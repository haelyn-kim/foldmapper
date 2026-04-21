"""FoldMapper: two-stage graph inverse-folding with an evolutionary-profile prior.

Stage 1  distillation encoder (k=30, edge=128) consumes a per-position Foldseek
         alignment profile (optionally with a BLOSUM62 transform) along with the
         backbone coordinates and emits distilled node features.
Stage 2  ProteinMPNN-style encoder/decoder (k=48, edge=512) predicts the amino
         acid sequence autoregressively.

All four paper variants (d_s1 in {256, 512} x {no BLOSUM, +BLOSUM}) share this
class; architecture is controlled by ``FoldMapperConfig``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field, asdict
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from .blosum import get_blosum62_prob
from .model_utils import (
    ProteinFeatures,
    EncLayer,
    DecLayer,
    gather_nodes,
    cat_neighbors_nodes,
)


@dataclass
class FoldMapperConfig:
    """Architecture + training hyperparameters for FoldMapper."""

    # Stage 1: distillation encoder
    distill_hidden_dim: int = 256
    distill_edge_features: int = 128
    distill_encoder_layers: int = 3
    distill_k_neighbors: int = 30
    distill_augment_eps: float = 0.1

    # Stage 2: MPNN backbone encoder/decoder
    hidden_dim: int = 512
    num_encoder_layers: int = 3
    num_decoder_layers: int = 3
    k_neighbors: int = 48
    augment_eps: float = 0.2

    # Regularization
    dropout: float = 0.1
    profile_dropout: float = 0.1

    # Vocabulary
    num_letters: int = 21
    vocab: int = 21

    # BLOSUM62 injection (Stage 1 node-init bias)
    use_blosum: bool = False
    aa_factor: float = 1.4


class FoldMapper(nn.Module):
    """FoldMapper / Distilled ProteinMPNN two-stage model."""

    def __init__(self, config: Optional[FoldMapperConfig] = None, **overrides):
        super().__init__()
        if config is None:
            config = FoldMapperConfig()
        if overrides:
            config = FoldMapperConfig(**{**asdict(config), **overrides})
        self.config = config

        distill_hidden_dim = config.distill_hidden_dim
        distill_edge_features = config.distill_edge_features
        hidden_dim = config.hidden_dim
        self.hidden_dim = hidden_dim
        self.profile_dropout = config.profile_dropout
        self.use_blosum = config.use_blosum

        # -- Stage 1: distillation encoder --
        self.distill_features = ProteinFeatures(
            distill_edge_features,
            distill_edge_features,
            top_k=config.distill_k_neighbors,
            augment_eps=config.distill_augment_eps,
        )
        self.distill_W_p = nn.Linear(config.vocab, distill_hidden_dim, bias=True)
        if config.use_blosum:
            self.distill_W_t = nn.Linear(config.vocab, distill_hidden_dim, bias=True)
            self.register_buffer("blosum62_prob", get_blosum62_prob(config.aa_factor))
        self.distill_W_e = nn.Linear(distill_edge_features, distill_hidden_dim, bias=True)
        self.distill_encoder = nn.ModuleList(
            [
                EncLayer(distill_hidden_dim, distill_hidden_dim * 2, dropout=config.dropout)
                for _ in range(config.distill_encoder_layers)
            ]
        )

        # Projection into MPNN hidden dim
        if distill_hidden_dim != hidden_dim:
            self.projection = nn.Linear(distill_hidden_dim, hidden_dim, bias=True)
        else:
            self.projection = nn.Identity()

        # -- Stage 2: ProteinMPNN-style backbone encoder/decoder --
        self.features = ProteinFeatures(
            hidden_dim,
            hidden_dim,
            top_k=config.k_neighbors,
            augment_eps=config.augment_eps,
        )
        self.W_e = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.W_s = nn.Embedding(config.vocab, hidden_dim)
        self.encoder_layers = nn.ModuleList(
            [
                EncLayer(hidden_dim, hidden_dim * 2, dropout=config.dropout)
                for _ in range(config.num_encoder_layers)
            ]
        )
        self.decoder_layers = nn.ModuleList(
            [
                DecLayer(hidden_dim, hidden_dim * 3, dropout=config.dropout)
                for _ in range(config.num_decoder_layers)
            ]
        )
        self.W_out = nn.Linear(hidden_dim, config.num_letters, bias=True)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # ------------------------------------------------------------------
    # Stage 1 encode
    # ------------------------------------------------------------------
    def _encode_profile(self, X, mask, residue_idx, chain_encoding_all, evo_profile):
        """Run Stage 1 and return an [B, L, hidden_dim] projected node tensor."""
        E_d, E_idx_d = self.distill_features(X, mask, residue_idx, chain_encoding_all)
        if self.use_blosum:
            transformed = evo_profile @ self.blosum62_prob  # [B, L, 21]
            h_V_d = (self.distill_W_p(evo_profile) + self.distill_W_t(transformed)) * mask.unsqueeze(-1)
        else:
            h_V_d = self.distill_W_p(evo_profile) * mask.unsqueeze(-1)
        h_E_d = self.distill_W_e(E_d)
        mask_attend_d = gather_nodes(mask.unsqueeze(-1), E_idx_d).squeeze(-1)
        mask_attend_d = mask.unsqueeze(-1) * mask_attend_d
        for layer in self.distill_encoder:
            if self.training:
                h_V_d, h_E_d = torch.utils.checkpoint.checkpoint(
                    layer, h_V_d, h_E_d, E_idx_d, mask, mask_attend_d
                )
            else:
                h_V_d, h_E_d = layer(h_V_d, h_E_d, E_idx_d, mask, mask_attend_d)
        return self.projection(h_V_d) * mask.unsqueeze(-1)

    # ------------------------------------------------------------------
    # Teacher-forced forward (training / perplexity)
    # ------------------------------------------------------------------
    def forward(self, X, S, mask, chain_M, residue_idx, chain_encoding_all, evo_profile=None):
        device = X.device

        # -- Stage 1 --
        drop_profile = (
            self.training
            and self.profile_dropout > 0
            and random.random() < self.profile_dropout
        )
        if evo_profile is not None and not drop_profile:
            h_V = self._encode_profile(X, mask, residue_idx, chain_encoding_all, evo_profile)
        else:
            h_V = torch.zeros(X.shape[0], X.shape[1], self.hidden_dim, device=device)

        # -- Stage 2 encoder --
        E, E_idx = self.features(X, mask, residue_idx, chain_encoding_all)
        h_E = self.W_e(E)
        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.encoder_layers:
            if self.training:
                h_V, h_E = torch.utils.checkpoint.checkpoint(
                    layer, h_V, h_E, E_idx, mask, mask_attend
                )
            else:
                h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)

        # -- Stage 2 autoregressive decoder (teacher forced) --
        h_S = self.W_s(S)
        h_ES = cat_neighbors_nodes(h_S, h_E, E_idx)
        h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)

        chain_M = chain_M * mask
        decoding_order = torch.argsort(
            (chain_M + 0.0001) * torch.abs(torch.randn(chain_M.shape, device=device))
        )
        mask_size = E_idx.shape[1]
        perm_rev = F.one_hot(decoding_order, num_classes=mask_size).float()
        order_mask_bw = torch.einsum(
            "ij, biq, bjp->bqp",
            (1 - torch.triu(torch.ones(mask_size, mask_size, device=device))),
            perm_rev,
            perm_rev,
        )
        mask_attend_dec = torch.gather(order_mask_bw, 2, E_idx).unsqueeze(-1)
        mask_1D = mask.view(mask.size(0), mask.size(1), 1, 1)
        mask_bw = mask_1D * mask_attend_dec
        mask_fw = mask_1D * (1.0 - mask_attend_dec)

        h_EXV_encoder_fw = mask_fw * h_EXV_encoder
        for layer in self.decoder_layers:
            h_ESV = cat_neighbors_nodes(h_V, h_ES, E_idx)
            h_ESV = mask_bw * h_ESV + h_EXV_encoder_fw
            if self.training:
                h_V = torch.utils.checkpoint.checkpoint(layer, h_V, h_ESV, mask)
            else:
                h_V = layer(h_V, h_ESV, mask)

        logits = self.W_out(h_V)
        return F.log_softmax(logits, dim=-1)

    # ------------------------------------------------------------------
    # Autoregressive sampling
    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample(
        self,
        X,
        mask,
        chain_M,
        residue_idx,
        chain_encoding_all,
        evo_profile=None,
        temperature: float = 0.1,
        randn: Optional[torch.Tensor] = None,
    ):
        """Autoregressive sampling. ``chain_M`` is 1 for positions to predict.

        Returns a dict with ``S`` (sampled indices) and ``log_probs`` (for
        teacher-forced perplexity, computed with the sampled sequence).

        NOTE: the current implementation assumes every masked position is
        freshly sampled; visible positions (``chain_M == 0``) are filled with
        the zero index rather than the native sequence. Single-chain
        inverse-folding works correctly. To support partial-redesign (mixed
        visible/masked contexts), pass native ``S`` through and restore it at
        visible positions in the scatter step below.
        """
        device = X.device
        B, N_nodes = X.shape[0], X.shape[1]

        if evo_profile is not None:
            h_V = self._encode_profile(X, mask, residue_idx, chain_encoding_all, evo_profile)
        else:
            h_V = torch.zeros(B, N_nodes, self.hidden_dim, device=device)

        E, E_idx = self.features(X, mask, residue_idx, chain_encoding_all)
        h_E = self.W_e(E)
        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)

        h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_V), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)

        chain_M_updated = chain_M * mask
        if randn is None:
            randn = torch.randn(B, N_nodes, device=device)
        decoding_order = torch.argsort((chain_M_updated + 0.0001) * torch.abs(randn))

        mask_size = N_nodes
        perm_rev = F.one_hot(decoding_order, num_classes=mask_size).float()
        order_mask_bw = torch.einsum(
            "ij, biq, bjp->bqp",
            (1 - torch.triu(torch.ones(mask_size, mask_size, device=device))),
            perm_rev,
            perm_rev,
        )
        mask_attend_dec = torch.gather(order_mask_bw, 2, E_idx).unsqueeze(-1)
        mask_1D = mask.view(B, N_nodes, 1, 1)
        mask_bw = mask_1D * mask_attend_dec
        mask_fw = mask_1D * (1.0 - mask_attend_dec)
        h_EXV_encoder_fw = mask_fw * h_EXV_encoder

        h_S = torch.zeros_like(h_V)
        S_sampled = torch.zeros(B, N_nodes, dtype=torch.int64, device=device)
        all_log_probs = torch.zeros(B, N_nodes, self.config.num_letters, device=device)
        h_V_stack = [h_V] + [torch.zeros_like(h_V) for _ in range(len(self.decoder_layers))]

        hidden_dim = self.hidden_dim
        for t_ in range(N_nodes):
            t = decoding_order[:, t_]
            chain_mask_gathered = torch.gather(chain_M_updated, 1, t[:, None])
            mask_gathered = torch.gather(mask, 1, t[:, None])

            if (mask_gathered == 0).all():
                S_t = torch.zeros((B, 1), dtype=torch.int64, device=device)
            else:
                E_idx_t = torch.gather(
                    E_idx, 1, t[:, None, None].repeat(1, 1, E_idx.shape[-1])
                )
                h_E_t = torch.gather(
                    h_E,
                    1,
                    t[:, None, None, None].repeat(1, 1, h_E.shape[-2], h_E.shape[-1]),
                )
                h_ES_t = cat_neighbors_nodes(h_S, h_E_t, E_idx_t)
                h_EXV_encoder_t = torch.gather(
                    h_EXV_encoder_fw,
                    1,
                    t[:, None, None, None].repeat(
                        1, 1, h_EXV_encoder_fw.shape[-2], h_EXV_encoder_fw.shape[-1]
                    ),
                )
                mask_t = torch.gather(mask, 1, t[:, None])

                for l_idx, layer in enumerate(self.decoder_layers):
                    h_ESV_decoder_t = cat_neighbors_nodes(h_V_stack[l_idx], h_ES_t, E_idx_t)
                    h_V_t_l = torch.gather(
                        h_V_stack[l_idx], 1, t[:, None, None].repeat(1, 1, hidden_dim)
                    )
                    h_ESV_t = (
                        torch.gather(
                            mask_bw,
                            1,
                            t[:, None, None, None].repeat(
                                1, 1, mask_bw.shape[-2], mask_bw.shape[-1]
                            ),
                        )
                        * h_ESV_decoder_t
                        + h_EXV_encoder_t
                    )
                    h_V_stack[l_idx + 1].scatter_(
                        1,
                        t[:, None, None].repeat(1, 1, hidden_dim),
                        layer(h_V_t_l, h_ESV_t, mask_V=mask_t),
                    )

                h_V_t_final = torch.gather(
                    h_V_stack[-1], 1, t[:, None, None].repeat(1, 1, hidden_dim)
                )[:, 0]
                logits = self.W_out(h_V_t_final) / max(temperature, 1e-6)
                probs = F.softmax(logits, dim=-1)
                S_t = torch.multinomial(probs, 1)
                log_probs_t = F.log_softmax(self.W_out(h_V_t_final), dim=-1)
                all_log_probs.scatter_(
                    1,
                    t[:, None, None].repeat(1, 1, self.config.num_letters),
                    log_probs_t.unsqueeze(1),
                )

            S_t = (S_t * chain_mask_gathered + 0 * (1.0 - chain_mask_gathered)).long()
            temp_h = self.W_s(S_t)
            h_S.scatter_(1, t[:, None, None].repeat(1, 1, hidden_dim), temp_h)
            S_sampled.scatter_(1, t[:, None], S_t)

        return {"S": S_sampled, "log_probs": all_log_probs, "decoding_order": decoding_order}


# ---------------------------------------------------------------------------
# Backwards-compatible alias used in original training checkpoints.
# ---------------------------------------------------------------------------
DistilledProteinMPNN = FoldMapper


# ---------------------------------------------------------------------------
# Checkpoint loading helper: auto-detects architecture from state_dict.
# ---------------------------------------------------------------------------
def infer_config_from_state_dict(state_dict) -> FoldMapperConfig:
    """Reconstruct a ``FoldMapperConfig`` from a saved state_dict."""
    hidden_dim = state_dict["W_s.weight"].shape[1]
    distill_hidden_dim = state_dict["distill_W_e.weight"].shape[0]
    distill_edge_features = state_dict["distill_W_e.weight"].shape[1]

    num_distill_layers = len(
        {int(k.split(".")[1]) for k in state_dict if k.startswith("distill_encoder.")}
    )
    num_encoder_layers = len(
        {int(k.split(".")[1]) for k in state_dict if k.startswith("encoder_layers.")}
    )
    num_decoder_layers = len(
        {int(k.split(".")[1]) for k in state_dict if k.startswith("decoder_layers.")}
    )
    use_blosum = any(k.startswith("distill_W_t.") for k in state_dict)

    return FoldMapperConfig(
        distill_hidden_dim=distill_hidden_dim,
        distill_edge_features=distill_edge_features,
        distill_encoder_layers=num_distill_layers,
        hidden_dim=hidden_dim,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        use_blosum=use_blosum,
        # Eval-time defaults: disable augmentation.
        augment_eps=0.0,
        distill_augment_eps=0.0,
        dropout=0.0,
        profile_dropout=0.0,
    )


def load_model_from_checkpoint(checkpoint_path: str, device: str = "cpu",
                               config: Optional[FoldMapperConfig] = None):
    """Load a ``FoldMapper`` from a training checkpoint.

    If ``config`` is None it is inferred from the state_dict (architectural
    hyper-parameters only; training-time regularisers are set to zero).
    Returns ``(model, checkpoint_dict)``.
    """
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt["model_state_dict"]

    if config is None:
        config = infer_config_from_state_dict(state_dict)
        # k_neighbors is not recoverable from weights alone; pick checkpoint hint if any.
        k_hint = ckpt.get("num_edges", 48)
        config = FoldMapperConfig(**{**asdict(config), "k_neighbors": k_hint})

    model = FoldMapper(config)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device).eval()
    return model, ckpt
