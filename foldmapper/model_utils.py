"""Core graph layers and featurization for FoldMapper.

See ``NOTICE`` for third-party attributions. FoldMapper-specific change here:
``featurize`` carries an optional ``evo_profile`` tensor (the SPSF profile)
for the Stage-1 evolutionary input pathway.
"""

from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Batch featurization
# ---------------------------------------------------------------------------
def featurize(batch, device):
    """Convert a list of sample dicts into padded tensors for the model.

    Each sample dict must provide ``seq``, ``masked_list``, ``visible_list``,
    ``num_of_chains``, and per-chain ``seq_chain_<L>`` / ``coords_chain_<L>``
    entries. Optional ``evo_profile`` is an ``[L, 21]`` array in Foldseek amino
    acid ordering ``A C D E F G H I K L M N P Q R S T V W Y X``.
    """
    alphabet = "ACDEFGHIKLMNPQRSTVWYX"
    B = len(batch)
    lengths = np.array([len(b["seq"]) for b in batch], dtype=np.int32)
    L_max = max(len(b["seq"]) for b in batch)
    X = np.zeros([B, L_max, 4, 3])
    residue_idx = -100 * np.ones([B, L_max], dtype=np.int32)
    chain_M = np.zeros([B, L_max], dtype=np.int32)       # 1 = predict, 0 = visible context
    mask_self = np.ones([B, L_max, L_max], dtype=np.int32)
    chain_encoding_all = np.zeros([B, L_max], dtype=np.int32)
    S = np.zeros([B, L_max], dtype=np.int32)
    evo_profile = np.zeros([B, L_max, 21], dtype=np.float32)

    for i, b in enumerate(batch):
        masked_chains = list(b["masked_list"])
        visible_chains = list(b["visible_list"])
        all_chains = masked_chains + visible_chains
        # Deduplicate: collapse identical visible chains into the masked list.
        visible_temp_dict = {}
        masked_temp_dict = {}
        for letter in all_chains:
            chain_seq = b[f"seq_chain_{letter}"]
            if letter in visible_chains:
                visible_temp_dict[letter] = chain_seq
            elif letter in masked_chains:
                masked_temp_dict[letter] = chain_seq
        for _, vm in masked_temp_dict.items():
            for kv, vv in visible_temp_dict.items():
                if vm == vv:
                    if kv not in masked_chains:
                        masked_chains.append(kv)
                    if kv in visible_chains:
                        visible_chains.remove(kv)
        all_chains = masked_chains + visible_chains
        random.shuffle(all_chains)

        x_chain_list = []
        chain_mask_list = []
        chain_seq_list = []
        chain_encoding_list = []
        c = 1
        l0 = 0
        l1 = 0
        for letter in all_chains:
            chain_seq = b[f"seq_chain_{letter}"]
            chain_length = len(chain_seq)
            chain_coords = b[f"coords_chain_{letter}"]
            chain_mask = np.ones(chain_length) if letter in masked_chains else np.zeros(chain_length)
            x_chain = np.stack(
                [
                    chain_coords[k] for k in [
                        f"N_chain_{letter}",
                        f"CA_chain_{letter}",
                        f"C_chain_{letter}",
                        f"O_chain_{letter}",
                    ]
                ],
                1,
            )
            x_chain_list.append(x_chain)
            chain_mask_list.append(chain_mask)
            chain_seq_list.append(chain_seq)
            chain_encoding_list.append(c * np.ones(chain_length))
            l1 += chain_length
            mask_self[i, l0:l1, l0:l1] = np.zeros([chain_length, chain_length])
            residue_idx[i, l0:l1] = 100 * (c - 1) + np.arange(l0, l1)
            l0 += chain_length
            c += 1

        x = np.concatenate(x_chain_list, 0)
        all_sequence = "".join(chain_seq_list)
        m = np.concatenate(chain_mask_list, 0)
        chain_encoding = np.concatenate(chain_encoding_list, 0)

        L = len(all_sequence)
        # Normalise to shape [L, 4, 3] even if a single-residue chain collapsed a dim.
        if x.ndim == 4:
            x = x.squeeze()
        if x.ndim != 3:
            x = x.reshape(-1, 4, 3)
        x_pad = np.pad(x, [[0, L_max - L], [0, 0], [0, 0]], "constant", constant_values=(np.nan,))
        X[i, :, :, :] = x_pad

        m_pad = np.pad(m, [[0, L_max - L]], "constant", constant_values=(0.0,))
        chain_M[i, :] = m_pad

        chain_encoding_pad = np.pad(chain_encoding, [[0, L_max - L]], "constant", constant_values=(0.0,))
        chain_encoding_all[i, :] = chain_encoding_pad

        indices = np.asarray([alphabet.index(a) for a in all_sequence], dtype=np.int32)
        S[i, :L] = indices

        # Optional SPSF profile (first masked chain only).
        if "evo_profile" in b and b["evo_profile"] is not None:
            ep = b["evo_profile"]
            if hasattr(ep, "numpy"):
                ep = ep.numpy()
            ep = np.array(ep, dtype=np.float32)
            if ep.shape[0] == L:
                evo_profile[i, :L, :] = ep
            elif ep.shape[0] < L:
                evo_profile[i, :ep.shape[0], :] = ep

    isnan = np.isnan(X)
    mask = np.isfinite(np.sum(X, (2, 3))).astype(np.float32)
    X[isnan] = 0.0

    residue_idx = torch.from_numpy(residue_idx).to(dtype=torch.long, device=device)
    S = torch.from_numpy(S).to(dtype=torch.long, device=device)
    X = torch.from_numpy(X).to(dtype=torch.float32, device=device)
    mask = torch.from_numpy(mask).to(dtype=torch.float32, device=device)
    mask_self = torch.from_numpy(mask_self).to(dtype=torch.float32, device=device)
    chain_M = torch.from_numpy(chain_M).to(dtype=torch.float32, device=device)
    chain_encoding_all = torch.from_numpy(chain_encoding_all).to(dtype=torch.long, device=device)
    evo_profile = torch.from_numpy(evo_profile).to(dtype=torch.float32, device=device)
    return X, S, mask, lengths, chain_M, residue_idx, mask_self, chain_encoding_all, evo_profile


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------
def loss_nll(S, log_probs, mask):
    """Negative log-likelihood with per-position argmax accuracy."""
    criterion = torch.nn.NLLLoss(reduction="none")
    loss = criterion(
        log_probs.contiguous().view(-1, log_probs.size(-1)),
        S.contiguous().view(-1),
    ).view(S.size())
    S_argmaxed = torch.argmax(log_probs, -1)
    true_false = (S == S_argmaxed).float()
    loss_av = torch.sum(loss * mask) / torch.sum(mask)
    return loss, loss_av, true_false


def loss_smoothed(S, log_probs, mask, weight: float = 0.1):
    """Label-smoothed negative log-likelihood."""
    S_onehot = F.one_hot(S, 21).float()
    S_onehot = S_onehot + weight / float(S_onehot.size(-1))
    S_onehot = S_onehot / S_onehot.sum(-1, keepdim=True)
    loss = -(S_onehot * log_probs).sum(-1)
    loss_av = torch.sum(loss * mask) / torch.sum(mask)
    return loss, loss_av


# ---------------------------------------------------------------------------
# Graph gather helpers
# ---------------------------------------------------------------------------
def gather_edges(edges, neighbor_idx):
    """Edge features [B, N, N, C] indexed by [B, N, K] -> [B, N, K, C]."""
    neighbors = neighbor_idx.unsqueeze(-1).expand(-1, -1, -1, edges.size(-1))
    return torch.gather(edges, 2, neighbors)


def gather_nodes(nodes, neighbor_idx):
    """Node features [B, N, C] indexed by [B, N, K] -> [B, N, K, C]."""
    neighbors_flat = neighbor_idx.view((neighbor_idx.shape[0], -1))
    neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    neighbor_features = torch.gather(nodes, 1, neighbors_flat)
    return neighbor_features.view(list(neighbor_idx.shape)[:3] + [-1])


def gather_nodes_t(nodes, neighbor_idx):
    """Node features [B, N, C] at index [B, K] -> [B, K, C]."""
    idx_flat = neighbor_idx.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    return torch.gather(nodes, 1, idx_flat)


def cat_neighbors_nodes(h_nodes, h_neighbors, E_idx):
    h_nodes = gather_nodes(h_nodes, E_idx)
    return torch.cat([h_neighbors, h_nodes], -1)


# ---------------------------------------------------------------------------
# Core layers
# ---------------------------------------------------------------------------
class PositionWiseFeedForward(nn.Module):
    def __init__(self, num_hidden, num_ff):
        super().__init__()
        self.W_in = nn.Linear(num_hidden, num_ff, bias=True)
        self.W_out = nn.Linear(num_ff, num_hidden, bias=True)
        self.act = nn.GELU()

    def forward(self, h_V):
        return self.W_out(self.act(self.W_in(h_V)))


class PositionalEncodings(nn.Module):
    def __init__(self, num_embeddings, max_relative_feature: int = 32):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.max_relative_feature = max_relative_feature
        self.linear = nn.Linear(2 * max_relative_feature + 1 + 1, num_embeddings)

    def forward(self, offset, mask):
        d = torch.clip(offset + self.max_relative_feature, 0, 2 * self.max_relative_feature) * mask + (1 - mask) * (2 * self.max_relative_feature + 1)
        d_onehot = F.one_hot(d, 2 * self.max_relative_feature + 1 + 1)
        return self.linear(d_onehot.float())


class ProteinFeatures(nn.Module):
    """Backbone graph featurizer: Calpha k-NN with RBF inter-atom distances."""

    def __init__(
        self,
        edge_features,
        node_features,
        num_positional_embeddings: int = 16,
        num_rbf: int = 16,
        top_k: int = 30,
        augment_eps: float = 0.0,
        num_chain_embeddings: int = 16,
    ):
        super().__init__()
        self.edge_features = edge_features
        self.node_features = node_features
        self.top_k = top_k
        self.augment_eps = augment_eps
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings

        self.embeddings = PositionalEncodings(num_positional_embeddings)
        edge_in = num_positional_embeddings + num_rbf * 25
        self.edge_embedding = nn.Linear(edge_in, edge_features, bias=False)
        self.norm_edges = nn.LayerNorm(edge_features)

    def _dist(self, X, mask, eps: float = 1e-6):
        mask_2D = torch.unsqueeze(mask, 1) * torch.unsqueeze(mask, 2)
        dX = torch.unsqueeze(X, 1) - torch.unsqueeze(X, 2)
        D = mask_2D * torch.sqrt(torch.sum(dX ** 2, 3) + eps)
        D_max, _ = torch.max(D, -1, keepdim=True)
        D_adjust = D + (1.0 - mask_2D) * D_max
        D_neighbors, E_idx = torch.topk(
            D_adjust, np.minimum(self.top_k, X.shape[1]), dim=-1, largest=False
        )
        return D_neighbors, E_idx

    def _rbf(self, D):
        device = D.device
        D_min, D_max, D_count = 2.0, 22.0, self.num_rbf
        D_mu = torch.linspace(D_min, D_max, D_count, device=device).view([1, 1, 1, -1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)
        return torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)

    def _get_rbf(self, A, B, E_idx):
        D_A_B = torch.sqrt(torch.sum((A[:, :, None, :] - B[:, None, :, :]) ** 2, -1) + 1e-6)
        D_A_B_neighbors = gather_edges(D_A_B[:, :, :, None], E_idx)[:, :, :, 0]
        return self._rbf(D_A_B_neighbors)

    def forward(self, X, mask, residue_idx, chain_labels):
        if self.training and self.augment_eps > 0:
            X = X + self.augment_eps * torch.randn_like(X)

        # Calpha-based virtual Cbeta, then build Calpha k-NN edges.
        b = X[:, :, 1, :] - X[:, :, 0, :]
        c = X[:, :, 2, :] - X[:, :, 1, :]
        a = torch.cross(b, c, dim=-1)
        Cb = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + X[:, :, 1, :]
        Ca = X[:, :, 1, :]
        N = X[:, :, 0, :]
        C = X[:, :, 2, :]
        O = X[:, :, 3, :]

        D_neighbors, E_idx = self._dist(Ca, mask)

        rbf_pairs = [
            (Ca, Ca), (N, N), (C, C), (O, O), (Cb, Cb),
            (Ca, N), (Ca, C), (Ca, O), (Ca, Cb),
            (N, C), (N, O), (N, Cb),
            (Cb, C), (Cb, O),
            (O, C),
            (N, Ca), (C, Ca), (O, Ca), (Cb, Ca),
            (C, N), (O, N), (Cb, N),
            (C, Cb), (O, Cb),
            (C, O),
        ]
        rbf_list = []
        for A_xyz, B_xyz in rbf_pairs:
            if A_xyz is Ca and B_xyz is Ca:
                rbf_list.append(self._rbf(D_neighbors))
            else:
                rbf_list.append(self._get_rbf(A_xyz, B_xyz, E_idx))
        RBF_all = torch.cat(rbf_list, dim=-1)

        offset = residue_idx[:, :, None] - residue_idx[:, None, :]
        offset = gather_edges(offset[:, :, :, None], E_idx)[:, :, :, 0]

        d_chains = ((chain_labels[:, :, None] - chain_labels[:, None, :]) == 0).long()
        E_chains = gather_edges(d_chains[:, :, :, None], E_idx)[:, :, :, 0]
        E_positional = self.embeddings(offset.long(), E_chains)
        E = torch.cat((E_positional, RBF_all), -1)
        E = self.edge_embedding(E)
        E = self.norm_edges(E)
        return E, E_idx


class EncLayer(nn.Module):
    """Encoder message-passing layer (updates nodes and edges). Shared by both stages."""

    def __init__(self, num_hidden, num_in, dropout: float = 0.1, num_heads=None, scale: int = 30):
        super().__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)
        self.norm3 = nn.LayerNorm(num_hidden)

        self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W11 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W12 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W13 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = nn.GELU()
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_V, h_E, E_idx, mask_V=None, mask_attend=None):
        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
        h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_EV.size(-2), -1)
        h_EV = torch.cat([h_V_expand, h_EV], -1)
        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message
        dh = torch.sum(h_message, -2) / self.scale
        h_V = self.norm1(h_V + self.dropout1(dh))

        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))
        if mask_V is not None:
            h_V = mask_V.unsqueeze(-1) * h_V

        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
        h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_EV.size(-2), -1)
        h_EV = torch.cat([h_V_expand, h_EV], -1)
        h_message = self.W13(self.act(self.W12(self.act(self.W11(h_EV)))))
        h_E = self.norm3(h_E + self.dropout3(h_message))
        return h_V, h_E


class DecLayer(nn.Module):
    """Decoder message-passing layer (updates nodes only). Used by the Stage 2 decoder."""

    def __init__(self, num_hidden, num_in, dropout: float = 0.1, num_heads=None, scale: int = 30):
        super().__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)

        self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = nn.GELU()
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_V, h_E, mask_V=None, mask_attend=None):
        h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_E.size(-2), -1)
        h_EV = torch.cat([h_V_expand, h_E], -1)

        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message
        dh = torch.sum(h_message, -2) / self.scale

        h_V = self.norm1(h_V + self.dropout1(dh))
        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))
        if mask_V is not None:
            h_V = mask_V.unsqueeze(-1) * h_V
        return h_V


# ---------------------------------------------------------------------------
# Noam LR scheduler
# ---------------------------------------------------------------------------
class NoamOpt:
    """Optimizer wrapper implementing the Noam (inverse-sqrt) learning rate schedule."""

    def __init__(self, model_size, factor, warmup, optimizer, step):
        self.optimizer = optimizer
        self._step = step
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def step(self):
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p["lr"] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        if step is None:
            step = self._step
        return self.factor * (
            self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5))
        )

    def zero_grad(self):
        self.optimizer.zero_grad()


def get_std_opt(parameters, d_model, step):
    return NoamOpt(
        d_model,
        2,
        4000,
        torch.optim.Adam(parameters, lr=0, betas=(0.9, 0.98), eps=1e-9),
        step,
    )
