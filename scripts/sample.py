#!/usr/bin/env python3
"""Sample sequences from a backbone using a FoldMapper checkpoint.

Two ways to provide the evolutionary profile:

  1. ``--profile profile.npy``   : precomputed ``[L, 21]`` array.
  2. ``--foldseek-db DB``         : run Foldseek at call time on ``--pdb``.

If neither is given, the model is run in backbone-only mode (profile-free),
which matches the ``profile_dropout`` training regime.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from foldmapper import load_model_from_checkpoint  # noqa: E402
from foldmapper.io import load_structure  # noqa: E402
from foldmapper.model_utils import featurize  # noqa: E402
from foldmapper.profile import profile_from_structure, IDX_TO_AA  # noqa: E402


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pdb", required=True, help="Input PDB or mmCIF file.")
    p.add_argument("--chain", default=None, help="Chain id to use (default: first chain).")
    p.add_argument("--checkpoint", required=True, help="FoldMapper *.pt checkpoint.")
    p.add_argument("--profile", default=None, help="Precomputed profile [L,21] .npy file.")
    p.add_argument("--foldseek-db", default=None, help="Foldseek DB path; runs easy-search on the input if provided.")
    p.add_argument("--foldseek-bin", default="foldseek", help="Foldseek executable.")
    p.add_argument("--foldseek-topk", type=int, default=100, help="Top-k alignments for profile.")
    p.add_argument("--n-samples", type=int, default=10, help="Sequences to sample.")
    p.add_argument("--temperature", type=float, default=0.1, help="Sampling temperature (>0).")
    p.add_argument("--seed", type=int, default=19720422)
    p.add_argument("--device", default=None, help="cuda | cpu (default: auto).")
    p.add_argument("--output-fasta", default=None, help="Write FASTA here; else stdout.")
    return p


def load_profile(args, seq_len: int) -> np.ndarray:
    if args.profile:
        prof = np.load(args.profile).astype(np.float32)
        if prof.shape[0] != seq_len:
            print(
                f"[warn] profile length {prof.shape[0]} != sequence length {seq_len};"
                " truncating/padding.",
                file=sys.stderr,
            )
            out = np.zeros((seq_len, 21), dtype=np.float32)
            L = min(prof.shape[0], seq_len)
            out[:L] = prof[:L]
            return out
        return prof
    if args.foldseek_db:
        print(f"[foldseek] generating profile from {args.pdb} against {args.foldseek_db}", file=sys.stderr)
        return profile_from_structure(
            args.pdb,
            args.foldseek_db,
            seq_len,
            foldseek_bin=args.foldseek_bin,
            topk=args.foldseek_topk,
        )
    return None


def main():
    args = build_parser().parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    entry = load_structure(args.pdb, chain=args.chain)
    seq_len = len(entry["seq"])
    print(f"[load] {args.pdb}: length={seq_len}, chain={entry.get('masked_list', ['?'])[0]}", file=sys.stderr)

    profile_np = load_profile(args, seq_len)
    has_profile = profile_np is not None

    model, _ = load_model_from_checkpoint(args.checkpoint, device=device)
    print(
        f"[model] use_blosum={model.config.use_blosum}, s1_hidden_dim={model.config.s1_hidden_dim},"
        f" hidden_dim={model.config.hidden_dim}",
        file=sys.stderr,
    )

    X, S, mask, lengths, chain_M, residue_idx, mask_self, chain_encoding_all, _ = featurize(
        [entry], device
    )

    if has_profile:
        evo_profile = torch.zeros(1, X.shape[1], 21, device=device)
        evo_profile[0, :seq_len] = torch.from_numpy(profile_np).to(device)
    else:
        evo_profile = None

    native_seq = entry["seq"]
    fasta_lines = []

    t0 = time.time()
    for i in range(args.n_samples):
        out = model.sample(
            X, mask, chain_M, residue_idx, chain_encoding_all,
            evo_profile=evo_profile,
            temperature=args.temperature,
        )
        sampled_idx = out["S"][0, :seq_len].tolist()
        pred = "".join(IDX_TO_AA.get(i, "X") for i in sampled_idx)
        matches = sum(a == b for a, b in zip(native_seq, pred))
        recovery = matches / seq_len if seq_len else 0.0
        fasta_lines.append(
            f">{entry['name']}|sample={i}|T={args.temperature}|recovery={recovery:.3f}\n{pred}"
        )
    dt = time.time() - t0
    print(f"[sample] {args.n_samples} sequences in {dt:.1f}s", file=sys.stderr)

    text = "\n".join(fasta_lines) + "\n"
    if args.output_fasta:
        Path(args.output_fasta).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_fasta, "w") as f:
            f.write(text)
        print(f"[write] {args.output_fasta}", file=sys.stderr)
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
