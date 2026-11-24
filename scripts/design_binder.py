#!/usr/bin/env python3
"""Design binder sequences for a fixed target structure.

The binder chain is fully redesigned while the target chain provides fixed
structural context (chain_M = 0). SPSF profiles can be supplied via
a precomputed .npy file or generated on-the-fly with Foldseek (searched
against the binder chain only).

Example
-------
python scripts/design_binder.py \\
    --pdb complex.pdb --binder-chain B --target-chain A \\
    --checkpoint model.pt --foldseek-db /path/to/pdb100_db \\
    --n-samples 50 --temperature 0.1 --output-fasta designs.fa
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from foldmapper import load_model_from_checkpoint  # noqa: E402
from foldmapper.io import load_multichain_structure  # noqa: E402
from foldmapper.model_utils import featurize  # noqa: E402
from foldmapper.profile import profile_from_structure, IDX_TO_AA  # noqa: E402


def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--pdb", required=True, help="Input PDB or mmCIF file.")
    p.add_argument("--binder-chain", required=True, help="Chain ID to redesign.")
    p.add_argument("--target-chain", required=True, help="Chain ID kept as fixed context.")
    p.add_argument("--checkpoint", required=True, help="FoldMapper *.pt checkpoint.")
    p.add_argument("--profile", default=None, help="Precomputed binder profile [L_binder, 21] .npy.")
    p.add_argument("--foldseek-db", default=None, help="Foldseek DB; runs easy-search on the binder chain.")
    p.add_argument("--foldseek-bin", default="foldseek", help="Foldseek executable.")
    p.add_argument("--foldseek-topk", type=int, default=100, help="Top-k alignments for profile.")
    p.add_argument("--n-samples", type=int, default=10, help="Sequences to sample.")
    p.add_argument("--temperature", type=float, default=0.1, help="Sampling temperature (>0).")
    p.add_argument("--seed", type=int, default=19720422, help="Random seed for reproducibility.")
    p.add_argument(
        "--fixed-residues", default=None,
        help="Residues to keep fixed, by PDB number. Format: A:1-10,A:25,B:3-7",
    )
    p.add_argument("--device", default=None, help="cuda | cpu (default: auto).")
    p.add_argument("--output-fasta", default=None, help="Write FASTA here; else stdout.")
    return p


def _extract_chain_pdb(pdb_path: str, chain_id: str, out_path: str) -> None:
    """Write a single chain to a PDB file for Foldseek searching."""
    from Bio.PDB import PDBParser, PDBIO, Select

    class _ChainSelect(Select):
        def accept_chain(self, chain):
            return chain.get_id() == chain_id

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("s", pdb_path)
    io = PDBIO()
    io.set_structure(structure)
    io.save(out_path, _ChainSelect())


def _parse_fixed_residues(spec: str) -> dict:
    """Parse 'A:1-10,A:25,B:3-7' into {chain: set_of_pdb_resnums}."""
    fixed: dict = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        chain, nums = part.split(":", 1)
        chain = chain.strip()
        fixed.setdefault(chain, set())
        if "-" in nums:
            lo, hi = nums.split("-", 1)
            fixed[chain].update(range(int(lo), int(hi) + 1))
        else:
            fixed[chain].add(int(nums))
    return fixed


def _apply_fixed_residues(chain_M, fixed: dict, entry: dict, starts: dict) -> None:
    """Set chain_M to 0 for fixed PDB residues (in-place)."""
    for ch, resnums_to_fix in fixed.items():
        if f"resnum_chain_{ch}" not in entry:
            print(f"[warn] --fixed-residues: chain {ch!r} not in structure, skipping", file=sys.stderr)
            continue
        resnum_list = entry[f"resnum_chain_{ch}"]
        resnum_to_idx = {r: i for i, r in enumerate(resnum_list)}
        ch_start = starts[ch]
        for r in sorted(resnums_to_fix):
            if r not in resnum_to_idx:
                print(f"[warn] --fixed-residues: residue {ch}:{r} not found (missing backbone?), skipping", file=sys.stderr)
                continue
            chain_M[0, ch_start + resnum_to_idx[r]] = 0.0


def _load_profile(args, binder_len: int):
    if args.profile:
        prof = np.load(args.profile).astype(np.float32)
        if prof.shape[0] != binder_len:
            print(
                f"[warn] profile length {prof.shape[0]} != binder length {binder_len};"
                " truncating/padding.",
                file=sys.stderr,
            )
            out = np.zeros((binder_len, 21), dtype=np.float32)
            L = min(prof.shape[0], binder_len)
            out[:L] = prof[:L]
            return out
        return prof
    if args.foldseek_db:
        print(
            f"[foldseek] generating profile for binder chain {args.binder_chain}",
            file=sys.stderr,
        )
        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as f:
            tmp_pdb = f.name
        try:
            _extract_chain_pdb(args.pdb, args.binder_chain, tmp_pdb)
            return profile_from_structure(
                tmp_pdb,
                args.foldseek_db,
                binder_len,
                foldseek_bin=args.foldseek_bin,
                topk=args.foldseek_topk,
            )
        finally:
            os.unlink(tmp_pdb)
    return None


def main():
    args = build_parser().parse_args()

    # Seed all three independent random states for full reproducibility.
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    entry = load_multichain_structure(
        args.pdb,
        masked_chains=[args.binder_chain],
        context_chains=[args.target_chain],
    )
    binder_len = len(entry[f"seq_chain_{args.binder_chain}"])
    target_len = len(entry[f"seq_chain_{args.target_chain}"])
    print(
        f"[load] {args.pdb}: binder={args.binder_chain} ({binder_len} res),"
        f" target={args.target_chain} ({target_len} res)",
        file=sys.stderr,
    )

    model, _ = load_model_from_checkpoint(args.checkpoint, device=device)
    print(
        f"[model] use_blosum={model.config.use_blosum}, hidden_dim={model.config.hidden_dim}",
        file=sys.stderr,
    )

    X, S, mask, lengths, chain_M, residue_idx, mask_self, chain_encoding_all, _ = featurize(
        [entry], device
    )

    # Find chain starts from chain_M (binder=1, target=0 among valid positions).
    valid = mask[0].bool()
    binder_pos = (chain_M[0] == 1.0).nonzero(as_tuple=True)[0]
    target_pos = ((chain_M[0] == 0.0) & valid).nonzero(as_tuple=True)[0]
    _starts = {
        args.binder_chain: int(binder_pos[0].item()) if len(binder_pos) > 0 else 0,
        args.target_chain: int(target_pos[0].item()) if len(target_pos) > 0 else 0,
    }

    # Capture binder positions before fixed-residues modifies chain_M.
    binder_output_mask = (chain_M[0] == 1.0).clone()

    if args.fixed_residues:
        fixed = _parse_fixed_residues(args.fixed_residues)
        _apply_fixed_residues(chain_M, fixed, entry, _starts)
        n_fixed = sum(len(v) for v in fixed.values())
        print(f"[fixed] {n_fixed} residue(s) locked to native sequence", file=sys.stderr)

    profile_np = _load_profile(args, binder_len)
    if profile_np is not None:
        # chain_M == 1 marks binder positions; they form one contiguous block.
        binder_positions = (chain_M[0] == 1.0).nonzero(as_tuple=True)[0]
        binder_start = binder_positions[0].item()
        evo_profile = torch.zeros(1, X.shape[1], 21, device=device)
        evo_profile[0, binder_start:binder_start + binder_len] = (
            torch.from_numpy(profile_np).to(device)
        )
    else:
        evo_profile = None

    native_binder = entry[f"seq_chain_{args.binder_chain}"]
    fasta_lines = []

    t0 = time.time()
    for i in range(args.n_samples):
        out = model.sample(
            X, mask, chain_M, residue_idx, chain_encoding_all,
            evo_profile=evo_profile,
            temperature=args.temperature,
            S=S,
        )
        # Extract sampled binder residues using the pre-fixed-residues mask,
        # so fixed positions (now chain_M == 0) are still included.
        sampled_idx = out["S"][0][binder_output_mask].tolist()
        pred = "".join(IDX_TO_AA.get(idx, "X") for idx in sampled_idx)
        recovery = (
            sum(a == b for a, b in zip(native_binder, pred)) / binder_len
            if binder_len else 0.0
        )
        fasta_lines.append(
            f">{entry['name']}|binder={args.binder_chain}|target={args.target_chain}"
            f"|sample={i}|T={args.temperature}|recovery={recovery:.3f}\n{pred}"
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
