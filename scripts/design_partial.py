#!/usr/bin/env python3
"""Partial and tied-oligomer sequence design with FoldMapper.

Redesigns one or more chains while keeping context chains fixed. Chains can be
tied so that they produce the same sequence — required for symmetric homo-oligomers.

Chain specification
-------------------
--chains-to-design A,B    redesign chains A and B
--context-chains   C      keep chain C as fixed structural context
--tie-chains       A,B    chains A and B must share one sequence (homo-oligomer)

Profile support
---------------
A single Foldseek profile ``[L_chain, 21]`` can be provided for the first
chain listed in ``--chains-to-design`` (or for the tied group). For tied
designs the profile is broadcast to all tied chain slices in the tensor.

Example
-------
# Homodimer: design both chains with the same sequence
python scripts/design_partial.py \\
    --pdb dimer.pdb --chains-to-design A,B --tie-chains A,B \\
    --checkpoint model.pt --foldseek-db /path/to/foldseek_db \\
    --n-samples 50 --temperature 0.1

# Asymmetric partial redesign: redesign chain B, keep A and C fixed
python scripts/design_partial.py \\
    --pdb complex.pdb --chains-to-design B --context-chains A,C \\
    --checkpoint model.pt --foldseek-db /path/to/foldseek_db \\
    --n-samples 20
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set

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
    p.add_argument(
        "--chains-to-design", required=True,
        help="Comma-separated chain IDs to redesign (e.g. A,B).",
    )
    p.add_argument(
        "--context-chains", default="",
        help="Comma-separated chain IDs to keep fixed as structural context.",
    )
    p.add_argument(
        "--tie-chains", default="",
        help="Comma-separated chain IDs that must share the same sequence (homo-oligomer).",
    )
    p.add_argument("--checkpoint", required=True, help="FoldMapper *.pt checkpoint.")
    p.add_argument("--profile", default=None, help="Precomputed profile [L_chain, 21] .npy.")
    p.add_argument("--foldseek-db", default=None, help="Foldseek DB; runs easy-search on the first design chain.")
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


def _parse_chains(s: str) -> List[str]:
    return [c.strip() for c in s.split(",") if c.strip()]


def _simulate_featurize_order(entry: Dict, seed: int) -> List[str]:
    """Return the chain order that featurize will use given this seed.

    Mirrors featurize's deduplication + shuffle logic so callers can
    determine the absolute position of each chain in the padded tensor.
    """
    masked = list(entry["masked_list"])
    visible = list(entry["visible_list"])

    # Deduplication: visible chains with the same sequence as a masked chain
    # get promoted to masked (featurize model_utils.py:60-66).
    masked_seqs = {l: entry[f"seq_chain_{l}"] for l in masked}
    visible_seqs = {l: entry[f"seq_chain_{l}"] for l in visible}
    for _, ms in masked_seqs.items():
        for vl, vs in list(visible_seqs.items()):
            if ms == vs and vl not in masked:
                masked.append(vl)
                visible.remove(vl)
                del visible_seqs[vl]

    all_chains = masked + visible
    random.seed(seed)
    random.shuffle(all_chains)
    return all_chains


def _chain_starts(entry: Dict, shuffled_order: List[str]) -> Dict[str, int]:
    """Map chain letter → starting absolute position in the concatenated tensor."""
    starts: Dict[str, int] = {}
    pos = 0
    for letter in shuffled_order:
        starts[letter] = pos
        pos += len(entry[f"seq_chain_{letter}"])
    return starts


def _build_tied_groups(
    entry: Dict,
    tie_set: Set[str],
    chain_starts: Dict[str, int],
) -> List[List[int]]:
    """Build tied_groups for model.sample() from a set of chain letters to tie.

    Each inner list is one position group across all tied chains. All chains
    in the tie set must have the same length (homo-oligomer assumption).
    """
    letters = sorted(tie_set)
    lengths = [len(entry[f"seq_chain_{l}"]) for l in letters]
    if len(set(lengths)) != 1:
        raise ValueError(
            f"Tied chains must have equal length. Got: {dict(zip(letters, lengths))}"
        )
    chain_len = lengths[0]
    return [
        [chain_starts[letter] + i for letter in letters]
        for i in range(chain_len)
    ]


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


def _load_profile(args, chain_len: int) -> Optional[np.ndarray]:
    if args.profile:
        prof = np.load(args.profile).astype(np.float32)
        if prof.shape[0] != chain_len:
            print(
                f"[warn] profile length {prof.shape[0]} != chain length {chain_len};"
                " truncating/padding.",
                file=sys.stderr,
            )
            out = np.zeros((chain_len, 21), dtype=np.float32)
            L = min(prof.shape[0], chain_len)
            out[:L] = prof[:L]
            return out
        return prof
    if args.foldseek_db:
        import tempfile
        from Bio.PDB import PDBIO, PDBParser, Select

        first_chain = _parse_chains(args.chains_to_design)[0]
        print(
            f"[foldseek] generating profile for chain {first_chain}",
            file=sys.stderr,
        )

        class _ChainSelect(Select):
            def accept_chain(self, chain):
                return chain.get_id() == first_chain

        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as f:
            tmp_pdb = f.name
        try:
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure("s", args.pdb)
            io = PDBIO()
            io.set_structure(structure)
            io.save(tmp_pdb, _ChainSelect())
            return profile_from_structure(
                tmp_pdb,
                args.foldseek_db,
                chain_len,
                foldseek_bin=args.foldseek_bin,
                topk=args.foldseek_topk,
            )
        finally:
            os.unlink(tmp_pdb)
    return None


def main():
    args = build_parser().parse_args()

    design_chains = _parse_chains(args.chains_to_design)
    context_chains = _parse_chains(args.context_chains)
    tie_chains: Set[str] = set(_parse_chains(args.tie_chains))

    if tie_chains and not tie_chains.issubset(set(design_chains)):
        sys.exit(
            f"[error] --tie-chains {sorted(tie_chains)} must be a subset of"
            f" --chains-to-design {design_chains}"
        )

    # Seed all three independent random states for full reproducibility.
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    entry = load_multichain_structure(
        args.pdb,
        masked_chains=design_chains,
        context_chains=context_chains,
    )
    design_lens = {c: len(entry[f"seq_chain_{c}"]) for c in design_chains}
    context_lens = {c: len(entry[f"seq_chain_{c}"]) for c in context_chains}
    print(
        f"[load] {args.pdb}: design={design_lens}, context={context_lens}",
        file=sys.stderr,
    )

    model, _ = load_model_from_checkpoint(args.checkpoint, device=device)
    print(
        f"[model] use_blosum={model.config.use_blosum}, hidden_dim={model.config.hidden_dim}",
        file=sys.stderr,
    )

    # featurize uses Python random.shuffle internally; seed is already set.
    X, S, mask, lengths, chain_M, residue_idx, mask_self, chain_encoding_all, _ = featurize(
        [entry], device
    )

    # Determine the featurize chain order with the same seed to build tied_groups.
    shuffled_order = _simulate_featurize_order(entry, args.seed)
    starts = _chain_starts(entry, shuffled_order)

    if args.fixed_residues:
        fixed = _parse_fixed_residues(args.fixed_residues)
        _apply_fixed_residues(chain_M, fixed, entry, starts)
        n_fixed = sum(len(v) for v in fixed.values())
        print(f"[fixed] {n_fixed} residue(s) locked to native sequence", file=sys.stderr)

    tied_groups: Optional[List[List[int]]] = None
    if tie_chains:
        tied_groups = _build_tied_groups(entry, tie_chains, starts)
        print(
            f"[tied] {sorted(tie_chains)}: {len(tied_groups)} position groups",
            file=sys.stderr,
        )

    # Build evolutionary profile tensor if provided.
    first_design_chain = design_chains[0]
    first_chain_len = design_lens[first_design_chain]
    profile_np = _load_profile(args, first_chain_len)
    if profile_np is not None:
        evo_profile = torch.zeros(1, X.shape[1], 21, device=device)
        if tie_chains and first_design_chain in tie_chains:
            # Broadcast profile to all tied chain slices.
            for letter in tie_chains:
                start = starts[letter]
                chain_len = design_lens[letter]
                evo_profile[0, start:start + chain_len] = torch.from_numpy(profile_np).to(device)
        else:
            start = starts[first_design_chain]
            evo_profile[0, start:start + first_chain_len] = (
                torch.from_numpy(profile_np).to(device)
            )
    else:
        evo_profile = None

    fasta_lines = []
    t0 = time.time()

    for i in range(args.n_samples):
        out = model.sample(
            X, mask, chain_M, residue_idx, chain_encoding_all,
            evo_profile=evo_profile,
            temperature=args.temperature,
            S=S,
            tied_groups=tied_groups,
        )
        S_out = out["S"][0]

        # Extract per-chain designed sequences and compute recovery.
        chain_results = {}
        for ch in design_chains:
            start = starts[ch]
            ch_len = design_lens[ch]
            sampled_idx = S_out[start:start + ch_len].tolist()
            pred = "".join(IDX_TO_AA.get(idx, "X") for idx in sampled_idx)
            native = entry[f"seq_chain_{ch}"]
            recovery = sum(a == b for a, b in zip(native, pred)) / ch_len if ch_len else 0.0
            chain_results[ch] = (pred, recovery)

        chains_tag = "+".join(design_chains)
        header = (
            f">{entry['name']}|chains={chains_tag}|sample={i}"
            f"|T={args.temperature}"
        )
        if tie_chains:
            # For tied design, report one representative chain.
            rep = sorted(tie_chains)[0]
            pred_rep, rec_rep = chain_results[rep]
            header += f"|tied={''.join(sorted(tie_chains))}|recovery={rec_rep:.3f}"
            seq_out = pred_rep
        else:
            # Multi-chain: concatenate with '/' separator, avg recovery.
            avg_rec = sum(r for _, r in chain_results.values()) / len(chain_results)
            header += f"|recovery={avg_rec:.3f}"
            seq_out = "/".join(pred for pred, _ in (chain_results[c] for c in design_chains))

        fasta_lines.append(f"{header}\n{seq_out}")

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
