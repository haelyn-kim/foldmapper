#!/usr/bin/env python3
"""Attach cached ``evo_profile`` tensors to a tree of training ``.pt`` shards.

``scripts/make_profile.sh`` builds a profile for a single structure by running
Foldseek on the fly. For training-scale corpora (~2.1M shards) Foldseek is run
once as a separate batch step, and this driver turns the resulting ``.m8``
alignments into the ``evo_profile`` tensor that ``scripts/train.py`` consumes.

The profile math lives in :mod:`foldmapper.profile` and is shared with
``make_profile.sh`` — this script only does I/O plumbing.

Query length is taken from the shard's own ``seq`` field, NOT from the ``.m8``
``qlen`` column: the two disagree whenever the sequence a structure yields is
shorter than the FASTA it was indexed under, and the trainer indexes profiles
against the shard sequence.

Usage:
    python scripts/make_profile_batch.py \
        --manifest chunk_00000.txt \
        --m8-dir /path/to/foldseek_results \
        --input-base /path/to/training_data_subset \
        --output-dir /path/to/pt_with_profile \
        --topk 100 --num-workers 2

The manifest is a newline-separated list of ``.pt`` paths. Each shard is written
to ``--output-dir`` under its path relative to ``--input-base``, so the shard
tree layout is preserved. We ran this as a SLURM array over 106 manifest chunks;
any array scheduler or a plain loop works equally well.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from typing import Optional

import numpy as np
import torch

# Allow running as a plain script from a source checkout.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from foldmapper.profile import compute_profile_from_alignments, parse_foldseek_m8


def process_shard(
    pt_path: str,
    m8_dir: str,
    input_base: str,
    output_dir: str,
    topk: int,
) -> tuple[bool, str, Counter]:
    """Add ``evo_profile`` to one shard. Returns (ok, message, stats)."""
    stats: Counter = Counter()
    try:
        basename = os.path.basename(pt_path)[: -len(".pt")]
        m8_path = os.path.join(m8_dir, f"{basename}.m8")

        data = torch.load(pt_path, map_location="cpu")
        query_len = len(data["seq"])

        if not os.path.exists(m8_path):
            stats["missing_m8"] += 1
        elif os.path.getsize(m8_path) == 0:
            stats["empty_m8"] += 1

        alignments = parse_foldseek_m8(m8_path, topk=topk)
        if not alignments:
            # No usable hits (absent, empty, or self-hits only) -> no conditioning.
            stats["no_alignments"] += 1
            profile = np.zeros((query_len, 21), dtype=np.float32)
        else:
            profile = compute_profile_from_alignments(alignments, query_len)

        stats["hits"] += len(alignments)
        if not profile.any():
            stats["all_zero_profile"] += 1
        stats["zero_rows"] += int((profile.sum(axis=1) == 0).sum())
        stats["positions"] += query_len

        data["evo_profile"] = torch.from_numpy(profile)

        out_path = os.path.join(output_dir, os.path.relpath(pt_path, input_base))
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        torch.save(data, out_path)

        stats["ok"] += 1
        return True, basename, stats
    except Exception as exc:  # noqa: BLE001 - one bad shard must not kill the chunk
        stats["failed"] += 1
        return False, f"{pt_path}: {exc}", stats


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Attach Foldseek evo_profile tensors to training .pt shards."
    )
    ap.add_argument("--manifest", required=True, help="file listing .pt paths, one per line")
    ap.add_argument("--m8-dir", required=True, help="flat directory of <shard>.m8 alignments")
    ap.add_argument("--input-base", required=True, help="root the manifest paths are relative to")
    ap.add_argument("--output-dir", required=True, help="destination root for updated shards")
    ap.add_argument("--topk", type=int, default=100, help="top alignments per query (default: 100)")
    ap.add_argument("--num-workers", type=int, default=1, help="parallel worker processes")
    args = ap.parse_args()

    with open(args.manifest) as f:
        pt_files = [line.strip() for line in f if line.strip()]

    print(f"manifest      : {args.manifest} ({len(pt_files)} shards)")
    print(f"m8 dir        : {args.m8_dir}")
    print(f"input base    : {args.input_base}")
    print(f"output dir    : {args.output_dir}")
    print(f"topk          : {args.topk}")
    print("weighting     : bit score (matches foldmapper.profile)")

    os.makedirs(args.output_dir, exist_ok=True)

    totals: Counter = Counter()
    failures: list[str] = []

    def record(ok: bool, message: str, stats: Counter) -> None:
        totals.update(stats)
        if not ok and len(failures) < 20:
            failures.append(message)

    if args.num_workers > 1:
        with ProcessPoolExecutor(max_workers=args.num_workers) as pool:
            futures = [
                pool.submit(
                    process_shard, p, args.m8_dir, args.input_base, args.output_dir, args.topk
                )
                for p in pt_files
            ]
            for fut in futures:
                record(*fut.result())
    else:
        for p in pt_files:
            record(*process_shard(p, args.m8_dir, args.input_base, args.output_dir, args.topk))

    ok, failed = totals["ok"], totals["failed"]
    avg_hits = totals["hits"] / ok if ok else 0.0
    print("\n--- summary ---")
    print(f"written            : {ok}")
    print(f"failed             : {failed}")
    print(f"avg hits/shard     : {avg_hits:.1f}")
    # A missing or empty .m8 silently yields an all-zero profile, which is
    # indistinguishable from "Foldseek found nothing". Surface both.
    print(f"missing .m8        : {totals['missing_m8']}")
    print(f"empty .m8          : {totals['empty_m8']}")
    print(f"no usable hits     : {totals['no_alignments']}")
    print(f"all-zero profiles  : {totals['all_zero_profile']}")
    if totals["positions"]:
        frac = totals["zero_rows"] / totals["positions"]
        print(f"zero rows          : {totals['zero_rows']} / {totals['positions']} ({frac:.3%})")

    for msg in failures:
        print(f"FAILED {msg}", file=sys.stderr)

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
