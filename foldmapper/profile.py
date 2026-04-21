"""Foldseek-based evolutionary profile utilities.

Pipeline:
    PDB/CIF query  --(foldseek easy-search)-->  .m8 alignments  -->
    per-position amino-acid profile in Foldseek alphabet (A C D E F G H I K L M N P Q R S T V W Y X)
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from typing import Iterable, List, Optional

import numpy as np

from .blosum import FOLDSEEK_AA_ALPHABET


AA_TO_IDX = {aa: i for i, aa in enumerate(FOLDSEEK_AA_ALPHABET)}
IDX_TO_AA = {i: aa for aa, i in AA_TO_IDX.items()}


def run_foldseek_search(
    query_structure: str,
    db_path: str,
    output_m8: str,
    foldseek_bin: str = "foldseek",
    alignment_type: int = 0,
    max_seqs: int = 100,
    threads: int = 4,
    evalue: float = 10.0,
    timeout: Optional[int] = 600,
    tmp_dir: Optional[str] = None,
) -> bool:
    """Run ``foldseek easy-search`` against a prebuilt Foldseek database.

    ``alignment_type=0`` uses 3Di-only alignments (recommended for FoldMapper).
    Returns True on success.
    """
    if not os.path.exists(query_structure):
        raise FileNotFoundError(query_structure)

    cleanup = False
    if tmp_dir is None:
        tmp_dir = tempfile.mkdtemp(prefix="foldseek_")
        cleanup = True

    cmd = [
        foldseek_bin, "easy-search",
        query_structure,
        db_path,
        output_m8,
        tmp_dir,
        "--format-output", "query,target,qstart,qend,tstart,tend,qlen,tlen,qaln,taln,evalue,bits",
        "--alignment-type", str(alignment_type),
        "--max-seqs", str(max_seqs),
        "--threads", str(threads),
        "-e", str(evalue),
        "-v", "0",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if result.returncode != 0:
            print(f"[foldseek] failed: {result.stderr}", file=sys.stderr)
            return False
        return os.path.exists(output_m8)
    finally:
        if cleanup:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)


def parse_foldseek_m8(m8_path: str, topk: int = 100) -> List[dict]:
    """Parse a Foldseek ``.m8`` file into a list of alignment dicts.

    Expected columns (from the ``--format-output`` above):
        query target qstart qend tstart tend qlen tlen qaln taln evalue bits
    """
    alignments: List[dict] = []
    if not os.path.exists(m8_path):
        return alignments
    with open(m8_path) as f:
        for i, line in enumerate(f):
            if i >= topk:
                break
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 12:
                continue
            query, target = parts[0], parts[1]
            if query == target:
                continue
            alignments.append(
                {
                    "query": query,
                    "target": target,
                    "qstart": int(parts[2]),
                    "qend": int(parts[3]),
                    "tstart": int(parts[4]),
                    "tend": int(parts[5]),
                    "qlen": int(parts[6]),
                    "tlen": int(parts[7]),
                    "qaln": parts[8],
                    "taln": parts[9],
                    "evalue": float(parts[10]),
                    "bits": float(parts[11]),
                }
            )
    return alignments


def compute_profile_from_alignments(
    alignments: Iterable[dict],
    query_len: int,
    weight_by_bits: bool = True,
) -> np.ndarray:
    """Build a per-position amino-acid distribution ``[query_len, 21]``.

    For each alignment, the target residue at every aligned query column contributes
    to that column's count, weighted by bit score (or uniformly if
    ``weight_by_bits=False``). Insertions in the query (``q_char == '-'``) are
    ignored; insertions in the target (``t_char == '-'``) advance the query cursor
    only.
    """
    counts = np.zeros((query_len, 21), dtype=np.float32)
    for aln in alignments:
        qaln = aln["qaln"]
        taln = aln["taln"]
        qstart = aln["qstart"]
        weight = aln["bits"] if weight_by_bits else 1.0

        q_pos = qstart - 1
        for q_char, t_char in zip(qaln, taln):
            if q_char == "-":
                continue
            if t_char == "-":
                q_pos += 1
                continue
            if 0 <= q_pos < query_len:
                aa_idx = AA_TO_IDX.get(t_char.upper())
                if aa_idx is not None:
                    counts[q_pos, aa_idx] += weight
            q_pos += 1

    row_sums = counts.sum(axis=1, keepdims=True)
    profile = np.divide(counts, row_sums, out=np.zeros_like(counts), where=row_sums > 0)
    return profile.astype(np.float32)


def profile_from_structure(
    query_structure: str,
    db_path: str,
    query_len: int,
    foldseek_bin: str = "foldseek",
    topk: int = 100,
    alignment_type: int = 0,
    threads: int = 4,
) -> np.ndarray:
    """Convenience: run Foldseek and return a ``[query_len, 21]`` profile."""
    with tempfile.TemporaryDirectory(prefix="foldseek_") as tmp:
        m8 = os.path.join(tmp, "query.m8")
        ok = run_foldseek_search(
            query_structure,
            db_path,
            m8,
            foldseek_bin=foldseek_bin,
            alignment_type=alignment_type,
            max_seqs=topk,
            threads=threads,
            tmp_dir=tmp,
        )
        if not ok:
            return np.zeros((query_len, 21), dtype=np.float32)
        alignments = parse_foldseek_m8(m8, topk=topk)
    return compute_profile_from_alignments(alignments, query_len)
