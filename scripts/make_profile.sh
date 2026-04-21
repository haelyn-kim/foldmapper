#!/usr/bin/env bash
# Build a FoldMapper-compatible Foldseek evolutionary profile.
#
# Usage:
#   bash scripts/make_profile.sh <input.pdb|.cif> <foldseek_db> <output_dir> [topk=100] [threads=4]
#
# Outputs (in <output_dir>):
#   <basename>.m8          raw Foldseek alignments (alignment-type 0, 3Di)
#   <basename>.npy         [L, 21] per-position profile in Foldseek alphabet
#
# Requires: foldseek (>= v10) on PATH, python with the foldmapper package importable.

set -euo pipefail

INPUT="${1:?input PDB/CIF}"
FOLDSEEK_DB="${2:?foldseek database}"
OUTDIR="${3:?output directory}"
TOPK="${4:-100}"
THREADS="${5:-4}"

mkdir -p "${OUTDIR}"
BASENAME="$(basename "${INPUT}")"
BASENAME="${BASENAME%.*}"

M8="${OUTDIR}/${BASENAME}.m8"
NPY="${OUTDIR}/${BASENAME}.npy"

TMPDIR="$(mktemp -d)"
trap 'rm -rf "${TMPDIR}"' EXIT

foldseek easy-search \
    "${INPUT}" \
    "${FOLDSEEK_DB}" \
    "${M8}" \
    "${TMPDIR}" \
    --format-output "query,target,qstart,qend,tstart,tend,qlen,tlen,qaln,taln,evalue,bits" \
    --alignment-type 0 \
    --max-seqs "${TOPK}" \
    --threads "${THREADS}" \
    -e 10 \
    -v 1

python - <<PY
from foldmapper.profile import parse_foldseek_m8, compute_profile_from_alignments
from foldmapper.io import load_structure
import numpy as np

entry = load_structure("${INPUT}")
aln = parse_foldseek_m8("${M8}", topk=${TOPK})
profile = compute_profile_from_alignments(aln, len(entry["seq"]))
np.save("${NPY}", profile)
print(f"wrote ${NPY}  shape=" + str(profile.shape))
PY
