#!/usr/bin/env bash
# Sample sequences for a single-chain backbone.
#
# Usage:
#   PDB=/path/to/monomer.pdb \
#   CKPT=/path/to/foldmapper_d256.pt \
#   FOLDSEEK_DB=/path/to/pdb100_db \
#   bash examples/01_monomer.sh
#
# Optional env vars:
#   CHAIN        chain id to design (default: first chain in the PDB)
#   N_SAMPLES    number of sequences to draw (default: 10)
#   TEMPERATURE  sampling temperature (default: 0.1)

set -euo pipefail

: "${PDB:?set PDB to your input structure}"
: "${CKPT:?set CKPT to a FoldMapper checkpoint}"
: "${FOLDSEEK_DB:?set FOLDSEEK_DB to a Foldseek PDB100 database}"

CHAIN="${CHAIN:-}"
N_SAMPLES="${N_SAMPLES:-10}"
TEMPERATURE="${TEMPERATURE:-0.1}"

cmd=(python scripts/sample.py
    --pdb "$PDB"
    --checkpoint "$CKPT"
    --foldseek-db "$FOLDSEEK_DB"
    --n-samples "$N_SAMPLES"
    --temperature "$TEMPERATURE")
[[ -n "$CHAIN" ]] && cmd+=(--chain "$CHAIN")

"${cmd[@]}"
