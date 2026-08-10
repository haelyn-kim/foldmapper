#!/usr/bin/env bash
# Redesign a binder chain while keeping the target chain as fixed context.
#
# Usage:
#   PDB=/path/to/complex.pdb \
#   BINDER_CHAIN=B TARGET_CHAIN=A \
#   CKPT=/path/to/foldmapper_d256.pt \
#   FOLDSEEK_DB=/path/to/pdb100_db \
#   bash examples/02_binder.sh
#
# The Foldseek profile is built from the binder chain only (the target is
# present as structural context but not redesigned).
#
# Optional env vars:
#   N_SAMPLES    sequences to draw (default: 10)
#   TEMPERATURE  sampling temperature (default: 0.1)

set -euo pipefail

: "${PDB:?set PDB to your two-chain complex}"
: "${BINDER_CHAIN:?set BINDER_CHAIN to the chain to redesign}"
: "${TARGET_CHAIN:?set TARGET_CHAIN to the chain kept as fixed context}"
: "${CKPT:?set CKPT to a FoldMapper checkpoint}"
: "${FOLDSEEK_DB:?set FOLDSEEK_DB to a Foldseek PDB100 database}"

N_SAMPLES="${N_SAMPLES:-10}"
TEMPERATURE="${TEMPERATURE:-0.1}"

python scripts/design_binder.py \
    --pdb "$PDB" \
    --binder-chain "$BINDER_CHAIN" \
    --target-chain "$TARGET_CHAIN" \
    --checkpoint "$CKPT" \
    --foldseek-db "$FOLDSEEK_DB" \
    --n-samples "$N_SAMPLES" \
    --temperature "$TEMPERATURE"
