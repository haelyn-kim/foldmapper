#!/usr/bin/env bash
# Binder design with a fixed-residue motif: redesign the binder chain except
# for a set of positions (e.g. interface hotspots) that are kept native.
#
# `FIXED_RESIDUES` uses PDB residue numbering: comma-separated
# `<chain>:<start>-<end>` (or `<chain>:<resnum>`) entries. Example:
#   FIXED_RESIDUES="B:1-5,B:12,B:20-22"
# keeps residues 1-5, 12, and 20-22 on chain B as their native amino acids.
#
# Usage:
#   PDB=/path/to/complex.pdb \
#   BINDER_CHAIN=B TARGET_CHAIN=A \
#   FIXED_RESIDUES="B:1-5" \
#   CKPT=/path/to/foldmapper_d256.pt \
#   FOLDSEEK_DB=/path/to/pdb100_db \
#   bash examples/04_fixed_residues.sh
#
# Optional env vars:
#   N_SAMPLES    sequences to draw (default: 10)
#   TEMPERATURE  sampling temperature (default: 0.1)

set -euo pipefail

: "${PDB:?set PDB to your two-chain complex}"
: "${BINDER_CHAIN:?set BINDER_CHAIN to the chain being redesigned}"
: "${TARGET_CHAIN:?set TARGET_CHAIN to the chain kept as fixed context}"
: "${FIXED_RESIDUES:?set FIXED_RESIDUES, e.g. B:1-5,B:12}"
: "${CKPT:?set CKPT to a FoldMapper checkpoint}"
: "${FOLDSEEK_DB:?set FOLDSEEK_DB to a Foldseek PDB100 database}"

N_SAMPLES="${N_SAMPLES:-10}"
TEMPERATURE="${TEMPERATURE:-0.1}"

python scripts/design_binder.py \
    --pdb "$PDB" \
    --binder-chain "$BINDER_CHAIN" \
    --target-chain "$TARGET_CHAIN" \
    --fixed-residues "$FIXED_RESIDUES" \
    --checkpoint "$CKPT" \
    --foldseek-db "$FOLDSEEK_DB" \
    --n-samples "$N_SAMPLES" \
    --temperature "$TEMPERATURE"
