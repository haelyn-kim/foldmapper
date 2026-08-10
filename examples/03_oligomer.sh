#!/usr/bin/env bash
# Redesign a homo-oligomer with a tied sequence across symmetric chains.
#
# Usage (C3 homo-trimer, chains A/B/C, all tied):
#   PDB=/path/to/trimer.pdb \
#   DESIGN_CHAINS=A,B,C TIE_CHAINS=A,B,C \
#   CKPT=/path/to/foldmapper_d256.pt \
#   FOLDSEEK_DB=/path/to/pdb100_db \
#   bash examples/03_oligomer.sh
#
# `TIE_CHAINS` must be a subset of `DESIGN_CHAINS`; the listed chains will
# share a single decoded sequence (broadcast at decoder time).
#
# Optional env vars:
#   CONTEXT_CHAINS  chains kept as fixed structural context (default: none)
#   N_SAMPLES       sequences to draw (default: 10)
#   TEMPERATURE     sampling temperature (default: 0.1)

set -euo pipefail

: "${PDB:?set PDB to your oligomer structure}"
: "${DESIGN_CHAINS:?set DESIGN_CHAINS, e.g. A,B,C}"
: "${TIE_CHAINS:?set TIE_CHAINS to the chains that must share a sequence}"
: "${CKPT:?set CKPT to a FoldMapper checkpoint}"
: "${FOLDSEEK_DB:?set FOLDSEEK_DB to a Foldseek PDB100 database}"

CONTEXT_CHAINS="${CONTEXT_CHAINS:-}"
N_SAMPLES="${N_SAMPLES:-10}"
TEMPERATURE="${TEMPERATURE:-0.1}"

cmd=(python scripts/design_partial.py
    --pdb "$PDB"
    --chains-to-design "$DESIGN_CHAINS"
    --tie-chains "$TIE_CHAINS"
    --checkpoint "$CKPT"
    --foldseek-db "$FOLDSEEK_DB"
    --n-samples "$N_SAMPLES"
    --temperature "$TEMPERATURE")
[[ -n "$CONTEXT_CHAINS" ]] && cmd+=(--context-chains "$CONTEXT_CHAINS")

"${cmd[@]}"
