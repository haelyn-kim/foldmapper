#!/bin/bash
# Integration test for foldmapper design scripts.
#
# Required environment variables:
#   FOLDMAPPER_CKPT          path to a FoldMapper *.pt checkpoint
#   FOLDMAPPER_TEST_MONOMER  single-chain PDB (any chain)
#   FOLDMAPPER_TEST_BINDER   two-chain PDB; chain B = binder (20 res),
#                            chain A = target. Native binder B:1-5 must be
#                            known for the fixed-residues assertion.
#   FOLDMAPPER_TEST_TRIMER   C3 symmetric homo-trimer PDB (chains A, B, C,
#                            all same length).
#
# Optional:
#   FOLDMAPPER_PYTHON        python interpreter (default: python)
#   FOLDMAPPER_REPO          repo root (default: parent of this script)
#   FOLDMAPPER_TRIMER_LEN    per-chain length of the trimer (default: 205)
#   FOLDMAPPER_BINDER_NATIVE_PREFIX   native first-5 AA of the binder
#                                      (default: RRKWQ, matches 9mxd_B_A)
#
# Usage:
#   export FOLDMAPPER_CKPT=/path/to/ckpt.pt
#   export FOLDMAPPER_TEST_MONOMER=/path/to/monomer.pdb
#   export FOLDMAPPER_TEST_BINDER=/path/to/binder_complex.pdb
#   export FOLDMAPPER_TEST_TRIMER=/path/to/trimer.pdb
#   bash tests/test_design.sh

set -e

PYTHON="${FOLDMAPPER_PYTHON:-python}"
REPO="${FOLDMAPPER_REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
CKPT="${FOLDMAPPER_CKPT:?FOLDMAPPER_CKPT not set}"
MONO_PDB="${FOLDMAPPER_TEST_MONOMER:?FOLDMAPPER_TEST_MONOMER not set}"
BINDER_PDB="${FOLDMAPPER_TEST_BINDER:?FOLDMAPPER_TEST_BINDER not set}"
TRIMER_PDB="${FOLDMAPPER_TEST_TRIMER:?FOLDMAPPER_TEST_TRIMER not set}"
TRIMER_LEN="${FOLDMAPPER_TRIMER_LEN:-205}"
BINDER_PREFIX="${FOLDMAPPER_BINDER_NATIVE_PREFIX:-RRKWQ}"

pass() { echo "[PASS] $1"; }
fail() { echo "[FAIL] $1"; FAILED=1; }
FAILED=0

echo "=========================================="
echo "Test 1: sample.py — single chain"
echo "=========================================="
OUT1=$($PYTHON $REPO/scripts/sample.py \
    --pdb "$MONO_PDB" --checkpoint "$CKPT" \
    --seed 42 --n-samples 2 2>/dev/null)
echo "$OUT1"
COUNT=$(echo "$OUT1" | grep -c "^>" || true)
[ "$COUNT" -eq 2 ] && pass "produced 2 sequences" || fail "expected 2 seqs, got $COUNT"

echo ""
echo "=========================================="
echo "Test 2: design_binder.py — no profile"
echo "=========================================="
OUT2=$($PYTHON $REPO/scripts/design_binder.py \
    --pdb "$BINDER_PDB" --binder-chain B --target-chain A \
    --checkpoint "$CKPT" --seed 42 --n-samples 2 2>/dev/null)
echo "$OUT2"
COUNT=$(echo "$OUT2" | grep -c "^>" || true)
[ "$COUNT" -eq 2 ] && pass "produced 2 sequences" || fail "expected 2 seqs, got $COUNT"

echo ""
echo "=========================================="
echo "Test 3: design_binder.py --fixed-residues B:1-5 → prefix must be $BINDER_PREFIX"
echo "=========================================="
OUT3=$($PYTHON $REPO/scripts/design_binder.py \
    --pdb "$BINDER_PDB" --binder-chain B --target-chain A \
    --checkpoint "$CKPT" --seed 42 --n-samples 3 \
    --fixed-residues "B:1-5" 2>/dev/null)
echo "$OUT3"
BAD=0
while IFS= read -r line; do
    [[ "$line" == ">"* ]] && continue
    [[ -z "$line" ]] && continue
    PREFIX="${line:0:5}"
    [ "$PREFIX" != "$BINDER_PREFIX" ] && { echo "  → got prefix '$PREFIX'"; BAD=1; }
done <<< "$OUT3"
[ "$BAD" -eq 0 ] && pass "all samples start with $BINDER_PREFIX" \
                 || fail "fixed-residues not honoured"

echo ""
echo "=========================================="
echo "Test 4: design_partial.py — tied trimer CLI (A,B,C same sequence)"
echo "=========================================="
OUT4=$($PYTHON $REPO/scripts/design_partial.py \
    --pdb "$TRIMER_PDB" \
    --chains-to-design A,B,C --tie-chains A,B,C \
    --checkpoint "$CKPT" --seed 42 --n-samples 1 2>/dev/null)
echo "$OUT4"
COUNT=$(echo "$OUT4" | grep -c "^>" || true)
[ "$COUNT" -eq 1 ] && pass "produced 1 sequence" || fail "expected 1 seq, got $COUNT"

echo ""
echo "=========================================="
echo "Test 5: tied trimer — verify all 3 chains identical at tensor level"
echo "=========================================="
FOLDMAPPER_CKPT="$CKPT" FOLDMAPPER_TEST_TRIMER="$TRIMER_PDB" \
FOLDMAPPER_TRIMER_LEN="$TRIMER_LEN" FOLDMAPPER_REPO="$REPO" \
$PYTHON - <<'PYEOF'
import os, sys, random
sys.path.insert(0, os.environ["FOLDMAPPER_REPO"])
import torch
import numpy as np

from foldmapper import load_model_from_checkpoint
from foldmapper.io import load_multichain_structure
from foldmapper.model_utils import featurize
from foldmapper.profile import IDX_TO_AA

CKPT = os.environ["FOLDMAPPER_CKPT"]
PDB  = os.environ["FOLDMAPPER_TEST_TRIMER"]
CHAIN_LEN = int(os.environ.get("FOLDMAPPER_TRIMER_LEN", "205"))
SEED = 42

random.seed(SEED); torch.manual_seed(SEED); np.random.seed(SEED)
entry = load_multichain_structure(PDB, ['A', 'B', 'C'])
model, _ = load_model_from_checkpoint(CKPT, device='cpu')
X, S, mask, lengths, chain_M, residue_idx, mask_self, chain_enc, _ = featurize([entry], 'cpu')

masked = list(entry['masked_list'])
visible = list(entry['visible_list'])
all_chains = masked + visible
random.seed(SEED)
random.shuffle(all_chains)
starts, pos = {}, 0
for ch in all_chains:
    starts[ch] = pos
    pos += len(entry[f'seq_chain_{ch}'])

tied_groups = [
    [starts['A'] + i, starts['B'] + i, starts['C'] + i]
    for i in range(CHAIN_LEN)
]

out = model.sample(X, mask, chain_M, residue_idx, chain_enc,
                   temperature=0.1, S=S, tied_groups=tied_groups)
S_out = out['S'][0]

seqs = {ch: ''.join(IDX_TO_AA.get(int(idx), 'X')
                    for idx in S_out[starts[ch]:starts[ch]+CHAIN_LEN])
        for ch in ['A', 'B', 'C']}

if seqs['A'] == seqs['B'] == seqs['C']:
    print('[PASS] tied trimer: A == B == C')
    print('  seq:', seqs['A'][:30], '...')
else:
    dAB = sum(a != b for a, b in zip(seqs['A'], seqs['B']))
    dAC = sum(a != b for a, b in zip(seqs['A'], seqs['C']))
    print(f'[FAIL] tied trimer: A-B diffs={dAB}, A-C diffs={dAC}')
    sys.exit(1)
PYEOF
[ $? -eq 0 ] || FAILED=1

echo ""
if [ "$FAILED" -eq 0 ]; then
    echo "=========================================="
    echo "All tests PASSED."
    echo "=========================================="
else
    echo "=========================================="
    echo "Some tests FAILED — check output above."
    echo "=========================================="
    exit 1
fi
