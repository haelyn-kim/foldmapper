# Integration tests

End-to-end tests covering:

1. `sample.py` — monomer inverse folding regression
2. `design_binder.py` — two-chain binder redesign
3. `--fixed-residues` — residue-level locking by PDB number
4. `design_partial.py` — tied homo-oligomer design (CLI)
5. Tied design tensor-level verification (all chains identical sampled sequence)

## What you need

A FoldMapper checkpoint and three PDB files matching these properties:

| Variable | Content |
|----------|---------|
| `FOLDMAPPER_CKPT` | Any FoldMapper `.pt` checkpoint |
| `FOLDMAPPER_TEST_MONOMER` | Single-chain PDB |
| `FOLDMAPPER_TEST_BINDER` | Two-chain PDB; chain `B` is the binder (any length), chain `A` is the target |
| `FOLDMAPPER_TEST_TRIMER` | C3 symmetric homo-trimer (chains `A`, `B`, `C`, equal lengths) |

For Test 3 to assert correctness, also set:

```bash
export FOLDMAPPER_BINDER_NATIVE_PREFIX=<first 5 AA of chain B in your binder PDB>
```

(Default is `RRKWQ`, matching the native sequence of `9mxd_B_A.pdb` used during development.)

The trimer per-chain length defaults to `205`; override with `FOLDMAPPER_TRIMER_LEN` if needed.

## Run

```bash
export FOLDMAPPER_PYTHON=/path/to/conda/env/bin/python   # optional
export FOLDMAPPER_CKPT=/path/to/checkpoint.pt
export FOLDMAPPER_TEST_MONOMER=/path/to/monomer.pdb
export FOLDMAPPER_TEST_BINDER=/path/to/binder_complex.pdb
export FOLDMAPPER_TEST_TRIMER=/path/to/trimer.pdb
export FOLDMAPPER_BINDER_NATIVE_PREFIX=<5-letter prefix>

bash tests/test_design.sh
```

CPU-only run takes ~1 minute for the whole suite at the default sample counts.

## SLURM example

`tests/run_test_design.sh` is a reference SLURM wrapper for HPC environments.
Edit the paths at the top and submit with `sbatch`.
