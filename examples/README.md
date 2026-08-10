# Example workflows

Four end-to-end usage examples for FoldMapper. Each script runs a single
scenario from PDB + checkpoint to sampled sequences. Bring your own input PDB
and checkpoint; nothing is hard-coded.

## Prerequisites

1. **FoldMapper installed** (from the repo root):
   ```bash
   pip install -e .
   ```

2. **Foldseek + PDB100 database** for evolutionary profiles. FoldMapper is
   trained against the official Foldseek PDB100 DB; install Foldseek (>= v10)
   and download/index the DB once:
   ```bash
   # https://github.com/steineggerlab/foldseek
   foldseek databases PDB100 /path/to/pdb100_db /path/to/tmp
   ```

3. **A checkpoint.** Slim checkpoints for the four released variants are
   attached to the GitHub release. Default to `foldmapper_d256.pt` unless
   you need a specific architecture/regularisation:

   | File | Stage 1 / Stage 2 dim | BLOSUM62 |
   |---|---|---|
   | `foldmapper_d256.pt`        | 256 / 256 | no  |
   | `foldmapper_d256_blosum.pt` | 256 / 256 | yes |
   | `foldmapper_d512.pt`        | 256 / 512 | no  |
   | `foldmapper_d512_blosum.pt` | 256 / 512 | yes |

## Common environment

All scripts read the same two env vars:

```bash
export CKPT=/path/to/foldmapper_d256.pt
export FOLDSEEK_DB=/path/to/pdb100_db
```

Each scenario takes its own `PDB`-style input (see the script header).

## Scenarios

| Script | Task |
|---|---|
| `01_monomer.sh` | Sample N sequences for a single-chain backbone. |
| `02_binder.sh`  | Redesign a binder chain while keeping the target chain as fixed structural context. |
| `03_oligomer.sh`| Redesign a homo-oligomer with a tied sequence across symmetric chains. |
| `04_fixed_residues.sh` | Binder design with a fixed-residue motif (e.g. interface hotspots kept native). |

Run any script with:

```bash
PDB=/path/to/input.pdb CKPT=... FOLDSEEK_DB=... bash examples/01_monomer.sh
```

Outputs (FASTA) go to `stdout` by default; redirect or pass `--output-fasta`
inside the script if you prefer files.

## Building the profile separately

If you want to inspect or cache the profile before sampling:

```bash
bash scripts/make_profile.sh input.pdb "$FOLDSEEK_DB" ./profile_out
# -> ./profile_out/input.npy  (per-position [L, 21] Foldseek-AA profile)
```

The sampling scripts can consume this via `--profile path/to/input.npy`
instead of `--foldseek-db`, which avoids re-running the search.
