# FoldMapper

Structure-based evolutionary-profile inverse folding.

FoldMapper is a two-stage graph neural network for protein inverse folding.
Stage 1 distils a per-residue evolutionary profile (obtained by Foldseek
structural alignment of the query against a structure database) into node
features over the backbone graph. Stage 2 is a graph encoder/decoder that
autoregressively samples a sequence consistent with the backbone + profile.

## Installation

Requires Python 3.9+, PyTorch 2.3.1 (CUDA 12.1 tested), and Foldseek (only
needed to generate new profiles).

```bash
conda env create -f environment.yml
conda activate foldmapper
pip install -e .
```

## Model variants

| Paper label             | `d_s1` | BLOSUM62 Stage-1 bias | Config                     |
|-------------------------|-------:|-----------------------|----------------------------|
| FoldMapper-29.5M        |    256 | off                   | `configs/d256.yaml`        |
| FoldMapper-40.0M        |    512 | off                   | `configs/d512.yaml`        |
| FoldMapper-29.5M+BLOSUM |    256 | on                    | `configs/d256_blosum.yaml` |
| FoldMapper-40.0M+BLOSUM |    512 | on                    | `configs/d512_blosum.yaml` |

Checkpoints are distributed separately. See `docs/checkpoints.md` for the
variant ↔ Zenodo DOI mapping.

## Quickstart: single PDB → sequences

```bash
# 1. Build a Foldseek evolutionary profile (requires the PDB100 database, ~20 GB)
#    See "Foldseek database" below for the exact download command.
bash scripts/make_profile.sh input.pdb /path/to/foldseek_db/pdb profiles/

# 2. Sample 100 sequences at temperature 0.1
python scripts/sample.py \
    --pdb input.pdb \
    --profile profiles/input.npy \
    --checkpoint /path/to/foldmapper_d256_blosum.pt \
    --n-samples 100 \
    --temperature 0.1 \
    --output-fasta samples.fasta
```

`sample.py` also accepts `--foldseek-db DB` instead of `--profile`, in which
case the profile is generated on the fly. If neither is supplied, the model
runs in backbone-only mode (the same regime triggered by `profile_dropout`
during training).

For the other design tasks (binder redesign, tied homo-oligomer, fixed-residue
motif scaffolding), see `examples/` — one runnable shell script per scenario.

## Foldseek database

FoldMapper is trained and evaluated against the official Foldseek PDB100
database. We do not redistribute it; use the upstream source:

```bash
conda install -c bioconda foldseek         # v10 or newer
foldseek databases PDB100 /path/to/db tmp  # ~20 GB
```

## Training

Reproduces the paper training run for any variant. The cluster splits, per-chain
metadata, and the pipeline that materializes the PDB/TED `.pt` shard tree are
distributed as a separate Zenodo record — see the "Data" section below.

The shards that pipeline produces carry backbone coordinates and the native
sequence. FoldMapper additionally conditions on a cached Foldseek profile, which
is not part of the dataset record — generate it before training:

```bash
# Run Foldseek once per shard first (same parameters as scripts/make_profile.sh;
# pinned Foldseek/PDB100 versions in docs/checkpoints.md), then attach profiles.
# --manifest is a newline-separated list of absolute .pt paths; each shard is
# rewritten under --output-dir at its path relative to --input-base, so the
# shard tree layout is preserved.
export SHARDS_WITH_PROFILE="${SUBSET_DIR}_with_profile"
find "${SUBSET_DIR}" -name '*.pt' > shards.txt
python scripts/make_profile_batch.py \
    --manifest shards.txt \
    --m8-dir <foldseek_results> \
    --input-base "${SUBSET_DIR}" \
    --output-dir "${SHARDS_WITH_PROFILE}"
```

Training silently falls back to a zero profile for any shard missing
`evo_profile`, so check the driver's summary counters (`missing .m8`,
`all-zero profiles`) rather than assuming success.

```bash
# Prereq: source the Zenodo training-data record's config.env, which exports
# SPLITS_DIR, METADATA_DIR, SUBSET_DIR, and FOLDMAPPER_DATA_BASE.
# Point the trainer at the profile-augmented tree produced above, not SUBSET_DIR.
export FOLDMAPPER_DATA_BASE="${SHARDS_WITH_PROFILE}"
python scripts/train.py \
    --config configs/d256_blosum.yaml \
    --path-for-training-data "${SPLITS_DIR}" \
    --ted-root "${SHARDS_WITH_PROFILE}/ted" \
    --ted-metadata "${METADATA_DIR}/ted_metadata.json" \
    --pdb-metadata "${METADATA_DIR}/pdb_metadata.json" \
    --path-for-outputs ./outputs \
    --use-wandb --wandb-project foldmapper
```

PDB shard paths are resolved via `pdb_metadata.json` + `FOLDMAPPER_DATA_BASE`;
TED shard paths combine `--ted-root` (for the initial existence check) with
`ted_metadata.json` (for sibling-domain lookup, also resolved through
`FOLDMAPPER_DATA_BASE`).

## Reproducing paper tables

Inference for the benchmarks in the paper (PDB/TED recovery, PPIBindBench,
scTM) is driven by the same `scripts/sample.py` — loop over test structures,
save profiles once, and sample 100 sequences per backbone at temperature 0.1.
Wrapper scripts will be added alongside the camera-ready release.

## Data

The training-data record (cluster splits + per-chain metadata + the data-prep
pipeline that materializes the `.pt` shard tree from PDB + TED sources) is
released separately as a Zenodo deposit; see `docs/checkpoints.md` for the DOI
mapping. The record ships splits, metadata, and the data-prep pipeline; the
shards themselves are regenerated from public PDB + TED sources and contain
backbone coords and the native sequence. The `evo_profile` tensor consumed by
the trainer is FoldMapper-specific and is produced by
`scripts/make_profile_batch.py` in Foldseek AA order
(`A C D E F G H I K L M N P Q R S T V W Y X`).

## Citation

If you use FoldMapper, please cite the FoldMapper paper (Kim et al., 2026).
The full reference and DOI are added here once the preprint is posted.

## License & attribution

FoldMapper is released under the Apache License 2.0 (`LICENSE`). Portions
are adapted from [ProteinMPNN](https://github.com/dauparas/ProteinMPNN)
(Dauparas et al., 2022) under the MIT License; see `NOTICE` for details.
