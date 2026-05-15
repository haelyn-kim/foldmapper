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

| Variant           | `d_s1` | BLOSUM62 Stage-1 bias | Config                          |
|-------------------|-------:|-----------------------|---------------------------------|
| FoldMapper-d256   |    256 | off                   | `configs/d256.yaml`             |
| FoldMapper-d512   |    512 | off                   | `configs/d512.yaml`             |
| FoldMapper-d256-B |    256 | on                    | `configs/d256_blosum.yaml`      |
| FoldMapper-d512-B |    512 | on                    | `configs/d512_blosum.yaml`      |

Checkpoints are distributed separately. See `docs/checkpoints.md` for the
variant ↔ Zenodo DOI mapping (available once the manuscript is accepted).

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

## Foldseek database

FoldMapper is trained and evaluated against the official Foldseek PDB100
database. We do not redistribute it; use the upstream source:

```bash
conda install -c bioconda foldseek         # v10 or newer
foldseek databases PDB100 /path/to/db tmp  # ~20 GB
```

## Training

Reproduces the paper training run for any variant. The PDB/TED `.pt` shard
dataset (per-protein backbone + `evo_profile` + metadata JSONs) is distributed
as a separate Zenodo record — see the "Data" section below.

```bash
# Prereq: source the Zenodo training-data record's config.env, which exports
# SPLITS_DIR, METADATA_DIR, SUBSET_DIR, and FOLDMAPPER_DATA_BASE.
export FOLDMAPPER_DATA_BASE="${SUBSET_DIR}"
python scripts/train.py \
    --config configs/d256_blosum.yaml \
    --path-for-training-data "${SPLITS_DIR}" \
    --ted-root "${SUBSET_DIR}/ted" \
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
mapping. Each `.pt` shard contains backbone coords, the native sequence, and a
cached `evo_profile` tensor in Foldseek AA order
(`A C D E F G H I K L M N P Q R S T V W Y X`).

## Citation

```
@article{foldmapper,
  title  = {FoldMapper},
  author = {Kim, Haelyn and colleagues},
  year   = {2026},
  journal= {Nature Communications (submitted)}
}
```

## License & attribution

FoldMapper is released under the Apache License 2.0 (`LICENSE`). Portions
are adapted from [ProteinMPNN](https://github.com/dauparas/ProteinMPNN)
(Dauparas et al., 2022) under the MIT License; see `NOTICE` for details.
