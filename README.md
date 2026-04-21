# FoldMapper

Structure-based evolutionary-profile inverse folding.

FoldMapper is a two-stage graph neural network for protein inverse folding.
Stage 1 distils a per-residue evolutionary profile (obtained by Foldseek
structural alignment of the query against a structure database) into node
features over the backbone graph. Stage 2 is a ProteinMPNN-style
encoder/decoder that autoregressively samples a sequence consistent with the
backbone + profile.

## Installation

Requires Python 3.9+, PyTorch 2.3.1 (CUDA 12.1 tested), and Foldseek (only
needed to generate new profiles).

```bash
conda env create -f environment.yml
conda activate foldmapper
pip install -e .
```

## Model variants

| Variant           | `distill_hidden_dim` | BLOSUM62 Stage-1 bias | Config                          |
|-------------------|----------------------|-----------------------|---------------------------------|
| FoldMapper-d256   | 256                  | off                   | `configs/d256.yaml`             |
| FoldMapper-d512   | 512                  | off                   | `configs/d512.yaml`             |
| FoldMapper-d256-B | 256                  | on                    | `configs/d256_blosum.yaml`      |
| FoldMapper-d512-B | 512                  | on                    | `configs/d512_blosum.yaml`      |

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

Reproduces the paper training run for any variant. Data pipeline (PDB/TED
`.pt` shards with a cached `evo_profile` tensor per entry and metadata JSON
files) must be built separately; see the "Data" section below for the expected
layout.

```bash
python scripts/train.py \
    --config configs/d256_blosum.yaml \
    --path-for-training-data /path/to/dataset_splits/data \
    --pdb-root /path/to/pt_with_profile/pdb \
    --ted-root /path/to/pt_with_profile/ted \
    --ted-metadata /path/to/ted_metadata.json \
    --pdb-metadata /path/to/pdb_metadata.json \
    --path-for-outputs ./outputs \
    --use-wandb --wandb-project foldmapper
```

## Reproducing paper tables

Inference for the benchmarks in the paper (PDB/TED recovery, PPIBindBench,
scTM) is driven by the same `scripts/sample.py` — loop over test structures,
save profiles once, and sample 100 sequences per backbone at temperature 0.1.
Wrapper scripts will be added alongside the camera-ready release.

## Data

Training uses two data shards per protein: a `.pt` with backbone coords, the
native sequence, and a cached `evo_profile` tensor in Foldseek AA order
(`A C D E F G H I K L M N P Q R S T V W Y X`). The split lists and cluster
files follow the ProteinMPNN convention.

## Citation

```
@article{foldmapper,
  title  = {FoldMapper},
  author = {Kim, Haelyn and colleagues},
  year   = {2026},
  journal= {Nature Communications (submitted)}
}
```

## License

Apache-2.0. See `LICENSE`.
