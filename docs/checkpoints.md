# Checkpoints

Checkpoints are distributed via Zenodo once the paper is accepted. Each variant
has one `epoch_last.pt` file (final epoch) plus the corresponding training
`log.txt` for reproducibility.

| Variant           | File                               | `use_blosum` | `distill_hidden_dim` | Training run (internal) |
|-------------------|------------------------------------|-------------:|---------------------:|-------------------------|
| FoldMapper-d256   | `foldmapper_d256.pt`               |        false |                  256 | 260213/1                |
| FoldMapper-d512   | `foldmapper_d512.pt`               |        false |                  512 | 260214/1                |
| FoldMapper-d256-B | `foldmapper_d256_blosum.pt`        |         true |                  256 | 260214/4                |
| FoldMapper-d512-B | `foldmapper_d512_blosum.pt`        |         true |                  512 | 260214/3                |

## Zenodo

DOI: *(to be assigned at paper acceptance)*

## Manual loading

```python
from foldmapper import load_model_from_checkpoint

# Architecture (use_blosum / distill_hidden_dim / hidden_dim) is inferred
# from the state_dict automatically.
model, ckpt = load_model_from_checkpoint("foldmapper_d256_blosum.pt", device="cuda")
print(model.config)
```

## Checkpoint format

```
{
  "epoch":  int,
  "step":   int,
  "num_edges":    int,          # MPNN top-k, for eval-time override
  "noise_level":  float,
  "model_state_dict":  dict,
  "optimizer_state_dict":  dict,
  "config":            dict     # FoldMapperConfig as dict (present for checkpoints trained with foldmapper/scripts/train.py)
}
```

## Foldseek database version used in the paper

Training profiles were generated against `Foldseek PDB100` downloaded on
*(fill in date)* using `foldseek databases PDB100` (Foldseek v*(fill in)*).
Recording this date is required for reproducibility because PDB100 is
periodically refreshed upstream.
