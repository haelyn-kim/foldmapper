# Checkpoints

Checkpoints are distributed via Zenodo once the paper is accepted. Each
variant has one `epoch_last.pt` file (final epoch) plus the corresponding
training `log.txt` for reproducibility.

| Variant           | File                            | `use_blosum` | `s1_hidden_dim` (`d_s1`) |
|-------------------|---------------------------------|-------------:|-------------------------:|
| FoldMapper-d256   | `foldmapper_d256.pt`            |        false |                      256 |
| FoldMapper-d512   | `foldmapper_d512.pt`            |        false |                      512 |
| FoldMapper-d256-B | `foldmapper_d256_blosum.pt`     |         true |                      256 |
| FoldMapper-d512-B | `foldmapper_d512_blosum.pt`     |         true |                      512 |

## Zenodo

DOI: *(to be assigned at paper acceptance)*

## Manual loading

```python
from foldmapper import load_model_from_checkpoint

# Architecture (use_blosum / s1_hidden_dim / hidden_dim) is inferred from the
# state_dict automatically; older Stage-1 key prefixes are normalised on load.
model, ckpt = load_model_from_checkpoint("foldmapper_d256_blosum.pt", device="cuda")
print(model.config)
```

## Checkpoint format

```
{
  "epoch":              int,
  "step":               int,
  "num_edges":          int,    # Stage 2 top-k, for eval-time override
  "noise_level":        float,
  "model_state_dict":   dict,
  "optimizer_state_dict": dict,
  "config":             dict    # FoldMapperConfig as dict
}
```

Weight keys follow the new paper-aligned naming:

- Stage 1: `s1_features.*`, `s1_W_p.*`, `s1_W_t.*` (BLOSUM variants only),
  `s1_W_e.*`, `s1_encoder_layers.*`, `projection.*`, `blosum62_prob` (buffer).
- Stage 2: `features.*`, `W_e.*`, `W_s.*`, `encoder_layers.*`,
  `decoder_layers.*`, `W_out.*`.

## Foldseek database version

Training profiles were generated against `Foldseek PDB100` using
`foldseek databases PDB100`. The exact download date and Foldseek version
are recorded here for reproducibility (fill in before release):

- Foldseek version: *(fill in)*
- PDB100 snapshot date: *(fill in)*
