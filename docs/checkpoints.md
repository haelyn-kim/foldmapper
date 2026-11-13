# Checkpoints

Checkpoints are distributed via Zenodo as four independent files, so you can
download only the variant you need. They are inference-ready weights: optimizer
state and training bookkeeping have been stripped.

| Paper label             | File                        | `use_blosum` | `s1_hidden_dim` (`d_s1`) |
|-------------------------|-----------------------------|-------------:|-------------------------:|
| FoldMapper-29.5M        | `foldmapper_d256.pt`        |        false |                      256 |
| FoldMapper-40.0M        | `foldmapper_d512.pt`        |        false |                      512 |
| FoldMapper-29.5M+BLOSUM | `foldmapper_d256_blosum.pt` |         true |                      256 |
| FoldMapper-40.0M+BLOSUM | `foldmapper_d512_blosum.pt` |         true |                      512 |

## Zenodo

DOI: 10.5281/zenodo.24186465

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
  "model_state_dict": dict,
  "num_edges":        int,    # Stage 2 top-k, for eval-time override
  "noise_level":      float,
  "config":           dict    # FoldMapperConfig as dict; absent in
}                             # foldmapper_d512.pt, where the architecture
                              # is inferred from the weights instead
```

All four released variants share `hidden_dim` 512, 3 encoder + 3 decoder
layers, `k_neighbors` 48, `num_edges` 48, `noise_level` 0.2, `dropout` 0.0.
Parameter counts: 29.5 M at `d_s1` 256, 40.0 M at 512.

The model parameter names are:

- Stage 1: `s1_features.*`, `s1_W_p.*`, `s1_W_t.*` (BLOSUM variants only),
  `s1_W_e.*`, `s1_encoder_layers.*`, `projection.*`, `blosum62_prob` (buffer).
- Stage 2: `features.*`, `W_e.*`, `W_s.*`, `encoder_layers.*`,
  `decoder_layers.*`, `W_out.*`.

## Foldseek database version

Training profiles were generated against `Foldseek PDB100` using
`foldseek databases PDB100`, with
`foldseek easy-search --alignment-type 0 --max-seqs 100 -e 10`. The exact
versions are recorded here for reproducibility:

- Foldseek version: 10.941cd33 (v10)
- PDB100 snapshot date: 2025-01-01 (PDB_DATE 250101)
- pdb100.tar.gz md5: aefd75e0a5d6acbe8a2b7791b53eb479
- DB build (foldseek databases PDB100) commit: 1815f0d76d7b5807e63b13f9d446dcef43c1f3b1
