#!/usr/bin/env python3
"""Normalise a FoldMapper checkpoint into the current ``s1_*`` key layout.

Loads any supported checkpoint (including older ones with pre-rename Stage-1
key prefixes) through ``load_model_from_checkpoint`` and re-saves it with the
current key layout and a ``dataclasses.asdict(config)`` entry.

Usage:
    python scripts/convert_checkpoint.py IN.pt OUT.pt
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from foldmapper import load_model_from_checkpoint  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("src", help="Input .pt checkpoint (any supported layout).")
    parser.add_argument("dst", help="Output .pt checkpoint (s1_* key layout).")
    args = parser.parse_args()

    model, ckpt = load_model_from_checkpoint(args.src, device="cpu")
    out = dict(ckpt)
    out["model_state_dict"] = model.state_dict()
    out["config"] = asdict(model.config)

    os.makedirs(os.path.dirname(os.path.abspath(args.dst)) or ".", exist_ok=True)
    torch.save(out, args.dst)

    n_s1 = sum(1 for k in out["model_state_dict"] if k.startswith("s1_"))
    print(
        f"wrote {args.dst}: use_blosum={model.config.use_blosum} "
        f"s1_hidden_dim={model.config.s1_hidden_dim} "
        f"hidden_dim={model.config.hidden_dim} s1_keys={n_s1}"
    )


if __name__ == "__main__":
    main()
