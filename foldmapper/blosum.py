"""BLOSUM62 utilities for FoldMapper.

Provides:
- BLOSUM62 substitution matrix in standard 20-AA ordering.
- ``get_blosum62_prob`` reordered to Foldseek profile alphabet
  (A C D E F G H I K L M N P Q R S T V W Y X) with softmax scaling.
"""

import torch


# Foldseek profile amino-acid ordering (same as featurize alphabet).
FOLDSEEK_AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"

# Standard BLOSUM62 ordering (20 canonical AAs).
BLOSUM62_AA_ORDER = "ARNDCQEGHILKMFPSTWYV"


# BLOSUM62 substitution matrix (symmetric, integer-valued) in BLOSUM62_AA_ORDER.
BLOSUM62_STD = [
    [ 4,-1,-2,-2, 0,-1,-1, 0,-2,-1,-1,-1,-1,-2,-1, 1, 0,-3,-2, 0],  # A
    [-1, 5, 0,-2,-3, 1, 0,-2, 0,-3,-2, 2,-1,-3,-2,-1,-1,-3,-2,-3],  # R
    [-2, 0, 6, 1,-3, 0, 0, 0, 1,-3,-3, 0,-2,-3,-2, 1, 0,-4,-2,-3],  # N
    [-2,-2, 1, 6,-3, 0, 2,-1,-1,-3,-4,-1,-3,-3,-1, 0,-1,-4,-3,-3],  # D
    [ 0,-3,-3,-3, 9,-3,-4,-3,-3,-1,-1,-3,-1,-2,-3,-1,-1,-2,-2,-1],  # C
    [-1, 1, 0, 0,-3, 5, 2,-2, 0,-3,-2, 1, 0,-3,-1, 0,-1,-2,-1,-2],  # Q
    [-1, 0, 0, 2,-4, 2, 5,-2, 0,-3,-3, 1,-2,-3,-1, 0,-1,-3,-2,-2],  # E
    [ 0,-2, 0,-1,-3,-2,-2, 6,-2,-4,-4,-2,-3,-3,-2, 0,-2,-2,-3,-3],  # G
    [-2, 0, 1,-1,-3, 0, 0,-2, 8,-3,-3,-1,-2,-1,-2,-1,-2,-2, 2,-3],  # H
    [-1,-3,-3,-3,-1,-3,-3,-4,-3, 4, 2,-3, 1, 0,-3,-2,-1,-3,-1, 3],  # I
    [-1,-2,-3,-4,-1,-2,-3,-4,-3, 2, 4,-2, 2, 0,-3,-2,-1,-2,-1, 1],  # L
    [-1, 2, 0,-1,-3, 1, 1,-2,-1,-3,-2, 5,-1,-3,-1, 0,-1,-3,-2,-2],  # K
    [-1,-1,-2,-3,-1, 0,-2,-3,-2, 1, 2,-1, 5, 0,-2,-1,-1,-1,-1, 1],  # M
    [-2,-3,-3,-3,-2,-3,-3,-3,-1, 0, 0,-3, 0, 6,-4,-2,-2, 1, 3,-1],  # F
    [-1,-2,-2,-1,-3,-1,-1,-2,-2,-3,-3,-1,-2,-4, 7,-1,-1,-4,-3,-2],  # P
    [ 1,-1, 1, 0,-1, 0, 0, 0,-1,-2,-2, 0,-1,-2,-1, 4, 1,-3,-2,-2],  # S
    [ 0,-1, 0,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,-2,-1, 1, 5,-2,-2, 0],  # T
    [-3,-3,-4,-4,-2,-2,-3,-2,-2,-3,-2,-3,-1, 1,-4,-3,-2,11, 2,-3],  # W
    [-2,-2,-2,-3,-2,-1,-2,-3, 2,-1,-1,-2,-1, 3,-3,-2,-2, 2, 7,-1],  # Y
    [ 0,-3,-3,-3,-1,-2,-2,-3,-3, 3, 1,-2, 1,-1,-2,-2, 0,-3,-1, 4],  # V
]


def _std_to_foldseek_permutation():
    """Permutation that reorders BLOSUM62_AA_ORDER rows/cols into FOLDSEEK_AA_ALPHABET (first 20 AAs)."""
    return [BLOSUM62_AA_ORDER.index(aa) for aa in FOLDSEEK_AA_ALPHABET[:20]]


def get_blosum62_prob(aa_factor: float = 1.4) -> torch.Tensor:
    """Return a [21, 21] BLOSUM62-derived probability matrix in Foldseek profile ordering.

    Softmax(aa_factor * BLOSUM62) per row, then expand with an identity row/col
    for the unknown amino-acid token 'X'. ``aa_factor=1.4`` matches Foldseek's
    aaFactor for alignment-type 2.
    """
    mat = torch.tensor(BLOSUM62_STD, dtype=torch.float32)
    perm = _std_to_foldseek_permutation()
    mat = mat[perm][:, perm]

    prob = torch.softmax(aa_factor * mat, dim=-1)  # [20, 20]

    full = torch.zeros(21, 21)
    full[:20, :20] = prob
    full[20, 20] = 1.0
    return full
