"""Structure I/O: parse PDB or mmCIF into FoldMapper's featurize-compatible dict.

Only backbone atoms N, CA, C, O are extracted. Missing heavy atoms are
rejected; rows with all four atoms present survive.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np


AA_3TO1 = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
    "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
    "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
    "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
    "UNK": "X", "MSE": "M",
}

BACKBONE_ATOMS = ("N", "CA", "C", "O")


def _parse_with_biopython(path: str, model_id: Optional[int] = None):
    """Yield (chain_id, residues) pairs with residues as list of (resnum, resname, coords_dict)."""
    from Bio.PDB import PDBParser, MMCIFParser

    ext = os.path.splitext(path)[1].lower()
    if ext in (".cif", ".mmcif"):
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)
    structure = parser.get_structure("s", path)
    models = list(structure.get_models())
    model = models[0] if model_id is None else models[model_id]
    for chain in model.get_chains():
        residues: List = []
        for residue in chain.get_residues():
            hetflag = residue.id[0]
            if hetflag.strip() and hetflag != " ":
                # skip heteroatoms except MSE (selenomet) which we treat as M
                if residue.get_resname() != "MSE":
                    continue
            coords = {}
            for atom_name in BACKBONE_ATOMS:
                if atom_name in residue:
                    coords[atom_name] = tuple(float(c) for c in residue[atom_name].get_coord())
            residues.append((residue.id[1], residue.get_resname(), coords))
        yield chain.id, residues


def load_structure(
    path: str,
    chain: Optional[str] = None,
    model_id: Optional[int] = None,
) -> Dict:
    """Load a backbone structure into a featurize-ready dict.

    Parameters
    ----------
    path : str
        PDB or mmCIF file.
    chain : Optional[str]
        If given, only this chain is returned. Otherwise the first chain is used.
    model_id : Optional[int]
        Model index (default 0).

    Returns
    -------
    dict
        ``{'name', 'seq', 'num_of_chains', 'masked_list', 'visible_list',
            'seq_chain_A', 'coords_chain_A': {'N_chain_A', 'CA_chain_A', 'C_chain_A', 'O_chain_A'}}``
    """
    selected = None
    for ch_id, residues in _parse_with_biopython(path, model_id=model_id):
        if chain is None or ch_id == chain:
            selected = (ch_id, residues)
            break
    if selected is None:
        raise ValueError(f"chain {chain!r} not found in {path}")

    ch_id, residues = selected
    seq = []
    N_list, CA_list, C_list, O_list = [], [], [], []
    for _, resname, coords in residues:
        if not all(atom in coords for atom in BACKBONE_ATOMS):
            continue
        aa = AA_3TO1.get(resname, "X")
        seq.append(aa)
        N_list.append(coords["N"])
        CA_list.append(coords["CA"])
        C_list.append(coords["C"])
        O_list.append(coords["O"])

    if not seq:
        raise ValueError(f"no residues with complete backbone in {path} (chain {ch_id})")

    name = os.path.splitext(os.path.basename(path))[0]
    return {
        "name": name,
        "seq": "".join(seq),
        "num_of_chains": 1,
        "masked_list": ["A"],
        "visible_list": [],
        "seq_chain_A": "".join(seq),
        "coords_chain_A": {
            "N_chain_A": N_list,
            "CA_chain_A": CA_list,
            "C_chain_A": C_list,
            "O_chain_A": O_list,
        },
    }
