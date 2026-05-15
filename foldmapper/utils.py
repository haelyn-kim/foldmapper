"""Training data loading utilities for FoldMapper.

See ``NOTICE`` for third-party attributions. FoldMapper extends the data
loader to consume the PDB + TED shard layout and a per-residue
``evo_profile`` tensor.

Data sources handled by ``loader_pdb``:

1. PDB shards: one ``.pt`` file per chain containing ``seq`` (string),
   ``xyz`` ([L, 4, 3] backbone atoms N/Calpha/C/O), and optionally
   ``evo_profile`` ([L, 21] in Foldseek amino acid ordering).
2. TED shards: one ``.pt`` per domain (same layout). The parent AlphaFold
   model is encoded in the filename
   (``AF-<UniProt>-F1-model_v<N>_TED<XX>``); domains belonging to the same
   parent are discovered through ``ted_metadata``: ``{base_id: [file_path,
   ...]}``.

The loader performs a per-sample assembly so that multi-domain TED proteins
or multi-chain PDB entries expose the requested chain as ``masked`` while
the rest of the assembly acts as visible context.
"""

from __future__ import annotations

import csv
import glob
import json
import os
import random
import re
import sys
import time

import numpy as np
import torch
from dateutil import parser


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------
def _resolve_data_path(p):
    """Resolve a metadata path. If relative, prepend ``$FOLDMAPPER_DATA_BASE``.

    Metadata distributed via the Zenodo record stores paths relative to the
    materialized training subset; users set ``FOLDMAPPER_DATA_BASE`` to point
    at their local copy. Absolute paths (legacy / local builds) pass through
    unchanged.
    """
    if not p:
        return p
    if os.path.isabs(p):
        return p
    base = os.environ.get("FOLDMAPPER_DATA_BASE", "")
    return os.path.join(base, p) if base else p


# ---------------------------------------------------------------------------
# TED metadata cache
# ---------------------------------------------------------------------------
_TED_METADATA_CACHE = None


def load_ted_metadata(metadata_path):
    """Load TED metadata JSON (``{base_id: [file_path, ...]}``) and cache it."""
    global _TED_METADATA_CACHE
    if _TED_METADATA_CACHE is not None:
        return _TED_METADATA_CACHE

    if metadata_path and os.path.isfile(metadata_path):
        print(f"Loading TED metadata from {metadata_path}...", flush=True)
        t0 = time.time()
        with open(metadata_path, "r") as f:
            _TED_METADATA_CACHE = json.load(f)
        print(
            f"TED metadata loaded: {len(_TED_METADATA_CACHE)} proteins in {time.time() - t0:.1f}s",
            flush=True,
        )
    else:
        print(f"TED metadata not found at {metadata_path}, falling back to glob lookups", flush=True)
        _TED_METADATA_CACHE = {}
    return _TED_METADATA_CACHE


# ---------------------------------------------------------------------------
# Structure dataset / loader
# ---------------------------------------------------------------------------
class StructureDataset:
    """Filters pre-loaded protein entries by sequence length and alphabet."""

    def __init__(
        self,
        pdb_dict_list,
        verbose: bool = True,
        truncate=None,
        max_length: int = 100,
        alphabet: str = "ACDEFGHIKLMNPQRSTVWYX",
    ):
        alphabet_set = set(alphabet)
        discard_count = {"bad_chars": 0, "too_long": 0, "bad_seq_length": 0}

        self.data = []
        print(
            f"StructureDataset: received {len(pdb_dict_list)} entries, max_length={max_length}",
            file=sys.stderr,
        )

        for i, entry in enumerate(pdb_dict_list):
            if (
                "seq" not in entry
                or not entry["seq"]
                or (isinstance(entry["seq"], str) and len(entry["seq"]) < 10)
            ):
                discard_count["bad_seq_length"] += 1
                if i < 10:
                    print(
                        f"StructureDataset: discarding entry {i} - short/empty seq="
                        f"{entry.get('seq', 'N/A')[:50]}",
                        file=sys.stderr,
                    )
                continue

            seq = entry["seq"]
            name = entry["name"]
            bad_chars = set(seq).difference(alphabet_set)

            if not bad_chars:
                if len(seq) <= max_length:
                    self.data.append(entry)
                else:
                    discard_count["too_long"] += 1
                    if discard_count["too_long"] <= 10:
                        print(
                            f"StructureDataset: discarding {name} - too_long, len={len(seq)}",
                            file=sys.stderr,
                        )
            else:
                discard_count["bad_chars"] += 1
                if discard_count["bad_chars"] <= 10:
                    print(
                        f"StructureDataset: discarding {name} - bad_chars={bad_chars}",
                        file=sys.stderr,
                    )

        print(
            f"StructureDataset: final size={len(self.data)}, discarded={discard_count}",
            file=sys.stderr,
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class StructureLoader:
    """Length-bucketed batcher: each batch packs up to ``batch_size`` tokens."""

    def __init__(self, dataset, batch_size: int = 100, shuffle: bool = True,
                 collate_fn=lambda x: x, drop_last: bool = False):
        self.dataset = dataset
        self.size = len(dataset)
        self.lengths = [len(dataset[i]["seq"]) for i in range(self.size)]
        self.batch_size = batch_size
        sorted_ix = np.argsort(self.lengths)

        clusters, batch = [], []
        for ix in sorted_ix:
            size = self.lengths[ix]
            if size * (len(batch) + 1) <= self.batch_size:
                batch.append(ix)
            else:
                if batch:
                    clusters.append(batch)
                batch = [ix]
        if batch:
            clusters.append(batch)
        self.clusters = clusters

    def __len__(self):
        return len(self.clusters)

    def __iter__(self):
        np.random.shuffle(self.clusters)
        for b_idx in self.clusters:
            yield [self.dataset[i] for i in b_idx]


def worker_init_fn(worker_id):
    np.random.seed()


# ---------------------------------------------------------------------------
# NoamOpt is defined in model_utils; re-exported here for scripts that
# import it directly from foldmapper.utils (e.g. training scripts written
# against the earlier module layout).
# ---------------------------------------------------------------------------
from .model_utils import NoamOpt, get_std_opt  # noqa: E402, F401


# ---------------------------------------------------------------------------
# Parallel loader
# ---------------------------------------------------------------------------
def get_pdbs(data_loader, repeat: int = 1, max_length: int = 10000, num_units: int = 1000000):
    """Load protein entries in parallel through a ThreadPoolExecutor."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 4))

    init_alphabet = [
        "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
        "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
        "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
        "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
    ]
    extra_alphabet = [str(item) for item in list(np.arange(300))]
    chain_alphabet = init_alphabet + extra_alphabet

    pdb_dict_list = []
    dataset = data_loader.dataset
    indices = list(range(len(dataset)))
    np.random.shuffle(indices)

    t0 = time.time()
    print(
        f"[loader] parallel loading with {num_workers} workers, target up to {num_units} samples",
        file=sys.stderr, flush=True,
    )

    def load_single(idx):
        try:
            return dataset[idx]
        except Exception:
            return None

    batch_size = num_workers * 10
    idx_ptr = 0
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        while len(pdb_dict_list) < num_units and idx_ptr < len(indices):
            batch_indices = indices[idx_ptr:idx_ptr + batch_size]
            idx_ptr += batch_size
            futures = {executor.submit(load_single, i): i for i in batch_indices}

            for fut in as_completed(futures):
                if len(pdb_dict_list) >= num_units:
                    break
                try:
                    t = fut.result()
                except Exception:
                    continue
                if t is None or "label" not in t:
                    continue

                # Pre-assembled entries coming straight from loader_pdb.
                if "coords_chain_A" in t and "seq_chain_A" in t:
                    my_dict = {
                        "name": t["name"],
                        "seq": t["seq"],
                        "masked_list": list(t["masked_list"]),
                        "visible_list": list(t["visible_list"]),
                        "num_of_chains": t["num_of_chains"],
                        "evo_profile": t.get("evo_profile", None),
                    }
                    all_chains = list(t["masked_list"]) + list(t["visible_list"])
                    missing_chains = []
                    for letter in all_chains:
                        seq_key = f"seq_chain_{letter}"
                        coords_key = f"coords_chain_{letter}"
                        if seq_key in t and coords_key in t:
                            my_dict[seq_key] = t[seq_key]
                            my_dict[coords_key] = t[coords_key]
                        else:
                            missing_chains.append(letter)
                    if missing_chains:
                        my_dict["masked_list"] = [c for c in my_dict["masked_list"] if c not in missing_chains]
                        my_dict["visible_list"] = [c for c in my_dict["visible_list"] if c not in missing_chains]
                        my_dict["num_of_chains"] = len(my_dict["masked_list"]) + len(my_dict["visible_list"])
                        if my_dict["num_of_chains"] == 0:
                            continue

                    if len(my_dict["seq"]) <= max_length:
                        pdb_dict_list.append(my_dict)
                        if len(pdb_dict_list) % 500 == 0:
                            print(
                                f"[loader] {len(pdb_dict_list)}/{num_units} loaded ({time.time() - t0:.1f}s)",
                                file=sys.stderr, flush=True,
                            )
                    continue

                # Flat-array format (older .pt shards with idx/xyz/masked keys): rebuild per-chain dict.
                my_dict = {}
                concat_seq = ""
                mask_list = []
                visible_list = []

                if len(list(np.unique(t["idx"]))) < 352:
                    seq_list_np = np.array(list(t["seq"]))
                    xyz_np = t["xyz"].cpu().numpy() if hasattr(t["xyz"], "cpu") else np.array(t["xyz"])
                    if xyz_np.ndim == 4 and xyz_np.shape[0] == 1:
                        xyz_np = xyz_np[0]

                    for chain_idx in list(np.unique(t["idx"])):
                        letter = chain_alphabet[chain_idx]
                        res = np.argwhere(t["idx"] == chain_idx)
                        res_flat = res.flatten()
                        initial_sequence = "".join(list(seq_list_np[res_flat]))

                        # Strip leading/trailing poly-His tags when present.
                        if initial_sequence[-6:] == "HHHHHH":
                            res = res[:, :-6]
                        if initial_sequence[0:6] == "HHHHHH":
                            res = res[:, 6:]
                        if initial_sequence[-7:-1] == "HHHHHH":
                            res = res[:, :-7]
                        if initial_sequence[-8:-2] == "HHHHHH":
                            res = res[:, :-8]
                        if initial_sequence[-9:-3] == "HHHHHH":
                            res = res[:, :-9]
                        if initial_sequence[-10:-4] == "HHHHHH":
                            res = res[:, :-10]
                        if initial_sequence[1:7] == "HHHHHH":
                            res = res[:, 7:]
                        if initial_sequence[2:8] == "HHHHHH":
                            res = res[:, 8:]
                        if initial_sequence[3:9] == "HHHHHH":
                            res = res[:, 9:]
                        if initial_sequence[4:10] == "HHHHHH":
                            res = res[:, 10:]

                        if res.shape[1] < 4:
                            continue

                        res_flat = res.flatten()
                        all_atoms = xyz_np[res_flat]
                        # Guard against collapse to 2D for single-residue chains.
                        if all_atoms.ndim == 2:
                            all_atoms = np.expand_dims(all_atoms, axis=0)

                        my_dict[f"seq_chain_{letter}"] = "".join(list(seq_list_np[res_flat]))
                        concat_seq += my_dict[f"seq_chain_{letter}"]
                        if chain_idx in t["masked"]:
                            mask_list.append(letter)
                        else:
                            visible_list.append(letter)

                        coords_dict_chain = {
                            f"N_chain_{letter}": all_atoms[:, 0, :].tolist(),
                            f"CA_chain_{letter}": all_atoms[:, 1, :].tolist(),
                            f"C_chain_{letter}": all_atoms[:, 2, :].tolist(),
                            f"O_chain_{letter}": all_atoms[:, 3, :].tolist(),
                        }
                        my_dict[f"coords_chain_{letter}"] = coords_dict_chain

                    my_dict["name"] = t["label"]
                    my_dict["masked_list"] = mask_list
                    my_dict["visible_list"] = visible_list
                    my_dict["num_of_chains"] = len(mask_list) + len(visible_list)
                    my_dict["seq"] = concat_seq

                    valid = all(
                        f"seq_chain_{ch}" in my_dict and f"coords_chain_{ch}" in my_dict
                        for ch in (mask_list + visible_list)
                    )
                    if not valid or my_dict["num_of_chains"] == 0:
                        continue
                    if len(concat_seq) <= max_length:
                        pdb_dict_list.append(my_dict)
                        if len(pdb_dict_list) % 500 == 0:
                            print(
                                f"[loader] {len(pdb_dict_list)}/{num_units} loaded ({time.time() - t0:.1f}s)",
                                file=sys.stderr, flush=True,
                            )

    print(f"[loader] finished: {len(pdb_dict_list)} entries", file=sys.stderr)
    return pdb_dict_list


# ---------------------------------------------------------------------------
# Per-cluster dataset
# ---------------------------------------------------------------------------
class PDB_dataset(torch.utils.data.Dataset):
    """Picks a random chain from each cluster key via ``loader``."""

    def __init__(self, IDs, loader, train_dict, params):
        self.IDs = IDs
        self.train_dict = train_dict
        self.loader = loader
        self.params = params

    def __len__(self):
        return len(self.IDs)

    def __getitem__(self, index):
        ID = self.IDs[index]
        for attempt in range(5):
            sel_idx = np.random.randint(0, len(self.train_dict[ID]))
            out = self.loader(self.train_dict[ID][sel_idx], self.params)
            if out is not None:
                return out
        return None


# ---------------------------------------------------------------------------
# TED path helpers
# ---------------------------------------------------------------------------
def get_ted_file_path(chain_id: str, base_dir: str):
    """Resolve a TED chain id to the on-disk ``.pt`` path.

    Layout: ``<base_dir>/<uniprot[:3]>/<uniprot[3:6]>/<chain_id>.pt``.
    ``chain_id`` format: ``AF-<UniProt>-F<N>-model_v<M>_TED<XX>``.
    """
    match = re.match(r"AF-([A-Z0-9]+)-F\d+-model_v\d+_TED\d+", chain_id)
    if not match:
        return None
    uniprot_id = match.group(1)

    if len(uniprot_id) >= 6:
        subdir1 = uniprot_id[:3]
        subdir2 = uniprot_id[3:6]
    else:
        subdir1 = uniprot_id[:3] if len(uniprot_id) >= 3 else uniprot_id
        subdir2 = uniprot_id[3:6] if len(uniprot_id) >= 6 else ""

    return os.path.join(base_dir, subdir1, subdir2, chain_id + ".pt")


def find_related_ted_domains(chain_id: str, base_dir: str, ted_metadata=None):
    """Find all TED domain files sharing the same parent AlphaFold model.

    Uses ``ted_metadata`` for O(1) lookups; falls back to a glob search when
    the metadata is unavailable.
    """
    match = re.match(r"(AF-[A-Z0-9]+-F\d+-model_v\d+)_TED\d+", chain_id)
    if not match:
        return []
    base_id = match.group(1)

    if ted_metadata and base_id in ted_metadata:
        return [_resolve_data_path(p) for p in ted_metadata[base_id]]

    uniprot_match = re.match(r"AF-([A-Z0-9]+)-F\d+-model_v\d+", base_id)
    if not uniprot_match:
        return []
    uniprot_id = uniprot_match.group(1)

    if len(uniprot_id) >= 6:
        subdir1 = uniprot_id[:3]
        subdir2 = uniprot_id[3:6]
    else:
        subdir1 = uniprot_id[:3] if len(uniprot_id) >= 3 else uniprot_id
        subdir2 = uniprot_id[3:6] if len(uniprot_id) >= 6 else ""

    pattern = os.path.join(base_dir, subdir1, subdir2, f"{base_id}_TED*.pt")
    return sorted(glob.glob(pattern))


# ---------------------------------------------------------------------------
# Per-chain loader (TED and PDB paths)
# ---------------------------------------------------------------------------
def loader_pdb(item, params):
    """Load a single protein entry (TED domain, generic new-format, or PDB chain)."""
    init_alphabet = [
        "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
        "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
    ]
    extra_alphabet = [str(i) for i in list(np.arange(300))]
    chain_alphabet = init_alphabet + extra_alphabet

    chain_id = item[0]
    is_ted_format = chain_id.startswith("AF-")
    is_pdb_format = "_" in chain_id and not is_ted_format and not chain_id.startswith("TEMP_")

    if is_ted_format:
        pt_file = get_ted_file_path(chain_id, params["DIR_TED"])
        if pt_file is None or not os.path.isfile(pt_file):
            return None

        try:
            ted_metadata = None
            if "TED_METADATA_DATA" in params and params["TED_METADATA_DATA"]:
                ted_metadata = params["TED_METADATA_DATA"]
            elif "TED_METADATA" in params:
                ted_metadata = load_ted_metadata(params["TED_METADATA"])

            related_files = find_related_ted_domains(chain_id, params["DIR_TED"], ted_metadata)
            if not related_files:
                related_files = [pt_file]

            all_seqs = []
            all_xyz = []
            all_idx = []
            chain_data = {}

            for chain_idx, ted_file in enumerate(related_files):
                data = torch.load(ted_file, map_location="cpu")
                if "xyz" in data and isinstance(data["xyz"], torch.Tensor):
                    data["xyz"] = data["xyz"].clone()
                if "seq" not in data:
                    continue
                seq = data["seq"]
                xyz_np = data["xyz"].cpu().numpy() if isinstance(data["xyz"], torch.Tensor) else np.array(data["xyz"])
                L = len(seq)

                all_seqs.append(seq)
                all_xyz.append(xyz_np)
                all_idx.append(np.full(L, chain_idx, dtype=np.int32))

                letter = chain_alphabet[chain_idx]
                chain_data[f"seq_chain_{letter}"] = seq
                chain_data[f"coords_chain_{letter}"] = {
                    f"N_chain_{letter}": xyz_np[:, 0, :].tolist(),
                    f"CA_chain_{letter}": xyz_np[:, 1, :].tolist(),
                    f"C_chain_{letter}": xyz_np[:, 2, :].tolist(),
                    f"O_chain_{letter}": xyz_np[:, 3, :].tolist(),
                }
                del data

            if not all_seqs:
                return None

            concat_seq = "".join(all_seqs)
            concat_xyz = np.concatenate(all_xyz, axis=0)
            concat_idx = np.concatenate(all_idx, axis=0)
            num_chains = len(all_seqs)

            # Locate the requested chain within the assembled set.
            requested_chain_idx = 0
            for idx, ted_file in enumerate(related_files):
                file_chain_id = os.path.basename(ted_file).replace(".pt", "")
                if file_chain_id == chain_id:
                    requested_chain_idx = idx
                    break

            masked_chain_letter = chain_alphabet[requested_chain_idx]
            visible_list = [chain_alphabet[i] for i in range(num_chains) if i != requested_chain_idx]

            evo_profile = None
            try:
                requested_data = torch.load(related_files[requested_chain_idx], map_location="cpu")
                if "evo_profile" in requested_data:
                    evo_profile = requested_data["evo_profile"]
                del requested_data
            except Exception:
                pass

            result = {
                "seq": concat_seq,
                "xyz": concat_xyz.copy(),
                "idx": concat_idx,
                "masked": np.array([requested_chain_idx], dtype=np.int32),
                "label": chain_id,
                "masked_list": [masked_chain_letter],
                "visible_list": visible_list,
                "num_of_chains": num_chains,
                "name": chain_id,
                "evo_profile": evo_profile,
            }
            result.update(chain_data)
            return result

        except Exception:
            import traceback
            traceback.print_exc(file=sys.stderr)
            return None

    elif not is_pdb_format:
        # Generic new-format entry (single file, not PDB_chain, not TED).
        pt_file = os.path.join(params["DIR_TED"], chain_id) + ".pt"
        if os.path.isfile(pt_file):
            try:
                data = torch.load(pt_file, map_location="cpu")
                if "xyz" in data and isinstance(data["xyz"], torch.Tensor):
                    data["xyz"] = data["xyz"].clone()
                if "seq" not in data:
                    return None

                seq = data["seq"]
                xyz_np = data["xyz"].cpu().numpy() if isinstance(data["xyz"], torch.Tensor) else np.array(data["xyz"])
                L = len(seq)
                evo_profile = data.get("evo_profile", None)

                result = {
                    "seq": seq,
                    "xyz": xyz_np.copy(),
                    "idx": np.zeros(L, dtype=np.int32),
                    "masked": np.array([0], dtype=np.int32),
                    "label": chain_id,
                    "seq_chain_A": seq,
                    "coords_chain_A": {
                        "N_chain_A": xyz_np[:, 0, :].tolist(),
                        "CA_chain_A": xyz_np[:, 1, :].tolist(),
                        "C_chain_A": xyz_np[:, 2, :].tolist(),
                        "O_chain_A": xyz_np[:, 3, :].tolist(),
                    },
                    "masked_list": ["A"],
                    "visible_list": [],
                    "num_of_chains": 1,
                    "name": chain_id,
                    "evo_profile": evo_profile,
                }
                del data
                del xyz_np
                return result
            except Exception:
                import traceback
                traceback.print_exc(file=sys.stderr)
                return None
        return None

    # ---------------------------------------------------------------------
    # PDB path: metadata-driven (pdb_metadata.json).
    # ---------------------------------------------------------------------
    pdb_metadata = params.get("PDB_METADATA", {})
    if chain_id not in pdb_metadata:
        return None

    entry = pdb_metadata[chain_id]
    chain_path = _resolve_data_path(entry["chain_path"])
    other_chains = entry["other_chains"]
    pdb_id = entry["pdb_id"]

    try:
        target_data = torch.load(chain_path, map_location="cpu")
        target_seq = target_data["seq"]
        target_xyz = target_data["xyz"].cpu().numpy() if hasattr(target_data["xyz"], "cpu") else np.array(target_data["xyz"])

        if not other_chains:
            L = len(target_seq)
            evo_profile = target_data.get("evo_profile", None)
            return {
                "seq": target_seq,
                "xyz": target_xyz.copy(),
                "idx": np.zeros(L, dtype=np.int32),
                "masked": np.array([0], dtype=np.int32),
                "label": chain_id,
                "seq_chain_A": target_seq,
                "coords_chain_A": {
                    "N_chain_A": target_xyz[:, 0, :].tolist(),
                    "CA_chain_A": target_xyz[:, 1, :].tolist(),
                    "C_chain_A": target_xyz[:, 2, :].tolist(),
                    "O_chain_A": target_xyz[:, 3, :].tolist(),
                },
                "masked_list": ["A"],
                "visible_list": [],
                "num_of_chains": 1,
                "name": chain_id,
                "evo_profile": evo_profile,
            }

        all_seqs = [target_seq]
        all_xyz = [target_xyz]
        all_idx = [np.zeros(len(target_seq), dtype=np.int32)]
        chain_data = {
            "seq_chain_A": target_seq,
            "coords_chain_A": {
                "N_chain_A": target_xyz[:, 0, :].tolist(),
                "CA_chain_A": target_xyz[:, 1, :].tolist(),
                "C_chain_A": target_xyz[:, 2, :].tolist(),
                "O_chain_A": target_xyz[:, 3, :].tolist(),
            },
        }

        # Cap context chains to avoid unbounded memory on large assemblies.
        max_context_chains = 5
        if len(other_chains) > max_context_chains:
            other_chains_selected = random.sample(other_chains, max_context_chains)
        else:
            other_chains_selected = other_chains

        visible_letters = []
        loaded_count = 0
        for other_ch in other_chains_selected:
            other_entry_id = f"{pdb_id}_{other_ch}"
            if other_entry_id in pdb_metadata:
                other_path = _resolve_data_path(pdb_metadata[other_entry_id]["chain_path"])
                if os.path.isfile(other_path):
                    other_data = torch.load(other_path, map_location="cpu")
                    other_seq = other_data["seq"]
                    other_xyz = other_data["xyz"].cpu().numpy() if hasattr(other_data["xyz"], "cpu") else np.array(other_data["xyz"])

                    chain_idx = loaded_count + 1
                    letter = chain_alphabet[chain_idx]
                    loaded_count += 1

                    all_seqs.append(other_seq)
                    all_xyz.append(other_xyz)
                    all_idx.append(np.full(len(other_seq), chain_idx, dtype=np.int32))
                    visible_letters.append(letter)

                    chain_data[f"seq_chain_{letter}"] = other_seq
                    chain_data[f"coords_chain_{letter}"] = {
                        f"N_chain_{letter}": other_xyz[:, 0, :].tolist(),
                        f"CA_chain_{letter}": other_xyz[:, 1, :].tolist(),
                        f"C_chain_{letter}": other_xyz[:, 2, :].tolist(),
                        f"O_chain_{letter}": other_xyz[:, 3, :].tolist(),
                    }
                    del other_data

        concat_seq = "".join(all_seqs)
        concat_xyz = np.concatenate(all_xyz, axis=0)
        concat_idx = np.concatenate(all_idx, axis=0)
        num_chains = 1 + len(visible_letters)
        evo_profile = target_data.get("evo_profile", None)

        result = {
            "seq": concat_seq,
            "xyz": concat_xyz.copy(),
            "idx": concat_idx,
            "masked": np.array([0], dtype=np.int32),
            "label": chain_id,
            "masked_list": ["A"],
            "visible_list": visible_letters,
            "num_of_chains": num_chains,
            "name": chain_id,
            "evo_profile": evo_profile,
        }
        result.update(chain_data)
        del target_data
        return result

    except Exception:
        import traceback
        traceback.print_exc(file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Training cluster builder
# ---------------------------------------------------------------------------
def build_training_clusters(params, debug: bool = False):
    """Read cluster CSV and split into train / valid / test by cluster id."""
    with open(params["VAL"]) as f:
        val_ids = {int(l) for l in f.readlines()}
    with open(params["TEST"]) as f:
        test_ids = {int(l) for l in f.readlines()}

    if debug:
        val_ids = set()
        test_ids = set()

    with open(params["LIST"], "r") as f:
        reader = csv.reader(f)
        next(reader)
        rows = [
            [r[0], r[3], int(r[5])]
            for r in reader
            if float(r[2]) <= params["RESCUT"]
            and parser.parse(r[1]) <= parser.parse(params["DATCUT"])
        ]

    train, valid, test = {}, {}, {}
    if debug:
        rows = rows[:20]

    for r in rows:
        bucket = valid if r[2] in val_ids else test if r[2] in test_ids else train
        bucket.setdefault(r[2], []).append(r[:2])

    if debug:
        valid = train

    return train, valid, test
