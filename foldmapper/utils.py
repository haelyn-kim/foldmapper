import torch
from torch.utils.data import DataLoader
import csv
from dateutil import parser
import numpy as np
import time
import random
import os
import json
import re

# 🌟 TED 메타데이터 전역 캐시
_TED_METADATA_CACHE = None

def load_ted_metadata(metadata_path):
    """
    TED 메타데이터를 로드하고 캐시합니다.
    메타데이터 구조: {base_id: [file_path1, file_path2, ...]}
    """
    global _TED_METADATA_CACHE
    if _TED_METADATA_CACHE is not None:
        return _TED_METADATA_CACHE

    if metadata_path and os.path.isfile(metadata_path):
        print(f"Loading TED metadata from {metadata_path}...", flush=True)
        t0 = time.time()
        with open(metadata_path, 'r') as f:
            _TED_METADATA_CACHE = json.load(f)
        print(f"TED metadata loaded: {len(_TED_METADATA_CACHE)} proteins in {time.time()-t0:.1f}s", flush=True)
    else:
        print(f"TED metadata not found at {metadata_path}, using glob fallback", flush=True)
        _TED_METADATA_CACHE = {}

    return _TED_METADATA_CACHE

class StructureDataset():
    def __init__(self, pdb_dict_list, verbose=True, truncate=None, max_length=100, # max_length 인수는 training.py에서 10000으로 전달됨
        alphabet='ACDEFGHIKLMNPQRSTVWYX'):
        import sys
        alphabet_set = set([a for a in alphabet])
        discard_count = {
            'bad_chars': 0,
            'too_long': 0,
            'bad_seq_length': 0
        }

        self.data = []

        print(f"StructureDataset: Received {len(pdb_dict_list)} entries, max_length={max_length}", file=sys.stderr)

        start = time.time()
        for i, entry in enumerate(pdb_dict_list):

            # 🌟🌟🌟 수정 1: loader_pdb 실패 항목 (빈 서열) 즉시 폐기 🌟🌟🌟
            # PDB_dataset에서 넘어온 None 또는 빈 문자열을 확실히 거릅니다.
            if 'seq' not in entry or not entry['seq'] or \
            (isinstance(entry['seq'], str) and len(entry['seq']) < 10):
                discard_count['bad_seq_length'] += 1
                if i < 10:  # 처음 10개만 출력
                    print(f"StructureDataset: Discarding entry {i} - bad_seq_length, seq={entry.get('seq', 'N/A')[:50]}", file=sys.stderr)
                continue

            seq = entry['seq']
            name = entry['name']

            bad_chars = set([s for s in seq]).difference(alphabet_set)

            # 🌟🌟🌟 수정 2: max_length 필터링 강화 (training.py의 10000을 사용) 🌟🌟🌟
            if len(bad_chars) == 0:
                if len(entry['seq']) <= max_length: # 이 max_length는 args.max_protein_length (10000)입니다.
                    self.data.append(entry)
                else:
                    discard_count['too_long'] += 1
                    if discard_count['too_long'] <= 10:  # 처음 10개만 출력
                        print(f"StructureDataset: Discarding entry {name} - too_long, len={len(entry['seq'])}", file=sys.stderr)
            else:
                discard_count['bad_chars'] += 1
                if discard_count['bad_chars'] <= 10:  # 처음 10개만 출력
                    print(f"StructureDataset: Discarding entry {name} - bad_chars={bad_chars}", file=sys.stderr)

        print(f"StructureDataset: Final dataset size={len(self.data)}, Discarded: {discard_count}", file=sys.stderr)
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class StructureLoader():
    def __init__(self, dataset, batch_size=100, shuffle=True,
        collate_fn=lambda x:x, drop_last=False):
        self.dataset = dataset
        self.size = len(dataset)
        self.lengths = [len(dataset[i]['seq']) for i in range(self.size)]
        self.batch_size = batch_size
        sorted_ix = np.argsort(self.lengths)

        # Cluster into batches of similar sizes
        clusters, batch = [], []
        batch_max = 0
        for ix in sorted_ix:
            size = self.lengths[ix]
            if size * (len(batch) + 1) <= self.batch_size:
                batch.append(ix)
                batch_max = size
            else:
                # 현재 배치가 비어있지 않으면 추가
                if len(batch) > 0:
                    clusters.append(batch)
                # 새 배치 시작 (현재 아이템 포함)
                batch = [ix]
                batch_max = size
        if len(batch) > 0:
            clusters.append(batch)
        self.clusters = clusters

    def __len__(self):
        return len(self.clusters)

    def __iter__(self):
        np.random.shuffle(self.clusters)
        for b_idx in self.clusters:
            batch = [self.dataset[i] for i in b_idx]
            yield batch


def worker_init_fn(worker_id):
    np.random.seed()

class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer, step):
        self.optimizer = optimizer
        self._step = step
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    @property
    def param_groups(self):
        """Return param_groups."""
        return self.optimizer.param_groups

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def zero_grad(self):
        self.optimizer.zero_grad()

def get_std_opt(parameters, d_model, step):
    return NoamOpt(
        d_model, 2, 4000, torch.optim.Adam(parameters, lr=0, betas=(0.9, 0.98), eps=1e-9), step
    )




def get_pdbs(data_loader, repeat=1, max_length=10000, num_units=1000000):
    import sys
    import numpy as np
    import time
    import torch
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import os

    # 🌟 병렬 로딩을 위한 worker 수 (SLURM CPU 기반)
    NUM_WORKERS = int(os.environ.get('SLURM_CPUS_PER_TASK', 4))

    init_alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G','H', 'I', 'J','K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T','U', 'V','W','X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g','h', 'i', 'j','k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't','u', 'v','w','x', 'y', 'z']
    extra_alphabet = [str(item) for item in list(np.arange(300))]
    chain_alphabet = init_alphabet + extra_alphabet
    c = 0
    c1 = 0
    pdb_dict_list = []
    t0 = time.time()

    # 🌟🌟🌟 병렬 로딩: ThreadPoolExecutor 사용 🌟🌟🌟
    dataset = data_loader.dataset
    indices = list(range(len(dataset)))
    np.random.shuffle(indices)  # 랜덤하게 섞기

    print(f"🌟 Parallel loading with {NUM_WORKERS} workers, loading up to {num_units} samples...", file=sys.stderr, flush=True)

    def load_single_item(idx):
        """단일 아이템 로드 (병렬 실행용)"""
        try:
            return dataset[idx]
        except Exception as e:
            return None

    # 병렬로 데이터 로드 (배치 단위로 제출하여 메모리 효율성 향상)
    BATCH_SIZE = NUM_WORKERS * 10  # 한 번에 제출할 작업 수
    idx_ptr = 0

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        while len(pdb_dict_list) < num_units and idx_ptr < len(indices):
            # 배치 단위로 작업 제출
            batch_indices = indices[idx_ptr:idx_ptr + BATCH_SIZE]
            idx_ptr += BATCH_SIZE
            futures = {executor.submit(load_single_item, idx): idx for idx in batch_indices}

            for future in as_completed(futures):
                if len(pdb_dict_list) >= num_units:
                    break

                try:
                    t = future.result()
                    if t is None:
                        continue
                except Exception as e:
                    continue

                c1 += 1

                # 🌟🌟🌟 loader_pdb 실패 항목 (None 반환)은 이미 PDB_dataset에서 걸러졌다고 가정합니다. 🌟🌟🌟
                if 'label' not in t:
                    continue

                # 🌟 loader_pdb에서 이미 완성된 형식으로 반환한 경우 (새로운 슬라이싱 데이터)
                if 'coords_chain_A' in t and 'seq_chain_A' in t:

                    my_dict = {
                        'name': t['name'],
                        'seq': t['seq'],
                        'masked_list': list(t['masked_list']),  # 🌟 tuple -> list 변환
                        'visible_list': list(t['visible_list']),  # 🌟 tuple -> list 변환
                        'num_of_chains': t['num_of_chains'],
                        'evo_profile': t.get('evo_profile', None),  # 🌟 evolutionary profile 추가
                    }

                    # 🌟 모든 체인의 seq_chain_* 및 coords_chain_* 복사 (다중 체인 지원)
                    all_chains = list(t['masked_list']) + list(t['visible_list'])
                    missing_chains = []
                    for chain_letter in all_chains:
                        seq_key = f'seq_chain_{chain_letter}'
                        coords_key = f'coords_chain_{chain_letter}'
                        if seq_key in t and coords_key in t:
                            my_dict[seq_key] = t[seq_key]
                            my_dict[coords_key] = t[coords_key]
                        else:
                            missing_chains.append(chain_letter)

                    # 🌟 누락된 체인이 있으면 masked_list/visible_list에서 제거
                    if missing_chains:
                        my_dict['masked_list'] = [ch for ch in my_dict['masked_list'] if ch not in missing_chains]
                        my_dict['visible_list'] = [ch for ch in my_dict['visible_list'] if ch not in missing_chains]
                        my_dict['num_of_chains'] = len(my_dict['masked_list']) + len(my_dict['visible_list'])
                        if my_dict['num_of_chains'] == 0:
                            # 유효한 체인이 없으면 스킵
                            continue

                    # max_length 필터링
                    if len(my_dict['seq']) <= max_length:
                        pdb_dict_list.append(my_dict)
                        c += 1

                        current_count = len(pdb_dict_list)
                        if current_count % 500 == 0:
                            elapsed_time = time.time() - t0
                            print(f"LOAD STATUS: {current_count}/{num_units} files loaded. Time: {elapsed_time:.1f}s", file=sys.stderr, flush=True)

                    continue

                # 기존 복잡한 체인 처리 로직 (PDB 파일용)
                my_dict = {}
                s = 0
                concat_seq = ''

                coords_dict = {}
                mask_list = []
                visible_list = []

                if len(list(np.unique(t['idx']))) < 352:

                    seq_list_np = np.array(list(t['seq']))

                    # 🌟 t['xyz']를 먼저 numpy 배열로 변환
                    xyz_np = t['xyz'].cpu().numpy() if hasattr(t['xyz'], 'cpu') else np.array(t['xyz'])
                    if xyz_np.ndim == 4 and xyz_np.shape[0] == 1:
                        xyz_np = xyz_np[0]

                    for idx in list(np.unique(t['idx'])):
                        letter = chain_alphabet[idx]
                        res = np.argwhere(t['idx']==idx)

                        res_flat = res.flatten()
                        initial_sequence= "".join(list(seq_list_np[res_flat]))

                        # HHHHHH 마스킹 로직 (유지)
                        if initial_sequence[-6:] == "HHHHHH":
                            res = res[:,:-6]
                        if initial_sequence[0:6] == "HHHHHH":
                            res = res[:,6:]
                        if initial_sequence[-7:-1] == "HHHHHH":
                            res = res[:,:-7]
                        if initial_sequence[-8:-2] == "HHHHHH":
                            res = res[:,:-8]
                        if initial_sequence[-9:-3] == "HHHHHH":
                            res = res[:,:-9]
                        if initial_sequence[-10:-4] == "HHHHHH":
                            res = res[:,:-10]
                        if initial_sequence[1:7] == "HHHHHH":
                            res = res[:,7:]
                        if initial_sequence[2:8] == "HHHHHH":
                            res = res[:,8:]
                        if initial_sequence[3:9] == "HHHHHH":
                            res = res[:,9:]
                        if initial_sequence[4:10] == "HHHHHH":
                            res = res[:,10:]

                        if res.shape[1] < 4:
                            pass
                        else:
                            res_flat = res.flatten()

                            # 🌟🌟🌟 핵심 수정: 단일 잔기 시 배열 차원 붕괴 방지 🌟🌟🌟
                            all_atoms = xyz_np[res_flat]
                            if all_atoms.ndim == 2:
                                # [14, 3] 형태로 붕괴된 경우, [1, 14, 3]으로 확장
                                all_atoms = np.expand_dims(all_atoms, axis=0)

                            my_dict['seq_chain_'+letter]= "".join(list(seq_list_np[res_flat]))
                            concat_seq += my_dict['seq_chain_'+letter]
                            if idx in t['masked']:
                                mask_list.append(letter)
                            else:
                                visible_list.append(letter)
                            coords_dict_chain = {}

                            coords_dict_chain['N_chain_'+letter]=all_atoms[:,0,:].tolist()
                            coords_dict_chain['CA_chain_'+letter]=all_atoms[:,1,:].tolist()
                            coords_dict_chain['C_chain_'+letter]=all_atoms[:,2,:].tolist()
                            coords_dict_chain['O_chain_'+letter]=all_atoms[:,3,:].tolist()
                            my_dict['coords_chain_'+letter]=coords_dict_chain

                    # my_dict 완성
                    my_dict['name']= t['label']
                    my_dict['masked_list']= mask_list
                    my_dict['visible_list']= visible_list
                    my_dict['num_of_chains'] = len(mask_list) + len(visible_list)
                    my_dict['seq'] = concat_seq

                    # 🌟 데이터 일관성 검증
                    all_chains_check = mask_list + visible_list
                    valid_entry = True
                    for chain_letter in all_chains_check:
                        if f'seq_chain_{chain_letter}' not in my_dict or f'coords_chain_{chain_letter}' not in my_dict:
                            print(f"WARNING: Skipping {t['label']} - missing data for chain {chain_letter}", file=sys.stderr)
                            valid_entry = False
                            break

                    if not valid_entry:
                        continue

                    if my_dict['num_of_chains'] == 0:
                        continue

                    if len(concat_seq) <= max_length:
                        pdb_dict_list.append(my_dict)

                        current_count = len(pdb_dict_list)
                        if current_count % 500 == 0:
                            elapsed_time = time.time() - t0
                            print(f"LOAD STATUS: {current_count}/{num_units} files loaded. Time: {elapsed_time:.1f}s", file=sys.stderr, flush=True)

    # 🌟🌟🌟 최종 반환 🌟🌟🌟
    print(f"LOAD STATUS: Finalizing load. Total {len(pdb_dict_list)} units ready.", file=sys.stderr)
    return pdb_dict_list



class PDB_dataset(torch.utils.data.Dataset):
    def __init__(self, IDs, loader, train_dict, params):
        self.IDs = IDs
        self.train_dict = train_dict
        self.loader = loader
        self.params = params

    def __len__(self):
        return len(self.IDs)

    def __getitem__(self, index):
        import sys
        ID = self.IDs[index]

        for attempt in range(5):
            sel_idx = np.random.randint(0, len(self.train_dict[ID]))
            out = self.loader(self.train_dict[ID][sel_idx], self.params)
            if out is not None:
                if index < 5 and attempt == 0:  # 처음 5개 아이템의 첫 시도만 출력
                    print(f"PDB_dataset.__getitem__: ID={ID}, seq_len={len(out.get('seq', ''))}", file=sys.stderr)
                return out
            elif index < 5:  # 처음 5개만 출력
                print(f"PDB_dataset.__getitem__: ID={ID}, attempt {attempt} returned None", file=sys.stderr)

        print(f"PDB_dataset.__getitem__: ID={ID} failed all 5 attempts, returning None", file=sys.stderr)
        return None



def get_ted_file_path(chain_id, base_dir):
    """
    TED 파일의 실제 경로를 찾는 함수

    chain_id 예시: AF-A0A0G4AJM1-F1-model_v4_TED01
    UniProt ID: A0A0G4AJM1
    Subdirectory: A0A/0G4

    Returns: 전체 파일 경로 또는 None
    """
    import os
    import re

    # AF-{UniProtID}-F1-model_v4_TED{XX} 패턴에서 UniProt ID 추출
    match = re.match(r'AF-([A-Z0-9]+)-F\d+-model_v\d+_TED\d+', chain_id)
    if not match:
        return None

    uniprot_id = match.group(1)

    # Subdirectory: 첫 3글자 / 다음 3글자
    if len(uniprot_id) >= 6:
        subdir1 = uniprot_id[:3]  # A0A
        subdir2 = uniprot_id[3:6]  # 0G4
    else:
        # 짧은 ID의 경우 fallback
        subdir1 = uniprot_id[:3] if len(uniprot_id) >= 3 else uniprot_id
        subdir2 = uniprot_id[3:6] if len(uniprot_id) >= 6 else ""

    pt_file = os.path.join(base_dir, subdir1, subdir2, chain_id + ".pt")
    return pt_file


def find_related_ted_domains(chain_id, base_dir, ted_metadata=None):
    """
    같은 UniProt ID를 가진 모든 TED 도메인 파일들을 찾는 함수

    chain_id 예시: AF-A0A0G4AJM1-F1-model_v4_TED01
    Returns: [TED01.pt 경로, TED02.pt 경로, ...] 정렬된 리스트

    ted_metadata가 제공되면 O(1) lookup, 없으면 glob fallback
    """
    import os
    import glob

    # base_id 추출: AF-A0A0G4AJM1-F1-model_v4 (_TED01 제거)
    match = re.match(r'(AF-[A-Z0-9]+-F\d+-model_v\d+)_TED\d+', chain_id)
    if not match:
        return []

    base_id = match.group(1)

    # 🌟 메타데이터가 있으면 사용 (O(1) lookup)
    if ted_metadata and base_id in ted_metadata:
        return ted_metadata[base_id]

    # 🌟 메타데이터가 없으면 glob fallback
    # UniProt ID 추출
    uniprot_match = re.match(r'AF-([A-Z0-9]+)-F\d+-model_v\d+', base_id)
    if not uniprot_match:
        return []

    uniprot_id = uniprot_match.group(1)

    # Subdirectory 결정
    if len(uniprot_id) >= 6:
        subdir1 = uniprot_id[:3]
        subdir2 = uniprot_id[3:6]
    else:
        subdir1 = uniprot_id[:3] if len(uniprot_id) >= 3 else uniprot_id
        subdir2 = uniprot_id[3:6] if len(uniprot_id) >= 6 else ""

    # glob으로 같은 base_id를 가진 모든 TED 파일 찾기
    pattern = os.path.join(base_dir, subdir1, subdir2, f"{base_id}_TED*.pt")
    related_files = sorted(glob.glob(pattern))

    return related_files


def loader_pdb(item,params):
    import os
    import torch
    import numpy as np
    import random
    import sys

    # chain_alphabet for multi-chain assembly
    init_alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G','H', 'I', 'J','K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T','U', 'V','W','X', 'Y', 'Z']
    extra_alphabet = [str(item_idx) for item_idx in list(np.arange(300))]
    chain_alphabet = init_alphabet + extra_alphabet

    # item[0]은 CHAINID/DOMAINID (예: '5naf_A' 또는 'AF-ID_TEDXX')
    chain_id = item[0]

    # ----------------------------------------------------------------------
    # 1. 새로운 슬라이싱된 DOMAINID 파일 경로 확인 (AF-ID_TEDXX 형태)

    # 새로운 데이터인지 확인: AF-로 시작하면 TED 데이터
    is_ted_format = chain_id.startswith('AF-')
    is_pdb_format = ('_' in chain_id and not is_ted_format and not chain_id.startswith('TEMP_'))

    if is_ted_format:
        # **TED 데이터 로딩 로직 (Multi-domain assembly 지원)**

        # TED 파일 경로 찾기 (subdirectory 구조 지원)
        pt_file = get_ted_file_path(chain_id, params['DIR_NEW'])
        print(f"Debug TED: Attempting TED file: {pt_file}", file=sys.stderr)

        if pt_file is None or not os.path.isfile(pt_file):
            print(f"Debug TED: File not found: {pt_file}", file=sys.stderr)
            return None

        try:
            # 🌟 TED 메타데이터: 미리 로드된 데이터 사용 (없으면 파일에서 로드)
            ted_metadata = None
            if 'TED_METADATA_DATA' in params and params['TED_METADATA_DATA']:
                ted_metadata = params['TED_METADATA_DATA']
            elif 'TED_METADATA' in params:
                ted_metadata = load_ted_metadata(params['TED_METADATA'])

            # 같은 UniProt의 모든 TED 도메인 찾기 (메타데이터 사용)
            related_files = find_related_ted_domains(chain_id, params['DIR_NEW'], ted_metadata)

            if not related_files:
                related_files = [pt_file]

            # 디버그: assembly 정보 출력
            if len(related_files) > 1:
                related_names = [os.path.basename(f).replace('.pt', '') for f in related_files]
                print(f"Debug TED: Assembly {chain_id} with {len(related_files)} domains: {related_names}", file=sys.stderr)

            # 모든 관련 도메인 로드 및 assembly
            all_seqs = []
            all_xyz = []
            all_idx = []
            chain_data = {}

            for chain_idx, ted_file in enumerate(related_files):
                data = torch.load(ted_file, map_location='cpu')

                if 'xyz' in data and isinstance(data['xyz'], torch.Tensor):
                    data['xyz'] = data['xyz'].clone()

                if 'seq' not in data:
                    continue

                seq = data['seq']
                xyz_np = data['xyz'].cpu().numpy() if isinstance(data['xyz'], torch.Tensor) else np.array(data['xyz'])
                L = len(seq)

                all_seqs.append(seq)
                all_xyz.append(xyz_np)
                all_idx.append(np.full(L, chain_idx, dtype=np.int32))

                # 체인별 데이터 저장
                letter = chain_alphabet[chain_idx]
                chain_data[f'seq_chain_{letter}'] = seq
                chain_data[f'coords_chain_{letter}'] = {
                    f'N_chain_{letter}': xyz_np[:, 0, :].tolist(),
                    f'CA_chain_{letter}': xyz_np[:, 1, :].tolist(),
                    f'C_chain_{letter}': xyz_np[:, 2, :].tolist(),
                    f'O_chain_{letter}': xyz_np[:, 3, :].tolist(),
                }

                del data

            if not all_seqs:
                return None

            # 전체 시퀀스 및 좌표 합치기
            concat_seq = ''.join(all_seqs)
            concat_xyz = np.concatenate(all_xyz, axis=0)
            concat_idx = np.concatenate(all_idx, axis=0)

            num_chains = len(all_seqs)

            # 🌟🌟🌟 버그 수정: 요청된 chain_id가 related_files에서 몇 번째인지 찾기 🌟🌟🌟
            # related_files는 정렬되어 있으므로 요청된 도메인이 첫 번째가 아닐 수 있음
            requested_chain_idx = 0  # 기본값
            for idx, ted_file in enumerate(related_files):
                # 파일명에서 chain_id 추출하여 비교
                file_chain_id = os.path.basename(ted_file).replace('.pt', '')
                if file_chain_id == chain_id:
                    requested_chain_idx = idx
                    break

            # masked_list: 요청된 도메인만 포함
            masked_chain_letter = chain_alphabet[requested_chain_idx]
            # visible_list: 요청된 도메인을 제외한 나머지
            visible_list = [chain_alphabet[i] for i in range(num_chains) if i != requested_chain_idx]

            # 디버그: multi-domain인 경우 masked 정보 출력
            if num_chains > 1:
                print(f"Debug TED: {chain_id} -> masked_idx={requested_chain_idx}, masked_list=['{masked_chain_letter}'], visible_list={visible_list}", file=sys.stderr)

            # 🌟 요청된 도메인의 evo_profile 로드
            evo_profile = None
            requested_file = related_files[requested_chain_idx]
            try:
                requested_data = torch.load(requested_file, map_location='cpu')
                if 'evo_profile' in requested_data:
                    evo_profile = requested_data['evo_profile']
                del requested_data
            except:
                pass

            result = {
                'seq': concat_seq,
                'xyz': concat_xyz.copy(),
                'idx': concat_idx,
                'masked': np.array([requested_chain_idx], dtype=np.int32),  # 🌟 요청된 도메인의 인덱스
                'label': chain_id,
                'masked_list': [masked_chain_letter],  # 🌟 요청된 도메인의 체인 문자
                'visible_list': visible_list,
                'num_of_chains': num_chains,
                'name': chain_id,
                'evo_profile': evo_profile,  # 🌟 evolutionary profile 추가
            }

            # 체인별 데이터 추가
            result.update(chain_data)

            return result

        except Exception as e:
            import traceback
            traceback.print_exc(file=sys.stderr)
            return None

    elif not is_pdb_format:
        # 기타 새로운 데이터 (TED가 아닌 경우)
        PREFIX_NEW = os.path.join(params['DIR_NEW'], chain_id)
        pt_file = PREFIX_NEW + ".pt"

        if os.path.isfile(pt_file):
            try:
                data = torch.load(pt_file, map_location='cpu')
                if 'xyz' in data and isinstance(data['xyz'], torch.Tensor):
                    data['xyz'] = data['xyz'].clone()

                if 'seq' not in data:
                    return None

                seq = data['seq']
                L = len(seq)
                xyz_np = data['xyz'].cpu().numpy() if isinstance(data['xyz'], torch.Tensor) else np.array(data['xyz'])

                # 🌟 evo_profile 로드
                evo_profile = data.get('evo_profile', None)

                result = {
                    'seq': data['seq'],
                    'xyz': xyz_np.copy(),
                    'idx': np.zeros(L, dtype=np.int32),
                    'masked': np.array([0], dtype=np.int32),
                    'label': chain_id,
                    'seq_chain_A': data['seq'],
                    'coords_chain_A': {
                        'N_chain_A': xyz_np[:, 0, :].tolist(),
                        'CA_chain_A': xyz_np[:, 1, :].tolist(),
                        'C_chain_A': xyz_np[:, 2, :].tolist(),
                        'O_chain_A': xyz_np[:, 3, :].tolist(),
                    },
                    'masked_list': ['A'],  # 🌟 수정: 원본 ProteinMPNN처럼 체인 A를 예측 대상으로 설정
                    'visible_list': [],
                    'num_of_chains': 1,
                    'name': chain_id,
                    'evo_profile': evo_profile,  # 🌟 evolutionary profile 추가
                }

                del data
                del xyz_np

                return result
            except Exception as e:
                import traceback
                traceback.print_exc(file=sys.stderr)
                return None

        return None


    # ----------------------------------------------------------------------
    # 2. PDB 파일 로딩 (메타데이터 기반 - 간소화된 버전)
    #    pdb_metadata.json을 사용하여 확실하게 존재하는 파일만 로드

    # 🌟 PDB 메타데이터 확인
    pdb_metadata = params.get('PDB_METADATA', {})

    if chain_id in pdb_metadata:
        # 메타데이터에서 정보 가져오기
        entry = pdb_metadata[chain_id]
        chain_path = entry['chain_path']
        other_chains = entry['other_chains']
        pdb_id = entry['pdb_id']
        target_chain_id = entry['chain_id']

        # 🔍 Debug: 처음 몇 개만 출력
        if 'DEBUG_PDB_COUNT' not in globals():
            globals()['DEBUG_PDB_COUNT'] = 0
        globals()['DEBUG_PDB_COUNT'] += 1
        if globals()['DEBUG_PDB_COUNT'] <= 5:
            print(f"Debug PDB: Loading {chain_id} from {chain_path}, other_chains={other_chains}", file=sys.stderr)

        try:
            # 타겟 체인 로드
            target_data = torch.load(chain_path, map_location='cpu')
            target_seq = target_data['seq']
            target_xyz = target_data['xyz'].cpu().numpy() if hasattr(target_data['xyz'], 'cpu') else np.array(target_data['xyz'])

            # 단일 체인인 경우 (other_chains가 비어있음)
            if not other_chains:
                L = len(target_seq)
                # 🌟 evo_profile 로드
                evo_profile = target_data.get('evo_profile', None)
                return {
                    'seq': target_seq,
                    'xyz': target_xyz.copy(),
                    'idx': np.zeros(L, dtype=np.int32),
                    'masked': np.array([0], dtype=np.int32),
                    'label': chain_id,
                    'seq_chain_A': target_seq,
                    'coords_chain_A': {
                        'N_chain_A': target_xyz[:, 0, :].tolist(),
                        'CA_chain_A': target_xyz[:, 1, :].tolist(),
                        'C_chain_A': target_xyz[:, 2, :].tolist(),
                        'O_chain_A': target_xyz[:, 3, :].tolist(),
                    },
                    'masked_list': ['A'],
                    'visible_list': [],
                    'num_of_chains': 1,
                    'name': chain_id,
                    'evo_profile': evo_profile,  # 🌟 evolutionary profile 추가
                }

            # 다중 체인인 경우 - 다른 체인들도 로드하여 assembly 구성
            all_seqs = [target_seq]
            all_xyz = [target_xyz]
            all_idx = [np.zeros(len(target_seq), dtype=np.int32)]

            chain_data = {
                'seq_chain_A': target_seq,
                'coords_chain_A': {
                    'N_chain_A': target_xyz[:, 0, :].tolist(),
                    'CA_chain_A': target_xyz[:, 1, :].tolist(),
                    'C_chain_A': target_xyz[:, 2, :].tolist(),
                    'O_chain_A': target_xyz[:, 3, :].tolist(),
                }
            }

            # 다른 체인들 로드 (context로 사용)
            # 🌟 성능 최적화: 최대 5개 체인만 랜덤 선택
            max_context_chains = 5
            if len(other_chains) > max_context_chains:
                other_chains_selected = random.sample(other_chains, max_context_chains)
            else:
                other_chains_selected = other_chains

            visible_letters = []
            loaded_chain_count = 0
            for other_ch in other_chains_selected:
                other_entry_id = f"{pdb_id}_{other_ch}"
                if other_entry_id in pdb_metadata:
                    other_path = pdb_metadata[other_entry_id]['chain_path']
                    if os.path.isfile(other_path):
                        other_data = torch.load(other_path, map_location='cpu')
                        other_seq = other_data['seq']
                        other_xyz = other_data['xyz'].cpu().numpy() if hasattr(other_data['xyz'], 'cpu') else np.array(other_data['xyz'])

                        chain_idx = loaded_chain_count + 1  # 0은 타겟 체인
                        letter = chain_alphabet[chain_idx]
                        loaded_chain_count += 1

                        all_seqs.append(other_seq)
                        all_xyz.append(other_xyz)
                        all_idx.append(np.full(len(other_seq), chain_idx, dtype=np.int32))
                        visible_letters.append(letter)

                        chain_data[f'seq_chain_{letter}'] = other_seq
                        chain_data[f'coords_chain_{letter}'] = {
                            f'N_chain_{letter}': other_xyz[:, 0, :].tolist(),
                            f'CA_chain_{letter}': other_xyz[:, 1, :].tolist(),
                            f'C_chain_{letter}': other_xyz[:, 2, :].tolist(),
                            f'O_chain_{letter}': other_xyz[:, 3, :].tolist(),
                        }

                        del other_data

            # 전체 시퀀스 및 좌표 합치기
            concat_seq = ''.join(all_seqs)
            concat_xyz = np.concatenate(all_xyz, axis=0)
            concat_idx = np.concatenate(all_idx, axis=0)

            num_chains = 1 + len(visible_letters)

            # 🌟 evo_profile 로드 (타겟 체인의 profile)
            evo_profile = target_data.get('evo_profile', None)

            result = {
                'seq': concat_seq,
                'xyz': concat_xyz.copy(),
                'idx': concat_idx,
                'masked': np.array([0], dtype=np.int32),  # 타겟 체인(A)만 masked
                'label': chain_id,
                'masked_list': ['A'],
                'visible_list': visible_letters,
                'num_of_chains': num_chains,
                'name': chain_id,
                'evo_profile': evo_profile,  # 🌟 evolutionary profile 추가
            }
            result.update(chain_data)

            del target_data
            return result

        except Exception as e:
            import traceback
            traceback.print_exc(file=sys.stderr)
            return None

    # 메타데이터에 없는 경우 - None 반환 (기존 복잡한 로직 제거)
    print(f"Debug PDB: {chain_id} not found in PDB metadata", file=sys.stderr)
    return None
    # ----------------------------------------------------------------------



def build_training_clusters(params, debug):
    val_ids = set([int(l) for l in open(params['VAL']).readlines()])
    test_ids = set([int(l) for l in open(params['TEST']).readlines()])
   
    if debug:
        val_ids = []
        test_ids = []
 
    # read & clean list.csv
    with open(params['LIST'], 'r') as f:
        reader = csv.reader(f)
        next(reader)
        rows = [[r[0],r[3],int(r[5])] for r in reader
                if float(r[2])<=params['RESCUT'] and
                parser.parse(r[1])<=parser.parse(params['DATCUT'])]
    
    # compile training and validation sets
    train = {}
    valid = {}
    test = {}

    if debug:
        rows = rows[:20]
    for r in rows:
        if r[2] in val_ids:
            if r[2] in valid.keys():
                valid[r[2]].append(r[:2])
            else:
                valid[r[2]] = [r[:2]]
        elif r[2] in test_ids:
            if r[2] in test.keys():
                test[r[2]].append(r[:2])
            else:
                test[r[2]] = [r[:2]]
        else:
            if r[2] in train.keys():
                train[r[2]].append(r[:2])
            else:
                train[r[2]] = [r[:2]]
    if debug:
        valid=train       
    return train, valid, test
