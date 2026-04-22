#!/usr/bin/env python3
"""Train FoldMapper.

All paths are CLI flags; architecture is selected via flags or a YAML config.

Typical call:
    python scripts/train.py \\
        --config configs/d256_blosum.yaml \\
        --path-for-training-data /path/to/dataset_splits/data \\
        --pdb-root /path/to/pt_with_profile/pdb \\
        --ted-root /path/to/pt_with_profile/ted \\
        --ted-metadata /path/to/ted_metadata.json \\
        --pdb-metadata /path/to/pdb_metadata.json \\
        --path-for-outputs ./outputs
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from foldmapper.model import FoldMapper, FoldMapperConfig  # noqa: E402
from foldmapper.model_utils import featurize, loss_smoothed, loss_nll, get_std_opt  # noqa: E402
from foldmapper.utils import (  # noqa: E402
    worker_init_fn,
    get_pdbs,
    loader_pdb,
    build_training_clusters,
    PDB_dataset,
    StructureDataset,
    StructureLoader,
    load_ted_metadata,
)


def get_slurm_cpu_count():
    for env in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE"):
        if env in os.environ:
            return int(os.environ[env])
    return os.cpu_count() or 4


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    # Config override
    p.add_argument("--config", type=str, default=None,
                   help="YAML with FoldMapperConfig fields + training defaults.")

    # Data paths
    p.add_argument("--path-for-training-data", dest="path_for_training_data", required=True,
                   help="Directory with list.csv, valid_clusters_ids.txt, test_clusters_ids.txt.")
    p.add_argument("--pdb-root", dest="pdb_root", required=True, help="PDB .pt shards root.")
    p.add_argument("--ted-root", dest="ted_root", required=True, help="TED .pt shards root.")
    p.add_argument("--ted-metadata", dest="ted_metadata", required=True)
    p.add_argument("--pdb-metadata", dest="pdb_metadata", required=True)
    p.add_argument("--path-for-outputs", dest="path_for_outputs", default="./outputs")
    p.add_argument("--previous-checkpoint", dest="previous_checkpoint", default="")

    # Training loop
    p.add_argument("--num-epochs", dest="num_epochs", type=int, default=200)
    p.add_argument("--save-model-every-n-epochs", dest="save_model_every_n_epochs", type=int, default=20)
    p.add_argument("--reload-data-every-n-epochs", dest="reload_data_every_n_epochs", type=int, default=1)
    p.add_argument("--num-examples-per-epoch", dest="num_examples_per_epoch", type=int, default=30000)
    p.add_argument("--batch-size", dest="batch_size", type=int, default=2000)
    p.add_argument("--max-protein-length", dest="max_protein_length", type=int, default=10000)
    p.add_argument("--gradient-norm", dest="gradient_norm", type=float, default=0.5)
    p.add_argument("--mixed-precision", dest="mixed_precision", action="store_true", default=True)
    p.add_argument("--rescut", type=float, default=3.5)
    p.add_argument("--debug", action="store_true")

    # Stage 1 (evolutionary encoder)
    p.add_argument("--s1-hidden-dim", dest="s1_hidden_dim", type=int, default=256)
    p.add_argument("--s1-edge-features", dest="s1_edge_features", type=int, default=128)
    p.add_argument("--s1-num-encoder-layers", dest="s1_num_encoder_layers", type=int, default=3)
    p.add_argument("--s1-k-neighbors", dest="s1_k_neighbors", type=int, default=30)
    p.add_argument("--s1-augment-eps", dest="s1_augment_eps", type=float, default=0.1)

    # Stage 2 (structure-conditioned MPNN)
    p.add_argument("--hidden-dim", dest="hidden_dim", type=int, default=512)
    p.add_argument("--num-encoder-layers", dest="num_encoder_layers", type=int, default=3)
    p.add_argument("--num-decoder-layers", dest="num_decoder_layers", type=int, default=3)
    p.add_argument("--num-neighbors", dest="num_neighbors", type=int, default=48)
    p.add_argument("--backbone-noise", dest="backbone_noise", type=float, default=0.2)

    # Regularisation
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--profile-dropout", dest="profile_dropout", type=float, default=0.1)

    # BLOSUM62
    p.add_argument("--use-blosum", dest="use_blosum", action="store_true")
    p.add_argument("--aa-factor", dest="aa_factor", type=float, default=1.4)

    # wandb
    p.add_argument("--use-wandb", dest="use_wandb", action="store_true")
    p.add_argument("--wandb-project", dest="wandb_project", default="foldmapper")
    p.add_argument("--wandb-run-name", dest="wandb_run_name", default=None)
    p.add_argument("--wandb-run-id", dest="wandb_run_id", default=None)

    return p


def apply_yaml_overrides(args, config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f) or {}
    for k, v in cfg.items():
        key = k.replace("-", "_")
        if hasattr(args, key):
            setattr(args, key, v)
        else:
            print(f"[warn] YAML key {k!r} not a known CLI arg; ignoring.", file=sys.stderr)
    return args


def main():
    args = build_parser().parse_args()
    if args.config:
        args = apply_yaml_overrides(args, args.config)

    if args.use_wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            id=args.wandb_run_id,
            resume="allow" if args.wandb_run_id else None,
            config=vars(args),
            dir=args.path_for_outputs,
            settings=wandb.Settings(init_timeout=300),
        )

    scaler = torch.cuda.amp.GradScaler()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    base_folder = time.strftime(args.path_for_outputs, time.localtime())
    if base_folder[-1] != "/":
        base_folder += "/"
    os.makedirs(base_folder + "model_weights", exist_ok=True)

    PATH = args.previous_checkpoint
    logfile = base_folder + "log.txt"
    if not PATH:
        with open(logfile, "w") as f:
            f.write("Epoch\tTrain\tValidation\n")

    data_path = args.path_for_training_data

    print(f"[data] pre-loading TED metadata from {args.ted_metadata}...", flush=True)
    ted_metadata_data = load_ted_metadata(args.ted_metadata)

    print(f"[data] loading PDB metadata from {args.pdb_metadata}...", flush=True)
    with open(args.pdb_metadata, "r") as f:
        pdb_metadata = json.load(f)
    print(f"[data] PDB metadata: {len(pdb_metadata)} entries", flush=True)

    params = {
        "LIST": f"{data_path}/list.csv",
        "VAL": f"{data_path}/valid_clusters_ids.txt",
        "TEST": f"{data_path}/test_clusters_ids.txt",
        "DIR_OLD": args.pdb_root,
        "DIR_NEW": args.ted_root,
        "TED_METADATA": args.ted_metadata,
        "TED_METADATA_DATA": ted_metadata_data,
        "PDB_METADATA": pdb_metadata,
        "DATCUT": "2030-Jan-01",
        "RESCUT": args.rescut,
        "HOMO": 0.70,
    }

    total_cpus = get_slurm_cpu_count()
    print(f"[data] cpus={total_cpus}, ThreadPoolExecutor(max_workers=2)", flush=True)

    load_param = {"batch_size": 1, "shuffle": True, "pin_memory": False, "num_workers": 0}

    if args.debug:
        args.num_examples_per_epoch = 50
        args.max_protein_length = 1000
        args.batch_size = 1000

    train, valid, _ = build_training_clusters(params, args.debug)
    train_set = PDB_dataset(list(train.keys()), loader_pdb, train, params)
    valid_set = PDB_dataset(list(valid.keys()), loader_pdb, valid, params)
    train_loader = torch.utils.data.DataLoader(train_set, worker_init_fn=worker_init_fn, **load_param)
    valid_loader = torch.utils.data.DataLoader(valid_set, worker_init_fn=worker_init_fn, **load_param)

    config = FoldMapperConfig(
        s1_hidden_dim=args.s1_hidden_dim,
        s1_edge_features=args.s1_edge_features,
        s1_num_encoder_layers=args.s1_num_encoder_layers,
        s1_k_neighbors=args.s1_k_neighbors,
        s1_augment_eps=args.s1_augment_eps,
        hidden_dim=args.hidden_dim,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        k_neighbors=args.num_neighbors,
        augment_eps=args.backbone_noise,
        dropout=args.dropout,
        profile_dropout=args.profile_dropout,
        use_blosum=args.use_blosum,
        aa_factor=args.aa_factor,
    )
    model = FoldMapper(config).to(device)
    print(f"[model] params={sum(p.numel() for p in model.parameters())}", flush=True)

    total_step = 0
    epoch = 0
    if PATH:
        checkpoint = torch.load(PATH)
        total_step = checkpoint["step"]
        epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["model_state_dict"])

    optimizer = get_std_opt(model.parameters(), args.hidden_dim, total_step)
    if PATH:
        optimizer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    num_valid_examples = min(10000, args.num_examples_per_epoch // 10)
    with ThreadPoolExecutor(max_workers=2) as executor:
        q = queue.Queue(maxsize=1)
        p_q = queue.Queue(maxsize=1)
        q.put_nowait(executor.submit(get_pdbs, train_loader, 1, args.max_protein_length, args.num_examples_per_epoch))
        p_q.put_nowait(executor.submit(get_pdbs, valid_loader, 1, args.max_protein_length, num_valid_examples))

        pdb_dict_train = q.get().result()
        pdb_dict_valid = p_q.get().result()
        q.put_nowait(executor.submit(get_pdbs, train_loader, 1, args.max_protein_length, args.num_examples_per_epoch))
        p_q.put_nowait(executor.submit(get_pdbs, valid_loader, 1, args.max_protein_length, num_valid_examples))

        dataset_train = StructureDataset(pdb_dict_train, truncate=None, max_length=args.max_protein_length)
        dataset_valid = StructureDataset(pdb_dict_valid, truncate=None, max_length=args.max_protein_length)
        loader_train = StructureLoader(dataset_train, batch_size=args.batch_size, drop_last=False)
        loader_valid = StructureLoader(dataset_valid, batch_size=args.batch_size, drop_last=False)

        reload_c = 0
        for e in range(args.num_epochs):
            t0_epoch = time.time()
            e_total = epoch + e
            model.train()
            train_sum, train_weights, train_acc = 0.0, 0.0, 0.0

            data_loading_time = 0.0
            if e_total % args.reload_data_every_n_epochs == 0:
                if reload_c != 0:
                    t_reload_start = time.time()
                    pdb_dict_train = q.get().result()
                    dataset_train = StructureDataset(pdb_dict_train, truncate=None, max_length=args.max_protein_length)
                    loader_train = StructureLoader(dataset_train, batch_size=args.batch_size, drop_last=False)
                    pdb_dict_valid = p_q.get().result()
                    dataset_valid = StructureDataset(pdb_dict_valid, truncate=None, max_length=args.max_protein_length)
                    loader_valid = StructureLoader(dataset_valid, batch_size=args.batch_size, drop_last=False)
                    q.put_nowait(executor.submit(get_pdbs, train_loader, 1, args.max_protein_length, args.num_examples_per_epoch))
                    p_q.put_nowait(executor.submit(get_pdbs, valid_loader, 1, args.max_protein_length, num_valid_examples))
                    data_loading_time = time.time() - t_reload_start
                reload_c += 1

            t_train_start = time.time()
            for batch in loader_train:
                if not batch:
                    continue
                X, S, mask, lengths, chain_M, residue_idx, mask_self, chain_encoding_all, evo_profile = featurize(
                    batch, device
                )
                if torch.isnan(X).any() or torch.isnan(S).any():
                    continue
                optimizer.zero_grad()
                mask_for_loss = mask * chain_M
                if torch.sum(mask_for_loss) == 0:
                    continue

                if args.mixed_precision:
                    with torch.cuda.amp.autocast():
                        log_probs = model(
                            X, S, mask, chain_M, residue_idx, chain_encoding_all,
                            evo_profile=evo_profile,
                        )
                        if torch.isnan(log_probs).any() or torch.isinf(log_probs).any():
                            continue
                        _, loss_av_smoothed = loss_smoothed(S, log_probs, mask_for_loss)
                        if torch.isnan(loss_av_smoothed) or torch.isinf(loss_av_smoothed):
                            continue
                    scaler.scale(loss_av_smoothed).backward()
                    if args.gradient_norm > 0:
                        scaler.unscale_(optimizer)
                        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_norm)
                        if torch.isnan(torch.tensor(total_norm)) or torch.isinf(torch.tensor(total_norm)):
                            optimizer.zero_grad()
                            scaler.update()
                            continue
                    scaler.step(optimizer)
                    scaler.update()
                    with torch.no_grad():
                        log_probs_logging = model(
                            X, S, mask, chain_M, residue_idx, chain_encoding_all,
                            evo_profile=evo_profile,
                        )
                        loss, loss_av, true_false = loss_nll(S, log_probs_logging, mask_for_loss)
                else:
                    log_probs = model(
                        X, S, mask, chain_M, residue_idx, chain_encoding_all,
                        evo_profile=evo_profile,
                    )
                    if torch.isnan(log_probs).any() or torch.isinf(log_probs).any():
                        continue
                    _, loss_av_smoothed = loss_smoothed(S, log_probs, mask_for_loss)
                    if torch.isnan(loss_av_smoothed) or torch.isinf(loss_av_smoothed):
                        continue
                    loss_av_smoothed.backward()
                    if args.gradient_norm > 0:
                        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_norm)
                        if torch.isnan(torch.tensor(total_norm)) or torch.isinf(torch.tensor(total_norm)):
                            optimizer.zero_grad()
                            continue
                    optimizer.step()
                    loss, loss_av, true_false = loss_nll(S, log_probs, mask_for_loss)

                train_sum += float(torch.sum(loss * mask_for_loss).cpu())
                train_acc += float(torch.sum(true_false * mask_for_loss).cpu())
                train_weights += float(torch.sum(mask_for_loss).cpu())
                total_step += 1

            train_time = time.time() - t_train_start

            model.eval()
            t_valid_start = time.time()
            validation_sum, validation_weights, validation_acc = 0.0, 0.0, 0.0
            with torch.no_grad():
                for batch in loader_valid:
                    X, S, mask, lengths, chain_M, residue_idx, mask_self, chain_encoding_all, evo_profile = featurize(
                        batch, device
                    )
                    log_probs = model(X, S, mask, chain_M, residue_idx, chain_encoding_all, evo_profile=evo_profile)
                    mask_for_loss = mask * chain_M
                    loss, loss_av, true_false = loss_nll(S, log_probs, mask_for_loss)
                    validation_sum += float(torch.sum(loss * mask_for_loss).cpu())
                    validation_acc += float(torch.sum(true_false * mask_for_loss).cpu())
                    validation_weights += float(torch.sum(mask_for_loss).cpu())
            valid_time = time.time() - t_valid_start

            train_loss = train_sum / max(train_weights, 1e-9)
            validation_loss = validation_sum / max(validation_weights, 1e-9)
            msg = (
                f"epoch: {e_total + 1}, step: {total_step}, "
                f"time: {time.time() - t0_epoch:.1f}s "
                f"(data: {data_loading_time:.1f}s, train: {train_time:.1f}s, valid: {valid_time:.1f}s), "
                f"train_ppl: {np.exp(train_loss):.3f}, valid_ppl: {np.exp(validation_loss):.3f}, "
                f"train_acc: {train_acc / max(train_weights, 1e-9):.3f}, "
                f"valid_acc: {validation_acc / max(validation_weights, 1e-9):.3f}"
            )
            with open(logfile, "a") as f:
                f.write(msg + "\n")
            print(msg, flush=True)

            if args.use_wandb:
                import wandb
                wandb.log({
                    "epoch": e_total + 1,
                    "step": total_step,
                    "train/loss": train_loss,
                    "train/perplexity": np.exp(train_loss),
                    "valid/loss": validation_loss,
                    "valid/perplexity": np.exp(validation_loss),
                })

            ckpt_dict = {
                "epoch": e_total + 1,
                "step": total_step,
                "num_edges": args.num_neighbors,
                "noise_level": args.backbone_noise,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.optimizer.state_dict(),
                "config": asdict(config),
            }
            torch.save(ckpt_dict, base_folder + "model_weights/epoch_last.pt")
            if (e_total + 1) % args.save_model_every_n_epochs == 0:
                torch.save(
                    ckpt_dict,
                    base_folder + f"model_weights/epoch{e_total + 1}_step{total_step}.pt",
                )


if __name__ == "__main__":
    if sys.platform != "darwin":
        try:
            torch.multiprocessing.set_start_method("spawn", force=True)
        except RuntimeError:
            pass
        torch.multiprocessing.set_sharing_strategy("file_system")
    main()
