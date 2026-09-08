#!/usr/bin/env python3
"""Train the public GeneQuery gene-aware head under the frozen double-disjoint split."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from common import (
    FEATURE_ROOT,
    RUN_ROOT,
    counterfactual_embeddings,
    load_manifest,
    load_panel_artifact,
    load_partition,
)
from model import GeneQueryHead, parameter_count


VARIANTS = ("semantic", "identity_shuffle", "random", "constant")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loader(
    features: np.ndarray,
    targets: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
    workers: int,
) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(features), torch.from_numpy(targets))
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
        drop_last=False,
    )


def forward_loss(
    model: nn.Module,
    image: torch.Tensor,
    target: torch.Tensor,
    genes: torch.Tensor,
    amp: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    device_type = image.device.type
    with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=amp):
        prediction = model(image, genes)
        loss = nn.functional.mse_loss(prediction, target)
    return prediction, loss


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    genes: torch.Tensor,
    device: torch.device,
    amp: bool,
) -> float:
    model.eval()
    squared_error = 0.0
    count = 0
    for image, target in loader:
        image = image.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        prediction, _ = forward_loss(model, image, target, genes, amp)
        squared_error += float(torch.square(prediction.float() - target).sum().item())
        count += int(target.numel())
    return squared_error / count


@torch.no_grad()
def predict(
    model: nn.Module,
    features: np.ndarray,
    genes: torch.Tensor,
    device: torch.device,
    batch_size: int,
    workers: int,
    amp: bool,
) -> np.ndarray:
    dummy = np.zeros((len(features), 1), dtype=np.float32)
    loader = make_loader(features, dummy, batch_size, False, 0, workers)
    model.eval()
    blocks = []
    for image, _ in loader:
        image = image.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            blocks.append(model(image, genes).float().cpu().numpy())
    return np.concatenate(blocks).astype(np.float32, copy=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANTS, default="semantic")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--feature-root", type=Path, default=FEATURE_ROOT)
    parser.add_argument("--output-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-3)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--heads", type=int, default=64)
    parser.add_argument("--dim-head", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--max-spots", type=int, default=0, help="Debug-only cap per partition.")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device(args.device)
    amp = device.type == "cuda" and not args.no_amp
    panel = load_panel_artifact()
    labels = panel["split_labels"].astype(str)
    train_mask = labels == "train"
    heldout_mask = labels == "heldout"
    gene_ids = panel["gene_ids"].astype(str)
    all_embeddings = counterfactual_embeddings(
        panel["semantic_embeddings"], labels, args.variant, args.seed
    )
    train_embeddings = torch.from_numpy(all_embeddings[train_mask]).to(device)
    heldout_embeddings = torch.from_numpy(all_embeddings[heldout_mask]).to(device)

    run_dir = args.output_root / ("seed_%d" % args.seed) / args.variant
    checkpoint_path = run_dir / "best.pt"
    prediction_path = run_dir / "predictions.npz"
    if prediction_path.exists() and not args.overwrite:
        raise FileExistsError("Run output already exists: %s" % prediction_path)
    run_dir.mkdir(parents=True, exist_ok=True)

    train_x, train_all_y, _, _ = load_partition(
        "train", gene_ids, args.feature_root, args.max_spots
    )
    val_x, val_all_y, _, _ = load_partition("val", gene_ids, args.feature_root, args.max_spots)
    train_y = train_all_y[:, train_mask]
    val_y = val_all_y[:, train_mask]
    train_loader = make_loader(
        train_x, train_y, args.batch_size, True, args.seed, args.workers
    )
    val_loader = make_loader(
        val_x, val_y, args.eval_batch_size, False, args.seed, args.workers
    )

    model = GeneQueryHead(
        image_dim=train_x.shape[1],
        gene_embedding_dim=all_embeddings.shape[1],
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        heads=args.heads,
        dim_head=args.dim_head,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    history = []
    best_val = math.inf
    best_epoch = 0
    stale = 0
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_sse = 0.0
        total_count = 0
        for image, target in train_loader:
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            prediction, loss = forward_loss(model, image, target, train_embeddings, amp)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_sse += float(torch.square(prediction.detach().float() - target).sum().item())
            total_count += int(target.numel())
        train_mse = total_sse / total_count
        val_mse = evaluate(model, val_loader, train_embeddings, device, amp)
        row = {"epoch": epoch, "train_mse": train_mse, "val_mse": val_mse}
        history.append(row)
        print(json.dumps(row), flush=True)
        if val_mse < best_val:
            best_val = val_mse
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "val_mse": val_mse,
                    "variant": args.variant,
                    "seed": args.seed,
                },
                checkpoint_path,
            )
        else:
            stale += 1
            if args.patience > 0 and stale >= args.patience:
                break

    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    test_x, test_all_y, test_samples, test_barcodes = load_partition(
        "test", gene_ids, args.feature_root, args.max_spots
    )
    test_true = test_all_y[:, heldout_mask]
    test_prediction = predict(
        model,
        test_x,
        heldout_embeddings,
        device,
        args.eval_batch_size,
        args.workers,
        amp,
    )
    patient_map = load_manifest().set_index("sample")["patient"].astype(str).to_dict()
    individuals = np.asarray([patient_map[sample] for sample in test_samples])
    np.savez_compressed(
        prediction_path,
        pred=test_prediction,
        true=test_true.astype(np.float32, copy=False),
        gene_ids=gene_ids[heldout_mask],
        symbols=panel["symbols"].astype(str)[heldout_mask],
        sample_ids=test_samples,
        individual_ids=individuals,
        barcodes=test_barcodes,
    )
    config = vars(args).copy()
    for key, value in list(config.items()):
        if isinstance(value, Path):
            config[key] = str(value)
    config.update(
        {
            "target_scale": "log1p_counts_per_million",
            "image_backbone": "frozen_timm_resnet50_a1_in1k",
            "model_parameters": parameter_count(model),
            "n_train_spots": int(len(train_x)),
            "n_val_spots": int(len(val_x)),
            "n_test_spots": int(len(test_x)),
            "n_train_genes": int(train_mask.sum()),
            "n_heldout_genes": int(heldout_mask.sum()),
            "best_epoch": best_epoch,
            "best_val_mse": best_val,
            "elapsed_seconds": time.time() - start,
            "history": history,
        }
    )
    (run_dir / "run.json").write_text(json.dumps(config, indent=2) + "\n")
    print("Saved %s" % prediction_path)


if __name__ == "__main__":
    main()
