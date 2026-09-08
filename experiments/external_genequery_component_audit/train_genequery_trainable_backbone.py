#!/usr/bin/env python3
"""Fine-tune the GeneQuery head and ResNet-50 under the double-disjoint split."""

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
from torch.utils.data import DataLoader

from common import (
    PATCH_ROOT,
    TRAINABLE_RUN_ROOT,
    counterfactual_embeddings,
    load_manifest,
    load_panel_artifact,
    sha256,
)
from extract_resnet50_features import DEFAULT_CHECKPOINT, load_backbone
from model import GeneQueryHead, parameter_count
from patch_dataset import PatchExpressionDataset


VARIANTS = ("semantic", "identity_shuffle", "random", "constant")


class EndToEndGeneQuery(nn.Module):
    def __init__(self, backbone: nn.Module, head: GeneQueryHead) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, image: torch.Tensor, genes: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(image), genes)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loader(
    dataset: PatchExpressionDataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
    workers: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
        drop_last=False,
        persistent_workers=workers > 0,
    )


def forward_loss(
    model: nn.Module,
    image: torch.Tensor,
    target: torch.Tensor,
    genes: torch.Tensor,
    amp: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    with torch.autocast(device_type=image.device.type, dtype=torch.float16, enabled=amp):
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
    loader: DataLoader,
    genes: torch.Tensor,
    device: torch.device,
    amp: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    predictions = []
    truths = []
    for image, target in loader:
        image = image.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            predictions.append(model(image, genes).float().cpu().numpy())
        truths.append(target.numpy())
    return (
        np.concatenate(predictions).astype(np.float32, copy=False),
        np.concatenate(truths).astype(np.float32, copy=False),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANTS, default="semantic")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patch-root", type=Path, default=PATCH_ROOT)
    parser.add_argument("--output-root", type=Path, default=TRAINABLE_RUN_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
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
    parser.add_argument("--augment", action="store_true")
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

    train_data = PatchExpressionDataset(
        "train", gene_ids, train_mask, args.patch_root, args.augment, args.max_spots
    )
    val_data = PatchExpressionDataset(
        "val", gene_ids, train_mask, args.patch_root, False, args.max_spots
    )
    train_loader = make_loader(
        train_data, args.batch_size, True, args.seed, args.workers
    )
    val_loader = make_loader(
        val_data, args.eval_batch_size, False, args.seed, args.workers
    )

    backbone = load_backbone(args.checkpoint, device)
    for parameter in backbone.parameters():
        parameter.requires_grad_(True)
    head = GeneQueryHead(
        image_dim=2048,
        gene_embedding_dim=all_embeddings.shape[1],
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        heads=args.heads,
        dim_head=args.dim_head,
        dropout=args.dropout,
    )
    model = EndToEndGeneQuery(backbone, head).to(device)
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
    test_data = PatchExpressionDataset(
        "test", gene_ids, heldout_mask, args.patch_root, False, args.max_spots
    )
    test_loader = make_loader(
        test_data, args.eval_batch_size, False, args.seed, args.workers
    )
    test_prediction, test_true = predict(
        model, test_loader, heldout_embeddings, device, amp
    )
    patient_map = load_manifest().set_index("sample")["patient"].astype(str).to_dict()
    individuals = np.asarray([patient_map[sample] for sample in test_data.sample_ids])
    np.savez_compressed(
        prediction_path,
        pred=test_prediction,
        true=test_true,
        gene_ids=gene_ids[heldout_mask],
        symbols=panel["symbols"].astype(str)[heldout_mask],
        sample_ids=test_data.sample_ids,
        individual_ids=individuals,
        barcodes=test_data.barcodes,
    )
    config = vars(args).copy()
    for key, value in list(config.items()):
        if isinstance(value, Path):
            config[key] = str(value)
    config.update(
        {
            "target_scale": "log1p_counts_per_million",
            "image_backbone": "trainable_timm_resnet50_a1_in1k",
            "checkpoint_sha256": sha256(args.checkpoint),
            "augmentation": "random_flip_rot90_per_access" if args.augment else "none",
            "model_parameters": parameter_count(model),
            "trainable_parameters": int(
                sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
            ),
            "n_train_spots": int(len(train_data)),
            "n_val_spots": int(len(val_data)),
            "n_test_spots": int(len(test_data)),
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
