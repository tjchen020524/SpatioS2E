#!/usr/bin/env python3
"""Train Hist2ST on Visium spot-level expression (train slides only)."""

from __future__ import annotations
from workflow_paths import DATA_ROOT

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

ROOT = DATA_ROOT
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import collections
import collections.abc
if not hasattr(collections, "Iterable"):
    collections.Iterable = collections.abc.Iterable  # type: ignore[attr-defined]

from experiments.hist2st_spatial.hist2st_dataset import Hist2STSpotDataset

# add Hist2ST repo to path
HIST2ST_ROOT = ROOT / "Hist2ST"
if str(HIST2ST_ROOT) not in sys.path:
    sys.path.append(str(HIST2ST_ROOT))

from HIST2ST import Hist2ST


def load_config(path: Path) -> Dict:
    if path.suffix in {".yaml", ".yml"}:
        import yaml  # type: ignore
        return yaml.safe_load(path.read_text())
    return json.loads(path.read_text())


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_gene_list(path: Path) -> List[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gene_list = read_gene_list(Path(cfg["paths"]["gene_list"]))

    train_samples = cfg["split"]["train"]
    val_samples = cfg["split"]["val"] if "val" in cfg.get("split", {}) else []

    train_dataset = Hist2STSpotDataset(
        samples=train_samples,
        expression_root=Path(cfg["paths"]["expression_root"]),
        raw_root=Path(cfg["paths"]["raw_root"]),
        split="train",
        gene_list=gene_list,
        patch_size=int(cfg["data"]["patch_size"]),
        n_pos=int(cfg["model"]["n_pos"]),
        neighbor_k=int(cfg["data"]["neighbor_k"]),
        prune=str(cfg["data"].get("prune", "Grid")),
        max_spots=(int(cfg["data"]["max_spots"]) if cfg["data"].get("max_spots") else None),
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=int(cfg["data"].get("num_workers", 0)),
        collate_fn=lambda batch: batch[0],
    )

    val_loader = None
    if val_samples:
        val_dataset = Hist2STSpotDataset(
            samples=val_samples,
            expression_root=Path(cfg["paths"]["expression_root"]),
            raw_root=Path(cfg["paths"]["raw_root"]),
            split="val",
            gene_list=gene_list,
            patch_size=int(cfg["data"]["patch_size"]),
            n_pos=int(cfg["model"]["n_pos"]),
            neighbor_k=int(cfg["data"]["neighbor_k"]),
            prune=str(cfg["data"].get("prune", "Grid")),
            max_spots=(int(cfg["data"]["max_spots"]) if cfg["data"].get("max_spots") else None),
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=int(cfg["data"].get("num_workers", 0)),
            collate_fn=lambda batch: batch[0],
        )

    model = Hist2ST(
        learning_rate=float(cfg["optim"]["lr"]),
        fig_size=int(cfg["data"]["patch_size"]),
        dropout=float(cfg["model"]["dropout"]),
        n_pos=int(cfg["model"]["n_pos"]),
        kernel_size=int(cfg["model"]["kernel_size"]),
        patch_size=int(cfg["model"]["patch_embed"]),
        n_genes=len(gene_list),
        depth1=int(cfg["model"]["depth1"]),
        depth2=int(cfg["model"]["depth2"]),
        depth3=int(cfg["model"]["depth3"]),
        heads=int(cfg["model"]["heads"]),
        channel=int(cfg["model"]["channel"]),
        zinb=float(cfg["model"].get("zinb", 0.0)),
        nb=bool(cfg["model"].get("nb", False)),
        bake=int(cfg["model"].get("bake", 0)),
        lamb=float(cfg["model"].get("lamb", 0.0)),
        policy=str(cfg["model"].get("policy", "mean")),
    ).to(device)

    optim = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg["optim"]["lr"]),
        weight_decay=float(cfg["optim"].get("weight_decay", 0.0)),
    )

    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / cfg["paths"].get("checkpoint_name", "best.pt")

    max_epochs = int(cfg["optim"]["max_epochs"])
    best_val = float("inf")

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss = 0.0
        steps = 0
        for sample in train_loader:
            patches = sample.patches.unsqueeze(0).to(device)
            centers = sample.centers.unsqueeze(0).to(device)
            expr = sample.expr.to(device)
            adj = sample.adj.to(device)

            optim.zero_grad()
            pred, _, _ = model(patches, centers, adj)
            loss = torch.nn.functional.mse_loss(pred, expr)
            loss.backward()
            optim.step()

            train_loss += loss.item()
            steps += 1

        avg_train = train_loss / max(steps, 1)
        print(f"[epoch {epoch}/{max_epochs}] train_mse={avg_train:.4f}")

        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            vsteps = 0
            with torch.no_grad():
                for sample in val_loader:
                    patches = sample.patches.unsqueeze(0).to(device)
                    centers = sample.centers.unsqueeze(0).to(device)
                    expr = sample.expr.to(device)
                    adj = sample.adj.to(device)
                    pred, _, _ = model(patches, centers, adj)
                    loss = torch.nn.functional.mse_loss(pred, expr)
                    val_loss += loss.item()
                    vsteps += 1
            avg_val = val_loss / max(vsteps, 1)
            print(f"  val_mse={avg_val:.4f}")
            if avg_val < best_val:
                best_val = avg_val
                torch.save({"model": model.state_dict(), "config": cfg}, best_path)
                print(f"  [ckpt] saved best to {best_path}")
        else:
            torch.save({"model": model.state_dict(), "config": cfg}, best_path)


if __name__ == "__main__":
    main()
