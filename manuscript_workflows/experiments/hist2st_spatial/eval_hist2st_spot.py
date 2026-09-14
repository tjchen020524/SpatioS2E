#!/usr/bin/env python3
"""Evaluate Hist2ST spot-level prediction (no Decima interaction)."""

from __future__ import annotations
from workflow_paths import DATA_ROOT

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")

ROOT = DATA_ROOT
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import collections
import collections.abc
if not hasattr(collections, "Iterable"):
    collections.Iterable = collections.abc.Iterable  # type: ignore[attr-defined]

HIST2ST_ROOT = ROOT / "Hist2ST"
if str(HIST2ST_ROOT) not in sys.path:
    sys.path.append(str(HIST2ST_ROOT))

from experiments.hist2st_spatial.hist2st_dataset import Hist2STSpotDataset
from HIST2ST import Hist2ST


def load_config(path: Path) -> Dict:
    if path.suffix in {".yaml", ".yml"}:
        import yaml  # type: ignore

        return yaml.safe_load(path.read_text())
    return json.loads(path.read_text())


def read_gene_list(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def load_gene_subset(path: Optional[Path]) -> Optional[list[str]]:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"Gene subset file not found: {path}")
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


class RunningStats:
    def __init__(self) -> None:
        self.count = 0
        self.sum_pred = 0.0
        self.sum_true = 0.0
        self.sum_pred2 = 0.0
        self.sum_true2 = 0.0
        self.sum_pred_true = 0.0
        self.sum_sq_error = 0.0

    def update(self, pred: np.ndarray, true: np.ndarray) -> None:
        mask = np.isfinite(pred) & np.isfinite(true)
        if not np.any(mask):
            return
        p = pred[mask].astype(np.float64, copy=False)
        t = true[mask].astype(np.float64, copy=False)
        self.count += int(p.size)
        self.sum_pred += float(p.sum())
        self.sum_true += float(t.sum())
        self.sum_pred2 += float((p * p).sum())
        self.sum_true2 += float((t * t).sum())
        self.sum_pred_true += float((p * t).sum())
        diff = p - t
        self.sum_sq_error += float((diff * diff).sum())

    def mse(self) -> float:
        return self.sum_sq_error / self.count if self.count else float("nan")

    def corr(self) -> float:
        if self.count <= 1:
            return float("nan")
        mean_p = self.sum_pred / self.count
        mean_t = self.sum_true / self.count
        var_p = self.sum_pred2 / self.count - mean_p * mean_p
        var_t = self.sum_true2 / self.count - mean_t * mean_t
        if var_p <= 0 or var_t <= 0:
            return float("nan")
        cov = self.sum_pred_true / self.count - mean_p * mean_t
        return float(cov / np.sqrt(var_p * var_t))


def write_tsv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def eval_split(
    cfg_path: Path,
    ckpt_path: Path,
    split: str,
    save_dir: Path,
    device_override: Optional[str],
    gene_subset: Optional[list[str]],
    no_pandas: bool,
) -> None:
    cfg = load_config(cfg_path)
    if device_override:
        device = torch.device(device_override)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(1)

    gene_list = read_gene_list(Path(cfg["paths"]["gene_list"]))
    gene_to_idx = {g: i for i, g in enumerate(gene_list)}
    subset_idx = None
    subset_genes = None
    if gene_subset:
        subset_idx = [gene_to_idx[g] for g in gene_subset if g in gene_to_idx]
        if not subset_idx:
            raise ValueError("No overlap between gene_subset and gene_list.")
        subset_genes = [gene_list[i] for i in subset_idx]

    samples = cfg["split"].get(split, [])
    if not samples:
        raise SystemExit(f"No samples for split {split}")

    ds = Hist2STSpotDataset(
        samples=samples,
        expression_root=Path(cfg["paths"]["expression_root"]),
        raw_root=Path(cfg["paths"]["raw_root"]),
        split=split,
        gene_list=gene_list,
        patch_size=int(cfg["data"]["patch_size"]),
        n_pos=int(cfg["model"]["n_pos"]),
        neighbor_k=int(cfg["data"]["neighbor_k"]),
        prune=str(cfg["data"].get("prune", "Grid")),
        max_spots=(int(cfg["data"]["max_spots"]) if cfg["data"].get("max_spots") else None),
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

    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state)
    model.eval()

    overall = RunningStats()
    sample_stats: Dict[str, RunningStats] = {}
    gene_stats: Dict[str, RunningStats] = {}

    with torch.no_grad():
        for sample in ds:
            patches = sample.patches.unsqueeze(0).to(device)
            centers = sample.centers.unsqueeze(0).to(device)
            adj = sample.adj.to(device)
            pred, _, _ = model(patches, centers, adj)
            pred = pred.squeeze(0)
            true = sample.expr.to(device)

            if subset_idx is not None:
                pred = pred[:, subset_idx]
                true = true[:, subset_idx]
                gene_ids = subset_genes
            else:
                gene_ids = gene_list

            pred_np = pred.detach().cpu().numpy()
            true_np = true.detach().cpu().numpy()
            sample_stats.setdefault(sample.sample, RunningStats())
            overall.update(pred_np.ravel(), true_np.ravel())
            sample_stats[sample.sample].update(pred_np.ravel(), true_np.ravel())

            for j, gid in enumerate(gene_ids):
                gene_stats.setdefault(gid, RunningStats()).update(pred_np[:, j], true_np[:, j])

    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / f"{split}_overall.json").write_text(
        json.dumps({"split": split, "mse": overall.mse(), "corr": overall.corr()}, indent=2)
    )

    sample_rows = [{"sample": k, "mse": v.mse(), "corr": v.corr()} for k, v in sample_stats.items()]
    gene_rows = [{"gene_id": k, "mse": v.mse(), "corr": v.corr()} for k, v in gene_stats.items()]
    if no_pandas:
        write_tsv(save_dir / f"{split}_sample_metrics.tsv", sample_rows)
        write_tsv(save_dir / f"{split}_gene_metrics.tsv", gene_rows)
    else:
        import pandas as pd

        pd.DataFrame(sample_rows).to_csv(save_dir / f"{split}_sample_metrics.tsv", sep="\t", index=False)
        pd.DataFrame(gene_rows).to_csv(save_dir / f"{split}_gene_metrics.tsv", sep="\t", index=False)
    print(json.dumps({"split": split, "mse": overall.mse(), "corr": overall.corr()}, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--save-dir", type=Path, default=Path("experiments/hist2st_spatial/results"))
    ap.add_argument("--device", type=str, default=None, help="Force device: 'cpu' or 'cuda'.")
    ap.add_argument("--gene-subset-file", type=Path, default=None)
    ap.add_argument("--no-pandas", action="store_true")
    args = ap.parse_args()
    gene_subset = load_gene_subset(args.gene_subset_file)
    eval_split(args.config, args.ckpt, args.split, args.save_dir, args.device, gene_subset, args.no_pandas)
