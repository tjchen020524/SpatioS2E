#!/usr/bin/env python3
"""Evaluate finetuned STPath baseline predictions (no Decima interaction)."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from experiments.stpath_finetune_spatios2e.common import (
    csr_to_dense,
    load_expression,
    load_yaml_or_json,
    resolve_split,
)


class FinetunedBaselineDataset(torch.utils.data.Dataset):
    def __init__(self, samples, expression_root: Path, pred_root: Path, split_name: str):
        self.samples = list(samples)
        self.expression_root = expression_root
        self.pred_root = pred_root
        self.split_name = split_name

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        pred_path = self.pred_root / sample / "stpath_pred.npz"
        if not pred_path.exists():
            raise FileNotFoundError(f"Missing prediction {pred_path}")
        pred_obj = np.load(pred_path, allow_pickle=True)
        pred = pred_obj["pred"].astype(np.float32)
        p_barc = pred_obj["barcodes"].astype(str).tolist()
        gene_ids = pred_obj["gene_ids"].astype(str).tolist()

        expr = load_expression(self.expression_root / sample / f"{self.split_name}.npz")
        y = csr_to_dense(expr["csr"])
        e_barc = expr["barcodes"]
        if gene_ids != expr["gene_ids"]:
            raise ValueError(f"Gene order mismatch in {sample}")

        e_index = {bc: i for i, bc in enumerate(e_barc)}
        keep_p, keep_e = [], []
        for i, bc in enumerate(p_barc):
            j = e_index.get(bc)
            if j is not None:
                keep_p.append(i)
                keep_e.append(j)
        if not keep_e:
            raise ValueError(f"No overlapping barcodes for {sample}")

        pred = pred[keep_p]
        y = y[keep_e]
        return {
            "sample": sample,
            "pred": torch.tensor(pred, dtype=torch.float32),
            "y": torch.tensor(y, dtype=torch.float32),
            "gene_ids": gene_ids,
        }


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
        d = p - t
        self.sum_sq_error += float((d * d).sum())

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


def load_gene_subset(path: Optional[Path]) -> Optional[list[str]]:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"Gene subset file not found: {path}")
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def write_tsv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def eval_split(cfg_path: Path, split_name: str, save_dir: Path, max_genes: Optional[int], gene_subset: Optional[list[str]], no_pandas: bool) -> None:
    cfg = load_yaml_or_json(cfg_path)
    split_map = resolve_split(cfg)
    samples = split_map.get(split_name, [])
    if not samples:
        raise SystemExit(f"No samples for split {split_name}")

    ds = FinetunedBaselineDataset(
        samples=samples,
        expression_root=Path(cfg["paths"]["expression_root"]),
        pred_root=Path(cfg["paths"]["pred_out_root"]),
        split_name=split_name,
    )
    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, collate_fn=lambda b: b[0])

    overall = RunningStats()
    sample_stats: Dict[str, RunningStats] = {}
    gene_stats: Dict[str, RunningStats] = {}

    for batch in loader:
        pred = batch["pred"]
        true = batch["y"]
        gene_ids = list(batch["gene_ids"]) if not isinstance(batch["gene_ids"], list) else batch["gene_ids"]

        if gene_subset:
            subset = set(gene_subset)
            idx = [i for i, gid in enumerate(gene_ids) if gid in subset]
            if not idx:
                continue
            gene_ids = [gene_ids[i] for i in idx]
            pred = pred[:, idx]
            true = true[:, idx]
        elif max_genes is not None and true.shape[1] > max_genes:
            perm = torch.randperm(true.shape[1])[:max_genes]
            pred = pred[:, perm]
            true = true[:, perm]
            gene_ids = [gene_ids[i] for i in perm.tolist()]

        pred_np = pred.cpu().numpy()
        true_np = true.cpu().numpy()
        sample_name = batch["sample"]

        overall.update(pred_np.ravel(), true_np.ravel())
        sample_stats.setdefault(sample_name, RunningStats()).update(pred_np.ravel(), true_np.ravel())
        for j, gid in enumerate(gene_ids):
            gene_stats.setdefault(gid, RunningStats()).update(pred_np[:, j], true_np[:, j])

    save_dir.mkdir(parents=True, exist_ok=True)
    out = {"split": split_name, "mse": overall.mse(), "corr": overall.corr(), "mode": "stpath_finetuned_only"}
    (save_dir / f"{split_name}_overall.json").write_text(json.dumps(out, indent=2))

    sample_rows = [{"sample": k, "mse": v.mse(), "corr": v.corr()} for k, v in sample_stats.items()]
    gene_rows = [{"gene_id": k, "mse": v.mse(), "corr": v.corr()} for k, v in gene_stats.items()]

    if no_pandas:
        write_tsv(save_dir / f"{split_name}_sample_metrics.tsv", sample_rows)
        write_tsv(save_dir / f"{split_name}_gene_metrics.tsv", gene_rows)
    else:
        import pandas as pd

        pd.DataFrame(sample_rows).to_csv(save_dir / f"{split_name}_sample_metrics.tsv", sep="\t", index=False)
        pd.DataFrame(gene_rows).to_csv(save_dir / f"{split_name}_gene_metrics.tsv", sep="\t", index=False)

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--save-dir", type=Path, default=Path("experiments/stpath_finetune_spatios2e/results_stpath_finetuned_only"))
    ap.add_argument("--max-genes-per-batch", type=int, default=None)
    ap.add_argument("--gene-subset-file", type=Path, default=None)
    ap.add_argument("--no-pandas", action="store_true")
    args = ap.parse_args()
    eval_split(
        cfg_path=args.config,
        split_name=args.split,
        save_dir=args.save_dir,
        max_genes=args.max_genes_per_batch,
        gene_subset=load_gene_subset(args.gene_subset_file),
        no_pandas=args.no_pandas,
    )
