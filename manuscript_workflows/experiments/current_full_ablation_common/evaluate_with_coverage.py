#!/usr/bin/env python3
"""Evaluate a current-model checkpoint with explicit gene-PCC coverage rules."""

from __future__ import annotations
from workflow_paths import DATA_ROOT

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch


ROOT = DATA_ROOT
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.append(str(path))

from experiments.sample_split_fullgenes_v4_structured_loss.eval_sample_split_v4 import (  # noqa: E402
    RunningStats,
    _resolve_max_genes,
    build_dataset,
    forward_model,
    load_config,
    to_dense,
    update_gene_stats,
)
from models.stage0_reg_model_factory import build_stage0_reg_model  # noqa: E402


def moments(stats: RunningStats) -> tuple[float, float, float, float]:
    if stats.count <= 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    mean_pred = stats.sum_pred / stats.count
    mean_true = stats.sum_true / stats.count
    var_pred = max(stats.sum_pred2 / stats.count - mean_pred * mean_pred, 0.0)
    var_true = max(stats.sum_true2 / stats.count - mean_true * mean_true, 0.0)
    return mean_pred, mean_true, float(np.sqrt(var_pred)), float(np.sqrt(var_true))


def correlation_with_thresholds(
    stats: RunningStats, true_std_threshold: float, pred_std_threshold: float
) -> tuple[float, float, bool, bool]:
    _, _, pred_std, true_std = moments(stats)
    eligible = bool(np.isfinite(true_std) and true_std > true_std_threshold)
    pred_variable = bool(np.isfinite(pred_std) and pred_std > pred_std_threshold)
    if not eligible:
        return float("nan"), float("nan"), eligible, pred_variable
    if not pred_variable:
        return 0.0, float("nan"), eligible, pred_variable
    corr = stats.corr()
    if not np.isfinite(corr):
        return 0.0, float("nan"), eligible, pred_variable
    corr = float(np.clip(corr, -1.0, 1.0))
    return corr, corr, eligible, pred_variable


def load_hvg_order(path: Path) -> list[str]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return [row["gene_id"] for row in reader]


def summarize_hvg(gene_df: pd.DataFrame, ranking: Sequence[str]) -> list[dict]:
    indexed = gene_df.set_index("gene_id")
    rows = []
    for k in (50, 100, 200, 500, 1000, 2000):
        panel = [gene_id for gene_id in ranking[:k] if gene_id in indexed.index]
        frame = indexed.loc[panel] if panel else indexed.iloc[0:0]
        eligible = frame[frame["eligible_true_variance"]]
        finite = eligible[np.isfinite(eligible["corr_finite"])]
        rows.append(
            {
                "k": k,
                "n_panel": len(panel),
                "n_eligible": int(len(eligible)),
                "n_finite": int(len(finite)),
                "coverage": float(len(finite) / len(eligible)) if len(eligible) else float("nan"),
                "mean_pcc_primary": float(eligible["corr_primary"].mean()) if len(eligible) else float("nan"),
                "mean_pcc_finite": float(finite["corr_finite"].mean()) if len(finite) else float("nan"),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--save-dir", type=Path, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    eval_cfg = cfg.get("eval", {})
    true_threshold = float(eval_cfg.get("true_std_threshold", 1.0e-6))
    pred_threshold = float(eval_cfg.get("pred_std_threshold", 1.0e-8))
    max_genes = _resolve_max_genes(cfg, None)
    samples = cfg.get("split", {}).get(args.split)
    ds = build_dataset(cfg, split=args.split, samples=samples)
    if len(ds) == 0:
        raise RuntimeError(f"No samples available for split {args.split}")

    first = ds[0]
    input_dim = int(first["x"].shape[1])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_stage0_reg_model(cfg, input_dim=input_dim).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()
    no_graph = bool(state.get("no_graph", False) or cfg.get("model", {}).get("no_graph", False))

    overall = RunningStats()
    sample_stats: dict[str, RunningStats] = {}
    gene_stats: dict[str, RunningStats] = {}
    with torch.no_grad():
        for sample_idx in range(len(ds)):
            item = ds[sample_idx]
            sample = str(item["sample"])
            x = item["x"].to(device)
            edge_index = item["edge_index"].to(device)
            edge_weight = item["edge_weight"].to(device)
            if no_graph:
                edge_weight = torch.zeros_like(edge_weight)
            true = to_dense(item["y"]).to(device)
            gene_ids = list(item["gene_ids"])
            sample_stats.setdefault(sample, RunningStats())
            chunk_size = max_genes or len(gene_ids)
            for start in range(0, len(gene_ids), chunk_size):
                end = min(start + chunk_size, len(gene_ids))
                chunk_genes = gene_ids[start:end]
                true_chunk = true[:, start:end]
                out = forward_model(
                    model, x, edge_index, edge_weight, chunk_genes, sample_name=sample
                )
                pred_cpu = out["log_mu"].detach().cpu()
                true_cpu = true_chunk.detach().cpu()
                overall.update(pred_cpu.reshape(-1), true_cpu.reshape(-1))
                sample_stats[sample].update(pred_cpu.reshape(-1), true_cpu.reshape(-1))
                update_gene_stats(gene_stats, chunk_genes, pred_cpu, true_cpu)
            print(f"[eval] {sample_idx + 1}/{len(ds)} {sample}", flush=True)

    gene_rows = []
    for gene_id, stats in gene_stats.items():
        corr_primary, corr_finite, eligible, pred_variable = correlation_with_thresholds(
            stats, true_std_threshold=true_threshold, pred_std_threshold=pred_threshold
        )
        _, _, pred_std, true_std = moments(stats)
        gene_rows.append(
            {
                "gene_id": gene_id,
                "n_observations": stats.count,
                "mse": stats.mse(),
                "true_std": true_std,
                "pred_std": pred_std,
                "eligible_true_variance": eligible,
                "pred_has_variance": pred_variable,
                "corr_primary": corr_primary,
                "corr_finite": corr_finite,
            }
        )
    gene_df = pd.DataFrame(gene_rows)
    eligible_df = gene_df[gene_df["eligible_true_variance"]]
    finite_df = eligible_df[np.isfinite(eligible_df["corr_finite"])]

    sample_rows = [
        {"sample": sample, "mse": stats.mse(), "corr": stats.corr()}
        for sample, stats in sample_stats.items()
    ]
    summary = {
        "split": args.split,
        "mse": overall.mse(),
        "corr": overall.corr(),
        "gene_corr_mean": float(eligible_df["corr_primary"].mean()),
        "gene_corr_median": float(eligible_df["corr_primary"].median()),
        "gene_corr_mean_primary": float(eligible_df["corr_primary"].mean()),
        "gene_corr_median_primary": float(eligible_df["corr_primary"].median()),
        "gene_corr_mean_finite": float(finite_df["corr_finite"].mean()) if len(finite_df) else float("nan"),
        "gene_corr_median_finite": float(finite_df["corr_finite"].median()) if len(finite_df) else float("nan"),
        "n_genes_total": int(len(gene_df)),
        "n_genes_eligible": int(len(eligible_df)),
        "n_gene_corr_finite": int(len(finite_df)),
        "gene_corr_coverage": float(len(finite_df) / len(eligible_df)) if len(eligible_df) else float("nan"),
        "true_std_threshold": true_threshold,
        "pred_std_threshold": pred_threshold,
        "no_graph": no_graph,
    }

    ranking_path = Path(eval_cfg["hvg_ranking_file"])
    if not ranking_path.is_absolute():
        ranking_path = ROOT / ranking_path
    hvg_rows = summarize_hvg(gene_df, load_hvg_order(ranking_path))
    for row in hvg_rows:
        summary[f"top{row['k']}_pcc_primary"] = row["mean_pcc_primary"]
        summary[f"top{row['k']}_coverage"] = row["coverage"]

    args.save_dir.mkdir(parents=True, exist_ok=True)
    gene_df.to_csv(args.save_dir / f"{args.split}_gene_metrics.tsv", sep="\t", index=False)
    pd.DataFrame(sample_rows).to_csv(
        args.save_dir / f"{args.split}_sample_metrics.tsv", sep="\t", index=False
    )
    pd.DataFrame(hvg_rows).to_csv(args.save_dir / f"{args.split}_hvg_metrics.csv", index=False)
    (args.save_dir / f"{args.split}_overall.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
