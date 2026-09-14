#!/usr/bin/env python3
"""Harmonize external-method metrics with the strict benchmark definitions."""

from __future__ import annotations
from workflow_paths import DATA_ROOT

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = DATA_ROOT
EXP = DATA_ROOT / 'experiments/hippocampus_donor_disjoint_external_benchmark'
STRICT = ROOT / "experiments/hippocampus_current_full_ablation"
METHODS = ("stnet", "hist2st", "bleep", "stpath")
SEEDS = (42, 123, 456)
HVG_K = (50, 100, 200, 500, 1000, 2000)


def metric_paths(method: str, seed: int) -> tuple[Path, Path]:
    results = EXP / f"runs/{method}/seed_{seed}/results"
    return results / "test_overall.json", results / "test_gene_metrics.tsv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--output-prefix", default="external_methods")
    args = parser.parse_args()

    reference = pd.read_csv(
        STRICT / "multiseed/seed_42/variants/uni2h/results/test_gene_metrics.tsv",
        sep="\t",
        usecols=["gene_id", "eligible_true_variance"],
    )
    reference["gene_id"] = reference["gene_id"].astype(str)
    rank = pd.read_csv(STRICT / "artifacts/train_gene_stats.tsv", sep="\t", usecols=["rank", "gene_id"])
    rank["gene_id"] = rank["gene_id"].astype(str)
    rank_map = dict(zip(rank["gene_id"], rank["rank"]))

    rows = []
    hvg_rows = []
    for method in METHODS:
        for seed in SEEDS:
            overall_path, gene_path = metric_paths(method, seed)
            if not overall_path.exists() or not gene_path.exists():
                if args.allow_incomplete:
                    continue
                raise FileNotFoundError(f"Incomplete {method} seed {seed}: {overall_path.parent}")
            overall = json.loads(overall_path.read_text())
            genes = pd.read_csv(gene_path, sep="\t")
            genes["gene_id"] = genes["gene_id"].astype(str).str.split(".", regex=False).str[0]
            genes = reference.merge(genes[["gene_id", "corr"]], on="gene_id", how="left", validate="one_to_one")
            eligible = genes["eligible_true_variance"].astype(bool).to_numpy()
            corr = pd.to_numeric(genes["corr"], errors="coerce").to_numpy(dtype=float)
            primary = np.where(eligible, np.where(np.isfinite(corr), corr, 0.0), np.nan)
            finite = eligible & np.isfinite(corr)

            row = {
                "method": method,
                "seed": seed,
                "mse": float(overall["mse"]),
                "pooled_pcc": float(overall["corr"]),
                "gene_pcc_mean_primary": float(np.nanmean(primary)),
                "gene_pcc_median_primary": float(np.nanmedian(primary)),
                "gene_pcc_mean_finite": float(np.mean(corr[finite])) if finite.any() else np.nan,
                "n_genes_eligible": int(eligible.sum()),
                "n_gene_pcc_finite": int(finite.sum()),
                "gene_pcc_coverage": float(finite.sum() / eligible.sum()),
            }
            for k in HVG_K:
                panel = genes["gene_id"].map(rank_map).le(k).to_numpy() & eligible
                panel_finite = panel & np.isfinite(corr)
                panel_primary = np.where(panel, np.where(np.isfinite(corr), corr, 0.0), np.nan)
                row[f"top{k}_pcc_primary"] = float(np.nanmean(panel_primary))
                row[f"top{k}_coverage"] = float(panel_finite.sum() / panel.sum())
                hvg_rows.append(
                    {
                        "method": method,
                        "seed": seed,
                        "k": k,
                        "n_panel": int(panel.sum()),
                        "n_finite": int(panel_finite.sum()),
                        "coverage": float(panel_finite.sum() / panel.sum()),
                        "mean_pcc_primary": float(np.nanmean(panel_primary)),
                        "mean_pcc_finite": float(np.mean(corr[panel_finite])) if panel_finite.any() else np.nan,
                    }
                )
            rows.append(row)

    if not rows:
        raise RuntimeError("No completed external-method results were found")
    per_seed = pd.DataFrame(rows)
    summary_dir = EXP / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_prefix
    per_seed.to_csv(summary_dir / f"{prefix}_per_seed.tsv", sep="\t", index=False)
    pd.DataFrame(hvg_rows).to_csv(summary_dir / f"{prefix}_hvg_per_seed.tsv", sep="\t", index=False)

    metrics = [column for column in per_seed.columns if column not in {"method", "seed"}]
    aggregates = []
    for method, frame in per_seed.groupby("method", sort=False):
        out = {"method": method, "n_seeds": int(len(frame))}
        for metric in metrics:
            out[f"{metric}_mean"] = float(frame[metric].mean())
            out[f"{metric}_sd"] = float(frame[metric].std(ddof=1))
        aggregates.append(out)
    aggregate = pd.DataFrame(aggregates)
    aggregate.to_csv(summary_dir / f"{prefix}_mean_sd.tsv", sep="\t", index=False)
    print(aggregate.to_string(index=False))


if __name__ == "__main__":
    main()
