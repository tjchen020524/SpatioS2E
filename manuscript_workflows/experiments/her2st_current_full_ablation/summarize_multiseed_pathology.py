#!/usr/bin/env python3
"""Aggregate HER2ST pathology-program metrics across three model seeds."""

from __future__ import annotations
from workflow_paths import DATA_ROOT

import json
from pathlib import Path

import pandas as pd


ROOT = DATA_ROOT
EXP = ROOT / "experiments/her2st_current_full_ablation"
SEEDS = (42, 123, 456)
VARIANTS = (
    "uni2h",
    "uni2h_gnn",
    "decima_prior_only",
    "decima_film_no_graph",
    "gene_mean_uni2h_residual",
    "gene_mean_uni2h_gnn",
    "decima_film_gnn",
    "decima_concat_gnn",
    "decima_crossattn_gnn",
)


def main() -> None:
    frames = []
    missing = []
    for seed in SEEDS:
        for variant in VARIANTS:
            if seed == 42:
                path = EXP / "variants" / variant / "pathology/pathology_program_metrics.csv"
            else:
                path = (
                    EXP
                    / "multiseed"
                    / f"seed_{seed}"
                    / "variants"
                    / variant
                    / "pathology/pathology_program_metrics.csv"
                )
            if not path.exists():
                missing.append(str(path))
                continue
            frame = pd.read_csv(path)
            frame["seed"] = seed
            frames.append(frame)

    output = EXP / "multiseed/results"
    output.mkdir(parents=True, exist_ok=True)
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    combined.to_csv(output / "pathology_program_per_seed.csv", index=False)
    if not combined.empty:
        metrics = [
            "predicted_program_auc",
            "predicted_vs_observed_program_pcc",
            "predicted_tumor_contrast",
        ]
        grouped = combined.groupby(["variant", "sample"], sort=False)[metrics]
        aggregate = pd.concat(
            [
                grouped.size().rename("n_seeds"),
                grouped.mean().add_suffix("_mean"),
                grouped.std(ddof=1).add_suffix("_sd"),
            ],
            axis=1,
        ).reset_index()
        aggregate.to_csv(output / "pathology_program_mean_sd.csv", index=False)
    summary = {
        "n_expected_runs": len(SEEDS) * len(VARIANTS),
        "n_completed_runs": len(frames),
        "missing": missing,
        "complete": not missing,
    }
    (output / "pathology_run_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
