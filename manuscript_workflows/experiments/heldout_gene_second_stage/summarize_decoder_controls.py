#!/usr/bin/env python3
"""Summarize shuffled-vector, fitted-target oracle and high-signal controls."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.multicohort_geneheldout_decima_clean_split.clean_split_config import (
    CLEAN_ROOT,
    COHORTS,
    ROOT,
    SEEDS,
)


BASE = ROOT / "experiments/heldout_gene_second_stage/decoder_controls"
OUT = BASE / "summary"
CONDITIONS = ("shuffled_pretrained", "fitted_target_oracle")
ENDPOINTS = (
    "full_matrix_pcc",
    "overall_mse",
    "abundance_pcc",
    "abundance_rmse",
    "mean_gene_pcc",
    "centered_full_matrix_pcc",
    "centered_rmse",
)


def load_summary(root: Path) -> dict[str, object]:
    if not root.exists():
        raise FileNotFoundError(root)
    return json.loads(root.read_text())


def effect(candidate: float, control: float, endpoint: str) -> float:
    return control - candidate if endpoint.endswith("mse") or endpoint.endswith("rmse") else candidate - control


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    absolute_rows: list[dict[str, object]] = []
    effect_rows: list[dict[str, object]] = []
    high_signal_rows: list[dict[str, object]] = []

    for cohort in COHORTS:
        for seed in SEEDS:
            primary_root = CLEAN_ROOT / "components/runs" / cohort / f"seed_{seed}"
            primary_summary = load_summary(primary_root / "decima/summary.json")
            random_summary = load_summary(primary_root / "random/summary.json")
            condition_summary = {
                condition: load_summary(
                    BASE
                    / condition
                    / "components/runs"
                    / cohort
                    / f"seed_{seed}/decima/summary.json"
                )
                for condition in CONDITIONS
            }
            summaries = {
                "pretrained": primary_summary,
                "random": random_summary,
                **condition_summary,
            }
            for condition, summary in summaries.items():
                absolute_rows.append(
                    {
                        "cohort": cohort,
                        "seed": seed,
                        "condition": condition,
                        **{endpoint: float(summary[endpoint]) for endpoint in ENDPOINTS},
                        "n_heldout_genes": int(summary["n_heldout_genes"]),
                        "n_eligible_gene_pcc": int(summary["n_eligible_gene_pcc"]),
                    }
                )
            contrasts = {
                "pretrained_vs_shuffled": (primary_summary, condition_summary["shuffled_pretrained"]),
                "shuffled_vs_random": (condition_summary["shuffled_pretrained"], random_summary),
                "pretrained_vs_random": (primary_summary, random_summary),
                "oracle_vs_heldout_pretrained": (condition_summary["fitted_target_oracle"], primary_summary),
            }
            for contrast, (candidate, control) in contrasts.items():
                for endpoint in ENDPOINTS:
                    effect_rows.append(
                        {
                            "cohort": cohort,
                            "seed": seed,
                            "contrast": contrast,
                            "endpoint": endpoint,
                            "effect": effect(float(candidate[endpoint]), float(control[endpoint]), endpoint),
                        }
                    )

            gene_tables = {
                "pretrained": pd.read_csv(primary_root / "decima/gene_components.tsv.gz", sep="\t"),
                "random": pd.read_csv(primary_root / "random/gene_components.tsv.gz", sep="\t"),
                **{
                    condition: pd.read_csv(
                        BASE
                        / condition
                        / "components/runs"
                        / cohort
                        / f"seed_{seed}/decima/gene_components.tsv.gz",
                        sep="\t",
                    )
                    for condition in CONDITIONS
                },
            }
            reference = gene_tables["pretrained"]
            observed_std = reference["observed_std"].to_numpy(dtype=float)
            eligible = reference["gene_pcc_eligible"].astype(bool).to_numpy()
            thresholds = {
                "all_eligible": -np.inf,
                "top_quartile_observed_spatial_sd": float(np.quantile(observed_std[eligible], 0.75)),
                "top_decile_observed_spatial_sd": float(np.quantile(observed_std[eligible], 0.90)),
            }
            for subset, threshold in thresholds.items():
                subset_mask = eligible & (observed_std >= threshold)
                for condition, table in gene_tables.items():
                    if table["gene_id"].astype(str).tolist() != reference["gene_id"].astype(str).tolist():
                        raise ValueError(f"Gene order mismatch for {cohort}/{seed}/{condition}")
                    values = table.loc[subset_mask, "gene_pcc"].to_numpy(dtype=float)
                    high_signal_rows.append(
                        {
                            "cohort": cohort,
                            "seed": seed,
                            "subset": subset,
                            "observed_std_threshold": threshold,
                            "condition": condition,
                            "n_genes": int(subset_mask.sum()),
                            "mean_gene_pcc": float(np.mean(values)),
                            "median_gene_pcc": float(np.median(values)),
                        }
                    )

    absolute = pd.DataFrame(absolute_rows)
    effects = pd.DataFrame(effect_rows)
    effect_summary = (
        effects.groupby(["cohort", "contrast", "endpoint"], as_index=False)
        .agg(
            mean_effect=("effect", "mean"),
            min_effect=("effect", "min"),
            max_effect=("effect", "max"),
            n_runs=("effect", "size"),
        )
    )
    high_signal = pd.DataFrame(high_signal_rows)
    high_pivot = high_signal.pivot_table(
        index=["cohort", "seed", "subset", "observed_std_threshold", "n_genes"],
        columns="condition",
        values="mean_gene_pcc",
    ).reset_index()
    high_pivot["pretrained_minus_random"] = high_pivot["pretrained"] - high_pivot["random"]
    high_pivot["pretrained_minus_shuffled"] = high_pivot["pretrained"] - high_pivot["shuffled_pretrained"]
    high_pivot["oracle_minus_heldout_pretrained"] = high_pivot["fitted_target_oracle"] - high_pivot["pretrained"]
    high_summary = (
        high_pivot.groupby(["cohort", "subset"], as_index=False)
        .agg(
            n_genes_mean=("n_genes", "mean"),
            pretrained_mean_gene_pcc=("pretrained", "mean"),
            random_mean_gene_pcc=("random", "mean"),
            shuffled_mean_gene_pcc=("shuffled_pretrained", "mean"),
            oracle_mean_gene_pcc=("fitted_target_oracle", "mean"),
            pretrained_minus_random=("pretrained_minus_random", "mean"),
            pretrained_minus_shuffled=("pretrained_minus_shuffled", "mean"),
            oracle_minus_heldout_pretrained=("oracle_minus_heldout_pretrained", "mean"),
        )
    )

    absolute.to_csv(OUT / "absolute_per_run.tsv", sep="\t", index=False)
    effects.to_csv(OUT / "paired_effects_per_run.tsv", sep="\t", index=False)
    effect_summary.to_csv(OUT / "paired_effects_summary.tsv", sep="\t", index=False)
    high_signal.to_csv(OUT / "observed_spatial_signal_absolute.tsv", sep="\t", index=False)
    high_pivot.to_csv(OUT / "observed_spatial_signal_effects_per_run.tsv", sep="\t", index=False)
    high_summary.to_csv(OUT / "observed_spatial_signal_summary.tsv", sep="\t", index=False)

    manifest = {
        "status": "PASS",
        "n_absolute_rows": len(absolute),
        "n_effect_rows": len(effects),
        "n_high_signal_rows": len(high_signal),
        "checks": {
            "absolute_complete": len(absolute) == 4 * 3 * 4,
            "effect_complete": len(effects) == 4 * 3 * 4 * len(ENDPOINTS),
            "common_gene_denominator": bool(
                absolute.groupby(["cohort", "seed"])["n_eligible_gene_pcc"].nunique().eq(1).all()
            ),
        },
        "interpretive_boundary": "The oracle is a fitted-target capacity control, not a deployable held-out-gene predictor; observed-variance subsets are defined from outcomes and estimate capacity rather than prospective performance.",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    print("\nKey paired effects")
    print(
        effect_summary.loc[
            effect_summary["endpoint"].isin(
                ["full_matrix_pcc", "abundance_pcc", "mean_gene_pcc", "centered_rmse"]
            )
        ].to_string(index=False)
    )
    print("\nHigh-signal subsets")
    print(high_summary.to_string(index=False))


if __name__ == "__main__":
    main()
