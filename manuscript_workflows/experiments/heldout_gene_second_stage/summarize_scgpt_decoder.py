#!/usr/bin/env python3
"""Summarize the matched static-scGPT held-out-gene decoder experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.multicohort_geneheldout_decima_clean_split.clean_split_config import (
    COHORTS,
    ROOT,
    SEEDS,
)


BASE = ROOT / "experiments/heldout_gene_second_stage/scgpt_decoder"
OUT = BASE / "summary"
CHECKPOINT_ROOTS = {
    "brain": ROOT / "experiments/heldout_gene_second_stage/scgpt_decoder",
    "whole_human": ROOT
    / "experiments/heldout_gene_second_stage/scgpt_whole_human_decoder",
}
CONDITIONS = (
    "pretrained_gene_tokens",
    "random_gene_vectors",
    "constant_gene_vector",
    "identity_shuffled_gene_tokens",
)
ENDPOINTS = (
    "full_matrix_pcc",
    "overall_mse",
    "abundance_pcc",
    "abundance_rmse",
    "abundance_mse",
    "mean_gene_pcc",
    "centered_full_matrix_pcc",
    "centered_rmse",
    "centered_mse",
)


def improvement(candidate: float, control: float, endpoint: str) -> float:
    """Orient every contrast so that positive values favour pretrained vectors."""
    if endpoint.endswith("mse") or endpoint.endswith("rmse"):
        return control - candidate
    return candidate - control


def main() -> None:
    global BASE, OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", choices=CHECKPOINT_ROOTS, default="brain")
    args = parser.parse_args()
    BASE = CHECKPOINT_ROOTS[args.checkpoint]
    OUT = BASE / "summary"
    OUT.mkdir(parents=True, exist_ok=True)
    absolute_rows: list[dict[str, object]] = []
    for condition in CONDITIONS:
        for cohort in COHORTS:
            for seed in SEEDS:
                path = (
                    BASE
                    / condition
                    / "components/runs"
                    / cohort
                    / f"seed_{seed}/decima/summary.json"
                )
                if not path.exists():
                    raise FileNotFoundError(path)
                summary = json.loads(path.read_text())
                endpoint_values = {
                    endpoint: (
                        float("nan")
                        if condition == "constant_gene_vector"
                        and endpoint == "abundance_pcc"
                        else float(summary[endpoint])
                    )
                    for endpoint in ENDPOINTS
                }
                absolute_rows.append(
                    {
                        "cohort": cohort,
                        "seed": seed,
                        "condition": condition,
                        **endpoint_values,
                        "n_heldout_genes": int(summary["n_heldout_genes"]),
                        "n_eligible_gene_pcc": int(summary["n_eligible_gene_pcc"]),
                    }
                )
    absolute = pd.DataFrame(absolute_rows)

    effects: list[dict[str, object]] = []
    contrasts = {
        "pretrained_vs_random": "random_gene_vectors",
        "pretrained_vs_constant": "constant_gene_vector",
        "pretrained_vs_identity_shuffled": "identity_shuffled_gene_tokens",
    }
    for cohort in COHORTS:
        for seed in SEEDS:
            subset = absolute.loc[
                (absolute["cohort"] == cohort) & (absolute["seed"] == seed)
            ].set_index("condition")
            candidate = subset.loc["pretrained_gene_tokens"]
            for contrast, control_name in contrasts.items():
                control = subset.loc[control_name]
                for endpoint in ENDPOINTS:
                    if endpoint == "abundance_pcc" and control_name == "constant_gene_vector":
                        value = float("nan")
                    else:
                        value = improvement(
                            float(candidate[endpoint]),
                            float(control[endpoint]),
                            endpoint,
                        )
                    effects.append(
                        {
                            "cohort": cohort,
                            "seed": seed,
                            "contrast": contrast,
                            "endpoint": endpoint,
                            "effect": value,
                        }
                    )
    effect_table = pd.DataFrame(effects)
    effect_summary = (
        effect_table.groupby(["cohort", "contrast", "endpoint"], as_index=False)
        .agg(
            mean_effect=("effect", "mean"),
            min_effect=("effect", "min"),
            max_effect=("effect", "max"),
            n_runs=("effect", "size"),
        )
    )
    absolute_summary = (
        absolute.groupby(["cohort", "condition"], as_index=False)
        .agg(
            **{
                f"{endpoint}_mean": (endpoint, "mean")
                for endpoint in ENDPOINTS
            },
            n_runs=("seed", "size"),
            n_heldout_genes=("n_heldout_genes", "first"),
            n_eligible_gene_pcc=("n_eligible_gene_pcc", "first"),
        )
    )

    absolute.to_csv(OUT / "absolute_per_run.tsv", sep="\t", index=False)
    absolute_summary.to_csv(OUT / "absolute_summary.tsv", sep="\t", index=False)
    effect_table.to_csv(OUT / "paired_effects_per_run.tsv", sep="\t", index=False)
    effect_summary.to_csv(OUT / "paired_effects_summary.tsv", sep="\t", index=False)

    denominator_ok = bool(
        absolute.groupby(["cohort", "seed"])[
            ["n_heldout_genes", "n_eligible_gene_pcc"]
        ]
        .nunique()
        .eq(1)
        .all()
        .all()
    )
    decomposition_error = np.abs(
        absolute["overall_mse"]
        - absolute["abundance_mse"]
        - absolute["centered_mse"]
    )
    manifest = {
        "status": "PASS",
        "checkpoint_scope": args.checkpoint,
        "design": "Matched factorized dot-product held-out-gene decoder using static scGPT gene-token, dimension-matched random, constant or train-time identity-shuffled vectors",
        "n_absolute_rows": int(len(absolute)),
        "n_effect_rows": int(len(effect_table)),
        "checks": {
            "complete": len(absolute) == len(CONDITIONS) * len(COHORTS) * len(SEEDS),
            "common_gene_denominator_within_cohort_and_run": denominator_ok,
            "maximum_absolute_mse_decomposition_error": float(decomposition_error.max()),
            "exact_mse_decomposition": bool(decomposition_error.max() < 1.0e-9),
        },
        "interpretive_boundary": "This assay tests a second frozen gene representation in the same decoder system; it does not establish behaviour in every gene-conditioned architecture.",
        "gene_mean_pcc_policy": "Undefined (NA) for the constant-vector condition because predicted gene means have no across-gene variance.",
    }
    if not all(manifest["checks"].values()):
        manifest["status"] = "FAIL"
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    print(
        effect_summary.loc[
            effect_summary["endpoint"].isin(
                ["full_matrix_pcc", "abundance_pcc", "mean_gene_pcc", "centered_rmse"]
            )
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
