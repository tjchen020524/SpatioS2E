#!/usr/bin/env python3
"""Stratify primary held-out-gene effects by nearest training-vector similarity."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from experiments.heldout_gene_second_stage.build_gene_mean_baselines import safe_corr
from experiments.multicohort_geneheldout_decima_clean_split.clean_split_config import (
    CLEAN_ROOT,
    COHORTS,
    ROOT,
    SEEDS,
)


BASE = ROOT / "experiments/heldout_gene_second_stage"
OUT = BASE / "similarity_stratified"
SIMILARITY = BASE / "gene_partitions/nearest_training_similarity_per_gene.tsv.gz"


def subset_effect(
    cohort: str,
    seed: int,
    stratum: str,
    table: pd.DataFrame,
) -> dict[str, object]:
    eligible = table["gene_pcc_eligible_pretrained"].astype(bool).to_numpy()
    weight = table["n_observations_pretrained"].to_numpy(dtype=float)
    weight /= weight.sum()
    observed = table["observed_mean_pretrained"].to_numpy(dtype=float)
    pretrained_mean = table["predicted_mean_pretrained"].to_numpy(dtype=float)
    random_mean = table["predicted_mean_random"].to_numpy(dtype=float)
    return {
        "cohort": cohort,
        "seed": seed,
        "similarity_stratum": stratum,
        "n_genes": len(table),
        "similarity_min": float(table["nearest_training_cosine"].min()),
        "similarity_max": float(table["nearest_training_cosine"].max()),
        "similarity_mean": float(table["nearest_training_cosine"].mean()),
        "abundance_pcc_gain": safe_corr(pretrained_mean, observed)
        - safe_corr(random_mean, observed),
        "mean_gene_pcc_gain": float(
            np.mean(
                table.loc[eligible, "gene_pcc_pretrained"].to_numpy(dtype=float)
                - table.loc[eligible, "gene_pcc_random"].to_numpy(dtype=float)
            )
        ),
        "abundance_mse_reduction": float(
            np.sum(
                weight
                * (
                    np.square(table["mean_error_random"].to_numpy(dtype=float))
                    - np.square(table["mean_error_pretrained"].to_numpy(dtype=float))
                )
            )
        ),
        "centered_mse_reduction": float(
            np.sum(
                weight
                * (
                    table["centered_mse_random"].to_numpy(dtype=float)
                    - table["centered_mse_pretrained"].to_numpy(dtype=float)
                )
            )
        ),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    similarity = pd.read_csv(SIMILARITY, sep="\t")
    similarity = similarity.loc[
        similarity["partition"] == "primary_expression_seed42",
        ["gene_id", "nearest_training_cosine"],
    ]
    rows: list[dict[str, object]] = []
    for cohort in COHORTS:
        for seed in SEEDS:
            pretrained = pd.read_csv(
                CLEAN_ROOT
                / "components/runs"
                / cohort
                / f"seed_{seed}/decima/gene_components.tsv.gz",
                sep="\t",
            )
            random = pd.read_csv(
                CLEAN_ROOT
                / "components/runs"
                / cohort
                / f"seed_{seed}/random/gene_components.tsv.gz",
                sep="\t",
            )
            columns = [
                "gene_id",
                "n_observations",
                "observed_mean",
                "predicted_mean",
                "mean_error",
                "centered_mse",
                "gene_pcc",
                "gene_pcc_eligible",
            ]
            merged = pretrained[columns].merge(
                random[columns], on="gene_id", suffixes=("_pretrained", "_random")
            ).merge(similarity, on="gene_id", how="left", validate="one_to_one")
            if merged["nearest_training_cosine"].isna().any():
                raise ValueError(f"Missing similarity for {cohort}/{seed}")
            merged["similarity_quartile"] = pd.qcut(
                merged["nearest_training_cosine"],
                q=4,
                labels=["Q1 least similar", "Q2", "Q3", "Q4 most similar"],
                duplicates="drop",
            )
            rows.append(subset_effect(cohort, seed, "all", merged))
            for stratum, group in merged.groupby("similarity_quartile", observed=True):
                rows.append(subset_effect(cohort, seed, str(stratum), group.copy()))

    per_run = pd.DataFrame(rows)
    summary = (
        per_run.groupby(["cohort", "similarity_stratum"], as_index=False)
        .agg(
            n_genes=("n_genes", "first"),
            similarity_mean=("similarity_mean", "mean"),
            abundance_pcc_gain=("abundance_pcc_gain", "mean"),
            mean_gene_pcc_gain=("mean_gene_pcc_gain", "mean"),
            abundance_mse_reduction=("abundance_mse_reduction", "mean"),
            centered_mse_reduction=("centered_mse_reduction", "mean"),
        )
    )
    per_run.to_csv(OUT / "effects_per_run.tsv", sep="\t", index=False)
    summary.to_csv(OUT / "effects_summary.tsv", sep="\t", index=False)
    quartiles = summary.loc[summary["similarity_stratum"] != "all"]
    manifest = {
        "status": "PASS",
        "checks": {
            "twelve_run_groups": per_run.groupby(["cohort", "seed"]).ngroups == 12,
            "four_quartiles_each_run": bool(
                per_run.loc[per_run["similarity_stratum"] != "all"]
                .groupby(["cohort", "seed"])["similarity_stratum"]
                .nunique()
                .eq(4)
                .all()
            ),
            "abundance_gain_positive_all_quartiles": bool(
                (quartiles["abundance_pcc_gain"] > 0).all()
            ),
        },
        "interpretive_boundary": "Similarity strata are post-hoc diagnostics within the primary downstream split; they do not replace cluster- or chromosome-blocked retraining.",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
