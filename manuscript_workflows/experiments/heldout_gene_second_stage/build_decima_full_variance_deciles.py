#!/usr/bin/env python3
"""Describe Decima spatial effects across training-variance deciles, all targets."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.append(str(REPOSITORY_ROOT))

from experiments.multicohort_geneheldout_decima_clean_split.clean_split_config import (
    CLEAN_ROOT,
    COHORTS,
    ROOT,
    SEEDS,
)


OUT = ROOT / "experiments/heldout_gene_second_stage/decima_full_variance_deciles"
PARTITION = ROOT / "experiments/heldout_gene_second_stage/partition_balance/primary_partition_per_gene.tsv.gz"
CONDITIONS = ("decima", "random", "constant")
CONTRASTS = {"pretrained_vs_random": "random", "pretrained_vs_constant": "constant"}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    annotations = pd.read_csv(PARTITION, sep="\t")
    annotations = annotations.loc[
        annotations["assignment"].eq("heldout"),
        ["cohort", "gene_id", "training_spot_std"],
    ].copy()
    annotations["training_variance_decile"] = (
        annotations.groupby("cohort")["training_spot_std"]
        .transform(lambda values: pd.qcut(values.rank(method="first"), 10, labels=False) + 1)
        .astype(int)
    )
    rows: list[pd.DataFrame] = []
    for condition in CONDITIONS:
        for cohort in COHORTS:
            cohort_annotations = annotations.loc[annotations["cohort"].eq(cohort)]
            for seed in SEEDS:
                pooled = pd.read_csv(
                    CLEAN_ROOT / "components/runs" / cohort / f"seed_{seed}/{condition}/gene_components.tsv.gz",
                    sep="\t",
                )[["gene_id", "gene_pcc", "gene_pcc_eligible"]]
                section = pd.read_csv(
                    CLEAN_ROOT / "section_centered/runs" / cohort / f"seed_{seed}/{condition}/concat_gene_pcc.tsv.gz",
                    sep="\t",
                )[["gene_id", "section_centered_concat_pcc", "gene_pcc_eligible"]]
                frame = pooled.merge(section, on="gene_id", suffixes=("_pooled", "_section"), validate="one_to_one").merge(
                    cohort_annotations, on="gene_id", validate="one_to_one"
                )
                frame["condition"] = condition
                frame["seed"] = seed
                rows.append(frame)
    absolute = pd.concat(rows, ignore_index=True)
    effect_rows: list[pd.DataFrame] = []
    for (cohort, seed), group in absolute.groupby(["cohort", "seed"], sort=False):
        for endpoint in ("gene_pcc", "section_centered_concat_pcc"):
            wide = group.pivot(index="gene_id", columns="condition", values=endpoint)
            annotation = annotations.loc[annotations["cohort"].eq(cohort)].set_index("gene_id")
            for contrast, control in CONTRASTS.items():
                frame = pd.DataFrame(
                    {"effect": wide["decima"] - wide[control]}, index=wide.index
                ).join(annotation).reset_index()
                frame["cohort"] = cohort
                frame["seed"] = seed
                frame["contrast"] = contrast
                frame["endpoint"] = (
                    "pooled_within_gene_pcc"
                    if endpoint == "gene_pcc"
                    else "section_centered_within_gene_pcc"
                )
                effect_rows.append(frame)
    per_run = pd.concat(effect_rows, ignore_index=True)
    technical_mean = (
        per_run.groupby(
            ["cohort", "gene_id", "contrast", "endpoint", "training_variance_decile"],
            as_index=False,
        )["effect"]
        .mean()
        .rename(columns={"effect": "technical_mean_effect"})
    )
    summary_rows: list[dict[str, object]] = []
    for keys, group in technical_mean.groupby(
        ["cohort", "contrast", "endpoint", "training_variance_decile"], sort=False
    ):
        values = group["technical_mean_effect"].dropna().to_numpy(float)
        summary_rows.append(
            {
                "cohort": keys[0],
                "contrast": keys[1],
                "endpoint": keys[2],
                "training_variance_decile": keys[3],
                "n_genes": len(values),
                "mean_effect": float(values.mean()),
                "median_effect": float(np.median(values)),
                "q25_effect": float(np.quantile(values, 0.25)),
                "q75_effect": float(np.quantile(values, 0.75)),
                "fraction_positive": float((values > 0).mean()),
            }
        )
    summary = pd.DataFrame(summary_rows)
    technical_mean.to_csv(OUT / "gene_effects_technical_mean.tsv.gz", sep="\t", index=False)
    summary.to_csv(OUT / "variance_decile_summary.tsv", sep="\t", index=False)
    manifest = {
        "status": "PASS",
        "analysis": "Decima pretrained-vector spatial effects across training-variance deciles",
        "target_universe": "complete primary downstream held-out panel",
        "decile_definition": "within-cohort deciles of pooled training-spot standard deviation",
        "technical_run_policy": "paired effects averaged per gene before decile summaries",
        "checks": {
            "four_cohorts": bool(summary["cohort"].nunique() == 4),
            "ten_deciles": bool(summary["training_variance_decile"].nunique() == 10),
            "three_runs_per_gene": bool(
                per_run.groupby(["cohort", "gene_id", "contrast", "endpoint"])["seed"]
                .nunique()
                .eq(3)
                .all()
            ),
        },
        "interpretive_boundary": "Descriptive target distribution; genes are not independent biological replicates.",
    }
    if not all(manifest["checks"].values()):
        manifest["status"] = "FAIL"
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    print(
        summary.loc[
            summary["contrast"].eq("pretrained_vs_random")
            & summary["endpoint"].eq("section_centered_within_gene_pcc")
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
