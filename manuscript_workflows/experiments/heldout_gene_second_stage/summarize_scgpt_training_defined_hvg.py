#!/usr/bin/env python3
"""Summarize static-scGPT performance on the frozen training-defined top 50."""

from __future__ import annotations

import argparse
import hashlib
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
OUT = BASE / "training_defined_top50"
SELECTION = ROOT / "paper/submission/source_data/supp_training_defined_top50_hvg_genes.tsv"
COVERAGE = (
    ROOT
    / "experiments/heldout_gene_second_stage/scgpt_static_gene_vectors/gene_coverage.tsv.gz"
)
CHECKPOINTS = {
    "brain": {
        "base": ROOT / "experiments/heldout_gene_second_stage/scgpt_decoder",
        "coverage": ROOT
        / "experiments/heldout_gene_second_stage/scgpt_static_gene_vectors/gene_coverage.tsv.gz",
        "label": "scGPT brain static gene-token vectors",
    },
    "whole_human": {
        "base": ROOT
        / "experiments/heldout_gene_second_stage/scgpt_whole_human_decoder",
        "coverage": ROOT
        / "experiments/heldout_gene_second_stage/scgpt_whole_human_static_gene_vectors/gene_coverage.tsv.gz",
        "label": "scGPT whole-human static gene-token vectors",
    },
}
DECIMA_POOLED_SUMMARY = (
    ROOT / "paper/submission/source_data/supp_training_defined_top50_hvg_summary.tsv"
)
DECIMA_SECTION_SUMMARY = (
    ROOT
    / "paper/submission/source_data/supp_training_defined_top50_hvg_section_centered_summary.tsv"
)
CONDITIONS = (
    "pretrained_gene_tokens",
    "random_gene_vectors",
    "constant_gene_vector",
    "identity_shuffled_gene_tokens",
)
CONTRASTS = {
    "pretrained_vs_random": ("pretrained_gene_tokens", "random_gene_vectors"),
    "pretrained_vs_constant": ("pretrained_gene_tokens", "constant_gene_vector"),
    "pretrained_vs_identity_shuffled": (
        "pretrained_gene_tokens",
        "identity_shuffled_gene_tokens",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize_effects(
    absolute: pd.DataFrame, pcc_column: str, rmse_column: str | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for (cohort, seed), group in absolute.groupby(["cohort", "seed"], sort=False):
        indexed = group.set_index("condition")
        for contrast, (candidate, control) in CONTRASTS.items():
            row: dict[str, object] = {
                "cohort": cohort,
                "seed": int(seed),
                "contrast": contrast,
                "n_genes": 50,
                "mean_within_gene_pcc_gain": float(
                    indexed.loc[candidate, pcc_column] - indexed.loc[control, pcc_column]
                ),
            }
            if rmse_column is not None:
                row["centered_rmse_reduction"] = float(
                    indexed.loc[control, rmse_column] - indexed.loc[candidate, rmse_column]
                )
            rows.append(row)
    effects = pd.DataFrame(rows)
    aggregations: dict[str, tuple[str, str]] = {
        "mean_within_gene_pcc_gain_mean": ("mean_within_gene_pcc_gain", "mean"),
        "mean_within_gene_pcc_gain_min": ("mean_within_gene_pcc_gain", "min"),
        "mean_within_gene_pcc_gain_max": ("mean_within_gene_pcc_gain", "max"),
        "n_runs": ("seed", "size"),
    }
    if rmse_column is not None:
        aggregations.update(
            {
                "centered_rmse_reduction_mean": ("centered_rmse_reduction", "mean"),
                "centered_rmse_reduction_min": ("centered_rmse_reduction", "min"),
                "centered_rmse_reduction_max": ("centered_rmse_reduction", "max"),
            }
        )
    summary = effects.groupby(["cohort", "contrast"], as_index=False).agg(**aggregations)
    return effects, summary


def summarize_gene_effects(
    per_gene: pd.DataFrame, value_column: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Average paired run effects per gene, then describe their distribution."""
    wide = per_gene.pivot(
        index=["cohort", "seed", "gene_id"],
        columns="condition",
        values=value_column,
    ).reset_index()
    run_rows: list[pd.DataFrame] = []
    for contrast, (candidate, control) in CONTRASTS.items():
        run_rows.append(
            wide[["cohort", "seed", "gene_id"]].assign(
                contrast=contrast,
                effect=wide[candidate] - wide[control],
            )
        )
    run_effects = pd.concat(run_rows, ignore_index=True)
    technical_mean = (
        run_effects.groupby(["cohort", "gene_id", "contrast"], as_index=False)["effect"]
        .mean()
        .rename(columns={"effect": "technical_mean_effect"})
    )
    summary_rows: list[dict[str, object]] = []
    for (cohort, contrast), group in technical_mean.groupby(
        ["cohort", "contrast"], sort=False
    ):
        values = group["technical_mean_effect"].to_numpy(dtype=float)
        summary_rows.append(
            {
                "cohort": cohort,
                "contrast": contrast,
                "n_genes": int(len(values)),
                "mean_effect": float(values.mean()),
                "median_effect": float(np.median(values)),
                "q25_effect": float(np.quantile(values, 0.25)),
                "q75_effect": float(np.quantile(values, 0.75)),
                "n_positive": int((values > 0).sum()),
                "fraction_positive": float((values > 0).mean()),
            }
        )
    return technical_mean, pd.DataFrame(summary_rows)


def main() -> None:
    global BASE, OUT, COVERAGE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", choices=CHECKPOINTS, default="brain")
    args = parser.parse_args()
    checkpoint = CHECKPOINTS[args.checkpoint]
    BASE = checkpoint["base"]
    OUT = BASE / "training_defined_top50"
    COVERAGE = checkpoint["coverage"]
    OUT.mkdir(parents=True, exist_ok=True)
    selected = pd.read_csv(SELECTION, sep="\t").sort_values(
        ["cohort", "training_hvg_rank"]
    )
    coverage = pd.read_csv(COVERAGE, sep="\t")
    covered = coverage.loc[coverage["in_scgpt_vocabulary"], ["cohort", "gene_id"]]
    selected_covered = selected.merge(
        covered, on=["cohort", "gene_id"], how="left", indicator=True
    )
    if len(selected) != len(COHORTS) * 50 or not selected_covered["_merge"].eq("both").all():
        raise ValueError("Frozen top-50 selection is not completely covered by scGPT")
    selected.to_csv(OUT / "selected_genes.tsv", sep="\t", index=False)

    pooled_rows: list[pd.DataFrame] = []
    for condition in CONDITIONS:
        for cohort in COHORTS:
            genes = selected.loc[
                selected["cohort"].eq(cohort),
                ["gene_id", "training_hvg_rank", "training_spot_std"],
            ]
            for seed in SEEDS:
                path = (
                    BASE
                    / condition
                    / "components/runs"
                    / cohort
                    / f"seed_{seed}/decima/gene_components.tsv.gz"
                )
                frame = pd.read_csv(path, sep="\t").merge(
                    genes, on="gene_id", how="inner", validate="one_to_one"
                )
                if len(frame) != 50 or not frame["gene_pcc_eligible"].astype(bool).all():
                    raise ValueError(f"Unexpected pooled top-50 denominator in {path}")
                frame["condition"] = condition
                pooled_rows.append(
                    frame[
                        [
                            "cohort",
                            "seed",
                            "condition",
                            "gene_id",
                            "training_hvg_rank",
                            "training_spot_std",
                            "gene_pcc",
                            "centered_mse",
                            "mse",
                            "gene_pcc_eligible",
                            "observed_std",
                        ]
                    ]
                )
    pooled_per_gene = pd.concat(pooled_rows, ignore_index=True)
    pooled_absolute = (
        pooled_per_gene.groupby(["cohort", "seed", "condition"], as_index=False)
        .agg(
            n_genes=("gene_id", "size"),
            mean_within_gene_pcc=("gene_pcc", "mean"),
            median_within_gene_pcc=("gene_pcc", "median"),
            centered_mse=("centered_mse", "mean"),
            overall_mse=("mse", "mean"),
        )
    )
    pooled_absolute["centered_rmse"] = np.sqrt(pooled_absolute["centered_mse"])
    pooled_effects, pooled_summary = summarize_effects(
        pooled_absolute, "mean_within_gene_pcc", "centered_rmse"
    )
    pooled_gene_effects, pooled_gene_summary = summarize_gene_effects(
        pooled_per_gene, "gene_pcc"
    )

    section_rows: list[pd.DataFrame] = []
    for condition in CONDITIONS:
        for cohort in COHORTS:
            genes = selected.loc[
                selected["cohort"].eq(cohort), ["gene_id", "training_hvg_rank"]
            ]
            for seed in SEEDS:
                path = (
                    BASE
                    / "section_centered"
                    / condition
                    / "runs"
                    / cohort
                    / f"seed_{seed}/decima/concat_gene_pcc.tsv.gz"
                )
                frame = pd.read_csv(path, sep="\t").merge(
                    genes, on="gene_id", how="inner", validate="one_to_one"
                )
                if len(frame) != 50 or not frame["gene_pcc_eligible"].astype(bool).all():
                    raise ValueError(f"Unexpected section-centred top-50 denominator in {path}")
                frame["condition"] = condition
                section_rows.append(
                    frame[
                        [
                            "cohort",
                            "seed",
                            "condition",
                            "gene_id",
                            "training_hvg_rank",
                            "section_centered_concat_pcc",
                            "gene_pcc_eligible",
                            "n_spots",
                        ]
                    ]
                )
    section_per_gene = pd.concat(section_rows, ignore_index=True)
    section_absolute = (
        section_per_gene.groupby(["cohort", "seed", "condition"], as_index=False)
        .agg(
            n_genes=("gene_id", "size"),
            section_centered_mean_within_gene_pcc=(
                "section_centered_concat_pcc",
                "mean",
            ),
            section_centered_median_within_gene_pcc=(
                "section_centered_concat_pcc",
                "median",
            ),
        )
    )
    section_effects, section_summary = summarize_effects(
        section_absolute, "section_centered_mean_within_gene_pcc"
    )
    section_gene_effects, section_gene_summary = summarize_gene_effects(
        section_per_gene, "section_centered_concat_pcc"
    )

    decima_pooled = pd.read_csv(DECIMA_POOLED_SUMMARY, sep="\t")
    decima_section = pd.read_csv(DECIMA_SECTION_SUMMARY, sep="\t")
    comparison_contrasts = {"pretrained_vs_random", "pretrained_vs_constant"}
    comparison = (
        pooled_summary.loc[
            pooled_summary["contrast"].isin(comparison_contrasts),
            ["cohort", "contrast", "mean_within_gene_pcc_gain_mean"]
        ]
        .rename(
            columns={"mean_within_gene_pcc_gain_mean": "scgpt_pooled_pcc_gain"}
        )
        .merge(
            section_summary.loc[
                section_summary["contrast"].isin(comparison_contrasts),
                ["cohort", "contrast", "mean_within_gene_pcc_gain_mean"]
            ].rename(
                columns={
                    "mean_within_gene_pcc_gain_mean": "scgpt_section_centered_pcc_gain"
                }
            ),
            on=["cohort", "contrast"],
            validate="one_to_one",
        )
        .merge(
            decima_pooled[
                ["cohort", "contrast", "mean_within_gene_pcc_gain_mean"]
            ].rename(
                columns={"mean_within_gene_pcc_gain_mean": "decima_pooled_pcc_gain"}
            ),
            on=["cohort", "contrast"],
            validate="one_to_one",
        )
        .merge(
            decima_section[
                [
                    "cohort",
                    "contrast",
                    "section_centered_mean_within_gene_pcc_gain_mean",
                ]
            ].rename(
                columns={
                    "section_centered_mean_within_gene_pcc_gain_mean": "decima_section_centered_pcc_gain"
                }
            ),
            on=["cohort", "contrast"],
            validate="one_to_one",
        )
    )

    pooled_per_gene.to_csv(
        OUT / "pooled_per_gene_per_run.tsv.gz", sep="\t", index=False, compression="gzip"
    )
    pooled_absolute.to_csv(OUT / "pooled_absolute_per_run.tsv", sep="\t", index=False)
    pooled_effects.to_csv(OUT / "pooled_effects_per_run.tsv", sep="\t", index=False)
    pooled_summary.to_csv(OUT / "pooled_effects_summary.tsv", sep="\t", index=False)
    pooled_gene_effects.to_csv(
        OUT / "pooled_gene_effects_technical_mean.tsv.gz",
        sep="\t",
        index=False,
        compression="gzip",
    )
    pooled_gene_summary.to_csv(
        OUT / "pooled_gene_effects_summary.tsv", sep="\t", index=False
    )
    section_per_gene.to_csv(
        OUT / "section_centered_per_gene_per_run.tsv.gz",
        sep="\t",
        index=False,
        compression="gzip",
    )
    section_absolute.to_csv(
        OUT / "section_centered_absolute_per_run.tsv", sep="\t", index=False
    )
    section_effects.to_csv(
        OUT / "section_centered_effects_per_run.tsv", sep="\t", index=False
    )
    section_summary.to_csv(
        OUT / "section_centered_effects_summary.tsv", sep="\t", index=False
    )
    section_gene_effects.to_csv(
        OUT / "section_centered_gene_effects_technical_mean.tsv.gz",
        sep="\t",
        index=False,
        compression="gzip",
    )
    section_gene_summary.to_csv(
        OUT / "section_centered_gene_effects_summary.tsv", sep="\t", index=False
    )
    comparison.to_csv(
        OUT / "aligned_target_representation_comparison.tsv", sep="\t", index=False
    )

    checks = {
        "same_frozen_top50_as_primary_decima_analysis": True,
        "all_selected_genes_in_scgpt_vocabulary": True,
        "complete_pooled_grid": len(pooled_per_gene) == 4 * 3 * 4 * 50,
        "complete_section_centered_grid": len(section_per_gene) == 4 * 3 * 4 * 50,
        "all_pooled_gene_pcc_eligible": bool(pooled_per_gene["gene_pcc_eligible"].all()),
        "all_section_centered_gene_pcc_eligible": bool(
            section_per_gene["gene_pcc_eligible"].all()
        ),
        "three_runs_per_condition_and_cohort": bool(
            pooled_absolute.groupby(["cohort", "condition"])["seed"].nunique().eq(3).all()
            and section_absolute.groupby(["cohort", "condition"])["seed"].nunique().eq(3).all()
        ),
    }
    manifest = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "representation": checkpoint["label"],
        "checkpoint_scope": args.checkpoint,
        "selection": "the exact top 50 downstream-held-out genes per cohort frozen by the primary Decima analysis, ranked using pooled training-spot expression standard deviation",
        "selection_sha256": sha256(SELECTION),
        "selection_uses_validation_or_test_expression": False,
        "section_centered_definition": "observed and predicted means removed separately within each test section; centred spots concatenated before per-gene PCC",
        "cross_representation_comparison_boundary": "Decima and scGPT use the same frozen top-50 evaluation targets and gene-partition logic, but the scGPT assay excludes vocabulary-uncovered training and held-out genes; the aligned-target table is descriptive rather than a strict representation-only comparison.",
        "conditions": list(CONDITIONS),
        "runs": list(SEEDS),
        "counts": {
            "selected_genes": int(len(selected)),
            "pooled_per_gene_rows": int(len(pooled_per_gene)),
            "pooled_absolute_rows": int(len(pooled_absolute)),
            "section_centered_per_gene_rows": int(len(section_per_gene)),
            "section_centered_absolute_rows": int(len(section_absolute)),
            "pooled_gene_effect_rows": int(len(pooled_gene_effects)),
            "section_centered_gene_effect_rows": int(len(section_gene_effects)),
            "representation_comparison_rows": int(len(comparison)),
        },
        "checks": checks,
        "interpretive_boundary": "This sensitivity analysis tests one additional frozen representation in the same factorized dot-product decoder; it does not establish behaviour in every gene-conditioned architecture.",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    print("\nPooled top-50 effects\n" + pooled_summary.to_string(index=False))
    print("\nSection-centred top-50 effects\n" + section_summary.to_string(index=False))
    print("\nSection-centred gene-level consistency\n" + section_gene_summary.to_string(index=False))
    print(
        "\nAligned-target descriptive representation comparison\n"
        + comparison.to_string(index=False)
    )


if __name__ == "__main__":
    main()
