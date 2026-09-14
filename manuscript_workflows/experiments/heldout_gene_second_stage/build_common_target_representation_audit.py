#!/usr/bin/env python3
"""Reaggregate Decima and scGPT assays on the exact common held-out targets.

This removes outcome-set differences without claiming a representation-only
head-to-head comparison: the downstream training-gene universes, vector
dimensions and upstream pretraining sources still differ.  All contrasts are
therefore computed within representation.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.append(str(REPOSITORY_ROOT))

from experiments.multicohort_geneheldout_decima_clean_split.evaluate_components import (
    safe_corr,
)
from experiments.multicohort_geneheldout_decima_clean_split.clean_split_config import (
    CLEAN_ROOT,
    COHORTS,
    ROOT,
    SEEDS,
)


OUT = ROOT / "experiments/heldout_gene_second_stage/common_target_representation_audit"
SCGPT_ROOT = ROOT / "experiments/heldout_gene_second_stage/scgpt_decoder"
COVERAGE = (
    ROOT
    / "experiments/heldout_gene_second_stage/scgpt_static_gene_vectors/gene_coverage.tsv.gz"
)
DECIMA_CONDITIONS = {
    "pretrained": "decima",
    "random": "random",
    "constant": "constant",
}
SCGPT_CONDITIONS = {
    "pretrained": "pretrained_gene_tokens",
    "random": "random_gene_vectors",
    "constant": "constant_gene_vector",
    "identity_shuffled": "identity_shuffled_gene_tokens",
}
SCGPT_CHECKPOINTS = {
    "brain": {
        "root": ROOT / "experiments/heldout_gene_second_stage/scgpt_decoder",
        "coverage": ROOT
        / "experiments/heldout_gene_second_stage/scgpt_static_gene_vectors/gene_coverage.tsv.gz",
        "out": ROOT
        / "experiments/heldout_gene_second_stage/common_target_representation_audit",
        "label": "scGPT brain gene-token",
    },
    "whole_human": {
        "root": ROOT
        / "experiments/heldout_gene_second_stage/scgpt_whole_human_decoder",
        "coverage": ROOT
        / "experiments/heldout_gene_second_stage/scgpt_whole_human_static_gene_vectors/gene_coverage.tsv.gz",
        "out": ROOT
        / "experiments/heldout_gene_second_stage/common_target_representation_audit_whole_human",
        "label": "scGPT whole-human gene-token",
    },
}


def aggregate_components(table: pd.DataFrame) -> dict[str, float | int]:
    count = table["n_observations"].to_numpy(dtype=np.float64)
    true_mean = table["observed_mean"].to_numpy(dtype=np.float64)
    pred_mean = table["predicted_mean"].to_numpy(dtype=np.float64)
    true_std = table["observed_std"].to_numpy(dtype=np.float64)
    pred_std = table["predicted_std"].to_numpy(dtype=np.float64)
    gene_corr = table["gene_pcc"].fillna(0.0).to_numpy(dtype=np.float64)
    total_count = float(count.sum())

    sum_true = count * true_mean
    sum_pred = count * pred_mean
    sum_true2 = count * (np.square(true_std) + np.square(true_mean))
    sum_pred2 = count * (np.square(pred_std) + np.square(pred_mean))
    sum_cross = count * (
        gene_corr * true_std * pred_std + true_mean * pred_mean
    )
    pooled_true_mean = float(sum_true.sum() / total_count)
    pooled_pred_mean = float(sum_pred.sum() / total_count)
    pooled_true_var = float(sum_true2.sum() / total_count - pooled_true_mean**2)
    pooled_pred_var = float(sum_pred2.sum() / total_count - pooled_pred_mean**2)
    pooled_cov = float(
        sum_cross.sum() / total_count - pooled_true_mean * pooled_pred_mean
    )

    weights = count / total_count
    overall_mse = float(np.sum(weights * table["mse"].to_numpy(dtype=float)))
    gene_mean_mse = float(
        np.sum(weights * np.square(pred_mean - true_mean))
    )
    centered_mse = float(
        np.sum(weights * table["centered_mse"].to_numpy(dtype=float))
    )
    eligible = table["gene_pcc_eligible"].astype(bool).to_numpy()
    centered_cov = float(np.sum(count * gene_corr * true_std * pred_std))
    centered_true_ss = float(np.sum(count * np.square(true_std)))
    centered_pred_ss = float(np.sum(count * np.square(pred_std)))
    centered_full_pcc = (
        centered_cov / math.sqrt(centered_true_ss * centered_pred_ss)
        if centered_true_ss > 0 and centered_pred_ss > 0
        else float("nan")
    )
    full_matrix_pcc = (
        pooled_cov / math.sqrt(pooled_true_var * pooled_pred_var)
        if pooled_true_var > 0 and pooled_pred_var > 0
        else float("nan")
    )
    return {
        "n_genes": int(len(table)),
        "n_gene_pcc_eligible": int(eligible.sum()),
        "full_matrix_pcc": full_matrix_pcc,
        "overall_mse": overall_mse,
        "gene_mean_pcc": safe_corr(pred_mean, true_mean),
        "gene_mean_rmse": math.sqrt(max(gene_mean_mse, 0.0)),
        "gene_mean_mse": gene_mean_mse,
        "mean_within_gene_pcc": float(gene_corr[eligible].mean()),
        "centered_full_matrix_pcc": centered_full_pcc,
        "centered_rmse": math.sqrt(max(centered_mse, 0.0)),
        "centered_mse": centered_mse,
        "decomposition_error": overall_mse - gene_mean_mse - centered_mse,
    }


def section_centered_mean(path: Path, targets: set[str]) -> tuple[float, int]:
    table = pd.read_csv(path, sep="\t")
    table = table.loc[table["gene_id"].isin(targets)].copy()
    if "gene_pcc_eligible" in table:
        table = table.loc[table["gene_pcc_eligible"].astype(bool)]
    return float(table["section_centered_concat_pcc"].mean()), int(len(table))


def improvement(candidate: float, control: float, endpoint: str) -> float:
    if endpoint.endswith("mse") or endpoint.endswith("rmse"):
        return control - candidate
    return candidate - control


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scgpt-checkpoint", choices=SCGPT_CHECKPOINTS, default="brain")
    args = parser.parse_args()
    checkpoint = SCGPT_CHECKPOINTS[args.scgpt_checkpoint]
    output = checkpoint["out"]
    output.mkdir(parents=True, exist_ok=True)
    coverage = pd.read_csv(checkpoint["coverage"], sep="\t")
    targets = coverage.loc[
        coverage["in_scgpt_vocabulary"]
        & coverage["downstream_partition"].eq("held_out"),
        ["cohort", "gene_id", "gene_symbol"],
    ].copy()
    targets.to_csv(output / "common_heldout_targets.tsv.gz", sep="\t", index=False)
    target_sets = {
        cohort: set(group["gene_id"].astype(str))
        for cohort, group in targets.groupby("cohort")
    }

    rows: list[dict[str, object]] = []
    validation_differences: list[float] = []
    representation_specs = {
        "Decima sequence-derived": (CLEAN_ROOT, DECIMA_CONDITIONS),
        checkpoint["label"]: (checkpoint["root"], SCGPT_CONDITIONS),
    }
    for representation, (root, conditions) in representation_specs.items():
        for condition, directory in conditions.items():
            condition_complete = all(
                (
                    root
                    / ("components/runs" if representation.startswith("Decima") else f"{directory}/components/runs")
                    / cohort
                    / f"seed_{seed}/decima/gene_components.tsv.gz"
                    if not representation.startswith("Decima")
                    else root
                    / "components/runs"
                    / cohort
                    / f"seed_{seed}/{directory}/gene_components.tsv.gz"
                ).exists()
                for cohort in COHORTS
                for seed in SEEDS
            )
            if not condition_complete:
                continue
            for cohort in COHORTS:
                common = target_sets[cohort]
                for seed in SEEDS:
                    if representation.startswith("Decima"):
                        component_path = (
                            root
                            / "components/runs"
                            / cohort
                            / f"seed_{seed}/{directory}/gene_components.tsv.gz"
                        )
                        section_path = (
                            root
                            / "section_centered/runs"
                            / cohort
                            / f"seed_{seed}/{directory}/concat_gene_pcc.tsv.gz"
                        )
                        summary_path = component_path.with_name("summary.json")
                    else:
                        component_path = (
                            root
                            / directory
                            / "components/runs"
                            / cohort
                            / f"seed_{seed}/decima/gene_components.tsv.gz"
                        )
                        section_path = (
                            root
                            / "section_centered"
                            / directory
                            / "runs"
                            / cohort
                            / f"seed_{seed}/decima/concat_gene_pcc.tsv.gz"
                        )
                        summary_path = component_path.with_name("summary.json")
                    full = pd.read_csv(component_path, sep="\t")
                    common_table = full.loc[full["gene_id"].isin(common)].copy()
                    if len(common_table) != len(common):
                        raise ValueError(f"Common-target mismatch: {component_path}")
                    metrics = aggregate_components(common_table)
                    section_pcc, section_n = section_centered_mean(section_path, common)
                    metrics["section_centered_mean_within_gene_pcc"] = section_pcc
                    metrics["n_section_centered_gene_pcc"] = section_n
                    rows.append(
                        {
                            "representation": representation,
                            "cohort": cohort,
                            "seed": seed,
                            "condition": condition,
                            **metrics,
                        }
                    )
                    # Validate the moment reconstruction on the complete table.
                    stored = json.loads(summary_path.read_text())
                    reconstructed = aggregate_components(full)
                    for reconstructed_key, stored_key in (
                        ("full_matrix_pcc", "full_matrix_pcc"),
                        ("overall_mse", "overall_mse"),
                        ("gene_mean_mse", "abundance_mse"),
                        ("centered_mse", "centered_mse"),
                    ):
                        validation_differences.append(
                            abs(
                                float(reconstructed[reconstructed_key])
                                - float(stored[stored_key])
                            )
                        )

    absolute = pd.DataFrame(rows)
    endpoints = [
        "full_matrix_pcc",
        "overall_mse",
        "gene_mean_pcc",
        "gene_mean_rmse",
        "gene_mean_mse",
        "mean_within_gene_pcc",
        "centered_full_matrix_pcc",
        "centered_rmse",
        "centered_mse",
        "section_centered_mean_within_gene_pcc",
    ]
    effects: list[dict[str, object]] = []
    for (representation, cohort, seed), group in absolute.groupby(
        ["representation", "cohort", "seed"], sort=False
    ):
        indexed = group.set_index("condition")
        candidate = indexed.loc["pretrained"]
        for control in ("random", "constant", "identity_shuffled"):
            if control not in indexed.index:
                continue
            for endpoint in endpoints:
                value = (
                    float("nan")
                    if endpoint == "gene_mean_pcc" and control == "constant"
                    else improvement(
                        float(candidate[endpoint]),
                        float(indexed.loc[control, endpoint]),
                        endpoint,
                    )
                )
                effects.append(
                    {
                        "representation": representation,
                        "cohort": cohort,
                        "seed": seed,
                        "contrast": f"pretrained_vs_{control}",
                        "endpoint": endpoint,
                        "effect": value,
                    }
                )
            gene_mean_reduction = float(
                indexed.loc[control, "gene_mean_mse"] - candidate["gene_mean_mse"]
            )
            total_reduction = float(
                indexed.loc[control, "overall_mse"] - candidate["overall_mse"]
            )
            effects.append(
                {
                    "representation": representation,
                    "cohort": cohort,
                    "seed": seed,
                    "contrast": f"pretrained_vs_{control}",
                    "endpoint": "gene_mean_share_of_mse_reduction",
                    "effect": gene_mean_reduction / total_reduction
                    if total_reduction > 0
                    else float("nan"),
                }
            )
    effects_frame = pd.DataFrame(effects)
    summary = (
        effects_frame.groupby(
            ["representation", "cohort", "contrast", "endpoint"], as_index=False
        )
        .agg(
            mean_effect=("effect", "mean"),
            min_effect=("effect", "min"),
            max_effect=("effect", "max"),
            n_runs=("effect", "size"),
        )
    )
    absolute.to_csv(output / "absolute_per_run.tsv", sep="\t", index=False)
    effects_frame.to_csv(output / "within_representation_effects_per_run.tsv", sep="\t", index=False)
    summary.to_csv(output / "within_representation_effects_summary.tsv", sep="\t", index=False)

    common_counts = targets.groupby("cohort")["gene_id"].nunique().to_dict()
    max_validation_difference = float(max(validation_differences, default=0.0))
    checks = {
        "four_cohorts": len(common_counts) == 4,
        "expected_common_target_counts": common_counts
        == {
            "hippocampus_donor_disjoint": 3574,
            "dlpfc": 3574,
            "nac": 3574,
            "her2st": 3179,
        },
        "component_decomposition_exact": bool(
            absolute["decomposition_error"].abs().max() < 1.0e-9
        ),
        "moment_reconstruction_matches_raw_summaries": max_validation_difference
        < 3.0e-6,
        "only_within_representation_effects": True,
    }
    manifest = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "scgpt_checkpoint_scope": args.scgpt_checkpoint,
        "scgpt_representation_label": checkpoint["label"],
        "analysis": "exact-common-held-out-target sensitivity for Decima and scGPT assays",
        "common_target_counts": common_counts,
        "included_scgpt_identity_shuffle": bool(
            "identity_shuffled" in absolute.loc[
                absolute["representation"].str.startswith("scGPT"), "condition"
            ].unique()
        ),
        "maximum_full-table_moment_reconstruction_difference": max_validation_difference,
        "checks": checks,
        "interpretive_boundary": (
            "The evaluation targets are exactly aligned, but vector dimensions, upstream "
            "training and downstream training-gene universes remain different. Results are "
            "parallel within-representation contrasts, not a Decima-versus-scGPT leaderboard."
        ),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    print(
        summary.loc[
            summary["endpoint"].isin(
                [
                    "full_matrix_pcc",
                    "gene_mean_pcc",
                    "mean_within_gene_pcc",
                    "gene_mean_share_of_mse_reduction",
                ]
            )
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
