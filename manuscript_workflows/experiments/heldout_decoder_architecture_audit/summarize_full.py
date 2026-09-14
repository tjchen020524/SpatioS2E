#!/usr/bin/env python3
"""Audit and summarize the complete held-out-decoder architecture experiment."""

from __future__ import annotations
from workflow_paths import DATA_ROOT

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = DATA_ROOT
AUDIT = ROOT / "experiments/heldout_decoder_architecture_audit"
CLEAN = ROOT / "experiments/multicohort_geneheldout_decima_clean_split/components/runs"
COHORTS = ("hippocampus_donor_disjoint", "dlpfc", "nac", "her2st")
SEEDS = (42, 123, 456)
VARIANTS = ("decima", "random", "constant")
CONDITIONS = (
    "bias_correct__interaction_correct",
    "bias_correct__interaction_permuted",
    "bias_permuted__interaction_correct",
    "bias_permuted__interaction_permuted",
    "bias_correct__interaction_zero",
    "bias_zero__interaction_correct",
)


def load_json(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def summarize_effects(per_run: pd.DataFrame, architecture: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    metrics = (
        ("full_matrix_pcc", "pooled_pcc", 1.0),
        ("overall_mse", "overall_mse", -1.0),
        ("abundance_pcc", "abundance_pcc", 1.0),
        ("abundance_rmse", "abundance_rmse", -1.0),
        ("mean_gene_pcc", "mean_gene_pcc", 1.0),
        ("centered_full_matrix_pcc", "centered_full_matrix_pcc", 1.0),
        ("centered_rmse", "centered_rmse", -1.0),
    )
    if architecture == "residual_only":
        metrics = (
            ("mean_gene_pcc", "primary_mean_within_section_gene_pcc", 1.0),
            ("centered_full_matrix_pcc", "primary_centered_full_matrix_pcc", 1.0),
            ("centered_rmse", "primary_section_centered_rmse", -1.0),
        )
    for cohort in COHORTS:
        cohort_frame = per_run.loc[per_run["cohort"] == cohort]
        for control in ("random", "constant"):
            paired = cohort_frame.pivot(index="seed", columns="variant")
            for public_name, column, sign in metrics:
                values = sign * (paired[column]["decima"] - paired[column][control])
                rows.append(
                    {
                        "architecture": architecture,
                        "cohort": cohort,
                        "contrast": f"pretrained_vs_{control}",
                        "endpoint": public_name,
                        "direction": "gain" if sign > 0 else "reduction",
                        "mean_effect": float(values.mean()),
                        "min_effect": float(values.min()),
                        "max_effect": float(values.max()),
                        "n_runs": int(values.size),
                    }
                )
            if architecture != "residual_only":
                normalized = {
                    "abundance_error_fraction_reduced": (
                        paired["abundance_mse"][control]
                        - paired["abundance_mse"]["decima"]
                    )
                    / paired["abundance_mse"][control],
                    "centered_error_fraction_reduced": (
                        paired["centered_mse"][control]
                        - paired["centered_mse"]["decima"]
                    )
                    / paired["centered_mse"][control],
                }
                for public_name, values in normalized.items():
                    rows.append(
                        {
                            "architecture": architecture,
                            "cohort": cohort,
                            "contrast": f"pretrained_vs_{control}",
                            "endpoint": public_name,
                            "direction": "fraction reduced",
                            "mean_effect": float(values.mean()),
                            "min_effect": float(values.min()),
                            "max_effect": float(values.max()),
                            "n_runs": int(values.size),
                        }
                    )
    return pd.DataFrame(rows)


def main() -> None:
    bias_rows: list[dict[str, object]] = []
    concat_rows: list[dict[str, object]] = []
    residual_rows: list[dict[str, object]] = []
    branch_frames: list[pd.DataFrame] = []
    stability_rows: list[dict[str, object]] = []
    reproduction_rows: list[dict[str, object]] = []

    for cohort in COHORTS:
        for seed in SEEDS:
            branch_path = AUDIT / "branch_intervention/runs" / cohort / f"seed_{seed}/per_run.tsv"
            branch = pd.read_csv(branch_path, sep="\t")
            if set(branch["condition"]) != set(CONDITIONS) or len(branch) != len(CONDITIONS):
                raise RuntimeError(f"Incomplete branch intervention: {branch_path}")
            branch_frames.append(branch)

            stored = load_json(CLEAN / cohort / f"seed_{seed}/decima/summary.json")
            untouched = branch.loc[branch["condition"] == CONDITIONS[0]].iloc[0]
            differences = {
                "full_matrix_pcc": abs(float(untouched["full_matrix_pcc"]) - float(stored["pooled_pcc"])),
                "overall_mse": abs(float(untouched["full_matrix_mse"]) - float(stored["overall_mse"])),
                "abundance_pcc": abs(float(untouched["abundance_pcc"]) - float(stored["abundance_pcc"])),
                "mean_gene_pcc": abs(float(untouched["mean_gene_pcc"]) - float(stored["mean_gene_pcc"])),
                "centered_full_matrix_pcc": abs(
                    float(untouched["centered_full_matrix_pcc"])
                    - float(stored["centered_full_matrix_pcc"])
                ),
            }
            reproduction_rows.append(
                {
                    "cohort": cohort,
                    "seed": seed,
                    **{f"abs_diff_{key}": value for key, value in differences.items()},
                    "pass": max(differences.values()) < 3e-6,
                }
            )

            for variant in VARIANTS:
                bias_path = AUDIT / "bias_free/components/runs" / cohort / f"seed_{seed}" / variant / "summary.json"
                bias = load_json(bias_path)
                bias_rows.append(bias)
                history_path = (
                    AUDIT / "bias_free/runs" / cohort / f"seed_{seed}" / variant / variant / "results/history.json"
                )
                history = load_json(history_path)
                bias_pass = bool(
                    len(history) == 6
                    and all(
                        np.isfinite(float(row["train_loss"])) and np.isfinite(float(row["score"]))
                        for row in history
                    )
                )
                stability_rows.append(
                    {
                        "architecture": "bias_free",
                        "cohort": cohort,
                        "seed": seed,
                        "variant": variant,
                        "pass": bias_pass,
                    }
                )

                concat_path = AUDIT / "concat_mlp/components/runs" / cohort / f"seed_{seed}" / variant / "summary.json"
                concat = load_json(concat_path)
                concat_rows.append(concat)
                concat_history_path = (
                    AUDIT / "concat_mlp/runs" / cohort / f"seed_{seed}" / variant / variant / "results/history.json"
                )
                concat_history = load_json(concat_history_path)
                concat_pass = bool(
                    len(concat_history) == 6
                    and all(
                        np.isfinite(float(row["train_loss"])) and np.isfinite(float(row["score"]))
                        for row in concat_history
                    )
                )
                stability_rows.append(
                    {
                        "architecture": "concat_mlp",
                        "cohort": cohort,
                        "seed": seed,
                        "variant": variant,
                        "pass": concat_pass,
                    }
                )

                residual_path = AUDIT / "residual_only/runs" / cohort / f"seed_{seed}" / variant / "results/summary.json"
                residual = load_json(residual_path)
                test = residual["heldout_test"]
                residual_rows.append(
                    {
                        "cohort": cohort,
                        "seed": seed,
                        "variant": variant,
                        "best_epoch": residual["best_epoch"],
                        **test,
                    }
                )
                residual_pass = bool(
                    int(residual["protocol"]["epochs"]) == 6
                    and float(test["max_abs_true_section_mean"]) < 1e-5
                    and float(test["max_abs_pred_section_mean"]) < 1e-5
                    and np.isfinite(float(test["primary_mean_within_section_gene_pcc"]))
                    and np.isfinite(float(test["primary_section_centered_rmse"]))
                )
                stability_rows.append(
                    {
                        "architecture": "residual_only",
                        "cohort": cohort,
                        "seed": seed,
                        "variant": variant,
                        "pass": residual_pass,
                    }
                )

    bias_frame = pd.DataFrame(bias_rows).sort_values(["cohort", "seed", "variant"])
    concat_frame = pd.DataFrame(concat_rows).sort_values(["cohort", "seed", "variant"])
    residual_frame = pd.DataFrame(residual_rows).sort_values(["cohort", "seed", "variant"])
    branch_frame = pd.concat(branch_frames, ignore_index=True).sort_values(["cohort", "seed", "condition"])
    stability = pd.DataFrame(stability_rows).sort_values(["architecture", "cohort", "seed", "variant"])
    reproduction = pd.DataFrame(reproduction_rows).sort_values(["cohort", "seed"])

    expected_runs = len(COHORTS) * len(SEEDS) * len(VARIANTS)
    if any(len(frame) != expected_runs for frame in (bias_frame, concat_frame, residual_frame)):
        raise RuntimeError("Architecture audit is incomplete")
    if not bool(stability["pass"].all()):
        raise RuntimeError("Training/centring stability audit failed:\n" + stability.loc[~stability["pass"]].to_string(index=False))
    if not bool(reproduction["pass"].all()):
        raise RuntimeError("Branch reproduction audit failed:\n" + reproduction.loc[~reproduction["pass"]].to_string(index=False))

    effects = pd.concat(
        [
            summarize_effects(bias_frame, "bias_free"),
            summarize_effects(concat_frame, "concat_mlp"),
            summarize_effects(residual_frame, "residual_only"),
        ],
        ignore_index=True,
    )

    branch_effect_rows: list[dict[str, object]] = []
    for cohort in COHORTS:
        cohort_frame = branch_frame.loc[branch_frame["cohort"] == cohort]
        for endpoint in (
            "full_matrix_pcc",
            "abundance_pcc",
            "mean_gene_pcc",
            "centered_full_matrix_pcc",
            "primary_mean_within_section_gene_pcc",
            "primary_centered_full_matrix_pcc",
        ):
            wide = cohort_frame.pivot(index="seed", columns="condition", values=endpoint)
            comparisons = [
                ("permute_interaction_given_correct_bias", CONDITIONS[1]),
                ("permute_bias_given_correct_interaction", CONDITIONS[2]),
                ("zero_bias_interaction_only", CONDITIONS[5]),
            ]
            if endpoint in ("full_matrix_pcc", "abundance_pcc"):
                comparisons.append(("zero_interaction_gene_vector_only", CONDITIONS[4]))
            for intervention, comparison in comparisons:
                loss = wide[CONDITIONS[0]] - wide[comparison]
                branch_effect_rows.append(
                    {
                        "cohort": cohort,
                        "intervention": intervention,
                        "endpoint": endpoint,
                        "mean_loss": float(loss.mean()),
                        "min_loss": float(loss.min()),
                        "max_loss": float(loss.max()),
                        "n_runs": int(loss.size),
                    }
                )
    branch_effects = pd.DataFrame(branch_effect_rows)

    branch_error_rows: list[dict[str, object]] = []
    for cohort in COHORTS:
        cohort_frame = branch_frame.loc[branch_frame["cohort"] == cohort]
        for endpoint in (
            "full_matrix_mse",
            "abundance_rmse",
            "centered_rmse",
            "gene_mean_mse",
            "centered_mse",
        ):
            wide = cohort_frame.pivot(index="seed", columns="condition", values=endpoint)
            for intervention, comparison in (
                ("permute_interaction_given_correct_bias", CONDITIONS[1]),
                ("permute_bias_given_correct_interaction", CONDITIONS[2]),
                ("zero_bias_interaction_only", CONDITIONS[5]),
                ("zero_interaction_gene_vector_only", CONDITIONS[4]),
            ):
                increase = wide[comparison] - wide[CONDITIONS[0]]
                branch_error_rows.append(
                    {
                        "cohort": cohort,
                        "intervention": intervention,
                        "endpoint": endpoint,
                        "direction": "perturbed minus complete; positive is worse",
                        "mean_increase": float(increase.mean()),
                        "min_increase": float(increase.min()),
                        "max_increase": float(increase.max()),
                        "n_runs": int(increase.size),
                    }
                )
    branch_error_effects = pd.DataFrame(branch_error_rows)

    retention_rows: list[dict[str, object]] = []
    for cohort in COHORTS:
        for seed in SEEDS:
            conditions = branch_frame.loc[
                (branch_frame["cohort"] == cohort) & (branch_frame["seed"] == seed)
            ].set_index("condition")
            complete = conditions.loc[CONDITIONS[0]]
            gene_only = conditions.loc[CONDITIONS[4]]
            random_control = load_json(CLEAN / cohort / f"seed_{seed}/random/summary.json")
            complete_pcc_gain = float(complete["full_matrix_pcc"] - random_control["pooled_pcc"])
            gene_only_pcc_gain = float(gene_only["full_matrix_pcc"] - random_control["pooled_pcc"])
            complete_mse_reduction = float(random_control["overall_mse"] - complete["full_matrix_mse"])
            gene_only_mse_reduction = float(random_control["overall_mse"] - gene_only["full_matrix_mse"])
            retention_rows.append(
                {
                    "cohort": cohort,
                    "seed": seed,
                    "complete_full_matrix_pcc": float(complete["full_matrix_pcc"]),
                    "gene_only_full_matrix_pcc": float(gene_only["full_matrix_pcc"]),
                    "gene_only_full_matrix_pcc_loss": float(
                        complete["full_matrix_pcc"] - gene_only["full_matrix_pcc"]
                    ),
                    "complete_overall_mse": float(complete["full_matrix_mse"]),
                    "gene_only_overall_mse": float(gene_only["full_matrix_mse"]),
                    "gene_only_overall_mse_increase": float(
                        gene_only["full_matrix_mse"] - complete["full_matrix_mse"]
                    ),
                    "complete_abundance_pcc": float(complete["abundance_pcc"]),
                    "gene_only_abundance_pcc": float(gene_only["abundance_pcc"]),
                    "gene_only_abundance_pcc_loss": float(
                        complete["abundance_pcc"] - gene_only["abundance_pcc"]
                    ),
                    "complete_abundance_rmse": float(complete["abundance_rmse"]),
                    "gene_only_abundance_rmse": float(gene_only["abundance_rmse"]),
                    "gene_only_abundance_rmse_increase": float(
                        gene_only["abundance_rmse"] - complete["abundance_rmse"]
                    ),
                    "complete_centered_rmse": float(complete["centered_rmse"]),
                    "gene_only_centered_rmse": float(gene_only["centered_rmse"]),
                    "gene_only_centered_rmse_increase": float(
                        gene_only["centered_rmse"] - complete["centered_rmse"]
                    ),
                    "complete_vs_random_full_matrix_pcc_gain": complete_pcc_gain,
                    "gene_only_vs_random_full_matrix_pcc_gain": gene_only_pcc_gain,
                    "full_matrix_pcc_gain_retained_fraction": gene_only_pcc_gain / complete_pcc_gain,
                    "complete_vs_random_overall_mse_reduction": complete_mse_reduction,
                    "gene_only_vs_random_overall_mse_reduction": gene_only_mse_reduction,
                    "overall_mse_reduction_retained_fraction": gene_only_mse_reduction
                    / complete_mse_reduction,
                    "gene_only_gene_mean_mse": float(gene_only["gene_mean_mse"]),
                    "gene_only_centered_mse": float(gene_only["centered_mse"]),
                    "gene_only_decomposition_error": float(gene_only["decomposition_error"]),
                }
            )
    retention = pd.DataFrame(retention_rows).sort_values(["cohort", "seed"])
    retention_summary = (
        retention.groupby("cohort", sort=True)
        .agg(
            full_matrix_pcc_loss_mean=("gene_only_full_matrix_pcc_loss", "mean"),
            full_matrix_pcc_loss_min=("gene_only_full_matrix_pcc_loss", "min"),
            full_matrix_pcc_loss_max=("gene_only_full_matrix_pcc_loss", "max"),
            overall_mse_increase_mean=("gene_only_overall_mse_increase", "mean"),
            overall_mse_increase_min=("gene_only_overall_mse_increase", "min"),
            overall_mse_increase_max=("gene_only_overall_mse_increase", "max"),
            pcc_gain_retained_fraction_mean=("full_matrix_pcc_gain_retained_fraction", "mean"),
            pcc_gain_retained_fraction_min=("full_matrix_pcc_gain_retained_fraction", "min"),
            pcc_gain_retained_fraction_max=("full_matrix_pcc_gain_retained_fraction", "max"),
            mse_reduction_retained_fraction_mean=("overall_mse_reduction_retained_fraction", "mean"),
            mse_reduction_retained_fraction_min=("overall_mse_reduction_retained_fraction", "min"),
            mse_reduction_retained_fraction_max=("overall_mse_reduction_retained_fraction", "max"),
        )
        .reset_index()
    )

    output = AUDIT / "full_summary"
    output.mkdir(parents=True, exist_ok=True)
    bias_frame.to_csv(output / "bias_free_per_run.tsv", sep="\t", index=False)
    concat_frame.to_csv(output / "concat_mlp_per_run.tsv", sep="\t", index=False)
    residual_frame.to_csv(output / "residual_only_per_run.tsv", sep="\t", index=False)
    branch_frame.to_csv(output / "branch_intervention_per_run.tsv", sep="\t", index=False)
    stability.to_csv(output / "stability.tsv", sep="\t", index=False)
    reproduction.to_csv(output / "branch_reproduction.tsv", sep="\t", index=False)
    effects.to_csv(output / "paired_effect_summary.tsv", sep="\t", index=False)
    branch_effects.to_csv(output / "branch_effect_summary.tsv", sep="\t", index=False)
    branch_error_effects.to_csv(output / "branch_error_effect_summary.tsv", sep="\t", index=False)
    retention.to_csv(output / "gene_only_retention_per_run.tsv", sep="\t", index=False)
    retention_summary.to_csv(output / "gene_only_retention_summary.tsv", sep="\t", index=False)
    manifest = {
        "status": "PASS",
        "design": "four cohorts x three analysis runs x three vector conditions",
        "gene_partition": "training-only clean split",
        "n_bias_free_runs": len(bias_frame),
        "n_concat_mlp_runs": len(concat_frame),
        "n_residual_only_runs": len(residual_frame),
        "n_branch_intervention_rows": len(branch_frame),
        "n_stability_checks": len(stability),
        "n_branch_reproduction_checks": len(reproduction),
        "n_gene_only_retention_runs": len(retention),
        "interpretive_boundary": (
            "Bias-free tests necessity of an explicit gene-conditioned bias; concat MLP tests "
            "decoder-family robustness; residual-only tests section-centred spatial transfer; "
            "branch intervention is an in-model pathway diagnostic."
        ),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    print("\nPaired effects\n" + effects.to_string(index=False))
    print("\nBranch effects\n" + branch_effects.to_string(index=False))
    print("\nBranch error effects\n" + branch_error_effects.to_string(index=False))
    print("\nGene-only retention\n" + retention_summary.to_string(index=False))


if __name__ == "__main__":
    main()
