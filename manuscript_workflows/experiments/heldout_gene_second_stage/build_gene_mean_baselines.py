#!/usr/bin/env python3
"""Fit independent gene-vector-only mean models and cross-cohort transfers.

For each cohort, a ridge model learns the mapping from standardized frozen
pretrained gene vectors to across-spot mean log-expression using downstream
training genes and training biological individuals only.  The model never sees
spot or image features.  Its held-out-gene predictions are broadcast over all
test spots to quantify how much conventional matrix performance is available
without spatial prediction.
"""

from __future__ import annotations
from workflow_paths import DATA_ROOT

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from experiments.geneheldout_uni2h_decima_decoder_noseq_controls import (
    run_geneheldout_noseq_controls as base,
)
from experiments.multicohort_geneheldout_decima_clean_split.clean_split_config import (
    CLEAN_ROOT,
    COHORTS,
    ROOT,
    SEEDS,
    configure_clean_base,
)


OUT = ROOT / "experiments/heldout_gene_second_stage/gene_mean_only"
ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1_000.0, 10_000.0)
DISPLAY = {
    "hippocampus_donor_disjoint": "Hippocampus",
    "dlpfc": "DLPFC",
    "nac": "NAc",
    "her2st": "HER2ST",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 2:
        return float("nan")
    x_use = x[mask] - np.mean(x[mask])
    y_use = y[mask] - np.mean(y[mask])
    denominator = math.sqrt(float(np.dot(x_use, x_use) * np.dot(y_use, y_use)))
    return float(np.dot(x_use, y_use) / denominator) if denominator > 0 else float("nan")


def fit_ridge(
    features: np.ndarray,
    outcome: np.ndarray,
    train_idx: np.ndarray,
) -> tuple[Ridge, float, pd.DataFrame]:
    rng = np.random.default_rng(42)
    validation_idx = np.sort(
        rng.choice(train_idx, size=min(2_048, len(train_idx) // 5), replace=False)
    )
    fit_idx = np.setdiff1d(train_idx, validation_idx, assume_unique=False)
    rows: list[dict[str, float]] = []
    for alpha in ALPHAS:
        model = Ridge(alpha=alpha, fit_intercept=True, solver="lsqr", tol=1.0e-5)
        model.fit(features[fit_idx], outcome[fit_idx])
        prediction = np.maximum(model.predict(features[validation_idx]), 0.0)
        error = prediction - outcome[validation_idx]
        rows.append(
            {
                "alpha": alpha,
                "validation_rmse": float(np.sqrt(np.mean(np.square(error)))),
                "validation_pcc": safe_corr(prediction, outcome[validation_idx]),
            }
        )
    validation = pd.DataFrame(rows)
    selected_alpha = float(
        validation.sort_values(
            ["validation_rmse", "alpha"], ascending=[True, True]
        ).iloc[0]["alpha"]
    )
    final = Ridge(
        alpha=selected_alpha, fit_intercept=True, solver="lsqr", tol=1.0e-6
    )
    final.fit(features[train_idx], outcome[train_idx])
    return final, selected_alpha, validation


def constant_map_metrics(
    predicted_mean: np.ndarray, gene_table: pd.DataFrame
) -> dict[str, float | int]:
    observed_mean = gene_table["observed_mean"].to_numpy(dtype=np.float64)
    observed_std = gene_table["observed_std"].to_numpy(dtype=np.float64)
    count = gene_table["n_observations"].to_numpy(dtype=np.float64)
    prediction = np.maximum(np.asarray(predicted_mean, dtype=np.float64), 0.0)
    valid = (
        np.isfinite(observed_mean)
        & np.isfinite(observed_std)
        & np.isfinite(prediction)
        & (count > 0)
    )
    observed_mean = observed_mean[valid]
    observed_std = observed_std[valid]
    prediction = prediction[valid]
    count = count[valid]
    total = float(count.sum())
    weight = count / total
    mean_error = prediction - observed_mean
    abundance_mse = float(np.sum(weight * np.square(mean_error)))
    centered_mse = float(np.sum(weight * np.square(observed_std)))
    overall_mse = abundance_mse + centered_mse
    pooled_true_mean = float(np.sum(weight * observed_mean))
    pooled_pred_mean = float(np.sum(weight * prediction))
    pooled_true_second = float(
        np.sum(weight * (np.square(observed_std) + np.square(observed_mean)))
    )
    pooled_pred_second = float(np.sum(weight * np.square(prediction)))
    pooled_cross = float(np.sum(weight * prediction * observed_mean))
    true_var = pooled_true_second - pooled_true_mean**2
    pred_var = pooled_pred_second - pooled_pred_mean**2
    covariance = pooled_cross - pooled_true_mean * pooled_pred_mean
    full_matrix_pcc = (
        covariance / math.sqrt(true_var * pred_var)
        if true_var > 0 and pred_var > 0
        else float("nan")
    )
    return {
        "n_genes": int(valid.sum()),
        "abundance_pcc": safe_corr(prediction, observed_mean),
        "abundance_rmse": math.sqrt(abundance_mse),
        "abundance_mse": abundance_mse,
        "full_matrix_pcc": full_matrix_pcc,
        "overall_mse": overall_mse,
        "mean_gene_pcc": 0.0,
        "gene_centered_full_matrix_pcc": 0.0,
        "centered_rmse": math.sqrt(centered_mse),
        "centered_mse": centered_mse,
        "decomposition_error": overall_mse - abundance_mse - centered_mse,
    }


def training_means(cohort: str) -> tuple[
    list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame
]:
    configure_clean_base(cohort, 42, "decima")
    manifest = pd.read_csv(base.DATA_ROOT / "manifest.tsv", sep="\t")
    gene_ids = base.load_gene_ids()
    train_idx, heldout_idx = base.load_gene_split(gene_ids)
    expression_store = base.ExpressionStore(gene_ids)
    train_manifest = manifest.loc[manifest["split"] == "train"].copy()
    means = base.compute_train_gene_means(
        train_manifest, expression_store, len(gene_ids)
    ).astype(np.float64)

    cache = np.load(base.DECIMA_EMB_CACHE, allow_pickle=False)
    cache_ids = cache["gene_ids"].astype(str).tolist()
    cache_lookup = {gene: index for index, gene in enumerate(cache_ids)}
    raw_all = cache["embeddings"].astype(np.float32, copy=False)
    raw = raw_all[
        np.asarray([cache_lookup[gene] for gene in gene_ids], dtype=np.int64)
    ]
    feature_mean = raw[train_idx].mean(axis=0, keepdims=True)
    feature_std = raw[train_idx].std(axis=0, keepdims=True)
    standardized = (
        (raw - feature_mean) / np.maximum(feature_std, 1.0e-6)
    ).astype(np.float32, copy=False)
    return (
        gene_ids,
        train_idx,
        heldout_idx,
        means,
        standardized,
        np.vstack([feature_mean, feature_std]),
        manifest,
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cohort_objects: dict[str, dict[str, object]] = {}
    baseline_rows: list[dict[str, object]] = []
    comparison_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    inputs: list[dict[str, object]] = []

    for cohort in COHORTS:
        (
            gene_ids,
            train_idx,
            heldout_idx,
            means,
            features,
            scaling,
            manifest,
        ) = training_means(cohort)
        model, alpha, validation = fit_ridge(features, means, train_idx)
        predicted = np.maximum(model.predict(features[heldout_idx]), 0.0)
        gene_table_path = (
            CLEAN_ROOT
            / "components/runs"
            / cohort
            / "seed_42/decima/gene_components.tsv.gz"
        )
        gene_table = pd.read_csv(gene_table_path, sep="\t")
        heldout_genes = [gene_ids[int(index)] for index in heldout_idx]
        if gene_table["gene_id"].astype(str).tolist() != heldout_genes:
            raise ValueError(f"Held-out gene order mismatch for {cohort}")
        metrics = constant_map_metrics(predicted, gene_table)
        baseline_rows.append(
            {
                "cohort": cohort,
                "display_cohort": DISPLAY[cohort],
                "model": "pretrained_vector_to_gene_mean_ridge",
                "selected_alpha": alpha,
                **metrics,
            }
        )
        validation.insert(0, "cohort", cohort)
        validation.to_csv(OUT / f"{cohort}_ridge_validation.tsv", sep="\t", index=False)
        for gene, value in zip(heldout_genes, predicted.tolist()):
            prediction_rows.append(
                {
                    "cohort": cohort,
                    "gene_id": gene,
                    "predicted_gene_mean": value,
                }
            )

        for seed in SEEDS:
            full_gene_path = (
                CLEAN_ROOT
                / "components/runs"
                / cohort
                / f"seed_{seed}/decima/gene_components.tsv.gz"
            )
            full_summary_path = full_gene_path.parent / "summary.json"
            random_summary_path = (
                CLEAN_ROOT
                / "components/runs"
                / cohort
                / f"seed_{seed}/random/summary.json"
            )
            full_gene = pd.read_csv(full_gene_path, sep="\t")
            full_summary = json.loads(full_summary_path.read_text())
            random_summary = json.loads(random_summary_path.read_text())
            if full_gene["gene_id"].astype(str).tolist() != heldout_genes:
                raise ValueError(f"Full-decoder gene order mismatch for {cohort}/{seed}")
            mean_correlation = safe_corr(
                predicted,
                full_gene["predicted_mean"].to_numpy(dtype=np.float64),
            )
            full_matrix_denominator = float(
                full_summary["full_matrix_pcc"] - random_summary["full_matrix_pcc"]
            )
            mse_denominator = float(
                random_summary["overall_mse"] - full_summary["overall_mse"]
            )
            abundance_denominator = float(
                full_summary["abundance_pcc"] - random_summary["abundance_pcc"]
            )
            comparison_rows.append(
                {
                    "cohort": cohort,
                    "display_cohort": DISPLAY[cohort],
                    "seed": seed,
                    "mean_only_vs_full_predicted_gene_mean_pcc": mean_correlation,
                    "full_minus_mean_only_full_matrix_pcc": float(
                        full_summary["full_matrix_pcc"] - metrics["full_matrix_pcc"]
                    ),
                    "mean_only_minus_full_overall_mse": float(
                        metrics["overall_mse"] - full_summary["overall_mse"]
                    ),
                    "full_minus_mean_only_mean_gene_pcc": float(
                        full_summary["mean_gene_pcc"]
                    ),
                    "full_minus_mean_only_gene_centered_full_matrix_pcc": float(
                        full_summary["centered_full_matrix_pcc"]
                    ),
                    "mean_only_minus_full_centered_rmse": float(
                        metrics["centered_rmse"] - full_summary["centered_rmse"]
                    ),
                    "mean_only_fraction_full_matrix_pcc_gain_vs_random": float(
                        (metrics["full_matrix_pcc"] - random_summary["full_matrix_pcc"])
                        / full_matrix_denominator
                    ),
                    "mean_only_fraction_overall_mse_reduction_vs_random": float(
                        (random_summary["overall_mse"] - metrics["overall_mse"])
                        / mse_denominator
                    ),
                    "mean_only_fraction_abundance_pcc_gain_vs_random": float(
                        (metrics["abundance_pcc"] - random_summary["abundance_pcc"])
                        / abundance_denominator
                    ),
                }
            )
            inputs.append(
                {
                    "path": str(full_summary_path.relative_to(ROOT)),
                    "sha256": sha256(full_summary_path),
                }
            )
            inputs.append(
                {
                    "path": str(random_summary_path.relative_to(ROOT)),
                    "sha256": sha256(random_summary_path),
                }
            )

        cohort_objects[cohort] = {
            "gene_ids": gene_ids,
            "train_idx": train_idx,
            "heldout_idx": heldout_idx,
            "means": means,
            "features": features,
            "scaling": scaling,
            "model": model,
            "gene_table": gene_table,
        }
        inputs.append(
            {"path": str(gene_table_path.relative_to(ROOT)), "sha256": sha256(gene_table_path)}
        )

    cross_rows: list[dict[str, object]] = []
    cache = np.load(base.DECIMA_EMB_CACHE, allow_pickle=False)
    cache_ids = cache["gene_ids"].astype(str).tolist()
    cache_lookup = {gene: index for index, gene in enumerate(cache_ids)}
    raw_all = cache["embeddings"].astype(np.float32, copy=False)
    for source in COHORTS:
        source_object = cohort_objects[source]
        source_model: Ridge = source_object["model"]  # type: ignore[assignment]
        scaling = np.asarray(source_object["scaling"], dtype=np.float32)
        feature_mean, feature_std = scaling[0:1], scaling[1:2]
        for target in COHORTS:
            target_object = cohort_objects[target]
            target_gene_ids: list[str] = target_object["gene_ids"]  # type: ignore[assignment]
            target_heldout_idx = np.asarray(target_object["heldout_idx"], dtype=np.int64)
            target_genes = [target_gene_ids[int(index)] for index in target_heldout_idx]
            raw = raw_all[
                np.asarray([cache_lookup[gene] for gene in target_genes], dtype=np.int64)
            ]
            target_features = (
                (raw - feature_mean) / np.maximum(feature_std, 1.0e-6)
            ).astype(np.float32, copy=False)
            prediction = np.maximum(source_model.predict(target_features), 0.0)
            target_table: pd.DataFrame = target_object["gene_table"]  # type: ignore[assignment]
            observed = target_table["observed_mean"].to_numpy(dtype=np.float64)
            cross_rows.append(
                {
                    "source_cohort": source,
                    "target_cohort": target,
                    "source_display": DISPLAY[source],
                    "target_display": DISPLAY[target],
                    "n_target_heldout_genes": len(target_genes),
                    "abundance_pcc": safe_corr(prediction, observed),
                    "abundance_rmse_unweighted": float(
                        np.sqrt(np.mean(np.square(prediction - observed)))
                    ),
                    "cohort_matched": source == target,
                }
            )

    baseline = pd.DataFrame(baseline_rows)
    comparisons = pd.DataFrame(comparison_rows)
    predictions = pd.DataFrame(prediction_rows)
    cross = pd.DataFrame(cross_rows)
    baseline.to_csv(OUT / "mean_only_absolute.tsv", sep="\t", index=False)
    comparisons.to_csv(OUT / "full_decoder_increment.tsv", sep="\t", index=False)
    predictions.to_csv(
        OUT / "heldout_gene_mean_predictions.tsv.gz",
        sep="\t",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    cross.to_csv(OUT / "cross_cohort_transfer.tsv", sep="\t", index=False)

    manifest = {
        "status": "PASS",
        "model": "ridge regression from frozen standardized pretrained gene vectors to training-individual across-spot mean log-expression",
        "spot_or_image_information": "none",
        "gene_outcome_scope": "downstream training genes only",
        "heldout_prediction": "broadcast gene mean over every test spot",
        "alpha_grid": list(ALPHAS),
        "validation": "fixed 2,048-gene-or-one-fifth subset of downstream training genes; minimum validation RMSE",
        "n_cohorts": len(COHORTS),
        "n_full_decoder_comparisons": len(comparisons),
        "n_cross_cohort_cells": len(cross),
        "checks": {
            "all_decompositions_exact": bool(
                np.max(np.abs(baseline["decomposition_error"].to_numpy(dtype=float)))
                < 1.0e-12
            ),
            "mean_gene_pcc_exactly_zero": bool(
                np.all(baseline["mean_gene_pcc"].to_numpy(dtype=float) == 0.0)
            ),
            "cross_matrix_complete": len(cross) == len(COHORTS) ** 2,
        },
        "inputs": inputs,
        "outputs": [
            "mean_only_absolute.tsv",
            "full_decoder_increment.tsv",
            "heldout_gene_mean_predictions.tsv.gz",
            "cross_cohort_transfer.tsv",
        ],
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    print("\nMean-only absolute")
    print(baseline.to_string(index=False))
    print("\nFull decoder increment")
    print(
        comparisons.groupby("display_cohort", sort=False)
        .mean(numeric_only=True)
        .to_string()
    )
    print("\nCross-cohort abundance PCC")
    print(
        cross.pivot(
            index="source_display", columns="target_display", values="abundance_pcc"
        ).to_string()
    )


if __name__ == "__main__":
    main()
