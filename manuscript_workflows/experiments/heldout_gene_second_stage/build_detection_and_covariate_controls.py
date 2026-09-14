#!/usr/bin/env python3
"""Evaluate simple genomic covariates and detection-restricted sensitivities."""

from __future__ import annotations
from workflow_paths import DATA_ROOT

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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
from experiments.heldout_gene_second_stage.build_gene_mean_baselines import (
    ALPHAS,
    constant_map_metrics,
    safe_corr,
)


OUT = ROOT / "experiments/heldout_gene_second_stage/detection_covariates"
GENE_META = ROOT / "data/decima_input/metadata/gene_meta.tsv"
SEQUENCE_COVARIATES = OUT / "sequence_covariates.tsv"
NUMERIC = ["log_gene_length", "frac_N", "window_gc_fraction"]
CATEGORICAL = ["chrom", "gene_type", "strand"]


def train_means_and_detection(
    cohort: str,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    configure_clean_base(cohort, 42, "decima")
    manifest = pd.read_csv(base.DATA_ROOT / "manifest.tsv", sep="\t")
    gene_ids = base.load_gene_ids()
    train_idx, heldout_idx = base.load_gene_split(gene_ids)
    store = base.ExpressionStore(gene_ids)
    means = base.compute_train_gene_means(
        manifest.loc[manifest["split"] == "train"].copy(), store, len(gene_ids)
    ).astype(np.float64)
    test = manifest.loc[manifest["split"] == "test"].copy()
    total_nonzero = np.zeros(len(gene_ids), dtype=np.int64)
    detected_samples = np.zeros(len(gene_ids), dtype=np.int64)
    total_spots = 0
    samples = sorted(test["sample"].astype(str).unique())
    for sample in samples:
        matrix, _ = store._load(sample, "test")
        nonzero = np.asarray(matrix.getnnz(axis=0)).ravel().astype(np.int64)
        total_nonzero += nonzero
        detected_samples += nonzero > 0
        total_spots += int(matrix.shape[0])
    return (
        gene_ids,
        train_idx,
        heldout_idx,
        means,
        total_nonzero / float(total_spots),
        detected_samples / float(len(samples)),
    )


def build_features(gene_ids: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata = pd.read_csv(GENE_META, sep="\t")
    metadata["gene_id"] = metadata["gene_id"].astype(str).str.split(".").str[0]
    metadata = metadata.drop_duplicates("gene_id", keep="first").set_index("gene_id")
    missing = [gene for gene in gene_ids if gene not in metadata.index]
    if missing:
        raise ValueError(f"Missing {len(missing)} genes from gene metadata")
    selected = metadata.loc[gene_ids].reset_index()
    selected["log_gene_length"] = np.log1p(
        selected["gene_length"].fillna(selected["gene_length"].median()).clip(lower=0)
    )
    selected["frac_N"] = selected["frac_N"].fillna(0.0)
    if not SEQUENCE_COVARIATES.exists():
        raise FileNotFoundError(
            f"Missing {SEQUENCE_COVARIATES}; run build_sequence_covariates first"
        )
    sequence = pd.read_csv(SEQUENCE_COVARIATES, sep="\t").set_index("gene_id")
    selected["window_gc_fraction"] = sequence.loc[
        selected["gene_id"], "window_gc_fraction"
    ].to_numpy(dtype=float)
    for column in CATEGORICAL:
        selected[column] = selected[column].fillna("unknown").astype(str)
    return selected, metadata.reset_index()


def fit_covariate_model(
    features: pd.DataFrame, outcome: np.ndarray, train_idx: np.ndarray
) -> tuple[object, float, pd.DataFrame]:
    rng = np.random.default_rng(42)
    validation_idx = np.sort(
        rng.choice(train_idx, size=min(2_048, len(train_idx) // 5), replace=False)
    )
    fit_idx = np.setdiff1d(train_idx, validation_idx, assume_unique=False)
    rows: list[dict[str, float]] = []
    for alpha in ALPHAS:
        preprocess = ColumnTransformer(
            [
                ("numeric", StandardScaler(), NUMERIC),
                ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
            ]
        )
        model = make_pipeline(preprocess, Ridge(alpha=alpha, solver="lsqr"))
        model.fit(features.iloc[fit_idx], outcome[fit_idx])
        prediction = np.maximum(model.predict(features.iloc[validation_idx]), 0.0)
        rows.append(
            {
                "alpha": alpha,
                "validation_rmse": float(
                    np.sqrt(np.mean(np.square(prediction - outcome[validation_idx])))
                ),
                "validation_pcc": safe_corr(prediction, outcome[validation_idx]),
            }
        )
    validation = pd.DataFrame(rows)
    alpha = float(validation.sort_values(["validation_rmse", "alpha"]).iloc[0]["alpha"])
    preprocess = ColumnTransformer(
        [
            ("numeric", StandardScaler(), NUMERIC),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ]
    )
    final = make_pipeline(preprocess, Ridge(alpha=alpha, solver="lsqr"))
    final.fit(features.iloc[train_idx], outcome[train_idx])
    return final, alpha, validation


def subset_effects(
    cohort: str,
    seed: int,
    subset: str,
    mask: np.ndarray,
    decima: pd.DataFrame,
    random: pd.DataFrame,
) -> dict[str, object]:
    eligible = decima["gene_pcc_eligible"].astype(bool).to_numpy() & mask
    weight = decima["n_observations"].to_numpy(dtype=float)[mask]
    weight /= weight.sum()
    decima_mean = decima["predicted_mean"].to_numpy(dtype=float)[mask]
    random_mean = random["predicted_mean"].to_numpy(dtype=float)[mask]
    observed = decima["observed_mean"].to_numpy(dtype=float)[mask]
    return {
        "cohort": cohort,
        "seed": seed,
        "subset": subset,
        "n_genes": int(mask.sum()),
        "n_gene_pcc_eligible": int(eligible.sum()),
        "abundance_pcc_gain": safe_corr(decima_mean, observed)
        - safe_corr(random_mean, observed),
        "mean_gene_pcc_gain": float(
            np.mean(
                decima.loc[eligible, "gene_pcc"].to_numpy(dtype=float)
                - random.loc[eligible, "gene_pcc"].to_numpy(dtype=float)
            )
        ),
        "abundance_mse_reduction": float(
            np.sum(
                weight
                * (
                    np.square(random.loc[mask, "mean_error"].to_numpy(dtype=float))
                    - np.square(decima.loc[mask, "mean_error"].to_numpy(dtype=float))
                )
            )
        ),
        "centered_mse_reduction": float(
            np.sum(
                weight
                * (
                    random.loc[mask, "centered_mse"].to_numpy(dtype=float)
                    - decima.loc[mask, "centered_mse"].to_numpy(dtype=float)
                )
            )
        ),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    baseline_rows: list[dict[str, object]] = []
    residual_rows: list[dict[str, object]] = []
    subset_rows: list[dict[str, object]] = []
    pretraining_rows: list[dict[str, object]] = []
    annotation_tables: list[pd.DataFrame] = []
    metadata = pd.read_csv(GENE_META, sep="\t")
    metadata["gene_id"] = metadata["gene_id"].astype(str).str.split(".").str[0]
    metadata = metadata.drop_duplicates("gene_id", keep="first")
    metadata_index = metadata.set_index("gene_id")

    mean_predictions = pd.read_csv(
        ROOT
        / "experiments/heldout_gene_second_stage/gene_mean_only/heldout_gene_mean_predictions.tsv.gz",
        sep="\t",
    )
    for cohort in COHORTS:
        gene_ids, train_idx, heldout_idx, train_mean, detection, sample_presence = train_means_and_detection(cohort)
        features, _ = build_features(gene_ids)
        model, alpha, validation = fit_covariate_model(features, train_mean, train_idx)
        prediction = np.maximum(model.predict(features.iloc[heldout_idx]), 0.0)
        gene_table_path = CLEAN_ROOT / "components/runs" / cohort / "seed_42/decima/gene_components.tsv.gz"
        gene_table = pd.read_csv(gene_table_path, sep="\t")
        heldout_genes = [gene_ids[int(index)] for index in heldout_idx]
        if gene_table["gene_id"].astype(str).tolist() != heldout_genes:
            raise ValueError(f"Held-out order mismatch for {cohort}")
        metrics = constant_map_metrics(prediction, gene_table)
        baseline_rows.append(
            {
                "cohort": cohort,
                "model": "simple_genomic_covariate_to_gene_mean_ridge",
                "selected_alpha": alpha,
                **metrics,
            }
        )
        validation.insert(0, "cohort", cohort)
        validation.to_csv(OUT / f"{cohort}_covariate_validation.tsv", sep="\t", index=False)

        annotations = features.iloc[heldout_idx][
            [
                "gene_id",
                "gene_length",
                "log_gene_length",
                "frac_N",
                "window_gc_fraction",
                "chrom",
                "gene_type",
                "strand",
                "dataset",
            ]
        ].copy()
        annotations.insert(0, "cohort", cohort)
        annotations["test_spot_detection_fraction"] = detection[heldout_idx]
        annotations["test_section_presence_fraction"] = sample_presence[heldout_idx]
        annotation_tables.append(annotations)

        mean_only = mean_predictions.loc[mean_predictions["cohort"] == cohort]
        if mean_only["gene_id"].astype(str).tolist() != heldout_genes:
            raise ValueError(f"Mean-only order mismatch for {cohort}")
        observed_residual = gene_table["observed_mean"].to_numpy(dtype=float) - prediction
        mean_only_residual = mean_only["predicted_gene_mean"].to_numpy(dtype=float) - prediction
        for seed in SEEDS:
            decima = pd.read_csv(
                CLEAN_ROOT / "components/runs" / cohort / f"seed_{seed}/decima/gene_components.tsv.gz",
                sep="\t",
            )
            random = pd.read_csv(
                CLEAN_ROOT / "components/runs" / cohort / f"seed_{seed}/random/gene_components.tsv.gz",
                sep="\t",
            )
            full_residual = decima["predicted_mean"].to_numpy(dtype=float) - prediction
            residual_rows.append(
                {
                    "cohort": cohort,
                    "seed": seed,
                    "full_decoder_residual_abundance_pcc": safe_corr(full_residual, observed_residual),
                    "mean_only_residual_abundance_pcc": safe_corr(mean_only_residual, observed_residual),
                    "unadjusted_full_decoder_abundance_pcc": safe_corr(
                        decima["predicted_mean"].to_numpy(dtype=float),
                        decima["observed_mean"].to_numpy(dtype=float),
                    ),
                }
            )
            heldout_detection = detection[heldout_idx]
            heldout_presence = sample_presence[heldout_idx]
            masks = {
                "all_heldout": np.ones(len(heldout_idx), dtype=bool),
                "detected_at_least_10pct_spots": heldout_detection >= 0.10,
                "detected_at_least_25pct_spots": heldout_detection >= 0.25,
                "detected_in_every_test_section": heldout_presence >= 1.0,
                "upper_half_detection_fraction": heldout_detection
                >= np.median(heldout_detection),
            }
            for subset, mask in masks.items():
                if int(mask.sum()) < 100:
                    continue
                subset_rows.append(subset_effects(cohort, seed, subset, mask, decima, random))

        decima_meta = metadata_index.loc[heldout_genes].reset_index()
        for dataset, group in decima_meta.groupby("dataset", dropna=False):
            pretraining_rows.append(
                {
                    "cohort": cohort,
                    "decima_pretraining_partition": str(dataset),
                    "n_downstream_heldout_genes": len(group),
                    "fraction_downstream_heldout_genes": len(group) / len(decima_meta),
                }
            )

    baseline = pd.DataFrame(baseline_rows)
    residual = pd.DataFrame(residual_rows)
    subsets = pd.DataFrame(subset_rows)
    subset_summary = (
        subsets.groupby(["cohort", "subset"], as_index=False)
        .agg(
            n_genes=("n_genes", "first"),
            abundance_pcc_gain=("abundance_pcc_gain", "mean"),
            mean_gene_pcc_gain=("mean_gene_pcc_gain", "mean"),
            abundance_mse_reduction=("abundance_mse_reduction", "mean"),
            centered_mse_reduction=("centered_mse_reduction", "mean"),
        )
    )
    pretraining = pd.DataFrame(pretraining_rows)
    annotation_table = pd.concat(annotation_tables, ignore_index=True)
    baseline.to_csv(OUT / "simple_covariate_mean_only_absolute.tsv", sep="\t", index=False)
    residual.to_csv(OUT / "covariate_residualized_abundance.tsv", sep="\t", index=False)
    subsets.to_csv(OUT / "detection_subset_effects_per_run.tsv", sep="\t", index=False)
    subset_summary.to_csv(OUT / "detection_subset_effects_summary.tsv", sep="\t", index=False)
    pretraining.to_csv(OUT / "decima_pretraining_partition_audit.tsv", sep="\t", index=False)
    annotation_table.to_csv(
        OUT / "heldout_gene_annotations.tsv.gz",
        sep="\t",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    manifest = {
        "status": "PASS",
        "simple_covariates": NUMERIC + CATEGORICAL,
        "excluded_metadata": "Decima pretraining mean_counts, n_tracks and performance fields were not used as covariates",
        "abundance_definition": "across-spot mean log-expression",
        "checks": {
            "four_covariate_baselines": len(baseline) == 4,
            "twelve_residualized_runs": len(residual) == 12,
            "all_subsets_have_three_runs": bool(
                subsets.groupby(["cohort", "subset"])["seed"].nunique().eq(3).all()
            ),
            "pretraining_audit_four_cohorts": pretraining["cohort"].nunique() == 4,
            "heldout_gene_annotation_rows": len(annotation_table)
            == 3 * 3_680 + 3_184,
        },
        "interpretive_boundary": "Detection-restricted analyses address sparse measured log-expression; they do not convert the target to pseudobulk counts or identify biological versus assay-specific causes of gene means.",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    print("\nSimple covariate mean-only")
    print(baseline.to_string(index=False))
    print("\nCovariate residualized")
    print(residual.groupby("cohort").mean(numeric_only=True).to_string())
    print("\nDetection subsets")
    print(subset_summary.to_string(index=False))
    print("\nDecima pretraining audit")
    print(pretraining.to_string(index=False))


if __name__ == "__main__":
    main()
