#!/usr/bin/env python3
"""Audit training/held-out balance under the primary downstream gene split."""

from __future__ import annotations
from workflow_paths import DATA_ROOT

import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.geneheldout_uni2h_decima_decoder_noseq_controls import (
    run_geneheldout_noseq_controls as base,
)
from experiments.heldout_gene_second_stage.build_detection_and_covariate_controls import (
    build_features,
)
from experiments.multicohort_geneheldout_decima_clean_split.clean_split_config import (
    COHORTS,
    ROOT,
    configure_clean_base,
)


OUT = ROOT / "experiments/heldout_gene_second_stage/partition_balance"
SOURCE = ROOT / "paper/submission/source_data"
NUMERIC_FEATURES = (
    "training_mean_log_expression",
    "training_spot_std",
    "log_gene_length",
    "window_gc_fraction",
    "standardized_pretrained_vector_norm",
)
CATEGORICAL_FEATURES = ("gene_type", "chrom")


def training_moments(
    manifest: pd.DataFrame, store: base.ExpressionStore, n_genes: int
) -> tuple[np.ndarray, np.ndarray, int]:
    total = np.zeros(n_genes, dtype=np.float64)
    total_sq = np.zeros(n_genes, dtype=np.float64)
    count = 0
    for sample in sorted(manifest["sample"].astype(str).unique()):
        matrix, _ = store._load(sample, "train")
        total += np.asarray(matrix.sum(axis=0)).ravel()
        total_sq += np.asarray(matrix.power(2).sum(axis=0)).ravel()
        count += int(matrix.shape[0])
    mean = total / count
    variance = np.maximum(total_sq / count - np.square(mean), 0.0)
    return mean, np.sqrt(variance), count


def numeric_summary(table: pd.DataFrame, cohort: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for feature in NUMERIC_FEATURES:
        train = table.loc[table["assignment"] == "training", feature].to_numpy(float)
        heldout = table.loc[table["assignment"] == "heldout", feature].to_numpy(float)
        pooled_sd = np.sqrt((np.var(train, ddof=1) + np.var(heldout, ddof=1)) / 2.0)
        standardized_difference = (
            (float(np.mean(heldout)) - float(np.mean(train))) / pooled_sd
            if pooled_sd > 0
            else 0.0
        )
        for assignment, values in (("training", train), ("heldout", heldout)):
            rows.append(
                {
                    "cohort": cohort,
                    "feature": feature,
                    "assignment": assignment,
                    "n_genes": len(values),
                    "mean": float(np.mean(values)),
                    "standard_deviation": float(np.std(values, ddof=1)),
                    "q05": float(np.quantile(values, 0.05)),
                    "median": float(np.median(values)),
                    "q95": float(np.quantile(values, 0.95)),
                    "heldout_minus_training_standardized_difference": standardized_difference,
                }
            )
    return rows


def categorical_summary(
    table: pd.DataFrame, cohort: str
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    distance_rows: list[dict[str, object]] = []
    for feature in CATEGORICAL_FEATURES:
        categories = sorted(table[feature].fillna("unknown").astype(str).unique())
        proportions: dict[str, np.ndarray] = {}
        for assignment in ("training", "heldout"):
            values = table.loc[table["assignment"] == assignment, feature].fillna("unknown").astype(str)
            counts = values.value_counts().reindex(categories, fill_value=0)
            proportions[assignment] = counts.to_numpy(float) / len(values)
            for category, count, proportion in zip(
                categories, counts.to_numpy(int), proportions[assignment]
            ):
                rows.append(
                    {
                        "cohort": cohort,
                        "feature": feature,
                        "category": category,
                        "assignment": assignment,
                        "n_genes": int(count),
                        "proportion": float(proportion),
                    }
                )
        difference = proportions["heldout"] - proportions["training"]
        distance_rows.append(
            {
                "cohort": cohort,
                "feature": feature,
                "total_variation_distance": float(0.5 * np.abs(difference).sum()),
                "maximum_absolute_proportion_difference": float(np.abs(difference).max()),
            }
        )
    return rows, distance_rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    SOURCE.mkdir(parents=True, exist_ok=True)
    numeric_rows: list[dict[str, object]] = []
    categorical_rows: list[dict[str, object]] = []
    distance_rows: list[dict[str, object]] = []
    per_gene_tables: list[pd.DataFrame] = []
    spot_counts: dict[str, int] = {}

    for cohort in COHORTS:
        configure_clean_base(cohort, 42, "decima")
        gene_ids = base.load_gene_ids()
        train_idx, heldout_idx = base.load_gene_split(gene_ids)
        assignment = np.full(len(gene_ids), "training", dtype=object)
        assignment[heldout_idx] = "heldout"

        manifest = pd.read_csv(base.DATA_ROOT / "manifest.tsv", sep="\t")
        train_manifest = manifest.loc[manifest["split"] == "train"].copy()
        mean, standard_deviation, n_spots = training_moments(
            train_manifest, base.ExpressionStore(gene_ids), len(gene_ids)
        )
        spot_counts[cohort] = n_spots

        features, _ = build_features(gene_ids)
        vectors = base.load_decima_embeddings(gene_ids, train_idx)
        table = features[
            ["gene_id", "log_gene_length", "window_gc_fraction", "gene_type", "chrom"]
        ].copy()
        table.insert(0, "cohort", cohort)
        table["assignment"] = assignment
        table["training_mean_log_expression"] = mean
        table["training_spot_std"] = standard_deviation
        table["standardized_pretrained_vector_norm"] = np.linalg.norm(vectors, axis=1)
        per_gene_tables.append(table)
        numeric_rows.extend(numeric_summary(table, cohort))
        categorical, distances = categorical_summary(table, cohort)
        categorical_rows.extend(categorical)
        distance_rows.extend(distances)

    numeric = pd.DataFrame(numeric_rows)
    categorical = pd.DataFrame(categorical_rows)
    distances = pd.DataFrame(distance_rows)
    per_gene = pd.concat(per_gene_tables, ignore_index=True)

    outputs = {
        "numeric": OUT / "primary_partition_numeric_balance.tsv",
        "categorical": OUT / "primary_partition_categorical_balance.tsv",
        "categorical_distance": OUT / "primary_partition_categorical_distance.tsv",
        "per_gene": OUT / "primary_partition_per_gene.tsv.gz",
    }
    numeric.to_csv(outputs["numeric"], sep="\t", index=False)
    categorical.to_csv(outputs["categorical"], sep="\t", index=False)
    distances.to_csv(outputs["categorical_distance"], sep="\t", index=False)
    per_gene.to_csv(
        outputs["per_gene"],
        sep="\t",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )

    for key in ("numeric", "categorical", "categorical_distance"):
        target = SOURCE / f"supp_stage3_partition_balance_{key}.tsv"
        pd.read_csv(outputs[key], sep="\t").to_csv(target, sep="\t", index=False)

    manifest_payload = {
        "analysis": "primary downstream gene-partition balance",
        "cohorts": list(COHORTS),
        "training_spots": spot_counts,
        "numeric_features": list(NUMERIC_FEATURES),
        "categorical_features": list(CATEGORICAL_FEATURES),
        "outputs": {key: str(path.relative_to(ROOT)) for key, path in outputs.items()},
        "interpretation": (
            "Descriptive balance audit only; the split was expression-stratified "
            "and was not optimized to balance every annotation."
        ),
    }
    with (OUT / "manifest.json").open("w") as handle:
        json.dump(manifest_payload, handle, indent=2)

    max_numeric = numeric[
        "heldout_minus_training_standardized_difference"
    ].abs().max()
    print(f"Wrote {len(per_gene):,} per-gene rows")
    print(f"Maximum absolute standardized numeric difference: {max_numeric:.4f}")
    print(distances.to_string(index=False))


if __name__ == "__main__":
    main()
