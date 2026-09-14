#!/usr/bin/env python3
"""Evaluate held-out-gene abundance under an alternative CPM aggregation.

The primary endpoint averages log(CPM + 1) across evaluated spots.  Here we
invert that transform spot by spot, average CPM, and then apply log1p.  This is
``log(1 + mean CPM)`` rather than ``mean log(CPM + 1)``.  We evaluate both the
independently trained gene-mean-only ridge model and the exported complete
decoder gene means against this alternative observed target.
"""

from __future__ import annotations
from workflow_paths import DATA_ROOT

import json

import numpy as np
import pandas as pd

from experiments.geneheldout_uni2h_decima_decoder_noseq_controls import (
    run_geneheldout_noseq_controls as base,
)
from experiments.heldout_gene_second_stage.build_gene_mean_baselines import (
    fit_ridge,
    safe_corr,
)
from experiments.multicohort_geneheldout_decima_clean_split.clean_split_config import (
    CLEAN_ROOT,
    COHORTS,
    ROOT,
    SEEDS,
    configure_clean_base,
)


OUT = ROOT / "experiments/heldout_gene_second_stage/alternative_abundance"


def log1p_mean_cpm(
    manifest: pd.DataFrame,
    store: base.ExpressionStore,
    n_genes: int,
) -> np.ndarray:
    """Return log1p of across-spot arithmetic mean CPM for manifest rows."""
    cpm_sum = np.zeros(n_genes, dtype=np.float64)
    n_spots = 0
    for (sample, split), group in manifest.groupby(["sample", "split"], sort=False):
        matrix, barcode_to_idx = store._load(str(sample), str(split))
        row_idx = np.asarray(
            [barcode_to_idx[str(barcode)] for barcode in group["barcode"].tolist()],
            dtype=np.int64,
        )
        subset = matrix[row_idx].copy().astype(np.float64)
        # Stored nonzero values are log(CPM + 1); implicit zeros remain zero.
        subset.data = np.expm1(subset.data)
        cpm_sum += np.asarray(subset.sum(axis=0)).ravel()
        n_spots += int(subset.shape[0])
    if n_spots == 0:
        raise ValueError("Cannot aggregate an empty manifest")
    return np.log1p(cpm_sum / float(n_spots))


def load_embeddings(
    gene_ids: list[str], train_idx: np.ndarray
) -> np.ndarray:
    cache = np.load(base.DECIMA_EMB_CACHE, allow_pickle=False)
    cache_ids = cache["gene_ids"].astype(str).tolist()
    lookup = {gene: index for index, gene in enumerate(cache_ids)}
    raw = cache["embeddings"].astype(np.float32, copy=False)[
        np.asarray([lookup[gene] for gene in gene_ids], dtype=np.int64)
    ]
    mean = raw[train_idx].mean(axis=0, keepdims=True)
    std = raw[train_idx].std(axis=0, keepdims=True)
    return ((raw - mean) / np.maximum(std, 1.0e-6)).astype(np.float32)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    target_rows: list[dict[str, object]] = []
    model_rows: list[dict[str, object]] = []

    for cohort in COHORTS:
        configure_clean_base(cohort, 42, "decima")
        manifest = pd.read_csv(base.DATA_ROOT / "manifest.tsv", sep="\t")
        gene_ids = base.load_gene_ids()
        train_idx, heldout_idx = base.load_gene_split(gene_ids)
        store = base.ExpressionStore(gene_ids)
        train_target = log1p_mean_cpm(
            manifest.loc[manifest["split"] == "train"].copy(), store, len(gene_ids)
        )
        test_target = log1p_mean_cpm(
            manifest.loc[manifest["split"] == "test"].copy(), store, len(gene_ids)
        )
        features = load_embeddings(gene_ids, train_idx)
        ridge, alpha, validation = fit_ridge(features, train_target, train_idx)
        mean_only_prediction = np.maximum(ridge.predict(features[heldout_idx]), 0.0)
        heldout_target = test_target[heldout_idx]
        heldout_genes = [gene_ids[int(index)] for index in heldout_idx]

        primary_gene_table = pd.read_csv(
            CLEAN_ROOT
            / "components/runs"
            / cohort
            / "seed_42/decima/gene_components.tsv.gz",
            sep="\t",
        )
        if primary_gene_table["gene_id"].astype(str).tolist() != heldout_genes:
            raise ValueError(f"Held-out gene order mismatch for {cohort}")
        primary_mean_log = primary_gene_table["observed_mean"].to_numpy(dtype=float)
        target_rows.append(
            {
                "cohort": cohort,
                "n_heldout_genes": len(heldout_idx),
                "mean_log_expression_vs_log_mean_cpm_pcc": safe_corr(
                    primary_mean_log, heldout_target
                ),
                "selected_ridge_alpha": alpha,
                "mean_only_log_mean_cpm_pcc": safe_corr(
                    mean_only_prediction, heldout_target
                ),
                "mean_only_log_mean_cpm_rmse": float(
                    np.sqrt(np.mean(np.square(mean_only_prediction - heldout_target)))
                ),
            }
        )
        validation.insert(0, "cohort", cohort)
        validation.to_csv(
            OUT / f"{cohort}_ridge_validation.tsv", sep="\t", index=False
        )

        for seed in SEEDS:
            tables = {
                variant: pd.read_csv(
                    CLEAN_ROOT
                    / "components/runs"
                    / cohort
                    / f"seed_{seed}/{variant}/gene_components.tsv.gz",
                    sep="\t",
                )
                for variant in ("decima", "random", "constant")
            }
            if any(
                table["gene_id"].astype(str).tolist() != heldout_genes
                for table in tables.values()
            ):
                raise ValueError(f"Component gene order mismatch for {cohort}/{seed}")
            row: dict[str, object] = {
                "cohort": cohort,
                "seed": seed,
                "n_heldout_genes": len(heldout_idx),
                "mean_only_log_mean_cpm_pcc": safe_corr(
                    mean_only_prediction, heldout_target
                ),
            }
            for variant, table in tables.items():
                row[f"{variant}_log_mean_cpm_pcc"] = safe_corr(
                    table["predicted_mean"].to_numpy(dtype=float), heldout_target
                )
            row["pretrained_minus_random_log_mean_cpm_pcc"] = float(
                row["decima_log_mean_cpm_pcc"]
            ) - float(row["random_log_mean_cpm_pcc"])
            row["pretrained_minus_constant_log_mean_cpm_pcc"] = float(
                row["decima_log_mean_cpm_pcc"]
            ) - float(row["constant_log_mean_cpm_pcc"])
            model_rows.append(row)

    targets = pd.DataFrame(target_rows)
    models = pd.DataFrame(model_rows)
    summary = (
        models.groupby("cohort", as_index=False)
        .agg(
            mean_only_log_mean_cpm_pcc=("mean_only_log_mean_cpm_pcc", "mean"),
            pretrained_log_mean_cpm_pcc=("decima_log_mean_cpm_pcc", "mean"),
            random_log_mean_cpm_pcc=("random_log_mean_cpm_pcc", "mean"),
            constant_log_mean_cpm_pcc=("constant_log_mean_cpm_pcc", "mean"),
            pretrained_minus_random=("pretrained_minus_random_log_mean_cpm_pcc", "mean"),
            pretrained_minus_constant=("pretrained_minus_constant_log_mean_cpm_pcc", "mean"),
        )
    )
    targets.to_csv(OUT / "alternative_target_summary.tsv", sep="\t", index=False)
    models.to_csv(OUT / "decoder_abundance_per_run.tsv", sep="\t", index=False)
    summary.to_csv(OUT / "decoder_abundance_summary.tsv", sep="\t", index=False)
    manifest = {
        "status": "PASS",
        "primary_abundance": "mean over spots of log(CPM + 1)",
        "alternative_abundance": "log(1 + arithmetic mean CPM across spots)",
        "checks": {
            "four_cohorts": len(targets) == 4,
            "twelve_decoder_runs": len(models) == 12,
            "finite_outputs": bool(
                np.isfinite(
                    models.select_dtypes(include=[np.number]).to_numpy(dtype=float)
                ).all()
            ),
        },
        "interpretive_boundary": (
            "The stored matrices are spot-normalized log(CPM + 1), so this is "
            "log of mean reconstructed CPM rather than a raw-count pseudobulk."
        ),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    print("\nAlternative target")
    print(targets.to_string(index=False))
    print("\nDecoder abundance")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
