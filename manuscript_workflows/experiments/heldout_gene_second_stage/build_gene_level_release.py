#!/usr/bin/env python3
"""Build a per-gene, per-run release table for the primary held-out assay."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

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


SECOND = ROOT / "experiments/heldout_gene_second_stage"
SOURCE = ROOT / "paper/submission/source_data"
CONDITIONS = {"decima": "pretrained", "random": "random", "constant": "constant"}
METRICS = [
    "predicted_mean",
    "mse",
    "centered_mse",
    "gene_pcc",
    "predicted_std",
    "prediction_variable",
]


def main() -> None:
    annotations = pd.read_csv(
        SECOND / "detection_covariates/heldout_gene_annotations.tsv.gz", sep="\t"
    )
    similarity = pd.read_csv(
        SECOND / "gene_partitions/nearest_training_similarity_per_gene.tsv.gz", sep="\t"
    )
    similarity = similarity.loc[
        similarity["partition"] == "primary_expression_seed42",
        ["gene_id", "nearest_training_cosine"],
    ]
    tables: list[pd.DataFrame] = []
    for cohort in COHORTS:
        configure_clean_base(cohort, 42, "decima")
        gene_ids = base.load_gene_ids()
        train_idx, heldout_idx = base.load_gene_split(gene_ids)
        embeddings = base.load_decima_embeddings(gene_ids, train_idx)
        heldout_genes = [gene_ids[int(index)] for index in heldout_idx]
        fixed = annotations.loc[annotations["cohort"] == cohort].copy()
        if fixed["gene_id"].astype(str).tolist() != heldout_genes:
            fixed = fixed.set_index("gene_id").loc[heldout_genes].reset_index()
        fixed["standardized_pretrained_vector_norm"] = np.linalg.norm(
            embeddings[heldout_idx], axis=1
        )
        fixed = fixed.merge(similarity, on="gene_id", how="left", validate="one_to_one")
        for seed in SEEDS:
            merged = fixed.copy()
            reference: pd.DataFrame | None = None
            for variant, display in CONDITIONS.items():
                component = pd.read_csv(
                    CLEAN_ROOT
                    / "components/runs"
                    / cohort
                    / f"seed_{seed}"
                    / variant
                    / "gene_components.tsv.gz",
                    sep="\t",
                )
                if component["gene_id"].astype(str).tolist() != heldout_genes:
                    raise ValueError(f"Gene order mismatch for {cohort}/{seed}/{variant}")
                if reference is None:
                    reference = component[
                        [
                            "gene_id",
                            "n_observations",
                            "observed_mean",
                            "observed_std",
                            "gene_pcc_eligible",
                        ]
                    ].copy()
                    merged = merged.merge(
                        reference, on="gene_id", how="left", validate="one_to_one"
                    )
                selected = component[["gene_id", *METRICS]].rename(
                    columns={metric: f"{display}_{metric}" for metric in METRICS}
                )
                merged = merged.merge(
                    selected, on="gene_id", how="left", validate="one_to_one"
                )
            merged.insert(1, "seed", seed)
            merged["pretrained_minus_random_gene_pcc"] = (
                merged["pretrained_gene_pcc"] - merged["random_gene_pcc"]
            )
            merged["pretrained_vs_random_mse_reduction"] = (
                merged["random_mse"] - merged["pretrained_mse"]
            )
            merged["pretrained_vs_random_centered_mse_reduction"] = (
                merged["random_centered_mse"] - merged["pretrained_centered_mse"]
            )
            tables.append(merged)
    release = pd.concat(tables, ignore_index=True)
    expected = 3 * (3 * 3_680 + 3_184)
    if len(release) != expected or release["gene_id"].isna().any():
        raise ValueError(f"Release table incomplete: {len(release)} rows, expected {expected}")
    SOURCE.mkdir(parents=True, exist_ok=True)
    output = SOURCE / "heldout_gene_benchmark_per_gene_per_run.tsv.gz"
    release.to_csv(
        output,
        sep="\t",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    print(f"Wrote {output} ({len(release):,} rows, {len(release.columns)} columns)")


if __name__ == "__main__":
    main()
