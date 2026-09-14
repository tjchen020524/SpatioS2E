#!/usr/bin/env python3
"""Consolidate completed CPU-only second-stage experiments and audit checks."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.multicohort_geneheldout_decima_clean_split.clean_split_config import ROOT


BASE = ROOT / "experiments/heldout_gene_second_stage"
OUT = BASE / "summary"
SOURCE = ROOT / "paper/submission/source_data"


def add_rows(
    rows: list[dict[str, object]],
    analysis: str,
    table: pd.DataFrame,
    cohort_column: str,
    metrics: list[str],
    source: str,
) -> None:
    for _, row in table.iterrows():
        for metric in metrics:
            rows.append(
                {
                    "analysis": analysis,
                    "cohort": row[cohort_column],
                    "endpoint": metric,
                    "value": float(row[metric]),
                    "source": source,
                }
            )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    SOURCE.mkdir(parents=True, exist_ok=True)
    key_rows: list[dict[str, object]] = []

    mean_abs_path = BASE / "gene_mean_only/mean_only_absolute.tsv"
    increment_path = BASE / "gene_mean_only/full_decoder_increment.tsv"
    cross_path = BASE / "gene_mean_only/cross_cohort_transfer.tsv"
    mean_abs = pd.read_csv(mean_abs_path, sep="\t")
    increment = pd.read_csv(increment_path, sep="\t")
    cross = pd.read_csv(cross_path, sep="\t")
    add_rows(
        key_rows,
        "independent_mean_only_absolute",
        mean_abs,
        "display_cohort",
        ["abundance_pcc", "full_matrix_pcc", "overall_mse", "mean_gene_pcc"],
        str(mean_abs_path.relative_to(ROOT)),
    )
    increment_mean = increment.groupby("display_cohort", as_index=False).mean(numeric_only=True)
    add_rows(
        key_rows,
        "independent_mean_only_vs_complete",
        increment_mean,
        "display_cohort",
        [
            "full_minus_mean_only_full_matrix_pcc",
            "mean_only_minus_full_overall_mse",
            "mean_only_vs_full_predicted_gene_mean_pcc",
            "mean_only_fraction_full_matrix_pcc_gain_vs_random",
            "mean_only_fraction_overall_mse_reduction_vs_random",
            "mean_only_fraction_abundance_pcc_gain_vs_random",
        ],
        str(increment_path.relative_to(ROOT)),
    )

    alt_path = BASE / "alternative_abundance/decoder_abundance_summary.tsv"
    alt = pd.read_csv(alt_path, sep="\t")
    add_rows(
        key_rows,
        "alternative_log_mean_cpm",
        alt,
        "cohort",
        ["mean_only_log_mean_cpm_pcc", "pretrained_log_mean_cpm_pcc", "pretrained_minus_random"],
        str(alt_path.relative_to(ROOT)),
    )

    detection_path = BASE / "detection_covariates/detection_subset_effects_summary.tsv"
    detection = pd.read_csv(detection_path, sep="\t")
    detected_25 = detection.loc[detection["subset"] == "detected_at_least_25pct_spots"]
    add_rows(
        key_rows,
        "detected_at_least_25pct",
        detected_25,
        "cohort",
        ["abundance_pcc_gain", "mean_gene_pcc_gain"],
        str(detection_path.relative_to(ROOT)),
    )

    covariate_path = BASE / "detection_covariates/simple_covariate_mean_only_absolute.tsv"
    covariate = pd.read_csv(covariate_path, sep="\t")
    add_rows(
        key_rows,
        "simple_genomic_covariate_mean_only",
        covariate,
        "cohort",
        ["abundance_pcc", "full_matrix_pcc"],
        str(covariate_path.relative_to(ROOT)),
    )
    residual_path = BASE / "detection_covariates/covariate_residualized_abundance.tsv"
    residual_per_run = pd.read_csv(residual_path, sep="\t")
    residual = residual_per_run.groupby("cohort", as_index=False).mean(numeric_only=True)
    add_rows(
        key_rows,
        "covariate_residualized_abundance",
        residual,
        "cohort",
        ["full_decoder_residual_abundance_pcc", "mean_only_residual_abundance_pcc"],
        str(residual_path.relative_to(ROOT)),
    )

    individual_path = BASE / "biological_uncertainty/individual_block_bootstrap_summary.tsv"
    individual = pd.read_csv(individual_path, sep="\t")
    for _, row in individual.iterrows():
        for endpoint in ("estimate", "ci_low", "ci_high"):
            key_rows.append(
                {
                    "analysis": f"individual_block_bootstrap_{row['metric']}",
                    "cohort": row["display_cohort"],
                    "endpoint": endpoint,
                    "value": float(row[endpoint]),
                    "source": str(individual_path.relative_to(ROOT)),
                }
            )

    similarity_path = BASE / "similarity_stratified/effects_summary.tsv"
    similarity = pd.read_csv(similarity_path, sep="\t")
    least_similar = similarity.loc[similarity["similarity_stratum"] == "Q1 least similar"]
    add_rows(
        key_rows,
        "least_similar_gene_quartile",
        least_similar,
        "cohort",
        ["similarity_mean", "abundance_pcc_gain", "mean_gene_pcc_gain"],
        str(similarity_path.relative_to(ROOT)),
    )

    key = pd.DataFrame(key_rows)
    key.to_csv(OUT / "offline_key_results.tsv", sep="\t", index=False)
    key.to_csv(SOURCE / "supp_stage2_offline_key_results.tsv", sep="\t", index=False)

    pretrain_fold = pd.read_csv(
        BASE / "detection_covariates/decima_pretraining_partition_audit.tsv", sep="\t"
    )
    direct_overlap = pd.read_csv(
        BASE / "decima_pretraining_audit/exact_downstream_overlap.tsv", sep="\t"
    )
    related = pd.read_csv(
        BASE / "decima_pretraining_audit/related_tissue_supervision.tsv", sep="\t"
    )

    # Publication-facing exports retain the complete summaries behind textual
    # sensitivity claims, not only the compact key-value audit above.
    covariate.to_csv(
        SOURCE / "supp_stage2_simple_genomic_covariates.tsv", sep="\t", index=False
    )
    residual_per_run.to_csv(
        SOURCE / "supp_stage2_covariate_residualized_abundance.tsv",
        sep="\t",
        index=False,
    )
    pd.read_csv(
        BASE / "biological_uncertainty/hierarchical_section_bootstrap_summary.tsv",
        sep="\t",
    ).to_csv(
        SOURCE / "supp_stage2_biological_hierarchical_bootstrap.tsv",
        sep="\t",
        index=False,
    )
    pd.read_csv(
        BASE / "biological_uncertainty/leave_one_individual_out.tsv", sep="\t"
    ).to_csv(
        SOURCE / "supp_stage2_biological_leave_one_out.tsv", sep="\t", index=False
    )
    pd.read_csv(
        BASE / "biological_uncertainty/gene_cluster_bootstrap_summary.tsv", sep="\t"
    ).to_csv(
        SOURCE / "supp_stage2_gene_cluster_bootstrap.tsv", sep="\t", index=False
    )
    pretrain_fold.to_csv(
        SOURCE / "supp_stage2_decima_pretraining_gene_partitions.tsv",
        sep="\t",
        index=False,
    )
    direct_overlap.to_csv(
        SOURCE / "supp_stage2_decima_pretraining_exact_identifier_audit.tsv",
        sep="\t",
        index=False,
    )
    related.to_csv(
        SOURCE / "supp_stage2_decima_related_tissue_supervision.tsv",
        sep="\t",
        index=False,
    )
    partition_manifest = json.loads((BASE / "gene_partitions/manifest.json").read_text())
    source_manifests = [
        BASE / "gene_mean_only/manifest.json",
        BASE / "alternative_abundance/manifest.json",
        BASE / "detection_covariates/manifest.json",
        BASE / "biological_uncertainty/manifest.json",
        BASE / "decima_pretraining_audit/manifest.json",
        BASE / "gene_partitions/manifest.json",
        BASE / "similarity_stratified/manifest.json",
    ]
    manifest = {
        "status": "PASS",
        "checks": {
            "all_source_manifests_pass": all(
                json.loads(path.read_text()).get("status") == "PASS"
                for path in source_manifests
            ),
            "four_mean_only_cohorts": len(mean_abs) == 4,
            "twelve_complete_decoder_comparisons": len(increment) == 12,
            "mean_only_has_zero_within_gene_pcc": bool(
                np.allclose(mean_abs["mean_gene_pcc"], 0.0)
            ),
            "alternative_abundance_effect_positive": bool(
                (alt["pretrained_minus_random"] > 0).all()
            ),
            "detected_subset_abundance_exceeds_gene_pcc": bool(
                (detected_25["abundance_pcc_gain"] > detected_25["mean_gene_pcc_gain"]).all()
            ),
            "no_exact_downstream_identifier_in_packaged_decima_metadata": bool(
                (~direct_overlap["exact_identifier_detected"]).all()
            ),
            "related_tissue_supervision_present": bool(
                (related["n_related_decima_supervision_rows"] > 0).all()
            ),
            "additional_gene_partitions_pass": partition_manifest.get("status") == "PASS",
            "least_similar_quartile_abundance_positive": bool(
                (least_similar["abundance_pcc_gain"] > 0).all()
            ),
        },
        "key_ranges": {
            "mean_only_full_matrix_pcc": [
                float(mean_abs["full_matrix_pcc"].min()),
                float(mean_abs["full_matrix_pcc"].max()),
            ],
            "mean_only_full_matrix_gain_retained_percent": [
                100 * float(increment_mean["mean_only_fraction_full_matrix_pcc_gain_vs_random"].min()),
                100 * float(increment_mean["mean_only_fraction_full_matrix_pcc_gain_vs_random"].max()),
            ],
            "alternative_abundance_pretrained_minus_random": [
                float(alt["pretrained_minus_random"].min()),
                float(alt["pretrained_minus_random"].max()),
            ],
            "cross_cohort_cells": len(cross),
            "decima_pretraining_train_fold_fraction": [
                float(
                    pretrain_fold.loc[
                        pretrain_fold["decima_pretraining_partition"] == "train",
                        "fraction_downstream_heldout_genes",
                    ].min()
                ),
                float(
                    pretrain_fold.loc[
                        pretrain_fold["decima_pretraining_partition"] == "train",
                        "fraction_downstream_heldout_genes",
                    ].max()
                ),
            ],
        },
        "source_manifests": [str(path.relative_to(ROOT)) for path in source_manifests],
    }
    (OUT / "offline_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
