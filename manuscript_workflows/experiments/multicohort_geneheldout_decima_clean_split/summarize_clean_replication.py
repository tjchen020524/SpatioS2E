#!/usr/bin/env python3
"""Summarize all 36 clean-partition component-resolved runs."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from experiments.heldout_metric_policy import assert_common_eligibility
from experiments.multicohort_geneheldout_decima_clean_split.clean_split_config import (
    CLEAN_ROOT,
    COHORTS,
    SEEDS,
    VARIANTS,
)


def main() -> None:
    rows: list[dict[str, object]] = []
    missing: list[str] = []
    for cohort in COHORTS:
        for seed in SEEDS:
            for variant in VARIANTS:
                path = CLEAN_ROOT / "components/runs" / cohort / f"seed_{seed}" / variant / "summary.json"
                if not path.exists():
                    missing.append(str(path))
                else:
                    rows.append(json.loads(path.read_text()))
    if missing:
        raise FileNotFoundError("Missing clean-split runs:\n" + "\n".join(missing))
    per_run = pd.DataFrame(rows).sort_values(["cohort", "seed", "variant"])
    # Across-gene PCC is undefined for the constant-vector condition because
    # every target receives the same predicted map and hence the same mean.
    # Stored non-zero values can arise solely from chunk-level round-off.
    per_run.loc[per_run["variant"].eq("constant"), "abundance_pcc"] = np.nan
    output = CLEAN_ROOT / "results"
    output.mkdir(parents=True, exist_ok=True)
    per_run.to_csv(output / "per_run.tsv", sep="\t", index=False)

    eligibility_rows: list[dict[str, object]] = []
    for cohort in COHORTS:
        for seed in SEEDS:
            eligibility: dict[str, np.ndarray] = {}
            gene_ids: np.ndarray | None = None
            for variant in VARIANTS:
                path = (
                    CLEAN_ROOT
                    / "components/runs"
                    / cohort
                    / f"seed_{seed}"
                    / variant
                    / "gene_components.tsv.gz"
                )
                table = pd.read_csv(
                    path,
                    sep="\t",
                    usecols=["gene_id", "gene_pcc_eligible", "prediction_variable"],
                )
                current_gene_ids = table["gene_id"].astype(str).to_numpy()
                if gene_ids is None:
                    gene_ids = current_gene_ids
                elif not np.array_equal(gene_ids, current_gene_ids):
                    raise AssertionError(f"Gene order differs for {cohort}/seed_{seed}/{variant}")
                eligibility[variant] = table["gene_pcc_eligible"].astype(bool).to_numpy()
                eligibility_rows.append(
                    {
                        "cohort": cohort,
                        "seed": seed,
                        "variant": variant,
                        "n_heldout_genes": int(len(table)),
                        "n_eligible_genes": int(table["gene_pcc_eligible"].astype(bool).sum()),
                        "n_constant_prediction_genes": int(
                            (
                                table["gene_pcc_eligible"].astype(bool)
                                & ~table["prediction_variable"].astype(bool)
                            ).sum()
                        ),
                    }
                )
            assert gene_ids is not None
            assert_common_eligibility(gene_ids, eligibility)
    eligibility_frame = pd.DataFrame(eligibility_rows).sort_values(
        ["cohort", "seed", "variant"]
    )
    eligibility_frame.to_csv(output / "eligibility_audit.tsv", sep="\t", index=False)

    metrics = [
        "full_matrix_pcc", "pooled_pcc", "overall_mse", "abundance_pcc", "abundance_rmse",
        "mean_gene_pcc", "centered_full_matrix_pcc", "centered_rmse",
    ]
    effect_rows: list[dict[str, object]] = []
    for cohort in COHORTS:
        for seed in SEEDS:
            indexed = per_run.loc[(per_run["cohort"] == cohort) & (per_run["seed"] == seed)].set_index("variant")
            decima = indexed.loc["decima"]
            for control in ("random", "constant"):
                comparator = indexed.loc[control]
                row: dict[str, object] = {"cohort": cohort, "seed": seed, "control": control}
                for metric in metrics:
                    direction = -1.0 if metric.endswith("rmse") or metric == "overall_mse" else 1.0
                    output_name = f"{metric}_{'reduction' if direction < 0 else 'gain'}"
                    if metric == "abundance_pcc" and control == "constant":
                        row[output_name] = np.nan
                    else:
                        row[output_name] = direction * (
                            float(decima[metric]) - float(comparator[metric])
                        )
                total_reduction = float(comparator["overall_mse"]) - float(decima["overall_mse"])
                abundance_reduction = float(comparator["abundance_mse"]) - float(decima["abundance_mse"])
                centered_reduction = float(comparator["centered_mse"]) - float(decima["centered_mse"])
                row["abundance_mse_reduction"] = abundance_reduction
                row["centered_mse_reduction"] = centered_reduction
                row["abundance_share_percent"] = 100.0 * abundance_reduction / total_reduction
                row["centered_share_percent"] = 100.0 * centered_reduction / total_reduction
                row["abundance_error_fraction_reduced"] = (
                    abundance_reduction / float(comparator["abundance_mse"])
                )
                row["centered_error_fraction_reduced"] = (
                    centered_reduction / float(comparator["centered_mse"])
                )
                row["n_eligible_gene_pcc"] = int(decima["n_eligible_gene_pcc"])
                effect_rows.append(row)
    effects = pd.DataFrame(effect_rows).sort_values(["cohort", "control", "seed"])
    effects.to_csv(output / "paired_effects.tsv", sep="\t", index=False)
    numeric = [column for column in effects if column not in {"cohort", "seed", "control"}]
    summary = effects.groupby(["cohort", "control"], sort=False)[numeric].agg(["mean", "min", "max"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary.reset_index().to_csv(output / "paired_effects_summary.tsv", sep="\t", index=False)
    max_identity_error = float(np.abs(per_run["decomposition_error"].to_numpy(float)).max())
    manifest = {
        "n_runs": len(per_run),
        "expected_runs": len(COHORTS) * len(SEEDS) * len(VARIANTS),
        "cohorts": list(COHORTS),
        "seeds": list(SEEDS),
        "variants": list(VARIANTS),
        "gene_pcc_policy": {
            "eligibility": "observed standard deviation > 1e-6",
            "constant_prediction": "eligible gene PCC assigned zero",
            "common_denominator_check": "PASS",
            "audit_rows": len(eligibility_frame),
        },
        "gene_mean_pcc_policy": {
            "constant_vector_condition": "undefined (NA)",
            "reason": "the predicted gene means have no across-gene variance",
        },
        "max_absolute_decomposition_error": max_identity_error,
        "partition_manifest": str((CLEAN_ROOT / "partition/manifest.json")),
    }
    if max_identity_error > 1.0e-9:
        raise AssertionError(max_identity_error)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
