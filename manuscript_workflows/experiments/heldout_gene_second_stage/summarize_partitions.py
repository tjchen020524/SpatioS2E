#!/usr/bin/env python3
"""Summarize pretrained-versus-random effects across additional gene splits."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from experiments.multicohort_geneheldout_decima_clean_split.clean_split_config import (
    COHORTS,
)
from experiments.heldout_gene_second_stage.build_gene_partitions import OUT, PARTITIONS


PARTITION_SEEDS = {
    "expression_seed123": (42,),
    "expression_seed456": (42,),
    "embedding_cluster": (42,),
    "chromosome_blocked": (42,),
}
ENDPOINTS = (
    "full_matrix_pcc",
    "overall_mse",
    "abundance_pcc",
    "abundance_rmse",
    "mean_gene_pcc",
    "centered_full_matrix_pcc",
    "centered_rmse",
    "abundance_mse",
    "centered_mse",
)


def main() -> None:
    rows: list[dict[str, object]] = []
    effects: list[dict[str, object]] = []
    for partition in PARTITIONS:
        root = OUT / partition
        for cohort in COHORTS:
            for seed in PARTITION_SEEDS[partition]:
                summaries: dict[str, dict[str, object]] = {}
                for variant in ("decima", "random"):
                    path = root / "components/runs" / cohort / f"seed_{seed}" / variant / "summary.json"
                    if not path.exists():
                        raise FileNotFoundError(path)
                    summary = json.loads(path.read_text())
                    summaries[variant] = summary
                    rows.append(
                        {
                            "partition": partition,
                            "cohort": cohort,
                            "seed": seed,
                            "variant": "pretrained" if variant == "decima" else variant,
                            "n_heldout_genes": int(summary["n_heldout_genes"]),
                            "n_eligible_gene_pcc": int(summary["n_eligible_gene_pcc"]),
                            **{endpoint: float(summary[endpoint]) for endpoint in ENDPOINTS},
                        }
                    )
                for endpoint in ENDPOINTS:
                    candidate = float(summaries["decima"][endpoint])
                    control = float(summaries["random"][endpoint])
                    value = control - candidate if endpoint.endswith("mse") or endpoint.endswith("rmse") else candidate - control
                    effects.append(
                        {
                            "partition": partition,
                            "cohort": cohort,
                            "seed": seed,
                            "endpoint": endpoint,
                            "effect": value,
                        }
                    )
    absolute = pd.DataFrame(rows)
    effect_table = pd.DataFrame(effects)
    summary = (
        effect_table.groupby(["partition", "cohort", "endpoint"], as_index=False)
        .agg(
            mean_effect=("effect", "mean"),
            min_effect=("effect", "min"),
            max_effect=("effect", "max"),
            n_runs=("effect", "size"),
            n_positive=("effect", lambda values: int((values > 0).sum())),
        )
    )
    absolute.to_csv(OUT / "partition_absolute_per_run.tsv", sep="\t", index=False)
    effect_table.to_csv(OUT / "partition_effects_per_run.tsv", sep="\t", index=False)
    summary.to_csv(OUT / "partition_effects_summary.tsv", sep="\t", index=False)
    manifest = {
        "status": "PASS",
        "n_absolute_rows": len(absolute),
        "n_effect_rows": len(effect_table),
        "checks": {
            "absolute_complete": len(absolute)
            == 4 * 2 * sum(len(PARTITION_SEEDS[name]) for name in PARTITIONS),
            "effects_complete": len(effect_table)
            == 4
            * len(ENDPOINTS)
            * sum(len(PARTITION_SEEDS[name]) for name in PARTITIONS),
            "common_denominator_within_run": bool(
                absolute.groupby(["partition", "cohort", "seed"])["n_eligible_gene_pcc"].nunique().eq(1).all()
            ),
        },
        "partition_analysis_seeds": {
            name: list(seeds) for name, seeds in PARTITION_SEEDS.items()
        },
        "interpretive_boundary": "Each additional target partition uses one paired seed-42 sensitivity run. These analyses test robustness of the large abundance effect to target assignment; they do not precisely establish the magnitude of the much smaller within-gene effect and are not biological replicates.",
    }
    (OUT / "results_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    print(
        summary.loc[
            summary["endpoint"].isin(
                ["full_matrix_pcc", "abundance_pcc", "mean_gene_pcc", "centered_rmse"]
            )
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
