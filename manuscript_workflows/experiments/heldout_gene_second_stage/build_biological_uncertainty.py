#!/usr/bin/env python3
"""Quantify biological-unit and gene-cluster uncertainty for primary effects."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.multicohort_geneheldout_decima_clean_split.clean_split_config import (
    CLEAN_ROOT,
    COHORTS,
    ROOT,
    SEEDS,
)


OUT = ROOT / "experiments/heldout_gene_second_stage/biological_uncertainty"
SECTION_ROOT = CLEAN_ROOT / "section_centered"
CLUSTER_ASSIGNMENT = (
    ROOT
    / "experiments/heldout_gene_second_stage/gene_partitions/embedding_cluster/master_assignment.tsv.gz"
)
N_BOOTSTRAP = 20_000
DISPLAY = {
    "hippocampus_donor_disjoint": "Hippocampus",
    "dlpfc": "DLPFC",
    "nac": "NAc",
    "her2st": "HER2ST",
}
EFFECT_SPECS = {
    "abundance_pcc_gain": ("abundance_pcc", 1.0),
    "within_section_gene_pcc_gain": ("mean_within_section_gene_pcc", 1.0),
    "section_centered_full_matrix_pcc_gain": ("centered_full_matrix_pcc", 1.0),
    "abundance_rmse_reduction": ("abundance_rmse", -1.0),
    "section_centered_rmse_reduction": ("section_centered_rmse", -1.0),
    "total_mse_reduction": ("total_mse", -1.0),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def paired_effects(table: pd.DataFrame, unit_columns: list[str]) -> pd.DataFrame:
    index = ["cohort", "seed", *unit_columns]
    decima = table.loc[table["variant"] == "decima"].set_index(index)
    random = table.loc[table["variant"] == "random"].set_index(index)
    if not decima.index.equals(random.index):
        missing_decima = random.index.difference(decima.index)
        missing_random = decima.index.difference(random.index)
        raise ValueError(
            f"Unpaired biological units: decima={len(missing_decima)} random={len(missing_random)}"
        )
    output = decima.reset_index()[index].copy()
    for output_name, (metric, direction) in EFFECT_SPECS.items():
        output[output_name] = direction * (
            decima[metric].to_numpy(dtype=float)
            - random[metric].to_numpy(dtype=float)
        )
    return output


def collapse_runs(effect: pd.DataFrame, unit_columns: list[str]) -> pd.DataFrame:
    columns = list(EFFECT_SPECS)
    collapsed = (
        effect.groupby(["cohort", *unit_columns], as_index=False)[columns]
        .mean()
    )
    collapsed["n_technical_runs_collapsed"] = len(SEEDS)
    return collapsed


def block_bootstrap(
    individual_effect: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, object]] = []
    draw_rows: list[dict[str, object]] = []
    for cohort in COHORTS:
        cohort_table = individual_effect.loc[individual_effect["cohort"] == cohort]
        individuals = cohort_table["individual"].astype(str).tolist()
        values = cohort_table[list(EFFECT_SPECS)].to_numpy(dtype=float)
        rng = np.random.default_rng(23_081 + COHORTS.index(cohort))
        draw_index = rng.integers(0, len(individuals), size=(N_BOOTSTRAP, len(individuals)))
        bootstrap = values[draw_index].mean(axis=1)
        for metric_index, metric in enumerate(EFFECT_SPECS):
            draws = bootstrap[:, metric_index]
            summary_rows.append(
                {
                    "cohort": cohort,
                    "display_cohort": DISPLAY[cohort],
                    "metric": metric,
                    "n_biological_individuals": len(individuals),
                    "estimate": float(np.mean(values[:, metric_index])),
                    "bootstrap_median": float(np.median(draws)),
                    "ci_low": float(np.quantile(draws, 0.025)),
                    "ci_high": float(np.quantile(draws, 0.975)),
                    "fraction_positive": float(np.mean(draws > 0)),
                    "uncertainty_unit": "biological individual; technical runs averaged first",
                }
            )
            draw_rows.extend(
                {
                    "cohort": cohort,
                    "metric": metric,
                    "draw": draw,
                    "effect": float(value),
                }
                for draw, value in enumerate(draws.tolist())
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(draw_rows)


def hierarchical_section_bootstrap(section_effect: pd.DataFrame) -> pd.DataFrame:
    collapsed = collapse_runs(section_effect, ["sample", "individual"])
    rows: list[dict[str, object]] = []
    for cohort in COHORTS:
        cohort_table = collapsed.loc[collapsed["cohort"] == cohort].copy()
        individuals = sorted(cohort_table["individual"].astype(str).unique())
        by_individual = {
            individual: cohort_table.loc[
                cohort_table["individual"].astype(str) == individual,
                list(EFFECT_SPECS),
            ].to_numpy(dtype=float)
            for individual in individuals
        }
        rng = np.random.default_rng(91_103 + COHORTS.index(cohort))
        draws = np.empty((N_BOOTSTRAP, len(EFFECT_SPECS)), dtype=np.float64)
        for draw in range(N_BOOTSTRAP):
            sampled_individuals = rng.choice(individuals, size=len(individuals), replace=True)
            individual_means: list[np.ndarray] = []
            for individual in sampled_individuals:
                section_values = by_individual[str(individual)]
                selected = rng.integers(0, len(section_values), size=len(section_values))
                individual_means.append(section_values[selected].mean(axis=0))
            draws[draw] = np.mean(individual_means, axis=0)
        point = np.mean(
            [values.mean(axis=0) for values in by_individual.values()], axis=0
        )
        for metric_index, metric in enumerate(EFFECT_SPECS):
            rows.append(
                {
                    "cohort": cohort,
                    "display_cohort": DISPLAY[cohort],
                    "metric": metric,
                    "n_biological_individuals": len(individuals),
                    "n_sections": len(cohort_table),
                    "estimate": float(point[metric_index]),
                    "ci_low": float(np.quantile(draws[:, metric_index], 0.025)),
                    "ci_high": float(np.quantile(draws[:, metric_index], 0.975)),
                    "fraction_positive": float(np.mean(draws[:, metric_index] > 0)),
                    "uncertainty_unit": "individual block, sections nested within individual; technical runs averaged first",
                }
            )
    return pd.DataFrame(rows)


def leave_one_individual_out(individual_effect: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for cohort in COHORTS:
        cohort_table = individual_effect.loc[individual_effect["cohort"] == cohort]
        individuals = cohort_table["individual"].astype(str).tolist()
        for omitted in individuals:
            retained = cohort_table.loc[cohort_table["individual"].astype(str) != omitted]
            for metric in EFFECT_SPECS:
                rows.append(
                    {
                        "cohort": cohort,
                        "display_cohort": DISPLAY[cohort],
                        "omitted_individual": omitted,
                        "n_retained_individuals": len(retained),
                        "metric": metric,
                        "effect": float(retained[metric].mean()),
                    }
                )
    return pd.DataFrame(rows)


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 2:
        return float("nan")
    return float(np.corrcoef(x[mask], y[mask])[0, 1])


def cluster_bootstrap() -> tuple[pd.DataFrame, pd.DataFrame]:
    cluster = pd.read_csv(CLUSTER_ASSIGNMENT, sep="\t")
    cluster = cluster[["gene_id", "embedding_cluster"]]
    summary_rows: list[dict[str, object]] = []
    run_rows: list[dict[str, object]] = []
    for cohort in COHORTS:
        seed_tables: list[pd.DataFrame] = []
        for seed in SEEDS:
            decima = pd.read_csv(
                CLEAN_ROOT
                / "components/runs"
                / cohort
                / f"seed_{seed}/decima/gene_components.tsv.gz",
                sep="\t",
            )
            random = pd.read_csv(
                CLEAN_ROOT
                / "components/runs"
                / cohort
                / f"seed_{seed}/random/gene_components.tsv.gz",
                sep="\t",
            )
            columns = [
                "gene_id",
                "n_observations",
                "observed_mean",
                "observed_std",
                "predicted_mean",
                "mean_error",
                "centered_mse",
                "gene_pcc",
                "gene_pcc_eligible",
            ]
            merged = decima[columns].merge(
                random[columns], on="gene_id", suffixes=("_pretrained", "_random")
            ).merge(cluster, on="gene_id", how="left", validate="one_to_one")
            if merged["embedding_cluster"].isna().any():
                raise ValueError(f"Missing embedding cluster for {cohort}/{seed}")
            merged["seed"] = seed
            seed_tables.append(merged)
        unique_clusters = np.sort(seed_tables[0]["embedding_cluster"].unique())
        rng = np.random.default_rng(72_001 + COHORTS.index(cohort))
        bootstrap = {
            "abundance_pcc_gain": np.empty(N_BOOTSTRAP),
            "mean_gene_pcc_gain": np.empty(N_BOOTSTRAP),
            "abundance_mse_reduction": np.empty(N_BOOTSTRAP),
            "centered_mse_reduction": np.empty(N_BOOTSTRAP),
        }
        prepared: list[dict[str, object]] = []
        for table in seed_tables:
            cluster_values = table["embedding_cluster"].to_numpy(dtype=np.int64)
            prepared.append(
                {
                    "groups": {
                        int(cluster_id): np.flatnonzero(cluster_values == cluster_id)
                        for cluster_id in unique_clusters
                    },
                    "weight": table["n_observations_pretrained"].to_numpy(dtype=float),
                    "observed_pretrained": table["observed_mean_pretrained"].to_numpy(dtype=float),
                    "observed_random": table["observed_mean_random"].to_numpy(dtype=float),
                    "predicted_pretrained": table["predicted_mean_pretrained"].to_numpy(dtype=float),
                    "predicted_random": table["predicted_mean_random"].to_numpy(dtype=float),
                    "mean_error_pretrained": table["mean_error_pretrained"].to_numpy(dtype=float),
                    "mean_error_random": table["mean_error_random"].to_numpy(dtype=float),
                    "centered_mse_pretrained": table["centered_mse_pretrained"].to_numpy(dtype=float),
                    "centered_mse_random": table["centered_mse_random"].to_numpy(dtype=float),
                    "gene_pcc_pretrained": table["gene_pcc_pretrained"].to_numpy(dtype=float),
                    "gene_pcc_random": table["gene_pcc_random"].to_numpy(dtype=float),
                    "eligible": table["gene_pcc_eligible_pretrained"].astype(bool).to_numpy(),
                }
            )
        for draw in range(N_BOOTSTRAP):
            sampled_clusters = rng.choice(
                unique_clusters, size=len(unique_clusters), replace=True
            )
            seed_effects: list[dict[str, float]] = []
            for seed_data in prepared:
                groups: dict[int, np.ndarray] = seed_data["groups"]  # type: ignore[assignment]
                sample_index = np.concatenate(
                    [groups[int(cluster_id)] for cluster_id in sampled_clusters]
                )
                weight = np.asarray(seed_data["weight"])[sample_index]
                weight /= weight.sum()
                eligible = np.asarray(seed_data["eligible"])[sample_index].astype(bool)
                observed_pretrained = np.asarray(seed_data["observed_pretrained"])[sample_index]
                observed_random = np.asarray(seed_data["observed_random"])[sample_index]
                predicted_pretrained = np.asarray(seed_data["predicted_pretrained"])[sample_index]
                predicted_random = np.asarray(seed_data["predicted_random"])[sample_index]
                mean_error_pretrained = np.asarray(seed_data["mean_error_pretrained"])[sample_index]
                mean_error_random = np.asarray(seed_data["mean_error_random"])[sample_index]
                centered_mse_pretrained = np.asarray(seed_data["centered_mse_pretrained"])[sample_index]
                centered_mse_random = np.asarray(seed_data["centered_mse_random"])[sample_index]
                gene_pcc_pretrained = np.asarray(seed_data["gene_pcc_pretrained"])[sample_index]
                gene_pcc_random = np.asarray(seed_data["gene_pcc_random"])[sample_index]
                seed_effects.append(
                    {
                        "abundance_pcc_gain": safe_corr(
                            predicted_pretrained,
                            observed_pretrained,
                        )
                        - safe_corr(
                            predicted_random,
                            observed_random,
                        ),
                        "mean_gene_pcc_gain": float(
                            np.mean(
                                gene_pcc_pretrained[eligible]
                                - gene_pcc_random[eligible]
                            )
                        ),
                        "abundance_mse_reduction": float(
                            np.sum(
                                weight
                                * (
                                    np.square(mean_error_random)
                                    - np.square(mean_error_pretrained)
                                )
                            )
                        ),
                        "centered_mse_reduction": float(
                            np.sum(
                                weight
                                * (
                                    centered_mse_random
                                    - centered_mse_pretrained
                                )
                            )
                        ),
                    }
                )
            for metric in bootstrap:
                bootstrap[metric][draw] = float(
                    np.mean([effect[metric] for effect in seed_effects])
                )
        for metric, values in bootstrap.items():
            summary_rows.append(
                {
                    "cohort": cohort,
                    "display_cohort": DISPLAY[cohort],
                    "metric": metric,
                    "n_gene_clusters": len(unique_clusters),
                    "estimate": float(np.mean(values)),
                    "ci_low": float(np.quantile(values, 0.025)),
                    "ci_high": float(np.quantile(values, 0.975)),
                    "fraction_positive": float(np.mean(values > 0)),
                    "uncertainty_unit": "Decima-embedding clusters; technical runs averaged within draw",
                }
            )
            run_rows.extend(
                {
                    "cohort": cohort,
                    "metric": metric,
                    "draw": draw,
                    "effect": float(value),
                }
                for draw, value in enumerate(values.tolist())
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(run_rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    individual_path = SECTION_ROOT / "per_individual.tsv"
    section_path = SECTION_ROOT / "per_section.tsv.gz"
    individuals = pd.read_csv(individual_path, sep="\t")
    sections = pd.read_csv(section_path, sep="\t")
    individual_effect_runs = paired_effects(individuals, ["individual"])
    individual_effect = collapse_runs(individual_effect_runs, ["individual"])
    section_effect_runs = paired_effects(sections, ["sample", "individual"])
    block_summary, block_draws = block_bootstrap(individual_effect)
    hierarchical_summary = hierarchical_section_bootstrap(section_effect_runs)
    loo = leave_one_individual_out(individual_effect)
    gene_summary, gene_draws = cluster_bootstrap()

    individual_effect_runs.to_csv(
        OUT / "per_individual_effect_per_run.tsv", sep="\t", index=False
    )
    individual_effect.to_csv(
        OUT / "per_individual_effect_technical_mean.tsv", sep="\t", index=False
    )
    block_summary.to_csv(
        OUT / "individual_block_bootstrap_summary.tsv", sep="\t", index=False
    )
    block_draws.to_csv(
        OUT / "individual_block_bootstrap_draws.tsv.gz",
        sep="\t",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    hierarchical_summary.to_csv(
        OUT / "hierarchical_section_bootstrap_summary.tsv", sep="\t", index=False
    )
    loo.to_csv(OUT / "leave_one_individual_out.tsv", sep="\t", index=False)
    gene_summary.to_csv(
        OUT / "gene_cluster_bootstrap_summary.tsv", sep="\t", index=False
    )
    gene_draws.to_csv(
        OUT / "gene_cluster_bootstrap_draws.tsv.gz",
        sep="\t",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )

    manifest = {
        "status": "PASS",
        "n_bootstrap_draws": N_BOOTSTRAP,
        "technical_run_policy": "paired effects computed within run, then averaged within biological unit before resampling",
        "individual_counts": individual_effect.groupby("cohort")["individual"].nunique().to_dict(),
        "checks": {
            "three_runs_per_individual": bool(
                individual_effect_runs.groupby(["cohort", "individual"])["seed"]
                .nunique()
                .eq(3)
                .all()
            ),
            "all_four_cohorts": individual_effect["cohort"].nunique() == 4,
            "individual_bootstrap_complete": len(block_summary)
            == len(COHORTS) * len(EFFECT_SPECS),
            "gene_cluster_bootstrap_complete": len(gene_summary) == len(COHORTS) * 4,
        },
        "interpretive_boundary": "Intervals describe sensitivity to the finite evaluated biological individuals or correlated gene clusters; cohorts with two test individuals cannot support precise population inference.",
        "inputs": [
            {"path": str(individual_path.relative_to(ROOT)), "sha256": sha256(individual_path)},
            {"path": str(section_path.relative_to(ROOT)), "sha256": sha256(section_path)},
            {"path": str(CLUSTER_ASSIGNMENT.relative_to(ROOT)), "sha256": sha256(CLUSTER_ASSIGNMENT)},
        ],
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    print("\nIndividual block bootstrap")
    print(block_summary.to_string(index=False))
    print("\nGene-cluster bootstrap")
    print(gene_summary.to_string(index=False))


if __name__ == "__main__":
    main()
