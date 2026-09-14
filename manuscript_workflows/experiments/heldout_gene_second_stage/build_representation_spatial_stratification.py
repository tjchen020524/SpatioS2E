#!/usr/bin/env python3
"""Stratify held-out spatial transfer by training variance and individual.

All cross-representation summaries use the exact scGPT-covered held-out target
set.  Contrasts remain within representation.  Technical runs are averaged
before gene- or individual-level distributions are described.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.append(str(REPOSITORY_ROOT))

from experiments.multicohort_geneheldout_decima_clean_split.clean_split_config import (
    CLEAN_ROOT,
    COHORTS,
    ROOT,
    SEEDS,
)


OUT = ROOT / "experiments/heldout_gene_second_stage/representation_spatial_stratification"
SCGPT_ROOT = ROOT / "experiments/heldout_gene_second_stage/scgpt_decoder"
COVERAGE = (
    ROOT
    / "experiments/heldout_gene_second_stage/scgpt_static_gene_vectors/gene_coverage.tsv.gz"
)
PARTITION = (
    ROOT
    / "experiments/heldout_gene_second_stage/partition_balance/primary_partition_per_gene.tsv.gz"
)
TOP50 = ROOT / "paper/submission/source_data/supp_training_defined_top50_hvg_genes.tsv"
REPRESENTATIONS = {
    "Decima sequence-derived": {
        "root": CLEAN_ROOT,
        "conditions": {
            "pretrained": "decima",
            "random": "random",
            "constant": "constant",
        },
    },
    "scGPT static gene-token": {
        "root": SCGPT_ROOT,
        "conditions": {
            "pretrained": "pretrained_gene_tokens",
            "random": "random_gene_vectors",
            "constant": "constant_gene_vector",
            "identity_shuffled": "identity_shuffled_gene_tokens",
        },
    },
}
SCGPT_CHECKPOINTS = {
    "brain": {
        "root": ROOT / "experiments/heldout_gene_second_stage/scgpt_decoder",
        "coverage": ROOT
        / "experiments/heldout_gene_second_stage/scgpt_static_gene_vectors/gene_coverage.tsv.gz",
        "out": ROOT
        / "experiments/heldout_gene_second_stage/representation_spatial_stratification",
        "label": "scGPT brain gene-token",
    },
    "whole_human": {
        "root": ROOT
        / "experiments/heldout_gene_second_stage/scgpt_whole_human_decoder",
        "coverage": ROOT
        / "experiments/heldout_gene_second_stage/scgpt_whole_human_static_gene_vectors/gene_coverage.tsv.gz",
        "out": ROOT
        / "experiments/heldout_gene_second_stage/representation_spatial_stratification_whole_human",
        "label": "scGPT whole-human gene-token",
    },
}


def paths(
    representation: str, root: Path, directory: str, cohort: str, seed: int
) -> tuple[Path, Path, Path]:
    if representation.startswith("Decima"):
        component = (
            root
            / "components/runs"
            / cohort
            / f"seed_{seed}/{directory}/gene_components.tsv.gz"
        )
        section = (
            root
            / "section_centered/runs"
            / cohort
            / f"seed_{seed}/{directory}/concat_gene_pcc.tsv.gz"
        )
        individual = (
            root
            / "section_centered/runs"
            / cohort
            / f"seed_{seed}/{directory}/per_individual.tsv"
        )
    else:
        component = (
            root
            / directory
            / "components/runs"
            / cohort
            / f"seed_{seed}/decima/gene_components.tsv.gz"
        )
        section = (
            root
            / "section_centered"
            / directory
            / "runs"
            / cohort
            / f"seed_{seed}/decima/concat_gene_pcc.tsv.gz"
        )
        individual = section.with_name("per_individual.tsv")
    return component, section, individual


def describe_effects(table: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in table.groupby(groups, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        values = group["technical_mean_effect"].dropna().to_numpy(dtype=float)
        row = dict(zip(groups, keys))
        row.update(
            {
                "n_units": int(len(values)),
                "mean_effect": float(values.mean()),
                "median_effect": float(np.median(values)),
                "q25_effect": float(np.quantile(values, 0.25)),
                "q75_effect": float(np.quantile(values, 0.75)),
                "n_positive": int((values > 0).sum()),
                "fraction_positive": float((values > 0).mean()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scgpt-checkpoint", choices=SCGPT_CHECKPOINTS, default="brain")
    args = parser.parse_args()
    checkpoint = SCGPT_CHECKPOINTS[args.scgpt_checkpoint]
    output = checkpoint["out"]
    scgpt_label = checkpoint["label"]
    representations = {
        "Decima sequence-derived": REPRESENTATIONS["Decima sequence-derived"],
        scgpt_label: {
            "root": checkpoint["root"],
            "conditions": REPRESENTATIONS["scGPT static gene-token"]["conditions"],
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    coverage = pd.read_csv(checkpoint["coverage"], sep="\t")
    common = coverage.loc[
        coverage["in_scgpt_vocabulary"]
        & coverage["downstream_partition"].eq("held_out"),
        ["cohort", "gene_id"],
    ]
    annotations = pd.read_csv(PARTITION, sep="\t")
    annotations = annotations.loc[
        annotations["assignment"].eq("heldout"),
        ["cohort", "gene_id", "training_mean_log_expression", "training_spot_std"],
    ].merge(common, on=["cohort", "gene_id"], how="inner", validate="one_to_one")
    annotations["training_variance_decile"] = (
        annotations.groupby("cohort")["training_spot_std"]
        .transform(
            lambda values: pd.qcut(
                values.rank(method="first"), 10, labels=False
            )
            + 1
        )
        .astype(int)
    )
    top50 = pd.read_csv(TOP50, sep="\t")[["cohort", "gene_id"]].assign(
        training_defined_top50=True
    )
    annotations = annotations.merge(
        top50, on=["cohort", "gene_id"], how="left", validate="one_to_one"
    )
    annotations["training_defined_top50"] = annotations[
        "training_defined_top50"
    ].eq(True)
    annotations.to_csv(output / "common_target_annotations.tsv.gz", sep="\t", index=False)

    gene_rows: list[pd.DataFrame] = []
    individual_rows: list[pd.DataFrame] = []
    included_conditions: dict[str, list[str]] = {}
    for representation, spec in representations.items():
        root = Path(spec["root"])
        conditions: dict[str, str] = spec["conditions"]
        available: list[str] = []
        for condition, directory in conditions.items():
            complete = all(
                all(path.exists() for path in paths(representation, root, directory, cohort, seed)[:2])
                for cohort in COHORTS
                for seed in SEEDS
            )
            if not complete:
                continue
            available.append(condition)
            for cohort in COHORTS:
                gene_annotation = annotations.loc[annotations["cohort"].eq(cohort)]
                target_set = set(gene_annotation["gene_id"])
                for seed in SEEDS:
                    component_path, section_path, individual_path = paths(
                        representation, root, directory, cohort, seed
                    )
                    pooled = pd.read_csv(component_path, sep="\t")
                    pooled = pooled.loc[pooled["gene_id"].isin(target_set), [
                        "gene_id", "gene_pcc", "gene_pcc_eligible"
                    ]]
                    section = pd.read_csv(section_path, sep="\t")
                    section = section.loc[section["gene_id"].isin(target_set), [
                        "gene_id", "section_centered_concat_pcc", "gene_pcc_eligible"
                    ]]
                    if len(pooled) != len(target_set) or len(section) != len(target_set):
                        raise ValueError(f"Common-target mismatch for {component_path}")
                    merged = pooled.merge(
                        section,
                        on="gene_id",
                        suffixes=("_pooled", "_section"),
                        validate="one_to_one",
                    ).merge(
                        gene_annotation, on="gene_id", how="left", validate="one_to_one"
                    )
                    merged["representation"] = representation
                    merged["condition"] = condition
                    merged["seed"] = seed
                    gene_rows.append(merged)
                    if representation.startswith("scGPT") and individual_path.exists():
                        individual = pd.read_csv(individual_path, sep="\t")
                        individual["representation"] = representation
                        individual["condition"] = condition
                        # The constant vector has no across-gene mean ordering.
                        if condition == "constant":
                            individual["abundance_pcc"] = np.nan
                        individual_rows.append(individual)
        included_conditions[representation] = available

    per_gene = pd.concat(gene_rows, ignore_index=True)
    gene_effect_rows: list[pd.DataFrame] = []
    value_columns = {
        "pooled_within_gene_pcc": "gene_pcc",
        "section_centered_within_gene_pcc": "section_centered_concat_pcc",
    }
    for (representation, cohort, seed), group in per_gene.groupby(
        ["representation", "cohort", "seed"], sort=False
    ):
        for endpoint, value_column in value_columns.items():
            wide = group.pivot(index="gene_id", columns="condition", values=value_column)
            annotation = annotations.loc[annotations["cohort"].eq(cohort)].set_index(
                "gene_id"
            )
            for control in ("random", "constant", "identity_shuffled"):
                if control not in wide.columns:
                    continue
                current = pd.DataFrame(
                    {
                        "gene_id": wide.index,
                        "effect": wide["pretrained"] - wide[control],
                    }
                ).set_index("gene_id").join(annotation).reset_index()
                current["representation"] = representation
                current["cohort"] = cohort
                current["seed"] = seed
                current["contrast"] = f"pretrained_vs_{control}"
                current["endpoint"] = endpoint
                gene_effect_rows.append(current)
    gene_effect_per_run = pd.concat(gene_effect_rows, ignore_index=True)
    gene_effect_technical_mean = (
        gene_effect_per_run.groupby(
            [
                "representation",
                "cohort",
                "gene_id",
                "contrast",
                "endpoint",
                "training_variance_decile",
                "training_defined_top50",
            ],
            as_index=False,
        )["effect"]
        .mean()
        .rename(columns={"effect": "technical_mean_effect"})
    )
    decile_summary = describe_effects(
        gene_effect_technical_mean,
        ["representation", "cohort", "contrast", "endpoint", "training_variance_decile"],
    )
    subset_frames: list[pd.DataFrame] = []
    for subset, mask in (
        ("all_common_targets", np.ones(len(gene_effect_technical_mean), dtype=bool)),
        (
            "training_defined_top50",
            gene_effect_technical_mean["training_defined_top50"].astype(bool).to_numpy(),
        ),
    ):
        described = describe_effects(
            gene_effect_technical_mean.loc[mask],
            ["representation", "cohort", "contrast", "endpoint"],
        )
        described.insert(0, "target_subset", subset)
        subset_frames.append(described)
    subset_summary = pd.concat(subset_frames, ignore_index=True)

    concordance_rows: list[dict[str, object]] = []
    candidate_effects = gene_effect_technical_mean.loc[
        gene_effect_technical_mean["contrast"].eq("pretrained_vs_random")
    ]
    for cohort in COHORTS:
        for endpoint in value_columns:
            group = candidate_effects.loc[
                candidate_effects["cohort"].eq(cohort)
                & candidate_effects["endpoint"].eq(endpoint)
            ]
            wide = group.pivot(
                index="gene_id", columns="representation", values="technical_mean_effect"
            ).join(
                annotations.loc[annotations["cohort"].eq(cohort)].set_index("gene_id")[
                    ["training_defined_top50"]
                ]
            )
            for subset, subset_frame in (
                ("all_common_targets", wide),
                ("training_defined_top50", wide.loc[wide["training_defined_top50"]]),
            ):
                x = subset_frame["Decima sequence-derived"]
                y = subset_frame[scgpt_label]
                concordance_rows.append(
                    {
                        "cohort": cohort,
                        "endpoint": endpoint,
                        "target_subset": subset,
                        "n_genes": len(subset_frame),
                        "pearson_r": x.corr(y, method="pearson"),
                        "spearman_r": x.corr(y, method="spearman"),
                        "both_positive": int(((x > 0) & (y > 0)).sum()),
                        "decima_only_positive": int(((x > 0) & (y <= 0)).sum()),
                        "scgpt_only_positive": int(((x <= 0) & (y > 0)).sum()),
                        "both_nonpositive": int(((x <= 0) & (y <= 0)).sum()),
                    }
                )
    concordance = pd.DataFrame(concordance_rows)

    individual_absolute = pd.concat(individual_rows, ignore_index=True)
    individual_endpoints = [
        "abundance_pcc",
        "mean_within_section_gene_pcc",
        "centered_full_matrix_pcc",
        "section_centered_rmse",
        "total_mse",
        "abundance_mse",
        "centered_mse",
    ]
    individual_effect_rows: list[dict[str, object]] = []
    for (cohort, seed, individual), group in individual_absolute.groupby(
        ["cohort", "seed", "individual"], sort=False
    ):
        indexed = group.set_index("condition")
        for control in ("random", "constant", "identity_shuffled"):
            if control not in indexed.index:
                continue
            for endpoint in individual_endpoints:
                if endpoint == "abundance_pcc" and control == "constant":
                    effect = np.nan
                elif endpoint.endswith("mse") or endpoint.endswith("rmse"):
                    effect = indexed.loc[control, endpoint] - indexed.loc["pretrained", endpoint]
                else:
                    effect = indexed.loc["pretrained", endpoint] - indexed.loc[control, endpoint]
                individual_effect_rows.append(
                    {
                        "cohort": cohort,
                        "seed": seed,
                        "individual": individual,
                        "contrast": f"pretrained_vs_{control}",
                        "endpoint": endpoint,
                        "effect": effect,
                    }
                )
    individual_effects = pd.DataFrame(individual_effect_rows)
    individual_technical_mean = (
        individual_effects.groupby(
            ["cohort", "individual", "contrast", "endpoint"], as_index=False
        )["effect"]
        .mean()
        .rename(columns={"effect": "technical_mean_effect"})
    )

    per_gene.to_csv(output / "common_target_absolute_per_gene_per_run.tsv.gz", sep="\t", index=False)
    gene_effect_per_run.to_csv(output / "gene_effects_per_run.tsv.gz", sep="\t", index=False)
    gene_effect_technical_mean.to_csv(
        output / "gene_effects_technical_mean.tsv.gz", sep="\t", index=False
    )
    decile_summary.to_csv(output / "variance_decile_summary.tsv", sep="\t", index=False)
    subset_summary.to_csv(output / "target_subset_summary.tsv", sep="\t", index=False)
    concordance.to_csv(output / "decima_scgpt_gene_effect_concordance.tsv", sep="\t", index=False)
    individual_absolute.to_csv(output / "scgpt_per_individual_absolute.tsv", sep="\t", index=False)
    individual_effects.to_csv(output / "scgpt_per_individual_effects_per_run.tsv", sep="\t", index=False)
    individual_technical_mean.to_csv(
        output / "scgpt_per_individual_effects_technical_mean.tsv", sep="\t", index=False
    )

    checks = {
        "four_cohorts": bool(per_gene["cohort"].nunique() == 4),
        "ten_variance_deciles": bool(
            decile_summary["training_variance_decile"].nunique() == 10
        ),
        "exact_top50_per_cohort": bool(
            annotations.loc[annotations["training_defined_top50"]]
            .groupby("cohort")["gene_id"]
            .nunique()
            .eq(50)
            .all()
        ),
        "technical_runs_averaged_before_gene_summary": True,
        "all_biological_individuals_retained": bool(
            individual_technical_mean["individual"].nunique() > 4
        ),
    }
    manifest = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "scgpt_checkpoint_scope": args.scgpt_checkpoint,
        "scgpt_representation_label": scgpt_label,
        "analysis": "training-variance continuum, top-50 distribution and biological-individual sensitivity",
        "target_universe": "exact scGPT-covered held-out targets for both representations",
        "included_conditions": included_conditions,
        "decile_definition": "within-cohort deciles of training-only pooled spot standard deviation",
        "gene_summary_policy": "paired effect per technical run, then average runs per gene before distribution summaries",
        "individual_summary_policy": "paired effect per technical run, then average runs within biological individual",
        "checks": checks,
        "interpretive_boundary": "Gene-level distributions are descriptive target distributions; genes are not treated as independent biological replicates.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    print(
        subset_summary.loc[
            subset_summary["contrast"].eq("pretrained_vs_random")
            & subset_summary["endpoint"].eq("section_centered_within_gene_pcc")
        ].to_string(index=False)
    )
    print("\nConcordance\n" + concordance.to_string(index=False))


if __name__ == "__main__":
    main()
