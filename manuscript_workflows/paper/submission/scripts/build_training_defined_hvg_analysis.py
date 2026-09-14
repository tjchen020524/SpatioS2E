#!/usr/bin/env python3
"""Summarize held-out-gene performance on training-defined high-variance targets.

The ranking is fixed from pooled training-spot standard deviation.  In addition
to the primary pooled-spot endpoints, this script exports a stricter diagnostic
that centres every test section before concatenating spots.  It also places the
same 50 genes in the train-time identity-shuffle and fitted-target controls and
exports gene-level effect distributions after averaging technical runs.
"""

from __future__ import annotations
from workflow_paths import DATA_ROOT

from pathlib import Path
import json

import numpy as np
import pandas as pd


ROOT = DATA_ROOT
SOURCE = ROOT / "paper/submission/source_data"
TABLES = ROOT / "paper/submission/tables"
PARTITION = (
    ROOT
    / "experiments/heldout_gene_second_stage/partition_balance/primary_partition_per_gene.tsv.gz"
)
PER_GENE = SOURCE / "heldout_gene_benchmark_per_gene_per_run.tsv.gz"
SECTION_CENTERED_ROOT = (
    ROOT
    / "experiments/multicohort_geneheldout_decima_clean_split/section_centered/runs"
)
CONTROL_ROOT = ROOT / "experiments/heldout_gene_second_stage/decoder_controls"

GENES_OUT = SOURCE / "supp_training_defined_top50_hvg_genes.tsv"
ABSOLUTE_OUT = SOURCE / "supp_training_defined_top50_hvg_absolute.tsv"
EFFECTS_OUT = SOURCE / "supp_training_defined_top50_hvg_effects.tsv"
SUMMARY_OUT = SOURCE / "supp_training_defined_top50_hvg_summary.tsv"
PER_GENE_OUT = SOURCE / "supp_training_defined_top50_hvg_per_gene.tsv.gz"
PER_GENE_TECHNICAL_MEAN_OUT = (
    SOURCE / "supp_training_defined_top50_hvg_per_gene_technical_mean.tsv.gz"
)
PER_GENE_SUMMARY_OUT = SOURCE / "supp_training_defined_top50_hvg_per_gene_summary.tsv"
MATCHED_ABSOLUTE_OUT = SOURCE / "supp_training_defined_top50_hvg_matched_absolute.tsv"
MATCHED_EFFECTS_OUT = SOURCE / "supp_training_defined_top50_hvg_matched_effects.tsv"
MATCHED_SUMMARY_OUT = SOURCE / "supp_training_defined_top50_hvg_matched_summary.tsv"
SECTION_PER_GENE_OUT = SOURCE / "supp_training_defined_top50_hvg_section_centered_per_gene.tsv.gz"
SECTION_ABSOLUTE_OUT = SOURCE / "supp_training_defined_top50_hvg_section_centered_absolute.tsv"
SECTION_EFFECTS_OUT = SOURCE / "supp_training_defined_top50_hvg_section_centered_effects.tsv"
SECTION_SUMMARY_OUT = SOURCE / "supp_training_defined_top50_hvg_section_centered_summary.tsv"
MANIFEST_OUT = SOURCE / "supp_training_defined_top50_hvg_manifest.json"
TABLE_OUT = TABLES / "supp_table_13_training_defined_hvg.tex"
MATCHED_TABLE_OUT = TABLES / "supp_table_14_training_defined_hvg_controls.tex"

CONDITIONS = ("pretrained", "random", "constant")
CONTRASTS = {
    "pretrained_vs_random": ("pretrained", "random"),
    "pretrained_vs_constant": ("pretrained", "constant"),
}
COHORT_LABELS = {
    "hippocampus_donor_disjoint": "Hippocampus",
    "dlpfc": "DLPFC",
    "nac": "NAc",
    "her2st": "HER2ST",
}
CONDITION_LABELS = {
    "pretrained": "Pretrained gene vectors",
    "random": "Random gene vectors",
    "constant": "Constant gene vector",
}
MATCHED_CONDITION_LABELS = {
    **CONDITION_LABELS,
    "shuffled_pretrained": "Identity-shuffled gene vectors",
    "fitted_target": "Fitted-target reference",
}
MATCHED_CONTRASTS = {
    **CONTRASTS,
    "pretrained_vs_shuffled": ("pretrained", "shuffled_pretrained"),
    "fitted_target_vs_pretrained": ("fitted_target", "pretrained"),
}
VARIANT_DIR = {"pretrained": "decima", "random": "random", "constant": "constant"}


def summarize_run(group: pd.DataFrame, condition: str) -> dict[str, float]:
    pcc = group[f"{condition}_gene_pcc"].to_numpy(dtype=float)
    centred_mse = group[f"{condition}_centered_mse"].to_numpy(dtype=float)
    return {
        "mean_within_gene_pcc": float(np.mean(pcc)),
        "median_within_gene_pcc": float(np.median(pcc)),
        "centered_rmse": float(np.sqrt(np.mean(centred_mse))),
        "overall_mse": float(np.mean(group[f"{condition}_mse"].to_numpy(dtype=float))),
    }


def fmt_range(values: pd.Series) -> str:
    return f"${values.mean():.3f}$ [${values.min():.3f}$, ${values.max():.3f}$]"


def write_table(absolute: pd.DataFrame, section_absolute: pd.DataFrame) -> None:
    lines = [
        r"\begin{table}[p]",
        r"  \centering",
        r"  \caption{Held-out-gene performance for training-defined top-50 high-variance targets. Within each cohort, held-out genes were ranked by pooled training-spot expression standard deviation using training biological individuals only. The selected genes were fixed across gene-vector conditions and analysis runs. Values are mean [minimum, maximum] across three runs.}",
        r"  \label{tab:supp-training-hvg}",
        r"  \small",
        r"  \setlength{\tabcolsep}{6pt}",
        r"  \begin{tabular}{llcccc}",
        r"    \toprule",
        r"    Cohort & Gene input & Genes & Mean within-gene PCC & Within-section PCC & Centred RMSE \\",
        r"    \midrule",
    ]
    for cohort_index, (cohort, display) in enumerate(COHORT_LABELS.items()):
        subset = absolute.loc[absolute["cohort"].eq(cohort)]
        for condition in CONDITIONS:
            rows = subset.loc[subset["condition"].eq(condition)]
            section_rows = section_absolute.loc[
                section_absolute["cohort"].eq(cohort)
                & section_absolute["condition"].eq(condition)
            ]
            lines.append(
                "    "
                + " & ".join(
                    [
                        display,
                        CONDITION_LABELS[condition],
                        "50",
                        fmt_range(rows["mean_within_gene_pcc"]),
                        fmt_range(section_rows["section_centered_mean_within_gene_pcc"]),
                        fmt_range(rows["centered_rmse"]),
                    ]
                )
                + r" \\"
            )
        if cohort_index < len(COHORT_LABELS) - 1:
            lines.append(r"    \addlinespace[0.35em]")
    lines.extend(
        [
            r"    \bottomrule",
            r"  \end{tabular}",
            r"  \vspace{0.45em}",
            r"  \begin{minipage}{0.96\linewidth}\footnotesize",
            r"  Ranking and target selection used no validation- or test-individual expression. The pooled training-spot ranking can reflect within-section variation, differences among training sections or donors and measurement noise; these genes are therefore described as training-defined high-variance targets rather than as a formal set of spatially variable genes. Within-section PCC removes each test section's observed and predicted gene mean before concatenating its spots. All 50 genes satisfied the observed-expression eligibility rule. Centred RMSE is calculated on the $\log(\mathrm{CPM}+1)$ scale. Run ranges describe computational variation, not biological confidence intervals.",
            r"  \end{minipage}",
            r"\end{table}",
        ]
    )
    TABLE_OUT.write_text("\n".join(lines) + "\n")


def write_matched_table(matched_summary: pd.DataFrame, per_gene_summary: pd.DataFrame) -> None:
    contrast_labels = {
        "pretrained_vs_random": "Pretrained minus random",
        "pretrained_vs_constant": "Pretrained minus constant",
        "pretrained_vs_shuffled": "Pretrained minus identity-shuffled",
        "fitted_target_vs_pretrained": "Target-fitted reference minus held out",
    }
    lines = [
        r"\begin{table}[p]",
        r"  \centering",
        r"  \caption{Matched controls on the same training-defined top-50 high-variance targets. Mean within-gene PCC effects are candidate minus control. Values are mean [minimum, maximum] across three paired analysis runs. The final two columns summarize gene-level effects after averaging the three analysis runs within each gene.}",
        r"  \label{tab:supp-training-hvg-controls}",
        r"  \small",
        r"  \setlength{\tabcolsep}{5pt}",
        r"  \begin{tabular}{llccc}",
        r"    \toprule",
        r"    Cohort & Comparison & Mean PCC effect & Median gene effect & Genes improved \\",
        r"    \midrule",
    ]
    for cohort_index, (cohort, display) in enumerate(COHORT_LABELS.items()):
        for contrast in MATCHED_CONTRASTS:
            row = matched_summary.loc[
                matched_summary["cohort"].eq(cohort)
                & matched_summary["contrast"].eq(contrast)
            ].iloc[0]
            gene_row = per_gene_summary.loc[
                per_gene_summary["cohort"].eq(cohort)
                & per_gene_summary["contrast"].eq(contrast)
            ].iloc[0]
            lines.append(
                "    "
                + " & ".join(
                    [
                        display,
                        contrast_labels[contrast],
                        f"${row['mean_within_gene_pcc_gain_mean']:.3f}$ "
                        f"[${row['mean_within_gene_pcc_gain_min']:.3f}$, "
                        f"${row['mean_within_gene_pcc_gain_max']:.3f}$]",
                        f"${gene_row['median_effect']:.3f}$",
                        f"{int(gene_row['n_positive'])}/{int(gene_row['n_genes'])}",
                    ]
                )
                + r" \\"
            )
        if cohort_index < len(COHORT_LABELS) - 1:
            lines.append(r"    \addlinespace[0.35em]")
    lines.extend(
        [
            r"    \bottomrule",
            r"  \end{tabular}",
            r"  \vspace{0.45em}",
            r"  \begin{minipage}{0.96\linewidth}\footnotesize",
            r"  Identity-shuffled vectors preserve the standardized pretrained-vector multiset but break gene--vector correspondence during downstream fitting. The target-fitted reference uses the same frozen spot representation and decoder family as the held-out-gene decoder but allows the evaluation targets to participate in fitting and checkpoint selection. Gene-level summaries are descriptive and do not treat the 50 correlated genes as independent biological replicates.",
            r"  \end{minipage}",
            r"\end{table}",
        ]
    )
    MATCHED_TABLE_OUT.write_text("\n".join(lines) + "\n")


def load_control_components(control: str, cohort: str, seed: int) -> pd.DataFrame:
    path = (
        CONTROL_ROOT
        / control
        / "components/runs"
        / cohort
        / f"seed_{seed}"
        / "decima/gene_components.tsv.gz"
    )
    frame = pd.read_csv(path, sep="\t")
    if frame["gene_id"].duplicated().any():
        raise ValueError(f"Duplicate genes in {path}")
    return frame


def build_section_centered(selected: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[pd.DataFrame] = []
    for cohort in COHORT_LABELS:
        genes = selected.loc[selected["cohort"].eq(cohort), ["gene_id", "training_hvg_rank"]]
        for seed in (42, 123, 456):
            for condition, variant in VARIANT_DIR.items():
                path = (
                    SECTION_CENTERED_ROOT
                    / cohort
                    / f"seed_{seed}"
                    / variant
                    / "concat_gene_pcc.tsv.gz"
                )
                frame = pd.read_csv(path, sep="\t").merge(
                    genes, on="gene_id", how="inner", validate="one_to_one"
                )
                if len(frame) != 50 or not frame["gene_pcc_eligible"].astype(bool).all():
                    raise ValueError(f"Unexpected section-centred top-50 grid in {path}")
                frame = frame.assign(condition=condition)
                rows.append(
                    frame[
                        [
                            "cohort",
                            "seed",
                            "condition",
                            "gene_id",
                            "training_hvg_rank",
                            "section_centered_concat_pcc",
                            "gene_pcc_eligible",
                            "n_spots",
                        ]
                    ]
                )
    per_gene = pd.concat(rows, ignore_index=True)
    absolute = (
        per_gene.groupby(["cohort", "seed", "condition"], as_index=False)
        .agg(
            n_genes=("gene_id", "size"),
            section_centered_mean_within_gene_pcc=("section_centered_concat_pcc", "mean"),
            section_centered_median_within_gene_pcc=("section_centered_concat_pcc", "median"),
        )
        .assign(subset="training_defined_top50_hvg")
    )
    effect_rows: list[dict[str, object]] = []
    for (cohort, seed), group in absolute.groupby(["cohort", "seed"], sort=False):
        indexed = group.set_index("condition")
        for contrast, (candidate, control) in CONTRASTS.items():
            effect_rows.append(
                {
                    "cohort": cohort,
                    "seed": int(seed),
                    "subset": "training_defined_top50_hvg",
                    "contrast": contrast,
                    "n_genes": 50,
                    "section_centered_mean_within_gene_pcc_gain": float(
                        indexed.loc[candidate, "section_centered_mean_within_gene_pcc"]
                        - indexed.loc[control, "section_centered_mean_within_gene_pcc"]
                    ),
                }
            )
    effects = pd.DataFrame(effect_rows)
    summary = (
        effects.groupby(["cohort", "subset", "contrast"], as_index=False)
        .agg(
            n_genes=("n_genes", "first"),
            section_centered_mean_within_gene_pcc_gain_mean=(
                "section_centered_mean_within_gene_pcc_gain",
                "mean",
            ),
            section_centered_mean_within_gene_pcc_gain_min=(
                "section_centered_mean_within_gene_pcc_gain",
                "min",
            ),
            section_centered_mean_within_gene_pcc_gain_max=(
                "section_centered_mean_within_gene_pcc_gain",
                "max",
            ),
            n_runs=("seed", "size"),
        )
    )
    return per_gene, absolute, effects, summary


def main() -> None:
    SOURCE.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    partition = pd.read_csv(PARTITION, sep="\t")
    per_gene = pd.read_csv(PER_GENE, sep="\t")

    selected_tables: list[pd.DataFrame] = []
    for cohort in COHORT_LABELS:
        cohort_partition = partition.loc[
            partition["cohort"].eq(cohort) & partition["assignment"].eq("heldout")
        ].copy()
        cohort_partition = cohort_partition.sort_values(
            ["training_spot_std", "gene_id"], ascending=[False, True]
        ).head(50)
        if len(cohort_partition) != 50:
            raise ValueError(f"Expected 50 held-out genes for {cohort}")
        cohort_partition.insert(
            cohort_partition.columns.get_loc("training_spot_std") + 1,
            "training_hvg_rank",
            np.arange(1, 51),
        )
        selected_tables.append(
            cohort_partition[
                [
                    "cohort",
                    "gene_id",
                    "training_hvg_rank",
                    "training_spot_std",
                    "training_mean_log_expression",
                ]
            ]
        )
    selected = pd.concat(selected_tables, ignore_index=True)
    selected.to_csv(GENES_OUT, sep="\t", index=False)

    selected_metrics = per_gene.merge(
        selected[["cohort", "gene_id", "training_hvg_rank", "training_spot_std"]],
        on=["cohort", "gene_id"],
        how="inner",
        validate="many_to_one",
    )
    if len(selected_metrics) != 4 * 3 * 50:
        raise ValueError(f"Unexpected selected metric rows: {len(selected_metrics)}")
    if not selected_metrics["gene_pcc_eligible"].astype(bool).all():
        raise ValueError("A training-defined top-50 gene is ineligible for within-gene PCC")

    absolute_rows: list[dict[str, object]] = []
    for (cohort, seed), group in selected_metrics.groupby(["cohort", "seed"], sort=False):
        if len(group) != 50:
            raise ValueError(f"Expected 50 genes for {cohort}/{seed}")
        for condition in CONDITIONS:
            absolute_rows.append(
                {
                    "cohort": cohort,
                    "seed": int(seed),
                    "subset": "training_defined_top50_hvg",
                    "condition": condition,
                    "n_genes": len(group),
                    **summarize_run(group, condition),
                }
            )
    absolute = pd.DataFrame(absolute_rows)
    absolute.to_csv(ABSOLUTE_OUT, sep="\t", index=False)

    effect_rows: list[dict[str, object]] = []
    for (cohort, seed), rows in absolute.groupby(["cohort", "seed"], sort=False):
        indexed = rows.set_index("condition")
        for contrast, (candidate, control) in CONTRASTS.items():
            effect_rows.append(
                {
                    "cohort": cohort,
                    "seed": int(seed),
                    "subset": "training_defined_top50_hvg",
                    "contrast": contrast,
                    "n_genes": 50,
                    "mean_within_gene_pcc_gain": float(
                        indexed.loc[candidate, "mean_within_gene_pcc"]
                        - indexed.loc[control, "mean_within_gene_pcc"]
                    ),
                    "centered_rmse_reduction": float(
                        indexed.loc[control, "centered_rmse"]
                        - indexed.loc[candidate, "centered_rmse"]
                    ),
                }
            )
    effects = pd.DataFrame(effect_rows)
    effects.to_csv(EFFECTS_OUT, sep="\t", index=False)

    summary = (
        effects.groupby(["cohort", "subset", "contrast"], as_index=False)
        .agg(
            n_genes=("n_genes", "first"),
            mean_within_gene_pcc_gain_mean=("mean_within_gene_pcc_gain", "mean"),
            mean_within_gene_pcc_gain_min=("mean_within_gene_pcc_gain", "min"),
            mean_within_gene_pcc_gain_max=("mean_within_gene_pcc_gain", "max"),
            centered_rmse_reduction_mean=("centered_rmse_reduction", "mean"),
            centered_rmse_reduction_min=("centered_rmse_reduction", "min"),
            centered_rmse_reduction_max=("centered_rmse_reduction", "max"),
            n_runs=("seed", "size"),
        )
    )
    summary.to_csv(SUMMARY_OUT, sep="\t", index=False)

    # Place the exact same genes in the identity-shuffled and fitted-target
    # controls.  These component files use the same observed-defined PCC
    # denominator as the primary release.
    enriched_groups: list[pd.DataFrame] = []
    for (cohort, seed), group in selected_metrics.groupby(["cohort", "seed"], sort=False):
        enriched = group.copy()
        for output_name, control_dir in (
            ("shuffled_pretrained", "shuffled_pretrained"),
            ("fitted_target", "fitted_target_oracle"),
        ):
            control = load_control_components(control_dir, str(cohort), int(seed))
            control = control.loc[
                control["gene_id"].isin(enriched["gene_id"]),
                ["gene_id", "gene_pcc", "centered_mse", "mse"],
            ].rename(
                columns={
                    "gene_pcc": f"{output_name}_gene_pcc",
                    "centered_mse": f"{output_name}_centered_mse",
                    "mse": f"{output_name}_mse",
                }
            )
            enriched = enriched.merge(control, on="gene_id", how="left", validate="one_to_one")
        if len(enriched) != 50 or enriched.filter(regex="^(shuffled|fitted)").isna().any().any():
            raise ValueError(f"Incomplete matched top-50 controls for {cohort}/{seed}")
        enriched_groups.append(enriched)
    enriched_metrics = pd.concat(enriched_groups, ignore_index=True)

    per_gene_columns = [
        "cohort",
        "seed",
        "gene_id",
        "training_hvg_rank",
        "training_spot_std",
        "observed_std",
        "pretrained_gene_pcc",
        "random_gene_pcc",
        "constant_gene_pcc",
        "shuffled_pretrained_gene_pcc",
        "fitted_target_gene_pcc",
    ]
    per_gene_export = enriched_metrics[per_gene_columns].copy()
    for contrast, (candidate, control) in MATCHED_CONTRASTS.items():
        per_gene_export[contrast] = (
            per_gene_export[f"{candidate}_gene_pcc"]
            - per_gene_export[f"{control}_gene_pcc"]
        )
    per_gene_export.to_csv(PER_GENE_OUT, sep="\t", index=False, compression="gzip")

    technical_mean = (
        per_gene_export.groupby(
            ["cohort", "gene_id", "training_hvg_rank", "training_spot_std", "observed_std"],
            as_index=False,
        )[list(MATCHED_CONTRASTS)]
        .mean()
    )
    technical_mean.to_csv(
        PER_GENE_TECHNICAL_MEAN_OUT, sep="\t", index=False, compression="gzip"
    )
    per_gene_summary_rows: list[dict[str, object]] = []
    for cohort, group in technical_mean.groupby("cohort", sort=False):
        for contrast in MATCHED_CONTRASTS:
            values = group[contrast].to_numpy(float)
            per_gene_summary_rows.append(
                {
                    "cohort": cohort,
                    "contrast": contrast,
                    "n_genes": len(values),
                    "mean_effect": float(values.mean()),
                    "median_effect": float(np.median(values)),
                    "q25_effect": float(np.quantile(values, 0.25)),
                    "q75_effect": float(np.quantile(values, 0.75)),
                    "n_positive": int((values > 0).sum()),
                    "fraction_positive": float((values > 0).mean()),
                }
            )
    per_gene_summary = pd.DataFrame(per_gene_summary_rows)
    per_gene_summary.to_csv(PER_GENE_SUMMARY_OUT, sep="\t", index=False)

    matched_absolute_rows: list[dict[str, object]] = []
    for (cohort, seed), group in enriched_metrics.groupby(["cohort", "seed"], sort=False):
        for condition in MATCHED_CONDITION_LABELS:
            matched_absolute_rows.append(
                {
                    "cohort": cohort,
                    "seed": int(seed),
                    "subset": "training_defined_top50_hvg",
                    "condition": condition,
                    "n_genes": len(group),
                    **summarize_run(group, condition),
                }
            )
    matched_absolute = pd.DataFrame(matched_absolute_rows)
    matched_absolute.to_csv(MATCHED_ABSOLUTE_OUT, sep="\t", index=False)

    matched_effect_rows: list[dict[str, object]] = []
    for (cohort, seed), group in matched_absolute.groupby(["cohort", "seed"], sort=False):
        indexed = group.set_index("condition")
        for contrast, (candidate, control) in MATCHED_CONTRASTS.items():
            matched_effect_rows.append(
                {
                    "cohort": cohort,
                    "seed": int(seed),
                    "subset": "training_defined_top50_hvg",
                    "contrast": contrast,
                    "n_genes": 50,
                    "mean_within_gene_pcc_gain": float(
                        indexed.loc[candidate, "mean_within_gene_pcc"]
                        - indexed.loc[control, "mean_within_gene_pcc"]
                    ),
                    "centered_rmse_reduction": float(
                        indexed.loc[control, "centered_rmse"]
                        - indexed.loc[candidate, "centered_rmse"]
                    ),
                }
            )
    matched_effects = pd.DataFrame(matched_effect_rows)
    matched_effects.to_csv(MATCHED_EFFECTS_OUT, sep="\t", index=False)
    matched_summary = (
        matched_effects.groupby(["cohort", "subset", "contrast"], as_index=False)
        .agg(
            n_genes=("n_genes", "first"),
            mean_within_gene_pcc_gain_mean=("mean_within_gene_pcc_gain", "mean"),
            mean_within_gene_pcc_gain_min=("mean_within_gene_pcc_gain", "min"),
            mean_within_gene_pcc_gain_max=("mean_within_gene_pcc_gain", "max"),
            centered_rmse_reduction_mean=("centered_rmse_reduction", "mean"),
            centered_rmse_reduction_min=("centered_rmse_reduction", "min"),
            centered_rmse_reduction_max=("centered_rmse_reduction", "max"),
            n_runs=("seed", "size"),
        )
    )
    matched_summary.to_csv(MATCHED_SUMMARY_OUT, sep="\t", index=False)

    section_per_gene, section_absolute, section_effects, section_summary = (
        build_section_centered(selected)
    )
    section_per_gene.to_csv(
        SECTION_PER_GENE_OUT, sep="\t", index=False, compression="gzip"
    )
    section_absolute.to_csv(SECTION_ABSOLUTE_OUT, sep="\t", index=False)
    section_effects.to_csv(SECTION_EFFECTS_OUT, sep="\t", index=False)
    section_summary.to_csv(SECTION_SUMMARY_OUT, sep="\t", index=False)

    write_table(absolute, section_absolute)
    write_matched_table(matched_summary, per_gene_summary)
    manifest = {
        "selection": (
            "top 50 downstream-held-out genes per cohort ranked by pooled "
            "training-spot log-expression standard deviation"
        ),
        "selection_boundary": (
            "the ranking can include within-section variation, between-section or "
            "donor shifts and measurement noise; it is not a formal spatially "
            "variable-gene call"
        ),
        "runs": [42, 123, 456],
        "counts": {
            "selected_genes": len(selected),
            "primary_absolute_rows": len(absolute),
            "primary_effect_rows": len(effects),
            "matched_absolute_rows": len(matched_absolute),
            "matched_effect_rows": len(matched_effects),
            "per_gene_rows": len(per_gene_export),
            "section_centered_per_gene_rows": len(section_per_gene),
        },
        "section_centered_definition": (
            "observed and predicted means removed separately within each test section; "
            "centred spots concatenated before per-gene PCC"
        ),
        "interpretive_boundary": (
            "constant gene vectors yield one shared spot-varying prediction map for all "
            "targets in the tested decoder; gains over constant isolate gene-dependent "
            "information more directly than absolute PCC"
        ),
        "outputs": [
            str(path.relative_to(ROOT))
            for path in (
                GENES_OUT,
                ABSOLUTE_OUT,
                EFFECTS_OUT,
                SUMMARY_OUT,
                PER_GENE_OUT,
                PER_GENE_TECHNICAL_MEAN_OUT,
                PER_GENE_SUMMARY_OUT,
                MATCHED_ABSOLUTE_OUT,
                MATCHED_EFFECTS_OUT,
                MATCHED_SUMMARY_OUT,
                SECTION_PER_GENE_OUT,
                SECTION_ABSOLUTE_OUT,
                SECTION_EFFECTS_OUT,
                SECTION_SUMMARY_OUT,
                TABLE_OUT,
                MATCHED_TABLE_OUT,
            )
        ],
    }
    MANIFEST_OUT.write_text(json.dumps(manifest, indent=2) + "\n")
    print(matched_summary.to_string(index=False))
    print("\nSection-centred effects\n", section_summary.to_string(index=False))


if __name__ == "__main__":
    main()
