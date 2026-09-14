#!/usr/bin/env python3
"""Freeze provenance and downstream vocabulary coverage for scGPT vectors.

The audit is intentionally descriptive.  The local checkpoint metadata
identifies an organ-specific model pretrained on CELLxGENE brain cells, but it
does not enumerate every upstream dataset.  Consequently, overlap with the
downstream spatial cohorts is reported as unresolved rather than absent.
"""

from __future__ import annotations
from workflow_paths import DATA_ROOT

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.append(str(REPOSITORY_ROOT))

from experiments.geneheldout_uni2h_decima_decoder_noseq_controls import (
    run_geneheldout_noseq_controls as base,
)
from experiments.multicohort_geneheldout_decima_clean_split.clean_split_config import (
    COHORTS,
    ROOT,
    configure_clean_base,
)


OUT = ROOT / "experiments/heldout_gene_second_stage/scgpt_provenance"
MODEL_DIR = ROOT / "scGPT/brain_model"
SCGPT_REPOSITORY = ROOT / "scGPT"
COVERAGE = (
    ROOT
    / "experiments/heldout_gene_second_stage/scgpt_static_gene_vectors/gene_coverage.tsv.gz"
)
CHECKPOINTS = {
    "brain": {
        "model_dir": ROOT / "scGPT/brain_model",
        "out": ROOT / "experiments/heldout_gene_second_stage/scgpt_provenance",
        "coverage": ROOT
        / "experiments/heldout_gene_second_stage/scgpt_static_gene_vectors/gene_coverage.tsv.gz",
        "label": "static gene-token vectors extracted from the scGPT brain checkpoint",
        "scope": "organ-specific brain pretraining checkpoint; not a downstream fine-tuned checkpoint",
        "training": "Official scGPT model-zoo description: pretrained on 13.2 million brain cells",
        "overlap": (
            "Unresolved: the packaged checkpoint metadata does not enumerate every "
            "upstream dataset, so overlap or close tissue exposure with the downstream "
            "brain cohorts cannot be excluded."
        ),
    },
    "whole_human": {
        "model_dir": ROOT / "scGPT/whole_human_model",
        "out": ROOT
        / "experiments/heldout_gene_second_stage/scgpt_whole_human_provenance",
        "coverage": ROOT
        / "experiments/heldout_gene_second_stage/scgpt_whole_human_static_gene_vectors/gene_coverage.tsv.gz",
        "label": "static gene-token vectors extracted from the scGPT whole-human checkpoint",
        "scope": "whole-human pretraining checkpoint; not a downstream fine-tuned checkpoint",
        "training": "Official scGPT model-zoo description: pretrained on 33 million normal human cells",
        "overlap": (
            "Unresolved: the packaged checkpoint metadata identifies the CELLxGENE human "
            "pretraining collection but does not enumerate every source dataset, so direct "
            "or closely related exposure to downstream tissues cannot be excluded."
        ),
    },
}
PARTITION = (
    ROOT
    / "experiments/heldout_gene_second_stage/partition_balance/primary_partition_per_gene.tsv.gz"
)
TOP50 = ROOT / "paper/submission/source_data/supp_training_defined_top50_hvg_genes.tsv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_value(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(SCGPT_REPOSITORY), *arguments], text=True
    ).strip()


def training_detection_fraction(cohort: str) -> tuple[list[str], np.ndarray, int]:
    configure_clean_base(cohort, 42, "decima")
    gene_ids = base.load_gene_ids()
    manifest = pd.read_csv(base.DATA_ROOT / "manifest.tsv", sep="\t")
    train = manifest.loc[manifest["split"] == "train"]
    store = base.ExpressionStore(gene_ids)
    nonzero = np.zeros(len(gene_ids), dtype=np.int64)
    n_spots = 0
    for sample in sorted(train["sample"].astype(str).unique()):
        matrix, _ = store._load(sample, "train")
        nonzero += np.asarray(matrix.getnnz(axis=0)).ravel().astype(np.int64)
        n_spots += int(matrix.shape[0])
    return gene_ids, nonzero / float(n_spots), n_spots


def summarize_numeric(table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (cohort, partition, covered), group in table.groupby(
        ["cohort", "downstream_partition", "in_scgpt_vocabulary"], sort=False
    ):
        rows.append(
            {
                "cohort": cohort,
                "downstream_partition": partition,
                "coverage_group": "covered" if covered else "OOV",
                "n_genes": len(group),
                "training_mean_log_expression_mean": group[
                    "training_mean_log_expression"
                ].mean(),
                "training_mean_log_expression_median": group[
                    "training_mean_log_expression"
                ].median(),
                "training_spot_std_mean": group["training_spot_std"].mean(),
                "training_spot_std_median": group["training_spot_std"].median(),
                "training_detection_fraction_mean": group[
                    "training_detection_fraction"
                ].mean(),
                "training_detection_fraction_median": group[
                    "training_detection_fraction"
                ].median(),
                "protein_coding_fraction": group["gene_type"]
                .eq("protein_coding")
                .mean(),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    global OUT, MODEL_DIR, COVERAGE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", choices=CHECKPOINTS, default="brain")
    args_cli = parser.parse_args()
    checkpoint = CHECKPOINTS[args_cli.checkpoint]
    OUT = checkpoint["out"]
    MODEL_DIR = checkpoint["model_dir"]
    COVERAGE = checkpoint["coverage"]
    OUT.mkdir(parents=True, exist_ok=True)
    coverage = pd.read_csv(COVERAGE, sep="\t")
    partition = pd.read_csv(PARTITION, sep="\t")
    partition["downstream_partition"] = partition["assignment"].replace(
        {"heldout": "held_out"}
    )
    merged = coverage.merge(
        partition[
            [
                "cohort",
                "gene_id",
                "downstream_partition",
                "gene_type",
                "training_mean_log_expression",
                "training_spot_std",
            ]
        ],
        on=["cohort", "gene_id", "downstream_partition"],
        how="left",
        validate="one_to_one",
    )
    if merged["gene_type"].isna().any():
        raise ValueError("Coverage-to-partition merge left missing gene annotations")

    spot_counts: dict[str, int] = {}
    for cohort in COHORTS:
        gene_ids, detection, n_spots = training_detection_fraction(cohort)
        detection_lookup = dict(zip(gene_ids, detection.tolist()))
        mask = merged["cohort"].eq(cohort)
        merged.loc[mask, "training_detection_fraction"] = merged.loc[
            mask, "gene_id"
        ].map(detection_lookup)
        spot_counts[cohort] = n_spots
    if merged["training_detection_fraction"].isna().any():
        raise ValueError("Training detection calculation left missing values")

    coverage_summary = (
        merged.groupby(
            ["cohort", "downstream_partition", "in_scgpt_vocabulary"],
            as_index=False,
        )
        .agg(n_genes=("gene_id", "size"))
    )
    coverage_summary["coverage_group"] = np.where(
        coverage_summary["in_scgpt_vocabulary"], "covered", "OOV"
    )
    denominators = coverage_summary.groupby(
        ["cohort", "downstream_partition"]
    )["n_genes"].transform("sum")
    coverage_summary["fraction"] = coverage_summary["n_genes"] / denominators
    coverage_summary = coverage_summary[
        [
            "cohort",
            "downstream_partition",
            "coverage_group",
            "n_genes",
            "fraction",
        ]
    ]
    characteristics = summarize_numeric(merged)
    selected = pd.read_csv(TOP50, sep="\t")[["cohort", "gene_id"]]
    selected_coverage = selected.merge(
        coverage[["cohort", "gene_id", "in_scgpt_vocabulary"]],
        on=["cohort", "gene_id"],
        how="left",
        validate="one_to_one",
    )
    oov = merged.loc[
        ~merged["in_scgpt_vocabulary"],
        [
            "cohort",
            "downstream_partition",
            "gene_id",
            "gene_symbol",
            "gene_type",
            "training_mean_log_expression",
            "training_spot_std",
            "training_detection_fraction",
        ],
    ].copy()

    coverage_summary.to_csv(
        OUT / "coverage_by_cohort_partition.tsv", sep="\t", index=False
    )
    characteristics.to_csv(
        OUT / "covered_vs_oov_characteristics.tsv", sep="\t", index=False
    )
    oov.to_csv(
        OUT / "oov_gene_list.tsv.gz",
        sep="\t",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )

    args = json.loads((MODEL_DIR / "args.json").read_text())
    state = torch.load(MODEL_DIR / "best_model.pt", map_location="cpu")
    embedding_shape = list(state["encoder.embedding.weight"].shape)
    provenance = {
        "representation_label": checkpoint["label"],
        "representation_family": "single-cell-transcriptomically pretrained gene representation",
        "checkpoint_scope": checkpoint["scope"],
        "upstream_modality": "single-cell transcriptomic profiles",
        "upstream_training_description": checkpoint["training"],
        "checkpoint_training_source_from_args": args.get("data_source"),
        "checkpoint_save_label_from_args": args.get("save_dir"),
        "upstream_overlap_status": checkpoint["overlap"],
        "checkpoint": str((MODEL_DIR / "best_model.pt").relative_to(ROOT)),
        "checkpoint_sha256": sha256(MODEL_DIR / "best_model.pt"),
        "checkpoint_args": str((MODEL_DIR / "args.json").relative_to(ROOT)),
        "checkpoint_args_sha256": sha256(MODEL_DIR / "args.json"),
        "vocabulary": str((MODEL_DIR / "vocab.json").relative_to(ROOT)),
        "vocabulary_sha256": sha256(MODEL_DIR / "vocab.json"),
        "vocabulary_size": int(embedding_shape[0]),
        "embedding_dimension": int(embedding_shape[1]),
        "repository_url": git_value("remote", "get-url", "origin"),
        "extraction_code_repository_commit": git_value("rev-parse", "HEAD"),
        "extraction_code_repository_commit_date": git_value(
            "log", "-1", "--format=%cI"
        ),
        "checkpoint_revision_identifier": (
            "not supplied in the packaged model directory; the SHA-256 digest "
            "above uniquely freezes the checkpoint used"
        ),
        "extraction": (
            "encoder.embedding.weight token row followed by the checkpoint's "
            "encoder.enc_norm LayerNorm (epsilon 1e-5); no cell-specific expression "
            "values or contextual transformer outputs"
        ),
        "downstream_standardization": (
            "feature-wise mean and standard deviation estimated from vocabulary-covered "
            "downstream training genes within each cohort"
        ),
        "coverage_rule": (
            "Genes absent from the scGPT vocabulary are excluded identically from the "
            "pretrained, random, constant and identity-shuffled downstream conditions."
        ),
    }
    (OUT / "representation_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    manifest = {
        "status": "PASS",
        "analysis": "scGPT checkpoint provenance and downstream vocabulary coverage",
        "training_spots_used_for_detection_summary": spot_counts,
        "outputs": {
            "provenance": str(
                (OUT / "representation_provenance.json").relative_to(ROOT)
            ),
            "coverage": str(
                (OUT / "coverage_by_cohort_partition.tsv").relative_to(ROOT)
            ),
            "covered_vs_oov": str(
                (OUT / "covered_vs_oov_characteristics.tsv").relative_to(ROOT)
            ),
            "oov_genes": str((OUT / "oov_gene_list.tsv.gz").relative_to(ROOT)),
        },
        "checks": {
            "all_four_cohorts": coverage_summary["cohort"].nunique() == 4,
            "all_partitions": sorted(
                coverage_summary["downstream_partition"].unique().tolist()
            )
            == ["held_out", "training"],
            "coverage_counts_sum_to_gene_universe": int(
                coverage_summary["n_genes"].sum()
            )
            == len(merged),
            "all_top50_targets_covered": bool(
                len(selected_coverage) == 4 * 50
                and selected_coverage["in_scgpt_vocabulary"].fillna(False).all()
            ),
        },
        "interpretive_boundary": provenance["upstream_overlap_status"],
    }
    if not all(manifest["checks"].values()):
        manifest["status"] = "FAIL"
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    print(coverage_summary.to_string(index=False))
    print(characteristics.to_string(index=False))


if __name__ == "__main__":
    main()
