#!/usr/bin/env python3
"""Define a train-only HER2ST tumor program for held-out pathology evaluation."""

from __future__ import annotations
from workflow_paths import DATA_ROOT

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = DATA_ROOT
if str(ROOT / "scripts") not in sys.path:
    sys.path.append(str(ROOT / "scripts"))

from gnn_dataloader import load_expression_split  # noqa: E402


EXP = ROOT / "experiments/her2st_current_full_ablation"
ANNOTATED_TRAIN = ("A1", "B1", "C1", "D1", "E1")
TUMOR_LABELS = {"invasive cancer", "cancer in situ"}
NONTUMOR_LABELS = {
    "adipose tissue",
    "breast glands",
    "connective tissue",
    "immune infiltrate",
}


def main() -> None:
    stats = pd.read_csv(EXP / "artifacts/train_gene_stats.tsv", sep="\t")
    stats_index = stats.set_index("gene_id")
    effects = []
    sample_rows = []
    reference_genes = None
    for sample in ANNOTATED_TRAIN:
        expression = load_expression_split(
            EXP / "data/expression" / sample / "train.npz", to_dense=True
        )
        gene_ids = list(expression["gene_ids"])
        if reference_genes is None:
            reference_genes = gene_ids
        elif gene_ids != reference_genes:
            raise ValueError(f"{sample}: gene order differs from other training sections")
        labels = pd.read_csv(EXP / "data/annotations" / f"{sample}.tsv", sep="\t")
        label_map = dict(zip(labels["spot_id"].astype(str), labels["pathology_label"].astype(str)))
        spot_labels = np.asarray([label_map.get(str(barcode), "") for barcode in expression["barcodes"]])
        tumor = np.isin(spot_labels, sorted(TUMOR_LABELS))
        nontumor = np.isin(spot_labels, sorted(NONTUMOR_LABELS))
        if int(tumor.sum()) < 10 or int(nontumor.sum()) < 10:
            raise ValueError(f"{sample}: insufficient labeled tumor or non-tumor spots")
        matrix = np.asarray(expression["matrix"], dtype=np.float32)
        difference = matrix[tumor].mean(axis=0) - matrix[nontumor].mean(axis=0)
        train_std = stats_index.loc[gene_ids, "train_std"].to_numpy(dtype=np.float32)
        effects.append(difference / np.maximum(train_std, 1.0e-3))
        sample_rows.append(
            {
                "sample": sample,
                "n_tumor": int(tumor.sum()),
                "n_nontumor": int(nontumor.sum()),
            }
        )

    assert reference_genes is not None
    mean_effect = np.mean(np.vstack(effects), axis=0)
    train_mean = stats_index.loc[reference_genes, "train_mean"].to_numpy(dtype=np.float32)
    train_std = stats_index.loc[reference_genes, "train_std"].to_numpy(dtype=np.float32)
    eligible = np.isfinite(mean_effect) & (train_std > 1.0e-3) & (train_mean > 0.05)
    eligible_indices = np.flatnonzero(eligible)
    ranked = eligible_indices[np.argsort(mean_effect[eligible_indices], kind="stable")]
    n_markers = 50
    down_indices = ranked[:n_markers]
    up_indices = ranked[-n_markers:][::-1]
    selected = np.concatenate([up_indices, down_indices])
    directions = np.concatenate(
        [np.ones(n_markers, dtype=np.float32), -np.ones(n_markers, dtype=np.float32)]
    )

    output_dir = EXP / "artifacts/pathology_program"
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_genes = [reference_genes[index] for index in selected]
    np.savez(
        output_dir / "tumor_program.npz",
        gene_ids=np.asarray(selected_genes, dtype=str),
        direction=directions,
        train_mean=train_mean[selected].astype(np.float32),
        train_std=train_std[selected].astype(np.float32),
        standardized_effect=mean_effect[selected].astype(np.float32),
    )
    pd.DataFrame(
        {
            "gene_id": selected_genes,
            "direction": ["tumor_up"] * n_markers + ["tumor_down"] * n_markers,
            "standardized_effect": mean_effect[selected],
            "train_mean": train_mean[selected],
            "train_std": train_std[selected],
        }
    ).to_csv(output_dir / "tumor_program_genes.tsv", sep="\t", index=False)
    summary = {
        "definition": "mean within-section tumor-minus-nontumor effect standardized by train-only gene SD",
        "training_sections": list(ANNOTATED_TRAIN),
        "tumor_labels": sorted(TUMOR_LABELS),
        "nontumor_labels": sorted(NONTUMOR_LABELS),
        "n_up_genes": n_markers,
        "n_down_genes": n_markers,
        "sample_counts": sample_rows,
        "test_sections_not_used": ["G2", "H1"],
    }
    (output_dir / "tumor_program.summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
