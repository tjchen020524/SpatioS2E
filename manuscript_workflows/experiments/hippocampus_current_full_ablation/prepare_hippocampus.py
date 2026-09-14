#!/usr/bin/env python3
"""Prepare a GSM-group-disjoint hippocampus cohort for current ablations."""

from __future__ import annotations
from workflow_paths import DATA_ROOT

import argparse
import csv
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np


ROOT = DATA_ROOT
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.append(str(path))

from gnn_dataloader import ModalityNormalizer, load_expression_split  # noqa: E402


EXP = ROOT / "experiments/hippocampus_current_full_ablation"
SOURCE_EXPRESSION = ROOT / "experiments/sample_split_fullgenes/expression_full"
SOURCE_MULTIMODAL = ROOT / "data/processed/multimodal_features_uni2h_zero_celltype"
SOURCE_GRAPHS = ROOT / "data/processed/spatial_graphs"
SOURCE_GENES = ROOT / "experiments/sample_split_fullgenes/gene_splits/train_genes.txt"
SOURCE_SAMPLE_LINKS = ROOT / "metadata/sample_links.csv"

GROUP_SPLIT = {
    "train": [
        "GSM8226199",
        "GSM8226201",
        "GSM8226203",
        "GSM8226204",
        "GSM8226205",
        "GSM8226206",
    ],
    "val": ["GSM8226202", "GSM8226208"],
    "test": ["GSM8226200", "GSM8226207"],
}


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def link_or_copy(source: Path, destination: Path, overwrite: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not overwrite:
            return
        destination.unlink()
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def sample_group(sample: str) -> str:
    return sample.split("_", maxsplit=1)[0]


def load_gsm_to_donor() -> dict[str, str]:
    mapping: dict[str, str] = {}
    with SOURCE_SAMPLE_LINKS.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["modality"] != "SRT":
                continue
            gsm = row["gsm_id"]
            donor = row["donor_id"]
            if gsm in mapping and mapping[gsm] != donor:
                raise ValueError(f"{gsm} maps to multiple donors")
            mapping[gsm] = donor
    expected_groups = {group for groups in GROUP_SPLIT.values() for group in groups}
    selected = {gsm: mapping[gsm] for gsm in expected_groups}
    if len(set(selected.values())) != 10:
        raise ValueError("The 10 hippocampus GSM groups must map to 10 unique donors")
    return selected


def build_split() -> dict[str, list[str]]:
    samples = sorted(path.name for path in SOURCE_EXPRESSION.iterdir() if path.is_dir())
    expected_groups = {group for groups in GROUP_SPLIT.values() for group in groups}
    observed_groups = {sample_group(sample) for sample in samples}
    if len(samples) != 34 or observed_groups != expected_groups:
        raise ValueError(
            f"Expected 34 arrays from 10 GSM groups; observed "
            f"{len(samples)} arrays and groups {sorted(observed_groups)}"
        )
    split = {
        split_name: [
            sample for sample in samples if sample_group(sample) in set(groups)
        ]
        for split_name, groups in GROUP_SPLIT.items()
    }
    flattened = [sample for split_name in ("train", "val", "test") for sample in split[split_name]]
    if len(flattened) != 34 or len(set(flattened)) != 34:
        raise ValueError("The GSM-group split must contain all 34 arrays exactly once")
    if {key: len(value) for key, value in split.items()} != {"train": 22, "val": 8, "test": 4}:
        raise ValueError("Unexpected array counts for the GSM-group split")
    return split


def prepare_sample(
    sample: str,
    split_name: str,
    gene_ids: list[str],
    gsm_to_donor: dict[str, str],
    overwrite: bool,
) -> dict:
    source_expression = SOURCE_EXPRESSION / sample / "train.npz"
    source_multimodal = SOURCE_MULTIMODAL / sample / "multimodal_features.npz"
    source_graph = SOURCE_GRAPHS / sample / "graph.npz"
    for path in (source_expression, source_multimodal, source_graph):
        if not path.exists():
            raise FileNotFoundError(path)

    expression = load_expression_split(source_expression, to_dense=False)
    expression_genes = list(expression["gene_ids"])
    if expression_genes != gene_ids:
        raise ValueError(f"{sample}: expression gene order differs from the cohort list")

    multimodal = np.load(source_multimodal, allow_pickle=True)
    barcodes = multimodal["barcodes"].astype(str)
    if "celltype_weights" in multimodal.files:
        celltype = multimodal["celltype_weights"]
        if celltype.size and np.max(np.abs(celltype)) > 1.0e-8:
            raise ValueError(f"{sample}: expected zero cell-type channels")
    expression_barcodes = np.asarray(expression["barcodes"]).astype(str)
    if not np.array_equal(barcodes, expression_barcodes):
        raise ValueError(f"{sample}: expression and UNI2-h feature barcodes differ")

    output_expression_dir = EXP / "data/expression" / sample
    for stale_split in {"train", "val", "test"} - {split_name}:
        stale_path = output_expression_dir / f"{stale_split}.npz"
        if stale_path.exists():
            stale_path.unlink()
    link_or_copy(
        source_expression,
        output_expression_dir / f"{split_name}.npz",
        overwrite=overwrite,
    )
    link_or_copy(
        source_multimodal,
        EXP / "data/multimodal" / sample / "multimodal_features.npz",
        overwrite=overwrite,
    )
    link_or_copy(
        source_graph,
        EXP / "data/graphs" / sample / "graph.npz",
        overwrite=overwrite,
    )
    return {
        "sample": sample,
        "gsm_group": sample_group(sample),
        "donor": gsm_to_donor[sample_group(sample)],
        "split": split_name,
        "n_spots": int(len(barcodes)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    split = build_split()
    gsm_to_donor = load_gsm_to_donor()
    gene_ids = [line.strip() for line in SOURCE_GENES.read_text().splitlines() if line.strip()]
    if len(gene_ids) != 18_397:
        raise ValueError(f"Expected 18,397 genes, found {len(gene_ids)}")

    rows = []
    for split_name in ("train", "val", "test"):
        for sample in split[split_name]:
            rows.append(
                prepare_sample(sample, split_name, gene_ids, gsm_to_donor, args.overwrite)
            )
            print(f"[prepare] {split_name} {sample}", flush=True)

    gene_dir = EXP / "data/gene_splits"
    gene_dir.mkdir(parents=True, exist_ok=True)
    gene_text = "\n".join(gene_ids) + "\n"
    for split_name in ("train", "val", "test"):
        (gene_dir / f"{split_name}_genes.txt").write_text(gene_text)

    normalizer = ModalityNormalizer.compute(
        samples=split["train"], multimodal_root=EXP / "data/multimodal"
    )
    stats_path = EXP / "data/multimodal/modality_stats.json"
    normalizer.save(stats_path)
    (EXP / "data/split.json").write_text(json.dumps(split, indent=2))

    paths = {
        "cohort_name": "hippocampus_gse264692_donor_disjoint",
        "expression_root": relative(EXP / "data/expression"),
        "multimodal_root": relative(EXP / "data/multimodal"),
        "graph_root": relative(EXP / "data/graphs"),
        "modality_stats": relative(stats_path),
        "gene_split_dir": relative(gene_dir),
        "decima_ckpt": "decima_weights/rep0.ckpt",
        "decima_h5": None,
        "decima_npz_dir": "data/decima_input/gene_inputs_npz",
        "seed": 42,
        "max_epochs": 12,
        "min_epochs": 8,
        "max_genes_per_batch": 96,
        "val_max_genes_per_batch": 128,
        "eval_max_genes_per_batch": 128,
        "train_gene_chunks_per_sample": 9,
        "require_full_gene_coverage_per_epoch": True,
        "residual_scale_floor": 1.0e-3,
    }
    (EXP / "data/paths.json").write_text(json.dumps(paths, indent=2))

    metadata = {
        "dataset": "GSE264692",
        "tissue": "human anterior hippocampus",
        "platform": "10x Genomics Visium",
        "n_arrays": 34,
        "n_gsm_groups": 10,
        "n_donors": 10,
        "split_unit": "donor",
        "split_sizes": {key: len(value) for key, value in split.items()},
        "donor_split_sizes": {key: len(value) for key, value in GROUP_SPLIT.items()},
        "donor_disjoint": True,
        "gsm_groups": GROUP_SPLIT,
        "donor_split": {
            split_name: [gsm_to_donor[gsm] for gsm in gsm_groups]
            for split_name, gsm_groups in GROUP_SPLIT.items()
        },
        "gsm_to_donor": gsm_to_donor,
        "n_spots": int(sum(row["n_spots"] for row in rows)),
        "n_genes": len(gene_ids),
        "uses_old_checkpoints_or_results": False,
        "reuses_precomputed_expression_uni2h_and_graph_inputs": True,
        "recomputes_train_only_normalization_and_priors": True,
        "matches_nac_donor_partitions": True,
        "validation_and_test_each_include_one_female_and_one_male_donor": True,
        "balanced_full_gene_coverage_per_epoch": True,
    }
    (EXP / "data/cohort_metadata.json").write_text(json.dumps(metadata, indent=2))
    (EXP / "data/sample_manifest.json").write_text(json.dumps(rows, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
