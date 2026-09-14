#!/usr/bin/env python3
"""Prepare GSE307403 for current UNI2-h full-ablation training."""

from __future__ import annotations
from workflow_paths import DATA_ROOT

import argparse
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

from gnn_dataloader import ModalityNormalizer  # noqa: E402


EXP = ROOT / "experiments/dlpfc_current_full_ablation"
SOURCE_EXPRESSION = ROOT / "experiments/dlpfc_external_generalization/data/expression_full"
SOURCE_MULTIMODAL = ROOT / "experiments/dlpfc_external_generalization/data/multimodal_features"
SOURCE_GRAPHS = ROOT / "experiments/dlpfc_external_generalization/data/spatial_graphs"
SOURCE_UNI2H = ROOT / "experiments/dlpfc_lightweight_sequence_prior_validation/data/histology_embeddings_uni2h"
SOURCE_SPLIT = ROOT / "experiments/dlpfc_lightweight_sequence_prior_validation/data/uni2h_fullgenes/split.json"
SOURCE_GENES = ROOT / "experiments/dlpfc_lightweight_sequence_prior_validation/data/uni2h_fullgenes/gene_ids.txt"


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


def prepare_sample(sample: str, split_name: str, overwrite: bool) -> dict:
    source_multi_path = SOURCE_MULTIMODAL / sample / "multimodal_features.npz"
    source_uni_path = SOURCE_UNI2H / sample / "embeddings.npz"
    source_graph_path = SOURCE_GRAPHS / sample / "graph.npz"
    source_expr_path = SOURCE_EXPRESSION / sample / f"{split_name}.npz"
    for path in (source_multi_path, source_uni_path, source_graph_path, source_expr_path):
        if not path.exists():
            raise FileNotFoundError(path)

    source_multi = np.load(source_multi_path, allow_pickle=True)
    source_barcodes = source_multi["barcodes"].astype(str)
    output_multi_path = EXP / "data/multimodal" / sample / "multimodal_features.npz"
    if overwrite or not output_multi_path.exists():
        uni = np.load(source_uni_path, allow_pickle=True)
        barcodes = source_barcodes
        uni_barcodes = uni["barcodes"].astype(str)
        uni_index = {barcode: idx for idx, barcode in enumerate(uni_barcodes)}
        missing = [barcode for barcode in barcodes if barcode not in uni_index]
        if missing:
            raise KeyError(f"{sample}: {len(missing)} multimodal barcodes lack UNI2-h embeddings")
        uni_order = np.fromiter(
            (uni_index[barcode] for barcode in barcodes),
            dtype=np.int64,
            count=len(barcodes),
        )
        histology = uni["embeddings"][uni_order].astype(np.float32, copy=False)
        spatial = source_multi["spatial_features"].astype(np.float32)
        celltype = source_multi["celltype_weights"].astype(np.float32)
        if np.max(np.abs(celltype)) > 1.0e-8:
            raise ValueError(f"{sample}: expected zero cell-type channels")
        combined = np.concatenate([spatial, histology, celltype], axis=1)
        output_multi_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_multi_path,
            barcodes=barcodes,
            spatial_features=spatial,
            histology_embeddings=histology,
            celltype_weights=celltype,
            combined_features=combined,
            spatial_feature_names=source_multi["spatial_feature_names"].astype(str),
            celltype_names=source_multi["celltype_names"].astype(str),
        )
    else:
        output_multi = np.load(output_multi_path, allow_pickle=True)
        barcodes = output_multi["barcodes"].astype(str)
        if not np.array_equal(barcodes, source_barcodes):
            raise ValueError(f"{sample}: existing output barcodes do not match source data")

    link_or_copy(
        source_graph_path,
        EXP / "data/graphs" / sample / "graph.npz",
        overwrite=overwrite,
    )
    link_or_copy(
        source_expr_path,
        EXP / "data/expression" / sample / f"{split_name}.npz",
        overwrite=overwrite,
    )
    return {"sample": sample, "split": split_name, "n_spots": int(len(barcodes))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source_split = json.loads(SOURCE_SPLIT.read_text())
    split = {
        "train": list(source_split["train_samples"]),
        "val": list(source_split["val_samples"]),
        "test": list(source_split["test_samples"]),
    }
    all_samples = split["train"] + split["val"] + split["test"]
    if len(all_samples) != 63 or len(set(all_samples)) != 63:
        raise ValueError("Expected 63 unique GSE307403 donor samples")

    rows = []
    for split_name in ("train", "val", "test"):
        for sample in split[split_name]:
            rows.append(prepare_sample(sample, split_name, overwrite=args.overwrite))
            print(f"[prepare] {split_name} {sample}", flush=True)

    gene_dir = EXP / "data/gene_splits"
    gene_dir.mkdir(parents=True, exist_ok=True)
    gene_text = SOURCE_GENES.read_text()
    for split_name in ("train", "val", "test"):
        (gene_dir / f"{split_name}_genes.txt").write_text(gene_text)

    normalizer = ModalityNormalizer.compute(
        samples=split["train"], multimodal_root=EXP / "data/multimodal"
    )
    stats_path = EXP / "data/multimodal/modality_stats.json"
    normalizer.save(stats_path)
    (EXP / "data/split.json").write_text(json.dumps(split, indent=2))
    paths = {
        "cohort_name": "dlpfc_gse307403",
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
        "train_gene_chunks_per_sample": 5,
        "require_full_gene_coverage_per_epoch": True,
        "residual_scale_floor": 1.0e-3,
    }
    (EXP / "data/paths.json").write_text(json.dumps(paths, indent=2))
    metadata = {
        "dataset": "GSE307403",
        "title": "Schizophrenia-linked gene expression changes across cortical layers and cellular microenvironments in human prefrontal cortex [Visium]",
        "tissue": "human dorsolateral prefrontal cortex",
        "platform": "10x Genomics Visium v1 3-prime Gene Expression",
        "n_donors": 63,
        "n_schizophrenia": 32,
        "n_neurotypical_controls": 31,
        "one_section_per_donor": True,
        "split_sizes": {key: len(value) for key, value in split.items()},
        "diagnosis_stratified_split": False,
        "diagnosis_split_note": "Public GEO supplementary files do not expose per-donor diagnosis labels.",
        "n_spots": int(sum(row["n_spots"] for row in rows)),
        "n_genes": len([line for line in gene_text.splitlines() if line.strip()]),
        "uses_old_checkpoints_or_results": False,
        "balanced_full_gene_coverage_per_epoch": True,
    }
    (EXP / "data/cohort_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
