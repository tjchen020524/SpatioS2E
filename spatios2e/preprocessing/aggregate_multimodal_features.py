#!/usr/bin/env python3
"""Aggregate spatial, histology, and cell type features into aligned per-spot matrices."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spatial-dir",
        type=Path,
        default=Path("data/processed/spatial_features"),
        help="Per-sample spatial feature CSVs.",
    )
    parser.add_argument(
        "--histology-dir",
        type=Path,
        default=Path("data/processed/histology_embeddings"),
        help="Per-sample histology embedding NPZs.",
    )
    parser.add_argument(
        "--celltype-dir",
        type=Path,
        default=Path("data/processed/cell_type_composition"),
        help="Per-sample RCTD outputs containing celltype_weights.csv.",
    )
    parser.add_argument(
        "--logcpm-dir",
        type=Path,
        default=Path("data/processed/log_cpm"),
        help="Per-sample log CPM folders containing *_barcodes.tsv.gz.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/multimodal_features"),
        help="Where to write aggregated outputs.",
    )
    parser.add_argument(
        "--samples",
        nargs="*",
        default=None,
        help="Optional subset of sample IDs (with GSM prefix) to process.",
    )
    parser.add_argument(
        "--drop-out-of-tissue",
        action="store_true",
        help="Filter out spots with in_tissue==0 using spatial metadata.",
    )
    parser.add_argument(
        "--min-common",
        type=int,
        default=50,
        help="Fail if the intersection of barcodes across modalities is below this threshold.",
    )
    return parser.parse_args()


def normalize_sample_id(sample: str) -> str:
    return sample.split("_", 1)[1] if sample.startswith("GSM") and "_" in sample else sample


def load_log_barcodes(path: Path) -> List[str]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        return [line.strip() for line in handle if line.strip()]


def load_spatial_features(path: Path, drop_out_of_tissue: bool) -> Tuple[List[str], Dict[str, List[float]]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {path}")
        feature_cols = [c for c in reader.fieldnames if c != "barcode"]
        rows: Dict[str, List[float]] = {}
        for row in reader:
            if drop_out_of_tissue and row.get("in_tissue") not in ("1", 1, "True", True):
                continue
            barcode = row["barcode"]
            rows[barcode] = [float(row[col]) for col in feature_cols]
    return feature_cols, rows


def load_celltype_weights(path: Path) -> Tuple[List[str], Dict[str, List[float]]]:
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {path}")
        weight_cols = [c for c in reader.fieldnames if c != "barcode"]
        rows: Dict[str, List[float]] = {}
        for row in reader:
            barcode = row["barcode"]
            rows[barcode] = [float(row[col]) for col in weight_cols]
    return weight_cols, rows


def load_histology_embeddings(path: Path) -> Tuple[np.ndarray, List[str]]:
    data = np.load(path)
    barcodes = data["barcodes"].astype(str).tolist()
    return data["embeddings"], barcodes


def intersection_order(log_order: Sequence[str], allowed: set[str]) -> List[str]:
    return [bc for bc in log_order if bc in allowed]


def aggregate_sample(
    sample_full: str,
    args: argparse.Namespace,
) -> Dict[str, object]:
    sample_base = normalize_sample_id(sample_full)
    log_dir = args.logcpm_dir / sample_full
    spatial_dir = args.spatial_dir / sample_full
    hist_dir = args.histology_dir / sample_full
    cell_dir = args.celltype_dir / sample_base

    spatial_path = spatial_dir / "spatial_features.csv.gz"
    histology_path = hist_dir / "embeddings.npz"
    celltype_path = cell_dir / "celltype_weights.csv"
    barcodes_path = log_dir / f"{sample_full}_barcodes.tsv.gz"

    if not spatial_path.exists() or not histology_path.exists() or not celltype_path.exists():
        raise FileNotFoundError(f"Missing modality for {sample_full}")

    log_barcodes = load_log_barcodes(barcodes_path)
    spatial_cols, spatial_rows = load_spatial_features(spatial_path, args.drop_out_of_tissue)
    celltype_cols, celltype_rows = load_celltype_weights(celltype_path)
    hist_embeddings, hist_barcodes = load_histology_embeddings(histology_path)
    hist_index = {bc: i for i, bc in enumerate(hist_barcodes)}

    common = set(log_barcodes) & set(spatial_rows) & set(celltype_rows) & set(hist_index)
    if len(common) < args.min_common:
        raise ValueError(f"{sample_full}: common barcodes {len(common)} below min_common={args.min_common}")

    ordered_barcodes = intersection_order(log_barcodes, common)
    log_index_map = {bc: i for i, bc in enumerate(log_barcodes)}
    log_indices = [log_index_map[bc] for bc in ordered_barcodes]

    spatial_matrix = np.asarray([spatial_rows[bc] for bc in ordered_barcodes], dtype=np.float32)
    histology_matrix = np.asarray([hist_embeddings[hist_index[bc]] for bc in ordered_barcodes], dtype=np.float32)
    celltype_matrix = np.asarray([celltype_rows[bc] for bc in ordered_barcodes], dtype=np.float32)
    combined = np.concatenate([spatial_matrix, histology_matrix, celltype_matrix], axis=1)

    out_dir = args.output_dir / sample_full
    out_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_dir / "multimodal_features.npz",
        barcodes=np.array(ordered_barcodes),
        spatial_features=spatial_matrix,
        histology_embeddings=histology_matrix,
        celltype_weights=celltype_matrix,
        combined_features=combined,
        spatial_feature_names=np.array(spatial_cols),
        celltype_names=np.array(celltype_cols),
    )

    with (out_dir / "selected_barcodes.tsv").open("w") as handle:
        handle.write("\n".join(ordered_barcodes))

    with (out_dir / "log_cpm_indices.txt").open("w") as handle:
        handle.write("\n".join(str(i) for i in log_indices))

    summary = {
        "sample": sample_full,
        "n_log_cpm": len(log_barcodes),
        "n_spatial": len(spatial_rows),
        "n_histology": len(hist_index),
        "n_celltype": len(celltype_rows),
        "n_common": len(ordered_barcodes),
        "spatial_dim": len(spatial_cols),
        "histology_dim": histology_matrix.shape[1],
        "celltype_dim": len(celltype_cols),
        "combined_dim": combined.shape[1],
        "drop_out_of_tissue": bool(args.drop_out_of_tissue),
        "outputs": {
            "npz": str(out_dir / "multimodal_features.npz"),
            "barcodes": str(out_dir / "selected_barcodes.tsv"),
            "log_cpm_indices": str(out_dir / "log_cpm_indices.txt"),
        },
    }
    with (out_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sample_dirs = sorted(p.name for p in args.logcpm_dir.iterdir() if p.is_dir())
    if args.samples:
        sample_dirs = [s for s in sample_dirs if s in set(args.samples)]
    summaries: List[Dict[str, object]] = []

    for sample_full in sample_dirs:
        try:
            summaries.append(aggregate_sample(sample_full, args))
        except Exception as exc:
            print(f"[WARN] {sample_full}: {exc}")

    with (args.output_dir / "aggregate_summary.json").open("w") as handle:
        json.dump(summaries, handle, indent=2)

    print(f"Completed {len(summaries)} samples. Summary -> {args.output_dir / 'aggregate_summary.json'}")


if __name__ == "__main__":
    main()
