#!/usr/bin/env python3
"""Sanity-check alignment between spatial, histology, cell type, and log-CPM inputs."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path
from typing import Dict, Iterable, List, Set

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
        "--output-json",
        type=Path,
        default=Path("reports/multimodal_alignment_summary.json"),
        help="Where to write a JSON summary table.",
    )
    parser.add_argument(
        "--drop-out-of-tissue",
        action="store_true",
        help="Ignore spatial/cell-type entries with in_tissue==0.",
    )
    return parser.parse_args()


def load_barcodes_from_tsv(path: Path) -> List[str]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        return [line.strip() for line in handle if line.strip()]


def load_spatial_barcodes(path: Path, drop_out_of_tissue: bool) -> Set[str]:
    barcodes: Set[str] = set()
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if drop_out_of_tissue and row.get("in_tissue") not in ("1", 1, "True", True):
                continue
            barcode = row.get("barcode")
            if barcode:
                barcodes.add(barcode)
    return barcodes


def load_celltype_barcodes(path: Path, drop_out_of_tissue: bool) -> Set[str]:
    barcodes: Set[str] = set()
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            barcode = row.get("barcode")
            if not barcode:
                continue
            if drop_out_of_tissue and row.get("in_tissue") not in ("1", 1, "True", True, None):
                # RCTD weights usually lack in_tissue, so this only filters when present.
                continue
            barcodes.add(barcode)
    return barcodes


def load_histology_barcodes(path: Path) -> Set[str]:
    data = np.load(path, allow_pickle=False)
    return set(data["barcodes"].astype(str))


def normalize_sample_id(sample: str) -> str:
    """Strip GSM prefix for matching against cell type outputs."""
    return sample.split("_", 1)[1] if sample.startswith("GSM") and "_" in sample else sample


def discover_samples(logcpm_dir: Path) -> List[str]:
    return sorted(p.name for p in logcpm_dir.iterdir() if p.is_dir())


def main() -> None:
    args = parse_args()
    args.output_json.parent.mkdir(parents=True, exist_ok=True)

    samples = discover_samples(args.logcpm_dir)
    if not samples:
        raise SystemExit(f"No samples found under {args.logcpm_dir}")

    summary: Dict[str, Dict[str, int]] = {}
    missing_modality = {"spatial": [], "histology": [], "celltype": []}

    for sample_full in samples:
        sample_base = normalize_sample_id(sample_full)
        log_barcodes_path = args.logcpm_dir / sample_full / f"{sample_full}_barcodes.tsv.gz"
        spatial_path = args.spatial_dir / sample_full / "spatial_features.csv.gz"
        histology_path = args.histology_dir / sample_full / "embeddings.npz"
        celltype_path = args.celltype_dir / sample_base / "celltype_weights.csv"

        if not spatial_path.exists():
            missing_modality["spatial"].append(sample_full)
        if not histology_path.exists():
            missing_modality["histology"].append(sample_full)
        if not celltype_path.exists():
            missing_modality["celltype"].append(sample_full)

        log_barcodes = set(load_barcodes_from_tsv(log_barcodes_path))
        spatial_barcodes = load_spatial_barcodes(spatial_path, args.drop_out_of_tissue)
        histology_barcodes = load_histology_barcodes(histology_path) if histology_path.exists() else set()
        celltype_barcodes = load_celltype_barcodes(celltype_path, args.drop_out_of_tissue) if celltype_path.exists() else set()

        intersect = log_barcodes & spatial_barcodes & histology_barcodes & celltype_barcodes
        summary[sample_full] = {
            "log_cpm": len(log_barcodes),
            "spatial_features": len(spatial_barcodes),
            "histology_embeddings": len(histology_barcodes),
            "cell_type_composition": len(celltype_barcodes),
            "intersection_all": len(intersect),
        }

    with args.output_json.open("w") as handle:
        json.dump({"per_sample": summary, "missing": missing_modality}, handle, indent=2)

    print(f"Checked {len(samples)} samples. Per-sample counts written to {args.output_json}")
    mismatched = [s for s, stats in summary.items() if len({stats[k] for k in ("log_cpm", "spatial_features", "histology_embeddings", "cell_type_composition")}) > 1]
    if mismatched:
        print(f"Samples with mismatched counts: {len(mismatched)} (e.g., {', '.join(mismatched[:5])})")
    if any(missing_modality.values()):
        print("Missing modalities:", missing_modality)


if __name__ == "__main__":
    main()
