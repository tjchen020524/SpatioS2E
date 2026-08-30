#!/usr/bin/env python3
"""
Subset Visium log-CPM matrices by predefined gene splits and align barcodes to multimodal features.

Inputs per sample:
  - data/processed/log_cpm/<sample>/matrix_log_cpm.mtx.gz
  - data/processed/log_cpm/<sample>/<sample>_features.tsv.gz (gene_id order)
  - data/processed/log_cpm/<sample>/<sample>_barcodes.tsv.gz (barcode order)
  - data/processed/multimodal_features/<sample>/selected_barcodes.tsv
  - data/processed/multimodal_features/<sample>/log_cpm_indices.txt (indices into log_cpm barcodes)

Gene splits:
  - data/processed/dataset_gene_split/train_genes.txt
  - data/processed/dataset_gene_split/val_genes.txt
  - data/processed/dataset_gene_split/test_genes.txt

Outputs per sample (under --output-dir/<sample>/):
  - train.npz / val.npz / test.npz : CSR sparse matrices with fields (data, indices, indptr, shape, gene_ids, barcodes)
  - summary.json : counts and missing genes

Usage:
  python scripts/subset_expression_by_gene_split.py --samples GSM8226199_V12F14-051_A1
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import logging
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from scipy import sparse
from scipy.io import mmread

LOGGER = logging.getLogger("spatios2e.subset_expression")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logcpm-dir", type=Path, default=Path("data/processed/log_cpm"))
    parser.add_argument("--multimodal-dir", type=Path, default=Path("data/processed/multimodal_features"))
    parser.add_argument("--splits-dir", type=Path, default=Path("data/processed/dataset_gene_split"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/expression_splits"))
    parser.add_argument("--samples", nargs="*", default=None, help="Optional list of sample IDs to process.")
    parser.add_argument(
        "--check-graph",
        action="store_true",
        help="Also verify barcodes match spatial graph barcodes (requires data/processed/spatial_graphs/<sample>/graph.npz).",
    )
    parser.add_argument("--graph-dir", type=Path, default=Path("data/processed/spatial_graphs"))
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(message)s")


def read_lines(path: Path) -> List[str]:
    """Read non-empty, stripped lines."""
    lines: List[str] = []
    with path.open("r") as handle:
        for line in handle:
            item = line.strip()
            if item:
                lines.append(item)
    return lines


def load_gene_lists(splits_dir: Path) -> Dict[str, List[str]]:
    gene_lists = {}
    for split in ("train", "val", "test"):
        path = splits_dir / f"{split}_genes.txt"
        if not path.exists():
            raise FileNotFoundError(f"Missing gene list: {path}")
        gene_lists[split] = read_lines(path)
    return gene_lists


def load_feature_genes(path: Path) -> List[str]:
    """Load gene_ids in order from 10x features file."""
    genes: List[str] = []
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if not row:
                continue
            genes.append(row[0].strip())
    return genes


def load_barcodes(path: Path) -> List[str]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        return [line.strip() for line in handle if line.strip()]


def to_csr(matrix_path: Path) -> sparse.csr_matrix:
    mat = mmread(str(matrix_path))
    if not sparse.issparse(mat):
        raise ValueError(f"{matrix_path} is not a sparse matrix")
    return mat.tocsr()


def save_csr(path: Path, matrix: sparse.csr_matrix, gene_ids: Sequence[str], barcodes: Sequence[str]) -> None:
    np.savez_compressed(
        path,
        data=matrix.data.astype(np.float32),
        indices=matrix.indices.astype(np.int32),
        indptr=matrix.indptr.astype(np.int32),
        shape=np.array(matrix.shape, dtype=np.int32),
        gene_ids=np.array(gene_ids),
        barcodes=np.array(barcodes),
    )


def verify_selected_barcodes(log_barcodes: Sequence[str], selected_indices: Sequence[int], selected_barcodes: Sequence[str]) -> None:
    derived = [log_barcodes[i] for i in selected_indices]
    if derived != list(selected_barcodes):
        raise ValueError("selected_barcodes.tsv does not match log_cpm_indices.txt and log_cpm barcodes.")


def maybe_check_graph(sample: str, selected_barcodes: Sequence[str], graph_dir: Path) -> None:
    graph_path = graph_dir / sample / "graph.npz"
    if not graph_path.exists():
        raise FileNotFoundError(f"--check-graph requested but {graph_path} not found")
    arr = np.load(graph_path, allow_pickle=True)
    graph_barcodes = arr["barcodes"].astype(str).tolist()
    if graph_barcodes != list(selected_barcodes):
        raise ValueError(f"Graph barcodes mismatch for {sample}")


def process_sample(
    sample: str,
    args: argparse.Namespace,
    gene_lists: Dict[str, List[str]],
) -> Dict[str, object]:
    log_dir = args.logcpm_dir / sample
    multimodal_dir = args.multimodal_dir / sample
    matrix_path = log_dir / "matrix_log_cpm.mtx.gz"
    features_path = log_dir / f"{sample}_features.tsv.gz"
    barcodes_path = log_dir / f"{sample}_barcodes.tsv.gz"
    selected_barcodes_path = multimodal_dir / "selected_barcodes.tsv"
    log_indices_path = multimodal_dir / "log_cpm_indices.txt"

    for p in (matrix_path, features_path, barcodes_path, selected_barcodes_path, log_indices_path):
        if not p.exists():
            raise FileNotFoundError(f"{sample}: missing required file {p}")

    feature_genes = load_feature_genes(features_path)
    gene_to_idx = {gid: i for i, gid in enumerate(feature_genes)}

    log_barcodes = load_barcodes(barcodes_path)
    selected_barcodes = read_lines(selected_barcodes_path)
    selected_indices = [int(x) for x in read_lines(log_indices_path)]
    verify_selected_barcodes(log_barcodes, selected_indices, selected_barcodes)

    if args.check_graph:
        maybe_check_graph(sample, selected_barcodes, args.graph_dir)

    full_matrix = to_csr(matrix_path)[:, selected_indices]

    out_dir = args.output_dir / sample
    out_dir.mkdir(parents=True, exist_ok=True)

    split_rows: Dict[str, List[int]] = {}
    missing: Dict[str, List[str]] = {}
    for split, genes in gene_lists.items():
        indices = [gene_to_idx[g] for g in genes if g in gene_to_idx]
        split_rows[split] = indices
        missing[split] = [g for g in genes if g not in gene_to_idx]

    for split, rows in split_rows.items():
        sub = full_matrix[rows, :]
        save_csr(out_dir / f"{split}.npz", sub, [feature_genes[i] for i in rows], selected_barcodes)
        LOGGER.info("%s: saved %s with shape %s", sample, split, sub.shape)

    summary = {
        "sample": sample,
        "matrix_path": str(matrix_path),
        "selected_barcodes": len(selected_barcodes),
        "splits": {split: {"genes": len(gene_lists[split]), "present": len(rows), "missing": len(missing[split])} for split, rows in split_rows.items()},
        "missing_genes": missing,
        "output_dir": str(out_dir),
    }
    with (out_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)

    gene_lists = load_gene_lists(args.splits_dir)

    samples = args.samples
    if samples is None or len(samples) == 0:
        samples = sorted([p.name for p in args.logcpm_dir.iterdir() if p.is_dir()])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for sample in samples:
        LOGGER.info("Processing %s", sample)
        summary = process_sample(sample, args, gene_lists)
        summaries.append(summary)

    overall = {
        "n_samples": len(summaries),
        "samples": summaries,
        "gene_split_dir": str(args.splits_dir),
        "output_dir": str(args.output_dir),
    }
    with (args.output_dir / "summary.json").open("w") as handle:
        json.dump(overall, handle, indent=2)


if __name__ == "__main__":
    main()
