#!/usr/bin/env python
"""
Normalize Visium UMI counts to log(CPM + 1) scale.

For each Visium dataset located under the raw directory (expected 10x Genomics
format with matrix.mtx.gz, features.tsv.gz, and barcodes.tsv.gz), this script:
    1. Reads the sparse matrix in Matrix Market coordinate format.
    2. Computes counts per million (CPM) for each spot (barcode).
    3. Applies log(CPM + 1) transform to align with Decima preprocessing.
    4. Writes a new Matrix Market file with floating-point entries along with
       copies of the original barcodes/features.

The output directory mirrors the raw structure, e.g.:
data/processed/log_cpm/<sample>/matrix.mtx.gz
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import math
import shutil
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


LOGGER = logging.getLogger("spatios2e.normalize_visium")


@dataclass
class MatrixEntry:
    """Sparse matrix entry in coordinate (row, column, value) form."""

    row: int
    col: int
    value: float


def configure_logging(verbose: bool = False) -> None:
    """Configure basic logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(message)s")


def load_matrix(mtx_path: Path) -> Tuple[int, int, List[MatrixEntry], Dict[int, float]]:
    """Load matrix entries and compute per-column sums."""
    if mtx_path.suffix == ".gz":
        opener = lambda: gzip.open(mtx_path, "rt")  # noqa: E731
    else:
        opener = lambda: mtx_path.open("rt")  # noqa: E731

    handle = opener()
    try:
        header = next(handle).strip()
        if not header.startswith("%%MatrixMarket matrix coordinate"):
            raise ValueError(f"Unsupported Matrix Market header: {header}")

        # Skip comment lines until we read the size specification
        line = next(handle).strip()
        while line.startswith("%"):
            line = next(handle).strip()
        n_rows, n_cols, n_nnz = map(int, line.split())

        entries: List[MatrixEntry] = []
        col_sums: Dict[int, float] = defaultdict(float)

        for line in handle:
            if not line or line.startswith("%"):
                continue
            row_str, col_str, val_str = line.strip().split()
            row = int(row_str)
            col = int(col_str)
            val = float(val_str)
            entries.append(MatrixEntry(row, col, val))
            col_sums[col] += val
    finally:
        handle.close()

    LOGGER.debug(
        "Loaded %s with shape (%d genes, %d spots); non-zero entries: %d.",
        mtx_path.name,
        n_rows,
        n_cols,
        n_nnz,
    )
    return n_rows, n_cols, entries, col_sums


def write_matrix(output_path: Path, n_rows: int, n_cols: int, entries: List[MatrixEntry]) -> None:
    """Write matrix entries back to a Matrix Market coordinate file (gzipped)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output_path, "wt") as handle:
        handle.write("%%MatrixMarket matrix coordinate real general\n")
        handle.write(f"{n_rows} {n_cols} {len(entries)}\n")
        for entry in entries:
            handle.write(f"{entry.row} {entry.col} {entry.value:.10f}\n")


def normalize_entries(entries: List[MatrixEntry], col_sums: Dict[int, float], n_cols: int) -> Dict[str, float]:
    """In-place log(CPM + 1) transform of matrix entries. Returns summary stats."""
    zero_spots = 0
    min_total = math.inf
    max_total = 0.0
    totals = []

    if not entries:
        return {"zero_spots": 0, "min_total": 0.0, "max_total": 0.0, "mean_total": 0.0}

    for col in range(1, n_cols + 1):
        total = col_sums.get(col, 0.0)
        totals.append(total)
        if total == 0:
            zero_spots += 1
        else:
            min_total = min(min_total, total)
            max_total = max(max_total, total)

    if min_total is math.inf:
        min_total = 0.0

    for entry in entries:
        total = col_sums.get(entry.col, 0.0)
        cpm = (entry.value / total * 1e6) if total > 0 else 0.0
        entry.value = math.log1p(cpm)

    mean_total = statistics.mean(totals) if totals else 0.0

    return {
        "zero_spots": zero_spots,
        "min_total": float(min_total),
        "max_total": float(max_total),
        "mean_total": float(mean_total),
    }


def copy_sidecar_files(sample_prefix: str, raw_dir: Path, output_dir: Path) -> None:
    """Copy features/barcodes/scalefactors to output directory for completeness."""
    sidecars = [
        f"{sample_prefix}_features.tsv.gz",
        f"{sample_prefix}_barcodes.tsv.gz",
        f"{sample_prefix}-scalefactors_json.json.gz",
        f"{sample_prefix}-tissue_positions_list.csv.gz",
    ]
    for filename in sidecars:
        src = raw_dir / filename
        if src.exists():
            dst = output_dir / filename
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)


def process_sample(sample_prefix: str, raw_dir: Path, output_dir: Path) -> Dict[str, float]:
    """Normalize a single Visium sample."""
    matrix_path = raw_dir / f"{sample_prefix}_matrix.mtx.gz"
    if not matrix_path.exists():
        raise FileNotFoundError(f"Matrix file not found for sample {sample_prefix}: {matrix_path}")

    n_rows, n_cols, entries, col_sums = load_matrix(matrix_path)
    stats = normalize_entries(entries, col_sums, n_cols)

    sample_out_dir = output_dir / sample_prefix
    sample_out_dir.mkdir(parents=True, exist_ok=True)

    out_matrix_path = sample_out_dir / "matrix_log_cpm.mtx.gz"
    write_matrix(out_matrix_path, n_rows, n_cols, entries)
    copy_sidecar_files(sample_prefix, raw_dir, sample_out_dir)

    stats.update(
        {
            "genes": n_rows,
            "spots": n_cols,
            "nnz": len(entries),
            "matrix_path": str(out_matrix_path),
        }
    )
    LOGGER.info(
        "Processed %s: %d genes, %d spots, %d nnz, zero spots %d.",
        sample_prefix,
        n_rows,
        n_cols,
        len(entries),
        stats["zero_spots"],
    )
    return stats


def find_samples(raw_dir: Path) -> List[str]:
    """Discover sample prefixes within the raw directory."""
    prefixes = set()
    for matrix in raw_dir.glob("*_matrix.mtx.gz"):
        prefix = matrix.name.replace("_matrix.mtx.gz", "")
        prefixes.add(prefix)
    return sorted(prefixes)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory containing Visium files (*.mtx.gz, *_barcodes.tsv.gz, *_features.tsv.gz).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/log_cpm"),
        help="Directory to write normalized outputs.",
    )
    parser.add_argument(
        "--samples",
        nargs="*",
        default=None,
        help="Optional list of sample prefixes to process. Defaults to all discovered samples.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    args = parser.parse_args()

    configure_logging(args.verbose)

    raw_dir = args.raw_dir
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw directory not found: {raw_dir}")

    samples = args.samples or find_samples(raw_dir)
    if not samples:
        raise FileNotFoundError(f"No matrix files found under {raw_dir}")

    summary = {}
    for sample in samples:
        summary[sample] = process_sample(sample, raw_dir, args.output_dir)

    summary_path = args.output_dir / "log_cpm_summary.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w") as handle:
        json.dump(summary, handle, indent=2)

    LOGGER.info("Wrote normalization summary to %s", summary_path)


if __name__ == "__main__":
    main()
