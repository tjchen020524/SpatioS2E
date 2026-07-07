#!/usr/bin/env python
"""
Prepare normalized spatial coordinates and 2D Fourier positional encodings for Visium spots.

For each Visium sample (identified by `*_tissue_positions_list.csv.gz` under --raw-dir),
this script:
  1. Loads spot metadata including pixel-level coordinates.
  2. Computes min-max normalized coordinates in [0, 1] (default using in-tissue spots).
  3. Generates sine/cosine Fourier features for both x and y axes.
  4. Writes a compressed CSV with the augmented features and a JSON summary.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

import csv
import gzip
import math


POSITION_COLUMNS = [
    "barcode",
    "in_tissue",
    "array_row",
    "array_col",
    "pxl_row",
    "pxl_col",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory containing Visium `*_tissue_positions_list.csv.gz` files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/spatial_features"),
        help="Directory to store per-sample spatial feature tables.",
    )
    parser.add_argument(
        "--num-frequencies",
        type=int,
        default=6,
        help="Number of Fourier frequency levels to generate.",
    )
    parser.add_argument(
        "--frequency-base",
        type=float,
        default=2.0,
        help="Geometric base for Fourier frequencies (freq = base**k).",
    )
    parser.add_argument(
        "--use-all-spots",
        action="store_true",
        help="Normalize coordinates using all spots (default uses in-tissue spots only).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs.",
    )
    return parser.parse_args()


def discover_samples(raw_dir: Path) -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    patterns = [
        "*tissue_positions_list.csv.gz",
        "*tissue_positions.csv.gz",
    ]
    for pattern in patterns:
        for path in raw_dir.glob(pattern):
            sample = path.name
            for suffix in (
                "-tissue_positions_list.csv.gz",
                "_tissue_positions_list.csv.gz",
                "-tissue_positions.csv.gz",
                "_tissue_positions.csv.gz",
            ):
                if sample.endswith(suffix):
                    sample = sample[: -len(suffix)]
                    break
            mapping[sample] = path
    return dict(sorted(mapping.items()))


def load_positions(path: Path) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="") as handle:
        reader = csv.reader(handle)
        for barcode, in_tissue, array_row, array_col, pxl_row, pxl_col in reader:
            if barcode == "barcode":
                continue
            rows.append(
                {
                    "barcode": barcode,
                    "in_tissue": int(in_tissue),
                    "array_row": int(array_row),
                    "array_col": int(array_col),
                    "pxl_row": float(pxl_row),
                    "pxl_col": float(pxl_col),
                }
            )
    return rows


def compute_normalized_coords(
    rows: List[Dict[str, float]], use_all_spots: bool = False
) -> Dict[str, float]:
    stats = {}
    for axis, coord_key in (("x", "pxl_col"), ("y", "pxl_row")):
        if use_all_spots:
            values = [row[coord_key] for row in rows]
        else:
            values = [row[coord_key] for row in rows if row["in_tissue"] == 1]
            if not values:
                values = [row[coord_key] for row in rows]
        vmin = min(values)
        vmax = max(values)
        span = vmax - vmin
        if span == 0:
            for row in rows:
                row[f"{axis}_norm"] = 0.0
        else:
            for row in rows:
                value = (row[coord_key] - vmin) / span
                if value < 0.0:
                    value = 0.0
                elif value > 1.0:
                    value = 1.0
                row[f"{axis}_norm"] = value
        stats[f"{axis}_min"] = vmin
        stats[f"{axis}_max"] = vmax
    return stats


def generate_fourier_features(rows: List[Dict[str, float]], frequencies: Iterable[float]) -> None:
    for row in rows:
        for axis in ("x", "y"):
            coord = row[f"{axis}_norm"]
            for freq in frequencies:
                angle = 2.0 * math.pi * freq * coord
                row[f"{axis}_sin_{freq:g}"] = math.sin(angle)
                row[f"{axis}_cos_{freq:g}"] = math.cos(angle)


def process_sample(
    sample: str,
    positions_path: Path,
    output_dir: Path,
    frequencies: Iterable[float],
    use_all_spots: bool,
    overwrite: bool,
) -> Dict[str, float]:

    sample_dir = output_dir / sample
    sample_dir.mkdir(parents=True, exist_ok=True)
    feature_path = sample_dir / "spatial_features.csv.gz"
    summary_path = sample_dir / "spatial_features_summary.json"
    if feature_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {feature_path} (use --overwrite to replace)")

    rows = load_positions(positions_path)
    stats = compute_normalized_coords(rows, use_all_spots=use_all_spots)
    generate_fourier_features(rows, frequencies)

    # Write compressed CSV
    with gzip.open(feature_path, "wt", newline="") as handle:
        fieldnames = list(rows[0].keys()) if rows else POSITION_COLUMNS
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    stats.update(
        {
            "sample": sample,
            "n_spots": len(rows),
            "n_in_tissue": sum(row["in_tissue"] == 1 for row in rows),
            "frequencies": list(frequencies),
            "feature_path": str(feature_path),
        }
    )
    with summary_path.open("w") as handle:
        json.dump(stats, handle, indent=2)
    return stats


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sample_map = discover_samples(args.raw_dir)
    if not sample_map:
        raise FileNotFoundError(f"No tissue position files found under {args.raw_dir}")

    frequencies = [args.frequency_base ** k for k in range(args.num_frequencies)]
    summary: Dict[str, Dict[str, float]] = {}
    for sample, positions_path in sample_map.items():
        stats = process_sample(
            sample=sample,
            positions_path=positions_path,
            output_dir=args.output_dir,
            frequencies=frequencies,
            use_all_spots=args.use_all_spots,
            overwrite=args.overwrite,
        )
        summary[sample] = stats

    aggregate_path = args.output_dir / "spatial_features_summary.json"
    with aggregate_path.open("w") as handle:
        json.dump(summary, handle, indent=2)


if __name__ == "__main__":
    main()
