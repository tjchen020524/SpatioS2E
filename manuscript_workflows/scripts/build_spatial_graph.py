#!/usr/bin/env python3
"""Construct kNN graphs for each sample using spatial coordinates.

Input:
  - data/processed/multimodal_features/<sample>/multimodal_features.npz
    (produced by scripts/aggregate_multimodal_features.py)

Output:
  - data/processed/spatial_graphs/<sample>/graph.npz containing:
      edge_index: shape (2, E) int64 COO edges
      edge_weight: optional shape (E,) float32 (exp(-dist/temperature))
      positions:   shape (N, 2) float32 (x_norm, y_norm)
      barcodes:    shape (N,) barcodes aligned with edge_index
      sample:      sample identifier
      metadata:    dict with k, radius, metric, normalize, n_nodes, n_edges
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--multimodal-dir",
        type=Path,
        default=Path("data/processed/multimodal_features"),
        help="Directory containing per-sample multimodal_features.npz files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/spatial_graphs"),
        help="Where to write per-sample graph npz files.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=6,
        help="Number of nearest neighbors (per node) excluding self.",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=None,
        help="Optional distance cutoff; edges beyond this threshold are dropped "
        "(Euclidean: squared distance, Cosine: cosine distance). If unset, keep top-k.",
    )
    parser.add_argument(
        "--metric",
        choices=("euclidean", "cosine"),
        default="euclidean",
        help="Distance metric for kNN.",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Z-score features (per column) before distance computation.",
    )
    parser.add_argument(
        "--edge-temperature",
        type=float,
        default=0.1,
        help="Scale for edge_weight = exp(-dist/temperature). Ignored if edge weights are not needed.",
    )
    parser.add_argument(
        "--samples",
        nargs="*",
        default=None,
        help="Optional list of sample folder names to process.",
    )
    return parser.parse_args()


def load_multimodal(sample_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    npz_path = sample_dir / "multimodal_features.npz"
    data = np.load(npz_path, allow_pickle=False)
    barcodes = data["barcodes"].astype(str)
    if "spatial_features" not in data.files or "spatial_feature_names" not in data.files:
        raise ValueError(f"{npz_path} missing spatial_features/spatial_feature_names for coordinate lookup.")

    feature_names = data["spatial_feature_names"].astype(str).tolist()
    try:
        xi = feature_names.index("x_norm")
        yi = feature_names.index("y_norm")
    except ValueError as exc:
        raise ValueError(f"{npz_path} spatial_feature_names must include x_norm and y_norm.") from exc

    positions = data["spatial_features"][:, [xi, yi]].astype(np.float32)
    return positions, barcodes


def preprocess_features(features: np.ndarray, normalize: bool, metric: str) -> np.ndarray:
    feats = features.astype(np.float32)
    if normalize:
        mean = feats.mean(axis=0, keepdims=True)
        std = feats.std(axis=0, keepdims=True) + 1e-8
        feats = (feats - mean) / std
    if metric == "cosine":
        norms = np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8
        feats = feats / norms
    return feats


def build_knn_edges(
    features: np.ndarray, k: int, radius: Optional[float], metric: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute kNN using dense distance matrix (fast BLAS, ~100MB for 5k nodes)."""
    n = features.shape[0]
    if n == 0:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0,), dtype=np.float32)

    if metric == "euclidean":
        norms = np.sum(features * features, axis=1, dtype=np.float64)
        dist2 = norms[:, None] + norms[None, :] - 2.0 * np.dot(features.astype(np.float64), features.T.astype(np.float64))
        dist2[dist2 < 0] = 0.0  # numerical floor
    elif metric == "cosine":
        sim = np.dot(features.astype(np.float64), features.T.astype(np.float64))
        # cosine distance in [0,2]
        dist2 = 2.0 - 2.0 * sim
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    np.fill_diagonal(dist2, np.inf)
    k_eff = min(k, max(n - 1, 1))
    neighbors = np.argpartition(dist2, kth=k_eff - 1, axis=1)[:, :k_eff]

    dists_flat = dist2[np.arange(n)[:, None], neighbors].reshape(-1)
    rows = np.repeat(np.arange(n, dtype=np.int64), k_eff)
    cols = neighbors.reshape(-1).astype(np.int64)

    if radius is not None:
        mask = dists_flat <= radius * radius if metric == "euclidean" else dists_flat <= radius
        rows = rows[mask]
        cols = cols[mask]
        dists_flat = dists_flat[mask]

    edge_index = np.stack([rows, cols], axis=0)
    return edge_index, dists_flat.astype(np.float32)


def process_sample(
    sample_dir: Path,
    output_dir: Path,
    k: int,
    radius: Optional[float],
    metric: str,
    normalize: bool,
    edge_temperature: float,
) -> Dict[str, object]:
    positions, barcodes = load_multimodal(sample_dir)
    feats = preprocess_features(positions, normalize=normalize, metric=metric)
    edge_index, edge_dist = build_knn_edges(feats, k=k, radius=radius, metric=metric)
    if edge_index.shape[1] == 0 and radius is not None:
        # Fallback: disable radius if it filtered everything.
        edge_index, edge_dist = build_knn_edges(feats, k=k, radius=None, metric=metric)
        radius_used = None
    else:
        radius_used = radius
    edge_weight = np.exp(-edge_dist / edge_temperature).astype(np.float32)

    out_dir = output_dir / sample_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "graph.npz"

    meta = {
        "sample": sample_dir.name,
        "k": k,
        "radius": radius_used,
        "n_nodes": positions.shape[0],
        "n_edges": int(edge_index.shape[1]),
        "feature_dim": int(positions.shape[1]),
        "metric": metric,
        "normalize": normalize,
        "edge_temperature": edge_temperature,
    }

    np.savez_compressed(
        out_path,
        edge_index=edge_index,
        edge_weight=edge_weight,
        positions=positions,
        barcodes=barcodes,
        metadata=json.dumps(meta),
    )
    return meta


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sample_dirs = sorted(p for p in args.multimodal_dir.iterdir() if p.is_dir())
    if args.samples:
        sample_set = set(args.samples)
        sample_dirs = [p for p in sample_dirs if p.name in sample_set]

    summaries = []
    for sample_dir in sample_dirs:
        try:
            summaries.append(
                process_sample(
                    sample_dir,
                    args.output_dir,
                    args.k,
                    args.radius,
                    metric=args.metric,
                    normalize=args.normalize,
                    edge_temperature=args.edge_temperature,
                )
            )
        except Exception as exc:
            print(f"[WARN] {sample_dir.name}: {exc}")

    summary_path = args.output_dir / "spatial_graphs_summary.json"
    with summary_path.open("w") as handle:
        json.dump(summaries, handle, indent=2)

    print(f"Finished {len(summaries)} samples. Summary -> {summary_path}")


if __name__ == "__main__":
    main()
