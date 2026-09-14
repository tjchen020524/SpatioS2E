#!/usr/bin/env python3
"""
Dataset/Dataloader skeleton for SpatioS2E graphs with aligned multimodal features and expression splits.

Each item corresponds to one Visium sample graph:
  - x: node features. Either raw `combined_features` or per-modality
    (spatial / histology / celltype) normalized then concatenated.
  - edge_index / edge_weight: spatial graph
  - y: expression targets for a chosen gene split (spots x n_genes)
  - gene_ids, barcodes, positions, metadata

Assumptions:
  - Expression splits are stored under data/processed/expression_splits/<sample>/<split>.npz
    with fields [data, indices, indptr, shape, gene_ids, barcodes] where shape = (n_genes, n_spots).
  - Multimodal features under data/processed/multimodal_features/<sample>/multimodal_features.npz
    with fields ['combined_features', 'barcodes', ...].
  - Graph under data/processed/spatial_graphs/<sample>/graph.npz with fields
    ['edge_index', 'edge_weight', 'positions', 'barcodes', 'metadata'].
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Set

import numpy as np
import torch
from scipy import sparse


def to_torch(array: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
    """Convert numpy-like arrays robustly across torch/numpy version combinations."""
    return torch.tensor(np.asarray(array), dtype=dtype)


def load_expression_split(path: Path, to_dense: bool = False) -> Dict[str, object]:
    arr = np.load(path)
    shape = tuple(arr["shape"].tolist())
    csr = sparse.csr_matrix((arr["data"], arr["indices"], arr["indptr"]), shape=shape)
    gene_ids = arr["gene_ids"].astype(str).tolist()
    barcodes = arr["barcodes"].astype(str).tolist()
    matrix = csr.toarray().T if to_dense else csr.transpose()  # spots x genes
    return {"matrix": matrix, "gene_ids": gene_ids, "barcodes": barcodes}


def load_multimodal_components(path: Path) -> Dict[str, object]:
    arr = np.load(path)
    return {
        "barcodes": arr["barcodes"].astype(str).tolist(),
        "spatial": arr["spatial_features"].astype(np.float32),
        "histology": arr["histology_embeddings"].astype(np.float32),
        "celltype": arr["celltype_weights"].astype(np.float32),
        "spatial_feature_names": arr["spatial_feature_names"].astype(str).tolist(),
        "celltype_names": arr["celltype_names"].astype(str).tolist(),
    }


def load_combined_features(path: Path, feature_key: str = "combined_features") -> Dict[str, object]:
    arr = np.load(path)
    barcodes = arr["barcodes"].astype(str).tolist()
    feats = arr[feature_key].astype(np.float32)
    return {"features": feats, "barcodes": barcodes, "feature_key": feature_key}


def load_graph(path: Path) -> Dict[str, object]:
    arr = np.load(path, allow_pickle=True)
    edge_index = arr["edge_index"].astype(np.int64)
    edge_weight = arr["edge_weight"].astype(np.float32)
    positions = arr.get("positions")
    positions = positions.astype(np.float32) if positions is not None else None
    barcodes = arr["barcodes"].astype(str).tolist()
    metadata = arr["metadata"].item() if arr.get("metadata") is not None else {}
    return {
        "edge_index": edge_index,
        "edge_weight": edge_weight,
        "positions": positions,
        "barcodes": barcodes,
        "metadata": metadata,
    }


def assert_alignment(bcs_a: Sequence[str], bcs_b: Sequence[str], bcs_c: Sequence[str], sample: str) -> None:
    if list(bcs_a) != list(bcs_b) or list(bcs_a) != list(bcs_c):
        raise ValueError(f"Barcode mismatch for {sample}: expression/multimodal/graph differ.")


class SpatioS2EDataset(torch.utils.data.Dataset):
    """Graph-level dataset; each item is a full sample graph."""

    def __init__(
        self,
        split: str,
        samples: Optional[Sequence[str]] = None,
        expression_root: Path = Path("data/processed/expression_splits"),
        multimodal_root: Path = Path("data/processed/multimodal_features"),
        graph_root: Path = Path("data/processed/spatial_graphs"),
        feature_key: str = "combined_features",
        targets_dense: bool = False,
        cache: bool = False,
        normalize_modalities: bool = False,
        modality_stats_path: Path = Path("data/processed/multimodal_features/modality_stats.json"),
        gene_filter: Optional[Set[str]] = None,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be one of train/val/test")
        self.split = split
        self.expression_root = expression_root
        self.multimodal_root = multimodal_root
        self.graph_root = graph_root
        self.feature_key = feature_key
        self.targets_dense = targets_dense
        self.cache = cache
        self.normalize_modalities = normalize_modalities
        self.modality_stats_path = modality_stats_path
        self.gene_filter = set(gene_filter) if gene_filter else None

        if samples is None:
            samples = sorted([p.name for p in expression_root.iterdir() if p.is_dir()])
        self.samples = list(samples)
        self._cache: Dict[str, Dict[str, object]] = {}
        self._normalizer = None
        if self.normalize_modalities:
            self._normalizer = ModalityNormalizer.load(modality_stats_path)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        sample = self.samples[idx]
        if self.cache and sample in self._cache:
            return self._cache[sample]

        expr_path = self.expression_root / sample / f"{self.split}.npz"
        multimodal_path = self.multimodal_root / sample / "multimodal_features.npz"
        graph_path = self.graph_root / sample / "graph.npz"

        expr = load_expression_split(expr_path, to_dense=self.targets_dense)
        if self.normalize_modalities:
            multi = load_multimodal_components(multimodal_path)
            feats = self._normalizer.transform(
                spatial=multi["spatial"],
                histology=multi["histology"],
                celltype=multi["celltype"],
            )
            multi_features = feats
            multi_barcodes = multi["barcodes"]
        else:
            multi = load_combined_features(multimodal_path, feature_key=self.feature_key)
            multi_features = multi["features"]
            multi_barcodes = multi["barcodes"]
        graph = load_graph(graph_path)

        assert_alignment(expr["barcodes"], multi_barcodes, graph["barcodes"], sample)

        if self.gene_filter is not None:
            keep_idx = [i for i, gid in enumerate(expr["gene_ids"]) if gid in self.gene_filter]
            if not keep_idx:
                raise ValueError(f"No genes kept after filtering for sample {sample}")
            expr["gene_ids"] = [expr["gene_ids"][i] for i in keep_idx]
            if self.targets_dense:
                expr["matrix"] = expr["matrix"][:, keep_idx]
            else:
                expr["matrix"] = expr["matrix"][:, keep_idx]

        x = to_torch(multi_features, dtype=torch.float32)
        edge_index = to_torch(graph["edge_index"], dtype=torch.long)
        edge_weight = to_torch(graph["edge_weight"], dtype=torch.float32)
        positions = to_torch(graph["positions"], dtype=torch.float32) if graph["positions"] is not None else None
        if self.targets_dense:
            y = to_torch(expr["matrix"], dtype=torch.float32)  # (spots, genes)
        else:
            # torch sparse COO: transpose back to genes x spots -> coalesce -> transpose for spots x genes if needed downstream
            coo = expr["matrix"].tocoo()
            indices = torch.stack(
                [to_torch(coo.row, dtype=torch.long), to_torch(coo.col, dtype=torch.long)], dim=0
            )
            values = to_torch(coo.data, dtype=torch.float32)
            shape = tuple(coo.shape)
            y = torch.sparse_coo_tensor(indices, values, size=shape)

        item = {
            "sample": sample,
            "x": x,
            "edge_index": edge_index,
            "edge_weight": edge_weight,
            "y": y,
            "gene_ids": expr["gene_ids"],
            "barcodes": expr["barcodes"],
            "positions": positions,
            "graph_metadata": graph["metadata"],
        }
        if self.cache:
            self._cache[sample] = item
        return item


def collate_graphs(batch: List[Dict[str, object]]) -> Dict[str, object]:
    """Simple collate that keeps graphs separate (graph-level batching can be handled by PyG if desired)."""
    return {k: [item[k] for item in batch] for k in batch[0]}


class RunningStats:
    """Streaming mean/std tracker (per feature)."""

    def __init__(self, dim: int) -> None:
        self.count = 0
        self.mean = np.zeros(dim, dtype=np.float64)
        self.M2 = np.zeros(dim, dtype=np.float64)

    def update(self, data: np.ndarray) -> None:
        if data.ndim != 2 or data.shape[1] != self.mean.shape[0]:
            raise ValueError("Data shape mismatch for RunningStats")
        # batch stats
        batch_count = data.shape[0]
        batch_mean = data.mean(axis=0, dtype=np.float64)
        batch_M2 = ((data - batch_mean) ** 2).sum(axis=0, dtype=np.float64)

        total_count = self.count + batch_count
        delta = batch_mean - self.mean
        new_mean = self.mean + delta * (batch_count / total_count)
        new_M2 = self.M2 + batch_M2 + (delta * delta) * self.count * batch_count / total_count

        self.count = total_count
        self.mean = new_mean
        self.M2 = new_M2

    def finalize(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.count < 2:
            raise ValueError("Not enough data to compute stats.")
        var = self.M2 / (self.count - 1)
        std = np.sqrt(var + 1e-8)
        return self.mean.astype(np.float32), std.astype(np.float32)


class ModalityNormalizer:
    """Per-modality z-score normalization."""

    def __init__(self, stats: Dict[str, Dict[str, List[float]]]) -> None:
        self.means = {
            k: np.asarray(v["mean"], dtype=np.float32) for k, v in stats.items()
        }
        self.stds = {
            k: np.asarray(v["std"], dtype=np.float32) for k, v in stats.items()
        }
        for k in self.means:
            if self.means[k].shape != self.stds[k].shape:
                raise ValueError(f"Mean/std shape mismatch for modality {k}")

    @classmethod
    def compute(
        cls,
        samples: Sequence[str],
        multimodal_root: Path,
    ) -> "ModalityNormalizer":
        stats = {}
        # Use first sample to infer dims
        first = np.load(multimodal_root / samples[0] / "multimodal_features.npz")
        dims = {
            "spatial": first["spatial_features"].shape[1],
            "histology": first["histology_embeddings"].shape[1],
            "celltype": first["celltype_weights"].shape[1],
        }
        trackers = {k: RunningStats(dim) for k, dim in dims.items()}
        for sample in samples:
            arr = np.load(multimodal_root / sample / "multimodal_features.npz")
            trackers["spatial"].update(arr["spatial_features"].astype(np.float32))
            trackers["histology"].update(arr["histology_embeddings"].astype(np.float32))
            trackers["celltype"].update(arr["celltype_weights"].astype(np.float32))
        stats = {
            k: {"mean": trackers[k].finalize()[0].tolist(), "std": trackers[k].finalize()[1].tolist()}
            for k in trackers
        }
        return cls(stats)

    @classmethod
    def load(cls, path: Path) -> "ModalityNormalizer":
        if not path.exists():
            raise FileNotFoundError(f"Modality stats not found at {path}")
        with path.open("r") as handle:
            stats = json.load(handle)
        return cls(stats)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {k: {"mean": v.tolist(), "std": self.stds[k].tolist()} for k, v in self.means.items()}
        with path.open("w") as handle:
            json.dump(payload, handle, indent=2)

    def transform(
        self,
        spatial: np.ndarray,
        histology: np.ndarray,
        celltype: np.ndarray,
    ) -> np.ndarray:
        def norm(x: np.ndarray, key: str) -> np.ndarray:
            mean = self.means[key]
            std = self.stds[key]
            if x.shape[1] != mean.shape[0]:
                raise ValueError(f"{key} feature dim mismatch: {x.shape[1]} vs {mean.shape[0]}")
            return (x - mean) / (std + 1e-6)

        spatial_n = norm(spatial, "spatial")
        histology_n = norm(histology, "histology")
        celltype_n = norm(celltype, "celltype")
        return np.concatenate([spatial_n, histology_n, celltype_n], axis=1).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick inspection of SpatioS2EDataset shapes.")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--samples", nargs="*", default=None)
    parser.add_argument("--dense-targets", action="store_true")
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--normalize-modalities", action="store_true", help="Apply per-modality z-score using stats file.")
    parser.add_argument(
        "--modality-stats-path",
        type=Path,
        default=Path("data/processed/multimodal_features/modality_stats.json"),
    )
    parser.add_argument(
        "--compute-stats",
        action="store_true",
        help="Compute modality stats over provided samples and save to modality-stats-path.",
    )
    args = parser.parse_args()

    if args.compute_stats:
        if not args.samples:
            raise SystemExit("--compute-stats requires --samples (typically train samples).")
        normalizer = ModalityNormalizer.compute(args.samples, multimodal_root=Path("data/processed/multimodal_features"))
        normalizer.save(args.modality_stats_path)
        print(f"Saved modality stats to {args.modality_stats_path}")
        return

    ds = SpatioS2EDataset(
        split=args.split,
        samples=args.samples,
        targets_dense=args.dense_targets,
        cache=args.cache,
        normalize_modalities=args.normalize_modalities,
        modality_stats_path=args.modality_stats_path,
    )
    print(f"Dataset split={args.split}, n_samples={len(ds)}")
    if len(ds) == 0:
        return
    example = ds[0]
    print(
        json.dumps(
            {
                "sample": example["sample"],
                "x_shape": tuple(example["x"].shape),
                "edge_index_shape": tuple(example["edge_index"].shape),
                "edge_weight_shape": tuple(example["edge_weight"].shape),
                "positions_shape": tuple(example["positions"].shape) if example["positions"] is not None else None,
                "y_shape": tuple(example["y"].shape),
                "gene_ids": len(example["gene_ids"]),
                "barcodes": len(example["barcodes"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
