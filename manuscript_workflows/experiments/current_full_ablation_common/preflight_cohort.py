#!/usr/bin/env python3
"""Validate a prepared cohort before launching the ablation array."""

from __future__ import annotations
from workflow_paths import DATA_ROOT

import argparse
import json
from pathlib import Path

import numpy as np
import yaml


ROOT = DATA_ROOT


def resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def load_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def validate_sample(
    sample: str,
    split_name: str,
    paths: dict,
    cohort_genes: list[str],
) -> dict:
    expression_path = resolve(paths["expression_root"]) / sample / f"{split_name}.npz"
    multimodal_path = resolve(paths["multimodal_root"]) / sample / "multimodal_features.npz"
    graph_path = resolve(paths["graph_root"]) / sample / "graph.npz"
    for path in (expression_path, multimodal_path, graph_path):
        if not path.exists() or path.stat().st_size == 0:
            raise FileNotFoundError(path)

    expression = np.load(expression_path, allow_pickle=False)
    multimodal = np.load(multimodal_path, allow_pickle=False)
    graph = np.load(graph_path, allow_pickle=True)
    expression_barcodes = expression["barcodes"].astype(str)
    multimodal_barcodes = multimodal["barcodes"].astype(str)
    graph_barcodes = graph["barcodes"].astype(str)
    if not (
        np.array_equal(expression_barcodes, multimodal_barcodes)
        and np.array_equal(expression_barcodes, graph_barcodes)
    ):
        raise ValueError(f"{sample}: expression, multimodal, and graph barcodes differ")

    gene_ids = expression["gene_ids"].astype(str).tolist()
    if gene_ids != cohort_genes:
        raise ValueError(f"{sample}: expression genes differ from the cohort gene list")
    n_genes, n_spots = map(int, expression["shape"].tolist())
    if n_genes != len(cohort_genes) or n_spots != len(expression_barcodes):
        raise ValueError(f"{sample}: expression shape does not match genes/barcodes")

    spatial_shape = multimodal["spatial_features"].shape
    histology_shape = multimodal["histology_embeddings"].shape
    celltype = multimodal["celltype_weights"]
    combined_shape = multimodal["combined_features"].shape
    component_width = spatial_shape[1] + histology_shape[1] + celltype.shape[1]
    if any(shape[0] != n_spots for shape in (spatial_shape, histology_shape, celltype.shape, combined_shape)):
        raise ValueError(f"{sample}: multimodal row count does not match expression spots")
    if combined_shape[1] != component_width:
        raise ValueError(f"{sample}: combined feature width does not match component widths")
    if celltype.size and float(np.max(np.abs(celltype))) > 1.0e-8:
        raise ValueError(f"{sample}: target-derived cell-type channels are not zero")

    edge_index = graph["edge_index"]
    if edge_index.shape[0] != 2 or edge_index.size == 0:
        raise ValueError(f"{sample}: graph has an invalid edge_index")
    if int(edge_index.min()) < 0 or int(edge_index.max()) >= n_spots:
        raise ValueError(f"{sample}: graph edge index is outside the spot range")
    if graph["edge_weight"].shape != (edge_index.shape[1],):
        raise ValueError(f"{sample}: graph edge weights do not match edge count")

    return {
        "sample": sample,
        "split": split_name,
        "n_spots": n_spots,
        "n_genes": n_genes,
        "spatial_dim": int(spatial_shape[1]),
        "histology_dim": int(histology_shape[1]),
        "celltype_dim": int(celltype.shape[1]),
        "n_edges": int(edge_index.shape[1]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-dir", type=Path, required=True)
    args = parser.parse_args()

    cohort_dir = resolve(args.cohort_dir)
    paths = json.loads((cohort_dir / "data/paths.json").read_text())
    split = json.loads((cohort_dir / "data/split.json").read_text())
    sample_sets = {name: set(split[name]) for name in ("train", "val", "test")}
    if any(not values for values in sample_sets.values()):
        raise ValueError("train, val, and test splits must all be nonempty")
    if (
        sample_sets["train"] & sample_sets["val"]
        or sample_sets["train"] & sample_sets["test"]
        or sample_sets["val"] & sample_sets["test"]
    ):
        raise ValueError("sample leakage detected across train, val, and test")

    gene_dir = resolve(paths["gene_split_dir"])
    cohort_genes = load_lines(gene_dir / "train_genes.txt")
    if len(cohort_genes) != len(set(cohort_genes)):
        raise ValueError("cohort gene list contains duplicates")
    for split_name in ("val", "test"):
        if load_lines(gene_dir / f"{split_name}_genes.txt") != cohort_genes:
            raise ValueError(f"{split_name} gene list differs from train gene list")

    rows = []
    for split_name in ("train", "val", "test"):
        for sample in split[split_name]:
            rows.append(validate_sample(sample, split_name, paths, cohort_genes))
            print(f"[preflight] {split_name} {sample}", flush=True)
    dimensions = {
        (row["spatial_dim"], row["histology_dim"], row["celltype_dim"])
        for row in rows
    }
    if len(dimensions) != 1:
        raise ValueError(f"multimodal dimensions differ across samples: {sorted(dimensions)}")

    for filename in ("train_decima_prior.npz", "train_gene_mean_prior.npz"):
        artifact_path = cohort_dir / "artifacts" / filename
        artifact = np.load(artifact_path, allow_pickle=False)
        if artifact["gene_ids"].astype(str).tolist() != cohort_genes:
            raise ValueError(f"{artifact_path}: genes differ from cohort list")

    manifest = json.loads((cohort_dir / "configs/variant_manifest.json").read_text())
    if len(manifest) != 9 or len({entry["name"] for entry in manifest}) != 9:
        raise ValueError("expected nine unique ablation variants")
    chunk_counts = set()
    capacities = set()
    for entry in manifest:
        config_path = Path(entry["config"])
        cfg = yaml.safe_load(config_path.read_text())
        if cfg["split"] != split:
            raise ValueError(f"{entry['name']}: config split differs from cohort split")
        optim = cfg["optim"]
        chunk_width = int(optim["max_genes_per_batch"])
        chunks_per_sample = int(optim.get("train_gene_chunks_per_sample", 1))
        gene_capacity = len(split["train"]) * chunks_per_sample * chunk_width
        chunk_counts.add(chunks_per_sample)
        capacities.add(gene_capacity)
        if (
            bool(optim.get("require_full_gene_coverage_per_epoch", False))
            and gene_capacity < len(cohort_genes)
        ):
            raise ValueError(
                f"{entry['name']}: per-epoch gene capacity {gene_capacity} is "
                f"smaller than the {len(cohort_genes)}-gene panel"
            )
        variant = cfg["model"]["variant"]
        target_mode = cfg["loss"]["delta_target_mode"]
        if variant == "image_graph_only" and target_mode != "centered_true":
            raise ValueError(f"{entry['name']}: image-only model has incompatible residual target")
        if variant != "image_graph_only" and target_mode != "baseline_relative":
            raise ValueError(f"{entry['name']}: sequence model has incompatible residual target")
    if len(chunk_counts) != 1 or len(capacities) != 1:
        raise ValueError("ablation variants use different per-epoch gene capacities")

    summary = {
        "cohort": paths["cohort_name"],
        "split_sizes": {name: len(split[name]) for name in ("train", "val", "test")},
        "n_samples": len(rows),
        "n_spots": int(sum(row["n_spots"] for row in rows)),
        "n_genes": len(cohort_genes),
        "train_gene_chunks_per_sample": int(next(iter(chunk_counts))),
        "per_epoch_gene_capacity": int(next(iter(capacities))),
        "feature_dimensions": list(next(iter(dimensions))),
        "n_variants": len(manifest),
        "status": "passed",
    }
    output = cohort_dir / "artifacts/preflight.json"
    output.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
