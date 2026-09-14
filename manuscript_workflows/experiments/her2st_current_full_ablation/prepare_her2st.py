#!/usr/bin/env python3
"""Prepare HER2ST expression, coordinates, annotations, multimodal files, and graphs."""

from __future__ import annotations
from workflow_paths import DATA_ROOT

import argparse
import gzip
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse


ROOT = DATA_ROOT
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.append(str(path))

from build_spatial_graph import process_sample as build_graph  # noqa: E402
from gnn_dataloader import ModalityNormalizer, load_expression_split  # noqa: E402


EXP = ROOT / "experiments/her2st_current_full_ablation"
COUNTS = EXP / "data/source/counts/count-matrices"
IMAGES = EXP / "data/source/images/images/HE"
SELECTIONS = EXP / "data/source/spot_selections"
META = EXP / "data/source/meta"
GENE_MAP = ROOT / "data/processed/gene_mapping/gene_intersection.tsv"
PATIENT_SPLIT = {
    "train": list("ABCDE"),
    "val": ["F"],
    "test": list("GH"),
}


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def discover_samples() -> list[str]:
    samples = sorted(path.name.removesuffix(".tsv.gz") for path in COUNTS.glob("*.tsv.gz"))
    if len(samples) != 36:
        raise ValueError(f"Expected 36 HER2ST sections, found {len(samples)}")
    return samples


def split_for_sample(sample: str) -> str:
    patient = sample[0]
    for split_name, patients in PATIENT_SPLIT.items():
        if patient in patients:
            return split_name
    raise KeyError(sample)


def load_symbol_mapping() -> dict[str, str]:
    mapping = pd.read_csv(GENE_MAP, sep="\t")
    symbols = mapping["st_gene_name"].astype(str)
    counts = Counter(symbols)
    return {
        symbol: gene_id
        for symbol, gene_id in zip(symbols, mapping["gene_id"].astype(str))
        if counts[symbol] == 1
    }


def save_expression(path: Path, matrix_spot_gene: np.ndarray, gene_ids: list[str], barcodes: list[str]) -> None:
    csr = sparse.csr_matrix(matrix_spot_gene.T)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        data=csr.data.astype(np.float32),
        indices=csr.indices.astype(np.int32),
        indptr=csr.indptr.astype(np.int64),
        shape=np.asarray(csr.shape, dtype=np.int64),
        gene_ids=np.asarray(gene_ids, dtype=str),
        barcodes=np.asarray(barcodes, dtype=str),
    )


def align_annotations(sample: str, barcodes: list[str]) -> pd.DataFrame:
    candidates = [META / f"{sample}_labeled_coordinates.tsv"]
    meta_path = next((path for path in candidates if path.exists()), None)
    labels: dict[str, str] = {}
    if meta_path is not None:
        frame = pd.read_csv(meta_path, sep="\t")
        for row in frame.itertuples(index=False):
            x = float(row.x)
            y = float(row.y)
            if not np.isfinite(x) or not np.isfinite(y):
                continue
            spot_id = f"{int(round(x))}x{int(round(y))}"
            labels[spot_id] = str(row.label)
    return pd.DataFrame(
        {
            "sample": sample,
            "spot_id": barcodes,
            "pathology_label": [labels.get(barcode, "") for barcode in barcodes],
        }
    )


def prepare_expression(overwrite: bool) -> None:
    samples = discover_samples()
    symbol_to_ensembl = load_symbol_mapping()
    train_symbols: set[str] = set()
    for sample in samples:
        if split_for_sample(sample) != "train":
            continue
        with gzip.open(COUNTS / f"{sample}.tsv.gz", "rt") as handle:
            train_symbols.update(handle.readline().rstrip("\n").split("\t")[1:])
    gene_symbols = [symbol for symbol in symbol_to_ensembl if symbol in train_symbols]
    gene_ids = [symbol_to_ensembl[symbol] for symbol in gene_symbols]
    if len(gene_ids) != len(set(gene_ids)):
        raise ValueError("HER2ST symbol mapping produced duplicate Ensembl IDs")

    split = {name: [] for name in PATIENT_SPLIT}
    sample_rows = []
    for sample in samples:
        split_name = split_for_sample(sample)
        split[split_name].append(sample)
        expression_path = EXP / "data/expression" / sample / f"{split_name}.npz"
        coordinate_path = EXP / "data/coordinates" / f"{sample}.npz"
        annotation_path = EXP / "data/annotations" / f"{sample}.tsv"
        if overwrite or not expression_path.exists() or not coordinate_path.exists():
            counts = pd.read_csv(COUNTS / f"{sample}.tsv.gz", sep="\t", index_col=0)
            counts = counts.reindex(columns=gene_symbols, fill_value=0)
            selection = pd.read_csv(SELECTIONS / f"{sample}_selection.tsv.gz", sep="\t")
            selection["spot_id"] = [
                f"{int(x)}x{int(y)}" for x, y in selection[["x", "y"]].itertuples(index=False)
            ]
            selection = selection.set_index("spot_id")
            barcodes = [str(barcode) for barcode in counts.index if str(barcode) in selection.index]
            counts = counts.loc[barcodes, gene_symbols]
            values = counts.to_numpy(dtype=np.float64, copy=False)
            library_size = values.sum(axis=1, keepdims=True)
            library_size[library_size <= 0] = 1.0
            log_cpm = np.log1p(values / library_size * 1.0e6).astype(np.float32)
            save_expression(expression_path, log_cpm, gene_ids=gene_ids, barcodes=barcodes)

            coords = selection.loc[barcodes]
            coordinate_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                coordinate_path,
                barcodes=np.asarray(barcodes, dtype=str),
                array_x=coords["x"].to_numpy(dtype=np.float32),
                array_y=coords["y"].to_numpy(dtype=np.float32),
                pixel_x=coords["pixel_x"].to_numpy(dtype=np.float32),
                pixel_y=coords["pixel_y"].to_numpy(dtype=np.float32),
            )
            annotation_path.parent.mkdir(parents=True, exist_ok=True)
            align_annotations(sample, barcodes).to_csv(annotation_path, sep="\t", index=False)
        else:
            barcodes = np.load(coordinate_path)["barcodes"].astype(str).tolist()
        sample_rows.append(
            {
                "sample": sample,
                "patient": sample[0],
                "split": split_name,
                "n_spots": len(barcodes),
                "has_pathology_labels": bool((META / f"{sample}_labeled_coordinates.tsv").exists()),
            }
        )
        print(f"[expression] {sample} spots={len(barcodes)}", flush=True)

    gene_dir = EXP / "data/gene_splits"
    gene_dir.mkdir(parents=True, exist_ok=True)
    gene_text = "\n".join(gene_ids) + "\n"
    for split_name in ("train", "val", "test"):
        (gene_dir / f"{split_name}_genes.txt").write_text(gene_text)
    (EXP / "data/split.json").write_text(json.dumps(split, indent=2))
    pd.DataFrame(sample_rows).to_csv(EXP / "data/sample_manifest.tsv", sep="\t", index=False)
    paths = {
        "cohort_name": "her2st",
        "expression_root": relative(EXP / "data/expression"),
        "multimodal_root": relative(EXP / "data/multimodal"),
        "graph_root": relative(EXP / "data/graphs"),
        "modality_stats": relative(EXP / "data/multimodal/modality_stats.json"),
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
        "train_gene_chunks_per_sample": 7,
        "require_full_gene_coverage_per_epoch": True,
        "residual_scale_floor": 1.0e-3,
    }
    (EXP / "data/paths.json").write_text(json.dumps(paths, indent=2))
    metadata = {
        "dataset": "HER2ST",
        "citation": "Andersson et al., Nature Communications 2021",
        "doi": "10.1038/s41467-021-26271-2",
        "data_doi": "10.5281/zenodo.4751624",
        "platform": "first-generation Spatial Transcriptomics array",
        "spot_diameter_micrometers": 100,
        "center_spacing_micrometers": 200,
        "n_patients": 8,
        "n_sections": 36,
        "patient_split": PATIENT_SPLIT,
        "split_sizes": {key: len(value) for key, value in split.items()},
        "n_spots": int(sum(row["n_spots"] for row in sample_rows)),
        "n_genes": len(gene_ids),
        "balanced_full_gene_coverage_per_epoch": True,
    }
    (EXP / "data/cohort_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


def spatial_features(array_x: np.ndarray, array_y: np.ndarray, pixel_x: np.ndarray, pixel_y: np.ndarray) -> tuple[np.ndarray, list[str]]:
    def normalize(values: np.ndarray) -> np.ndarray:
        span = float(values.max() - values.min())
        return np.zeros_like(values, dtype=np.float32) if span == 0 else ((values - values.min()) / span).astype(np.float32)

    x_norm = normalize(pixel_x)
    y_norm = normalize(pixel_y)
    columns = [
        np.ones_like(array_x, dtype=np.float32),
        array_y.astype(np.float32),
        array_x.astype(np.float32),
        pixel_y.astype(np.float32),
        pixel_x.astype(np.float32),
        x_norm,
        y_norm,
    ]
    names = ["in_tissue", "array_row", "array_col", "pxl_row", "pxl_col", "x_norm", "y_norm"]
    for axis_name, coord in (("x", x_norm), ("y", y_norm)):
        for frequency in (1, 2, 4, 8, 16, 32):
            angle = 2.0 * math.pi * frequency * coord
            columns.extend([np.sin(angle).astype(np.float32), np.cos(angle).astype(np.float32)])
            names.extend([f"{axis_name}_sin_{frequency}", f"{axis_name}_cos_{frequency}"])
    return np.column_stack(columns).astype(np.float32), names


def finalize(overwrite: bool) -> None:
    split = json.loads((EXP / "data/split.json").read_text())
    for split_name in ("train", "val", "test"):
        for sample in split[split_name]:
            output = EXP / "data/multimodal" / sample / "multimodal_features.npz"
            if output.exists() and not overwrite:
                continue
            expr = load_expression_split(
                EXP / "data/expression" / sample / f"{split_name}.npz", to_dense=False
            )
            barcodes = list(expr["barcodes"])
            coords = np.load(EXP / "data/coordinates" / f"{sample}.npz")
            coord_index = {barcode: idx for idx, barcode in enumerate(coords["barcodes"].astype(str))}
            emb = np.load(EXP / "data/uni2h_embeddings" / sample / "embeddings.npz", allow_pickle=True)
            emb_index = {barcode: idx for idx, barcode in enumerate(emb["barcodes"].astype(str))}
            missing = [barcode for barcode in barcodes if barcode not in coord_index or barcode not in emb_index]
            if missing:
                raise KeyError(f"{sample}: {len(missing)} spots missing coordinates or UNI2-h embeddings")
            idx_coord = [coord_index[barcode] for barcode in barcodes]
            idx_emb = [emb_index[barcode] for barcode in barcodes]
            spatial, spatial_names = spatial_features(
                coords["array_x"][idx_coord],
                coords["array_y"][idx_coord],
                coords["pixel_x"][idx_coord],
                coords["pixel_y"][idx_coord],
            )
            histology = emb["embeddings"][idx_emb].astype(np.float32)
            celltype = np.zeros((len(barcodes), 1), dtype=np.float32)
            combined = np.concatenate([spatial, histology, celltype], axis=1)
            output.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                output,
                barcodes=np.asarray(barcodes, dtype=str),
                spatial_features=spatial,
                histology_embeddings=histology,
                celltype_weights=celltype,
                combined_features=combined,
                spatial_feature_names=np.asarray(spatial_names, dtype=str),
                celltype_names=np.asarray(["none"], dtype=str),
            )
            print(f"[multimodal] {sample}", flush=True)

    graph_root = EXP / "data/graphs"
    for sample_dir in sorted(path for path in (EXP / "data/multimodal").iterdir() if path.is_dir()):
        graph_path = graph_root / sample_dir.name / "graph.npz"
        if graph_path.exists() and not overwrite:
            continue
        build_graph(
            sample_dir=sample_dir,
            output_dir=graph_root,
            k=6,
            radius=None,
            metric="euclidean",
            normalize=False,
            edge_temperature=0.1,
        )
        print(f"[graph] {sample_dir.name}", flush=True)

    normalizer = ModalityNormalizer.compute(
        samples=split["train"], multimodal_root=EXP / "data/multimodal"
    )
    normalizer.save(EXP / "data/multimodal/modality_stats.json")
    print(json.dumps({"status": "finalized", "n_samples": sum(map(len, split.values()))}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("expression", "finalize", "all"), default="all")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.stage in ("expression", "all"):
        prepare_expression(overwrite=args.overwrite)
    if args.stage in ("finalize", "all"):
        finalize(overwrite=args.overwrite)


if __name__ == "__main__":
    main()
