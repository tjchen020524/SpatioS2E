#!/usr/bin/env python3
"""Prepare donor-disjoint GSE307586 human nucleus accumbens Visium data."""

from __future__ import annotations
from workflow_paths import DATA_ROOT

import argparse
import csv
import gzip
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmread


ROOT = DATA_ROOT
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.append(str(path))

from build_spatial_graph import process_sample as build_graph  # noqa: E402
from gnn_dataloader import ModalityNormalizer, load_expression_split  # noqa: E402


EXP = ROOT / "experiments/nac_current_full_ablation"
RAW = EXP / "data/source/raw"
SAMPLE_KEY = EXP / "data/source/sample_key_spatial_NAc.csv"
GENE_MAP = ROOT / "data/processed/gene_mapping/gene_intersection.tsv"

GSM_TO_DONOR = {
    **{f"GSM{number}": "Br2720" for number in range(9227428, 9227432)},
    **{f"GSM{number}": "Br8492" for number in range(9227432, 9227436)},
    **{f"GSM{number}": "Br6522" for number in range(9227436, 9227440)},
    "GSM9227440": "Br6423",
    "GSM9227441": "Br6423",
    "GSM9227442": "Br8325",
    "GSM9227443": "Br6423",
    **{f"GSM{number}": "Br8325" for number in range(9227444, 9227448)},
    **{f"GSM{number}": "Br6432" for number in range(9227448, 9227452)},
    **{f"GSM{number}": "Br6471" for number in range(9227452, 9227456)},
    **{f"GSM{number}": "Br2743" for number in range(9227456, 9227460)},
    **{f"GSM{number}": "Br3942" for number in range(9227460, 9227464)},
    **{f"GSM{number}": "Br8667" for number in range(9227464, 9227466)},
}

DONOR_SPLIT = {
    "train": ["Br2720", "Br8325", "Br6522", "Br6432", "Br6471", "Br3942"],
    "val": ["Br8667", "Br6423"],
    "test": ["Br8492", "Br2743"],
}

# These donors also occur in the independently processed GSE307403 dlPFC cohort.
# Within-cohort NAc evaluation remains donor-disjoint, but cross-tissue and joint
# experiments must group these identities across cohorts before splitting.
DLPFC_DONOR_OVERLAP = ["Br2720", "Br6432", "Br8325", "Br8492", "Br8667"]


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def discover_samples() -> list[dict[str, str]]:
    samples = []
    for position_path in sorted(RAW.glob("*tissue_positions_list.csv.gz")):
        suffix = "-tissue_positions_list.csv.gz"
        if not position_path.name.endswith(suffix):
            raise ValueError(position_path)
        prefix = position_path.name[: -len(suffix)]
        gsm = prefix.split("_", 1)[0]
        if gsm not in GSM_TO_DONOR:
            raise KeyError(f"No donor mapping for {gsm}")
        samples.append({"sample": prefix, "gsm": gsm, "donor": GSM_TO_DONOR[gsm]})
    if len(samples) != 38 or len({row["gsm"] for row in samples}) != 38:
        raise ValueError(f"Expected 38 unique GSE307586 samples, found {len(samples)}")
    if set(GSM_TO_DONOR) != {row["gsm"] for row in samples}:
        raise ValueError("GEO sample mapping does not match downloaded archive")
    return samples


def split_for_donor(donor: str) -> str:
    matches = [name for name, donors in DONOR_SPLIT.items() if donor in donors]
    if len(matches) != 1:
        raise ValueError(f"Donor {donor} has invalid split assignment: {matches}")
    return matches[0]


def load_lines_gzip(path: Path) -> list[str]:
    with gzip.open(path, "rt") as handle:
        return [line.rstrip("\n").split("\t")[0] for line in handle if line.strip()]


def load_positions(path: Path) -> pd.DataFrame:
    rows = []
    with gzip.open(path, "rt", newline="") as handle:
        for record in csv.reader(handle):
            if not record or record[0] == "barcode":
                continue
            barcode, in_tissue, array_row, array_col, pixel_row, pixel_col = record
            if int(in_tissue) != 1:
                continue
            rows.append(
                {
                    "barcode": barcode,
                    "array_row": int(array_row),
                    "array_col": int(array_col),
                    "pixel_row_fullres": float(pixel_row),
                    "pixel_col_fullres": float(pixel_col),
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty or frame["barcode"].duplicated().any():
        raise ValueError(f"Invalid tissue positions in {path}")
    return frame


def load_scalefactors(path: Path) -> dict:
    with gzip.open(path, "rt") as handle:
        return json.load(handle)


def save_expression(
    path: Path,
    matrix_gene_spot: sparse.spmatrix,
    gene_ids: list[str],
    barcodes: list[str],
) -> None:
    csr = matrix_gene_spot.tocsr()
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


def normalize_log_cpm(matrix_gene_spot: sparse.spmatrix) -> sparse.csr_matrix:
    matrix = matrix_gene_spot.tocoo(copy=True)
    library_size = np.asarray(matrix.sum(axis=0)).ravel().astype(np.float64)
    library_size[library_size <= 0] = 1.0
    matrix.data = np.log1p(matrix.data.astype(np.float64) / library_size[matrix.col] * 1.0e6)
    return matrix.tocsr().astype(np.float32)


def select_cohort_genes(samples: list[dict[str, str]]) -> list[str]:
    common = None
    for row in samples:
        feature_path = RAW / f"{row['sample']}_features.tsv.gz"
        genes = set(load_lines_gzip(feature_path))
        common = genes if common is None else common & genes
    mapping = pd.read_csv(GENE_MAP, sep="\t")
    ordered = mapping["gene_id"].astype(str).tolist()
    selected = [gene_id for gene_id in ordered if gene_id in (common or set())]
    if len(selected) < 15000 or len(selected) != len(set(selected)):
        raise ValueError(f"Unexpected NAc cohort gene universe: {len(selected)}")
    return selected


def donor_metadata() -> dict[str, dict]:
    frame = pd.read_csv(SAMPLE_KEY)
    frame.columns = [column.strip() for column in frame.columns]
    frame = frame[frame["In analysis"].astype(str).str.lower() == "yes"].copy()
    output = {}
    for donor, donor_frame in frame.groupby("Brain"):
        output[str(donor)] = {
            "age": float(donor_frame["Age"].iloc[0]),
            "sex": str(donor_frame["Sex"].iloc[0]),
            "diagnosis": str(donor_frame["Diagnosis"].iloc[0]),
        }
    return output


def prepare_expression(overwrite: bool) -> None:
    samples = discover_samples()
    genes = select_cohort_genes(samples)
    gene_set = set(genes)
    donor_info = donor_metadata()
    split = {name: [] for name in DONOR_SPLIT}
    manifest_rows = []

    for index, row in enumerate(samples):
        sample = row["sample"]
        donor = row["donor"]
        split_name = split_for_donor(donor)
        split[split_name].append(sample)
        expression_path = EXP / "data/expression" / sample / f"{split_name}.npz"
        coordinate_path = EXP / "data/coordinates" / f"{sample}.npz"

        if overwrite or not expression_path.exists() or not coordinate_path.exists():
            feature_ids = load_lines_gzip(RAW / f"{sample}_features.tsv.gz")
            row_lookup = {gene_id: idx for idx, gene_id in enumerate(feature_ids)}
            if not gene_set.issubset(row_lookup):
                raise ValueError(f"{sample}: selected genes missing from feature table")
            barcodes_all = load_lines_gzip(RAW / f"{sample}_barcodes.tsv.gz")
            positions = load_positions(RAW / f"{sample}-tissue_positions_list.csv.gz")
            barcode_lookup = {barcode: idx for idx, barcode in enumerate(barcodes_all)}
            missing = [barcode for barcode in positions["barcode"] if barcode not in barcode_lookup]
            if missing:
                raise ValueError(f"{sample}: {len(missing)} tissue barcodes absent from matrix")

            with gzip.open(RAW / f"{sample}_matrix.mtx.gz", "rb") as handle:
                matrix = mmread(handle).tocsr()
            if matrix.shape != (len(feature_ids), len(barcodes_all)):
                raise ValueError(f"{sample}: matrix shape {matrix.shape} does not match features/barcodes")
            gene_indices = np.asarray([row_lookup[gene_id] for gene_id in genes], dtype=np.int64)
            spot_indices = np.asarray(
                [barcode_lookup[barcode] for barcode in positions["barcode"]], dtype=np.int64
            )
            matrix = normalize_log_cpm(matrix[gene_indices][:, spot_indices])
            barcodes = positions["barcode"].astype(str).tolist()
            save_expression(expression_path, matrix, genes, barcodes)

            scales = load_scalefactors(RAW / f"{sample}-scalefactors_json.json.gz")
            hires_scale = float(scales["tissue_hires_scalef"])
            coordinate_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                coordinate_path,
                barcodes=np.asarray(barcodes, dtype=str),
                array_x=positions["array_col"].to_numpy(dtype=np.float32),
                array_y=positions["array_row"].to_numpy(dtype=np.float32),
                pixel_x=(positions["pixel_col_fullres"] * hires_scale).to_numpy(dtype=np.float32),
                pixel_y=(positions["pixel_row_fullres"] * hires_scale).to_numpy(dtype=np.float32),
                tissue_hires_scalef=np.asarray([hires_scale], dtype=np.float32),
                spot_diameter_hires=np.asarray(
                    [float(scales["spot_diameter_fullres"]) * hires_scale], dtype=np.float32
                ),
            )
        else:
            barcodes = np.load(coordinate_path, allow_pickle=False)["barcodes"].astype(str).tolist()

        info = donor_info[donor]
        manifest_rows.append(
            {
                **row,
                "split": split_name,
                "age": info["age"],
                "sex": info["sex"],
                "diagnosis": info["diagnosis"],
                "n_spots": len(barcodes),
            }
        )
        print(f"[expression] {index + 1}/38 {split_name} {sample} spots={len(barcodes)}", flush=True)

    split_donors = {
        name: sorted({row["donor"] for row in manifest_rows if row["split"] == name})
        for name in DONOR_SPLIT
    }
    if split_donors != {name: sorted(donors) for name, donors in DONOR_SPLIT.items()}:
        raise ValueError(f"Donor split mismatch: {split_donors}")

    gene_dir = EXP / "data/gene_splits"
    gene_dir.mkdir(parents=True, exist_ok=True)
    gene_text = "\n".join(genes) + "\n"
    for split_name in ("train", "val", "test"):
        (gene_dir / f"{split_name}_genes.txt").write_text(gene_text)
    (EXP / "data/split.json").write_text(json.dumps(split, indent=2))
    pd.DataFrame(manifest_rows).to_csv(EXP / "data/sample_manifest.tsv", sep="\t", index=False)

    paths = {
        "cohort_name": "nac_gse307586",
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
        "train_gene_chunks_per_sample": 8,
        "require_full_gene_coverage_per_epoch": True,
        "residual_scale_floor": 1.0e-3,
    }
    (EXP / "data/paths.json").write_text(json.dumps(paths, indent=2))
    metadata = {
        "dataset": "GSE307586",
        "title": "Spatiomolecular mapping reveals anatomical organization of heterogeneous cell types in the human nucleus accumbens",
        "citation_pmid": "40964296",
        "zenodo_doi": "10.5281/zenodo.17089020",
        "tissue": "human nucleus accumbens",
        "platform": "10x Genomics Visium",
        "n_donors": 10,
        "n_sections": 38,
        "donor_split": DONOR_SPLIT,
        "hippocampus_gse264692_donor_overlap": sorted(
            donor for donors in DONOR_SPLIT.values() for donor in donors
        ),
        "matches_hippocampus_donor_partitions": True,
        "dlpfc_gse307403_donor_overlap": DLPFC_DONOR_OVERLAP,
        "cross_cohort_split_constraint": (
            "For zero-shot or joint training, donor identities must remain in one "
            "partition across hippocampus, NAc, and dlPFC."
        ),
        "split_sizes": {name: len(values) for name, values in split.items()},
        "sex_by_split": {
            name: [donor_info[donor]["sex"] for donor in donors]
            for name, donors in DONOR_SPLIT.items()
        },
        "n_spots": int(sum(row["n_spots"] for row in manifest_rows)),
        "n_genes": len(genes),
        "image_source": "GEO Space Ranger tissue_hires_image",
        "image_max_dimension_pixels": 2000,
        "original_wsi_available_in_geo": False,
        "uses_old_checkpoints_or_results": False,
        "balanced_full_gene_coverage_per_epoch": True,
    }
    (EXP / "data/cohort_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


def spatial_features(
    array_x: np.ndarray,
    array_y: np.ndarray,
    pixel_x: np.ndarray,
    pixel_y: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    def normalize(values: np.ndarray) -> np.ndarray:
        span = float(values.max() - values.min())
        if span == 0:
            return np.zeros_like(values, dtype=np.float32)
        return ((values - values.min()) / span).astype(np.float32)

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
            expression = load_expression_split(
                EXP / "data/expression" / sample / f"{split_name}.npz", to_dense=False
            )
            barcodes = list(expression["barcodes"])
            coords = np.load(EXP / "data/coordinates" / f"{sample}.npz", allow_pickle=False)
            coord_index = {barcode: idx for idx, barcode in enumerate(coords["barcodes"].astype(str))}
            embedding_path = EXP / "data/uni2h_embeddings" / sample / "embeddings.npz"
            embeddings = np.load(embedding_path, allow_pickle=True)
            embedding_index = {
                barcode: idx for idx, barcode in enumerate(embeddings["barcodes"].astype(str))
            }
            missing = [
                barcode
                for barcode in barcodes
                if barcode not in coord_index or barcode not in embedding_index
            ]
            if missing:
                raise KeyError(f"{sample}: {len(missing)} spots lack coordinates or UNI2-h embeddings")
            coord_order = [coord_index[barcode] for barcode in barcodes]
            embedding_order = [embedding_index[barcode] for barcode in barcodes]
            spatial, spatial_names = spatial_features(
                coords["array_x"][coord_order],
                coords["array_y"][coord_order],
                coords["pixel_x"][coord_order],
                coords["pixel_y"][coord_order],
            )
            histology = embeddings["embeddings"][embedding_order].astype(np.float32)
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
