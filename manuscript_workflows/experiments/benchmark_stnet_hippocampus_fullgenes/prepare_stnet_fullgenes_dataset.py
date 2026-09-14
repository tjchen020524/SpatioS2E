#!/usr/bin/env python3
from __future__ import annotations
from workflow_paths import DATA_ROOT

import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image
from scipy import sparse


ROOT = DATA_ROOT
CONFIG_PATH = ROOT / "experiments/sample_split_fullgenes/configs/stage0_reg_sample_split_fullgenes.yaml"
EXPRESSION_ROOT = ROOT / "experiments/sample_split_fullgenes/expression_full"
RAW_ROOT = ROOT / "data/raw"
OUT_ROOT = ROOT / "experiments/benchmark_stnet_hippocampus_fullgenes"
RAW_OUT = OUT_ROOT / "raw"


def parse_sample(sample: str) -> tuple[str, str, str]:
    gsm, slide, sec = sample.split("_")
    return gsm, slide, sec


def load_positions(sample_dir: Path) -> pd.DataFrame:
    with gzip.open(sample_dir / "tissue_positions_list.csv.gz", "rt") as f:
        first = f.readline().strip()
    with gzip.open(sample_dir / "tissue_positions_list.csv.gz", "rt") as f:
        df = pd.read_csv(f, header=0 if first.startswith("barcode") else None)
    if first.startswith("barcode"):
        df = df.rename(
            columns={
                "barcode": "barcode",
                "in_tissue": "in_tissue",
                "array_row": "array_row",
                "array_col": "array_col",
                "pxl_row_in_fullres": "pxl_row_in_fullres",
                "pxl_col_in_fullres": "pxl_col_in_fullres",
            }
        )
        return df[["barcode", "in_tissue", "array_row", "array_col", "pxl_row_in_fullres", "pxl_col_in_fullres"]].copy()
    df.columns = ["barcode", "in_tissue", "array_row", "array_col", "pxl_row_in_fullres", "pxl_col_in_fullres"]
    return df


def load_expression(sample: str) -> tuple[sparse.csr_matrix, list[str], list[str]]:
    arr = np.load(EXPRESSION_ROOT / sample / "test.npz", allow_pickle=True)
    shape = tuple(arr["shape"].tolist())
    csr = sparse.csr_matrix((arr["data"], arr["indices"], arr["indptr"]), shape=shape)
    gene_ids = arr["gene_ids"].astype(str).tolist()
    barcodes = arr["barcodes"].astype(str).tolist()
    return csr, gene_ids, barcodes


def build_count_table(sample: str, positions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    csr, gene_ids, barcodes = load_expression(sample)
    barcode_to_idx = {bc: i for i, bc in enumerate(barcodes)}
    positions = positions[positions["barcode"].astype(str).isin(barcode_to_idx)].copy()
    dense = csr.T.toarray().astype(np.float32)
    keep = [barcode_to_idx[b] for b in positions["barcode"].astype(str)]
    sub = dense[keep]
    out = pd.DataFrame(sub, columns=gene_ids)
    out.insert(0, "spot", [f"{int(r)}x{int(c)}" for r, c in zip(positions["array_row"], positions["array_col"])])
    return out, positions, gene_ids


def load_hires_image(src: Path) -> Image.Image:
    with gzip.open(src, "rb") as fi:
        img = Image.open(fi).convert("RGB")
        img.load()
    return img


def ensure_slide_images(src: Path, jpg_path: Path, tif_path: Path) -> None:
    jpg_path.parent.mkdir(parents=True, exist_ok=True)
    img = load_hires_image(src)
    if not jpg_path.exists():
        img.save(jpg_path, quality=95)
    tmp_tif = tif_path.with_name(f"{tif_path.stem}.tmp.tif")
    img.save(tmp_tif, format="TIFF", compression=None)
    tmp_tif.replace(tif_path)


def main() -> None:
    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    train_samples = list(cfg["split"]["train"])
    val_samples = list(cfg["split"]["val"])
    test_samples = list(cfg["split"]["test"])
    samples = train_samples + val_samples + test_samples

    RAW_OUT.mkdir(parents=True, exist_ok=True)
    mapping_rows: list[dict[str, str]] = []
    gene_ids_ref: list[str] | None = None

    for idx, sample in enumerate(samples):
        patient = f"P{idx:03d}"
        section = "S1"
        split_name = "train" if sample in train_samples else ("val" if sample in val_samples else "test")
        gsm, slide, sec = parse_sample(sample)
        src = RAW_ROOT / gsm / slide / sec
        dst = RAW_OUT / "hippocampus" / patient
        dst.mkdir(parents=True, exist_ok=True)

        positions = load_positions(src)
        positions = positions[positions["in_tissue"].astype(int) == 1].copy()
        with gzip.open(src / "scalefactors_json.json.gz", "rt") as f:
            sf = json.load(f)
        hires_scale = float(sf.get("tissue_hires_scalef", 1.0))

        jpg_path = dst / f"{patient}_{section}.jpg"
        tif_path = dst / f"{patient}_{section}.tif"
        ensure_slide_images(src / "tissue_hires_image.png.gz", jpg_path, tif_path)

        counts_path = dst / f"{patient}_{section}.tsv.gz"
        expr_positions = None
        if not counts_path.exists():
            count, expr_positions, gene_ids = build_count_table(sample, positions)
            gene_ids_ref = gene_ids if gene_ids_ref is None else gene_ids_ref
            with gzip.open(counts_path, "wt") as f:
                count.to_csv(f, sep="\t", index=False)
        elif gene_ids_ref is None:
            _, gene_ids_ref, _ = load_expression(sample)
        if expr_positions is None:
            _, expr_positions, _ = build_count_table(sample, positions)

        spots_path = dst / f"{patient}_{section}.spots.txt"
        if not spots_path.exists():
            spots = pd.DataFrame(
                {
                    "x": expr_positions["array_row"].astype(int),
                    "y": expr_positions["array_col"].astype(int),
                    "pixel_x": np.round(expr_positions["pxl_col_in_fullres"].astype(float) * hires_scale).astype(int),
                    "pixel_y": np.round(expr_positions["pxl_row_in_fullres"].astype(float) * hires_scale).astype(int),
                }
            )
            spots.to_csv(spots_path, sep="\t", index=False)

        coords_path = dst / f"{patient}_{section}_Coords.tsv"
        if not coords_path.exists():
            coords = pd.DataFrame(
                {
                    "barcode": expr_positions["barcode"].astype(str),
                    "x": expr_positions["array_row"].astype(int),
                    "y": expr_positions["array_col"].astype(int),
                    "region": "all",
                    "label": "tumor",
                }
            )
            coords.to_csv(coords_path, sep="\t", index=False)

        mapping_rows.append(
            {
                "sample": sample,
                "patient": patient,
                "section": section,
                "split": split_name,
            }
        )

    meta_dir = OUT_ROOT / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(mapping_rows).to_csv(meta_dir / "patient_mapping.tsv", sep="\t", index=False)
    (meta_dir / "split.json").write_text(
        json.dumps(
            {
                "train_samples": train_samples,
                "val_samples": val_samples,
                "test_samples": test_samples,
                "train_patients": [r["patient"] for r in mapping_rows if r["split"] == "train"],
                "val_patients": [r["patient"] for r in mapping_rows if r["split"] == "val"],
                "test_patients": [r["patient"] for r in mapping_rows if r["split"] == "test"],
                "n_genes": len(gene_ids_ref or []),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
