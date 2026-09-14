#!/usr/bin/env python3
"""Dataset for training Hist2ST on Visium spot-level expression."""

from __future__ import annotations

import csv
import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy import sparse

try:
    from PIL import Image
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("Pillow is required for Hist2ST dataset.") from exc

from Hist2ST.graph_construction import calcADJ


def open_image(path: Path) -> Image.Image:
    if path.suffix == ".gz":
        with gzip.open(path, "rb") as handle:
            with Image.open(handle) as img:
                return img.convert("RGB")
    with Image.open(path) as img:
        return img.convert("RGB")


def load_scalefactors(path: Path) -> Dict[str, float]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        return json.load(handle)


def resolve_image_path(raw_dir: Path) -> Tuple[Path, float]:
    scalefactors_path = raw_dir / "scalefactors_json.json.gz"
    if not scalefactors_path.exists():
        scalefactors_path = raw_dir / "scalefactors_json.json"
    scalefactors = load_scalefactors(scalefactors_path)
    hires_scale = float(scalefactors.get("tissue_hires_scalef", 1.0))
    lowres_scale = float(scalefactors.get("tissue_lowres_scalef", hires_scale))

    candidates = [
        ("tissue_hires_image.png.gz", hires_scale),
        ("tissue_hires_image.png", hires_scale),
        ("detected_tissue_image.jpg.gz", hires_scale),
        ("detected_tissue_image.jpg", hires_scale),
        ("tissue_lowres_image.png.gz", lowres_scale),
        ("tissue_lowres_image.png", lowres_scale),
    ]
    for fname, scale in candidates:
        path = raw_dir / fname
        if path.exists():
            return path, scale
    raise FileNotFoundError(f"No histology image found in {raw_dir}")


def load_positions(path: Path) -> List[Dict[str, float]]:
    opener = gzip.open if path.suffix == ".gz" else open
    rows: List[Dict[str, float]] = []
    with opener(path, "rt", newline="") as handle:
        reader = csv.reader(handle)
        for barcode, in_tissue, array_row, array_col, pxl_row, pxl_col in reader:
            if int(in_tissue) != 1:
                continue
            rows.append(
                {
                    "barcode": barcode,
                    "array_row": int(array_row),
                    "array_col": int(array_col),
                    "pxl_row": float(pxl_row),
                    "pxl_col": float(pxl_col),
                }
            )
    return rows


def crop_patch(image_array: np.ndarray, center_x: float, center_y: float, crop_size: int) -> np.ndarray:
    half = crop_size / 2.0
    left = int(np.floor(center_x - half))
    top = int(np.floor(center_y - half))
    right = left + crop_size
    bottom = top + crop_size

    pad_left = max(0, -left)
    pad_top = max(0, -top)
    pad_right = max(0, right - image_array.shape[1])
    pad_bottom = max(0, bottom - image_array.shape[0])

    if pad_left or pad_top or pad_right or pad_bottom:
        image_array = np.pad(
            image_array,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode="constant",
        )
        left += pad_left
        right += pad_left
        top += pad_top
        bottom += pad_top

    patch = image_array[top:bottom, left:right]
    if patch.shape[0] != crop_size or patch.shape[1] != crop_size:
        # fallback resize
        patch = np.asarray(Image.fromarray(patch).resize((crop_size, crop_size), Image.BILINEAR))
    return patch


def load_expression(npz_path: Path) -> Tuple[sparse.csr_matrix, List[str], List[str]]:
    arr = np.load(npz_path)
    shape = tuple(arr["shape"].tolist())
    csr = sparse.csr_matrix((arr["data"], arr["indices"], arr["indptr"]), shape=shape)
    gene_ids = arr["gene_ids"].astype(str).tolist()
    barcodes = arr["barcodes"].astype(str).tolist()
    return csr, gene_ids, barcodes


def parse_sample_name(sample: str) -> Tuple[str, str, str]:
    parts = sample.split("_")
    if len(parts) < 3:
        raise ValueError(f"Unexpected sample format: {sample}")
    gsm = parts[0]
    section = parts[-1]
    slide = "_".join(parts[1:-1])
    return gsm, slide, section


def normalize_positions(coords: np.ndarray, n_pos: int) -> np.ndarray:
    coords = coords.astype(np.float32)
    coords = coords - coords.min(axis=0, keepdims=True)
    maxv = coords.max(axis=0, keepdims=True)
    maxv[maxv == 0] = 1.0
    coords = coords / maxv
    coords = np.clip(np.rint(coords * (n_pos - 1)), 0, n_pos - 1).astype(np.int64)
    return coords


@dataclass
class Hist2STSample:
    patches: torch.Tensor
    centers: torch.Tensor
    expr: torch.Tensor
    adj: torch.Tensor
    barcodes: List[str]
    sample: str


class Hist2STSpotDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        samples: Sequence[str],
        expression_root: Path,
        raw_root: Path,
        split: str,
        gene_list: Sequence[str],
        patch_size: int = 112,
        n_pos: int = 64,
        neighbor_k: int = 6,
        prune: str = "Grid",
        max_spots: int | None = None,
    ) -> None:
        self.samples = list(samples)
        self.expression_root = expression_root
        self.raw_root = raw_root
        self.split = split
        self.gene_list = list(gene_list)
        self.patch_size = patch_size
        self.n_pos = n_pos
        self.neighbor_k = neighbor_k
        self.prune = prune
        self.max_spots = max_spots

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Hist2STSample:
        sample = self.samples[idx]
        gsm, slide, section = parse_sample_name(sample)
        raw_dir = self.raw_root / gsm / slide / section

        pos_path = raw_dir / "tissue_positions_list.csv.gz"
        if not pos_path.exists():
            pos_path = raw_dir / "tissue_positions_list.csv"
        positions = load_positions(pos_path)
        if not positions:
            raise ValueError(f"No in-tissue spots for {sample}")

        img_path, scale = resolve_image_path(raw_dir)
        image = np.asarray(open_image(img_path))

        expr_path = self.expression_root / sample / f"{self.split}.npz"
        expr_csr, gene_ids, barcodes = load_expression(expr_path)
        gene_idx = {g: i for i, g in enumerate(gene_ids)}
        keep = [gene_idx[g] for g in self.gene_list if g in gene_idx]
        if not keep:
            raise ValueError(f"No genes from list found in {sample}")

        # align barcodes
        expr_barcode_index = {bc: i for i, bc in enumerate(barcodes)}
        rows = []
        centers = []
        arrays = []
        kept_barcodes = []
        for row in positions:
            bc = row["barcode"]
            if bc not in expr_barcode_index:
                continue
            rows.append(expr_barcode_index[bc])
            centers.append([row["pxl_row"] * scale, row["pxl_col"] * scale])
            arrays.append([row["array_row"], row["array_col"]])
            kept_barcodes.append(bc)

        if not rows:
            raise ValueError(f"No overlapping barcodes for {sample}")

        if self.max_spots is not None and len(rows) > self.max_spots:
            rng = np.random.default_rng(abs(hash(sample)) % (2**32))
            sel = rng.choice(len(rows), size=self.max_spots, replace=False)
            sel = np.sort(sel)
            rows = [rows[i] for i in sel]
            centers = [centers[i] for i in sel]
            arrays = [arrays[i] for i in sel]
            kept_barcodes = [kept_barcodes[i] for i in sel]

        expr = expr_csr[keep, :][:, rows].toarray().T.astype(np.float32)  # (N, G)

        # build patches
        patches = []
        for center_y, center_x in centers:
            patch = crop_patch(image, center_x, center_y, self.patch_size)
            patches.append(patch)
        patches = np.stack(patches, axis=0)
        patches = patches.astype(np.float32) / 255.0
        patches = torch.from_numpy(patches).permute(0, 3, 1, 2)  # (N, 3, H, W)

        centers_arr = np.asarray(arrays, dtype=np.float32)
        centers_norm = normalize_positions(centers_arr, self.n_pos)
        centers_t = torch.from_numpy(centers_norm).long()

        adj = calcADJ(centers_arr, k=self.neighbor_k, pruneTag=self.prune)
        adj = torch.as_tensor(adj, dtype=torch.float32)
        row_sum = adj.sum(1)
        if torch.any(row_sum == 0):
            idx = torch.nonzero(row_sum == 0, as_tuple=False).squeeze(1)
            adj = adj.clone()
            adj[idx, idx] = 1.0

        return Hist2STSample(
            patches=patches,
            centers=centers_t,
            expr=torch.from_numpy(expr),
            adj=adj,
            barcodes=kept_barcodes,
            sample=sample,
        )
