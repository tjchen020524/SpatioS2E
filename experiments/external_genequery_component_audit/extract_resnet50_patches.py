#!/usr/bin/env python3
"""Materialize the exact 224-pixel HER2 crops used by the ResNet audit."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from common import HER2, PATCH_ROOT, load_manifest
from extract_resnet50_features import crop_square


def atomic_save(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=PATCH_ROOT)
    parser.add_argument("--sample", action="append", default=[])
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-spots", type=int, default=0, help="Debug-only cap per sample.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    manifest = load_manifest()
    samples = manifest["sample"].astype(str).tolist()
    if args.sample:
        requested = set(args.sample)
        samples = [sample for sample in samples if sample in requested]
        missing = requested - set(samples)
        if missing:
            raise ValueError("Unknown samples: %s" % sorted(missing))
    if args.max_samples > 0:
        samples = samples[: args.max_samples]

    summary = {}
    for position, sample in enumerate(samples, start=1):
        sample_dir = args.output_root / sample
        patch_path = sample_dir / "patches.npy"
        barcode_path = sample_dir / "barcodes.npy"
        if patch_path.exists() and barcode_path.exists() and not args.overwrite:
            patches = np.load(patch_path, mmap_mode="r", allow_pickle=False)
            barcodes = np.load(barcode_path, allow_pickle=False).astype(str)
            if patches.shape != (len(barcodes), 224, 224, 3) or patches.dtype != np.uint8:
                raise ValueError("Invalid existing patch artifact for %s: %s" % (sample, patches.shape))
            summary[sample] = {"n_spots": int(len(barcodes)), "status": "existing"}
            continue

        coords_path = HER2 / "data/coordinates" / (sample + ".npz")
        with np.load(coords_path, allow_pickle=False) as coords:
            barcodes = coords["barcodes"].astype(str)
            points = np.column_stack([coords["pixel_x"], coords["pixel_y"]]).astype(np.float32)
        if args.max_spots > 0:
            barcodes = barcodes[: args.max_spots]
            points = points[: args.max_spots]
        image = Image.open(HER2 / "data/source/images/images/HE" / (sample + ".jpg")).convert("RGB")
        patches = np.empty((len(points), 224, 224, 3), dtype=np.uint8)
        for index, point in enumerate(points):
            patches[index] = crop_square(image, float(point[0]), float(point[1]))
        atomic_save(patch_path, patches)
        atomic_save(barcode_path, barcodes)
        summary[sample] = {
            "n_spots": int(len(barcodes)),
            "shape": list(patches.shape),
            "dtype": str(patches.dtype),
            "status": "written",
        }
        print("[patches] %d/%d %s spots=%d" % (position, len(samples), sample, len(barcodes)), flush=True)

    args.output_root.mkdir(parents=True, exist_ok=True)
    metadata = {
        "source_images": str(HER2 / "data/source/images/images/HE"),
        "crop_pixels": 224,
        "color_space": "RGB",
        "padding": "white",
        "storage": "per-sample uint8 NPY",
        "samples": summary,
    }
    (args.output_root / "patch_manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
