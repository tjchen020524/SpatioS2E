#!/usr/bin/env python3
"""Extract spot-centered UNI2-h embeddings from custom HER2ST coordinates."""

from __future__ import annotations
from workflow_paths import DATA_ROOT

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


ROOT = DATA_ROOT
for path in (ROOT, ROOT / "scripts", ROOT / "UNI"):
    if str(path) not in sys.path:
        sys.path.append(str(path))

from experiments.dlpfc_lightweight_sequence_prior_validation.extract_dlpfc_uni2h_embeddings import (  # noqa: E402
    load_uni2h_backbone,
    preprocess_patch_contiguous,
)


EXP = ROOT / "experiments/her2st_current_full_ablation"


def median_nearest_spacing(points: np.ndarray) -> float:
    """Return median nearest-neighbor spacing without a SciPy dependency."""
    tensor = torch.from_numpy(points.astype(np.float32, copy=False))
    distances = torch.cdist(tensor, tensor)
    distances.fill_diagonal_(float("inf"))
    nearest = distances.min(dim=1).values
    return float(nearest[torch.isfinite(nearest)].median().item())


def crop_square(image: Image.Image, center_x: float, center_y: float, side: int) -> np.ndarray:
    half = side / 2.0
    left = int(round(center_x - half))
    top = int(round(center_y - half))
    right = left + side
    bottom = top + side
    canvas = Image.new("RGB", (side, side), color=(255, 255, 255))
    src_left = max(left, 0)
    src_top = max(top, 0)
    src_right = min(right, image.width)
    src_bottom = min(bottom, image.height)
    if src_right > src_left and src_bottom > src_top:
        region = image.crop((src_left, src_top, src_right, src_bottom))
        canvas.paste(region, (src_left - left, src_top - top))
    return np.asarray(canvas)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "UNI/UNI2-h/pytorch_model.bin")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patch-scale", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    split = json.loads((EXP / "data/split.json").read_text())
    samples = split["train"] + split["val"] + split["test"]
    model, feature_dim, mean, std = load_uni2h_backbone(args.device, args.checkpoint)
    summaries = {}
    for sample_idx, sample in enumerate(samples):
        output = EXP / "data/uni2h_embeddings" / sample / "embeddings.npz"
        if output.exists() and not args.overwrite:
            data = np.load(output)
            summaries[sample] = {"n_spots": int(len(data["barcodes"])), "status": "existing"}
            continue
        coords = np.load(EXP / "data/coordinates" / f"{sample}.npz")
        barcodes = coords["barcodes"].astype(str)
        points = np.column_stack([coords["pixel_x"], coords["pixel_y"]]).astype(np.float32)
        center_spacing_pixels = median_nearest_spacing(points)
        spot_diameter_pixels = max(32, int(round(center_spacing_pixels * 0.5 * args.patch_scale)))
        image = Image.open(EXP / "data/source/images/images/HE" / f"{sample}.jpg").convert("RGB")

        embeddings = []
        norms = []
        batch = []
        with torch.inference_mode():
            for point in points:
                patch = crop_square(image, float(point[0]), float(point[1]), spot_diameter_pixels)
                batch.append(preprocess_patch_contiguous(patch, 224, mean, std))
                if len(batch) == args.batch_size:
                    features = model(torch.stack(batch).to(args.device)).detach().cpu().float().numpy()
                    embeddings.append(features)
                    norms.extend(np.linalg.norm(features, axis=1).tolist())
                    batch.clear()
            if batch:
                features = model(torch.stack(batch).to(args.device)).detach().cpu().float().numpy()
                embeddings.append(features)
                norms.extend(np.linalg.norm(features, axis=1).tolist())
        matrix = np.concatenate(embeddings).astype(np.float32)
        if matrix.shape != (len(barcodes), feature_dim):
            raise RuntimeError(f"{sample}: unexpected embedding shape {matrix.shape}")
        output.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "sample": sample,
            "model": "UNI2-h",
            "checkpoint": str(args.checkpoint),
            "feature_dim": feature_dim,
            "center_spacing_pixels": center_spacing_pixels,
            "spot_diameter_pixels": spot_diameter_pixels,
            "spot_diameter_micrometers": 100,
            "patch_scale": args.patch_scale,
        }
        np.savez_compressed(
            output,
            embeddings=matrix,
            barcodes=barcodes,
            metadata=np.asarray([json.dumps(metadata)], dtype=object),
        )
        summaries[sample] = {
            **metadata,
            "n_spots": len(barcodes),
            "embedding_norm_mean": float(np.mean(norms)),
        }
        print(f"[UNI2-h] {sample_idx + 1}/{len(samples)} {sample} spots={len(barcodes)}", flush=True)

    summary_path = EXP / "data/uni2h_embeddings/histology_embeddings_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
