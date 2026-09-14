#!/usr/bin/env python
"""
Extract per-spot histology embeddings from Visium H&E tissue images.

This script loads the 10x Visium histology images alongside the
``tissue_positions_list.csv`` coordinates and ``scalefactors_json.json`` files,
crops a patch for each spot, and feeds it through a pretrained CNN backbone
(default: torchvision ResNet-50 with ImageNet weights). The resulting feature
vectors are written to ``.npz`` files that contain the embeddings, barcodes,
and metadata needed for downstream modelling.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

try:
    from PIL import Image
except ImportError as exc:  # pragma: no cover - defensive guard
    raise RuntimeError("Pillow is required for histology embedding extraction.") from exc

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - fallback if tqdm missing
    tqdm = lambda it, **kw: it  # type: ignore

LOGGER = logging.getLogger("spatios2e.extract_histology_embeddings")


@dataclass
class SampleConfig:
    """Metadata required to process a single Visium sample."""

    sample: str
    positions_path: Path
    scalefactors_path: Path
    image_path: Path
    image_resolution: str
    scale_factor: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("SpatioS2E/data/raw"),
        help="Directory containing Visium raw data (images, scalefactors, tissue positions).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("SpatioS2E/data/processed/histology_embeddings"),
        help="Directory to store per-sample embedding archives.",
    )
    parser.add_argument(
        "--image-resolution",
        choices=("auto", "fullres", "hires", "lowres", "detected"),
        default="auto",
        help="Which image to use for patch extraction. 'auto' tries hires→detected→lowres.",
    )
    parser.add_argument(
        "--model",
        choices=("resnet50", "ctranspath"),
        default="resnet50",
        help="Backbone used to extract embeddings.",
    )
    parser.add_argument(
        "--ctranspath-ckpt",
        type=Path,
        default=None,
        help="Checkpoint path for CTransPath (required when --model ctranspath).",
    )
    parser.add_argument(
        "--ctranspath-arch",
        choices=("vit_b_16",),
        default="vit_b_16",
        help="Backbone architecture used for CTransPath checkpoint loading.",
    )
    parser.add_argument(
        "--patch-scale",
        type=float,
        default=1.0,
        help="Factor multiplied with spot_diameter_fullres (after scaling) for crop size.",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=224,
        help="Spatial size (pixels) of resized patches fed into the backbone.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Number of patches processed per forward pass.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run inference on (e.g. 'cuda', 'cuda:0', 'cpu').",
    )
    parser.add_argument(
        "--include-background",
        action="store_true",
        help="Include spots marked as out-of-tissue (in_tissue == 0).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing embedding archives.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Reserved for API compatibility; patches are processed in-Python.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    return parser.parse_args()


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(message)s")


def discover_samples(raw_dir: Path) -> Dict[str, Path]:
    """Find available tissue position files under the raw directory."""
    mapping: Dict[str, Path] = {}
    for path in raw_dir.glob("*tissue_positions_list.csv.gz"):
        sample = path.name.replace("-tissue_positions_list.csv.gz", "")
        sample = sample.replace("_tissue_positions_list.csv.gz", "")
        mapping[sample] = path
    return dict(sorted(mapping.items()))


def load_scalefactors(path: Path) -> Dict[str, float]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        data = json.load(handle)
    return data


def resolve_image_path(
    sample: str,
    scalefactors: Dict[str, float],
    raw_dir: Path,
    requested: str = "auto",
) -> Tuple[Path, str, float]:
    """Locate the best available histology image and return with scale factor."""
    candidates: List[Tuple[str, str, float]] = []
    hires_scale = float(scalefactors.get("tissue_hires_scalef", 1.0))
    lowres_scale = float(scalefactors.get("tissue_lowres_scalef", hires_scale))
    resolution_order = {
        "fullres": [("tissue_hires_image.png", "hires", hires_scale)],
        "hires": [("tissue_hires_image.png", "hires", hires_scale)],
        "detected": [("detected_tissue_image.jpg", "detected", hires_scale)],
        "lowres": [("tissue_lowres_image.png", "lowres", lowres_scale)],
        "auto": [
            ("tissue_hires_image.png", "hires", hires_scale),
            ("detected_tissue_image.jpg", "detected", hires_scale),
            ("tissue_lowres_image.png", "lowres", lowres_scale),
        ],
    }
    for filename, label, scale in resolution_order[requested]:
        for suffix in ("", ".gz"):
            path = raw_dir / f"{sample}-{filename}{suffix}"
            if path.exists():
                return path, label, scale
    raise FileNotFoundError(
        f"No histology image found for sample {sample} with resolution preference '{requested}'."
    )


def open_image(path: Path) -> Image.Image:
    """Open PNG/JPG images (optionally gzipped) as RGB PIL Image."""
    if path.suffix == ".gz":
        with gzip.open(path, "rb") as handle:
            with Image.open(handle) as img:
                return img.convert("RGB")
    with Image.open(path) as img:
        return img.convert("RGB")


def load_positions(path: Path, include_background: bool) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="") as handle:
        reader = csv.reader(handle)
        for barcode, in_tissue, array_row, array_col, pxl_row, pxl_col in reader:
            row = {
                "barcode": barcode,
                "in_tissue": int(in_tissue),
                "array_row": int(array_row),
                "array_col": int(array_col),
                "pxl_row": float(pxl_row),
                "pxl_col": float(pxl_col),
            }
            if include_background or row["in_tissue"] == 1:
                rows.append(row)
    return rows


def crop_patch(
    image_array: np.ndarray,
    center_x: float,
    center_y: float,
    crop_size: int,
) -> np.ndarray:
    """Crop a padded square patch from an image."""
    half = crop_size / 2.0
    left = int(math.floor(center_x - half))
    top = int(math.floor(center_y - half))
    right = left + crop_size
    bottom = top + crop_size

    pad_left = max(0, -left)
    pad_top = max(0, -top)
    pad_right = max(0, right - image_array.shape[1])
    pad_bottom = max(0, bottom - image_array.shape[0])

    if any(v > 0 for v in (pad_left, pad_top, pad_right, pad_bottom)):
        image_array = np.pad(
            image_array,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode="reflect",
        )
        left += pad_left
        right += pad_left
        top += pad_top
        bottom += pad_top

    patch = image_array[top:bottom, left:right, :]
    return patch


def preprocess_patch(
    patch: np.ndarray,
    patch_size: int,
    mean: Iterable[float],
    std: Iterable[float],
) -> torch.Tensor:
    """Resize and normalise a patch for CNN input."""
    pil_image = Image.fromarray(patch)
    if pil_image.width != patch_size or pil_image.height != patch_size:
        pil_image = pil_image.resize((patch_size, patch_size), Image.BILINEAR)
    arr = np.asarray(pil_image, dtype=np.float32) / 255.0
    arr = (arr - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)
    arr = arr.transpose(2, 0, 1)
    return torch.from_numpy(arr)


def _strip_state_prefix(state: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    if not prefix:
        return state
    out = {}
    for k, v in state.items():
        if k.startswith(prefix):
            out[k[len(prefix) :]] = v
        else:
            out[k] = v
    return out


def _extract_state_dict(checkpoint: object) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
        return checkpoint
    raise ValueError("Unsupported checkpoint format")


def load_backbone(
    model_name: str,
    device: str,
    ctranspath_ckpt: Optional[Path] = None,
    ctranspath_arch: str = "vit_b_16",
) -> Tuple[torch.nn.Module, int, Tuple[float, ...], Tuple[float, ...]]:
    """Instantiate the requested embedding backbone."""
    if model_name == "resnet50":
        try:
            from torchvision import models
        except ImportError as exc:  # pragma: no cover - provide actionable error
            raise RuntimeError(
                "torchvision is required for the ResNet-50 histology embeddings. "
                "Install torchvision or choose a different backbone."
            ) from exc

        weights = models.ResNet50_Weights.DEFAULT
        backbone = models.resnet50(weights=weights)
        feature_dim = backbone.fc.in_features
        backbone.fc = torch.nn.Identity()
        backbone.eval()
        backbone.to(device)
        for param in backbone.parameters():
            param.requires_grad_(False)
        mean = tuple(float(x) for x in weights.transforms().mean)
        std = tuple(float(x) for x in weights.transforms().std)
        return backbone, feature_dim, mean, std

    if model_name == "ctranspath":
        if ctranspath_ckpt is None:
            raise ValueError("--ctranspath-ckpt is required when --model ctranspath")
        if ctranspath_arch != "vit_b_16":
            raise ValueError(f"Unsupported ctranspath-arch {ctranspath_arch}")
        try:
            from torchvision.models.vision_transformer import vit_b_16
        except ImportError as exc:  # pragma: no cover - actionable error
            raise RuntimeError(
                "torchvision is required for CTransPath embedding extraction. "
                "Install torchvision or choose a different backbone."
            ) from exc

        backbone = vit_b_16(weights=None)
        feature_dim = backbone.hidden_dim
        backbone.heads = torch.nn.Identity()

        ckpt = torch.load(ctranspath_ckpt, map_location="cpu")
        state = _extract_state_dict(ckpt)
        if any(k.startswith("module.") for k in state):
            state = _strip_state_prefix(state, "module.")
        missing, unexpected = backbone.load_state_dict(state, strict=False)
        if missing:
            LOGGER.warning("CTransPath load missing keys: %s", len(missing))
        if unexpected:
            LOGGER.warning("CTransPath load unexpected keys: %s", len(unexpected))

        backbone.eval()
        backbone.to(device)
        for param in backbone.parameters():
            param.requires_grad_(False)
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        return backbone, feature_dim, mean, std

    raise ValueError(f"Unsupported model '{model_name}'.")


def process_sample(
    cfg: SampleConfig,
    backbone: torch.nn.Module,
    feature_dim: int,
    mean: Tuple[float, ...],
    std: Tuple[float, ...],
    args: argparse.Namespace,
) -> Dict[str, object]:
    """Extract embeddings for one sample and persist outputs."""
    sample_dir = args.output_dir / cfg.sample
    sample_dir.mkdir(parents=True, exist_ok=True)
    archive_path = sample_dir / "embeddings.npz"
    qc_path = sample_dir / "qc_summary.json"

    if archive_path.exists() and not args.overwrite:
        LOGGER.info("Skipping %s (embeddings already exist).", cfg.sample)
        return {
            "sample": cfg.sample,
            "skipped": True,
            "embedding_path": str(archive_path),
        }

    LOGGER.info("Processing sample %s using image %s", cfg.sample, cfg.image_path.name)
    image = np.asarray(open_image(cfg.image_path))
    rows = load_positions(cfg.positions_path, include_background=args.include_background)

    if not rows:
        raise RuntimeError(f"No spots selected for sample {cfg.sample} (check --include-background).")

    spot_diameter_fullres = float(
        load_scalefactors(cfg.scalefactors_path).get("spot_diameter_fullres", 100.0)
    )
    crop_size = int(round(spot_diameter_fullres * cfg.scale_factor * args.patch_scale))
    crop_size = max(crop_size, 2)

    device = next(backbone.parameters()).device
    batches: List[torch.Tensor] = []
    embeddings: List[np.ndarray] = []
    barcodes: List[str] = []
    norms: List[float] = []

    iterator = tqdm(rows, desc=f"{cfg.sample}", disable=LOGGER.level > logging.INFO)
    for row in iterator:
        center_x = row["pxl_col"] * cfg.scale_factor
        center_y = row["pxl_row"] * cfg.scale_factor
        patch = crop_patch(image, center_x, center_y, crop_size)
        tensor = preprocess_patch(patch, args.patch_size, mean, std)
        batches.append(tensor)
        barcodes.append(row["barcode"])

        if len(batches) == args.batch_size:
            batch_tensor = torch.stack(batches, dim=0).to(device)
            with torch.no_grad():
                feats = backbone(batch_tensor).cpu().numpy()
            embeddings.append(feats)
            norms.extend(np.linalg.norm(feats, axis=1).tolist())
            batches.clear()

    if batches:
        batch_tensor = torch.stack(batches, dim=0).to(device)
        with torch.no_grad():
            feats = backbone(batch_tensor).cpu().numpy()
        embeddings.append(feats)
        norms.extend(np.linalg.norm(feats, axis=1).tolist())
        batches.clear()

    all_embeddings = np.concatenate(embeddings, axis=0)
    if all_embeddings.shape[0] != len(barcodes):
        raise RuntimeError(
            f"Embedding count mismatch for {cfg.sample}: {all_embeddings.shape[0]} vs {len(barcodes)}."
        )

    np.savez_compressed(
        archive_path,
        embeddings=all_embeddings.astype(np.float32),
        barcodes=np.array(barcodes),
        metadata=np.array(
            [
                json.dumps(
                    {
                        "sample": cfg.sample,
                        "image_path": str(cfg.image_path),
                        "image_resolution": cfg.image_resolution,
                        "scale_factor": cfg.scale_factor,
                        "patch_scale": args.patch_scale,
                        "patch_size": args.patch_size,
                        "feature_dim": feature_dim,
                        "model": args.model,
                    }
                )
            ],
            dtype=object,
        ),
    )

    qc_summary = {
        "sample": cfg.sample,
        "n_embeddings": int(all_embeddings.shape[0]),
        "embedding_dim": int(all_embeddings.shape[1]),
        "n_spots_selected": len(rows),
        "include_background": args.include_background,
        "l2_mean": float(np.mean(norms)),
        "l2_std": float(np.std(norms)),
        "image_path": str(cfg.image_path),
        "image_resolution": cfg.image_resolution,
        "scale_factor": cfg.scale_factor,
        "crop_size_pixels": crop_size,
        "patch_size": args.patch_size,
        "patch_scale": args.patch_scale,
        "model": args.model,
        "device": device.type if isinstance(device, torch.device) else str(device),
    }
    with qc_path.open("w") as handle:
        json.dump(qc_summary, handle, indent=2)

    LOGGER.info(
        "Saved embeddings for %s to %s (spots=%d, dim=%d).",
        cfg.sample,
        archive_path,
        all_embeddings.shape[0],
        all_embeddings.shape[1],
    )

    return {
        "sample": cfg.sample,
        "embedding_path": str(archive_path),
        "qc_path": str(qc_path),
        "n_spots": int(all_embeddings.shape[0]),
        "embedding_dim": int(all_embeddings.shape[1]),
        "image_resolution": cfg.image_resolution,
        "scale_factor": cfg.scale_factor,
    }


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)

    if not args.raw_dir.exists():
        raise FileNotFoundError(f"Raw directory not found: {args.raw_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    sample_positions = discover_samples(args.raw_dir)
    if not sample_positions:
        raise FileNotFoundError(f"No tissue position files found under {args.raw_dir}.")

    backbone, feature_dim, mean, std = load_backbone(
        args.model,
        args.device,
        ctranspath_ckpt=args.ctranspath_ckpt,
        ctranspath_arch=args.ctranspath_arch,
    )

    aggregate: Dict[str, Dict[str, object]] = {}
    for sample, positions_path in sample_positions.items():
        scalefactors_path = args.raw_dir / f"{sample}-scalefactors_json.json.gz"
        if not scalefactors_path.exists():
            scalefactors_path = args.raw_dir / f"{sample}_scalefactors_json.json.gz"
        if not scalefactors_path.exists():
            raise FileNotFoundError(f"Missing scalefactors for sample {sample}.")

        scalefactors = load_scalefactors(scalefactors_path)
        image_path, resolution, scale_factor = resolve_image_path(
            sample,
            scalefactors,
            args.raw_dir,
            requested=args.image_resolution,
        )
        cfg = SampleConfig(
            sample=sample,
            positions_path=positions_path,
            scalefactors_path=scalefactors_path,
            image_path=image_path,
            image_resolution=resolution,
            scale_factor=scale_factor,
        )

        stats = process_sample(cfg, backbone, feature_dim, mean, std, args)
        aggregate[cfg.sample] = stats

    summary_path = args.output_dir / "histology_embeddings_summary.json"
    with summary_path.open("w") as handle:
        json.dump(aggregate, handle, indent=2)

    LOGGER.info("Wrote summary to %s", summary_path)


if __name__ == "__main__":
    main()
