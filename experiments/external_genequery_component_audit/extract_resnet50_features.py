#!/usr/bin/env python3
"""Extract deterministic ImageNet ResNet-50 features from 224-pixel HER2 crops."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import timm
from PIL import Image
from safetensors.torch import load_file

from common import EXP, FEATURE_ROOT, HER2, load_manifest, sha256


DEFAULT_CHECKPOINT = EXP.parents[1] / "weights/resnet50.a1_in1k/model.safetensors"
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def crop_square(image: Image.Image, center_x: float, center_y: float, side: int = 224) -> np.ndarray:
    half = side / 2.0
    left = int(round(center_x - half))
    top = int(round(center_y - half))
    canvas = Image.new("RGB", (side, side), color=(255, 255, 255))
    src_left = max(left, 0)
    src_top = max(top, 0)
    src_right = min(left + side, image.width)
    src_bottom = min(top + side, image.height)
    if src_right > src_left and src_bottom > src_top:
        region = image.crop((src_left, src_top, src_right, src_bottom))
        canvas.paste(region, (src_left - left, src_top - top))
    return np.asarray(canvas)


def preprocess(patch: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(np.array(patch, copy=True)).permute(2, 0, 1).float().div_(255.0)
    return (tensor - IMAGENET_MEAN) / IMAGENET_STD


def load_backbone(checkpoint: Path, device: torch.device) -> torch.nn.Module:
    if not checkpoint.exists():
        raise FileNotFoundError("ResNet-50 checkpoint not found: %s" % checkpoint)
    model = timm.create_model("resnet50", pretrained=False, num_classes=0, global_pool="avg")
    state = load_file(str(checkpoint), device="cpu")
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys or set(incompatible.unexpected_keys) != {"fc.bias", "fc.weight"}:
        raise RuntimeError("Unexpected timm ResNet-50 checkpoint keys: %s" % (incompatible,))
    model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-root", type=Path, default=FEATURE_ROOT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--sample", action="append", default=[])
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-spots", type=int, default=0, help="Debug-only cap per sample.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    model = load_backbone(args.checkpoint, device)
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
        output = args.output_root / sample / "features.npz"
        if output.exists() and not args.overwrite:
            with np.load(output, allow_pickle=False) as existing:
                summary[sample] = {"n_spots": int(len(existing["barcodes"])), "status": "existing"}
            continue
        coords_path = HER2 / "data/coordinates" / (sample + ".npz")
        with np.load(coords_path, allow_pickle=False) as coords:
            barcodes = coords["barcodes"].astype(str)
            points = np.column_stack([coords["pixel_x"], coords["pixel_y"]]).astype(np.float32)
        if args.max_spots > 0:
            barcodes = barcodes[: args.max_spots]
            points = points[: args.max_spots]
        image = Image.open(HER2 / "data/source/images/images/HE" / (sample + ".jpg")).convert("RGB")
        feature_blocks = []
        batch = []
        with torch.inference_mode():
            for point in points:
                batch.append(preprocess(crop_square(image, float(point[0]), float(point[1]))))
                if len(batch) == args.batch_size:
                    feature_blocks.append(model(torch.stack(batch).to(device)).cpu().float().numpy())
                    batch.clear()
            if batch:
                feature_blocks.append(model(torch.stack(batch).to(device)).cpu().float().numpy())
        features = np.concatenate(feature_blocks).astype(np.float32, copy=False)
        if features.shape != (len(barcodes), 2048):
            raise RuntimeError("%s: unexpected feature shape %s" % (sample, features.shape))
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output, features=features, barcodes=barcodes)
        summary[sample] = {
            "n_spots": int(len(barcodes)),
            "feature_dim": 2048,
            "mean_feature_norm": float(np.linalg.norm(features, axis=1).mean()),
            "status": "written",
        }
        print("[ResNet-50] %d/%d %s spots=%d" % (position, len(samples), sample, len(barcodes)), flush=True)

    args.output_root.mkdir(parents=True, exist_ok=True)
    metadata = {
        "extractor": "timm.create_model('resnet50', num_classes=0, global_pool='avg')",
        "resolved_pretraining": "timm/resnet50.a1_in1k",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "crop_pixels": 224,
        "padding": "white",
        "preprocessing": "ImageNet mean/std",
        "backbone_trainable": False,
        "samples": summary,
    }
    (args.output_root / "feature_manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
