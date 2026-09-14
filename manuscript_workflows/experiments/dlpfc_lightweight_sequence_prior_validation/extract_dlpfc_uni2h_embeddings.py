#!/usr/bin/env python3
from __future__ import annotations
from workflow_paths import DATA_ROOT

import argparse
import csv
import gzip
import json
import logging
import os
import sys
from pathlib import Path

import torch
import numpy as np
from PIL import Image


ROOT = DATA_ROOT
for path in (ROOT, ROOT / "scripts", ROOT / "UNI"):
    if str(path) not in sys.path:
        sys.path.append(str(path))

import extract_histology_embeddings as base  # noqa: E402


def patch_pytree() -> None:
    pytree = torch.utils._pytree
    if hasattr(pytree, "register_pytree_node"):
        return
    if not hasattr(pytree, "_register_pytree_node"):
        return
    old_register = pytree._register_pytree_node

    def compat_register_pytree_node(cls, flatten_fn, unflatten_fn, *args, **kwargs):
        kwargs.pop("serialized_type_name", None)
        kwargs.pop("serialized_fields", None)
        return old_register(cls, flatten_fn, unflatten_fn)

    pytree.register_pytree_node = compat_register_pytree_node  # type: ignore[attr-defined]


def load_positions_robust(path: Path, include_background: bool):
    rows = []
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="") as handle:
        reader = csv.reader(handle)
        for record in reader:
            if not record or record[0] == "barcode":
                continue
            barcode, in_tissue, array_row, array_col, pxl_row, pxl_col = record
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


def preprocess_patch_contiguous(patch, patch_size: int, mean, std) -> torch.Tensor:
    pil_image = Image.fromarray(patch)
    if pil_image.width != patch_size or pil_image.height != patch_size:
        pil_image = pil_image.resize((patch_size, patch_size), Image.BILINEAR)
    arr = np.asarray(pil_image, dtype=np.float32) / 255.0
    arr = (arr - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)
    arr = np.ascontiguousarray(arr.transpose(2, 0, 1), dtype=np.float32)
    # torch.from_numpy can fail in this environment when torch and numpy were
    # built against different ndarray ABIs; torch.tensor copies safely.
    return torch.tensor(arr, dtype=torch.float32)


def load_uni2h_backbone(device: str, ckpt_path: Path) -> tuple[torch.nn.Module, int, tuple[float, ...], tuple[float, ...]]:
    patch_pytree()
    import timm

    timm_kwargs = {
        "model_name": "vit_giant_patch14_224",
        "img_size": 224,
        "patch_size": 14,
        "depth": 24,
        "num_heads": 24,
        "init_values": 1e-5,
        "embed_dim": 1536,
        "mlp_ratio": 2.66667 * 2,
        "num_classes": 0,
        "no_embed_class": True,
        "mlp_layer": timm.layers.SwiGLUPacked,
        "act_layer": torch.nn.SiLU,
        "reg_tokens": 8,
        "dynamic_img_size": True,
    }
    model = timm.create_model(pretrained=False, **timm_kwargs)
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval().to(device)
    for param in model.parameters():
        param.requires_grad_(False)
    return model, int(model.num_features), (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract UNI2-h spot embeddings for the prepared dlPFC Visium cohort.")
    p.add_argument("--raw-dir", type=Path, default=ROOT / "experiments/dlpfc_external_generalization/data/raw_dlpfc")
    p.add_argument("--output-dir", type=Path, default=ROOT / "experiments/dlpfc_lightweight_sequence_prior_validation/data/histology_embeddings_uni2h")
    p.add_argument("--ckpt-path", type=Path, default=ROOT / "UNI/UNI2-h/pytorch_model.bin")
    p.add_argument("--image-resolution", choices=("auto", "fullres", "hires", "lowres", "detected"), default="auto")
    p.add_argument("--patch-scale", type=float, default=1.0)
    p.add_argument("--patch-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--include-background", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s")
    base.LOGGER.setLevel(logging.DEBUG if args.verbose else logging.INFO)
    base.load_positions = load_positions_robust
    base.preprocess_patch = preprocess_patch_contiguous

    if not args.raw_dir.exists():
        raise FileNotFoundError(f"Raw directory not found: {args.raw_dir}")
    if not args.ckpt_path.exists():
        raise FileNotFoundError(f"UNI2-h checkpoint not found: {args.ckpt_path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample_positions = base.discover_samples(args.raw_dir)
    if not sample_positions:
        raise FileNotFoundError(f"No tissue position files found under {args.raw_dir}")

    backbone, feature_dim, mean, std = load_uni2h_backbone(args.device, args.ckpt_path)
    args.model = "uni2h"

    aggregate = {}
    for sample, positions_path in sample_positions.items():
        scalefactors_path = args.raw_dir / f"{sample}-scalefactors_json.json.gz"
        if not scalefactors_path.exists():
            scalefactors_path = args.raw_dir / f"{sample}_scalefactors_json.json.gz"
        if not scalefactors_path.exists():
            raise FileNotFoundError(f"Missing scalefactors for sample {sample}")
        scalefactors = base.load_scalefactors(scalefactors_path)
        image_path, resolution, scale_factor = base.resolve_image_path(
            sample,
            scalefactors,
            args.raw_dir,
            requested=args.image_resolution,
        )
        cfg = base.SampleConfig(
            sample=sample,
            positions_path=positions_path,
            scalefactors_path=scalefactors_path,
            image_path=image_path,
            image_resolution=resolution,
            scale_factor=scale_factor,
        )
        aggregate[sample] = base.process_sample(cfg, backbone, feature_dim, mean, std, args)

    summary = {
        "model": "UNI2-h",
        "checkpoint": str(args.ckpt_path),
        "n_samples": len(aggregate),
        "samples": aggregate,
    }
    (args.output_dir / "histology_embeddings_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"output_dir": str(args.output_dir), "n_samples": len(aggregate)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
