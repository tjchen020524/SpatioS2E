#!/usr/bin/env python3
"""Extract per-spot UNI2-h histology embeddings without touching the base extractor."""

from __future__ import annotations
from workflow_paths import DATA_ROOT

import argparse
import json
import sys
import csv
import gzip
from pathlib import Path
from typing import Dict, Tuple

import torch

try:
    import torch.utils._pytree as _torch_pytree

    if (
        hasattr(_torch_pytree, "_register_pytree_node")
        and not hasattr(_torch_pytree, "register_pytree_node")
    ):
        _orig_register = _torch_pytree._register_pytree_node  # type: ignore[attr-defined]

        def _compat_register_pytree_node(*args, **kwargs):
            kwargs.pop("serialized_type_name", None)
            kwargs.pop("serialized_fields", None)
            return _orig_register(*args, **kwargs)

        _torch_pytree.register_pytree_node = _compat_register_pytree_node  # type: ignore[attr-defined]
except Exception:
    pass

import timm

ROOT = DATA_ROOT
for path in (ROOT / "scripts", ROOT / "UNI"):
    if str(path) not in sys.path:
        sys.path.append(str(path))

from extract_histology_embeddings import (  # type: ignore
    SampleConfig,
    configure_logging,
    load_scalefactors,
    process_sample,
)
import extract_histology_embeddings as base_extractor  # type: ignore
from uni.get_encoder.get_encoder import get_norm_constants


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "processed" / "histology_embeddings_uni2h",
    )
    parser.add_argument("--uni-root", type=Path, default=ROOT / "UNI")
    parser.add_argument(
        "--image-resolution",
        choices=("auto", "fullres", "hires", "lowres", "detected"),
        default="auto",
    )
    parser.add_argument("--patch-scale", type=float, default=1.0)
    parser.add_argument("--patch-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--include-background", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--samples", nargs="*", default=None)
    return parser.parse_args()


def load_uni2h_backbone(uni_root: Path, device: torch.device, patch_size: int) -> Tuple[torch.nn.Module, int, Tuple[float, ...], Tuple[float, ...]]:
    ckpt_path = uni_root / "UNI2-h" / "pytorch_model.bin"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"UNI2-h checkpoint not found: {ckpt_path}")

    uni_kwargs = {
        "model_name": "vit_giant_patch14_224",
        "img_size": patch_size,
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
    backbone = timm.create_model(**uni_kwargs)
    state_dict = torch.load(ckpt_path, map_location="cpu")
    missing_keys, unexpected_keys = backbone.load_state_dict(state_dict, strict=True)
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            f"UNI2-h checkpoint load mismatch: missing={len(missing_keys)} unexpected={len(unexpected_keys)}"
        )
    backbone.eval()
    backbone.to(device)
    for param in backbone.parameters():
        param.requires_grad_(False)
    with torch.no_grad():
        dummy = torch.zeros((1, 3, patch_size, patch_size), device=device)
        feature_dim = int(backbone(dummy).shape[1])
    mean, std = get_norm_constants("imagenet")
    return backbone, feature_dim, tuple(float(x) for x in mean), tuple(float(x) for x in std)


def discover_nested_samples(raw_dir: Path) -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    for positions_path in raw_dir.glob("GSM*/*/*/tissue_positions_list.csv.gz"):
        rel = positions_path.relative_to(raw_dir)
        if len(rel.parts) != 4:
            continue
        gsm, block, section, _ = rel.parts
        sample = f"{gsm}_{block}_{section}"
        mapping[sample] = positions_path.parent
    for positions_path in raw_dir.glob("GSM*/*/*/tissue_positions_list.csv"):
        rel = positions_path.relative_to(raw_dir)
        if len(rel.parts) != 4:
            continue
        gsm, block, section, _ = rel.parts
        sample = f"{gsm}_{block}_{section}"
        mapping.setdefault(sample, positions_path.parent)
    return dict(sorted(mapping.items()))


def resolve_image_path_from_dir(sample_dir: Path, scalefactors: Dict[str, float], requested: str = "auto") -> Tuple[Path, str, float]:
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
            path = sample_dir / f"{filename}{suffix}"
            if path.exists():
                return path, label, scale
    raise FileNotFoundError(f"No histology image found under {sample_dir} for resolution '{requested}'.")


def load_positions_robust(path: Path, include_background: bool):
    rows = []
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row:
                continue
            if row[0].lower() == "barcode" or (len(row) > 1 and row[1].lower() == "in_tissue"):
                continue
            if len(row) != 6:
                raise ValueError(f"Unexpected tissue_positions row with {len(row)} columns in {path}: {row[:3]}")
            barcode, in_tissue, array_row, array_col, pxl_row, pxl_col = row
            item = {
                "barcode": barcode,
                "in_tissue": int(in_tissue),
                "array_row": int(array_row),
                "array_col": int(array_col),
                "pxl_row": float(pxl_row),
                "pxl_col": float(pxl_col),
            }
            if include_background or item["in_tissue"] == 1:
                rows.append(item)
    return rows


base_extractor.load_positions = load_positions_robust


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)
    args.model = "uni2h"

    if not args.raw_dir.exists():
        raise FileNotFoundError(f"Raw directory not found: {args.raw_dir}")
    ckpt_path = args.uni_root / "UNI2-h" / "pytorch_model.bin"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"UNI2-h checkpoint not found: {ckpt_path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    sample_dirs = discover_nested_samples(args.raw_dir)
    if args.samples:
        keep = set(args.samples)
        sample_dirs = {k: v for k, v in sample_dirs.items() if k in keep}
    if not sample_dirs:
        raise FileNotFoundError("No samples selected for UNI2-h extraction.")

    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    backbone, feature_dim, mean, std = load_uni2h_backbone(args.uni_root, device, args.patch_size)

    aggregate = {}
    for sample, sample_dir in sample_dirs.items():
        positions_path = sample_dir / "tissue_positions_list.csv.gz"
        if not positions_path.exists():
            positions_path = sample_dir / "tissue_positions_list.csv"
        scalefactors_path = sample_dir / "scalefactors_json.json.gz"
        if not scalefactors_path.exists():
            scalefactors_path = sample_dir / "scalefactors_json.json"
        if not scalefactors_path.exists() or not positions_path.exists():
            raise FileNotFoundError(f"Missing scalefactors for sample {sample}.")

        scalefactors = load_scalefactors(scalefactors_path)
        image_path, resolution, scale_factor = resolve_image_path_from_dir(
            sample_dir,
            scalefactors,
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
    summary_path.write_text(json.dumps(aggregate, indent=2))
    print(f"Wrote UNI2-h histology embedding summary to {summary_path}")


if __name__ == "__main__":
    main()
