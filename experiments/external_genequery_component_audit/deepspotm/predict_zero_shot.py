#!/usr/bin/env python3
"""Run a component-ready DeepSpot-M full-checkpoint audit on HER2 test patients."""

from __future__ import annotations

import argparse
import bisect
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "experiments/external_genequery_component_audit"
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

from common import (  # noqa: E402
    PATCH_ROOT,
    counterfactual_embeddings,
    load_expression,
    load_manifest,
    load_panel_artifact,
    sha256,
)


DEFAULT_MODEL_ROOT = ROOT / "DeepSpotM"
DEFAULT_PANEL = EXP / "artifacts/deepspotm/her2_panel_scgpt.npz"
DEFAULT_OUTPUT_ROOT = EXP / "deepspotm/runs"
VARIANTS = ("semantic", "identity_shuffle", "random", "constant")
SOURCES = ("evo2", "orthrus", "prott5", "scgpt", "apertus")


class DeepSpotTestDataset(Dataset):
    """Raw 224-pixel HER2 patches and aligned held-out expression targets."""

    def __init__(
        self,
        gene_ids: np.ndarray,
        patch_root: Path,
        max_spots: int = 0,
    ) -> None:
        manifest = load_manifest()
        samples = manifest.loc[manifest["split"] == "test", "sample"].astype(str).tolist()
        self.patches = []
        self.targets = []
        sample_blocks = []
        barcode_blocks = []
        offsets = [0]
        remaining = max_spots if max_spots > 0 else None
        for sample in samples:
            patches = np.load(patch_root / sample / "patches.npy", mmap_mode="r", allow_pickle=False)
            patch_barcodes = np.load(
                patch_root / sample / "barcodes.npy", allow_pickle=False
            ).astype(str)
            expression, expression_barcodes = load_expression(sample, gene_ids)
            if not np.array_equal(patch_barcodes, expression_barcodes):
                raise ValueError("Patch/expression barcode mismatch for %s" % sample)
            if patches.ndim != 4 or tuple(patches.shape[1:]) != (224, 224, 3):
                raise ValueError("Unexpected patch shape for %s: %s" % (sample, patches.shape))
            take = len(patches)
            if remaining is not None:
                take = min(take, remaining)
            if take == 0:
                break
            self.patches.append(patches[:take])
            self.targets.append(expression[:take])
            sample_blocks.append(np.repeat(sample, take))
            barcode_blocks.append(patch_barcodes[:take])
            offsets.append(offsets[-1] + take)
            if remaining is not None:
                remaining -= take
        self.offsets = offsets
        self.sample_ids = np.concatenate(sample_blocks)
        self.barcodes = np.concatenate(barcode_blocks)

    def __len__(self) -> int:
        return self.offsets[-1]

    def __getitem__(self, index: int):
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        block = bisect.bisect_right(self.offsets, index) - 1
        within = index - self.offsets[block]
        patch = torch.from_numpy(np.array(self.patches[block][within], copy=True))
        patch = patch.permute(2, 0, 1).float().div_(127.5).sub_(1.0)
        target = torch.from_numpy(self.targets[block][within])
        return patch, target


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_amp_dtype(device: torch.device, requested: str):
    if device.type != "cuda" or requested == "none":
        return None
    if requested == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("bf16 was requested but is not supported by this GPU")
        return torch.bfloat16
    if requested == "fp16":
        return torch.float16
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--panel-artifact", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--patch-root", type=Path, default=PATCH_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--source", choices=SOURCES, default="scgpt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--amp-dtype", choices=("auto", "bf16", "fp16", "none"), default="auto")
    parser.add_argument("--max-spots", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    amp_dtype = resolve_amp_dtype(device, args.amp_dtype)
    panel = load_panel_artifact(args.panel_artifact)
    labels = panel["split_labels"].astype(str)
    heldout_mask = labels == "heldout"
    train_mask = labels == "train"
    if "model_indices" not in panel:
        raise ValueError("DeepSpot-M panel artifact lacks model_indices")
    if int(train_mask.sum()) == 0 or int(heldout_mask.sum()) == 0:
        raise ValueError("Panel artifact must contain fitted and held-out genes")

    heldout_gene_ids = panel["gene_ids"].astype(str)[heldout_mask]
    dataset = DeepSpotTestDataset(heldout_gene_ids, args.patch_root, args.max_spots)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
        drop_last=False,
    )
    patient_map = load_manifest().set_index("sample")["patient"].astype(str).to_dict()
    individuals = np.asarray([patient_map[sample] for sample in dataset.sample_ids])

    for variant in VARIANTS:
        output = args.output_root / ("seed_%d" % args.seed) / variant / "predictions.npz"
        if output.exists() and not args.overwrite:
            raise FileExistsError(output)

    from deepspotm import DeepSpotM

    start = time.time()
    model, official_processor = DeepSpotM.from_pretrained(
        str(args.model_root), source=args.source, device=device
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if model.current_source != args.source:
        raise RuntimeError("Requested source was not activated")
    gene_names = list(model.gene_names)
    model_indices = panel["model_indices"].astype(np.int64)
    if [gene_names[index] for index in model_indices] != panel["symbols"].astype(str).tolist():
        raise ValueError("Panel symbols do not match DeepSpot-M model indices")

    multi_source = model.gene_decoder.multi_source
    if multi_source is None:
        raise RuntimeError("Expected a multi-source released checkpoint")
    bio_parameter = getattr(multi_source, "bio_%s" % args.source)
    panel_vectors = bio_parameter.detach()[torch.as_tensor(model_indices, device=device)].float().cpu().numpy()
    artifact_vectors = panel["semantic_embeddings"].astype(np.float32)
    embedding_parity_error = float(np.max(np.abs(panel_vectors - artifact_vectors)))
    if embedding_parity_error != 0.0:
        raise AssertionError("Panel/checkpoint embedding mismatch: %g" % embedding_parity_error)

    variant_vectors = {}
    for variant in VARIANTS:
        vectors = counterfactual_embeddings(artifact_vectors, labels, variant, args.seed)
        variant_vectors[variant] = torch.from_numpy(vectors[heldout_mask]).to(device)
    heldout_model_indices = torch.as_tensor(model_indices[heldout_mask], dtype=torch.long, device=device)

    # Existing stored patches are already exactly 224x224, so x / 127.5 - 1 is
    # algebraically the released Midnight transform. Verify this once against
    # the package-provided processor before scoring any spot.
    first_raw = np.array(dataset.patches[0][0], copy=True)
    official_first = official_processor(first_raw)
    direct_first, _ = dataset[0]
    transform_parity_error = float(torch.max(torch.abs(official_first - direct_first)).item())
    if transform_parity_error != 0.0:
        raise AssertionError("Image-transform mismatch: %g" % transform_parity_error)

    predictions = {variant: [] for variant in VARIANTS}
    truths = []
    forward_parity_error = None
    amp_enabled = amp_dtype is not None
    with torch.inference_mode():
        for batch_index, (images, target) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            truths.append(target.numpy())
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype if amp_dtype is not None else torch.float32,
                enabled=amp_enabled,
            ):
                patch_tokens = model.backbone.forward_patch_tokens(images)
                for variant in VARIANTS:
                    bio_parameter.index_copy_(
                        0, heldout_model_indices, variant_vectors[variant]
                    )
                    expression, _, _ = model.gene_decoder(
                        patch_tokens, need_weights=False, gene_indices=heldout_model_indices
                    )
                    predictions[variant].append(expression.float().cpu().numpy())
                if batch_index == 0:
                    bio_parameter.index_copy_(
                        0, heldout_model_indices, variant_vectors["semantic"]
                    )
                    full_expression, _, _ = model(images, gene_indices=heldout_model_indices)
                    forward_parity_error = float(
                        torch.max(
                            torch.abs(
                                full_expression.float()
                                - torch.from_numpy(predictions["semantic"][-1]).to(device)
                            )
                        ).item()
                    )
            if batch_index % 25 == 0:
                print(
                    json.dumps(
                        {
                            "batch": batch_index,
                            "n_batches": len(loader),
                            "elapsed_seconds": time.time() - start,
                        }
                    ),
                    flush=True,
                )
    bio_parameter.index_copy_(0, heldout_model_indices, variant_vectors["semantic"])

    true = np.concatenate(truths).astype(np.float32, copy=False)
    elapsed = time.time() - start
    checkpoint_sha256 = sha256(args.model_root / "model.safetensors")
    for variant in VARIANTS:
        pred = np.concatenate(predictions[variant]).astype(np.float32, copy=False)
        if pred.shape != true.shape or not np.isfinite(pred).all():
            raise ValueError("Invalid %s prediction array: %s" % (variant, pred.shape))
        run_dir = args.output_root / ("seed_%d" % args.seed) / variant
        run_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            run_dir / "predictions.npz",
            pred=pred,
            true=true,
            gene_ids=heldout_gene_ids,
            symbols=panel["symbols"].astype(str)[heldout_mask],
            sample_ids=dataset.sample_ids,
            individual_ids=individuals,
            barcodes=dataset.barcodes,
        )
        metadata = {
            "method": "DeepSpot-M",
            "variant": variant,
            "seed": args.seed,
            "evaluation_mode": "released_full_checkpoint_zero_shot",
            "upstream_spatial_target_exposure": True,
            "source": args.source,
            "source_tokens": "spatially_trained_checkpoint_parameters",
            "target_scale": "log1p_counts_per_million",
            "model_root": str(args.model_root),
            "model_checkpoint_sha256": checkpoint_sha256,
            "panel_artifact": str(args.panel_artifact),
            "n_test_spots": int(len(dataset)),
            "n_train_panel_genes": int(train_mask.sum()),
            "n_heldout_genes": int(heldout_mask.sum()),
            "batch_size": args.batch_size,
            "amp_dtype": str(amp_dtype).replace("torch.", "") if amp_dtype else "none",
            "embedding_parity_error": embedding_parity_error,
            "transform_parity_error": transform_parity_error,
            "forward_parity_error": forward_parity_error,
            "elapsed_seconds_shared_job": elapsed,
            "max_cuda_memory_gib": (
                float(torch.cuda.max_memory_allocated() / 1024**3)
                if device.type == "cuda"
                else None
            ),
            "counterfactual_definition": (
                "Reassign or replace the released source-token rows used jointly by "
                "the query adapter and gene router; random/constant moments use fitted "
                "HER2 panel genes only."
            ),
        }
        (run_dir / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print("Saved %s" % (run_dir / "predictions.npz"), flush=True)


if __name__ == "__main__":
    main()
