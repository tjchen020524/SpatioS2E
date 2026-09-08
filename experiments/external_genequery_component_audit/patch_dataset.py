#!/usr/bin/env python3
"""Memory-mapped image-patch dataset for end-to-end GeneQuery training."""

from __future__ import annotations

import bisect
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from common import PATCH_ROOT, load_expression, load_manifest


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


class PatchExpressionDataset(Dataset):
    """A partition of patches aligned to a requested ordered gene panel."""

    def __init__(
        self,
        partition: str,
        gene_ids: Iterable[str],
        gene_mask: np.ndarray,
        patch_root: Path = PATCH_ROOT,
        augment: bool = False,
        max_spots: int = 0,
    ) -> None:
        manifest = load_manifest()
        samples = manifest.loc[manifest["split"] == partition, "sample"].astype(str).tolist()
        if not samples:
            raise ValueError("No samples for partition %s" % partition)
        requested = np.asarray(list(gene_ids)).astype(str)
        mask = np.asarray(gene_mask, dtype=bool)
        if len(mask) != len(requested):
            raise ValueError("Gene mask length does not match the requested panel")

        self.augment = bool(augment)
        self.patches = []
        self.targets = []
        sample_blocks = []
        barcode_blocks = []
        offsets = [0]
        remaining = max_spots if max_spots > 0 else None
        for sample in samples:
            patch_path = patch_root / sample / "patches.npy"
            barcode_path = patch_root / sample / "barcodes.npy"
            if not patch_path.exists() or not barcode_path.exists():
                raise FileNotFoundError("Missing patch artifact for %s under %s" % (sample, patch_root))
            patches = np.load(patch_path, mmap_mode="r", allow_pickle=False)
            patch_barcodes = np.load(barcode_path, allow_pickle=False).astype(str)
            expression, expression_barcodes = load_expression(sample, requested)
            if not np.array_equal(patch_barcodes, expression_barcodes):
                raise ValueError("Barcode order differs between patches and expression for %s" % sample)
            take = len(patch_barcodes)
            if remaining is not None:
                take = min(take, remaining)
            if take == 0:
                break
            self.patches.append(patches[:take])
            self.targets.append(expression[:take, mask])
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
        patch = torch.from_numpy(np.array(self.patches[block][within], copy=True)).permute(2, 0, 1)
        patch = patch.float().div_(255.0)
        if self.augment:
            if bool(torch.rand(()) < 0.5):
                patch = torch.flip(patch, dims=(2,))
            if bool(torch.rand(()) < 0.5):
                patch = torch.flip(patch, dims=(1,))
            patch = torch.rot90(patch, int(torch.randint(0, 4, ()).item()), dims=(1, 2))
        patch = (patch - IMAGENET_MEAN) / IMAGENET_STD
        target = torch.from_numpy(self.targets[block][within])
        return patch, target
