#!/usr/bin/env python3
"""Run one strict-split ST-Net seed using the existing method implementation."""

from __future__ import annotations
from workflow_paths import CODE_ROOT, DATA_ROOT

import argparse
import functools
import importlib.util
import sys
from pathlib import Path


ROOT = DATA_ROOT
EXP = DATA_ROOT / 'experiments/hippocampus_donor_disjoint_external_benchmark'
SOURCE = CODE_ROOT / "experiments/benchmark_stnet_hippocampus_fullgenes"


def patch_torch_pytree() -> None:
    import torch.utils._pytree as pytree

    if hasattr(pytree, "register_pytree_node") or not hasattr(pytree, "_register_pytree_node"):
        return

    def register_pytree_node(node_type, flatten_fn, unflatten_fn, *args, **kwargs):
        return pytree._register_pytree_node(node_type, flatten_fn, unflatten_fn)

    pytree.register_pytree_node = register_pytree_node


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    run_dir = EXP / f"runs/stnet/seed_{args.seed}"

    patch_torch_pytree()
    from PIL import ImageFile

    ImageFile.LOAD_TRUNCATED_IMAGES = True
    train = load_module("strict_stnet_train", SOURCE / "run_stnet_fullgenes_fixedsplit.py")
    train.EXP_ROOT = run_dir
    spatial_dataset = train.stnet.datasets.Spatial
    train.stnet.datasets.Spatial = functools.partial(spatial_dataset, root=str(run_dir / "processed"))
    sys.argv = [
        str(SOURCE / "run_stnet_fullgenes_fixedsplit.py"),
        "--epochs", str(args.epochs),
        "--batch", "4",
        "--test-batch", "1",
        "--workers", "2",
        "--window", "224",
        "--model", "densenet121",
        "--lr", "1e-6",
        "--seed", str(args.seed),
    ]
    train.main()

    evaluate = load_module("strict_stnet_eval", SOURCE / "eval_stnet_fullgenes.py")
    evaluate.EXP_ROOT = run_dir
    evaluate.SAVE_DIR = run_dir / "results"
    evaluate.OUT_NPZ = run_dir / "output/stnet_fullgenes_test.npz"
    evaluate.main()


if __name__ == "__main__":
    main()
