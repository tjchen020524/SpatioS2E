#!/usr/bin/env python3
"""Run Hist2ST train or evaluation with the cluster PyTorch compatibility shim."""

from __future__ import annotations
from workflow_paths import CODE_ROOT, DATA_ROOT

import argparse
import runpy
import sys
from pathlib import Path


ROOT = DATA_ROOT
EXP = DATA_ROOT / 'experiments/hippocampus_donor_disjoint_external_benchmark'


def patch_torch_pytree() -> None:
    import torch.utils._pytree as pytree

    if hasattr(pytree, "register_pytree_node") or not hasattr(pytree, "_register_pytree_node"):
        return

    def register_pytree_node(node_type, flatten_fn, unflatten_fn, *args, **kwargs):
        return pytree._register_pytree_node(node_type, flatten_fn, unflatten_fn)

    pytree.register_pytree_node = register_pytree_node


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("train", "eval"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    run = EXP / f"runs/hist2st/seed_{args.seed}"

    patch_torch_pytree()
    if args.mode == "train":
        script = CODE_ROOT / "experiments/hist2st_spatial/train_hist2st_spot.py"
        sys.argv = [str(script), "--config", str(EXP / f"configs/hist2st_seed_{args.seed}.json")]
    else:
        script = CODE_ROOT / "experiments/hist2st_spatial/eval_hist2st_spot.py"
        sys.argv = [
            str(script),
            "--config", str(EXP / f"configs/hist2st_eval_seed_{args.seed}.json"),
            "--ckpt", str(run / "checkpoints/hist2st_best.pt"),
            "--split", "test",
            "--save-dir", str(run / "results"),
        ]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()

