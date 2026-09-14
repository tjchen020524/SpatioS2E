#!/usr/bin/env python3
"""Run one strict-split BLEEP seed using the existing method implementation."""

from __future__ import annotations
from workflow_paths import CODE_ROOT, DATA_ROOT

import argparse
import importlib.util
import sys
from pathlib import Path


ROOT = DATA_ROOT
EXP = DATA_ROOT / 'experiments/hippocampus_donor_disjoint_external_benchmark'
SOURCE = CODE_ROOT / "experiments/benchmark_bleep_hippocampus_fullgenes/run_bleep_fullgenes.py"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    run_dir = EXP / f"runs/bleep/seed_{args.seed}"

    spec = importlib.util.spec_from_file_location("strict_bleep", SOURCE)
    if spec is None or spec.loader is None:
        raise ImportError(SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.EXP_ROOT = run_dir
    module.DATA_ROOT = EXP / "data/bleep"
    module.EXPRESSION_ROOT = EXP / "data/bleep/expression_proxy"
    sys.argv = [
        str(SOURCE),
        "--epochs", "10",
        "--batch-size", "64",
        "--eval-batch-size", "128",
        "--workers", "2",
        "--lr", "1e-4",
        "--weight-decay", "1e-4",
        "--top-k", "50",
        "--impute-chunk-size", "64",
        "--seed", str(args.seed),
        "--pretrained",
    ]
    module.main()


if __name__ == "__main__":
    main()

