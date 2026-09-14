#!/usr/bin/env python3
"""Train one clean-partition replication and export component metrics."""

from __future__ import annotations

import argparse
import sys

import torch

from experiments.geneheldout_uni2h_decima_decoder_noseq_controls import (
    run_geneheldout_noseq_controls as base,
)
from experiments.multicohort_geneheldout_decima_clean_split.clean_split_config import (
    COHORTS,
    SEEDS,
    VARIANTS,
    configure_clean_base,
)
from experiments.multicohort_geneheldout_decima_clean_split.evaluate_components import run_one


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", required=True, choices=COHORTS)
    parser.add_argument("--seed", required=True, type=int, choices=SEEDS)
    parser.add_argument("--variant", required=True, choices=VARIANTS)
    args = parser.parse_args()
    configure_clean_base(args.cohort, args.seed, args.variant)
    forwarded = [
        "--epochs", "6",
        "--batch-size", "256",
        "--eval-batch-size", "256",
        "--workers", "0",
        "--hidden-dim", "512",
        "--program-dim", "96",
        "--train-genes-per-batch", "512",
        "--val-train-genes", "2048",
        "--report-train-genes", "2048",
        "--eval-gene-chunk-size", "512",
        "--lambda-gene-corr", "0.15",
        "--lr", "2.0e-4",
        "--weight-decay", "1.0e-4",
        "--seed", str(args.seed),
        "--variants", args.variant,
        "--paired-rng",
        "--no-train-mean-baseline",
    ]
    sys.argv = [sys.argv[0], *forwarded]
    base.main()
    run_one(
        cohort=args.cohort,
        seed=args.seed,
        variant=args.variant,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )


if __name__ == "__main__":
    main()
