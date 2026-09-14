#!/usr/bin/env python3
"""Train and evaluate one decoder under an additional gene partition."""

from __future__ import annotations
from workflow_paths import DATA_ROOT

import argparse
import json
from pathlib import Path
import sys

import torch

from experiments.geneheldout_uni2h_decima_decoder_noseq_controls import (
    run_geneheldout_noseq_controls as base,
)
from experiments.multicohort_geneheldout_decima.run_cohort_geneheldout import (
    configure_base,
)
from experiments.multicohort_geneheldout_decima_clean_split import (
    evaluate_components as component_eval,
)
from experiments.multicohort_geneheldout_decima_clean_split.clean_split_config import (
    COHORTS,
    SEEDS,
)
from experiments.heldout_gene_second_stage.build_gene_partitions import PARTITIONS


ROOT = DATA_ROOT
PARTITION_ROOT = ROOT / "experiments/heldout_gene_second_stage/gene_partitions"
VARIANTS = ("decima", "random")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partition", required=True, choices=PARTITIONS)
    parser.add_argument("--cohort", required=True, choices=COHORTS)
    parser.add_argument("--seed", required=True, type=int, choices=SEEDS)
    parser.add_argument("--variant", required=True, choices=VARIANTS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    configure_base(args.cohort, args.seed, args.variant)
    root = PARTITION_ROOT / args.partition
    base.EXP_ROOT = root / "runs" / args.cohort / f"seed_{args.seed}" / args.variant
    base.GENE_SPLIT_DIR = root / "data" / args.cohort / "gene_splits"
    gene_ids = base.load_gene_ids()
    train_idx, heldout_idx = base.load_gene_split(gene_ids)
    metadata = {
        "partition": args.partition,
        "cohort": args.cohort,
        "seed": args.seed,
        "variant": args.variant,
        "n_training_genes": int(len(train_idx)),
        "n_heldout_genes": int(len(heldout_idx)),
        "partition_manifest": str((PARTITION_ROOT / "manifest.json").relative_to(ROOT)),
    }
    base.EXP_ROOT.mkdir(parents=True, exist_ok=True)
    (base.EXP_ROOT / "partition_run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    if args.dry_run:
        print(json.dumps(metadata, indent=2))
        return

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

    component_eval.configure_clean_base = lambda cohort, seed, variant: None
    component_eval.CLEAN_ROOT = root
    summary = component_eval.run_one(
        cohort=args.cohort,
        seed=args.seed,
        variant=args.variant,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    summary["partition"] = args.partition
    output = (
        root
        / "components/runs"
        / args.cohort
        / f"seed_{args.seed}"
        / args.variant
        / "summary.json"
    )
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
