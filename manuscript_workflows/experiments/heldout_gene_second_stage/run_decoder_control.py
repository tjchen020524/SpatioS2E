#!/usr/bin/env python3
"""Train one shuffled-vector or fitted-target oracle decoder and evaluate it.

The shuffled condition permutes standardized pretrained vectors separately
within the downstream training and held-out gene sets.  This exactly preserves
the vector multiset in each partition while breaking gene--vector identity
before optimization.  The oracle condition keeps the original held-out target
set for evaluation but allows those genes to participate in decoder fitting;
the embedding standardization remains fixed to the original training genes.
"""

from __future__ import annotations
from workflow_paths import DATA_ROOT

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

from experiments.geneheldout_uni2h_decima_decoder_noseq_controls import (
    run_geneheldout_noseq_controls as base,
)
from experiments.multicohort_geneheldout_decima_clean_split import (
    evaluate_components as component_eval,
)
from experiments.multicohort_geneheldout_decima_clean_split.clean_split_config import (
    COHORTS,
    SEEDS,
    configure_clean_base,
)


ROOT = DATA_ROOT
SECOND_ROOT = ROOT / "experiments/heldout_gene_second_stage"
CONDITIONS = ("shuffled_pretrained", "fitted_target_oracle")


def forwarded_args(seed: int) -> list[str]:
    return [
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
        "--seed", str(seed),
        "--variants", "decima",
        "--paired-rng",
        "--no-train-mean-baseline",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", required=True, choices=CONDITIONS)
    parser.add_argument("--cohort", required=True, choices=COHORTS)
    parser.add_argument("--seed", required=True, type=int, choices=SEEDS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    configure_clean_base(args.cohort, args.seed, "decima")
    original_split_loader = base.load_gene_split
    original_embedding_loader = base.load_decima_embeddings
    gene_ids = base.load_gene_ids()
    original_train_idx, original_heldout_idx = original_split_loader(gene_ids)
    original_embeddings = original_embedding_loader(gene_ids, original_train_idx)

    run_root = (
        SECOND_ROOT
        / "decoder_controls"
        / args.condition
        / "runs"
        / args.cohort
        / f"seed_{args.seed}"
    )
    run_root.mkdir(parents=True, exist_ok=True)
    base.EXP_ROOT = run_root

    mapping_rows: list[dict[str, object]] = []
    if args.condition == "shuffled_pretrained":
        rng = np.random.default_rng(args.seed + 810_001)
        source_index = np.arange(len(gene_ids), dtype=np.int64)
        for partition, indices in (
            ("training", original_train_idx),
            ("held_out", original_heldout_idx),
        ):
            permuted = rng.permutation(indices)
            source_index[indices] = permuted
            mapping_rows.extend(
                {
                    "cohort": args.cohort,
                    "seed": args.seed,
                    "partition": partition,
                    "target_gene": gene_ids[int(target)],
                    "source_vector_gene": gene_ids[int(source)],
                }
                for target, source in zip(indices.tolist(), permuted.tolist())
            )
        condition_embeddings = original_embeddings[source_index]

        def condition_split_loader(current_gene_ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
            if current_gene_ids != gene_ids:
                raise ValueError("Gene order changed after shuffled-control initialization")
            return original_train_idx, original_heldout_idx

    else:
        condition_embeddings = original_embeddings

        def condition_split_loader(current_gene_ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
            if current_gene_ids != gene_ids:
                raise ValueError("Gene order changed after oracle-control initialization")
            return np.arange(len(gene_ids), dtype=np.int64), original_heldout_idx

    def condition_embedding_loader(
        current_gene_ids: list[str], train_idx: np.ndarray
    ) -> np.ndarray:
        if current_gene_ids != gene_ids:
            raise ValueError("Gene order changed after control initialization")
        # Both controls use the original training-gene standardization.  The
        # oracle therefore changes target availability, not feature scaling.
        return condition_embeddings

    base.load_gene_split = condition_split_loader
    base.load_decima_embeddings = condition_embedding_loader

    metadata = {
        "condition": args.condition,
        "cohort": args.cohort,
        "seed": args.seed,
        "n_original_training_genes": int(len(original_train_idx)),
        "n_original_heldout_genes": int(len(original_heldout_idx)),
        "n_decoder_training_genes": int(
            len(original_train_idx)
            if args.condition == "shuffled_pretrained"
            else len(gene_ids)
        ),
        "embedding_standardization": "original downstream training genes",
        "shuffling": (
            "independent within-partition permutations before training"
            if args.condition == "shuffled_pretrained"
            else "none"
        ),
        "interpretation": (
            "distribution- and covariance-matched identity control"
            if args.condition == "shuffled_pretrained"
            else "subject-disjoint fitted-target capacity ceiling"
        ),
    }
    (run_root / "control_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if mapping_rows:
        import pandas as pd

        pd.DataFrame(mapping_rows).to_csv(
            run_root / "gene_vector_permutation.tsv.gz",
            sep="\t",
            index=False,
            compression={"method": "gzip", "mtime": 0},
        )
    if args.dry_run:
        print(json.dumps(metadata, indent=2))
        return

    sys.argv = [sys.argv[0], *forwarded_args(args.seed)]
    base.main()

    # Reuse the publication evaluator, but keep the control-specific split,
    # embedding matrix and checkpoint root installed above.
    component_eval.configure_clean_base = lambda cohort, seed, variant: None
    component_eval.embedding_matrix = (
        lambda variant, current_gene_ids, train_idx, seed: condition_embeddings
    )
    component_eval.CLEAN_ROOT = (
        SECOND_ROOT / "decoder_controls" / args.condition
    )
    summary = component_eval.run_one(
        cohort=args.cohort,
        seed=args.seed,
        variant="decima",
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    summary["condition"] = args.condition
    summary["control_metadata"] = str(
        (run_root / "control_metadata.json").relative_to(ROOT)
    )
    output = (
        SECOND_ROOT
        / "decoder_controls"
        / args.condition
        / "components"
        / "runs"
        / args.cohort
        / f"seed_{args.seed}"
        / "decima"
        / "summary.json"
    )
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
