#!/usr/bin/env python3
"""Run the held-out-gene decoder with frozen static scGPT gene-token vectors.

This experiment is deliberately matched to the primary Decima assay.  The
spot inputs, subject split, original gene partition, decoder, optimizer and
metric policy are unchanged.  The only differences are that genes absent from
the scGPT brain-model vocabulary are removed from *all* conditions and the
gene vectors are either the correct frozen token vectors, dimension-matched
random vectors, one identical constant vector or the same pretrained vector
multiset with gene identity permuted independently within each partition.

The static token representation is not the donor-specific single-cell context
used by the older fitted-gene scGPT experiments in this repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

from experiments.geneheldout_uni2h_decima_decoder_noseq_controls import (
    run_geneheldout_noseq_controls as base,
)
from experiments.heldout_gene_second_stage.audit_scgpt_static_gene_vectors import (
    cohort_features,
    token_feature_lookup,
)
from experiments.multicohort_geneheldout_decima_clean_split import (
    evaluate_components as component_eval,
)
from experiments.multicohort_geneheldout_decima_clean_split.clean_split_config import (
    COHORTS,
    ROOT,
    SEEDS,
    configure_clean_base,
)


CHECKPOINTS = {
    "brain": {
        "model_dir": ROOT / "scGPT/brain_model",
        "out": ROOT / "experiments/heldout_gene_second_stage/scgpt_decoder",
        "label": "scGPT brain static gene-token vectors",
    },
    "whole_human": {
        "model_dir": ROOT / "scGPT/whole_human_model",
        "out": ROOT
        / "experiments/heldout_gene_second_stage/scgpt_whole_human_decoder",
        "label": "scGPT whole-human static gene-token vectors",
    },
}
CONDITIONS = (
    "pretrained_gene_tokens",
    "random_gene_vectors",
    "constant_gene_vector",
    "identity_shuffled_gene_tokens",
)
INTERNAL_VARIANT = "decima"


def build_embeddings(
    condition: str,
    raw_features: np.ndarray,
    train_idx: np.ndarray,
    heldout_idx: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, dict[str, object]]:
    """Construct one vector condition and freeze its identity mapping."""
    eligible_idx = np.concatenate([train_idx, heldout_idx])
    metadata: dict[str, object] = {}
    if condition == "pretrained_gene_tokens":
        embeddings = raw_features.copy()
        mean = embeddings[train_idx].mean(axis=0, keepdims=True)
        std = embeddings[train_idx].std(axis=0, keepdims=True)
        embeddings = (embeddings - mean) / np.maximum(std, 1.0e-6)
    elif condition == "random_gene_vectors":
        rng = np.random.default_rng(seed + 12_345)
        embeddings = np.zeros_like(raw_features, dtype=np.float32)
        embeddings[eligible_idx] = rng.standard_normal(
            (len(eligible_idx), raw_features.shape[1])
        ).astype(np.float32)
        metadata["random_seed"] = int(seed + 12_345)
    elif condition == "constant_gene_vector":
        embeddings = np.zeros_like(raw_features, dtype=np.float32)
    elif condition == "identity_shuffled_gene_tokens":
        permutation_seed = int(seed + 810_001)
        rng = np.random.default_rng(permutation_seed)
        source_index = np.arange(len(raw_features), dtype=np.int64)
        # Preserve the exact vector multiset separately within the downstream
        # training and held-out partitions while breaking gene identity.
        source_index[train_idx] = rng.permutation(train_idx)
        source_index[heldout_idx] = rng.permutation(heldout_idx)
        if not np.array_equal(np.sort(source_index[train_idx]), np.sort(train_idx)):
            raise AssertionError("Training-partition vector multiset was not preserved")
        if not np.array_equal(np.sort(source_index[heldout_idx]), np.sort(heldout_idx)):
            raise AssertionError("Held-out-partition vector multiset was not preserved")
        embeddings = raw_features[source_index].copy()
        mean = embeddings[train_idx].mean(axis=0, keepdims=True)
        std = embeddings[train_idx].std(axis=0, keepdims=True)
        embeddings = (embeddings - mean) / np.maximum(std, 1.0e-6)
        metadata.update(
            {
                "permutation_seed": permutation_seed,
                "permutation_scope": "independent within downstream training and held-out partitions",
                "permutation_fixed_throughout_training_validation_and_test": True,
                "source_index_sha256": hashlib.sha256(
                    source_index.astype("<i8", copy=False).tobytes()
                ).hexdigest(),
            }
        )
    else:
        raise ValueError(condition)
    return embeddings.astype(np.float32, copy=False), metadata


def forwarded_args(seed: int) -> list[str]:
    """Return the hyperparameters frozen for the primary held-out assay."""
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
        "--variants", INTERNAL_VARIANT,
        "--paired-rng",
        "--no-train-mean-baseline",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", required=True, choices=CONDITIONS)
    parser.add_argument("--cohort", required=True, choices=COHORTS)
    parser.add_argument("--seed", required=True, type=int, choices=SEEDS)
    parser.add_argument("--checkpoint", choices=CHECKPOINTS, default="brain")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    checkpoint = CHECKPOINTS[args.checkpoint]
    out = checkpoint["out"]

    # Configure the same cohort and original gene split used by the primary
    # assay before installing the scGPT-specific loaders below.
    configure_clean_base(args.cohort, args.seed, INTERNAL_VARIANT)
    gene_ids = base.load_gene_ids()
    original_train_idx, original_heldout_idx = base.load_gene_split(gene_ids)
    lookup, model_metadata = token_feature_lookup(checkpoint["model_dir"])
    raw_features, covered, coverage = cohort_features(gene_ids, lookup)
    train_idx = original_train_idx[covered[original_train_idx]]
    heldout_idx = original_heldout_idx[covered[original_heldout_idx]]
    if len(train_idx) < 2 or len(heldout_idx) < 2:
        raise ValueError("Insufficient scGPT-vocabulary-covered genes")

    embeddings, condition_metadata = build_embeddings(
        args.condition, raw_features, train_idx, heldout_idx, args.seed
    )

    def split_loader(current_gene_ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
        if current_gene_ids != gene_ids:
            raise ValueError("Gene order changed after scGPT initialization")
        return train_idx, heldout_idx

    def embedding_loader(
        current_gene_ids: list[str], current_train_idx: np.ndarray
    ) -> np.ndarray:
        if current_gene_ids != gene_ids:
            raise ValueError("Gene order changed after scGPT initialization")
        if not np.array_equal(current_train_idx, train_idx):
            raise ValueError("Training-gene set changed after scGPT initialization")
        return embeddings

    run_root = out / args.condition / "runs" / args.cohort / f"seed_{args.seed}"
    run_root.mkdir(parents=True, exist_ok=True)
    base.EXP_ROOT = run_root
    base.load_gene_split = split_loader
    base.load_decima_embeddings = embedding_loader

    metadata = {
        "condition": args.condition,
        "cohort": args.cohort,
        "seed": args.seed,
        "representation": (
            model_metadata["representation"]
            if args.condition in (
                "pretrained_gene_tokens",
                "identity_shuffled_gene_tokens",
            )
            else args.condition.replace("_", " ")
        ),
        "n_original_training_genes": int(len(original_train_idx)),
        "n_original_heldout_genes": int(len(original_heldout_idx)),
        "n_training_genes": int(len(train_idx)),
        "n_heldout_genes": int(len(heldout_idx)),
        "training_gene_coverage": float(len(train_idx) / len(original_train_idx)),
        "heldout_gene_coverage": float(len(heldout_idx) / len(original_heldout_idx)),
        "checkpoint_scope": args.checkpoint,
        "coverage_rule": f"present in local scGPT {args.checkpoint.replace('_', '-')}-model vocabulary; identical denominator for all conditions",
        "embedding_standardization": (
            "feature-wise mean and standard deviation from covered downstream training genes"
            if args.condition in (
                "pretrained_gene_tokens",
                "identity_shuffled_gene_tokens",
            )
            else "not applicable"
        ),
        "decoder_and_training": "identical to the primary held-out-gene factorized dot-product decoder assay",
        **condition_metadata,
        **model_metadata,
    }
    (run_root / "control_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    if args.dry_run:
        print(json.dumps(metadata, indent=2))
        return

    sys.argv = [sys.argv[0], *forwarded_args(args.seed)]
    base.main()

    # Reuse the publication evaluator with the matched covered-gene split and
    # exact embedding matrix installed above.
    component_eval.configure_clean_base = lambda cohort, seed, variant: None
    component_eval.embedding_matrix = (
        lambda variant, current_gene_ids, current_train_idx, seed: embeddings
    )
    component_eval.CLEAN_ROOT = out / args.condition
    summary = component_eval.run_one(
        cohort=args.cohort,
        seed=args.seed,
        variant=INTERNAL_VARIANT,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    summary["condition"] = args.condition
    summary["representation_family"] = checkpoint["label"]
    summary["coverage_metadata"] = str(
        (run_root / "control_metadata.json").relative_to(ROOT)
    )
    output = (
        out
        / args.condition
        / "components/runs"
        / args.cohort
        / f"seed_{args.seed}"
        / INTERNAL_VARIANT
        / "summary.json"
    )
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
