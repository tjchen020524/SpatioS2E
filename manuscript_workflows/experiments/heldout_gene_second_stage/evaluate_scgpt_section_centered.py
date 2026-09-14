#!/usr/bin/env python3
"""Re-evaluate frozen scGPT-decoder checkpoints after centring each section.

No parameters are fitted here.  The script reconstructs the exact covered-gene
denominator and condition-specific vectors used during training, reloads one
frozen checkpoint and delegates the streaming evaluation to the publication
section-centred evaluator.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.geneheldout_uni2h_decima_decoder_noseq_controls import (
    run_geneheldout_noseq_controls as base,
)
from experiments.heldout_gene_second_stage.audit_scgpt_static_gene_vectors import (
    cohort_features,
    token_feature_lookup,
)
from experiments.heldout_gene_second_stage.run_scgpt_gene_token_decoder import (
    build_embeddings,
    CHECKPOINTS,
    CONDITIONS,
    INTERNAL_VARIANT,
)
from experiments.multicohort_geneheldout_decima import (
    evaluate_section_centered as section_eval,
)
from experiments.multicohort_geneheldout_decima_clean_split.clean_split_config import (
    COHORTS,
    ROOT,
    SEEDS,
    configure_clean_base,
)


def prepare_inputs(
    condition: str, cohort: str, seed: int, checkpoint_scope: str = "brain"
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    """Reconstruct the exact split and vectors used for a completed run."""
    configure_clean_base(cohort, seed, INTERNAL_VARIANT)
    gene_ids = base.load_gene_ids()
    original_train_idx, original_heldout_idx = base.load_gene_split(gene_ids)
    checkpoint = CHECKPOINTS[checkpoint_scope]
    lookup, model_metadata = token_feature_lookup(checkpoint["model_dir"])
    raw_features, covered, _ = cohort_features(gene_ids, lookup)
    train_idx = original_train_idx[covered[original_train_idx]]
    heldout_idx = original_heldout_idx[covered[original_heldout_idx]]

    embeddings, condition_metadata = build_embeddings(
        condition, raw_features, train_idx, heldout_idx, seed
    )

    metadata = {
        "condition": condition,
        "cohort": cohort,
        "seed": seed,
        "checkpoint_scope": checkpoint_scope,
        "n_training_genes": int(len(train_idx)),
        "n_heldout_genes": int(len(heldout_idx)),
        "embedding_dim": int(embeddings.shape[1]),
        "representation": (
            model_metadata["representation"]
            if condition in (
                "pretrained_gene_tokens",
                "identity_shuffled_gene_tokens",
            )
            else condition.replace("_", " ")
        ),
        **condition_metadata,
    }
    return gene_ids, train_idx, heldout_idx, embeddings.astype(np.float32), metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", required=True, choices=CONDITIONS)
    parser.add_argument("--cohort", required=True, choices=COHORTS)
    parser.add_argument("--seed", required=True, type=int, choices=SEEDS)
    parser.add_argument("--checkpoint", choices=CHECKPOINTS, default="brain")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--gene-chunk-size", type=int, default=512)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    gene_ids, train_idx, heldout_idx, embeddings, metadata = prepare_inputs(
        args.condition, args.cohort, args.seed, args.checkpoint
    )
    checkpoint_config = CHECKPOINTS[args.checkpoint]
    out = checkpoint_config["out"]
    run_root = out / args.condition / "runs" / args.cohort / f"seed_{args.seed}"
    checkpoint = run_root / INTERNAL_VARIANT / "checkpoints/best.pt"
    metadata["checkpoint"] = str(checkpoint.relative_to(ROOT))
    metadata["checkpoint_exists"] = checkpoint.exists()
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    if args.dry_run:
        print(json.dumps(metadata, indent=2))
        return

    def split_loader(current_gene_ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
        if current_gene_ids != gene_ids:
            raise ValueError("Gene order changed during section-centred evaluation")
        return train_idx, heldout_idx

    base.EXP_ROOT = run_root
    base.load_gene_split = split_loader
    section_eval.configure_clean_base = lambda cohort, seed, variant: None
    section_eval.embedding_matrix = (
        lambda variant, current_gene_ids, current_train_idx, seed: embeddings
    )
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    output_root = out / "section_centered" / args.condition
    summary = section_eval.evaluate_one(
        cohort=args.cohort,
        seed=args.seed,
        variant=INTERNAL_VARIANT,
        device=device,
        batch_size=args.batch_size,
        gene_chunk_size=args.gene_chunk_size,
        max_sections=None,
        max_genes=None,
        output_root=output_root,
        partition="training_only",
    )
    summary.update(
        {
            "condition": args.condition,
            "representation_family": checkpoint_config["label"],
            "evaluation": "centre observed and predicted expression separately within each test section",
        }
    )
    summary_path = (
        output_root
        / "runs"
        / args.cohort
        / f"seed_{args.seed}"
        / INTERNAL_VARIANT
        / "summary.json"
    )
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
