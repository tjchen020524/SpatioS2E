#!/usr/bin/env python3
"""Train a non-bilinear concatenation-MLP held-out-gene decoder.

The first joint layer is written as separate spot and gene projections for
memory efficiency, but is algebraically identical to a linear layer applied to
their concatenation.  There is no separate gene-only skip branch.  The model
can nevertheless learn an implicit gene main effect, so this experiment tests
decoder-family robustness rather than removal of an abundance pathway.
"""

from __future__ import annotations
from workflow_paths import DATA_ROOT

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = DATA_ROOT
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from experiments.geneheldout_uni2h_decima_decoder_noseq_controls import (
    run_geneheldout_noseq_controls as base,
)
from experiments.multicohort_geneheldout_decima.run_cohort_geneheldout import configure_base
from experiments.multicohort_geneheldout_decima_clean_split import evaluate_components as component_eval
from experiments.multicohort_geneheldout_decima_clean_split.clean_split_config import (
    CLEAN_ROOT,
    COHORTS,
    SEEDS,
    VARIANTS,
)


ROOT = DATA_ROOT
EXPERIMENT_ROOT = ROOT / "experiments/heldout_decoder_architecture_audit/concat_mlp"


class ConcatenationMLPDecoder(torch.nn.Module):
    """One-hidden-layer MLP on concatenated spot and gene vectors."""

    def __init__(
        self,
        spot_dim: int,
        gene_dim: int,
        hidden_dim: int,
        program_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        del program_dim
        self.spot_norm = torch.nn.LayerNorm(spot_dim)
        self.gene_norm = torch.nn.LayerNorm(gene_dim)
        # One bias across the two terms exactly matches Linear([x || e]).
        self.spot_projection = torch.nn.Linear(spot_dim, hidden_dim, bias=True)
        self.gene_projection = torch.nn.Linear(gene_dim, hidden_dim, bias=False)
        self.dropout = torch.nn.Dropout(dropout)
        self.output = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, gene_emb: torch.Tensor) -> torch.Tensor:
        spot_hidden = self.spot_projection(self.spot_norm(x))
        gene_hidden = self.gene_projection(self.gene_norm(gene_emb))
        joint = F.gelu(spot_hidden[:, None, :] + gene_hidden[None, :, :])
        logits = self.output(self.dropout(joint)).squeeze(-1)
        return F.softplus(logits)


def configure_audit(cohort: str, seed: int, variant: str) -> None:
    configure_base(cohort, seed, variant)
    base.EXP_ROOT = EXPERIMENT_ROOT / "runs" / cohort / f"seed_{seed}" / variant
    base.GENE_SPLIT_DIR = CLEAN_ROOT / "data" / cohort / "gene_splits"


def evaluation_decoder(joint_hidden_dim: int) -> type[ConcatenationMLPDecoder]:
    """Bind the frozen joint width for evaluators with a legacy 512 argument."""

    class FixedWidthConcatenationMLPDecoder(ConcatenationMLPDecoder):
        def __init__(
            self,
            spot_dim: int,
            gene_dim: int,
            hidden_dim: int,
            program_dim: int,
            dropout: float,
        ) -> None:
            del hidden_dim
            super().__init__(
                spot_dim=spot_dim,
                gene_dim=gene_dim,
                hidden_dim=joint_hidden_dim,
                program_dim=program_dim,
                dropout=dropout,
            )

    return FixedWidthConcatenationMLPDecoder


def smoke_test() -> None:
    torch.manual_seed(7)
    model = ConcatenationMLPDecoder(32, 24, 28, 8, 0.1)
    output = model(torch.randn(5, 32), torch.randn(11, 24))
    assert output.shape == (5, 11)
    assert bool(torch.isfinite(output).all() and (output > 0).all())
    expected = 32 * 28 + 28 + 24 * 28 + 28 + 1 + 2 * 32 + 2 * 24
    assert sum(parameter.numel() for parameter in model.parameters()) == expected
    print("concatenation-MLP smoke test: PASS")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", choices=COHORTS)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--joint-hidden-dim", type=int, default=704)
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    if args.smoke_test:
        smoke_test()
        return
    if args.cohort is None or args.seed is None or args.variant is None:
        parser.error("--cohort, --seed and --variant are required")

    configure_audit(args.cohort, args.seed, args.variant)
    base.GeneConditionedProgramDecoder = ConcatenationMLPDecoder
    if not args.evaluate_only:
        forwarded = [
            "--epochs", str(args.epochs),
            "--batch-size", "256",
            "--eval-batch-size", "256",
            "--workers", "0",
            "--hidden-dim", str(args.joint_hidden_dim),
            "--program-dim", "96",
            "--train-genes-per-batch", "512",
            "--val-train-genes", "2048",
            "--report-train-genes", "2048",
            "--eval-gene-chunk-size", "512",
            "--lambda-gene-corr", "0.15",
            "--selection-mse-weight", "0.02",
            "--lr", "2.0e-4",
            "--weight-decay", "1.0e-4",
            "--seed", str(args.seed),
            "--variants", args.variant,
            "--paired-rng",
            "--no-train-mean-baseline",
        ]
        sys.argv = [sys.argv[0], *forwarded]
        base.main()

    component_eval.CLEAN_ROOT = EXPERIMENT_ROOT
    component_eval.configure_clean_base = configure_audit
    component_eval.base.GeneConditionedProgramDecoder = evaluation_decoder(args.joint_hidden_dim)
    component_eval.run_one(
        cohort=args.cohort,
        seed=args.seed,
        variant=args.variant,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )


if __name__ == "__main__":
    main()
