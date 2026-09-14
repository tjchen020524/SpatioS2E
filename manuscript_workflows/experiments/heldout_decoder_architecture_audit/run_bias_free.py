#!/usr/bin/env python3
"""Train a held-out-gene bilinear decoder without a gene-conditioned bias.

The only intercept is one scalar shared by every spot and gene.  This tests
whether the explicit gene-bias path is necessary for the abundance-dominant
effect seen in the primary held-out-gene assay.  It does not prevent the
bilinear interaction from representing gene means implicitly.
"""

from __future__ import annotations
from workflow_paths import DATA_ROOT

import argparse
import math
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
from experiments.multicohort_geneheldout_decima.run_cohort_geneheldout import (
    configure_base,
)
from experiments.multicohort_geneheldout_decima_clean_split import (
    evaluate_components as component_eval,
)
from experiments.multicohort_geneheldout_decima_clean_split.clean_split_config import (
    CLEAN_ROOT,
    COHORTS,
    SEEDS,
    VARIANTS,
)


ROOT = DATA_ROOT
AUDIT_ROOT = ROOT / "experiments/heldout_decoder_architecture_audit"
EXPERIMENT_ROOT = AUDIT_ROOT / "bias_free"


class BiasFreeProgramDecoder(torch.nn.Module):
    """Bilinear decoder with a globally shared intercept and no gene bias."""

    def __init__(
        self,
        spot_dim: int,
        gene_dim: int,
        hidden_dim: int,
        program_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.spot_encoder = torch.nn.Sequential(
            torch.nn.LayerNorm(spot_dim),
            torch.nn.Linear(spot_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, program_dim),
        )
        self.gene_encoder = torch.nn.Sequential(
            torch.nn.LayerNorm(gene_dim),
            torch.nn.Linear(gene_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, program_dim),
        )
        self.global_intercept = torch.nn.Parameter(torch.zeros(()))
        self.program_dim = program_dim

    def forward(self, x: torch.Tensor, gene_emb: torch.Tensor) -> torch.Tensor:
        spot_program = self.spot_encoder(x)
        gene_program = self.gene_encoder(gene_emb)
        interaction = spot_program @ gene_program.transpose(0, 1)
        interaction = interaction / math.sqrt(float(self.program_dim))
        return F.softplus(interaction + self.global_intercept)


def configure_audit(cohort: str, seed: int, variant: str) -> None:
    configure_base(cohort, seed, variant)
    base.EXP_ROOT = EXPERIMENT_ROOT / "runs" / cohort / f"seed_{seed}" / variant
    base.GENE_SPLIT_DIR = CLEAN_ROOT / "data" / cohort / "gene_splits"


def smoke_test() -> None:
    torch.manual_seed(7)
    model = BiasFreeProgramDecoder(32, 24, 16, 8, 0.1)
    output = model(torch.randn(5, 32), torch.randn(11, 24))
    assert output.shape == (5, 11)
    assert set(model.state_dict()) == {
        "global_intercept",
        "spot_encoder.0.weight",
        "spot_encoder.0.bias",
        "spot_encoder.1.weight",
        "spot_encoder.1.bias",
        "spot_encoder.4.weight",
        "spot_encoder.4.bias",
        "gene_encoder.0.weight",
        "gene_encoder.0.bias",
        "gene_encoder.1.weight",
        "gene_encoder.1.bias",
        "gene_encoder.4.weight",
        "gene_encoder.4.bias",
    }
    print("bias-free model smoke test: PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", choices=COHORTS)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    if args.smoke_test:
        smoke_test()
        return
    if args.cohort is None or args.seed is None or args.variant is None:
        parser.error("--cohort, --seed and --variant are required")

    configure_audit(args.cohort, args.seed, args.variant)
    base.GeneConditionedProgramDecoder = BiasFreeProgramDecoder
    if not args.evaluate_only:
        forwarded = [
            "--epochs", str(args.epochs),
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

    # Reuse the frozen component evaluator with the audited model and paths.
    component_eval.CLEAN_ROOT = EXPERIMENT_ROOT
    component_eval.configure_clean_base = configure_audit
    component_eval.base.GeneConditionedProgramDecoder = BiasFreeProgramDecoder
    component_eval.run_one(
        cohort=args.cohort,
        seed=args.seed,
        variant=args.variant,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )


if __name__ == "__main__":
    main()
