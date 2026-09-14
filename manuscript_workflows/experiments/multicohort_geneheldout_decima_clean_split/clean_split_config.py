"""Shared paths and base-module configuration for the clean-split replication."""

from __future__ import annotations
from workflow_paths import DATA_ROOT

from pathlib import Path

from experiments.multicohort_geneheldout_decima.run_cohort_geneheldout import configure_base
from experiments.geneheldout_uni2h_decima_decoder_noseq_controls import (
    run_geneheldout_noseq_controls as base,
)


ROOT = DATA_ROOT
CLEAN_ROOT = ROOT / "experiments/multicohort_geneheldout_decima_clean_split"
SOURCE_ROOT = ROOT / "experiments/multicohort_geneheldout_decima"
COHORTS = ("hippocampus_donor_disjoint", "dlpfc", "nac", "her2st")
SEEDS = (42, 123, 456)
VARIANTS = ("decima", "random", "constant")


def configure_clean_base(cohort: str, seed: int, variant: str) -> None:
    configure_base(cohort, seed, variant)
    base.EXP_ROOT = CLEAN_ROOT / "runs" / cohort / f"seed_{seed}" / variant
    base.GENE_SPLIT_DIR = CLEAN_ROOT / "data" / cohort / "gene_splits"
