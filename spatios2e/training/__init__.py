"""Training entry points for fitted- and held-out-target assays."""

from spatios2e.training.heldout import (
    HeldOutTrainingConfig,
    HeldOutTrainingResult,
    evaluate_heldout_decoder,
    fit_heldout_decoder,
    gene_correlation_loss,
)

__all__ = [
    "HeldOutTrainingConfig",
    "HeldOutTrainingResult",
    "evaluate_heldout_decoder",
    "fit_heldout_decoder",
    "gene_correlation_loss",
]
