"""Evaluation entry points for SpatioS2E."""

from spatios2e.evaluation.components import (
    center_within_gene,
    center_within_section,
    component_metrics,
    pearson_correlation,
)
from spatios2e.evaluation.gene_mean import (
    GeneMeanRidge,
    broadcast_gene_means,
    fit_gene_mean_counterfactual,
)

__all__ = [
    "center_within_gene",
    "center_within_section",
    "component_metrics",
    "pearson_correlation",
    "GeneMeanRidge",
    "broadcast_gene_means",
    "fit_gene_mean_counterfactual",
]
