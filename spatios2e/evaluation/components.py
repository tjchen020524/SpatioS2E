"""Component-resolved endpoints for spatial-expression matrices.

Matrices are expected in ``[spot, gene]`` order. Gene means describe the
across-gene ordering of mean normalized log-expression; subtracting those
means isolates within-gene variation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Sequence

import numpy as np


def _matrix(values: np.ndarray | Sequence[Sequence[float]], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional [spot, gene] matrix")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def pearson_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation, returning NaN when either input is constant."""

    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.shape != y.shape:
        raise ValueError("correlation inputs must have the same shape")
    if x.size < 2:
        return float("nan")
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denominator = np.sqrt(np.sum(x_centered**2) * np.sum(y_centered**2))
    if denominator <= 0:
        return float("nan")
    return float(np.sum(x_centered * y_centered) / denominator)


def center_within_gene(matrix: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
    """Subtract each gene's mean over the supplied spots."""

    values = _matrix(matrix, "matrix")
    return values - values.mean(axis=0, keepdims=True)


def center_within_section(
    matrix: np.ndarray | Sequence[Sequence[float]],
    section_ids: Sequence[object],
) -> np.ndarray:
    """Subtract gene means independently within every tissue section."""

    values = _matrix(matrix, "matrix")
    labels = np.asarray(section_ids)
    if labels.ndim != 1 or labels.shape[0] != values.shape[0]:
        raise ValueError("section_ids must contain one label per matrix row")
    centered = np.empty_like(values)
    for section in np.unique(labels):
        mask = labels == section
        centered[mask] = values[mask] - values[mask].mean(axis=0, keepdims=True)
    return centered


def component_metrics(
    observed: np.ndarray | Sequence[Sequence[float]],
    predicted: np.ndarray | Sequence[Sequence[float]],
    *,
    observed_std_min: float = 1.0e-6,
    predicted_std_min: float = 1.0e-6,
) -> Dict[str, float | int]:
    """Compute full-matrix, gene-mean and within-gene endpoints.

    The returned MSE terms obey the exact identity

    ``full_matrix_mse = gene_mean_mse + centered_mse``

    up to floating-point error for a rectangular finite matrix.
    """

    true = _matrix(observed, "observed")
    pred = _matrix(predicted, "predicted")
    if true.shape != pred.shape:
        raise ValueError("observed and predicted matrices must have the same shape")
    if true.shape[0] < 1 or true.shape[1] < 1:
        raise ValueError("observed and predicted matrices must be non-empty")

    mean_true = true.mean(axis=0)
    mean_pred = pred.mean(axis=0)
    centered_true = true - mean_true[None, :]
    centered_pred = pred - mean_pred[None, :]

    squared_error = (pred - true) ** 2
    centered_squared_error = (centered_pred - centered_true) ** 2
    gene_mean_squared_error = (mean_pred - mean_true) ** 2

    observed_std = true.std(axis=0)
    predicted_std = pred.std(axis=0)
    eligible = observed_std > observed_std_min
    prediction_variable = predicted_std > predicted_std_min
    gene_correlations = np.full(true.shape[1], np.nan, dtype=np.float64)
    gene_correlations[eligible] = 0.0
    for gene in np.flatnonzero(eligible & prediction_variable):
        gene_correlations[gene] = pearson_correlation(true[:, gene], pred[:, gene])
    eligible_gene_correlations = gene_correlations[eligible]
    if not eligible.any():
        centered_full_matrix_pcc = float("nan")
    elif prediction_variable[eligible].any():
        centered_full_matrix_pcc = pearson_correlation(
            centered_true[:, eligible], centered_pred[:, eligible]
        )
    else:
        centered_full_matrix_pcc = 0.0

    full_matrix_mse = float(squared_error.mean())
    gene_mean_mse = float(gene_mean_squared_error.mean())
    centered_mse = float(centered_squared_error.mean())
    gene_mean_pcc = pearson_correlation(mean_true, mean_pred)
    gene_mean_rmse = float(np.sqrt(gene_mean_mse))
    return {
        "n_spots": int(true.shape[0]),
        "n_genes": int(true.shape[1]),
        "full_matrix_pcc": pearson_correlation(true, pred),
        "full_matrix_mse": full_matrix_mse,
        "gene_mean_pcc": gene_mean_pcc,
        "gene_mean_rmse": gene_mean_rmse,
        # Compatibility aliases used by v0.1 and the frozen analysis scripts.
        "abundance_pcc": gene_mean_pcc,
        "abundance_rmse": gene_mean_rmse,
        "mean_gene_pcc": (
            float(eligible_gene_correlations.mean())
            if eligible_gene_correlations.size
            else float("nan")
        ),
        "median_gene_pcc": (
            float(np.median(eligible_gene_correlations))
            if eligible_gene_correlations.size
            else float("nan")
        ),
        "q25_gene_pcc": (
            float(np.quantile(eligible_gene_correlations, 0.25))
            if eligible_gene_correlations.size
            else float("nan")
        ),
        "q75_gene_pcc": (
            float(np.quantile(eligible_gene_correlations, 0.75))
            if eligible_gene_correlations.size
            else float("nan")
        ),
        "n_eligible_gene_pcc": int(eligible.sum()),
        "n_finite_gene_pcc": int(eligible.sum()),
        "n_constant_prediction_gene_pcc": int((eligible & ~prediction_variable).sum()),
        "centered_full_matrix_pcc": centered_full_matrix_pcc,
        "centered_rmse": float(np.sqrt(centered_mse)),
        "gene_mean_mse": gene_mean_mse,
        "centered_mse": centered_mse,
        "decomposition_error": full_matrix_mse - gene_mean_mse - centered_mse,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="NPZ containing observed and predicted [spot, gene] arrays")
    parser.add_argument("--observed-key", default="observed")
    parser.add_argument("--predicted-key", default="predicted")
    parser.add_argument("--section-key", default=None, help="Optionally centre both matrices within section first")
    parser.add_argument("--output", type=Path, default=None, help="JSON output path; stdout if omitted")
    args = parser.parse_args()

    arrays = np.load(args.input, allow_pickle=False)
    observed = arrays[args.observed_key]
    predicted = arrays[args.predicted_key]
    if args.section_key:
        section_ids = arrays[args.section_key]
        observed = center_within_section(observed, section_ids)
        predicted = center_within_section(predicted, section_ids)
    result = component_metrics(observed, predicted)
    payload = json.dumps(result, indent=2, allow_nan=True) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)


if __name__ == "__main__":
    main()
