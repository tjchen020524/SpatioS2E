"""No-image gene-mean counterfactual for target-disjoint evaluation."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from scipy.sparse.linalg import lsqr


DEFAULT_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1_000.0, 10_000.0)


def _safe_pcc(x: np.ndarray, y: np.ndarray) -> float:
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denominator = np.linalg.norm(x_centered) * np.linalg.norm(y_centered)
    return float(np.dot(x_centered, y_centered) / denominator) if denominator > 0 else float("nan")


@dataclass(frozen=True)
class GeneMeanRidge:
    """Fitted ridge mapping from fixed gene vectors to one mean per gene."""

    coefficient: np.ndarray
    intercept: float
    alpha: float

    def predict(self, vectors: np.ndarray, clip_minimum: float | None = 0.0) -> np.ndarray:
        values = np.asarray(vectors, dtype=np.float64)
        prediction = values @ self.coefficient + self.intercept
        if clip_minimum is not None:
            prediction = np.maximum(prediction, float(clip_minimum))
        return prediction


def _fit_ridge(
    vectors: np.ndarray,
    targets: np.ndarray,
    alpha: float,
    tolerance: float,
) -> GeneMeanRidge:
    x_mean = vectors.mean(axis=0)
    y_mean = float(targets.mean())
    centered_x = vectors - x_mean
    centered_y = targets - y_mean
    coefficient = lsqr(
        centered_x,
        centered_y,
        damp=float(np.sqrt(alpha)),
        atol=tolerance,
        btol=tolerance,
    )[0]
    intercept = y_mean - float(np.dot(x_mean, coefficient))
    return GeneMeanRidge(coefficient=coefficient, intercept=intercept, alpha=float(alpha))


def fit_gene_mean_counterfactual(
    vectors: np.ndarray,
    training_gene_means: np.ndarray,
    training_indices: Sequence[int],
    *,
    alphas: Iterable[float] = DEFAULT_ALPHAS,
    validation_indices: Sequence[int] | None = None,
    validation_seed: int = 42,
    maximum_validation_genes: int = 2_048,
    clip_minimum: float | None = 0.0,
) -> tuple[GeneMeanRidge, list[dict[str, float]]]:
    """Select and fit the manuscript's no-image gene-mean ridge model.

    The outcome must be computed from training biological individuals only.
    When ``validation_indices`` is omitted, a fixed subset is sampled from the
    downstream training genes and excluded during alpha selection.  The final
    model is refitted on every downstream training gene.
    """

    x = np.asarray(vectors, dtype=np.float64)
    y = np.asarray(training_gene_means, dtype=np.float64)
    train = np.asarray(training_indices, dtype=np.int64)
    alpha_values = tuple(float(value) for value in alphas)
    if x.ndim != 2 or y.ndim != 1 or x.shape[0] != y.shape[0]:
        raise ValueError("vectors and training_gene_means must align on the gene axis")
    if train.ndim != 1 or len(train) < 5:
        raise ValueError("training_indices must contain at least five genes")
    if not alpha_values or any(value <= 0 for value in alpha_values):
        raise ValueError("alphas must contain positive values")
    if validation_indices is None:
        validation_count = min(maximum_validation_genes, len(train) // 5)
        rng = np.random.default_rng(validation_seed)
        validation = np.sort(rng.choice(train, size=validation_count, replace=False))
    else:
        validation = np.asarray(validation_indices, dtype=np.int64)
        if not np.isin(validation, train).all():
            raise ValueError("validation_indices must be a subset of training_indices")
    fit = np.setdiff1d(train, validation, assume_unique=False)
    if not len(fit) or not len(validation):
        raise ValueError("both ridge-fitting and validation gene sets must be non-empty")

    diagnostics: list[dict[str, float]] = []
    for alpha in alpha_values:
        model = _fit_ridge(x[fit], y[fit], alpha, tolerance=1.0e-5)
        prediction = model.predict(x[validation], clip_minimum=clip_minimum)
        error = prediction - y[validation]
        diagnostics.append(
            {
                "alpha": alpha,
                "validation_rmse": float(np.sqrt(np.mean(np.square(error)))),
                "validation_pcc": _safe_pcc(prediction, y[validation]),
            }
        )
    selected = min(diagnostics, key=lambda row: (row["validation_rmse"], row["alpha"]))
    final = _fit_ridge(x[train], y[train], selected["alpha"], tolerance=1.0e-6)
    return final, diagnostics


def broadcast_gene_means(predicted_gene_means: np.ndarray, number_of_spots: int) -> np.ndarray:
    """Return a read-only spot-by-gene view with no within-gene variation."""

    means = np.asarray(predicted_gene_means, dtype=np.float64)
    if means.ndim != 1 or number_of_spots < 1:
        raise ValueError("predicted_gene_means must be one-dimensional and number_of_spots positive")
    return np.broadcast_to(means[None, :], (int(number_of_spots), len(means)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="NPZ with vectors, gene_means, training_indices and heldout_indices")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-seed", type=int, default=42)
    parser.add_argument("--no-clip", action="store_true")
    args = parser.parse_args()
    data = np.load(args.input, allow_pickle=False)
    clip = None if args.no_clip else 0.0
    model, diagnostics = fit_gene_mean_counterfactual(
        data["vectors"],
        data["gene_means"],
        data["training_indices"],
        validation_seed=args.validation_seed,
        clip_minimum=clip,
    )
    heldout = np.asarray(data["heldout_indices"], dtype=np.int64)
    prediction = model.predict(data["vectors"][heldout], clip_minimum=clip)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        heldout_indices=heldout,
        predicted_gene_means=prediction,
        coefficient=model.coefficient,
        intercept=np.asarray(model.intercept),
        selected_alpha=np.asarray(model.alpha),
    )
    args.output.with_suffix(".validation.json").write_text(json.dumps(diagnostics, indent=2) + "\n")


if __name__ == "__main__":
    main()
