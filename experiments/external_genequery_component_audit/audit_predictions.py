#!/usr/bin/env python3
"""Apply the manuscript component decomposition to a model-agnostic prediction file."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.heldout_metric_policy import gene_correlations_from_moments  # noqa: E402


def finite_corr(x: np.ndarray, y: np.ndarray, *, std_min: float = 1.0e-6) -> float:
    """Return PCC, or NaN when either vector has negligible variation."""
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 2:
        return float("nan")
    xc = x[mask] - x[mask].mean()
    yc = y[mask] - y[mask].mean()
    x_ss = float(np.dot(xc, xc))
    y_ss = float(np.dot(yc, yc))
    if min(x_ss, y_ss) <= int(mask.sum()) * std_min**2:
        return float("nan")
    denominator = math.sqrt(x_ss * y_ss)
    return float(np.clip(np.dot(xc, yc) / denominator, -1.0, 1.0))


def matrix_metrics(pred: np.ndarray, true: np.ndarray) -> Dict[str, object]:
    if pred.shape != true.shape or pred.ndim != 2:
        raise ValueError("Expected matched spot-by-gene matrices; got %s and %s" % (pred.shape, true.shape))
    finite = np.isfinite(pred) & np.isfinite(true)
    if not finite.all():
        raise ValueError("Prediction matrices contain non-finite values")
    count = np.repeat(pred.shape[0], pred.shape[1]).astype(np.float64)
    sum_pred = pred.sum(axis=0, dtype=np.float64)
    sum_true = true.sum(axis=0, dtype=np.float64)
    sum_pred2 = np.square(pred, dtype=np.float64).sum(axis=0)
    sum_true2 = np.square(true, dtype=np.float64).sum(axis=0)
    sum_cross = (pred.astype(np.float64) * true.astype(np.float64)).sum(axis=0)
    squared_error = np.square(pred.astype(np.float64) - true.astype(np.float64))
    gene_sse = squared_error.sum(axis=0)
    mean_pred = sum_pred / count
    mean_true = sum_true / count
    abundance_gene_sse = count * np.square(mean_pred - mean_true)
    centered_gene_sse = np.maximum(gene_sse - abundance_gene_sse, 0.0)
    policy = gene_correlations_from_moments(
        count, sum_pred, sum_true, sum_pred2, sum_true2, sum_cross
    )
    centered_pcc = policy.centered_full_matrix_pcc
    centered_count = float(count[policy.eligible].sum())
    if centered_count == 0 or min(
        float(policy.centered_pred_ss[policy.eligible].sum()),
        float(policy.centered_true_ss[policy.eligible].sum()),
    ) <= centered_count * 1.0e-12:
        centered_pcc = float("nan")
    total_count = int(pred.size)
    total_sse = float(gene_sse.sum())
    abundance_sse = float(abundance_gene_sse.sum())
    centered_sse = float(centered_gene_sse.sum())
    return {
        "n_spots": int(pred.shape[0]),
        "n_genes": int(pred.shape[1]),
        "n_eligible_gene_pcc": policy.n_eligible,
        "n_constant_prediction_gene_pcc": policy.n_constant_prediction,
        "full_matrix_pcc": finite_corr(pred, true),
        "abundance_pcc": finite_corr(mean_pred, mean_true),
        "abundance_rmse": float(np.sqrt(np.mean(np.square(mean_pred - mean_true)))),
        "mean_gene_pcc": policy.mean,
        "median_gene_pcc": policy.median,
        "centered_full_matrix_pcc": centered_pcc,
        "overall_mse": total_sse / total_count,
        "abundance_mse": abundance_sse / total_count,
        "centered_mse": centered_sse / total_count,
        "centered_rmse": math.sqrt(max(centered_sse / total_count, 0.0)),
        "decomposition_error": total_sse / total_count - abundance_sse / total_count - centered_sse / total_count,
        "mean_pred": mean_pred,
        "mean_true": mean_true,
        "gene_mse": gene_sse / count,
        "gene_abundance_mse": abundance_gene_sse / count,
        "gene_centered_mse": centered_gene_sse / count,
        "gene_pcc": policy.correlation,
        "gene_pcc_eligible": policy.eligible,
        "observed_std": policy.observed_std,
        "predicted_std": policy.predicted_std,
    }


def scalar_metrics(metrics: Dict[str, object]) -> Dict[str, object]:
    return {
        key: value
        for key, value in metrics.items()
        if not isinstance(value, np.ndarray)
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prediction", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    with np.load(args.prediction, allow_pickle=False) as data:
        pred = data["pred"].astype(np.float64)
        true = data["true"].astype(np.float64)
        gene_ids = data["gene_ids"].astype(str)
        symbols = data["symbols"].astype(str)
        samples = data["sample_ids"].astype(str)
        individuals = data["individual_ids"].astype(str)
    if len(samples) != len(pred) or len(individuals) != len(pred):
        raise ValueError("Spot metadata does not match prediction rows")
    output_dir = args.output_dir or args.prediction.parent / "audit"
    output_dir.mkdir(parents=True, exist_ok=True)

    overall = matrix_metrics(pred, true)
    per_gene = pd.DataFrame(
        {
            "gene_id": gene_ids,
            "symbol": symbols,
            "mean_pred": overall["mean_pred"],
            "mean_true": overall["mean_true"],
            "gene_mse": overall["gene_mse"],
            "gene_abundance_mse": overall["gene_abundance_mse"],
            "gene_centered_mse": overall["gene_centered_mse"],
            "gene_pcc": overall["gene_pcc"],
            "gene_pcc_eligible": overall["gene_pcc_eligible"],
            "observed_std": overall["observed_std"],
            "predicted_std": overall["predicted_std"],
        }
    )
    per_gene.to_csv(output_dir / "per_gene.tsv", sep="\t", index=False)

    section_rows = []
    centered_pred_blocks = []
    centered_true_blocks = []
    for sample in sorted(set(samples.tolist())):
        mask = samples == sample
        if len(set(individuals[mask].tolist())) != 1:
            raise ValueError("Section %s maps to multiple individuals" % sample)
        section = matrix_metrics(pred[mask], true[mask])
        row = scalar_metrics(section)
        row.update({"sample": sample, "individual": individuals[mask][0]})
        section_rows.append(row)
        centered_pred_blocks.append(pred[mask] - pred[mask].mean(axis=0, keepdims=True))
        centered_true_blocks.append(true[mask] - true[mask].mean(axis=0, keepdims=True))
    section_frame = pd.DataFrame(section_rows)
    section_frame.to_csv(output_dir / "per_section.tsv", sep="\t", index=False)

    numeric = [
        "full_matrix_pcc",
        "abundance_pcc",
        "abundance_rmse",
        "mean_gene_pcc",
        "centered_full_matrix_pcc",
        "overall_mse",
        "abundance_mse",
        "centered_mse",
        "centered_rmse",
    ]
    individual_rows = []
    for individual, group in section_frame.groupby("individual", sort=True):
        row = {"individual": individual, "n_sections": int(len(group))}
        for column in numeric:
            row[column] = float(group[column].mean())
        individual_rows.append(row)
    individual_frame = pd.DataFrame(individual_rows)
    individual_frame.to_csv(output_dir / "per_individual.tsv", sep="\t", index=False)

    section_centered = matrix_metrics(
        np.concatenate(centered_pred_blocks), np.concatenate(centered_true_blocks)
    )
    summary = scalar_metrics(overall)
    summary.update(
        {
            "prediction_file": str(args.prediction),
            "n_sections": int(len(section_frame)),
            "n_individuals": int(len(individual_frame)),
            "section_centered_full_matrix_pcc": float(section_centered["full_matrix_pcc"]),
            "section_centered_mean_gene_pcc": float(section_centered["mean_gene_pcc"]),
            "section_centered_mse": float(section_centered["overall_mse"]),
        }
    )
    for column in numeric:
        summary["individual_equal_%s" % column] = float(individual_frame[column].mean())
    if abs(float(summary["decomposition_error"])) > 1.0e-10:
        raise AssertionError("MSE decomposition failed: %s" % summary["decomposition_error"])
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
