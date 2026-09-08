#!/usr/bin/env python3
"""Fit an image-free gene-mean ridge baseline using the same query vectors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from common import (
    PANEL_ARTIFACT,
    RUN_ROOT,
    counterfactual_embeddings,
    load_expression_partition,
    load_manifest,
    load_panel_artifact,
)


ALPHAS = np.asarray([0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0])


def ridge_path(
    train_x: np.ndarray,
    train_y: np.ndarray,
    query_x: np.ndarray,
    alphas: np.ndarray,
) -> dict:
    x = train_x.astype(np.float64)
    y = train_y.astype(np.float64)
    q = query_x.astype(np.float64)
    x_mean = x.mean(axis=0)
    y_mean = float(y.mean())
    xc = x - x_mean
    yc = y - y_mean
    u, singular, vt = np.linalg.svd(xc, full_matrices=False)
    projected_y = u.T @ yc
    predictions = {}
    for alpha in alphas:
        weights = vt.T @ ((singular / (np.square(singular) + float(alpha))) * projected_y)
        predictions[float(alpha)] = ((q - x_mean) @ weights + y_mean).astype(np.float32)
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        choices=("semantic", "identity_shuffle", "random", "constant"),
        default="semantic",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--panel-artifact", type=Path, default=PANEL_ARTIFACT)
    parser.add_argument("--output-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--max-spots", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    panel = load_panel_artifact(args.panel_artifact)
    labels = panel["split_labels"].astype(str)
    train_mask = labels == "train"
    heldout_mask = labels == "heldout"
    gene_ids = panel["gene_ids"].astype(str)
    embeddings = counterfactual_embeddings(
        panel["semantic_embeddings"], labels, args.variant, args.seed
    )
    train_y, _, _ = load_expression_partition("train", gene_ids, args.max_spots)
    test_y, samples, barcodes = load_expression_partition("test", gene_ids, args.max_spots)
    train_means = train_y[:, train_mask].mean(axis=0)
    train_embeddings = embeddings[train_mask].astype(np.float64)
    feature_mean = train_embeddings.mean(axis=0, keepdims=True)
    feature_std = train_embeddings.std(axis=0, keepdims=True)
    feature_std = np.where(feature_std > 1.0e-8, feature_std, 1.0)
    standardized = (embeddings.astype(np.float64) - feature_mean) / feature_std
    n_train_genes = int(train_mask.sum())
    rng = np.random.default_rng(42)
    validation_local = np.sort(
        rng.choice(n_train_genes, size=n_train_genes // 5, replace=False)
    )
    fit_local = np.setdiff1d(np.arange(n_train_genes), validation_local)
    validation_path = ridge_path(
        standardized[train_mask][fit_local],
        train_means[fit_local],
        standardized[train_mask][validation_local],
        ALPHAS,
    )
    scores = {}
    for alpha in ALPHAS:
        fitted = np.maximum(validation_path[float(alpha)], 0.0)
        scores[str(float(alpha))] = float(
            np.mean(np.square(fitted - train_means[validation_local]))
        )
    best_alpha = min(ALPHAS.tolist(), key=lambda alpha: scores[str(float(alpha))])
    heldout_means = ridge_path(
        standardized[train_mask],
        train_means,
        standardized[heldout_mask],
        np.asarray([best_alpha]),
    )[float(best_alpha)]
    heldout_means = np.maximum(heldout_means, 0.0)
    prediction = np.repeat(heldout_means[None, :], len(test_y), axis=0)
    patient_map = load_manifest().set_index("sample")["patient"].astype(str).to_dict()
    individuals = np.asarray([patient_map[sample] for sample in samples])
    run_dir = args.output_root / ("seed_%d" % args.seed) / (args.variant + "_mean_only")
    output = run_dir / "predictions.npz"
    if output.exists() and not args.overwrite:
        raise FileExistsError(output)
    run_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        pred=prediction.astype(np.float32),
        true=test_y[:, heldout_mask].astype(np.float32),
        gene_ids=gene_ids[heldout_mask],
        symbols=panel["symbols"].astype(str)[heldout_mask],
        sample_ids=samples,
        individual_ids=individuals,
        barcodes=barcodes,
    )
    metadata = {
        "variant": args.variant + "_mean_only",
        "seed": args.seed,
        "selected_alpha": float(best_alpha),
        "internal_gene_validation_mse_by_alpha": scores,
        "ridge_feature_scaling": "featurewise mean/std fitted on downstream training genes",
        "ridge_validation": "fixed seed-42 20% internal holdout from downstream training genes",
        "nonnegative_prediction_clip": True,
        "target_scale": "log1p_counts_per_million",
        "panel_artifact": str(args.panel_artifact),
        "n_train_genes": int(train_mask.sum()),
        "n_heldout_genes": int(heldout_mask.sum()),
    }
    (run_dir / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
