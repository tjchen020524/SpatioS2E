#!/usr/bin/env python3
"""Export component-resolved metrics for one clean-split checkpoint."""

from __future__ import annotations
from workflow_paths import DATA_ROOT

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from experiments.heldout_metric_policy import (
    OBSERVED_STD_MIN,
    PREDICTED_STD_MIN,
    gene_correlations_from_moments,
)
from experiments.geneheldout_uni2h_decima_decoder_noseq_controls import (
    run_geneheldout_noseq_controls as base,
)
from experiments.multicohort_geneheldout_decima_clean_split.clean_split_config import (
    CLEAN_ROOT,
    COHORTS,
    ROOT,
    SEEDS,
    VARIANTS,
    configure_clean_base,
)


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Return PCC only when both across-gene quantities have real variance.

    In particular, a constant gene vector produces one shared prediction map
    and therefore one shared predicted gene mean.  Chunk-level floating-point
    noise must not turn that undefined across-gene correlation into a number.
    """
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 2:
        return float("nan")
    x_raw = np.asarray(x[mask], dtype=np.float64)
    y_raw = np.asarray(y[mask], dtype=np.float64)
    if float(np.std(x_raw)) <= PREDICTED_STD_MIN:
        return float("nan")
    if float(np.std(y_raw)) <= OBSERVED_STD_MIN:
        return float("nan")
    x_use = x_raw - np.mean(x_raw)
    y_use = y_raw - np.mean(y_raw)
    denominator = math.sqrt(float(np.dot(x_use, x_use) * np.dot(y_use, y_use)))
    return float(np.dot(x_use, y_use) / denominator) if denominator > 0.0 else float("nan")


def embedding_matrix(variant: str, gene_ids: list[str], train_idx: np.ndarray, seed: int) -> np.ndarray:
    decima = base.load_decima_embeddings(gene_ids, train_idx)
    if variant == "decima":
        return decima
    if variant == "random":
        return base.load_random_embeddings(decima.shape, seed + 12345)
    return base.load_constant_embeddings(decima.shape)


def run_one(cohort: str, seed: int, variant: str, device: torch.device) -> dict[str, object]:
    configure_clean_base(cohort, seed, variant)
    base.set_seed(seed)
    manifest = pd.read_csv(base.DATA_ROOT / "manifest.tsv", sep="\t")
    gene_ids = base.load_gene_ids()
    train_idx, heldout_idx = base.load_gene_split(gene_ids)
    embeddings = embedding_matrix(variant, gene_ids, train_idx, seed)
    feature_store = base.FeatureStore()
    expression_store = base.ExpressionStore(gene_ids)
    test_dataset = base.SpotDataset(
        manifest.loc[manifest["split"] == "test"].copy(), feature_store, expression_store
    )
    loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=512,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
        collate_fn=base.collate,
    )
    model = base.GeneConditionedProgramDecoder(
        spot_dim=int(test_dataset[0]["x"].shape[0]),
        gene_dim=int(embeddings.shape[1]),
        hidden_dim=512,
        program_dim=96,
        dropout=0.10,
    ).to(device)
    checkpoint = base.EXP_ROOT / variant / "checkpoints/best.pt"
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()
    gene_tensor = base.numpy_to_tensor(embeddings).to(device)

    n_genes = len(heldout_idx)
    count = np.zeros(n_genes, dtype=np.int64)
    sum_true = np.zeros(n_genes, dtype=np.float64)
    sum_pred = np.zeros(n_genes, dtype=np.float64)
    sum_true2 = np.zeros(n_genes, dtype=np.float64)
    sum_pred2 = np.zeros(n_genes, dtype=np.float64)
    sum_cross = np.zeros(n_genes, dtype=np.float64)
    sum_sq_error = np.zeros(n_genes, dtype=np.float64)
    chunks = base.make_gene_chunks(n_genes, 512)
    with torch.no_grad():
        for batch_number, batch in enumerate(loader, start=1):
            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            for local_idx in chunks:
                full_idx_np = heldout_idx[local_idx]
                full_idx = torch.tensor(full_idx_np, dtype=torch.long, device=device)
                pred = base.tensor_to_numpy(model(x, gene_tensor.index_select(0, full_idx))).astype(np.float64)
                true = base.tensor_to_numpy(y.index_select(1, full_idx)).astype(np.float64)
                mask = np.isfinite(pred) & np.isfinite(true)
                pred = np.where(mask, pred, 0.0)
                true = np.where(mask, true, 0.0)
                count[local_idx] += mask.sum(axis=0).astype(np.int64)
                sum_true[local_idx] += true.sum(axis=0)
                sum_pred[local_idx] += pred.sum(axis=0)
                sum_true2[local_idx] += np.square(true).sum(axis=0)
                sum_pred2[local_idx] += np.square(pred).sum(axis=0)
                sum_cross[local_idx] += (pred * true).sum(axis=0)
                sum_sq_error[local_idx] += np.square(pred - true).sum(axis=0)
            if batch_number % 10 == 0 or batch_number == len(loader):
                print(f"[clean components] {cohort} seed={seed} {variant} {batch_number}/{len(loader)}", flush=True)

    valid = count > 0
    mean_true = np.full(n_genes, np.nan)
    mean_pred = np.full(n_genes, np.nan)
    mean_true[valid] = sum_true[valid] / count[valid]
    mean_pred[valid] = sum_pred[valid] / count[valid]
    mean_error = mean_pred - mean_true
    total_gene_mse = np.full(n_genes, np.nan)
    total_gene_mse[valid] = sum_sq_error[valid] / count[valid]
    centered_gene_mse = np.maximum(total_gene_mse - np.square(mean_error), 0.0)
    policy = gene_correlations_from_moments(
        count,
        sum_pred,
        sum_true,
        sum_pred2,
        sum_true2,
        sum_cross,
    )
    gene_pcc = policy.correlation

    output = CLEAN_ROOT / "components/runs" / cohort / f"seed_{seed}" / variant
    output.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(
        {
            "cohort": cohort,
            "seed": seed,
            "variant": variant,
            "gene_id": [gene_ids[int(index)] for index in heldout_idx],
            "n_observations": count,
            "observed_mean": mean_true,
            "predicted_mean": mean_pred,
            "mean_error": mean_error,
            "mse": total_gene_mse,
            "centered_mse": centered_gene_mse,
            "gene_pcc": gene_pcc,
            "gene_pcc_eligible": policy.eligible,
            "prediction_variable": policy.prediction_variable,
            "observed_std": policy.observed_std,
            "predicted_std": policy.predicted_std,
        }
    )
    table.to_csv(
        output / "gene_components.tsv.gz",
        sep="\t",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )

    total_count = float(count.sum())
    overall_mse = float(sum_sq_error.sum() / total_count)
    abundance_mse = float(np.nansum(np.square(mean_error) * count) / total_count)
    centered_mse = float(np.nansum(centered_gene_mse * count) / total_count)
    pooled_mean_true = float(sum_true.sum() / total_count)
    pooled_mean_pred = float(sum_pred.sum() / total_count)
    pooled_var_true = float(sum_true2.sum() / total_count - pooled_mean_true**2)
    pooled_var_pred = float(sum_pred2.sum() / total_count - pooled_mean_pred**2)
    pooled_cov = float(sum_cross.sum() / total_count - pooled_mean_true * pooled_mean_pred)
    summary: dict[str, object] = {
        "cohort": cohort,
        "seed": seed,
        "variant": variant,
        "n_heldout_genes": n_genes,
        "n_eligible_gene_pcc": policy.n_eligible,
        "n_finite_gene_pcc": policy.n_eligible,
        "n_constant_prediction_gene_pcc": policy.n_constant_prediction,
        "observed_std_min": OBSERVED_STD_MIN,
        "predicted_std_min": PREDICTED_STD_MIN,
        "abundance_pcc": safe_corr(mean_pred, mean_true),
        "abundance_rmse": math.sqrt(max(abundance_mse, 0.0)),
        "abundance_mse": abundance_mse,
        "mean_gene_pcc": policy.mean,
        "median_gene_pcc": policy.median,
        "q25_gene_pcc": policy.q25,
        "q75_gene_pcc": policy.q75,
        "centered_full_matrix_pcc": policy.centered_full_matrix_pcc,
        "centered_rmse": math.sqrt(max(centered_mse, 0.0)),
        "centered_mse": centered_mse,
        "pooled_pcc": pooled_cov / math.sqrt(pooled_var_true * pooled_var_pred),
        "full_matrix_pcc": pooled_cov / math.sqrt(pooled_var_true * pooled_var_pred),
        "overall_mse": overall_mse,
        "decomposition_error": overall_mse - abundance_mse - centered_mse,
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "gene_table": str((output / "gene_components.tsv.gz").relative_to(ROOT)),
    }
    if abs(float(summary["decomposition_error"])) > 1.0e-9:
        raise AssertionError(summary["decomposition_error"])
    stored = json.loads((base.EXP_ROOT / variant / "results/heldout_test_overall.json").read_text())
    for current, expected, metric in (
        (summary["overall_mse"], stored["mse"], "MSE"),
        (summary["pooled_pcc"], stored["corr"], "pooled PCC"),
    ):
        if not np.isclose(float(current), float(expected), rtol=0.0, atol=3.0e-6):
            raise ValueError(f"Stored {metric} mismatch: {current} != {expected}")
    summary["legacy_finite_only_mean_gene_pcc"] = float(stored["gene_corr_mean"])
    summary["common_denominator_minus_legacy_mean_gene_pcc"] = float(
        summary["mean_gene_pcc"] - stored["gene_corr_mean"]
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", required=True, choices=COHORTS)
    parser.add_argument("--seed", required=True, type=int, choices=SEEDS)
    parser.add_argument("--variant", required=True, choices=VARIANTS)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)
    print(json.dumps(run_one(args.cohort, args.seed, args.variant, device), indent=2))


if __name__ == "__main__":
    main()
