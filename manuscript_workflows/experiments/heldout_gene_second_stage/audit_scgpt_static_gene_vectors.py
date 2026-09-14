#!/usr/bin/env python3
"""Audit static scGPT gene-token vectors in the no-image target assay.

This is an independent no-image control for a second pretrained gene representation. It
uses the frozen, layer-normalized gene-token embeddings from the local scGPT
brain checkpoint, not donor-specific cell embeddings or expression-weighted
single-cell context.  Genes absent from the pretrained vocabulary are excluded
from every condition before fitting and evaluation.  Correct, random and
identity-shuffled vectors use identical downstream splits and ridge selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.append(str(REPOSITORY_ROOT))

from experiments.heldout_gene_second_stage import build_gene_mean_baselines as mean_only
from experiments.multicohort_geneheldout_decima_clean_split.clean_split_config import (
    COHORTS,
    ROOT,
    SEEDS,
)


OUT = ROOT / "experiments/heldout_gene_second_stage/scgpt_static_gene_vectors"
MODEL_DIR = ROOT / "scGPT/brain_model"
CHECKPOINTS = {
    "brain": (
        ROOT / "scGPT/brain_model",
        ROOT / "experiments/heldout_gene_second_stage/scgpt_static_gene_vectors",
    ),
    "whole_human": (
        ROOT / "scGPT/whole_human_model",
        ROOT
        / "experiments/heldout_gene_second_stage/scgpt_whole_human_static_gene_vectors",
    ),
}
GENE_MAP = ROOT / "data/processed/gene_mapping/gene_intersection.tsv"
CONDITIONS = ("scgpt_gene_token", "random", "identity_shuffled_scgpt")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def token_feature_lookup(
    model_dir=None,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    model_dir = MODEL_DIR if model_dir is None else model_dir
    vocab = json.loads((model_dir / "vocab.json").read_text())
    state = torch.load(model_dir / "best_model.pt", map_location="cpu")
    embedding = state["encoder.embedding.weight"].detach().cpu().numpy().astype(
        np.float32, copy=False
    )
    weight = state["encoder.enc_norm.weight"].detach().cpu().numpy().astype(
        np.float32, copy=False
    )
    bias = state["encoder.enc_norm.bias"].detach().cpu().numpy().astype(
        np.float32, copy=False
    )
    row_mean = embedding.mean(axis=1, keepdims=True)
    row_var = embedding.var(axis=1, keepdims=True)
    normalized = (embedding - row_mean) / np.sqrt(row_var + 1.0e-5)
    normalized = normalized * weight[None, :] + bias[None, :]
    lookup = {
        symbol: normalized[int(index)].astype(np.float32, copy=False)
        for symbol, index in vocab.items()
        if int(index) < normalized.shape[0]
    }
    metadata = {
        "checkpoint": str((model_dir / "best_model.pt").relative_to(ROOT)),
        "checkpoint_sha256": sha256(model_dir / "best_model.pt"),
        "vocab": str((model_dir / "vocab.json").relative_to(ROOT)),
        "vocab_sha256": sha256(model_dir / "vocab.json"),
        "embedding_dim": int(normalized.shape[1]),
        "representation": "frozen layer-normalized scGPT gene-token embedding",
    }
    return lookup, metadata


def cohort_features(
    gene_ids: list[str], lookup: dict[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    mapping = pd.read_csv(GENE_MAP, sep="\t", dtype=str)
    mapping = mapping.drop_duplicates("gene_id", keep="first").set_index("gene_id")
    symbols = mapping.reindex(gene_ids)["st_gene_name"]
    covered = symbols.isin(lookup).to_numpy(bool)
    dimension = len(next(iter(lookup.values())))
    features = np.full((len(gene_ids), dimension), np.nan, dtype=np.float32)
    for index in np.flatnonzero(covered):
        features[index] = lookup[str(symbols.iloc[index])]
    coverage = pd.DataFrame(
        {
            "gene_id": gene_ids,
            "gene_symbol": symbols.astype("string").to_numpy(),
            "in_scgpt_vocabulary": covered,
        }
    ).reset_index(drop=True)
    return features, covered, coverage


def fit_condition(
    base_features: np.ndarray,
    training_gene_means: np.ndarray,
    train_idx: np.ndarray,
    heldout_idx: np.ndarray,
    condition: str,
    seed: int,
) -> tuple[np.ndarray, float, pd.DataFrame, np.ndarray]:
    all_indices = np.concatenate([train_idx, heldout_idx])
    if condition == "scgpt_gene_token":
        features = base_features.copy()
        source_index = np.arange(len(features), dtype=np.int64)
    elif condition == "random":
        rng = np.random.default_rng(seed + 12_345)
        features = base_features.copy()
        features[all_indices] = rng.standard_normal(
            size=(len(all_indices), base_features.shape[1])
        ).astype(np.float32)
        source_index = np.arange(len(features), dtype=np.int64)
    elif condition == "identity_shuffled_scgpt":
        rng = np.random.default_rng(seed + 810_001)
        source_index = np.arange(len(base_features), dtype=np.int64)
        for indices in (train_idx, heldout_idx):
            source_index[indices] = rng.permutation(indices)
        features = base_features[source_index]
    else:
        raise ValueError(condition)
    train_mean = features[train_idx].mean(axis=0, keepdims=True)
    train_std = features[train_idx].std(axis=0, keepdims=True)
    features = (features - train_mean) / np.maximum(train_std, 1.0e-6)
    model, alpha, validation = mean_only.fit_ridge(
        features, training_gene_means, train_idx
    )
    prediction = np.maximum(model.predict(features[heldout_idx]), 0.0)
    return prediction, alpha, validation, source_index


def main() -> None:
    global MODEL_DIR, OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", choices=COHORTS)
    parser.add_argument("--checkpoint", choices=CHECKPOINTS, default="brain")
    args = parser.parse_args()
    MODEL_DIR, OUT = CHECKPOINTS[args.checkpoint]
    cohorts = (args.cohort,) if args.cohort else COHORTS
    OUT.mkdir(parents=True, exist_ok=True)
    lookup, model_metadata = token_feature_lookup()
    absolute_rows: list[dict[str, object]] = []
    validation_rows: list[dict[str, object]] = []
    coverage_rows: list[pd.DataFrame] = []
    prediction_rows: list[dict[str, object]] = []

    for cohort in cohorts:
        (
            gene_ids,
            full_train_idx,
            full_heldout_idx,
            training_gene_means,
            _decima_features,
            _scaling,
            _manifest,
        ) = mean_only.training_means(cohort)
        raw_features, covered, coverage = cohort_features(gene_ids, lookup)
        train_idx = full_train_idx[covered[full_train_idx]]
        heldout_idx = full_heldout_idx[covered[full_heldout_idx]]
        coverage.insert(0, "cohort", cohort)
        coverage["downstream_partition"] = "excluded"
        coverage.loc[full_train_idx, "downstream_partition"] = "training"
        coverage.loc[full_heldout_idx, "downstream_partition"] = "held_out"
        coverage_rows.append(coverage)
        gene_table_path = (
            mean_only.CLEAN_ROOT
            / "components/runs"
            / cohort
            / "seed_42/decima/gene_components.tsv.gz"
        )
        gene_table = pd.read_csv(gene_table_path, sep="\t")
        table = gene_table.set_index("gene_id").loc[
            [gene_ids[int(index)] for index in heldout_idx]
        ].reset_index()

        for condition in CONDITIONS:
            condition_seeds = (42,) if condition == "scgpt_gene_token" else SEEDS
            for seed in condition_seeds:
                prediction, alpha, validation, _source_index = fit_condition(
                    raw_features,
                    training_gene_means,
                    train_idx,
                    heldout_idx,
                    condition,
                    seed,
                )
                metrics = mean_only.constant_map_metrics(prediction, table)
                absolute_rows.append(
                    {
                        "cohort": cohort,
                        "display_cohort": mean_only.DISPLAY[cohort],
                        "condition": condition,
                        "seed": seed,
                        "selected_alpha": alpha,
                        "n_training_genes": len(train_idx),
                        "n_heldout_genes": len(heldout_idx),
                        "training_coverage": len(train_idx) / len(full_train_idx),
                        "heldout_coverage": len(heldout_idx) / len(full_heldout_idx),
                        **metrics,
                    }
                )
                validation = validation.copy()
                validation.insert(0, "seed", seed)
                validation.insert(0, "condition", condition)
                validation.insert(0, "cohort", cohort)
                validation_rows.extend(validation.to_dict(orient="records"))
                prediction_rows.extend(
                    {
                        "cohort": cohort,
                        "condition": condition,
                        "seed": seed,
                        "gene_id": gene_ids[int(index)],
                        "predicted_gene_mean": value,
                    }
                    for index, value in zip(heldout_idx, prediction.tolist())
                )

    absolute = pd.DataFrame(absolute_rows)
    validation = pd.DataFrame(validation_rows)
    coverage = pd.concat(coverage_rows, ignore_index=True)
    predictions = pd.DataFrame(prediction_rows)
    absolute.to_csv(OUT / "absolute.tsv", sep="\t", index=False)
    validation.to_csv(OUT / "ridge_validation.tsv", sep="\t", index=False)
    coverage.to_csv(OUT / "gene_coverage.tsv.gz", sep="\t", index=False)
    predictions.to_csv(
        OUT / "predictions.tsv.gz",
        sep="\t",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )

    effect_rows: list[dict[str, object]] = []
    for cohort in cohorts:
        subset = absolute.loc[absolute["cohort"] == cohort]
        pretrained = subset.loc[subset["condition"] == "scgpt_gene_token"].iloc[0]
        for row in subset.loc[subset["condition"] != "scgpt_gene_token"].itertuples():
            effect_rows.append(
                {
                    "cohort": cohort,
                    "display_cohort": row.display_cohort,
                    "control": row.condition,
                    "seed": row.seed,
                    "abundance_pcc_gain": pretrained.abundance_pcc
                    - row.abundance_pcc,
                    "full_matrix_pcc_gain": pretrained.full_matrix_pcc
                    - row.full_matrix_pcc,
                    "overall_mse_reduction": row.overall_mse
                    - pretrained.overall_mse,
                }
            )
    effects = pd.DataFrame(effect_rows)
    effects.to_csv(OUT / "effects_per_run.tsv", sep="\t", index=False)
    summary = (
        effects.groupby(["cohort", "display_cohort", "control"], sort=False)
        .agg(
            abundance_pcc_gain_mean=("abundance_pcc_gain", "mean"),
            abundance_pcc_gain_min=("abundance_pcc_gain", "min"),
            abundance_pcc_gain_max=("abundance_pcc_gain", "max"),
            full_matrix_pcc_gain_mean=("full_matrix_pcc_gain", "mean"),
            full_matrix_pcc_gain_min=("full_matrix_pcc_gain", "min"),
            full_matrix_pcc_gain_max=("full_matrix_pcc_gain", "max"),
            overall_mse_reduction_mean=("overall_mse_reduction", "mean"),
            overall_mse_reduction_min=("overall_mse_reduction", "min"),
            overall_mse_reduction_max=("overall_mse_reduction", "max"),
        )
        .reset_index()
    )
    summary.to_csv(OUT / "effects_summary.tsv", sep="\t", index=False)
    manifest = {
        "status": "PASS",
        "purpose": "independent no-image gene-mean control for a second static pretrained gene representation",
        "model": model_metadata,
        "gene_mapping": {
            "path": str(GENE_MAP.relative_to(ROOT)),
            "sha256": sha256(GENE_MAP),
        },
        "conditions": list(CONDITIONS),
        "cohorts": list(cohorts),
        "spot_or_image_information": "none",
        "interpretation_boundary": "static gene-token vectors; not the older donor-specific scGPT cell-context experiments and not a spatial predictor",
        "checks": {
            "all_decompositions_exact": bool(
                np.max(np.abs(absolute["decomposition_error"].to_numpy(float)))
                < 1.0e-12
            ),
            "within_gene_metrics_zero": bool(
                np.all(absolute["mean_gene_pcc"].to_numpy(float) == 0.0)
            ),
            "minimum_training_coverage": float(absolute["training_coverage"].min()),
            "minimum_heldout_coverage": float(absolute["heldout_coverage"].min()),
        },
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    print(absolute.to_string(index=False))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
