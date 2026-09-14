#!/usr/bin/env python3
"""Fit matched no-image gene-mean controls for the held-out-gene assay.

The existing independent gene-mean assay maps standardized pretrained gene
vectors to training-individual gene means and broadcasts one prediction over
all test spots.  This script adds model-matched random-vector and
identity-shuffled-pretrained-vector controls.  Every condition uses the same
training genes, validation genes, ridge family and alpha grid.  Random-vector
realizations and within-partition identity permutations follow the seed rules
used by the corresponding spatial-decoder controls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.append(str(REPOSITORY_ROOT))

from experiments.heldout_gene_second_stage import build_gene_mean_baselines as mean_only
from experiments.multicohort_geneheldout_decima_clean_split.clean_split_config import (
    COHORTS,
    ROOT,
    SEEDS,
)


OUT = ROOT / "experiments/heldout_gene_second_stage/gene_mean_only"
RUN_ROOT = OUT / "matched_runs"
CONDITIONS = ("pretrained", "random", "identity_shuffled_pretrained")
CORRELATION_ENDPOINTS = (
    "abundance_pcc",
    "full_matrix_pcc",
    "mean_gene_pcc",
    "gene_centered_full_matrix_pcc",
)
ERROR_ENDPOINTS = (
    "abundance_rmse",
    "abundance_mse",
    "overall_mse",
    "centered_rmse",
    "centered_mse",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def condition_features(
    standardized_pretrained: np.ndarray,
    train_idx: np.ndarray,
    heldout_idx: np.ndarray,
    condition: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if condition == "pretrained":
        return standardized_pretrained, np.arange(len(standardized_pretrained))
    if condition == "random":
        rng = np.random.default_rng(seed + 12_345)
        features = rng.standard_normal(size=standardized_pretrained.shape).astype(
            np.float32
        )
        return features, np.arange(len(features))
    if condition == "identity_shuffled_pretrained":
        rng = np.random.default_rng(seed + 810_001)
        source_index = np.arange(len(standardized_pretrained), dtype=np.int64)
        for indices in (train_idx, heldout_idx):
            source_index[indices] = rng.permutation(indices)
        return standardized_pretrained[source_index], source_index
    raise ValueError(f"Unsupported condition: {condition}")


def run_cohort(cohort: str) -> None:
    run_dir = RUN_ROOT / cohort
    run_dir.mkdir(parents=True, exist_ok=True)
    (
        gene_ids,
        train_idx,
        heldout_idx,
        training_gene_means,
        pretrained_features,
        _scaling,
        _manifest,
    ) = mean_only.training_means(cohort)
    gene_table_path = (
        mean_only.CLEAN_ROOT
        / "components/runs"
        / cohort
        / "seed_42/decima/gene_components.tsv.gz"
    )
    gene_table = pd.read_csv(gene_table_path, sep="\t")
    heldout_genes = [gene_ids[int(index)] for index in heldout_idx]
    if gene_table["gene_id"].astype(str).tolist() != heldout_genes:
        raise ValueError(f"Held-out gene order mismatch for {cohort}")

    absolute_rows: list[dict[str, object]] = []
    validation_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    mapping_rows: list[dict[str, object]] = []

    for condition in CONDITIONS:
        condition_seeds = (42,) if condition == "pretrained" else SEEDS
        for seed in condition_seeds:
            features, source_index = condition_features(
                pretrained_features, train_idx, heldout_idx, condition, seed
            )
            model, selected_alpha, validation = mean_only.fit_ridge(
                features, training_gene_means, train_idx
            )
            prediction = np.maximum(model.predict(features[heldout_idx]), 0.0)
            metrics = mean_only.constant_map_metrics(prediction, gene_table)
            absolute_rows.append(
                {
                    "cohort": cohort,
                    "display_cohort": mean_only.DISPLAY[cohort],
                    "condition": condition,
                    "seed": seed,
                    "selected_alpha": selected_alpha,
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
                    "gene_id": gene,
                    "predicted_gene_mean": value,
                }
                for gene, value in zip(heldout_genes, prediction.tolist())
            )
            if condition == "identity_shuffled_pretrained":
                for partition, indices in (
                    ("training", train_idx),
                    ("held_out", heldout_idx),
                ):
                    mapping_rows.extend(
                        {
                            "cohort": cohort,
                            "seed": seed,
                            "partition": partition,
                            "target_gene": gene_ids[int(target)],
                            "source_vector_gene": gene_ids[int(source_index[target])],
                        }
                        for target in indices.tolist()
                    )

    absolute = pd.DataFrame(absolute_rows)
    validation = pd.DataFrame(validation_rows)
    predictions = pd.DataFrame(prediction_rows)
    mappings = pd.DataFrame(mapping_rows)
    absolute.to_csv(run_dir / "absolute.tsv", sep="\t", index=False)
    validation.to_csv(run_dir / "ridge_validation.tsv", sep="\t", index=False)
    predictions.to_csv(
        run_dir / "predictions.tsv.gz",
        sep="\t",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    mappings.to_csv(
        run_dir / "identity_permutations.tsv.gz",
        sep="\t",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    manifest = {
        "status": "PASS",
        "cohort": cohort,
        "conditions": list(CONDITIONS),
        "control_seeds": list(SEEDS),
        "pretrained_fit_seed": 42,
        "alpha_grid": list(mean_only.ALPHAS),
        "validation": "same fixed training-gene subset in every condition; minimum validation RMSE",
        "spot_or_image_information": "none",
        "n_absolute_rows": len(absolute),
        "n_predictions": len(predictions),
        "checks": {
            "expected_absolute_rows": len(absolute) == 1 + 2 * len(SEEDS),
            "all_decompositions_exact": bool(
                np.max(np.abs(absolute["decomposition_error"].to_numpy(float)))
                < 1.0e-12
            ),
            "all_within_gene_correlations_zero": bool(
                np.all(absolute["mean_gene_pcc"].to_numpy(float) == 0.0)
                and np.all(
                    absolute["gene_centered_full_matrix_pcc"].to_numpy(float)
                    == 0.0
                )
            ),
            "shuffle_preserves_partitions": bool(
                len(mappings)
                == len(SEEDS) * (len(train_idx) + len(heldout_idx))
            ),
        },
        "input_gene_table": {
            "path": str(gene_table_path.relative_to(ROOT)),
            "sha256": sha256(gene_table_path),
        },
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if not all(manifest["checks"].values()):
        raise RuntimeError(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2), flush=True)
    print(absolute.to_string(index=False), flush=True)


def summarize() -> None:
    frames = []
    manifests = []
    for cohort in COHORTS:
        run_dir = RUN_ROOT / cohort
        frames.append(pd.read_csv(run_dir / "absolute.tsv", sep="\t"))
        manifest_path = run_dir / "manifest.json"
        manifests.append(
            {
                "path": str(manifest_path.relative_to(ROOT)),
                "sha256": sha256(manifest_path),
            }
        )
    absolute = pd.concat(frames, ignore_index=True)
    reference = (
        absolute.loc[absolute["condition"] == "pretrained"]
        .set_index("cohort")
        .to_dict(orient="index")
    )
    effect_rows: list[dict[str, object]] = []
    for row in absolute.loc[absolute["condition"] != "pretrained"].to_dict(
        orient="records"
    ):
        pretrained = reference[str(row["cohort"])]
        out: dict[str, object] = {
            "cohort": row["cohort"],
            "display_cohort": row["display_cohort"],
            "control": row["condition"],
            "seed": row["seed"],
        }
        for endpoint in CORRELATION_ENDPOINTS:
            out[f"{endpoint}_gain"] = float(pretrained[endpoint]) - float(
                row[endpoint]
            )
        for endpoint in ERROR_ENDPOINTS:
            out[f"{endpoint}_reduction"] = float(row[endpoint]) - float(
                pretrained[endpoint]
            )
        effect_rows.append(out)
    effects = pd.DataFrame(effect_rows)
    value_columns = [
        column
        for column in effects.columns
        if column not in {"cohort", "display_cohort", "control", "seed"}
    ]
    summary_rows: list[dict[str, object]] = []
    for keys, group in effects.groupby(
        ["cohort", "display_cohort", "control"], sort=False
    ):
        for endpoint in value_columns:
            values = group[endpoint].to_numpy(float)
            summary_rows.append(
                {
                    "cohort": keys[0],
                    "display_cohort": keys[1],
                    "control": keys[2],
                    "endpoint": endpoint,
                    "mean_effect": float(np.mean(values)),
                    "min_effect": float(np.min(values)),
                    "max_effect": float(np.max(values)),
                    "n_runs": len(values),
                    "n_positive": int(np.sum(values > 0)),
                }
            )
    summary = pd.DataFrame(summary_rows)
    absolute.to_csv(OUT / "matched_absolute.tsv", sep="\t", index=False)
    effects.to_csv(OUT / "matched_effects_per_run.tsv", sep="\t", index=False)
    summary.to_csv(OUT / "matched_effects_summary.tsv", sep="\t", index=False)
    manifest = {
        "status": "PASS",
        "description": "architecture-matched no-image gene-mean ridge assay",
        "conditions": list(CONDITIONS),
        "spot_or_image_information": "none",
        "n_cohorts": len(COHORTS),
        "control_seeds": list(SEEDS),
        "n_absolute_rows": len(absolute),
        "n_effect_rows": len(effects),
        "checks": {
            "all_cohorts_present": set(absolute["cohort"]) == set(COHORTS),
            "expected_absolute_rows": len(absolute)
            == len(COHORTS) * (1 + 2 * len(SEEDS)),
            "expected_effect_rows": len(effects)
            == len(COHORTS) * 2 * len(SEEDS),
            "all_decompositions_exact": bool(
                np.max(np.abs(absolute["decomposition_error"].to_numpy(float)))
                < 1.0e-12
            ),
            "all_within_gene_correlations_zero": bool(
                np.all(absolute["mean_gene_pcc"].to_numpy(float) == 0.0)
                and np.all(
                    absolute["gene_centered_full_matrix_pcc"].to_numpy(float)
                    == 0.0
                )
            ),
        },
        "inputs": manifests,
        "outputs": [
            "matched_absolute.tsv",
            "matched_effects_per_run.tsv",
            "matched_effects_summary.tsv",
        ],
    }
    (OUT / "matched_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if not all(manifest["checks"].values()):
        raise RuntimeError(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    selected = summary.loc[
        summary["endpoint"].isin(
            [
                "abundance_pcc_gain",
                "full_matrix_pcc_gain",
                "overall_mse_reduction",
            ]
        )
    ]
    print(selected.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", choices=COHORTS)
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if args.summarize:
        summarize()
    elif args.cohort:
        run_cohort(args.cohort)
    else:
        parser.error("provide --cohort or --summarize")


if __name__ == "__main__":
    main()
