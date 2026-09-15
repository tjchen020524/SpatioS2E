#!/usr/bin/env python3
"""Evaluate held-out-gene predictions after centring within each test section.

This script reloads one frozen checkpoint and streams sufficient statistics; it
does not retrain a model or save a dense spot-by-gene prediction matrix.  The
primary aggregation gives each biological individual equal weight.  A
centre-within-section-then-concatenate gene-wise PCC and observation-weighted
MSE decomposition are retained as sensitivity analyses.
"""

from __future__ import annotations
from workflow_paths import DATA_ROOT

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = DATA_ROOT
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from experiments.heldout_metric_policy import (  # noqa: E402
    OBSERVED_STD_MIN,
    PREDICTED_STD_MIN,
    gene_correlations_from_moments,
)
from experiments.multicohort_geneheldout_decima.run_cohort_geneheldout import (  # noqa: E402
    configure_base,
)
from experiments.geneheldout_uni2h_decima_decoder_noseq_controls import (  # noqa: E402
    run_geneheldout_noseq_controls as base,
)
from experiments.multicohort_geneheldout_decima_clean_split.clean_split_config import (  # noqa: E402
    CLEAN_ROOT,
    configure_clean_base,
)


EXP_ROOT = ROOT / "experiments/multicohort_geneheldout_decima"
OUT_ROOT = EXP_ROOT / "section_centered"
COHORTS = ("hippocampus_donor_disjoint", "dlpfc", "nac", "her2st")
VARIANTS = ("decima", "random", "constant")
SEEDS = (42, 123, 456)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Across-gene PCC, excluding constant or numerically constant vectors."""
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 2:
        return float("nan")
    x_use = x[mask].astype(np.float64, copy=False)
    y_use = y[mask].astype(np.float64, copy=False)
    if float(np.std(x_use)) <= OBSERVED_STD_MIN or float(np.std(y_use)) <= PREDICTED_STD_MIN:
        return float("nan")
    x_centered = x_use - x_use.mean()
    y_centered = y_use - y_use.mean()
    denominator = math.sqrt(float(np.dot(x_centered, x_centered) * np.dot(y_centered, y_centered)))
    if denominator <= 0.0:
        return float("nan")
    return float(np.dot(x_centered, y_centered) / denominator)


@dataclass
class VectorMoments:
    count: np.ndarray
    sum_pred: np.ndarray
    sum_true: np.ndarray
    sum_pred2: np.ndarray
    sum_true2: np.ndarray
    sum_cross: np.ndarray
    sum_sq_error: np.ndarray

    @classmethod
    def zeros(cls, n_genes: int) -> "VectorMoments":
        return cls(
            count=np.zeros(n_genes, dtype=np.int64),
            sum_pred=np.zeros(n_genes, dtype=np.float64),
            sum_true=np.zeros(n_genes, dtype=np.float64),
            sum_pred2=np.zeros(n_genes, dtype=np.float64),
            sum_true2=np.zeros(n_genes, dtype=np.float64),
            sum_cross=np.zeros(n_genes, dtype=np.float64),
            sum_sq_error=np.zeros(n_genes, dtype=np.float64),
        )

    def update(self, local_idx: np.ndarray, pred: np.ndarray, true: np.ndarray) -> None:
        mask = np.isfinite(pred) & np.isfinite(true)
        pred64 = np.where(mask, pred, 0.0).astype(np.float64, copy=False)
        true64 = np.where(mask, true, 0.0).astype(np.float64, copy=False)
        self.count[local_idx] += mask.sum(axis=0).astype(np.int64)
        self.sum_pred[local_idx] += pred64.sum(axis=0)
        self.sum_true[local_idx] += true64.sum(axis=0)
        self.sum_pred2[local_idx] += np.square(pred64).sum(axis=0)
        self.sum_true2[local_idx] += np.square(true64).sum(axis=0)
        self.sum_cross[local_idx] += (pred64 * true64).sum(axis=0)
        self.sum_sq_error[local_idx] += np.square(pred64 - true64).sum(axis=0)

    def derived(self) -> dict[str, np.ndarray | float | int]:
        valid = self.count > 0
        mean_pred = np.full(self.count.shape, np.nan, dtype=np.float64)
        mean_true = np.full(self.count.shape, np.nan, dtype=np.float64)
        mean_pred[valid] = self.sum_pred[valid] / self.count[valid]
        mean_true[valid] = self.sum_true[valid] / self.count[valid]

        centered_pred_ss = np.zeros(self.count.shape, dtype=np.float64)
        centered_true_ss = np.zeros(self.count.shape, dtype=np.float64)
        centered_cross = np.zeros(self.count.shape, dtype=np.float64)
        centered_pred_ss[valid] = (
            self.sum_pred2[valid] - np.square(self.sum_pred[valid]) / self.count[valid]
        )
        centered_true_ss[valid] = (
            self.sum_true2[valid] - np.square(self.sum_true[valid]) / self.count[valid]
        )
        centered_cross[valid] = (
            self.sum_cross[valid] - self.sum_pred[valid] * self.sum_true[valid] / self.count[valid]
        )
        centered_pred_ss = np.maximum(centered_pred_ss, 0.0)
        centered_true_ss = np.maximum(centered_true_ss, 0.0)
        policy = gene_correlations_from_moments(
            self.count,
            self.sum_pred,
            self.sum_true,
            self.sum_pred2,
            self.sum_true2,
            self.sum_cross,
        )

        abundance_sq_error = np.zeros(self.count.shape, dtype=np.float64)
        abundance_sq_error[valid] = self.count[valid] * np.square(
            mean_pred[valid] - mean_true[valid]
        )
        centered_sq_error = np.maximum(self.sum_sq_error - abundance_sq_error, 0.0)
        total_count = int(self.count.sum())
        total_sse = float(self.sum_sq_error.sum())
        abundance_sse = float(abundance_sq_error.sum())
        centered_sse = float(centered_sq_error.sum())
        return {
            "valid": valid,
            "mean_pred": mean_pred,
            "mean_true": mean_true,
            "centered_pred_ss": centered_pred_ss,
            "centered_true_ss": centered_true_ss,
            "centered_cross": centered_cross,
            "correlation": policy.correlation,
            "gene_pcc_eligible": policy.eligible,
            "prediction_variable": policy.prediction_variable,
            "observed_std": policy.observed_std,
            "predicted_std": policy.predicted_std,
            "centered_full_matrix_pcc": policy.centered_full_matrix_pcc,
            "n_eligible_gene_pcc": policy.n_eligible,
            "n_constant_prediction_gene_pcc": policy.n_constant_prediction,
            "abundance_sq_error": abundance_sq_error,
            "centered_sq_error": centered_sq_error,
            "total_count": total_count,
            "total_sse": total_sse,
            "abundance_sse": abundance_sse,
            "centered_sse": centered_sse,
        }


class ConcatenatedCenteredMoments:
    def __init__(self, n_genes: int) -> None:
        self.count = np.zeros(n_genes, dtype=np.int64)
        self.pred_ss = np.zeros(n_genes, dtype=np.float64)
        self.true_ss = np.zeros(n_genes, dtype=np.float64)
        self.cross = np.zeros(n_genes, dtype=np.float64)

    def add_section(self, moments: VectorMoments, derived: dict[str, object]) -> None:
        self.count += moments.count
        self.pred_ss += np.asarray(derived["centered_pred_ss"], dtype=np.float64)
        self.true_ss += np.asarray(derived["centered_true_ss"], dtype=np.float64)
        self.cross += np.asarray(derived["centered_cross"], dtype=np.float64)

    def corr(self) -> np.ndarray:
        return self.metrics()["correlation"]

    def metrics(self) -> dict[str, object]:
        zeros = np.zeros(self.count.shape, dtype=np.float64)
        policy = gene_correlations_from_moments(
            self.count,
            zeros,
            zeros,
            self.pred_ss,
            self.true_ss,
            self.cross,
        )
        return {
            "correlation": policy.correlation,
            "eligible": policy.eligible,
            "centered_full_matrix_pcc": policy.centered_full_matrix_pcc,
            "mean_gene_pcc": policy.mean,
            "median_gene_pcc": policy.median,
            "n_eligible_gene_pcc": policy.n_eligible,
            "n_constant_prediction_gene_pcc": policy.n_constant_prediction,
        }


def load_individual_map(cohort: str) -> dict[str, str]:
    if cohort == "hippocampus_donor_disjoint":
        path = ROOT / "experiments/hippocampus_current_full_ablation/data/sample_manifest.json"
        rows = json.loads(path.read_text())
        return {str(row["sample"]): str(row["donor"]) for row in rows}
    if cohort == "nac":
        path = ROOT / "experiments/nac_current_full_ablation/data/sample_manifest.tsv"
        frame = pd.read_csv(path, sep="\t", usecols=["sample", "donor"])
        return dict(zip(frame["sample"].astype(str), frame["donor"].astype(str)))
    if cohort == "her2st":
        path = ROOT / "experiments/her2st_current_full_ablation/data/sample_manifest.tsv"
        frame = pd.read_csv(path, sep="\t", usecols=["sample", "patient"])
        return dict(zip(frame["sample"].astype(str), frame["patient"].astype(str)))
    if cohort == "dlpfc":
        path = base.DATA_ROOT / "manifest.tsv"
        samples = pd.read_csv(path, sep="\t", usecols=["sample"])["sample"].astype(str).unique()
        return {sample: sample for sample in samples}
    raise ValueError(cohort)


def embedding_matrix(variant: str, gene_ids: list[str], train_idx: np.ndarray, seed: int) -> np.ndarray:
    decima = base.load_decima_embeddings(gene_ids, train_idx)
    if variant == "decima":
        return decima
    if variant == "random":
        return base.load_random_embeddings(decima.shape, seed + 12345)
    return base.load_constant_embeddings(decima.shape)


def section_metrics(sample: str, individual: str, derived: dict[str, object]) -> dict[str, object]:
    correlation = np.asarray(derived["correlation"], dtype=np.float64)
    eligible = np.asarray(derived["gene_pcc_eligible"], dtype=bool)
    eligible_gene_corr = correlation[eligible]
    mean_pred = np.asarray(derived["mean_pred"], dtype=np.float64)
    mean_true = np.asarray(derived["mean_true"], dtype=np.float64)
    mean_valid = np.isfinite(mean_pred) & np.isfinite(mean_true)
    abundance_mse_unweighted = float(np.mean(np.square(mean_pred[mean_valid] - mean_true[mean_valid])))
    total_count = int(derived["total_count"])
    total_mse = float(derived["total_sse"]) / total_count
    abundance_mse = float(derived["abundance_sse"]) / total_count
    centered_mse = float(derived["centered_sse"]) / total_count
    return {
        "sample": sample,
        "individual": individual,
        "n_spots": int(np.max(np.asarray(derived["total_count"])) // int(mean_valid.sum())),
        "n_genes": int(mean_valid.sum()),
        "n_eligible_gene_pcc": int(eligible_gene_corr.size),
        "n_finite_gene_pcc": int(eligible_gene_corr.size),
        "n_constant_prediction_gene_pcc": int(derived["n_constant_prediction_gene_pcc"]),
        "abundance_pcc": finite_corr(mean_true, mean_pred),
        "abundance_rmse": math.sqrt(max(abundance_mse_unweighted, 0.0)),
        "mean_within_section_gene_pcc": float(np.mean(eligible_gene_corr)) if eligible_gene_corr.size else float("nan"),
        "median_within_section_gene_pcc": float(np.median(eligible_gene_corr)) if eligible_gene_corr.size else float("nan"),
        "centered_full_matrix_pcc": float(derived["centered_full_matrix_pcc"]),
        "section_centered_rmse": math.sqrt(max(centered_mse, 0.0)),
        "total_mse": total_mse,
        "abundance_mse": abundance_mse,
        "centered_mse": centered_mse,
        "decomposition_error": total_mse - abundance_mse - centered_mse,
        "total_count": total_count,
        "total_sse": float(derived["total_sse"]),
        "abundance_sse": float(derived["abundance_sse"]),
        "centered_sse": float(derived["centered_sse"]),
    }


def aggregate_individuals(section_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for individual, group in section_frame.groupby("individual", sort=True):
        total_mse = float(group["total_mse"].mean())
        abundance_mse = float(group["abundance_mse"].mean())
        centered_mse = float(group["centered_mse"].mean())
        rows.append(
            {
                "individual": individual,
                "n_sections": int(len(group)),
                "abundance_pcc": float(group["abundance_pcc"].mean()),
                "abundance_rmse": math.sqrt(max(abundance_mse, 0.0)),
                "mean_within_section_gene_pcc": float(group["mean_within_section_gene_pcc"].mean()),
                "centered_full_matrix_pcc": float(group["centered_full_matrix_pcc"].mean()),
                "section_centered_rmse": math.sqrt(max(centered_mse, 0.0)),
                "total_mse": total_mse,
                "abundance_mse": abundance_mse,
                "centered_mse": centered_mse,
                "decomposition_error": total_mse - abundance_mse - centered_mse,
            }
        )
    return pd.DataFrame(rows)


def evaluate_one(
    cohort: str,
    seed: int,
    variant: str,
    device: torch.device,
    batch_size: int,
    gene_chunk_size: int,
    max_sections: int | None,
    max_genes: int | None,
    output_root: Path,
    partition: str = "prespecified",
) -> dict[str, object]:
    if partition == "training_only":
        configure_clean_base(cohort, seed, variant)
    elif partition == "prespecified":
        configure_base(cohort, seed, variant)
    else:
        raise ValueError(f"Unknown partition: {partition}")
    base.set_seed(seed)
    manifest = pd.read_csv(base.DATA_ROOT / "manifest.tsv", sep="\t")
    test_manifest = manifest.loc[manifest["split"] == "test"].copy()
    samples = sorted(test_manifest["sample"].astype(str).unique().tolist())
    if max_sections is not None:
        samples = samples[:max_sections]
        test_manifest = test_manifest.loc[test_manifest["sample"].astype(str).isin(samples)].copy()
    individual_map = load_individual_map(cohort)
    missing_individuals = sorted(set(samples) - set(individual_map))
    if missing_individuals:
        raise ValueError(f"Missing individual mapping for {cohort}: {missing_individuals}")

    gene_ids = base.load_gene_ids()
    train_idx, heldout_idx = base.load_gene_split(gene_ids)
    if max_genes is not None:
        heldout_idx = heldout_idx[:max_genes]
    heldout_gene_ids = [gene_ids[int(idx)] for idx in heldout_idx]
    embeddings = embedding_matrix(variant, gene_ids, train_idx, seed)
    feature_store = base.FeatureStore()
    expression_store = base.ExpressionStore(gene_ids)
    split = json.loads((base.DATA_ROOT / "split.json").read_text())
    model = base.GeneConditionedProgramDecoder(
        spot_dim=int(split["embedding_dim"]),
        gene_dim=int(embeddings.shape[1]),
        hidden_dim=512,
        program_dim=96,
        dropout=0.10,
    ).to(device)
    checkpoint = base.EXP_ROOT / variant / "checkpoints/best.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()
    gene_tensor = base.numpy_to_tensor(embeddings).to(device)
    chunks = base.make_gene_chunks(len(heldout_idx), gene_chunk_size)

    section_rows: list[dict[str, object]] = []
    concatenated = ConcatenatedCenteredMoments(len(heldout_idx))
    for section_number, sample in enumerate(samples, start=1):
        sample_manifest = test_manifest.loc[test_manifest["sample"].astype(str) == sample].copy()
        dataset = base.SpotDataset(sample_manifest, feature_store, expression_store)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
            pin_memory=device.type == "cuda",
            collate_fn=base.collate,
        )
        moments = VectorMoments.zeros(len(heldout_idx))
        with torch.no_grad():
            for batch in loader:
                x = batch["x"].to(device, non_blocking=True)
                y = batch["y"].to(device, non_blocking=True)
                for local_idx in chunks:
                    full_idx_np = heldout_idx[local_idx]
                    full_idx = torch.tensor(full_idx_np, dtype=torch.long, device=device)
                    pred = model(x, gene_tensor.index_select(0, full_idx))
                    true = y.index_select(1, full_idx)
                    moments.update(local_idx, base.tensor_to_numpy(pred), base.tensor_to_numpy(true))
        derived = moments.derived()
        concatenated.add_section(moments, derived)
        section_rows.append(section_metrics(sample, individual_map[sample], derived))
        print(
            f"[section-centred] {cohort} seed={seed} {variant} "
            f"section={section_number}/{len(samples)} sample={sample}",
            flush=True,
        )

    section_frame = pd.DataFrame(section_rows)
    individual_frame = aggregate_individuals(section_frame)
    concat_metrics = concatenated.metrics()
    concat_corr = np.asarray(concat_metrics["correlation"], dtype=np.float64)
    concat_eligible = np.asarray(concat_metrics["eligible"], dtype=bool)

    primary_total_mse = float(individual_frame["total_mse"].mean())
    primary_abundance_mse = float(individual_frame["abundance_mse"].mean())
    primary_centered_mse = float(individual_frame["centered_mse"].mean())
    observation_total_count = int(section_frame["total_count"].sum())
    observation_total_mse = float(section_frame["total_sse"].sum()) / observation_total_count
    observation_abundance_mse = float(section_frame["abundance_sse"].sum()) / observation_total_count
    observation_centered_mse = float(section_frame["centered_sse"].sum()) / observation_total_count

    output_dir = output_root / "runs" / cohort / f"seed_{seed}" / variant
    output_dir.mkdir(parents=True, exist_ok=True)
    section_frame.insert(0, "variant", variant)
    section_frame.insert(0, "seed", seed)
    section_frame.insert(0, "cohort", cohort)
    individual_frame.insert(0, "variant", variant)
    individual_frame.insert(0, "seed", seed)
    individual_frame.insert(0, "cohort", cohort)
    section_frame.to_csv(output_dir / "per_section.tsv", sep="\t", index=False, na_rep="NA")
    individual_frame.to_csv(output_dir / "per_individual.tsv", sep="\t", index=False, na_rep="NA")
    pd.DataFrame(
        {
            "cohort": cohort,
            "seed": seed,
            "variant": variant,
            "gene_id": heldout_gene_ids,
            "section_centered_concat_pcc": concat_corr,
            "gene_pcc_eligible": concat_eligible,
            "n_spots": concatenated.count,
        }
    ).to_csv(
        output_dir / "concat_gene_pcc.tsv.gz",
        sep="\t",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )

    summary: dict[str, object] = {
        "cohort": cohort,
        "seed": seed,
        "variant": variant,
        "n_sections": int(len(section_frame)),
        "n_individuals": int(len(individual_frame)),
        "n_heldout_genes": int(len(heldout_idx)),
        "primary_aggregation": "mean sections within biological individual, then equal-weight mean across individuals",
        "primary_abundance_pcc": float(individual_frame["abundance_pcc"].mean()),
        "primary_abundance_rmse": math.sqrt(max(primary_abundance_mse, 0.0)),
        "primary_mean_within_section_gene_pcc": float(individual_frame["mean_within_section_gene_pcc"].mean()),
        "primary_centered_full_matrix_pcc": float(individual_frame["centered_full_matrix_pcc"].mean()),
        "primary_section_centered_rmse": math.sqrt(max(primary_centered_mse, 0.0)),
        "primary_total_mse": primary_total_mse,
        "primary_abundance_mse": primary_abundance_mse,
        "primary_centered_mse": primary_centered_mse,
        "primary_decomposition_error": primary_total_mse - primary_abundance_mse - primary_centered_mse,
        "concat_centered_mean_gene_pcc": float(concat_metrics["mean_gene_pcc"]),
        "concat_centered_median_gene_pcc": float(concat_metrics["median_gene_pcc"]),
        "concat_centered_full_matrix_pcc": float(concat_metrics["centered_full_matrix_pcc"]),
        "concat_centered_n_eligible_gene_pcc": int(concat_metrics["n_eligible_gene_pcc"]),
        "concat_centered_n_finite_gene_pcc": int(concat_metrics["n_eligible_gene_pcc"]),
        "concat_centered_n_constant_prediction_gene_pcc": int(
            concat_metrics["n_constant_prediction_gene_pcc"]
        ),
        "observation_weighted_total_mse": observation_total_mse,
        "observation_weighted_abundance_mse": observation_abundance_mse,
        "observation_weighted_centered_mse": observation_centered_mse,
        "observation_weighted_section_centered_rmse": math.sqrt(max(observation_centered_mse, 0.0)),
        "observation_weighted_decomposition_error": observation_total_mse - observation_abundance_mse - observation_centered_mse,
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "checkpoint_sha256": sha256(checkpoint),
        "gene_split_dir": str(base.GENE_SPLIT_DIR.relative_to(ROOT)),
        "gene_split_train_sha256": sha256(base.GENE_SPLIT_DIR / "train_genes.txt"),
        "gene_split_heldout_sha256": sha256(base.GENE_SPLIT_DIR / "heldout_genes.txt"),
        "test_manifest_sha256": sha256(base.DATA_ROOT / "manifest.tsv"),
        "max_sections": max_sections,
        "max_genes": max_genes,
        "gene_partition": partition,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def self_test() -> None:
    true = np.asarray([[1.0, 3.0], [2.0, 2.0], [3.0, 1.0]], dtype=np.float32)
    pred = np.asarray([[2.0, 2.0], [3.0, 2.5], [4.0, 3.0]], dtype=np.float32)
    moments = VectorMoments.zeros(2)
    moments.update(np.arange(2), pred, true)
    derived = moments.derived()
    total_mse = float(np.square(pred - true).mean())
    abundance_mse = float(np.square(pred.mean(axis=0) - true.mean(axis=0)).mean())
    centered_mse = float(
        np.square((pred - pred.mean(axis=0)) - (true - true.mean(axis=0))).mean()
    )
    assert abs(float(derived["total_sse"]) / int(derived["total_count"]) - total_mse) < 1.0e-10
    assert abs(float(derived["abundance_sse"]) / int(derived["total_count"]) - abundance_mse) < 1.0e-10
    assert abs(float(derived["centered_sse"]) / int(derived["total_count"]) - centered_mse) < 1.0e-10
    assert abs(total_mse - abundance_mse - centered_mse) < 1.0e-10
    print("section-centred sufficient-statistics self-test: PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", choices=COHORTS)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--gene-chunk-size", type=int, default=512)
    parser.add_argument("--max-sections", type=int)
    parser.add_argument("--max-genes", type=int)
    parser.add_argument("--output-root", type=Path, default=OUT_ROOT)
    parser.add_argument(
        "--partition",
        choices=("prespecified", "training_only"),
        default="prespecified",
    )
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.cohort is None or args.seed is None or args.variant is None:
        parser.error("--cohort, --seed and --variant are required unless --self-test is used")
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    evaluate_one(
        cohort=args.cohort,
        seed=args.seed,
        variant=args.variant,
        device=device,
        batch_size=args.batch_size,
        gene_chunk_size=args.gene_chunk_size,
        max_sections=args.max_sections,
        max_genes=args.max_genes,
        output_root=(
            CLEAN_ROOT / "section_centered"
            if args.partition == "training_only" and args.output_root == OUT_ROOT
            else args.output_root
        ),
        partition=args.partition,
    )


if __name__ == "__main__":
    main()
