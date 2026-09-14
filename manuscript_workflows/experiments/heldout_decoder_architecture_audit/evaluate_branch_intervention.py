#!/usr/bin/env python3
"""Intervene on gene-vector identity separately in decoder branches.

The frozen clean-split pretrained-vector checkpoint is evaluated with correct
or held-out-gene-permuted Decima vectors supplied independently to the learned
gene-bias and spot--gene interaction branches.  This is an inference-time
pathway intervention, not branch-specific retraining or causal mediation.
"""

from __future__ import annotations
from workflow_paths import DATA_ROOT

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

REPO_ROOT = DATA_ROOT
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from experiments.geneheldout_uni2h_decima_decoder_noseq_controls import (
    run_geneheldout_noseq_controls as base,
)
from experiments.multicohort_geneheldout_decima.evaluate_section_centered import (
    VectorMoments,
    finite_corr,
    load_individual_map,
)
from experiments.multicohort_geneheldout_decima_clean_split.clean_split_config import (
    COHORTS,
    SEEDS,
    configure_clean_base,
)


ROOT = DATA_ROOT
OUTPUT_ROOT = ROOT / "experiments/heldout_decoder_architecture_audit/branch_intervention"
CONDITIONS = (
    "bias_correct__interaction_correct",
    "bias_correct__interaction_permuted",
    "bias_permuted__interaction_correct",
    "bias_permuted__interaction_permuted",
    "bias_correct__interaction_zero",
    "bias_zero__interaction_correct",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def flattened_corr(moments: VectorMoments) -> float:
    count = int(moments.count.sum())
    sum_pred = float(moments.sum_pred.sum())
    sum_true = float(moments.sum_true.sum())
    pred_ss = float(moments.sum_pred2.sum()) - sum_pred * sum_pred / count
    true_ss = float(moments.sum_true2.sum()) - sum_true * sum_true / count
    cross = float(moments.sum_cross.sum()) - sum_pred * sum_true / count
    if pred_ss <= 0.0 or true_ss <= 0.0:
        return float("nan")
    return cross / math.sqrt(pred_ss * true_ss)


def raw_metrics(moments: VectorMoments) -> dict[str, float | int]:
    derived = moments.derived()
    mean_pred = np.asarray(derived["mean_pred"], dtype=np.float64)
    mean_true = np.asarray(derived["mean_true"], dtype=np.float64)
    correlation = np.asarray(derived["correlation"], dtype=np.float64)
    eligible = np.asarray(derived["gene_pcc_eligible"], dtype=bool)
    eligible_correlation = correlation[eligible]
    valid = np.isfinite(mean_pred) & np.isfinite(mean_true)
    count = int(derived["total_count"])
    total_mse = float(derived["total_sse"]) / count
    mean_mse = float(derived["abundance_sse"]) / count
    centered_mse = float(derived["centered_sse"]) / count
    return {
        "n_observations": count,
        "n_genes": int(valid.sum()),
        "n_eligible_gene_pcc": int(eligible_correlation.size),
        "n_finite_gene_pcc": int(eligible_correlation.size),
        "n_constant_prediction_gene_pcc": int(derived["n_constant_prediction_gene_pcc"]),
        "full_matrix_pcc": flattened_corr(moments),
        "full_matrix_mse": total_mse,
        "abundance_pcc": finite_corr(mean_true, mean_pred),
        "abundance_rmse": math.sqrt(float(np.mean(np.square(mean_pred[valid] - mean_true[valid])))),
        "mean_gene_pcc": float(np.mean(eligible_correlation)) if eligible_correlation.size else float("nan"),
        "centered_full_matrix_pcc": float(derived["centered_full_matrix_pcc"]),
        "centered_rmse": math.sqrt(max(centered_mse, 0.0)),
        "gene_mean_mse": mean_mse,
        "centered_mse": centered_mse,
        "decomposition_error": total_mse - mean_mse - centered_mse,
    }


def section_metrics(moments: VectorMoments) -> dict[str, float | int]:
    derived = moments.derived()
    correlation = np.asarray(derived["correlation"], dtype=np.float64)
    eligible = np.asarray(derived["gene_pcc_eligible"], dtype=bool)
    eligible_correlation = correlation[eligible]
    centered_mse = float(derived["centered_sse"]) / int(derived["total_count"])
    return {
        "mean_within_section_gene_pcc": float(np.mean(eligible_correlation)) if eligible_correlation.size else float("nan"),
        "centered_full_matrix_pcc": float(derived["centered_full_matrix_pcc"]),
        "section_centered_mse": centered_mse,
        "section_centered_rmse": math.sqrt(max(centered_mse, 0.0)),
        "n_eligible_gene_pcc": int(eligible_correlation.size),
        "n_finite_gene_pcc": int(eligible_correlation.size),
        "n_constant_prediction_gene_pcc": int(derived["n_constant_prediction_gene_pcc"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", required=True, choices=COHORTS)
    parser.add_argument("--seed", required=True, type=int, choices=SEEDS)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--gene-chunk-size", type=int, default=512)
    parser.add_argument("--max-sections", type=int)
    parser.add_argument("--max-genes", type=int)
    args = parser.parse_args()

    configure_clean_base(args.cohort, args.seed, "decima")
    base.set_seed(args.seed)
    manifest = pd.read_csv(base.DATA_ROOT / "manifest.tsv", sep="\t")
    test_manifest = manifest.loc[manifest["split"] == "test"].copy()
    samples = sorted(test_manifest["sample"].astype(str).unique().tolist())
    if args.max_sections is not None:
        samples = samples[: args.max_sections]
    individual_map = load_individual_map(args.cohort)
    gene_ids = base.load_gene_ids()
    train_idx, heldout_idx = base.load_gene_split(gene_ids)
    if args.max_genes is not None:
        heldout_idx = heldout_idx[: args.max_genes]
    embeddings = base.load_decima_embeddings(gene_ids, train_idx)
    heldout_embeddings = embeddings[heldout_idx]
    permutation = np.random.default_rng(args.seed + 54321).permutation(len(heldout_idx))
    permuted_embeddings = heldout_embeddings[permutation]
    correct_tensor = base.numpy_to_tensor(heldout_embeddings).to("cuda" if torch.cuda.is_available() else "cpu")
    permuted_tensor = base.numpy_to_tensor(permuted_embeddings).to(correct_tensor.device)
    device = correct_tensor.device

    split = json.loads((base.DATA_ROOT / "split.json").read_text())
    model = base.GeneConditionedProgramDecoder(
        spot_dim=int(split["embedding_dim"]),
        gene_dim=int(embeddings.shape[1]),
        hidden_dim=512,
        program_dim=96,
        dropout=0.10,
    ).to(device)
    checkpoint = base.EXP_ROOT / "decima/checkpoints/best.pt"
    payload = torch.load(checkpoint, map_location=device)
    model.load_state_dict(payload["model"])
    model.eval()
    feature_store = base.FeatureStore()
    expression_store = base.ExpressionStore(gene_ids)
    chunks = base.make_gene_chunks(len(heldout_idx), args.gene_chunk_size)
    global_moments = {condition: VectorMoments.zeros(len(heldout_idx)) for condition in CONDITIONS}
    section_rows: list[dict[str, object]] = []

    with torch.inference_mode():
        for section_number, sample in enumerate(samples, start=1):
            sample_manifest = test_manifest.loc[test_manifest["sample"].astype(str) == sample].copy()
            dataset = base.SpotDataset(sample_manifest, feature_store, expression_store)
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=0,
                pin_memory=device.type == "cuda",
                collate_fn=base.collate,
            )
            local_moments = {condition: VectorMoments.zeros(len(heldout_idx)) for condition in CONDITIONS}
            for batch in loader:
                x = batch["x"].to(device, non_blocking=True)
                y = batch["y"].to(device, non_blocking=True)
                spot_program = model.spot_encoder(x)
                for local_idx in chunks:
                    full_idx_np = heldout_idx[local_idx]
                    full_idx = torch.tensor(full_idx_np, dtype=torch.long, device=device)
                    correct = correct_tensor.index_select(0, torch.tensor(local_idx, dtype=torch.long, device=device))
                    permuted = permuted_tensor.index_select(0, torch.tensor(local_idx, dtype=torch.long, device=device))
                    interaction_correct = spot_program @ model.gene_encoder(correct).transpose(0, 1)
                    interaction_permuted = spot_program @ model.gene_encoder(permuted).transpose(0, 1)
                    interaction_correct = interaction_correct / math.sqrt(float(model.program_dim))
                    interaction_permuted = interaction_permuted / math.sqrt(float(model.program_dim))
                    bias_correct = model.gene_bias(correct).transpose(0, 1)
                    bias_permuted = model.gene_bias(permuted).transpose(0, 1)
                    predictions = {
                        "bias_correct__interaction_correct": F.softplus(interaction_correct + bias_correct),
                        "bias_correct__interaction_permuted": F.softplus(interaction_permuted + bias_correct),
                        "bias_permuted__interaction_correct": F.softplus(interaction_correct + bias_permuted),
                        "bias_permuted__interaction_permuted": F.softplus(interaction_permuted + bias_permuted),
                        # Post-hoc pathway counterfactuals from the same frozen full decoder.
                        # The first uses no tissue/spot information; neither is an
                        # independently trained baseline.
                        "bias_correct__interaction_zero": F.softplus(bias_correct).expand_as(interaction_correct),
                        "bias_zero__interaction_correct": F.softplus(interaction_correct),
                    }
                    true = base.tensor_to_numpy(y.index_select(1, full_idx))
                    for condition, prediction in predictions.items():
                        pred = base.tensor_to_numpy(prediction)
                        local_moments[condition].update(local_idx, pred, true)
                        global_moments[condition].update(local_idx, pred, true)
            for condition, moments in local_moments.items():
                metrics = section_metrics(moments)
                section_rows.append(
                    {
                        "cohort": args.cohort,
                        "seed": args.seed,
                        "condition": condition,
                        "sample": sample,
                        "individual": individual_map[sample],
                        **metrics,
                    }
                )
            print(f"[branch intervention] {section_number}/{len(samples)} {sample}", flush=True)

    section_frame = pd.DataFrame(section_rows)
    run_rows: list[dict[str, object]] = []
    for condition, moments in global_moments.items():
        condition_sections = section_frame.loc[section_frame["condition"] == condition]
        individual = condition_sections.groupby("individual", sort=True).agg(
            mean_within_section_gene_pcc=("mean_within_section_gene_pcc", "mean"),
            centered_full_matrix_pcc=("centered_full_matrix_pcc", "mean"),
            section_centered_mse=("section_centered_mse", "mean"),
        )
        primary_centered_mse = float(individual["section_centered_mse"].mean())
        metrics = raw_metrics(moments)
        run_rows.append(
            {
                "cohort": args.cohort,
                "seed": args.seed,
                "condition": condition,
                **metrics,
                "primary_mean_within_section_gene_pcc": float(individual["mean_within_section_gene_pcc"].mean()),
                "primary_centered_full_matrix_pcc": float(individual["centered_full_matrix_pcc"].mean()),
                "primary_section_centered_rmse": math.sqrt(max(primary_centered_mse, 0.0)),
            }
        )
    run_frame = pd.DataFrame(run_rows)

    # The untouched condition must reproduce the frozen clean-partition export.
    if args.max_sections is None and args.max_genes is None:
        stored_path = (
            ROOT
            / "experiments/multicohort_geneheldout_decima_clean_split/components/runs"
            / args.cohort
            / f"seed_{args.seed}/decima/summary.json"
        )
        stored = json.loads(stored_path.read_text())
        untouched = run_frame.loc[run_frame["condition"] == CONDITIONS[0]].iloc[0]
        for current_name, stored_name in (
            ("full_matrix_pcc", "pooled_pcc"),
            ("full_matrix_mse", "overall_mse"),
            ("abundance_pcc", "abundance_pcc"),
            ("abundance_rmse", "abundance_rmse"),
            ("mean_gene_pcc", "mean_gene_pcc"),
            ("centered_rmse", "centered_rmse"),
        ):
            if not np.isclose(float(untouched[current_name]), float(stored[stored_name]), atol=3e-6, rtol=0):
                raise ValueError(f"Untouched condition mismatch for {current_name}")

    output = OUTPUT_ROOT / "runs" / args.cohort / f"seed_{args.seed}"
    output.mkdir(parents=True, exist_ok=True)
    run_frame.to_csv(output / "per_run.tsv", sep="\t", index=False)
    section_frame.to_csv(output / "per_section.tsv", sep="\t", index=False)
    metadata = {
        "cohort": args.cohort,
        "seed": args.seed,
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "checkpoint_sha256": sha256(checkpoint),
        "gene_split": str(base.GENE_SPLIT_DIR.relative_to(ROOT)),
        "permutation_seed": args.seed + 54321,
        "permutation_scope": "held-out genes within cohort",
        "conditions": list(CONDITIONS),
        "interpretive_boundary": "frozen inference-time pathway intervention; not branch-specific retraining or causal mediation",
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(run_frame.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
