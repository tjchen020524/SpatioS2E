#!/usr/bin/env python3
"""Train a section-centred held-out-gene residual decoder.

Targets and predictions are centred exactly within each tissue section and the
decoder has a signed linear output.  The task therefore removes gene abundance
by construction and asks whether a gene representation transfers within-section
variation.  Checkpoints are selected only with residual-specific validation
endpoints.
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
from experiments.multicohort_geneheldout_decima.run_cohort_geneheldout import (
    configure_base,
)
from experiments.multicohort_geneheldout_decima.evaluate_section_centered import (
    VectorMoments,
    load_individual_map,
)
from experiments.multicohort_geneheldout_decima_clean_split.clean_split_config import (
    CLEAN_ROOT,
    COHORTS,
    SEEDS,
    VARIANTS,
)


ROOT = DATA_ROOT
AUDIT_ROOT = ROOT / "experiments/heldout_decoder_architecture_audit"
EXPERIMENT_ROOT = AUDIT_ROOT / "residual_only"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class ResidualProgramDecoder(torch.nn.Module):
    def __init__(
        self,
        spot_dim: int,
        gene_dim: int,
        hidden_dim: int = 512,
        program_dim: int = 96,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.spot_encoder = torch.nn.Sequential(
            torch.nn.LayerNorm(spot_dim),
            torch.nn.Linear(spot_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, program_dim),
        )
        self.gene_encoder = torch.nn.Sequential(
            torch.nn.LayerNorm(gene_dim),
            torch.nn.Linear(gene_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, program_dim),
        )
        self.program_dim = program_dim

    def forward(self, x: torch.Tensor, gene_emb: torch.Tensor) -> torch.Tensor:
        spot_program = self.spot_encoder(x)
        gene_program = self.gene_encoder(gene_emb)
        raw = spot_program @ gene_program.transpose(0, 1)
        return raw / math.sqrt(float(self.program_dim))


class SectionStore:
    def __init__(self, manifest: pd.DataFrame, gene_ids: list[str]) -> None:
        self.manifest = manifest.copy()
        self.feature_store = base.FeatureStore()
        self.expression_store = base.ExpressionStore(gene_ids)
        self.by_sample = {
            str(sample): group.copy()
            for sample, group in self.manifest.groupby("sample", sort=False)
        }

    def samples(self, split: str, limit: int | None = None) -> list[str]:
        values = sorted(
            self.manifest.loc[self.manifest["split"] == split, "sample"]
            .astype(str)
            .unique()
            .tolist()
        )
        return values if limit is None else values[:limit]

    def x(self, sample: str) -> torch.Tensor:
        group = self.by_sample[sample]
        values = np.stack(
            [self.feature_store.get_row(sample, str(barcode)) for barcode in group["barcode"]]
        ).astype(np.float32, copy=False)
        return base.numpy_to_tensor(values)

    def y(self, sample: str, gene_indices: np.ndarray) -> torch.Tensor:
        group = self.by_sample[sample]
        splits = group["split"].astype(str).unique().tolist()
        if len(splits) != 1:
            raise ValueError(f"Section {sample} spans multiple splits: {splits}")
        matrix, barcode_to_idx = self.expression_store._load(sample, splits[0])
        rows = np.asarray(
            [barcode_to_idx[str(barcode)] for barcode in group["barcode"]],
            dtype=np.int64,
        )
        values = matrix[rows][:, gene_indices].toarray().astype(np.float32, copy=False)
        return base.numpy_to_tensor(values)


def configure_audit(cohort: str, seed: int, variant: str) -> None:
    configure_base(cohort, seed, variant)
    base.GENE_SPLIT_DIR = CLEAN_ROOT / "data" / cohort / "gene_splits"


def embedding_matrix(
    variant: str,
    gene_ids: list[str],
    train_idx: np.ndarray,
    seed: int,
) -> np.ndarray:
    pretrained = base.load_decima_embeddings(gene_ids, train_idx)
    if variant == "decima":
        return pretrained
    if variant == "random":
        return base.load_random_embeddings(pretrained.shape, seed + 12345)
    if variant == "constant":
        return base.load_constant_embeddings(pretrained.shape)
    raise ValueError(variant)


def finite_gene_metrics(
    moments: VectorMoments,
) -> tuple[float, float, int, float, float, int]:
    derived = moments.derived()
    correlations = np.asarray(derived["correlation"], dtype=np.float64)
    eligible = np.asarray(derived["gene_pcc_eligible"], dtype=bool)
    eligible_correlations = correlations[eligible]
    count = int(derived["total_count"])
    mse = float(derived["total_sse"]) / count
    return (
        float(np.mean(eligible_correlations)) if eligible_correlations.size else float("nan"),
        mse,
        int(eligible_correlations.size),
        float(np.nanmax(np.abs(np.asarray(derived["mean_pred"], dtype=np.float64)))),
        float(derived["centered_full_matrix_pcc"]),
        int(derived["n_constant_prediction_gene_pcc"]),
    )


@torch.no_grad()
def evaluate(
    model: ResidualProgramDecoder,
    store: SectionStore,
    samples: list[str],
    gene_indices: np.ndarray,
    gene_tensor: torch.Tensor,
    device: torch.device,
    gene_chunk_size: int,
    individual_map: dict[str, str] | None = None,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    model.eval()
    chunks = base.make_gene_chunks(len(gene_indices), gene_chunk_size)
    section_rows: list[dict[str, object]] = []
    for section_number, sample in enumerate(samples, start=1):
        x = store.x(sample).to(device, non_blocking=True)
        moments = VectorMoments.zeros(len(gene_indices))
        true_mean_max = 0.0
        pred_mean_max = 0.0
        for local_idx in chunks:
            full_idx_np = gene_indices[local_idx]
            full_idx = torch.tensor(full_idx_np, dtype=torch.long, device=device)
            true = store.y(sample, full_idx_np).to(device, non_blocking=True)
            true = true - true.mean(dim=0, keepdim=True)
            pred = model(x, gene_tensor.index_select(0, full_idx))
            pred = pred - pred.mean(dim=0, keepdim=True)
            true_mean_max = max(true_mean_max, float(true.mean(dim=0).abs().max().cpu()))
            pred_mean_max = max(pred_mean_max, float(pred.mean(dim=0).abs().max().cpu()))
            moments.update(local_idx, base.tensor_to_numpy(pred), base.tensor_to_numpy(true))
        (
            mean_pcc,
            mse,
            n_eligible,
            moment_mean_max,
            centered_full_matrix_pcc,
            n_constant_prediction,
        ) = finite_gene_metrics(moments)
        section_rows.append(
            {
                "sample": sample,
                "individual": individual_map[sample] if individual_map is not None else sample,
                "n_spots": int(x.shape[0]),
                "n_genes": int(len(gene_indices)),
                "n_eligible_gene_pcc": n_eligible,
                "n_finite_gene_pcc": n_eligible,
                "n_constant_prediction_gene_pcc": n_constant_prediction,
                "mean_within_section_gene_pcc": mean_pcc,
                "centered_full_matrix_pcc": centered_full_matrix_pcc,
                "section_centered_mse": mse,
                "section_centered_rmse": math.sqrt(max(mse, 0.0)),
                "max_abs_true_section_mean": true_mean_max,
                "max_abs_pred_section_mean": max(pred_mean_max, moment_mean_max),
            }
        )
        print(
            f"[residual eval] section={section_number}/{len(samples)} sample={sample}",
            flush=True,
        )
    section_frame = pd.DataFrame(section_rows)
    individual_rows: list[dict[str, object]] = []
    for individual, group in section_frame.groupby("individual", sort=True):
        individual_rows.append(
            {
                "individual": individual,
                "n_sections": int(len(group)),
                "mean_within_section_gene_pcc": float(group["mean_within_section_gene_pcc"].mean()),
                "centered_full_matrix_pcc": float(group["centered_full_matrix_pcc"].mean()),
                "section_centered_mse": float(group["section_centered_mse"].mean()),
            }
        )
    individual_frame = pd.DataFrame(individual_rows)
    primary_mse = float(individual_frame["section_centered_mse"].mean())
    summary = {
        "n_sections": int(len(section_frame)),
        "n_individuals": int(len(individual_frame)),
        "n_genes": int(len(gene_indices)),
        "primary_aggregation": "mean sections within individual, then equal-weight mean across individuals",
        "primary_mean_within_section_gene_pcc": float(individual_frame["mean_within_section_gene_pcc"].mean()),
        "primary_centered_full_matrix_pcc": float(individual_frame["centered_full_matrix_pcc"].mean()),
        "primary_section_centered_mse": primary_mse,
        "primary_section_centered_rmse": math.sqrt(max(primary_mse, 0.0)),
        "max_abs_true_section_mean": float(section_frame["max_abs_true_section_mean"].max()),
        "max_abs_pred_section_mean": float(section_frame["max_abs_pred_section_mean"].max()),
    }
    return summary, section_frame, individual_frame


def next_gene_chunk(
    rng: np.random.Generator,
    train_idx: np.ndarray,
    chunk_size: int,
    state: dict[str, object],
) -> np.ndarray:
    chunks = state.get("chunks")
    position = int(state.get("position", 0))
    if chunks is None or position >= len(chunks):
        order = rng.permutation(train_idx)
        chunks = [order[start : start + chunk_size] for start in range(0, len(order), chunk_size)]
        state["chunks"] = chunks
        position = 0
    output = np.sort(chunks[position].astype(np.int64))
    state["position"] = position + 1
    return output


def smoke_test() -> None:
    torch.manual_seed(7)
    model = ResidualProgramDecoder(32, 24, 16, 8, 0.1)
    raw = model(torch.randn(9, 32), torch.randn(11, 24))
    residual = raw - raw.mean(dim=0, keepdim=True)
    assert residual.shape == (9, 11)
    assert float(residual.mean(dim=0).abs().max().detach()) < 1.0e-6
    assert bool((residual < 0).any() and (residual > 0).any())
    print("residual-only model smoke test: PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", choices=COHORTS)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--gene-chunk-size", type=int, default=512)
    parser.add_argument("--val-genes", type=int, default=2048)
    parser.add_argument("--lambda-gene-corr", type=float, default=0.15)
    parser.add_argument("--selection-mse-weight", type=float, default=0.02)
    parser.add_argument("--lr", type=float, default=2.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--max-train-sections", type=int)
    parser.add_argument("--max-val-sections", type=int)
    parser.add_argument("--max-test-sections", type=int)
    parser.add_argument("--max-train-genes", type=int)
    parser.add_argument("--max-heldout-genes", type=int)
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    if args.smoke_test:
        smoke_test()
        return
    if args.cohort is None or args.seed is None or args.variant is None:
        parser.error("--cohort, --seed and --variant are required")

    configure_audit(args.cohort, args.seed, args.variant)
    base.set_seed(args.seed)
    manifest = pd.read_csv(base.DATA_ROOT / "manifest.tsv", sep="\t")
    split = json.loads((base.DATA_ROOT / "split.json").read_text())
    gene_ids = base.load_gene_ids()
    train_idx, heldout_idx = base.load_gene_split(gene_ids)
    if args.max_train_genes is not None:
        train_idx = train_idx[: args.max_train_genes]
    if args.max_heldout_genes is not None:
        heldout_idx = heldout_idx[: args.max_heldout_genes]
    embeddings = embedding_matrix(args.variant, gene_ids, train_idx, args.seed)
    gene_tensor = base.numpy_to_tensor(embeddings).to("cuda" if torch.cuda.is_available() else "cpu")
    device = gene_tensor.device
    store = SectionStore(manifest, gene_ids)
    train_samples = store.samples("train", args.max_train_sections)
    val_samples = store.samples("val", args.max_val_sections)
    test_samples = store.samples("test", args.max_test_sections)
    rng = np.random.default_rng(args.seed)
    val_gene_idx = np.sort(
        rng.choice(train_idx, size=min(args.val_genes, len(train_idx)), replace=False).astype(np.int64)
    )
    model = ResidualProgramDecoder(
        spot_dim=int(split["embedding_dim"]),
        gene_dim=int(embeddings.shape[1]),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    output = EXPERIMENT_ROOT / "runs" / args.cohort / f"seed_{args.seed}" / args.variant
    checkpoint_dir = output / "checkpoints"
    result_dir = output / "results"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    if not args.evaluate_only:
        best_score = -float("inf")
        history: list[dict[str, object]] = []
        gene_state: dict[str, object] = {}
        for epoch in range(1, args.epochs + 1):
            model.train()
            losses: list[float] = []
            mses: list[float] = []
            corr_losses: list[float] = []
            for section_number in rng.permutation(len(train_samples)):
                sample = train_samples[int(section_number)]
                gene_idx_np = next_gene_chunk(rng, train_idx, args.gene_chunk_size, gene_state)
                gene_idx = torch.tensor(gene_idx_np, dtype=torch.long, device=device)
                x = store.x(sample).to(device, non_blocking=True)
                true = store.y(sample, gene_idx_np).to(device, non_blocking=True)
                true = true - true.mean(dim=0, keepdim=True)
                pred = model(x, gene_tensor.index_select(0, gene_idx))
                pred = pred - pred.mean(dim=0, keepdim=True)
                mse = F.mse_loss(pred, true)
                corr = base.corr_loss(pred, true)
                loss = mse + args.lambda_gene_corr * corr
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
                mses.append(float(mse.detach().cpu()))
                corr_losses.append(float(corr.detach().cpu()))

            val_summary, _, _ = evaluate(
                model,
                store,
                val_samples,
                val_gene_idx,
                gene_tensor,
                device,
                args.gene_chunk_size,
            )
            score = float(val_summary["primary_mean_within_section_gene_pcc"]) - args.selection_mse_weight * float(
                val_summary["primary_section_centered_mse"]
            )
            record = {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "train_mse": float(np.mean(mses)),
                "train_corr_loss": float(np.mean(corr_losses)),
                "validation": val_summary,
                "score": score,
            }
            history.append(record)
            print(f"[residual epoch {epoch}] {json.dumps(record)}", flush=True)
            payload = {"model": model.state_dict(), "epoch": epoch, "score": score, "validation": val_summary}
            torch.save(payload, checkpoint_dir / "last.pt")
            if score > best_score:
                best_score = score
                torch.save(payload, checkpoint_dir / "best.pt")

        (result_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    checkpoint = torch.load(checkpoint_dir / "best.pt", map_location=device)
    model.load_state_dict(checkpoint["model"])
    test_summary, section_frame, individual_frame = evaluate(
        model,
        store,
        test_samples,
        heldout_idx,
        gene_tensor,
        device,
        args.gene_chunk_size,
        load_individual_map(args.cohort),
    )
    section_frame.insert(0, "variant", args.variant)
    section_frame.insert(0, "seed", args.seed)
    section_frame.insert(0, "cohort", args.cohort)
    individual_frame.insert(0, "variant", args.variant)
    individual_frame.insert(0, "seed", args.seed)
    individual_frame.insert(0, "cohort", args.cohort)
    section_frame.to_csv(result_dir / "per_section.tsv", sep="\t", index=False)
    individual_frame.to_csv(result_dir / "per_individual.tsv", sep="\t", index=False)
    summary = {
        "cohort": args.cohort,
        "seed": args.seed,
        "variant": args.variant,
        "task": "section-centred signed residual prediction",
        "architecture": "bias-free bilinear interaction with exact section centring",
        "best_epoch": int(checkpoint["epoch"]),
        "best_score": float(checkpoint["score"]),
        "validation": checkpoint["validation"],
        "heldout_test": test_summary,
        "checkpoint": str((checkpoint_dir / "best.pt").relative_to(ROOT)),
        "checkpoint_sha256": sha256(checkpoint_dir / "best.pt"),
        "gene_split": str(base.GENE_SPLIT_DIR.relative_to(ROOT)),
        "gene_split_train_sha256": sha256(base.GENE_SPLIT_DIR / "train_genes.txt"),
        "gene_split_heldout_sha256": sha256(base.GENE_SPLIT_DIR / "heldout_genes.txt"),
        "protocol": {
            "epochs": args.epochs,
            "gene_chunk_size": args.gene_chunk_size,
            "val_genes": args.val_genes,
            "lambda_gene_corr": args.lambda_gene_corr,
            "selection_mse_weight": args.selection_mse_weight,
            "selection": "mean within-section gene PCC - weight * section-centred MSE",
            "softplus": False,
            "target_centring": "exact within section",
            "prediction_centring": "exact within section",
        },
    }
    (result_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
