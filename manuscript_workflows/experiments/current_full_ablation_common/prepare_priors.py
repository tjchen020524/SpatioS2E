#!/usr/bin/env python3
"""Compute train-only Decima, gene-mean, residual-scale, and HVG artifacts."""

from __future__ import annotations
from workflow_paths import DATA_ROOT

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = DATA_ROOT
for path in (ROOT, ROOT / "scripts", ROOT / "decima/src"):
    if str(path) not in sys.path:
        sys.path.append(str(path))

from gnn_dataloader import load_expression_split  # noqa: E402
from models.decima_wrapper import DecimaSequenceWrapper  # noqa: E402
from models.stage0_reg_model_v3 import EPS  # noqa: E402


def resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def compute_decima_baseline(
    gene_ids: list[str], paths: dict, device: torch.device, chunk_size: int
) -> np.ndarray:
    model = DecimaSequenceWrapper(
        ckpt_path=resolve(paths["decima_ckpt"]),
        h5_path=resolve(paths["decima_h5"]) if paths.get("decima_h5") else None,
        npz_dir=resolve(paths["decima_npz_dir"]) if paths.get("decima_npz_dir") else None,
        freeze_backbone=True,
        freeze_head=False,
    ).to(device)
    model.eval()
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(gene_ids), chunk_size):
            chunk = gene_ids[start : start + chunk_size]
            embedding = model.encode_genes(chunk, device=device)
            mu, _ = model.forward_pseudobulk(embedding)
            chunks.append(torch.log(mu + EPS).detach().cpu().float().numpy())
    return np.concatenate(chunks).astype(np.float32, copy=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-dir", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cohort_dir = resolve(args.cohort_dir)
    paths = load_json(cohort_dir / "data/paths.json")
    split = load_json(cohort_dir / "data/split.json")
    gene_split_dir = resolve(paths["gene_split_dir"])
    expression_root = resolve(paths["expression_root"])
    gene_ids = [
        line.strip()
        for line in (gene_split_dir / "train_genes.txt").read_text().splitlines()
        if line.strip()
    ]
    if not gene_ids:
        raise RuntimeError("The cohort gene list is empty")

    artifacts = cohort_dir / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    decima_path = artifacts / "train_decima_prior.npz"
    mean_path = artifacts / "train_gene_mean_prior.npz"
    stats_path = artifacts / "train_gene_stats.tsv"
    summary_path = artifacts / "train_priors.summary.json"
    expected = (decima_path, mean_path, stats_path, summary_path)
    if all(path.exists() and path.stat().st_size > 0 for path in expected) and not args.overwrite:
        print(json.dumps({"status": "already_complete", "cohort_dir": str(cohort_dir)}, indent=2))
        return

    n_genes = len(gene_ids)
    sum_y = np.zeros(n_genes, dtype=np.float64)
    sumsq_y = np.zeros(n_genes, dtype=np.float64)
    n_spots = 0
    for sample in split["train"]:
        expr = load_expression_split(expression_root / sample / "train.npz", to_dense=True)
        sample_gene_ids = list(expr["gene_ids"])
        matrix = np.asarray(expr["matrix"], dtype=np.float32)
        if sample_gene_ids != gene_ids:
            index = {gene_id: idx for idx, gene_id in enumerate(sample_gene_ids)}
            missing = [gene_id for gene_id in gene_ids if gene_id not in index]
            if missing:
                raise KeyError(f"{sample} is missing {len(missing)} cohort genes")
            matrix = matrix[:, [index[gene_id] for gene_id in gene_ids]]
        matrix64 = matrix.astype(np.float64, copy=False)
        sum_y += matrix64.sum(axis=0)
        sumsq_y += np.square(matrix64).sum(axis=0)
        n_spots += int(matrix.shape[0])

    if n_spots < 2:
        raise RuntimeError("Not enough training spots to compute cohort priors")
    train_mean = sum_y / float(n_spots)
    train_var = np.maximum(sumsq_y / float(n_spots) - np.square(train_mean), 0.0)
    train_std = np.sqrt(train_var)
    scale_floor = float(paths.get("residual_scale_floor", 1.0e-3))
    residual_scale = np.maximum(train_std, scale_floor).astype(np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    decima_log_mu = compute_decima_baseline(
        gene_ids, paths=paths, device=device, chunk_size=args.chunk_size
    )
    decima_residual_mean = train_mean - decima_log_mu.astype(np.float64)

    np.savez(
        decima_path,
        gene_ids=np.asarray(gene_ids, dtype=str),
        residual_scale=residual_scale,
        residual_scale_raw=train_std.astype(np.float32),
        residual_mean=decima_residual_mean.astype(np.float32),
        log_mu_base=decima_log_mu,
    )
    np.savez(
        mean_path,
        gene_ids=np.asarray(gene_ids, dtype=str),
        residual_scale=residual_scale,
        residual_scale_raw=train_std.astype(np.float32),
        residual_mean=np.zeros(n_genes, dtype=np.float32),
        log_mu_base=train_mean.astype(np.float32),
    )

    ranking = np.argsort(-train_std, kind="stable")
    with stats_path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["rank", "gene_id", "train_mean", "train_std"])
        for rank, idx in enumerate(ranking, start=1):
            writer.writerow([rank, gene_ids[idx], float(train_mean[idx]), float(train_std[idx])])

    summary = {
        "cohort_dir": str(cohort_dir),
        "n_train_samples": len(split["train"]),
        "n_train_spots": n_spots,
        "n_genes": n_genes,
        "scale_floor": scale_floor,
        "train_mean_mean": float(np.mean(train_mean)),
        "train_std_mean": float(np.mean(train_std)),
        "decima_log_mu_mean": float(np.mean(decima_log_mu)),
        "decima_prior": str(decima_path),
        "gene_mean_prior": str(mean_path),
        "gene_ranking": str(stats_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
