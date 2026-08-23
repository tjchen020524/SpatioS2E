#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
import yaml

from spatios2e.data.dataset import load_expression_split
from spatios2e.models.decima_wrapper import DecimaSequenceWrapper
from spatios2e.models.residual_model import EPS


def load_config(path: Path) -> Dict:
    return yaml.safe_load(path.read_text())


def resolve_gene_ids(cfg: Dict) -> list[str]:
    gene_list_path = Path(cfg["paths"]["gene_split_dir"]) / "train_genes.txt"
    gene_ids = [line.strip() for line in gene_list_path.read_text().splitlines() if line.strip()]
    if not gene_ids:
        raise RuntimeError(f"No gene IDs found in {gene_list_path}")
    return gene_ids


def compute_log_mu_base(cfg: Dict, gene_ids: list[str], device: torch.device, chunk_size: int) -> np.ndarray:
    decima = DecimaSequenceWrapper(
        ckpt_path=Path(cfg["paths"]["decima_ckpt"]),
        h5_path=Path(cfg["paths"]["decima_h5"]) if cfg["paths"].get("decima_h5") else None,
        npz_dir=Path(cfg["paths"]["decima_npz_dir"]) if cfg["paths"].get("decima_npz_dir") else None,
        freeze_backbone=True,
        freeze_head=bool(cfg["model"].get("freeze_pseudobulk_head", False)),
    )
    decima = decima.to(device)
    decima.eval()

    chunks = []
    with torch.no_grad():
        for start in range(0, len(gene_ids), chunk_size):
            gids = gene_ids[start : start + chunk_size]
            e_gene = decima.encode_genes(gids, device=device)
            mu_pb, _ = decima.forward_pseudobulk(e_gene)
            chunks.append(torch.log(mu_pb + EPS).detach().cpu().float().numpy())
    return np.concatenate(chunks, axis=0).astype(np.float32, copy=False)


def load_aligned_training_matrix(cfg: Dict, gene_ids: list[str], sample: str) -> np.ndarray:
    """Load one training section and align it to the configured gene order."""

    expr_root = Path(cfg["paths"]["expression_root"])
    expr = load_expression_split(expr_root / sample / "train.npz", to_dense=True)
    sample_gene_ids = expr["gene_ids"]
    if sample_gene_ids != gene_ids:
        order = {gid: i for i, gid in enumerate(sample_gene_ids)}
        missing = [gid for gid in gene_ids if gid not in order]
        if missing:
            preview = ", ".join(missing[:5])
            raise KeyError(f"Training section {sample} is missing {len(missing)} genes, e.g. {preview}")
        indices = [order[gid] for gid in gene_ids]
        return np.asarray(expr["matrix"], dtype=np.float32)[:, indices]
    return np.asarray(expr["matrix"], dtype=np.float32)


def compute_training_tissue_mean(cfg: Dict, gene_ids: list[str]) -> tuple[np.ndarray, int]:
    """Compute the spot-weighted gene mean using training sections only."""

    total = np.zeros(len(gene_ids), dtype=np.float64)
    n_spots = 0
    for sample in cfg["split"]["train"]:
        values = load_aligned_training_matrix(cfg, gene_ids, sample)
        total += values.sum(axis=0, dtype=np.float64)
        n_spots += int(values.shape[0])
    if n_spots <= 0:
        raise RuntimeError("No training spots found while computing the tissue mean")
    return (total / float(n_spots)).astype(np.float32), n_spots


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Compute train-derived gene residual scales for SpatioS2E.")
    parser.add_argument("config", type=Path)
    args = parser.parse_args()

    cfg_path = args.config
    cfg = load_config(cfg_path)

    out_path = Path(cfg["paths"]["residual_scale_npz"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    gene_ids = resolve_gene_ids(cfg)
    chunk_size = int(cfg.get("eval", {}).get("max_genes_per_batch", 256))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    abundance_source = str(cfg.get("model", {}).get("abundance_source", "decima")).lower()
    if abundance_source in {"training_tissue_mean", "training_mean", "tissue_mean"}:
        log_mu_base, expected_n_spots = compute_training_tissue_mean(cfg, gene_ids)
        abundance_source = "training_tissue_mean"
    elif abundance_source in {"decima", "sequence", "sequence_head"}:
        log_mu_base = compute_log_mu_base(cfg, gene_ids, device=device, chunk_size=chunk_size)
        expected_n_spots = None
        abundance_source = "decima"
    else:
        raise ValueError(f"Unsupported model.abundance_source={abundance_source!r}")

    n_gene = len(gene_ids)
    sum_resid = np.zeros(n_gene, dtype=np.float64)
    sumsq_resid = np.zeros(n_gene, dtype=np.float64)
    n_spots = 0
    for sample in cfg["split"]["train"]:
        y = load_aligned_training_matrix(cfg, gene_ids, sample)
        resid = y.astype(np.float64, copy=False) - log_mu_base[np.newaxis, :]
        sum_resid += resid.sum(axis=0)
        sumsq_resid += np.square(resid).sum(axis=0)
        n_spots += int(y.shape[0])

    if n_spots <= 0:
        raise RuntimeError("No training spots found while computing residual scale")
    if expected_n_spots is not None and n_spots != expected_n_spots:
        raise RuntimeError("Training spot count changed between abundance and scale passes")

    resid_mean = sum_resid / float(n_spots)
    resid_var = np.maximum(sumsq_resid / float(n_spots) - resid_mean * resid_mean, 0.0)
    resid_scale_raw = np.sqrt(resid_var)
    scale_floor = float(cfg["model"].get("residual_scale_floor", 1.0e-3))
    resid_scale = np.maximum(resid_scale_raw, scale_floor)

    np.savez(
        out_path,
        gene_ids=np.asarray(gene_ids, dtype=str),
        residual_scale=resid_scale.astype(np.float32),
        residual_scale_raw=resid_scale_raw.astype(np.float32),
        residual_mean=resid_mean.astype(np.float32),
        log_mu_base=log_mu_base.astype(np.float32),
    )

    ranking_value = cfg.get("eval", {}).get("hvg_ranking_file")
    ranking_path = Path(ranking_value) if ranking_value else out_path.with_suffix(".gene_stats.tsv")
    ranking_path.parent.mkdir(parents=True, exist_ok=True)
    training_mean = log_mu_base.astype(np.float64) + resid_mean
    ranking_order = np.argsort(-resid_scale_raw, kind="stable")
    pd.DataFrame(
        {
            "gene_id": np.asarray(gene_ids, dtype=str)[ranking_order],
            "training_mean": training_mean[ranking_order],
            "training_std": resid_scale_raw[ranking_order],
            "variance_rank": np.arange(1, n_gene + 1),
        }
    ).to_csv(ranking_path, sep="\t", index=False)

    summary = {
        "config": str(cfg_path),
        "output": str(out_path),
        "hvg_ranking_file": str(ranking_path),
        "n_genes": int(n_gene),
        "n_train_samples": int(len(cfg["split"]["train"])),
        "n_train_spots": int(n_spots),
        "abundance_source": abundance_source,
        "residual_scale_mean": float(np.mean(resid_scale)),
        "residual_scale_median": float(np.median(resid_scale)),
        "residual_scale_p90": float(np.quantile(resid_scale, 0.9)),
        "residual_scale_raw_mean": float(np.mean(resid_scale_raw)),
        "residual_scale_floor": scale_floor,
    }
    out_path.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
