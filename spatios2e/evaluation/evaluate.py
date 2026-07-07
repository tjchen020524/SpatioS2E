#!/usr/bin/env python3
"""Evaluate a SpatioS2E checkpoint on held-out spatial transcriptomics samples."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch

from spatios2e.data.dataset import SpatioS2EDataset, collate_graphs
from spatios2e.models.factory import build_model


def to_dense(y: torch.Tensor) -> torch.Tensor:
    return y.to_dense() if y.is_sparse else y


def forward_model(
    model,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    gene_ids: Sequence[str],
    sample_name: str,
) -> Dict[str, torch.Tensor]:
    if getattr(model, "requires_sample_name", False):
        return model(x, edge_index, edge_weight, gene_ids, sample_name=sample_name)
    return model(x, edge_index, edge_weight, gene_ids)


def load_config(path: Path) -> Dict:
    if path.suffix in {".yaml", ".yml"}:
        import yaml  # type: ignore

        with path.open("r") as handle:
            return yaml.safe_load(handle)
    with path.open("r") as handle:
        return json.load(handle)


def build_dataset(cfg: Dict, split: str, samples: Sequence[str] | None = None) -> SpatioS2EDataset:
    gene_list_path = Path(cfg["paths"]["gene_split_dir"]) / f"{split}_genes.txt"
    gene_filter = None
    if gene_list_path.exists():
        gene_filter = [line.strip() for line in gene_list_path.read_text().splitlines() if line.strip()]

    ds = SpatioS2EDataset(
        split=split,
        samples=samples,
        expression_root=Path(cfg["paths"]["expression_root"]),
        multimodal_root=Path(cfg["paths"]["multimodal_root"]),
        graph_root=Path(cfg["paths"]["graph_root"]),
        targets_dense=True,
        cache=False,
        normalize_modalities=bool(cfg.get("normalize_modalities", True)),
        modality_stats_path=Path(cfg["paths"]["modality_stats"]),
        gene_filter=set(gene_filter) if gene_filter else None,
    )
    return ds


@dataclass
class RunningStats:
    count: int = 0
    sum_pred: float = 0.0
    sum_true: float = 0.0
    sum_pred2: float = 0.0
    sum_true2: float = 0.0
    sum_pred_true: float = 0.0
    sum_sq_error: float = 0.0

    def update(self, pred, true) -> None:
        if not isinstance(pred, torch.Tensor):
            pred = torch.tensor(np.asarray(pred), dtype=torch.float64)
        else:
            pred = pred.detach().to(dtype=torch.float64)

        if not isinstance(true, torch.Tensor):
            true = torch.tensor(np.asarray(true), dtype=torch.float64)
        else:
            true = true.detach().to(dtype=torch.float64)

        mask = torch.isfinite(pred) & torch.isfinite(true)
        if not torch.any(mask):
            return

        p = pred[mask]
        t = true[mask]
        self.count += int(p.numel())
        self.sum_pred += float(p.sum().item())
        self.sum_true += float(t.sum().item())
        self.sum_pred2 += float((p * p).sum().item())
        self.sum_true2 += float((t * t).sum().item())
        self.sum_pred_true += float((p * t).sum().item())
        diff = p - t
        self.sum_sq_error += float((diff * diff).sum().item())

    def mse(self) -> float:
        return self.sum_sq_error / self.count if self.count else float("nan")

    def corr(self) -> float:
        if self.count <= 1:
            return float("nan")
        mean_p = self.sum_pred / self.count
        mean_t = self.sum_true / self.count
        var_p = self.sum_pred2 / self.count - mean_p * mean_p
        var_t = self.sum_true2 / self.count - mean_t * mean_t
        if var_p <= 0 or var_t <= 0:
            return float("nan")
        cov = self.sum_pred_true / self.count - mean_p * mean_t
        return float(cov / np.sqrt(var_p * var_t))


def _resolve_max_genes(cfg: Dict, override: Optional[int]) -> Optional[int]:
    if override is not None:
        return int(override)
    eval_cfg = cfg.get("eval", {})
    if eval_cfg.get("max_genes_per_batch") is not None:
        return int(eval_cfg["max_genes_per_batch"])
    optim_cfg = cfg.get("optim", {})
    if optim_cfg.get("max_genes_per_batch") is not None:
        return int(optim_cfg["max_genes_per_batch"])
    return None


def eval_split(
    cfg_path: Path,
    ckpt_path: Path,
    split: str,
    save_dir: Path,
    max_genes_per_batch: Optional[int],
    gene_subset: Optional[List[str]] = None,
    no_graph_override: bool = False,
) -> None:
    cfg = load_config(cfg_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    samples_cfg = cfg.get("split", {})
    samples = samples_cfg.get(split) if isinstance(samples_cfg, dict) else None
    if samples == []:
        samples = None

    ds = build_dataset(cfg, split=split, samples=samples)
    if len(ds) == 0:
        raise SystemExit(f"No samples found for split {split}")
    sample_batch = collate_graphs([ds[0]])
    input_dim = sample_batch["x"][0].shape[1]

    model = build_model(cfg, input_dim=input_dim).to(device)

    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()
    no_graph = bool(no_graph_override or state.get("no_graph", False) or cfg.get("model", {}).get("no_graph", False))

    overall_stats = RunningStats()
    sample_stats: Dict[str, RunningStats] = {}
    gene_stats: Dict[str, RunningStats] = {}

    with torch.no_grad():
        for idx in range(len(ds)):
            item = ds[idx]
            batch = collate_graphs([item])
            x = batch["x"][0].to(device)
            edge_index = batch["edge_index"][0].to(device)
            edge_weight = batch["edge_weight"][0].to(device)
            if no_graph:
                edge_weight = torch.zeros_like(edge_weight)
            y_spot = to_dense(batch["y"][0]).to(device)
            gene_ids_full = batch["gene_ids"][0]
            sample_name = batch["sample"][0]
            if gene_subset:
                subset_set = set(gene_subset)
                idx_keep = [i for i, gid in enumerate(gene_ids_full) if gid in subset_set]
                if not idx_keep:
                    continue
                gene_ids_full = [gene_ids_full[i] for i in idx_keep]
                y_spot = y_spot[:, idx_keep]
            n_genes = len(gene_ids_full)
            sample_stats.setdefault(sample_name, RunningStats())

            if max_genes_per_batch is None or max_genes_per_batch <= 0:
                ranges = [(0, n_genes)]
            else:
                ranges = [(s, min(s + max_genes_per_batch, n_genes)) for s in range(0, n_genes, max_genes_per_batch)]

            for start, end in ranges:
                gene_ids = gene_ids_full[start:end]
                y_chunk = y_spot[:, start:end]
                out = forward_model(model, x, edge_index, edge_weight, gene_ids, sample_name=sample_name)
                pred = out["log_mu"]
                pred_cpu = pred.detach().cpu()
                true_cpu = y_chunk.detach().cpu()

                overall_stats.update(pred_cpu.reshape(-1), true_cpu.reshape(-1))
                sample_stats[sample_name].update(pred_cpu.reshape(-1), true_cpu.reshape(-1))
                for j, gid in enumerate(gene_ids):
                    gene_stats.setdefault(gid, RunningStats()).update(pred_cpu[:, j], true_cpu[:, j])

    if not overall_stats.count:
        raise SystemExit("No samples evaluated (gene filter removed all genes).")

    overall_mse = overall_stats.mse()
    overall_corr = overall_stats.corr()

    gene_rows = [{"gene_id": gid, "mse": stats.mse(), "corr": stats.corr()} for gid, stats in gene_stats.items()]
    sample_rows = [{"sample": name, "mse": stats.mse(), "corr": stats.corr()} for name, stats in sample_stats.items()]

    gene_corr_vals = [row["corr"] for row in gene_rows if np.isfinite(row["corr"])]
    gene_corr_mean = float(np.mean(gene_corr_vals)) if gene_corr_vals else float("nan")
    gene_corr_median = float(np.median(gene_corr_vals)) if gene_corr_vals else float("nan")

    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / f"{split}_overall.json").write_text(
        json.dumps(
            {
                "split": split,
                "mse": overall_mse,
                "corr": overall_corr,
                "gene_corr_mean": gene_corr_mean,
                "gene_corr_median": gene_corr_median,
                "no_graph": no_graph,
            },
            indent=2,
        )
    )
    pd.DataFrame(sample_rows).to_csv(save_dir / f"{split}_sample_metrics.tsv", sep="\t", index=False)
    pd.DataFrame(gene_rows).to_csv(save_dir / f"{split}_gene_metrics.tsv", sep="\t", index=False)
    print(
        json.dumps(
            {
                "split": split,
                "mse": overall_mse,
                "corr": overall_corr,
                "gene_corr_mean": gene_corr_mean,
                "gene_corr_median": gene_corr_median,
                "no_graph": no_graph,
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument("--max-genes-per-batch", type=int, default=None)
    parser.add_argument("--gene-subset-file", type=Path, default=None)
    parser.add_argument("--no-graph", action="store_true", help="Disable spatial graph message passing by zeroing edge weights.")
    args = parser.parse_args()
    cfg = load_config(args.config)
    max_genes = _resolve_max_genes(cfg, args.max_genes_per_batch)
    gene_subset = None
    if args.gene_subset_file is not None:
        gene_subset = [line.strip() for line in args.gene_subset_file.read_text().splitlines() if line.strip()]
    eval_split(args.config, args.ckpt, args.split, args.save_dir, max_genes, gene_subset, no_graph_override=bool(args.no_graph))


if __name__ == "__main__":
    main()
