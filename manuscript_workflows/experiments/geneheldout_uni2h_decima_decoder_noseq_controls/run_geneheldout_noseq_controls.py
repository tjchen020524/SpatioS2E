#!/usr/bin/env python3
from __future__ import annotations
from workflow_paths import DATA_ROOT

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.utils._pytree as _torch_pytree
from scipy import sparse


ROOT = DATA_ROOT
EXP_ROOT = ROOT / "experiments/geneheldout_uni2h_decima_decoder_noseq_controls"
DATA_ROOT = ROOT / "experiments/benchmark_uni2h_hippocampus_fullgenes/data"
EXPRESSION_ROOT = ROOT / "experiments/sample_split_fullgenes/expression_full"
EMBEDDING_ROOT = ROOT / "data/processed/histology_embeddings_uni2h"
GENE_SPLIT_DIR = ROOT / "experiments/stpath_seq_mechanism_controls/artifacts"
DECIMA_EMB_CACHE = (
    ROOT
    / "experiments/sample_split_fullgenes_v12_uni2h_direct_seq_residual_no_celltype/artifacts/decima_gene_embeddings_fullgenes.npz"
)


def patch_pytorch_transformers_compat() -> None:
    """Patch a PyTorch 2.1 / recent Transformers pytree API mismatch in decima_env."""
    if hasattr(_torch_pytree, "register_pytree_node"):
        return
    if not hasattr(_torch_pytree, "_register_pytree_node"):
        return

    def register_pytree_node(typ, flatten_fn, unflatten_fn, **kwargs):  # type: ignore[no-untyped-def]
        allowed = {
            key: kwargs[key]
            for key in ("to_dumpable_context", "from_dumpable_context")
            if key in kwargs
        }
        return _torch_pytree._register_pytree_node(typ, flatten_fn, unflatten_fn, **allowed)

    _torch_pytree.register_pytree_node = register_pytree_node  # type: ignore[attr-defined]


patch_pytorch_transformers_compat()


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Cluster-safe torch->numpy conversion for the current environment."""
    return np.asarray(tensor.detach().cpu().float().tolist(), dtype=np.float32)


def numpy_to_tensor(array: np.ndarray) -> torch.Tensor:
    """Avoid torch.from_numpy ABI issues seen on the current cluster stack."""
    return torch.tensor(np.asarray(array, dtype=np.float32), dtype=torch.float32)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = True


@dataclass
class RunningStats:
    count: int = 0
    sum_pred: float = 0.0
    sum_true: float = 0.0
    sum_pred2: float = 0.0
    sum_true2: float = 0.0
    sum_pred_true: float = 0.0
    sum_sq_error: float = 0.0

    def update(self, pred: np.ndarray, true: np.ndarray) -> None:
        pred = np.asarray(pred, dtype=np.float32).reshape(-1)
        true = np.asarray(true, dtype=np.float32).reshape(-1)
        mask = np.isfinite(pred) & np.isfinite(true)
        if not np.any(mask):
            return
        p = pred[mask].astype(np.float64, copy=False)
        t = true[mask].astype(np.float64, copy=False)
        self.count += int(p.size)
        self.sum_pred += float(p.sum())
        self.sum_true += float(t.sum())
        self.sum_pred2 += float((p * p).sum())
        self.sum_true2 += float((t * t).sum())
        self.sum_pred_true += float((p * t).sum())
        self.sum_sq_error += float(((p - t) ** 2).sum())

    def mse(self) -> float:
        return self.sum_sq_error / self.count if self.count else float("nan")

    def corr(self) -> float:
        if self.count <= 1:
            return float("nan")
        mean_pred = self.sum_pred / self.count
        mean_true = self.sum_true / self.count
        var_pred = self.sum_pred2 / self.count - mean_pred * mean_pred
        var_true = self.sum_true2 / self.count - mean_true * mean_true
        if var_pred <= 0.0 or var_true <= 0.0:
            return float("nan")
        cov = self.sum_pred_true / self.count - mean_pred * mean_true
        return float(cov / math.sqrt(var_pred * var_true))


class GeneVectorStats:
    def __init__(self, n_genes: int) -> None:
        self.count = np.zeros(n_genes, dtype=np.int64)
        self.sum_pred = np.zeros(n_genes, dtype=np.float64)
        self.sum_true = np.zeros(n_genes, dtype=np.float64)
        self.sum_pred2 = np.zeros(n_genes, dtype=np.float64)
        self.sum_true2 = np.zeros(n_genes, dtype=np.float64)
        self.sum_pred_true = np.zeros(n_genes, dtype=np.float64)
        self.sum_sq_error = np.zeros(n_genes, dtype=np.float64)

    def update_chunk(self, pred: np.ndarray, true: np.ndarray, local_idx: np.ndarray) -> None:
        mask = np.isfinite(pred) & np.isfinite(true)
        counts = mask.sum(axis=0).astype(np.int64)
        p = np.where(mask, pred, 0.0).astype(np.float64, copy=False)
        t = np.where(mask, true, 0.0).astype(np.float64, copy=False)
        self.count[local_idx] += counts
        self.sum_pred[local_idx] += p.sum(axis=0)
        self.sum_true[local_idx] += t.sum(axis=0)
        self.sum_pred2[local_idx] += (p * p).sum(axis=0)
        self.sum_true2[local_idx] += (t * t).sum(axis=0)
        self.sum_pred_true[local_idx] += (p * t).sum(axis=0)
        self.sum_sq_error[local_idx] += ((p - t) ** 2).sum(axis=0)

    def mse(self) -> np.ndarray:
        out = np.full(self.count.shape, np.nan, dtype=np.float64)
        mask = self.count > 0
        out[mask] = self.sum_sq_error[mask] / self.count[mask]
        return out

    def corr(self) -> np.ndarray:
        out = np.full(self.count.shape, np.nan, dtype=np.float64)
        mask = self.count > 1
        mean_pred = np.zeros_like(self.sum_pred)
        mean_true = np.zeros_like(self.sum_true)
        mean_pred[mask] = self.sum_pred[mask] / self.count[mask]
        mean_true[mask] = self.sum_true[mask] / self.count[mask]
        var_pred = np.zeros_like(self.sum_pred)
        var_true = np.zeros_like(self.sum_true)
        var_pred[mask] = self.sum_pred2[mask] / self.count[mask] - mean_pred[mask] * mean_pred[mask]
        var_true[mask] = self.sum_true2[mask] / self.count[mask] - mean_true[mask] * mean_true[mask]
        good = mask & (var_pred > 0.0) & (var_true > 0.0)
        cov = np.zeros_like(self.sum_pred)
        cov[good] = self.sum_pred_true[good] / self.count[good] - mean_pred[good] * mean_true[good]
        out[good] = cov[good] / np.sqrt(var_pred[good] * var_true[good])
        return out


class FeatureStore:
    def __init__(self) -> None:
        self._cache: dict[str, tuple[np.ndarray, dict[str, int]]] = {}

    def _load(self, sample: str) -> tuple[np.ndarray, dict[str, int]]:
        cached = self._cache.get(sample)
        if cached is not None:
            return cached
        arr = np.load(EMBEDDING_ROOT / sample / "embeddings.npz", allow_pickle=True)
        embeddings = arr["embeddings"].astype(np.float32, copy=False)
        barcodes = arr["barcodes"].astype(str).tolist()
        cached = (embeddings, {bc: i for i, bc in enumerate(barcodes)})
        self._cache[sample] = cached
        return cached

    def get_row(self, sample: str, barcode: str) -> np.ndarray:
        embeddings, barcode_to_idx = self._load(sample)
        return embeddings[barcode_to_idx[barcode]]


class ExpressionStore:
    def __init__(self, gene_ids: list[str]) -> None:
        self.gene_ids = gene_ids
        self._cache: dict[tuple[str, str], tuple[sparse.csr_matrix, dict[str, int]]] = {}

    def _load(self, sample: str, split: str) -> tuple[sparse.csr_matrix, dict[str, int]]:
        key = (sample, split)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        arr = np.load(EXPRESSION_ROOT / sample / f"{split}.npz", allow_pickle=True)
        shape = tuple(arr["shape"].tolist())
        csr = sparse.csr_matrix((arr["data"], arr["indices"], arr["indptr"]), shape=shape)
        gene_ids = arr["gene_ids"].astype(str).tolist()
        if gene_ids != self.gene_ids:
            raise ValueError(f"Gene order mismatch for {sample}/{split}")
        barcodes = arr["barcodes"].astype(str).tolist()
        cached = (csr.T.tocsr(), {bc: i for i, bc in enumerate(barcodes)})
        self._cache[key] = cached
        return cached

    def get_row(self, sample: str, split: str, barcode: str) -> np.ndarray:
        spot_csr, barcode_to_idx = self._load(sample, split)
        return spot_csr.getrow(barcode_to_idx[barcode]).toarray().ravel().astype(np.float32, copy=False)


class SpotDataset(torch.utils.data.Dataset):
    def __init__(self, manifest: pd.DataFrame, feature_store: FeatureStore, expression_store: ExpressionStore) -> None:
        self.manifest = manifest.reset_index(drop=True)
        self.feature_store = feature_store
        self.expression_store = expression_store

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> dict[str, object]:
        row = self.manifest.iloc[idx]
        sample = str(row["sample"])
        split = str(row["split"])
        barcode = str(row.barcode)
        return {
            "x": numpy_to_tensor(self.feature_store.get_row(sample, barcode)),
            "y": numpy_to_tensor(self.expression_store.get_row(sample, split, barcode)),
            "sample": sample,
            "barcode": barcode,
        }


def collate(batch: list[dict[str, object]]) -> dict[str, object]:
    return {
        "x": torch.stack([x["x"] for x in batch]),  # type: ignore[list-item]
        "y": torch.stack([x["y"] for x in batch]),  # type: ignore[list-item]
        "sample": [x["sample"] for x in batch],
        "barcode": [x["barcode"] for x in batch],
    }


class GeneConditionedProgramDecoder(torch.nn.Module):
    def __init__(
        self,
        spot_dim: int,
        gene_dim: int,
        hidden_dim: int,
        program_dim: int,
        dropout: float,
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
        self.gene_bias = torch.nn.Sequential(
            torch.nn.LayerNorm(gene_dim),
            torch.nn.Linear(gene_dim, hidden_dim // 2),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim // 2, 1),
        )
        self.program_dim = program_dim

    def forward(self, x: torch.Tensor, gene_emb: torch.Tensor) -> torch.Tensor:
        spot_program = self.spot_encoder(x)
        gene_program = self.gene_encoder(gene_emb)
        bias = self.gene_bias(gene_emb).transpose(0, 1)
        raw = spot_program @ gene_program.transpose(0, 1) / math.sqrt(float(self.program_dim))
        return F.softplus(raw + bias)


def corr_loss(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    pred_c = pred.float() - pred.float().mean(dim=0, keepdim=True)
    true_c = true.float() - true.float().mean(dim=0, keepdim=True)
    pred_std = torch.sqrt(pred_c.pow(2).mean(dim=0) + 1.0e-6)
    true_std = torch.sqrt(true_c.pow(2).mean(dim=0) + 1.0e-6)
    corr = (pred_c * true_c).mean(dim=0) / (pred_std * true_std)
    return 1.0 - corr.mean()


def make_gene_chunks(n_genes: int, chunk_size: int) -> list[np.ndarray]:
    return [np.arange(start, min(start + chunk_size, n_genes), dtype=np.int64) for start in range(0, n_genes, chunk_size)]


def load_gene_ids() -> list[str]:
    return [line.strip() for line in (DATA_ROOT / "gene_ids.txt").read_text().splitlines() if line.strip()]


def load_gene_split(gene_ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    gene_to_idx = {gid: i for i, gid in enumerate(gene_ids)}
    train_genes = [x.strip() for x in (GENE_SPLIT_DIR / "train_genes.txt").read_text().splitlines() if x.strip()]
    heldout_genes = [x.strip() for x in (GENE_SPLIT_DIR / "heldout_genes.txt").read_text().splitlines() if x.strip()]
    train_idx = np.asarray([gene_to_idx[g] for g in train_genes], dtype=np.int64)
    heldout_idx = np.asarray([gene_to_idx[g] for g in heldout_genes], dtype=np.int64)
    return train_idx, heldout_idx


def load_decima_embeddings(gene_ids: list[str], train_idx: np.ndarray) -> np.ndarray:
    arr = np.load(DECIMA_EMB_CACHE, allow_pickle=False)
    cached_gene_ids = arr["gene_ids"].astype(str).tolist()
    if cached_gene_ids != gene_ids:
        raise ValueError(f"Gene order mismatch in {DECIMA_EMB_CACHE}")
    emb = arr["embeddings"].astype(np.float32, copy=False)
    mean = emb[train_idx].mean(axis=0, keepdims=True)
    std = emb[train_idx].std(axis=0, keepdims=True)
    emb = (emb - mean) / np.maximum(std, 1.0e-6)
    return emb.astype(np.float32, copy=False)


def load_random_embeddings(shape: tuple[int, int], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    emb = rng.standard_normal(size=shape).astype(np.float32)
    return emb


def load_constant_embeddings(shape: tuple[int, int]) -> np.ndarray:
    return np.zeros(shape, dtype=np.float32)


def compute_train_gene_means(train_manifest: pd.DataFrame, expression_store: ExpressionStore, n_genes: int) -> np.ndarray:
    sums = np.zeros(n_genes, dtype=np.float64)
    count = 0
    for (sample, split), group in train_manifest.groupby(["sample", "split"], sort=False):
        spot_csr, barcode_to_idx = expression_store._load(str(sample), str(split))
        row_idx = np.asarray([barcode_to_idx[str(barcode)] for barcode in group["barcode"].tolist()], dtype=np.int64)
        sub = spot_csr[row_idx]
        sums += np.asarray(sub.sum(axis=0)).ravel().astype(np.float64, copy=False)
        count += int(sub.shape[0])
    if count == 0:
        raise ValueError("Cannot compute train gene means from an empty manifest")
    return (sums / count).astype(np.float32)


@torch.no_grad()
def evaluate_gene_mean_baseline(
    loader: torch.utils.data.DataLoader,
    gene_means: np.ndarray,
    gene_indices: np.ndarray,
    gene_ids: list[str],
    device: torch.device,
    gene_chunk_size: int,
    out_dir: Path | None = None,
    label: str = "heldout_test",
) -> dict[str, float]:
    overall = RunningStats()
    stats = GeneVectorStats(len(gene_indices))
    sample_stats: dict[str, RunningStats] = {}
    chunks = make_gene_chunks(len(gene_indices), gene_chunk_size)
    mean_tensor = numpy_to_tensor(gene_means).to(device)
    for batch in loader:
        y = batch["y"].to(device, non_blocking=True)
        pred_full_parts = []
        true_full_parts = []
        batch_size = int(y.shape[0])
        for local_idx in chunks:
            full_idx_np = gene_indices[local_idx]
            full_idx = torch.tensor(full_idx_np, dtype=torch.long, device=device)
            pred = mean_tensor.index_select(0, full_idx).unsqueeze(0).expand(batch_size, -1)
            true = y.index_select(1, full_idx)
            pred_np = tensor_to_numpy(pred)
            true_np = tensor_to_numpy(true)
            overall.update(pred_np, true_np)
            stats.update_chunk(pred_np, true_np, local_idx)
            pred_full_parts.append(pred_np)
            true_full_parts.append(true_np)
        pred_full = np.concatenate(pred_full_parts, axis=1)
        true_full = np.concatenate(true_full_parts, axis=1)
        for sample in sorted(set(batch["sample"])):
            mask = np.asarray([s == sample for s in batch["sample"]])
            sample_stats.setdefault(sample, RunningStats()).update(pred_full[mask], true_full[mask])

    gene_mse = stats.mse()
    gene_corr = stats.corr()
    finite_gene_corr = gene_corr[np.isfinite(gene_corr)]
    metrics = {
        "mse": overall.mse(),
        "corr": overall.corr(),
        "gene_corr_mean": float(np.mean(finite_gene_corr)) if finite_gene_corr.size else float("nan"),
        "gene_corr_median": float(np.median(finite_gene_corr)) if finite_gene_corr.size else float("nan"),
        "n_genes": int(len(gene_indices)),
        "n_finite_gene_corr": int(finite_gene_corr.size),
    }
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{label}_overall.json").write_text(json.dumps(metrics, indent=2))
        pd.DataFrame(
            [
                {"sample": sample, "mse": stat.mse(), "corr": stat.corr()}
                for sample, stat in sorted(sample_stats.items())
            ]
        ).to_csv(out_dir / f"{label}_sample_metrics.tsv", sep="\t", index=False)
        pd.DataFrame(
            [
                {
                    "gene_id": gene_ids[int(full_idx)],
                    "mse": float(mse),
                    "corr": float(corr),
                    "full_gene_index": int(full_idx),
                }
                for full_idx, mse, corr in zip(gene_indices.tolist(), gene_mse.tolist(), gene_corr.tolist())
            ]
        ).to_csv(out_dir / f"{label}_gene_metrics.tsv", sep="\t", index=False)
    return metrics


@torch.no_grad()
def evaluate(
    model: GeneConditionedProgramDecoder,
    loader: torch.utils.data.DataLoader,
    gene_emb: torch.Tensor,
    gene_indices: np.ndarray,
    gene_ids: list[str],
    device: torch.device,
    gene_chunk_size: int,
    out_dir: Path | None = None,
    label: str = "",
) -> dict[str, float]:
    model.eval()
    overall = RunningStats()
    stats = GeneVectorStats(len(gene_indices))
    sample_stats: dict[str, RunningStats] = {}

    chunks = make_gene_chunks(len(gene_indices), gene_chunk_size)
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        pred_full_parts = []
        true_full_parts = []
        for local_idx in chunks:
            full_idx_np = gene_indices[local_idx]
            full_idx = torch.tensor(full_idx_np, dtype=torch.long, device=device)
            pred = model(x, gene_emb.index_select(0, full_idx))
            true = y.index_select(1, full_idx)
            pred_np = tensor_to_numpy(pred)
            true_np = tensor_to_numpy(true)
            overall.update(pred_np, true_np)
            stats.update_chunk(pred_np, true_np, local_idx)
            pred_full_parts.append(pred_np)
            true_full_parts.append(true_np)
        pred_full = np.concatenate(pred_full_parts, axis=1)
        true_full = np.concatenate(true_full_parts, axis=1)
        for sample in sorted(set(batch["sample"])):
            mask = np.asarray([s == sample for s in batch["sample"]])
            sample_stats.setdefault(sample, RunningStats()).update(pred_full[mask], true_full[mask])

    gene_mse = stats.mse()
    gene_corr = stats.corr()
    finite_gene_corr = gene_corr[np.isfinite(gene_corr)]
    metrics = {
        "mse": overall.mse(),
        "corr": overall.corr(),
        "gene_corr_mean": float(np.mean(finite_gene_corr)) if finite_gene_corr.size else float("nan"),
        "gene_corr_median": float(np.median(finite_gene_corr)) if finite_gene_corr.size else float("nan"),
        "n_genes": int(len(gene_indices)),
        "n_finite_gene_corr": int(finite_gene_corr.size),
    }
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{label}_overall.json").write_text(json.dumps(metrics, indent=2))
        pd.DataFrame(
            [
                {"sample": sample, "mse": stat.mse(), "corr": stat.corr()}
                for sample, stat in sorted(sample_stats.items())
            ]
        ).to_csv(out_dir / f"{label}_sample_metrics.tsv", sep="\t", index=False)
        pd.DataFrame(
            [
                {
                    "gene_id": gene_ids[int(full_idx)],
                    "mse": float(mse),
                    "corr": float(corr),
                    "full_gene_index": int(full_idx),
                }
                for full_idx, mse, corr in zip(gene_indices.tolist(), gene_mse.tolist(), gene_corr.tolist())
            ]
        ).to_csv(out_dir / f"{label}_gene_metrics.tsv", sep="\t", index=False)
    return metrics


def train_variant(
    args: argparse.Namespace,
    variant: str,
    gene_emb_np: np.ndarray,
    train_idx: np.ndarray,
    heldout_idx: np.ndarray,
    gene_ids: list[str],
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    spot_dim: int,
) -> dict[str, object]:
    variant_dir = EXP_ROOT / variant
    ckpt_dir = variant_dir / "checkpoints"
    results_dir = variant_dir / "results"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    gene_emb = numpy_to_tensor(gene_emb_np).to(device)
    model = GeneConditionedProgramDecoder(
        spot_dim=spot_dim,
        gene_dim=int(gene_emb.shape[1]),
        hidden_dim=args.hidden_dim,
        program_dim=args.program_dim,
        dropout=args.dropout,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    rng_offset = 0 if args.paired_rng else {"decima": 0, "random": 1000, "constant": 2000}.get(variant, 3000)
    rng = np.random.default_rng(args.seed + rng_offset)
    val_gene_subset = np.sort(
        rng.choice(train_idx, size=min(args.val_train_genes, len(train_idx)), replace=False).astype(np.int64)
    )

    best_path = ckpt_dir / "best.pt"
    best_score = -float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0
        loss_sum = 0.0
        mse_sum = 0.0
        corr_sum = 0.0
        for batch in train_loader:
            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            gene_idx_np = np.sort(
                rng.choice(train_idx, size=min(args.train_genes_per_batch, len(train_idx)), replace=False).astype(np.int64)
            )
            gene_idx = torch.tensor(gene_idx_np, dtype=torch.long, device=device)
            pred = model(x, gene_emb.index_select(0, gene_idx))
            target = y.index_select(1, gene_idx)
            mse = F.mse_loss(pred, target)
            c_loss = corr_loss(pred, target)
            loss = mse + args.lambda_gene_corr * c_loss
            if not torch.isfinite(loss):
                opt.zero_grad(set_to_none=True)
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip, error_if_nonfinite=False)
            if not torch.isfinite(grad_norm):
                opt.zero_grad(set_to_none=True)
                continue
            opt.step()
            n = int(x.shape[0])
            total += n
            loss_sum += float(loss.detach().cpu()) * n
            mse_sum += float(mse.detach().cpu()) * n
            corr_sum += float(c_loss.detach().cpu()) * n
        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            gene_emb=gene_emb,
            gene_indices=val_gene_subset,
            gene_ids=gene_ids,
            device=device,
            gene_chunk_size=args.eval_gene_chunk_size,
            out_dir=None,
            label="val_train_gene_subset",
        )
        score = val_metrics["gene_corr_mean"] - args.selection_mse_weight * val_metrics["mse"]
        record = {
            "epoch": int(epoch),
            "train_loss": float(loss_sum / max(total, 1)),
            "train_mse": float(mse_sum / max(total, 1)),
            "train_corr_loss": float(corr_sum / max(total, 1)),
            "val": val_metrics,
            "score": float(score),
        }
        history.append(record)
        print(f"[{variant} epoch {epoch}] {json.dumps(record)}", flush=True)
        torch.save({"model": model.state_dict(), "epoch": epoch, "score": score, "val": val_metrics}, ckpt_dir / "last.pt")
        if score > best_score:
            best_score = score
            torch.save({"model": model.state_dict(), "epoch": epoch, "score": score, "val": val_metrics}, best_path)
            print(f"[{variant} ckpt] saved best epoch={epoch} score={score:.6f}", flush=True)

    (results_dir / "history.json").write_text(json.dumps(history, indent=2))
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    print(f"[{variant} best] epoch={ckpt['epoch']} score={ckpt['score']:.6f} val={ckpt['val']}", flush=True)

    heldout_test = evaluate(
        model=model,
        loader=test_loader,
        gene_emb=gene_emb,
        gene_indices=heldout_idx,
        gene_ids=gene_ids,
        device=device,
        gene_chunk_size=args.eval_gene_chunk_size,
        out_dir=results_dir,
        label="heldout_test",
    )
    train_test_subset_idx = np.sort(rng.choice(train_idx, size=min(args.report_train_genes, len(train_idx)), replace=False))
    train_test = evaluate(
        model=model,
        loader=test_loader,
        gene_emb=gene_emb,
        gene_indices=train_test_subset_idx,
        gene_ids=gene_ids,
        device=device,
        gene_chunk_size=args.eval_gene_chunk_size,
        out_dir=results_dir,
        label="train_gene_test_subset",
    )
    summary = {
        "variant": variant,
        "best_epoch": int(ckpt["epoch"]),
        "best_score": float(ckpt["score"]),
        "heldout_test": heldout_test,
        "train_gene_test_subset": train_test,
    }
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--program-dim", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=2.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--train-genes-per-batch", type=int, default=512)
    parser.add_argument("--val-train-genes", type=int, default=2048)
    parser.add_argument("--report-train-genes", type=int, default=2048)
    parser.add_argument("--eval-gene-chunk-size", type=int, default=512)
    parser.add_argument("--lambda-gene-corr", type=float, default=0.15)
    parser.add_argument("--selection-mse-weight", type=float, default=0.02)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--paired-rng",
        action="store_true",
        help="Use the same validation-gene subset and training-gene minibatch RNG across learned controls.",
    )
    parser.add_argument("--variants", nargs="+", default=["decima", "random", "constant"], choices=["decima", "random", "constant"])
    parser.add_argument("--train-mean-baseline", action="store_true", default=True)
    parser.add_argument("--no-train-mean-baseline", dest="train_mean_baseline", action="store_false")
    args = parser.parse_args()

    set_seed(args.seed)
    EXP_ROOT.mkdir(parents=True, exist_ok=True)
    (EXP_ROOT / "results").mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(DATA_ROOT / "manifest.tsv", sep="\t")
    split = json.loads((DATA_ROOT / "split.json").read_text())
    gene_ids = load_gene_ids()
    train_idx, heldout_idx = load_gene_split(gene_ids)
    feature_store = FeatureStore()
    expression_store = ExpressionStore(gene_ids)
    loader_kwargs = {"num_workers": args.workers, "pin_memory": torch.cuda.is_available(), "collate_fn": collate}
    train_ds = SpotDataset(manifest[manifest["split"] == "train"].copy(), feature_store, expression_store)
    val_ds = SpotDataset(manifest[manifest["split"] == "val"].copy(), feature_store, expression_store)
    test_ds = SpotDataset(manifest[manifest["split"] == "test"].copy(), feature_store, expression_store)
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        **loader_kwargs,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    decima_emb = load_decima_embeddings(gene_ids, train_idx)
    random_emb = load_random_embeddings(decima_emb.shape, args.seed + 12345)
    constant_emb = load_constant_embeddings(decima_emb.shape)
    metadata = {
        "experiment": str(EXP_ROOT),
        "design": "Frozen UNI2-h spot embeddings + gene-conditioned bilinear decoder; train genes only; heldout genes test. No-sequence controls include random gene vectors, identical constant gene vectors, and an empirical train-slide gene-mean baseline.",
        "data_root": str(DATA_ROOT),
        "expression_root": str(EXPRESSION_ROOT),
        "embedding_root": str(EMBEDDING_ROOT),
        "decima_embedding_cache": str(DECIMA_EMB_CACHE),
        "n_train_spots": int(len(train_ds)),
        "n_val_spots": int(len(val_ds)),
        "n_test_spots": int(len(test_ds)),
        "n_genes_total": int(len(gene_ids)),
        "n_train_genes": int(len(train_idx)),
        "n_heldout_genes": int(len(heldout_idx)),
        "spot_dim": int(split["embedding_dim"]),
        "gene_dim": int(decima_emb.shape[1]),
        "args": vars(args),
    }
    (EXP_ROOT / "experiment_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2), flush=True)

    emb_by_variant = {
        "decima": decima_emb,
        "random": random_emb,
        "constant": constant_emb,
    }
    summaries = []
    for variant in args.variants:
        emb = emb_by_variant[variant]
        summaries.append(
            train_variant(
                args=args,
                variant=variant,
                gene_emb_np=emb,
                train_idx=train_idx,
                heldout_idx=heldout_idx,
                gene_ids=gene_ids,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                device=device,
                spot_dim=int(split["embedding_dim"]),
            )
        )
    summary = {"variants": summaries}
    if args.train_mean_baseline:
        print("[train_gene_mean] computing empirical train-slide gene means", flush=True)
        train_gene_means = compute_train_gene_means(train_ds.manifest, expression_store, len(gene_ids))
        mean_dir = EXP_ROOT / "train_gene_mean" / "results"
        heldout_mean = evaluate_gene_mean_baseline(
            loader=test_loader,
            gene_means=train_gene_means,
            gene_indices=heldout_idx,
            gene_ids=gene_ids,
            device=device,
            gene_chunk_size=args.eval_gene_chunk_size,
            out_dir=mean_dir,
            label="heldout_test",
        )
        train_test_subset_idx = np.sort(np.random.default_rng(args.seed + 3000).choice(train_idx, size=min(args.report_train_genes, len(train_idx)), replace=False))
        train_subset_mean = evaluate_gene_mean_baseline(
            loader=test_loader,
            gene_means=train_gene_means,
            gene_indices=train_test_subset_idx,
            gene_ids=gene_ids,
            device=device,
            gene_chunk_size=args.eval_gene_chunk_size,
            out_dir=mean_dir,
            label="train_gene_test_subset",
        )
        mean_summary = {
            "variant": "train_gene_mean",
            "note": "No sequence or learned gene embedding; uses empirical expression means from training slides, including held-out genes.",
            "heldout_test": heldout_mean,
            "train_gene_test_subset": train_subset_mean,
        }
        (mean_dir / "summary.json").write_text(json.dumps(mean_summary, indent=2))
        summaries.append(mean_summary)
        summary["variants"] = summaries
    if len(summaries) >= 2:
        by_name = {s["variant"]: s for s in summaries}
        if "decima" in by_name:
            decima = by_name["decima"]["heldout_test"]
            for control_name in ("random", "constant", "train_gene_mean"):
                if control_name not in by_name:
                    continue
                control = by_name[control_name]["heldout_test"]
                summary[f"heldout_decima_minus_{control_name}"] = {
                    key: float(decima[key] - control[key])
                    for key in ("mse", "corr", "gene_corr_mean", "gene_corr_median")
                }
    (EXP_ROOT / "results/summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
