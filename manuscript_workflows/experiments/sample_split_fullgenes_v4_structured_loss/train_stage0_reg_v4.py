#!/usr/bin/env python3
"""Stage-0 v4 training: v3 backbone with structured gene/spot correlation losses."""

from __future__ import annotations
from workflow_paths import DATA_ROOT

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

ROOT = DATA_ROOT
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.append(str(path))

from gnn_dataloader import SpatioS2EDataset, collate_graphs
from models.stage0_reg_model_factory import build_stage0_reg_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def laplacian_loss(delta: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
    src, dst = edge_index
    diff = delta[src] - delta[dst]
    sq = (diff * diff) * edge_weight.unsqueeze(-1)
    return sq.sum()


def centered_cosine_loss(
    pred: torch.Tensor,
    true: torch.Tensor,
    dim: int,
    min_true_std: float = 1.0e-6,
) -> torch.Tensor:
    pred_c = pred - pred.mean(dim=dim, keepdim=True)
    true_c = true - true.mean(dim=dim, keepdim=True)

    pred_norm = torch.linalg.norm(pred_c, dim=dim)
    true_norm = torch.linalg.norm(true_c, dim=dim)
    valid = true_norm > float(min_true_std)
    if not torch.any(valid):
        return pred.new_tensor(0.0)

    pred_unit = pred_c / pred_norm.clamp_min(1.0e-8).unsqueeze(dim)
    true_unit = true_c / true_norm.clamp_min(1.0e-8).unsqueeze(dim)
    cos = (pred_unit * true_unit).sum(dim=dim)
    return 1.0 - cos[valid].mean()


def gene_variance_ratio_loss(
    pred: torch.Tensor,
    true: torch.Tensor,
    dim: int,
    min_true_std: float = 1.0e-6,
    eps: float = 1.0e-4,
) -> torch.Tensor:
    pred_c = pred - pred.mean(dim=dim, keepdim=True)
    true_c = true - true.mean(dim=dim, keepdim=True)
    pred_std = torch.sqrt(torch.mean(pred_c * pred_c, dim=dim) + float(eps))
    true_std = torch.sqrt(torch.mean(true_c * true_c, dim=dim) + float(eps))
    valid = true_std > float(min_true_std)
    if not torch.any(valid):
        return pred.new_tensor(0.0)
    log_ratio = torch.log(pred_std[valid]) - torch.log(true_std[valid])
    return torch.mean(log_ratio * log_ratio)


def upper_tail_underprediction_loss(
    pred: torch.Tensor,
    true: torch.Tensor,
    dim: int,
    quantile: float = 0.9,
    min_true_std: float = 1.0e-6,
    beta: float = 0.25,
    scale_by_true_std: bool = True,
) -> torch.Tensor:
    quantile = float(quantile)
    if not (0.0 < quantile < 1.0):
        raise ValueError(f"quantile must be in (0, 1), got {quantile}")

    true_c = true - true.mean(dim=dim, keepdim=True)
    true_std = torch.sqrt(torch.mean(true_c * true_c, dim=dim) + 1.0e-8)
    valid = true_std > float(min_true_std)
    if not torch.any(valid):
        return pred.new_tensor(0.0)

    thresholds = torch.quantile(true, q=quantile, dim=dim, keepdim=True)
    valid_mask = valid.unsqueeze(dim)
    tail_mask = (true >= thresholds) & valid_mask
    if not torch.any(tail_mask):
        return pred.new_tensor(0.0)

    gap = F.relu(true - pred)
    if bool(scale_by_true_std):
        gap = gap / true_std.clamp_min(1.0e-2).unsqueeze(dim)

    selected = gap[tail_mask]
    return F.smooth_l1_loss(
        selected,
        torch.zeros_like(selected),
        beta=float(beta),
    )


def resolve_spatial_residual_pred(out: Dict[str, torch.Tensor]) -> torch.Tensor:
    return resolve_spatial_residual_pred_with_mode(out, mode="effective")


def resolve_spatial_residual_pred_with_mode(out: Dict[str, torch.Tensor], mode: str) -> torch.Tensor:
    mode = str(mode).lower()
    delta = out["delta"]
    if mode in {"raw", "normalized", "z"}:
        return delta
    if mode not in {"effective", "scaled", "default"}:
        raise ValueError(f"Unsupported delta_pred_space: {mode}")
    if "delta_scale" in out:
        return delta * out["delta_scale"].unsqueeze(0)
    return delta


def build_delta_target(
    out: Dict[str, torch.Tensor],
    true: torch.Tensor,
    mode: str,
) -> torch.Tensor:
    mode = str(mode).lower()
    if mode in {"centered_true", "centered", "demean_true"}:
        return true - true.mean(dim=0, keepdim=True)
    if mode in {"baseline_relative", "base_relative"}:
        if "log_mu_base" not in out:
            raise ValueError("delta_target_mode=baseline_relative requires model output log_mu_base")
        return true - out["log_mu_base"]
    if mode in {"normalized_baseline_relative", "prior_normalized_baseline", "baseline_relative_normalized"}:
        if "log_mu_base" not in out:
            raise ValueError(f"delta_target_mode={mode} requires model output log_mu_base")
        if "delta_scale_prior" in out:
            scale = out["delta_scale_prior"]
        elif "delta_scale" in out:
            scale = out["delta_scale"]
        else:
            raise ValueError(f"delta_target_mode={mode} requires model output delta_scale_prior or delta_scale")
        return (true - out["log_mu_base"]) / scale.unsqueeze(0).clamp_min(1.0e-4)
    raise ValueError(f"Unsupported delta_target_mode: {mode}")


def delta_supervision_losses(
    out: Dict[str, torch.Tensor],
    true: torch.Tensor,
    cfg_loss: Dict,
) -> Dict[str, torch.Tensor]:
    spatial_pred = resolve_spatial_residual_pred_with_mode(
        out,
        mode=str(cfg_loss.get("delta_pred_space", "effective")),
    )
    target = build_delta_target(
        out=out,
        true=true,
        mode=str(cfg_loss.get("delta_target_mode", "centered_true")),
    )
    l_delta_point = point_loss(
        spatial_pred,
        target,
        loss_type=str(cfg_loss.get("delta_point_loss", cfg_loss.get("point_loss", "huber"))),
        huber_beta=float(cfg_loss.get("delta_huber_beta", cfg_loss.get("huber_beta", 1.0))),
    )
    l_delta_gene = centered_cosine_loss(
        spatial_pred,
        target,
        dim=0,
        min_true_std=float(cfg_loss.get("min_true_std", 1.0e-6)),
    )
    return {
        "l_delta_point": l_delta_point,
        "l_delta_gene": l_delta_gene,
    }


def point_loss(pred: torch.Tensor, true: torch.Tensor, loss_type: str, huber_beta: float) -> torch.Tensor:
    loss_type = str(loss_type).lower()
    if loss_type == "mse":
        return torch.mean((pred - true) ** 2)
    if loss_type in {"huber", "smoothl1", "smooth_l1"}:
        return F.smooth_l1_loss(pred, true, beta=float(huber_beta))
    raise ValueError(f"Unsupported point loss type: {loss_type}")


class RunningStats:
    def __init__(self) -> None:
        self.count = 0
        self.sum_pred = 0.0
        self.sum_true = 0.0
        self.sum_pred2 = 0.0
        self.sum_true2 = 0.0
        self.sum_pred_true = 0.0
        self.sum_sq_error = 0.0

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
        d = p - t
        self.sum_sq_error += float((d * d).sum().item())

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


def load_config(path: Path) -> Dict:
    if path.suffix in {".yaml", ".yml"}:
        import yaml  # type: ignore

        with path.open("r") as handle:
            return yaml.safe_load(handle)
    with path.open("r") as handle:
        return json.load(handle)


def build_dataloader(cfg: Dict, split: str) -> torch.utils.data.DataLoader:
    gene_list_path = Path(cfg.get("paths", {}).get("gene_split_dir", "data/processed/dataset_gene_split")) / f"{split}_genes.txt"
    gene_filter = None
    if gene_list_path.exists():
        gene_filter = [line.strip() for line in gene_list_path.read_text().splitlines() if line.strip()]

    samples_cfg = cfg.get("split", {})
    samples = samples_cfg[split] if split in samples_cfg else None
    if samples == []:
        samples = None

    ds = SpatioS2EDataset(
        split="train" if split == "train" else "val",
        samples=samples,
        expression_root=Path(cfg["paths"]["expression_root"]),
        multimodal_root=Path(cfg["paths"]["multimodal_root"]),
        graph_root=Path(cfg["paths"]["graph_root"]),
        targets_dense=bool(cfg.get("targets_dense", True)),
        cache=False,
        normalize_modalities=bool(cfg.get("normalize_modalities", True)),
        modality_stats_path=Path(cfg["paths"]["modality_stats"]),
        gene_filter=set(gene_filter) if gene_filter else None,
    )
    return torch.utils.data.DataLoader(ds, batch_size=1, shuffle=(split == "train"), collate_fn=collate_graphs)


def compute_losses(
    pred: torch.Tensor,
    true: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    cfg_loss: Dict,
    out: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, torch.Tensor]:
    if out is None:
        out = {}
    delta = out["delta"]
    l_point = point_loss(
        pred,
        true,
        loss_type=str(cfg_loss.get("point_loss", "huber")),
        huber_beta=float(cfg_loss.get("huber_beta", 1.0)),
    )
    l_gene = centered_cosine_loss(pred, true, dim=0, min_true_std=float(cfg_loss.get("min_true_std", 1.0e-6)))
    l_spot = centered_cosine_loss(pred, true, dim=1, min_true_std=float(cfg_loss.get("min_true_std", 1.0e-6)))
    l_var = gene_variance_ratio_loss(
        pred,
        true,
        dim=0,
        min_true_std=float(cfg_loss.get("min_true_std", 1.0e-6)),
        eps=float(cfg_loss.get("variance_eps", 1.0e-4)),
    )
    l_hi_under = upper_tail_underprediction_loss(
        pred,
        true,
        dim=0,
        quantile=float(cfg_loss.get("high_expr_quantile", 0.9)),
        min_true_std=float(cfg_loss.get("min_true_std", 1.0e-6)),
        beta=float(cfg_loss.get("high_expr_beta", 0.25)),
        scale_by_true_std=bool(cfg_loss.get("high_expr_scale_by_true_std", True)),
    )
    l_lap_sum = laplacian_loss(delta, edge_index, edge_weight)
    denom_lap = edge_weight.numel() * true.shape[1]
    l_lap = l_lap_sum / max(int(denom_lap), 1)
    delta_losses = delta_supervision_losses(out, true, cfg_loss)
    l_delta_point = delta_losses["l_delta_point"]
    l_delta_gene = delta_losses["l_delta_gene"]

    loss = (
        float(cfg_loss.get("lambda_point", 1.0)) * l_point
        + float(cfg_loss.get("lambda_gene", 0.0)) * l_gene
        + float(cfg_loss.get("lambda_spot", 0.0)) * l_spot
        + float(cfg_loss.get("lambda_gene_var", 0.0)) * l_var
        + float(cfg_loss.get("lambda_hi_under", 0.0)) * l_hi_under
        + float(cfg_loss.get("lambda_delta_point", 0.0)) * l_delta_point
        + float(cfg_loss.get("lambda_delta_gene", 0.0)) * l_delta_gene
        + float(cfg_loss.get("lambda_lap", 0.0)) * l_lap
    )
    return {
        "loss": loss,
        "l_point": l_point.detach(),
        "l_gene": l_gene.detach(),
        "l_spot": l_spot.detach(),
        "l_var": l_var.detach(),
        "l_hi_under": l_hi_under.detach(),
        "l_delta_point": l_delta_point.detach(),
        "l_delta_gene": l_delta_gene.detach(),
        "l_lap": l_lap.detach(),
    }


def calibration_identity_loss(out: Dict[str, torch.Tensor]) -> torch.Tensor:
    pieces = []
    if "raw_base_logscale" not in out:
        ref = out["log_mu"]
    else:
        ref = out["log_mu"]
        pieces.extend(
            [
                out["raw_base_logscale"].pow(2).mean(),
                out["raw_delta_logscale"].pow(2).mean(),
                out["calib_bias"].pow(2).mean(),
            ]
        )
    if "residual_scale_logcorr" in out:
        pieces.append(out["residual_scale_logcorr"].pow(2).mean())
    if not pieces:
        return ref.new_tensor(0.0)
    return torch.stack(pieces).sum()


def choose_val_metric(metrics: Dict[str, float], cfg_optim: Dict) -> tuple[float, str]:
    mode = str(cfg_optim.get("selection_metric", "legacy")).lower()
    gene_corr = float(metrics.get("gene_corr_mean", float("nan")))
    flat_corr = float(metrics.get("flat_corr", float("nan")))
    val_loss = float(metrics.get("val_loss", float("nan")))

    if mode in {"legacy", "default"}:
        if np.isfinite(gene_corr):
            return gene_corr, "val_gene_corr_mean"
        if np.isfinite(flat_corr):
            return flat_corr, "val_flat_corr"
        return -val_loss, "-val_loss"

    if mode in {"gene_flat_combo", "gene_corr_plus_flat"}:
        if np.isfinite(gene_corr):
            flat_weight = float(cfg_optim.get("selection_flat_weight", 0.25))
            score = gene_corr
            name = "val_gene_corr_mean"
            if np.isfinite(flat_corr):
                score += flat_weight * flat_corr
                name = f"val_gene_corr_mean+{flat_weight:g}*val_flat_corr"
            return score, name
        if np.isfinite(flat_corr):
            return flat_corr, "val_flat_corr"
        return -val_loss, "-val_loss"

    raise ValueError(f"Unsupported selection_metric: {mode}")


def forward_train_batch(
    model,
    batch: Dict[str, Sequence],
    cfg_loss: Dict,
    max_genes: Optional[int],
    device: torch.device,
    no_graph: bool = False,
) -> Dict[str, torch.Tensor]:
    x = batch["x"][0].to(device)
    edge_index = batch["edge_index"][0].to(device)
    edge_weight = batch["edge_weight"][0].to(device)
    if no_graph:
        edge_weight = torch.zeros_like(edge_weight)
    y_spot = to_dense(batch["y"][0]).to(device)
    gene_ids = batch["gene_ids"][0]
    sample_name = batch["sample"][0]

    if max_genes is not None and y_spot.shape[1] > max_genes:
        perm = torch.randperm(y_spot.shape[1], device=y_spot.device)[:max_genes]
        y_spot = y_spot[:, perm]
        gene_ids = [gene_ids[i] for i in perm.tolist()]

    out = forward_model(model, x, edge_index, edge_weight, gene_ids, sample_name=sample_name)
    pred = out["log_mu"]
    delta = out["delta"]
    losses = compute_losses(pred, y_spot, edge_index, edge_weight, cfg_loss, out=out)
    l_calib = calibration_identity_loss(out)
    losses["loss"] = losses["loss"] + float(cfg_loss.get("lambda_calib_identity", 0.0)) * l_calib
    losses["l_calib"] = l_calib.detach()
    return losses


def validate(
    model,
    val_loader: torch.utils.data.DataLoader,
    cfg_loss: Dict,
    chunk_size: Optional[int],
    device: torch.device,
    no_graph: bool = False,
) -> Dict[str, float]:
    model.eval()
    vloss = vpoint = vgene = vspot = vvar = vhi = vdelta_p = vdelta_g = vlap = vcalib = 0.0
    steps = 0
    overall = RunningStats()
    gene_stats: Dict[str, RunningStats] = {}

    with torch.no_grad():
        for batch in val_loader:
            x = batch["x"][0].to(device)
            edge_index = batch["edge_index"][0].to(device)
            edge_weight = batch["edge_weight"][0].to(device)
            if no_graph:
                edge_weight = torch.zeros_like(edge_weight)
            y_spot = to_dense(batch["y"][0]).to(device)
            gene_ids_full = list(batch["gene_ids"][0])
            sample_name = batch["sample"][0]
            n_genes = len(gene_ids_full)

            if chunk_size is None or chunk_size <= 0 or chunk_size >= n_genes:
                ranges = [(0, n_genes)]
            else:
                ranges = [(s, min(s + chunk_size, n_genes)) for s in range(0, n_genes, chunk_size)]

            for start, end in ranges:
                gene_ids = gene_ids_full[start:end]
                y_chunk = y_spot[:, start:end]
                out = forward_model(model, x, edge_index, edge_weight, gene_ids, sample_name=sample_name)
                pred = out["log_mu"]
                delta = out["delta"]
                losses = compute_losses(pred, y_chunk, edge_index, edge_weight, cfg_loss, out=out)
                l_calib = calibration_identity_loss(out)
                losses["loss"] = losses["loss"] + float(cfg_loss.get("lambda_calib_identity", 0.0)) * l_calib

                vloss += float(losses["loss"].item())
                vpoint += float(losses["l_point"].item())
                vgene += float(losses["l_gene"].item())
                vspot += float(losses["l_spot"].item())
                vvar += float(losses["l_var"].item())
                vhi += float(losses["l_hi_under"].item())
                vdelta_p += float(losses["l_delta_point"].item())
                vdelta_g += float(losses["l_delta_gene"].item())
                vlap += float(losses["l_lap"].item())
                vcalib += float(l_calib.item())
                steps += 1

                pred_cpu = pred.detach().cpu()
                true_cpu = y_chunk.detach().cpu()
                overall.update(pred_cpu.reshape(-1), true_cpu.reshape(-1))
                for j, gid in enumerate(gene_ids):
                    gene_stats.setdefault(gid, RunningStats()).update(pred_cpu[:, j], true_cpu[:, j])

    val_loss_avg = vloss / max(steps, 1)
    val_point_avg = vpoint / max(steps, 1)
    val_gene_avg = vgene / max(steps, 1)
    val_spot_avg = vspot / max(steps, 1)
    val_var_avg = vvar / max(steps, 1)
    val_hi_avg = vhi / max(steps, 1)
    val_delta_point_avg = vdelta_p / max(steps, 1)
    val_delta_gene_avg = vdelta_g / max(steps, 1)
    val_lap_avg = vlap / max(steps, 1)
    flat_corr = overall.corr()
    gene_corrs = [s.corr() for s in gene_stats.values()]
    gene_corrs = [c for c in gene_corrs if np.isfinite(c)]
    gene_corr_mean = float(np.mean(gene_corrs)) if gene_corrs else float("nan")

    return {
        "val_loss": val_loss_avg,
        "val_point": val_point_avg,
        "val_gene_loss": val_gene_avg,
        "val_spot_loss": val_spot_avg,
        "val_var_loss": val_var_avg,
        "val_hi_under_loss": val_hi_avg,
        "val_delta_point_loss": val_delta_point_avg,
        "val_delta_gene_loss": val_delta_gene_avg,
        "val_lap": val_lap_avg,
        "val_calib_identity": vcalib / max(steps, 1),
        "flat_corr": flat_corr,
        "gene_corr_mean": gene_corr_mean,
    }


def train(cfg_path: Path, no_graph: bool = False) -> None:
    cfg = load_config(cfg_path)
    set_seed(int(cfg.get("seed", 42)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = build_dataloader(cfg, split="train")
    val_loader = build_dataloader(cfg, split="val") if ("val" in cfg.get("split", {})) else None

    sample_batch = next(iter(train_loader))
    input_dim = sample_batch["x"][0].shape[1]

    model = build_stage0_reg_model(cfg, input_dim=input_dim).to(device)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=float(cfg["optim"]["lr"]),
        weight_decay=float(cfg["optim"].get("weight_decay", 0.0)),
    )

    cfg_loss = cfg["loss"]
    max_genes = cfg["optim"].get("max_genes_per_batch", None)
    if max_genes is not None:
        max_genes = int(max_genes)
    val_max_genes = cfg["optim"].get("val_max_genes_per_batch", max_genes)
    if val_max_genes is not None:
        val_max_genes = int(val_max_genes)

    max_epochs = int(cfg["optim"]["max_epochs"])
    log_every = int(cfg["optim"].get("log_every", 1))
    patience = int(cfg["optim"].get("early_stop_patience", 0))

    ckpt_dir = Path(cfg["paths"].get("checkpoint_dir", "checkpoints/stage0_reg_v4"))
    ckpt_name = cfg["paths"].get("checkpoint_name", "best.pt")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / ckpt_name

    best_metric = float("-inf")
    best_epoch = 0
    no_improve = 0
    saved_once = False

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = point_sum = gene_sum = spot_sum = var_sum = hi_sum = delta_p_sum = delta_g_sum = lap_sum = calib_sum = 0.0
        steps = 0
        for batch in train_loader:
            optimizer.zero_grad()
            losses = forward_train_batch(
                model,
                batch,
                cfg_loss=cfg_loss,
                max_genes=max_genes,
                device=device,
                no_graph=no_graph,
            )
            if not torch.isfinite(losses["loss"]):
                print("[warn] non-finite loss, skipping batch")
                continue
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_norm=1.0)
            optimizer.step()
            total_loss += float(losses["loss"].item())
            point_sum += float(losses["l_point"].item())
            gene_sum += float(losses["l_gene"].item())
            spot_sum += float(losses["l_spot"].item())
            var_sum += float(losses["l_var"].item())
            hi_sum += float(losses["l_hi_under"].item())
            delta_p_sum += float(losses["l_delta_point"].item())
            delta_g_sum += float(losses["l_delta_gene"].item())
            lap_sum += float(losses["l_lap"].item())
            calib_sum += float(losses["l_calib"].item())
            steps += 1

        avg_loss = total_loss / max(steps, 1)
        avg_point = point_sum / max(steps, 1)
        avg_gene = gene_sum / max(steps, 1)
        avg_spot = spot_sum / max(steps, 1)
        avg_var = var_sum / max(steps, 1)
        avg_hi = hi_sum / max(steps, 1)
        avg_delta_p = delta_p_sum / max(steps, 1)
        avg_delta_g = delta_g_sum / max(steps, 1)
        avg_lap = lap_sum / max(steps, 1)
        avg_calib = calib_sum / max(steps, 1)
        if epoch % log_every == 0:
            print(
                f"[epoch {epoch}/{max_epochs}] train_loss={avg_loss:.4f} "
                f"(point={avg_point:.4f}, gene={avg_gene:.4f}, spot={avg_spot:.4f}, "
                f"var={avg_var:.4f}, hi={avg_hi:.4f}, delta_p={avg_delta_p:.4f}, "
                f"delta_g={avg_delta_g:.4f}, lap={avg_lap:.4f}, calib={avg_calib:.4f})"
            )

        if val_loader is not None:
            metrics = validate(
                model=model,
                val_loader=val_loader,
                cfg_loss=cfg_loss,
                chunk_size=val_max_genes,
                device=device,
                no_graph=no_graph,
            )
            print(
                "  val_loss={val_loss:.4f} (point={val_point:.4f}, gene={val_gene_loss:.4f}, "
                "spot={val_spot_loss:.4f}, var={val_var_loss:.4f}, hi={val_hi_under_loss:.4f}, "
                "delta_p={val_delta_point_loss:.4f}, delta_g={val_delta_gene_loss:.4f}, "
                "lap={val_lap:.4f}, calib={val_calib_identity:.4f}); "
                "val_flat_corr={flat_corr:.4f}, val_gene_corr_mean={gene_corr_mean:.4f}".format(**metrics)
            )

            metric, metric_name = choose_val_metric(metrics, cfg["optim"])

            if metric > best_metric:
                best_metric = metric
                best_epoch = epoch
                no_improve = 0
                torch.save(
                    {
                        "model": model.state_dict(),
                        "epoch": epoch,
                        "config": cfg,
                        "best_metric_name": metric_name,
                        "no_graph": bool(no_graph),
                    },
                    best_path,
                )
                print(f"  [ckpt] saved best to {best_path} ({metric_name}={metric:.4f})")
                saved_once = True
            else:
                no_improve += 1

        if patience > 0 and no_improve >= patience:
            print(f"[early stop] no improvement for {no_improve} epochs (best@{best_epoch} metric={best_metric:.4f})")
            break

    if val_loader is None or not saved_once:
        torch.save({"model": model.state_dict(), "epoch": epoch, "config": cfg, "no_graph": bool(no_graph)}, best_path)
        print(f"[ckpt] saved final model to {best_path} (no val or no improvement) at epoch {epoch}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--no-graph", action="store_true", help="Disable spatial graph message passing by zeroing edge weights.")
    args = parser.parse_args()
    train(args.config, no_graph=bool(args.no_graph))
