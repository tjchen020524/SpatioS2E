#!/usr/bin/env python3
"""Train current ablations with balanced full-gene coverage each epoch."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from experiments.sample_split_fullgenes_v4_structured_loss import train_stage0_reg_v4 as base
from models.stage0_reg_model_factory import build_stage0_reg_model


def take_balanced_chunk(
    order: torch.Tensor,
    cursor: int,
    width: int,
    n_genes: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    pieces = []
    remaining = width
    while remaining > 0:
        available = n_genes - cursor
        take = min(remaining, available)
        pieces.append(order[cursor : cursor + take])
        cursor += take
        remaining -= take
        if cursor == n_genes:
            order = torch.randperm(n_genes)
            cursor = 0
    return torch.cat(pieces), order, cursor


def forward_selected_batch(
    model,
    batch,
    cfg_loss: dict,
    gene_indices: torch.Tensor,
    device: torch.device,
    no_graph: bool,
) -> dict[str, torch.Tensor]:
    x = batch["x"][0].to(device)
    edge_index = batch["edge_index"][0].to(device)
    edge_weight = batch["edge_weight"][0].to(device)
    if no_graph:
        edge_weight = torch.zeros_like(edge_weight)
    true = base.to_dense(batch["y"][0]).to(device)
    indices = gene_indices.to(device=device)
    true = true.index_select(1, indices)
    all_gene_ids = batch["gene_ids"][0]
    gene_ids = [all_gene_ids[idx] for idx in gene_indices.tolist()]
    sample_name = batch["sample"][0]

    out = base.forward_model(
        model,
        x,
        edge_index,
        edge_weight,
        gene_ids,
        sample_name=sample_name,
    )
    losses = base.compute_losses(
        out["log_mu"],
        true,
        edge_index,
        edge_weight,
        cfg_loss,
        out=out,
    )
    calibration = base.calibration_identity_loss(out)
    losses["loss"] = losses["loss"] + float(
        cfg_loss.get("lambda_calib_identity", 0.0)
    ) * calibration
    losses["l_calib"] = calibration.detach()
    return losses


def train(config_path: Path, no_graph: bool) -> None:
    cfg = base.load_config(config_path)
    base.set_seed(int(cfg.get("seed", 42)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = base.build_dataloader(cfg, split="train")
    val_loader = base.build_dataloader(cfg, split="val")
    sample_batch = next(iter(train_loader))
    input_dim = int(sample_batch["x"][0].shape[1])
    n_genes = int(sample_batch["y"][0].shape[1])

    model = build_stage0_reg_model(cfg, input_dim=input_dim).to(device)
    optimizer = torch.optim.Adam(
        filter(lambda parameter: parameter.requires_grad, model.parameters()),
        lr=float(cfg["optim"]["lr"]),
        weight_decay=float(cfg["optim"].get("weight_decay", 0.0)),
    )

    optim_cfg = cfg["optim"]
    cfg_loss = cfg["loss"]
    chunk_width = int(optim_cfg["max_genes_per_batch"])
    chunks_per_sample = int(optim_cfg.get("train_gene_chunks_per_sample", 1))
    capacity = len(train_loader) * chunks_per_sample * chunk_width
    if bool(optim_cfg.get("require_full_gene_coverage_per_epoch", False)) and capacity < n_genes:
        raise ValueError(
            f"Per-epoch gene capacity {capacity} is smaller than the {n_genes}-gene panel"
        )

    max_epochs = int(optim_cfg["max_epochs"])
    min_epochs = int(optim_cfg.get("min_epochs", 1))
    patience = int(optim_cfg.get("early_stop_patience", 0))
    val_chunk = int(optim_cfg.get("val_max_genes_per_batch", chunk_width))
    checkpoint_dir = Path(cfg["paths"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / cfg["paths"].get("checkpoint_name", "best.pt")

    metric_keys = (
        "loss",
        "l_point",
        "l_gene",
        "l_spot",
        "l_var",
        "l_hi_under",
        "l_delta_point",
        "l_delta_gene",
        "l_lap",
        "l_calib",
    )
    best_metric = float("-inf")
    best_epoch = 0
    no_improve = 0
    saved_once = False

    for epoch in range(1, max_epochs + 1):
        model.train()
        totals = {key: 0.0 for key in metric_keys}
        steps = 0
        order = torch.randperm(n_genes)
        cursor = 0
        seen = torch.zeros(n_genes, dtype=torch.bool)
        for batch in train_loader:
            if int(batch["y"][0].shape[1]) != n_genes:
                raise ValueError("All training samples must use the same ordered gene panel")
            for _ in range(chunks_per_sample):
                indices, order, cursor = take_balanced_chunk(
                    order, cursor, chunk_width, n_genes
                )
                seen[indices] = True
                optimizer.zero_grad()
                losses = forward_selected_batch(
                    model,
                    batch,
                    cfg_loss=cfg_loss,
                    gene_indices=indices,
                    device=device,
                    no_graph=no_graph,
                )
                if not torch.isfinite(losses["loss"]):
                    print("[warn] non-finite loss, skipping chunk", flush=True)
                    continue
                losses["loss"].backward()
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda parameter: parameter.requires_grad, model.parameters()),
                    max_norm=1.0,
                )
                optimizer.step()
                for key in metric_keys:
                    totals[key] += float(losses[key].item())
                steps += 1

        coverage = float(seen.float().mean().item())
        averages = {key: value / max(steps, 1) for key, value in totals.items()}
        print(
            f"[epoch {epoch}/{max_epochs}] train_loss={averages['loss']:.4f} "
            f"gene_coverage={coverage:.4f} chunks={steps} "
            f"point={averages['l_point']:.4f} gene={averages['l_gene']:.4f} "
            f"spot={averages['l_spot']:.4f} lap={averages['l_lap']:.4f}",
            flush=True,
        )
        if bool(optim_cfg.get("require_full_gene_coverage_per_epoch", False)) and coverage < 1.0:
            raise RuntimeError(f"Epoch {epoch} covered only {coverage:.3%} of target genes")

        metrics = base.validate(
            model=model,
            val_loader=val_loader,
            cfg_loss=cfg_loss,
            chunk_size=val_chunk,
            device=device,
            no_graph=no_graph,
        )
        metric, metric_name = base.choose_val_metric(metrics, optim_cfg)
        print(
            f"  val_loss={metrics['val_loss']:.4f} "
            f"val_flat_corr={metrics['flat_corr']:.4f} "
            f"val_gene_corr_mean={metrics['gene_corr_mean']:.4f} "
            f"selection={metric_name}:{metric:.4f}",
            flush=True,
        )
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
                    "balanced_full_gene_coverage": True,
                },
                checkpoint_path,
            )
            saved_once = True
            print(f"  [ckpt] saved best to {checkpoint_path}", flush=True)
        else:
            no_improve += 1

        if epoch >= min_epochs and patience > 0 and no_improve >= patience:
            print(
                f"[early stop] best epoch={best_epoch} metric={best_metric:.4f}",
                flush=True,
            )
            break

    if not saved_once:
        raise RuntimeError("Training completed without a finite validation checkpoint")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--no-graph", action="store_true")
    args = parser.parse_args()
    train(args.config, no_graph=bool(args.no_graph))


if __name__ == "__main__":
    main()
