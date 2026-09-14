#!/usr/bin/env python3
"""Finetune STPath on SpatioS2E splits and full-gene targets."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from experiments.stpath_finetune_spatios2e.common import (
    STPathFinetuneDataset,
    STPathSubsetPredictor,
    fit_or_load_adapter,
    load_aligned_sample,
    load_yaml_or_json,
    resolve_split,
    set_seed,
    build_gene2token_map,
)


class RunningStats:
    def __init__(self) -> None:
        self.count = 0
        self.sum_pred = 0.0
        self.sum_true = 0.0
        self.sum_pred2 = 0.0
        self.sum_true2 = 0.0
        self.sum_pred_true = 0.0

    def update(self, pred: torch.Tensor, true: torch.Tensor) -> None:
        mask = torch.isfinite(pred) & torch.isfinite(true)
        if not mask.any():
            return
        p = pred[mask].double()
        t = true[mask].double()
        self.count += int(p.numel())
        self.sum_pred += float(p.sum().item())
        self.sum_true += float(t.sum().item())
        self.sum_pred2 += float((p * p).sum().item())
        self.sum_true2 += float((t * t).sum().item())
        self.sum_pred_true += float((p * t).sum().item())

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


def select_genes(token_ids: torch.Tensor, max_genes: int | None, random_pick: bool) -> torch.Tensor:
    valid = torch.where(token_ids >= 0)[0]
    if valid.numel() == 0:
        return valid
    if max_genes is None or valid.numel() <= max_genes:
        return valid
    if random_pick:
        perm = torch.randperm(valid.numel(), device=valid.device)[:max_genes]
        return valid[perm]
    return valid[:max_genes]


def forward_batch(
    predictor: STPathSubsetPredictor,
    batch: dict,
    tech_id: int,
    organ_id: int,
    max_genes: int | None,
    max_spots: int | None,
    random_gene_pick: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = batch["x"]
    coords = batch["coords"]
    y = batch["y"]
    token_ids = batch["token_ids"]

    if max_spots is not None and x.shape[0] > max_spots:
        keep = torch.randperm(x.shape[0], device=x.device)[:max_spots]
        x = x[keep]
        coords = coords[keep]
        y = y[keep]

    gene_sel = select_genes(token_ids, max_genes, random_pick=random_gene_pick)
    if gene_sel.numel() == 0:
        raise RuntimeError(f"No mappable genes in sample {batch['sample']}")

    token_sub = token_ids[gene_sel]
    y_sub = y[:, gene_sel]

    tech_tokens = torch.full((x.shape[0],), tech_id, dtype=torch.long, device=x.device)
    organ_tokens = torch.full((x.shape[0],), organ_id, dtype=torch.long, device=x.device)
    h_spot = predictor.encode_spots(x, coords, tech_tokens, organ_tokens)
    pred_sub = predictor.predict_from_hidden(h_spot, token_sub)
    mse = torch.mean((pred_sub - y_sub) ** 2)
    return mse, pred_sub, y_sub


def train(cfg_path: Path) -> None:
    cfg = load_yaml_or_json(cfg_path)
    split_map = resolve_split(cfg)
    seed = int(cfg.get("seed", 42))
    set_seed(seed)
    torch.set_num_threads(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    paths = cfg["paths"]
    hist_root = Path(paths["hist_root"])
    spatial_root = Path(paths["spatial_root"])
    expr_root = Path(paths["expression_root"])
    adapter_path = Path(paths["adapter_path"])
    gene_vocab = Path(paths["gene_vocab"])

    fit_on = str(cfg["data"].get("fit_on", "train"))
    all_samples = split_map.get("train", []) + split_map.get("val", []) + split_map.get("test", [])
    if fit_on == "train":
        fit_samples = split_map.get("train", [])
    elif fit_on == "trainval":
        fit_samples = split_map.get("train", []) + split_map.get("val", [])
    else:
        fit_samples = all_samples
    if not fit_samples:
        raise RuntimeError("No fit samples for PCA adapter")

    first_x, _, _ = load_aligned_sample(hist_root, spatial_root, fit_samples[0])
    in_dim = int(first_x.shape[1])
    adapter = fit_or_load_adapter(
        adapter_path=adapter_path,
        fit_samples=fit_samples,
        hist_root=hist_root,
        spatial_root=spatial_root,
        in_dim=in_dim,
        out_dim=1536,
    )
    gene2token = build_gene2token_map(gene_vocab)

    train_ds = STPathFinetuneDataset(
        samples=split_map["train"],
        expression_root=expr_root,
        hist_root=hist_root,
        spatial_root=spatial_root,
        adapter=adapter,
        gene2token=gene2token,
        split_name="train",
    )
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=lambda b: b[0])

    val_loader = None
    if split_map.get("val"):
        val_ds = STPathFinetuneDataset(
            samples=split_map["val"],
            expression_root=expr_root,
            hist_root=hist_root,
            spatial_root=spatial_root,
            adapter=adapter,
            gene2token=gene2token,
            split_name="val",
        )
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=lambda b: b[0])

    predictor = STPathSubsetPredictor(
        stpath_root=Path(paths["stpath_root"]),
        stpath_weight=Path(paths["stpath_weight"]),
        device=device,
    ).to(device)
    predictor.set_trainable(
        train_gene_head=bool(cfg["model"].get("train_gene_head", True)),
        train_image_embed=bool(cfg["model"].get("train_image_embed", True)),
        train_gene_embed=bool(cfg["model"].get("train_gene_embed", False)),
        unfreeze_last_n_blocks=int(cfg["model"].get("unfreeze_last_n_blocks", 0)),
    )

    tech_label = str(cfg["model"].get("tech_type", "Visium"))
    organ_label = str(cfg["model"].get("organ_type", "Brain"))
    tech_id = int(predictor.tokenizer.tech_tokenizer.encode(tech_label, align_first=True))
    organ_id = int(predictor.tokenizer.organ_tokenizer.encode(organ_label, align_first=True))

    trainable = [p for p in predictor.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable parameters selected")
    optim = torch.optim.AdamW(
        trainable,
        lr=float(cfg["optim"]["lr"]),
        weight_decay=float(cfg["optim"].get("weight_decay", 0.0)),
    )

    max_epochs = int(cfg["optim"]["max_epochs"])
    train_max_genes = cfg["optim"].get("max_genes_per_step", 512)
    train_max_genes = int(train_max_genes) if train_max_genes is not None else None
    train_max_spots = cfg["optim"].get("max_spots_per_step", 2048)
    train_max_spots = int(train_max_spots) if train_max_spots is not None else None
    val_max_genes = cfg["optim"].get("val_max_genes_per_step", 1024)
    val_max_genes = int(val_max_genes) if val_max_genes is not None else None
    val_max_spots = cfg["optim"].get("val_max_spots_per_step", None)
    val_max_spots = int(val_max_spots) if val_max_spots is not None else None

    ckpt_dir = Path(paths["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / paths.get("checkpoint_name", "best.pt")
    best_metric = float("-inf")

    print(
        f"[setup] device={device} train_samples={len(train_ds)} val_samples={0 if val_loader is None else len(val_loader.dataset)} "
        f"in_dim={in_dim} trainable_params={sum(p.numel() for p in trainable)}"
    )

    for epoch in range(1, max_epochs + 1):
        predictor.train()
        tr_mse = 0.0
        tr_steps = 0
        for batch in train_loader:
            batch["x"] = batch["x"].to(device)
            batch["coords"] = batch["coords"].to(device)
            batch["y"] = batch["y"].to(device)
            batch["token_ids"] = batch["token_ids"].to(device)

            optim.zero_grad()
            mse, _, _ = forward_batch(
                predictor=predictor,
                batch=batch,
                tech_id=tech_id,
                organ_id=organ_id,
                max_genes=train_max_genes,
                max_spots=train_max_spots,
                random_gene_pick=True,
            )
            mse.backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            optim.step()

            tr_mse += float(mse.item())
            tr_steps += 1

        print(f"[epoch {epoch}/{max_epochs}] train_mse={tr_mse/max(tr_steps,1):.4f}")

        if val_loader is not None:
            predictor.eval()
            val_mse = 0.0
            val_steps = 0
            stats = RunningStats()
            with torch.no_grad():
                for batch in val_loader:
                    batch["x"] = batch["x"].to(device)
                    batch["coords"] = batch["coords"].to(device)
                    batch["y"] = batch["y"].to(device)
                    batch["token_ids"] = batch["token_ids"].to(device)

                    mse, pred, true = forward_batch(
                        predictor=predictor,
                        batch=batch,
                        tech_id=tech_id,
                        organ_id=organ_id,
                        max_genes=val_max_genes,
                        max_spots=val_max_spots,
                        random_gene_pick=False,
                    )
                    val_mse += float(mse.item())
                    val_steps += 1
                    stats.update(pred.flatten(), true.flatten())

            v_mse = val_mse / max(val_steps, 1)
            v_corr = stats.corr()
            print(f"  val_mse={v_mse:.4f} val_corr={v_corr:.4f}")
            metric = v_corr if np.isfinite(v_corr) else -v_mse
            if metric > best_metric:
                best_metric = metric
                torch.save(
                    {
                        "model": predictor.state_dict(),
                        "config": cfg,
                        "epoch": epoch,
                        "tech_id": tech_id,
                        "organ_id": organ_id,
                    },
                    best_path,
                )
                print(f"  [ckpt] saved best to {best_path}")
        else:
            torch.save(
                {
                    "model": predictor.state_dict(),
                    "config": cfg,
                    "epoch": epoch,
                    "tech_id": tech_id,
                    "organ_id": organ_id,
                },
                best_path,
            )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    args = ap.parse_args()
    train(args.config)
