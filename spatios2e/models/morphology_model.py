#!/usr/bin/env python3
"""Morphology-only spatial graph baseline.

This ablation removes all sequence-derived quantities. It keeps the same
UNI2-h spot features and spatial graph backbone used by SpatioS2E, then
predicts requested genes with gene-specific trainable output weights.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

from spatios2e.models.residual_model import SpatialGraphEncoder


class MorphologyGraphModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        gnn_hidden: int,
        gnn_layers: int,
        gnn_dropout: float,
        spot_head_hidden: int,
        gene_ids_path: Path,
        gene_bias_init_path: Path | None = None,
        gene_bias_init_key: str = "log_mu_base",
        output_weight_init_std: float | None = None,
        clamp_min: float | None = None,
        gnn_post_mlp_layers: int = 1,
        gnn_residual: bool = True,
    ):
        super().__init__()
        gene_ids = [line.strip() for line in Path(gene_ids_path).read_text().splitlines() if line.strip()]
        if not gene_ids:
            raise ValueError(f"No gene IDs found in {gene_ids_path}")

        self.gene_ids = gene_ids
        self.gene_to_idx = {gid: i for i, gid in enumerate(gene_ids)}
        self.clamp_min = clamp_min
        self.spatial_backbone = SpatialGraphEncoder(
            in_dim=input_dim,
            hidden_dim=gnn_hidden,
            gnn_layers=gnn_layers,
            dropout=gnn_dropout,
            post_mlp_layers=gnn_post_mlp_layers,
            residual=gnn_residual,
        )
        self.spot_head = nn.Sequential(
            nn.Linear(gnn_hidden, spot_head_hidden),
            nn.ReLU(),
            nn.Dropout(gnn_dropout),
        )
        self.gene_weight = nn.Parameter(torch.empty(len(gene_ids), spot_head_hidden))
        self.gene_bias = nn.Parameter(torch.zeros(len(gene_ids)))
        if output_weight_init_std is None:
            nn.init.xavier_uniform_(self.gene_weight)
        else:
            nn.init.normal_(self.gene_weight, mean=0.0, std=float(output_weight_init_std))

        if gene_bias_init_path is not None:
            self._init_gene_bias(Path(gene_bias_init_path), gene_bias_init_key)

    def _init_gene_bias(self, path: Path, key: str) -> None:
        arr = np.load(path, allow_pickle=True)
        if key not in arr:
            raise KeyError(f"{path} does not contain key {key!r}; available keys: {sorted(arr.files)}")
        if "gene_ids" not in arr:
            raise KeyError(f"{path} does not contain gene_ids")
        source_gene_ids = arr["gene_ids"].astype(str).tolist()
        values = np.asarray(arr[key], dtype=np.float32)
        if values.shape[0] != len(source_gene_ids):
            raise ValueError(f"{key} length {values.shape[0]} does not match gene_ids length {len(source_gene_ids)}")
        source = {gid: float(v) for gid, v in zip(source_gene_ids, values.tolist())}
        missing = [gid for gid in self.gene_ids if gid not in source]
        if missing:
            preview = ", ".join(missing[:5])
            raise KeyError(f"{len(missing)} genes missing from bias init file {path}: {preview}")
        bias = torch.tensor([source[gid] for gid in self.gene_ids], dtype=self.gene_bias.dtype)
        with torch.no_grad():
            self.gene_bias.copy_(bias)

    def _gene_index(self, gene_ids: Sequence[str], device: torch.device) -> torch.Tensor:
        missing = [gid for gid in gene_ids if gid not in self.gene_to_idx]
        if missing:
            preview = ", ".join(missing[:5])
            raise KeyError(f"{len(missing)} requested genes are absent from model gene list: {preview}")
        return torch.tensor([self.gene_to_idx[gid] for gid in gene_ids], dtype=torch.long, device=device)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        gene_ids: Sequence[str],
        seq_ablation_mode: str = "none",
        seq_ablation_seed: int = 42,
    ) -> dict[str, torch.Tensor]:
        del seq_ablation_mode, seq_ablation_seed

        h_spot = self.spatial_backbone(x, edge_index, edge_weight)
        h = self.spot_head(h_spot)
        gene_idx = self._gene_index(gene_ids, x.device)
        weight = self.gene_weight.index_select(0, gene_idx)
        bias = self.gene_bias.index_select(0, gene_idx)
        pred = h @ weight.t() + bias.unsqueeze(0)
        if self.clamp_min is not None:
            pred = torch.clamp(pred, min=float(self.clamp_min))

        return {
            "log_mu": pred,
            "delta": pred,
            "h_spot": h_spot,
        }
