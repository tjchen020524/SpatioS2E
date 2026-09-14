#!/usr/bin/env python3
"""Stage-0 regression concat-fusion baselines."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from models.stage0_reg_model_v11 import Stage0RegModelV11


class ConcatConditioner(nn.Module):
    """MLP([spot, gene]) conditioner with an equivalent memory-efficient first layer."""

    def __init__(self, spot_dim: int, gene_dim: int, hidden_dim: int):
        super().__init__()
        self.spot_linear = nn.Linear(spot_dim, hidden_dim, bias=False)
        self.gene_linear = nn.Linear(gene_dim, hidden_dim, bias=True)
        self.out = nn.Linear(hidden_dim, spot_dim)

    def forward(self, h_spot: torch.Tensor, e_gene: torch.Tensor) -> torch.Tensor:
        h = self.spot_linear(h_spot).unsqueeze(1) + self.gene_linear(e_gene).unsqueeze(0)
        return self.out(torch.relu(h))


class Stage0RegModelV11Concat(Stage0RegModelV11):
    """v11/v11b residual-scale model with direct concat spot-gene fusion."""

    def __init__(
        self,
        input_dim: int,
        gnn_hidden: int,
        gnn_layers: int,
        gnn_dropout: float,
        film_hidden: int,
        spot_head_hidden: int,
        decima_ckpt: Path,
        decima_h5: Path | None = None,
        decima_npz_dir: Path | None = None,
        freeze_decima_backbone: bool = True,
        freeze_pseudobulk_head: bool = False,
        gnn_post_mlp_layers: int = 1,
        gnn_residual: bool = True,
        delta_head_init_std: float = 0.0,
        residual_scale_path: Path | None = None,
        residual_scale_floor: float = 1.0e-3,
        learnable_residual_scale: bool = False,
        residual_scale_hidden: int = 128,
        residual_scale_logscale_clip: float = 1.0,
    ):
        super().__init__(
            input_dim=input_dim,
            gnn_hidden=gnn_hidden,
            gnn_layers=gnn_layers,
            gnn_dropout=gnn_dropout,
            film_hidden=film_hidden,
            spot_head_hidden=spot_head_hidden,
            decima_ckpt=decima_ckpt,
            decima_h5=decima_h5,
            decima_npz_dir=decima_npz_dir,
            freeze_decima_backbone=freeze_decima_backbone,
            freeze_pseudobulk_head=freeze_pseudobulk_head,
            gnn_post_mlp_layers=gnn_post_mlp_layers,
            gnn_residual=gnn_residual,
            delta_head_init_std=delta_head_init_std,
            residual_scale_path=residual_scale_path,
            residual_scale_floor=residual_scale_floor,
            learnable_residual_scale=learnable_residual_scale,
            residual_scale_hidden=residual_scale_hidden,
            residual_scale_logscale_clip=residual_scale_logscale_clip,
        )
        self.conditioner = ConcatConditioner(
            spot_dim=gnn_hidden,
            gene_dim=self.decima.embed_dim,
            hidden_dim=film_hidden,
        )
