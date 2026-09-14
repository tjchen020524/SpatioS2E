#!/usr/bin/env python3
"""Stage-0 regression v7: v4b backbone plus gene-wise output calibration."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

import torch
import torch.nn as nn

from models.decima_wrapper import DecimaSequenceWrapper
from models.stage0_reg_model_v3 import EPS, FiLMConditioner, SpatialBackboneV3, SpotDeltaHead


class GeneCalibrationHead(nn.Module):
    """Predict gene-wise calibration parameters around identity."""

    def __init__(self, gene_dim: int, hidden_dim: int = 128):
        super().__init__()
        if int(hidden_dim) > 0:
            self.net = nn.Sequential(
                nn.Linear(gene_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 3),
            )
            last = self.net[-1]
        else:
            self.net = nn.Linear(gene_dim, 3)
            last = self.net
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(self, e_gene: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raw = self.net(e_gene)
        raw_base_logscale, raw_delta_logscale, bias = raw.unbind(dim=-1)
        return raw_base_logscale, raw_delta_logscale, bias


class Stage0RegModelV7(nn.Module):
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
        calibration_hidden: int = 128,
        calibration_logscale_clip: float = 2.0,
    ):
        super().__init__()
        self.spatial_backbone = SpatialBackboneV3(
            in_dim=input_dim,
            hidden_dim=gnn_hidden,
            gnn_layers=gnn_layers,
            dropout=gnn_dropout,
            post_mlp_layers=gnn_post_mlp_layers,
            residual=gnn_residual,
        )
        self.decima = DecimaSequenceWrapper(
            ckpt_path=decima_ckpt,
            h5_path=decima_h5,
            npz_dir=decima_npz_dir,
            freeze_backbone=freeze_decima_backbone,
            freeze_head=freeze_pseudobulk_head,
        )
        gene_dim = self.decima.embed_dim
        self.conditioner = FiLMConditioner(spot_dim=gnn_hidden, gene_dim=gene_dim, hidden_dim=film_hidden)
        self.spot_head = SpotDeltaHead(
            in_dim=gnn_hidden,
            hidden_dim=spot_head_hidden,
            init_std=float(delta_head_init_std),
        )
        self.calibration_head = GeneCalibrationHead(gene_dim=gene_dim, hidden_dim=int(calibration_hidden))
        self.calibration_logscale_clip = float(calibration_logscale_clip)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        gene_ids: Sequence[str],
    ) -> Dict[str, torch.Tensor]:
        device = x.device
        e_gene = self.decima.encode_genes(gene_ids, device=device)
        mu_pb, pb_logits = self.decima.forward_pseudobulk(e_gene)
        h_spot = self.spatial_backbone(x, edge_index, edge_weight)
        h_cond = self.conditioner(h_spot, e_gene)
        delta = self.spot_head(h_cond)

        log_mu_base = torch.log(mu_pb + EPS)
        raw_base_logscale, raw_delta_logscale, calib_bias = self.calibration_head(e_gene)
        raw_base_logscale = raw_base_logscale.clamp(
            min=-self.calibration_logscale_clip,
            max=self.calibration_logscale_clip,
        )
        raw_delta_logscale = raw_delta_logscale.clamp(
            min=-self.calibration_logscale_clip,
            max=self.calibration_logscale_clip,
        )
        base_scale = torch.exp(raw_base_logscale)
        delta_scale = torch.exp(raw_delta_logscale)

        log_mu_uncalibrated = log_mu_base.unsqueeze(0) + delta
        log_mu = (
            base_scale.unsqueeze(0) * log_mu_base.unsqueeze(0)
            + delta_scale.unsqueeze(0) * delta
            + calib_bias.unsqueeze(0)
        )
        return {
            "e_gene": e_gene,
            "mu_pb": mu_pb,
            "pb_logits": pb_logits,
            "h_spot": h_spot,
            "h_cond": h_cond,
            "delta": delta,
            "log_mu_base": log_mu_base.unsqueeze(0),
            "log_mu_uncalibrated": log_mu_uncalibrated,
            "base_scale": base_scale,
            "delta_scale": delta_scale,
            "calib_bias": calib_bias,
            "raw_base_logscale": raw_base_logscale,
            "raw_delta_logscale": raw_delta_logscale,
            "log_mu": log_mu,
        }
