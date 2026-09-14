#!/usr/bin/env python3
"""Stage-0 regression cross-attention fusion variants."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from models.stage0_reg_model_v11 import Stage0RegModelV11


class CrossAttentionConditioner(nn.Module):
    """Fuse spot and gene embeddings with gene-to-spot feature-token attention."""

    def __init__(
        self,
        spot_dim: int,
        gene_dim: int,
        hidden_dim: int,
        num_heads: int = 4,
        num_spot_tokens: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim)
        num_heads = int(num_heads)
        num_spot_tokens = int(num_spot_tokens)
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if num_spot_tokens <= 0:
            raise ValueError("num_spot_tokens must be positive")
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
            )

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_spot_tokens = num_spot_tokens
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim**-0.5

        self.spot_norm = nn.LayerNorm(spot_dim)
        self.gene_norm = nn.LayerNorm(gene_dim)
        self.gene_query = nn.Linear(gene_dim, hidden_dim)
        self.spot_key = nn.Linear(spot_dim, num_spot_tokens * hidden_dim)
        self.spot_value = nn.Linear(spot_dim, num_spot_tokens * hidden_dim)
        self.context_out = nn.Linear(hidden_dim, spot_dim)
        self.spot_skip = nn.Linear(spot_dim, spot_dim)
        self.gene_skip = nn.Linear(gene_dim, spot_dim)
        self.dropout = nn.Dropout(dropout)
        self.out_norm = nn.LayerNorm(spot_dim)

    def forward(self, h_spot: torch.Tensor, e_gene: torch.Tensor) -> torch.Tensor:
        n_spot = h_spot.shape[0]
        n_gene = e_gene.shape[0]

        spot = self.spot_norm(h_spot)
        gene = self.gene_norm(e_gene)

        q = self.gene_query(gene).view(n_gene, self.num_heads, self.head_dim)
        k = self.spot_key(spot).view(
            n_spot,
            self.num_spot_tokens,
            self.num_heads,
            self.head_dim,
        )
        v = self.spot_value(spot).view(
            n_spot,
            self.num_spot_tokens,
            self.num_heads,
            self.head_dim,
        )

        scores = torch.einsum("ghd,nthd->nght", q, k) * self.scale
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        context = torch.einsum("nght,nthd->nghd", attn, v)
        context = context.reshape(n_spot, n_gene, self.hidden_dim)

        fused = (
            self.spot_skip(spot).unsqueeze(1)
            + self.gene_skip(gene).unsqueeze(0)
            + self.dropout(self.context_out(context))
        )
        return self.out_norm(fused)


class Stage0RegModelV11CrossAttention(Stage0RegModelV11):
    """v11/v11b residual-scale model with cross-attention spot-gene fusion."""

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
        log_mu_base_path: Path | None = None,
        donor_log_mu_base_path: Path | None = None,
        mu_only: bool = False,
        cross_attn_hidden: int | None = None,
        cross_attn_heads: int = 4,
        cross_attn_spot_tokens: int = 4,
        cross_attn_dropout: float | None = None,
    ) -> None:
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
            log_mu_base_path=log_mu_base_path,
            donor_log_mu_base_path=donor_log_mu_base_path,
            mu_only=mu_only,
        )
        attn_hidden = int(film_hidden if cross_attn_hidden is None else cross_attn_hidden)
        attn_dropout = float(gnn_dropout if cross_attn_dropout is None else cross_attn_dropout)
        self.conditioner = CrossAttentionConditioner(
            spot_dim=gnn_hidden,
            gene_dim=self.decima.embed_dim,
            hidden_dim=attn_hidden,
            num_heads=int(cross_attn_heads),
            num_spot_tokens=int(cross_attn_spot_tokens),
            dropout=attn_dropout,
        )
