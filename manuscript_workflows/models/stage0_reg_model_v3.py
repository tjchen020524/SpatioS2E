#!/usr/bin/env python3
"""Stage-0 regression v3: multi-step message passing GNN + FiLM."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Sequence

import torch
import torch.nn as nn

from models.decima_wrapper import DecimaSequenceWrapper

EPS = 1e-8


class GraphMessagePassingLayer(nn.Module):
    """One round of message passing with separate self/neighbor transforms."""

    def __init__(self, hidden_dim: int, dropout: float = 0.1, residual: bool = True):
        super().__init__()
        self.lin_self = nn.Linear(hidden_dim, hidden_dim)
        self.lin_nei = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.residual = residual

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        nei = torch.sparse.mm(adj, h)
        out = self.lin_self(h) + self.lin_nei(nei)
        out = self.norm(out)
        out = torch.relu(out)
        out = self.dropout(out)
        if self.residual:
            out = out + h
        return out


class SpatialBackboneV3(nn.Module):
    """True multi-layer graph message passing backbone."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        gnn_layers: int,
        dropout: float = 0.1,
        post_mlp_layers: int = 1,
        residual: bool = True,
    ):
        super().__init__()
        if gnn_layers < 1:
            raise ValueError("gnn_layers must be >= 1")
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        self.mp_layers = nn.ModuleList(
            [GraphMessagePassingLayer(hidden_dim, dropout=dropout, residual=residual) for _ in range(gnn_layers)]
        )

        mlp = []
        for _ in range(max(0, post_mlp_layers)):
            mlp.append(nn.Linear(hidden_dim, hidden_dim))
            mlp.append(nn.ReLU())
            mlp.append(nn.Dropout(dropout))
        self.post_mlp = nn.Sequential(*mlp) if mlp else nn.Identity()

    @staticmethod
    def _row_normalized_adj(
        n: int,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        src = edge_index[0]
        deg = torch.zeros(n, device=device, dtype=edge_weight.dtype)
        deg.index_add_(0, src, edge_weight)
        norm = edge_weight / deg[src].clamp_min(1.0e-8)
        return torch.sparse_coo_tensor(edge_index, norm, (n, n), device=device).coalesce()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        adj = self._row_normalized_adj(x.size(0), edge_index, edge_weight, x.device)
        for layer in self.mp_layers:
            h = layer(h, adj)
        return self.post_mlp(h)


class FiLMConditioner(nn.Module):
    def __init__(self, spot_dim: int, gene_dim: int, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(gene_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * spot_dim),
        )

    def forward(self, h_spot: torch.Tensor, e_gene: torch.Tensor) -> torch.Tensor:
        film = self.mlp(e_gene)
        scale, shift = film.chunk(2, dim=-1)
        h = h_spot.unsqueeze(1)
        return scale.unsqueeze(0) * h + shift.unsqueeze(0)


class SpotDeltaHead(nn.Module):
    """Predict only delta (additive in log-space)."""

    def __init__(self, in_dim: int, hidden_dim: int, init_std: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        last = self.net[-1]
        init_std = float(init_std)
        if init_std > 0:
            nn.init.normal_(last.weight, mean=0.0, std=init_std)
            nn.init.normal_(last.bias, mean=0.0, std=init_std)
        else:
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        n_spot, n_gene, hdim = h.shape
        return self.net(h.reshape(n_spot * n_gene, hdim)).view(n_spot, n_gene)


class Stage0RegModelV3(nn.Module):
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

    @staticmethod
    def _random_vec_for_gene(gid: str, dim: int, seed: int, device: torch.device) -> torch.Tensor:
        key = f"{gid}|{seed}".encode("utf-8")
        digest = hashlib.sha256(key).hexdigest()[:16]
        gseed = int(digest, 16) % (2**31 - 1)
        gen = torch.Generator(device="cpu")
        gen.manual_seed(gseed)
        vec = torch.randn(dim, generator=gen, dtype=torch.float32)
        return vec.to(device)

    def _apply_seq_ablation(
        self,
        e_gene: torch.Tensor,
        gene_ids: Sequence[str],
        mode: str,
        seed: int = 42,
    ) -> torch.Tensor:
        mode = str(mode).lower()
        if mode in {"none", "real", ""}:
            return e_gene
        if mode == "shuffle":
            gen = torch.Generator(device="cpu")
            gen.manual_seed(int(seed))
            perm = torch.randperm(e_gene.shape[0], generator=gen, device="cpu").to(e_gene.device)
            return e_gene[perm]
        if mode in {"noise", "random"}:
            gen = torch.Generator(device="cpu")
            gen.manual_seed(int(seed))
            z = torch.randn(e_gene.shape, generator=gen, dtype=e_gene.dtype, device="cpu")
            return z.to(e_gene.device)
        if mode == "random_per_gene":
            rows = [self._random_vec_for_gene(gid, e_gene.shape[1], int(seed), e_gene.device) for gid in gene_ids]
            return torch.stack(rows, dim=0)
        if mode in {"random_per_gene_matched", "random_matched"}:
            rows = [self._random_vec_for_gene(gid, e_gene.shape[1], int(seed), e_gene.device) for gid in gene_ids]
            z = torch.stack(rows, dim=0)
            mean = e_gene.mean(dim=0, keepdim=True)
            std = e_gene.std(dim=0, unbiased=False, keepdim=True).clamp_min(1.0e-6)
            return z * std + mean
        if mode == "zero":
            return torch.zeros_like(e_gene)
        raise ValueError(f"Unknown seq_ablation_mode={mode}")

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        gene_ids: Sequence[str],
        seq_ablation_mode: str = "none",
        seq_ablation_seed: int = 42,
    ) -> Dict[str, torch.Tensor]:
        device = x.device
        e_gene = self.decima.encode_genes(gene_ids, device=device)
        e_gene_used = self._apply_seq_ablation(
            e_gene=e_gene,
            gene_ids=gene_ids,
            mode=seq_ablation_mode,
            seed=seq_ablation_seed,
        )
        mu_pb, pb_logits = self.decima.forward_pseudobulk(e_gene_used)
        h_spot = self.spatial_backbone(x, edge_index, edge_weight)
        h_cond = self.conditioner(h_spot, e_gene_used)
        delta = self.spot_head(h_cond)
        log_mu_base = torch.log(mu_pb + EPS).unsqueeze(0)
        log_mu = log_mu_base + delta
        return {
            "e_gene": e_gene,
            "e_gene_used": e_gene_used,
            "mu_pb": mu_pb,
            "pb_logits": pb_logits,
            "h_spot": h_spot,
            "h_cond": h_cond,
            "delta": delta,
            "log_mu_base": log_mu_base,
            "log_mu": log_mu,
        }
