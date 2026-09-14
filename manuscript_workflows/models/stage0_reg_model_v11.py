#!/usr/bin/env python3
"""Stage-0 regression v11: predict standardized residual with fixed gene-wise scale."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import torch
import torch.nn as nn

from models.stage0_reg_model_v3 import Stage0RegModelV3


class GeneResidualScaleHead(nn.Module):
    """Predict a small gene-wise log-scale correction around a fixed prior."""

    def __init__(self, gene_dim: int, hidden_dim: int = 128):
        super().__init__()
        if int(hidden_dim) > 0:
            self.net = nn.Sequential(
                nn.Linear(gene_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
            last = self.net[-1]
        else:
            self.net = nn.Linear(gene_dim, 1)
            last = self.net
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(self, e_gene: torch.Tensor) -> torch.Tensor:
        return self.net(e_gene).squeeze(-1)


class Stage0RegModelV11(Stage0RegModelV3):
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
        )
        if residual_scale_path is None:
            raise ValueError("residual_scale_path is required for v11")
        arr = np.load(residual_scale_path, allow_pickle=False)
        gene_ids = arr["gene_ids"].astype(str).tolist()
        scales = arr["residual_scale"].astype(np.float32)
        floor = float(residual_scale_floor)
        self.residual_scale_floor = floor
        self.residual_scale_map = {
            gid: max(float(scale), floor) for gid, scale in zip(gene_ids, scales)
        }
        self.learnable_residual_scale = bool(learnable_residual_scale)
        self.mu_only = bool(mu_only)
        self.residual_scale_logscale_clip = float(residual_scale_logscale_clip)
        self.log_mu_base_map = None
        self.donor_log_mu_base = None
        self.donor_log_mu_gene_to_idx = None
        self.donor_log_mu_donor_to_idx = None
        self.sample_to_donor = None
        self.requires_sample_name = False
        if log_mu_base_path is not None:
            base_arr = np.load(log_mu_base_path, allow_pickle=False)
            base_gene_ids = base_arr["gene_ids"].astype(str).tolist()
            base_vals = base_arr["log_mu_base"].astype(np.float32)
            self.log_mu_base_map = {gid: float(val) for gid, val in zip(base_gene_ids, base_vals)}
        if donor_log_mu_base_path is not None:
            donor_arr = np.load(donor_log_mu_base_path, allow_pickle=False)
            donor_gene_ids = donor_arr["gene_ids"].astype(str).tolist()
            donor_ids = donor_arr["donor_ids"].astype(str).tolist()
            sample_ids = donor_arr["sample_ids"].astype(str).tolist()
            sample_donor_ids = donor_arr["sample_donor_ids"].astype(str).tolist()
            self.donor_log_mu_base = donor_arr["donor_log_mu_base"].astype(np.float32)
            self.donor_log_mu_gene_to_idx = {gid: i for i, gid in enumerate(donor_gene_ids)}
            self.donor_log_mu_donor_to_idx = {donor: i for i, donor in enumerate(donor_ids)}
            self.sample_to_donor = {
                sample: donor for sample, donor in zip(sample_ids, sample_donor_ids)
            }
            self.requires_sample_name = True
        if self.learnable_residual_scale:
            self.residual_scale_head = GeneResidualScaleHead(
                gene_dim=self.decima.embed_dim,
                hidden_dim=int(residual_scale_hidden),
            )

    def _lookup_residual_scale(self, gene_ids: Sequence[str], device: torch.device) -> torch.Tensor:
        missing = [gid for gid in gene_ids if gid not in self.residual_scale_map]
        if missing:
            preview = ", ".join(missing[:5])
            raise KeyError(f"Missing residual scale for {len(missing)} genes, e.g. {preview}")
        vals = [self.residual_scale_map[gid] for gid in gene_ids]
        return torch.tensor(vals, dtype=torch.float32, device=device)

    def _lookup_log_mu_base(self, gene_ids: Sequence[str], device: torch.device) -> torch.Tensor:
        if self.log_mu_base_map is None:
            raise RuntimeError("log_mu_base_map is not configured")
        missing = [gid for gid in gene_ids if gid not in self.log_mu_base_map]
        if missing:
            preview = ", ".join(missing[:5])
            raise KeyError(f"Missing log_mu_base for {len(missing)} genes, e.g. {preview}")
        vals = [self.log_mu_base_map[gid] for gid in gene_ids]
        return torch.tensor(vals, dtype=torch.float32, device=device).unsqueeze(0)

    def _lookup_donor_log_mu_base(
        self,
        gene_ids: Sequence[str],
        sample_name: str | None,
        device: torch.device,
    ) -> torch.Tensor:
        if self.donor_log_mu_base is None:
            raise RuntimeError("donor_log_mu_base is not configured")
        if sample_name is None:
            raise ValueError("sample_name is required for donor-matched log_mu_base")
        assert self.sample_to_donor is not None
        assert self.donor_log_mu_donor_to_idx is not None
        assert self.donor_log_mu_gene_to_idx is not None
        if sample_name not in self.sample_to_donor:
            raise KeyError(f"Missing donor mapping for sample {sample_name}")
        donor_id = self.sample_to_donor[sample_name]
        if donor_id not in self.donor_log_mu_donor_to_idx:
            raise KeyError(f"Missing donor baseline for donor {donor_id}")
        missing = [gid for gid in gene_ids if gid not in self.donor_log_mu_gene_to_idx]
        if missing:
            preview = ", ".join(missing[:5])
            raise KeyError(f"Missing donor log_mu_base for {len(missing)} genes, e.g. {preview}")
        donor_idx = self.donor_log_mu_donor_to_idx[donor_id]
        gene_idx = [self.donor_log_mu_gene_to_idx[gid] for gid in gene_ids]
        vals = self.donor_log_mu_base[donor_idx, gene_idx]
        return torch.tensor(vals, dtype=torch.float32, device=device).unsqueeze(0)

    def _lookup_configured_log_mu_base(
        self,
        gene_ids: Sequence[str],
        device: torch.device,
        sample_name: str | None = None,
    ) -> torch.Tensor:
        if self.donor_log_mu_base is not None:
            return self._lookup_donor_log_mu_base(gene_ids, sample_name=sample_name, device=device)
        if self.log_mu_base_map is not None:
            return self._lookup_log_mu_base(gene_ids, device=device)
        raise RuntimeError("No configured log_mu_base override")

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        gene_ids: Sequence[str],
        seq_ablation_mode: str = "none",
        seq_ablation_seed: int = 42,
        sample_name: str | None = None,
    ) -> Dict[str, torch.Tensor]:
        if self.mu_only:
            device = x.device
            e_gene = self.decima.encode_genes(gene_ids, device=device)
            e_gene_used = self._apply_seq_ablation(
                e_gene=e_gene,
                gene_ids=gene_ids,
                mode=seq_ablation_mode,
                seed=seq_ablation_seed,
            )
            mu_pb, pb_logits = self.decima.forward_pseudobulk(e_gene_used)
            log_mu_base = torch.log(mu_pb + 1.0e-8).unsqueeze(0)
            if self.log_mu_base_map is not None or self.donor_log_mu_base is not None:
                log_mu_base = self._lookup_configured_log_mu_base(
                    gene_ids,
                    device=device,
                    sample_name=sample_name,
                )
            residual_scale_prior = self._lookup_residual_scale(gene_ids, device=device)
            delta = torch.zeros((x.shape[0], len(gene_ids)), dtype=x.dtype, device=device)
            log_mu = log_mu_base.expand_as(delta)
            return {
                "e_gene": e_gene,
                "e_gene_used": e_gene_used,
                "mu_pb": mu_pb,
                "pb_logits": pb_logits,
                "delta": delta,
                "log_mu_base": log_mu_base,
                "log_mu": log_mu,
                "delta_scale_prior": residual_scale_prior,
                "delta_scale": residual_scale_prior,
                "delta_effective": delta,
            }
        out = super().forward(
            x=x,
            edge_index=edge_index,
            edge_weight=edge_weight,
            gene_ids=gene_ids,
            seq_ablation_mode=seq_ablation_mode,
            seq_ablation_seed=seq_ablation_seed,
        )
        if self.log_mu_base_map is not None or self.donor_log_mu_base is not None:
            out["log_mu_base_decima"] = out["log_mu_base"]
            out["log_mu_base"] = self._lookup_configured_log_mu_base(
                gene_ids,
                device=x.device,
                sample_name=sample_name,
            )
        residual_scale_prior = self._lookup_residual_scale(gene_ids, device=x.device)
        residual_scale = residual_scale_prior
        if self.learnable_residual_scale:
            residual_scale_logcorr = self.residual_scale_head(out["e_gene_used"])
            residual_scale_logcorr = residual_scale_logcorr.clamp(
                min=-self.residual_scale_logscale_clip,
                max=self.residual_scale_logscale_clip,
            )
            residual_scale = residual_scale_prior * torch.exp(residual_scale_logcorr)
            out["residual_scale_logcorr"] = residual_scale_logcorr
        delta_effective = out["delta"] * residual_scale.unsqueeze(0)
        out["delta_scale_prior"] = residual_scale_prior
        out["delta_scale"] = residual_scale
        out["delta_effective"] = delta_effective
        if self.mu_only:
            out["delta"] = torch.zeros_like(out["delta"])
            out["delta_effective"] = torch.zeros_like(delta_effective)
            out["log_mu"] = out["log_mu_base"].expand_as(delta_effective)
        else:
            out["log_mu"] = out["log_mu_base"] + delta_effective
        return out
