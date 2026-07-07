#!/usr/bin/env python3
"""Single-cell RNA prior extensions for SpatioS2E."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import torch
import torch.nn as nn

from spatios2e.models.spatios2e_model import SpatioS2EModel


class SnRNAGeneProfileEncoder(nn.Module):
    """Project low-dimensional donor snRNA gene features into Decima embedding space."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        hidden_dim = int(hidden_dim)
        if hidden_dim <= 0:
            self.net = nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.LayerNorm(output_dim),
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(float(dropout)),
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SingleCellPriorModel(SpatioS2EModel):
    """Replace Decima FiLM conditioning with donor-matched snRNA gene-profile embeddings.

    This keeps the residual-decomposition machinery intact.
    Decima is still loaded to provide the reference sequence embedding and pseudobulk
    diagnostics, but the spatial residual branch conditions on the snRNA MLP output.
    """

    def __init__(
        self,
        *args,
        snrna_gene_profile_path: Path,
        snrna_profile_hidden: int = 256,
        snrna_profile_dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.snrna_gene_to_idx = None
        self.snrna_donor_to_idx = None
        self.snrna_sample_to_donor = None
        self.snrna_feature_names: list[str] = []
        self._load_snrna_gene_profiles(
            Path(snrna_gene_profile_path),
            hidden_dim=int(snrna_profile_hidden),
            dropout=float(snrna_profile_dropout),
        )
        self.requires_sample_name = True

    @staticmethod
    def _zscore(x: np.ndarray, axis: int | None = None) -> np.ndarray:
        mean = np.mean(x, axis=axis, keepdims=True)
        std = np.std(x, axis=axis, keepdims=True)
        return (x - mean) / np.maximum(std, 1.0e-6)

    def _load_snrna_gene_profiles(
        self,
        path: Path,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        arr = np.load(path, allow_pickle=False)
        gene_ids = arr["gene_ids"].astype(str).tolist()
        donor_ids = arr["donor_ids"].astype(str).tolist()
        sample_ids = arr["sample_ids"].astype(str).tolist()
        sample_donor_ids = arr["sample_donor_ids"].astype(str).tolist()
        donor_snrna = arr["donor_log_mu_base"].astype(np.float32)

        if donor_snrna.shape != (len(donor_ids), len(gene_ids)):
            raise ValueError(
                "donor_log_mu_base shape does not match donor_ids/gene_ids: "
                f"{donor_snrna.shape} vs ({len(donor_ids)}, {len(gene_ids)})"
            )

        gene_mean = donor_snrna.mean(axis=0, keepdims=True)
        gene_std = donor_snrna.std(axis=0, keepdims=True)
        donor_dev = donor_snrna - gene_mean
        donor_dev_z = donor_dev / np.maximum(gene_std, 1.0e-6)
        raw_global_z = self._zscore(donor_snrna, axis=None)
        gene_mean_global_z = self._zscore(np.repeat(gene_mean, len(donor_ids), axis=0), axis=None)
        gene_std_global_z = self._zscore(np.repeat(gene_std, len(donor_ids), axis=0), axis=None)
        features = np.stack(
            [
                raw_global_z,
                donor_dev_z,
                gene_mean_global_z,
                gene_std_global_z,
            ],
            axis=-1,
        ).astype(np.float32)

        self.snrna_gene_to_idx = {gid: i for i, gid in enumerate(gene_ids)}
        self.snrna_donor_to_idx = {donor: i for i, donor in enumerate(donor_ids)}
        self.snrna_sample_to_donor = {
            sample: donor for sample, donor in zip(sample_ids, sample_donor_ids)
        }
        self.snrna_feature_names = [
            "raw_global_z",
            "donor_dev_gene_z",
            "gene_mean_global_z",
            "gene_std_global_z",
        ]
        self.register_buffer(
            "snrna_gene_features",
            torch.tensor(features, dtype=torch.float32),
            persistent=False,
        )
        self.snrna_profile_encoder = SnRNAGeneProfileEncoder(
            input_dim=features.shape[-1],
            output_dim=self.decima.embed_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    def _lookup_snrna_gene_features(
        self,
        gene_ids: Sequence[str],
        sample_name: str | None,
        device: torch.device,
    ) -> torch.Tensor:
        if sample_name is None:
            raise ValueError("sample_name is required for donor-matched snRNA gene profiles")
        assert self.snrna_sample_to_donor is not None
        assert self.snrna_donor_to_idx is not None
        assert self.snrna_gene_to_idx is not None
        if sample_name not in self.snrna_sample_to_donor:
            raise KeyError(f"Missing snRNA donor mapping for sample {sample_name}")
        donor = self.snrna_sample_to_donor[sample_name]
        if donor not in self.snrna_donor_to_idx:
            raise KeyError(f"Missing snRNA donor profile for donor {donor}")
        missing = [gid for gid in gene_ids if gid not in self.snrna_gene_to_idx]
        if missing:
            preview = ", ".join(missing[:5])
            raise KeyError(f"Missing snRNA gene profile for {len(missing)} genes, e.g. {preview}")
        donor_idx = self.snrna_donor_to_idx[donor]
        gene_idx = [self.snrna_gene_to_idx[gid] for gid in gene_ids]
        return self.snrna_gene_features[donor_idx, gene_idx].to(device=device)

    def encode_snrna_genes(
        self,
        gene_ids: Sequence[str],
        sample_name: str | None,
        device: torch.device,
    ) -> torch.Tensor:
        features = self._lookup_snrna_gene_features(gene_ids, sample_name=sample_name, device=device)
        return self.snrna_profile_encoder(features)

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
        if self.baseline_only:
            raise ValueError("SingleCellPriorModel is intended for residual prediction, not baseline-only use")

        device = x.device
        e_gene_decima = self.decima.encode_genes(gene_ids, device=device)
        e_gene_snrna = self.encode_snrna_genes(gene_ids, sample_name=sample_name, device=device)
        if seq_ablation_mode not in {"none", "real", ""}:
            e_gene_snrna = self._apply_seq_ablation(
                e_gene=e_gene_snrna,
                gene_ids=gene_ids,
                mode=seq_ablation_mode,
                seed=seq_ablation_seed,
            )

        mu_pb, pb_logits = self.decima.forward_pseudobulk(e_gene_decima)
        h_spot = self.spatial_backbone(x, edge_index, edge_weight)
        h_cond = self.conditioner(h_spot, e_gene_snrna)
        delta = self.spot_head(h_cond)
        log_mu_base_decima = torch.log(mu_pb + 1.0e-8).unsqueeze(0)
        if self.log_mu_base_map is not None or self.donor_log_mu_base is not None:
            log_mu_base = self._lookup_configured_log_mu_base(
                gene_ids,
                device=device,
                sample_name=sample_name,
            )
        else:
            log_mu_base = log_mu_base_decima

        residual_scale_prior = self._lookup_residual_scale(gene_ids, device=device)
        residual_scale = residual_scale_prior
        out: Dict[str, torch.Tensor] = {
            "e_gene": e_gene_decima,
            "e_gene_decima": e_gene_decima,
            "e_gene_used": e_gene_snrna,
            "e_gene_snrna": e_gene_snrna,
            "mu_pb": mu_pb,
            "pb_logits": pb_logits,
            "h_spot": h_spot,
            "h_cond": h_cond,
            "delta": delta,
            "log_mu_base_decima": log_mu_base_decima,
            "log_mu_base": log_mu_base,
        }
        if self.learnable_residual_scale:
            residual_scale_logcorr = self.residual_scale_head(e_gene_snrna)
            residual_scale_logcorr = residual_scale_logcorr.clamp(
                min=-self.residual_scale_logscale_clip,
                max=self.residual_scale_logscale_clip,
            )
            residual_scale = residual_scale_prior * torch.exp(residual_scale_logcorr)
            out["residual_scale_logcorr"] = residual_scale_logcorr

        delta_effective = delta * residual_scale.unsqueeze(0)
        out["delta_scale_prior"] = residual_scale_prior
        out["delta_scale"] = residual_scale
        out["delta_effective"] = delta_effective
        out["log_mu"] = log_mu_base + delta_effective
        return out


class GatedSingleCellPriorModel(SingleCellPriorModel):
    """Use Decima embeddings with a learnable gated snRNA additive prior."""

    def __init__(
        self,
        *args,
        snrna_gate_init: float = 0.02,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        init = float(snrna_gate_init)
        init = min(max(init, 1.0e-4), 1.0 - 1.0e-4)
        self.snrna_gate_logit = nn.Parameter(torch.tensor(np.log(init / (1.0 - init)), dtype=torch.float32))

    def fuse_gene_embeddings(
        self,
        e_gene_decima: torch.Tensor,
        e_gene_snrna: torch.Tensor,
    ) -> torch.Tensor:
        gate = torch.sigmoid(self.snrna_gate_logit)
        return e_gene_decima + gate * e_gene_snrna

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
        if self.baseline_only:
            raise ValueError("GatedSingleCellPriorModel is intended for residual prediction, not baseline-only use")

        device = x.device
        e_gene_decima = self.decima.encode_genes(gene_ids, device=device)
        e_gene_snrna = self.encode_snrna_genes(gene_ids, sample_name=sample_name, device=device)
        e_gene_fused = self.fuse_gene_embeddings(e_gene_decima, e_gene_snrna)
        if seq_ablation_mode not in {"none", "real", ""}:
            e_gene_fused = self._apply_seq_ablation(
                e_gene=e_gene_fused,
                gene_ids=gene_ids,
                mode=seq_ablation_mode,
                seed=seq_ablation_seed,
            )

        mu_pb, pb_logits = self.decima.forward_pseudobulk(e_gene_decima)
        h_spot = self.spatial_backbone(x, edge_index, edge_weight)
        h_cond = self.conditioner(h_spot, e_gene_fused)
        delta = self.spot_head(h_cond)
        log_mu_base_decima = torch.log(mu_pb + 1.0e-8).unsqueeze(0)
        if self.log_mu_base_map is not None or self.donor_log_mu_base is not None:
            log_mu_base = self._lookup_configured_log_mu_base(
                gene_ids,
                device=device,
                sample_name=sample_name,
            )
        else:
            log_mu_base = log_mu_base_decima

        residual_scale_prior = self._lookup_residual_scale(gene_ids, device=device)
        residual_scale = residual_scale_prior
        out: Dict[str, torch.Tensor] = {
            "e_gene": e_gene_decima,
            "e_gene_decima": e_gene_decima,
            "e_gene_used": e_gene_fused,
            "e_gene_snrna": e_gene_snrna,
            "mu_pb": mu_pb,
            "pb_logits": pb_logits,
            "h_spot": h_spot,
            "h_cond": h_cond,
            "delta": delta,
            "log_mu_base_decima": log_mu_base_decima,
            "log_mu_base": log_mu_base,
            "snrna_gate": torch.sigmoid(self.snrna_gate_logit),
        }
        if self.learnable_residual_scale:
            residual_scale_logcorr = self.residual_scale_head(e_gene_fused)
            residual_scale_logcorr = residual_scale_logcorr.clamp(
                min=-self.residual_scale_logscale_clip,
                max=self.residual_scale_logscale_clip,
            )
            residual_scale = residual_scale_prior * torch.exp(residual_scale_logcorr)
            out["residual_scale_logcorr"] = residual_scale_logcorr

        delta_effective = delta * residual_scale.unsqueeze(0)
        out["delta_scale_prior"] = residual_scale_prior
        out["delta_scale"] = residual_scale
        out["delta_effective"] = delta_effective
        out["log_mu"] = log_mu_base + delta_effective
        return out


class ScGPTGeneContextPriorModel(SingleCellPriorModel):
    """Replace Decima FiLM conditioning with donor-specific scGPT gene context."""

    def __init__(
        self,
        *args,
        snrna_scgpt_include_expr_features: bool = True,
        **kwargs,
    ):
        self.snrna_scgpt_include_expr_features = bool(snrna_scgpt_include_expr_features)
        super().__init__(*args, **kwargs)

    def _load_snrna_gene_profiles(
        self,
        path: Path,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        arr = np.load(path, allow_pickle=False)
        gene_ids = arr["gene_ids"].astype(str).tolist()
        donor_ids = arr["donor_ids"].astype(str).tolist()
        sample_ids = arr["sample_ids"].astype(str).tolist()
        sample_donor_ids = arr["sample_donor_ids"].astype(str).tolist()
        context = arr["donor_scgpt_gene_context"].astype(np.float32)
        if context.shape[:2] != (len(donor_ids), len(gene_ids)):
            raise ValueError(
                "donor_scgpt_gene_context shape does not match donor_ids/gene_ids: "
                f"{context.shape} vs ({len(donor_ids)}, {len(gene_ids)}, dim)"
            )

        flat = context.reshape(-1, context.shape[-1])
        context_mean = flat.mean(axis=0, keepdims=True)
        context_std = flat.std(axis=0, keepdims=True)
        features = ((context - context_mean.reshape(1, 1, -1)) / np.maximum(
            context_std.reshape(1, 1, -1),
            1.0e-6,
        )).astype(np.float32)
        feature_names = [f"scgpt_context_z_{idx}" for idx in range(features.shape[-1])]

        if self.snrna_scgpt_include_expr_features:
            expr_sum = arr["donor_scgpt_gene_expr_sum"].astype(np.float32)
            if expr_sum.shape != (len(donor_ids), len(gene_ids)):
                raise ValueError(
                    "donor_scgpt_gene_expr_sum shape does not match donor_ids/gene_ids: "
                    f"{expr_sum.shape} vs ({len(donor_ids)}, {len(gene_ids)})"
                )
            log_expr = np.log1p(expr_sum)
            global_z = self._zscore(log_expr, axis=None)
            gene_mean = log_expr.mean(axis=0, keepdims=True)
            gene_std = log_expr.std(axis=0, keepdims=True)
            donor_dev_z = (log_expr - gene_mean) / np.maximum(gene_std, 1.0e-6)
            expr_features = np.stack([global_z, donor_dev_z], axis=-1).astype(np.float32)
            features = np.concatenate([features, expr_features], axis=-1)
            feature_names.extend(["scgpt_gene_log_expr_global_z", "scgpt_gene_log_expr_donor_dev_z"])

        self.snrna_gene_to_idx = {gid: i for i, gid in enumerate(gene_ids)}
        self.snrna_donor_to_idx = {donor: i for i, donor in enumerate(donor_ids)}
        self.snrna_sample_to_donor = {
            sample: donor for sample, donor in zip(sample_ids, sample_donor_ids)
        }
        self.snrna_feature_names = feature_names
        self.register_buffer(
            "snrna_gene_features",
            torch.tensor(features, dtype=torch.float32),
            persistent=False,
        )
        self.snrna_profile_encoder = SnRNAGeneProfileEncoder(
            input_dim=features.shape[-1],
            output_dim=self.decima.embed_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )


class GatedScGPTGeneContextPriorModel(GatedSingleCellPriorModel):
    """Use Decima embeddings with a learnable gated scGPT gene-context prior."""

    def __init__(
        self,
        *args,
        snrna_scgpt_include_expr_features: bool = True,
        **kwargs,
    ):
        self.snrna_scgpt_include_expr_features = bool(snrna_scgpt_include_expr_features)
        super().__init__(*args, **kwargs)

    _load_snrna_gene_profiles = ScGPTGeneContextPriorModel._load_snrna_gene_profiles
