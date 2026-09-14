#!/usr/bin/env python3
"""Factory for stage-0 regression model variants."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

from models.stage0_reg_model_v3 import Stage0RegModelV3
from models.stage0_reg_model_v11 import Stage0RegModelV11
from models.stage0_reg_model_v7 import Stage0RegModelV7
from models.stage0_reg_model_concat import Stage0RegModelV11Concat
from models.stage0_reg_model_crossattn import Stage0RegModelV11CrossAttention
from models.stage0_image_graph_model import Stage0ImageGraphOnly
from models.stage0_reg_model_snrna import (
    Stage0RegModelV11SnRNAGatedFusion,
    Stage0RegModelV11SnRNAGeneMLP,
    Stage0RegModelV11SnRNAScGPTGeneGatedFusion,
    Stage0RegModelV11SnRNAScGPTGeneMLP,
)


def build_stage0_reg_model(cfg: Dict, input_dim: int):
    model_cfg = cfg["model"]
    paths_cfg = cfg["paths"]
    variant = str(model_cfg.get("variant", "v3")).lower()
    if variant in {"image_graph_only", "image_graph", "no_sequence_graph"}:
        return Stage0ImageGraphOnly(
            input_dim=input_dim,
            gnn_hidden=int(model_cfg["gnn_hidden"]),
            gnn_layers=int(model_cfg["gnn_layers"]),
            gnn_dropout=float(model_cfg["gnn_dropout"]),
            spot_head_hidden=int(model_cfg["spot_head_hidden"]),
            gene_ids_path=Path(paths_cfg["gene_split_dir"]) / "train_genes.txt",
            gene_bias_init_path=Path(paths_cfg["gene_bias_init_npz"]) if paths_cfg.get("gene_bias_init_npz") else None,
            gene_bias_init_key=str(model_cfg.get("gene_bias_init_key", "log_mu_base")),
            output_weight_init_std=(
                float(model_cfg["output_weight_init_std"])
                if model_cfg.get("output_weight_init_std") is not None
                else None
            ),
            clamp_min=(
                float(model_cfg["clamp_min"])
                if model_cfg.get("clamp_min") is not None
                else None
            ),
            gnn_post_mlp_layers=int(model_cfg.get("gnn_post_mlp_layers", 1)),
            gnn_residual=bool(model_cfg.get("gnn_residual", True)),
        )

    common_kwargs = dict(
        input_dim=input_dim,
        gnn_hidden=int(model_cfg["gnn_hidden"]),
        gnn_layers=int(model_cfg["gnn_layers"]),
        gnn_dropout=float(model_cfg["gnn_dropout"]),
        film_hidden=int(model_cfg["film_hidden"]),
        spot_head_hidden=int(model_cfg["spot_head_hidden"]),
        decima_ckpt=Path(paths_cfg["decima_ckpt"]),
        decima_h5=Path(paths_cfg["decima_h5"]) if paths_cfg.get("decima_h5") else None,
        decima_npz_dir=Path(paths_cfg["decima_npz_dir"]) if paths_cfg.get("decima_npz_dir") else None,
        freeze_decima_backbone=bool(model_cfg.get("freeze_decima_backbone", True)),
        freeze_pseudobulk_head=bool(model_cfg.get("freeze_pseudobulk_head", False)),
        gnn_post_mlp_layers=int(model_cfg.get("gnn_post_mlp_layers", 1)),
        gnn_residual=bool(model_cfg.get("gnn_residual", True)),
        delta_head_init_std=float(model_cfg.get("delta_head_init_std", 0.0)),
    )
    if variant in {"v3", "v4b", "default"}:
        return Stage0RegModelV3(**common_kwargs)
    if variant in {"v11", "v11_fixed_scale", "fixed_scale"}:
        return Stage0RegModelV11(
            **common_kwargs,
            residual_scale_path=Path(paths_cfg["residual_scale_npz"]),
            residual_scale_floor=float(model_cfg.get("residual_scale_floor", 1.0e-3)),
        )
    if variant in {"v11b", "v11_learnable_scale", "learnable_scale", "v11b_donor_snrna_base"}:
        return Stage0RegModelV11(
            **common_kwargs,
            residual_scale_path=Path(paths_cfg["residual_scale_npz"]),
            residual_scale_floor=float(model_cfg.get("residual_scale_floor", 1.0e-3)),
            learnable_residual_scale=True,
            residual_scale_hidden=int(model_cfg.get("residual_scale_hidden", 128)),
            residual_scale_logscale_clip=float(model_cfg.get("residual_scale_logscale_clip", 1.0)),
            log_mu_base_path=Path(paths_cfg["log_mu_base_npz"]) if paths_cfg.get("log_mu_base_npz") else None,
            donor_log_mu_base_path=(
                Path(paths_cfg["donor_log_mu_base_npz"])
                if paths_cfg.get("donor_log_mu_base_npz")
                else None
            ),
            mu_only=bool(model_cfg.get("mu_only", False)),
        )
    if variant in {"v11b_mu_only", "mu_only"}:
        return Stage0RegModelV11(
            **common_kwargs,
            residual_scale_path=Path(paths_cfg["residual_scale_npz"]),
            residual_scale_floor=float(model_cfg.get("residual_scale_floor", 1.0e-3)),
            learnable_residual_scale=False,
            log_mu_base_path=Path(paths_cfg["log_mu_base_npz"]) if paths_cfg.get("log_mu_base_npz") else None,
            donor_log_mu_base_path=(
                Path(paths_cfg["donor_log_mu_base_npz"])
                if paths_cfg.get("donor_log_mu_base_npz")
                else None
            ),
            mu_only=True,
        )
    if variant in {"v11b_snrna_gene_mlp", "snrna_gene_mlp", "learnable_scale_snrna_gene_mlp"}:
        return Stage0RegModelV11SnRNAGeneMLP(
            **common_kwargs,
            residual_scale_path=Path(paths_cfg["residual_scale_npz"]),
            residual_scale_floor=float(model_cfg.get("residual_scale_floor", 1.0e-3)),
            learnable_residual_scale=True,
            residual_scale_hidden=int(model_cfg.get("residual_scale_hidden", 128)),
            residual_scale_logscale_clip=float(model_cfg.get("residual_scale_logscale_clip", 1.0)),
            log_mu_base_path=Path(paths_cfg["log_mu_base_npz"]) if paths_cfg.get("log_mu_base_npz") else None,
            donor_log_mu_base_path=(
                Path(paths_cfg["donor_log_mu_base_npz"])
                if paths_cfg.get("donor_log_mu_base_npz")
                else None
            ),
            mu_only=bool(model_cfg.get("mu_only", False)),
            snrna_gene_profile_path=Path(paths_cfg["snrna_gene_profile_npz"]),
            snrna_profile_hidden=int(model_cfg.get("snrna_profile_hidden", 256)),
            snrna_profile_dropout=float(model_cfg.get("snrna_profile_dropout", model_cfg.get("gnn_dropout", 0.1))),
        )
    if variant in {"v11b_snrna_gene_gated", "snrna_gene_gated", "learnable_scale_snrna_gene_gated"}:
        return Stage0RegModelV11SnRNAGatedFusion(
            **common_kwargs,
            residual_scale_path=Path(paths_cfg["residual_scale_npz"]),
            residual_scale_floor=float(model_cfg.get("residual_scale_floor", 1.0e-3)),
            learnable_residual_scale=True,
            residual_scale_hidden=int(model_cfg.get("residual_scale_hidden", 128)),
            residual_scale_logscale_clip=float(model_cfg.get("residual_scale_logscale_clip", 1.0)),
            log_mu_base_path=Path(paths_cfg["log_mu_base_npz"]) if paths_cfg.get("log_mu_base_npz") else None,
            donor_log_mu_base_path=(
                Path(paths_cfg["donor_log_mu_base_npz"])
                if paths_cfg.get("donor_log_mu_base_npz")
                else None
            ),
            mu_only=bool(model_cfg.get("mu_only", False)),
            snrna_gene_profile_path=Path(paths_cfg["snrna_gene_profile_npz"]),
            snrna_profile_hidden=int(model_cfg.get("snrna_profile_hidden", 256)),
            snrna_profile_dropout=float(model_cfg.get("snrna_profile_dropout", model_cfg.get("gnn_dropout", 0.1))),
            snrna_gate_init=float(model_cfg.get("snrna_gate_init", 0.02)),
        )
    if variant in {"v11b_snrna_scgpt_gene_mlp", "snrna_scgpt_gene_mlp"}:
        return Stage0RegModelV11SnRNAScGPTGeneMLP(
            **common_kwargs,
            residual_scale_path=Path(paths_cfg["residual_scale_npz"]),
            residual_scale_floor=float(model_cfg.get("residual_scale_floor", 1.0e-3)),
            learnable_residual_scale=True,
            residual_scale_hidden=int(model_cfg.get("residual_scale_hidden", 128)),
            residual_scale_logscale_clip=float(model_cfg.get("residual_scale_logscale_clip", 1.0)),
            log_mu_base_path=Path(paths_cfg["log_mu_base_npz"]) if paths_cfg.get("log_mu_base_npz") else None,
            donor_log_mu_base_path=(
                Path(paths_cfg["donor_log_mu_base_npz"])
                if paths_cfg.get("donor_log_mu_base_npz")
                else None
            ),
            mu_only=bool(model_cfg.get("mu_only", False)),
            snrna_gene_profile_path=Path(paths_cfg["snrna_gene_profile_npz"]),
            snrna_profile_hidden=int(model_cfg.get("snrna_profile_hidden", 512)),
            snrna_profile_dropout=float(model_cfg.get("snrna_profile_dropout", model_cfg.get("gnn_dropout", 0.1))),
            snrna_scgpt_include_expr_features=bool(model_cfg.get("snrna_scgpt_include_expr_features", True)),
        )
    if variant in {"v11b_snrna_scgpt_gene_gated", "snrna_scgpt_gene_gated"}:
        return Stage0RegModelV11SnRNAScGPTGeneGatedFusion(
            **common_kwargs,
            residual_scale_path=Path(paths_cfg["residual_scale_npz"]),
            residual_scale_floor=float(model_cfg.get("residual_scale_floor", 1.0e-3)),
            learnable_residual_scale=True,
            residual_scale_hidden=int(model_cfg.get("residual_scale_hidden", 128)),
            residual_scale_logscale_clip=float(model_cfg.get("residual_scale_logscale_clip", 1.0)),
            log_mu_base_path=Path(paths_cfg["log_mu_base_npz"]) if paths_cfg.get("log_mu_base_npz") else None,
            donor_log_mu_base_path=(
                Path(paths_cfg["donor_log_mu_base_npz"])
                if paths_cfg.get("donor_log_mu_base_npz")
                else None
            ),
            mu_only=bool(model_cfg.get("mu_only", False)),
            snrna_gene_profile_path=Path(paths_cfg["snrna_gene_profile_npz"]),
            snrna_profile_hidden=int(model_cfg.get("snrna_profile_hidden", 512)),
            snrna_profile_dropout=float(model_cfg.get("snrna_profile_dropout", model_cfg.get("gnn_dropout", 0.1))),
            snrna_gate_init=float(model_cfg.get("snrna_gate_init", 0.02)),
            snrna_scgpt_include_expr_features=bool(model_cfg.get("snrna_scgpt_include_expr_features", True)),
        )
    if variant in {"v11b_concat", "concat", "learnable_scale_concat"}:
        return Stage0RegModelV11Concat(
            **common_kwargs,
            residual_scale_path=Path(paths_cfg["residual_scale_npz"]),
            residual_scale_floor=float(model_cfg.get("residual_scale_floor", 1.0e-3)),
            learnable_residual_scale=True,
            residual_scale_hidden=int(model_cfg.get("residual_scale_hidden", 128)),
            residual_scale_logscale_clip=float(model_cfg.get("residual_scale_logscale_clip", 1.0)),
        )
    if variant in {"v11b_crossattn", "crossattn", "cross_attention", "learnable_scale_crossattn"}:
        return Stage0RegModelV11CrossAttention(
            **common_kwargs,
            residual_scale_path=Path(paths_cfg["residual_scale_npz"]),
            residual_scale_floor=float(model_cfg.get("residual_scale_floor", 1.0e-3)),
            learnable_residual_scale=True,
            residual_scale_hidden=int(model_cfg.get("residual_scale_hidden", 128)),
            residual_scale_logscale_clip=float(model_cfg.get("residual_scale_logscale_clip", 1.0)),
            log_mu_base_path=Path(paths_cfg["log_mu_base_npz"]) if paths_cfg.get("log_mu_base_npz") else None,
            donor_log_mu_base_path=(
                Path(paths_cfg["donor_log_mu_base_npz"])
                if paths_cfg.get("donor_log_mu_base_npz")
                else None
            ),
            mu_only=bool(model_cfg.get("mu_only", False)),
            cross_attn_hidden=int(model_cfg.get("cross_attn_hidden", model_cfg["film_hidden"])),
            cross_attn_heads=int(model_cfg.get("cross_attn_heads", 4)),
            cross_attn_spot_tokens=int(model_cfg.get("cross_attn_spot_tokens", 4)),
            cross_attn_dropout=(
                float(model_cfg["cross_attn_dropout"])
                if model_cfg.get("cross_attn_dropout") is not None
                else None
            ),
        )
    if variant in {"v7", "v7_calibrated", "calibrated"}:
        return Stage0RegModelV7(
            **common_kwargs,
            calibration_hidden=int(model_cfg.get("calibration_hidden", 128)),
            calibration_logscale_clip=float(model_cfg.get("calibration_logscale_clip", 2.0)),
        )
    raise ValueError(f"Unsupported stage0 regression model variant: {variant}")
