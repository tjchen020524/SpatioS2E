#!/usr/bin/env python3
"""Factory for public SpatioS2E model variants."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

from spatios2e.models.morphology_model import MorphologyGraphModel
from spatios2e.models.residual_model import GeneConditionedResidualModel
from spatios2e.models.single_cell_prior import GatedScGPTGeneContextPriorModel
from spatios2e.models.spatios2e_model import SpatioS2EModel


def _optional_path(paths_cfg: Dict, key: str) -> Path | None:
    value = paths_cfg.get(key)
    return Path(value) if value else None


def _common_kwargs(model_cfg: Dict, paths_cfg: Dict, input_dim: int) -> Dict:
    return dict(
        input_dim=input_dim,
        gnn_hidden=int(model_cfg["gnn_hidden"]),
        gnn_layers=int(model_cfg["gnn_layers"]),
        gnn_dropout=float(model_cfg["gnn_dropout"]),
        film_hidden=int(model_cfg["film_hidden"]),
        spot_head_hidden=int(model_cfg["spot_head_hidden"]),
        decima_ckpt=Path(paths_cfg["decima_ckpt"]),
        decima_h5=_optional_path(paths_cfg, "decima_h5"),
        decima_npz_dir=_optional_path(paths_cfg, "decima_npz_dir"),
        freeze_decima_backbone=bool(model_cfg.get("freeze_decima_backbone", True)),
        freeze_pseudobulk_head=bool(model_cfg.get("freeze_pseudobulk_head", False)),
        gnn_post_mlp_layers=int(model_cfg.get("gnn_post_mlp_layers", 1)),
        gnn_residual=bool(model_cfg.get("gnn_residual", True)),
        delta_head_init_std=float(model_cfg.get("delta_head_init_std", 0.0)),
    )


def _scale_kwargs(model_cfg: Dict, paths_cfg: Dict) -> Dict:
    return dict(
        residual_scale_path=Path(paths_cfg["residual_scale_npz"]),
        residual_scale_floor=float(model_cfg.get("residual_scale_floor", 1.0e-3)),
        residual_scale_hidden=int(model_cfg.get("residual_scale_hidden", 128)),
        residual_scale_logscale_clip=float(model_cfg.get("residual_scale_logscale_clip", 1.0)),
    )


def build_model(cfg: Dict, input_dim: int):
    """Build a SpatioS2E model from a YAML/JSON configuration dictionary.

    Public release variants:
    - ``spatios2e``: H&E morphology + spatial graph + DNA sequence prior.
    - ``morphology_graph``: H&E morphology + spatial graph without molecular prior.
    - ``sequence_prior``: DNA prior baseline without a learned spatial residual.
    - ``single_cell_prior``: DNA conditioning with additive gated donor-matched scGPT gene context.
    """

    model_cfg = cfg["model"]
    paths_cfg = cfg["paths"]
    variant = str(model_cfg.get("variant", "spatios2e")).lower()

    if variant in {"morphology_graph", "morphology", "histology_graph"}:
        return MorphologyGraphModel(
            input_dim=input_dim,
            gnn_hidden=int(model_cfg["gnn_hidden"]),
            gnn_layers=int(model_cfg["gnn_layers"]),
            gnn_dropout=float(model_cfg["gnn_dropout"]),
            spot_head_hidden=int(model_cfg["spot_head_hidden"]),
            gene_ids_path=Path(paths_cfg["gene_split_dir"]) / "train_genes.txt",
            gene_bias_init_path=_optional_path(paths_cfg, "gene_bias_init_npz"),
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

    common_kwargs = _common_kwargs(model_cfg, paths_cfg, input_dim)

    if variant in {"gene_conditioned_residual", "dna_residual"}:
        return GeneConditionedResidualModel(**common_kwargs)

    if variant in {"spatios2e", "dna_sequence_prior", "residual_decomposition"}:
        return SpatioS2EModel(
            **common_kwargs,
            residual_scale_path=Path(paths_cfg["residual_scale_npz"]),
            residual_scale_floor=float(model_cfg.get("residual_scale_floor", 1.0e-3)),
            learnable_residual_scale=True,
            residual_scale_hidden=int(model_cfg.get("residual_scale_hidden", 128)),
            residual_scale_logscale_clip=float(model_cfg.get("residual_scale_logscale_clip", 1.0)),
            log_mu_base_path=_optional_path(paths_cfg, "log_mu_base_npz"),
            donor_log_mu_base_path=_optional_path(paths_cfg, "donor_log_mu_base_npz"),
            baseline_only=bool(model_cfg.get("baseline_only", False)),
        )

    if variant in {"sequence_prior", "dna_prior", "baseline_only"}:
        return SpatioS2EModel(
            **common_kwargs,
            residual_scale_path=Path(paths_cfg["residual_scale_npz"]),
            residual_scale_floor=float(model_cfg.get("residual_scale_floor", 1.0e-3)),
            learnable_residual_scale=False,
            log_mu_base_path=_optional_path(paths_cfg, "log_mu_base_npz"),
            donor_log_mu_base_path=_optional_path(paths_cfg, "donor_log_mu_base_npz"),
            baseline_only=True,
        )

    if variant in {
        "single_cell_prior",
        "gated_scgpt_gene_context_prior",
        "scgpt_gene_context_gated",
        "snrna_scgpt_gene_gated",
        "v11b_snrna_scgpt_gene_gated",
    }:
        return GatedScGPTGeneContextPriorModel(
            **common_kwargs,
            **_scale_kwargs(model_cfg, paths_cfg),
            learnable_residual_scale=True,
            log_mu_base_path=_optional_path(paths_cfg, "log_mu_base_npz"),
            donor_log_mu_base_path=_optional_path(paths_cfg, "donor_log_mu_base_npz"),
            snrna_gene_profile_path=Path(paths_cfg["snrna_gene_profile_npz"]),
            snrna_profile_hidden=int(model_cfg.get("snrna_profile_hidden", 512)),
            snrna_profile_dropout=float(model_cfg.get("snrna_profile_dropout", model_cfg.get("gnn_dropout", 0.1))),
            snrna_gate_init=float(model_cfg.get("snrna_gate_init", 0.02)),
            snrna_scgpt_include_expr_features=bool(model_cfg.get("snrna_scgpt_include_expr_features", True)),
        )

    raise ValueError(f"Unsupported SpatioS2E model variant: {variant}")
