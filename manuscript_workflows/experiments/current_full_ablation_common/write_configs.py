#!/usr/bin/env python3
"""Generate a matched current-model ablation matrix for one prepared cohort."""

from __future__ import annotations
from workflow_paths import DATA_ROOT

import argparse
import json
from pathlib import Path

import yaml


ROOT = DATA_ROOT


def resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


VARIANTS = {
    "uni2h": {"variant": "image_graph_only", "no_graph": True},
    "uni2h_gnn": {"variant": "image_graph_only", "no_graph": False},
    "decima_prior_only": {"variant": "v11b_mu_only", "no_graph": True},
    "decima_film_no_graph": {"variant": "v11b", "no_graph": True},
    "gene_mean_uni2h_residual": {
        "variant": "image_graph_only",
        "no_graph": True,
        "gene_mean_init": True,
    },
    "gene_mean_uni2h_gnn": {
        "variant": "image_graph_only",
        "no_graph": False,
        "gene_mean_init": True,
    },
    "decima_film_gnn": {"variant": "v11b", "no_graph": False},
    "decima_concat_gnn": {"variant": "v11b_concat", "no_graph": False},
    "decima_crossattn_gnn": {"variant": "v11b_crossattn", "no_graph": False},
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-dir", type=Path, required=True)
    args = parser.parse_args()

    cohort_dir = resolve(args.cohort_dir)
    paths = json.loads((cohort_dir / "data/paths.json").read_text())
    split = json.loads((cohort_dir / "data/split.json").read_text())
    config_dir = cohort_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    artifacts = cohort_dir / "artifacts"

    generated = []
    for name, spec in VARIANTS.items():
        variant_dir = cohort_dir / "variants" / name
        model = {
            "variant": spec["variant"],
            "no_graph": bool(spec["no_graph"]),
            "gnn_hidden": 256,
            "gnn_layers": 3,
            "gnn_dropout": 0.1,
            "gnn_post_mlp_layers": 1,
            "gnn_residual": True,
            "film_hidden": 256,
            "spot_head_hidden": 128,
            "freeze_decima_backbone": True,
            "freeze_pseudobulk_head": False,
            "delta_head_init_std": 0.0,
            "residual_scale_floor": float(paths.get("residual_scale_floor", 1.0e-3)),
            "residual_scale_hidden": 128,
            "residual_scale_logscale_clip": 0.75,
        }
        config_paths = {
            "expression_root": paths["expression_root"],
            "multimodal_root": paths["multimodal_root"],
            "graph_root": paths["graph_root"],
            "modality_stats": paths["modality_stats"],
            "gene_split_dir": paths["gene_split_dir"],
            "decima_ckpt": paths["decima_ckpt"],
            "decima_h5": paths.get("decima_h5"),
            "decima_npz_dir": paths["decima_npz_dir"],
            "residual_scale_npz": str(artifacts.relative_to(ROOT) / "train_decima_prior.npz"),
            "checkpoint_dir": str(variant_dir.relative_to(ROOT) / "checkpoints"),
            "checkpoint_name": "best.pt",
        }
        if spec.get("gene_mean_init"):
            config_paths["gene_bias_init_npz"] = str(
                artifacts.relative_to(ROOT) / "train_gene_mean_prior.npz"
            )
            model["gene_bias_init_key"] = "log_mu_base"
            model["output_weight_init_std"] = 0.0

        cfg = {
            "run_name": f"{paths['cohort_name']}_{name}",
            "paths": config_paths,
            "split": split,
            "targets_dense": True,
            "normalize_modalities": True,
            "model": model,
            "loss": {
                "point_loss": "huber",
                "huber_beta": 1.0,
                "lambda_point": 1.0,
                "lambda_gene": 0.6,
                "lambda_spot": 0.05,
                "lambda_delta_point": 0.0,
                "lambda_delta_gene": 0.0,
                "delta_target_mode": (
                    "centered_true"
                    if spec["variant"] == "image_graph_only"
                    else "baseline_relative"
                ),
                "lambda_lap": 6.0e-4,
                "lambda_calib_identity": 5.0e-4 if "decima" in name else 0.0,
                "min_true_std": 1.0e-6,
            },
            "optim": {
                "lr": float(paths.get("lr", 2.5e-4)),
                "weight_decay": 1.0e-4,
                "max_epochs": int(paths.get("max_epochs", 10)),
                "min_epochs": int(paths.get("min_epochs", 1)),
                "max_genes_per_batch": int(paths.get("max_genes_per_batch", 96)),
                "val_max_genes_per_batch": int(paths.get("val_max_genes_per_batch", 128)),
                "train_gene_chunks_per_sample": int(
                    paths.get("train_gene_chunks_per_sample", 1)
                ),
                "require_full_gene_coverage_per_epoch": bool(
                    paths.get("require_full_gene_coverage_per_epoch", False)
                ),
                "log_every": 1,
                "early_stop_patience": 4,
                "selection_metric": "gene_flat_combo",
                "selection_flat_weight": 0.25,
            },
            "eval": {
                "max_genes_per_batch": int(paths.get("eval_max_genes_per_batch", 128)),
                "true_std_threshold": 1.0e-6,
                "pred_std_threshold": 1.0e-6,
                "hvg_ranking_file": str(artifacts.relative_to(ROOT) / "train_gene_stats.tsv"),
            },
            "seed": int(paths.get("seed", 42)),
        }
        config_path = config_dir / f"{name}.yaml"
        config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        generated.append(
            {
                "name": name,
                "config": str(config_path),
                "no_graph": spec["no_graph"],
                "spatial_pcc_applicable": name != "decima_prior_only",
            }
        )

    (config_dir / "variant_manifest.json").write_text(json.dumps(generated, indent=2))
    print(json.dumps({"cohort_dir": str(cohort_dir), "variants": generated}, indent=2))


if __name__ == "__main__":
    main()
