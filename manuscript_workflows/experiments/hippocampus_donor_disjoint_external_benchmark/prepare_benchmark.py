#!/usr/bin/env python3
"""Prepare strict-split inputs and per-seed configs for external benchmarks."""

from __future__ import annotations
from workflow_paths import DATA_ROOT

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = DATA_ROOT
EXP = DATA_ROOT / 'experiments/hippocampus_donor_disjoint_external_benchmark'
STRICT = ROOT / "experiments/hippocampus_current_full_ablation"
OLD_STNET = ROOT / "experiments/benchmark_stnet_hippocampus_fullgenes"
OLD_BLEEP = ROOT / "experiments/benchmark_bleep_hippocampus_fullgenes"
SEEDS = (42, 123, 456)


def ensure_symlink(link: Path, target: Path) -> None:
    target = target.resolve()
    if not target.exists():
        raise FileNotFoundError(target)
    if link.is_symlink():
        if link.resolve() != target:
            raise RuntimeError(f"Refusing to replace mismatched symlink: {link}")
        return
    if link.exists():
        raise RuntimeError(f"Refusing to replace existing path: {link}")
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target, target_is_directory=target.is_dir())


def donor(sample: str) -> str:
    return sample.split("_", 1)[0]


def load_split() -> dict[str, list[str]]:
    split = json.loads((STRICT / "data/split.json").read_text())
    if set(split) != {"train", "val", "test"}:
        raise ValueError(f"Unexpected split keys: {sorted(split)}")
    sample_sets = {key: set(value) for key, value in split.items()}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = sample_sets[left] & sample_sets[right]
        if overlap:
            raise ValueError(f"Sample overlap between {left} and {right}: {sorted(overlap)}")
        donor_overlap = {donor(x) for x in split[left]} & {donor(x) for x in split[right]}
        if donor_overlap:
            raise ValueError(f"Donor overlap between {left} and {right}: {sorted(donor_overlap)}")
    return split


def load_gene_ids(split: dict[str, list[str]]) -> list[str]:
    sample = split["train"][0]
    path = STRICT / "data/expression" / sample / "train.npz"
    with np.load(path, allow_pickle=True) as arr:
        genes = arr["gene_ids"].astype(str).tolist()
    if len(genes) != 18397 or len(set(genes)) != len(genes):
        raise ValueError(f"Unexpected target gene universe: {len(genes)} genes")
    return genes


def prepare_stnet(split: dict[str, list[str]]) -> None:
    mapping = pd.read_csv(OLD_STNET / "meta/patient_mapping.tsv", sep="\t")
    expected = {sample for values in split.values() for sample in values}
    if set(mapping["sample"]) != expected:
        missing = sorted(expected - set(mapping["sample"]))
        extra = sorted(set(mapping["sample"]) - expected)
        raise ValueError(f"ST-Net mapping mismatch; missing={missing}, extra={extra}")

    sample_to_split = {sample: key for key, values in split.items() for sample in values}
    mapping["split"] = mapping["sample"].map(sample_to_split)
    meta = EXP / "data/stnet/meta"
    meta.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(meta / "patient_mapping.tsv", sep="\t", index=False)
    payload = {
        f"{key}_patients": mapping.loc[mapping["split"] == key, "patient"].astype(str).tolist()
        for key in ("train", "val", "test")
    }
    payload["source_split"] = str(STRICT / "data/split.json")
    (meta / "split.json").write_text(json.dumps(payload, indent=2))

    for seed in SEEDS:
        run = EXP / f"runs/stnet/seed_{seed}"
        run.mkdir(parents=True, exist_ok=True)
        ensure_symlink(run / "raw", OLD_STNET / "raw")
        ensure_symlink(run / "processed", OLD_STNET / "processed")
        ensure_symlink(run / "meta", meta)


def prepare_bleep(split: dict[str, list[str]], genes: list[str]) -> None:
    manifest = pd.read_csv(OLD_BLEEP / "data/manifest.tsv", sep="\t")
    expected = {sample for values in split.values() for sample in values}
    if set(manifest["sample"].astype(str)) != expected:
        raise ValueError("BLEEP manifest does not cover the strict 34-section cohort")
    sample_to_split = {sample: key for key, values in split.items() for sample in values}
    manifest["split"] = manifest["sample"].map(sample_to_split)

    data_root = EXP / "data/bleep"
    data_root.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(data_root / "manifest.tsv", sep="\t", index=False)
    (data_root / "gene_ids.txt").write_text("\n".join(genes) + "\n")

    expression_proxy = data_root / "expression_proxy"
    for split_name, samples in split.items():
        for sample in samples:
            ensure_symlink(
                expression_proxy / sample / "test.npz",
                STRICT / "data/expression" / sample / f"{split_name}.npz",
            )


def hist2st_config(split: dict[str, list[str]], seed: int, *, evaluate: bool) -> dict:
    run = EXP / f"runs/hist2st/seed_{seed}"
    return {
        "run_name": f"hist2st_donor_disjoint_seed_{seed}",
        "paths": {
            "raw_root": str(ROOT / "data/raw"),
            "expression_root": str(STRICT / "data/expression"),
            "gene_list": str(EXP / "data/all_genes.txt"),
            "checkpoint_dir": str(run / "checkpoints"),
            "checkpoint_name": "hist2st_best.pt",
        },
        "split": split,
        "data": {
            "patch_size": 112,
            "neighbor_k": 6,
            "prune": "NA",
            "max_spots": None if evaluate else 1024,
            "num_workers": 0,
        },
        "model": {
            "n_pos": 64,
            "kernel_size": 5,
            "patch_embed": 7,
            "depth1": 2,
            "depth2": 8,
            "depth3": 4,
            "heads": 16,
            "channel": 32,
            "dropout": 0.2,
            "zinb": 0.0,
            "nb": False,
            "bake": 0,
            "lamb": 0.0,
            "policy": "mean",
        },
        "optim": {"lr": 1.0e-4, "weight_decay": 1.0e-4, "max_epochs": 10},
        "seed": seed,
    }


def stpath_config(split: dict[str, list[str]], seed: int) -> dict:
    run = EXP / f"runs/stpath/seed_{seed}"
    return {
        "run_name": f"stpath_donor_disjoint_seed_{seed}",
        "paths": {
            "expression_root": str(STRICT / "data/expression"),
            "hist_root": str(ROOT / "data/processed/histology_embeddings"),
            "spatial_root": str(ROOT / "data/processed/spatial_features"),
            "stpath_root": str(ROOT / "STPath"),
            "stpath_weight": str(ROOT / "STPath/stfm.pth"),
            "gene_vocab": str(ROOT / "STPath/utils_data/symbol2ensembl.json"),
            "adapter_path": str(EXP / "artifacts/pca_train_donors_2048_to_1536.npz"),
            "checkpoint_dir": str(run / "checkpoints"),
            "checkpoint_name": "best.pt",
            "embed_out_root": str(run / "embeddings"),
            "pred_out_root": str(run / "predictions"),
        },
        "split": split,
        "data": {"fit_on": "train"},
        "model": {
            "tech_type": "Visium",
            "organ_type": "Brain",
            "train_gene_head": True,
            "train_image_embed": True,
            "train_gene_embed": False,
            "unfreeze_last_n_blocks": 1,
        },
        "optim": {
            "lr": 1.0e-4,
            "weight_decay": 1.0e-4,
            "max_epochs": 8,
            "max_genes_per_step": 512,
            "max_spots_per_step": 2048,
            "val_max_genes_per_step": 1024,
            "val_max_spots_per_step": 2048,
        },
        "export": {"spot_chunk_size": 2048, "gene_chunk_size": 512},
        "seed": seed,
    }


def prepare_configs(split: dict[str, list[str]], genes: list[str]) -> None:
    data = EXP / "data"
    data.mkdir(parents=True, exist_ok=True)
    (data / "all_genes.txt").write_text("\n".join(genes) + "\n")
    configs = EXP / "configs"
    configs.mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        for method in ("hist2st", "stpath", "bleep"):
            (EXP / f"runs/{method}/seed_{seed}").mkdir(parents=True, exist_ok=True)
        (configs / f"hist2st_seed_{seed}.json").write_text(
            json.dumps(hist2st_config(split, seed, evaluate=False), indent=2)
        )
        (configs / f"hist2st_eval_seed_{seed}.json").write_text(
            json.dumps(hist2st_config(split, seed, evaluate=True), indent=2)
        )
        (configs / f"stpath_seed_{seed}.json").write_text(json.dumps(stpath_config(split, seed), indent=2))


def fit_stpath_adapter(split: dict[str, list[str]]) -> None:
    from experiments.stpath_finetune_spatios2e.common import fit_or_load_adapter, load_aligned_sample

    hist_root = ROOT / "data/processed/histology_embeddings"
    spatial_root = ROOT / "data/processed/spatial_features"
    adapter_path = EXP / "artifacts/pca_train_donors_2048_to_1536.npz"
    first_x, _, _ = load_aligned_sample(hist_root, spatial_root, split["train"][0])
    fit_or_load_adapter(
        adapter_path=adapter_path,
        fit_samples=split["train"],
        hist_root=hist_root,
        spatial_root=spatial_root,
        in_dim=int(first_x.shape[1]),
        out_dim=1536,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-stpath-adapter", action="store_true")
    args = parser.parse_args()

    split = load_split()
    genes = load_gene_ids(split)
    prepare_stnet(split)
    prepare_bleep(split, genes)
    prepare_configs(split, genes)
    if args.fit_stpath_adapter:
        fit_stpath_adapter(split)

    manifest = pd.read_csv(EXP / "data/bleep/manifest.tsv", sep="\t")
    report = {
        "split_source": str(STRICT / "data/split.json"),
        "samples": {key: len(value) for key, value in split.items()},
        "donors": {key: sorted({donor(x) for x in value}) for key, value in split.items()},
        "spots": {key: int((manifest["split"] == key).sum()) for key in split},
        "n_genes": len(genes),
        "seeds": list(SEEDS),
        "stpath_adapter_ready": (EXP / "artifacts/pca_train_donors_2048_to_1536.npz").exists(),
    }
    (EXP / "data/preparation_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

