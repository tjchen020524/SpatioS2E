#!/usr/bin/env python3
from __future__ import annotations
from workflow_paths import DATA_ROOT

import argparse
import sys
from pathlib import Path

import numpy as np

from experiments.geneheldout_uni2h_decima_decoder_noseq_controls import (
    run_geneheldout_noseq_controls as base,
)


ROOT = DATA_ROOT
EXP_ROOT = ROOT / "experiments/multicohort_geneheldout_decima"
DECIMA_EMB_CACHE = (
    ROOT
    / "experiments/sample_split_fullgenes_v12_uni2h_direct_seq_residual_no_celltype/artifacts/decima_gene_embeddings_fullgenes.npz"
)
COHORT_ROOTS = {
    "hippocampus": ROOT / "experiments/hippocampus_current_full_ablation",
    "hippocampus_donor_disjoint": ROOT / "experiments/hippocampus_current_full_ablation",
    "dlpfc": ROOT / "experiments/dlpfc_current_full_ablation",
    "nac": ROOT / "experiments/nac_current_full_ablation",
    "her2st": ROOT / "experiments/her2st_current_full_ablation",
}


def configure_base(cohort: str, seed: int, variant: str) -> None:
    cohort_root = COHORT_ROOTS[cohort]
    base.EXP_ROOT = EXP_ROOT / "runs" / cohort / f"seed_{seed}" / variant
    base.DECIMA_EMB_CACHE = DECIMA_EMB_CACHE
    if cohort == "hippocampus":
        base.DATA_ROOT = ROOT / "experiments/benchmark_uni2h_hippocampus_fullgenes/data"
        base.EXPRESSION_ROOT = ROOT / "experiments/sample_split_fullgenes/expression_full"
        base.EMBEDDING_ROOT = ROOT / "data/processed/histology_embeddings_uni2h"
        base.GENE_SPLIT_DIR = ROOT / "experiments/stpath_seq_mechanism_controls/artifacts"
        return

    metadata_root = EXP_ROOT / "data" / cohort
    multimodal_root = cohort_root / "data/multimodal"
    base.DATA_ROOT = metadata_root
    base.EXPRESSION_ROOT = cohort_root / "data/expression"
    base.EMBEDDING_ROOT = multimodal_root
    base.GENE_SPLIT_DIR = metadata_root / "gene_splits"

    class MultimodalFeatureStore:
        def __init__(self) -> None:
            self._cache: dict[str, tuple[np.ndarray, dict[str, int]]] = {}

        def _load(self, sample: str) -> tuple[np.ndarray, dict[str, int]]:
            cached = self._cache.get(sample)
            if cached is not None:
                return cached
            arr = np.load(multimodal_root / sample / "multimodal_features.npz", allow_pickle=True)
            embeddings = arr["histology_embeddings"].astype(np.float32, copy=False)
            barcodes = arr["barcodes"].astype(str).tolist()
            cached = (embeddings, {barcode: idx for idx, barcode in enumerate(barcodes)})
            self._cache[sample] = cached
            return cached

        def get_row(self, sample: str, barcode: str) -> np.ndarray:
            embeddings, barcode_to_idx = self._load(sample)
            return embeddings[barcode_to_idx[barcode]]

    def load_cohort_decima_embeddings(gene_ids: list[str], train_idx: np.ndarray) -> np.ndarray:
        arr = np.load(DECIMA_EMB_CACHE, allow_pickle=False)
        cached_gene_ids = arr["gene_ids"].astype(str).tolist()
        cache_index = {gene: idx for idx, gene in enumerate(cached_gene_ids)}
        missing = [gene for gene in gene_ids if gene not in cache_index]
        if missing:
            raise ValueError(f"Missing {len(missing)} {cohort} genes from Decima embedding cache")
        source = arr["embeddings"].astype(np.float32, copy=False)
        emb = source[np.asarray([cache_index[gene] for gene in gene_ids], dtype=np.int64)]
        mean = emb[train_idx].mean(axis=0, keepdims=True)
        std = emb[train_idx].std(axis=0, keepdims=True)
        return ((emb - mean) / np.maximum(std, 1.0e-6)).astype(np.float32, copy=False)

    base.FeatureStore = MultimodalFeatureStore
    base.load_decima_embeddings = load_cohort_decima_embeddings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", required=True, choices=sorted(COHORT_ROOTS))
    parser.add_argument("--seed", required=True, type=int, choices=[42, 123, 456])
    parser.add_argument("--variant", required=True, choices=["decima", "random", "constant"])
    args = parser.parse_args()
    configure_base(args.cohort, args.seed, args.variant)

    forwarded = [
        "--epochs", "6",
        "--batch-size", "256",
        "--eval-batch-size", "256",
        "--workers", "0",
        "--hidden-dim", "512",
        "--program-dim", "96",
        "--train-genes-per-batch", "512",
        "--val-train-genes", "2048",
        "--report-train-genes", "2048",
        "--eval-gene-chunk-size", "512",
        "--lambda-gene-corr", "0.15",
        "--lr", "2.0e-4",
        "--weight-decay", "1.0e-4",
        "--seed", str(args.seed),
        "--variants", args.variant,
        "--paired-rng",
    ]
    if args.variant != "decima":
        forwarded.append("--no-train-mean-baseline")
    sys.argv = [sys.argv[0], *forwarded]
    base.main()


if __name__ == "__main__":
    main()
