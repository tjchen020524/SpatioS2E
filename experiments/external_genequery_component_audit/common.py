#!/usr/bin/env python3
"""Shared data and provenance utilities for the external GeneQuery audit."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy import sparse


ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments/external_genequery_component_audit"
HER2 = Path(os.environ.get("SPATIOS2E_HER2_ROOT", ROOT / "data/her2st"))
CLEAN_SPLIT = ROOT / "configs/manuscript/gene_splits/her2st"
PANEL_PATH = Path(os.environ.get("SPATIOS2E_GENEQUERY_PANEL", ROOT / "data/her_hvg_cut_1000.npy"))
GENE_META_PATH = Path(os.environ.get("SPATIOS2E_GENE_META", ROOT / "data/gene_meta.tsv"))
PANEL_ARTIFACT = EXP / "artifacts/panel/genequery_her2_panel.npz"
FEATURE_ROOT = EXP / "artifacts/resnet50_features"
PATCH_ROOT = EXP / "artifacts/resnet50_patches"
RUN_ROOT = EXP / "runs"
TRAINABLE_RUN_ROOT = EXP / "trainable_backbone_runs"

OFFICIAL_COMMIT = "7d61d38497316cce384fc11cdf7158cea752d413"
OFFICIAL_EMBEDDING_SHA256 = (
    "61f482f4a5e4377f1085d6a95d28ff59cf721a9d4311e1abae9fc7813efbd83d"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_split() -> Dict[str, List[str]]:
    return json.loads((HER2 / "data/split.json").read_text())


def load_manifest() -> pd.DataFrame:
    frame = pd.read_csv(HER2 / "data/sample_manifest.tsv", sep="\t")
    required = {"sample", "patient", "split", "n_spots"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError("HER2 manifest is missing columns: %s" % sorted(missing))
    return frame


def load_panel_artifact(path: Path = PANEL_ARTIFACT) -> Dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(
            "%s is missing; run prepare_panel.py with the official GeneQuery embedding" % path
        )
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def load_expression(sample: str, gene_ids: Iterable[str]) -> Tuple[np.ndarray, np.ndarray]:
    manifest = load_manifest().set_index("sample")
    split = str(manifest.loc[sample, "split"])
    path = HER2 / "data/expression" / sample / (split + ".npz")
    with np.load(path, allow_pickle=False) as data:
        matrix = sparse.csr_matrix(
            (data["data"], data["indices"], data["indptr"]),
            shape=tuple(data["shape"].tolist()),
        )
        available = data["gene_ids"].astype(str)
        barcodes = data["barcodes"].astype(str)
    lookup = {gene_id: idx for idx, gene_id in enumerate(available.tolist())}
    requested = list(gene_ids)
    missing = [gene_id for gene_id in requested if gene_id not in lookup]
    if missing:
        raise ValueError("%s is missing %d requested genes" % (sample, len(missing)))
    indices = np.asarray([lookup[gene_id] for gene_id in requested], dtype=np.int64)
    expression = matrix[indices].toarray().T.astype(np.float32, copy=False)
    return expression, barcodes


def load_feature_sample(sample: str, root: Path = FEATURE_ROOT) -> Tuple[np.ndarray, np.ndarray]:
    path = root / sample / "features.npz"
    if not path.exists():
        raise FileNotFoundError("Missing ResNet feature file: %s" % path)
    with np.load(path, allow_pickle=False) as data:
        features = data["features"].astype(np.float32, copy=False)
        barcodes = data["barcodes"].astype(str)
    return features, barcodes


def load_partition(
    partition: str,
    gene_ids: Iterable[str],
    feature_root: Path = FEATURE_ROOT,
    max_spots: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return features, expression, sample IDs, and barcodes for one patient split."""

    manifest = load_manifest()
    samples = manifest.loc[manifest["split"] == partition, "sample"].astype(str).tolist()
    if not samples:
        raise ValueError("No samples for partition %s" % partition)
    feature_blocks = []
    expression_blocks = []
    sample_blocks = []
    barcode_blocks = []
    for sample in samples:
        features, feature_barcodes = load_feature_sample(sample, feature_root)
        expression, expression_barcodes = load_expression(sample, gene_ids)
        if not np.array_equal(feature_barcodes, expression_barcodes):
            raise ValueError("Barcode order differs between features and expression for %s" % sample)
        feature_blocks.append(features)
        expression_blocks.append(expression)
        sample_blocks.append(np.repeat(sample, len(feature_barcodes)))
        barcode_blocks.append(feature_barcodes)
    x = np.concatenate(feature_blocks)
    y = np.concatenate(expression_blocks)
    sample_ids = np.concatenate(sample_blocks)
    barcodes = np.concatenate(barcode_blocks)
    if max_spots > 0:
        x = x[:max_spots]
        y = y[:max_spots]
        sample_ids = sample_ids[:max_spots]
        barcodes = barcodes[:max_spots]
    return x, y, sample_ids, barcodes


def load_expression_partition(
    partition: str,
    gene_ids: Iterable[str],
    max_spots: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return expression, sample IDs, and barcodes without requiring image features."""

    manifest = load_manifest()
    samples = manifest.loc[manifest["split"] == partition, "sample"].astype(str).tolist()
    if not samples:
        raise ValueError("No samples for partition %s" % partition)
    expression_blocks = []
    sample_blocks = []
    barcode_blocks = []
    for sample in samples:
        expression, barcodes = load_expression(sample, gene_ids)
        expression_blocks.append(expression)
        sample_blocks.append(np.repeat(sample, len(barcodes)))
        barcode_blocks.append(barcodes)
    y = np.concatenate(expression_blocks)
    sample_ids = np.concatenate(sample_blocks)
    barcodes = np.concatenate(barcode_blocks)
    if max_spots > 0:
        y = y[:max_spots]
        sample_ids = sample_ids[:max_spots]
        barcodes = barcodes[:max_spots]
    return y, sample_ids, barcodes


def counterfactual_embeddings(
    embeddings: np.ndarray,
    split_labels: np.ndarray,
    variant: str,
    seed: int,
) -> np.ndarray:
    """Construct matched controls without changing the train/held-out membership."""

    source = np.asarray(embeddings, dtype=np.float32)
    labels = np.asarray(split_labels).astype(str)
    rng = np.random.default_rng(seed + 1907)
    if variant == "semantic":
        return source.copy()
    if variant == "identity_shuffle":
        output = source.copy()
        for label in ("train", "heldout"):
            indices = np.flatnonzero(labels == label)
            output[indices] = source[rng.permutation(indices)]
        return output
    if variant == "random":
        train = source[labels == "train"].astype(np.float64)
        mean = train.mean(axis=0)
        std = train.std(axis=0)
        std = np.where(std > 1.0e-8, std, 1.0)
        return rng.normal(mean, std, size=source.shape).astype(np.float32)
    if variant == "constant":
        mean = source[labels == "train"].mean(axis=0, keepdims=True)
        return np.repeat(mean, source.shape[0], axis=0).astype(np.float32)
    raise ValueError("Unknown counterfactual variant: %s" % variant)
