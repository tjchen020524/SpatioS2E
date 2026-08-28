"""Utilities for matched pretrained, random and constant gene-vector assays."""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np


def standardize_from_training_genes(
    vectors: np.ndarray,
    training_indices: Sequence[int],
    minimum_scale: float = 1.0e-6,
) -> np.ndarray:
    """Standardize every vector feature using training genes only."""

    values = np.asarray(vectors, dtype=np.float32)
    indices = np.asarray(training_indices, dtype=np.int64)
    if values.ndim != 2:
        raise ValueError("vectors must be a two-dimensional [gene, feature] array")
    if indices.ndim != 1 or indices.size == 0:
        raise ValueError("training_indices must be a non-empty one-dimensional sequence")
    if indices.min() < 0 or indices.max() >= values.shape[0]:
        raise IndexError("training_indices contain an out-of-range gene index")
    training = values[indices]
    mean = training.mean(axis=0, keepdims=True)
    scale = np.maximum(training.std(axis=0, keepdims=True), float(minimum_scale))
    return ((values - mean) / scale).astype(np.float32, copy=False)


def random_gene_vectors(shape: Tuple[int, int], seed: int) -> np.ndarray:
    """Draw a fixed dimension-matched standard-normal control matrix."""

    if len(shape) != 2 or shape[0] < 1 or shape[1] < 1:
        raise ValueError("shape must contain positive gene and feature dimensions")
    rng = np.random.default_rng(seed)
    return rng.standard_normal(size=shape).astype(np.float32)


def constant_gene_vectors(shape: Tuple[int, int], value: float = 0.0) -> np.ndarray:
    """Return the same vector for every gene."""

    if len(shape) != 2 or shape[0] < 1 or shape[1] < 1:
        raise ValueError("shape must contain positive gene and feature dimensions")
    return np.full(shape, value, dtype=np.float32)


def permute_gene_identity(vectors: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Permute gene-to-vector correspondence while preserving the vector set."""

    values = np.asarray(vectors)
    if values.ndim != 2:
        raise ValueError("vectors must be a two-dimensional [gene, feature] array")
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(values.shape[0])
    return values[permutation].copy(), permutation


def permute_gene_identity_within_partitions(
    vectors: np.ndarray,
    partitions: Sequence[Sequence[int]],
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Break gene identity while preserving each partition's vector multiset.

    The manuscript permutes downstream training and held-out targets
    independently so no held-out vector is reassigned to a training gene (or
    conversely). Indices not listed in a partition retain their original row.
    """

    values = np.asarray(vectors)
    if values.ndim != 2:
        raise ValueError("vectors must be a two-dimensional [gene, feature] array")
    permutation = np.arange(values.shape[0], dtype=np.int64)
    assigned = np.zeros(values.shape[0], dtype=bool)
    rng = np.random.default_rng(seed)
    for raw_partition in partitions:
        partition = np.asarray(raw_partition, dtype=np.int64)
        if partition.ndim != 1 or not len(partition):
            raise ValueError("every partition must be a non-empty one-dimensional sequence")
        if partition.min() < 0 or partition.max() >= values.shape[0]:
            raise IndexError("a partition contains an out-of-range gene index")
        if len(np.unique(partition)) != len(partition) or assigned[partition].any():
            raise ValueError("partitions must contain unique, mutually disjoint indices")
        permutation[partition] = rng.permutation(partition)
        assigned[partition] = True
    return values[permutation].copy(), permutation
