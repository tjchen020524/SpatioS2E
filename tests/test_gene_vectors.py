import numpy as np

from spatios2e.models import (
    constant_gene_vectors,
    permute_gene_identity,
    random_gene_vectors,
    standardize_from_training_genes,
)


def test_vector_controls_are_reproducible_and_dimension_matched():
    first = random_gene_vectors((7, 5), seed=123)
    second = random_gene_vectors((7, 5), seed=123)
    constant = constant_gene_vectors((7, 5))

    assert first.shape == constant.shape
    assert np.array_equal(first, second)
    assert np.all(constant == 0.0)


def test_pretrained_standardization_uses_training_genes_only():
    vectors = np.arange(30, dtype=np.float32).reshape(6, 5)
    standardized = standardize_from_training_genes(vectors, [0, 1, 2, 3])

    assert np.allclose(standardized[:4].mean(axis=0), 0.0, atol=1.0e-6)
    assert np.allclose(standardized[:4].std(axis=0), 1.0, atol=1.0e-6)


def test_identity_permutation_preserves_vector_set():
    vectors = np.arange(24, dtype=np.float32).reshape(6, 4)
    permuted, permutation = permute_gene_identity(vectors, seed=456)

    assert np.array_equal(permuted, vectors[permutation])
    assert sorted(permutation.tolist()) == list(range(vectors.shape[0]))
