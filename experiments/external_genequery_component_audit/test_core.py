#!/usr/bin/env python3
"""Small dependency-light regression tests for the external audit core."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from audit_predictions import matrix_metrics
from common import counterfactual_embeddings
from model import GeneQueryHead


class AuditCoreTest(unittest.TestCase):
    def test_exact_mse_decomposition(self) -> None:
        true = np.asarray([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
        pred = true + np.asarray([[1.0, -2.0], [1.0, 0.0], [1.0, 2.0]])
        metrics = matrix_metrics(pred, true)
        self.assertAlmostEqual(
            metrics["overall_mse"],
            metrics["abundance_mse"] + metrics["centered_mse"],
            places=12,
        )

    def test_counterfactuals_preserve_shape_and_partitions(self) -> None:
        embeddings = np.arange(48, dtype=np.float32).reshape(8, 6)
        labels = np.asarray(["train"] * 5 + ["heldout"] * 3)
        shuffled = counterfactual_embeddings(embeddings, labels, "identity_shuffle", 42)
        self.assertEqual(shuffled.shape, embeddings.shape)
        for label in ("train", "heldout"):
            mask = labels == label
            self.assertEqual(
                sorted(map(tuple, shuffled[mask].tolist())),
                sorted(map(tuple, embeddings[mask].tolist())),
            )
        constant = counterfactual_embeddings(embeddings, labels, "constant", 42)
        self.assertTrue(np.allclose(constant, constant[0]))

    def test_genequery_head_accepts_unseen_query_count(self) -> None:
        model = GeneQueryHead(
            image_dim=16,
            gene_embedding_dim=12,
            hidden_dim=8,
            depth=2,
            heads=2,
            dim_head=4,
            dropout=0.0,
        )
        train_output = model(torch.randn(3, 16), torch.randn(7, 12))
        heldout_output = model(torch.randn(3, 16), torch.randn(5, 12))
        self.assertEqual(tuple(train_output.shape), (3, 7))
        self.assertEqual(tuple(heldout_output.shape), (3, 5))


if __name__ == "__main__":
    unittest.main()
