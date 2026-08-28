import numpy as np
import torch

from spatios2e.models import FactorizedDotProductDecoder
from spatios2e.training import (
    HeldOutTrainingConfig,
    evaluate_heldout_decoder,
    fit_heldout_decoder,
)


def test_heldout_training_never_requires_heldout_targets_for_fitting():
    generator = torch.Generator().manual_seed(17)
    gene_vectors = torch.randn(8, 5, generator=generator)
    spots = torch.randn(18, 4, generator=generator)
    expression = torch.nn.functional.softplus(
        spots[:, :3] @ gene_vectors[:, :3].transpose(0, 1)
    )
    training_batches = [
        {"spot_features": spots[:10], "expression": expression[:10]},
    ]
    validation_batches = [
        {"spot_features": spots[10:14], "expression": expression[10:14]},
    ]
    test_batches = [
        {"spot_features": spots[14:], "expression": expression[14:]},
    ]
    model = FactorizedDotProductDecoder(
        spot_dim=4,
        gene_dim=5,
        hidden_dim=10,
        program_dim=3,
        dropout=0.0,
    )
    result = fit_heldout_decoder(
        model,
        training_batches,
        validation_batches,
        gene_vectors,
        training_gene_indices=np.arange(6),
        config=HeldOutTrainingConfig(
            epochs=2,
            genes_per_batch=4,
            validation_genes=4,
            seed=3,
        ),
    )
    metrics = evaluate_heldout_decoder(
        model,
        test_batches,
        gene_vectors,
        gene_indices=[6, 7],
    )

    assert result.best_epoch in (1, 2)
    assert len(result.history) == 2
    assert set(result.validation_gene_indices).issubset(set(range(6)))
    assert metrics["n_genes"] == 2
    assert abs(metrics["decomposition_error"]) < 1.0e-12
