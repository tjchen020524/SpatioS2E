import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from spatios2e.models import FactorizedDotProductDecoder
from spatios2e.training import (
    HeldOutTrainingConfig,
    evaluate_heldout_decoder,
    fit_heldout_decoder,
)


@pytest.mark.parametrize("batch_source", ["list", "dataloader", "generator"])
def test_heldout_training_never_requires_heldout_targets_for_fitting(batch_source):
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
    epochs = 2
    if batch_source == "dataloader":
        training_batches = DataLoader(TensorDataset(spots[:10], expression[:10]), batch_size=10)
        validation_batches = DataLoader(TensorDataset(spots[10:14], expression[10:14]), batch_size=4)
    elif batch_source == "generator":
        training_batches = (batch for batch in training_batches)
        validation_batches = (batch for batch in validation_batches)
        epochs = 1
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
            epochs=epochs,
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

    assert 1 <= result.best_epoch <= epochs
    assert len(result.history) == epochs
    assert set(result.validation_gene_indices).issubset(set(range(6)))
    assert metrics["n_genes"] == 2
    assert abs(metrics["decomposition_error"]) < 1.0e-12


@pytest.mark.parametrize("source_name", ["training_batches", "validation_batches"])
@pytest.mark.parametrize("iterator_kind", ["generator", "list_iterator"])
def test_multiple_epochs_reject_one_shot_batches_before_training(source_name, iterator_kind):
    batch = {"spot_features": torch.ones(4, 4), "expression": torch.ones(4, 8)}
    one_shot = (item for item in [batch]) if iterator_kind == "generator" else iter([batch])
    batches = {"training_batches": [batch], "validation_batches": [batch]}
    batches[source_name] = one_shot
    model = FactorizedDotProductDecoder(
        spot_dim=4,
        gene_dim=5,
        hidden_dim=10,
        program_dim=3,
        dropout=0.0,
    )
    initial_state = {name: value.clone() for name, value in model.state_dict().items()}

    with pytest.raises(TypeError, match=f"{source_name} must be re-iterable"):
        fit_heldout_decoder(
            model,
            **batches,
            gene_vectors=torch.ones(8, 5),
            training_gene_indices=np.arange(6),
            config=HeldOutTrainingConfig(epochs=2),
        )

    assert next(one_shot) is batch
    for name, value in model.state_dict().items():
        assert torch.equal(value, initial_state[name])
