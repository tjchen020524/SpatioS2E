import numpy as np

from spatios2e.evaluation import (
    broadcast_gene_means,
    component_metrics,
    fit_gene_mean_counterfactual,
)


def test_gene_mean_counterfactual_recovers_vector_signal_without_spatial_variation():
    rng = np.random.default_rng(11)
    vectors = rng.normal(size=(120, 8))
    gene_means = 2.0 + vectors @ rng.normal(size=8) * 0.15
    training = np.arange(90)
    heldout = np.arange(90, 120)
    model, diagnostics = fit_gene_mean_counterfactual(
        vectors,
        gene_means,
        training,
        alphas=(0.01, 1.0, 100.0),
        maximum_validation_genes=18,
    )
    predicted_means = model.predict(vectors[heldout])
    observed = np.broadcast_to(gene_means[heldout], (12, len(heldout))).copy()
    observed += rng.normal(scale=0.05, size=observed.shape)
    predicted = broadcast_gene_means(predicted_means, number_of_spots=12)
    metrics = component_metrics(observed, predicted)

    assert len(diagnostics) == 3
    assert metrics["abundance_pcc"] > 0.99
    assert metrics["mean_gene_pcc"] == 0.0
    assert metrics["centered_full_matrix_pcc"] == 0.0
    assert abs(metrics["decomposition_error"]) < 1.0e-12
