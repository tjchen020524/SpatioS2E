"""Run a weight-free illustration of matrix-level versus spatial agreement."""
import json

import numpy as np

from spatios2e.evaluation import component_metrics


def example_metrics():
    rng = np.random.default_rng(42)
    gene_means = np.linspace(0.5, 4.0, 24)
    spatial = rng.normal(0, 0.2, size=(80, 24))
    spatial -= spatial.mean(axis=0, keepdims=True)
    observed = gene_means[None, :] + spatial
    mean_only = np.broadcast_to(gene_means, observed.shape)
    predictions = {"mean_only": mean_only, "mean_and_spatial": mean_only + 0.8 * spatial}
    keys = ("full_matrix_pcc", "gene_mean_pcc", "mean_gene_pcc", "centered_rmse", "decomposition_error")
    results = {}
    for name, prediction in predictions.items():
        metrics = component_metrics(observed, prediction)
        assert abs(metrics["decomposition_error"]) < 1e-12
        results[name] = {key: float(metrics[key]) for key in keys}
    return results


if __name__ == "__main__":
    print(json.dumps(example_metrics(), indent=2, allow_nan=False))
