import numpy as np

from spatios2e.evaluation import center_within_section, component_metrics


def test_component_metrics_obey_exact_mse_identity():
    observed = np.asarray(
        [
            [1.0, 4.0, 2.0],
            [2.0, 3.0, 4.0],
            [3.0, 5.0, 3.0],
            [4.0, 6.0, 5.0],
        ]
    )
    predicted = observed + np.asarray(
        [
            [0.7, -0.2, 0.1],
            [0.4, 0.0, -0.3],
            [0.8, 0.4, 0.2],
            [0.5, 0.2, -0.4],
        ]
    )
    metrics = component_metrics(observed, predicted)

    assert abs(metrics["decomposition_error"]) < 1.0e-12
    assert np.isclose(
        metrics["full_matrix_mse"],
        metrics["gene_mean_mse"] + metrics["centered_mse"],
    )
    assert metrics["n_finite_gene_pcc"] == observed.shape[1]
    assert metrics["q25_gene_pcc"] <= metrics["median_gene_pcc"] <= metrics["q75_gene_pcc"]


def test_abundance_only_prediction_has_no_within_gene_variation():
    observed = np.asarray(
        [
            [0.0, 2.0, 5.0],
            [2.0, 4.0, 7.0],
            [4.0, 6.0, 9.0],
        ]
    )
    predicted = np.broadcast_to(observed.mean(axis=0), observed.shape)
    metrics = component_metrics(observed, predicted)

    assert np.isclose(metrics["abundance_pcc"], 1.0)
    assert np.isclose(metrics["abundance_rmse"], 0.0)
    assert metrics["mean_gene_pcc"] == 0.0
    assert metrics["centered_full_matrix_pcc"] == 0.0
    assert metrics["n_constant_prediction_gene_pcc"] == observed.shape[1]
    assert metrics["centered_rmse"] > 0


def test_section_centering_removes_section_specific_gene_means():
    matrix = np.asarray(
        [
            [1.0, 2.0],
            [3.0, 6.0],
            [10.0, 20.0],
            [14.0, 24.0],
        ]
    )
    section_ids = np.asarray(["a", "a", "b", "b"])
    centered = center_within_section(matrix, section_ids)

    assert np.allclose(centered[:2].mean(axis=0), 0.0)
    assert np.allclose(centered[2:].mean(axis=0), 0.0)
