"""External-model targets retain the full-panel library-size denominator."""

import sys

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from experiments.external_genequery_component_audit import common
from experiments.external_genequery_component_audit.deepspotm import predict_zero_shot


def test_query_subset_uses_full_library_denominator(tmp_path, monkeypatch):
    counts = np.array([[0.0, 2.0, 98.0], [20.0, 5.0, 175.0]])
    cpm = counts / counts.sum(axis=1, keepdims=True) * 1e6
    stored = sparse.csr_matrix(np.log1p(cpm).T.astype(np.float32))
    expression_dir = tmp_path / "data/expression/G1"
    expression_dir.mkdir(parents=True)
    np.savez(
        expression_dir / "test.npz",
        data=stored.data,
        indices=stored.indices,
        indptr=stored.indptr,
        shape=np.array(stored.shape),
        gene_ids=np.array(["g1", "g2", "g3"]),
        barcodes=np.array(["a", "b"]),
    )
    pd.DataFrame([dict(sample="G1", patient="G", split="test", n_spots=2)]).to_csv(
        tmp_path / "data/sample_manifest.tsv", sep="\t", index=False
    )
    monkeypatch.setattr(common, "HER2", tmp_path)
    actual, barcodes = common.load_expression("G1", ["g2", "g1"], target_scale="log1p_cp10k")
    expected = np.log1p(counts[:, [1, 0]] / counts.sum(axis=1, keepdims=True) * 1e4)
    np.testing.assert_allclose(actual, expected, atol=1e-6)
    np.testing.assert_array_equal(barcodes, ["a", "b"])
    default, _ = common.load_expression("G1", ["g2", "g1"])
    np.testing.assert_array_equal(default, np.log1p(cpm[:, [1, 0]]).astype(np.float32))
    partition, samples, _ = common.load_expression_partition("test", ["g2", "g1"], target_scale="log1p_cp10k")
    np.testing.assert_array_equal(partition, actual)
    np.testing.assert_array_equal(samples, ["G1", "G1"])


def test_deepspot_dataset_requests_cp10k(tmp_path, monkeypatch):
    directory = tmp_path / "G1"
    directory.mkdir()
    np.save(directory / "patches.npy", np.zeros((2, 224, 224, 3), dtype=np.uint8))
    np.save(directory / "barcodes.npy", np.array(["a", "b"]))
    monkeypatch.setattr(predict_zero_shot, "load_manifest", lambda: pd.DataFrame([dict(sample="G1", split="test")]))
    calls = []

    def expression(sample, genes, *, target_scale):
        calls.append((sample, list(genes), target_scale))
        return np.array([[1.0], [2.0]], dtype=np.float32), np.array(["a", "b"])

    monkeypatch.setattr(predict_zero_shot, "load_expression", expression)
    dataset = predict_zero_shot.DeepSpotTestDataset(np.array(["g1"]), tmp_path)
    assert calls == [("G1", ["g1"], "log1p_cp10k")]
    np.testing.assert_array_equal(dataset[1][1].numpy(), [2.0])


def test_ridge_fits_and_exports_on_selected_scale(tmp_path, monkeypatch):
    # Import as the CLI does, sharing its plain 'common' module.
    from experiments.external_genequery_component_audit import mean_only_baseline as ridge

    labels = np.array(["train"] * 10 + ["heldout"] * 2)
    vectors = np.arange(36, dtype=float).reshape(12, 3)
    panel = dict(
        split_labels=labels,
        gene_ids=np.array([f"g{i}" for i in range(12)]),
        symbols=np.array([f"G{i}" for i in range(12)]),
        semantic_embeddings=vectors,
    )
    monkeypatch.setattr(ridge, "load_panel_artifact", lambda _: panel)
    scales = []

    def expression(partition, genes, max_spots, *, target_scale):
        scales.append(target_scale)
        return np.tile(np.arange(12, dtype=float), (2, 1)), np.array(["G1", "G1"]), np.array(["a", "b"])

    monkeypatch.setattr(ridge, "load_expression_partition", expression)
    monkeypatch.setattr(ridge, "load_manifest", lambda: pd.DataFrame([dict(sample="G1", patient="G")]))
    monkeypatch.setattr(sys, "argv", ["ridge", "--output-root", str(tmp_path), "--target-scale", "log1p_cp10k"])
    ridge.main()
    import json

    run = tmp_path / "seed_42/semantic_mean_only"
    assert scales == ["log1p_cp10k", "log1p_cp10k"]
    assert json.loads((run / "run.json").read_text())["target_scale"] == "log1p_counts_per_10000"
    with np.load(run / "predictions.npz") as data:
        np.testing.assert_array_equal(data["true"], [[10.0, 11.0], [10.0, 11.0]])


@pytest.mark.parametrize("bad", [np.array([-1.0]), np.array([np.nan])])
def test_cp10k_rejects_invalid_expression(bad):
    with pytest.raises(ValueError, match="finite nonnegative"):
        common.expression_on_scale(bad, "log1p_cp10k")
