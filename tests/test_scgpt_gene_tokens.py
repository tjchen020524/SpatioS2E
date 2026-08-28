import numpy as np
import torch

from spatios2e.preprocessing.extract_scgpt_gene_tokens import (
    layer_normalized_gene_tokens,
)


def test_static_scgpt_tokens_apply_checkpoint_layer_normalization():
    embedding = torch.tensor(
        [
            [1.0, 2.0, 4.0],
            [8.0, 3.0, 1.0],
            [0.0, 0.0, 0.0],
        ]
    )
    state = {
        "encoder.embedding.weight": embedding,
        "encoder.enc_norm.weight": torch.tensor([1.0, 2.0, 0.5]),
        "encoder.enc_norm.bias": torch.tensor([0.1, -0.2, 0.3]),
    }
    symbols, vectors = layer_normalized_gene_tokens(
        state,
        {"GENE_B": 1, "GENE_A": 0},
    )

    expected = torch.nn.functional.layer_norm(
        embedding[:2],
        normalized_shape=(3,),
        weight=state["encoder.enc_norm.weight"],
        bias=state["encoder.enc_norm.bias"],
        eps=1.0e-5,
    ).numpy()
    assert symbols == ["GENE_A", "GENE_B"]
    assert vectors.dtype == np.float32
    assert np.allclose(vectors, expected, atol=1.0e-6)


def test_static_scgpt_tokens_reject_duplicate_vocabulary_indices():
    state = {
        "encoder.embedding.weight": torch.zeros(2, 3),
        "encoder.enc_norm.weight": torch.ones(3),
        "encoder.enc_norm.bias": torch.zeros(3),
    }
    try:
        layer_normalized_gene_tokens(state, {"A": 0, "B": 0})
    except ValueError as error:
        assert "duplicate token index" in str(error)
    else:
        raise AssertionError("duplicate vocabulary indices must be rejected")
