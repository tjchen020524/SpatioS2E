import torch

from spatios2e.models import (
    BiasFreeFactorizedDecoder,
    ConcatenationMLPDecoder,
    FactorizedDotProductDecoder,
    SectionCenteredResidualDecoder,
)


def _inputs():
    torch.manual_seed(7)
    return torch.randn(5, 12), torch.randn(9, 16)


def test_factorized_decoder_branch_accounting():
    spots, genes = _inputs()
    decoder = FactorizedDotProductDecoder(12, 16, hidden_dim=20, program_dim=8, dropout=0.0)
    decoder.eval()
    branches = decoder.forward_branches(spots, genes)

    assert branches["prediction"].shape == (5, 9)
    assert branches["gene_bias"].shape == (1, 9)
    assert torch.allclose(
        branches["preactivation"],
        branches["interaction"] + branches["gene_bias"],
    )
    assert torch.allclose(decoder(spots, genes), branches["prediction"])
    assert bool((branches["prediction"] > 0).all())


def test_audit_decoders_have_expected_outputs():
    spots, genes = _inputs()
    bias_free = BiasFreeFactorizedDecoder(12, 16, hidden_dim=20, program_dim=8, dropout=0.0)
    concat = ConcatenationMLPDecoder(12, 16, hidden_dim=24, dropout=0.0)
    residual = SectionCenteredResidualDecoder(12, 16, hidden_dim=20, program_dim=8, dropout=0.0)

    assert bias_free(spots, genes).shape == (5, 9)
    assert concat(spots, genes).shape == (5, 9)
    assert residual(spots, genes).shape == (5, 9)
    assert "gene_bias" not in bias_free.forward_branches(spots, genes)
    assert not hasattr(concat, "gene_bias")


def test_manuscript_decoder_parameter_counts():
    models = {
        "full": FactorizedDotProductDecoder(1536, 1920),
        "bias_free": BiasFreeFactorizedDecoder(1536, 1920),
        "concat": ConcatenationMLPDecoder(1536, 1920),
        "residual": SectionCenteredResidualDecoder(1536, 1920),
    }
    observed = {name: sum(parameter.numel() for parameter in model.parameters()) for name, model in models.items()}
    assert observed == {
        "full": 2_371_777,
        "bias_free": 1_875_905,
        "concat": 2_441_345,
        "residual": 1_875_904,
    }
