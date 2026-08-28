"""Decoders used in the target-disjoint held-out-target assays.

These modules operate on precomputed spot representations and fixed gene
vectors.  They do not load or fine-tune the model that produced either input.
"""

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


def _program_encoder(input_dim: int, hidden_dim: int, program_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(input_dim),
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, program_dim),
    )


class FactorizedDotProductDecoder(nn.Module):
    """Primary held-out-target decoder with gene-bias and interaction paths.

    For spot representation ``x_i`` and gene vector ``e_g``, the model returns
    ``softplus(<f_spot(x_i), f_gene(e_g)> / sqrt(k) + f_bias(e_g))``.
    """

    def __init__(
        self,
        spot_dim: int,
        gene_dim: int,
        hidden_dim: int = 512,
        program_dim: int = 96,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_dim < 2:
            raise ValueError("hidden_dim must be at least 2")
        if program_dim < 1:
            raise ValueError("program_dim must be positive")
        self.spot_encoder = _program_encoder(spot_dim, hidden_dim, program_dim, dropout)
        self.gene_encoder = _program_encoder(gene_dim, hidden_dim, program_dim, dropout)
        self.gene_bias = nn.Sequential(
            nn.LayerNorm(gene_dim),
            nn.Linear(gene_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.program_dim = int(program_dim)

    def forward_branches(self, spot_features: torch.Tensor, gene_vectors: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Return the complete prediction and its two pre-softplus paths."""

        spot_program = self.spot_encoder(spot_features)
        gene_program = self.gene_encoder(gene_vectors)
        interaction = spot_program @ gene_program.transpose(0, 1)
        interaction = interaction / math.sqrt(float(self.program_dim))
        gene_bias = self.gene_bias(gene_vectors).transpose(0, 1)
        preactivation = interaction + gene_bias
        return {
            "prediction": F.softplus(preactivation),
            "preactivation": preactivation,
            "gene_bias": gene_bias,
            "interaction": interaction,
            "spot_program": spot_program,
            "gene_program": gene_program,
        }

    def forward(self, spot_features: torch.Tensor, gene_vectors: torch.Tensor) -> torch.Tensor:
        return self.forward_branches(spot_features, gene_vectors)["prediction"]


class BiasFreeFactorizedDecoder(nn.Module):
    """Factorized decoder with one shared intercept and no gene-bias path."""

    def __init__(
        self,
        spot_dim: int,
        gene_dim: int,
        hidden_dim: int = 512,
        program_dim: int = 96,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if program_dim < 1:
            raise ValueError("program_dim must be positive")
        self.spot_encoder = _program_encoder(spot_dim, hidden_dim, program_dim, dropout)
        self.gene_encoder = _program_encoder(gene_dim, hidden_dim, program_dim, dropout)
        self.global_intercept = nn.Parameter(torch.zeros(()))
        self.program_dim = int(program_dim)

    def forward_branches(self, spot_features: torch.Tensor, gene_vectors: torch.Tensor) -> Dict[str, torch.Tensor]:
        spot_program = self.spot_encoder(spot_features)
        gene_program = self.gene_encoder(gene_vectors)
        interaction = spot_program @ gene_program.transpose(0, 1)
        interaction = interaction / math.sqrt(float(self.program_dim))
        preactivation = interaction + self.global_intercept
        return {
            "prediction": F.softplus(preactivation),
            "preactivation": preactivation,
            "interaction": interaction,
            "global_intercept": self.global_intercept,
            "spot_program": spot_program,
            "gene_program": gene_program,
        }

    def forward(self, spot_features: torch.Tensor, gene_vectors: torch.Tensor) -> torch.Tensor:
        return self.forward_branches(spot_features, gene_vectors)["prediction"]


class ConcatenationMLPDecoder(nn.Module):
    """One-hidden-layer MLP on every spot--gene vector pair.

    Separate spot and gene projections are algebraically equivalent to the
    first linear layer on their concatenation, while avoiding materializing a
    repeated ``[n_spots, n_genes, spot_dim + gene_dim]`` tensor.
    """

    def __init__(
        self,
        spot_dim: int,
        gene_dim: int,
        hidden_dim: int = 704,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.spot_norm = nn.LayerNorm(spot_dim)
        self.gene_norm = nn.LayerNorm(gene_dim)
        self.spot_projection = nn.Linear(spot_dim, hidden_dim, bias=True)
        self.gene_projection = nn.Linear(gene_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, spot_features: torch.Tensor, gene_vectors: torch.Tensor) -> torch.Tensor:
        spot_hidden = self.spot_projection(self.spot_norm(spot_features))
        gene_hidden = self.gene_projection(self.gene_norm(gene_vectors))
        joint = F.gelu(spot_hidden[:, None, :] + gene_hidden[None, :, :])
        return F.softplus(self.output(self.dropout(joint)).squeeze(-1))


class SectionCenteredResidualDecoder(nn.Module):
    """Signed factorized decoder used when the target is centred expression.

    The returned values are uncentred logits.  Centre predictions separately
    within each tissue section before computing the residual loss or endpoints.
    """

    def __init__(
        self,
        spot_dim: int,
        gene_dim: int,
        hidden_dim: int = 512,
        program_dim: int = 96,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if program_dim < 1:
            raise ValueError("program_dim must be positive")
        self.spot_encoder = _program_encoder(spot_dim, hidden_dim, program_dim, dropout)
        self.gene_encoder = _program_encoder(gene_dim, hidden_dim, program_dim, dropout)
        self.program_dim = int(program_dim)

    def forward(self, spot_features: torch.Tensor, gene_vectors: torch.Tensor) -> torch.Tensor:
        spot_program = self.spot_encoder(spot_features)
        gene_program = self.gene_encoder(gene_vectors)
        return (spot_program @ gene_program.transpose(0, 1)) / math.sqrt(float(self.program_dim))
