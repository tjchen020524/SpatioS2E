#!/usr/bin/env python3
"""Clean-room port of the public GeneQuery gene-aware prediction head."""

from __future__ import annotations


import torch
from torch import nn


class ProjectionHead(nn.Module):
    """Residual projection used by the official GeneQuery implementation."""

    def __init__(self, input_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(output_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        return self.layer_norm(x + projected)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int, dropout: float) -> None:
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, genes, _ = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = [
            value.reshape(batch, genes, self.heads, self.dim_head).transpose(1, 2)
            for value in qkv
        ]
        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        weights = scores.softmax(dim=-1)
        output = torch.matmul(weights, v)
        output = output.transpose(1, 2).reshape(batch, genes, self.heads * self.dim_head)
        return self.to_out(output)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class TransformerLayer(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int, dropout: float) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(dim)
        self.attention = Attention(dim, heads, dim_head, dropout)
        self.ff_norm = nn.LayerNorm(dim)
        self.feed_forward = FeedForward(dim, 2 * dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x))
        return x + self.feed_forward(self.ff_norm(x))


class GeneQueryHead(nn.Module):
    """GeneDesTransformer operating on precomputed ResNet-50 spot features.

    The public code passes ``dim_head`` as the number of heads and leaves the
    attention implementation's per-head dimension at 64.  Defaults here retain
    that behavior (64 heads x 64 dimensions) for fidelity.
    """

    def __init__(
        self,
        image_dim: int = 2048,
        gene_embedding_dim: int = 768,
        hidden_dim: int = 256,
        depth: int = 2,
        heads: int = 64,
        dim_head: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.image_projection = ProjectionHead(image_dim, hidden_dim, dropout)
        self.gene_projection = nn.Linear(gene_embedding_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        self.transformer_input_dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [TransformerLayer(hidden_dim, heads, dim_head, dropout) for _ in range(depth)]
        )
        self.regression = nn.Linear(hidden_dim, 1)

    def forward(self, image_features: torch.Tensor, gene_embeddings: torch.Tensor) -> torch.Tensor:
        image = self.image_projection(image_features).unsqueeze(1)
        gene = self.gene_projection(gene_embeddings).unsqueeze(0)
        x = self.activation(self.dropout(image + gene))
        x = self.transformer_input_dropout(x)
        for layer in self.layers:
            x = layer(x)
        return self.regression(x).squeeze(-1)


def parameter_count(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))
