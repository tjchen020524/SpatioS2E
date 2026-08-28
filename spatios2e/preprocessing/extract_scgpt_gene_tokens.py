"""Extract static gene-token vectors from a packaged scGPT checkpoint.

The manuscript uses one fixed vector per gene: the corresponding row of
``encoder.embedding.weight`` after the checkpoint's ``encoder.enc_norm``
layer normalization. No cell-specific expression values or contextual
transformer outputs enter this representation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

EMBEDDING_KEY = "encoder.embedding.weight"
NORM_WEIGHT_KEY = "encoder.enc_norm.weight"
NORM_BIAS_KEY = "encoder.enc_norm.bias"


def sha256_file(path: Path) -> str:
    """Return a lowercase SHA-256 digest for ``path``."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _model_state(checkpoint: object) -> Mapping[str, torch.Tensor]:
    if not isinstance(checkpoint, Mapping):
        raise TypeError("The scGPT checkpoint must contain a mapping of tensor names")
    if EMBEDDING_KEY in checkpoint:
        state = checkpoint
    elif isinstance(checkpoint.get("model_state_dict"), Mapping):
        state = checkpoint["model_state_dict"]
    elif isinstance(checkpoint.get("model"), Mapping):
        state = checkpoint["model"]
    else:
        raise KeyError(f"The checkpoint does not contain {EMBEDDING_KEY!r}")

    if EMBEDDING_KEY not in state and f"module.{EMBEDDING_KEY}" in state:
        state = {
            key.removeprefix("module."): value
            for key, value in state.items()
        }
    missing = [
        key
        for key in (EMBEDDING_KEY, NORM_WEIGHT_KEY, NORM_BIAS_KEY)
        if key not in state
    ]
    if missing:
        raise KeyError(f"Missing required scGPT checkpoint tensors: {missing}")
    return state


def layer_normalized_gene_tokens(
    checkpoint: object,
    vocabulary: Mapping[str, int],
    *,
    epsilon: float = 1.0e-5,
) -> tuple[list[str], np.ndarray]:
    """Return vocabulary-ordered static scGPT gene-token vectors.

    Special tokens are retained because their interpretation belongs to the
    supplied vocabulary. Downstream code should match requested gene symbols
    explicitly and record any out-of-vocabulary targets.
    """

    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    state = _model_state(checkpoint)
    embedding = state[EMBEDDING_KEY].detach().cpu().float()
    weight = state[NORM_WEIGHT_KEY].detach().cpu().float()
    bias = state[NORM_BIAS_KEY].detach().cpu().float()
    if embedding.ndim != 2:
        raise ValueError("scGPT embedding weight must be two-dimensional")
    if weight.shape != (embedding.shape[1],) or bias.shape != (embedding.shape[1],):
        raise ValueError("scGPT layer-normalization tensors do not match the embedding dimension")

    entries: list[tuple[int, str]] = []
    seen_indices: set[int] = set()
    for symbol, raw_index in vocabulary.items():
        index = int(raw_index)
        if index < 0 or index >= embedding.shape[0]:
            raise IndexError(f"Vocabulary index for {symbol!r} is outside the embedding table")
        if index in seen_indices:
            raise ValueError(f"Vocabulary contains duplicate token index {index}")
        seen_indices.add(index)
        entries.append((index, str(symbol)))
    entries.sort()

    selected = embedding[[index for index, _ in entries]]
    row_mean = selected.mean(dim=1, keepdim=True)
    row_variance = selected.var(dim=1, keepdim=True, unbiased=False)
    normalized = (selected - row_mean) / torch.sqrt(row_variance + float(epsilon))
    normalized = normalized * weight[None, :] + bias[None, :]
    symbols = [symbol for _, symbol in entries]
    return symbols, normalized.numpy().astype(np.float32, copy=False)


def load_checkpoint(path: Path) -> object:
    """Load tensors with PyTorch's restricted loader when available."""

    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - compatibility with early PyTorch 2.x
        return torch.load(path, map_location="cpu")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vocabulary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epsilon", type=float, default=1.0e-5)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {args.output}; pass --overwrite to replace it")
    vocabulary = json.loads(args.vocabulary.read_text())
    if not isinstance(vocabulary, dict):
        raise TypeError("The scGPT vocabulary JSON must contain a symbol-to-index object")
    symbols, vectors = layer_normalized_gene_tokens(
        load_checkpoint(args.checkpoint),
        vocabulary,
        epsilon=args.epsilon,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        gene_symbols=np.asarray(symbols, dtype=np.str_),
        vectors=vectors,
    )
    metadata = {
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "vocabulary_sha256": sha256_file(args.vocabulary),
        "number_of_tokens": len(symbols),
        "embedding_dimension": int(vectors.shape[1]),
        "extraction": (
            "encoder.embedding.weight token row followed by encoder.enc_norm "
            "LayerNorm; no cell-specific or contextual transformer output"
        ),
        "layer_norm_epsilon": args.epsilon,
    }
    args.output.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
