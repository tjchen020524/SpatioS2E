"""Training utilities for individual- and target-disjoint decoders.

The caller supplies iterable batches with ``spot_features`` (or ``x``) and a
complete ``expression`` (or ``y``) matrix. Only ``training_gene_indices`` are
sampled during optimization and checkpoint selection. Held-out targets can be
passed to :func:`evaluate_heldout_decoder` only after fitting is complete.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from spatios2e.evaluation import component_metrics


@dataclass(frozen=True)
class HeldOutTrainingConfig:
    """Optimization settings used by the manuscript held-out-target assay."""

    epochs: int = 6
    learning_rate: float = 2.0e-4
    weight_decay: float = 1.0e-4
    genes_per_batch: int = 512
    validation_genes: int = 2_048
    gene_correlation_weight: float = 0.15
    selection_mse_weight: float = 0.02
    gradient_clip_norm: float = 1.0
    seed: int = 42

    def validate(self) -> None:
        if self.epochs < 1:
            raise ValueError("epochs must be positive")
        if self.genes_per_batch < 1 or self.validation_genes < 1:
            raise ValueError("gene-sampling sizes must be positive")
        if self.learning_rate <= 0 or self.gradient_clip_norm <= 0:
            raise ValueError("learning_rate and gradient_clip_norm must be positive")


@dataclass(frozen=True)
class HeldOutTrainingResult:
    """Checkpoint-selection record returned after fitting."""

    best_epoch: int
    best_score: float
    validation_gene_indices: np.ndarray
    history: list[dict[str, object]]
    config: dict[str, object]


def gene_correlation_loss(predicted: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
    """Return ``1 - mean PCC`` across genes for one spot minibatch."""

    if predicted.shape != observed.shape or predicted.ndim != 2:
        raise ValueError("predicted and observed must be aligned [spot, gene] matrices")
    pred_centered = predicted.float() - predicted.float().mean(dim=0, keepdim=True)
    true_centered = observed.float() - observed.float().mean(dim=0, keepdim=True)
    pred_scale = torch.sqrt(pred_centered.square().mean(dim=0) + 1.0e-6)
    true_scale = torch.sqrt(true_centered.square().mean(dim=0) + 1.0e-6)
    correlation = (pred_centered * true_centered).mean(dim=0) / (pred_scale * true_scale)
    return 1.0 - correlation.mean()


def _batch_tensors(batch: object) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(batch, Mapping):
        features = batch.get("spot_features", batch.get("x"))
        expression = batch.get("expression", batch.get("y"))
    elif isinstance(batch, (tuple, list)) and len(batch) >= 2:
        features, expression = batch[:2]
    else:
        raise TypeError("Each batch must be a mapping or a (spot_features, expression) pair")
    if not isinstance(features, torch.Tensor) or not isinstance(expression, torch.Tensor):
        raise TypeError("Batch spot features and expression must be torch tensors")
    if features.ndim != 2 or expression.ndim != 2 or features.shape[0] != expression.shape[0]:
        raise ValueError("Batch tensors must be aligned two-dimensional matrices")
    return features, expression


@torch.no_grad()
def evaluate_heldout_decoder(
    model: torch.nn.Module,
    batches: Iterable[object],
    gene_vectors: np.ndarray | torch.Tensor,
    gene_indices: Sequence[int],
    *,
    device: torch.device | str = "cpu",
    gene_chunk_size: int = 512,
) -> dict[str, float | int]:
    """Evaluate a fitted decoder with the package's component endpoints.

    This convenience implementation materializes the requested prediction
    matrix. For very large cohorts, callers can evaluate one section at a time
    and aggregate the exported sufficient statistics externally.
    """

    if gene_chunk_size < 1:
        raise ValueError("gene_chunk_size must be positive")
    device = torch.device(device)
    vectors = torch.as_tensor(gene_vectors, dtype=torch.float32, device=device)
    indices = np.asarray(gene_indices, dtype=np.int64)
    if indices.ndim != 1 or not len(indices):
        raise ValueError("gene_indices must be a non-empty one-dimensional sequence")
    if indices.min() < 0 or indices.max() >= vectors.shape[0]:
        raise IndexError("gene_indices contain an out-of-range index")

    model.eval()
    predicted_batches: list[np.ndarray] = []
    observed_batches: list[np.ndarray] = []
    for batch in batches:
        features, expression = _batch_tensors(batch)
        features = features.to(device)
        expression = expression.to(device)
        predicted_chunks: list[torch.Tensor] = []
        observed_chunks: list[torch.Tensor] = []
        for start in range(0, len(indices), gene_chunk_size):
            chunk = torch.as_tensor(
                indices[start : start + gene_chunk_size],
                dtype=torch.long,
                device=device,
            )
            predicted_chunks.append(model(features, vectors.index_select(0, chunk)))
            observed_chunks.append(expression.index_select(1, chunk))
        predicted_batches.append(torch.cat(predicted_chunks, dim=1).cpu().numpy())
        observed_batches.append(torch.cat(observed_chunks, dim=1).cpu().numpy())
    if not predicted_batches:
        raise ValueError("No evaluation batches were supplied")
    return component_metrics(
        np.concatenate(observed_batches, axis=0),
        np.concatenate(predicted_batches, axis=0),
    )


def fit_heldout_decoder(
    model: torch.nn.Module,
    training_batches: Iterable[object],
    validation_batches: Iterable[object],
    gene_vectors: np.ndarray | torch.Tensor,
    training_gene_indices: Sequence[int],
    *,
    config: HeldOutTrainingConfig = HeldOutTrainingConfig(),
    device: torch.device | str = "cpu",
) -> HeldOutTrainingResult:
    """Fit a shared decoder without exposing held-out targets to optimization."""

    config.validate()
    device = torch.device(device)
    model.to(device)
    vectors = torch.as_tensor(gene_vectors, dtype=torch.float32, device=device)
    training = np.asarray(training_gene_indices, dtype=np.int64)
    if training.ndim != 1 or len(training) < 2:
        raise ValueError("training_gene_indices must contain at least two genes")
    if training.min() < 0 or training.max() >= vectors.shape[0]:
        raise IndexError("training_gene_indices contain an out-of-range index")

    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    validation = np.sort(
        rng.choice(
            training,
            size=min(config.validation_genes, len(training)),
            replace=False,
        )
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    best_epoch = 0
    best_score = -float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, object]] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        weighted_loss = 0.0
        number_of_spots = 0
        for batch in training_batches:
            features, expression = _batch_tensors(batch)
            features = features.to(device)
            expression = expression.to(device)
            sampled = np.sort(
                rng.choice(
                    training,
                    size=min(config.genes_per_batch, len(training)),
                    replace=False,
                )
            )
            gene_index = torch.as_tensor(sampled, dtype=torch.long, device=device)
            prediction = model(features, vectors.index_select(0, gene_index))
            target = expression.index_select(1, gene_index)
            mse = F.mse_loss(prediction, target)
            loss = mse + config.gene_correlation_weight * gene_correlation_loss(
                prediction,
                target,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            optimizer.step()
            batch_size = int(features.shape[0])
            weighted_loss += float(loss.detach().cpu()) * batch_size
            number_of_spots += batch_size

        validation_metrics = evaluate_heldout_decoder(
            model,
            validation_batches,
            vectors,
            validation,
            device=device,
            gene_chunk_size=min(512, len(validation)),
        )
        score = float(validation_metrics["mean_gene_pcc"]) - (
            config.selection_mse_weight * float(validation_metrics["full_matrix_mse"])
        )
        history.append(
            {
                "epoch": epoch,
                "training_loss": weighted_loss / max(number_of_spots, 1),
                "validation": validation_metrics,
                "checkpoint_score": score,
            }
        )
        if score > best_score:
            best_epoch = epoch
            best_score = score
            best_state = deepcopy(
                {name: value.detach().cpu() for name, value in model.state_dict().items()}
            )

    if best_state is None:
        raise RuntimeError("Training did not produce a finite checkpoint score")
    model.load_state_dict(best_state)
    model.to(device)
    return HeldOutTrainingResult(
        best_epoch=best_epoch,
        best_score=best_score,
        validation_gene_indices=validation,
        history=history,
        config=asdict(config),
    )
