#!/usr/bin/env python3
"""Run a bounded target-disjoint smoke test on frozen real-cohort inputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import sparse

from spatios2e.models import FactorizedDotProductDecoder
from spatios2e.training import HeldOutTrainingConfig, evaluate_heldout_decoder, fit_heldout_decoder


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_genes(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def _validate_manifest_biological_split(manifest: pd.DataFrame, split_path: Path) -> dict[str, list[str]]:
    """Require every manifest row to follow the frozen manuscript section split."""
    required_columns = {"sample", "split", "barcode"}
    missing_columns = sorted(required_columns - set(manifest.columns))
    if missing_columns:
        raise ValueError(f"Spot manifest lacks required columns: {', '.join(missing_columns)}")

    payload = json.loads(split_path.read_text())
    required_splits = ("train", "val", "test")
    missing_splits = [split for split in required_splits if split not in payload]
    if missing_splits:
        raise ValueError(f"Biological split lacks keys: {', '.join(missing_splits)}")

    normalized = {split: [str(sample) for sample in payload[split]] for split in required_splits}
    sample_to_split: dict[str, str] = {}
    for split, samples in normalized.items():
        if len(samples) != len(set(samples)):
            raise ValueError(f"Biological split contains duplicate samples within {split}")
        for sample in samples:
            previous = sample_to_split.setdefault(sample, split)
            if previous != split:
                raise ValueError(f"Sample {sample} occurs in both {previous} and {split}")

    observed_samples = set(manifest["sample"].astype(str))
    expected_samples = set(sample_to_split)
    missing_samples = sorted(expected_samples - observed_samples)
    unexpected_samples = sorted(observed_samples - expected_samples)
    if missing_samples or unexpected_samples:
        raise ValueError(
            "Spot manifest and biological split contain different samples; "
            f"missing={missing_samples[:5]}, unexpected={unexpected_samples[:5]}"
        )

    observed_split = manifest["split"].astype(str)
    expected_split = manifest["sample"].astype(str).map(sample_to_split)
    mismatch = observed_split != expected_split
    if mismatch.any():
        examples = manifest.loc[mismatch, ["sample", "split"]].drop_duplicates().head(5)
        rendered = ", ".join(
            f"{row.sample}: observed={row.split}, expected={sample_to_split[str(row.sample)]}"
            for row in examples.itertuples(index=False)
        )
        raise ValueError(f"Spot manifest violates the frozen biological split ({rendered})")
    return normalized


def _load_spots(
    manifest: pd.DataFrame,
    split: str,
    expression_root: Path,
    embedding_root: Path,
    selected_genes: list[str],
    spots_per_split: int,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, object]]]:
    rows = manifest.loc[manifest["split"] == split].head(spots_per_split).copy()
    if len(rows) != spots_per_split:
        raise ValueError(f"Requested {spots_per_split} {split} spots but found {len(rows)}")
    feature_blocks: list[np.ndarray] = []
    expression_blocks: list[np.ndarray] = []
    inputs: list[dict[str, object]] = []
    for sample, sample_rows in rows.groupby("sample", sort=False):
        sample = str(sample)
        requested_barcodes = sample_rows["barcode"].astype(str).tolist()

        expression_path = expression_root / sample / f"{split}.npz"
        expression_archive = np.load(expression_path, allow_pickle=True)
        matrix = sparse.csr_matrix(
            (
                expression_archive["data"],
                expression_archive["indices"],
                expression_archive["indptr"],
            ),
            shape=tuple(expression_archive["shape"].tolist()),
        )
        expression_genes = expression_archive["gene_ids"].astype(str).tolist()
        gene_index = {gene: index for index, gene in enumerate(expression_genes)}
        missing = [gene for gene in selected_genes if gene not in gene_index]
        if missing:
            raise ValueError(f"Expression file lacks {len(missing)} selected genes: {expression_path}")
        expression_barcodes = expression_archive["barcodes"].astype(str).tolist()
        barcode_index = {barcode: index for index, barcode in enumerate(expression_barcodes)}
        expression_columns = [barcode_index[barcode] for barcode in requested_barcodes]
        expression_rows = [gene_index[gene] for gene in selected_genes]
        values = matrix[expression_rows, :][:, expression_columns].transpose().toarray().astype(np.float32)

        embedding_path = embedding_root / sample / "embeddings.npz"
        embedding_archive = np.load(embedding_path, allow_pickle=True)
        embedding_barcodes = embedding_archive["barcodes"].astype(str).tolist()
        embedding_index = {barcode: index for index, barcode in enumerate(embedding_barcodes)}
        feature_rows = [embedding_index[barcode] for barcode in requested_barcodes]
        features = embedding_archive["embeddings"][feature_rows].astype(np.float32, copy=False)

        expression_blocks.append(values)
        feature_blocks.append(features)
        inputs.extend(
            [
                {
                    "role": "expression",
                    "sample": sample,
                    "split": split,
                    "name": expression_path.name,
                    "bytes": expression_path.stat().st_size,
                    "sha256": _sha256(expression_path),
                },
                {
                    "role": "spot representation",
                    "sample": sample,
                    "name": embedding_path.name,
                    "bytes": embedding_path.stat().st_size,
                    "sha256": _sha256(embedding_path),
                },
            ]
        )

    return (
        torch.tensor(np.concatenate(feature_blocks), dtype=torch.float32),
        torch.tensor(np.concatenate(expression_blocks), dtype=torch.float32),
        inputs,
    )


def _batches(features: torch.Tensor, expression: torch.Tensor, batch_size: int) -> list[dict[str, torch.Tensor]]:
    return [
        {"spot_features": features[start : start + batch_size], "expression": expression[start : start + batch_size]}
        for start in range(0, len(features), batch_size)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--biological-split", type=Path, required=True)
    parser.add_argument("--expression-root", type=Path, required=True)
    parser.add_argument("--embedding-root", type=Path, required=True)
    parser.add_argument("--gene-vectors", type=Path, required=True)
    parser.add_argument("--gene-split-dir", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--spots-per-split", type=int, default=128)
    parser.add_argument("--training-genes", type=int, default=128)
    parser.add_argument("--heldout-genes", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest, sep="\t", dtype=str)
    biological_split = _validate_manifest_biological_split(manifest, args.biological_split)
    vector_archive = np.load(args.gene_vectors, allow_pickle=False)
    vector_genes = vector_archive["gene_ids"].astype(str).tolist()
    vectors = vector_archive["embeddings"].astype(np.float32, copy=False)
    vector_index = {gene: index for index, gene in enumerate(vector_genes)}

    complete_training = _read_genes(args.gene_split_dir / "train_genes.txt")
    complete_heldout = _read_genes(args.gene_split_dir / "heldout_genes.txt")
    training_genes = complete_training[: args.training_genes]
    heldout_genes = complete_heldout[: args.heldout_genes]
    if len(training_genes) != args.training_genes or len(heldout_genes) != args.heldout_genes:
        raise ValueError("Requested gene count exceeds the archived partition")
    selected_genes = training_genes + heldout_genes

    complete_training_rows = np.asarray([vector_index[gene] for gene in complete_training], dtype=np.int64)
    mean = vectors[complete_training_rows].mean(axis=0, keepdims=True)
    scale = vectors[complete_training_rows].std(axis=0, keepdims=True)
    selected_rows = np.asarray([vector_index[gene] for gene in selected_genes], dtype=np.int64)
    pretrained = ((vectors[selected_rows] - mean) / np.maximum(scale, 1.0e-6)).astype(np.float32)

    split_tensors: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    input_records = [
        {
            "role": "spot manifest",
            "name": args.manifest.name,
            "bytes": args.manifest.stat().st_size,
            "sha256": _sha256(args.manifest),
        },
        {
            "role": "frozen biological split",
            "name": args.biological_split.name,
            "bytes": args.biological_split.stat().st_size,
            "sha256": _sha256(args.biological_split),
        },
        {
            "role": "gene representation",
            "name": args.gene_vectors.name,
            "bytes": args.gene_vectors.stat().st_size,
            "sha256": _sha256(args.gene_vectors),
        },
    ]
    for split in ("train", "val", "test"):
        features, expression, records = _load_spots(
            manifest,
            split,
            args.expression_root,
            args.embedding_root,
            selected_genes,
            args.spots_per_split,
        )
        split_tensors[split] = (features, expression)
        input_records.extend(records)

    training_indices = np.arange(len(training_genes), dtype=np.int64)
    heldout_indices = np.arange(len(training_genes), len(selected_genes), dtype=np.int64)
    config = HeldOutTrainingConfig(
        epochs=1,
        genes_per_batch=min(512, len(training_genes)),
        validation_genes=min(2_048, len(training_genes)),
        seed=args.seed,
    )
    outputs = []
    for variant in ("pretrained", "random"):
        torch.manual_seed(args.seed)
        if variant == "pretrained":
            gene_vectors = pretrained
        else:
            gene_vectors = np.random.default_rng(args.seed + 12_345).standard_normal(pretrained.shape).astype(np.float32)
        model = FactorizedDotProductDecoder(
            spot_dim=split_tensors["train"][0].shape[1],
            gene_dim=gene_vectors.shape[1],
            hidden_dim=512,
            program_dim=96,
            dropout=0.1,
        )
        result = fit_heldout_decoder(
            model,
            _batches(*split_tensors["train"], batch_size=args.spots_per_split),
            _batches(*split_tensors["val"], batch_size=args.spots_per_split),
            gene_vectors,
            training_indices,
            config=config,
        )
        metrics = evaluate_heldout_decoder(
            model,
            _batches(*split_tensors["test"], batch_size=args.spots_per_split),
            gene_vectors,
            heldout_indices,
        )
        outputs.append({"variant": variant, "best_epoch": result.best_epoch, **metrics})

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    table_path = args.output_prefix.with_suffix(".tsv")
    fieldnames = list(outputs[0])
    with table_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(outputs)

    report = {
        "schema_version": 1,
        "status": "PASS",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "Bounded real-data hippocampus smoke test using one epoch, 128 spots per biological split, "
            "128 downstream training genes and 32 held-out genes. It exercises the manuscript decoder "
            "dimensions and target-disjoint API but is not a reproduction or estimate of manuscript performance."
        ),
        "seed": args.seed,
        "counts": {
            "spots_per_split": args.spots_per_split,
            "training_genes": len(training_genes),
            "heldout_genes": len(heldout_genes),
            "sections": {split: len(samples) for split, samples in biological_split.items()},
        },
        "training_gene_sha256": hashlib.sha256("\n".join(training_genes).encode()).hexdigest(),
        "heldout_gene_sha256": hashlib.sha256("\n".join(heldout_genes).encode()).hexdigest(),
        "inputs": input_records,
        "output_table": {"name": table_path.name, "sha256": _sha256(table_path)},
        "results": outputs,
    }
    report_path = args.output_prefix.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"PASS: wrote {table_path} and {report_path}")


if __name__ == "__main__":
    main()
