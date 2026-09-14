#!/usr/bin/env python3
"""Audit and freeze the training-only master gene partition for replication."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from experiments.multicohort_geneheldout_decima_clean_split.clean_split_config import (
    CLEAN_ROOT,
    COHORTS,
    ROOT,
    SOURCE_ROOT,
)


MASTER = CLEAN_ROOT / "partition/master"
HIP_ROOT = ROOT / "experiments/hippocampus_current_full_ablation"
GENERATOR = CODE_ROOT / "experiments/stpath_seq_mechanism_controls/build_cold_gene_split.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def compute_training_means(samples: list[str]) -> tuple[list[str], np.ndarray, int, list[dict[str, object]]]:
    gene_ids_ref: list[str] | None = None
    gene_sum: np.ndarray | None = None
    total_spots = 0
    inputs: list[dict[str, object]] = []
    for sample in samples:
        path = HIP_ROOT / "data/expression" / sample / "train.npz"
        values = np.load(path, allow_pickle=True)
        shape = tuple(values["shape"].tolist())
        matrix = sparse.csr_matrix(
            (values["data"], values["indices"], values["indptr"]), shape=shape
        )
        gene_ids = values["gene_ids"].astype(str).tolist()
        if gene_ids_ref is None:
            gene_ids_ref = gene_ids
            gene_sum = np.asarray(matrix.sum(axis=1)).ravel().astype(np.float64)
        else:
            if gene_ids != gene_ids_ref:
                raise ValueError(f"Gene order mismatch: {path}")
            assert gene_sum is not None
            gene_sum += np.asarray(matrix.sum(axis=1)).ravel().astype(np.float64)
        total_spots += int(shape[1])
        inputs.append(
            {
                "sample": sample,
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256(path),
                "n_genes": int(shape[0]),
                "n_spots": int(shape[1]),
            }
        )
    if gene_ids_ref is None or gene_sum is None:
        raise ValueError("No final-training hippocampus expression matrices were found")
    return gene_ids_ref, gene_sum / total_spots, total_spots, inputs


def reproduce_assignment(mean_expression: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    order = np.argsort(mean_expression)
    bins = np.array_split(order, 10)
    heldout: list[int] = []
    bin_index = np.zeros(len(mean_expression), dtype=np.int64)
    for bin_number, members in enumerate(bins, start=1):
        bin_index[members] = bin_number
        n_holdout = max(1, int(round(members.size * 0.20)))
        heldout.extend(int(value) for value in rng.choice(members, size=n_holdout, replace=False))
    return np.asarray(sorted(set(heldout)), dtype=np.int64), bin_index


def main() -> None:
    split_path = HIP_ROOT / "data/split.json"
    split = json.loads(split_path.read_text())
    final_train_samples = [str(sample) for sample in split["train"]]
    if len(final_train_samples) != 22:
        raise ValueError(f"Expected 22 final-training hippocampus sections, found {len(final_train_samples)}")
    gene_ids, mean_expression, total_spots, input_rows = compute_training_means(final_train_samples)
    heldout_idx, bin_index = reproduce_assignment(mean_expression)
    heldout_set = set(heldout_idx.tolist())
    expected_heldout = [gene_ids[index] for index in heldout_idx]
    expected_train = [gene for index, gene in enumerate(gene_ids) if index not in heldout_set]
    stored_train = read_lines(MASTER / "train_genes.txt")
    stored_heldout = read_lines(MASTER / "heldout_genes.txt")
    if stored_train != expected_train or stored_heldout != expected_heldout:
        raise ValueError("Frozen partition differs from the original ten-bin seed-42 algorithm")
    if len(stored_train) != 14717 or len(stored_heldout) != 3680:
        raise ValueError("Unexpected clean master partition size")

    assignment = np.full(len(gene_ids), "train", dtype=object)
    assignment[heldout_idx] = "heldout"
    pd.DataFrame(
        {
            "gene_id": gene_ids,
            "training_mean_expression": mean_expression,
            "expression_bin": bin_index,
            "assignment": assignment,
        }
    ).to_csv(
        MASTER / "stratification.tsv.gz",
        sep="\t",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )

    cohort_rows: list[dict[str, object]] = []
    master_train = set(stored_train)
    master_heldout = set(stored_heldout)
    for cohort in COHORTS:
        source_data = SOURCE_ROOT / "data" / cohort
        cohort_gene_ids = read_lines(source_data / "gene_ids.txt")
        train_genes = [gene for gene in cohort_gene_ids if gene in master_train]
        heldout_genes = [gene for gene in cohort_gene_ids if gene in master_heldout]
        if len(train_genes) + len(heldout_genes) != len(cohort_gene_ids):
            raise ValueError(f"Master partition does not cover {cohort}")
        output = CLEAN_ROOT / "data" / cohort
        split_dir = output / "gene_splits"
        split_dir.mkdir(parents=True, exist_ok=True)
        (output / "gene_ids.txt").write_text("\n".join(cohort_gene_ids) + "\n")
        (split_dir / "train_genes.txt").write_text("\n".join(train_genes) + "\n")
        (split_dir / "heldout_genes.txt").write_text("\n".join(heldout_genes) + "\n")
        cohort_rows.append(
            {
                "cohort": cohort,
                "n_genes": len(cohort_gene_ids),
                "n_train_genes": len(train_genes),
                "n_heldout_genes": len(heldout_genes),
                "train_genes_sha256": sha256(split_dir / "train_genes.txt"),
                "heldout_genes_sha256": sha256(split_dir / "heldout_genes.txt"),
                "source_gene_ids": str((source_data / "gene_ids.txt").relative_to(ROOT)),
                "source_manifest_sha256": sha256(source_data / "manifest.tsv"),
                "source_split_sha256": sha256(source_data / "split.json"),
            }
        )
    pd.DataFrame(cohort_rows).to_csv(CLEAN_ROOT / "partition/cohort_partitions.tsv", sep="\t", index=False)

    historical_root = ROOT / "experiments/stpath_seq_mechanism_controls/artifacts"
    historical_train = read_lines(historical_root / "train_genes.txt")
    historical_heldout = read_lines(historical_root / "heldout_genes.txt")
    historical_heldout_set = set(historical_heldout)
    clean_heldout_set = set(stored_heldout)
    manifest = {
        "design": "training-only clean replication of the historical master gene partition",
        "algorithm": {
            "source_script": str(GENERATOR.relative_to(ROOT)),
            "source_script_sha256": sha256(GENERATOR),
            "holdout_fraction": 0.20,
            "n_equal_rank_bins": 10,
            "random_seed": 42,
            "only_design_change": "expression stratification uses the final hippocampus training individuals only",
        },
        "hippocampus_final_split": str(split_path.relative_to(ROOT)),
        "hippocampus_final_split_sha256": sha256(split_path),
        "n_training_sections": len(final_train_samples),
        "training_sections": final_train_samples,
        "total_training_spots": total_spots,
        "training_expression_inputs": input_rows,
        "n_master_genes": len(gene_ids),
        "n_train_genes": len(stored_train),
        "n_heldout_genes": len(stored_heldout),
        "master_train_sha256": sha256(MASTER / "train_genes.txt"),
        "master_heldout_sha256": sha256(MASTER / "heldout_genes.txt"),
        "stratification_sha256": sha256(MASTER / "stratification.tsv.gz"),
        "historical_partition_comparison": {
            "historical_train_sha256": sha256(historical_root / "train_genes.txt"),
            "historical_heldout_sha256": sha256(historical_root / "heldout_genes.txt"),
            "n_historical_heldout": len(historical_heldout),
            "n_clean_heldout": len(stored_heldout),
            "n_heldout_overlap": len(historical_heldout_set & clean_heldout_set),
            "n_changed_on_each_side": len(historical_heldout_set - clean_heldout_set),
            "heldout_jaccard": len(historical_heldout_set & clean_heldout_set)
            / len(historical_heldout_set | clean_heldout_set),
            "historical_train_count": len(historical_train),
        },
    }
    (CLEAN_ROOT / "partition/manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({key: value for key, value in manifest.items() if key != "training_expression_inputs"}, indent=2))


if __name__ == "__main__":
    main()
