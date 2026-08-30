#!/usr/bin/env python3
"""Build and validate the manuscript gene-partition checksum manifest."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "manuscript"
COHORTS = ("hippocampus_donor_disjoint", "dlpfc", "nac", "her2st")
SENSITIVITIES = (
    "historical",
    "expression_seed123",
    "expression_seed456",
    "embedding_cluster",
    "chromosome_blocked",
)


def _lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _partition_record(directory: Path) -> dict[str, object]:
    train_path = directory / "train_genes.txt"
    heldout_path = directory / "heldout_genes.txt"
    train = _lines(train_path)
    heldout = _lines(heldout_path)
    if len(train) != len(set(train)):
        raise ValueError(f"Duplicate training genes: {train_path}")
    if len(heldout) != len(set(heldout)):
        raise ValueError(f"Duplicate held-out genes: {heldout_path}")
    if set(train) & set(heldout):
        raise ValueError(f"Overlapping train/held-out genes: {directory}")
    return {
        "train": {
            "path": train_path.relative_to(ROOT).as_posix(),
            "n_genes": len(train),
            "sha256": _sha256(train_path),
        },
        "heldout": {
            "path": heldout_path.relative_to(ROOT).as_posix(),
            "n_genes": len(heldout),
            "sha256": _sha256(heldout_path),
        },
        "universe_n_genes": len(train) + len(heldout),
    }


def _render_manifest() -> str:
    partitions: dict[str, dict[str, dict[str, object]]] = {"primary": {}}
    for cohort in COHORTS:
        partitions["primary"][cohort] = _partition_record(CONFIG / "gene_splits" / cohort)

    sensitivity_root = CONFIG / "gene_splits_sensitivity"
    for name in SENSITIVITIES:
        partitions[name] = {}
        for cohort in COHORTS:
            partitions[name][cohort] = _partition_record(sensitivity_root / name / cohort)

    overlap: dict[str, dict[str, float | int]] = {}
    for cohort in COHORTS:
        primary_path = CONFIG / "gene_splits" / cohort / "heldout_genes.txt"
        historical_path = sensitivity_root / "historical" / cohort / "heldout_genes.txt"
        primary = set(_lines(primary_path))
        historical = set(_lines(historical_path))
        union = primary | historical
        overlap[cohort] = {
            "intersection_n_genes": len(primary & historical),
            "union_n_genes": len(union),
            "jaccard": len(primary & historical) / len(union),
        }

    payload = {
        "schema_version": 1,
        "identifier": "stable Ensembl identifiers in file order",
        "partitions": partitions,
        "primary_vs_historical_heldout_overlap": overlap,
        "construction": {
            "primary": "ten training-expression rank bins; 20% sampled within bin; seed 42",
            "historical": "prespecified earlier working-set partition retained only as sensitivity analysis",
            "expression_seed123": "primary construction repeated with sampling seed 123",
            "expression_seed456": "primary construction repeated with sampling seed 456",
            "embedding_cluster": (
                "standardized Decima vectors; randomized PCA(50, seed 42); ten MiniBatchKMeans "
                "clusters within each expression bin; full clusters assigned held out"
            ),
            "chromosome_blocked": "chromosomes 1, 14, 15 and X held out",
        },
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def main() -> None:
    output = CONFIG / "gene_partition_manifest.json"
    rendered = _render_manifest()
    if "--check" in sys.argv[1:]:
        if not output.exists() or output.read_text() != rendered:
            raise SystemExit(f"STALE: regenerate {output.relative_to(ROOT)}")
        print(f"PASS: {output.relative_to(ROOT)} is current")
        return
    if sys.argv[1:]:
        raise SystemExit("usage: build_manuscript_partition_manifest.py [--check]")
    output.write_text(rendered)
    print(f"Wrote {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
