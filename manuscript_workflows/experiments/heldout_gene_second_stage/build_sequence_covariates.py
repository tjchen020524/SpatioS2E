#!/usr/bin/env python3
"""Compute simple sequence-composition covariates from packaged Decima inputs."""

from __future__ import annotations
from workflow_paths import DATA_ROOT

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = DATA_ROOT
INPUT = ROOT / "data/decima_input/gene_inputs_npz"
OUTPUT = (
    ROOT
    / "experiments/heldout_gene_second_stage/detection_covariates/sequence_covariates.tsv"
)


def summarize(path_string: str) -> dict[str, object]:
    path = Path(path_string)
    with np.load(path) as archive:
        sequence = archive["seq_mask"][:4].astype(np.float64, copy=False)
    valid = float(sequence.sum())
    gc = float(sequence[1:3].sum())
    at = float(sequence[[0, 3]].sum())
    return {
        "gene_id": path.stem,
        "window_gc_fraction": gc / valid if valid > 0 else np.nan,
        "window_at_fraction": at / valid if valid > 0 else np.nan,
        "window_valid_bases": int(round(valid)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    paths = sorted(str(path) for path in INPUT.glob("*.npz"))
    if len(paths) != 18_397:
        raise ValueError(f"Expected 18,397 packaged gene inputs; found {len(paths)}")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        table = pd.DataFrame(pool.map(summarize, paths, chunksize=16))
    if table["gene_id"].nunique() != len(paths) or table["window_gc_fraction"].isna().any():
        raise ValueError("Incomplete sequence-covariate table")
    table.to_csv(OUTPUT, sep="\t", index=False)
    print(table.describe(include="all").to_string())
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
