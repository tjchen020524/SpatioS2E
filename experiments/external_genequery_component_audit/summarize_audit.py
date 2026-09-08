#!/usr/bin/env python3
"""Summarize seeds and compute exact component attribution against controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import RUN_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--expected-seeds", type=int, nargs="+", help="Require exactly these completed seeds.")
    args = parser.parse_args()
    output_dir = args.output_dir or args.run_root / "summary"
    rows = []
    for path in sorted(args.run_root.glob("seed_*/*/audit/summary.json")):
        seed = int(path.parents[2].name.split("_", 1)[1])
        variant = path.parents[1].name
        values = json.loads(path.read_text())
        rows.append({"seed": seed, "variant": variant, **values})
    if not rows:
        raise FileNotFoundError("No audit summaries found below %s" % args.run_root)
    frame = pd.DataFrame(rows).sort_values(["seed", "variant"]).reset_index(drop=True)
    if frame.duplicated(["seed", "variant"]).any():
        raise ValueError("Duplicate seed/condition summaries")
    if args.expected_seeds is not None and set(frame["seed"]) != set(args.expected_seeds):
        raise ValueError("Completed seeds do not match --expected-seeds")
    required = {"semantic", "identity_shuffle", "random", "constant"}
    for seed, group in frame.groupby("seed"):
        if not required.issubset(set(group["variant"])):
            raise ValueError("Incomplete matched conditions for seed %s" % seed)
    if not np.allclose(frame["overall_mse"], frame["abundance_mse"] + frame["centered_mse"], rtol=0, atol=1e-10):
        raise ValueError("Input MSE components do not sum to total error")
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "all_runs.tsv", sep="\t", index=False)

    metric_columns = [
        "full_matrix_pcc",
        "abundance_pcc",
        "mean_gene_pcc",
        "centered_full_matrix_pcc",
        "section_centered_full_matrix_pcc",
        "section_centered_mean_gene_pcc",
        "overall_mse",
        "abundance_mse",
        "centered_mse",
        "section_centered_mse",
    ]
    aggregate_rows = []
    for variant, group in frame.groupby("variant", sort=True):
        row = {"variant": variant, "n_seeds": int(group["seed"].nunique())}
        for metric in metric_columns:
            row[metric + "_mean"] = float(group[metric].mean())
            row[metric + "_sd"] = float(group[metric].std(ddof=1)) if len(group) > 1 else float("nan")
        aggregate_rows.append(row)
    pd.DataFrame(aggregate_rows).to_csv(output_dir / "aggregate.tsv", sep="\t", index=False)

    comparisons = []
    for seed, group in frame.groupby("seed", sort=True):
        indexed = group.set_index("variant")
        if "semantic" not in indexed.index:
            continue
        semantic = indexed.loc["semantic"]
        for control in ("identity_shuffle", "random", "constant"):
            if control not in indexed.index:
                continue
            reference = indexed.loc[control]
            delta_total = float(reference["overall_mse"] - semantic["overall_mse"])
            delta_abundance = float(reference["abundance_mse"] - semantic["abundance_mse"])
            delta_centered = float(reference["centered_mse"] - semantic["centered_mse"])
            comparisons.append(
                {
                    "seed": seed,
                    "comparison": "semantic_vs_%s" % control,
                    "mse_improvement": delta_total,
                    "mse_change_direction": (
                        "unchanged" if abs(delta_total) <= 1e-12 else "reduction" if delta_total > 0 else "increase"
                    ),
                    "gene_mean_contribution": delta_abundance,
                    "centered_contribution": delta_centered,
                    "gene_mean_fraction": delta_abundance / delta_total if abs(delta_total) > 1.0e-12 else np.nan,
                    "attribution_error": delta_total - delta_abundance - delta_centered,
                    "delta_abundance_pcc": float(semantic["abundance_pcc"] - reference["abundance_pcc"]),
                    "delta_mean_gene_pcc": float(semantic["mean_gene_pcc"] - reference["mean_gene_pcc"]),
                    "delta_section_centered_full_matrix_pcc": float(
                        semantic["section_centered_full_matrix_pcc"]
                        - reference["section_centered_full_matrix_pcc"]
                    ),
                }
            )
    comparison_frame = pd.DataFrame(comparisons)
    comparison_frame.to_csv(output_dir / "component_attribution.tsv", sep="\t", index=False)
    print("Wrote %s (%d runs, %d comparisons)" % (output_dir, len(frame), len(comparison_frame)))


if __name__ == "__main__":
    main()
