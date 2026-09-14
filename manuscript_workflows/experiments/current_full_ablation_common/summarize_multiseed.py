#!/usr/bin/env python3
"""Aggregate per-seed current full-ablation metrics into mean and SD tables."""

from __future__ import annotations
from workflow_paths import DATA_ROOT

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = DATA_ROOT


def resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument(
        "--reuse-base-seed",
        type=int,
        help="Read this seed from variants/<name>/results instead of multiseed/.",
    )
    args = parser.parse_args()

    cohort_dir = resolve(args.cohort_dir)
    manifest = json.loads((cohort_dir / "configs/variant_manifest.json").read_text())
    rows = []
    missing = []
    for seed in args.seeds:
        for entry in manifest:
            name = entry["name"]
            if args.reuse_base_seed == seed:
                results = cohort_dir / "variants" / name / "results"
            else:
                results = cohort_dir / "multiseed" / f"seed_{seed}" / "variants" / name / "results"
            overall_path = results / "test_overall.json"
            if not overall_path.exists():
                missing.append({"seed": seed, "variant": name, "path": str(overall_path)})
                continue
            row = json.loads(overall_path.read_text())
            row["seed"] = int(seed)
            row["variant"] = name
            row["result_dir"] = str(results.relative_to(ROOT))
            rows.append(row)

    output_dir = cohort_dir / "multiseed/results"
    output_dir.mkdir(parents=True, exist_ok=True)
    per_seed = pd.DataFrame(rows)
    if not per_seed.empty:
        front = ["variant", "seed", "mse", "corr", "gene_corr_mean_primary"]
        ordered = [column for column in front if column in per_seed] + [
            column for column in per_seed if column not in front
        ]
        per_seed[ordered].to_csv(output_dir / "per_seed_ablation_summary.csv", index=False)

        excluded = {"seed", "variant", "split", "result_dir", "no_graph"}
        numeric = [
            column
            for column in per_seed.columns
            if column not in excluded and pd.api.types.is_numeric_dtype(per_seed[column])
        ]
        grouped = per_seed.groupby("variant", sort=False)
        mean = grouped[numeric].mean().add_suffix("_mean")
        sd = grouped[numeric].std(ddof=1).add_suffix("_sd")
        count = grouped["seed"].nunique().rename("n_seeds_completed")
        aggregate = pd.concat([count, mean, sd], axis=1).reset_index()
        aggregate.to_csv(output_dir / "ablation_summary_mean_sd.csv", index=False)

        core_metrics = [
            metric
            for metric in (
                "mse",
                "corr",
                "gene_corr_mean_primary",
                "top50_pcc_primary",
                "top2000_pcc_primary",
            )
            if metric in numeric
        ]
        long_rows = []
        for row in aggregate.itertuples(index=False):
            for metric in core_metrics:
                long_rows.append(
                    {
                        "variant": row.variant,
                        "metric": metric,
                        "mean": getattr(row, f"{metric}_mean"),
                        "sd": getattr(row, f"{metric}_sd"),
                        "n_seeds": row.n_seeds_completed,
                    }
                )
        pd.DataFrame(long_rows).to_csv(output_dir / "core_metrics_mean_sd_long.csv", index=False)

    summary = {
        "cohort_dir": str(cohort_dir.relative_to(ROOT)),
        "seeds_expected": [int(seed) for seed in args.seeds],
        "n_expected_runs": len(args.seeds) * len(manifest),
        "n_completed_runs": len(rows),
        "missing": missing,
        "complete": len(missing) == 0,
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
