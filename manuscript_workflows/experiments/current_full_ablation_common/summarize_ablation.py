#!/usr/bin/env python3
"""Collect one cohort's completed ablations and common-gene sensitivities."""

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
    args = parser.parse_args()
    cohort_dir = resolve(args.cohort_dir)
    manifest = json.loads((cohort_dir / "configs/variant_manifest.json").read_text())

    summary_rows = []
    gene_frames: dict[str, pd.DataFrame] = {}
    spatial_pcc_applicable = {
        entry["name"]: bool(entry.get("spatial_pcc_applicable", True))
        for entry in manifest
    }
    missing = []
    for entry in manifest:
        name = entry["name"]
        results = cohort_dir / "variants" / name / "results"
        overall_path = results / "test_overall.json"
        genes_path = results / "test_gene_metrics.tsv"
        if not overall_path.exists() or not genes_path.exists():
            missing.append(name)
            continue
        overall = json.loads(overall_path.read_text())
        overall["variant"] = name
        summary_rows.append(overall)
        frame = pd.read_csv(genes_path, sep="\t").set_index("gene_id")
        gene_frames[name] = frame

    output_dir = cohort_dir / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        first_cols = ["variant", "mse", "corr", "gene_corr_mean_primary", "gene_corr_mean_finite", "gene_corr_coverage"]
        ordered = [col for col in first_cols if col in summary_df] + [
            col for col in summary_df.columns if col not in first_cols
        ]
        summary_df[ordered].to_csv(output_dir / "ablation_summary.csv", index=False)

    common_rows = []
    sensitivity_frames = {
        name: frame
        for name, frame in gene_frames.items()
        if spatial_pcc_applicable.get(name, True)
    }
    if sensitivity_frames:
        common = None
        for frame in sensitivity_frames.values():
            finite_ids = set(frame.index[np.isfinite(frame["corr_finite"])])
            common = finite_ids if common is None else common & finite_ids
        common = common or set()
        for name, frame in gene_frames.items():
            applicable = spatial_pcc_applicable.get(name, True)
            values = (
                frame.loc[sorted(common), "corr_finite"]
                if common and applicable
                else pd.Series(dtype=float)
            )
            common_rows.append(
                {
                    "variant": name,
                    "spatial_pcc_applicable": applicable,
                    "n_common_finite_genes": len(common) if applicable else 0,
                    "common_finite_gene_corr_mean": float(values.mean()) if len(values) else float("nan"),
                    "common_finite_gene_corr_median": float(values.median()) if len(values) else float("nan"),
                }
            )
    pd.DataFrame(common_rows).to_csv(output_dir / "common_finite_sensitivity.csv", index=False)
    run_summary = {
        "cohort_dir": str(cohort_dir),
        "n_expected_variants": len(manifest),
        "n_completed_variants": len(summary_rows),
        "missing_variants": missing,
    }
    (output_dir / "run_summary.json").write_text(json.dumps(run_summary, indent=2))
    print(json.dumps(run_summary, indent=2))


if __name__ == "__main__":
    main()
