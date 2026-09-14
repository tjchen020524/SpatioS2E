#!/usr/bin/env python3
"""Audit across-gene mean PCC under constant gene-vector conditions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.heldout_metric_policy import PREDICTED_STD_MIN
from experiments.multicohort_geneheldout_decima_clean_split.clean_split_config import (
    COHORTS,
    ROOT,
    SEEDS,
)


OUT = ROOT / "experiments/heldout_gene_second_stage/constant_gene_mean_pcc_audit"


def sources() -> list[tuple[str, str, Path]]:
    rows: list[tuple[str, str, Path]] = []
    decima = ROOT / "experiments/multicohort_geneheldout_decima_clean_split/components/runs"
    scgpt = ROOT / "experiments/heldout_gene_second_stage/scgpt_whole_human_decoder"
    for cohort in COHORTS:
        for seed in SEEDS:
            rows.append(
                (
                    "Decima-derived vectors",
                    cohort,
                    decima / cohort / f"seed_{seed}/constant",
                )
            )
            rows.append(
                (
                    "scGPT whole-human static gene-token vectors",
                    cohort,
                    scgpt
                    / "constant_gene_vector/components/runs"
                    / cohort
                    / f"seed_{seed}/decima",
                )
            )
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    audit_rows: list[dict[str, object]] = []
    for representation, cohort, run_root in sources():
        table_path = run_root / "gene_components.tsv.gz"
        summary_path = run_root / "summary.json"
        table = pd.read_csv(
            table_path, sep="\t", usecols=["seed", "predicted_mean", "observed_mean"]
        )
        summary = json.loads(summary_path.read_text())
        predicted = table["predicted_mean"].to_numpy(dtype=np.float64)
        observed = table["observed_mean"].to_numpy(dtype=np.float64)
        predicted_std = float(np.std(predicted))
        audit_rows.append(
            {
                "representation": representation,
                "cohort": cohort,
                "seed": int(table["seed"].iloc[0]),
                "n_heldout_genes": int(len(table)),
                "predicted_gene_mean_std": predicted_std,
                "predicted_gene_mean_range": float(np.ptp(predicted)),
                "observed_gene_mean_std": float(np.std(observed)),
                "stored_raw_abundance_pcc": summary.get("abundance_pcc"),
                "gene_mean_pcc_defined": bool(predicted_std > PREDICTED_STD_MIN),
                "publication_value": "NA",
                "component_table": str(table_path.relative_to(ROOT)),
            }
        )
    audit = pd.DataFrame(audit_rows).sort_values(["representation", "cohort", "seed"])
    output = OUT / "constant_condition_per_run.tsv"
    audit.to_csv(output, sep="\t", index=False)
    checks = {
        "complete_24_run_grid": len(audit) == 2 * len(COHORTS) * len(SEEDS),
        "all_predicted_gene_mean_std_at_or_below_threshold": bool(
            audit["predicted_gene_mean_std"].le(PREDICTED_STD_MIN).all()
        ),
        "all_gene_mean_pcc_undefined": bool(~audit["gene_mean_pcc_defined"].any()),
        "all_publication_values_na": bool(audit["publication_value"].eq("NA").all()),
    }
    manifest = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "question": "Is across-gene PCC of predicted gene means defined for a constant gene vector?",
        "answer": "No. Every target receives the same predicted map; chunk-level floating-point noise is below the registered 1e-6 variability threshold.",
        "predicted_std_threshold": PREDICTED_STD_MIN,
        "checks": checks,
        "maximum_predicted_gene_mean_std": float(audit["predicted_gene_mean_std"].max()),
        "output": str(output.relative_to(ROOT)),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if manifest["status"] != "PASS":
        raise AssertionError(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
