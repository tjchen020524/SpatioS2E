#!/usr/bin/env python3
"""Audit overlap between downstream cohorts and Decima supervision metadata."""

from __future__ import annotations

import json

import pandas as pd

from experiments.multicohort_geneheldout_decima_clean_split.clean_split_config import ROOT


CELL_META = ROOT / "data/decima_input/metadata/cell_meta.tsv"
OUT = ROOT / "experiments/heldout_gene_second_stage/decima_pretraining_audit"
DOWNSTREAM = {
    "Hippocampus": ("GSE264692", "GSE264692|Br2743|Br8492"),
    "DLPFC": ("GSE307403", "GSE307403"),
    "NAc": ("GSE307586", "GSE307586"),
    "HER2ST": ("HER2ST / Andersson et al. 2021", "HER2ST|Andersson"),
}
RELATED = {
    "Hippocampus": r"hippocamp",
    "DLPFC": r"dorsolateral prefrontal|dorsalateral prefrontal|prefrontal cortex",
    "NAc": r"nucleus accumbens",
    "HER2ST": r"breast|mammary",
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(CELL_META, sep="\t", low_memory=False)
    searchable_columns = [
        column
        for column in ("tissue", "organ", "disease", "study", "dataset", "region", "subregion")
        if column in table.columns
    ]
    searchable = table[searchable_columns].fillna("").astype(str).agg(" | ".join, axis=1)
    exact_rows: list[dict[str, object]] = []
    related_rows: list[dict[str, object]] = []
    for cohort, (declared_source, pattern) in DOWNSTREAM.items():
        mask = searchable.str.contains(pattern, case=False, regex=True)
        exact_rows.append(
            {
                "downstream_cohort": cohort,
                "downstream_identifier": declared_source,
                "n_matching_decima_supervision_rows": int(mask.sum()),
                "exact_identifier_detected": bool(mask.any()),
            }
        )
        related_mask = searchable.str.contains(RELATED[cohort], case=False, regex=True)
        related = table.loc[related_mask, searchable_columns].copy()
        studies = sorted(
            set(related.get("study", pd.Series(dtype=str)).dropna().astype(str))
        )
        related_rows.append(
            {
                "downstream_cohort": cohort,
                "related_tissue_pattern": RELATED[cohort],
                "n_related_decima_supervision_rows": int(related_mask.sum()),
                "n_related_studies": len(studies),
                "example_related_studies": "; ".join(studies[:12]),
            }
        )
    exact = pd.DataFrame(exact_rows)
    related = pd.DataFrame(related_rows)
    exact.to_csv(OUT / "exact_downstream_overlap.tsv", sep="\t", index=False)
    related.to_csv(OUT / "related_tissue_supervision.tsv", sep="\t", index=False)
    manifest = {
        "status": "PASS",
        "n_decima_supervision_rows": len(table),
        "searched_columns": searchable_columns,
        "checks": {
            "four_downstream_cohorts": len(exact) == 4,
            "no_exact_downstream_identifier_detected": bool(
                (~exact["exact_identifier_detected"]).all()
            ),
            "related_tissue_supervision_present": bool(
                (related["n_related_decima_supervision_rows"] > 0).all()
            ),
        },
        "interpretive_boundary": (
            "String matching of packaged Decima supervision metadata found no exact "
            "downstream cohort identifier, but it cannot prove absence from every upstream "
            "source or derivative. Related brain and breast supervision is present."
        ),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    print("\nExact overlap")
    print(exact.to_string(index=False))
    print("\nRelated tissue supervision")
    print(related.to_string(index=False))


if __name__ == "__main__":
    main()
