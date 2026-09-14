#!/usr/bin/env python3
"""Validate and summarize the frozen DLPFC/HER2ST architecture-audit pilot."""

from __future__ import annotations
from workflow_paths import DATA_ROOT

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = DATA_ROOT
AUDIT = ROOT / "experiments/heldout_decoder_architecture_audit"
COHORTS = ("dlpfc", "her2st")
VARIANTS = ("decima", "random", "constant")


def load_json(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def main() -> None:
    bias_rows: list[dict[str, object]] = []
    residual_rows: list[dict[str, object]] = []
    stability_rows: list[dict[str, object]] = []
    for cohort in COHORTS:
        for variant in VARIANTS:
            bias_path = AUDIT / "bias_free/components/runs" / cohort / "seed_42" / variant / "summary.json"
            bias = load_json(bias_path)
            bias_rows.append(bias)
            history_path = (
                AUDIT
                / "bias_free/runs"
                / cohort
                / "seed_42"
                / variant
                / variant
                / "results/history.json"
            )
            history = load_json(history_path)
            finite_bias = bool(
                len(history) == 6
                and all(np.isfinite(float(row["train_loss"])) and np.isfinite(float(row["score"])) for row in history)
            )
            stability_rows.append(
                {"architecture": "bias_free", "cohort": cohort, "variant": variant, "pass": finite_bias}
            )

            residual_path = AUDIT / "residual_only/runs" / cohort / "seed_42" / variant / "results/summary.json"
            residual = load_json(residual_path)
            test = residual["heldout_test"]
            residual_rows.append(
                {
                    "cohort": cohort,
                    "seed": 42,
                    "variant": variant,
                    "best_epoch": residual["best_epoch"],
                    **test,
                }
            )
            stable_residual = bool(
                int(residual["protocol"]["epochs"]) == 6
                and float(test["max_abs_true_section_mean"]) < 1e-5
                and float(test["max_abs_pred_section_mean"]) < 1e-5
                and np.isfinite(float(test["primary_mean_within_section_gene_pcc"]))
                and np.isfinite(float(test["primary_section_centered_rmse"]))
            )
            stability_rows.append(
                {"architecture": "residual_only", "cohort": cohort, "variant": variant, "pass": stable_residual}
            )

    bias_frame = pd.DataFrame(bias_rows).sort_values(["cohort", "variant"])
    residual_frame = pd.DataFrame(residual_rows).sort_values(["cohort", "variant"])
    branch_frame = pd.concat(
        [
            pd.read_csv(AUDIT / "branch_intervention/runs" / cohort / "seed_42/per_run.tsv", sep="\t")
            for cohort in COHORTS
        ],
        ignore_index=True,
    ).sort_values(["cohort", "condition"])
    expected_branch_conditions = {
        "bias_correct__interaction_correct",
        "bias_correct__interaction_permuted",
        "bias_permuted__interaction_correct",
        "bias_permuted__interaction_permuted",
        "bias_correct__interaction_zero",
        "bias_zero__interaction_correct",
    }
    for cohort in COHORTS:
        observed = set(branch_frame.loc[branch_frame["cohort"] == cohort, "condition"])
        if observed != expected_branch_conditions:
            raise RuntimeError(f"Incomplete branch conditions for {cohort}: {sorted(observed)}")
    stability = pd.DataFrame(stability_rows)
    if not bool(stability["pass"].all()):
        raise RuntimeError("Pilot stability gate failed:\n" + stability.to_string(index=False))

    effect_rows: list[dict[str, object]] = []
    for cohort in COHORTS:
        bias_index = bias_frame.loc[bias_frame["cohort"] == cohort].set_index("variant")
        residual_index = residual_frame.loc[residual_frame["cohort"] == cohort].set_index("variant")
        for control in ("random", "constant"):
            effect_rows.append(
                {
                    "architecture": "bias_free",
                    "cohort": cohort,
                    "control": control,
                    "full_matrix_pcc_gain": float(bias_index.loc["decima", "pooled_pcc"] - bias_index.loc[control, "pooled_pcc"]),
                    "overall_mse_reduction": float(bias_index.loc[control, "overall_mse"] - bias_index.loc["decima", "overall_mse"]),
                    "abundance_pcc_gain": float(bias_index.loc["decima", "abundance_pcc"] - bias_index.loc[control, "abundance_pcc"]),
                    "mean_gene_pcc_gain": float(bias_index.loc["decima", "mean_gene_pcc"] - bias_index.loc[control, "mean_gene_pcc"]),
                    "centered_rmse_reduction": float(bias_index.loc[control, "centered_rmse"] - bias_index.loc["decima", "centered_rmse"]),
                }
            )
            effect_rows.append(
                {
                    "architecture": "residual_only",
                    "cohort": cohort,
                    "control": control,
                    "full_matrix_pcc_gain": float("nan"),
                    "overall_mse_reduction": float("nan"),
                    "abundance_pcc_gain": float("nan"),
                    "mean_gene_pcc_gain": float(
                        residual_index.loc["decima", "primary_mean_within_section_gene_pcc"]
                        - residual_index.loc[control, "primary_mean_within_section_gene_pcc"]
                    ),
                    "centered_rmse_reduction": float(
                        residual_index.loc[control, "primary_section_centered_rmse"]
                        - residual_index.loc["decima", "primary_section_centered_rmse"]
                    ),
                }
            )
    effects = pd.DataFrame(effect_rows)
    output = AUDIT / "pilot_summary"
    output.mkdir(parents=True, exist_ok=True)
    bias_frame.to_csv(output / "bias_free_per_run.tsv", sep="\t", index=False)
    residual_frame.to_csv(output / "residual_only_per_run.tsv", sep="\t", index=False)
    branch_frame.to_csv(output / "branch_intervention_per_run.tsv", sep="\t", index=False)
    stability.to_csv(output / "stability.tsv", sep="\t", index=False)
    effects.to_csv(output / "paired_effects.tsv", sep="\t", index=False)
    manifest = {
        "status": "PASS",
        "pilot": "DLPFC and HER2ST; seed 42; training-only gene split",
        "n_bias_free_runs": len(bias_frame),
        "n_residual_only_runs": len(residual_frame),
        "n_branch_intervention_rows": len(branch_frame),
        "stability_checks": int(len(stability)),
        "stability_passed": int(stability["pass"].sum()),
        "expansion_rule_triggered": True,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    print("\nPaired effects\n" + effects.to_string(index=False))


if __name__ == "__main__":
    main()
