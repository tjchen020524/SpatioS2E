#!/usr/bin/env python3
"""Validate a raw-expression decoder pilot without using effect direction."""

from __future__ import annotations
from workflow_paths import DATA_ROOT

import argparse
import json
import math
from pathlib import Path


ROOT = DATA_ROOT
AUDIT = ROOT / "experiments/heldout_decoder_architecture_audit"
COHORTS = ("dlpfc", "her2st")
VARIANTS = ("decima", "random", "constant")
ARCHITECTURES = ("bias_free", "concat_mlp")
METRICS = (
    "pooled_pcc",
    "overall_mse",
    "abundance_pcc",
    "abundance_rmse",
    "mean_gene_pcc",
    "centered_rmse",
)


def load(path: Path) -> object:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("architecture", choices=ARCHITECTURES)
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for cohort in COHORTS:
        for variant in VARIANTS:
            run_root = AUDIT / args.architecture / "runs" / cohort / "seed_42" / variant / variant
            history = load(run_root / "results/history.json")
            component = load(
                AUDIT
                / args.architecture
                / "components/runs"
                / cohort
                / "seed_42"
                / variant
                / "summary.json"
            )
            assert isinstance(history, list)
            finite_history = len(history) == 6 and all(
                math.isfinite(float(row["train_loss"])) and math.isfinite(float(row["score"]))
                for row in history
            )
            finite_metrics = all(math.isfinite(float(component[key])) for key in METRICS)
            checkpoint_exists = (run_root / "checkpoints/best.pt").exists()
            rows.append(
                {
                    "architecture": args.architecture,
                    "cohort": cohort,
                    "seed": 42,
                    "variant": variant,
                    "n_epochs": len(history),
                    "finite_history": finite_history,
                    "finite_component_metrics": finite_metrics,
                    "checkpoint_exists": checkpoint_exists,
                    "pass": finite_history and finite_metrics and checkpoint_exists,
                }
            )

    passed = all(bool(row["pass"]) for row in rows)
    output = AUDIT / "pilot_gate"
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "architecture": args.architecture,
        "status": "PASS" if passed else "FAIL",
        "criterion": "training/validation stability, selected checkpoint and finite exported endpoints; no effect-direction gate",
        "checks": rows,
    }
    (output / f"{args.architecture}.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
