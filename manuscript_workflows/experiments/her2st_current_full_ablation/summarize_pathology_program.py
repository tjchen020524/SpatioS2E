from workflow_paths import DATA_ROOT
#!/usr/bin/env python3
"""Collect held-out HER2ST pathology-program metrics across ablations."""

from pathlib import Path

import pandas as pd


ROOT = DATA_ROOT
EXP = ROOT / "experiments/her2st_current_full_ablation"


def main() -> None:
    frames = []
    for path in sorted((EXP / "variants").glob("*/pathology/pathology_program_metrics.csv")):
        frames.append(pd.read_csv(path))
    if len(frames) != 9:
        raise RuntimeError(f"Expected pathology outputs from 9 variants, found {len(frames)}")
    combined = pd.concat(frames, ignore_index=True)
    output = EXP / "results/pathology_program_summary.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output, index=False)
    pooled = combined[combined["sample"] == "pooled_G2_H1"].sort_values(
        "predicted_program_auc", ascending=False
    )
    pooled.to_csv(EXP / "results/pathology_program_pooled.csv", index=False)
    print(pooled.to_string(index=False))


if __name__ == "__main__":
    main()
