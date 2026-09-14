#!/usr/bin/env python3
"""Clone one cohort's matched ablation configs for multiple random seeds."""

from __future__ import annotations
from workflow_paths import DATA_ROOT

import argparse
import json
from pathlib import Path

import yaml


ROOT = DATA_ROOT


def resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    args = parser.parse_args()

    cohort_dir = resolve(args.cohort_dir)
    base_manifest = json.loads((cohort_dir / "configs/variant_manifest.json").read_text())
    cohort_relative = cohort_dir.relative_to(ROOT)
    generated = []
    for seed in args.seeds:
        seed_config_dir = cohort_dir / "configs/multiseed" / f"seed_{seed}"
        seed_config_dir.mkdir(parents=True, exist_ok=True)
        for entry in base_manifest:
            name = entry["name"]
            cfg = yaml.safe_load(Path(entry["config"]).read_text())
            cfg["seed"] = int(seed)
            cfg["run_name"] = f"{cfg['run_name']}_seed{seed}"
            run_root = cohort_relative / "multiseed" / f"seed_{seed}" / "variants" / name
            cfg["paths"]["checkpoint_dir"] = str(run_root / "checkpoints")
            cfg["paths"]["checkpoint_name"] = "best.pt"
            config_path = seed_config_dir / f"{name}.yaml"
            config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
            generated.append(
                {
                    "seed": int(seed),
                    "name": name,
                    "config": str(config_path.relative_to(ROOT)),
                    "checkpoint": str(run_root / "checkpoints/best.pt"),
                    "results": str(run_root / "results"),
                    "no_graph": bool(entry["no_graph"]),
                    "spatial_pcc_applicable": bool(
                        entry.get("spatial_pcc_applicable", True)
                    ),
                }
            )

    output = cohort_dir / "configs/multiseed_manifest.json"
    output.write_text(
        json.dumps(
            {
                "cohort_dir": str(cohort_relative),
                "seeds": [int(seed) for seed in args.seeds],
                "runs": generated,
            },
            indent=2,
        )
    )
    print(json.dumps({"manifest": str(output), "n_runs": len(generated)}, indent=2))


if __name__ == "__main__":
    main()
