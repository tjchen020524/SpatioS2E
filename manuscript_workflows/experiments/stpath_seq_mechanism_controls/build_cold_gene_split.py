#!/usr/bin/env python3
"""Build train/heldout gene lists for cold-gene split (stratified by mean expression)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse


def load_config(path: Path) -> dict[str, Any]:
    if path.suffix in {".yml", ".yaml"}:
        import yaml  # type: ignore

        return yaml.safe_load(path.read_text())
    return json.loads(path.read_text())


def resolve_train_samples(cfg: dict[str, Any], split_key: str) -> list[str]:
    if split_key in cfg and isinstance(cfg[split_key], list):
        return list(cfg[split_key])
    if "split" in cfg and split_key in cfg["split"]:
        return list(cfg["split"][split_key])
    raise KeyError(f"Cannot find split '{split_key}' in {cfg.keys()}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split-config", type=Path, required=True)
    ap.add_argument("--expression-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--split-key", type=str, default="train")
    ap.add_argument("--split-name", type=str, default="train", help="which npz name to read under each sample")
    ap.add_argument("--holdout-frac", type=float, default=0.20)
    ap.add_argument("--n-bins", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = load_config(args.split_config)
    train_samples = resolve_train_samples(cfg, args.split_key)
    if not train_samples:
        raise SystemExit("No train samples found")

    rng = np.random.default_rng(args.seed)
    gene_ids_ref: list[str] | None = None
    gene_sum: np.ndarray | None = None
    total_spots = 0

    for sample in train_samples:
        npz_path = args.expression_root / sample / f"{args.split_name}.npz"
        if not npz_path.exists():
            raise FileNotFoundError(f"Missing expression file: {npz_path}")
        arr = np.load(npz_path)
        shape = tuple(arr["shape"].tolist())
        csr = sparse.csr_matrix((arr["data"], arr["indices"], arr["indptr"]), shape=shape)
        gene_ids = arr["gene_ids"].astype(str).tolist()

        if gene_ids_ref is None:
            gene_ids_ref = gene_ids
            gene_sum = np.asarray(csr.sum(axis=1)).reshape(-1).astype(np.float64)
        else:
            if gene_ids != gene_ids_ref:
                raise ValueError(f"Gene order mismatch for sample {sample}")
            gene_sum += np.asarray(csr.sum(axis=1)).reshape(-1)
        total_spots += int(shape[1])

    assert gene_ids_ref is not None and gene_sum is not None
    mean_expr = gene_sum / max(total_spots, 1)
    n_genes = len(gene_ids_ref)

    order = np.argsort(mean_expr)
    bins = np.array_split(order, max(args.n_bins, 1))
    heldout_idx: list[int] = []
    for b in bins:
        if b.size == 0:
            continue
        n_hold = max(1, int(round(b.size * args.holdout_frac)))
        pick = rng.choice(b, size=min(n_hold, b.size), replace=False)
        heldout_idx.extend(int(x) for x in pick.tolist())

    heldout_idx = sorted(set(heldout_idx))
    heldout_set = set(heldout_idx)
    train_idx = [i for i in range(n_genes) if i not in heldout_set]

    train_genes = [gene_ids_ref[i] for i in train_idx]
    heldout_genes = [gene_ids_ref[i] for i in heldout_idx]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "train_genes.txt").write_text("\n".join(train_genes) + "\n")
    (args.out_dir / "heldout_genes.txt").write_text("\n".join(heldout_genes) + "\n")

    summary = {
        "n_samples_train": len(train_samples),
        "n_genes_total": n_genes,
        "n_genes_train": len(train_genes),
        "n_genes_heldout": len(heldout_genes),
        "holdout_frac_target": args.holdout_frac,
        "holdout_frac_actual": len(heldout_genes) / max(n_genes, 1),
        "total_spots_train": total_spots,
        "mean_expr_train_genes": float(np.mean(mean_expr[train_idx])) if train_idx else float("nan"),
        "mean_expr_heldout_genes": float(np.mean(mean_expr[heldout_idx])) if heldout_idx else float("nan"),
    }
    (args.out_dir / "cold_gene_split_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
