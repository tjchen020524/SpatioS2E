#!/usr/bin/env python3
"""Map the frozen HER2 audit panel to one released DeepSpot-M token source."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from safetensors import safe_open


ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "experiments/external_genequery_component_audit"
DEFAULT_PARENT_PANEL = EXP / "artifacts/panel/genequery_her2_panel.npz"
DEFAULT_MODEL_ROOT = ROOT / "DeepSpotM"
DEFAULT_OUTPUT = EXP / "artifacts/deepspotm/her2_panel_scgpt.npz"
SOURCES = ("evo2", "orthrus", "prott5", "scgpt", "apertus")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-panel", type=Path, default=DEFAULT_PARENT_PANEL)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--source", choices=SOURCES, default="scgpt")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    with np.load(args.parent_panel, allow_pickle=False) as data:
        parent = {key: data[key] for key in data.files}
    required = {"symbols", "gene_ids", "split_labels"}
    missing = required - set(parent)
    if missing:
        raise ValueError("Parent panel is missing fields: %s" % sorted(missing))

    token_frame = pd.read_csv(args.model_root / "tokens.csv")
    genes = token_frame.loc[token_frame["token_type"] == "gene"].copy()
    genes["token"] = genes["token"].astype(str)
    if len(genes) != 19338 or genes["token"].duplicated().any():
        raise ValueError("Unexpected DeepSpot-M gene vocabulary")
    expected_ids = np.arange(len(genes), dtype=np.int64)
    if not np.array_equal(genes["token_id"].to_numpy(dtype=np.int64), expected_ids):
        raise ValueError("DeepSpot-M gene token IDs are not contiguous in gene order")
    lookup = dict(zip(genes["token"], expected_ids.tolist()))

    symbols = parent["symbols"].astype(str)
    keep = np.asarray([symbol in lookup for symbol in symbols], dtype=bool)
    model_indices = np.asarray([lookup[symbol] for symbol in symbols[keep]], dtype=np.int64)
    weight_key = "gene_decoder.multi_source.bio_%s" % args.source
    with safe_open(args.model_root / "model.safetensors", framework="pt", device="cpu") as handle:
        if weight_key not in handle.keys():
            raise KeyError("Checkpoint does not contain %s" % weight_key)
        source_tokens = handle.get_tensor(weight_key).numpy()
    if source_tokens.shape[0] != len(genes):
        raise ValueError("Gene-token matrix and vocabulary have different row counts")
    embeddings = source_tokens[model_indices].astype(np.float32, copy=False)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        symbols=symbols[keep],
        gene_ids=parent["gene_ids"].astype(str)[keep],
        split_labels=parent["split_labels"].astype(str)[keep],
        model_indices=model_indices,
        semantic_embeddings=embeddings,
    )
    code_root = args.model_root / "code"
    try:
        code_commit = subprocess.check_output(
            ["git", "-C", str(code_root), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        code_commit = None
    provenance = {
        "method": "DeepSpot-M",
        "release_type": "released_full_checkpoint",
        "upstream_spatial_target_exposure": True,
        "source": args.source,
        "checkpoint_gene_token_key": weight_key,
        "checkpoint_gene_tokens_are_trained_parameters": True,
        "model_checkpoint": str(args.model_root / "model.safetensors"),
        "model_checkpoint_sha256": sha256(args.model_root / "model.safetensors"),
        "config_sha256": sha256(args.model_root / "config.json"),
        "tokens_sha256": sha256(args.model_root / "tokens.csv"),
        "official_code_commit": code_commit,
        "parent_panel": str(args.parent_panel),
        "n_parent_genes": int(len(symbols)),
        "n_intersection_genes": int(keep.sum()),
        "n_train_genes": int(np.sum(parent["split_labels"].astype(str)[keep] == "train")),
        "n_heldout_genes": int(np.sum(parent["split_labels"].astype(str)[keep] == "heldout")),
        "missing_symbols": symbols[~keep].tolist(),
        "interpretation": (
            "The full checkpoint and its per-gene source tokens were spatially trained. "
            "This is an upstream-exposed zero-shot audit, not reproduction of the paper's "
            "chromosome-held-out pretraining checkpoint."
        ),
    }
    provenance_path = args.output.with_suffix(".provenance.json")
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
