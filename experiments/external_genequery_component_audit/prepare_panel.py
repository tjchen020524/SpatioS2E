#!/usr/bin/env python3
"""Align the official GeneQuery HER2 embedding to the frozen clean gene split."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import (
    CLEAN_SPLIT,
    HER2,
    EXP,
    GENE_META_PATH,
    OFFICIAL_COMMIT,
    OFFICIAL_EMBEDDING_SHA256,
    PANEL_ARTIFACT,
    PANEL_PATH,
    sha256,
)


OFFICIAL_PUBLISHED_PREFIX = [
    "ISG15", "TNFRSF4", "SCNN1D", "DVL1", "AURKAIP1", "MIB2", "TNFRSF14",
    "FAM213B", "KCNAB2", "PHF13", "SPSB1", "PEX14", "EXOSC10", "MIIP",
    "VPS13D", "EFHD2", "CROCC", "SDHB", "ARHGEF10L", "C1QB", "TCEA3",
    "RSRP1", "STMN1", "CD52", "MAP3K6", "LCK", "ZNF362", "C1orf216",
    "COL8A2", "EVA1B",
]


def read_gene_set(path: Path) -> set:
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--official-embedding",
        type=Path,
        required=True,
        help="Official gene-aware/src/her2_gpt_description_emb.npy from the pinned commit.",
    )
    parser.add_argument("--output", type=Path, default=PANEL_ARTIFACT)
    args = parser.parse_args()

    observed_sha = sha256(args.official_embedding)
    if observed_sha != OFFICIAL_EMBEDDING_SHA256:
        raise ValueError(
            "Official embedding SHA256 mismatch: expected %s, found %s"
            % (OFFICIAL_EMBEDDING_SHA256, observed_sha)
        )
    embeddings = np.load(args.official_embedding, allow_pickle=False)
    panel_symbols = np.load(PANEL_PATH, allow_pickle=False).astype(str)
    count_path = (
        HER2 / "data/source/counts/count-matrices/A1.tsv.gz"
    )
    with gzip.open(count_path, "rt") as handle:
        count_header = handle.readline().rstrip("\n").split("\t")[1:]
    panel_set = set(panel_symbols.tolist())
    symbols = np.asarray([symbol for symbol in count_header if symbol in panel_set])
    if panel_symbols.shape != (785,) or symbols.shape != (785,) or embeddings.shape != (785, 768):
        raise ValueError(
            "Expected 785 local symbols, 785 count-ordered symbols and a (785,768) vector; "
            "got %s, %s and %s" % (panel_symbols.shape, symbols.shape, embeddings.shape)
        )
    if symbols[: len(OFFICIAL_PUBLISHED_PREFIX)].tolist() != OFFICIAL_PUBLISHED_PREFIX:
        raise ValueError("Count-derived order does not match the 30-gene official published prefix")

    meta = pd.read_csv(GENE_META_PATH, sep="\t")
    symbol_column = meta.columns[0]
    pairs = meta[[symbol_column, "gene_id"]].dropna().copy()
    pairs[symbol_column] = pairs[symbol_column].astype(str)
    pairs["gene_id"] = pairs["gene_id"].astype(str).str.split(".").str[0]
    duplicated = set(pairs.loc[pairs[symbol_column].duplicated(False), symbol_column].tolist())
    unambiguous = pairs.loc[~pairs[symbol_column].isin(duplicated)].set_index(symbol_column)["gene_id"]
    symbol_to_gene = unambiguous.to_dict()

    train_genes = read_gene_set(CLEAN_SPLIT / "train_genes.txt")
    heldout_genes = read_gene_set(CLEAN_SPLIT / "heldout_genes.txt")
    rows = []
    unmapped = []
    for panel_index, symbol in enumerate(symbols.tolist()):
        gene_id = symbol_to_gene.get(symbol)
        if gene_id is None:
            unmapped.append(symbol)
            continue
        if gene_id in train_genes:
            label = "train"
        elif gene_id in heldout_genes:
            label = "heldout"
        else:
            continue
        rows.append((panel_index, symbol, gene_id, label))

    panel_indices = np.asarray([row[0] for row in rows], dtype=np.int64)
    output_symbols = np.asarray([row[1] for row in rows])
    gene_ids = np.asarray([row[2] for row in rows])
    split_labels = np.asarray([row[3] for row in rows])
    if len(set(gene_ids.tolist())) != len(gene_ids):
        raise ValueError("The mapped GeneQuery panel contains duplicate Ensembl IDs")
    counts = {label: int(np.sum(split_labels == label)) for label in ("train", "heldout")}
    expected = {"train": 613, "heldout": 138}
    if counts != expected or len(gene_ids) != 751:
        raise ValueError("Unexpected frozen-panel intersection: %s (expected %s)" % (counts, expected))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        symbols=output_symbols,
        gene_ids=gene_ids,
        split_labels=split_labels,
        official_panel_indices=panel_indices,
        semantic_embeddings=embeddings[panel_indices].astype(np.float32),
    )
    table = pd.DataFrame(
        {
            "official_panel_index": panel_indices,
            "symbol": output_symbols,
            "gene_id": gene_ids,
            "split": split_labels,
        }
    )
    table.to_csv(args.output.with_suffix(".tsv"), sep="\t", index=False)
    provenance = {
        "official_repository": "https://github.com/xy-always/GeneQuery",
        "official_commit": OFFICIAL_COMMIT,
        "official_embedding_relative_path": "gene-aware/src/her2_gpt_description_emb.npy",
        "official_embedding_sha256": observed_sha,
        "local_original_panel": str(PANEL_PATH.relative_to(EXP.parents[1])),
        "row_order_source": str(count_path.relative_to(EXP.parents[1])),
        "row_order_validation": (
            "30/30 official description rows match the count-header-filtered panel prefix; "
            "Bio_ClinicalBERT recomputation matched embedding rows 0-29 at cosine 1.0"
        ),
        "mapped_panel_size": int(len(gene_ids)),
        "train_genes": counts["train"],
        "heldout_genes": counts["heldout"],
        "unmapped_or_ambiguous_symbols": unmapped,
    }
    args.output.with_suffix(".provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
