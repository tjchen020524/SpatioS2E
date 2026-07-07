#!/usr/bin/env python
"""
Utility to map Visium spatial transcriptomics genes to the Decima hg38 gene set.

The script reads all `*_features.tsv.gz` files under the Visium raw data directory,
collects unique genes, and intersects them with the genes present in the Decima
metadata AnnData file. Outputs include the intersection as well as genes that are
exclusive to each modality.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import logging
import os
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple


LOGGER = logging.getLogger("spatios2e.map_genes")


def configure_logging(verbose: bool = False) -> None:
    """Configure basic logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(message)s")


def find_gene_id_list(
    explicit: Optional[str], reference_dir: Path, env_var: str = "DECIMA_GENE_IDS"
) -> Path:
    """Resolve the Decima gene ID list location."""
    candidates: Iterable[Optional[str]] = (
        explicit,
        os.environ.get(env_var),
        str(reference_dir / "decima_gene_ids.csv"),
    )
    for candidate in candidates:
        if not candidate:
            continue
        candidate_path = Path(candidate).expanduser()
        if candidate_path.exists():
            LOGGER.info("Using Decima gene list at %s", candidate_path)
            return candidate_path

    raise FileNotFoundError(
        "Could not locate Decima gene IDs. Provide --decima-gene-ids, set DECIMA_GENE_IDS, "
        "or place decima_gene_ids.csv under data/reference."
    )


def load_st_genes(raw_dir: Path, feature_type: str = "Gene Expression") -> Dict[str, str]:
    """Collect unique Visium gene IDs -> gene name."""
    feature_files = sorted(raw_dir.glob("**/*_features.tsv.gz"))
    if not feature_files:
        raise FileNotFoundError(f"No *_features.tsv.gz files found under {raw_dir}.")

    LOGGER.info("Found %d Visium feature files.", len(feature_files))
    gene_map: Dict[str, str] = {}
    duplicates = 0

    for path in feature_files:
        with gzip.open(path, "rt") as handle:
            reader = csv.reader(handle, delimiter="\t")
            for row in reader:
                if len(row) < 3:
                    continue
                gene_id, gene_name, ft = row[0].strip(), row[1].strip(), row[2].strip()
                if ft != feature_type:
                    continue
                if gene_id not in gene_map:
                    gene_map[gene_id] = gene_name
                elif gene_map[gene_id] != gene_name:
                    duplicates += 1

    if duplicates:
        LOGGER.warning(
            "Detected %d gene_id entries with conflicting names; keeping the first encountered mapping.",
            duplicates,
        )

    LOGGER.info("Unique Visium genes: %d", len(gene_map))
    return gene_map


def load_decima_gene_ids(path: Path) -> Dict[str, str]:
    """Load Decima gene identifiers from a CSV/TSV/plain-text list."""
    LOGGER.info("Loading Decima gene identifiers.")
    gene_map: Dict[str, str] = {}
    with path.open("r", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row:
                continue
            entries = [item.strip() for item in row if item.strip()]
            if not entries:
                continue
            # Prefer Ensembl-like entries
            gene_id, gene_name = _extract_gene_fields(entries)
            if not gene_id:
                continue
            if gene_id not in gene_map:
                gene_map[gene_id] = gene_name or gene_id
    LOGGER.info("Decima genes available: %d", len(gene_map))
    return gene_map


def _extract_gene_fields(entries: Iterable[str]) -> Tuple[Optional[str], Optional[str]]:
    """Infer gene_id and optional gene_name from a list of tokens in a CSV row."""
    gene_id = None
    for token in entries:
        if token.upper().startswith(("ENS", "ENSG")):
            gene_id = token
            break
    if gene_id is None:
        first = next(iter(entries))
        if first.upper().startswith(("ENS", "ENSG")):
            gene_id = first
    return gene_id, None


def write_outputs(
    intersection: Dict[str, Tuple[str, str]],
    st_only: Dict[str, str],
    decima_only: Dict[str, str],
    output_dir: Path,
) -> None:
    """Persist intersection and exclusives to TSV/JSON files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    intersection_path = output_dir / "gene_intersection.tsv"
    st_only_path = output_dir / "genes_st_only.tsv"
    decima_only_path = output_dir / "genes_decima_only.tsv"
    summary_path = output_dir / "gene_mapping_summary.json"

    with intersection_path.open("w", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(["gene_id", "st_gene_name", "decima_gene_name"])
        for gene_id in sorted(intersection):
            st_name, decima_name = intersection[gene_id]
            writer.writerow([gene_id, st_name, decima_name])

    with st_only_path.open("w", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(["gene_id", "st_gene_name"])
        for gene_id in sorted(st_only):
            writer.writerow([gene_id, st_only[gene_id]])

    with decima_only_path.open("w", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(["gene_id", "decima_gene_name"])
        for gene_id in sorted(decima_only):
            writer.writerow([gene_id, decima_only[gene_id]])

    intersection_count = len(intersection)
    st_total = intersection_count + len(st_only)
    decima_total = intersection_count + len(decima_only)

    summary = {
        "n_visium_genes": st_total,
        "n_decima_genes": decima_total,
        "n_intersection": intersection_count,
        "fraction_visium_covered": float(intersection_count / st_total) if st_total else 0.0,
        "fraction_decima_covered": float(intersection_count / decima_total) if decima_total else 0.0,
        "intersection_path": str(intersection_path),
        "visium_only_path": str(st_only_path),
        "decima_only_path": str(decima_only_path),
    }

    with summary_path.open("w") as fh:
        json.dump(summary, fh, indent=2)

    LOGGER.info("Intersection written to %s", intersection_path)
    LOGGER.info("Summary written to %s", summary_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory containing Visium donor subdirectories with *_features.tsv.gz files.",
    )
    parser.add_argument(
        "--decima-gene-ids",
        type=str,
        default=None,
        help="Path to Decima gene identifiers (CSV/TSV/plain-text). "
        "Overrides DECIMA_GENE_IDS env var and data/reference/decima_gene_ids.csv fallback.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory to store mapping outputs.",
    )
    parser.add_argument(
        "--feature-type",
        type=str,
        default="Gene Expression",
        help="Feature type to keep from Visium features.tsv files.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    args = parser.parse_args()

    configure_logging(args.verbose)

    reference_dir = args.raw_dir.parent / "reference"
    gene_list_path = find_gene_id_list(args.decima_gene_ids, reference_dir)

    st_genes = load_st_genes(args.raw_dir, feature_type=args.feature_type)
    decima_genes = load_decima_gene_ids(gene_list_path)

    intersection = {}
    for gene_id, st_name in st_genes.items():
        if gene_id in decima_genes:
            intersection[gene_id] = (st_name, decima_genes[gene_id])

    st_only = {gid: name for gid, name in st_genes.items() if gid not in intersection}
    decima_only = {gid: name for gid, name in decima_genes.items() if gid not in intersection}

    LOGGER.info(
        "Intersection size: %d genes (%.2f%% of Visium, %.2f%% of Decima).",
        len(intersection),
        100.0 * len(intersection) / max(len(intersection) + len(st_only), 1),
        100.0 * len(intersection) / max(len(intersection) + len(decima_only), 1),
    )

    write_outputs(intersection, st_only, decima_only, args.output_dir)


if __name__ == "__main__":
    main()
