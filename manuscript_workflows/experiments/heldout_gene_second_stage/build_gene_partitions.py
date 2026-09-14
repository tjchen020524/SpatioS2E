#!/usr/bin/env python3
"""Build additional training-only and embedding-cluster-disjoint gene splits."""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from experiments.multicohort_geneheldout_decima_clean_split.clean_split_config import (
    CLEAN_ROOT,
    COHORTS,
    ROOT,
    SOURCE_ROOT,
)


OUT = ROOT / "experiments/heldout_gene_second_stage/gene_partitions"
PARTITIONS = (
    "expression_seed123",
    "expression_seed456",
    "embedding_cluster",
    "chromosome_blocked",
)
DECIMA_CACHE = (
    ROOT
    / "experiments/sample_split_fullgenes_v12_uni2h_direct_seq_residual_no_celltype/artifacts/decima_gene_embeddings_fullgenes.npz"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def expression_assignment(stratification: pd.DataFrame, seed: int) -> np.ndarray:
    heldout: list[int] = []
    rng = np.random.default_rng(seed)
    for expression_bin in sorted(stratification["expression_bin"].unique()):
        members = np.flatnonzero(
            stratification["expression_bin"].to_numpy(dtype=int) == expression_bin
        )
        n_holdout = max(1, int(round(len(members) * 0.20)))
        heldout.extend(
            rng.choice(members, size=n_holdout, replace=False).astype(int).tolist()
        )
    assignment = np.full(len(stratification), "train", dtype=object)
    assignment[np.asarray(sorted(heldout), dtype=np.int64)] = "heldout"
    return assignment


def closest_cluster_subset(cluster_sizes: np.ndarray, target: int) -> set[int]:
    # Exact small dynamic program over the within-bin clusters.
    states: dict[int, tuple[int, ...]] = {0: ()}
    for cluster, size in enumerate(cluster_sizes.tolist()):
        additions = {
            total + int(size): selected + (cluster,)
            for total, selected in list(states.items())
        }
        for total, selected in additions.items():
            states.setdefault(total, selected)
    positive = [(total, selected) for total, selected in states.items() if selected]
    best_total, best_selected = min(
        positive,
        key=lambda item: (abs(item[0] - target), len(item[1]), item[0]),
    )
    if abs(best_total - target) > max(10, int(round(target * 0.05))):
        raise ValueError(f"Could not match cluster holdout target {target}: {best_total}")
    return set(best_selected)


def embedding_cluster_assignment(
    stratification: pd.DataFrame, embedding: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    standardized = StandardScaler().fit_transform(embedding).astype(np.float32)
    pca = PCA(n_components=50, svd_solver="randomized", random_state=42)
    reduced = pca.fit_transform(standardized).astype(np.float32)
    assignment = np.full(len(stratification), "train", dtype=object)
    cluster_id = np.full(len(stratification), -1, dtype=np.int64)
    global_cluster = 0
    expression_bins = stratification["expression_bin"].to_numpy(dtype=int)
    for expression_bin in sorted(np.unique(expression_bins)):
        members = np.flatnonzero(expression_bins == expression_bin)
        n_clusters = 10
        model = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=42 + int(expression_bin),
            batch_size=512,
            n_init=10,
            max_iter=300,
            reassignment_ratio=0.0,
        )
        labels = model.fit_predict(reduced[members])
        sizes = np.bincount(labels, minlength=n_clusters)
        selected = closest_cluster_subset(sizes, int(round(len(members) * 0.20)))
        for local_cluster in range(n_clusters):
            cluster_members = members[labels == local_cluster]
            cluster_id[cluster_members] = global_cluster
            if local_cluster in selected:
                assignment[cluster_members] = "heldout"
            global_cluster += 1
    if np.any(cluster_id < 0):
        raise AssertionError("Unassigned embedding cluster")
    return assignment, cluster_id, reduced, float(pca.explained_variance_ratio_.sum())


def nearest_similarity(reduced: np.ndarray, assignment: np.ndarray) -> np.ndarray:
    train = reduced[assignment == "train"]
    heldout = reduced[assignment == "heldout"]
    train = train / np.maximum(np.linalg.norm(train, axis=1, keepdims=True), 1.0e-8)
    heldout = heldout / np.maximum(np.linalg.norm(heldout, axis=1, keepdims=True), 1.0e-8)
    model = NearestNeighbors(n_neighbors=1, metric="cosine", n_jobs=-1).fit(train)
    distance, _ = model.kneighbors(heldout)
    return 1.0 - distance[:, 0]


def chromosome_blocked_assignment(
    stratification: pd.DataFrame, gene_metadata: pd.DataFrame
) -> tuple[np.ndarray, list[str], float]:
    metadata = gene_metadata.copy()
    metadata["gene_id"] = metadata["gene_id"].astype(str).str.split(".").str[0]
    metadata = metadata.drop_duplicates("gene_id", keep="first").set_index("gene_id")
    genes = stratification["gene_id"].astype(str).tolist()
    chromosome = metadata.loc[genes, "chrom"].fillna("unknown").astype(str).to_numpy()
    chromosomes = sorted(set(chromosome) - {"unknown", "chrM"})
    target = int(round(len(genes) * 0.20))
    expression_bins = stratification["expression_bin"].to_numpy(dtype=int)
    global_bin_fraction = np.bincount(expression_bins, minlength=11)[1:] / len(genes)
    candidates: list[tuple[float, int, tuple[str, ...], int]] = []
    for n_blocks in range(1, min(5, len(chromosomes) + 1)):
        for selected in itertools.combinations(chromosomes, n_blocks):
            mask = np.isin(chromosome, selected)
            n_genes = int(mask.sum())
            if n_genes < int(target * 0.70) or n_genes > int(target * 1.30):
                continue
            bin_fraction = np.bincount(expression_bins[mask], minlength=11)[1:] / n_genes
            distribution_distance = float(np.abs(bin_fraction - global_bin_fraction).sum())
            size_distance = abs(n_genes - target) / target
            score = size_distance + 0.35 * distribution_distance
            candidates.append((score, n_blocks, selected, n_genes))
    if not candidates:
        raise ValueError("No chromosome-blocked subset approached the 20% target")
    _, _, selected, n_selected = min(candidates, key=lambda item: (item[0], item[1], item[2]))
    assignment = np.full(len(genes), "train", dtype=object)
    assignment[np.isin(chromosome, selected)] = "heldout"
    return assignment, list(selected), n_selected / len(genes)


def write_partition(
    name: str,
    stratification: pd.DataFrame,
    assignment: np.ndarray,
    cluster_id: np.ndarray | None,
) -> list[dict[str, object]]:
    root = OUT / name
    root.mkdir(parents=True, exist_ok=True)
    table = stratification.copy()
    table["assignment"] = assignment
    if cluster_id is not None:
        table["embedding_cluster"] = cluster_id
    table.to_csv(
        root / "master_assignment.tsv.gz",
        sep="\t",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    gene_ids = table["gene_id"].astype(str).tolist()
    master_train = {
        gene for gene, state in zip(gene_ids, assignment.tolist()) if state == "train"
    }
    master_heldout = set(gene_ids) - master_train
    rows: list[dict[str, object]] = []
    for cohort in COHORTS:
        cohort_genes = read_lines(SOURCE_ROOT / "data" / cohort / "gene_ids.txt")
        train_genes = [gene for gene in cohort_genes if gene in master_train]
        heldout_genes = [gene for gene in cohort_genes if gene in master_heldout]
        if len(train_genes) + len(heldout_genes) != len(cohort_genes):
            raise ValueError(f"Partition {name} does not cover cohort {cohort}")
        split_dir = root / "data" / cohort / "gene_splits"
        split_dir.mkdir(parents=True, exist_ok=True)
        (split_dir / "train_genes.txt").write_text("\n".join(train_genes) + "\n")
        (split_dir / "heldout_genes.txt").write_text("\n".join(heldout_genes) + "\n")
        rows.append(
            {
                "partition": name,
                "cohort": cohort,
                "n_genes": len(cohort_genes),
                "n_train_genes": len(train_genes),
                "n_heldout_genes": len(heldout_genes),
                "train_sha256": sha256(split_dir / "train_genes.txt"),
                "heldout_sha256": sha256(split_dir / "heldout_genes.txt"),
            }
        )
    pd.DataFrame(rows).to_csv(root / "cohort_partitions.tsv", sep="\t", index=False)
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    stratification = pd.read_csv(
        CLEAN_ROOT / "partition/master/stratification.tsv.gz", sep="\t"
    )
    cache = np.load(DECIMA_CACHE, allow_pickle=False)
    cache_ids = cache["gene_ids"].astype(str).tolist()
    cache_lookup = {gene: index for index, gene in enumerate(cache_ids)}
    genes = stratification["gene_id"].astype(str).tolist()
    missing = [gene for gene in genes if gene not in cache_lookup]
    if missing:
        raise ValueError(f"Missing {len(missing)} master genes from Decima cache")
    embedding = cache["embeddings"].astype(np.float32, copy=False)[
        np.asarray([cache_lookup[gene] for gene in genes], dtype=np.int64)
    ]

    assignments = {
        "expression_seed123": expression_assignment(stratification, 123),
        "expression_seed456": expression_assignment(stratification, 456),
    }
    cluster_assignment, cluster_id, reduced, pca_variance = embedding_cluster_assignment(
        stratification, embedding
    )
    assignments["embedding_cluster"] = cluster_assignment
    gene_metadata = pd.read_csv(ROOT / "data/decima_input/metadata/gene_meta.tsv", sep="\t")
    chromosome_assignment, heldout_chromosomes, chromosome_fraction = (
        chromosome_blocked_assignment(stratification, gene_metadata)
    )
    assignments["chromosome_blocked"] = chromosome_assignment

    primary_assignment = stratification["assignment"].astype(str).to_numpy()
    similarity_rows: list[dict[str, object]] = []
    similarity_gene_rows: list[dict[str, object]] = []
    for name, assignment in (
        ("primary_expression_seed42", primary_assignment),
        *assignments.items(),
    ):
        assignment_array = np.asarray(assignment)
        heldout_index = np.flatnonzero(assignment_array == "heldout")
        similarity = nearest_similarity(reduced, assignment_array)
        similarity_rows.append(
            {
                "partition": name,
                "n_heldout": int(np.sum(np.asarray(assignment) == "heldout")),
                "nearest_training_cosine_mean": float(np.mean(similarity)),
                "nearest_training_cosine_median": float(np.median(similarity)),
                "nearest_training_cosine_q05": float(np.quantile(similarity, 0.05)),
                "nearest_training_cosine_q95": float(np.quantile(similarity, 0.95)),
            }
        )
        similarity_gene_rows.extend(
            {
                "partition": name,
                "gene_id": genes[int(index)],
                "nearest_training_cosine": float(value),
            }
            for index, value in zip(heldout_index.tolist(), similarity.tolist())
        )
    similarity_table = pd.DataFrame(similarity_rows)
    similarity_table.to_csv(OUT / "nearest_training_similarity.tsv", sep="\t", index=False)
    pd.DataFrame(similarity_gene_rows).to_csv(
        OUT / "nearest_training_similarity_per_gene.tsv.gz",
        sep="\t",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )

    cohort_rows: list[dict[str, object]] = []
    for name, assignment in assignments.items():
        clusters = cluster_id if name == "embedding_cluster" else None
        cohort_rows.extend(write_partition(name, stratification, assignment, clusters))
    cohort_table = pd.DataFrame(cohort_rows)

    overlap_rows: list[dict[str, object]] = []
    all_assignments = {
        "primary_expression_seed42": primary_assignment,
        **assignments,
    }
    for left_name, left in all_assignments.items():
        left_set = set(np.flatnonzero(np.asarray(left) == "heldout").tolist())
        for right_name, right in all_assignments.items():
            right_set = set(np.flatnonzero(np.asarray(right) == "heldout").tolist())
            union = left_set | right_set
            overlap_rows.append(
                {
                    "partition_a": left_name,
                    "partition_b": right_name,
                    "n_overlap": len(left_set & right_set),
                    "jaccard": len(left_set & right_set) / len(union),
                }
            )
    pd.DataFrame(overlap_rows).to_csv(OUT / "partition_overlap.tsv", sep="\t", index=False)

    cluster_check = pd.DataFrame(
        {
            "cluster": cluster_id,
            "assignment": cluster_assignment,
        }
    ).groupby("cluster")["assignment"].nunique()
    manifest = {
        "status": "PASS",
        "partitions": list(PARTITIONS),
        "source_expression_stratification": str(
            (CLEAN_ROOT / "partition/master/stratification.tsv.gz").relative_to(ROOT)
        ),
        "source_expression_scope": "final hippocampus training individuals only",
        "embedding_cluster_design": "50-PC Decima space; ten coarse clusters within each of 10 expression strata; entire clusters assigned to one side",
        "chromosome_blocked_design": {
            "heldout_chromosomes": heldout_chromosomes,
            "heldout_fraction": chromosome_fraction,
            "selection": "one to four whole chromosomes chosen using gene-count and training-expression-bin balance only",
        },
        "pca_explained_variance_fraction": pca_variance,
        "n_gene_similarity_rows": len(similarity_gene_rows),
        "checks": {
            "all_partitions_have_four_cohorts": bool(
                cohort_table.groupby("partition")["cohort"].nunique().eq(4).all()
            ),
            "all_genes_covered": bool(
                (cohort_table["n_train_genes"] + cohort_table["n_heldout_genes"])
                .eq(cohort_table["n_genes"])
                .all()
            ),
            "embedding_clusters_do_not_cross_split": bool(cluster_check.max() == 1),
        },
        "decima_cache": str(DECIMA_CACHE.relative_to(ROOT)),
        "decima_cache_sha256": sha256(DECIMA_CACHE),
        "outputs": [
            "nearest_training_similarity.tsv",
            "nearest_training_similarity_per_gene.tsv.gz",
            "partition_overlap.tsv",
            *[f"{name}/master_assignment.tsv.gz" for name in PARTITIONS],
            *[f"{name}/cohort_partitions.tsv" for name in PARTITIONS],
        ],
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    print("\nCohort partitions")
    print(cohort_table.to_string(index=False))
    print("\nNearest training similarity")
    print(similarity_table.to_string(index=False))


if __name__ == "__main__":
    main()
