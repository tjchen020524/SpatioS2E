import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from scripts.validate_real_heldout_smoke import _validate_manifest_biological_split

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "manuscript"

EXPECTED_GENE_SPLITS = {
    "hippocampus_donor_disjoint": (
        14_717,
        3_680,
        "92e7cb74e03615cb33771ea195e411ceb02c7449f7df2b2104a31325afc03e13",
        "f3ff669018819b0d47afa921b252989510b986c8c443435e894f35cb11a69baa",
    ),
    "dlpfc": (
        14_717,
        3_680,
        "92e7cb74e03615cb33771ea195e411ceb02c7449f7df2b2104a31325afc03e13",
        "f3ff669018819b0d47afa921b252989510b986c8c443435e894f35cb11a69baa",
    ),
    "nac": (
        14_717,
        3_680,
        "748d9b486cae9f4962afe1c8cfccd184e2bd635d23aaeab873cc5581cf0a0476",
        "72b6f1429628cbaefcaaf880b9a9243680a2b330b9830dc30f3fc1c1daf7dab2",
    ),
    "her2st": (
        12_730,
        3_184,
        "bb961fa62b755ee1dca447ca645953cd6d623e9e4b83147f0bbd31215e6dbe7d",
        "413a0ca2cb25ca8fe5049712bd09ebcd39bdc346cb0248a2c7e71f5cef2c48f8",
    ),
}


def _lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_biological_split_counts_and_disjointness():
    cohorts = yaml.safe_load((CONFIG / "cohorts.yaml").read_text())["cohorts"]
    for cohort, record in cohorts.items():
        split = json.loads((CONFIG / record["biological_split"]).read_text())
        observed = {key: len(split[key]) for key in ("train", "val", "test")}
        expected = {
            "train": record["sections"]["train"],
            "val": record["sections"]["validation"],
            "test": record["sections"]["test"],
        }
        assert observed == expected, cohort
        assert len(set(split["train"]) & set(split["val"])) == 0
        assert len(set(split["train"]) & set(split["test"])) == 0
        assert len(set(split["val"]) & set(split["test"])) == 0


def test_real_smoke_manifest_is_checked_against_frozen_biological_split(tmp_path):
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps({"train": ["sample-a"], "val": ["sample-b"], "test": ["sample-c"]}))
    manifest = pd.DataFrame(
        {
            "sample": ["sample-a", "sample-b", "sample-c"],
            "split": ["train", "val", "test"],
            "barcode": ["a-1", "b-1", "c-1"],
        }
    )
    assert _validate_manifest_biological_split(manifest, split_path) == {
        "train": ["sample-a"],
        "val": ["sample-b"],
        "test": ["sample-c"],
    }


def test_real_smoke_manifest_rejects_stale_section_labels(tmp_path):
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps({"train": ["sample-a"], "val": ["sample-b"], "test": ["sample-c"]}))
    manifest = pd.DataFrame(
        {
            "sample": ["sample-a", "sample-b", "sample-c"],
            "split": ["test", "val", "train"],
            "barcode": ["a-1", "b-1", "c-1"],
        }
    )
    with pytest.raises(ValueError, match="violates the frozen biological split"):
        _validate_manifest_biological_split(manifest, split_path)


def test_primary_gene_split_counts_disjointness_and_checksums():
    for cohort, (n_train, n_heldout, train_sha, heldout_sha) in EXPECTED_GENE_SPLITS.items():
        directory = CONFIG / "gene_splits" / cohort
        train_path = directory / "train_genes.txt"
        heldout_path = directory / "heldout_genes.txt"
        train = _lines(train_path)
        heldout = _lines(heldout_path)
        assert len(train) == n_train
        assert len(heldout) == n_heldout
        assert len(set(train)) == n_train
        assert len(set(heldout)) == n_heldout
        assert set(train).isdisjoint(heldout)
        assert _sha256(train_path) == train_sha
        assert _sha256(heldout_path) == heldout_sha


def test_complete_gene_partition_manifest_matches_archived_files():
    manifest = json.loads((CONFIG / "gene_partition_manifest.json").read_text())
    expected_names = {
        "primary",
        "historical",
        "expression_seed123",
        "expression_seed456",
        "embedding_cluster",
        "chromosome_blocked",
    }
    assert set(manifest["partitions"]) == expected_names
    for partition_name, cohorts in manifest["partitions"].items():
        assert set(cohorts) == set(EXPECTED_GENE_SPLITS)
        for cohort, record in cohorts.items():
            observed_sets = {}
            for split in ("train", "heldout"):
                split_record = record[split]
                path = ROOT / split_record["path"]
                genes = _lines(path)
                observed_sets[split] = set(genes)
                assert len(genes) == split_record["n_genes"], (partition_name, cohort, split)
                assert len(genes) == len(observed_sets[split])
                assert _sha256(path) == split_record["sha256"]
            assert observed_sets["train"].isdisjoint(observed_sets["heldout"])
            assert sum(len(values) for values in observed_sets.values()) == record["universe_n_genes"]

    for cohort, overlap in manifest["primary_vs_historical_heldout_overlap"].items():
        primary = set(_lines(CONFIG / "gene_splits" / cohort / "heldout_genes.txt"))
        historical = set(
            _lines(
                CONFIG
                / "gene_splits_sensitivity"
                / "historical"
                / cohort
                / "heldout_genes.txt"
            )
        )
        assert len(primary & historical) == overlap["intersection_n_genes"]
        assert len(primary | historical) == overlap["union_n_genes"]
        assert len(primary & historical) / len(primary | historical) == overlap["jaccard"]


def test_heldout_assay_records_both_representation_contracts():
    assay = yaml.safe_load((CONFIG / "heldout_assay.yaml").read_text())
    representations = assay["gene_representations"]
    assert representations["decima"]["dimension"] == 1_920
    assert representations["scgpt_whole_human"]["dimension"] == 512
    assert representations["scgpt_whole_human"]["heldout_targets"] == {
        "hippocampus": 3_574,
        "dlpfc": 3_574,
        "nac": 3_574,
        "her2st": 3_179,
    }
    assert assay["evaluation"]["constant_vector_gene_mean_pcc"].startswith("undefined")


def test_external_model_hashes_are_frozen_without_local_paths():
    text = (CONFIG / "external_models.yaml").read_text()
    models = yaml.safe_load(text)
    assert models["Decima"]["sha256"] == (
        "9b4efc2967d09d05c34ced1877744ad1d499d3899863463d5107d072046bfb31"
    )
    assert models["scGPT_whole_human"]["checkpoint_sha256"] == (
        "6cb5d451ab5c4b33eb673adbe4fddc61d2389df1b89b7651a9fe2e557572b922"
    )
    assert "/dcs04/" not in text
    assert "/users/" not in text
    assert "/scratch/" not in text
