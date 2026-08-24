import hashlib
import json
from pathlib import Path

import yaml


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
