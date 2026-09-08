"""Exercise the distributed external workflow without data, weights or GPUs."""

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "experiments/external_genequery_component_audit"


def run_script(name, *args):
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    return subprocess.run(
        [sys.executable, str(AUDIT / name), *map(str, args)],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=60,
    )


def make_runs(tmp_path):
    metrics = {
        "full_matrix_pcc": 0.2, "abundance_pcc": 0.4,
        "mean_gene_pcc": 0.01, "centered_full_matrix_pcc": 0.02,
        "section_centered_full_matrix_pcc": 0.02,
        "section_centered_mean_gene_pcc": 0.01, "section_centered_mse": 1.0,
    }
    for seed in (42, 123, 456):
        for variant in ("semantic", "identity_shuffle", "random", "constant"):
            path = tmp_path / f"seed_{seed}" / variant / "audit/summary.json"
            path.parent.mkdir(parents=True)
            # Semantic has higher error: a positive fraction is not an improvement.
            mean = 2.0 if variant == "semantic" else 1.0
            path.write_text(json.dumps(dict(metrics, overall_mse=mean+1, abundance_mse=mean, centered_mse=1.0)))


def test_three_seed_summary_and_signed_error(tmp_path):
    make_runs(tmp_path)
    result = run_script("summarize_audit.py", "--run-root", tmp_path, "--expected-seeds", 42, 123, 456)
    assert result.returncode == 0, result.stderr
    frame = pd.read_csv(tmp_path / "summary/component_attribution.tsv", sep="\t")
    assert len(frame) == 9
    assert (frame.mse_improvement == -1).all()
    assert (frame.gene_mean_fraction == 1).all()
    assert (frame.mse_change_direction == "increase").all()
    assert (frame.attribution_error.abs() < 1e-12).all()


def test_summary_rejects_missing_condition(tmp_path):
    make_runs(tmp_path)
    (tmp_path / "seed_123/random/audit/summary.json").unlink()
    result = run_script("summarize_audit.py", "--run-root", tmp_path)
    assert result.returncode != 0
    assert "Incomplete matched conditions" in result.stderr


def test_summary_rejects_missing_seed(tmp_path):
    make_runs(tmp_path)
    result = run_script("summarize_audit.py", "--run-root", tmp_path, "--expected-seeds", 42, 123)
    assert result.returncode != 0
    assert "Completed seeds" in result.stderr


def test_prediction_cli_constant_spatial_output(tmp_path):
    true = np.array([[0., 1.], [2., 3.], [1., 2.], [3., 4.]])
    path = tmp_path / "predictions.npz"
    np.savez(path, pred=np.broadcast_to(true.mean(0), true.shape), true=true,
             gene_ids=np.array(["g1", "g2"]), symbols=np.array(["G1", "G2"]),
             sample_ids=np.array(["G1", "G1", "H1", "H1"]),
             individual_ids=np.array(["G", "G", "H", "H"]))
    result = run_script("audit_predictions.py", path)
    assert result.returncode == 0, result.stderr
    row = json.loads((tmp_path / "audit/summary.json").read_text())
    assert row["mean_gene_pcc"] == 0
    assert row["section_centered_full_matrix_pcc"] == 0
    assert abs(row["overall_mse"]-row["abundance_mse"]-row["centered_mse"]) < 1e-12


def test_external_configuration_uses_release_split():
    result = subprocess.run(
        [sys.executable, "-c", "import common; print(common.CLEAN_SPLIT)"],
        cwd=AUDIT, capture_output=True, text=True, check=True,
    )
    split = Path(result.stdout.strip())
    assert split == ROOT / "configs/manuscript/gene_splits/her2st"
    assert (split / "train_genes.txt").is_file()
    assert (split / "heldout_genes.txt").is_file()
