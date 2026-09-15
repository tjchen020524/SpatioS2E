"""Check undefined gene-mean correlations in the section-centred workflow."""
from pathlib import Path
import subprocess
import sys


def test_section_centered_constant_means_and_aggregation(tmp_path):
    archive = Path(__file__).resolve().parents[1] / "manuscript_workflows"
    program = f"""
import os, sys
os.environ['SPATIOS2E_DATA_ROOT'] = {str(tmp_path)!r}
sys.path.insert(0, {str(archive)!r})
import numpy as np
import pandas as pd
from experiments.multicohort_geneheldout_decima import evaluate_section_centered as ev

variable = np.arange(4, dtype=float)
for predicted in (np.zeros(4), np.full(4, .1), .1 + 1e-9 * variable):
    assert np.isnan(ev.finite_corr(variable, predicted))
    assert np.isnan(ev.finite_corr(predicted, variable))
assert abs(ev.finite_corr(variable, variable) - 1) < 1e-12
assert ev.finite_corr(np.array([-1., 0., 1.]), np.array([1., -2., 1.])) == 0

true = np.arange(20, dtype=float).reshape(5, 4)
shared = np.broadcast_to(np.arange(5, dtype=float)[:, None], true.shape)
rows = []
for sample, noise in [('a', 0), ('b', 1e-9)]:
    pred = shared + noise * np.arange(4)[None, :]
    moments = ev.VectorMoments.zeros(4)
    moments.update(np.arange(4), pred, true)
    row = ev.section_metrics(sample, 'donor1', moments.derived())
    assert np.isnan(row['abundance_pcc'])
    assert np.isfinite(row['centered_full_matrix_pcc'])
    assert np.isfinite(row['mean_within_section_gene_pcc'])
    assert abs(row['total_mse'] - np.mean((pred - true)**2)) < 1e-12
    assert abs(row['decomposition_error']) < 1e-12
    rows.append(row)
individual = ev.aggregate_individuals(pd.DataFrame(rows))
assert individual['abundance_pcc'].isna().all()
assert np.isnan(0.8 - float(individual['abundance_pcc'].mean()))

# A spatially constant prediction for an observed-variable gene still scores zero.
moments = ev.VectorMoments.zeros(4)
moments.update(np.arange(4), np.broadcast_to(true.mean(0), true.shape), true)
row = ev.section_metrics('c', 'donor1', moments.derived())
assert row['mean_within_section_gene_pcc'] == 0
assert abs(row['abundance_pcc'] - 1) < 1e-12
"""
    subprocess.run([sys.executable, "-I", "-c", program], cwd=tmp_path,
                   check=True, capture_output=True, timeout=60)
