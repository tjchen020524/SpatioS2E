import math

import numpy as np

from spatios2e.evaluation.evaluate import RunningStats, correlation_with_thresholds


def test_constant_prediction_counts_as_zero_for_eligible_gene():
    stats = RunningStats()
    stats.update(np.asarray([2.0, 2.0, 2.0]), np.asarray([1.0, 2.0, 4.0]))

    primary, finite, eligible, pred_variable = correlation_with_thresholds(
        stats,
        true_std_threshold=1.0e-6,
        pred_std_threshold=1.0e-6,
    )

    assert eligible
    assert not pred_variable
    assert primary == 0.0
    assert math.isnan(finite)


def test_low_variance_true_gene_is_not_eligible():
    stats = RunningStats()
    stats.update(np.asarray([1.0, 2.0, 3.0]), np.asarray([4.0, 4.0, 4.0]))

    primary, finite, eligible, _ = correlation_with_thresholds(
        stats,
        true_std_threshold=1.0e-6,
        pred_std_threshold=1.0e-6,
    )

    assert not eligible
    assert math.isnan(primary)
    assert math.isnan(finite)
