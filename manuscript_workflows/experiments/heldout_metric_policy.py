"""Shared metric policy for held-out-gene evaluations.

Gene eligibility is determined only from observed expression.  Every eligible
gene remains in the denominator for every model condition; a spatially
constant prediction receives gene-wise PCC zero rather than being excluded.
The same observed-eligible set is used for the gene-centred full-matrix PCC.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


OBSERVED_STD_MIN = 1.0e-6
PREDICTED_STD_MIN = 1.0e-6


@dataclass(frozen=True)
class GeneCorrelationPolicyResult:
    correlation: np.ndarray
    eligible: np.ndarray
    prediction_variable: np.ndarray
    observed_std: np.ndarray
    predicted_std: np.ndarray
    centered_pred_ss: np.ndarray
    centered_true_ss: np.ndarray
    centered_cross: np.ndarray

    @property
    def n_eligible(self) -> int:
        return int(self.eligible.sum())

    @property
    def n_constant_prediction(self) -> int:
        return int((self.eligible & ~self.prediction_variable).sum())

    @property
    def mean(self) -> float:
        values = self.correlation[self.eligible]
        return float(values.mean()) if values.size else float("nan")

    @property
    def median(self) -> float:
        values = self.correlation[self.eligible]
        return float(np.median(values)) if values.size else float("nan")

    @property
    def q25(self) -> float:
        values = self.correlation[self.eligible]
        return float(np.quantile(values, 0.25)) if values.size else float("nan")

    @property
    def q75(self) -> float:
        values = self.correlation[self.eligible]
        return float(np.quantile(values, 0.75)) if values.size else float("nan")

    @property
    def centered_full_matrix_pcc(self) -> float:
        """Pearson correlation after centring each eligible gene over spots."""

        if not self.eligible.any():
            return float("nan")
        pred_ss = float(self.centered_pred_ss[self.eligible].sum())
        true_ss = float(self.centered_true_ss[self.eligible].sum())
        if true_ss <= 0.0:
            return float("nan")
        if pred_ss <= 0.0:
            return 0.0
        cross = float(self.centered_cross[self.eligible].sum())
        return float(np.clip(cross / np.sqrt(pred_ss * true_ss), -1.0, 1.0))


def gene_correlations_from_moments(
    count: np.ndarray,
    sum_pred: np.ndarray,
    sum_true: np.ndarray,
    sum_pred2: np.ndarray,
    sum_true2: np.ndarray,
    sum_cross: np.ndarray,
    *,
    observed_std_min: float = OBSERVED_STD_MIN,
    predicted_std_min: float = PREDICTED_STD_MIN,
) -> GeneCorrelationPolicyResult:
    """Apply the common observed-defined eligibility rule to sufficient stats."""

    arrays = [
        np.asarray(value, dtype=np.float64)
        for value in (count, sum_pred, sum_true, sum_pred2, sum_true2, sum_cross)
    ]
    shape = arrays[0].shape
    if any(value.shape != shape for value in arrays[1:]):
        raise ValueError("All sufficient-statistic arrays must have the same shape")
    count64, pred, true, pred2, true2, cross = arrays
    valid = np.isfinite(count64) & (count64 > 1)

    centered_pred_ss = np.zeros(shape, dtype=np.float64)
    centered_true_ss = np.zeros(shape, dtype=np.float64)
    centered_cross = np.zeros(shape, dtype=np.float64)
    centered_pred_ss[valid] = pred2[valid] - np.square(pred[valid]) / count64[valid]
    centered_true_ss[valid] = true2[valid] - np.square(true[valid]) / count64[valid]
    centered_cross[valid] = cross[valid] - pred[valid] * true[valid] / count64[valid]
    centered_pred_ss = np.maximum(centered_pred_ss, 0.0)
    centered_true_ss = np.maximum(centered_true_ss, 0.0)

    observed_std = np.full(shape, np.nan, dtype=np.float64)
    predicted_std = np.full(shape, np.nan, dtype=np.float64)
    observed_std[valid] = np.sqrt(centered_true_ss[valid] / count64[valid])
    predicted_std[valid] = np.sqrt(centered_pred_ss[valid] / count64[valid])

    eligible = valid & np.isfinite(observed_std) & (observed_std > observed_std_min)
    prediction_variable = (
        valid & np.isfinite(predicted_std) & (predicted_std > predicted_std_min)
    )
    correlation = np.full(shape, np.nan, dtype=np.float64)
    correlation[eligible] = 0.0
    computable = eligible & prediction_variable
    correlation[computable] = centered_cross[computable] / np.sqrt(
        centered_pred_ss[computable] * centered_true_ss[computable]
    )
    correlation[computable] = np.clip(correlation[computable], -1.0, 1.0)

    return GeneCorrelationPolicyResult(
        correlation=correlation,
        eligible=eligible,
        prediction_variable=prediction_variable,
        observed_std=observed_std,
        predicted_std=predicted_std,
        centered_pred_ss=centered_pred_ss,
        centered_true_ss=centered_true_ss,
        centered_cross=centered_cross,
    )


def assert_common_eligibility(
    gene_ids: np.ndarray,
    eligibility_by_condition: dict[str, np.ndarray],
) -> None:
    """Fail if model conditions do not use exactly the same eligible genes."""

    if not eligibility_by_condition:
        raise ValueError("No eligibility arrays supplied")
    reference_name = next(iter(eligibility_by_condition))
    reference = np.asarray(eligibility_by_condition[reference_name], dtype=bool)
    for name, current in eligibility_by_condition.items():
        current_bool = np.asarray(current, dtype=bool)
        if current_bool.shape != reference.shape:
            raise AssertionError(f"Eligibility shape differs for {name}")
        mismatch = np.flatnonzero(current_bool != reference)
        if mismatch.size:
            examples = np.asarray(gene_ids)[mismatch[:10]].tolist()
            raise AssertionError(
                f"Observed-defined eligibility differs between {reference_name} and {name}; "
                f"n_mismatch={mismatch.size}, examples={examples}"
            )
