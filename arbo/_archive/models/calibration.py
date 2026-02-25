"""Calibration utilities for the XGBoost value model.

Implements Platt scaling, Brier score calculation, and
reliability diagram data generation.

PM-101 acceptance: Brier score < 0.22 on holdout test set.
"""

from __future__ import annotations

import numpy as np
from sklearn.calibration import _SigmoidCalibration
from sklearn.metrics import brier_score_loss

# NOTE: _SigmoidCalibration is a private sklearn API used because
# CalibratedClassifierCV removed cv="prefit" in sklearn 1.8.
# If sklearn changes internals, fallback: fit LogisticRegression on
# log-odds of raw_probs vs y_true, then sigmoid(a*x + b) for calibration.
# See: sklearn.calibration._SigmoidCalibration source — it's just
# Platt's method: P(y=1|f) = 1/(1 + exp(A*f + B)), fitted via MLE.


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Calculate Brier score (lower is better).

    BS = (1/N) * sum((forecast - actual)^2)
    Perfect: 0.0, Worst: 1.0, Baseline (always 0.5): 0.25

    PM-101 acceptance threshold: < 0.22
    """
    return float(brier_score_loss(y_true, y_prob))


def reliability_diagram_data(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate data for a reliability (calibration) diagram.

    Returns:
        bin_centers: Midpoints of probability bins.
        observed_freq: Observed frequency of positives in each bin.
        bin_counts: Number of samples in each bin.
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    observed_freq = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if i == n_bins - 1:
            mask = mask | (y_prob == bin_edges[i + 1])
        count = int(mask.sum())
        bin_counts[i] = count
        if count > 0:
            observed_freq[i] = float(y_true[mask].mean())

    return bin_centers, observed_freq, bin_counts


class PlattCalibrator:
    """Platt scaling calibrator using sklearn's sigmoid calibration.

    Takes raw predicted probabilities from XGBoost and maps them through
    a fitted sigmoid (Platt scaling) for better calibration.
    """

    def __init__(self) -> None:
        self._sigmoid = _SigmoidCalibration()
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        """Whether the calibrator has been fitted."""
        return self._is_fitted

    def fit(self, raw_probs: np.ndarray, y_true: np.ndarray) -> None:
        """Fit Platt scaling on calibration data.

        Args:
            raw_probs: Uncalibrated predicted probabilities (1D array).
            y_true: True binary labels (0/1).
        """
        self._sigmoid.fit(raw_probs, y_true)
        self._is_fitted = True

    def predict(self, raw_probs: np.ndarray) -> np.ndarray:
        """Get calibrated probabilities from raw predictions.

        Args:
            raw_probs: Uncalibrated predicted probabilities (1D array).

        Returns:
            1D array of calibrated probabilities.
        """
        if not self._is_fitted:
            raise RuntimeError("Calibrator not fitted. Call fit() first.")
        return self._sigmoid.predict(raw_probs)
