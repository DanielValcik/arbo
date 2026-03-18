"""EMOSEnsembleModel — Forward-looking sigma from GFS ensemble spread.

Uses real GEFS 31-member ensemble disagreement to estimate per-day
forecast uncertainty. Replaces backward-looking rolling sigma for
cities with sufficient training data.

sigma = c + d * ensemble_std

Where (c, d) are fit from historical (ensemble_std, actual_error) pairs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class Observation:
    """A (forecast, actual) pair for EMOS training."""

    forecast: float
    actual: float
    date: str
    city: str = ""


def _normal_cdf(x: float, mu: float, sigma: float) -> float:
    """Standard normal CDF."""
    if sigma <= 0:
        return 1.0 if x >= mu else 0.0
    return 0.5 * (1.0 + math.erf((x - mu) / (sigma * math.sqrt(2.0))))


class EMOSEnsembleModel:
    """EMOS with real ensemble spread (forward-looking sigma).

    sigma = c + d * ensemble_std

    Parameters c, d are fit via linear regression of |corrected_error|
    on ensemble_std over a rolling training window.
    """

    def __init__(
        self,
        training_window: int = 66,
        sigma_floor: float = 0.6,
        bias_method: str = "rolling_mean",
        ewma_alpha: float = 0.3,
    ):
        self.training_window = training_window
        self.sigma_floor = sigma_floor
        self.bias_method = bias_method
        self.ewma_alpha = ewma_alpha

        self._bias: float = 0.0
        self._c: float = 0.5
        self._d: float = 1.0
        self._fitted: bool = False
        self._n_obs: int = 0

    def fit(
        self,
        observations: list[Observation],
        ensemble_stds: dict[str, float],
    ) -> None:
        """Fit EMOS from (forecast, actual, ensemble_std) triples."""
        paired = []
        for obs in observations[-self.training_window :]:
            std = ensemble_stds.get(obs.date)
            if std is not None:
                error = obs.actual - obs.forecast
                paired.append((obs.forecast, obs.actual, error, std))

        self._n_obs = len(paired)
        if len(paired) < 5:
            self._bias = 0.0
            self._c = 1.5
            self._d = 1.0
            self._fitted = len(paired) > 0
            return

        errors = [p[2] for p in paired]
        stds = [p[3] for p in paired]

        # Bias correction
        if self.bias_method == "rolling_mean":
            self._bias = sum(errors) / len(errors)
        elif self.bias_method == "ewma":
            self._bias = errors[0]
            for e in errors[1:]:
                self._bias = self.ewma_alpha * e + (1 - self.ewma_alpha) * self._bias
        else:
            self._bias = 0.0

        # Fit sigma = c + d * ensemble_std via linear regression
        corrected = [abs(e - self._bias) for e in errors]
        n = len(stds)
        mean_x = sum(stds) / n
        mean_y = sum(corrected) / n
        cov_xy = sum((stds[i] - mean_x) * (corrected[i] - mean_y) for i in range(n))
        var_x = sum((s - mean_x) ** 2 for s in stds)

        if var_x > 0.001:
            self._d = max(0.1, cov_xy / var_x)
            self._c = max(0.1, mean_y - self._d * mean_x)
        else:
            self._d = 0.0
            self._c = max(self.sigma_floor, mean_y)

        self._fitted = True

    def predict(
        self, forecast_temp: float, ensemble_std: float | None = None,
    ) -> tuple[float, float]:
        """Return (corrected_mean, calibrated_sigma)."""
        if not self._fitted:
            return forecast_temp, 2.0

        corrected = forecast_temp + self._bias
        if ensemble_std is not None:
            sigma = self._c + self._d * ensemble_std
        else:
            sigma = self._c + self._d * 1.5
        sigma = max(sigma, self.sigma_floor)
        return corrected, sigma

    def bucket_probability(
        self,
        forecast_temp: float,
        bucket_low: float | None,
        bucket_high: float | None,
        bucket_type: str,
        ensemble_std: float | None = None,
    ) -> float:
        """Compute P(temp in bucket) using ensemble-calibrated Gaussian."""
        mean, sigma = self.predict(forecast_temp, ensemble_std)

        if bucket_type == "below" and bucket_high is not None:
            return _normal_cdf(bucket_high, mean, sigma)
        elif bucket_type == "above" and bucket_low is not None:
            return 1.0 - _normal_cdf(bucket_low, mean, sigma)
        elif bucket_low is not None and bucket_high is not None:
            return _normal_cdf(bucket_high, mean, sigma) - _normal_cdf(
                bucket_low, mean, sigma
            )
        return 0.0

    @property
    def bias(self) -> float:
        return self._bias

    @property
    def sigma_params(self) -> tuple[float, float]:
        """Return (c, d) — sigma = c + d * ensemble_std."""
        return self._c, self._d

    @property
    def n_observations(self) -> int:
        return self._n_obs
