"""EMOS (Ensemble Model Output Statistics) Probability Model.

Replaces fixed Normal CDF with adaptive, data-driven probability estimation.
Key improvements over baseline (AR-0134):
  1. Adaptive sigma: learns from recent forecast errors (not fixed per-city)
  2. Adaptive bias: rolling bias correction (not fixed METAR calibration)
  3. Confidence scaling: bigger bets when model is accurate, smaller when not

Reference: Gneiting et al. (2005), "Calibrated Probabilistic Forecasting
Using EMOS and Minimum CRPS Estimation", Monthly Weather Review, Vol 133.

Usage:
    model = EMOSModel(training_window=30, sigma_method="rolling_rmse")
    model.fit(observations)  # list of (forecast, actual) pairs
    mean, sigma = model.predict(forecast_temp)
    prob = model.bucket_probability(forecast_temp, 5.0, 7.0, "range")
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class Observation:
    """A single (forecast, actual) pair for EMOS training."""

    forecast: float  # Open-Meteo archive temperature (°C)
    actual: float  # Actual temperature from resolved bucket midpoint (°C)
    date: str  # ISO date for ordering
    city: str = ""


class EMOSModel:
    """Ensemble Model Output Statistics probability model.

    Fits a calibrated Gaussian distribution:
        mean = a + b * forecast  (bias-corrected)
        variance = (c + d * error_spread)^2  (adaptive uncertainty)

    Parameters are fit on a rolling training window of (forecast, actual) pairs.
    """

    def __init__(
        self,
        training_window: int = 30,
        sigma_method: str = "rolling_rmse",
        bias_method: str = "rolling_mean",
        sigma_floor: float = 0.5,
        sigma_scale: float = 1.0,
        ewma_alpha: float = 0.1,
    ):
        self.training_window = training_window
        self.sigma_method = sigma_method
        self.bias_method = bias_method
        self.sigma_floor = sigma_floor
        self.sigma_scale = sigma_scale
        self.ewma_alpha = ewma_alpha

        # Fitted parameters
        self._bias: float = 0.0  # additive bias correction
        self._sigma: float = 2.0  # calibrated sigma
        self._n_obs: int = 0
        self._fitted: bool = False

    def fit(self, observations: list[Observation]) -> None:
        """Fit EMOS parameters from (forecast, actual) pairs.

        Uses the most recent `training_window` observations.
        """
        if not observations:
            self._fitted = False
            return

        # Take most recent observations
        obs = observations[-self.training_window :]
        self._n_obs = len(obs)

        if self._n_obs < 3:
            # Not enough data — use conservative defaults
            self._bias = 0.0
            self._sigma = max(3.0, self.sigma_floor)
            self._fitted = True
            return

        errors = [o.actual - o.forecast for o in obs]

        # ── Bias correction ──
        if self.bias_method == "rolling_mean":
            self._bias = sum(errors) / len(errors)
        elif self.bias_method == "ewma":
            self._bias = self._ewma(errors)
        elif self.bias_method == "none":
            self._bias = 0.0
        else:
            self._bias = sum(errors) / len(errors)

        # ── Sigma (adaptive uncertainty) ──
        corrected_errors = [e - self._bias for e in errors]

        if self.sigma_method == "rolling_rmse":
            mse = sum(e**2 for e in corrected_errors) / len(corrected_errors)
            self._sigma = math.sqrt(mse)
        elif self.sigma_method == "rolling_mae":
            # MAE → sigma conversion: sigma ≈ MAE * sqrt(pi/2)
            mae = sum(abs(e) for e in corrected_errors) / len(corrected_errors)
            self._sigma = mae * math.sqrt(math.pi / 2)
        elif self.sigma_method == "ewma_rmse":
            ewma_var = self._ewma([e**2 for e in corrected_errors])
            self._sigma = math.sqrt(max(ewma_var, 0.01))
        else:
            mse = sum(e**2 for e in corrected_errors) / len(corrected_errors)
            self._sigma = math.sqrt(mse)

        # Apply scale and floor
        self._sigma = max(self._sigma * self.sigma_scale, self.sigma_floor)
        self._fitted = True

    def _ewma(self, values: list[float]) -> float:
        """Exponentially Weighted Moving Average."""
        if not values:
            return 0.0
        alpha = self.ewma_alpha
        result = values[0]
        for v in values[1:]:
            result = alpha * v + (1 - alpha) * result
        return result

    def predict(self, forecast_temp: float) -> tuple[float, float]:
        """Return (corrected_mean, calibrated_sigma).

        If not fitted, returns conservative defaults.
        """
        if not self._fitted:
            return forecast_temp, 3.0

        corrected_mean = forecast_temp + self._bias
        return corrected_mean, self._sigma

    def bucket_probability(
        self,
        forecast_temp: float,
        bucket_low: float | None,
        bucket_high: float | None,
        bucket_type: str,
    ) -> float:
        """Compute P(temp in bucket) using calibrated Gaussian.

        Args:
            forecast_temp: Raw forecast temperature (°C)
            bucket_low: Lower bound (°C), None for "above" buckets
            bucket_high: Upper bound (°C), None for "below" buckets
            bucket_type: "range", "above", or "below"

        Returns:
            Probability in [0, 1]
        """
        mean, sigma = self.predict(forecast_temp)

        if bucket_type == "below" and bucket_high is not None:
            return _normal_cdf(bucket_high, mean, sigma)
        elif bucket_type == "above" and bucket_low is not None:
            return 1.0 - _normal_cdf(bucket_low, mean, sigma)
        elif bucket_low is not None and bucket_high is not None:
            return _normal_cdf(bucket_high, mean, sigma) - _normal_cdf(
                bucket_low, mean, sigma
            )
        else:
            return 0.0

    @property
    def bias(self) -> float:
        return self._bias

    @property
    def sigma(self) -> float:
        return self._sigma

    @property
    def n_observations(self) -> int:
        return self._n_obs


def _normal_cdf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """Standard normal CDF using math.erf."""
    if sigma <= 0:
        return 1.0 if x >= mu else 0.0
    return 0.5 * (1.0 + math.erf((x - mu) / (sigma * math.sqrt(2.0))))


class EMOSEnsembleModel:
    """Full EMOS with real ensemble spread (forward-looking sigma).

    Unlike EMOSModel (backward-looking rolling errors), this uses actual
    GFS ensemble member spread to estimate forecast uncertainty per day.

    sigma = c + d * ensemble_std

    Where ensemble_std comes from 31 GFS members disagreeing about
    tomorrow's temperature. High disagreement = high uncertainty = wider sigma.

    Parameters c, d are fit from (ensemble_std, actual_error) pairs
    over a rolling training window.
    """

    def __init__(
        self,
        training_window: int = 45,
        sigma_floor: float = 0.5,
        bias_method: str = "rolling_mean",
        ewma_alpha: float = 0.1,
    ):
        self.training_window = training_window
        self.sigma_floor = sigma_floor
        self.bias_method = bias_method
        self.ewma_alpha = ewma_alpha

        self._bias: float = 0.0
        self._c: float = 0.5  # intercept
        self._d: float = 1.0  # slope (ensemble_std coefficient)
        self._fitted: bool = False
        self._n_obs: int = 0

    def fit(
        self,
        observations: list[Observation],
        ensemble_stds: dict[str, float],  # date -> ensemble_std
    ) -> None:
        """Fit EMOS from (forecast, actual, ensemble_std) triples.

        observations: list of Observation(forecast, actual, date)
        ensemble_stds: {date_str: ensemble_std_for_that_date}
        """
        # Filter to observations with ensemble data
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

        # Fit sigma = c + d * ensemble_std via simple linear regression
        # of |corrected_error| on ensemble_std
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
            # Ensemble spread is constant — fall back to mean error
            self._d = 0.0
            self._c = max(self.sigma_floor, mean_y)

        self._fitted = True

    def predict(
        self, forecast_temp: float, ensemble_std: float | None = None
    ) -> tuple[float, float]:
        """Return (corrected_mean, calibrated_sigma)."""
        if not self._fitted:
            return forecast_temp, 2.0

        corrected = forecast_temp + self._bias

        if ensemble_std is not None:
            sigma = self._c + self._d * ensemble_std
        else:
            sigma = self._c + self._d * 1.5  # fallback: assume moderate spread

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


def actual_temp_from_bucket(
    low_c: float | None,
    high_c: float | None,
    bucket_type: str | None,
) -> float | None:
    """Estimate actual temperature from winning bucket boundaries.

    Returns midpoint for range buckets, or offset from boundary for
    above/below buckets. Returns None if boundaries are missing.
    """
    if bucket_type == "range" and low_c is not None and high_c is not None:
        return (low_c + high_c) / 2.0
    elif bucket_type == "below" and high_c is not None:
        # Assume actual is ~1°C below the upper threshold
        return high_c - 1.0
    elif bucket_type == "above" and low_c is not None:
        # Assume actual is ~1°C above the lower threshold
        return low_c + 1.0
    elif low_c is not None and high_c is not None:
        return (low_c + high_c) / 2.0
    return None
