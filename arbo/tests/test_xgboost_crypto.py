"""Tests for crypto XGBoost model (PM-401)."""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from arbo.models.crypto_features import (
    CRYPTO_FEATURE_COLUMNS,
    CryptoFeatures,
    compute_distance_pct,
    compute_spot_vs_strike,
    crypto_features_to_dataframe,
    extract_crypto_feature_vector,
)
from arbo.models.xgboost_crypto import CryptoValueModel

# ---------------------------------------------------------------------------
# Helper: generate synthetic training data
# ---------------------------------------------------------------------------


def _make_training_data(n: int = 300, seed: int = 42) -> tuple[pd.DataFrame, np.ndarray]:
    """Generate synthetic crypto training data.

    Features mimic real crypto markets:
    - spot_vs_strike near 1.0 with noise
    - time_to_expiry 1-720 hours
    - volatility, RSI, momentum, etc.
    - label = 1 if spot_vs_strike > 1 + noise (correlated with features)
    """
    rng = np.random.RandomState(seed)

    spot_ratio = rng.normal(1.0, 0.1, n)
    time_exp = rng.uniform(1, 720, n)
    vol_24h = rng.uniform(0.1, 2.0, n)
    vol_7d = rng.uniform(0.1, 1.5, n)
    vol_log = rng.uniform(10, 25, n)
    vol_trend = rng.uniform(0.5, 2.0, n)
    funding = rng.normal(0.0001, 0.0005, n)
    rsi = rng.uniform(20, 80, n)
    momentum = rng.normal(0, 0.05, n)
    distance = np.abs(spot_ratio - 1.0)
    poly_mid = rng.uniform(0.1, 0.9, n)

    data = {
        "spot_vs_strike": spot_ratio,
        "time_to_expiry": time_exp,
        "time_to_expiry_log": np.log1p(time_exp),
        "volatility_24h": vol_24h,
        "volatility_7d": vol_7d,
        "volume_24h_log": vol_log,
        "volume_trend": vol_trend,
        "funding_rate": funding,
        "rsi_14": rsi,
        "momentum_24h": momentum,
        "distance_pct": distance,
        "polymarket_mid": poly_mid,
    }
    X = pd.DataFrame(data)[CRYPTO_FEATURE_COLUMNS]  # noqa: N806

    # Labels: correlated with spot_vs_strike > 1 and momentum
    prob = 1 / (1 + np.exp(-(spot_ratio - 1.0) * 20 - momentum * 5))
    y = (rng.random(n) < prob).astype(int)

    return X, y


# ---------------------------------------------------------------------------
# Feature extraction tests
# ---------------------------------------------------------------------------


class TestCryptoFeatureExtraction:
    def test_feature_extraction_complete(self) -> None:
        """All 12 features present in extracted vector."""
        features = CryptoFeatures(
            spot_vs_strike=1.05,
            time_to_expiry=24.0,
            volatility_24h=0.65,
            volatility_7d=0.55,
            volume_24h_log=20.5,
            volume_trend=1.3,
            funding_rate=0.0001,
            rsi_14=55.0,
            momentum_24h=0.02,
            distance_pct=0.05,
            polymarket_mid=0.60,
        )
        vec = extract_crypto_feature_vector(features)
        assert set(vec.keys()) == set(CRYPTO_FEATURE_COLUMNS)
        assert vec["spot_vs_strike"] == 1.05
        assert vec["rsi_14"] == 55.0

    def test_dataframe_column_order(self) -> None:
        """DataFrame has correct column order matching CRYPTO_FEATURE_COLUMNS."""
        features = CryptoFeatures(spot_vs_strike=1.0, time_to_expiry=24.0)
        vec = extract_crypto_feature_vector(features)
        df = crypto_features_to_dataframe([vec])
        assert list(df.columns) == CRYPTO_FEATURE_COLUMNS

    def test_missing_values_are_nan(self) -> None:
        """Missing optional fields become NaN."""
        features = CryptoFeatures()  # All defaults
        vec = extract_crypto_feature_vector(features)
        assert math.isnan(vec["spot_vs_strike"])
        assert math.isnan(vec["volatility_24h"])


class TestSpotVsStrike:
    def test_computation(self) -> None:
        """spot_vs_strike = spot / strike."""
        assert abs(compute_spot_vs_strike(105000, 100000) - 1.05) < 0.001

    def test_zero_strike(self) -> None:
        """Returns NaN for zero strike."""
        assert math.isnan(compute_spot_vs_strike(100, 0))

    def test_distance_pct(self) -> None:
        """distance_pct = |spot - strike| / strike."""
        assert abs(compute_distance_pct(105000, 100000) - 0.05) < 0.001


# ---------------------------------------------------------------------------
# Model training and prediction tests
# ---------------------------------------------------------------------------


class TestCryptoValueModelTraining:
    def test_train_returns_metrics(self) -> None:
        """Train returns dict with expected metric keys."""
        X, y = _make_training_data(200)  # noqa: N806
        model = CryptoValueModel()
        metrics = model.train(X, y)
        assert "brier_train" in metrics
        assert "brier_cal_calibrated" in metrics
        assert "n_train" in metrics
        assert metrics["n_train"] > 0

    def test_predict_output_range(self) -> None:
        """Predictions are in [0, 1]."""
        X, y = _make_training_data(200)  # noqa: N806
        model = CryptoValueModel()
        model.train(X, y)

        probs = model.predict_proba_df(X)
        assert all(0 <= p <= 1 for p in probs)

    def test_predict_single(self) -> None:
        """predict_single returns a float in [0, 1]."""
        X, y = _make_training_data(200)  # noqa: N806
        model = CryptoValueModel()
        model.train(X, y)

        features = CryptoFeatures(
            spot_vs_strike=1.05,
            time_to_expiry=24.0,
            volatility_24h=0.6,
            volatility_7d=0.5,
            volume_24h_log=20.0,
            volume_trend=1.2,
            funding_rate=0.0001,
            rsi_14=55.0,
            momentum_24h=0.02,
            distance_pct=0.05,
            polymarket_mid=0.55,
        )
        prob = model.predict_single(features)
        assert 0.0 <= prob <= 1.0

    def test_is_trained_flag(self) -> None:
        """is_trained starts False, becomes True after train()."""
        model = CryptoValueModel()
        assert model.is_trained is False

        X, y = _make_training_data(200)  # noqa: N806
        model.train(X, y)
        assert model.is_trained is True

    def test_untrained_predict_raises(self) -> None:
        """Predicting without training raises RuntimeError."""
        model = CryptoValueModel()
        features = CryptoFeatures(spot_vs_strike=1.0)
        with pytest.raises(RuntimeError, match="not trained"):
            model.predict_proba(features)


class TestCryptoValueModelPersistence:
    def test_save_load_roundtrip(self) -> None:
        """Model survives save/load roundtrip."""
        X, y = _make_training_data(200)  # noqa: N806
        model = CryptoValueModel()
        model.train(X, y)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_crypto.joblib"
            model.save(path)

            loaded = CryptoValueModel()
            loaded.load(path)

            assert loaded.is_trained
            assert loaded.brier_score_val == model.brier_score_val

            # Predictions should match
            features = CryptoFeatures(
                spot_vs_strike=1.02,
                time_to_expiry=48.0,
                volatility_24h=0.5,
            )
            orig_prob = model.predict_single(features)
            loaded_prob = loaded.predict_single(features)
            assert abs(orig_prob - loaded_prob) < 1e-6

    def test_save_untrained_raises(self) -> None:
        """Cannot save untrained model."""
        model = CryptoValueModel()
        with pytest.raises(RuntimeError, match="untrained"):
            model.save("/tmp/should_not_exist.joblib")


class TestCryptoValueModelEvaluate:
    def test_evaluate_returns_metrics(self) -> None:
        """Evaluate returns brier_score, accuracy, n_samples."""
        X, y = _make_training_data(200)  # noqa: N806
        model = CryptoValueModel()
        model.train(X, y)

        metrics = model.evaluate(X, y)
        assert "brier_score" in metrics
        assert "accuracy" in metrics
        assert metrics["n_samples"] == 200
        assert 0 <= metrics["brier_score"] <= 1

    def test_calibration_improves_brier(self) -> None:
        """Calibrated Brier should be <= raw Brier (or very close)."""
        X, y = _make_training_data(300)  # noqa: N806
        model = CryptoValueModel()
        metrics = model.train(X, y)

        # Calibration improvement should be >= 0 (or close to it)
        # In practice, Platt scaling may not always improve, but should not worsen much
        assert metrics["calibration_improvement"] > -0.05
