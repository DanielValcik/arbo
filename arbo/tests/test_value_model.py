"""Tests for PM-101: XGBoost Value Model.

Tests feature engineering, calibration, model training,
prediction, serialization, and backtest simulation.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from arbo.models.calibration import PlattCalibrator, brier_score, reliability_diagram_data
from arbo.models.feature_engineering import (
    CATEGORY_ENCODING,
    FEATURE_COLUMNS,
    MarketFeatures,
    extract_feature_vector,
    features_to_dataframe,
)
from arbo.models.xgboost_value import ValueModel, backtest_strategy

# ================================================================
# Synthetic data generator
# ================================================================


def _generate_synthetic_data(
    n: int = 500, seed: int = 42
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic training data with learnable signal.

    Returns:
        X: Feature DataFrame
        y: Binary outcomes (0/1)
        market_prices: Polymarket YES prices
        fee_rates: Fee rates per market
    """
    rng = np.random.default_rng(seed)

    # True probabilities from Beta(2, 2) — centered ~0.5 with spread
    true_probs = rng.beta(2, 2, size=n)

    # Pinnacle prob: sharp estimate (close to true)
    pinnacle_probs = np.clip(true_probs + rng.normal(0, 0.03, n), 0.01, 0.99)

    # Polymarket mid: noisier (less efficient market)
    poly_mids = np.clip(true_probs + rng.normal(0, 0.07, n), 0.01, 0.99)

    # Outcomes from true probability
    y = rng.binomial(1, true_probs).astype(np.int64)

    # Features
    time_hours = rng.uniform(1, 720, n)
    volume_24h = rng.lognormal(8, 2, n)
    volume_30d = volume_24h * rng.uniform(0.5, 2.0, n)
    liquidity = rng.lognormal(7, 1.5, n)
    categories = rng.choice(list(CATEGORY_ENCODING.keys()), n)
    spreads = np.abs(1.0 - poly_mids - (1.0 - poly_mids) + rng.normal(0, 0.02, n))
    fee_enabled = rng.choice([True, False], n)

    feature_dicts = []
    for i in range(n):
        mf = MarketFeatures(
            pinnacle_prob=float(pinnacle_probs[i]),
            polymarket_mid=float(poly_mids[i]),
            time_to_event_hours=float(time_hours[i]),
            category=str(categories[i]),
            volume_24h=float(volume_24h[i]),
            volume_30d_avg=float(volume_30d[i]),
            liquidity=float(liquidity[i]),
            spread=float(spreads[i]),
            fee_enabled=bool(fee_enabled[i]),
        )
        feature_dicts.append(extract_feature_vector(mf))

    X = features_to_dataframe(feature_dicts)
    market_prices = poly_mids.copy()
    fee_rates = np.where(fee_enabled, market_prices * (1 - market_prices) * 0.02, 0.0)

    return X, y, market_prices, fee_rates


# ================================================================
# Feature Engineering Tests
# ================================================================


class TestFeatureEngineering:
    """Tests for feature_engineering.py."""

    def test_extract_all_features_present(self) -> None:
        """All FEATURE_COLUMNS keys should appear in extracted dict."""
        mf = MarketFeatures(pinnacle_prob=0.6, polymarket_mid=0.55)
        result = extract_feature_vector(mf)
        for col in FEATURE_COLUMNS:
            assert col in result, f"Missing feature: {col}"

    def test_missing_pinnacle_is_nan(self) -> None:
        """Missing pinnacle_prob should produce NaN."""
        mf = MarketFeatures(polymarket_mid=0.5)
        result = extract_feature_vector(mf)
        assert math.isnan(result["pinnacle_prob"])

    def test_missing_polymarket_is_nan(self) -> None:
        """Missing polymarket_mid should produce NaN."""
        mf = MarketFeatures(pinnacle_prob=0.6)
        result = extract_feature_vector(mf)
        assert math.isnan(result["polymarket_mid"])

    def test_price_divergence_calculated(self) -> None:
        """Divergence = pinnacle - polymarket."""
        mf = MarketFeatures(pinnacle_prob=0.7, polymarket_mid=0.6)
        result = extract_feature_vector(mf)
        assert abs(result["price_divergence"] - 0.1) < 1e-9

    def test_price_divergence_nan_when_missing(self) -> None:
        """Divergence is NaN if either price missing."""
        mf = MarketFeatures(pinnacle_prob=0.7)
        result = extract_feature_vector(mf)
        assert math.isnan(result["price_divergence"])

    def test_volume_trend_with_zero_avg(self) -> None:
        """Zero 30d average should return 1.0 (neutral)."""
        mf = MarketFeatures(volume_24h=1000, volume_30d_avg=0)
        result = extract_feature_vector(mf)
        assert result["volume_trend"] == 1.0

    def test_volume_trend_ratio(self) -> None:
        """Volume trend = 24h / 30d_avg."""
        mf = MarketFeatures(volume_24h=2000, volume_30d_avg=1000)
        result = extract_feature_vector(mf)
        assert abs(result["volume_trend"] - 2.0) < 1e-9

    def test_category_encoding_soccer(self) -> None:
        """Soccer should encode to 0."""
        mf = MarketFeatures(category="soccer")
        result = extract_feature_vector(mf)
        assert result["category"] == 0.0

    def test_category_encoding_unknown(self) -> None:
        """Unknown category should encode to 'other' (6)."""
        mf = MarketFeatures(category="magic")
        result = extract_feature_vector(mf)
        assert result["category"] == 6.0

    def test_log_transforms(self) -> None:
        """Log transforms should use log1p."""
        mf = MarketFeatures(volume_24h=1000, liquidity=500, time_to_event_hours=48)
        result = extract_feature_vector(mf)
        assert abs(result["volume_24h_log"] - math.log1p(1000)) < 1e-9
        assert abs(result["liquidity_log"] - math.log1p(500)) < 1e-9
        assert abs(result["time_log"] - math.log1p(48)) < 1e-9

    def test_fee_flag(self) -> None:
        """Fee enabled should encode to 1.0."""
        mf = MarketFeatures(fee_enabled=True)
        assert extract_feature_vector(mf)["fee_enabled"] == 1.0
        mf2 = MarketFeatures(fee_enabled=False)
        assert extract_feature_vector(mf2)["fee_enabled"] == 0.0

    def test_features_to_dataframe_column_order(self) -> None:
        """DataFrame must have columns in exact FEATURE_COLUMNS order."""
        mf = MarketFeatures(pinnacle_prob=0.6, polymarket_mid=0.5)
        df = features_to_dataframe([extract_feature_vector(mf)])
        assert list(df.columns) == FEATURE_COLUMNS

    def test_features_to_dataframe_missing_cols_filled(self) -> None:
        """Missing columns in input should be filled with NaN."""
        df = features_to_dataframe([{"pinnacle_prob": 0.5}])
        assert list(df.columns) == FEATURE_COLUMNS
        assert math.isnan(df["polymarket_mid"].iloc[0])


# ================================================================
# Calibration Tests
# ================================================================


class TestCalibration:
    """Tests for calibration.py."""

    def test_brier_perfect(self) -> None:
        """Perfect predictions should have Brier = 0."""
        y = np.array([1, 0, 1, 0])
        p = np.array([1.0, 0.0, 1.0, 0.0])
        assert brier_score(y, p) == 0.0

    def test_brier_worst(self) -> None:
        """Worst predictions should have Brier = 1."""
        y = np.array([1, 0, 1, 0])
        p = np.array([0.0, 1.0, 0.0, 1.0])
        assert brier_score(y, p) == 1.0

    def test_brier_baseline(self) -> None:
        """Always predicting 0.5 should have Brier = 0.25."""
        y = np.array([1, 0, 1, 0])
        p = np.array([0.5, 0.5, 0.5, 0.5])
        assert abs(brier_score(y, p) - 0.25) < 1e-9

    def test_brier_range(self) -> None:
        """Brier score should be in [0, 1]."""
        rng = np.random.default_rng(42)
        y = rng.binomial(1, 0.5, 100)
        p = rng.uniform(0, 1, 100)
        bs = brier_score(y, p)
        assert 0.0 <= bs <= 1.0

    def test_reliability_diagram_bins(self) -> None:
        """Should return correct number of bins."""
        y = np.array([1, 0, 1, 0, 1])
        p = np.array([0.9, 0.1, 0.8, 0.2, 0.7])
        centers, freq, counts = reliability_diagram_data(y, p, n_bins=5)
        assert len(centers) == 5
        assert len(freq) == 5
        assert len(counts) == 5

    def test_reliability_diagram_sums(self) -> None:
        """Total bin counts should equal number of samples."""
        rng = np.random.default_rng(42)
        n = 200
        y = rng.binomial(1, 0.5, n)
        p = rng.uniform(0, 1, n)
        _, _, counts = reliability_diagram_data(y, p, n_bins=10)
        assert counts.sum() == n

    def test_platt_calibrator_unfitted_raises(self) -> None:
        """Predicting before fitting should raise RuntimeError."""
        cal = PlattCalibrator()
        with pytest.raises(RuntimeError, match="not fitted"):
            cal.predict(np.array([0.5, 0.3]))

    def test_platt_calibrator_fit_predict(self) -> None:
        """Fitted calibrator should return valid probabilities."""
        raw_probs = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 0.2, 0.8, 0.4, 0.6, 0.95])
        y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0, 1, 1])

        cal = PlattCalibrator()
        cal.fit(raw_probs, y_true)
        assert cal.is_fitted

        calibrated = cal.predict(raw_probs)
        assert len(calibrated) == 10
        assert all(0 <= p <= 1 for p in calibrated)


# ================================================================
# Value Model Tests
# ================================================================


class TestValueModel:
    """Tests for xgboost_value.py — ValueModel class."""

    @pytest.fixture()
    def trained_model(self) -> tuple[ValueModel, pd.DataFrame, np.ndarray]:
        """Train a model on synthetic data and return model + test data."""
        X, y, _, _ = _generate_synthetic_data(n=500, seed=42)

        # Split 70/30
        n_train = 350
        X_train, y_train = X.iloc[:n_train], y[:n_train]
        X_test, y_test = X.iloc[n_train:], y[n_train:]

        model = ValueModel()
        model.train(X_train, y_train)
        return model, X_test, y_test

    def test_untrained_predict_raises(self) -> None:
        """Predicting before training should raise."""
        model = ValueModel()
        mf = MarketFeatures(pinnacle_prob=0.6, polymarket_mid=0.5)
        with pytest.raises(RuntimeError, match="not trained"):
            model.predict_proba(mf)

    def test_untrained_save_raises(self) -> None:
        """Saving untrained model should raise."""
        model = ValueModel()
        with pytest.raises(RuntimeError, match="untrained"):
            model.save("/tmp/test.joblib")

    def test_train_returns_metrics(
        self, trained_model: tuple[ValueModel, pd.DataFrame, np.ndarray]
    ) -> None:
        """Training should return dict with expected metric keys."""
        # Retrain to get metrics
        X, y, _, _ = _generate_synthetic_data(n=300, seed=99)
        model = ValueModel()
        metrics = model.train(X, y)

        assert "brier_train" in metrics
        assert "brier_cal_raw" in metrics
        assert "brier_cal_calibrated" in metrics
        assert "n_train" in metrics
        assert "n_cal" in metrics
        assert metrics["n_train"] > 0
        assert metrics["n_cal"] > 0

    def test_model_is_trained_flag(
        self, trained_model: tuple[ValueModel, pd.DataFrame, np.ndarray]
    ) -> None:
        """After training, is_trained should be True."""
        model, _, _ = trained_model
        assert model.is_trained is True

    def test_brier_score_stored(
        self, trained_model: tuple[ValueModel, pd.DataFrame, np.ndarray]
    ) -> None:
        """Brier score should be stored after training."""
        model, _, _ = trained_model
        assert model.brier_score_val is not None
        assert 0 < model.brier_score_val < 1

    def test_feature_importance_available(
        self, trained_model: tuple[ValueModel, pd.DataFrame, np.ndarray]
    ) -> None:
        """Feature importance should be populated after training."""
        model, _, _ = trained_model
        assert model.feature_importance is not None
        assert len(model.feature_importance) == len(FEATURE_COLUMNS)
        assert all(v >= 0 for v in model.feature_importance.values())

    def test_predict_single_returns_float(
        self, trained_model: tuple[ValueModel, pd.DataFrame, np.ndarray]
    ) -> None:
        """predict_single should return a float in [0, 1]."""
        model, _, _ = trained_model
        mf = MarketFeatures(pinnacle_prob=0.6, polymarket_mid=0.5, category="soccer")
        prob = model.predict_single(mf)
        assert isinstance(prob, float)
        assert 0 <= prob <= 1

    def test_predict_multiple_returns_array(
        self, trained_model: tuple[ValueModel, pd.DataFrame, np.ndarray]
    ) -> None:
        """predict_proba with list should return array matching length."""
        model, _, _ = trained_model
        features = [
            MarketFeatures(pinnacle_prob=0.3, polymarket_mid=0.4),
            MarketFeatures(pinnacle_prob=0.7, polymarket_mid=0.6),
            MarketFeatures(pinnacle_prob=0.5, polymarket_mid=0.5),
        ]
        probs = model.predict_proba(features)
        assert len(probs) == 3
        assert all(0 <= p <= 1 for p in probs)

    def test_predict_df(self, trained_model: tuple[ValueModel, pd.DataFrame, np.ndarray]) -> None:
        """predict_proba_df should work with raw DataFrame."""
        model, X_test, _ = trained_model
        probs = model.predict_proba_df(X_test)
        assert len(probs) == len(X_test)
        assert all(0 <= p <= 1 for p in probs)

    def test_evaluate_returns_metrics(
        self, trained_model: tuple[ValueModel, pd.DataFrame, np.ndarray]
    ) -> None:
        """evaluate should return dict with brier_score, accuracy, etc."""
        model, X_test, y_test = trained_model
        metrics = model.evaluate(X_test, y_test)

        assert "brier_score" in metrics
        assert "accuracy" in metrics
        assert "n_samples" in metrics
        assert metrics["n_samples"] == len(y_test)
        assert 0 <= metrics["brier_score"] <= 1
        assert 0 <= metrics["accuracy"] <= 1

    def test_brier_below_threshold(
        self, trained_model: tuple[ValueModel, pd.DataFrame, np.ndarray]
    ) -> None:
        """Brier score on synthetic data should be < 0.25 (better than random).

        PM-101 acceptance is < 0.22 on real data. Synthetic data with
        strong signal should achieve better than baseline.
        """
        model, X_test, y_test = trained_model
        metrics = model.evaluate(X_test, y_test)
        assert (
            metrics["brier_score"] < 0.25
        ), f"Brier {metrics['brier_score']:.4f} not below 0.25 baseline"

    def test_save_and_load(
        self, trained_model: tuple[ValueModel, pd.DataFrame, np.ndarray]
    ) -> None:
        """Model should be serializable and produce same predictions after load."""
        model, X_test, _ = trained_model
        probs_before = model.predict_proba_df(X_test)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_model.joblib"
            model.save(path)
            assert path.exists()

            loaded = ValueModel()
            loaded.load(path)
            assert loaded.is_trained
            assert loaded.brier_score_val == model.brier_score_val

            probs_after = loaded.predict_proba_df(X_test)
            np.testing.assert_array_almost_equal(probs_before, probs_after)

    def test_custom_params(self) -> None:
        """Custom params should override defaults."""
        model = ValueModel(params={"max_depth": 3, "n_estimators": 50})
        X, y, _, _ = _generate_synthetic_data(n=200, seed=55)
        metrics = model.train(X, y)
        assert metrics["n_train"] > 0


# ================================================================
# Optuna Tuning Tests
# ================================================================


class TestTuning:
    """Tests for tune_hyperparameters function."""

    def test_tune_returns_params(self) -> None:
        """Optuna tuning should return a valid param dict."""
        from arbo.models.xgboost_value import tune_hyperparameters

        X, y, _, _ = _generate_synthetic_data(n=200, seed=77)

        best = tune_hyperparameters(X, y, n_trials=3)

        assert "max_depth" in best
        assert "learning_rate" in best
        assert "n_estimators" in best
        assert best["objective"] == "binary:logistic"
        assert isinstance(best["max_depth"], int)

    def test_tuned_params_produce_model(self) -> None:
        """Model trained with tuned params should work."""
        from arbo.models.xgboost_value import tune_hyperparameters

        X, y, _, _ = _generate_synthetic_data(n=200, seed=88)
        best = tune_hyperparameters(X, y, n_trials=3)

        model = ValueModel(params=best)
        metrics = model.train(X, y)
        assert model.is_trained
        assert metrics["brier_cal_calibrated"] < 1.0


# ================================================================
# Backtest Tests
# ================================================================


class TestBacktest:
    """Tests for backtest_strategy function."""

    @pytest.fixture()
    def backtest_setup(self) -> tuple[ValueModel, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare trained model + test data for backtest."""
        X, y, prices, fees = _generate_synthetic_data(n=500, seed=42)
        n_train = 350
        X_train, y_train = X.iloc[:n_train], y[:n_train]
        X_test, y_test = X.iloc[n_train:], y[n_train:]
        prices_test = prices[n_train:]
        fees_test = fees[n_train:]

        model = ValueModel()
        model.train(X_train, y_train)
        return model, X_test, y_test, prices_test, fees_test

    def test_backtest_returns_metrics(
        self,
        backtest_setup: tuple[ValueModel, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """Backtest should return dict with expected keys."""
        model, X_test, y_test, prices, fees = backtest_setup
        result = backtest_strategy(model, X_test, y_test, prices, fees)

        assert "roi" in result
        assert "n_bets" in result
        assert "final_bankroll" in result
        assert "max_drawdown" in result
        assert "sharpe" in result
        assert "avg_edge" in result

    def test_backtest_places_bets(
        self,
        backtest_setup: tuple[ValueModel, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """With edge_threshold=0.01, should place some bets."""
        model, X_test, y_test, prices, fees = backtest_setup
        result = backtest_strategy(model, X_test, y_test, prices, fees, edge_threshold=0.01)
        assert result["n_bets"] > 0

    def test_backtest_high_threshold_no_bets(
        self,
        backtest_setup: tuple[ValueModel, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """Very high threshold should result in 0 bets."""
        model, X_test, y_test, prices, fees = backtest_setup
        result = backtest_strategy(model, X_test, y_test, prices, fees, edge_threshold=0.99)
        assert result["n_bets"] == 0
        assert result["roi"] == 0.0

    def test_backtest_drawdown_bounded(
        self,
        backtest_setup: tuple[ValueModel, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """Max drawdown should be in [0, 1]."""
        model, X_test, y_test, prices, fees = backtest_setup
        result = backtest_strategy(model, X_test, y_test, prices, fees, edge_threshold=0.01)
        assert 0 <= result["max_drawdown"] <= 1.0

    def test_backtest_bankroll_positive(
        self,
        backtest_setup: tuple[ValueModel, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """Final bankroll should remain positive (Kelly protects from ruin)."""
        model, X_test, y_test, prices, fees = backtest_setup
        result = backtest_strategy(model, X_test, y_test, prices, fees, edge_threshold=0.01)
        assert result["final_bankroll"] > 0
