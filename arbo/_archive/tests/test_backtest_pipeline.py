"""Tests for backtest pipeline: process_data and run_backtest.

Tests verify:
1. Training data shape correct (3 rows per match)
2. Simulated prices bounded [0.02, 0.98]
3. Time-series split no leakage (train dates < test dates)
4. Outcome encoding correct
5. Full pipeline on synthetic data completes
6. Report contains required fields
7. Feature columns match FEATURE_COLUMNS
8. Fee rates correct for soccer (0.0)
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

from arbo.models.feature_engineering import FEATURE_COLUMNS

# Add project root to sys.path before importing scripts
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.backfill_data import BacktestDataset, HistoricalMatch
from scripts.process_data import build_training_data
from scripts.run_backtest import (
    GATES,
    check_gate,
    generate_report,
    run_pipeline,
    time_series_split,
)

# ================================================================
# Fixtures
# ================================================================


def _make_match(
    match_id: str = "test_001",
    date: str = "2025-06-01T15:00:00Z",
    home: str = "Arsenal",
    away: str = "Chelsea",
    home_goals: int = 2,
    away_goals: int = 1,
    result: str = "H",
    home_odds: float = 2.10,
    draw_odds: float = 3.40,
    away_odds: float = 3.50,
) -> HistoricalMatch:
    """Create a test HistoricalMatch."""
    return HistoricalMatch(
        match_id=match_id,
        date=date,
        league="Premier League",
        season="2425",
        home_team=home,
        away_team=away,
        home_goals=home_goals,
        away_goals=away_goals,
        result=result,
        fd_home_odds=home_odds,
        fd_draw_odds=draw_odds,
        fd_away_odds=away_odds,
    )


def _make_dataset(n_matches: int = 50, seed: int = 42) -> BacktestDataset:
    """Create a synthetic BacktestDataset for testing."""
    rng = np.random.default_rng(seed)
    matches: list[HistoricalMatch] = []

    for i in range(n_matches):
        # Spread dates over 6 months
        day_offset = i * 3  # ~3 days apart
        date = f"2025-{1 + (day_offset // 30) % 12:02d}-{1 + day_offset % 28:02d}T15:00:00Z"

        home_goals = int(rng.poisson(1.5))
        away_goals = int(rng.poisson(1.2))
        if home_goals > away_goals:
            result = "H"
        elif away_goals > home_goals:
            result = "A"
        else:
            result = "D"

        home_odds = float(rng.uniform(1.5, 4.0))
        draw_odds = float(rng.uniform(2.5, 5.0))
        away_odds = float(rng.uniform(1.8, 5.0))

        matches.append(
            _make_match(
                match_id=f"test_{i:04d}",
                date=date,
                home=f"Team_H_{i}",
                away=f"Team_A_{i}",
                home_goals=home_goals,
                away_goals=away_goals,
                result=result,
                home_odds=home_odds,
                draw_odds=draw_odds,
                away_odds=away_odds,
            )
        )

    return BacktestDataset(
        n_matches=len(matches),
        leagues=["Premier League"],
        seasons=["2425"],
        matches=matches,
    )


# ================================================================
# Training data shape
# ================================================================


class TestTrainingDataShape:
    """Training data should have 3 rows per match."""

    def test_three_rows_per_match(self) -> None:
        """Each match produces exactly 3 entries (H, A, D)."""
        dataset = _make_dataset(n_matches=10)
        _X, y, _prices, _fees = build_training_data(dataset)
        assert len(y) == 30  # 10 matches * 3

    def test_single_match_three_entries(self) -> None:
        """A single match should produce 3 entries."""
        match = _make_match()
        dataset = BacktestDataset(
            n_matches=1,
            leagues=["Premier League"],
            seasons=["2425"],
            matches=[match],
        )
        X, y, _prices, _fees = build_training_data(dataset)
        assert len(y) == 3
        assert X.shape[0] == 3


# ================================================================
# Simulated prices bounded
# ================================================================


class TestSimulatedPrices:
    """Simulated Polymarket prices must be bounded [0.02, 0.98]."""

    def test_prices_bounded(self) -> None:
        """All simulated prices should be within [0.02, 0.98]."""
        dataset = _make_dataset(n_matches=100)
        _X, _y, prices, _fees = build_training_data(dataset)
        assert prices.min() >= 0.02
        assert prices.max() <= 0.98

    def test_prices_not_degenerate(self) -> None:
        """Prices should have reasonable variance (not all the same)."""
        dataset = _make_dataset(n_matches=50)
        _X, _y, prices, _fees = build_training_data(dataset)
        assert prices.std() > 0.01


# ================================================================
# Time-series split no leakage
# ================================================================


class TestTimeSeriesSplit:
    """Train dates must be strictly before test dates."""

    def test_no_temporal_leakage(self) -> None:
        """All training match dates must precede all test match dates."""
        dataset = _make_dataset(n_matches=50)
        train_ds, test_ds = time_series_split(dataset, train_fraction=0.7)

        if train_ds.matches and test_ds.matches:
            max_train_date = max(m.date for m in train_ds.matches)
            min_test_date = min(m.date for m in test_ds.matches)
            assert max_train_date <= min_test_date

    def test_split_sizes(self) -> None:
        """Split should respect the 70/30 ratio."""
        dataset = _make_dataset(n_matches=100)
        train_ds, test_ds = time_series_split(dataset, train_fraction=0.7)
        assert train_ds.n_matches == 70
        assert test_ds.n_matches == 30


# ================================================================
# Outcome encoding
# ================================================================


class TestOutcomeEncoding:
    """Binary outcome encoding must be correct."""

    def test_exactly_one_positive_per_match(self) -> None:
        """Each match should have exactly 1 positive outcome among its 3 entries."""
        dataset = _make_dataset(n_matches=20)
        _X, y, _prices, _fees = build_training_data(dataset)

        # Every 3 rows should sum to exactly 1
        for i in range(0, len(y), 3):
            group_sum = y[i : i + 3].sum()
            assert group_sum == 1, f"Match group at index {i} has {group_sum} positives"

    def test_outcome_values_binary(self) -> None:
        """All outcomes should be 0 or 1."""
        dataset = _make_dataset(n_matches=20)
        _X, y, _prices, _fees = build_training_data(dataset)
        assert set(np.unique(y)).issubset({0, 1})


# ================================================================
# Full pipeline
# ================================================================


class TestFullPipeline:
    """Full pipeline on synthetic data should complete."""

    def test_pipeline_completes(self) -> None:
        """Run full pipeline on synthetic dataset and get a report."""
        dataset = _make_dataset(n_matches=100)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            input_path = tmpdir_path / "dataset.json"
            output_path = tmpdir_path / "model.joblib"

            # Save dataset
            with open(input_path, "w") as f:
                json.dump(dataset.model_dump(), f, default=str)

            report = run_pipeline(
                input_path=input_path,
                output_path=output_path,
                tune=False,
            )

            assert report is not None
            assert "overall_verdict" in report
            assert report["overall_verdict"] in ("PASS", "FAIL")
            assert output_path.exists()


# ================================================================
# Report fields
# ================================================================


class TestReportFields:
    """Report must contain all required fields."""

    def test_report_has_gates(self) -> None:
        """Report should have gates section with pass/fail."""
        report = generate_report(
            eval_metrics={"brier_score": 0.20, "accuracy": 0.65, "n_samples": 300},
            backtest_results={
                "roi": 0.05,
                "n_bets": 250,
                "final_bankroll": 10500,
                "max_drawdown": 0.15,
                "sharpe": 1.2,
                "avg_edge": 0.04,
            },
            train_metrics={
                "brier_train": 0.18,
                "brier_cal_calibrated": 0.19,
                "n_train": 700,
                "n_cal": 100,
            },
            n_train_matches=500,
            n_test_matches=200,
            tuned=False,
        )

        assert "overall_verdict" in report
        assert "gates" in report
        assert "evaluation" in report
        assert "backtest" in report
        assert "training" in report
        assert "data" in report
        assert "config" in report

        # Check all gates are present
        for gate_name in GATES:
            assert gate_name in report["gates"]
            assert "value" in report["gates"][gate_name]
            assert "passed" in report["gates"][gate_name]

    def test_passing_report(self) -> None:
        """All gates should pass with good metrics."""
        report = generate_report(
            eval_metrics={"brier_score": 0.18, "accuracy": 0.70, "n_samples": 500},
            backtest_results={
                "roi": 0.08,
                "n_bets": 300,
                "final_bankroll": 10800,
                "max_drawdown": 0.10,
                "sharpe": 1.5,
                "avg_edge": 0.05,
            },
            train_metrics={
                "brier_train": 0.15,
                "brier_cal_calibrated": 0.16,
                "n_train": 1000,
                "n_cal": 150,
            },
            n_train_matches=700,
            n_test_matches=300,
            tuned=True,
        )

        assert report["overall_verdict"] == "PASS"

    def test_check_gate_below(self) -> None:
        """Gate with direction 'below' should pass when value < threshold."""
        assert check_gate("brier_score", 0.20) is True
        assert check_gate("brier_score", 0.25) is False

    def test_check_gate_above(self) -> None:
        """Gate with direction 'above' should pass when value >= threshold."""
        assert check_gate("roi", 0.05) is True
        assert check_gate("roi", 0.01) is False


# ================================================================
# Feature columns match
# ================================================================


class TestFeatureColumns:
    """Feature columns in training data must match FEATURE_COLUMNS."""

    def test_columns_match(self) -> None:
        """DataFrame columns should exactly match FEATURE_COLUMNS."""
        dataset = _make_dataset(n_matches=10)
        X, _y, _prices, _fees = build_training_data(dataset)
        assert list(X.columns) == FEATURE_COLUMNS


# ================================================================
# Fee rates
# ================================================================


class TestFeeRates:
    """Fee rates must be correct for soccer (0.0)."""

    def test_soccer_fees_zero(self) -> None:
        """All fee rates for soccer matches should be 0.0."""
        dataset = _make_dataset(n_matches=20)
        _X, _y, _prices, fees = build_training_data(dataset)
        assert np.all(fees == 0.0)
