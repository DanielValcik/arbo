"""Tests for Social Momentum Divergence calculator (B2-03)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from arbo.strategies.social_divergence import (
    DivergenceSignal,
    MomentumSnapshot,
    SocialDivergenceCalculator,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_snapshot(
    symbol: str = "BTC",
    social_dominance: float = 25.0,
    sentiment: int = 60,
    interactions_24h: int = 5000000,
    price: float = 95000.0,
    percent_change_24h: float = 0.0,
    hours_ago: int = 0,
) -> MomentumSnapshot:
    """Create a test snapshot."""
    return MomentumSnapshot(
        symbol=symbol,
        social_dominance=social_dominance,
        sentiment=sentiment,
        interactions_24h=interactions_24h,
        price=price,
        percent_change_24h=percent_change_24h,
        captured_at=datetime.now(UTC) - timedelta(hours=hours_ago),
    )


# ---------------------------------------------------------------------------
# Tests: Momentum Score Calculation
# ---------------------------------------------------------------------------


class TestMomentumScore:
    """Tests for social momentum score computation."""

    def test_flat_metrics_zero_score(self) -> None:
        """No change in metrics → score near zero."""
        calc = SocialDivergenceCalculator(sigma_threshold=0.1)
        snapshots = {
            "BTC": [
                _make_snapshot(social_dominance=25.0, sentiment=60, interactions_24h=5000000, hours_ago=6),
                _make_snapshot(social_dominance=25.0, sentiment=60, interactions_24h=5000000, hours_ago=0),
            ]
        }

        signals = calc.calculate_signals(snapshots)

        # With no change, momentum is 0, and no signal should be generated
        # (depends on threshold, but with flat data divergence ~ 0)
        # We're testing the momentum calculation indirectly
        assert len(signals) == 0

    def test_social_spike_generates_long_signal(self) -> None:
        """Social metrics spike → positive momentum → LONG signal."""
        calc = SocialDivergenceCalculator(sigma_threshold=0.5)
        snapshots = {
            "SOL": [
                _make_snapshot(
                    symbol="SOL", social_dominance=5.0, sentiment=50,
                    interactions_24h=1000000, percent_change_24h=0.5, hours_ago=6,
                ),
                _make_snapshot(
                    symbol="SOL", social_dominance=15.0, sentiment=80,
                    interactions_24h=3000000, percent_change_24h=0.5, hours_ago=0,
                ),
            ]
        }

        signals = calc.calculate_signals(snapshots)

        assert len(signals) == 1
        assert signals[0].direction == "LONG"
        assert signals[0].social_momentum_score > 0
        assert signals[0].symbol == "SOL"

    def test_social_drop_generates_short_signal(self) -> None:
        """Social metrics drop with flat price → SHORT signal."""
        calc = SocialDivergenceCalculator(sigma_threshold=0.5)
        snapshots = {
            "DOGE": [
                _make_snapshot(
                    symbol="DOGE", social_dominance=10.0, sentiment=70,
                    interactions_24h=2000000, percent_change_24h=5.0, hours_ago=6,
                ),
                _make_snapshot(
                    symbol="DOGE", social_dominance=3.0, sentiment=30,
                    interactions_24h=500000, percent_change_24h=5.0, hours_ago=0,
                ),
            ]
        }

        signals = calc.calculate_signals(snapshots)

        assert len(signals) == 1
        assert signals[0].direction == "SHORT"
        assert signals[0].social_momentum_score < 0

    def test_no_divergence_when_aligned(self) -> None:
        """Social up AND price up → no divergence signal."""
        calc = SocialDivergenceCalculator(sigma_threshold=2.0)
        snapshots = {
            "ETH": [
                _make_snapshot(
                    symbol="ETH", social_dominance=10.0, sentiment=50,
                    interactions_24h=1000000, percent_change_24h=15.0, hours_ago=6,
                ),
                _make_snapshot(
                    symbol="ETH", social_dominance=20.0, sentiment=75,
                    interactions_24h=3000000, percent_change_24h=15.0, hours_ago=0,
                ),
            ]
        }

        signals = calc.calculate_signals(snapshots)

        # Both social and price up → aligned → below threshold (likely)
        # The divergence should be small
        assert len(signals) == 0


# ---------------------------------------------------------------------------
# Tests: Threshold Filtering
# ---------------------------------------------------------------------------


class TestThresholdFiltering:
    """Tests for sigma threshold-based signal filtering."""

    def test_below_threshold_no_signal(self) -> None:
        """Divergence below threshold → no signal."""
        calc = SocialDivergenceCalculator(sigma_threshold=99.0)  # Very high threshold
        snapshots = {
            "BTC": [
                _make_snapshot(social_dominance=25.0, sentiment=60, hours_ago=6),
                _make_snapshot(social_dominance=30.0, sentiment=65, hours_ago=0),
            ]
        }

        signals = calc.calculate_signals(snapshots)

        assert len(signals) == 0

    def test_above_threshold_generates_signal(self) -> None:
        """Divergence above threshold → signal generated."""
        calc = SocialDivergenceCalculator(sigma_threshold=0.1)  # Very low threshold
        snapshots = {
            "BTC": [
                _make_snapshot(social_dominance=10.0, sentiment=40, interactions_24h=1000000, hours_ago=6),
                _make_snapshot(social_dominance=30.0, sentiment=80, interactions_24h=5000000, hours_ago=0),
            ]
        }

        signals = calc.calculate_signals(snapshots)

        assert len(signals) == 1


# ---------------------------------------------------------------------------
# Tests: Coin Mapping
# ---------------------------------------------------------------------------


class TestCoinMapping:
    """Tests for coin → Polymarket contract mapping."""

    def test_signal_includes_pm_contracts(self) -> None:
        """Signal includes mapped PM condition IDs."""
        mapping = {"SOL": ["cond_sol_200", "cond_sol_ath"]}
        calc = SocialDivergenceCalculator(
            sigma_threshold=0.1,
            coin_mapping=mapping,
        )
        snapshots = {
            "SOL": [
                _make_snapshot(symbol="SOL", social_dominance=5.0, sentiment=40, hours_ago=6),
                _make_snapshot(symbol="SOL", social_dominance=20.0, sentiment=80, hours_ago=0),
            ]
        }

        signals = calc.calculate_signals(snapshots)

        assert len(signals) == 1
        assert signals[0].polymarket_condition_ids == ["cond_sol_200", "cond_sol_ath"]

    def test_unmapped_coin_empty_contracts(self) -> None:
        """Unmapped coin → signal with empty contract list."""
        calc = SocialDivergenceCalculator(
            sigma_threshold=0.1,
            coin_mapping={},  # No mappings
        )
        snapshots = {
            "XRP": [
                _make_snapshot(symbol="XRP", social_dominance=3.0, sentiment=40, hours_ago=6),
                _make_snapshot(symbol="XRP", social_dominance=15.0, sentiment=80, hours_ago=0),
            ]
        }

        signals = calc.calculate_signals(snapshots)

        assert len(signals) == 1
        assert signals[0].polymarket_condition_ids == []

    def test_add_coin_mapping(self) -> None:
        """add_coin_mapping() updates mapping."""
        calc = SocialDivergenceCalculator()
        calc.add_coin_mapping("AVAX", ["cond_avax_1"])

        assert "AVAX" in calc._coin_mapping
        assert calc._coin_mapping["AVAX"] == ["cond_avax_1"]

    def test_set_coin_mapping_replaces(self) -> None:
        """set_coin_mapping() replaces entire mapping."""
        calc = SocialDivergenceCalculator(coin_mapping={"OLD": ["old_cond"]})
        calc.set_coin_mapping({"NEW": ["new_cond"]})

        assert "OLD" not in calc._coin_mapping
        assert "NEW" in calc._coin_mapping


# ---------------------------------------------------------------------------
# Tests: Edge Cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_insufficient_snapshots_skipped(self) -> None:
        """Coins with < min_snapshots are skipped."""
        calc = SocialDivergenceCalculator(min_snapshots=3)
        snapshots = {
            "BTC": [
                _make_snapshot(hours_ago=6),
                _make_snapshot(hours_ago=0),
            ]
        }

        signals = calc.calculate_signals(snapshots)

        assert len(signals) == 0

    def test_zero_social_dominance_handled(self) -> None:
        """Zero starting social dominance doesn't divide by zero."""
        calc = SocialDivergenceCalculator(sigma_threshold=0.1)
        snapshots = {
            "NEW": [
                _make_snapshot(symbol="NEW", social_dominance=0.0, sentiment=50, interactions_24h=0, hours_ago=6),
                _make_snapshot(symbol="NEW", social_dominance=5.0, sentiment=70, interactions_24h=10000, hours_ago=0),
            ]
        }

        # Should not raise
        signals = calc.calculate_signals(snapshots)
        # May or may not generate a signal depending on threshold

    def test_zero_interactions_handled(self) -> None:
        """Zero starting interactions doesn't divide by zero."""
        calc = SocialDivergenceCalculator(sigma_threshold=0.1)
        snapshots = {
            "MICRO": [
                _make_snapshot(symbol="MICRO", social_dominance=1.0, sentiment=50, interactions_24h=0, hours_ago=6),
                _make_snapshot(symbol="MICRO", social_dominance=2.0, sentiment=60, interactions_24h=500, hours_ago=0),
            ]
        }

        signals = calc.calculate_signals(snapshots)
        # Should not raise

    def test_multiple_coins_independent(self) -> None:
        """Multiple coins analyzed independently."""
        calc = SocialDivergenceCalculator(sigma_threshold=0.1)
        snapshots = {
            "BTC": [
                _make_snapshot(symbol="BTC", social_dominance=25.0, sentiment=60, hours_ago=6),
                _make_snapshot(symbol="BTC", social_dominance=25.0, sentiment=60, hours_ago=0),
            ],
            "SOL": [
                _make_snapshot(symbol="SOL", social_dominance=5.0, sentiment=40, hours_ago=6),
                _make_snapshot(symbol="SOL", social_dominance=20.0, sentiment=85, hours_ago=0),
            ],
        }

        signals = calc.calculate_signals(snapshots)

        # BTC is flat (no signal), SOL has big spike
        sol_signals = [s for s in signals if s.symbol == "SOL"]
        assert len(sol_signals) >= 1


# ---------------------------------------------------------------------------
# Tests: Stats
# ---------------------------------------------------------------------------


class TestStats:
    """Tests for calculator statistics."""

    def test_stats_after_analysis(self) -> None:
        """Stats reflect analysis activity."""
        calc = SocialDivergenceCalculator(sigma_threshold=0.1)
        snapshots = {
            "BTC": [
                _make_snapshot(social_dominance=10.0, sentiment=40, hours_ago=6),
                _make_snapshot(social_dominance=30.0, sentiment=80, hours_ago=0),
            ],
        }

        calc.calculate_signals(snapshots)
        stats = calc.stats

        assert stats["coins_analyzed"] >= 1
        assert stats["sigma_threshold"] == 0.1

    def test_stats_initial(self) -> None:
        """Initial stats are zeroed."""
        calc = SocialDivergenceCalculator()
        stats = calc.stats

        assert stats["signals_generated"] == 0
        assert stats["coins_analyzed"] == 0


# ---------------------------------------------------------------------------
# Tests: Data Models
# ---------------------------------------------------------------------------


class TestDataModels:
    """Tests for DivergenceSignal dataclass."""

    def test_signal_frozen(self) -> None:
        """DivergenceSignal is immutable."""
        signal = DivergenceSignal(
            symbol="BTC",
            social_momentum_score=0.5,
            price_momentum=1.0,
            divergence=0.3,
            z_score=2.5,
            direction="LONG",
            confidence=0.7,
            polymarket_condition_ids=["cond_1"],
        )
        with pytest.raises(AttributeError):
            signal.symbol = "ETH"  # type: ignore[misc]

    def test_signal_to_dict(self) -> None:
        """to_dict() serializes all fields."""
        signal = DivergenceSignal(
            symbol="SOL",
            social_momentum_score=0.65,
            price_momentum=-2.3,
            divergence=0.45,
            z_score=3.1,
            direction="LONG",
            confidence=0.82,
            polymarket_condition_ids=["cond_sol_1"],
        )
        d = signal.to_dict()

        assert d["symbol"] == "SOL"
        assert d["direction"] == "LONG"
        assert d["z_score"] == 3.1
        assert d["polymarket_condition_ids"] == ["cond_sol_1"]
        assert "detected_at" in d

    def test_snapshot_frozen(self) -> None:
        """MomentumSnapshot is immutable."""
        snap = _make_snapshot()
        with pytest.raises(AttributeError):
            snap.symbol = "ETH"  # type: ignore[misc]
