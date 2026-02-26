"""Tests for Social Momentum Divergence calculator v2 (B2-13).

Tests for the rewritten calculator using Santiment + CoinGecko metrics.
"""

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
    daily_active_addresses: float = 500000.0,
    transactions_count: float = 300000.0,
    dev_activity: float = 50.0,
    volume_24h: float = 45000000000.0,
    price: float = 95000.0,
    price_change_24h: float = 0.0,
    hours_ago: int = 0,
) -> MomentumSnapshot:
    """Create a test snapshot."""
    return MomentumSnapshot(
        symbol=symbol,
        daily_active_addresses=daily_active_addresses,
        transactions_count=transactions_count,
        dev_activity=dev_activity,
        volume_24h=volume_24h,
        price=price,
        price_change_24h=price_change_24h,
        captured_at=datetime.now(UTC) - timedelta(hours=hours_ago),
    )


# ---------------------------------------------------------------------------
# Tests: Momentum Score Calculation
# ---------------------------------------------------------------------------


class TestMomentumScore:
    """Tests for on-chain momentum score computation."""

    def test_flat_metrics_zero_score(self) -> None:
        """No change in metrics → score near zero."""
        calc = SocialDivergenceCalculator(sigma_threshold=0.1)
        snapshots = {
            "BTC": [
                _make_snapshot(hours_ago=6),
                _make_snapshot(hours_ago=0),
            ]
        }

        signals = calc.calculate_signals(snapshots)

        # With no change, momentum is 0, and no signal should be generated
        assert len(signals) == 0

    def test_onchain_spike_generates_long_signal(self) -> None:
        """On-chain activity spike → positive momentum → LONG signal."""
        calc = SocialDivergenceCalculator(sigma_threshold=0.5)
        snapshots = {
            "SOL": [
                _make_snapshot(
                    symbol="SOL",
                    daily_active_addresses=100000.0,
                    transactions_count=50000.0,
                    dev_activity=30.0,
                    volume_24h=5000000000.0,
                    price_change_24h=0.5,
                    hours_ago=6,
                ),
                _make_snapshot(
                    symbol="SOL",
                    daily_active_addresses=300000.0,  # 3x DAA
                    transactions_count=150000.0,  # 3x tx
                    dev_activity=60.0,  # 2x dev
                    volume_24h=15000000000.0,  # 3x volume
                    price_change_24h=0.5,  # price flat
                    hours_ago=0,
                ),
            ]
        }

        signals = calc.calculate_signals(snapshots)

        assert len(signals) == 1
        assert signals[0].direction == "LONG"
        assert signals[0].social_momentum_score > 0
        assert signals[0].symbol == "SOL"

    def test_onchain_drop_generates_short_signal(self) -> None:
        """On-chain metrics drop with flat price → SHORT signal."""
        calc = SocialDivergenceCalculator(sigma_threshold=0.5)
        snapshots = {
            "DOGE": [
                _make_snapshot(
                    symbol="DOGE",
                    daily_active_addresses=200000.0,
                    transactions_count=100000.0,
                    dev_activity=40.0,
                    volume_24h=10000000000.0,
                    price_change_24h=5.0,
                    hours_ago=6,
                ),
                _make_snapshot(
                    symbol="DOGE",
                    daily_active_addresses=50000.0,  # 75% drop
                    transactions_count=25000.0,  # 75% drop
                    dev_activity=10.0,  # 75% drop
                    volume_24h=2500000000.0,  # 75% drop
                    price_change_24h=5.0,  # price still up
                    hours_ago=0,
                ),
            ]
        }

        signals = calc.calculate_signals(snapshots)

        assert len(signals) == 1
        assert signals[0].direction == "SHORT"
        assert signals[0].social_momentum_score < 0

    def test_no_divergence_when_aligned(self) -> None:
        """On-chain up AND price up → no divergence signal."""
        calc = SocialDivergenceCalculator(sigma_threshold=2.0)
        snapshots = {
            "ETH": [
                _make_snapshot(
                    symbol="ETH",
                    daily_active_addresses=200000.0,
                    transactions_count=100000.0,
                    dev_activity=40.0,
                    volume_24h=10000000000.0,
                    price_change_24h=15.0,
                    hours_ago=6,
                ),
                _make_snapshot(
                    symbol="ETH",
                    daily_active_addresses=400000.0,
                    transactions_count=200000.0,
                    dev_activity=80.0,
                    volume_24h=20000000000.0,
                    price_change_24h=15.0,
                    hours_ago=0,
                ),
            ]
        }

        signals = calc.calculate_signals(snapshots)

        # Both on-chain and price up → aligned → below threshold (likely)
        assert len(signals) == 0


# ---------------------------------------------------------------------------
# Tests: Weight Distribution
# ---------------------------------------------------------------------------


class TestWeightDistribution:
    """Tests for the new weight distribution (DAA x 0.30, TX x 0.20, DEV x 0.10, VOL x 0.40)."""

    def test_volume_dominates(self) -> None:
        """Volume has highest weight (0.40), so volume spike should dominate."""
        calc = SocialDivergenceCalculator(sigma_threshold=0.1)

        # Only volume changes, everything else flat
        snapshots = {
            "BTC": [
                _make_snapshot(
                    daily_active_addresses=500000.0,
                    transactions_count=300000.0,
                    dev_activity=50.0,
                    volume_24h=10000000000.0,
                    hours_ago=6,
                ),
                _make_snapshot(
                    daily_active_addresses=500000.0,  # same
                    transactions_count=300000.0,  # same
                    dev_activity=50.0,  # same
                    volume_24h=30000000000.0,  # 3x volume spike
                    hours_ago=0,
                ),
            ]
        }

        signals = calc.calculate_signals(snapshots)
        assert len(signals) == 1
        assert signals[0].direction == "LONG"
        # Volume-driven momentum should be significant
        assert signals[0].social_momentum_score > 0.3

    def test_dev_activity_smallest_weight(self) -> None:
        """Dev activity has smallest weight (0.10), so dev-only spike is weak."""
        calc = SocialDivergenceCalculator(sigma_threshold=0.5)

        # Only dev_activity changes, everything else flat
        snapshots = {
            "LINK": [
                _make_snapshot(
                    symbol="LINK",
                    daily_active_addresses=100000.0,
                    transactions_count=50000.0,
                    dev_activity=10.0,
                    volume_24h=5000000000.0,
                    hours_ago=6,
                ),
                _make_snapshot(
                    symbol="LINK",
                    daily_active_addresses=100000.0,  # same
                    transactions_count=50000.0,  # same
                    dev_activity=30.0,  # 3x dev spike
                    volume_24h=5000000000.0,  # same
                    hours_ago=0,
                ),
            ]
        }

        signals = calc.calculate_signals(snapshots)
        # With only 0.10 weight, dev-only change may not cross threshold
        if len(signals) > 0:
            assert signals[0].social_momentum_score < 0.3


# ---------------------------------------------------------------------------
# Tests: Threshold Filtering
# ---------------------------------------------------------------------------


class TestThresholdFiltering:
    """Tests for sigma threshold-based signal filtering."""

    def test_below_threshold_no_signal(self) -> None:
        """Divergence below threshold → no signal."""
        calc = SocialDivergenceCalculator(sigma_threshold=99.0)
        snapshots = {
            "BTC": [
                _make_snapshot(
                    daily_active_addresses=500000.0,
                    volume_24h=45000000000.0,
                    hours_ago=6,
                ),
                _make_snapshot(
                    daily_active_addresses=600000.0,
                    volume_24h=50000000000.0,
                    hours_ago=0,
                ),
            ]
        }

        signals = calc.calculate_signals(snapshots)

        assert len(signals) == 0

    def test_above_threshold_generates_signal(self) -> None:
        """Divergence above threshold → signal generated."""
        calc = SocialDivergenceCalculator(sigma_threshold=0.1)
        snapshots = {
            "BTC": [
                _make_snapshot(
                    daily_active_addresses=100000.0,
                    transactions_count=50000.0,
                    volume_24h=5000000000.0,
                    hours_ago=6,
                ),
                _make_snapshot(
                    daily_active_addresses=500000.0,
                    transactions_count=250000.0,
                    volume_24h=25000000000.0,
                    hours_ago=0,
                ),
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
                _make_snapshot(
                    symbol="SOL",
                    daily_active_addresses=100000.0,
                    transactions_count=50000.0,
                    volume_24h=5000000000.0,
                    hours_ago=6,
                ),
                _make_snapshot(
                    symbol="SOL",
                    daily_active_addresses=500000.0,
                    transactions_count=250000.0,
                    volume_24h=25000000000.0,
                    hours_ago=0,
                ),
            ]
        }

        signals = calc.calculate_signals(snapshots)

        assert len(signals) == 1
        assert signals[0].polymarket_condition_ids == ["cond_sol_200", "cond_sol_ath"]

    def test_unmapped_coin_empty_contracts(self) -> None:
        """Unmapped coin → signal with empty contract list."""
        calc = SocialDivergenceCalculator(
            sigma_threshold=0.1,
            coin_mapping={},
        )
        snapshots = {
            "XRP": [
                _make_snapshot(
                    symbol="XRP",
                    daily_active_addresses=50000.0,
                    transactions_count=20000.0,
                    volume_24h=2000000000.0,
                    hours_ago=6,
                ),
                _make_snapshot(
                    symbol="XRP",
                    daily_active_addresses=250000.0,
                    transactions_count=100000.0,
                    volume_24h=10000000000.0,
                    hours_ago=0,
                ),
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

    def test_zero_daa_handled(self) -> None:
        """Zero starting DAA doesn't divide by zero."""
        calc = SocialDivergenceCalculator(sigma_threshold=0.1)
        snapshots = {
            "NEW": [
                _make_snapshot(
                    symbol="NEW",
                    daily_active_addresses=0.0,
                    transactions_count=0.0,
                    dev_activity=0.0,
                    volume_24h=0.0,
                    hours_ago=6,
                ),
                _make_snapshot(
                    symbol="NEW",
                    daily_active_addresses=50000.0,
                    transactions_count=20000.0,
                    dev_activity=10.0,
                    volume_24h=1000000000.0,
                    hours_ago=0,
                ),
            ]
        }

        # Should not raise
        calc.calculate_signals(snapshots)

    def test_zero_volume_handled(self) -> None:
        """Zero starting volume doesn't divide by zero."""
        calc = SocialDivergenceCalculator(sigma_threshold=0.1)
        snapshots = {
            "MICRO": [
                _make_snapshot(
                    symbol="MICRO",
                    daily_active_addresses=1000.0,
                    transactions_count=500.0,
                    dev_activity=5.0,
                    volume_24h=0.0,
                    hours_ago=6,
                ),
                _make_snapshot(
                    symbol="MICRO",
                    daily_active_addresses=2000.0,
                    transactions_count=1000.0,
                    dev_activity=10.0,
                    volume_24h=500000.0,
                    hours_ago=0,
                ),
            ]
        }

        calc.calculate_signals(snapshots)
        # Should not raise

    def test_multiple_coins_independent(self) -> None:
        """Multiple coins analyzed independently."""
        calc = SocialDivergenceCalculator(sigma_threshold=0.1)
        snapshots = {
            "BTC": [
                _make_snapshot(symbol="BTC", hours_ago=6),
                _make_snapshot(symbol="BTC", hours_ago=0),  # flat
            ],
            "SOL": [
                _make_snapshot(
                    symbol="SOL",
                    daily_active_addresses=100000.0,
                    transactions_count=50000.0,
                    volume_24h=5000000000.0,
                    hours_ago=6,
                ),
                _make_snapshot(
                    symbol="SOL",
                    daily_active_addresses=500000.0,
                    transactions_count=250000.0,
                    volume_24h=25000000000.0,
                    hours_ago=0,
                ),
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
                _make_snapshot(
                    daily_active_addresses=100000.0,
                    volume_24h=5000000000.0,
                    hours_ago=6,
                ),
                _make_snapshot(
                    daily_active_addresses=500000.0,
                    volume_24h=25000000000.0,
                    hours_ago=0,
                ),
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

    def test_snapshot_has_v2_fields(self) -> None:
        """MomentumSnapshot v2 has on-chain metric fields."""
        snap = _make_snapshot(
            daily_active_addresses=500000.0,
            transactions_count=300000.0,
            dev_activity=50.0,
            volume_24h=45000000000.0,
        )
        assert snap.daily_active_addresses == 500000.0
        assert snap.transactions_count == 300000.0
        assert snap.dev_activity == 50.0
        assert snap.volume_24h == 45000000000.0
