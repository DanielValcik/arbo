"""Tests for Whale Position Monitor (PM-202).

Tests verify:
1. Position diffing: new, increased, decreased, closed
2. Signal generation: direction, layer, details
3. Multi-whale confluence: boosted confidence when ≥2 whales
4. Polling cycle: full cycle with changes, no changes
"""

from __future__ import annotations

import re
from decimal import Decimal

import pytest
from aioresponses import aioresponses

from arbo.core.scanner import SignalDirection
from arbo.strategies.whale_discovery import WhaleDiscovery, WhaleWallet
from arbo.strategies.whale_monitor import (
    ChangeType,
    PositionChange,
    WalletPosition,
    WhaleMonitor,
)

# ================================================================
# Helpers
# ================================================================


def _make_wallet(address: str = "0xwhale1") -> WhaleWallet:
    return WhaleWallet(
        address=address,
        display_name=address[:10],
        roi_pct=25.0,
        win_rate=70.0,
        resolved_positions=100,
        total_volume=100_000.0,
        discovery_source="test",
    )


def _make_position(
    address: str = "0xwhale1",
    condition_id: str = "cond_1",
    token_id: str = "tok_1",
    size: str = "1000",
    avg_price: str = "0.60",
) -> WalletPosition:
    return WalletPosition(
        address=address,
        condition_id=condition_id,
        token_id=token_id,
        size=Decimal(size),
        avg_price=Decimal(avg_price),
        outcome="Yes",
    )


def _make_discovery(wallets: list[WhaleWallet] | None = None) -> WhaleDiscovery:
    disc = WhaleDiscovery()
    if wallets:
        disc._wallets = {w.address: w for w in wallets}
    return disc


# ================================================================
# TestPositionDiff
# ================================================================


class TestPositionDiff:
    """Position diff detection."""

    def test_new_position_detected(self) -> None:
        """New position not in old snapshot → NEW change."""
        disc = _make_discovery()
        monitor = WhaleMonitor(disc)

        new_positions = [_make_position(token_id="tok_1", size="1000")]
        changes = monitor._diff_positions("0xwhale1", new_positions)

        assert len(changes) == 1
        assert changes[0].change_type == ChangeType.NEW
        assert changes[0].new_size == Decimal("1000")
        assert changes[0].old_size == Decimal("0")

    def test_increased_position_detected(self) -> None:
        """Position size increased → INCREASED change."""
        disc = _make_discovery()
        monitor = WhaleMonitor(disc)

        # Set up old snapshot
        old_pos = _make_position(token_id="tok_1", size="500")
        monitor._positions["0xwhale1"] = {"tok_1": old_pos}

        # New snapshot with bigger size
        new_positions = [_make_position(token_id="tok_1", size="1000")]
        changes = monitor._diff_positions("0xwhale1", new_positions)

        assert len(changes) == 1
        assert changes[0].change_type == ChangeType.INCREASED
        assert changes[0].delta == Decimal("500")

    def test_decreased_position_detected(self) -> None:
        """Position size decreased → DECREASED change."""
        disc = _make_discovery()
        monitor = WhaleMonitor(disc)

        old_pos = _make_position(token_id="tok_1", size="1000")
        monitor._positions["0xwhale1"] = {"tok_1": old_pos}

        new_positions = [_make_position(token_id="tok_1", size="500")]
        changes = monitor._diff_positions("0xwhale1", new_positions)

        assert len(changes) == 1
        assert changes[0].change_type == ChangeType.DECREASED
        assert changes[0].delta == Decimal("-500")

    def test_closed_position_detected(self) -> None:
        """Position disappears from new snapshot → CLOSED change."""
        disc = _make_discovery()
        monitor = WhaleMonitor(disc)

        old_pos = _make_position(token_id="tok_1", size="1000")
        monitor._positions["0xwhale1"] = {"tok_1": old_pos}

        new_positions: list[WalletPosition] = []  # Position gone
        changes = monitor._diff_positions("0xwhale1", new_positions)

        assert len(changes) == 1
        assert changes[0].change_type == ChangeType.CLOSED
        assert changes[0].new_size == Decimal("0")


# ================================================================
# TestSignalGeneration
# ================================================================


class TestSignalGeneration:
    """Signal generation from position changes."""

    def test_new_generates_signal(self) -> None:
        """NEW position change generates a signal."""
        disc = _make_discovery()
        monitor = WhaleMonitor(disc)

        changes = [
            PositionChange(
                address="0xwhale1",
                condition_id="cond_1",
                token_id="tok_1",
                change_type=ChangeType.NEW,
                old_size=Decimal("0"),
                new_size=Decimal("1000"),
                delta=Decimal("1000"),
            )
        ]
        signals = monitor._changes_to_signals(changes)
        assert len(signals) == 1

    def test_signal_direction_mirrors_whale(self) -> None:
        """Signal direction should mirror whale's BUY."""
        disc = _make_discovery()
        monitor = WhaleMonitor(disc)

        changes = [
            PositionChange(
                address="0xwhale1",
                condition_id="cond_1",
                token_id="tok_1",
                change_type=ChangeType.NEW,
                old_size=Decimal("0"),
                new_size=Decimal("1000"),
                delta=Decimal("1000"),
            )
        ]
        signals = monitor._changes_to_signals(changes)
        assert signals[0].direction == SignalDirection.BUY_YES

    def test_signal_layer_is_4(self) -> None:
        """Whale signals should be Layer 4."""
        disc = _make_discovery()
        monitor = WhaleMonitor(disc)

        changes = [
            PositionChange(
                address="0xwhale1",
                condition_id="cond_1",
                token_id="tok_1",
                change_type=ChangeType.INCREASED,
                old_size=Decimal("500"),
                new_size=Decimal("1000"),
                delta=Decimal("500"),
            )
        ]
        signals = monitor._changes_to_signals(changes)
        assert signals[0].layer == 4

    def test_signal_details_contain_address(self) -> None:
        """Signal details should contain whale address."""
        disc = _make_discovery()
        monitor = WhaleMonitor(disc)

        changes = [
            PositionChange(
                address="0xwhale1",
                condition_id="cond_1",
                token_id="tok_1",
                change_type=ChangeType.NEW,
                old_size=Decimal("0"),
                new_size=Decimal("1000"),
                delta=Decimal("1000"),
            )
        ]
        signals = monitor._changes_to_signals(changes)
        assert signals[0].details["whale_address"] == "0xwhale1"

    def test_decreased_no_signal(self) -> None:
        """DECREASED changes should NOT generate signals."""
        disc = _make_discovery()
        monitor = WhaleMonitor(disc)

        changes = [
            PositionChange(
                address="0xwhale1",
                condition_id="cond_1",
                token_id="tok_1",
                change_type=ChangeType.DECREASED,
                old_size=Decimal("1000"),
                new_size=Decimal("500"),
                delta=Decimal("-500"),
            )
        ]
        signals = monitor._changes_to_signals(changes)
        assert len(signals) == 0


# ================================================================
# TestMultiWhaleConfluence
# ================================================================


class TestMultiWhaleConfluence:
    """Multi-whale confluence detection."""

    def test_two_whales_strong_signal(self) -> None:
        """≥2 whales in same market → boosted signal."""
        disc = _make_discovery()
        monitor = WhaleMonitor(disc)

        changes = [
            PositionChange(
                address="0xwhale1",
                condition_id="cond_1",
                token_id="tok_1",
                change_type=ChangeType.NEW,
                old_size=Decimal("0"),
                new_size=Decimal("1000"),
                delta=Decimal("1000"),
            ),
            PositionChange(
                address="0xwhale2",
                condition_id="cond_1",
                token_id="tok_1",
                change_type=ChangeType.INCREASED,
                old_size=Decimal("500"),
                new_size=Decimal("1500"),
                delta=Decimal("1000"),
            ),
        ]
        signals = monitor._check_multi_whale(changes)
        assert len(signals) == 1
        assert signals[0].details["multi_whale"] is True
        assert signals[0].details["whale_count"] == 2

    def test_one_whale_no_boost(self) -> None:
        """Single whale in market → no multi-whale signal."""
        disc = _make_discovery()
        monitor = WhaleMonitor(disc)

        changes = [
            PositionChange(
                address="0xwhale1",
                condition_id="cond_1",
                token_id="tok_1",
                change_type=ChangeType.NEW,
                old_size=Decimal("0"),
                new_size=Decimal("1000"),
                delta=Decimal("1000"),
            ),
        ]
        signals = monitor._check_multi_whale(changes)
        assert len(signals) == 0

    def test_boosted_confidence(self) -> None:
        """Multi-whale signal has confidence 0.85 (vs 0.65 normal)."""
        disc = _make_discovery()
        monitor = WhaleMonitor(disc)

        changes = [
            PositionChange(
                address="0xwhale1",
                condition_id="cond_1",
                token_id="tok_1",
                change_type=ChangeType.NEW,
                old_size=Decimal("0"),
                new_size=Decimal("1000"),
                delta=Decimal("1000"),
            ),
            PositionChange(
                address="0xwhale2",
                condition_id="cond_1",
                token_id="tok_1",
                change_type=ChangeType.NEW,
                old_size=Decimal("0"),
                new_size=Decimal("2000"),
                delta=Decimal("2000"),
            ),
        ]
        signals = monitor._check_multi_whale(changes)
        assert signals[0].confidence == Decimal("0.85")


# ================================================================
# TestPollingCycle
# ================================================================


class TestPollingCycle:
    """Full polling cycle tests."""

    @pytest.mark.asyncio
    async def test_poll_with_changes(self) -> None:
        """Poll cycle with position changes generates signals."""
        wallet = _make_wallet("0xwhale1")
        disc = _make_discovery([wallet])
        monitor = WhaleMonitor(disc)
        await monitor.initialize()

        positions_data = [
            {
                "conditionId": "cond_1",
                "tokenId": "tok_1",
                "size": "1000",
                "avgPrice": "0.60",
                "outcome": "Yes",
            }
        ]

        with aioresponses() as mocked:
            mocked.get(
                re.compile(r".*/positions.*"),
                payload=positions_data,
            )

            signals = await monitor.poll_cycle()
            # First poll: everything is NEW
            assert len(signals) >= 1
            assert any(s.layer == 4 for s in signals)

        await monitor.close()

    @pytest.mark.asyncio
    async def test_poll_no_changes_no_signals(self) -> None:
        """Poll cycle without changes generates no signals."""
        wallet = _make_wallet("0xwhale1")
        disc = _make_discovery([wallet])
        monitor = WhaleMonitor(disc)
        await monitor.initialize()

        # Pre-populate with existing position
        monitor._positions["0xwhale1"] = {
            "tok_1": _make_position(address="0xwhale1", token_id="tok_1", size="1000")
        }

        # Same positions returned (no change)
        positions_data = [
            {
                "conditionId": "cond_1",
                "tokenId": "tok_1",
                "size": "1000",
                "avgPrice": "0.60",
                "outcome": "Yes",
            }
        ]

        with aioresponses() as mocked:
            mocked.get(
                re.compile(r".*/positions.*"),
                payload=positions_data,
            )

            signals = await monitor.poll_cycle()
            assert len(signals) == 0

        await monitor.close()
