"""Tests for Whale Wallet Discovery (PM-201).

Tests verify:
1. WhaleWallet creation and filter checks
2. Filtering by win_rate, resolved_positions, volume, combined
3. Leaderboard scraping mock, Data API enrichment, refresh interval, API error fallback
4. Integration: ≥15 wallets, get by address
"""

from __future__ import annotations

import re
import time
from unittest.mock import AsyncMock, patch

import pytest
from aioresponses import aioresponses

from arbo.strategies.whale_discovery import (
    WhaleDiscovery,
    WhaleWallet,
)

# ================================================================
# Factory helpers
# ================================================================


def _make_wallet(
    address: str = "0xabc123",
    win_rate: float = 65.0,
    resolved_positions: int = 100,
    total_volume: float = 100_000.0,
    roi_pct: float = 25.0,
) -> WhaleWallet:
    return WhaleWallet(
        address=address,
        display_name=address[:10],
        roi_pct=roi_pct,
        win_rate=win_rate,
        resolved_positions=resolved_positions,
        total_volume=total_volume,
        discovery_source="test",
    )


def _make_leaderboard_entry(
    address: str = "0xabc123",
    win_rate: float = 65.0,
    resolved: int = 100,
    volume: float = 100_000.0,
    roi: float = 25.0,
) -> dict:
    return {
        "address": address,
        "displayName": address[:10],
        "roi": roi,
        "winRate": win_rate,
        "resolvedPositions": resolved,
        "totalVolume": volume,
    }


# ================================================================
# TestWhaleWallet
# ================================================================


class TestWhaleWallet:
    """WhaleWallet creation and threshold checks."""

    def test_creation(self) -> None:
        """WhaleWallet stores all fields correctly."""
        w = _make_wallet(address="0xdef456", win_rate=70.0, resolved_positions=200)
        assert w.address == "0xdef456"
        assert w.win_rate == 70.0
        assert w.resolved_positions == 200

    def test_passes_filters(self) -> None:
        """Wallet meeting all thresholds passes filters."""
        w = _make_wallet(win_rate=65.0, resolved_positions=100, total_volume=100_000.0)
        assert w.passes_filters() is True

    def test_fails_win_rate(self) -> None:
        """Wallet below win rate threshold fails."""
        w = _make_wallet(win_rate=50.0)
        assert w.passes_filters() is False

    def test_fails_resolved(self) -> None:
        """Wallet below resolved positions threshold fails."""
        w = _make_wallet(resolved_positions=10)
        assert w.passes_filters() is False

    def test_fails_volume(self) -> None:
        """Wallet below volume threshold fails."""
        w = _make_wallet(total_volume=1000.0)
        assert w.passes_filters() is False


# ================================================================
# TestWhaleDiscoveryFiltering
# ================================================================


class TestWhaleDiscoveryFiltering:
    """Filter logic for discovered wallets."""

    def test_win_rate_filter(self) -> None:
        """Only wallets with win_rate >= 60% pass."""
        disc = WhaleDiscovery()
        wallets = [
            _make_wallet(address="0x1", win_rate=55.0),
            _make_wallet(address="0x2", win_rate=65.0),
        ]
        filtered = disc._apply_filters(wallets)
        assert len(filtered) == 1
        assert filtered[0].address == "0x2"

    def test_resolved_filter(self) -> None:
        """Only wallets with resolved >= 50 pass."""
        disc = WhaleDiscovery()
        wallets = [
            _make_wallet(address="0x1", resolved_positions=30),
            _make_wallet(address="0x2", resolved_positions=60),
        ]
        filtered = disc._apply_filters(wallets)
        assert len(filtered) == 1
        assert filtered[0].address == "0x2"

    def test_volume_filter(self) -> None:
        """Only wallets with volume >= $50K pass."""
        disc = WhaleDiscovery()
        wallets = [
            _make_wallet(address="0x1", total_volume=10_000.0),
            _make_wallet(address="0x2", total_volume=75_000.0),
        ]
        filtered = disc._apply_filters(wallets)
        assert len(filtered) == 1
        assert filtered[0].address == "0x2"

    def test_combined_filters(self) -> None:
        """All three filters applied together."""
        disc = WhaleDiscovery()
        wallets = [
            _make_wallet(
                address="0x1", win_rate=55.0, resolved_positions=100, total_volume=100_000.0
            ),
            _make_wallet(
                address="0x2", win_rate=65.0, resolved_positions=30, total_volume=100_000.0
            ),
            _make_wallet(
                address="0x3", win_rate=65.0, resolved_positions=100, total_volume=1_000.0
            ),
            _make_wallet(
                address="0x4", win_rate=70.0, resolved_positions=200, total_volume=200_000.0
            ),
        ]
        filtered = disc._apply_filters(wallets)
        assert len(filtered) == 1
        assert filtered[0].address == "0x4"


# ================================================================
# TestWhaleDiscoveryAPI
# ================================================================


class TestWhaleDiscoveryAPI:
    """API interaction tests with mocked HTTP."""

    @pytest.mark.asyncio
    async def test_leaderboard_scrape(self) -> None:
        """Leaderboard endpoint returns parsed wallets."""
        disc = WhaleDiscovery()
        await disc.initialize()

        entries = [_make_leaderboard_entry(address=f"0x{i:04x}") for i in range(5)]

        with aioresponses() as mocked:
            mocked.get(
                re.compile(r".*/leaderboard/ranked-wallets.*"),
                payload=entries,
            )
            # Second page returns empty to stop pagination
            mocked.get(
                re.compile(r".*/leaderboard/ranked-wallets.*"),
                payload=[],
            )

            wallets = await disc._scrape_leaderboard()
            assert len(wallets) == 5

        await disc.close()

    @pytest.mark.asyncio
    async def test_enrich_updates_data(self) -> None:
        """Enrichment via Data API updates position count and volume."""
        disc = WhaleDiscovery()
        await disc.initialize()

        wallet = _make_wallet(resolved_positions=10, total_volume=1000.0)
        positions = [{"size": "5000"} for _ in range(60)]

        with aioresponses() as mocked:
            mocked.get(
                re.compile(r".*/positions.*"),
                payload=positions,
            )

            enriched = await disc._enrich_via_data_api(wallet)
            assert enriched.resolved_positions == 60
            assert enriched.total_volume >= 5000 * 60

        await disc.close()

    @pytest.mark.asyncio
    async def test_refresh_interval(self) -> None:
        """Refresh respects 7-day interval."""
        disc = WhaleDiscovery()
        disc._wallets = {"0x1": _make_wallet(address="0x1")}
        disc._last_refresh = time.monotonic()  # just refreshed

        # Should return cached wallets without calling discover()
        with patch.object(disc, "discover", new_callable=AsyncMock) as mock_discover:
            result = await disc.refresh_if_stale()
            mock_discover.assert_not_called()
            assert len(result) == 1

    @pytest.mark.asyncio
    async def test_api_error_fallback(self) -> None:
        """API errors return empty list gracefully."""
        disc = WhaleDiscovery()
        await disc.initialize()

        with aioresponses() as mocked:
            mocked.get(
                re.compile(r".*/leaderboard/ranked-wallets.*"),
                status=500,
            )

            wallets = await disc._scrape_leaderboard()
            assert wallets == []

        await disc.close()


# ================================================================
# TestWhaleDiscoveryIntegration
# ================================================================


class TestWhaleDiscoveryIntegration:
    """Integration-style tests for full discover pipeline."""

    @pytest.mark.asyncio
    async def test_discover_returns_filtered_wallets(self) -> None:
        """Full discover pipeline returns ≥15 wallets when leaderboard has enough."""
        disc = WhaleDiscovery()
        await disc.initialize()

        # Create 30 good entries, 5 bad ones
        good_entries = [
            _make_leaderboard_entry(
                address=f"0x{i:04x}",
                win_rate=70.0,
                resolved=100,
                volume=100_000.0,
            )
            for i in range(30)
        ]
        bad_entries = [
            _make_leaderboard_entry(
                address=f"0xbad{i:02x}",
                win_rate=30.0,
                resolved=5,
                volume=500.0,
            )
            for i in range(5)
        ]

        with aioresponses() as mocked:
            # Leaderboard returns mixed entries
            mocked.get(
                re.compile(r".*/leaderboard/ranked-wallets.*"),
                payload=good_entries + bad_entries,
            )
            mocked.get(
                re.compile(r".*/leaderboard/ranked-wallets.*"),
                payload=[],
            )
            # Enrich calls for each wallet
            for _ in range(35):
                mocked.get(
                    re.compile(r".*/positions.*"),
                    payload=[{"size": "5000"} for _ in range(100)],
                )

            result = await disc.discover()
            assert len(result) >= 15

        await disc.close()

    @pytest.mark.asyncio
    async def test_get_wallet_by_address(self) -> None:
        """get_wallet returns specific wallet or None."""
        disc = WhaleDiscovery()
        disc._wallets["0xabc"] = _make_wallet(address="0xabc")

        assert disc.get_wallet("0xabc") is not None
        assert disc.get_wallet("0xabc").address == "0xabc"
        assert disc.get_wallet("0xnonexistent") is None
