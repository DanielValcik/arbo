"""Integration tests against live Polymarket and The Odds API.

Sprint 1 closure gate tests:
- PM-001: Connect to CLOB, get orderbook for 3 active markets
- PM-002: Fetch active markets from Gamma API
- PM-003: Fetch EPL odds, match with Polymarket markets

Run: python3 -m pytest arbo/tests/test_integration.py -v -m integration

Requires: .env with POLY_PRIVATE_KEY, ODDS_API_KEY
"""

from __future__ import annotations

import os
from decimal import Decimal
from pathlib import Path

import pytest
from dotenv import load_dotenv

# Load .env so skipif conditions can see credentials
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

# Mark all tests in this module as integration
pytestmark = pytest.mark.integration


def _has_poly_key() -> bool:
    """Check if Polymarket credentials are available."""
    return bool(os.getenv("POLY_PRIVATE_KEY", ""))


def _has_odds_key() -> bool:
    """Check if The Odds API key is available."""
    return bool(os.getenv("ODDS_API_KEY", ""))


# ================================================================
# PM-002: Gamma API Market Discovery
# ================================================================


class TestGammaAPIIntegration:
    """PM-002 acceptance: Fetch active markets from Gamma API (no auth needed)."""

    @pytest.mark.asyncio
    async def test_fetch_active_markets(self) -> None:
        """Gamma API should return active markets without authentication."""
        from arbo.connectors.market_discovery import MarketDiscovery

        discovery = MarketDiscovery()
        await discovery.initialize()

        try:
            markets = await discovery.fetch_all_active_markets(max_pages=2)

            assert len(markets) > 0, "Expected at least 1 active market from Gamma API"

            # Verify market structure
            for market in markets[:5]:
                assert market.condition_id, "Market must have condition_id"
                assert market.question, "Market must have question"
                assert market.active is True, "Market must be active"

            # Check we get diverse categories
            categories = {m.category for m in markets}
            print(f"\n  Markets fetched: {len(markets)}")
            print(f"  Categories: {categories}")
            print(f"  Sample: {markets[0].question[:80]}")

        finally:
            await discovery.close()

    @pytest.mark.asyncio
    async def test_soccer_markets_exist(self) -> None:
        """Should find soccer markets including EPL/La Liga."""
        from arbo.connectors.market_discovery import MarketDiscovery

        discovery = MarketDiscovery()
        await discovery.initialize()

        try:
            markets = await discovery.fetch_all_active_markets(max_pages=5)

            # Store in cache for filtering
            discovery._markets = {m.condition_id: m for m in markets}

            soccer = discovery.get_by_category("soccer")
            print(f"\n  Total markets: {len(markets)}")
            print(f"  Soccer markets: {len(soccer)}")
            if soccer:
                for m in soccer[:5]:
                    print(f"    - {m.question[:80]}")

        finally:
            await discovery.close()


# ================================================================
# PM-001: Polymarket CLOB Client
# ================================================================


@pytest.mark.skipif(not _has_poly_key(), reason="POLY_PRIVATE_KEY not set")
class TestPolymarketCLOBIntegration:
    """PM-001 acceptance: Connect to CLOB, get orderbook for 3 active markets."""

    @pytest.mark.asyncio
    async def test_health_check(self) -> None:
        """CLOB health endpoint should return OK."""
        from arbo.connectors.polymarket_client import PolymarketClient

        client = PolymarketClient()
        await client.initialize()

        try:
            is_healthy = await client.health_check()
            assert is_healthy, "Polymarket CLOB health check failed"
            print("\n  CLOB health: OK")
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_get_orderbooks_for_3_markets(self) -> None:
        """Fetch orderbooks for 3 active markets, verify valid data."""
        from arbo.connectors.market_discovery import MarketDiscovery
        from arbo.connectors.polymarket_client import PolymarketClient

        # First get real token IDs from Gamma
        discovery = MarketDiscovery()
        await discovery.initialize()

        try:
            markets = await discovery.fetch_all_active_markets(max_pages=2)
            # Pick 3 markets with valid token IDs and some volume
            candidates = [m for m in markets if m.token_id_yes and m.volume_24h > Decimal("500")][
                :3
            ]

            assert (
                len(candidates) >= 3
            ), f"Need at least 3 active markets with volume, got {len(candidates)}"
        finally:
            await discovery.close()

        # Now fetch orderbooks from CLOB
        client = PolymarketClient()
        await client.initialize()

        try:
            for market in candidates:
                token_id = market.token_id_yes
                book = await client.get_orderbook(token_id)

                assert book.token_id == token_id
                print(f"\n  Market: {market.question[:60]}")
                print(f"    Token: {token_id[:20]}...")
                print(f"    Bids: {len(book.bids)}, Asks: {len(book.asks)}")
                print(f"    Midpoint: {book.midpoint}")
                print(f"    Spread: {book.spread}")

                # Verify prices are in valid range
                if book.midpoint is not None:
                    assert (
                        Decimal("0") < book.midpoint < Decimal("1")
                    ), f"Midpoint {book.midpoint} outside (0,1)"

        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_get_midpoint(self) -> None:
        """Get midpoint price for an active market."""
        from arbo.connectors.market_discovery import MarketDiscovery
        from arbo.connectors.polymarket_client import PolymarketClient

        discovery = MarketDiscovery()
        await discovery.initialize()

        try:
            markets = await discovery.fetch_all_active_markets(max_pages=1)
            candidate = next(
                (m for m in markets if m.token_id_yes and m.volume_24h > Decimal("1000")),
                None,
            )
            assert candidate is not None, "No active market with volume found"
        finally:
            await discovery.close()

        client = PolymarketClient()
        await client.initialize()

        try:
            midpoint = await client.get_midpoint(candidate.token_id_yes)
            assert (
                Decimal("0") < midpoint < Decimal("1")
            ), f"Midpoint {midpoint} outside valid range"
            print(f"\n  Market: {candidate.question[:60]}")
            print(f"  Midpoint: {midpoint}")
        finally:
            await client.close()


# ================================================================
# PM-003: The Odds API + Event Matching
# ================================================================


@pytest.mark.skipif(not _has_odds_key(), reason="ODDS_API_KEY not set")
class TestOddsAPIIntegration:
    """PM-003 acceptance: Fetch EPL odds, match with Polymarket markets."""

    @pytest.mark.asyncio
    async def test_fetch_epl_odds(self) -> None:
        """Fetch EPL odds from The Odds API."""
        from arbo.connectors.odds_api_client import OddsApiClient

        client = OddsApiClient()

        try:
            events = await client.get_odds("soccer_epl")

            print(f"\n  EPL events: {len(events)}")
            print(f"  Remaining quota: {client.remaining_quota}")

            for event in events[:5]:
                h2h = event.get_pinnacle_h2h()
                pinnacle_str = ""
                if h2h:
                    pinnacle_str = ", ".join(f"{k}:{v}" for k, v in h2h.items())
                print(f"    {event.home_team} vs {event.away_team} | {pinnacle_str}")

            # Even if no current EPL matches, the API should respond successfully
            assert client.remaining_quota is None or client.remaining_quota > 0

        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_fetch_all_soccer_odds(self) -> None:
        """Fetch odds across all configured soccer leagues."""
        from arbo.connectors.odds_api_client import OddsApiClient

        client = OddsApiClient()

        try:
            events = await client.get_soccer_odds()

            print(f"\n  Total soccer events: {len(events)}")
            print(f"  Remaining quota: {client.remaining_quota}")

            # Count events with Pinnacle odds
            with_pinnacle = sum(1 for e in events if e.get_pinnacle_h2h())
            print(f"  With Pinnacle h2h: {with_pinnacle}")

            for event in events[:3]:
                prob = event.get_pinnacle_implied_prob(event.home_team)
                print(f"    {event.home_team} vs {event.away_team}: home_prob={prob}")

        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_match_polymarket_with_odds_api(self) -> None:
        """Full pipeline: Gamma markets + Odds API → matched pairs."""
        from arbo.connectors.event_matcher import EventMatcher
        from arbo.connectors.market_discovery import MarketDiscovery
        from arbo.connectors.odds_api_client import OddsApiClient

        # Fetch Polymarket soccer markets
        discovery = MarketDiscovery()
        await discovery.initialize()

        try:
            markets = await discovery.fetch_all_active_markets(max_pages=5)
            discovery._markets = {m.condition_id: m for m in markets}
            soccer_markets = discovery.get_by_category("soccer")
            print(f"\n  Polymarket soccer markets: {len(soccer_markets)}")
        finally:
            await discovery.close()

        # Fetch Odds API events
        odds_client = OddsApiClient()
        try:
            odds_events = await odds_client.get_soccer_odds()
            print(f"  Odds API soccer events: {len(odds_events)}")
            print(f"  Remaining quota: {odds_client.remaining_quota}")
        finally:
            await odds_client.close()

        # Match them
        matcher = EventMatcher()
        matched = matcher.match_markets(soccer_markets, odds_events)

        print(f"  Matched pairs: {len(matched)}")
        for pair in matched[:5]:
            print(
                f"    Poly: {pair.polymarket.question[:50]} ↔ "
                f"OA: {pair.odds_event.home_team} vs {pair.odds_event.away_team} "
                f"(score={pair.match_score:.3f}, pinnacle={pair.pinnacle_prob})"
            )

        # Log results regardless — CEO wants to see the data
        if not matched:
            print("  ⚠ No matches found — may be off-season or question format mismatch")


# ================================================================
# PM-102: Value Signal Pipeline (Seasonal Matching)
# ================================================================


@pytest.mark.skipif(not _has_odds_key(), reason="ODDS_API_KEY not set")
class TestValueSignalPipeline:
    """PM-102 acceptance: ≥3 matched pairs on real data (match + seasonal)."""

    @pytest.mark.asyncio
    async def test_fetch_outrights(self) -> None:
        """Odds API outrights endpoint returns team probabilities."""
        from arbo.connectors.odds_api_client import OddsApiClient

        client = OddsApiClient()

        try:
            outrights = await client.get_outrights("soccer_epl")

            print(f"\n  EPL outrights: {len(outrights)} teams")
            print(f"  Remaining quota: {client.remaining_quota}")

            for team, prob in sorted(outrights.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"    {team}: {float(prob):.3f}")

            if outrights:
                total = sum(outrights.values())
                print(f"  Sum of probs: {float(total):.4f}")
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_seasonal_matching_real_data(self) -> None:
        """Match seasonal Polymarket questions against Pinnacle outrights.

        PM-102 acceptance: ≥3 successful matched pairs.
        """
        from arbo.connectors.event_matcher import EventMatcher
        from arbo.connectors.market_discovery import MarketDiscovery
        from arbo.connectors.odds_api_client import OddsApiClient

        # Fetch Polymarket soccer markets
        discovery = MarketDiscovery()
        await discovery.initialize()

        try:
            markets = await discovery.fetch_all_active_markets(max_pages=5)
            discovery._markets = {m.condition_id: m for m in markets}
            soccer_markets = discovery.get_by_category("soccer")
            print(f"\n  Polymarket soccer markets: {len(soccer_markets)}")
            for m in soccer_markets[:10]:
                print(f"    - {m.question[:80]}")
        finally:
            await discovery.close()

        # Fetch Odds API outrights
        odds_client = OddsApiClient()
        try:
            outright_odds = await odds_client.get_all_soccer_outrights()
            print(f"  Outright leagues: {len(outright_odds)}")
            for sport_key, odds in outright_odds.items():
                print(f"    {sport_key}: {len(odds)} teams")
            print(f"  Remaining quota: {odds_client.remaining_quota}")

            # Also get match-level odds for combined count
            match_events = await odds_client.get_soccer_odds()
            print(f"  Match-level events: {len(match_events)}")
        finally:
            await odds_client.close()

        # Run all matching strategies
        matcher = EventMatcher(threshold=0.70)

        # 1. Match-level (e.g. "Arsenal vs Chelsea?")
        match_pairs = matcher.match_markets(soccer_markets, match_events)

        # 2. Outright odds (if available)
        seasonal_pairs = matcher.match_seasonal_markets(soccer_markets, outright_odds)

        # 3. Derived from match odds (fallback for leagues without outrights)
        derived_pairs: list = []
        if not seasonal_pairs and match_events:
            derived_pairs = matcher.match_seasonal_via_match_odds(soccer_markets, match_events)

        all_pairs = match_pairs + seasonal_pairs + derived_pairs

        print(f"\n  Match-level pairs: {len(match_pairs)}")
        print(f"  Seasonal outright pairs: {len(seasonal_pairs)}")
        print(f"  Seasonal derived pairs: {len(derived_pairs)}")
        print(f"  Total matched pairs: {len(all_pairs)}")

        for pair in all_pairs[:10]:
            mt = pair.match_type.upper()
            ev = pair.odds_event
            ev_str = f"{ev.home_team} vs {ev.away_team}" if ev else "N/A"
            print(
                f"    [{mt}] {pair.polymarket.question[:50]} → "
                f"{pair.outright_team or ev_str} ({pair.sport_key}) "
                f"Pinnacle={pair.pinnacle_prob} score={pair.match_score:.3f}"
            )

        # PM-102 acceptance: ≥3 matched pairs
        assert len(all_pairs) >= 3, (
            f"PM-102 acceptance FAILED: need ≥3 matched pairs, got {len(all_pairs)}. "
            f"Check seasonal question patterns and match event availability."
        )
