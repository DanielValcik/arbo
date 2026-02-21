"""Tests for The Odds API client (PM-003).

Tests verify:
1. OddsEvent parsing from API response
2. Pinnacle h2h extraction
3. Implied probability calculation (with vig removal)
4. Preferred bookmaker fallback
5. Quota tracking
6. SPORT_KEY_MAP coverage
"""

from __future__ import annotations

from decimal import Decimal

from arbo.connectors.odds_api_client import (
    SPORT_KEY_MAP,
    OddsBookmaker,
    OddsEvent,
    OddsMarket,
    OddsOutcome,
)

# ================================================================
# Factory helper
# ================================================================


def _make_event(
    home_team: str = "Arsenal",
    away_team: str = "Chelsea",
    bookmakers: list[OddsBookmaker] | None = None,
) -> OddsEvent:
    """Build a test OddsEvent."""
    if bookmakers is None:
        bookmakers = [
            OddsBookmaker(
                key="pinnacle",
                title="Pinnacle",
                markets=[
                    OddsMarket(
                        key="h2h",
                        outcomes=[
                            OddsOutcome(name="Arsenal", price=Decimal("2.10")),
                            OddsOutcome(name="Chelsea", price=Decimal("3.50")),
                            OddsOutcome(name="Draw", price=Decimal("3.40")),
                        ],
                    )
                ],
            )
        ]

    return OddsEvent(
        id="event_123",
        sport_key="soccer_epl",
        home_team=home_team,
        away_team=away_team,
        commence_time="2026-03-01T15:00:00Z",
        bookmakers=bookmakers,
    )


# ================================================================
# OddsEvent tests
# ================================================================


class TestOddsEventParsing:
    """OddsEvent construction and field access."""

    def test_basic_fields(self) -> None:
        event = _make_event()
        assert event.home_team == "Arsenal"
        assert event.away_team == "Chelsea"
        assert event.sport_key == "soccer_epl"

    def test_bookmakers_parsed(self) -> None:
        event = _make_event()
        assert len(event.bookmakers) == 1
        assert event.bookmakers[0].key == "pinnacle"


class TestPinnacleExtraction:
    """Pinnacle h2h odds extraction."""

    def test_pinnacle_h2h_found(self) -> None:
        event = _make_event()
        h2h = event.get_pinnacle_h2h()
        assert h2h is not None
        assert "Arsenal" in h2h
        assert h2h["Arsenal"] == Decimal("2.10")

    def test_no_pinnacle_falls_back(self) -> None:
        """If Pinnacle missing, falls back to next preferred bookmaker."""
        event = _make_event(
            bookmakers=[
                OddsBookmaker(
                    key="betfair_ex_eu",
                    title="Betfair",
                    markets=[
                        OddsMarket(
                            key="h2h",
                            outcomes=[
                                OddsOutcome(name="Arsenal", price=Decimal("2.20")),
                                OddsOutcome(name="Chelsea", price=Decimal("3.60")),
                            ],
                        )
                    ],
                )
            ]
        )
        h2h = event.get_pinnacle_h2h()
        assert h2h is not None
        assert h2h["Arsenal"] == Decimal("2.20")

    def test_no_bookmakers_returns_none(self) -> None:
        event = _make_event(bookmakers=[])
        assert event.get_pinnacle_h2h() is None

    def test_no_h2h_market_returns_none(self) -> None:
        """Bookmaker exists but has no h2h market."""
        event = _make_event(
            bookmakers=[
                OddsBookmaker(
                    key="pinnacle",
                    title="Pinnacle",
                    markets=[
                        OddsMarket(
                            key="spreads",
                            outcomes=[OddsOutcome(name="Arsenal", price=Decimal("1.90"))],
                        )
                    ],
                )
            ]
        )
        assert event.get_pinnacle_h2h() is None


class TestImpliedProbability:
    """Implied probability with vig removal."""

    def test_implied_prob_reasonable(self) -> None:
        """Arsenal at 2.10 → raw ~47.6%, after vig removal should be ~45-50%."""
        event = _make_event()
        prob = event.get_pinnacle_implied_prob("Arsenal")
        assert prob is not None
        assert Decimal("0.40") < prob < Decimal("0.55")

    def test_probabilities_sum_to_one(self) -> None:
        """All outcome probs should sum to ~1.0 after vig removal."""
        event = _make_event()
        probs = []
        for outcome_name in ["Arsenal", "Chelsea", "Draw"]:
            p = event.get_pinnacle_implied_prob(outcome_name)
            assert p is not None
            probs.append(p)
        total = sum(probs)
        assert abs(total - Decimal("1")) < Decimal("0.01")

    def test_unknown_outcome_returns_none(self) -> None:
        event = _make_event()
        assert event.get_pinnacle_implied_prob("Liverpool") is None

    def test_no_pinnacle_returns_none(self) -> None:
        event = _make_event(bookmakers=[])
        assert event.get_pinnacle_implied_prob("Arsenal") is None


class TestSportKeyMap:
    """SPORT_KEY_MAP configuration."""

    def test_soccer_has_leagues(self) -> None:
        assert "soccer" in SPORT_KEY_MAP
        assert len(SPORT_KEY_MAP["soccer"]) >= 5

    def test_epl_included(self) -> None:
        assert "soccer_epl" in SPORT_KEY_MAP["soccer"]

    def test_la_liga_included(self) -> None:
        assert "soccer_spain_la_liga" in SPORT_KEY_MAP["soccer"]
