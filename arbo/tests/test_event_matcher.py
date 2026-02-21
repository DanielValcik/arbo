"""Tests for Event Matcher (PM-003).

Tests verify:
1. Team name extraction from Polymarket questions
2. Fuzzy name matching with aliases
3. Polymarket ↔ Odds API event matching
4. Name normalization (FC stripping, case, whitespace)
5. Alias resolution
6. Score calculation (normal + swapped)
"""

from __future__ import annotations

from decimal import Decimal

from arbo.connectors.event_matcher import (
    DEFAULT_THRESHOLD,
    EventMatcher,
    extract_teams_from_question,
)
from arbo.connectors.market_discovery import GammaMarket
from arbo.connectors.odds_api_client import (
    OddsBookmaker,
    OddsEvent,
    OddsMarket,
    OddsOutcome,
)

# ================================================================
# Team extraction from questions
# ================================================================


class TestTeamExtraction:
    """Extract team names from Polymarket market questions."""

    def test_vs_format(self) -> None:
        result = extract_teams_from_question("Arsenal vs Chelsea?")
        assert result is not None
        assert result[0] == "Arsenal"
        assert result[1] == "Chelsea"

    def test_v_format(self) -> None:
        result = extract_teams_from_question("Arsenal v Chelsea?")
        assert result is not None
        assert result[0] == "Arsenal"
        assert result[1] == "Chelsea"

    def test_will_win_format(self) -> None:
        result = extract_teams_from_question("Will Arsenal win against Chelsea?")
        assert result is not None
        assert result[0] == "Arsenal"
        assert result[1] == "Chelsea"

    def test_who_will_win_format(self) -> None:
        result = extract_teams_from_question("Who will win: Arsenal or Chelsea?")
        assert result is not None
        assert result[0] == "Arsenal"
        assert result[1] == "Chelsea"

    def test_no_teams_returns_none(self) -> None:
        """Non-match question returns None."""
        result = extract_teams_from_question("Will Bitcoin hit $100k?")
        assert result is None

    def test_single_word_teams(self) -> None:
        result = extract_teams_from_question("Liverpool vs Everton?")
        assert result is not None
        assert result[0] == "Liverpool"
        assert result[1] == "Everton"


# ================================================================
# Name normalization and aliases
# ================================================================


class TestNameNormalization:
    """EventMatcher name normalization."""

    def test_lowercase(self) -> None:
        matcher = EventMatcher()
        assert matcher._normalize_name("ARSENAL") == "arsenal"

    def test_strip_fc(self) -> None:
        matcher = EventMatcher()
        assert matcher._normalize_name("Arsenal FC") == "arsenal"

    def test_strip_multiple_suffixes(self) -> None:
        matcher = EventMatcher()
        assert matcher._normalize_name("AC Milan") == "milan"

    def test_extra_whitespace(self) -> None:
        matcher = EventMatcher()
        assert matcher._normalize_name("  Real   Madrid  ") == "real madrid"

    def test_alias_resolution(self) -> None:
        matcher = EventMatcher(aliases={"Manchester United": ["Man United", "Man Utd", "MUFC"]})
        sim = matcher._name_similarity("Man Utd", "Manchester United")
        assert sim == 1.0  # Both resolve to same canonical

    def test_alias_with_normalization(self) -> None:
        matcher = EventMatcher(aliases={"Atletico Madrid": ["Atleti", "Atletico"]})
        sim = matcher._name_similarity("Atleti", "Atletico Madrid")
        assert sim == 1.0


# ================================================================
# Score calculation
# ================================================================


class TestScoreCalculation:
    """Match scoring between team pairs."""

    def test_exact_match(self) -> None:
        matcher = EventMatcher()
        event = OddsEvent(
            id="e1",
            sport_key="soccer_epl",
            home_team="Arsenal",
            away_team="Chelsea",
            commence_time="2026-03-01T15:00:00Z",
            bookmakers=[],
        )
        score = matcher._score_match("Arsenal", "Chelsea", event)
        assert score == 1.0

    def test_swapped_teams_still_match(self) -> None:
        """Teams in different order should still get high score."""
        matcher = EventMatcher()
        event = OddsEvent(
            id="e1",
            sport_key="soccer_epl",
            home_team="Arsenal",
            away_team="Chelsea",
            commence_time="2026-03-01T15:00:00Z",
            bookmakers=[],
        )
        score = matcher._score_match("Chelsea", "Arsenal", event)
        assert score == 1.0

    def test_partial_match(self) -> None:
        """Similar but not identical names get reasonable score."""
        matcher = EventMatcher()
        event = OddsEvent(
            id="e1",
            sport_key="soccer_epl",
            home_team="Arsenal",
            away_team="Chelsea",
            commence_time="2026-03-01T15:00:00Z",
            bookmakers=[],
        )
        score = matcher._score_match("Arsenal FC", "Chelsea FC", event)
        assert score > 0.8

    def test_completely_different(self) -> None:
        matcher = EventMatcher()
        event = OddsEvent(
            id="e1",
            sport_key="soccer_epl",
            home_team="Arsenal",
            away_team="Chelsea",
            commence_time="2026-03-01T15:00:00Z",
            bookmakers=[],
        )
        score = matcher._score_match("Barcelona", "Real Madrid", event)
        assert score < DEFAULT_THRESHOLD  # Below matching threshold


# ================================================================
# Full matching pipeline
# ================================================================


def _make_gamma_market(
    condition_id: str,
    question: str,
    category: str = "soccer",
    outcome_prices: list[str] | None = None,
) -> GammaMarket:
    """Build a GammaMarket for testing."""
    raw = {
        "conditionId": condition_id,
        "question": question,
        "slug": "test",
        "outcomes": ["Yes", "No"],
        "outcomePrices": outcome_prices or ["0.55", "0.45"],
        "clobTokenIds": ["tok_yes", "tok_no"],
        "volume": "100000",
        "volume24hr": "5000",
        "liquidity": "10000",
        "active": True,
        "closed": False,
        "feesEnabled": False,
        "enableNegRisk": False,
        "tags": [{"label": "Premier League"}] if category == "soccer" else [],
    }
    return GammaMarket(raw)


def _make_odds_event(
    event_id: str,
    home_team: str,
    away_team: str,
) -> OddsEvent:
    """Build a test OddsEvent with Pinnacle h2h."""
    return OddsEvent(
        id=event_id,
        sport_key="soccer_epl",
        home_team=home_team,
        away_team=away_team,
        commence_time="2026-03-01T15:00:00Z",
        bookmakers=[
            OddsBookmaker(
                key="pinnacle",
                title="Pinnacle",
                markets=[
                    OddsMarket(
                        key="h2h",
                        outcomes=[
                            OddsOutcome(name=home_team, price=Decimal("2.10")),
                            OddsOutcome(name=away_team, price=Decimal("3.50")),
                            OddsOutcome(name="Draw", price=Decimal("3.40")),
                        ],
                    )
                ],
            )
        ],
    )


class TestFullMatching:
    """End-to-end matching pipeline."""

    def test_simple_match(self) -> None:
        matcher = EventMatcher()
        polymarkets = [
            _make_gamma_market("c1", "Arsenal vs Chelsea?"),
        ]
        odds_events = [
            _make_odds_event("e1", "Arsenal", "Chelsea"),
        ]
        matched = matcher.match_markets(polymarkets, odds_events)
        assert len(matched) == 1
        assert matched[0].match_score >= 0.85

    def test_no_match_for_non_soccer(self) -> None:
        """Non-soccer Polymarket markets are skipped."""
        matcher = EventMatcher()
        polymarkets = [
            _make_gamma_market("c1", "Will Bitcoin hit $100k?", category="crypto"),
        ]
        odds_events = [
            _make_odds_event("e1", "Arsenal", "Chelsea"),
        ]
        matched = matcher.match_markets(polymarkets, odds_events)
        assert len(matched) == 0

    def test_no_match_for_unrelated_teams(self) -> None:
        """Different teams should not match."""
        matcher = EventMatcher()
        polymarkets = [
            _make_gamma_market("c1", "Barcelona vs Real Madrid?"),
        ]
        odds_events = [
            _make_odds_event("e1", "Arsenal", "Chelsea"),
        ]
        matched = matcher.match_markets(polymarkets, odds_events)
        assert len(matched) == 0

    def test_matched_pair_has_pinnacle_prob(self) -> None:
        matcher = EventMatcher()
        polymarkets = [
            _make_gamma_market("c1", "Arsenal vs Chelsea?"),
        ]
        odds_events = [
            _make_odds_event("e1", "Arsenal", "Chelsea"),
        ]
        matched = matcher.match_markets(polymarkets, odds_events)
        assert len(matched) == 1
        assert matched[0].pinnacle_prob is not None
        assert Decimal("0") < matched[0].pinnacle_prob < Decimal("1")

    def test_one_to_one_matching(self) -> None:
        """Each Odds API event used at most once."""
        matcher = EventMatcher()
        polymarkets = [
            _make_gamma_market("c1", "Arsenal vs Chelsea?"),
            _make_gamma_market("c2", "Will Arsenal beat Chelsea?"),  # same match
        ]
        odds_events = [
            _make_odds_event("e1", "Arsenal", "Chelsea"),
        ]
        matched = matcher.match_markets(polymarkets, odds_events)
        # Only one should match (first wins)
        assert len(matched) == 1

    def test_multiple_matches(self) -> None:
        matcher = EventMatcher()
        polymarkets = [
            _make_gamma_market("c1", "Arsenal vs Chelsea?"),
            _make_gamma_market("c2", "Liverpool vs Everton?"),
        ]
        odds_events = [
            _make_odds_event("e1", "Arsenal", "Chelsea"),
            _make_odds_event("e2", "Liverpool", "Everton"),
        ]
        matched = matcher.match_markets(polymarkets, odds_events)
        assert len(matched) == 2
