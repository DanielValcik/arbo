"""Tests for fuzzy event matching."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from src.data.event_matcher import EventMatcher, load_aliases
from src.data.odds_api import OddsApiBookmaker, OddsApiEvent, OddsApiMarket, OddsApiOutcome
from src.exchanges.base import EventStatus, ExchangeEvent


def _mb_event(
    event_id: int = 1,
    home: str = "Liverpool",
    away: str = "Arsenal",
    start: datetime | None = None,
) -> ExchangeEvent:
    """Helper to create Matchbook events."""
    return ExchangeEvent(
        event_id=event_id,
        name=f"{home} vs {away}",
        sport="football",
        league="EPL",
        home_team=home,
        away_team=away,
        start_time=start or datetime(2026, 3, 1, 15, 0, tzinfo=UTC),
        status=EventStatus.UPCOMING,
        markets=[],
    )


def _oa_event(
    event_id: str = "abc123",
    home: str = "Liverpool",
    away: str = "Arsenal",
    start: datetime | None = None,
) -> OddsApiEvent:
    """Helper to create Odds API events."""
    return OddsApiEvent(
        id=event_id,
        sport_key="soccer_epl",
        home_team=home,
        away_team=away,
        commence_time=start or datetime(2026, 3, 1, 15, 0, tzinfo=UTC),
        bookmakers=[
            OddsApiBookmaker(
                key="bet365",
                title="Bet365",
                markets=[
                    OddsApiMarket(
                        key="h2h",
                        outcomes=[
                            OddsApiOutcome(name=home, price=Decimal("2.10")),
                            OddsApiOutcome(name="Draw", price=Decimal("3.40")),
                            OddsApiOutcome(name=away, price=Decimal("3.60")),
                        ],
                    )
                ],
            )
        ],
    )


SAMPLE_ALIASES = {
    "Liverpool": ["Liverpool FC", "LFC"],
    "Arsenal": ["Arsenal FC", "AFC"],
    "Manchester United": ["Man Utd", "Man United", "MUFC"],
    "Manchester City": ["Man City", "MCFC"],
    "Tottenham Hotspur": ["Tottenham", "Spurs"],
}


class TestLoadAliases:
    def test_load_from_file(self, tmp_path) -> None:
        """Load aliases from a YAML file."""
        yaml_content = """
football:
  "Liverpool": ["LFC", "L'pool"]
basketball:
  "Los Angeles Lakers": ["LA Lakers", "Lakers"]
"""
        path = tmp_path / "aliases.yaml"
        path.write_text(yaml_content)

        aliases = load_aliases(path)
        assert "Liverpool" in aliases
        assert "LFC" in aliases["Liverpool"]
        assert "Los Angeles Lakers" in aliases
        assert "LA Lakers" in aliases["Los Angeles Lakers"]

    def test_missing_file_returns_empty(self, tmp_path) -> None:
        path = tmp_path / "nonexistent.yaml"
        assert load_aliases(path) == {}


class TestEventMatcherExact:
    def test_exact_name_match(self) -> None:
        matcher = EventMatcher()
        mb = [_mb_event(home="Liverpool", away="Arsenal")]
        oa = [_oa_event(home="Liverpool", away="Arsenal")]

        result = matcher.match_events(mb, oa)
        assert len(result) == 1
        assert result[0].match_score == 1.0

    def test_no_match_different_teams(self) -> None:
        matcher = EventMatcher()
        mb = [_mb_event(home="Liverpool", away="Arsenal")]
        oa = [_oa_event(home="Bayern Munich", away="Dortmund")]

        result = matcher.match_events(mb, oa)
        assert len(result) == 0


class TestEventMatcherFuzzy:
    def test_fc_suffix_stripped(self) -> None:
        """'Liverpool FC' matches 'Liverpool' after stripping FC."""
        matcher = EventMatcher()
        mb = [_mb_event(home="Liverpool FC", away="Arsenal FC")]
        oa = [_oa_event(home="Liverpool", away="Arsenal")]

        result = matcher.match_events(mb, oa)
        assert len(result) == 1
        assert result[0].match_score == 1.0

    def test_alias_expansion(self) -> None:
        """Alias 'Man Utd' resolves to 'Manchester United'."""
        matcher = EventMatcher(aliases=SAMPLE_ALIASES)
        mb = [_mb_event(home="Man Utd", away="Man City")]
        oa = [_oa_event(home="Manchester United", away="Manchester City")]

        result = matcher.match_events(mb, oa)
        assert len(result) == 1
        assert result[0].match_score == 1.0

    def test_partial_fuzzy_match(self) -> None:
        """Fuzzy match with minor spelling difference."""
        matcher = EventMatcher(threshold=0.75)
        mb = [_mb_event(home="Tottenham Hotspur", away="Arsenal")]
        oa = [_oa_event(home="Tottenham", away="Arsenal")]

        result = matcher.match_events(mb, oa)
        assert len(result) == 1
        assert result[0].match_score >= 0.75


class TestEventMatcherTime:
    def test_time_within_threshold(self) -> None:
        """Events within 2 hours match."""
        matcher = EventMatcher()
        t1 = datetime(2026, 3, 1, 15, 0, tzinfo=UTC)
        t2 = t1 + timedelta(minutes=30)

        mb = [_mb_event(start=t1)]
        oa = [_oa_event(start=t2)]

        result = matcher.match_events(mb, oa)
        assert len(result) == 1

    def test_time_exceeds_threshold(self) -> None:
        """Events >2 hours apart don't match."""
        matcher = EventMatcher()
        t1 = datetime(2026, 3, 1, 15, 0, tzinfo=UTC)
        t2 = t1 + timedelta(hours=3)

        mb = [_mb_event(start=t1)]
        oa = [_oa_event(start=t2)]

        result = matcher.match_events(mb, oa)
        assert len(result) == 0


class TestEventMatcherEdgeCases:
    def test_multiple_matches(self) -> None:
        """Multiple Matchbook events matched to different Odds API events."""
        matcher = EventMatcher()
        mb = [
            _mb_event(event_id=1, home="Liverpool", away="Arsenal"),
            _mb_event(event_id=2, home="Chelsea", away="Everton"),
        ]
        oa = [
            _oa_event(event_id="a1", home="Liverpool", away="Arsenal"),
            _oa_event(event_id="a2", home="Chelsea", away="Everton"),
        ]

        result = matcher.match_events(mb, oa)
        assert len(result) == 2

    def test_no_duplicate_oa_match(self) -> None:
        """An Odds API event is only matched once (greedy first match)."""
        matcher = EventMatcher()
        mb = [
            _mb_event(event_id=1, home="Liverpool", away="Arsenal"),
            _mb_event(event_id=2, home="Liverpool", away="Arsenal"),
        ]
        oa = [_oa_event(event_id="a1", home="Liverpool", away="Arsenal")]

        result = matcher.match_events(mb, oa)
        assert len(result) == 1

    def test_below_threshold(self) -> None:
        """Score below threshold is not matched."""
        matcher = EventMatcher(threshold=0.99)
        mb = [_mb_event(home="Liverpool FC", away="Arsenal FC")]
        oa = [_oa_event(home="Liverpool Reserves", away="Arsenal U23")]

        result = matcher.match_events(mb, oa)
        assert len(result) == 0

    def test_empty_inputs(self) -> None:
        matcher = EventMatcher()
        assert matcher.match_events([], []) == []
        assert matcher.match_events([_mb_event()], []) == []
        assert matcher.match_events([], [_oa_event()]) == []
