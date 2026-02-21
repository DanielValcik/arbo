"""Tests for PM-102: Value Signal Generator.

Tests verify:
1. Seasonal question extraction (team + league)
2. League identification (name → sport key mapping)
3. Outright odds parsing (vig removal)
4. Seasonal market matching pipeline
5. Signal generation (edge calculation, direction, confidence)
6. ValueSignalGenerator orchestration
7. Integration with ValueModel (when trained)

Acceptance: ≥3 matched pairs on real data (tested in test_integration.py).
"""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from arbo.connectors.event_matcher import (
    EventMatcher,
    MatchedPair,
    extract_team_from_seasonal,
    extract_teams_from_question,
    identify_league,
)
from arbo.connectors.market_discovery import GammaMarket
from arbo.connectors.odds_api_client import (
    OddsApiClient,
    OddsBookmaker,
    OddsEvent,
    OddsMarket,
    OddsOutcome,
)
from arbo.core.scanner import Signal, SignalDirection
from arbo.strategies.value_signal import (
    ValueSignalGenerator,
    build_market_features,
    evaluate_pair,
)

# ================================================================
# Factory helpers
# ================================================================


def _make_gamma_market(
    condition_id: str = "cond_1",
    question: str = "Will Arsenal win the Premier League?",
    category: str = "soccer",
    price_yes: str = "0.35",
    price_no: str = "0.65",
    volume_24h: str = "5000",
    liquidity: str = "10000",
    fee_enabled: bool = False,
    end_date: str | None = "2026-05-24T00:00:00Z",
) -> GammaMarket:
    """Build a GammaMarket for testing."""
    raw = {
        "conditionId": condition_id,
        "question": question,
        "slug": "test-market",
        "outcomes": '["Yes", "No"]',
        "outcomePrices": f'["{price_yes}", "{price_no}"]',
        "clobTokenIds": '["tok_yes_1", "tok_no_1"]',
        "volume": "100000",
        "volume24hr": volume_24h,
        "liquidity": liquidity,
        "active": True,
        "closed": False,
        "feesEnabled": fee_enabled,
        "enableNegRisk": False,
        "endDate": end_date,
        "tags": [{"label": "Premier League"}] if category == "soccer" else [],
    }
    return GammaMarket(raw)


def _make_outright_odds() -> dict[str, dict[str, Decimal]]:
    """Build sample outright odds for testing."""
    return {
        "soccer_epl": {
            "Arsenal": Decimal("0.35"),
            "Manchester City": Decimal("0.25"),
            "Liverpool": Decimal("0.20"),
            "Chelsea": Decimal("0.10"),
            "Manchester United": Decimal("0.05"),
            "Tottenham Hotspur": Decimal("0.05"),
        },
        "soccer_spain_la_liga": {
            "Barcelona": Decimal("0.40"),
            "Real Madrid": Decimal("0.35"),
            "Atletico Madrid": Decimal("0.15"),
            "Athletic Bilbao": Decimal("0.10"),
        },
    }


# ================================================================
# Seasonal question extraction
# ================================================================


class TestSeasonalExtraction:
    """Extract team + league from seasonal Polymarket questions."""

    def test_will_win_the_league(self) -> None:
        result = extract_team_from_seasonal("Will Arsenal win the Premier League?")
        assert result is not None
        assert result[0] == "Arsenal"
        assert result[1] == "Premier League"

    def test_to_win_the_league(self) -> None:
        result = extract_team_from_seasonal("Arsenal to win the Premier League")
        assert result is not None
        assert result[0] == "Arsenal"
        assert result[1] == "Premier League"

    def test_with_season_year(self) -> None:
        result = extract_team_from_seasonal("Will Arsenal win the 2025/26 EPL?")
        assert result is not None
        assert result[0] == "Arsenal"
        assert "EPL" in result[1]

    def test_champions_pattern(self) -> None:
        result = extract_team_from_seasonal("Will Arsenal be EPL champions?")
        assert result is not None
        assert result[0] == "Arsenal"
        assert "EPL" in result[1]

    def test_multi_word_team(self) -> None:
        result = extract_team_from_seasonal("Will Manchester City win the Premier League?")
        assert result is not None
        assert result[0] == "Manchester City"

    def test_la_liga(self) -> None:
        result = extract_team_from_seasonal("Will Barcelona win La Liga?")
        assert result is not None
        assert result[0] == "Barcelona"
        assert "La Liga" in result[1]

    def test_en_dash_year(self) -> None:
        """Polymarket uses en-dash in '2025\u201326'."""
        result = extract_team_from_seasonal(
            "Will Liverpool win the 2025\u201326 English Premier League?"
        )
        assert result is not None
        assert result[0] == "Liverpool"
        assert "English Premier League" in result[1]

    def test_non_seasonal_returns_none(self) -> None:
        """Match-level questions should NOT match seasonal patterns."""
        result = extract_team_from_seasonal("Arsenal vs Chelsea?")
        assert result is None

    def test_crypto_returns_none(self) -> None:
        result = extract_team_from_seasonal("Will Bitcoin hit $100k?")
        assert result is None

    def test_match_level_still_works(self) -> None:
        """Existing match-level extraction is not broken."""
        result = extract_teams_from_question("Arsenal vs Chelsea?")
        assert result is not None
        assert result[0] == "Arsenal"
        assert result[1] == "Chelsea"


# ================================================================
# League identification
# ================================================================


class TestLeagueIdentification:
    """Map league strings to Odds API sport keys."""

    def test_premier_league(self) -> None:
        assert identify_league("Premier League") == "soccer_epl"

    def test_epl_abbreviation(self) -> None:
        assert identify_league("EPL") == "soccer_epl"

    def test_la_liga(self) -> None:
        assert identify_league("La Liga") == "soccer_spain_la_liga"

    def test_bundesliga(self) -> None:
        assert identify_league("Bundesliga") == "soccer_germany_bundesliga"

    def test_serie_a(self) -> None:
        assert identify_league("Serie A") == "soccer_italy_serie_a"

    def test_champions_league(self) -> None:
        assert identify_league("Champions League") == "soccer_uefa_champs_league"

    def test_ucl_abbreviation(self) -> None:
        assert identify_league("UCL") == "soccer_uefa_champs_league"

    def test_ligue_1(self) -> None:
        assert identify_league("Ligue 1") == "soccer_france_ligue_one"

    def test_case_insensitive(self) -> None:
        assert identify_league("PREMIER LEAGUE") == "soccer_epl"

    def test_unknown_league_returns_none(self) -> None:
        assert identify_league("MLS") is None

    def test_partial_match_in_longer_string(self) -> None:
        assert identify_league("the 2025/26 Premier League title") == "soccer_epl"


# ================================================================
# Outright odds parsing
# ================================================================


class TestOutrightsParsing:
    """OddsApiClient._parse_outrights correctness."""

    def test_parse_pinnacle_outrights(self) -> None:
        client = OddsApiClient.__new__(OddsApiClient)
        data = [
            {
                "id": "outright_1",
                "sport_key": "soccer_epl",
                "bookmakers": [
                    {
                        "key": "pinnacle",
                        "title": "Pinnacle",
                        "markets": [
                            {
                                "key": "outrights",
                                "outcomes": [
                                    {"name": "Arsenal", "price": 2.50},
                                    {"name": "Man City", "price": 3.00},
                                    {"name": "Liverpool", "price": 5.00},
                                ],
                            }
                        ],
                    }
                ],
            }
        ]

        result = client._parse_outrights(data)
        assert len(result) == 3
        assert "Arsenal" in result
        assert "Man City" in result
        assert "Liverpool" in result

    def test_outrights_sum_to_one(self) -> None:
        """Vig-removed probabilities should sum to ~1.0."""
        client = OddsApiClient.__new__(OddsApiClient)
        data = [
            {
                "id": "outright_1",
                "sport_key": "soccer_epl",
                "bookmakers": [
                    {
                        "key": "pinnacle",
                        "title": "Pinnacle",
                        "markets": [
                            {
                                "key": "outrights",
                                "outcomes": [
                                    {"name": "Arsenal", "price": 2.50},
                                    {"name": "Man City", "price": 3.00},
                                    {"name": "Liverpool", "price": 5.00},
                                    {"name": "Chelsea", "price": 10.00},
                                ],
                            }
                        ],
                    }
                ],
            }
        ]

        result = client._parse_outrights(data)
        total = sum(result.values())
        assert abs(total - Decimal("1")) < Decimal("0.01")

    def test_outrights_preferred_bookmaker_fallback(self) -> None:
        """Falls back to betfair if no Pinnacle."""
        client = OddsApiClient.__new__(OddsApiClient)
        data = [
            {
                "id": "outright_1",
                "sport_key": "soccer_epl",
                "bookmakers": [
                    {
                        "key": "betfair_ex_eu",
                        "title": "Betfair",
                        "markets": [
                            {
                                "key": "outrights",
                                "outcomes": [
                                    {"name": "Arsenal", "price": 2.50},
                                    {"name": "Man City", "price": 3.00},
                                ],
                            }
                        ],
                    }
                ],
            }
        ]

        result = client._parse_outrights(data)
        assert len(result) == 2

    def test_empty_response(self) -> None:
        client = OddsApiClient.__new__(OddsApiClient)
        assert client._parse_outrights([]) == {}

    def test_no_outrights_market(self) -> None:
        """Bookmaker has h2h but not outrights."""
        client = OddsApiClient.__new__(OddsApiClient)
        data = [
            {
                "id": "event_1",
                "bookmakers": [
                    {
                        "key": "pinnacle",
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [{"name": "Arsenal", "price": 2.10}],
                            }
                        ],
                    }
                ],
            }
        ]

        assert client._parse_outrights(data) == {}


# ================================================================
# Seasonal market matching
# ================================================================


class TestSeasonalMatching:
    """EventMatcher.match_seasonal_markets pipeline."""

    def test_simple_seasonal_match(self) -> None:
        matcher = EventMatcher()
        markets = [
            _make_gamma_market("c1", "Will Arsenal win the Premier League?"),
        ]
        outrights = _make_outright_odds()

        matched = matcher.match_seasonal_markets(markets, outrights)
        assert len(matched) == 1
        assert matched[0].match_type == "seasonal"
        assert matched[0].pinnacle_prob == Decimal("0.35")
        assert matched[0].outright_team == "Arsenal"
        assert matched[0].sport_key == "soccer_epl"

    def test_multiple_leagues(self) -> None:
        matcher = EventMatcher()
        markets = [
            _make_gamma_market("c1", "Will Arsenal win the Premier League?"),
            _make_gamma_market("c2", "Will Barcelona win La Liga?"),
        ]
        outrights = _make_outright_odds()

        matched = matcher.match_seasonal_markets(markets, outrights)
        assert len(matched) == 2
        teams = {m.outright_team for m in matched}
        assert "Arsenal" in teams
        assert "Barcelona" in teams

    def test_fuzzy_team_name(self) -> None:
        """Polymarket might say 'Man City' but outrights have 'Manchester City'."""
        matcher = EventMatcher(threshold=0.6)  # Relaxed for fuzzy test
        markets = [
            _make_gamma_market("c1", "Will Man City win the Premier League?"),
        ]
        outrights = _make_outright_odds()

        matched = matcher.match_seasonal_markets(markets, outrights)
        assert len(matched) == 1
        assert matched[0].outright_team == "Manchester City"

    def test_alias_resolution(self) -> None:
        matcher = EventMatcher(
            aliases={"Manchester City": ["Man City", "MCFC"]},
        )
        markets = [
            _make_gamma_market("c1", "Will Man City win the Premier League?"),
        ]
        outrights = _make_outright_odds()

        matched = matcher.match_seasonal_markets(markets, outrights)
        assert len(matched) == 1
        assert matched[0].match_score == 1.0

    def test_non_soccer_skipped(self) -> None:
        matcher = EventMatcher()
        markets = [
            _make_gamma_market("c1", "Will Bitcoin hit $100k?", category="crypto"),
        ]
        outrights = _make_outright_odds()

        matched = matcher.match_seasonal_markets(markets, outrights)
        assert len(matched) == 0

    def test_unknown_league_skipped(self) -> None:
        matcher = EventMatcher()
        markets = [
            _make_gamma_market("c1", "Will Arsenal win the MLS Cup?"),
        ]
        outrights = _make_outright_odds()

        matched = matcher.match_seasonal_markets(markets, outrights)
        assert len(matched) == 0

    def test_no_outrights_available(self) -> None:
        matcher = EventMatcher()
        markets = [
            _make_gamma_market("c1", "Will Arsenal win the Premier League?"),
        ]

        matched = matcher.match_seasonal_markets(markets, {})
        assert len(matched) == 0

    def test_team_not_in_outrights(self) -> None:
        """Team from question is not listed in outright odds."""
        matcher = EventMatcher()
        markets = [
            _make_gamma_market("c1", "Will Wolverhampton win the Premier League?"),
        ]
        outrights = _make_outright_odds()  # Wolves not in list

        matched = matcher.match_seasonal_markets(markets, outrights)
        assert len(matched) == 0


# ================================================================
# Derived matching (match-level → seasonal)
# ================================================================


def _make_epl_events() -> list[OddsEvent]:
    """Build sample EPL match events with Pinnacle h2h."""
    teams = [
        ("Arsenal", "Chelsea", Decimal("1.80"), Decimal("4.50"), Decimal("3.80")),
        ("Arsenal", "Everton", Decimal("1.40"), Decimal("7.00"), Decimal("5.00")),
        ("Liverpool", "Arsenal", Decimal("2.10"), Decimal("3.50"), Decimal("3.40")),
        ("Liverpool", "Chelsea", Decimal("1.90"), Decimal("4.00"), Decimal("3.50")),
        ("Brighton", "Newcastle", Decimal("2.50"), Decimal("3.00"), Decimal("3.20")),
    ]
    events = []
    for i, (home, away, h_price, a_price, d_price) in enumerate(teams):
        events.append(
            OddsEvent(
                id=f"e{i}",
                sport_key="soccer_epl",
                home_team=home,
                away_team=away,
                commence_time="2026-03-01T15:00:00Z",
                bookmakers=[
                    OddsBookmaker(
                        key="pinnacle",
                        title="Pinnacle",
                        markets=[
                            OddsMarket(
                                key="h2h",
                                outcomes=[
                                    OddsOutcome(name=home, price=h_price),
                                    OddsOutcome(name=away, price=a_price),
                                    OddsOutcome(name="Draw", price=d_price),
                                ],
                            )
                        ],
                    )
                ],
            )
        )
    return events


class TestDerivedMatching:
    """EventMatcher.match_seasonal_via_match_odds pipeline."""

    def test_derives_prob_from_matches(self) -> None:
        matcher = EventMatcher(threshold=0.7)
        markets = [
            _make_gamma_market("c1", "Will Arsenal win the 2025\u201326 English Premier League?"),
        ]
        events = _make_epl_events()

        matched = matcher.match_seasonal_via_match_odds(markets, events)
        assert len(matched) == 1
        assert matched[0].match_type == "seasonal_derived"
        assert matched[0].outright_team is not None
        # Arsenal has Pinnacle probs from 3 matches — average should be reasonable
        assert matched[0].pinnacle_prob is not None
        assert Decimal("0.20") < matched[0].pinnacle_prob < Decimal("0.80")

    def test_multiple_teams_matched(self) -> None:
        matcher = EventMatcher(threshold=0.7)
        markets = [
            _make_gamma_market("c1", "Will Arsenal win the 2025\u201326 English Premier League?"),
            _make_gamma_market("c2", "Will Liverpool win the 2025\u201326 English Premier League?"),
        ]
        events = _make_epl_events()

        matched = matcher.match_seasonal_via_match_odds(markets, events)
        assert len(matched) == 2
        teams = {m.outright_team for m in matched}
        assert "Arsenal" in teams
        assert "Liverpool" in teams

    def test_league_filter(self) -> None:
        """Only matches events from the right league."""
        matcher = EventMatcher(threshold=0.7)
        markets = [
            _make_gamma_market("c1", "Will Arsenal win La Liga?"),
        ]
        events = _make_epl_events()  # All EPL events

        matched = matcher.match_seasonal_via_match_odds(markets, events)
        assert len(matched) == 0  # Arsenal is in EPL events but question is La Liga

    def test_no_match_events(self) -> None:
        matcher = EventMatcher(threshold=0.7)
        markets = [
            _make_gamma_market("c1", "Will Arsenal win the 2025\u201326 English Premier League?"),
        ]

        matched = matcher.match_seasonal_via_match_odds(markets, [])
        assert len(matched) == 0


# ================================================================
# Market features building
# ================================================================


class TestBuildMarketFeatures:
    """build_market_features correctness."""

    def test_basic_features(self) -> None:
        market = _make_gamma_market(price_yes="0.35", volume_24h="5000", liquidity="10000")
        pair = MatchedPair(
            polymarket=market,
            odds_event=None,
            match_score=0.95,
            pinnacle_prob=Decimal("0.40"),
            match_type="seasonal",
        )

        features = build_market_features(pair, market)
        assert features.pinnacle_prob == 0.40
        assert features.polymarket_mid == 0.35
        assert features.category == "soccer"
        assert features.volume_24h == 5000.0
        assert features.liquidity == 10000.0

    def test_missing_pinnacle_gives_none(self) -> None:
        market = _make_gamma_market()
        pair = MatchedPair(
            polymarket=market,
            odds_event=None,
            match_score=0.9,
            pinnacle_prob=None,
        )
        features = build_market_features(pair, market)
        assert features.pinnacle_prob is None

    def test_time_to_event_calculated(self) -> None:
        market = _make_gamma_market(end_date="2027-01-01T00:00:00Z")
        pair = MatchedPair(
            polymarket=market,
            odds_event=None,
            match_score=0.9,
            pinnacle_prob=Decimal("0.5"),
        )
        features = build_market_features(pair, market)
        assert features.time_to_event_hours is not None
        assert features.time_to_event_hours > 0  # Future date


# ================================================================
# Signal evaluation
# ================================================================


class TestEvaluatePair:
    """evaluate_pair edge calculation and signal generation."""

    def test_generates_signal_with_edge(self) -> None:
        """Pinnacle 0.45, Polymarket 0.35 → ~10% edge → signal."""
        market = _make_gamma_market(price_yes="0.35")
        pair = MatchedPair(
            polymarket=market,
            odds_event=None,
            match_score=0.95,
            pinnacle_prob=Decimal("0.45"),
            match_type="seasonal",
        )

        signal = evaluate_pair(pair, edge_threshold=0.03)
        assert signal is not None
        assert signal.layer == 2
        assert signal.direction == SignalDirection.BUY_YES
        assert signal.edge > Decimal("0.03")

    def test_no_signal_below_threshold(self) -> None:
        """Tiny edge → no signal."""
        market = _make_gamma_market(price_yes="0.35")
        pair = MatchedPair(
            polymarket=market,
            odds_event=None,
            match_score=0.95,
            pinnacle_prob=Decimal("0.36"),  # Only 1% edge
            match_type="seasonal",
        )

        signal = evaluate_pair(pair, edge_threshold=0.03)
        assert signal is None

    def test_buy_no_direction(self) -> None:
        """Model says lower than Poly → BUY NO."""
        market = _make_gamma_market(price_yes="0.55")
        pair = MatchedPair(
            polymarket=market,
            odds_event=None,
            match_score=0.95,
            pinnacle_prob=Decimal("0.45"),  # 10% below Poly
            match_type="seasonal",
        )

        signal = evaluate_pair(pair, edge_threshold=0.03)
        assert signal is not None
        assert signal.direction == SignalDirection.BUY_NO

    def test_fee_reduces_edge(self) -> None:
        """Fee-enabled market reduces effective edge."""
        market = _make_gamma_market(price_yes="0.50", fee_enabled=True)
        pair = MatchedPair(
            polymarket=market,
            odds_event=None,
            match_score=0.95,
            pinnacle_prob=Decimal("0.54"),  # 4% raw edge
            match_type="seasonal",
        )

        # Without fee: 4% edge
        # With fee at 0.50: 0.50 * 0.50 * 0.0315 ≈ 0.79%
        # Net edge ≈ 3.2% → still above 3%
        signal = evaluate_pair(pair, edge_threshold=0.03)
        assert signal is not None
        assert signal.edge < Decimal("0.04")  # Less than raw 4%

    def test_no_signal_without_pinnacle(self) -> None:
        market = _make_gamma_market()
        pair = MatchedPair(
            polymarket=market,
            odds_event=None,
            match_score=0.9,
            pinnacle_prob=None,
        )

        signal = evaluate_pair(pair, edge_threshold=0.03)
        assert signal is None

    def test_signal_details_complete(self) -> None:
        """Signal details contain audit trail."""
        market = _make_gamma_market(price_yes="0.35")
        pair = MatchedPair(
            polymarket=market,
            odds_event=None,
            match_score=0.95,
            pinnacle_prob=Decimal("0.45"),
            match_type="seasonal",
            outright_team="Arsenal",
            sport_key="soccer_epl",
        )

        signal = evaluate_pair(pair, edge_threshold=0.03)
        assert signal is not None
        assert "pinnacle_prob" in signal.details
        assert "model_prob" in signal.details
        assert "poly_price" in signal.details
        assert "fee" in signal.details
        assert signal.details["match_type"] == "seasonal"
        assert signal.details["outright_team"] == "Arsenal"

    def test_confidence_scales_with_edge(self) -> None:
        """Higher edge → higher confidence."""
        market_small = _make_gamma_market(price_yes="0.35")
        pair_small = MatchedPair(
            polymarket=market_small,
            odds_event=None,
            match_score=0.95,
            pinnacle_prob=Decimal("0.40"),
        )

        market_large = _make_gamma_market(price_yes="0.30")
        pair_large = MatchedPair(
            polymarket=market_large,
            odds_event=None,
            match_score=0.95,
            pinnacle_prob=Decimal("0.45"),
        )

        sig_small = evaluate_pair(pair_small, edge_threshold=0.03)
        sig_large = evaluate_pair(pair_large, edge_threshold=0.03)

        assert sig_small is not None
        assert sig_large is not None
        assert sig_large.confidence > sig_small.confidence

    def test_with_value_model(self) -> None:
        """When ValueModel is trained, uses model prediction instead of Pinnacle."""
        market = _make_gamma_market(price_yes="0.35")
        pair = MatchedPair(
            polymarket=market,
            odds_event=None,
            match_score=0.95,
            pinnacle_prob=Decimal("0.40"),
            match_type="seasonal",
        )

        # Mock a trained model that predicts 0.50
        mock_model = MagicMock()
        mock_model.is_trained = True
        mock_model.predict_single.return_value = 0.50

        signal = evaluate_pair(pair, edge_threshold=0.03, value_model=mock_model)
        assert signal is not None
        # Model says 0.50, poly is 0.35 → edge ~15%
        assert signal.edge > Decimal("0.10")
        assert signal.details["used_model"] is True


# ================================================================
# ValueSignalGenerator orchestrator
# ================================================================


class TestValueSignalGenerator:
    """ValueSignalGenerator.scan() orchestration."""

    @pytest.fixture
    def mock_discovery(self) -> MagicMock:
        discovery = MagicMock()
        discovery.get_by_category.return_value = [
            _make_gamma_market("c1", "Will Arsenal win the Premier League?", price_yes="0.35"),
            _make_gamma_market("c2", "Will Barcelona win La Liga?", price_yes="0.30"),
            _make_gamma_market("c3", "Arsenal vs Chelsea?", price_yes="0.55"),
        ]
        return discovery

    @pytest.fixture
    def mock_odds_client(self) -> AsyncMock:
        client = AsyncMock()
        client.get_soccer_odds.return_value = [
            OddsEvent(
                id="e1",
                sport_key="soccer_epl",
                home_team="Arsenal",
                away_team="Chelsea",
                commence_time="2026-03-01T15:00:00Z",
                bookmakers=[
                    OddsBookmaker(
                        key="pinnacle",
                        title="Pinnacle",
                        markets=[
                            OddsMarket(
                                key="h2h",
                                outcomes=[
                                    OddsOutcome(name="Arsenal", price=Decimal("1.50")),
                                    OddsOutcome(name="Chelsea", price=Decimal("5.00")),
                                    OddsOutcome(name="Draw", price=Decimal("4.50")),
                                ],
                            )
                        ],
                    )
                ],
            )
        ]
        client.get_all_soccer_outrights.return_value = _make_outright_odds()
        return client

    @pytest.fixture
    def matcher(self) -> EventMatcher:
        return EventMatcher(threshold=0.6)  # Relaxed for test names

    async def test_scan_returns_signals(
        self, mock_discovery: MagicMock, mock_odds_client: AsyncMock, matcher: EventMatcher
    ) -> None:
        gen = ValueSignalGenerator(
            discovery=mock_discovery,
            odds_client=mock_odds_client,
            matcher=matcher,
            edge_threshold=0.03,
        )

        signals = await gen.scan()
        # Should find at least the seasonal Arsenal EPL match
        assert len(signals) >= 1
        assert all(isinstance(s, Signal) for s in signals)
        assert all(s.layer == 2 for s in signals)

    async def test_scan_increments_counters(
        self, mock_discovery: MagicMock, mock_odds_client: AsyncMock, matcher: EventMatcher
    ) -> None:
        gen = ValueSignalGenerator(
            discovery=mock_discovery,
            odds_client=mock_odds_client,
            matcher=matcher,
        )

        assert gen.scan_count == 0
        await gen.scan()
        assert gen.scan_count == 1

        await gen.scan()
        assert gen.scan_count == 2

    async def test_scan_no_soccer_markets(
        self, mock_odds_client: AsyncMock, matcher: EventMatcher
    ) -> None:
        """Empty soccer catalog → no signals."""
        empty_discovery = MagicMock()
        empty_discovery.get_by_category.return_value = []

        gen = ValueSignalGenerator(
            discovery=empty_discovery,
            odds_client=mock_odds_client,
            matcher=matcher,
        )

        signals = await gen.scan()
        assert signals == []

    async def test_scan_no_odds_data(
        self, mock_discovery: MagicMock, matcher: EventMatcher
    ) -> None:
        """No Pinnacle odds → no signals."""
        empty_client = AsyncMock()
        empty_client.get_soccer_odds.return_value = []
        empty_client.get_all_soccer_outrights.return_value = {}

        gen = ValueSignalGenerator(
            discovery=mock_discovery,
            odds_client=empty_client,
            matcher=matcher,
        )

        signals = await gen.scan()
        assert signals == []

    async def test_scan_with_value_model(
        self, mock_discovery: MagicMock, mock_odds_client: AsyncMock, matcher: EventMatcher
    ) -> None:
        """With trained model, signals use model predictions."""
        mock_model = MagicMock()
        mock_model.is_trained = True
        mock_model.predict_single.return_value = 0.50

        gen = ValueSignalGenerator(
            discovery=mock_discovery,
            odds_client=mock_odds_client,
            matcher=matcher,
            value_model=mock_model,
            edge_threshold=0.03,
        )

        signals = await gen.scan()
        # With model predicting 0.50 and poly prices at 0.30-0.35,
        # should generate signals
        assert len(signals) >= 1
        for sig in signals:
            assert sig.details["used_model"] is True

    async def test_scan_edge_threshold_filtering(
        self, mock_discovery: MagicMock, mock_odds_client: AsyncMock, matcher: EventMatcher
    ) -> None:
        """Higher threshold → fewer signals."""
        gen_loose = ValueSignalGenerator(
            discovery=mock_discovery,
            odds_client=mock_odds_client,
            matcher=matcher,
            edge_threshold=0.01,
        )

        gen_strict = ValueSignalGenerator(
            discovery=mock_discovery,
            odds_client=mock_odds_client,
            matcher=matcher,
            edge_threshold=0.15,
        )

        signals_loose = await gen_loose.scan()
        signals_strict = await gen_strict.scan()

        assert len(signals_loose) >= len(signals_strict)
