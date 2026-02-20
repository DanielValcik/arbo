"""Tests for arbitrage scanner."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from src.agents.arb_scanner import ArbScanner
from src.data.event_matcher import MatchedEvent
from src.data.odds_api import OddsApiBookmaker, OddsApiEvent, OddsApiMarket, OddsApiOutcome
from src.exchanges.base import (
    EventStatus,
    ExchangeEvent,
    Market,
    Runner,
    RunnerPrice,
    Side,
)

START = datetime(2026, 3, 1, 15, 0, tzinfo=UTC)


def _mb_event(
    runners: list[Runner] | None = None,
    market_type: str = "h2h",
) -> ExchangeEvent:
    """Create a Matchbook event with given runners."""
    if runners is None:
        runners = [
            Runner(
                runner_id=1,
                name="Liverpool",
                prices=[
                    RunnerPrice(odds=Decimal("2.10"), available_amount=500, side=Side.BACK),
                    RunnerPrice(odds=Decimal("2.14"), available_amount=300, side=Side.LAY),
                ],
            ),
            Runner(
                runner_id=2,
                name="Draw",
                prices=[
                    RunnerPrice(odds=Decimal("3.40"), available_amount=200, side=Side.BACK),
                    RunnerPrice(odds=Decimal("3.50"), available_amount=150, side=Side.LAY),
                ],
            ),
            Runner(
                runner_id=3,
                name="Arsenal",
                prices=[
                    RunnerPrice(odds=Decimal("3.60"), available_amount=400, side=Side.BACK),
                    RunnerPrice(odds=Decimal("3.70"), available_amount=250, side=Side.LAY),
                ],
            ),
        ]
    return ExchangeEvent(
        event_id=100,
        name="Liverpool vs Arsenal",
        sport="football",
        league="EPL",
        home_team="Liverpool",
        away_team="Arsenal",
        start_time=START,
        status=EventStatus.UPCOMING,
        markets=[
            Market(market_id=200, name="Match Odds", market_type=market_type, runners=runners)
        ],
    )


def _oa_event(
    outcomes: list[OddsApiOutcome] | None = None,
    bookmaker_key: str = "bet365",
) -> OddsApiEvent:
    """Create an Odds API event with given outcomes."""
    if outcomes is None:
        outcomes = [
            OddsApiOutcome(name="Liverpool", price=Decimal("2.10")),
            OddsApiOutcome(name="Draw", price=Decimal("3.40")),
            OddsApiOutcome(name="Arsenal", price=Decimal("3.60")),
        ]
    return OddsApiEvent(
        id="abc123",
        sport_key="soccer_epl",
        home_team="Liverpool",
        away_team="Arsenal",
        commence_time=START,
        bookmakers=[
            OddsApiBookmaker(
                key=bookmaker_key,
                title=bookmaker_key.title(),
                markets=[OddsApiMarket(key="h2h", outcomes=outcomes)],
            )
        ],
    )


def _matched(mb: ExchangeEvent | None = None, oa: OddsApiEvent | None = None) -> MatchedEvent:
    return MatchedEvent(
        matchbook_event=mb or _mb_event(),
        odds_api_event=oa or _oa_event(),
        match_score=1.0,
    )


class TestCrossPlatformArb:
    def test_arb_detected(self) -> None:
        """Bookmaker back > Matchbook lay (commission adjusted) = arb."""
        # Bookmaker offers back at 2.30, Matchbook lay at 2.14
        # adj_lay = 1 + 1.14 * 0.96 = 2.0944
        # margin = 1/2.30 + 1/2.0944 - 1 = 0.4348 + 0.4775 - 1 = -0.0877
        oa = _oa_event(
            outcomes=[
                OddsApiOutcome(name="Liverpool", price=Decimal("2.30")),
                OddsApiOutcome(name="Draw", price=Decimal("3.40")),
                OddsApiOutcome(name="Arsenal", price=Decimal("3.60")),
            ]
        )
        scanner = ArbScanner(min_edge=Decimal("0.02"))
        result = scanner.scan([_matched(oa=oa)])

        assert len(result) >= 1
        arb = result[0]
        assert arb.selection == "Liverpool"
        assert arb.back_source == "bet365"
        assert arb.lay_source == "matchbook"
        assert arb.back_odds == Decimal("2.30")
        assert arb.lay_odds == Decimal("2.14")
        assert arb.edge > Decimal("0.02")

    def test_no_arb_when_lay_too_high(self) -> None:
        """No arb when bookmaker back is lower than commission-adjusted lay."""
        # back=1.80, lay=2.14 → adj_lay=2.0944, 1/1.80+1/2.0944 = 0.5556+0.4775 = 1.033 > 1
        oa = _oa_event(
            outcomes=[
                OddsApiOutcome(name="Liverpool", price=Decimal("1.80")),
                OddsApiOutcome(name="Draw", price=Decimal("3.40")),
                OddsApiOutcome(name="Arsenal", price=Decimal("3.60")),
            ]
        )
        scanner = ArbScanner()
        result = scanner.scan([_matched(oa=oa)])

        # Only Liverpool should be checked; no arb expected
        liverpool_arbs = [a for a in result if a.selection == "Liverpool"]
        assert len(liverpool_arbs) == 0

    def test_edge_below_min(self) -> None:
        """Arb with edge below min_edge is not reported."""
        oa = _oa_event(
            outcomes=[
                OddsApiOutcome(name="Liverpool", price=Decimal("2.16")),
                OddsApiOutcome(name="Draw", price=Decimal("3.40")),
                OddsApiOutcome(name="Arsenal", price=Decimal("3.60")),
            ]
        )
        # With min_edge=0.10 the tiny arb won't qualify
        scanner = ArbScanner(min_edge=Decimal("0.10"))
        result = scanner.scan([_matched(oa=oa)])

        liverpool_arbs = [a for a in result if a.selection == "Liverpool"]
        assert len(liverpool_arbs) == 0

    def test_commission_impact(self) -> None:
        """Higher commission reduces arb profitability."""
        oa = _oa_event(
            outcomes=[
                OddsApiOutcome(name="Liverpool", price=Decimal("2.20")),
                OddsApiOutcome(name="Draw", price=Decimal("3.40")),
                OddsApiOutcome(name="Arsenal", price=Decimal("3.60")),
            ]
        )
        # Low commission: arb should exist
        scanner_low = ArbScanner(commission=Decimal("0.02"), min_edge=Decimal("0.02"))
        result_low = scanner_low.scan([_matched(oa=oa)])

        # High commission: arb might not exist
        scanner_high = ArbScanner(commission=Decimal("0.10"), min_edge=Decimal("0.02"))
        result_high = scanner_high.scan([_matched(oa=oa)])

        low_arbs = [a for a in result_low if a.selection == "Liverpool"]
        high_arbs = [a for a in result_high if a.selection == "Liverpool"]

        # Low commission should find more/bigger arbs than high commission
        assert len(low_arbs) >= len(high_arbs)

    def test_no_lay_prices_no_crash(self) -> None:
        """Runners without lay prices are skipped gracefully."""
        runners = [
            Runner(
                runner_id=1,
                name="Liverpool",
                prices=[
                    RunnerPrice(odds=Decimal("2.10"), available_amount=500, side=Side.BACK),
                ],
            ),
        ]
        mb = _mb_event(runners=runners)
        scanner = ArbScanner()
        result = scanner.scan([_matched(mb=mb)])

        assert result == []

    def test_multiple_bookmakers_scanned(self) -> None:
        """Scanner checks all bookmakers, finds best arb."""
        oa = OddsApiEvent(
            id="abc123",
            sport_key="soccer_epl",
            home_team="Liverpool",
            away_team="Arsenal",
            commence_time=START,
            bookmakers=[
                OddsApiBookmaker(
                    key="bet365",
                    title="Bet365",
                    markets=[
                        OddsApiMarket(
                            key="h2h",
                            outcomes=[
                                OddsApiOutcome(name="Liverpool", price=Decimal("2.15")),
                            ],
                        )
                    ],
                ),
                OddsApiBookmaker(
                    key="williamhill",
                    title="William Hill",
                    markets=[
                        OddsApiMarket(
                            key="h2h",
                            outcomes=[
                                OddsApiOutcome(name="Liverpool", price=Decimal("2.35")),
                            ],
                        )
                    ],
                ),
            ],
        )
        scanner = ArbScanner(min_edge=Decimal("0.02"))
        result = scanner.scan([_matched(oa=oa)])

        # Both bookmakers should produce arbs (if edge is sufficient)
        sources = {a.back_source for a in result if a.selection == "Liverpool"}
        assert "williamhill" in sources


class TestSameExchangeArb:
    def test_same_exchange_arb_detected(self) -> None:
        """Back all 3 outcomes if sum of commission-adjusted implied < 1."""
        # Need very generous back odds to get sum < 1 after 4% commission
        runners = [
            Runner(
                runner_id=1,
                name="Liverpool",
                prices=[RunnerPrice(odds=Decimal("3.50"), available_amount=500, side=Side.BACK)],
            ),
            Runner(
                runner_id=2,
                name="Draw",
                prices=[RunnerPrice(odds=Decimal("4.00"), available_amount=200, side=Side.BACK)],
            ),
            Runner(
                runner_id=3,
                name="Arsenal",
                prices=[RunnerPrice(odds=Decimal("4.50"), available_amount=400, side=Side.BACK)],
            ),
        ]
        mb = _mb_event(runners=runners, market_type="h2h")
        scanner = ArbScanner(commission=Decimal("0.02"), min_edge=Decimal("0.01"))
        result = scanner.scan([_matched(mb=mb)])

        exchange_arbs = [a for a in result if a.selection == "all_outcomes"]
        assert len(exchange_arbs) == 1
        assert exchange_arbs[0].edge > Decimal("0.01")

    def test_no_same_exchange_arb_typical(self) -> None:
        """Typical market odds don't produce same-exchange arbs."""
        scanner = ArbScanner()
        result = scanner.scan([_matched()])

        exchange_arbs = [a for a in result if a.selection == "all_outcomes"]
        assert len(exchange_arbs) == 0

    def test_non_h2h_market_skipped(self) -> None:
        """Same-exchange arb only applies to h2h markets."""
        runners = [
            Runner(
                runner_id=1,
                name="Over 2.5",
                prices=[RunnerPrice(odds=Decimal("3.50"), available_amount=500, side=Side.BACK)],
            ),
            Runner(
                runner_id=2,
                name="Under 2.5",
                prices=[RunnerPrice(odds=Decimal("3.50"), available_amount=500, side=Side.BACK)],
            ),
        ]
        mb = _mb_event(runners=runners, market_type="totals")
        scanner = ArbScanner(commission=Decimal("0.00"), min_edge=Decimal("0.01"))
        result = scanner.scan([_matched(mb=mb)])

        exchange_arbs = [a for a in result if a.selection == "all_outcomes"]
        assert len(exchange_arbs) == 0


class TestScanEmpty:
    def test_empty_events(self) -> None:
        scanner = ArbScanner()
        assert scanner.scan([]) == []
