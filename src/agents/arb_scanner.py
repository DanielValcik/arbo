"""Arbitrage scanner — detects cross-platform and same-exchange arbs.

Compares Matchbook lay odds with bookmaker back odds to find
guaranteed profit opportunities.
"""

from __future__ import annotations

from decimal import Decimal

from pydantic import BaseModel

from src.data.event_matcher import MatchedEvent  # noqa: TC001
from src.utils.logger import get_logger
from src.utils.odds import commission_adjusted_odds

log = get_logger("arb_scanner")


class ArbOpportunity(BaseModel):
    event_name: str
    market_type: str
    selection: str
    back_source: str
    back_odds: Decimal
    lay_source: str
    lay_odds: Decimal
    edge: Decimal
    commission: Decimal
    recommended_stake: Decimal | None = None


class ArbScanner:
    """Scans matched events for arbitrage opportunities."""

    def __init__(
        self,
        commission: Decimal = Decimal("0.04"),
        min_edge: Decimal = Decimal("0.02"),
    ) -> None:
        self._commission = commission
        self._min_edge = min_edge

    def scan(self, matched_events: list[MatchedEvent]) -> list[ArbOpportunity]:
        """Scan all matched events for arb opportunities."""
        opportunities: list[ArbOpportunity] = []

        for matched in matched_events:
            cross = self._find_cross_platform_arbs(matched)
            opportunities.extend(cross)

            exchange = self._find_same_exchange_arbs(matched)
            opportunities.extend(exchange)

        log.info("arb_scan_complete", events=len(matched_events), arbs_found=len(opportunities))
        return opportunities

    def _find_cross_platform_arbs(self, matched: MatchedEvent) -> list[ArbOpportunity]:
        """Find arbs: bookmaker BACK vs Matchbook LAY.

        For each runner in Matchbook with a lay price, compare against
        each bookmaker's back price for the same selection.
        """
        arbs: list[ArbOpportunity] = []
        mb = matched.matchbook_event
        oa = matched.odds_api_event
        event_name = f"{mb.home_team} vs {mb.away_team}"

        for market in mb.markets:
            for runner in market.runners:
                lay = runner.best_lay
                if not lay:
                    continue

                adj_lay = commission_adjusted_odds(lay.odds, self._commission)

                # Check each bookmaker for better back odds
                for bookmaker in oa.bookmakers:
                    for bm_market in bookmaker.markets:
                        if not self._markets_compatible(market.market_type, bm_market.key):
                            continue

                        for outcome in bm_market.outcomes:
                            if not self._selections_match(runner.name, outcome.name):
                                continue

                            back_odds = outcome.price
                            margin = Decimal(1) / back_odds + Decimal(1) / adj_lay - Decimal(1)

                            if margin < 0:
                                edge = -margin
                                if edge >= self._min_edge:
                                    arbs.append(
                                        ArbOpportunity(
                                            event_name=event_name,
                                            market_type=market.market_type,
                                            selection=runner.name,
                                            back_source=bookmaker.key,
                                            back_odds=back_odds,
                                            lay_source="matchbook",
                                            lay_odds=lay.odds,
                                            edge=edge,
                                            commission=self._commission,
                                        )
                                    )
                                    log.info(
                                        "arb_found",
                                        match=event_name,
                                        selection=runner.name,
                                        back=f"{back_odds}@{bookmaker.key}",
                                        lay=f"{lay.odds}@matchbook",
                                        edge=f"{edge:.4f}",
                                    )

        return arbs

    def _find_same_exchange_arbs(self, matched: MatchedEvent) -> list[ArbOpportunity]:
        """Find arbs within Matchbook: BACK selection A vs LAY selection B.

        Only applies to h2h 3-way markets (Home/Draw/Away).
        If sum(1/back_i for all i) < 1, there's a same-exchange arb.
        """
        arbs: list[ArbOpportunity] = []
        mb = matched.matchbook_event
        event_name = f"{mb.home_team} vs {mb.away_team}"

        for market in mb.markets:
            if market.market_type != "h2h" or len(market.runners) < 3:
                continue

            # Collect best back odds for all runners
            back_prices: list[tuple[str, Decimal]] = []
            for runner in market.runners:
                back = runner.best_back
                if back:
                    adj = commission_adjusted_odds(back.odds, self._commission)
                    back_prices.append((runner.name, adj))

            if len(back_prices) < 3:
                continue

            # Check if sum of implied probs < 1
            total_implied = sum(Decimal(1) / odds for _, odds in back_prices)
            if total_implied < Decimal(1):
                edge = Decimal(1) - total_implied
                if edge >= self._min_edge:
                    arbs.append(
                        ArbOpportunity(
                            event_name=event_name,
                            market_type="h2h",
                            selection="all_outcomes",
                            back_source="matchbook",
                            back_odds=Decimal(0),
                            lay_source="matchbook",
                            lay_odds=Decimal(0),
                            edge=edge,
                            commission=self._commission,
                        )
                    )
                    log.info(
                        "same_exchange_arb_found",
                        match=event_name,
                        edge=f"{edge:.4f}",
                        total_implied=f"{total_implied:.4f}",
                    )

        return arbs

    @staticmethod
    def _markets_compatible(mb_type: str, oa_key: str) -> bool:
        """Check if Matchbook market type maps to Odds API market key."""
        mapping = {
            "h2h": "h2h",
            "one_x_two": "h2h",
            "spreads": "spreads",
            "totals": "totals",
        }
        return mapping.get(mb_type) == oa_key

    @staticmethod
    def _selections_match(mb_name: str, oa_name: str) -> bool:
        """Check if two selection names refer to the same outcome."""
        return mb_name.lower().strip() == oa_name.lower().strip()
