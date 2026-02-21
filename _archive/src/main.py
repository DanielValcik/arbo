"""Arbo main entry point.

Asyncio event loop running Matchbook + Odds API pollers, event matching,
arb scanning, paper bet logging, and Slack bot.
Handles graceful shutdown, error recovery, and kill switch.
"""

from __future__ import annotations

import argparse
import asyncio
import signal
import time
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import redis.asyncio as aioredis
from sqlalchemy import insert, select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from src.agents.arb_scanner import ArbScanner
from src.alerts.slack_bot import ArboSlackBot
from src.data.event_matcher import EventMatcher, load_aliases
from src.data.odds_api import OddsApiClient
from src.exchanges.base import ExchangeEvent  # noqa: TC001
from src.exchanges.matchbook import MatchbookClient, MatchbookError
from src.execution.position_tracker import PaperTracker
from src.utils.config import get_config
from src.utils.db import (
    Event,
    OddsSnapshot,
    Opportunity,
    get_engine,
    get_session_factory,
)
from src.utils.logger import get_logger, setup_logging

log = get_logger("main")

# Global kill switch — set by /kill command or repeated crashes
KILL_SWITCH = False


class Arbo:
    """Main application orchestrator."""

    def __init__(self) -> None:
        self.config = get_config()
        self.matchbook: MatchbookClient | None = None
        self.redis: aioredis.Redis | None = None  # type: ignore[type-arg]
        self.odds_api: OddsApiClient | None = None
        self.matcher: EventMatcher | None = None
        self.arb_scanner: ArbScanner | None = None
        self.slack_bot: ArboSlackBot | None = None
        self._shutdown_event = asyncio.Event()
        self._error_tracker: dict[str, list[float]] = defaultdict(list)
        self._consecutive_errors: int = 0
        self._background_tasks: set[asyncio.Task[Any]] = set()

        # Shared state between pollers
        self._matchbook_events: list[ExchangeEvent] = []
        self._event_id_map: dict[int, int] = {}  # external_id → DB id
        self._last_odds_api_poll: float = 0.0

    async def startup(self) -> None:
        """Initialize all connections and Sprint 2 components."""
        log.info("arbo_starting", mode=self.config.mode)

        # Redis
        try:
            self.redis = aioredis.from_url(self.config.redis_url, decode_responses=True)
            await self.redis.ping()
            log.info("redis_connected")
        except Exception as e:
            log.warning("redis_connection_failed", error=str(e))
            self.redis = None

        # Matchbook
        self.matchbook = MatchbookClient(redis_client=self.redis)
        await self.matchbook.login()
        log.info("matchbook_connected")

        # Odds API
        if self.config.odds_api_key:
            self.odds_api = OddsApiClient()
            log.info("odds_api_initialized")

        # Event matcher
        aliases = load_aliases()
        self.matcher = EventMatcher(aliases=aliases)
        log.info("event_matcher_initialized", alias_count=len(aliases))

        # Arb scanner
        self.arb_scanner = ArbScanner(
            commission=Decimal(str(self.config.matchbook.commission_pct)),
            min_edge=Decimal(str(self.config.thresholds.min_edge)),
        )

        # Slack bot
        if self.config.slack_bot_token and self.config.slack_app_token:
            self.slack_bot = ArboSlackBot(
                bot_token=self.config.slack_bot_token,
                app_token=self.config.slack_app_token,
                channel_id=self.config.slack_channel_id,
            )
            self.slack_bot._on_kill = self._activate_kill_switch
            task = asyncio.create_task(self.slack_bot.start())
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)
            log.info("slack_bot_started")

    async def shutdown(self) -> None:
        """Gracefully close all connections."""
        log.info("arbo_shutting_down")
        if self.matchbook:
            await self.matchbook.close()
        if self.odds_api:
            await self.odds_api.close()
        if self.redis:
            await self.redis.close()

        engine = get_engine()
        await engine.dispose()
        log.info("arbo_shutdown_complete")

    def _activate_kill_switch(self) -> None:
        """Activate kill switch (called from Slack /kill command)."""
        global KILL_SWITCH
        KILL_SWITCH = True
        self._shutdown_event.set()
        log.critical("kill_switch_activated")

    def _check_repeated_errors(self, error_type: str) -> bool:
        """Check if same error type occurred 3x in 10 minutes. Returns True if should kill."""
        now = time.monotonic()
        self._error_tracker[error_type] = [
            t for t in self._error_tracker[error_type] if now - t < 600
        ]
        self._error_tracker[error_type].append(now)
        return len(self._error_tracker[error_type]) >= 3

    async def _send_error_if_slack(self, error: str) -> None:
        """Send error alert via Slack if bot is configured."""
        self._consecutive_errors += 1
        if self.slack_bot and self._consecutive_errors >= 3:
            await self.slack_bot.send_error_alert(error)
            self._consecutive_errors = 0

    async def _upsert_events(self, events: list[Any]) -> dict[int, int]:
        """Upsert events into database. Returns mapping of external_id -> db_id."""
        session_factory = get_session_factory()
        id_map: dict[int, int] = {}

        async with session_factory() as session:
            for event in events:
                stmt = pg_insert(Event).values(
                    external_id=str(event.event_id),
                    source="matchbook",
                    sport=event.sport,
                    league=event.league,
                    home_team=event.home_team,
                    away_team=event.away_team,
                    start_time=event.start_time,
                    status=event.status.value,
                )
                stmt = stmt.on_conflict_do_update(
                    constraint="uq_events_source_external",
                    set_={
                        "status": stmt.excluded.status,
                        "updated_at": datetime.now(UTC),
                    },
                )
                await session.execute(stmt)

                # Get the DB id
                result = await session.execute(
                    select(Event.id).where(
                        Event.source == "matchbook",
                        Event.external_id == str(event.event_id),
                    )
                )
                row = result.scalar_one_or_none()
                if row:
                    id_map[event.event_id] = row

            await session.commit()

        return id_map

    async def _write_odds_snapshots(self, events: list[Any], id_map: dict[int, int]) -> int:
        """Batch write odds snapshots to database. Returns count written."""
        snapshots = []

        for event in events:
            db_id = id_map.get(event.event_id)
            if not db_id:
                continue

            for market in event.markets:
                for runner in market.runners:
                    best_back = runner.best_back
                    best_lay = runner.best_lay

                    if best_back or best_lay:
                        snapshots.append(
                            {
                                "event_id": db_id,
                                "source": "matchbook",
                                "market_type": market.market_type,
                                "selection": runner.name,
                                "back_odds": float(best_back.odds) if best_back else None,
                                "lay_odds": float(best_lay.odds) if best_lay else None,
                                "back_stake": (
                                    float(best_back.available_amount) if best_back else None
                                ),
                                "lay_stake": (
                                    float(best_lay.available_amount) if best_lay else None
                                ),
                                "bookmaker": None,
                            }
                        )

        if not snapshots:
            return 0

        session_factory = get_session_factory()
        async with session_factory() as session:
            await session.execute(insert(OddsSnapshot), snapshots)
            await session.commit()

        return len(snapshots)

    async def _store_opportunities(self, arbs: list[Any], id_map: dict[int, int]) -> None:
        """Store detected arb opportunities in the database."""
        if not arbs:
            return

        session_factory = get_session_factory()
        async with session_factory() as session:
            for arb in arbs:
                event_id = id_map.get(arb.matchbook_event_id, 0)
                if event_id == 0:
                    log.warning(
                        "store_opp_missing_event",
                        matchbook_event_id=arb.matchbook_event_id,
                    )
                    continue

                opp = Opportunity(
                    event_id=event_id,
                    strategy="arb",
                    expected_edge=arb.edge,
                    details={
                        "event_name": arb.event_name,
                        "market_type": arb.market_type,
                        "selection": arb.selection,
                        "back_source": arb.back_source,
                        "back_odds": str(arb.back_odds),
                        "lay_source": arb.lay_source,
                        "lay_odds": str(arb.lay_odds),
                        "commission": str(arb.commission),
                    },
                    status="detected",
                )
                session.add(opp)

            await session.commit()

    async def poll_matchbook(self) -> None:
        """Single Matchbook polling cycle: fetch events -> write snapshots."""
        if self.matchbook is None:
            raise RuntimeError("Matchbook client not initialized")

        total_snapshots = 0
        all_events: list[ExchangeEvent] = []

        for sport in self.config.sports:
            now = datetime.now(UTC)
            events = await self.matchbook.get_events(
                sport=sport,
                date_from=now,
                date_to=now + timedelta(days=7),
            )

            if not events:
                continue

            # Upsert events
            id_map = await self._upsert_events(events)
            self._event_id_map.update(id_map)

            # For markets without embedded prices, fetch them individually
            for event in events:
                for market in event.markets:
                    has_prices = any(r.prices for r in market.runners)
                    if not has_prices:
                        runners = await self.matchbook.get_prices(event.event_id, market.market_id)
                        market.runners = runners

            # Write snapshots
            count = await self._write_odds_snapshots(events, id_map)
            total_snapshots += count
            all_events.extend(events)

        self._matchbook_events = all_events
        log.info("matchbook_poll_complete", snapshots_written=total_snapshots)

        # Update Slack bot state
        if self.slack_bot:
            self.slack_bot._last_poll_time = datetime.now(UTC).strftime("%H:%M:%S")
            self.slack_bot._daily_get_count = getattr(self.matchbook, "_daily_get_count", 0)

    async def poll_odds_api(self) -> None:
        """Fetch odds from The Odds API (runs less frequently than Matchbook)."""
        if self.odds_api is None:
            return

        now = time.monotonic()
        interval = self.config.polling.odds_api_batch
        if now - self._last_odds_api_poll < interval:
            return  # Not time yet

        self._last_odds_api_poll = now

        try:
            oa_events = await self.odds_api.get_all_odds()
            log.info("odds_api_poll_complete", events=len(oa_events))

            # Update Slack bot quota
            if self.slack_bot and self.odds_api.remaining_quota is not None:
                self.slack_bot._odds_api_quota = self.odds_api.remaining_quota

            # Match events + scan for arbs
            if self._matchbook_events and oa_events and self.matcher and self.arb_scanner:
                matched = self.matcher.match_events(self._matchbook_events, oa_events)
                if matched:
                    arbs = self.arb_scanner.scan(matched)
                    if arbs:
                        log.info("arbs_detected", count=len(arbs))

                        # Store opportunities + paper bets using DB id map
                        await self._store_opportunities(arbs, self._event_id_map)
                        await self._log_paper_bets(arbs, self._event_id_map)

                        # Alert via Slack
                        if self.slack_bot:
                            for arb in arbs:
                                await self.slack_bot.send_arb_alert(arb)

            # Full Odds API cycle succeeded — reset error counter
            self._consecutive_errors = 0

        except Exception as e:
            log.error("odds_api_poll_error", error=str(e))
            await self._send_error_if_slack(f"Odds API error: {e}")

    async def _log_paper_bets(self, arbs: list[Any], id_map: dict[int, int]) -> None:
        """Log paper bets for detected arbs."""
        if self.config.mode != "paper":
            return

        session_factory = get_session_factory()
        async with session_factory() as session:
            tracker = PaperTracker(
                session=session,
                bankroll=Decimal(str(self.config.bankroll)),
            )
            for arb in arbs:
                # Same-exchange arbs (back_odds=0) can't be paper-bet as single leg
                if arb.back_odds <= 1:
                    continue
                event_id = id_map.get(arb.matchbook_event_id, 0)
                if event_id == 0:
                    log.warning(
                        "paper_bet_missing_event",
                        matchbook_event_id=arb.matchbook_event_id,
                    )
                    continue
                try:
                    bet_id = await tracker.place_paper_bet(arb, event_id=event_id)
                    log.info("paper_bet_logged", bet_id=bet_id, selection=arb.selection)
                except Exception as e:
                    log.error("paper_bet_error", error=str(e))

            await session.commit()

    async def run_polling_loop(self) -> None:
        """Main polling loop with error handling per Section 8."""
        global KILL_SWITCH
        interval = self.config.polling.matchbook

        while not self._shutdown_event.is_set() and not KILL_SWITCH:
            try:
                await self.poll_matchbook()
            except MatchbookError as e:
                error_type = type(e).__name__
                log.error("poll_error", error_type=error_type, error=str(e))
                await self._send_error_if_slack(f"Matchbook {error_type}: {e}")

                if self._check_repeated_errors(error_type):
                    log.critical("repeated_errors_kill_switch", error_type=error_type)
                    KILL_SWITCH = True
                    break
            except Exception as e:
                error_type = type(e).__name__
                log.error("poll_unhandled_error", error_type=error_type, error=str(e))
                await self._send_error_if_slack(f"Unhandled {error_type}: {e}")

                if self._check_repeated_errors(error_type):
                    log.critical("repeated_errors_kill_switch", error_type=error_type)
                    KILL_SWITCH = True
                    break

                # Sleep 60s before resuming on unhandled exception
                await asyncio.sleep(60)
                continue

            # Poll Odds API (respects its own interval internally)
            await self.poll_odds_api()

            try:
                await asyncio.wait_for(self._shutdown_event.wait(), timeout=interval)
                break  # Shutdown requested
            except TimeoutError:
                pass  # Normal: timeout means it's time to poll again

    def request_shutdown(self) -> None:
        """Signal the polling loop to stop."""
        self._shutdown_event.set()


async def fetch_test() -> None:
    """One-shot fetch test: login, fetch events from both sources, print matches."""
    setup_logging(log_level="INFO", json_output=False)
    log.info("fetch_test_starting")

    config = get_config()
    redis_client = None
    try:
        redis_client = aioredis.from_url(config.redis_url, decode_responses=True)
        await redis_client.ping()
    except Exception:
        redis_client = None

    client = MatchbookClient(redis_client=redis_client)
    odds_api = OddsApiClient() if config.odds_api_key else None

    try:
        await client.login()

        # Discover sport IDs
        sports = await client.get_sports()
        print("\n=== Matchbook Sport IDs ===")
        for s in sports:
            print(f"  {s.get('name', '?')}: id={s.get('id', '?')}")

        all_mb_events: list[ExchangeEvent] = []

        # Fetch Matchbook events
        for sport in config.sports:
            now = datetime.now(UTC)
            events = await client.get_events(
                sport=sport,
                date_from=now,
                date_to=now + timedelta(days=3),
            )
            print(f"\n=== {sport.upper()} Events ({len(events)}) ===")
            for ev in events[:15]:
                print(f"  [{ev.event_id}] {ev.home_team} vs {ev.away_team}")
                print(f"    Start: {ev.start_time}, League: {ev.league}")
                for m in ev.markets[:2]:
                    print(f"    Market: {m.name} ({m.market_type})")
                    for r in m.runners:
                        back = r.best_back
                        lay = r.best_lay
                        print(
                            f"      {r.name}: "
                            f"back={back.odds if back else '-'} "
                            f"lay={lay.odds if lay else '-'}"
                        )
            all_mb_events.extend(events)

        # Fetch Odds API events and match
        if odds_api:
            oa_events = await odds_api.get_all_odds()
            print(f"\n=== Odds API Events ({len(oa_events)}) ===")
            for ev in oa_events[:10]:
                print(f"  [{ev.id}] {ev.home_team} vs {ev.away_team} ({ev.sport_key})")
                print(f"    Bookmakers: {len(ev.bookmakers)}")

            # Match events
            aliases = load_aliases()
            matcher = EventMatcher(aliases=aliases)
            matched = matcher.match_events(all_mb_events, oa_events)
            print(f"\n=== Matched Events ({len(matched)}) ===")
            for m in matched:
                mb = m.matchbook_event
                oa = m.odds_api_event
                print(
                    f"  {mb.home_team} vs {mb.away_team} <-> "
                    f"{oa.home_team} vs {oa.away_team} "
                    f"(score={m.match_score:.3f})"
                )

            # Scan for arbs
            scanner = ArbScanner(
                commission=Decimal(str(config.matchbook.commission_pct)),
                min_edge=Decimal(str(config.thresholds.min_edge)),
            )
            arbs = scanner.scan(matched)
            print(f"\n=== Arb Opportunities ({len(arbs)}) ===")
            for arb in arbs:
                print(
                    f"  {arb.event_name} | {arb.selection} | "
                    f"back={arb.back_odds}@{arb.back_source} | "
                    f"lay={arb.lay_odds}@{arb.lay_source} | "
                    f"edge={float(arb.edge) * 100:.2f}%"
                )

    finally:
        await client.close()
        if odds_api:
            await odds_api.close()
        if redis_client:
            await redis_client.close()


async def run() -> None:
    """Main async entry point."""
    config = get_config()
    setup_logging(log_level=config.log_level)

    app = Arbo()

    # Signal handlers for graceful shutdown
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, app.request_shutdown)

    try:
        await app.startup()
        await app.run_polling_loop()
    except Exception as e:
        log.critical("fatal_error", error=str(e))
    finally:
        await app.shutdown()


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Arbo — Sports Betting Intelligence")
    parser.add_argument(
        "--fetch-test",
        action="store_true",
        help="Fetch events once, print to stdout, and exit",
    )
    args = parser.parse_args()

    if args.fetch_test:
        asyncio.run(fetch_test())
    else:
        asyncio.run(run())


if __name__ == "__main__":
    main()
