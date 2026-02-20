"""Arbo main entry point.

Asyncio event loop running Matchbook poller at configured interval.
Handles graceful shutdown, error recovery, and kill switch.
"""

from __future__ import annotations

import argparse
import asyncio
import signal
import time
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from typing import Any

import redis.asyncio as aioredis
from sqlalchemy import insert, select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from src.exchanges.matchbook import MatchbookClient, MatchbookError
from src.utils.config import get_config
from src.utils.db import (
    Event,
    OddsSnapshot,
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
        self._shutdown_event = asyncio.Event()
        self._error_tracker: dict[str, list[float]] = defaultdict(list)

    async def startup(self) -> None:
        """Initialize all connections."""
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

    async def shutdown(self) -> None:
        """Gracefully close all connections."""
        log.info("arbo_shutting_down")
        if self.matchbook:
            await self.matchbook.close()
        if self.redis:
            await self.redis.close()

        engine = get_engine()
        await engine.dispose()
        log.info("arbo_shutdown_complete")

    def _check_repeated_errors(self, error_type: str) -> bool:
        """Check if same error type occurred 3x in 10 minutes. Returns True if should kill."""
        now = time.monotonic()
        self._error_tracker[error_type] = [
            t for t in self._error_tracker[error_type] if now - t < 600
        ]
        self._error_tracker[error_type].append(now)
        return len(self._error_tracker[error_type]) >= 3

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
                                "lay_stake": float(best_lay.available_amount) if best_lay else None,
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

    async def poll_matchbook(self) -> None:
        """Single polling cycle: fetch events → write snapshots."""
        if self.matchbook is None:
            raise RuntimeError("Matchbook client not initialized — call startup() first")

        total_snapshots = 0
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

        log.info("poll_cycle_complete", snapshots_written=total_snapshots)

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

                if self._check_repeated_errors(error_type):
                    log.critical("repeated_errors_kill_switch", error_type=error_type)
                    KILL_SWITCH = True
                    break
            except Exception as e:
                error_type = type(e).__name__
                log.error("poll_unhandled_error", error_type=error_type, error=str(e))

                if self._check_repeated_errors(error_type):
                    log.critical("repeated_errors_kill_switch", error_type=error_type)
                    KILL_SWITCH = True
                    break

                # Sleep 60s before resuming on unhandled exception
                await asyncio.sleep(60)
                continue

            try:
                await asyncio.wait_for(self._shutdown_event.wait(), timeout=interval)
                break  # Shutdown requested
            except TimeoutError:
                pass  # Normal: timeout means it's time to poll again

    def request_shutdown(self) -> None:
        """Signal the polling loop to stop."""
        self._shutdown_event.set()


async def fetch_test() -> None:
    """One-shot fetch test: login, fetch events, print to stdout, exit."""
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
    try:
        await client.login()

        # Discover sport IDs
        sports = await client.get_sports()
        print("\n=== Matchbook Sport IDs ===")
        for s in sports:
            print(f"  {s.get('name', '?')}: id={s.get('id', '?')}")

        # Fetch events
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
    finally:
        await client.close()
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
