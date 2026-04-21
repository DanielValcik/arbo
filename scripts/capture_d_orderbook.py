"""Capture NBA orderbook snapshots for future Strategy D ML training.

Polls Polymarket CLOB `/book` endpoint for active NBA moneyline tokens
every N minutes, stores (bid, ask, depth, spread) in
`orderbook_snapshots_d` (alembic migration 015).

**Why:** PolymarketData.co Pass 2 never captured bid/ask
(LEARNINGS D3b). This script builds a forward-captured microstructure
dataset at zero API cost so a future ML iteration (v3) can add
spread + depth + volume_imbalance features.

**Design notes:**
- Runs standalone — NOT wired into main arbo service. Can be started/
  stopped independently via systemd unit.
- Reuses `arbo.connectors.orderbook_provider.OrderbookProvider` for
  book fetch (handles NegRisk semantics + retry).
- Discovers tokens fresh each poll via
  `arbo.strategies.strategy_d_discovery_nba.discover_markets` so new
  games auto-add without restart.
- Write-only: no reads from production tables. Safe to run alongside
  live trading.

**Deployment:**
  1. Run alembic upgrade 015
  2. scp this + systemd unit to arbo-dublin
  3. `sudo systemctl enable --now arbo-capture-d-orderbook.service`

Usage (local):
  PYTHONPATH=. python3 scripts/capture_d_orderbook.py --interval 300 --dry-run
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import signal
import sys
from datetime import datetime, date, timezone
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arbo.utils.logger import get_logger, setup_logging
from arbo.utils.db import get_session_factory

setup_logging()
logger = get_logger("capture_d_orderbook")


# ── Discovery (ephemeral, per poll) ───────────────────────────────────


async def _discover_nba_tokens(window_hours: float = 48.0) -> list[dict]:
    """Find active NBA moneyline markets. Returns list of dicts with
    (token_id_yes, token_id_no, condition_id, neg_risk, game_date, question).

    Uses `strategy_d_discovery_nba.discover_nba_markets` which internally
    calls Gamma API. Window is filtered inside discover_nba_markets
    already (it returns active-today markets) — we additionally filter
    to within `window_hours` of game start.
    """
    from arbo.strategies.strategy_d_discovery_nba import discover_nba_markets

    markets = await discover_nba_markets(gamma_client=None)
    today = datetime.now(timezone.utc).date()
    out = []
    for m in markets:
        # Optional filter: skip markets too far in future
        try:
            gd = datetime.strptime(m.game_date, "%Y-%m-%d").date()
            days_to = (gd - today).days
            if days_to > (window_hours / 24):
                continue
            if days_to < -2:  # already 2+ days past → resolved
                continue
        except (ValueError, AttributeError):
            pass  # Accept if can't parse
        out.append({
            "token_id_yes": m.token_id_yes,
            "token_id_no": m.token_id_no,
            "condition_id": m.condition_id,
            "neg_risk": bool(m.neg_risk),
            "game_date": m.game_date,
            "question": m.question,
        })
    return out


# ── Insert row ────────────────────────────────────────────────────────


# Dust filter: skip inserts where bid-ask spread exceeds this fraction of mid.
# Pre-game NBA orderbooks often show only market-maker dust orders at 0.01 and
# 0.99 (spread > 90%) with no real trading. Those rows are useless for ML
# training and waste DB storage. Real NBA liquidity typically appears within
# 6-12h of tipoff with spread < 20% (bps < 2000).
DUST_SPREAD_THRESHOLD = 0.50  # 50% = 5000 bps


async def _insert_snapshot(
    session,
    token_id: str,
    condition_id: str | None,
    snapshot,
    neg_risk: bool,
    game_date: str | None,
    question: str | None,
) -> bool:
    """Write one orderbook snapshot to DB. Returns True on success.

    Computes true top-of-book as max(bids) / min(asks) rather than relying on
    snapshot.best_bid/ask (which reflect the Polymarket CLOB response order
    — bids ascending from dust, asks descending from dust — so index 0 is the
    FAR side, not the best side).

    Skips dust-only snapshots (spread > DUST_SPREAD_THRESHOLD of mid) —
    these indicate a market with no real trading yet. Returns False so the
    caller counts this as 'failed' (effectively 'filtered').
    """
    from sqlalchemy import text

    bids_raw = [(float(p), float(s)) for (p, s) in (snapshot.bids or [])
                if p is not None and s is not None]
    asks_raw = [(float(p), float(s)) for (p, s) in (snapshot.asks or [])
                if p is not None and s is not None]

    # True top-of-book: highest bid price, lowest ask price.
    bid = ask = bid_size = ask_size = None
    if bids_raw:
        b = max(bids_raw, key=lambda x: x[0])
        bid, bid_size = b[0], b[1]
    if asks_raw:
        a = min(asks_raw, key=lambda x: x[0])
        ask, ask_size = a[0], a[1]

    # Standard orderbook convention for JSON storage: bids descending
    # (highest first), asks ascending (lowest first), top 5 levels each.
    bids_json = (
        [[p, s] for p, s in sorted(bids_raw, key=lambda x: -x[0])[:5]]
        if bids_raw else None
    )
    asks_json = (
        [[p, s] for p, s in sorted(asks_raw, key=lambda x: x[0])[:5]]
        if asks_raw else None
    )

    mid = (bid + ask) / 2 if (bid is not None and ask is not None) else None
    spread_bps = None
    if bid is not None and ask is not None and mid and mid > 1e-6:
        spread_bps = (ask - bid) / mid * 10000

    # Dust filter — skip insert if spread exceeds threshold
    if bid is not None and ask is not None:
        raw_spread = ask - bid
        if raw_spread > DUST_SPREAD_THRESHOLD:
            logger.debug(
                "dust_snapshot_skipped",
                extra={
                    "token_id": token_id[:20],
                    "bid": bid, "ask": ask, "spread": raw_spread,
                },
            )
            return False

    # Convert game_date string → date object for asyncpg binding
    game_date_obj: date | None = None
    if game_date:
        try:
            game_date_obj = datetime.strptime(game_date, "%Y-%m-%d").date()
        except ValueError:
            game_date_obj = None

    try:
        await session.execute(
            text("""
                INSERT INTO orderbook_snapshots_d
                (token_id, condition_id, sport, ts,
                 best_bid, best_ask, bid_size, ask_size,
                 bids, asks, mid, spread_bps,
                 neg_risk, game_date, question, source)
                VALUES
                (:token_id, :condition_id, 'nba', now(),
                 :bid, :ask, :bid_size, :ask_size,
                 CAST(:bids AS JSON), CAST(:asks AS JSON), :mid, :spread_bps,
                 :neg_risk, :game_date, :question, 'clob_rest')
            """),
            {
                "token_id": token_id,
                "condition_id": condition_id,
                "bid": bid,
                "ask": ask,
                "bid_size": bid_size,
                "ask_size": ask_size,
                "bids": json.dumps(bids_json) if bids_json else None,
                "asks": json.dumps(asks_json) if asks_json else None,
                "mid": mid,
                "spread_bps": spread_bps,
                "neg_risk": neg_risk,
                "game_date": game_date_obj,
                "question": question,
            },
        )
        await session.commit()
        return True
    except Exception as e:
        logger.warning(
            "orderbook_snapshot_insert_failed",
            extra={"err": str(e), "token_id": token_id[:24]},
        )
        await session.rollback()
        return False


# ── Main loop ────────────────────────────────────────────────────────


async def _poll_once(orderbook_provider, tokens: list[dict], dry_run: bool = False) -> dict:
    """One polling pass across all tokens. Returns counts."""
    counts = {"total": 0, "fetched": 0, "inserted": 0, "failed": 0}
    if dry_run:
        logger.info("dry_run_skip_db", extra={"n_tokens": sum(1 for t in tokens) * 2})

    # Single session for entire poll iteration — cheaper than per-token.
    factory = get_session_factory() if not dry_run else None
    session_cm = factory() if factory else None

    try:
        session = await session_cm.__aenter__() if session_cm else None
        for tok in tokens:
            for side in ("yes", "no"):
                token_id = tok[f"token_id_{side}"]
                if not token_id:
                    continue
                counts["total"] += 1
                try:
                    snap = await orderbook_provider.get_snapshot(
                        token_id, neg_risk=tok["neg_risk"],
                    )
                    if snap is None or snap.best_bid is None or snap.best_ask is None:
                        counts["failed"] += 1
                        continue
                    counts["fetched"] += 1
                    if dry_run:
                        logger.info(
                            "dry_run_snapshot",
                            extra={
                                "token": token_id[:20],
                                "side": side,
                                "bid": float(snap.best_bid),
                                "ask": float(snap.best_ask),
                                "neg_risk": tok["neg_risk"],
                            },
                        )
                    else:
                        ok = await _insert_snapshot(
                            session,
                            token_id=token_id,
                            condition_id=tok["condition_id"],
                            snapshot=snap,
                            neg_risk=tok["neg_risk"],
                            game_date=tok.get("game_date"),
                            question=tok.get("question"),
                        )
                        if ok:
                            counts["inserted"] += 1
                        else:
                            counts["failed"] += 1
                except Exception as e:
                    counts["failed"] += 1
                    logger.warning(
                        "snapshot_fetch_failed",
                        extra={"err": str(e), "token": token_id[:24]},
                    )
    finally:
        if session_cm:
            await session_cm.__aexit__(None, None, None)

    return counts


async def _main_loop(interval_s: int, window_h: float, dry_run: bool):
    """Top-level async loop. Discovers markets + polls orderbook forever."""
    from arbo.connectors.orderbook_provider import OrderbookProvider
    from arbo.connectors.polymarket_client import PolymarketClient

    logger.info(
        "capture_starting",
        extra={"interval_s": interval_s, "window_h": window_h, "dry_run": dry_run},
    )

    # Read-only CLOB client is required — OrderbookProvider w/o client
    # silently returns None for every get_snapshot call.
    poly_client = PolymarketClient()
    await poly_client.initialize()
    orderbook = OrderbookProvider(poly_client=poly_client, cache_ttl_s=30.0)
    logger.info("clob_client_initialized")

    # Graceful shutdown
    stop_event = asyncio.Event()

    def _handle_stop(*_):
        logger.info("shutdown_requested")
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            asyncio.get_running_loop().add_signal_handler(sig, _handle_stop)
        except NotImplementedError:
            pass  # Windows fallback

    iter_count = 0
    while not stop_event.is_set():
        iter_count += 1
        t0 = asyncio.get_running_loop().time()
        try:
            tokens = await _discover_nba_tokens(window_hours=window_h)
            logger.info(
                "discovery_complete",
                extra={"iter": iter_count, "n_markets": len(tokens)},
            )
            if tokens:
                counts = await _poll_once(orderbook, tokens, dry_run=dry_run)
                elapsed = asyncio.get_running_loop().time() - t0
                logger.info(
                    "poll_complete",
                    extra={
                        "iter": iter_count,
                        "elapsed_s": f"{elapsed:.1f}",
                        **counts,
                    },
                )
        except Exception as e:
            logger.error(
                "poll_iteration_failed",
                extra={"iter": iter_count, "err": str(e), "type": type(e).__name__},
            )

        # Sleep until next interval (or stop event)
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval_s)
        except asyncio.TimeoutError:
            pass

    # Cleanup
    try:
        await poly_client.close()
    except Exception:
        pass
    logger.info("capture_stopped", extra={"total_iterations": iter_count})


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capture Polymarket CLOB orderbook snapshots for NBA markets.",
    )
    parser.add_argument(
        "--interval", type=int, default=300,
        help="Seconds between polls (default: 300 = 5 min)",
    )
    parser.add_argument(
        "--window-hours", type=float, default=12.0,
        help="Capture window before game start (default: 12h). NBA orderbooks "
             "are mostly dust > 12h out; narrowing improves data density.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Poll orderbook but don't write to DB",
    )
    args = parser.parse_args()

    if args.interval < 30:
        print("ERROR: --interval < 30s risks rate-limit. Refusing.")
        sys.exit(1)

    try:
        asyncio.run(
            _main_loop(args.interval, args.window_hours, args.dry_run),
        )
    except KeyboardInterrupt:
        print("\nInterrupted.")


if __name__ == "__main__":
    main()
