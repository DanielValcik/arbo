"""Mid-trade orderbook sampler — Phase 3.1.

After a trade is placed, captures the token's orderbook midpoint at
fixed offsets (+30s, +60s) and writes them into paper_trades.trade_details.
Used by composite reward computation (PerformanceAnalyzer Phase 3.3) and
by future Optuna BO sweeps that score on early-trade dynamics rather
than terminal PnL.

The sampler is fire-and-forget — schedule_mid_capture(token_id, trade_id)
returns immediately and runs the captures as background asyncio tasks.

Spec: docs/PROJECT_PARALLEL_ROADMAP.md §Phase 3
"""
from __future__ import annotations

import asyncio
from typing import Any

import aiohttp

from arbo.utils.logger import get_logger

logger = get_logger("mid_sampler")

CLOB_URL = "https://clob.polymarket.com"
SAMPLE_OFFSETS_S = (30.0, 60.0)


async def _fetch_mid(session: aiohttp.ClientSession, token_id: str) -> float | None:
    """Return current orderbook midpoint for a token, or None on error."""
    try:
        async with session.get(
            f"{CLOB_URL}/book", params={"token_id": token_id},
        ) as resp:
            if resp.status != 200:
                return None
            book = await resp.json()
        asks = book.get("asks", []) or []
        bids = book.get("bids", []) or []
        if not asks or not bids:
            return None
        best_ask = min(float(a["price"]) for a in asks)
        best_bid = max(float(b["price"]) for b in bids)
        return round((best_ask + best_bid) / 2.0, 4)
    except Exception:
        return None


async def _persist_mid(trade_id: int, key: str, value: float) -> None:
    """Merge {key: value} into paper_trades.trade_details JSONB for trade_id."""
    try:
        from arbo.utils.db import get_session_factory
        import sqlalchemy as sa
        factory = get_session_factory()
        async with factory() as session:
            await session.execute(
                sa.text("""
                    UPDATE paper_trades
                    SET trade_details = COALESCE(trade_details, '{}'::jsonb)
                                        || jsonb_build_object(:k, :v)
                    WHERE id = :id
                """),
                {"k": key, "v": value, "id": trade_id},
            )
            await session.commit()
    except Exception as e:
        logger.debug(
            "mid_persist_error", trade_id=trade_id, key=key, error=str(e),
        )


async def _sample_loop(token_id: str, trade_id: int) -> None:
    """Background task: sleep + fetch + persist for each offset."""
    timeout = aiohttp.ClientTimeout(total=8)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for offset in SAMPLE_OFFSETS_S:
            try:
                await asyncio.sleep(offset if offset == SAMPLE_OFFSETS_S[0]
                                    else offset - SAMPLE_OFFSETS_S[0])
            except asyncio.CancelledError:
                return
            mid = await _fetch_mid(session, token_id)
            if mid is not None:
                key = f"mid_at_{int(offset)}s"
                await _persist_mid(trade_id, key, mid)


def schedule_mid_capture(token_id: str, trade_id: int | None) -> None:
    """Fire-and-forget — start mid sampling for a trade.

    Safe to call from within strategy poll loops: if trade_id is None or
    no event loop, this is a no-op.
    """
    if not token_id or trade_id is None:
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    loop.create_task(
        _sample_loop(token_id, int(trade_id)),
        name=f"mid_capture_{trade_id}",
    )


def composite_reward(
    *,
    direction: int,
    entry_price: float,
    mid_at_60s: float | None,
    pnl_per_share: float | None,
) -> float | None:
    """Compute mid-trade composite reward.

    composite = 0.4 * dir_60s + 0.6 * norm_pnl

    where:
      dir_60s = +1 if mid moved our direction by ≥1 cent at 60s, -1 if
                opposite by ≥1 cent, 0 otherwise.
      norm_pnl = pnl_per_share / max(entry_price * (1-entry_price), 0.01)
                 (Brier-like normalization keeps it roughly in [-1, 1]).

    Returns None if either input missing.
    """
    if mid_at_60s is None or pnl_per_share is None:
        return None
    if entry_price <= 0 or entry_price >= 1:
        return None

    delta = mid_at_60s - entry_price
    if direction == -1:
        delta = -delta
    if abs(delta) < 0.01:
        dir_60s = 0.0
    else:
        dir_60s = 1.0 if delta > 0 else -1.0

    denom = max(entry_price * (1.0 - entry_price), 0.01)
    norm = pnl_per_share / denom
    norm = max(-1.5, min(1.5, norm))
    return round(0.4 * dir_60s + 0.6 * norm, 4)
