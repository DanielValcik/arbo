#!/usr/bin/env python3
"""Audit B3 live trades against actual Chainlink resolutions from Gamma API.

Identifies trades where live_exit_price in DB doesn't match the true Chainlink
outcome. Root cause: _resolve_b3_from_redeem matched on condition_id instead
of token_id, so trades on the LOSING side of a resolved market got falsely
marked as WIN ($1) when the WINNING side was redeemed.

This script is READ-ONLY by default. Pass --apply to write corrections to DB.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import ssl
import sys
import urllib.request
from pathlib import Path

import asyncpg

# Gamma API — reliable, public, no auth
GAMMA_URL = "https://gamma-api.polymarket.com"


def _ssl_ctx():
    ctx = ssl.create_default_context()
    try:
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        pass
    return ctx


async def fetch_gamma_market(condition_id: str) -> dict | None:
    """Fetch market resolution from Gamma API by condition_id."""
    url = f"{GAMMA_URL}/markets?condition_ids={condition_id}&closed=true"
    req = urllib.request.Request(url, headers={"User-Agent": "Arbo-Audit/1.0"})
    ctx = _ssl_ctx()

    def _fetch():
        return json.loads(urllib.request.urlopen(req, timeout=10, context=ctx).read())

    try:
        markets = await asyncio.get_event_loop().run_in_executor(None, _fetch)
    except Exception as e:
        return {"_error": str(e)}
    if not markets:
        return None
    m = markets[0]
    outcomes_raw = m.get("outcomes", "[]")
    prices_raw = m.get("outcomePrices", "[]")
    clob_raw = m.get("clobTokenIds", "[]")
    try:
        outcomes = json.loads(outcomes_raw) if isinstance(outcomes_raw, str) else outcomes_raw
        prices = json.loads(prices_raw) if isinstance(prices_raw, str) else prices_raw
        clob_token_ids = json.loads(clob_raw) if isinstance(clob_raw, str) else clob_raw
    except Exception:
        return None
    return {
        "condition_id": m.get("conditionId"),
        "closed": m.get("closed"),
        "outcomes": outcomes,
        "outcome_prices": prices,
        "clob_token_ids": clob_token_ids,
    }


def winning_token_id(gamma: dict) -> str | None:
    """Return clob_token_id of the winning outcome, or None if not resolved."""
    prices = gamma.get("outcome_prices") or []
    tokens = gamma.get("clob_token_ids") or []
    if len(prices) != len(tokens):
        return None
    for p, t in zip(prices, tokens):
        if str(p) in ("1", "1.0"):
            return str(t)
    return None


async def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--apply", action="store_true", help="Write corrections to DB")
    p.add_argument("--strategy", default="B3", help="Strategy filter (B3 or B3_15M)")
    p.add_argument("--limit", type=int, default=None, help="Max trades to audit")
    p.add_argument("--dsn", default=os.environ.get("DATABASE_URL", "postgresql://arbo@/arbo"))
    args = p.parse_args()

    conn = await asyncpg.connect(args.dsn)

    where_clause = """
        strategy = $1
        AND status IN ('won','lost','sold')
        AND (trade_details->>'live_fill_status') IN ('filled','partial')
        AND (trade_details->>'live_entry_shares')::float > 0
        AND trade_details->>'live_exit_status' IN ('resolution','filled','partial')
    """
    rows = await conn.fetch(
        f"""
        SELECT id, market_condition_id, token_id,
               trade_details->>'direction' AS direction,
               (trade_details->>'live_entry_price')::float AS entry,
               (trade_details->>'live_entry_shares')::float AS shares,
               (trade_details->>'live_exit_price')::float AS exit_price,
               trade_details AS details
        FROM paper_trades
        WHERE {where_clause}
        ORDER BY placed_at
        {'LIMIT ' + str(args.limit) if args.limit else ''}
        """,
        args.strategy,
    )
    print(f"Auditing {len(rows)} B3 resolution trades...")

    # Group by condition_id to fetch Gamma once per market
    cid_to_trades: dict[str, list] = {}
    for r in rows:
        cid_to_trades.setdefault(r["market_condition_id"], []).append(r)

    print(f"Unique markets: {len(cid_to_trades)}")

    # Sequential fetch (low rate) to avoid hammering Gamma
    cid_to_gamma: dict[str, dict] = {}
    for i, cid in enumerate(cid_to_trades.keys(), 1):
        g = await fetch_gamma_market(cid)
        if g and not g.get("_error"):
            cid_to_gamma[cid] = g
        if i % 20 == 0:
            print(f"  fetched {i}/{len(cid_to_trades)}")
        await asyncio.sleep(0.05)  # 20 req/s, well under 50/s limit

    # Audit each trade
    correct = 0
    needs_fix = 0
    unresolved = 0
    old_pnl_sum = 0.0
    new_pnl_sum = 0.0
    fixes: list[tuple[int, float, float, float, float]] = []
    for r in rows:
        cid = r["market_condition_id"]
        g = cid_to_gamma.get(cid)
        if not g or not g.get("closed"):
            unresolved += 1
            continue
        winner = winning_token_id(g)
        if winner is None:
            unresolved += 1
            continue
        real_exit = 1.0 if str(r["token_id"]) == winner else 0.0
        db_exit = r["exit_price"]
        entry = r["entry"]
        shares = r["shares"]
        old_pnl = (db_exit - entry) * shares
        new_pnl = (real_exit - entry) * shares
        old_pnl_sum += old_pnl
        new_pnl_sum += new_pnl
        if abs(real_exit - db_exit) > 0.01:
            needs_fix += 1
            fixes.append((r["id"], entry, shares, db_exit, real_exit))
        else:
            correct += 1

    print(f"\n=== AUDIT SUMMARY ===")
    print(f"Total trades audited: {len(rows)}")
    print(f"  Correct in DB:     {correct}")
    print(f"  NEED FIX:          {needs_fix}")
    print(f"  Unresolved/error:  {unresolved}")
    print(f"\nOLD PnL (current DB): ${old_pnl_sum:+.2f}")
    print(f"NEW PnL (corrected):  ${new_pnl_sum:+.2f}")
    print(f"DELTA:                ${new_pnl_sum - old_pnl_sum:+.2f}")

    if fixes:
        print(f"\n=== Top 10 largest corrections ===")
        fixes.sort(key=lambda f: abs(f[3] - f[4]) * f[2], reverse=True)
        for fid, entry, shares, old_x, new_x in fixes[:10]:
            old_p = (old_x - entry) * shares
            new_p = (new_x - entry) * shares
            print(f"  id={fid}  entry={entry:.3f} × {shares:.0f}  "
                  f"exit: {old_x:.1f}→{new_x:.1f}  PnL: ${old_p:+.2f}→${new_p:+.2f}")

    if args.apply and fixes:
        print(f"\nApplying {len(fixes)} corrections to DB...")
        for fid, _, _, _, new_x in fixes:
            await conn.execute(
                """
                UPDATE paper_trades
                SET trade_details = jsonb_set(
                    jsonb_set(trade_details,
                             '{live_exit_price}', to_jsonb($1::float)),
                             '{resolution_source}', to_jsonb('gamma_audit'::text))
                WHERE id = $2
                """,
                new_x, fid,
            )
        print("Done.")
    elif fixes:
        print(f"\n(read-only mode — pass --apply to write {len(fixes)} corrections)")

    await conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
