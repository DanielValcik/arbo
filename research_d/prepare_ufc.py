"""Strategy D UFC — Backtest Harness.

UFC-specific backtest based on prepare.py (NBA). Key differences:
  - Fighter name parsing (last-name key, not team abbrev)
  - Pinnacle-only model (UFC has no Elo ratings)
  - UFC-specific keywords filter
  - Shorter GAME_DURATION (fights ~1.5h vs NBA 2.5h)

Usage:
    PYTHONPATH=. python3 research_d/prepare_ufc.py
    PYTHONPATH=. python3 research_d/prepare_ufc.py --limit 5000
"""

from __future__ import annotations

import math
import re
import sqlite3
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).parent / "data"
DEFAULT_DB_PATH = DATA_DIR / "sports_backtest.sqlite"
TIME_BUDGET = 7200

# ── UFC parsing ───────────────────────────────────────────────────────

_FIGHT_RE = [
    re.compile(r"^(?:UFC\s+(?:Fight\s+Night|\d+|\w+)\s*:\s*)?(.+?)\s+vs?\.?\s+(.+?)(?:\s*\([^)]*\))?[\?\s]*$", re.IGNORECASE),
]


def _name_key(name: str) -> str:
    """Normalize full fighter name for matching (lowercase, alphanum only)."""
    if not name:
        return ""
    return re.sub(r"[^a-z]", "", name.lower())

_NON_FIGHT_KEYWORDS = [
    "o/u", "rounds", "method", "round ", "won by", "distance",
    "submission", "ko or tko", "decision", "fight of the night",
    "retirement", "belt", "title",
]


def _fighter_key(name: str) -> str:
    """Normalize fighter name to uppercase last name."""
    if not name:
        return ""
    # Strip parentheses and weightclass
    cleaned = re.sub(r"\([^)]*\)", "", name)
    cleaned = re.sub(r"[^a-zA-Z\s-]", "", cleaned).strip()
    parts = cleaned.split()
    if not parts:
        return ""
    # Use last word (last name), uppercase
    last = parts[-1].upper()
    # Reject generic words
    if last in {"CARD", "PRELIMS", "MAIN", "EVENT", "UFC", "FIGHT", "NIGHT"}:
        # Try second-to-last
        if len(parts) >= 2:
            return parts[-2].upper()
        return ""
    return last


def _parse_ufc_fighters(question: str) -> tuple[str, str, str, str] | None:
    """Returns (last_name_a, last_name_b, fullname_key_a, fullname_key_b)."""
    if not question:
        return None
    lower = question.lower()
    if any(kw in lower for kw in _NON_FIGHT_KEYWORDS):
        return None
    q = question.strip().rstrip("?").strip()

    for pat in _FIGHT_RE:
        m = pat.match(q)
        if not m or not m.group(2):
            continue
        a_raw = m.group(1).strip()
        b_raw = m.group(2).strip()
        if len(a_raw) > 60 or len(b_raw) > 60:
            continue
        a = _fighter_key(a_raw)
        b = _fighter_key(b_raw)
        fk_a = _name_key(a_raw)
        fk_b = _name_key(b_raw)
        if a and b and a != b and len(a) >= 3 and len(b) >= 3:
            return a, b, fk_a, fk_b
    return None


# ── Data structures ───────────────────────────────────────────────────

@dataclass
class UFCMarket:
    token_id: str
    game_id: str
    question: str
    won: int
    game_date: str
    fighter_a: str        # Last name uppercase (e.g., "EMMETT")
    fighter_b: str
    fullname_key_a: str   # Normalized full name (e.g., "joshemmett")
    fullname_key_b: str
    prices: list[tuple[int, float]]


# ── Data loading ──────────────────────────────────────────────────────

def iter_ufc_markets(conn, min_prices: int):
    """Stream UFC markets."""
    t0 = time.time()
    rows = conn.execute("""
        SELECT m.token_id, m.game_id, m.question, m.won, g.game_date
        FROM markets m JOIN games g ON m.game_id = g.game_id
        WHERE m.won IS NOT NULL AND g.sport = 'ufc'
    """).fetchall()
    print(f"  UFC resolved markets: {len(rows)}", flush=True)

    candidates = []
    for token_id, game_id, question, won, game_date in rows:
        fighters = _parse_ufc_fighters(question)
        if not fighters:
            continue
        candidates.append((token_id, game_id, question, won, game_date, *fighters))
    print(f"  With parsed fighters: {len(candidates)}", flush=True)

    def _gen():
        loaded = 0
        skipped = 0
        for token_id, game_id, question, won, game_date, fa, fb, fk_a, fk_b in candidates:
            price_rows = conn.execute(
                "SELECT ts, price FROM prices WHERE token_id = ? ORDER BY ts",
                (token_id,),
            ).fetchall()
            prices = [(ts, p) for ts, p in price_rows if p and 0 < p < 1]
            if len(prices) < min_prices:
                skipped += 1
                continue
            loaded += 1
            if loaded % 500 == 0:
                print(f"  ... {loaded} loaded ({time.time()-t0:.0f}s)", flush=True)
            yield UFCMarket(
                token_id=token_id, game_id=game_id, question=question,
                won=won, game_date=game_date,
                fighter_a=fa, fighter_b=fb,
                fullname_key_a=fk_a, fullname_key_b=fk_b,
                prices=prices,
            )
        print(f"  Done: {loaded} markets, {skipped} skipped (no prices), {time.time()-t0:.0f}s", flush=True)

    return len(candidates), _gen()


def load_pinnacle_ufc(conn) -> dict:
    """Load UFC Pinnacle odds indexed by (date, fighter_a_full_key, fighter_b_full_key).

    Uses extra_json home_full/away_full (real fighter names) from Pinnacle games.
    """
    import json
    rows = conn.execute("""
        SELECT p.game_id, p.home_prob_novig, p.away_prob_novig, g.game_date, g.extra_json
        FROM pinnacle_odds p JOIN games g ON p.game_id = g.game_id
        WHERE g.sport = 'ufc' AND p.home_prob_novig IS NOT NULL
    """).fetchall()
    pin = {}
    for gid, hp, ap, game_date, extra in rows:
        if not (hp and ap and hp > 0 and ap > 0 and extra):
            continue
        try:
            ext = json.loads(extra)
        except Exception:
            continue
        home_full = ext.get("home_full", "")
        away_full = ext.get("away_full", "")
        if not home_full or not away_full:
            continue
        fk_home = _name_key(home_full)
        fk_away = _name_key(away_full)
        # Store both orderings: (date, first_fighter, second_fighter) → (prob_first, prob_second)
        pin[(game_date, fk_home, fk_away)] = (float(hp), float(ap))
        pin[(game_date, fk_away, fk_home)] = (float(ap), float(hp))
    print(f"  UFC Pinnacle lookup keys: {len(pin)}", flush=True)
    return pin


# ── Model (Pinnacle-only for UFC) ─────────────────────────────────────

def model_prob_ufc(game_date: str, fullname_key_a: str, fullname_key_b: str,
                   pinnacle: dict) -> float | None:
    """Pinnacle-only model. Match by date + full-name keys (fuzzy substring)."""
    if not game_date or not fullname_key_a or not fullname_key_b:
        return None
    # Try direct exact match
    p = pinnacle.get((game_date, fullname_key_a, fullname_key_b))
    if p:
        return p[0]
    # Try ± 1 day tolerance
    from datetime import datetime, timedelta
    try:
        dt = datetime.strptime(game_date, "%Y-%m-%d")
        for delta_d in (-1, 1):
            alt = (dt + timedelta(days=delta_d)).strftime("%Y-%m-%d")
            p = pinnacle.get((alt, fullname_key_a, fullname_key_b))
            if p:
                return p[0]
    except Exception:
        pass
    # Fuzzy: iterate Pinnacle entries looking for substring match on same date (±1)
    from datetime import datetime, timedelta
    check_dates = [game_date]
    try:
        dt = datetime.strptime(game_date, "%Y-%m-%d")
        check_dates += [(dt + timedelta(days=d)).strftime("%Y-%m-%d") for d in (-1, 1)]
    except Exception:
        pass
    for (pd, fa, fb), (pa, pb) in pinnacle.items():
        if pd not in check_dates:
            continue
        # Check if full-name keys substring-match (either direction)
        a_match = fullname_key_a in fa or fa in fullname_key_a
        b_match = fullname_key_b in fb or fb in fullname_key_b
        if a_match and b_match and fullname_key_a != fullname_key_b:
            return pa
    return None


# ── Kelly sizing ──────────────────────────────────────────────────────

def kelly_size(edge: float, price: float, capital: float, params: dict) -> float:
    if price <= 0 or price >= 1:
        return 0.0
    p = max(0.01, min(0.99, price + edge))
    q = 1 - p
    b = (1 / price) - 1
    if b <= 0:
        return 0.0
    kelly_raw = max(0, min((b * p - q) / b, params["kelly_raw_cap"]))
    size = capital * params["kelly_fraction"] * kelly_raw
    return min(size, capital * params["max_position_pct"])


# ── Trade simulation ──────────────────────────────────────────────────

@dataclass
class TradeResult:
    game_date: str
    fighter_a: str
    fighter_b: str
    side: str
    entry_price: float
    exit_price: float
    edge: float
    position_usd: float
    n_contracts: int
    pnl: float
    green_booked: bool
    stopped_out: bool
    time_exited: bool
    hold_fraction: float


def simulate_ufc_trade(market: UFCMarket, prob: float, capital: float, params: dict):
    prices = market.prices
    if len(prices) < 2:
        return None

    entry_price = prices[0][1]

    # Determine side (YES=fighter_a wins)
    raw_edge = prob - entry_price
    both = params.get("both_sides", False)

    if raw_edge >= params["min_edge"] and raw_edge <= params["max_edge"]:
        side = "yes"
        edge = raw_edge
        trade_price = entry_price
    elif both and -raw_edge >= params["min_edge"] and -raw_edge <= params["max_edge"]:
        side = "no"
        edge = -raw_edge
        trade_price = 1 - entry_price
    else:
        return None

    if trade_price < params["min_price"] or trade_price > params["max_price"]:
        return None

    size = kelly_size(edge, trade_price, capital, params)
    n = int(size / trade_price)
    if n < 1:
        return None
    cost = n * trade_price

    delta = params["green_book_delta"]
    sl = params.get("stop_loss_delta", 0.15)
    sl_on = params.get("stop_loss_enabled", True)
    mhf = params.get("max_hold_fraction", 1.0)

    if side == "yes":
        target = entry_price + delta
        stop = entry_price - sl if sl_on else 0
    else:
        target = entry_price - delta
        stop = entry_price + sl if sl_on else 2.0

    max_hold_idx = int(len(prices) * mhf) if mhf < 1.0 else len(prices)

    gb = False
    stopped = False
    timed = False
    exit_price = entry_price
    exit_idx = len(prices) - 1

    for idx, (ts, price) in enumerate(prices[1:], 1):
        if side == "yes":
            if price >= target:
                gb = True; exit_price = price; exit_idx = idx; break
            if sl_on and price <= stop:
                stopped = True; exit_price = price; exit_idx = idx; break
        else:
            if price <= target:
                gb = True; exit_price = price; exit_idx = idx; break
            if sl_on and price >= stop:
                stopped = True; exit_price = price; exit_idx = idx; break
        if idx >= max_hold_idx:
            timed = True; exit_price = price; exit_idx = idx; break
    else:
        exit_price = prices[-1][1]

    if gb or stopped or timed:
        pnl = n * (exit_price - entry_price) if side == "yes" else n * (entry_price - exit_price)
    else:
        if side == "yes":
            pnl = n * (1 - entry_price) if market.won == 1 else -cost
        else:
            pnl = n * entry_price if market.won == 0 else -cost

    return TradeResult(
        game_date=market.game_date,
        fighter_a=market.fighter_a, fighter_b=market.fighter_b,
        side=side, entry_price=entry_price, exit_price=exit_price,
        edge=edge, position_usd=cost, n_contracts=n, pnl=pnl,
        green_booked=gb, stopped_out=stopped, time_exited=timed,
        hold_fraction=exit_idx / max(len(prices) - 1, 1),
    )


# ── Metrics ───────────────────────────────────────────────────────────

def compute_metrics(trades, initial_capital):
    if not trades:
        return {"score": 0, "n_trades": 0, "total_pnl": 0, "win_rate": 0,
                "green_book_rate": 0, "sharpe": 0, "max_drawdown": 0,
                "profit_factor": 0, "turnover": 0, "avg_edge": 0}
    n = len(trades)
    pnls = [t.pnl for t in trades]
    total_pnl = sum(pnls)
    wins = sum(1 for t in trades if t.pnl > 0)
    gbs = sum(1 for t in trades if t.green_booked)

    daily = defaultdict(float)
    for t in trades:
        daily[t.game_date] += t.pnl
    if len(daily) > 1:
        vals = list(daily.values())
        mu = sum(vals) / len(vals)
        var = sum((v - mu) ** 2 for v in vals) / (len(vals) - 1)
        sharpe = (mu / max(var ** 0.5, 1e-9)) * (252 ** 0.5)
    else:
        sharpe = 0

    cum = peak = 0.0
    max_dd = 0.0
    for p in pnls:
        cum += p
        peak = max(peak, cum)
        max_dd = max(max_dd, (peak - cum) / max(initial_capital, 1))

    gp = sum(p for p in pnls if p > 0)
    gl = abs(sum(p for p in pnls if p < 0))
    pf = gp / max(gl, 1e-9)
    turnover = sum(t.position_usd for t in trades) / max(initial_capital, 1)

    pnl_f = total_pnl / max(initial_capital, 1) * 100
    sharpe_f = min(max(sharpe, 0) / 3, 2)
    trade_f = min(n / 100, 2)
    dd_f = max(0, 1 - max_dd * 2)
    turn_f = min(turnover / 10, 1.5)
    gb_f = 1 + (gbs / n) * 0.5
    score = pnl_f * (1 + sharpe_f) * trade_f * dd_f * turn_f * gb_f

    return {
        "score": score, "n_trades": n, "total_pnl": total_pnl,
        "win_rate": wins / n, "green_book_rate": gbs / n,
        "sharpe": sharpe, "max_drawdown": max_dd, "profit_factor": pf,
        "turnover": turnover, "avg_edge": sum(t.edge for t in trades) / n,
    }


# ── Evaluate ──────────────────────────────────────────────────────────

DEFAULT_PARAMS = {
    "min_edge": 0.05,
    "max_edge": 0.30,
    "min_price": 0.20,
    "max_price": 0.70,
    "min_prices": 10,
    "green_book_enabled": True,
    "green_book_delta": 0.20,
    "stop_loss_enabled": True,
    "stop_loss_delta": 0.20,
    "max_hold_fraction": 0.50,
    "kelly_fraction": 0.15,
    "kelly_raw_cap": 0.10,
    "max_position_pct": 0.03,
    "initial_capital": 1000.0,
    "both_sides": True,
}


def evaluate(params=None, db_path=None, limit=0):
    params = params or DEFAULT_PARAMS
    db_path = db_path or DEFAULT_DB_PATH

    t0 = time.time()
    print(f"Loading UFC data from {db_path}...", flush=True)
    conn = sqlite3.connect(f"file:{db_path}?mode=ro&immutable=1", uri=True, timeout=60)

    pinnacle = load_pinnacle_ufc(conn)
    n_candidates, market_gen = iter_ufc_markets(conn, params["min_prices"])

    if n_candidates == 0:
        conn.close()
        return {"score": 0, "n_trades": 0, "total_pnl": 0,
                "win_rate": 0, "green_book_rate": 0, "sharpe": 0,
                "max_drawdown": 0, "profit_factor": 0, "turnover": 0,
                "avg_edge": 0, "skipped": {"no_model": 0, "no_edge": 0}}

    print(f"\nRunning UFC backtest on {n_candidates} markets...", flush=True)
    capital = params["initial_capital"]
    trades = []
    skipped = {"no_model": 0, "no_edge": 0}
    count = 0

    for market in market_gen:
        count += 1
        if limit > 0 and count > limit:
            print(f"  LIMIT {limit} reached", flush=True)
            break
        if time.time() - t0 > TIME_BUDGET:
            print(f"  TIME BUDGET exceeded", flush=True)
            break

        prob = model_prob_ufc(market.game_date, market.fullname_key_a, market.fullname_key_b, pinnacle)
        if prob is None:
            skipped["no_model"] += 1
            continue
        result = simulate_ufc_trade(market, prob, capital, params)
        if result is None:
            skipped["no_edge"] += 1
            continue
        trades.append(result)
        capital += result.pnl
        market.prices = []  # free memory

    conn.close()
    metrics = compute_metrics(trades, params["initial_capital"])
    metrics["skipped"] = skipped
    return metrics


def print_results(r):
    print(f"\nscore={r['score']:.1f}")
    print(f"pnl=${r['total_pnl']:.2f}")
    print(f"trades={r['n_trades']}")
    print(f"win_rate={r['win_rate']:.1%}")
    print(f"gb_rate={r['green_book_rate']:.1%}")
    print(f"sharpe={r['sharpe']:.2f}")
    print(f"max_dd={r['max_drawdown']:.1%}")
    print(f"pf={r['profit_factor']:.2f}")
    print(f"turnover={r['turnover']:.1f}x")
    print(f"avg_edge={r['avg_edge']:.3f}")
    if "skipped" in r:
        print(f"skipped={r['skipped']}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH))
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    r = evaluate(DEFAULT_PARAMS, Path(args.db), limit=args.limit)
    print_results(r)


if __name__ == "__main__":
    main()
