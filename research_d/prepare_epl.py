"""Strategy D EPL — Backtest Harness.

EPL (English Premier League) moneyline + draw green booking.

Key differences from NBA/UFC:
  - 3-way outcomes: home_win / draw / away_win
  - Polymarket splits into binary markets per outcome
  - Question formats:
    * "Will X beat Y?" — X wins (home or away team A)
    * "Will X win on DATE?" — X wins (single-team question)
    * "Will X vs Y end in a draw?" — draw outcome
  - REAL game dates (unlike UFC's 2025-01-01 placeholder)
  - Pinnacle from football-data.co.uk (1,940 games, free historical)

Usage:
    PYTHONPATH=. python3 research_d/prepare_epl.py
    PYTHONPATH=. python3 research_d/prepare_epl.py --limit 5000
"""

from __future__ import annotations

import math
import re
import sqlite3
import sys
import time
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).parent / "data"
DEFAULT_DB_PATH = DATA_DIR / "sports_backtest.sqlite"
TIME_BUDGET = 7200


def _name_key(name: str) -> str:
    """Normalize team name (unicode, lowercase, alphanum)."""
    if not name:
        return ""
    normalized = unicodedata.normalize("NFKD", name).encode("ASCII", "ignore").decode()
    # Strip common suffixes: FC, F.C., United, City, etc. (keep only unique part)
    normalized = re.sub(r"\b(f\.?c\.?|fc)\b", "", normalized.lower())
    return re.sub(r"[^a-z]", "", normalized)


# Question patterns (in priority order)
_BEAT_RE = re.compile(r"[Ww]ill\s+(.+?)\s+beat\s+(.+?)[\?\s]*$")
_SINGLE_WIN_RE = re.compile(r"[Ww]ill\s+(.+?)\s+win\s+on\s+\d{4}-\d{2}-\d{2}[\?\s]*$")
_VS_DRAW_RE = re.compile(r"[Ww]ill\s+(.+?)\s+vs\.?\s+(.+?)\s+end\s+in\s+(?:a\s+)?draw[\?\s]*$")

_NON_GAME_KEYWORDS = [
    "mvp", "top scorer", "golden boot", "ballon d'or",
    "relegation", "champions league spot", "title race",
    "make the top 4", "finish", "total wins", "winner of",
]


def _is_game_market(question: str) -> bool:
    if not question:
        return False
    lower = question.lower()
    return not any(kw in lower for kw in _NON_GAME_KEYWORDS)


@dataclass
class EPLMarket:
    token_id: str
    game_id: str
    question: str
    won: int
    game_date: str
    team_a: str          # Normalized team key
    team_b: str          # Empty for single-team questions
    outcome_type: str    # "team_a_wins", "draw", "single_team"
    prices: list = None  # [(ts, price), ...]
    def __post_init__(self):
        if self.prices is None:
            self.prices = []


def _parse_epl_market(question: str) -> tuple[str, str, str] | None:
    """Returns (team_a_key, team_b_key, outcome_type) or None."""
    if not question or not _is_game_market(question):
        return None
    q = question.strip().rstrip("?").strip()

    # Draw markets: "Will X vs Y end in a draw?"
    m = _VS_DRAW_RE.search(q)
    if m:
        a = _name_key(m.group(1))
        b = _name_key(m.group(2))
        if a and b and a != b and len(a) >= 3 and len(b) >= 3:
            return a, b, "draw"
        return None

    # Beat markets: "Will X beat Y?"
    m = _BEAT_RE.search(q)
    if m:
        a = _name_key(m.group(1))
        b = _name_key(m.group(2))
        if a and b and a != b and len(a) >= 3 and len(b) >= 3:
            return a, b, "team_a_wins"
        return None

    # Single team: "Will X win on DATE?"
    m = _SINGLE_WIN_RE.search(q)
    if m:
        a = _name_key(m.group(1))
        if a and len(a) >= 3:
            return a, "", "single_team"
        return None

    return None


# ── Data loading ──────────────────────────────────────────────────────

def iter_epl_markets(conn, min_prices: int):
    t0 = time.time()
    rows = conn.execute("""
        SELECT m.token_id, m.game_id, m.question, m.won, g.game_date
        FROM markets m JOIN games g ON m.game_id = g.game_id
        WHERE m.won IS NOT NULL AND g.sport = 'epl'
    """).fetchall()
    print(f"  EPL resolved markets: {len(rows)}", flush=True)

    candidates = []
    for token_id, game_id, question, won, game_date in rows:
        parsed = _parse_epl_market(question)
        if not parsed:
            continue
        ta, tb, outcome = parsed
        candidates.append((token_id, game_id, question, won, game_date, ta, tb, outcome))
    print(f"  Parsed: {len(candidates)}", flush=True)

    def _gen():
        loaded = 0
        skipped = 0
        for tid, gid, q, won, gd, ta, tb, outcome in candidates:
            price_rows = conn.execute(
                "SELECT ts, price FROM prices WHERE token_id = ? ORDER BY ts",
                (tid,),
            ).fetchall()
            prices = [(ts, p) for ts, p in price_rows if p and 0 < p < 1]
            if len(prices) < min_prices:
                skipped += 1
                continue
            loaded += 1
            if loaded % 500 == 0:
                print(f"  ... {loaded} loaded ({time.time()-t0:.0f}s)", flush=True)
            yield EPLMarket(
                token_id=tid, game_id=gid, question=q, won=won,
                game_date=gd, team_a=ta, team_b=tb, outcome_type=outcome,
                prices=prices,
            )
        print(f"  Done: {loaded} markets, {skipped} no-prices, {time.time()-t0:.0f}s", flush=True)

    return len(candidates), _gen()


def load_pinnacle_epl(conn) -> dict:
    """Load EPL Pinnacle 3-way odds (home_win, away_win, draw).

    Keyed by (team_a_key, team_b_key, date) — date helps disambiguate
    multiple matches between same teams across season.

    Returns: {(key_a, key_b): [(date, home_prob, away_prob, draw_prob?), ...]}
    """
    import json
    rows = conn.execute("""
        SELECT p.game_id, p.home_prob_novig, p.away_prob_novig, p.draw_odds,
               g.game_date, g.extra_json
        FROM pinnacle_odds p JOIN games g ON p.game_id = g.game_id
        WHERE g.sport = 'epl' AND p.home_prob_novig IS NOT NULL
    """).fetchall()
    pin = defaultdict(list)
    for gid, hp, ap, draw_odds, gdate, extra in rows:
        if not (hp and ap and hp > 0 and ap > 0 and extra):
            continue
        try:
            ext = json.loads(extra)
        except Exception:
            continue
        home_full = ext.get("home_team_full", "") or ext.get("home_full", "")
        away_full = ext.get("away_team_full", "") or ext.get("away_full", "")
        if not home_full or not away_full:
            continue
        fk_home = _name_key(home_full)
        fk_away = _name_key(away_full)
        if not fk_home or not fk_away:
            continue
        # Approx draw probability from remaining mass (if 2-way), or from draw_odds
        # For football-data.co.uk, they provide home/draw/away odds. Let's check.
        sum_2way = hp + ap
        draw_prob = max(0.0, 1.0 - sum_2way)  # Default inferred from 2-way

        entry = {
            "date": gdate,
            "home_prob": float(hp),
            "away_prob": float(ap),
            "draw_prob": draw_prob,
        }
        pin[(fk_home, fk_away)].append(entry)
        # Also store reversed for lookups
        pin[(fk_away, fk_home)].append({
            "date": gdate,
            "home_prob": float(ap),
            "away_prob": float(hp),
            "draw_prob": draw_prob,
        })

    print(f"  EPL Pinnacle pairs: {len(pin)} ({len(pin)//2} unique fixtures)", flush=True)
    return dict(pin)


# ── Model ─────────────────────────────────────────────────────────────

def model_prob_epl(market: EPLMarket, pinnacle: dict) -> float | None:
    """Return probability that the YES side of market resolves true."""
    ta, tb = market.team_a, market.team_b
    outcome = market.outcome_type

    if outcome == "single_team":
        # "Will X win on DATE?" — need to find X as either home or away
        # Search pinnacle for any fixture with X on given date
        for (pk_a, pk_b), entries in pinnacle.items():
            if ta != pk_a:
                continue
            for e in entries:
                if e["date"] == market.game_date:
                    # X is "home" side in our stored orientation (pk_a)
                    # Return home_prob (which is ta winning when ta is first)
                    return e["home_prob"]
        # Fuzzy fallback: date-agnostic if team name substring matches
        for (pk_a, pk_b), entries in pinnacle.items():
            if ta in pk_a or pk_a in ta:
                if entries and len(ta) >= 4 and len(pk_a) >= 4:
                    # Prefer closest-date entry
                    closest = min(entries, key=lambda e: abs(_date_diff(e["date"], market.game_date)))
                    if _date_diff(closest["date"], market.game_date) <= 3:
                        return closest["home_prob"]
        return None

    # Two-team markets
    if not ta or not tb:
        return None
    # Try exact pair match first
    entries = pinnacle.get((ta, tb), [])
    if not entries:
        # Fuzzy substring match
        for (pk_a, pk_b), e_list in pinnacle.items():
            a_match = ta in pk_a or pk_a in ta
            b_match = tb in pk_b or pk_b in tb
            if a_match and b_match and len(ta) >= 4 and len(tb) >= 4:
                entries = e_list
                break
    if not entries:
        return None

    # Pick closest date
    closest = min(entries, key=lambda e: abs(_date_diff(e["date"], market.game_date)))
    if _date_diff(closest["date"], market.game_date) > 3:
        return None  # Too far — probably wrong match

    if outcome == "team_a_wins":
        return closest["home_prob"]
    elif outcome == "draw":
        return closest["draw_prob"]
    return None


def _date_diff(date1: str, date2: str) -> int:
    """Absolute day difference."""
    from datetime import datetime
    try:
        d1 = datetime.strptime(date1, "%Y-%m-%d")
        d2 = datetime.strptime(date2, "%Y-%m-%d")
        return abs((d1 - d2).days)
    except Exception:
        return 9999


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


# ── Simulation ────────────────────────────────────────────────────────

@dataclass
class TradeResult:
    game_date: str
    team_a: str
    team_b: str
    outcome_type: str
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


def simulate_epl_trade(market: EPLMarket, prob: float, capital: float, params: dict):
    prices = market.prices
    if len(prices) < 2:
        return None

    entry_price = prices[0][1]
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

    gb = stopped = timed = False
    exit_price = entry_price

    for idx, (ts, price) in enumerate(prices[1:], 1):
        if side == "yes":
            if price >= target:
                gb = True; exit_price = price; break
            if sl_on and price <= stop:
                stopped = True; exit_price = price; break
        else:
            if price <= target:
                gb = True; exit_price = price; break
            if sl_on and price >= stop:
                stopped = True; exit_price = price; break
        if idx >= max_hold_idx:
            timed = True; exit_price = price; break
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
        game_date=market.game_date, team_a=market.team_a, team_b=market.team_b,
        outcome_type=market.outcome_type,
        side=side, entry_price=entry_price, exit_price=exit_price,
        edge=edge, position_usd=cost, n_contracts=n, pnl=pnl,
        green_booked=gb, stopped_out=stopped, time_exited=timed,
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
    "min_price": 0.15,
    "max_price": 0.70,
    "min_prices": 10,
    "green_book_enabled": True,
    "green_book_delta": 0.15,
    "stop_loss_enabled": True,
    "stop_loss_delta": 0.20,
    "max_hold_fraction": 0.50,
    "kelly_fraction": 0.12,
    "kelly_raw_cap": 0.10,
    "max_position_pct": 0.03,
    "initial_capital": 1000.0,
    "both_sides": True,
}


def evaluate(params=None, db_path=None, limit=0):
    params = params or DEFAULT_PARAMS
    db_path = db_path or DEFAULT_DB_PATH
    t0 = time.time()
    print(f"Loading EPL data from {db_path}...", flush=True)
    conn = sqlite3.connect(f"file:{db_path}?mode=ro&immutable=1", uri=True, timeout=60)

    pinnacle = load_pinnacle_epl(conn)
    n_candidates, market_gen = iter_epl_markets(conn, params["min_prices"])

    if n_candidates == 0:
        conn.close()
        return {"score": 0, "n_trades": 0, "total_pnl": 0, "win_rate": 0,
                "green_book_rate": 0, "sharpe": 0, "max_drawdown": 0,
                "profit_factor": 0, "turnover": 0, "avg_edge": 0,
                "skipped": {"no_model": 0, "no_edge": 0}}

    print(f"\nRunning EPL backtest on {n_candidates} markets...", flush=True)
    capital = params["initial_capital"]
    trades = []
    skipped = {"no_model": 0, "no_edge": 0}
    count = 0

    # Attach prices to market
    for market in market_gen:
        count += 1
        if limit > 0 and count > limit:
            print(f"  LIMIT {limit} reached", flush=True)
            break
        if time.time() - t0 > TIME_BUDGET:
            print(f"  TIME BUDGET exceeded", flush=True)
            break

        prob = model_prob_epl(market, pinnacle)
        if prob is None:
            skipped["no_model"] += 1
            continue
        result = simulate_epl_trade(market, prob, capital, params)
        if result is None:
            skipped["no_edge"] += 1
            continue
        trades.append(result)
        capital += result.pnl
        market.prices = []

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
