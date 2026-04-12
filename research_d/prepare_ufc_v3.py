"""Strategy D UFC v3 — Expanded markets (moneyline + method + O/U rounds).

Research-based improvements:
  1. Method-of-victory model: P(KO) = moneyline_prob × weight_class_KO_rate
  2. O/U rounds: time-decay model based on historical fight duration distribution
  3. Distance markets: binary based on moneyline × base rate

UFC base rates from 2015-2023 data (n≈5000):
  Weight class  | KO/TKO | Sub  | Decision | Avg rounds
  Heavyweight   | 58%    | 12%  | 30%      | 2.2
  Light Heavy   | 48%    | 15%  | 37%      | 2.5
  Middleweight  | 38%    | 18%  | 44%      | 2.7
  Welterweight  | 34%    | 20%  | 46%      | 2.8
  Lightweight   | 32%    | 22%  | 46%      | 2.8
  Featherweight | 28%    | 20%  | 52%      | 3.0
  Bantamweight  | 25%    | 21%  | 54%      | 3.1
  Flyweight     | 20%    | 22%  | 58%      | 3.2
  Women's SW    | 30%    | 22%  | 48%      | 2.9
  Women's FLW   | 22%    | 18%  | 60%      | 3.2

Note: Fee-aware edge — Polymarket now charges 0.75% on sports at peak
(fee = p*(1-p)*fee_rate). We subtract fee_cost from edge.

Source: Research agent findings (Collier/Johnson 2021 Pinnacle efficiency paper,
Hitkul et al. 2019 UFC ML models).
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

DATA_DIR = Path(__file__).parent / "data"
DEFAULT_DB_PATH = DATA_DIR / "sports_backtest.sqlite"
TIME_BUDGET = 7200

# Weight class → (ko_rate, sub_rate, decision_rate, avg_rounds)
WEIGHT_CLASS_RATES = {
    "heavyweight":    (0.58, 0.12, 0.30, 2.2),
    "light heavy":    (0.48, 0.15, 0.37, 2.5),
    "middleweight":   (0.38, 0.18, 0.44, 2.7),
    "welterweight":   (0.34, 0.20, 0.46, 2.8),
    "lightweight":    (0.32, 0.22, 0.46, 2.8),
    "featherweight":  (0.28, 0.20, 0.52, 3.0),
    "bantamweight":   (0.25, 0.21, 0.54, 3.1),
    "flyweight":      (0.20, 0.22, 0.58, 3.2),
    "women's strawweight": (0.30, 0.22, 0.48, 2.9),
    "women's bantamweight": (0.25, 0.21, 0.54, 3.1),
    "women's flyweight":   (0.22, 0.18, 0.60, 3.2),
    "women's featherweight": (0.28, 0.20, 0.52, 3.0),
}
DEFAULT_RATES = (0.34, 0.20, 0.46, 2.8)  # Welterweight avg

# Polymarket taker fee on sports (as of March 2026)
FEE_RATE = 0.0075  # 0.75% at peak


def _parse_weight_class(question: str) -> tuple[float, float, float, float]:
    """Extract weight class rates from question text like '...(Welterweight, Main Card)'."""
    lower = question.lower()
    for wc, rates in WEIGHT_CLASS_RATES.items():
        if wc in lower:
            return rates
    return DEFAULT_RATES


def _name_key(name: str) -> str:
    if not name:
        return ""
    normalized = unicodedata.normalize("NFKD", name).encode("ASCII", "ignore").decode("ASCII")
    return re.sub(r"[^a-z]", "", normalized.lower())


# ── Parse question types ──────────────────────────────────────────────

_FIGHT_RE = re.compile(r"^(?:UFC(?:\s+(?:Fight\s+Night|\d+|\w+))?\s*:\s*)?(.+?)\s+vs?\.?\s+(.+?)(?:\s*\([^)]*\))?[\?\s]*$", re.IGNORECASE)
_METHOD_X_WINS_KO = re.compile(r"[Ww]ill\s+(.+?)\s+win\s+by\s+(?:KO|TKO)", re.IGNORECASE)
_METHOD_X_WINS_SUB = re.compile(r"[Ww]ill\s+(.+?)\s+win\s+by\s+(?:submission|sub)", re.IGNORECASE)
_METHOD_X_WINS_DEC = re.compile(r"[Ww]ill\s+(.+?)\s+win\s+by\s+decision", re.IGNORECASE)
_METHOD_FIGHT_KO = re.compile(r"[Ww]ill\s+the\s+fight\s+be\s+won\s+by\s+(?:KO|TKO)", re.IGNORECASE)
_METHOD_FIGHT_SUB = re.compile(r"[Ww]ill\s+the\s+fight\s+be\s+won\s+by\s+submission", re.IGNORECASE)
_DISTANCE = re.compile(r"[Ff]ight\s+to\s+[Gg]o\s+the\s+[Dd]istance|[Gg]oes\s+to\s+(?:the\s+)?decision", re.IGNORECASE)
_OU_ROUNDS = re.compile(r"O/U\s+(\d+\.?5?)\s+[Rr]ounds", re.IGNORECASE)


@dataclass
class UFCMarket:
    token_id: str
    game_id: str
    question: str
    won: int
    game_date: str
    market_type: str         # "moneyline", "method_fighter_ko", "method_fighter_sub",
                              # "method_fight_ko", "method_fight_sub", "distance", "ou_rounds"
    fighter_a: str           # For moneyline (last name key)
    fighter_b: str
    fullname_key_a: str
    fullname_key_b: str
    weight_rates: tuple      # (ko_rate, sub_rate, dec_rate, avg_rounds)
    round_threshold: float   # For O/U markets (e.g., 2.5)
    prices: list = None

    def __post_init__(self):
        if self.prices is None:
            self.prices = []


def _parse_market(question: str) -> dict | None:
    """Parse question into market info. Returns dict or None if unparseable."""
    if not question:
        return None
    q = question.strip().rstrip("?").strip()
    weight_rates = _parse_weight_class(q)

    # Method markets (fighter-specific)
    for rx, mtype in [
        (_METHOD_X_WINS_KO, "method_fighter_ko"),
        (_METHOD_X_WINS_SUB, "method_fighter_sub"),
        (_METHOD_X_WINS_DEC, "method_fighter_dec"),
    ]:
        m = rx.search(q)
        if m:
            name = m.group(1).strip()
            return {
                "type": mtype,
                "fighter_a": name,
                "fullname_key_a": _name_key(name),
                "weight_rates": weight_rates,
            }

    # Fight-generic method markets
    if _METHOD_FIGHT_KO.search(q):
        return {"type": "method_fight_ko", "weight_rates": weight_rates}
    if _METHOD_FIGHT_SUB.search(q):
        return {"type": "method_fight_sub", "weight_rates": weight_rates}

    # Distance
    if _DISTANCE.search(q):
        return {"type": "distance", "weight_rates": weight_rates}

    # Over/Under rounds
    m = _OU_ROUNDS.search(q)
    if m:
        return {
            "type": "ou_rounds",
            "round_threshold": float(m.group(1)),
            "weight_rates": weight_rates,
        }

    # Moneyline (fallback — skip if contains method keywords)
    lower = q.lower()
    method_kw = ["o/u", "rounds", "method", "won by", "ko", "tko",
                 "submission", "decision", "distance"]
    if any(kw in lower for kw in method_kw):
        return None

    m = _FIGHT_RE.match(q)
    if m and m.group(2):
        a_raw = m.group(1).strip()
        b_raw = m.group(2).strip()
        if len(a_raw) > 60 or len(b_raw) > 60:
            return None
        return {
            "type": "moneyline",
            "fighter_a": a_raw, "fighter_b": b_raw,
            "fullname_key_a": _name_key(a_raw),
            "fullname_key_b": _name_key(b_raw),
            "weight_rates": weight_rates,
        }
    return None


# ── Pinnacle loader ───────────────────────────────────────────────────

def load_pinnacle_ufc(conn) -> dict:
    """Load UFC Pinnacle keyed by fighter pair (date-agnostic)."""
    import json
    rows = conn.execute("""
        SELECT p.game_id, p.home_prob_novig, p.away_prob_novig, g.extra_json
        FROM pinnacle_odds p JOIN games g ON p.game_id=g.game_id
        WHERE g.sport='ufc' AND p.home_prob_novig IS NOT NULL
    """).fetchall()
    pin = {}
    pin_by_name = defaultdict(list)  # fk_name → list of (prob, opponent_key)
    for gid, hp, ap, extra in rows:
        if not (hp and ap and hp > 0 and ap > 0 and extra):
            continue
        try:
            ext = json.loads(extra)
        except Exception:
            continue
        hf, af = ext.get("home_full", ""), ext.get("away_full", "")
        if not hf or not af:
            continue
        fk_h, fk_a = _name_key(hf), _name_key(af)
        if not fk_h or not fk_a:
            continue
        pin[(fk_h, fk_a)] = (float(hp), float(ap))
        pin[(fk_a, fk_h)] = (float(ap), float(hp))
        pin_by_name[fk_h].append((float(hp), fk_a))
        pin_by_name[fk_a].append((float(ap), fk_h))
    print(f"  UFC Pinnacle pairs: {len(pin)}, fighters: {len(pin_by_name)}", flush=True)
    return {"pair": pin, "by_name": dict(pin_by_name)}


def _lookup_moneyline(fk_a: str, fk_b: str | None, pin: dict) -> float | None:
    """Return P(fighter_a wins). If fk_b None, find any opponent match."""
    if not fk_a or len(fk_a) < 4:
        return None
    # Pair match
    if fk_b:
        p = pin["pair"].get((fk_a, fk_b))
        if p:
            return p[0]
        # Fuzzy: substring in both names
        for (pk_a, pk_b), (pa, pb) in pin["pair"].items():
            a_match = fk_a in pk_a or pk_a in fk_a
            b_match = fk_b in pk_b or pk_b in fk_b
            if a_match and b_match:
                return pa
        return None
    # Single-fighter lookup (for method markets)
    by_name = pin["by_name"]
    # Exact
    if fk_a in by_name and by_name[fk_a]:
        return by_name[fk_a][0][0]  # first entry
    # Fuzzy
    for name, matches in by_name.items():
        if (fk_a in name or name in fk_a) and matches and len(fk_a) >= 5:
            return matches[0][0]
    return None


# ── Model probabilities ───────────────────────────────────────────────

def model_prob_ufc(market: UFCMarket, pin: dict) -> float | None:
    """Compute model probability for market's YES outcome."""
    mt = market.market_type
    ko_rate, sub_rate, dec_rate, avg_rounds = market.weight_rates

    if mt == "moneyline":
        return _lookup_moneyline(market.fullname_key_a, market.fullname_key_b, pin)

    if mt in ("method_fighter_ko", "method_fighter_sub", "method_fighter_dec"):
        # P(X wins by method) = P(X wins) × method_rate_given_win
        p_win = _lookup_moneyline(market.fullname_key_a, None, pin)
        if p_win is None:
            return None
        if mt == "method_fighter_ko":
            return p_win * ko_rate
        elif mt == "method_fighter_sub":
            return p_win * sub_rate
        elif mt == "method_fighter_dec":
            return p_win * dec_rate

    if mt == "method_fight_ko":
        # P(fight ends by KO/TKO) = ko_rate (either fighter)
        return ko_rate

    if mt == "method_fight_sub":
        return sub_rate

    if mt == "distance":
        # P(goes to distance) = dec_rate
        return dec_rate

    if mt == "ou_rounds":
        # P(over X.5 rounds) — use avg_rounds with Normal approx
        # σ ≈ 1.0 rounds for UFC fight duration distribution
        # P(over T) = 1 - Φ((T - avg_rounds)/σ)
        sigma = 1.0
        z = (market.round_threshold - avg_rounds) / sigma
        return 1.0 - 0.5 * (1.0 + math.erf(z / math.sqrt(2)))

    return None


# ── Fee-aware edge ────────────────────────────────────────────────────

def fee_cost(price: float) -> float:
    """Polymarket fee: p * (1-p) * fee_rate. Max at p=0.5 = 0.75%/4."""
    return price * (1 - price) * FEE_RATE


# ── Data loading ──────────────────────────────────────────────────────

def iter_ufc_markets(conn, min_prices: int):
    t0 = time.time()
    rows = conn.execute("""
        SELECT m.token_id, m.game_id, m.question, m.won, g.game_date
        FROM markets m JOIN games g ON m.game_id=g.game_id
        WHERE m.won IS NOT NULL AND g.sport='ufc'
    """).fetchall()
    print(f"  UFC resolved markets: {len(rows)}", flush=True)

    type_counts = defaultdict(int)
    candidates = []
    for tid, gid, q, won, gd in rows:
        parsed = _parse_market(q)
        if not parsed:
            continue
        type_counts[parsed["type"]] += 1
        candidates.append((tid, gid, q, won, gd, parsed))

    print(f"  Parsed by type:", flush=True)
    for t, n in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"    {t}: {n}", flush=True)
    print(f"  Total parseable: {len(candidates)}", flush=True)

    def _gen():
        loaded = skipped = 0
        for tid, gid, q, won, gd, parsed in candidates:
            price_rows = conn.execute(
                "SELECT ts, price FROM prices WHERE token_id=? ORDER BY ts", (tid,)
            ).fetchall()
            prices = [(ts, p) for ts, p in price_rows if p and 0 < p < 1]
            if len(prices) < min_prices:
                skipped += 1
                continue
            loaded += 1
            if loaded % 500 == 0:
                print(f"  ... {loaded} loaded ({time.time()-t0:.0f}s)", flush=True)

            yield UFCMarket(
                token_id=tid, game_id=gid, question=q, won=won, game_date=gd,
                market_type=parsed["type"],
                fighter_a=parsed.get("fighter_a", ""),
                fighter_b=parsed.get("fighter_b", ""),
                fullname_key_a=parsed.get("fullname_key_a", ""),
                fullname_key_b=parsed.get("fullname_key_b", ""),
                weight_rates=parsed["weight_rates"],
                round_threshold=parsed.get("round_threshold", 0.0),
                prices=prices,
            )
        print(f"  Done: {loaded} loaded, {skipped} no-prices, {time.time()-t0:.0f}s", flush=True)

    return len(candidates), _gen()


# ── Sizing + simulation (same as prepare_ufc.py) ──────────────────────

def kelly_size(edge: float, price: float, capital: float, params: dict) -> float:
    if price <= 0 or price >= 1: return 0.0
    p = max(0.01, min(0.99, price + edge))
    q = 1 - p
    b = (1 / price) - 1
    if b <= 0: return 0.0
    kelly_raw = max(0, min((b * p - q) / b, params["kelly_raw_cap"]))
    size = capital * params["kelly_fraction"] * kelly_raw
    return min(size, capital * params["max_position_pct"])


@dataclass
class TradeResult:
    game_date: str
    market_type: str
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


def simulate_trade(market: UFCMarket, prob: float, capital: float, params: dict):
    prices = market.prices
    if len(prices) < 2:
        return None
    entry_price = prices[0][1]
    raw_edge = prob - entry_price
    both = params.get("both_sides", False)

    # Fee-adjusted edge: subtract expected fee cost from edge
    fee_entry = fee_cost(entry_price) if params.get("fee_aware", True) else 0

    if raw_edge - fee_entry >= params["min_edge"] and raw_edge <= params["max_edge"]:
        side = "yes"; edge = raw_edge - fee_entry
        trade_price = entry_price
    elif both and -raw_edge - fee_entry >= params["min_edge"] and -raw_edge <= params["max_edge"]:
        side = "no"; edge = -raw_edge - fee_entry
        trade_price = 1 - entry_price
    else:
        return None

    if trade_price < params["min_price"] or trade_price > params["max_price"]:
        return None

    size = kelly_size(edge, trade_price, capital, params)
    n = int(size / trade_price)
    if n < 1: return None
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
            if price >= target: gb = True; exit_price = price; break
            if sl_on and price <= stop: stopped = True; exit_price = price; break
        else:
            if price <= target: gb = True; exit_price = price; break
            if sl_on and price >= stop: stopped = True; exit_price = price; break
        if idx >= max_hold_idx: timed = True; exit_price = price; break
    else:
        exit_price = prices[-1][1]

    # Subtract exit fee
    fee_exit = fee_cost(exit_price) if params.get("fee_aware", True) else 0
    total_fee_per_contract = fee_entry + fee_exit

    if gb or stopped or timed:
        if side == "yes":
            pnl = n * (exit_price - entry_price - total_fee_per_contract)
        else:
            pnl = n * (entry_price - exit_price - total_fee_per_contract)
    else:
        if side == "yes":
            pnl = n * (1 - entry_price - total_fee_per_contract) if market.won == 1 else -(cost + n * total_fee_per_contract)
        else:
            pnl = n * (entry_price - total_fee_per_contract) if market.won == 0 else -(cost + n * total_fee_per_contract)

    return TradeResult(
        game_date=market.game_date, market_type=market.market_type,
        side=side, entry_price=entry_price, exit_price=exit_price, edge=edge,
        position_usd=cost, n_contracts=n, pnl=pnl,
        green_booked=gb, stopped_out=stopped, time_exited=timed,
    )


# ── Metrics ───────────────────────────────────────────────────────────

def compute_metrics(trades, initial_capital):
    if not trades:
        return {"score": 0, "n_trades": 0, "total_pnl": 0, "win_rate": 0,
                "green_book_rate": 0, "sharpe": 0, "max_drawdown": 0,
                "profit_factor": 0, "turnover": 0, "avg_edge": 0, "by_type": {}}
    n = len(trades)
    pnls = [t.pnl for t in trades]
    total_pnl = sum(pnls)
    wins = sum(1 for t in trades if t.pnl > 0)
    gbs = sum(1 for t in trades if t.green_booked)

    daily = defaultdict(float)
    for t in trades: daily[t.game_date] += t.pnl
    if len(daily) > 1:
        vals = list(daily.values())
        mu = sum(vals) / len(vals)
        var = sum((v - mu) ** 2 for v in vals) / (len(vals) - 1)
        sharpe = (mu / max(var ** 0.5, 1e-9)) * (252 ** 0.5)
    else:
        sharpe = 0

    cum = peak = 0.0; max_dd = 0.0
    for p in pnls:
        cum += p; peak = max(peak, cum)
        max_dd = max(max_dd, (peak - cum) / max(initial_capital, 1))

    gp = sum(p for p in pnls if p > 0); gl = abs(sum(p for p in pnls if p < 0))
    pf = gp / max(gl, 1e-9)
    turnover = sum(t.position_usd for t in trades) / max(initial_capital, 1)

    pnl_f = total_pnl / max(initial_capital, 1) * 100
    sharpe_f = min(max(sharpe, 0) / 3, 2)
    trade_f = min(n / 100, 2)
    dd_f = max(0, 1 - max_dd * 2)
    turn_f = min(turnover / 10, 1.5)
    gb_f = 1 + (gbs / n) * 0.5
    score = pnl_f * (1 + sharpe_f) * trade_f * dd_f * turn_f * gb_f

    # By-type breakdown
    by_type = defaultdict(lambda: {"n": 0, "pnl": 0.0, "wins": 0})
    for t in trades:
        by_type[t.market_type]["n"] += 1
        by_type[t.market_type]["pnl"] += t.pnl
        if t.pnl > 0: by_type[t.market_type]["wins"] += 1

    return {
        "score": score, "n_trades": n, "total_pnl": total_pnl,
        "win_rate": wins / n, "green_book_rate": gbs / n,
        "sharpe": sharpe, "max_drawdown": max_dd, "profit_factor": pf,
        "turnover": turnover, "avg_edge": sum(t.edge for t in trades) / n,
        "by_type": dict(by_type),
    }


# ── Evaluate ──────────────────────────────────────────────────────────

DEFAULT_PARAMS = {
    "min_edge": 0.05,
    "max_edge": 0.30,
    "min_price": 0.15,
    "max_price": 0.75,
    "min_prices": 10,
    "green_book_enabled": True,
    "green_book_delta": 0.15,
    "stop_loss_enabled": True,
    "stop_loss_delta": 0.20,
    "max_hold_fraction": 0.70,
    "kelly_fraction": 0.10,
    "kelly_raw_cap": 0.08,
    "max_position_pct": 0.02,
    "initial_capital": 1000.0,
    "both_sides": True,
    "fee_aware": False,   # UFC moneyline on Polymarket = 0% fee. See CLAUDE.md
}


def evaluate(params=None, db_path=None, limit=0):
    params = params or DEFAULT_PARAMS
    db_path = db_path or DEFAULT_DB_PATH
    t0 = time.time()
    print(f"Loading UFC v3 data from {db_path}...", flush=True)
    conn = sqlite3.connect(f"file:{db_path}?mode=ro&immutable=1", uri=True, timeout=60)

    pin = load_pinnacle_ufc(conn)
    n_cand, gen = iter_ufc_markets(conn, params["min_prices"])
    if n_cand == 0:
        conn.close()
        return {"score": 0, "n_trades": 0, "total_pnl": 0}

    print(f"\nRunning UFC v3 backtest ({n_cand} markets)...", flush=True)
    capital = params["initial_capital"]
    trades = []
    skipped = defaultdict(int)
    count = 0

    for market in gen:
        count += 1
        if limit > 0 and count > limit: break
        if time.time() - t0 > TIME_BUDGET: break
        prob = model_prob_ufc(market, pin)
        if prob is None:
            skipped[f"no_model_{market.market_type}"] += 1
            continue
        result = simulate_trade(market, prob, capital, params)
        if result is None:
            skipped[f"no_edge_{market.market_type}"] += 1
            continue
        trades.append(result)
        capital += result.pnl
        market.prices = []

    conn.close()
    metrics = compute_metrics(trades, params["initial_capital"])
    metrics["skipped"] = dict(skipped)
    return metrics


def print_results(r):
    print(f"\nscore={r['score']:.1f} pnl=${r['total_pnl']:.2f} trades={r['n_trades']}")
    print(f"wr={r['win_rate']:.1%} gb={r['green_book_rate']:.1%} sharpe={r['sharpe']:.2f}")
    print(f"max_dd={r['max_drawdown']:.1%} pf={r['profit_factor']:.2f} turnover={r['turnover']:.1f}x")
    if r.get("by_type"):
        print("\nBy market type:")
        for t, d in sorted(r["by_type"].items(), key=lambda x: -x[1]["pnl"]):
            wr = d["wins"] / max(d["n"], 1)
            print(f"  {t}: {d['n']} trades, ${d['pnl']:.0f} pnl, wr={wr:.0%}")
    if "skipped" in r:
        total_skip = sum(r["skipped"].values())
        print(f"\nSkipped: {total_skip}")
        for k, v in sorted(r["skipped"].items(), key=lambda x: -x[1])[:10]:
            print(f"  {k}: {v}")


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
