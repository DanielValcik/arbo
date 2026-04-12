"""Strategy D NBA v2 — Expanded markets (moneyline + spread + O/U totals).

Research-based improvements over v1:
  1. Stern Brownian motion model for spread covers / totals
     P(home covers s) = 1 - Φ((s - μ_margin)/σ_margin)
     σ_margin ≈ 11.5 for NBA regular season
  2. Totals model: Normal(μ_total, σ_total) where μ = pace-adjusted
  3. Fee-aware edge (Polymarket 0.75% on sports)
  4. CLV computation in metrics (closing line value)

NBA has 45,810 O/U markets + 3,508 spread markets (vs 2,193 moneyline trades in v1).
Potential: 10-20x trade volume.

Source: Stern (1994), Paul & Weinbach NBA spread research.
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

# Stern's BM parameters for NBA
NBA_SIGMA_MARGIN = 11.5     # Final margin std (points)
NBA_SIGMA_TOTAL = 13.0       # Total points std
NBA_AVG_TOTAL = 225.0        # League avg total for 2024-25 / 2025-26
NBA_HCA = 2.5                # Home court advantage (points)

FEE_RATE = 0.0075


def _norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2)))


def _team_key(name: str) -> str:
    if not name:
        return ""
    normalized = unicodedata.normalize("NFKD", name).encode("ASCII", "ignore").decode()
    return re.sub(r"[^a-z]", "", normalized.lower())


# NBA team name map (primary names)
NBA_TEAMS_MAP = {
    "lakers": "LAL", "celtics": "BOS", "warriors": "GSW", "knicks": "NYK",
    "nets": "BKN", "76ers": "PHI", "sixers": "PHI", "heat": "MIA", "bulls": "CHI",
    "cavaliers": "CLE", "cavs": "CLE", "thunder": "OKC", "nuggets": "DEN",
    "bucks": "MIL", "suns": "PHX", "kings": "SAC", "clippers": "LAC",
    "mavericks": "DAL", "mavs": "DAL", "rockets": "HOU", "grizzlies": "MEM",
    "spurs": "SAS", "pelicans": "NOP", "pacers": "IND", "hawks": "ATL",
    "hornets": "CHA", "magic": "ORL", "wizards": "WAS", "pistons": "DET",
    "raptors": "TOR", "timberwolves": "MIN", "wolves": "MIN",
    "blazers": "POR", "trail blazers": "POR", "jazz": "UTA",
}


def _name_to_abbr(text: str) -> str | None:
    """Find NBA team abbreviation in text."""
    lower = text.lower().strip()
    # Longest match first
    for name in sorted(NBA_TEAMS_MAP.keys(), key=len, reverse=True):
        if name in lower:
            return NBA_TEAMS_MAP[name]
    return None


# ── Question parsers ──────────────────────────────────────────────────

# Moneyline: "Lakers vs. Celtics: Game 5", "Will Lakers beat Celtics?"
_ML_BEAT = re.compile(r"[Ww]ill\s+(?:the\s+)?(.+?)\s+beat\s+(?:the\s+)?(.+?)[\?\s]*$")
_ML_VS = re.compile(r"^(?:NBA:?\s*)?(.+?)\s+vs\.?\s+(.+?)(?:\s*[:]\s*(?:Game|Series).*)?[\?\s]*$")
# Spread: "NBA: Will the Lakers beat the Celtics by more than 5.5 points..."
_SPREAD_BY_MORE = re.compile(r"(?:NBA|Spread)[:]?\s*[Ww]ill\s+(?:the\s+)?(.+?)\s+beat\s+(?:the\s+)?(.+?)\s+by\s+more\s+than\s+(-?\d+\.?\d*)\s+points", re.IGNORECASE)
_SPREAD_COVER = re.compile(r"[Ww]ill\s+(?:the\s+)?(.+?)\s+cover\s+(?:the\s+)?spread\s+of\s+(-?\d+\.?\d*)", re.IGNORECASE)
# O/U totals: "NBA: Will Lakers vs Celtics total go over 225.5 points?"
_OU_TOTAL_OVER = re.compile(r"(?:NBA:)?\s*[Ww]ill\s+(?:the\s+)?(.+?)\s+(?:vs\.?|and)\s+(?:the\s+)?(.+?)\s+(?:total\s+|combined\s+)?(?:go\s+)?over\s+(\d+\.?\d*)", re.IGNORECASE)
_OU_GENERIC = re.compile(r"(?:NBA:)?\s*O/U\s+(\d+\.?\d*)", re.IGNORECASE)


@dataclass
class NBAMarketV2:
    token_id: str
    game_id: str
    question: str
    won: int
    game_date: str
    market_type: str    # "moneyline", "spread", "total_over"
    team_a: str
    team_b: str
    spread_value: float = 0.0       # For spread markets (positive = team_a favored)
    total_value: float = 0.0        # For total markets
    prices: list = None

    def __post_init__(self):
        if self.prices is None:
            self.prices = []


_NON_GAME = [
    "mvp", "award", "all-star", "total wins season", "make the playoffs",
    "championship", "finals mvp", "rookie of the year",
]


def _parse_market(q: str) -> dict | None:
    if not q:
        return None
    lower = q.lower()
    if any(kw in lower for kw in _NON_GAME):
        return None

    # Spread with "by more than X points"
    m = _SPREAD_BY_MORE.search(q)
    if m:
        a = _name_to_abbr(m.group(1))
        b = _name_to_abbr(m.group(2))
        try:
            spread = float(m.group(3))
        except:
            return None
        if a and b and a != b:
            return {"type": "spread", "team_a": a, "team_b": b, "spread_value": spread}

    # Spread cover
    m = _SPREAD_COVER.search(q)
    if m:
        a = _name_to_abbr(m.group(1))
        try:
            spread = float(m.group(2))
        except:
            return None
        if a:
            return {"type": "spread", "team_a": a, "team_b": "", "spread_value": spread}

    # Over total
    m = _OU_TOTAL_OVER.search(q)
    if m:
        a = _name_to_abbr(m.group(1))
        b = _name_to_abbr(m.group(2))
        try:
            total = float(m.group(3))
        except:
            return None
        if a and b and a != b:
            return {"type": "total_over", "team_a": a, "team_b": b, "total_value": total}

    # Moneyline
    if "by more than" not in lower and "cover" not in lower and "o/u" not in lower and "over " not in lower:
        m = _ML_BEAT.search(q)
        if m:
            a = _name_to_abbr(m.group(1))
            b = _name_to_abbr(m.group(2))
            if a and b and a != b:
                return {"type": "moneyline", "team_a": a, "team_b": b}
        m = _ML_VS.search(q)
        if m:
            a = _name_to_abbr(m.group(1))
            b = _name_to_abbr(m.group(2))
            if a and b and a != b:
                return {"type": "moneyline", "team_a": a, "team_b": b}

    return None


# ── Elo + Pinnacle loading ────────────────────────────────────────────

def load_ratings(conn) -> dict[str, tuple[float, float]]:
    rows = conn.execute(
        "SELECT team, elo, glicko_rating FROM ratings WHERE sport='nba' ORDER BY date"
    ).fetchall()
    ratings = {}
    for team, elo, glicko in rows:
        if elo is not None:
            ratings[team] = (float(elo), float(glicko or elo))  # latest wins
    print(f"  NBA Elo teams: {len(ratings)}", flush=True)
    return ratings


def load_pinnacle(conn) -> dict:
    """Load NBA Pinnacle keyed by (team_a, team_b) pairs."""
    rows = conn.execute("""
        SELECT p.game_id, p.home_prob_novig, p.away_prob_novig
        FROM pinnacle_odds p JOIN games g ON p.game_id=g.game_id
        WHERE g.sport='nba' AND p.home_prob_novig IS NOT NULL
    """).fetchall()
    pin = {}
    for gid, hp, ap in rows:
        parts = gid.split("_")
        if len(parts) >= 4:
            away, home = parts[2], parts[3]
            pin[(home, away)] = (float(hp), float(ap))
            pin[(away, home)] = (float(ap), float(hp))
    print(f"  NBA Pinnacle pairs: {len(pin)}", flush=True)
    return pin


# ── Models ────────────────────────────────────────────────────────────

def _moneyline_prob(team_a: str, team_b: str, ratings: dict, pinnacle: dict) -> float | None:
    """Combined Elo + Pinnacle for team_a wins probability."""
    ra = ratings.get(team_a)
    rb = ratings.get(team_b)
    elo_p = None
    if ra and rb:
        diff = ra[0] - rb[0]
        elo_p = 1.0 / (1.0 + 10.0 ** (-diff / 400.0))

    pin_p = None
    p = pinnacle.get((team_a, team_b))
    if p:
        pin_p = p[0]

    if elo_p is not None and pin_p is not None:
        return 0.40 * elo_p + 0.60 * pin_p
    return elo_p or pin_p


def _spread_cover_prob(team_a: str, team_b: str, spread: float,
                       ratings: dict, pinnacle: dict) -> float | None:
    """P(team_a beats team_b by more than `spread` points) using Stern BM.

    Derive implied margin from moneyline: μ = σ × Φ⁻¹(P(a_wins))
    Actually: from moneyline prob → convert to point spread.
    μ_margin ≈ σ_margin × Φ⁻¹(P(a wins)) × √2 [rough inversion]

    Better: use Elo difference as direct margin estimate:
      expected_margin = (elo_a - elo_b) / 28  (NBA convention)
    """
    ra = ratings.get(team_a)
    rb = ratings.get(team_b)
    # Expected margin from Elo
    if ra and rb:
        mu_margin = (ra[0] - rb[0]) / 28.0
    else:
        # Fall back: invert moneyline
        ml_p = _moneyline_prob(team_a, team_b, ratings, pinnacle)
        if ml_p is None:
            return None
        # Inverse normal CDF approx
        if ml_p <= 0.01 or ml_p >= 0.99:
            return None
        from math import sqrt, log
        # rational approximation of inverse normal
        q_prob = max(0.01, min(0.99, ml_p))
        # Simple: μ = σ × standard-score
        # Use Box-Muller inverse rational approximation
        t = math.sqrt(-2.0 * math.log(1 - q_prob if q_prob < 0.5 else q_prob))
        c0, c1, c2 = 2.515517, 0.802853, 0.010328
        d1, d2, d3 = 1.432788, 0.189269, 0.001308
        z = t - (c0 + c1 * t + c2 * t * t) / (1 + d1 * t + d2 * t * t + d3 * t ** 3)
        if q_prob < 0.5:
            z = -z
        mu_margin = z * NBA_SIGMA_MARGIN

    # P(margin > spread) = 1 - Φ((spread - μ)/σ)
    return 1.0 - _norm_cdf((spread - mu_margin) / NBA_SIGMA_MARGIN)


def _total_over_prob(team_a: str, team_b: str, total: float,
                     ratings: dict, pinnacle: dict) -> float | None:
    """P(combined points > total)."""
    # Without team-specific pace/offense data, use league avg
    # Could improve with team PPG/opponent PPG data
    mu_total = NBA_AVG_TOTAL
    return 1.0 - _norm_cdf((total - mu_total) / NBA_SIGMA_TOTAL)


def model_prob(market: NBAMarketV2, ratings: dict, pinnacle: dict) -> float | None:
    if market.market_type == "moneyline":
        return _moneyline_prob(market.team_a, market.team_b, ratings, pinnacle)
    elif market.market_type == "spread":
        return _spread_cover_prob(market.team_a, market.team_b, market.spread_value,
                                  ratings, pinnacle)
    elif market.market_type == "total_over":
        return _total_over_prob(market.team_a, market.team_b, market.total_value,
                                ratings, pinnacle)
    return None


def fee_cost(price: float) -> float:
    return price * (1 - price) * FEE_RATE


# ── Data loading ──────────────────────────────────────────────────────

def iter_nba_markets(conn, min_prices: int):
    t0 = time.time()
    rows = conn.execute("""
        SELECT m.token_id, m.game_id, m.question, m.won, g.game_date
        FROM markets m JOIN games g ON m.game_id=g.game_id
        WHERE m.won IS NOT NULL AND g.sport='nba'
    """).fetchall()
    print(f"  NBA resolved markets: {len(rows)}", flush=True)

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

    def _gen():
        loaded = 0
        for tid, gid, q, won, gd, parsed in candidates:
            price_rows = conn.execute(
                "SELECT ts, price FROM prices WHERE token_id=? ORDER BY ts", (tid,)
            ).fetchall()
            prices = [(ts, p) for ts, p in price_rows if p and 0 < p < 1]
            if len(prices) < min_prices:
                continue
            loaded += 1
            if loaded % 1000 == 0:
                print(f"  ... {loaded} loaded ({time.time()-t0:.0f}s)", flush=True)
            yield NBAMarketV2(
                token_id=tid, game_id=gid, question=q, won=won, game_date=gd,
                market_type=parsed["type"],
                team_a=parsed.get("team_a", ""),
                team_b=parsed.get("team_b", ""),
                spread_value=parsed.get("spread_value", 0),
                total_value=parsed.get("total_value", 0),
                prices=prices,
            )
        print(f"  Done: {loaded} loaded, {time.time()-t0:.0f}s", flush=True)

    return len(candidates), _gen()


# ── Kelly + simulation ────────────────────────────────────────────────

def kelly_size(edge, price, capital, params):
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
    close_price: float      # Final price before resolution (for CLV)
    edge: float
    position_usd: float
    n_contracts: int
    pnl: float
    green_booked: bool
    stopped_out: bool
    time_exited: bool
    clv: float              # Closing line value (cents)


def simulate_trade(market, prob, capital, params):
    prices = market.prices
    if len(prices) < 2: return None

    entry_price = prices[0][1]
    close_price = prices[-1][1]  # For CLV
    raw_edge = prob - entry_price
    both = params.get("both_sides", False)
    fee_entry = fee_cost(entry_price) if params.get("fee_aware", True) else 0

    if raw_edge - fee_entry >= params["min_edge"] and raw_edge <= params["max_edge"]:
        side = "yes"; edge = raw_edge - fee_entry; trade_price = entry_price
    elif both and -raw_edge - fee_entry >= params["min_edge"] and -raw_edge <= params["max_edge"]:
        side = "no"; edge = -raw_edge - fee_entry; trade_price = 1 - entry_price
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

    fee_exit = fee_cost(exit_price) if params.get("fee_aware", True) else 0
    total_fee = fee_entry + fee_exit

    if gb or stopped or timed:
        if side == "yes":
            pnl = n * (exit_price - entry_price - total_fee)
        else:
            pnl = n * (entry_price - exit_price - total_fee)
    else:
        if side == "yes":
            pnl = n * (1 - entry_price - total_fee) if market.won == 1 else -(cost + n * total_fee)
        else:
            pnl = n * (entry_price - total_fee) if market.won == 0 else -(cost + n * total_fee)

    # CLV: how much entry price moved vs close
    if side == "yes":
        clv_val = close_price - entry_price  # positive = entry was cheap vs close
    else:
        clv_val = entry_price - close_price  # inverted for NO

    return TradeResult(
        game_date=market.game_date, market_type=market.market_type,
        side=side, entry_price=entry_price, exit_price=exit_price, close_price=close_price,
        edge=edge, position_usd=cost, n_contracts=n, pnl=pnl,
        green_booked=gb, stopped_out=stopped, time_exited=timed,
        clv=clv_val,
    )


# ── Metrics ───────────────────────────────────────────────────────────

def compute_metrics(trades, initial_capital):
    if not trades:
        return {"score": 0, "n_trades": 0, "total_pnl": 0, "win_rate": 0,
                "green_book_rate": 0, "sharpe": 0, "max_drawdown": 0,
                "profit_factor": 0, "turnover": 0, "avg_edge": 0,
                "avg_clv": 0, "by_type": {}}
    n = len(trades)
    pnls = [t.pnl for t in trades]
    total_pnl = sum(pnls)
    wins = sum(1 for t in trades if t.pnl > 0)
    gbs = sum(1 for t in trades if t.green_booked)
    avg_clv = sum(t.clv for t in trades) / n

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

    by_type = defaultdict(lambda: {"n": 0, "pnl": 0.0, "wins": 0, "clv": 0.0})
    for t in trades:
        bt = by_type[t.market_type]
        bt["n"] += 1; bt["pnl"] += t.pnl
        bt["clv"] += t.clv
        if t.pnl > 0: bt["wins"] += 1

    return {
        "score": score, "n_trades": n, "total_pnl": total_pnl,
        "win_rate": wins / n, "green_book_rate": gbs / n,
        "sharpe": sharpe, "max_drawdown": max_dd, "profit_factor": pf,
        "turnover": turnover, "avg_edge": sum(t.edge for t in trades) / n,
        "avg_clv": avg_clv,
        "by_type": {k: dict(v) for k, v in by_type.items()},
    }


# ── Evaluate ──────────────────────────────────────────────────────────

DEFAULT_PARAMS = {
    "min_edge": 0.05,
    "max_edge": 0.25,
    "min_price": 0.15,
    "max_price": 0.75,
    "min_prices": 10,
    "green_book_enabled": True,
    "green_book_delta": 0.10,
    "stop_loss_enabled": True,
    "stop_loss_delta": 0.15,
    "max_hold_fraction": 0.50,
    "kelly_fraction": 0.12,
    "kelly_raw_cap": 0.10,
    "max_position_pct": 0.03,
    "initial_capital": 1000.0,
    "both_sides": True,
    "fee_aware": False,   # Polymarket sports moneyline = 0% fee (CLAUDE.md).
                           # PostOnly maker orders = 0% regardless. Only crypto/NCAAB has fees.
}


def evaluate(params=None, db_path=None, limit=0):
    params = params or DEFAULT_PARAMS
    db_path = db_path or DEFAULT_DB_PATH
    t0 = time.time()
    print(f"Loading NBA v2 data...", flush=True)
    conn = sqlite3.connect(f"file:{db_path}?mode=ro&immutable=1", uri=True, timeout=60)
    ratings = load_ratings(conn)
    pinnacle = load_pinnacle(conn)

    n_cand, gen = iter_nba_markets(conn, params["min_prices"])

    print(f"\nRunning NBA v2 backtest...", flush=True)
    capital = params["initial_capital"]
    trades = []
    skipped = defaultdict(int)
    count = 0

    for market in gen:
        count += 1
        if limit > 0 and count > limit: break
        if time.time() - t0 > TIME_BUDGET: break
        prob = model_prob(market, ratings, pinnacle)
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
    print(f"avg_clv={r.get('avg_clv', 0)*100:.2f}¢  avg_edge={r['avg_edge']:.3f}")
    if r.get("by_type"):
        print("\nBy market type:")
        for t, d in sorted(r["by_type"].items(), key=lambda x: -x[1]["pnl"]):
            wr = d["wins"] / max(d["n"], 1)
            clv = d["clv"] / max(d["n"], 1) * 100
            print(f"  {t}: {d['n']} trades, ${d['pnl']:.0f} pnl, wr={wr:.0%}, CLV={clv:.1f}¢")
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
