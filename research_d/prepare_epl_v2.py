"""Strategy D EPL v2 — Dixon-Coles probability model.

Research improvements over v1:
  1. Dixon-Coles (1997) correction for low-score correlations (0-0, 1-1)
     τ(x,y; λ, μ, ρ) function that better captures draws
  2. Derive draw probability from DC model, compare to Pinnacle
  3. Keep moneyline markets (v1 approach works)
  4. NO FEES (EPL on Polymarket = 0%, PostOnly = 0%)
  5. CLV tracking

Reference: Dixon & Coles (1997) JRSS-C.
  P(X=x, Y=y) = τ(x,y) × Pois(x; λ) × Pois(y; μ)
  τ(0,0) = 1 - λμρ
  τ(1,0) = 1 + μρ
  τ(0,1) = 1 + λρ
  τ(1,1) = 1 - ρ
  τ(x,y) = 1 otherwise

Typical EPL fitted: ρ ∈ [-0.15, -0.05], γ (home adv) ≈ 1.3.

Approach: For each fixture, estimate λ, μ from Pinnacle no-vig 2-way
(back out via optimization). Then simulate 5x5 score grid, derive true
P(home_win), P(away_win), P(draw) via DC τ correction.

Trade when our P differs from market price by min_edge.
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

# Typical EPL fitted parameters
DC_RHO = -0.10     # Dixon-Coles low-score correlation
EPL_HOME_ADV = 1.3 # Home advantage multiplier


def _name_key(name: str) -> str:
    if not name:
        return ""
    normalized = unicodedata.normalize("NFKD", name).encode("ASCII", "ignore").decode()
    normalized = re.sub(r"\b(f\.?c\.?|fc)\b", "", normalized.lower())
    return re.sub(r"[^a-z]", "", normalized)


# ── Dixon-Coles core ─────────────────────────────────────────────────

def _dc_tau(x: int, y: int, lam: float, mu: float, rho: float) -> float:
    """Dixon-Coles correction factor for low-score correlation."""
    if x == 0 and y == 0:
        return 1 - lam * mu * rho
    if x == 0 and y == 1:
        return 1 + lam * rho
    if x == 1 and y == 0:
        return 1 + mu * rho
    if x == 1 and y == 1:
        return 1 - rho
    return 1.0


def _poisson_pmf(k: int, lam: float) -> float:
    """Poisson PMF."""
    if lam <= 0 or k < 0:
        return 0.0
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def dc_outcome_probs(lam: float, mu: float, rho: float = DC_RHO,
                     max_goals: int = 8) -> tuple[float, float, float]:
    """Compute P(home_win), P(draw), P(away_win) from Dixon-Coles model."""
    p_home = p_draw = p_away = 0.0
    for x in range(max_goals + 1):
        p_x = _poisson_pmf(x, lam)
        for y in range(max_goals + 1):
            p_y = _poisson_pmf(y, mu)
            tau = _dc_tau(x, y, lam, mu, rho)
            p = tau * p_x * p_y
            if x > y:
                p_home += p
            elif x < y:
                p_away += p
            else:
                p_draw += p
    total = p_home + p_draw + p_away
    if total > 0:
        p_home /= total
        p_draw /= total
        p_away /= total
    return p_home, p_draw, p_away


def invert_pinnacle_to_lambda_mu(
    p_home_novig: float, p_away_novig: float,
    rho: float = DC_RHO,
) -> tuple[float, float]:
    """Back out λ (home expected goals) and μ (away expected goals) from Pinnacle 2-way.

    Uses iterative search: find (lam, mu) such that DC model's implied
    home_prob / away_prob matches Pinnacle. Assumes total goals ~2.6 (EPL avg).

    This is an approximation — real Dixon-Coles fitting uses MLE on historical data.
    """
    # Start with EPL averages: home scores 1.5, away scores 1.1 on average
    total_goals = 2.6  # EPL typical match total
    # Use ratio from Pinnacle to split goals
    # If p_home > p_away, home attacks more
    ratio = p_home_novig / max(p_away_novig, 0.01)

    # Initial guess based on ratio
    lam = total_goals * 0.55 * math.sqrt(ratio)
    mu = total_goals * 0.45 / math.sqrt(ratio)

    # Simple gradient descent: adjust lam/mu to match Pinnacle moneyline
    for _ in range(20):
        ph, pd, pa = dc_outcome_probs(lam, mu, rho, max_goals=6)
        err_home = ph - p_home_novig * (ph + pa)  # normalize 2-way
        err_away = pa - p_away_novig * (ph + pa)

        # Adjust: more lam = more home wins
        lam *= 1 - 0.3 * err_home
        mu *= 1 - 0.3 * err_away
        lam = max(0.1, min(4.0, lam))
        mu = max(0.1, min(4.0, mu))

        if abs(err_home) < 0.005:
            break

    return lam, mu


# ── Data structures ───────────────────────────────────────────────────

_BEAT_RE = re.compile(r"[Ww]ill\s+(.+?)\s+beat\s+(.+?)[\?\s]*$")
_SINGLE_WIN_RE = re.compile(r"[Ww]ill\s+(.+?)\s+win\s+on\s+\d{4}-\d{2}-\d{2}[\?\s]*$")
_VS_DRAW_RE = re.compile(r"[Ww]ill\s+(.+?)\s+vs\.?\s+(.+?)\s+end\s+in\s+(?:a\s+)?draw[\?\s]*$")

_NON_GAME_KEYWORDS = [
    "mvp", "top scorer", "golden boot", "ballon d'or",
    "relegation", "title race", "make the top", "finish",
    "total wins", "winner of", "premier league winner",
]


@dataclass
class EPLMarketV2:
    token_id: str
    game_id: str
    question: str
    won: int
    game_date: str
    team_a: str
    team_b: str
    outcome_type: str  # "team_a_wins", "draw", "single_team"
    prices: list = None

    def __post_init__(self):
        if self.prices is None:
            self.prices = []


def _parse_market(question: str) -> tuple[str, str, str] | None:
    if not question:
        return None
    lower = question.lower()
    if any(kw in lower for kw in _NON_GAME_KEYWORDS):
        return None
    q = question.strip().rstrip("?").strip()

    m = _VS_DRAW_RE.search(q)
    if m:
        a = _name_key(m.group(1))
        b = _name_key(m.group(2))
        if a and b and a != b and len(a) >= 3 and len(b) >= 3:
            return a, b, "draw"
        return None

    m = _BEAT_RE.search(q)
    if m:
        a = _name_key(m.group(1))
        b = _name_key(m.group(2))
        if a and b and a != b and len(a) >= 3 and len(b) >= 3:
            return a, b, "team_a_wins"
        return None

    m = _SINGLE_WIN_RE.search(q)
    if m:
        a = _name_key(m.group(1))
        if a and len(a) >= 3:
            return a, "", "single_team"
        return None

    return None


def iter_epl_markets(conn, min_prices: int):
    t0 = time.time()
    rows = conn.execute("""
        SELECT m.token_id, m.game_id, m.question, m.won, g.game_date
        FROM markets m JOIN games g ON m.game_id=g.game_id
        WHERE m.won IS NOT NULL AND g.sport='epl'
    """).fetchall()
    print(f"  EPL markets: {len(rows)}", flush=True)

    candidates = []
    for tid, gid, q, won, gd in rows:
        parsed = _parse_market(q)
        if parsed:
            candidates.append((tid, gid, q, won, gd, *parsed))
    print(f"  Parsed: {len(candidates)}", flush=True)

    def _gen():
        loaded = 0
        for tid, gid, q, won, gd, ta, tb, ot in candidates:
            price_rows = conn.execute(
                "SELECT ts, price FROM prices WHERE token_id=? ORDER BY ts", (tid,)
            ).fetchall()
            prices = [(ts, p) for ts, p in price_rows if p and 0 < p < 1]
            if len(prices) < min_prices:
                continue
            loaded += 1
            if loaded % 500 == 0:
                print(f"  ... {loaded} loaded ({time.time()-t0:.0f}s)", flush=True)
            yield EPLMarketV2(
                token_id=tid, game_id=gid, question=q, won=won,
                game_date=gd, team_a=ta, team_b=tb, outcome_type=ot,
                prices=prices,
            )
        print(f"  Done: {loaded} loaded", flush=True)

    return len(candidates), _gen()


def load_pinnacle_epl(conn):
    """Load EPL Pinnacle indexed by team pair with date."""
    import json
    rows = conn.execute("""
        SELECT p.home_prob_novig, p.away_prob_novig, g.game_date, g.extra_json
        FROM pinnacle_odds p JOIN games g ON p.game_id=g.game_id
        WHERE g.sport='epl' AND p.home_prob_novig IS NOT NULL
    """).fetchall()
    pin = defaultdict(list)
    for hp, ap, gdate, extra in rows:
        if not (hp and ap and hp > 0 and ap > 0 and extra):
            continue
        try:
            ext = json.loads(extra)
        except Exception:
            continue
        hf = ext.get("home_team_full", "") or ext.get("home_full", "")
        af = ext.get("away_team_full", "") or ext.get("away_full", "")
        if not hf or not af:
            continue
        fk_h = _name_key(hf)
        fk_a = _name_key(af)
        if not fk_h or not fk_a:
            continue
        entry = {"date": gdate, "home_prob": float(hp), "away_prob": float(ap)}
        pin[(fk_h, fk_a)].append(entry)
        pin[(fk_a, fk_h)].append({"date": gdate, "home_prob": float(ap), "away_prob": float(hp)})
    print(f"  Pinnacle pairs: {len(pin)} ({len(pin)//2} fixtures)", flush=True)
    return dict(pin)


def _date_diff(d1: str, d2: str) -> int:
    from datetime import datetime
    try:
        a = datetime.strptime(d1, "%Y-%m-%d")
        b = datetime.strptime(d2, "%Y-%m-%d")
        return abs((a - b).days)
    except Exception:
        return 9999


def _find_pinnacle(team_a: str, team_b: str, date: str, pin: dict) -> tuple[float, float] | None:
    """Return (p_home, p_away) from Pinnacle for team_a vs team_b."""
    if not team_a:
        return None
    # Single-team lookup: find any fixture with team_a
    if not team_b:
        for (pk_a, pk_b), entries in pin.items():
            if team_a == pk_a:
                closest = min(entries, key=lambda e: _date_diff(e["date"], date))
                if _date_diff(closest["date"], date) <= 3:
                    return closest["home_prob"], closest["away_prob"]
        # Fuzzy
        for (pk_a, _), entries in pin.items():
            if (team_a in pk_a or pk_a in team_a) and len(team_a) >= 4:
                closest = min(entries, key=lambda e: _date_diff(e["date"], date))
                if _date_diff(closest["date"], date) <= 3:
                    return closest["home_prob"], closest["away_prob"]
        return None

    # Pair lookup
    entries = pin.get((team_a, team_b), [])
    if not entries:
        for (pk_a, pk_b), e_list in pin.items():
            if (team_a in pk_a or pk_a in team_a) and (team_b in pk_b or pk_b in team_b) \
                    and len(team_a) >= 4 and len(team_b) >= 4:
                entries = e_list
                break
    if not entries:
        return None
    closest = min(entries, key=lambda e: _date_diff(e["date"], date))
    if _date_diff(closest["date"], date) > 3:
        return None
    return closest["home_prob"], closest["away_prob"]


def model_prob_dc(market: EPLMarketV2, pin: dict) -> float | None:
    """Return P(YES side) using Dixon-Coles model."""
    pinnacle = _find_pinnacle(market.team_a, market.team_b, market.game_date, pin)
    if pinnacle is None:
        return None
    p_home_pin, p_away_pin = pinnacle

    # Back out λ, μ from Pinnacle
    lam, mu = invert_pinnacle_to_lambda_mu(p_home_pin, p_away_pin)

    # Compute DC outcome probs
    p_home_dc, p_draw_dc, p_away_dc = dc_outcome_probs(lam, mu, DC_RHO)

    if market.outcome_type == "team_a_wins":
        return p_home_dc
    elif market.outcome_type == "draw":
        return p_draw_dc
    elif market.outcome_type == "single_team":
        return p_home_dc  # team_a is "home" in our key ordering
    return None


# ── Kelly + simulation (same structure as v1) ─────────────────────────

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
    outcome_type: str
    side: str
    entry_price: float
    exit_price: float
    close_price: float
    edge: float
    position_usd: float
    n_contracts: int
    pnl: float
    green_booked: bool
    stopped_out: bool
    time_exited: bool
    clv: float


def simulate_trade(market, prob, capital, params):
    prices = market.prices
    if len(prices) < 2: return None
    entry_price = prices[0][1]
    close_price = prices[-1][1]
    raw_edge = prob - entry_price
    both = params.get("both_sides", False)

    if raw_edge >= params["min_edge"] and raw_edge <= params["max_edge"]:
        side = "yes"; edge = raw_edge; trade_price = entry_price
    elif both and -raw_edge >= params["min_edge"] and -raw_edge <= params["max_edge"]:
        side = "no"; edge = -raw_edge; trade_price = 1 - entry_price
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

    if gb or stopped or timed:
        pnl = n * (exit_price - entry_price) if side == "yes" else n * (entry_price - exit_price)
    else:
        if side == "yes":
            pnl = n * (1 - entry_price) if market.won == 1 else -cost
        else:
            pnl = n * entry_price if market.won == 0 else -cost

    clv_val = (close_price - entry_price) if side == "yes" else (entry_price - close_price)

    return TradeResult(
        game_date=market.game_date, outcome_type=market.outcome_type,
        side=side, entry_price=entry_price, exit_price=exit_price, close_price=close_price,
        edge=edge, position_usd=cost, n_contracts=n, pnl=pnl,
        green_booked=gb, stopped_out=stopped, time_exited=timed, clv=clv_val,
    )


def compute_metrics(trades, initial_capital):
    if not trades:
        return {"score": 0, "n_trades": 0, "total_pnl": 0, "win_rate": 0,
                "green_book_rate": 0, "sharpe": 0, "max_drawdown": 0,
                "profit_factor": 0, "turnover": 0, "avg_edge": 0, "avg_clv": 0, "by_type": {}}
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
        bt = by_type[t.outcome_type]
        bt["n"] += 1; bt["pnl"] += t.pnl; bt["clv"] += t.clv
        if t.pnl > 0: bt["wins"] += 1

    return {
        "score": score, "n_trades": n, "total_pnl": total_pnl,
        "win_rate": wins / n, "green_book_rate": gbs / n,
        "sharpe": sharpe, "max_drawdown": max_dd, "profit_factor": pf,
        "turnover": turnover, "avg_edge": sum(t.edge for t in trades) / n,
        "avg_clv": avg_clv,
        "by_type": {k: dict(v) for k, v in by_type.items()},
    }


DEFAULT_PARAMS = {
    "min_edge": 0.05,
    "max_edge": 0.30,
    "min_price": 0.15,
    "max_price": 0.70,
    "min_prices": 10,
    "green_book_enabled": True,
    "green_book_delta": 0.15,
    "stop_loss_enabled": True,
    "stop_loss_delta": 0.25,
    "max_hold_fraction": 1.0,
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
    print(f"Loading EPL v2 (Dixon-Coles) data...", flush=True)
    conn = sqlite3.connect(f"file:{db_path}?mode=ro&immutable=1", uri=True, timeout=60)
    pin = load_pinnacle_epl(conn)
    n_cand, gen = iter_epl_markets(conn, params["min_prices"])

    print(f"\nRunning EPL v2 backtest...", flush=True)
    capital = params["initial_capital"]
    trades = []
    skipped = defaultdict(int)
    count = 0

    for market in gen:
        count += 1
        if limit > 0 and count > limit: break
        if time.time() - t0 > TIME_BUDGET: break
        prob = model_prob_dc(market, pin)
        if prob is None:
            skipped[f"no_model_{market.outcome_type}"] += 1
            continue
        result = simulate_trade(market, prob, capital, params)
        if result is None:
            skipped[f"no_edge_{market.outcome_type}"] += 1
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
        print("\nBy outcome type:")
        for t, d in sorted(r["by_type"].items(), key=lambda x: -x[1]["pnl"]):
            wr = d["wins"] / max(d["n"], 1)
            clv = d["clv"] / max(d["n"], 1) * 100
            print(f"  {t}: {d['n']} trades, ${d['pnl']:.0f} pnl, wr={wr:.0%}, CLV={clv:.1f}¢")
    if "skipped" in r:
        total_skip = sum(r["skipped"].values())
        print(f"\nSkipped: {total_skip}")
        for k, v in sorted(r["skipped"].items(), key=lambda x: -x[1])[:5]:
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
