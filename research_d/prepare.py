"""Strategy D — Backtest Harness v2 (PMD Dataset).

Immutable evaluation harness for the Live Edge Harvester (LEH).
Agent modifies strategy_params.py only.

Designed for the full PMD dataset (~958M prices, 250K+ markets).
Handles PMD-specific data format: team names from questions,
separate game result DB for Elo/Pinnacle matching.

Usage:
    PYTHONPATH=. python3 research_d/prepare.py
    PYTHONPATH=. python3 research_d/prepare.py --report detailed
    PYTHONPATH=. python3 research_d/prepare.py --walk-forward

Output:
    score=X.X  pnl=$X.XX  trades=N  win_rate=X.X%  green_book_rate=X.X%
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from research_d.strategy_params import PARAMS

DATA_DIR = Path(__file__).parent / "data"
DEFAULT_DB_PATH = DATA_DIR / "sports_backtest.sqlite"
TIME_BUDGET = 7200  # 2 hours max

# ── Team name extraction from questions ───────────────────────────────

# Import team maps for abbreviation lookup
try:
    from research_d.download_sports_prices import (
        team_to_abbreviation,
        NBA_TEAMS,
        EPL_TEAMS,
        NFL_TEAMS,
    )
    _TEAM_MAPS = {"nba": NBA_TEAMS, "epl": EPL_TEAMS, "nfl": NFL_TEAMS}
except ImportError:
    _TEAM_MAPS = {}

    def team_to_abbreviation(name: str, sport: str) -> str:
        cleaned = re.sub(r"[^a-zA-Z\\s]", "", name).strip()
        words = cleaned.split()
        if len(words) >= 2:
            return "".join(w[0].upper() for w in words[:3])
        return cleaned[:3].upper() or "UNK"


_TWO_TEAM_RE = [
    # "Will the Nets beat the Warriors by more than 2.5 points..."
    re.compile(r"(?:NBA|NFL|EPL)?:?\s*[Ww]ill (?:the )?(.+?) (?:beat|defeat|win against|win over) (?:the )?(.+?)(?:\s+by\b.+)?[\?\s]*$"),
    # "Dallas Mavericks vs. Oklahoma City Thunder: Game 6"
    re.compile(r"^(.+?)\s+vs?\.?\s+(.+?)(?:\s*[:]\s*.+)?$"),
    # "X vs Y" (clean)
    re.compile(r"^(.+?)\s+vs?\.?\s+(.+?)[\?\s]*$"),
    # "Spread: Lakers (-4.5)" → extract team from spread market
    re.compile(r"[Ss]pread:\s*(.+?)\s*\([+-][\d.]+\)\s*(?:vs\.?\s*(.+?))?[\?\s]*$"),
    # "Over/Under X.5: Team A vs Team B"
    re.compile(r"(?:Over|Under)\s*[\d.]+[:\s]+(.+?)\s+vs?\.?\s+(.+?)[\?\s]*$"),
    # "Team A to beat Team B"
    re.compile(r"^(.+?)\s+to\s+beat\s+(.+?)[\?\s]*$"),
    # "Moneyline: Team A vs Team B"
    re.compile(r"[Mm]oneyline:\s*(.+?)\s+vs?\.?\s+(.+?)[\?\s]*$"),
]

# Non-game market keywords — skip these entirely
_NON_GAME_KEYWORDS = [
    "mvp", "rookie of the year", "ballon d'or", "top scorer",
    "final four", "overtime", "championship", "wins the",
    "win the nba", "win the premier", "win the super",
    "how many", "total wins", "make the playoffs",
    "player prop", "which team", "run for a td",
    "award", "all-star", "draft", "trade",
]


def _is_game_market(question: str) -> bool:
    """Check if a question is about a specific game (not futures/props)."""
    if not question:
        return False
    lower = question.lower()
    return not any(kw in lower for kw in _NON_GAME_KEYWORDS)


def _parse_teams(question: str, sport: str) -> tuple[str, str] | None:
    """Extract (team_a_abbr, team_b_abbr) from market question."""
    if not question or not _is_game_market(question):
        return None

    # Strip common prefixes
    q = re.sub(r"^(?:NBA|NFL|EPL|UFC|NHL)\s*:\s*", "", question.strip())
    q = q.strip().rstrip("?").strip()

    for pat in _TWO_TEAM_RE:
        m = pat.match(q)
        if m and m.lastindex and m.lastindex >= 2 and m.group(2):
            a_raw = m.group(1).strip()
            b_raw = m.group(2).strip()
            if not a_raw or not b_raw:
                continue
            # Skip if either side is too long (likely not a team name)
            if len(a_raw) > 40 or len(b_raw) > 40:
                continue
            a = team_to_abbreviation(a_raw, sport)
            b = team_to_abbreviation(b_raw, sport)
            if a != "UNK" and b != "UNK" and a != b:
                return a, b
    return None


# ── Data Loading ──────────────────────────────────────────────────────


@dataclass
class MarketData:
    """All data needed for one backtest trade."""
    token_id: str
    game_id: str
    sport: str
    question: str
    outcome: str
    won: int              # 1 = YES won, 0 = NO won
    game_date: str
    volume: float
    team_a: str           # Parsed from question
    team_b: str
    prices: list[tuple[int, float]]  # [(ts, price), ...]


@dataclass
class EloSnapshot:
    """Elo/Glicko ratings for a team at a point in time."""
    elo: float
    glicko: float
    glicko_rd: float


def iter_markets(
    conn: sqlite3.Connection,
    enabled_sports: list[str],
    min_prices: int,
    min_volume: float,
) -> tuple[int, any]:
    """Yield backtest-eligible markets one at a time (memory-efficient).

    Returns (total_eligible, generator of MarketData).
    Streams prices from DB instead of loading all into RAM.
    """
    t0 = time.time()

    # Step 1: Find eligible markets with team names parseable from question
    sports_filter = ",".join(f"'{s}'" for s in enabled_sports)
    rows = conn.execute(f"""
        SELECT m.token_id, m.game_id, g.sport, m.question, m.outcome,
               m.won, g.game_date, COALESCE(m.volume, 0) as volume
        FROM markets m
        JOIN games g ON m.game_id = g.game_id
        WHERE m.won IS NOT NULL
        AND g.sport IN ({sports_filter})
    """).fetchall()

    print(f"  Eligible markets (resolved, known sport): {len(rows)}", flush=True)

    # Step 2: Parse team names and filter to game markets only
    candidates: list[tuple] = []
    no_teams = 0
    for token_id, game_id, sport, question, outcome, won, game_date, volume in rows:
        teams = _parse_teams(question, sport)
        if not teams:
            no_teams += 1
            continue
        candidates.append((token_id, game_id, sport, question, outcome, won, game_date, volume, teams[0], teams[1]))

    print(f"  With parsed teams: {len(candidates)} (skipped {no_teams} without teams)", flush=True)

    def _gen():
        loaded = 0
        skipped_prices = 0
        for token_id, game_id, sport, question, outcome, won, game_date, volume, team_a, team_b in candidates:
            # Load prices on-demand (one market at a time, no bulk RAM)
            price_rows = conn.execute(
                "SELECT ts, price FROM prices WHERE token_id = ? ORDER BY ts",
                (token_id,),
            ).fetchall()

            prices = [(ts, price) for ts, price in price_rows if 0 < price < 1]
            if len(prices) < min_prices:
                skipped_prices += 1
                continue

            loaded += 1
            if loaded % 1000 == 0:
                elapsed = time.time() - t0
                print(f"  ... loaded {loaded} markets ({elapsed:.0f}s)", flush=True)

            yield MarketData(
                token_id=token_id,
                game_id=game_id,
                sport=sport,
                question=question,
                outcome=outcome,
                won=won,
                game_date=game_date,
                volume=volume,
                team_a=team_a,
                team_b=team_b,
                prices=prices,
            )

        elapsed = time.time() - t0
        print(f"  Done: {loaded} markets loaded, {skipped_prices} skipped (no prices), {elapsed:.0f}s", flush=True)

    return len(candidates), _gen()


def load_ratings(conn: sqlite3.Connection) -> dict[str, dict[str, EloSnapshot]]:
    """Load Elo/Glicko ratings indexed by (team, date).

    Returns: {team: {date: EloSnapshot}}
    """
    rows = conn.execute(
        "SELECT team, date, elo, glicko_rating, glicko_rd FROM ratings ORDER BY team, date"
    ).fetchall()

    ratings: dict[str, dict[str, EloSnapshot]] = defaultdict(dict)
    for team, date_str, elo, glicko, glicko_rd in rows:
        ratings[team][date_str] = EloSnapshot(
            elo=elo, glicko=glicko, glicko_rd=glicko_rd or 350.0,
        )

    print(f"  Elo/Glicko ratings: {len(ratings)} teams, {len(rows)} snapshots", flush=True)
    return ratings


def load_pinnacle(conn: sqlite3.Connection) -> dict[str, tuple[float, float]]:
    """Load Pinnacle no-vig odds indexed by game_id.

    Returns: {game_id: (home_prob, away_prob)}
    """
    rows = conn.execute(
        "SELECT game_id, home_prob_novig, away_prob_novig FROM pinnacle_odds"
    ).fetchall()

    pinnacle: dict[str, tuple[float, float]] = {}
    for game_id, hp, ap in rows:
        if hp and ap and hp > 0 and ap > 0:
            pinnacle[game_id] = (hp, ap)

    print(f"  Pinnacle odds: {len(pinnacle)} games", flush=True)
    return pinnacle


# ── Probability Model ─────────────────────────────────────────────────


def _get_rating(
    ratings: dict[str, dict[str, EloSnapshot]],
    team: str,
    date: str,
) -> EloSnapshot | None:
    """Get the latest rating for a team on or before a given date."""
    team_hist = ratings.get(team)
    if not team_hist:
        return None
    # Find the latest date <= game_date
    best_date = None
    for d in team_hist:
        if d <= date:
            if best_date is None or d > best_date:
                best_date = d
    if best_date:
        return team_hist[best_date]
    return None


def _find_pinnacle_game_id(
    pinnacle: dict[str, tuple[float, float]],
    sport: str,
    date: str,
    team_a: str,
    team_b: str,
) -> str | None:
    """Find matching Pinnacle game_id by sport + date + teams.

    Pinnacle game_ids are formatted: sport_YYYYMMDD_AWAY_HOME
    (or sport_YYYYMMDD_HOME_AWAY depending on source).
    We try both orders.
    """
    date_compact = date.replace("-", "")
    candidates = [
        f"{sport}_{date_compact}_{team_a}_{team_b}",
        f"{sport}_{date_compact}_{team_b}_{team_a}",
    ]
    for cand in candidates:
        if cand in pinnacle:
            return cand

    # Fuzzy: search for any game_id with matching sport + date + both teams
    prefix = f"{sport}_{date_compact}_"
    for gid in pinnacle:
        if gid.startswith(prefix) and team_a in gid and team_b in gid:
            return gid

    return None


def compute_model_prob(
    team_a: str,
    team_b: str,
    sport: str,
    date: str,
    ratings: dict[str, dict[str, EloSnapshot]],
    pinnacle: dict[str, tuple[float, float]],
    params: dict,
) -> float | None:
    """Compute ensemble model probability for team_a winning.

    Model: weighted average of Elo/Glicko-2 and Pinnacle no-vig.
    """
    # Get Elo ratings
    rating_a = _get_rating(ratings, team_a, date)
    rating_b = _get_rating(ratings, team_b, date)

    elo_prob = None
    glicko_prob = None

    if rating_a and rating_b:
        # Elo win probability
        elo_diff = rating_a.elo - rating_b.elo
        elo_prob = 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))

        # Glicko win probability
        glicko_diff = rating_a.glicko - rating_b.glicko
        q = math.log(10) / 400
        g_rd = 1.0 / math.sqrt(1.0 + 3 * q**2 * (rating_b.glicko_rd**2) / (math.pi**2))
        glicko_prob = 1.0 / (1.0 + 10.0 ** (-g_rd * glicko_diff / 400.0))

    # Get Pinnacle odds
    pinnacle_prob_a = None
    pin_gid = _find_pinnacle_game_id(pinnacle, sport, date, team_a, team_b)
    if pin_gid:
        hp, ap = pinnacle[pin_gid]
        # Determine which side is team_a
        # Game ID format: sport_date_X_Y — check if team_a matches first or second
        parts = pin_gid.split("_")
        if len(parts) >= 4:
            if parts[2] == team_a:
                # team_a is the first team in game_id (could be home or away)
                pinnacle_prob_a = hp
            elif parts[3] == team_a:
                pinnacle_prob_a = ap
            else:
                pinnacle_prob_a = hp  # default: assume first position

    # Ensemble
    if elo_prob is not None and pinnacle_prob_a is not None:
        elo_blend = params["elo_weight"]
        pin_blend = params["pinnacle_weight"]
        # Blend Elo and Glicko first, then ensemble with Pinnacle
        elo_glicko = 0.45 * elo_prob + 0.55 * (glicko_prob or elo_prob)
        return elo_blend * elo_glicko + pin_blend * pinnacle_prob_a

    elif elo_prob is not None:
        # Elo/Glicko only
        ew = params["elo_only_weight_elo"]
        gw = params["elo_only_weight_glicko"]
        return ew * elo_prob + gw * (glicko_prob or elo_prob)

    elif pinnacle_prob_a is not None:
        # Pinnacle only
        return pinnacle_prob_a

    return None


# ── Kelly Sizing ──────────────────────────────────────────────────────


def kelly_size(
    edge: float,
    entry_price: float,
    capital: float,
    params: dict,
) -> float:
    """Compute position size in dollars using fractional Kelly."""
    if entry_price <= 0 or entry_price >= 1:
        return 0.0

    # Kelly formula: f* = (bp - q) / b where b = (1/p - 1)
    p = entry_price + edge  # Our estimated true probability
    p = max(0.01, min(0.99, p))
    q = 1.0 - p
    b = (1.0 / entry_price) - 1.0
    if b <= 0:
        return 0.0

    kelly_raw = (b * p - q) / b
    kelly_raw = max(0, min(kelly_raw, params["kelly_raw_cap"]))

    size = capital * params["kelly_fraction"] * kelly_raw
    max_size = capital * params["max_position_pct"]
    return min(size, max_size)


# ── Green Book Simulator ─────────────────────────────────────────────


@dataclass
class TradeResult:
    """Result of a single simulated trade."""
    token_id: str
    sport: str
    game_date: str
    team_a: str
    team_b: str
    model_prob: float
    entry_price: float
    edge: float
    position_usd: float
    n_contracts: int
    won: int
    green_booked: bool
    exit_price: float
    pnl: float
    max_price: float
    min_price: float
    n_prices: int
    hold_fraction: float    # Fraction of price series before exit


def simulate_trade(
    market: MarketData,
    model_prob: float,
    capital: float,
    params: dict,
) -> TradeResult | None:
    """Simulate a single D1 Green Book trade on a market's price trajectory."""
    prices = market.prices
    if len(prices) < 2:
        return None

    # Entry: use first available price
    entry_ts, entry_price = prices[0]

    # Compute edge — support both sides
    raw_edge = model_prob - entry_price
    both_sides = params.get("both_sides", False)

    # Determine trade direction
    if raw_edge >= params["min_edge"] and raw_edge <= params["max_edge"]:
        # BUY YES — model says underpriced
        side = "yes"
        edge = raw_edge
        trade_price = entry_price
    elif both_sides and (-raw_edge) >= params["min_edge"] and (-raw_edge) <= params["max_edge"]:
        # BUY NO (= sell YES) — model says overpriced
        side = "no"
        edge = -raw_edge
        trade_price = 1.0 - entry_price  # NO price = 1 - YES price
    else:
        return None

    if trade_price < params["min_price"] or trade_price > params["max_price"]:
        return None

    # Position size (based on edge magnitude)
    pos_usd = kelly_size(edge, trade_price, capital, params)
    n_contracts = int(pos_usd / trade_price)
    if n_contracts < 1:
        return None
    actual_cost = n_contracts * trade_price

    # Green book target + stop loss
    sport = market.sport
    delta_key = f"green_book_delta_{sport}"
    delta = params.get(delta_key, params["green_book_delta_default"])
    stop_loss_enabled = params.get("stop_loss_enabled", False)
    sl_delta = params.get("stop_loss_delta", 0.15)

    if side == "yes":
        target = entry_price + delta
        stop_loss_price = entry_price - sl_delta if stop_loss_enabled else 0
    else:
        # NO side: we profit when YES price DROPS
        target = entry_price - delta  # YES price drops = NO profits
        stop_loss_price = entry_price + sl_delta if stop_loss_enabled else 2.0

    # Time-based exit
    max_hold_frac = params.get("max_hold_fraction", 1.0)
    max_hold_idx = int(len(prices) * max_hold_frac) if max_hold_frac < 1.0 else len(prices)

    # Walk through prices
    green_booked = False
    stopped_out = False
    time_exited = False
    exit_price = entry_price
    max_price = entry_price
    min_price = entry_price
    exit_idx = len(prices) - 1

    for idx, (ts, price) in enumerate(prices[1:], 1):
        max_price = max(max_price, price)
        min_price = min(min_price, price)

        if side == "yes":
            gb_hit = params["green_book_enabled"] and price >= target
            sl_hit = stop_loss_enabled and price <= stop_loss_price
        else:
            gb_hit = params["green_book_enabled"] and price <= target
            sl_hit = stop_loss_enabled and price >= stop_loss_price

        if gb_hit:
            green_booked = True
            exit_price = price
            exit_idx = idx
            break
        elif sl_hit:
            stopped_out = True
            exit_price = price
            exit_idx = idx
            break
        elif idx >= max_hold_idx:
            time_exited = True
            exit_price = price
            exit_idx = idx
            break
    else:
        exit_price = prices[-1][1]

    # P&L calculation
    if green_booked or stopped_out or time_exited:
        if side == "yes":
            pnl = n_contracts * (exit_price - entry_price)
        else:
            pnl = n_contracts * (entry_price - exit_price)  # NO: profit when price drops
    else:
        # Hold to resolution
        if side == "yes":
            pnl = n_contracts * (1.0 - entry_price) if market.won == 1 else -actual_cost
        else:
            pnl = n_contracts * (entry_price - 0.0) if market.won == 0 else -actual_cost

    hold_fraction = exit_idx / max(len(prices) - 1, 1)

    return TradeResult(
        token_id=market.token_id,
        sport=sport,
        game_date=market.game_date,
        team_a=market.team_a,
        team_b=market.team_b,
        model_prob=model_prob,
        entry_price=entry_price,
        edge=edge,
        position_usd=actual_cost,
        n_contracts=n_contracts,
        won=market.won,
        green_booked=green_booked,
        exit_price=exit_price,
        pnl=pnl,
        max_price=max_price,
        min_price=min_price,
        n_prices=len(prices),
        hold_fraction=hold_fraction,
    )


# ── Metrics ───────────────────────────────────────────────────────────


@dataclass
class BacktestMetrics:
    """Aggregate metrics from a backtest run."""
    n_trades: int = 0
    total_pnl: float = 0.0
    win_rate: float = 0.0
    green_book_rate: float = 0.0
    could_gb_rate: float = 0.0
    avg_edge: float = 0.0
    sharpe: float = 0.0
    max_drawdown: float = 0.0
    profit_factor: float = 0.0
    turnover: float = 0.0
    avg_hold_fraction: float = 0.0
    score: float = 0.0
    # Per-sport breakdown
    sport_pnl: dict[str, float] = field(default_factory=dict)
    sport_trades: dict[str, int] = field(default_factory=dict)


def compute_metrics(
    trades: list[TradeResult],
    initial_capital: float,
) -> BacktestMetrics:
    """Compute all backtest metrics from trade results."""
    m = BacktestMetrics()
    if not trades:
        return m

    m.n_trades = len(trades)

    # Basic stats
    pnls = [t.pnl for t in trades]
    m.total_pnl = sum(pnls)
    wins = sum(1 for t in trades if t.pnl > 0)
    m.win_rate = wins / m.n_trades

    gb_count = sum(1 for t in trades if t.green_booked)
    m.green_book_rate = gb_count / m.n_trades

    # Could have green booked (max price >= target)
    could_gb = sum(1 for t in trades if t.max_price >= t.entry_price + 0.05)
    m.could_gb_rate = could_gb / m.n_trades

    m.avg_edge = sum(t.edge for t in trades) / m.n_trades
    m.avg_hold_fraction = sum(t.hold_fraction for t in trades) / m.n_trades

    # Sharpe (daily returns approximation)
    # Group trades by date, compute daily P&L
    daily_pnl: dict[str, float] = defaultdict(float)
    for t in trades:
        daily_pnl[t.game_date] += t.pnl

    if len(daily_pnl) > 1:
        daily_returns = list(daily_pnl.values())
        mean_r = sum(daily_returns) / len(daily_returns)
        var_r = sum((r - mean_r) ** 2 for r in daily_returns) / (len(daily_returns) - 1)
        std_r = math.sqrt(var_r) if var_r > 0 else 1e-9
        m.sharpe = (mean_r / std_r) * math.sqrt(252)  # Annualized

    # Max drawdown
    cumulative = 0.0
    peak = 0.0
    max_dd = 0.0
    for pnl in pnls:
        cumulative += pnl
        peak = max(peak, cumulative)
        dd = (peak - cumulative) / max(initial_capital, 1)
        max_dd = max(max_dd, dd)
    m.max_drawdown = max_dd

    # Profit factor
    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p < 0))
    m.profit_factor = gross_profit / max(gross_loss, 1e-9)

    # Capital turnover
    total_traded = sum(t.position_usd for t in trades)
    m.turnover = total_traded / max(initial_capital, 1)

    # Per-sport
    for t in trades:
        m.sport_pnl[t.sport] = m.sport_pnl.get(t.sport, 0) + t.pnl
        m.sport_trades[t.sport] = m.sport_trades.get(t.sport, 0) + 1

    # Composite score
    m.score = _composite_score(m, initial_capital)

    return m


def _composite_score(m: BacktestMetrics, initial_capital: float) -> float:
    """Strategy D composite score.

    Balances profitability, capital turnover, and consistency.
    """
    pnl_factor = m.total_pnl / max(initial_capital, 1) * 100
    sharpe_factor = min(max(m.sharpe, 0) / 3.0, 2.0)
    trade_factor = min(m.n_trades / 100, 2.0)
    dd_factor = max(0, 1.0 - m.max_drawdown * 2)
    turnover_factor = min(m.turnover / 10.0, 1.5)
    gb_factor = 1.0 + m.green_book_rate * 0.5

    score = (
        pnl_factor
        * (1 + sharpe_factor)
        * trade_factor
        * dd_factor
        * turnover_factor
        * gb_factor
    )
    return score


# ── Walk-Forward Validation ───────────────────────────────────────────


def walk_forward(
    markets: list[MarketData],
    ratings: dict[str, dict[str, EloSnapshot]],
    pinnacle: dict[str, tuple[float, float]],
    params: dict,
) -> list[BacktestMetrics]:
    """Run walk-forward validation with expanding training window.

    Returns metrics for each test fold.
    """
    # Sort markets by date
    sorted_markets = sorted(markets, key=lambda m: m.game_date)
    if not sorted_markets:
        return []

    dates = sorted(set(m.game_date for m in sorted_markets))
    if len(dates) < 60:  # Need at least 2 months
        print("  WARNING: Not enough date range for walk-forward", flush=True)
        return []

    # Create monthly boundaries
    months: list[str] = []
    current_month = ""
    for d in dates:
        m = d[:7]  # YYYY-MM
        if m != current_month:
            months.append(d)
            current_month = m

    if len(months) < 4:
        print(f"  WARNING: Only {len(months)} months, need >= 4 for walk-forward", flush=True)
        return []

    train_months = params["wf_train_months"]
    test_months = params["wf_test_months"]

    folds: list[BacktestMetrics] = []

    for fold_start in range(train_months, len(months) - test_months + 1, test_months):
        train_end_date = months[fold_start]
        test_end_idx = min(fold_start + test_months, len(months) - 1)
        test_end_date = months[test_end_idx] if test_end_idx < len(months) else dates[-1]

        # Test markets: between train_end and test_end
        test_markets = [
            m for m in sorted_markets
            if train_end_date <= m.game_date < test_end_date
        ]

        if not test_markets:
            continue

        # Run backtest on test fold
        capital = params["initial_capital"]
        trades: list[TradeResult] = []

        for market in test_markets:
            prob = compute_model_prob(
                market.team_a, market.team_b, market.sport, market.game_date,
                ratings, pinnacle, params,
            )
            if prob is None:
                continue

            result = simulate_trade(market, prob, capital, params)
            if result:
                trades.append(result)
                capital += result.pnl

        metrics = compute_metrics(trades, params["initial_capital"])
        folds.append(metrics)

        print(
            f"  Fold {len(folds)}: {train_end_date}→{test_end_date} "
            f"trades={metrics.n_trades} pnl=${metrics.total_pnl:.2f} "
            f"wr={metrics.win_rate:.1%} gb={metrics.green_book_rate:.1%}",
            flush=True,
        )

    return folds


# ── Main Evaluate ─────────────────────────────────────────────────────


def evaluate(params: dict | None = None, db_path: Path | None = None, limit: int = 0) -> dict:
    """Run full backtest with given parameters.

    Returns dict with all metrics for autoresearch scoring.
    """
    if params is None:
        params = PARAMS
    if db_path is None:
        db_path = DEFAULT_DB_PATH

    t0 = time.time()
    print(f"Loading data from {db_path}...", flush=True)

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA cache_size=-64000")

    # Load ratings + Pinnacle (small, fits in memory)
    ratings = load_ratings(conn)
    pinnacle = load_pinnacle(conn)

    # Stream markets (prices loaded one at a time)
    n_candidates, market_gen = iter_markets(
        conn,
        enabled_sports=params["enabled_sports"],
        min_prices=params["min_prices"],
        min_volume=params["min_volume"],
    )

    if n_candidates == 0:
        conn.close()
        print("ERROR: No eligible markets found!", flush=True)
        return {
            "score": 0, "n_trades": 0, "total_pnl": 0, "win_rate": 0,
            "green_book_rate": 0, "could_gb_rate": 0, "avg_edge": 0,
            "sharpe": 0, "max_drawdown": 0, "profit_factor": 0,
            "turnover": 0, "avg_hold_fraction": 0, "sport_pnl": {},
            "sport_trades": {},
        }

    # Run simulation (streaming — one market at a time)
    print(f"\nRunning backtest on {n_candidates} candidate markets...", flush=True)
    capital = params["initial_capital"]
    trades: list[TradeResult] = []
    skipped = {"no_model": 0, "no_edge": 0, "no_size": 0}

    market_count = 0
    for market in market_gen:
        market_count += 1
        if limit > 0 and market_count > limit:
            print(f"  LIMIT {limit} reached, stopping", flush=True)
            break
        if time.time() - t0 > TIME_BUDGET:
            print(f"  TIME BUDGET ({TIME_BUDGET}s) exceeded, stopping", flush=True)
            break

        prob = compute_model_prob(
            market.team_a, market.team_b, market.sport, market.game_date,
            ratings, pinnacle, params,
        )
        if prob is None:
            skipped["no_model"] += 1
            continue

        result = simulate_trade(market, prob, capital, params)
        if result is None:
            skipped["no_edge"] += 1
            continue

        trades.append(result)
        capital += result.pnl
        # Free price data immediately (GC)
        market.prices = []

    conn.close()

    # Compute metrics
    metrics = compute_metrics(trades, params["initial_capital"])

    elapsed = time.time() - t0
    print(f"\nBacktest done in {elapsed:.1f}s", flush=True)
    print(f"Skipped: {skipped}", flush=True)

    return {
        "score": metrics.score,
        "n_trades": metrics.n_trades,
        "total_pnl": metrics.total_pnl,
        "win_rate": metrics.win_rate,
        "green_book_rate": metrics.green_book_rate,
        "could_gb_rate": metrics.could_gb_rate,
        "avg_edge": metrics.avg_edge,
        "sharpe": metrics.sharpe,
        "max_drawdown": metrics.max_drawdown,
        "profit_factor": metrics.profit_factor,
        "turnover": metrics.turnover,
        "avg_hold_fraction": metrics.avg_hold_fraction,
        "sport_pnl": metrics.sport_pnl,
        "sport_trades": metrics.sport_trades,
    }


# ── CLI ───────────────────────────────────────────────────────────────


def print_results(results: dict) -> None:
    """Print results in machine-parseable + human-readable format."""
    print(f"\nscore={results['score']:.1f}")
    print(f"pnl=${results['total_pnl']:.2f}")
    print(f"trades={results['n_trades']}")
    print(f"win_rate={results['win_rate']:.1%}")
    print(f"green_book_rate={results['green_book_rate']:.1%}")
    print(f"could_gb_rate={results['could_gb_rate']:.1%}")
    print(f"sharpe={results['sharpe']:.2f}")
    print(f"max_dd={results['max_drawdown']:.1%}")
    print(f"profit_factor={results['profit_factor']:.2f}")
    print(f"turnover={results['turnover']:.1f}x")
    print(f"avg_edge={results['avg_edge']:.3f}")
    print(f"avg_hold={results['avg_hold_fraction']:.1%}")

    if results.get("sport_pnl"):
        print("\nPer-sport breakdown:")
        for sport in sorted(results["sport_pnl"]):
            pnl = results["sport_pnl"][sport]
            n = results["sport_trades"].get(sport, 0)
            print(f"  {sport}: ${pnl:.2f} ({n} trades)")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Strategy D Backtest v2")
    parser.add_argument(
        "--db", type=str, default=str(DEFAULT_DB_PATH),
        help="Path to SQLite database",
    )
    parser.add_argument(
        "--report", choices=["summary", "detailed"], default="summary",
        help="Report verbosity",
    )
    parser.add_argument(
        "--walk-forward", action="store_true",
        help="Run walk-forward validation",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Limit number of candidate markets (for quick testing)",
    )
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"ERROR: Database not found: {db_path}")
        sys.exit(1)

    results = evaluate(PARAMS, db_path, limit=args.limit)
    print_results(results)

    if args.walk_forward:
        print("\n=== WALK-FORWARD VALIDATION ===")
        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA cache_size=-64000")

        _, market_gen = iter_markets(
            conn,
            enabled_sports=PARAMS["enabled_sports"],
            min_prices=PARAMS["min_prices"],
            min_volume=PARAMS["min_volume"],
        )
        # For walk-forward we need all markets in memory (sorted by date)
        markets = list(market_gen)
        ratings_data = load_ratings(conn)
        pinnacle_data = load_pinnacle(conn)
        conn.close()

        folds = walk_forward(markets, ratings_data, pinnacle_data, PARAMS)
        if folds:
            profitable_folds = sum(1 for f in folds if f.total_pnl > 0)
            avg_pnl = sum(f.total_pnl for f in folds) / len(folds)
            avg_wr = sum(f.win_rate for f in folds) / len(folds)
            print(f"\nWalk-forward: {profitable_folds}/{len(folds)} profitable folds")
            print(f"Avg fold P&L: ${avg_pnl:.2f}")
            print(f"Avg fold WR: {avg_wr:.1%}")


if __name__ == "__main__":
    main()
