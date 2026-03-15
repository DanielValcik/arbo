"""Strategy D — Sports Backtest Engine (Phase 1: Small Realistic).

Simulates the Green Book Engine (D1) and Overreaction Fade (D2) on
historical Polymarket sports price data. Uses Elo/Glicko-2 as the
probability model (Pinnacle added when overlap exists).

Phase 1 uses available NBA playoff data (~65 markets with hourly prices)
to validate whether the green book concept works on real data.

Key question answered:
    "If we enter when our model says Polymarket is mispriced, will the
     price move favorably at ANY point before resolution?"

Usage:
    PYTHONPATH=. python3 research_d/backtest_sports.py
    PYTHONPATH=. python3 research_d/backtest_sports.py --min-edge 0.03 --green-book-delta 0.05
    PYTHONPATH=. python3 research_d/backtest_sports.py --report detailed

Output:
    Prints backtest results: P&L, win rate, green book rate, Sharpe, etc.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from research_d.sports_db import SportsDB
from research_d.elo_glicko_engine import EloGlickoEngine


# ── Trade Result ─────────────────────────────────────────────────────

@dataclass
class TradeResult:
    """Result of a single simulated trade."""
    game_id: str
    market_id: str
    sport: str
    question: str
    game_date: str
    home_team: str
    away_team: str
    # Model
    model_prob: float          # Our estimated probability (Elo/ensemble)
    market_price: float        # Polymarket entry price
    edge: float                # model_prob - market_price
    # Sizing
    position_size: float       # Dollar amount bet
    n_contracts: int           # Number of YES contracts
    # Outcome
    won_resolution: bool       # Did this market resolve YES?
    resolution_pnl: float      # P&L if held to resolution
    # Green book
    green_booked: bool         # Did we green book?
    green_book_price: float    # Exit price (if green booked)
    green_book_pnl: float      # P&L from green book
    max_price_during: float    # Highest price seen during game
    min_price_during: float    # Lowest price seen during game
    # Final
    actual_pnl: float          # Green book PnL if green booked, else resolution PnL
    hold_hours: float          # How long position was held
    n_prices: int              # Number of price points during game


# ── Backtest Parameters ──────────────────────────────────────────────

@dataclass
class BacktestParams:
    """All tunable parameters for the backtest."""
    # Quality gate
    min_edge: float = 0.03         # Minimum edge to enter
    max_edge: float = 0.30         # Maximum edge (anomaly filter)
    min_price: float = 0.10        # Min Polymarket price (no extreme longshots)
    max_price: float = 0.85        # Max price (limited upside)
    min_prices: int = 10           # Min price data points for a market

    # Green book
    green_book_delta: float = 0.05  # Exit when price rises by this much
    green_book_enabled: bool = True

    # Model weights
    elo_weight: float = 0.45       # Elo contribution (when no Pinnacle)
    glicko_weight: float = 0.55    # Glicko contribution (when no Pinnacle)
    pinnacle_weight: float = 0.60  # Pinnacle weight (when available)

    # Sizing
    initial_capital: float = 1000.0
    kelly_fraction: float = 0.25   # Quarter-Kelly
    kelly_raw_cap: float = 0.15    # Max kelly before fraction
    max_position_pct: float = 0.05 # Max 5% per trade

    # Side: which side do we trade?
    trade_side: str = "undervalued"  # "undervalued" (buy YES when model > market)


# ── Probability Model ────────────────────────────────────────────────

# Team abbreviation aliases (different sources use different names)
TEAM_ALIASES = {
    "NYK": "NY", "NY": "NY",
    "GSW": "GS", "GS": "GS",
    "NOP": "NO", "NO": "NO",
    "SAS": "SA", "SA": "SA",
    "PHO": "PHX", "PHX": "PHX",
    "WSH": "WAS", "WAS": "WSH",
    "BRK": "BKN", "BKN": "BKN",
    "UTA": "UTAH", "UTAH": "UTAH",
    "CHA": "CHA", "CHO": "CHA",
}


def _normalize_team(team: str) -> str:
    """Normalize team abbreviation across different data sources."""
    return TEAM_ALIASES.get(team, team)


def _find_team_in_engine(team: str, engine: EloGlickoEngine) -> str | None:
    """Find a team in the Elo engine, trying aliases."""
    if team in engine.teams:
        return team
    normalized = _normalize_team(team)
    if normalized in engine.teams:
        return normalized
    # Reverse lookup
    for alias, canonical in TEAM_ALIASES.items():
        if canonical == team and alias in engine.teams:
            return alias
        if alias == team and canonical in engine.teams:
            return canonical
    return None


def compute_model_prob(
    home_team: str,
    away_team: str,
    sport: str,
    elo_engine: EloGlickoEngine,
    pinnacle_home_prob: float | None,
    params: BacktestParams,
    outcome: str = "home_win",
) -> float | None:
    """Compute our model's probability for a market outcome.

    Combines Elo/Glicko-2 with Pinnacle (when available).

    Args:
        home_team: Home team abbreviation.
        away_team: Away team abbreviation.
        sport: Sport identifier.
        elo_engine: Initialized Elo/Glicko engine.
        pinnacle_home_prob: Pinnacle no-vig home probability (or None).
        params: Backtest parameters.
        outcome: Market outcome type.

    Returns:
        Model probability for the market's YES outcome, or None if
        insufficient data.
    """
    # Normalize team names across data sources
    elo_home_team = _find_team_in_engine(home_team, elo_engine)
    elo_away_team = _find_team_in_engine(away_team, elo_engine)
    if not elo_home_team or not elo_away_team:
        return None

    pred = elo_engine.predict(elo_home_team, elo_away_team)
    elo_home = pred["ensemble_home_prob"]

    if pinnacle_home_prob is not None and pinnacle_home_prob > 0:
        # Weighted ensemble with Pinnacle
        model_home = (
            (1 - params.pinnacle_weight) * elo_home
            + params.pinnacle_weight * pinnacle_home_prob
        )
    else:
        # Elo/Glicko only
        model_home = elo_home

    if outcome in ("home_win", "moneyline"):
        # For moneyline markets, YES = first team mentioned wins
        # In our DB, this corresponds to away_team (the team traveling)
        # because game_id format is sport_date_away_home
        # But the market may be asking about either side
        # Default: treat as home team winning
        return model_home
    elif outcome == "away_win":
        return 1.0 - model_home
    else:
        # Unknown outcome type — use home probability as default
        return model_home


# ── Kelly Sizing ─────────────────────────────────────────────────────

def kelly_size(
    edge: float,
    market_price: float,
    capital: float,
    params: BacktestParams,
) -> float:
    """Compute position size using quarter-Kelly.

    Args:
        edge: model_prob - market_price.
        market_price: Polymarket YES price.
        capital: Available capital.
        params: Backtest parameters.

    Returns:
        Dollar amount to invest.
    """
    if edge <= 0 or market_price <= 0 or market_price >= 1:
        return 0.0

    # Kelly formula for binary outcome
    b = (1.0 / market_price) - 1.0  # Net odds
    p = market_price + edge           # Our estimated prob
    q = 1.0 - p

    kelly_raw = (b * p - q) / b
    kelly_raw = max(0, min(kelly_raw, params.kelly_raw_cap))

    kelly_adj = kelly_raw * params.kelly_fraction
    size = capital * kelly_adj
    size = min(size, capital * params.max_position_pct)

    return round(size, 2)


# ── Green Book Simulation ────────────────────────────────────────────

def simulate_green_book(
    prices: list[tuple[int, float]],
    entry_price: float,
    delta: float,
) -> tuple[bool, float, float, float]:
    """Simulate green booking on a price trajectory.

    Walks through minute/hourly prices and checks if the price
    ever reaches entry_price + delta. If yes, "green books" at that price.

    Args:
        prices: List of (timestamp, price) sorted by time.
        entry_price: Our entry price.
        delta: Green book target delta.

    Returns:
        (green_booked, exit_price, max_price, min_price)
    """
    target = entry_price + delta
    max_price = entry_price
    min_price = entry_price

    for _, price in prices:
        max_price = max(max_price, price)
        min_price = min(min_price, price)

        if price >= target:
            return True, price, max_price, min_price

    return False, prices[-1][1] if prices else entry_price, max_price, min_price


# ── Market Matching ──────────────────────────────────────────────────

def find_matching_game(
    market_game_id: str,
    home_team: str,
    away_team: str,
    game_date: str,
    db: SportsDB,
) -> dict | None:
    """Find a game matching a Polymarket market via fuzzy team+date match.

    The game_ids from Polymarket discovery and ESPN download differ
    in format (team order, abbreviation style). This function tries
    multiple matching strategies.

    Returns:
        Game row dict or None.
    """
    # Direct match
    game = db.get_game(market_game_id)
    if game and game["home_score"] is not None:
        return dict(game)

    # Try swapped team order
    parts = market_game_id.split("_")
    if len(parts) >= 4:
        swapped_id = f"{parts[0]}_{parts[1]}_{parts[3]}_{parts[2]}"
        game = db.get_game(swapped_id)
        if game and game["home_score"] is not None:
            return dict(game)

    # Fuzzy: find games on same date with matching teams
    if game_date and home_team and away_team:
        candidates = db.get_games(min_date=game_date, max_date=game_date)
        for c in candidates:
            teams = {c["home_team"], c["away_team"]}
            if home_team in teams or away_team in teams:
                if c["home_score"] is not None:
                    return dict(c)

    return None


# ── Main Backtest ────────────────────────────────────────────────────

def run_backtest(
    params: BacktestParams | None = None,
    db_path: str | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run the full Strategy D backtest.

    1. Load all markets with price data
    2. For each, compute model probability
    3. Apply quality gate
    4. Simulate entry + green book
    5. Compute P&L and metrics

    Args:
        params: Backtest parameters (default if None).
        db_path: Path to SQLite DB.
        verbose: Print progress.

    Returns:
        Results dict with trades, metrics, and summary.
    """
    if params is None:
        params = BacktestParams()

    db = SportsDB(db_path)

    # Initialize Elo engines per sport
    engines: dict[str, EloGlickoEngine] = {}
    for sport in ["nba", "epl", "nfl"]:
        engine = EloGlickoEngine(sport)
        games = db.get_games(sport=sport, status="final")
        for g in games:
            if g["home_score"] is not None:
                engine.process_game(
                    g["home_team"], g["away_team"],
                    g["home_score"], g["away_score"],
                    g["game_date"], g["season"],
                )
        engines[sport] = engine
        if verbose:
            print(f"  Elo engine {sport}: {engine._games_processed} games, "
                  f"{len(engine.teams)} teams")

    # Find all markets with price data
    markets_with_prices = db.conn.execute("""
        SELECT m.token_id, m.game_id, m.question, m.outcome, m.won,
               g.sport, g.home_team, g.away_team, g.game_date,
               g.home_score, g.away_score,
               COUNT(p.ts) as n_prices,
               MIN(p.price) as min_price_raw,
               MAX(p.price) as max_price_raw
        FROM markets m
        JOIN games g ON m.game_id = g.game_id
        JOIN prices p ON m.token_id = p.token_id
        WHERE m.game_id NOT LIKE '%HOME_AWAY%'
        GROUP BY m.token_id
        HAVING n_prices >= ?
        ORDER BY g.game_date
    """, (params.min_prices,)).fetchall()

    if verbose:
        print(f"\n  Markets with {params.min_prices}+ prices: {len(markets_with_prices)}")

    # Run backtest
    trades: list[TradeResult] = []
    skipped = defaultdict(int)
    capital = params.initial_capital

    for row in markets_with_prices:
        token_id = row["token_id"]
        sport = row["sport"]
        question = row["question"] or ""
        outcome = row["outcome"] or "home_win"

        # Skip non-game markets (futures, props, non-sport)
        if not _is_game_market(question):
            skipped["not_game_market"] += 1
            continue

        # Need sport engine
        if sport not in engines:
            skipped["no_engine"] += 1
            continue

        engine = engines[sport]
        home_team = row["home_team"]
        away_team = row["away_team"]

        if not home_team or not away_team or home_team == "HOME":
            skipped["no_teams"] += 1
            continue

        # Get prices
        prices = db.get_prices(token_id)
        if len(prices) < params.min_prices:
            skipped["too_few_prices"] += 1
            continue

        price_data = [(p["ts"], p["price"]) for p in prices]

        # Entry price = first available price (pre-game or early game)
        entry_price = price_data[0][1]

        # Quality gate
        if entry_price < params.min_price or entry_price > params.max_price:
            skipped["price_out_of_range"] += 1
            continue

        # Compute model probability
        # Get Pinnacle if available
        pinnacle = db.get_latest_pinnacle_odds(row["game_id"])
        pinnacle_home_prob = pinnacle["home_prob_novig"] if pinnacle else None

        model_prob = compute_model_prob(
            home_team, away_team, sport, engine,
            pinnacle_home_prob, params, outcome,
        )
        if model_prob is None:
            skipped["no_model_prob"] += 1
            continue

        edge = model_prob - entry_price

        # Edge filter
        if edge < params.min_edge:
            skipped["edge_too_small"] += 1
            continue
        if edge > params.max_edge:
            skipped["edge_too_large"] += 1
            continue

        # Position sizing
        size = kelly_size(edge, entry_price, capital, params)
        if size < 1.0:
            skipped["size_too_small"] += 1
            continue

        n_contracts = int(size / entry_price)
        if n_contracts < 1:
            skipped["zero_contracts"] += 1
            continue

        actual_cost = n_contracts * entry_price

        # Simulate green book
        if params.green_book_enabled:
            gb_ok, gb_price, max_p, min_p = simulate_green_book(
                price_data[1:],  # Skip entry point
                entry_price,
                params.green_book_delta,
            )
        else:
            gb_ok = False
            gb_price = entry_price
            max_p = max(p for _, p in price_data)
            min_p = min(p for _, p in price_data)

        # Resolution P&L
        won = row["won"]
        if won == 1:
            resolution_pnl = n_contracts * (1.0 - entry_price)
        elif won == 0:
            resolution_pnl = -actual_cost
        else:
            # Unresolved — use last price
            last_price = price_data[-1][1]
            resolution_pnl = n_contracts * (last_price - entry_price)

        # Green book P&L
        if gb_ok:
            green_book_pnl = n_contracts * (gb_price - entry_price)
            actual_pnl = green_book_pnl
        else:
            green_book_pnl = 0.0
            actual_pnl = resolution_pnl

        # Time held
        hold_seconds = price_data[-1][0] - price_data[0][0]
        hold_hours = hold_seconds / 3600.0

        trade = TradeResult(
            game_id=row["game_id"],
            market_id=token_id,
            sport=sport,
            question=question[:80],
            game_date=row["game_date"] or "",
            home_team=home_team,
            away_team=away_team,
            model_prob=model_prob,
            market_price=entry_price,
            edge=edge,
            position_size=actual_cost,
            n_contracts=n_contracts,
            won_resolution=won == 1,
            resolution_pnl=resolution_pnl,
            green_booked=gb_ok,
            green_book_price=gb_price if gb_ok else 0.0,
            green_book_pnl=green_book_pnl,
            max_price_during=max_p,
            min_price_during=min_p,
            actual_pnl=actual_pnl,
            hold_hours=hold_hours,
            n_prices=len(price_data),
        )
        trades.append(trade)
        capital += actual_pnl

    db.close()

    # Compute metrics
    metrics = compute_metrics(trades, params)
    metrics["skipped"] = dict(skipped)

    if verbose:
        print_report(trades, metrics, params)

    return {
        "trades": trades,
        "metrics": metrics,
        "params": params,
    }


def _is_game_market(question: str) -> bool:
    """Check if a market question is about a specific game (not futures/props)."""
    q = question.lower()
    # Game indicators
    game_keywords = [
        "will the", "vs.", "vs ", "beat", "win game",
        "series winner", "game 1", "game 2", "game 3", "game 4",
        "game 5", "game 6", "game 7", "conference finals",
        "nba finals", "premier league",
    ]
    # Non-game indicators
    non_game = [
        "mvp", "draft", "retire", "coach", "trade", "sign",
        "bitcoin", "crypto", "price", "election", "president",
        "ban", "list", "coinbase", "ethereum", "token",
    ]
    if any(kw in q for kw in non_game):
        return False
    return any(kw in q for kw in game_keywords)


# ── Metrics Computation ──────────────────────────────────────────────

def compute_metrics(
    trades: list[TradeResult],
    params: BacktestParams,
) -> dict[str, Any]:
    """Compute backtest performance metrics."""
    if not trades:
        return {"n_trades": 0, "total_pnl": 0, "error": "no trades"}

    n = len(trades)
    pnls = [t.actual_pnl for t in trades]
    total_pnl = sum(pnls)

    # Win rate
    wins = sum(1 for t in trades if t.actual_pnl > 0)
    win_rate = wins / n

    # Green book rate
    gb_count = sum(1 for t in trades if t.green_booked)
    gb_rate = gb_count / n

    # Would-have-been green book (price touched target but we didn't GB)
    could_gb = sum(
        1 for t in trades
        if t.max_price_during >= t.market_price + params.green_book_delta
    )
    could_gb_rate = could_gb / n

    # Average edge
    avg_edge = sum(t.edge for t in trades) / n

    # Max drawdown
    cumulative = 0.0
    peak = 0.0
    max_dd = 0.0
    for pnl in pnls:
        cumulative += pnl
        peak = max(peak, cumulative)
        dd = (peak - cumulative) / max(peak, params.initial_capital)
        max_dd = max(max_dd, dd)

    # Sharpe (daily approximation)
    if len(pnls) > 1:
        mean_pnl = total_pnl / n
        std_pnl = (sum((p - mean_pnl) ** 2 for p in pnls) / (n - 1)) ** 0.5
        sharpe = (mean_pnl / std_pnl * (252 ** 0.5)) if std_pnl > 0 else 0.0
    else:
        sharpe = 0.0

    # Capital turnover
    total_invested = sum(t.position_size for t in trades)
    turnover = total_invested / params.initial_capital

    # Average hold time
    avg_hold = sum(t.hold_hours for t in trades) / n

    # Profit factor
    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # ROI
    roi = total_pnl / params.initial_capital * 100

    # Composite score (Strategy D formula)
    pnl_factor = total_pnl / params.initial_capital * 100
    sharpe_factor = min(max(sharpe, 0) / 3.0, 2.0)
    trade_factor = min(n / 50, 2.0)
    dd_factor = max(0, 1.0 - max_dd * 2)
    turnover_factor = min(turnover / 5.0, 1.5)
    gb_factor = 1.0 + gb_rate * 0.5

    score = (
        pnl_factor
        * (1 + sharpe_factor)
        * trade_factor
        * dd_factor
        * turnover_factor
        * gb_factor
    )

    return {
        "n_trades": n,
        "total_pnl": round(total_pnl, 2),
        "roi_pct": round(roi, 2),
        "win_rate": round(win_rate, 4),
        "green_book_rate": round(gb_rate, 4),
        "could_gb_rate": round(could_gb_rate, 4),
        "avg_edge": round(avg_edge, 4),
        "sharpe": round(sharpe, 2),
        "max_drawdown": round(max_dd, 4),
        "profit_factor": round(profit_factor, 2),
        "turnover": round(turnover, 2),
        "avg_hold_hours": round(avg_hold, 1),
        "gross_profit": round(gross_profit, 2),
        "gross_loss": round(gross_loss, 2),
        "composite_score": round(score, 1),
    }


# ── Report Printing ──────────────────────────────────────────────────

def print_report(
    trades: list[TradeResult],
    metrics: dict[str, Any],
    params: BacktestParams,
) -> None:
    """Print a comprehensive backtest report."""
    print()
    print("=" * 65)
    print("  STRATEGY D BACKTEST — Phase 1 (Small Realistic)")
    print("=" * 65)

    # Parameters
    print(f"\n  Parameters:")
    print(f"    min_edge={params.min_edge}  max_edge={params.max_edge}")
    print(f"    min_price={params.min_price}  max_price={params.max_price}")
    print(f"    green_book_delta={params.green_book_delta}")
    print(f"    kelly_fraction={params.kelly_fraction}  kelly_cap={params.kelly_raw_cap}")
    print(f"    initial_capital=${params.initial_capital:.0f}")

    if not trades:
        print("\n  NO TRADES — adjust parameters or add more data")
        if "skipped" in metrics:
            print(f"\n  Skipped reasons:")
            for reason, count in sorted(metrics["skipped"].items(), key=lambda x: -x[1]):
                print(f"    {reason}: {count}")
        return

    # Summary
    print(f"\n  {'─'*50}")
    print(f"  Results:")
    print(f"    Trades:           {metrics['n_trades']}")
    print(f"    Total P&L:        ${metrics['total_pnl']:+.2f}")
    print(f"    ROI:              {metrics['roi_pct']:+.1f}%")
    print(f"    Win Rate:         {metrics['win_rate']:.1%}")
    print(f"    Green Book Rate:  {metrics['green_book_rate']:.1%}")
    print(f"    Could-GB Rate:    {metrics['could_gb_rate']:.1%}")
    print(f"    Avg Edge:         {metrics['avg_edge']:.3f}")
    print(f"    Sharpe:           {metrics['sharpe']:.2f}")
    print(f"    Max Drawdown:     {metrics['max_drawdown']:.1%}")
    print(f"    Profit Factor:    {metrics['profit_factor']:.2f}")
    print(f"    Capital Turnover: {metrics['turnover']:.1f}x")
    print(f"    Avg Hold Time:    {metrics['avg_hold_hours']:.0f}h")
    print(f"    Composite Score:  {metrics['composite_score']:.1f}")

    # Skipped
    if "skipped" in metrics and metrics["skipped"]:
        print(f"\n  Skipped reasons:")
        for reason, count in sorted(metrics["skipped"].items(), key=lambda x: -x[1]):
            print(f"    {reason}: {count}")

    # Per-trade detail
    print(f"\n  {'─'*50}")
    print(f"  Trade Details:")
    print(f"  {'Game':<35s} {'Entry':>5s} {'Model':>5s} {'Edge':>5s} "
          f"{'MaxP':>5s} {'GB':>3s} {'P&L':>7s}")
    print(f"  {'─'*70}")
    for t in trades:
        gb_mark = "YES" if t.green_booked else "no"
        q = t.question[:33]
        print(f"  {q:<35s} {t.market_price:5.2f} {t.model_prob:5.2f} "
              f"{t.edge:5.3f} {t.max_price_during:5.2f} {gb_mark:>3s} "
              f"${t.actual_pnl:+7.2f}")

    # Green book analysis
    print(f"\n  {'─'*50}")
    print(f"  Green Book Analysis:")
    gb_trades = [t for t in trades if t.green_booked]
    non_gb = [t for t in trades if not t.green_booked]

    if gb_trades:
        avg_gb_pnl = sum(t.actual_pnl for t in gb_trades) / len(gb_trades)
        print(f"    Green booked trades: {len(gb_trades)} "
              f"(avg P&L: ${avg_gb_pnl:+.2f})")
    if non_gb:
        avg_held_pnl = sum(t.actual_pnl for t in non_gb) / len(non_gb)
        print(f"    Held to resolution: {len(non_gb)} "
              f"(avg P&L: ${avg_held_pnl:+.2f})")

    # Price movement analysis
    print(f"\n  Price Movement Analysis:")
    for t in trades:
        move_up = t.max_price_during - t.market_price
        move_down = t.market_price - t.min_price_during
        print(f"    {t.question[:40]:<42s} "
              f"up={move_up:+.2f} down={move_down:-.2f} "
              f"range={t.max_price_during - t.min_price_during:.2f}")

    print(f"\n{'='*65}")


# ── CLI ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Strategy D sports backtest engine.",
    )
    parser.add_argument("--min-edge", type=float, default=0.03)
    parser.add_argument("--max-edge", type=float, default=0.30)
    parser.add_argument("--min-price", type=float, default=0.10)
    parser.add_argument("--max-price", type=float, default=0.85)
    parser.add_argument("--green-book-delta", type=float, default=0.05)
    parser.add_argument("--no-green-book", action="store_true")
    parser.add_argument("--kelly-fraction", type=float, default=0.25)
    parser.add_argument("--capital", type=float, default=1000.0)
    parser.add_argument("--min-prices", type=int, default=5)
    parser.add_argument("--db", default=None)
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    params = BacktestParams(
        min_edge=args.min_edge,
        max_edge=args.max_edge,
        min_price=args.min_price,
        max_price=args.max_price,
        green_book_delta=args.green_book_delta,
        green_book_enabled=not args.no_green_book,
        kelly_fraction=args.kelly_fraction,
        initial_capital=args.capital,
        min_prices=args.min_prices,
    )

    result = run_backtest(params, args.db, verbose=not args.json)

    if args.json:
        # Serialize for autoresearch
        output = {
            "metrics": result["metrics"],
            "n_trades": result["metrics"]["n_trades"],
            "score": result["metrics"].get("composite_score", 0),
        }
        print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
