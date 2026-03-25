"""
Early Exit Backtest — Profit-Taking vs Hold-to-Resolution
==========================================================

Analyzes whether closing trades early into a counter-position (selling
YES tokens at profit) beats holding to binary resolution ($0 or $1).

Uses PMD 10-minute price data (7.4M price points) to simulate realistic
price paths and exit timing.

Key questions:
  1. How often does price hit a profit target before resolution?
  2. What's the optimal profit target vs hold-to-resolution?
  3. How much does capital recycling improve total portfolio P&L?
  4. What's the MFE/MAE distribution (max favorable/adverse excursion)?

Usage:
    python3 research/backtest_early_exit.py

Output: Console report + JSON results
"""

from __future__ import annotations

import bisect
import json
import math
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "innovations"))

from pmd_loader import load_pmd_data
from experiment_framework import (
    CITY_COORDS,
    SimulationData,
    compute_prob,
    quality_gate,
    compute_size,
    load_forecasts,
    SLIPPAGE_PCT,
    GAS_COST_USD,
    INITIAL_CAPITAL,
    MIN_TRADE_SIZE,
)

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# Model params (AR-0134 baseline from production)
MODEL_PARAMS = {
    "min_edge": 0.03,
    "min_price": 0.04,
    "max_price": 0.70,
    "min_forecast_prob": 0.04,
    "min_volume": 50,        # PMD volume = price count, lower threshold
    "prob_sharpening": 0.90,
    "base_sigma": 3.0,
    "kelly_fraction": 0.25,
    "kelly_raw_cap": 0.15,
    "max_position_pct": 0.05,
    "max_aggregate_pct": 0.80,
    "city_max_exposure": 0.25,
    "no_compounding": True,
    "shrinkage_prior": 0.125,
    "shrinkage_weight": 0.03,
}

# Entry horizon (hours before market close)
ENTRY_HOURS = [48, 24]

# Exit strategies to test
EXIT_STRATEGIES = {
    "hold": {
        "type": "hold",
    },
    "target_3c": {
        "type": "profit_target",
        "target": 0.03,        # Exit when price rises 3 cents above entry
    },
    "target_5c": {
        "type": "profit_target",
        "target": 0.05,
    },
    "target_8c": {
        "type": "profit_target",
        "target": 0.08,
    },
    "target_10c": {
        "type": "profit_target",
        "target": 0.10,
    },
    "target_15c": {
        "type": "profit_target",
        "target": 0.15,
    },
    "target_20c": {
        "type": "profit_target",
        "target": 0.20,
    },
    "trail_30pct": {
        "type": "trailing_stop",
        "trail_pct": 0.30,     # Exit when price drops 30% from peak since entry
    },
    "trail_50pct": {
        "type": "trailing_stop",
        "trail_pct": 0.50,
    },
    "t5c_tr30": {
        "type": "combo",
        "target": 0.05,
        "trail_pct": 0.30,     # Target + trail: exit at first trigger
    },
    "t10c_tr30": {
        "type": "combo",
        "target": 0.10,
        "trail_pct": 0.30,
    },
    "t5c_tr50": {
        "type": "combo",
        "target": 0.05,
        "trail_pct": 0.50,
    },
}

# Slippage on exit sell (separate from entry — can be different if using limit orders)
EXIT_SLIPPAGE_PCT = SLIPPAGE_PCT  # Default: same as entry (taker)

# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class PricePath:
    """Full price trajectory for a trade from entry to resolution."""
    token_id: str
    event_id: str
    city: str
    entry_ts: int
    close_ts: int
    entry_price: float        # Mid price at entry time
    entry_fill: float         # Actual fill (entry_price * (1 + slippage))
    won: bool                 # Did this bucket win at resolution?
    prices: list[tuple[int, float]]  # All (ts, price) from entry to close
    # Pre-computed path statistics
    mfe: float = 0.0          # Max Favorable Excursion (highest price - entry)
    mae: float = 0.0          # Max Adverse Excursion (entry - lowest price)
    mfe_time_pct: float = 0.0 # % of hold time when MFE occurs
    hold_hours: float = 0.0


@dataclass
class ExitResult:
    """Result of applying an exit strategy to a single trade."""
    strategy: str
    exited_early: bool
    exit_ts: int | None = None
    exit_price: float | None = None
    pnl: float = 0.0
    hold_hours: float = 0.0       # Actual time held (to exit or resolution)
    resolution_pnl: float = 0.0   # Counterfactual: what if held to resolution


@dataclass
class StrategyMetrics:
    """Aggregate metrics for one exit strategy."""
    name: str
    n_trades: int = 0
    n_exited_early: int = 0
    total_pnl: float = 0.0
    total_size: float = 0.0
    wins: int = 0
    losses: int = 0
    avg_hold_hours: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe: float = 0.0
    # Comparison to hold
    resolution_total_pnl: float = 0.0  # What hold-to-resolution would have made
    saves: int = 0             # Times exit beat hold
    regrets: int = 0           # Times hold would have been better
    avg_save_amount: float = 0.0
    avg_regret_amount: float = 0.0
    # Capital recycling metrics
    capital_turns: float = 0.0  # How many times capital was recycled
    pnl_per_dollar_day: float = 0.0  # Capital efficiency


# ═══════════════════════════════════════════════════════════════════════════════
# PRICE PATH EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════


def extract_price_paths(
    sim_data: SimulationData,
    params: dict,
    entry_hours_list: list[float],
) -> list[PricePath]:
    """Extract price paths for all trades the model would enter.

    For each entry opportunity (event × entry_hour), checks if the model
    would trade it (using quality gate + edge), then extracts the full
    10-minute price path from entry to resolution.
    """
    paths: list[PricePath] = []
    seen_entries: set[str] = set()  # Deduplicate: one entry per (event, bucket, hours)

    for entry_hours in entry_hours_list:
        entry_index = sim_data.build_entry_index(entry_hours)
        days_out = max(0, int(entry_hours / 24))

        for entry_ts, entries in sorted(entry_index.items()):
            for entry_item in entries:
                if isinstance(entry_item, tuple):
                    ev, d_out = entry_item
                else:
                    ev = entry_item
                    d_out = days_out

                if not ev.city or ev.city not in CITY_COORDS:
                    continue

                close_ts = sim_data.get_close_ts(ev.event_id)
                if close_ts is None or close_ts <= entry_ts:
                    continue

                forecast_temp = sim_data.forecasts.get(ev.city, {}).get(
                    ev.target_date
                )
                if forecast_temp is None:
                    continue

                buckets = sim_data.buckets_by_event.get(ev.event_id, [])

                for bucket in buckets:
                    dedup_key = f"{ev.event_id}:{bucket.token_id}:{int(entry_hours)}"
                    if dedup_key in seen_entries:
                        continue

                    price = sim_data.get_price(bucket.token_id, entry_ts)
                    if price is None or price <= 0.001:
                        continue

                    our_prob = compute_prob(
                        forecast_temp, bucket, d_out, ev.city, params
                    )
                    edge = our_prob - price

                    if not quality_gate(
                        edge, our_prob, price, bucket.volume, ev.city, params
                    ):
                        continue

                    seen_entries.add(dedup_key)

                    # Extract full price path from entry to close
                    raw_prices = sim_data._prices.get(bucket.token_id, [])
                    if not raw_prices:
                        continue

                    # Get prices between entry_ts and close_ts
                    start_idx = bisect.bisect_left(raw_prices, (entry_ts, -1.0))
                    end_idx = bisect.bisect_right(raw_prices, (close_ts, float("inf")))

                    path_prices = raw_prices[start_idx:end_idx]
                    if len(path_prices) < 2:
                        continue

                    entry_fill = price * (1 + SLIPPAGE_PCT)

                    # Compute MFE/MAE
                    max_price = max(p for _, p in path_prices)
                    min_price = min(p for _, p in path_prices)
                    mfe = max_price - price
                    mae = price - min_price

                    # MFE timing (what % of hold time)
                    mfe_ts = next(ts for ts, p in path_prices if p == max_price)
                    total_duration = close_ts - entry_ts
                    mfe_time_pct = (
                        (mfe_ts - entry_ts) / total_duration * 100
                        if total_duration > 0 else 0
                    )

                    hold_hours = total_duration / 3600

                    path = PricePath(
                        token_id=bucket.token_id,
                        event_id=ev.event_id,
                        city=ev.city,
                        entry_ts=entry_ts,
                        close_ts=close_ts,
                        entry_price=price,
                        entry_fill=entry_fill,
                        won=bucket.won,
                        prices=path_prices,
                        mfe=mfe,
                        mae=mae,
                        mfe_time_pct=mfe_time_pct,
                        hold_hours=hold_hours,
                    )
                    paths.append(path)

    return paths


# ═══════════════════════════════════════════════════════════════════════════════
# EXIT STRATEGY SIMULATION
# ═══════════════════════════════════════════════════════════════════════════════


def apply_exit_strategy(
    path: PricePath,
    strategy_name: str,
    strategy_config: dict,
) -> ExitResult:
    """Apply an exit strategy to a price path and compute P&L.

    Returns ExitResult with actual P&L and counterfactual (hold) P&L.
    """
    entry_fill = path.entry_fill
    tokens = 1.0 / entry_fill  # Shares per $1 invested

    # Resolution P&L (hold to end)
    if path.won:
        resolution_pnl = (1.0 / entry_fill - 1.0) - GAS_COST_USD
    else:
        resolution_pnl = -1.0 - GAS_COST_USD

    strategy_type = strategy_config.get("type", "hold")

    if strategy_type == "hold":
        return ExitResult(
            strategy=strategy_name,
            exited_early=False,
            pnl=resolution_pnl,
            hold_hours=path.hold_hours,
            resolution_pnl=resolution_pnl,
        )

    # Walk through price path
    peak_price = path.entry_price
    target = strategy_config.get("target", 999)
    trail_pct = strategy_config.get("trail_pct", 1.0)

    for ts, price in path.prices:
        peak_price = max(peak_price, price)

        # Check profit target
        if strategy_type in ("profit_target", "combo"):
            if price >= path.entry_price + target:
                exit_fill = price * (1 - EXIT_SLIPPAGE_PCT)
                pnl = tokens * exit_fill - 1.0 - 2 * GAS_COST_USD  # Double gas
                hours_held = (ts - path.entry_ts) / 3600
                return ExitResult(
                    strategy=strategy_name,
                    exited_early=True,
                    exit_ts=ts,
                    exit_price=price,
                    pnl=pnl,
                    hold_hours=hours_held,
                    resolution_pnl=resolution_pnl,
                )

        # Check trailing stop (only after some profit to avoid instant trigger)
        if strategy_type in ("trailing_stop", "combo"):
            if peak_price > path.entry_price and price > path.entry_price:
                gain_from_entry = peak_price - path.entry_price
                pullback = peak_price - price
                if gain_from_entry > 0.01 and pullback / gain_from_entry >= trail_pct:
                    exit_fill = price * (1 - EXIT_SLIPPAGE_PCT)
                    pnl = tokens * exit_fill - 1.0 - 2 * GAS_COST_USD
                    hours_held = (ts - path.entry_ts) / 3600
                    return ExitResult(
                        strategy=strategy_name,
                        exited_early=True,
                        exit_ts=ts,
                        exit_price=price,
                        pnl=pnl,
                        hold_hours=hours_held,
                        resolution_pnl=resolution_pnl,
                    )

    # Never triggered — hold to resolution
    return ExitResult(
        strategy=strategy_name,
        exited_early=False,
        pnl=resolution_pnl,
        hold_hours=path.hold_hours,
        resolution_pnl=resolution_pnl,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PORTFOLIO SIMULATION WITH CAPITAL RECYCLING
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class PortfolioResult:
    """Result of running a strategy with capital recycling."""
    strategy: str
    total_pnl: float = 0.0
    n_trades: int = 0
    n_exited_early: int = 0
    n_skipped_capital: int = 0   # Trades skipped due to capital constraints
    wins: int = 0
    avg_hold_hours: float = 0.0
    sharpe: float = 0.0
    max_drawdown_pct: float = 0.0
    capital_turns: float = 0.0
    final_capital: float = 0.0
    pnl_per_dollar_day: float = 0.0
    # Comparison
    saves: int = 0
    regrets: int = 0


def simulate_portfolio_with_recycling(
    paths: list[PricePath],
    strategy_name: str,
    strategy_config: dict,
    initial_capital: float = INITIAL_CAPITAL,
) -> PortfolioResult:
    """Simulate portfolio with capital recycling.

    Key difference from hold: when a trade exits early, the freed capital
    + profit can immediately be deployed into the next available trade.
    """
    # Sort paths by entry time
    sorted_paths = sorted(paths, key=lambda p: p.entry_ts)

    capital = initial_capital
    deployed: dict[str, tuple[PricePath, float, int]]  = {}  # token_id -> (path, size, expected_free_ts)
    trades_executed: list[ExitResult] = []
    skipped_capital = 0
    total_deployed = 0.0

    # Process events chronologically
    # Build a timeline of events: entries and exits
    events_timeline: list[tuple[int, str, object]] = []  # (ts, type, data)

    for path in sorted_paths:
        events_timeline.append((path.entry_ts, "entry", path))

    events_timeline.sort(key=lambda x: x[0])

    # Pre-compute exit times for each path
    exit_cache: dict[str, ExitResult] = {}
    for path in sorted_paths:
        result = apply_exit_strategy(path, strategy_name, strategy_config)
        exit_cache[path.token_id] = result

    # Track equity curve for drawdown
    equity_curve = [initial_capital]
    peak = initial_capital
    max_dd = 0.0

    for ts, event_type, data in events_timeline:
        path = data

        # Free up completed positions
        for token_id in list(deployed.keys()):
            dep_path, dep_size, free_ts = deployed[token_id]
            if ts >= free_ts:
                result = exit_cache[token_id]
                pnl = result.pnl * dep_size  # Scale by position size
                capital += dep_size + pnl
                trades_executed.append(result)

                equity = capital + sum(s for _, s, _ in deployed.values() if _ > ts)
                equity_curve.append(equity)
                peak = max(peak, equity)
                dd = (peak - equity) / peak * 100 if peak > 0 else 0
                max_dd = max(max_dd, dd)

                del deployed[token_id]

        # Try to enter new position
        if event_type == "entry":
            # Skip if already in this token
            if path.token_id in deployed:
                continue

            # Position sizing (simplified: fixed fraction)
            sizing_capital = min(capital, initial_capital)  # no_compounding
            max_size = sizing_capital * MODEL_PARAMS["max_position_pct"]
            size = min(max_size, capital * 0.1)  # Conservative: max 10% of available

            if size < MIN_TRADE_SIZE or capital < size:
                skipped_capital += 1
                continue

            # Determine when capital is freed
            result = exit_cache[path.token_id]
            if result.exited_early and result.exit_ts is not None:
                free_ts = result.exit_ts
            else:
                free_ts = path.close_ts

            deployed[path.token_id] = (path, size, free_ts)
            capital -= size
            total_deployed += size

    # Close remaining positions
    for token_id in list(deployed.keys()):
        dep_path, dep_size, free_ts = deployed[token_id]
        result = exit_cache[token_id]
        pnl = result.pnl * dep_size
        capital += dep_size + pnl
        trades_executed.append(result)
        del deployed[token_id]

    # Compute metrics
    n_trades = len(trades_executed)
    if n_trades == 0:
        return PortfolioResult(strategy=strategy_name, final_capital=initial_capital)

    pnls = [r.pnl for r in trades_executed]
    total_pnl = capital - initial_capital
    wins = sum(1 for p in pnls if p > 0)
    n_exited = sum(1 for r in trades_executed if r.exited_early)
    avg_hold = sum(r.hold_hours for r in trades_executed) / n_trades
    hold_hours_total = sum(r.hold_hours for r in trades_executed)

    # Sharpe
    mean_pnl = sum(pnls) / n_trades
    if n_trades > 1:
        var = sum((p - mean_pnl) ** 2 for p in pnls) / (n_trades - 1)
        std = math.sqrt(var) if var > 0 else 1e-6
        sharpe = (mean_pnl / std) * math.sqrt(252)
    else:
        sharpe = 0.0

    # Capital turns
    capital_turns = total_deployed / initial_capital if initial_capital > 0 else 0

    # P&L per dollar-day
    dollar_days = hold_hours_total / 24  # Rough proxy
    pnl_per_dd = total_pnl / dollar_days if dollar_days > 0 else 0

    # Saves vs regrets
    saves = sum(1 for r in trades_executed if r.exited_early and r.pnl > r.resolution_pnl)
    regrets = sum(1 for r in trades_executed if r.exited_early and r.pnl < r.resolution_pnl)

    return PortfolioResult(
        strategy=strategy_name,
        total_pnl=round(total_pnl, 2),
        n_trades=n_trades,
        n_exited_early=n_exited,
        n_skipped_capital=skipped_capital,
        wins=wins,
        avg_hold_hours=round(avg_hold, 1),
        sharpe=round(sharpe, 4),
        max_drawdown_pct=round(max_dd, 2),
        capital_turns=round(capital_turns, 2),
        final_capital=round(capital, 2),
        pnl_per_dollar_day=round(pnl_per_dd, 4),
        saves=saves,
        regrets=regrets,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS & REPORTING
# ═══════════════════════════════════════════════════════════════════════════════


def analyze_price_paths(paths: list[PricePath]) -> dict:
    """Analyze MFE/MAE distributions across all price paths."""
    if not paths:
        return {}

    won_paths = [p for p in paths if p.won]
    lost_paths = [p for p in paths if not p.won]

    def percentiles(values: list[float]) -> dict:
        if not values:
            return {"p10": 0, "p25": 0, "p50": 0, "p75": 0, "p90": 0, "mean": 0}
        s = sorted(values)
        n = len(s)
        return {
            "p10": round(s[int(n * 0.10)], 4),
            "p25": round(s[int(n * 0.25)], 4),
            "p50": round(s[int(n * 0.50)], 4),
            "p75": round(s[int(n * 0.75)], 4),
            "p90": round(s[int(n * 0.90)], 4),
            "mean": round(sum(s) / n, 4),
        }

    # MFE/MAE for winning trades
    won_mfe = percentiles([p.mfe for p in won_paths])
    won_mae = percentiles([p.mae for p in won_paths])

    # MFE/MAE for losing trades
    lost_mfe = percentiles([p.mfe for p in lost_paths])
    lost_mae = percentiles([p.mae for p in lost_paths])

    # When does MFE occur? (early = good for profit-taking)
    won_mfe_timing = percentiles([p.mfe_time_pct for p in won_paths])
    lost_mfe_timing = percentiles([p.mfe_time_pct for p in lost_paths])

    # How often does price reach various targets?
    targets = [0.03, 0.05, 0.08, 0.10, 0.15, 0.20]
    hit_rates = {}
    for target in targets:
        won_hits = sum(1 for p in won_paths if p.mfe >= target)
        lost_hits = sum(1 for p in lost_paths if p.mfe >= target)
        all_hits = won_hits + lost_hits
        hit_rates[f"+{target:.2f}"] = {
            "all": round(all_hits / len(paths) * 100, 1) if paths else 0,
            "winners": round(won_hits / len(won_paths) * 100, 1) if won_paths else 0,
            "losers": round(lost_hits / len(lost_paths) * 100, 1) if lost_paths else 0,
        }

    return {
        "total_paths": len(paths),
        "won": len(won_paths),
        "lost": len(lost_paths),
        "win_rate_pct": round(len(won_paths) / len(paths) * 100, 1),
        "avg_hold_hours": round(sum(p.hold_hours for p in paths) / len(paths), 1),
        "avg_entry_price": round(sum(p.entry_price for p in paths) / len(paths), 4),
        "mfe": {
            "winners": won_mfe,
            "losers": lost_mfe,
        },
        "mae": {
            "winners": won_mae,
            "losers": lost_mae,
        },
        "mfe_timing_pct": {
            "winners": won_mfe_timing,
            "losers": lost_mfe_timing,
        },
        "target_hit_rates": hit_rates,
    }


def analyze_per_trade(
    paths: list[PricePath],
    strategies: dict[str, dict],
) -> dict[str, list[ExitResult]]:
    """Run all exit strategies on all paths (without capital constraints)."""
    results: dict[str, list[ExitResult]] = {name: [] for name in strategies}

    for path in paths:
        for name, config in strategies.items():
            result = apply_exit_strategy(path, name, config)
            results[name].append(result)

    return results


def compute_strategy_metrics(
    name: str,
    results: list[ExitResult],
) -> StrategyMetrics:
    """Compute aggregate metrics for one strategy (per-trade, no capital constraints)."""
    n = len(results)
    if n == 0:
        return StrategyMetrics(name=name)

    n_exited = sum(1 for r in results if r.exited_early)
    pnls = [r.pnl for r in results]
    total_pnl = sum(pnls)
    wins = sum(1 for p in pnls if p > 0)
    avg_hold = sum(r.hold_hours for r in results) / n

    # Sharpe
    mean_pnl = total_pnl / n
    if n > 1:
        var = sum((p - mean_pnl) ** 2 for p in pnls) / (n - 1)
        std = math.sqrt(var) if var > 0 else 1e-6
        sharpe = (mean_pnl / std) * math.sqrt(252)
    else:
        sharpe = 0.0

    # Saves vs regrets
    exited = [r for r in results if r.exited_early]
    saves = sum(1 for r in exited if r.pnl > r.resolution_pnl)
    regrets = sum(1 for r in exited if r.pnl < r.resolution_pnl)
    save_amounts = [r.pnl - r.resolution_pnl for r in exited if r.pnl > r.resolution_pnl]
    regret_amounts = [r.resolution_pnl - r.pnl for r in exited if r.pnl < r.resolution_pnl]

    resolution_pnl = sum(r.resolution_pnl for r in results)

    return StrategyMetrics(
        name=name,
        n_trades=n,
        n_exited_early=n_exited,
        total_pnl=round(total_pnl, 4),
        wins=wins,
        avg_hold_hours=round(avg_hold, 1),
        sharpe=round(sharpe, 4),
        resolution_total_pnl=round(resolution_pnl, 4),
        saves=saves,
        regrets=regrets,
        avg_save_amount=round(sum(save_amounts) / len(save_amounts), 4) if save_amounts else 0,
        avg_regret_amount=round(sum(regret_amounts) / len(regret_amounts), 4) if regret_amounts else 0,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    t0 = time.time()

    # ── Load data ──
    print("Loading PMD data...")
    events, buckets_by_event, prices = load_pmd_data()

    print("Loading forecasts...")
    forecasts = load_forecasts(events, CITY_COORDS)

    sim_data = SimulationData(events, buckets_by_event, prices, forecasts)

    # ── Extract price paths ──
    print(f"\nExtracting price paths (entry hours: {ENTRY_HOURS})...")
    paths = extract_price_paths(sim_data, MODEL_PARAMS, ENTRY_HOURS)
    print(f"Found {len(paths)} valid trade entries with price paths")

    if not paths:
        print("ERROR: No price paths found. Check model params / data.")
        return

    # ── Phase 1: Price Path Analysis ──
    print("\n" + "=" * 90)
    print("PHASE 1: PRICE PATH ANALYSIS (MFE / MAE)")
    print("=" * 90)

    path_analysis = analyze_price_paths(paths)

    print(f"\n  Total trades: {path_analysis['total_paths']}")
    print(f"  Win rate: {path_analysis['win_rate_pct']}%")
    print(f"  Avg hold: {path_analysis['avg_hold_hours']}h")
    print(f"  Avg entry price: ${path_analysis['avg_entry_price']:.4f}")

    print(f"\n  Max Favorable Excursion (how high does price go above entry?):")
    print(f"    Winners:  median +${path_analysis['mfe']['winners']['p50']:.4f}  "
          f"mean +${path_analysis['mfe']['winners']['mean']:.4f}  "
          f"p90 +${path_analysis['mfe']['winners']['p90']:.4f}")
    print(f"    Losers:   median +${path_analysis['mfe']['losers']['p50']:.4f}  "
          f"mean +${path_analysis['mfe']['losers']['mean']:.4f}  "
          f"p90 +${path_analysis['mfe']['losers']['p90']:.4f}")

    print(f"\n  Max Adverse Excursion (how low does price drop below entry?):")
    print(f"    Winners:  median -${path_analysis['mae']['winners']['p50']:.4f}  "
          f"mean -${path_analysis['mae']['winners']['mean']:.4f}")
    print(f"    Losers:   median -${path_analysis['mae']['losers']['p50']:.4f}  "
          f"mean -${path_analysis['mae']['losers']['mean']:.4f}")

    print(f"\n  MFE timing (when does peak price occur, % of hold time):")
    print(f"    Winners:  median {path_analysis['mfe_timing_pct']['winners']['p50']:.0f}%  "
          f"(earlier = better for profit-taking)")
    print(f"    Losers:   median {path_analysis['mfe_timing_pct']['losers']['p50']:.0f}%")

    print(f"\n  Target hit rates (% of trades where price reaches target):")
    print(f"    {'Target':<10} {'All':>8} {'Winners':>10} {'Losers':>10}")
    print(f"    {'-'*38}")
    for target, rates in path_analysis["target_hit_rates"].items():
        print(f"    {target:<10} {rates['all']:>7.1f}% {rates['winners']:>9.1f}% {rates['losers']:>9.1f}%")

    # ── Phase 2: Strategy Comparison (per-trade, no capital constraint) ──
    print("\n" + "=" * 90)
    print("PHASE 2: EXIT STRATEGY COMPARISON (per-$1 invested, no capital constraint)")
    print("=" * 90)

    all_results = analyze_per_trade(paths, EXIT_STRATEGIES)
    strategy_metrics: dict[str, StrategyMetrics] = {}

    for name, results in all_results.items():
        metrics = compute_strategy_metrics(name, results)
        strategy_metrics[name] = metrics

    # Sort by total P&L
    sorted_strategies = sorted(
        strategy_metrics.values(),
        key=lambda m: -m.total_pnl,
    )

    print(f"\n  {'Strategy':<14} {'PnL/trade':>10} {'WinRate':>8} {'ExitRate':>9} "
          f"{'AvgHold':>8} {'Sharpe':>8} {'Saves':>6} {'Regrets':>8}")
    print(f"  {'-'*78}")

    hold_pnl = strategy_metrics["hold"].total_pnl / strategy_metrics["hold"].n_trades

    for m in sorted_strategies:
        pnl_per_trade = m.total_pnl / m.n_trades if m.n_trades > 0 else 0
        wr = m.wins / m.n_trades * 100 if m.n_trades > 0 else 0
        exit_rate = m.n_exited_early / m.n_trades * 100 if m.n_trades > 0 else 0
        delta = pnl_per_trade - hold_pnl

        marker = ">>>" if pnl_per_trade > hold_pnl and m.name != "hold" else "   "
        print(f"{marker}{m.name:<14} ${pnl_per_trade:>8.4f} {wr:>7.1f}% {exit_rate:>8.1f}% "
              f"{m.avg_hold_hours:>7.1f}h {m.sharpe:>7.2f} {m.saves:>6} {m.regrets:>7}  "
              f"{'(' + '+' if delta >= 0 else '('}{delta:.4f} vs hold)")

    # ── Phase 3: Portfolio Simulation with Capital Recycling ──
    print("\n" + "=" * 90)
    print("PHASE 3: PORTFOLIO WITH CAPITAL RECYCLING ($1000 initial)")
    print("=" * 90)

    portfolio_results: list[PortfolioResult] = []
    for name, config in EXIT_STRATEGIES.items():
        pr = simulate_portfolio_with_recycling(paths, name, config)
        portfolio_results.append(pr)

    portfolio_results.sort(key=lambda r: -r.total_pnl)

    print(f"\n  {'Strategy':<14} {'Total PnL':>10} {'Trades':>7} {'Exited':>7} "
          f"{'AvgHold':>8} {'CapTurns':>9} {'$/day':>8} {'Saves':>6} {'Regrets':>8}")
    print(f"  {'-'*85}")

    hold_portfolio_pnl = next(
        r.total_pnl for r in portfolio_results if r.strategy == "hold"
    )

    for r in portfolio_results:
        delta = r.total_pnl - hold_portfolio_pnl
        marker = ">>>" if r.total_pnl > hold_portfolio_pnl and r.strategy != "hold" else "   "
        wr = r.wins / r.n_trades * 100 if r.n_trades > 0 else 0
        print(f"{marker}{r.strategy:<14} ${r.total_pnl:>9.2f} {r.n_trades:>7} "
              f"{r.n_exited_early:>7} {r.avg_hold_hours:>7.1f}h {r.capital_turns:>8.1f}x "
              f"${r.pnl_per_dollar_day:>7.4f} {r.saves:>6} {r.regrets:>7}  "
              f"({'+' if delta >= 0 else ''}{delta:.2f} vs hold)")

    # ── Summary ──
    best_per_trade = sorted_strategies[0]
    best_portfolio = portfolio_results[0]

    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    ppt_hold = strategy_metrics["hold"].total_pnl / strategy_metrics["hold"].n_trades
    ppt_best = best_per_trade.total_pnl / best_per_trade.n_trades if best_per_trade.n_trades else 0
    print(f"  Best per-trade: {best_per_trade.name} "
          f"(${ppt_best:.4f}/trade vs hold ${ppt_hold:.4f})")
    print(f"  Best portfolio: {best_portfolio.strategy} "
          f"(${best_portfolio.total_pnl:.2f} vs hold ${hold_portfolio_pnl:.2f})")
    print(f"  Capital recycling benefit: "
          f"{best_portfolio.capital_turns:.1f}x turns vs hold")

    # ── Save results ──
    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "model_params": MODEL_PARAMS,
            "entry_hours": ENTRY_HOURS,
            "exit_slippage_pct": EXIT_SLIPPAGE_PCT,
            "initial_capital": INITIAL_CAPITAL,
        },
        "path_analysis": path_analysis,
        "per_trade_metrics": {
            name: {
                "pnl_per_trade": round(m.total_pnl / m.n_trades, 6) if m.n_trades else 0,
                "win_rate": round(m.wins / m.n_trades * 100, 1) if m.n_trades else 0,
                "exit_rate": round(m.n_exited_early / m.n_trades * 100, 1) if m.n_trades else 0,
                "avg_hold_hours": m.avg_hold_hours,
                "sharpe": m.sharpe,
                "saves": m.saves,
                "regrets": m.regrets,
            }
            for name, m in strategy_metrics.items()
        },
        "portfolio_results": {
            r.strategy: asdict(r) for r in portfolio_results
        },
    }

    results_path = Path(__file__).parent / "data" / "early_exit_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {results_path}")

    elapsed = time.time() - t0
    print(f"  Runtime: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
