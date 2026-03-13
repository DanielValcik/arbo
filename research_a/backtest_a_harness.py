#!/usr/bin/env python3
"""
Strategy A (Theta Decay) — Backtest Harness
============================================

FIXED FILE — do NOT modify during autoresearch.
Only strategy_a_experiment.py gets modified.

Simulates a universe of longshot prediction markets with realistic:
- Optimism bias (longshots overpriced by 2-8pp, per Snowberg & Wolfers 2010)
- Taker flow with periodic 3σ+ spikes (news/social hype events)
- Price dynamics (mean reversion after spikes + random walk)
- Resolution outcomes (Bernoulli based on true probability)

Walk-forward validation with composite scoring.

Usage: python3 research_a/backtest_a_harness.py
"""

import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

# Import experiment (mutable — the agent edits this)
sys.path.insert(0, str(Path(__file__).parent))
import strategy_a_experiment as strategy


# ═══════════════════════════════════════════════════════════════════════════════
# FIXED CONSTANTS — DO NOT CHANGE
# ═══════════════════════════════════════════════════════════════════════════════

INITIAL_CAPITAL = 400.0             # Strategy A allocation (€400)
BASE_SEED = 73                      # Different from C (42) and B (137)
TICKS_PER_DAY = 6                   # 4-hour ticks

# Walk-forward: 5 windows, 90 days each, different random universes
WINDOWS = [
    {"label": "W1", "n_days": 90, "seed_offset": 0},
    {"label": "W2", "n_days": 90, "seed_offset": 10000},
    {"label": "W3", "n_days": 90, "seed_offset": 20000},
    {"label": "W4", "n_days": 90, "seed_offset": 30000},
    {"label": "W5", "n_days": 90, "seed_offset": 40000},
]


# ═══════════════════════════════════════════════════════════════════════════════
# MARKET SIMULATION MODEL (FIXED — calibrated to academic research)
# ═══════════════════════════════════════════════════════════════════════════════

# True YES probability: Beta(1.5, 20) → mode ~2.4%, mean ~7%
# This matches observed longshot bias in prediction markets
# (Snowberg & Wolfers 2010, Moskowitz 2021)
SIM_TRUE_PROB_ALPHA = 1.5
SIM_TRUE_PROB_BETA = 20.0

# Optimism bias: market overprices longshots by this amount
SIM_BIAS_MIN = 0.02                 # Minimum bias (2pp)
SIM_BIAS_MAX = 0.08                 # Maximum bias (8pp)

# Market generation
SIM_NEW_MARKETS_PER_DAY = 4.0       # Poisson rate of new longshot markets
SIM_VOLUME_LOG_MEAN = 10.0          # LogNormal mean → median ~$22K
SIM_VOLUME_LOG_STD = 1.2            # LogNormal std → wide range
SIM_DAYS_TO_RES_MIN = 5             # Min days to resolution
SIM_DAYS_TO_RES_MAX = 45            # Max days to resolution

# Taker flow simulation
SIM_SPIKE_RATE = 0.008              # P(spike) per tick (~1 spike per 5 days per market)
SIM_SPIKE_MAG_MIN = 3.5             # Min spike magnitude (σ)
SIM_SPIKE_MAG_MAX = 7.0             # Max spike magnitude (σ)

# Price dynamics
SIM_SPIKE_PRICE_IMPACT = 0.006      # YES price impact per σ during spike
SIM_NORMAL_PRICE_IMPACT = 0.001     # YES price impact per σ normal flow
SIM_REVERSION_RATE = 0.06           # Mean reversion per tick toward equilibrium
SIM_PRICE_NOISE_STD = 0.002         # Random walk per tick

# Slippage & fees
SIM_BASE_SLIPPAGE = 0.005           # 0.5% base slippage on NO entry
SIM_THIN_SLIPPAGE = 0.010           # +1% for thin markets
SIM_THIN_VOLUME = 5000.0            # Volume threshold for thin market
SIM_FEE_RATE = 0.0                  # Most longshot markets are fee-free

# Capital limits
MAX_SIZING_CAPITAL = 1000.0         # Cap to prevent unrealistic compounding


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SimMarket:
    id: int
    true_prob: float
    equilibrium_yes: float           # Fair YES price = true_prob + bias
    current_yes: float
    volume_24h: float
    start_day: int
    resolution_day: int
    resolves_yes: bool
    is_resolved: bool = False


@dataclass
class Position:
    market_id: int
    entry_price_no: float
    initial_size: float
    initial_shares: float            # initial_size / entry_price_no
    remaining_shares: float = 0.0
    entry_tick: int = 0
    partial_exited: bool = False
    peak_pnl_pct: float = 0.0
    cash_received: float = 0.0

    def __post_init__(self):
        if self.remaining_shares == 0.0:
            self.remaining_shares = self.initial_shares


@dataclass
class TradeResult:
    market_id: int
    pnl: float
    pnl_pct: float
    won: bool
    exit_reason: str


@dataclass
class WindowResult:
    sharpe: float
    win_rate: float
    max_drawdown: float
    num_trades: int
    total_pnl: float
    pnl_pct: float
    profit_factor: float


# ═══════════════════════════════════════════════════════════════════════════════
# MARKET GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def generate_markets(rng: np.random.Generator, n_days: int) -> list[SimMarket]:
    """Generate simulated longshot market universe."""
    markets = []
    mid = 0

    for day in range(n_days):
        n_new = rng.poisson(SIM_NEW_MARKETS_PER_DAY)
        for _ in range(n_new):
            true_prob = float(rng.beta(SIM_TRUE_PROB_ALPHA, SIM_TRUE_PROB_BETA))
            true_prob = max(0.01, min(true_prob, 0.25))

            bias = float(rng.uniform(SIM_BIAS_MIN, SIM_BIAS_MAX))
            eq_yes = min(true_prob + bias, 0.30)

            vol = float(rng.lognormal(SIM_VOLUME_LOG_MEAN, SIM_VOLUME_LOG_STD))
            vol = max(500, vol)

            days_to_res = int(rng.integers(SIM_DAYS_TO_RES_MIN, SIM_DAYS_TO_RES_MAX + 1))
            resolves_yes = float(rng.random()) < true_prob

            init_yes = eq_yes + float(rng.normal(0, 0.008))
            init_yes = max(0.01, min(init_yes, 0.30))

            markets.append(SimMarket(
                id=mid,
                true_prob=true_prob,
                equilibrium_yes=eq_yes,
                current_yes=init_yes,
                volume_24h=vol,
                start_day=day,
                resolution_day=day + days_to_res,
                resolves_yes=resolves_yes,
            ))
            mid += 1

    return markets


# ═══════════════════════════════════════════════════════════════════════════════
# SIMULATION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def simulate_tick(market: SimMarket, rng: np.random.Generator) -> float:
    """
    Simulate one tick of market activity.
    Returns the taker flow value for this tick.
    """
    # Generate taker flow (normal baseline + rare spikes)
    if float(rng.random()) < SIM_SPIKE_RATE:
        flow = float(rng.uniform(SIM_SPIKE_MAG_MIN, SIM_SPIKE_MAG_MAX))
    else:
        flow = float(rng.normal(0, 1))

    # Price evolution
    if flow > 2.0:
        impact = flow * SIM_SPIKE_PRICE_IMPACT
    else:
        impact = flow * SIM_NORMAL_PRICE_IMPACT

    # Mean reversion toward equilibrium
    reversion = (market.equilibrium_yes - market.current_yes) * SIM_REVERSION_RATE

    # Random walk
    noise = float(rng.normal(0, SIM_PRICE_NOISE_STD))

    market.current_yes += impact + reversion + noise
    market.current_yes = max(0.01, min(market.current_yes, 0.50))

    return flow


def apply_slippage(no_price: float, volume: float) -> float:
    """Apply slippage to NO entry price (makes it more expensive = worse fill)."""
    slip = SIM_BASE_SLIPPAGE
    if volume < SIM_THIN_VOLUME:
        slip += SIM_THIN_SLIPPAGE
    return no_price * (1 + slip)


# ═══════════════════════════════════════════════════════════════════════════════
# WINDOW SIMULATION
# ═══════════════════════════════════════════════════════════════════════════════

def simulate_window(seed: int, n_days: int, label: str) -> WindowResult:
    """Simulate one walk-forward window."""
    rng = np.random.default_rng(seed)

    # Reset strategy state
    strategy.reset_state()

    # Generate market universe
    markets = generate_markets(rng, n_days)
    market_lookup = {m.id: m for m in markets}

    # State
    capital = INITIAL_CAPITAL
    positions: dict[int, Position] = {}
    closed_trades: list[TradeResult] = []
    peak_capital = INITIAL_CAPITAL
    max_drawdown = 0.0

    n_ticks = n_days * TICKS_PER_DAY

    for tick in range(n_ticks):
        day = tick // TICKS_PER_DAY

        # Get active markets (started but not yet resolved)
        active = [m for m in markets if m.start_day <= day and not m.is_resolved]

        # --- 1. Simulate flows and get z-scores ---
        zscores: dict[int, float] = {}
        for market in active:
            flow = simulate_tick(market, rng)
            zscores[market.id] = strategy.detect_spike(market.id, flow)

        # --- 2. Check resolutions ---
        for market in active:
            if day >= market.resolution_day and not market.is_resolved:
                market.is_resolved = True
                if market.id in positions:
                    pos = positions.pop(market.id)
                    if market.resolves_yes:
                        resolve_value = 0.0  # NO is worthless
                    else:
                        resolve_value = pos.remaining_shares * 1.0  # NO pays $1

                    remaining_cost = pos.remaining_shares * pos.entry_price_no
                    pnl = resolve_value - remaining_cost
                    pnl_pct = pnl / remaining_cost if remaining_cost > 0 else 0

                    capital += resolve_value
                    closed_trades.append(TradeResult(
                        market_id=market.id,
                        pnl=pnl,
                        pnl_pct=pnl_pct,
                        won=(pnl > 0),
                        exit_reason="resolution",
                    ))

        # --- 3. Check exits for open positions ---
        to_close = []
        for mid, pos in positions.items():
            market = market_lookup[mid]
            if market.is_resolved:
                to_close.append(mid)
                continue

            current_no = 1.0 - market.current_yes
            pnl_pct = (current_no - pos.entry_price_no) / pos.entry_price_no
            pos.peak_pnl_pct = max(pos.peak_pnl_pct, pnl_pct)

            days_to_res = market.resolution_day - day

            exit_reason, exit_fraction = strategy.check_exit(
                pnl_pct=pnl_pct,
                peak_pnl_pct=pos.peak_pnl_pct,
                days_to_resolution=days_to_res,
                partial_exited=pos.partial_exited,
            )

            if exit_reason is None:
                continue

            if exit_fraction >= 1.0:
                # Full exit
                sell_value = pos.remaining_shares * current_no * (1 - SIM_BASE_SLIPPAGE)
                remaining_cost = pos.remaining_shares * pos.entry_price_no
                pnl = sell_value - remaining_cost
                trade_pnl_pct = pnl / remaining_cost if remaining_cost > 0 else 0

                capital += sell_value
                closed_trades.append(TradeResult(
                    market_id=mid,
                    pnl=pnl,
                    pnl_pct=trade_pnl_pct,
                    won=(pnl > 0),
                    exit_reason=exit_reason,
                ))
                to_close.append(mid)
            else:
                # Partial exit
                sell_shares = pos.remaining_shares * exit_fraction
                sell_value = sell_shares * current_no * (1 - SIM_BASE_SLIPPAGE)
                sell_cost = sell_shares * pos.entry_price_no
                partial_pnl = sell_value - sell_cost

                capital += sell_value
                pos.remaining_shares -= sell_shares
                pos.partial_exited = True

                closed_trades.append(TradeResult(
                    market_id=mid,
                    pnl=partial_pnl,
                    pnl_pct=partial_pnl / sell_cost if sell_cost > 0 else 0,
                    won=(partial_pnl > 0),
                    exit_reason="partial",
                ))

        for mid in to_close:
            positions.pop(mid, None)

        # --- 4. Check entries ---
        deployed = sum(p.remaining_shares * p.entry_price_no for p in positions.values())
        for market in active:
            if market.is_resolved:
                continue
            if market.id in positions:
                continue
            if len(positions) >= strategy.MAX_CONCURRENT:
                break
            if deployed >= INITIAL_CAPITAL * strategy.MAX_CAPITAL_DEPLOYED_PCT:
                break

            zscore = zscores.get(market.id, 0.0)
            days_to_res = market.resolution_day - day

            # Compute edge
            edge, model_no_prob = strategy.compute_entry(market.current_yes, zscore)

            # Quality gate
            if not strategy.should_trade(
                edge=edge,
                price_yes=market.current_yes,
                volume_24h=market.volume_24h,
                days_to_resolution=days_to_res,
                zscore=zscore,
            ):
                continue

            # Position sizing
            no_price = 1.0 - market.current_yes
            available = min(capital, MAX_SIZING_CAPITAL)
            size = strategy.position_size(
                edge=edge,
                no_price=no_price,
                available_capital=available,
                total_capital=capital,
            )

            # Minimum 1% of initial capital (prevents micro-position gaming)
            if size <= 0 or size > capital or size < INITIAL_CAPITAL * 0.01:
                continue

            # Apply slippage to entry
            entry_no = apply_slippage(no_price, market.volume_24h)
            shares = size / entry_no

            # Enter position
            capital -= size
            deployed += size
            positions[market.id] = Position(
                market_id=market.id,
                entry_price_no=entry_no,
                initial_size=size,
                initial_shares=shares,
                entry_tick=tick,
            )

        # --- 5. Track drawdown ---
        mtm = capital
        for mid, pos in positions.items():
            m = market_lookup[mid]
            mtm += pos.remaining_shares * (1.0 - m.current_yes)
        if mtm > peak_capital:
            peak_capital = mtm
        if peak_capital > 0:
            dd = (peak_capital - mtm) / peak_capital * 100.0
            max_drawdown = max(max_drawdown, dd)

    # --- 6. Force-close remaining positions ---
    for mid, pos in list(positions.items()):
        market = market_lookup[mid]
        # Assume market continues at current price
        current_no = 1.0 - market.current_yes
        sell_value = pos.remaining_shares * current_no * (1 - SIM_BASE_SLIPPAGE)
        remaining_cost = pos.remaining_shares * pos.entry_price_no
        pnl = sell_value - remaining_cost

        capital += sell_value
        closed_trades.append(TradeResult(
            market_id=mid,
            pnl=pnl,
            pnl_pct=pnl / remaining_cost if remaining_cost > 0 else 0,
            won=(pnl > 0),
            exit_reason="window_end",
        ))

    return _compute_metrics(closed_trades, max_drawdown)


# ═══════════════════════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_metrics(trades: list[TradeResult], max_drawdown: float) -> WindowResult:
    """Compute performance metrics from closed trades."""
    if not trades:
        return WindowResult(
            sharpe=0, win_rate=0, max_drawdown=max_drawdown,
            num_trades=0, total_pnl=0, pnl_pct=0, profit_factor=0,
        )

    wins = sum(1 for t in trades if t.won)
    win_rate = wins / len(trades) * 100.0

    total_pnl = sum(t.pnl for t in trades)
    pnl_pct = total_pnl / INITIAL_CAPITAL * 100.0

    # Per-trade returns as fraction of initial capital
    returns = [t.pnl / INITIAL_CAPITAL for t in trades]
    avg_return = sum(returns) / len(returns)

    if len(returns) > 1:
        std_return = math.sqrt(
            sum((r - avg_return) ** 2 for r in returns) / (len(returns) - 1)
        )
    else:
        std_return = abs(avg_return) if avg_return != 0 else 1.0

    # Annualized Sharpe (assuming ~90-day windows)
    trades_per_year = len(trades) * (365.0 / 90.0)
    sharpe = (avg_return / std_return * math.sqrt(trades_per_year)) if std_return > 0 else 0

    # Profit factor
    gross_wins = sum(t.pnl for t in trades if t.pnl > 0)
    gross_losses = abs(sum(t.pnl for t in trades if t.pnl < 0))
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else (
        float("inf") if gross_wins > 0 else 0
    )

    return WindowResult(
        sharpe=sharpe,
        win_rate=win_rate,
        max_drawdown=max_drawdown,
        num_trades=len(trades),
        total_pnl=total_pnl,
        pnl_pct=pnl_pct,
        profit_factor=profit_factor,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# COMPOSITE SCORE
# ═══════════════════════════════════════════════════════════════════════════════

def compute_composite_score(results: list[WindowResult]) -> dict:
    """
    Compute composite score from walk-forward window results.

    Formula:
        composite_score = avg_sharpe * sqrt(total_trades/40) * (1 - max_dd/30) * consistency

    Where:
        avg_sharpe = mean Sharpe across windows
        total_trades = sum of trades across ALL windows
        max_dd = worst drawdown across ALL windows (%)
        consistency = profitable_windows / total_windows

    Minimum 15 total trades for non-zero score.
    Max drawdown > 30% -> score = 0.
    """
    if not results:
        return _empty_scores()

    sharpes = [r.sharpe for r in results]
    avg_sharpe = sum(sharpes) / len(sharpes)

    total_trades = sum(r.num_trades for r in results)
    worst_drawdown = max(r.max_drawdown for r in results)
    profitable_windows = sum(1 for r in results if r.total_pnl > 0)

    avg_win_rate = sum(r.win_rate for r in results) / len(results)
    avg_pnl_pct = sum(r.pnl_pct for r in results) / len(results)
    avg_profit_factor = sum(
        min(r.profit_factor, 100) for r in results
    ) / len(results)

    # Composite score components
    if total_trades < 15:
        composite = 0.0
    elif worst_drawdown >= 30.0:
        composite = 0.0
    elif avg_sharpe <= 0:
        composite = 0.0
    else:
        trade_factor = math.sqrt(total_trades / 40.0)
        drawdown_factor = max(0.0, 1.0 - worst_drawdown / 30.0)
        consistency = profitable_windows / len(results)
        composite = avg_sharpe * trade_factor * drawdown_factor * consistency

    return {
        "composite_score": composite,
        "avg_sharpe": avg_sharpe,
        "avg_pnl_pct": avg_pnl_pct,
        "max_drawdown_pct": worst_drawdown,
        "avg_win_rate": avg_win_rate,
        "num_trades": total_trades,
        "avg_profit_factor": avg_profit_factor,
        "profitable_windows": f"{profitable_windows}/{len(results)}",
    }


def _empty_scores() -> dict:
    return {
        "composite_score": 0.0, "avg_sharpe": 0.0,
        "avg_pnl_pct": 0.0, "max_drawdown_pct": 0.0, "avg_win_rate": 0.0,
        "num_trades": 0, "avg_profit_factor": 0.0,
        "profitable_windows": "0/0",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()

    results: list[WindowResult] = []

    for window in WINDOWS:
        seed = BASE_SEED + window["seed_offset"]
        label = window["label"]

        print(f"\n=== {label} (seed={seed}) ===")
        result = simulate_window(seed, window["n_days"], label)
        results.append(result)

        print(f"Trades: {result.num_trades}, Win rate: {result.win_rate:.1f}%, "
              f"Sharpe: {result.sharpe:.2f}, Max DD: {result.max_drawdown:.2f}%, "
              f"PnL: ${result.total_pnl:.2f} ({result.pnl_pct:.1f}%)")

    # Compute composite score
    scores = compute_composite_score(results)
    elapsed = time.time() - t0

    # Print greppable summary (same format as B harness)
    print("\n--- RESULTS ---")
    print(f"composite_score:    {scores['composite_score']:.6f}")
    print(f"avg_sharpe:         {scores['avg_sharpe']:.4f}")
    print(f"avg_pnl_pct:        {scores['avg_pnl_pct']:.2f}")
    print(f"max_drawdown_pct:   {scores['max_drawdown_pct']:.2f}")
    print(f"avg_win_rate:       {scores['avg_win_rate']:.1f}")
    print(f"num_trades:         {scores['num_trades']}")
    print(f"avg_profit_factor:  {scores['avg_profit_factor']:.4f}")
    print(f"profitable_windows: {scores['profitable_windows']}")
    print(f"backtest_seconds:   {elapsed:.1f}")


if __name__ == "__main__":
    main()
