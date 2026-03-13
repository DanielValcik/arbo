"""
Autoresearch Backtest Harness for Strategy B (Reflexivity Surfer)
================================================================

FIXED — DO NOT MODIFY. The agent edits strategy_b_experiment.py only.

Loads historical crypto data (CoinGecko), generates synthetic Polymarket-style
binary markets, simulates divergence-based reflexivity trading through
walk-forward validation, and reports metrics.

Data: 20 coins, daily price + volume, 2024-2025
Markets: Synthetic binary "Will {coin} be above ${threshold} in N days?"
Edge: Our divergence signal vs market maker's price-only momentum

Usage: python3 research_b/backtest_b_harness.py
"""

import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

# Import the strategy module (this is what the agent modifies)
sys.path.insert(0, str(Path(__file__).parent))
import strategy_b_experiment as strategy


# ═══════════════════════════════════════════════════════════════════════════════
# FIXED CONSTANTS — DO NOT CHANGE
# ═══════════════════════════════════════════════════════════════════════════════

DATA_DIR = Path(__file__).parent / "data"
CACHE_FILE = DATA_DIR / "crypto_history.json"

COINS = [
    "bitcoin", "ethereum", "solana", "ripple", "cardano",
    "dogecoin", "avalanche-2", "polkadot", "chainlink", "matic-network",
    "uniswap", "cosmos", "litecoin", "near", "arbitrum",
    "optimism", "aptos", "sui", "filecoin", "aave",
]

COIN_SYMBOLS = {
    "bitcoin": "BTC", "ethereum": "ETH", "solana": "SOL", "ripple": "XRP",
    "cardano": "ADA", "dogecoin": "DOGE", "avalanche-2": "AVAX",
    "polkadot": "DOT", "chainlink": "LINK", "matic-network": "MATIC",
    "uniswap": "UNI", "cosmos": "ATOM", "litecoin": "LTC", "near": "NEAR",
    "arbitrum": "ARB", "optimism": "OP", "aptos": "APT", "sui": "SUI",
    "filecoin": "FIL", "aave": "AAVE",
}

# Walk-forward windows: 5 overlapping periods, 3-month test each
WALK_FORWARD_WINDOWS = [
    {"train": ("2024-01-01", "2024-06-30"), "test": ("2024-07-01", "2024-09-30")},
    {"train": ("2024-04-01", "2024-09-30"), "test": ("2024-10-01", "2024-12-31")},
    {"train": ("2024-07-01", "2024-12-31"), "test": ("2025-01-01", "2025-03-31")},
    {"train": ("2024-10-01", "2025-03-31"), "test": ("2025-04-01", "2025-06-30")},
    {"train": ("2025-01-01", "2025-06-30"), "test": ("2025-07-01", "2025-09-30")},
]

INITIAL_CAPITAL = 400.0         # Strategy B allocation (€400)
SLIPPAGE_PCT = 0.01             # 1% slippage (crypto markets)
MAX_SIZING_CAPITAL = 2000.0     # Cap to prevent unrealistic compounding
BASE_SEED = 137                 # Different from Strategy C

# Market maker model (less accurate — only uses price momentum, no divergence)
MM_MOMENTUM_SCALE = 0.15        # How much price trend affects MM pricing
MM_NOISE_STD = 0.04             # Random noise on MM contract prices

# Polymarket volume simulation (fraction of spot volume)
PM_VOLUME_SCALE_LOW = 0.000005
PM_VOLUME_SCALE_HIGH = 0.00005
PM_LIQUIDITY_RATIO_LOW = 0.1
PM_LIQUIDITY_RATIO_HIGH = 0.5


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Position:
    coin: str
    direction: str              # "YES" or "NO"
    entry_price: float          # Contract price at entry
    size_usd: float             # Dollar amount invested
    num_contracts: float        # size_usd / entry_price
    entry_date: date
    resolution_date: date
    threshold: float            # Price threshold for binary market
    entry_coin_price: float     # Coin price at entry
    phase: int                  # Phase that triggered this trade
    daily_volatility: float     # For intermediate valuation
    partially_exited: bool = False


@dataclass
class TradeResult:
    pnl: float
    won: bool
    date: str
    coin: str
    direction: str
    phase: int
    entry_price: float
    exit_reason: str            # "resolution", "stop_loss", "partial_exit"


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
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def _norm_cdf(x: float) -> float:
    """Standard normal CDF using error function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _contract_fair_value(
    current_coin_price: float,
    threshold: float,
    days_remaining: int,
    daily_volatility: float,
) -> float:
    """
    Approximate fair value of YES contract using log-normal model.

    Similar to Black-Scholes: P(S_T > K) where S_T is future price.
    """
    if days_remaining <= 0:
        return 1.0 if current_coin_price >= threshold else 0.0
    if threshold <= 0 or current_coin_price <= 0:
        return 0.5
    sigma = daily_volatility * math.sqrt(days_remaining)
    if sigma <= 0:
        return 1.0 if current_coin_price >= threshold else 0.0
    z = math.log(current_coin_price / threshold) / sigma
    return _norm_cdf(z)


def _compute_daily_volatility(price_series: list[float]) -> float:
    """Compute annualized daily log-return volatility."""
    if len(price_series) < 3:
        return 0.02  # Default 2% daily vol
    log_returns = []
    for i in range(1, len(price_series)):
        if price_series[i] > 0 and price_series[i - 1] > 0:
            log_returns.append(math.log(price_series[i] / price_series[i - 1]))
    if len(log_returns) < 2:
        return 0.02
    mean_r = sum(log_returns) / len(log_returns)
    var_r = sum((r - mean_r) ** 2 for r in log_returns) / (len(log_returns) - 1)
    return math.sqrt(var_r)


def _market_maker_price(price_series: list[float], rng: random.Random) -> float:
    """
    Market maker prices based on price momentum only.

    The MM sees recent price trend but does NOT have divergence signal.
    This is the information asymmetry that creates our edge.
    """
    if len(price_series) < 2:
        return 0.50

    # Recent price trend (7 days or available)
    lookback = min(7, len(price_series) - 1)
    recent_change = (price_series[-1] - price_series[-1 - lookback]) / price_series[-1 - lookback]
    momentum_adj = math.tanh(recent_change / 0.10) * MM_MOMENTUM_SCALE

    # Random noise
    noise = rng.gauss(0, MM_NOISE_STD)

    # At-the-money base
    mm_price = 0.50 + momentum_adj + noise
    return max(0.08, min(0.92, mm_price))


def _get_series(
    coin_dates: dict[str, dict], current: date, lookback: int
) -> list[dict]:
    """Get the most recent `lookback` days of data up to and including current."""
    entries = []
    d = current - timedelta(days=lookback + 30)  # Extra buffer for gaps
    while d <= current:
        ds = d.isoformat()
        if ds in coin_dates:
            entries.append(coin_dates[ds])
        d += timedelta(days=1)
    return entries[-lookback:] if len(entries) > lookback else entries


# ═══════════════════════════════════════════════════════════════════════════════
# WINDOW SIMULATION
# ═══════════════════════════════════════════════════════════════════════════════

def simulate_window(
    window: dict, all_data: dict[str, list[dict]], window_idx: int
) -> WindowResult:
    """Simulate one walk-forward window."""
    test_start = date.fromisoformat(window["test"][0])
    test_end = date.fromisoformat(window["test"][1])

    # Reset strategy state
    strategy.reset_state()

    capital = INITIAL_CAPITAL
    positions: list[Position] = []
    closed_trades: list[TradeResult] = []
    peak_capital = capital
    max_drawdown = 0.0

    rng = random.Random(BASE_SEED + window_idx * 1000)

    # Build daily data index per coin: {date_str: {price, volume, ...}}
    coin_data: dict[str, dict[str, dict]] = {}
    for coin in COINS:
        if coin not in all_data:
            continue
        coin_data[coin] = {e["date"]: e for e in all_data[coin]}

    # Pre-compute: run signals through train period to build up divergence history
    train_start = date.fromisoformat(window["train"][0])
    _warmup_signals(coin_data, train_start, test_start)

    # Main simulation loop: iterate through each day in the test window
    current = test_start
    while current <= test_end:
        date_str = current.isoformat()

        # --- 1. Generate signals and check for new trade entries ---
        for coin in COINS:
            if coin not in coin_data:
                continue

            # Get price/volume series for lookback
            lookback_needed = strategy.MOMENTUM_LOOKBACK + strategy.DIVERGENCE_HISTORY + 10
            series = _get_series(coin_data[coin], current, lookback_needed)
            if len(series) < strategy.MOMENTUM_LOOKBACK + 2:
                continue

            price_series = [s["price"] for s in series]
            volume_series = [s["volume"] for s in series]
            # Use Santiment DAA if available, else Binance num_trades as proxy
            if "daa" in series[0]:
                daa_series = [s["daa"] for s in series]
            elif "num_trades" in series[0]:
                daa_series = [float(s["num_trades"]) for s in series]
            else:
                daa_series = None
            current_price = price_series[-1]
            current_volume = volume_series[-1]

            if current_price <= 0:
                continue

            # Compute signals
            signal = strategy.compute_signals(coin, price_series, volume_series, daa_series)

            # Phase transition
            phase = strategy.get_phase_transition(
                coin, signal.divergence, signal.z_score, current
            )

            # Only trade in active phases
            if phase not in (2, 3, 4):
                continue

            # Skip if we already have a position in this coin
            if any(p.coin == coin for p in positions):
                continue

            # Max concurrent check
            if len(positions) >= strategy.MAX_CONCURRENT:
                continue

            # Generate synthetic Polymarket binary market
            holding_period = strategy.HOLDING_PERIOD_DAYS
            resolution_date = current + timedelta(days=holding_period)

            # Need future data for resolution
            resolution_str = resolution_date.isoformat()
            if resolution_date > test_end:
                continue
            if resolution_str not in coin_data[coin]:
                # Try nearby dates
                found = False
                for offset in range(1, 4):
                    alt = (resolution_date + timedelta(days=offset)).isoformat()
                    if alt in coin_data[coin]:
                        resolution_date = resolution_date + timedelta(days=offset)
                        resolution_str = alt
                        found = True
                        break
                if not found:
                    continue

            # Threshold = current price (at-the-money binary market)
            threshold = current_price

            # Market maker pricing (based on price momentum only)
            mm_price = _market_maker_price(price_series, rng)

            # Our probability estimate (uses divergence signal)
            our_prob = strategy.estimate_probability(
                signal.divergence, signal.z_score, phase
            )

            # Determine direction and edge
            if phase == 2:  # BOOM -> buy YES (expect price UP)
                direction = "YES"
                market_price = mm_price
                edge = our_prob - mm_price
            else:  # PEAK/BUST -> buy NO (expect price DOWN)
                direction = "NO"
                market_price = 1.0 - mm_price  # NO contract price
                edge = (1.0 - our_prob) - (1.0 - mm_price)

            if edge <= 0:
                continue

            # Simulated Polymarket volume/liquidity (derived from spot volume)
            pm_volume = current_volume * rng.uniform(PM_VOLUME_SCALE_LOW, PM_VOLUME_SCALE_HIGH)
            pm_liquidity = pm_volume * rng.uniform(PM_LIQUIDITY_RATIO_LOW, PM_LIQUIDITY_RATIO_HIGH)

            # Quality gate
            if not strategy.should_trade(
                edge=edge,
                market_price=market_price,
                divergence=signal.divergence,
                confidence=signal.confidence,
                volume_24h=pm_volume,
                liquidity=pm_liquidity,
                phase=phase,
            ):
                continue

            # Position sizing
            sizing_capital = min(capital, MAX_SIZING_CAPITAL)
            size = strategy.position_size(
                edge=edge,
                market_price=market_price,
                available_capital=sizing_capital,
                total_capital=capital,
                phase=phase,
            )

            if size <= 0 or size > capital:
                continue

            # Apply slippage
            entry_price = market_price * (1.0 + SLIPPAGE_PCT)
            entry_price = min(entry_price, 0.95)

            num_contracts = size / entry_price

            # Daily volatility for intermediate valuation
            daily_vol = _compute_daily_volatility(price_series[-60:])

            pos = Position(
                coin=coin,
                direction=direction,
                entry_price=entry_price,
                size_usd=size,
                num_contracts=num_contracts,
                entry_date=current,
                resolution_date=resolution_date,
                threshold=threshold,
                entry_coin_price=current_price,
                phase=phase,
                daily_volatility=daily_vol,
            )

            positions.append(pos)
            capital -= size

        # --- 2. Check existing positions for exits ---
        still_open: list[Position] = []
        for pos in positions:
            coin_today = coin_data.get(pos.coin, {}).get(date_str)
            if not coin_today:
                still_open.append(pos)
                continue

            current_coin_price = coin_today["price"]

            if current >= pos.resolution_date:
                # Resolution
                won = (current_coin_price >= pos.threshold) == (pos.direction == "YES")
                if won:
                    pnl = pos.num_contracts * 1.0 - pos.size_usd
                else:
                    pnl = -pos.size_usd

                capital += pos.size_usd + pnl
                closed_trades.append(TradeResult(
                    pnl=pnl, won=won, date=date_str, coin=pos.coin,
                    direction=pos.direction, phase=pos.phase,
                    entry_price=pos.entry_price, exit_reason="resolution",
                ))
                strategy.reset_phase(pos.coin, current)
            else:
                # Intermediate: check stop loss and partial exit
                days_remaining = (pos.resolution_date - current).days
                contract_val = _contract_fair_value(
                    current_coin_price, pos.threshold, days_remaining, pos.daily_volatility
                )

                if pos.direction == "YES":
                    unrealized_pnl_pct = (contract_val - pos.entry_price) / pos.entry_price
                else:
                    no_val = 1.0 - contract_val
                    unrealized_pnl_pct = (no_val - pos.entry_price) / pos.entry_price

                stop_loss = (
                    strategy.PHASE2_STOP_LOSS if pos.phase == 2
                    else strategy.PHASE3_STOP_LOSS
                )

                if unrealized_pnl_pct < -stop_loss:
                    # Stop loss hit
                    realized_loss = stop_loss * pos.size_usd
                    pnl = -realized_loss
                    capital += pos.size_usd + pnl
                    closed_trades.append(TradeResult(
                        pnl=pnl, won=False, date=date_str, coin=pos.coin,
                        direction=pos.direction, phase=pos.phase,
                        entry_price=pos.entry_price, exit_reason="stop_loss",
                    ))
                    strategy.reset_phase(pos.coin, current)
                elif (
                    unrealized_pnl_pct > strategy.PARTIAL_EXIT_GAIN
                    and not pos.partially_exited
                ):
                    # Partial exit
                    exit_frac = strategy.PARTIAL_EXIT_PCT
                    exit_pnl = exit_frac * pos.size_usd * unrealized_pnl_pct
                    capital += exit_frac * pos.size_usd + exit_pnl

                    closed_trades.append(TradeResult(
                        pnl=exit_pnl, won=True, date=date_str, coin=pos.coin,
                        direction=pos.direction, phase=pos.phase,
                        entry_price=pos.entry_price, exit_reason="partial_exit",
                    ))

                    pos.size_usd *= (1.0 - exit_frac)
                    pos.num_contracts *= (1.0 - exit_frac)
                    pos.partially_exited = True
                    still_open.append(pos)
                else:
                    still_open.append(pos)

        positions = still_open

        # --- 3. Track drawdown ---
        invested = sum(p.size_usd for p in positions)
        total_value = capital + invested
        if total_value > peak_capital:
            peak_capital = total_value
        if peak_capital > 0:
            dd = (peak_capital - total_value) / peak_capital * 100.0
            if dd > max_drawdown:
                max_drawdown = dd

        current += timedelta(days=1)

    # --- 4. Force-close remaining positions at window end ---
    for pos in positions:
        final_str = test_end.isoformat()
        coin_today = coin_data.get(pos.coin, {}).get(final_str)
        if coin_today:
            current_coin_price = coin_today["price"]
            won = (current_coin_price >= pos.threshold) == (pos.direction == "YES")
            pnl = (pos.num_contracts * 1.0 - pos.size_usd) if won else -pos.size_usd
        else:
            pnl = 0.0
            won = False

        capital += pos.size_usd + pnl
        closed_trades.append(TradeResult(
            pnl=pnl, won=won, date=final_str, coin=pos.coin,
            direction=pos.direction, phase=pos.phase,
            entry_price=pos.entry_price, exit_reason="window_end",
        ))

    # --- 5. Compute metrics ---
    return _compute_metrics(closed_trades, max_drawdown)


def _warmup_signals(
    coin_data: dict[str, dict[str, dict]], train_start: date, test_start: date
) -> None:
    """Run signals through training period to build up divergence history."""
    current = train_start
    while current < test_start:
        for coin in coin_data:
            lookback = strategy.MOMENTUM_LOOKBACK + strategy.DIVERGENCE_HISTORY + 10
            series = _get_series(coin_data[coin], current, lookback)
            if len(series) < strategy.MOMENTUM_LOOKBACK + 2:
                continue
            price_series = [s["price"] for s in series]
            volume_series = [s["volume"] for s in series]
            if series and "daa" in series[0]:
                daa_series = [s["daa"] for s in series]
            elif series and "num_trades" in series[0]:
                daa_series = [float(s["num_trades"]) for s in series]
            else:
                daa_series = None
            strategy.compute_signals(coin, price_series, volume_series, daa_series)
        current += timedelta(days=1)


def _compute_metrics(
    trades: list[TradeResult], max_drawdown: float
) -> WindowResult:
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

    # Annualized Sharpe
    trades_per_year = len(trades) * (365.0 / 90.0)  # ~90 day test windows
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
        composite_score = avg_sharpe * sqrt(total_trades/30) * (1 - max_dd/40) * consistency

    Where:
        avg_sharpe = mean Sharpe across windows
        total_trades = sum of trades across ALL windows
        max_dd = worst drawdown across ALL windows (%)
        consistency = profitable_windows / total_windows

    Minimum 10 total trades for non-zero score.
    Max drawdown > 40% -> score = 0.
    """
    if not results:
        return _empty_scores()

    sharpes = [r.sharpe for r in results]
    avg_sharpe = sum(sharpes) / len(sharpes)
    sharpe_std = (
        math.sqrt(sum((s - avg_sharpe) ** 2 for s in sharpes) / len(sharpes))
        if len(sharpes) > 1 else 0.0
    )

    total_trades = sum(r.num_trades for r in results)
    worst_drawdown = max(r.max_drawdown for r in results)
    profitable_windows = sum(1 for r in results if r.total_pnl > 0)

    avg_win_rate = sum(r.win_rate for r in results) / len(results)
    avg_pnl_pct = sum(r.pnl_pct for r in results) / len(results)
    avg_return_pct = sum(r.pnl_pct for r in results) / len(results)
    avg_profit_factor = sum(
        min(r.profit_factor, 100) for r in results
    ) / len(results)

    # Composite score components
    if total_trades < 10:
        composite = 0.0
    elif worst_drawdown >= 40.0:
        composite = 0.0
    elif avg_sharpe <= 0:
        composite = 0.0
    else:
        trade_factor = math.sqrt(total_trades / 30.0)
        drawdown_factor = max(0.0, 1.0 - worst_drawdown / 40.0)
        consistency = profitable_windows / len(results)
        composite = avg_sharpe * trade_factor * drawdown_factor * consistency

    return {
        "composite_score": composite,
        "avg_sharpe": avg_sharpe,
        "sharpe_std": sharpe_std,
        "avg_pnl_pct": avg_pnl_pct,
        "max_drawdown_pct": worst_drawdown,
        "avg_win_rate": avg_win_rate,
        "num_trades": total_trades,
        "avg_profit_factor": avg_profit_factor,
        "avg_return_pct": avg_return_pct,
        "profitable_windows": f"{profitable_windows}/{len(results)}",
    }


def _empty_scores() -> dict:
    return {
        "composite_score": 0.0, "avg_sharpe": 0.0, "sharpe_std": 0.0,
        "avg_pnl_pct": 0.0, "max_drawdown_pct": 0.0, "avg_win_rate": 0.0,
        "num_trades": 0, "avg_profit_factor": 0.0, "avg_return_pct": 0.0,
        "profitable_windows": "0/0",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_data() -> dict[str, list[dict]]:
    """Load historical crypto data from cache file."""
    if not CACHE_FILE.exists():
        print(f"ERROR: Data file not found: {CACHE_FILE}")
        print("Run: python3 research_b/fetch_data.py")
        sys.exit(1)

    with open(CACHE_FILE) as f:
        data = json.load(f)

    # Validate
    total_entries = sum(len(v) for v in data.values())
    coins_found = [c for c in COINS if c in data]

    if len(coins_found) < 5:
        print(f"ERROR: Only {len(coins_found)} coins found, need at least 5")
        sys.exit(1)

    print(f"Loaded {len(coins_found)} coins, {total_entries} entries")

    # Check date coverage
    min_date = None
    max_date = None
    for coin_entries in data.values():
        if coin_entries:
            d0 = coin_entries[0]["date"]
            d1 = coin_entries[-1]["date"]
            if min_date is None or d0 < min_date:
                min_date = d0
            if max_date is None or d1 > max_date:
                max_date = d1

    print(f"Date range: {min_date} to {max_date}")

    # Check for DAA data
    has_daa = any(
        "daa" in data[coin][0]
        for coin in data
        if data[coin]
    )
    has_num_trades = any(
        "num_trades" in data[coin][0]
        for coin in data
        if data[coin]
    )
    if has_daa:
        print("DAA data: available (Santiment)")
    elif has_num_trades:
        print("DAA data: using Binance num_trades (high quality proxy)")
    else:
        print("DAA data: using volume/price proxy")

    return data


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()

    all_data = load_data()

    window_results: list[WindowResult] = []

    for i, window in enumerate(WALK_FORWARD_WINDOWS):
        test_start_str = window["test"][0]
        test_end_str = window["test"][1]

        # Check if we have data for this window
        has_data = False
        for coin in COINS:
            if coin in all_data:
                dates = {e["date"] for e in all_data[coin]}
                if test_start_str in dates or any(
                    d >= test_start_str and d <= test_end_str for d in dates
                ):
                    has_data = True
                    break

        if not has_data:
            print(f"\n=== Window {i + 1} === SKIPPED (no data for {test_start_str} to {test_end_str})")
            continue

        print(f"\n=== Walk-Forward Window {i + 1} ===")
        print(f"Train: {window['train'][0]} to {window['train'][1]}")
        print(f"Test:  {test_start_str} to {test_end_str}")

        result = simulate_window(window, all_data, i)
        window_results.append(result)

        print(f"Trades: {result.num_trades}, Win rate: {result.win_rate:.1f}%, "
              f"Sharpe: {result.sharpe:.2f}, Max DD: {result.max_drawdown:.2f}%, "
              f"PnL: ${result.total_pnl:.2f} ({result.pnl_pct:.1f}%)")

    if not window_results:
        print("\nERROR: No windows could be evaluated")
        scores = _empty_scores()
    else:
        scores = compute_composite_score(window_results)

    elapsed = time.time() - t0

    # Print greppable summary
    print("\n---")
    print(f"composite_score:    {scores['composite_score']:.6f}")
    print(f"avg_sharpe:         {scores['avg_sharpe']:.4f}")
    print(f"sharpe_std:         {scores['sharpe_std']:.4f}")
    print(f"avg_pnl_pct:        {scores['avg_pnl_pct']:.2f}")
    print(f"max_drawdown_pct:   {scores['max_drawdown_pct']:.2f}")
    print(f"avg_win_rate:       {scores['avg_win_rate']:.1f}")
    print(f"num_trades:         {scores['num_trades']}")
    print(f"avg_profit_factor:  {scores['avg_profit_factor']:.4f}")
    print(f"avg_return_pct:     {scores['avg_return_pct']:.2f}")
    print(f"profitable_windows: {scores['profitable_windows']}")
    print(f"backtest_seconds:   {elapsed:.1f}")


if __name__ == "__main__":
    main()
