"""
Strategy B3 — Binance Oracle Scalper Autoresearch
==================================================

Simulates scalping BTC 5min/15min Up/Down markets on Polymarket
using Binance real-time price as a fair value oracle.

Core insight:
  Binance trades BTC at $50B+/day volume. Polymarket Up/Down markets
  have ~$150K volume. Binance price moves FIRST — we compute fair
  value from Binance and trade Polymarket when it diverges.

Fair value model:
  P(Up) = Φ( log(S_now / S_start) / (σ_per_min × σ_scale × √t_remaining) )

  Where S_start is BTC price at event window open, S_now is current
  Binance price, σ is realized volatility from recent 1-min returns.

Strategy:
  - Enter when |fair_value - 0.50| > entry_threshold (BTC has moved)
  - Exit on: profit target, stop loss, edge gone, time limit, near resolution
  - PostOnly orders = 0% maker fee + 20% rebate
  - NEVER hold to resolution by default → bounded risk, continuous scalping

Data: 89,419 Binance 1-min BTCUSDT klines (89 days, 2025-12-28 → 2026-03-27)
  → 8,571 × 15-min windows
  → 25,715 × 5-min windows

Fee model (crypto_fees):
  Taker: fee = 0.10 × min(p, 1-p)² per share
  Maker: 0% fee + rebate = 0.20 × taker_fee
  PostOnly guarantees maker status.

Phases:
  Gen 0: Random params (1000 trials)
  Gen 1: Fine-tune top-15 (600 trials)
  Gen 2: Exit + window tuning (300 trials)
  WF:    Walk-forward 70/30 split on top-5

Usage:
    python3 research/innovations/sweep_b3_scalper.py
"""

from __future__ import annotations

import json
import math
import random
import sqlite3
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════════════════════════════════════════

DATA_DIR = Path(__file__).parent.parent / "data"
DB_PATH = DATA_DIR / "crypto_price_pmd.sqlite"
RESULTS_DIR = DATA_DIR / "experiments"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
SWEEP_PATH = RESULTS_DIR / "sweep_b3_scalper.json"
LOG_PATH = Path(__file__).parent.parent / "sweep_b3_scalper_log.txt"

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

INITIAL_CAPITAL = 1000.0
MIN_ORDER_SIZE = 5.0  # Polymarket minimum $5
MAX_BET_SIZE = 100.0  # Max per trade (conservative for ~$40K book)
MAX_SHARES = 500      # Max shares per trade (prevents extreme OTM leverage)
MIN_ENTRY_MKT_FV = 0.15  # Don't buy tokens priced < 15% (no liquidity at deep OTM)
MAX_ENTRY_MKT_FV = 0.85  # Don't buy tokens priced > 85% (too expensive, low edge)
SIGMA_FLOOR = 0.00005  # ~2.6% annualized minimum
MIN_TRADES_FOR_SCORE = 30

# Phase config
GEN0_TRIALS = 1000
GEN1_TRIALS = 600
GEN2_TRIALS = 300
TOP_K = 15

# Walk-forward split
WF_IS_PCT = 0.70  # 70% in-sample

# ═══════════════════════════════════════════════════════════════════════════════
# MATH
# ═══════════════════════════════════════════════════════════════════════════════

_SQRT2 = math.sqrt(2.0)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / _SQRT2))


def _taker_fee(price: float) -> float:
    """Crypto taker fee: 0.10 × min(p, 1-p)²."""
    p = min(price, 1.0 - price)
    return 0.10 * p * p


def _maker_rebate(price: float) -> float:
    """Maker rebate: 20% of what taker pays."""
    return 0.20 * _taker_fee(price)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════


def load_klines(db_path: Path = DB_PATH) -> tuple[list[float], list[int]]:
    """Load Binance 1-min BTCUSDT closes and timestamps.

    Returns:
        (closes, timestamps) — parallel lists, sorted by time.
    """
    conn = sqlite3.connect(str(db_path))
    rows = conn.execute(
        "SELECT ts, close FROM binance_klines "
        "WHERE symbol='BTCUSDT' ORDER BY ts"
    ).fetchall()
    conn.close()

    timestamps = [r[0] for r in rows]
    closes = [r[1] for r in rows]
    print(f"Loaded {len(closes):,} BTCUSDT 1-min klines")
    print(f"  Range: {datetime.utcfromtimestamp(timestamps[0]).strftime('%Y-%m-%d')} → "
          f"{datetime.utcfromtimestamp(timestamps[-1]).strftime('%Y-%m-%d')}")
    return closes, timestamps


def build_windows(
    closes: list[float], timestamps: list[int], window_min: int,
) -> list[int]:
    """Return list of start indices for non-overlapping windows.

    Each window = window_min + 1 consecutive klines (entry through resolution).
    We step by window_min to avoid overlap.
    """
    indices = []
    i = 0
    while i + window_min < len(closes):
        # Check gap — skip if timestamps aren't consecutive minutes
        gap = timestamps[i + 1] - timestamps[i]
        if gap <= 120:  # Allow up to 2-min gap (occasional missing kline)
            indices.append(i)
        i += window_min  # Non-overlapping
    return indices


def precompute_log_returns(closes: list[float]) -> list[float]:
    """Compute log returns for every consecutive pair."""
    lr = [0.0]  # Index 0 has no return
    for i in range(1, len(closes)):
        if closes[i] > 0 and closes[i - 1] > 0:
            lr.append(math.log(closes[i] / closes[i - 1]))
        else:
            lr.append(0.0)
    return lr


# ═══════════════════════════════════════════════════════════════════════════════
# VOLATILITY
# ═══════════════════════════════════════════════════════════════════════════════


def compute_sigma(
    log_returns: list[float], end_idx: int, window: int, method: str,
) -> float:
    """Compute per-minute σ from preceding log returns."""
    start = max(1, end_idx - window)
    rets = log_returns[start:end_idx]
    if len(rets) < 10:
        return SIGMA_FLOOR

    if method == "ewma":
        alpha = 0.06
        var = rets[0] ** 2
        for r in rets[1:]:
            var = alpha * r * r + (1 - alpha) * var
        return max(math.sqrt(var), SIGMA_FLOOR)

    elif method == "garch":
        omega, a, b = 1e-8, 0.10, 0.85
        var = sum(r * r for r in rets) / len(rets)
        for r in rets:
            var = omega + a * r * r + b * var
        return max(math.sqrt(var), SIGMA_FLOOR)

    else:  # realized
        n = len(rets)
        mean = sum(rets) / n
        var = sum((r - mean) ** 2 for r in rets) / (n - 1)
        return max(math.sqrt(var), SIGMA_FLOOR)


# ═══════════════════════════════════════════════════════════════════════════════
# PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class B3Params:
    # Volatility
    sigma_window: int = 360       # minutes of history for σ
    sigma_method: str = "realized"
    sigma_scale: float = 1.0

    # Entry
    entry_threshold: float = 0.08  # min |fair_value - 0.50| to enter
    min_entry_min: int = 1         # earliest minute to enter
    max_entry_min: int = 12        # latest minute to enter
    contrarian: bool = False       # bet AGAINST the move (mean reversion)

    # Exit
    profit_target: float = 0.08    # take profit per share ($)
    stop_loss: float = 0.10        # max loss per share ($)
    max_hold_min: int = 8          # max minutes to hold
    edge_exit: float = 0.03        # exit if |position_fv - 0.50| drops below
    exit_before_end: int = 1       # exit N minutes before resolution
    allow_resolution: bool = False # hold through resolution if still open

    # Market model
    spread: float = 0.02           # bid-ask spread
    maker: bool = True             # PostOnly (0% fee) vs taker

    # Sizing
    position_pct: float = 0.05    # max % of capital per trade
    edge_scaling: float = 2.0     # size multiplier per unit of edge

    # Window
    window_min: int = 15          # 5 or 15

    # Re-entry
    reentry_cooldown: int = 1     # minutes after exit before re-entry


# ═══════════════════════════════════════════════════════════════════════════════
# SIMULATION — SINGLE WINDOW
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class Trade:
    direction: int      # +1 = long Up, -1 = long Down
    entry_min: int
    exit_min: int
    entry_fv: float
    exit_fv: float
    shares: float
    gross_pnl: float
    fees: float
    net_pnl: float
    exit_reason: str


def simulate_window(
    closes: list[float],
    start_idx: int,
    sigma_per_min: float,
    capital: float,
    p: B3Params,
) -> list[Trade]:
    """Simulate one 5/15-min window. Returns list of trades.

    KEY DESIGN: Separates SIGNAL from PRICE.
    - market_fv: fair value with true σ (σ_scale=1.0) — what the market trades at
    - signal_fv: fair value with model σ (σ_scale param) — entry/exit trigger
    Entry/exit PRICES always use market_fv. σ_scale only affects signal sensitivity.
    """
    wm = p.window_min
    S_start = closes[start_idx]
    if S_start <= 0:
        return []

    # Effective entry limits for this window size
    max_entry = min(p.max_entry_min, wm - p.exit_before_end - 1)
    if max_entry < p.min_entry_min:
        return []

    trades: list[Trade] = []
    position = None  # (direction, entry_mkt_fv, entry_min, shares, entry_fee_per_share)
    last_exit_min = -999

    for t in range(1, wm):
        S_now = closes[start_idx + t]
        if S_now <= 0:
            continue
        t_remaining = wm - t
        if t_remaining <= 0:
            continue

        log_ratio = math.log(S_now / S_start)
        sqrt_t = math.sqrt(t_remaining)

        # ── MARKET fair value (σ_scale=1.0) — realistic price ──
        sigma_rem_true = sigma_per_min * sqrt_t
        if sigma_rem_true > 1e-12:
            market_up = _norm_cdf(log_ratio / sigma_rem_true)
        else:
            market_up = 1.0 if log_ratio > 0 else 0.0 if log_ratio < 0 else 0.5
        market_up = max(0.02, min(0.98, market_up))

        # ── SIGNAL fair value (σ_scale from params) — entry trigger ──
        sigma_rem_model = sigma_per_min * p.sigma_scale * sqrt_t
        if sigma_rem_model > 1e-12:
            signal_up = _norm_cdf(log_ratio / sigma_rem_model)
        else:
            signal_up = 1.0 if log_ratio > 0 else 0.0 if log_ratio < 0 else 0.5
        signal_up = max(0.01, min(0.99, signal_up))

        if position is None:
            # ══ ENTRY ══
            if t - last_exit_min < p.reentry_cooldown:
                continue
            if t < p.min_entry_min or t > max_entry:
                continue

            # Signal based on model's view (amplified by σ_scale)
            signal_dev = signal_up - 0.50
            if abs(signal_dev) < p.entry_threshold:
                continue

            # Direction: momentum or contrarian
            if p.contrarian:
                direction = -1 if signal_dev > 0 else 1
            else:
                direction = 1 if signal_dev > 0 else -1

            # Entry at MARKET price (not model price!)
            entry_mkt_fv = market_up if direction == 1 else (1.0 - market_up)

            # Skip if market price is too extreme (no liquidity at deep OTM/ITM)
            if entry_mkt_fv < MIN_ENTRY_MKT_FV or entry_mkt_fv > MAX_ENTRY_MKT_FV:
                continue

            entry_price = entry_mkt_fv + p.spread / 2  # Buy at ask

            # Fee
            if p.maker:
                entry_fee = -_maker_rebate(entry_mkt_fv)
            else:
                entry_fee = _taker_fee(entry_mkt_fv)

            # Sizing: proportional to signal edge, capped
            edge = abs(signal_dev)
            raw_pct = min(p.position_pct, edge * p.edge_scaling)
            bet_size = min(capital * raw_pct, MAX_BET_SIZE)
            if bet_size < MIN_ORDER_SIZE:
                continue
            shares = bet_size / entry_price if entry_price > 0.01 else 0
            shares = min(shares, MAX_SHARES)
            if shares < 1:
                continue

            position = (direction, entry_mkt_fv, t, shares, entry_fee)

        else:
            # ══ EXIT CHECK ══
            direction, entry_mkt_fv, entry_t, shares, entry_fee = position
            hold_min = t - entry_t

            # Current MARKET fair value for our position
            pos_mkt_fv = market_up if direction == 1 else (1.0 - market_up)
            unrealized = pos_mkt_fv - entry_mkt_fv

            # Signal-based edge check (using model)
            pos_signal_fv = signal_up if direction == 1 else (1.0 - signal_up)

            should_exit = False
            reason = ""

            if unrealized >= p.profit_target:
                should_exit, reason = True, "profit"
            elif unrealized <= -p.stop_loss:
                should_exit, reason = True, "stop"
            elif hold_min >= p.max_hold_min:
                should_exit, reason = True, "time"
            elif abs(pos_signal_fv - 0.50) < p.edge_exit and hold_min >= 1:
                should_exit, reason = True, "edge_gone"
            elif t >= wm - p.exit_before_end:
                should_exit, reason = True, "near_end"

            if should_exit:
                exit_mkt_fv = pos_mkt_fv
                exit_price = exit_mkt_fv - p.spread / 2  # Sell at bid

                if p.maker:
                    exit_fee = -_maker_rebate(exit_mkt_fv)
                else:
                    exit_fee = _taker_fee(exit_mkt_fv)

                gross = (exit_price - (entry_mkt_fv + p.spread / 2)) * shares
                fee_total = (entry_fee + exit_fee) * shares
                net = gross - fee_total

                trades.append(Trade(
                    direction=direction,
                    entry_min=entry_t, exit_min=t,
                    entry_fv=entry_mkt_fv, exit_fv=exit_mkt_fv,
                    shares=shares,
                    gross_pnl=gross, fees=fee_total, net_pnl=net,
                    exit_reason=reason,
                ))
                position = None
                last_exit_min = t

    # ── Handle unresolved position ──
    if position is not None:
        direction, entry_mkt_fv, entry_t, shares, entry_fee = position
        if p.allow_resolution:
            resolved_up = closes[start_idx + wm] >= S_start
            won = (direction == 1 and resolved_up) or (direction == -1 and not resolved_up)
            # Resolution: token redeems at exactly $1 or $0. No fee, no spread.
            exit_mkt_fv = 1.0 if won else 0.0
            exit_fee = 0.0  # Resolution is automatic, no trading fee

            gross = (exit_mkt_fv - (entry_mkt_fv + p.spread / 2)) * shares
            fee_total = entry_fee * shares  # Only entry fee
            net = gross - fee_total
            trades.append(Trade(
                direction=direction,
                entry_min=entry_t, exit_min=wm,
                entry_fv=entry_mkt_fv, exit_fv=exit_mkt_fv,
                shares=shares,
                gross_pnl=gross, fees=fee_total, net_pnl=net,
                exit_reason="resolution_win" if won else "resolution_lose",
            ))
        else:
            # Force exit at market fair value (last computed)
            exit_mkt_fv = market_up if direction == 1 else (1.0 - market_up)
            exit_price = exit_mkt_fv - p.spread / 2

            if p.maker:
                exit_fee = -_maker_rebate(exit_mkt_fv)
            else:
                exit_fee = _taker_fee(exit_mkt_fv)

            gross = (exit_price - (entry_mkt_fv + p.spread / 2)) * shares
            fee_total = (entry_fee + exit_fee) * shares
            net = gross - fee_total
            trades.append(Trade(
                direction=direction,
                entry_min=entry_t, exit_min=wm - 1,
                entry_fv=entry_mkt_fv, exit_fv=exit_mkt_fv,
                shares=shares,
                gross_pnl=gross, fees=fee_total, net_pnl=net,
                exit_reason="forced_end",
            ))

    return trades


# ═══════════════════════════════════════════════════════════════════════════════
# PORTFOLIO SIMULATION
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class SimResult:
    total_pnl: float = 0.0
    total_trades: int = 0
    wins: int = 0
    total_fees: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe: float = 0.0
    win_rate: float = 0.0
    avg_hold_min: float = 0.0
    exit_reasons: dict = field(default_factory=dict)
    daily_pnls: list = field(default_factory=list)
    score: float = 0.0


def simulate_portfolio(
    closes: list[float],
    timestamps: list[int],
    log_returns: list[float],
    windows: list[int],
    params: B3Params,
) -> SimResult:
    """Run strategy across all windows, tracking capital and P&L."""
    capital = INITIAL_CAPITAL
    peak_capital = INITIAL_CAPITAL
    max_dd = 0.0

    all_trades: list[Trade] = []
    daily_pnl: dict[str, float] = defaultdict(float)
    exit_reasons: dict[str, int] = defaultdict(int)
    total_fees = 0.0

    for start_idx in windows:
        if capital < MIN_ORDER_SIZE:
            break

        # Compute σ for this window
        sigma = compute_sigma(
            log_returns, start_idx, params.sigma_window, params.sigma_method,
        )

        # Simulate
        trades = simulate_window(closes, start_idx, sigma, capital, params)

        for tr in trades:
            all_trades.append(tr)
            capital += tr.net_pnl
            total_fees += tr.fees

            # Track daily P&L
            ts = timestamps[start_idx + tr.entry_min]
            day = datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d")
            daily_pnl[day] += tr.net_pnl

            exit_reasons[tr.exit_reason] += 1

            # Drawdown
            if capital > peak_capital:
                peak_capital = capital
            dd = (peak_capital - capital) / peak_capital if peak_capital > 0 else 0
            if dd > max_dd:
                max_dd = dd

    # ── Aggregate ──
    n = len(all_trades)
    result = SimResult()
    result.total_pnl = capital - INITIAL_CAPITAL
    result.total_trades = n
    result.total_fees = total_fees
    result.max_drawdown_pct = max_dd * 100
    result.exit_reasons = dict(exit_reasons)

    if n > 0:
        result.wins = sum(1 for t in all_trades if t.net_pnl > 0)
        result.win_rate = result.wins / n * 100
        result.avg_hold_min = sum(t.exit_min - t.entry_min for t in all_trades) / n

    # Sharpe from daily P&L
    daily_vals = list(daily_pnl.values())
    if len(daily_vals) >= 5:
        mean_d = sum(daily_vals) / len(daily_vals)
        diffs = [(d - mean_d) for d in daily_vals]
        var_d = sum(d * d for d in diffs) / (len(daily_vals) - 1)
        std_d = math.sqrt(max(var_d, 1e-10))
        if std_d > 1e-8:
            result.sharpe = min((mean_d / std_d) * math.sqrt(365), 100.0)
        else:
            result.sharpe = 0.0
    result.daily_pnls = daily_vals

    # Score
    result.score = experiment_score(result)
    return result


def experiment_score(r: SimResult) -> float:
    """Score that rewards P&L, trade count, Sharpe, and penalizes drawdown."""
    if r.total_trades < MIN_TRADES_FOR_SCORE:
        return 0.0

    roi = r.total_pnl / INITIAL_CAPITAL * 100  # ROI %
    if roi <= 0:
        return 0.0

    trade_factor = 1.0 + math.log10(max(r.total_trades, 1))
    dd_penalty = max(0.0, 1.0 - r.max_drawdown_pct / 100.0)
    sharpe_factor = min(max(r.sharpe, 0), 15) / 15.0

    return roi * trade_factor * dd_penalty * sharpe_factor


# ═══════════════════════════════════════════════════════════════════════════════
# PARAMETER GENERATORS
# ═══════════════════════════════════════════════════════════════════════════════


def random_params() -> B3Params:
    wm = random.choice([5, 15])
    max_e = wm - 2
    return B3Params(
        sigma_window=random.choice([30, 60, 120, 240, 360, 720, 1440]),
        sigma_method=random.choice(["realized", "ewma", "garch"]),
        sigma_scale=round(random.uniform(0.3, 3.0), 2),
        entry_threshold=round(random.uniform(0.02, 0.30), 3),
        min_entry_min=random.randint(0, min(3, max_e - 1)),
        max_entry_min=random.randint(max(2, max_e - 4), max_e),
        contrarian=random.choices([True, False], weights=[30, 70])[0],
        profit_target=round(random.uniform(0.01, 0.30), 3),
        stop_loss=round(random.uniform(0.02, 0.30), 3),
        max_hold_min=random.randint(1, max(2, wm - 2)),
        edge_exit=round(random.uniform(0.0, 0.15), 3),
        exit_before_end=random.randint(0, min(3, wm - 2)),
        allow_resolution=random.choices([True, False], weights=[20, 80])[0],
        spread=round(random.uniform(0.01, 0.05), 3),
        maker=random.choice([True, False]),
        position_pct=round(random.uniform(0.02, 0.20), 3),
        edge_scaling=round(random.uniform(0.5, 8.0), 2),
        window_min=wm,
        reentry_cooldown=random.randint(0, min(3, max_e)),
    )


def mutate_params(base: B3Params, intensity: float = 0.3) -> B3Params:
    """Create a mutated version of base params."""
    p = B3Params(**asdict(base))

    def jitter_f(val: float, lo: float, hi: float) -> float:
        delta = (hi - lo) * intensity * random.gauss(0, 1)
        return round(max(lo, min(hi, val + delta)), 3)

    def jitter_i(val: int, lo: int, hi: int) -> int:
        delta = max(1, int((hi - lo) * intensity))
        return max(lo, min(hi, val + random.randint(-delta, delta)))

    max_e = p.window_min - 2

    p.sigma_window = random.choice([30, 60, 120, 240, 360, 720, 1440])
    if random.random() < 0.2:
        p.sigma_method = random.choice(["realized", "ewma", "garch"])
    p.sigma_scale = jitter_f(p.sigma_scale, 0.3, 3.0)
    p.entry_threshold = jitter_f(p.entry_threshold, 0.02, 0.30)
    p.min_entry_min = jitter_i(p.min_entry_min, 0, min(3, max_e - 1))
    p.max_entry_min = jitter_i(p.max_entry_min, max(2, max_e - 4), max_e)
    p.profit_target = jitter_f(p.profit_target, 0.01, 0.30)
    p.stop_loss = jitter_f(p.stop_loss, 0.02, 0.30)
    p.max_hold_min = jitter_i(p.max_hold_min, 1, max(2, p.window_min - 2))
    p.edge_exit = jitter_f(p.edge_exit, 0.0, 0.15)
    p.exit_before_end = jitter_i(p.exit_before_end, 0, min(3, p.window_min - 2))
    p.spread = jitter_f(p.spread, 0.01, 0.05)
    p.position_pct = jitter_f(p.position_pct, 0.02, 0.20)
    p.edge_scaling = jitter_f(p.edge_scaling, 0.5, 8.0)
    p.reentry_cooldown = jitter_i(p.reentry_cooldown, 0, min(3, max_e))

    if random.random() < 0.15:
        p.contrarian = not p.contrarian
    if random.random() < 0.15:
        p.maker = not p.maker
    if random.random() < 0.15:
        p.allow_resolution = not p.allow_resolution

    return p


# ═══════════════════════════════════════════════════════════════════════════════
# WALK-FORWARD VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════


def walk_forward(
    closes: list[float],
    timestamps: list[int],
    log_returns: list[float],
    all_windows_5: list[int],
    all_windows_15: list[int],
    params: B3Params,
) -> tuple[SimResult, SimResult]:
    """Run walk-forward: 70% IS, 30% OOS."""
    windows = all_windows_15 if params.window_min == 15 else all_windows_5
    split = int(len(windows) * WF_IS_PCT)

    is_windows = windows[:split]
    oos_windows = windows[split:]

    is_result = simulate_portfolio(closes, timestamps, log_returns, is_windows, params)
    oos_result = simulate_portfolio(closes, timestamps, log_returns, oos_windows, params)
    return is_result, oos_result


# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════════════════


_log_fh = None


def log(msg: str) -> None:
    global _log_fh
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    if _log_fh is None:
        _log_fh = open(LOG_PATH, "w")
    _log_fh.write(line + "\n")
    _log_fh.flush()


def log_result(label: str, r: SimResult, p: B3Params) -> None:
    log(
        f"  {label}: score={r.score:.1f}  trades={r.total_trades}  "
        f"WR={r.win_rate:.1f}%  PnL=${r.total_pnl:.0f}  "
        f"Sharpe={r.sharpe:.2f}  DD={r.max_drawdown_pct:.1f}%  "
        f"hold={r.avg_hold_min:.1f}min"
    )
    log(
        f"    VOL: window={p.sigma_window}min, method={p.sigma_method}, "
        f"σ_scale={p.sigma_scale}"
    )
    log(
        f"    ENTRY: thr={p.entry_threshold}, min_t={p.min_entry_min}, "
        f"max_t={p.max_entry_min}, {'contrarian' if p.contrarian else 'momentum'}"
    )
    log(
        f"    EXIT: pt={p.profit_target}, sl={p.stop_loss}, "
        f"hold_max={p.max_hold_min}min, edge_exit={p.edge_exit}, "
        f"end={p.exit_before_end}, resolution={'yes' if p.allow_resolution else 'no'}"
    )
    log(
        f"    MKT: spread={p.spread}, {'maker' if p.maker else 'taker'}, "
        f"pos%={p.position_pct}, edge_scale={p.edge_scaling}, "
        f"window={p.window_min}min, cooldown={p.reentry_cooldown}"
    )
    exits = r.exit_reasons
    log(f"    EXITS: {exits}")


# ═══════════════════════════════════════════════════════════════════════════════
# SERIALIZATION
# ═══════════════════════════════════════════════════════════════════════════════


def serialize_trial(
    gen: int, trial: int, params: B3Params, result: SimResult,
) -> dict:
    return {
        "gen": gen,
        "trial": trial,
        "params": asdict(params),
        "score": result.score,
        "total_pnl": result.total_pnl,
        "total_trades": result.total_trades,
        "win_rate": result.win_rate,
        "sharpe": result.sharpe,
        "max_drawdown_pct": result.max_drawdown_pct,
        "avg_hold_min": result.avg_hold_min,
        "total_fees": result.total_fees,
        "exit_reasons": result.exit_reasons,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN SWEEP
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    log("Strategy B3 — Binance Oracle Scalper Autoresearch")
    log("=" * 60)
    log("Loading data...")

    closes, timestamps = load_klines()
    log_returns = precompute_log_returns(closes)

    # Build windows for both 5min and 15min
    windows_5 = build_windows(closes, timestamps, 5)
    windows_15 = build_windows(closes, timestamps, 15)
    log(f"Windows: {len(windows_5)} × 5min, {len(windows_15)} × 15min")

    all_results: list[dict] = []
    best_score = 0.0
    best_params: B3Params | None = None

    # ── GEN 0: RANDOM SEARCH ──
    log(f"\n=== GEN 0: RANDOM SEARCH ({GEN0_TRIALS} trials) ===")
    gen0_start = time.time()

    for i in range(GEN0_TRIALS):
        p = random_params()
        windows = windows_15 if p.window_min == 15 else windows_5

        r = simulate_portfolio(closes, timestamps, log_returns, windows, p)
        all_results.append(serialize_trial(0, i, p, r))

        if r.score > best_score:
            best_score = r.score
            best_params = p
            log_result(f"★ NEW BEST #{i}", r, p)

    gen0_time = time.time() - gen0_start
    log(f"Gen 0 done in {gen0_time:.0f}s. Best score: {best_score:.1f}")

    # ── TOP-K from Gen 0 ──
    all_results.sort(key=lambda x: x["score"], reverse=True)
    top_k_params: list[B3Params] = []
    for entry in all_results[:TOP_K]:
        top_k_params.append(B3Params(**entry["params"]))

    top_scores = [f"{e['score']:.1f}" for e in all_results[:TOP_K]]
    log(f"\nTop-{TOP_K} scores: {top_scores}")

    # ── GEN 1: FINE-TUNE ──
    log(f"\n=== GEN 1: FINE-TUNE TOP-{TOP_K} ({GEN1_TRIALS} trials) ===")
    gen1_start = time.time()

    for i in range(GEN1_TRIALS):
        base = random.choice(top_k_params)
        p = mutate_params(base, intensity=0.25)
        windows = windows_15 if p.window_min == 15 else windows_5

        r = simulate_portfolio(closes, timestamps, log_returns, windows, p)
        all_results.append(serialize_trial(1, i, p, r))

        if r.score > best_score:
            best_score = r.score
            best_params = p
            log_result(f"★ NEW BEST gen1#{i}", r, p)

    gen1_time = time.time() - gen1_start
    log(f"Gen 1 done in {gen1_time:.0f}s. Best score: {best_score:.1f}")

    # Refresh top-K
    all_results.sort(key=lambda x: x["score"], reverse=True)
    top_k_params = [B3Params(**e["params"]) for e in all_results[:TOP_K]]

    # ── GEN 2: EXIT + WINDOW TUNING ──
    log(f"\n=== GEN 2: EXIT + WINDOW TUNING ({GEN2_TRIALS} trials) ===")
    gen2_start = time.time()

    for i in range(GEN2_TRIALS):
        base = random.choice(top_k_params[:5])  # Top 5 only
        p = mutate_params(base, intensity=0.15)
        # Occasionally force different window size
        if random.random() < 0.3:
            new_wm = 5 if p.window_min == 15 else 15
            p.window_min = new_wm
            max_e = new_wm - 2
            p.max_entry_min = min(p.max_entry_min, max_e)
            p.min_entry_min = min(p.min_entry_min, max_e - 1)
            p.max_hold_min = min(p.max_hold_min, max(2, new_wm - 2))
            p.exit_before_end = min(p.exit_before_end, new_wm - 2)
            p.reentry_cooldown = min(p.reentry_cooldown, max_e)

        # Focus on exit params mutation
        p.profit_target = round(random.uniform(0.01, 0.25), 3)
        p.stop_loss = round(random.uniform(0.02, 0.25), 3)
        p.edge_exit = round(random.uniform(0.0, 0.12), 3)
        p.exit_before_end = random.randint(0, min(3, p.window_min - 2))
        p.allow_resolution = random.choice([True, False])

        windows = windows_15 if p.window_min == 15 else windows_5
        r = simulate_portfolio(closes, timestamps, log_returns, windows, p)
        all_results.append(serialize_trial(2, i, p, r))

        if r.score > best_score:
            best_score = r.score
            best_params = p
            log_result(f"★ NEW BEST gen2#{i}", r, p)

    gen2_time = time.time() - gen2_start
    log(f"Gen 2 done in {gen2_time:.0f}s. Best score: {best_score:.1f}")

    # ── WALK-FORWARD VALIDATION ──
    all_results.sort(key=lambda x: x["score"], reverse=True)
    top_5 = [B3Params(**e["params"]) for e in all_results[:5]]

    log(f"\n=== WALK-FORWARD VALIDATION (top 5) ===")
    wf_results = []

    for rank, p in enumerate(top_5):
        is_r, oos_r = walk_forward(
            closes, timestamps, log_returns, windows_5, windows_15, p,
        )
        wf_results.append({
            "rank": rank + 1,
            "params": asdict(p),
            "is_score": is_r.score,
            "is_pnl": is_r.total_pnl,
            "is_trades": is_r.total_trades,
            "is_wr": is_r.win_rate,
            "is_sharpe": is_r.sharpe,
            "is_dd": is_r.max_drawdown_pct,
            "oos_score": oos_r.score,
            "oos_pnl": oos_r.total_pnl,
            "oos_trades": oos_r.total_trades,
            "oos_wr": oos_r.win_rate,
            "oos_sharpe": oos_r.sharpe,
            "oos_dd": oos_r.max_drawdown_pct,
        })

        log(f"\n  #{rank+1} (window={p.window_min}min, "
            f"{'contrarian' if p.contrarian else 'momentum'}, "
            f"{'maker' if p.maker else 'taker'}):")
        log(f"    IS:  score={is_r.score:.1f}  trades={is_r.total_trades}  "
            f"WR={is_r.win_rate:.1f}%  PnL=${is_r.total_pnl:.0f}  "
            f"Sharpe={is_r.sharpe:.2f}  DD={is_r.max_drawdown_pct:.1f}%")
        log(f"    OOS: score={oos_r.score:.1f}  trades={oos_r.total_trades}  "
            f"WR={oos_r.win_rate:.1f}%  PnL=${oos_r.total_pnl:.0f}  "
            f"Sharpe={oos_r.sharpe:.2f}  DD={oos_r.max_drawdown_pct:.1f}%")
        log(f"    Params: entry_thr={p.entry_threshold}, σ_scale={p.sigma_scale}, "
            f"σ_win={p.sigma_window}, pt={p.profit_target}, sl={p.stop_loss}")
        log(f"    Exits: {oos_r.exit_reasons}")

    # ── SAVE ──
    output = {
        "strategy": "B3",
        "description": "Binance Oracle Scalper — BTC 5/15min Up/Down markets",
        "data": f"{len(closes):,} Binance 1-min klines, {len(windows_5)} × 5min + {len(windows_15)} × 15min windows",
        "initial_capital": INITIAL_CAPITAL,
        "generations": {
            "gen0": GEN0_TRIALS,
            "gen1": GEN1_TRIALS,
            "gen2": GEN2_TRIALS,
        },
        "total_trials": len(all_results),
        "best_params": asdict(best_params) if best_params else None,
        "best_score": best_score,
        "walk_forward": wf_results,
        "top_20": [
            {
                "rank": i + 1,
                "score": e["score"],
                "pnl": e["total_pnl"],
                "trades": e["total_trades"],
                "wr": e["win_rate"],
                "sharpe": e["sharpe"],
                "dd": e["max_drawdown_pct"],
                "params": e["params"],
            }
            for i, e in enumerate(all_results[:20])
        ],
    }

    with open(SWEEP_PATH, "w") as f:
        json.dump(output, f, indent=2)

    log(f"\nResults saved to {SWEEP_PATH}")
    log(f"Log saved to {LOG_PATH}")
    log(f"\nTotal time: Gen0={gen0_time:.0f}s, Gen1={gen1_time:.0f}s, Gen2={gen2_time:.0f}s")

    if _log_fh:
        _log_fh.close()


if __name__ == "__main__":
    main()
