"""
B3 Improvement Study — Stop Loss & Exit Optimization
=====================================================

Systematic study of B3 scalper modifications to improve paper trading alignment.

Key findings from 139 paper trades:
  - Paper WR=43.9% vs Backtest OOS WR=52.1% (-8.2%)
  - Paper avg loss=$13.31 vs expected ~$10 (+33%)
  - Stop loss overshoot: threshold=0.038 but median FV drop=0.102 (2.7x)
  - Missing resolution_wins: 28.4% in backtest vs 2.9% in paper
  - 30%+ edge trades are unprofitable (36.8% WR, -$3.42/trade)
  - Paper checks exits every 10s (backtest every 60s) — catches intra-minute dips

Tests:
  A. Baseline Config #1 (reproduce)
  B. Wider stop loss (0.06, 0.08, 0.10, 0.12, 0.15)
  C. BTC-price-based stop (linear, no CDF nonlinearity)
  D. Max edge cap (skip 30%+ edge entries)
  E. Stop loss penalty (simulate paper-like overshoot in backtest)
  F. Combinations of best individual improvements

All tests use the same walk-forward 70/30 split.
No changes to entry logic (same signals, same trade count except D).

Usage:
    python3 research/innovations/b3_improvement_study.py
"""

from __future__ import annotations

import json
import math
import sqlite3
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════════
# PATHS & CONSTANTS (from sweep_b3_scalper.py)
# ═══════════════════════════════════════════════════════════════════════════════

DATA_DIR = Path(__file__).parent.parent / "data"
DB_PATH = DATA_DIR / "crypto_price_pmd.sqlite"
LOG_PATH = Path(__file__).parent.parent / "b3_improvement_study_log.txt"

INITIAL_CAPITAL = 1000.0
MIN_ORDER_SIZE = 5.0
MAX_BET_SIZE = 100.0
MAX_SHARES = 500
MIN_ENTRY_MKT_FV = 0.15
MAX_ENTRY_MKT_FV = 0.85
SIGMA_FLOOR = 0.00005
MIN_TRADES_FOR_SCORE = 30
WF_IS_PCT = 0.70

_SQRT2 = math.sqrt(2.0)
_log_lines: list[str] = []


def log(msg: str = "") -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}" if msg else ""
    print(line)
    _log_lines.append(line)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / _SQRT2))


def _taker_fee(price: float) -> float:
    p = min(price, 1.0 - price)
    return 0.10 * p * p


def _maker_rebate(price: float) -> float:
    return 0.20 * _taker_fee(price)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════════════════════════════════


def load_klines() -> tuple[list[float], list[int]]:
    conn = sqlite3.connect(str(DB_PATH))
    rows = conn.execute(
        "SELECT ts, close FROM binance_klines "
        "WHERE symbol='BTCUSDT' ORDER BY ts"
    ).fetchall()
    conn.close()
    timestamps = [r[0] for r in rows]
    closes = [r[1] for r in rows]
    log(f"Loaded {len(closes):,} BTCUSDT 1-min klines")
    log(f"  Range: {datetime.utcfromtimestamp(timestamps[0]).strftime('%Y-%m-%d')} → "
        f"{datetime.utcfromtimestamp(timestamps[-1]).strftime('%Y-%m-%d')}")
    return closes, timestamps


def build_windows(
    closes: list[float], timestamps: list[int], window_min: int,
) -> list[int]:
    indices = []
    i = 0
    while i + window_min < len(closes):
        gap = timestamps[i + 1] - timestamps[i]
        if gap <= 120:
            indices.append(i)
        i += window_min
    return indices


def precompute_log_returns(closes: list[float]) -> list[float]:
    lr = [0.0]
    for i in range(1, len(closes)):
        if closes[i] > 0 and closes[i - 1] > 0:
            lr.append(math.log(closes[i] / closes[i - 1]))
        else:
            lr.append(0.0)
    return lr


def compute_sigma(
    log_returns: list[float], end_idx: int, window: int, method: str,
) -> float:
    start = max(1, end_idx - window)
    rets = log_returns[start:end_idx]
    if len(rets) < 10:
        return SIGMA_FLOOR
    n = len(rets)
    mean = sum(rets) / n
    var = sum((r - mean) ** 2 for r in rets) / (n - 1)
    return max(math.sqrt(var), SIGMA_FLOOR)


# ═══════════════════════════════════════════════════════════════════════════════
# SIMULATION WITH MODIFICATIONS
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class Trade:
    direction: int
    entry_min: int
    exit_min: int
    entry_fv: float
    exit_fv: float
    shares: float
    gross_pnl: float
    fees: float
    net_pnl: float
    exit_reason: str
    entry_edge: float = 0.0
    fv_drop: float = 0.0  # How far past stop the FV actually dropped


@dataclass
class ModConfig:
    """Modification config — what to change vs baseline."""
    name: str = "baseline"

    # Stop loss
    stop_loss: float = 0.038          # FV-based stop threshold
    use_btc_stop: bool = False        # Use BTC-% stop instead of FV stop
    btc_stop_pct: float = 0.0015     # BTC reversal % for btc stop (0.15%)

    # Entry filters
    max_edge: float = 1.0             # Skip if edge > this (1.0 = no cap)
    max_entry_fv: float = 0.85        # Skip if market_fv > this

    # Stop overshoot penalty (simulate paper-like behavior)
    stop_penalty_mult: float = 1.0    # Multiply stop FV drop by this (>1 = worse)

    # Profit target
    profit_target: float = 0.207

    # Edge exit
    edge_exit: float = 0.076

    # Max hold
    max_hold_min: int = 3

    # Original params (fixed)
    sigma_window: int = 720
    sigma_scale: float = 0.644
    entry_threshold: float = 0.095
    min_entry_min: int = 2
    max_entry_min: int = 2
    spread: float = 0.01
    position_pct: float = 0.067
    edge_scaling: float = 4.838
    window_min: int = 5
    reentry_cooldown: int = 3
    allow_resolution: bool = True


def simulate_window(
    closes: list[float],
    start_idx: int,
    sigma_per_min: float,
    capital: float,
    m: ModConfig,
) -> list[Trade]:
    wm = m.window_min
    S_start = closes[start_idx]
    if S_start <= 0:
        return []

    max_entry = min(m.max_entry_min, wm - 1)
    if max_entry < m.min_entry_min:
        return []

    trades: list[Trade] = []
    position = None
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

        # Market FV (σ_scale=1.0)
        sigma_rem_true = sigma_per_min * sqrt_t
        if sigma_rem_true > 1e-12:
            market_up = _norm_cdf(log_ratio / sigma_rem_true)
        else:
            market_up = 1.0 if log_ratio > 0 else 0.0 if log_ratio < 0 else 0.5
        market_up = max(0.02, min(0.98, market_up))

        # Signal FV (σ_scale from params)
        sigma_rem_model = sigma_per_min * m.sigma_scale * sqrt_t
        if sigma_rem_model > 1e-12:
            signal_up = _norm_cdf(log_ratio / sigma_rem_model)
        else:
            signal_up = 1.0 if log_ratio > 0 else 0.0 if log_ratio < 0 else 0.5
        signal_up = max(0.01, min(0.99, signal_up))

        if position is None:
            # ══ ENTRY ══
            if t - last_exit_min < m.reentry_cooldown:
                continue
            if t < m.min_entry_min or t > max_entry:
                continue

            signal_dev = signal_up - 0.50
            if abs(signal_dev) < m.entry_threshold:
                continue

            direction = 1 if signal_dev > 0 else -1
            entry_mkt_fv = market_up if direction == 1 else (1.0 - market_up)
            edge = abs(signal_dev)

            # ── MAX EDGE CAP (modification D) ──
            if edge > m.max_edge:
                continue

            # ── MAX ENTRY FV (modification) ──
            if entry_mkt_fv > m.max_entry_fv or entry_mkt_fv < (1.0 - m.max_entry_fv):
                continue

            if entry_mkt_fv < MIN_ENTRY_MKT_FV or entry_mkt_fv > MAX_ENTRY_MKT_FV:
                continue

            entry_price = entry_mkt_fv + m.spread / 2
            entry_fee = -_maker_rebate(entry_mkt_fv)

            raw_pct = min(m.position_pct, edge * m.edge_scaling)
            bet_size = min(capital * raw_pct, MAX_BET_SIZE)
            if bet_size < MIN_ORDER_SIZE:
                continue
            shares = bet_size / entry_price if entry_price > 0.01 else 0
            shares = min(shares, MAX_SHARES)
            if shares < 1:
                continue

            position = (direction, entry_mkt_fv, t, shares, entry_fee, edge, S_now)

        else:
            # ══ EXIT CHECK ══
            direction, entry_mkt_fv, entry_t, shares, entry_fee, entry_edge, S_entry = position
            hold_min = t - entry_t

            pos_mkt_fv = market_up if direction == 1 else (1.0 - market_up)
            unrealized = pos_mkt_fv - entry_mkt_fv

            pos_signal_fv = signal_up if direction == 1 else (1.0 - signal_up)

            should_exit = False
            reason = ""

            # ── PROFIT TARGET ──
            if unrealized >= m.profit_target:
                should_exit, reason = True, "profit"

            # ── STOP LOSS — FV-based or BTC-based ──
            elif m.use_btc_stop:
                # BTC-price-based stop: did BTC reverse X% from our direction?
                if direction == 1:
                    # Long Up: BTC should go up. Stop if BTC dropped btc_stop_pct from entry
                    btc_change = (S_now - S_entry) / S_entry
                    if btc_change <= -m.btc_stop_pct:
                        should_exit, reason = True, "stop"
                else:
                    # Long Down: BTC should go down. Stop if BTC rose btc_stop_pct
                    btc_change = (S_now - S_entry) / S_entry
                    if btc_change >= m.btc_stop_pct:
                        should_exit, reason = True, "stop"
            elif unrealized <= -m.stop_loss:
                should_exit, reason = True, "stop"

            # ── TIME / EDGE exits ──
            if not should_exit:
                if hold_min >= m.max_hold_min:
                    should_exit, reason = True, "time"
                elif abs(pos_signal_fv - 0.50) < m.edge_exit and hold_min >= 1:
                    should_exit, reason = True, "edge_gone"

            if should_exit:
                exit_mkt_fv = pos_mkt_fv

                # ── STOP PENALTY (simulate paper-like overshoot) ──
                if reason == "stop" and m.stop_penalty_mult > 1.0:
                    fv_drop = entry_mkt_fv - exit_mkt_fv
                    extra_drop = fv_drop * (m.stop_penalty_mult - 1.0)
                    exit_mkt_fv = max(0.02, exit_mkt_fv - extra_drop)

                exit_price = exit_mkt_fv - m.spread / 2
                exit_fee = -_maker_rebate(exit_mkt_fv)

                gross = (exit_price - (entry_mkt_fv + m.spread / 2)) * shares
                fee_total = (entry_fee + exit_fee) * shares
                net = gross - fee_total

                trades.append(Trade(
                    direction=direction,
                    entry_min=entry_t, exit_min=t,
                    entry_fv=entry_mkt_fv, exit_fv=exit_mkt_fv,
                    shares=shares,
                    gross_pnl=gross, fees=fee_total, net_pnl=net,
                    exit_reason=reason,
                    entry_edge=entry_edge,
                    fv_drop=entry_mkt_fv - exit_mkt_fv if reason == "stop" else 0.0,
                ))
                position = None
                last_exit_min = t

    # ── Handle unresolved position ──
    if position is not None:
        direction, entry_mkt_fv, entry_t, shares, entry_fee, entry_edge, S_entry = position
        if m.allow_resolution:
            resolved_up = closes[start_idx + wm] >= S_start
            won = (direction == 1 and resolved_up) or (direction == -1 and not resolved_up)
            exit_mkt_fv = 1.0 if won else 0.0
            exit_fee = 0.0

            gross = (exit_mkt_fv - (entry_mkt_fv + m.spread / 2)) * shares
            fee_total = entry_fee * shares
            net = gross - fee_total
            trades.append(Trade(
                direction=direction,
                entry_min=entry_t, exit_min=wm,
                entry_fv=entry_mkt_fv, exit_fv=exit_mkt_fv,
                shares=shares,
                gross_pnl=gross, fees=fee_total, net_pnl=net,
                exit_reason="resolution_win" if won else "resolution_lose",
                entry_edge=entry_edge,
            ))
        else:
            exit_mkt_fv = market_up if direction == 1 else (1.0 - market_up)
            exit_price = exit_mkt_fv - m.spread / 2
            exit_fee = -_maker_rebate(exit_mkt_fv)

            gross = (exit_price - (entry_mkt_fv + m.spread / 2)) * shares
            fee_total = (entry_fee + exit_fee) * shares
            net = gross - fee_total
            trades.append(Trade(
                direction=direction,
                entry_min=entry_t, exit_min=wm - 1,
                entry_fv=entry_mkt_fv, exit_fv=exit_mkt_fv,
                shares=shares,
                gross_pnl=gross, fees=fee_total, net_pnl=net,
                exit_reason="forced_end",
                entry_edge=entry_edge,
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
    score: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_stop_fv_drop: float = 0.0  # For analysis


def simulate_portfolio(
    closes: list[float],
    timestamps: list[int],
    log_returns: list[float],
    windows: list[int],
    mod: ModConfig,
) -> SimResult:
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

        sigma = compute_sigma(log_returns, start_idx, mod.sigma_window, "realized")
        trades = simulate_window(closes, start_idx, sigma, capital, mod)

        for tr in trades:
            all_trades.append(tr)
            capital += tr.net_pnl
            total_fees += tr.fees

            ts = timestamps[start_idx + tr.entry_min]
            day = datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d")
            daily_pnl[day] += tr.net_pnl
            exit_reasons[tr.exit_reason] += 1

            if capital > peak_capital:
                peak_capital = capital
            dd = (peak_capital - capital) / peak_capital if peak_capital > 0 else 0
            if dd > max_dd:
                max_dd = dd

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
        winners = [t.net_pnl for t in all_trades if t.net_pnl > 0]
        losers = [t.net_pnl for t in all_trades if t.net_pnl <= 0]
        result.avg_win = sum(winners) / len(winners) if winners else 0
        result.avg_loss = sum(losers) / len(losers) if losers else 0
        stop_drops = [t.fv_drop for t in all_trades if t.exit_reason == "stop" and t.fv_drop > 0]
        result.avg_stop_fv_drop = sum(stop_drops) / len(stop_drops) if stop_drops else 0

    daily_vals = list(daily_pnl.values())
    if len(daily_vals) >= 5:
        mean_d = sum(daily_vals) / len(daily_vals)
        diffs = [(d - mean_d) for d in daily_vals]
        var_d = sum(d * d for d in diffs) / (len(daily_vals) - 1)
        std_d = math.sqrt(max(var_d, 1e-10))
        if std_d > 1e-8:
            result.sharpe = min((mean_d / std_d) * math.sqrt(365), 100.0)

    result.score = experiment_score(result)
    return result


def experiment_score(r: SimResult) -> float:
    if r.total_trades < MIN_TRADES_FOR_SCORE:
        return 0.0
    roi = r.total_pnl / INITIAL_CAPITAL * 100
    if roi <= 0:
        return 0.0
    trade_factor = 1.0 + math.log10(max(r.total_trades, 1))
    dd_penalty = max(0.0, 1.0 - r.max_drawdown_pct / 100.0)
    sharpe_factor = min(max(r.sharpe, 0), 15) / 15.0
    return roi * trade_factor * dd_penalty * sharpe_factor


# ═══════════════════════════════════════════════════════════════════════════════
# WALK-FORWARD EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════


def walk_forward(
    closes: list[float],
    timestamps: list[int],
    log_returns: list[float],
    all_windows: list[int],
    mod: ModConfig,
) -> tuple[SimResult, SimResult]:
    split = int(len(all_windows) * WF_IS_PCT)
    is_windows = all_windows[:split]
    oos_windows = all_windows[split:]

    is_result = simulate_portfolio(closes, timestamps, log_returns, is_windows, mod)
    oos_result = simulate_portfolio(closes, timestamps, log_returns, oos_windows, mod)
    return is_result, oos_result


def log_result(label: str, r: SimResult, mod: ModConfig) -> None:
    log(f"  {label}: score={r.score:.1f}  trades={r.total_trades}  "
        f"WR={r.win_rate:.1f}%  PnL=${r.total_pnl:.0f}  "
        f"Sharpe={r.sharpe:.2f}  DD={r.max_drawdown_pct:.1f}%  "
        f"hold={r.avg_hold_min:.1f}min")
    log(f"    AvgWin=${r.avg_win:.2f}  AvgLoss=${r.avg_loss:.2f}  "
        f"W/L={abs(r.avg_win / r.avg_loss) if r.avg_loss != 0 else 999:.2f}  "
        f"AvgStopDrop={r.avg_stop_fv_drop:.3f}")
    log(f"    EXITS: {r.exit_reasons}")


# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════


def define_experiments() -> list[ModConfig]:
    experiments = []

    # ═══ A. BASELINE (Config #1 exact) ═══
    experiments.append(ModConfig(name="A_baseline"))

    # ═══ B. WIDER STOP LOSS (same entries, wider stop) ═══
    for sl in [0.06, 0.08, 0.10, 0.12, 0.15, 0.20]:
        experiments.append(ModConfig(name=f"B_stop_{sl}", stop_loss=sl))

    # ═══ C. BTC-PRICE-BASED STOP (linear, no CDF nonlinearity) ═══
    for pct in [0.0005, 0.001, 0.0015, 0.002, 0.003, 0.004, 0.005]:
        experiments.append(ModConfig(
            name=f"C_btc_stop_{pct*100:.2f}pct",
            use_btc_stop=True,
            btc_stop_pct=pct,
            stop_loss=99.0,  # Disable FV stop when using BTC stop
        ))

    # ═══ D. MAX EDGE CAP (reduce extreme-FV entries) ═══
    for me in [0.20, 0.25, 0.30, 0.35, 0.40]:
        experiments.append(ModConfig(name=f"D_max_edge_{me}", max_edge=me))

    # ═══ E. STOP PENALTY (simulate paper-like overshoot) ═══
    # This is for ANALYSIS — shows what backtest looks like if stops overshoot
    for mult in [1.5, 2.0, 2.7, 3.0]:
        experiments.append(ModConfig(
            name=f"E_penalty_{mult}x",
            stop_penalty_mult=mult,
        ))

    # ═══ F. NO RESOLUTION (disable allow_resolution) ═══
    experiments.append(ModConfig(name="F_no_resolution", allow_resolution=False))

    # ═══ G. WIDER STOP + NO RESOLUTION (paper-like: stop wider, no holdthrough) ═══
    for sl in [0.08, 0.10, 0.15]:
        experiments.append(ModConfig(
            name=f"G_wide_stop_{sl}_no_res",
            stop_loss=sl,
            allow_resolution=False,
        ))

    # ═══ H. BTC STOP + wider profit target ═══
    for pct in [0.001, 0.0015, 0.002]:
        for pt in [0.207, 0.25, 0.30]:
            experiments.append(ModConfig(
                name=f"H_btc_{pct*100:.1f}pct_pt_{pt}",
                use_btc_stop=True,
                btc_stop_pct=pct,
                stop_loss=99.0,
                profit_target=pt,
            ))

    # ═══ I. WIDER STOP + MAX EDGE CAP (combo) ═══
    for sl in [0.08, 0.10, 0.12]:
        for me in [0.25, 0.30, 0.35]:
            experiments.append(ModConfig(
                name=f"I_stop_{sl}_edge_{me}",
                stop_loss=sl,
                max_edge=me,
            ))

    # ═══ J. BTC STOP + MAX EDGE CAP (combo) ═══
    for pct in [0.001, 0.0015, 0.002]:
        for me in [0.25, 0.30, 0.35]:
            experiments.append(ModConfig(
                name=f"J_btc_{pct*100:.1f}pct_edge_{me}",
                use_btc_stop=True,
                btc_stop_pct=pct,
                stop_loss=99.0,
                max_edge=me,
            ))

    # ═══ K. ADJUSTED PROFIT TARGET (keep stops, change PT) ═══
    for pt in [0.15, 0.18, 0.25, 0.30]:
        experiments.append(ModConfig(name=f"K_pt_{pt}", profit_target=pt))

    # ═══ L. WIDER STOP + PENALTY (realistic pessimistic scenario) ═══
    for sl in [0.08, 0.10, 0.12, 0.15]:
        experiments.append(ModConfig(
            name=f"L_stop_{sl}_pen_2x",
            stop_loss=sl,
            stop_penalty_mult=2.0,  # Simulate 2x overshoot
        ))

    return experiments


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    log("=" * 80)
    log("B3 IMPROVEMENT STUDY")
    log("=" * 80)

    t0 = time.time()

    # Load data
    closes, timestamps = load_klines()
    log_returns = precompute_log_returns(closes)
    windows_5 = build_windows(closes, timestamps, 5)
    log(f"Built {len(windows_5):,} × 5-min windows")

    split = int(len(windows_5) * WF_IS_PCT)
    log(f"Walk-forward split: IS={split:,} windows, OOS={len(windows_5)-split:,} windows")

    # Define experiments
    experiments = define_experiments()
    log(f"\n{len(experiments)} experiments to run\n")

    # Run all experiments
    results: list[dict] = []
    baseline_oos = None

    for i, mod in enumerate(experiments):
        log(f"\n{'='*60}")
        log(f"[{i+1}/{len(experiments)}] {mod.name}")
        log(f"{'='*60}")

        is_r, oos_r = walk_forward(closes, timestamps, log_returns, windows_5, mod)
        log_result("IS ", is_r, mod)
        log_result("OOS", oos_r, mod)

        if mod.name == "A_baseline":
            baseline_oos = oos_r
            log(f"  → BASELINE SET: OOS score={oos_r.score:.1f}")

        # Compare vs baseline
        if baseline_oos and mod.name != "A_baseline":
            d_score = oos_r.score - baseline_oos.score
            d_pnl = oos_r.total_pnl - baseline_oos.total_pnl
            d_trades = oos_r.total_trades - baseline_oos.total_trades
            d_wr = oos_r.win_rate - baseline_oos.win_rate
            log(f"  → vs BASELINE: score {d_score:+.1f}  PnL ${d_pnl:+.0f}  "
                f"trades {d_trades:+d}  WR {d_wr:+.1f}%")

        results.append({
            "name": mod.name,
            "is_score": round(is_r.score, 1),
            "oos_score": round(oos_r.score, 1),
            "is_trades": is_r.total_trades,
            "oos_trades": oos_r.total_trades,
            "is_wr": round(is_r.win_rate, 1),
            "oos_wr": round(oos_r.win_rate, 1),
            "is_pnl": round(is_r.total_pnl),
            "oos_pnl": round(oos_r.total_pnl),
            "oos_sharpe": round(oos_r.sharpe, 2),
            "oos_dd": round(oos_r.max_drawdown_pct, 1),
            "oos_avg_win": round(oos_r.avg_win, 2),
            "oos_avg_loss": round(oos_r.avg_loss, 2),
            "oos_avg_stop_drop": round(oos_r.avg_stop_fv_drop, 4),
            "oos_exits": oos_r.exit_reasons,
        })

    elapsed = time.time() - t0

    # ═══ SUMMARY TABLE ═══
    log("\n" + "=" * 80)
    log("SUMMARY — SORTED BY OOS SCORE (descending)")
    log("=" * 80)
    log(f"{'Name':<35} {'OOS Score':>10} {'Trades':>7} {'WR%':>6} {'PnL$':>8} "
        f"{'Sharpe':>7} {'DD%':>6} {'AvgW$':>7} {'AvgL$':>7} {'StopDr':>7}")
    log("-" * 105)

    sorted_results = sorted(results, key=lambda x: x["oos_score"], reverse=True)
    for r in sorted_results:
        marker = " ★" if r["name"] == "A_baseline" else ""
        log(f"{r['name']:<35} {r['oos_score']:>10.1f} {r['oos_trades']:>7} "
            f"{r['oos_wr']:>6.1f} {r['oos_pnl']:>8} {r['oos_sharpe']:>7.2f} "
            f"{r['oos_dd']:>6.1f} {r['oos_avg_win']:>7.2f} {r['oos_avg_loss']:>7.02f} "
            f"{r['oos_avg_stop_drop']:>7.4f}{marker}")

    # ═══ TOP 10 vs BASELINE ═══
    log("\n" + "=" * 80)
    log("TOP 10 IMPROVEMENTS vs BASELINE")
    log("=" * 80)
    baseline_score = next(r["oos_score"] for r in results if r["name"] == "A_baseline")
    baseline_trades = next(r["oos_trades"] for r in results if r["name"] == "A_baseline")
    baseline_pnl = next(r["oos_pnl"] for r in results if r["name"] == "A_baseline")

    improvements = [r for r in sorted_results if r["name"] != "A_baseline"]
    for r in improvements[:10]:
        d_score = r["oos_score"] - baseline_score
        d_trades = r["oos_trades"] - baseline_trades
        d_pnl = r["oos_pnl"] - baseline_pnl
        trade_pct = (d_trades / baseline_trades * 100) if baseline_trades else 0
        log(f"  {r['name']:<33} score {d_score:+8.1f}  PnL ${d_pnl:+6}  "
            f"trades {d_trades:+5} ({trade_pct:+5.1f}%)  WR={r['oos_wr']:.1f}%")

    # ═══ CATEGORY WINNERS ═══
    log("\n" + "=" * 80)
    log("CATEGORY WINNERS")
    log("=" * 80)
    categories = set(r["name"].split("_")[0] for r in results)
    for cat in sorted(categories):
        cat_results = [r for r in sorted_results if r["name"].startswith(cat + "_")]
        if cat_results:
            best = cat_results[0]
            log(f"  {cat}: {best['name']:<33} score={best['oos_score']:.1f}  "
                f"PnL=${best['oos_pnl']}  trades={best['oos_trades']}  WR={best['oos_wr']:.1f}%")

    log(f"\n✓ Done in {elapsed:.1f}s")

    # Save results
    results_path = DATA_DIR / "experiments" / "b3_improvement_study.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    log(f"Results saved to {results_path}")

    # Save log
    with open(LOG_PATH, "w") as f:
        f.write("\n".join(_log_lines))
    log(f"Log saved to {LOG_PATH}")


if __name__ == "__main__":
    main()
