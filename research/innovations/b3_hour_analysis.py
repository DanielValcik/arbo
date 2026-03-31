"""
B3 Hour-of-Day Analysis — Is overnight performance systematically worse?
========================================================================

Uses production B3 config (BTC stop) on 89,419 BTCUSDT 1-min klines.
Groups trades by hour of day (UTC + CET) to find time-based patterns.

Walk-forward 70/30 split — reports IS and OOS separately.

Usage:
    python3 research/innovations/b3_hour_analysis.py
"""

from __future__ import annotations

import json
import math
import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════════════════════════════════════════

DATA_DIR = Path(__file__).parent.parent / "data"
DB_PATH = DATA_DIR / "crypto_price_pmd.sqlite"
RESULTS_DIR = DATA_DIR / "experiments"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PATH = RESULTS_DIR / "b3_hour_analysis.json"

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS — Production B3 config (BTC stop)
# ═══════════════════════════════════════════════════════════════════════════════

INITIAL_CAPITAL = 1000.0
MIN_ORDER_SIZE = 5.0
MAX_BET_SIZE = 100.0
MAX_SHARES = 500
MIN_ENTRY_MKT_FV = 0.15
MAX_ENTRY_MKT_FV = 0.85
SIGMA_FLOOR = 0.00005

# Production params from b3_quality_gate.py
SIGMA_WINDOW = 720
SIGMA_SCALE = 0.644
ENTRY_THRESHOLD = 0.095
MIN_ENTRY_MIN = 2
MAX_ENTRY_MIN = 2
PROFIT_TARGET = 0.207
STOP_LOSS = 99.0  # Disabled — BTC stop used instead
MAX_HOLD_MIN = 3
EDGE_EXIT = 0.076
ALLOW_RESOLUTION = True
SPREAD = 0.01
POSITION_PCT = 0.067
EDGE_SCALING = 4.838
WINDOW_MIN = 5
REENTRY_COOLDOWN = 3

# BTC stop
USE_BTC_STOP = True
BTC_STOP_PCT = 0.0015

# Walk-forward
WF_IS_PCT = 0.70

# ═══════════════════════════════════════════════════════════════════════════════
# MATH
# ═══════════════════════════════════════════════════════════════════════════════

_SQRT2 = math.sqrt(2.0)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / _SQRT2))


def _taker_fee(price: float) -> float:
    p = min(price, 1.0 - price)
    return 0.10 * p * p


def _maker_rebate(price: float) -> float:
    return 0.20 * _taker_fee(price)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════


def load_klines(db_path: Path = DB_PATH) -> tuple[list[float], list[int]]:
    conn = sqlite3.connect(str(db_path))
    rows = conn.execute(
        "SELECT ts, close FROM binance_klines "
        "WHERE symbol='BTCUSDT' ORDER BY ts"
    ).fetchall()
    conn.close()
    timestamps = [r[0] for r in rows]
    closes = [r[1] for r in rows]
    print(f"Loaded {len(closes):,} BTCUSDT 1-min klines")
    print(
        f"  Range: {datetime.utcfromtimestamp(timestamps[0]).strftime('%Y-%m-%d')} → "
        f"{datetime.utcfromtimestamp(timestamps[-1]).strftime('%Y-%m-%d')}"
    )
    return closes, timestamps


def build_windows(closes: list[float], timestamps: list[int]) -> list[int]:
    indices = []
    i = 0
    while i + WINDOW_MIN < len(closes):
        gap = timestamps[i + 1] - timestamps[i]
        if gap <= 120:
            indices.append(i)
        i += WINDOW_MIN
    return indices


def precompute_log_returns(closes: list[float]) -> list[float]:
    lr = [0.0]
    for i in range(1, len(closes)):
        if closes[i] > 0 and closes[i - 1] > 0:
            lr.append(math.log(closes[i] / closes[i - 1]))
        else:
            lr.append(0.0)
    return lr


def compute_sigma(log_returns: list[float], end_idx: int) -> float:
    start = max(1, end_idx - SIGMA_WINDOW)
    rets = log_returns[start:end_idx]
    if len(rets) < 10:
        return SIGMA_FLOOR
    n = len(rets)
    mean = sum(rets) / n
    var = sum((r - mean) ** 2 for r in rets) / (n - 1)
    return max(math.sqrt(var), SIGMA_FLOOR)


# ═══════════════════════════════════════════════════════════════════════════════
# SIMULATION
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class Trade:
    direction: int
    entry_min: int
    exit_min: int
    entry_fv: float
    exit_fv: float
    shares: float
    net_pnl: float
    exit_reason: str
    hour_utc: int = 0
    hour_cet: int = 0
    timestamp: int = 0


def simulate_window(
    closes: list[float],
    timestamps: list[int],
    start_idx: int,
    sigma_per_min: float,
    capital: float,
) -> list[Trade]:
    wm = WINDOW_MIN
    S_start = closes[start_idx]
    if S_start <= 0:
        return []

    max_entry = min(MAX_ENTRY_MIN, wm - 1)
    if max_entry < MIN_ENTRY_MIN:
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
        sigma_rem_model = sigma_per_min * SIGMA_SCALE * sqrt_t
        if sigma_rem_model > 1e-12:
            signal_up = _norm_cdf(log_ratio / sigma_rem_model)
        else:
            signal_up = 1.0 if log_ratio > 0 else 0.0 if log_ratio < 0 else 0.5
        signal_up = max(0.01, min(0.99, signal_up))

        if position is None:
            # ══ ENTRY ══
            if t - last_exit_min < REENTRY_COOLDOWN:
                continue
            if t < MIN_ENTRY_MIN or t > max_entry:
                continue

            signal_dev = signal_up - 0.50
            if abs(signal_dev) < ENTRY_THRESHOLD:
                continue

            direction = 1 if signal_dev > 0 else -1
            entry_mkt_fv = market_up if direction == 1 else (1.0 - market_up)

            if entry_mkt_fv < MIN_ENTRY_MKT_FV or entry_mkt_fv > MAX_ENTRY_MKT_FV:
                continue

            entry_price = entry_mkt_fv + SPREAD / 2
            entry_fee = -_maker_rebate(entry_mkt_fv)

            edge = abs(signal_dev)
            raw_pct = min(POSITION_PCT, edge * EDGE_SCALING)
            bet_size = min(capital * raw_pct, MAX_BET_SIZE)
            if bet_size < MIN_ORDER_SIZE:
                continue
            shares = bet_size / entry_price if entry_price > 0.01 else 0
            shares = min(shares, MAX_SHARES)
            if shares < 1:
                continue

            position = (direction, entry_mkt_fv, t, shares, entry_fee, S_now)

        else:
            # ══ EXIT CHECK ══
            direction, entry_mkt_fv, entry_t, shares, entry_fee, S_entry = position
            hold_min = t - entry_t

            pos_mkt_fv = market_up if direction == 1 else (1.0 - market_up)
            unrealized = pos_mkt_fv - entry_mkt_fv

            pos_signal_fv = signal_up if direction == 1 else (1.0 - signal_up)

            should_exit = False
            reason = ""

            # Profit target
            if unrealized >= PROFIT_TARGET:
                should_exit, reason = True, "profit"

            # BTC stop
            elif USE_BTC_STOP:
                btc_change = (S_now - S_entry) / S_entry
                if (direction == 1 and btc_change <= -BTC_STOP_PCT) or (
                    direction == -1 and btc_change >= BTC_STOP_PCT
                ):
                    should_exit, reason = True, "stop"

            # FV stop (disabled)
            elif unrealized <= -STOP_LOSS:
                should_exit, reason = True, "stop"

            # Time / edge exits
            if not should_exit:
                if hold_min >= MAX_HOLD_MIN:
                    should_exit, reason = True, "time"
                elif abs(pos_signal_fv - 0.50) < EDGE_EXIT and hold_min >= 1:
                    should_exit, reason = True, "edge_gone"

            if should_exit:
                exit_mkt_fv = pos_mkt_fv
                exit_price = exit_mkt_fv - SPREAD / 2
                exit_fee = -_maker_rebate(exit_mkt_fv)

                gross = (exit_price - (entry_mkt_fv + SPREAD / 2)) * shares
                fee_total = (entry_fee + exit_fee) * shares
                net = gross - fee_total

                ts = timestamps[start_idx + entry_t]
                dt_utc = datetime.utcfromtimestamp(ts)
                dt_cet = dt_utc + timedelta(hours=1)  # CET = UTC+1

                trades.append(Trade(
                    direction=direction,
                    entry_min=entry_t,
                    exit_min=t,
                    entry_fv=entry_mkt_fv,
                    exit_fv=exit_mkt_fv,
                    shares=shares,
                    net_pnl=net,
                    exit_reason=reason,
                    hour_utc=dt_utc.hour,
                    hour_cet=dt_cet.hour,
                    timestamp=ts,
                ))
                position = None
                last_exit_min = t

    # Handle unresolved position
    if position is not None:
        direction, entry_mkt_fv, entry_t, shares, entry_fee, S_entry = position
        if ALLOW_RESOLUTION:
            resolved_up = closes[start_idx + wm] >= S_start
            won = (direction == 1 and resolved_up) or (direction == -1 and not resolved_up)
            exit_mkt_fv = 1.0 if won else 0.0
            exit_fee = 0.0

            gross = (exit_mkt_fv - (entry_mkt_fv + SPREAD / 2)) * shares
            fee_total = entry_fee * shares
            net = gross - fee_total

            ts = timestamps[start_idx + entry_t]
            dt_utc = datetime.utcfromtimestamp(ts)
            dt_cet = dt_utc + timedelta(hours=1)

            trades.append(Trade(
                direction=direction,
                entry_min=entry_t,
                exit_min=wm,
                entry_fv=entry_mkt_fv,
                exit_fv=exit_mkt_fv,
                shares=shares,
                net_pnl=net,
                exit_reason="resolution_win" if won else "resolution_lose",
                hour_utc=dt_utc.hour,
                hour_cet=dt_cet.hour,
                timestamp=ts,
            ))

    return trades


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class HourStats:
    hour: int
    trades: int = 0
    wins: int = 0
    total_pnl: float = 0.0
    win_pnl: float = 0.0
    loss_pnl: float = 0.0
    exit_reasons: dict = field(default_factory=lambda: defaultdict(int))

    @property
    def wr(self) -> float:
        return self.wins / self.trades * 100 if self.trades > 0 else 0

    @property
    def avg_pnl(self) -> float:
        return self.total_pnl / self.trades if self.trades > 0 else 0

    @property
    def avg_win(self) -> float:
        return self.win_pnl / self.wins if self.wins > 0 else 0

    @property
    def avg_loss(self) -> float:
        losses = self.trades - self.wins
        return self.loss_pnl / losses if losses > 0 else 0

    @property
    def wl_ratio(self) -> float:
        return self.avg_win / abs(self.avg_loss) if self.avg_loss != 0 else 999


def analyze_trades(trades: list[Trade], label: str) -> dict:
    """Analyze trades by hour and produce summary."""

    # Per-hour stats (CET)
    hours: dict[int, HourStats] = {h: HourStats(hour=h) for h in range(24)}

    for tr in trades:
        h = hours[tr.hour_cet]
        h.trades += 1
        h.exit_reasons[tr.exit_reason] += 1
        if tr.net_pnl > 0:
            h.wins += 1
            h.win_pnl += tr.net_pnl
        else:
            h.loss_pnl += tr.net_pnl
        h.total_pnl += tr.net_pnl

    # Print hourly table
    print(f"\n{'='*90}")
    print(f"  {label} — {len(trades)} trades, PnL: ${sum(t.net_pnl for t in trades):,.0f}")
    print(f"{'='*90}")
    print(f"{'Hour':>4}  {'Trades':>6}  {'WR%':>5}  {'TotalPnL':>10}  {'AvgPnL':>8}  "
          f"{'AvgWin':>8}  {'AvgLoss':>8}  {'W/L':>5}  {'Exits'}")
    print("-" * 90)

    for h in range(24):
        hs = hours[h]
        if hs.trades == 0:
            print(f"  {h:02d}   {'—':>6}")
            continue
        exits_str = ", ".join(f"{k}={v}" for k, v in sorted(hs.exit_reasons.items(), key=lambda x: -x[1]))
        marker = " ◄◄◄" if hs.avg_pnl < -2 else " ★" if hs.avg_pnl > 8 else ""
        print(
            f"  {h:02d}   {hs.trades:>6}  {hs.wr:>5.1f}  ${hs.total_pnl:>9,.1f}  "
            f"${hs.avg_pnl:>7.2f}  ${hs.avg_win:>7.2f}  ${hs.avg_loss:>7.2f}  "
            f"{hs.wl_ratio:>5.2f}  {exits_str}{marker}"
        )

    # Day vs night summary
    day_trades = [t for t in trades if 8 <= t.hour_cet <= 22]
    night_trades = [t for t in trades if t.hour_cet < 8 or t.hour_cet > 22]

    for name, subset in [("DAYTIME (8-22h CET)", day_trades), ("OVERNIGHT (23-7h CET)", night_trades)]:
        if not subset:
            continue
        n = len(subset)
        wins = sum(1 for t in subset if t.net_pnl > 0)
        pnl = sum(t.net_pnl for t in subset)
        avg_w = sum(t.net_pnl for t in subset if t.net_pnl > 0) / max(wins, 1)
        losses = n - wins
        avg_l = sum(t.net_pnl for t in subset if t.net_pnl <= 0) / max(losses, 1)
        wl = avg_w / abs(avg_l) if avg_l != 0 else 999

        reasons = defaultdict(int)
        for t in subset:
            reasons[t.exit_reason] += 1
        exits = ", ".join(f"{k}={v}" for k, v in sorted(reasons.items(), key=lambda x: -x[1]))

        print(f"\n  {name}: {n} trades, WR {wins/n*100:.1f}%, PnL ${pnl:,.0f} (${pnl/n:.2f}/trade)")
        print(f"    Avg win: ${avg_w:.2f}, Avg loss: ${avg_l:.2f}, W/L ratio: {wl:.2f}")
        print(f"    Exits: {exits}")

    # What-if: filter out worst hours
    print(f"\n  WHAT-IF analysis — removing hours:")
    for exclude_hours in [
        [7],
        [4, 7],
        [1, 4, 7],
        [1, 4, 7, 22],
        list(range(0, 8)),  # All night (0-7h)
        list(range(2, 8)),  # Late night only (2-7h)
    ]:
        filtered = [t for t in trades if t.hour_cet not in exclude_hours]
        if not filtered:
            continue
        n = len(filtered)
        pnl = sum(t.net_pnl for t in filtered)
        wins = sum(1 for t in filtered if t.net_pnl > 0)
        label_hrs = ",".join(str(h) for h in sorted(exclude_hours))
        print(f"    Exclude h={label_hrs}: {n} trades, WR {wins/n*100:.1f}%, "
              f"PnL ${pnl:,.0f} (${pnl/n:.2f}/trade)")

    # Build result dict
    result = {
        "label": label,
        "total_trades": len(trades),
        "total_pnl": round(sum(t.net_pnl for t in trades), 2),
        "win_rate": round(sum(1 for t in trades if t.net_pnl > 0) / max(len(trades), 1) * 100, 1),
        "hours": {},
    }
    for h in range(24):
        hs = hours[h]
        if hs.trades > 0:
            result["hours"][str(h)] = {
                "trades": hs.trades,
                "wr": round(hs.wr, 1),
                "total_pnl": round(hs.total_pnl, 2),
                "avg_pnl": round(hs.avg_pnl, 2),
                "avg_win": round(hs.avg_win, 2),
                "avg_loss": round(hs.avg_loss, 2),
                "wl_ratio": round(hs.wl_ratio, 2),
                "exits": dict(hs.exit_reasons),
            }
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    print("B3 Hour-of-Day Analysis")
    print("=" * 60)
    print(f"Config: BTC stop {BTC_STOP_PCT*100:.2f}%, PT ${PROFIT_TARGET:.3f}")
    print(f"Walk-forward: {WF_IS_PCT*100:.0f}% IS / {(1-WF_IS_PCT)*100:.0f}% OOS")
    print()

    # Load data
    closes, timestamps = load_klines()
    log_returns = precompute_log_returns(closes)
    windows = build_windows(closes, timestamps)
    print(f"  {len(windows):,} × {WINDOW_MIN}-min windows")

    # Walk-forward split
    split = int(len(windows) * WF_IS_PCT)
    is_windows = windows[:split]
    oos_windows = windows[split:]
    print(f"  IS: {len(is_windows):,} windows, OOS: {len(oos_windows):,} windows")

    # Simulate all windows
    all_trades_is: list[Trade] = []
    all_trades_oos: list[Trade] = []

    capital = INITIAL_CAPITAL
    for start_idx in is_windows:
        if capital < MIN_ORDER_SIZE:
            break
        sigma = compute_sigma(log_returns, start_idx)
        trades = simulate_window(closes, timestamps, start_idx, sigma, capital)
        for tr in trades:
            all_trades_is.append(tr)
            capital += tr.net_pnl

    capital_oos = capital  # Continue from IS ending capital
    for start_idx in oos_windows:
        if capital_oos < MIN_ORDER_SIZE:
            break
        sigma = compute_sigma(log_returns, start_idx)
        trades = simulate_window(closes, timestamps, start_idx, sigma, capital_oos)
        for tr in trades:
            all_trades_oos.append(tr)
            capital_oos += tr.net_pnl

    # Analyze
    result_is = analyze_trades(all_trades_is, "IN-SAMPLE (70%)")
    result_oos = analyze_trades(all_trades_oos, "OUT-OF-SAMPLE (30%)")
    result_all = analyze_trades(all_trades_is + all_trades_oos, "ALL DATA (100%)")

    # Day-of-week analysis
    print(f"\n{'='*60}")
    print("  DAY-OF-WEEK ANALYSIS (all data)")
    print(f"{'='*60}")
    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    dow_trades: dict[int, list[Trade]] = defaultdict(list)
    for t in all_trades_is + all_trades_oos:
        dow = datetime.utcfromtimestamp(t.timestamp).weekday()
        dow_trades[dow].append(t)
    print(f"{'Day':>4}  {'Trades':>6}  {'WR%':>5}  {'TotalPnL':>10}  {'AvgPnL':>8}")
    print("-" * 45)
    for d in range(7):
        ts = dow_trades[d]
        if not ts:
            continue
        n = len(ts)
        wins = sum(1 for t in ts if t.net_pnl > 0)
        pnl = sum(t.net_pnl for t in ts)
        print(f"  {dow_names[d]}  {n:>6}  {wins/n*100:>5.1f}  ${pnl:>9,.1f}  ${pnl/n:>7.2f}")

    # Volatility regime analysis
    print(f"\n{'='*60}")
    print("  VOLATILITY REGIME × HOUR")
    print(f"{'='*60}")
    # Group by sigma tercile and day/night
    all_trades = all_trades_is + all_trades_oos
    # Recalculate sigma for each trade's window
    sigma_vals = []
    for start_idx in windows:
        sigma = compute_sigma(log_returns, start_idx)
        sigma_vals.append(sigma)
    # Map each trade to its sigma
    trade_sigmas = []
    for start_idx in windows:
        sigma = compute_sigma(log_returns, start_idx)
        trades = simulate_window(closes, timestamps, start_idx, sigma, 10000)  # large capital
        for tr in trades:
            trade_sigmas.append((sigma, tr))

    if trade_sigmas:
        sigmas_only = [s for s, _ in trade_sigmas]
        sigmas_only.sort()
        p33 = sigmas_only[len(sigmas_only) // 3]
        p66 = sigmas_only[2 * len(sigmas_only) // 3]

        print(f"  Sigma terciles: low < {p33:.6f}, mid {p33:.6f}-{p66:.6f}, high > {p66:.6f}")
        print()
        print(f"{'Regime':>8} {'Period':>12} {'Trades':>6} {'WR%':>5} {'AvgPnL':>8} {'TotalPnL':>10}")
        print("-" * 60)

        for regime, lo, hi in [("low", 0, p33), ("mid", p33, p66), ("high", p66, 999)]:
            for period, hour_filter in [("day(8-22)", lambda h: 8 <= h <= 22),
                                         ("night(23-7)", lambda h: h < 8 or h > 22)]:
                subset = [t for s, t in trade_sigmas if lo <= s < hi and hour_filter(t.hour_cet)]
                if not subset:
                    continue
                n = len(subset)
                wins = sum(1 for t in subset if t.net_pnl > 0)
                pnl = sum(t.net_pnl for t in subset)
                print(f"  {regime:>6} {period:>12} {n:>6} {wins/n*100:>5.1f} "
                      f"${pnl/n:>7.2f} ${pnl:>9,.1f}")

    # Save results
    output = {
        "config": {
            "btc_stop_pct": BTC_STOP_PCT,
            "profit_target": PROFIT_TARGET,
            "sigma_scale": SIGMA_SCALE,
            "entry_threshold": ENTRY_THRESHOLD,
            "edge_exit": EDGE_EXIT,
            "window_min": WINDOW_MIN,
        },
        "in_sample": result_is,
        "out_of_sample": result_oos,
        "all_data": result_all,
    }
    RESULTS_PATH.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
