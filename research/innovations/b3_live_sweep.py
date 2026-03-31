"""
B3 Live-Realistic Sweep — Optimize for sell failures
=====================================================

Paper model assumes instant sells. Live reality: 28% sell success rate
on 5-min markets. This sweep simulates realistic sell conditions:
- Sell at extreme prices (>0.95, <0.05) = 10% success
- Sell at moderate prices (0.05-0.95) = 40-70% success
- Failed sells → position resolves at $1 (win) or $0 (loss)

Tests baseline + "never sell" + parameter variants with live exit model.

Usage:
    python3 research/innovations/b3_live_sweep.py
"""

from __future__ import annotations

import json
import math
import random
import sqlite3
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
DB_PATH = DATA_DIR / "crypto_price_pmd.sqlite"
RESULTS_DIR = DATA_DIR / "experiments"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PATH = RESULTS_DIR / "b3_live_sweep.json"
LOG_PATH = Path(__file__).parent.parent / "b3_live_sweep_log.txt"

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


def _maker_rebate(price: float) -> float:
    p = min(price, 1.0 - price)
    return 0.20 * 0.10 * p * p


# ═══════════════════════════════════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════════════════════════════════


def load_klines() -> tuple[list[float], list[int]]:
    conn = sqlite3.connect(str(DB_PATH))
    rows = conn.execute(
        "SELECT ts, close FROM binance_klines WHERE symbol='BTCUSDT' ORDER BY ts"
    ).fetchall()
    conn.close()
    return [r[1] for r in rows], [r[0] for r in rows]


def build_windows(closes: list[float], timestamps: list[int], wm: int) -> list[int]:
    indices = []
    i = 0
    while i + wm < len(closes):
        if timestamps[i + 1] - timestamps[i] <= 120:
            indices.append(i)
        i += wm
    return indices


def precompute_log_returns(closes: list[float]) -> list[float]:
    lr = [0.0]
    for i in range(1, len(closes)):
        if closes[i] > 0 and closes[i - 1] > 0:
            lr.append(math.log(closes[i] / closes[i - 1]))
        else:
            lr.append(0.0)
    return lr


def compute_sigma(log_returns: list[float], end_idx: int, window: int) -> float:
    start = max(1, end_idx - window)
    rets = log_returns[start:end_idx]
    if len(rets) < 10:
        return SIGMA_FLOOR
    n = len(rets)
    mean = sum(rets) / n
    var = sum((r - mean) ** 2 for r in rets) / (n - 1)
    return max(math.sqrt(var), SIGMA_FLOOR)


# ═══════════════════════════════════════════════════════════════════════════════
# LIVE EXIT MODEL
# ═══════════════════════════════════════════════════════════════════════════════


def sell_succeeds(exit_fv: float) -> bool:
    """Model sell success probability based on exit price.

    Calibrated from March 31 live data:
    - Extreme prices (<0.05 or >0.95): ~10% success (CLOB rejects)
    - Low liquidity (0.05-0.20 or 0.80-0.95): ~40% success
    - Moderate (0.20-0.80): ~70% success
    """
    if exit_fv < 0.05 or exit_fv > 0.95:
        return random.random() < 0.10
    if exit_fv < 0.20 or exit_fv > 0.80:
        return random.random() < 0.40
    return random.random() < 0.70


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class Config:
    name: str = "baseline"

    # Entry (fixed from autoresearch)
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

    # Exit params (sweep these)
    profit_target: float = 0.207
    use_btc_stop: bool = True
    btc_stop_pct: float = 0.0015
    stop_loss: float = 99.0  # FV stop disabled when BTC stop active
    edge_exit: float = 0.076
    max_hold_min: int = 3
    allow_resolution: bool = True

    # Live exit model
    live_exit: bool = False  # Simulate sell failures
    never_sell: bool = False  # Skip all early exits, always resolve


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


def simulate_window(
    closes: list[float], start_idx: int, sigma_per_min: float,
    capital: float, c: Config,
) -> list[Trade]:
    wm = c.window_min
    S_start = closes[start_idx]
    if S_start <= 0:
        return []

    max_entry = min(c.max_entry_min, wm - 1)
    if max_entry < c.min_entry_min:
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

        # Market FV (sigma_scale=1.0)
        sigma_rem_true = sigma_per_min * sqrt_t
        if sigma_rem_true > 1e-12:
            market_up = _norm_cdf(log_ratio / sigma_rem_true)
        else:
            market_up = 1.0 if log_ratio > 0 else 0.0 if log_ratio < 0 else 0.5
        market_up = max(0.02, min(0.98, market_up))

        # Signal FV
        sigma_rem_model = sigma_per_min * c.sigma_scale * sqrt_t
        if sigma_rem_model > 1e-12:
            signal_up = _norm_cdf(log_ratio / sigma_rem_model)
        else:
            signal_up = 1.0 if log_ratio > 0 else 0.0 if log_ratio < 0 else 0.5
        signal_up = max(0.01, min(0.99, signal_up))

        if position is None:
            # ══ ENTRY ══
            if t - last_exit_min < c.reentry_cooldown:
                continue
            if t < c.min_entry_min or t > max_entry:
                continue

            signal_dev = signal_up - 0.50
            if abs(signal_dev) < c.entry_threshold:
                continue

            direction = 1 if signal_dev > 0 else -1
            entry_mkt_fv = market_up if direction == 1 else (1.0 - market_up)

            if entry_mkt_fv < MIN_ENTRY_MKT_FV or entry_mkt_fv > MAX_ENTRY_MKT_FV:
                continue

            entry_price = entry_mkt_fv + c.spread / 2
            entry_fee = -_maker_rebate(entry_mkt_fv)
            edge = abs(signal_dev)

            raw_pct = min(c.position_pct, edge * c.edge_scaling)
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

            # Never-sell mode: skip all early exits
            if c.never_sell:
                continue

            should_exit = False
            reason = ""

            if unrealized >= c.profit_target:
                should_exit, reason = True, "profit"
            elif c.use_btc_stop:
                btc_change = (S_now - S_entry) / S_entry
                if (direction == 1 and btc_change <= -c.btc_stop_pct) or (
                    direction == -1 and btc_change >= c.btc_stop_pct
                ):
                    should_exit, reason = True, "stop"
            elif unrealized <= -c.stop_loss:
                should_exit, reason = True, "stop"

            if not should_exit:
                if hold_min >= c.max_hold_min:
                    should_exit, reason = True, "time"
                elif abs(pos_signal_fv - 0.50) < c.edge_exit and hold_min >= 1:
                    should_exit, reason = True, "edge_gone"

            if should_exit:
                exit_mkt_fv = pos_mkt_fv

                # ── LIVE EXIT MODEL ──
                if c.live_exit:
                    if not sell_succeeds(exit_mkt_fv):
                        # Sell failed → don't exit, hold to resolution
                        continue

                exit_price = exit_mkt_fv - c.spread / 2
                exit_fee = -_maker_rebate(exit_mkt_fv)
                gross = (exit_price - (entry_mkt_fv + c.spread / 2)) * shares
                fee_total = (entry_fee + exit_fee) * shares
                net = gross - fee_total

                trades.append(Trade(
                    direction=direction, entry_min=entry_t, exit_min=t,
                    entry_fv=entry_mkt_fv, exit_fv=exit_mkt_fv,
                    shares=shares, net_pnl=net, exit_reason=reason,
                ))
                position = None
                last_exit_min = t

    # ── Resolution ──
    if position is not None:
        direction, entry_mkt_fv, entry_t, shares, entry_fee, S_entry = position
        if c.allow_resolution:
            resolved_up = closes[start_idx + wm] >= S_start
            won = (direction == 1 and resolved_up) or (direction == -1 and not resolved_up)
            exit_mkt_fv = 1.0 if won else 0.0
            gross = (exit_mkt_fv - (entry_mkt_fv + c.spread / 2)) * shares
            fee_total = entry_fee * shares
            net = gross - fee_total
            trades.append(Trade(
                direction=direction, entry_min=entry_t, exit_min=wm,
                entry_fv=entry_mkt_fv, exit_fv=exit_mkt_fv,
                shares=shares, net_pnl=net,
                exit_reason="resolution_win" if won else "resolution_lose",
            ))

    return trades


# ═══════════════════════════════════════════════════════════════════════════════
# PORTFOLIO + SCORING
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class Result:
    total_pnl: float = 0.0
    total_trades: int = 0
    wins: int = 0
    win_rate: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe: float = 0.0
    score: float = 0.0
    exit_reasons: dict = field(default_factory=dict)
    avg_pnl: float = 0.0


def simulate_portfolio(
    closes: list[float], timestamps: list[int], log_returns: list[float],
    windows: list[int], cfg: Config,
) -> Result:
    capital = INITIAL_CAPITAL
    peak = INITIAL_CAPITAL
    max_dd = 0.0
    all_trades: list[Trade] = []
    daily_pnl: dict[str, float] = defaultdict(float)
    exit_reasons: dict[str, int] = defaultdict(int)

    for si in windows:
        if capital < MIN_ORDER_SIZE:
            break
        sigma = compute_sigma(log_returns, si, cfg.sigma_window)
        for tr in simulate_window(closes, si, sigma, capital, cfg):
            all_trades.append(tr)
            capital += tr.net_pnl
            day = datetime.utcfromtimestamp(timestamps[si + tr.entry_min]).strftime("%Y-%m-%d")
            daily_pnl[day] += tr.net_pnl
            exit_reasons[tr.exit_reason] += 1
            if capital > peak:
                peak = capital
            dd = (peak - capital) / peak if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd

    n = len(all_trades)
    r = Result()
    r.total_pnl = capital - INITIAL_CAPITAL
    r.total_trades = n
    r.max_drawdown_pct = max_dd * 100
    r.exit_reasons = dict(exit_reasons)
    if n > 0:
        r.wins = sum(1 for t in all_trades if t.net_pnl > 0)
        r.win_rate = r.wins / n * 100
        r.avg_pnl = r.total_pnl / n

    daily_vals = list(daily_pnl.values())
    if len(daily_vals) >= 5:
        mean_d = sum(daily_vals) / len(daily_vals)
        var_d = sum((d - mean_d) ** 2 for d in daily_vals) / (len(daily_vals) - 1)
        std_d = math.sqrt(max(var_d, 1e-10))
        if std_d > 1e-8:
            r.sharpe = min((mean_d / std_d) * math.sqrt(365), 100.0)

    r.score = experiment_score(r)
    return r


def experiment_score(r: Result) -> float:
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
# EXPERIMENTS
# ═══════════════════════════════════════════════════════════════════════════════


def define_experiments() -> list[Config]:
    exps: list[Config] = []

    # ═══ 1. BASELINES ═══
    # Paper baseline (no live model)
    exps.append(Config(name="paper_baseline", live_exit=False))
    # Live baseline (current params + sell failures)
    exps.append(Config(name="live_baseline", live_exit=True))
    # Never sell (buy only, always resolve)
    exps.append(Config(name="never_sell", live_exit=True, never_sell=True))

    # ═══ 2. NEVER-SELL VARIANTS (entry params only) ═══
    for threshold in [0.05, 0.08, 0.095, 0.12, 0.15, 0.20]:
        exps.append(Config(
            name=f"ns_thresh_{threshold}",
            live_exit=True, never_sell=True,
            entry_threshold=threshold,
        ))
    for scale in [0.4, 0.5, 0.644, 0.8, 1.0, 1.5]:
        exps.append(Config(
            name=f"ns_scale_{scale}",
            live_exit=True, never_sell=True,
            sigma_scale=scale,
        ))

    # ═══ 3. WIDER PROFIT TARGET (sell less often, more resolution wins) ═══
    for pt in [0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
        exps.append(Config(
            name=f"live_pt_{pt}", live_exit=True,
            profit_target=pt,
        ))

    # ═══ 4. WIDER BTC STOP (fewer stops, more resolution) ═══
    for pct in [0.001, 0.002, 0.003, 0.005, 0.010, 99.0]:
        label = "none" if pct > 1 else f"{pct*100:.1f}pct"
        exps.append(Config(
            name=f"live_btcstop_{label}", live_exit=True,
            btc_stop_pct=pct,
        ))

    # ═══ 5. WIDER EDGE EXIT (fewer edge_gone, more resolution) ═══
    for ee in [0.03, 0.05, 0.076, 0.10, 0.15, 0.20, 0.50]:
        exps.append(Config(
            name=f"live_edge_{ee}", live_exit=True,
            edge_exit=ee,
        ))

    # ═══ 6. MAX HOLD (shorter = fewer failed sells) ═══
    for mh in [1, 2, 3, 4]:
        exps.append(Config(
            name=f"live_hold_{mh}min", live_exit=True,
            max_hold_min=mh,
        ))

    # ═══ 7. COMBOS: wide PT + wide stop + wide edge ═══
    for pt in [0.30, 0.50]:
        for btc in [0.003, 0.005, 99.0]:
            for ee in [0.10, 0.20, 0.50]:
                label_btc = "none" if btc > 1 else f"{btc*100:.1f}"
                exps.append(Config(
                    name=f"combo_pt{pt}_btc{label_btc}_ee{ee}",
                    live_exit=True,
                    profit_target=pt,
                    btc_stop_pct=btc,
                    edge_exit=ee,
                ))

    # ═══ 8. COMBOS: never_sell with entry threshold variants ═══
    for thresh in [0.08, 0.095, 0.12, 0.15]:
        for scale in [0.5, 0.644, 0.8]:
            exps.append(Config(
                name=f"ns_t{thresh}_s{scale}",
                live_exit=True, never_sell=True,
                entry_threshold=thresh,
                sigma_scale=scale,
            ))

    log(f"Total experiments: {len(exps)}")
    return exps


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    log("B3 Live-Realistic Sweep")
    log("=" * 60)

    closes, timestamps = load_klines()
    log_returns = precompute_log_returns(closes)
    windows = build_windows(closes, timestamps, 5)
    log(f"Data: {len(closes):,} klines, {len(windows):,} windows")

    split = int(len(windows) * WF_IS_PCT)
    is_windows = windows[:split]
    oos_windows = windows[split:]
    log(f"Walk-forward: IS={len(is_windows):,}, OOS={len(oos_windows):,}")

    experiments = define_experiments()
    results = []

    # Run with fixed seed for reproducibility (live_exit uses random)
    random.seed(42)

    for i, cfg in enumerate(experiments):
        t0 = time.time()

        # Run 3 times for stochastic configs and average
        if cfg.live_exit:
            scores = []
            for run in range(3):
                random.seed(42 + run)
                is_r = simulate_portfolio(closes, timestamps, log_returns, is_windows, cfg)
                oos_r = simulate_portfolio(closes, timestamps, log_returns, oos_windows, cfg)
                scores.append((is_r, oos_r))
            # Average OOS metrics
            avg_oos = Result()
            avg_oos.total_pnl = sum(s[1].total_pnl for s in scores) / 3
            avg_oos.total_trades = scores[0][1].total_trades
            avg_oos.wins = round(sum(s[1].wins for s in scores) / 3)
            avg_oos.win_rate = sum(s[1].win_rate for s in scores) / 3
            avg_oos.max_drawdown_pct = sum(s[1].max_drawdown_pct for s in scores) / 3
            avg_oos.sharpe = sum(s[1].sharpe for s in scores) / 3
            avg_oos.score = sum(s[1].score for s in scores) / 3
            avg_oos.exit_reasons = scores[0][1].exit_reasons
            avg_oos.avg_pnl = avg_oos.total_pnl / max(avg_oos.total_trades, 1)
            is_r = scores[0][0]
            oos_r = avg_oos
        else:
            is_r = simulate_portfolio(closes, timestamps, log_returns, is_windows, cfg)
            oos_r = simulate_portfolio(closes, timestamps, log_returns, oos_windows, cfg)

        dt = time.time() - t0

        log(f"\n[{i+1}/{len(experiments)}] {cfg.name} ({dt:.1f}s)")
        log(f"  IS:  score={is_r.score:>8.1f}  trades={is_r.total_trades:>5}  "
            f"WR={is_r.win_rate:>5.1f}%  PnL=${is_r.total_pnl:>9,.0f}  "
            f"avg=${is_r.avg_pnl:>6.2f}  DD={is_r.max_drawdown_pct:>5.1f}%  "
            f"Sharpe={is_r.sharpe:>5.1f}")
        log(f"  OOS: score={oos_r.score:>8.1f}  trades={oos_r.total_trades:>5}  "
            f"WR={oos_r.win_rate:>5.1f}%  PnL=${oos_r.total_pnl:>9,.0f}  "
            f"avg=${oos_r.avg_pnl:>6.2f}  DD={oos_r.max_drawdown_pct:>5.1f}%  "
            f"Sharpe={oos_r.sharpe:>5.1f}")
        log(f"  Exits: {oos_r.exit_reasons}")

        results.append({
            "name": cfg.name,
            "config": asdict(cfg),
            "is": {"score": is_r.score, "trades": is_r.total_trades,
                   "wr": round(is_r.win_rate, 1), "pnl": round(is_r.total_pnl, 2),
                   "avg_pnl": round(is_r.avg_pnl, 2), "dd": round(is_r.max_drawdown_pct, 1),
                   "sharpe": round(is_r.sharpe, 2), "exits": is_r.exit_reasons},
            "oos": {"score": oos_r.score, "trades": oos_r.total_trades,
                    "wr": round(oos_r.win_rate, 1), "pnl": round(oos_r.total_pnl, 2),
                    "avg_pnl": round(oos_r.avg_pnl, 2), "dd": round(oos_r.max_drawdown_pct, 1),
                    "sharpe": round(oos_r.sharpe, 2), "exits": oos_r.exit_reasons},
        })

    # Sort by OOS score
    results.sort(key=lambda x: x["oos"]["score"], reverse=True)

    log("\n" + "=" * 80)
    log("TOP 20 BY OOS SCORE")
    log("=" * 80)
    log(f"{'#':>3} {'Name':<35} {'OOS Score':>9} {'Trades':>6} {'WR%':>5} "
        f"{'OOS PnL':>9} {'Avg':>7} {'DD%':>5} {'Sharpe':>6}")
    log("-" * 90)
    for i, r in enumerate(results[:20]):
        o = r["oos"]
        log(f"{i+1:>3} {r['name']:<35} {o['score']:>9.1f} {o['trades']:>6} "
            f"{o['wr']:>5.1f} ${o['pnl']:>8,.0f} ${o['avg_pnl']:>6.2f} "
            f"{o['dd']:>5.1f} {o['sharpe']:>6.1f}")

    # Save
    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    LOG_PATH.write_text("\n".join(_log_lines))
    log(f"\nResults: {RESULTS_PATH}")
    log(f"Log: {LOG_PATH}")


if __name__ == "__main__":
    main()
