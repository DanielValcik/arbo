"""
Overnight B3 Analysis — Compounding vs Fixed + Time-of-Day
===========================================================

Takes the top 5 configs from livereal sweep and runs:
1. Fixed $300 capital (baseline)
2. Compounding (reinvest profits, $100/trade cap)
3. Compounding with $500 start
4. Time-of-day analysis (split by hour, find best/worst times)
5. Day vs Night comparison

Produces morning report for CEO.

Usage:
    python3 research/innovations/overnight_b3_analysis.py
"""

from __future__ import annotations

import json
import math
import random
import sqlite3
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
DB_PATH = DATA_DIR / "crypto_price_pmd.sqlite"
RESULTS_PATH = DATA_DIR / "experiments" / "b3_overnight_analysis.json"
LOG_PATH = Path(__file__).parent.parent / "b3_overnight_analysis_log.txt"
LIVEREAL_PATH = DATA_DIR / "experiments" / "b3_livereal_sweep.json"

MIN_ORDER_SIZE = 5.0
MAX_BET_SIZE = 100.0
MAX_SHARES = 500
SIGMA_FLOOR = 0.00005
WINDOW_MIN = 5

_SQRT2 = math.sqrt(2.0)
_log_fh = None

_FILL_DIST = json.loads(Path("/tmp/b3_live_fill_distribution.json").read_text())
_REAL_GAPS = _FILL_DIST["gaps"]
_FILL_RATE = _FILL_DIST["fill_rate"]
_PARTIAL_RATE = _FILL_DIST["partial_rate"]


def log(msg: str = "") -> None:
    global _log_fh
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}" if msg else ""
    print(line)
    if _log_fh is None:
        _log_fh = open(LOG_PATH, "w")
    _log_fh.write(line + "\n")
    _log_fh.flush()


def _norm_cdf(x): return 0.5 * (1.0 + math.erf(x / _SQRT2))
def _maker_rebate(p):
    pp = min(p, 1.0 - p)
    return 0.20 * 0.10 * pp * pp


def load_data():
    conn = sqlite3.connect(str(DB_PATH))
    rows = conn.execute("SELECT ts, close FROM binance_klines WHERE symbol='BTCUSDT' ORDER BY ts").fetchall()
    cl_rows = conn.execute("SELECT ts, up_won FROM chainlink_resolutions").fetchall()
    conn.close()
    closes = [r[1] for r in rows]
    timestamps = [r[0] for r in rows]
    chainlink = {r[0]: bool(r[1]) for r in cl_rows}
    log_returns = [0.0]
    for i in range(1, len(closes)):
        if closes[i] > 0 and closes[i-1] > 0:
            log_returns.append(math.log(closes[i] / closes[i-1]))
        else:
            log_returns.append(0.0)
    windows = []
    i = 0
    while i + WINDOW_MIN < len(closes):
        if timestamps[i+1] - timestamps[i] <= 120:
            windows.append(i)
        i += WINDOW_MIN
    return closes, timestamps, log_returns, windows, chainlink


def compute_sigma(log_returns, end_idx, window):
    start = max(1, end_idx - window)
    rets = log_returns[start:end_idx]
    if len(rets) < 10: return SIGMA_FLOOR
    n = len(rets); mean = sum(rets)/n
    return max(math.sqrt(sum((r-mean)**2 for r in rets)/(n-1)), SIGMA_FLOOR)


def simulate(closes, timestamps, log_returns, windows, chainlink, cfg,
             capital=300.0, compounding=False, seed=42, hour_filter=None):
    """Simulate with optional compounding and hour filter."""
    rng = random.Random(seed)
    peak = capital
    start_capital = capital
    max_dd = 0.0
    trades = []
    daily_pnl = defaultdict(float)
    hourly_stats = defaultdict(lambda: {"wins": 0, "losses": 0, "pnl": 0.0})

    for si in windows:
        if capital < MIN_ORDER_SIZE:
            break

        # Hour filter
        if hour_filter is not None:
            hour = datetime.utcfromtimestamp(timestamps[si]).hour
            if hour not in hour_filter:
                continue

        sigma = compute_sigma(log_returns, si, cfg["sigma_window"])
        S_start = closes[si]
        if S_start <= 0: continue

        entered = False
        for t in range(cfg["min_entry_min"], min(cfg["max_entry_min"]+1, WINDOW_MIN)):
            if entered: break
            S_now = closes[si + t]
            if S_now <= 0: continue
            t_rem = WINDOW_MIN - t
            if t_rem <= 0: continue

            log_ratio = math.log(S_now / S_start)
            sqrt_t = math.sqrt(t_rem)
            sigma_rem = sigma * sqrt_t
            market_up = _norm_cdf(log_ratio / sigma_rem) if sigma_rem > 1e-12 else 0.5
            market_up = max(0.02, min(0.98, market_up))
            sigma_model = sigma * cfg["sigma_scale"] * sqrt_t
            signal_up = _norm_cdf(log_ratio / sigma_model) if sigma_model > 1e-12 else 0.5
            signal_dev = signal_up - 0.50
            if abs(signal_dev) < cfg["entry_threshold"]: continue

            direction = 1 if signal_dev > 0 else -1
            entry_fv = market_up if direction == 1 else (1 - market_up)
            if entry_fv < cfg.get("min_entry_fv", 0.3): continue

            # Realistic fill
            if rng.random() > _FILL_RATE: continue
            gap = rng.choice(_REAL_GAPS)
            fill_price = max(0.03, min(0.97, entry_fv + gap))
            if fill_price > cfg["max_entry_price"]: continue

            share_ratio = 1.0
            if rng.random() < _PARTIAL_RATE:
                share_ratio = max(0.2, min(0.9, 0.60 + rng.uniform(-0.15, 0.15)))

            edge = abs(signal_dev)
            raw_pct = min(cfg["position_pct"], edge * cfg["edge_scaling"])
            bet = min(capital * raw_pct, MAX_BET_SIZE)
            if bet < MIN_ORDER_SIZE: continue
            shares = min(bet / fill_price, MAX_SHARES) * share_ratio
            if shares < 1: continue

            # Resolution
            wts = (timestamps[si] // 300) * 300
            resolved_up = chainlink[wts] if wts in chainlink else closes[si+WINDOW_MIN] >= S_start
            won = (direction == 1 and resolved_up) or (direction == -1 and not resolved_up)
            exit_p = 1.0 if won else 0.0
            net = (exit_p - fill_price) * shares + _maker_rebate(fill_price) * shares

            if compounding:
                capital += net
            daily_pnl[datetime.utcfromtimestamp(timestamps[si]).strftime("%Y-%m-%d")] += net

            hour = datetime.utcfromtimestamp(timestamps[si]).hour
            if won:
                hourly_stats[hour]["wins"] += 1
            else:
                hourly_stats[hour]["losses"] += 1
            hourly_stats[hour]["pnl"] += net

            trades.append({"won": won, "pnl": net, "fill": fill_price, "hour": hour})

            if compounding:
                if capital > peak: peak = capital
                dd = (peak - capital) / peak if peak > 0 else 0
                if dd > max_dd: max_dd = dd

            entered = True

    n = len(trades)
    if n == 0:
        return {"trades": 0}

    wins = sum(1 for t in trades if t["won"])
    total_pnl = sum(t["pnl"] for t in trades)
    avg_fill = sum(t["fill"] for t in trades) / n

    if not compounding:
        # Compute DD from cumulative PnL
        cum = 0; peak_cum = 0; max_dd = 0
        for d in sorted(daily_pnl.keys()):
            cum += daily_pnl[d]
            if cum > peak_cum: peak_cum = cum
            dd = peak_cum - cum
            if dd > max_dd: max_dd = dd
        max_dd_pct = max_dd / start_capital * 100
    else:
        max_dd_pct = max_dd * 100

    daily_vals = list(daily_pnl.values())
    sharpe = 0
    if len(daily_vals) >= 5:
        m = sum(daily_vals) / len(daily_vals)
        var = sum((d-m)**2 for d in daily_vals) / (len(daily_vals)-1)
        std = math.sqrt(max(var, 1e-10))
        if std > 1e-8: sharpe = min((m/std)*math.sqrt(365), 100)

    return {
        "trades": n, "wins": wins, "wr": round(wins/n*100, 1),
        "total_pnl": round(total_pnl, 2),
        "avg_pnl": round(total_pnl/n, 2),
        "avg_fill": round(avg_fill, 3),
        "dd_pct": round(max_dd_pct, 1),
        "sharpe": round(sharpe, 1),
        "daily_avg": round(sum(daily_vals)/max(len(daily_vals),1), 2),
        "final_capital": round(start_capital + total_pnl if not compounding else capital, 2),
        "hourly": dict(hourly_stats),
    }


def main():
    log("B3 Overnight Analysis — Compounding + Time-of-Day")
    log("=" * 60)

    closes, timestamps, log_returns, windows, chainlink = load_data()
    split = int(len(windows) * 0.70)
    oos_windows = windows[split:]
    log(f"OOS windows: {len(oos_windows)}")

    # Load top configs from livereal sweep
    livereal = json.loads(LIVEREAL_PATH.read_text())
    configs = [wf["config"] for wf in livereal["walk_forward"]]
    log(f"Configs to test: {len(configs)}")

    all_results = []

    for ci, cfg in enumerate(configs):
        log(f"\n{'='*60}")
        log(f"CONFIG #{ci+1}: maxP={cfg['max_entry_price']} σs={cfg['sigma_scale']} thr={cfg['entry_threshold']}")

        # 1. Fixed $300 (3 seeds average)
        fixed_results = [simulate(closes, timestamps, log_returns, oos_windows, chainlink,
                                   cfg, capital=300, compounding=False, seed=s) for s in [42,43,44]]
        fixed_avg = {k: sum(r[k] for r in fixed_results)/3 for k in ["trades","wr","total_pnl","avg_pnl","avg_fill","dd_pct","sharpe","daily_avg"]}
        fixed_avg["trades"] = fixed_results[0]["trades"]
        log(f"  FIXED $300:    {fixed_avg['trades']} trades, WR={fixed_avg['wr']:.0f}%, "
            f"PnL=${fixed_avg['total_pnl']:.0f}, ${fixed_avg['daily_avg']:.1f}/day, DD={fixed_avg['dd_pct']:.0f}%")

        # 2. Compounding $300
        comp300 = [simulate(closes, timestamps, log_returns, oos_windows, chainlink,
                            cfg, capital=300, compounding=True, seed=s) for s in [42,43,44]]
        comp300_avg = {k: sum(r[k] for r in comp300)/3 for k in ["trades","wr","total_pnl","avg_pnl","avg_fill","dd_pct","sharpe","daily_avg","final_capital"]}
        comp300_avg["trades"] = comp300[0]["trades"]
        log(f"  COMPOUND $300: {comp300_avg['trades']} trades, WR={comp300_avg['wr']:.0f}%, "
            f"PnL=${comp300_avg['total_pnl']:.0f}, final=${comp300_avg['final_capital']:.0f}, DD={comp300_avg['dd_pct']:.0f}%")

        # 3. Compounding $500
        comp500 = [simulate(closes, timestamps, log_returns, oos_windows, chainlink,
                            cfg, capital=500, compounding=True, seed=s) for s in [42,43,44]]
        comp500_avg = {k: sum(r[k] for r in comp500)/3 for k in ["trades","wr","total_pnl","avg_pnl","avg_fill","dd_pct","sharpe","daily_avg","final_capital"]}
        comp500_avg["trades"] = comp500[0]["trades"]
        log(f"  COMPOUND $500: {comp500_avg['trades']} trades, WR={comp500_avg['wr']:.0f}%, "
            f"PnL=${comp500_avg['total_pnl']:.0f}, final=${comp500_avg['final_capital']:.0f}, DD={comp500_avg['dd_pct']:.0f}%")

        # 4. Time-of-day analysis (using fixed $300, seed=42)
        full_result = simulate(closes, timestamps, log_returns, oos_windows, chainlink,
                               cfg, capital=300, compounding=False, seed=42)
        hourly = full_result.get("hourly", {})

        log(f"\n  TIME-OF-DAY (OOS, fixed $300):")
        log(f"  {'Hour':>6} {'Wins':>5} {'Losses':>6} {'WR':>5} {'PnL':>8} {'$/trade':>8}")
        log(f"  {'-'*42}")
        best_hours = []
        for h in range(24):
            hs = hourly.get(h, {"wins":0,"losses":0,"pnl":0})
            total = hs["wins"] + hs["losses"]
            if total == 0: continue
            wr = hs["wins"] / total * 100
            avg = hs["pnl"] / total
            log(f"  {h:>4}:00 {hs['wins']:>5} {hs['losses']:>6} {wr:>4.0f}% ${hs['pnl']:>+7.0f} ${avg:>+7.2f}")
            best_hours.append((h, wr, hs["pnl"], total))

        # 5. Day (6-18 UTC) vs Night (18-6 UTC)
        day_hours = set(range(6, 18))
        night_hours = set(range(0, 6)) | set(range(18, 24))

        day_r = [simulate(closes, timestamps, log_returns, oos_windows, chainlink,
                          cfg, capital=300, compounding=False, seed=s, hour_filter=day_hours) for s in [42,43,44]]
        night_r = [simulate(closes, timestamps, log_returns, oos_windows, chainlink,
                            cfg, capital=300, compounding=False, seed=s, hour_filter=night_hours) for s in [42,43,44]]

        day_avg = {k: sum(r.get(k,0) for r in day_r)/3 for k in ["trades","wr","total_pnl","daily_avg","dd_pct"]}
        night_avg = {k: sum(r.get(k,0) for r in night_r)/3 for k in ["trades","wr","total_pnl","daily_avg","dd_pct"]}
        day_avg["trades"] = day_r[0]["trades"]
        night_avg["trades"] = night_r[0]["trades"]

        log(f"\n  DAY (06-18 UTC):   {day_avg['trades']} trades, WR={day_avg['wr']:.0f}%, "
            f"PnL=${day_avg['total_pnl']:.0f}, ${day_avg['daily_avg']:.1f}/day, DD={day_avg['dd_pct']:.0f}%")
        log(f"  NIGHT (18-06 UTC): {night_avg['trades']} trades, WR={night_avg['wr']:.0f}%, "
            f"PnL=${night_avg['total_pnl']:.0f}, ${night_avg['daily_avg']:.1f}/day, DD={night_avg['dd_pct']:.0f}%")

        all_results.append({
            "config": cfg,
            "fixed_300": fixed_avg,
            "compound_300": comp300_avg,
            "compound_500": comp500_avg,
            "day": day_avg,
            "night": night_avg,
            "hourly": {str(h): hourly.get(h, {}) for h in range(24)},
        })

    # Final report
    log(f"\n{'='*60}")
    log(f"MORNING REPORT")
    log(f"{'='*60}")
    log(f"\nTop config comparison (OOS, 28 days):")
    log(f"{'#':>2} {'Mode':<15} {'Trades':>6} {'WR':>4} {'PnL':>9} {'$/day':>7} {'DD%':>5} {'Final$':>8}")
    log(f"{'-'*60}")

    for i, r in enumerate(all_results):
        c = r["config"]
        for mode, data in [("Fixed $300", r["fixed_300"]), ("Compound $300", r["compound_300"]), ("Compound $500", r["compound_500"])]:
            final = data.get("final_capital", "")
            final_s = f"${final:.0f}" if final else ""
            log(f"{i+1:>2} {mode:<15} {data['trades']:>6} {data['wr']:>3.0f}% ${data['total_pnl']:>8.0f} ${data['daily_avg']:>6.1f} {data['dd_pct']:>4.0f}% {final_s:>8}")

    json.dump(all_results, open(str(RESULTS_PATH), "w"), indent=2, default=str)
    log(f"\nResults: {RESULTS_PATH}")
    log(f"Log: {LOG_PATH}")


if __name__ == "__main__":
    random.seed(42)
    main()
