"""
B3 15-min Chainlink Oracle Autoresearch
========================================

Same model as 5-min B3 but for 15-minute BTC Up/Down markets.
15-min markets have 7x more volume and 1.5x more depth.

Key differences from 5-min:
- window_min = 15 (3x longer)
- Entry window: minute 2-12 (vs 2-3 for 5-min)
- Higher sigma accumulation: sigma * sqrt(15) vs sqrt(5)
- Resolution from chainlink_resolutions_15m table

Usage:
    python3 research/innovations/sweep_b3_15min.py
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
RESULTS_PATH = RESULTS_DIR / "b3_15min_chainlink_sweep.json"
LOG_PATH = Path(__file__).parent.parent / "b3_15min_chainlink_sweep_log.txt"

INITIAL_CAPITAL = 300.0  # Fixed, matches live
MIN_ORDER_SIZE = 5.0
MAX_BET_SIZE = 100.0
MAX_SHARES = 500
MIN_ENTRY_MKT_FV = 0.15
MAX_ENTRY_MKT_FV = 0.85
SIGMA_FLOOR = 0.00005
MIN_TRADES_FOR_SCORE = 30
WF_IS_PCT = 0.70
WINDOW_MIN = 15

GEN0_TRIALS = 1200
GEN1_TRIALS = 800
GEN2_TRIALS = 400
TOP_N_GEN1 = 15
TOP_N_GEN2 = 5
WF_TOP_N = 5

_SQRT2 = math.sqrt(2.0)
_log_fh = None


def log(msg: str = "") -> None:
    global _log_fh
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}" if msg else ""
    print(line)
    if _log_fh is None:
        _log_fh = open(LOG_PATH, "w")
    _log_fh.write(line + "\n")
    _log_fh.flush()


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / _SQRT2))

def _maker_rebate(price: float) -> float:
    p = min(price, 1.0 - price)
    return 0.20 * 0.10 * p * p


def load_klines() -> tuple[list[float], list[int]]:
    conn = sqlite3.connect(str(DB_PATH))
    rows = conn.execute(
        "SELECT ts, close FROM binance_klines WHERE symbol='BTCUSDT' ORDER BY ts"
    ).fetchall()
    conn.close()
    return [r[1] for r in rows], [r[0] for r in rows]


def load_chainlink_resolutions() -> dict[int, bool]:
    conn = sqlite3.connect(str(DB_PATH))
    try:
        rows = conn.execute("SELECT ts, up_won FROM chainlink_resolutions_15m").fetchall()
    except sqlite3.OperationalError:
        return {}
    conn.close()
    return {r[0]: bool(r[1]) for r in rows}


def build_windows(closes: list[float], timestamps: list[int]) -> list[int]:
    indices = []
    i = 0
    while i + WINDOW_MIN < len(closes):
        if timestamps[i + 1] - timestamps[i] <= 120:
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


def compute_sigma(log_returns: list[float], end_idx: int, window: int) -> float:
    start = max(1, end_idx - window)
    rets = log_returns[start:end_idx]
    if len(rets) < 10:
        return SIGMA_FLOOR
    n = len(rets)
    mean = sum(rets) / n
    var = sum((r - mean) ** 2 for r in rets) / (n - 1)
    return max(math.sqrt(var), SIGMA_FLOOR)


@dataclass
class Config:
    sigma_window: int = 720
    sigma_scale: float = 0.5
    entry_threshold: float = 0.10
    min_entry_min: int = 2
    max_entry_min: int = 8
    spread: float = 0.01
    position_pct: float = 0.029
    edge_scaling: float = 5.0
    reentry_cooldown: int = 3


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
    chainlink: dict[int, bool], timestamps: list[int],
) -> list[Trade]:
    wm = WINDOW_MIN
    S_start = closes[start_idx]
    if S_start <= 0:
        return []

    max_entry = min(c.max_entry_min, wm - 2)
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

        sigma_rem_true = sigma_per_min * sqrt_t
        if sigma_rem_true > 1e-12:
            market_up = _norm_cdf(log_ratio / sigma_rem_true)
        else:
            market_up = 1.0 if log_ratio > 0 else 0.0 if log_ratio < 0 else 0.5
        market_up = max(0.02, min(0.98, market_up))

        sigma_rem_model = sigma_per_min * c.sigma_scale * sqrt_t
        if sigma_rem_model > 1e-12:
            signal_up = _norm_cdf(log_ratio / sigma_rem_model)
        else:
            signal_up = 1.0 if log_ratio > 0 else 0.0 if log_ratio < 0 else 0.5
        signal_up = max(0.01, min(0.99, signal_up))

        if position is None:
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
            # Never-sell: hold to resolution

    # Resolution (Chainlink oracle truth)
    if position is not None:
        direction, entry_mkt_fv, entry_t, shares, entry_fee, S_entry = position
        window_ts = timestamps[start_idx]
        window_start_aligned = (window_ts // 900) * 900

        if window_start_aligned in chainlink:
            resolved_up = chainlink[window_start_aligned]
        else:
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
    chainlink_pct: float = 0.0


def simulate_portfolio(
    closes: list[float], timestamps: list[int], log_returns: list[float],
    windows: list[int], cfg: Config, chainlink: dict[int, bool],
) -> Result:
    capital = INITIAL_CAPITAL
    peak = INITIAL_CAPITAL
    max_dd = 0.0
    all_trades: list[Trade] = []
    daily_pnl: dict[str, float] = defaultdict(float)
    exit_reasons: dict[str, int] = defaultdict(int)
    cl_count = 0

    for si in windows:
        if capital < MIN_ORDER_SIZE:
            break
        sigma = compute_sigma(log_returns, si, cfg.sigma_window)
        wts = (timestamps[si] // 900) * 900
        has_cl = wts in chainlink

        for tr in simulate_window(closes, si, sigma, capital, cfg, chainlink, timestamps):
            all_trades.append(tr)
            capital += tr.net_pnl
            day = datetime.utcfromtimestamp(timestamps[si + tr.entry_min]).strftime("%Y-%m-%d")
            daily_pnl[day] += tr.net_pnl
            exit_reasons[tr.exit_reason] += 1
            if has_cl:
                cl_count += 1
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
    r.chainlink_pct = cl_count / max(n, 1) * 100
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


def random_config() -> Config:
    return Config(
        sigma_window=random.choice([60, 120, 240, 360, 720, 1440]),
        sigma_scale=round(random.uniform(0.2, 3.0), 3),
        entry_threshold=round(random.uniform(0.02, 0.30), 3),
        min_entry_min=random.randint(2, 5),
        max_entry_min=random.randint(5, 12),
        spread=round(random.uniform(0.005, 0.02), 3),
        position_pct=round(random.uniform(0.015, 0.060), 3),
        edge_scaling=round(random.uniform(0.5, 10.0), 2),
        reentry_cooldown=random.randint(0, 5),
    )


def mutate_config(base: Config, intensity: float = 0.25) -> Config:
    c = Config(**asdict(base))

    def jf(val: float, lo: float, hi: float) -> float:
        delta = (hi - lo) * intensity * random.gauss(0, 1)
        return round(max(lo, min(hi, val + delta)), 3)

    def ji(val: int, lo: int, hi: int) -> int:
        delta = max(1, int((hi - lo) * intensity))
        return max(lo, min(hi, val + random.randint(-delta, delta)))

    c.sigma_window = random.choice([60, 120, 240, 360, 720, 1440])
    c.sigma_scale = jf(c.sigma_scale, 0.2, 3.0)
    c.entry_threshold = jf(c.entry_threshold, 0.02, 0.30)
    c.min_entry_min = ji(c.min_entry_min, 2, 5)
    c.max_entry_min = ji(c.max_entry_min, 5, 12)
    c.spread = jf(c.spread, 0.005, 0.02)
    c.position_pct = jf(c.position_pct, 0.015, 0.060)
    c.edge_scaling = jf(c.edge_scaling, 0.5, 10.0)
    c.reentry_cooldown = ji(c.reentry_cooldown, 0, 5)
    return c


def main() -> None:
    log("B3 15-min Chainlink Oracle Autoresearch")
    log("=" * 60)

    closes, timestamps = load_klines()
    log_returns = precompute_log_returns(closes)
    windows = build_windows(closes, timestamps)
    chainlink = load_chainlink_resolutions()

    cl_hits = sum(1 for si in windows if (timestamps[si] // 900) * 900 in chainlink)
    log(f"Data: {len(closes):,} klines, {len(windows):,} 15-min windows")
    log(f"Chainlink 15m resolutions: {len(chainlink):,} ({cl_hits}/{len(windows)} = {cl_hits/max(len(windows),1)*100:.1f}%)")

    if cl_hits < 1000:
        log(f"WARNING: Only {cl_hits} Chainlink resolutions. Waiting for fetcher...")
        return

    all_results: list[dict] = []
    best_score = 0.0
    best_cfg: Config | None = None

    # Gen 0
    log(f"\n{'='*60}\nGEN 0: RANDOM ({GEN0_TRIALS})\n{'='*60}")
    t0 = time.time()
    for i in range(GEN0_TRIALS):
        cfg = random_config()
        r = simulate_portfolio(closes, timestamps, log_returns, windows, cfg, chainlink)
        all_results.append({"gen": 0, "config": asdict(cfg), "score": r.score,
            "pnl": r.total_pnl, "trades": r.total_trades, "wr": round(r.win_rate, 1),
            "sharpe": round(r.sharpe, 2), "dd": round(r.max_drawdown_pct, 1),
            "cl_pct": round(r.chainlink_pct, 1), "exits": r.exit_reasons})
        if r.score > best_score:
            best_score = r.score; best_cfg = cfg
        if (i + 1) % 200 == 0:
            log(f"  [{i+1}/{GEN0_TRIALS}] best={best_score:.1f}")
    log(f"Gen 0 done in {time.time()-t0:.0f}s. Best: {best_score:.1f}")

    all_results.sort(key=lambda x: x["score"], reverse=True)
    top = [Config(**r["config"]) for r in all_results[:TOP_N_GEN1]]
    for i, r in enumerate(all_results[:5]):
        c = r["config"]
        log(f"  {i+1}. score={r['score']:.1f} PnL=${r['pnl']:.0f} trades={r['trades']} WR={r['wr']}% "
            f"σs={c['sigma_scale']} thr={c['entry_threshold']} entry={c['min_entry_min']}-{c['max_entry_min']}")

    # Gen 1
    log(f"\n{'='*60}\nGEN 1: MUTATIONS ({GEN1_TRIALS})\n{'='*60}")
    t1 = time.time()
    for i in range(GEN1_TRIALS):
        cfg = mutate_config(random.choice(top))
        r = simulate_portfolio(closes, timestamps, log_returns, windows, cfg, chainlink)
        all_results.append({"gen": 1, "config": asdict(cfg), "score": r.score,
            "pnl": r.total_pnl, "trades": r.total_trades, "wr": round(r.win_rate, 1),
            "sharpe": round(r.sharpe, 2), "dd": round(r.max_drawdown_pct, 1),
            "cl_pct": round(r.chainlink_pct, 1), "exits": r.exit_reasons})
        if r.score > best_score:
            best_score = r.score; best_cfg = cfg
        if (i + 1) % 200 == 0:
            log(f"  [{i+1}/{GEN1_TRIALS}] best={best_score:.1f}")
    log(f"Gen 1 done in {time.time()-t1:.0f}s. Best: {best_score:.1f}")

    all_results.sort(key=lambda x: x["score"], reverse=True)
    top5 = [Config(**r["config"]) for r in all_results[:TOP_N_GEN2]]

    # Gen 2
    log(f"\n{'='*60}\nGEN 2: FINE-TUNE ({GEN2_TRIALS})\n{'='*60}")
    t2 = time.time()
    for i in range(GEN2_TRIALS):
        cfg = mutate_config(random.choice(top5), intensity=0.15)
        r = simulate_portfolio(closes, timestamps, log_returns, windows, cfg, chainlink)
        all_results.append({"gen": 2, "config": asdict(cfg), "score": r.score,
            "pnl": r.total_pnl, "trades": r.total_trades, "wr": round(r.win_rate, 1),
            "sharpe": round(r.sharpe, 2), "dd": round(r.max_drawdown_pct, 1),
            "cl_pct": round(r.chainlink_pct, 1), "exits": r.exit_reasons})
        if r.score > best_score:
            best_score = r.score; best_cfg = cfg
    log(f"Gen 2 done in {time.time()-t2:.0f}s. Best: {best_score:.1f}")

    # Walk-forward
    log(f"\n{'='*60}\nWALK-FORWARD (top-{WF_TOP_N})\n{'='*60}")
    all_results.sort(key=lambda x: x["score"], reverse=True)
    split = int(len(windows) * WF_IS_PCT)
    wf_results = []

    for i, r in enumerate(all_results[:WF_TOP_N]):
        cfg = Config(**r["config"])
        is_r = simulate_portfolio(closes, timestamps, log_returns, windows[:split], cfg, chainlink)
        oos_r = simulate_portfolio(closes, timestamps, log_returns, windows[split:], cfg, chainlink)
        c = r["config"]
        log(f"\n  #{i+1} σs={c['sigma_scale']} thr={c['entry_threshold']} entry={c['min_entry_min']}-{c['max_entry_min']} "
            f"pos%={c['position_pct']} es={c['edge_scaling']}")
        log(f"    IS:  score={is_r.score:.1f} trades={is_r.total_trades} WR={is_r.win_rate:.1f}% "
            f"PnL=${is_r.total_pnl:.0f} Sharpe={is_r.sharpe:.1f} DD={is_r.max_drawdown_pct:.1f}%")
        log(f"    OOS: score={oos_r.score:.1f} trades={oos_r.total_trades} WR={oos_r.win_rate:.1f}% "
            f"PnL=${oos_r.total_pnl:.0f} Sharpe={oos_r.sharpe:.1f} DD={oos_r.max_drawdown_pct:.1f}%")
        log(f"    Exits: {oos_r.exit_reasons}")
        wf_results.append({"rank": i+1, "config": c,
            "is": {"score": is_r.score, "trades": is_r.total_trades, "wr": round(is_r.win_rate, 1),
                   "pnl": round(is_r.total_pnl, 2), "sharpe": round(is_r.sharpe, 2), "dd": round(is_r.max_drawdown_pct, 1)},
            "oos": {"score": oos_r.score, "trades": oos_r.total_trades, "wr": round(oos_r.win_rate, 1),
                    "pnl": round(oos_r.total_pnl, 2), "sharpe": round(oos_r.sharpe, 2), "dd": round(oos_r.max_drawdown_pct, 1),
                    "exits": oos_r.exit_reasons}})

    # Compare with 5-min deployed
    log(f"\n{'='*60}\nCOMPARISON: 15-min vs 5-min deployed\n{'='*60}")
    best_wf = max(wf_results, key=lambda x: x["oos"]["score"])
    log(f"15-min best OOS: score={best_wf['oos']['score']:.1f} trades={best_wf['oos']['trades']} "
        f"WR={best_wf['oos']['wr']}% PnL=${best_wf['oos']['pnl']:.0f} DD={best_wf['oos']['dd']}%")
    log(f"5-min deployed:  score=7996 trades=4423 WR=70.7% PnL=$2690 DD=33%")

    output = {"meta": {"sweep": "b3_15min_chainlink", "timestamp": datetime.now().isoformat(),
        "windows": len(windows), "chainlink": len(chainlink), "trials": len(all_results)},
        "walk_forward": wf_results, "all_trials": all_results[:50]}
    RESULTS_PATH.write_text(json.dumps(output, indent=2))
    log(f"\nResults: {RESULTS_PATH}")
    log(f"Total: {(time.time()-t0)/60:.1f} min, {len(all_results)} trials")


if __name__ == "__main__":
    random.seed(42)
    main()
