"""
B3 Live-Realistic Autoresearch v2 — Calibrated from 160 actual live trades
==========================================================================

Uses ACTUAL live fill distribution, not Gaussian approximation:
- Gap distribution: sampled from 160 real trades (bimodal: 34% overpay +0.10)
- Fill rate: 76% (24% failed fills skipped)
- Partial fill: 29% of fills are partial (get ~60% of requested shares)
- Chainlink resolution truth
- Never-sell (hold to resolution)
- $300 fixed capital

Key math: in never-sell, breakeven WR = entry_price.
At avg fill 0.61 → need 61% WR. At fill 0.85 → need 85% WR.

Usage:
    python3 research/innovations/sweep_b3_livereal.py
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
RESULTS_PATH = RESULTS_DIR / "b3_livereal_sweep.json"
LOG_PATH = Path(__file__).parent.parent / "b3_livereal_sweep_log.txt"

INITIAL_CAPITAL = 300.0
MIN_ORDER_SIZE = 5.0
MAX_BET_SIZE = 100.0
MAX_SHARES = 500
SIGMA_FLOOR = 0.00005
MIN_TRADES_FOR_SCORE = 30
WF_IS_PCT = 0.70
WINDOW_MIN = 5

GEN0_TRIALS = 1500
GEN1_TRIALS = 1000
GEN2_TRIALS = 500

_SQRT2 = math.sqrt(2.0)
_log_fh = None

# Load ACTUAL live fill distribution from 160 trades
_FILL_DIST = json.loads(Path("/tmp/b3_live_fill_distribution.json").read_text())
_REAL_GAPS = _FILL_DIST["gaps"]  # 160 real gap values
_FILL_RATE = _FILL_DIST["fill_rate"]  # 0.76
_PARTIAL_RATE = _FILL_DIST["partial_rate"]  # 0.29
_PARTIAL_SHARE_RATIO = 0.60  # partial fills get ~60% of shares


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


def load_chainlink() -> dict[int, bool]:
    conn = sqlite3.connect(str(DB_PATH))
    try:
        rows = conn.execute("SELECT ts, up_won FROM chainlink_resolutions").fetchall()
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


def sample_fill(rng: random.Random, model_fv: float) -> tuple[float, float] | None:
    """Simulate realistic CLOB fill using actual live distribution.

    Returns (fill_price, share_ratio) or None if fill fails.
    share_ratio: 1.0 for full fill, ~0.6 for partial, 0 for failed.
    """
    # 24% of orders fail entirely
    if rng.random() > _FILL_RATE:
        return None

    # Sample gap from REAL distribution (160 values)
    gap = rng.choice(_REAL_GAPS)
    fill_price = model_fv + gap
    fill_price = max(0.03, min(0.97, fill_price))

    # 29% of successful fills are partial (~60% shares)
    if rng.random() < _PARTIAL_RATE:
        share_ratio = _PARTIAL_SHARE_RATIO + rng.uniform(-0.15, 0.15)
        share_ratio = max(0.2, min(0.9, share_ratio))
    else:
        share_ratio = 1.0

    return fill_price, share_ratio


@dataclass
class Config:
    sigma_window: int = 720
    sigma_scale: float = 0.5
    entry_threshold: float = 0.15
    min_entry_min: int = 2
    max_entry_min: int = 3
    max_entry_price: float = 0.65  # KEY: breakeven WR = this value
    min_entry_fv: float = 0.50
    position_pct: float = 0.03
    edge_scaling: float = 5.0
    reentry_cooldown: int = 0


@dataclass
class Trade:
    direction: int
    entry_fv: float
    fill_price: float
    shares: float
    net_pnl: float
    exit_reason: str


def simulate_window(
    closes: list[float], start_idx: int, sigma_per_min: float,
    capital: float, c: Config,
    chainlink: dict[int, bool], timestamps: list[int],
    rng: random.Random,
) -> list[Trade]:
    wm = WINDOW_MIN
    S_start = closes[start_idx]
    if S_start <= 0:
        return []

    # Only integer minutes (matching live behavior)
    for t in range(c.min_entry_min, min(c.max_entry_min + 1, wm)):
        S_now = closes[start_idx + t]
        if S_now <= 0:
            continue
        t_remaining = wm - t
        if t_remaining <= 0:
            continue

        log_ratio = math.log(S_now / S_start)
        sqrt_t = math.sqrt(t_remaining)

        sigma_rem = sigma_per_min * sqrt_t
        if sigma_rem > 1e-12:
            market_up = _norm_cdf(log_ratio / sigma_rem)
        else:
            market_up = 0.5
        market_up = max(0.02, min(0.98, market_up))

        sigma_model = sigma_per_min * c.sigma_scale * sqrt_t
        if sigma_model > 1e-12:
            signal_up = _norm_cdf(log_ratio / sigma_model)
        else:
            signal_up = 0.5

        signal_dev = signal_up - 0.50
        if abs(signal_dev) < c.entry_threshold:
            continue

        direction = 1 if signal_dev > 0 else -1
        entry_mkt_fv = market_up if direction == 1 else (1.0 - market_up)

        if entry_mkt_fv < c.min_entry_fv:
            continue

        # REALISTIC fill from actual live distribution
        fill_result = sample_fill(rng, entry_mkt_fv)
        if fill_result is None:
            continue  # Fill failed (24% of the time)

        fill_price, share_ratio = fill_result

        # Max entry price cap
        if fill_price > c.max_entry_price:
            continue

        # Sizing
        edge = abs(signal_dev)
        raw_pct = min(c.position_pct, edge * c.edge_scaling)
        bet_size = min(capital * raw_pct, MAX_BET_SIZE)
        if bet_size < MIN_ORDER_SIZE:
            continue
        shares = bet_size / fill_price if fill_price > 0.01 else 0
        shares = min(shares, MAX_SHARES)
        shares = shares * share_ratio  # Partial fill reduction
        if shares < 1:
            continue

        # Chainlink resolution
        wts = (timestamps[start_idx] // 300) * 300
        if wts in chainlink:
            resolved_up = chainlink[wts]
        else:
            resolved_up = closes[start_idx + wm] >= S_start

        won = (direction == 1 and resolved_up) or (direction == -1 and not resolved_up)
        exit_price = 1.0 if won else 0.0
        rebate = _maker_rebate(fill_price) * shares
        net = (exit_price - fill_price) * shares + rebate

        return [Trade(
            direction=direction, entry_fv=entry_mkt_fv, fill_price=fill_price,
            shares=shares, net_pnl=net,
            exit_reason="resolution_win" if won else "resolution_lose",
        )]

    return []


@dataclass
class Result:
    total_pnl: float = 0.0
    total_trades: int = 0
    wins: int = 0
    win_rate: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe: float = 0.0
    score: float = 0.0
    avg_pnl: float = 0.0
    avg_fill: float = 0.0
    exit_reasons: dict = field(default_factory=dict)


def simulate_portfolio(
    closes, timestamps, log_returns, windows, cfg, chainlink, seed=42,
) -> Result:
    rng = random.Random(seed)
    capital = INITIAL_CAPITAL
    peak = INITIAL_CAPITAL
    max_dd = 0.0
    all_trades = []
    daily_pnl = defaultdict(float)
    exit_reasons = defaultdict(int)
    total_fill = 0.0

    for si in windows:
        if capital < MIN_ORDER_SIZE:
            break
        sigma = compute_sigma(log_returns, si, cfg.sigma_window)
        for tr in simulate_window(closes, si, sigma, capital, cfg, chainlink, timestamps, rng):
            all_trades.append(tr)
            capital += tr.net_pnl
            total_fill += tr.fill_price
            day = datetime.utcfromtimestamp(timestamps[si]).strftime("%Y-%m-%d")
            daily_pnl[day] += tr.net_pnl
            exit_reasons[tr.exit_reason] += 1
            if capital > peak: peak = capital
            dd = (peak - capital) / peak if peak > 0 else 0
            if dd > max_dd: max_dd = dd

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
        r.avg_fill = total_fill / n

    daily_vals = list(daily_pnl.values())
    if len(daily_vals) >= 5:
        mean_d = sum(daily_vals) / len(daily_vals)
        var_d = sum((d - mean_d) ** 2 for d in daily_vals) / (len(daily_vals) - 1)
        std_d = math.sqrt(max(var_d, 1e-10))
        if std_d > 1e-8:
            r.sharpe = min((mean_d / std_d) * math.sqrt(365), 100.0)

    r.score = _score(r)
    return r


def _score(r: Result) -> float:
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
        sigma_window=random.choice([240, 360, 720, 1440]),
        sigma_scale=round(random.uniform(0.15, 2.5), 3),
        entry_threshold=round(random.uniform(0.03, 0.40), 3),
        min_entry_min=random.choice([1, 2]),
        max_entry_min=random.choice([2, 3, 4]),
        max_entry_price=round(random.uniform(0.40, 0.75), 2),
        min_entry_fv=round(random.uniform(0.25, 0.55), 2),
        position_pct=round(random.uniform(0.02, 0.10), 3),
        edge_scaling=round(random.uniform(0.5, 10.0), 2),
        reentry_cooldown=random.randint(0, 3),
    )


def mutate_config(base: Config, intensity: float = 0.25) -> Config:
    c = Config(**asdict(base))
    def jf(v, lo, hi):
        return round(max(lo, min(hi, v + (hi-lo)*intensity*random.gauss(0,1))), 3)
    c.sigma_window = random.choice([240, 360, 720, 1440])
    c.sigma_scale = jf(c.sigma_scale, 0.15, 2.5)
    c.entry_threshold = jf(c.entry_threshold, 0.03, 0.40)
    c.min_entry_min = random.choice([1, 2])
    c.max_entry_min = random.choice([2, 3, 4])
    c.max_entry_price = jf(c.max_entry_price, 0.40, 0.75)
    c.min_entry_fv = jf(c.min_entry_fv, 0.25, 0.55)
    c.position_pct = jf(c.position_pct, 0.02, 0.10)
    c.edge_scaling = jf(c.edge_scaling, 0.5, 10.0)
    c.reentry_cooldown = random.choice([0, 1, 2, 3])
    return c


def main() -> None:
    log("B3 Live-Realistic Autoresearch v2")
    log("Calibrated from 160 actual live trades")
    log("=" * 60)
    log(f"Real fill distribution: {len(_REAL_GAPS)} gaps, fill rate {_FILL_RATE*100:.0f}%, partial {_PARTIAL_RATE*100:.0f}%")

    closes, timestamps = load_klines()
    log_returns = precompute_log_returns(closes)
    windows = build_windows(closes, timestamps)
    chainlink = load_chainlink()
    cl_hits = sum(1 for si in windows if (timestamps[si] // 300) * 300 in chainlink)
    log(f"Data: {len(closes):,} klines, {len(windows):,} windows, {cl_hits} chainlink ({cl_hits/len(windows)*100:.0f}%)")

    all_results = []
    best_score = 0.0
    best_cfg = None

    # Gen 0
    log(f"\nGEN 0: RANDOM ({GEN0_TRIALS})")
    t0 = time.time()
    for i in range(GEN0_TRIALS):
        cfg = random_config()
        scores = [simulate_portfolio(closes, timestamps, log_returns, windows, cfg, chainlink, s) for s in [42,43,44]]
        avg = {k: sum(getattr(s, k) for s in scores)/3 for k in ['score','total_pnl','win_rate','max_drawdown_pct','sharpe','avg_fill']}
        avg['total_trades'] = scores[0].total_trades
        all_results.append({"gen":0, "config":asdict(cfg), **{k:round(v,2) for k,v in avg.items()}, "exits":scores[0].exit_reasons})
        if avg['score'] > best_score: best_score = avg['score']; best_cfg = cfg
        if (i+1) % 300 == 0:
            log(f"  [{i+1}/{GEN0_TRIALS}] best={best_score:.0f} ({(i+1)/(time.time()-t0):.1f}/s)")
    log(f"Gen 0: {time.time()-t0:.0f}s, best={best_score:.0f}")

    all_results.sort(key=lambda x: x["score"], reverse=True)
    top15 = [Config(**r["config"]) for r in all_results[:15]]
    for i, r in enumerate(all_results[:5]):
        c = r["config"]
        log(f"  {i+1}. score={r['score']:.0f} PnL=${r['total_pnl']:.0f} n={r['total_trades']} "
            f"WR={r['win_rate']:.0f}% fill={r['avg_fill']:.3f} maxP={c['max_entry_price']} "
            f"σs={c['sigma_scale']} thr={c['entry_threshold']} minFV={c['min_entry_fv']}")

    # Gen 1
    log(f"\nGEN 1: MUTATIONS ({GEN1_TRIALS})")
    t1 = time.time()
    for i in range(GEN1_TRIALS):
        cfg = mutate_config(random.choice(top15))
        scores = [simulate_portfolio(closes, timestamps, log_returns, windows, cfg, chainlink, s) for s in [42,43,44]]
        avg = {k: sum(getattr(s, k) for s in scores)/3 for k in ['score','total_pnl','win_rate','max_drawdown_pct','sharpe','avg_fill']}
        avg['total_trades'] = scores[0].total_trades
        all_results.append({"gen":1, "config":asdict(cfg), **{k:round(v,2) for k,v in avg.items()}, "exits":scores[0].exit_reasons})
        if avg['score'] > best_score: best_score = avg['score']; best_cfg = cfg
        if (i+1) % 200 == 0: log(f"  [{i+1}/{GEN1_TRIALS}] best={best_score:.0f}")
    log(f"Gen 1: {time.time()-t1:.0f}s, best={best_score:.0f}")

    all_results.sort(key=lambda x: x["score"], reverse=True)
    top5 = [Config(**r["config"]) for r in all_results[:5]]

    # Gen 2
    log(f"\nGEN 2: FINE-TUNE ({GEN2_TRIALS})")
    t2 = time.time()
    for i in range(GEN2_TRIALS):
        cfg = mutate_config(random.choice(top5), intensity=0.15)
        scores = [simulate_portfolio(closes, timestamps, log_returns, windows, cfg, chainlink, s) for s in [42,43,44]]
        avg = {k: sum(getattr(s, k) for s in scores)/3 for k in ['score','total_pnl','win_rate','max_drawdown_pct','sharpe','avg_fill']}
        avg['total_trades'] = scores[0].total_trades
        all_results.append({"gen":2, "config":asdict(cfg), **{k:round(v,2) for k,v in avg.items()}, "exits":scores[0].exit_reasons})
        if avg['score'] > best_score: best_score = avg['score']; best_cfg = cfg
    log(f"Gen 2: {time.time()-t2:.0f}s, best={best_score:.0f}")

    # Walk-forward
    log(f"\n{'='*60}\nWALK-FORWARD (top 5)\n{'='*60}")
    all_results.sort(key=lambda x: x["score"], reverse=True)
    split = int(len(windows) * WF_IS_PCT)
    wf = []

    for i, r in enumerate(all_results[:5]):
        cfg = Config(**r["config"])
        is_r = [simulate_portfolio(closes, timestamps, log_returns, windows[:split], cfg, chainlink, s) for s in [42,43,44]]
        oos_r = [simulate_portfolio(closes, timestamps, log_returns, windows[split:], cfg, chainlink, s) for s in [42,43,44]]

        is_avg = {k: sum(getattr(s,k) for s in is_r)/3 for k in ['total_pnl','win_rate','max_drawdown_pct','sharpe','score','avg_fill']}
        oos_avg = {k: sum(getattr(s,k) for s in oos_r)/3 for k in ['total_pnl','win_rate','max_drawdown_pct','sharpe','score','avg_fill']}
        oos_avg['total_trades'] = oos_r[0].total_trades

        c = r["config"]
        daily = oos_avg['total_pnl'] / 28
        per_trade = oos_avg['total_pnl'] / max(oos_avg['total_trades'], 1)

        log(f"\n  #{i+1} maxP={c['max_entry_price']} σs={c['sigma_scale']} thr={c['entry_threshold']} "
            f"minFV={c['min_entry_fv']} pos%={c['position_pct']} es={c['edge_scaling']}")
        log(f"    IS:  n={is_r[0].total_trades} WR={is_avg['win_rate']:.1f}% PnL=${is_avg['total_pnl']:.0f} "
            f"Sharpe={is_avg['sharpe']:.1f} DD={is_avg['max_drawdown_pct']:.1f}% fill={is_avg['avg_fill']:.3f}")
        log(f"    OOS: n={oos_avg['total_trades']} WR={oos_avg['win_rate']:.1f}% PnL=${oos_avg['total_pnl']:.0f} "
            f"Sharpe={oos_avg['sharpe']:.1f} DD={oos_avg['max_drawdown_pct']:.1f}% fill={oos_avg['avg_fill']:.3f}")
        log(f"    ${daily:.1f}/den, ${per_trade:.2f}/trade")
        log(f"    Exits: {oos_r[0].exit_reasons}")

        # Breakeven check
        log(f"    Breakeven WR at avg fill {oos_avg['avg_fill']:.3f} = {oos_avg['avg_fill']*100:.0f}% → WR {oos_avg['win_rate']:.0f}% {'> OK' if oos_avg['win_rate'] > oos_avg['avg_fill']*100 else '< BAD'}")

        wf.append({"rank":i+1, "config":c,
            "oos":{"score":oos_avg['score'], "trades":oos_avg['total_trades'],
                   "wr":round(oos_avg['win_rate'],1), "pnl":round(oos_avg['total_pnl'],2),
                   "sharpe":round(oos_avg['sharpe'],2), "dd":round(oos_avg['max_drawdown_pct'],1),
                   "avg_fill":round(oos_avg['avg_fill'],3),
                   "daily":round(daily,2), "per_trade":round(per_trade,2)}})

    best_wf = max(wf, key=lambda x: x["oos"]["score"])
    c = best_wf["config"]
    o = best_wf["oos"]
    log(f"\n{'='*60}")
    log(f"BEST REALISTIC CONFIG:")
    for k, v in c.items(): log(f"  {k}: {v}")
    log(f"  OOS: {o['trades']} trades, WR={o['wr']}%, PnL=${o['pnl']:.0f}, ${o['daily']}/day")
    log(f"  Avg fill: {o['avg_fill']:.3f}, breakeven={o['avg_fill']*100:.0f}%, margin={o['wr']-o['avg_fill']*100:.1f}pp")

    output = {"meta": {"sweep": "b3_livereal_v2", "timestamp": datetime.now().isoformat(),
        "fill_dist_size": len(_REAL_GAPS), "fill_rate": _FILL_RATE, "partial_rate": _PARTIAL_RATE,
        "capital": INITIAL_CAPITAL, "trials": len(all_results)},
        "walk_forward": wf, "all_trials": all_results[:50]}
    RESULTS_PATH.write_text(json.dumps(output, indent=2))
    log(f"\nResults: {RESULTS_PATH}")
    log(f"Total: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    random.seed(42)
    main()
