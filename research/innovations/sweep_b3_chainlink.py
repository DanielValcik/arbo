"""
B3 Chainlink Oracle Autoresearch — Full 3-Generation Sweep
===========================================================

Recalibrates B3 model using Polymarket Chainlink resolution truth instead
of Binance klines for determining win/loss. Signal generation still uses
Binance (faster), but resolution matches live Polymarket oracle.

Key difference from previous sweeps:
  OLD: resolved_up = closes[start_idx + wm] >= S_start  (Binance)
  NEW: resolved_up = chainlink_resolutions[window_ts]    (Chainlink oracle)

This eliminates the 27% wrong-resolution rate found in live trading.

Data:
  - Binance 1-min klines: 89,419 candles (signal + volatility)
  - Chainlink resolutions: from Gamma API (ground truth outcomes)

Phases:
  Gen 0: 1200 random trials (never_sell mode)
  Gen 1: 800 mutations from top-15
  Gen 2: 400 fine-tuning from top-5
  WF:    Walk-forward 70/30 on top-5

Usage:
    python3 research/innovations/sweep_b3_chainlink.py
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
RESULTS_PATH = RESULTS_DIR / "b3_chainlink_sweep.json"
LOG_PATH = Path(__file__).parent.parent / "b3_chainlink_sweep_log.txt"

INITIAL_CAPITAL = 1000.0
MIN_ORDER_SIZE = 5.0
MAX_BET_SIZE = 100.0
MAX_SHARES = 500
MIN_ENTRY_MKT_FV = 0.15
MAX_ENTRY_MKT_FV = 0.85
SIGMA_FLOOR = 0.00005
MIN_TRADES_FOR_SCORE = 30
WF_IS_PCT = 0.70

GEN0_TRIALS = 1200
GEN1_TRIALS = 800
GEN2_TRIALS = 400
TOP_N_GEN1 = 15
TOP_N_GEN2 = 5
WF_TOP_N = 5

_SQRT2 = math.sqrt(2.0)


# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════════
# MATH
# ═══════════════════════════════════════════════════════════════════════════════


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / _SQRT2))


def _maker_rebate(price: float) -> float:
    p = min(price, 1.0 - price)
    return 0.20 * 0.10 * p * p


# ═══════════════════════════════════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════════════════════════════════


def load_klines() -> tuple[list[float], list[int]]:
    """Load Binance 1-min klines (for signal generation + volatility)."""
    conn = sqlite3.connect(str(DB_PATH))
    rows = conn.execute(
        "SELECT ts, close FROM binance_klines WHERE symbol='BTCUSDT' ORDER BY ts"
    ).fetchall()
    conn.close()
    return [r[1] for r in rows], [r[0] for r in rows]


def load_chainlink_resolutions() -> dict[int, bool]:
    """Load Chainlink resolution truth from DB.

    Returns dict mapping event_start_ts → True (Up won) / False (Down won).
    """
    conn = sqlite3.connect(str(DB_PATH))
    try:
        rows = conn.execute(
            "SELECT ts, up_won FROM chainlink_resolutions"
        ).fetchall()
    except sqlite3.OperationalError:
        log("WARNING: chainlink_resolutions table not found!")
        return {}
    conn.close()
    return {r[0]: bool(r[1]) for r in rows}


def build_windows(closes: list[float], timestamps: list[int], wm: int) -> list[int]:
    """Build list of valid window start indices."""
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
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class Config:
    name: str = "baseline"

    # Volatility
    sigma_window: int = 720
    sigma_scale: float = 0.400

    # Entry
    entry_threshold: float = 0.095
    min_entry_min: int = 2
    max_entry_min: int = 3
    spread: float = 0.01
    position_pct: float = 0.067
    edge_scaling: float = 4.838
    window_min: int = 5
    reentry_cooldown: int = 3

    # Exit (never-sell: these only affect paper model FV tracking)
    profit_target: float = 0.207
    use_btc_stop: bool = True
    btc_stop_pct: float = 0.0015
    stop_loss: float = 99.0
    edge_exit: float = 0.076
    max_hold_min: int = 3
    allow_resolution: bool = True

    # Live mode: always never-sell (validated +$155 vs -$50)
    never_sell: bool = True


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
    chainlink: dict[int, bool], timestamps: list[int],
) -> list[Trade]:
    """Simulate one 5-min window. Resolution uses Chainlink oracle truth."""
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
            # Never-sell: don't check exits, always hold to resolution

    # ── Resolution (Chainlink oracle truth) ──
    if position is not None:
        direction, entry_mkt_fv, entry_t, shares, entry_fee, S_entry = position
        if c.allow_resolution:
            # Use Chainlink resolution truth if available
            window_ts = timestamps[start_idx]
            window_start_aligned = (window_ts // 300) * 300

            if window_start_aligned in chainlink:
                # CHAINLINK ORACLE (source of truth for live)
                resolved_up = chainlink[window_start_aligned]
            else:
                # Fallback: Binance (for windows without Polymarket data)
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
    chainlink_pct: float = 0.0  # % of resolutions from Chainlink


def simulate_portfolio(
    closes: list[float], timestamps: list[int], log_returns: list[float],
    windows: list[int], cfg: Config,
    chainlink: dict[int, bool],
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
        window_ts = (timestamps[si] // 300) * 300
        has_cl = window_ts in chainlink

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


# ═══════════════════════════════════════════════════════════════════════════════
# PARAMETER GENERATORS
# ═══════════════════════════════════════════════════════════════════════════════


def random_config() -> Config:
    return Config(
        name="rand",
        sigma_window=random.choice([30, 60, 120, 240, 360, 720, 1440]),
        sigma_scale=round(random.uniform(0.2, 3.0), 3),
        entry_threshold=round(random.uniform(0.02, 0.30), 3),
        min_entry_min=random.choice([1, 2]),
        max_entry_min=random.choice([2, 3]),
        spread=round(random.uniform(0.005, 0.03), 3),
        position_pct=round(random.uniform(0.02, 0.20), 3),
        edge_scaling=round(random.uniform(0.5, 10.0), 2),
        reentry_cooldown=random.randint(0, 3),
        # Never-sell: these don't matter but keep for reference
        profit_target=round(random.uniform(0.05, 0.50), 3),
        use_btc_stop=random.choice([True, False]),
        btc_stop_pct=round(random.uniform(0.001, 0.01), 4),
        edge_exit=round(random.uniform(0.03, 0.20), 3),
        max_hold_min=random.randint(1, 4),
        allow_resolution=True,
        never_sell=True,
    )


def mutate_config(base: Config, intensity: float = 0.3) -> Config:
    c = Config(**asdict(base))
    c.name = "mut"

    def jf(val: float, lo: float, hi: float) -> float:
        delta = (hi - lo) * intensity * random.gauss(0, 1)
        return round(max(lo, min(hi, val + delta)), 3)

    def ji(val: int, lo: int, hi: int) -> int:
        delta = max(1, int((hi - lo) * intensity))
        return max(lo, min(hi, val + random.randint(-delta, delta)))

    c.sigma_window = random.choice([30, 60, 120, 240, 360, 720, 1440])
    c.sigma_scale = jf(c.sigma_scale, 0.2, 3.0)
    c.entry_threshold = jf(c.entry_threshold, 0.02, 0.30)
    c.min_entry_min = random.choice([1, 2])
    c.max_entry_min = random.choice([2, 3])
    c.spread = jf(c.spread, 0.005, 0.03)
    c.position_pct = jf(c.position_pct, 0.02, 0.20)
    c.edge_scaling = jf(c.edge_scaling, 0.5, 10.0)
    c.reentry_cooldown = ji(c.reentry_cooldown, 0, 3)
    return c


# ═══════════════════════════════════════════════════════════════════════════════
# WALK-FORWARD
# ═══════════════════════════════════════════════════════════════════════════════


def walk_forward(
    closes: list[float], timestamps: list[int], log_returns: list[float],
    windows: list[int], cfg: Config, chainlink: dict[int, bool],
) -> tuple[Result, Result]:
    split = int(len(windows) * WF_IS_PCT)
    is_r = simulate_portfolio(closes, timestamps, log_returns, windows[:split], cfg, chainlink)
    oos_r = simulate_portfolio(closes, timestamps, log_returns, windows[split:], cfg, chainlink)
    return is_r, oos_r


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN SWEEP
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    log("B3 Chainlink Oracle Autoresearch")
    log("=" * 60)

    closes, timestamps = load_klines()
    log_returns = precompute_log_returns(closes)
    windows = build_windows(closes, timestamps, 5)
    chainlink = load_chainlink_resolutions()

    # Count coverage
    cl_hits = sum(1 for si in windows if (timestamps[si] // 300) * 300 in chainlink)
    log(f"Data: {len(closes):,} klines, {len(windows):,} windows")
    log(f"Chainlink resolutions: {len(chainlink):,} ({cl_hits}/{len(windows)} windows = {cl_hits/len(windows)*100:.1f}%)")

    if cl_hits < 1000:
        log(f"WARNING: Only {cl_hits} Chainlink resolutions. Waiting for fetcher to complete...")
        log("Run: python3 research/fetch_chainlink_resolutions.py")
        return

    all_results: list[dict] = []
    best_score = 0.0
    best_cfg: Config | None = None

    # ── GEN 0: RANDOM SEARCH ──
    log(f"\n{'='*60}")
    log(f"GEN 0: RANDOM SEARCH ({GEN0_TRIALS} trials)")
    log(f"{'='*60}")
    t0 = time.time()

    for i in range(GEN0_TRIALS):
        cfg = random_config()
        r = simulate_portfolio(closes, timestamps, log_returns, windows, cfg, chainlink)
        all_results.append({
            "gen": 0, "trial": i, "config": asdict(cfg),
            "score": r.score, "pnl": r.total_pnl, "trades": r.total_trades,
            "wr": round(r.win_rate, 1), "sharpe": round(r.sharpe, 2),
            "dd": round(r.max_drawdown_pct, 1), "cl_pct": round(r.chainlink_pct, 1),
            "exits": r.exit_reasons,
        })

        if r.score > best_score:
            best_score = r.score
            best_cfg = cfg

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            log(f"  [{i+1}/{GEN0_TRIALS}] best={best_score:.1f} ({rate:.0f}/s)")

    log(f"Gen 0 done in {time.time()-t0:.0f}s. Best score: {best_score:.1f}")

    # Top N for Gen 1
    all_results.sort(key=lambda x: x["score"], reverse=True)
    top_gen0 = [Config(**r["config"]) for r in all_results[:TOP_N_GEN1]]
    log(f"\nTop {TOP_N_GEN1} Gen 0:")
    for i, r in enumerate(all_results[:TOP_N_GEN1]):
        c = r["config"]
        log(f"  {i+1}. score={r['score']:.1f} PnL=${r['pnl']:.0f} trades={r['trades']} "
            f"WR={r['wr']}% σ_scale={c['sigma_scale']} thr={c['entry_threshold']} "
            f"σ_win={c['sigma_window']} pos%={c['position_pct']} "
            f"edge_scale={c['edge_scaling']} CL={r['cl_pct']:.0f}%")

    # ── GEN 1: MUTATIONS ──
    log(f"\n{'='*60}")
    log(f"GEN 1: MUTATIONS ({GEN1_TRIALS} trials from top-{TOP_N_GEN1})")
    log(f"{'='*60}")
    t1 = time.time()

    gen1_results = []
    for i in range(GEN1_TRIALS):
        base = random.choice(top_gen0)
        cfg = mutate_config(base, intensity=0.25)
        r = simulate_portfolio(closes, timestamps, log_returns, windows, cfg, chainlink)
        entry = {
            "gen": 1, "trial": i, "config": asdict(cfg),
            "score": r.score, "pnl": r.total_pnl, "trades": r.total_trades,
            "wr": round(r.win_rate, 1), "sharpe": round(r.sharpe, 2),
            "dd": round(r.max_drawdown_pct, 1), "cl_pct": round(r.chainlink_pct, 1),
            "exits": r.exit_reasons,
        }
        gen1_results.append(entry)
        all_results.append(entry)

        if r.score > best_score:
            best_score = r.score
            best_cfg = cfg

        if (i + 1) % 100 == 0:
            log(f"  [{i+1}/{GEN1_TRIALS}] best={best_score:.1f}")

    log(f"Gen 1 done in {time.time()-t1:.0f}s. Best score: {best_score:.1f}")

    # Top N for Gen 2
    all_results.sort(key=lambda x: x["score"], reverse=True)
    top_gen1 = [Config(**r["config"]) for r in all_results[:TOP_N_GEN2]]

    # ── GEN 2: FINE-TUNING ──
    log(f"\n{'='*60}")
    log(f"GEN 2: FINE-TUNING ({GEN2_TRIALS} trials from top-{TOP_N_GEN2})")
    log(f"{'='*60}")
    t2 = time.time()

    for i in range(GEN2_TRIALS):
        base = random.choice(top_gen1)
        cfg = mutate_config(base, intensity=0.15)  # Finer mutations
        r = simulate_portfolio(closes, timestamps, log_returns, windows, cfg, chainlink)
        entry = {
            "gen": 2, "trial": i, "config": asdict(cfg),
            "score": r.score, "pnl": r.total_pnl, "trades": r.total_trades,
            "wr": round(r.win_rate, 1), "sharpe": round(r.sharpe, 2),
            "dd": round(r.max_drawdown_pct, 1), "cl_pct": round(r.chainlink_pct, 1),
            "exits": r.exit_reasons,
        }
        all_results.append(entry)

        if r.score > best_score:
            best_score = r.score
            best_cfg = cfg

        if (i + 1) % 100 == 0:
            log(f"  [{i+1}/{GEN2_TRIALS}] best={best_score:.1f}")

    log(f"Gen 2 done in {time.time()-t2:.0f}s. Best score: {best_score:.1f}")

    # ── WALK-FORWARD VALIDATION ──
    log(f"\n{'='*60}")
    log(f"WALK-FORWARD VALIDATION (top-{WF_TOP_N})")
    log(f"{'='*60}")

    all_results.sort(key=lambda x: x["score"], reverse=True)
    wf_results = []

    for i, r in enumerate(all_results[:WF_TOP_N]):
        cfg = Config(**r["config"])
        cfg.name = f"wf_{i+1}"
        is_r, oos_r = walk_forward(closes, timestamps, log_returns, windows, cfg, chainlink)
        c = r["config"]
        log(f"\n  #{i+1} σ_scale={c['sigma_scale']} thr={c['entry_threshold']} "
            f"σ_win={c['sigma_window']} pos%={c['position_pct']} "
            f"edge_scale={c['edge_scaling']}")
        log(f"    IS:  score={is_r.score:.1f} trades={is_r.total_trades} "
            f"WR={is_r.win_rate:.1f}% PnL=${is_r.total_pnl:.0f} "
            f"Sharpe={is_r.sharpe:.1f} DD={is_r.max_drawdown_pct:.1f}%")
        log(f"    OOS: score={oos_r.score:.1f} trades={oos_r.total_trades} "
            f"WR={oos_r.win_rate:.1f}% PnL=${oos_r.total_pnl:.0f} "
            f"Sharpe={oos_r.sharpe:.1f} DD={oos_r.max_drawdown_pct:.1f}%")
        log(f"    Exits: {oos_r.exit_reasons}")
        wf_results.append({
            "rank": i + 1,
            "config": c,
            "is": {"score": is_r.score, "trades": is_r.total_trades,
                   "wr": round(is_r.win_rate, 1), "pnl": round(is_r.total_pnl, 2),
                   "sharpe": round(is_r.sharpe, 2), "dd": round(is_r.max_drawdown_pct, 1)},
            "oos": {"score": oos_r.score, "trades": oos_r.total_trades,
                    "wr": round(oos_r.win_rate, 1), "pnl": round(oos_r.total_pnl, 2),
                    "sharpe": round(oos_r.sharpe, 2), "dd": round(oos_r.max_drawdown_pct, 1),
                    "exits": oos_r.exit_reasons},
        })

    # ── FINAL RESULTS ──
    log(f"\n{'='*60}")
    log(f"FINAL RESULTS")
    log(f"{'='*60}")
    log(f"Total trials: {len(all_results):,}")
    log(f"Best full-data score: {best_score:.1f}")
    if best_cfg:
        log(f"Best config:")
        log(f"  sigma_window:    {best_cfg.sigma_window}")
        log(f"  sigma_scale:     {best_cfg.sigma_scale}")
        log(f"  entry_threshold: {best_cfg.entry_threshold}")
        log(f"  position_pct:    {best_cfg.position_pct}")
        log(f"  edge_scaling:    {best_cfg.edge_scaling}")
        log(f"  spread:          {best_cfg.spread}")
        log(f"  min/max entry:   {best_cfg.min_entry_min}/{best_cfg.max_entry_min}")
        log(f"  reentry_cooldown:{best_cfg.reentry_cooldown}")

    # Save all
    output = {
        "meta": {
            "sweep": "b3_chainlink",
            "timestamp": datetime.now().isoformat(),
            "klines": len(closes),
            "windows": len(windows),
            "chainlink_resolutions": len(chainlink),
            "trials": len(all_results),
        },
        "walk_forward": wf_results,
        "all_trials": all_results[:100],  # Top 100 only
    }
    RESULTS_PATH.write_text(json.dumps(output, indent=2))
    log(f"\nResults saved: {RESULTS_PATH}")
    log(f"Log: {LOG_PATH}")

    total_time = time.time() - t0
    log(f"Total time: {total_time/60:.1f} min")


if __name__ == "__main__":
    random.seed(42)
    main()
