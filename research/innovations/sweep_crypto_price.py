"""
Crypto Price Edge Autoresearch
===============================

Optimizes Strategy B2 parameters using historical crypto price prediction
market data. Uses volatility-adjusted CDF model instead of EMOS.

Phases:
  Gen 0: Random volatility + gate + exit params (600 trials)
  Gen 1: Fine-tune top-10 (400 trials)
  Gen 2: Per-asset overrides — BTC vs ETH tuning (200 trials)
  Gen 3: Exit parameter tuning (200 trials)
  WF:    Walk-forward validation on best

Monkey-patches compute_prob() and check_exit() in experiment_framework
to use the volatility model and Binance exchange prices.

Usage:
    python3 research/innovations/sweep_crypto_price.py
"""

from __future__ import annotations

import json
import math
import random
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from crypto_price_loader import CryptoSimulationData, load_crypto_simulation_data

import experiment_framework as ef
from experiment_framework import (
    ExperimentResult,
    experiment_score,
    GAS_COST_USD,
    INITIAL_CAPITAL,
)

from arbo.models.volatility_model import (
    compute_realized_vol,
    estimate_crypto_prob,
)

# ═══════════════════════════════════════════════════════════════════════════════
# OUTPUT
# ═══════════════════════════════════════════════════════════════════════════════

RESULTS_DIR = Path(__file__).parent.parent / "data" / "experiments"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
SWEEP_PATH = RESULTS_DIR / "sweep_crypto_price.json"
LOG_PATH = Path(__file__).parent.parent / "sweep_crypto_price_log.txt"

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

GEN0_TRIALS = 1000
GEN1_TRIALS = 600
GEN2_TRIALS = 200   # Per-asset overrides
GEN3_TRIALS = 400   # Exit tuning
TOP_K = 15

# Multiple entry windows: scan markets at 1h, 2h, 4h, 8h, 12h, 24h, 48h before close
# This gives 7x more entry opportunities than single 24h
ENTRY_HOURS = [1, 2, 4, 8, 12, 24, 48]

# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL STATE (for monkey-patched functions)
# ═══════════════════════════════════════════════════════════════════════════════

_sim: CryptoSimulationData | None = None
_params: dict = {}
_vol_enabled: bool = False
_peak_prices: dict[str, float] = {}
_exit_slippage_pct: float = 0.02

# Save originals
_original_compute_prob = ef.compute_prob
_original_check_exit = ef.check_exit
_original_calc_exit_pnl = ef.calc_exit_pnl

# ═══════════════════════════════════════════════════════════════════════════════
# MONKEY-PATCHED FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════


def _compute_prob_crypto(forecast_temp, bucket, days_out, city, params):
    """Volatility-based probability for crypto price markets.

    Interface matches experiment_framework.compute_prob() signature.
    Reinterprets the arguments:
      forecast_temp → exchange price (from forecasts[asset][date])
      bucket.low_c  → strike price
      bucket.bucket_type → direction ("above" or "below")
      city          → asset (BTC, ETH)
      days_out      → used to compute hours_to_expiry
    """
    if not _vol_enabled or _sim is None:
        return _original_compute_prob(forecast_temp, bucket, days_out, city, params)

    exchange_price = forecast_temp
    strike = bucket.low_c  # strike stored in low_c field
    direction = bucket.bucket_type or "above"

    if exchange_price is None or exchange_price <= 0 or strike is None or strike <= 0:
        return 0.0

    # Convert days_out to hours
    hours_to_expiry = max(days_out * 24, 1)

    # Get volatility params
    vol_window = params.get("volatility_window", 24)
    vol_method = params.get("volatility_method", "realized")
    sigma_scale = params.get("sigma_scale", 1.0)

    # Compute hourly sigma from historical prices
    # Use forecast_temp as current price, look up volatility from Binance history
    symbol = f"{city}USDT"  # city = asset (e.g. "BTC" → "BTCUSDT")

    # Get recent prices for volatility computation
    # For backtesting, we pre-computed this from Binance klines
    # Use a simple proxy: log return std from exchange price vs strike
    sigma_hourly = params.get("_sigma_cache", {}).get(city, 0.02)

    # Market type from bucket metadata
    # bucket.unit = "USD" for crypto, and we store market_type info in the question
    question = (bucket.question or "").lower()
    market_type = "monthly_hit" if "hit" in question or "what price" in question else "daily_above"

    prob = estimate_crypto_prob(
        current_price=exchange_price,
        strike=strike,
        hours_to_expiry=hours_to_expiry,
        sigma_per_hour=sigma_hourly,
        market_type=market_type,
        sigma_scale=sigma_scale,
        direction=direction,
    )

    # Post-processing: shrinkage + sharpening
    asset = city  # city = asset
    asset_ov = params.get("city_overrides", {}).get(asset, {})
    shrinkage = asset_ov.get("shrinkage", params.get("shrinkage", 0.0))
    prob = prob * (1.0 - shrinkage) + 0.5 * shrinkage  # Shrink toward 0.5 (uninformative)

    sharpening = asset_ov.get("prob_sharpening", params.get("prob_sharpening", 1.0))
    if sharpening != 1.0 and 0 < prob < 1:
        prob = prob ** sharpening

    return max(0.001, min(0.999, prob))


def _check_exit_crypto(pos, current_price, hour_ts, params):
    """Edge-based exit for crypto: re-estimate prob with latest info."""
    if not params.get("exit_enabled", False):
        return None

    # Track peak price
    token_id = pos.token_id
    if token_id not in _peak_prices:
        _peak_prices[token_id] = pos.entry_price
    _peak_prices[token_id] = max(_peak_prices[token_id], current_price)

    # Recompute probability with updated time
    hours_to_close = max(0, (pos.closes_at_ts - hour_ts) / 3600)
    days_out = max(0, int(hours_to_close / 24))

    updated_prob = _compute_prob_crypto(
        pos.forecast_temp, pos.bucket, days_out, pos.city, params
    )
    updated_edge = updated_prob - current_price

    # Edge-based exit
    min_hold_edge = params.get("min_hold_edge", 0.03)
    if updated_edge < min_hold_edge:
        return "edge_lost"

    # Probability floor
    prob_floor = params.get("prob_exit_floor", 0.0)
    if prob_floor > 0 and updated_prob < prob_floor:
        return "prob_floor"

    # Profit take (absolute price gain)
    if params.get("profit_take_also", True):
        target = params.get("profit_target_abs", 0.10)
        if current_price >= pos.entry_price + target:
            return "profit_take"

    return None


def _crypto_fee(price: float) -> float:
    """Polymarket crypto taker fee: price * (1-price) * 0.25."""
    if price <= 0 or price >= 1:
        return 0.0
    return price * (1 - price) * 0.25


def _calc_resolution_pnl_crypto(pos):
    """Resolution P&L — maker entry (0% fee), resolution = no trade = no fee."""
    # PostOnly maker entry: 0% fee (+ rebate, but we ignore rebate for conservative estimate)
    # Resolution: shares become worth $1 or $0, no trading fee
    if pos.bucket.won:
        return pos.size * (1.0 / pos.entry_fill - 1.0) - GAS_COST_USD
    else:
        return -pos.size - GAS_COST_USD


def _calc_exit_pnl_crypto(pos, exit_price):
    """Exit P&L — maker entry (0% fee), TAKER exit (pays crypto fee)."""
    # Entry: PostOnly maker = 0% fee
    # Exit: taker sell = pays crypto fee
    exit_fee = _crypto_fee(exit_price)
    exit_fill = exit_price * (1 - _exit_slippage_pct) - exit_fee

    tokens = pos.size / pos.entry_fill
    proceeds = tokens * max(exit_fill, 0.001)
    return proceeds - pos.size - GAS_COST_USD


def _crypto_score(result) -> float:
    """Crypto-optimized scoring: rewards high profit AND high trade count.

    Compared to default:
    - Stronger trade count bonus (saturates at 500 not 100)
    - Turnover rate bonus
    - Lower Sharpe floor (crypto is volatile)
    """
    if result.trades < 5:
        return 0.0
    if result.max_drawdown_pct > 60:
        return 0.0
    if result.sharpe < 0.3:
        return 0.0

    # Profitability (45%)
    roi_score = min(result.roi_pct / 200, 2.0)                     # 20%
    sharpe_score = min(max(result.sharpe, 0) / 5.0, 2.0)           # 15%
    dd_score = max(0, 1.0 - result.max_drawdown_pct / 50)          # 10%

    # Capital Turnover (35%) — HIGHER weight for crypto
    util_score = min(result.capital_utilization / 10, 2.0)          # 15%
    pph = max(result.avg_pnl_per_hour, 0)
    pph_score = min(pph / 3.0, 2.0)                                # 10%
    trade_score = min(result.trades / 500, 2.0)                     # 10%

    # Validation (20%)
    if result.oos_pnl is not None and result.oos_pnl > 0:
        oos_score = min(result.oos_pnl / 500, 2.0)                 # 10%
    else:
        oos_score = 0
    # Reward consistent profitability
    wr_score = min(max(result.win_rate - 40, 0) / 30, 2.0)         # 10%

    return round(
        roi_score * 20
        + sharpe_score * 15
        + dd_score * 10
        + util_score * 15
        + pph_score * 10
        + trade_score * 10
        + oos_score * 10
        + wr_score * 10,
        2,
    )


def install_patches():
    """Install crypto probability + exit patches + scoring."""
    ef.compute_prob = _compute_prob_crypto
    ef.check_exit = _check_exit_crypto
    ef.calc_exit_pnl = _calc_exit_pnl_crypto
    ef.calc_resolution_pnl = _calc_resolution_pnl_crypto
    ef.experiment_score = _crypto_score  # Custom scoring for high-frequency crypto

    # Add crypto assets to CITY_COORDS so experiment_framework doesn't skip them
    ef.CITY_COORDS["BTC"] = (0, 0)
    ef.CITY_COORDS["ETH"] = (0, 0)
    ef.CITY_COORDS["SOL"] = (0, 0)


def reset_state():
    _peak_prices.clear()


# ═══════════════════════════════════════════════════════════════════════════════
# VOLATILITY PRECOMPUTATION
# ═══════════════════════════════════════════════════════════════════════════════


def precompute_volatility(sim: CryptoSimulationData, params: dict) -> dict[str, float]:
    """Precompute hourly sigma per asset from Binance klines.

    Returns dict[asset, sigma_hourly].
    """
    vol_window = params.get("volatility_window", 24)
    vol_method = params.get("volatility_method", "realized")

    sigmas: dict[str, float] = {}
    for asset_name in set(ev.city for ev in sim.events if ev.city):
        symbol = f"{asset_name}USDT"
        klines = sim._binance.get(symbol, [])
        if len(klines) < 2:
            sigmas[asset_name] = 0.02  # Default
            continue

        # Use hourly close prices (sample every 60 minutes from 1-min klines)
        hourly_prices: list[float] = []
        last_hour = -1
        for ts, close in klines:
            hour = ts // 3600
            if hour != last_hour:
                hourly_prices.append(close)
                last_hour = hour

        sigma = compute_realized_vol(hourly_prices, vol_window, vol_method)
        sigmas[asset_name] = sigma

    return sigmas


# ═══════════════════════════════════════════════════════════════════════════════
# PARAMETER GENERATION
# ═══════════════════════════════════════════════════════════════════════════════


def random_params(rng: random.Random) -> dict:
    """Random volatility + gate + exit parameters.

    Expanded parameter space for high-frequency crypto trading:
    - Lower edge thresholds (crypto is efficient, edges are small)
    - Higher position limits (deep liquidity)
    - Aggressive sizing
    - Wide price range (trade near-money AND tails)
    """
    return {
        # Volatility model
        "volatility_window": rng.choice([6, 12, 24, 48, 72, 96, 168, 336]),
        "volatility_method": rng.choice(["realized", "ewma", "garch"]),
        "sigma_scale": rng.choice([0.3, 0.5, 0.7, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0]),
        # Exit
        "exit_enabled": rng.choice([True, True, True, False]),  # Sometimes hold to resolution
        "min_hold_edge": rng.choice([-0.05, -0.02, 0.0, 0.005, 0.01, 0.02, 0.03, 0.05]),
        "exit_slippage_pct": rng.choice([0.005, 0.01, 0.015, 0.02, 0.03]),
        "prob_exit_floor": rng.choice([0.0, 0.0, 0.0, 0.05, 0.10]),
        "profit_take_also": rng.choice([True, True, False]),
        "profit_target_abs": rng.choice([0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30]),
        # Quality gate — LOOSER for more trades
        "min_edge": rng.choice([0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.08]),
        "max_edge": rng.choice([0.30, 0.40, 0.50, 0.70, 0.90]),
        "max_price": rng.choice([0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]),
        "min_price": rng.choice([0.01, 0.02, 0.03, 0.05, 0.10]),
        "min_prob": rng.choice([0.01, 0.02, 0.05, 0.10]),
        "min_volume": 0,  # PMD doesn't have volume, always pass
        "kelly_raw_cap": rng.choice([0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]),
        "prob_sharpening": rng.choice([0.70, 0.80, 0.85, 0.90, 0.95, 1.0, 1.05, 1.10, 1.20]),
        "shrinkage": rng.choice([0.0, 0.01, 0.02, 0.03, 0.05]),
        # Aggressive capital deployment for high turnover
        "max_aggregate_pct": rng.choice([0.40, 0.50, 0.60, 0.70, 0.80, 0.90]),
        "city_max_exposure": rng.choice([0.30, 0.40, 0.50, 0.60, 0.80, 1.0]),
        "no_compounding": rng.choice([True, False]),  # Allow compounding for growth
        "max_position_pct": rng.choice([0.03, 0.05, 0.08, 0.10, 0.15]),
        # Time filter — allow short-term trades
        "min_time_to_expiry_h": rng.choice([0, 1, 2, 4, 8]),
        # Re-entry cooldown
        "reentry_cooldown_h": rng.choice([0, 0, 0, 1, 2, 4]),
        # Asset
        "excluded_cities": set(),
        "city_overrides": {},
    }


def perturb_params(base: dict, rng: random.Random) -> dict:
    """Perturb 1-4 params from base."""
    p = dict(base)
    p["excluded_cities"] = set(p.get("excluded_cities", set()))
    p["city_overrides"] = dict(p.get("city_overrides", {}))

    mutations = {
        "volatility_window": [6, 12, 24, 48, 72, 96, 168, 336],
        "sigma_scale": [0.3, 0.5, 0.7, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0],
        "min_hold_edge": [-0.05, -0.02, 0.0, 0.005, 0.01, 0.02, 0.03, 0.05],
        "exit_slippage_pct": [0.005, 0.01, 0.015, 0.02, 0.03],
        "profit_target_abs": [0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30],
        "min_edge": [0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.08],
        "max_edge": [0.30, 0.40, 0.50, 0.70, 0.90],
        "max_price": [0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95],
        "min_price": [0.01, 0.02, 0.03, 0.05, 0.10],
        "min_prob": [0.01, 0.02, 0.05, 0.10],
        "kelly_raw_cap": [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50],
        "prob_sharpening": [0.70, 0.80, 0.85, 0.90, 0.95, 1.0, 1.05, 1.10, 1.20],
        "max_aggregate_pct": [0.40, 0.50, 0.60, 0.70, 0.80, 0.90],
        "max_position_pct": [0.03, 0.05, 0.08, 0.10, 0.15],
        "min_time_to_expiry_h": [0, 1, 2, 4, 8],
        "reentry_cooldown_h": [0, 0, 1, 2, 4],
        "city_max_exposure": [0.30, 0.40, 0.50, 0.60, 0.80, 1.0],
    }

    n = rng.randint(1, 4)
    keys = rng.sample(list(mutations.keys()), min(n, len(mutations)))
    for key in keys:
        pool = mutations[key]
        current = p.get(key)
        candidates = [v for v in pool if v != current]
        if candidates:
            p[key] = rng.choice(candidates)

    # Occasionally flip method
    if rng.random() < 0.15:
        p["volatility_method"] = rng.choice(["realized", "ewma", "garch"])
    if rng.random() < 0.10:
        p["profit_take_also"] = not p.get("profit_take_also", True)

    return p


# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT RUNNER
# ═══════════════════════════════════════════════════════════════════════════════


def run_experiment(sim_data, params, experiment_id, entry_hours=None):
    """Run crypto price experiment with volatility model."""
    global _vol_enabled, _params, _exit_slippage_pct

    if entry_hours is None:
        entry_hours = ENTRY_HOURS

    _vol_enabled = True
    _params = params
    _exit_slippage_pct = params.get("exit_slippage_pct", 0.02)
    reset_state()

    # Precompute volatility for this param set
    sigmas = precompute_volatility(sim_data, params)
    params["_sigma_cache"] = sigmas

    try:
        result = ef.simulate_portfolio(
            sim_data, params,
            entry_hours=entry_hours,
            experiment_id=experiment_id,
        )
        _vol_enabled = False
        return result
    except Exception as e:
        _vol_enabled = False
        print(f"  ERROR {experiment_id}: {e}", flush=True)
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# WALK-FORWARD (adapted for crypto SimulationData)
# ═══════════════════════════════════════════════════════════════════════════════


def walk_forward_validate(sim_data, params, entry_hours=24, n_folds=3):
    """Walk-forward cross-validation for crypto data."""
    events = sorted(
        [ev for ev in sim_data.events if ev.target_date],
        key=lambda e: e.target_date,
    )
    if len(events) < n_folds + 1:
        return {"score": 0, "oos_pnl": 0, "n_folds": 0, "total_trades": 0, "folds": []}

    fold_size = len(events) // (n_folds + 1)
    folds = []
    total_pnl = 0
    total_trades = 0

    for fold in range(n_folds):
        test_start = (fold + 1) * fold_size
        test_end = (fold + 2) * fold_size
        test_events = events[test_start:test_end]
        if not test_events:
            continue

        min_date = test_events[0].target_date
        max_date = test_events[-1].target_date
        test_sim = sim_data.filter_events(min_date=min_date, max_date=max_date)

        result = run_experiment(test_sim, params, f"wf_fold{fold}", entry_hours)
        if result:
            folds.append({
                "fold": fold,
                "pnl": result.total_pnl,
                "trades": result.trades,
                "wr": result.win_rate,
                "score": result.score,
            })
            total_pnl += result.total_pnl
            total_trades += result.trades

    avg_pnl = total_pnl / max(len(folds), 1)
    avg_score = sum(f["score"] for f in folds) / max(len(folds), 1) if folds else 0

    return {
        "score": avg_score,
        "oos_pnl": avg_pnl,
        "n_folds": len(folds),
        "total_trades": total_trades,
        "folds": folds,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    global _sim, _vol_enabled, _params, _exit_slippage_pct

    t_start = time.time()
    log_f = open(LOG_PATH, "w", buffering=1)

    def log(msg):
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        log_f.write(line + "\n")

    log("Crypto Price Edge Autoresearch — Strategy B2")
    log("Loading data...")

    sim_data = load_crypto_simulation_data()
    _sim = sim_data

    log(f"Data: {len(sim_data.events)} events, "
        f"{sum(len(b) for b in sim_data.buckets_by_event.values())} buckets")

    install_patches()
    log("Patches installed: volatility compute_prob + crypto check_exit")

    # ── Tracking ──
    all_results: list[dict] = []
    top_results: list[tuple[float, dict, ExperimentResult]] = []
    best_score = 0.0
    best_params: dict = {}
    best_result: ExperimentResult | None = None
    trial_n = 0

    def record(params, result, phase):
        nonlocal best_score, best_params, best_result, trial_n
        trial_n += 1
        score = result.score

        all_results.append({
            "trial": trial_n, "phase": phase, "score": score,
            "trades": result.trades, "win_rate": result.win_rate,
            "total_pnl": result.total_pnl, "roi_pct": result.roi_pct,
            "sharpe": result.sharpe, "max_drawdown_pct": result.max_drawdown_pct,
            "capital_utilization": result.capital_utilization,
            "total_exits": result.total_exits,
            "params": _ser(params),
        })

        top_results.append((score, params, result))
        top_results.sort(key=lambda x: -x[0])
        if len(top_results) > TOP_K * 3:
            top_results[:] = top_results[:TOP_K * 3]

        if score > best_score:
            best_score = score
            best_params = params
            best_result = result
            log(f"  ★ NEW BEST: score={score:.1f}  trades={result.trades}  "
                f"WR={result.win_rate:.1f}%  PnL=${result.total_pnl:.0f}  "
                f"Sharpe={result.sharpe:.2f}  DD={result.max_drawdown_pct:.1f}%")
            p = params
            log(f"    VOL: window={p.get('volatility_window')}h, "
                f"method={p.get('volatility_method')}, "
                f"σ_scale={p.get('sigma_scale')}")
            log(f"    GATE: edge≥{p.get('min_edge')}, "
                f"p=[{p.get('min_price')},{p.get('max_price')}], "
                f"kelly={p.get('kelly_raw_cap')}, "
                f"min_h={p.get('min_time_to_expiry_h')}")
            log(f"    EXIT: mhe={p.get('min_hold_edge')}, "
                f"pt={p.get('profit_target_abs')}, "
                f"slip={p.get('exit_slippage_pct')}")

    # ══════════════════════════════════════════════════════════════════════
    # GEN 0: RANDOM SEARCH
    # ══════════════════════════════════════════════════════════════════════

    log(f"\n=== GEN 0: RANDOM SEARCH ({GEN0_TRIALS} trials) ===")
    rng = random.Random(42)

    for i in range(GEN0_TRIALS):
        params = random_params(rng)
        result = run_experiment(sim_data, params, f"g0_{i:04d}")
        if result:
            record(params, result, "gen0")
        if (i + 1) % 100 == 0:
            scored = sum(1 for r in all_results if r["score"] > 0)
            log(f"  Gen 0: {i+1}/{GEN0_TRIALS}  best={best_score:.1f}  scored={scored}")

    log(f"Gen 0 done: best={best_score:.1f}")

    # ══════════════════════════════════════════════════════════════════════
    # GEN 1: FINE-TUNE TOP-K
    # ══════════════════════════════════════════════════════════════════════

    log(f"\n=== GEN 1: FINE-TUNE ({GEN1_TRIALS} trials, top-{TOP_K}) ===")
    top_k = [p for _, p, _ in top_results[:TOP_K]]

    for i in range(GEN1_TRIALS):
        base = top_k[i % len(top_k)] if top_k else random_params(rng)
        params = perturb_params(base, rng)
        result = run_experiment(sim_data, params, f"g1_{i:04d}")
        if result:
            record(params, result, "gen1")
        if (i + 1) % 100 == 0:
            log(f"  Gen 1: {i+1}/{GEN1_TRIALS}  best={best_score:.1f}")

    log(f"Gen 1 done: best={best_score:.1f}")

    # ══════════════════════════════════════════════════════════════════════
    # GEN 2: PER-ASSET OVERRIDES (BTC vs ETH)
    # ══════════════════════════════════════════════════════════════════════

    log(f"\n=== GEN 2: PER-ASSET OVERRIDES ({GEN2_TRIALS} trials) ===")

    # Get per-asset results from best model
    asset_pnl: dict[str, float] = {}
    asset_trades: dict[str, int] = {}
    if best_result and best_result.city_results:
        for asset_name, cr in best_result.city_results.items():
            asset_pnl[asset_name] = cr.get("pnl", 0)
            asset_trades[asset_name] = cr.get("trades", 0)

    losing_assets = sorted([a for a, p in asset_pnl.items() if p < 0],
                           key=lambda a: asset_pnl.get(a, 0))
    log(f"  Losing assets: {losing_assets}")

    top_k = [p for _, p, _ in top_results[:TOP_K]]

    # Test excluding losing assets
    trial_i = 0
    for asset_name in losing_assets:
        if trial_i >= GEN2_TRIALS // 2:
            break
        for base in top_k[:5]:
            if trial_i >= GEN2_TRIALS // 2:
                break
            params = dict(base)
            params["excluded_cities"] = set(base.get("excluded_cities", set())) | {asset_name}
            params["city_overrides"] = dict(base.get("city_overrides", {}))
            result = run_experiment(sim_data, params, f"g2_excl_{trial_i:04d}")
            if result:
                record(params, result, "gen2_excl")
            trial_i += 1

    # Tune per-asset overrides
    tunable_assets = [a for a, t in asset_trades.items()
                      if t >= 5 and a not in best_params.get("excluded_cities", set())]

    while trial_i < GEN2_TRIALS:
        base = rng.choice(top_k) if top_k else random_params(rng)
        params = dict(base)
        params["excluded_cities"] = set(base.get("excluded_cities", set()))
        params["city_overrides"] = dict(base.get("city_overrides", {}))

        if tunable_assets:
            asset_name = rng.choice(tunable_assets)
            params["city_overrides"][asset_name] = {
                "min_edge": rng.choice([0.01, 0.02, 0.03, 0.05, 0.08]),
                "max_price": rng.choice([0.30, 0.40, 0.50, 0.60, 0.70]),
                "min_price": rng.choice([0.01, 0.03, 0.05]),
                "kelly_raw_cap": rng.choice([0.10, 0.15, 0.20, 0.25, 0.30]),
                "prob_sharpening": rng.choice([0.85, 0.90, 0.95, 1.0, 1.05]),
            }

        result = run_experiment(sim_data, params, f"g2_ov_{trial_i:04d}")
        if result:
            record(params, result, "gen2_override")
        trial_i += 1
        if trial_i % 50 == 0:
            log(f"  Gen 2: {trial_i}/{GEN2_TRIALS}  best={best_score:.1f}")

    log(f"Gen 2 done: best={best_score:.1f}")

    # ══════════════════════════════════════════════════════════════════════
    # GEN 3: EXIT PARAMETER TUNING
    # ══════════════════════════════════════════════════════════════════════

    log(f"\n=== GEN 3: EXIT TUNING ({GEN3_TRIALS} trials) ===")
    top_k = [p for _, p, _ in top_results[:TOP_K]]

    for i in range(GEN3_TRIALS):
        base = top_k[i % len(top_k)] if top_k else random_params(rng)
        params = dict(base)
        params["excluded_cities"] = set(base.get("excluded_cities", set()))
        params["city_overrides"] = dict(base.get("city_overrides", {}))

        # Focus perturbation on exit params
        exit_mutations = {
            "min_hold_edge": [0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10],
            "profit_target_abs": [0.03, 0.05, 0.08, 0.10, 0.15, 0.20],
            "exit_slippage_pct": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
            "prob_exit_floor": [0.0, 0.05, 0.10, 0.15],
        }
        n = rng.randint(1, 3)
        keys = rng.sample(list(exit_mutations.keys()), min(n, len(exit_mutations)))
        for key in keys:
            params[key] = rng.choice(exit_mutations[key])

        if rng.random() < 0.3:
            params["profit_take_also"] = not params.get("profit_take_also", True)

        # Also sometimes perturb gate params
        if rng.random() < 0.2:
            params = perturb_params(params, rng)

        result = run_experiment(sim_data, params, f"g3_{i:04d}")
        if result:
            record(params, result, "gen3")
        if (i + 1) % 50 == 0:
            log(f"  Gen 3: {i+1}/{GEN3_TRIALS}  best={best_score:.1f}")

    log(f"Gen 3 done: best={best_score:.1f}")

    # ══════════════════════════════════════════════════════════════════════
    # WALK-FORWARD VALIDATION
    # ══════════════════════════════════════════════════════════════════════

    log("\n=== WALK-FORWARD VALIDATION ===")
    if best_result:
        wf = walk_forward_validate(sim_data, best_params, entry_hours=24)
        log(f"WF: OOS score={wf['score']:.1f}  OOS PnL=${wf['oos_pnl']:.0f}  "
            f"folds={wf['n_folds']}  trades={wf['total_trades']}")
        for fold in wf.get("folds", []):
            log(f"  Fold {fold['fold']}: PnL=${fold['pnl']:.0f}  "
                f"trades={fold['trades']}  WR={fold['wr']:.1f}%")
    else:
        wf = {"score": 0, "oos_pnl": 0, "n_folds": 0, "total_trades": 0}

    # ══════════════════════════════════════════════════════════════════════
    # RESULTS
    # ══════════════════════════════════════════════════════════════════════

    elapsed = time.time() - t_start

    # Per-asset from best
    if best_result and best_result.city_results:
        log("\n=== BEST MODEL — PER-ASSET ===")
        log(f"  {'Asset':<10} {'Trades':>6} {'WR':>6} {'PnL':>10}")
        log(f"  {'-'*36}")
        for asset_name in sorted(best_result.city_results.keys(),
                                 key=lambda a: -best_result.city_results[a].get("pnl", 0)):
            cr = best_result.city_results[asset_name]
            log(f"  {asset_name:<10} {cr.get('trades', 0):>6} "
                f"{cr.get('win_rate', 0):>5.1f}% ${cr.get('pnl', 0):>9.0f}")

    log(f"\n{'='*80}")
    log("FINAL SUMMARY")
    log(f"{'='*80}")
    if best_result:
        log(f"  Best score:       {best_score:.1f}")
        log(f"  Trades:           {best_result.trades}")
        log(f"  Win rate:         {best_result.win_rate:.1f}%")
        log(f"  Total PnL:        ${best_result.total_pnl:.0f}")
        log(f"  ROI:              {best_result.roi_pct:.1f}%")
        log(f"  Sharpe:           {best_result.sharpe:.2f}")
        log(f"  Max drawdown:     {best_result.max_drawdown_pct:.1f}%")
        log(f"  Exits:            {best_result.total_exits}")
        log(f"  OOS PnL:          ${wf.get('oos_pnl', 0):.0f}")
        log(f"  OOS score:        {wf.get('score', 0):.1f}")
        excl = best_params.get("excluded_cities", set())
        log(f"  Excluded assets:  {sorted(excl) if excl else 'none'}")
    log(f"  Trials:           {trial_n}")
    log(f"  Runtime:          {elapsed:.0f}s ({elapsed/60:.1f}min)")

    # ── Save JSON ──
    output = {
        "sweep_id": "crypto_price_b2",
        "sweep_type": "Crypto Price Edge Autoresearch — Strategy B2",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_trials": trial_n,
        "runtime_seconds": round(elapsed, 1),
        "best": {
            "score": best_score,
            "params": _ser(best_params),
        },
        "walk_forward": wf,
        "top_results": [
            {"rank": i + 1, "score": s, "params": _ser(p)}
            for i, (s, p, _) in enumerate(top_results[:20])
        ],
        "all_results_count": len(all_results),
    }

    with open(SWEEP_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)
    log(f"\nResults saved to {SWEEP_PATH}")
    log_f.close()


def _ser(params):
    """JSON-serializable params (strip internal caches)."""
    out = {}
    for k, v in params.items():
        if k.startswith("_"):
            continue  # Skip internal caches
        if isinstance(v, set):
            out[k] = sorted(v)
        elif isinstance(v, dict):
            out[k] = {str(kk): vv for kk, vv in v.items()}
        else:
            out[k] = v
    return out


if __name__ == "__main__":
    main()
