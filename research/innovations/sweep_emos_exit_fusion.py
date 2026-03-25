"""
EMOS + Edge Exit Fusion Autoresearch
=====================================

Combines the two best innovations:
  1. EMOS adaptive probability model (from sweep_emos_v2)
  2. Edge-based early exit (winner from sweep_early_exit)

Phases:
  Gen 0: Random EMOS + exit params (600 trials)
  Gen 1: Fine-tune top-10 (400 trials)
  Gen 2: City exclusion sweep — test every losing city combination (200)
  Gen 3: Per-city override tuning (200 trials)
  WF:    Walk-forward validation on best

Uses PMD 10-min price data. Monkey-patches both compute_prob (EMOS)
and check_exit (edge-based exit).

Usage:
    python3 research/innovations/sweep_emos_exit_fusion.py
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

from emos_model import EMOSModel, Observation, actual_temp_from_bucket
from pmd_loader import load_pmd_data

import experiment_framework as ef
from experiment_framework import (
    CITY_COORDS,
    SimulationData,
    ExperimentResult,
    experiment_score,
    load_forecasts,
    serialize_result,
    walk_forward_validate,
    GAS_COST_USD,
    INITIAL_CAPITAL,
)

# ═══════════════════════════════════════════════════════════════════════════════
# OUTPUT
# ═══════════════════════════════════════════════════════════════════════════════

RESULTS_DIR = Path(__file__).parent.parent / "data" / "experiments"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
SWEEP_PATH = RESULTS_DIR / "sweep_emos_exit_fusion.json"
LOG_PATH = Path(__file__).parent.parent / "sweep_emos_exit_fusion_log.txt"

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

GEN0_TRIALS = 600
GEN1_TRIALS = 400
GEN2_TRIALS = 200    # City exclusion
GEN3_TRIALS = 200    # Per-city overrides
TOP_K = 10

# ═══════════════════════════════════════════════════════════════════════════════
# EMOS STATE (from sweep_emos_v2)
# ═══════════════════════════════════════════════════════════════════════════════

_emos_models: dict[str, EMOSModel] = {}
_emos_obs: dict[str, list[Observation]] = {}
_emos_params: dict = {}
_emos_enabled: bool = False

# ═══════════════════════════════════════════════════════════════════════════════
# EDGE EXIT STATE (from sweep_early_exit)
# ═══════════════════════════════════════════════════════════════════════════════

_peak_prices: dict[str, float] = {}
_exit_slippage_pct: float = 0.01

# ═══════════════════════════════════════════════════════════════════════════════
# EMOS FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════


def _build_training_observations(
    events: list, buckets_by_event: dict, forecasts: dict,
) -> dict[str, list[Observation]]:
    """Build (forecast, actual) pairs per city from resolved events."""
    city_obs: dict[str, list[Observation]] = defaultdict(list)
    for ev in events:
        if not ev.city or not ev.target_date:
            continue
        forecast_temp = forecasts.get(ev.city, {}).get(ev.target_date)
        if forecast_temp is None:
            continue
        buckets = buckets_by_event.get(ev.event_id, [])
        actual_temp = None
        for bucket in buckets:
            if bucket.won:
                actual_temp = actual_temp_from_bucket(
                    bucket.low_c, bucket.high_c, bucket.bucket_type
                )
                break
        if actual_temp is None:
            continue
        city_obs[ev.city].append(
            Observation(forecast=forecast_temp, actual=actual_temp,
                        date=ev.target_date, city=ev.city)
        )
    for city in city_obs:
        city_obs[city].sort(key=lambda o: o.date)
    return dict(city_obs)


def _get_emos_model(city: str, target_date: str) -> EMOSModel:
    """Walk-forward EMOS model (fit only on prior data)."""
    cache_key = f"{city}:{target_date}"
    if cache_key in _emos_models:
        return _emos_models[cache_key]
    obs = _emos_obs.get(city, [])
    prior = [o for o in obs if o.date < target_date]
    model = EMOSModel(
        training_window=_emos_params.get("emos_training_window", 30),
        sigma_method=_emos_params.get("emos_sigma_method", "rolling_rmse"),
        bias_method=_emos_params.get("emos_bias_method", "rolling_mean"),
        sigma_floor=_emos_params.get("emos_sigma_floor", 0.5),
        sigma_scale=_emos_params.get("emos_sigma_scale", 1.0),
        ewma_alpha=_emos_params.get("emos_ewma_alpha", 0.1),
    )
    model.fit(prior)
    _emos_models[cache_key] = model
    return model


def _prefit_emos_all(params: dict) -> None:
    """Pre-fit EMOS for all dates per city (walk-forward safe)."""
    global _emos_models
    _emos_models = {}
    for city, obs in _emos_obs.items():
        dates = sorted(set(o.date for o in obs))
        for target_date in dates:
            _get_emos_model(city, target_date)
        if obs:
            _get_emos_model(city, "9999-99-99")


# ═══════════════════════════════════════════════════════════════════════════════
# MONKEY PATCHES — EMOS compute_prob + Edge exit check_exit
# ═══════════════════════════════════════════════════════════════════════════════

_original_compute_prob = ef.compute_prob
_original_check_exit = ef.check_exit
_original_calc_exit_pnl = ef.calc_exit_pnl


def _compute_prob_emos(forecast_temp, bucket, days_out, city, params):
    """EMOS-enhanced probability."""
    if not _emos_enabled:
        return _original_compute_prob(forecast_temp, bucket, days_out, city, params)

    obs = _emos_obs.get(city, [])
    model = None
    if obs:
        model = _get_emos_model(city, obs[-1].date if obs else "9999-99-99")

    if model is None or not model._fitted:
        return _original_compute_prob(forecast_temp, bucket, days_out, city, params)

    raw = model.bucket_probability(
        forecast_temp, bucket.low_c, bucket.high_c,
        bucket.bucket_type or "range",
    )

    # Post-processing
    city_ov = params.get("city_overrides", {}).get(city, {})
    shrinkage = city_ov.get("shrinkage", params.get("shrinkage", 0.0))
    raw = raw * (1.0 - shrinkage) + 0.125 * shrinkage

    sharpening = city_ov.get("prob_sharpening", params.get("prob_sharpening", 1.0))
    if sharpening != 1.0 and raw > 0:
        raw = raw ** sharpening

    return raw


def _check_exit_edge(pos, current_price, hour_ts, params):
    """Edge-based exit: sell when model says edge < threshold."""
    if not params.get("exit_enabled", False):
        return None

    # Track peak for optional trailing
    token_id = pos.token_id
    if token_id not in _peak_prices:
        _peak_prices[token_id] = pos.entry_price
    _peak_prices[token_id] = max(_peak_prices[token_id], current_price)

    # Updated probability with time-decayed sigma
    hours_to_close = max(0, (pos.closes_at_ts - hour_ts) / 3600)
    days_out = max(0, int(hours_to_close / 24))

    updated_prob = _compute_prob_emos(
        pos.forecast_temp, pos.bucket, days_out, pos.city, params
    )
    updated_edge = updated_prob - current_price

    # Edge-based exit
    min_hold_edge = params.get("min_hold_edge", 0.10)
    if updated_edge < min_hold_edge:
        return "edge_lost"

    # Optional probability floor
    prob_floor = params.get("prob_exit_floor", 0.0)
    if prob_floor > 0 and updated_prob < prob_floor:
        return "prob_floor"

    # Optional profit take (price-based, as supplement to edge)
    if params.get("profit_take_also", False):
        target = params.get("profit_target_abs", 0.15)
        if current_price >= pos.entry_price + target:
            return "profit_take"

    return None


def _calc_exit_pnl_custom(pos, exit_price):
    """Exit P&L with configurable slippage."""
    exit_fill = exit_price * (1 - _exit_slippage_pct)
    tokens = pos.size / pos.entry_fill
    proceeds = tokens * exit_fill
    return proceeds - pos.size - GAS_COST_USD


def install_patches():
    """Install both EMOS + edge exit patches."""
    ef.compute_prob = _compute_prob_emos
    ef.check_exit = _check_exit_edge
    ef.calc_exit_pnl = _calc_exit_pnl_custom


def reset_state():
    _peak_prices.clear()


# ═══════════════════════════════════════════════════════════════════════════════
# PARAMETER GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

# Winning edge-exit params from sweep_early_exit (base for EMOS overlay)
_EDGE_EXIT_BASE = {
    "exit_enabled": True,
    "min_hold_edge": 0.10,
    "exit_slippage_pct": 0.01,
    "min_edge": 0.01,
    "max_edge": 0.90,
    "max_price": 0.45,
    "min_price": 0.01,
    "min_prob": 0.02,
    "min_volume": 0,
    "prob_sharpening": 0.92,
    "kelly_raw_cap": 0.18,
    "shrinkage": 0.0,
    "max_aggregate_pct": 0.60,
    "city_max_exposure": 0.25,
    "no_compounding": True,
    "max_position_pct": 0.05,
}


def random_params(rng: random.Random) -> dict:
    """Random EMOS + edge exit parameters."""
    return {
        # EMOS
        "emos_training_window": rng.choice([7, 14, 21, 30, 45, 60, 90]),
        "emos_sigma_method": rng.choice(
            ["rolling_rmse", "rolling_mae", "ewma_rmse"]
        ),
        "emos_bias_method": rng.choice(["rolling_mean", "ewma", "none"]),
        "emos_sigma_floor": rng.choice([0.3, 0.5, 0.7, 1.0, 1.2, 1.5]),
        "emos_sigma_scale": rng.choice([0.6, 0.8, 1.0, 1.2, 1.5, 2.0]),
        "emos_ewma_alpha": rng.choice([0.05, 0.1, 0.15, 0.2, 0.3]),
        # Edge exit
        "exit_enabled": True,
        "min_hold_edge": rng.choice([-0.05, 0.0, 0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20]),
        "exit_slippage_pct": rng.choice([0.005, 0.01, 0.015, 0.02, 0.03, 0.04]),
        "prob_exit_floor": rng.choice([0.0, 0.0, 0.05, 0.10, 0.15]),
        "profit_take_also": rng.choice([False, False, False, True]),
        "profit_target_abs": rng.choice([0.10, 0.15, 0.20, 0.25, 0.30]),
        # Quality gate
        "min_edge": rng.choice([0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.08]),
        "max_edge": 0.90,
        "max_price": rng.choice([0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]),
        "min_price": rng.choice([0.01, 0.02, 0.03, 0.05, 0.08]),
        "min_prob": rng.choice([0.01, 0.02, 0.03, 0.05, 0.08, 0.10]),
        "min_volume": rng.choice([0, 10, 30, 50]),
        "kelly_raw_cap": rng.choice([0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30]),
        "prob_sharpening": rng.choice([0.80, 0.85, 0.90, 0.92, 0.95, 1.0, 1.05, 1.10]),
        "shrinkage": rng.choice([0.0, 0.01, 0.02, 0.03, 0.05]),
        "max_aggregate_pct": rng.choice([0.40, 0.50, 0.60, 0.70, 0.80]),
        "city_max_exposure": rng.choice([0.15, 0.20, 0.25, 0.30]),
        "no_compounding": True,
        "max_position_pct": 0.05,
        # City
        "excluded_cities": set(),
        "city_overrides": {},
    }


def perturb_params(base: dict, rng: random.Random) -> dict:
    """Perturb 1-4 params from base."""
    p = dict(base)
    p["excluded_cities"] = set(p.get("excluded_cities", set()))
    p["city_overrides"] = dict(p.get("city_overrides", {}))

    mutations = {
        # EMOS
        "emos_training_window": [7, 14, 21, 30, 45, 60, 90],
        "emos_sigma_floor": [0.3, 0.5, 0.7, 1.0, 1.2, 1.5],
        "emos_sigma_scale": [0.6, 0.8, 1.0, 1.2, 1.5, 2.0],
        "emos_ewma_alpha": [0.05, 0.1, 0.15, 0.2, 0.3],
        # Exit
        "min_hold_edge": [-0.05, 0.0, 0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20],
        "exit_slippage_pct": [0.005, 0.01, 0.015, 0.02, 0.03, 0.04],
        "prob_exit_floor": [0.0, 0.05, 0.10, 0.15],
        # Gate
        "min_edge": [0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.08],
        "max_price": [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60],
        "min_price": [0.01, 0.02, 0.03, 0.05, 0.08],
        "min_prob": [0.01, 0.02, 0.03, 0.05, 0.08, 0.10],
        "kelly_raw_cap": [0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30],
        "prob_sharpening": [0.80, 0.85, 0.90, 0.92, 0.95, 1.0, 1.05, 1.10],
        "shrinkage": [0.0, 0.01, 0.02, 0.03, 0.05],
        "max_aggregate_pct": [0.40, 0.50, 0.60, 0.70, 0.80],
    }

    n = rng.randint(1, 4)
    keys = rng.sample(list(mutations.keys()), min(n, len(mutations)))
    for key in keys:
        pool = mutations[key]
        current = p.get(key)
        candidates = [v for v in pool if v != current]
        if candidates:
            p[key] = rng.choice(candidates)

    # Occasionally flip EMOS method
    if rng.random() < 0.15:
        p["emos_sigma_method"] = rng.choice(["rolling_rmse", "rolling_mae", "ewma_rmse"])
    if rng.random() < 0.15:
        p["emos_bias_method"] = rng.choice(["rolling_mean", "ewma", "none"])
    if rng.random() < 0.10:
        p["profit_take_also"] = not p.get("profit_take_also", False)

    return p


# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT RUNNER
# ═══════════════════════════════════════════════════════════════════════════════


def run_experiment(sim_data, params, experiment_id, entry_hours=24):
    """Run EMOS + edge exit experiment."""
    global _emos_enabled, _emos_params, _exit_slippage_pct

    _emos_enabled = True
    _emos_params = params
    _exit_slippage_pct = params.get("exit_slippage_pct", 0.01)
    reset_state()

    try:
        _prefit_emos_all(params)
        result = ef.simulate_portfolio(
            sim_data, params,
            entry_hours=entry_hours,
            experiment_id=experiment_id,
        )
        _emos_enabled = False
        return result
    except Exception as e:
        _emos_enabled = False
        print(f"  ERROR {experiment_id}: {e}", flush=True)
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    global _emos_obs, _emos_enabled, _emos_params, _exit_slippage_pct

    t_start = time.time()
    log_f = open(LOG_PATH, "w", buffering=1)

    def log(msg):
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        log_f.write(line + "\n")

    log("EMOS + Edge Exit Fusion Autoresearch")
    log("Loading data...")

    events, buckets_by_event, prices = load_pmd_data()
    forecasts = load_forecasts(events, CITY_COORDS)
    sim_data = SimulationData(events, buckets_by_event, prices, forecasts)

    _emos_obs = _build_training_observations(events, buckets_by_event, forecasts)
    obs_total = sum(len(v) for v in _emos_obs.values())
    log(f"Data: {len(events)} events, EMOS obs: {obs_total} across {len(_emos_obs)} cities")

    install_patches()
    log("Patches installed: EMOS compute_prob + edge check_exit + custom exit_pnl")

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
            "exit_saves": result.exit_saves, "exit_regrets": result.exit_regrets,
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
            ex = f"exits={result.total_exits} saves={result.exit_saves}" if result.total_exits else ""
            log(f"  ★ NEW BEST: score={score:.1f}  trades={result.trades}  "
                f"WR={result.win_rate:.1f}%  PnL=${result.total_pnl:.0f}  "
                f"Sharpe={result.sharpe:.2f}  DD={result.max_drawdown_pct:.1f}%  {ex}")
            p = params
            log(f"    EMOS: win={p.get('emos_training_window')}, "
                f"σ_method={p.get('emos_sigma_method')}, "
                f"σ_floor={p.get('emos_sigma_floor')}, "
                f"σ_scale={p.get('emos_sigma_scale')}")
            log(f"    EXIT: mhe={p.get('min_hold_edge')}, "
                f"slip={p.get('exit_slippage_pct')}, "
                f"edge≥{p.get('min_edge')}, "
                f"p=[{p.get('min_price')},{p.get('max_price')}], "
                f"sharp={p.get('prob_sharpening')}, "
                f"kelly={p.get('kelly_raw_cap')}")
            excl = p.get("excluded_cities", set())
            if excl:
                log(f"    EXCL: {sorted(excl)}")

    # ══════════════════════════════════════════════════════════════════════
    # BASELINE: Edge exit WITHOUT EMOS (from previous sweep winner)
    # ══════════════════════════════════════════════════════════════════════

    log("\n=== BASELINE: Edge exit (no EMOS) ===")
    _emos_enabled = False
    baseline_params = dict(_EDGE_EXIT_BASE)
    baseline_params["excluded_cities"] = set()
    baseline_params["city_overrides"] = {}
    _exit_slippage_pct = 0.01
    reset_state()
    baseline_result = ef.simulate_portfolio(
        sim_data, baseline_params, entry_hours=24, experiment_id="baseline_edge",
    )
    _emos_enabled = True
    if baseline_result:
        record(baseline_params, baseline_result, "baseline")
        log(f"Baseline (edge, no EMOS): score={baseline_result.score:.1f}  "
            f"PnL=${baseline_result.total_pnl:.0f}")

    # ══════════════════════════════════════════════════════════════════════
    # GEN 0: RANDOM SEARCH — EMOS + edge exit
    # ══════════════════════════════════════════════════════════════════════

    log(f"\n=== GEN 0: RANDOM SEARCH ({GEN0_TRIALS} trials) ===")
    rng = random.Random(123)

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
        base = top_k[i % len(top_k)]
        params = perturb_params(base, rng)
        result = run_experiment(sim_data, params, f"g1_{i:04d}")
        if result:
            record(params, result, "gen1")
        if (i + 1) % 100 == 0:
            log(f"  Gen 1: {i+1}/{GEN1_TRIALS}  best={best_score:.1f}")

    log(f"Gen 1 done: best={best_score:.1f}")

    # ══════════════════════════════════════════════════════════════════════
    # GEN 2: SYSTEMATIC CITY EXCLUSION
    # ══════════════════════════════════════════════════════════════════════

    log(f"\n=== GEN 2: CITY EXCLUSION ({GEN2_TRIALS} trials) ===")

    # Get per-city results from best model
    city_pnl = {}
    if best_result and best_result.city_results:
        for city, cr in best_result.city_results.items():
            city_pnl[city] = cr.get("pnl", 0)

    losing_cities = sorted([c for c, p in city_pnl.items() if p < 0],
                           key=lambda c: city_pnl.get(c, 0))
    marginal_cities = sorted([c for c, p in city_pnl.items() if 0 <= p < 30],
                             key=lambda c: city_pnl.get(c, 0))

    log(f"  Losing: {losing_cities}")
    log(f"  Marginal: {marginal_cities}")

    trial_i = 0
    top_k = [p for _, p, _ in top_results[:TOP_K]]

    # Test single exclusions
    for city in losing_cities + marginal_cities:
        if trial_i >= GEN2_TRIALS:
            break
        for base in top_k[:3]:
            if trial_i >= GEN2_TRIALS:
                break
            params = dict(base)
            params["excluded_cities"] = set(base.get("excluded_cities", set())) | {city}
            params["city_overrides"] = dict(base.get("city_overrides", {}))
            result = run_experiment(sim_data, params, f"g2_excl1_{trial_i:04d}")
            if result:
                record(params, result, "gen2_excl1")
            trial_i += 1

    # Test pair exclusions
    candidates = losing_cities + marginal_cities[:3]
    for c1, c2 in combinations(candidates, 2):
        if trial_i >= GEN2_TRIALS:
            break
        for base in top_k[:2]:
            if trial_i >= GEN2_TRIALS:
                break
            params = dict(base)
            params["excluded_cities"] = set(base.get("excluded_cities", set())) | {c1, c2}
            params["city_overrides"] = dict(base.get("city_overrides", {}))
            result = run_experiment(sim_data, params, f"g2_excl2_{trial_i:04d}")
            if result:
                record(params, result, "gen2_excl2")
            trial_i += 1

    # Test triple exclusions of worst
    if len(losing_cities) >= 3:
        for combo in combinations(losing_cities[:5], 3):
            if trial_i >= GEN2_TRIALS:
                break
            base = top_k[0]
            params = dict(base)
            params["excluded_cities"] = set(base.get("excluded_cities", set())) | set(combo)
            params["city_overrides"] = dict(base.get("city_overrides", {}))
            result = run_experiment(sim_data, params, f"g2_excl3_{trial_i:04d}")
            if result:
                record(params, result, "gen2_excl3")
            trial_i += 1

    # Fill remaining with random exclusion combos
    while trial_i < GEN2_TRIALS:
        base = rng.choice(top_k)
        n_excl = rng.randint(0, 3)
        excl_pool = losing_cities + marginal_cities
        excl = set(rng.sample(excl_pool, min(n_excl, len(excl_pool)))) if excl_pool else set()
        params = dict(base)
        params["excluded_cities"] = set(base.get("excluded_cities", set())) | excl
        params["city_overrides"] = dict(base.get("city_overrides", {}))
        result = run_experiment(sim_data, params, f"g2_rand_{trial_i:04d}")
        if result:
            record(params, result, "gen2_rand")
        trial_i += 1

    log(f"Gen 2 done: best={best_score:.1f}")

    # ══════════════════════════════════════════════════════════════════════
    # GEN 3: PER-CITY OVERRIDE TUNING
    # ══════════════════════════════════════════════════════════════════════

    log(f"\n=== GEN 3: PER-CITY OVERRIDES ({GEN3_TRIALS} trials) ===")

    # Refresh city rankings from best
    city_pnl = {}
    city_trades = {}
    if best_result and best_result.city_results:
        for city, cr in best_result.city_results.items():
            city_pnl[city] = cr.get("pnl", 0)
            city_trades[city] = cr.get("trades", 0)

    # Focus on cities with enough trades but mediocre PnL (room for improvement)
    tunable_cities = [c for c, t in city_trades.items()
                      if t >= 20 and c not in best_params.get("excluded_cities", set())]

    top_k = [p for _, p, _ in top_results[:TOP_K]]

    for i in range(GEN3_TRIALS):
        base = top_k[i % len(top_k)]
        params = dict(base)
        params["excluded_cities"] = set(base.get("excluded_cities", set()))
        params["city_overrides"] = dict(base.get("city_overrides", {}))

        if tunable_cities:
            # Pick 1-3 cities to override
            n_cities = rng.randint(1, min(3, len(tunable_cities)))
            cities = rng.sample(tunable_cities, n_cities)

            for city in cities:
                params["city_overrides"][city] = {
                    "min_edge": rng.choice([0.005, 0.01, 0.02, 0.03, 0.05, 0.08]),
                    "max_price": rng.choice([0.25, 0.30, 0.35, 0.40, 0.50, 0.60]),
                    "min_price": rng.choice([0.01, 0.02, 0.03, 0.05]),
                    "kelly_raw_cap": rng.choice([0.08, 0.10, 0.12, 0.15, 0.20, 0.25]),
                    "prob_sharpening": rng.choice([0.80, 0.85, 0.90, 0.95, 1.0, 1.05]),
                    "shrinkage": rng.choice([0.0, 0.01, 0.03, 0.05, 0.08]),
                }

        # Also perturb some global params
        if rng.random() < 0.3:
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
        _exit_slippage_pct = best_params.get("exit_slippage_pct", 0.01)
        _emos_enabled = True
        _emos_params.update(best_params)
        reset_state()
        _prefit_emos_all(best_params)

        wf = walk_forward_validate(sim_data, best_params, entry_hours=24)
        log(f"WF: OOS score={wf['score']:.1f}  OOS PnL=${wf['oos_pnl']:.0f}  "
            f"folds={wf['n_folds']}  trades={wf['total_trades']}")
        for fold in wf.get("folds", []):
            log(f"  Fold: PnL=${fold['pnl']:.0f}  trades={fold['trades']}  "
                f"WR={fold['wr']:.1f}%  score={fold['score']:.1f}")
    else:
        wf = {"score": 0, "oos_pnl": 0, "n_folds": 0}

    # ══════════════════════════════════════════════════════════════════════
    # RESULTS
    # ══════════════════════════════════════════════════════════════════════

    elapsed = time.time() - t_start

    # Per-city from best
    if best_result and best_result.city_results:
        log("\n=== BEST MODEL — PER-CITY ===")
        log(f"  {'City':<16} {'Trades':>6} {'WR':>6} {'PnL':>10}")
        log(f"  {'-'*42}")
        for city in sorted(best_result.city_results.keys(),
                           key=lambda c: -best_result.city_results[c].get("pnl", 0)):
            cr = best_result.city_results[city]
            log(f"  {city:<16} {cr.get('trades', 0):>6} "
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
        log(f"  Capital util:     {best_result.capital_utilization:.1f}%")
        log(f"  Exits:            {best_result.total_exits}  "
            f"(saves={best_result.exit_saves}, regrets={best_result.exit_regrets})")
        log(f"  OOS PnL:          ${wf.get('oos_pnl', 0):.0f}")
        log(f"  OOS score:        {wf.get('score', 0):.1f}")
        excl = best_params.get("excluded_cities", set())
        log(f"  Excluded cities:  {sorted(excl) if excl else 'none'}")
        ovr = best_params.get("city_overrides", {})
        if ovr:
            log(f"  City overrides:   {list(ovr.keys())}")
    log(f"  Trials:           {trial_n}")
    log(f"  Runtime:          {elapsed:.0f}s ({elapsed/60:.1f}min)")

    # Comparison
    if baseline_result:
        log(f"\n  vs baseline (edge, no EMOS):")
        log(f"    Score: {best_score:.1f} vs {baseline_result.score:.1f} "
            f"(+{best_score - baseline_result.score:.1f})")
        log(f"    PnL:   ${best_result.total_pnl:.0f} vs ${baseline_result.total_pnl:.0f}")

    # ── Save JSON ──
    output = {
        "sweep_id": "emos_exit_fusion",
        "sweep_type": "EMOS + Edge Exit Fusion Autoresearch",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_trials": trial_n,
        "runtime_seconds": round(elapsed, 1),
        "baseline": {
            "score": baseline_result.score if baseline_result else 0,
            "trades": baseline_result.trades if baseline_result else 0,
            "total_pnl": baseline_result.total_pnl if baseline_result else 0,
            "params": _ser(baseline_params),
        },
        "best": {
            "score": best_score,
            "params": _ser(best_params),
            **(serialize_result(best_result) if best_result else {}),
        },
        "walk_forward": wf,
        "top_results": [
            {
                "rank": i + 1, "score": s, "params": _ser(p),
                **(serialize_result(r) if r else {}),
            }
            for i, (s, p, r) in enumerate(top_results[:20])
        ],
        "all_results": all_results,
    }

    with open(SWEEP_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)
    log(f"\nResults saved to {SWEEP_PATH}")
    log_f.close()


def _ser(params):
    """JSON-serializable params."""
    out = {}
    for k, v in params.items():
        if isinstance(v, set):
            out[k] = sorted(v)
        elif isinstance(v, dict):
            out[k] = {str(kk): vv for kk, vv in v.items()}
        else:
            out[k] = v
    return out


if __name__ == "__main__":
    main()
