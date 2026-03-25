"""
Early Exit Autoresearch — Profit-Taking Strategy Optimization
==============================================================

Three-phase autoresearch (Karpathy pattern) to find optimal early exit
parameters for Strategy C weather markets.

Instead of holding to binary resolution ($0/$1), this explores:
  - Profit targets (sell when price rises +X above entry)
  - Trailing stops (sell when price pulls back Y% from peak)
  - Combo strategies (target + trail)
  - Edge-based exit (existing approach, for comparison)

Uses PMD 10-minute price data through experiment_framework's hourly
simulation. Monkey-patches check_exit() to add profit target + trailing
stop logic.

Phases:
  Gen 0: Random search (800 trials) — explore full parameter space
  Gen 1: Fine-tune (400 trials) — perturb top-10 performers
  Gen 2: City optimization (200 trials) — exclude/tune per-city
  Final: Walk-forward validation on best result

Usage:
    python3 research/innovations/sweep_early_exit.py

Output: research/data/experiments/sweep_early_exit.json
"""

from __future__ import annotations

import json
import math
import random
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from pmd_loader import load_pmd_data

import experiment_framework as ef
from experiment_framework import (
    CITY_COORDS,
    SimulationData,
    ExperimentResult,
    experiment_score,
    load_forecasts,
    simulate_portfolio,
    serialize_result,
    walk_forward_validate,
    GAS_COST_USD,
    INITIAL_CAPITAL,
)

# ═══════════════════════════════════════════════════════════════════════════════
# OUTPUT PATHS
# ═══════════════════════════════════════════════════════════════════════════════

RESULTS_DIR = Path(__file__).parent.parent / "data" / "experiments"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
SWEEP_PATH = RESULTS_DIR / "sweep_early_exit.json"
LOG_PATH = Path(__file__).parent.parent / "sweep_early_exit_log.txt"

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

GEN0_TRIALS = 800    # Random search
GEN1_TRIALS = 400    # Fine-tune top-10
GEN2_TRIALS = 200    # City optimization
TOP_K = 10           # Keep top-K for fine-tuning

# ═══════════════════════════════════════════════════════════════════════════════
# EARLY EXIT MONKEY-PATCH
# ═══════════════════════════════════════════════════════════════════════════════

# State for trailing stops: token_id -> peak price since entry
_peak_prices: dict[str, float] = {}

# Custom exit slippage (separate from entry slippage, configurable)
_exit_slippage_pct: float = 0.005

# Store original functions
_original_check_exit = ef.check_exit
_original_calc_exit_pnl = ef.calc_exit_pnl


def _check_exit_early(pos, current_price, hour_ts, params):
    """Enhanced exit logic with profit targets and trailing stops.

    Exit types:
      "none"   — hold to resolution (baseline)
      "target" — exit when price >= entry + profit_target_abs
      "trail"  — exit when price pulls back trail_pct from peak
      "combo"  — exit on first trigger (target OR trail)
      "edge"   — edge-based exit (original logic)
    """
    if not params.get("exit_enabled", False):
        return None

    exit_type = params.get("exit_type", "none")
    if exit_type == "none":
        return None

    # Edge-based exit (original logic)
    if exit_type == "edge":
        return _original_check_exit(pos, current_price, hour_ts, params)

    # Track peak price for trailing stop
    token_id = pos.token_id
    if token_id not in _peak_prices:
        _peak_prices[token_id] = pos.entry_price
    _peak_prices[token_id] = max(_peak_prices[token_id], current_price)

    # ── Profit target ──
    if exit_type in ("target", "combo"):
        target = params.get("profit_target_abs", 0.10)
        if current_price >= pos.entry_price + target:
            return "profit_take"

    # ── Trailing stop ──
    if exit_type in ("trail", "combo"):
        peak = _peak_prices[token_id]
        trail_pct = params.get("trailing_pct", 0.30)
        activation = params.get("trail_activation", 0.02)

        gain_from_entry = peak - pos.entry_price
        if gain_from_entry >= activation:
            pullback = peak - current_price
            if gain_from_entry > 0 and pullback / gain_from_entry >= trail_pct:
                return "trailing_stop"

    # ── Optional edge check (for combo+edge) ──
    if params.get("edge_exit_also", False):
        return _original_check_exit(pos, current_price, hour_ts, params)

    return None


def _calc_exit_pnl_custom(pos, exit_price):
    """Exit P&L with configurable slippage."""
    exit_fill = exit_price * (1 - _exit_slippage_pct)
    tokens = pos.size / pos.entry_fill
    proceeds = tokens * exit_fill
    return proceeds - pos.size - GAS_COST_USD


def install_monkey_patches():
    """Install early exit logic into experiment framework."""
    ef.check_exit = _check_exit_early
    ef.calc_exit_pnl = _calc_exit_pnl_custom


def reset_state():
    """Reset peak prices between trials."""
    _peak_prices.clear()


# ═══════════════════════════════════════════════════════════════════════════════
# PARAMETER GENERATION
# ═══════════════════════════════════════════════════════════════════════════════


def random_params(rng: random.Random) -> dict:
    """Generate random parameter set (Gen 0)."""
    exit_type = rng.choice(["target", "trail", "combo", "none", "edge"])

    # Exit parameters
    profit_target_abs = rng.choice([0.02, 0.03, 0.04, 0.05, 0.06, 0.08,
                                     0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30])
    trailing_pct = rng.choice([0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70])
    trail_activation = rng.choice([0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.08])
    exit_slippage = rng.choice([0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05])

    # Entry parameters (same as existing sweeps)
    min_edge = rng.choice([0.01, 0.02, 0.03, 0.04, 0.05, 0.08, 0.10, 0.12, 0.15])
    max_price = rng.choice([0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80])
    min_price = rng.choice([0.01, 0.02, 0.03, 0.04, 0.05, 0.08, 0.10])
    min_prob = rng.choice([0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25])
    min_volume = rng.choice([0, 10, 30, 50, 100])
    prob_sharpening = rng.choice([0.80, 0.85, 0.88, 0.90, 0.92, 0.95,
                                   1.0, 1.05, 1.10, 1.15, 1.20])
    kelly_raw_cap = rng.choice([0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30, 0.35, 0.40])
    shrinkage = rng.choice([0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10])
    shrinkage_prior = rng.choice([0.10, 0.125, 0.15])
    base_sigma = rng.choice([2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])

    # Capital management
    max_aggregate_pct = rng.choice([0.30, 0.40, 0.50, 0.60, 0.70, 0.80])
    city_max_exposure = rng.choice([0.15, 0.20, 0.25, 0.30, 0.35])

    # Edge-based exit params (for "edge" type)
    min_hold_edge = rng.choice([-0.05, 0.0, 0.02, 0.05, 0.08, 0.10, 0.15])

    return {
        # Exit params
        "exit_enabled": exit_type != "none",
        "exit_type": exit_type,
        "profit_target_abs": profit_target_abs,
        "trailing_pct": trailing_pct,
        "trail_activation": trail_activation,
        "exit_slippage_pct": exit_slippage,
        # Edge-based exit (for "edge" type)
        "min_hold_edge": min_hold_edge,
        "edge_exit_also": False,
        # Entry params
        "min_edge": min_edge,
        "max_edge": 0.90,
        "max_price": max_price,
        "min_price": min_price,
        "min_prob": min_prob,
        "min_volume": min_volume,
        "prob_sharpening": prob_sharpening,
        "kelly_fraction": 0.25,
        "kelly_raw_cap": kelly_raw_cap,
        "max_position_pct": 0.05,
        "shrinkage": shrinkage,
        "shrinkage_prior": shrinkage_prior,
        "shrinkage_weight": shrinkage,
        "base_sigma": base_sigma,
        # Capital management
        "max_aggregate_pct": max_aggregate_pct,
        "city_max_exposure": city_max_exposure,
        "no_compounding": True,
        # City overrides
        "excluded_cities": set(),
        "city_overrides": {},
    }


def perturb_params(base: dict, rng: random.Random, n_mutations: int = 0) -> dict:
    """Perturb a parameter set (Gen 1 fine-tuning).

    Randomly modify 1-4 parameters from the base set.
    """
    params = {k: v for k, v in base.items()}  # shallow copy
    params["city_overrides"] = dict(base.get("city_overrides", {}))
    params["excluded_cities"] = set(base.get("excluded_cities", set()))

    if n_mutations <= 0:
        n_mutations = rng.randint(1, 4)

    # Mutable parameters with their option pools
    mutations = {
        "exit_type": ["target", "trail", "combo", "none", "edge"],
        "profit_target_abs": [0.02, 0.03, 0.04, 0.05, 0.06, 0.08,
                               0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30],
        "trailing_pct": [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70],
        "trail_activation": [0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.08],
        "exit_slippage_pct": [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05],
        "min_edge": [0.01, 0.02, 0.03, 0.04, 0.05, 0.08, 0.10, 0.12, 0.15],
        "max_price": [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80],
        "min_price": [0.01, 0.02, 0.03, 0.04, 0.05, 0.08, 0.10],
        "min_prob": [0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25],
        "prob_sharpening": [0.80, 0.85, 0.88, 0.90, 0.92, 0.95,
                             1.0, 1.05, 1.10, 1.15, 1.20],
        "kelly_raw_cap": [0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30, 0.35, 0.40],
        "shrinkage": [0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10],
        "base_sigma": [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
        "max_aggregate_pct": [0.30, 0.40, 0.50, 0.60, 0.70, 0.80],
        "city_max_exposure": [0.15, 0.20, 0.25, 0.30, 0.35],
        "min_hold_edge": [-0.05, 0.0, 0.02, 0.05, 0.08, 0.10, 0.15],
    }

    keys = rng.sample(list(mutations.keys()), min(n_mutations, len(mutations)))

    for key in keys:
        pool = mutations[key]
        current = params.get(key)
        # Pick a different value
        candidates = [v for v in pool if v != current]
        if candidates:
            params[key] = rng.choice(candidates)

    # Keep exit_enabled consistent
    params["exit_enabled"] = params.get("exit_type", "none") != "none"
    params["shrinkage_weight"] = params.get("shrinkage", 0.0)

    return params


def city_optimize_params(base: dict, rng: random.Random,
                          city_results: dict[str, dict]) -> dict:
    """Generate city-optimized variant (Gen 2).

    Options:
      - Exclude worst-performing cities
      - Add per-city overrides for mid-performers
    """
    params = {k: v for k, v in base.items()}
    params["city_overrides"] = dict(base.get("city_overrides", {}))
    params["excluded_cities"] = set(base.get("excluded_cities", set()))

    # Rank cities by P&L
    ranked = sorted(
        city_results.items(),
        key=lambda x: x[1].get("pnl", 0),
    )
    losing_cities = [c for c, r in ranked if r.get("pnl", 0) < 0]
    mid_cities = [c for c, r in ranked
                  if 0 <= r.get("pnl", 0) < 50 and r.get("trades", 0) >= 5]

    action = rng.choice(["exclude", "override", "both", "reset"])

    if action == "exclude" and losing_cities:
        n = rng.randint(1, min(3, len(losing_cities)))
        params["excluded_cities"] = set(rng.sample(losing_cities, n))

    elif action == "override" and mid_cities:
        city = rng.choice(mid_cities)
        params["city_overrides"][city] = {
            "min_edge": rng.choice([0.02, 0.03, 0.05, 0.08]),
            "max_price": rng.choice([0.30, 0.40, 0.50, 0.60]),
            "min_price": rng.choice([0.02, 0.03, 0.05]),
            "kelly_raw_cap": rng.choice([0.10, 0.15, 0.20]),
        }

    elif action == "both" and losing_cities:
        n = rng.randint(1, min(2, len(losing_cities)))
        params["excluded_cities"] = set(rng.sample(losing_cities, n))
        if mid_cities:
            city = rng.choice(mid_cities)
            params["city_overrides"][city] = {
                "min_edge": rng.choice([0.02, 0.05, 0.08]),
                "max_price": rng.choice([0.40, 0.50, 0.60]),
            }

    elif action == "reset":
        params["excluded_cities"] = set()
        params["city_overrides"] = {}

    return params


# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT RUNNER
# ═══════════════════════════════════════════════════════════════════════════════


def run_experiment(
    sim_data: SimulationData,
    params: dict,
    experiment_id: str,
    entry_hours: float = 24,
) -> ExperimentResult | None:
    """Run one experiment with given parameters.

    Sets up exit slippage, resets state, runs portfolio simulation.
    """
    global _exit_slippage_pct

    # Set exit slippage for this trial
    _exit_slippage_pct = params.get("exit_slippage_pct", 0.005)

    # Reset trailing stop state
    reset_state()

    try:
        result = simulate_portfolio(
            sim_data, params,
            entry_hours=entry_hours,
            experiment_id=experiment_id,
        )
        return result
    except Exception as e:
        print(f"  ERROR {experiment_id}: {e}", flush=True)
        return None


def format_params_short(params: dict) -> str:
    """Short param summary for logging."""
    et = params.get("exit_type", "none")
    parts = [f"exit={et}"]

    if et in ("target", "combo"):
        parts.append(f"tgt={params.get('profit_target_abs', '?')}")
    if et in ("trail", "combo"):
        parts.append(f"tr={params.get('trailing_pct', '?')}")
        parts.append(f"act={params.get('trail_activation', '?')}")
    if et == "edge":
        parts.append(f"mhe={params.get('min_hold_edge', '?')}")

    parts.append(f"slip={params.get('exit_slippage_pct', '?')}")
    parts.append(f"edge≥{params.get('min_edge', '?')}")
    parts.append(f"p=[{params.get('min_price', '?')},{params.get('max_price', '?')}]")
    parts.append(f"sharp={params.get('prob_sharpening', '?')}")
    parts.append(f"kelly={params.get('kelly_raw_cap', '?')}")

    return " ".join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN SWEEP
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    t_start = time.time()
    log_f = open(LOG_PATH, "w", buffering=1)

    def log(msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        log_f.write(line + "\n")

    log("Early Exit Autoresearch — Loading data...")

    # ── Load PMD data ──
    events, buckets_by_event, prices = load_pmd_data()
    forecasts = load_forecasts(events, CITY_COORDS)
    sim_data = SimulationData(events, buckets_by_event, prices, forecasts)

    log(f"Data: {len(events)} events, {sum(len(b) for b in buckets_by_event.values())} buckets")

    # ── Install monkey patches ──
    install_monkey_patches()
    log("Monkey-patched check_exit() and calc_exit_pnl()")

    # ── Track results ──
    all_results: list[dict] = []
    top_results: list[tuple[float, dict, ExperimentResult]] = []  # (score, params, result)
    best_score = 0.0
    best_params: dict = {}
    best_result: ExperimentResult | None = None
    trial_count = 0

    def record_result(params: dict, result: ExperimentResult, phase: str):
        nonlocal best_score, best_params, best_result, trial_count
        trial_count += 1

        score = result.score
        entry = {
            "trial": trial_count,
            "phase": phase,
            "score": score,
            "trades": result.trades,
            "win_rate": result.win_rate,
            "total_pnl": result.total_pnl,
            "roi_pct": result.roi_pct,
            "sharpe": result.sharpe,
            "max_drawdown_pct": result.max_drawdown_pct,
            "capital_utilization": result.capital_utilization,
            "avg_pnl_per_hour": result.avg_pnl_per_hour,
            "total_exits": result.total_exits,
            "exit_saves": result.exit_saves,
            "exit_regrets": result.exit_regrets,
            "exit_type": params.get("exit_type", "none"),
            "params": _serialize_params(params),
        }
        all_results.append(entry)

        # Update top-K
        top_results.append((score, params, result))
        top_results.sort(key=lambda x: -x[0])
        if len(top_results) > TOP_K * 2:
            top_results[:] = top_results[:TOP_K * 2]

        if score > best_score:
            best_score = score
            best_params = params
            best_result = result
            log(f"  ★ NEW BEST: score={score:.1f}  trades={result.trades}  "
                f"WR={result.win_rate:.1f}%  PnL=${result.total_pnl:.0f}  "
                f"Sharpe={result.sharpe:.2f}  DD={result.max_drawdown_pct:.1f}%  "
                f"exits={result.total_exits}  saves={result.exit_saves}  "
                f"regrets={result.exit_regrets}")
            log(f"    {format_params_short(params)}")

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 0: BASELINE (hold to resolution)
    # ══════════════════════════════════════════════════════════════════════

    log("\n=== PHASE 0: BASELINE (hold to resolution) ===")
    baseline_params = {
        "exit_enabled": False,
        "exit_type": "none",
        "min_edge": 0.03,
        "max_edge": 0.90,
        "max_price": 0.55,
        "min_price": 0.04,
        "min_prob": 0.04,
        "min_volume": 50,
        "prob_sharpening": 0.90,
        "kelly_fraction": 0.25,
        "kelly_raw_cap": 0.15,
        "max_position_pct": 0.05,
        "shrinkage": 0.03,
        "shrinkage_prior": 0.125,
        "shrinkage_weight": 0.03,
        "base_sigma": 3.0,
        "max_aggregate_pct": 0.40,
        "city_max_exposure": 0.25,
        "no_compounding": True,
        "excluded_cities": set(),
        "city_overrides": {},
        "exit_slippage_pct": 0.005,
    }
    baseline_result = run_experiment(sim_data, baseline_params, "baseline_hold", entry_hours=24)
    if baseline_result:
        record_result(baseline_params, baseline_result, "baseline")
        log(f"Baseline: score={baseline_result.score:.1f}  trades={baseline_result.trades}  "
            f"PnL=${baseline_result.total_pnl:.0f}  Sharpe={baseline_result.sharpe:.2f}")

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 1: RANDOM SEARCH (Gen 0)
    # ══════════════════════════════════════════════════════════════════════

    log(f"\n=== PHASE 1: RANDOM SEARCH ({GEN0_TRIALS} trials) ===")
    rng = random.Random(42)

    for i in range(GEN0_TRIALS):
        params = random_params(rng)
        eid = f"gen0_{i:04d}"

        result = run_experiment(sim_data, params, eid, entry_hours=24)
        if result is None:
            continue

        record_result(params, result, "gen0")

        if (i + 1) % 100 == 0:
            log(f"  Gen 0: {i + 1}/{GEN0_TRIALS}  best={best_score:.1f}  "
                f"trials_with_score={sum(1 for r in all_results if r['score'] > 0)}")

    log(f"Gen 0 complete: best={best_score:.1f}")

    # Also test entry_hours=48 for top-5 Gen 0
    log("  Testing entry_hours=48 for top-5...")
    for rank, (score, params, _) in enumerate(top_results[:5]):
        result = run_experiment(sim_data, params, f"gen0_48h_{rank}", entry_hours=48)
        if result:
            record_result(params, result, "gen0_48h")

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 2: FINE-TUNING (Gen 1)
    # ══════════════════════════════════════════════════════════════════════

    log(f"\n=== PHASE 2: FINE-TUNING ({GEN1_TRIALS} trials, top-{TOP_K}) ===")

    top_k_params = [p for _, p, _ in top_results[:TOP_K]]

    for i in range(GEN1_TRIALS):
        base = top_k_params[i % len(top_k_params)]
        n_mut = rng.choice([1, 1, 2, 2, 3, 4])
        params = perturb_params(base, rng, n_mutations=n_mut)
        eid = f"gen1_{i:04d}"

        result = run_experiment(sim_data, params, eid, entry_hours=24)
        if result is None:
            continue

        record_result(params, result, "gen1")

        if (i + 1) % 100 == 0:
            log(f"  Gen 1: {i + 1}/{GEN1_TRIALS}  best={best_score:.1f}")

    log(f"Gen 1 complete: best={best_score:.1f}")

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 3: CITY OPTIMIZATION (Gen 2)
    # ══════════════════════════════════════════════════════════════════════

    log(f"\n=== PHASE 3: CITY OPTIMIZATION ({GEN2_TRIALS} trials) ===")

    # Get city results from best experiment
    best_city_results = {}
    if best_result and best_result.city_results:
        best_city_results = best_result.city_results

    top_k_params = [p for _, p, _ in top_results[:TOP_K]]

    for i in range(GEN2_TRIALS):
        base = top_k_params[i % len(top_k_params)]
        params = city_optimize_params(base, rng, best_city_results)
        eid = f"gen2_{i:04d}"

        result = run_experiment(sim_data, params, eid, entry_hours=24)
        if result is None:
            continue

        record_result(params, result, "gen2")

        # Update city_results from new best for next iterations
        if result.score > best_score * 0.95 and result.city_results:
            best_city_results = result.city_results

        if (i + 1) % 50 == 0:
            log(f"  Gen 2: {i + 1}/{GEN2_TRIALS}  best={best_score:.1f}")

    log(f"Gen 2 complete: best={best_score:.1f}")

    # ══════════════════════════════════════════════════════════════════════
    # WALK-FORWARD VALIDATION
    # ══════════════════════════════════════════════════════════════════════

    log("\n=== WALK-FORWARD VALIDATION ===")

    if best_result is not None:
        # Set exit slippage for WF run
        _exit_slippage_pct = best_params.get("exit_slippage_pct", 0.005)
        reset_state()

        wf = walk_forward_validate(sim_data, best_params, entry_hours=24)
        log(f"WF: OOS score={wf['score']:.1f}  OOS PnL=${wf['oos_pnl']:.0f}  "
            f"folds={wf['n_folds']}  trades={wf['total_trades']}")

        for fold in wf.get("folds", []):
            log(f"  Fold: PnL=${fold['pnl']:.0f}  trades={fold['trades']}  "
                f"WR={fold['wr']:.1f}%  score={fold['score']:.1f}")
    else:
        wf = {"score": 0, "oos_pnl": 0, "n_folds": 0}

    # ══════════════════════════════════════════════════════════════════════
    # RESULTS ANALYSIS & OUTPUT
    # ══════════════════════════════════════════════════════════════════════

    elapsed = time.time() - t_start

    # Best result per exit type
    log("\n=== BEST PER EXIT TYPE ===")
    by_type: dict[str, list[dict]] = defaultdict(list)
    for r in all_results:
        by_type[r["exit_type"]].append(r)

    for etype in ["none", "target", "trail", "combo", "edge"]:
        type_results = by_type.get(etype, [])
        if not type_results:
            continue
        best_of_type = max(type_results, key=lambda r: r["score"])
        log(f"  {etype:<8}  score={best_of_type['score']:.1f}  "
            f"trades={best_of_type['trades']}  WR={best_of_type['win_rate']:.1f}%  "
            f"PnL=${best_of_type['total_pnl']:.0f}  Sharpe={best_of_type['sharpe']:.2f}  "
            f"exits={best_of_type.get('total_exits', 0)}")

    # Per-city breakdown for best
    if best_result and best_result.city_results:
        log("\n=== BEST MODEL — PER-CITY RESULTS ===")
        log(f"  {'City':<16} {'Trades':>6} {'WR':>6} {'PnL':>10}")
        log(f"  {'-'*40}")
        for city in sorted(best_result.city_results.keys(),
                           key=lambda c: -best_result.city_results[c].get("pnl", 0)):
            cr = best_result.city_results[city]
            log(f"  {city:<16} {cr.get('trades', 0):>6} "
                f"{cr.get('win_rate', 0):>5.1f}% ${cr.get('pnl', 0):>9.0f}")

    # Summary
    log("\n" + "=" * 80)
    log("FINAL SUMMARY")
    log("=" * 80)
    if best_result:
        log(f"  Best score:       {best_score:.1f}")
        log(f"  Exit type:        {best_params.get('exit_type', 'none')}")
        log(f"  Trades:           {best_result.trades}")
        log(f"  Win rate:         {best_result.win_rate:.1f}%")
        log(f"  Total PnL:        ${best_result.total_pnl:.0f}")
        log(f"  ROI:              {best_result.roi_pct:.1f}%")
        log(f"  Sharpe:           {best_result.sharpe:.2f}")
        log(f"  Max drawdown:     {best_result.max_drawdown_pct:.1f}%")
        log(f"  Capital util:     {best_result.capital_utilization:.1f}%")
        log(f"  PnL/hour:         ${best_result.avg_pnl_per_hour:.2f}")
        log(f"  Exits:            {best_result.total_exits}")
        log(f"  Saves:            {best_result.exit_saves}")
        log(f"  Regrets:          {best_result.exit_regrets}")
        log(f"  OOS PnL:          ${wf.get('oos_pnl', 0):.0f}")
        log(f"  OOS score:        {wf.get('score', 0):.1f}")
    log(f"  Total trials:     {trial_count}")
    log(f"  Runtime:          {elapsed:.0f}s ({elapsed/60:.1f}min)")
    log(f"  Params: {format_params_short(best_params)}")

    # ── Save JSON ──
    output = {
        "sweep_id": "early_exit",
        "sweep_type": "Early Exit Autoresearch",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_trials": trial_count,
        "runtime_seconds": round(elapsed, 1),
        "baseline": {
            "score": baseline_result.score if baseline_result else 0,
            "trades": baseline_result.trades if baseline_result else 0,
            "total_pnl": baseline_result.total_pnl if baseline_result else 0,
            "sharpe": baseline_result.sharpe if baseline_result else 0,
            "params": _serialize_params(baseline_params),
        },
        "best": {
            "score": best_score,
            "params": _serialize_params(best_params),
            **(serialize_result(best_result) if best_result else {}),
        },
        "walk_forward": wf,
        "top_results": [
            {
                "rank": i + 1,
                "score": s,
                "params": _serialize_params(p),
                **(serialize_result(r) if r else {}),
            }
            for i, (s, p, r) in enumerate(top_results[:20])
        ],
        "best_per_type": {
            etype: {
                "score": max(rs, key=lambda r: r["score"])["score"],
                "best": max(rs, key=lambda r: r["score"]),
            }
            for etype, rs in by_type.items()
            if rs
        },
        "all_results": all_results,
    }

    with open(SWEEP_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)
    log(f"\nResults saved to {SWEEP_PATH}")

    log_f.close()


def _serialize_params(params: dict) -> dict:
    """Make params JSON-serializable."""
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
