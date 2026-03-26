"""EMOS Per-City Autoresearch — Hyperpersonalized Parameters.

Takes the global EMOS baseline (score=156.5) and finds optimal per-city
EMOS + quality gate parameters for each city independently.

Approach:
  1. For each city, run 200 trials sweeping EMOS + gate params
  2. Best per-city params become city_overrides
  3. Final combined run: global params + all per-city overrides
  4. Compare: global-only vs per-city optimized

Output: sweep_emos_percity.json for /experiments dashboard.

Usage:
    python3 research/innovations/sweep_emos_percity.py
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

from emos_model import EMOSModel, Observation, actual_temp_from_bucket
from pmd_loader import load_pmd_data

import experiment_framework as ef
from experiment_framework import (
    CITY_COORDS,
    SimulationData,
    load_forecasts,
    serialize_result,
    simulate_portfolio,
    walk_forward_validate,
)

# Import EMOS machinery from sweep_emos_v2
from sweep_emos_v2 import (
    _build_training_observations,
    _compute_prob_emos,
    _emos_models,
    _emos_obs,
    _emos_params,
    _prefit_emos_all,
    run_experiment,
)
import sweep_emos_v2 as emos_v2

# ── Paths ────────────────────────────────────────────────────────────

RESULTS_DIR = Path(__file__).parent.parent / "data" / "experiments"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
SWEEP_PATH = RESULTS_DIR / "sweep_emos_percity.json"
LOG_PATH = Path(__file__).parent.parent / "sweep_emos_percity_log.txt"

# ── Global EMOS baseline params (from sweep_emos_v2 best) ───────────

GLOBAL_BASELINE = {
    "emos_training_window": 45,
    "emos_sigma_method": "rolling_mae",
    "emos_bias_method": "rolling_mean",
    "emos_sigma_floor": 0.5,
    "emos_sigma_scale": 1.5,
    "emos_ewma_alpha": 0.3,
    "min_edge": 0.15,
    "max_edge": 0.90,
    "max_price": 0.66,
    "min_price": 0.08,
    "min_prob": 0.07,
    "min_volume": 0,
    "kelly_raw_cap": 0.35,
    "prob_sharpening": 1.0,
    "shrinkage": 0.10,
    "excluded_cities": set(),
    "city_overrides": {},
    "exit_enabled": False,
}

# ── Per-city parameter space ─────────────────────────────────────────

# Parameters that can be overridden per city
CITY_OVERRIDE_KEYS = [
    "min_edge", "max_price", "min_price", "min_prob",
    "kelly_raw_cap", "prob_sharpening", "shrinkage",
]

# EMOS params that affect all cities (swept globally, not per-city)
EMOS_GLOBAL_KEYS = [
    "emos_training_window", "emos_sigma_floor", "emos_sigma_scale",
]


def random_city_override(rng: random.Random) -> dict:
    """Generate random per-city quality gate overrides."""
    return {
        "min_edge": rng.choice([0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20]),
        "max_price": rng.choice([0.30, 0.40, 0.50, 0.55, 0.60, 0.70, 0.80, 0.90]),
        "min_price": rng.choice([0.01, 0.02, 0.03, 0.05, 0.08, 0.10]),
        "min_prob": rng.choice([0.02, 0.05, 0.08, 0.10, 0.15, 0.25]),
        "kelly_raw_cap": rng.choice([0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]),
        "prob_sharpening": rng.choice([0.85, 0.90, 0.95, 1.0, 1.05, 1.10, 1.15, 1.20]),
        "shrinkage": rng.choice([0.0, 0.01, 0.03, 0.05, 0.10, 0.15]),
    }


def perturb_city_override(base: dict, rng: random.Random) -> dict:
    """Perturb 1-3 city override parameters."""
    p = dict(base)
    keys = list(CITY_OVERRIDE_KEYS)
    for key in rng.sample(keys, min(rng.randint(1, 3), len(keys))):
        val = p.get(key, 0)
        if key == "prob_sharpening":
            p[key] = round(val + rng.choice([-0.10, -0.05, 0.05, 0.10]), 2)
        elif key == "shrinkage":
            p[key] = round(max(0, val + rng.choice([-0.03, -0.02, 0.02, 0.03])), 2)
        elif key == "kelly_raw_cap":
            p[key] = round(max(0.05, min(0.60, val + rng.choice([-0.05, 0.05]))), 2)
        else:
            p[key] = round(max(0.01, val + rng.choice([-0.05, -0.03, -0.02, 0.02, 0.03, 0.05])), 2)
    return p


def random_emos_global_perturb(base_params: dict, rng: random.Random) -> dict:
    """Perturb global EMOS params slightly."""
    p = dict(base_params)
    if rng.random() < 0.3:
        p["emos_training_window"] = max(7, p["emos_training_window"] + rng.choice([-7, -3, 3, 7]))
    if rng.random() < 0.3:
        p["emos_sigma_floor"] = round(max(0.1, p["emos_sigma_floor"] + rng.choice([-0.2, -0.1, 0.1, 0.2])), 2)
    if rng.random() < 0.3:
        p["emos_sigma_scale"] = round(max(0.3, p["emos_sigma_scale"] + rng.choice([-0.2, -0.1, 0.1, 0.2])), 2)
    if rng.random() < 0.15:
        p["emos_sigma_method"] = rng.choice(["rolling_rmse", "rolling_mae", "ewma_rmse"])
    if rng.random() < 0.15:
        p["emos_bias_method"] = rng.choice(["rolling_mean", "ewma", "none"])
    return p


# ── Single-city evaluation ───────────────────────────────────────────


def evaluate_single_city(
    sim_data: SimulationData,
    city: str,
    city_override: dict,
    base_params: dict,
) -> dict:
    """Run experiment with only one city enabled (all others excluded)."""
    params = dict(base_params)
    # Exclude all cities except the target
    all_cities = set(CITY_COORDS.keys())
    params["excluded_cities"] = all_cities - {city}
    params["city_overrides"] = {city: city_override}

    result = run_experiment(sim_data, params, f"city_{city}")

    return {
        "city": city,
        "score": result.score,
        "trades": result.trades,
        "wins": result.wins,
        "win_rate": result.win_rate,
        "pnl": result.total_pnl,
        "max_dd": result.max_drawdown_pct,
        "sharpe": result.sharpe,
        "utilization": result.capital_utilization,
        "override": city_override,
    }


# ── Main ─────────────────────────────────────────────────────────────


def main():
    t_start = time.time()
    rng = random.Random(42)

    # ── Load data ──
    print("Loading PMD data...")
    events, buckets_by_event, prices = load_pmd_data()
    forecasts = load_forecasts(events)
    sim_data = SimulationData(events, buckets_by_event, prices, forecasts)

    # Build EMOS training observations
    emos_v2._emos_obs = _build_training_observations(events, buckets_by_event, forecasts)

    def log(msg):
        print(msg, flush=True)
        with open(LOG_PATH, "a") as f:
            f.write(msg + "\n")

    log(f"\n{'=' * 70}")
    log(f"EMOS PER-CITY AUTORESEARCH")
    log(f"Events: {len(events)}, Start: {datetime.now().isoformat()}")
    log(f"Global baseline: score=156.5, WR=39.6%, PnL=$32,918")
    log(f"{'=' * 70}\n")

    # ── Run global baseline for reference ──
    log("Running global EMOS baseline...")
    global_result = run_experiment(sim_data, GLOBAL_BASELINE, "global_baseline")
    log(f"  Global: score={global_result.score:.1f}  trades={global_result.trades}  "
        f"WR={global_result.win_rate}%  PnL=${global_result.total_pnl:.2f}")

    # Get cities that have enough data
    cities_with_data = set()
    for ev in events:
        if ev.city and ev.city in CITY_COORDS:
            fc = forecasts.get(ev.city, {}).get(ev.target_date)
            if fc is not None:
                cities_with_data.add(ev.city)

    # Count events per city
    city_event_counts = defaultdict(int)
    for ev in events:
        if ev.city:
            city_event_counts[ev.city] += 1

    tradeable_cities = sorted(
        [c for c in cities_with_data if city_event_counts[c] >= 10],
        key=lambda c: -city_event_counts[c],
    )
    log(f"\nTradeable cities ({len(tradeable_cities)}): "
        f"{[(c, city_event_counts[c]) for c in tradeable_cities]}\n")

    # ── Per-city sweep ──
    N_TRIALS_PER_CITY = 200
    city_best: dict[str, dict] = {}

    for city in tradeable_cities:
        log(f"--- {city.upper()} ({city_event_counts[city]} events) ---")

        # First, evaluate city with global params (no override)
        global_city = evaluate_single_city(
            sim_data, city, {}, GLOBAL_BASELINE
        )
        log(f"  Global params: score={global_city['score']:.1f}  "
            f"trades={global_city['trades']}  WR={global_city['win_rate']}%  "
            f"PnL=${global_city['pnl']:.2f}")

        best_score = global_city["score"]
        best_override = {}
        best_result = global_city
        all_city_results = [global_city]

        # Phase 1: Random search (120 trials)
        for i in range(120):
            override = random_city_override(rng)
            r = evaluate_single_city(sim_data, city, override, GLOBAL_BASELINE)
            all_city_results.append(r)

            if r["score"] > best_score:
                best_score = r["score"]
                best_override = override
                best_result = r
                if i < 30 or r["score"] > best_score * 0.95:
                    log(f"  Trial {i:>3}: score={r['score']:>6.1f}  "
                        f"trades={r['trades']:>3}  WR={r['win_rate']:>5.1f}%  "
                        f"PnL=${r['pnl']:>8.2f}  ★")

        # Phase 2: Fine-tune around best (80 trials)
        if best_override:
            for i in range(80):
                override = perturb_city_override(best_override, rng)
                # Occasionally also perturb global EMOS params
                base = GLOBAL_BASELINE
                if rng.random() < 0.2:
                    base = random_emos_global_perturb(GLOBAL_BASELINE, rng)
                r = evaluate_single_city(sim_data, city, override, base)
                all_city_results.append(r)

                if r["score"] > best_score:
                    best_score = r["score"]
                    best_override = override
                    best_result = r
                    log(f"  Fine {120 + i:>3}: score={r['score']:>6.1f}  "
                        f"PnL=${r['pnl']:>8.2f}  ★")

        city_best[city] = {
            "best_override": best_override,
            "best_result": best_result,
            "global_result": global_city,
            "n_trials": len(all_city_results),
            "improvement": best_score - global_city["score"],
        }

        log(f"  BEST: score={best_result['score']:.1f}  trades={best_result['trades']}  "
            f"WR={best_result['win_rate']}%  PnL=${best_result['pnl']:.2f}  "
            f"(vs global: {'+' if best_result['score'] >= global_city['score'] else ''}"
            f"{best_result['score'] - global_city['score']:.1f})")
        if best_override:
            changed = {k: v for k, v in best_override.items()
                       if v != GLOBAL_BASELINE.get(k)}
            log(f"  Override: {changed}")
        log("")

    # ── Build combined params with all per-city overrides ──
    log(f"{'=' * 70}")
    log("COMBINED: Global + All Per-City Overrides")
    log(f"{'=' * 70}\n")

    combined_params = dict(GLOBAL_BASELINE)
    combined_params["city_overrides"] = {}

    # Only include overrides that improved score
    for city, data in city_best.items():
        if data["improvement"] > 0 and data["best_override"]:
            combined_params["city_overrides"][city] = data["best_override"]
            log(f"  {city}: +{data['improvement']:.1f} score → override included")
        else:
            log(f"  {city}: no improvement → using global params")

    # Determine excluded cities (score=0 even with best override)
    excluded = set()
    for city, data in city_best.items():
        if data["best_result"]["score"] <= 0 and data["best_result"]["pnl"] < 0:
            excluded.add(city)
            log(f"  {city}: EXCLUDED (score=0, negative PnL)")
    combined_params["excluded_cities"] = excluded

    log(f"\n  Excluded: {sorted(excluded)}")
    log(f"  Overrides: {len(combined_params['city_overrides'])} cities")

    # ── Run combined experiment ──
    log("\nRunning combined experiment...")
    combined_result = run_experiment(
        sim_data, combined_params, "emos_percity_combined"
    )
    log(f"  Combined: score={combined_result.score:.1f}  "
        f"trades={combined_result.trades}  WR={combined_result.win_rate}%  "
        f"PnL=${combined_result.total_pnl:.2f}  DD={combined_result.max_drawdown_pct}%  "
        f"Sharpe={combined_result.sharpe}")

    # ── Walk-forward validation ──
    log("\nWalk-forward validation...")
    emos_v2._emos_enabled = True
    emos_v2._emos_params = combined_params
    emos_v2._emos_models = {}
    _prefit_emos_all(combined_params)
    wf = walk_forward_validate(sim_data, combined_params, entry_hours=24, n_folds=3)
    log(f"  WF OOS: score={wf['score']:.1f}  PnL=${wf.get('oos_pnl', 0):.2f}  "
        f"folds={wf.get('n_folds', 0)}")
    if wf.get("folds"):
        for fi, fold in enumerate(wf["folds"]):
            log(f"    Fold {fi + 1}: PnL=${fold['pnl']:.2f}  trades={fold['trades']}  "
                f"WR={fold['wr']}%  score={fold['score']:.1f}")
    emos_v2._emos_enabled = False

    # ── Also run some full combined sweeps (perturb combined params) ──
    log(f"\nPhase 3: Full combined sweep (200 trials)")
    log("-" * 50)

    all_combined_results = []
    best_combined_score = combined_result.score
    best_combined_result = serialize_result(combined_result)
    best_combined_params = combined_params

    rd = serialize_result(combined_result)
    rd["trial"] = 0
    rd["phase"] = "combined"
    all_combined_results.append(rd)

    for i in range(200):
        params = random_emos_global_perturb(dict(combined_params), rng)
        # Perturb some city overrides too
        for city in list(params.get("city_overrides", {}).keys()):
            if rng.random() < 0.3:
                params["city_overrides"][city] = perturb_city_override(
                    params["city_overrides"][city], rng
                )

        result = run_experiment(sim_data, params, f"EMOS-PC-{i:04d}")
        rd = serialize_result(result)
        rd["trial"] = i + 1
        rd["phase"] = "combined_sweep"
        all_combined_results.append(rd)

        if result.score > best_combined_score:
            best_combined_score = result.score
            best_combined_result = rd
            best_combined_params = params
            log(f"  Trial {i:>3}: score={result.score:>6.1f}  "
                f"trades={result.trades:>3}  PnL=${result.total_pnl:>8.2f}  ★")

        if (i + 1) % 50 == 0:
            log(f"  ... {i + 1}/200 done, best: {best_combined_score:.1f}")

    # ── Save results ──
    sorted_combined = sorted(all_combined_results, key=lambda r: -r["score"])

    output = {
        "sweep_id": "emos_percity",
        "sweep_type": "EMOS Per-City Autoresearch",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "global_baseline": serialize_result(global_result),
        "combined_result": serialize_result(combined_result),
        "walk_forward": wf,
        "city_analysis": {
            city: {
                "best_override": data["best_override"],
                "global_score": data["global_result"]["score"],
                "best_score": data["best_result"]["score"],
                "improvement": data["improvement"],
                "best_trades": data["best_result"]["trades"],
                "best_wr": data["best_result"]["win_rate"],
                "best_pnl": data["best_result"]["pnl"],
            }
            for city, data in city_best.items()
        },
        "excluded_cities": sorted(excluded),
        "best": best_combined_result,
        "top_results": sorted_combined[:20],
        "all_results": sorted_combined,
    }

    with open(SWEEP_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)

    elapsed = time.time() - t_start

    # ── Final report ──
    log(f"\n{'=' * 70}")
    log(f"PER-CITY SWEEP COMPLETE — {elapsed / 60:.1f} minutes")
    log(f"{'=' * 70}")

    log(f"\n  GLOBAL EMOS:   score={global_result.score:.1f}  trades={global_result.trades}  "
        f"WR={global_result.win_rate}%  PnL=${global_result.total_pnl:.2f}")
    log(f"  PER-CITY EMOS: score={best_combined_score:.1f}  "
        f"trades={best_combined_result['trades']}  "
        f"WR={best_combined_result['win_rate']}%  "
        f"PnL=${best_combined_result['total_pnl']:.2f}")

    delta = best_combined_score - global_result.score
    delta_pnl = best_combined_result["total_pnl"] - global_result.total_pnl
    log(f"\n  Delta: score {'+' if delta >= 0 else ''}{delta:.1f}, "
        f"PnL {'+' if delta_pnl >= 0 else ''}${delta_pnl:.2f}")

    log(f"\n  Per-city improvements:")
    for city in sorted(city_best, key=lambda c: -city_best[c]["improvement"]):
        d = city_best[city]
        marker = "★" if d["improvement"] > 0 else "  "
        log(f"    {marker} {city:<16} global={d['global_result']['score']:>6.1f}  "
            f"best={d['best_result']['score']:>6.1f}  "
            f"Δ={d['improvement']:>+6.1f}  "
            f"WR={d['best_result']['win_rate']}%  "
            f"PnL=${d['best_result']['pnl']:.2f}")

    log(f"\n  Results: {SWEEP_PATH}")
    log(f"  Dashboard: /experiments → select 'emos_percity'")


if __name__ == "__main__":
    main()
