"""EMOS V3 — Dynamic Sizing + Aggregate Exposure Cap.

Changes from V2:
- Removed $200 flat position cap → pure % of capital
- Added 40% aggregate exposure cap
- Position size scales with capital AND edge (true Kelly)
- Sweep includes max_aggregate_pct as tunable

Usage:
    python3 research/innovations/sweep_emos_v3.py
"""

from __future__ import annotations

import json
import random
import sys
import time
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
from sweep_emos_v2 import (
    _build_training_observations,
    _prefit_emos_all,
    run_experiment,
)
import sweep_emos_v2 as emos_v2

RESULTS_DIR = Path(__file__).parent.parent / "data" / "experiments"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
SWEEP_PATH = RESULTS_DIR / "sweep_emos_v3.json"
LOG_PATH = Path(__file__).parent.parent / "sweep_emos_v3_log.txt"


def random_params(rng: random.Random) -> dict:
    """V3 params: dynamic sizing, aggregate cap, EMOS + gate."""
    return {
        # EMOS
        "emos_training_window": rng.choice([14, 21, 30, 45, 60]),
        "emos_sigma_method": rng.choice(["rolling_rmse", "rolling_mae", "ewma_rmse"]),
        "emos_bias_method": rng.choice(["rolling_mean", "ewma", "none"]),
        "emos_sigma_floor": rng.choice([0.3, 0.5, 0.7, 1.0, 1.2]),
        "emos_sigma_scale": rng.choice([0.8, 1.0, 1.2, 1.5, 2.0]),
        "emos_ewma_alpha": rng.choice([0.1, 0.15, 0.2, 0.3]),
        # Quality gate
        "min_edge": rng.choice([0.02, 0.03, 0.05, 0.08, 0.10, 0.15]),
        "max_edge": 0.90,
        "max_price": rng.choice([0.40, 0.50, 0.55, 0.60, 0.70, 0.80]),
        "min_price": rng.choice([0.01, 0.02, 0.03, 0.05, 0.08]),
        "min_prob": rng.choice([0.02, 0.05, 0.08, 0.10, 0.15]),
        "min_volume": 0,
        # Dynamic sizing (V3 new)
        "kelly_raw_cap": rng.choice([0.20, 0.25, 0.30, 0.35, 0.40, 0.50]),
        "max_aggregate_pct": rng.choice([0.25, 0.30, 0.35, 0.40, 0.50, 0.60]),
        # No flat USD cap
        "prob_sharpening": rng.choice([0.90, 0.95, 1.0, 1.05, 1.10, 1.15, 1.20]),
        "shrinkage": rng.choice([0.0, 0.03, 0.05, 0.10]),
        "excluded_cities": set(),
        "city_overrides": {},
        "exit_enabled": False,
    }


def perturb_params(base: dict, rng: random.Random) -> dict:
    p = dict(base)
    p["excluded_cities"] = set(p.get("excluded_cities", set()))
    p["city_overrides"] = dict(p.get("city_overrides", {}))

    keys = [
        "emos_training_window", "emos_sigma_floor", "emos_sigma_scale",
        "min_edge", "max_price", "min_price", "min_prob",
        "kelly_raw_cap", "max_aggregate_pct",
        "prob_sharpening", "shrinkage",
    ]

    for key in rng.sample(keys, min(rng.randint(1, 4), len(keys))):
        val = p.get(key, 0)
        if key == "emos_training_window":
            p[key] = max(7, val + rng.choice([-7, -3, 3, 7, 14]))
        elif key in ("emos_sigma_floor", "emos_sigma_scale"):
            p[key] = round(max(0.1, val + rng.choice([-0.2, -0.1, 0.1, 0.2])), 2)
        elif key in ("kelly_raw_cap", "max_aggregate_pct"):
            p[key] = round(max(0.10, min(0.70, val + rng.choice([-0.05, 0.05, 0.10]))), 2)
        elif key == "prob_sharpening":
            p[key] = round(val + rng.choice([-0.05, 0.05, 0.10]), 2)
        elif key == "shrinkage":
            p[key] = round(max(0, val + rng.choice([-0.03, 0.03])), 2)
        else:
            p[key] = round(max(0.01, val + rng.choice([-0.03, -0.02, 0.02, 0.03])), 2)

    if rng.random() < 0.15:
        p["emos_sigma_method"] = rng.choice(["rolling_rmse", "rolling_mae", "ewma_rmse"])
    if rng.random() < 0.15:
        p["emos_bias_method"] = rng.choice(["rolling_mean", "ewma", "none"])
    return p


def main():
    t_start = time.time()
    rng = random.Random(42)

    print("Loading data...")
    events, buckets_by_event, prices = load_pmd_data()
    forecasts = load_forecasts(events)
    sim_data = SimulationData(events, buckets_by_event, prices, forecasts)
    emos_v2._emos_obs = _build_training_observations(events, buckets_by_event, forecasts)

    all_results = []
    best_score = -1
    best_result = None
    best_params = None

    def log(msg):
        print(msg, flush=True)
        with open(LOG_PATH, "a") as f:
            f.write(msg + "\n")

    log(f"\n{'=' * 70}")
    log(f"EMOS V3 — Dynamic Sizing + Aggregate Cap")
    log(f"Events: {len(events)}, Start: {datetime.now().isoformat()}")
    log(f"No flat $200 cap. Sizing = min(quarter-Kelly, 5% capital).")
    log(f"Aggregate exposure cap (tunable, default 40%).")
    log(f"{'=' * 70}\n")

    # ── V2 baseline for comparison (with old $200 cap) ──
    log("Running V2 reference (old $200 cap)...")
    v2_params = {
        "emos_training_window": 45, "emos_sigma_method": "rolling_mae",
        "emos_bias_method": "rolling_mean", "emos_sigma_floor": 0.5,
        "emos_sigma_scale": 1.5, "emos_ewma_alpha": 0.3,
        "min_edge": 0.15, "max_edge": 0.90, "max_price": 0.66,
        "min_price": 0.08, "min_prob": 0.07, "min_volume": 0,
        "kelly_raw_cap": 0.35, "prob_sharpening": 1.0, "shrinkage": 0.10,
        "max_position_usd": 200,  # Old flat cap for comparison
        "max_aggregate_pct": 1.0,  # No aggregate limit (V2 behavior)
        "excluded_cities": set(), "city_overrides": {}, "exit_enabled": False,
    }
    v2_ref = run_experiment(sim_data, v2_params, "v2_reference")
    log(f"  V2 ($200 cap): score={v2_ref.score:.1f}  trades={v2_ref.trades}  "
        f"WR={v2_ref.win_rate}%  PnL=${v2_ref.total_pnl:.2f}  "
        f"Util={v2_ref.capital_utilization:.1f}%  Concurrent={v2_ref.concurrent_positions:.1f}")

    # ── Phase 1: Random search (600 trials) ──
    N1 = 600
    log(f"\nPhase 1: Random Search ({N1} trials)")
    log("-" * 50)

    for trial in range(N1):
        params = random_params(rng)
        eid = f"V3-{trial:04d}"
        result = run_experiment(sim_data, params, eid)
        rd = serialize_result(result)
        rd["trial"] = trial
        rd["phase"] = 1
        all_results.append(rd)

        if result.score > best_score:
            best_score = result.score
            best_result = rd
            best_params = params
            log(f"  {eid}: score={result.score:>6.1f}  trades={result.trades:>3}  "
                f"WR={result.win_rate:>5.1f}%  PnL=${result.total_pnl:>8.2f}  "
                f"Util={result.capital_utilization:.1f}%  "
                f"Conc={result.concurrent_positions:.1f}  "
                f"Agg={params['max_aggregate_pct']:.0%}  ★")

        if (trial + 1) % 100 == 0:
            log(f"  ... {trial + 1}/{N1} done, best: {best_score:.1f}")

    # ── Phase 2: Fine-tuning (400 trials) ──
    N2 = 400
    log(f"\nPhase 2: Fine-tuning ({N2} trials)")
    log("-" * 50)
    top10 = sorted(all_results, key=lambda r: -r["score"])[:10]
    log(f"  Top 10: {[r['score'] for r in top10]}")

    for i in range(N2):
        base_r = top10[i % 10]
        base_p = dict(base_r["params"])
        base_p["excluded_cities"] = set(base_p.get("excluded_cities", []))
        base_p["city_overrides"] = dict(base_p.get("city_overrides", {}))

        params = perturb_params(base_p, rng)
        eid = f"V3-{N1 + i:04d}"
        result = run_experiment(sim_data, params, eid)
        rd = serialize_result(result)
        rd["trial"] = N1 + i
        rd["phase"] = 2
        all_results.append(rd)

        if result.score > best_score:
            best_score = result.score
            best_result = rd
            best_params = params
            log(f"  {eid}: score={result.score:>6.1f}  PnL=${result.total_pnl:>8.2f}  "
                f"Util={result.capital_utilization:.1f}%  ★")

        if (i + 1) % 100 == 0:
            log(f"  ... {i + 1}/{N2} done, best: {best_score:.1f}")

    # ── Phase 3: City optimization (200 trials) ──
    N3 = 200
    log(f"\nPhase 3: City optimization ({N3} trials)")
    log("-" * 50)

    losing_cities, winning_cities = [], []
    if best_result:
        cr = best_result.get("city_results", {})
        losing_cities = sorted([c for c, v in cr.items() if v["pnl"] < 0], key=lambda c: cr[c]["pnl"])
        winning_cities = sorted([c for c, v in cr.items() if v["pnl"] > 0], key=lambda c: -cr[c]["pnl"])
        log(f"  Losing: {losing_cities}")
        log(f"  Winning: {winning_cities}")

    for i in range(N3):
        params = dict(best_params)
        params["excluded_cities"] = set(params.get("excluded_cities", set()))
        params["city_overrides"] = dict(params.get("city_overrides", {}))

        if i < 50 and losing_cities:
            params["excluded_cities"] = set(rng.sample(losing_cities, min(rng.randint(1, len(losing_cities)), 5)))
        elif i < 100 and winning_cities:
            params["excluded_cities"] = set(losing_cities)
            for city in rng.sample(winning_cities, min(len(winning_cities), 3)):
                params["city_overrides"][city] = {
                    "max_price": rng.choice([0.50, 0.60, 0.70, 0.80]),
                    "min_edge": rng.choice([0.01, 0.02, 0.03, 0.05]),
                    "kelly_raw_cap": rng.choice([0.25, 0.30, 0.35, 0.40, 0.50]),
                }
        else:
            params = perturb_params(params, rng)
            if losing_cities and rng.random() < 0.5:
                params["excluded_cities"] = set(losing_cities)

        eid = f"V3-{N1 + N2 + i:04d}"
        result = run_experiment(sim_data, params, eid)
        rd = serialize_result(result)
        rd["trial"] = N1 + N2 + i
        rd["phase"] = 3
        all_results.append(rd)

        if result.score > best_score:
            best_score = result.score
            best_result = rd
            best_params = params
            log(f"  {eid}: score={result.score:>6.1f}  PnL=${result.total_pnl:>8.2f}  "
                f"excluded={sorted(params['excluded_cities'])}  ★")

        if (i + 1) % 50 == 0:
            log(f"  ... {i + 1}/{N3} done, best: {best_score:.1f}")

    # ── Walk-forward ──
    log("\nWalk-forward validation...")
    emos_v2._emos_enabled = True
    emos_v2._emos_params = best_params
    emos_v2._emos_models = {}
    _prefit_emos_all(best_params)
    wf = walk_forward_validate(sim_data, best_params, entry_hours=24, n_folds=3)
    log(f"  WF: score={wf['score']:.1f}  OOS PnL=${wf.get('oos_pnl', 0):.2f}")
    if wf.get("folds"):
        for fi, fold in enumerate(wf["folds"]):
            log(f"    Fold {fi + 1}: PnL=${fold['pnl']:.2f}  trades={fold['trades']}  "
                f"WR={fold['wr']}%  util={fold.get('utilization', 0):.1f}%")
    emos_v2._emos_enabled = False

    # ── Save ──
    sorted_results = sorted(all_results, key=lambda r: -r["score"])
    output = {
        "sweep_id": "emos_v3",
        "sweep_type": "EMOS V3 — Dynamic Sizing + Aggregate Cap",
        "meta": {
            "sweep_id": "emos_v3",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_trials": len(all_results),
            "sweep_type": "EMOS V3 Dynamic Sizing",
        },
        "v2_reference": serialize_result(v2_ref),
        "walk_forward": wf,
        "best": best_result,
        "top_results": sorted_results[:20],
        "all_results": sorted_results,
    }
    with open(SWEEP_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)

    elapsed = time.time() - t_start

    # ── Report ──
    log(f"\n{'=' * 70}")
    log(f"V3 SWEEP COMPLETE — {elapsed / 60:.1f} minutes")
    log(f"{'=' * 70}")
    log(f"\n  V2 ($200 cap):  score={v2_ref.score:.1f}  trades={v2_ref.trades}  "
        f"Util={v2_ref.capital_utilization:.1f}%  PnL=${v2_ref.total_pnl:.2f}")
    log(f"  V3 (dynamic):   score={best_score:.1f}  trades={best_result['trades']}  "
        f"Util={best_result['capital_utilization']:.1f}%  PnL=${best_result['total_pnl']:.2f}")
    log(f"\n  Delta: score {best_score - v2_ref.score:+.1f}, "
        f"util {best_result['capital_utilization'] - v2_ref.capital_utilization:+.1f}%, "
        f"PnL ${best_result['total_pnl'] - v2_ref.total_pnl:+,.2f}")

    bp = best_result["params"]
    log(f"\n  Best V3 params:")
    for k in ["emos_training_window", "emos_sigma_method", "emos_bias_method",
              "emos_sigma_floor", "emos_sigma_scale", "kelly_raw_cap",
              "max_aggregate_pct", "min_edge", "max_price", "min_price",
              "prob_sharpening", "shrinkage"]:
        log(f"    {k}: {bp.get(k)}")
    log(f"    excluded_cities: {bp.get('excluded_cities', [])}")

    log(f"\n  Per-city:")
    cr = best_result.get("city_results", {})
    for city in sorted(cr, key=lambda c: -cr[c]["pnl"]):
        v = cr[city]
        log(f"    {city:<16} ${v['pnl']:>8.2f}  ({v['trades']} trades, WR={v['win_rate']}%)")

    log(f"\n  Results: {SWEEP_PATH}")
    log(f"  Dashboard: /experiments → select 'emos_v3'")


if __name__ == "__main__":
    main()
