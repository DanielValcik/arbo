"""EMOS V4 Targeted — Shrinkage Variants A/B Test.

NOT a full autoresearch. Tests exactly 2 variants against V3 baseline:

V4a: Soft shrinkage (k=15), NO exposure cap.
     Blending works, but city exposure is flat 25% for all.
     Shrinkage alone dampens overconfidence.

V4b: No shrinkage, ONLY progressive exposure cap.
     Full per-city params (like V3), but √(n/30) × 25% caps sizing.
     Edge detection maximal, sizing conservative.

All other params locked to V3 best values.
Walk-forward OOS on all variants.

Usage:
    python3 research/innovations/sweep_emos_v4_targeted.py
"""

from __future__ import annotations

import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from pmd_loader import load_pmd_data
import experiment_framework as ef
from experiment_framework import (
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
SWEEP_PATH = RESULTS_DIR / "sweep_emos_v4_targeted.json"
LOG_PATH = Path(__file__).parent.parent / "sweep_emos_v4_targeted_log.txt"

# ── V3 best params (LOCKED — do not modify) ─────────────────────────

V3_BASE = {
    "emos_training_window": 66,
    "emos_sigma_method": "rolling_rmse",
    "emos_bias_method": "rolling_mean",
    "emos_sigma_floor": 0.6,
    "emos_sigma_scale": 1.1,
    "emos_ewma_alpha": 0.3,
    "min_edge": 0.10,
    "max_edge": 0.90,
    "max_price": 0.56,
    "min_price": 0.08,
    "min_prob": 0.07,
    "min_volume": 0,
    "kelly_raw_cap": 0.40,
    "prob_sharpening": 1.10,
    "shrinkage": 0.03,
    "max_aggregate_pct": 0.55,
    "excluded_cities": {"lucknow", "wellington"},
    "city_overrides": {
        "munich": {"min_edge": 0.05, "max_price": 0.80, "kelly_raw_cap": 0.50},
        "seoul": {"min_edge": 0.05, "max_price": 0.70, "kelly_raw_cap": 0.45},
        "atlanta": {"min_edge": 0.08, "max_price": 0.70, "kelly_raw_cap": 0.40},
        "nyc": {"min_edge": 0.08, "max_price": 0.70},
        "toronto": {"min_edge": 0.08, "max_price": 0.60},
    },
    "exit_enabled": False,
}


def make_v3_reference() -> dict:
    """V3: no shrinkage, no exposure cap (original V3 behavior)."""
    p = dict(V3_BASE)
    p["shrinkage_pseudocount"] = 999999  # effectively disables blending (α≈1)
    p["city_max_exposure"] = 1.0  # no per-city cap
    return p


def make_v4a(k: int) -> dict:
    """V4a: soft shrinkage (tunable k), NO exposure cap."""
    p = dict(V3_BASE)
    p["shrinkage_pseudocount"] = k
    p["city_max_exposure"] = 1.0  # NO progressive exposure cap
    return p


def make_v4b() -> dict:
    """V4b: no shrinkage, ONLY progressive exposure cap."""
    p = dict(V3_BASE)
    p["shrinkage_pseudocount"] = 999999  # disable blending
    p["city_max_exposure"] = 0.25  # progressive cap active
    return p


def run_variant(
    sim_data: SimulationData,
    params: dict,
    name: str,
    log_fn,
) -> dict:
    """Run one variant with full metrics + walk-forward."""
    result = run_experiment(sim_data, params, name)

    # Walk-forward
    emos_v2._emos_enabled = True
    emos_v2._emos_params = params
    emos_v2._emos_models = {}
    _prefit_emos_all(params)
    wf = walk_forward_validate(sim_data, params, entry_hours=24, n_folds=3)
    emos_v2._emos_enabled = False

    rd = serialize_result(result)
    rd["walk_forward"] = wf

    log_fn(f"\n  {name}:")
    log_fn(f"    Score: {result.score:.1f}  Trades: {result.trades}  WR: {result.win_rate}%")
    log_fn(f"    PnL: ${result.total_pnl:,.2f}  DD: {result.max_drawdown_pct:.1f}%  Sharpe: {result.sharpe}")
    log_fn(f"    Util: {result.capital_utilization:.1f}%  Concurrent: {result.concurrent_positions:.1f}")
    log_fn(f"    Turnover: {result.turnover_rate:.1f}  PnL/hour: ${result.avg_pnl_per_hour:.2f}")

    # City concentration
    cr = result.city_results
    total = result.total_pnl
    if total > 0:
        max_city = max(cr.items(), key=lambda x: x[1]["pnl"])
        max_pct = max_city[1]["pnl"] / total * 100
        log_fn(f"    Max city: {max_city[0]} = {max_pct:.1f}% of PnL ({max_city[1]['trades']} trades)")
    else:
        max_pct = 0
        log_fn(f"    PnL negative — no concentration check")

    log_fn(f"    WF OOS: score={wf['score']:.1f}  PnL=${wf.get('oos_pnl', 0):.2f}")
    if wf.get("folds"):
        for fi, fold in enumerate(wf["folds"]):
            log_fn(f"      Fold {fi+1}: PnL=${fold['pnl']:.2f}  "
                   f"trades={fold['trades']}  WR={fold['wr']}%  "
                   f"util={fold.get('utilization', 0):.1f}%")

    # Per-city breakdown
    log_fn(f"    Per-city:")
    for city in sorted(cr, key=lambda c: -cr[c]["pnl"]):
        v = cr[city]
        pct = v["pnl"] / total * 100 if total > 0 else 0
        log_fn(f"      {city:<14} ${v['pnl']:>8,.2f}  ({v['trades']:>3} tr, WR={v['win_rate']:>5.1f}%)  {pct:>5.1f}%")

    return rd


def main():
    t_start = time.time()

    print("Loading data...")
    events, buckets, prices = load_pmd_data()
    forecasts = load_forecasts(events)
    sim_data = SimulationData(events, buckets, prices, forecasts)
    emos_v2._emos_obs = _build_training_observations(events, buckets, forecasts)

    def log(msg):
        print(msg, flush=True)
        with open(LOG_PATH, "a") as f:
            f.write(msg + "\n")

    log(f"\n{'=' * 70}")
    log(f"EMOS V4 TARGETED — Shrinkage A/B Test")
    log(f"Events: {len(events)}, {datetime.now().isoformat()}")
    log(f"All params locked to V3 best. Only shrinkage varies.")
    log(f"{'=' * 70}")

    results = {}

    # ── V3 reference (no shrinkage, no cap) ──
    log(f"\n{'=' * 50}")
    log("V3 REFERENCE (no shrinkage, no exposure cap)")
    results["v3_ref"] = run_variant(sim_data, make_v3_reference(), "v3_reference", log)

    # ── V4a variants (soft shrinkage, no exposure cap) ──
    for k in [10, 15, 20, 25, 30]:
        log(f"\n{'=' * 50}")
        log(f"V4a (k={k}, soft shrinkage, NO exposure cap)")
        results[f"v4a_k{k}"] = run_variant(sim_data, make_v4a(k), f"v4a_k{k}", log)

    # ── V4b (no shrinkage, only exposure cap) ──
    log(f"\n{'=' * 50}")
    log("V4b (NO shrinkage, progressive exposure cap √(n/30) × 25%)")
    results["v4b"] = run_variant(sim_data, make_v4b(), "v4b_expcap", log)

    # ── Summary table ──
    log(f"\n{'=' * 70}")
    log("COMPARISON TABLE")
    log(f"{'=' * 70}")
    log(f"{'Variant':<16} {'Score':>6} {'Trades':>7} {'WR':>6} {'PnL':>10} {'DD':>6} {'Sharpe':>7} {'Util':>6} {'MaxCity':>8} {'OOS PnL':>9} {'OOS Score':>9}")
    log("-" * 100)

    for name in ["v3_ref", "v4a_k10", "v4a_k15", "v4a_k20", "v4a_k25", "v4a_k30", "v4b"]:
        r = results[name]
        cr = r.get("city_results", {})
        total = r["total_pnl"]
        max_city_pct = 0
        if total > 0 and cr:
            max_city_pct = max(v["pnl"] / total * 100 for v in cr.values())

        wf = r.get("walk_forward", {})
        oos_pnl = wf.get("oos_pnl", 0)
        oos_score = wf.get("score", 0)

        label = name.replace("_ref", "").replace("_expcap", "")
        log(f"{label:<16} {r['score']:>6.1f} {r['trades']:>7} {r['win_rate']:>5.1f}% "
            f"${r['total_pnl']:>9,.2f} {r['max_drawdown_pct']:>5.1f}% {r['sharpe']:>7.2f} "
            f"{r['capital_utilization']:>5.1f}% {max_city_pct:>7.1f}% "
            f"${oos_pnl:>8,.2f} {oos_score:>9.1f}")

    # ── Save ──
    output = {
        "sweep_id": "emos_v4_targeted",
        "sweep_type": "EMOS V4 Targeted — Shrinkage A/B",
        "meta": {
            "sweep_id": "emos_v4_targeted",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_trials": len(results),
            "sweep_type": "EMOS V4 Shrinkage A/B",
        },
        "variants": results,
        # Dashboard compatibility: put all variants as "all_results"
        "best": max(results.values(), key=lambda r: r.get("score", 0)),
        "top_results": sorted(results.values(), key=lambda r: -r.get("score", 0)),
        "all_results": sorted(results.values(), key=lambda r: -r.get("score", 0)),
    }

    with open(SWEEP_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)

    elapsed = time.time() - t_start
    log(f"\nDone in {elapsed:.0f}s. Results: {SWEEP_PATH}")


if __name__ == "__main__":
    main()
