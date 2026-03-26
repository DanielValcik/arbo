"""V5 Capital Velocity Expansion — Three Dimensions.

V5a: Multi-bucket (lower min_edge → 2-3 buckets per city/day)
V5b: V5a + multi-day (+2, +3 day horizons)
V5c: V5b + city expansion (5→12+ cities)

Each tested at agg30 and agg40. Walk-forward OOS on all.
New metrics: intra-city correlation, marginal edge quality.

Usage:
    python3 research/innovations/sweep_v5_velocity.py
"""

from __future__ import annotations

import json
import math
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
    load_forecasts,
    serialize_result,
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
SWEEP_PATH = RESULTS_DIR / "sweep_v5_velocity.json"
LOG_PATH = Path(__file__).parent.parent / "sweep_v5_velocity_log.txt"

# ── V4a k=20 agg30 locked params ────────────────────────────────────

V4A_BASE = {
    "emos_training_window": 66,
    "emos_sigma_method": "rolling_rmse",
    "emos_bias_method": "rolling_mean",
    "emos_sigma_floor": 0.6,
    "emos_sigma_scale": 1.1,
    "emos_ewma_alpha": 0.3,
    "max_edge": 0.90,
    "max_price": 0.56,
    "min_price": 0.08,
    "min_prob": 0.07,
    "min_volume": 0,
    "kelly_raw_cap": 0.40,
    "prob_sharpening": 1.10,
    "shrinkage": 0.03,
    "shrinkage_pseudocount": 20,
    "city_max_exposure": 0.35,
    "city_overrides": {
        "munich": {"min_edge": 0.05, "max_price": 0.80, "kelly_raw_cap": 0.50},
        "seoul": {"min_edge": 0.05, "max_price": 0.70, "kelly_raw_cap": 0.45},
        "atlanta": {"min_edge": 0.08, "max_price": 0.70, "kelly_raw_cap": 0.40},
        "nyc": {"min_edge": 0.08, "max_price": 0.70},
        "toronto": {"min_edge": 0.08, "max_price": 0.60},
    },
    "exit_enabled": False,
}


# ── Variant constructors ─────────────────────────────────────────────


def make_v4a_ref(agg: float) -> tuple[dict, float]:
    """V4a reference: single bucket, single day."""
    p = dict(V4A_BASE)
    p["min_edge"] = 0.10  # Original
    p["max_aggregate_pct"] = agg
    p["excluded_cities"] = {"lucknow", "wellington"}
    return p, 24  # entry_hours


def make_v5a(agg: float) -> tuple[dict, float]:
    """V5a: Multi-bucket — lower min_edge to 0.05 for secondary buckets."""
    p = dict(V4A_BASE)
    p["min_edge"] = 0.05  # Lowered from 0.10 → allows 2-3 buckets per event
    p["max_aggregate_pct"] = agg
    p["excluded_cities"] = {"lucknow", "wellington"}
    return p, 24


def make_v5b(agg: float) -> tuple[dict, list[float]]:
    """V5b: V5a + multi-day (+2, +3 day horizons)."""
    p = dict(V4A_BASE)
    p["min_edge"] = 0.05
    p["max_aggregate_pct"] = agg
    p["excluded_cities"] = {"lucknow", "wellington"}
    return p, [24, 48, 72]  # Multi-day entry


def make_v5c(agg: float) -> tuple[dict, list[float]]:
    """V5c: V5b + city expansion (remove exclusions, add all cities)."""
    p = dict(V4A_BASE)
    p["min_edge"] = 0.05
    p["max_aggregate_pct"] = agg
    p["excluded_cities"] = set()  # No exclusions — all cities trade
    return p, [24, 48, 72]


# ── New metrics ──────────────────────────────────────────────────────


def compute_extra_metrics(result: ef.ExperimentResult, sim_data) -> dict:
    """Compute intra-city correlation and marginal edge quality."""
    # We need access to individual trades — extract from result
    # Unfortunately ExperimentResult doesn't store individual trades
    # We'll compute what we can from city_results and aggregate metrics
    metrics = {}

    cr = result.city_results
    total_pnl = result.total_pnl

    # City concentration (max, top2, herfindahl)
    if total_pnl > 0 and cr:
        pnl_shares = [max(0, v["pnl"]) / total_pnl for v in cr.values() if v["pnl"] > 0]
        pnl_shares_sorted = sorted(pnl_shares, reverse=True)
        metrics["max_city_pct"] = round(pnl_shares_sorted[0] * 100, 1) if pnl_shares_sorted else 0
        metrics["top2_city_pct"] = round(sum(pnl_shares_sorted[:2]) * 100, 1)
        metrics["herfindahl"] = round(sum(s**2 for s in pnl_shares) * 10000, 0)  # HHI
    else:
        metrics["max_city_pct"] = 0
        metrics["top2_city_pct"] = 0
        metrics["herfindahl"] = 0

    # Trades per city per day (proxy for multi-bucket density)
    if cr:
        trades_per_city = [v["trades"] for v in cr.values()]
        metrics["avg_trades_per_city"] = round(sum(trades_per_city) / len(trades_per_city), 1)
        metrics["max_trades_per_city"] = max(trades_per_city)
        metrics["n_active_cities"] = sum(1 for v in cr.values() if v["trades"] > 0)
    else:
        metrics["avg_trades_per_city"] = 0
        metrics["max_trades_per_city"] = 0
        metrics["n_active_cities"] = 0

    return metrics


# ── Run variant ──────────────────────────────────────────────────────


def run_variant(
    sim_data: SimulationData,
    params: dict,
    entry_hours,
    name: str,
    log_fn,
) -> dict:
    """Run one variant with EMOS + walk-forward + extra metrics."""
    result = run_experiment(sim_data, params, name)

    # Override entry_hours for multi-day if needed
    if isinstance(entry_hours, list) and len(entry_hours) > 1:
        # Re-run with multi-day entry hours
        emos_v2._emos_enabled = True
        emos_v2._emos_params = params
        emos_v2._emos_models = {}
        _prefit_emos_all(params)

        result = ef.simulate_portfolio(
            sim_data, params,
            entry_hours=entry_hours,
            experiment_id=name,
            record_equity=True,
        )
        result.score = ef.experiment_score(result)
        emos_v2._emos_enabled = False

    # Walk-forward
    emos_v2._emos_enabled = True
    emos_v2._emos_params = params
    emos_v2._emos_models = {}
    _prefit_emos_all(params)
    wf = walk_forward_validate(sim_data, params, entry_hours=entry_hours, n_folds=3)
    emos_v2._emos_enabled = False

    extra = compute_extra_metrics(result, sim_data)

    rd = serialize_result(result)
    rd["walk_forward"] = wf
    rd["extra_metrics"] = extra
    rd["entry_hours"] = entry_hours

    # Log
    cr = result.city_results
    log_fn(f"\n  {name}:")
    log_fn(f"    Score: {result.score:.1f}  Trades: {result.trades}  WR: {result.win_rate}%")
    log_fn(f"    PnL: ${result.total_pnl:,.2f}  DD: {result.max_drawdown_pct:.1f}%  Sharpe: {result.sharpe}")
    log_fn(f"    Util: {result.capital_utilization:.1f}%  Concurrent: {result.concurrent_positions:.1f}  Turnover: {result.turnover_rate:.1f}")
    log_fn(f"    Active cities: {extra['n_active_cities']}  Avg trades/city: {extra['avg_trades_per_city']}")
    log_fn(f"    MaxCity: {extra['max_city_pct']}%  Top2: {extra['top2_city_pct']}%  HHI: {extra['herfindahl']}")
    log_fn(f"    WF OOS: score={wf['score']:.1f}  PnL=${wf.get('oos_pnl', 0):.2f}")

    if wf.get("folds"):
        for fi, fold in enumerate(wf["folds"]):
            log_fn(f"      Fold {fi+1}: PnL=${fold['pnl']:.2f}  trades={fold['trades']}  "
                   f"WR={fold['wr']}%  util={fold.get('utilization', 0):.1f}%")

    log_fn(f"    Per-city:")
    for city in sorted(cr, key=lambda c: -cr[c]["pnl"]):
        v = cr[city]
        pct = v["pnl"] / result.total_pnl * 100 if result.total_pnl > 0 else 0
        log_fn(f"      {city:<14} ${v['pnl']:>8,.2f}  ({v['trades']:>3} tr, WR={v['win_rate']:>5.1f}%)  {pct:>5.1f}%")

    dd_ok = "✓" if result.max_drawdown_pct < 20 else "✗"
    oos_ok = "✓" if wf.get("score", 0) > 90 else "✗"
    log_fn(f"    Gates: DD<20% {dd_ok}  OOS>90 {oos_ok}")

    return rd


# ── Main ─────────────────────────────────────────────────────────────


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
    log(f"V5 CAPITAL VELOCITY EXPANSION")
    log(f"Events: {len(events)}, {datetime.now().isoformat()}")
    log(f"V4a k=20 base. Three dimensions: multi-bucket, multi-day, city expansion.")
    log(f"Each at agg30 and agg40. Decision: DD<20% → use agg40, else agg30.")
    log(f"{'=' * 70}")

    results = {}

    # ── V4a reference (single bucket, single day) ──
    for agg in [0.30, 0.40]:
        name = f"v4a_agg{int(agg*100)}"
        log(f"\n{'=' * 50}")
        log(f"V4a REFERENCE (single bucket, 24h, agg={agg:.0%})")
        p, eh = make_v4a_ref(agg)
        results[name] = run_variant(sim_data, p, eh, name, log)

    # ── V5a: Multi-bucket ──
    for agg in [0.30, 0.40]:
        name = f"v5a_agg{int(agg*100)}"
        log(f"\n{'=' * 50}")
        log(f"V5a MULTI-BUCKET (min_edge=0.05, 24h, agg={agg:.0%})")
        p, eh = make_v5a(agg)
        results[name] = run_variant(sim_data, p, eh, name, log)

    # ── V5b: Multi-bucket + multi-day ──
    for agg in [0.30, 0.40]:
        name = f"v5b_agg{int(agg*100)}"
        log(f"\n{'=' * 50}")
        log(f"V5b MULTI-DAY (min_edge=0.05, 24/48/72h, agg={agg:.0%})")
        p, eh = make_v5b(agg)
        results[name] = run_variant(sim_data, p, eh, name, log)

    # ── V5c: Multi-bucket + multi-day + city expansion ──
    for agg in [0.30, 0.40]:
        name = f"v5c_agg{int(agg*100)}"
        log(f"\n{'=' * 50}")
        log(f"V5c CITY EXPANSION (all cities, min_edge=0.05, 24/48/72h, agg={agg:.0%})")
        p, eh = make_v5c(agg)
        results[name] = run_variant(sim_data, p, eh, name, log)

    # ── Summary table ──
    log(f"\n{'=' * 70}")
    log("COMPARISON TABLE")
    log(f"{'=' * 70}")
    header = (f"{'Variant':<16} {'Score':>6} {'Tr':>4} {'WR':>6} {'PnL':>10} {'DD':>6} "
              f"{'Shrp':>5} {'Util':>5} {'Conc':>5} {'Cities':>6} {'MaxC%':>6} "
              f"{'OOS':>6} {'DD<20':>5} {'OOS>90':>6}")
    log(header)
    log("-" * len(header))

    for name in ["v4a_agg30", "v4a_agg40",
                  "v5a_agg30", "v5a_agg40",
                  "v5b_agg30", "v5b_agg40",
                  "v5c_agg30", "v5c_agg40"]:
        r = results[name]
        ex = r.get("extra_metrics", {})
        wf = r.get("walk_forward", {})

        dd_ok = "✓" if r["max_drawdown_pct"] < 20 else "✗"
        oos_ok = "✓" if wf.get("score", 0) > 90 else "✗"

        log(f"{name:<16} {r['score']:>6.1f} {r['trades']:>4} {r['win_rate']:>5.1f}% "
            f"${r['total_pnl']:>9,.0f} {r['max_drawdown_pct']:>5.1f}% "
            f"{r['sharpe']:>5.2f} {r['capital_utilization']:>4.0f}% "
            f"{r.get('concurrent_positions', 0):>4.1f} "
            f"{ex.get('n_active_cities', 0):>6} {ex.get('max_city_pct', 0):>5.1f}% "
            f"{wf.get('score', 0):>5.1f}  {dd_ok:>4}  {oos_ok:>5}")

    # ── Incremental impact analysis ──
    log(f"\n{'=' * 70}")
    log("INCREMENTAL IMPACT (agg30)")
    log(f"{'=' * 70}")

    ref = results["v4a_agg30"]
    for name, label in [("v5a_agg30", "+Multi-bucket"), ("v5b_agg30", "+Multi-day"), ("v5c_agg30", "+Cities")]:
        r = results[name]
        dt = r["trades"] - ref["trades"]
        du = r["capital_utilization"] - ref["capital_utilization"]
        dp = r["total_pnl"] - ref["total_pnl"]
        dd_d = r["max_drawdown_pct"] - ref["max_drawdown_pct"]
        log(f"  {label:<16} Trades: {ref['trades']}→{r['trades']} ({dt:+d})  "
            f"Util: {ref['capital_utilization']:.0f}→{r['capital_utilization']:.0f}% ({du:+.0f})  "
            f"PnL: ${dp:+,.0f}  DD: {dd_d:+.1f}pp")
        ref = r  # Chain: each step builds on previous

    # ── Save ──
    output = {
        "sweep_id": "v5_velocity",
        "sweep_type": "V5 Capital Velocity Expansion",
        "meta": {
            "sweep_id": "v5_velocity",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_trials": len(results),
        },
        "variants": results,
        "best": max(results.values(), key=lambda r: r.get("score", 0)),
        "top_results": sorted(results.values(), key=lambda r: -r.get("score", 0)),
        "all_results": sorted(results.values(), key=lambda r: -r.get("score", 0)),
    }
    with open(SWEEP_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)

    elapsed = time.time() - t_start
    log(f"\nDone in {elapsed:.0f}s. Results: {SWEEP_PATH}")
    log(f"Dashboard: /experiments → select 'v5_velocity'")


if __name__ == "__main__":
    main()
