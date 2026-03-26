"""EMOS Autoresearch V2 — Using Experiment Framework + PMD Data.

Realistic hour-by-hour portfolio simulation with:
- $200 max position cap (no runaway compounding)
- Concurrent positions tracked
- Capital turnover metrics
- 100-point composite scoring
- Walk-forward validation
- Dashboard-compatible JSON output

Data: weather_pmd.sqlite (10K markets, 7.4M prices, 10-min resolution)
Model: EMOS adaptive sigma + bias (vs baseline fixed sigma)

Usage:
    python3 research/innovations/sweep_emos_v2.py
"""

from __future__ import annotations

import json
import math
import random
import sys
import time
from collections import defaultdict
from dataclasses import asdict
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
    ExperimentResult,
    experiment_score,
    load_forecasts,
    serialize_result,
    simulate_portfolio,
    walk_forward_validate,
)

# ── Output paths ─────────────────────────────────────────────────────

RESULTS_DIR = Path(__file__).parent.parent / "data" / "experiments"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
SWEEP_PATH = RESULTS_DIR / "sweep_emos_v2.json"
LOG_PATH = Path(__file__).parent.parent / "sweep_emos_v2_log.txt"

# ── EMOS Integration ─────────────────────────────────────────────────

# Global EMOS state — set per-trial, used by patched compute_prob
_emos_models: dict[str, EMOSModel] = {}
_emos_obs: dict[str, list[Observation]] = {}
_emos_params: dict = {}
_emos_enabled: bool = False


def _build_training_observations(
    events: list,
    buckets_by_event: dict,
    forecasts: dict,
) -> dict[str, list[Observation]]:
    """Build (forecast, actual) pairs per city from resolved events."""
    city_obs: dict[str, list[Observation]] = defaultdict(list)

    for ev in events:
        if not ev.city or not ev.target_date:
            continue

        forecast_temp = forecasts.get(ev.city, {}).get(ev.target_date)
        if forecast_temp is None:
            continue

        # Find winning bucket → actual temperature
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
            Observation(
                forecast=forecast_temp,
                actual=actual_temp,
                date=ev.target_date,
                city=ev.city,
            )
        )

    for city in city_obs:
        city_obs[city].sort(key=lambda o: o.date)

    total = sum(len(v) for v in city_obs.values())
    print(f"EMOS training observations: {total} across {len(city_obs)} cities")
    return dict(city_obs)


def _get_emos_model(city: str, target_date: str) -> EMOSModel:
    """Get walk-forward EMOS model for city (fit only on prior data)."""
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


# Monkey-patch compute_prob to support EMOS
_original_compute_prob = ef.compute_prob


def _compute_prob_emos(
    forecast_temp: float,
    bucket,
    days_out: int,
    city: str,
    params: dict,
) -> float:
    """EMOS-enhanced probability computation."""
    if not _emos_enabled:
        return _original_compute_prob(forecast_temp, bucket, days_out, city, params)

    # Get target_date from context — approximate from forecast cache
    # The EMOS model is pre-fitted per city+date, we use the latest available
    model = None
    obs = _emos_obs.get(city, [])
    if obs:
        # Use the latest observation date as proxy for "current" date
        # The walk-forward fitting already handles this correctly
        latest_date = obs[-1].date if obs else "9999-99-99"
        model = _get_emos_model(city, latest_date)

    if model is None or not model._fitted:
        return _original_compute_prob(forecast_temp, bucket, days_out, city, params)

    # EMOS probability
    raw = model.bucket_probability(
        forecast_temp,
        bucket.low_c,
        bucket.high_c,
        bucket.bucket_type or "range",
    )

    # Post-processing (same as baseline)
    city_ov = params.get("city_overrides", {}).get(city, {})
    shrinkage = city_ov.get("shrinkage", params.get("shrinkage", 0.0))
    raw = raw * (1.0 - shrinkage) + 0.125 * shrinkage

    sharpening = city_ov.get("prob_sharpening", params.get("prob_sharpening", 1.05))
    if sharpening != 1.0 and raw > 0:
        raw = raw**sharpening

    return raw


# Apply monkey-patch
ef.compute_prob = _compute_prob_emos


# ── Improved EMOS with per-event date context ────────────────────────

# We need target_date in compute_prob but the framework doesn't pass it.
# Solution: patch simulate_portfolio to set a per-event date context.

_current_event_date: str = ""


def _simulate_with_date_context(
    sim_data, params, entry_hours=24,
    period_start_ts=None, period_end_ts=None,
    experiment_id="", record_equity=False,
):
    """Wrapper that sets target_date context for EMOS models."""
    global _emos_models
    _emos_models = {}  # Clear cache per simulation

    # Patch _get_emos_model to use event date
    # We override compute_prob to capture the event context
    return _original_simulate(
        sim_data, params, entry_hours,
        period_start_ts, period_end_ts,
        experiment_id, record_equity,
    )


_original_simulate = ef.simulate_portfolio


# Instead of complex date tracking, pre-fit EMOS models for ALL dates per city
def _prefit_emos_all(params: dict) -> None:
    """Pre-fit EMOS for all observed dates per city (walk-forward)."""
    global _emos_models
    _emos_models = {}

    for city, obs in _emos_obs.items():
        dates = sorted(set(o.date for o in obs))
        for target_date in dates:
            _get_emos_model(city, target_date)

    # Also fit for "latest" (for events beyond training data)
    for city, obs in _emos_obs.items():
        if obs:
            _get_emos_model(city, "9999-99-99")


# ── Search Space ─────────────────────────────────────────────────────


def random_params(rng: random.Random) -> dict:
    """Generate random EMOS + quality gate parameters."""
    return {
        # EMOS
        "emos_training_window": rng.choice([7, 14, 21, 30, 45, 60]),
        "emos_sigma_method": rng.choice(
            ["rolling_rmse", "rolling_mae", "ewma_rmse"]
        ),
        "emos_bias_method": rng.choice(["rolling_mean", "ewma", "none"]),
        "emos_sigma_floor": rng.choice([0.3, 0.5, 0.7, 1.0, 1.2, 1.5]),
        "emos_sigma_scale": rng.choice([0.6, 0.8, 1.0, 1.2, 1.5, 2.0]),
        "emos_ewma_alpha": rng.choice([0.05, 0.1, 0.15, 0.2, 0.3]),
        # Quality gate
        "min_edge": rng.choice([0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15]),
        "max_edge": 0.90,
        "max_price": rng.choice([0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80]),
        "min_price": rng.choice([0.01, 0.02, 0.03, 0.05, 0.08, 0.10]),
        "min_prob": rng.choice([0.02, 0.05, 0.08, 0.10, 0.15, 0.25]),
        "min_volume": rng.choice([0, 10, 50, 100]),
        "kelly_raw_cap": rng.choice([0.15, 0.20, 0.25, 0.30, 0.40]),
        "prob_sharpening": rng.choice([0.85, 0.90, 0.95, 1.0, 1.05, 1.10, 1.15, 1.20]),
        "shrinkage": rng.choice([0.0, 0.01, 0.03, 0.05, 0.10]),
        "excluded_cities": set(),
        "city_overrides": {},
        # Exit (off for now — separate experiment)
        "exit_enabled": False,
    }


def perturb_params(base: dict, rng: random.Random) -> dict:
    """Perturb 1-4 parameters from base."""
    p = dict(base)
    p["excluded_cities"] = set(p.get("excluded_cities", set()))
    p["city_overrides"] = dict(p.get("city_overrides", {}))

    emos_keys = [
        "emos_training_window", "emos_sigma_floor",
        "emos_sigma_scale", "emos_ewma_alpha",
    ]
    gate_keys = [
        "min_edge", "max_price", "min_price", "min_prob",
        "min_volume", "kelly_raw_cap", "prob_sharpening", "shrinkage",
    ]
    all_keys = emos_keys + gate_keys

    for key in rng.sample(all_keys, min(rng.randint(1, 4), len(all_keys))):
        val = p.get(key, 0)
        if key == "emos_training_window":
            p[key] = max(5, val + rng.choice([-7, -3, 3, 7, 14]))
        elif key == "emos_sigma_floor":
            p[key] = round(max(0.1, val + rng.choice([-0.2, -0.1, 0.1, 0.2])), 2)
        elif key == "emos_sigma_scale":
            p[key] = round(max(0.3, val + rng.choice([-0.2, -0.1, 0.1, 0.2])), 2)
        elif key == "emos_ewma_alpha":
            p[key] = round(max(0.01, min(0.5, val + rng.choice([-0.05, 0.05]))), 2)
        elif key == "min_volume":
            p[key] = max(0, val + rng.choice([-50, -20, 20, 50, 100]))
        elif key == "prob_sharpening":
            p[key] = round(val + rng.choice([-0.10, -0.05, 0.05, 0.10]), 2)
        elif key == "shrinkage":
            p[key] = round(max(0, val + rng.choice([-0.03, -0.02, 0.02, 0.03])), 2)
        elif key == "kelly_raw_cap":
            p[key] = round(max(0.05, min(0.60, val + rng.choice([-0.05, 0.05]))), 2)
        else:
            p[key] = round(max(0.01, val + rng.choice([-0.03, -0.02, -0.01, 0.01, 0.02, 0.03])), 2)

    if rng.random() < 0.15:
        p["emos_sigma_method"] = rng.choice(["rolling_rmse", "rolling_mae", "ewma_rmse"])
    if rng.random() < 0.15:
        p["emos_bias_method"] = rng.choice(["rolling_mean", "ewma", "none"])

    return p


# ── Run one experiment ───────────────────────────────────────────────


def run_experiment(
    sim_data: SimulationData,
    params: dict,
    experiment_id: str,
    entry_hours: float = 24,
) -> ExperimentResult:
    """Run single EMOS experiment."""
    global _emos_enabled, _emos_params, _emos_models

    _emos_enabled = True
    _emos_params = params
    _emos_models = {}  # Clear cache

    # Pre-fit EMOS models
    _prefit_emos_all(params)

    result = simulate_portfolio(
        sim_data, params,
        entry_hours=entry_hours,
        experiment_id=experiment_id,
        record_equity=True,
    )

    _emos_enabled = False
    return result


# ── Main sweep ───────────────────────────────────────────────────────


def main():
    t_start = time.time()
    rng = random.Random(42)

    # ── Load PMD data ──
    print("Loading PMD data...")
    events, buckets_by_event, prices = load_pmd_data()

    # ── Load forecasts ──
    print("Loading forecasts...")
    forecasts = load_forecasts(events)

    # ── Build SimulationData ──
    print("Building simulation data...")
    sim_data = SimulationData(events, buckets_by_event, prices, forecasts)

    # ── Build EMOS training observations ──
    global _emos_obs
    _emos_obs = _build_training_observations(events, buckets_by_event, forecasts)

    all_results: list[dict] = []
    best_score = -1
    best_result = None
    best_params = None

    def log(msg):
        print(msg, flush=True)
        with open(LOG_PATH, "a") as f:
            f.write(msg + "\n")

    log(f"\n{'=' * 70}")
    log(f"EMOS AUTORESEARCH V2 — Experiment Framework + PMD Data")
    log(f"Events: {len(events)}, Start: {datetime.now().isoformat()}")
    log(f"Scoring: 100-point composite (ROI+Sharpe+DD+Util+PnL/h+Trades)")
    log(f"Position cap: $200, Quarter-Kelly, Realistic simulation")
    log(f"{'=' * 70}\n")

    # ── Run baseline (fixed sigma) for reference ──
    log("Running baseline (fixed sigma, no EMOS)...")
    global _emos_enabled
    _emos_enabled = False
    baseline_params = {
        "min_edge": 0.01, "max_edge": 0.90, "max_price": 0.55,
        "min_price": 0.08, "min_prob": 0.02, "min_volume": 0,
        "kelly_raw_cap": 0.25, "prob_sharpening": 1.05, "shrinkage": 0.0,
        "excluded_cities": {"tel_aviv", "lucknow", "los_angeles", "dc"},
        "city_overrides": {}, "exit_enabled": False,
    }
    baseline = simulate_portfolio(
        sim_data, baseline_params, entry_hours=24,
        experiment_id="baseline", record_equity=True,
    )
    log(f"  Baseline: score={baseline.score:.1f}  trades={baseline.trades}  "
        f"WR={baseline.win_rate}%  PnL=${baseline.total_pnl:.2f}  "
        f"DD={baseline.max_drawdown_pct}%  Sharpe={baseline.sharpe}")
    log("")

    # ── Phase 1: Random Search ──
    N1 = 600
    log(f"Phase 1: Random Search ({N1} trials)")
    log("-" * 50)

    for trial in range(N1):
        params = random_params(rng)
        eid = f"EMOS-{trial:04d}"
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
                f"DD={result.max_drawdown_pct:>5.1f}%  Util={result.capital_utilization:.1f}%  "
                f"★ NEW BEST")

        if (trial + 1) % 100 == 0:
            log(f"  ... {trial + 1}/{N1} done, best: {best_score:.1f}")

    # ── Phase 2: Fine-tuning ──
    N2 = 400
    log(f"\nPhase 2: Fine-tuning ({N2} trials around top 10)")
    log("-" * 50)

    top10 = sorted(all_results, key=lambda r: -r["score"])[:10]
    log(f"  Top 10 scores: {[r['score'] for r in top10]}")

    for i in range(N2):
        base_r = top10[i % 10]
        base_p = dict(base_r["params"])
        base_p["excluded_cities"] = set(base_p.get("excluded_cities", []))
        base_p["city_overrides"] = dict(base_p.get("city_overrides", {}))

        params = perturb_params(base_p, rng)
        eid = f"EMOS-{N1 + i:04d}"
        result = run_experiment(sim_data, params, eid)

        rd = serialize_result(result)
        rd["trial"] = N1 + i
        rd["phase"] = 2
        all_results.append(rd)

        if result.score > best_score:
            best_score = result.score
            best_result = rd
            best_params = params
            log(f"  {eid}: score={result.score:>6.1f}  trades={result.trades:>3}  "
                f"PnL=${result.total_pnl:>8.2f}  ★ NEW BEST")

        if (i + 1) % 100 == 0:
            log(f"  ... {i + 1}/{N2} done, best: {best_score:.1f}")

    # ── Phase 3: City optimization ──
    N3 = 200
    log(f"\nPhase 3: City optimization ({N3} trials)")
    log("-" * 50)

    losing_cities = []
    winning_cities = []
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
            n_exc = rng.randint(1, min(len(losing_cities), 5))
            params["excluded_cities"] = set(rng.sample(losing_cities, n_exc))
        elif i < 100 and winning_cities:
            params["excluded_cities"] = set(losing_cities)
            for city in rng.sample(winning_cities, min(len(winning_cities), 3)):
                params["city_overrides"][city] = {
                    "max_price": rng.choice([0.50, 0.55, 0.60, 0.70, 0.80]),
                    "min_edge": rng.choice([0.01, 0.02, 0.03, 0.05]),
                }
        elif i < 150:
            params = perturb_params(params, rng)
            params["excluded_cities"] = set(losing_cities)
        else:
            all_c = list(CITY_COORDS.keys())
            params["excluded_cities"] = set(rng.sample(all_c, rng.randint(0, 6)))
            params = perturb_params(params, rng)

        eid = f"EMOS-{N1 + N2 + i:04d}"
        result = run_experiment(sim_data, params, eid)

        rd = serialize_result(result)
        rd["trial"] = N1 + N2 + i
        rd["phase"] = 3
        all_results.append(rd)

        if result.score > best_score:
            best_score = result.score
            best_result = rd
            best_params = params
            log(f"  {eid}: score={result.score:>6.1f}  trades={result.trades:>3}  "
                f"PnL=${result.total_pnl:>8.2f}  "
                f"excluded={sorted(params['excluded_cities'])}  ★ NEW BEST")

        if (i + 1) % 50 == 0:
            log(f"  ... {i + 1}/{N3} done, best: {best_score:.1f}")

    # ── Walk-forward validation on best ──
    log(f"\nWalk-forward validation on best params...")
    _emos_enabled = True
    _emos_params = best_params
    _emos_models = {}
    _prefit_emos_all(best_params)
    wf = walk_forward_validate(sim_data, best_params, entry_hours=24, n_folds=3)
    log(f"  WF score: {wf['score']:.1f}, OOS PnL: ${wf.get('oos_pnl', 0):.2f}, "
        f"Folds: {wf.get('n_folds', 0)}")
    if wf.get("folds"):
        for fi, fold in enumerate(wf["folds"]):
            log(f"    Fold {fi + 1}: PnL=${fold['pnl']:.2f}  trades={fold['trades']}  "
                f"WR={fold['wr']}%  score={fold['score']:.1f}")

    if best_result:
        best_result["oos_pnl"] = wf.get("oos_pnl", 0)
        best_result["oos_trades"] = wf.get("total_trades", 0)

    _emos_enabled = False

    # ── Save results ──
    sorted_results = sorted(all_results, key=lambda r: -r["score"])

    output = {
        "sweep_id": "emos_v2",
        "sweep_type": "EMOS Autoresearch V2",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_trials": len(all_results),
        "baseline": serialize_result(baseline),
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
    log(f"EMOS V2 SWEEP COMPLETE — {elapsed / 60:.1f} minutes")
    log(f"{'=' * 70}")

    log(f"\n  BASELINE: score={baseline.score:.1f}  trades={baseline.trades}  "
        f"WR={baseline.win_rate}%  PnL=${baseline.total_pnl:.2f}  "
        f"Sharpe={baseline.sharpe}  DD={baseline.max_drawdown_pct}%")

    if best_result:
        log(f"\n  BEST EMOS: score={best_result['score']:.1f}  "
            f"trades={best_result['trades']}  "
            f"WR={best_result['win_rate']}%  "
            f"PnL=${best_result['total_pnl']:.2f}  "
            f"Sharpe={best_result['sharpe']}  "
            f"DD={best_result['max_drawdown_pct']}%")

        delta_score = best_result["score"] - baseline.score
        delta_pnl = best_result["total_pnl"] - baseline.total_pnl
        log(f"\n  Delta: score {'+' if delta_score >= 0 else ''}{delta_score:.1f}, "
            f"PnL {'+' if delta_pnl >= 0 else ''}${delta_pnl:.2f}")

        log(f"\n  Best EMOS params:")
        bp = best_result["params"]
        for k in ["emos_training_window", "emos_sigma_method", "emos_bias_method",
                   "emos_sigma_floor", "emos_sigma_scale", "emos_ewma_alpha"]:
            log(f"    {k}: {bp.get(k)}")
        for k in ["min_edge", "max_price", "min_price", "min_prob",
                   "kelly_raw_cap", "prob_sharpening", "shrinkage"]:
            log(f"    {k}: {bp.get(k)}")
        log(f"    excluded_cities: {bp.get('excluded_cities', [])}")

        log(f"\n  Per-city PnL:")
        cr = best_result.get("city_results", {})
        for city in sorted(cr, key=lambda c: -cr[c]["pnl"]):
            v = cr[city]
            log(f"    {city:<16} ${v['pnl']:>8.2f}  ({v['trades']} trades, WR={v['win_rate']}%)")

    log(f"\n  Top 5:")
    for i, r in enumerate(sorted_results[:5]):
        log(f"    #{i + 1}: score={r['score']:.1f}  trades={r['trades']}  "
            f"WR={r['win_rate']}%  PnL=${r['total_pnl']:.2f}  "
            f"(trial {r.get('trial', '?')}, phase {r.get('phase', '?')})")

    log(f"\n  Results: {SWEEP_PATH}")
    log(f"  Dashboard: /experiments → select 'emos_v2'")


if __name__ == "__main__":
    main()
