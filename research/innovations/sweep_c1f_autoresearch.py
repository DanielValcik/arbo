"""C1f Autoresearch — Optimize ensemble EMOS parameters.

Searches over:
- Ensemble model params: training_window, sigma_floor, min_n_threshold
- Quality gate: min_edge, max_price, min_price, kelly_raw_cap, etc.
- City exclusions and overrides

Base: C1f (ensemble EMOS + EMOS-lite fallback for low-n cities)
Output: Dashboard-compatible sweep JSON for /experiments

Usage:
    python3 research/innovations/sweep_c1f_autoresearch.py
    python3 research/innovations/sweep_c1f_autoresearch.py --trials 200
"""

from __future__ import annotations

import json
import random
import sqlite3
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from emos_model import EMOSEnsembleModel, EMOSModel, Observation, actual_temp_from_bucket
from pmd_loader import load_pmd_data

import experiment_framework as ef
from experiment_framework import (
    SimulationData, ExperimentResult,
    experiment_score, load_forecasts, serialize_result,
    simulate_portfolio, walk_forward_validate,
)
from sweep_emos_v2 import (
    _build_training_observations, _prefit_emos_all,
)
import sweep_emos_v2 as emos_v2

# ── Paths ────────────────────────────────────────────────────────────

RESULTS_DIR = Path(__file__).parent.parent / "data" / "experiments"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
SWEEP_PATH = RESULTS_DIR / "sweep_c1f_autoresearch.json"
LOG_PATH = Path(__file__).parent.parent / "sweep_c1f_autoresearch_log.txt"
GEFS_DB = Path(__file__).parent.parent / "data" / "gefs_ensemble.sqlite"

# ── Ensemble data (loaded once) ──────────────────────────────────────

_ensemble_stats: dict[str, dict[str, float]] = {}
_city_mean_std: dict[str, float] = {}
_ensemble_models: dict[str, EMOSEnsembleModel] = {}
_current_event_date: str | None = None
_event_dates: dict[str, str] = {}
_bucket_event: dict[str, str] = {}

# ── Mode control ─────────────────────────────────────────────────────

_c1f_enabled: bool = False
_c1f_params: dict = {}
_min_n_threshold: int = 10

_original_compute_prob = ef.compute_prob


def _get_ensemble_std(city: str, target_date: str | None = None) -> float | None:
    """Get ensemble_std for city+date, falling back to nearest prior date."""
    city_stds = _ensemble_stats.get(city, {})
    if not city_stds:
        return None
    if target_date and target_date in city_stds:
        return city_stds[target_date]
    if target_date:
        prior = [d for d in sorted(city_stds.keys()) if d <= target_date]
        if prior:
            return city_stds[prior[-1]]
    return None


def _compute_prob_c1f(forecast_temp, bucket, days_out, city, params):
    """C1f: Ensemble EMOS for high-n cities, EMOS-lite fallback for rest."""
    if not _c1f_enabled:
        return _original_compute_prob(forecast_temp, bucket, days_out, city, params)

    # Resolve event date from bucket
    global _current_event_date
    event_id = _bucket_event.get(bucket.token_id)
    _current_event_date = _event_dates.get(event_id) if event_id else None

    # Try ensemble model
    model = _ensemble_models.get(city)
    if model and model._fitted and model.n_observations >= _min_n_threshold:
        ens_std = _get_ensemble_std(city, _current_event_date)
        if ens_std is not None:
            raw = model.bucket_probability(
                forecast_temp,
                bucket.low_c, bucket.high_c,
                bucket.bucket_type or "range",
                ensemble_std=ens_std,
            )

            # Post-processing
            shrinkage = ef._get_blended_city_param("shrinkage", city, params)
            raw = raw * (1.0 - shrinkage) + 0.125 * shrinkage
            sharpening = ef._get_blended_city_param("prob_sharpening", city, params)
            if sharpening != 1.0 and raw > 0:
                raw = raw ** sharpening
            return raw

    # Fallback to EMOS-lite
    emos_v2._emos_enabled = True
    emos_v2._emos_params = params
    result = _original_compute_prob(forecast_temp, bucket, days_out, city, params)
    emos_v2._emos_enabled = False
    return result


def _fit_ensemble_models(params: dict) -> None:
    """Fit ensemble EMOS models per city with given params."""
    global _ensemble_models
    _ensemble_models = {}

    for city, obs_list in emos_v2._emos_obs.items():
        city_stds = _ensemble_stats.get(city, {})
        if not city_stds:
            continue

        model = EMOSEnsembleModel(
            training_window=params.get("emos_training_window", 66),
            sigma_floor=params.get("emos_sigma_floor", 0.6),
            bias_method=params.get("emos_bias_method", "rolling_mean"),
            ewma_alpha=params.get("emos_ewma_alpha", 0.3),
        )
        model.fit(obs_list, city_stds)
        _ensemble_models[city] = model


def run_c1f_experiment(
    sim_data: SimulationData,
    params: dict,
    experiment_id: str,
) -> ExperimentResult:
    """Run one C1f experiment."""
    global _c1f_enabled, _c1f_params, _min_n_threshold

    # Fit ensemble models with these params
    _fit_ensemble_models(params)
    _min_n_threshold = params.get("ensemble_min_n", 10)

    # Also pre-fit EMOS-lite for fallback
    emos_v2._emos_obs = emos_v2._emos_obs  # already set
    emos_v2._emos_params = params
    emos_v2._emos_models = {}
    _prefit_emos_all(params)

    # Enable C1f mode
    _c1f_enabled = True
    _c1f_params = params
    ef.compute_prob = _compute_prob_c1f

    result = simulate_portfolio(
        sim_data, params,
        entry_hours=24,
        experiment_id=experiment_id,
        record_equity=True,
    )

    _c1f_enabled = False
    ef.compute_prob = _original_compute_prob
    return result


# ── Search Space ─────────────────────────────────────────────────────

# V4a agg30 as base
V4A_BASE = {
    "emos_training_window": 66, "emos_sigma_method": "rolling_rmse",
    "emos_bias_method": "rolling_mean", "emos_sigma_floor": 0.6,
    "emos_sigma_scale": 1.1, "emos_ewma_alpha": 0.3,
    "min_edge": 0.10, "max_edge": 0.90, "max_price": 0.56,
    "min_price": 0.08, "min_prob": 0.07, "min_volume": 0,
    "kelly_raw_cap": 0.40, "prob_sharpening": 1.10, "shrinkage": 0.03,
    "max_aggregate_pct": 0.30, "city_max_exposure": 0.35,
    "shrinkage_pseudocount": 20,
    "excluded_cities": {"lucknow", "wellington"},
    "city_overrides": {
        "munich": {"min_edge": 0.05, "max_price": 0.80, "kelly_raw_cap": 0.50},
        "seoul": {"min_edge": 0.05, "max_price": 0.70, "kelly_raw_cap": 0.45},
        "atlanta": {"min_edge": 0.08, "max_price": 0.70, "kelly_raw_cap": 0.40},
        "nyc": {"min_edge": 0.08, "max_price": 0.70},
        "toronto": {"min_edge": 0.08, "max_price": 0.60},
    },
    "exit_enabled": False,
    "ensemble_min_n": 10,
}


def random_params(rng: random.Random) -> dict:
    """Generate random C1f parameters."""
    excluded = set()
    for city in ["lucknow", "wellington", "chicago", "miami", "sao_paulo"]:
        if rng.random() < 0.3:
            excluded.add(city)
    # Always exclude lucknow + wellington (no forecast data)
    excluded.add("lucknow")
    excluded.add("wellington")

    return {
        "emos_training_window": rng.choice([30, 45, 60, 66, 75, 90]),
        "emos_sigma_method": rng.choice(["rolling_rmse", "rolling_mae"]),
        "emos_bias_method": rng.choice(["rolling_mean", "ewma"]),
        "emos_sigma_floor": rng.choice([0.4, 0.5, 0.6, 0.7, 0.8, 1.0]),
        "emos_sigma_scale": rng.choice([0.8, 0.9, 1.0, 1.1, 1.2, 1.5]),
        "emos_ewma_alpha": rng.choice([0.1, 0.2, 0.3, 0.4]),
        "min_edge": rng.choice([0.05, 0.08, 0.10, 0.12, 0.15]),
        "max_edge": 0.90,
        "max_price": rng.choice([0.45, 0.50, 0.56, 0.60, 0.65, 0.70]),
        "min_price": rng.choice([0.05, 0.08, 0.10]),
        "min_prob": rng.choice([0.05, 0.07, 0.10]),
        "min_volume": 0,
        "kelly_raw_cap": rng.choice([0.25, 0.30, 0.35, 0.40, 0.45, 0.50]),
        "prob_sharpening": rng.choice([0.95, 1.00, 1.05, 1.10, 1.15, 1.20]),
        "shrinkage": rng.choice([0.01, 0.03, 0.05]),
        "max_aggregate_pct": rng.choice([0.25, 0.30, 0.35]),
        "city_max_exposure": rng.choice([0.30, 0.35, 0.40]),
        "shrinkage_pseudocount": rng.choice([10, 15, 20, 25]),
        "excluded_cities": excluded,
        "city_overrides": _random_city_overrides(rng),
        "exit_enabled": False,
        "ensemble_min_n": rng.choice([5, 10, 15, 20]),
    }


def _random_city_overrides(rng: random.Random) -> dict:
    """Generate random per-city overrides."""
    overrides = {}
    # Cities that benefit from overrides (based on V4a research)
    candidate_cities = ["munich", "seoul", "atlanta", "nyc", "toronto", "dallas"]
    for city in candidate_cities:
        if rng.random() < 0.5:
            ov = {}
            if rng.random() < 0.6:
                ov["min_edge"] = rng.choice([0.03, 0.05, 0.08, 0.10])
            if rng.random() < 0.6:
                ov["max_price"] = rng.choice([0.55, 0.60, 0.65, 0.70, 0.80])
            if rng.random() < 0.4:
                ov["kelly_raw_cap"] = rng.choice([0.30, 0.40, 0.45, 0.50])
            if ov:
                overrides[city] = ov
    return overrides


def perturb_params(base: dict, rng: random.Random) -> dict:
    """Perturb 1-3 parameters from base."""
    p = dict(base)
    p["excluded_cities"] = set(p.get("excluded_cities", set()))
    p["city_overrides"] = dict(p.get("city_overrides", {}))

    keys = [
        "emos_training_window", "emos_sigma_floor", "emos_sigma_scale",
        "emos_ewma_alpha", "min_edge", "max_price", "min_price",
        "kelly_raw_cap", "prob_sharpening", "shrinkage", "ensemble_min_n",
        "city_max_exposure", "max_aggregate_pct", "shrinkage_pseudocount",
    ]

    for key in rng.sample(keys, min(rng.randint(1, 3), len(keys))):
        val = p.get(key, 0)
        if key == "emos_training_window":
            p[key] = max(14, val + rng.choice([-14, -7, 7, 14]))
        elif key == "ensemble_min_n":
            p[key] = max(3, val + rng.choice([-5, -3, 3, 5]))
        elif key == "shrinkage_pseudocount":
            p[key] = max(3, val + rng.choice([-5, -3, 3, 5]))
        elif key in ("emos_sigma_floor", "emos_sigma_scale"):
            p[key] = round(max(0.2, val + rng.choice([-0.2, -0.1, 0.1, 0.2])), 2)
        elif key == "emos_ewma_alpha":
            p[key] = round(max(0.05, min(0.5, val + rng.choice([-0.1, 0.1]))), 2)
        elif key in ("kelly_raw_cap", "city_max_exposure", "max_aggregate_pct"):
            p[key] = round(max(0.10, min(0.60, val + rng.choice([-0.05, 0.05]))), 2)
        elif key == "prob_sharpening":
            p[key] = round(val + rng.choice([-0.05, 0.05]), 2)
        elif key == "shrinkage":
            p[key] = round(max(0, min(0.10, val + rng.choice([-0.02, 0.02]))), 2)
        else:
            p[key] = round(max(0.01, val + rng.choice([-0.03, -0.02, 0.02, 0.03])), 2)

    # Occasionally perturb city overrides
    if rng.random() < 0.2:
        p["city_overrides"] = _random_city_overrides(rng)

    # Occasionally flip excluded cities
    if rng.random() < 0.1:
        toggleable = ["chicago", "miami", "sao_paulo"]
        c = rng.choice(toggleable)
        if c in p["excluded_cities"]:
            p["excluded_cities"].discard(c)
        else:
            p["excluded_cities"].add(c)

    return p


# ── Main ─────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    t_start = time.time()
    rng = random.Random(args.seed)

    # Clear log
    LOG_PATH.write_text("")

    def log(msg):
        print(msg, flush=True)
        with open(LOG_PATH, "a") as f:
            f.write(msg + "\n")

    # Load data
    log("Loading data...")
    events, buckets, prices = load_pmd_data()
    forecasts = load_forecasts(events)
    sim_data = SimulationData(events, buckets, prices, forecasts)

    # Build EMOS training observations
    emos_v2._emos_obs = _build_training_observations(events, buckets, forecasts)

    # Load ensemble stats
    global _ensemble_stats, _city_mean_std, _event_dates, _bucket_event
    _ensemble_stats = load_ensemble_stats()
    _city_mean_std = compute_city_mean_std(_ensemble_stats)

    # Build event/bucket lookups
    for ev in events:
        _event_dates[ev.event_id] = ev.target_date
    for ev in events:
        for b in buckets.get(ev.event_id, []):
            _bucket_event[b.token_id] = ev.event_id

    log(f"Data ready: {len(events)} events, {len(_ensemble_stats)} cities with ensemble")

    # ── Autoresearch loop ──
    all_results: list[dict] = []
    top_results: list[dict] = []
    best_score = 0.0
    best_params = V4A_BASE
    gen1_end = args.trials // 3  # Random exploration
    gen2_end = 2 * args.trials // 3  # Exploitation

    log(f"\n{'=' * 70}")
    log(f"C1f Autoresearch — {args.trials} trials")
    log(f"  Gen 1 (random): trials 1-{gen1_end}")
    log(f"  Gen 2 (perturb best): trials {gen1_end+1}-{gen2_end}")
    log(f"  Gen 3 (fine-tune top 3): trials {gen2_end+1}-{args.trials}")
    log(f"{'=' * 70}")

    for trial in range(1, args.trials + 1):
        # Parameter generation
        if trial <= gen1_end:
            # Gen 1: Random exploration
            if trial == 1:
                params = V4A_BASE  # Always test baseline first
            else:
                params = random_params(rng)
        elif trial <= gen2_end:
            # Gen 2: Perturb from best
            params = perturb_params(best_params, rng)
        else:
            # Gen 3: Fine-tune from top 3
            if len(top_results) >= 3:
                base = rng.choice(top_results[:3])["params"]
            else:
                base = best_params
            params = perturb_params(base, rng)

        eid = f"c1f_{trial:04d}"

        try:
            result = run_c1f_experiment(sim_data, params, eid)
            rd = serialize_result(result)
            rd["params"] = _serialize_params(params)
            all_results.append(rd)

            score = rd.get("score", 0)

            # Track best
            if score > best_score:
                best_score = score
                best_params = dict(params)

            # Update top results (sorted by score)
            top_results.append(rd)
            top_results.sort(key=lambda r: -r.get("score", 0))
            top_results = top_results[:20]

            gen = "G1" if trial <= gen1_end else "G2" if trial <= gen2_end else "G3"
            trades = rd.get("trades", 0)
            pnl = rd.get("total_pnl", 0)
            wr = rd.get("win_rate", 0)
            dd = rd.get("max_drawdown_pct", 0)

            is_best = " ★" if score == best_score and trial > 1 else ""
            log(f"  [{gen}] {trial:>3}/{args.trials} score={score:>6.1f} "
                f"trades={trades:>3} WR={wr:>4.1f}% PnL=${pnl:>9,.0f} "
                f"DD={dd:>4.1f}%{is_best}")

        except Exception as e:
            log(f"  [{trial:>3}] ERROR: {e}")

        # Periodic save
        if trial % 25 == 0 or trial == args.trials:
            _save_results(all_results, top_results, trial, args.trials, t_start)

    # ── Walk-forward validation of top 5 ──
    log(f"\n{'=' * 70}")
    log("Walk-Forward Validation — Top 5")
    log(f"{'=' * 70}")

    for i, rd in enumerate(top_results[:5]):
        params = _deserialize_params(rd["params"])
        _fit_ensemble_models(params)
        _min_n = params.get("ensemble_min_n", 10)

        global _c1f_enabled, _min_n_threshold
        _c1f_enabled = True
        _min_n_threshold = _min_n

        emos_v2._emos_params = params
        emos_v2._emos_models = {}
        _prefit_emos_all(params)

        ef.compute_prob = _compute_prob_c1f
        wf = walk_forward_validate(sim_data, params, entry_hours=24, n_folds=3)
        ef.compute_prob = _original_compute_prob
        _c1f_enabled = False

        rd["walk_forward"] = wf
        oos = wf.get("score", 0)
        oos_pnl = wf.get("oos_pnl", 0)

        log(f"  #{i+1} (IS={rd['score']:.1f}) → OOS={oos:.1f} PnL=${oos_pnl:.0f}")
        if wf.get("folds"):
            for fi, f in enumerate(wf["folds"]):
                log(f"    Fold {fi+1}: PnL=${f['pnl']:.0f} trades={f['trades']} WR={f['wr']:.0f}%")

    # Final save
    _save_results(all_results, top_results, args.trials, args.trials, t_start)

    elapsed = time.time() - t_start
    log(f"\nDone in {elapsed:.0f}s ({elapsed/60:.1f}min)")
    log(f"Best IS score: {best_score:.1f}")
    log(f"Best OOS score: {top_results[0].get('walk_forward', {}).get('score', 'N/A')}")
    log(f"Results: {SWEEP_PATH}")


def _serialize_params(params: dict) -> dict:
    """Make params JSON-serializable."""
    p = dict(params)
    if "excluded_cities" in p:
        p["excluded_cities"] = sorted(list(p["excluded_cities"]))
    return p


def _deserialize_params(params: dict) -> dict:
    """Restore params from JSON."""
    p = dict(params)
    if "excluded_cities" in p:
        p["excluded_cities"] = set(p["excluded_cities"])
    return p


def load_ensemble_stats() -> dict[str, dict[str, float]]:
    conn = sqlite3.connect(str(GEFS_DB))
    rows = conn.execute(
        "SELECT city, target_date, ensemble_std FROM ensemble_stats"
    ).fetchall()
    conn.close()
    result: dict[str, dict[str, float]] = defaultdict(dict)
    for city, date_str, std in rows:
        result[city][date_str] = std
    total = sum(len(v) for v in result.values())
    print(f"Ensemble stats loaded: {total} city-dates across {len(result)} cities")
    return dict(result)


def compute_city_mean_std(ensemble_stats):
    result = {}
    for city, stds in ensemble_stats.items():
        vals = list(stds.values())
        if vals:
            result[city] = statistics.mean(vals)
    return result


def _save_results(all_results, top_results, current_trial, total_trials, t_start):
    """Save dashboard-compatible JSON."""
    output = {
        "meta": {
            "sweep_id": "c1f_autoresearch",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_trials": current_trial,
            "sweep_type": "C1f Ensemble EMOS Autoresearch",
            "gefs_days": 75,
            "gefs_members": 31,
            "elapsed_s": round(time.time() - t_start),
        },
        "top_results": top_results[:20],
        "all_results": sorted(all_results, key=lambda r: -r.get("score", 0)),
    }
    with open(SWEEP_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)


if __name__ == "__main__":
    main()
