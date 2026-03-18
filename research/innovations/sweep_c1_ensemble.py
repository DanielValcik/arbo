"""C1 vs C — EMOS Ensemble Backtest (v2 — proper date matching + hybrid).

C  = V4a agg30 (rolling MAE sigma, backward-looking)
C1 = Full replacement: sigma = c + d * ensemble_std (forward-looking)
C1b = Hybrid: sigma = rolling_sigma * modulator(ensemble_std / mean_std)
C1c = Hybrid + sizing: Kelly multiplied by confidence from ensemble agreement

Uses real GEFS 31-member ensemble spread from gefs_ensemble.sqlite.
Proper per-event date matching via global context variable.

Usage:
    python3 research/innovations/sweep_c1_ensemble.py
"""

from __future__ import annotations

import json
import math
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
    CITY_COORDS, SimulationData, load_forecasts,
    serialize_result, walk_forward_validate,
)
from sweep_emos_v2 import (
    _build_training_observations, _prefit_emos_all, run_experiment,
)
import sweep_emos_v2 as emos_v2

RESULTS_DIR = Path(__file__).parent.parent / "data" / "experiments"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
SWEEP_PATH = RESULTS_DIR / "sweep_c1_ensemble.json"
LOG_PATH = Path(__file__).parent.parent / "sweep_c1_ensemble_log.txt"
GEFS_DB = Path(__file__).parent.parent / "data" / "gefs_ensemble.sqlite"

# V4a agg30 locked params
V4A_PARAMS = {
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
}


# ── Data loading ─────────────────────────────────────────────────────

def load_ensemble_stats() -> dict[str, dict[str, float]]:
    """Load ensemble_std per city per date from GEFS DB."""
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


def compute_city_mean_std(
    ensemble_stats: dict[str, dict[str, float]],
) -> dict[str, float]:
    """Compute mean ensemble_std per city (for normalization)."""
    result = {}
    for city, stds in ensemble_stats.items():
        vals = list(stds.values())
        if vals:
            result[city] = statistics.mean(vals)
    return result


# ── Global context for per-event date matching ───────────────────────

_current_event_date: str | None = None  # Set by patched simulate loop
_ensemble_stats: dict[str, dict[str, float]] = {}
_city_mean_std: dict[str, float] = {}
_variant: str = "C"  # "C", "C1", "C1b", "C1c"


def _get_ensemble_std(city: str, target_date: str | None = None) -> float | None:
    """Get ensemble_std for city+date, falling back to nearest available date."""
    city_stds = _ensemble_stats.get(city, {})
    if not city_stds:
        return None

    # Try exact date
    if target_date and target_date in city_stds:
        return city_stds[target_date]

    # Fallback: nearest prior date (walk-forward safe)
    if target_date:
        prior = [d for d in sorted(city_stds.keys()) if d <= target_date]
        if prior:
            return city_stds[prior[-1]]

    return None


# ── Patched compute_prob variants ────────────────────────────────────

_original_compute_prob = ef.compute_prob
_original_compute_size = ef.compute_size


def _compute_prob_hybrid(forecast_temp, bucket, days_out, city, params):
    """C1b: Hybrid — modulate rolling sigma with ensemble spread ratio.

    sigma_final = rolling_sigma * (1 + alpha * (ens_std/mean_std - 1))

    When ensemble agrees (low std): sigma shrinks → narrower dist → bigger edge
    When ensemble disagrees (high std): sigma grows → wider dist → smaller edge
    """
    if _variant == "C":
        return _original_compute_prob(forecast_temp, bucket, days_out, city, params)

    # Get the V4a rolling sigma first
    import strategy_experiment as strategy
    base_sigma = strategy._get_sigma(days_out, city)
    bias = strategy.CITY_BIAS.get(city, 0.0)
    adjusted_temp = forecast_temp + bias

    ens_std = _get_ensemble_std(city, _current_event_date)
    mean_std = _city_mean_std.get(city)

    if _variant in ("C1", "C1f"):
        # Full replacement: use EMOSEnsembleModel
        model = _ensemble_models.get(city)
        if model and model._fitted and model.n_observations >= 10 and ens_std is not None:
            raw = model.bucket_probability(
                forecast_temp,
                bucket.low_c, bucket.high_c,
                bucket.bucket_type or "range",
                ensemble_std=ens_std,
            )
        elif _variant == "C1f":
            # C1f: fallback to EMOS-lite (V4a) instead of raw compute_prob
            emos_v2._emos_enabled = True
            emos_v2._emos_params = params
            _prefit_emos_all(params)
            result = _original_compute_prob(forecast_temp, bucket, days_out, city, params)
            emos_v2._emos_enabled = False
            return result
        else:
            return _original_compute_prob(forecast_temp, bucket, days_out, city, params)
    else:
        # C1b/C1c: Hybrid modulation
        alpha = params.get("ensemble_alpha", 0.3)

        if ens_std is not None and mean_std and mean_std > 0.01:
            ratio = ens_std / mean_std
            modulator = 1.0 + alpha * (ratio - 1.0)
            modulator = max(0.5, min(2.0, modulator))  # clamp
            sigma = base_sigma * modulator
        else:
            sigma = base_sigma

        cdf = lambda x: strategy._normal_cdf(x, adjusted_temp, sigma)

        if bucket.low_c is None and bucket.high_c is not None:
            raw = cdf(bucket.high_c)
        elif bucket.high_c is None and bucket.low_c is not None:
            raw = 1.0 - cdf(bucket.low_c)
        elif bucket.low_c is not None and bucket.high_c is not None:
            raw = cdf(bucket.high_c) - cdf(bucket.low_c)
        else:
            return 0.0

    # Post-processing (same as V4a)
    shrinkage = ef._get_blended_city_param("shrinkage", city, params)
    raw = raw * (1.0 - shrinkage) + 0.125 * shrinkage

    sharpening = ef._get_blended_city_param("prob_sharpening", city, params)
    if sharpening != 1.0 and raw > 0:
        raw = raw ** sharpening

    return raw


def _compute_size_confidence(
    edge, price, available, total_capital, params, city=None,
):
    """C1c: Kelly × ensemble confidence multiplier.

    When ensemble members agree, we're more confident → full Kelly.
    When they disagree, scale down position size.
    """
    base_size = _original_compute_size(edge, price, available, total_capital, params, city=city)

    if _variant != "C1c":
        return base_size

    ens_std = _get_ensemble_std(city, _current_event_date)
    mean_std = _city_mean_std.get(city)

    if ens_std is not None and mean_std and mean_std > 0.01:
        # confidence = 1 when std < mean (agree), < 1 when std > mean (disagree)
        ratio = ens_std / mean_std
        confidence = max(0.5, min(1.2, 1.0 / ratio))
        return base_size * confidence

    return base_size


# ── Patched simulate_portfolio to set event date context ─────────────

_original_simulate = ef.simulate_portfolio


def _simulate_with_date_context(sim_data, params, **kwargs):
    """Wrapper that sets _current_event_date before each compute_prob call.

    We monkey-patch the event iteration to inject date context.
    """
    # We can't easily inject into the tight loop in simulate_portfolio.
    # Instead, we patch at the simulate_portfolio level by modifying
    # the events to carry date info that compute_prob can access.
    #
    # Simpler approach: pre-compute ensemble_std per (city, target_date)
    # and store in a lookup that compute_prob accesses via the bucket/event.
    #
    # Actually, the simplest correct approach: patch the iteration.
    # We'll re-implement the key loop with date context.
    return _original_simulate(sim_data, params, **kwargs)


# Since we can't easily inject into ef.simulate_portfolio's tight loop,
# use a different approach: pre-compute the sigma modulation per (city, date)
# and inject it into the strategy module's _get_sigma function.

_original_get_sigma = None
_sigma_overrides: dict[tuple[str, str], float] = {}


def _precompute_sigma_overrides(params: dict) -> None:
    """Pre-compute per (city, date) sigma values incorporating ensemble spread."""
    import strategy_experiment as strategy

    alpha = params.get("ensemble_alpha", 0.3)
    _sigma_overrides.clear()

    for city, date_stds in _ensemble_stats.items():
        mean_std = _city_mean_std.get(city)
        if not mean_std or mean_std < 0.01:
            continue

        for date_str, ens_std in date_stds.items():
            base_sigma = strategy._get_sigma(1, city)  # days_out=1 for weather
            ratio = ens_std / mean_std
            modulator = 1.0 + alpha * (ratio - 1.0)
            modulator = max(0.5, min(2.0, modulator))
            _sigma_overrides[(city, date_str)] = base_sigma * modulator


# ── Ensemble EMOS models (for C1 full replacement) ───────────────────

_ensemble_models: dict[str, EMOSEnsembleModel] = {}


# ── Runner ───────────────────────────────────────────────────────────

def run_variant(sim_data, params, variant_name, eid):
    """Run a specific variant."""
    global _variant
    _variant = variant_name

    ef.compute_prob = _compute_prob_hybrid
    if variant_name == "C1c":
        ef.compute_size = _compute_size_confidence

    result = ef.simulate_portfolio(
        sim_data, params, entry_hours=24,
        experiment_id=eid, record_equity=True,
    )
    result.score = ef.experiment_score(result)

    wf = walk_forward_validate(sim_data, params, entry_hours=24, n_folds=3)

    # Restore originals
    ef.compute_prob = _original_compute_prob
    ef.compute_size = _original_compute_size
    _variant = "C"

    return result, wf


def log_result(log, name, result, wf):
    """Pretty-print one variant's results."""
    cr = result.city_results
    log(f"\n{'=' * 50}")
    log(f"{name}")
    log(f"  Score: {result.score:.1f}  Trades: {result.trades}  WR: {result.win_rate}%")
    log(f"  PnL: ${result.total_pnl:,.2f}  DD: {result.max_drawdown_pct:.1f}%  Sharpe: {result.sharpe}")
    log(f"  WF OOS: score={wf['score']:.1f}  PnL=${wf.get('oos_pnl', 0):.2f}")
    if wf.get("folds"):
        for fi, f in enumerate(wf["folds"]):
            log(f"    Fold {fi+1}: PnL=${f['pnl']:.2f}  trades={f['trades']}  WR={f['wr']}%")
    log(f"  Per-city:")
    for city in sorted(cr, key=lambda c: -cr[c]["pnl"]):
        v = cr[city]
        log(f"    {city:<14} ${v['pnl']:>8,.2f}  ({v['trades']} tr, WR={v['win_rate']}%)")


def main():
    t_start = time.time()

    # Clear log
    LOG_PATH.write_text("")

    print("Loading data...")
    events, buckets, prices = load_pmd_data()
    forecasts = load_forecasts(events)
    sim_data = SimulationData(events, buckets, prices, forecasts)

    # EMOS-lite training data (for C baseline)
    emos_v2._emos_obs = _build_training_observations(events, buckets, forecasts)

    # Ensemble data
    global _ensemble_stats, _city_mean_std
    _ensemble_stats = load_ensemble_stats()
    _city_mean_std = compute_city_mean_std(_ensemble_stats)

    print(f"\nCity mean ensemble std (for normalization):")
    for city in sorted(_city_mean_std, key=lambda c: _city_mean_std[c]):
        print(f"  {city:<14} mean_std={_city_mean_std[city]:.3f}")

    # Build ensemble EMOS models per city (for C1)
    global _ensemble_models
    for city, obs_list in emos_v2._emos_obs.items():
        city_stds = _ensemble_stats.get(city, {})
        if not city_stds:
            continue
        model = EMOSEnsembleModel(
            training_window=66, sigma_floor=0.6, bias_method="rolling_mean",
        )
        model.fit(obs_list, city_stds)
        _ensemble_models[city] = model

    print(f"\nEnsemble EMOS models: {len(_ensemble_models)} cities")
    for city, model in sorted(_ensemble_models.items()):
        c, d = model.sigma_params
        print(f"  {city:<14} c={c:.3f} d={d:.3f} bias={model.bias:.2f} n={model.n_observations}")

    # Inject _current_event_date into simulation loop.
    # Approach: monkey-patch ef.compute_prob to extract date from global event context.
    # We'll do this by patching the simulate_portfolio loop to set the context.
    #
    # Since ef.simulate_portfolio iterates events which have .target_date,
    # we patch it at a higher level: before each call to compute_prob within simulate,
    # we need the date. The cleanest way without modifying experiment_framework:
    # make compute_prob look up the ensemble_std from (city, target_date) where
    # target_date comes from a thread-local / global context.
    #
    # We can set this by patching ef.quality_gate (called right after compute_prob
    # with the same event context) — but that's fragile.
    #
    # Better approach: for C1b/C1c, the ensemble effect is purely on sigma.
    # We can pre-compute a sigma_scale per (city, date) and pass it via
    # city_overrides mechanism or by patching strategy._get_sigma.
    #
    # SIMPLEST: override strategy._get_sigma to check a lookup table keyed on
    # (city, current_date). Set current_date in a patched compute_prob
    # that also calls _get_sigma (circular...).
    #
    # Actually let's just modify experiment_framework.compute_prob properly:
    # We know it's called from simulate_portfolio at line 845 with `ev` in scope.
    # The simplest working approach: scan through the events to build a
    # per-(city, date) → ensemble_std map. Then in compute_prob, we don't
    # need the exact date — we can use city + bucket characteristics to
    # look up events. But that's not reliable either.
    #
    # PRAGMATIC: For C1b hybrid, the modulation alpha is typically small (0.3),
    # and the ensemble_std varies day-to-day within a narrow range for each city.
    # Using per-city mean is a ~0.7 correlation proxy. The signal we're testing
    # is whether ANY ensemble information helps — the per-day precision is
    # secondary. Let's test with mean_std (always available) first, then
    # per-day precision as a refinement IF the mean_std signal shows promise.
    #
    # Even better: test per-day by scanning forward through each unique event date
    # and setting the global context. We monkey-patch simulate_portfolio to do this.

    # Instead of complex monkey-patching, directly patch the framework's compute_prob
    # call site. We wrap the entire simulate_portfolio call such that each event
    # iteration sets the global date.

    # Let's take a step back and use the most robust approach:
    # Modify ef.compute_prob to accept an optional date parameter via **kwargs,
    # and in our patched version, extract it. But ef.simulate_portfolio doesn't
    # pass date. So we use the approach of scanning events to build
    # event_id → target_date mapping, then from the bucket (which has event_id)
    # we can look up the date.

    # Build event_id → target_date mapping
    _event_dates: dict[str, str] = {}
    for ev in events:
        _event_dates[ev.event_id] = ev.target_date

    # Build bucket → event_id mapping
    _bucket_event: dict[str, str] = {}
    for ev in events:
        for b in buckets.get(ev.event_id, []):
            _bucket_event[b.token_id] = ev.event_id

    def _compute_prob_with_context(forecast_temp, bucket, days_out, city, params):
        """Patched compute_prob that resolves date from bucket → event → date."""
        global _current_event_date
        event_id = _bucket_event.get(bucket.token_id)
        _current_event_date = _event_dates.get(event_id) if event_id else None
        return _compute_prob_hybrid(forecast_temp, bucket, days_out, city, params)

    def _compute_size_with_context(edge, price, available, total_capital, params, city=None):
        """Patched compute_size with ensemble date context (already set by compute_prob)."""
        return _compute_size_confidence(edge, price, available, total_capital, params, city=city)

    def log(msg):
        print(msg, flush=True)
        with open(LOG_PATH, "a") as f:
            f.write(msg + "\n")

    log(f"\n{'=' * 70}")
    log(f"C1 vs C — EMOS Ensemble Backtest (v2)")
    log(f"  GEFS: 31 members, 75 days, 19 cities")
    log(f"  C  = V4a agg30 (rolling MAE sigma, backward-looking)")
    log(f"  C1 = Full replacement: sigma = c + d*ensemble_std")
    log(f"  C1b = Hybrid: sigma = rolling_sigma * (1 + alpha*(ens/mean - 1))")
    log(f"  C1c = Hybrid + confidence sizing")
    log(f"{'=' * 70}")

    all_results = {}

    # ── C: Baseline ──
    # run_experiment sets _emos_enabled=False after IS. Must re-enable for WF.
    c_result = run_experiment(sim_data, V4A_PARAMS, "C_baseline")

    emos_v2._emos_enabled = True
    emos_v2._emos_params = V4A_PARAMS
    _prefit_emos_all(V4A_PARAMS)
    c_wf = walk_forward_validate(sim_data, V4A_PARAMS, entry_hours=24, n_folds=3)
    emos_v2._emos_enabled = False

    log_result(log, "C (V4a baseline — rolling MAE sigma)", c_result, c_wf)
    c_rd = serialize_result(c_result)
    c_rd["walk_forward"] = c_wf
    all_results["C_baseline"] = c_rd

    # ── C1: Full ensemble replacement ──
    global _variant
    _variant = "C1"
    ef.compute_prob = _compute_prob_with_context

    c1_result = ef.simulate_portfolio(
        sim_data, V4A_PARAMS, entry_hours=24,
        experiment_id="C1_ensemble", record_equity=True,
    )
    c1_result.score = ef.experiment_score(c1_result)
    c1_wf = walk_forward_validate(sim_data, V4A_PARAMS, entry_hours=24, n_folds=3)
    ef.compute_prob = _original_compute_prob
    _variant = "C"

    log_result(log, "C1 (ensemble EMOS — sigma = c + d*ens_std)", c1_result, c1_wf)
    c1_rd = serialize_result(c1_result)
    c1_rd["walk_forward"] = c1_wf
    all_results["C1_ensemble"] = c1_rd

    # ── C1f: Ensemble with EMOS-lite fallback ──
    _variant = "C1f"
    ef.compute_prob = _compute_prob_with_context

    c1f_result = ef.simulate_portfolio(
        sim_data, V4A_PARAMS, entry_hours=24,
        experiment_id="C1f_fallback", record_equity=True,
    )
    c1f_result.score = ef.experiment_score(c1f_result)
    c1f_wf = walk_forward_validate(sim_data, V4A_PARAMS, entry_hours=24, n_folds=3)
    ef.compute_prob = _original_compute_prob
    _variant = "C"

    log_result(log, "C1f (ensemble + EMOS-lite fallback for low-n cities)", c1f_result, c1f_wf)
    c1f_rd = serialize_result(c1f_result)
    c1f_rd["walk_forward"] = c1f_wf
    all_results["C1f_fallback"] = c1f_rd

    # ── C1b: Hybrid alpha sweep ──
    for alpha in [0.15, 0.30, 0.50, 0.75]:
        _variant = "C1b"
        test_params = {**V4A_PARAMS, "ensemble_alpha": alpha}
        ef.compute_prob = _compute_prob_with_context

        r = ef.simulate_portfolio(
            sim_data, test_params, entry_hours=24,
            experiment_id=f"C1b_a{alpha}", record_equity=True,
        )
        r.score = ef.experiment_score(r)
        wf = walk_forward_validate(sim_data, test_params, entry_hours=24, n_folds=3)
        ef.compute_prob = _original_compute_prob
        _variant = "C"

        name = f"C1b (hybrid alpha={alpha})"
        log_result(log, name, r, wf)
        rd = serialize_result(r)
        rd["walk_forward"] = wf
        rd["ensemble_alpha"] = alpha
        all_results[f"C1b_a{alpha}"] = rd

    # ── C1c: Hybrid + confidence sizing (best C1b alpha) ──
    best_c1b = max(
        [(k, v) for k, v in all_results.items() if k.startswith("C1b_")],
        key=lambda x: x[1].get("walk_forward", {}).get("score", 0),
    )
    best_alpha = all_results[best_c1b[0]].get("ensemble_alpha", 0.3)

    _variant = "C1c"
    test_params = {**V4A_PARAMS, "ensemble_alpha": best_alpha}
    ef.compute_prob = _compute_prob_with_context
    ef.compute_size = _compute_size_with_context

    c1c_result = ef.simulate_portfolio(
        sim_data, test_params, entry_hours=24,
        experiment_id="C1c_best", record_equity=True,
    )
    c1c_result.score = ef.experiment_score(c1c_result)
    c1c_wf = walk_forward_validate(sim_data, test_params, entry_hours=24, n_folds=3)
    ef.compute_prob = _original_compute_prob
    ef.compute_size = _original_compute_size
    _variant = "C"

    log_result(log, f"C1c (hybrid + confidence sizing, alpha={best_alpha})", c1c_result, c1c_wf)
    c1c_rd = serialize_result(c1c_result)
    c1c_rd["walk_forward"] = c1c_wf
    all_results["C1c_best"] = c1c_rd

    # ── Summary comparison ──
    log(f"\n{'=' * 80}")
    log("SUMMARY — All Variants")
    log(f"{'=' * 80}")
    log(f"{'Variant':<22} {'Score':>6} {'OOS':>6} {'Trades':>7} {'WR%':>5} "
        f"{'PnL':>10} {'DD%':>5} {'Shrp':>5} {'Util%':>6}")
    log("-" * 80)

    for name, rd in sorted(all_results.items(), key=lambda x: -x[1].get("walk_forward", {}).get("score", 0)):
        wf = rd.get("walk_forward", {})
        log(f"{name:<22} {rd.get('score', 0):>6.1f} {wf.get('score', 0):>6.1f} "
            f"{rd.get('trades', 0):>7} {rd.get('win_rate', 0):>5.1f} "
            f"${rd.get('total_pnl', 0):>9,.0f} {rd.get('max_drawdown_pct', 0):>5.1f} "
            f"{rd.get('sharpe', 0):>5.2f} {rd.get('capital_utilization', 0):>6.1f}")

    # ── Verdict ──
    c_oos = all_results["C_baseline"].get("walk_forward", {}).get("score", 0)
    best_variant = max(
        all_results.items(),
        key=lambda x: x[1].get("walk_forward", {}).get("score", 0),
    )
    best_oos = best_variant[1].get("walk_forward", {}).get("score", 0)

    log(f"\n{'=' * 70}")
    if best_variant[0] == "C_baseline":
        log("VERDICT: V4a baseline WINS. Ensemble does NOT improve OOS performance.")
        log("Recommendation: Stay on V4a, accept 55% utilization.")
    elif best_oos > c_oos * 1.10:  # Need >10% OOS improvement
        log(f"VERDICT: {best_variant[0]} improves OOS by {(best_oos/c_oos - 1)*100:.0f}%.")
        log(f"Recommendation: Consider deploying {best_variant[0]}.")
    else:
        log(f"VERDICT: Best variant ({best_variant[0]}) OOS improvement is marginal.")
        log("Recommendation: Stay on V4a — ensemble signal too weak to justify complexity.")
    log(f"{'=' * 70}")

    # ── Save ──
    output = {
        "sweep_id": "c1_ensemble_v2",
        "sweep_type": "C1 vs C — EMOS Ensemble Backtest v2",
        "meta": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_trials": len(all_results),
            "gefs_days": 75,
            "gefs_members": 31,
        },
        "variants": all_results,
        "best": best_variant[1],
        "best_name": best_variant[0],
        "top_results": sorted(
            all_results.values(),
            key=lambda r: -r.get("walk_forward", {}).get("score", 0),
        ),
        "all_results": sorted(
            all_results.values(), key=lambda r: -r.get("score", 0),
        ),
    }
    with open(SWEEP_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)

    elapsed = time.time() - t_start
    log(f"\nDone in {elapsed:.0f}s. Results: {SWEEP_PATH}")


if __name__ == "__main__":
    main()
