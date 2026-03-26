"""
Full Parameter Sweep for Strategy C — Quarter-Kelly Fixed
==========================================================

Systematically searches the best combination of:
  1. Quality gate thresholds (MIN_EDGE, MAX_PRICE, MIN_PRICE, MIN_FORECAST_PROB)
  2. Probability model (PROB_SHARPENING, shrinkage weight, kelly_raw_cap)
  3. Per-city overrides (exclusions + widenings)

KELLY_FRACTION is FIXED at 0.25 (quarter-Kelly, architectural constant).

Strategy: 3-phase search
  Phase 1: Random search across full parameter space (500 trials)
  Phase 2: Grid search around top-10 from Phase 1 (fine-tuning)
  Phase 3: Per-city override optimization on best global params

Usage: python3 research/sweep_full_v2.py
Output: research/sweep_full_v2_results.json
"""

import copy
import json
import itertools
import math
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

# Setup path
sys.path.insert(0, str(Path(__file__).parent))
import backtest_harness as harness

# Load data once
harness.download_historical_data()
DATA = harness.load_data()
NORMALS = harness.compute_monthly_normals(DATA)

RESULTS_FILE = Path(__file__).parent / "sweep_full_v2_results.json"
LOG_FILE = Path(__file__).parent / "sweep_full_v2_log.txt"

ALL_CITIES = list(harness.CITIES.keys())

# Current best (baseline) — loaded from strategy_experiment.py
import strategy_experiment as current_strategy

BASELINE_PARAMS = {
    "MIN_EDGE": current_strategy.MIN_EDGE,
    "MAX_PRICE": current_strategy.MAX_PRICE,
    "MIN_PRICE": current_strategy.MIN_PRICE,
    "MIN_FORECAST_PROB": current_strategy.MIN_FORECAST_PROB,
    "MIN_VOLUME": current_strategy.MIN_VOLUME,
    "MIN_LIQUIDITY": current_strategy.MIN_LIQUIDITY,
    "PROB_SHARPENING": current_strategy.PROB_SHARPENING,
    "SHRINKAGE_WEIGHT": 0.03,
    "KELLY_RAW_CAP": 0.40,
    "CONVICTION_RATIO": current_strategy.CONVICTION_RATIO,
    "CITY_OVERRIDES": copy.deepcopy(current_strategy.CITY_OVERRIDES),
    "CITY_SIGMA": copy.deepcopy(current_strategy.CITY_SIGMA),
    "CITY_BIAS": copy.deepcopy(current_strategy.CITY_BIAS),
}


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def apply_params(params):
    """Apply parameter dict to strategy_experiment module."""
    import strategy_experiment as s

    s.MIN_EDGE = params["MIN_EDGE"]
    s.MAX_PRICE = params["MAX_PRICE"]
    s.MIN_PRICE = params["MIN_PRICE"]
    s.MIN_FORECAST_PROB = params["MIN_FORECAST_PROB"]
    s.MIN_VOLUME = params["MIN_VOLUME"]
    s.MIN_LIQUIDITY = params["MIN_LIQUIDITY"]
    s.PROB_SHARPENING = params["PROB_SHARPENING"]
    s.CONVICTION_RATIO = params["CONVICTION_RATIO"]

    # Shrinkage weight is hardcoded in estimate_probability — patch it
    # We monkey-patch the function
    shrinkage = params["SHRINKAGE_WEIGHT"]
    kelly_cap = params["KELLY_RAW_CAP"]

    orig_sigma_fn = s._get_sigma

    if "CITY_SIGMA" in params:
        s.CITY_SIGMA = params["CITY_SIGMA"]
    if "CITY_BIAS" in params:
        s.CITY_BIAS = params["CITY_BIAS"]
    if "CITY_OVERRIDES" in params:
        s.CITY_OVERRIDES = params["CITY_OVERRIDES"]

    # Patch estimate_probability with current shrinkage/sharpening
    sharpening = params["PROB_SHARPENING"]

    def patched_estimate_probability(forecast_temp_c, bucket_low_c, bucket_high_c,
                                      days_out, *, city=None):
        sigma = s._get_sigma(days_out, city)

        if s.DISTRIBUTION == "student_t":
            cdf = lambda x: s._student_t_cdf(x, forecast_temp_c, sigma, s.STUDENT_T_DF)
        else:
            cdf = lambda x: s._normal_cdf(x, forecast_temp_c, sigma)

        if bucket_low_c is None and bucket_high_c is not None:
            raw = cdf(bucket_high_c)
        elif bucket_high_c is None and bucket_low_c is not None:
            raw = 1.0 - cdf(bucket_low_c)
        elif bucket_low_c is not None and bucket_high_c is not None:
            raw = cdf(bucket_high_c) - cdf(bucket_low_c)
        else:
            return 0.0

        uniform_prior = 0.125
        raw = raw * (1.0 - shrinkage) + uniform_prior * shrinkage

        if sharpening != 1.0 and raw > 0:
            raw = raw ** sharpening
        return raw

    s.estimate_probability = patched_estimate_probability

    # Patch position_size with kelly_raw_cap
    def patched_position_size(edge, market_price, available_capital, total_capital,
                               *, city=None):
        if market_price <= 0 or market_price >= 1 or edge <= 0:
            return 0.0
        prob = market_price + edge
        if prob <= 0 or prob >= 1:
            return 0.0
        odds = (1.0 / market_price) - 1.0
        kelly_raw = (prob * odds - (1.0 - prob)) / odds
        if kelly_raw <= 0:
            return 0.0
        kelly_raw = min(kelly_raw, kelly_cap)
        kelly_adjusted = kelly_raw * s.KELLY_FRACTION
        size = available_capital * kelly_adjusted
        max_size = total_capital * s.MAX_POSITION_PCT
        size = min(size, max_size)
        if size < 1.0:
            return 0.0
        return round(size, 2)

    s.position_size = patched_position_size


def evaluate(params):
    """Run walk-forward evaluation with given params. Returns results dict."""
    apply_params(params)
    results = harness.walk_forward_evaluate(DATA, NORMALS)

    # Collect per-city stats
    all_trades = []
    for wm in results["windows"]:
        seed = harness.BASE_SEED + (wm["window"] - 1) * 7919
        test_start, test_end = harness.WALK_FORWARD_WINDOWS[wm["window"] - 1]["test"]
        trades_w, _ = harness.run_single_backtest(DATA, NORMALS, test_start, test_end, seed)
        all_trades.extend(trades_w)

    city_pnl = {}
    for t in all_trades:
        city_pnl.setdefault(t.city, 0.0)
        city_pnl[t.city] += t.pnl

    unprofitable_cities = [c for c, p in city_pnl.items() if p < 0]

    return {
        "composite_score": results["composite_score"],
        "avg_sharpe": results["avg_sharpe"],
        "total_trades": results["total_trades"],
        "avg_win_rate": results["avg_win_rate"],
        "max_drawdown_pct": results["max_drawdown_pct"],
        "avg_pnl_pct": results["avg_pnl_pct"],
        "profitable_windows": results["profitable_windows"],
        "total_windows": results["total_windows"],
        "city_pnl": {c: round(p, 2) for c, p in sorted(city_pnl.items(), key=lambda x: -x[1])},
        "unprofitable_cities": unprofitable_cities,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1: Random Search
# ═══════════════════════════════════════════════════════════════════════════════

def random_params(rng):
    """Generate random parameter combination."""
    p = copy.deepcopy(BASELINE_PARAMS)

    p["MIN_EDGE"] = rng.choice([0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.12, 0.15])
    p["MAX_PRICE"] = rng.choice([0.38, 0.40, 0.43, 0.45, 0.48, 0.50, 0.55])
    p["MIN_PRICE"] = rng.choice([0.20, 0.25, 0.28, 0.30, 0.33, 0.35])
    p["MIN_FORECAST_PROB"] = rng.choice([0.50, 0.55, 0.58, 0.60, 0.62, 0.65, 0.68, 0.70])
    p["PROB_SHARPENING"] = rng.choice([0.95, 0.98, 1.0, 1.02, 1.05, 1.08, 1.10, 1.15])
    p["SHRINKAGE_WEIGHT"] = rng.choice([0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10])
    p["KELLY_RAW_CAP"] = rng.choice([0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.80])
    p["CONVICTION_RATIO"] = rng.choice([0.0, 1.2, 1.3, 1.4, 1.5, 1.6])
    p["MIN_VOLUME"] = rng.choice([500, 1000, 2000, 5000])
    p["MIN_LIQUIDITY"] = rng.choice([100, 200, 500, 1000])

    return p


def phase1_random_search(n_trials=500):
    """Phase 1: Explore the parameter space with random search."""
    log(f"=== PHASE 1: Random Search ({n_trials} trials) ===")

    rng = random.Random(42)
    results = []

    # First evaluate baseline
    log("Evaluating baseline...")
    baseline_result = evaluate(BASELINE_PARAMS)
    baseline_entry = {
        "trial": 0,
        "params": {k: v for k, v in BASELINE_PARAMS.items()
                   if k not in ("CITY_SIGMA", "CITY_BIAS", "CITY_OVERRIDES")},
        "result": baseline_result,
        "label": "BASELINE",
    }
    results.append(baseline_entry)
    log(f"  BASELINE: score={baseline_result['composite_score']:.4f} "
        f"sharpe={baseline_result['avg_sharpe']:.4f} "
        f"trades={baseline_result['total_trades']} "
        f"wr={baseline_result['avg_win_rate']:.1f}% "
        f"dd={baseline_result['max_drawdown_pct']:.2f}%")

    best_score = baseline_result["composite_score"]
    best_trial = 0

    for i in range(1, n_trials + 1):
        params = random_params(rng)
        result = evaluate(params)

        entry = {
            "trial": i,
            "params": {k: v for k, v in params.items()
                       if k not in ("CITY_SIGMA", "CITY_BIAS", "CITY_OVERRIDES")},
            "result": result,
        }
        results.append(entry)

        marker = ""
        if result["composite_score"] > best_score:
            best_score = result["composite_score"]
            best_trial = i
            marker = " *** NEW BEST ***"

        if i % 25 == 0 or marker:
            log(f"  Trial {i}/{n_trials}: score={result['composite_score']:.4f} "
                f"sharpe={result['avg_sharpe']:.4f} "
                f"trades={result['total_trades']} "
                f"wr={result['avg_win_rate']:.1f}% "
                f"dd={result['max_drawdown_pct']:.2f}%{marker}")

    log(f"Phase 1 complete. Best score: {best_score:.4f} (trial {best_trial})")
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2: Fine-tune top-10
# ═══════════════════════════════════════════════════════════════════════════════

def fine_tune_params(base_params, rng):
    """Small perturbation around a good parameter set."""
    p = copy.deepcopy(base_params)

    # Perturb one or two params slightly
    param_choices = ["MIN_EDGE", "MAX_PRICE", "MIN_PRICE", "MIN_FORECAST_PROB",
                     "PROB_SHARPENING", "SHRINKAGE_WEIGHT", "KELLY_RAW_CAP"]
    n_perturb = rng.randint(1, 3)

    for _ in range(n_perturb):
        param = rng.choice(param_choices)
        if param == "MIN_EDGE":
            p[param] = max(0.03, min(0.20, p[param] + rng.uniform(-0.02, 0.02)))
        elif param == "MAX_PRICE":
            p[param] = max(0.35, min(0.60, p[param] + rng.uniform(-0.03, 0.03)))
        elif param == "MIN_PRICE":
            p[param] = max(0.15, min(0.40, p[param] + rng.uniform(-0.03, 0.03)))
        elif param == "MIN_FORECAST_PROB":
            p[param] = max(0.45, min(0.75, p[param] + rng.uniform(-0.03, 0.03)))
        elif param == "PROB_SHARPENING":
            p[param] = max(0.90, min(1.20, p[param] + rng.uniform(-0.03, 0.03)))
        elif param == "SHRINKAGE_WEIGHT":
            p[param] = max(0.0, min(0.15, p[param] + rng.uniform(-0.02, 0.02)))
        elif param == "KELLY_RAW_CAP":
            p[param] = max(0.20, min(0.90, p[param] + rng.uniform(-0.05, 0.05)))

    # Round for cleanliness
    for k in ["MIN_EDGE", "MAX_PRICE", "MIN_PRICE", "MIN_FORECAST_PROB",
              "PROB_SHARPENING", "SHRINKAGE_WEIGHT", "KELLY_RAW_CAP"]:
        p[k] = round(p[k], 3)

    return p


def phase2_fine_tune(phase1_results, n_per_top=30):
    """Phase 2: Fine-tune around top-10 from Phase 1."""
    sorted_results = sorted(phase1_results, key=lambda x: -x["result"]["composite_score"])
    top10 = sorted_results[:10]

    log(f"\n=== PHASE 2: Fine-tuning top-10 ({n_per_top} perturbations each) ===")
    for i, entry in enumerate(top10):
        log(f"  Top {i+1}: trial={entry['trial']} score={entry['result']['composite_score']:.4f}")

    rng = random.Random(123)
    results = list(phase1_results)  # Keep all phase 1 results
    best_score = top10[0]["result"]["composite_score"]
    improvements = 0

    trial_offset = len(phase1_results)
    for rank, top_entry in enumerate(top10):
        base_params = copy.deepcopy(BASELINE_PARAMS)
        # Apply the non-city params from this top entry
        for k, v in top_entry["params"].items():
            base_params[k] = v

        for j in range(n_per_top):
            params = fine_tune_params(base_params, rng)
            result = evaluate(params)
            trial_id = trial_offset + rank * n_per_top + j

            entry = {
                "trial": trial_id,
                "params": {k: v for k, v in params.items()
                           if k not in ("CITY_SIGMA", "CITY_BIAS", "CITY_OVERRIDES")},
                "result": result,
                "label": f"finetune_top{rank+1}",
            }
            results.append(entry)

            if result["composite_score"] > best_score:
                best_score = result["composite_score"]
                improvements += 1
                log(f"  Fine-tune top{rank+1} #{j}: score={result['composite_score']:.4f} "
                    f"*** NEW BEST ***")

    log(f"Phase 2 complete. {improvements} improvements found. Best: {best_score:.4f}")
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3: Per-city override optimization
# ═══════════════════════════════════════════════════════════════════════════════

def phase3_city_overrides(all_results):
    """Phase 3: Optimize which cities to exclude/widen using best global params."""
    sorted_results = sorted(all_results, key=lambda x: -x["result"]["composite_score"])
    best_entry = sorted_results[0]

    log(f"\n=== PHASE 3: Per-city Override Optimization ===")
    log(f"Starting from best: trial={best_entry['trial']} "
        f"score={best_entry['result']['composite_score']:.4f}")

    # Get best global params
    best_global = copy.deepcopy(BASELINE_PARAMS)
    for k, v in best_entry["params"].items():
        best_global[k] = v

    # First: evaluate with NO city overrides to see raw per-city performance
    no_override_params = copy.deepcopy(best_global)
    no_override_params["CITY_OVERRIDES"] = {}
    no_override_result = evaluate(no_override_params)
    log(f"No overrides: score={no_override_result['composite_score']:.4f} "
        f"trades={no_override_result['total_trades']}")

    city_pnl = no_override_result["city_pnl"]
    log(f"Per-city PnL (no overrides):")
    for city, pnl in city_pnl.items():
        log(f"  {city:>16}: ${pnl:>10.2f}")

    # Identify unprofitable cities (candidates for exclusion)
    unprofitable = [c for c, p in city_pnl.items() if p < 0]
    marginal = [c for c, p in city_pnl.items() if 0 <= p < 500]
    strong = [c for c, p in city_pnl.items() if p >= 2000]

    log(f"Unprofitable: {unprofitable}")
    log(f"Marginal (< $500): {marginal}")
    log(f"Strong (> $2000): {strong}")

    results = list(all_results)
    trial_offset = len(all_results)
    best_score = best_entry["result"]["composite_score"]

    # Try excluding different combinations of unprofitable + marginal cities
    exclude_candidates = unprofitable + marginal
    widen_candidates = strong

    # Strategy: try excluding each unprofitable/marginal city individually,
    # then combine the best exclusions
    log(f"\nTesting individual exclusions...")
    exclusion_scores = {}

    for city in exclude_candidates:
        params = copy.deepcopy(best_global)
        params["CITY_OVERRIDES"] = copy.deepcopy(BASELINE_PARAMS.get("CITY_OVERRIDES", {}))
        params["CITY_OVERRIDES"][city] = {"min_edge": 0.99}
        result = evaluate(params)
        exclusion_scores[city] = result["composite_score"]
        marker = " ***" if result["composite_score"] > best_score else ""
        log(f"  Exclude {city:>16}: score={result['composite_score']:.4f} "
            f"trades={result['total_trades']}{marker}")

        results.append({
            "trial": trial_offset,
            "params": {k: v for k, v in params.items()
                       if k not in ("CITY_SIGMA", "CITY_BIAS")},
            "result": result,
            "label": f"exclude_{city}",
        })
        trial_offset += 1

    # Sort exclusions by score improvement
    helpful_exclusions = [c for c, s in exclusion_scores.items()
                          if s > best_entry["result"]["composite_score"]]
    log(f"Helpful exclusions: {helpful_exclusions}")

    # Try combining helpful exclusions
    if helpful_exclusions:
        log(f"\nTesting exclusion combinations...")
        for n in range(2, min(len(helpful_exclusions) + 1, 8)):
            for combo in itertools.combinations(helpful_exclusions, n):
                params = copy.deepcopy(best_global)
                params["CITY_OVERRIDES"] = {}
                for city in combo:
                    params["CITY_OVERRIDES"][city] = {"min_edge": 0.99}

                result = evaluate(params)
                marker = ""
                if result["composite_score"] > best_score:
                    best_score = result["composite_score"]
                    marker = " *** NEW BEST ***"

                combo_str = "+".join(combo)
                log(f"  Exclude [{combo_str}]: score={result['composite_score']:.4f} "
                    f"trades={result['total_trades']}{marker}")

                results.append({
                    "trial": trial_offset,
                    "params": {k: v for k, v in params.items()
                               if k not in ("CITY_SIGMA", "CITY_BIAS")},
                    "result": result,
                    "label": f"exclude_combo_{combo_str}",
                })
                trial_offset += 1

    # Try widening price for strong cities
    log(f"\nTesting price widening for strong cities...")
    # Start from best exclusion combo
    best_so_far = sorted(results, key=lambda x: -x["result"]["composite_score"])[0]
    base_overrides = best_so_far.get("params", {}).get("CITY_OVERRIDES",
                                                        BASELINE_PARAMS.get("CITY_OVERRIDES", {}))

    for widen_price in [0.48, 0.50, 0.55, 0.60]:
        for n in range(1, min(len(widen_candidates) + 1, 7)):
            for combo in itertools.combinations(widen_candidates, n):
                params = copy.deepcopy(best_global)
                # Apply best exclusions
                params["CITY_OVERRIDES"] = copy.deepcopy(base_overrides) if isinstance(base_overrides, dict) else {}

                for city in combo:
                    if city not in params["CITY_OVERRIDES"] or \
                       params["CITY_OVERRIDES"].get(city, {}).get("min_edge") != 0.99:
                        params["CITY_OVERRIDES"][city] = {"max_price": widen_price}

                result = evaluate(params)
                marker = ""
                if result["composite_score"] > best_score:
                    best_score = result["composite_score"]
                    marker = " *** NEW BEST ***"

                if marker or n <= 2:
                    combo_str = "+".join(combo)
                    log(f"  Widen [{combo_str}] to {widen_price}: "
                        f"score={result['composite_score']:.4f} "
                        f"trades={result['total_trades']}{marker}")

                results.append({
                    "trial": trial_offset,
                    "params": {k: v for k, v in params.items()
                               if k not in ("CITY_SIGMA", "CITY_BIAS")},
                    "result": result,
                    "label": f"widen_{widen_price}",
                })
                trial_offset += 1

    log(f"Phase 3 complete. Best score: {best_score:.4f}")
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    t_start = time.time()
    log(f"Starting full parameter sweep at {datetime.now().isoformat()}")
    log(f"Sizing FIXED: KELLY_FRACTION=0.25 (quarter-Kelly)")
    log(f"Initial capital: ${harness.INITIAL_CAPITAL}")
    log(f"Walk-forward: {len(harness.WALK_FORWARD_WINDOWS)} windows")
    log("")

    # Phase 1
    results = phase1_random_search(n_trials=500)

    # Phase 2
    results = phase2_fine_tune(results, n_per_top=30)

    # Phase 3
    results = phase3_city_overrides(results)

    # Final summary
    sorted_all = sorted(results, key=lambda x: -x["result"]["composite_score"])
    top5 = sorted_all[:5]

    elapsed = time.time() - t_start
    log(f"\n{'='*78}")
    log(f"SWEEP COMPLETE — {len(results)} total evaluations in {elapsed:.0f}s")
    log(f"{'='*78}")

    for i, entry in enumerate(top5):
        r = entry["result"]
        log(f"\nTop {i+1}: score={r['composite_score']:.4f} "
            f"sharpe={r['avg_sharpe']:.4f} "
            f"trades={r['total_trades']} "
            f"wr={r['avg_win_rate']:.1f}% "
            f"dd={r['max_drawdown_pct']:.2f}% "
            f"pnl={r['avg_pnl_pct']:.1f}%")
        log(f"  Params: {json.dumps(entry['params'], default=str)}")
        if entry.get("label"):
            log(f"  Label: {entry['label']}")

    # Save results
    output = {
        "sweep_date": datetime.now().isoformat(),
        "elapsed_seconds": round(elapsed, 1),
        "total_evaluations": len(results),
        "sizing": "FIXED quarter-Kelly 0.25",
        "top_20": [{
            "rank": i + 1,
            "trial": e["trial"],
            "label": e.get("label", ""),
            "params": e["params"],
            "result": e["result"],
        } for i, e in enumerate(sorted_all[:20])],
        "baseline": {
            "params": {k: v for k, v in BASELINE_PARAMS.items()
                       if k not in ("CITY_SIGMA", "CITY_BIAS", "CITY_OVERRIDES")},
            "result": results[0]["result"],  # First entry is always baseline
        },
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2, default=str)

    log(f"\nResults saved to {RESULTS_FILE}")

    # Print production-ready params for best result
    best = top5[0]
    log(f"\n{'='*78}")
    log(f"BEST PARAMS (ready for production):")
    log(f"{'='*78}")
    for k, v in sorted(best["params"].items()):
        log(f"  {k} = {v}")


if __name__ == "__main__":
    main()
