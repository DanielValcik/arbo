"""
Full Parameter Sweep V3 — VOLUME OPTIMIZED
============================================

Same as V2 but with modified scoring that rewards HIGH TRADE VOLUME.
Goal: maximize capital turnover while staying profitable.

Scoring: composite = avg_sharpe * (total_trades/100) * dd_factor * consistency
  - Uses linear trade factor (not sqrt) — strongly rewards more trades
  - Minimum 500 trades (over 15 months) or score = 0
  - Minimum 70% win rate or score = 0

Usage: python3 research/sweep_full_v3_volume.py
Output: research/sweep_full_v3_results.json
"""

import copy
import json
import itertools
import math
import random
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import backtest_harness as harness

# Load data once
harness.download_historical_data()
DATA = harness.load_data()
NORMALS = harness.compute_monthly_normals(DATA)

RESULTS_FILE = Path(__file__).parent / "sweep_full_v3_results.json"
LOG_FILE = Path(__file__).parent / "sweep_full_v3_log.txt"

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

# ── Volume scoring thresholds ──
MIN_TRADES = 500       # Minimum 500 trades over 15 months (~33/month, ~1/day)
MIN_WIN_RATE = 70.0    # Must win at least 70%
MIN_SHARPE = 5.0       # Must have positive Sharpe


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

    shrinkage = params["SHRINKAGE_WEIGHT"]
    kelly_cap = params["KELLY_RAW_CAP"]
    sharpening = params["PROB_SHARPENING"]

    if "CITY_SIGMA" in params:
        s.CITY_SIGMA = params["CITY_SIGMA"]
    if "CITY_BIAS" in params:
        s.CITY_BIAS = params["CITY_BIAS"]
    if "CITY_OVERRIDES" in params:
        s.CITY_OVERRIDES = params["CITY_OVERRIDES"]

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


def volume_composite_score(avg_sharpe, total_trades, max_dd, profitable_windows,
                            total_windows, avg_win_rate):
    """Composite score that rewards HIGH VOLUME trading.

    Key difference from V2: linear trade factor (not sqrt), plus minimum thresholds.
    """
    if total_trades < MIN_TRADES:
        return 0.0
    if avg_win_rate < MIN_WIN_RATE:
        return 0.0
    if avg_sharpe < MIN_SHARPE:
        return 0.0

    # Linear trade factor — strongly rewards more trades
    trade_factor = total_trades / 100.0
    dd_factor = max(0, 1.0 - max_dd / 50.0)
    consistency = profitable_windows / total_windows

    return avg_sharpe * trade_factor * dd_factor * consistency


def evaluate(params):
    """Run walk-forward evaluation with given params."""
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

    # Calculate volume-optimized composite score
    vol_score = volume_composite_score(
        results["avg_sharpe"], results["total_trades"],
        results["max_drawdown_pct"], results["profitable_windows"],
        results["total_windows"], results["avg_win_rate"],
    )

    return {
        "volume_score": round(vol_score, 4),
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
        "trades_per_month": round(results["total_trades"] / 15, 1),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1: Random Search — biased toward high-volume params
# ═══════════════════════════════════════════════════════════════════════════════

def random_params(rng):
    """Generate random params biased toward high trade volume."""
    p = copy.deepcopy(BASELINE_PARAMS)

    # Bias toward LOWER min_edge (more trades)
    p["MIN_EDGE"] = rng.choice([0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.12])
    # Bias toward WIDER price range (more trades)
    p["MAX_PRICE"] = rng.choice([0.43, 0.45, 0.48, 0.50, 0.55, 0.60, 0.65, 0.70])
    p["MIN_PRICE"] = rng.choice([0.10, 0.15, 0.20, 0.25, 0.28, 0.30])
    # Bias toward LOWER min forecast prob (more trades)
    p["MIN_FORECAST_PROB"] = rng.choice([0.40, 0.45, 0.50, 0.55, 0.58, 0.60, 0.62])
    p["PROB_SHARPENING"] = rng.choice([0.95, 0.98, 1.0, 1.02, 1.05, 1.08, 1.10, 1.15])
    p["SHRINKAGE_WEIGHT"] = rng.choice([0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10])
    p["KELLY_RAW_CAP"] = rng.choice([0.25, 0.30, 0.35, 0.40, 0.50, 0.60])
    p["CONVICTION_RATIO"] = rng.choice([0.0, 1.2, 1.3, 1.4, 1.5])
    # Lower volume thresholds = more trades
    p["MIN_VOLUME"] = rng.choice([200, 500, 1000, 2000])
    p["MIN_LIQUIDITY"] = rng.choice([50, 100, 200, 500])

    return p


def phase1_random_search(n_trials=600):
    """Phase 1: Random search biased toward high volume."""
    log(f"=== PHASE 1: Random Search — VOLUME OPTIMIZED ({n_trials} trials) ===")
    log(f"Minimum trades: {MIN_TRADES} | Min win rate: {MIN_WIN_RATE}% | Min Sharpe: {MIN_SHARPE}")

    rng = random.Random(42)
    results = []

    # Baseline
    log("Evaluating baseline...")
    baseline_result = evaluate(BASELINE_PARAMS)
    results.append({
        "trial": 0, "params": {k: v for k, v in BASELINE_PARAMS.items()
                                if k not in ("CITY_SIGMA", "CITY_BIAS", "CITY_OVERRIDES")},
        "result": baseline_result, "label": "BASELINE",
    })
    log(f"  BASELINE: vol_score={baseline_result['volume_score']:.2f} "
        f"sharpe={baseline_result['avg_sharpe']:.2f} "
        f"trades={baseline_result['total_trades']} ({baseline_result['trades_per_month']}/mo) "
        f"wr={baseline_result['avg_win_rate']:.1f}% "
        f"dd={baseline_result['max_drawdown_pct']:.2f}%")

    best_score = baseline_result["volume_score"]
    best_trial = 0
    qualifying = 1 if best_score > 0 else 0

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

        if result["volume_score"] > 0:
            qualifying += 1

        marker = ""
        if result["volume_score"] > best_score:
            best_score = result["volume_score"]
            best_trial = i
            marker = " *** NEW BEST ***"

        if i % 25 == 0 or marker:
            log(f"  Trial {i}/{n_trials}: vol_score={result['volume_score']:.2f} "
                f"sharpe={result['avg_sharpe']:.2f} "
                f"trades={result['total_trades']} ({result['trades_per_month']}/mo) "
                f"wr={result['avg_win_rate']:.1f}% "
                f"dd={result['max_drawdown_pct']:.2f}%{marker}")

    log(f"Phase 1 done. Best vol_score: {best_score:.2f} (trial {best_trial}). "
        f"{qualifying}/{n_trials+1} qualifying configs.")
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2: Fine-tune top-10
# ═══════════════════════════════════════════════════════════════════════════════

def fine_tune_params(base_params, rng):
    """Small perturbation around a good parameter set."""
    p = copy.deepcopy(base_params)
    param_choices = ["MIN_EDGE", "MAX_PRICE", "MIN_PRICE", "MIN_FORECAST_PROB",
                     "PROB_SHARPENING", "SHRINKAGE_WEIGHT", "KELLY_RAW_CAP",
                     "MIN_VOLUME", "MIN_LIQUIDITY"]
    n_perturb = rng.randint(1, 3)

    for _ in range(n_perturb):
        param = rng.choice(param_choices)
        if param == "MIN_EDGE":
            p[param] = max(0.02, min(0.15, p[param] + rng.uniform(-0.015, 0.015)))
        elif param == "MAX_PRICE":
            p[param] = max(0.35, min(0.75, p[param] + rng.uniform(-0.03, 0.03)))
        elif param == "MIN_PRICE":
            p[param] = max(0.05, min(0.35, p[param] + rng.uniform(-0.03, 0.03)))
        elif param == "MIN_FORECAST_PROB":
            p[param] = max(0.35, min(0.70, p[param] + rng.uniform(-0.03, 0.03)))
        elif param == "PROB_SHARPENING":
            p[param] = max(0.90, min(1.20, p[param] + rng.uniform(-0.03, 0.03)))
        elif param == "SHRINKAGE_WEIGHT":
            p[param] = max(0.0, min(0.15, p[param] + rng.uniform(-0.02, 0.02)))
        elif param == "KELLY_RAW_CAP":
            p[param] = max(0.20, min(0.70, p[param] + rng.uniform(-0.05, 0.05)))
        elif param == "MIN_VOLUME":
            p[param] = max(100, min(5000, p[param] + rng.uniform(-300, 300)))
        elif param == "MIN_LIQUIDITY":
            p[param] = max(25, min(1000, p[param] + rng.uniform(-100, 100)))

    for k in ["MIN_EDGE", "MAX_PRICE", "MIN_PRICE", "MIN_FORECAST_PROB",
              "PROB_SHARPENING", "SHRINKAGE_WEIGHT", "KELLY_RAW_CAP"]:
        p[k] = round(p[k], 3)
    for k in ["MIN_VOLUME", "MIN_LIQUIDITY"]:
        p[k] = round(p[k], 0)

    return p


def phase2_fine_tune(phase1_results, n_per_top=40):
    """Phase 2: Fine-tune around top-10 from Phase 1."""
    valid = [r for r in phase1_results if r["result"]["volume_score"] > 0]
    sorted_results = sorted(valid, key=lambda x: -x["result"]["volume_score"])
    top10 = sorted_results[:10]

    log(f"\n=== PHASE 2: Fine-tuning top-{len(top10)} ({n_per_top} each) ===")
    for i, entry in enumerate(top10):
        r = entry["result"]
        log(f"  Top {i+1}: trial={entry['trial']} vol_score={r['volume_score']:.2f} "
            f"trades={r['total_trades']} wr={r['avg_win_rate']:.1f}%")

    rng = random.Random(456)
    results = list(phase1_results)
    best_score = top10[0]["result"]["volume_score"]
    improvements = 0

    trial_offset = len(phase1_results)
    for rank, top_entry in enumerate(top10):
        base_params = copy.deepcopy(BASELINE_PARAMS)
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

            if result["volume_score"] > best_score:
                best_score = result["volume_score"]
                improvements += 1
                log(f"  FT top{rank+1} #{j}: vol_score={result['volume_score']:.2f} "
                    f"trades={result['total_trades']} wr={result['avg_win_rate']:.1f}% *** BEST ***")

    log(f"Phase 2 done. {improvements} improvements. Best: {best_score:.2f}")
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3: Per-city override optimization
# ═══════════════════════════════════════════════════════════════════════════════

def phase3_city_overrides(all_results):
    """Phase 3: City overrides on best global params."""
    valid = [r for r in all_results if r["result"]["volume_score"] > 0]
    sorted_results = sorted(valid, key=lambda x: -x["result"]["volume_score"])
    best_entry = sorted_results[0]

    log(f"\n=== PHASE 3: Per-city Override Optimization ===")
    log(f"Starting from best: vol_score={best_entry['result']['volume_score']:.2f} "
        f"trades={best_entry['result']['total_trades']}")

    best_global = copy.deepcopy(BASELINE_PARAMS)
    for k, v in best_entry["params"].items():
        best_global[k] = v

    # Evaluate with NO overrides
    no_override_params = copy.deepcopy(best_global)
    no_override_params["CITY_OVERRIDES"] = {}
    no_override_result = evaluate(no_override_params)
    log(f"No overrides: vol_score={no_override_result['volume_score']:.2f} "
        f"trades={no_override_result['total_trades']}")

    city_pnl = no_override_result["city_pnl"]
    log(f"Per-city PnL (no overrides):")
    for city, pnl in city_pnl.items():
        log(f"  {city:>16}: ${pnl:>10.2f}")

    unprofitable = [c for c, p in city_pnl.items() if p < 0]
    marginal = [c for c, p in city_pnl.items() if 0 <= p < 200]
    strong = [c for c, p in city_pnl.items() if p >= 2000]

    log(f"Unprofitable: {unprofitable}")
    log(f"Marginal (< $200): {marginal}")
    log(f"Strong (> $2000): {strong}")

    results = list(all_results)
    trial_offset = len(all_results)
    best_score = best_entry["result"]["volume_score"]

    # Only exclude clearly unprofitable cities (we want volume!)
    log(f"\nTesting exclusions (only unprofitable cities)...")
    exclusion_scores = {}

    for city in unprofitable:
        params = copy.deepcopy(best_global)
        params["CITY_OVERRIDES"] = {}
        params["CITY_OVERRIDES"][city] = {"min_edge": 0.99}
        result = evaluate(params)
        exclusion_scores[city] = result["volume_score"]
        marker = " ***" if result["volume_score"] > best_score else ""
        log(f"  Exclude {city:>16}: vol_score={result['volume_score']:.2f} "
            f"trades={result['total_trades']}{marker}")
        results.append({
            "trial": trial_offset, "params": {k: v for k, v in params.items()
                                               if k not in ("CITY_SIGMA", "CITY_BIAS")},
            "result": result, "label": f"exclude_{city}",
        })
        trial_offset += 1

    helpful_exclusions = [c for c, s in exclusion_scores.items() if s > best_score]

    if helpful_exclusions:
        log(f"\nTesting exclusion combos: {helpful_exclusions}")
        for n in range(2, min(len(helpful_exclusions) + 1, 8)):
            for combo in itertools.combinations(helpful_exclusions, n):
                params = copy.deepcopy(best_global)
                params["CITY_OVERRIDES"] = {}
                for city in combo:
                    params["CITY_OVERRIDES"][city] = {"min_edge": 0.99}
                result = evaluate(params)
                marker = ""
                if result["volume_score"] > best_score:
                    best_score = result["volume_score"]
                    marker = " *** NEW BEST ***"
                combo_str = "+".join(combo)
                log(f"  Exclude [{combo_str}]: vol_score={result['volume_score']:.2f} "
                    f"trades={result['total_trades']}{marker}")
                results.append({
                    "trial": trial_offset, "params": {k: v for k, v in params.items()
                                                       if k not in ("CITY_SIGMA", "CITY_BIAS")},
                    "result": result, "label": f"exclude_combo",
                })
                trial_offset += 1

    # Widening for strong cities
    log(f"\nTesting widenings for strong cities...")
    best_so_far = sorted(results, key=lambda x: -x["result"]["volume_score"])[0]
    base_overrides = best_so_far.get("params", {}).get("CITY_OVERRIDES", {})

    for widen_price in [0.55, 0.60, 0.65, 0.70]:
        for n in range(1, min(len(strong) + 1, 6)):
            for combo in itertools.combinations(strong, n):
                params = copy.deepcopy(best_global)
                params["CITY_OVERRIDES"] = copy.deepcopy(base_overrides) if isinstance(base_overrides, dict) else {}
                for city in combo:
                    if params["CITY_OVERRIDES"].get(city, {}).get("min_edge") != 0.99:
                        params["CITY_OVERRIDES"][city] = {"max_price": widen_price}
                result = evaluate(params)
                marker = ""
                if result["volume_score"] > best_score:
                    best_score = result["volume_score"]
                    marker = " *** NEW BEST ***"
                if marker or n <= 2:
                    combo_str = "+".join(combo)
                    log(f"  Widen [{combo_str}] to {widen_price}: "
                        f"vol_score={result['volume_score']:.2f} "
                        f"trades={result['total_trades']}{marker}")
                results.append({
                    "trial": trial_offset, "params": {k: v for k, v in params.items()
                                                       if k not in ("CITY_SIGMA", "CITY_BIAS")},
                    "result": result, "label": f"widen_{widen_price}",
                })
                trial_offset += 1

    log(f"Phase 3 done. Best vol_score: {best_score:.2f}")
    return results


def main():
    t_start = time.time()
    # Clear log
    with open(LOG_FILE, "w") as f:
        f.write("")

    log(f"Starting VOLUME-OPTIMIZED sweep at {datetime.now().isoformat()}")
    log(f"Sizing FIXED: KELLY_FRACTION=0.25 (quarter-Kelly)")
    log(f"Min trades: {MIN_TRADES} | Min win rate: {MIN_WIN_RATE}% | Min Sharpe: {MIN_SHARPE}")
    log("")

    results = phase1_random_search(n_trials=600)
    results = phase2_fine_tune(results, n_per_top=40)
    results = phase3_city_overrides(results)

    # Final summary
    valid = [r for r in results if r["result"]["volume_score"] > 0]
    sorted_all = sorted(valid, key=lambda x: -x["result"]["volume_score"])
    top5 = sorted_all[:5]

    elapsed = time.time() - t_start
    log(f"\n{'='*78}")
    log(f"SWEEP COMPLETE — {len(results)} evaluations in {elapsed:.0f}s")
    log(f"{'='*78}")

    for i, entry in enumerate(top5):
        r = entry["result"]
        log(f"\nTop {i+1}: vol_score={r['volume_score']:.2f} "
            f"sharpe={r['avg_sharpe']:.2f} "
            f"trades={r['total_trades']} ({r['trades_per_month']}/mo) "
            f"wr={r['avg_win_rate']:.1f}% "
            f"dd={r['max_drawdown_pct']:.2f}% "
            f"pnl={r['avg_pnl_pct']:.1f}%")
        log(f"  Params: {json.dumps(entry['params'], default=str)}")

    # Save results
    output = {
        "sweep_date": datetime.now().isoformat(),
        "elapsed_seconds": round(elapsed, 1),
        "total_evaluations": len(results),
        "qualifying_configs": len(valid),
        "sizing": "FIXED quarter-Kelly 0.25",
        "scoring": "volume-optimized: sharpe * (trades/100) * dd_factor * consistency",
        "min_trades": MIN_TRADES,
        "min_win_rate": MIN_WIN_RATE,
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
            "result": results[0]["result"],
        },
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2, default=str)

    log(f"\nResults saved to {RESULTS_FILE}")

    best = top5[0]
    log(f"\n{'='*78}")
    log(f"BEST PARAMS (volume-optimized):")
    log(f"{'='*78}")
    for k, v in sorted(best["params"].items()):
        log(f"  {k} = {v}")


if __name__ == "__main__":
    main()
