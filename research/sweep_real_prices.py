"""Autoresearch Sweep on Real Polymarket Price History.

Optimizes Strategy C parameters using actual Polymarket CLOB prices
instead of synthetic market maker. This finds parameters that work
in real market conditions.

Data source: price_history.sqlite (downloaded via download_price_history.py)
Scoring: profit-focused composite score for real market conditions.

Usage:
    python3 research/sweep_real_prices.py
"""

import json
import math
import os
import random
import ssl
import sys
import time
import urllib.request
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

try:
    import certifi
    SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    SSL_CTX = ssl.create_default_context()
    SSL_CTX.check_hostname = False
    SSL_CTX.verify_mode = ssl.CERT_NONE

sys.path.insert(0, str(Path(__file__).parent))
import strategy_experiment as strategy
from price_history_db import PriceHistoryDB

# ── Constants ─────────────────────────────────────────────────────────

INITIAL_CAPITAL = 1000.0
SLIPPAGE_PCT = 0.005
GAS_COST_USD = 0.007
KELLY_FRACTION = 0.25  # Quarter-Kelly — FIXED, not tunable
MAX_POSITION_PCT = 0.05

RESULTS_PATH = Path(__file__).parent / "data" / "sweep_real_results.json"
LOG_PATH = Path(__file__).parent / "sweep_real_log.txt"

CITY_COORDS = {
    "chicago": (41.88, -87.63), "london": (51.51, -0.13),
    "seoul": (37.57, 126.98), "ankara": (39.93, 32.86),
    "sao_paulo": (-23.55, -46.63), "miami": (25.76, -80.19),
    "paris": (48.86, 2.35), "dallas": (32.78, -96.80),
    "seattle": (47.61, -122.33), "munich": (48.14, 11.58),
    "lucknow": (26.85, 80.95), "tel_aviv": (32.09, 34.78),
    "tokyo": (35.68, 139.65), "los_angeles": (34.05, -118.24),
}

# All cities that could potentially trade
ALL_CITIES = set(CITY_COORDS.keys())

# ── Forecast cache ────────────────────────────────────────────────────

_forecast_cache: dict[str, dict[str, float]] = {}


def _load_forecasts(events):
    """Pre-fetch all needed forecasts from Open-Meteo archive."""
    needed = set()
    for ev in events:
        if ev.city and ev.city in CITY_COORDS and ev.target_date:
            needed.add((ev.city, ev.target_date))

    # Remove already cached
    needed = {
        (c, d) for c, d in needed
        if d not in _forecast_cache.get(c, {})
    }

    if not needed:
        return

    # Group by city for efficient batch fetching
    by_city = defaultdict(list)
    for city, dt in needed:
        by_city[city].append(dt)

    for city, dates in by_city.items():
        lat, lon = CITY_COORDS[city]
        dates_sorted = sorted(dates)
        min_d, max_d = dates_sorted[0], dates_sorted[-1]

        url = (
            f"https://archive-api.open-meteo.com/v1/archive"
            f"?latitude={lat}&longitude={lon}"
            f"&start_date={min_d}&end_date={max_d}"
            f"&daily=temperature_2m_max&timezone=auto"
        )
        req = urllib.request.Request(
            url, headers={"User-Agent": "ArboResearch/1.0"}
        )
        try:
            with urllib.request.urlopen(req, timeout=30, context=SSL_CTX) as resp:
                data = json.loads(resp.read().decode())
            daily = data.get("daily", {})
            for i, dt_str in enumerate(daily.get("time", [])):
                val = daily["temperature_2m_max"][i]
                if val is not None:
                    _forecast_cache.setdefault(city, {})[dt_str] = val
        except Exception:
            pass
        time.sleep(0.05)

    total = sum(len(v) for v in _forecast_cache.values())
    print(f"Forecasts cached: {total} city-dates")


# ── Evaluation ────────────────────────────────────────────────────────


def evaluate_params(
    db: PriceHistoryDB,
    events: list,
    params: dict,
    entry_hours: float = 24,
) -> dict:
    """Evaluate a parameter set on real price data.

    Returns dict with: trades, wins, pnl, win_rate, max_dd, sharpe, etc.
    """
    min_edge = params["min_edge"]
    max_price = params["max_price"]
    min_price = params["min_price"]
    min_prob = params["min_prob"]
    min_volume = params.get("min_volume", 100)
    kelly_raw_cap = params.get("kelly_raw_cap", 0.40)
    prob_sharpening = params.get("prob_sharpening", 1.05)
    shrinkage = params.get("shrinkage", 0.03)
    excluded = params.get("excluded_cities", set())
    city_overrides = params.get("city_overrides", {})

    capital = INITIAL_CAPITAL
    peak_capital = capital
    max_dd = 0.0
    daily_pnl = defaultdict(float)
    trades = []

    for ev in events:
        if not ev.city or not ev.target_date:
            continue
        if ev.city in excluded:
            continue
        if ev.city not in CITY_COORDS:
            continue

        forecast_temp = _forecast_cache.get(ev.city, {}).get(ev.target_date)
        if forecast_temp is None:
            continue

        snap = db.get_price_snapshot(ev.event_id, entry_hours)
        if not snap or not snap.prices:
            continue

        # Per-city overrides
        city_min_edge = city_overrides.get(ev.city, {}).get(
            "min_edge", min_edge
        )
        city_max_price = city_overrides.get(ev.city, {}).get(
            "max_price", max_price
        )

        for bucket in snap.buckets:
            price = snap.prices.get(bucket.token_id)
            if price is None or price <= 0.001:
                continue

            # Calculate probability with current params
            sigma = strategy._get_sigma(0, ev.city)
            cdf = lambda x, fc=forecast_temp, s=sigma: strategy._normal_cdf(
                x, fc, s
            )

            if bucket.low_c is None and bucket.high_c is not None:
                raw_prob = cdf(bucket.high_c)
            elif bucket.high_c is None and bucket.low_c is not None:
                raw_prob = 1.0 - cdf(bucket.low_c)
            elif (
                bucket.low_c is not None and bucket.high_c is not None
            ):
                raw_prob = cdf(bucket.high_c) - cdf(bucket.low_c)
            else:
                continue

            # Apply shrinkage
            raw_prob = raw_prob * (1.0 - shrinkage) + 0.125 * shrinkage

            # Apply sharpening
            if prob_sharpening != 1.0 and raw_prob > 0:
                raw_prob = raw_prob ** prob_sharpening

            our_prob = raw_prob
            edge = our_prob - price

            # Quality gate
            if edge < city_min_edge:
                continue
            if price < min_price or price > city_max_price:
                continue
            if our_prob < min_prob:
                continue
            if bucket.volume < min_volume:
                continue

            # Sizing (quarter-Kelly)
            prob_est = price + edge
            if prob_est <= 0 or prob_est >= 1:
                continue
            odds = (1.0 / price) - 1.0
            if odds <= 0:
                continue
            kelly_raw = (prob_est * odds - (1 - prob_est)) / odds
            if kelly_raw <= 0:
                continue
            kelly_raw = min(kelly_raw, kelly_raw_cap)
            kelly_adj = kelly_raw * KELLY_FRACTION
            size = capital * kelly_adj
            size = min(size, capital * MAX_POSITION_PCT)
            if size < 1.0:
                continue
            size = round(size, 2)

            # PnL
            fill = price * (1 + SLIPPAGE_PCT)
            if bucket.won:
                pnl = size * (1.0 / fill - 1.0) - GAS_COST_USD
            else:
                pnl = -size - GAS_COST_USD

            capital += pnl
            peak_capital = max(peak_capital, capital)
            dd = (peak_capital - capital) / peak_capital if peak_capital > 0 else 0
            max_dd = max(max_dd, dd)

            daily_pnl[ev.target_date] += pnl

            trades.append({
                "city": ev.city, "won": bucket.won, "pnl": pnl,
                "edge": edge, "price": price, "size": size,
                "date": ev.target_date,
            })

    n_trades = len(trades)
    if n_trades == 0:
        return {
            "trades": 0, "wins": 0, "pnl": 0, "win_rate": 0,
            "max_dd": 0, "sharpe": 0, "score": 0,
            "final_capital": INITIAL_CAPITAL,
        }

    wins = sum(1 for t in trades if t["won"])
    total_pnl = sum(t["pnl"] for t in trades)
    win_rate = wins / n_trades

    # Sharpe on daily PnL
    daily_vals = list(daily_pnl.values())
    if len(daily_vals) >= 2:
        mean_d = sum(daily_vals) / len(daily_vals)
        var_d = sum((d - mean_d) ** 2 for d in daily_vals) / (len(daily_vals) - 1)
        std_d = math.sqrt(var_d) if var_d > 0 else 0.001
        sharpe = (mean_d / std_d) * math.sqrt(365) if std_d > 0 else 0
    else:
        sharpe = 0

    # Per-city PnL
    city_pnl = defaultdict(float)
    city_trades = defaultdict(int)
    for t in trades:
        city_pnl[t["city"]] += t["pnl"]
        city_trades[t["city"]] += 1
    unprofitable_cities = [c for c, p in city_pnl.items() if p < 0]

    # Composite score: profit × consistency
    # Reward: high PnL, high Sharpe, many trades
    # Penalize: high drawdown, unprofitable cities
    pnl_factor = total_pnl / INITIAL_CAPITAL  # return as fraction
    dd_factor = max(0, 1.0 - max_dd * 2)  # penalize >50% DD heavily
    trade_factor = min(n_trades / 50, 2.0)  # reward up to 100 trades
    city_penalty = max(0, 1.0 - len(unprofitable_cities) * 0.1)
    sharpe_factor = min(max(sharpe, 0) / 5.0, 2.0)  # normalize

    score = (
        pnl_factor * 100  # Primary: profit
        * (1 + sharpe_factor)  # Boost for consistency
        * dd_factor  # Penalize drawdown
        * trade_factor  # Reward volume
        * city_penalty  # Penalize losing cities
    )

    return {
        "trades": n_trades,
        "wins": wins,
        "pnl": round(total_pnl, 2),
        "win_rate": round(win_rate * 100, 1),
        "max_dd": round(max_dd * 100, 2),
        "sharpe": round(sharpe, 2),
        "score": round(score, 2),
        "final_capital": round(INITIAL_CAPITAL + total_pnl, 2),
        "unprofitable_cities": unprofitable_cities,
        "city_pnl": {c: round(p, 2) for c, p in city_pnl.items()},
        "city_trades": dict(city_trades),
    }


# ── Search space ──────────────────────────────────────────────────────


def random_params(rng: random.Random) -> dict:
    """Generate random parameter set from search space."""
    return {
        "min_edge": rng.choice([0.01, 0.02, 0.03, 0.04, 0.05, 0.06,
                                 0.08, 0.10, 0.12, 0.15]),
        "max_price": rng.choice([0.30, 0.35, 0.40, 0.45, 0.50, 0.55,
                                  0.60, 0.70, 0.80]),
        "min_price": rng.choice([0.02, 0.03, 0.05, 0.08, 0.10, 0.12,
                                  0.15, 0.20]),
        "min_prob": rng.choice([0.05, 0.08, 0.10, 0.15, 0.20, 0.25,
                                 0.30, 0.40]),
        "min_volume": rng.choice([0, 50, 100, 200, 500, 1000]),
        "kelly_raw_cap": rng.choice([0.15, 0.20, 0.25, 0.30, 0.35,
                                      0.40, 0.50]),
        "prob_sharpening": rng.choice([0.80, 0.90, 0.95, 1.0, 1.05,
                                        1.10, 1.15, 1.20, 1.30]),
        "shrinkage": rng.choice([0.0, 0.01, 0.03, 0.05, 0.08, 0.10,
                                  0.15, 0.20]),
        "excluded_cities": set(),  # Phase 1: no exclusions
        "city_overrides": {},
    }


def perturb_params(base: dict, rng: random.Random) -> dict:
    """Create a small perturbation of a parameter set."""
    p = dict(base)
    p["excluded_cities"] = set(p.get("excluded_cities", set()))
    p["city_overrides"] = dict(p.get("city_overrides", {}))

    # Perturb 1-3 parameters
    keys = ["min_edge", "max_price", "min_price", "min_prob",
            "min_volume", "kelly_raw_cap", "prob_sharpening", "shrinkage"]
    n_perturb = rng.randint(1, 3)
    for key in rng.sample(keys, n_perturb):
        val = p[key]
        if key == "min_volume":
            p[key] = max(0, val + rng.choice([-200, -100, -50, 50, 100, 200]))
        elif key in ("prob_sharpening",):
            p[key] = round(val + rng.choice([-0.10, -0.05, 0.05, 0.10]), 2)
        elif key == "shrinkage":
            p[key] = round(max(0, val + rng.choice([-0.03, -0.02, 0.02, 0.03])), 2)
        elif key == "kelly_raw_cap":
            p[key] = round(
                max(0.05, min(0.60, val + rng.choice([-0.05, 0.05]))), 2
            )
        else:
            p[key] = round(
                max(0.01, val + rng.choice([-0.03, -0.02, -0.01, 0.01, 0.02, 0.03])),
                2,
            )
    return p


# ── Main sweep ────────────────────────────────────────────────────────


def main():
    t_start = time.time()
    rng = random.Random(42)

    db = PriceHistoryDB()
    events = db.get_events(with_prices=True)
    print(f"Events with price data: {len(events)}")

    # Load forecasts
    _load_forecasts(events)

    all_results = []
    best_score = -999
    best_params = None
    best_result = None

    def log(msg):
        print(msg, flush=True)
        with open(LOG_PATH, "a") as f:
            f.write(msg + "\n")

    log(f"\n{'=' * 70}")
    log(f"SWEEP ON REAL POLYMARKET PRICES")
    log(f"Events: {len(events)}, Start: {datetime.now().isoformat()}")
    log(f"{'=' * 70}\n")

    # ── Phase 1: Random Search (500 trials) ──
    log("Phase 1: Random Search (500 trials)")
    log("-" * 50)

    for trial in range(500):
        params = random_params(rng)
        result = evaluate_params(db, events, params)
        result["params"] = {k: v if not isinstance(v, set) else list(v)
                            for k, v in params.items()}
        result["trial"] = trial
        result["phase"] = 1
        all_results.append(result)

        if result["score"] > best_score:
            best_score = result["score"]
            best_params = params
            best_result = result
            log(f"  Trial {trial:>3}: score={result['score']:>8.2f}  "
                f"trades={result['trades']:>3}  "
                f"WR={result['win_rate']:>5.1f}%  "
                f"PnL=${result['pnl']:>8.2f}  "
                f"DD={result['max_dd']:>5.2f}%  "
                f"Sharpe={result['sharpe']:>6.2f}  ★ NEW BEST")

        if (trial + 1) % 50 == 0:
            log(f"  ... {trial + 1}/500 done, best score: {best_score:.2f}")

    # ── Phase 2: Fine-tuning around top 10 (300 trials) ──
    log(f"\nPhase 2: Fine-tuning (300 trials around top 10)")
    log("-" * 50)

    # Get top 10 from phase 1
    phase1_sorted = sorted(all_results, key=lambda r: -r["score"])[:10]
    log(f"  Top 10 scores: {[r['score'] for r in phase1_sorted]}")

    trial_offset = 500
    for i in range(300):
        base = phase1_sorted[i % 10]
        base_params = dict(base["params"])
        base_params["excluded_cities"] = set(
            base_params.get("excluded_cities", [])
        )
        base_params["city_overrides"] = dict(
            base_params.get("city_overrides", {})
        )

        params = perturb_params(base_params, rng)
        result = evaluate_params(db, events, params)
        result["params"] = {k: v if not isinstance(v, set) else list(v)
                            for k, v in params.items()}
        result["trial"] = trial_offset + i
        result["phase"] = 2
        all_results.append(result)

        if result["score"] > best_score:
            best_score = result["score"]
            best_params = params
            best_result = result
            log(f"  Trial {trial_offset + i}: score={result['score']:>8.2f}  "
                f"trades={result['trades']:>3}  "
                f"WR={result['win_rate']:>5.1f}%  "
                f"PnL=${result['pnl']:>8.2f}  "
                f"DD={result['max_dd']:>5.2f}%  ★ NEW BEST")

        if (i + 1) % 50 == 0:
            log(f"  ... {i + 1}/300 done, best score: {best_score:.2f}")

    # ── Phase 3: City exclusions & overrides (200 trials) ──
    log(f"\nPhase 3: City optimization (200 trials)")
    log("-" * 50)

    # Find unprofitable cities from best result
    if best_result:
        city_pnl = best_result.get("city_pnl", {})
        losing_cities = sorted(
            [c for c, p in city_pnl.items() if p < 0],
            key=lambda c: city_pnl[c],
        )
        winning_cities = sorted(
            [c for c, p in city_pnl.items() if p > 0],
            key=lambda c: -city_pnl[c],
        )
        log(f"  Losing cities: {losing_cities}")
        log(f"  Winning cities: {winning_cities}")

    trial_offset = 800
    for i in range(200):
        params = dict(best_params)
        params["excluded_cities"] = set(params.get("excluded_cities", set()))
        params["city_overrides"] = dict(params.get("city_overrides", {}))

        # Try different city exclusion combos
        if i < 50 and losing_cities:
            # Exclude subsets of losing cities
            n_exclude = rng.randint(1, min(len(losing_cities), 5))
            params["excluded_cities"] = set(
                rng.sample(losing_cities, n_exclude)
            )
        elif i < 100 and winning_cities:
            # Widen max_price for winning cities
            params["excluded_cities"] = set(losing_cities)  # exclude all losers
            for city in rng.sample(
                winning_cities, min(len(winning_cities), 3)
            ):
                params["city_overrides"][city] = {
                    "max_price": rng.choice([0.50, 0.55, 0.60, 0.70, 0.80]),
                    "min_edge": rng.choice([0.01, 0.02, 0.03, 0.05]),
                }
        elif i < 150:
            # Perturb best + exclude losers
            params = perturb_params(params, rng)
            params["excluded_cities"] = set(losing_cities)
        else:
            # Random combos of exclusions
            all_cities_list = list(ALL_CITIES)
            n_exc = rng.randint(0, 6)
            params["excluded_cities"] = set(
                rng.sample(all_cities_list, n_exc)
            )
            params = perturb_params(params, rng)

        result = evaluate_params(db, events, params)
        result["params"] = {k: v if not isinstance(v, set) else list(v)
                            for k, v in params.items()}
        result["trial"] = trial_offset + i
        result["phase"] = 3
        all_results.append(result)

        if result["score"] > best_score:
            best_score = result["score"]
            best_params = params
            best_result = result
            log(f"  Trial {trial_offset + i}: score={result['score']:>8.2f}  "
                f"trades={result['trades']:>3}  "
                f"WR={result['win_rate']:>5.1f}%  "
                f"PnL=${result['pnl']:>8.2f}  "
                f"excluded={list(params['excluded_cities'])}  ★ NEW BEST")

        if (i + 1) % 50 == 0:
            log(f"  ... {i + 1}/200 done, best score: {best_score:.2f}")

    # ── Save results ──
    # Sort serializable
    for r in all_results:
        p = r.get("params", {})
        if "excluded_cities" in p and isinstance(p["excluded_cities"], set):
            p["excluded_cities"] = list(p["excluded_cities"])
        if "city_overrides" in p:
            p["city_overrides"] = dict(p["city_overrides"])

    with open(RESULTS_PATH, "w") as f:
        json.dump(
            {"best": best_result, "all": all_results},
            f, indent=2, default=str,
        )

    elapsed = time.time() - t_start

    # ── Final report ──
    log(f"\n{'=' * 70}")
    log(f"SWEEP COMPLETE — {elapsed / 60:.1f} minutes")
    log(f"{'=' * 70}")
    log(f"\n  Best score: {best_score:.2f}")
    log(f"  Trades: {best_result['trades']}")
    log(f"  Win rate: {best_result['win_rate']}%")
    log(f"  PnL: ${best_result['pnl']:,.2f} "
        f"({best_result['pnl'] / INITIAL_CAPITAL * 100:.1f}%)")
    log(f"  Max DD: {best_result['max_dd']}%")
    log(f"  Sharpe: {best_result['sharpe']}")
    log(f"  Final capital: ${best_result['final_capital']:,.2f}")

    log(f"\n  Best parameters:")
    bp = best_result["params"]
    for k in ["min_edge", "max_price", "min_price", "min_prob",
              "min_volume", "kelly_raw_cap", "prob_sharpening", "shrinkage"]:
        log(f"    {k}: {bp[k]}")
    log(f"    excluded_cities: {bp.get('excluded_cities', [])}")
    log(f"    city_overrides: {bp.get('city_overrides', {})}")

    log(f"\n  Per-city PnL:")
    city_pnl = best_result.get("city_pnl", {})
    for city in sorted(city_pnl, key=lambda c: -city_pnl[c]):
        ct = best_result.get("city_trades", {}).get(city, 0)
        log(f"    {city:<16} ${city_pnl[city]:>10.2f}  ({ct} trades)")

    log(f"\n  Top 5 results:")
    top5 = sorted(all_results, key=lambda r: -r["score"])[:5]
    for i, r in enumerate(top5):
        log(f"    #{i + 1}: score={r['score']:.2f}  trades={r['trades']}  "
            f"WR={r['win_rate']}%  PnL=${r['pnl']:.2f}  "
            f"DD={r['max_dd']}%  (trial {r['trial']}, phase {r['phase']})")

    log(f"\n  Results saved to: {RESULTS_PATH}")
    log(f"  Log saved to: {LOG_PATH}")

    db.close()


if __name__ == "__main__":
    main()
