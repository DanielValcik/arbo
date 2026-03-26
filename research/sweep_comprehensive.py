"""Comprehensive Autoresearch Sweep on Real + Goldsky Data.

5-phase parameter optimization using all available real Polymarket data:
  Phase 1: Broad random search (2000 trials)
  Phase 2: Fine-tuning around top 20 (1000 trials)
  Phase 3: City exclusion & overrides (500 trials)
  Phase 4: Multi-entry-timing evaluation (500 trials)
  Phase 5: Walk-forward validation on top 10 (100 trials)

Data sources:
  - CLOB /prices-history (30 days, hourly)
  - Goldsky on-chain trades (all history, hourly aggregated)

Usage:
    python3 research/sweep_comprehensive.py
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
KELLY_FRACTION = 0.25
MAX_POSITION_PCT = 0.05

RESULTS_PATH = Path(__file__).parent / "data" / "sweep_comprehensive_results.json"
LOG_PATH = Path(__file__).parent / "sweep_comprehensive_log.txt"

CITY_COORDS = {
    "chicago": (41.88, -87.63), "london": (51.51, -0.13),
    "seoul": (37.57, 126.98), "ankara": (39.93, 32.86),
    "sao_paulo": (-23.55, -46.63), "miami": (25.76, -80.19),
    "paris": (48.86, 2.35), "dallas": (32.78, -96.80),
    "seattle": (47.61, -122.33), "munich": (48.14, 11.58),
    "lucknow": (26.85, 80.95), "tel_aviv": (32.09, 34.78),
    "tokyo": (35.68, 139.65), "los_angeles": (34.05, -118.24),
    "nyc": (40.71, -74.01), "toronto": (43.65, -79.38),
    "buenos_aires": (-34.60, -58.38), "atlanta": (33.75, -84.39),
    "wellington": (-41.29, 174.78), "dc": (38.91, -77.04),
}

ALL_CITIES = set(CITY_COORDS.keys())

# ── Forecast cache ────────────────────────────────────────────────────

_forecast_cache: dict[str, dict[str, float]] = {}


def _load_forecasts(events):
    """Pre-fetch all needed forecasts from Open-Meteo archive."""
    needed = set()
    for ev in events:
        if ev.city and ev.city in CITY_COORDS and ev.target_date:
            needed.add((ev.city, ev.target_date))

    needed = {
        (c, d) for c, d in needed
        if d not in _forecast_cache.get(c, {})
    }

    if not needed:
        return

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
    print(f"Forecasts cached: {total} city-dates", flush=True)


# ── Evaluation ────────────────────────────────────────────────────────


def evaluate_params(
    db: PriceHistoryDB,
    events: list,
    params: dict,
    entry_hours: float = 24,
) -> dict:
    """Evaluate a parameter set on real price data."""
    min_edge = params["min_edge"]
    max_price = params["max_price"]
    min_price = params["min_price"]
    min_prob = params["min_prob"]
    min_volume = params.get("min_volume", 100)
    kelly_raw_cap = params.get("kelly_raw_cap", 0.40)
    prob_sharpening = params.get("prob_sharpening", 1.05)
    shrinkage = params.get("shrinkage", 0.03)
    max_edge = params.get("max_edge", 0.90)
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

        city_min_edge = city_overrides.get(ev.city, {}).get(
            "min_edge", min_edge
        )
        city_max_price = city_overrides.get(ev.city, {}).get(
            "max_price", max_price
        )
        city_min_price = city_overrides.get(ev.city, {}).get(
            "min_price", min_price
        )

        for bucket in snap.buckets:
            price = snap.prices.get(bucket.token_id)
            if price is None or price <= 0.001:
                continue

            sigma = strategy._get_sigma(0, ev.city)
            cdf = lambda x, fc=forecast_temp, s=sigma: strategy._normal_cdf(
                x, fc, s
            )

            if bucket.low_c is None and bucket.high_c is not None:
                raw_prob = cdf(bucket.high_c)
            elif bucket.high_c is None and bucket.low_c is not None:
                raw_prob = 1.0 - cdf(bucket.low_c)
            elif bucket.low_c is not None and bucket.high_c is not None:
                raw_prob = cdf(bucket.high_c) - cdf(bucket.low_c)
            else:
                continue

            raw_prob = raw_prob * (1.0 - shrinkage) + 0.125 * shrinkage

            if prob_sharpening != 1.0 and raw_prob > 0:
                raw_prob = raw_prob ** prob_sharpening

            our_prob = raw_prob
            edge = our_prob - price

            if edge < city_min_edge or edge > max_edge:
                continue
            if price < city_min_price or price > city_max_price:
                continue
            if our_prob < min_prob:
                continue
            if bucket.volume < min_volume:
                continue

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

    daily_vals = list(daily_pnl.values())
    if len(daily_vals) >= 2:
        mean_d = sum(daily_vals) / len(daily_vals)
        var_d = sum((d - mean_d) ** 2 for d in daily_vals) / (len(daily_vals) - 1)
        std_d = math.sqrt(var_d) if var_d > 0 else 0.001
        sharpe = (mean_d / std_d) * math.sqrt(365) if std_d > 0 else 0
    else:
        sharpe = 0

    city_pnl = defaultdict(float)
    city_trades = defaultdict(int)
    for t in trades:
        city_pnl[t["city"]] += t["pnl"]
        city_trades[t["city"]] += 1
    unprofitable_cities = [c for c, p in city_pnl.items() if p < 0]

    pnl_factor = total_pnl / INITIAL_CAPITAL
    dd_factor = max(0, 1.0 - max_dd * 2)
    trade_factor = min(n_trades / 50, 2.0)
    city_penalty = max(0, 1.0 - len(unprofitable_cities) * 0.1)
    sharpe_factor = min(max(sharpe, 0) / 5.0, 2.0)

    score = (
        pnl_factor * 100
        * (1 + sharpe_factor)
        * dd_factor
        * trade_factor
        * city_penalty
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


def evaluate_multi_entry(
    db: PriceHistoryDB,
    events: list,
    params: dict,
    entry_hours_list: list[float] = [48, 36, 24, 12, 6],
) -> dict:
    """Evaluate across multiple entry timings and average results.

    This tests robustness: good params should work at different entry times.
    """
    results = []
    for hours in entry_hours_list:
        r = evaluate_params(db, events, params, entry_hours=hours)
        results.append(r)

    if not results:
        return {"score": 0, "trades": 0, "pnl": 0}

    avg_score = sum(r["score"] for r in results) / len(results)
    min_score = min(r["score"] for r in results)
    total_trades = sum(r["trades"] for r in results)
    avg_pnl = sum(r["pnl"] for r in results) / len(results)
    avg_wr = sum(r["win_rate"] for r in results) / len(results)
    max_dd = max(r["max_dd"] for r in results)
    avg_sharpe = sum(r["sharpe"] for r in results) / len(results)

    # Robust score: penalize if some entry times are bad
    robust_score = avg_score * 0.6 + min_score * 0.4

    return {
        "score": round(robust_score, 2),
        "avg_score": round(avg_score, 2),
        "min_score": round(min_score, 2),
        "trades": total_trades,
        "avg_pnl": round(avg_pnl, 2),
        "avg_wr": round(avg_wr, 1),
        "max_dd": round(max_dd, 2),
        "avg_sharpe": round(avg_sharpe, 2),
        "per_entry": {str(h): {"score": r["score"], "trades": r["trades"],
                                "pnl": r["pnl"]}
                      for h, r in zip(entry_hours_list, results)},
    }


def walk_forward_validate(
    db: PriceHistoryDB,
    all_events: list,
    params: dict,
    n_folds: int = 3,
) -> dict:
    """Walk-forward cross-validation.

    Splits events chronologically into n_folds. Trains (selects best entry)
    on each fold, tests on the next fold. Reports out-of-sample performance.
    """
    dated_events = [e for e in all_events if e.target_date]
    dated_events.sort(key=lambda e: e.target_date)

    if len(dated_events) < 20:
        return {"score": 0, "oos_pnl": 0, "n_folds": 0}

    fold_size = len(dated_events) // n_folds
    oos_results = []

    for fold in range(n_folds - 1):
        test_start = (fold + 1) * fold_size
        test_end = (fold + 2) * fold_size if fold < n_folds - 2 else len(dated_events)
        test_events = dated_events[test_start:test_end]

        if len(test_events) < 5:
            continue

        result = evaluate_params(db, test_events, params, entry_hours=24)
        oos_results.append(result)

    if not oos_results:
        return {"score": 0, "oos_pnl": 0, "n_folds": 0}

    avg_oos_pnl = sum(r["pnl"] for r in oos_results) / len(oos_results)
    avg_oos_score = sum(r["score"] for r in oos_results) / len(oos_results)
    total_trades = sum(r["trades"] for r in oos_results)

    return {
        "score": round(avg_oos_score, 2),
        "oos_pnl": round(avg_oos_pnl, 2),
        "n_folds": len(oos_results),
        "total_trades": total_trades,
        "folds": [{"pnl": r["pnl"], "trades": r["trades"],
                    "wr": r["win_rate"], "score": r["score"]}
                   for r in oos_results],
    }


# ── Search space ──────────────────────────────────────────────────────


def random_params(rng: random.Random) -> dict:
    """Generate random parameter set — wider search space than V1."""
    return {
        "min_edge": rng.choice([0.005, 0.01, 0.015, 0.02, 0.025, 0.03,
                                 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15]),
        "max_edge": rng.choice([0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]),
        "max_price": rng.choice([0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
                                  0.55, 0.60, 0.70, 0.80, 0.90]),
        "min_price": rng.choice([0.01, 0.02, 0.03, 0.04, 0.05, 0.06,
                                  0.08, 0.10, 0.12, 0.15, 0.20, 0.25]),
        "min_prob": rng.choice([0.01, 0.02, 0.03, 0.05, 0.08, 0.10,
                                 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]),
        "min_volume": rng.choice([0, 10, 25, 50, 100, 200, 500, 1000, 2000]),
        "kelly_raw_cap": rng.choice([0.10, 0.15, 0.20, 0.25, 0.30, 0.35,
                                      0.40, 0.50, 0.60]),
        "prob_sharpening": rng.choice([0.70, 0.80, 0.85, 0.90, 0.95, 1.0,
                                        1.05, 1.10, 1.15, 1.20, 1.30, 1.50]),
        "shrinkage": rng.choice([0.0, 0.01, 0.02, 0.03, 0.05, 0.08,
                                  0.10, 0.15, 0.20, 0.25]),
        "excluded_cities": set(),
        "city_overrides": {},
    }


def perturb_params(base: dict, rng: random.Random, n_changes: int = 0) -> dict:
    """Create a perturbation of a parameter set."""
    p = dict(base)
    p["excluded_cities"] = set(p.get("excluded_cities", set()))
    p["city_overrides"] = dict(p.get("city_overrides", {}))

    keys = ["min_edge", "max_edge", "max_price", "min_price", "min_prob",
            "min_volume", "kelly_raw_cap", "prob_sharpening", "shrinkage"]
    n_perturb = n_changes if n_changes > 0 else rng.randint(1, 3)
    for key in rng.sample(keys, min(n_perturb, len(keys))):
        val = p[key]
        if key == "min_volume":
            p[key] = max(0, val + rng.choice([-500, -200, -100, -50, 50, 100, 200, 500]))
        elif key in ("prob_sharpening",):
            p[key] = round(max(0.5, val + rng.choice([-0.15, -0.10, -0.05, 0.05, 0.10, 0.15])), 2)
        elif key == "shrinkage":
            p[key] = round(max(0, min(0.40, val + rng.choice([-0.05, -0.03, -0.02, 0.02, 0.03, 0.05]))), 2)
        elif key == "kelly_raw_cap":
            p[key] = round(max(0.05, min(0.70, val + rng.choice([-0.10, -0.05, 0.05, 0.10]))), 2)
        elif key == "max_edge":
            p[key] = round(max(0.15, min(0.95, val + rng.choice([-0.10, -0.05, 0.05, 0.10]))), 2)
        else:
            p[key] = round(
                max(0.005, val + rng.choice([-0.04, -0.03, -0.02, -0.01, 0.01, 0.02, 0.03, 0.04])),
                3,
            )
    return p


# ── Main sweep ────────────────────────────────────────────────────────

def serialize_params(params: dict) -> dict:
    """Make params JSON-serializable."""
    p = dict(params)
    if isinstance(p.get("excluded_cities"), set):
        p["excluded_cities"] = sorted(list(p["excluded_cities"]))
    if "city_overrides" in p:
        p["city_overrides"] = dict(p["city_overrides"])
    return p


def main():
    t_start = time.time()
    rng = random.Random(2026)

    db = PriceHistoryDB()
    stats = db.stats()
    events = db.get_events(with_prices=True)

    def log(msg):
        print(msg, flush=True)
        with open(LOG_PATH, "a") as f:
            f.write(msg + "\n")

    log(f"\n{'=' * 70}")
    log(f"COMPREHENSIVE SWEEP — Real Polymarket + Goldsky Data")
    log(f"{'=' * 70}")
    log(f"Events with prices: {len(events)}")
    log(f"Database stats: {json.dumps(stats, indent=2, default=str)}")
    log(f"Start: {datetime.now().isoformat()}")
    log(f"{'=' * 70}\n")

    # Load forecasts
    _load_forecasts(events)

    all_results = []
    best_score = -999
    best_params = None
    best_result = None

    def update_best(result, params, trial_num, phase):
        nonlocal best_score, best_params, best_result
        result["params"] = serialize_params(params)
        result["trial"] = trial_num
        result["phase"] = phase
        all_results.append(result)

        if result["score"] > best_score:
            best_score = result["score"]
            best_params = params
            best_result = result
            log(f"  Trial {trial_num:>4}: score={result['score']:>8.2f}  "
                f"trades={result['trades']:>3}  "
                f"WR={result['win_rate']:>5.1f}%  "
                f"PnL=${result['pnl']:>8.2f}  "
                f"DD={result['max_dd']:>5.2f}%  "
                f"Sharpe={result['sharpe']:>6.2f}  ★ NEW BEST")
            return True
        return False

    # ══════════════════════════════════════════════════════════════════
    # Phase 1: Broad Random Search (2000 trials)
    # ══════════════════════════════════════════════════════════════════
    log("Phase 1: Broad Random Search (2000 trials)")
    log("-" * 50)

    for trial in range(2000):
        params = random_params(rng)
        result = evaluate_params(db, events, params)
        update_best(result, params, trial, 1)

        if (trial + 1) % 200 == 0:
            elapsed = time.time() - t_start
            log(f"  ... {trial + 1}/2000 done, best score: {best_score:.2f} "
                f"({elapsed/60:.1f}min)")

    # ══════════════════════════════════════════════════════════════════
    # Phase 2: Fine-tuning around top 20 (1000 trials)
    # ══════════════════════════════════════════════════════════════════
    log(f"\nPhase 2: Fine-tuning (1000 trials around top 20)")
    log("-" * 50)

    phase1_sorted = sorted(all_results, key=lambda r: -r["score"])[:20]
    log(f"  Top 5 scores from P1: {[r['score'] for r in phase1_sorted[:5]]}")

    for i in range(1000):
        base = phase1_sorted[i % 20]
        base_params = dict(base["params"])
        base_params["excluded_cities"] = set(
            base_params.get("excluded_cities", [])
        )
        base_params["city_overrides"] = dict(
            base_params.get("city_overrides", {})
        )

        # Alternate between small and medium perturbations
        n_changes = 1 if i % 3 == 0 else (2 if i % 3 == 1 else 3)
        params = perturb_params(base_params, rng, n_changes)
        result = evaluate_params(db, events, params)
        update_best(result, params, 2000 + i, 2)

        if (i + 1) % 200 == 0:
            elapsed = time.time() - t_start
            log(f"  ... {i + 1}/1000 done, best score: {best_score:.2f} "
                f"({elapsed/60:.1f}min)")

    # ══════════════════════════════════════════════════════════════════
    # Phase 3: City Optimization (500 trials)
    # ══════════════════════════════════════════════════════════════════
    log(f"\nPhase 3: City optimization (500 trials)")
    log("-" * 50)

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
    else:
        losing_cities = []
        winning_cities = []

    for i in range(500):
        params = dict(best_params) if best_params else random_params(rng)
        params["excluded_cities"] = set(params.get("excluded_cities", set()))
        params["city_overrides"] = dict(params.get("city_overrides", {}))

        if i < 80 and losing_cities:
            # Exclude subsets of losing cities
            n_exclude = rng.randint(1, min(len(losing_cities), 6))
            params["excluded_cities"] = set(
                rng.sample(losing_cities, n_exclude)
            )
        elif i < 200 and winning_cities:
            # Optimize overrides for winning cities
            params["excluded_cities"] = set(losing_cities)
            for city in rng.sample(
                winning_cities, min(len(winning_cities), 4)
            ):
                params["city_overrides"][city] = {
                    "max_price": rng.choice([0.40, 0.45, 0.50, 0.55, 0.60,
                                              0.70, 0.80]),
                    "min_edge": rng.choice([0.005, 0.01, 0.015, 0.02, 0.03,
                                             0.05]),
                    "min_price": rng.choice([0.01, 0.02, 0.03, 0.05, 0.08]),
                }
        elif i < 350:
            # Perturb best + exclude losers
            params = perturb_params(params, rng)
            params["excluded_cities"] = set(losing_cities)
        else:
            # Random combos of exclusions + perturbation
            all_cities_list = list(ALL_CITIES)
            n_exc = rng.randint(0, 8)
            params["excluded_cities"] = set(
                rng.sample(all_cities_list, min(n_exc, len(all_cities_list)))
            )
            params = perturb_params(params, rng)

        result = evaluate_params(db, events, params)
        update_best(result, params, 3000 + i, 3)

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t_start
            log(f"  ... {i + 1}/500 done, best score: {best_score:.2f} "
                f"({elapsed/60:.1f}min)")

    # ══════════════════════════════════════════════════════════════════
    # Phase 4: Multi-Entry-Timing Evaluation (500 trials)
    # ══════════════════════════════════════════════════════════════════
    log(f"\nPhase 4: Multi-entry-timing robustness (500 trials)")
    log("-" * 50)

    # Take top 50 from all phases and evaluate at multiple entry times
    all_sorted = sorted(all_results, key=lambda r: -r["score"])[:50]
    multi_results = []

    for i, base in enumerate(all_sorted):
        base_params = dict(base["params"])
        base_params["excluded_cities"] = set(
            base_params.get("excluded_cities", [])
        )
        base_params["city_overrides"] = dict(
            base_params.get("city_overrides", {})
        )

        # Evaluate at multiple entry times
        multi_r = evaluate_multi_entry(db, events, base_params)
        multi_r["params"] = serialize_params(base_params)
        multi_r["base_trial"] = base["trial"]
        multi_results.append(multi_r)

        if (i + 1) % 10 == 0:
            log(f"  ... {i + 1}/50 multi-entry done")

    # Also try perturbations of top 5 multi-entry results
    multi_sorted = sorted(multi_results, key=lambda r: -r["score"])[:5]
    log(f"  Top 5 multi-entry scores: {[r['score'] for r in multi_sorted]}")

    for i in range(450):
        base = multi_sorted[i % 5]
        base_params = dict(base["params"])
        base_params["excluded_cities"] = set(
            base_params.get("excluded_cities", [])
        )
        base_params["city_overrides"] = dict(
            base_params.get("city_overrides", {})
        )

        params = perturb_params(base_params, rng)
        multi_r = evaluate_multi_entry(db, events, params)
        multi_r["params"] = serialize_params(params)
        multi_r["trial"] = 3500 + 50 + i
        multi_results.append(multi_r)

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t_start
            log(f"  ... {50 + i + 1}/500 done, best multi-score: "
                f"{max(r['score'] for r in multi_results):.2f} ({elapsed/60:.1f}min)")

    # ══════════════════════════════════════════════════════════════════
    # Phase 5: Walk-Forward Validation on Top 10 (100 trials)
    # ══════════════════════════════════════════════════════════════════
    log(f"\nPhase 5: Walk-forward validation (top 10 + perturbations)")
    log("-" * 50)

    # Take top 10 from multi-entry and validate with walk-forward
    top10_multi = sorted(multi_results, key=lambda r: -r["score"])[:10]
    wf_results = []

    for i, base in enumerate(top10_multi):
        base_params = dict(base["params"])
        base_params["excluded_cities"] = set(
            base_params.get("excluded_cities", [])
        )
        base_params["city_overrides"] = dict(
            base_params.get("city_overrides", {})
        )

        wf_r = walk_forward_validate(db, events, base_params, n_folds=3)
        wf_r["params"] = serialize_params(base_params)
        wf_r["multi_score"] = base["score"]
        wf_results.append(wf_r)
        log(f"  WF #{i+1}: multi_score={base['score']:.2f}  "
            f"oos_pnl=${wf_r['oos_pnl']:.2f}  "
            f"oos_score={wf_r['score']:.2f}  "
            f"trades={wf_r.get('total_trades', 0)}")

    # Perturbations of best walk-forward
    wf_sorted = sorted(wf_results, key=lambda r: -r["score"])[:3]
    for i in range(90):
        base = wf_sorted[i % 3]
        base_params = dict(base["params"])
        base_params["excluded_cities"] = set(
            base_params.get("excluded_cities", [])
        )
        base_params["city_overrides"] = dict(
            base_params.get("city_overrides", {})
        )

        params = perturb_params(base_params, rng, n_changes=1)
        wf_r = walk_forward_validate(db, events, params, n_folds=3)
        wf_r["params"] = serialize_params(params)

        # Also get single-entry score for comparison
        single = evaluate_params(db, events, params)
        wf_r["single_score"] = single["score"]
        wf_r["single_pnl"] = single["pnl"]
        wf_results.append(wf_r)

    # ══════════════════════════════════════════════════════════════════
    # Save all results
    # ══════════════════════════════════════════════════════════════════
    output = {
        "meta": {
            "timestamp": datetime.now().isoformat(),
            "events": len(events),
            "data_stats": stats,
            "total_trials": len(all_results),
            "phases": "5 (random, finetune, city, multi-entry, walk-forward)",
        },
        "best_single_entry": best_result,
        "top10_multi_entry": sorted(multi_results, key=lambda r: -r["score"])[:10],
        "walk_forward": sorted(wf_results, key=lambda r: -r["score"])[:10],
        "all_single_entry": all_results,
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)

    elapsed = time.time() - t_start

    # ── Final Report ──
    log(f"\n{'=' * 70}")
    log(f"COMPREHENSIVE SWEEP COMPLETE — {elapsed / 60:.1f} minutes")
    log(f"{'=' * 70}")

    log(f"\n─── Best Single-Entry (24h) ───")
    log(f"  Score: {best_result['score']:.2f}")
    log(f"  Trades: {best_result['trades']}")
    log(f"  Win rate: {best_result['win_rate']}%")
    log(f"  PnL: ${best_result['pnl']:,.2f} ({best_result['pnl']/INITIAL_CAPITAL*100:.1f}%)")
    log(f"  Max DD: {best_result['max_dd']}%")
    log(f"  Sharpe: {best_result['sharpe']}")

    bp = best_result["params"]
    log(f"\n  Parameters:")
    for k in ["min_edge", "max_edge", "max_price", "min_price", "min_prob",
              "min_volume", "kelly_raw_cap", "prob_sharpening", "shrinkage"]:
        log(f"    {k}: {bp.get(k, '—')}")
    log(f"    excluded_cities: {bp.get('excluded_cities', [])}")
    log(f"    city_overrides: {bp.get('city_overrides', {})}")

    log(f"\n  Per-city PnL:")
    city_pnl = best_result.get("city_pnl", {})
    for city in sorted(city_pnl, key=lambda c: -city_pnl[c]):
        ct = best_result.get("city_trades", {}).get(city, 0)
        log(f"    {city:<16} ${city_pnl[city]:>10.2f}  ({ct} trades)")

    log(f"\n─── Best Multi-Entry (robust) ───")
    best_multi = sorted(multi_results, key=lambda r: -r["score"])[0]
    log(f"  Robust score: {best_multi['score']:.2f}")
    log(f"  Avg PnL: ${best_multi['avg_pnl']:.2f}")
    log(f"  Total trades (5 timings): {best_multi['trades']}")
    if "per_entry" in best_multi:
        for hours, data in sorted(best_multi["per_entry"].items()):
            log(f"    {hours}h: score={data['score']:.2f}  "
                f"trades={data['trades']}  PnL=${data['pnl']:.2f}")

    log(f"\n─── Walk-Forward Validation ───")
    best_wf = sorted(wf_results, key=lambda r: -r["score"])[0]
    log(f"  OOS score: {best_wf['score']:.2f}")
    log(f"  OOS PnL: ${best_wf['oos_pnl']:.2f}")
    log(f"  Folds: {best_wf.get('n_folds', 0)}")
    if "folds" in best_wf:
        for i, fold in enumerate(best_wf["folds"]):
            log(f"    Fold {i+1}: PnL=${fold['pnl']:.2f}  "
                f"trades={fold['trades']}  WR={fold['wr']}%")

    bp_wf = best_wf["params"]
    log(f"\n  Walk-forward best parameters:")
    for k in ["min_edge", "max_edge", "max_price", "min_price", "min_prob",
              "min_volume", "kelly_raw_cap", "prob_sharpening", "shrinkage"]:
        log(f"    {k}: {bp_wf.get(k, '—')}")
    log(f"    excluded_cities: {bp_wf.get('excluded_cities', [])}")
    log(f"    city_overrides: {bp_wf.get('city_overrides', {})}")

    log(f"\n  Top 5 single-entry results:")
    top5 = sorted(all_results, key=lambda r: -r["score"])[:5]
    for i, r in enumerate(top5):
        log(f"    #{i+1}: score={r['score']:.2f}  trades={r['trades']}  "
            f"WR={r['win_rate']}%  PnL=${r['pnl']:.2f}  "
            f"DD={r['max_dd']}%  (trial {r['trial']}, phase {r['phase']})")

    log(f"\n  Results saved to: {RESULTS_PATH}")
    log(f"  Total time: {elapsed / 60:.1f} minutes")

    db.close()


if __name__ == "__main__":
    main()
