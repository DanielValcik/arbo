"""Autoresearch — Karpathy-style autonomous parameter optimization.

Deep per-city optimization: each city gets its own tuned parameters.
Runs forever until Ctrl+C. Saves periodically.

Goals (EQUAL weight):
  1. Profitability — maximize risk-adjusted returns
  2. Capital turnover — capital must WORK, not sit idle

Data: 571K real price points, 420 days, 20 cities, $1000 starting capital.
Per-city overrides: min_edge, max_price, min_price, min_volume,
                    prob_sharpening, shrinkage, kelly_raw_cap

Usage:
    python3 research/autoresearch.py          # run forever (Ctrl+C to stop)
    python3 research/autoresearch.py --max 3000  # limit experiments
"""

import copy
import json
import random
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from experiment_framework import (
    CITY_COORDS,
    INITIAL_CAPITAL,
    experiment_score,
    load_forecasts,
    preload_data,
    simulate_portfolio,
    walk_forward_validate,
)
from price_history_db import PriceHistoryDB

# ── Constants ──
TRAIN_END = "2026-01-31"
ALL_CITIES = sorted(CITY_COORDS.keys())
TSV_PATH = Path(__file__).parent / "results.tsv"
RESULTS_JSON = Path(__file__).parent / "data" / "experiments" / "autoresearch_latest.json"
SAVE_EVERY = 100
REPORT_EVERY = 50

# ── Parameter ranges ──
GLOBAL_RANGES = {
    "min_edge":             (0.005, 0.25),
    "max_edge":             (0.20, 0.95),
    "max_price":            (0.20, 0.90),
    "min_price":            (0.03, 0.30),
    "min_prob":             (0.01, 0.50),
    "min_volume":           (0, 2000),
    "kelly_raw_cap":        (0.05, 0.60),
    "prob_sharpening":      (0.40, 1.60),
    "shrinkage":            (0.0, 0.25),
    "profit_take_threshold": (0.20, 4.00),
    "profit_take_min_hours": (1, 24),
    "min_hold_edge":        (0.0, 0.20),
    "prob_exit_floor":      (0.0, 0.50),
    "reentry_cooldown_h":   (1, 12),
    "stop_loss_pct":        (0.10, 1.0),
}

# Per-city override ranges
CITY_OVERRIDE_RANGES = {
    "min_edge":        (0.005, 0.20),
    "max_price":       (0.20, 0.90),
    "min_price":       (0.03, 0.25),
    "min_volume":      (0, 2000),
    "prob_sharpening": (0.50, 1.40),
    "shrinkage":       (0.0, 0.20),
    "kelly_raw_cap":   (0.05, 0.60),
}

INT_PARAMS = {"min_volume", "profit_take_min_hours", "reentry_cooldown_h"}

# ── Grid values ──
GLOBAL_GRID = {
    "min_edge":        [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15],
    "max_price":       [0.30, 0.40, 0.50, 0.60, 0.70, 0.80],
    "kelly_raw_cap":   [0.10, 0.15, 0.20, 0.25, 0.30, 0.40],
    "prob_sharpening": [0.70, 0.80, 0.85, 0.90, 0.95, 1.0, 1.05, 1.10],
    "shrinkage":       [0.0, 0.01, 0.02, 0.03, 0.05, 0.10],
    "min_hold_edge":   [0.0, 0.02, 0.04, 0.06, 0.08, 0.10],
    "prob_exit_floor": [0.0, 0.10, 0.20, 0.25, 0.30, 0.40],
    "profit_take_threshold": [0.50, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0],
    "min_price":       [0.03, 0.05, 0.08, 0.10, 0.15, 0.20],
}

CITY_GRID = {
    "min_edge":        [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15],
    "max_price":       [0.30, 0.40, 0.50, 0.55, 0.60, 0.70, 0.80],
    "min_price":       [0.03, 0.05, 0.08, 0.10, 0.15],
    "min_volume":      [0, 50, 100, 200, 500, 1000],
    "prob_sharpening": [0.70, 0.80, 0.85, 0.90, 0.95, 1.0, 1.10, 1.20],
    "shrinkage":       [0.0, 0.01, 0.02, 0.03, 0.05, 0.10],
    "kelly_raw_cap":   [0.10, 0.15, 0.20, 0.30, 0.40, 0.50],
}

# ── Baseline ──
BASELINE = {
    "min_edge": 0.10, "max_edge": 0.70, "max_price": 0.50, "min_price": 0.05,
    "min_prob": 0.06, "min_volume": 510, "kelly_raw_cap": 0.40,
    "prob_sharpening": 0.90, "shrinkage": 0.03,
    "excluded_cities": {"chicago", "seoul"},
    "city_overrides": {
        "seattle": {"max_price": 0.55, "min_edge": 0.005, "min_price": 0.15},
        "toronto": {"max_price": 0.4, "min_edge": 0.005, "min_price": 0.05},
        "atlanta": {"max_price": 0.55, "min_edge": 0.02, "min_price": 0.05},
        "ankara": {"max_price": 0.55, "min_edge": 0.005, "min_price": 0.08},
        "buenos_aires": {"max_price": 0.7, "min_edge": 0.02, "min_price": 0.05},
        "dallas": {"max_price": 0.8, "min_edge": 0.05, "min_price": 0.05},
        "wellington": {"max_price": 0.5, "min_edge": 0.02, "min_price": 0.05},
        "miami": {"max_price": 0.4, "min_edge": 0.05, "min_price": 0.08},
    },
    "exit_enabled": True, "min_hold_edge": 0.0, "prob_exit_floor": 0.25,
    "profit_take_enabled": True, "profit_take_threshold": 1.5,
    "profit_take_min_hours": 4, "reentry_enabled": False,
    "reentry_cooldown_h": 1, "stop_loss_pct": 1.0,
}


def log(msg: str):
    print(msg, flush=True)


def clone(p: dict) -> dict:
    p2 = copy.deepcopy(p)
    if isinstance(p2.get("excluded_cities"), list):
        p2["excluded_cities"] = set(p2["excluded_cities"])
    return p2


def rand_val(key: str, ranges: dict) -> float:
    lo, hi = ranges[key]
    if key in INT_PARAMS:
        return random.randint(int(lo), int(hi))
    return round(random.uniform(lo, hi), 4)


def perturb(key: str, current: float, scale: float, ranges: dict) -> float:
    lo, hi = ranges[key]
    delta = random.gauss(0, (hi - lo) * scale)
    val = max(lo, min(hi, current + delta))
    if key in INT_PARAMS:
        return int(round(val))
    return round(val, 4)


class AutoResearch:
    def __init__(self, max_experiments: int = 0):
        self.max_experiments = max_experiments
        self.n = 0
        self.best_score = 0.0
        self.best_params = None
        self.best_result = None
        self.history: list[dict] = []
        self.keeps = 0
        self.discards = 0
        self.sim = None
        self.stop = False
        self.stagnation = 0
        self._tried: set[int] = set()

        # Per-city solo results (from phase 1)
        self.city_solo: dict[str, dict] = {}

    def load_data(self):
        log("Loading data...")
        db = PriceHistoryDB()
        events = db.get_events(with_prices=True)
        log(f"  Events: {len(events)}")
        forecasts = load_forecasts(events)
        sim_all = preload_data(db, events, forecasts)
        self.sim_all = sim_all
        self.sim = sim_all.filter_events(max_date=TRAIN_END)
        log(f"  Train events: {len(self.sim.events)}")
        log(f"  All events: {len(self.sim_all.events)}")
        log(f"  Capital: ${INITIAL_CAPITAL}")

    def _hash(self, p: dict) -> int:
        parts = []
        for k in sorted(p.keys()):
            v = p[k]
            if isinstance(v, set):
                v = tuple(sorted(v))
            elif isinstance(v, dict):
                v = tuple(sorted((kk, tuple(sorted(vv.items())) if isinstance(vv, dict) else vv)
                                  for kk, vv in v.items()))
            parts.append((k, v))
        return hash(tuple(parts))

    def run_one(self, params: dict, desc: str, entry_hours: float = 24) -> dict:
        self.n += 1
        h = self._hash(params)
        if h in self._tried:
            self.n -= 1
            return {}
        self._tried.add(h)

        result = simulate_portfolio(
            self.sim, params, entry_hours=entry_hours,
            experiment_id=f"AR-{self.n:04d}",
        )
        experiment_score(result)

        r = {
            "id": f"AR-{self.n:04d}", "score": result.score,
            "trades": result.trades, "win_rate": result.win_rate,
            "pnl": result.total_pnl, "roi_pct": result.roi_pct,
            "dd": result.max_drawdown_pct, "sharpe": result.sharpe,
            "util": result.capital_utilization,
            "pph": result.avg_pnl_per_hour,
            "exits": result.total_exits, "saves": result.exit_saves,
            "regrets": result.exit_regrets,
            "params": params, "entry_hours": entry_hours,
            "city_results": result.city_results,
        }

        improved = r["score"] > self.best_score
        if improved:
            delta = r["score"] - self.best_score
            r["status"] = "keep"
            self.best_score = r["score"]
            self.best_params = clone(params)
            self.best_result = r
            self.keeps += 1
            self.stagnation = 0
            marker = f"★ KEEP (+{delta:.1f})"
        else:
            r["status"] = "discard"
            self.discards += 1
            self.stagnation += 1
            marker = "  discard"

        r["description"] = desc
        self.history.append(r)

        log(f"  {r['id']}: score={r['score']:6.1f}  t={r['trades']:4d}  "
            f"WR={r['win_rate']:5.1f}%  pnl=${r['pnl']:>9.0f}  "
            f"dd={r['dd']:5.1f}%  util={r['util']:5.1f}%  "
            f"Sh={r['sharpe']:5.1f}  {marker}  [{desc}]")

        # TSV
        hdr = "id\tscore\ttrades\twin_rate\tpnl\tdd\tsharpe\tutil\tstatus\tdescription\n"
        if not TSV_PATH.exists():
            TSV_PATH.write_text(hdr)
        with open(TSV_PATH, "a") as f:
            f.write(f"{r['id']}\t{r['score']:.2f}\t{r['trades']}\t"
                    f"{r['win_rate']:.1f}\t{r['pnl']:.0f}\t{r['dd']:.1f}\t"
                    f"{r['sharpe']:.1f}\t{r['util']:.1f}\t"
                    f"{r['status']}\t{desc}\n")

        if self.n % SAVE_EVERY == 0:
            self._save()
        if self.n % REPORT_EVERY == 0:
            log(f"\n  ── STATUS: {self.n} exp, {self.keeps} keeps, "
                f"best={self.best_score:.1f}, stagnation={self.stagnation} ──\n")
        return r

    def ok(self) -> bool:
        if self.stop:
            return False
        if self.max_experiments and self.n >= self.max_experiments:
            return False
        return True

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 1: Global parameter sweep + 2D/3D grids
    # ══════════════════════════════════════════════════════════════════════

    def phase_global_sweep(self):
        log("\n" + "=" * 70)
        log("PHASE 1: Global Parameter Sweep + Multi-Dim Grids")
        log("=" * 70)

        # 1a. Individual sweeps
        for param, values in GLOBAL_GRID.items():
            if not self.ok():
                return
            log(f"\n── {param} ──")
            for val in values:
                if not self.ok():
                    return
                p = clone(self.best_params)
                p[param] = val
                self.run_one(p, f"sweep {param}={val}")

        # 1b. 2D grids — all pairs of top 4 most impactful params
        top4 = ["min_edge", "prob_sharpening", "kelly_raw_cap", "prob_exit_floor"]
        for i in range(len(top4)):
            for j in range(i + 1, len(top4)):
                if not self.ok():
                    return
                k1, k2 = top4[i], top4[j]
                log(f"\n── 2D: {k1} × {k2} ──")
                for v1 in GLOBAL_GRID[k1]:
                    for v2 in GLOBAL_GRID[k2]:
                        if not self.ok():
                            return
                        p = clone(self.best_params)
                        p[k1] = v1
                        p[k2] = v2
                        self.run_one(p, f"2D {k1}={v1} {k2}={v2}")

        # 1c. 3D grid — min_edge × prob_sharpening × kelly
        log("\n── 3D: min_edge × prob_sharpening × kelly ──")
        for me in [0.04, 0.06, 0.08, 0.10, 0.12]:
            for ps in [0.80, 0.85, 0.90, 0.95, 1.0]:
                for kc in [0.10, 0.15, 0.20, 0.30, 0.40]:
                    if not self.ok():
                        return
                    p = clone(self.best_params)
                    p["min_edge"] = me
                    p["prob_sharpening"] = ps
                    p["kelly_raw_cap"] = kc
                    self.run_one(p, f"3D me={me} ps={ps} kc={kc}")

        # 1d. Entry + exit joint grid
        log("\n── Entry+Exit joint grid ──")
        for me in [0.06, 0.08, 0.10, 0.12]:
            for mhe in [0.0, 0.02, 0.04, 0.06]:
                for pef in [0.10, 0.20, 0.25, 0.30, 0.40]:
                    for pt in [0.75, 1.0, 1.5, 2.0]:
                        if not self.ok():
                            return
                        p = clone(self.best_params)
                        p["min_edge"] = me
                        p["min_hold_edge"] = mhe
                        p["prob_exit_floor"] = pef
                        p["profit_take_threshold"] = pt
                        self.run_one(p, f"e+x me={me} mhe={mhe} pef={pef} pt={pt}")

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 2: Per-city profiling — test each city solo, find best params
    # ══════════════════════════════════════════════════════════════════════

    def phase_city_profiling(self):
        log("\n" + "=" * 70)
        log("PHASE 2: Per-City Profiling (solo + per-city param optimization)")
        log("=" * 70)

        # 2a. Test each city solo with default params
        log("\n── Solo city test ──")
        for city in ALL_CITIES:
            if not self.ok():
                return
            p = clone(self.best_params)
            p["excluded_cities"] = set(ALL_CITIES) - {city}
            p["city_overrides"] = {}
            r = self.run_one(p, f"solo: {city}")
            if r:
                self.city_solo[city] = r

        # Rank
        ranked = sorted(self.city_solo.items(),
                        key=lambda x: (-x[1].get("score", 0)))
        log("\n  City ranking:")
        for city, r in ranked:
            log(f"    {city:20s}: score={r['score']:6.1f}  "
                f"pnl=${r['pnl']:>8.0f}  trades={r['trades']}")

        active = [c for c, r in ranked if r.get("trades", 0) >= 3 and r.get("pnl", 0) > 0]
        inactive = [c for c in ALL_CITIES if c not in active]
        log(f"\n  Active ({len(active)}): {active}")
        log(f"  Inactive ({len(inactive)}): {inactive}")

        # 2b. For each active city, optimize its override parameters
        log("\n── Per-city parameter optimization ──")
        for city in active:
            if not self.ok():
                return
            log(f"\n  ── Optimizing {city} ──")

            # Sweep each city override param
            for param, values in CITY_GRID.items():
                if not self.ok():
                    return
                for val in values:
                    if not self.ok():
                        return
                    p = clone(self.best_params)
                    p.setdefault("city_overrides", {})[city] = (
                        {**p.get("city_overrides", {}).get(city, {}), param: val}
                    )
                    self.run_one(p, f"{city}: {param}={val}")

            # 2D combos for this city: min_edge × max_price
            for me in [0.005, 0.01, 0.02, 0.05, 0.08, 0.10]:
                for mp in [0.40, 0.50, 0.60, 0.70, 0.80]:
                    if not self.ok():
                        return
                    p = clone(self.best_params)
                    existing = p.get("city_overrides", {}).get(city, {})
                    p.setdefault("city_overrides", {})[city] = {
                        **existing, "min_edge": me, "max_price": mp,
                    }
                    self.run_one(p, f"{city}: me={me} mp={mp}")

            # 3D: min_edge × max_price × prob_sharpening
            for me in [0.005, 0.02, 0.05, 0.10]:
                for mp in [0.40, 0.55, 0.70]:
                    for ps in [0.80, 0.90, 1.0, 1.10]:
                        if not self.ok():
                            return
                        p = clone(self.best_params)
                        existing = p.get("city_overrides", {}).get(city, {})
                        p.setdefault("city_overrides", {})[city] = {
                            **existing, "min_edge": me,
                            "max_price": mp, "prob_sharpening": ps,
                        }
                        self.run_one(p, f"{city}: me={me} mp={mp} ps={ps}")

        # 2c. Try excluding inactive cities
        for city_set_name, city_set in [
            ("all active", set(inactive)),
            ("top-5", set(ALL_CITIES) - set([c for c, _ in ranked[:5]])),
            ("top-8", set(ALL_CITIES) - set([c for c, _ in ranked[:8]])),
            ("top-10", set(ALL_CITIES) - set([c for c, _ in ranked[:10]])),
        ]:
            if not self.ok():
                return
            p = clone(self.best_params)
            p["excluded_cities"] = city_set
            self.run_one(p, f"cities: {city_set_name}")

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 3: Deep per-city tuning — optimize kelly, sharpening, shrinkage
    #           per city in combination with global params
    # ══════════════════════════════════════════════════════════════════════

    def phase_deep_city_tuning(self):
        log("\n" + "=" * 70)
        log("PHASE 3: Deep Per-City Tuning (kelly, sharpening, shrinkage per city)")
        log("=" * 70)

        # Get cities that have >10 trades
        active = [c for c, r in self.city_solo.items()
                  if r.get("trades", 0) >= 5 and r.get("pnl", 0) > -50]

        for city in active:
            if not self.ok():
                return
            log(f"\n  ── Deep tuning: {city} ──")

            # Per-city kelly_raw_cap × prob_sharpening
            for kc in [0.10, 0.15, 0.20, 0.30, 0.40, 0.50]:
                for ps in [0.75, 0.85, 0.90, 0.95, 1.0, 1.10]:
                    if not self.ok():
                        return
                    p = clone(self.best_params)
                    existing = p.get("city_overrides", {}).get(city, {})
                    p.setdefault("city_overrides", {})[city] = {
                        **existing, "kelly_raw_cap": kc, "prob_sharpening": ps,
                    }
                    self.run_one(p, f"{city}: kc={kc} ps={ps}")

            # Per-city shrinkage × prob_sharpening
            for sh in [0.0, 0.01, 0.03, 0.05, 0.10]:
                for ps in [0.80, 0.90, 1.0, 1.10]:
                    if not self.ok():
                        return
                    p = clone(self.best_params)
                    existing = p.get("city_overrides", {}).get(city, {})
                    p.setdefault("city_overrides", {})[city] = {
                        **existing, "shrinkage": sh, "prob_sharpening": ps,
                    }
                    self.run_one(p, f"{city}: sh={sh} ps={ps}")

            # Full per-city 4D: min_edge × max_price × kelly × sharpening
            for me in [0.005, 0.02, 0.05, 0.10]:
                for mp in [0.40, 0.55, 0.70]:
                    for kc in [0.15, 0.25, 0.40]:
                        for ps in [0.85, 0.95, 1.05]:
                            if not self.ok():
                                return
                            p = clone(self.best_params)
                            existing = p.get("city_overrides", {}).get(city, {})
                            p.setdefault("city_overrides", {})[city] = {
                                **existing,
                                "min_edge": me, "max_price": mp,
                                "kelly_raw_cap": kc, "prob_sharpening": ps,
                            }
                            self.run_one(
                                p, f"{city}: me={me} mp={mp} kc={kc} ps={ps}")

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 4: Autonomous loop — random strategies, never stops
    # ══════════════════════════════════════════════════════════════════════

    def phase_autonomous_loop(self):
        log("\n" + "=" * 70)
        log("PHASE 4: Autonomous Loop (random strategies)")
        log("=" * 70)

        strategies = [
            (25, self._strat_perturb_global),
            (20, self._strat_perturb_city),
            (15, self._strat_grid_global),
            (10, self._strat_random_city_override),
            (10, self._strat_entry_exit_combo),
            (8, self._strat_multi_city_override),
            (5, self._strat_full_random),
            (4, self._strat_random_restart),
            (3, self._strat_flip_city),
        ]
        names = ["perturb_global", "perturb_city", "grid_global",
                 "random_city_ov", "entry_exit", "multi_city_ov",
                 "full_random", "restart", "flip_city"]
        weights = [w for w, _ in strategies]
        fns = [f for _, f in strategies]

        while self.ok():
            # Increase exploration when stagnating
            w = list(weights)
            if self.stagnation > 100:
                w[4] = 20  # entry_exit
                w[6] = 15  # full_random
                w[7] = 15  # restart
            elif self.stagnation > 200:
                w[6] = 25
                w[7] = 25

            idx = random.choices(range(len(fns)), weights=w, k=1)[0]
            params, desc = fns[idx]()
            self.run_one(params, desc)

    def _strat_perturb_global(self):
        p = clone(self.best_params)
        n = random.randint(1, 4)
        keys = random.sample([k for k in GLOBAL_RANGES if k not in
                               ("excluded_cities", "city_overrides")],
                              min(n, len(GLOBAL_RANGES)))
        ch = []
        for k in keys:
            old = p.get(k, GLOBAL_RANGES[k][0])
            new = perturb(k, old, 0.25, GLOBAL_RANGES)
            p[k] = new
            ch.append(f"{k}={new}")
        return p, f"perturb-{n}: {', '.join(ch[:3])}"

    def _strat_perturb_city(self):
        p = clone(self.best_params)
        city = random.choice(ALL_CITIES)
        existing = p.get("city_overrides", {}).get(city, {})
        n = random.randint(1, 3)
        keys = random.sample(list(CITY_OVERRIDE_RANGES.keys()), n)
        ch = []
        for k in keys:
            old = existing.get(k, p.get(k, CITY_OVERRIDE_RANGES[k][0]))
            new = perturb(k, old, 0.3, CITY_OVERRIDE_RANGES)
            existing[k] = new
            ch.append(f"{k}={new}")
        p.setdefault("city_overrides", {})[city] = existing
        return p, f"{city} perturb: {', '.join(ch)}"

    def _strat_grid_global(self):
        p = clone(self.best_params)
        n = random.randint(2, 3)
        keys = random.sample(list(GLOBAL_GRID.keys()), n)
        ch = []
        for k in keys:
            val = random.choice(GLOBAL_GRID[k])
            p[k] = val
            ch.append(f"{k}={val}")
        return p, f"grid-{n}: {', '.join(ch)}"

    def _strat_random_city_override(self):
        p = clone(self.best_params)
        city = random.choice(ALL_CITIES)
        n = random.randint(2, 4)
        keys = random.sample(list(CITY_GRID.keys()), n)
        ov = {}
        ch = []
        for k in keys:
            val = random.choice(CITY_GRID[k])
            ov[k] = val
            ch.append(f"{k}={val}")
        p.setdefault("city_overrides", {})[city] = {
            **p.get("city_overrides", {}).get(city, {}), **ov
        }
        return p, f"{city}: {', '.join(ch)}"

    def _strat_entry_exit_combo(self):
        p = clone(self.best_params)
        p["min_edge"] = random.choice(GLOBAL_GRID["min_edge"])
        p["min_hold_edge"] = random.choice(GLOBAL_GRID["min_hold_edge"])
        p["prob_exit_floor"] = random.choice(GLOBAL_GRID["prob_exit_floor"])
        p["profit_take_threshold"] = random.choice(GLOBAL_GRID["profit_take_threshold"])
        p["kelly_raw_cap"] = random.choice(GLOBAL_GRID["kelly_raw_cap"])
        return p, (f"e+x me={p['min_edge']} mhe={p['min_hold_edge']} "
                   f"pef={p['prob_exit_floor']} pt={p['profit_take_threshold']} "
                   f"kc={p['kelly_raw_cap']}")

    def _strat_multi_city_override(self):
        """Override 2-4 cities at once with random params."""
        p = clone(self.best_params)
        n_cities = random.randint(2, 4)
        cities = random.sample(ALL_CITIES, n_cities)
        desc_parts = []
        for city in cities:
            n_params = random.randint(1, 3)
            keys = random.sample(list(CITY_GRID.keys()), n_params)
            ov = {}
            for k in keys:
                ov[k] = random.choice(CITY_GRID[k])
            p.setdefault("city_overrides", {})[city] = {
                **p.get("city_overrides", {}).get(city, {}), **ov
            }
            desc_parts.append(f"{city}({n_params})")
        return p, f"multi-city: {', '.join(desc_parts)}"

    def _strat_full_random(self):
        p = clone(BASELINE)
        for k in GLOBAL_RANGES:
            p[k] = rand_val(k, GLOBAL_RANGES)
        p["exit_enabled"] = random.random() < 0.7
        p["profit_take_enabled"] = random.random() < 0.6
        p["reentry_enabled"] = random.random() < 0.3
        n_exclude = random.randint(0, 10)
        p["excluded_cities"] = set(random.sample(ALL_CITIES, n_exclude))
        p["city_overrides"] = {}
        return p, "full-random"

    def _strat_random_restart(self):
        p = clone(BASELINE)
        n = random.randint(3, 7)
        keys = random.sample(list(GLOBAL_RANGES.keys()), n)
        ch = []
        for k in keys:
            p[k] = rand_val(k, GLOBAL_RANGES)
            ch.append(f"{k}={p[k]}")
        return p, f"restart: {', '.join(ch[:4])}"

    def _strat_flip_city(self):
        p = clone(self.best_params)
        excluded = p.get("excluded_cities", set())
        if random.random() < 0.5 and excluded:
            city = random.choice(sorted(excluded))
            p["excluded_cities"].discard(city)
            return p, f"include: {city}"
        else:
            included = set(ALL_CITIES) - excluded
            if len(included) > 3:
                city = random.choice(sorted(included))
                p["excluded_cities"].add(city)
                return p, f"exclude: {city}"
        return p, "flip-noop"

    # ══════════════════════════════════════════════════════════════════════
    # MAIN
    # ══════════════════════════════════════════════════════════════════════

    def run(self):
        t_start = time.time()

        log("=" * 70)
        log("AUTORESEARCH — Deep Per-City Optimization")
        log(f"Start: {datetime.now().isoformat()}")
        log(f"Data: 571K price points, 20 cities, ${INITIAL_CAPITAL} capital")
        log(f"Goals: Profitability + Capital Turnover (EQUAL weight)")
        log(f"Per-city overrides: min_edge, max_price, min_price, min_volume,")
        log(f"                    prob_sharpening, shrinkage, kelly_raw_cap")
        log(f"Max experiments: {self.max_experiments or 'unlimited (Ctrl+C)'}")
        log("=" * 70)

        signal.signal(signal.SIGINT, lambda *_: setattr(self, 'stop', True))
        signal.signal(signal.SIGTERM, lambda *_: setattr(self, 'stop', True))

        self.load_data()

        # Baseline
        log("\n── BASELINE ──")
        self.run_one(BASELINE, "baseline (production)")
        log(f"  Baseline score: {self.best_score:.1f}")

        # Phase 1: Global sweep + multi-dim grids
        self.phase_global_sweep()
        log(f"\n  After Phase 1: best={self.best_score:.1f} ({self.n} experiments)")

        # Phase 2: Per-city profiling and optimization
        self.phase_city_profiling()
        log(f"\n  After Phase 2: best={self.best_score:.1f} ({self.n} experiments)")

        # Phase 3: Deep per-city tuning
        self.phase_deep_city_tuning()
        log(f"\n  After Phase 3: best={self.best_score:.1f} ({self.n} experiments)")

        # Phase 4: Autonomous loop
        self.phase_autonomous_loop()

        elapsed = time.time() - t_start
        self._print_final(elapsed)
        self._finalize()
        self._save()

    def _print_final(self, elapsed: float):
        log("\n" + "=" * 70)
        log("AUTORESEARCH COMPLETE")
        log("=" * 70)
        log(f"  Total: {self.n} experiments in {elapsed/60:.1f} min ({elapsed:.0f}s)")
        log(f"  Keeps: {self.keeps} / Discards: {self.discards}")
        log(f"  Best score: {self.best_score:.1f}")
        log(f"  Exp/sec: {self.n/max(elapsed,1):.1f}")

        if self.best_result:
            r = self.best_result
            log(f"\n─── BEST CONFIG ───")
            log(f"  Score: {r['score']:.1f}")
            log(f"  Trades: {r['trades']}  WR: {r['win_rate']:.1f}%")
            log(f"  PnL: ${r['pnl']:.0f}  ROI: {r['roi_pct']:.0f}%")
            log(f"  DD: {r['dd']:.1f}%  Sharpe: {r['sharpe']:.1f}")
            log(f"  Util: {r['util']:.1f}%")
            log(f"\n  Params:")
            for k in sorted(self.best_params.keys()):
                v = self.best_params[k]
                if isinstance(v, set):
                    v = sorted(v)
                if k == "city_overrides" and isinstance(v, dict):
                    log(f"    {k}:")
                    for city in sorted(v.keys()):
                        log(f"      {city}: {v[city]}")
                else:
                    log(f"    {k}: {v}")

    def _finalize(self):
        """Re-run top configs with equity curves + OOS validation."""
        log("\n" + "=" * 70)
        log("FINALIZE: Enriching top configs (equity curves + OOS validation)")
        log("=" * 70)

        # OOS data: everything after training cutoff
        sim_oos = self.sim_all.filter_events(min_date=TRAIN_END)
        log(f"  OOS events: {len(sim_oos.events)}")

        # Get top 20 unique configs by score
        sorted_hist = sorted(self.history, key=lambda h: -h.get("score", 0))
        seen: set[float] = set()
        top = []
        for h in sorted_hist:
            key = round(h["score"], 1)
            if key not in seen and h.get("params"):
                seen.add(key)
                top.append(h)
            if len(top) >= 20:
                break

        self.top_enriched = []
        for i, h in enumerate(top):
            params = clone(h["params"])
            log(f"  [{i+1}/{len(top)}] {h['id']} (score={h['score']:.1f})")

            # Re-run on training data WITH equity curve
            result = simulate_portfolio(
                self.sim, params, entry_hours=h.get("entry_hours", 24),
                experiment_id=h["id"],
                record_equity=True,
            )
            experiment_score(result)

            # OOS test on holdout period
            oos_result = simulate_portfolio(
                sim_oos, params, entry_hours=h.get("entry_hours", 24),
                experiment_id=f"{h['id']}_oos",
            )
            experiment_score(oos_result)

            # Walk-forward on training data
            wf = walk_forward_validate(
                self.sim, params, entry_hours=h.get("entry_hours", 24), n_folds=3,
            )

            enriched = {k: v for k, v in h.items() if k != "city_results"}
            enriched["city_results"] = result.city_results
            enriched["equity_curve"] = result.equity_curve
            enriched["oos_pnl"] = round(oos_result.total_pnl, 2)
            enriched["oos_trades"] = oos_result.trades
            enriched["oos_win_rate"] = round(oos_result.win_rate, 1)
            enriched["oos_roi_pct"] = round(oos_result.roi_pct, 1)
            enriched["oos_sharpe"] = round(oos_result.sharpe, 2)
            enriched["wf_oos_pnl"] = wf.get("oos_pnl", 0)
            enriched["wf_score"] = wf.get("score", 0)
            enriched["wf_folds"] = wf.get("folds", [])

            # Re-score with OOS data
            result.oos_pnl = oos_result.total_pnl
            enriched["score_with_oos"] = experiment_score(result)

            self.top_enriched.append(enriched)
            log(f"    → score_oos={enriched['score_with_oos']:.1f}  "
                f"OOS: pnl=${oos_result.total_pnl:.0f} trades={oos_result.trades} "
                f"WF: pnl=${wf.get('oos_pnl', 0):.0f}")

    def _save(self):
        RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)

        def ser(p):
            if p is None:
                return None
            p2 = dict(p)
            if isinstance(p2.get("excluded_cities"), set):
                p2["excluded_cities"] = sorted(p2["excluded_cities"])
            return p2

        def ser_history(h):
            """Serialize a history entry (strip params & city_results for compactness)."""
            out = {}
            for k, v in h.items():
                if k in ("params", "city_results"):
                    continue
                if isinstance(v, set):
                    out[k] = sorted(v)
                else:
                    out[k] = v
            return out

        def ser_enriched(h):
            """Serialize an enriched top result (include all data)."""
            out = {}
            for k, v in h.items():
                if k == "params":
                    out[k] = ser(v)
                elif isinstance(v, set):
                    out[k] = sorted(v)
                else:
                    out[k] = v
            return out

        output = {
            "meta": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "total_experiments": self.n,
                "keeps": self.keeps, "discards": self.discards,
                "best_score": self.best_score,
            },
            "best_params": ser(self.best_params) if self.best_params else None,
            "best_result": {k: v for k, v in (self.best_result or {}).items()
                           if k not in ("params", "city_results")},
            "city_solo": {c: {k: v for k, v in r.items()
                              if k not in ("params", "city_results")}
                          for c, r in self.city_solo.items()},
            "top_enriched": [ser_enriched(h) for h in getattr(self, "top_enriched", [])],
            "history": [ser_history(h) for h in self.history],
        }

        with open(RESULTS_JSON, "w") as f:
            json.dump(output, f, indent=2, default=str)
        log(f"  Saved → {RESULTS_JSON}")


def main():
    max_exp = 0
    if "--max" in sys.argv:
        idx = sys.argv.index("--max")
        max_exp = int(sys.argv[idx + 1])

    ar = AutoResearch(max_experiments=max_exp)
    ar.run()


if __name__ == "__main__":
    main()
