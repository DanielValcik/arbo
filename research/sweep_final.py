"""Final Autoresearch Sweep — Chronological Portfolio Simulator.

6-phase parameter optimization on real Polymarket data:
  Phase 1: Broad random search (2000 trials)
  Phase 2: Fine-tuning around top 20 (1000 trials)
  Phase 3: City exclusion & overrides (500 trials)
  Phase 4: Exit optimization (500 trials)
  Phase 5: Multi-entry-timing robustness (500 trials)
  Phase 6: Walk-forward validation (200 trials)

Uses the chronological portfolio simulator (experiment_framework.py)
with entry + exit params optimized JOINTLY.

Data: Goldsky on-chain trades + CLOB prices (571K+ price points, 420 days).
Train: Jan 2025 → Jan 2026 | Test: Feb-Mar 2026 (used ONCE on final config).

Usage:
    python3 research/sweep_final.py
    python3 research/sweep_final.py --quick    # 200 trials only (smoke test)
"""

import json
import random
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from experiment_framework import (
    CITY_COORDS,
    ExperimentResult,
    SimulationData,
    experiment_score,
    load_forecasts,
    preload_data,
    serialize_result,
    simulate_portfolio,
    walk_forward_validate,
)
from price_history_db import PriceHistoryDB

# ── Constants ──────────────────────────────────────────────────────────────────

TRAIN_END = "2026-01-31"
TEST_START = "2026-02-01"
RESULTS_DIR = Path(__file__).parent / "data" / "experiments"
LOG_PATH = Path(__file__).parent / "sweep_final_log.txt"
ALL_CITIES = set(CITY_COORDS.keys())

# ── Logging ────────────────────────────────────────────────────────────────────

_log_file = None


def log(msg: str):
    print(msg, flush=True)
    global _log_file
    if _log_file is None:
        _log_file = open(LOG_PATH, "a")
    _log_file.write(msg + "\n")
    _log_file.flush()


# ── Search Space ───────────────────────────────────────────────────────────────


def random_params(rng: random.Random) -> dict:
    """Generate random parameter set — quality gate + exit params together."""
    exit_enabled = rng.choice([True, False])

    p = {
        # Quality gate
        "min_edge": rng.choice(
            [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04,
             0.05, 0.06, 0.08, 0.10, 0.12, 0.15]
        ),
        "max_edge": rng.choice([0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]),
        "max_price": rng.choice(
            [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55,
             0.60, 0.70, 0.80, 0.90]
        ),
        "min_price": rng.choice(
            [0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]
        ),
        "min_prob": rng.choice(
            [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15,
             0.20, 0.25, 0.30, 0.40, 0.50]
        ),
        "min_volume": rng.choice([0, 10, 25, 50, 100, 200, 500, 1000, 2000]),
        "kelly_raw_cap": rng.choice(
            [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60]
        ),
        "prob_sharpening": rng.choice(
            [0.70, 0.80, 0.85, 0.90, 0.95, 1.0, 1.05,
             1.10, 1.15, 1.20, 1.30, 1.50]
        ),
        "shrinkage": rng.choice(
            [0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25]
        ),
        "excluded_cities": set(),
        "city_overrides": {},
        # Exit params
        "exit_enabled": exit_enabled,
        "min_hold_edge": rng.choice(
            [0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20]
        ) if exit_enabled else 0.0,
        "prob_exit_floor": rng.choice(
            [0.0, 0.20, 0.30, 0.40, 0.50]
        ) if exit_enabled else 0.0,
        "profit_take_enabled": rng.choice([True, False]) if exit_enabled else False,
        "profit_take_threshold": rng.choice(
            [0.30, 0.50, 0.75, 1.00, 1.50, 2.00, 3.00]
        ),
        "profit_take_min_hours": rng.choice([2, 4, 6, 12]),
        "reentry_enabled": rng.choice([True, False]) if exit_enabled else False,
        "reentry_cooldown_h": rng.choice([1, 2, 4, 6]),
        "stop_loss_pct": 1.0,  # Disabled — proven harmful
    }

    # Ensure min_hold_edge <= min_edge (avoid immediate exit after entry)
    if exit_enabled and p["min_hold_edge"] > p["min_edge"]:
        p["min_hold_edge"] = p["min_edge"] * rng.choice([0.3, 0.5, 0.7, 0.9])

    return p


def perturb_params(base: dict, rng: random.Random, n_changes: int = 0) -> dict:
    """Create a perturbation of a parameter set."""
    p = dict(base)
    p["excluded_cities"] = set(p.get("excluded_cities", set()))
    if isinstance(p["excluded_cities"], list):
        p["excluded_cities"] = set(p["excluded_cities"])
    p["city_overrides"] = dict(p.get("city_overrides", {}))

    # Quality gate keys
    qg_keys = [
        "min_edge", "max_edge", "max_price", "min_price", "min_prob",
        "min_volume", "kelly_raw_cap", "prob_sharpening", "shrinkage",
    ]
    # Exit keys
    exit_keys = [
        "min_hold_edge", "prob_exit_floor", "profit_take_threshold",
        "profit_take_min_hours", "reentry_cooldown_h",
    ]

    all_keys = qg_keys + (exit_keys if p.get("exit_enabled") else [])
    n_perturb = n_changes if n_changes > 0 else rng.randint(1, 3)

    for key in rng.sample(all_keys, min(n_perturb, len(all_keys))):
        val = p.get(key, 0)
        if key == "min_volume":
            p[key] = max(0, val + rng.choice(
                [-500, -200, -100, -50, 50, 100, 200, 500]
            ))
        elif key == "prob_sharpening":
            p[key] = round(max(0.5, val + rng.choice(
                [-0.15, -0.10, -0.05, 0.05, 0.10, 0.15]
            )), 2)
        elif key == "shrinkage":
            p[key] = round(max(0, min(0.40, val + rng.choice(
                [-0.05, -0.03, -0.02, 0.02, 0.03, 0.05]
            ))), 2)
        elif key == "kelly_raw_cap":
            p[key] = round(max(0.05, min(0.70, val + rng.choice(
                [-0.10, -0.05, 0.05, 0.10]
            ))), 2)
        elif key == "max_edge":
            p[key] = round(max(0.15, min(0.95, val + rng.choice(
                [-0.10, -0.05, 0.05, 0.10]
            ))), 2)
        elif key == "min_hold_edge":
            p[key] = round(max(0, min(0.25, val + rng.choice(
                [-0.05, -0.03, -0.02, 0.02, 0.03, 0.05]
            ))), 3)
        elif key == "prob_exit_floor":
            p[key] = round(max(0, min(0.60, val + rng.choice(
                [-0.10, -0.05, 0.05, 0.10]
            ))), 2)
        elif key == "profit_take_threshold":
            p[key] = round(max(0.10, min(5.0, val + rng.choice(
                [-0.50, -0.25, 0.25, 0.50]
            ))), 2)
        elif key == "profit_take_min_hours":
            p[key] = max(1, val + rng.choice([-4, -2, 2, 4]))
        elif key == "reentry_cooldown_h":
            p[key] = max(1, val + rng.choice([-2, -1, 1, 2]))
        elif key == "min_price":
            p[key] = round(max(0.05, min(0.30, val + rng.choice(
                [-0.04, -0.03, -0.02, -0.01, 0.01, 0.02, 0.03, 0.04]
            ))), 3)
        else:
            p[key] = round(max(0.005, val + rng.choice(
                [-0.04, -0.03, -0.02, -0.01, 0.01, 0.02, 0.03, 0.04]
            )), 3)

    # Safety: min_hold_edge <= min_edge
    if p.get("exit_enabled") and p.get("min_hold_edge", 0) > p.get("min_edge", 0.08):
        p["min_hold_edge"] = round(p["min_edge"] * 0.7, 3)

    return p


def random_exit_params(rng: random.Random) -> dict:
    """Generate random exit params for Phase 4 (exit-only sweep)."""
    return {
        "exit_enabled": True,
        "min_hold_edge": rng.choice(
            [0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05,
             0.06, 0.08, 0.10]
        ),
        "prob_exit_floor": rng.choice([0.0, 0.15, 0.20, 0.25, 0.30, 0.40]),
        "profit_take_enabled": rng.choice([True, False]),
        "profit_take_threshold": rng.choice(
            [0.30, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00, 3.00]
        ),
        "profit_take_min_hours": rng.choice([2, 3, 4, 6, 8, 12]),
        "reentry_enabled": rng.choice([True, False]),
        "reentry_cooldown_h": rng.choice([1, 2, 3, 4, 6]),
        "stop_loss_pct": 1.0,
    }


# ── Helpers ────────────────────────────────────────────────────────────────────


def result_summary(r: ExperimentResult) -> str:
    """One-line summary string for logging."""
    exit_str = ""
    if r.total_exits > 0:
        exit_str = f"  ex={r.total_exits}(s{r.exit_saves}/r{r.exit_regrets})"
    return (
        f"score={r.score:>8.2f}  trades={r.trades:>3}  "
        f"WR={r.win_rate:>5.1f}%  PnL=${r.total_pnl:>8.2f}  "
        f"DD={r.max_drawdown_pct:>5.2f}%  Sharpe={r.sharpe:>6.2f}  "
        f"util={r.capital_utilization:>4.1f}%  pph=${r.avg_pnl_per_hour:>5.2f}"
        f"{exit_str}"
    )


# ── Main Sweep ─────────────────────────────────────────────────────────────────


def main():
    t_start = time.time()
    rng = random.Random(2026)

    # Parse args
    quick_mode = "--quick" in sys.argv
    if quick_mode:
        P1, P2, P3, P4, P5, P6 = 100, 50, 50, 50, 20, 10
        log("*** QUICK MODE — reduced trial counts ***")
    else:
        P1, P2, P3, P4, P5, P6 = 2000, 1000, 500, 500, 500, 200

    # Load data
    log(f"\n{'=' * 70}")
    log(f"FINAL SWEEP — Chronological Portfolio Simulator")
    log(f"Start: {datetime.now().isoformat()}")
    log(f"{'=' * 70}\n")

    db = PriceHistoryDB()
    events = db.get_events(with_prices=True)
    log(f"Total events: {len(events)}")

    # Train/test split
    train_events = [
        e for e in events
        if e.target_date and e.target_date <= TRAIN_END
    ]
    test_events = [
        e for e in events
        if e.target_date and e.target_date >= TEST_START
    ]
    log(f"Train events (≤{TRAIN_END}): {len(train_events)}")
    log(f"Test events  (≥{TEST_START}): {len(test_events)}")

    # Load forecasts and preload ALL data (shared across train/test)
    forecasts = load_forecasts(events)
    sim_all = preload_data(db, events, forecasts)

    # Create train-only SimulationData
    sim_train = sim_all.filter_events(max_date=TRAIN_END)
    log(f"Train simulation events: {len(sim_train.events)}")

    # Track all results
    all_results: list[ExperimentResult] = []
    best_score = -999.0
    best_result: ExperimentResult | None = None
    trial_counter = 0

    def run_trial(params, phase, entry_hours=24, record_equity=False):
        nonlocal trial_counter, best_score, best_result
        trial_counter += 1
        exp_id = f"EXP-{trial_counter:04d}"

        result = simulate_portfolio(
            sim_train, params, entry_hours=entry_hours,
            experiment_id=exp_id, record_equity=record_equity,
        )
        all_results.append(result)

        if result.score > best_score:
            best_score = result.score
            best_result = result
            log(f"  {exp_id}: {result_summary(result)}  ★ NEW BEST")
            return True
        return False

    # ══════════════════════════════════════════════════════════════════════
    # Phase 1: Broad Random Search
    # ══════════════════════════════════════════════════════════════════════
    log(f"\nPhase 1: Broad Random Search ({P1} trials)")
    log("-" * 50)

    for i in range(P1):
        params = random_params(rng)
        run_trial(params, phase=1)

        if (i + 1) % max(P1 // 5, 1) == 0:
            elapsed = time.time() - t_start
            log(f"  ... {i + 1}/{P1} done, best: {best_score:.2f} "
                f"({elapsed/60:.1f}min)")

    # ══════════════════════════════════════════════════════════════════════
    # Phase 2: Fine-tuning around top 20
    # ══════════════════════════════════════════════════════════════════════
    log(f"\nPhase 2: Fine-tuning ({P2} trials around top 20)")
    log("-" * 50)

    top20 = sorted(all_results, key=lambda r: -r.score)[:20]
    log(f"  Top 5 scores: {[r.score for r in top20[:5]]}")

    for i in range(P2):
        base = top20[i % 20]
        n_changes = 1 if i % 3 == 0 else (2 if i % 3 == 1 else 3)
        params = perturb_params(base.params, rng, n_changes)
        run_trial(params, phase=2)

        if (i + 1) % max(P2 // 5, 1) == 0:
            elapsed = time.time() - t_start
            log(f"  ... {i + 1}/{P2} done, best: {best_score:.2f} "
                f"({elapsed/60:.1f}min)")

    # ══════════════════════════════════════════════════════════════════════
    # Phase 3: City Optimization
    # ══════════════════════════════════════════════════════════════════════
    log(f"\nPhase 3: City optimization ({P3} trials)")
    log("-" * 50)

    if best_result:
        city_pnl = best_result.city_results
        losing_cities = sorted(
            [c for c, cr in city_pnl.items() if cr["pnl"] < 0],
            key=lambda c: city_pnl[c]["pnl"],
        )
        winning_cities = sorted(
            [c for c, cr in city_pnl.items() if cr["pnl"] > 0],
            key=lambda c: -city_pnl[c]["pnl"],
        )
        log(f"  Losing: {losing_cities}")
        log(f"  Winning: {winning_cities[:5]}")
    else:
        losing_cities = []
        winning_cities = []

    for i in range(P3):
        params = dict(best_result.params) if best_result else random_params(rng)
        params["excluded_cities"] = set(params.get("excluded_cities", []))
        params["city_overrides"] = dict(params.get("city_overrides", {}))

        if i < P3 * 0.3 and losing_cities:
            n_exc = rng.randint(1, min(len(losing_cities), 6))
            params["excluded_cities"] = set(rng.sample(losing_cities, n_exc))
        elif i < P3 * 0.6 and winning_cities:
            params["excluded_cities"] = set(losing_cities)
            for city in rng.sample(
                winning_cities, min(len(winning_cities), 4)
            ):
                params["city_overrides"][city] = {
                    "max_price": rng.choice(
                        [0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80]
                    ),
                    "min_edge": rng.choice(
                        [0.005, 0.01, 0.015, 0.02, 0.03, 0.05]
                    ),
                    "min_price": rng.choice(
                        [0.05, 0.08, 0.10, 0.15]
                    ),
                }
        else:
            params = perturb_params(params, rng)
            if losing_cities:
                params["excluded_cities"] = set(losing_cities)

        run_trial(params, phase=3)

        if (i + 1) % max(P3 // 5, 1) == 0:
            elapsed = time.time() - t_start
            log(f"  ... {i + 1}/{P3} done, best: {best_score:.2f} "
                f"({elapsed/60:.1f}min)")

    # ══════════════════════════════════════════════════════════════════════
    # Phase 4: Exit Optimization
    # ══════════════════════════════════════════════════════════════════════
    log(f"\nPhase 4: Exit optimization ({P4} trials)")
    log("-" * 50)

    # Use best QG params, sweep ONLY exit params
    base_qg = dict(best_result.params) if best_result else random_params(rng)

    for i in range(P4):
        params = dict(base_qg)
        exit_p = random_exit_params(rng)
        params.update(exit_p)

        # Ensure min_hold_edge <= min_edge
        if params["min_hold_edge"] > params.get("min_edge", 0.08):
            params["min_hold_edge"] = round(
                params["min_edge"] * rng.choice([0.3, 0.5, 0.7, 0.9]), 3
            )

        run_trial(params, phase=4)

        if (i + 1) % max(P4 // 5, 1) == 0:
            elapsed = time.time() - t_start
            log(f"  ... {i + 1}/{P4} done, best: {best_score:.2f} "
                f"({elapsed/60:.1f}min)")

    # ══════════════════════════════════════════════════════════════════════
    # Phase 5: Multi-Entry-Timing Robustness
    # ══════════════════════════════════════════════════════════════════════
    log(f"\nPhase 5: Multi-entry-timing ({P5} configs × 5 timings)")
    log("-" * 50)

    top50 = sorted(all_results, key=lambda r: -r.score)[:min(50, P5)]
    entry_timings = [48, 36, 24, 12, 6]
    timing_weights = {48: 0.15, 36: 0.20, 24: 0.30, 12: 0.20, 6: 0.15}

    multi_results: list[dict] = []

    for i, base_r in enumerate(top50):
        timing_scores = {}
        timing_details = {}

        for hours in entry_timings:
            r = simulate_portfolio(
                sim_train, base_r.params, entry_hours=hours,
                experiment_id=f"MT-{i+1}-{hours}h",
            )
            timing_scores[hours] = r.score
            timing_details[f"{hours}h"] = {
                "score": r.score, "trades": r.trades,
                "pnl": r.total_pnl, "utilization": r.capital_utilization,
            }

        weighted = sum(
            timing_scores[h] * timing_weights[h] for h in entry_timings
        )
        min_score = min(timing_scores.values())
        robust_score = weighted * 0.6 + min_score * 0.4

        multi_results.append({
            "experiment_id": base_r.experiment_id,
            "params": base_r.params,
            "robust_score": round(robust_score, 2),
            "weighted_score": round(weighted, 2),
            "min_score": round(min_score, 2),
            "per_timing": timing_details,
            "base_score": base_r.score,
        })

        if (i + 1) % max(len(top50) // 5, 1) == 0:
            best_mt = max(multi_results, key=lambda r: r["robust_score"])
            log(f"  ... {i + 1}/{len(top50)} done, "
                f"best robust: {best_mt['robust_score']:.2f}")

    multi_results.sort(key=lambda r: -r["robust_score"])
    log(f"  Top 5 robust scores: "
        f"{[r['robust_score'] for r in multi_results[:5]]}")

    # Perturb top 5 multi-timing results
    remaining = P5 - len(top50)
    multi_top5 = multi_results[:5]
    for i in range(max(0, remaining)):
        base_mr = multi_top5[i % 5]
        params = perturb_params(base_mr["params"], rng, n_changes=1)

        timing_scores = {}
        timing_details = {}
        for hours in entry_timings:
            r = simulate_portfolio(
                sim_train, params, entry_hours=hours,
                experiment_id=f"MT-P-{i}",
            )
            timing_scores[hours] = r.score
            timing_details[f"{hours}h"] = {
                "score": r.score, "trades": r.trades,
                "pnl": r.total_pnl,
            }

        weighted = sum(
            timing_scores[h] * timing_weights[h] for h in entry_timings
        )
        min_score = min(timing_scores.values())
        robust_score = weighted * 0.6 + min_score * 0.4

        multi_results.append({
            "experiment_id": f"MT-P-{i}",
            "params": params,
            "robust_score": round(robust_score, 2),
            "weighted_score": round(weighted, 2),
            "min_score": round(min_score, 2),
            "per_timing": timing_details,
        })

    multi_results.sort(key=lambda r: -r["robust_score"])

    # ══════════════════════════════════════════════════════════════════════
    # Phase 6: Walk-Forward Validation
    # ══════════════════════════════════════════════════════════════════════
    log(f"\nPhase 6: Walk-forward validation (top {min(P6, 20)})")
    log("-" * 50)

    top_for_wf = multi_results[:min(P6, 20)]
    wf_results = []

    for i, mr in enumerate(top_for_wf):
        params = mr["params"]
        wf = walk_forward_validate(sim_train, params, entry_hours=24, n_folds=3)
        wf["experiment_id"] = mr["experiment_id"]
        wf["params"] = params
        wf["robust_score"] = mr["robust_score"]
        wf_results.append(wf)

        log(f"  WF #{i+1} ({mr['experiment_id']}): "
            f"oos_pnl=${wf['oos_pnl']:.2f}  "
            f"oos_score={wf['score']:.2f}  "
            f"trades={wf.get('total_trades', 0)}  "
            f"robust={mr['robust_score']:.2f}")

    # Perturb top 5 WF results
    wf_sorted = sorted(wf_results, key=lambda r: -r["score"])[:5]
    for i in range(min(P6 - len(top_for_wf), 180)):
        base = wf_sorted[i % len(wf_sorted)]
        params = perturb_params(base["params"], rng, n_changes=1)
        wf = walk_forward_validate(sim_train, params, entry_hours=24, n_folds=3)
        wf["params"] = params
        wf["experiment_id"] = f"WF-P-{i}"
        wf_results.append(wf)

    wf_results.sort(key=lambda r: -r["score"])

    # ══════════════════════════════════════════════════════════════════════
    # Save Results
    # ══════════════════════════════════════════════════════════════════════
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    sweep_id = datetime.now().strftime("sweep_%Y%m%d_%H%M")
    output_path = RESULTS_DIR / f"{sweep_id}.json"

    # Top 20 with equity curves — DEDUPLICATED
    # Skip experiments with nearly identical metrics (same trades ±5, same PnL ±2%)
    def _is_duplicate(r, selected):
        for s in selected:
            if (abs(r.trades - s.trades) <= 5
                    and abs(r.total_pnl - s.total_pnl) / max(abs(s.total_pnl), 1) < 0.02):
                return True
        return False

    top20_unique = []
    for r in sorted(all_results, key=lambda r: -r.score):
        if not _is_duplicate(r, top20_unique):
            top20_unique.append(r)
        if len(top20_unique) >= 20:
            break

    log(f"\n  Top 20 unique experiments selected (from {len(all_results)} total)")

    # Re-run top 20 with equity curves
    top20_with_equity = []
    for r in top20_unique:
        r_eq = simulate_portfolio(
            sim_train, r.params, entry_hours=24,
            experiment_id=r.experiment_id, record_equity=True,
        )
        top20_with_equity.append(serialize_result(r_eq))

    # Compact all results (no equity curves)
    all_compact = []
    for r in sorted(all_results, key=lambda r: -r.score):
        all_compact.append({
            "experiment_id": r.experiment_id,
            "score": r.score,
            "trades": r.trades,
            "wins": r.wins,
            "win_rate": r.win_rate,
            "total_pnl": r.total_pnl,
            "roi_pct": r.roi_pct,
            "max_drawdown_pct": r.max_drawdown_pct,
            "sharpe": r.sharpe,
            "capital_utilization": r.capital_utilization,
            "avg_pnl_per_hour": r.avg_pnl_per_hour,
            "turnover_rate": r.turnover_rate,
            "total_exits": r.total_exits,
            "exit_saves": r.exit_saves,
            "exit_regrets": r.exit_regrets,
            "params": serialize_result(r)["params"],
        })

    # Serialize multi-timing results
    def _ser_params(p):
        if isinstance(p, dict):
            p2 = dict(p)
            if isinstance(p2.get("excluded_cities"), set):
                p2["excluded_cities"] = sorted(p2["excluded_cities"])
            return p2
        return p

    multi_ser = []
    for mr in multi_results[:20]:
        mr2 = dict(mr)
        mr2["params"] = _ser_params(mr2["params"])
        multi_ser.append(mr2)

    wf_ser = []
    for wr in wf_results[:20]:
        wr2 = dict(wr)
        wr2["params"] = _ser_params(wr2["params"])
        wf_ser.append(wr2)

    output = {
        "meta": {
            "sweep_id": sweep_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "train_end": TRAIN_END,
            "test_start": TEST_START,
            "train_events": len(train_events),
            "test_events": len(test_events),
            "total_trials": trial_counter,
            "phases": f"P1={P1} P2={P2} P3={P3} P4={P4} P5={P5} P6={P6}",
            "quick_mode": quick_mode,
        },
        "top_results": top20_with_equity,
        "multi_timing": multi_ser,
        "walk_forward": wf_ser,
        "all_results": all_compact,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    elapsed = time.time() - t_start

    # ── Final Report ──
    log(f"\n{'=' * 70}")
    log(f"SWEEP COMPLETE — {elapsed / 60:.1f} minutes")
    log(f"{'=' * 70}")
    log(f"  Sweep ID: {sweep_id}")
    log(f"  Total trials: {trial_counter}")
    log(f"  Results: {output_path}")

    log(f"\n─── Best Overall (24h entry, training set) ───")
    if best_result:
        log(f"  {best_result.experiment_id}: {result_summary(best_result)}")
        bp = serialize_result(best_result)["params"]
        log(f"  Params:")
        for k in sorted(bp.keys()):
            log(f"    {k}: {bp[k]}")

    log(f"\n─── Top 5 Multi-Timing (robust) ───")
    for i, mr in enumerate(multi_results[:5]):
        log(f"  #{i+1} ({mr['experiment_id']}): "
            f"robust={mr['robust_score']:.2f}  "
            f"weighted={mr.get('weighted_score', 0):.2f}  "
            f"min={mr.get('min_score', 0):.2f}")
        if "per_timing" in mr:
            for t, d in sorted(mr["per_timing"].items()):
                log(f"    {t}: score={d['score']:.2f} trades={d['trades']}")

    log(f"\n─── Top 5 Walk-Forward ───")
    for i, wr in enumerate(wf_results[:5]):
        log(f"  #{i+1} ({wr.get('experiment_id', '?')}): "
            f"oos_pnl=${wr['oos_pnl']:.2f}  "
            f"oos_score={wr['score']:.2f}  "
            f"trades={wr.get('total_trades', 0)}")

    log(f"\n  Total time: {elapsed / 60:.1f} minutes")

    db.close()


if __name__ == "__main__":
    main()
