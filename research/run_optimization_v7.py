"""
Optimization v7: 20-city with FIXED bias simulation in harness.

v6 had a critical bug: harness generated unbiased forecasts but strategy applied
bias correction → systematic error for high-bias cities (BA: 8.7% WR).
Now harness generates biased forecasts matching real-world behavior.

Baseline: composite=187.78, Sharpe=24.3, WR=89.3%, 6023 trades

Focus: per-city parameter profiles to squeeze maximum edge.
"""
import subprocess
import re
import sys
import os
import json
import time

EXPERIMENT_FILE = os.path.join(os.path.dirname(__file__), "strategy_experiment.py")
HARNESS_CMD = [sys.executable, os.path.join(os.path.dirname(__file__), "backtest_harness.py")]
WORK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

with open(EXPERIMENT_FILE) as f:
    BASELINE = f.read()

results = []
t_start = time.time()


def write_file(content):
    with open(EXPERIMENT_FILE, "w") as f:
        f.write(content)


def run_backtest():
    result = subprocess.run(HARNESS_CMD, capture_output=True, text=True, timeout=300, cwd=WORK_DIR)
    output = result.stdout + result.stderr
    metrics = {}
    for line in output.split("\n"):
        for key in [
            "composite_score", "avg_sharpe", "num_trades", "avg_win_rate",
            "max_drawdown_pct", "avg_pnl_pct", "avg_profit_factor", "avg_return_pct",
            "profitable_windows",
        ]:
            if line.strip().startswith(f"{key}:"):
                val = line.split(":")[1].strip().split("/")[0]
                try:
                    metrics[key] = float(val)
                except ValueError:
                    metrics[key] = val
    return metrics


def set_param(content, param, value):
    val_str = str(value)
    pattern = rf'^({param}\s*=\s*)([^\n#]+)'
    return re.sub(pattern, rf'\g<1>{val_str}', content, flags=re.MULTILINE)


def set_shrinkage(content, model_w):
    prior_w = round(1.0 - model_w, 3)
    return re.sub(
        r'raw = raw \* [\d.]+ \+ uniform_prior \* [\d.]+',
        f'raw = raw * {model_w} + uniform_prior * {prior_w}',
        content,
    )


def set_city_overrides(content, overrides_dict):
    override_str = json.dumps(overrides_dict)
    return re.sub(
        r'^(CITY_OVERRIDES\s*=\s*)([^\n]+)',
        rf'\g<1>{override_str}',
        content,
        flags=re.MULTILINE,
    )


def set_max_edge_cap(content, val):
    return re.sub(r'if edge > [\d.]+:', f'if edge > {val}:', content)


def apply_params(params):
    c = BASELINE
    for k, v in params.items():
        if k == "_shrinkage":
            c = set_shrinkage(c, v)
        elif k == "_max_edge_cap":
            c = set_max_edge_cap(c, v)
        elif k == "_city_overrides":
            c = set_city_overrides(c, v)
        else:
            c = set_param(c, k, v)
    return c


def run_exp(name, params):
    content = apply_params(params)
    write_file(content)
    try:
        metrics = run_backtest()
        r = {
            "name": name, "params": params,
            "composite": metrics.get("composite_score", 0.0),
            "trades": int(metrics.get("num_trades", 0)),
            "sharpe": metrics.get("avg_sharpe", 0.0),
            "win_rate": metrics.get("avg_win_rate", 0.0),
            "pnl_pct": metrics.get("avg_pnl_pct", 0.0),
            "dd_pct": metrics.get("max_drawdown_pct", 0.0),
            "pf": metrics.get("avg_profit_factor", 0.0),
        }
        results.append(r)
        elapsed = time.time() - t_start
        print(
            f"  [{len(results):>3d}] {name:<65s} | comp={r['composite']:>10.4f} | "
            f"trades={r['trades']:>4d} | sharpe={r['sharpe']:>7.2f} | "
            f"win={r['win_rate']:>5.1f}% | pnl={r['pnl_pct']:>6.1f}% | "
            f"dd={r['dd_pct']:>5.2f}% | {elapsed:.0f}s"
        )
        return r["composite"]
    except Exception as e:
        print(f"  ERROR: {name}: {e}")
        return 0.0


print("=" * 160)
print("STRATEGY C OPTIMIZATION v7 — FIXED BIAS SIMULATION (20 CITIES)")
print("=" * 160)

# ============================================================================
# Phase 1: Baseline
# ============================================================================
print("\n--- Phase 1: Baseline ---")
run_exp("baseline", {})

# ============================================================================
# Phase 2: Single-param sweeps
# ============================================================================
print("\n--- Phase 2: Single-param sweeps ---")

for mp in [0.35, 0.40, 0.43, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80]:
    run_exp(f"maxp={mp}", {"MAX_PRICE": mp})

for mfp in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.68, 0.70, 0.75]:
    run_exp(f"mfp={mfp}", {"MIN_FORECAST_PROB": mfp})

for s in [0.80, 0.85, 0.90, 0.92, 0.95, 1.0, 1.05, 1.10, 1.15, 1.20]:
    run_exp(f"sharp={s}", {"PROB_SHARPENING": s})

for me in [0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.12, 0.15]:
    run_exp(f"min_edge={me}", {"MIN_EDGE": me})

for minp in [0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30]:
    run_exp(f"minp={minp}", {"MIN_PRICE": minp})

for mec in [0.42, 0.50, 0.60, 0.70, 0.80]:
    run_exp(f"max_edge_cap={mec}", {"_max_edge_cap": mec})

# ============================================================================
# Phase 3: Combined sweeps
# ============================================================================
print("\n--- Phase 3: Combined sweeps ---")
best1 = max(results, key=lambda x: x["composite"])
print(f"  Best Phase 2: {best1['name']} = {best1['composite']:.4f}")

for mfp in [0.45, 0.50, 0.55, 0.60, 0.65, 0.68]:
    for maxp in [0.43, 0.50, 0.55, 0.60, 0.70]:
        for sharp in [0.85, 0.90, 0.92, 0.95, 1.0, 1.05]:
            run_exp(
                f"mfp={mfp} maxp={maxp} s={sharp}",
                {"MIN_FORECAST_PROB": mfp, "MAX_PRICE": maxp, "PROB_SHARPENING": sharp},
            )

# ============================================================================
# Phase 4: Fine-tune around Phase 3 winner
# ============================================================================
print("\n--- Phase 4: Fine-tuning ---")
best3 = max(results, key=lambda x: x["composite"])
print(f"  Best Phase 3: {best3['name']} = {best3['composite']:.4f}")
bp = best3.get("params", {})

b_mfp = bp.get("MIN_FORECAST_PROB", 0.65)
b_maxp = bp.get("MAX_PRICE", 0.43)
b_sharp = bp.get("PROB_SHARPENING", 0.92)

for mfp in [b_mfp - 0.03, b_mfp - 0.01, b_mfp, b_mfp + 0.01, b_mfp + 0.03]:
    for maxp in [b_maxp - 0.03, b_maxp, b_maxp + 0.03]:
        for sharp in [b_sharp - 0.03, b_sharp, b_sharp + 0.03]:
            for me in [0.06, 0.08, 0.10]:
                run_exp(
                    f"fine mfp={mfp:.2f} mp={maxp:.2f} s={sharp:.2f} me={me}",
                    {
                        "MIN_FORECAST_PROB": round(mfp, 2),
                        "MAX_PRICE": round(maxp, 2),
                        "PROB_SHARPENING": round(sharp, 2),
                        "MIN_EDGE": me,
                    },
                )

# ============================================================================
# Phase 5: Per-city quality gate overrides
# ============================================================================
print("\n--- Phase 5: Per-city overrides ---")
best4 = max(results, key=lambda x: x["composite"])
print(f"  Best Phase 4: {best4['name']} = {best4['composite']:.4f}")
bp4 = best4.get("params", {})

# High-variance cities (sigma > 1.5): nyc, dc, tokyo, los_angeles, munich
HIGH_VAR = ["nyc", "dc", "tokyo", "los_angeles", "munich"]
# Low-variance cities (sigma < 1.0): paris, london, seattle, miami, lucknow
LOW_VAR = ["paris", "london", "seattle", "miami", "lucknow"]

for tight in [0.10, 0.12, 0.15]:
    overrides = {c: {"min_edge": tight} for c in HIGH_VAR}
    run_exp(f"hv_edge={tight}", {**bp4, "_city_overrides": overrides})

for loose in [0.04, 0.05, 0.06]:
    overrides = {c: {"min_edge": loose} for c in LOW_VAR}
    run_exp(f"lv_edge={loose}", {**bp4, "_city_overrides": overrides})

for tight in [0.10, 0.12, 0.15]:
    for loose in [0.04, 0.05, 0.06]:
        overrides = {}
        for c in HIGH_VAR:
            overrides[c] = {"min_edge": tight}
        for c in LOW_VAR:
            overrides[c] = {"min_edge": loose}
        run_exp(f"split t={tight} l={loose}", {**bp4, "_city_overrides": overrides})

# Also try per-city max_price overrides
# Low-var cities with high WR can handle higher max_price
for lv_maxp in [0.55, 0.60, 0.70]:
    overrides = {c: {"max_price": lv_maxp} for c in LOW_VAR}
    run_exp(f"lv_maxp={lv_maxp}", {**bp4, "_city_overrides": overrides})

# High-var cities need tighter max_price
for hv_maxp in [0.35, 0.40]:
    overrides = {c: {"max_price": hv_maxp} for c in HIGH_VAR}
    run_exp(f"hv_maxp={hv_maxp}", {**bp4, "_city_overrides": overrides})

# ============================================================================
# Phase 6: Kelly, shrinkage, max_edge
# ============================================================================
print("\n--- Phase 6: Sizing ---")
best5 = max(results, key=lambda x: x["composite"])
bp5 = best5.get("params", {})
print(f"  Best Phase 5: {best5['name']} = {best5['composite']:.4f}")

for kf in [0.005, 0.01, 0.015, 0.02, 0.03]:
    run_exp(f"kelly={kf}", {**bp5, "KELLY_FRACTION": kf})

for sw in [0.93, 0.95, 0.97, 0.99]:
    run_exp(f"shrink={sw}", {**bp5, "_shrinkage": sw})

# ============================================================================
# RESULTS
# ============================================================================
elapsed = time.time() - t_start
print("\n" + "=" * 160)
print(f"TOP 20 EXPERIMENTS (v7 — fixed bias, {len(results)} total, {elapsed:.0f}s = {elapsed/60:.1f}min)")
print("=" * 160)
sr = sorted(results, key=lambda x: x["composite"], reverse=True)
for i, r in enumerate(sr[:20]):
    print(
        f"  #{i+1:>2d}  {r['name']:<70s} | comp={r['composite']:>10.4f} | "
        f"trades={r['trades']:>4d} | sharpe={r['sharpe']:>7.2f} | "
        f"win={r['win_rate']:>5.1f}% | pnl={r['pnl_pct']:>6.1f}% | dd={r['dd_pct']:>5.2f}%"
    )

print(f"\nTotal experiments: {len(results)}")
best = sr[0]
print(f"\nBEST: {best['name']}")
print(f"  composite_score = {best['composite']:.6f}")
print(f"  params = {json.dumps(best.get('params', {}), indent=2)}")

results_file = os.path.join(os.path.dirname(__file__), "optimization_results_v7.json")
with open(results_file, "w") as f:
    json.dump(sr, f, indent=2)
print(f"\nResults saved to {results_file}")

write_file(BASELINE)
print("Restored strategy_experiment.py to baseline.")
print(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f}min)")
