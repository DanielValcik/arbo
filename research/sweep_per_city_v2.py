"""Per-city sweep v2 — fine-tuning marginal cities + top performer expansion."""

import copy
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import strategy_experiment as strategy
import backtest_harness as harness


def run_with_overrides(overrides, data, normals):
    original = strategy.CITY_OVERRIDES
    strategy.CITY_OVERRIDES = overrides
    results = harness.walk_forward_evaluate(data, normals)
    all_trades = []
    for i, window in enumerate(harness.WALK_FORWARD_WINDOWS):
        seed = harness.BASE_SEED + i * 7919
        trades_w, _ = harness.run_single_backtest(
            data, normals, window["test"][0], window["test"][1], seed
        )
        all_trades.extend(trades_w)
    strategy.CITY_OVERRIDES = original
    return results, len(all_trades)


def main():
    harness.download_historical_data()
    data = harness.load_data()
    normals = harness.compute_monthly_normals(data)

    # Always exclude these 3 (consistently unprofitable)
    EXCLUDE = {"dc": {"min_edge": 0.99}, "toronto": {"min_edge": 0.99}, "buenos_aires": {"min_edge": 0.99}}

    base_results, _ = run_with_overrides(EXCLUDE, data, normals)
    base_score = base_results["composite_score"]
    print(f"Baseline (excl dc/tor/ba): {base_score:.4f}")

    best_score = base_score
    best_overrides = copy.deepcopy(EXCLUDE)
    best_desc = "excl_3"

    experiments = []

    # === Try raising MIN_EDGE for marginal cities instead of excluding ===
    for nyc_edge in [0.12, 0.15, 0.18, 0.20, 0.25, 0.99]:
        for atl_edge in [0.10, 0.12, 0.15, 0.20, 0.99]:
            for wel_edge in [0.08, 0.10, 0.12, 0.15, 0.20, 0.99]:
                o = copy.deepcopy(EXCLUDE)
                if nyc_edge < 0.99:
                    o["nyc"] = {"min_edge": nyc_edge}
                else:
                    o["nyc"] = {"min_edge": 0.99}
                if atl_edge < 0.99:
                    o["atlanta"] = {"min_edge": atl_edge}
                else:
                    o["atlanta"] = {"min_edge": 0.99}
                if wel_edge < 0.99:
                    o["wellington"] = {"min_edge": wel_edge}
                else:
                    o["wellington"] = {"min_edge": 0.99}
                desc = f"nyc={nyc_edge}_atl={atl_edge}_wel={wel_edge}"
                experiments.append((desc, o))

    # === Try wider MAX_PRICE for top performers ===
    top_cities = ["paris", "seattle", "london", "lucknow", "miami", "tel_aviv"]
    for mp in [0.45, 0.48, 0.50]:
        o = copy.deepcopy(EXCLUDE)
        o["nyc"] = {"min_edge": 0.99}
        o["atlanta"] = {"min_edge": 0.99}
        o["wellington"] = {"min_edge": 0.99}
        for c in top_cities:
            o[c] = {"max_price": mp}
        experiments.append((f"top6_mp={mp}_excl6", o))

    # === Combined: best marginal + top expansion ===
    for mp in [0.45, 0.48, 0.50]:
        o = copy.deepcopy(EXCLUDE)
        o["nyc"] = {"min_edge": 0.99}
        o["atlanta"] = {"min_edge": 0.99}
        # Keep wellington with higher edge
        o["wellington"] = {"min_edge": 0.12}
        for c in top_cities:
            o[c] = {"max_price": mp}
        experiments.append((f"top6_mp={mp}_wel=0.12_excl5", o))

    # Run
    print(f"\nRunning {len(experiments)} experiments...")
    print(f"{'#':>4} {'Description':<40} {'Score':>10} {'Trades':>7} {'Sharpe':>8} {'WR':>6} {'delta':>8}")
    print("-" * 85)

    for i, (desc, overrides) in enumerate(experiments):
        results, n = run_with_overrides(overrides, data, normals)
        score = results["composite_score"]
        delta = score - base_score
        marker = " **" if score > best_score else ""
        print(f"{i+1:>4} {desc:<40} {score:>10.4f} {n:>7} "
              f"{results['avg_sharpe']:>8.2f} {results['avg_win_rate']:>5.1f}% "
              f"{delta:>+8.2f}{marker}")
        if score > best_score:
            best_score = score
            best_overrides = overrides
            best_desc = desc

    print(f"\n{'=' * 78}")
    print(f"BEST: {best_desc}")
    print(f"  Score: {best_score:.4f} (baseline: {base_score:.4f}, +{best_score - base_score:.2f})")
    print(f"\nCITY_OVERRIDES = {json.dumps(best_overrides, indent=2)}")


if __name__ == "__main__":
    main()
