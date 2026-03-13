"""
Per-city parameter sweep for Strategy C.

Systematically tests CITY_OVERRIDES combinations to find optimal
per-city thresholds (min_edge, min_price, max_price).
"""

import copy
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import strategy_experiment as strategy
import backtest_harness as harness


def run_with_overrides(overrides, data, normals):
    """Run backtest with specific CITY_OVERRIDES, return composite + per-city stats."""
    original = strategy.CITY_OVERRIDES
    strategy.CITY_OVERRIDES = overrides

    results = harness.walk_forward_evaluate(data, normals)

    # Collect per-city stats
    all_trades = []
    for i, window in enumerate(harness.WALK_FORWARD_WINDOWS):
        seed = harness.BASE_SEED + i * 7919
        trades_w, _ = harness.run_single_backtest(
            data, normals, window["test"][0], window["test"][1], seed
        )
        all_trades.extend(trades_w)

    city_stats = {}
    for t in all_trades:
        if t.city not in city_stats:
            city_stats[t.city] = {"trades": 0, "wins": 0, "pnl": 0.0, "size": 0.0}
        cs = city_stats[t.city]
        cs["trades"] += 1
        cs["wins"] += 1 if t.won else 0
        cs["pnl"] += t.pnl
        cs["size"] += t.size

    strategy.CITY_OVERRIDES = original
    return results, city_stats, len(all_trades)


def main():
    harness.download_historical_data()
    data = harness.load_data()
    normals = harness.compute_monthly_normals(data)

    # Baseline
    print("=" * 78)
    print("BASELINE (no city overrides)")
    print("=" * 78)
    base_results, base_city, base_n = run_with_overrides({}, data, normals)
    base_score = base_results["composite_score"]
    print(f"  Score: {base_score:.4f}  Trades: {base_n}  Sharpe: {base_results['avg_sharpe']:.2f}  "
          f"WinRate: {base_results['avg_win_rate']:.1f}%")

    best_score = base_score
    best_overrides = {}
    best_desc = "baseline"

    experiments = []

    # === Phase 1: Exclude unprofitable cities ===
    unprofitable = ["dc", "toronto", "buenos_aires"]
    marginal = ["nyc", "atlanta", "wellington"]

    # Test: exclude unprofitable (set min_edge very high)
    experiments.append(("exclude_unprofitable", {
        c: {"min_edge": 0.99} for c in unprofitable
    }))

    # Test: exclude unprofitable + marginal
    experiments.append(("exclude_unprof+marginal", {
        c: {"min_edge": 0.99} for c in unprofitable + marginal
    }))

    # === Phase 2: Per-city min_edge based on sigma ===
    # Higher sigma = less accurate = need more edge
    sigma_based = {}
    for city, sigmas in strategy.CITY_SIGMA.items():
        s = sigmas.get(0, 2.0)
        if s < 1.0:      # Tight sigma (Paris, Seattle, London, Lucknow)
            sigma_based[city] = {"min_edge": 0.06}
        elif s < 1.2:    # Medium-tight (Miami, Tel Aviv, NYC, Chicago)
            sigma_based[city] = {"min_edge": 0.07}
        elif s < 1.4:    # Medium (Dallas, Seoul, Atlanta, Sao Paulo)
            sigma_based[city] = {"min_edge": 0.09}
        elif s < 1.6:    # Wide (Toronto, Buenos Aires, Ankara, Munich)
            sigma_based[city] = {"min_edge": 0.12}
        else:            # Very wide (Tokyo, DC, LA)
            sigma_based[city] = {"min_edge": 0.15}
    experiments.append(("sigma_based_min_edge", sigma_based))

    # === Phase 3: Sigma-based + exclude worst ===
    sigma_plus_exclude = copy.deepcopy(sigma_based)
    for c in unprofitable:
        sigma_plus_exclude[c] = {"min_edge": 0.99}
    experiments.append(("sigma_based+exclude_worst", sigma_plus_exclude))

    # === Phase 4: Per-city price ranges ===
    # Tight sigma cities can trade wider price range
    sigma_price = copy.deepcopy(sigma_based)
    for city, sigmas in strategy.CITY_SIGMA.items():
        s = sigmas.get(0, 2.0)
        if s < 1.0:
            sigma_price[city]["max_price"] = 0.48
        elif s < 1.2:
            sigma_price[city]["max_price"] = 0.45
    for c in unprofitable:
        sigma_price[c] = {"min_edge": 0.99}
    experiments.append(("sigma_price_range+exclude", sigma_price))

    # === Phase 5: Fine-tune min_edge ===
    for edge_tight in [0.04, 0.05, 0.06]:
        for edge_wide in [0.12, 0.15, 0.18, 0.20]:
            overrides = {}
            for city, sigmas in strategy.CITY_SIGMA.items():
                s = sigmas.get(0, 2.0)
                if s < 1.0:
                    overrides[city] = {"min_edge": edge_tight}
                elif s < 1.2:
                    overrides[city] = {"min_edge": edge_tight + 0.01}
                elif s < 1.4:
                    overrides[city] = {"min_edge": 0.09}
                else:
                    overrides[city] = {"min_edge": edge_wide}
            for c in unprofitable:
                overrides[c] = {"min_edge": 0.99}
            experiments.append((f"tight={edge_tight}_wide={edge_wide}+excl", overrides))

    # === Phase 6: Per-city max_price sweep for tight-sigma cities ===
    for mp in [0.45, 0.48, 0.50, 0.52]:
        overrides = copy.deepcopy(best_overrides) if best_overrides else {}
        for city, sigmas in strategy.CITY_SIGMA.items():
            s = sigmas.get(0, 2.0)
            if s < 1.0:
                if city not in overrides:
                    overrides[city] = {}
                overrides[city]["max_price"] = mp
        for c in unprofitable:
            overrides[c] = {"min_edge": 0.99}
        experiments.append((f"tight_max_price={mp}+excl", overrides))

    # Run all experiments
    print(f"\nRunning {len(experiments)} experiments...")
    print(f"{'#':>3} {'Description':<40} {'Score':>10} {'Trades':>7} {'Sharpe':>8} {'WR':>6} {'vs base':>8}")
    print("-" * 85)

    for i, (desc, overrides) in enumerate(experiments):
        t0 = time.time()
        results, city_stats, n_trades = run_with_overrides(overrides, data, normals)
        score = results["composite_score"]
        delta = score - base_score
        marker = " **" if score > best_score else ""

        print(f"{i+1:>3} {desc:<40} {score:>10.4f} {n_trades:>7} "
              f"{results['avg_sharpe']:>8.2f} {results['avg_win_rate']:>5.1f}% "
              f"{delta:>+8.2f}{marker}")

        if score > best_score:
            best_score = score
            best_overrides = overrides
            best_desc = desc

    # Summary
    print("\n" + "=" * 78)
    print(f"BEST: {best_desc}")
    print(f"  Score: {best_score:.4f} (baseline: {base_score:.4f}, +{best_score - base_score:.2f})")
    print(f"\nCITY_OVERRIDES = {json.dumps(best_overrides, indent=2)}")

    # Show per-city stats for best
    print("\nPer-city with best overrides:")
    _, best_city, _ = run_with_overrides(best_overrides, data, normals)
    print(f"  {'City':<16} {'Trades':>6} {'WR':>6} {'PnL':>10} {'ROI':>8}")
    for city in sorted(best_city.keys(), key=lambda c: -best_city[c]["pnl"]):
        cs = best_city[city]
        wr = cs["wins"] / cs["trades"] * 100 if cs["trades"] > 0 else 0
        roi = cs["pnl"] / cs["size"] * 100 if cs["size"] > 0 else 0
        print(f"  {city:<16} {cs['trades']:>6} {wr:>5.1f}% ${cs['pnl']:>9.2f} {roi:>7.1f}%")


if __name__ == "__main__":
    main()
