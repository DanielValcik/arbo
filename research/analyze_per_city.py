"""
Per-city performance analysis for Strategy C.

Runs the baseline backtest and breaks down results by city to identify:
- Which cities drive PnL
- Which cities have best/worst win rates
- Optimal parameter ranges per city
- Where per-city tuning would help most
"""
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

# Add research dir to path
sys.path.insert(0, str(Path(__file__).parent))
import backtest_harness as harness
import strategy_experiment as strategy

# Run baseline backtest across all windows, collect trades
harness.download_historical_data()
data = harness.load_data()
normals = harness.compute_monthly_normals(data)

all_trades = []
for i, window in enumerate(harness.WALK_FORWARD_WINDOWS):
    test_start, test_end = window["test"]
    seed = harness.BASE_SEED + i * 7919
    trades, _ = harness.run_single_backtest(data, normals, test_start, test_end, seed)
    all_trades.extend(trades)

print(f"Total trades: {len(all_trades)}")
print()

# Per-city breakdown
city_stats = defaultdict(lambda: {
    "trades": 0, "wins": 0, "pnl": 0.0, "total_size": 0.0,
    "edges": [], "prices": [], "probs": [],
})

for t in all_trades:
    cs = city_stats[t.city]
    cs["trades"] += 1
    cs["wins"] += 1 if t.won else 0
    cs["pnl"] += t.pnl
    cs["total_size"] += t.size
    cs["edges"].append(t.edge)
    cs["prices"].append(t.market_price)
    cs["probs"].append(t.forecast_prob)

# Print per-city table
print(f"{'City':<16} {'Trades':>6} {'Wins':>5} {'WR%':>6} {'PnL':>10} {'ROI%':>7} "
      f"{'Avg Edge':>8} {'Avg Price':>9} {'Med Price':>9} {'Avg Prob':>8}")
print("-" * 110)

city_rows = []
for city_id in sorted(city_stats.keys()):
    cs = city_stats[city_id]
    wr = cs["wins"] / cs["trades"] * 100 if cs["trades"] else 0
    roi = cs["pnl"] / cs["total_size"] * 100 if cs["total_size"] else 0
    avg_edge = sum(cs["edges"]) / len(cs["edges"]) if cs["edges"] else 0
    avg_price = sum(cs["prices"]) / len(cs["prices"]) if cs["prices"] else 0
    med_price = sorted(cs["prices"])[len(cs["prices"])//2] if cs["prices"] else 0
    avg_prob = sum(cs["probs"]) / len(cs["probs"]) if cs["probs"] else 0

    city_rows.append({
        "city": city_id, "trades": cs["trades"], "wins": cs["wins"],
        "wr": wr, "pnl": cs["pnl"], "roi": roi,
        "avg_edge": avg_edge, "avg_price": avg_price, "med_price": med_price,
        "avg_prob": avg_prob,
    })
    print(f"{city_id:<16} {cs['trades']:>6} {cs['wins']:>5} {wr:>5.1f}% "
          f"{cs['pnl']:>10.1f} {roi:>6.1f}% "
          f"{avg_edge:>8.3f} {avg_price:>9.3f} {med_price:>9.3f} {avg_prob:>8.3f}")

# Sort by PnL contribution
print()
print("=== TOP CITIES BY PnL ===")
for r in sorted(city_rows, key=lambda x: x["pnl"], reverse=True)[:10]:
    print(f"  {r['city']:<16} PnL={r['pnl']:>8.1f}  trades={r['trades']:>4}  WR={r['wr']:.1f}%  "
          f"avg_edge={r['avg_edge']:.3f}  avg_price={r['avg_price']:.3f}")

print()
print("=== WORST CITIES BY PnL ===")
for r in sorted(city_rows, key=lambda x: x["pnl"])[:5]:
    print(f"  {r['city']:<16} PnL={r['pnl']:>8.1f}  trades={r['trades']:>4}  WR={r['wr']:.1f}%  "
          f"avg_edge={r['avg_edge']:.3f}  avg_price={r['avg_price']:.3f}")

# Edge distribution per city
print()
print("=== EDGE DISTRIBUTION ===")
print(f"{'City':<16} {'p10':>6} {'p25':>6} {'p50':>6} {'p75':>6} {'p90':>6} {'max':>6}")
print("-" * 58)
for city_id in sorted(city_stats.keys()):
    edges = sorted(city_stats[city_id]["edges"])
    if not edges:
        continue
    n = len(edges)
    def pct(p):
        return edges[min(int(n * p / 100), n-1)]
    print(f"{city_id:<16} {pct(10):>6.3f} {pct(25):>6.3f} {pct(50):>6.3f} "
          f"{pct(75):>6.3f} {pct(90):>6.3f} {max(edges):>6.3f}")

# Price distribution per city
print()
print("=== PRICE DISTRIBUTION (where we trade) ===")
print(f"{'City':<16} {'p10':>6} {'p25':>6} {'p50':>6} {'p75':>6} {'p90':>6}")
print("-" * 52)
for city_id in sorted(city_stats.keys()):
    prices = sorted(city_stats[city_id]["prices"])
    if not prices:
        continue
    n = len(prices)
    def pct(p):
        return prices[min(int(n * p / 100), n-1)]
    print(f"{city_id:<16} {pct(10):>6.3f} {pct(25):>6.3f} {pct(50):>6.3f} "
          f"{pct(75):>6.3f} {pct(90):>6.3f}")

# Suggestions for per-city tuning
print()
print("=== PER-CITY TUNING SUGGESTIONS ===")
for r in sorted(city_rows, key=lambda x: x["pnl"], reverse=True):
    city = r["city"]
    suggestions = []
    if r["wr"] > 72:
        suggestions.append("high WR → can lower min_edge or raise max_price")
    if r["wr"] < 62:
        suggestions.append("low WR → raise min_edge or lower max_price")
    if r["avg_price"] > 0.40:
        suggestions.append(f"avg_price={r['avg_price']:.2f} → raise max_price")
    if r["pnl"] < 0:
        suggestions.append("NEGATIVE PnL → consider excluding or tighter gates")
    if suggestions:
        print(f"  {city:<16} {' | '.join(suggestions)}")

# Save detailed results
results = {
    "total_trades": len(all_trades),
    "per_city": city_rows,
}
results_file = Path(__file__).parent / "data" / "per_city_analysis.json"
with open(results_file, "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nSaved to {results_file}")
