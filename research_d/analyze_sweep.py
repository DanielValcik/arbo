"""Analyze sweep results."""
import csv, sys
from pathlib import Path

tsv = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("research_d/data/sweep_d_results.tsv")
rows = []
with open(tsv) as f:
    for r in csv.DictReader(f, delimiter="\t"):
        rows.append(r)

print(f"Total experiments: {len(rows)}\n")

hdr = f"{'#':>3} {'score':>7} {'pnl':>9} {'trades':>6} {'WR':>5} {'GB':>5} {'DD':>5} {'Sharpe':>7} {'edge':>5} {'delta':>5} {'SL':>6}"

by_score = sorted(rows, key=lambda r: float(r["score"]), reverse=True)
print("=== TOP 10 BY SCORE ===")
print(hdr)
for r in by_score[:10]:
    sl = r["sl_delta"] if r["sl_on"] == "True" else "off"
    wr = float(r["win_rate"])
    gb = float(r["gb_rate"])
    dd = float(r["max_dd"])
    print(f"{r['experiment']:>3} {float(r['score']):>7.1f} {float(r['pnl']):>8.0f}$ {int(r['trades']):>6} {wr:>4.0%} {gb:>4.0%} {dd:>4.0%} {float(r['sharpe']):>7.2f} {r['min_edge']:>5} {r['gb_delta']:>5} {sl:>6}")

print()
by_pnl = sorted(rows, key=lambda r: float(r["pnl"]), reverse=True)
print("=== TOP 10 BY PNL ===")
print(hdr)
for r in by_pnl[:10]:
    sl = r["sl_delta"] if r["sl_on"] == "True" else "off"
    wr = float(r["win_rate"])
    gb = float(r["gb_rate"])
    dd = float(r["max_dd"])
    print(f"{r['experiment']:>3} {float(r['score']):>7.1f} {float(r['pnl']):>8.0f}$ {int(r['trades']):>6} {wr:>4.0%} {gb:>4.0%} {dd:>4.0%} {float(r['sharpe']):>7.2f} {r['min_edge']:>5} {r['gb_delta']:>5} {sl:>6}")

print()
profitable = [r for r in rows if float(r["pnl"]) > 0]
print(f"Profitable: {len(profitable)}/{len(rows)} ({len(profitable)/len(rows)*100:.0f}%)")

# Pattern analysis
print("\n=== PATTERN: SL impact ===")
for sl_val in ["0.1", "0.15", "off"]:
    subset = [r for r in rows if (r["sl_delta"] if r["sl_on"]=="True" else "off") == sl_val]
    if subset:
        avg_pnl = sum(float(r["pnl"]) for r in subset) / len(subset)
        n_prof = sum(1 for r in subset if float(r["pnl"]) > 0)
        print(f"  SL={sl_val}: avg PnL=${avg_pnl:.0f}, {n_prof}/{len(subset)} profitable")

print("\n=== PATTERN: Delta impact ===")
for d in ["0.05", "0.1", "0.15", "0.2"]:
    subset = [r for r in rows if r["gb_delta"] == d]
    if subset:
        avg_pnl = sum(float(r["pnl"]) for r in subset) / len(subset)
        n_prof = sum(1 for r in subset if float(r["pnl"]) > 0)
        print(f"  Delta={d}: avg PnL=${avg_pnl:.0f}, {n_prof}/{len(subset)} profitable")

print("\n=== PATTERN: min_edge impact ===")
for e in ["0.03", "0.05", "0.08", "0.12", "0.15"]:
    subset = [r for r in rows if r["min_edge"] == e]
    if subset:
        avg_pnl = sum(float(r["pnl"]) for r in subset) / len(subset)
        n_prof = sum(1 for r in subset if float(r["pnl"]) > 0)
        best = max(subset, key=lambda r: float(r["pnl"]))
        print(f"  edge={e}: avg PnL=${avg_pnl:.0f}, {n_prof}/{len(subset)} profitable, best=${float(best['pnl']):.0f}")
