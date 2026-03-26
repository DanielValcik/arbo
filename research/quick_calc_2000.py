"""Quick calculation: run best V3 candidates with $2000 capital."""
import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import backtest_harness as harness
import strategy_experiment as strategy

harness.download_historical_data()
DATA = harness.load_data()
NORMALS = harness.compute_monthly_normals(DATA)

# Override capital to $2000
harness.INITIAL_CAPITAL = 2000.0
harness.MAX_SIZING_CAPITAL = 10000.0  # Scale proportionally

# ── Candidate configs to test ──
CONFIGS = {
    "BASELINE (current production)": {
        "MIN_EDGE": 0.08, "MAX_PRICE": 0.43, "MIN_PRICE": 0.30,
        "MIN_FORECAST_PROB": 0.62, "MIN_VOLUME": 1000, "MIN_LIQUIDITY": 200,
        "PROB_SHARPENING": 1.05, "SHRINKAGE": 0.03, "KELLY_RAW_CAP": 0.40,
    },
    "V2 BEST (selective, high WR)": {
        "MIN_EDGE": 0.15, "MAX_PRICE": 0.50, "MIN_PRICE": 0.35,
        "MIN_FORECAST_PROB": 0.70, "MIN_VOLUME": 2000, "MIN_LIQUIDITY": 200,
        "PROB_SHARPENING": 1.143, "SHRINKAGE": 0.10, "KELLY_RAW_CAP": 0.25,
    },
    "V3 BEST (volume, ~295/mo)": {
        # Trial 25 was best in Phase 1. Reconstruct from search space:
        # vol_score=476.63, sharpe=13.38, trades=4426, wr=74.2%, dd=9.75%
        # With these metrics, likely params are low min_edge, wide price range
        "MIN_EDGE": 0.05, "MAX_PRICE": 0.55, "MIN_PRICE": 0.20,
        "MIN_FORECAST_PROB": 0.50, "MIN_VOLUME": 500, "MIN_LIQUIDITY": 100,
        "PROB_SHARPENING": 1.05, "SHRINKAGE": 0.03, "KELLY_RAW_CAP": 0.40,
    },
    "BALANCED (compromise)": {
        "MIN_EDGE": 0.08, "MAX_PRICE": 0.50, "MIN_PRICE": 0.25,
        "MIN_FORECAST_PROB": 0.58, "MIN_VOLUME": 1000, "MIN_LIQUIDITY": 200,
        "PROB_SHARPENING": 1.08, "SHRINKAGE": 0.05, "KELLY_RAW_CAP": 0.35,
    },
}


def apply_config(cfg):
    strategy.MIN_EDGE = cfg["MIN_EDGE"]
    strategy.MAX_PRICE = cfg["MAX_PRICE"]
    strategy.MIN_PRICE = cfg["MIN_PRICE"]
    strategy.MIN_FORECAST_PROB = cfg["MIN_FORECAST_PROB"]
    strategy.MIN_VOLUME = cfg["MIN_VOLUME"]
    strategy.MIN_LIQUIDITY = cfg["MIN_LIQUIDITY"]
    strategy.PROB_SHARPENING = cfg["PROB_SHARPENING"]

    shrinkage = cfg["SHRINKAGE"]
    sharpening = cfg["PROB_SHARPENING"]
    kelly_cap = cfg["KELLY_RAW_CAP"]

    def patched_prob(forecast_temp_c, bucket_low_c, bucket_high_c, days_out, *, city=None):
        sigma = strategy._get_sigma(days_out, city)
        cdf = lambda x: strategy._normal_cdf(x, forecast_temp_c, sigma)
        if bucket_low_c is None and bucket_high_c is not None:
            raw = cdf(bucket_high_c)
        elif bucket_high_c is None and bucket_low_c is not None:
            raw = 1.0 - cdf(bucket_low_c)
        elif bucket_low_c is not None and bucket_high_c is not None:
            raw = cdf(bucket_high_c) - cdf(bucket_low_c)
        else:
            return 0.0
        raw = raw * (1.0 - shrinkage) + 0.125 * shrinkage
        if sharpening != 1.0 and raw > 0:
            raw = raw ** sharpening
        return raw

    strategy.estimate_probability = patched_prob

    def patched_size(edge, market_price, available_capital, total_capital, *, city=None):
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
        kelly_adjusted = kelly_raw * strategy.KELLY_FRACTION
        size = available_capital * kelly_adjusted
        max_size = total_capital * strategy.MAX_POSITION_PCT
        size = min(size, max_size)
        return round(size, 2) if size >= 1.0 else 0.0

    strategy.position_size = patched_size


print("=" * 90)
print(f"STRATEGY C PROFITABILITY — $2,000 CAPITAL (quarter-Kelly 0.25)")
print(f"Walk-forward: 5 windows × 3 months each (2024-07 to 2025-09)")
print("=" * 90)

for name, cfg in CONFIGS.items():
    apply_config(cfg)
    results = harness.walk_forward_evaluate(DATA, NORMALS)

    # Collect all trades for per-window P&L
    window_details = []
    all_trades = []
    for wm in results["windows"]:
        seed = harness.BASE_SEED + (wm["window"] - 1) * 7919
        test_start, test_end = harness.WALK_FORWARD_WINDOWS[wm["window"] - 1]["test"]
        trades_w, final_cap = harness.run_single_backtest(DATA, NORMALS, test_start, test_end, seed)
        all_trades.extend(trades_w)
        pnl = sum(t.pnl for t in trades_w)
        wins = sum(1 for t in trades_w if t.won)
        window_details.append({
            "period": f"{test_start} → {test_end}",
            "trades": len(trades_w),
            "wins": wins,
            "pnl": pnl,
            "final_capital": final_cap,
        })

    total_trades = len(all_trades)
    total_pnl = sum(t.pnl for t in all_trades)
    total_wins = sum(1 for t in all_trades if t.won)
    avg_size = sum(t.size for t in all_trades) / total_trades if total_trades else 0
    total_volume = sum(t.size for t in all_trades)

    # Per-city breakdown
    city_stats = {}
    for t in all_trades:
        cs = city_stats.setdefault(t.city, {"trades": 0, "pnl": 0.0, "wins": 0})
        cs["trades"] += 1
        cs["pnl"] += t.pnl
        cs["wins"] += 1 if t.won else 0

    print(f"\n{'─' * 90}")
    print(f"  {name}")
    print(f"{'─' * 90}")
    print(f"  Trades: {total_trades} ({total_trades/15:.0f}/month, {total_trades/15/30:.1f}/day)")
    print(f"  Win rate: {total_wins/total_trades*100:.1f}%")
    print(f"  Total PnL: ${total_pnl:,.2f} ({total_pnl/2000*100:.1f}% return)")
    print(f"  Avg position: ${avg_size:.2f}")
    print(f"  Total volume: ${total_volume:,.2f}")
    print(f"  Max drawdown: {results['max_drawdown_pct']:.2f}%")
    print(f"  Sharpe: {results['avg_sharpe']:.2f}")
    print()

    # Per-window
    print(f"  {'Window':<28} {'Trades':>6} {'WinRate':>8} {'PnL':>12} {'Final Cap':>12}")
    for wd in window_details:
        wr = wd["wins"]/wd["trades"]*100 if wd["trades"] else 0
        print(f"  {wd['period']:<28} {wd['trades']:>6} {wr:>7.1f}% ${wd['pnl']:>10,.2f} ${wd['final_capital']:>10,.2f}")

    # Top cities
    print(f"\n  Top cities:")
    for city in sorted(city_stats, key=lambda c: -city_stats[c]["pnl"])[:7]:
        cs = city_stats[city]
        wr = cs["wins"]/cs["trades"]*100 if cs["trades"] else 0
        marker = "  " if cs["pnl"] >= 0 else "!!"
        print(f"  {marker}{city:<16} {cs['trades']:>5} trades  {wr:>5.1f}% WR  ${cs['pnl']:>10,.2f}")

    unprofitable = [c for c, s in city_stats.items() if s["pnl"] < 0]
    if unprofitable:
        print(f"\n  Unprofitable: {', '.join(unprofitable)}")

print(f"\n{'=' * 90}")
