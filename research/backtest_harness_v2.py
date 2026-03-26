"""
Autoresearch Backtest Harness V2 — EXIT LOGIC
==============================================

Extension of backtest_harness.py that adds:
- Position exit checks at day-0 for multi-day holds
- Rolling to adjacent buckets on exit
- Partial profit-taking
- Counterfactual tracking (A/B: what if we held to resolution?)

Uses strategy_experiment_v2.py for parameters + exit logic.

Usage: python3 research/backtest_harness_v2.py

Compare with baseline: python3 research/backtest_harness.py
"""

import math
import random
import sys
import time
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import strategy_experiment_v2 as strategy

# Import shared infrastructure from v1 (data management, market simulation)
from backtest_harness import (
    download_historical_data,
    load_data,
    compute_monthly_normals,
    CITIES,
    WALK_FORWARD_WINDOWS,
    INITIAL_CAPITAL,
    SLIPPAGE_PCT,
    GAS_COST_USD,
    MAX_SIZING_CAPITAL,
    BASE_SEED,
    OUR_FORECAST_NOISE,
    generate_buckets,
    bucket_contains,
    generate_market,
)


# ═══════════════════════════════════════════════════════════════════════════════
# SIMULATE FORECAST (uses v2 strategy module)
# ═══════════════════════════════════════════════════════════════════════════════

def simulate_forecast(actual_temp_c, days_out, rng, city=None):
    """Simulate our weather forecast (same as v1 but references v2 strategy)."""
    noise_std = OUR_FORECAST_NOISE.get(days_out, 5.0)
    forecast_bias = 0.0
    if city:
        forecast_bias = -strategy.CITY_BIAS.get(city, 0.0)
    return actual_temp_c + forecast_bias + rng.gauss(0, noise_std)


# ═══════════════════════════════════════════════════════════════════════════════
# EXTENDED TRADE DATA
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Trade:
    city: str
    date: str
    bucket_label: str
    days_out: int
    market_price: float
    forecast_prob: float
    edge: float
    size: float
    fill_price: float
    won: bool
    pnl: float
    capital_after: float
    # V2 extensions
    exit_reason: str = "resolution"        # resolution | stop_loss | edge_lost | prob_floor | profit_take | roll_entry
    counterfactual_pnl: float | None = None  # P&L if held to resolution (for exits)
    counterfactual_won: bool | None = None   # Would have won if held (for exits)


# ═══════════════════════════════════════════════════════════════════════════════
# BACKTEST ENGINE WITH EXIT LOGIC
# ═══════════════════════════════════════════════════════════════════════════════

def run_single_backtest(data, normals, test_start, test_end, seed):
    """Run backtest over a single date range with exit logic.

    Key change from v1: for trades entered at days_out > 0, a day-0
    exit check is performed before resolution. The day-0 forecast is
    more accurate (lower sigma), simulating the METAR-informed exit
    decision in production.

    Returns (trades, final_capital).
    """
    forecast_rng = random.Random(seed)
    market_rng = random.Random(seed + 10000)
    # Separate RNG for exit simulation — does NOT affect entry RNG sequence
    exit_rng = random.Random(seed + 20000)

    capital = INITIAL_CAPITAL
    trades = []
    daily_deployed = 0.0

    start_d = date.fromisoformat(test_start)
    end_d = date.fromisoformat(test_end)
    dates = []
    d = start_d
    while d <= end_d:
        dates.append(d.isoformat())
        d += timedelta(days=1)

    for dt_str in dates:
        month = int(dt_str.split("-")[1])
        daily_deployed = 0.0

        # ── Pre-compute day-0 forecasts for exit checks ──
        # One forecast per city (temperature forecast is city-level, not bucket-level)
        day0_city_forecasts = {}
        if strategy.EXIT_ENABLED:
            for city_id in sorted(CITIES.keys()):
                if dt_str in data.get(city_id, {}):
                    actual_high = data[city_id][dt_str]["high_c"]
                    day0_city_forecasts[city_id] = simulate_forecast(
                        actual_high, 0, exit_rng, city_id
                    )

        # ── Collect trade candidates (identical to v1) ──
        candidates = []
        for city_id in CITIES:
            if dt_str not in data.get(city_id, {}):
                continue

            actual_high = data[city_id][dt_str]["high_c"]
            monthly_avg = normals[city_id].get(month)
            if monthly_avg is None:
                continue

            buckets = generate_buckets(monthly_avg)

            for bucket_idx, bucket in enumerate(buckets):
                best_candidate = None
                best_edge = -1.0

                for days_out in strategy.DAYS_OUT_TO_TRADE:
                    forecast = simulate_forecast(
                        actual_high, days_out, forecast_rng, city=city_id
                    )
                    market_price, volume, liquidity = generate_market(
                        actual_high, bucket, days_out, market_rng
                    )

                    prob = strategy.estimate_probability(
                        forecast, bucket["low_c"], bucket["high_c"], days_out,
                        city=city_id,
                    )
                    edge = prob - market_price

                    if not strategy.should_trade(
                        edge, prob, market_price, days_out, volume, liquidity,
                        city=city_id,
                    ):
                        continue

                    if edge > best_edge:
                        best_edge = edge
                        best_candidate = {
                            "city": city_id,
                            "bucket_idx": bucket_idx,
                            "bucket": bucket,
                            "buckets": buckets,  # All buckets for roll logic
                            "days_out": days_out,
                            "market_price": market_price,
                            "forecast_prob": prob,
                            "edge": edge,
                            "actual_high": actual_high,
                        }

                if best_candidate is not None:
                    candidates.append(best_candidate)

        # Sort by edge descending
        candidates.sort(key=lambda c: -c["edge"])

        # Deduplicate: max one trade per (city, bucket)
        seen = set()
        for cand in candidates:
            key = (cand["city"], cand["bucket_idx"])
            if key in seen:
                continue
            seen.add(key)

            # Capital check
            sizing_capital = min(capital, MAX_SIZING_CAPITAL)
            max_exposure = sizing_capital * getattr(strategy, "MAX_TOTAL_EXPOSURE_PCT", 0.80)
            available = max(0, max_exposure - daily_deployed)
            if available < 1.0:
                break

            size = strategy.position_size(
                cand["edge"], cand["market_price"], available, sizing_capital,
                city=cand["city"],
            )
            if size <= 0 or size > available:
                continue

            # ── Execute trade ──
            fill_price = min(0.99, cand["market_price"] * (1.0 + SLIPPAGE_PCT))
            won = bucket_contains(cand["bucket"], cand["actual_high"])
            shares = size / fill_price

            # Always compute resolution P&L (needed for counterfactual)
            if won:
                resolution_pnl = size * (1.0 / fill_price - 1.0) - GAS_COST_USD
            else:
                resolution_pnl = -size - GAS_COST_USD

            exit_reason = "resolution"
            actual_pnl = resolution_pnl
            cf_pnl = None
            cf_won = None

            # ── EXIT CHECK for multi-day positions ──
            if cand["days_out"] > 0 and strategy.EXIT_ENABLED:
                city = cand["city"]
                day0_forecast = day0_city_forecasts.get(city)

                if day0_forecast is not None:
                    # Day-0 market price for this specific bucket
                    day0_price, _, _ = generate_market(
                        cand["actual_high"], cand["bucket"], 0, exit_rng
                    )
                    day0_prob = strategy.estimate_probability(
                        day0_forecast, cand["bucket"]["low_c"],
                        cand["bucket"]["high_c"], 0, city=city,
                    )
                    day0_edge = day0_prob - day0_price
                    pnl_pct = (day0_price - fill_price) / fill_price

                    exit_decision = strategy.check_exit(
                        entry_price=fill_price,
                        current_price=day0_price,
                        updated_prob=day0_prob,
                        updated_edge=day0_edge,
                        edge_at_entry=cand["edge"],
                        pnl_pct=pnl_pct,
                        city=city,
                    )

                    if exit_decision["action"] in ("exit", "roll"):
                        # Sell at day-0 price with taker slippage
                        sell_price = max(0.01, day0_price * (1.0 - SLIPPAGE_PCT))
                        # Double gas: entry + exit
                        actual_pnl = shares * sell_price - size - 2 * GAS_COST_USD
                        exit_reason = exit_decision["reason"]
                        cf_pnl = resolution_pnl
                        cf_won = won

                        capital += actual_pnl
                        daily_deployed += size

                        trades.append(Trade(
                            city=city, date=dt_str,
                            bucket_label=cand["bucket"]["label"],
                            days_out=cand["days_out"],
                            market_price=cand["market_price"],
                            forecast_prob=cand["forecast_prob"],
                            edge=cand["edge"],
                            size=size, fill_price=fill_price,
                            won=won, pnl=actual_pnl,
                            capital_after=capital,
                            exit_reason=exit_reason,
                            counterfactual_pnl=cf_pnl,
                            counterfactual_won=cf_won,
                        ))

                        # ── ROLL: find better adjacent bucket ──
                        if exit_decision["action"] == "roll" and strategy.ROLL_ENABLED:
                            best_roll_edge = strategy.MIN_ROLL_EDGE
                            best_roll = None

                            for alt_idx, alt_bucket in enumerate(cand["buckets"]):
                                if alt_idx == cand["bucket_idx"]:
                                    continue
                                alt_key = (city, alt_idx)
                                if alt_key in seen:
                                    continue

                                alt_price, alt_vol, alt_liq = generate_market(
                                    cand["actual_high"], alt_bucket, 0, exit_rng
                                )
                                alt_prob = strategy.estimate_probability(
                                    day0_forecast,
                                    alt_bucket["low_c"],
                                    alt_bucket["high_c"],
                                    0, city=city,
                                )
                                alt_edge = alt_prob - alt_price

                                if (alt_edge > best_roll_edge
                                        and strategy.should_trade(
                                            alt_edge, alt_prob, alt_price,
                                            0, alt_vol, alt_liq, city=city)):
                                    best_roll_edge = alt_edge
                                    best_roll = {
                                        "idx": alt_idx,
                                        "bucket": alt_bucket,
                                        "price": alt_price,
                                        "prob": alt_prob,
                                        "edge": alt_edge,
                                    }

                            if best_roll is not None:
                                roll_available = max(0, max_exposure - daily_deployed)
                                roll_size = strategy.position_size(
                                    best_roll["edge"], best_roll["price"],
                                    roll_available, sizing_capital, city=city,
                                )
                                if roll_size > 0:
                                    roll_fill = min(0.99, best_roll["price"] * (1.0 + SLIPPAGE_PCT))
                                    roll_won = bucket_contains(
                                        best_roll["bucket"], cand["actual_high"]
                                    )
                                    if roll_won:
                                        roll_pnl = roll_size * (1.0 / roll_fill - 1.0) - GAS_COST_USD
                                    else:
                                        roll_pnl = -roll_size - GAS_COST_USD

                                    capital += roll_pnl
                                    daily_deployed += roll_size
                                    seen.add((city, best_roll["idx"]))

                                    trades.append(Trade(
                                        city=city, date=dt_str,
                                        bucket_label=best_roll["bucket"]["label"],
                                        days_out=0,
                                        market_price=best_roll["price"],
                                        forecast_prob=best_roll["prob"],
                                        edge=best_roll["edge"],
                                        size=roll_size, fill_price=roll_fill,
                                        won=roll_won, pnl=roll_pnl,
                                        capital_after=capital,
                                        exit_reason="roll_entry",
                                    ))

                        if capital <= 0:
                            return trades, 0.0
                        continue  # Skip normal resolution

                    elif exit_decision["action"] == "partial_exit":
                        fraction = exit_decision.get("fraction", 0.5)
                        sell_price = max(0.01, day0_price * (1.0 - SLIPPAGE_PCT))

                        # Partial: sell fraction at day-0 price
                        exit_shares = shares * fraction
                        partial_pnl = exit_shares * sell_price - size * fraction - GAS_COST_USD

                        # Hold: remaining fraction resolves normally
                        hold_size = size * (1 - fraction)
                        if won:
                            hold_pnl = hold_size * (1.0 / fill_price - 1.0) - GAS_COST_USD
                        else:
                            hold_pnl = -hold_size - GAS_COST_USD

                        actual_pnl = partial_pnl + hold_pnl
                        exit_reason = "profit_take"
                        cf_pnl = resolution_pnl
                        cf_won = won

                        capital += actual_pnl
                        daily_deployed += size

                        trades.append(Trade(
                            city=cand["city"], date=dt_str,
                            bucket_label=cand["bucket"]["label"],
                            days_out=cand["days_out"],
                            market_price=cand["market_price"],
                            forecast_prob=cand["forecast_prob"],
                            edge=cand["edge"],
                            size=size, fill_price=fill_price,
                            won=won, pnl=actual_pnl,
                            capital_after=capital,
                            exit_reason=exit_reason,
                            counterfactual_pnl=cf_pnl,
                            counterfactual_won=cf_won,
                        ))

                        if capital <= 0:
                            return trades, 0.0
                        continue  # Skip normal resolution

            # ── Normal resolution (day-0 immediate OR no exit triggered) ──
            capital += actual_pnl
            daily_deployed += size

            trades.append(Trade(
                city=cand["city"], date=dt_str,
                bucket_label=cand["bucket"]["label"],
                days_out=cand["days_out"],
                market_price=cand["market_price"],
                forecast_prob=cand["forecast_prob"],
                edge=cand["edge"],
                size=size, fill_price=fill_price,
                won=won, pnl=actual_pnl,
                capital_after=capital,
                exit_reason=exit_reason,
            ))

            if capital <= 0:
                return trades, 0.0

    return trades, capital


# ═══════════════════════════════════════════════════════════════════════════════
# METRICS (extended with exit analysis)
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_metrics(trades, initial_capital):
    """Calculate performance metrics + exit-specific analysis."""
    if not trades:
        return {
            "composite_score": 0.0, "sharpe_ratio": 0.0, "total_pnl": 0.0,
            "total_pnl_pct": 0.0, "max_drawdown_pct": 0.0, "win_rate_pct": 0.0,
            "num_trades": 0, "profit_factor": 0.0, "avg_return_pct": 0.0,
            "avg_win_pct": 0.0, "avg_loss_pct": 0.0,
            "exit_count": 0, "exit_by_reason": {},
            "saves": 0, "regrets": 0,
            "avg_save_amount": 0.0, "avg_regret_amount": 0.0,
            "roll_count": 0, "roll_win_rate": 0.0,
        }

    # ── Standard metrics (same as v1) ──
    returns = [t.pnl / t.size if t.size > 0 else 0 for t in trades]
    pnls = [t.pnl for t in trades]
    n = len(returns)
    total_pnl = sum(pnls)

    win_returns = [r for r in returns if r > 0]
    loss_returns = [r for r in returns if r <= 0]
    win_rate = len(win_returns) / n * 100

    mean_ret = sum(returns) / n
    if n > 1:
        variance = sum((r - mean_ret) ** 2 for r in returns) / (n - 1)
        std_ret = math.sqrt(variance) if variance > 0 else 1e-6
        sharpe = (mean_ret / std_ret) * math.sqrt(252)
    else:
        sharpe = 0.0

    peak = initial_capital
    max_dd = 0.0
    equity = initial_capital
    for t in trades:
        equity += t.pnl
        peak = max(peak, equity)
        dd = (peak - equity) / peak * 100 if peak > 0 else 0
        max_dd = max(max_dd, dd)

    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p <= 0)) or 1e-6
    profit_factor = gross_profit / gross_loss

    if n >= 10:
        trade_factor = math.sqrt(n / 100)
        dd_factor = max(0, 1.0 - max_dd / 50.0)
        composite = sharpe * trade_factor * dd_factor
    else:
        composite = 0.0

    # ── Exit-specific metrics ──
    exit_trades = [t for t in trades if t.exit_reason not in ("resolution", "roll_entry")]
    exit_count = len(exit_trades)

    exit_by_reason = {}
    for t in exit_trades:
        exit_by_reason[t.exit_reason] = exit_by_reason.get(t.exit_reason, 0) + 1

    # Saves vs regrets: compare exit P&L with counterfactual (hold to resolution)
    saves = 0
    regrets = 0
    save_amounts = []
    regret_amounts = []

    for t in exit_trades:
        if t.counterfactual_pnl is not None:
            benefit = t.pnl - t.counterfactual_pnl  # Positive = saved, negative = regret
            if benefit >= 0:
                saves += 1
                save_amounts.append(benefit)
            else:
                regrets += 1
                regret_amounts.append(benefit)

    avg_save = sum(save_amounts) / len(save_amounts) if save_amounts else 0.0
    avg_regret = sum(regret_amounts) / len(regret_amounts) if regret_amounts else 0.0

    # Roll stats
    roll_trades = [t for t in trades if t.exit_reason == "roll_entry"]
    roll_count = len(roll_trades)
    roll_win_rate = (
        sum(1 for t in roll_trades if t.won) / roll_count * 100
        if roll_count > 0 else 0.0
    )

    return {
        "composite_score": round(composite, 6),
        "sharpe_ratio": round(sharpe, 4),
        "total_pnl": round(total_pnl, 2),
        "total_pnl_pct": round(total_pnl / initial_capital * 100, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "win_rate_pct": round(win_rate, 1),
        "num_trades": n,
        "profit_factor": round(profit_factor, 4),
        "avg_return_pct": round(mean_ret * 100, 2),
        "avg_win_pct": round(sum(win_returns) / len(win_returns) * 100, 2) if win_returns else 0.0,
        "avg_loss_pct": round(sum(loss_returns) / len(loss_returns) * 100, 2) if loss_returns else 0.0,
        # Exit metrics
        "exit_count": exit_count,
        "exit_by_reason": exit_by_reason,
        "saves": saves,
        "regrets": regrets,
        "avg_save_amount": round(avg_save, 4),
        "avg_regret_amount": round(avg_regret, 4),
        "roll_count": roll_count,
        "roll_win_rate": round(roll_win_rate, 1),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# WALK-FORWARD VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

def walk_forward_evaluate(data, normals):
    """Run walk-forward validation across all windows (same as v1 + exit metrics)."""
    window_metrics = []

    for i, window in enumerate(WALK_FORWARD_WINDOWS):
        test_start, test_end = window["test"]
        seed = BASE_SEED + i * 7919

        trades, final_capital = run_single_backtest(
            data, normals, test_start, test_end, seed
        )

        metrics = calculate_metrics(trades, INITIAL_CAPITAL)
        metrics["window"] = i + 1
        metrics["test_period"] = f"{test_start} -> {test_end}"
        metrics["final_capital"] = round(final_capital, 2)
        window_metrics.append(metrics)

    n_windows = len(window_metrics)

    sharpes = [m["sharpe_ratio"] for m in window_metrics]
    avg_sharpe = sum(sharpes) / n_windows if sharpes else 0
    sharpe_std = 0.0
    if n_windows > 1:
        sharpe_std = (sum((s - avg_sharpe) ** 2 for s in sharpes) / (n_windows - 1)) ** 0.5

    total_trades = sum(m["num_trades"] for m in window_metrics)
    avg_win_rate = sum(m["win_rate_pct"] for m in window_metrics) / n_windows
    max_max_dd = max(m["max_drawdown_pct"] for m in window_metrics)
    avg_pnl_pct = sum(m["total_pnl_pct"] for m in window_metrics) / n_windows
    avg_profit_factor = sum(m["profit_factor"] for m in window_metrics) / n_windows
    avg_return_pct = sum(m["avg_return_pct"] for m in window_metrics) / n_windows
    profitable_windows = sum(1 for m in window_metrics if m["total_pnl"] > 0)

    # Aggregate exit metrics
    total_exits = sum(m["exit_count"] for m in window_metrics)
    total_saves = sum(m["saves"] for m in window_metrics)
    total_regrets = sum(m["regrets"] for m in window_metrics)
    total_rolls = sum(m["roll_count"] for m in window_metrics)

    # Aggregate exit_by_reason
    agg_exit_reasons = {}
    for m in window_metrics:
        for reason, count in m["exit_by_reason"].items():
            agg_exit_reasons[reason] = agg_exit_reasons.get(reason, 0) + count

    if total_trades >= 50:
        trade_factor = math.sqrt(total_trades / 100)
        dd_factor = max(0, 1.0 - max_max_dd / 50.0)
        consistency = profitable_windows / n_windows
        composite = avg_sharpe * trade_factor * dd_factor * consistency
    else:
        composite = 0.0

    return {
        "windows": window_metrics,
        "composite_score": round(composite, 6),
        "avg_sharpe": round(avg_sharpe, 4),
        "sharpe_std": round(sharpe_std, 4),
        "total_trades": total_trades,
        "avg_win_rate": round(avg_win_rate, 1),
        "max_drawdown_pct": round(max_max_dd, 2),
        "avg_pnl_pct": round(avg_pnl_pct, 2),
        "avg_profit_factor": round(avg_profit_factor, 4),
        "avg_return_pct": round(avg_return_pct, 2),
        "profitable_windows": profitable_windows,
        "total_windows": n_windows,
        # Exit aggregates
        "total_exits": total_exits,
        "total_saves": total_saves,
        "total_regrets": total_regrets,
        "total_rolls": total_rolls,
        "exit_by_reason": agg_exit_reasons,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    t_start = time.time()

    download_historical_data()
    data = load_data()
    normals = compute_monthly_normals(data)

    print(f"Data: {len(data)} cities, "
          f"{sum(len(v) for v in data.values())} total day-records")

    results = walk_forward_evaluate(data, normals)

    t_end = time.time()

    # ── Per-city stats ──
    all_trades = []
    for wm in results["windows"]:
        seed = BASE_SEED + (wm["window"] - 1) * 7919
        test_start, test_end = WALK_FORWARD_WINDOWS[wm["window"] - 1]["test"]
        trades_w, _ = run_single_backtest(data, normals, test_start, test_end, seed)
        all_trades.extend(trades_w)

    city_stats = {}
    for t in all_trades:
        if t.city not in city_stats:
            city_stats[t.city] = {"trades": 0, "wins": 0, "pnl": 0.0, "size_total": 0.0,
                                  "exits": 0, "saves": 0, "regrets": 0}
        cs = city_stats[t.city]
        cs["trades"] += 1
        cs["wins"] += 1 if t.won else 0
        cs["pnl"] += t.pnl
        cs["size_total"] += t.size
        if t.exit_reason not in ("resolution", "roll_entry"):
            cs["exits"] += 1
            if t.counterfactual_pnl is not None:
                if t.pnl >= t.counterfactual_pnl:
                    cs["saves"] += 1
                else:
                    cs["regrets"] += 1

    print("\n" + "=" * 90)
    print("PER-CITY PERFORMANCE")
    print("=" * 90)
    print(f"  {'City':<16} {'Trades':>6} {'WinRate':>8} {'PnL':>10} {'ROI':>8} {'Exits':>6} {'Saves':>6} {'Regrets':>7}")
    print("  " + "-" * 72)

    for city in sorted(city_stats.keys(), key=lambda c: -city_stats[c]["pnl"]):
        cs = city_stats[city]
        wr = cs["wins"] / cs["trades"] * 100 if cs["trades"] > 0 else 0
        roi = cs["pnl"] / cs["size_total"] * 100 if cs["size_total"] > 0 else 0
        marker = "  " if cs["pnl"] >= 0 else "!!"
        print(f"{marker}{city:<16} {cs['trades']:>6} {wr:>7.1f}% ${cs['pnl']:>9.2f} {roi:>7.1f}% "
              f"{cs['exits']:>6} {cs['saves']:>6} {cs['regrets']:>7}")

    print(f"\n  Total: {len(all_trades)} trades across {len(city_stats)} cities")

    # ── Exit analysis ──
    if results["total_exits"] > 0:
        print("\n" + "=" * 90)
        print("EXIT ANALYSIS")
        print("=" * 90)
        print(f"  Total exits: {results['total_exits']} / {results['total_trades']} "
              f"({results['total_exits']/results['total_trades']*100:.1f}%)")
        for reason, count in sorted(results["exit_by_reason"].items()):
            print(f"    {reason}: {count}")
        print(f"  Saves (exit P&L > hold P&L): {results['total_saves']}")
        print(f"  Regrets (exit P&L < hold P&L): {results['total_regrets']}")
        if results["total_saves"] + results["total_regrets"] > 0:
            save_rate = results["total_saves"] / (results["total_saves"] + results["total_regrets"]) * 100
            print(f"  Save rate: {save_rate:.1f}%")
        if results["total_rolls"] > 0:
            print(f"  Rolls: {results['total_rolls']}")

    # ── Walk-forward results ──
    print("\n" + "=" * 90)
    print("WALK-FORWARD RESULTS")
    print("=" * 90)

    for wm in results["windows"]:
        status = "+" if wm["total_pnl"] > 0 else "-"
        exit_str = f"E:{wm['exit_count']}" if wm["exit_count"] > 0 else ""
        print(f"  W{wm['window']} {wm['test_period']}  |  "
              f"Sharpe {wm['sharpe_ratio']:>7.3f}  |  "
              f"PnL {wm['total_pnl_pct']:>7.1f}%  ({status})  |  "
              f"Win {wm['win_rate_pct']:>5.1f}%  |  "
              f"Trades {wm['num_trades']:>4d}  |  "
              f"MaxDD {wm['max_drawdown_pct']:>5.2f}% {exit_str}")

    # ── Summary ──
    print("\n" + "=" * 90)
    print("AGGREGATE (per-window averages)")
    print("=" * 90)
    print(f"  Profitable windows: {results['profitable_windows']}/{results['total_windows']}")
    print(f"  Avg Sharpe: {results['avg_sharpe']:.4f} +/- {results['sharpe_std']:.4f}")
    print(f"  Avg PnL: {results['avg_pnl_pct']:.1f}%  |  "
          f"Win rate: {results['avg_win_rate']:.1f}%  |  "
          f"Worst DD: {results['max_drawdown_pct']:.1f}%")

    # ── Greppable output ──
    print("\n---")
    print(f"composite_score:    {results['composite_score']:.6f}")
    print(f"avg_sharpe:         {results['avg_sharpe']:.4f}")
    print(f"sharpe_std:         {results['sharpe_std']:.4f}")
    print(f"avg_pnl_pct:        {results['avg_pnl_pct']:.2f}")
    print(f"max_drawdown_pct:   {results['max_drawdown_pct']:.2f}")
    print(f"avg_win_rate:       {results['avg_win_rate']:.1f}")
    print(f"num_trades:         {results['total_trades']}")
    print(f"avg_profit_factor:  {results['avg_profit_factor']:.4f}")
    print(f"avg_return_pct:     {results['avg_return_pct']:.2f}")
    print(f"profitable_windows: {results['profitable_windows']}/{results['total_windows']}")
    print(f"total_exits:        {results['total_exits']}")
    print(f"total_saves:        {results['total_saves']}")
    print(f"total_regrets:      {results['total_regrets']}")
    print(f"total_rolls:        {results['total_rolls']}")
    print(f"backtest_seconds:   {t_end - t_start:.1f}")


if __name__ == "__main__":
    main()
