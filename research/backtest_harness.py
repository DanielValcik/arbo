"""
Autoresearch Backtest Harness for Strategy C (Weather)
======================================================

FIXED — DO NOT MODIFY. The agent edits strategy_experiment.py only.

Downloads historical weather data from Open-Meteo archive API,
generates synthetic Polymarket-style weather markets, runs the
strategy through walk-forward validation, and reports metrics.

Usage: python3 research/backtest_harness.py

Output: greppable summary (composite_score, sharpe_ratio, etc.)
"""

import json
import math
import os
import random
import ssl
import sys
import time
import urllib.request
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

try:
    import certifi
    SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    SSL_CONTEXT = ssl.create_default_context()
    SSL_CONTEXT.check_hostname = False
    SSL_CONTEXT.verify_mode = ssl.CERT_NONE

# Import the strategy module (this is what the agent modifies)
sys.path.insert(0, str(Path(__file__).parent))
import strategy_experiment as strategy

# ═══════════════════════════════════════════════════════════════════════════════
# FIXED CONSTANTS — DO NOT CHANGE
# ═══════════════════════════════════════════════════════════════════════════════

DATA_DIR = Path(__file__).parent / "data"
CACHE_FILE = DATA_DIR / "weather_history.json"

CITIES = {
    "nyc": {"lat": 40.7128, "lon": -74.0060},
    "chicago": {"lat": 41.8781, "lon": -87.6298},
    "london": {"lat": 51.5074, "lon": -0.1278},
    "seoul": {"lat": 37.5665, "lon": 126.9780},
    "buenos_aires": {"lat": -34.6037, "lon": -58.3816},
    "atlanta": {"lat": 33.749, "lon": -84.388},
    "toronto": {"lat": 43.6532, "lon": -79.3832},
    "ankara": {"lat": 39.9334, "lon": 32.8597},
    "sao_paulo": {"lat": -23.5505, "lon": -46.6333},
    "miami": {"lat": 25.7617, "lon": -80.1918},
    "paris": {"lat": 48.8566, "lon": 2.3522},
    "dallas": {"lat": 32.7767, "lon": -96.7970},
    "seattle": {"lat": 47.6062, "lon": -122.3321},
    "wellington": {"lat": -41.2865, "lon": 174.7762},
    "los_angeles": {"lat": 34.0522, "lon": -118.2437},
    "dc": {"lat": 38.9072, "lon": -77.0369},
    "tokyo": {"lat": 35.6762, "lon": 139.6503},
    "munich": {"lat": 48.1351, "lon": 11.5820},
    "tel_aviv": {"lat": 32.0853, "lon": 34.7818},
    "lucknow": {"lat": 26.8467, "lon": 80.9462},
}

DATA_START = "2024-01-01"
DATA_END = "2025-12-31"

# Walk-forward windows: 5 overlapping periods, each ~3 month test
WALK_FORWARD_WINDOWS = [
    {"train": ("2024-01-01", "2024-06-30"), "test": ("2024-07-01", "2024-09-30")},
    {"train": ("2024-04-01", "2024-09-30"), "test": ("2024-10-01", "2024-12-31")},
    {"train": ("2024-07-01", "2024-12-31"), "test": ("2025-01-01", "2025-03-31")},
    {"train": ("2024-10-01", "2025-03-31"), "test": ("2025-04-01", "2025-06-30")},
    {"train": ("2025-01-01", "2025-06-30"), "test": ("2025-07-01", "2025-09-30")},
]

INITIAL_CAPITAL = 1000.0
SLIPPAGE_PCT = 0.005  # 0.5% slippage on fill
MAX_SIZING_CAPITAL = 5000.0  # Cap capital used for sizing at 5x initial (prevents unrealistic compounding)
BASE_SEED = 42

# Market simulation — how realistic synthetic markets are generated
BUCKET_WIDTH_C = 2.5        # 2.5°C per bucket (~4.5°F)
NUM_RANGE_BUCKETS = 6       # + "below" + "above" = 8 total

# Market maker model (less accurate than our strategy — this creates edge)
MM_FORECAST_NOISE = {0: 1.5, 1: 2.5, 2: 3.5, 3: 4.5, 4: 5.5, 5: 6.5, 6: 7.5}
MM_SIGMA = {0: 2.5, 1: 3.5, 2: 4.5, 3: 5.5, 4: 6.5, 5: 7.5, 6: 8.5}
MM_PRICING_NOISE = 0.05     # Additional bid-ask noise on market prices

# Our forecast simulation (better than market maker — reflects real data advantage)
OUR_FORECAST_NOISE = {0: 0.5, 1: 1.2, 2: 2.0, 3: 2.8, 4: 3.5, 5: 4.2, 6: 5.0}


# ═══════════════════════════════════════════════════════════════════════════════
# DATA MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════

def download_historical_data():
    """Download 2 years of daily temperatures from Open-Meteo archive API."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if CACHE_FILE.exists():
        size_kb = CACHE_FILE.stat().st_size / 1024
        print(f"Data: cached at {CACHE_FILE} ({size_kb:.0f} KB)")
        return

    print("Data: downloading historical weather from Open-Meteo archive...")
    all_data = {}

    for city_id, coords in CITIES.items():
        url = (
            f"https://archive-api.open-meteo.com/v1/archive"
            f"?latitude={coords['lat']}&longitude={coords['lon']}"
            f"&start_date={DATA_START}&end_date={DATA_END}"
            f"&daily=temperature_2m_max,temperature_2m_min"
            f"&timezone=auto"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "ArboResearch/1.0"})
        with urllib.request.urlopen(req, timeout=30, context=SSL_CONTEXT) as resp:
            data = json.loads(resp.read().decode())

        daily = data["daily"]
        city_data = {}
        for i, dt in enumerate(daily["time"]):
            high = daily["temperature_2m_max"][i]
            low = daily["temperature_2m_min"][i]
            if high is not None and low is not None:
                city_data[dt] = {"high_c": high, "low_c": low}
        all_data[city_id] = city_data
        print(f"  {city_id}: {len(city_data)} days")

    with open(CACHE_FILE, "w") as f:
        json.dump(all_data, f)
    print(f"Data: saved to {CACHE_FILE}")


def load_data():
    """Load cached historical data."""
    with open(CACHE_FILE) as f:
        return json.load(f)


def compute_monthly_normals(data):
    """Compute monthly average high temperature per city."""
    normals = {}
    for city_id, days in data.items():
        monthly = {}
        for dt_str, temps in days.items():
            month = int(dt_str.split("-")[1])
            if month not in monthly:
                monthly[month] = []
            monthly[month].append(temps["high_c"])
        normals[city_id] = {
            m: sum(v) / len(v) for m, v in monthly.items()
        }
    return normals


# ═══════════════════════════════════════════════════════════════════════════════
# MARKET SIMULATION
# ═══════════════════════════════════════════════════════════════════════════════

def _normal_cdf(x, mu=0.0, sigma=1.0):
    """Normal CDF (used by market maker model)."""
    if sigma <= 0:
        return 1.0 if x >= mu else 0.0
    return 0.5 * (1.0 + math.erf((x - mu) / (sigma * math.sqrt(2.0))))


def generate_buckets(monthly_avg_c):
    """Generate temperature buckets centered on monthly climatological average."""
    center = round(monthly_avg_c / BUCKET_WIDTH_C) * BUCKET_WIDTH_C
    half_range = (NUM_RANGE_BUCKETS / 2) * BUCKET_WIDTH_C

    buckets = []
    low_bound = center - half_range

    # "Below" bucket
    buckets.append({
        "low_c": None, "high_c": low_bound,
        "type": "below", "label": f"Below {low_bound:.1f}C",
    })

    # Range buckets
    for i in range(NUM_RANGE_BUCKETS):
        bl = low_bound + i * BUCKET_WIDTH_C
        bh = bl + BUCKET_WIDTH_C
        buckets.append({
            "low_c": bl, "high_c": bh,
            "type": "range", "label": f"{bl:.1f}-{bh:.1f}C",
        })

    # "Above" bucket
    high_bound = center + half_range
    buckets.append({
        "low_c": high_bound, "high_c": None,
        "type": "above", "label": f"Above {high_bound:.1f}C",
    })

    return buckets


def bucket_contains(bucket, actual_temp_c):
    """Check if actual temperature falls in this bucket."""
    if bucket["type"] == "below":
        return actual_temp_c < bucket["high_c"]
    elif bucket["type"] == "above":
        return actual_temp_c >= bucket["low_c"]
    else:
        return bucket["low_c"] <= actual_temp_c < bucket["high_c"]


def generate_market(actual_temp_c, bucket, days_out, rng):
    """Generate synthetic market price, volume, and liquidity for a bucket."""
    # Market maker forecast (noisier than ours)
    mm_noise_std = MM_FORECAST_NOISE.get(days_out, 7.5)
    mm_forecast = actual_temp_c + rng.gauss(0, mm_noise_std)

    # Market maker probability estimate
    mm_sigma = MM_SIGMA.get(days_out, 8.5)
    if bucket["type"] == "below":
        mm_prob = _normal_cdf(bucket["high_c"], mm_forecast, mm_sigma)
    elif bucket["type"] == "above":
        mm_prob = 1.0 - _normal_cdf(bucket["low_c"], mm_forecast, mm_sigma)
    else:
        mm_prob = (_normal_cdf(bucket["high_c"], mm_forecast, mm_sigma)
                   - _normal_cdf(bucket["low_c"], mm_forecast, mm_sigma))

    # Market price = MM estimate + pricing noise
    price = mm_prob + rng.gauss(0, MM_PRICING_NOISE)
    price = max(0.03, min(0.97, price))

    # Volume and liquidity (correlated with probability / activity)
    base_vol = 3000 + abs(price - 0.5) * 40000
    volume = max(500, base_vol * rng.uniform(0.3, 2.5))
    liquidity = volume * rng.uniform(0.2, 0.7)

    return price, volume, liquidity


def simulate_forecast(actual_temp_c, days_out, rng, city=None):
    """Simulate our weather forecast (better than market maker's).

    Includes per-city bias to match real-world forecast behavior:
    positive bias cities → forecast reads systematically LOW vs METAR actual.
    The strategy's bias correction then compensates for this.
    """
    noise_std = OUR_FORECAST_NOISE.get(days_out, 5.0)
    # Real forecasts have systematic bias vs METAR resolution data.
    # Negative of CITY_BIAS: if bias=+1.53 (forecast reads low), subtract 1.53 from actual.
    forecast_bias = 0.0
    if city:
        forecast_bias = -strategy.CITY_BIAS.get(city, 0.0)
    return actual_temp_c + forecast_bias + rng.gauss(0, noise_std)


# ═══════════════════════════════════════════════════════════════════════════════
# BACKTEST ENGINE
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


def run_single_backtest(data, normals, test_start, test_end, seed):
    """Run backtest over a single date range. Returns (trades, final_capital)."""
    forecast_rng = random.Random(seed)
    market_rng = random.Random(seed + 10000)

    capital = INITIAL_CAPITAL
    trades = []
    daily_deployed = 0.0

    # Generate date range
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

        # Collect all trade candidates for this day
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
                    forecast = simulate_forecast(actual_high, days_out, forecast_rng, city=city_id)
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
                            "days_out": days_out,
                            "market_price": market_price,
                            "forecast_prob": prob,
                            "edge": edge,
                            "actual_high": actual_high,
                        }

                if best_candidate is not None:
                    candidates.append(best_candidate)

        # Sort by edge descending — trade best opportunities first
        candidates.sort(key=lambda c: -c["edge"])

        # Deduplicate: max one trade per (city, bucket)
        seen = set()
        for cand in candidates:
            key = (cand["city"], cand["bucket_idx"])
            if key in seen:
                continue
            seen.add(key)

            # Check capital availability (cap sizing capital to prevent unrealistic compounding)
            sizing_capital = min(capital, MAX_SIZING_CAPITAL)
            max_exposure = sizing_capital * getattr(strategy, "MAX_TOTAL_EXPOSURE_PCT", 0.80)
            available = max(0, max_exposure - daily_deployed)
            if available < 1.0:
                break  # No more capital for today

            size = strategy.position_size(
                cand["edge"], cand["market_price"], available, sizing_capital,
                city=cand["city"],
            )
            if size <= 0 or size > available:
                continue

            # Execute trade
            fill_price = min(0.99, cand["market_price"] * (1.0 + SLIPPAGE_PCT))
            won = bucket_contains(cand["bucket"], cand["actual_high"])

            if won:
                pnl = size * (1.0 / fill_price - 1.0)
            else:
                pnl = -size

            capital += pnl
            daily_deployed += size

            trades.append(Trade(
                city=cand["city"],
                date=dt_str,
                bucket_label=cand["bucket"]["label"],
                days_out=cand["days_out"],
                market_price=cand["market_price"],
                forecast_prob=cand["forecast_prob"],
                edge=cand["edge"],
                size=size,
                fill_price=fill_price,
                won=won,
                pnl=pnl,
                capital_after=capital,
            ))

            # Blow-up protection
            if capital <= 0:
                return trades, 0.0

    return trades, capital


# ═══════════════════════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_metrics(trades, initial_capital):
    """Calculate performance metrics from trade list.

    Uses per-trade percentage returns (pnl/size) for Sharpe calculation,
    which is independent of capital compounding and comparable across experiments.
    """
    if not trades:
        return {
            "composite_score": 0.0,
            "sharpe_ratio": 0.0,
            "total_pnl": 0.0,
            "total_pnl_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "win_rate_pct": 0.0,
            "num_trades": 0,
            "profit_factor": 0.0,
            "avg_return_pct": 0.0,
            "avg_win_pct": 0.0,
            "avg_loss_pct": 0.0,
        }

    # Per-trade percentage returns (independent of capital level)
    returns = [t.pnl / t.size if t.size > 0 else 0 for t in trades]
    pnls = [t.pnl for t in trades]
    n = len(returns)
    total_pnl = sum(pnls)

    win_returns = [r for r in returns if r > 0]
    loss_returns = [r for r in returns if r <= 0]
    win_rate = len(win_returns) / n * 100

    # Sharpe ratio on percentage returns (annualized)
    mean_ret = sum(returns) / n
    if n > 1:
        variance = sum((r - mean_ret) ** 2 for r in returns) / (n - 1)
        std_ret = math.sqrt(variance) if variance > 0 else 1e-6
        sharpe = (mean_ret / std_ret) * math.sqrt(252)
    else:
        sharpe = 0.0

    # Max drawdown (on actual equity curve)
    peak = initial_capital
    max_dd = 0.0
    equity = initial_capital
    for t in trades:
        equity += t.pnl
        peak = max(peak, equity)
        dd = (peak - equity) / peak * 100 if peak > 0 else 0
        max_dd = max(max_dd, dd)

    # Profit factor (on absolute PnL)
    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p <= 0)) or 1e-6
    profit_factor = gross_profit / gross_loss

    # Composite score: Sharpe * sqrt(trades/100) * (1 - drawdown/50)
    if n >= 10:
        trade_factor = math.sqrt(n / 100)
        dd_factor = max(0, 1.0 - max_dd / 50.0)
        composite = sharpe * trade_factor * dd_factor
    else:
        composite = 0.0

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
    }


# ═══════════════════════════════════════════════════════════════════════════════
# WALK-FORWARD VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

def walk_forward_evaluate(data, normals):
    """Run walk-forward validation across all windows.

    The composite_score is computed from per-window AVERAGES, not by
    merging all trades (which would be dominated by compounding effects).
    """
    window_metrics = []

    for i, window in enumerate(WALK_FORWARD_WINDOWS):
        test_start, test_end = window["test"]
        seed = BASE_SEED + i * 7919  # Different prime-offset seed per window

        trades, final_capital = run_single_backtest(
            data, normals, test_start, test_end, seed
        )

        metrics = calculate_metrics(trades, INITIAL_CAPITAL)
        metrics["window"] = i + 1
        metrics["test_period"] = f"{test_start} -> {test_end}"
        metrics["final_capital"] = round(final_capital, 2)
        window_metrics.append(metrics)

    n_windows = len(window_metrics)

    # Per-window averages (robust to compounding effects)
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

    # Composite score from per-window averages
    # Sharpe * sqrt(total_trades/100) * (1 - worst_drawdown/50) * consistency
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
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    t_start = time.time()

    # 1. Ensure data is available
    download_historical_data()
    data = load_data()
    normals = compute_monthly_normals(data)

    print(f"Data: {len(data)} cities, "
          f"{sum(len(v) for v in data.values())} total day-records")

    # 2. Run walk-forward evaluation
    results = walk_forward_evaluate(data, normals)

    t_end = time.time()

    # 3. Per-window results
    print("\n" + "=" * 78)
    print("WALK-FORWARD RESULTS")
    print("=" * 78)

    for wm in results["windows"]:
        status = "+" if wm["total_pnl"] > 0 else "-"
        print(f"  W{wm['window']} {wm['test_period']}  |  "
              f"Sharpe {wm['sharpe_ratio']:>7.3f}  |  "
              f"PnL {wm['total_pnl_pct']:>7.1f}%  ({status})  |  "
              f"Win {wm['win_rate_pct']:>5.1f}%  |  "
              f"Trades {wm['num_trades']:>4d}  |  "
              f"MaxDD {wm['max_drawdown_pct']:>5.2f}%")

    # 4. Summary
    print("\n" + "=" * 78)
    print("AGGREGATE (per-window averages)")
    print("=" * 78)
    print(f"  Profitable windows: {results['profitable_windows']}/{results['total_windows']}")
    print(f"  Avg Sharpe: {results['avg_sharpe']:.4f} +/- {results['sharpe_std']:.4f}")
    print(f"  Avg PnL: {results['avg_pnl_pct']:.1f}%  |  "
          f"Win rate: {results['avg_win_rate']:.1f}%  |  "
          f"Worst DD: {results['max_drawdown_pct']:.1f}%")

    # 5. Greppable output (agent reads this)
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
    print(f"backtest_seconds:   {t_end - t_start:.1f}")


if __name__ == "__main__":
    main()
