"""Experiment Framework — Chronological Portfolio Simulator.

Hour-by-hour portfolio simulator for Strategy C (weather markets).
Replaces the old event-by-event evaluator from sweep_comprehensive.py.

Key features:
- Hourly time-step across entire period
- Concurrent positions with capital allocation
- Exit logic (edge-based, profit-take, re-entry)
- Capital turnover tracking (utilization, idle hours, concurrent positions)
- Compound sizing with MAX_POSITION_USD cap
- Walk-forward cross-validation

Usage:
    from experiment_framework import preload_data, simulate_portfolio, experiment_score

    forecasts = load_forecasts(events)
    sim = preload_data(db, events, forecasts)
    result = simulate_portfolio(sim, params, entry_hours=24)
    print(result.score)
"""

import bisect
import json
import math
import ssl
import sys
import time
import urllib.request
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import median as _median

sys.path.insert(0, str(Path(__file__).parent))
import strategy_experiment as strategy
from price_history_db import Bucket, Event, PriceHistoryDB

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

INITIAL_CAPITAL = 1000.0
MAX_POSITION_USD = 200.0
SLIPPAGE_PCT = 0.005
GAS_COST_USD = 0.007
KELLY_FRACTION = 0.25
MAX_POSITION_PCT = 0.05
MIN_TRADE_SIZE = 1.0

CITY_COORDS = {
    "chicago": (41.88, -87.63), "london": (51.51, -0.13),
    "seoul": (37.57, 126.98), "ankara": (39.93, 32.86),
    "sao_paulo": (-23.55, -46.63), "miami": (25.76, -80.19),
    "paris": (48.86, 2.35), "dallas": (32.78, -96.80),
    "seattle": (47.61, -122.33), "munich": (48.14, 11.58),
    "lucknow": (26.85, 80.95), "tel_aviv": (32.09, 34.78),
    "tokyo": (35.68, 139.65), "los_angeles": (34.05, -118.24),
    "nyc": (40.71, -74.01), "toronto": (43.65, -79.38),
    "buenos_aires": (-34.60, -58.38), "atlanta": (33.75, -84.39),
    "wellington": (-41.29, 174.78), "dc": (38.91, -77.04),
}

try:
    import certifi
    SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    SSL_CTX = ssl.create_default_context()
    SSL_CTX.check_hostname = False
    SSL_CTX.verify_mode = ssl.CERT_NONE


# ═══════════════════════════════════════════════════════════════════════════════
# DATACLASSES
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class OpenPosition:
    """Active position in portfolio."""
    token_id: str
    event_id: str
    city: str
    bucket: Bucket
    entry_price: float      # Raw market price at entry
    entry_fill: float       # Fill price (with slippage)
    entry_ts: int            # Hour timestamp of entry
    size: float              # USDC committed
    edge: float
    our_prob: float
    forecast_temp: float
    closes_at_ts: int        # Event close timestamp


@dataclass
class TradeRecord:
    """Completed trade (resolved or exited)."""
    token_id: str
    event_id: str
    city: str
    entry_price: float
    exit_price: float | None  # None for resolution
    entry_ts: int
    exit_ts: int
    size: float
    pnl: float
    edge: float
    won: bool | None          # None for early exit
    exit_reason: str          # "resolution", "edge_lost", "profit_take", "prob_floor"
    counterfactual_pnl: float | None = None


@dataclass
class HourlyState:
    """Portfolio snapshot at one hour."""
    ts: int
    capital: float       # cash + deployed (total committed value)
    deployed: float      # sum of position sizes
    n_positions: int


@dataclass
class CityResult:
    """Per-city aggregated results."""
    pnl: float
    trades: int
    wins: int
    win_rate: float


@dataclass
class ExperimentResult:
    """Complete result of one experiment evaluation."""
    # Identification
    experiment_id: str
    params: dict
    entry_hours: float
    timestamp: str

    # Core metrics
    trades: int
    wins: int
    win_rate: float
    total_pnl: float
    final_capital: float
    roi_pct: float
    max_drawdown_pct: float
    sharpe: float

    # Capital turnover
    capital_utilization: float     # % of time capital is deployed (0-100)
    avg_time_in_trade_h: float
    turnover_rate: float           # sum(trade_sizes) / avg_capital
    idle_hours: int                # hours with 0 deployed
    concurrent_positions: float    # avg simultaneous positions
    avg_pnl_per_hour: float        # $ PnL / hour in trade

    # Per-trade
    avg_pnl_per_trade: float
    avg_edge: float
    avg_price: float
    median_pnl: float

    # Breakdowns
    city_results: dict[str, dict]

    # Exit stats
    total_exits: int
    exit_saves: int
    exit_regrets: int

    # Equity curve (daily samples: [[ts, capital], ...])
    equity_curve: list[list] | None

    # OOS (filled by walk-forward)
    oos_pnl: float | None = None
    oos_trades: int | None = None
    oos_win_rate: float | None = None

    # Composite score
    score: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# SIMULATION DATA
# ═══════════════════════════════════════════════════════════════════════════════


class SimulationData:
    """Pre-loaded data with indexes for fast portfolio simulation."""

    def __init__(self, events, buckets_by_event, prices, forecasts):
        self.events = events
        self.buckets_by_event = buckets_by_event     # event_id -> [Bucket]
        self._prices = prices                        # token_id -> sorted [(ts, price)]
        self.forecasts = forecasts                   # city -> {date_str -> temp_c}

        # Parse close timestamps + build lookups
        self._close_ts: dict[str, int] = {}
        self._events_by_id: dict[str, Event] = {}
        for ev in events:
            self._events_by_id[ev.event_id] = ev
            if ev.closed_time:
                try:
                    ct = ev.closed_time.replace("Z", "+00:00")
                    self._close_ts[ev.event_id] = int(
                        datetime.fromisoformat(ct).timestamp()
                    )
                except (ValueError, AttributeError):
                    pass

    def get_price(self, token_id: str, hour_ts: int) -> float | None:
        """Get most recent price at or before hour_ts (bisect O(log n))."""
        pts = self._prices.get(token_id)
        if not pts:
            return None
        idx = bisect.bisect_right(pts, (hour_ts, float("inf"))) - 1
        if idx < 0:
            return None
        # Don't use prices more than 7 days stale
        if hour_ts - pts[idx][0] > 7 * 86400:
            return None
        return pts[idx][1]

    def get_close_ts(self, event_id: str) -> int | None:
        return self._close_ts.get(event_id)

    def build_entry_index(self, entry_hours: float) -> dict[int, list[Event]]:
        """Map: entry_hour_ts -> events to consider entering that hour."""
        index: dict[int, list[Event]] = defaultdict(list)
        offset = int(entry_hours * 3600)
        for ev in self.events:
            close_ts = self._close_ts.get(ev.event_id)
            if close_ts is None or not ev.city:
                continue
            entry_ts = close_ts - offset
            entry_hour = (entry_ts // 3600) * 3600
            index[entry_hour].append(ev)
        return dict(index)

    def build_close_index(self) -> dict[int, list[Event]]:
        """Map: close_hour_ts -> events resolving that hour."""
        index: dict[int, list[Event]] = defaultdict(list)
        for ev in self.events:
            close_ts = self._close_ts.get(ev.event_id)
            if close_ts is None:
                continue
            close_hour = (close_ts // 3600) * 3600
            index[close_hour].append(ev)
        return dict(index)

    def get_period(self, entry_hours: float = 48) -> tuple[int, int]:
        """Get simulation period (earliest entry, latest close)."""
        all_close_ts = list(self._close_ts.values())
        if not all_close_ts:
            return (0, 0)
        offset = int(entry_hours * 3600)
        start = min(all_close_ts) - offset
        end = max(all_close_ts)
        return ((start // 3600) * 3600, ((end // 3600) + 1) * 3600)

    def filter_events(
        self, min_date: str | None = None, max_date: str | None = None
    ) -> "SimulationData":
        """Return a new SimulationData with events filtered by target_date."""
        filtered = [
            ev for ev in self.events
            if ev.target_date
            and (min_date is None or ev.target_date >= min_date)
            and (max_date is None or ev.target_date <= max_date)
        ]
        # Reuse same prices/forecasts (shared, read-only)
        return SimulationData(
            filtered, self.buckets_by_event, self._prices, self.forecasts
        )


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════


def load_forecasts(
    events: list[Event], city_coords: dict | None = None
) -> dict[str, dict[str, float]]:
    """Fetch Open-Meteo archive forecasts for all events.

    Returns: city -> {date_str -> temp_high_c}
    """
    if city_coords is None:
        city_coords = CITY_COORDS

    cache: dict[str, dict[str, float]] = {}
    needed: dict[str, list[str]] = defaultdict(list)
    for ev in events:
        if ev.city and ev.city in city_coords and ev.target_date:
            needed[ev.city].append(ev.target_date)

    for city, dates in needed.items():
        lat, lon = city_coords[city]
        dates_sorted = sorted(set(dates))
        if not dates_sorted:
            continue

        url = (
            f"https://archive-api.open-meteo.com/v1/archive"
            f"?latitude={lat}&longitude={lon}"
            f"&start_date={dates_sorted[0]}&end_date={dates_sorted[-1]}"
            f"&daily=temperature_2m_max&timezone=auto"
        )
        req = urllib.request.Request(
            url, headers={"User-Agent": "ArboResearch/1.0"}
        )
        try:
            with urllib.request.urlopen(req, timeout=30, context=SSL_CTX) as resp:
                data = json.loads(resp.read().decode())
            daily = data.get("daily", {})
            city_data: dict[str, float] = {}
            for i, dt_str in enumerate(daily.get("time", [])):
                val = daily["temperature_2m_max"][i]
                if val is not None:
                    city_data[dt_str] = val
            cache[city] = city_data
        except Exception:
            pass
        time.sleep(0.05)

    total = sum(len(v) for v in cache.values())
    print(f"Forecasts cached: {total} city-dates", flush=True)
    return cache


def preload_data(
    db: PriceHistoryDB, events: list[Event], forecasts: dict
) -> SimulationData:
    """Load all data into memory for fast simulation.

    Bulk-loads prices (2 SQL queries instead of 12,845 per-token queries).
    CLOB prices take priority over Goldsky for overlapping hours.
    """
    # Load buckets
    buckets_by_event: dict[str, list[Bucket]] = {}
    token_ids_needed: set[str] = set()
    for ev in events:
        buckets = db.get_buckets(ev.event_id)
        buckets_by_event[ev.event_id] = buckets
        for b in buckets:
            token_ids_needed.add(b.token_id)

    # Bulk load prices — CLOB first (priority), then Goldsky for gaps
    prices_dict: dict[str, dict[int, float]] = defaultdict(dict)

    for row in db._conn.execute(
        "SELECT token_id, ts, price FROM prices"
    ).fetchall():
        tid = row["token_id"]
        if tid in token_ids_needed:
            prices_dict[tid][row["ts"]] = row["price"]

    if db._has_goldsky():
        for row in db._conn.execute(
            "SELECT token_id, ts, price FROM goldsky_trades"
        ).fetchall():
            tid = row["token_id"]
            if tid in token_ids_needed:
                ts = row["ts"]
                if ts not in prices_dict[tid]:  # CLOB takes priority
                    prices_dict[tid][ts] = row["price"]

    # Convert to sorted lists for bisect lookups
    sorted_prices: dict[str, list[tuple[int, float]]] = {}
    for tid, ts_dict in prices_dict.items():
        sorted_prices[tid] = sorted(ts_dict.items())

    n_points = sum(len(v) for v in sorted_prices.values())
    print(
        f"Loaded: {len(sorted_prices)} tokens, {n_points:,} price points",
        flush=True,
    )

    return SimulationData(events, buckets_by_event, sorted_prices, forecasts)


# ═══════════════════════════════════════════════════════════════════════════════
# PROBABILITY, QUALITY GATE, SIZING
# ═══════════════════════════════════════════════════════════════════════════════


def compute_prob(
    forecast_temp: float,
    bucket: Bucket,
    days_out: int,
    city: str,
    params: dict,
) -> float:
    """Compute our probability estimate for a bucket winning."""
    sigma = strategy._get_sigma(days_out, city)

    # Bias correction
    bias = strategy.CITY_BIAS.get(city, 0.0)
    adjusted_temp = forecast_temp + bias

    cdf = lambda x: strategy._normal_cdf(x, adjusted_temp, sigma)

    if bucket.low_c is None and bucket.high_c is not None:
        raw = cdf(bucket.high_c)
    elif bucket.high_c is None and bucket.low_c is not None:
        raw = 1.0 - cdf(bucket.low_c)
    elif bucket.low_c is not None and bucket.high_c is not None:
        raw = cdf(bucket.high_c) - cdf(bucket.low_c)
    else:
        return 0.0

    # Per-city overrides for shrinkage and sharpening
    city_ov = params.get("city_overrides", {}).get(city, {})

    # Bayesian shrinkage
    shrinkage = city_ov.get("shrinkage", params.get("shrinkage", 0.03))
    raw = raw * (1.0 - shrinkage) + 0.125 * shrinkage

    # Probability sharpening
    sharpening = city_ov.get("prob_sharpening", params.get("prob_sharpening", 1.05))
    if sharpening != 1.0 and raw > 0:
        raw = raw ** sharpening

    return raw


def quality_gate(
    edge: float,
    our_prob: float,
    price: float,
    volume: float,
    city: str,
    params: dict,
) -> bool:
    """Check if a signal passes quality gate filters."""
    min_edge = params.get("min_edge", 0.08)
    max_edge = params.get("max_edge", 0.90)
    min_price = params.get("min_price", 0.30)
    max_price = params.get("max_price", 0.43)
    min_prob = params.get("min_prob", 0.10)
    min_volume = params.get("min_volume", 1000)

    # Per-city overrides
    city_ov = params.get("city_overrides", {}).get(city, {})
    min_edge = city_ov.get("min_edge", min_edge)
    max_price = city_ov.get("max_price", max_price)
    min_price = city_ov.get("min_price", min_price)
    min_volume = city_ov.get("min_volume", min_volume)

    if edge < min_edge or edge > max_edge:
        return False
    if price < min_price or price > max_price:
        return False
    if our_prob < min_prob:
        return False
    if volume < min_volume:
        return False
    return True


def compute_size(
    edge: float,
    price: float,
    available: float,
    total_capital: float,
    params: dict,
    city: str = "",
) -> float:
    """Compute position size using Kelly criterion."""
    if price <= 0 or price >= 1 or edge <= 0:
        return 0.0

    prob = price + edge
    if prob <= 0 or prob >= 1:
        return 0.0

    odds = (1.0 / price) - 1.0
    if odds <= 0:
        return 0.0

    kelly_raw = (prob * odds - (1 - prob)) / odds
    if kelly_raw <= 0:
        return 0.0

    # Per-city kelly_raw_cap override
    city_ov = params.get("city_overrides", {}).get(city, {})
    kelly_raw_cap = city_ov.get("kelly_raw_cap", params.get("kelly_raw_cap", 0.40))
    kelly_raw = min(kelly_raw, kelly_raw_cap)

    kelly_adj = kelly_raw * KELLY_FRACTION
    size = available * kelly_adj
    size = min(size, total_capital * MAX_POSITION_PCT)
    size = min(size, MAX_POSITION_USD)

    if size < MIN_TRADE_SIZE:
        return 0.0

    return round(size, 2)


# ═══════════════════════════════════════════════════════════════════════════════
# EXIT LOGIC
# ═══════════════════════════════════════════════════════════════════════════════


def check_exit(
    pos: OpenPosition,
    current_price: float,
    hour_ts: int,
    params: dict,
) -> str | None:
    """Check if position should be exited.

    Returns exit reason string ("edge_lost", "profit_take", "prob_floor")
    or None to hold.
    """
    if not params.get("exit_enabled", False):
        return None

    # Updated probability (sigma shrinks closer to resolution)
    hours_to_close = max(0, (pos.closes_at_ts - hour_ts) / 3600)
    days_out = max(0, int(hours_to_close / 24))

    updated_prob = compute_prob(
        pos.forecast_temp, pos.bucket, days_out, pos.city, params
    )
    updated_edge = updated_prob - current_price

    # Edge-based exit
    min_hold_edge = params.get("min_hold_edge", 0.15)
    if updated_edge < min_hold_edge:
        return "edge_lost"

    # Probability floor
    prob_floor = params.get("prob_exit_floor", 0.0)
    if prob_floor > 0 and updated_prob < prob_floor:
        return "prob_floor"

    # Profit-take
    if params.get("profit_take_enabled", False):
        threshold = params.get("profit_take_threshold", 1.0)
        min_hours = params.get("profit_take_min_hours", 6)
        hours_held = (hour_ts - pos.entry_ts) / 3600

        if hours_held >= min_hours and pos.entry_price > 0:
            unrealized = (current_price - pos.entry_price) / pos.entry_price
            if unrealized > threshold:
                return "profit_take"

    return None


def calc_resolution_pnl(pos: OpenPosition) -> float:
    """P&L when position resolves at market close."""
    if pos.bucket.won:
        return pos.size * (1.0 / pos.entry_fill - 1.0) - GAS_COST_USD
    else:
        return -pos.size - GAS_COST_USD


def calc_exit_pnl(pos: OpenPosition, exit_price: float) -> float:
    """P&L when selling position early at exit_price."""
    exit_fill = exit_price * (1 - SLIPPAGE_PCT)
    tokens = pos.size / pos.entry_fill
    proceeds = tokens * exit_fill
    return proceeds - pos.size - GAS_COST_USD


# ═══════════════════════════════════════════════════════════════════════════════
# CORE SIMULATION
# ═══════════════════════════════════════════════════════════════════════════════


def simulate_portfolio(
    sim_data: SimulationData,
    params: dict,
    entry_hours: float = 24,
    period_start_ts: int | None = None,
    period_end_ts: int | None = None,
    experiment_id: str = "",
    record_equity: bool = False,
) -> ExperimentResult:
    """Run chronological portfolio simulation.

    Iterates hour-by-hour, managing entries, exits, and resolutions.
    Tracks capital deployment and turnover metrics.
    """
    # Build indexes
    entry_index = sim_data.build_entry_index(entry_hours)
    close_index = sim_data.build_close_index()

    if period_start_ts is None or period_end_ts is None:
        p_start, p_end = sim_data.get_period(entry_hours)
        period_start_ts = period_start_ts or p_start
        period_end_ts = period_end_ts or p_end

    # Excluded cities
    excluded = params.get("excluded_cities", set())
    if isinstance(excluded, list):
        excluded = set(excluded)

    # State
    cash = INITIAL_CAPITAL
    deployed: dict[str, OpenPosition] = {}
    all_trades: list[TradeRecord] = []
    hourly_log: list[HourlyState] = []
    entered_events: set[str] = set()  # Avoid double-entry per event+bucket

    # Re-entry cooldown: city -> earliest reentry hour_ts
    cooldowns: dict[str, int] = {}
    cooldown_h = params.get("reentry_cooldown_h", 1)

    # Pre-compute interesting hours (entry + close hours that exist)
    # This avoids iterating 10K empty hours
    interesting_hours: set[int] = set()
    interesting_hours.update(entry_index.keys())
    interesting_hours.update(close_index.keys())

    # Main loop
    for hour_ts in range(period_start_ts, period_end_ts, 3600):

        has_activity = (
            hour_ts in interesting_hours or len(deployed) > 0
        )

        if not has_activity:
            # Fast path: nothing to do this hour
            total_deployed = 0.0
            total_value = cash
            hourly_log.append(
                HourlyState(hour_ts, total_value, 0.0, 0)
            )
            continue

        # ── 1. RESOLUTIONS (events closing this hour) ──
        for ev in close_index.get(hour_ts, []):
            buckets = sim_data.buckets_by_event.get(ev.event_id, [])
            for bucket in buckets:
                pos = deployed.get(bucket.token_id)
                if pos is None:
                    continue

                pnl = calc_resolution_pnl(pos)
                cash += pos.size + pnl  # Return committed capital + P&L

                all_trades.append(TradeRecord(
                    token_id=pos.token_id,
                    event_id=pos.event_id,
                    city=pos.city,
                    entry_price=pos.entry_price,
                    exit_price=None,
                    entry_ts=pos.entry_ts,
                    exit_ts=hour_ts,
                    size=pos.size,
                    pnl=pnl,
                    edge=pos.edge,
                    won=bucket.won,
                    exit_reason="resolution",
                ))
                del deployed[bucket.token_id]

        # ── 2. EXIT CHECKS on open positions ──
        if params.get("exit_enabled", False):
            for token_id in list(deployed.keys()):
                pos = deployed[token_id]
                current_price = sim_data.get_price(token_id, hour_ts)
                if current_price is None:
                    continue

                exit_reason = check_exit(pos, current_price, hour_ts, params)
                if exit_reason is None:
                    continue

                exit_pnl = calc_exit_pnl(pos, current_price)
                cash += pos.size + exit_pnl

                # Counterfactual: what would have happened if we held?
                cf_pnl = calc_resolution_pnl(pos)

                all_trades.append(TradeRecord(
                    token_id=pos.token_id,
                    event_id=pos.event_id,
                    city=pos.city,
                    entry_price=pos.entry_price,
                    exit_price=current_price,
                    entry_ts=pos.entry_ts,
                    exit_ts=hour_ts,
                    size=pos.size,
                    pnl=exit_pnl,
                    edge=pos.edge,
                    won=None,
                    exit_reason=exit_reason,
                    counterfactual_pnl=cf_pnl,
                ))
                del deployed[token_id]

                # Set cooldown for re-entry in same city
                cooldowns[pos.city] = hour_ts + cooldown_h * 3600

        # ── 3. NEW ENTRIES ──
        total_deployed = sum(p.size for p in deployed.values())
        available = cash - total_deployed  # Free cash for new trades
        total_capital = cash  # For MAX_POSITION_PCT calc

        for ev in entry_index.get(hour_ts, []):
            if available < MIN_TRADE_SIZE:
                break
            if not ev.city or ev.city in excluded:
                continue
            if ev.city not in CITY_COORDS:
                continue

            # Re-entry cooldown check
            if ev.city in cooldowns and hour_ts < cooldowns[ev.city]:
                continue

            forecast_temp = sim_data.forecasts.get(ev.city, {}).get(
                ev.target_date
            )
            if forecast_temp is None:
                continue

            close_ts = sim_data.get_close_ts(ev.event_id)
            if close_ts is None:
                continue

            days_out = max(0, int(entry_hours / 24))
            buckets = sim_data.buckets_by_event.get(ev.event_id, [])

            for bucket in buckets:
                if bucket.token_id in deployed:
                    continue

                # Check if we already entered this specific bucket
                entry_key = f"{ev.event_id}:{bucket.token_id}"
                if entry_key in entered_events:
                    continue

                price = sim_data.get_price(bucket.token_id, hour_ts)
                if price is None or price <= 0.001:
                    continue

                our_prob = compute_prob(
                    forecast_temp, bucket, days_out, ev.city, params
                )
                edge = our_prob - price

                if not quality_gate(
                    edge, our_prob, price, bucket.volume, ev.city, params
                ):
                    continue

                size = compute_size(
                    edge, price, available, total_capital, params,
                    city=ev.city,
                )
                if size < MIN_TRADE_SIZE or available < size:
                    continue

                fill_price = price * (1 + SLIPPAGE_PCT)

                deployed[bucket.token_id] = OpenPosition(
                    token_id=bucket.token_id,
                    event_id=ev.event_id,
                    city=ev.city,
                    bucket=bucket,
                    entry_price=price,
                    entry_fill=fill_price,
                    entry_ts=hour_ts,
                    size=size,
                    edge=edge,
                    our_prob=our_prob,
                    forecast_temp=forecast_temp,
                    closes_at_ts=close_ts,
                )
                entered_events.add(entry_key)

                cash -= size
                available -= size
                total_deployed += size

        # ── 4. LOG STATE ──
        total_deployed = sum(p.size for p in deployed.values())
        total_value = cash + total_deployed

        # Track peak and drawdown
        hourly_log.append(
            HourlyState(hour_ts, total_value, total_deployed, len(deployed))
        )

    # ── CLOSE ANY REMAINING POSITIONS (shouldn't happen if close_index is right) ──
    for token_id, pos in list(deployed.items()):
        pnl = calc_resolution_pnl(pos)
        cash += pos.size + pnl
        all_trades.append(TradeRecord(
            token_id=pos.token_id,
            event_id=pos.event_id,
            city=pos.city,
            entry_price=pos.entry_price,
            exit_price=None,
            entry_ts=pos.entry_ts,
            exit_ts=period_end_ts,
            size=pos.size,
            pnl=pnl,
            edge=pos.edge,
            won=pos.bucket.won,
            exit_reason="period_end",
        ))
    deployed.clear()

    return _build_result(
        experiment_id, params, entry_hours,
        all_trades, hourly_log,
        record_equity=record_equity,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# METRICS & SCORING
# ═══════════════════════════════════════════════════════════════════════════════


def _build_result(
    experiment_id: str,
    params: dict,
    entry_hours: float,
    trades: list[TradeRecord],
    hourly_log: list[HourlyState],
    record_equity: bool = False,
) -> ExperimentResult:
    """Compute all metrics from trades and hourly log."""
    n_trades = len(trades)

    if n_trades == 0:
        return ExperimentResult(
            experiment_id=experiment_id, params=params,
            entry_hours=entry_hours,
            timestamp=datetime.now(timezone.utc).isoformat(),
            trades=0, wins=0, win_rate=0, total_pnl=0,
            final_capital=INITIAL_CAPITAL, roi_pct=0,
            max_drawdown_pct=0, sharpe=0,
            capital_utilization=0, avg_time_in_trade_h=0,
            turnover_rate=0, idle_hours=len(hourly_log),
            concurrent_positions=0, avg_pnl_per_hour=0,
            avg_pnl_per_trade=0, avg_edge=0, avg_price=0,
            median_pnl=0, city_results={},
            total_exits=0, exit_saves=0, exit_regrets=0,
            equity_curve=None,
        )

    # ── Core metrics ──
    # Win = profitable trade (resolved won OR exited with profit)
    wins = sum(1 for t in trades if t.pnl > 0)
    win_rate = wins / n_trades * 100

    total_pnl = sum(t.pnl for t in trades)
    final_capital = INITIAL_CAPITAL + total_pnl
    roi_pct = total_pnl / INITIAL_CAPITAL * 100

    # Drawdown from hourly log
    peak = INITIAL_CAPITAL
    max_dd = 0.0
    for state in hourly_log:
        peak = max(peak, state.capital)
        if peak > 0:
            dd = (peak - state.capital) / peak
            max_dd = max(max_dd, dd)

    # Sharpe (daily P&L, annualized)
    daily_pnl: dict[int, float] = defaultdict(float)
    for t in trades:
        day = t.exit_ts // 86400
        daily_pnl[day] += t.pnl
    daily_vals = list(daily_pnl.values())

    if len(daily_vals) >= 2:
        mean_d = sum(daily_vals) / len(daily_vals)
        var_d = sum((d - mean_d) ** 2 for d in daily_vals) / (len(daily_vals) - 1)
        std_d = math.sqrt(var_d) if var_d > 0 else 0.001
        sharpe = (mean_d / std_d) * math.sqrt(365)
    else:
        sharpe = 0.0

    # ── Capital turnover metrics ──
    if hourly_log:
        # capital_utilization = % of hours with at least 1 open position
        hours_with_pos = sum(1 for s in hourly_log if s.n_positions > 0)
        capital_util = hours_with_pos / len(hourly_log) * 100
        idle_hrs = sum(1 for s in hourly_log if s.deployed == 0)
        concurrent = (
            sum(s.n_positions for s in hourly_log) / len(hourly_log)
        )
    else:
        capital_util = 0.0
        idle_hrs = 0
        concurrent = 0.0

    total_time_in_trade = sum(
        (t.exit_ts - t.entry_ts) / 3600 for t in trades
    )
    avg_time = total_time_in_trade / n_trades if n_trades else 0

    total_sizes = sum(t.size for t in trades)
    avg_capital = (
        sum(s.capital for s in hourly_log) / len(hourly_log)
        if hourly_log else INITIAL_CAPITAL
    )
    turnover = total_sizes / avg_capital if avg_capital > 0 else 0

    pnl_per_hour = (
        total_pnl / total_time_in_trade if total_time_in_trade > 0 else 0
    )

    # ── Per-trade stats ──
    pnls = [t.pnl for t in trades]
    avg_pnl_trade = total_pnl / n_trades
    avg_edge = sum(t.edge for t in trades) / n_trades
    avg_price = sum(t.entry_price for t in trades) / n_trades
    med_pnl = _median(pnls) if pnls else 0

    # ── Per-city breakdown ──
    city_pnl: dict[str, float] = defaultdict(float)
    city_trades: dict[str, int] = defaultdict(int)
    city_wins: dict[str, int] = defaultdict(int)
    for t in trades:
        city_pnl[t.city] += t.pnl
        city_trades[t.city] += 1
        if t.pnl > 0:
            city_wins[t.city] += 1

    city_results = {}
    for city in city_pnl:
        ct = city_trades[city]
        cw = city_wins[city]
        city_results[city] = {
            "pnl": round(city_pnl[city], 2),
            "trades": ct,
            "wins": cw,
            "win_rate": round(cw / ct * 100, 1) if ct > 0 else 0,
        }

    # ── Exit stats ──
    exits = [t for t in trades if t.exit_reason not in ("resolution", "period_end")]
    exit_saves = sum(
        1 for t in exits
        if t.counterfactual_pnl is not None and t.pnl > t.counterfactual_pnl
    )
    exit_regrets = len(exits) - exit_saves

    # ── Equity curve (daily samples) ──
    equity_curve = None
    if record_equity and hourly_log:
        daily_equity: dict[int, float] = {}
        for state in hourly_log:
            day = state.ts // 86400
            daily_equity[day] = state.capital
        equity_curve = [
            [day * 86400, round(cap, 2)]
            for day, cap in sorted(daily_equity.items())
        ]

    result = ExperimentResult(
        experiment_id=experiment_id,
        params=params,
        entry_hours=entry_hours,
        timestamp=datetime.now(timezone.utc).isoformat(),
        trades=n_trades,
        wins=wins,
        win_rate=round(win_rate, 1),
        total_pnl=round(total_pnl, 2),
        final_capital=round(final_capital, 2),
        roi_pct=round(roi_pct, 1),
        max_drawdown_pct=round(max_dd * 100, 2),
        sharpe=round(sharpe, 2),
        capital_utilization=round(capital_util, 1),
        avg_time_in_trade_h=round(avg_time, 1),
        turnover_rate=round(turnover, 2),
        idle_hours=idle_hrs,
        concurrent_positions=round(concurrent, 2),
        avg_pnl_per_hour=round(pnl_per_hour, 2),
        avg_pnl_per_trade=round(avg_pnl_trade, 2),
        avg_edge=round(avg_edge, 4),
        avg_price=round(avg_price, 4),
        median_pnl=round(med_pnl, 2),
        city_results=city_results,
        total_exits=len(exits),
        exit_saves=exit_saves,
        exit_regrets=exit_regrets,
        equity_curve=equity_curve,
    )
    result.score = experiment_score(result)
    return result


def experiment_score(result: ExperimentResult) -> float:
    """Composite score for ranking experiments.

    Profitability (50%) + Capital Turnover (30%) + Validation (20%).
    All components capped at 2.0 → max theoretical score = 200.

    Mandatory filters (return 0 if failed):
    - trades < 10
    - max_drawdown > 50%
    - Sharpe < 0.5
    """
    if result.trades < 10:
        return 0.0
    if result.max_drawdown_pct > 50:
        return 0.0
    if result.sharpe < 0.5:
        return 0.0

    # ── Profitability (50%) ──
    # ROI capped: 500% = max. Prevents lottery strategies from dominating.
    roi_score = min(result.roi_pct / 250, 2.0)                    # 25%
    sharpe_score = min(max(result.sharpe, 0) / 5.0, 2.0)          # 15%
    dd_score = max(0, 1.0 - result.max_drawdown_pct / 50)         # 10%

    # ── Capital Turnover (30%) ──
    # Utilization scaled: 20% = max. Rewards capital at work.
    util_score = min(result.capital_utilization / 10, 2.0)         # 15%
    pph = max(result.avg_pnl_per_hour, 0)
    pph_score = min(pph / 5.0, 2.0)                               # 15%

    # ── Validation (20%) ──
    if result.oos_pnl is not None and result.oos_pnl > 0:
        oos_score = min(result.oos_pnl / 1000, 2.0)              # 10%
    else:
        oos_score = 0
    trade_score = min(result.trades / 100, 2.0)                   # 10%

    return round(
        roi_score * 25
        + sharpe_score * 15
        + dd_score * 10
        + util_score * 15
        + pph_score * 15
        + oos_score * 10
        + trade_score * 10,
        2,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# WALK-FORWARD VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════


def walk_forward_validate(
    sim_data: SimulationData,
    params: dict,
    entry_hours: float = 24,
    n_folds: int = 3,
) -> dict:
    """Walk-forward cross-validation with expanding window.

    Fold 1: Train [Jan-Jun 2025] → Test [Jul-Sep 2025]
    Fold 2: Train [Jan-Sep 2025] → Test [Oct-Dec 2025]
    Fold 3: Train [Jan-Dec 2025] → Test [Jan 2026]
    """
    dated_events = [e for e in sim_data.events if e.target_date]
    dated_events.sort(key=lambda e: e.target_date)

    if len(dated_events) < 20:
        return {"score": 0, "oos_pnl": 0, "n_folds": 0}

    fold_size = len(dated_events) // (n_folds + 1)
    oos_results = []

    for fold in range(n_folds):
        # Test window: fold_size events after training
        test_start_idx = (fold + 1) * fold_size
        test_end_idx = (fold + 2) * fold_size
        if fold == n_folds - 1:
            test_end_idx = len(dated_events)

        test_events = dated_events[test_start_idx:test_end_idx]
        if len(test_events) < 5:
            continue

        # Get date range for test events
        test_min_date = test_events[0].target_date
        test_max_date = test_events[-1].target_date

        # Create filtered SimulationData for test period
        test_sim = sim_data.filter_events(
            min_date=test_min_date, max_date=test_max_date
        )

        result = simulate_portfolio(
            test_sim, params, entry_hours=entry_hours,
            experiment_id=f"wf_fold_{fold + 1}",
        )
        oos_results.append(result)

    if not oos_results:
        return {"score": 0, "oos_pnl": 0, "n_folds": 0}

    avg_oos_pnl = sum(r.total_pnl for r in oos_results) / len(oos_results)
    avg_oos_score = sum(r.score for r in oos_results) / len(oos_results)
    total_trades = sum(r.trades for r in oos_results)

    return {
        "score": round(avg_oos_score, 2),
        "oos_pnl": round(avg_oos_pnl, 2),
        "n_folds": len(oos_results),
        "total_trades": total_trades,
        "folds": [
            {
                "pnl": r.total_pnl,
                "trades": r.trades,
                "wr": r.win_rate,
                "score": r.score,
                "utilization": r.capital_utilization,
            }
            for r in oos_results
        ],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SERIALIZATION
# ═══════════════════════════════════════════════════════════════════════════════


def serialize_result(result: ExperimentResult) -> dict:
    """Convert ExperimentResult to JSON-serializable dict."""
    d = {
        "experiment_id": result.experiment_id,
        "params": _serialize_params(result.params),
        "entry_hours": result.entry_hours,
        "timestamp": result.timestamp,
        "score": result.score,
        "trades": result.trades,
        "wins": result.wins,
        "win_rate": result.win_rate,
        "total_pnl": result.total_pnl,
        "final_capital": result.final_capital,
        "roi_pct": result.roi_pct,
        "max_drawdown_pct": result.max_drawdown_pct,
        "sharpe": result.sharpe,
        "capital_utilization": result.capital_utilization,
        "avg_time_in_trade_h": result.avg_time_in_trade_h,
        "turnover_rate": result.turnover_rate,
        "idle_hours": result.idle_hours,
        "concurrent_positions": result.concurrent_positions,
        "avg_pnl_per_hour": result.avg_pnl_per_hour,
        "avg_pnl_per_trade": result.avg_pnl_per_trade,
        "avg_edge": result.avg_edge,
        "avg_price": result.avg_price,
        "median_pnl": result.median_pnl,
        "city_results": result.city_results,
        "total_exits": result.total_exits,
        "exit_saves": result.exit_saves,
        "exit_regrets": result.exit_regrets,
    }
    if result.equity_curve is not None:
        d["equity_curve"] = result.equity_curve
    if result.oos_pnl is not None:
        d["oos_pnl"] = result.oos_pnl
        d["oos_trades"] = result.oos_trades
        d["oos_win_rate"] = result.oos_win_rate
    return d


def _serialize_params(params: dict) -> dict:
    """Make params JSON-serializable."""
    p = dict(params)
    if isinstance(p.get("excluded_cities"), set):
        p["excluded_cities"] = sorted(list(p["excluded_cities"]))
    if "city_overrides" in p:
        p["city_overrides"] = dict(p["city_overrides"])
    return p


# ═══════════════════════════════════════════════════════════════════════════════
# QUICK TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=== Experiment Framework — Smoke Test ===\n")

    db = PriceHistoryDB()
    events = db.get_events(with_prices=True)
    print(f"Events: {len(events)}")

    forecasts = load_forecasts(events)
    sim = preload_data(db, events, forecasts)

    # Test with default production params
    params = {
        "min_edge": 0.08,
        "max_edge": 0.90,
        "max_price": 0.43,
        "min_price": 0.30,
        "min_prob": 0.10,
        "min_volume": 1000,
        "kelly_raw_cap": 0.40,
        "prob_sharpening": 1.05,
        "shrinkage": 0.03,
        "excluded_cities": set(),
        "city_overrides": {},
        # Exit params
        "exit_enabled": False,
    }

    print("\n--- No exit ---")
    r1 = simulate_portfolio(sim, params, entry_hours=24,
                            experiment_id="test_no_exit", record_equity=True)
    print(f"  Trades: {r1.trades}, Wins: {r1.wins}, WR: {r1.win_rate}%")
    print(f"  PnL: ${r1.total_pnl:,.2f}, ROI: {r1.roi_pct}%")
    print(f"  Max DD: {r1.max_drawdown_pct}%, Sharpe: {r1.sharpe}")
    print(f"  Utilization: {r1.capital_utilization}%, Idle: {r1.idle_hours}h")
    print(f"  Concurrent: {r1.concurrent_positions}, Turnover: {r1.turnover_rate}")
    print(f"  PnL/hour: ${r1.avg_pnl_per_hour}")
    print(f"  Score: {r1.score}")

    params["exit_enabled"] = True
    params["min_hold_edge"] = 0.15
    params["profit_take_enabled"] = True
    params["profit_take_threshold"] = 1.0
    params["profit_take_min_hours"] = 6
    params["reentry_enabled"] = True
    params["reentry_cooldown_h"] = 1

    print("\n--- With exit ---")
    r2 = simulate_portfolio(sim, params, entry_hours=24,
                            experiment_id="test_with_exit", record_equity=True)
    print(f"  Trades: {r2.trades}, Wins: {r2.wins}, WR: {r2.win_rate}%")
    print(f"  PnL: ${r2.total_pnl:,.2f}, ROI: {r2.roi_pct}%")
    print(f"  Max DD: {r2.max_drawdown_pct}%, Sharpe: {r2.sharpe}")
    print(f"  Utilization: {r2.capital_utilization}%, Idle: {r2.idle_hours}h")
    print(f"  Exits: {r2.total_exits} (saves: {r2.exit_saves}, regrets: {r2.exit_regrets})")
    print(f"  PnL/hour: ${r2.avg_pnl_per_hour}")
    print(f"  Score: {r2.score}")

    if r2.city_results:
        print("\n  Per-city:")
        for city in sorted(r2.city_results, key=lambda c: -r2.city_results[c]["pnl"]):
            cr = r2.city_results[city]
            print(f"    {city:<16} ${cr['pnl']:>8.2f}  ({cr['trades']} trades)")

    db.close()
    print("\n=== Done ===")
