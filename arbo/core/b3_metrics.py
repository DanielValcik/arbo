"""B3 Metrics Engine — SQL queries + rolling analytics for Watchdog.

Queries paper_trades WHERE strategy='B3', computes rolling WR, PnL, Sharpe,
regime breakdown (sigma, velocity, spread, TA), PSI feature drift, and
ECE calibration error. Returns structured dicts consumed by the Watchdog
decision engine.

All queries use asyncpg via SQLAlchemy async sessions.

See: docs/B3_WATCHDOG_SPEC.md (Section 3: Metriky)
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

from sqlalchemy import text

from arbo.utils.logger import get_logger

logger = get_logger("b3_metrics")


@dataclass
class B3MetricsSnapshot:
    """Complete metrics snapshot for Watchdog evaluation cycle.

    IMPORTANT: Primary metrics are LIVE (real money). Paper metrics are
    secondary context. Watchdog decisions should prioritize live_* fields.
    """

    timestamp: float
    total_trades: int

    # Rolling outcome metrics (last N trades)
    rolling_n: int

    # LIVE METRICS (PRIMARY — used for Watchdog decisions)
    rolling_n_live: int          # Count of live-filled resolved trades
    rolling_wr_live: float       # Win rate on live trades
    rolling_avg_pnl_live: float  # Average live PnL per trade
    rolling_total_pnl_live: float  # Sum of live PnL
    rolling_sharpe_live: float   # Live-only Sharpe
    max_consecutive_losses_live: int

    # PAPER METRICS (SECONDARY — context only, paper PnL is approximate)
    rolling_wr_paper: float
    rolling_avg_pnl: float
    rolling_sharpe: float
    max_consecutive_losses_paper: int

    # Regime breakdown (dict of bucket_name → {wr, n, pnl})
    sigma_regime: dict[str, dict[str, float]]
    velocity_regime: dict[str, dict[str, float]]
    spread_regime: dict[str, dict[str, float]]

    # TA regime (may be empty if no TA data yet)
    adx_regime: dict[str, dict[str, float]]
    rsi_regime: dict[str, dict[str, float]]

    # Feature drift (PSI scores)
    psi_scores: dict[str, float]

    # Calibration
    ece: float

    # Market microstructure
    avg_spread: float
    avg_cl_delta: float
    avg_fill_price: float
    avg_liquidity: float

    # Daily PnL (last 7 days)
    daily_pnl: list[dict[str, Any]]


# ── SQL Queries ──────────────────────────────────────────────────────

_QUERY_B3_TRADES = text("""
    SELECT
        id, status, actual_pnl, placed_at, resolved_at,
        trade_details->>'direction' AS direction,
        (trade_details->>'velocity_paper')::float AS velocity,
        (trade_details->>'sigma_norm')::float AS sigma_norm,
        (trade_details->>'orderbook_spread')::float AS spread,
        (trade_details->>'combined_risk')::float AS combined_risk,
        (trade_details->>'edge')::float AS edge,
        (trade_details->>'live_fill_status') AS live_status,
        (trade_details->>'live_exit_status') AS live_exit_status,
        (trade_details->>'live_entry_price')::float AS live_fill_price,
        (trade_details->>'live_exit_price')::float AS live_exit_price,
        (trade_details->>'live_entry_shares')::float AS live_shares,
        -- Live PnL computed directly from live prices (not paper actual_pnl)
        CASE
            WHEN (trade_details->>'live_fill_status') IN ('filled', 'partial')
                AND (trade_details->>'live_exit_status') = 'resolution'
                AND (trade_details->>'live_exit_price') IS NOT NULL
                AND (trade_details->>'live_entry_shares')::float > 0
            THEN ((trade_details->>'live_exit_price')::float
                  - (trade_details->>'live_entry_price')::float)
                 * (trade_details->>'live_entry_shares')::float
            ELSE NULL
        END AS live_pnl,
        (trade_details->>'liq_available_usd')::float AS liquidity,
        (trade_details->>'btc_binance_chainlink_delta')::float AS cl_delta,
        (trade_details->>'ta_adx_5m')::float AS ta_adx,
        (trade_details->>'ta_rsi_5m')::float AS ta_rsi,
        (trade_details->>'ta_adx_regime') AS ta_adx_regime,
        (trade_details->>'ta_rsi_zone') AS ta_rsi_zone,
        (trade_details->>'ta_multi_tf_aligned') AS ta_mtf_aligned
    FROM paper_trades
    WHERE strategy = 'B3'
      AND status IN ('won', 'lost', 'sold')
      AND actual_pnl IS NOT NULL
    ORDER BY resolved_at DESC
    LIMIT :limit
""")

_QUERY_DAILY_PNL = text("""
    SELECT
        DATE(resolved_at) AS trade_date,
        COUNT(*) AS trades,
        SUM(CASE WHEN actual_pnl > 0 THEN 1 ELSE 0 END) AS wins,
        SUM(actual_pnl) AS daily_pnl,
        SUM(CASE WHEN trade_details->>'live_fill_status' = 'filled' THEN 1 ELSE 0 END) AS live_trades,
        SUM(CASE WHEN trade_details->>'live_fill_status' = 'filled'
                 AND actual_pnl > 0 THEN 1 ELSE 0 END) AS live_wins
    FROM paper_trades
    WHERE strategy = 'B3'
      AND status IN ('won', 'lost', 'sold')
      AND actual_pnl IS NOT NULL
      AND resolved_at >= NOW() - INTERVAL '7 days'
    GROUP BY DATE(resolved_at)
    ORDER BY trade_date DESC
""")

_QUERY_TRADE_COUNT = text("""
    SELECT COUNT(*) AS cnt
    FROM paper_trades
    WHERE strategy = 'B3'
      AND status IN ('won', 'lost', 'sold')
      AND actual_pnl IS NOT NULL
""")


async def fetch_b3_metrics(
    session_factory: Any,
    rolling_window: int = 200,
) -> B3MetricsSnapshot | None:
    """Fetch and compute all B3 metrics.

    Args:
        session_factory: SQLAlchemy async session factory.
        rolling_window: Number of recent trades to analyze.

    Returns:
        B3MetricsSnapshot or None if insufficient data.
    """
    try:
        async with session_factory() as session:
            # Get total trade count
            result = await session.execute(_QUERY_TRADE_COUNT)
            row = result.fetchone()
            total_trades = row.cnt if row else 0

            if total_trades < 20:
                logger.info("b3_metrics_insufficient_data", trades=total_trades)
                return None

            # Fetch recent trades
            result = await session.execute(
                _QUERY_B3_TRADES, {"limit": rolling_window}
            )
            rows = result.fetchall()

            # Fetch daily PnL
            result_daily = await session.execute(_QUERY_DAILY_PNL)
            daily_rows = result_daily.fetchall()

        if not rows:
            return None

        # Convert to dicts for processing
        trades = [dict(r._mapping) for r in rows]
        daily_pnl = [dict(r._mapping) for r in daily_rows]

        return _compute_metrics(trades, daily_pnl, total_trades)

    except Exception as e:
        logger.error("b3_metrics_fetch_error", error=str(e))
        return None


def _compute_metrics(
    trades: list[dict],
    daily_pnl: list[dict],
    total_trades: int,
) -> B3MetricsSnapshot:
    """Compute all metrics from trade data."""
    now = time.time()
    n = len(trades)

    # Paper outcome metrics (secondary — context only, paper PnL is approximate)
    pnls = [float(t["actual_pnl"]) for t in trades if t["actual_pnl"] is not None]
    wins_paper = sum(
        1 for t in trades
        if t.get("actual_pnl") is not None and float(t["actual_pnl"]) > 0
    )
    wr_paper = wins_paper / n if n > 0 else 0.0

    # LIVE metrics (PRIMARY) — computed from live_pnl (real money PnL)
    # Includes both "filled" and "partial" fill statuses, only those with resolution
    live_trades = [
        t for t in trades
        if t.get("live_status") in ("filled", "partial")
        and t.get("live_pnl") is not None
    ]
    live_pnls = [float(t["live_pnl"]) for t in live_trades]
    live_wins = sum(1 for pnl in live_pnls if pnl > 0)
    live_n = len(live_trades)
    wr_live = live_wins / live_n if live_n > 0 else 0.0
    avg_pnl_live = sum(live_pnls) / live_n if live_n > 0 else 0.0
    total_pnl_live = sum(live_pnls) if live_pnls else 0.0

    # Live Sharpe
    if len(live_pnls) > 1:
        mean_l = avg_pnl_live
        std_l = math.sqrt(sum((p - mean_l) ** 2 for p in live_pnls) / (len(live_pnls) - 1))
        sharpe_live = (mean_l / std_l) * math.sqrt(33) if std_l > 0 else 0.0
    else:
        sharpe_live = 0.0

    # Average PnL and Sharpe
    avg_pnl = sum(pnls) / len(pnls) if pnls else 0.0
    if len(pnls) > 1:
        mean_pnl = avg_pnl
        std_pnl = math.sqrt(sum((p - mean_pnl) ** 2 for p in pnls) / (len(pnls) - 1))
        sharpe = (mean_pnl / std_pnl) * math.sqrt(33) if std_pnl > 0 else 0.0  # ~33 trades/day
    else:
        sharpe = 0.0

    # Consecutive losses
    max_consec_paper = _max_consecutive_losses(trades, live_only=False)
    max_consec_live = _max_consecutive_losses(trades, live_only=True)

    # Regime breakdowns
    sigma_regime = _bucket_regime(trades, "sigma_norm", [
        ("CALM", 0, 1.5), ("NORMAL", 1.5, 2.0),
        ("ACTIVE", 2.0, 2.5), ("VOLATILE", 2.5, 100),
    ])
    velocity_regime = _bucket_regime(trades, "velocity", [
        ("SLOW", 0, 30), ("MEDIUM", 30, 60), ("FAST", 60, 10000),
    ])
    spread_regime = _bucket_regime(trades, "spread", [
        ("TIGHT", 0, 0.02), ("NORMAL", 0.02, 0.05), ("WIDE", 0.05, 100),
    ])

    # TA regimes (from trade_details, may have None values)
    adx_regime = _bucket_regime_categorical(trades, "ta_adx_regime")
    rsi_regime = _bucket_regime_categorical(trades, "ta_rsi_zone")

    # PSI drift
    psi_scores = _compute_psi_scores(trades)

    # ECE calibration
    ece = _compute_ece(trades)

    # Market microstructure averages
    spreads = [float(t["spread"]) for t in trades if t.get("spread") is not None]
    cl_deltas = [float(t["cl_delta"]) for t in trades if t.get("cl_delta") is not None]
    fill_prices = [float(t["live_fill_price"]) for t in live_trades if t.get("live_fill_price")]
    liquidities = [float(t["liquidity"]) for t in trades if t.get("liquidity") is not None]

    # Format daily PnL
    daily_formatted = []
    for d in daily_pnl:
        daily_formatted.append({
            "date": str(d["trade_date"]),
            "trades": int(d["trades"]),
            "wins": int(d["wins"]),
            "pnl": float(d["daily_pnl"]) if d["daily_pnl"] else 0,
            "live_trades": int(d["live_trades"]) if d.get("live_trades") else 0,
            "live_wins": int(d["live_wins"]) if d.get("live_wins") else 0,
        })

    return B3MetricsSnapshot(
        timestamp=now,
        total_trades=total_trades,
        rolling_n=n,
        # LIVE metrics (primary)
        rolling_n_live=live_n,
        rolling_wr_live=round(wr_live, 4),
        rolling_avg_pnl_live=round(avg_pnl_live, 4),
        rolling_total_pnl_live=round(total_pnl_live, 2),
        rolling_sharpe_live=round(sharpe_live, 2),
        max_consecutive_losses_live=max_consec_live,
        # Paper metrics (secondary)
        rolling_wr_paper=round(wr_paper, 4),
        rolling_avg_pnl=round(avg_pnl, 4),
        rolling_sharpe=round(sharpe, 2),
        max_consecutive_losses_paper=max_consec_paper,
        sigma_regime=sigma_regime,
        velocity_regime=velocity_regime,
        spread_regime=spread_regime,
        adx_regime=adx_regime,
        rsi_regime=rsi_regime,
        psi_scores=psi_scores,
        ece=round(ece, 4),
        avg_spread=round(sum(spreads) / len(spreads), 4) if spreads else 0.0,
        avg_cl_delta=round(sum(cl_deltas) / len(cl_deltas), 2) if cl_deltas else 0.0,
        avg_fill_price=round(sum(fill_prices) / len(fill_prices), 4) if fill_prices else 0.0,
        avg_liquidity=round(sum(liquidities) / len(liquidities), 0) if liquidities else 0.0,
        daily_pnl=daily_formatted,
    )


# ── Helpers ──────────────────────────────────────────────────────────

def _max_consecutive_losses(trades: list[dict], live_only: bool) -> int:
    """Find longest losing streak.

    For live_only=True: uses live_pnl (real money).
    For paper: uses actual_pnl.
    """
    max_streak = 0
    current = 0
    for t in reversed(trades):  # oldest first
        if live_only:
            if t.get("live_status") not in ("filled", "partial"):
                continue
            pnl = t.get("live_pnl")
        else:
            pnl = t.get("actual_pnl")

        if pnl is not None and float(pnl) < 0:
            current += 1
            max_streak = max(max_streak, current)
        elif pnl is not None and float(pnl) > 0:
            current = 0
    return max_streak


def _bucket_regime(
    trades: list[dict],
    field: str,
    buckets: list[tuple[str, float, float]],
) -> dict[str, dict[str, float]]:
    """Bucket trades by a numeric field and compute WR/PnL per bucket."""
    result: dict[str, dict[str, float]] = {}
    for name, low, high in buckets:
        bucket_trades = [
            t for t in trades
            if t.get(field) is not None and low <= float(t[field]) < high
        ]
        n = len(bucket_trades)
        if n == 0:
            result[name] = {"wr": 0, "n": 0, "pnl": 0}
            continue
        wins = sum(
            1 for t in bucket_trades
            if t.get("actual_pnl") is not None and float(t["actual_pnl"]) > 0
        )
        pnl = sum(float(t.get("actual_pnl") or 0) for t in bucket_trades)
        result[name] = {
            "wr": round(wins / n, 4),
            "n": n,
            "pnl": round(pnl, 2),
        }
    return result


def _bucket_regime_categorical(
    trades: list[dict],
    field: str,
) -> dict[str, dict[str, float]]:
    """Bucket trades by a categorical field (TA regime names)."""
    buckets: dict[str, list[dict]] = {}
    for t in trades:
        val = t.get(field)
        if val is None or val == "UNKNOWN":
            continue
        buckets.setdefault(val, []).append(t)

    result: dict[str, dict[str, float]] = {}
    for name, bucket_trades in buckets.items():
        n = len(bucket_trades)
        wins = sum(
            1 for t in bucket_trades
            if t.get("actual_pnl") is not None and float(t["actual_pnl"]) > 0
        )
        pnl = sum(float(t.get("actual_pnl") or 0) for t in bucket_trades)
        result[name] = {
            "wr": round(wins / n, 4),
            "n": n,
            "pnl": round(pnl, 2),
        }
    return result


def _compute_psi_scores(trades: list[dict]) -> dict[str, float]:
    """Compute PSI (Population Stability Index) for key features.

    Compares first half vs second half of trades.
    PSI < 0.10: no drift. 0.10-0.25: moderate. >= 0.25: significant.
    """
    if len(trades) < 40:
        return {}

    mid = len(trades) // 2
    # Trades are newest-first, so first half = recent, second half = baseline
    recent = trades[:mid]
    baseline = trades[mid:]

    features = ["velocity", "spread", "sigma_norm", "cl_delta"]
    if any(t.get("ta_adx") is not None for t in trades):
        features.extend(["ta_adx", "ta_rsi"])

    result: dict[str, float] = {}
    for feat in features:
        ref_vals = [float(t[feat]) for t in baseline if t.get(feat) is not None]
        cur_vals = [float(t[feat]) for t in recent if t.get(feat) is not None]
        if len(ref_vals) < 10 or len(cur_vals) < 10:
            continue
        psi = _psi(ref_vals, cur_vals)
        result[feat] = round(psi, 4)

    return result


def _psi(reference: list[float], current: list[float], n_bins: int = 10) -> float:
    """Calculate Population Stability Index between two distributions."""
    if not reference or not current:
        return 0.0

    # Create bins from reference distribution
    sorted_ref = sorted(reference)
    step = max(1, len(sorted_ref) // n_bins)
    breakpoints = [sorted_ref[0] - 1e-9]
    for i in range(step, len(sorted_ref), step):
        breakpoints.append(sorted_ref[i])
    breakpoints.append(sorted_ref[-1] + 1e-9)

    # Count in each bin
    def _bin_counts(vals: list[float]) -> list[int]:
        counts = [0] * (len(breakpoints) - 1)
        for v in vals:
            for j in range(len(breakpoints) - 1):
                if breakpoints[j] <= v < breakpoints[j + 1]:
                    counts[j] += 1
                    break
            else:
                counts[-1] += 1
        return counts

    ref_counts = _bin_counts(reference)
    cur_counts = _bin_counts(current)

    ref_n = len(reference)
    cur_n = len(current)
    if ref_n == 0 or cur_n == 0:
        return 0.0

    psi_val = 0.0
    for rc, cc in zip(ref_counts, cur_counts, strict=False):
        ref_pct = max(rc / ref_n, 1e-6)
        cur_pct = max(cc / cur_n, 1e-6)
        psi_val += (cur_pct - ref_pct) * math.log(cur_pct / ref_pct)

    return max(psi_val, 0.0)


def _compute_ece(trades: list[dict], n_bins: int = 5) -> float:
    """Expected Calibration Error: predicted edge vs actual win rate.

    Lower ECE = model better predicts outcomes.
    ECE < 0.05: excellent. 0.05-0.10: acceptable. > 0.15: problem.
    """
    edges = [float(t["edge"]) for t in trades if t.get("edge") is not None]
    outcomes = [
        1.0 if (t.get("actual_pnl") is not None and float(t["actual_pnl"]) > 0) else 0.0
        for t in trades if t.get("edge") is not None
    ]

    if len(edges) < 20:
        return 0.0

    # Sort by edge and split into bins
    paired = sorted(zip(edges, outcomes, strict=True))
    bin_size = max(1, len(paired) // n_bins)

    ece = 0.0
    total = len(paired)
    for i in range(0, len(paired), bin_size):
        bin_data = paired[i:i + bin_size]
        if len(bin_data) < 3:
            continue
        avg_edge = sum(e for e, _ in bin_data) / len(bin_data)
        actual_wr = sum(o for _, o in bin_data) / len(bin_data)
        # Expected WR from edge (rough: edge is |signal_fv - 0.50|)
        expected_wr = 0.50 + avg_edge
        ece += (len(bin_data) / total) * abs(actual_wr - expected_wr)

    return ece
