"""Web dashboard for Arbo trading system monitoring.

FastAPI app serving a single-page dashboard with real-time API endpoints.
Integrated into the orchestrator as an internal task (same pattern as Slack bot).
"""

from __future__ import annotations

import os
import secrets
import time
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import psutil
import sqlalchemy as sa
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.templating import Jinja2Templates

from arbo.utils.logger import get_logger

logger = get_logger("web_dashboard")

# ---------------------------------------------------------------------------
# CryptoArb state reader (file-based bridge)
# ---------------------------------------------------------------------------

CRYPTOARB_STATE_PATH = os.getenv(
    "CRYPTOARB_STATE_PATH", "/opt/cryptoarb/production_data/state.json"
)


class CryptoArbReader:
    """Reads CryptoArb state.json with caching and stale detection."""

    CACHE_TTL = 10  # seconds
    STALE_THRESHOLD = 7200  # 2 hours

    def __init__(self, path: str = CRYPTOARB_STATE_PATH):
        self.path = path
        self._cache: dict[str, Any] | None = None
        self._cache_time: float = 0.0

    def read(self) -> dict[str, Any] | None:
        """Read state.json with 10s cache. Returns None if file missing."""
        now = time.time()
        if self._cache is not None and (now - self._cache_time) < self.CACHE_TTL:
            return self._cache
        try:
            import json as _json

            with open(self.path) as f:
                data = _json.load(f)
            self._cache = data
            self._cache_time = now
            return data
        except FileNotFoundError:
            return None
        except Exception:
            # JSON parse error — return cached value if available
            return self._cache

    def is_stale(self) -> bool:
        """Check if CryptoArb data is stale.

        Uses last_daily_update from state: stale if it's more than 1 day old.
        Falls back to file mtime (24h threshold) if field is missing.
        """
        data = self.read()
        if data is None:
            return True
        last_daily = data.get("last_daily_update", "")
        if last_daily:
            try:
                last_date = datetime.strptime(last_daily, "%Y-%m-%d").date()
                today = datetime.now(UTC).date()
                return (today - last_date).days > 1
            except ValueError:
                pass
        # Fallback: file mtime > 24h
        try:
            mtime = os.path.getmtime(self.path)
            return (time.time() - mtime) > 86400
        except OSError:
            return True


_cryptoarb = CryptoArbReader()


# ---------------------------------------------------------------------------
# Shared state — injected by orchestrator at init
# ---------------------------------------------------------------------------


class DashboardState:
    """Shared state injected by orchestrator."""

    orchestrator: Any = None


state = DashboardState()

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="Arbo Dashboard", docs_url=None, redoc_url=None)
security = HTTPBasic()
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))


def _verify_credentials(credentials: HTTPBasicCredentials = Depends(security)) -> str:
    """Verify HTTP Basic Auth credentials."""
    expected_user = os.getenv("DASHBOARD_USER", "arbo")
    expected_pass = os.getenv("DASHBOARD_PASSWORD", "")

    if not expected_pass:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Dashboard password not configured",
        )

    user_ok = secrets.compare_digest(credentials.username.encode(), expected_user.encode())
    pass_ok = secrets.compare_digest(credentials.password.encode(), expected_pass.encode())

    if not (user_ok and pass_ok):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


# ---------------------------------------------------------------------------
# Helper: decimal-safe JSON serialization
# ---------------------------------------------------------------------------


def _dec(val: Any) -> float | None:
    """Convert Decimal/numeric to float for JSON serialization."""
    if val is None:
        return None
    if isinstance(val, Decimal):
        return float(val)
    return float(val)


# Strategy metadata (RDH architecture)
_STRATEGY_META: dict[str, dict[str, str]] = {
    "A": {"name": "Theta Decay", "category": "Longshots", "description": "Sell optimism premium on longshot YES contracts"},
    "B": {"name": "Reflexivity Surfer", "category": "Trending", "description": "Ride reflexive momentum in trending markets"},
    "B2": {"name": "Crypto Price Edge", "category": "Crypto", "description": "Volatility model vs Binance price on crypto prediction markets"},
    "B3": {"name": "Binance Oracle Scalper", "category": "Crypto", "description": "BTC 5-min Up/Down momentum scalper via Binance price oracle"},
    "B3_15M": {"name": "Binance Oracle Scalper 15m", "category": "Crypto", "description": "BTC 15-min Up/Down momentum scalper — same Binance→Chainlink edge, longer window, higher PnL/trade"},
    "C": {"name": "Compound Weather", "category": "Weather", "description": "Weather temperature ladder trades"},
    "C2": {"name": "EMOS Exit Fusion", "category": "Weather", "description": "EMOS adaptive probability + edge-based early exit"},
    "D": {"name": "NBA Green Book", "category": "Sports", "description": "NBA pre-game entry, green book exit mid-game (both sides, always-close)"},
    "D_UFC": {"name": "UFC Green Book", "category": "Sports", "description": "UFC fight green book — sweep v2 winner (355 trades, DD 3%, Sharpe 14)"},
    "D_EPL": {"name": "EPL Green Book", "category": "Sports", "description": "EPL + cups green book — sweep winner (100% profitable, WR 64%, DD 3%)"},
}


def _infer_category(question: str | None, details: dict[str, Any] | None = None) -> str:
    """Infer market category from signal details or market question."""
    if details:
        cat = details.get("category", "")
        if cat:
            return cat.capitalize()
    if not question:
        return "Other"
    q = question.lower()
    if any(kw in q for kw in ("btc", "bitcoin", "eth", "ethereum", "sol", "solana", "crypto")):
        return "Crypto"
    if any(kw in q for kw in ("election", "trump", "biden", "senate", "congress", "president")):
        return "Politics"
    if any(
        kw in q
        for kw in (
            "win on",
            "beat",
            "draw",
            "premier league",
            "la liga",
            "bundesliga",
            "serie a",
            "soccer",
            "football",
        )
    ):
        return "Soccer"
    return "Other"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
async def dashboard_page(
    request: Request, _user: str = Depends(_verify_credentials)
) -> HTMLResponse:
    """Serve the main dashboard HTML page."""
    orch = state.orchestrator
    mode = orch._mode if orch else "unknown"
    response = templates.TemplateResponse("dashboard.html", {
        "request": request,
        "mode": mode,
        "nightcap_api_url": os.environ.get("NIGHTCAP_API_URL", ""),
        "nightcap_api_key": os.environ.get("NIGHTCAP_API_KEY", ""),
        "skinny_api_url": os.environ.get("SKINNY_API_URL", ""),
        "skinny_api_key": os.environ.get("SKINNY_API_KEY", ""),
    })
    # Force fresh HTML on every load — dashboard updates frequently and
    # users were getting stale cached cards after deploys (Phase 1 lesson).
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


# ── Echo proxy endpoints (forward to Singapore VPS) ──
ECHO_API = "http://13.212.182.85:8081"

@app.get("/api/echo/stats")
async def api_echo_stats(_user: str = Depends(_verify_credentials)):
    import aiohttp
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{ECHO_API}/api/stats", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                return await resp.json()
    except Exception:
        return {"total_trades": 0, "total_pnl": 0, "win_rate": 0, "wins": 0, "losses": 0, "per_pair": {}, "daily": [], "equity_curve": [], "uptime_hours": 0}

@app.get("/api/echo/config")
async def api_echo_config(_user: str = Depends(_verify_credentials)):
    import aiohttp
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{ECHO_API}/api/config", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                return await resp.json()
    except Exception:
        return {}

@app.get("/api/echo/trades")
async def api_echo_trades(_user: str = Depends(_verify_credentials)):
    import aiohttp
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{ECHO_API}/api/trades", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                return await resp.json()
    except Exception:
        return []


@app.get("/api/portfolio")
async def api_portfolio(_user: str = Depends(_verify_credentials)) -> dict[str, Any]:
    """Portfolio summary: balance, P&L, snapshot history, per-category breakdown."""
    orch = state.orchestrator
    if orch is None:
        return {"error": "Orchestrator not available"}

    engine = orch._paper_engine
    balance = _dec(engine.balance) if engine else 0.0
    total_value = _dec(engine.total_value) if engine else 0.0
    initial = _dec(orch._capital)

    # Query DB for snapshots and per-category breakdown
    snapshots: list[dict[str, Any]] = []
    category_pnl: dict[str, dict[str, Any]] = {}

    try:
        from arbo.utils.db import (
            Market,
            PaperPosition,
            PaperSnapshot,
            PaperTrade,
            get_session_factory,
        )

        factory = get_session_factory()
        async with factory() as session:
            # Per-category P&L: join trades with markets to get category
            # Exclude pre-validation trades from dashboard metrics
            result = await session.execute(
                sa.select(
                    Market.category,
                    sa.func.count(PaperTrade.id),
                    sa.func.coalesce(sa.func.sum(PaperTrade.actual_pnl), 0),
                    sa.func.coalesce(
                        sa.func.sum(
                            sa.case(
                                (PaperTrade.status == "open", PaperTrade.size),
                                else_=sa.literal(0),
                            )
                        ),
                        0,
                    ),
                )
                .outerjoin(Market, PaperTrade.market_condition_id == Market.condition_id)
                .where(
                    sa.or_(
                        PaperTrade.notes.is_(None),
                        PaperTrade.notes != "pre-validation",
                    )
                )
                .group_by(Market.category)
            )
            for row in result.all():
                cat = (row[0] or "other").capitalize()
                category_pnl[cat] = {
                    "trades": row[1],
                    "realized_pnl": _dec(row[2]) or 0.0,
                    "invested": _dec(row[3]) or 0.0,
                }

            # Open positions per category
            result = await session.execute(
                sa.select(
                    Market.category,
                    sa.func.count(PaperPosition.id),
                    sa.func.coalesce(sa.func.sum(PaperPosition.size), 0),
                    sa.func.coalesce(sa.func.sum(PaperPosition.unrealized_pnl), 0),
                )
                .outerjoin(Market, PaperPosition.market_condition_id == Market.condition_id)
                .group_by(Market.category)
            )
            for row in result.all():
                cat = (row[0] or "other").capitalize()
                if cat not in category_pnl:
                    category_pnl[cat] = {"trades": 0, "realized_pnl": 0.0, "invested": 0.0}
                category_pnl[cat]["open_positions"] = row[1]
                category_pnl[cat]["invested_open"] = _dec(row[2]) or 0.0
                category_pnl[cat]["unrealized_pnl"] = _dec(row[3]) or 0.0

            # Snapshots for chart (last 7 days)
            result = await session.execute(
                sa.select(
                    PaperSnapshot.balance,
                    PaperSnapshot.total_value,
                    PaperSnapshot.unrealized_pnl,
                    PaperSnapshot.snapshot_at,
                )
                .order_by(PaperSnapshot.snapshot_at.desc())
                .limit(168)
            )
            for row in result.all():
                snapshots.append(
                    {
                        "balance": _dec(row.balance),
                        "total_value": _dec(row.total_value),
                        "unrealized_pnl": _dec(row.unrealized_pnl),
                        "timestamp": row.snapshot_at.isoformat() if row.snapshot_at else None,
                    }
                )
            snapshots.reverse()

            # Reconstruct corrected chart from cumulative realized P&L at each snapshot time.
            # Paper engine balance drifts after restarts; use trade-level P&L instead.
            if snapshots:
                import bisect

                result = await session.execute(
                    sa.select(PaperTrade.resolved_at, PaperTrade.actual_pnl)
                    .where(PaperTrade.status.in_(["won", "lost", "sold"]))
                    .where(PaperTrade.resolved_at.isnot(None))
                    .where(
                        sa.or_(
                            PaperTrade.notes.is_(None),
                            PaperTrade.notes != "pre-validation",
                        )
                    )
                    .order_by(PaperTrade.resolved_at)
                )
                trades_pnl = [
                    (row.resolved_at.isoformat(), _dec(row.actual_pnl) or 0.0)
                    for row in result.all()
                ]

                # Build cumulative P&L series from trades
                trade_times = [t[0] for t in trades_pnl]
                cumulative = []
                running = 0.0
                for _, pnl in trades_pnl:
                    running += pnl
                    cumulative.append(running)

                # For each snapshot, compute cumulative P&L (realized + unrealized)
                # This is capital-agnostic — no jumps when adding strategies
                for s in snapshots:
                    ts = s["timestamp"] or ""
                    idx = bisect.bisect_right(trade_times, ts) - 1
                    realized_at_time = cumulative[idx] if idx >= 0 else 0.0
                    s["total_value"] = round(
                        realized_at_time + (s["unrealized_pnl"] or 0.0), 2
                    )

                # Append live "now" point so chart always shows current P&L
                now_realized = cumulative[-1] if cumulative else 0.0
                now_unrealized = 0.0
                if engine:
                    pos_dict = getattr(engine, "_positions", None) or {}
                    for pos in pos_dict.values():
                        now_unrealized += _dec(getattr(pos, "unrealized_pnl", 0)) or 0.0
                snapshots.append({
                    "balance": 0.0,
                    "total_value": round(now_realized + now_unrealized, 2),
                    "unrealized_pnl": round(now_unrealized, 2),
                    "timestamp": datetime.now(UTC).isoformat(),
                })

    except Exception as e:
        logger.warning("portfolio_query_error", error=str(e))

    # Per-strategy P&L from risk manager (reliable source of truth)
    strategy_pnl: dict[str, dict[str, Any]] = {}
    realized_total_pnl = 0.0
    realized_weekly_pnl = 0.0
    if orch._risk_manager:
        rm = orch._risk_manager
        for sid, meta in _STRATEGY_META.items():
            ss = rm.get_strategy_state(sid)
            if ss:
                s_total = _dec(ss.total_pnl) or 0.0
                s_weekly = _dec(ss.weekly_pnl) or 0.0
                realized_total_pnl += s_total
                realized_weekly_pnl += s_weekly
                strategy_pnl[sid] = {
                    "name": meta["name"],
                    "allocated": _dec(ss.allocated),
                    "deployed": _dec(ss.deployed),
                    "available": _dec(ss.available),
                    "weekly_pnl": _dec(ss.weekly_pnl),
                    "total_pnl": _dec(ss.total_pnl),
                    "positions": ss.position_count,
                    "is_halted": ss.is_halted,
                }

    # Unrealized P&L from open positions
    unrealized_pnl = 0.0
    if engine:
        pos_dict = getattr(engine, "_positions", None) or getattr(engine, "positions", {})
        for pos in pos_dict.values():
            unrealized_pnl += _dec(getattr(pos, "unrealized_pnl", 0)) or 0.0

    # Use strategy-level P&L (reliable) instead of snapshot-based (drifts after restarts)
    total_pnl = realized_total_pnl + unrealized_pnl
    corrected_total_value = (initial or 0.0) + total_pnl
    roi_pct = (total_pnl / initial * 100) if initial else 0.0

    # Per-strategy daily + monthly P&L from resolved trades
    monthly_pnl = 0.0
    realized_daily_pnl = 0.0
    try:
        from arbo.utils.db import PaperTrade, get_session_factory as _gsf

        factory = _gsf()
        async with factory() as session:
            now_utc = datetime.now(UTC)
            today_start = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
            month_start = now_utc.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

            # Per-strategy daily + monthly + total in one query (from DB — survives restarts)
            week_start = today_start - timedelta(days=today_start.weekday())
            result = await session.execute(
                sa.select(
                    PaperTrade.strategy,
                    sa.func.coalesce(
                        sa.func.sum(
                            sa.case(
                                (PaperTrade.resolved_at >= today_start, PaperTrade.actual_pnl),
                                else_=sa.literal(0),
                            )
                        ),
                        0,
                    ).label("daily"),
                    sa.func.coalesce(
                        sa.func.sum(
                            sa.case(
                                (PaperTrade.resolved_at >= week_start, PaperTrade.actual_pnl),
                                else_=sa.literal(0),
                            )
                        ),
                        0,
                    ).label("weekly"),
                    sa.func.coalesce(
                        sa.func.sum(
                            sa.case(
                                (PaperTrade.resolved_at >= month_start, PaperTrade.actual_pnl),
                                else_=sa.literal(0),
                            )
                        ),
                        0,
                    ).label("monthly"),
                    sa.func.coalesce(
                        sa.func.sum(PaperTrade.actual_pnl), 0
                    ).label("total"),
                )
                .where(PaperTrade.status.in_(["won", "lost", "sold"]))
                .where(
                    sa.or_(
                        PaperTrade.notes.is_(None),
                        PaperTrade.notes != "pre-validation",
                    )
                )
                .group_by(PaperTrade.strategy)
            )
            for row in result.all():
                sid = row[0] or ""
                s_daily = _dec(row[1]) or 0.0
                s_weekly = _dec(row[2]) or 0.0
                s_monthly = _dec(row[3]) or 0.0
                s_total = _dec(row[4]) or 0.0
                realized_daily_pnl += s_daily
                realized_weekly_pnl = realized_weekly_pnl  # already from RM
                monthly_pnl += s_monthly
                if sid in strategy_pnl:
                    strategy_pnl[sid]["daily_pnl"] = round(s_daily, 2)
                    strategy_pnl[sid]["weekly_pnl"] = round(s_weekly, 2)
                    strategy_pnl[sid]["monthly_pnl"] = round(s_monthly, 2)
                    # Use DB total — risk manager resets after restart
                    strategy_pnl[sid]["total_pnl"] = round(s_total, 2)
                    realized_total_pnl = sum(
                        sp.get("total_pnl", 0) or 0 for sp in strategy_pnl.values()
                    )

            # Ensure all strategies have all P&L keys
            for sid in strategy_pnl:
                strategy_pnl[sid].setdefault("daily_pnl", 0.0)
                strategy_pnl[sid].setdefault("weekly_pnl", 0.0)
                strategy_pnl[sid].setdefault("monthly_pnl", 0.0)
                # ROI per strategy
                alloc = strategy_pnl[sid].get("allocated", 0) or 1
                strategy_pnl[sid]["roi_pct"] = round(
                    (strategy_pnl[sid].get("total_pnl", 0) or 0) / alloc * 100, 2
                )

    except Exception as e:
        logger.warning("strategy_pnl_query_error", error=str(e))

    return {
        "balance": round(corrected_total_value, 2),
        "total_value": round(corrected_total_value, 2),
        "initial_capital": initial,
        "daily_pnl": round(realized_daily_pnl, 2),
        "weekly_pnl": round(realized_weekly_pnl, 2),
        "monthly_pnl": round(monthly_pnl, 2),
        "total_pnl": round(total_pnl, 2),
        "roi_pct": round(roi_pct, 2),
        "snapshots": snapshots,
        "category_pnl": category_pnl,
        "strategy_pnl": strategy_pnl,
    }


@app.get("/api/positions")
async def api_positions(_user: str = Depends(_verify_credentials)) -> dict[str, Any]:
    """Open positions with market names and category."""
    orch = state.orchestrator
    if orch is None:
        return {"positions": []}

    positions: list[dict[str, Any]] = []
    try:
        from arbo.utils.db import Market, PaperPosition, get_session_factory

        factory = get_session_factory()
        async with factory() as session:
            result = await session.execute(
                sa.select(
                    PaperPosition, Market.question, Market.category, Market.end_date
                )
                .outerjoin(Market, PaperPosition.market_condition_id == Market.condition_id)
                .order_by(PaperPosition.opened_at.desc())
            )
            now = datetime.now(UTC)
            for row in result.all():
                pos = row[0]
                question = row[1] or pos.market_condition_id[:20]
                category = (row[2] or "other").capitalize()
                end_date = row[3]
                if category == "Other":
                    category = _infer_category(question)
                positions.append(
                    {
                        "market": question,
                        "condition_id": pos.market_condition_id,
                        "token_id": pos.token_id,
                        "side": pos.side,
                        "size": _dec(pos.size),
                        "avg_price": _dec(pos.avg_price),
                        "current_price": _dec(pos.current_price),
                        "unrealized_pnl": _dec(pos.unrealized_pnl),
                        "layer": pos.layer,
                        "strategy": getattr(pos, "strategy", "") or "",
                        "category": category,
                        "opened_at": pos.opened_at.isoformat() if pos.opened_at else None,
                        "end_date": end_date.isoformat() if end_date else None,
                    }
                )
    except Exception as e:
        logger.warning("positions_query_error", error=str(e))

    return {"positions": positions}


@app.get("/api/signals")
async def api_signals(
    _user: str = Depends(_verify_credentials),
    strategy: str | None = None,
) -> dict[str, Any]:
    """Last 100 tradeable signals (edge ≤ 12%) with market names and category.

    Signals with edge > 12% are excluded (model error / line mismatch).
    By default only RDH signals are shown (layer=0). Pass strategy=A/B/C to
    filter further, or strategy=all to include legacy signals.
    """
    signals: list[dict[str, Any]] = []
    max_edge = Decimal("0.12")
    try:
        from arbo.utils.db import Market, get_session_factory
        from arbo.utils.db import Signal as DBSignal

        factory = get_session_factory()
        async with factory() as session:
            query = (
                sa.select(DBSignal, Market.question, Market.category)
                .outerjoin(Market, DBSignal.market_condition_id == Market.condition_id)
                .where(sa.or_(DBSignal.edge <= max_edge, DBSignal.edge.is_(None)))
            )

            # Strategy filter: default = RDH only (layer 0)
            if strategy and strategy.upper() in ("A", "B", "C"):
                # Filter to specific strategy via JSONB details->>'strategy'
                query = query.where(
                    DBSignal.details["strategy"].astext == strategy.upper()
                )
            elif strategy and strategy.lower() == "all":
                pass  # Show everything including legacy
            else:
                # Default: only RDH signals (layer=0)
                query = query.where(DBSignal.layer == 0)

            query = query.order_by(DBSignal.detected_at.desc()).limit(100)
            result = await session.execute(query)
            for row in result.all():
                sig = row[0]
                question = row[1] or sig.market_condition_id[:20]
                details = sig.details or {}
                reason = details.get("reason", details.get("match_type", ""))
                # Category from signal details (most accurate) or market table
                category = details.get("category", row[2] or "other")
                if isinstance(category, str):
                    category = category.capitalize()
                sig_strategy = details.get("strategy", "")
                signals.append(
                    {
                        "timestamp": sig.detected_at.isoformat() if sig.detected_at else None,
                        "layer": sig.layer,
                        "strategy": sig_strategy,
                        "market": question,
                        "direction": sig.direction,
                        "edge": _dec(sig.edge),
                        "confidence": _dec(sig.confidence),
                        "confluence_score": sig.confluence_score,
                        "category": category,
                        "reason": str(reason)[:100] if reason else "",
                    }
                )
    except Exception as e:
        logger.warning("signals_query_error", error=str(e))

    return {"signals": signals}


@app.get("/api/layers")
async def api_layers(_user: str = Depends(_verify_credentials)) -> dict[str, Any]:
    """Per-task status (RDH: _tasks, legacy: _layers)."""
    orch = state.orchestrator
    if orch is None:
        return {"layers": []}

    now = time.monotonic()
    layers: list[dict[str, Any]] = []

    # Use _tasks (RDH) or _layers (legacy)
    task_dict: dict[str, Any] = getattr(orch, "_tasks", None) or getattr(
        orch, "_layers", {}
    )

    # Strategy name mapping for RDH tasks
    task_labels: dict[str, str] = {
        "discovery": "Discovery",
        "strategy_A": "Strategy A — Theta Decay",
        "strategy_B": "Strategy B — Reflexivity Surfer",
        "strategy_C": "Strategy C — Compound Weather",
    }

    for name, ts in task_dict.items():
        heartbeat_ago = now - ts.last_heartbeat
        task_running = ts.task is not None and not ts.task.done()

        # Determine status color
        if ts.permanent_stop or (ts.task and ts.task.done()) or heartbeat_ago > 300:
            status_color = "red"
        elif ts.restart_count > 0:
            status_color = "yellow"
        else:
            status_color = "green"

        layers.append(
            {
                "name": name,
                "label": task_labels.get(name, name),
                "status": status_color,
                "running": task_running,
                "enabled": ts.enabled,
                "permanent_stop": ts.permanent_stop,
                "heartbeat_ago_s": int(heartbeat_ago),
                "restart_count": ts.restart_count,
            }
        )

    return {"layers": layers}


@app.get("/api/risk")
async def api_risk(_user: str = Depends(_verify_credentials)) -> dict[str, Any]:
    """Risk metrics: daily/weekly loss utilization, category exposure."""
    orch = state.orchestrator
    if orch is None:
        return {"error": "Orchestrator not available"}

    rm = orch._risk_manager
    if rm is None:
        return {"error": "Risk manager not available"}

    capital = _dec(orch._capital) or 1.0

    # Get exposure state from risk manager
    rs = rm._state if hasattr(rm, "_state") else None
    daily_pnl = _dec(rs.daily_pnl if rs else Decimal("0")) or 0.0
    weekly_pnl = _dec(rs.weekly_pnl if rs else Decimal("0")) or 0.0
    category_exposure: dict[str, float] = {}
    raw_cat = rs.category_exposure if rs else {}
    for cat, val in raw_cat.items():
        category_exposure[cat] = _dec(val) or 0.0

    # Risk limits (hardcoded constants)
    daily_limit = capital * 0.10
    weekly_limit = capital * 0.20

    # Only show utilization on LOSSES (positive P&L = no risk concern)
    daily_loss = max(0.0, -daily_pnl)
    weekly_loss = max(0.0, -weekly_pnl)
    daily_utilization = daily_loss / daily_limit * 100 if daily_limit else 0.0
    weekly_utilization = weekly_loss / weekly_limit * 100 if weekly_limit else 0.0

    return {
        "daily_pnl": daily_pnl,
        "daily_limit": daily_limit,
        "daily_utilization_pct": round(min(daily_utilization, 100), 1),
        "weekly_pnl": weekly_pnl,
        "weekly_limit": weekly_limit,
        "weekly_utilization_pct": round(min(weekly_utilization, 100), 1),
        "category_exposure": category_exposure,
        "max_category_pct": 30.0,
        "shutdown_active": getattr(rm, "_shutdown", False),
    }


@app.get("/api/drawdown")
async def api_drawdown(_user: str = Depends(_verify_credentials)) -> dict[str, Any]:
    """Per-strategy drawdown monitoring with history chart data."""
    orch = state.orchestrator
    if orch is None:
        return {"error": "Orchestrator not available"}

    rm = orch._risk_manager
    if rm is None:
        return {"error": "Risk manager not available"}

    from arbo.core.risk_manager import STRATEGY_WEEKLY_DRAWDOWN_PCT

    threshold_pct = float(STRATEGY_WEEKLY_DRAWDOWN_PCT) * 100  # 25%

    # Per-strategy current state
    strategies: list[dict[str, Any]] = []
    for sid in ["A", "B", "C"]:
        ss = rm.get_strategy_state(sid)
        if ss is None:
            continue
        allocated = float(ss.allocated)
        weekly = float(ss.weekly_pnl)
        dd_pct = abs(weekly) / allocated * 100 if weekly < 0 and allocated > 0 else 0.0
        strategies.append({
            "id": sid,
            "name": _STRATEGY_META.get(sid, {}).get("name", sid),
            "allocated": allocated,
            "weekly_pnl": round(weekly, 2),
            "drawdown_pct": round(dd_pct, 1),
            "threshold_pct": threshold_pct,
            "utilization_pct": round(min(dd_pct / threshold_pct * 100, 100), 1) if threshold_pct else 0,
            "is_halted": ss.is_halted,
            "headroom": round(threshold_pct - dd_pct, 1),
        })

    # Daily drawdown chart — cumulative resolved PnL per day per strategy (this week)
    chart_data: list[dict[str, Any]] = []
    try:
        from arbo.utils.db import PaperTrade, get_session_factory as _gsf

        now = datetime.now(UTC)
        monday = now - timedelta(days=now.weekday())
        week_start = monday.replace(hour=0, minute=0, second=0, microsecond=0)

        factory = _gsf()
        async with factory() as session:
            result = await session.execute(
                sa.select(
                    sa.func.date_trunc("day", PaperTrade.resolved_at).label("day"),
                    PaperTrade.strategy,
                    sa.func.coalesce(sa.func.sum(PaperTrade.actual_pnl), 0),
                    sa.func.count(PaperTrade.id),
                )
                .where(PaperTrade.status.in_(["won", "lost", "sold"]))
                .where(PaperTrade.resolved_at >= week_start)
                .where(PaperTrade.strategy.isnot(None))
                .where(
                    sa.or_(
                        PaperTrade.notes.is_(None),
                        PaperTrade.notes != "pre-validation",
                    )
                )
                .group_by(sa.text("1"), sa.text("2"))
                .order_by(sa.text("1"))
            )

            # Build cumulative series per strategy
            all_sids = list(_STRATEGY_META.keys())
            cumulative: dict[str, float] = {s: 0.0 for s in all_sids}
            day_data: dict[str, dict[str, Any]] = {}

            for row in result:
                day_str = row[0].strftime("%Y-%m-%d") if row[0] else None
                if not day_str:
                    continue
                sid = row[1]
                pnl = float(row[2])
                trades = row[3]

                if sid not in cumulative:
                    continue

                cumulative[sid] += pnl

                if day_str not in day_data:
                    day_data[day_str] = {}
                day_data[day_str][sid] = {
                    "pnl": round(pnl, 2),
                    "cumulative": round(cumulative[sid], 2),
                    "trades": trades,
                }

            for day_str in sorted(day_data.keys()):
                entry: dict[str, Any] = {"date": day_str}
                for sid in all_sids:
                    d = day_data[day_str].get(sid, {})
                    entry[f"{sid}_pnl"] = d.get("pnl", 0)
                    entry[f"{sid}_cumulative"] = d.get("cumulative", cumulative.get(sid, 0))
                    entry[f"{sid}_trades"] = d.get("trades", 0)
                chart_data.append(entry)

    except Exception as e:
        logger.warning("drawdown_chart_error", error=str(e))

    # Backtest reference
    backtest_max_dd = 15.6  # AR-0134 max drawdown %

    return {
        "strategies": strategies,
        "threshold_pct": threshold_pct,
        "backtest_max_dd_pct": backtest_max_dd,
        "chart": chart_data,
        "week_start": week_start.isoformat() if chart_data else None,
    }


@app.get("/api/download-progress")
async def api_download_progress(
    _user: str = Depends(_verify_credentials),
) -> dict[str, Any]:
    """Strategy D data download progress from download VPS."""
    import json as _json

    status_path = Path("/opt/arbo/research_d/data/download_status.json")
    if status_path.exists():
        try:
            data = _json.loads(status_path.read_text())
            done = data.get("markets_done", 0)
            total = data.get("markets_total", 1)
            prices = data.get("prices_total", 0)
            db_gb = data.get("db_size_bytes", 0) / (1024**3)
            workers = data.get("workers_alive", 0)
            updated = data.get("updated_at", "?")
            current_pass = data.get("pass", "?")
            pct = done / max(total, 1) * 100
            # Use ETA from update_progress.sh (computed from actual rate)
            eta_h = data.get("eta_hours", 0)

            # Cap progress at 100% for display
            pct = min(pct, 100.0)

            # Pass 2 info (if available)
            p2_ml = data.get("pass2_ml_done", 0)
            p2_sp = data.get("pass2_sp_done", 0)
            p2_total = data.get("pass2_total", 0)
            p2_done = data.get("pass2_done", p2_ml + p2_sp)
            p2_active = current_pass.startswith("2")

            return {
                "active": workers > 0,
                "pass": current_pass,
                "markets_done": min(done, total),
                "markets_total": total,
                "progress_pct": round(pct, 1),
                "prices_total": prices,
                "prices_millions": round(prices / 1_000_000, 1),
                "db_size_gb": round(db_gb, 1),
                "workers_alive": workers,
                "eta_hours": round(eta_h, 1),
                "updated_at": updated,
                "pass2_active": p2_active,
                "pass2_ml_done": p2_ml,
                "pass2_sp_done": p2_sp,
                "pass2_total": p2_total,
                "pass2_done": p2_done,
            }
        except Exception:
            pass

    return {
        "active": False,
        "pass": "unknown",
        "markets_done": 0,
        "markets_total": 0,
        "progress_pct": 0,
        "prices_total": 0,
        "prices_millions": 0,
        "db_size_gb": 0,
        "workers_alive": 0,
        "eta_hours": 0,
        "updated_at": "no data",
    }


@app.get("/api/weather-download-progress")
async def api_weather_download_progress(
    _user: str = Depends(_verify_credentials),
) -> dict[str, Any]:
    """Strategy C weather data download progress."""
    import json as _json

    status_path = Path("/opt/arbo/research/data/weather_status.json")
    if status_path.exists():
        try:
            data = _json.loads(status_path.read_text())
            done = data.get("markets_done", 0)
            total = data.get("markets_total", 1)
            pct = done / max(total, 1) * 100
            remaining = total - done
            if remaining <= 0:
                eta_min = 0
            else:
                eta_min = remaining / 100  # ~100 markets/min (10-min res, 5 workers)
            return {
                "active": done > 0 and done < total,
                "markets_done": done,
                "markets_total": total,
                "progress_pct": round(pct, 1),
                "prices_total": data.get("prices_total", 0),
                "prices_millions": round(data.get("prices_total", 0) / 1_000_000, 1),
                "db_size_mb": data.get("db_size_mb", 0),
                "cities_done": data.get("cities_done", 0),
                "cities_total": 20,
                "eta_hours": round(eta_min / 60, 1),
                "updated_at": data.get("updated_at", "?"),
            }
        except Exception:
            pass
    return {"active": False, "markets_done": 0, "markets_total": 0, "progress_pct": 0}


@app.get("/api/infra")
async def api_infra(_user: str = Depends(_verify_credentials)) -> dict[str, Any]:
    """Infrastructure: uptime, system resources, strategy task count."""
    orch = state.orchestrator
    uptime_s = int(time.monotonic() - orch._start_time) if orch and orch._start_time else 0

    # System resources
    process = psutil.Process()
    mem_info = process.memory_info()
    cpu_pct = process.cpu_percent(interval=0)

    # Odds API quota (available on legacy orchestrator, optional on RDH)
    odds_quota: dict[str, Any] = {"remaining": "N/A", "total": "N/A"}
    odds_client = getattr(orch, "_odds_client", None) if orch else None
    if odds_client is not None:
        remaining = getattr(odds_client, "_remaining_quota", None)
        if remaining is not None:
            odds_quota = {"remaining": remaining, "total": 20000}

    # Active strategy/task count
    task_dict = getattr(orch, "_tasks", {}) if orch else {}
    active_tasks = sum(1 for ts in task_dict.values() if ts.enabled and not ts.permanent_stop)

    return {
        "uptime_s": uptime_s,
        "uptime_human": _format_uptime(uptime_s),
        "memory_mb": round(mem_info.rss / 1024 / 1024, 1),
        "cpu_pct": round(cpu_pct, 1),
        "system_memory_pct": round(psutil.virtual_memory().percent, 1),
        "odds_api_quota": odds_quota,
        "active_tasks": active_tasks,
        "mode": getattr(orch, "_mode", "unknown") if orch else "unknown",
    }


@app.get("/api/trades")
async def api_trades(_user: str = Depends(_verify_credentials)) -> dict[str, Any]:
    """Last 50 trades with market names and category."""
    trades: list[dict[str, Any]] = []
    try:
        from arbo.utils.db import Market, PaperTrade, get_session_factory

        factory = get_session_factory()
        async with factory() as session:
            # Exclude pre-validation trades from trade log
            result = await session.execute(
                sa.select(PaperTrade, Market.question, Market.category)
                .outerjoin(Market, PaperTrade.market_condition_id == Market.condition_id)
                .where(
                    sa.or_(
                        PaperTrade.notes.is_(None),
                        PaperTrade.notes != "pre-validation",
                    )
                )
                .order_by(PaperTrade.placed_at.desc())
                .limit(50)
            )
            for row in result.all():
                trade = row[0]
                question = row[1] or trade.market_condition_id[:20]
                category = (row[2] or "other").capitalize()
                if category == "Other":
                    category = _infer_category(question)
                trades.append(
                    {
                        "id": trade.id,
                        "market": question,
                        "layer": trade.layer,
                        "strategy": getattr(trade, "strategy", "") or "",
                        "side": trade.side,
                        "price": _dec(trade.price),
                        "size": _dec(trade.size),
                        "edge": _dec(trade.edge_at_exec),
                        "confluence_score": trade.confluence_score,
                        "status": trade.status,
                        "actual_pnl": _dec(trade.actual_pnl),
                        "fee_paid": _dec(trade.fee_paid),
                        "category": category,
                        "placed_at": trade.placed_at.isoformat() if trade.placed_at else None,
                        "resolved_at": (
                            trade.resolved_at.isoformat() if trade.resolved_at else None
                        ),
                    }
                )
    except Exception as e:
        logger.warning("trades_query_error", error=str(e))

    return {"trades": trades}


@app.get("/api/closed-positions")
async def api_closed_positions(
    request: Request,
    _user: str = Depends(_verify_credentials),
) -> dict[str, Any]:
    """Closed (resolved) positions — won and lost trades.

    Query params:
      strategy: filter by strategy name (e.g. "B3", "B3_15M")
      live_only: "1" → return only trades with live_entry_shares > 0
                 (prevents live trades from being dropped by LIMIT when
                 paper trades dominate volume, e.g., B3 with 825 paper
                 since V6.0 but only 20 live fills)
      limit: max trades to return (default 500, capped 10000)
    """
    strategy_filter = request.query_params.get("strategy")
    live_only = request.query_params.get("live_only") == "1"
    try:
        limit = int(request.query_params.get("limit") or 500)
    except ValueError:
        limit = 500
    limit = max(1, min(limit, 10000))
    trades: list[dict[str, Any]] = []
    try:
        from arbo.utils.db import Market, PaperTrade, get_session_factory

        factory = get_session_factory()
        async with factory() as session:
            query = (
                sa.select(PaperTrade, Market.question, Market.category)
                .outerjoin(Market, PaperTrade.market_condition_id == Market.condition_id)
                .where(PaperTrade.status.in_(["won", "lost", "sold"]))
                .where(
                    sa.or_(
                        PaperTrade.notes.is_(None),
                        PaperTrade.notes != "pre-validation",
                    )
                )
            )
            if strategy_filter:
                query = query.where(PaperTrade.strategy == strategy_filter)
            if live_only:
                # Only trades with actual live position: live_entry_price set
                # and live_entry_shares > 0. Uses JSONB operators.
                query = query.where(
                    sa.and_(
                        PaperTrade.trade_details["live_entry_price"].isnot(None),
                        sa.cast(
                            PaperTrade.trade_details["live_entry_shares"].astext,
                            sa.Float,
                        ) > 0,
                    )
                )
            result = await session.execute(
                query.order_by(PaperTrade.resolved_at.desc()).limit(limit)
            )
            for row in result.all():
                trade = row[0]
                question = row[1] or trade.market_condition_id[:20]
                category = (row[2] or "other").capitalize()
                if category == "Other":
                    category = _infer_category(question)
                trades.append(
                    {
                        "id": trade.id,
                        "market": question,
                        "strategy": getattr(trade, "strategy", "") or "",
                        "side": trade.side,
                        "price": _dec(trade.price),
                        "fill_price": _dec(getattr(trade, "fill_price", None)),
                        "size": _dec(trade.size),
                        "edge": _dec(getattr(trade, "edge", None)),
                        "status": trade.status,
                        "actual_pnl": _dec(trade.actual_pnl),
                        "exit_price": _dec(getattr(trade, "exit_price", None)),
                        "exit_reason": getattr(trade, "exit_reason", None),
                        "trade_details": getattr(trade, "trade_details", None),
                        "category": category,
                        "placed_at": trade.placed_at.isoformat() if trade.placed_at else None,
                        "resolved_at": (
                            trade.resolved_at.isoformat() if trade.resolved_at else None
                        ),
                    }
                )
    except Exception as e:
        logger.warning("closed_positions_query_error", error=str(e))

    # Return both keys for backwards compatibility (JS uses "positions")
    return {"trades": trades, "positions": trades}


@app.get("/api/daily-pnl")
async def api_daily_pnl(_user: str = Depends(_verify_credentials)) -> dict[str, Any]:
    """Daily realized P&L aggregated by date, plus per-strategy daily averages."""
    days: list[dict[str, Any]] = []
    avg_by_strategy: dict[str, float] = {}
    try:
        from arbo.utils.db import PaperTrade, get_session_factory

        factory = get_session_factory()
        async with factory() as session:
            # Daily totals
            result = await session.execute(
                sa.select(
                    sa.func.date(PaperTrade.resolved_at).label("day"),
                    sa.func.sum(PaperTrade.actual_pnl).label("pnl"),
                    sa.func.count().label("trades"),
                )
                .where(PaperTrade.status.in_(["won", "lost", "sold"]))
                .where(PaperTrade.resolved_at.isnot(None))
                .group_by(sa.func.date(PaperTrade.resolved_at))
                .order_by(sa.func.date(PaperTrade.resolved_at))
            )
            for row in result.all():
                days.append(
                    {
                        "date": row.day.isoformat() if row.day else None,
                        "pnl": round(float(row.pnl or 0), 2),
                        "trades": row.trades,
                    }
                )

            # Per-strategy: total P&L and number of distinct days
            strat_result = await session.execute(
                sa.select(
                    PaperTrade.strategy,
                    sa.func.sum(PaperTrade.actual_pnl).label("total_pnl"),
                    sa.func.count(sa.func.distinct(sa.func.date(PaperTrade.resolved_at))).label(
                        "num_days"
                    ),
                )
                .where(PaperTrade.status.in_(["won", "lost", "sold"]))
                .where(PaperTrade.resolved_at.isnot(None))
                .where(PaperTrade.strategy.isnot(None))
                .group_by(PaperTrade.strategy)
            )
            for row in strat_result.all():
                strat = row.strategy or "?"
                total = float(row.total_pnl or 0)
                num_days = row.num_days or 1
                avg_by_strategy[strat] = round(total / num_days, 2)

    except Exception as e:
        logger.warning("daily_pnl_query_error", error=str(e))

    # Overall daily average
    num_total_days = len(days) or 1
    total_pnl = sum(d["pnl"] for d in days)
    avg_total = round(total_pnl / num_total_days, 2)

    return {
        "days": days,
        "avg_daily": avg_total,
        "avg_by_strategy": avg_by_strategy,
    }


@app.get("/api/strategies")
async def api_strategies(_user: str = Depends(_verify_credentials)) -> dict[str, Any]:
    """Per-strategy status: allocation, P&L, positions, halt state."""
    orch = state.orchestrator
    if orch is None:
        return {"strategies": []}

    rm = orch._risk_manager
    strategies: list[dict[str, Any]] = []

    for sid, meta in _STRATEGY_META.items():
        ss = rm.get_strategy_state(sid) if rm else None
        entry: dict[str, Any] = {
            "id": sid,
            "name": meta["name"],
            "category": meta["category"],
            "description": meta["description"],
        }
        if ss:
            entry.update(
                {
                    "allocated": _dec(ss.allocated),
                    "deployed": _dec(ss.deployed),
                    "available": _dec(ss.available),
                    "weekly_pnl": _dec(ss.weekly_pnl),
                    "total_pnl": _dec(ss.total_pnl),
                    "positions": ss.position_count,
                    "is_halted": ss.is_halted,
                }
            )
        else:
            entry.update(
                {
                    "allocated": 0.0,
                    "deployed": 0.0,
                    "available": 0.0,
                    "weekly_pnl": 0.0,
                    "total_pnl": 0.0,
                    "positions": 0,
                    "is_halted": False,
                }
            )
        # Add model metadata for Strategy C
        if sid == "C":
            entry["model"] = {
                "name": "C1f-ensemble",
                "score": 139.7,
                "train_trades": 219,
                "train_wr": 36.5,
                "oos_pnl": 1877,
                "oos_wr": 37.6,
                "walkforward_pnl": 1877,
                "max_drawdown_pct": 15.6,
                "cities_active": 19,
                "expected_daily_trades": "3-5",
                "min_edge": 0.10,
                "kelly_fraction": 0.25,
                "prob_sharpening": 1.10,
                "excluded_cities": ["Lucknow", "Wellington"],
                "ensemble_members": 31,
                "ensemble_source": "GEFS (NOAA S3)",
            }
        elif sid == "C2":
            entry["model"] = {
                "name": "EMOS-Exit-Fusion",
                "score": 138.1,
                "train_trades": 1878,
                "train_wr": 54.1,
                "oos_pnl": 3411,
                "oos_wr": 53.7,
                "walkforward_pnl": 3411,
                "max_drawdown_pct": 8.3,
                "cities_active": 15,
                "expected_daily_trades": "8-15",
                "min_edge": 0.03,
                "kelly_fraction": 0.25,
                "prob_sharpening": 0.85,
                "excluded_cities": ["São Paulo", "Tel Aviv", "Tokyo", "Lucknow"],
                "exit_type": "edge-based",
                "min_hold_edge": 0.05,
                "profit_target": 0.15,
                "emos_window": 21,
                "emos_method": "rolling_mae + ewma bias",
                "exit_slippage_pct": 6.0,
            }
        elif sid == "B2":
            entry["model"] = {
                "name": "Crypto-Price-Edge-v5",
                "score": 155.9,
                "train_trades": 902,
                "train_wr": 84.7,
                "oos_pnl": 1229,
                "oos_wr": 76.5,
                "walkforward_pnl": 1229,
                "max_drawdown_pct": 4.4,
                "assets": ["BTC", "ETH"],
                "expected_daily_trades": "5-6",
                "min_edge": 0.08,
                "kelly_fraction": 0.25,
                "prob_sharpening": 1.2,
                "excluded_assets": ["SOL", "XRP", "DOGE", "ADA", "BNB"],
                "exit_type": "edge-based (hold to resolution preferred)",
                "min_hold_edge": 0.03,
                "profit_target": 0.30,
                "volatility_window": 168,
                "volatility_method": "ewma",
                "sigma_scale": 0.8,
                "fee_model": "maker entry 0%, taker exit = p*(1-p)*0.25",
                "market_type": "daily above (NOT NegRisk)",
                "data_source": "Binance real-time price",
                "backtest_period": "87 days (2025-12-28 → 2026-03-26)",
                "backtest_data": "3,745 markets (BTC+ETH), 95M price points",
            }
        elif sid == "B3":
            entry["model"] = {
                "name": "Binance-Oracle-Scalper-v1",
                "oos_score": 6140,
                "train_trades": 7324,
                "train_wr": 57.4,
                "oos_pnl": 20285,
                "oos_wr": 52.1,
                "oos_sharpe": 22.0,
                "max_drawdown_pct": 33.1,
                "expected_daily_trades": "33",
                "entry_trigger": "|signal_fv - 0.50| > 0.095",
                "sigma_window": 720,
                "sigma_method": "realized",
                "sigma_scale": 0.644,
                "profit_target": 0.207,
                "stop_loss": 0.038,
                "max_hold_min": 3,
                "fee_model": "PostOnly maker 0% + 20% rebate",
                "market_type": "BTC 5-min Up/Down (NOT NegRisk)",
                "data_source": "Binance real-time price",
                "backtest_period": "89 days (2025-12-28 → 2026-03-27)",
                "backtest_data": "89,419 1-min klines, 17,883 windows",
            }
        elif sid == "B3_15M":
            entry["model"] = {
                "name": "B3-15m-Scalper-Shadow-Rank1",
                "source": "shadow_autoresearch_2026-04-12",
                "oos_trades": 49,
                "oos_pnl": 9.89,
                "oos_wr": 94.4,
                "folds": 5,
                "folds_all_positive": True,
                "std_avg_pnl": 0.103,
                "expected_daily_trades": "5-8",
                "entry_trigger": "|signal_fv - 0.50| > 0.089",
                "sigma_window": 1440,
                "sigma_method": "realized",
                "sigma_scale": 0.526,
                "min_edge": 0.30,
                "max_btc_move_usd": 80,
                "max_market_gap": 0.30,
                "max_fill_price": "uncapped (OPPOSITE of 5-min 0.75)",
                "entry_minutes": "4-11",
                "fee_model": "PostOnly maker 0% + 20% rebate",
                "market_type": "BTC 15-min Up/Down (NOT NegRisk)",
                "data_source": "Binance real-time + Chainlink RTDS",
                "live_capital": 75,
                "daily_loss_limit": 20,
                "max_bet_size": 20,
            }

        strategies.append(entry)

    return {"strategies": strategies}


@app.get("/api/capital")
async def api_capital(_user: str = Depends(_verify_credentials)) -> dict[str, Any]:
    """Capital allocation overview across strategies + reserve."""
    orch = state.orchestrator
    if orch is None:
        return {"error": "Orchestrator not available"}

    rm = orch._risk_manager
    total_capital = _dec(orch._capital) or 0.0

    allocations: list[dict[str, Any]] = []
    total_deployed = 0.0
    total_available = 0.0

    if rm:
        for sid, meta in _STRATEGY_META.items():
            ss = rm.get_strategy_state(sid)
            if ss:
                deployed = _dec(ss.deployed) or 0.0
                available = _dec(ss.available) or 0.0
                allocations.append(
                    {
                        "strategy": sid,
                        "name": meta["name"],
                        "allocated": _dec(ss.allocated),
                        "deployed": deployed,
                        "available": available,
                        "utilization_pct": round(
                            deployed / float(ss.allocated) * 100, 1
                        )
                        if ss.allocated
                        else 0.0,
                        "is_halted": ss.is_halted,
                    }
                )
                total_deployed += deployed
                total_available += available

    # Reserve (10% = $200)
    from arbo.core.risk_manager import RESERVE_CAPITAL

    reserve = _dec(RESERVE_CAPITAL) or 0.0

    return {
        "total_capital": total_capital,
        "total_deployed": round(total_deployed, 2),
        "total_available": round(total_available, 2),
        "reserve": reserve,
        "allocations": allocations,
    }


@app.get("/api/taker-flow")
async def api_taker_flow(_user: str = Depends(_verify_credentials)) -> dict[str, Any]:
    """Recent taker flow snapshots from DB (last 24h)."""
    snapshots: list[dict[str, Any]] = []
    try:
        from arbo.utils.db import TakerFlowSnapshot, get_session_factory

        factory = get_session_factory()
        async with factory() as session:
            cutoff = datetime.now(UTC) - timedelta(hours=24)
            result = await session.execute(
                sa.select(TakerFlowSnapshot)
                .where(TakerFlowSnapshot.captured_at > cutoff)
                .order_by(TakerFlowSnapshot.captured_at.desc())
                .limit(200)
            )
            for row in result.scalars().all():
                snapshots.append(
                    {
                        "market_condition_id": row.market_condition_id,
                        "yes_volume": _dec(row.yes_taker_volume),
                        "no_volume": _dec(row.no_taker_volume),
                        "yes_ratio": _dec(row.yes_no_ratio),
                        "z_score": _dec(row.z_score),
                        "is_peak": row.is_peak_optimism,
                        "captured_at": row.captured_at.isoformat() if row.captured_at else None,
                    }
                )
    except Exception as e:
        logger.warning("taker_flow_query_error", error=str(e))

    return {"snapshots": snapshots}


@app.get("/api/drift")
async def api_drift(_user: str = Depends(_verify_credentials)) -> dict[str, Any]:
    """Price drift analytics from Strategy C latency re-check."""
    orch = state.orchestrator
    if orch is None:
        return {"samples": 0, "avg_drift": 0, "avg_abs_drift": 0, "max_abs_drift": 0,
                "skipped_volatility": 0, "recent": []}

    sc = getattr(orch, "_strategy_c", None)
    if sc is None:
        return {"samples": 0, "avg_drift": 0, "avg_abs_drift": 0, "max_abs_drift": 0,
                "skipped_volatility": 0, "recent": []}

    log = sc.get_drift_log()
    stats = sc.stats
    # Last 50 entries for the chart
    recent = log[-50:] if log else []
    return {
        "samples": stats.get("drift_samples", 0),
        "avg_drift": stats.get("avg_drift", 0),
        "avg_abs_drift": stats.get("avg_abs_drift", 0),
        "max_abs_drift": stats.get("max_abs_drift", 0),
        "skipped_volatility": stats.get("trades_skipped_volatility", 0),
        "recent": recent,
    }


# ---------------------------------------------------------------------------
# CryptoArb API endpoints
# ---------------------------------------------------------------------------


@app.get("/api/cryptoarb/overview")
async def api_cryptoarb_overview(
    _user: str = Depends(_verify_credentials),
) -> dict[str, Any]:
    """Portfolio overview + all pair summaries."""
    data = _cryptoarb.read()
    if data is None:
        return {"error": "CryptoArb state not available"}

    # Compute win rate from trade returns
    trade_returns = data.get("trade_returns", [])
    wins = sum(1 for r in trade_returns if r > 0)
    win_rate = (wins / len(trade_returns) * 100) if trade_returns else 0.0

    # Compute drawdown from equity history
    equity_hist = data.get("equity_history", [])
    drawdown = 0.0
    if equity_hist:
        peak = equity_hist[0]
        for v in equity_hist:
            if v > peak:
                peak = v
            dd = (v - peak) / peak if peak > 0 else 0.0
            if dd < drawdown:
                drawdown = dd

    # Compute Sharpe estimate from trade returns
    import math

    sharpe = 0.0
    if len(trade_returns) > 1:
        mean_r = sum(trade_returns) / len(trade_returns)
        var_r = sum((r - mean_r) ** 2 for r in trade_returns) / (len(trade_returns) - 1)
        if var_r > 0:
            sharpe = mean_r / math.sqrt(var_r) * math.sqrt(365)

    pairs = {}
    for name, ps in data.get("pairs", {}).items():
        z_hist = ps.get("z_history", [])
        current_z = z_hist[-1] if z_hist else 0.0
        dz = (z_hist[-1] - z_hist[-4]) if len(z_hist) >= 4 else 0.0
        pos = ps.get("position", 0)
        pairs[name] = {
            "position": "LONG" if pos == 1 else ("SHORT" if pos == -1 else "FLAT"),
            "z": round(current_z, 4),
            "dz_3d": round(dz, 4),
            "allocation": ps.get("allocation", 0.0),
            "trade_pnl": round(ps.get("trade_pnl", 0.0), 6),
            "acf": round(ps.get("acf", 0.0), 3),
            "beta_stability": round(ps.get("beta_stability", 1.0), 3),
            "entry_z": ps.get("entry_z", 0.0),
            "entry_time": ps.get("entry_time", ""),
            "hold_count": ps.get("hold_count", 0),
        }

    return {
        "equity": data.get("portfolio_equity", 1.0),
        "drawdown_pct": round(drawdown * 100, 2),
        "total_trades": data.get("total_trades", 0),
        "win_rate": round(win_rate, 1),
        "sharpe": round(sharpe, 2),
        "last_daily_update": data.get("last_daily_update", ""),
        "last_signal_eval": data.get("last_signal_eval", ""),
        "stale": _cryptoarb.is_stale(),
        "mode": "paper",
        "pairs": pairs,
    }


@app.get("/api/cryptoarb/charts")
async def api_cryptoarb_charts(
    _user: str = Depends(_verify_credentials),
) -> dict[str, Any]:
    """Timestamped equity + Z-score series for Chart.js."""
    data = _cryptoarb.read()
    if data is None:
        return {"error": "CryptoArb state not available"}

    # Equity series
    equity_snapshots = data.get("equity_snapshots", [])
    if not equity_snapshots:
        # Fallback: index-based from equity_history
        equity_hist = data.get("equity_history", [])
        equity_snapshots = [{"t": str(i), "v": v} for i, v in enumerate(equity_hist[-365:])]

    # Z-score series per pair
    z_series: dict[str, list[dict]] = {}
    for name, ps in data.get("pairs", {}).items():
        snaps = ps.get("z_snapshots", [])
        if snaps:
            z_series[name] = snaps
        else:
            # Fallback: index-based from z_history
            z_hist = ps.get("z_history", [])
            z_series[name] = [{"t": str(i), "z": v} for i, v in enumerate(z_hist)]

    return {
        "equity": equity_snapshots,
        "z_scores": z_series,
    }


@app.get("/api/cryptoarb/trades")
async def api_cryptoarb_trades(
    _user: str = Depends(_verify_credentials),
) -> dict[str, Any]:
    """Trade log (most recent first)."""
    data = _cryptoarb.read()
    if data is None:
        return {"error": "CryptoArb state not available"}

    trade_log = data.get("trade_log", [])
    # Return most recent first
    return {"trades": list(reversed(trade_log))}


@app.get("/api/cryptoarb/alerts")
async def api_cryptoarb_alerts(
    _user: str = Depends(_verify_credentials),
) -> dict[str, Any]:
    """Alert history (most recent first)."""
    data = _cryptoarb.read()
    if data is None:
        return {"error": "CryptoArb state not available"}

    alerts = data.get("alerts", [])
    return {"alerts": list(reversed(alerts))}


# ---------------------------------------------------------------------------
# CryptoArb Expected vs Reality
# ---------------------------------------------------------------------------

# Run 3 backtest baseline (5.5y, 738 trades, composite 1.282)
_CRYPTOARB_BASELINE = {
    "model": "Run 3 (vol_scale + z_mom + max_hold)",
    "composite": 1.282,
    "trades_per_year": 133,
    "win_rate": 58.4,
    "annual_roi_pct": 73.7,
    "monthly_roi_pct": 6.1,
    "weekly_roi_pct": 1.42,
    "daily_roi_pct": 0.202,
    "sharpe": 1.27,
    "avg_dur_days": 9.3,
    "max_dd_pct": -24.3,
    "per_pair": {
        "SOL_ETH": {"sharpe": 1.06, "ann_ret": 15.4, "wr": 62.7, "trades_yr": 12.1},
        "BNB_ETH": {"sharpe": 1.77, "ann_ret": 15.0, "wr": 57.7, "trades_yr": 37.5},
        "ADA_ETH": {"sharpe": 0.90, "ann_ret": 11.0, "wr": 62.0, "trades_yr": 12.7},
        "ATOM_ETH": {"sharpe": 1.38, "ann_ret": 17.7, "wr": 49.3, "trades_yr": 19.9},
        "XRP_ETH": {"sharpe": 1.26, "ann_ret": 14.6, "wr": 60.2, "trades_yr": 15.7},
    },
}


@app.get("/api/cryptoarb/expected-vs-reality")
async def api_cryptoarb_expected_vs_reality(
    _user: str = Depends(_verify_credentials),
) -> dict[str, Any]:
    """Compare Run 3 backtest expectations to actual paper trading."""
    try:
        data = _cryptoarb.read()
        if data is None:
            return {
                "error": "CryptoArb state not available",
                "too_early": True,
                "comparison": [],
                "daily_series": [],
                "expected_line": [],
            }

        bl = _CRYPTOARB_BASELINE

        # Actuals from state
        trade_returns = data.get("trade_returns", [])
        total_trades = data.get("total_trades", 0)
        wins = sum(1 for r in trade_returns if r > 0)
        win_rate = (wins / len(trade_returns) * 100) if trade_returns else 0.0
        equity = data.get("portfolio_equity", 1.0)
        total_roi = (equity - 1.0) * 100

        # Days active
        last_eval = data.get("last_signal_eval", "")
        equity_snaps = data.get("equity_snapshots", [])
        days_active = 0.0
        start_date = None
        if equity_snaps:
            try:
                first_t = equity_snaps[0].get("t", "")
                start_date = datetime.fromisoformat(first_t.replace("Z", "+00:00"))
                days_active = (datetime.now(UTC) - start_date).total_seconds() / 86400
            except (ValueError, TypeError):
                pass

        # Daily/weekly/monthly ROI (annualized from actual)
        actual_daily = total_roi / days_active if days_active > 1 else 0.0
        actual_weekly = actual_daily * 7
        actual_monthly = actual_daily * 30

        # Trade pace
        expected_trades_so_far = bl["trades_per_year"] / 365 * days_active if days_active > 0 else 0
        trades_per_year_actual = total_trades / days_active * 365 if days_active > 1 else 0

        # Open positions
        open_count = sum(
            1 for ps in data.get("pairs", {}).values() if ps.get("position", 0) != 0
        )

        # Avg trade duration
        avg_dur = 0.0
        if total_trades > 0:
            total_hold = sum(
                ps.get("hold_count", 0)
                for ps in data.get("pairs", {}).values()
                if ps.get("position", 0) != 0
            )
            # rough: total_hold is for open trades; closed trade dur not stored
            avg_dur = 0.0  # will show "—" until enough data

        too_early = total_trades < 20

        def _status(actual_val: float, expected_val: float, higher_is_better: bool = True, threshold: float = 0.5) -> str:
            if too_early:
                return "too_early"
            if expected_val == 0:
                return "info"
            ratio = actual_val / expected_val if expected_val != 0 else 0
            if higher_is_better:
                return "ok" if ratio >= threshold else "warning"
            else:  # lower is better (e.g. drawdown)
                return "ok" if ratio <= (2.0 - threshold) else "warning"

        comparison = [
            {
                "metric": "Umisteno obchodu",
                "expected": f"~{expected_trades_so_far:.0f} za {days_active:.0f}d",
                "actual": str(total_trades),
                "status": "info",
            },
            {
                "metric": "Win Rate",
                "expected": f"{bl['win_rate']:.1f}%",
                "actual": f"{win_rate:.1f}%" if trade_returns else "—",
                "status": _status(win_rate, bl["win_rate"]),
            },
            {
                "metric": "Celkove P&L",
                "expected": f"+{bl['daily_roi_pct'] * days_active:.2f}%" if days_active > 0 else "—",
                "actual": f"{total_roi:+.2f}%",
                "status": _status(total_roi, bl["daily_roi_pct"] * days_active) if days_active > 1 else "too_early",
            },
            {
                "metric": "Open positions",
                "expected": "—",
                "actual": f"{open_count} ({open_count * 10}% deployed)",
                "status": "info",
            },
            {"metric": "_separator", "expected": "", "actual": "", "status": ""},
            {
                "metric": "Daily ROI",
                "expected": f"+{bl['daily_roi_pct']:.3f}%",
                "actual": f"{actual_daily:+.3f}%" if days_active > 1 else "—",
                "status": _status(actual_daily, bl["daily_roi_pct"]) if days_active > 1 else "too_early",
            },
            {
                "metric": "Weekly ROI",
                "expected": f"+{bl['weekly_roi_pct']:.2f}%",
                "actual": f"{actual_weekly:+.2f}%" if days_active > 7 else "—",
                "status": _status(actual_weekly, bl["weekly_roi_pct"]) if days_active > 7 else "too_early",
            },
            {
                "metric": "Monthly ROI",
                "expected": f"+{bl['monthly_roi_pct']:.1f}%",
                "actual": f"{actual_monthly:+.1f}%" if days_active > 30 else "—",
                "status": _status(actual_monthly, bl["monthly_roi_pct"]) if days_active > 30 else "too_early",
            },
            {"metric": "_separator", "expected": "", "actual": "", "status": ""},
            {
                "metric": "Sharpe",
                "expected": f"{bl['sharpe']:.2f}",
                "actual": "—",
                "status": "too_early",
            },
            {
                "metric": "Trades/rok",
                "expected": f"~{bl['trades_per_year']:.0f}",
                "actual": f"~{trades_per_year_actual:.0f}" if days_active > 7 else "—",
                "status": _status(trades_per_year_actual, bl["trades_per_year"], threshold=0.4) if days_active > 7 else "too_early",
            },
            {
                "metric": "Avg trade dur.",
                "expected": f"{bl['avg_dur_days']:.1f}d",
                "actual": "—",
                "status": "too_early",
            },
        ]

        # Build daily equity series for chart
        daily_series = []
        expected_line = []
        if equity_snaps:
            # Group snapshots by date, take last value per day
            from collections import OrderedDict

            by_date: OrderedDict[str, float] = OrderedDict()
            for snap in equity_snaps:
                t = snap.get("t", "")
                v = snap.get("v", 1.0)
                if t:
                    date_str = t[:10]
                    by_date[date_str] = v

            cumulative = 0.0
            day_num = 0
            daily_expected_pnl = bl["daily_roi_pct"] / 100  # fraction per day
            for date_str, eq_val in by_date.items():
                day_num += 1
                pnl = (eq_val - 1.0) * 100  # total ROI in %
                daily_series.append({
                    "date": date_str,
                    "cumulative": round(pnl, 3),
                })
                expected_line.append({
                    "date": date_str,
                    "expected_cumulative": round(daily_expected_pnl * day_num * 100, 3),
                })

        return {
            "baseline": bl,
            "too_early": too_early,
            "actual": {
                "total_trades": total_trades,
                "wins": wins,
                "win_rate": round(win_rate, 1),
                "equity": equity,
                "total_roi": round(total_roi, 3),
                "days_active": round(days_active, 1),
                "open_positions": open_count,
            },
            "comparison": comparison,
            "daily_series": daily_series,
            "expected_line": expected_line,
        }
    except Exception as e:
        logger.error("cryptoarb_evr_error", error=str(e))
        return {"error": str(e), "too_early": True, "comparison": [], "daily_series": [], "expected_line": []}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_uptime(seconds: int) -> str:
    """Format uptime seconds into human-readable string."""
    days, remainder = divmod(seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, _ = divmod(remainder, 60)
    parts = []
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    parts.append(f"{minutes}m")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Experiments Dashboard
# ---------------------------------------------------------------------------

_EXPERIMENTS_DIR = Path(__file__).parent.parent.parent / "research" / "data" / "experiments"


def _load_latest_sweep() -> dict[str, Any] | None:
    """Load the most recent sweep or autoresearch results JSON."""
    if not _EXPERIMENTS_DIR.exists():
        return None

    # Try autoresearch first (newer), then sweep
    ar_file = _EXPERIMENTS_DIR / "autoresearch_latest.json"
    if ar_file.exists():
        try:
            return _convert_autoresearch(ar_file)
        except Exception:
            pass

    sweep_files = sorted(_EXPERIMENTS_DIR.glob("sweep_*.json"), reverse=True)
    if not sweep_files:
        return None
    try:
        import json as _json
        with open(sweep_files[0]) as f:
            return _json.load(f)
    except Exception:
        return None


def _convert_autoresearch(path: Path) -> dict[str, Any]:
    """Convert autoresearch JSON to sweep-compatible format for the dashboard."""
    import json as _json
    with open(path) as f:
        ar = _json.load(f)

    meta = ar.get("meta", {})

    # Convert history to all_results format (compact — no params/city_results)
    all_results = []
    for h in ar.get("history", []):
        all_results.append({
            "experiment_id": h.get("id", ""),
            "score": h.get("score", 0),
            "trades": h.get("trades", 0),
            "win_rate": h.get("win_rate", 0),
            "total_pnl": h.get("pnl", 0),
            "roi_pct": h.get("roi_pct", 0),
            "max_drawdown_pct": h.get("dd", 0),
            "sharpe": h.get("sharpe", 0),
            "capital_utilization": h.get("util", 0),
            "avg_pnl_per_hour": h.get("pph", 0),
            "total_exits": h.get("exits", 0),
            "exit_saves": h.get("saves", 0),
            "exit_regrets": h.get("regrets", 0),
            "status": h.get("status", ""),
            "description": h.get("description", ""),
        })

    # Top results from enriched data (includes params, city_results, equity_curve, OOS)
    top_enriched = ar.get("top_enriched", [])
    if top_enriched:
        top_results = []
        for h in top_enriched:
            top_results.append({
                "experiment_id": h.get("id", ""),
                "score": h.get("score_with_oos", h.get("score", 0)),
                "score_train": h.get("score", 0),
                "trades": h.get("trades", 0),
                "win_rate": h.get("win_rate", 0),
                "total_pnl": h.get("pnl", 0),
                "roi_pct": h.get("roi_pct", 0),
                "max_drawdown_pct": h.get("dd", 0),
                "sharpe": h.get("sharpe", 0),
                "capital_utilization": h.get("util", 0),
                "avg_pnl_per_hour": h.get("pph", 0),
                "total_exits": h.get("exits", 0),
                "exit_saves": h.get("saves", 0),
                "exit_regrets": h.get("regrets", 0),
                "status": h.get("status", ""),
                "description": h.get("description", ""),
                "params": h.get("params"),
                "city_results": h.get("city_results"),
                "equity_curve": h.get("equity_curve"),
                "oos_pnl": h.get("oos_pnl"),
                "oos_trades": h.get("oos_trades"),
                "oos_win_rate": h.get("oos_win_rate"),
                "oos_roi_pct": h.get("oos_roi_pct"),
                "wf_oos_pnl": h.get("wf_oos_pnl"),
                "wf_score": h.get("wf_score"),
            })
    else:
        # Fallback: derive from all_results (no enrichment)
        sorted_results = sorted(all_results, key=lambda r: -r["score"])
        seen_scores: set[float] = set()
        top_results = []
        for r in sorted_results:
            key = round(r["score"], 1)
            if key not in seen_scores:
                seen_scores.add(key)
                top_results.append(r)
            if len(top_results) >= 20:
                break

    return {
        "meta": {
            "sweep_id": "autoresearch",
            "timestamp": meta.get("timestamp", ""),
            "total_trials": meta.get("total_experiments", len(all_results)),
            "phases": f"keeps={meta.get('keeps', 0)}, discards={meta.get('discards', 0)}",
            "quick_mode": False,
        },
        "top_results": top_results,
        "all_results": all_results,
        "best_params": ar.get("best_params"),
        "city_solo": ar.get("city_solo"),
    }


@app.get("/experiments", response_class=HTMLResponse)
async def experiments_page(
    request: Request, _user: str = Depends(_verify_credentials)
) -> HTMLResponse:
    """Serve the experiments dashboard page."""
    return templates.TemplateResponse("experiments.html", {"request": request})


@app.get("/api/experiments/results")
async def api_experiments_results(
    _user: str = Depends(_verify_credentials),
) -> dict[str, Any]:
    """Return latest sweep results for the experiments dashboard."""
    data = _load_latest_sweep()
    if data is None:
        return {"error": "No sweep results found. Run: python3 research/sweep_final.py"}
    return data


@app.get("/api/experiments/sweeps")
async def api_experiments_sweeps(
    _user: str = Depends(_verify_credentials),
) -> dict[str, Any]:
    """List all available sweep result files."""
    if not _EXPERIMENTS_DIR.exists():
        return {"sweeps": []}

    sweeps = []

    # Autoresearch (show first)
    ar_file = _EXPERIMENTS_DIR / "autoresearch_latest.json"
    if ar_file.exists():
        try:
            import json as _json
            with open(ar_file) as fh:
                meta = _json.load(fh).get("meta", {})
            sweeps.append({
                "filename": ar_file.name,
                "sweep_id": "autoresearch",
                "timestamp": meta.get("timestamp", ""),
                "total_trials": meta.get("total_experiments", 0),
                "quick_mode": False,
            })
        except Exception:
            sweeps.append({"filename": ar_file.name, "sweep_id": "autoresearch"})

    # Sweep files
    sweep_files = sorted(_EXPERIMENTS_DIR.glob("sweep_*.json"), reverse=True)
    for f in sweep_files:
        try:
            import json as _json
            with open(f) as fh:
                data = _json.load(fh)
            meta = data.get("meta", {})
            sweep_type = meta.get("sweep_type", "") or data.get("sweep_type", "")
            sweeps.append({
                "filename": f.name,
                "sweep_id": meta.get("sweep_id", f.stem),
                "sweep_type": sweep_type,
                "timestamp": meta.get("timestamp", ""),
                "total_trials": meta.get("total_trials", 0),
                "quick_mode": meta.get("quick_mode", False),
            })
        except Exception:
            sweeps.append({"filename": f.name, "sweep_id": f.stem})
    return {"sweeps": sweeps}


@app.get("/api/experiments/sweep/{sweep_id}")
async def api_experiments_sweep_by_id(
    sweep_id: str, _user: str = Depends(_verify_credentials),
) -> dict[str, Any]:
    """Load a specific sweep by ID."""
    if not _EXPERIMENTS_DIR.exists():
        return {"error": "No experiments directory"}

    # Autoresearch
    if sweep_id == "autoresearch":
        ar_file = _EXPERIMENTS_DIR / "autoresearch_latest.json"
        if ar_file.exists():
            try:
                return _convert_autoresearch(ar_file)
            except Exception as e:
                return {"error": f"Failed to load autoresearch: {e}"}
        return {"error": "Autoresearch results not found"}

    # Sweep files
    for f in _EXPERIMENTS_DIR.glob("sweep_*.json"):
        if sweep_id in f.stem:
            try:
                import json as _json
                with open(f) as fh:
                    return _json.load(fh)
            except Exception:
                return {"error": f"Failed to load {f.name}"}
    return {"error": f"Sweep {sweep_id} not found"}


@app.get("/api/paper-trades")
async def api_paper_trades(
    strategy: str = "C",
    status: str = "won,lost",
    _user: str = Depends(_verify_credentials),
) -> dict[str, Any]:
    """Export paper trades for experiment validation."""
    orch = state.orchestrator
    if orch is None:
        return {"error": "Orchestrator not available"}

    statuses = [s.strip() for s in status.split(",")]

    try:
        async with orch._db_engine.begin() as conn:
            rows = await conn.execute(
                sa.text(
                    "SELECT id, market_condition_id, token_id, side, price, "
                    "size, edge_at_exec, confluence_score, kelly_fraction, "
                    "status, actual_pnl, fee_paid, placed_at, resolved_at, "
                    "strategy, notes "
                    "FROM paper_trades "
                    "WHERE strategy = :strategy AND status = ANY(:statuses) "
                    "ORDER BY placed_at"
                ),
                {"strategy": strategy, "statuses": statuses},
            )
            trades = []
            for row in rows:
                trades.append({
                    "id": row.id,
                    "token_id": row.token_id,
                    "side": row.side,
                    "price": float(row.price) if row.price else None,
                    "size": float(row.size) if row.size else None,
                    "edge": float(row.edge_at_exec) if row.edge_at_exec else None,
                    "status": row.status,
                    "pnl": float(row.actual_pnl) if row.actual_pnl else None,
                    "placed_at": row.placed_at.isoformat() if row.placed_at else None,
                    "resolved_at": row.resolved_at.isoformat() if row.resolved_at else None,
                    "strategy": row.strategy,
                })
            return {"trades": trades, "count": len(trades)}
    except Exception as e:
        return {"error": str(e), "trades": [], "count": 0}


# ---------------------------------------------------------------------------
# Health Check, Expected vs Reality, Seasonality, City Volumes
# ---------------------------------------------------------------------------


@app.get("/api/health-checks")
async def api_health_checks(_user: str = Depends(_verify_credentials)) -> dict[str, Any]:
    """Last 14 health check results for timeline display."""
    try:
        from arbo.utils.db import HealthCheck, get_session_factory

        factory = get_session_factory()
        async with factory() as session:
            result = await session.execute(
                sa.select(
                    HealthCheck.id,
                    HealthCheck.check_at,
                    HealthCheck.verdict,
                    HealthCheck.window_hours,
                    HealthCheck.metrics,
                    HealthCheck.notes,
                )
                .order_by(HealthCheck.check_at.desc())
                .limit(14)
            )
            checks = []
            for row in result.all():
                checks.append({
                    "id": row[0],
                    "check_at": row[1].isoformat() if row[1] else None,
                    "verdict": row[2],
                    "window_hours": row[3],
                    "metrics": row[4],
                    "notes": row[5],
                })
            return {"checks": list(reversed(checks))}
    except Exception as e:
        return {"checks": [], "error": str(e)}


@app.get("/api/expected-vs-reality")
async def api_expected_vs_reality(
    _user: str = Depends(_verify_credentials),
) -> dict[str, Any]:
    """Compare backtest expectations to actual paper trading performance."""
    try:
        from arbo.core.health_check import get_expected_vs_reality

        return await get_expected_vs_reality()
    except Exception as e:
        return {"error": str(e), "too_early": True, "comparison": [], "actual": {}}


@app.get("/api/expected-vs-reality-c2")
async def api_expected_vs_reality_c2(
    _user: str = Depends(_verify_credentials),
) -> dict[str, Any]:
    """Compare C2 model expectations to actual paper trading performance."""
    try:
        from arbo.core.health_check import get_expected_vs_reality_c2

        return await get_expected_vs_reality_c2()
    except Exception as e:
        return {"error": str(e), "too_early": True, "comparison": [], "actual": {}}


@app.get("/api/city-performance-c2")
async def api_city_performance_c2(
    _user: str = Depends(_verify_credentials),
) -> dict[str, Any]:
    """Per-city Strategy C2 performance from resolved trades."""
    try:
        from arbo.utils.db import Market, PaperTrade, get_session_factory

        factory = get_session_factory()
        async with factory() as session:
            result = await session.execute(
                sa.select(
                    PaperTrade.trade_details["city"].astext,
                    Market.question,
                    PaperTrade.status,
                    PaperTrade.actual_pnl,
                    PaperTrade.edge_at_exec,
                )
                .outerjoin(Market, PaperTrade.market_condition_id == Market.condition_id)
                .where(PaperTrade.strategy == "C2")
                .where(PaperTrade.status.in_(["won", "lost", "sold"]))
                .where(sa.or_(PaperTrade.notes.is_(None), PaperTrade.notes != "pre-validation"))
            )
            from arbo.strategies.weather_scanner import parse_city

            city_stats: dict[str, dict] = {}
            for row in result.all():
                city_name = row[0]
                if not city_name:
                    question = row[1] or ""
                    city_obj = parse_city(question)
                    city_name = city_obj.value if city_obj else "unknown"
                if city_name not in city_stats:
                    city_stats[city_name] = {"wins": 0, "losses": 0, "pnl": 0.0, "edges": []}
                cs = city_stats[city_name]
                pnl_val = float(row[3] or 0)
                status = row[2]
                # For "sold" (early exit), win/loss is determined by P&L sign
                if status == "won" or (status == "sold" and pnl_val >= 0):
                    cs["wins"] += 1
                else:
                    cs["losses"] += 1
                cs["pnl"] += pnl_val
                if row[4] is not None:
                    cs["edges"].append(float(row[4]))

            cities = []
            for city_name, cs in sorted(city_stats.items(), key=lambda x: x[1]["pnl"], reverse=True):
                resolved = cs["wins"] + cs["losses"]
                cities.append({
                    "city": city_name,
                    "resolved": resolved,
                    "wins": cs["wins"],
                    "losses": cs["losses"],
                    "wr": round(cs["wins"] / resolved, 4) if resolved > 0 else None,
                    "pnl": round(cs["pnl"], 2),
                    "avg_edge": round(sum(cs["edges"]) / len(cs["edges"]), 4) if cs["edges"] else 0,
                })
            return {"cities": cities}
    except Exception as e:
        return {"cities": [], "error": str(e)}


@app.get("/api/variants")
async def api_variants(
    request: Request,
    _user: str = Depends(_verify_credentials),
) -> dict[str, Any]:
    """Variant Leaderboard data — Phase 1 (read-only, champion-only).

    Returns aggregate stats per variant for a strategy. Phase 1 only the
    champion has trades; challengers exist as YAML files but no orchestrator
    is wired to them yet.

    Spec: docs/VARIANT_LEADERBOARD_SPEC.md §3
    """
    strategy = request.query_params.get("strategy", "").strip()
    if not strategy:
        return {"variants": [], "error": "strategy query param required"}

    # Load declarative pool
    try:
        from arbo.core.variant_pool import load_variants
        pool = load_variants(strategy)
    except Exception as e:
        logger.warning("variant_pool_load_error", strategy=strategy, error=str(e))
        return {"variants": [], "error": f"pool load failed: {e}"}

    if not pool:
        return {"variants": [], "error": f"no variants defined for {strategy}"}

    # Phase 2C.E: load raw YAML flags (auto_generated, retired_at, etc.)
    raw_meta: dict[str, dict[str, Any]] = {}
    try:
        import yaml as _yaml
        from pathlib import Path as _Path
        repo_root = _Path(__file__).resolve().parents[2]
        sdir = repo_root / "arbo" / "config" / "variants" / strategy.lower()
        if sdir.is_dir():
            for f in sdir.glob("*.yaml"):
                try:
                    with open(f) as fp:
                        raw_meta[f.stem] = _yaml.safe_load(fp) or {}
                except Exception:
                    pass
    except Exception as _e:
        pass

    # Phase 2C.E: pending promotion candidates (Tier 1/2 with p_better>=0.65)
    pending_promotions: set[str] = set()
    try:
        from arbo.core.promotion_engine import PromotionEngine, MIN_P_BETTER_CEO
        cands = await PromotionEngine(strategy).evaluate()
        for c in cands:
            if c.tier in (1, 2) and not c.reject_reason and c.p_better >= MIN_P_BETTER_CEO:
                pending_promotions.add(c.challenger_id)
    except Exception as _e:
        pass

    # Aggregate stats per variant from paper_trades
    rows: list[dict[str, Any]] = []
    try:
        from arbo.utils.db import PaperTrade, get_session_factory
        factory = get_session_factory()
        async with factory() as session:
            for v in pool:
                # Live-only filter: trades where live actually filled (so PnL is real)
                # Match by trade_details.variant_id key.
                stmt = sa.text("""
                    SELECT
                        COUNT(*) AS n_total,
                        COUNT(*) FILTER (
                            WHERE trade_details->>'live_fill_status' IN ('filled','partial')
                              AND (trade_details->>'live_entry_shares')::float > 0
                        ) AS n_live,
                        COUNT(*) FILTER (
                            WHERE trade_details->>'live_fill_status' IN ('filled','partial')
                              AND (trade_details->>'live_entry_shares')::float > 0
                              AND (trade_details->>'live_exit_price')::float > 0.5
                        ) AS wins,
                        COUNT(*) FILTER (
                            WHERE trade_details->>'live_fill_status' IN ('filled','partial')
                              AND (trade_details->>'live_entry_shares')::float > 0
                              AND (trade_details->>'live_exit_price')::float <= 0.5
                              AND (trade_details->>'live_exit_status') IN ('resolution','filled','partial')
                        ) AS losses,
                        COALESCE(SUM(
                            ((trade_details->>'live_exit_price')::float
                             - (trade_details->>'live_entry_price')::float)
                            * (trade_details->>'live_entry_shares')::float
                        ) FILTER (
                            WHERE trade_details->>'live_fill_status' IN ('filled','partial')
                              AND (trade_details->>'live_entry_shares')::float > 0
                              AND trade_details->>'live_exit_price' IS NOT NULL
                              AND trade_details->>'live_exit_status' IN ('resolution','filled','partial')
                        ), 0) AS total_pnl
                    FROM paper_trades
                    WHERE strategy = :strat
                      AND trade_details->>'variant_id' = :vid
                """)
                result = await session.execute(
                    stmt, {"strat": strategy, "vid": v.variant_id}
                )
                row = result.first()
                n_total = int(row.n_total) if row else 0
                n_live = int(row.n_live) if row else 0
                wins = int(row.wins) if row else 0
                losses = int(row.losses) if row else 0
                resolved = wins + losses
                wr = (100.0 * wins / resolved) if resolved > 0 else None
                pnl = float(row.total_pnl) if row else 0.0

                # Phase 2A.5: shadow stats from shadow_variant_signals.
                # Challengers have no paper_trades — their performance is
                # inferred from counterfactual "would-have-filled" PnL.
                shadow_stmt = sa.text("""
                    SELECT
                        COUNT(*) AS n_signals,
                        COUNT(*) FILTER (WHERE qualified) AS n_qualified,
                        COUNT(*) FILTER (
                            WHERE qualified AND resolution_outcome IS NOT NULL
                        ) AS n_resolved,
                        COUNT(*) FILTER (
                            WHERE qualified AND would_pnl_per_share > 0
                        ) AS s_wins,
                        COUNT(*) FILTER (
                            WHERE qualified AND would_pnl_per_share < 0
                        ) AS s_losses,
                        COALESCE(SUM(would_pnl_per_share) FILTER (
                            WHERE qualified AND would_pnl_per_share IS NOT NULL
                        ), 0) AS shadow_pnl_per_share
                    FROM shadow_variant_signals
                    WHERE strategy = :strat
                      AND variant_id = :vid
                """)
                s_result = await session.execute(
                    shadow_stmt, {"strat": strategy, "vid": v.variant_id}
                )
                s_row = s_result.first()
                shadow_n_sig = int(s_row.n_signals) if s_row else 0
                shadow_n_qual = int(s_row.n_qualified) if s_row else 0
                shadow_n_resolved = int(s_row.n_resolved) if s_row else 0
                shadow_wins = int(s_row.s_wins) if s_row else 0
                shadow_losses = int(s_row.s_losses) if s_row else 0
                shadow_decided = shadow_wins + shadow_losses
                shadow_wr = (
                    100.0 * shadow_wins / shadow_decided
                    if shadow_decided > 0 else None
                )
                shadow_pnl = float(s_row.shadow_pnl_per_share) if s_row else 0.0

                # Capital allocation: 100% to champion in Phase 1 (no MAB yet)
                cap_pct = 100.0 if v.status == "champion" else 0.0

                meta = raw_meta.get(v.variant_id, {})
                rows.append({
                    "variant_id": v.variant_id,
                    "strategy": v.strategy,
                    "status": v.status,
                    "n_total": n_total,
                    "n_live": n_live,
                    "wins": wins,
                    "losses": losses,
                    "wr_pct": round(wr, 1) if wr is not None else None,
                    "total_pnl": round(pnl, 2),
                    "capital_pct": cap_pct,
                    "dsr": None,  # Phase 2: implement DSR computation
                    "drift_status": "ok",  # Phase 4: real drift detector state
                    "notes": v.notes,
                    "parent_variant": v.parent_variant,
                    "params_summary": _variant_params_diff(v, pool),
                    # Shadow counterfactual stats (challenger readiness signal)
                    "shadow_n_signals": shadow_n_sig,
                    "shadow_n_qualified": shadow_n_qual,
                    "shadow_n_resolved": shadow_n_resolved,
                    "shadow_wr_pct": round(shadow_wr, 1) if shadow_wr is not None else None,
                    "shadow_pnl_per_share": round(shadow_pnl, 4),
                    # Phase 2C.E: badges
                    "auto_generated": bool(meta.get("auto_generated", False)),
                    "promoted_by": meta.get("promoted_by"),
                    "retired_reason": meta.get("retired_reason"),
                    "pending_promotion": v.variant_id in pending_promotions,
                })
    except Exception as e:
        logger.warning("api_variants_query_error", strategy=strategy, error=str(e))
        return {"variants": rows, "error": str(e)}

    # Sort: champion first, then challengers by total_pnl desc, retired last
    def sort_key(r: dict) -> tuple:
        status_order = {"champion": 0, "live": 1, "incubate": 2, "challenger": 3, "shadow": 4, "retired": 9}
        return (status_order.get(r["status"], 8), -r["total_pnl"])
    rows.sort(key=sort_key)

    return {
        "strategy": strategy,
        "variants": rows,
        "active_count": sum(1 for r in rows if r["status"] != "retired"),
        "champion": next((r["variant_id"] for r in rows if r["status"] == "champion"), None),
        "phase": "A — read-only champion display",
    }


def _variant_params_diff(v: "Any", pool: "list") -> dict[str, str]:
    """Return params that differ from champion (for tooltip/drill-down)."""
    if v.status == "champion":
        return {}
    champ = next((p for p in pool if p.status == "champion"), None)
    if champ is None:
        return {}
    diffs = {}
    for k, val in v.params.items():
        cval = champ.params.get(k)
        if cval != val:
            diffs[k] = f"{cval} → {val}"
    return diffs


@app.get("/api/expected-vs-reality-b3")
async def api_expected_vs_reality_b3(
    _user: str = Depends(_verify_credentials),
) -> dict[str, Any]:
    """Compare B3 backtest expectations to actual paper trading."""
    try:
        from arbo.utils.db import PaperTrade, get_session_factory

        factory = get_session_factory()
        async with factory() as session:
            result = await session.execute(
                sa.select(
                    PaperTrade.actual_pnl,
                    PaperTrade.status,
                    PaperTrade.exit_reason,
                    PaperTrade.placed_at,
                    PaperTrade.resolved_at,
                    PaperTrade.size,
                    PaperTrade.trade_details["direction"].astext,
                    PaperTrade.trade_details["btc_chainlink"].astext,
                    PaperTrade.trade_details["btc_binance_chainlink_delta"].astext,
                )
                .where(PaperTrade.strategy == "B3")
                .where(PaperTrade.status.in_(["won", "lost", "sold"]))
                .order_by(PaperTrade.placed_at)
            )
            rows = result.all()

            if not rows:
                return {"too_early": True, "actual": {}, "expected": {}, "daily_series": []}

            # Compute actual stats
            total_pnl = 0.0
            wins = 0
            losses = 0
            stop_count = 0
            profit_count = 0
            resolution_count = 0
            time_count = 0
            edge_count = 0
            chainlink_deltas: list[float] = []
            daily_pnl: dict[str, float] = {}
            win_pnls: list[float] = []
            loss_pnls: list[float] = []

            first_trade = None
            last_trade = None

            for row in rows:
                pnl = float(row[0] or 0)
                total_pnl += pnl
                exit_reason = row[2] or ""

                if pnl > 0:
                    wins += 1
                    win_pnls.append(pnl)
                else:
                    losses += 1
                    loss_pnls.append(abs(pnl))

                if exit_reason == "stop":
                    stop_count += 1
                elif exit_reason == "profit":
                    profit_count += 1
                elif exit_reason == "resolution":
                    resolution_count += 1
                elif exit_reason == "time":
                    time_count += 1
                elif exit_reason == "edge_gone":
                    edge_count += 1

                if row[7]:  # btc_chainlink
                    try:
                        chainlink_deltas.append(float(row[8] or 0))
                    except (ValueError, TypeError):
                        pass

                if row[3]:  # placed_at
                    if first_trade is None:
                        first_trade = row[3]
                    last_trade = row[3]
                    day = row[3].strftime("%Y-%m-%d")
                    daily_pnl[day] = daily_pnl.get(day, 0) + pnl

            total_resolved = wins + losses
            days_active = len(daily_pnl)
            wr = wins / total_resolved if total_resolved > 0 else 0
            avg_win = sum(win_pnls) / len(win_pnls) if win_pnls else 0
            avg_loss = sum(loss_pnls) / len(loss_pnls) if loss_pnls else 0
            pnl_per_day = total_pnl / days_active if days_active > 0 else 0
            trades_per_day = total_resolved / days_active if days_active > 0 else 0

            # Build daily cumulative series
            cum = 0.0
            daily_series = []
            for day in sorted(daily_pnl.keys()):
                cum += daily_pnl[day]
                daily_series.append({"date": day, "pnl": round(daily_pnl[day], 2), "cumulative": round(cum, 2)})

            # Backtest expectations (OOS config #1)
            expected = {
                "win_rate": 0.521,
                "avg_pnl_per_trade": 6.09,
                "avg_win": 20.70,
                "avg_loss": 9.79,
                "trades_per_day": 33,
                "pnl_per_day": 202,
                "sharpe": 22.0,
                "max_dd_pct": 33.1,
                "stop_loss_rate": 0.48,
            }

            actual = {
                "total_resolved": total_resolved,
                "wins": wins,
                "losses": losses,
                "win_rate": round(wr, 4),
                "total_pnl": round(total_pnl, 2),
                "avg_pnl_per_trade": round(total_pnl / total_resolved, 2) if total_resolved else 0,
                "avg_win": round(avg_win, 2),
                "avg_loss": round(avg_loss, 2),
                "trades_per_day": round(trades_per_day, 1),
                "pnl_per_day": round(pnl_per_day, 2),
                "days_active": days_active,
                "exit_reasons": {
                    "profit": profit_count,
                    "stop": stop_count,
                    "resolution": resolution_count,
                    "time": time_count,
                    "edge_gone": edge_count,
                },
                "chainlink_avg_delta": round(sum(chainlink_deltas) / len(chainlink_deltas), 2) if chainlink_deltas else None,
                "chainlink_trades": len(chainlink_deltas),
            }

            return {
                "expected": expected,
                "actual": actual,
                "daily_series": daily_series,
                "too_early": total_resolved < 10,
            }
    except Exception as e:
        return {"error": str(e), "too_early": True, "actual": {}, "expected": {}}


@app.get("/api/polymarket-wallet")
async def api_polymarket_wallet(
    _user: str = Depends(_verify_credentials),
) -> dict[str, Any]:
    """Live Polymarket wallet: balance, positions, P&L."""
    import json as _json
    import os
    import ssl
    import urllib.request

    funder = os.getenv("POLY_FUNDER_ADDRESS", "")
    result: dict[str, Any] = {"balance": 0, "positions": [], "total_value": 0}

    # Get positions from Data API
    try:
        _ssl = ssl.create_default_context()
        try:
            import certifi
            _ssl = ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            pass
        url = f"https://data-api.polymarket.com/positions?user={funder}"
        req = urllib.request.Request(url, headers={"User-Agent": "Arbo/1.0"})
        import asyncio
        data = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: _json.loads(urllib.request.urlopen(req, timeout=10, context=_ssl).read()),
        )
        positions = []
        for pos in data:
            size = float(pos.get("size", 0))
            if size <= 0:
                continue
            cur_val = float(pos.get("currentValue", 0))
            # Skip dead loser tokens. IMPORTANT: Polymarket's `redeemable` flag
            # is TRUE for both winners AND losers (it means "market resolved,
            # redeem tx is possible"). Not an indicator of payout > 0.
            # The real discriminator is currentValue = size × curPrice:
            #   - Winner: curPrice=1 → currentValue = size (full value)
            #   - Loser:  curPrice=0 → currentValue = 0 (dust, skip)
            #   - Active: 0 < curPrice < 1 → currentValue partial
            if cur_val < 0.10:
                continue
            avg_price = float(pos.get("avgPrice", 0))
            positions.append({
                "title": pos.get("title", ""),
                "outcome": pos.get("outcome", ""),
                "size": size,
                "avg_price": round(avg_price, 3),
                "current_value": round(cur_val, 2),
                "pnl": round(cur_val - size * avg_price, 2),
                "redeemable": bool(pos.get("redeemable", False)),
                "asset": str(pos.get("asset") or ""),
                "condition_id": pos.get("conditionId", ""),
            })
        result["positions"] = positions
        result["total_value"] = round(sum(p["current_value"] for p in positions), 2)
    except Exception as e:
        result["positions_error"] = str(e)

    # Get USDC balance from CLOB
    try:
        from py_clob_client.client import ClobClient
        from py_clob_client.clob_types import AssetType, BalanceAllowanceParams
        import asyncio

        def _get_bal():
            c = ClobClient(
                host="https://clob.polymarket.com", chain_id=137,
                key=os.getenv("POLY_PRIVATE_KEY", ""), signature_type=2,
                funder=funder or None,
            )
            creds = c.create_or_derive_api_creds()
            c.set_api_creds(creds)
            r = c.get_balance_allowance(
                BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
            )
            return int(r.get("balance", 0)) / 1_000_000

        result["balance"] = await asyncio.get_event_loop().run_in_executor(None, _get_bal)
        result["portfolio"] = round(result["balance"] + result["total_value"], 2)
        result["starting_capital"] = float(os.getenv("B3_LIVE_STARTING_CAPITAL", "300"))
    except Exception as e:
        result["balance_error"] = str(e)

    return result


@app.get("/api/strategy-d")
async def api_strategy_d(_user: str = Depends(_verify_credentials)) -> dict[str, Any]:
    """Strategy D status across all sport variants (NBA, UFC, EPL) + CLV tracking.

    CLV = Closing Line Value. Measures entry price vs market's final price.
    Positive CLV = entry was cheaper than close → real edge.
    Research: CLV > 2¢ over 500 trades = long-term profitable strategy.
    """
    orch = state.orchestrator
    if orch is None:
        return {"active": False, "variants": {}, "aggregated": {}}

    # Per-variant status (from in-memory positions)
    variants: dict[str, Any] = {}
    for attr, sid in [
        ("_strategy_d", "D"),
        ("_strategy_d_ufc", "D_UFC"),
        ("_strategy_d_epl", "D_EPL"),
    ]:
        strat = getattr(orch, attr, None)
        if strat is None:
            continue
        variants[sid] = strat.get_status()

    # Aggregated trades from DB across all D variants
    recent_trades: list[dict[str, Any]] = []
    per_strategy: dict[str, dict[str, Any]] = {}
    total_trades = 0
    wins = 0
    green_books = 0
    total_pnl = 0.0
    clv_values: list[float] = []

    try:
        from arbo.utils.db import PaperTrade, get_session_factory
        from sqlalchemy import desc, select, or_
        factory = get_session_factory()
        async with factory() as session:
            result = await session.execute(
                select(PaperTrade)
                .where(or_(
                    PaperTrade.strategy == "D",
                    PaperTrade.strategy == "D_UFC",
                    PaperTrade.strategy == "D_EPL",
                ))
                .order_by(desc(PaperTrade.placed_at)).limit(200)
            )
            trades = result.scalars().all()
            total_trades = len(trades)
            for t in trades:
                tp = float(_dec(t.actual_pnl) or 0)
                total_pnl += tp
                if tp > 0:
                    wins += 1

                details = getattr(t, "trade_details", None) or {}
                exit_reason = t.exit_reason or (details.get("exit_reason", "") if isinstance(details, dict) else "")
                if exit_reason == "green_book":
                    green_books += 1

                # Extract CLV from trade_details if stored
                clv = details.get("clv", None) if isinstance(details, dict) else None
                if clv is not None:
                    clv_values.append(float(clv))

                # Per-strategy breakdown
                sname = t.strategy or "D"
                if sname not in per_strategy:
                    per_strategy[sname] = {"trades": 0, "pnl": 0.0, "wins": 0, "gb": 0, "clv_sum": 0.0, "clv_n": 0}
                ps = per_strategy[sname]
                ps["trades"] += 1
                ps["pnl"] += tp
                if tp > 0: ps["wins"] += 1
                if exit_reason == "green_book": ps["gb"] += 1
                if clv is not None:
                    ps["clv_sum"] += float(clv)
                    ps["clv_n"] += 1

                if len(recent_trades) < 30:
                    recent_trades.append({
                        "time": t.placed_at.isoformat() if t.placed_at else "",
                        "strategy": sname,
                        "market": (details.get("question", "") if isinstance(details, dict) else "")[:60],
                        "side": details.get("side", "") if isinstance(details, dict) else t.side,
                        "entry_price": _dec(t.price),
                        "exit_price": _dec(t.exit_price),
                        "exit_reason": exit_reason,
                        "pnl": tp,
                        "clv": float(clv) if clv is not None else None,
                    })
    except Exception as e:
        logger.warning("strategy_d_trades_query_failed: %s", e)

    # Finalize per-strategy avg CLV
    for sname, ps in per_strategy.items():
        ps["win_rate"] = round(ps["wins"] / max(ps["trades"], 1), 3)
        ps["gb_rate"] = round(ps["gb"] / max(ps["trades"], 1), 3)
        ps["avg_clv"] = round(ps["clv_sum"] / max(ps["clv_n"], 1), 4) if ps["clv_n"] > 0 else None
        ps["pnl"] = round(ps["pnl"], 2)
        del ps["clv_sum"]
        del ps["clv_n"]

    avg_clv = (sum(clv_values) / len(clv_values)) if clv_values else None

    # CLV health status
    clv_health = "no_data"
    if len(clv_values) >= 50:
        if avg_clv >= 0.02:
            clv_health = "excellent"   # >2¢ avg CLV
        elif avg_clv >= 0.01:
            clv_health = "good"
        elif avg_clv >= 0.0:
            clv_health = "marginal"
        else:
            clv_health = "warning"      # Negative CLV = losing edge

    return {
        "active": True,
        "variants": variants,
        "aggregated": {
            "pnl": round(total_pnl, 2),
            "trades": total_trades,
            "win_rate": round(wins / total_trades, 3) if total_trades > 0 else 0.0,
            "gb_rate": round(green_books / total_trades, 3) if total_trades > 0 else 0.0,
            "avg_clv": round(avg_clv, 4) if avg_clv is not None else None,
            "avg_clv_cents": round(avg_clv * 100, 2) if avg_clv is not None else None,
            "clv_sample_size": len(clv_values),
            "clv_health": clv_health,
        },
        "per_strategy": per_strategy,
        "recent_trades": recent_trades,
        # Backward-compat fields (for existing JS)
        "pnl": round(total_pnl, 2),
        "trades": total_trades,
        "win_rate": round(wins / total_trades, 3) if total_trades > 0 else 0.0,
        "gb_rate": round(green_books / total_trades, 3) if total_trades > 0 else 0.0,
        "open_positions": sum(v.get("open_positions", 0) for v in variants.values()),
        "trades_today": sum(v.get("trades_today", 0) for v in variants.values()),
        "daily_pnl": sum(v.get("daily_pnl", 0) for v in variants.values()),
        "params": variants.get("D", {}).get("params", {}),
        "positions": [p for v in variants.values() for p in v.get("positions", [])],
    }


@app.get("/api/expected-vs-reality-b2")
async def api_expected_vs_reality_b2(
    _user: str = Depends(_verify_credentials),
) -> dict[str, Any]:
    """Compare B2 backtest expectations to actual paper trading.

    Breaks down by live_entry_shares:
      - `actual` — all B2 trades (paper + live-filled)
      - `actual_live` — subset where live_entry_shares > 0 (real CLOB fills
        in dual mode). Paper side mirrors the real fill so PnL reflects
        what live capital actually earned.
      - `actual_paper_only` — paper-only trades (live skipped or pre-deploy)
    """
    try:
        from arbo.utils.db import PaperTrade, get_session_factory

        factory = get_session_factory()
        async with factory() as session:
            result = await session.execute(
                sa.select(
                    PaperTrade.actual_pnl,
                    PaperTrade.status,
                    PaperTrade.exit_reason,
                    PaperTrade.placed_at,
                    PaperTrade.resolved_at,
                    PaperTrade.size,
                    PaperTrade.trade_details["asset"].astext,
                    PaperTrade.trade_details["strike"].astext,
                    PaperTrade.trade_details["direction"].astext,
                    PaperTrade.trade_details["live_entry_shares"].astext,
                    PaperTrade.trade_details["paper_match_live"].astext,
                    PaperTrade.trade_details["live_exit_status"].astext,
                )
                .where(PaperTrade.strategy == "B2")
                .where(PaperTrade.status.in_(["won", "lost", "sold"]))
                # Exclude archived pre-reset trades — dashboard shows only
                # data collected after the 2026-04-16 reset. The pre-reset
                # rows are preserved in DB for audit but don't pollute the
                # clean-slate Live counter. Same filter on restore query.
                .where(sa.or_(PaperTrade.notes.is_(None), ~PaperTrade.notes.ilike("%pre_reset%")))
                .order_by(PaperTrade.placed_at)
            )
            rows = result.all()

            if not rows:
                return {"too_early": True, "actual": {}, "expected": {}, "daily_series": []}

            total_pnl = 0.0
            wins = 0
            losses = 0
            resolution_count = 0
            edge_count = 0
            profit_count = 0
            daily_pnl: dict[str, float] = {}
            # Per-segment daily for live-only and paper-only lines on charts
            daily_pnl_live: dict[str, float] = {}
            daily_pnl_paper: dict[str, float] = {}
            win_pnls: list[float] = []
            loss_pnls: list[float] = []
            # Live-filled subset (dual mode, real CLOB fills)
            live_pnl = 0.0
            live_wins = 0
            live_losses = 0
            live_count = 0
            # Per-trade time series (resolved_at, pnl) for the per-trade
            # cumulative chart — split by live/paper.
            trade_series_live: list[tuple[object, float]] = []
            trade_series_paper: list[tuple[object, float]] = []

            # Live classification requires BOTH legs to have happened on CLOB.
            # - Entry: live_entry_shares > 0 (position actually filled on-chain)
            # - Exit: live_exit_status recorded as a real settlement outcome
            # Trades where paper closed but the live leg never executed (e.g.
            # ExitManager not yet synced → paper-only path taken) count as
            # paper, not live. Otherwise we'd attribute paper's phantom PnL
            # to real capital and mismatch the Slack cumulative tracker,
            # which only counts actual _notify_b2_live_resolve events.
            live_exit_terminal = {"resolution", "filled", "partial", "maker", "taker", "maker+taker"}
            for row in rows:
                pnl = float(row[0] or 0)
                total_pnl += pnl
                exit_reason = row[2] or ""
                live_shares_raw = row[9]
                live_exit_status = (row[11] or "").lower() if len(row) > 11 else ""
                try:
                    live_shares = int(live_shares_raw) if live_shares_raw else 0
                except (ValueError, TypeError):
                    live_shares = 0
                # Count as Live only when BOTH legs executed on CLOB.
                # Paper-path phantom closes (live_exit_status empty) stay
                # with paper-only so Live PnL matches the Slack tracker.
                is_live = live_shares > 0 and live_exit_status in live_exit_terminal

                if pnl > 0:
                    wins += 1
                    win_pnls.append(pnl)
                else:
                    losses += 1
                    loss_pnls.append(abs(pnl))

                if is_live:
                    live_pnl += pnl
                    live_count += 1
                    if pnl > 0:
                        live_wins += 1
                    else:
                        live_losses += 1

                if exit_reason == "resolution":
                    resolution_count += 1
                elif exit_reason == "edge_lost":
                    edge_count += 1
                elif exit_reason == "profit_take":
                    profit_count += 1

                if row[3]:
                    day = row[3].strftime("%Y-%m-%d")
                    daily_pnl[day] = daily_pnl.get(day, 0) + pnl
                    if is_live:
                        daily_pnl_live[day] = daily_pnl_live.get(day, 0) + pnl
                    else:
                        daily_pnl_paper[day] = daily_pnl_paper.get(day, 0) + pnl

                # Per-trade timeline keyed by resolution time (row[4]) —
                # falls back to placed_at (row[3]) if unresolved. Skip rows
                # with no timestamp at all.
                ts = row[4] or row[3]
                if ts is not None:
                    if is_live:
                        trade_series_live.append((ts, pnl))
                    else:
                        trade_series_paper.append((ts, pnl))

            total_resolved = wins + losses
            days_active = len(daily_pnl)
            wr = wins / total_resolved if total_resolved > 0 else 0
            avg_win = sum(win_pnls) / len(win_pnls) if win_pnls else 0
            avg_loss = sum(loss_pnls) / len(loss_pnls) if loss_pnls else 0
            pnl_per_day = total_pnl / days_active if days_active > 0 else 0

            cum = 0.0
            cum_live = 0.0
            cum_paper = 0.0
            daily_series = []
            for day in sorted(daily_pnl.keys()):
                cum += daily_pnl[day]
                cum_live += daily_pnl_live.get(day, 0)
                cum_paper += daily_pnl_paper.get(day, 0)
                daily_series.append({
                    "date": day,
                    "pnl": round(daily_pnl[day], 2),
                    "cumulative": round(cum, 2),
                    "pnl_live": round(daily_pnl_live.get(day, 0), 2),
                    "cumulative_live": round(cum_live, 2),
                    "pnl_paper": round(daily_pnl_paper.get(day, 0), 2),
                    "cumulative_paper": round(cum_paper, 2),
                })

            # Per-trade cumulative series (chronological, split by segment).
            def _build_trade_cum(items: list) -> list[dict]:
                items_sorted = sorted(items, key=lambda x: x[0])
                out: list[dict] = []
                running = 0.0
                for ts, pnl in items_sorted:
                    running += pnl
                    out.append({
                        "ts": ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
                        "pnl": round(pnl, 4),
                        "cumulative": round(running, 2),
                    })
                return out
            trade_series_live_out = _build_trade_cum(trade_series_live)
            trade_series_paper_out = _build_trade_cum(trade_series_paper)

            expected = {
                "win_rate": 0.847,
                "avg_pnl_per_trade": 30.0,
                "trades_per_day": 10,
                "pnl_per_day": 300,
                "sharpe": 7.95,
                "max_dd_pct": 4.4,
            }

            actual = {
                "total_resolved": total_resolved,
                "wins": wins,
                "losses": losses,
                "win_rate": round(wr, 4),
                "total_pnl": round(total_pnl, 2),
                "avg_pnl_per_trade": round(total_pnl / total_resolved, 2) if total_resolved else 0,
                "avg_win": round(avg_win, 2),
                "avg_loss": round(avg_loss, 2),
                "trades_per_day": round(total_resolved / days_active, 1) if days_active else 0,
                "pnl_per_day": round(pnl_per_day, 2),
                "days_active": days_active,
                "exit_reasons": {
                    "resolution": resolution_count,
                    "edge_lost": edge_count,
                    "profit_take": profit_count,
                },
            }

            live_resolved = live_wins + live_losses
            actual_live = {
                "total_resolved": live_resolved,
                "wins": live_wins,
                "losses": live_losses,
                "win_rate": round(live_wins / live_resolved, 4) if live_resolved else 0,
                "total_pnl": round(live_pnl, 2),
                "avg_pnl_per_trade": round(live_pnl / live_resolved, 2) if live_resolved else 0,
                "count": live_count,
            }
            paper_resolved = total_resolved - live_resolved
            paper_pnl = total_pnl - live_pnl
            actual_paper_only = {
                "total_resolved": paper_resolved,
                "total_pnl": round(paper_pnl, 2),
                "avg_pnl_per_trade": round(paper_pnl / paper_resolved, 2) if paper_resolved else 0,
            }

            return {
                "expected": expected,
                "actual": actual,
                "actual_live": actual_live,
                "actual_paper_only": actual_paper_only,
                "daily_series": daily_series,
                "trade_series_live": trade_series_live_out,
                "trade_series_paper": trade_series_paper_out,
                "too_early": total_resolved < 5,
            }
    except Exception as e:
        return {"error": str(e), "too_early": True, "actual": {}, "expected": {}}


@app.get("/api/seasonality")
async def api_seasonality(_user: str = Depends(_verify_credentials)) -> dict[str, Any]:
    """Seasonality analysis for Strategy C."""
    try:
        from arbo.core.health_check import get_seasonality_analysis

        return await get_seasonality_analysis()
    except Exception as e:
        return {"error": str(e), "monthly_actual": []}


@app.get("/api/city-volumes")
async def api_city_volumes(_user: str = Depends(_verify_credentials)) -> dict[str, Any]:
    """Per-city volume stats: current, 7d average, max position size."""
    try:
        from arbo.utils.db import CityVolumeDaily, get_session_factory

        factory = get_session_factory()
        async with factory() as session:
            # Last 7 days of volume data per city
            seven_days_ago = datetime.now(UTC) - timedelta(days=7)
            result = await session.execute(
                sa.select(
                    CityVolumeDaily.city,
                    CityVolumeDaily.date,
                    CityVolumeDaily.volume_24h,
                    CityVolumeDaily.liquidity,
                    CityVolumeDaily.num_markets,
                    CityVolumeDaily.avg_price,
                )
                .where(CityVolumeDaily.date >= seven_days_ago.date())
                .order_by(CityVolumeDaily.city, CityVolumeDaily.date)
            )

            from collections import defaultdict

            city_data: dict[str, list[dict]] = defaultdict(list)
            for row in result.all():
                city_data[row[0]].append({
                    "date": row[1].isoformat() if row[1] else None,
                    "volume_24h": _dec(row[2]),
                    "liquidity": _dec(row[3]),
                    "num_markets": row[4],
                    "avg_price": row[5],
                })

            # Compute summary per city
            cities: list[dict] = []
            for city_name, days in sorted(city_data.items()):
                volumes = [d["volume_24h"] for d in days if d["volume_24h"]]
                avg_vol = sum(volumes) / len(volumes) if volumes else 0
                latest = days[-1] if days else {}
                today_vol = latest.get("volume_24h", 0) or 0

                # Max position = 5% of avg 24h volume
                max_position = round(avg_vol * 0.05, 2) if avg_vol > 0 else 0

                # Trend: compare today vs 7d average
                trend = "stable"
                if avg_vol > 0 and today_vol > 0:
                    ratio = today_vol / avg_vol
                    if ratio > 1.3:
                        trend = "up"
                    elif ratio < 0.7:
                        trend = "down"

                cities.append({
                    "city": city_name,
                    "today_volume": round(today_vol, 2),
                    "avg_7d_volume": round(avg_vol, 2),
                    "today_liquidity": latest.get("liquidity", 0),
                    "num_markets": latest.get("num_markets", 0),
                    "max_position": max_position,
                    "trend": trend,
                    "days": days,
                })

            return {"cities": cities}
    except Exception as e:
        return {"cities": [], "error": str(e)}


@app.get("/api/city-performance")
async def api_city_performance(_user: str = Depends(_verify_credentials)) -> dict[str, Any]:
    """Per-city Strategy C performance from resolved trades."""
    try:
        from arbo.utils.db import Market, PaperTrade, get_session_factory

        factory = get_session_factory()
        async with factory() as session:
            # Get resolved C trades with market question for city parsing
            result = await session.execute(
                sa.select(
                    PaperTrade.trade_details["city"].astext,
                    Market.question,
                    PaperTrade.status,
                    PaperTrade.actual_pnl,
                    PaperTrade.edge_at_exec,
                )
                .outerjoin(Market, PaperTrade.market_condition_id == Market.condition_id)
                .where(PaperTrade.strategy == "C")
                .where(PaperTrade.status.in_(["won", "lost", "sold"]))
                .where(sa.or_(PaperTrade.notes.is_(None), PaperTrade.notes != "pre-validation"))
            )

            # Parse city from trade_details or market question
            from arbo.strategies.weather_scanner import parse_city

            city_stats: dict[str, dict] = {}
            for row in result.all():
                city_name = row[0]  # from trade_details JSONB
                if not city_name:
                    # Fallback: parse from market question
                    question = row[1] or ""
                    city_obj = parse_city(question)
                    city_name = city_obj.value if city_obj else "unknown"

                if city_name not in city_stats:
                    city_stats[city_name] = {"wins": 0, "losses": 0, "pnl": 0.0, "edges": []}
                cs = city_stats[city_name]
                pnl_val = float(row[3] or 0)
                status = row[2]
                if status == "won" or (status == "sold" and pnl_val >= 0):
                    cs["wins"] += 1
                else:
                    cs["losses"] += 1
                cs["pnl"] += pnl_val
                if row[4] is not None:
                    cs["edges"].append(float(row[4]))

            cities = []
            for city_name, cs in sorted(city_stats.items(), key=lambda x: x[1]["pnl"], reverse=True):
                resolved = cs["wins"] + cs["losses"]
                cities.append({
                    "city": city_name,
                    "resolved": resolved,
                    "wins": cs["wins"],
                    "losses": cs["losses"],
                    "wr": round(cs["wins"] / resolved, 4) if resolved > 0 else None,
                    "pnl": round(cs["pnl"], 2),
                    "avg_edge": round(sum(cs["edges"]) / len(cs["edges"]), 4) if cs["edges"] else 0,
                })
            return {"cities": cities}
    except Exception as e:
        return {"cities": [], "error": str(e)}


@app.get("/api/forecast-accuracy")
async def api_forecast_accuracy(_user: str = Depends(_verify_credentials)) -> dict[str, Any]:
    """Forecast accuracy: predicted vs actual METAR temperatures per city."""
    try:
        from arbo.utils.db import WeatherForecastRecord, get_session_factory

        factory = get_session_factory()
        async with factory() as session:
            result = await session.execute(
                sa.select(
                    WeatherForecastRecord.city,
                    WeatherForecastRecord.source,
                    sa.func.count(WeatherForecastRecord.id),
                    sa.func.avg(
                        sa.func.abs(
                            WeatherForecastRecord.temp_high_c
                            - WeatherForecastRecord.actual_temp_high_c
                        )
                    ).label("mae_high"),
                    sa.func.avg(
                        WeatherForecastRecord.temp_high_c
                        - WeatherForecastRecord.actual_temp_high_c
                    ).label("bias_high"),
                    sa.func.avg(
                        sa.func.abs(
                            WeatherForecastRecord.temp_low_c
                            - WeatherForecastRecord.actual_temp_low_c
                        )
                    ).label("mae_low"),
                )
                .where(WeatherForecastRecord.actual_temp_high_c.isnot(None))
                .group_by(WeatherForecastRecord.city, WeatherForecastRecord.source)
                .order_by(WeatherForecastRecord.city)
            )
            rows = []
            for row in result.all():
                rows.append({
                    "city": row[0],
                    "source": row[1],
                    "samples": row[2],
                    "mae_high": round(float(row[3] or 0), 2),
                    "bias_high": round(float(row[4] or 0), 2),
                    "mae_low": round(float(row[5] or 0), 2),
                })
            total_samples = sum(r["samples"] for r in rows)
            return {"accuracy": rows, "total_samples": total_samples, "has_data": total_samples > 0}
    except Exception as e:
        return {"accuracy": [], "total_samples": 0, "has_data": False, "error": str(e)}


@app.get("/api/go-live-readiness")
async def api_go_live_readiness(_user: str = Depends(_verify_credentials)) -> dict[str, Any]:
    """Go-live readiness checklist with clear READY/NOT READY verdict."""
    try:
        from arbo.core.health_check import get_expected_vs_reality

        evr = await get_expected_vs_reality()
        actual = evr.get("actual", {})
        baseline = evr.get("baseline", {})

        days = actual.get("days_active", 0)
        resolved = actual.get("total_resolved", 0)
        wr = actual.get("win_rate")
        pnl = actual.get("resolved_pnl", 0)

        # Thresholds
        MIN_DAYS = 28  # 4 weeks
        MIN_RESOLVED = 50
        WR_LOW = baseline.get("oos_wr", 0.38) - 0.15  # 23%
        WR_HIGH = baseline.get("oos_wr", 0.38) + 0.15  # 53%

        checks = [
            {
                "name": "Paper trading doba",
                "required": f"{MIN_DAYS} dni",
                "actual": f"{days:.0f} dni",
                "passed": days >= MIN_DAYS,
                "progress": min(days / MIN_DAYS * 100, 100),
            },
            {
                "name": "Resolved trades",
                "required": f"{MIN_RESOLVED}+",
                "actual": str(resolved),
                "passed": resolved >= MIN_RESOLVED,
                "progress": min(resolved / MIN_RESOLVED * 100, 100),
            },
            {
                "name": "Win Rate",
                "required": f"{WR_LOW:.0%} - {WR_HIGH:.0%}",
                "actual": f"{wr:.1%}" if wr is not None else "—",
                "passed": wr is not None and WR_LOW <= wr <= WR_HIGH,
                "progress": 100 if (wr is not None and WR_LOW <= wr <= WR_HIGH) else 0,
            },
            {
                "name": "Kladny P&L",
                "required": "> $0",
                "actual": f"${pnl:.2f}",
                "passed": pnl > 0,
                "progress": 100 if pnl > 0 else 0,
            },
        ]

        all_passed = all(c["passed"] for c in checks)
        verdict = "READY" if all_passed else "NOT READY"
        return {"verdict": verdict, "checks": checks, "days_remaining": max(MIN_DAYS - days, 0)}
    except Exception as e:
        return {"verdict": "ERROR", "checks": [], "error": str(e)}


@app.get("/api/pnl-projection")
async def api_pnl_projection(_user: str = Depends(_verify_credentials)) -> dict[str, Any]:
    """P&L projection based on daily realized performance."""
    try:
        from arbo.utils.db import PaperTrade, get_session_factory

        import math

        factory = get_session_factory()
        async with factory() as session:
            result = await session.execute(
                sa.select(
                    sa.func.date_trunc("day", PaperTrade.resolved_at).label("day"),
                    sa.func.sum(PaperTrade.actual_pnl),
                )
                .where(PaperTrade.status.in_(["won", "lost", "sold"]))
                .where(sa.or_(PaperTrade.notes.is_(None), PaperTrade.notes != "pre-validation"))
                .group_by(sa.text("1"))
                .order_by(sa.text("1"))
            )
            daily_values = [float(row[1] or 0) for row in result.all()]

        n = len(daily_values)
        if n < 3:
            return {
                "has_data": False,
                "reason": f"Potreba alespon 3 dny s resolved trades (aktualne {n})",
            }

        avg = sum(daily_values) / n
        variance = sum((x - avg) ** 2 for x in daily_values) / max(n - 1, 1)
        std = math.sqrt(variance)

        monthly = round(avg * 30, 2)
        yearly = round(avg * 365, 2)
        ci_monthly = round(1.96 * std * math.sqrt(30), 2)
        ci_yearly = round(1.96 * std * math.sqrt(365), 2)

        return {
            "has_data": True,
            "days_sampled": n,
            "avg_daily_pnl": round(avg, 2),
            "std_daily_pnl": round(std, 2),
            "monthly": {
                "projected": monthly,
                "ci_low": round(monthly - ci_monthly, 2),
                "ci_high": round(monthly + ci_monthly, 2),
            },
            "yearly": {
                "projected": yearly,
                "ci_low": round(yearly - ci_yearly, 2),
                "ci_high": round(yearly + ci_yearly, 2),
            },
        }
    except Exception as e:
        return {"has_data": False, "error": str(e)}


@app.get("/api/strategy-pnl-series")
async def api_strategy_pnl_series(
    _user: str = Depends(_verify_credentials),
) -> dict[str, Any]:
    """Cumulative P&L series per strategy for comparison chart."""
    try:
        from arbo.utils.db import PaperTrade, get_session_factory

        factory = get_session_factory()
        async with factory() as session:
            result = await session.execute(
                sa.select(
                    PaperTrade.strategy,
                    sa.func.date_trunc("day", PaperTrade.resolved_at).label("day"),
                    sa.func.sum(PaperTrade.actual_pnl),
                )
                .where(PaperTrade.status.in_(["won", "lost", "sold"]))
                .where(PaperTrade.resolved_at.isnot(None))
                .where(sa.or_(PaperTrade.notes.is_(None), PaperTrade.notes != "pre-validation"))
                .group_by(PaperTrade.strategy, sa.text("2"))
                .order_by(sa.text("2"))
            )
            # Collect all dates and per-strategy daily P&L
            all_dates: set[str] = set()
            strat_daily: dict[str, dict[str, float]] = {}
            for row in result.all():
                strat = row[0] or "?"
                day = row[1].strftime("%Y-%m-%d") if row[1] else None
                if not day:
                    continue
                all_dates.add(day)
                if strat not in strat_daily:
                    strat_daily[strat] = {}
                strat_daily[strat][day] = float(row[2] or 0)

            labels = sorted(all_dates)
            series: dict[str, list[float]] = {}
            for strat in list(_STRATEGY_META.keys()):
                cum = 0.0
                values: list[float] = []
                for day in labels:
                    cum += strat_daily.get(strat, {}).get(day, 0)
                    values.append(round(cum, 2))
                series[strat] = values

            return {"labels": [d[5:] for d in labels], "series": series}  # MM-DD format
    except Exception as e:
        return {"labels": [], "series": {}, "error": str(e)}


@app.get("/api/export/trades")
async def api_export_trades(
    strategy: str = "",
    _user: str = Depends(_verify_credentials),
) -> Any:
    """Export paper trades as CSV."""
    import csv
    import io

    from fastapi.responses import StreamingResponse

    from arbo.utils.db import PaperTrade, get_session_factory

    try:
        factory = get_session_factory()
        async with factory() as session:
            q = sa.select(PaperTrade).order_by(PaperTrade.placed_at)
            if strategy:
                q = q.where(PaperTrade.strategy == strategy)
            result = await session.execute(q)
            trades = result.scalars().all()

        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow([
            "id", "strategy", "market_condition_id", "side", "price", "size",
            "edge", "status", "actual_pnl", "fee_paid", "placed_at", "resolved_at",
            "city", "forecast_prob", "forecast_temp_c",
        ])
        for t in trades:
            td = t.trade_details or {}
            writer.writerow([
                t.id, t.strategy, t.market_condition_id, t.side,
                float(t.price) if t.price else "", float(t.size) if t.size else "",
                float(t.edge_at_exec) if t.edge_at_exec else "",
                t.status, float(t.actual_pnl) if t.actual_pnl else "",
                float(t.fee_paid) if t.fee_paid else "",
                t.placed_at.isoformat() if t.placed_at else "",
                t.resolved_at.isoformat() if t.resolved_at else "",
                td.get("city", ""), td.get("forecast_prob", ""), td.get("forecast_temp_c", ""),
            ])
        output.seek(0)
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=arbo_trades.csv"},
        )
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/export/scan-log")
async def api_export_scan_log(_user: str = Depends(_verify_credentials)) -> Any:
    """Export weather scan log as CSV."""
    import csv
    import io

    from fastapi.responses import StreamingResponse

    from arbo.utils.db import WeatherScanLog, get_session_factory

    try:
        factory = get_session_factory()
        async with factory() as session:
            result = await session.execute(
                sa.select(WeatherScanLog).order_by(WeatherScanLog.scan_at)
            )
            rows = result.scalars().all()

        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow([
            "id", "scan_at", "city", "target_date", "question",
            "forecast_temp_c", "forecast_prob", "market_price", "edge",
            "direction", "volume_24h", "liquidity",
            "quality_gate_passed", "quality_gate_reason", "traded", "trade_size",
        ])
        for r in rows:
            writer.writerow([
                r.id, r.scan_at.isoformat() if r.scan_at else "",
                r.city, r.target_date.isoformat() if r.target_date else "",
                r.question[:100], r.forecast_temp_c, r.forecast_prob, r.market_price,
                r.edge, r.direction, r.volume_24h, r.liquidity,
                r.quality_gate_passed, r.quality_gate_reason or "", r.traded, r.trade_size or "",
            ])
        output.seek(0)
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=arbo_scan_log.csv"},
        )
    except Exception as e:
        return {"error": str(e)}


def create_app(orchestrator: Any) -> FastAPI:
    """Factory function for RDH orchestrator integration.

    Called by main_rdh.py to inject the orchestrator into dashboard state.
    """
    state.orchestrator = orchestrator
    return app
