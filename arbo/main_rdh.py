"""RDH Orchestrator — 3-strategy architecture (RDH-305).

Entry point: python -m arbo.main --mode rdh

Architecture:
- RDHOrchestrator: task-per-strategy model (A, B, C + shared infrastructure)
- Per-strategy quality gates (NOT confluence scoring)
- Health monitor with crash detection + auto-restart
- Graceful shutdown on SIGINT/SIGTERM
- Capital allocation engine: A=$400, B=$400, C=$1000, Reserve=$200
"""

from __future__ import annotations

import asyncio
import contextlib
import signal
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from arbo.config.settings import get_config
from arbo.utils.logger import get_logger

logger = get_logger("rdh_orchestrator")


# ---------------------------------------------------------------------------
# Task state tracking
# ---------------------------------------------------------------------------


@dataclass
class TaskState:
    """Runtime state for a managed task."""

    name: str
    task: asyncio.Task[None] | None = None
    restart_count: int = 0
    error_timestamps: list[float] = field(default_factory=list)
    last_heartbeat: float = field(default_factory=time.monotonic)
    enabled: bool = True
    permanent_stop: bool = False


# Error thresholds
_ERROR_WINDOW_S = 600  # 10 minutes
_ERROR_THRESHOLD = 3


class RDHOrchestrator:
    """3-strategy orchestrator for Arbo RDH architecture.

    Strategies:
    - A: Theta Decay (taker flow peak optimism → sell NO on longshots)
    - B: Reflexivity Surfer (Kaito mindshare divergence, stub mode)
    - C: Compound Weather (multi-source weather → temperature ladders)
    """

    def __init__(self, mode: str = "paper") -> None:
        self._mode = mode
        self._config = get_config()
        self._orch_cfg = self._config.orchestrator
        self._capital = Decimal(str(self._config.bankroll))

        # Shutdown coordination
        self._shutdown_event = asyncio.Event()

        # Task states
        self._tasks: dict[str, TaskState] = {}
        self._internal_tasks: list[asyncio.Task[None]] = []

        # Components (initialized in start())
        self._risk_manager: Any = None
        self._paper_engine: Any = None
        self._discovery: Any = None
        self._gemini: Any = None
        self._flow_tracker: Any = None
        self._kaito: Any = None

        # Strategies
        self._strategy_a: Any = None
        self._strategy_b: Any = None
        self._strategy_c: Any = None

        # Infrastructure
        self._report_generator: Any = None
        self._slack_bot: Any = None
        self._web_dashboard: Any = None

        # Runtime state
        self._start_time: float = 0.0
        self._markets: list[Any] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Initialize components and start all strategy tasks."""
        logger.info("rdh_starting", mode=self._mode, capital=str(self._capital))
        self._start_time = time.monotonic()

        self._install_signal_handlers()
        await self._init_components()
        self._start_strategy_tasks()
        self._start_internal_tasks()

        logger.info(
            "rdh_started",
            strategies=len([t for t in self._tasks.values() if t.enabled]),
            mode=self._mode,
        )

        # Block until shutdown
        await self._shutdown_event.wait()
        await self.stop()

    async def stop(self) -> None:
        """Graceful shutdown: cancel all tasks, close sessions."""
        logger.info("rdh_stopping")

        # Cancel internal tasks
        for task in self._internal_tasks:
            if not task.done():
                task.cancel()
        for task in self._internal_tasks:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task

        # Cancel strategy tasks
        for state in self._tasks.values():
            if state.task and not state.task.done():
                state.task.cancel()
        for state in self._tasks.values():
            if state.task:
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await state.task

        # Close strategies
        for strategy in [self._strategy_a, self._strategy_b, self._strategy_c]:
            if strategy is not None and hasattr(strategy, "close"):
                with contextlib.suppress(Exception):
                    await strategy.close()

        # Close discovery session
        if self._discovery is not None and hasattr(self._discovery, "close"):
            with contextlib.suppress(Exception):
                await self._discovery.close()

        logger.info("rdh_stopped")

    # ------------------------------------------------------------------
    # Signal handlers
    # ------------------------------------------------------------------

    def _install_signal_handlers(self) -> None:
        """Install SIGINT/SIGTERM handlers for graceful shutdown."""
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self._handle_signal, sig)

    def _handle_signal(self, sig: signal.Signals) -> None:
        logger.info("signal_received", signal=sig.name)
        self._shutdown_event.set()

    # ------------------------------------------------------------------
    # Component initialization
    # ------------------------------------------------------------------

    async def _init_components(self) -> None:
        """Initialize all components. Failures are logged, not fatal (except core)."""
        # Risk manager (required)
        from arbo.core.risk_manager import RiskManager

        RiskManager.reset()
        self._risk_manager = RiskManager(self._capital)

        # Paper engine (required)
        from arbo.core.paper_engine import PaperTradingEngine

        self._paper_engine = PaperTradingEngine(
            initial_capital=self._capital,
            risk_manager=self._risk_manager,
        )

        # Market discovery
        self._discovery = await self._init_optional("MarketDiscovery", self._init_discovery)

        # Gemini agent
        self._gemini = await self._init_optional("GeminiAgent", self._init_gemini)

        # Flow tracker (for Strategy A)
        self._flow_tracker = await self._init_optional(
            "MarketFlowTracker", self._init_flow_tracker
        )

        # Kaito client (for Strategy B — stub mode)
        self._kaito = await self._init_optional("KaitoClient", self._init_kaito)

        # Strategy A: Theta Decay
        self._strategy_a = await self._init_optional("StrategyA", self._init_strategy_a)

        # Strategy B: Reflexivity Surfer
        self._strategy_b = await self._init_optional("StrategyB", self._init_strategy_b)

        # Strategy C: Compound Weather
        self._strategy_c = await self._init_optional("StrategyC", self._init_strategy_c)

        # Reports
        self._report_generator = await self._init_optional(
            "ReportGenerator", self._init_report_generator
        )

        # Database check
        await self._init_optional("Database", self._init_db)

        # Slack bot
        self._slack_bot = await self._init_optional("SlackBot", self._init_slack_bot)

        # Web dashboard
        self._web_dashboard = await self._init_optional("WebDashboard", self._init_web_dashboard)

        # Restore paper engine state from DB
        if self._paper_engine is not None:
            try:
                await self._paper_engine.load_state_from_db()
            except Exception as e:
                logger.warning("load_state_from_db_skipped", error=str(e))

        # Sync risk manager state from restored positions
        self._sync_risk_from_positions()

    async def _init_optional(self, name: str, factory: Any) -> Any:
        """Initialize an optional component. Returns None on failure."""
        try:
            return await factory()
        except Exception as e:
            logger.warning("init_optional_failed", component=name, error=str(e))
            return None

    async def _init_discovery(self) -> Any:
        from arbo.connectors.market_discovery import MarketDiscovery

        d = MarketDiscovery()
        await d.initialize()
        markets = await d.refresh()
        self._markets = markets
        logger.info("discovery_initial", markets=len(markets))
        return d

    async def _init_gemini(self) -> Any:
        from arbo.agents.gemini_agent import GeminiAgent

        g = GeminiAgent()
        await g.initialize()
        return g

    async def _init_flow_tracker(self) -> Any:
        from arbo.connectors.polygon_flow import MarketFlowTracker

        return MarketFlowTracker()

    async def _init_kaito(self) -> Any:
        from arbo.connectors.kaito_api import KaitoClient

        return KaitoClient(mode="stub")

    async def _init_strategy_a(self) -> Any:
        from arbo.strategies.theta_decay import ThetaDecay

        s = ThetaDecay(
            risk_manager=self._risk_manager,
            flow_tracker=self._flow_tracker,
            paper_engine=self._paper_engine,
        )
        await s.init()
        return s

    async def _init_strategy_b(self) -> Any:
        from arbo.strategies.reflexivity_surfer import ReflexivitySurfer

        s = ReflexivitySurfer(
            risk_manager=self._risk_manager,
            kaito_client=self._kaito,
            paper_engine=self._paper_engine,
        )
        await s.init()
        return s

    async def _init_strategy_c(self) -> Any:
        from arbo.strategies.strategy_c import StrategyC

        s = StrategyC(
            risk_manager=self._risk_manager,
            paper_engine=self._paper_engine,
            metoffice_api_key=self._config.metoffice_api_key,
        )
        await s.init()
        return s

    async def _init_report_generator(self) -> Any:
        from arbo.dashboard.report_generator import ReportGenerator

        return ReportGenerator()

    async def _init_db(self) -> Any:
        from arbo.utils.db import get_session_factory

        factory = get_session_factory()
        async with factory() as session:
            from sqlalchemy import text

            await session.execute(text("SELECT 1"))
        logger.info("database_connected")

    async def _init_slack_bot(self) -> Any:
        from arbo.dashboard.slack_bot import SlackBot

        bot = SlackBot(orchestrator=self)
        return bot

    async def _init_web_dashboard(self) -> Any:
        from arbo.dashboard.web import create_app

        return create_app(self)

    def _sync_risk_from_positions(self) -> None:
        """Sync risk manager per-strategy deployed capital from open positions."""
        if self._paper_engine is None or self._risk_manager is None:
            return
        for pos in self._paper_engine.open_positions:
            strategy = getattr(pos, "strategy", "")
            if strategy in ("A", "B", "C"):
                self._risk_manager.strategy_post_trade(strategy, pos.size)

    # ------------------------------------------------------------------
    # Strategy tasks
    # ------------------------------------------------------------------

    def _start_strategy_tasks(self) -> None:
        """Create asyncio tasks for each strategy + discovery."""
        task_defs: list[tuple[str, Any, float]] = [
            ("discovery", self._run_discovery, 300),  # 5 min
        ]

        if self._strategy_a is not None:
            td_cfg = self._config.theta_decay
            task_defs.append(("strategy_A", self._run_strategy_a, td_cfg.snapshot_interval_s))
        if self._strategy_b is not None:
            rx_cfg = self._config.reflexivity
            task_defs.append(("strategy_B", self._run_strategy_b, rx_cfg.scan_interval_s))
        if self._strategy_c is not None:
            wx_cfg = self._config.weather
            task_defs.append(("strategy_C", self._run_strategy_c, wx_cfg.scan_interval_s))

        for name, coro_factory, interval in task_defs:
            state = TaskState(name=name)
            self._tasks[name] = state
            state.task = asyncio.create_task(
                self._run_task_loop(name, coro_factory, interval),
                name=f"rdh_{name}",
            )

    def _start_internal_tasks(self) -> None:
        """Start infrastructure tasks."""
        self._internal_tasks.append(
            asyncio.create_task(self._health_monitor(), name="health_monitor")
        )
        self._internal_tasks.append(
            asyncio.create_task(self._snapshot_scheduler(), name="snapshot_scheduler")
        )
        self._internal_tasks.append(
            asyncio.create_task(self._daily_report_scheduler(), name="daily_report")
        )
        self._internal_tasks.append(
            asyncio.create_task(self._weekly_report_scheduler(), name="weekly_report")
        )
        self._internal_tasks.append(
            asyncio.create_task(self._position_price_updater(), name="price_updater")
        )
        self._internal_tasks.append(
            asyncio.create_task(self._resolution_checker(), name="resolution_checker")
        )
        self._internal_tasks.append(
            asyncio.create_task(self._data_collector(), name="data_collector")
        )
        if self._slack_bot is not None:
            self._internal_tasks.append(
                asyncio.create_task(self._slack_bot.start(), name="slack_bot")
            )
        if self._web_dashboard is not None:
            self._internal_tasks.append(
                asyncio.create_task(self._run_web_dashboard(), name="web_dashboard")
            )

    async def _run_web_dashboard(self) -> None:
        """Run web dashboard in background."""
        import uvicorn

        config = uvicorn.Config(
            self._web_dashboard,
            host=self._config.dashboard.host,
            port=self._config.dashboard.port,
            log_level="warning",
        )
        server = uvicorn.Server(config)
        await server.serve()

    # ------------------------------------------------------------------
    # Task loop
    # ------------------------------------------------------------------

    async def _run_task_loop(
        self,
        name: str,
        coro_factory: Any,
        interval_s: float,
    ) -> None:
        """Generic task loop with heartbeat and error handling."""
        state = self._tasks[name]

        while not self._shutdown_event.is_set():
            if state.permanent_stop or not state.enabled:
                await asyncio.sleep(1)
                continue

            try:
                state.last_heartbeat = time.monotonic()
                await coro_factory()
                state.last_heartbeat = time.monotonic()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                state.error_timestamps.append(time.monotonic())
                logger.warning("task_error", task=name, error=str(e))

                # Check error rate
                cutoff = time.monotonic() - _ERROR_WINDOW_S
                state.error_timestamps = [t for t in state.error_timestamps if t > cutoff]
                if len(state.error_timestamps) >= _ERROR_THRESHOLD:
                    logger.error("task_error_threshold", task=name, count=len(state.error_timestamps))
                    state.permanent_stop = True
                    continue

            # Sleep between iterations
            for _ in range(int(interval_s)):
                if self._shutdown_event.is_set():
                    return
                await asyncio.sleep(1)
                state.last_heartbeat = time.monotonic()

    # ------------------------------------------------------------------
    # Strategy runners
    # ------------------------------------------------------------------

    async def _run_discovery(self) -> None:
        """Refresh market catalog from Gamma API."""
        if self._discovery is None:
            return
        markets = await self._discovery.refresh()
        self._markets = markets
        logger.info("discovery_refreshed", markets=len(markets))

    async def _run_strategy_a(self) -> None:
        """Run Strategy A: Theta Decay poll cycle + exit checks."""
        if self._strategy_a is None:
            return
        trades = await self._strategy_a.poll_cycle(self._markets)
        if trades:
            logger.info("strategy_a_trades", count=len(trades))
            await self._save_trades_to_db(trades)

        # Check exits
        prices = self._get_current_prices()
        exits = self._strategy_a.check_exits(prices)
        if exits:
            logger.info("strategy_a_exits", count=len(exits))

    async def _run_strategy_b(self) -> None:
        """Run Strategy B: Reflexivity Surfer poll cycle + exit checks."""
        if self._strategy_b is None:
            return
        trades = await self._strategy_b.poll_cycle(self._markets)
        if trades:
            logger.info("strategy_b_trades", count=len(trades))
            await self._save_trades_to_db(trades)

        # Check exits
        prices = self._get_current_prices()
        exits = self._strategy_b.check_exits(prices)
        if exits:
            logger.info("strategy_b_exits", count=len(exits))

    async def _run_strategy_c(self) -> None:
        """Run Strategy C: Compound Weather poll cycle."""
        if self._strategy_c is None:
            return
        trades = await self._strategy_c.poll_cycle(self._markets)
        if trades:
            logger.info("strategy_c_trades", count=len(trades))
            await self._save_trades_to_db(trades)

    def _get_current_prices(self) -> dict[str, Decimal]:
        """Get current YES prices from discovery cache for exit checks."""
        prices: dict[str, Decimal] = {}
        if self._discovery is None:
            return prices
        for market in self._markets:
            if hasattr(market, "condition_id") and hasattr(market, "price_yes"):
                if market.price_yes is not None:
                    prices[market.condition_id] = market.price_yes
        return prices

    async def _save_trades_to_db(self, trades: list[dict[str, Any]]) -> None:
        """Best-effort save trade records to database."""
        try:
            from arbo.utils.db import PaperTradeRow, get_session_factory

            factory = get_session_factory()
            async with factory() as session:
                for trade in trades:
                    row = PaperTradeRow(
                        market_condition_id=trade.get("condition_id", ""),
                        token_id=trade.get("token_id", ""),
                        side=trade.get("side", ""),
                        price=float(trade.get("price", 0)),
                        size=float(trade.get("size", 0)),
                        strategy=trade.get("strategy", ""),
                    )
                    session.add(row)
                await session.commit()
        except Exception as e:
            logger.debug("save_trades_db_error", error=str(e))

    # ------------------------------------------------------------------
    # Health monitor
    # ------------------------------------------------------------------

    async def _health_monitor(self) -> None:
        """Check task health, detect crashes, restart with backoff."""
        interval = self._orch_cfg.health_check_interval_s
        timeout = self._orch_cfg.heartbeat_timeout_s

        while not self._shutdown_event.is_set():
            await asyncio.sleep(interval)
            now = time.monotonic()

            for name, state in self._tasks.items():
                if state.permanent_stop or not state.enabled:
                    continue

                # Crashed task
                if state.task is not None and state.task.done():
                    exc = state.task.exception() if not state.task.cancelled() else None
                    if exc is not None:
                        logger.warning("health_task_crashed", task=name, error=str(exc))
                    await self._restart_task(name)

                # Hung task
                elif now - state.last_heartbeat > timeout:
                    logger.warning(
                        "health_task_hung",
                        task=name,
                        last_heartbeat_ago_s=int(now - state.last_heartbeat),
                    )
                    await self._restart_task(name)

    async def _restart_task(self, name: str) -> None:
        """Restart a crashed/hung task."""
        state = self._tasks.get(name)
        if state is None or state.permanent_stop:
            return

        state.restart_count += 1
        if state.restart_count >= self._orch_cfg.max_restart_count:
            state.permanent_stop = True
            logger.warning("task_max_restarts", task=name)
            return

        # Cancel old task
        if state.task and not state.task.done():
            state.task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await state.task

        # Get task config
        task_map = self._get_task_map()
        if name not in task_map:
            return

        coro_factory, interval = task_map[name]
        state.last_heartbeat = time.monotonic()
        state.task = asyncio.create_task(
            self._run_task_loop(name, coro_factory, interval),
            name=f"rdh_{name}",
        )
        logger.info("task_restarted", task=name, restart_count=state.restart_count)

    def _get_task_map(self) -> dict[str, tuple[Any, float]]:
        """Map task names to (coro_factory, interval)."""
        result: dict[str, tuple[Any, float]] = {
            "discovery": (self._run_discovery, 300),
        }
        if self._strategy_a is not None:
            result["strategy_A"] = (
                self._run_strategy_a,
                self._config.theta_decay.snapshot_interval_s,
            )
        if self._strategy_b is not None:
            result["strategy_B"] = (
                self._run_strategy_b,
                self._config.reflexivity.scan_interval_s,
            )
        if self._strategy_c is not None:
            result["strategy_C"] = (
                self._run_strategy_c,
                self._config.weather.scan_interval_s,
            )
        return result

    # ------------------------------------------------------------------
    # Infrastructure tasks
    # ------------------------------------------------------------------

    async def _snapshot_scheduler(self) -> None:
        """Hourly portfolio snapshots."""
        interval = self._orch_cfg.snapshot_interval_s
        while not self._shutdown_event.is_set():
            await asyncio.sleep(interval)
            if self._paper_engine is None:
                continue
            try:
                snapshot = self._paper_engine.take_snapshot()
                await self._paper_engine.save_snapshot_to_db(snapshot)
                await self._paper_engine.sync_positions_to_db()
            except Exception as e:
                logger.error("snapshot_error", error=str(e))

    async def _position_price_updater(self) -> None:
        """Update open position prices every 5 minutes."""
        while not self._shutdown_event.is_set():
            await asyncio.sleep(300)
            if self._paper_engine is None or self._discovery is None:
                continue
            try:
                updated = 0
                for pos in self._paper_engine.open_positions:
                    market = self._discovery.get_by_condition_id(pos.market_condition_id)
                    if market is None:
                        continue
                    if pos.token_id == market.token_id_yes and market.price_yes is not None:
                        self._paper_engine.update_position_price(pos.token_id, market.price_yes)
                        updated += 1
                    elif pos.token_id == market.token_id_no and market.price_no is not None:
                        self._paper_engine.update_position_price(pos.token_id, market.price_no)
                        updated += 1
                if updated > 0:
                    await self._paper_engine.sync_positions_to_db()
            except Exception as e:
                logger.error("price_update_error", error=str(e))

    async def _resolution_checker(self) -> None:
        """Check for resolved markets every 5 minutes."""
        while not self._shutdown_event.is_set():
            await asyncio.sleep(300)
            if self._paper_engine is None or self._discovery is None:
                continue
            try:
                positions = list(self._paper_engine.open_positions)
                if not positions:
                    continue

                now = datetime.now(UTC)
                fetched: dict[str, Any] = {}

                for pos in positions:
                    cid = pos.market_condition_id

                    # Skip if end_date in future
                    cached = self._discovery.get_by_condition_id(cid)
                    if cached is not None and cached.end_date:
                        try:
                            end_dt = datetime.fromisoformat(
                                cached.end_date.replace("Z", "+00:00")
                            )
                            if end_dt > now:
                                continue
                        except (ValueError, TypeError):
                            pass

                    # Fetch market state
                    if cid not in fetched:
                        market = await self._discovery.fetch_by_token_id(pos.token_id)
                        fetched[cid] = market

                    market = fetched[cid]
                    if market is None or not market.closed:
                        continue

                    yes_price = market.price_yes
                    if yes_price is None:
                        continue

                    yes_won = yes_price > Decimal("0.5")
                    if pos.token_id == market.token_id_yes:
                        winning = yes_won
                    elif pos.token_id == market.token_id_no:
                        winning = not yes_won
                    else:
                        continue

                    pnl = self._paper_engine.resolve_market(pos.token_id, winning)

                    # Notify strategy
                    strategy = getattr(pos, "strategy", "")
                    if strategy == "A" and self._strategy_a is not None:
                        self._strategy_a.handle_resolution(cid, pnl)
                    elif strategy == "B" and self._strategy_b is not None:
                        self._strategy_b.handle_resolution(cid, pnl)
                    elif strategy == "C" and self._strategy_c is not None:
                        self._strategy_c.handle_resolution(cid, pnl)

                    await self._paper_engine.update_resolved_trades_in_db(pos.token_id)

                    logger.info(
                        "position_resolved",
                        condition_id=cid[:20],
                        strategy=strategy,
                        pnl=str(pnl),
                    )

            except Exception as e:
                logger.error("resolution_check_error", error=str(e))

    async def _daily_report_scheduler(self) -> None:
        """Daily report at configured UTC hour."""
        target_hour = self._orch_cfg.daily_report_hour_utc

        while not self._shutdown_event.is_set():
            now = datetime.now(UTC)
            if now.hour == target_hour and now.minute < 5:
                await self._send_daily_report()
                await asyncio.sleep(3600)
            else:
                await asyncio.sleep(60)

    async def _weekly_report_scheduler(self) -> None:
        """Weekly report on configured day/hour."""
        target_day = self._orch_cfg.weekly_report_day
        target_hour = self._orch_cfg.weekly_report_hour_utc

        while not self._shutdown_event.is_set():
            now = datetime.now(UTC)
            if now.weekday() == target_day and now.hour == target_hour and now.minute < 5:
                await self._send_weekly_report()
                await asyncio.sleep(3600)
            else:
                await asyncio.sleep(60)

    async def _send_daily_report(self) -> None:
        """Generate and send daily report via Slack."""
        if self._report_generator is None or self._paper_engine is None:
            return
        try:
            trades = [
                {
                    "layer": t.layer,
                    "actual_pnl": str(t.actual_pnl) if t.actual_pnl else None,
                    "size": str(t.size),
                    "notes": getattr(t, "notes", ""),
                    "strategy": getattr(t, "strategy", ""),
                }
                for t in self._paper_engine.trade_history
            ]
            report = self._report_generator.generate_daily(trades=trades, signals=[])
            formatted = self._report_generator.format_slack_report(report)
            if self._slack_bot is not None:
                await self._slack_bot.send_daily_report(formatted["blocks"])
            logger.info("daily_report_sent")
        except Exception as e:
            logger.error("daily_report_error", error=str(e))

    async def _send_weekly_report(self) -> None:
        """Generate and send weekly report via Slack."""
        if self._report_generator is None or self._paper_engine is None:
            return
        try:
            trades = [
                {
                    "layer": t.layer,
                    "actual_pnl": str(t.actual_pnl) if t.actual_pnl else None,
                    "size": str(t.size),
                    "confluence_score": t.confluence_score,
                    "token_id": t.token_id,
                    "strategy": getattr(t, "strategy", ""),
                }
                for t in self._paper_engine.trade_history
            ]
            daily = self._report_generator.generate_daily(trades=trades, signals=[])
            weekly = self._report_generator.generate_weekly(
                daily_reports=[daily],
                trades=trades,
                portfolio_balance=self._paper_engine.balance,
            )
            formatted = self._report_generator.format_slack_weekly_report(weekly)
            if self._slack_bot is not None:
                await self._slack_bot.send_weekly_report(formatted["blocks"])
            logger.info("weekly_report_sent")
        except Exception as e:
            logger.error("weekly_report_error", error=str(e))

    async def _data_collector(self) -> None:
        """Hourly market data persistence."""
        interval = self._orch_cfg.snapshot_interval_s
        while not self._shutdown_event.is_set():
            await asyncio.sleep(interval)
            if self._discovery is None:
                continue
            try:
                from arbo.utils.db import RealMarketData, get_session_factory

                markets = self._discovery.get_all_active()
                if not markets:
                    continue

                factory = get_session_factory()
                now = datetime.now(UTC)
                async with factory() as session:
                    for m in markets[:200]:  # Cap to avoid memory issues
                        if m.price_yes is not None:
                            row = RealMarketData(
                                condition_id=m.condition_id,
                                timestamp=now,
                                polymarket_mid=float(m.price_yes),
                                volume_24h=float(m.volume_24h),
                                liquidity=float(m.liquidity),
                            )
                            session.add(row)
                    await session.commit()
                logger.info("data_collected", markets=min(len(markets), 200))
            except Exception as e:
                logger.debug("data_collector_error", error=str(e))

    # ------------------------------------------------------------------
    # Properties (for dashboard/Slack integration)
    # ------------------------------------------------------------------

    @property
    def strategy_stats(self) -> dict[str, dict[str, Any]]:
        """Get stats for all 3 strategies."""
        result: dict[str, dict[str, Any]] = {}
        if self._strategy_a is not None:
            result["A"] = self._strategy_a.stats
        if self._strategy_b is not None:
            result["B"] = self._strategy_b.stats
        if self._strategy_c is not None:
            result["C"] = self._strategy_c.stats
        return result

    @property
    def task_states(self) -> dict[str, dict[str, Any]]:
        """Get state of all tasks (for health dashboard)."""
        return {
            name: {
                "enabled": s.enabled,
                "restart_count": s.restart_count,
                "permanent_stop": s.permanent_stop,
                "heartbeat_ago_s": int(time.monotonic() - s.last_heartbeat),
                "running": s.task is not None and not s.task.done() if s.task else False,
            }
            for name, s in self._tasks.items()
        }

    @property
    def uptime_s(self) -> float:
        """Seconds since orchestrator start."""
        return time.monotonic() - self._start_time if self._start_time else 0.0
