"""Main orchestrator -- starts all 9 layers + signal processing (PM-301).

Entry point: python -m arbo.main --mode paper [--log-level INFO]

Architecture:
- ArboOrchestrator class owns all component instances
- Task-per-layer model: each of 9 layers = independent asyncio.Task
- asyncio.Queue signal bus: all layers -> queue -> signal processor -> confluence -> paper engine
- Health monitor task (every 30s): detect crashed/hung tasks, auto-restart with backoff
- Graceful shutdown on SIGINT/SIGTERM: set asyncio.Event, cancel tasks, close sessions
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import signal
import time
from dataclasses import dataclass, field
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import sqlalchemy as sa

from arbo.config.settings import get_config
from arbo.utils.logger import get_logger, setup_logging

if TYPE_CHECKING:
    from arbo.core.scanner import Signal

logger = get_logger("orchestrator")


# ---------------------------------------------------------------------------
# Layer state tracking
# ---------------------------------------------------------------------------


@dataclass
class LayerState:
    """Runtime state for a single layer task."""

    name: str
    task: asyncio.Task[None] | None = None
    restart_count: int = 0
    error_timestamps: list[float] = field(default_factory=list)
    last_heartbeat: float = field(default_factory=time.monotonic)
    enabled: bool = True
    permanent_stop: bool = False


# Maximum errors in a window before emergency shutdown
_ERROR_WINDOW_S = 600  # 10 minutes
_ERROR_THRESHOLD = 3

# LLM re-check interval when degraded
_LLM_RECHECK_S = 300  # 5 minutes


class ArboOrchestrator:
    """Main orchestrator that ties all 9 strategy layers together.

    Lifecycle: ``__init__`` -> ``start()`` -> runs until ``stop()`` or signal.
    """

    def __init__(self, mode: str = "paper") -> None:
        self._mode = mode
        self._config = get_config()
        self._orch_cfg = self._config.orchestrator
        self._capital = Decimal(str(self._config.bankroll))

        # Signal bus
        self._signal_queue: asyncio.Queue[Signal] = asyncio.Queue()

        # Shutdown event
        self._shutdown_event = asyncio.Event()

        # Layer states
        self._layers: dict[str, LayerState] = {}

        # Internal tasks (signal processor, health monitor)
        self._internal_tasks: list[asyncio.Task[None]] = []

        # --- Components (initialized in start()) ---
        self._risk_manager: Any = None
        self._paper_engine: Any = None
        self._confluence: Any = None
        self._discovery: Any = None
        self._poly_client: Any = None
        self._odds_client: Any = None
        self._event_matcher: Any = None
        self._gemini: Any = None
        self._whale_discovery: Any = None

        # Layer components
        self._market_maker: Any = None
        self._value_signal: Any = None
        self._market_graph: Any = None
        self._whale_monitor: Any = None
        self._logical_arb: Any = None
        self._temporal_crypto: Any = None
        self._order_flow: Any = None
        self._attention: Any = None
        self._sports_latency: Any = None
        self._arb_monitor: Any = None

        # Dashboard / reports
        self._cli_dashboard: Any = None
        self._report_generator: Any = None
        self._slack_bot: Any = None

        # LLM degraded mode flag
        self._llm_degraded = False
        self._llm_last_check: float = 0.0

        # Startup timestamp for uptime tracking
        self._start_time: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Initialize all components and start all layer tasks."""
        logger.info("orchestrator_starting", mode=self._mode, capital=str(self._capital))
        self._start_time = time.monotonic()

        self._install_signal_handlers()
        await self._init_components()
        self._start_layer_tasks()
        self._start_internal_tasks()

        logger.info(
            "orchestrator_started",
            layers=len(self._layers),
            mode=self._mode,
        )

        # Block until shutdown
        await self._shutdown_event.wait()
        await self.stop()

    async def stop(self) -> None:
        """Graceful shutdown: cancel all tasks, close sessions."""
        logger.info("orchestrator_stopping")

        # Cancel internal tasks
        for task in self._internal_tasks:
            if not task.done():
                task.cancel()
        for task in self._internal_tasks:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task

        # Cancel layer tasks
        for state in self._layers.values():
            if state.task and not state.task.done():
                state.task.cancel()
        for state in self._layers.values():
            if state.task:
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await state.task

        # Close sessions
        await self._close_components()

        logger.info("orchestrator_stopped")

    # ------------------------------------------------------------------
    # Component initialization
    # ------------------------------------------------------------------

    async def _init_components(self) -> None:
        """Initialize all components. Each is optional -- failures are logged."""
        # Risk manager (required)
        try:
            from arbo.core.risk_manager import RiskManager

            self._risk_manager = RiskManager(self._capital)
        except Exception as e:
            logger.error("init_risk_manager_failed", error=str(e))
            raise

        # Paper engine (required in paper mode)
        try:
            from arbo.core.paper_engine import PaperTradingEngine

            self._paper_engine = PaperTradingEngine(
                initial_capital=self._capital,
                risk_manager=self._risk_manager,
            )
        except Exception as e:
            logger.error("init_paper_engine_failed", error=str(e))
            raise

        # Confluence scorer (required)
        try:
            from arbo.core.confluence import ConfluenceScorer

            self._confluence = ConfluenceScorer(
                risk_manager=self._risk_manager,
                capital=self._capital,
            )
        except Exception as e:
            logger.error("init_confluence_failed", error=str(e))
            raise

        # Market discovery
        self._discovery = await self._init_optional("MarketDiscovery", self._init_discovery)

        # Polymarket client
        self._poly_client = await self._init_optional("PolymarketClient", self._init_poly_client)

        # Odds API client
        self._odds_client = await self._init_optional("OddsApiClient", self._init_odds_client)

        # Event matcher
        self._event_matcher = await self._init_optional("EventMatcher", self._init_event_matcher)

        # Gemini agent
        self._gemini = await self._init_optional("GeminiAgent", self._init_gemini)

        # Whale discovery
        self._whale_discovery = await self._init_optional(
            "WhaleDiscovery", self._init_whale_discovery
        )

        # Layer components
        self._market_maker = await self._init_optional("MarketMaker", self._init_market_maker)
        self._value_signal = await self._init_optional(
            "ValueSignalGenerator", self._init_value_signal
        )
        self._market_graph = await self._init_optional(
            "SemanticMarketGraph", self._init_market_graph
        )
        self._whale_monitor = await self._init_optional("WhaleMonitor", self._init_whale_monitor)
        self._logical_arb = await self._init_optional("LogicalArbScanner", self._init_logical_arb)
        self._temporal_crypto = await self._init_optional(
            "TemporalCryptoScanner", self._init_temporal_crypto
        )
        self._order_flow = await self._init_optional("OrderFlowMonitor", self._init_order_flow)
        self._attention = await self._init_optional("AttentionMarketsScanner", self._init_attention)
        self._sports_latency = await self._init_optional(
            "SportsLatencyScanner", self._init_sports_latency
        )
        self._arb_monitor = await self._init_optional("NegRiskArbMonitor", self._init_arb_monitor)

        # Dashboard / reports
        self._cli_dashboard = await self._init_optional("CLIDashboard", self._init_dashboard)
        self._report_generator = await self._init_optional(
            "ReportGenerator", self._init_report_generator
        )

        # Database connectivity check
        await self._init_optional("Database", self._init_db)

        # Slack bot
        self._slack_bot = await self._init_optional("SlackBot", self._init_slack_bot)

        # Restore paper engine state from DB
        if self._paper_engine is not None:
            try:
                await self._paper_engine.load_state_from_db()
            except Exception as e:
                logger.warning("load_state_from_db_skipped", error=str(e))

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
        return d

    async def _init_poly_client(self) -> Any:
        from arbo.connectors.polymarket_client import PolymarketClient

        c = PolymarketClient()
        await c.initialize()
        return c

    async def _init_odds_client(self) -> Any:
        from arbo.connectors.odds_api_client import OddsApiClient

        return OddsApiClient()

    async def _init_event_matcher(self) -> Any:
        from arbo.connectors.event_matcher import EventMatcher

        return EventMatcher()

    async def _init_gemini(self) -> Any:
        from arbo.agents.gemini_agent import GeminiAgent

        g = GeminiAgent()
        await g.initialize()
        return g

    async def _init_whale_discovery(self) -> Any:
        from arbo.strategies.whale_discovery import WhaleDiscovery

        d = WhaleDiscovery()
        await d.initialize()
        return d

    async def _init_market_maker(self) -> Any:
        if not self._poly_client:
            return None
        from arbo.strategies.market_maker import MarketMaker

        return MarketMaker(poly_client=self._poly_client, capital=self._capital)

    async def _init_value_signal(self) -> Any:
        if not self._discovery or not self._odds_client or not self._event_matcher:
            return None
        from arbo.strategies.value_signal import ValueSignalGenerator

        return ValueSignalGenerator(
            discovery=self._discovery,
            odds_client=self._odds_client,
            matcher=self._event_matcher,
        )

    async def _init_market_graph(self) -> Any:
        from arbo.models.market_graph import SemanticMarketGraph

        g = SemanticMarketGraph(discovery=self._discovery, gemini=self._gemini)
        await g.initialize()
        return g

    async def _init_whale_monitor(self) -> Any:
        if not self._whale_discovery:
            return None
        from arbo.strategies.whale_monitor import WhaleMonitor

        m = WhaleMonitor(discovery=self._whale_discovery)
        await m.initialize()
        return m

    async def _init_logical_arb(self) -> Any:
        if not self._market_graph:
            return None
        from arbo.strategies.logical_arb import LogicalArbScanner

        return LogicalArbScanner(
            market_graph=self._market_graph,
            gemini=self._gemini,
            discovery=self._discovery,
        )

    async def _init_temporal_crypto(self) -> Any:
        if not self._discovery:
            return None
        from arbo.strategies.temporal_crypto import TemporalCryptoScanner

        s = TemporalCryptoScanner(discovery=self._discovery)
        await s.initialize()
        return s

    async def _init_order_flow(self) -> Any:
        from arbo.connectors.polygon_flow import OrderFlowMonitor

        def _enqueue_signal(sig: Signal) -> None:
            try:
                self._signal_queue.put_nowait(sig)
            except asyncio.QueueFull:
                logger.warning("signal_queue_full", layer=7)

        return OrderFlowMonitor(on_signal=_enqueue_signal)

    async def _init_attention(self) -> Any:
        if not self._discovery:
            return None
        from arbo.strategies.attention_markets import AttentionMarketsScanner

        return AttentionMarketsScanner(discovery=self._discovery, gemini=self._gemini)

    async def _init_sports_latency(self) -> Any:
        # Layer 9 disabled during paper trading — scores are hardcoded to 0, burns API credits
        logger.info("sports_latency_disabled_paper_mode")
        return None

    async def _init_arb_monitor(self) -> Any:
        if not self._discovery:
            return None
        from arbo.strategies.arb_monitor import NegRiskArbMonitor

        return NegRiskArbMonitor(discovery=self._discovery)

    async def _init_dashboard(self) -> Any:
        from arbo.dashboard.cli_dashboard import CLIDashboard

        return CLIDashboard()

    async def _init_report_generator(self) -> Any:
        from arbo.dashboard.report_generator import ReportGenerator

        return ReportGenerator()

    async def _init_db(self) -> Any:
        """Verify database connectivity."""
        from arbo.utils.db import get_engine

        engine = get_engine()
        async with engine.connect() as conn:
            await conn.execute(sa.text("SELECT 1"))
        logger.info("database_connected")
        return True

    async def _init_slack_bot(self) -> Any:
        """Initialize Slack bot with dependency-injected callbacks."""
        cfg = self._config
        if not cfg.slack_bot_token or not cfg.slack_app_token:
            logger.warning("slack_bot_skipped", reason="missing tokens")
            return None

        from arbo.dashboard.slack_bot import SlackBot

        return SlackBot(
            bot_token=cfg.slack_bot_token,
            app_token=cfg.slack_app_token,
            channel_id=cfg.slack_channel_id,
            get_status_fn=self._get_status_for_slack,
            get_pnl_fn=self._get_pnl_for_slack,
            shutdown_fn=self._slack_shutdown,
        )

    # ------------------------------------------------------------------
    # Slack callbacks
    # ------------------------------------------------------------------

    async def _get_status_for_slack(self) -> dict[str, Any]:
        """Build system status dict for /status command."""
        active = sum(
            1
            for s in self._layers.values()
            if s.enabled and not s.permanent_stop and s.task and not s.task.done()
        )
        balance = str(self._paper_engine.balance) if self._paper_engine else "N/A"
        positions = len(self._paper_engine.open_positions) if self._paper_engine else 0
        uptime = int(time.monotonic() - self._start_time) if self._start_time else 0

        return {
            "mode": self._mode,
            "uptime_s": uptime,
            "layers_active": active,
            "layers_total": len(self._layers),
            "balance": balance,
            "open_positions": positions,
        }

    async def _get_pnl_for_slack(self) -> dict[str, Any]:
        """Build P&L dict for /pnl command."""
        if self._paper_engine is None:
            return {}
        return self._paper_engine.get_stats()

    async def _slack_shutdown(self) -> None:
        """Emergency shutdown triggered via /kill."""
        logger.critical("emergency_shutdown_via_slack")
        self._shutdown_event.set()

    # ------------------------------------------------------------------
    # Close components
    # ------------------------------------------------------------------

    async def _close_components(self) -> None:
        """Close all component sessions."""
        # Sync final paper engine state to DB
        if self._paper_engine is not None:
            try:
                snapshot = self._paper_engine.take_snapshot()
                await self._paper_engine.save_snapshot_to_db(snapshot)
                await self._paper_engine.sync_positions_to_db()
            except Exception as e:
                logger.warning("final_state_sync_failed", error=str(e))

        # Close Slack bot
        if self._slack_bot is not None:
            try:
                await self._slack_bot.close()
            except Exception as e:
                logger.warning("close_slack_bot_error", error=str(e))

        closables = [
            ("discovery", self._discovery),
            ("poly_client", self._poly_client),
            ("odds_client", self._odds_client),
            ("market_graph", self._market_graph),
            ("whale_monitor", self._whale_monitor),
            ("temporal_crypto", self._temporal_crypto),
            ("order_flow", self._order_flow),
            ("sports_latency", self._sports_latency),
        ]
        for name, comp in closables:
            if comp is not None and hasattr(comp, "close"):
                try:
                    await comp.close()
                except Exception as e:
                    logger.warning("close_component_error", component=name, error=str(e))

        if self._order_flow is not None and hasattr(self._order_flow, "stop"):
            try:
                await self._order_flow.stop()
            except Exception as e:
                logger.warning("close_order_flow_error", error=str(e))

    # ------------------------------------------------------------------
    # Layer task management
    # ------------------------------------------------------------------

    def _start_layer_tasks(self) -> None:
        """Create and start an asyncio.Task for each available layer."""
        layer_defs: list[tuple[str, Any, float]] = [
            ("discovery", self._run_discovery, 300),
            ("L1_market_maker", self._run_market_maker, 300),
            ("L2_value_signal", self._run_value_signal, 300),
            ("L3_semantic_graph", self._run_semantic_graph, 86400),
            ("L4_whale_monitor", self._run_whale_monitor, 4),
            ("L5_logical_arb", self._run_logical_arb, 600),
            ("L6_temporal_crypto", self._run_temporal_crypto, 60),
            ("L7_order_flow", self._run_order_flow, 0),  # continuous
            ("L8_attention", self._run_attention, 1800),
            ("L9_sports_latency", self._run_sports_latency, 5),
            ("negrisk_monitor", self._run_arb_monitor, 60),
        ]

        for name, coro_factory, interval in layer_defs:
            state = LayerState(name=name)
            self._layers[name] = state
            if interval == 0:
                # Continuous task (e.g., WebSocket)
                state.task = asyncio.create_task(
                    self._run_layer_task(name, coro_factory, 0),
                    name=f"layer_{name}",
                )
            else:
                state.task = asyncio.create_task(
                    self._run_layer_task(name, coro_factory, interval),
                    name=f"layer_{name}",
                )

    def _start_internal_tasks(self) -> None:
        """Start signal processor, health monitor, schedulers, and data collector."""
        self._internal_tasks.append(
            asyncio.create_task(self._signal_processor(), name="signal_processor")
        )
        self._internal_tasks.append(
            asyncio.create_task(self._health_monitor(), name="health_monitor")
        )
        self._internal_tasks.append(
            asyncio.create_task(self._snapshot_scheduler(), name="snapshot_scheduler")
        )
        self._internal_tasks.append(
            asyncio.create_task(self._daily_report_scheduler(), name="daily_report_scheduler")
        )
        self._internal_tasks.append(
            asyncio.create_task(self._weekly_report_scheduler(), name="weekly_report_scheduler")
        )
        self._internal_tasks.append(
            asyncio.create_task(self._data_collector(), name="data_collector")
        )
        # Start Slack bot as internal task
        if self._slack_bot is not None:
            self._internal_tasks.append(
                asyncio.create_task(self._slack_bot.start(), name="slack_bot")
            )

    async def _run_layer_task(
        self,
        name: str,
        coro_factory: Any,
        interval_s: float,
    ) -> None:
        """Generic layer task loop with error handling.

        Args:
            name: Layer name for logging.
            coro_factory: Async callable that runs one iteration.
            interval_s: Sleep interval between iterations (0 = single run).
        """
        state = self._layers[name]

        while not self._shutdown_event.is_set():
            if state.permanent_stop or not state.enabled:
                await asyncio.sleep(1)
                continue

            try:
                state.last_heartbeat = time.monotonic()

                if interval_s <= 0:
                    # Continuous task: run coro in background, keep heartbeat alive
                    inner = asyncio.create_task(coro_factory(), name=f"{name}_inner")
                    try:
                        while not inner.done() and not self._shutdown_event.is_set():
                            await asyncio.sleep(1)
                            state.last_heartbeat = time.monotonic()
                    finally:
                        if not inner.done():
                            inner.cancel()
                            try:
                                await inner
                            except (asyncio.CancelledError, Exception):
                                pass
                        elif inner.exception():
                            raise inner.exception()
                    break
                else:
                    await coro_factory()
                    state.last_heartbeat = time.monotonic()

                    # Sleep in small increments; keep heartbeat alive during idle
                    deadline = time.monotonic() + interval_s
                    while time.monotonic() < deadline and not self._shutdown_event.is_set():
                        await asyncio.sleep(min(1.0, deadline - time.monotonic()))
                        state.last_heartbeat = time.monotonic()

            except asyncio.CancelledError:
                raise
            except Exception as e:
                self._handle_layer_error(name, e)
                # Back off after error
                await asyncio.sleep(min(60.0, 2**state.restart_count))

    def _handle_layer_error(self, name: str, error: Exception) -> None:
        """Track errors and decide on restart or shutdown.

        - 3x same error in 10min -> emergency shutdown
        - Max restart count exceeded -> stop layer permanently
        """
        state = self._layers[name]
        now = time.monotonic()

        state.error_timestamps.append(now)
        # Prune old timestamps
        state.error_timestamps = [t for t in state.error_timestamps if now - t < _ERROR_WINDOW_S]

        state.restart_count += 1

        logger.error(
            "layer_error",
            layer=name,
            error=str(error),
            error_type=type(error).__name__,
            restart_count=state.restart_count,
            errors_in_window=len(state.error_timestamps),
        )

        # Check for repeated errors -> emergency shutdown
        if len(state.error_timestamps) >= _ERROR_THRESHOLD:
            logger.critical(
                "repeated_errors_emergency_shutdown",
                layer=name,
                errors=len(state.error_timestamps),
                window_s=_ERROR_WINDOW_S,
            )
            self._shutdown_event.set()
            return

        # Check max restart count
        if state.restart_count >= self._orch_cfg.max_restart_count:
            logger.warning(
                "layer_permanently_stopped",
                layer=name,
                restart_count=state.restart_count,
            )
            state.permanent_stop = True

    # ------------------------------------------------------------------
    # Layer coroutines
    # ------------------------------------------------------------------

    async def _run_discovery(self) -> None:
        """Refresh market catalog."""
        if self._discovery is None:
            return
        await self._discovery.refresh()

    async def _run_market_maker(self) -> None:
        """Layer 1: Market making heartbeat."""
        if self._market_maker is None or self._discovery is None:
            return
        markets = self._discovery.get_mm_candidates()
        if not markets:
            return
        # Start runs the internal heartbeat loop; we just refresh market list
        if not getattr(self._market_maker, "_running", False):
            await self._market_maker.start(markets)
        else:
            self._market_maker._active_markets = self._market_maker.filter_markets(markets)

    async def _run_value_signal(self) -> None:
        """Layer 2: Value signal scan."""
        if self._value_signal is None:
            return
        signals = await self._value_signal.scan()
        for sig in signals:
            await self._signal_queue.put(sig)

    async def _run_semantic_graph(self) -> None:
        """Layer 3: Refresh semantic market graph."""
        if self._market_graph is None:
            return
        await self._market_graph.refresh_if_stale()

    async def _run_whale_monitor(self) -> None:
        """Layer 4: Whale position polling."""
        if self._whale_monitor is None:
            return
        signals = await self._whale_monitor.poll_cycle()
        for sig in signals:
            await self._signal_queue.put(sig)

    async def _run_logical_arb(self) -> None:
        """Layer 5: Logical/combinatorial arb scan."""
        if self._logical_arb is None:
            return
        if self._llm_degraded:
            logger.debug("layer5_skipped_llm_degraded")
            return
        signals = await self._logical_arb.scan()
        for sig in signals:
            await self._signal_queue.put(sig)

    async def _run_temporal_crypto(self) -> None:
        """Layer 6: Temporal crypto scan."""
        if self._temporal_crypto is None:
            return
        signals = await self._temporal_crypto.scan()
        for sig in signals:
            await self._signal_queue.put(sig)

    async def _run_order_flow(self) -> None:
        """Layer 7: Order flow monitor (continuous WebSocket)."""
        if self._order_flow is None:
            return
        await self._order_flow.initialize()
        await self._order_flow.start()
        # start() returns after the read_loop task is created; we wait on shutdown
        while not self._shutdown_event.is_set():
            await asyncio.sleep(1)

    async def _run_attention(self) -> None:
        """Layer 8: Attention markets scan."""
        if self._attention is None:
            return
        if self._llm_degraded:
            logger.debug("layer8_skipped_llm_degraded")
            return
        signals = await self._attention.scan()
        for sig in signals:
            await self._signal_queue.put(sig)

    async def _run_sports_latency(self) -> None:
        """Layer 9: Sports latency poll."""
        if self._sports_latency is None:
            return
        signals = await self._sports_latency.poll_cycle()
        for sig in signals:
            await self._signal_queue.put(sig)

    async def _run_arb_monitor(self) -> None:
        """NegRisk arb monitor (monitoring only)."""
        if self._arb_monitor is None:
            return
        self._arb_monitor.scan()

    # ------------------------------------------------------------------
    # Signal processor
    # ------------------------------------------------------------------

    async def _signal_processor(self) -> None:
        """Batch signals from queue, feed confluence scorer, route to paper engine.

        Uses a 2-second batch window to accumulate signals before scoring.
        """
        batch_timeout = self._orch_cfg.signal_batch_timeout_s

        while not self._shutdown_event.is_set():
            batch: list[Signal] = []

            # Collect signals for batch_timeout seconds
            try:
                first = await asyncio.wait_for(self._signal_queue.get(), timeout=batch_timeout)
                batch.append(first)
            except TimeoutError:
                continue
            except asyncio.CancelledError:
                raise

            # Drain any remaining signals within the window
            deadline = time.monotonic() + batch_timeout
            while time.monotonic() < deadline:
                try:
                    sig = self._signal_queue.get_nowait()
                    batch.append(sig)
                except asyncio.QueueEmpty:
                    await asyncio.sleep(0.1)

            if not batch:
                continue

            logger.info("signal_batch", count=len(batch))

            try:
                await self._process_signal_batch(batch)
            except Exception as e:
                logger.error("signal_processor_error", error=str(e))

    async def _process_signal_batch(self, signals: list[Signal]) -> None:
        """Score signals via confluence and route to paper engine."""
        if self._confluence is None or self._paper_engine is None:
            return

        # Build market category map from discovery
        category_map: dict[str, str] = {}
        if self._discovery is not None:
            for sig in signals:
                market = self._discovery.get_by_condition_id(sig.market_condition_id)
                if market is not None:
                    category_map[sig.market_condition_id] = market.category

        tradeable = self._confluence.get_tradeable(signals, category_map)

        for opp in tradeable:
            # Get market for fee info
            fee_enabled = False
            market_category = category_map.get(opp.market_condition_id, "other")
            if self._discovery is not None:
                market = self._discovery.get_by_condition_id(opp.market_condition_id)
                if market is not None:
                    fee_enabled = market.fee_enabled

            side = "BUY" if "BUY" in opp.direction.value else "SELL"
            market_price = Decimal("0.50")  # placeholder

            trade = self._paper_engine.place_trade(
                market_condition_id=opp.market_condition_id,
                token_id=opp.token_id,
                side=side,
                market_price=market_price,
                model_prob=market_price + opp.best_edge,
                layer=min(opp.contributing_layers) if opp.contributing_layers else 0,
                market_category=market_category,
                fee_enabled=fee_enabled,
                confluence_score=opp.score,
            )
            if trade is not None:
                await self._paper_engine.save_trade_to_db(trade)

    # ------------------------------------------------------------------
    # Health monitor
    # ------------------------------------------------------------------

    async def _health_monitor(self) -> None:
        """Check task health every health_check_interval_s.

        Detects crashed and hung tasks and triggers restarts.
        Also checks LLM availability for degraded mode.
        """
        interval = self._orch_cfg.health_check_interval_s
        timeout = self._orch_cfg.heartbeat_timeout_s

        while not self._shutdown_event.is_set():
            await asyncio.sleep(interval)

            now = time.monotonic()

            for name, state in self._layers.items():
                if state.permanent_stop or not state.enabled:
                    continue

                # Check if task crashed
                if state.task is not None and state.task.done():
                    exc = state.task.exception() if not state.task.cancelled() else None
                    if exc is not None:
                        logger.warning(
                            "health_task_crashed",
                            layer=name,
                            error=str(exc),
                        )
                    self._restart_layer(name)

                # Check heartbeat timeout
                elif now - state.last_heartbeat > timeout:
                    logger.warning(
                        "health_task_hung",
                        layer=name,
                        last_heartbeat_ago_s=int(now - state.last_heartbeat),
                    )
                    self._restart_layer(name)

            # Check LLM degraded mode
            await self._check_llm_health()

    def _restart_layer(self, name: str) -> None:
        """Restart a single layer task."""
        state = self._layers.get(name)
        if state is None or state.permanent_stop:
            return

        if state.restart_count >= self._orch_cfg.max_restart_count:
            state.permanent_stop = True
            logger.warning("layer_max_restarts_reached", layer=name)
            return

        # Cancel old task
        if state.task and not state.task.done():
            state.task.cancel()

        # Find the original coro factory and interval
        layer_map = self._get_layer_map()
        if name not in layer_map:
            return

        coro_factory, interval = layer_map[name]
        state.task = asyncio.create_task(
            self._run_layer_task(name, coro_factory, interval),
            name=f"layer_{name}",
        )

        logger.info("layer_restarted", layer=name, restart_count=state.restart_count)

    def _get_layer_map(self) -> dict[str, tuple[Any, float]]:
        """Return mapping of layer name to (coro_factory, interval)."""
        return {
            "discovery": (self._run_discovery, 300),
            "L1_market_maker": (self._run_market_maker, 300),
            "L2_value_signal": (self._run_value_signal, 300),
            "L3_semantic_graph": (self._run_semantic_graph, 86400),
            "L4_whale_monitor": (self._run_whale_monitor, 4),
            "L5_logical_arb": (self._run_logical_arb, 600),
            "L6_temporal_crypto": (self._run_temporal_crypto, 60),
            "L7_order_flow": (self._run_order_flow, 0),
            "L8_attention": (self._run_attention, 1800),
            "L9_sports_latency": (self._run_sports_latency, 5),
            "negrisk_monitor": (self._run_arb_monitor, 60),
        }

    async def _check_llm_health(self) -> None:
        """Check if LLM is available; toggle degraded mode for layers 5 and 8."""
        now = time.monotonic()
        if now - self._llm_last_check < _LLM_RECHECK_S:
            return
        self._llm_last_check = now

        if self._gemini is None:
            if not self._llm_degraded:
                self._llm_degraded = True
                logger.warning("llm_degraded_mode_enabled", reason="Gemini agent not available")
            return

        # Try a lightweight check
        try:
            # If gemini has a health check, use it; otherwise just check it exists
            if hasattr(self._gemini, "is_healthy"):
                healthy = await self._gemini.is_healthy()
            else:
                healthy = True

            if healthy and self._llm_degraded:
                self._llm_degraded = False
                logger.info("llm_degraded_mode_disabled")
            elif not healthy and not self._llm_degraded:
                self._llm_degraded = True
                logger.warning("llm_degraded_mode_enabled", reason="LLM health check failed")
        except Exception as e:
            if not self._llm_degraded:
                self._llm_degraded = True
                logger.warning("llm_degraded_mode_enabled", reason=str(e))

    # ------------------------------------------------------------------
    # Scheduled tasks
    # ------------------------------------------------------------------

    async def _snapshot_scheduler(self) -> None:
        """Take hourly portfolio snapshots and persist to DB."""
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
                logger.error("snapshot_scheduler_error", error=str(e))

    async def _daily_report_scheduler(self) -> None:
        """Generate and send daily report at configured hour (UTC)."""
        from datetime import UTC, datetime

        target_hour = self._orch_cfg.daily_report_hour_utc

        while not self._shutdown_event.is_set():
            now = datetime.now(UTC)
            # Sleep until target hour
            if now.hour == target_hour and now.minute < 5:
                await self._send_daily_report()
                # Sleep past this window to avoid double-send
                await asyncio.sleep(3600)
            else:
                await asyncio.sleep(60)

    async def _weekly_report_scheduler(self) -> None:
        """Generate and send weekly report on configured day/hour (UTC)."""
        from datetime import UTC, datetime

        target_day = self._orch_cfg.weekly_report_day  # 6 = Sunday
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
                    "notes": t.notes,
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
        """Hourly: snapshot polymarket_mid/spread/volume for active markets."""
        from datetime import UTC, datetime

        interval = self._orch_cfg.snapshot_interval_s  # same as snapshot interval (1h)

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
                    for m in markets:
                        mid = m.price_yes if m.price_yes is not None else 0.5
                        row = RealMarketData(
                            market_condition_id=m.condition_id,
                            polymarket_mid=float(mid),
                            spread=float(m.spread) if hasattr(m, "spread") and m.spread else None,
                            volume_24h=m.volume_24h,
                            liquidity=m.liquidity,
                            source="gamma_snapshot",
                            captured_at=now,
                        )
                        session.add(row)
                    await session.commit()

                logger.info("data_collector_snapshot", markets=len(markets))
            except Exception as e:
                logger.warning("data_collector_error", error=str(e))

    # ------------------------------------------------------------------
    # Signal handlers
    # ------------------------------------------------------------------

    def _install_signal_handlers(self) -> None:
        """Install SIGINT/SIGTERM handlers to trigger graceful shutdown."""
        loop = asyncio.get_running_loop()

        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self._on_signal, sig)

    def _on_signal(self, sig: signal.Signals) -> None:
        """Handle OS signal by setting shutdown event."""
        logger.info("shutdown_signal_received", signal=sig.name)
        self._shutdown_event.set()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for Arbo orchestrator."""
    parser = argparse.ArgumentParser(description="Arbo Trading System")
    parser.add_argument("--mode", default="paper", choices=["paper", "live"])
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(log_level=args.log_level)

    orchestrator = ArboOrchestrator(mode=args.mode)
    asyncio.run(orchestrator.start())


if __name__ == "__main__":
    main()
