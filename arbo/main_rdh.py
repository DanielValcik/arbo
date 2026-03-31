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
import json
import os
import signal
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
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
        self._order_flow_monitor: Any = None
        self._flow_tracker: Any = None
        self._kaito: Any = None

        # Strategies
        self._strategy_a: Any = None
        self._strategy_b: Any = None
        self._strategy_c: Any = None
        self._strategy_c2: Any = None  # EMOS + Edge Exit Fusion (weather variant)
        self._strategy_b2: Any = None  # Crypto Price Edge
        self._strategy_b3: Any = None  # Binance Oracle Scalper (5-min BTC Up/Down)
        self._binance_ws: Any = None   # Binance WebSocket feed for B2/B3
        self._exit_manager: Any = None  # Persistent exit manager for live C2/B2

        # B2 social momentum (Santiment + CoinGecko)
        self._santiment: Any = None
        self._coingecko: Any = None
        self._social_divergence: Any = None

        # Strategy C live price layer
        self._poly_client_readonly: Any = None
        self._orderbook_provider: Any = None
        self._iem_client: Any = None
        self._weather_resolver: Any = None
        self._shadow_exit_tracker: Any = None

        # Infrastructure
        self._report_generator: Any = None
        self._slack_bot: Any = None
        self._web_dashboard: Any = None

        # Runtime state
        self._start_time: float = 0.0
        self._markets: list[Any] = []
        self._latest_divergence_signals: list[Any] = []

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
        for strategy in [self._strategy_a, self._strategy_b, self._strategy_c, self._strategy_c2]:
            if strategy is not None and hasattr(strategy, "close"):
                with contextlib.suppress(Exception):
                    await strategy.close()

        # Close discovery session
        if self._discovery is not None and hasattr(self._discovery, "close"):
            with contextlib.suppress(Exception):
                await self._discovery.close()

        # Close order flow monitor
        if self._order_flow_monitor is not None:
            with contextlib.suppress(Exception):
                await self._order_flow_monitor.stop()

        # Close Santiment + CoinGecko clients
        if self._santiment is not None:
            with contextlib.suppress(Exception):
                await self._santiment.close()
        if self._coingecko is not None:
            with contextlib.suppress(Exception):
                await self._coingecko.close()

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

        # Order flow monitor (for Strategy A — on-chain taker flow from Polygon)
        self._order_flow_monitor = await self._init_optional(
            "OrderFlowMonitor", self._init_order_flow_monitor
        )
        # Flow tracker is exposed from the monitor (or standalone fallback)
        if self._order_flow_monitor is not None:
            self._flow_tracker = self._order_flow_monitor.market_tracker
        else:
            self._flow_tracker = await self._init_optional(
                "MarketFlowTracker", self._init_flow_tracker
            )

        # Kaito client (for Strategy B — stub mode)
        self._kaito = await self._init_optional("KaitoClient", self._init_kaito)

        # Santiment client (for Strategy B2 — on-chain metrics)
        self._santiment = await self._init_optional("Santiment", self._init_santiment)

        # CoinGecko client (for Strategy B2 — market + community metrics)
        self._coingecko = await self._init_optional("CoinGecko", self._init_coingecko)

        # Social divergence calculator (for Strategy B2)
        self._social_divergence = await self._init_optional(
            "SocialDivergence", self._init_social_divergence
        )

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

        # Restore paper engine state from DB (BEFORE C2 init — C2 needs open positions)
        if self._paper_engine is not None:
            try:
                await self._paper_engine.load_state_from_db()
            except Exception as e:
                logger.warning("load_state_from_db_skipped", error=str(e))

        # Sync risk manager state from restored positions
        self._sync_risk_from_positions()

        # Strategy C2: EMOS + Edge Exit Fusion (AFTER load_state_from_db so it can restore positions)
        self._strategy_c2 = await self._init_optional("StrategyC2", self._init_strategy_c2)

        # Strategy B2: Crypto Price Edge
        self._strategy_b2 = await self._init_optional("StrategyB2", self._init_strategy_b2)

        # Strategy B3: Binance Oracle Scalper
        self._strategy_b3 = await self._init_optional("StrategyB3", self._init_strategy_b3)

        # Restore realized P&L from DB into risk manager
        await self._restore_pnl_from_db()

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

    async def _init_order_flow_monitor(self) -> Any:
        """Initialize OrderFlowMonitor for on-chain taker flow (Strategy A)."""
        from arbo.connectors.polygon_flow import OrderFlowMonitor

        session_factory = None
        try:
            from arbo.utils.db import get_session_factory

            session_factory = get_session_factory()
        except Exception:
            pass

        monitor = OrderFlowMonitor(
            on_signal=None,  # Strategy A uses peak detection, not L7 signals
            poll_interval=60,
            session_factory=session_factory,
        )
        await monitor.initialize()
        return monitor

    async def _init_flow_tracker(self) -> Any:
        from arbo.connectors.polygon_flow import MarketFlowTracker

        return MarketFlowTracker()

    async def _init_kaito(self) -> Any:
        from arbo.connectors.kaito_api import KaitoClient

        return KaitoClient(live_mode=False)

    async def _init_santiment(self) -> Any:
        from arbo.connectors.santiment_client import SantimentClient

        return SantimentClient()

    async def _init_coingecko(self) -> Any:
        import os

        from arbo.connectors.coingecko_client import CoinGeckoClient

        api_key = os.environ.get("COINGECKO_API_KEY", "")
        if not api_key:
            logger.info("coingecko_no_key", msg="COINGECKO_API_KEY not set, skipping")
            return None
        return CoinGeckoClient(api_key=api_key)

    async def _init_social_divergence(self) -> Any:
        from arbo.strategies.social_divergence import SocialDivergenceCalculator

        return SocialDivergenceCalculator()

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
            paper_engine=self._paper_engine,
            divergence_calc=self._social_divergence,
        )
        await s.init()
        return s

    async def _init_strategy_c(self) -> Any:
        from arbo.connectors.orderbook_provider import OrderbookProvider
        from arbo.connectors.polymarket_client import PolymarketClient
        from arbo.connectors.weather_iem import IEMClient
        from arbo.strategies.strategy_c import StrategyC
        from arbo.strategies.weather_resolution import WeatherResolutionChecker

        # Read-only CLOB client for live NegRisk prices
        try:
            pc = PolymarketClient()
            await pc.initialize()
            self._poly_client_readonly = pc
            self._orderbook_provider = OrderbookProvider(poly_client=pc, cache_ttl_s=30.0)
            logger.info("clob_live_price_layer_ready")
        except Exception as e:
            logger.warning("clob_live_price_layer_failed", error=str(e))
            self._orderbook_provider = None

        # IEM METAR client for weather resolution
        try:
            iem = IEMClient()
            await iem.initialize()
            self._iem_client = iem
            self._weather_resolver = WeatherResolutionChecker(iem)
            logger.info("iem_metar_resolution_ready")
        except Exception as e:
            logger.warning("iem_init_failed", error=str(e))
            self._iem_client = None
            self._weather_resolver = None

        s = StrategyC(
            risk_manager=self._risk_manager,
            paper_engine=self._paper_engine,
            metoffice_api_key=self._config.metoffice_api_key,
            orderbook_provider=self._orderbook_provider,
        )
        await s.init()

        # Shadow exit tracker for A/B testing exit logic
        from arbo.strategies.shadow_exit_tracker import ShadowExitTracker

        self._shadow_exit_tracker = ShadowExitTracker(min_hold_edge=0.15)
        logger.info("shadow_exit_tracker_initialized", min_hold_edge=0.15)

        return s

    async def _init_strategy_c2(self) -> Any:
        """Initialize Strategy C2: EMOS + Edge Exit Fusion.

        Supports paper and live execution modes.
        Set C2_EXECUTION_MODE=live in .env to enable real CLOB trading.
        """
        import os

        from arbo.strategies.strategy_c2 import StrategyC2

        execution_mode = os.getenv("C2_EXECUTION_MODE", "paper")
        live_executor = None

        if execution_mode == "live":
            from arbo.core.exit_manager import ExitManager
            from arbo.core.live_executor import LiveExecutor

            if self._poly_client_readonly is not None:
                live_executor = LiveExecutor(self._poly_client_readonly)
                self._exit_manager = ExitManager(live_executor)
                logger.info("c2_live_executor_ready")
            else:
                logger.warning("c2_live_executor_no_client", msg="Falling back to paper mode")
                execution_mode = "paper"
                self._exit_manager = None
        else:
            self._exit_manager = None

        s = StrategyC2(
            risk_manager=self._risk_manager,
            paper_engine=self._paper_engine,
            metoffice_api_key=self._config.metoffice_api_key,
            orderbook_provider=self._orderbook_provider,
            execution_mode=execution_mode,
            live_executor=live_executor,
        )
        await s.init()

        # Give C2 reference to exit manager (prevents buying tokens being exited)
        if self._exit_manager:
            s._exit_manager_ref = self._exit_manager

        return s

    async def _init_strategy_b2(self) -> Any:
        """Initialize Strategy B2: Crypto Price Edge.

        Supports paper and live execution modes.
        Set B2_EXECUTION_MODE=live in .env to enable real CLOB trading.
        """
        import os

        from arbo.connectors.binance_ws import BinanceWSFeed
        from arbo.strategies.strategy_b2 import StrategyB2

        execution_mode = os.getenv("B2_EXECUTION_MODE", "paper")
        live_executor = None

        # Initialize Binance WebSocket for real-time prices
        self._binance_ws = BinanceWSFeed(symbols=["BTCUSDT", "ETHUSDT"])
        await self._binance_ws.start()
        logger.info("binance_ws_started_for_b2")

        if execution_mode == "live":
            from arbo.core.exit_manager import ExitManager
            from arbo.core.live_executor import LiveExecutor

            if self._poly_client_readonly is not None:
                live_executor = LiveExecutor(self._poly_client_readonly)
                # Reuse exit manager from C2, or create one if not yet created
                if self._exit_manager is None:
                    self._exit_manager = ExitManager(live_executor)
                logger.info("b2_live_executor_ready")
            else:
                logger.warning("b2_live_executor_no_client", msg="Falling back to paper mode")
                execution_mode = "paper"

        s = StrategyB2(
            risk_manager=self._risk_manager,
            paper_engine=self._paper_engine,
            binance_ws=self._binance_ws,
            orderbook_provider=self._orderbook_provider,
            execution_mode=execution_mode,
            live_executor=live_executor,
        )
        await s.init()

        # Give B2 reference to exit manager (prevents buying tokens being exited)
        if self._exit_manager:
            s._exit_manager_ref = self._exit_manager

        return s

    async def _init_strategy_b3(self) -> Any:
        """Initialize Strategy B3: Binance Oracle Scalper.

        Reuses Binance WebSocket from B2 (starts it if not already running).
        Also starts RTDS Chainlink feed for resolution price comparison.
        Supports paper/dual/live modes via B3_EXECUTION_MODE env var.
        """
        import os

        from arbo.strategies.strategy_b3 import StrategyB3

        execution_mode = os.getenv("B3_EXECUTION_MODE", "paper")
        live_executor = None

        # Ensure Binance WS is running (may already be started by B2)
        if self._binance_ws is None:
            from arbo.connectors.binance_ws import BinanceWSFeed

            self._binance_ws = BinanceWSFeed(symbols=["BTCUSDT", "ETHUSDT"])
            await self._binance_ws.start()
            logger.info("binance_ws_started_for_b3")

        # Start RTDS Chainlink feed (resolution price + Binance via same WS)
        rtds_feed = None
        try:
            from arbo.connectors.rtds_chainlink import RTDSChainlinkFeed

            rtds_feed = RTDSChainlinkFeed()
            await rtds_feed.start()
            logger.info("rtds_chainlink_started_for_b3")
        except Exception as e:
            logger.warning("rtds_chainlink_init_failed", error=str(e))

        # Live executor for dual/live mode
        if execution_mode in ("dual", "live"):
            from arbo.core.live_executor import LiveExecutor

            if self._poly_client_readonly is not None:
                live_executor = LiveExecutor(self._poly_client_readonly)
                logger.info("b3_live_executor_ready", mode=execution_mode,
                           msg="capital from wallet balance, % sizing")
            else:
                logger.warning("b3_live_executor_no_client", msg="Falling back to paper mode")
                execution_mode = "paper"

        s = StrategyB3(
            risk_manager=self._risk_manager,
            paper_engine=self._paper_engine,
            binance_ws=self._binance_ws,
            rtds_feed=rtds_feed,
            execution_mode=execution_mode,
            live_executor=live_executor,
        )
        await s.init()
        return s

    async def _init_gefs_background(self) -> None:
        """Background task: download GEFS + init C1f models (takes ~3 min)."""
        try:
            await self._init_gefs_ensemble()
        except Exception as e:
            logger.warning("gefs_background_failed", error=str(e))

    async def _init_gefs_ensemble(self) -> Any:
        """Download today's GEFS ensemble + init C1f EMOSEnsembleModel.

        Daily GEFS download: 31 members × ~400KB = ~12MB via Range requests.
        Fits EMOSEnsembleModel per city for forward-looking sigma estimation.
        """
        from arbo.connectors.gefs_downloader import download_today
        from arbo.strategies.weather_scanner import init_ensemble_models

        # Download today's ensemble forecast
        try:
            ensemble_stats = await download_today()
            logger.info(
                "gefs_download_ok",
                cities=len(ensemble_stats),
            )
        except Exception as e:
            logger.warning("gefs_download_failed", error=str(e))
            return None

        # Build training observations from DB (historical forecast vs actual)
        # For now, load from ensemble_stats table for all historical dates
        try:
            from arbo.utils.db import get_session_factory
            from sqlalchemy import text

            factory = get_session_factory()
            async with factory() as session:
                rows = await session.execute(
                    text("SELECT city, target_date, ensemble_std FROM ensemble_stats")
                )
                all_stds: dict[str, dict[str, float]] = {}
                for city, dt, std in rows:
                    if city not in all_stds:
                        all_stds[city] = {}
                    all_stds[city][dt] = std

            # Training observations: GEFS mean vs Open-Meteo actual
            # Primary: weather_training_obs (backfilled from GEFS + Open-Meteo)
            # Fallback: weather_scan_log (if it has actual_temp column)
            rows2 = None
            async with factory() as session:
                try:
                    rows2 = await session.execute(text("""
                        SELECT city, target_date, forecast_temp, actual_temp
                        FROM weather_training_obs
                        ORDER BY target_date
                    """))
                except Exception:
                    # Fallback to weather_scan_log
                    try:
                        rows2 = await session.execute(text("""
                            SELECT city, target_date, forecast_temp_c, 0.0
                            FROM weather_scan_log
                            WHERE forecast_temp_c IS NOT NULL
                            LIMIT 0
                        """))
                    except Exception:
                        pass

            training_obs: dict[str, list] = {}
            if rows2:
                from arbo.models.emos_ensemble import Observation
                for city, dt, forecast, actual in rows2:
                    if city not in training_obs:
                        training_obs[city] = []
                    training_obs[city].append(
                        Observation(forecast=forecast, actual=actual, date=dt, city=city)
                    )

            if training_obs and all_stds:
                n_cities = init_ensemble_models(
                    training_obs=training_obs,
                    ensemble_stds=all_stds,
                )
                logger.info("c1f_ensemble_ready", n_cities=n_cities)
            else:
                logger.info(
                    "c1f_ensemble_skipped",
                    reason="insufficient training data",
                    n_training_cities=len(training_obs),
                    n_ensemble_cities=len(all_stds),
                )

            # Pass ensemble_stds to Strategy C for scan_weather_market
            if self._strategy_c is not None and all_stds:
                self._strategy_c._ensemble_stds = all_stds
                logger.info(
                    "c1f_ensemble_stds_injected",
                    cities=len(all_stds),
                )

            # Store today's ensemble stats to DB for accumulation
            try:
                await self._store_ensemble_to_db(session, ensemble_stats)
            except Exception as e:
                logger.warning("ensemble_db_store_failed", error=str(e))

        except Exception as e:
            logger.warning("c1f_ensemble_init_failed", error=str(e))

        return ensemble_stats

    async def _store_ensemble_to_db(
        self, session: Any, ensemble_stats: dict[str, dict[str, float]],
    ) -> None:
        """Store today's ensemble stats to PostgreSQL for accumulation."""
        from datetime import date as _date
        from sqlalchemy import text

        from arbo.utils.db import get_session_factory

        target_date = _date.today().isoformat()
        factory = get_session_factory()
        async with factory() as sess:
            for city, stats in ensemble_stats.items():
                await sess.execute(
                    text("""
                        INSERT INTO ensemble_stats
                            (city, target_date, ensemble_mean, ensemble_std,
                             ensemble_min, ensemble_max, n_members)
                        VALUES (:city, :date, :mean, :std, :min, :max, :n)
                        ON CONFLICT (city, target_date) DO UPDATE SET
                            ensemble_mean = :mean, ensemble_std = :std,
                            ensemble_min = :min, ensemble_max = :max, n_members = :n
                    """),
                    {
                        "city": city, "date": target_date,
                        "mean": stats["mean"], "std": stats["std"],
                        "min": stats["min"], "max": stats["max"],
                        "n": stats["n_members"],
                    },
                )
            await sess.commit()
            logger.info("ensemble_stats_stored_to_db", target_date=target_date, cities=len(ensemble_stats))

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

        cfg = self._config
        return SlackBot(
            bot_token=cfg.slack_bot_token,
            app_token=cfg.slack_app_token,
            channel_id=cfg.slack_channel_id,
            get_status_fn=self._get_status_for_slack,
            get_pnl_fn=self._get_pnl_for_slack,
            shutdown_fn=self._slack_shutdown,
            daily_brief_channel_id=cfg.slack_daily_brief_channel_id,
            review_queue_channel_id=cfg.slack_review_queue_channel_id,
            weekly_report_channel_id=cfg.slack_weekly_report_channel_id,
            get_cryptoarb_fn=self._get_cryptoarb_for_slack,
            get_skinny_fn=self._get_skinny_for_slack,
        )

    async def _init_web_dashboard(self) -> Any:
        from arbo.dashboard.web import create_app

        return create_app(self)

    def _sync_risk_from_positions(self) -> None:
        """Sync risk manager state from open positions after restart.

        Uses restore_position() to update both global exposure tracking
        (open_positions_value, market_positions, category_exposure) and
        per-strategy state (deployed, position_count).
        """
        if self._paper_engine is None or self._risk_manager is None:
            return
        # Strategy → category mapping for exposure tracking
        _strategy_category = {"C": "weather", "C2": "weather", "A": "crypto", "B": "crypto"}
        for pos in self._paper_engine.open_positions:
            strategy = getattr(pos, "strategy", "")
            category = _strategy_category.get(strategy, "other")
            self._risk_manager.restore_position(
                market_id=pos.market_condition_id,
                size=pos.size,
                strategy=strategy,
                market_category=category,
            )

    async def _restore_pnl_from_db(self) -> None:
        """Restore realized P&L from paper_trades into risk manager on startup.

        Queries resolved trades (won/lost) from DB, groups by strategy,
        and sets weekly_pnl and total_pnl on each StrategyState.
        """
        if self._risk_manager is None:
            return
        try:
            import sqlalchemy as sa

            from arbo.utils.db import PaperTrade, get_session_factory

            factory = get_session_factory()
            async with factory() as session:
                # Total P&L per strategy (all resolved trades, excluding pre-validation)
                _no_preval = sa.or_(
                    PaperTrade.notes.is_(None),
                    PaperTrade.notes != "pre-validation",
                )
                result = await session.execute(
                    sa.select(
                        PaperTrade.strategy,
                        sa.func.coalesce(sa.func.sum(PaperTrade.actual_pnl), 0),
                    )
                    .where(PaperTrade.status.in_(["won", "lost"]))
                    .where(PaperTrade.strategy.isnot(None))
                    .where(_no_preval)
                    .group_by(PaperTrade.strategy)
                )
                for row in result.all():
                    strategy = row[0]
                    total_pnl = row[1]
                    ss = self._risk_manager.get_strategy_state(strategy)
                    if ss is not None:
                        ss.total_pnl = Decimal(str(total_pnl))

                # Weekly P&L per strategy (resolved this ISO week)
                from datetime import UTC, datetime, timedelta

                now = datetime.now(UTC)
                monday = now - timedelta(days=now.weekday())
                week_start = monday.replace(hour=0, minute=0, second=0, microsecond=0)

                result = await session.execute(
                    sa.select(
                        PaperTrade.strategy,
                        sa.func.coalesce(sa.func.sum(PaperTrade.actual_pnl), 0),
                    )
                    .where(PaperTrade.status.in_(["won", "lost"]))
                    .where(PaperTrade.strategy.isnot(None))
                    .where(_no_preval)
                    .where(PaperTrade.resolved_at >= week_start)
                    .group_by(PaperTrade.strategy)
                )
                for row in result.all():
                    strategy = row[0]
                    weekly_pnl = row[1]
                    ss = self._risk_manager.get_strategy_state(strategy)
                    if ss is not None:
                        ss.weekly_pnl = Decimal(str(weekly_pnl))

                # Also restore global daily/weekly P&L
                today = now.replace(hour=0, minute=0, second=0, microsecond=0)
                result = await session.execute(
                    sa.select(
                        sa.func.coalesce(sa.func.sum(PaperTrade.actual_pnl), 0),
                    )
                    .where(PaperTrade.status.in_(["won", "lost"]))
                    .where(_no_preval)
                    .where(PaperTrade.resolved_at >= today)
                )
                daily_total = result.scalar()
                if daily_total:
                    self._risk_manager._state.daily_pnl = Decimal(str(daily_total))

                result = await session.execute(
                    sa.select(
                        sa.func.coalesce(sa.func.sum(PaperTrade.actual_pnl), 0),
                    )
                    .where(PaperTrade.status.in_(["won", "lost"]))
                    .where(_no_preval)
                    .where(PaperTrade.resolved_at >= week_start)
                )
                weekly_total = result.scalar()
                if weekly_total:
                    self._risk_manager._state.weekly_pnl = Decimal(str(weekly_total))

            logger.info(
                "pnl_restored_from_db",
                strategies={
                    s: str(ss.total_pnl)
                    for s, ss in self._risk_manager._state.strategies.items()
                },
            )
        except Exception as e:
            logger.warning("pnl_restore_failed", error=str(e))

    # ------------------------------------------------------------------
    # Slack callbacks
    # ------------------------------------------------------------------

    async def _get_status_for_slack(self) -> dict[str, Any]:
        """Build system status dict for /status command."""
        active = sum(
            1
            for s in self._tasks.values()
            if s.enabled and not s.permanent_stop and s.task and not s.task.done()
        )
        balance = str(self._paper_engine.balance) if self._paper_engine else "N/A"
        positions = len(self._paper_engine.open_positions) if self._paper_engine else 0
        uptime = int(time.monotonic() - self._start_time) if self._start_time else 0

        status: dict[str, Any] = {
            "mode": self._mode,
            "uptime_s": uptime,
            "layers_active": active,
            "layers_total": len(self._tasks),
            "balance": balance,
            "open_positions": positions,
        }

        # Append CryptoArb summary if available
        ca = await self._get_cryptoarb_for_slack()
        if ca is not None:
            status["cryptoarb"] = ca

        return status

    async def _get_pnl_for_slack(self) -> dict[str, Any]:
        """Build P&L dict for /pnl command."""
        if self._paper_engine is None:
            return {}
        return self._paper_engine.get_stats()

    async def _slack_shutdown(self) -> None:
        """Emergency shutdown triggered via /kill."""
        logger.critical("emergency_shutdown_via_slack")
        if self._slack_bot is not None:
            await self._slack_bot.send_alert("Emergency shutdown triggered via `/kill` command")
        self._shutdown_event.set()

    # ------------------------------------------------------------------
    # CryptoArb integration (file-based bridge via state.json)
    # ------------------------------------------------------------------

    def _read_nightcap_state(self) -> dict[str, Any] | None:
        """Read Nightcap dashboard_state.json from disk."""
        path = os.getenv(
            "NIGHTCAP_STATE_PATH",
            "/opt/nightcap/data/dashboard_state.json",
        )
        try:
            with open(path) as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError, OSError) as e:
            logger.debug("nightcap_state_read_error", error=str(e), path=path)
            return None

    def _read_cryptoarb_state(self) -> dict[str, Any] | None:
        """Read CryptoArb state.json from disk. Returns raw dict or None."""
        path = os.getenv(
            "CRYPTOARB_STATE_PATH",
            "/opt/cryptoarb/production_data/state.json",
        )
        try:
            with open(path) as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError, OSError) as e:
            logger.debug("cryptoarb_state_read_error", error=str(e), path=path)
            return None

    async def _get_cryptoarb_for_slack(self) -> dict[str, Any] | None:
        """Build CryptoArb summary dict for Slack from state.json."""
        raw = self._read_cryptoarb_state()
        if raw is None:
            return None

        equity = raw.get("portfolio_equity", 1.0)
        equity_hist = raw.get("equity_history", [])
        peak = max(equity_hist) if equity_hist else equity
        drawdown_pct = ((peak - equity) / peak * 100) if peak > 0 else 0.0

        trade_returns = raw.get("trade_returns", [])
        total_trades = raw.get("total_trades", 0)
        wins = sum(1 for r in trade_returns if r > 0)
        win_rate = (wins / len(trade_returns) * 100) if trade_returns else 0.0

        pairs_raw = raw.get("pairs", {})
        pairs: dict[str, dict[str, Any]] = {}
        for name, info in pairs_raw.items():
            z_hist = info.get("z_history", [])
            pairs[name] = {
                "position": info.get("position", 0),
                "z": z_hist[-1] if z_hist else 0.0,
            }

        return {
            "equity": equity,
            "drawdown_pct": drawdown_pct,
            "trades": total_trades,
            "win_rate": win_rate,
            "mode": "live" if raw.get("last_signal_eval") else "idle",
            "last_signal_eval": raw.get("last_signal_eval", ""),
            "pairs": pairs,
            "alerts": raw.get("alerts", []),
        }

    async def _get_skinny_for_slack(self) -> dict[str, Any] | None:
        """Fetch Skinny CS2 trading status via HTTP from Skinny API."""
        import os

        import aiohttp

        skinny_url = os.getenv("SKINNY_API_URL", "")
        skinny_key = os.getenv("SKINNY_API_KEY", "")
        if not skinny_url:
            return None

        try:
            headers = {"X-API-Key": skinny_key} if skinny_key else {}
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{skinny_url}/api/portfolio", headers=headers, timeout=aiohttp.ClientTimeout(total=5)
                ) as resp:
                    if resp.status != 200:
                        return None
                    portfolio = await resp.json()

                async with session.get(
                    f"{skinny_url}/api/status", headers=headers, timeout=aiohttp.ClientTimeout(total=5)
                ) as resp:
                    status = await resp.json() if resp.status == 200 else {}

                async with session.get(
                    f"{skinny_url}/api/risk", headers=headers, timeout=aiohttp.ClientTimeout(total=5)
                ) as resp:
                    risk = await resp.json() if resp.status == 200 else {}

            return {"portfolio": portfolio, "status": status, "risk": risk}
        except Exception as e:
            logger.debug("skinny_api_error", error=str(e))
            return None

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
        if self._strategy_c2 is not None:
            task_defs.append(("strategy_C2", self._run_strategy_c2, 300))       # entry scan every 5 min
            task_defs.append(("C2_exit_monitor", self._run_c2_exit_check, 60))  # exit check every 60s
        if self._strategy_b2 is not None:
            task_defs.append(("strategy_B2", self._run_strategy_b2, 60))        # entry scan every 60s (faster)
            task_defs.append(("B2_exit_monitor", self._run_b2_exit_check, 30))  # exit check every 30s
        if self._strategy_b3 is not None:
            task_defs.append(("strategy_B3", self._run_strategy_b3, 15))        # entry scan every 15s (5-min windows)
            task_defs.append(("B3_exit_monitor", self._run_b3_exit_check, 10))  # exit check every 10s (fast exits)

        # Auto-redeem resolved positions (gasless, every 30 min)
        task_defs.append(("auto_redeem", self._run_auto_redeem, 1800))

        for name, coro_factory, interval in task_defs:
            state = TaskState(name=name)
            self._tasks[name] = state
            state.task = asyncio.create_task(
                self._run_task_loop(name, coro_factory, interval),
                name=f"rdh_{name}",
            )

    def _start_internal_tasks(self) -> None:
        """Start infrastructure tasks."""
        # Start OrderFlowMonitor polling (Strategy A on-chain flow)
        if self._order_flow_monitor is not None:
            self._internal_tasks.append(
                asyncio.create_task(
                    self._order_flow_monitor.start(), name="order_flow_monitor"
                )
            )
        self._internal_tasks.append(
            asyncio.create_task(self._health_monitor(), name="health_monitor")
        )
        # GEFS ensemble download (background — takes ~3 min, must not block startup)
        self._internal_tasks.append(
            asyncio.create_task(self._init_gefs_background(), name="gefs_ensemble")
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
        if self._santiment is not None or self._coingecko is not None:
            self._internal_tasks.append(
                asyncio.create_task(
                    self._social_momentum_collector(), name="social_momentum_collector"
                )
            )
        if self._slack_bot is not None:
            self._internal_tasks.append(
                asyncio.create_task(self._slack_bot.start(), name="slack_bot")
            )
        self._internal_tasks.append(
            asyncio.create_task(self._health_check_scheduler(), name="health_check")
        )
        self._internal_tasks.append(
            asyncio.create_task(self._morning_briefing_scheduler(), name="morning_briefing")
        )
        self._internal_tasks.append(
            asyncio.create_task(self._anomaly_check_scheduler(), name="anomaly_check")
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
                    logger.error(
                        "task_error_threshold", task=name, count=len(state.error_timestamps)
                    )
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
            await self._save_signals_to_db(trades, strategy="A")
            await self._paper_engine.sync_positions_to_db()

        # Check exits — Strategy A holds NO tokens, needs NO prices
        prices = self._get_current_prices(price_type="no")
        exits = self._strategy_a.check_exits(prices)
        if exits:
            logger.info("strategy_a_exits", count=len(exits))

    async def _run_strategy_b(self) -> None:
        """Run Strategy B: Reflexivity Surfer poll cycle + exit checks.

        Feeds latest divergence signals from Santiment + CoinGecko before
        each poll cycle. Strategy B's data_source_live guard will return []
        if signals are empty or stale (> 7h old).
        """
        if self._strategy_b is None:
            return

        # Feed latest divergence signals before poll cycle
        self._strategy_b.update_signals(self._latest_divergence_signals)

        trades = await self._strategy_b.poll_cycle(self._markets)
        if trades:
            logger.info("strategy_b_trades", count=len(trades))
            await self._save_trades_to_db(trades)
            await self._save_signals_to_db(trades, strategy="B")
            await self._paper_engine.sync_positions_to_db()

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
            await self._save_signals_to_db(trades, strategy="C")
            await self._paper_engine.sync_positions_to_db()

            # Register new trades with shadow exit tracker
            if self._shadow_exit_tracker is not None:
                for signal in trades:
                    token_id = (
                        signal.market.token_id_yes
                        if signal.direction == "BUY_YES"
                        else signal.market.token_id_no
                    )
                    self._shadow_exit_tracker.register_position(signal, token_id)

        # Shadow exit check: evaluate all open C positions
        if (
            self._shadow_exit_tracker is not None
            and self._paper_engine is not None
            and self._strategy_c is not None
        ):
            # Get current CLOB prices for open positions
            c_positions = [
                p for p in self._paper_engine.open_positions
                if getattr(p, "strategy", "") == "C"
            ]
            if c_positions and self._orderbook_provider is not None:
                from arbo.connectors.orderbook_provider import OrderbookProvider

                token_ids = [p.token_id for p in c_positions]
                snapshots = await self._orderbook_provider.get_snapshots_batch(
                    token_ids, neg_risk=True
                )
                current_prices: dict[str, Decimal] = {}
                for tid, snap in snapshots.items():
                    if snap is not None and snap.best_bid is not None:
                        current_prices[tid] = snap.best_bid

                shadow_exits = self._shadow_exit_tracker.check_exits(
                    c_positions,
                    current_prices,
                    self._strategy_c._forecasts,
                )
                if shadow_exits:
                    logger.info(
                        "shadow_exits_triggered",
                        count=len(shadow_exits),
                        tokens=[e.token_id[:20] for e in shadow_exits],
                    )

    async def _run_strategy_c2(self) -> None:
        """Run Strategy C2: EMOS + Edge Exit Fusion — entry scan only.

        Fetches forecasts, scans markets, places new paper trades.
        Exit checks run separately via _run_c2_exit_check (every 60s).
        """
        if self._strategy_c2 is None:
            return
        trades = await self._strategy_c2.poll_cycle(self._markets)
        if trades:
            logger.info("strategy_c2_trades", count=len(trades))
            await self._save_trades_to_db(trades)
            await self._save_signals_to_db(trades, strategy="C2")
            await self._paper_engine.sync_positions_to_db()

            # Slack notification for LIVE entries only
            if self._strategy_c2._execution_mode == "live" and self._slack_bot:
                for sig in trades:
                    city = sig.market.city.value if sig.market.city else "?"
                    try:
                        await self._slack_bot._post(
                            "C0AP2QLLM2N",
                            text=f":arrow_right: *C2 LIVE BUY* — {city.upper()} {sig.direction}\nEdge: {sig.edge:.1%}  |  Price: ${sig.market.market_price:.4f}",
                        )
                    except Exception:
                        pass

    async def _run_c2_exit_check(self) -> None:
        """Check C2 positions for exit triggers every 60 seconds.

        Lightweight: fetches CLOB bid prices and checks thresholds.
        Runs independently from the entry scan (every 30 min).
        """
        if (
            self._strategy_c2 is None
            or self._paper_engine is None
            or self._orderbook_provider is None
        ):
            return

        c2_positions = [
            p for p in self._paper_engine.open_positions
            if getattr(p, "strategy", "") == "C2"
        ]
        if not c2_positions:
            return

        token_ids = [p.token_id for p in c2_positions]
        snapshots = await self._orderbook_provider.get_snapshots_batch(
            token_ids, neg_risk=True
        )
        current_prices: dict[str, Decimal] = {}
        for tid, snap in snapshots.items():
            if snap is not None and snap.best_bid is not None:
                current_prices[tid] = snap.best_bid

        # Check for new exit triggers
        exits = await self._strategy_c2.check_exits(
            current_prices, self._strategy_c2._forecasts,
        )

        is_live = self._strategy_c2._execution_mode == "live"

        # Register new exits with ExitManager (live) or execute immediately (paper)
        for token_id, exit_reason in (exits or []):
            bid_price = current_prices.get(token_id)
            if bid_price is None:
                continue

            pos = next(
                (p for p in c2_positions if p.token_id == token_id),
                None,
            )
            if pos is None:
                continue

            if is_live and self._exit_manager:
                # Register with ExitManager — it will keep trying until all sold
                pos_data = self._strategy_c2._open_positions.get(token_id, {})
                shares = self._strategy_c2._live_executor._shares_owned.get(token_id, 0)
                if shares > 0:
                    self._exit_manager.register_exit(
                        token_id=token_id,
                        city=pos_data.get("city", "?"),
                        exit_reason=exit_reason,
                        shares=shares,
                        entry_price=pos_data.get("entry_price", float(pos.avg_price)),
                    )
            else:
                # Paper: instant sell
                pnl = self._paper_engine.sell_position(
                    token_id=token_id, sell_price=bid_price, exit_reason=exit_reason,
                )
                self._risk_manager.post_trade_update(
                    pos.market_condition_id, "weather", pos.size, pnl=pnl,
                )
                self._risk_manager.strategy_post_trade("C2", pos.size, pnl=pnl)
                await self._paper_engine.update_resolved_trades_in_db(
                    token_id=token_id, pnl=pnl,
                    exit_price=bid_price, exit_reason=exit_reason,
                )

        # Process pending exits (ExitManager keeps trying for live)
        if is_live and self._exit_manager and self._exit_manager.has_pending:
            completed = await self._exit_manager.process_exits()
            for comp in completed:
                # Record completed live exit
                token_id = comp["token_id"]
                pnl_val = comp["pnl"]
                city = comp["city"]

                # Find paper position to close
                pos = next(
                    (p for p in self._paper_engine.open_positions
                     if p.token_id == token_id and getattr(p, "strategy", "") == "C2"),
                    None,
                )
                if pos:
                    sell_price = Decimal(str(comp["avg_sell_price"])) if comp["avg_sell_price"] > 0 else current_prices.get(token_id, Decimal("0"))
                    pnl = self._paper_engine.sell_position(
                        token_id=token_id, sell_price=sell_price,
                        exit_reason=comp["exit_reason"],
                    )
                    self._risk_manager.post_trade_update(
                        pos.market_condition_id, "weather", pos.size, pnl=pnl,
                    )
                    self._risk_manager.strategy_post_trade("C2", pos.size, pnl=pnl)
                    await self._paper_engine.update_resolved_trades_in_db(
                        token_id=token_id, pnl=pnl,
                        exit_price=sell_price, exit_reason=comp["exit_reason"],
                    )

                # Slack notification
                await self._notify_c2_trade_close(
                    pos, comp["exit_reason"], pnl_val,
                    comp["avg_sell_price"], True,
                )

                logger.info(
                    "c2_live_exit_complete",
                    city=city,
                    shares_sold=comp["shares_sold"],
                    pnl=round(pnl_val, 2),
                    reason=comp["exit_reason"],
                )

        if exits:
            logger.info(
                "c2_exits_processed",
                new_triggers=len(exits),
                pending=len(self._exit_manager.pending_exits) if self._exit_manager else 0,
                tokens=[t[:20] for t, _ in exits],
        )
        await self._paper_engine.sync_positions_to_db()

    # ═══════════════════════════════════════════════════════════════════════
    # STRATEGY B2: Crypto Price Edge
    # ═══════════════════════════════════════════════════════════════════════

    async def _run_strategy_b2(self) -> None:
        """B2 entry scan every 60 seconds.

        Gets crypto markets from discovery, scans with volatility model,
        applies quality gate, and executes trades.
        """
        if self._strategy_b2 is None or self._discovery is None:
            return

        markets = self._discovery.get_by_category("crypto")
        if not markets:
            return

        trades = await self._strategy_b2.poll_cycle(markets)

        if trades:
            await self._save_trades_to_db(trades)
            for sig in trades:
                logger.info(
                    "b2_trade_executed",
                    asset=sig.asset,
                    strike=float(sig.strike),
                    edge=f"{sig.edge:.3f}",
                    exchange_price=f"${sig.current_exchange_price:.0f}",
                )
            await self._paper_engine.sync_positions_to_db()

    async def _run_b2_exit_check(self) -> None:
        """Check B2 positions for exit triggers every 30 seconds.

        Fetches CLOB prices and recomputes probability with latest
        Binance exchange price.
        """
        if (
            self._strategy_b2 is None
            or self._paper_engine is None
            or self._orderbook_provider is None
        ):
            return

        b2_positions = [
            p for p in self._paper_engine.open_positions
            if getattr(p, "strategy", "") == "B2"
        ]
        if not b2_positions:
            return

        token_ids = [p.token_id for p in b2_positions]
        # Use /price endpoint (neg_risk=True forces it) — RFQM pricing
        snapshots = await self._orderbook_provider.get_snapshots_batch(
            token_ids, neg_risk=True
        )
        current_prices: dict[str, float] = {}  # midpoint for edge check
        sell_prices: dict[str, float] = {}     # BUY price = what seller receives
        for tid, snap in snapshots.items():
            if snap is None:
                continue
            best_bid = float(snap.best_bid) if snap.best_bid else 0
            best_ask = float(snap.best_ask) if snap.best_ask else 0
            if best_bid > 0.001 and best_ask > 0.001:
                current_prices[tid] = (best_bid + best_ask) / 2
                sell_prices[tid] = best_bid  # Sell at BUY price (taker sell)

        # Check for exit triggers
        exits = await self._strategy_b2.check_exits(current_prices)

        is_live = self._strategy_b2._execution_mode == "live"

        for token_id, exit_reason in (exits or []):
            bid_price = sell_prices.get(token_id)
            if bid_price is None or bid_price < 0.02:
                continue

            pos = next(
                (p for p in b2_positions if p.token_id == token_id),
                None,
            )
            if pos is None:
                continue

            if is_live and self._exit_manager:
                b2_pos = self._strategy_b2._open_positions.get(token_id)
                shares = self._strategy_b2._live_executor._shares_owned.get(token_id, 0) if self._strategy_b2._live_executor else 0
                if shares > 0:
                    from arbo.core.exit_manager import PendingExit

                    self._exit_manager.register_exit(
                        token_id=token_id,
                        city=b2_pos.asset if b2_pos else "?",
                        exit_reason=exit_reason,
                        shares=shares,
                        entry_price=float(pos.avg_price),
                    )
                    # Set non-NegRisk flag on the PendingExit
                    pe = self._exit_manager._pending.get(token_id)
                    if pe:
                        pe.neg_risk = False
                        pe.label = f"{b2_pos.asset}_{int(b2_pos.strike)}" if b2_pos else ""
            else:
                # Paper: instant sell
                from decimal import Decimal as D
                pnl = self._paper_engine.sell_position(
                    token_id=token_id,
                    sell_price=D(str(bid_price)),
                    exit_reason=exit_reason,
                )
                self._risk_manager.strategy_post_trade("B2", pos.size, pnl=pnl)
                await self._paper_engine.update_resolved_trades_in_db(
                    token_id=token_id, pnl=pnl,
                    exit_price=D(str(bid_price)), exit_reason=exit_reason,
                )

        # Process pending exits (shared ExitManager)
        if is_live and self._exit_manager and self._exit_manager.has_pending:
            await self._exit_manager.process_exits()

        if exits:
            logger.info(
                "b2_exits_processed",
                new_triggers=len(exits),
                tokens=[t[:20] for t, _ in exits],
            )
            await self._paper_engine.sync_positions_to_db()

    # ═══════════════════════════════════════════════════════════════════════
    # STRATEGY B3: Binance Oracle Scalper
    # ═══════════════════════════════════════════════════════════════════════

    async def _run_strategy_b3(self) -> None:
        """B3 entry scan every 15 seconds.

        Fetches BTC Up/Down events, checks for entry signals at minute 2,
        and executes PostOnly trades.
        """
        if self._strategy_b3 is None:
            return

        trades = await self._strategy_b3.poll_cycle()

        if trades:
            for sig in trades:
                direction = "UP" if sig.direction == 1 else "DOWN"
                logger.info(
                    "b3_trade_executed",
                    direction=direction,
                    edge=f"{sig.edge:.3f}",
                    btc=f"${sig.btc_now:.0f}",
                )
                # Save trade to DB (find matching in-memory trade by condition_id)
                for t in reversed(self._paper_engine.trade_history):
                    if t.market_condition_id == sig.condition_id and t.status.value == "open":
                        try:
                            await self._paper_engine.save_trade_to_db(t)
                        except Exception as e:
                            logger.warning("b3_save_trade_db_error", error=str(e))
                        break
                # Sync B3 market to DB so dashboard shows question text
                await self._sync_b3_market_to_db(sig)
                # Slack notification
                await self._notify_b3_entry(sig)
            await self._paper_engine.sync_positions_to_db()

    async def _run_b3_exit_check(self) -> None:
        """Check B3 positions for exit triggers every 10 seconds.

        Recomputes fair value with latest Binance price and checks
        profit target, stop loss, time limit, and edge exit.
        Handles both paper and live sells (dual mode).
        """
        if self._strategy_b3 is None or self._paper_engine is None:
            return

        exits = await self._strategy_b3.check_exits()

        for token_id, exit_reason, exit_price, live_shares, b3_direction, live_entry_price in (exits or []):
            pos = next(
                (p for p in self._paper_engine.open_positions
                 if p.token_id == token_id),
                None,
            )
            if pos is None:
                continue

            from decimal import Decimal as D

            pnl = self._paper_engine.sell_position(
                token_id=token_id,
                sell_price=D(str(round(exit_price, 4))),
                exit_reason=exit_reason,
            )
            self._risk_manager.post_trade_update(
                pos.market_condition_id, "crypto_5min", pos.size, pnl=pnl,
            )
            self._risk_manager.strategy_post_trade("B3", pos.size, pnl=pnl)
            await self._paper_engine.update_resolved_trades_in_db(
                token_id=token_id, pnl=pnl,
                exit_price=D(str(round(exit_price, 4))),
                exit_reason=exit_reason,
            )

            # Live sell (dual mode) — skip resolution (token auto-redeems)
            live_exit_info = None
            if live_shares > 0 and exit_reason != "resolution":
                live_exit_info = await self._strategy_b3.sell_live_position(
                    token_id=token_id,
                    exit_reason=exit_reason,
                    paper_exit_price=exit_price,
                )
            elif live_shares > 0 and exit_reason == "resolution":
                # Resolution: token auto-redeems at exit_price ($1 or $0)
                live_exit_info = {
                    "live_exit_price": exit_price,
                    "live_exit_shares": live_shares,
                    "live_exit_status": "resolution",
                    "live_exit_latency_ms": 0,
                }

            # Store live exit info in trade_details for dashboard
            if live_exit_info or live_shares > 0:
                try:
                    import sqlalchemy as _sa

                    from arbo.utils.db import PaperTrade, get_session_factory
                    live_upd = {
                        "live_exit_price": live_exit_info.get("live_exit_price", 0) if live_exit_info else 0,
                        "live_exit_shares": live_exit_info.get("live_exit_shares", 0) if live_exit_info else 0,
                        "live_exit_status": live_exit_info.get("live_exit_status", "resolution") if live_exit_info else "resolution",
                        "live_exit_latency_ms": live_exit_info.get("live_exit_latency_ms", 0) if live_exit_info else 0,
                    }
                    factory = get_session_factory()
                    async with factory() as session:
                        # Find the trade
                        row = await session.execute(
                            _sa.select(PaperTrade)
                            .where(PaperTrade.token_id == token_id)
                            .where(PaperTrade.strategy == "B3")
                            .order_by(PaperTrade.placed_at.desc())
                            .limit(1)
                        )
                        trade = row.scalar_one_or_none()
                        if trade and trade.trade_details:
                            trade.trade_details = {**trade.trade_details, **live_upd}
                            await session.commit()
                            logger.info("b3_live_exit_saved_to_db", token=token_id[:20])
                        else:
                            logger.warning("b3_live_exit_no_trade", token=token_id[:20])
                except Exception as e:
                    logger.warning("b3_live_exit_db_error", error=str(e))

            # Slack notification
            await self._notify_b3_exit(
                pos, exit_reason, float(pnl), exit_price, live_exit_info,
                b3_direction=b3_direction,
                live_entry_price=live_entry_price,
            )

        if exits:
            logger.info(
                "b3_exits_processed",
                count=len(exits),
                reasons=[r for _, r, _, _, _, _ in exits],
            )
            await self._paper_engine.sync_positions_to_db()

    async def _notify_c2_trade_close(
        self,
        pos: Any,
        exit_reason: str,
        pnl: float,
        exit_price: float,
        is_live: bool,
    ) -> None:
        """Send Slack notification when C2 closes a trade."""
        C2_SLACK_CHANNEL = "C0AP2QLLM2N"
        if self._slack_bot is None:
            return
        try:
            td = self._paper_engine.get_trade_details(pos.token_id) if self._paper_engine else {}
            if not td:
                td = {}
            city = td.get("city", "?")
            direction = td.get("direction", "?")
            entry = float(pos.avg_price)
            size = float(pos.size)

            # Get current C2 state
            ss = self._risk_manager.get_strategy_state("C2") if self._risk_manager else None
            total_pnl = float(ss.total_pnl) if ss else 0
            deployed = float(ss.deployed) if ss else 0
            available = float(ss.available) if ss else 0

            mode = "LIVE" if is_live else "PAPER"
            emoji = ":white_check_mark:" if pnl >= 0 else ":x:"
            pnl_sign = "+" if pnl >= 0 else ""

            text = (
                f"{emoji} *C2 {mode} — {city.upper()}* {direction}\n"
                f"Entry: ${entry:.4f} → Exit: ${exit_price:.4f} ({exit_reason})\n"
                f"Size: ${size:.2f}  |  P&L: *{pnl_sign}${pnl:.2f}*\n"
                f"Balance: ${total_pnl:.2f} total P&L  |  ${deployed:.0f} deployed  |  ${available:.0f} available"
            )
            await self._slack_bot._post(C2_SLACK_CHANNEL, text=text)
        except Exception as e:
            logger.debug("c2_slack_notify_error", error=str(e))

    async def _get_b3_total_pnl_from_db(self) -> float:
        """Get B3 total P&L from DB (source of truth, survives restarts)."""
        try:
            import sqlalchemy as _sa

            from arbo.utils.db import PaperTrade, get_session_factory

            factory = get_session_factory()
            async with factory() as session:
                result = await session.execute(
                    _sa.select(_sa.func.coalesce(_sa.func.sum(PaperTrade.actual_pnl), 0))
                    .where(PaperTrade.strategy == "B3")
                    .where(PaperTrade.status.in_(["won", "lost", "sold"]))
                )
                return float(result.scalar() or 0)
        except Exception:
            ss = self._risk_manager.get_strategy_state("B3") if self._risk_manager else None
            return float(ss.total_pnl) if ss else 0

    async def _notify_b3_entry(self, sig: Any) -> None:
        """Send Slack notification on B3 trade entry.

        Paper notifications → C0APFCD4M9U (B3 paper channel)
        Live notifications  → C0APX4K8Z2N (B3 live channel)
        """
        b3_paper_channel = "C0APFCD4M9U"
        b3_live_channel = "C0APX4K8Z2N"
        if self._slack_bot is None:
            return
        try:
            direction = "UP" if sig.direction == 1 else "DOWN"
            total_pnl = await self._get_b3_total_pnl_from_db()
            ss = self._risk_manager.get_strategy_state("B3") if self._risk_manager else None
            deployed = float(ss.deployed) if ss else 0

            # Compute trade size (same logic as poll_cycle)
            raw_pct = min(0.067, sig.edge * 4.838)
            available = float(ss.allocated - ss.deployed) if ss else 0
            bet_size = min(available * raw_pct, 100.0)

            text = (
                f":zap: *B3 ENTRY — BTC {direction}*\n"
                f"BTC: ${sig.btc_now:,.0f}  |  FV: {sig.entry_price:.3f}  |  Edge: {sig.edge:.1%}\n"
                f"Size: ~${bet_size:.0f}  |  Deployed: ${deployed:.0f}  |  B3 Total: ${total_pnl:+.2f}"
            )
            await self._slack_bot._post(b3_paper_channel, text=text)

            # Send live fill to separate channel
            if self._strategy_b3 and hasattr(self._strategy_b3, '_open_positions'):
                for _tid, bpos in self._strategy_b3._open_positions.items():
                    if bpos.condition_id == sig.condition_id and bpos.live_fill_status:
                        st = bpos.live_fill_status
                        if st in ("filled", "partial") and bpos.live_shares > 0:
                            gap = bpos.live_entry_price - sig.entry_price
                            live_text = (
                                f":zap: *B3 LIVE BUY — BTC {direction}*\n"
                                f"*{st}*  |  "
                                f"{bpos.live_shares} shares @ "
                                f"{bpos.live_entry_price:.3f}  |  "
                                f"${bpos.live_shares * bpos.live_entry_price:.1f}  |  "
                                f"{bpos.live_latency_ms}ms\n"
                                f"Model FV: {sig.entry_price:.3f}  |  "
                                f"Gap: {gap:+.3f}"
                            )
                            await self._slack_bot._post(b3_live_channel, text=live_text)
                        elif st in ("failed", "error", "too_small"):
                            live_text = (
                                f":no_entry: *B3 LIVE BUY SKIP — "
                                f"BTC {direction}*\n"
                                f"Reason: {st}  |  "
                                f"{bpos.live_latency_ms}ms"
                            )
                            await self._slack_bot._post(b3_live_channel, text=live_text)
                        break

        except Exception as e:
            logger.debug("b3_slack_entry_error", error=str(e))

    async def _notify_b3_exit(
        self, pos: Any, exit_reason: str, pnl: float, exit_price: float,
        live_exit_info: dict | None = None,
        b3_direction: int = 0,
        live_entry_price: float = 0.0,
    ) -> None:
        """Send Slack notification on B3 trade exit.

        Paper notifications → C0APFCD4M9U (B3 paper channel)
        Live notifications  → C0APX4K8Z2N (B3 live channel)
        """
        b3_paper_channel = "C0APFCD4M9U"
        b3_live_channel = "C0APX4K8Z2N"
        if self._slack_bot is None:
            return
        try:
            dir_str = "UP" if b3_direction == 1 else "DOWN" if b3_direction == -1 else "?"
            if dir_str == "?":
                td = self._paper_engine.get_trade_details(pos.token_id) if self._paper_engine else {}
                dir_str = (td.get("direction", "?")).upper() if td else "?"
            size = float(pos.size)

            total_pnl = await self._get_b3_total_pnl_from_db()

            emoji = ":white_check_mark:" if pnl >= 0 else ":x:"
            pnl_sign = "+" if pnl >= 0 else ""
            hold_s = ""
            if hasattr(pos, "opened_at") and pos.opened_at:
                from datetime import UTC, datetime
                delta = (datetime.now(UTC) - pos.opened_at).total_seconds()
                hold_s = f"  |  Hold: {delta:.0f}s"

            # Paper exit → paper channel
            text = (
                f"{emoji} *B3 EXIT — BTC {dir_str}* ({exit_reason})\n"
                f"Entry: {float(pos.avg_price):.3f} -> "
                f"Exit: {exit_price:.3f}  |  "
                f"Size: ${size:.0f}{hold_s}\n"
                f"P&L: *{pnl_sign}${pnl:.2f}*  |  "
                f"B3 Total: ${total_pnl:+.2f}"
            )
            await self._slack_bot._post(b3_paper_channel, text=text)

            # Live exit → live channel
            if live_exit_info:
                live_xp = live_exit_info.get("live_exit_price", 0)
                live_xs = live_exit_info.get("live_exit_shares", 0)
                live_status = live_exit_info.get("live_exit_status", "?")
                latency = live_exit_info.get("live_exit_latency_ms", 0)

                # Compute actual live PnL
                live_pnl = 0.0
                if live_entry_price > 0 and live_xp > 0 and live_xs > 0:
                    live_pnl = (live_xp - live_entry_price) * live_xs
                live_pnl_s = f"{'+' if live_pnl >= 0 else ''}${live_pnl:.2f}"
                live_emoji = ":white_check_mark:" if live_pnl >= 0 else ":x:"

                if live_status == "resolution":
                    # Auto-resolve: token redeems at $1 or $0
                    live_pnl = (live_xp - live_entry_price) * live_shares if live_entry_price > 0 else 0
                    live_pnl_s = f"{'+' if live_pnl >= 0 else ''}${live_pnl:.2f}"
                    res_emoji = ":white_check_mark:" if live_xp > 0.5 else ":x:"
                    paper_entry = float(pos.avg_price) if pos else 0
                    live_text = (
                        f"{res_emoji} *B3 LIVE RESOLVE — "
                        f"BTC {dir_str}*\n"
                        f"Live:  {live_entry_price:.3f} -> "
                        f"{'$1' if live_xp > 0.5 else '$0'}  |  "
                        f"P&L: *{live_pnl_s}*  |  "
                        f"{live_shares} shares\n"
                        f"Paper: {paper_entry:.3f} -> "
                        f"{'$1' if exit_price > 0.5 else '$0'}  |  "
                        f"P&L: *{pnl_sign}${pnl:.2f}*"
                    )
                elif live_status in ("filled", "partial") and live_xs > 0:
                    paper_entry = float(pos.avg_price) if pos else 0
                    live_text = (
                        f"{live_emoji} *B3 LIVE SELL — BTC {dir_str}*"
                        f" ({exit_reason})\n"
                        f"Live:  {live_entry_price:.3f} -> "
                        f"{live_xp:.3f}  |  "
                        f"P&L: *{live_pnl_s}*  |  "
                        f"{live_xs} shares  |  {latency}ms\n"
                        f"Paper: {paper_entry:.3f} -> "
                        f"{exit_price:.3f}  |  "
                        f"P&L: *{pnl_sign}${pnl:.2f}*"
                    )
                else:
                    live_text = (
                        f":warning: *B3 LIVE SELL FAILED — "
                        f"BTC {dir_str}* ({exit_reason})\n"
                        f"Status: {live_status}  |  "
                        f"{latency}ms\n"
                        f"Entry: {live_entry_price:.3f}  |  "
                        f"{live_shares} shares → will auto-resolve"
                    )
                await self._slack_bot._post(b3_live_channel, text=live_text)

        except Exception as e:
            logger.debug("b3_slack_exit_error", error=str(e))

    async def _run_auto_redeem(self) -> None:
        """Auto-redeem resolved Polymarket positions every 30 min."""
        try:
            from arbo.core.auto_redeem import redeem_resolved_positions
            result = await redeem_resolved_positions()
            if result.get("redeemed", 0) > 0:
                # Notify on Slack
                if self._slack_bot:
                    await self._slack_bot._post(
                        "C0APX4K8Z2N",  # B3 live channel
                        text=(
                            f":moneybag: *AUTO-REDEEM*\n"
                            f"Redeemed {result['redeemed']} positions"
                        ),
                    )
        except Exception as e:
            logger.debug("auto_redeem_task_error", error=str(e))

    async def _sync_b3_market_to_db(self, sig: Any) -> None:
        """Sync B3 market to DB so dashboard shows question text instead of hash."""
        try:
            import sqlalchemy as _sa

            from arbo.utils.db import Market, get_session_factory

            factory = get_session_factory()
            async with factory() as session:
                stmt = (
                    _sa.dialects.postgresql.insert(Market)
                    .values(
                        condition_id=sig.condition_id,
                        question=sig.question,
                        category="crypto",
                    )
                    .on_conflict_do_update(
                        index_elements=["condition_id"],
                        set_={"question": sig.question, "category": "crypto"},
                    )
                )
                await session.execute(stmt)
                await session.commit()
        except Exception as e:
            logger.debug("b3_market_sync_error", error=str(e))

    def _get_current_prices(
        self, price_type: str = "yes"
    ) -> dict[str, Decimal]:
        """Get current prices from discovery cache for exit checks.

        Args:
            price_type: "yes" for YES prices, "no" for NO prices.
        """
        prices: dict[str, Decimal] = {}
        if self._discovery is None:
            return prices
        for market in self._markets:
            if not hasattr(market, "condition_id"):
                continue
            if price_type == "no":
                p = getattr(market, "price_no", None)
            else:
                p = getattr(market, "price_yes", None)
            if p is not None:
                prices[market.condition_id] = p
        return prices

    async def _save_trades_to_db(self, trades: list[Any]) -> None:
        """Best-effort save trade records to database via paper engine.

        Handles both dict-type trades (Strategy A/B) and WeatherSignal
        objects (Strategy C) by extracting condition_id from the appropriate
        location.
        """
        if self._paper_engine is None:
            return
        for trade_item in trades:
            # Extract condition_id from dict, WeatherSignal, or CryptoSignal
            if isinstance(trade_item, dict):
                cond = trade_item.get("condition_id", "")
            elif hasattr(trade_item, "condition_id"):
                # CryptoSignal: condition_id directly on signal
                cond = trade_item.condition_id
            elif hasattr(trade_item, "market"):
                # WeatherSignal: condition_id is at signal.market.condition_id
                cond = getattr(trade_item.market, "condition_id", "")
            else:
                cond = ""

            if not cond:
                continue

            # Find the matching in-memory trade from paper engine
            trade_obj = None
            for t in reversed(self._paper_engine.trade_history):
                if t.market_condition_id == cond and t.status.value == "open":
                    trade_obj = t
                    break
            if trade_obj is not None:
                try:
                    await self._paper_engine.save_trade_to_db(trade_obj)
                except (ValueError, TypeError) as e:
                    logger.warning("save_trade_db_error", error=str(e))

    async def _save_signals_to_db(self, trades: list[Any], strategy: str) -> None:
        """Best-effort save traded signals to the signals DB table.

        Handles both dict-type trades (Strategy A/B) and WeatherSignal
        objects (Strategy C).
        """
        try:
            from arbo.utils.db import Signal as SignalDB
            from arbo.utils.db import get_session_factory

            factory = get_session_factory()
            async with factory() as session:
                for trade_item in trades:
                    if isinstance(trade_item, dict):
                        # Strategy A/B dict format
                        cond = trade_item.get("condition_id", "")
                        direction = trade_item.get("side", "BUY_YES")
                        edge = trade_item.get("edge")
                        details = {
                            "strategy": strategy,
                            "category": "crypto",
                            **{
                                k: v
                                for k, v in trade_item.items()
                                if k not in ("condition_id",)
                            },
                        }
                    elif hasattr(trade_item, "market"):
                        # WeatherSignal
                        cond = trade_item.market.condition_id
                        direction = trade_item.direction
                        edge = trade_item.edge
                        details = {
                            "strategy": strategy,
                            "category": "weather",
                            "city": trade_item.market.city.value,
                            "forecast_temp_c": trade_item.forecast_temp_c,
                            "forecast_prob": round(trade_item.forecast_probability, 4),
                            "market_price": trade_item.market.market_price,
                            "confidence": trade_item.confidence,
                        }
                    else:
                        continue

                    row = SignalDB(
                        layer=0,
                        market_condition_id=cond,
                        direction=direction,
                        edge=Decimal(str(round(edge, 4))) if edge else None,
                        confidence=None,
                        details=details,
                        confluence_score=None,
                    )
                    session.add(row)
                await session.commit()
            logger.info("signals_saved_to_db", count=len(trades), strategy=strategy)
        except Exception as e:
            logger.warning("signals_save_failed", error=str(e))

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

            # Log OrderFlowMonitor stats for Strategy A diagnostics
            if self._order_flow_monitor is not None:
                ofm_stats = self._order_flow_monitor.stats
                logger.info(
                    "order_flow_health",
                    is_healthy=self._order_flow_monitor.is_healthy,
                    total_events=ofm_stats["total_events"],
                    signals_emitted=ofm_stats["signals_emitted"],
                    active_tokens=ofm_stats["active_tokens"],
                    registered_markets=(
                        self._order_flow_monitor.market_tracker.registered_markets
                    ),
                )

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
        """Update open position prices every 5 minutes.

        Priority: CLOB /price (real-time) > Gamma API (cached).
        """
        while not self._shutdown_event.is_set():
            await asyncio.sleep(300)
            if self._paper_engine is None:
                continue
            try:
                updated = 0
                for pos in self._paper_engine.open_positions:
                    new_price = None
                    is_b2 = getattr(pos, "strategy", "") in ("B2", "B3")

                    # All strategies use /price endpoint (neg_risk=True)
                    # Works for both NegRisk weather AND RFQM crypto markets
                    if self._orderbook_provider is not None:
                        try:
                            snap = await self._orderbook_provider.get_snapshot(
                                pos.token_id, neg_risk=True
                            )
                            if snap is not None and snap.midpoint is not None:
                                new_price = snap.midpoint
                        except Exception:
                            pass  # Fall through to Gamma

                    # Fallback: Gamma API cached prices (NOT for B2/B3)
                    if new_price is None and not is_b2 and self._discovery is not None:
                        market = self._discovery.get_by_condition_id(
                            pos.market_condition_id
                        )
                        if market is not None:
                            if (
                                pos.token_id == market.token_id_yes
                                and market.price_yes is not None
                            ):
                                new_price = market.price_yes
                            elif (
                                pos.token_id == market.token_id_no
                                and market.price_no is not None
                            ):
                                new_price = market.price_no

                    if new_price is not None:
                        self._paper_engine.update_position_price(
                            pos.token_id, new_price
                        )
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
                logger.info(
                    "resolution_check_cycle",
                    open_positions=len(positions),
                )
                if not positions:
                    continue

                now = datetime.now(UTC)
                fetched: dict[str, Any] = {}

                for pos in positions:
                  try:
                    cid = pos.market_condition_id
                    strategy = getattr(pos, "strategy", "")

                    # --- Step 1: Get market from cache or Gamma API ---
                    if cid not in fetched:
                        # Try cache first (fast, works for active markets)
                        market = self._discovery.get_by_condition_id(cid)
                        if market is None:
                            # Cache miss → fetch from Gamma API (works for closed too)
                            market = await self._discovery.fetch_by_token_id(
                                pos.token_id
                            )
                        fetched[cid] = market
                    market = fetched[cid]

                    # Skip if end_date in future — but NOT for Strategy C
                    # Strategy C uses METAR target_date (from question),
                    # not Polymarket end_date which is set days after resolution
                    if strategy != "C" and market is not None and market.end_date:
                        try:
                            end_dt = datetime.fromisoformat(
                                market.end_date.replace("Z", "+00:00")
                            )
                            if end_dt > now:
                                continue
                        except (ValueError, TypeError):
                            pass

                    # --- Step 2: Strategy C — METAR resolution ---
                    winning = None
                    if strategy == "C" and self._weather_resolver is not None:
                        # Build market substitute from trade_details if needed
                        resolution_market = market
                        if resolution_market is None:
                            td = self._paper_engine.get_trade_details(pos.token_id)
                            if td and td.get("bucket_text"):
                                from types import SimpleNamespace

                                direction = td.get("direction", "BUY_YES")
                                is_yes = "YES" in direction.upper()
                                resolution_market = SimpleNamespace(
                                    question=td["bucket_text"],
                                    token_id_yes=(
                                        pos.token_id if is_yes else "__no__"
                                    ),
                                    token_id_no=(
                                        pos.token_id if not is_yes else "__no__"
                                    ),
                                )
                                logger.info(
                                    "resolution_using_trade_details",
                                    condition_id=cid[:20],
                                    city=td.get("city"),
                                    direction=direction,
                                )

                        if resolution_market is not None:
                            try:
                                metar_result = (
                                    await self._weather_resolver.check_resolution(
                                        pos, resolution_market
                                    )
                                )
                                if metar_result is not None:
                                    _is_resolved, winning = metar_result
                                    logger.info(
                                        "metar_resolution_used",
                                        condition_id=cid[:20],
                                        winning=winning,
                                    )
                            except Exception as metar_err:
                                logger.warning(
                                    "metar_resolution_error",
                                    error=str(metar_err),
                                )

                    # --- Step 3: Price-based resolution ---
                    if winning is None and market is not None:
                        yes_price = market.price_yes

                        # Method 1: Market officially closed by Polymarket
                        if market.closed and yes_price is not None:
                            yes_won = yes_price > Decimal("0.5")
                            if pos.token_id == market.token_id_yes:
                                winning = yes_won
                            elif pos.token_id == market.token_id_no:
                                winning = not yes_won

                        # Method 2: Price converged (>0.95 or <0.05) after end_date
                        if winning is None and yes_price is not None:
                            end_passed = False
                            if market.end_date:
                                try:
                                    end_dt = datetime.fromisoformat(
                                        market.end_date.replace("Z", "+00:00")
                                    )
                                    end_passed = now > end_dt
                                except (ValueError, TypeError):
                                    pass

                            price_converged_yes = yes_price >= Decimal("0.95")
                            price_converged_no = yes_price <= Decimal("0.05")

                            if end_passed and (
                                price_converged_yes or price_converged_no
                            ):
                                yes_won = price_converged_yes
                                if pos.token_id == market.token_id_yes:
                                    winning = yes_won
                                elif pos.token_id == market.token_id_no:
                                    winning = not yes_won
                                if winning is not None:
                                    logger.info(
                                        "price_convergence_resolution",
                                        condition_id=cid[:20],
                                        yes_price=str(yes_price),
                                        strategy=strategy,
                                    )

                    # --- Step 4: CLOB price fallback (A/B when market unavailable) ---
                    if (
                        winning is None
                        and market is None
                        and strategy in ("A", "B")
                        and self._orderbook_provider is not None
                    ):
                        try:
                            snap = await self._orderbook_provider.get_snapshot(
                                pos.token_id, neg_risk=True
                            )
                            if snap is not None and snap.midpoint is not None:
                                token_price = snap.midpoint
                                if token_price >= Decimal("0.97"):
                                    winning = True  # This token won
                                    logger.info(
                                        "clob_price_resolution",
                                        condition_id=cid[:20],
                                        token_price=str(token_price),
                                        strategy=strategy,
                                        outcome="won",
                                    )
                                elif token_price <= Decimal("0.03"):
                                    winning = False  # This token lost
                                    logger.info(
                                        "clob_price_resolution",
                                        condition_id=cid[:20],
                                        token_price=str(token_price),
                                        strategy=strategy,
                                        outcome="lost",
                                    )
                        except Exception as clob_err:
                            logger.debug(
                                "clob_resolution_error",
                                error=str(clob_err),
                            )

                    if winning is None:
                        continue

                    pnl = self._paper_engine.resolve_market(pos.token_id, winning)

                    # Notify strategy
                    strategy = getattr(pos, "strategy", "")
                    if strategy == "A" and self._strategy_a is not None:
                        self._strategy_a.handle_resolution(cid, pnl)
                    elif strategy == "B" and self._strategy_b is not None:
                        self._strategy_b.handle_resolution(cid, pnl)
                    elif strategy == "C2" and self._strategy_c2 is not None:
                        self._strategy_c2.handle_resolution(cid, pnl)
                    elif strategy == "B2" and self._strategy_b2 is not None:
                        self._strategy_b2.handle_resolution(cid, float(pnl))
                    elif strategy == "B3" and self._strategy_b3 is not None:
                        self._strategy_b3.handle_resolution(cid, float(pnl))
                    elif strategy == "C" and self._strategy_c is not None:
                        self._strategy_c.handle_resolution(cid, pnl)
                        # Shadow exit A/B comparison
                        if self._shadow_exit_tracker is not None:
                            comparison = self._shadow_exit_tracker.resolve(
                                pos.token_id, pnl
                            )
                            if comparison is not None:
                                logger.info(
                                    "shadow_exit_comparison",
                                    city=comparison.city,
                                    saved=comparison.saved,
                                    actual_pnl=str(comparison.actual_pnl),
                                    shadow_pnl=comparison.shadow_exit_pnl,
                                )

                    await self._paper_engine.update_resolved_trades_in_db(
                        pos.token_id, winning=winning, pnl=pnl
                    )

                    logger.info(
                        "position_resolved",
                        condition_id=cid[:20],
                        strategy=strategy,
                        pnl=str(pnl),
                    )

                    # Slack notification
                    if self._slack_bot is not None:
                        market_name = getattr(market, "question", cid[:40]) or cid[:40]
                        result_status = "won" if winning else "lost"
                        try:
                            await self._slack_bot.send_resolution_alert(
                                market=market_name,
                                strategy=strategy or "?",
                                side=pos.side or "?",
                                size=float(getattr(pos, "size", 0)),
                                pnl=float(pnl),
                                status=result_status,
                            )
                        except Exception as slack_err:
                            logger.warning("resolution_slack_error", error=str(slack_err))

                  except Exception as pos_err:
                    logger.error(
                        "resolution_position_error",
                        token_id=getattr(pos, "token_id", "?")[:20],
                        error=str(pos_err),
                    )
                    # Continue to next position — don't let one crash block all

            except Exception as e:
                logger.error("resolution_check_error", error=str(e))

    async def _health_check_scheduler(self) -> None:
        """Run health check every 12 hours and persist results."""
        interval = 12 * 3600  # 12 hours
        # Initial delay: run first check after 1 hour of data
        await asyncio.sleep(3600)
        while not self._shutdown_event.is_set():
            try:
                from arbo.core.health_check import run_health_check, save_health_check

                report = await run_health_check(window_hours=12)
                await save_health_check(report)
                logger.info(
                    "health_check_complete",
                    verdict=report.verdict,
                    notes=report.notes[:3] if report.notes else [],
                )

                # Slack alert if not OK
                if report.verdict != "ok" and self._slack_bot is not None:
                    verdict_emoji = (
                        ":warning:" if report.verdict == "needs_attention" else ":red_circle:"
                    )
                    msg = (
                        f"{verdict_emoji} *Health Check: {report.verdict.upper()}*\n"
                        + "\n".join(f"• {n}" for n in report.notes)
                    )
                    await self._slack_bot.send_message(msg, channel="review-queue")

            except Exception as e:
                logger.error("health_check_scheduler_error", error=str(e))

            await asyncio.sleep(interval)

    async def _daily_report_scheduler(self) -> None:
        """Daily report at configured UTC hour and minute."""
        target_hour = self._orch_cfg.daily_report_hour_utc
        target_minute = self._orch_cfg.daily_report_minute_utc

        while not self._shutdown_event.is_set():
            now = datetime.now(UTC)
            if now.hour == target_hour and target_minute <= now.minute < target_minute + 5:
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
        """Daily report — now redirects to consolidated morning briefing.

        Previously sent separate Arbo report + CryptoArb message.
        Now everything is in _send_morning_briefing() as one message.
        """
        # Skip — morning briefing handles everything
        logger.debug("daily_report_skipped_use_morning_briefing")

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
    # Social momentum collector (B2-14: Santiment + CoinGecko, 4x daily)
    # ------------------------------------------------------------------

    async def _social_momentum_collector(self) -> None:
        """Fetch Santiment + CoinGecko data, store in DB, run divergence calc.

        Runs every 6 hours (4x daily): 00:00, 06:00, 12:00, 18:00 UTC.
        Sequence: fetch on-chain (Santiment) + market (CoinGecko) → merge → store → divergence.
        """
        interval = 21600  # 6 hours in seconds

        # Coins to track (top 20 by market cap)
        from arbo.connectors.santiment_client import SYMBOL_TO_SLUG

        symbols = list(SYMBOL_TO_SLUG.keys())

        while not self._shutdown_event.is_set():
            try:
                # 1. Fetch CoinGecko market data (bulk — 1 API call)
                cg_data: dict[str, Any] = {}
                if self._coingecko is not None:
                    markets = await self._coingecko.get_markets_bulk(per_page=100)
                    for coin in markets:
                        cg_data[coin.symbol.upper()] = coin
                    logger.info(
                        "coingecko_fetched",
                        coins=len(markets),
                        **self._coingecko.usage_stats,
                    )

                # 2. Fetch Santiment on-chain metrics (batch — 3 API calls)
                san_data: dict[str, dict[str, float]] = {}
                if self._santiment is not None:
                    from arbo.connectors.santiment_client import SantimentClient

                    slugs = SantimentClient.slugs_for_symbols(symbols)
                    all_metrics = await self._santiment.get_all_metrics(slugs, days=2)

                    # Reverse map slug → symbol for merging
                    slug_to_sym = {v: k for k, v in SYMBOL_TO_SLUG.items()}
                    for slug, metrics in all_metrics.items():
                        sym = slug_to_sym.get(slug, "")
                        if not sym:
                            continue
                        san_data[sym] = {
                            "daily_active_addresses": (
                                metrics["daily_active_addresses"][-1].value
                                if metrics["daily_active_addresses"]
                                else 0.0
                            ),
                            "dev_activity": (
                                metrics["dev_activity"][-1].value
                                if metrics["dev_activity"]
                                else 0.0
                            ),
                            "transactions_count": (
                                metrics["transaction_volume"][-1].value
                                if metrics["transaction_volume"]
                                else 0.0
                            ),
                        }
                    logger.info(
                        "santiment_fetched",
                        coins=len(san_data),
                        **self._santiment.usage_stats,
                    )

                if not cg_data and not san_data:
                    logger.warning("social_momentum_no_data")
                    await asyncio.sleep(interval)
                    continue

                # 3. Merge and store in DB
                await self._store_social_momentum_v2(symbols, san_data, cg_data)

                # 4. Run divergence calculation
                await self._run_divergence_calc()

            except Exception as e:
                logger.error("social_momentum_collector_error", error=str(e))

            # Sleep until next cycle
            for _ in range(interval):
                if self._shutdown_event.is_set():
                    return
                await asyncio.sleep(1)

    async def _store_social_momentum_v2(
        self,
        symbols: list[str],
        san_data: dict[str, dict[str, float]],
        cg_data: dict[str, Any],
    ) -> None:
        """Merge Santiment + CoinGecko data and store in social_momentum_v2."""
        try:
            from arbo.connectors.coingecko_client import SYMBOL_TO_COINGECKO_ID
            from arbo.connectors.santiment_client import SYMBOL_TO_SLUG
            from arbo.utils.db import SocialMomentumV2, get_session_factory

            factory = get_session_factory()
            count = 0
            async with factory() as session:
                for sym in symbols:
                    slug = SYMBOL_TO_SLUG.get(sym, "")
                    cg_id = SYMBOL_TO_COINGECKO_ID.get(sym, "")
                    san = san_data.get(sym, {})
                    cg = cg_data.get(sym)

                    # Skip if no data at all
                    if not san and cg is None:
                        continue

                    row = SocialMomentumV2(
                        symbol=sym,
                        slug=slug,
                        coingecko_id=cg_id or None,
                        # Santiment metrics
                        daily_active_addresses=san.get("daily_active_addresses"),
                        transactions_count=san.get("transactions_count"),
                        dev_activity=san.get("dev_activity"),
                        # CoinGecko market metrics
                        price=cg.current_price if cg else None,
                        market_cap=cg.market_cap if cg else None,
                        volume_24h=cg.total_volume if cg else None,
                        price_change_24h=(cg.price_change_percentage_24h if cg else None),
                        price_change_7d=(cg.price_change_percentage_7d if cg else None),
                        price_change_30d=(cg.price_change_percentage_30d if cg else None),
                        # CoinGecko community (populated if detail fetched)
                        twitter_followers=None,
                        reddit_subscribers=None,
                        source="santiment+coingecko",
                    )
                    session.add(row)
                    count += 1
                await session.commit()
            logger.info("social_momentum_v2_stored", count=count)
        except Exception as e:
            logger.warning("social_momentum_store_error", error=str(e))

    async def _run_divergence_calc(self) -> None:
        """Load recent snapshots from DB, run divergence calculator."""
        if self._social_divergence is None:
            return

        try:
            from sqlalchemy import text

            from arbo.strategies.social_divergence import MomentumSnapshot
            from arbo.utils.db import get_session_factory

            factory = get_session_factory()
            async with factory() as session:
                # Load latest 4 snapshots per symbol (24h window)
                result = await session.execute(text("""
                        SELECT symbol, daily_active_addresses, transactions_count,
                               dev_activity, volume_24h, price, price_change_24h,
                               captured_at
                        FROM social_momentum_v2
                        WHERE captured_at > NOW() - INTERVAL '25 hours'
                        ORDER BY symbol, captured_at ASC
                    """))
                rows = result.fetchall()

            if not rows:
                return

            # Group by symbol
            snapshots_by_coin: dict[str, list[MomentumSnapshot]] = {}
            for row in rows:
                symbol = row[0]
                snap = MomentumSnapshot(
                    symbol=symbol,
                    daily_active_addresses=float(row[1] or 0),
                    transactions_count=float(row[2] or 0),
                    dev_activity=float(row[3] or 0),
                    volume_24h=float(row[4] or 0),
                    price=float(row[5] or 0),
                    price_change_24h=float(row[6] or 0),
                    captured_at=row[7],
                )
                if symbol not in snapshots_by_coin:
                    snapshots_by_coin[symbol] = []
                snapshots_by_coin[symbol].append(snap)

            # Keep only latest 4 per coin
            for symbol in snapshots_by_coin:
                snapshots_by_coin[symbol] = snapshots_by_coin[symbol][-4:]

            # Build coin → PM condition_ids mapping from discovered crypto markets
            symbol_names = {
                "BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana",
                "XRP": "xrp", "ADA": "cardano", "DOGE": "dogecoin",
                "AVAX": "avalanche", "DOT": "polkadot", "LINK": "chainlink",
                "MATIC": "matic", "UNI": "uniswap", "ATOM": "cosmos",
                "LTC": "litecoin", "NEAR": "near", "ARB": "arbitrum",
                "OP": "optimism", "APT": "aptos",
            }
            coin_mapping: dict[str, list[str]] = {}
            for mkt in self._markets:
                if not getattr(mkt, "active", False) or getattr(mkt, "closed", False):
                    continue
                q_lower = getattr(mkt, "question", "").lower()
                for symbol, name in symbol_names.items():
                    if symbol.lower() in q_lower or name in q_lower:
                        if symbol not in coin_mapping:
                            coin_mapping[symbol] = []
                        cid = getattr(mkt, "condition_id", "")
                        if cid and cid not in coin_mapping[symbol]:
                            coin_mapping[symbol].append(cid)
            self._social_divergence.set_coin_mapping(coin_mapping)
            logger.info(
                "coin_mapping_built",
                mapped_coins=len(coin_mapping),
                total_contracts=sum(len(v) for v in coin_mapping.values()),
                symbols=list(coin_mapping.keys())[:10],
            )

            # Calculate divergence signals
            signals = self._social_divergence.calculate_signals(snapshots_by_coin)

            # Store for Strategy B consumption
            self._latest_divergence_signals = signals

            if signals:
                logger.info(
                    "divergence_signals_found",
                    count=len(signals),
                    symbols=[s.symbol for s in signals[:5]],
                )

        except Exception as e:
            logger.warning("divergence_calc_error", error=str(e))

    # ------------------------------------------------------------------
    # Morning briefing + anomaly detection
    # ------------------------------------------------------------------

    async def _morning_briefing_scheduler(self) -> None:
        """Send morning briefing to Slack at 08:00 UTC."""
        while not self._shutdown_event.is_set():
            now = datetime.now(UTC)
            if now.hour == 8 and now.minute == 0:
                try:
                    await self._send_morning_briefing()
                except Exception as e:
                    logger.error("morning_briefing_error", error=str(e))
                await asyncio.sleep(3600)  # Don't send again for 1 hour
            else:
                await asyncio.sleep(55)

    async def _send_morning_briefing(self) -> None:
        """Generate and send consolidated morning briefing via Slack.

        Single message combining Arbo (A+B+C), CryptoArb, and Nightcap status.
        Replaces: old daily_report + old morning_briefing + CryptoArb separate msg.
        """
        if self._slack_bot is None or self._paper_engine is None:
            return

        from sqlalchemy import text

        from arbo.utils.db import get_session_factory

        factory = get_session_factory()
        now = datetime.now(UTC)
        yesterday_start = (now - timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        yesterday_end = now.replace(hour=0, minute=0, second=0, microsecond=0)

        lines = [f":sunrise: *Denni prehled — {now.strftime('%d.%m.%Y')}*"]

        # ── ARBO (Polymarket) ──
        async with factory() as session:
            # Yesterday's resolved trades per strategy
            result = await session.execute(
                text(
                    "SELECT strategy, count(*), "
                    "sum(case when status='won' then 1 else 0 end), "
                    "coalesce(sum(actual_pnl), 0) "
                    "FROM paper_trades "
                    "WHERE resolved_at >= :start AND resolved_at < :end "
                    "AND status IN ('won','lost') "
                    "GROUP BY strategy ORDER BY strategy"
                ),
                {"start": yesterday_start, "end": yesterday_end},
            )
            strat_rows = result.all()

            # Open positions per strategy
            result = await session.execute(
                text(
                    "SELECT strategy, count(*), coalesce(sum(size), 0) "
                    "FROM paper_positions GROUP BY strategy ORDER BY strategy"
                )
            )
            pos_rows = result.all()

            # Total P&L (all time)
            result = await session.execute(
                text(
                    "SELECT coalesce(sum(actual_pnl), 0) FROM paper_trades "
                    "WHERE status IN ('won','lost')"
                )
            )
            total_pnl = float(result.scalar())

            # Notable trades (top 3 by absolute P&L)
            result = await session.execute(
                text(
                    "SELECT id, strategy, actual_pnl, trade_details->>'city' as city "
                    "FROM paper_trades "
                    "WHERE resolved_at >= :start AND resolved_at < :end "
                    "AND status IN ('won','lost') "
                    "ORDER BY abs(actual_pnl) DESC LIMIT 3"
                ),
                {"start": yesterday_start, "end": yesterday_end},
            )
            notable = result.all()

        total_yesterday_pnl = sum(float(r[3]) for r in strat_rows)
        total_resolved = sum(r[1] for r in strat_rows)
        total_wins = sum(r[2] for r in strat_rows)
        total_open = sum(r[1] for r in pos_rows)
        total_deployed = sum(float(r[2]) for r in pos_rows)

        lines.append("")
        lines.append("*ARBO (Polymarket)*")

        # Yesterday summary
        if total_resolved > 0:
            wr = total_wins / total_resolved * 100
            sign = "+" if total_yesterday_pnl >= 0 else ""
            lines.append(
                f"  Vcera: {total_resolved} trades ({total_wins}W/"
                f"{total_resolved - total_wins}L, {wr:.0f}%), "
                f"{sign}${total_yesterday_pnl:.2f}"
            )
            for r in strat_rows:
                sid, cnt, wins, pnl = r[0], r[1], r[2], float(r[3])
                s = "+" if pnl >= 0 else ""
                sname = {"A": "Theta", "B": "Reflexivity", "C": "Weather"}.get(sid, sid)
                lines.append(f"    {sid} ({sname}): {cnt}t ({wins}W), {s}${pnl:.2f}")
        else:
            lines.append("  Vcera: zadne resolved trades")

        # Notable
        if notable:
            for n in notable:
                pnl = float(n[2])
                emoji = ":white_check_mark:" if pnl > 0 else ":x:"
                city = f" ({n[3]})" if n[3] else ""
                s = "+" if pnl >= 0 else ""
                lines.append(f"    {emoji} #{n[0]} {n[1]}{city}: {s}${pnl:.2f}")

        # Current state
        sign = "+" if total_pnl >= 0 else ""
        from arbo.core.risk_manager import RESERVE_CAPITAL, STRATEGY_ALLOCATIONS
        total_capital = float(sum(
            v for v in STRATEGY_ALLOCATIONS.values()
        ) + RESERVE_CAPITAL)
        roi_pct = total_pnl / total_capital * 100 if total_capital > 0 else 0
        roi_sign = "+" if roi_pct >= 0 else ""
        yesterday_roi = total_yesterday_pnl / total_capital * 100 if total_capital > 0 else 0
        yesterday_roi_sign = "+" if yesterday_roi >= 0 else ""

        lines.append(
            f"  *ROI: {roi_sign}{roi_pct:.1f}%* (vcera {yesterday_roi_sign}{yesterday_roi:.1f}%)"
        )
        lines.append(
            f"  Celkovy P&L: {sign}${total_pnl:.2f}, "
            f"Pozice: {total_open} (${total_deployed:.0f} deployed)"
        )
        for r in pos_rows:
            sid, cnt, deployed = r[0], r[1], float(r[2])
            sname = {"A": "Theta", "B": "Reflexivity", "C": "Weather"}.get(sid, sid)
            lines.append(f"    {sid} ({sname}): {cnt} pozic, ${deployed:.0f}")

        # ── CRYPTOARB ──
        try:
            ca = await self._get_cryptoarb_for_slack()
            if ca is not None:
                lines.append("")
                lines.append("*CRYPTOARB (statarb)*")
                eq = ca.get("equity", 1.0)
                dd = ca.get("drawdown_pct", 0)
                trades = ca.get("total_trades", 0)
                wr = ca.get("win_rate", 0)
                mode = ca.get("mode", "?")
                pnl_pct = (eq - 1.0) * 100

                lines.append(
                    f"  Equity: {eq:.4f} ({'+' if pnl_pct >= 0 else ''}{pnl_pct:.1f}%), "
                    f"DD: {dd:.1f}%, Trades: {trades}, WR: {wr:.0f}%, Mode: {mode}"
                )

                pairs = ca.get("pairs", [])
                if pairs:
                    active = [p for p in pairs if p.get("position") != "FLAT"]
                    flat = [p for p in pairs if p.get("position") == "FLAT"]
                    if active:
                        lines.append(
                            "  Aktivni: "
                            + ", ".join(
                                f"{p['name']} {p['position']} ({p.get('spread', 0):+.2f}σ)"
                                for p in active
                            )
                        )
                    lines.append(f"  Flat: {len(flat)}/{len(pairs)} paru")
        except Exception as e:
            logger.debug("morning_brief_cryptoarb_error", error=str(e))

        # ── NIGHTCAP ──
        try:
            nc = self._read_nightcap_state()
            if nc is not None:
                lines.append("")
                lines.append("*NIGHTCAP (VIX futures)*")
                regime = nc.get("regime", "?")
                vix = nc.get("vix_level", 0)
                eq = nc.get("equity", 0)
                dd = nc.get("drawdown_pct", 0)
                env = nc.get("environment", "?")
                pos = nc.get("position", {})
                pos_dir = pos.get("direction", "")

                lines.append(
                    f"  Rezim: {regime}, VIX: {vix:.1f}, "
                    f"Equity: ${eq:,.0f}, DD: {dd:.1f}%, Mode: {env}"
                )
                if pos_dir:
                    lines.append(
                        f"  Pozice: {pos.get('symbol', '?')} {pos_dir} "
                        f"x{pos.get('contracts', 0)}, "
                        f"P&L: ${pos.get('unrealized_pnl', 0):.2f}"
                    )
                else:
                    lines.append("  Pozice: zadna")
        except Exception as e:
            logger.debug("morning_brief_nightcap_error", error=str(e))

        # ── Footer ──
        lines.append("")
        lines.append(f"_Dashboard: arbo.click_")

        await self._slack_bot.send_message("\n".join(lines))
        logger.info("morning_briefing_sent")

    async def _anomaly_check_scheduler(self) -> None:
        """Check for anomalies every 2 hours and alert via Slack."""
        await asyncio.sleep(7200)  # First check after 2 hours
        while not self._shutdown_event.is_set():
            try:
                await self._check_anomalies()
            except Exception as e:
                logger.error("anomaly_check_error", error=str(e))
            await asyncio.sleep(7200)

    async def _check_anomalies(self) -> None:
        """Run anomaly checks and send Slack alerts if needed.

        Deduplication: each alert type is sent at most once per day.
        State persisted to /tmp/arbo_anomaly_state.json so restarts don't re-send.
        """
        if self._slack_bot is None:
            return

        import json as _json
        from pathlib import Path

        from sqlalchemy import text

        from arbo.utils.db import get_session_factory

        alerts: list[str] = []
        factory = get_session_factory()
        now = datetime.now(UTC)
        today_key = now.strftime("%Y-%m-%d")

        # Load persisted dedup state (survives restarts)
        state_path = Path("/opt/arbo/logs/anomaly_state.json")
        sent_today: dict[str, str] = {}
        try:
            if state_path.exists():
                sent_today = _json.loads(state_path.read_text())
        except Exception:
            sent_today = {}
        # Reset on new day
        if sent_today.get("_date") != today_key:
            sent_today = {"_date": today_key}

        async with factory() as session:
            # 1. No activity in 24h (no new trades AND no resolutions)
            if "no_trades" not in sent_today:
                result = await session.execute(
                    text(
                        "SELECT count(*) FROM paper_trades "
                        "WHERE placed_at > :since OR resolved_at > :since"
                    ),
                    {"since": now - timedelta(hours=24)},
                )
                if result.scalar() == 0:
                    alerts.append(":warning: Zadna aktivita za poslednich 24 hodin")
                    sent_today["no_trades"] = today_key

            # 2. Daily loss > $150 (realistic threshold for 3-strategy portfolio)
            today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            result = await session.execute(
                text(
                    "SELECT coalesce(sum(actual_pnl), 0) FROM paper_trades "
                    "WHERE resolved_at >= :start AND status IN ('won','lost')"
                ),
                {"start": today_start},
            )
            daily_pnl = float(result.scalar())
            if daily_pnl < -150 and "daily_loss" not in sent_today:
                alerts.append(
                    f":red_circle: Dnesni ztrata ${abs(daily_pnl):.2f} prekrocila $150 limit"
                )
                sent_today["daily_loss"] = today_key

        # 3. Tasks health (always check — task crash is critical)
        stopped = [name for name, ts in self._tasks.items() if ts.task and ts.task.done()]
        if stopped:
            alerts.append(f":red_circle: Strategie zastaveny: {', '.join(stopped)}")

        if alerts:
            msg = "*Anomalie detekovana:*\n" + "\n".join(alerts)
            await self._slack_bot.send_message(msg)
            logger.warning("anomaly_alerts_sent", count=len(alerts))

        # Persist state to disk (survives restarts)
        try:
            state_path.write_text(_json.dumps(sent_today))
        except Exception as write_err:
            logger.warning("anomaly_state_write_failed", error=str(write_err), path=str(state_path))

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
        if self._strategy_c2 is not None:
            result["C2"] = self._strategy_c2.stats
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
