"""Database engine, session management, and SQLAlchemy 2.0 async models.

Polymarket-specific schema. All timestamps UTC. All monetary values USDC.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator  # noqa: TC003 — used in return type at runtime
from datetime import date as date_type  # noqa: TC003 — SQLAlchemy Mapped needs at runtime
from datetime import datetime  # noqa: TC003 — SQLAlchemy Mapped needs at runtime
from decimal import Decimal

from sqlalchemy import (
    BigInteger,
    Boolean,
    Date,
    DateTime,
    Float,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from arbo.config.settings import get_config


class Base(DeclarativeBase):
    pass


# ================================================================
# MARKETS: Polymarket market catalog from Gamma API
# ================================================================
class Market(Base):
    __tablename__ = "markets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    condition_id: Mapped[str] = mapped_column(String(128), nullable=False, unique=True)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    slug: Mapped[str | None] = mapped_column(String(256), nullable=True)
    category: Mapped[str] = mapped_column(String(64), nullable=False)
    outcomes: Mapped[dict] = mapped_column(JSONB, nullable=False)  # type: ignore[type-arg]
    clob_token_ids: Mapped[dict] = mapped_column(JSONB, nullable=False)  # type: ignore[type-arg]
    fee_enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    neg_risk: Mapped[bool] = mapped_column(Boolean, default=False)
    volume_24h: Mapped[Decimal | None] = mapped_column(Numeric(16, 2), nullable=True)
    liquidity: Mapped[Decimal | None] = mapped_column(Numeric(16, 2), nullable=True)
    end_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    last_price_yes: Mapped[float | None] = mapped_column(Float, nullable=True)
    last_price_no: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        Index("idx_markets_category", "category"),
        Index("idx_markets_active", "active"),
        Index("idx_markets_volume", volume_24h.desc()),
    )


# ================================================================
# SIGNALS: Every detected opportunity from any layer (1-9)
# ================================================================
class Signal(Base):
    __tablename__ = "signals"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    layer: Mapped[int] = mapped_column(Integer, nullable=False)
    market_condition_id: Mapped[str] = mapped_column(String(128), nullable=False)
    direction: Mapped[str] = mapped_column(String(8), nullable=False)  # BUY_YES, BUY_NO
    edge: Mapped[Decimal | None] = mapped_column(Numeric(8, 4), nullable=True)
    confidence: Mapped[Decimal | None] = mapped_column(Numeric(4, 3), nullable=True)
    details: Mapped[dict] = mapped_column(JSONB, nullable=False)  # type: ignore[type-arg]
    confluence_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    detected_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (
        Index("idx_signals_layer", "layer", detected_at.desc()),
        Index("idx_signals_market", "market_condition_id"),
    )


# ================================================================
# PAPER_TRADES: Simulated trades from paper trading engine
# ================================================================
class PaperTrade(Base):
    __tablename__ = "paper_trades"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    signal_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    market_condition_id: Mapped[str] = mapped_column(String(128), nullable=False)
    token_id: Mapped[str] = mapped_column(String(128), nullable=False)
    layer: Mapped[int] = mapped_column(Integer, nullable=False)
    side: Mapped[str] = mapped_column(String(8), nullable=False)  # BUY, SELL
    price: Mapped[Decimal] = mapped_column(Numeric(8, 4), nullable=False)
    size: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    slippage: Mapped[Decimal] = mapped_column(Numeric(6, 4), default=Decimal("0.005"))
    edge_at_exec: Mapped[Decimal | None] = mapped_column(Numeric(8, 4), nullable=True)
    confluence_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    kelly_fraction: Mapped[Decimal | None] = mapped_column(Numeric(6, 4), nullable=True)
    status: Mapped[str] = mapped_column(String(16), default="open")  # open, won, lost, cancelled
    actual_pnl: Mapped[Decimal | None] = mapped_column(Numeric(12, 2), nullable=True)
    fee_paid: Mapped[Decimal] = mapped_column(Numeric(8, 4), default=Decimal("0"))
    placed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    strategy: Mapped[str | None] = mapped_column(String(8), nullable=True)  # RDH: "A", "B", "C"

    __table_args__ = (
        Index("idx_paper_trades_status", "status", placed_at.desc()),
        Index("idx_paper_trades_layer", "layer"),
        Index("idx_paper_trades_strategy", "strategy"),
    )


# ================================================================
# PAPER_POSITIONS: Current open positions in paper trading
# ================================================================
class PaperPosition(Base):
    __tablename__ = "paper_positions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    market_condition_id: Mapped[str] = mapped_column(String(128), nullable=False)
    token_id: Mapped[str] = mapped_column(String(128), nullable=False)
    side: Mapped[str] = mapped_column(String(8), nullable=False)
    avg_price: Mapped[Decimal] = mapped_column(Numeric(8, 4), nullable=False)
    size: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    current_price: Mapped[Decimal | None] = mapped_column(Numeric(8, 4), nullable=True)
    unrealized_pnl: Mapped[Decimal | None] = mapped_column(Numeric(12, 2), nullable=True)
    layer: Mapped[int] = mapped_column(Integer, nullable=False)
    opened_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        UniqueConstraint("market_condition_id", "token_id", "side", name="uq_paper_position"),
    )


# ================================================================
# PAPER_SNAPSHOTS: Hourly portfolio snapshots
# ================================================================
class PaperSnapshot(Base):
    __tablename__ = "paper_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    balance: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    unrealized_pnl: Mapped[Decimal] = mapped_column(Numeric(12, 2), default=Decimal("0"))
    total_value: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    num_open_positions: Mapped[int] = mapped_column(Integer, default=0)
    per_layer_pnl: Mapped[dict] = mapped_column(JSONB, nullable=False)  # type: ignore[type-arg]
    snapshot_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


# ================================================================
# WHALE_WALLETS: Tracked profitable wallets
# ================================================================
class WhaleWallet(Base):
    __tablename__ = "whale_wallets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    address: Mapped[str] = mapped_column(String(42), nullable=False, unique=True)
    win_rate: Mapped[Decimal | None] = mapped_column(Numeric(5, 3), nullable=True)
    total_volume: Mapped[Decimal | None] = mapped_column(Numeric(16, 2), nullable=True)
    resolved_positions: Mapped[int] = mapped_column(Integer, default=0)
    specialization: Mapped[str | None] = mapped_column(String(32), nullable=True)
    last_active: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


# ================================================================
# ORDER_FLOW: On-chain order flow data from Polygon
# ================================================================
class OrderFlow(Base):
    __tablename__ = "order_flow"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    market_condition_id: Mapped[str] = mapped_column(String(128), nullable=False)
    buy_volume: Mapped[Decimal] = mapped_column(Numeric(16, 2), default=Decimal("0"))
    sell_volume: Mapped[Decimal] = mapped_column(Numeric(16, 2), default=Decimal("0"))
    volume_zscore: Mapped[float | None] = mapped_column(Float, nullable=True)
    flow_imbalance: Mapped[float | None] = mapped_column(Float, nullable=True)
    cumulative_delta: Mapped[Decimal | None] = mapped_column(Numeric(16, 2), nullable=True)
    window_seconds: Mapped[int] = mapped_column(Integer, nullable=False)
    captured_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (Index("idx_order_flow_market", "market_condition_id", captured_at.desc()),)


# ================================================================
# DAILY_PNL: Aggregated daily performance
# ================================================================
class DailyPnl(Base):
    __tablename__ = "daily_pnl"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    date: Mapped[date_type] = mapped_column(Date, nullable=False, unique=True)
    num_signals: Mapped[int] = mapped_column(Integer, default=0)
    num_trades: Mapped[int] = mapped_column(Integer, default=0)
    num_wins: Mapped[int] = mapped_column(Integer, default=0)
    total_size: Mapped[Decimal] = mapped_column(Numeric(12, 2), default=Decimal("0"))
    gross_pnl: Mapped[Decimal] = mapped_column(Numeric(12, 2), default=Decimal("0"))
    total_fees: Mapped[Decimal] = mapped_column(Numeric(8, 4), default=Decimal("0"))
    net_pnl: Mapped[Decimal] = mapped_column(Numeric(12, 2), default=Decimal("0"))
    per_layer_pnl: Mapped[dict | None] = mapped_column(JSONB, nullable=True)  # type: ignore[type-arg]
    bankroll_start: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    bankroll_end: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    roi_pct: Mapped[Decimal | None] = mapped_column(Numeric(6, 4), nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)


# ================================================================
# DAILY_TRADE_COUNTER: Persist daily trade count across restarts
# ================================================================
class DailyTradeCounter(Base):
    __tablename__ = "daily_trade_counter"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    trade_date: Mapped[date_type] = mapped_column(Date, nullable=False, unique=True)
    trade_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


# ================================================================
# SYSTEM_STATE: Generic key-value store for persistent system state
# ================================================================
class SystemState(Base):
    __tablename__ = "system_state"

    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[str] = mapped_column(Text, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


# ================================================================
# NEWS_ITEMS: Raw news for LLM analysis (Layer 8 / news agent)
# ================================================================
class NewsItem(Base):
    __tablename__ = "news_items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source: Mapped[str] = mapped_column(String(32), nullable=False)
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    body: Mapped[str | None] = mapped_column(Text, nullable=True)
    url: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    analyzed: Mapped[bool] = mapped_column(Boolean, default=False)
    analysis_result: Mapped[dict | None] = mapped_column(JSONB, nullable=True)  # type: ignore[type-arg]

    __table_args__ = (Index("idx_news_pending", "analyzed", fetched_at.desc()),)


# ================================================================
# REAL_MARKET_DATA: Snapshots of market prices for model retraining
# ================================================================
class RealMarketData(Base):
    __tablename__ = "real_market_data"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    market_condition_id: Mapped[str] = mapped_column(String(128), nullable=False)
    polymarket_mid: Mapped[float] = mapped_column(Float, nullable=False)
    pinnacle_prob: Mapped[float | None] = mapped_column(Float, nullable=True)
    spread: Mapped[float | None] = mapped_column(Float, nullable=True)
    volume_24h: Mapped[Decimal | None] = mapped_column(Numeric(16, 2), nullable=True)
    liquidity: Mapped[Decimal | None] = mapped_column(Numeric(16, 2), nullable=True)
    source: Mapped[str] = mapped_column(String(32), nullable=False)
    captured_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (
        Index("idx_real_market_data_cond_time", "market_condition_id", captured_at.desc()),
    )


# ================================================================
# RDH: WEATHER_FORECASTS — Temperature forecast data (Strategy C)
# ================================================================
class WeatherForecastRecord(Base):
    __tablename__ = "weather_forecasts"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    city: Mapped[str] = mapped_column(String(32), nullable=False)
    forecast_date: Mapped[date_type] = mapped_column(Date, nullable=False)
    source: Mapped[str] = mapped_column(String(32), nullable=False)
    temp_high_c: Mapped[float] = mapped_column(Float, nullable=False)
    temp_low_c: Mapped[float] = mapped_column(Float, nullable=False)
    condition: Mapped[str | None] = mapped_column(String(128), nullable=True)
    precip_probability: Mapped[float | None] = mapped_column(Float, nullable=True)
    actual_temp_high_c: Mapped[float | None] = mapped_column(Float, nullable=True)
    actual_temp_low_c: Mapped[float | None] = mapped_column(Float, nullable=True)
    accuracy_high: Mapped[float | None] = mapped_column(Float, nullable=True)
    accuracy_low: Mapped[float | None] = mapped_column(Float, nullable=True)
    fetched_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (
        Index("idx_weather_city_date", "city", "forecast_date"),
        Index("idx_weather_source", "source"),
    )


# ================================================================
# RDH: TAKER_FLOW_SNAPSHOTS — On-chain taker flow (Strategy A)
# ================================================================
class TakerFlowSnapshot(Base):
    __tablename__ = "taker_flow_snapshots"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    market_condition_id: Mapped[str] = mapped_column(String(128), nullable=False)
    yes_taker_volume: Mapped[Decimal] = mapped_column(Numeric(16, 2), nullable=False)
    no_taker_volume: Mapped[Decimal] = mapped_column(Numeric(16, 2), nullable=False)
    yes_no_ratio: Mapped[float] = mapped_column(Float, nullable=False)
    z_score: Mapped[float] = mapped_column(Float, nullable=False)
    window_seconds: Mapped[int] = mapped_column(Integer, nullable=False)
    is_peak_optimism: Mapped[bool] = mapped_column(Boolean, default=False)
    captured_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (
        Index("idx_taker_flow_market", "market_condition_id", captured_at.desc()),
    )


# ================================================================
# RDH: ATTENTION_MARKET_STATE — Kaito AI mindshare (Strategy B)
# ================================================================
class AttentionMarketState(Base):
    __tablename__ = "attention_market_state"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    market_condition_id: Mapped[str] = mapped_column(String(128), nullable=False)
    phase: Mapped[str] = mapped_column(String(16), nullable=False)
    kaito_mindshare: Mapped[float | None] = mapped_column(Float, nullable=True)
    pm_price: Mapped[float] = mapped_column(Float, nullable=False)
    divergence: Mapped[float | None] = mapped_column(Float, nullable=True)
    phase_entered_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        Index("idx_attention_market", "market_condition_id"),
        Index("idx_attention_phase", "phase"),
    )


# ================================================================
# RDH: RESOLUTION_CHAINS — Capital chaining (Strategy C)
# ================================================================
class ResolutionChain(Base):
    __tablename__ = "resolution_chains"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    chain_id: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    city_sequence: Mapped[dict] = mapped_column(JSONB, nullable=False)  # type: ignore[type-arg]
    current_city_index: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    initial_capital: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    current_capital: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    cumulative_pnl: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False, default=0)
    num_resolutions: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="active")
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


# ================================================================
# RDH: STRATEGY_ALLOCATIONS — Per-strategy capital tracking
# ================================================================
class StrategyAllocation(Base):
    __tablename__ = "strategy_allocations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    strategy: Mapped[str] = mapped_column(String(1), nullable=False, unique=True)
    allocated: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    deployed: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False, default=0)
    available: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    weekly_pnl: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False, default=0)
    total_pnl: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False, default=0)
    is_halted: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


# ================================================================
# Engine & Session Factory
# ================================================================

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def get_engine() -> AsyncEngine:
    """Get or create the async SQLAlchemy engine (singleton)."""
    global _engine
    if _engine is None:
        config = get_config()
        _engine = create_async_engine(
            config.database_url,
            echo=False,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,
        )
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Get or create the async session factory (singleton)."""
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            get_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
        )
    return _session_factory


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Async context manager yielding a database session."""
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
