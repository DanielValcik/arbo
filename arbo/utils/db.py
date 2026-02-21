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

    __table_args__ = (
        Index("idx_paper_trades_status", "status", placed_at.desc()),
        Index("idx_paper_trades_layer", "layer"),
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
