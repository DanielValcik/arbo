"""Database engine, session management, and SQLAlchemy 2.0 async models.

All 7 tables from the dev brief Section 4.
All timestamps UTC. All monetary values EUR.
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

from src.utils.config import get_config


class Base(DeclarativeBase):
    pass


# ================================================================
# EVENTS: Central event registry, normalized across all sources
# ================================================================
class Event(Base):
    __tablename__ = "events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    external_id: Mapped[str] = mapped_column(String(64), nullable=False)
    source: Mapped[str] = mapped_column(String(32), nullable=False)
    sport: Mapped[str] = mapped_column(String(32), nullable=False)
    league: Mapped[str | None] = mapped_column(String(128), nullable=True)
    home_team: Mapped[str] = mapped_column(String(128), nullable=False)
    away_team: Mapped[str] = mapped_column(String(128), nullable=False)
    start_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    status: Mapped[str] = mapped_column(String(16), default="upcoming")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        UniqueConstraint("source", "external_id", name="uq_events_source_external"),
        Index("idx_events_start", "start_time"),
        Index("idx_events_sport_status", "sport", "status"),
    )


# ================================================================
# EVENT_MAPPINGS: Links same real-world event across platforms
# ================================================================
class EventMapping(Base):
    __tablename__ = "event_mappings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    canonical_id: Mapped[int] = mapped_column(Integer, nullable=False)
    mapped_id: Mapped[int] = mapped_column(Integer, nullable=False)
    match_score: Mapped[Decimal] = mapped_column(Numeric(4, 3), nullable=False)
    matched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        UniqueConstraint("canonical_id", "mapped_id", name="uq_event_mappings_canonical_mapped"),
    )


# ================================================================
# ODDS_SNAPSHOTS: Time-series of every price update
# Most write-heavy table. Expect millions of rows per month.
# ================================================================
class OddsSnapshot(Base):
    __tablename__ = "odds_snapshots"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    event_id: Mapped[int] = mapped_column(Integer, nullable=False)
    source: Mapped[str] = mapped_column(String(32), nullable=False)
    market_type: Mapped[str] = mapped_column(String(32), nullable=False)
    selection: Mapped[str] = mapped_column(String(64), nullable=False)
    back_odds: Mapped[Decimal | None] = mapped_column(Numeric(8, 4), nullable=True)
    lay_odds: Mapped[Decimal | None] = mapped_column(Numeric(8, 4), nullable=True)
    back_stake: Mapped[Decimal | None] = mapped_column(Numeric(12, 2), nullable=True)
    lay_stake: Mapped[Decimal | None] = mapped_column(Numeric(12, 2), nullable=True)
    bookmaker: Mapped[str | None] = mapped_column(String(64), nullable=True)
    captured_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (
        Index("idx_odds_event_time", "event_id", captured_at.desc()),
        Index("idx_odds_source_market", "source", "market_type"),
    )


# ================================================================
# OPPORTUNITIES: Every detected arb/value/situational signal
# ================================================================
class Opportunity(Base):
    __tablename__ = "opportunities"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_id: Mapped[int] = mapped_column(Integer, nullable=False)
    strategy: Mapped[str] = mapped_column(String(32), nullable=False)
    expected_edge: Mapped[Decimal] = mapped_column(Numeric(6, 4), nullable=False)
    details: Mapped[dict] = mapped_column(JSONB, nullable=False)  # type: ignore[type-arg]
    status: Mapped[str] = mapped_column(String(16), default="detected")
    detected_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    expired_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (Index("idx_opps_status", "status", detected_at.desc()),)


# ================================================================
# BETS: Every bet placed (paper and live)
# ================================================================
class Bet(Base):
    __tablename__ = "bets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    opportunity_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    event_id: Mapped[int] = mapped_column(Integer, nullable=False)
    strategy: Mapped[str] = mapped_column(String(32), nullable=False)
    platform: Mapped[str] = mapped_column(String(32), nullable=False)
    external_bet_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    side: Mapped[str] = mapped_column(String(8), nullable=False)
    selection: Mapped[str] = mapped_column(String(64), nullable=False)
    odds: Mapped[Decimal] = mapped_column(Numeric(8, 4), nullable=False)
    stake: Mapped[Decimal] = mapped_column(Numeric(10, 2), nullable=False)
    potential_pnl: Mapped[Decimal | None] = mapped_column(Numeric(10, 2), nullable=True)
    actual_pnl: Mapped[Decimal | None] = mapped_column(Numeric(10, 2), nullable=True)
    commission_paid: Mapped[Decimal] = mapped_column(Numeric(8, 2), default=Decimal("0"))
    edge_at_exec: Mapped[Decimal | None] = mapped_column(Numeric(6, 4), nullable=True)
    fill_pct: Mapped[Decimal] = mapped_column(Numeric(4, 3), default=Decimal("1.000"))
    status: Mapped[str] = mapped_column(String(16), default="pending")
    is_paper: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    placed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    matched_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    settled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    __table_args__ = (
        Index("idx_bets_status", "status", placed_at.desc()),
        Index("idx_bets_event", "event_id"),
        Index("idx_bets_paper", "is_paper", "status"),
    )


# ================================================================
# DAILY_PNL: Aggregated daily performance
# ================================================================
class DailyPnl(Base):
    __tablename__ = "daily_pnl"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    date: Mapped[date_type] = mapped_column(Date, nullable=False, unique=True)
    num_opportunities: Mapped[int] = mapped_column(Integer, default=0)
    num_bets: Mapped[int] = mapped_column(Integer, default=0)
    num_wins: Mapped[int] = mapped_column(Integer, default=0)
    total_staked: Mapped[Decimal] = mapped_column(Numeric(12, 2), default=Decimal("0"))
    gross_pnl: Mapped[Decimal] = mapped_column(Numeric(10, 2), default=Decimal("0"))
    total_commission: Mapped[Decimal] = mapped_column(Numeric(8, 2), default=Decimal("0"))
    net_pnl: Mapped[Decimal] = mapped_column(Numeric(10, 2), default=Decimal("0"))
    bankroll_start: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    bankroll_end: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    roi_pct: Mapped[Decimal | None] = mapped_column(Numeric(6, 4), nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)


# ================================================================
# NEWS_ITEMS: Raw news and Reddit posts for LLM analysis
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

_engine = None
_session_factory = None


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
