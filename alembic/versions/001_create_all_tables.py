"""Create all tables.

Revision ID: 001
Revises: None
Create Date: 2026-02-20
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # --- EVENTS ---
    op.create_table(
        "events",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("external_id", sa.String(64), nullable=False),
        sa.Column("source", sa.String(32), nullable=False),
        sa.Column("sport", sa.String(32), nullable=False),
        sa.Column("league", sa.String(128), nullable=True),
        sa.Column("home_team", sa.String(128), nullable=False),
        sa.Column("away_team", sa.String(128), nullable=False),
        sa.Column("start_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("status", sa.String(16), server_default="upcoming"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint("source", "external_id", name="uq_events_source_external"),
    )
    op.create_index("idx_events_start", "events", ["start_time"])
    op.create_index("idx_events_sport_status", "events", ["sport", "status"])

    # --- EVENT_MAPPINGS ---
    op.create_table(
        "event_mappings",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("canonical_id", sa.Integer, nullable=False),
        sa.Column("mapped_id", sa.Integer, nullable=False),
        sa.Column("match_score", sa.Numeric(4, 3), nullable=False),
        sa.Column("matched_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint(
            "canonical_id", "mapped_id", name="uq_event_mappings_canonical_mapped"
        ),
        sa.ForeignKeyConstraint(["mapped_id"], ["events.id"]),
    )

    # --- ODDS_SNAPSHOTS ---
    op.create_table(
        "odds_snapshots",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("event_id", sa.Integer, nullable=False),
        sa.Column("source", sa.String(32), nullable=False),
        sa.Column("market_type", sa.String(32), nullable=False),
        sa.Column("selection", sa.String(64), nullable=False),
        sa.Column("back_odds", sa.Numeric(8, 4), nullable=True),
        sa.Column("lay_odds", sa.Numeric(8, 4), nullable=True),
        sa.Column("back_stake", sa.Numeric(12, 2), nullable=True),
        sa.Column("lay_stake", sa.Numeric(12, 2), nullable=True),
        sa.Column("bookmaker", sa.String(64), nullable=True),
        sa.Column("captured_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.ForeignKeyConstraint(["event_id"], ["events.id"]),
    )
    op.create_index("idx_odds_event_time", "odds_snapshots", ["event_id", sa.text("captured_at DESC")])
    op.create_index("idx_odds_source_market", "odds_snapshots", ["source", "market_type"])

    # --- OPPORTUNITIES ---
    op.create_table(
        "opportunities",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("event_id", sa.Integer, nullable=False),
        sa.Column("strategy", sa.String(32), nullable=False),
        sa.Column("expected_edge", sa.Numeric(6, 4), nullable=False),
        sa.Column("details", JSONB, nullable=False),
        sa.Column("status", sa.String(16), server_default="detected"),
        sa.Column("detected_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("expired_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["event_id"], ["events.id"]),
    )
    op.create_index("idx_opps_status", "opportunities", ["status", sa.text("detected_at DESC")])

    # --- BETS ---
    op.create_table(
        "bets",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("opportunity_id", sa.Integer, nullable=True),
        sa.Column("event_id", sa.Integer, nullable=False),
        sa.Column("strategy", sa.String(32), nullable=False),
        sa.Column("platform", sa.String(32), nullable=False),
        sa.Column("external_bet_id", sa.String(128), nullable=True),
        sa.Column("side", sa.String(8), nullable=False),
        sa.Column("selection", sa.String(64), nullable=False),
        sa.Column("odds", sa.Numeric(8, 4), nullable=False),
        sa.Column("stake", sa.Numeric(10, 2), nullable=False),
        sa.Column("potential_pnl", sa.Numeric(10, 2), nullable=True),
        sa.Column("actual_pnl", sa.Numeric(10, 2), nullable=True),
        sa.Column("commission_paid", sa.Numeric(8, 2), server_default="0"),
        sa.Column("edge_at_exec", sa.Numeric(6, 4), nullable=True),
        sa.Column("fill_pct", sa.Numeric(4, 3), server_default="1.000"),
        sa.Column("status", sa.String(16), server_default="pending"),
        sa.Column("is_paper", sa.Boolean, nullable=False, server_default="true"),
        sa.Column("placed_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("matched_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("settled_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("notes", sa.Text, nullable=True),
        sa.ForeignKeyConstraint(["opportunity_id"], ["opportunities.id"]),
        sa.ForeignKeyConstraint(["event_id"], ["events.id"]),
    )
    op.create_index("idx_bets_status", "bets", ["status", sa.text("placed_at DESC")])
    op.create_index("idx_bets_event", "bets", ["event_id"])
    op.create_index("idx_bets_paper", "bets", ["is_paper", "status"])

    # --- DAILY_PNL ---
    op.create_table(
        "daily_pnl",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("date", sa.Date, nullable=False, unique=True),
        sa.Column("num_opportunities", sa.Integer, server_default="0"),
        sa.Column("num_bets", sa.Integer, server_default="0"),
        sa.Column("num_wins", sa.Integer, server_default="0"),
        sa.Column("total_staked", sa.Numeric(12, 2), server_default="0"),
        sa.Column("gross_pnl", sa.Numeric(10, 2), server_default="0"),
        sa.Column("total_commission", sa.Numeric(8, 2), server_default="0"),
        sa.Column("net_pnl", sa.Numeric(10, 2), server_default="0"),
        sa.Column("bankroll_start", sa.Numeric(12, 2), nullable=False),
        sa.Column("bankroll_end", sa.Numeric(12, 2), nullable=False),
        sa.Column("roi_pct", sa.Numeric(6, 4), nullable=True),
        sa.Column("notes", sa.Text, nullable=True),
    )

    # --- NEWS_ITEMS ---
    op.create_table(
        "news_items",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("source", sa.String(32), nullable=False),
        sa.Column("title", sa.String(512), nullable=False),
        sa.Column("body", sa.Text, nullable=True),
        sa.Column("url", sa.String(1024), nullable=True),
        sa.Column("content_hash", sa.String(64), nullable=False, unique=True),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("fetched_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("analyzed", sa.Boolean, server_default="false"),
        sa.Column("analysis_result", JSONB, nullable=True),
    )
    op.create_index("idx_news_pending", "news_items", ["analyzed", sa.text("fetched_at DESC")])


def downgrade() -> None:
    op.drop_table("news_items")
    op.drop_table("daily_pnl")
    op.drop_table("bets")
    op.drop_table("opportunities")
    op.drop_table("odds_snapshots")
    op.drop_table("event_mappings")
    op.drop_table("events")
