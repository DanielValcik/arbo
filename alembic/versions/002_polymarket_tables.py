"""Drop Matchbook-era tables, create all Polymarket tables.

Revision ID: 002
Revises: 001
Create Date: 2026-02-21
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "002"
down_revision: Union[str, None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # --- Drop old Matchbook-era tables ---
    op.drop_table("bets")
    op.drop_table("opportunities")
    op.drop_table("odds_snapshots")
    op.drop_table("event_mappings")
    op.drop_table("events")
    op.drop_table("daily_pnl")
    op.drop_table("news_items")

    # --- MARKETS ---
    op.create_table(
        "markets",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("condition_id", sa.String(128), nullable=False, unique=True),
        sa.Column("question", sa.Text, nullable=False),
        sa.Column("slug", sa.String(256), nullable=True),
        sa.Column("category", sa.String(64), nullable=False),
        sa.Column("outcomes", JSONB, nullable=False),
        sa.Column("clob_token_ids", JSONB, nullable=False),
        sa.Column("fee_enabled", sa.Boolean, server_default="false"),
        sa.Column("neg_risk", sa.Boolean, server_default="false"),
        sa.Column("volume_24h", sa.Numeric(16, 2), nullable=True),
        sa.Column("liquidity", sa.Numeric(16, 2), nullable=True),
        sa.Column("end_date", sa.DateTime(timezone=True), nullable=True),
        sa.Column("active", sa.Boolean, server_default="true"),
        sa.Column("last_price_yes", sa.Float, nullable=True),
        sa.Column("last_price_no", sa.Float, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            onupdate=sa.func.now(),
        ),
    )
    op.create_index("idx_markets_category", "markets", ["category"])
    op.create_index("idx_markets_active", "markets", ["active"])
    op.create_index("idx_markets_volume", "markets", [sa.text("volume_24h DESC")])

    # --- SIGNALS ---
    op.create_table(
        "signals",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("layer", sa.Integer, nullable=False),
        sa.Column("market_condition_id", sa.String(128), nullable=False),
        sa.Column("direction", sa.String(8), nullable=False),
        sa.Column("edge", sa.Numeric(8, 4), nullable=True),
        sa.Column("confidence", sa.Numeric(4, 3), nullable=True),
        sa.Column("details", JSONB, nullable=False),
        sa.Column("confluence_score", sa.Integer, nullable=True),
        sa.Column("detected_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("idx_signals_layer", "signals", ["layer", sa.text("detected_at DESC")])
    op.create_index("idx_signals_market", "signals", ["market_condition_id"])

    # --- PAPER_TRADES ---
    op.create_table(
        "paper_trades",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("signal_id", sa.BigInteger, nullable=True),
        sa.Column("market_condition_id", sa.String(128), nullable=False),
        sa.Column("token_id", sa.String(128), nullable=False),
        sa.Column("layer", sa.Integer, nullable=False),
        sa.Column("side", sa.String(8), nullable=False),
        sa.Column("price", sa.Numeric(8, 4), nullable=False),
        sa.Column("size", sa.Numeric(12, 2), nullable=False),
        sa.Column("slippage", sa.Numeric(6, 4), server_default="0.0050"),
        sa.Column("edge_at_exec", sa.Numeric(8, 4), nullable=True),
        sa.Column("confluence_score", sa.Integer, nullable=True),
        sa.Column("kelly_fraction", sa.Numeric(6, 4), nullable=True),
        sa.Column("status", sa.String(16), server_default="open"),
        sa.Column("actual_pnl", sa.Numeric(12, 2), nullable=True),
        sa.Column("fee_paid", sa.Numeric(8, 4), server_default="0"),
        sa.Column("placed_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("notes", sa.Text, nullable=True),
    )
    op.create_index(
        "idx_paper_trades_status", "paper_trades", ["status", sa.text("placed_at DESC")]
    )
    op.create_index("idx_paper_trades_layer", "paper_trades", ["layer"])

    # --- PAPER_POSITIONS ---
    op.create_table(
        "paper_positions",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("market_condition_id", sa.String(128), nullable=False),
        sa.Column("token_id", sa.String(128), nullable=False),
        sa.Column("side", sa.String(8), nullable=False),
        sa.Column("avg_price", sa.Numeric(8, 4), nullable=False),
        sa.Column("size", sa.Numeric(12, 2), nullable=False),
        sa.Column("current_price", sa.Numeric(8, 4), nullable=True),
        sa.Column("unrealized_pnl", sa.Numeric(12, 2), nullable=True),
        sa.Column("layer", sa.Integer, nullable=False),
        sa.Column("opened_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            onupdate=sa.func.now(),
        ),
        sa.UniqueConstraint("market_condition_id", "token_id", "side", name="uq_paper_position"),
    )

    # --- PAPER_SNAPSHOTS ---
    op.create_table(
        "paper_snapshots",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("balance", sa.Numeric(12, 2), nullable=False),
        sa.Column("unrealized_pnl", sa.Numeric(12, 2), server_default="0"),
        sa.Column("total_value", sa.Numeric(12, 2), nullable=False),
        sa.Column("num_open_positions", sa.Integer, server_default="0"),
        sa.Column("per_layer_pnl", JSONB, nullable=False),
        sa.Column("snapshot_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # --- WHALE_WALLETS ---
    op.create_table(
        "whale_wallets",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("address", sa.String(42), nullable=False, unique=True),
        sa.Column("win_rate", sa.Numeric(5, 3), nullable=True),
        sa.Column("total_volume", sa.Numeric(16, 2), nullable=True),
        sa.Column("resolved_positions", sa.Integer, server_default="0"),
        sa.Column("specialization", sa.String(32), nullable=True),
        sa.Column("last_active", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            onupdate=sa.func.now(),
        ),
    )

    # --- ORDER_FLOW ---
    op.create_table(
        "order_flow",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("market_condition_id", sa.String(128), nullable=False),
        sa.Column("buy_volume", sa.Numeric(16, 2), server_default="0"),
        sa.Column("sell_volume", sa.Numeric(16, 2), server_default="0"),
        sa.Column("volume_zscore", sa.Float, nullable=True),
        sa.Column("flow_imbalance", sa.Float, nullable=True),
        sa.Column("cumulative_delta", sa.Numeric(16, 2), nullable=True),
        sa.Column("window_seconds", sa.Integer, nullable=False),
        sa.Column("captured_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index(
        "idx_order_flow_market",
        "order_flow",
        ["market_condition_id", sa.text("captured_at DESC")],
    )

    # --- DAILY_PNL (Polymarket version) ---
    op.create_table(
        "daily_pnl",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("date", sa.Date, nullable=False, unique=True),
        sa.Column("num_signals", sa.Integer, server_default="0"),
        sa.Column("num_trades", sa.Integer, server_default="0"),
        sa.Column("num_wins", sa.Integer, server_default="0"),
        sa.Column("total_size", sa.Numeric(12, 2), server_default="0"),
        sa.Column("gross_pnl", sa.Numeric(12, 2), server_default="0"),
        sa.Column("total_fees", sa.Numeric(8, 4), server_default="0"),
        sa.Column("net_pnl", sa.Numeric(12, 2), server_default="0"),
        sa.Column("per_layer_pnl", JSONB, nullable=True),
        sa.Column("bankroll_start", sa.Numeric(12, 2), nullable=False),
        sa.Column("bankroll_end", sa.Numeric(12, 2), nullable=False),
        sa.Column("roi_pct", sa.Numeric(6, 4), nullable=True),
        sa.Column("notes", sa.Text, nullable=True),
    )

    # --- NEWS_ITEMS (Polymarket version) ---
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

    # --- REAL_MARKET_DATA ---
    op.create_table(
        "real_market_data",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("market_condition_id", sa.String(128), nullable=False),
        sa.Column("polymarket_mid", sa.Float, nullable=False),
        sa.Column("pinnacle_prob", sa.Float, nullable=True),
        sa.Column("spread", sa.Float, nullable=True),
        sa.Column("volume_24h", sa.Numeric(16, 2), nullable=True),
        sa.Column("liquidity", sa.Numeric(16, 2), nullable=True),
        sa.Column("source", sa.String(32), nullable=False),
        sa.Column("captured_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index(
        "idx_real_market_data_cond_time",
        "real_market_data",
        ["market_condition_id", sa.text("captured_at DESC")],
    )


def downgrade() -> None:
    op.drop_table("real_market_data")
    op.drop_table("news_items")
    op.drop_table("daily_pnl")
    op.drop_table("order_flow")
    op.drop_table("whale_wallets")
    op.drop_table("paper_snapshots")
    op.drop_table("paper_positions")
    op.drop_table("paper_trades")
    op.drop_table("signals")
    op.drop_table("markets")

    # Recreate old Matchbook tables (reverse of 001)
    # In practice, this is never needed — paper trading only.
