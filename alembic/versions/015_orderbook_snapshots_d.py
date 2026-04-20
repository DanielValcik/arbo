"""Orderbook snapshots for Strategy D — bid/ask capture for future ML.

Populated by `arbo/scripts/capture_d_orderbook.py` via Polymarket CLOB
REST `/book?token_id=X` polling. Gives us bid/ask + depth features that
the PolymarketData.co Pass 2 pipeline never captured (D3b finding).

Used by (future) model v3 trained on microstructure features.

Rows: ~tens per token per day (5-min polling cadence × game window).
At ~30 active NBA tokens per day → ~8K rows/day = ~3M rows/year.
Cleanup: keep 90 days by default (ALTER to adjust retention later).

Revision ID: 015
Revises: 014
"""
from alembic import op
import sqlalchemy as sa

revision = "015"
down_revision = "014"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "orderbook_snapshots_d",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        # Identity
        sa.Column("token_id", sa.String(80), nullable=False),
        sa.Column("condition_id", sa.String(80), nullable=True),
        sa.Column("sport", sa.String(16), nullable=False, server_default="nba"),
        sa.Column("ts", sa.DateTime(timezone=True), nullable=False),
        # Top of book (raw — UNADJUSTED for the inverted semantics we sometimes see)
        sa.Column("best_bid", sa.Numeric(6, 4), nullable=True),
        sa.Column("best_ask", sa.Numeric(6, 4), nullable=True),
        sa.Column("bid_size", sa.Numeric(14, 2), nullable=True),
        sa.Column("ask_size", sa.Numeric(14, 2), nullable=True),
        # Depth beyond top (up to 5 levels each side, JSONB for flex)
        sa.Column("bids", sa.JSON, nullable=True),  # [[price, size], ...]
        sa.Column("asks", sa.JSON, nullable=True),
        # Derived / cached (nullable — we fill client-side)
        sa.Column("mid", sa.Numeric(6, 4), nullable=True),
        sa.Column("spread_bps", sa.Numeric(10, 2), nullable=True),
        # Context
        sa.Column("neg_risk", sa.Boolean, nullable=False, server_default=sa.text("false")),
        sa.Column("game_date", sa.Date, nullable=True),
        sa.Column("question", sa.Text, nullable=True),
        sa.Column("source", sa.String(16), nullable=False, server_default="clob_rest"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_index(
        "ix_ob_snap_d_token_ts",
        "orderbook_snapshots_d",
        ["token_id", "ts"],
    )
    op.create_index("ix_ob_snap_d_ts", "orderbook_snapshots_d", ["ts"])
    op.create_index("ix_ob_snap_d_game_date", "orderbook_snapshots_d", ["game_date"])


def downgrade() -> None:
    op.drop_index("ix_ob_snap_d_game_date", table_name="orderbook_snapshots_d")
    op.drop_index("ix_ob_snap_d_ts", table_name="orderbook_snapshots_d")
    op.drop_index("ix_ob_snap_d_token_ts", table_name="orderbook_snapshots_d")
    op.drop_table("orderbook_snapshots_d")
