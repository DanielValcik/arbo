"""Add social_momentum table for LunarCrush data (Strategy B2).

Stores periodic snapshots of crypto social metrics from LunarCrush API v4.
Used by the Social Momentum Divergence calculator to detect divergences
between social attention and Polymarket contract prices.

Revision ID: 005
Revises: 004
Create Date: 2026-02-26
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "005"
down_revision: Union[str, None] = "004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "social_momentum",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("symbol", sa.String(16), nullable=False),
        sa.Column("name", sa.String(128), nullable=False),
        sa.Column("social_dominance", sa.Float, nullable=True),
        sa.Column("sentiment", sa.Integer, nullable=True),  # 0-100
        sa.Column("galaxy_score", sa.Float, nullable=True),  # 0-100
        sa.Column("alt_rank", sa.Integer, nullable=True),
        sa.Column("interactions_24h", sa.BigInteger, nullable=True),
        sa.Column("social_volume_24h", sa.BigInteger, nullable=True),
        sa.Column("price", sa.Float, nullable=True),
        sa.Column("market_cap", sa.Numeric(20, 2), nullable=True),
        sa.Column("percent_change_24h", sa.Float, nullable=True),
        sa.Column("percent_change_7d", sa.Float, nullable=True),
        sa.Column("percent_change_30d", sa.Float, nullable=True),
        sa.Column("source", sa.String(16), nullable=False, server_default="lunarcrush"),
        sa.Column(
            "captured_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index(
        "idx_social_momentum_symbol_ts",
        "social_momentum",
        ["symbol", sa.text("captured_at DESC")],
    )
    op.create_index(
        "idx_social_momentum_captured",
        "social_momentum",
        ["captured_at"],
    )


def downgrade() -> None:
    op.drop_table("social_momentum")
