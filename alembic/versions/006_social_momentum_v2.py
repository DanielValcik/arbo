"""Replace LunarCrush social_momentum with Santiment+CoinGecko schema.

Drops the LunarCrush-specific social_momentum table and creates
social_momentum_v2 with on-chain metrics (Santiment) + community
metrics (CoinGecko).

Revision ID: 006
Revises: 005
Create Date: 2026-02-26
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "006"
down_revision: Union[str, None] = "005"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Drop old LunarCrush table (no production data yet)
    op.drop_table("social_momentum")

    # Create new table with Santiment + CoinGecko metrics
    op.create_table(
        "social_momentum_v2",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("symbol", sa.String(16), nullable=False),
        sa.Column("slug", sa.String(64), nullable=False),  # Santiment slug
        sa.Column("coingecko_id", sa.String(64), nullable=True),  # CoinGecko ID
        # Santiment free-access metrics
        sa.Column("daily_active_addresses", sa.Float, nullable=True),
        sa.Column("transactions_count", sa.Float, nullable=True),
        sa.Column("dev_activity", sa.Float, nullable=True),
        # CoinGecko market metrics
        sa.Column("price", sa.Float, nullable=True),
        sa.Column("market_cap", sa.Numeric(20, 2), nullable=True),
        sa.Column("volume_24h", sa.Numeric(20, 2), nullable=True),
        sa.Column("price_change_24h", sa.Float, nullable=True),
        sa.Column("price_change_7d", sa.Float, nullable=True),
        sa.Column("price_change_30d", sa.Float, nullable=True),
        # CoinGecko community metrics
        sa.Column("twitter_followers", sa.BigInteger, nullable=True),
        sa.Column("reddit_subscribers", sa.BigInteger, nullable=True),
        # Metadata
        sa.Column("source", sa.String(32), nullable=False, server_default="santiment+coingecko"),
        sa.Column(
            "captured_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index(
        "idx_social_momentum_v2_symbol_ts",
        "social_momentum_v2",
        ["symbol", sa.text("captured_at DESC")],
    )
    op.create_index(
        "idx_social_momentum_v2_slug",
        "social_momentum_v2",
        ["slug"],
    )
    op.create_index(
        "idx_social_momentum_v2_captured",
        "social_momentum_v2",
        ["captured_at"],
    )


def downgrade() -> None:
    op.drop_table("social_momentum_v2")

    # Recreate old LunarCrush table
    op.create_table(
        "social_momentum",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("symbol", sa.String(16), nullable=False),
        sa.Column("name", sa.String(128), nullable=False),
        sa.Column("social_dominance", sa.Float, nullable=True),
        sa.Column("sentiment", sa.Integer, nullable=True),
        sa.Column("galaxy_score", sa.Float, nullable=True),
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
