"""Add RDH (Reflexive Decay Harvester) tables.

New tables for the 3-strategy architecture:
- weather_forecasts: Temperature forecast data from NOAA/Met Office/Open-Meteo
- taker_flow_snapshots: On-chain taker flow analysis for Strategy A
- attention_market_state: Kaito AI mindshare tracking for Strategy B
- resolution_chains: Capital chaining across weather markets for Strategy C
- strategy_allocations: Per-strategy capital allocation tracking

Revision ID: 004
Revises: 003
Create Date: 2026-02-25
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "004"
down_revision: Union[str, None] = "003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ================================================================
    # weather_forecasts: Temperature forecast data
    # ================================================================
    op.create_table(
        "weather_forecasts",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("city", sa.String(32), nullable=False),
        sa.Column("forecast_date", sa.Date, nullable=False),
        sa.Column("source", sa.String(32), nullable=False),
        sa.Column("temp_high_c", sa.Float, nullable=False),
        sa.Column("temp_low_c", sa.Float, nullable=False),
        sa.Column("condition", sa.String(128), nullable=True),
        sa.Column("precip_probability", sa.Float, nullable=True),
        sa.Column("actual_temp_high_c", sa.Float, nullable=True),
        sa.Column("actual_temp_low_c", sa.Float, nullable=True),
        sa.Column("accuracy_high", sa.Float, nullable=True),
        sa.Column("accuracy_low", sa.Float, nullable=True),
        sa.Column(
            "fetched_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index(
        "idx_weather_city_date",
        "weather_forecasts",
        ["city", "forecast_date"],
    )
    op.create_index(
        "idx_weather_source",
        "weather_forecasts",
        ["source"],
    )

    # ================================================================
    # taker_flow_snapshots: On-chain taker flow analysis (Strategy A)
    # ================================================================
    op.create_table(
        "taker_flow_snapshots",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("market_condition_id", sa.String(128), nullable=False),
        sa.Column("yes_taker_volume", sa.Numeric(16, 2), nullable=False),
        sa.Column("no_taker_volume", sa.Numeric(16, 2), nullable=False),
        sa.Column("yes_no_ratio", sa.Float, nullable=False),
        sa.Column("z_score", sa.Float, nullable=False),
        sa.Column("window_seconds", sa.Integer, nullable=False),
        sa.Column("is_peak_optimism", sa.Boolean, default=False),
        sa.Column(
            "captured_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index(
        "idx_taker_flow_market",
        "taker_flow_snapshots",
        ["market_condition_id", sa.text("captured_at DESC")],
    )

    # ================================================================
    # attention_market_state: Kaito AI mindshare tracking (Strategy B)
    # ================================================================
    op.create_table(
        "attention_market_state",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("market_condition_id", sa.String(128), nullable=False),
        sa.Column("phase", sa.String(16), nullable=False),
        sa.Column("kaito_mindshare", sa.Float, nullable=True),
        sa.Column("pm_price", sa.Float, nullable=False),
        sa.Column("divergence", sa.Float, nullable=True),
        sa.Column("phase_entered_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index(
        "idx_attention_market",
        "attention_market_state",
        ["market_condition_id"],
    )
    op.create_index(
        "idx_attention_phase",
        "attention_market_state",
        ["phase"],
    )

    # ================================================================
    # resolution_chains: Capital chaining across weather markets (Strategy C)
    # ================================================================
    op.create_table(
        "resolution_chains",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("chain_id", sa.String(64), nullable=False, unique=True),
        sa.Column(
            "city_sequence",
            sa.dialects.postgresql.JSONB,
            nullable=False,
        ),
        sa.Column("current_city_index", sa.Integer, nullable=False, default=0),
        sa.Column("initial_capital", sa.Numeric(12, 2), nullable=False),
        sa.Column("current_capital", sa.Numeric(12, 2), nullable=False),
        sa.Column("cumulative_pnl", sa.Numeric(12, 2), nullable=False, default=0),
        sa.Column("num_resolutions", sa.Integer, nullable=False, default=0),
        sa.Column("status", sa.String(16), nullable=False, default="active"),
        sa.Column(
            "started_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
    )

    # ================================================================
    # strategy_allocations: Per-strategy capital tracking
    # ================================================================
    op.create_table(
        "strategy_allocations",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("strategy", sa.String(1), nullable=False, unique=True),
        sa.Column("allocated", sa.Numeric(12, 2), nullable=False),
        sa.Column("deployed", sa.Numeric(12, 2), nullable=False, default=0),
        sa.Column("available", sa.Numeric(12, 2), nullable=False),
        sa.Column("weekly_pnl", sa.Numeric(12, 2), nullable=False, default=0),
        sa.Column("total_pnl", sa.Numeric(12, 2), nullable=False, default=0),
        sa.Column("is_halted", sa.Boolean, nullable=False, default=False),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )


def downgrade() -> None:
    op.drop_table("strategy_allocations")
    op.drop_table("resolution_chains")
    op.drop_table("attention_market_state")
    op.drop_table("taker_flow_snapshots")
    op.drop_table("weather_forecasts")
