"""Add health_checks, city_volume_daily, weather_scan_log tables.

Also adds trade_details JSONB column to paper_trades for comprehensive
data capture (forecast details, CLOB prices, quality gate thresholds).

Revision ID: 009
Revises: 008
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = "009"
down_revision = "008"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # -- health_checks --
    op.create_table(
        "health_checks",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column(
            "check_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("verdict", sa.String(24), nullable=False),
        sa.Column("window_hours", sa.Integer, nullable=False, server_default="12"),
        sa.Column("metrics", JSONB, nullable=False),
        sa.Column("expected", JSONB, nullable=False),
        sa.Column("notes", sa.Text, nullable=True),
    )

    # -- city_volume_daily --
    op.create_table(
        "city_volume_daily",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("city", sa.String(32), nullable=False),
        sa.Column("date", sa.Date, nullable=False),
        sa.Column("volume_24h", sa.Numeric(16, 2), nullable=False),
        sa.Column("liquidity", sa.Numeric(16, 2), nullable=True),
        sa.Column("num_markets", sa.Integer, nullable=False, server_default="0"),
        sa.Column("avg_price", sa.Float, nullable=True),
        sa.Column(
            "captured_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_unique_constraint("uq_city_volume_daily", "city_volume_daily", ["city", "date"])
    op.create_index("idx_city_volume_city_date", "city_volume_daily", ["city", "date"])

    # -- weather_scan_log --
    op.create_table(
        "weather_scan_log",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column(
            "scan_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("city", sa.String(32), nullable=False),
        sa.Column("target_date", sa.Date, nullable=False),
        sa.Column("condition_id", sa.String(128), nullable=False),
        sa.Column("question", sa.Text, nullable=False),
        sa.Column("forecast_temp_c", sa.Float, nullable=False),
        sa.Column("forecast_prob", sa.Float, nullable=False),
        sa.Column("market_price", sa.Float, nullable=False),
        sa.Column("edge", sa.Float, nullable=False),
        sa.Column("direction", sa.String(8), nullable=False),
        sa.Column("volume_24h", sa.Float, nullable=False),
        sa.Column("liquidity", sa.Float, nullable=False),
        sa.Column("quality_gate_passed", sa.Boolean, nullable=False),
        sa.Column("quality_gate_reason", sa.Text, nullable=True),
        sa.Column("traded", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("trade_size", sa.Float, nullable=True),
    )
    op.create_index("idx_scan_log_city_date", "weather_scan_log", ["city", "scan_at"])
    op.create_index("idx_scan_log_traded", "weather_scan_log", ["traded"])

    # -- trade_details JSONB on paper_trades --
    op.add_column(
        "paper_trades",
        sa.Column("trade_details", JSONB, nullable=True),
    )


def downgrade() -> None:
    op.drop_column("paper_trades", "trade_details")
    op.drop_table("weather_scan_log")
    op.drop_table("city_volume_daily")
    op.drop_table("health_checks")
