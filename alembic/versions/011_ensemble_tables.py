"""Add ensemble_forecasts and ensemble_stats tables for GEFS data.

Stores daily GEFS 31-member ensemble TMAX forecasts per city.
Used by C1f EMOSEnsembleModel for forward-looking sigma estimation.

Revision ID: 011
Revises: 010
"""

import sqlalchemy as sa
from alembic import op

revision = "011"
down_revision = "010"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "ensemble_forecasts",
        sa.Column("city", sa.String(50), nullable=False),
        sa.Column("target_date", sa.String(10), nullable=False),
        sa.Column("member", sa.String(10), nullable=False),
        sa.Column("tmax_c", sa.Float, nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
        ),
        sa.PrimaryKeyConstraint("city", "target_date", "member"),
    )

    op.create_table(
        "ensemble_stats",
        sa.Column("city", sa.String(50), nullable=False),
        sa.Column("target_date", sa.String(10), nullable=False),
        sa.Column("ensemble_mean", sa.Float, nullable=True),
        sa.Column("ensemble_std", sa.Float, nullable=True),
        sa.Column("ensemble_min", sa.Float, nullable=True),
        sa.Column("ensemble_max", sa.Float, nullable=True),
        sa.Column("n_members", sa.Integer, nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
        ),
        sa.PrimaryKeyConstraint("city", "target_date"),
    )

    # Index for quick lookups by city
    op.create_index("ix_ensemble_stats_city", "ensemble_stats", ["city"])
    op.create_index(
        "ix_ensemble_forecasts_city_date",
        "ensemble_forecasts",
        ["city", "target_date"],
    )


def downgrade() -> None:
    op.drop_index("ix_ensemble_forecasts_city_date")
    op.drop_index("ix_ensemble_stats_city")
    op.drop_table("ensemble_stats")
    op.drop_table("ensemble_forecasts")
