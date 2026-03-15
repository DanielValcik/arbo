"""Add unique constraint to weather_forecasts for upsert support.

Enables ON CONFLICT DO UPDATE for forecast persistence — same city+date+source
gets updated with latest forecast values instead of creating duplicates.

Revision ID: 010
Revises: 009
"""
from alembic import op

revision = "010"
down_revision = "009"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_unique_constraint(
        "uq_weather_city_date_source",
        "weather_forecasts",
        ["city", "forecast_date", "source"],
    )


def downgrade() -> None:
    op.drop_constraint("uq_weather_city_date_source", "weather_forecasts")
