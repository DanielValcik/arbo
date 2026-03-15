"""Add strategy column to paper_positions.

Revision ID: 008
Revises: 007
"""
from alembic import op
import sqlalchemy as sa

revision = "008"
down_revision = "007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "paper_positions",
        sa.Column("strategy", sa.String(8), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("paper_positions", "strategy")
