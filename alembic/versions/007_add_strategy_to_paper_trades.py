"""Add strategy column to paper_trades.

Revision ID: 007
Revises: 006
"""
from alembic import op
import sqlalchemy as sa

revision = "007"
down_revision = "006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "paper_trades",
        sa.Column("strategy", sa.String(8), nullable=True),
    )
    op.create_index("idx_paper_trades_strategy", "paper_trades", ["strategy"])


def downgrade() -> None:
    op.drop_index("idx_paper_trades_strategy", "paper_trades")
    op.drop_column("paper_trades", "strategy")
