"""Add exit_price and exit_reason columns to paper_trades.

Supports Strategy C2 early exit (sell before market resolution).

Revision ID: 012
Revises: 011
"""

from alembic import op
import sqlalchemy as sa

revision = "012"
down_revision = "011"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("paper_trades", sa.Column("exit_price", sa.Numeric(8, 4), nullable=True))
    op.add_column("paper_trades", sa.Column("exit_reason", sa.String(32), nullable=True))


def downgrade() -> None:
    op.drop_column("paper_trades", "exit_reason")
    op.drop_column("paper_trades", "exit_price")
