"""Shadow variant signals — model_prob + meta_json columns.

Project PARALLEL extension to B2 + D_NBA Phase 1.

Adds two NULLABLE columns to shadow_variant_signals so non-B3 strategies
can store strategy-specific signal context without overloading existing
columns:

- model_prob FLOAT — model's predicted probability (Elo+Pinnacle for D,
  log-normal CDF for B2). Stored separately from `entry_price` (which is
  the would-fill price).
- meta_json JSONB — free-form per-strategy context (e.g. team_a/team_b
  for D, asset/strike/direction for B2). Avoids overloading existing
  columns and gives auto-challenger Gemini richer context.

Backward-compatible: all existing INSERTs continue to work unchanged.

Spec: docs/PROJECT_PARALLEL_B2_DNBA_PLAN.md §5.2

Revision ID: 014
Revises: 013
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "014"
down_revision = "013"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "shadow_variant_signals",
        sa.Column("model_prob", sa.Float, nullable=True),
    )
    op.add_column(
        "shadow_variant_signals",
        sa.Column("meta_json", postgresql.JSONB, nullable=True),
    )


def downgrade() -> None:
    op.drop_column("shadow_variant_signals", "meta_json")
    op.drop_column("shadow_variant_signals", "model_prob")
