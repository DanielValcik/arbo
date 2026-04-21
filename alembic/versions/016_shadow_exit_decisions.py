"""Shadow-exit decision log — paired ML vs fixed-rule exit telemetry.

For each open Strategy D position, at every check_exits tick, compute
what the ML exit model WOULD have said, and log alongside the real
rule-based decision. Used for paired-sample P(better) analysis on live
(or paper) positions without affecting real exits.

One row is emitted per position per "decision event":
  1. When ML first says should_exit=True during the position's life
  2. When the real fixed rule actually exits the position
  3. (Optional later) Periodic sampling for trajectory reconstruction

Table is write-once per (token_id, side, event_type, variant_id) — the
strategy keeps an in-memory dedup set; post-restart re-logs are expected
and acceptable.

Related:
  docs/STRATEGY_D_ML_DESIGN.md §15 Shadow-Exit Logger
  arbo/strategies/strategy_d_core.py::_log_shadow_exit_decision
  arbo/data/models/strategy_d_exit_v1.ubj (the model being evaluated)

Revision ID: 016
Revises: 015
"""
from alembic import op
import sqlalchemy as sa

revision = "016"
down_revision = "015"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "shadow_exit_decisions",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        # Identity
        sa.Column("strategy", sa.String(32), nullable=False),   # D / D_UFC / D_EPL
        sa.Column("variant_id", sa.String(64), nullable=True),  # active variant on position
        sa.Column("token_id", sa.String(80), nullable=False),
        sa.Column("condition_id", sa.String(80), nullable=True),
        sa.Column("side", sa.String(8), nullable=False),        # yes | no
        sa.Column("tick_ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("tick_idx", sa.Integer, nullable=True),       # n_price_checks at log time
        # Event type: what triggered this log entry
        #   "ml_first_exit" — ML said should_exit=True for the first time
        #   "real_exit"     — fixed rule actually exited position
        sa.Column("event_type", sa.String(20), nullable=False),
        # Market state at decision tick
        sa.Column("current_price", sa.Numeric(6, 4), nullable=True),
        sa.Column("entry_price", sa.Numeric(6, 4), nullable=True),
        sa.Column("unrealized_pnl_per_share", sa.Numeric(8, 4), nullable=True),
        sa.Column("hold_minutes", sa.Numeric(8, 2), nullable=True),
        # ML model decision
        sa.Column("ml_should_exit", sa.Boolean, nullable=False),
        sa.Column("ml_pred_log_t", sa.Numeric(14, 4), nullable=True),
        sa.Column("ml_threshold", sa.Numeric(14, 4), nullable=True),
        sa.Column("ml_reason", sa.String(64), nullable=True),
        sa.Column("ml_model_path", sa.String(256), nullable=True),
        # Real-rule decision snapshot (present when event_type='real_exit')
        sa.Column("real_exit_reason", sa.String(32), nullable=True),
        sa.Column("real_exit_price", sa.Numeric(6, 4), nullable=True),
        # Context
        sa.Column("game_date", sa.Date, nullable=True),
        sa.Column("sport", sa.String(16), nullable=True),
        sa.Column("team_a", sa.String(32), nullable=True),
        sa.Column("team_b", sa.String(32), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_index(
        "ix_shadow_exit_token_side",
        "shadow_exit_decisions",
        ["token_id", "side"],
    )
    op.create_index(
        "ix_shadow_exit_strategy_ts",
        "shadow_exit_decisions",
        ["strategy", "tick_ts"],
    )
    op.create_index(
        "ix_shadow_exit_event_type",
        "shadow_exit_decisions",
        ["event_type"],
    )


def downgrade() -> None:
    op.drop_index("ix_shadow_exit_event_type", table_name="shadow_exit_decisions")
    op.drop_index("ix_shadow_exit_strategy_ts", table_name="shadow_exit_decisions")
    op.drop_index("ix_shadow_exit_token_side", table_name="shadow_exit_decisions")
    op.drop_table("shadow_exit_decisions")
