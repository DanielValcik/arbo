"""Shadow variant signals table — Project PARALLEL Phase 2A.2.

Stores per-variant decisions for every signal that flows through the
ShadowOrchestrator. Champion's decision is also logged here (mirror of
its paper_trades row) so champion vs challenger comparisons are paired.

One row = (one signal × one variant).

Spec: docs/PROJECT_PARALLEL_ROADMAP.md §Phase 2A
Spec: docs/VARIANT_LEADERBOARD_SPEC.md §3b

Revision ID: 013
Revises: 012
"""

from alembic import op
import sqlalchemy as sa

revision = "013"
down_revision = "012"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "shadow_variant_signals",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("strategy", sa.String(32), nullable=False),
        sa.Column("variant_id", sa.String(64), nullable=False),
        sa.Column("condition_id", sa.String(80), nullable=False),
        sa.Column("token_id", sa.String(80), nullable=True),
        sa.Column("signal_ts", sa.DateTime(timezone=True), nullable=False),
        # Decision
        sa.Column("qualified", sa.Boolean, nullable=False),
        sa.Column("skip_reason", sa.String(64), nullable=True),
        sa.Column("direction", sa.String(8), nullable=True),  # "up" | "down"
        # Signal context
        sa.Column("entry_price", sa.Float, nullable=True),
        sa.Column("edge", sa.Float, nullable=True),
        sa.Column("sigma", sa.Float, nullable=True),
        sa.Column("btc_at_start", sa.Float, nullable=True),
        sa.Column("btc_now", sa.Float, nullable=True),
        sa.Column("btc_move", sa.Float, nullable=True),
        sa.Column("market_gap", sa.Float, nullable=True),
        sa.Column("velocity", sa.Float, nullable=True),
        sa.Column("dir_delta", sa.Float, nullable=True),
        sa.Column("would_fill_at", sa.Float, nullable=True),
        # Resolution (filled in later by backfill when event resolves)
        sa.Column("resolution_outcome", sa.Boolean, nullable=True),  # True=UP won
        sa.Column("resolution_ts", sa.DateTime(timezone=True), nullable=True),
        sa.Column("would_pnl_per_share", sa.Float, nullable=True),
        # Provenance
        sa.Column("event_start_ts", sa.Float, nullable=True),
        sa.Column("event_end_ts", sa.Float, nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("NOW()"),
            nullable=False,
        ),
    )
    # Primary lookup index — leaderboard queries by (strategy, variant_id)
    op.create_index(
        "idx_svs_variant_ts",
        "shadow_variant_signals",
        ["strategy", "variant_id", "signal_ts"],
        postgresql_using="btree",
    )
    # Backfill index — resolution updater queries by (strategy, condition_id)
    op.create_index(
        "idx_svs_strategy_cid",
        "shadow_variant_signals",
        ["strategy", "condition_id"],
    )
    # Unique constraint — same variant cannot fire twice on same signal
    # (defined by condition_id + entry minute via signal_ts truncation)
    op.create_unique_constraint(
        "uq_svs_variant_signal",
        "shadow_variant_signals",
        ["strategy", "variant_id", "condition_id", "signal_ts"],
    )


def downgrade() -> None:
    op.drop_constraint("uq_svs_variant_signal", "shadow_variant_signals", type_="unique")
    op.drop_index("idx_svs_strategy_cid", table_name="shadow_variant_signals")
    op.drop_index("idx_svs_variant_ts", table_name="shadow_variant_signals")
    op.drop_table("shadow_variant_signals")
