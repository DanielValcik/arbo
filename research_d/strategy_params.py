"""Strategy D tunable parameters.

This is the ONLY file the autoresearch agent may modify.
"""

PARAMS = {
    # ── Quality Gate (D1) — Sweep winner #44 baseline ─────────────────
    "min_edge": 0.12,              # Sweep winner: 0.12
    "max_edge": 0.25,
    "min_price": 0.20,
    "max_price": 0.65,
    "min_volume": 0,
    "min_prices": 20,
    "competitive_threshold": 999,

    # ── Green Book ────────────────────────────────────────────────────
    "green_book_enabled": True,
    "green_book_delta_nba": 0.15,  # Sweep winner: 0.15
    "green_book_delta_epl": 0.10,
    "green_book_delta_nfl": 0.08,
    "green_book_delta_ufc": 0.12,
    "green_book_delta_default": 0.15,
    "stop_loss_enabled": True,
    "stop_loss_delta": 0.15,       # Sweep winner: 0.15

    # ── Probability Model ─────────────────────────────────────────────
    "elo_weight": 0.40,
    "pinnacle_weight": 0.60,
    "elo_only_weight_elo": 0.45,
    "elo_only_weight_glicko": 0.55,

    # ── Position Sizing ───────────────────────────────────────────────
    "initial_capital": 1000.0,
    "kelly_fraction": 0.15,
    "kelly_raw_cap": 0.10,
    "max_position_pct": 0.03,

    # ── Both Sides Trading ─────────────────────────────────────────────
    "both_sides": True,            # Also buy NO when model < market

    # ── Sport Selection ───────────────────────────────────────────────
    "enabled_sports": ["nba"],
    "excluded_teams": [],

    # ── Time Exit (Always Close) ─────────────────────────────────────
    "max_hold_fraction": 0.70,     # Exit at 70% of game if no GB/SL (never hold to resolution)

    # ── Walk-Forward ──────────────────────────────────────────────────
    "wf_train_months": 3,
    "wf_test_months": 1,
}
