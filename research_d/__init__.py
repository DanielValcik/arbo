"""Strategy D: Live Edge Harvester (LEH) — Research & Backtest Module.

This package contains all research, data collection, backtesting, and
optimization code for Strategy D.

Modules:
    sports_db               — SQLite schema + CRUD helpers
    elo_glicko_engine       — Elo + Glicko-2 team strength ratings
    download_game_results   — NBA + EPL game results downloader
    download_sports_prices  — Polymarket sports price history downloader
    download_pinnacle_odds  — Pinnacle odds via The Odds API
    validate_data           — Cross-validation and data quality checks

Sprint D-0 (Data Infrastructure):
    D-001: download_sports_prices.py    — Polymarket CLOB price history
    D-003: download_game_results.py     — ESPN/football-data.co.uk game results
    D-004: download_pinnacle_odds.py    — Pinnacle odds via The Odds API
    D-005: elo_glicko_engine.py         — Elo/Glicko-2 rating engine
    D-006: sports_db.py                 — SQLite schema + data pipeline
    D-007: validate_data.py             — Data validation

Usage:
    # Download data
    python3 research_d/download_game_results.py --sport all
    python3 research_d/download_sports_prices.py --sport all
    python3 research_d/download_pinnacle_odds.py --sport all

    # Compute ratings
    python3 research_d/elo_glicko_engine.py --sport all

    # Validate
    python3 research_d/validate_data.py
"""
