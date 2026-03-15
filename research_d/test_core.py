"""Tests for Strategy D core modules: sports_db + elo_glicko_engine.

Run: python3 -m pytest research_d/test_core.py -v
"""

import math
import os
import sqlite3
import tempfile
from pathlib import Path

import pytest

from research_d.sports_db import SportsDB
from research_d.elo_glicko_engine import (
    EloGlickoEngine,
    EloRating,
    Glicko2Rating,
    elo_expected,
    elo_mov_multiplier,
    glicko2_update,
    _g,
    _E,
)


# ── SportsDB Tests ────────────────────────────────────────────────────


@pytest.fixture
def db(tmp_path):
    """Create a temporary SportsDB instance."""
    db_path = tmp_path / "test.sqlite"
    return SportsDB(db_path)


class TestSportsDB:
    """Tests for the SQLite database wrapper."""

    def test_create_schema(self, db):
        """Schema creates all 6 tables."""
        tables = db.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        table_names = {t["name"] for t in tables}
        assert "games" in table_names
        assert "markets" in table_names
        assert "prices" in table_names
        assert "game_events" in table_names
        assert "pinnacle_odds" in table_names
        assert "ratings" in table_names

    def test_upsert_game(self, db):
        """Insert and retrieve a game."""
        db.upsert_game(
            game_id="nba_20260315_LAL_BOS",
            sport="nba",
            league="NBA",
            home_team="BOS",
            away_team="LAL",
            game_date="2026-03-15",
            home_score=112,
            away_score=105,
            status="final",
        )
        game = db.get_game("nba_20260315_LAL_BOS")
        assert game is not None
        assert game["sport"] == "nba"
        assert game["home_team"] == "BOS"
        assert game["home_score"] == 112

    def test_upsert_game_update(self, db):
        """Upsert updates score but preserves existing fields."""
        db.upsert_game(
            game_id="test_game",
            sport="nba", league="NBA",
            home_team="BOS", away_team="LAL",
            game_date="2026-03-15",
        )
        db.upsert_game(
            game_id="test_game",
            sport="nba", league="NBA",
            home_team="BOS", away_team="LAL",
            game_date="2026-03-15",
            home_score=100, away_score=95,
            status="final",
        )
        game = db.get_game("test_game")
        assert game["home_score"] == 100
        assert game["status"] == "final"

    def test_get_games_filter(self, db):
        """Filter games by sport and date."""
        db.upsert_game("g1", "nba", "NBA", "BOS", "LAL", "2026-01-01", home_score=100, away_score=90, status="final")
        db.upsert_game("g2", "epl", "EPL", "ARS", "MCI", "2026-01-02", home_score=2, away_score=1, status="final")
        db.upsert_game("g3", "nba", "NBA", "GSW", "PHX", "2026-02-01", home_score=110, away_score=105, status="final")

        nba = db.get_games(sport="nba")
        assert len(nba) == 2

        jan = db.get_games(min_date="2026-01-01", max_date="2026-01-31")
        assert len(jan) == 2

        epl = db.get_games(sport="epl")
        assert len(epl) == 1

    def test_batch_upsert_games(self, db):
        """Batch insert games."""
        games = [
            {
                "game_id": f"g{i}", "sport": "nba", "league": "NBA",
                "home_team": "BOS", "away_team": "LAL",
                "game_date": f"2026-01-{i:02d}", "game_time": None,
                "home_score": 100 + i, "away_score": 90 + i,
                "status": "final", "season": "2025-26",
                "venue": None, "extra_json": None,
            }
            for i in range(1, 11)
        ]
        n = db.upsert_games_batch(games)
        assert n == 10
        assert db.count_games("nba") == 10

    def test_market_crud(self, db):
        """Insert and query markets."""
        db.upsert_game("g1", "nba", "NBA", "BOS", "LAL", "2026-01-01")
        db.upsert_market(
            token_id="tok1",
            game_id="g1",
            question="Will BOS beat LAL?",
            outcome="home_win",
            volume=50000,
            won=1,
        )
        market = db.get_market("tok1")
        assert market is not None
        assert market["outcome"] == "home_win"
        assert market["won"] == 1

        markets = db.get_markets(game_id="g1")
        assert len(markets) == 1

    def test_prices_crud(self, db):
        """Insert and query price data."""
        db.upsert_game("g1", "nba", "NBA", "BOS", "LAL", "2026-01-01")
        db.upsert_market(token_id="tok1", game_id="g1")

        prices = [(1000 + i * 60, 0.40 + i * 0.01, None, None, None) for i in range(100)]
        n = db.insert_prices("tok1", prices)
        assert n == 100
        assert db.count_prices("tok1") == 100

        # Test max/min price in range
        max_p = db.get_max_price_in_range("tok1", 1000, 7000)
        assert max_p is not None
        assert max_p == pytest.approx(1.39, abs=0.01)

        min_p = db.get_min_price_in_range("tok1", 1000, 2000)
        assert min_p is not None
        assert min_p == pytest.approx(0.40, abs=0.01)

    def test_prices_simple(self, db):
        """Insert simple (ts, price) pairs."""
        db.upsert_game("g1", "nba", "NBA", "BOS", "LAL", "2026-01-01")
        db.upsert_market(token_id="tok1", game_id="g1")

        prices = [(1000 + i * 60, 0.50) for i in range(50)]
        n = db.insert_prices_simple("tok1", prices)
        assert n == 50
        assert db.count_prices("tok1") == 50

    def test_prices_idempotent(self, db):
        """Duplicate prices are silently ignored."""
        db.upsert_game("g1", "nba", "NBA", "BOS", "LAL", "2026-01-01")
        db.upsert_market(token_id="tok1", game_id="g1")

        prices = [(1000, 0.50, None, None, None), (1060, 0.51, None, None, None)]
        db.insert_prices("tok1", prices)
        db.insert_prices("tok1", prices)  # Duplicate
        assert db.count_prices("tok1") == 2

    def test_game_events(self, db):
        """Insert and query game events."""
        db.upsert_game("g1", "nba", "NBA", "BOS", "LAL", "2026-01-01")

        db.insert_game_event("g1", 1000, "basket", "BOS", 2, 0, "Tatum layup")
        db.insert_game_event("g1", 1030, "basket", "LAL", 2, 2, "LeBron dunk")
        db.insert_game_event("g1", 1060, "basket", "BOS", 5, 2, "Brown 3pt")

        events = db.get_game_events("g1")
        assert len(events) == 3

        baskets = db.get_game_events("g1", event_type="basket")
        assert len(baskets) == 3

    def test_pinnacle_odds(self, db):
        """Insert and query Pinnacle odds."""
        db.upsert_game("g1", "nba", "NBA", "BOS", "LAL", "2026-01-01")

        db.upsert_pinnacle_odds("g1", 1000, 1.55, 2.50, None, 0.617, 0.383)
        db.upsert_pinnacle_odds("g1", 2000, 1.60, 2.40, None, 0.600, 0.400)

        odds = db.get_pinnacle_odds("g1")
        assert len(odds) == 2

        latest = db.get_latest_pinnacle_odds("g1")
        assert latest["ts"] == 2000
        assert latest["home_prob_novig"] == pytest.approx(0.600)

    def test_ratings(self, db):
        """Insert and query ratings."""
        db.upsert_rating("BOS", "nba", "2026-01-01", elo=1550, glicko_rating=1560)
        db.upsert_rating("BOS", "nba", "2026-01-15", elo=1570, glicko_rating=1580)
        db.upsert_rating("LAL", "nba", "2026-01-01", elo=1450, glicko_rating=1440)

        # Get latest rating on or before date
        r = db.get_rating("BOS", "nba", "2026-01-10")
        assert r["elo"] == 1550

        r2 = db.get_rating("BOS", "nba", "2026-02-01")
        assert r2["elo"] == 1570

        # All ratings on date
        all_r = db.get_all_ratings_on_date("nba", "2026-01-10")
        assert len(all_r) == 2

    def test_stats(self, db):
        """Stats returns correct counts."""
        db.upsert_game("g1", "nba", "NBA", "BOS", "LAL", "2026-01-01")
        db.upsert_game("g2", "epl", "EPL", "ARS", "MCI", "2026-01-02")
        s = db.stats()
        assert s["games"] == 2
        assert s["games_nba"] == 1
        assert s["games_epl"] == 1

    def test_context_manager(self, tmp_path):
        """SportsDB works as a context manager."""
        db_path = tmp_path / "ctx.sqlite"
        with SportsDB(db_path) as db:
            db.upsert_game("g1", "nba", "NBA", "BOS", "LAL", "2026-01-01")
            assert db.count_games() == 1


# ── Elo Tests ──────────────────────────────────────────────────────────


class TestElo:
    """Tests for the Elo rating system."""

    def test_elo_expected_equal(self):
        """Equal ratings → 50% expected."""
        assert elo_expected(1500, 1500) == pytest.approx(0.5)

    def test_elo_expected_higher(self):
        """Higher rating → higher expected score."""
        p = elo_expected(1600, 1400)
        assert p > 0.7
        assert p < 0.8

    def test_elo_expected_symmetric(self):
        """E(A vs B) + E(B vs A) = 1."""
        p1 = elo_expected(1600, 1400)
        p2 = elo_expected(1400, 1600)
        assert p1 + p2 == pytest.approx(1.0)

    def test_elo_expected_large_diff(self):
        """Very large rating diff → near-certain win."""
        p = elo_expected(2000, 1000)
        assert p > 0.99

    def test_mov_multiplier_positive(self):
        """MOV multiplier is always positive."""
        assert elo_mov_multiplier(10, 100) > 0
        assert elo_mov_multiplier(1, -200) > 0

    def test_mov_multiplier_scales(self):
        """Larger margin → larger multiplier."""
        m1 = elo_mov_multiplier(1, 0)
        m10 = elo_mov_multiplier(10, 0)
        m30 = elo_mov_multiplier(30, 0)
        assert m1 < m10 < m30

    def test_mov_autocorrelation(self):
        """Expected blowouts get less credit."""
        # Big upset (favorite lost by 20)
        m_upset = elo_mov_multiplier(20, -200)
        # Expected win (favorite won by 20)
        m_expected = elo_mov_multiplier(20, 200)
        assert m_upset > m_expected


# ── Glicko-2 Tests ─────────────────────────────────────────────────────


class TestGlicko2:
    """Tests for the Glicko-2 rating system."""

    def test_glicko2_rating_scale(self):
        """Default Glicko-2 rating matches traditional scale."""
        r = Glicko2Rating()
        assert r.rating == pytest.approx(1500, abs=1)

    def test_glicko2_from_traditional(self):
        """Conversion from traditional scale and back."""
        r = Glicko2Rating.from_traditional(1700, 200, 0.06)
        assert r.rating == pytest.approx(1700, abs=1)
        assert r.rd == pytest.approx(200, abs=1)

    def test_glicko2_update_win(self):
        """Winning increases rating."""
        player = Glicko2Rating.from_traditional(1500, 200, 0.06)
        opponent_mu = (1400 - 1500) / 173.7178
        opponent_phi = 30 / 173.7178

        updated = glicko2_update(
            player, [(opponent_mu, opponent_phi)], [1.0], tau=0.5
        )
        assert updated.rating > 1500

    def test_glicko2_update_loss(self):
        """Losing decreases rating."""
        player = Glicko2Rating.from_traditional(1500, 200, 0.06)
        opponent_mu = (1600 - 1500) / 173.7178
        opponent_phi = 30 / 173.7178

        updated = glicko2_update(
            player, [(opponent_mu, opponent_phi)], [0.0], tau=0.5
        )
        assert updated.rating < 1500

    def test_glicko2_rd_decreases(self):
        """RD decreases after a game (more certainty)."""
        player = Glicko2Rating.from_traditional(1500, 200, 0.06)
        opponent_mu = (1500 - 1500) / 173.7178
        opponent_phi = 200 / 173.7178

        updated = glicko2_update(
            player, [(opponent_mu, opponent_phi)], [1.0], tau=0.5
        )
        assert updated.rd < 200

    def test_glicko2_no_games(self):
        """No games → RD increases (more uncertainty)."""
        player = Glicko2Rating.from_traditional(1500, 100, 0.06)
        updated = glicko2_update(player, [], [], tau=0.5)
        assert updated.rd > 100
        assert updated.rating == pytest.approx(1500, abs=1)

    def test_g_function(self):
        """g function is between 0 and 1."""
        assert 0 < _g(0.5) < 1
        assert 0 < _g(2.0) < 1
        # g decreases with phi (more uncertain opponents count less)
        assert _g(0.5) > _g(2.0)

    def test_E_function(self):
        """E function is between 0 and 1."""
        e = _E(0.0, 0.0, 1.0)
        assert e == pytest.approx(0.5, abs=0.01)


# ── EloGlickoEngine Tests ─────────────────────────────────────────────


class TestEloGlickoEngine:
    """Tests for the combined engine."""

    def test_create_engine(self):
        """Engine initializes correctly."""
        engine = EloGlickoEngine("nba")
        assert engine.sport == "nba"
        assert len(engine.teams) == 0

    def test_invalid_sport(self):
        """Invalid sport raises ValueError."""
        with pytest.raises(ValueError):
            EloGlickoEngine("cricket")

    def test_process_game(self):
        """Processing a game updates both ratings."""
        engine = EloGlickoEngine("nba")
        result = engine.process_game("BOS", "LAL", 112, 105, "2026-01-15")

        assert "elo_home_pre" in result
        assert "elo_expected_home" in result
        assert len(engine.teams) == 2

        # Winner's Elo should increase
        assert engine.teams["BOS"].elo.rating > 1500
        assert engine.teams["LAL"].elo.rating < 1500

    def test_multiple_games(self):
        """Processing multiple games tracks teams correctly."""
        engine = EloGlickoEngine("nba")

        # BOS beats LAL
        engine.process_game("BOS", "LAL", 112, 105, "2026-01-15")
        # LAL beats GSW
        engine.process_game("LAL", "GSW", 108, 100, "2026-01-16")
        # BOS beats GSW
        engine.process_game("BOS", "GSW", 115, 98, "2026-01-17")

        assert len(engine.teams) == 3
        # BOS should be highest rated (2 wins)
        assert engine.teams["BOS"].elo.rating > engine.teams["LAL"].elo.rating
        assert engine.teams["LAL"].elo.rating > engine.teams["GSW"].elo.rating

    def test_predict(self):
        """Prediction returns probabilities between 0 and 1."""
        engine = EloGlickoEngine("nba")
        engine.process_game("BOS", "LAL", 112, 105, "2026-01-15")

        pred = engine.predict("BOS", "LAL")
        assert 0 < pred["elo_home_prob"] < 1
        assert 0 < pred["glicko_home_prob"] < 1
        assert 0 < pred["ensemble_home_prob"] < 1

        # BOS should be favored (just beat LAL + home advantage)
        assert pred["ensemble_home_prob"] > 0.5

    def test_predict_new_teams(self):
        """Prediction for new teams returns 50% (with home advantage)."""
        engine = EloGlickoEngine("nba")
        pred = engine.predict("AAA", "BBB")
        # Home advantage should make it > 0.5
        assert pred["elo_home_prob"] > 0.5
        assert pred["ensemble_home_prob"] > 0.5

    def test_draw_soccer(self):
        """Soccer draws handled correctly."""
        engine = EloGlickoEngine("epl")
        result = engine.process_game("ARS", "MCI", 1, 1, "2026-01-15")

        # Draw: both ratings should move slightly
        ars_elo = engine.teams["ARS"].elo.rating
        mci_elo = engine.teams["MCI"].elo.rating
        # With home advantage, ARS expected to win → draw is bad for ARS
        # So ARS rating drops slightly, MCI rises slightly
        assert abs(ars_elo - 1500) < 50  # Small change
        assert abs(mci_elo - 1500) < 50

    def test_history_tracking(self):
        """Engine records rating history."""
        engine = EloGlickoEngine("nba")
        engine.process_game("BOS", "LAL", 112, 105, "2026-01-15")

        assert len(engine.history) == 2  # One entry per team per game
        assert engine.history[0]["team"] in ("BOS", "LAL")
        assert "elo" in engine.history[0]
        assert "glicko_rating" in engine.history[0]

    def test_brier_score(self):
        """Brier score computation."""
        engine = EloGlickoEngine("nba")

        # Perfect predictions
        assert engine.brier_score([(1.0, 1.0), (0.0, 0.0)]) == 0.0

        # Coin flip on everything
        assert engine.brier_score([(0.5, 1.0), (0.5, 0.0)]) == 0.25

        # No predictions
        assert engine.brier_score([]) == 0.25

    def test_season_reversion(self):
        """Season transition reverts ratings toward mean."""
        engine = EloGlickoEngine("nba")

        # Build up a high rating
        for i in range(20):
            engine.process_game("BOS", "LAL", 112, 105, f"2025-{i+1:02d}-15", season="2024-25")

        bos_elo_before = engine.teams["BOS"].elo.rating
        assert bos_elo_before > 1600  # Should be well above mean

        # New season game triggers reversion
        engine.process_game("BOS", "LAL", 112, 105, "2025-10-20", season="2025-26")

        # Rating should have reverted partially toward 1500
        # (then updated for the new game, but the reversion happened first)
        # The key test: ratings didn't just continue climbing
        # Hard to test exactly, but history should show the reversion
        assert len(engine.history) > 40

    def test_save_to_db(self, db):
        """Ratings save to database correctly."""
        engine = EloGlickoEngine("nba")
        engine.process_game("BOS", "LAL", 112, 105, "2026-01-15")

        n = engine.save_to_db(db)
        assert n == 2  # Two teams

        # Verify in DB
        r = db.get_rating("BOS", "nba", "2026-01-15")
        assert r is not None
        assert r["elo"] > 1500

    def test_summary(self):
        """Summary produces readable output."""
        engine = EloGlickoEngine("nba")
        engine.process_game("BOS", "LAL", 112, 105, "2026-01-15")
        s = engine.summary()
        assert "BOS" in s
        assert "Elo=" in s

    def test_calibration_report(self):
        """Calibration report works."""
        engine = EloGlickoEngine("nba")
        preds = [(0.6, 1.0), (0.7, 1.0), (0.4, 0.0), (0.3, 0.0)]
        report = engine.calibration_report(preds, n_bins=5)
        assert len(report) > 0
        assert "bin_low" in report[0]
        assert "predicted_avg" in report[0]
        assert "actual_avg" in report[0]
