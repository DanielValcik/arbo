"""Strategy D — Sports Backtest Database.

SQLite schema and CRUD helpers for the Live Edge Harvester (LEH) backtest.
Stores games, Polymarket markets, minute-level prices, live game events,
Pinnacle odds, and Elo/Glicko-2 ratings.

Tables:
    games          — Individual sports matches (NBA, EPL, NFL, etc.)
    markets        — Polymarket binary contracts linked to games
    prices         — Minute-level Polymarket price history during games
    game_events    — Live scoring events (goals, baskets, touchdowns)
    pinnacle_odds  — Pinnacle no-vig odds history per game
    ratings        — Elo/Glicko-2 team strength ratings over time

Usage:
    from research_d.sports_db import SportsDB

    db = SportsDB()                          # default: research_d/data/sports_backtest.sqlite
    db = SportsDB("path/to/custom.sqlite")   # custom path

    db.upsert_game(game_id="nba_20260315_LAL_BOS", sport="nba", ...)
    db.upsert_market(token_id="abc123", game_id="nba_20260315_LAL_BOS", ...)
    db.insert_prices("abc123", [(ts1, 0.45, 0.44, 0.46, 1200.0), ...])
    db.insert_game_event("nba_20260315_LAL_BOS", ts, "basket", "LAL", 24, 22, "LeBron 3pt")

    games = db.get_games(sport="nba", min_date="2025-10-01")
    prices = db.get_prices("abc123", start_ts=..., end_ts=...)
    db.close()
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).parent / "data"
DEFAULT_DB_PATH = DATA_DIR / "sports_backtest.sqlite"


class SportsDB:
    """SQLite wrapper for Strategy D sports backtest data."""

    def __init__(self, db_path: str | Path | None = None):
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self._configure()
        self._create_schema()

    def _configure(self) -> None:
        """Set SQLite performance pragmas."""
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA cache_size=-64000")  # 64MB cache
        self.conn.execute("PRAGMA foreign_keys=ON")

    def _create_schema(self) -> None:
        """Create all tables and indexes."""
        self.conn.executescript(_SCHEMA_SQL)
        self.conn.commit()

    # ── Games ──────────────────────────────────────────────────────────

    def upsert_game(
        self,
        game_id: str,
        sport: str,
        league: str,
        home_team: str,
        away_team: str,
        game_date: str,
        game_time: str | None = None,
        home_score: int | None = None,
        away_score: int | None = None,
        status: str = "scheduled",
        season: str | None = None,
        venue: str | None = None,
        extra_json: str | None = None,
    ) -> None:
        """Insert or update a game record."""
        self.conn.execute(
            """INSERT INTO games (
                game_id, sport, league, home_team, away_team,
                game_date, game_time, home_score, away_score,
                status, season, venue, extra_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(game_id) DO UPDATE SET
                home_score = COALESCE(excluded.home_score, games.home_score),
                away_score = COALESCE(excluded.away_score, games.away_score),
                status = excluded.status,
                extra_json = COALESCE(excluded.extra_json, games.extra_json)
            """,
            (
                game_id, sport, league, home_team, away_team,
                game_date, game_time, home_score, away_score,
                status, season, venue, extra_json,
            ),
        )
        self.conn.commit()

    def upsert_games_batch(self, games: list[dict[str, Any]]) -> int:
        """Batch upsert games. Returns count inserted/updated."""
        self.conn.executemany(
            """INSERT INTO games (
                game_id, sport, league, home_team, away_team,
                game_date, game_time, home_score, away_score,
                status, season, venue, extra_json
            ) VALUES (
                :game_id, :sport, :league, :home_team, :away_team,
                :game_date, :game_time, :home_score, :away_score,
                :status, :season, :venue, :extra_json
            )
            ON CONFLICT(game_id) DO UPDATE SET
                home_score = COALESCE(excluded.home_score, games.home_score),
                away_score = COALESCE(excluded.away_score, games.away_score),
                status = excluded.status,
                extra_json = COALESCE(excluded.extra_json, games.extra_json)
            """,
            games,
        )
        self.conn.commit()
        return len(games)

    def get_games(
        self,
        sport: str | None = None,
        league: str | None = None,
        min_date: str | None = None,
        max_date: str | None = None,
        status: str | None = None,
        team: str | None = None,
    ) -> list[sqlite3.Row]:
        """Query games with optional filters."""
        conditions: list[str] = []
        params: list[Any] = []

        if sport:
            conditions.append("sport = ?")
            params.append(sport)
        if league:
            conditions.append("league = ?")
            params.append(league)
        if min_date:
            conditions.append("game_date >= ?")
            params.append(min_date)
        if max_date:
            conditions.append("game_date <= ?")
            params.append(max_date)
        if status:
            conditions.append("status = ?")
            params.append(status)
        if team:
            conditions.append("(home_team = ? OR away_team = ?)")
            params.extend([team, team])

        where = " AND ".join(conditions) if conditions else "1=1"
        return self.conn.execute(
            f"SELECT * FROM games WHERE {where} ORDER BY game_date, game_time",
            params,
        ).fetchall()

    def get_game(self, game_id: str) -> sqlite3.Row | None:
        """Get a single game by ID."""
        return self.conn.execute(
            "SELECT * FROM games WHERE game_id = ?", (game_id,)
        ).fetchone()

    def count_games(self, sport: str | None = None) -> int:
        """Count games, optionally filtered by sport."""
        if sport:
            row = self.conn.execute(
                "SELECT COUNT(*) FROM games WHERE sport = ?", (sport,)
            ).fetchone()
        else:
            row = self.conn.execute("SELECT COUNT(*) FROM games").fetchone()
        return row[0]

    # ── Markets ────────────────────────────────────────────────────────

    def upsert_market(
        self,
        token_id: str,
        game_id: str,
        event_id: str | None = None,
        condition_id: str | None = None,
        token_id_no: str | None = None,
        question: str | None = None,
        outcome: str | None = None,
        volume: float | None = None,
        neg_risk: int = 0,
        won: int | None = None,
        end_date: str | None = None,
        extra_json: str | None = None,
    ) -> None:
        """Insert or update a Polymarket market."""
        self.conn.execute(
            """INSERT INTO markets (
                token_id, token_id_no, game_id, event_id, condition_id,
                question, outcome, volume, neg_risk, won, end_date, extra_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(token_id) DO UPDATE SET
                volume = COALESCE(excluded.volume, markets.volume),
                won = COALESCE(excluded.won, markets.won),
                end_date = COALESCE(excluded.end_date, markets.end_date),
                extra_json = COALESCE(excluded.extra_json, markets.extra_json)
            """,
            (
                token_id, token_id_no, game_id, event_id, condition_id,
                question, outcome, volume, neg_risk, won, end_date, extra_json,
            ),
        )
        self.conn.commit()

    def upsert_markets_batch(self, markets: list[dict[str, Any]]) -> int:
        """Batch upsert markets."""
        self.conn.executemany(
            """INSERT INTO markets (
                token_id, token_id_no, game_id, event_id, condition_id,
                question, outcome, volume, neg_risk, won, end_date, extra_json
            ) VALUES (
                :token_id, :token_id_no, :game_id, :event_id, :condition_id,
                :question, :outcome, :volume, :neg_risk, :won, :end_date, :extra_json
            )
            ON CONFLICT(token_id) DO UPDATE SET
                volume = COALESCE(excluded.volume, markets.volume),
                won = COALESCE(excluded.won, markets.won),
                end_date = COALESCE(excluded.end_date, markets.end_date),
                extra_json = COALESCE(excluded.extra_json, markets.extra_json)
            """,
            markets,
        )
        self.conn.commit()
        return len(markets)

    def get_markets(
        self,
        game_id: str | None = None,
        sport: str | None = None,
        won: int | None = None,
    ) -> list[sqlite3.Row]:
        """Query markets with optional filters."""
        conditions: list[str] = []
        params: list[Any] = []

        if game_id:
            conditions.append("m.game_id = ?")
            params.append(game_id)
        if sport:
            conditions.append("g.sport = ?")
            params.append(sport)
        if won is not None:
            conditions.append("m.won = ?")
            params.append(won)

        where = " AND ".join(conditions) if conditions else "1=1"
        return self.conn.execute(
            f"""SELECT m.*, g.sport, g.league, g.home_team, g.away_team,
                       g.game_date, g.home_score, g.away_score
                FROM markets m
                JOIN games g ON m.game_id = g.game_id
                WHERE {where}
                ORDER BY g.game_date""",
            params,
        ).fetchall()

    def get_market(self, token_id: str) -> sqlite3.Row | None:
        """Get a single market by token_id."""
        return self.conn.execute(
            "SELECT * FROM markets WHERE token_id = ?", (token_id,)
        ).fetchone()

    def count_markets(self, sport: str | None = None) -> int:
        """Count markets, optionally filtered by sport (via join)."""
        if sport:
            row = self.conn.execute(
                """SELECT COUNT(*) FROM markets m
                   JOIN games g ON m.game_id = g.game_id
                   WHERE g.sport = ?""",
                (sport,),
            ).fetchone()
        else:
            row = self.conn.execute("SELECT COUNT(*) FROM markets").fetchone()
        return row[0]

    # ── Prices ─────────────────────────────────────────────────────────

    def insert_prices(
        self,
        token_id: str,
        prices: list[tuple[int, float, float | None, float | None, float | None]],
    ) -> int:
        """Batch insert price records: (ts, price, bid, ask, volume_1m).

        Duplicates are silently ignored (INSERT OR IGNORE).
        """
        self.conn.executemany(
            """INSERT OR IGNORE INTO prices (token_id, ts, price, bid, ask, volume_1m)
               VALUES (?, ?, ?, ?, ?, ?)""",
            [(token_id, ts, p, bid, ask, vol) for ts, p, bid, ask, vol in prices],
        )
        self.conn.commit()
        return len(prices)

    def insert_prices_simple(
        self,
        token_id: str,
        prices: list[tuple[int, float]],
    ) -> int:
        """Batch insert simple price records: (ts, price).

        Used when only timestamp and price are available (e.g., CLOB /prices-history).
        """
        self.conn.executemany(
            """INSERT OR IGNORE INTO prices (token_id, ts, price)
               VALUES (?, ?, ?)""",
            [(token_id, ts, p) for ts, p in prices],
        )
        self.conn.commit()
        return len(prices)

    def get_prices(
        self,
        token_id: str,
        start_ts: int | None = None,
        end_ts: int | None = None,
    ) -> list[sqlite3.Row]:
        """Get price history for a token, optionally filtered by time range."""
        conditions = ["token_id = ?"]
        params: list[Any] = [token_id]

        if start_ts:
            conditions.append("ts >= ?")
            params.append(start_ts)
        if end_ts:
            conditions.append("ts <= ?")
            params.append(end_ts)

        where = " AND ".join(conditions)
        return self.conn.execute(
            f"SELECT * FROM prices WHERE {where} ORDER BY ts", params
        ).fetchall()

    def get_price_range(self, token_id: str) -> tuple[int, int] | None:
        """Get (min_ts, max_ts) for a token's price data."""
        row = self.conn.execute(
            "SELECT MIN(ts), MAX(ts) FROM prices WHERE token_id = ?",
            (token_id,),
        ).fetchone()
        if row and row[0] is not None:
            return (row[0], row[1])
        return None

    def count_prices(self, token_id: str | None = None) -> int:
        """Count price records, optionally for a specific token."""
        if token_id:
            row = self.conn.execute(
                "SELECT COUNT(*) FROM prices WHERE token_id = ?", (token_id,)
            ).fetchone()
        else:
            row = self.conn.execute("SELECT COUNT(*) FROM prices").fetchone()
        return row[0]

    def get_price_at(self, token_id: str, ts: int) -> float | None:
        """Get the price closest to (but not after) a given timestamp."""
        row = self.conn.execute(
            """SELECT price FROM prices
               WHERE token_id = ? AND ts <= ?
               ORDER BY ts DESC LIMIT 1""",
            (token_id, ts),
        ).fetchone()
        return row["price"] if row else None

    def get_max_price_in_range(
        self, token_id: str, start_ts: int, end_ts: int
    ) -> float | None:
        """Get the maximum price in a time range. Used for green book simulation."""
        row = self.conn.execute(
            """SELECT MAX(price) as max_price FROM prices
               WHERE token_id = ? AND ts >= ? AND ts <= ?""",
            (token_id, start_ts, end_ts),
        ).fetchone()
        return row["max_price"] if row else None

    def get_min_price_in_range(
        self, token_id: str, start_ts: int, end_ts: int
    ) -> float | None:
        """Get the minimum price in a time range."""
        row = self.conn.execute(
            """SELECT MIN(price) as min_price FROM prices
               WHERE token_id = ? AND ts >= ? AND ts <= ?""",
            (token_id, start_ts, end_ts),
        ).fetchone()
        return row["min_price"] if row else None

    # ── Game Events ────────────────────────────────────────────────────

    def insert_game_event(
        self,
        game_id: str,
        ts: int,
        event_type: str,
        team: str | None = None,
        score_home: int | None = None,
        score_away: int | None = None,
        detail: str | None = None,
    ) -> None:
        """Insert a live game event (goal, basket, touchdown, etc.)."""
        self.conn.execute(
            """INSERT OR IGNORE INTO game_events
               (game_id, ts, event_type, team, score_home, score_away, detail)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (game_id, ts, event_type, team, score_home, score_away, detail),
        )
        self.conn.commit()

    def insert_game_events_batch(
        self, events: list[dict[str, Any]]
    ) -> int:
        """Batch insert game events."""
        self.conn.executemany(
            """INSERT OR IGNORE INTO game_events
               (game_id, ts, event_type, team, score_home, score_away, detail)
               VALUES (:game_id, :ts, :event_type, :team,
                       :score_home, :score_away, :detail)""",
            events,
        )
        self.conn.commit()
        return len(events)

    def get_game_events(
        self,
        game_id: str,
        event_type: str | None = None,
    ) -> list[sqlite3.Row]:
        """Get events for a game, optionally filtered by type."""
        if event_type:
            return self.conn.execute(
                """SELECT * FROM game_events
                   WHERE game_id = ? AND event_type = ?
                   ORDER BY ts""",
                (game_id, event_type),
            ).fetchall()
        return self.conn.execute(
            "SELECT * FROM game_events WHERE game_id = ? ORDER BY ts",
            (game_id,),
        ).fetchall()

    # ── Pinnacle Odds ──────────────────────────────────────────────────

    def upsert_pinnacle_odds(
        self,
        game_id: str,
        ts: int,
        home_odds: float | None = None,
        away_odds: float | None = None,
        draw_odds: float | None = None,
        home_prob_novig: float | None = None,
        away_prob_novig: float | None = None,
    ) -> None:
        """Insert or update Pinnacle odds for a game at a given time."""
        self.conn.execute(
            """INSERT INTO pinnacle_odds
               (game_id, ts, home_odds, away_odds, draw_odds,
                home_prob_novig, away_prob_novig)
               VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(game_id, ts) DO UPDATE SET
                   home_odds = excluded.home_odds,
                   away_odds = excluded.away_odds,
                   draw_odds = excluded.draw_odds,
                   home_prob_novig = excluded.home_prob_novig,
                   away_prob_novig = excluded.away_prob_novig
            """,
            (game_id, ts, home_odds, away_odds, draw_odds,
             home_prob_novig, away_prob_novig),
        )
        self.conn.commit()

    def upsert_pinnacle_odds_batch(self, odds: list[dict[str, Any]]) -> int:
        """Batch upsert Pinnacle odds."""
        self.conn.executemany(
            """INSERT INTO pinnacle_odds
               (game_id, ts, home_odds, away_odds, draw_odds,
                home_prob_novig, away_prob_novig)
               VALUES (:game_id, :ts, :home_odds, :away_odds, :draw_odds,
                       :home_prob_novig, :away_prob_novig)
               ON CONFLICT(game_id, ts) DO UPDATE SET
                   home_odds = excluded.home_odds,
                   away_odds = excluded.away_odds,
                   draw_odds = excluded.draw_odds,
                   home_prob_novig = excluded.home_prob_novig,
                   away_prob_novig = excluded.away_prob_novig
            """,
            odds,
        )
        self.conn.commit()
        return len(odds)

    def get_pinnacle_odds(
        self, game_id: str
    ) -> list[sqlite3.Row]:
        """Get all Pinnacle odds snapshots for a game."""
        return self.conn.execute(
            "SELECT * FROM pinnacle_odds WHERE game_id = ? ORDER BY ts",
            (game_id,),
        ).fetchall()

    def get_latest_pinnacle_odds(self, game_id: str) -> sqlite3.Row | None:
        """Get the most recent Pinnacle odds for a game (closing line)."""
        return self.conn.execute(
            """SELECT * FROM pinnacle_odds
               WHERE game_id = ?
               ORDER BY ts DESC LIMIT 1""",
            (game_id,),
        ).fetchone()

    # ── Ratings ────────────────────────────────────────────────────────

    def upsert_rating(
        self,
        team: str,
        sport: str,
        date: str,
        elo: float | None = None,
        glicko_rating: float | None = None,
        glicko_rd: float | None = None,
        glicko_vol: float | None = None,
    ) -> None:
        """Insert or update a team's rating on a given date."""
        self.conn.execute(
            """INSERT INTO ratings
               (team, sport, date, elo, glicko_rating, glicko_rd, glicko_vol)
               VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(team, sport, date) DO UPDATE SET
                   elo = COALESCE(excluded.elo, ratings.elo),
                   glicko_rating = COALESCE(excluded.glicko_rating, ratings.glicko_rating),
                   glicko_rd = COALESCE(excluded.glicko_rd, ratings.glicko_rd),
                   glicko_vol = COALESCE(excluded.glicko_vol, ratings.glicko_vol)
            """,
            (team, sport, date, elo, glicko_rating, glicko_rd, glicko_vol),
        )
        self.conn.commit()

    def upsert_ratings_batch(self, ratings: list[dict[str, Any]]) -> int:
        """Batch upsert ratings."""
        self.conn.executemany(
            """INSERT INTO ratings
               (team, sport, date, elo, glicko_rating, glicko_rd, glicko_vol)
               VALUES (:team, :sport, :date, :elo,
                       :glicko_rating, :glicko_rd, :glicko_vol)
               ON CONFLICT(team, sport, date) DO UPDATE SET
                   elo = COALESCE(excluded.elo, ratings.elo),
                   glicko_rating = COALESCE(excluded.glicko_rating, ratings.glicko_rating),
                   glicko_rd = COALESCE(excluded.glicko_rd, ratings.glicko_rd),
                   glicko_vol = COALESCE(excluded.glicko_vol, ratings.glicko_vol)
            """,
            ratings,
        )
        self.conn.commit()
        return len(ratings)

    def get_rating(
        self, team: str, sport: str, date: str
    ) -> sqlite3.Row | None:
        """Get a team's rating on or before a given date."""
        return self.conn.execute(
            """SELECT * FROM ratings
               WHERE team = ? AND sport = ? AND date <= ?
               ORDER BY date DESC LIMIT 1""",
            (team, sport, date),
        ).fetchone()

    def get_all_ratings_on_date(
        self, sport: str, date: str
    ) -> list[sqlite3.Row]:
        """Get latest ratings for all teams on or before a date."""
        return self.conn.execute(
            """SELECT r.* FROM ratings r
               INNER JOIN (
                   SELECT team, MAX(date) as max_date
                   FROM ratings
                   WHERE sport = ? AND date <= ?
                   GROUP BY team
               ) latest ON r.team = latest.team AND r.date = latest.max_date
               WHERE r.sport = ?""",
            (sport, date, sport),
        ).fetchall()

    # ── Utility ────────────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        """Return summary statistics for the database."""
        return {
            "games": self.count_games(),
            "games_nba": self.count_games("nba"),
            "games_epl": self.count_games("epl"),
            "games_nfl": self.count_games("nfl"),
            "markets": self.count_markets(),
            "prices": self.count_prices(),
            "game_events": self.conn.execute(
                "SELECT COUNT(*) FROM game_events"
            ).fetchone()[0],
            "pinnacle_odds": self.conn.execute(
                "SELECT COUNT(*) FROM pinnacle_odds"
            ).fetchone()[0],
            "ratings": self.conn.execute(
                "SELECT COUNT(*) FROM ratings"
            ).fetchone()[0],
        }

    def close(self) -> None:
        """Close the database connection."""
        self.conn.close()

    def __enter__(self) -> SportsDB:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        s = self.stats()
        return (
            f"SportsDB({self.db_path.name}: "
            f"{s['games']} games, {s['markets']} markets, "
            f"{s['prices']} prices)"
        )


# ── Schema SQL ──────────────────────────────────────────────────────────

_SCHEMA_SQL = """
-- Individual sports matches
CREATE TABLE IF NOT EXISTS games (
    game_id     TEXT PRIMARY KEY,     -- e.g., "nba_20260315_LAL_BOS"
    sport       TEXT NOT NULL,        -- "nba", "epl", "nfl", "ufc"
    league      TEXT NOT NULL,        -- "NBA", "Premier League", "NFL"
    home_team   TEXT NOT NULL,        -- team abbreviation or full name
    away_team   TEXT NOT NULL,
    game_date   TEXT NOT NULL,        -- YYYY-MM-DD
    game_time   TEXT,                 -- HH:MM UTC
    home_score  INTEGER,             -- final score (NULL if not yet played)
    away_score  INTEGER,
    status      TEXT DEFAULT 'final', -- "scheduled", "live", "final", "postponed"
    season      TEXT,                 -- e.g., "2025-26"
    venue       TEXT,
    extra_json  TEXT                  -- arbitrary metadata as JSON string
);

-- Polymarket binary contracts linked to games
CREATE TABLE IF NOT EXISTS markets (
    token_id    TEXT PRIMARY KEY,     -- YES token ID from Polymarket
    token_id_no TEXT,                 -- NO token ID
    game_id     TEXT NOT NULL,        -- FK to games
    event_id    TEXT,                 -- Polymarket event ID
    condition_id TEXT,                -- Polymarket condition ID
    question    TEXT,                 -- Market question text
    outcome     TEXT,                 -- "home_win", "away_win", "draw", "over", etc.
    volume      REAL,                -- Total trading volume
    neg_risk    INTEGER DEFAULT 0,   -- 1 if NegRisk market
    won         INTEGER,             -- 1 if resolved YES, 0 if NO, NULL if unresolved
    end_date    TEXT,                -- Market end date (ISO)
    extra_json  TEXT,                -- arbitrary metadata as JSON string
    FOREIGN KEY (game_id) REFERENCES games(game_id)
);

-- Minute-level price history during games
-- Primary data source for green book + overreaction backtesting
CREATE TABLE IF NOT EXISTS prices (
    token_id    TEXT NOT NULL,        -- FK to markets
    ts          INTEGER NOT NULL,     -- Unix timestamp
    price       REAL NOT NULL,        -- YES price [0.0, 1.0]
    bid         REAL,                -- Best bid (if available)
    ask         REAL,                -- Best ask (if available)
    volume_1m   REAL,                -- 1-minute volume (if available)
    PRIMARY KEY (token_id, ts)
);

-- Live game events (scores, cards, etc.)
-- Used for D2 overreaction detection and Bayesian updating
CREATE TABLE IF NOT EXISTS game_events (
    game_id     TEXT NOT NULL,        -- FK to games
    ts          INTEGER NOT NULL,     -- Unix timestamp of event
    event_type  TEXT NOT NULL,        -- "goal", "basket", "touchdown", "red_card",
                                     -- "quarter_end", "half_end", "game_start",
                                     -- "game_end", "injury", "substitution"
    team        TEXT,                 -- Which team (abbreviation)
    score_home  INTEGER,             -- Score after this event
    score_away  INTEGER,
    detail      TEXT,                 -- Player name, event specifics
    FOREIGN KEY (game_id) REFERENCES games(game_id)
);

-- Pinnacle odds history per game
-- Used as sharp benchmark for probability model
CREATE TABLE IF NOT EXISTS pinnacle_odds (
    game_id     TEXT NOT NULL,        -- FK to games
    ts          INTEGER NOT NULL,     -- Unix timestamp of odds snapshot
    home_odds   REAL,                -- Decimal odds for home win
    away_odds   REAL,                -- Decimal odds for away win
    draw_odds   REAL,                -- Decimal odds for draw (soccer only)
    home_prob_novig REAL,            -- Vig-removed probability
    away_prob_novig REAL,
    PRIMARY KEY (game_id, ts),
    FOREIGN KEY (game_id) REFERENCES games(game_id)
);

-- Elo/Glicko-2 team strength ratings over time
-- Computed from game results, used as 40% of probability model
CREATE TABLE IF NOT EXISTS ratings (
    team        TEXT NOT NULL,        -- Team abbreviation
    sport       TEXT NOT NULL,        -- "nba", "epl", "nfl"
    date        TEXT NOT NULL,        -- YYYY-MM-DD (date rating was computed)
    elo         REAL,                -- Elo rating (default 1500)
    glicko_rating REAL,              -- Glicko-2 rating
    glicko_rd   REAL,                -- Glicko-2 rating deviation
    glicko_vol  REAL,                -- Glicko-2 volatility
    PRIMARY KEY (team, sport, date)
);

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_games_sport_date ON games(sport, game_date);
CREATE INDEX IF NOT EXISTS idx_games_status ON games(status);
CREATE INDEX IF NOT EXISTS idx_games_teams ON games(home_team, away_team);
CREATE INDEX IF NOT EXISTS idx_markets_game ON markets(game_id);
CREATE INDEX IF NOT EXISTS idx_markets_outcome ON markets(outcome);
CREATE INDEX IF NOT EXISTS idx_prices_token ON prices(token_id);
CREATE INDEX IF NOT EXISTS idx_prices_ts ON prices(ts);
CREATE INDEX IF NOT EXISTS idx_game_events_game ON game_events(game_id);
CREATE INDEX IF NOT EXISTS idx_game_events_type ON game_events(event_type);
CREATE INDEX IF NOT EXISTS idx_pinnacle_game ON pinnacle_odds(game_id);
CREATE INDEX IF NOT EXISTS idx_ratings_team ON ratings(team, sport);
CREATE INDEX IF NOT EXISTS idx_ratings_date ON ratings(sport, date);
"""
