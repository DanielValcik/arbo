"""Strategy D — PMD Data Enrichment Pipeline.

Fills in missing data in the PolymarketData.co sports database:
  1. resolve   — Backfill markets.won from PMD API (re-query winner field)
  2. teams     — Parse team names from question text, update games table
  3. classify  — Classify unknown sports from question/slug patterns
  4. indexes   — Add optimized indexes for backtest queries
  5. stats     — Print data quality report

Designed to run on VPS where the 312GB SQLite DB lives.
Resume-safe: tracks progress to avoid re-processing.

Usage:
    # On VPS:
    cd /opt/arbo
    PYTHONPATH=. python3 research_d/enrich_pmd_data.py --step resolve
    PYTHONPATH=. python3 research_d/enrich_pmd_data.py --step teams
    PYTHONPATH=. python3 research_d/enrich_pmd_data.py --step classify
    PYTHONPATH=. python3 research_d/enrich_pmd_data.py --step indexes
    PYTHONPATH=. python3 research_d/enrich_pmd_data.py --step stats
    PYTHONPATH=. python3 research_d/enrich_pmd_data.py --step all
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import ssl
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

try:
    import certifi
    SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    SSL_CTX = ssl.create_default_context()
    SSL_CTX.check_hostname = False
    SSL_CTX.verify_mode = ssl.CERT_NONE

DATA_DIR = Path(__file__).parent / "data"
DEFAULT_DB_PATH = DATA_DIR / "sports_backtest.sqlite"
PROGRESS_FILE = DATA_DIR / "enrich_progress.json"

# PMD API
PMD_API_BASE = "https://api.polymarketdata.co/v1"

# Gamma API (free, no auth)
GAMMA_API_BASE = "https://gamma-api.polymarket.com"


def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ── HTTP helpers ──────────────────────────────────────────────────────


def _http_get(url: str, headers: dict | None = None, timeout: int = 30) -> dict | list | None:
    """GET request with retry. Returns parsed JSON or None."""
    hdrs = {"User-Agent": "ArboEnrich/1.0", "Accept": "application/json"}
    if headers:
        hdrs.update(headers)

    for attempt in range(3):
        req = urllib.request.Request(url, headers=hdrs)
        try:
            with urllib.request.urlopen(req, timeout=timeout, context=SSL_CTX) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = 2 ** (attempt + 1)
                log(f"  Rate limited, waiting {wait}s...")
                time.sleep(wait)
            elif e.code in (403, 404):
                return None
            else:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                else:
                    return None
        except Exception:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                return None
    return None


def pmd_get(path: str, api_key: str) -> dict | list | None:
    """PMD API request with auth."""
    return _http_get(
        f"{PMD_API_BASE}{path}",
        headers={"X-API-Key": api_key},
    )


def gamma_get(path: str) -> dict | list | None:
    """Gamma API request (no auth)."""
    return _http_get(f"{GAMMA_API_BASE}{path}")


# ── Progress tracking ─────────────────────────────────────────────────


def load_progress() -> dict:
    if PROGRESS_FILE.exists():
        return json.loads(PROGRESS_FILE.read_text())
    return {}


def save_progress(data: dict) -> None:
    PROGRESS_FILE.write_text(json.dumps(data, indent=2))


# ── Step 1: Resolution backfill ───────────────────────────────────────


def step_resolve(db_path: Path) -> None:
    """Backfill markets.won using PMD API and Gamma API fallback.

    Strategy:
      1. Re-query PMD API for each event_id — get token winner field
      2. For remaining: use Gamma API condition_id lookup
      3. For remaining: infer from final price in trajectory
    """
    log("=== STEP: RESOLVE (backfill markets.won) ===")

    api_key = os.environ.get("POLYMARKETDATA_API_KEY", "")
    if not api_key:
        log("WARNING: No POLYMARKETDATA_API_KEY — skipping PMD resolution, using Gamma only")

    conn = sqlite3.connect(str(db_path), timeout=300)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA cache_size=-64000")
    conn.execute("PRAGMA busy_timeout=300000")  # 5 min wait for lock

    # Count unresolved
    total_unresolved = conn.execute(
        "SELECT COUNT(*) FROM markets WHERE won IS NULL"
    ).fetchone()[0]
    total_resolved = conn.execute(
        "SELECT COUNT(*) FROM markets WHERE won IS NOT NULL"
    ).fetchone()[0]
    log(f"Markets: {total_resolved} resolved, {total_unresolved} unresolved")

    if total_unresolved == 0:
        log("Nothing to resolve!")
        conn.close()
        return

    progress = load_progress()
    resolved_pmd_ids = set(progress.get("resolved_pmd_ids", []))

    # --- Phase 1: PMD API re-query ---
    if api_key:
        log("Phase 1: PMD API re-query for winner field...")

        # Get unique event_ids (= PMD market IDs) with unresolved markets
        rows = conn.execute("""
            SELECT DISTINCT event_id FROM markets
            WHERE won IS NULL AND event_id IS NOT NULL AND event_id != ''
        """).fetchall()
        pmd_ids = [r[0] for r in rows if r[0] not in resolved_pmd_ids]
        log(f"  {len(pmd_ids)} PMD market IDs to query ({len(resolved_pmd_ids)} already done)")

        updated = 0
        batch_updates: list[tuple[int, str]] = []

        for i, pmd_id in enumerate(pmd_ids):
            if i > 0 and i % 100 == 0:
                # Flush batch
                if batch_updates:
                    conn.executemany(
                        "UPDATE markets SET won = ? WHERE token_id = ? AND won IS NULL",
                        batch_updates,
                    )
                    conn.commit()
                    updated += len(batch_updates)
                    batch_updates = []

                # Save progress
                resolved_pmd_ids.add(pmd_id)
                progress["resolved_pmd_ids"] = list(resolved_pmd_ids)
                save_progress(progress)
                log(f"  [{i}/{len(pmd_ids)}] updated={updated}")

            # Rate limit: 2000 RPM = ~33/s
            time.sleep(0.035)

            resp = pmd_get(f"/markets/{pmd_id}", api_key)
            if not resp:
                resolved_pmd_ids.add(pmd_id)
                continue

            # PMD returns resolved_token_id at market level (not winner on token)
            status = resp.get("status", "")
            resolved_token_id = str(resp.get("resolved_token_id") or "")
            tokens = resp.get("tokens", [])

            if status in ("closed", "resolved") and resolved_token_id:
                for token in tokens:
                    token_id = str(token.get("id", token.get("token_id", "")))
                    if not token_id:
                        continue
                    if token_id == resolved_token_id:
                        batch_updates.append((1, token_id))
                    else:
                        batch_updates.append((0, token_id))

            resolved_pmd_ids.add(pmd_id)

        # Flush remaining
        if batch_updates:
            conn.executemany(
                "UPDATE markets SET won = ? WHERE token_id = ? AND won IS NULL",
                batch_updates,
            )
            conn.commit()
            updated += len(batch_updates)

        progress["resolved_pmd_ids"] = list(resolved_pmd_ids)
        save_progress(progress)
        log(f"  Phase 1 done: {updated} markets resolved via PMD API")

    # --- Phase 2: Gamma API fallback ---
    remaining = conn.execute(
        "SELECT COUNT(*) FROM markets WHERE won IS NULL"
    ).fetchone()[0]
    log(f"Phase 2: Gamma API fallback ({remaining} still unresolved)...")

    if remaining > 0:
        resolved_conditions = set(progress.get("resolved_conditions", []))

        rows = conn.execute("""
            SELECT DISTINCT condition_id FROM markets
            WHERE won IS NULL AND condition_id IS NOT NULL AND condition_id != ''
        """).fetchall()
        condition_ids = [r[0] for r in rows if r[0] not in resolved_conditions]
        log(f"  {len(condition_ids)} condition IDs to query Gamma API")

        updated = 0
        batch_updates = []

        for i, cid in enumerate(condition_ids):
            if i > 0 and i % 100 == 0:
                if batch_updates:
                    conn.executemany(
                        "UPDATE markets SET won = ? WHERE token_id = ? AND won IS NULL",
                        batch_updates,
                    )
                    conn.commit()
                    updated += len(batch_updates)
                    batch_updates = []

                resolved_conditions.add(cid)
                progress["resolved_conditions"] = list(resolved_conditions)
                save_progress(progress)
                log(f"  [{i}/{len(condition_ids)}] updated={updated}")

            # Rate limit: 500/10s = 50/s
            time.sleep(0.025)

            resp = gamma_get(f"/markets?condition_id={cid}")
            if not resp:
                resolved_conditions.add(cid)
                continue

            # Gamma returns a list of markets
            markets_list = resp if isinstance(resp, list) else [resp]
            for mkt in markets_list:
                outcomes = mkt.get("outcomes", [])
                outcome_prices = mkt.get("outcomePrices", [])
                tokens = mkt.get("tokens", mkt.get("clobTokenIds", []))

                if not outcome_prices or len(outcome_prices) < 2:
                    continue

                # Check if resolved: prices are "0"/"1" exactly
                try:
                    prices = [float(p) for p in outcome_prices]
                except (ValueError, TypeError):
                    continue

                if not (
                    (prices[0] >= 0.99 and prices[1] <= 0.01)
                    or (prices[0] <= 0.01 and prices[1] >= 0.99)
                ):
                    # Not resolved yet
                    continue

                # Find which tokens in our DB match
                yes_won = 1 if prices[0] >= 0.99 else 0

                # Get token_ids for this condition from our DB
                db_tokens = conn.execute(
                    "SELECT token_id FROM markets WHERE condition_id = ? AND won IS NULL",
                    (cid,),
                ).fetchall()

                for row in db_tokens:
                    # All tokens with this condition_id in NegRisk have separate outcomes
                    # For simple binary: first token = YES
                    batch_updates.append((yes_won, row[0]))

            resolved_conditions.add(cid)

        if batch_updates:
            conn.executemany(
                "UPDATE markets SET won = ? WHERE token_id = ? AND won IS NULL",
                batch_updates,
            )
            conn.commit()
            updated += len(batch_updates)

        progress["resolved_conditions"] = list(resolved_conditions)
        save_progress(progress)
        log(f"  Phase 2 done: {updated} markets resolved via Gamma API")

    # --- Phase 3: Infer from final prices ---
    remaining = conn.execute(
        "SELECT COUNT(*) FROM markets WHERE won IS NULL"
    ).fetchone()[0]
    log(f"Phase 3: Infer from final prices ({remaining} still unresolved)...")

    if remaining > 0:
        # For each unresolved market, check if last price is near 0 or 1
        updated = conn.execute("""
            UPDATE markets SET won = (
                SELECT CASE
                    WHEN p.last_price >= 0.95 THEN 1
                    WHEN p.last_price <= 0.05 THEN 0
                    ELSE NULL
                END
                FROM (
                    SELECT token_id, price as last_price
                    FROM prices p2
                    WHERE p2.token_id = markets.token_id
                    ORDER BY p2.ts DESC
                    LIMIT 1
                ) p
            )
            WHERE won IS NULL
            AND EXISTS (SELECT 1 FROM prices WHERE prices.token_id = markets.token_id)
        """).rowcount
        conn.commit()
        log(f"  Phase 3 done: {updated} markets resolved from final prices")

    # Final stats
    final_resolved = conn.execute(
        "SELECT COUNT(*) FROM markets WHERE won IS NOT NULL"
    ).fetchone()[0]
    final_unresolved = conn.execute(
        "SELECT COUNT(*) FROM markets WHERE won IS NULL"
    ).fetchone()[0]
    log(f"Resolution complete: {final_resolved} resolved, {final_unresolved} still unresolved")

    conn.close()


# ── Step 2: Team name parsing ─────────────────────────────────────────

# Import team maps from existing module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    from research_d.download_sports_prices import (
        team_to_abbreviation,
        NBA_TEAMS,
        EPL_TEAMS,
        NFL_TEAMS,
    )
except ImportError:
    log("WARNING: Could not import team maps, using built-in")
    NBA_TEAMS = {}
    EPL_TEAMS = {}
    NFL_TEAMS = {}

    def team_to_abbreviation(name: str, sport: str) -> str:
        cleaned = re.sub(r"[^a-zA-Z\s]", "", name).strip()
        words = cleaned.split()
        if len(words) >= 2:
            return "".join(w[0].upper() for w in words[:3])
        return cleaned[:3].upper() or "UNK"


# Regex patterns for extracting team names from Polymarket questions
_TEAM_PATTERNS = [
    # "Will the Golden State Warriors beat the Cleveland Cavaliers?"
    re.compile(
        r"[Ww]ill (?:the )?(.+?) (?:beat|defeat|win against|win over) (?:the )?(.+?)\??$"
    ),
    # "Golden State Warriors vs Cleveland Cavaliers"
    re.compile(r"^(.+?)\s+vs?\.?\s+(.+?)(?:\s*\?)?$"),
    # "Lakers vs. Celtics: Who wins?"
    re.compile(r"^(.+?)\s+vs?\.?\s+(.+?)(?::\s*.+)?$"),
    # "Will X win against Y on DATE?"
    re.compile(
        r"[Ww]ill (?:the )?(.+?) win (?:against|vs\.?|over) (?:the )?(.+?)(?:\s+on\s+.+)?\??$"
    ),
]

# Patterns for moneyline questions (single team)
_MONEYLINE_PATTERNS = [
    # "Will the Lakers win?" or "Lakers to win?"
    re.compile(r"[Ww]ill (?:the )?(.+?) win\b"),
    # "Lakers win the NBA Cup?"
    re.compile(r"^(.+?) (?:wins?|to win)\b"),
]


def _extract_teams_from_question(question: str) -> tuple[str, str] | None:
    """Extract two team names from a Polymarket market question.

    Returns (team_a, team_b) or None if cannot parse.
    team_a is typically the team whose YES outcome is being asked about.
    """
    if not question:
        return None

    # Clean up question
    q = question.strip().rstrip("?").strip()

    for pattern in _TEAM_PATTERNS:
        m = pattern.match(q)
        if m:
            return m.group(1).strip(), m.group(2).strip()

    return None


def _extract_single_team(question: str) -> str | None:
    """Extract a single team name from moneyline-style question."""
    if not question:
        return None

    for pattern in _MONEYLINE_PATTERNS:
        m = pattern.search(question)
        if m:
            return m.group(1).strip()

    return None


def _detect_sport_from_question(question: str, slug: str = "") -> str | None:
    """Detect sport from question text or slug."""
    text = f"{question} {slug}".lower()

    # NBA indicators
    nba_teams = {"lakers", "celtics", "warriors", "knicks", "nets", "76ers",
                 "heat", "bulls", "cavaliers", "thunder", "nuggets", "bucks",
                 "suns", "kings", "clippers", "mavericks", "rockets", "grizzlies",
                 "spurs", "pelicans", "pacers", "hawks", "hornets", "magic",
                 "wizards", "pistons", "raptors", "timberwolves", "blazers", "jazz"}
    if any(t in text for t in nba_teams) or "nba" in text:
        return "nba"

    # EPL indicators
    epl_teams = {"arsenal", "chelsea", "liverpool", "manchester city", "man city",
                 "manchester united", "man united", "tottenham", "west ham",
                 "newcastle", "aston villa", "brighton", "brentford", "fulham",
                 "crystal palace", "everton", "nottingham forest", "bournemouth",
                 "wolverhampton", "wolves", "southampton", "leicester", "ipswich"}
    if any(t in text for t in epl_teams) or "premier league" in text or "epl" in text:
        return "epl"

    # NFL indicators
    nfl_teams = {"patriots", "eagles", "chiefs", "cowboys", "49ers", "seahawks",
                 "ravens", "steelers", "bills", "bengals", "dolphins", "chargers",
                 "rams", "cardinals", "commanders", "jaguars", "texans", "titans",
                 "colts", "broncos", "raiders", "saints", "falcons", "panthers",
                 "bears", "lions", "packers", "vikings", "buccaneers", "giants", "jets"}
    if any(t in text for t in nfl_teams) or "nfl" in text:
        return "nfl"

    # UFC/MMA
    if "ufc" in text or "mma" in text or "fight" in text:
        return "ufc"

    # NHL
    if "nhl" in text or "hockey" in text:
        return "nhl"

    # F1
    if "formula" in text or "f1" in text or "grand prix" in text:
        return "f1"

    # UCL
    if "champions league" in text or "ucl" in text:
        return "ucl"

    # Soccer (generic)
    if "la liga" in text or "serie a" in text or "bundesliga" in text:
        return "soccer"

    # MLB
    if "mlb" in text or "baseball" in text:
        return "mlb"

    # NCAAB
    if "ncaab" in text or "march madness" in text or "college basketball" in text:
        return "ncaab"

    return None


def step_teams(db_path: Path) -> None:
    """Parse team names from question text and update games table."""
    log("=== STEP: TEAMS (parse team names from questions) ===")

    conn = sqlite3.connect(str(db_path), timeout=300)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA cache_size=-64000")
    conn.execute("PRAGMA busy_timeout=300000")

    # Find games with HOME/AWAY placeholder team names
    rows = conn.execute("""
        SELECT g.game_id, g.sport, g.home_team, g.away_team, g.extra_json,
               m.question, m.outcome
        FROM games g
        JOIN markets m ON m.game_id = g.game_id
        WHERE (g.home_team = 'HOME' OR g.away_team = 'AWAY'
               OR g.home_team = 'UNK' OR g.away_team = 'UNK')
        AND g.sport != 'unknown'
        GROUP BY g.game_id
    """).fetchall()

    log(f"Games with placeholder teams: {len(rows)}")

    updated = 0
    for game_id, sport, home_team, away_team, extra_json, question, outcome in rows:
        # Try extracting from question
        teams = _extract_teams_from_question(question)
        if teams:
            team_a_name, team_b_name = teams
            # In Polymarket, team_a is typically the subject (YES outcome team)
            team_a = team_to_abbreviation(team_a_name, sport)
            team_b = team_to_abbreviation(team_b_name, sport)

            conn.execute(
                "UPDATE games SET home_team = ?, away_team = ? WHERE game_id = ?",
                (team_a, team_b, game_id),
            )
            updated += 1
        else:
            # Try single team from moneyline question
            team_name = _extract_single_team(question)
            if team_name:
                team_abbr = team_to_abbreviation(team_name, sport)
                # Update only the placeholder side
                if home_team in ("HOME", "UNK"):
                    conn.execute(
                        "UPDATE games SET home_team = ? WHERE game_id = ?",
                        (team_abbr, game_id),
                    )
                    updated += 1

    conn.commit()
    log(f"Updated {updated} games with parsed team names")

    # Also extract team names from slug in extra_json
    # PMD slugs like: "nba-den-mem-2026-01-25-spread-home-12pt5"
    slug_rows = conn.execute("""
        SELECT g.game_id, g.sport, g.extra_json
        FROM games g
        WHERE (g.home_team = 'HOME' OR g.away_team = 'AWAY'
               OR g.home_team = 'UNK' OR g.away_team = 'UNK')
        AND g.extra_json LIKE '%slug%'
    """).fetchall()

    slug_updated = 0
    slug_pattern = re.compile(
        r"(?:nba|nfl|nhl|epl|ucl|mlb|ufc|ncaab)-([a-z]+)-([a-z]+)-\d{4}-\d{2}-\d{2}"
    )

    for game_id, sport, extra_json in slug_rows:
        if not extra_json:
            continue
        try:
            ej = json.loads(extra_json)
        except (json.JSONDecodeError, TypeError):
            continue

        slug = ej.get("slug", "")
        m = slug_pattern.search(slug)
        if m:
            team_a = m.group(1).upper()
            team_b = m.group(2).upper()
            # Try abbreviation lookup
            team_a = team_to_abbreviation(team_a, sport) if len(team_a) > 3 else team_a
            team_b = team_to_abbreviation(team_b, sport) if len(team_b) > 3 else team_b

            conn.execute(
                "UPDATE games SET home_team = ?, away_team = ? WHERE game_id = ?",
                (team_a, team_b, game_id),
            )
            slug_updated += 1

    conn.commit()
    log(f"Updated {slug_updated} games from slug patterns")
    log(f"Total team updates: {updated + slug_updated}")

    # Report remaining
    remaining = conn.execute("""
        SELECT COUNT(*) FROM games
        WHERE (home_team = 'HOME' OR away_team = 'AWAY'
               OR home_team = 'UNK' OR away_team = 'UNK')
        AND sport != 'unknown'
    """).fetchone()[0]
    log(f"Games still with placeholder teams: {remaining}")

    conn.close()


# ── Step 3: Sport classification ──────────────────────────────────────


def step_classify(db_path: Path) -> None:
    """Classify 'unknown' sport games using question text and slug patterns."""
    log("=== STEP: CLASSIFY (detect sport for 'unknown' games) ===")

    conn = sqlite3.connect(str(db_path), timeout=300)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA cache_size=-64000")
    conn.execute("PRAGMA busy_timeout=300000")

    total_unknown = conn.execute(
        "SELECT COUNT(*) FROM games WHERE sport = 'unknown'"
    ).fetchone()[0]
    log(f"Unknown sport games: {total_unknown}")

    if total_unknown == 0:
        log("Nothing to classify!")
        conn.close()
        return

    # Get unknown games with their market questions
    rows = conn.execute("""
        SELECT g.game_id, g.extra_json, m.question
        FROM games g
        LEFT JOIN markets m ON m.game_id = g.game_id
        WHERE g.sport = 'unknown'
        GROUP BY g.game_id
    """).fetchall()

    classified = 0
    non_sports = 0
    sport_counts: dict[str, int] = {}

    batch: list[tuple[str, str, str]] = []  # (sport, league, game_id)

    for game_id, extra_json, question in rows:
        slug = ""
        if extra_json:
            try:
                ej = json.loads(extra_json)
                slug = ej.get("slug", "")
            except (json.JSONDecodeError, TypeError):
                pass

        sport = _detect_sport_from_question(question or "", slug)
        if sport:
            league = sport.upper()
            batch.append((sport, league, game_id))
            sport_counts[sport] = sport_counts.get(sport, 0) + 1
            classified += 1
        else:
            # Mark as non-sports (politics, crypto, entertainment, etc.)
            non_sports += 1

    # Batch update
    if batch:
        conn.executemany(
            "UPDATE games SET sport = ?, league = ? WHERE game_id = ?",
            batch,
        )
        conn.commit()

    log(f"Classified: {classified} games")
    for sport, count in sorted(sport_counts.items(), key=lambda x: -x[1]):
        log(f"  {sport}: {count}")
    log(f"Non-sports (kept as 'unknown'): {non_sports}")

    conn.close()


# ── Step 4: Index optimization ────────────────────────────────────────


def step_indexes(db_path: Path) -> None:
    """Add optimized indexes for backtest queries."""
    log("=== STEP: INDEXES (optimize for backtest) ===")

    conn = sqlite3.connect(str(db_path))

    indexes = [
        # Resolution queries
        "CREATE INDEX IF NOT EXISTS idx_markets_won ON markets(won)",
        # Market-game join with sport filter
        "CREATE INDEX IF NOT EXISTS idx_markets_game_won ON markets(game_id, won)",
        # NOTE: idx_prices_token already exists (from schema), compound (token_id, ts)
        # would take hours on 958M rows and isn't needed — single-column index + in-memory
        # sort is fast enough for per-market queries (1K-50K rows).
        # Sport + date for game selection
        "CREATE INDEX IF NOT EXISTS idx_games_sport_date_status ON games(sport, game_date, status)",
        # Markets by event_id (for PMD re-query)
        "CREATE INDEX IF NOT EXISTS idx_markets_event_id ON markets(event_id)",
        # Markets by condition_id (for Gamma API)
        "CREATE INDEX IF NOT EXISTS idx_markets_condition_id ON markets(condition_id)",
    ]

    for idx_sql in indexes:
        idx_name = idx_sql.split("IF NOT EXISTS ")[1].split(" ON")[0]
        log(f"  Creating {idx_name}...")
        conn.execute(idx_sql)

    conn.commit()
    log("  Running ANALYZE for query planner...")
    conn.execute("ANALYZE")
    conn.commit()
    log("Indexes done")
    conn.close()


# ── Step 5: Stats report ─────────────────────────────────────────────


def step_stats(db_path: Path) -> None:
    """Print data quality report."""
    log("=== DATA QUALITY REPORT ===")

    conn = sqlite3.connect(str(db_path))

    # Games by sport
    log("\nGames by sport:")
    rows = conn.execute(
        "SELECT sport, COUNT(*) FROM games GROUP BY sport ORDER BY COUNT(*) DESC"
    ).fetchall()
    for sport, count in rows:
        log(f"  {sport}: {count:,}")

    # Markets resolution status
    log("\nMarket resolution:")
    for label, query in [
        ("won=1 (YES)", "SELECT COUNT(*) FROM markets WHERE won = 1"),
        ("won=0 (NO)", "SELECT COUNT(*) FROM markets WHERE won = 0"),
        ("unresolved", "SELECT COUNT(*) FROM markets WHERE won IS NULL"),
    ]:
        count = conn.execute(query).fetchone()[0]
        log(f"  {label}: {count:,}")

    # Markets with prices
    log("\nPrice data:")
    total_prices = conn.execute("SELECT COUNT(*) FROM prices").fetchone()[0]
    tokens_with_prices = conn.execute(
        "SELECT COUNT(DISTINCT token_id) FROM prices"
    ).fetchone()[0]
    log(f"  Total price points: {total_prices:,}")
    log(f"  Tokens with prices: {tokens_with_prices:,}")

    # Sports markets with prices + resolution (backtest-ready)
    log("\nBacktest-ready markets (resolved + has prices + known sport):")
    rows = conn.execute("""
        SELECT g.sport, COUNT(DISTINCT m.token_id) as n_markets,
               SUM(CASE WHEN m.won = 1 THEN 1 ELSE 0 END) as won,
               SUM(CASE WHEN m.won = 0 THEN 1 ELSE 0 END) as lost
        FROM markets m
        JOIN games g ON m.game_id = g.game_id
        WHERE m.won IS NOT NULL
        AND g.sport != 'unknown'
        AND EXISTS (SELECT 1 FROM prices WHERE prices.token_id = m.token_id)
        GROUP BY g.sport
        ORDER BY n_markets DESC
    """).fetchall()
    total_bt = 0
    for sport, n, won, lost in rows:
        log(f"  {sport}: {n:,} markets (won={won:,}, lost={lost:,})")
        total_bt += n
    log(f"  TOTAL: {total_bt:,} backtest-ready markets")

    # Team name quality
    log("\nTeam name quality:")
    placeholder_games = conn.execute("""
        SELECT COUNT(*) FROM games
        WHERE (home_team = 'HOME' OR away_team = 'AWAY'
               OR home_team = 'UNK' OR away_team = 'UNK')
        AND sport != 'unknown'
    """).fetchone()[0]
    total_sport_games = conn.execute(
        "SELECT COUNT(*) FROM games WHERE sport != 'unknown'"
    ).fetchone()[0]
    log(f"  Sports games with real teams: {total_sport_games - placeholder_games:,} / {total_sport_games:,}")
    log(f"  Still placeholder: {placeholder_games:,}")

    # Pinnacle odds coverage
    pinnacle_count = conn.execute("SELECT COUNT(*) FROM pinnacle_odds").fetchone()[0]
    log(f"\nPinnacle odds: {pinnacle_count:,}")

    # Elo/Glicko ratings
    ratings_count = conn.execute("SELECT COUNT(*) FROM ratings").fetchone()[0]
    log(f"Elo/Glicko ratings: {ratings_count:,}")

    # Game events
    events_count = conn.execute("SELECT COUNT(*) FROM game_events").fetchone()[0]
    log(f"Game events: {events_count:,}")

    conn.close()


# ── Main ──────────────────────────────────────────────────────────────


STEPS = {
    "resolve": step_resolve,
    "teams": step_teams,
    "classify": step_classify,
    "indexes": step_indexes,
    "stats": step_stats,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="PMD Data Enrichment Pipeline")
    parser.add_argument(
        "--step",
        choices=list(STEPS.keys()) + ["all"],
        required=True,
        help="Enrichment step to run",
    )
    parser.add_argument(
        "--db",
        type=str,
        default=str(DEFAULT_DB_PATH),
        help=f"Path to SQLite database (default: {DEFAULT_DB_PATH})",
    )
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        log(f"ERROR: Database not found: {db_path}")
        sys.exit(1)

    log(f"Database: {db_path} ({db_path.stat().st_size / 1e9:.1f} GB)")

    if args.step == "all":
        for step_name in ["indexes", "classify", "teams", "resolve", "stats"]:
            STEPS[step_name](db_path)
            log("")
    else:
        STEPS[args.step](db_path)


if __name__ == "__main__":
    main()
