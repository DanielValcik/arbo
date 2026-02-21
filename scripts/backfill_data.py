"""Backfill historical match data for XGBoost value model training.

Downloads football-data.org CSVs for major leagues and fetches
historical Pinnacle odds from The Odds API. Produces a JSON dataset
used by process_data.py and run_backtest.py.

Usage:
    python3 scripts/backfill_data.py [--output path] [--seasons 2425,2324]

See Sprint 4 Phase A specification.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import ssl
import sys
from datetime import UTC, datetime, timedelta
from difflib import SequenceMatcher
from pathlib import Path

import aiohttp
import certifi
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arbo.utils.logger import get_logger

logger = get_logger("backfill")


# ================================================================
# DTOs
# ================================================================


class HistoricalOdds(BaseModel):
    """Odds snapshot at a point in time."""

    bookmaker: str = ""
    home_win: float | None = None
    draw: float | None = None
    away_win: float | None = None
    over_25: float | None = None
    under_25: float | None = None


class HistoricalMatch(BaseModel):
    """A single historical match with results and odds."""

    match_id: str
    date: str  # ISO-8601
    league: str
    season: str
    home_team: str
    away_team: str
    home_goals: int
    away_goals: int
    result: str  # "H", "D", "A"
    # Football-data.org odds (Pinnacle columns if available)
    fd_home_odds: float | None = None
    fd_draw_odds: float | None = None
    fd_away_odds: float | None = None
    fd_over_25: float | None = None
    fd_under_25: float | None = None
    # The Odds API historical snapshots
    opening_odds: HistoricalOdds | None = None
    closing_odds: HistoricalOdds | None = None


class BacktestDataset(BaseModel):
    """Full backtest dataset with metadata."""

    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    n_matches: int = 0
    leagues: list[str] = Field(default_factory=list)
    seasons: list[str] = Field(default_factory=list)
    matches: list[HistoricalMatch] = Field(default_factory=list)


# ================================================================
# Football-data.org league mapping
# ================================================================

# Maps (league_code, season) to football-data.org CSV URL
# season format: "2425" means 2024-2025
FOOTBALL_DATA_LEAGUES: dict[str, dict[str, str]] = {
    "E0": {
        "name": "Premier League",
        "odds_api_key": "soccer_epl",
    },
    "SP1": {
        "name": "La Liga",
        "odds_api_key": "soccer_spain_la_liga",
    },
    "D1": {
        "name": "Bundesliga",
        "odds_api_key": "soccer_germany_bundesliga",
    },
    "I1": {
        "name": "Serie A",
        "odds_api_key": "soccer_italy_serie_a",
    },
    "F1": {
        "name": "Ligue 1",
        "odds_api_key": "soccer_france_ligue_one",
    },
}

# Champions League uses a different source (football-data.org doesn't have CSVs)
CHAMPIONS_LEAGUE_KEY = "soccer_uefa_champs_league"

# Default seasons to fetch
DEFAULT_SEASONS = ["2425", "2324"]


def _football_data_url(league_code: str, season: str) -> str:
    """Build football-data.org CSV URL for a league and season."""
    return f"https://www.football-data.co.uk/mmz4281/{season}/{league_code}.csv"


# ================================================================
# Team name fuzzy matching
# ================================================================

# Common name mappings between football-data.org and The Odds API
TEAM_NAME_ALIASES: dict[str, list[str]] = {
    "Man United": ["Manchester United", "Man Utd"],
    "Man City": ["Manchester City"],
    "Tottenham": ["Tottenham Hotspur", "Spurs"],
    "Newcastle": ["Newcastle United"],
    "Wolves": ["Wolverhampton Wanderers", "Wolverhampton"],
    "West Ham": ["West Ham United"],
    "Nott'm Forest": ["Nottingham Forest", "Nottingham"],
    "Sheffield United": ["Sheffield Utd"],
    "Ath Madrid": ["Atletico Madrid", "Atletico de Madrid", "Atlético Madrid"],
    "Ath Bilbao": ["Athletic Bilbao", "Athletic Club"],
    "Betis": ["Real Betis"],
    "Sociedad": ["Real Sociedad"],
    "Celta": ["Celta Vigo", "Celta de Vigo"],
    "Espanol": ["Espanyol", "RCD Espanyol"],
    "Vallecano": ["Rayo Vallecano"],
    "Leverkusen": ["Bayer Leverkusen", "Bayer 04 Leverkusen"],
    "M'gladbach": ["Borussia Monchengladbach", "Monchengladbach", "Mönchengladbach"],
    "Dortmund": ["Borussia Dortmund"],
    "Bayern Munich": ["FC Bayern Munich", "Bayern München"],
    "Ein Frankfurt": ["Eintracht Frankfurt"],
    "FC Koln": ["FC Cologne", "1. FC Köln", "Köln"],
    "Hertha": ["Hertha Berlin", "Hertha BSC"],
    "St Pauli": ["FC St. Pauli"],
    "Milan": ["AC Milan"],
    "Inter": ["Inter Milan", "Internazionale"],
    "Verona": ["Hellas Verona"],
    "Parma": ["Parma Calcio"],
    "Paris SG": ["Paris Saint-Germain", "Paris Saint Germain", "PSG"],
    "St Etienne": ["Saint-Etienne", "AS Saint-Étienne"],
    "Marseille": ["Olympique Marseille", "Olympique de Marseille"],
    "Lyon": ["Olympique Lyonnais", "Olympique Lyon"],
}


def _normalize_name(name: str) -> str:
    """Normalize a team name for comparison."""
    return name.strip().lower().replace("fc ", "").replace(" fc", "")


def fuzzy_match_team(fd_name: str, odds_api_names: list[str]) -> str | None:
    """Match a football-data.org team name to an Odds API team name.

    Uses alias table first, then falls back to fuzzy matching.

    Args:
        fd_name: Team name from football-data.org.
        odds_api_names: List of team names from The Odds API.

    Returns:
        Best matching Odds API team name, or None if no match above threshold.
    """
    fd_norm = _normalize_name(fd_name)

    # Check alias table
    for canonical, aliases in TEAM_NAME_ALIASES.items():
        if fd_name == canonical or fd_name in aliases:
            all_names = [canonical, *aliases]
            for api_name in odds_api_names:
                if api_name in all_names or _normalize_name(api_name) in [
                    _normalize_name(n) for n in all_names
                ]:
                    return api_name

    # Fuzzy match
    best_match: str | None = None
    best_score = 0.0
    for api_name in odds_api_names:
        score = SequenceMatcher(None, fd_norm, _normalize_name(api_name)).ratio()
        if score > best_score:
            best_score = score
            best_match = api_name

    if best_score >= 0.6:
        return best_match
    return None


# ================================================================
# Data download
# ================================================================


async def download_football_data_csv(
    session: aiohttp.ClientSession,
    league_code: str,
    season: str,
) -> list[HistoricalMatch]:
    """Download and parse a football-data.org CSV file.

    Args:
        session: aiohttp session.
        league_code: E0, SP1, D1, I1, F1.
        season: e.g. "2425".

    Returns:
        List of HistoricalMatch objects parsed from the CSV.
    """
    import csv

    url = _football_data_url(league_code, season)
    league_info = FOOTBALL_DATA_LEAGUES[league_code]

    try:
        async with session.get(url) as resp:
            if resp.status != 200:
                logger.warning(
                    "football_data_download_failed",
                    league=league_code,
                    season=season,
                    status=resp.status,
                )
                return []
            text = await resp.text(encoding="utf-8", errors="replace")
    except Exception as e:
        logger.error("football_data_exception", league=league_code, error=str(e))
        return []

    matches: list[HistoricalMatch] = []
    reader = csv.DictReader(io.StringIO(text))

    for row in reader:
        try:
            # Parse date — football-data uses DD/MM/YYYY
            date_str = row.get("Date", "")
            if not date_str:
                continue

            # Handle both DD/MM/YYYY and DD/MM/YY formats
            for fmt in ("%d/%m/%Y", "%d/%m/%y"):
                try:
                    match_date = datetime.strptime(date_str, fmt)
                    break
                except ValueError:
                    continue
            else:
                logger.warning("football_data_bad_date", date=date_str)
                continue

            home = row.get("HomeTeam", "").strip()
            away = row.get("AwayTeam", "").strip()
            fthg = row.get("FTHG", "")
            ftag = row.get("FTAG", "")
            ftr = row.get("FTR", "")

            if not all([home, away, fthg, ftag, ftr]):
                continue

            # Pinnacle odds columns (if available)
            fd_home = _safe_float(row.get("PSH") or row.get("B365H"))
            fd_draw = _safe_float(row.get("PSD") or row.get("B365D"))
            fd_away = _safe_float(row.get("PSA") or row.get("B365A"))
            fd_over = _safe_float(row.get("P>2.5") or row.get("BbAv>2.5"))
            fd_under = _safe_float(row.get("P<2.5") or row.get("BbAv<2.5"))

            match_id = f"{league_code}_{season}_{match_date.strftime('%Y%m%d')}_{home}_{away}"

            matches.append(
                HistoricalMatch(
                    match_id=match_id,
                    date=match_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    league=league_info["name"],
                    season=season,
                    home_team=home,
                    away_team=away,
                    home_goals=int(fthg),
                    away_goals=int(ftag),
                    result=ftr,
                    fd_home_odds=fd_home,
                    fd_draw_odds=fd_draw,
                    fd_away_odds=fd_away,
                    fd_over_25=fd_over,
                    fd_under_25=fd_under,
                )
            )
        except (ValueError, KeyError) as e:
            logger.warning("football_data_parse_error", error=str(e))
            continue

    logger.info(
        "football_data_parsed",
        league=league_code,
        season=season,
        matches=len(matches),
    )
    return matches


def _safe_float(value: str | None) -> float | None:
    """Safely convert a string to float, returning None on failure."""
    if value is None or value.strip() == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


async def fetch_historical_odds_for_match(
    odds_client: object,
    sport_key: str,
    match: HistoricalMatch,
    markets: str = "h2h,totals",
) -> None:
    """Fetch opening and closing odds for a match from The Odds API.

    Opening: 24h before match start.
    Closing: 2h before match start.
    Modifies match in place.

    Args:
        odds_client: OddsApiClient instance.
        sport_key: The Odds API sport key.
        match: HistoricalMatch to enrich with odds.
        markets: Comma-separated market types.
    """
    from arbo.connectors.odds_api_client import OddsApiClient

    if not isinstance(odds_client, OddsApiClient):
        return

    match_dt = datetime.fromisoformat(match.date.replace("Z", "+00:00"))

    # Opening odds: 24h before
    opening_date = (match_dt - timedelta(hours=24)).strftime("%Y-%m-%dT%H:%M:%SZ")
    events_open, _, _ = await odds_client.get_historical_odds(
        sport_key, opening_date, markets=markets
    )

    # Find matching event
    open_event = _find_matching_event(events_open, match.home_team, match.away_team)
    if open_event:
        match.opening_odds = _extract_odds(open_event)

    # Rate limiting
    await asyncio.sleep(1.5)

    # Closing odds: 2h before
    closing_date = (match_dt - timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%SZ")
    events_close, _, _ = await odds_client.get_historical_odds(
        sport_key, closing_date, markets=markets
    )

    close_event = _find_matching_event(events_close, match.home_team, match.away_team)
    if close_event:
        match.closing_odds = _extract_odds(close_event)

    await asyncio.sleep(1.5)


def _find_matching_event(
    events: list,
    home_team: str,
    away_team: str,
) -> object | None:
    """Find an event matching the home/away teams using fuzzy matching."""
    for event in events:
        api_teams = [event.home_team, event.away_team]
        home_match = fuzzy_match_team(home_team, api_teams)
        away_match = fuzzy_match_team(away_team, api_teams)
        if home_match and away_match and home_match != away_match:
            return event
    return None


def _extract_odds(event: object) -> HistoricalOdds:
    """Extract odds from an OddsEvent into HistoricalOdds."""
    from arbo.connectors.odds_api_client import OddsEvent

    if not isinstance(event, OddsEvent):
        return HistoricalOdds()

    h2h = event.get_pinnacle_h2h()
    odds = HistoricalOdds(bookmaker="pinnacle")

    if h2h:
        # Map outcomes — h2h has home_team, away_team, Draw
        for name, price in h2h.items():
            p = float(price)
            if name == event.home_team:
                odds.home_win = p
            elif name == event.away_team:
                odds.away_win = p
            elif name.lower() == "draw":
                odds.draw = p

    # Extract totals if available
    for bm in event.bookmakers:
        if bm.key != "pinnacle":
            continue
        for market in bm.markets:
            if market.key == "totals":
                for oc in market.outcomes:
                    if oc.name == "Over" and float(oc.price) > 0:
                        odds.over_25 = float(oc.price)
                    elif oc.name == "Under" and float(oc.price) > 0:
                        odds.under_25 = float(oc.price)

    return odds


# ================================================================
# Main pipeline
# ================================================================


async def run_backfill(
    output_path: Path,
    seasons: list[str],
    fetch_odds_api: bool = False,
) -> BacktestDataset:
    """Run the full backfill pipeline.

    Args:
        output_path: Path to save the JSON dataset.
        seasons: List of season codes (e.g. ["2425", "2324"]).
        fetch_odds_api: Whether to fetch historical odds from The Odds API.

    Returns:
        The assembled BacktestDataset.
    """
    # Load existing dataset for resume capability
    existing_ids: set[str] = set()
    existing_matches: list[HistoricalMatch] = []
    if output_path.exists():
        try:
            with open(output_path) as f:
                existing_data = json.load(f)
            existing_dataset = BacktestDataset.model_validate(existing_data)
            existing_matches = existing_dataset.matches
            existing_ids = {m.match_id for m in existing_matches}
            logger.info("resume_from_existing", n_existing=len(existing_ids))
        except Exception as e:
            logger.warning("resume_load_failed", error=str(e))

    ssl_ctx = ssl.create_default_context(cafile=certifi.where())
    connector = aiohttp.TCPConnector(ssl=ssl_ctx)
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=60),
        connector=connector,
    ) as session:
        all_matches: list[HistoricalMatch] = list(existing_matches)
        all_leagues: set[str] = set()

        # Download football-data.org CSVs
        for league_code, league_info in FOOTBALL_DATA_LEAGUES.items():
            for season in seasons:
                matches = await download_football_data_csv(session, league_code, season)
                new_matches = [m for m in matches if m.match_id not in existing_ids]
                all_matches.extend(new_matches)
                existing_ids.update(m.match_id for m in new_matches)
                all_leagues.add(league_info["name"])

                logger.info(
                    "league_season_done",
                    league=league_code,
                    season=season,
                    new=len(new_matches),
                    total=len(all_matches),
                )

        # Optionally fetch historical odds from The Odds API
        if fetch_odds_api:
            try:
                from arbo.connectors.odds_api_client import OddsApiClient

                odds_client = OddsApiClient(session=session)
                try:
                    for match in all_matches:
                        if match.opening_odds is not None:
                            continue  # Already fetched

                        # Find the sport key for this league
                        sport_key = None
                        for _lc, info in FOOTBALL_DATA_LEAGUES.items():
                            if info["name"] == match.league:
                                sport_key = info["odds_api_key"]
                                break

                        if sport_key is None:
                            continue

                        await fetch_historical_odds_for_match(
                            odds_client, sport_key, match, markets="h2h,totals"
                        )

                        # Check quota
                        if (
                            odds_client.remaining_quota is not None
                            and odds_client.remaining_quota < 100
                        ):
                            logger.warning(
                                "odds_api_quota_low_stopping",
                                remaining=odds_client.remaining_quota,
                            )
                            break
                finally:
                    # Don't close the session we were given
                    odds_client._session = None
            except ImportError:
                logger.warning("odds_api_client_not_available")

    # Sort by date
    all_matches.sort(key=lambda m: m.date)

    dataset = BacktestDataset(
        n_matches=len(all_matches),
        leagues=sorted(all_leagues),
        seasons=sorted(set(seasons)),
        matches=all_matches,
    )

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(dataset.model_dump(), f, indent=2, default=str)

    logger.info(
        "backfill_complete",
        n_matches=len(all_matches),
        leagues=len(all_leagues),
        output=str(output_path),
    )
    return dataset


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Backfill historical match data")
    parser.add_argument(
        "--output",
        type=str,
        default="data/backtest_dataset.json",
        help="Output path for the dataset JSON",
    )
    parser.add_argument(
        "--seasons",
        type=str,
        default="2425,2324",
        help="Comma-separated season codes (e.g. 2425,2324)",
    )
    parser.add_argument(
        "--fetch-odds",
        action="store_true",
        help="Also fetch historical odds from The Odds API (uses quota)",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    seasons = [s.strip() for s in args.seasons.split(",")]

    asyncio.run(run_backfill(output_path, seasons, fetch_odds_api=args.fetch_odds))


if __name__ == "__main__":
    main()
