"""Fuzzy event matching: Polymarket markets ↔ The Odds API events.

Matches Polymarket prediction market questions to bookmaker events
using team name similarity (rapidfuzz) and time proximity.

Adapted from Sprint 2 Matchbook matcher for Polymarket.
See brief PM-003 for full specification.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta
from decimal import Decimal  # noqa: TC003 — used at runtime in MatchedPair
from pathlib import Path

import yaml
from rapidfuzz import fuzz

from arbo.connectors.market_discovery import GammaMarket  # noqa: TC001
from arbo.connectors.odds_api_client import OddsEvent  # noqa: TC001
from arbo.utils.logger import get_logger

logger = get_logger("event_matcher")

_CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "config"

# Suffixes to strip during normalization
_STRIP_SUFFIXES = re.compile(r"\b(FC|CF|SC|BC|AC|SS|SSC|ACF|AFC|SL|AS|VfB|VfL|TSG|RB|SK)\b", re.I)

# Market type detection patterns (based on production Polymarket questions)
_OU_PATTERN = re.compile(r":\s*O/U\s+(\d+\.?\d*)")  # ": O/U 2.5"
_SPREAD_PATTERN = re.compile(r"Spread:\s*.+\(([+-]?\d+\.?\d*)\)")  # "Spread: Nice (-2.5)"
_BTTS_PATTERN = re.compile(r"Both Teams to Score", re.I)

# Common patterns in Polymarket soccer questions (match-level: 2 teams)
_TEAM_PATTERNS = [
    # "Will Arsenal win against Chelsea?"
    re.compile(r"will\s+(.+?)\s+(?:win|beat|defeat)\s+(?:against\s+)?(.+?)[\?\.]", re.I),
    # "Arsenal vs Chelsea" or "Arsenal v Chelsea"
    re.compile(r"(.+?)\s+(?:vs\.?|v\.?|versus)\s+(.+?)[\?\.]?$", re.I),
    # "Who will win: Arsenal or Chelsea?"
    re.compile(r"who\s+will\s+win.*?:\s*(.+?)\s+or\s+(.+?)[\?\.]", re.I),
    # "Arsenal to win EPL match against Chelsea"
    re.compile(r"(.+?)\s+to\s+win\s+.*?(?:match|game)\s+(?:against|vs)\s+(.+?)[\?\.]", re.I),
]

# Seasonal/futures patterns: team + league (1 team, whole season)
# Year formats: 2025/26, 2025-26, en-dash (U+2013), em-dash (U+2014)
_YEAR_OPT = r"(?:\d{4}[/\-\u2013\u2014]?\d{2,4}\s+)?"

_SEASONAL_PATTERNS = [
    # "Will Arsenal win the Premier League?" / "Will Arsenal win the 2025-26 EPL?"
    re.compile(rf"will\s+(.+?)\s+win\s+(?:the\s+)?{_YEAR_OPT}(.+?)[\?\.]?\s*$", re.I),
    # "Arsenal to win the Premier League"
    re.compile(rf"^(.+?)\s+to\s+win\s+(?:the\s+)?{_YEAR_OPT}(.+?)[\?\.]?\s*$", re.I),
    # "Will Arsenal be EPL champions?" / "Will Arsenal be 2025/26 champions?"
    re.compile(
        rf"will\s+(.+?)\s+(?:be|become)\s+(?:the\s+)?{_YEAR_OPT}(.+?)\s+champions?[\?\.]?\s*$",
        re.I,
    ),
]

# League name keywords → The Odds API sport key
LEAGUE_SPORT_KEYS: dict[str, str] = {
    "epl": "soccer_epl",
    "premier league": "soccer_epl",
    "english premier league": "soccer_epl",
    "la liga": "soccer_spain_la_liga",
    "spanish la liga": "soccer_spain_la_liga",
    "bundesliga": "soccer_germany_bundesliga",
    "german bundesliga": "soccer_germany_bundesliga",
    "serie a": "soccer_italy_serie_a",
    "italian serie a": "soccer_italy_serie_a",
    "ligue 1": "soccer_france_ligue_one",
    "ligue one": "soccer_france_ligue_one",
    "french ligue 1": "soccer_france_ligue_one",
    "champions league": "soccer_uefa_champs_league",
    "uefa champions league": "soccer_uefa_champs_league",
    "ucl": "soccer_uefa_champs_league",
}

# Default match threshold (0.0-1.0)
DEFAULT_THRESHOLD = 0.85

# Max time difference for matching
MAX_TIME_DELTA = timedelta(hours=24)


def _ensure_utc(dt: datetime) -> datetime:
    """Ensure datetime is timezone-aware (assume UTC if naive)."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt


def extract_teams_from_question(question: str) -> tuple[str, str] | None:
    """Extract two team names from a Polymarket market question.

    Tries multiple regex patterns to parse team names from natural language.

    Args:
        question: Market question string.

    Returns:
        (team_a, team_b) tuple or None if no teams found.
    """
    for pattern in _TEAM_PATTERNS:
        match = pattern.search(question)
        if match:
            team_a = match.group(1).strip()
            team_b = match.group(2).strip()
            if len(team_a) > 1 and len(team_b) > 1:
                return (team_a, team_b)
    return None


def extract_team_from_seasonal(question: str) -> tuple[str, str] | None:
    """Extract (team, league) from a seasonal/futures Polymarket question.

    Handles patterns like:
    - "Will Arsenal win the Premier League?"
    - "Arsenal to win the 2025/26 EPL"
    - "Will Arsenal be EPL champions?"

    Returns:
        (team_name, league_string) or None.
    """
    for pattern in _SEASONAL_PATTERNS:
        match = pattern.search(question)
        if match:
            team = match.group(1).strip()
            league = match.group(2).strip()
            if len(team) > 1 and len(league) > 1:
                return (team, league)
    return None


def identify_league(league_str: str) -> str | None:
    """Map a league string from a market question to Odds API sport key.

    Searches for known league keywords within the string (case-insensitive).

    Args:
        league_str: League name extracted from a Polymarket question.

    Returns:
        Odds API sport key (e.g. "soccer_epl") or None.
    """
    lower = league_str.lower()
    for keyword, sport_key in LEAGUE_SPORT_KEYS.items():
        if keyword in lower:
            return sport_key
    return None


def detect_market_type(question: str) -> tuple[str, float | None]:
    """Detect the market type from a Polymarket question.

    Args:
        question: Polymarket market question string.

    Returns:
        (market_type, line) where market_type is one of
        "moneyline", "totals", "spreads", "btts" and line is
        the numeric line value (or None for moneyline/btts).
    """
    ou_match = _OU_PATTERN.search(question)
    if ou_match:
        return ("totals", float(ou_match.group(1)))

    spread_match = _SPREAD_PATTERN.search(question)
    if spread_match:
        return ("spreads", float(spread_match.group(1)))

    if _BTTS_PATTERN.search(question):
        return ("btts", None)

    return ("moneyline", None)


class MatchedPair:
    """A matched Polymarket market + Odds API event pair.

    Supports both match-level pairs (with odds_event) and seasonal pairs
    (with outright odds, no individual event).
    """

    def __init__(
        self,
        polymarket: GammaMarket,
        odds_event: OddsEvent | None,
        match_score: float,
        pinnacle_prob: Decimal | None = None,
        match_type: str = "match",
        outright_team: str | None = None,
        sport_key: str | None = None,
        market_type: str | None = None,
        market_line: float | None = None,
    ) -> None:
        self.polymarket = polymarket
        self.odds_event = odds_event
        self.match_score = match_score
        self.pinnacle_prob = pinnacle_prob
        self.match_type = match_type  # "match" or "seasonal"
        self.outright_team = outright_team
        self.sport_key = sport_key
        self.market_type = market_type  # "moneyline", "totals", "spreads", "btts"
        self.market_line = market_line  # 2.5, -1.5, etc.


def load_aliases(path: Path | None = None) -> dict[str, list[str]]:
    """Load team aliases from YAML config.

    Returns:
        Dict mapping canonical name → list of aliases.
    """
    if path is None:
        path = _CONFIG_DIR / "team_aliases.yaml"
    if not path.exists():
        return {}

    with open(path) as f:
        data = yaml.safe_load(f)

    flat: dict[str, list[str]] = {}
    if not isinstance(data, dict):
        return flat

    for _sport, teams in data.items():
        if isinstance(teams, dict):
            for canonical, aliases in teams.items():
                flat[canonical] = aliases if isinstance(aliases, list) else []

    return flat


class EventMatcher:
    """Matches Polymarket markets to The Odds API events.

    Uses fuzzy team name matching (rapidfuzz) with alias resolution.
    """

    def __init__(
        self,
        aliases: dict[str, list[str]] | None = None,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> None:
        self._threshold = threshold
        self._alias_map: dict[str, str] = {}
        if aliases:
            for canonical, alias_list in aliases.items():
                norm = self._normalize_name(canonical)
                self._alias_map[norm] = norm
                for alias in alias_list:
                    self._alias_map[self._normalize_name(alias)] = norm

    def match_markets(
        self,
        polymarkets: list[GammaMarket],
        odds_events: list[OddsEvent],
    ) -> list[MatchedPair]:
        """Match Polymarket markets to Odds API events.

        Only processes soccer-category markets that have extractable team names.

        Args:
            polymarkets: Polymarket soccer markets.
            odds_events: Events from The Odds API.

        Returns:
            List of matched pairs with scores and Pinnacle probabilities.
        """
        matched: list[MatchedPair] = []
        used_oa_ids: set[str] = set()

        for market in polymarkets:
            if market.category != "soccer":
                continue

            # Detect market type before team extraction
            market_type, market_line = detect_market_type(market.question)

            # Pre-process question to strip suffixes before team extraction
            clean_q = market.question
            clean_q = re.sub(r":\s*O/U\s+\d+\.?\d*", "", clean_q)
            clean_q = re.sub(r":\s*Both Teams to Score", "", clean_q, flags=re.I)

            teams = extract_teams_from_question(clean_q)
            if teams is None:
                continue

            # BTTS: skip — no BTTS in The Odds API
            if market_type == "btts":
                continue

            team_a, team_b = teams
            best_match: MatchedPair | None = None
            best_score = 0.0

            for event in odds_events:
                if event.id in used_oa_ids:
                    continue

                score = self._score_match(team_a, team_b, event)
                if score >= self._threshold and score > best_score:
                    # Route probability by market type
                    if market_type == "totals" and market_line is not None:
                        pinnacle_prob = event.get_pinnacle_totals_prob(
                            market_line, over=True
                        )
                    elif market_type == "spreads" and market_line is not None:
                        pinnacle_prob = event.get_pinnacle_spreads_prob(
                            event.home_team, market_line
                        )
                    else:
                        # moneyline: use h2h prob for home team
                        pinnacle_prob = event.get_pinnacle_implied_prob(event.home_team)

                    if pinnacle_prob is None:
                        continue  # No matching odds → skip (better than wrong prob)

                    best_score = score
                    best_match = MatchedPair(
                        polymarket=market,
                        odds_event=event,
                        match_score=score,
                        pinnacle_prob=pinnacle_prob,
                        market_type=market_type,
                        market_line=market_line,
                    )

            if best_match is not None:
                used_oa_ids.add(best_match.odds_event.id)
                matched.append(best_match)
                logger.info(
                    "market_matched",
                    poly_q=market.question[:80],
                    oa=f"{best_match.odds_event.home_team} vs {best_match.odds_event.away_team}",
                    score=round(best_score, 3),
                    pinnacle_prob=str(best_match.pinnacle_prob),
                    market_type=market_type,
                    market_line=market_line,
                )

        logger.info(
            "matching_complete",
            polymarket_count=len(polymarkets),
            odds_api_count=len(odds_events),
            matched=len(matched),
        )
        return matched

    def match_seasonal_markets(
        self,
        polymarkets: list[GammaMarket],
        outright_odds: dict[str, dict[str, Decimal]],
    ) -> list[MatchedPair]:
        """Match seasonal Polymarket questions to outright odds.

        Parses "Will X win the Y?" style questions, identifies the league,
        then fuzzy-matches the team name against outright odds teams.

        Args:
            polymarkets: Polymarket markets (filtered to soccer).
            outright_odds: Dict of sport_key → {team_name → implied_prob}.

        Returns:
            List of seasonal matched pairs with probabilities.
        """
        matched: list[MatchedPair] = []

        for market in polymarkets:
            if market.category != "soccer":
                continue

            seasonal = extract_team_from_seasonal(market.question)
            if seasonal is None:
                continue

            team, league_str = seasonal
            sport_key = identify_league(league_str)
            if sport_key is None:
                continue

            odds = outright_odds.get(sport_key, {})
            if not odds:
                continue

            # Fuzzy match team name against outright outcomes
            best_name: str | None = None
            best_score = 0.0
            for outright_team in odds:
                score = self._name_similarity(team, outright_team)
                if score > best_score:
                    best_score = score
                    best_name = outright_team

            if best_name is not None and best_score >= self._threshold:
                pinnacle_prob = odds[best_name]
                matched.append(
                    MatchedPair(
                        polymarket=market,
                        odds_event=None,
                        match_score=best_score,
                        pinnacle_prob=pinnacle_prob,
                        match_type="seasonal",
                        outright_team=best_name,
                        sport_key=sport_key,
                    )
                )
                logger.info(
                    "seasonal_market_matched",
                    poly_q=market.question[:80],
                    team=best_name,
                    sport_key=sport_key,
                    score=round(best_score, 3),
                    pinnacle_prob=str(pinnacle_prob),
                )

        logger.info(
            "seasonal_matching_complete",
            polymarket_count=len(polymarkets),
            outright_leagues=len(outright_odds),
            matched=len(matched),
        )
        return matched

    def match_seasonal_via_match_odds(
        self,
        polymarkets: list[GammaMarket],
        match_events: list[OddsEvent],
    ) -> list[MatchedPair]:
        """Match seasonal Polymarket questions to teams in match-level events.

        When outright odds aren't available (e.g., The Odds API doesn't support
        domestic league outrights), derives team strength from average Pinnacle
        win probability across upcoming match-level events.

        For "Will Arsenal win the EPL?":
        1. Find "Arsenal" in EPL match events (as home or away team)
        2. Average their Pinnacle implied win probability across matches
        3. Use this as a proxy signal for team strength

        Note: This is a weaker signal than true outright odds — the average
        match win probability (e.g. 0.60) doesn't directly translate to
        league win probability (e.g. 0.35). But it provides usable features
        for the XGBoost value model to learn the mapping.

        Args:
            polymarkets: Polymarket soccer markets.
            match_events: Match-level events from The Odds API.

        Returns:
            List of seasonal matched pairs with derived probabilities.
        """
        matched: list[MatchedPair] = []

        # Build index: team → list of (event, pinnacle_prob)
        team_probs: dict[str, list[tuple[OddsEvent, Decimal]]] = {}
        for event in match_events:
            # Home team
            prob_home = event.get_pinnacle_implied_prob(event.home_team)
            if prob_home is not None:
                norm_home = self._resolve_alias(self._normalize_name(event.home_team))
                team_probs.setdefault(norm_home, []).append((event, prob_home))

            # Away team
            prob_away = event.get_pinnacle_implied_prob(event.away_team)
            if prob_away is not None:
                norm_away = self._resolve_alias(self._normalize_name(event.away_team))
                team_probs.setdefault(norm_away, []).append((event, prob_away))

        for market in polymarkets:
            if market.category != "soccer":
                continue

            seasonal = extract_team_from_seasonal(market.question)
            if seasonal is None:
                continue

            team, league_str = seasonal
            sport_key = identify_league(league_str)
            if sport_key is None:
                continue

            norm_team = self._resolve_alias(self._normalize_name(team))

            # Find best fuzzy match in team_probs index
            best_key: str | None = None
            best_score = 0.0
            for indexed_team in team_probs:
                if norm_team == indexed_team:
                    score = 1.0
                else:
                    from rapidfuzz import fuzz

                    score = fuzz.token_sort_ratio(norm_team, indexed_team) / 100.0

                if score > best_score:
                    best_score = score
                    best_key = indexed_team

            if best_key is None or best_score < self._threshold:
                continue

            # Filter to events from the right league
            league_events = [
                (ev, prob) for ev, prob in team_probs[best_key] if ev.sport_key == sport_key
            ]

            if not league_events:
                continue

            # Average Pinnacle win probability across matches
            avg_prob = sum(p for _, p in league_events) / len(league_events)

            # Use the first event for reference
            ref_event = league_events[0][0]
            matched_team_name = ref_event.home_team
            for ev, _ in league_events:
                norm_h = self._resolve_alias(self._normalize_name(ev.home_team))
                norm_a = self._resolve_alias(self._normalize_name(ev.away_team))
                if norm_h == best_key:
                    matched_team_name = ev.home_team
                    break
                if norm_a == best_key:
                    matched_team_name = ev.away_team
                    break

            matched.append(
                MatchedPair(
                    polymarket=market,
                    odds_event=ref_event,
                    match_score=best_score,
                    pinnacle_prob=avg_prob,
                    match_type="seasonal_derived",
                    outright_team=matched_team_name,
                    sport_key=sport_key,
                )
            )
            logger.info(
                "seasonal_derived_match",
                poly_q=market.question[:80],
                team=matched_team_name,
                sport_key=sport_key,
                score=round(best_score, 3),
                avg_pinnacle_prob=str(round(avg_prob, 4)),
                n_matches=len(league_events),
            )

        logger.info(
            "seasonal_derived_matching_complete",
            polymarket_count=len(polymarkets),
            match_events=len(match_events),
            matched=len(matched),
        )
        return matched

    def _score_match(self, team_a: str, team_b: str, event: OddsEvent) -> float:
        """Score match between Polymarket teams and Odds API event.

        Tries both normal and swapped orderings.
        """
        home_score = self._name_similarity(team_a, event.home_team)
        away_score = self._name_similarity(team_b, event.away_team)
        normal = (home_score + away_score) / 2.0

        home_swap = self._name_similarity(team_a, event.away_team)
        away_swap = self._name_similarity(team_b, event.home_team)
        swapped = (home_swap + away_swap) / 2.0

        return max(normal, swapped)

    def _name_similarity(self, name_a: str, name_b: str) -> float:
        """Compare two team names after alias expansion and normalization."""
        norm_a = self._resolve_alias(self._normalize_name(name_a))
        norm_b = self._resolve_alias(self._normalize_name(name_b))

        if norm_a == norm_b:
            return 1.0

        return fuzz.token_sort_ratio(norm_a, norm_b) / 100.0

    def _normalize_name(self, name: str) -> str:
        """Lowercase, strip common suffixes (FC, CF, etc.), extra whitespace."""
        name = name.lower().strip()
        name = _STRIP_SUFFIXES.sub("", name)
        name = re.sub(r"\s+", " ", name).strip()
        return name

    def _resolve_alias(self, normalized_name: str) -> str:
        """Resolve a normalized name to its canonical form via alias map."""
        return self._alias_map.get(normalized_name, normalized_name)
