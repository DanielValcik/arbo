"""Fuzzy event matching across data sources using rapidfuzz.

Matches Matchbook exchange events to The Odds API bookmaker events
based on team name similarity and start time proximity.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta
from pathlib import Path

import yaml
from pydantic import BaseModel
from rapidfuzz import fuzz

from src.data.odds_api import OddsApiEvent  # noqa: TC001
from src.exchanges.base import ExchangeEvent  # noqa: TC001
from src.utils.logger import get_logger

log = get_logger("event_matcher")

_CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "config"

# Suffixes to strip during normalization
_STRIP_SUFFIXES = re.compile(r"\b(FC|CF|SC|BC|AC|SS|SSC|ACF|AFC|SL|AS|VfB|VfL|TSG|RB|SK)\b", re.I)

# Default match threshold (0.0-1.0)
DEFAULT_THRESHOLD = 0.85

# Max time difference for matching (2 hours)
MAX_TIME_DELTA = timedelta(hours=2)


def _ensure_utc(dt: datetime) -> datetime:
    """Ensure datetime is timezone-aware (assume UTC if naive)."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt


class MatchedEvent(BaseModel):
    matchbook_event: ExchangeEvent
    odds_api_event: OddsApiEvent
    match_score: float


def load_aliases(path: Path | None = None) -> dict[str, list[str]]:
    """Load team aliases from YAML. Returns flat dict: canonical → [aliases]."""
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
    """Matches events across Matchbook and The Odds API sources."""

    def __init__(
        self,
        aliases: dict[str, list[str]] | None = None,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> None:
        self._threshold = threshold
        # Build reverse lookup: alias → canonical name
        self._alias_map: dict[str, str] = {}
        if aliases:
            for canonical, alias_list in aliases.items():
                norm = self._normalize_name(canonical)
                self._alias_map[norm] = norm
                for alias in alias_list:
                    self._alias_map[self._normalize_name(alias)] = norm

    def match_events(
        self,
        matchbook_events: list[ExchangeEvent],
        odds_api_events: list[OddsApiEvent],
    ) -> list[MatchedEvent]:
        """Match Matchbook events to Odds API events by team names and start time."""
        matched: list[MatchedEvent] = []
        used_oa_ids: set[str] = set()

        for mb_event in matchbook_events:
            best_match: MatchedEvent | None = None
            best_score = 0.0

            for oa_event in odds_api_events:
                if oa_event.id in used_oa_ids:
                    continue

                # Check time proximity (ensure both timezone-aware)
                mb_time = _ensure_utc(mb_event.start_time)
                oa_time = _ensure_utc(oa_event.commence_time)
                if abs(mb_time - oa_time) > MAX_TIME_DELTA:
                    continue

                # Score team name similarity
                score = self._score_match(mb_event, oa_event)
                if score >= self._threshold and score > best_score:
                    best_score = score
                    best_match = MatchedEvent(
                        matchbook_event=mb_event,
                        odds_api_event=oa_event,
                        match_score=score,
                    )

            if best_match is not None:
                used_oa_ids.add(best_match.odds_api_event.id)
                matched.append(best_match)
                log.info(
                    "event_matched",
                    mb=f"{mb_event.home_team} vs {mb_event.away_team}",
                    oa=f"{best_match.odds_api_event.home_team} vs {best_match.odds_api_event.away_team}",
                    score=round(best_score, 3),
                )

        log.info(
            "matching_complete",
            matchbook_count=len(matchbook_events),
            odds_api_count=len(odds_api_events),
            matched=len(matched),
        )
        return matched

    def _score_match(self, mb: ExchangeEvent, oa: OddsApiEvent) -> float:
        """Score match between two events. Average of home + away name similarity.

        Tries both normal and swapped home/away to handle sources that
        list teams in different order.
        """
        home_score = self._name_similarity(mb.home_team, oa.home_team)
        away_score = self._name_similarity(mb.away_team, oa.away_team)
        normal = (home_score + away_score) / 2.0

        # Also try swapped (some sources swap home/away)
        home_swap = self._name_similarity(mb.home_team, oa.away_team)
        away_swap = self._name_similarity(mb.away_team, oa.home_team)
        swapped = (home_swap + away_swap) / 2.0

        return max(normal, swapped)

    def _name_similarity(self, name_a: str, name_b: str) -> float:
        """Compare two team names after alias expansion and normalization."""
        norm_a = self._resolve_alias(self._normalize_name(name_a))
        norm_b = self._resolve_alias(self._normalize_name(name_b))

        # Exact match after normalization
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
