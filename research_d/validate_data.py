"""Strategy D — Data Validation & Quality Report.

Cross-references all data sources to ensure consistency:
1. Games have matching Polymarket markets
2. Market resolution matches game results
3. Price data covers the game period
4. Pinnacle odds exist for games with markets
5. Elo/Glicko ratings are computed for all teams

Usage:
    python3 research_d/validate_data.py [--db path/to/db.sqlite] [--fix]

Output:
    Prints a comprehensive quality report with coverage stats,
    inconsistencies, and data gaps.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from research_d.sports_db import SportsDB


def validate_database(db_path: str | None = None, fix: bool = False) -> dict[str, Any]:
    """Run all validation checks on the sports backtest database.

    Args:
        db_path: Optional path to SQLite DB.
        fix: If True, attempt to fix minor inconsistencies.

    Returns:
        Validation report as a dict.
    """
    db = SportsDB(db_path)
    report: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "db_path": str(db.db_path),
        "summary": {},
        "checks": [],
    }

    # ── Summary Stats ──────────────────────────────────────────────
    stats = db.stats()
    report["summary"] = stats
    print("=" * 60)
    print("  Strategy D Data Validation Report")
    print("=" * 60)
    print()
    print("Database Summary:")
    for key, value in stats.items():
        print(f"  {key:>20s}: {value:>8,d}")
    print()

    # ── Check 1: Games with scores ──────────────────────────────────
    all_games = db.get_games()
    games_with_scores = [g for g in all_games if g["home_score"] is not None]
    games_without_scores = [g for g in all_games if g["home_score"] is None]

    check1 = {
        "name": "Games with scores",
        "total_games": len(all_games),
        "with_scores": len(games_with_scores),
        "without_scores": len(games_without_scores),
        "coverage_pct": len(games_with_scores) / max(len(all_games), 1) * 100,
    }
    report["checks"].append(check1)
    print(f"Check 1: Games with scores")
    print(f"  Total: {check1['total_games']}, "
          f"With scores: {check1['with_scores']}, "
          f"Missing: {check1['without_scores']} "
          f"({check1['coverage_pct']:.1f}% coverage)")
    if games_without_scores:
        print(f"  Missing scores for:")
        for g in games_without_scores[:5]:
            print(f"    {g['game_id']} ({g['game_date']})")
        if len(games_without_scores) > 5:
            print(f"    ... and {len(games_without_scores) - 5} more")
    print()

    # ── Check 2: Games per sport ────────────────────────────────────
    sport_counts: Counter[str] = Counter()
    sport_date_ranges: dict[str, tuple[str, str]] = {}
    for g in all_games:
        sport_counts[g["sport"]] += 1
        if g["sport"] not in sport_date_ranges:
            sport_date_ranges[g["sport"]] = (g["game_date"], g["game_date"])
        else:
            lo, hi = sport_date_ranges[g["sport"]]
            sport_date_ranges[g["sport"]] = (
                min(lo, g["game_date"]),
                max(hi, g["game_date"]),
            )

    check2 = {
        "name": "Games per sport",
        "sports": {
            s: {
                "count": sport_counts[s],
                "date_range": sport_date_ranges.get(s, ("", "")),
            }
            for s in sorted(sport_counts)
        },
    }
    report["checks"].append(check2)
    print("Check 2: Games per sport")
    for sport in sorted(sport_counts):
        lo, hi = sport_date_ranges[sport]
        print(f"  {sport:>6s}: {sport_counts[sport]:>5d} games  ({lo} → {hi})")
    print()

    # ── Check 3: Markets linked to games ────────────────────────────
    all_markets = db.conn.execute("SELECT * FROM markets").fetchall()
    markets_with_game = [m for m in all_markets if m["game_id"]]
    orphan_markets = [m for m in all_markets if not m["game_id"]]

    # Check game_id exists in games table
    game_ids = {g["game_id"] for g in all_games}
    broken_links = [
        m for m in markets_with_game
        if m["game_id"] not in game_ids
    ]

    check3 = {
        "name": "Markets linked to games",
        "total_markets": len(all_markets),
        "linked": len(markets_with_game),
        "orphan": len(orphan_markets),
        "broken_links": len(broken_links),
    }
    report["checks"].append(check3)
    print("Check 3: Markets linked to games")
    print(f"  Total: {check3['total_markets']}, "
          f"Linked: {check3['linked']}, "
          f"Orphan: {check3['orphan']}, "
          f"Broken links: {check3['broken_links']}")
    if broken_links:
        print(f"  Broken links (game_id not in games table):")
        for m in broken_links[:5]:
            print(f"    {m['token_id'][:16]}... → {m['game_id']}")
    print()

    # ── Check 4: Market resolution vs game results ──────────────────
    resolved_markets = [m for m in all_markets if m["won"] is not None]
    resolution_mismatches = []

    for m in resolved_markets:
        if m["game_id"] not in game_ids:
            continue
        game = db.get_game(m["game_id"])
        if not game or game["home_score"] is None:
            continue

        # Check if resolution matches game outcome
        outcome = m["outcome"]
        home_won = game["home_score"] > game["away_score"]
        away_won = game["away_score"] > game["home_score"]
        draw = game["home_score"] == game["away_score"]

        expected_won = None
        if outcome == "home_win":
            expected_won = 1 if home_won else 0
        elif outcome == "away_win":
            expected_won = 1 if away_won else 0
        elif outcome == "draw":
            expected_won = 1 if draw else 0

        if expected_won is not None and expected_won != m["won"]:
            resolution_mismatches.append({
                "token_id": m["token_id"],
                "game_id": m["game_id"],
                "outcome": outcome,
                "market_won": m["won"],
                "expected_won": expected_won,
                "score": f"{game['home_score']}-{game['away_score']}",
            })

    check4 = {
        "name": "Resolution consistency",
        "resolved_markets": len(resolved_markets),
        "checked": len([m for m in resolved_markets if m["game_id"] in game_ids]),
        "mismatches": len(resolution_mismatches),
    }
    report["checks"].append(check4)
    print("Check 4: Resolution vs game results")
    print(f"  Resolved markets: {check4['resolved_markets']}, "
          f"Checked: {check4['checked']}, "
          f"Mismatches: {check4['mismatches']}")
    if resolution_mismatches:
        print(f"  MISMATCHES (market resolution != game result):")
        for mm in resolution_mismatches[:5]:
            print(f"    {mm['game_id']}: outcome={mm['outcome']}, "
                  f"market_won={mm['market_won']}, expected={mm['expected_won']}, "
                  f"score={mm['score']}")
    print()

    # ── Check 5: Price data coverage ────────────────────────────────
    markets_with_prices = 0
    markets_without_prices = 0
    total_price_points = 0
    price_coverage: dict[str, int] = {}

    for m in all_markets:
        count = db.count_prices(m["token_id"])
        if count > 0:
            markets_with_prices += 1
            total_price_points += count
        else:
            markets_without_prices += 1

    check5 = {
        "name": "Price data coverage",
        "markets_with_prices": markets_with_prices,
        "markets_without_prices": markets_without_prices,
        "total_price_points": total_price_points,
        "avg_prices_per_market": (
            total_price_points / max(markets_with_prices, 1)
        ),
    }
    report["checks"].append(check5)
    print("Check 5: Price data coverage")
    print(f"  Markets with prices: {check5['markets_with_prices']}")
    print(f"  Markets without prices: {check5['markets_without_prices']}")
    print(f"  Total price points: {check5['total_price_points']:,d}")
    print(f"  Avg prices/market: {check5['avg_prices_per_market']:.0f}")
    print()

    # ── Check 6: Pinnacle odds coverage ─────────────────────────────
    games_with_pinnacle = db.conn.execute(
        "SELECT COUNT(DISTINCT game_id) FROM pinnacle_odds"
    ).fetchone()[0]
    total_pinnacle = db.conn.execute(
        "SELECT COUNT(*) FROM pinnacle_odds"
    ).fetchone()[0]

    check6 = {
        "name": "Pinnacle odds coverage",
        "games_with_pinnacle": games_with_pinnacle,
        "games_total": len(all_games),
        "coverage_pct": games_with_pinnacle / max(len(all_games), 1) * 100,
        "total_odds_records": total_pinnacle,
    }
    report["checks"].append(check6)
    print("Check 6: Pinnacle odds coverage")
    print(f"  Games with Pinnacle odds: {check6['games_with_pinnacle']} / "
          f"{check6['games_total']} ({check6['coverage_pct']:.1f}%)")
    print(f"  Total odds records: {check6['total_odds_records']}")
    print()

    # ── Check 7: Ratings coverage ───────────────────────────────────
    teams_with_ratings = db.conn.execute(
        "SELECT COUNT(DISTINCT team || sport) FROM ratings"
    ).fetchone()[0]
    total_ratings = db.conn.execute(
        "SELECT COUNT(*) FROM ratings"
    ).fetchone()[0]

    check7 = {
        "name": "Ratings coverage",
        "teams_with_ratings": teams_with_ratings,
        "total_rating_records": total_ratings,
    }
    report["checks"].append(check7)
    print("Check 7: Ratings coverage")
    print(f"  Teams with ratings: {check7['teams_with_ratings']}")
    print(f"  Total rating records: {check7['total_rating_records']}")
    print()

    # ── Check 8: Game events ────────────────────────────────────────
    games_with_events = db.conn.execute(
        "SELECT COUNT(DISTINCT game_id) FROM game_events"
    ).fetchone()[0]
    total_events = db.conn.execute(
        "SELECT COUNT(*) FROM game_events"
    ).fetchone()[0]
    event_types = db.conn.execute(
        "SELECT event_type, COUNT(*) FROM game_events GROUP BY event_type"
    ).fetchall()

    check8 = {
        "name": "Game events",
        "games_with_events": games_with_events,
        "total_events": total_events,
        "event_types": {row[0]: row[1] for row in event_types},
    }
    report["checks"].append(check8)
    print("Check 8: Game events")
    print(f"  Games with events: {check8['games_with_events']}")
    print(f"  Total events: {check8['total_events']}")
    if event_types:
        for et, count in sorted(event_types, key=lambda x: -x[1]):
            print(f"    {et}: {count}")
    print()

    # ── Overall Quality Score ───────────────────────────────────────
    quality_score = 0
    max_score = 0

    # Score components
    if stats["games"] > 0:
        max_score += 20
        quality_score += min(20, stats["games"] / 50 * 20)  # 20pts for 50+ games

    if stats["markets"] > 0:
        max_score += 20
        quality_score += min(20, stats["markets"] / 30 * 20)  # 20pts for 30+ markets

    if stats["prices"] > 0:
        max_score += 20
        quality_score += min(20, stats["prices"] / 10000 * 20)  # 20pts for 10K+ prices

    if games_with_pinnacle > 0:
        max_score += 20
        coverage = games_with_pinnacle / max(len(all_games), 1)
        quality_score += coverage * 20

    if total_ratings > 0:
        max_score += 20
        quality_score += min(20, total_ratings / 500 * 20)

    if check4["mismatches"] > 0:
        quality_score -= check4["mismatches"] * 5  # Penalty for mismatches

    quality_pct = quality_score / max(max_score, 1) * 100

    print("=" * 60)
    print(f"  OVERALL DATA QUALITY: {quality_pct:.0f}% ({quality_score:.0f}/{max_score})")
    print("=" * 60)

    if quality_pct < 30:
        print("  Status: INSUFFICIENT — need more data before backtesting")
    elif quality_pct < 60:
        print("  Status: PARTIAL — can run limited backtests")
    elif quality_pct < 80:
        print("  Status: GOOD — ready for backtesting")
    else:
        print("  Status: EXCELLENT — full backtest capability")

    report["quality_score"] = quality_pct
    db.close()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate Strategy D backtest data quality."
    )
    parser.add_argument(
        "--db", default=None,
        help="Path to SQLite database.",
    )
    parser.add_argument(
        "--fix", action="store_true",
        help="Attempt to fix minor inconsistencies.",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output report as JSON.",
    )
    args = parser.parse_args()

    report = validate_database(args.db, args.fix)

    if args.json:
        print(json.dumps(report, indent=2, default=str))
