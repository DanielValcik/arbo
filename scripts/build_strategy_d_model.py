"""Build Strategy D model cache from research_d DB.

Extracts latest Elo ratings + Pinnacle odds from research_d/sports_backtest.sqlite
and saves as compact JSON for production Strategy D to load at init.

Output: arbo/data/strategy_d_model.json (~100KB)

Usage:
    # Local (from small research DB)
    python3 scripts/build_strategy_d_model.py

    # From VPS (large 291GB DB)
    ssh arbo-download
    cd /opt/arbo
    python3 scripts/build_strategy_d_model.py --db /mnt/arbo-data/sports_backtest.sqlite
    scp arbo-download:/opt/arbo/arbo/data/strategy_d_model.json arbo/data/
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

OUTPUT = Path(__file__).resolve().parent.parent / "arbo" / "data" / "strategy_d_model.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="research_d/data/sports_backtest.sqlite")
    parser.add_argument("--out", default=str(OUTPUT))
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"ERROR: DB not found: {db_path}")
        sys.exit(1)

    conn = sqlite3.connect(str(db_path), timeout=60)
    conn.execute("PRAGMA busy_timeout=60000")

    # Get latest Elo per team (NBA only)
    elo: dict[str, list[float]] = {}
    rows = conn.execute("""
        SELECT team, MAX(date) as latest_date, elo, glicko_rating
        FROM ratings
        WHERE sport = 'nba'
        GROUP BY team
    """).fetchall()
    for team, date, elo_val, glicko_val in rows:
        if elo_val is not None:
            elo[team] = [float(elo_val), float(glicko_val or elo_val)]
    print(f"Elo teams: {len(elo)}")

    # Get Pinnacle odds — keyed by simple "nba_team_a_team_b" pattern
    # ESPN game_id format: nba_YYYYMMDD_AWAY_HOME
    pinnacle: dict[str, list[float]] = {}
    rows = conn.execute("""
        SELECT p.game_id, p.home_prob_novig, p.away_prob_novig
        FROM pinnacle_odds p
        JOIN games g ON p.game_id = g.game_id
        WHERE g.sport = 'nba' AND p.home_prob_novig IS NOT NULL
        ORDER BY p.ts DESC
    """).fetchall()

    for game_id, hp, ap in rows:
        if hp and ap and hp > 0 and ap > 0:
            # Extract teams from game_id: nba_YYYYMMDD_AWAY_HOME
            parts = game_id.split("_")
            if len(parts) >= 4:
                away, home = parts[2], parts[3]
                # Store under both orderings for flexible lookup
                key1 = f"nba_{home}_{away}"    # home listed first
                key2 = f"nba_{away}_{home}"    # away listed first
                if key1 not in pinnacle:
                    pinnacle[key1] = [float(hp), float(ap)]
                if key2 not in pinnacle:
                    pinnacle[key2] = [float(ap), float(hp)]

    print(f"Pinnacle games: {len(pinnacle)}")

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"elo": elo, "pinnacle": pinnacle}, indent=None))

    size_kb = output.stat().st_size / 1024
    print(f"Wrote {output} ({size_kb:.0f} KB)")

    conn.close()


if __name__ == "__main__":
    main()
