"""Backfill ensemble_stats from SQLite to PostgreSQL.

Copies 75 days of historical GEFS ensemble data (1,425 rows)
from research/data/gefs_ensemble.sqlite into the production
PostgreSQL ensemble_stats table.

Usage (on VPS):
    python3 scripts/backfill_ensemble_pg.py
"""

import os
import sqlite3
import sys

from dotenv import load_dotenv

load_dotenv()


def main():
    sqlite_path = os.path.join(
        os.path.dirname(__file__), "..", "research", "data", "gefs_ensemble.sqlite"
    )
    if not os.path.exists(sqlite_path):
        print(f"SQLite not found: {sqlite_path}")
        sys.exit(1)

    # Read from SQLite
    sconn = sqlite3.connect(sqlite_path)
    rows = sconn.execute(
        "SELECT city, target_date, ensemble_mean, ensemble_std, "
        "ensemble_min, ensemble_max, n_members FROM ensemble_stats"
    ).fetchall()
    sconn.close()
    print(f"Read {len(rows)} rows from SQLite")

    # Write to PostgreSQL
    import psycopg2

    pg_url = os.getenv("DATABASE_URL", "")
    if pg_url:
        conn = psycopg2.connect(pg_url)
    else:
        conn = psycopg2.connect(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=int(os.getenv("POSTGRES_PORT", "5432")),
            dbname=os.getenv("POSTGRES_DB", "arbo"),
            user=os.getenv("POSTGRES_USER", "arbo"),
            password=os.getenv("POSTGRES_PASSWORD", ""),
        )
    conn.autocommit = False
    cur = conn.cursor()

    inserted = 0
    for city, date_str, mean, std, mn, mx, n_members in rows:
        cur.execute(
            """INSERT INTO ensemble_stats
                (city, target_date, ensemble_mean, ensemble_std,
                 ensemble_min, ensemble_max, n_members)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (city, target_date) DO UPDATE SET
                ensemble_mean = EXCLUDED.ensemble_mean,
                ensemble_std = EXCLUDED.ensemble_std,
                ensemble_min = EXCLUDED.ensemble_min,
                ensemble_max = EXCLUDED.ensemble_max,
                n_members = EXCLUDED.n_members
            """,
            (city, date_str, mean, std, mn, mx, n_members),
        )
        inserted += 1

    conn.commit()
    cur.close()
    conn.close()
    print(f"Inserted {inserted} rows into PostgreSQL ensemble_stats")

    # Verify
    conn = psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        dbname=os.getenv("POSTGRES_DB", "arbo"),
        user=os.getenv("POSTGRES_USER", "arbo"),
        password=os.getenv("POSTGRES_PASSWORD", ""),
    )
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*), COUNT(DISTINCT city), MIN(target_date), MAX(target_date) FROM ensemble_stats")
    count, cities, min_d, max_d = cur.fetchone()
    print(f"Verified: {count} rows, {cities} cities, {min_d} to {max_d}")
    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
