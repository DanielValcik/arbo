"""Export paper trades from VPS dashboard API to local JSON.

Fetches resolved paper trades (strategy=C) from the VPS dashboard
endpoint and saves them locally for backtest validation.

Usage:
    python3 research/export_paper_trades.py
    python3 research/export_paper_trades.py --host 18.135.109.36 --port 8099
"""

import argparse
import json
import ssl
import sys
import urllib.request
from base64 import b64encode
from datetime import datetime, timezone
from getpass import getpass
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "data"
OUTPUT_FILE = OUTPUT_DIR / "paper_trades_export.json"

DEFAULT_HOST = "18.135.109.36"
DEFAULT_PORT = 8099


def fetch_paper_trades(
    host: str,
    port: int,
    username: str,
    password: str,
    strategy: str = "C",
    statuses: str = "won,lost",
) -> dict:
    """Fetch paper trades from VPS dashboard API."""
    url = f"http://{host}:{port}/api/paper-trades?strategy={strategy}&status={statuses}"
    credentials = b64encode(f"{username}:{password}".encode()).decode()

    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Basic {credentials}",
            "Accept": "application/json",
        },
    )

    # Skip SSL verification for local/VPS connections
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    try:
        with urllib.request.urlopen(req, timeout=30, context=ctx) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        print(f"HTTP Error {e.code}: {e.reason}")
        if e.code == 401:
            print("Check dashboard credentials (DASHBOARD_USER / DASHBOARD_PASS)")
        sys.exit(1)
    except urllib.error.URLError as e:
        print(f"Connection failed: {e.reason}")
        print(f"Is the dashboard running on {host}:{port}?")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export paper trades from VPS")
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"VPS host (default: {DEFAULT_HOST})")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Dashboard port (default: {DEFAULT_PORT})")
    parser.add_argument("--user", default="arbo", help="Dashboard username")
    parser.add_argument("--password", default=None, help="Dashboard password (or set DASHBOARD_PASS env var)")
    parser.add_argument("--strategy", default="C", help="Strategy filter (default: C)")
    parser.add_argument("--status", default="won,lost", help="Status filter (default: won,lost)")
    args = parser.parse_args()

    import os

    password = args.password or os.environ.get("DASHBOARD_PASS")
    if not password:
        password = getpass("Dashboard password: ")

    print(f"Fetching paper trades from {args.host}:{args.port}...")
    data = fetch_paper_trades(
        host=args.host,
        port=args.port,
        username=args.user,
        password=password,
        strategy=args.strategy,
        statuses=args.status,
    )

    if "error" in data:
        print(f"API error: {data['error']}")
        sys.exit(1)

    trades = data.get("trades", [])
    print(f"Fetched {len(trades)} paper trades")

    if not trades:
        print("No trades to export.")
        return

    # Add export metadata
    export = {
        "meta": {
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "source": f"{args.host}:{args.port}",
            "strategy": args.strategy,
            "status_filter": args.status,
            "count": len(trades),
        },
        "trades": trades,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(export, f, indent=2)

    print(f"Saved to {OUTPUT_FILE}")

    # Summary stats
    pnls = [t["pnl"] for t in trades if t.get("pnl") is not None]
    if pnls:
        total = sum(pnls)
        wins = sum(1 for p in pnls if p > 0)
        print(f"  Total PnL: ${total:.2f}")
        print(f"  Win rate:  {wins}/{len(pnls)} ({100*wins/len(pnls):.1f}%)")
        print(f"  Avg PnL:   ${total/len(pnls):.2f}")


if __name__ == "__main__":
    main()
