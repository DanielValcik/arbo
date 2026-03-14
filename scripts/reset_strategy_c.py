"""Reset Strategy C paper trading data for clean paper trading start.

Deletes all Strategy C trades, positions, and snapshots from the database.
Run on VPS before starting paper trading with new model.

Usage:
    python3 scripts/reset_strategy_c.py
    python3 scripts/reset_strategy_c.py --dry-run
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


async def reset_strategy_c(dry_run: bool = False) -> None:
    """Delete all Strategy C paper trading data."""
    from sqlalchemy import delete, func, select, text

    from arbo.utils.db import (
        PaperPosition,
        PaperSnapshot,
        PaperTrade,
        get_session_factory,
    )

    factory = get_session_factory()
    async with factory() as session:
        # Count existing data
        trades_count = (
            await session.execute(
                select(func.count()).select_from(PaperTrade).where(
                    PaperTrade.strategy == "C"
                )
            )
        ).scalar() or 0

        positions_count = (
            await session.execute(
                select(func.count()).select_from(PaperPosition).where(
                    PaperPosition.strategy == "C"
                )
            )
        ).scalar() or 0

        snapshots_count = (
            await session.execute(
                select(func.count()).select_from(PaperSnapshot)
            )
        ).scalar() or 0

        print(f"Strategy C data found:")
        print(f"  Trades:    {trades_count}")
        print(f"  Positions: {positions_count}")
        print(f"  Snapshots: {snapshots_count} (all strategies)")

        if dry_run:
            print("\n--dry-run: No data deleted.")
            return

        if trades_count == 0 and positions_count == 0:
            print("\nNo Strategy C data to delete.")
            return

        # Delete Strategy C trades
        result = await session.execute(
            delete(PaperTrade).where(PaperTrade.strategy == "C")
        )
        print(f"\nDeleted {result.rowcount} Strategy C trades")

        # Delete Strategy C positions
        result = await session.execute(
            delete(PaperPosition).where(PaperPosition.strategy == "C")
        )
        print(f"Deleted {result.rowcount} Strategy C positions")

        await session.commit()
        print("\nStrategy C paper data reset complete.")
        print("Restart the service to begin clean paper trading.")


def main() -> None:
    dry_run = "--dry-run" in sys.argv
    asyncio.run(reset_strategy_c(dry_run=dry_run))


if __name__ == "__main__":
    main()
