"""Strategy D — Data Pipeline Orchestrator.

Runs all data collection steps in the correct order:
1. Download game results (NBA + EPL)
2. Compute Elo/Glicko-2 ratings
3. Download Pinnacle odds
4. Download Polymarket sports prices
5. Validate data quality

Usage:
    python3 research_d/run_pipeline.py [--sport nba|epl|all] [--step 1-5|all]

Steps:
    1 = Download game results
    2 = Compute Elo/Glicko-2 ratings (requires step 1)
    3 = Download Pinnacle odds (requires step 1 for game matching)
    4 = Download Polymarket sports prices
    5 = Validate all data

Example:
    # Run everything for NBA
    python3 research_d/run_pipeline.py --sport nba

    # Only download game results for all sports
    python3 research_d/run_pipeline.py --step 1

    # Only validate
    python3 research_d/run_pipeline.py --step 5
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

RESEARCH_DIR = Path(__file__).parent
PYTHON = sys.executable


def run_step(step_num: int, description: str, cmd: list[str]) -> bool:
    """Run a pipeline step with timing and error handling.

    Args:
        step_num: Step number (1-5).
        description: Human-readable description.
        cmd: Command to execute.

    Returns:
        True if step succeeded, False otherwise.
    """
    print()
    print("=" * 60)
    print(f"  Step {step_num}: {description}")
    print("=" * 60)
    print(f"  Command: {' '.join(cmd)}")
    print()

    start = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd=str(RESEARCH_DIR.parent),
            capture_output=False,
            text=True,
            timeout=3600,  # 1 hour max per step
        )
        elapsed = time.time() - start

        if result.returncode == 0:
            print(f"\n  Step {step_num} COMPLETED in {elapsed:.1f}s")
            return True
        else:
            print(f"\n  Step {step_num} FAILED (exit code {result.returncode}) "
                  f"in {elapsed:.1f}s")
            return False

    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        print(f"\n  Step {step_num} TIMED OUT after {elapsed:.1f}s")
        return False
    except Exception as e:
        print(f"\n  Step {step_num} ERROR: {e}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Strategy D data pipeline orchestrator.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--sport",
        choices=["nba", "epl", "nfl", "all"],
        default="all",
        help="Sport to process (default: all).",
    )
    parser.add_argument(
        "--step",
        type=str,
        default="all",
        help="Step to run: 1-5 or 'all' (default: all).",
    )
    args = parser.parse_args()

    sport = args.sport
    if args.step == "all":
        steps = [1, 2, 3, 4, 5]
    else:
        steps = [int(s) for s in args.step.split(",")]

    print()
    print("╔" + "═" * 58 + "╗")
    print("║    Strategy D — Data Pipeline Orchestrator              ║")
    print("╠" + "═" * 58 + "╣")
    print(f"║  Sport: {sport:<49s}║")
    print(f"║  Steps: {str(steps):<49s}║")
    print("╚" + "═" * 58 + "╝")

    results: dict[int, bool] = {}
    start_total = time.time()

    # Step 1: Game results
    if 1 in steps:
        results[1] = run_step(
            1,
            f"Download game results ({sport})",
            [PYTHON, "research_d/download_game_results.py", "--sport", sport],
        )

    # Step 2: Elo/Glicko-2 ratings
    if 2 in steps:
        if 1 in results and not results[1]:
            print("\n  Skipping step 2 (step 1 failed)")
            results[2] = False
        else:
            results[2] = run_step(
                2,
                f"Compute Elo/Glicko-2 ratings ({sport})",
                [PYTHON, "research_d/elo_glicko_engine.py", "--sport", sport],
            )

    # Step 3: Pinnacle odds
    if 3 in steps:
        results[3] = run_step(
            3,
            f"Download Pinnacle odds ({sport})",
            [PYTHON, "research_d/download_pinnacle_odds.py", "--sport", sport],
        )

    # Step 4: Polymarket sports prices
    if 4 in steps:
        results[4] = run_step(
            4,
            f"Download Polymarket sports prices ({sport})",
            [PYTHON, "research_d/download_sports_prices.py", "--sport", sport],
        )

    # Step 5: Validate
    if 5 in steps:
        results[5] = run_step(
            5,
            "Validate data quality",
            [PYTHON, "research_d/validate_data.py"],
        )

    # Summary
    elapsed_total = time.time() - start_total
    print()
    print("╔" + "═" * 58 + "╗")
    print("║    Pipeline Summary                                     ║")
    print("╠" + "═" * 58 + "╣")
    for step_num in sorted(results):
        status = "PASS" if results[step_num] else "FAIL"
        emoji = "✓" if results[step_num] else "✗"
        step_names = {
            1: "Game results download",
            2: "Elo/Glicko-2 ratings",
            3: "Pinnacle odds download",
            4: "Polymarket prices download",
            5: "Data validation",
        }
        name = step_names.get(step_num, f"Step {step_num}")
        print(f"║  {emoji} Step {step_num}: {name:<35s} [{status}]  ║")
    print("╠" + "═" * 58 + "╣")

    passed = sum(1 for v in results.values() if v)
    total = len(results)
    print(f"║  Total: {passed}/{total} passed in {elapsed_total:.1f}s"
          f"{' ' * (37 - len(f'{passed}/{total} passed in {elapsed_total:.1f}s'))}║")
    print("╚" + "═" * 58 + "╝")

    if all(results.values()):
        print("\nAll steps passed! Data is ready for backtesting.")
    else:
        print("\nSome steps failed. Fix errors and re-run failed steps.")
        sys.exit(1)


if __name__ == "__main__":
    main()
