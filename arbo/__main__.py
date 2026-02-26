"""Allow running as: python -m arbo

Usage:
  python -m arbo --mode rdh      # New 3-strategy orchestrator (RDH)
  python -m arbo --mode legacy   # Legacy 9-layer orchestrator
  python -m arbo --mode paper    # Alias for rdh (default)
"""

import argparse
import asyncio

from arbo.utils.logger import setup_logging


def main() -> None:
    """CLI entry point with mode routing."""
    parser = argparse.ArgumentParser(description="Arbo Trading System")
    parser.add_argument(
        "--mode",
        default="paper",
        choices=["paper", "rdh", "legacy", "live"],
        help="Orchestrator mode: rdh (3-strategy), legacy (9-layer), paper (=rdh), live",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(log_level=args.log_level)

    if args.mode == "legacy":
        # Legacy 9-layer orchestrator
        from arbo.main import ArboOrchestrator

        orchestrator = ArboOrchestrator(mode="paper")
        asyncio.run(orchestrator.start())
    else:
        # RDH 3-strategy orchestrator (default)
        from arbo.main_rdh import RDHOrchestrator

        effective_mode = "paper" if args.mode in ("rdh", "paper") else args.mode
        orchestrator = RDHOrchestrator(mode=effective_mode)
        asyncio.run(orchestrator.start())


main()
