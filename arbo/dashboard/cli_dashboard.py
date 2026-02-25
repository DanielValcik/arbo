"""CLI dashboard for terminal monitoring (PM-210).

Plain text dashboard for real-time system status monitoring.
Displays layer status, portfolio summary, and risk state.

No external TUI library — uses str formatting only.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from arbo.utils.logger import get_logger

logger = get_logger("cli_dashboard")

# Layer names by number
LAYER_NAMES: dict[int, str] = {
    1: "Market Making",
    2: "Value Betting",
    3: "Semantic Graph",
    4: "Whale Copy",
    5: "Logical Arb",
    6: "Temporal Crypto",
    7: "Order Flow",
    8: "Attention Markets",
    9: "Sports Latency",
}

# Strategy names (RDH architecture)
STRATEGY_NAMES: dict[str, str] = {
    "A": "Theta Decay",
    "B": "Reflexivity Surfer",
    "C": "Compound Weather",
}


@dataclass
class LayerStatus:
    """Status of a single strategy layer."""

    layer: int
    name: str
    active: bool
    signals_count: int = 0
    last_signal_at: datetime | None = None
    error: str | None = None


@dataclass
class StrategyStatus:
    """Status of a single RDH strategy."""

    strategy_id: str
    name: str
    allocated: Decimal = Decimal("0")
    deployed: Decimal = Decimal("0")
    available: Decimal = Decimal("0")
    positions: int = 0
    weekly_pnl: Decimal = Decimal("0")
    is_halted: bool = False


class CLIDashboard:
    """Terminal dashboard for real-time system monitoring.

    Renders plain text tables showing:
    - Layer status (active/inactive, signal counts, errors)
    - Portfolio summary (balance, P&L, open positions)
    - Risk state (daily/weekly loss, exposure)
    """

    SEPARATOR = "=" * 60

    def render(
        self,
        layers: list[LayerStatus],
        portfolio_stats: dict[str, Any] | None = None,
        confluence_stats: dict[str, Any] | None = None,
        risk_state: dict[str, Any] | None = None,
        strategies: list[StrategyStatus] | None = None,
    ) -> str:
        """Render full dashboard as a string.

        Args:
            layers: Status of all 9 layers.
            portfolio_stats: Stats from PaperTradingEngine.get_stats().
            confluence_stats: Stats from ConfluenceScorer.stats.
            risk_state: Risk manager state dict.
            strategies: Per-strategy status (RDH architecture).

        Returns:
            Formatted dashboard string for terminal display.
        """
        sections: list[str] = []

        header = f"\n{self.SEPARATOR}\n  ARBO — Polymarket Trading System\n  {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}\n{self.SEPARATOR}"
        sections.append(header)

        if strategies:
            sections.append(self.format_strategy_table(strategies))

        sections.append(self.format_layer_table(layers))

        if portfolio_stats:
            sections.append(self.format_portfolio_summary(portfolio_stats))

        if risk_state:
            sections.append(self.format_risk_summary(risk_state))

        if confluence_stats:
            sections.append(self._format_confluence(confluence_stats))

        sections.append(self.SEPARATOR)
        return "\n".join(sections)

    def format_layer_table(self, layers: list[LayerStatus]) -> str:
        """Format layer status as a table.

        Args:
            layers: List of LayerStatus objects.

        Returns:
            Formatted table string.
        """
        lines = ["\n  LAYERS", "  " + "-" * 56]
        header = f"  {'#'.ljust(4)}{'Name'.ljust(20)}{'Status'.ljust(10)}{'Signals'.ljust(10)}{'Error'.ljust(14)}"
        lines.append(header)
        lines.append("  " + "-" * 56)

        for ls in layers:
            status = "ACTIVE" if ls.active else "OFF"
            error = ls.error[:12] if ls.error else "-"
            line = f"  {str(ls.layer).ljust(4)}{ls.name.ljust(20)}{status.ljust(10)}{str(ls.signals_count).ljust(10)}{error.ljust(14)}"
            lines.append(line)

        return "\n".join(lines)

    def format_strategy_table(self, strategies: list[StrategyStatus]) -> str:
        """Format per-strategy status as a table.

        Args:
            strategies: List of StrategyStatus objects.

        Returns:
            Formatted table string.
        """
        lines = ["\n  STRATEGIES", "  " + "-" * 56]
        header = f"  {'ID'.ljust(4)}{'Name'.ljust(22)}{'Deploy'.ljust(10)}{'Avail'.ljust(10)}{'Pos'.ljust(5)}{'Status'.ljust(8)}"
        lines.append(header)
        lines.append("  " + "-" * 56)

        for ss in strategies:
            status = "HALTED" if ss.is_halted else "ACTIVE"
            pnl_sign = "+" if ss.weekly_pnl >= 0 else ""
            lines.append(
                f"  {ss.strategy_id.ljust(4)}"
                f"{ss.name.ljust(22)}"
                f"${str(ss.deployed).ljust(9)}"
                f"${str(ss.available).ljust(9)}"
                f"{str(ss.positions).ljust(5)}"
                f"{status.ljust(8)}"
            )

        return "\n".join(lines)

    def format_portfolio_summary(self, stats: dict[str, Any]) -> str:
        """Format portfolio summary.

        Args:
            stats: Dict from PaperTradingEngine.get_stats().

        Returns:
            Formatted summary string.
        """
        lines = ["\n  PORTFOLIO", "  " + "-" * 56]

        balance = stats.get("current_balance", Decimal("0"))
        total_value = stats.get("total_value", Decimal("0"))
        total_pnl = stats.get("total_pnl", Decimal("0"))
        roi = stats.get("roi_pct", 0)
        win_rate = stats.get("win_rate", 0)
        open_trades = stats.get("open_trades", 0)
        resolved = stats.get("resolved_trades", 0)

        lines.append(f"  Balance:       ${balance:>10}")
        lines.append(f"  Total Value:   ${total_value:>10}")
        lines.append(f"  Total P&L:     ${total_pnl:>10}")
        lines.append(f"  ROI:           {roi:>10.2f}%")
        lines.append(f"  Win Rate:      {win_rate:>10.1%}")
        lines.append(f"  Open Trades:   {open_trades:>10}")
        lines.append(f"  Resolved:      {resolved:>10}")

        return "\n".join(lines)

    def format_risk_summary(self, state: dict[str, Any]) -> str:
        """Format risk manager state.

        Args:
            state: Risk state dict with daily_pnl, weekly_pnl, etc.

        Returns:
            Formatted summary string.
        """
        lines = ["\n  RISK", "  " + "-" * 56]

        daily_pnl = state.get("daily_pnl", Decimal("0"))
        weekly_pnl = state.get("weekly_pnl", Decimal("0"))
        shutdown = state.get("shutdown", False)

        lines.append(f"  Daily P&L:     ${daily_pnl:>10}")
        lines.append(f"  Weekly P&L:    ${weekly_pnl:>10}")
        lines.append(f"  Shutdown:      {'YES' if shutdown else 'NO':>10}")

        return "\n".join(lines)

    def _format_confluence(self, stats: dict[str, Any]) -> str:
        """Format confluence scorer stats."""
        lines = ["\n  CONFLUENCE", "  " + "-" * 56]

        lines.append(f"  Scored:        {stats.get('total_scored', 0):>10}")
        lines.append(f"  Tradeable:     {stats.get('total_tradeable', 0):>10}")
        lines.append(f"  Rejected:      {stats.get('total_rejected', 0):>10}")

        return "\n".join(lines)
