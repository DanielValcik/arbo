"""Strategy B: Reflexivity Surfer — Trade Soros-style reflexive feedback loops.

Monitors markets where social attention (Kaito mindshare/sentiment) diverges
from Polymarket price. Uses a 4-phase state machine per market:

  Phase 1 (Start) — No divergence, monitoring
  Phase 2 (Boom)  — Price trails below actual attention (buy YES)
  Phase 3 (Peak)  — Price overshoots actual attention (sell YES / buy NO)
  Phase 4 (Bust)  — Price reverts (hold NO, wait for resolution)

Entry:
  - Phase 2: divergence < -10% → buy YES (momentum, $10-20)
  - Phase 3: divergence > +20% → buy NO (reversal, $20-50)

Exit:
  - Phase 2: stop loss -15%
  - Phase 3-4: stop loss -25%
  - Partial exit at +30% gain

Uses Kaito API (stub mode) for mindshare/sentiment data.
Falls back to Gemini Flash for LLM-based sentiment estimation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from enum import IntEnum
from typing import Any

from arbo.config.settings import get_config
from arbo.connectors.kaito_api import KaitoClient
from arbo.core.risk_manager import (
    MAX_POSITION_PCT,
    RiskManager,
    TradeRequest,
)
from arbo.utils.logger import get_logger

logger = get_logger("reflexivity_surfer")

STRATEGY_ID = "B"
KELLY_FRACTION = Decimal("0.25")  # Quarter-Kelly


class Phase(IntEnum):
    """Reflexivity phase for a market."""

    START = 1  # No divergence, monitoring
    BOOM = 2  # Price trails actual attention (buy YES)
    PEAK = 3  # Price overshoots actual (buy NO)
    BUST = 4  # Price reverting (hold NO)


@dataclass
class MarketPhase:
    """Tracks the reflexivity phase state for a single market."""

    condition_id: str
    topic: str
    phase: Phase = Phase.START
    last_divergence: float = 0.0
    last_kaito_prob: float = 0.5
    last_market_price: float = 0.5
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    entered_phase_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    transition_count: int = 0


@dataclass
class _ReflexPosition:
    """Internal tracking of an active reflexivity position."""

    condition_id: str
    token_id: str
    side: str  # "BUY_YES" or "BUY_NO"
    phase_at_entry: Phase
    entry_price: Decimal
    size: Decimal
    entered_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    partial_exited: bool = False


class ReflexivitySurfer:
    """Strategy B: Reflexivity Surfer — trade reflexive feedback loops.

    Lifecycle:
    1. Scan markets and fetch Kaito attention data
    2. Compute divergence: (market_price - kaito_prob) / kaito_prob
    3. Transition phase state machine per market
    4. Trade on phase transitions (Phase 2 → buy YES, Phase 3 → buy NO)
    5. Manage exits (stop loss, partial exit)
    """

    def __init__(
        self,
        risk_manager: RiskManager,
        kaito_client: KaitoClient,
        paper_engine: Any = None,
    ) -> None:
        cfg = get_config().reflexivity
        self._risk = risk_manager
        self._kaito = kaito_client
        self._paper_engine = paper_engine

        # Config
        self._boom_threshold = cfg.boom_divergence_threshold  # -0.10
        self._peak_threshold = cfg.peak_divergence_threshold  # +0.20
        self._phase2_max = Decimal(str(cfg.phase2_max_position))  # $20
        self._phase3_max = Decimal(str(cfg.phase3_max_position))  # $50
        self._phase2_stop = cfg.phase2_stop_loss  # 0.15
        self._phase3_stop = cfg.phase3_stop_loss  # 0.25
        self._max_per_phase = cfg.max_concurrent_per_phase  # 5
        self._min_volume = Decimal(str(cfg.min_volume_24h))
        self._min_liquidity = Decimal(str(cfg.min_liquidity))

        # State
        self._phases: dict[str, MarketPhase] = {}
        self._active_positions: dict[str, _ReflexPosition] = {}
        self._signals_generated: int = 0
        self._trades_placed: int = 0
        self._last_scan: datetime | None = None

    async def init(self) -> None:
        """Initialize strategy."""
        logger.info(
            "reflexivity_init",
            boom_threshold=self._boom_threshold,
            peak_threshold=self._peak_threshold,
            kaito_mode="stub" if self._kaito.is_stub else "live",
        )

    async def close(self) -> None:
        """Clean up resources."""
        await self._kaito.close()
        logger.info(
            "reflexivity_close",
            signals=self._signals_generated,
            trades=self._trades_placed,
            tracked_markets=len(self._phases),
        )

    async def poll_cycle(self, markets: list[Any]) -> list[dict[str, Any]]:
        """Run one poll cycle: fetch attention, compute divergence, trade.

        Args:
            markets: List of GammaMarket objects from market discovery.

        Returns:
            List of trade results for markets where trades were placed.
        """
        self._last_scan = datetime.now(UTC)

        # Check strategy allocation
        strategy_state = self._risk.get_strategy_state(STRATEGY_ID)
        if strategy_state is None:
            logger.warning("strategy_b_no_allocation")
            return []
        if strategy_state.is_halted:
            logger.warning("strategy_b_halted")
            return []

        available_capital = strategy_state.available
        if available_capital <= Decimal("0"):
            return []

        # Filter candidate markets
        candidates = self._filter_candidates(markets)
        if not candidates:
            return []

        traded = []
        for mkt in candidates:
            # Already have position in this market
            if mkt.condition_id in self._active_positions:
                continue

            # Fetch Kaito attention data
            topic = mkt.question if hasattr(mkt, "question") else mkt.condition_id
            attention = await self._kaito.get_market_attention(mkt.condition_id, topic)

            # Compute divergence
            market_price = float(mkt.price_yes or Decimal("0.5"))
            kaito_prob = attention.kaito_probability
            if kaito_prob <= 0:
                continue

            divergence = (market_price - kaito_prob) / kaito_prob

            # Update phase state machine
            old_phase = self._get_phase(mkt.condition_id)
            new_phase = self._transition_phase(mkt.condition_id, topic, divergence, kaito_prob, market_price)

            # Count positions per phase
            phase2_count = sum(
                1 for p in self._active_positions.values() if p.phase_at_entry == Phase.BOOM
            )
            phase3_count = sum(
                1 for p in self._active_positions.values() if p.phase_at_entry in (Phase.PEAK, Phase.BUST)
            )

            # Trade on phase transitions
            result = None
            if new_phase == Phase.BOOM and old_phase == Phase.START:
                # Phase 2: buy YES (momentum)
                if phase2_count < self._max_per_phase:
                    result = self._execute_entry(
                        mkt, Phase.BOOM, divergence, available_capital
                    )
            elif new_phase == Phase.PEAK and old_phase in (Phase.START, Phase.BOOM):
                # Phase 3: buy NO (reversal)
                if phase3_count < self._max_per_phase:
                    result = self._execute_entry(
                        mkt, Phase.PEAK, divergence, available_capital
                    )

            if result is not None:
                traded.append(result)
                available_capital -= Decimal(str(result["size"]))
                self._signals_generated += 1

        return traded

    def _filter_candidates(self, markets: list[Any]) -> list[Any]:
        """Filter markets suitable for reflexivity trading.

        Criteria:
        - Active, not closed
        - 24h volume >= min_volume ($5K)
        - Liquidity >= min_liquidity ($2K)
        - Has YES/NO token IDs
        - Has price data
        """
        candidates = []
        for mkt in markets:
            if not mkt.active or mkt.closed:
                continue
            if mkt.volume_24h < self._min_volume:
                continue
            if hasattr(mkt, "liquidity") and mkt.liquidity < self._min_liquidity:
                continue
            if not getattr(mkt, "token_id_yes", None) or not getattr(mkt, "token_id_no", None):
                continue
            if mkt.price_yes is None:
                continue
            candidates.append(mkt)
        return candidates

    def _get_phase(self, condition_id: str) -> Phase:
        """Get current phase for a market."""
        mp = self._phases.get(condition_id)
        return mp.phase if mp else Phase.START

    def _transition_phase(
        self,
        condition_id: str,
        topic: str,
        divergence: float,
        kaito_prob: float,
        market_price: float,
    ) -> Phase:
        """Transition the phase state machine for a market.

        Phase transitions:
          START → BOOM:  divergence < boom_threshold (-10%)
          START → PEAK:  divergence > peak_threshold (+20%)
          BOOM  → PEAK:  divergence > peak_threshold (+20%)
          PEAK  → BUST:  divergence returns below +10% (mean-reverting)
          BUST  → START: divergence returns to [-5%, +5%] (normalized)

        Returns:
            New phase after transition.
        """
        now = datetime.now(UTC)

        if condition_id not in self._phases:
            self._phases[condition_id] = MarketPhase(
                condition_id=condition_id,
                topic=topic,
            )

        mp = self._phases[condition_id]
        old_phase = mp.phase

        # State machine transitions
        if mp.phase == Phase.START:
            if divergence < self._boom_threshold:
                mp.phase = Phase.BOOM
            elif divergence > self._peak_threshold:
                mp.phase = Phase.PEAK

        elif mp.phase == Phase.BOOM:
            if divergence > self._peak_threshold:
                mp.phase = Phase.PEAK

        elif mp.phase == Phase.PEAK:
            if divergence < 0.10:  # Mean reversion starting
                mp.phase = Phase.BUST

        elif mp.phase == Phase.BUST:
            if -0.05 <= divergence <= 0.05:  # Normalized
                mp.phase = Phase.START

        # Update tracking
        if mp.phase != old_phase:
            mp.entered_phase_at = now
            mp.transition_count += 1
            logger.info(
                "reflexivity_phase_transition",
                condition_id=condition_id[:20],
                old_phase=old_phase.name,
                new_phase=mp.phase.name,
                divergence=round(divergence, 4),
            )

        mp.last_divergence = divergence
        mp.last_kaito_prob = kaito_prob
        mp.last_market_price = market_price
        mp.updated_at = now

        return mp.phase

    def _execute_entry(
        self,
        mkt: Any,
        phase: Phase,
        divergence: float,
        available_capital: Decimal,
    ) -> dict[str, Any] | None:
        """Execute entry trade based on phase.

        Phase 2 (BOOM): buy YES at market price
        Phase 3 (PEAK): buy NO at market price
        """
        if phase == Phase.BOOM:
            # Buy YES — price is below where it should be
            side = "BUY"
            token_id = mkt.token_id_yes
            price = mkt.price_yes
            max_pos = self._phase2_max
        elif phase == Phase.PEAK:
            # Buy NO — price is above where it should be
            side = "BUY"
            token_id = mkt.token_id_no
            price = mkt.price_no
            max_pos = self._phase3_max
        else:
            return None

        if price is None or price <= Decimal("0.01") or price >= Decimal("0.99"):
            return None

        # Compute edge from divergence magnitude
        edge = Decimal(str(abs(divergence)))

        # Quarter-Kelly sizing
        odds = float(price)
        if odds <= 0 or odds >= 1:
            return Decimal("0")

        kelly = (float(edge) / (1 - odds)) * float(KELLY_FRACTION)
        raw_size = Decimal(str(max(0, kelly))) * available_capital

        # Clamp to phase-specific limits
        min_pos = Decimal("10")
        size = max(min_pos, min(max_pos, raw_size))
        size = min(size, available_capital)

        # Cap at MAX_POSITION_PCT of total capital
        max_pct_cap = self._risk._state.capital * MAX_POSITION_PCT
        size = min(size, max_pct_cap)
        size = size.quantize(Decimal("0.01"))

        if size < min_pos:
            return None

        # Build trade request for risk check
        trade_req = TradeRequest(
            market_id=mkt.condition_id,
            token_id=token_id,
            side=side,
            price=price,
            size=size,
            layer=0,
            market_category=getattr(mkt, "category", "other"),
            strategy=STRATEGY_ID,
        )

        decision = self._risk.pre_trade_check(trade_req)
        if not decision.approved:
            logger.info(
                "reflexivity_trade_rejected",
                condition_id=mkt.condition_id[:20],
                reason=decision.reason,
            )
            return None

        final_size = decision.adjusted_size if decision.adjusted_size else size

        # Execute via paper engine
        if self._paper_engine is None:
            return None

        model_prob = Decimal(str(max(0.01, min(0.99, 1 - float(price) + float(edge)))))

        trade = self._paper_engine.place_trade(
            market_condition_id=mkt.condition_id,
            token_id=token_id,
            side=side,
            market_price=price,
            model_prob=model_prob,
            layer=0,
            market_category=getattr(mkt, "category", "other"),
            strategy=STRATEGY_ID,
            pre_computed_size=final_size,
        )

        if trade is None:
            return None

        # Post-trade accounting
        self._risk.strategy_post_trade(STRATEGY_ID, final_size)

        # Track position
        side_label = "BUY_YES" if phase == Phase.BOOM else "BUY_NO"
        self._active_positions[mkt.condition_id] = _ReflexPosition(
            condition_id=mkt.condition_id,
            token_id=token_id,
            side=side_label,
            phase_at_entry=phase,
            entry_price=price,
            size=final_size,
        )
        self._trades_placed += 1

        logger.info(
            "reflexivity_entry",
            condition_id=mkt.condition_id[:20],
            phase=phase.name,
            side=side_label,
            price=str(price),
            size=str(final_size),
            divergence=round(divergence, 4),
        )

        return {
            "condition_id": mkt.condition_id,
            "side": side_label,
            "phase": phase.name,
            "price": float(price),
            "size": float(final_size),
            "divergence": round(divergence, 4),
            "strategy": STRATEGY_ID,
        }

    def check_exits(self, current_prices: dict[str, Decimal]) -> list[dict[str, Any]]:
        """Check active positions for stop loss or partial exit.

        Stop loss:
          - Phase 2 (BOOM): -15%
          - Phase 3/4 (PEAK/BUST): -25%
        Partial exit: +30% gain → sell 50%

        Args:
            current_prices: {condition_id: current_price_of_held_token}

        Returns:
            List of exit actions taken.
        """
        exits = []
        to_remove = []

        for cond_id, pos in self._active_positions.items():
            current = current_prices.get(cond_id)
            if current is None:
                continue

            pnl_pct = (current - pos.entry_price) / pos.entry_price

            # Stop loss threshold depends on entry phase
            stop_pct = self._phase2_stop if pos.phase_at_entry == Phase.BOOM else self._phase3_stop

            if pnl_pct <= -Decimal(str(stop_pct)):
                exits.append({
                    "condition_id": cond_id,
                    "action": "stop_loss",
                    "phase": pos.phase_at_entry.name,
                    "entry_price": float(pos.entry_price),
                    "current_price": float(current),
                    "pnl_pct": float(pnl_pct),
                    "size": float(pos.size),
                })
                to_remove.append(cond_id)
                logger.info(
                    "reflexivity_stop_loss",
                    condition_id=cond_id[:20],
                    phase=pos.phase_at_entry.name,
                    pnl_pct=float(pnl_pct),
                )

            elif pnl_pct >= Decimal("0.30") and not pos.partial_exited:
                exit_size = pos.size * Decimal("0.5")
                pos.size -= exit_size
                pos.partial_exited = True
                exits.append({
                    "condition_id": cond_id,
                    "action": "partial_exit",
                    "phase": pos.phase_at_entry.name,
                    "entry_price": float(pos.entry_price),
                    "current_price": float(current),
                    "pnl_pct": float(pnl_pct),
                    "exit_size": float(exit_size),
                    "remaining_size": float(pos.size),
                })
                logger.info(
                    "reflexivity_partial_exit",
                    condition_id=cond_id[:20],
                    pnl_pct=float(pnl_pct),
                )

        for cond_id in to_remove:
            del self._active_positions[cond_id]

        return exits

    def handle_resolution(self, condition_id: str, pnl: Decimal) -> None:
        """Handle market resolution for a reflexivity position."""
        if condition_id in self._active_positions:
            del self._active_positions[condition_id]
        if condition_id in self._phases:
            del self._phases[condition_id]

        self._risk.strategy_post_trade(STRATEGY_ID, abs(pnl), pnl=pnl)
        logger.info(
            "reflexivity_resolution",
            condition_id=condition_id[:20],
            pnl=str(pnl),
        )

    @property
    def stats(self) -> dict[str, Any]:
        """Current strategy statistics."""
        strategy_state = self._risk.get_strategy_state(STRATEGY_ID)
        phase_counts = {}
        for mp in self._phases.values():
            name = mp.phase.name
            phase_counts[name] = phase_counts.get(name, 0) + 1

        return {
            "strategy": STRATEGY_ID,
            "signals_generated": self._signals_generated,
            "trades_placed": self._trades_placed,
            "active_positions": len(self._active_positions),
            "tracked_markets": len(self._phases),
            "phase_distribution": phase_counts,
            "kaito_mode": "stub" if self._kaito.is_stub else "live",
            "last_scan": self._last_scan.isoformat() if self._last_scan else None,
            "deployed": str(strategy_state.deployed) if strategy_state else "0",
            "available": str(strategy_state.available) if strategy_state else "0",
            "is_halted": strategy_state.is_halted if strategy_state else False,
        }
