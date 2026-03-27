"""Strategy B2 — Crypto Price Edge.

Trades Polymarket crypto price prediction markets using real-time
exchange prices (Binance) and a volatility-based probability model.

Key differences from C2 (weather):
- NOT NegRisk: standard binary markets with real orderbooks
- Exchange price source: Binance WebSocket (not weather forecast)
- Model: Log-normal volatility CDF (not EMOS)
- Higher liquidity: $1-5M per bucket (vs $100-500)
- Faster cycle: 60s poll, 30s exit check (vs 300s/60s)
- Fee model: daily markets have crypto fees, monthly = 0%
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from arbo.connectors.binance_ws import BinanceWSFeed
from arbo.core.fee_model import calculate_taker_fee
from arbo.models.volatility_model import VolatilityEstimator, estimate_crypto_prob
from arbo.strategies.crypto_price_scanner import (
    CryptoSignal,
    compute_hours_to_expiry,
    scan_crypto_markets,
)
from arbo.strategies.crypto_quality_gate import (
    KELLY_RAW_CAP,
    MIN_HOLD_EDGE,
    PROFIT_TARGET_ABS,
    VOLATILITY_METHOD,
    VOLATILITY_WINDOW,
    SIGMA_SCALE,
    filter_signals,
)
from arbo.utils.logger import get_logger

logger = get_logger("strategy_b2")

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

STRATEGY_NAME = "B2"
KELLY_FRACTION = Decimal("0.25")  # Quarter-Kelly
MAX_TRADES_PER_SCAN = 3
MIN_HOURS_BETWEEN_TRADES = 0.5  # 30 min cooldown per token
LATENCY_RECHECK_DELAY_S = 2
VOLATILITY_GUARD_MAX_MOVE = 0.05  # Skip if price moved >5c during recheck
VOLUME_CAP_PCT = 0.05  # Max 5% of 24h volume per trade


@dataclass
class OpenPosition:
    """Tracked open B2 position for exit monitoring."""

    token_id: str
    condition_id: str
    asset: str
    symbol: str
    strike: float
    direction: str
    market_type: str
    entry_price: float  # Polymarket price at entry
    entry_exchange_price: float  # Binance price at entry
    sigma_at_entry: float
    hours_to_expiry_at_entry: float
    entry_time: float
    shares: int = 0


class StrategyB2:
    """Crypto Price Edge strategy orchestrator."""

    def __init__(
        self,
        risk_manager: Any,
        paper_engine: Any | None = None,
        binance_ws: BinanceWSFeed | None = None,
        orderbook_provider: Any | None = None,
        execution_mode: str = "paper",
        live_executor: Any | None = None,
    ) -> None:
        self._risk_manager = risk_manager
        self._paper_engine = paper_engine
        self._binance_ws = binance_ws
        self._orderbook_provider = orderbook_provider
        self._execution_mode = execution_mode
        self._live_executor = live_executor
        self._exit_manager_ref: Any | None = None

        # Volatility estimator (maintains rolling price buffer)
        self._vol_estimator = VolatilityEstimator(
            window=VOLATILITY_WINDOW,
            method=VOLATILITY_METHOD,
        )

        # Open position tracking
        self._open_positions: dict[str, OpenPosition] = {}
        self._last_trade_time: dict[str, float] = {}

    async def init(self) -> None:
        """Initialize strategy — restore state from paper engine."""
        if self._paper_engine:
            for tid, pos in self._paper_engine.open_positions.items():
                if pos.strategy == STRATEGY_NAME:
                    self._open_positions[tid] = OpenPosition(
                        token_id=tid,
                        condition_id="",
                        asset="BTC",  # Will be refined on next scan
                        symbol="BTCUSDT",
                        strike=0,
                        direction="above",
                        market_type="daily_above",
                        entry_price=float(pos.avg_price),
                        entry_exchange_price=0,
                        sigma_at_entry=0.02,
                        hours_to_expiry_at_entry=24,
                        entry_time=time.time(),
                        shares=int(pos.shares),
                    )

        n_restored = len(self._open_positions)
        if n_restored:
            logger.info("b2_positions_restored", count=n_restored)

    async def close(self) -> None:
        """Clean up resources."""
        pass  # Binance WS lifecycle managed by main_rdh

    # ═══════════════════════════════════════════════════════════════════════
    # ENTRY: Poll Cycle
    # ═══════════════════════════════════════════════════════════════════════

    async def poll_cycle(self, markets: list[Any]) -> list[CryptoSignal]:
        """Main entry scan. Called every 60 seconds.

        1. Get exchange prices from Binance WS
        2. Compute volatility
        3. Scan markets → signals
        4. Quality gate filter
        5. Fetch CLOB orderbook (non-NegRisk)
        6. Kelly sizing + risk check
        7. Execute (paper or live)

        Returns:
            List of executed signals.
        """
        executed: list[CryptoSignal] = []

        # 1. Get exchange prices
        exchange_prices = {}
        if self._binance_ws:
            exchange_prices = self._binance_ws.get_prices()
            # Update vol estimator with latest prices
            for symbol, price in exchange_prices.items():
                self._vol_estimator.update(symbol, price, time.time())

        if not exchange_prices:
            logger.debug("b2_no_exchange_prices")
            return []

        # 2. Scan markets
        signals = scan_crypto_markets(
            markets,
            exchange_prices,
            self._vol_estimator,
            current_time=time.time(),
        )

        if not signals:
            return []

        # 3. Quality gate
        qualified = filter_signals(signals)
        if not qualified:
            return []

        # 4. Process top signals
        now = time.time()
        for sig in qualified[:MAX_TRADES_PER_SCAN]:
            token_id = sig.token_id

            # Skip if already have position
            if token_id in self._open_positions:
                continue

            # Skip if in pending exit
            if self._exit_manager_ref and token_id in self._exit_manager_ref._pending:
                continue

            # Cooldown check
            last_trade = self._last_trade_time.get(token_id, 0)
            if now - last_trade < MIN_HOURS_BETWEEN_TRADES * 3600:
                continue

            # 5. Fetch CLOB orderbook (non-NegRisk: real orderbook)
            if self._orderbook_provider:
                try:
                    ob_snap = await self._orderbook_provider.get_snapshot(
                        token_id, neg_risk=False
                    )
                    if ob_snap and ob_snap.best_ask is not None:
                        clob_price = float(ob_snap.best_ask)
                    else:
                        continue
                except Exception:
                    continue
            else:
                clob_price = sig.market_price

            # 6. Revalidate edge with CLOB price
            clob_edge = sig.model_prob - clob_price
            if abs(clob_edge) < 0.01:  # Edge gone at CLOB price
                continue

            # 7. Kelly sizing
            prob = sig.model_prob
            price = clob_price
            if prob <= 0 or prob >= 1 or price <= 0 or price >= 1:
                continue

            odds = (1.0 / price) - 1.0
            if odds <= 0:
                continue

            kelly_raw = (prob * odds - (1 - prob)) / odds
            if kelly_raw <= 0:
                continue
            kelly_raw = min(kelly_raw, KELLY_RAW_CAP)
            kelly_adj = kelly_raw * float(KELLY_FRACTION)

            # Get available capital from risk manager
            strat_state = self._risk_manager.get_strategy_state(STRATEGY_NAME)
            if strat_state is None or strat_state.is_halted:
                continue
            available = float(strat_state.allocated - strat_state.deployed)
            if available <= 0:
                continue

            trade_size = round(available * kelly_adj, 2)
            trade_size = min(trade_size, float(strat_state.allocated) * 0.10)  # Max 10% of allocation

            # Volume cap
            if sig.volume_24h > 0:
                trade_size = min(trade_size, sig.volume_24h * VOLUME_CAP_PCT)

            if trade_size < 1.0:
                continue

            # 8. Risk check
            from arbo.core.risk_manager import TradeRequest

            trade_req = TradeRequest(
                market_id=sig.condition_id,
                token_id=token_id,
                side="BUY",
                price=Decimal(str(clob_price)),
                size=Decimal(str(trade_size)),
                layer=0,
                market_category="crypto",
                strategy=STRATEGY_NAME,
            )
            decision = self._risk_manager.pre_trade_check(trade_req)
            if not decision.approved:
                logger.debug("b2_risk_rejected", reason=decision.reason)
                continue

            actual_size = float(decision.adjusted_size or trade_size)

            # 9. Execute
            if self._execution_mode == "live" and self._live_executor:
                fill = await self._live_executor.buy(
                    token_id=token_id,
                    price=clob_price,
                    size_usdc=actual_size,
                    neg_risk=False,  # NOT NegRisk!
                    tick_size="0.01",
                )
                if fill.status in ("filled", "partial") and fill.shares_filled > 0:
                    shares = fill.shares_filled
                    fill_price = float(fill.fill_price) if fill.fill_price else clob_price
                else:
                    continue
            else:
                shares = int(actual_size / clob_price)
                fill_price = clob_price

            # 10. Record trade
            if self._paper_engine:
                self._paper_engine.place_trade(
                    market_condition_id=sig.condition_id,
                    token_id=token_id,
                    side="BUY",
                    market_price=clob_price,
                    model_prob=sig.model_prob,
                    layer=0,
                    market_category="crypto",
                    fee_enabled=sig.fee_enabled,
                    strategy=STRATEGY_NAME,
                    pre_computed_size=actual_size,
                    clob_fill_price=fill_price,
                    trade_details={
                        "asset": sig.asset,
                        "strike": float(sig.strike),
                        "direction": sig.direction,
                        "exchange_price": sig.current_exchange_price,
                        "sigma": sig.sigma_hourly,
                        "hours_to_expiry": sig.hours_to_expiry,
                        "market_type": sig.market_type,
                    },
                )

            # Track position
            self._open_positions[token_id] = OpenPosition(
                token_id=token_id,
                condition_id=sig.condition_id,
                asset=sig.asset,
                symbol=sig.symbol,
                strike=float(sig.strike),
                direction=sig.direction,
                market_type=sig.market_type,
                entry_price=fill_price,
                entry_exchange_price=sig.current_exchange_price,
                sigma_at_entry=sig.sigma_hourly,
                hours_to_expiry_at_entry=sig.hours_to_expiry,
                entry_time=time.time(),
                shares=shares,
            )
            self._last_trade_time[token_id] = time.time()

            logger.info(
                "b2_entry",
                asset=sig.asset,
                strike=float(sig.strike),
                direction=sig.direction,
                edge=f"{sig.edge:.3f}",
                price=f"{fill_price:.3f}",
                size=f"${actual_size:.2f}",
                exchange_price=f"${sig.current_exchange_price:.0f}",
                mode=self._execution_mode,
            )

            executed.append(sig)
            self._risk_manager.record_trade(trade_req)

        return executed

    # ═══════════════════════════════════════════════════════════════════════
    # EXIT: Check Exits
    # ═══════════════════════════════════════════════════════════════════════

    async def check_exits(
        self, current_polymarket_prices: dict[str, float],
    ) -> list[tuple[str, str]]:
        """Check open positions for exit triggers.

        Called every 30 seconds. Recomputes probability with latest
        Binance exchange price.

        Args:
            current_polymarket_prices: token_id → current Polymarket YES price.

        Returns:
            List of (token_id, exit_reason) for triggered exits.
        """
        triggered: list[tuple[str, str]] = []

        # Get latest exchange prices
        exchange_prices = {}
        if self._binance_ws:
            exchange_prices = self._binance_ws.get_prices()

        for token_id, pos in list(self._open_positions.items()):
            current_poly_price = current_polymarket_prices.get(token_id)
            if current_poly_price is None:
                continue

            # 1. Profit take
            if current_poly_price >= pos.entry_price + PROFIT_TARGET_ABS:
                triggered.append((token_id, "profit_take"))
                continue

            # 2. Edge-based exit using latest exchange price
            exchange_price = exchange_prices.get(pos.symbol)
            if exchange_price is not None:
                # Recompute probability with latest exchange price
                # Time has passed since entry → hours_to_expiry decreased
                elapsed_hours = (time.time() - pos.entry_time) / 3600
                updated_hours = max(
                    pos.hours_to_expiry_at_entry - elapsed_hours, 0.01
                )

                sigma = self._vol_estimator.get_sigma(
                    pos.symbol, time.time()
                )

                updated_prob = estimate_crypto_prob(
                    current_price=exchange_price,
                    strike=pos.strike,
                    hours_to_expiry=updated_hours,
                    sigma_per_hour=sigma,
                    market_type=pos.market_type,
                    sigma_scale=SIGMA_SCALE,
                    direction=pos.direction,
                )

                updated_edge = updated_prob - current_poly_price

                if updated_edge < MIN_HOLD_EDGE:
                    triggered.append((token_id, "edge_lost"))
                    continue

        # Clean up triggered positions
        for tid, reason in triggered:
            self._open_positions.pop(tid, None)
            logger.info(
                "b2_exit_triggered",
                token=tid[:20],
                reason=reason,
            )

        return triggered

    def handle_resolution(self, condition_id: str, pnl: float) -> None:
        """Clean up position after market resolves."""
        to_remove = [
            tid for tid, pos in self._open_positions.items()
            if pos.condition_id == condition_id
        ]
        for tid in to_remove:
            self._open_positions.pop(tid, None)
