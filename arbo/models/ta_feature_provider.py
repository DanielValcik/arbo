"""Background Technical Analysis feature cache for B3/B2 strategies.

Fetches BTC technical indicators (RSI, ADX, MACD, Bollinger Bands) from
TradingView via the tradingview-ta library every 60 seconds. Strategies
read from in-memory cache with zero latency.

tradingview-ta is sync (blocking HTTP) — all calls run in
loop.run_in_executor() to avoid blocking the asyncio event loop.

Graceful degradation: if tradingview-ta is not installed or the endpoint
fails, all features return None and strategies continue without TA.

See: docs/TA_INTEGRATION_SPEC.md
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

from arbo.utils.logger import get_logger

logger = get_logger("ta_feature_provider")

# Max consecutive failures before disabling updates.
_MAX_CONSECUTIVE_FAILURES = 10


@dataclass
class TAFeatures:
    """Cached technical analysis features for a single symbol."""

    timestamp: float

    # 5-minute timeframe (primary for B3)
    rsi_5m: float | None = None
    adx_5m: float | None = None
    macd_hist_5m: float | None = None  # macd - signal
    bb_width_5m: float | None = None  # (upper - lower) / middle
    atr_5m: float | None = None
    recommend_5m: str | None = None  # "STRONG_BUY"|"BUY"|"NEUTRAL"|"SELL"|"STRONG_SELL"

    # 1-hour timeframe (context)
    rsi_1h: float | None = None
    adx_1h: float | None = None
    recommend_1h: str | None = None

    # 4-hour timeframe (macro)
    rsi_4h: float | None = None
    adx_4h: float | None = None
    recommend_4h: str | None = None

    # Daily timeframe (for B2 integration)
    rsi_1d: float | None = None
    macd_hist_1d: float | None = None
    recommend_1d: str | None = None

    @property
    def multi_tf_aligned(self) -> bool:
        """True if 5m, 1h, 4h all agree on direction."""
        recs = [self.recommend_5m, self.recommend_1h, self.recommend_4h]
        if any(r is None for r in recs):
            return False
        buy_signals = {"STRONG_BUY", "BUY"}
        sell_signals = {"STRONG_SELL", "SELL"}
        all_buy = all(r in buy_signals for r in recs)
        all_sell = all(r in sell_signals for r in recs)
        return all_buy or all_sell

    @property
    def adx_regime(self) -> str:
        """ADX-based regime classification."""
        if self.adx_5m is None:
            return "UNKNOWN"
        if self.adx_5m < 15:
            return "RANGING"
        if self.adx_5m < 25:
            return "WEAK_TREND"
        return "STRONG_TREND"

    @property
    def rsi_zone(self) -> str:
        """RSI zone for mean-reversion risk assessment."""
        if self.rsi_5m is None:
            return "UNKNOWN"
        if self.rsi_5m > 70:
            return "OVERBOUGHT"
        if self.rsi_5m < 30:
            return "OVERSOLD"
        return "NEUTRAL"

    @property
    def is_stale(self) -> bool:
        """Cache older than 90s is considered stale."""
        return time.time() - self.timestamp > 90


class TAFeatureProvider:
    """Background TA feature cache.

    Usage::

        provider = TAFeatureProvider()
        asyncio.create_task(provider.start())

        # In strategy (zero latency):
        ta = provider.get("BTCUSDT")
        if ta and not ta.is_stale:
            rsi = ta.rsi_5m
    """

    def __init__(self, update_interval: float = 300.0) -> None:
        self._cache: dict[str, TAFeatures] = {}
        self._update_interval = update_interval
        self._running = False
        self._consecutive_failures = 0
        self._ta_available: bool | None = None  # None = not checked yet

    async def start(self) -> None:
        """Background update loop. Run as asyncio.create_task()."""
        self._running = True
        # Check if tradingview-ta is importable at all
        if not self._check_import():
            logger.warning(
                "ta_provider_disabled",
                reason="tradingview_ta not installed — TA features unavailable",
            )
            return

        logger.info("ta_provider_started", interval_s=self._update_interval)

        while self._running:
            try:
                loop = asyncio.get_event_loop()
                features = await loop.run_in_executor(None, self._fetch_btc_ta)
                if features:
                    self._cache["BTCUSDT"] = features
                    self._consecutive_failures = 0
                    logger.debug(
                        "ta_update_ok",
                        rsi_5m=f"{features.rsi_5m:.1f}" if features.rsi_5m else "N/A",
                        adx_5m=f"{features.adx_5m:.1f}" if features.adx_5m else "N/A",
                        regime=features.adx_regime,
                        aligned=features.multi_tf_aligned,
                    )
            except Exception as e:
                self._consecutive_failures += 1
                logger.warning(
                    "ta_update_error",
                    error=str(e),
                    failures=self._consecutive_failures,
                )
                if self._consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
                    logger.error(
                        "ta_provider_too_many_failures",
                        msg=f"{_MAX_CONSECUTIVE_FAILURES} consecutive failures — stopping TA updates",
                    )
                    break

            # Sleep in 1-second chunks for responsive shutdown
            for _ in range(int(self._update_interval)):
                if not self._running:
                    return
                await asyncio.sleep(1)

    def _check_import(self) -> bool:
        """Check if tradingview-ta is installed."""
        if self._ta_available is not None:
            return self._ta_available
        try:
            import tradingview_ta  # noqa: F401

            self._ta_available = True
        except ImportError:
            self._ta_available = False
        return self._ta_available

    def _fetch_btc_ta(self) -> TAFeatures | None:
        """Fetch BTC TA across 4 timeframes (sync, runs in executor)."""
        from tradingview_ta import Interval, TA_Handler

        now = time.time()
        result = TAFeatures(timestamp=now)

        # 5-minute (primary for B3)
        try:
            h5 = TA_Handler(
                symbol="BTCUSDT",
                screener="crypto",
                exchange="BINANCE",
                interval=Interval.INTERVAL_5_MINUTES,
            )
            a5 = h5.get_analysis()
            ind = a5.indicators
            result.rsi_5m = ind.get("RSI")
            result.adx_5m = ind.get("ADX")

            macd_val = ind.get("MACD.macd")
            signal_val = ind.get("MACD.signal")
            if macd_val is not None and signal_val is not None:
                result.macd_hist_5m = macd_val - signal_val

            bb_upper = ind.get("BB.upper")
            bb_lower = ind.get("BB.lower")
            bb_middle = ind.get("BB.middle")
            if bb_upper and bb_lower and bb_middle and bb_middle > 0:
                result.bb_width_5m = (bb_upper - bb_lower) / bb_middle

            result.atr_5m = ind.get("ATR")
            result.recommend_5m = a5.summary.get("RECOMMENDATION")
        except Exception as e:
            logger.warning("ta_fetch_5m_error", error=str(e))

        # Delay between requests to avoid TradingView 429 rate limit
        time.sleep(5)

        # 1-hour (context)
        try:
            h1h = TA_Handler(
                symbol="BTCUSDT",
                screener="crypto",
                exchange="BINANCE",
                interval=Interval.INTERVAL_1_HOUR,
            )
            a1h = h1h.get_analysis()
            result.rsi_1h = a1h.indicators.get("RSI")
            result.adx_1h = a1h.indicators.get("ADX")
            result.recommend_1h = a1h.summary.get("RECOMMENDATION")
        except Exception as e:
            logger.warning("ta_fetch_1h_error", error=str(e))

        time.sleep(5)

        # 4-hour (macro)
        try:
            h4h = TA_Handler(
                symbol="BTCUSDT",
                screener="crypto",
                exchange="BINANCE",
                interval=Interval.INTERVAL_4_HOURS,
            )
            a4h = h4h.get_analysis()
            result.rsi_4h = a4h.indicators.get("RSI")
            result.adx_4h = a4h.indicators.get("ADX")
            result.recommend_4h = a4h.summary.get("RECOMMENDATION")
        except Exception as e:
            logger.warning("ta_fetch_4h_error", error=str(e))

        time.sleep(5)

        # Daily (for B2)
        try:
            h1d = TA_Handler(
                symbol="BTCUSDT",
                screener="crypto",
                exchange="BINANCE",
                interval=Interval.INTERVAL_1_DAY,
            )
            a1d = h1d.get_analysis()
            ind_d = a1d.indicators
            result.rsi_1d = ind_d.get("RSI")

            macd_d = ind_d.get("MACD.macd")
            signal_d = ind_d.get("MACD.signal")
            if macd_d is not None and signal_d is not None:
                result.macd_hist_1d = macd_d - signal_d

            result.recommend_1d = a1d.summary.get("RECOMMENDATION")
        except Exception as e:
            logger.warning("ta_fetch_1d_error", error=str(e))

        # Return if we got at least 5-min data
        if result.rsi_5m is not None:
            return result
        return None

    def get(self, symbol: str = "BTCUSDT") -> TAFeatures | None:
        """Read cached TA features (zero latency).

        Returns None if cache is empty or stale (>90s).
        Strategies should handle None gracefully.
        """
        features = self._cache.get(symbol)
        if features and not features.is_stale:
            return features
        return None

    async def stop(self) -> None:
        """Signal the background loop to stop."""
        self._running = False

    def get_status(self) -> dict:
        """Status for health monitoring / Slack reports."""
        btc = self._cache.get("BTCUSDT")
        return {
            "running": self._running,
            "ta_available": self._ta_available,
            "consecutive_failures": self._consecutive_failures,
            "btc_cached": btc is not None,
            "btc_stale": btc.is_stale if btc else True,
            "btc_age_s": round(time.time() - btc.timestamp, 0) if btc else None,
            "btc_adx_regime": btc.adx_regime if btc else None,
            "btc_rsi_zone": btc.rsi_zone if btc else None,
        }
