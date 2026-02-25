"""Tests for resolution chaining engine (Strategy C)."""

from __future__ import annotations

from decimal import Decimal

import pytest

from arbo.connectors.weather_models import City
from arbo.strategies.resolution_chain import (
    CITY_CHAIN_ORDER,
    ChainStatus,
    ResolutionChainEngine,
    ResolutionChainState,
)


@pytest.fixture
def engine() -> ResolutionChainEngine:
    return ResolutionChainEngine()


class TestChainStart:
    """Test starting new chains."""

    def test_start_chain_default_order(self, engine: ResolutionChainEngine) -> None:
        chain = engine.start_chain(Decimal("200"))
        assert chain.chain_id.startswith("chain_")
        assert chain.initial_capital == Decimal("200")
        assert chain.current_capital == Decimal("200")
        assert chain.status == ChainStatus.ACTIVE
        assert chain.city_sequence == CITY_CHAIN_ORDER
        assert chain.current_city == CITY_CHAIN_ORDER[0]

    def test_start_chain_custom_order(self, engine: ResolutionChainEngine) -> None:
        custom = [City.NYC, City.LONDON, City.SEOUL]
        chain = engine.start_chain(Decimal("100"), city_sequence=custom)
        assert chain.city_sequence == custom
        assert chain.current_city == City.NYC

    def test_chain_stored_in_engine(self, engine: ResolutionChainEngine) -> None:
        chain = engine.start_chain(Decimal("100"))
        retrieved = engine.get_chain(chain.chain_id)
        assert retrieved is chain


class TestChainDeployment:
    """Test market deployment tracking."""

    def test_set_active_market(self, engine: ResolutionChainEngine) -> None:
        chain = engine.start_chain(Decimal("200"))
        result = engine.set_active_market(chain.chain_id, "weather_nyc_123")
        assert result is True
        assert chain.active_market_id == "weather_nyc_123"
        assert chain.status == ChainStatus.WAITING_RESOLUTION

    def test_set_active_market_unknown_chain(self, engine: ResolutionChainEngine) -> None:
        result = engine.set_active_market("nonexistent", "mkt_1")
        assert result is False


class TestChainResolution:
    """Test market resolution and capital advancement."""

    def test_resolution_advances_city(self, engine: ResolutionChainEngine) -> None:
        chain = engine.start_chain(Decimal("200"), city_sequence=[City.NYC, City.CHICAGO, City.LONDON])
        engine.set_active_market(chain.chain_id, "nyc_mkt")

        next_city = engine.resolve(chain.chain_id, "nyc_mkt", pnl=Decimal("20"))
        assert next_city == City.CHICAGO
        assert chain.current_city == City.CHICAGO
        assert chain.current_capital == Decimal("220")
        assert chain.cumulative_pnl == Decimal("20")
        assert chain.num_resolutions == 1

    def test_resolution_records_history(self, engine: ResolutionChainEngine) -> None:
        chain = engine.start_chain(Decimal("100"), city_sequence=[City.NYC, City.CHICAGO])
        engine.set_active_market(chain.chain_id, "nyc_mkt")
        engine.resolve(chain.chain_id, "nyc_mkt", pnl=Decimal("15"))

        assert len(chain.resolutions) == 1
        res = chain.resolutions[0]
        assert res.city == City.NYC
        assert res.pnl == Decimal("15")
        assert res.capital_before == Decimal("100")
        assert res.capital_after == Decimal("115")

    def test_chain_completes_after_all_cities(self, engine: ResolutionChainEngine) -> None:
        chain = engine.start_chain(Decimal("100"), city_sequence=[City.NYC, City.CHICAGO])
        engine.set_active_market(chain.chain_id, "nyc_mkt")
        engine.resolve(chain.chain_id, "nyc_mkt", pnl=Decimal("10"))

        engine.set_active_market(chain.chain_id, "chi_mkt")
        next_city = engine.resolve(chain.chain_id, "chi_mkt", pnl=Decimal("5"))

        assert next_city is None
        assert chain.status == ChainStatus.COMPLETED
        assert chain.is_complete
        assert chain.cumulative_pnl == Decimal("15")
        assert chain.completed_at is not None

    def test_negative_pnl_reduces_capital(self, engine: ResolutionChainEngine) -> None:
        chain = engine.start_chain(Decimal("100"), city_sequence=[City.NYC, City.CHICAGO])
        engine.set_active_market(chain.chain_id, "nyc_mkt")
        engine.resolve(chain.chain_id, "nyc_mkt", pnl=Decimal("-30"))

        assert chain.current_capital == Decimal("70")
        assert chain.cumulative_pnl == Decimal("-30")

    def test_chain_halts_on_zero_capital(self, engine: ResolutionChainEngine) -> None:
        chain = engine.start_chain(Decimal("50"), city_sequence=[City.NYC, City.CHICAGO])
        engine.set_active_market(chain.chain_id, "nyc_mkt")
        next_city = engine.resolve(chain.chain_id, "nyc_mkt", pnl=Decimal("-50"))

        assert next_city is None
        assert chain.status == ChainStatus.HALTED

    def test_unknown_chain_raises(self, engine: ResolutionChainEngine) -> None:
        with pytest.raises(ValueError, match="not found"):
            engine.resolve("nonexistent", "mkt_1", pnl=Decimal("10"))


class TestChainROI:
    """Test ROI calculation."""

    def test_positive_roi(self, engine: ResolutionChainEngine) -> None:
        chain = engine.start_chain(Decimal("100"), city_sequence=[City.NYC])
        engine.set_active_market(chain.chain_id, "nyc_mkt")
        engine.resolve(chain.chain_id, "nyc_mkt", pnl=Decimal("20"))
        assert chain.roi == Decimal("0.2")  # 20%

    def test_negative_roi(self, engine: ResolutionChainEngine) -> None:
        chain = engine.start_chain(Decimal("100"), city_sequence=[City.NYC])
        engine.set_active_market(chain.chain_id, "nyc_mkt")
        engine.resolve(chain.chain_id, "nyc_mkt", pnl=Decimal("-15"))
        assert chain.roi == Decimal("-0.15")  # -15%


class TestChainManagement:
    """Test chain listing and management."""

    def test_get_active_chains(self, engine: ResolutionChainEngine) -> None:
        c1 = engine.start_chain(Decimal("100"), city_sequence=[City.NYC])
        c2 = engine.start_chain(Decimal("100"), city_sequence=[City.CHICAGO])
        # Complete c1
        engine.set_active_market(c1.chain_id, "nyc_mkt")
        engine.resolve(c1.chain_id, "nyc_mkt", pnl=Decimal("10"))

        active = engine.get_active_chains()
        assert len(active) == 1
        assert active[0].chain_id == c2.chain_id

    def test_get_all_chains(self, engine: ResolutionChainEngine) -> None:
        engine.start_chain(Decimal("100"))
        engine.start_chain(Decimal("200"))
        assert len(engine.get_all_chains()) == 2

    def test_halt_chain(self, engine: ResolutionChainEngine) -> None:
        chain = engine.start_chain(Decimal("100"))
        result = engine.halt_chain(chain.chain_id, "kill switch")
        assert result is True
        assert chain.status == ChainStatus.HALTED

    def test_halt_unknown_chain(self, engine: ResolutionChainEngine) -> None:
        result = engine.halt_chain("nonexistent", "test")
        assert result is False


class TestChainDBSerialization:
    """Test DB serialization."""

    def test_to_db_dict(self, engine: ResolutionChainEngine) -> None:
        chain = engine.start_chain(Decimal("200"), city_sequence=[City.NYC, City.LONDON])
        db_dict = chain.to_db_dict()
        assert db_dict["chain_id"] == chain.chain_id
        assert db_dict["city_sequence"] == ["nyc", "london"]
        assert db_dict["initial_capital"] == Decimal("200")
        assert db_dict["status"] == "active"


class TestFullChainCycle:
    """Integration test: full chain lifecycle."""

    def test_full_5_city_chain(self, engine: ResolutionChainEngine) -> None:
        """Simulate a complete 5-city chain with mixed results."""
        chain = engine.start_chain(Decimal("200"))
        pnls = [Decimal("10"), Decimal("-5"), Decimal("15"), Decimal("-3"), Decimal("8")]

        for i, city in enumerate(CITY_CHAIN_ORDER):
            assert chain.current_city == city
            engine.set_active_market(chain.chain_id, f"mkt_{city.value}")
            next_city = engine.resolve(chain.chain_id, f"mkt_{city.value}", pnl=pnls[i])

            if i < len(CITY_CHAIN_ORDER) - 1:
                assert next_city == CITY_CHAIN_ORDER[i + 1]
            else:
                assert next_city is None

        assert chain.status == ChainStatus.COMPLETED
        assert chain.num_resolutions == 5
        assert chain.cumulative_pnl == sum(pnls)
        assert chain.current_capital == Decimal("200") + sum(pnls)
        assert len(chain.resolutions) == 5
