"""Tests for web dashboard (arbo/dashboard/web.py)."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, ClassVar
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# Set env vars before importing web module
os.environ.setdefault("DASHBOARD_USER", "testuser")
os.environ.setdefault("DASHBOARD_PASSWORD", "testpass")

from arbo.dashboard.web import app, state

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@dataclass
class MockLayerState:
    """Mock LayerState for testing."""

    name: str
    task: Any = None
    restart_count: int = 0
    error_timestamps: list[float] = field(default_factory=list)
    last_heartbeat: float = field(default_factory=time.monotonic)
    enabled: bool = True
    permanent_stop: bool = False


class MockPaperEngine:
    """Mock paper trading engine."""

    balance = Decimal("2050.00")
    total_value = Decimal("2100.00")
    open_positions: ClassVar[list[Any]] = []
    trade_history: ClassVar[list[Any]] = []

    def get_stats(self) -> dict[str, Any]:
        return {
            "initial_capital": "2000.00",
            "current_balance": "2050.00",
            "total_value": "2100.00",
            "total_trades": 5,
            "wins": 3,
            "losses": 2,
        }


@dataclass
class MockStrategyState:
    """Mock StrategyState for testing."""

    allocated: Decimal = Decimal("400")
    deployed: Decimal = Decimal("50")
    weekly_pnl: Decimal = Decimal("12.50")
    total_pnl: Decimal = Decimal("25.00")
    position_count: int = 2
    is_halted: bool = False

    @property
    def available(self) -> Decimal:
        return self.allocated - self.deployed


class MockRiskManager:
    """Mock risk manager."""

    _daily_pnl = Decimal("-50.00")
    _weekly_pnl = Decimal("-100.00")
    _category_exposure: ClassVar[dict[str, Decimal]] = {
        "soccer": Decimal("150.00"),
        "crypto": Decimal("80.00"),
    }
    _shutdown = False

    def __init__(self) -> None:
        self._strategies: dict[str, MockStrategyState] = {
            "A": MockStrategyState(),
            "B": MockStrategyState(allocated=Decimal("400"), deployed=Decimal("0")),
            "C": MockStrategyState(allocated=Decimal("1000"), deployed=Decimal("200")),
        }

    def get_strategy_state(self, strategy: str) -> MockStrategyState | None:
        return self._strategies.get(strategy)


class MockConfig:
    """Mock ArboConfig."""

    class Dashboard:
        port = 8080
        host = "0.0.0.0"

    class Confluence:
        min_score = 1

    dashboard = Dashboard()
    confluence = Confluence()

    dashboard_user = "testuser"
    dashboard_password = "testpass"
    bankroll = 2000.0


class MockOrchestrator:
    """Mock orchestrator for testing dashboard endpoints (RDH architecture)."""

    _mode = "paper"
    _capital = Decimal("2000.00")
    _start_time = time.monotonic() - 3600  # 1 hour ago
    _paper_engine = MockPaperEngine()
    _config = MockConfig()

    def __init__(self) -> None:
        self._risk_manager = MockRiskManager()
        self._tasks = {
            "discovery": MockLayerState("discovery"),
            "strategy_A": MockLayerState("strategy_A"),
            "strategy_B": MockLayerState("strategy_B"),
            "strategy_C": MockLayerState("strategy_C", restart_count=2),
        }


@pytest.fixture()
def mock_orch() -> MockOrchestrator:
    """Create a mock orchestrator and inject into dashboard state."""
    orch = MockOrchestrator()
    state.orchestrator = orch
    yield orch
    state.orchestrator = None


@pytest.fixture()
def client(mock_orch: MockOrchestrator) -> TestClient:
    """Create a test client with auth."""
    return TestClient(app)


AUTH = ("testuser", "testpass")


# ---------------------------------------------------------------------------
# Auth tests
# ---------------------------------------------------------------------------


class TestAuth:
    """Test HTTP Basic Auth."""

    def test_no_auth_returns_401(self) -> None:
        state.orchestrator = MockOrchestrator()
        c = TestClient(app, raise_server_exceptions=False)
        resp = c.get("/")
        assert resp.status_code == 401
        state.orchestrator = None

    def test_wrong_password_returns_401(self) -> None:
        state.orchestrator = MockOrchestrator()
        c = TestClient(app, raise_server_exceptions=False)
        resp = c.get("/", auth=("testuser", "wrong"))
        assert resp.status_code == 401
        state.orchestrator = None

    def test_valid_auth_returns_200(self, client: TestClient) -> None:
        resp = client.get("/", auth=AUTH)
        assert resp.status_code == 200

    def test_no_password_env_returns_503(self) -> None:
        state.orchestrator = MockOrchestrator()
        c = TestClient(app, raise_server_exceptions=False)
        with patch.dict(os.environ, {"DASHBOARD_PASSWORD": ""}):
            resp = c.get("/", auth=AUTH)
            assert resp.status_code == 503
        state.orchestrator = None


# ---------------------------------------------------------------------------
# Dashboard page
# ---------------------------------------------------------------------------


class TestDashboardPage:
    """Test main HTML page."""

    def test_renders_html(self, client: TestClient) -> None:
        resp = client.get("/", auth=AUTH)
        assert resp.status_code == 200
        assert "ARBO" in resp.text
        assert "paper" in resp.text

    def test_contains_chart_js(self, client: TestClient) -> None:
        resp = client.get("/", auth=AUTH)
        assert "chart.js" in resp.text or "Chart" in resp.text

    def test_contains_api_endpoints(self, client: TestClient) -> None:
        resp = client.get("/", auth=AUTH)
        assert "/api/portfolio" in resp.text
        assert "/api/strategies" in resp.text
        assert "/api/layers" in resp.text


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------


class TestPortfolioAPI:
    """Test /api/portfolio endpoint."""

    def test_returns_portfolio_data(self, client: TestClient) -> None:
        with patch("arbo.utils.db.get_session_factory") as mock_factory:
            mock_session = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=False)

            # Mock scalar results for daily/weekly PnL
            mock_result = MagicMock()
            mock_result.scalar.return_value = Decimal("25.00")
            mock_session.execute = AsyncMock(return_value=mock_result)

            # Mock all() for snapshots — return empty
            mock_result.all.return_value = []

            mock_factory.return_value = MagicMock(return_value=mock_session)

            resp = client.get("/api/portfolio", auth=AUTH)
            assert resp.status_code == 200
            data = resp.json()
            assert "balance" in data
            assert "total_value" in data
            assert "daily_pnl" in data
            assert "roi_pct" in data
            assert "snapshots" in data

    def test_portfolio_without_db(self, client: TestClient) -> None:
        """Portfolio endpoint handles DB errors gracefully."""
        resp = client.get("/api/portfolio", auth=AUTH)
        assert resp.status_code == 200
        data = resp.json()
        assert data["balance"] == 2050.0


class TestLayersAPI:
    """Test /api/layers endpoint (RDH tasks)."""

    def test_returns_task_statuses(self, client: TestClient) -> None:
        resp = client.get("/api/layers", auth=AUTH)
        assert resp.status_code == 200
        data = resp.json()
        assert "layers" in data
        layers = data["layers"]
        assert len(layers) == 4

        # Check task structure
        task = layers[0]
        assert "name" in task
        assert "label" in task
        assert "status" in task
        assert "heartbeat_ago_s" in task
        assert "restart_count" in task

    def test_task_status_colors(self, client: TestClient, mock_orch: MockOrchestrator) -> None:
        # strategy_C has restart_count=2 → should be yellow
        resp = client.get("/api/layers", auth=AUTH)
        data = resp.json()
        sc = next(t for t in data["layers"] if t["name"] == "strategy_C")
        assert sc["status"] == "yellow"
        assert sc["restart_count"] == 2

    def test_task_labels(self, client: TestClient) -> None:
        resp = client.get("/api/layers", auth=AUTH)
        data = resp.json()
        labels = {t["name"]: t["label"] for t in data["layers"]}
        assert labels["discovery"] == "Discovery"
        assert labels["strategy_A"] == "Strategy A — Theta Decay"
        assert labels["strategy_B"] == "Strategy B — Reflexivity Surfer"
        assert labels["strategy_C"] == "Strategy C — Compound Weather"


class TestRiskAPI:
    """Test /api/risk endpoint."""

    def test_returns_risk_data(self, client: TestClient) -> None:
        resp = client.get("/api/risk", auth=AUTH)
        assert resp.status_code == 200
        data = resp.json()
        assert "daily_pnl" in data
        assert "weekly_pnl" in data
        assert "daily_utilization_pct" in data
        assert "weekly_utilization_pct" in data
        assert "category_exposure" in data

    def test_daily_utilization_calculation(self, client: TestClient) -> None:
        resp = client.get("/api/risk", auth=AUTH)
        data = resp.json()
        # daily_pnl = -50, limit = 2000 * 0.10 = 200
        # utilization = 50/200 * 100 = 25%
        assert data["daily_utilization_pct"] == 25.0

    def test_category_exposure(self, client: TestClient) -> None:
        resp = client.get("/api/risk", auth=AUTH)
        data = resp.json()
        assert data["category_exposure"]["soccer"] == 150.0
        assert data["category_exposure"]["crypto"] == 80.0


class TestInfraAPI:
    """Test /api/infra endpoint."""

    def test_returns_infra_data(self, client: TestClient) -> None:
        resp = client.get("/api/infra", auth=AUTH)
        assert resp.status_code == 200
        data = resp.json()
        assert "uptime_s" in data
        assert "uptime_human" in data
        assert "memory_mb" in data
        assert "cpu_pct" in data
        assert "active_tasks" in data
        assert data["mode"] == "paper"

    def test_uptime_is_positive(self, client: TestClient) -> None:
        resp = client.get("/api/infra", auth=AUTH)
        data = resp.json()
        assert data["uptime_s"] > 0

    def test_active_tasks_count(self, client: TestClient) -> None:
        resp = client.get("/api/infra", auth=AUTH)
        data = resp.json()
        # 4 tasks, all enabled, none permanently stopped
        assert data["active_tasks"] == 4

    def test_odds_quota_na_without_client(self, client: TestClient) -> None:
        resp = client.get("/api/infra", auth=AUTH)
        data = resp.json()
        assert data["odds_api_quota"]["remaining"] == "N/A"


class TestSignalsAPI:
    """Test /api/signals endpoint."""

    def test_handles_no_db(self, client: TestClient) -> None:
        resp = client.get("/api/signals", auth=AUTH)
        assert resp.status_code == 200
        data = resp.json()
        assert "signals" in data


class TestTradesAPI:
    """Test /api/trades endpoint."""

    def test_handles_no_db(self, client: TestClient) -> None:
        resp = client.get("/api/trades", auth=AUTH)
        assert resp.status_code == 200
        data = resp.json()
        assert "trades" in data


class TestPositionsAPI:
    """Test /api/positions endpoint."""

    def test_handles_no_db(self, client: TestClient) -> None:
        resp = client.get("/api/positions", auth=AUTH)
        assert resp.status_code == 200
        data = resp.json()
        assert "positions" in data


# ---------------------------------------------------------------------------
# Strategy endpoints (RDH)
# ---------------------------------------------------------------------------


class TestStrategiesAPI:
    """Test /api/strategies endpoint."""

    def test_returns_all_strategies(self, client: TestClient) -> None:
        resp = client.get("/api/strategies", auth=AUTH)
        assert resp.status_code == 200
        data = resp.json()
        assert "strategies" in data
        strategies = data["strategies"]
        assert len(strategies) == 3

        ids = {s["id"] for s in strategies}
        assert ids == {"A", "B", "C"}

    def test_strategy_structure(self, client: TestClient) -> None:
        resp = client.get("/api/strategies", auth=AUTH)
        data = resp.json()
        strat_a = next(s for s in data["strategies"] if s["id"] == "A")
        assert strat_a["name"] == "Theta Decay"
        assert strat_a["allocated"] == 400.0
        assert strat_a["deployed"] == 50.0
        assert strat_a["available"] == 350.0
        assert strat_a["positions"] == 2
        assert strat_a["is_halted"] is False

    def test_no_orchestrator(self) -> None:
        state.orchestrator = None
        c = TestClient(app, raise_server_exceptions=False)
        resp = c.get("/api/strategies", auth=AUTH)
        assert resp.status_code == 200
        assert resp.json() == {"strategies": []}


class TestCapitalAPI:
    """Test /api/capital endpoint."""

    def test_returns_capital_allocation(self, client: TestClient) -> None:
        resp = client.get("/api/capital", auth=AUTH)
        assert resp.status_code == 200
        data = resp.json()
        assert "total_capital" in data
        assert "allocations" in data
        assert "reserve" in data
        assert len(data["allocations"]) == 3

    def test_capital_utilization(self, client: TestClient) -> None:
        resp = client.get("/api/capital", auth=AUTH)
        data = resp.json()
        alloc_a = next(a for a in data["allocations"] if a["strategy"] == "A")
        # deployed=50, allocated=400 → 12.5%
        assert alloc_a["utilization_pct"] == 12.5
        assert alloc_a["name"] == "Theta Decay"

    def test_reserve_shown(self, client: TestClient) -> None:
        resp = client.get("/api/capital", auth=AUTH)
        data = resp.json()
        assert data["reserve"] == 200.0


class TestTakerFlowAPI:
    """Test /api/taker-flow endpoint."""

    def test_handles_no_db(self, client: TestClient) -> None:
        resp = client.get("/api/taker-flow", auth=AUTH)
        assert resp.status_code == 200
        data = resp.json()
        assert "snapshots" in data


class TestPortfolioStrategyPnl:
    """Test per-strategy P&L in /api/portfolio."""

    def test_portfolio_includes_strategy_pnl(self, client: TestClient) -> None:
        resp = client.get("/api/portfolio", auth=AUTH)
        assert resp.status_code == 200
        data = resp.json()
        assert "strategy_pnl" in data
        sp = data["strategy_pnl"]
        assert "A" in sp
        assert sp["A"]["name"] == "Theta Decay"
        assert sp["A"]["allocated"] == 400.0
        assert sp["A"]["deployed"] == 50.0
        assert sp["A"]["total_pnl"] == 25.0


# ---------------------------------------------------------------------------
# Helper tests
# ---------------------------------------------------------------------------


class TestHelpers:
    """Test utility functions."""

    def test_format_uptime(self) -> None:
        from arbo.dashboard.web import _format_uptime

        assert _format_uptime(0) == "0m"
        assert _format_uptime(3600) == "1h 0m"
        assert _format_uptime(90061) == "1d 1h 1m"
        assert _format_uptime(300) == "5m"

    def test_dec_conversion(self) -> None:
        from arbo.dashboard.web import _dec

        assert _dec(None) is None
        assert _dec(Decimal("1.23")) == 1.23
        assert _dec(42) == 42.0
        assert _dec(3.14) == 3.14

    def test_create_app_injects_orchestrator(self) -> None:
        from arbo.dashboard.web import create_app, state

        mock = MockOrchestrator()
        result = create_app(mock)
        assert result is app
        assert state.orchestrator is mock
        state.orchestrator = None
