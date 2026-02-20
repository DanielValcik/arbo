"""Shared test fixtures for Arbo."""

from __future__ import annotations

import os
from unittest.mock import AsyncMock

import pytest

# Set test environment before any imports
os.environ.setdefault("MODE", "paper")
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://arbo:password@localhost:5432/arbo_test")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/1")
os.environ.setdefault("MATCHBOOK_USERNAME", "test_user")
os.environ.setdefault("MATCHBOOK_PASSWORD", "test_pass")
os.environ.setdefault("LOG_LEVEL", "DEBUG")


@pytest.fixture
def mock_redis() -> AsyncMock:
    """Mock Redis client."""
    redis = AsyncMock()
    redis.get.return_value = None
    redis.set.return_value = True
    redis.ping.return_value = True
    return redis


@pytest.fixture
def matchbook_session_response() -> dict:
    """Mock Matchbook login response."""
    return {
        "session-token": "test-session-token-12345",
        "user-id": 12345,
        "role": "TRADER",
        "account": {"balance": 2000.00, "currency": "EUR"},
    }


@pytest.fixture
def matchbook_events_response() -> dict:
    """Mock Matchbook events response."""
    return {
        "events": [
            {
                "id": 100001,
                "name": "Liverpool vs Arsenal",
                "start": "2026-03-01T15:00:00.000Z",
                "status": "open",
                "category-name": "EPL",
                "markets": [
                    {
                        "id": 200001,
                        "name": "Match Odds",
                        "market-type": "one_x_two",
                        "runners": [
                            {
                                "id": 300001,
                                "name": "Liverpool",
                                "prices": [
                                    {"odds": 2.10, "available-amount": 500.0, "side": "back"},
                                    {"odds": 2.14, "available-amount": 300.0, "side": "lay"},
                                ],
                            },
                            {
                                "id": 300002,
                                "name": "Draw",
                                "prices": [
                                    {"odds": 3.40, "available-amount": 200.0, "side": "back"},
                                    {"odds": 3.50, "available-amount": 150.0, "side": "lay"},
                                ],
                            },
                            {
                                "id": 300003,
                                "name": "Arsenal",
                                "prices": [
                                    {"odds": 3.60, "available-amount": 400.0, "side": "back"},
                                    {"odds": 3.70, "available-amount": 250.0, "side": "lay"},
                                ],
                            },
                        ],
                    }
                ],
            },
            {
                "id": 100002,
                "name": "Man City vs Chelsea",
                "start": "2026-03-01T17:30:00.000Z",
                "status": "open",
                "category-name": "EPL",
                "markets": [],
            },
        ]
    }


@pytest.fixture
def matchbook_offer_response() -> dict:
    """Mock Matchbook offer placement response."""
    return {
        "offers": [
            {
                "id": 999001,
                "status": "matched",
                "odds": 2.10,
                "stake": 50.0,
                "matched-amount": 50.0,
            }
        ]
    }
