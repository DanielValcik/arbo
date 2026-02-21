"""Shared test fixtures for Arbo test suite."""

from __future__ import annotations

from decimal import Decimal

import pytest


@pytest.fixture
def sample_capital() -> Decimal:
    """Default capital for testing."""
    return Decimal("2000.00")


@pytest.fixture
def sample_token_id() -> str:
    """Sample Polymarket token ID for testing."""
    return "71321045679252212594626385532706912750332728571942532289631379312455583992563"


@pytest.fixture
def sample_condition_id() -> str:
    """Sample market condition ID for testing."""
    return "0xe3b423dfad8c22ff1b8c312345678901234567890abcdef"
