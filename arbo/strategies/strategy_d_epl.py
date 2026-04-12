"""Strategy D — EPL Variant (v2 with Dixon-Coles model).

EPL + cup competitions green booking. Strategy ID = "D_EPL".

v2 improvements (Dixon-Coles 1997):
  - Low-score correlation correction (ρ = -0.10 for EPL)
  - Back out λ (home expected goals), μ (away expected goals) from Pinnacle
  - Simulate 8x8 score grid with τ(x,y) correction for draws
  - Result: CLV 0.6¢ → 3.15¢ (+425%), Sharpe 6.76 → 8.81 (+30%)

Latest backtest (v2 Dixon-Coles, single_team trades only):
  - Score: 5.3, +$162 on $1K over 20 months
  - 82 trades, WR 65%, DD 2.4%, Sharpe 8.81
  - CLV +3.15¢ — MUCH better edge per trade than v1

Key characteristics:
  - 3-way outcomes: home_win / draw / away_win
  - Polymarket splits into binary markets per outcome
  - Question formats: "Will X beat Y?", "Will X win on DATE?",
    "Will X vs Y end in a draw?"
  - 1,940 Pinnacle odds from football-data.co.uk (free)
  - Includes EPL + FA Cup + League Cup + UCL + Community Shield
  - Pinnacle coverage limits us to ~280 unique EPL fixtures

Architecture: docs/STRATEGY_D_ARCHITECTURE.md
"""

from __future__ import annotations

import math

from arbo.strategies.strategy_d_core import StrategyDCore


# ── Dixon-Coles helpers (EPL-specific) ────────────────────────────────

_DC_RHO = -0.10      # EPL low-score correlation
_DC_MAX_GOALS = 8


def _poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0 or k < 0:
        return 0.0
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def _dc_tau(x: int, y: int, lam: float, mu: float, rho: float) -> float:
    if x == 0 and y == 0: return 1 - lam * mu * rho
    if x == 0 and y == 1: return 1 + lam * rho
    if x == 1 and y == 0: return 1 + mu * rho
    if x == 1 and y == 1: return 1 - rho
    return 1.0


def dixon_coles_probs(p_home_pin: float, p_away_pin: float,
                      rho: float = _DC_RHO) -> tuple[float, float, float]:
    """Given Pinnacle no-vig 2-way probs, return (p_home, p_draw, p_away)
    using Dixon-Coles model with EPL-typical parameters."""
    # Back out λ, μ via iterative search
    total_goals = 2.6
    ratio = p_home_pin / max(p_away_pin, 0.01)
    lam = total_goals * 0.55 * math.sqrt(ratio)
    mu = total_goals * 0.45 / math.sqrt(ratio)

    for _ in range(15):
        p_h = p_d = p_a = 0.0
        for x in range(_DC_MAX_GOALS + 1):
            pxl = _poisson_pmf(x, lam)
            for y in range(_DC_MAX_GOALS + 1):
                pym = _poisson_pmf(y, mu)
                t = _dc_tau(x, y, lam, mu, rho)
                p = t * pxl * pym
                if x > y: p_h += p
                elif x < y: p_a += p
                else: p_d += p
        total = p_h + p_d + p_a
        if total > 0:
            p_h /= total; p_d /= total; p_a /= total

        # Adjust for 2-way mismatch
        err_h = p_h - p_home_pin * (p_h + p_a)
        err_a = p_a - p_away_pin * (p_h + p_a)
        lam *= 1 - 0.3 * err_h
        mu *= 1 - 0.3 * err_a
        lam = max(0.1, min(4.0, lam))
        mu = max(0.1, min(4.0, mu))
        if abs(err_h) < 0.005:
            break

    return p_h, p_d, p_a


class StrategyDEpl(StrategyDCore):
    """EPL green book engine — Dixon-Coles model (v2)."""

    SPORT_NAME = "epl"
    STRATEGY_NAME = "D_EPL"
    STRATEGY_LABEL = "EPL Green Book"

    # Sweep winner #12 params
    MIN_EDGE = 0.03          # Lower edge — EPL has small but consistent edges
    MAX_EDGE = 0.30
    MIN_PRICE = 0.15
    MAX_PRICE = 0.70

    GREEN_BOOK_DELTA = 0.15  # Smaller than UFC (0.20) — EPL has gradual moves
    STOP_LOSS_DELTA = 0.25   # Wide SL (robust across 288 experiments)
    MAX_HOLD_FRACTION = 1.0  # Hold to resolution — EPL Pinnacle very accurate
    GAME_DURATION_HOURS = 2.0  # 90min match + stoppage + pre-game window

    BOTH_SIDES = True
    MAX_CONCURRENT = 6        # Multiple matches per gameweek
    COOLDOWN_AFTER_TRADE_S = 60

    # Conservative sizing (small P&L in backtest)
    KELLY_FRACTION = 0.12
    KELLY_RAW_CAP = 0.10
    MAX_POSITION_PCT = 0.03

    # Pinnacle-weighted (EPL Pinnacle very reliable)
    ELO_WEIGHT = 0.20
    PINNACLE_WEIGHT = 0.80

    RISK_LAYER = 9

    # ── Dixon-Coles probability override ──────────────────────────────

    def compute_model_prob(self, team_a: str, team_b: str) -> float | None:
        """Use Dixon-Coles model for EPL probabilities.

        Returns P(team_a wins) for 2-way moneyline markets.
        For draw-specific markets, the caller can override outcome.
        """
        # Get raw Pinnacle 2-way
        pin_prob = None
        for key_teams in [(team_a, team_b), (team_b, team_a)]:
            pin_key = f"{self.SPORT_NAME}_{key_teams[0]}_{key_teams[1]}"
            if pin_key in self._pinnacle:
                hp, ap = self._pinnacle[pin_key]
                pin_prob = (hp, ap) if key_teams[0] == team_a else (ap, hp)
                break

        if pin_prob is None:
            # No Pinnacle — fall back to base class (Elo/Glicko if available)
            return super().compute_model_prob(team_a, team_b)

        # Apply Dixon-Coles correction
        p_h_pin, p_a_pin = pin_prob
        p_home, _p_draw, _p_away = dixon_coles_probs(p_h_pin, p_a_pin)

        # Blend with Elo if available (small weight since DC is sport-specific)
        elo_a = self._elo.get(team_a)
        elo_b = self._elo.get(team_b)
        if elo_a and elo_b:
            diff = elo_a[0] - elo_b[0]
            elo_p = 1.0 / (1.0 + 10.0 ** (-diff / 400.0))
            return self.ELO_WEIGHT * elo_p + self.PINNACLE_WEIGHT * p_home
        return p_home
