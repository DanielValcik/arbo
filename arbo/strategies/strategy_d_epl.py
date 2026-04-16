"""Strategy D — EPL Variant (v2 with Dixon-Coles model).

EPL + cup competitions green booking. Strategy ID = "D_EPL".

v2 improvements (Dixon-Coles 1997):
  - Low-score correlation correction (ρ = -0.10 for EPL)
  - Back out λ (home expected goals), μ (away expected goals) from Pinnacle
  - Simulate 8x8 score grid with τ(x,y) correction for draws
  - Result: CLV 0.6¢ → 3.15¢ (+425%), Sharpe 6.76 → 8.81 (+30%)

Sweep v2 winner #18 (240 experiments, 100% profitable):
  - Score: 11.1, +$252 on $1K over 20 months
  - 96 trades, WR 67%, DD 4%, Sharpe 9.05
  - **CLV +4.06¢** — 2x research benchmark of 2¢ for long-term profit

Sweep patterns:
  - min_edge=0.02 optimal (DC has enough signal for small edges)
  - delta=0.20 sweet spot across all mhf values
  - SL=0.30 safer than tighter SL
  - mhf=1.0 dominant (hold to resolution) — DC + Pinnacle very accurate

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

    # Sweep v2 winner #18 params (Dixon-Coles sweep, CLV +4.06¢)
    MIN_EDGE = 0.02          # Very low — DC gives us high-quality signal
    MAX_EDGE = 0.30
    MIN_PRICE = 0.15
    MAX_PRICE = 0.70

    GREEN_BOOK_DELTA = 0.20  # Up from 0.15 — sweep optimal
    STOP_LOSS_DELTA = 0.30   # Up from 0.25 — safer SL in volatile markets
    MAX_HOLD_FRACTION = 1.0  # Hold to resolution — DC + Pinnacle very accurate
    GAME_DURATION_HOURS = 2.0

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

    def _dc_probs(self, team_a: str, team_b: str) -> tuple[float, float, float] | None:
        """Return (p_home, p_draw, p_away) Dixon-Coles probs or None."""
        pin_prob = None
        for key_teams in [(team_a, team_b), (team_b, team_a)]:
            pin_key = f"{self.SPORT_NAME}_{key_teams[0]}_{key_teams[1]}"
            if pin_key in self._pinnacle:
                hp, ap = self._pinnacle[pin_key]
                pin_prob = (hp, ap) if key_teams[0] == team_a else (ap, hp)
                break
        if pin_prob is None:
            return None
        p_h_pin, p_a_pin = pin_prob
        return dixon_coles_probs(p_h_pin, p_a_pin)

    def compute_model_prob(self, team_a: str, team_b: str,
                           outcome_type: str = "moneyline") -> float | None:
        """Probability for the YES outcome given market type.

        - moneyline → P(team_a wins)
        - draw → P(match ends in a draw)
        """
        dc = self._dc_probs(team_a, team_b)
        if dc is None:
            return super().compute_model_prob(team_a, team_b)
        p_home, p_draw, _p_away = dc

        if outcome_type == "draw":
            # Draw markets are not Elo-blended (Elo doesn't model draws directly)
            return p_draw

        # Moneyline: blend Dixon-Coles home with Elo
        elo_a = self._elo.get(team_a)
        elo_b = self._elo.get(team_b)
        if elo_a and elo_b:
            diff = elo_a[0] - elo_b[0]
            elo_p = 1.0 / (1.0 + 10.0 ** (-diff / 400.0))
            return self.ELO_WEIGHT * elo_p + self.PINNACLE_WEIGHT * p_home
        return p_home
