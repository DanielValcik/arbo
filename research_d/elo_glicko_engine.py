"""Strategy D — Elo/Glicko-2 Team Strength Rating Engine.

Computes Elo and Glicko-2 ratings from historical game results.
These ratings form 40% of Strategy D's probability model
(the other 60% comes from Pinnacle's no-vig line).

Two rating systems are computed in parallel:

1. **Elo with Margin-of-Victory (MOV)**:
   - Standard Elo with K-factor adjustment
   - MOV multiplier: log(|margin| + 1) × autocorrelation_correction
   - Home advantage: +50 Elo points (~3.5% probability boost)
   - Sport-specific K-factors and home advantages

2. **Glicko-2** (Glickman 2012):
   - Rating + Rating Deviation (RD) + Volatility (σ)
   - RD increases during inactivity (captures uncertainty)
   - Volatility adapts to inconsistent performance
   - More accurate for teams with few games or variable form

Usage:
    from research_d.elo_glicko_engine import EloGlickoEngine
    from research_d.sports_db import SportsDB

    db = SportsDB()
    engine = EloGlickoEngine(sport="nba")

    # Process all games chronologically
    games = db.get_games(sport="nba", status="final")
    for game in games:
        engine.process_game(
            home_team=game["home_team"],
            away_team=game["away_team"],
            home_score=game["home_score"],
            away_score=game["away_score"],
            game_date=game["game_date"],
        )

    # Get current ratings
    ratings = engine.get_all_ratings()

    # Predict win probability
    p_home = engine.predict(home_team="LAL", away_team="BOS")

    # Save to database
    engine.save_to_db(db)

References:
    - Elo, A. (1978). "The Rating System in Chess."
    - Glickman, M. (2012). "Example of the Glicko-2 System."
      http://www.glicko.net/glicko/glicko2.pdf
    - Silver, N. (FiveThirtyEight). "How Our NBA/NFL/Soccer Predictions Work."
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

# ── Constants ──────────────────────────────────────────────────────────

# Sport-specific parameters
SPORT_PARAMS = {
    "nba": {
        "elo_k": 20,             # K-factor (learning rate)
        "elo_home_adv": 100,     # Home court advantage in Elo points
        "elo_initial": 1500,
        "elo_mov_enabled": True, # Enable margin-of-victory adjustment
        "elo_season_revert": 0.25,  # Revert 25% to mean between seasons
        "glicko_tau": 0.5,       # System volatility constant
        "glicko_initial_rd": 200,
        "glicko_initial_vol": 0.06,
        "glicko_initial_rating": 1500,
    },
    "epl": {
        "elo_k": 25,             # Slightly higher K for soccer (fewer games)
        "elo_home_adv": 65,      # Home advantage stronger in soccer
        "elo_initial": 1500,
        "elo_mov_enabled": True,
        "elo_season_revert": 0.20,  # Soccer teams more stable
        "glicko_tau": 0.6,       # Higher tau for soccer (more volatile)
        "glicko_initial_rd": 250,
        "glicko_initial_vol": 0.06,
        "glicko_initial_rating": 1500,
    },
    "nfl": {
        "elo_k": 20,
        "elo_home_adv": 48,      # NFL home advantage (~2.5 points)
        "elo_initial": 1500,
        "elo_mov_enabled": True,
        "elo_season_revert": 0.33,  # NFL teams change more between seasons
        "glicko_tau": 0.5,
        "glicko_initial_rd": 300,   # Higher uncertainty (fewer games)
        "glicko_initial_vol": 0.06,
        "glicko_initial_rating": 1500,
    },
}


# ── Elo System ─────────────────────────────────────────────────────────


@dataclass
class EloRating:
    """A team's Elo rating state."""
    rating: float = 1500.0
    games_played: int = 0
    last_date: str = ""


def elo_expected(rating_a: float, rating_b: float) -> float:
    """Calculate expected score for team A against team B.

    Uses the logistic curve:
        E(A) = 1 / (1 + 10^((R_B - R_A) / 400))

    Args:
        rating_a: Elo rating of team A.
        rating_b: Elo rating of team B.

    Returns:
        Expected score for team A (0 to 1).
    """
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def elo_mov_multiplier(margin: int, elo_diff: float) -> float:
    """Margin-of-victory multiplier for Elo updates.

    Adapts the FiveThirtyEight approach:
        mov_mult = log(|margin| + 1) × (2.2 / (elo_diff × 0.001 + 2.2))

    The autocorrelation correction (2.2 / ...) ensures that a team
    that was already expected to win big doesn't get as much credit.

    Args:
        margin: Absolute point/goal margin (always positive).
        elo_diff: Winner's Elo minus loser's Elo before the game.

    Returns:
        Multiplier (typically 0.5 to 3.0).
    """
    return math.log(abs(margin) + 1) * (2.2 / (elo_diff * 0.001 + 2.2))


# ── Glicko-2 System ───────────────────────────────────────────────────


@dataclass
class Glicko2Rating:
    """A team's Glicko-2 rating state.

    Attributes:
        mu: Rating on Glicko-2 internal scale (default 0).
        phi: Rating deviation (uncertainty).
        sigma: Rating volatility (expected fluctuation).
        rating: Rating on traditional scale (1500 ± ~700).
        rd: RD on traditional scale.
    """
    mu: float = 0.0        # Internal scale rating
    phi: float = 2.0148     # Internal scale RD (350.0 / 173.7178)
    sigma: float = 0.06     # Volatility
    games_played: int = 0
    last_date: str = ""

    @property
    def rating(self) -> float:
        """Convert internal mu to traditional Elo-like scale."""
        return self.mu * 173.7178 + 1500.0

    @property
    def rd(self) -> float:
        """Convert internal phi to traditional RD scale."""
        return self.phi * 173.7178

    @classmethod
    def from_traditional(
        cls,
        rating: float = 1500.0,
        rd: float = 350.0,
        vol: float = 0.06,
    ) -> Glicko2Rating:
        """Create from traditional-scale values."""
        return cls(
            mu=(rating - 1500.0) / 173.7178,
            phi=rd / 173.7178,
            sigma=vol,
        )


def _g(phi: float) -> float:
    """Glicko-2 g function: reduces impact of opponents with high RD."""
    return 1.0 / math.sqrt(1.0 + 3.0 * phi ** 2 / (math.pi ** 2))


def _E(mu: float, mu_j: float, phi_j: float) -> float:
    """Glicko-2 E function: expected score against opponent j."""
    return 1.0 / (1.0 + math.exp(-_g(phi_j) * (mu - mu_j)))


def _compute_variance(mu: float, opponents: list[tuple[float, float]]) -> float:
    """Compute estimated variance of team's rating.

    Args:
        mu: Team's current mu.
        opponents: List of (mu_j, phi_j) for each opponent.

    Returns:
        Estimated variance v.
    """
    total = 0.0
    for mu_j, phi_j in opponents:
        g_val = _g(phi_j)
        e_val = _E(mu, mu_j, phi_j)
        total += g_val ** 2 * e_val * (1.0 - e_val)
    return 1.0 / total if total > 0 else 1e6


def _compute_delta(
    mu: float,
    opponents: list[tuple[float, float]],
    scores: list[float],
    v: float,
) -> float:
    """Compute the estimated improvement in rating.

    Args:
        mu: Team's current mu.
        opponents: List of (mu_j, phi_j).
        scores: Actual scores (1.0=win, 0.5=draw, 0.0=loss).
        v: Estimated variance.

    Returns:
        Delta value.
    """
    total = 0.0
    for (mu_j, phi_j), s in zip(opponents, scores):
        g_val = _g(phi_j)
        e_val = _E(mu, mu_j, phi_j)
        total += g_val * (s - e_val)
    return v * total


def _new_volatility(
    sigma: float, phi: float, v: float, delta: float, tau: float
) -> float:
    """Compute new volatility using the Illinois algorithm.

    This is Step 5 of the Glicko-2 algorithm (Glickman 2012).
    Iteratively solves for the new volatility σ'.

    Args:
        sigma: Current volatility.
        phi: Current phi (internal RD).
        v: Estimated variance.
        delta: Estimated improvement.
        tau: System constant (controls volatility change speed).

    Returns:
        New volatility σ'.
    """
    a = math.log(sigma ** 2)
    epsilon = 0.000001

    def f(x: float) -> float:
        ex = math.exp(x)
        d2 = delta ** 2
        p2 = phi ** 2
        num1 = ex * (d2 - p2 - v - ex)
        den1 = 2.0 * (p2 + v + ex) ** 2
        return num1 / den1 - (x - a) / (tau ** 2)

    # Initial values for Illinois algorithm
    A = a
    if delta ** 2 > phi ** 2 + v:
        B = math.log(delta ** 2 - phi ** 2 - v)
    else:
        k = 1
        while f(a - k * tau) < 0:
            k += 1
        B = a - k * tau

    f_A = f(A)
    f_B = f(B)

    # Iterative search
    for _ in range(100):  # Max iterations (usually converges in <10)
        if abs(B - A) < epsilon:
            break
        C = A + (A - B) * f_A / (f_B - f_A)
        f_C = f(C)
        if f_C * f_B <= 0:
            A = B
            f_A = f_B
        else:
            f_A = f_A / 2.0
        B = C
        f_B = f_C

    return math.exp(A / 2.0)


def glicko2_update(
    player: Glicko2Rating,
    opponents: list[tuple[float, float]],
    scores: list[float],
    tau: float = 0.5,
) -> Glicko2Rating:
    """Update a player's Glicko-2 rating after a rating period.

    Implements the full Glicko-2 algorithm from Glickman (2012).

    Args:
        player: Current player rating.
        opponents: List of (mu_j, phi_j) for each opponent faced.
        scores: Actual scores (1.0=win, 0.5=draw, 0.0=loss).
        tau: System constant.

    Returns:
        Updated Glicko2Rating.
    """
    if not opponents:
        # No games: increase RD toward initial uncertainty
        new_phi = math.sqrt(player.phi ** 2 + player.sigma ** 2)
        return Glicko2Rating(
            mu=player.mu,
            phi=new_phi,
            sigma=player.sigma,
            games_played=player.games_played,
            last_date=player.last_date,
        )

    # Step 3: Compute variance v
    v = _compute_variance(player.mu, opponents)

    # Step 4: Compute delta
    delta = _compute_delta(player.mu, opponents, scores, v)

    # Step 5: New volatility
    new_sigma = _new_volatility(player.sigma, player.phi, v, delta, tau)

    # Step 6: Update phi to pre-rating period value
    phi_star = math.sqrt(player.phi ** 2 + new_sigma ** 2)

    # Step 7: Update phi and mu
    new_phi = 1.0 / math.sqrt(1.0 / (phi_star ** 2) + 1.0 / v)
    new_mu = player.mu + new_phi ** 2 * sum(
        _g(phi_j) * (s - _E(player.mu, mu_j, phi_j))
        for (mu_j, phi_j), s in zip(opponents, scores)
    )

    return Glicko2Rating(
        mu=new_mu,
        phi=new_phi,
        sigma=new_sigma,
        games_played=player.games_played + len(scores),
        last_date=player.last_date,
    )


# ── Combined Engine ───────────────────────────────────────────────────


@dataclass
class TeamState:
    """Combined Elo + Glicko-2 state for one team."""
    elo: EloRating = field(default_factory=EloRating)
    glicko: Glicko2Rating = field(default_factory=Glicko2Rating)


class EloGlickoEngine:
    """Computes and maintains Elo + Glicko-2 ratings for all teams in a sport.

    Processes games chronologically, updating both rating systems
    after each game. Stores rating snapshots at configurable intervals
    (default: after every game).

    Attributes:
        sport: Sport identifier ("nba", "epl", "nfl").
        teams: Dict mapping team abbreviation to TeamState.
        params: Sport-specific parameters from SPORT_PARAMS.
        history: List of (date, team, elo, glicko_rating, glicko_rd, glicko_vol).
    """

    def __init__(self, sport: str):
        if sport not in SPORT_PARAMS:
            raise ValueError(
                f"Unknown sport '{sport}'. Supported: {list(SPORT_PARAMS.keys())}"
            )
        self.sport = sport
        self.params = SPORT_PARAMS[sport]
        self.teams: dict[str, TeamState] = {}
        self.history: list[dict[str, Any]] = []
        self._current_season: str | None = None
        self._games_processed: int = 0

    def _get_or_create_team(self, team: str) -> TeamState:
        """Get a team's state, creating with defaults if new."""
        if team not in self.teams:
            p = self.params
            self.teams[team] = TeamState(
                elo=EloRating(rating=p["elo_initial"]),
                glicko=Glicko2Rating.from_traditional(
                    rating=p["glicko_initial_rating"],
                    rd=p["glicko_initial_rd"],
                    vol=p["glicko_initial_vol"],
                ),
            )
        return self.teams[team]

    def process_game(
        self,
        home_team: str,
        away_team: str,
        home_score: int,
        away_score: int,
        game_date: str,
        season: str | None = None,
    ) -> dict[str, float]:
        """Process a single game result and update ratings.

        Args:
            home_team: Home team abbreviation.
            away_team: Away team abbreviation.
            home_score: Home team's final score.
            away_score: Away team's final score.
            game_date: Game date as YYYY-MM-DD.
            season: Optional season identifier for between-season reversion.

        Returns:
            Dict with pre-game predictions and rating changes:
            {
                "elo_home_pre": float, "elo_away_pre": float,
                "elo_expected_home": float,
                "elo_home_change": float, "elo_away_change": float,
                "glicko_home_pre": float, "glicko_away_pre": float,
                "glicko_expected_home": float,
            }
        """
        # Handle season transition
        if season and season != self._current_season and self._current_season is not None:
            self._apply_season_reversion()
        if season:
            self._current_season = season

        home = self._get_or_create_team(home_team)
        away = self._get_or_create_team(away_team)

        # ── Pre-game state ──
        elo_home_pre = home.elo.rating
        elo_away_pre = away.elo.rating

        # ── Elo Update ──
        p = self.params
        # Apply home advantage
        elo_home_adj = home.elo.rating + p["elo_home_adv"]
        expected_home = elo_expected(elo_home_adj, away.elo.rating)

        # Actual score (1.0=win, 0.5=draw, 0.0=loss)
        if home_score > away_score:
            actual_home = 1.0
        elif home_score < away_score:
            actual_home = 0.0
        else:
            actual_home = 0.5  # Draw (soccer)

        # K-factor with optional MOV adjustment
        k = p["elo_k"]
        if p["elo_mov_enabled"] and home_score != away_score:
            margin = abs(home_score - away_score)
            winner_elo = elo_home_adj if actual_home == 1.0 else away.elo.rating
            loser_elo = away.elo.rating if actual_home == 1.0 else elo_home_adj
            mov_mult = elo_mov_multiplier(margin, winner_elo - loser_elo)
            k = k * mov_mult

        elo_change = k * (actual_home - expected_home)
        home.elo.rating += elo_change
        away.elo.rating -= elo_change
        home.elo.games_played += 1
        away.elo.games_played += 1
        home.elo.last_date = game_date
        away.elo.last_date = game_date

        # ── Glicko-2 Update ──
        # For game-by-game updating, each game is a separate rating period
        glicko_expected_home = _E(home.glicko.mu, away.glicko.mu, away.glicko.phi)

        home.glicko = glicko2_update(
            player=home.glicko,
            opponents=[(away.glicko.mu, away.glicko.phi)],
            scores=[actual_home],
            tau=p["glicko_tau"],
        )
        home.glicko.last_date = game_date

        # For away team, score is inverted
        away.glicko = glicko2_update(
            player=away.glicko,
            opponents=[(home.glicko.mu, home.glicko.phi)],
            scores=[1.0 - actual_home],
            tau=p["glicko_tau"],
        )
        away.glicko.last_date = game_date

        # ── Record history ──
        for team_name, state in [(home_team, home), (away_team, away)]:
            self.history.append({
                "team": team_name,
                "sport": self.sport,
                "date": game_date,
                "elo": round(state.elo.rating, 1),
                "glicko_rating": round(state.glicko.rating, 1),
                "glicko_rd": round(state.glicko.rd, 1),
                "glicko_vol": round(state.glicko.sigma, 6),
            })

        self._games_processed += 1

        return {
            "elo_home_pre": elo_home_pre,
            "elo_away_pre": elo_away_pre,
            "elo_expected_home": expected_home,
            "elo_home_change": elo_change,
            "elo_away_change": -elo_change,
            "glicko_home_pre": home.glicko.rating,
            "glicko_away_pre": away.glicko.rating,
            "glicko_expected_home": glicko_expected_home,
        }

    def _apply_season_reversion(self) -> None:
        """Revert all team ratings toward the mean between seasons.

        This prevents ratings from diverging too far and accounts for
        roster changes, coaching changes, etc.
        """
        revert_pct = self.params["elo_season_revert"]
        mean_elo = self.params["elo_initial"]

        for team in self.teams.values():
            # Elo: revert toward 1500
            team.elo.rating = team.elo.rating * (1 - revert_pct) + mean_elo * revert_pct

            # Glicko-2: increase RD (more uncertainty after off-season)
            new_rd = min(team.glicko.rd * 1.3, self.params["glicko_initial_rd"])
            team.glicko = Glicko2Rating.from_traditional(
                rating=team.glicko.rating,
                rd=new_rd,
                vol=team.glicko.sigma,
            )
            team.glicko.games_played = team.glicko.games_played

    def predict(
        self,
        home_team: str,
        away_team: str,
        include_home_advantage: bool = True,
    ) -> dict[str, float]:
        """Predict win probability for a matchup.

        Args:
            home_team: Home team abbreviation.
            away_team: Away team abbreviation.
            include_home_advantage: Whether to add home Elo bonus.

        Returns:
            Dict with probabilities:
            {
                "elo_home_prob": float,
                "glicko_home_prob": float,
                "ensemble_home_prob": float,  # average of both
            }
        """
        home = self._get_or_create_team(home_team)
        away = self._get_or_create_team(away_team)

        # Elo prediction
        home_elo = home.elo.rating
        if include_home_advantage:
            home_elo += self.params["elo_home_adv"]
        elo_prob = elo_expected(home_elo, away.elo.rating)

        # Glicko-2 prediction
        glicko_prob = _E(home.glicko.mu, away.glicko.mu, away.glicko.phi)

        # Ensemble (simple average — Glicko-2 weighted slightly more
        # since it captures uncertainty)
        ensemble = 0.45 * elo_prob + 0.55 * glicko_prob

        return {
            "elo_home_prob": round(elo_prob, 4),
            "glicko_home_prob": round(glicko_prob, 4),
            "ensemble_home_prob": round(ensemble, 4),
            "elo_home": round(home.elo.rating, 1),
            "elo_away": round(away.elo.rating, 1),
            "glicko_home": round(home.glicko.rating, 1),
            "glicko_away": round(away.glicko.rating, 1),
            "glicko_home_rd": round(home.glicko.rd, 1),
            "glicko_away_rd": round(away.glicko.rd, 1),
        }

    def get_all_ratings(self) -> dict[str, dict[str, float]]:
        """Get current ratings for all teams.

        Returns:
            Dict mapping team abbreviation to rating details.
        """
        result = {}
        for team_name, state in sorted(self.teams.items()):
            result[team_name] = {
                "elo": round(state.elo.rating, 1),
                "elo_games": state.elo.games_played,
                "glicko_rating": round(state.glicko.rating, 1),
                "glicko_rd": round(state.glicko.rd, 1),
                "glicko_vol": round(state.glicko.sigma, 6),
                "glicko_games": state.glicko.games_played,
            }
        return result

    def save_to_db(self, db: Any) -> int:
        """Save all rating history to database.

        Args:
            db: SportsDB instance.

        Returns:
            Number of rating records saved.
        """
        if not self.history:
            return 0
        return db.upsert_ratings_batch(self.history)

    def brier_score(
        self,
        predictions: list[tuple[float, float]],
    ) -> float:
        """Compute Brier score for a set of predictions.

        Args:
            predictions: List of (predicted_probability, actual_outcome)
                where actual_outcome is 1.0 for home win, 0.0 for away win.

        Returns:
            Brier score (lower is better, 0.0 = perfect, 0.25 = coin flip).
        """
        if not predictions:
            return 0.25
        total = sum((p - a) ** 2 for p, a in predictions)
        return total / len(predictions)

    def calibration_report(
        self,
        predictions: list[tuple[float, float]],
        n_bins: int = 10,
    ) -> list[dict[str, float]]:
        """Compute calibration report (predicted vs actual by bucket).

        Args:
            predictions: List of (predicted_probability, actual_outcome).
            n_bins: Number of probability bins.

        Returns:
            List of dicts with keys: bin_low, bin_high, n, predicted_avg, actual_avg.
        """
        bins: list[list[tuple[float, float]]] = [[] for _ in range(n_bins)]
        for pred, actual in predictions:
            bin_idx = min(int(pred * n_bins), n_bins - 1)
            bins[bin_idx].append((pred, actual))

        report = []
        for i, bin_data in enumerate(bins):
            if not bin_data:
                continue
            preds = [p for p, _ in bin_data]
            actuals = [a for _, a in bin_data]
            report.append({
                "bin_low": i / n_bins,
                "bin_high": (i + 1) / n_bins,
                "n": len(bin_data),
                "predicted_avg": sum(preds) / len(preds),
                "actual_avg": sum(actuals) / len(actuals),
            })
        return report

    def summary(self) -> str:
        """Return a human-readable summary of the engine state."""
        lines = [
            f"EloGlickoEngine(sport={self.sport})",
            f"  Teams: {len(self.teams)}",
            f"  Games processed: {self._games_processed}",
            f"  History records: {len(self.history)}",
        ]
        if self.teams:
            ratings = sorted(
                self.teams.items(),
                key=lambda x: x[1].elo.rating,
                reverse=True,
            )
            lines.append("  Top 5 by Elo:")
            for team, state in ratings[:5]:
                lines.append(
                    f"    {team:>6s}: Elo={state.elo.rating:7.1f}  "
                    f"Glicko={state.glicko.rating:7.1f} "
                    f"(RD={state.glicko.rd:.1f})"
                )
            lines.append("  Bottom 5 by Elo:")
            for team, state in ratings[-5:]:
                lines.append(
                    f"    {team:>6s}: Elo={state.elo.rating:7.1f}  "
                    f"Glicko={state.glicko.rating:7.1f} "
                    f"(RD={state.glicko.rd:.1f})"
                )
        return "\n".join(lines)


# ── CLI: Compute ratings from database ─────────────────────────────────


def compute_ratings_from_db(sport: str, db_path: str | None = None) -> EloGlickoEngine:
    """Load games from database and compute ratings.

    Args:
        sport: Sport identifier.
        db_path: Optional path to SQLite DB (default: research_d/data/sports_backtest.sqlite).

    Returns:
        Populated EloGlickoEngine.
    """
    from research_d.sports_db import SportsDB

    db = SportsDB(db_path)
    engine = EloGlickoEngine(sport)

    games = db.get_games(sport=sport, status="final")
    print(f"Processing {len(games)} {sport.upper()} games...")

    predictions = []
    for game in games:
        if game["home_score"] is None or game["away_score"] is None:
            continue

        # Get pre-game prediction for calibration
        pred = engine.predict(game["home_team"], game["away_team"])

        # Process the game
        engine.process_game(
            home_team=game["home_team"],
            away_team=game["away_team"],
            home_score=game["home_score"],
            away_score=game["away_score"],
            game_date=game["game_date"],
            season=game["season"],
        )

        # Record prediction for calibration
        actual = 1.0 if game["home_score"] > game["away_score"] else (
            0.5 if game["home_score"] == game["away_score"] else 0.0
        )
        predictions.append((pred["ensemble_home_prob"], actual))

    # Save to database
    n_saved = engine.save_to_db(db)
    print(f"Saved {n_saved} rating records to database.")

    # Report calibration
    if predictions:
        brier = engine.brier_score(predictions)
        print(f"\nCalibration Report:")
        print(f"  Brier Score: {brier:.4f} (0.25=coin flip, lower=better)")
        print(f"  Predictions: {len(predictions)}")
        print()
        report = engine.calibration_report(predictions)
        print(f"  {'Bin':>10s}  {'N':>5s}  {'Pred':>6s}  {'Actual':>6s}  {'Gap':>6s}")
        for row in report:
            gap = row["actual_avg"] - row["predicted_avg"]
            print(
                f"  {row['bin_low']:.1f}-{row['bin_high']:.1f}  "
                f"{row['n']:5d}  {row['predicted_avg']:6.3f}  "
                f"{row['actual_avg']:6.3f}  {gap:+6.3f}"
            )

    print()
    print(engine.summary())
    db.close()
    return engine


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute Elo/Glicko-2 ratings from game results in SQLite DB."
    )
    parser.add_argument(
        "--sport",
        choices=["nba", "epl", "nfl", "all"],
        default="all",
        help="Sport to compute ratings for (default: all).",
    )
    parser.add_argument(
        "--db",
        default=None,
        help="Path to SQLite database (default: research_d/data/sports_backtest.sqlite).",
    )
    args = parser.parse_args()

    sports = ["nba", "epl", "nfl"] if args.sport == "all" else [args.sport]

    for sport in sports:
        print(f"\n{'='*60}")
        print(f"  Computing {sport.upper()} ratings")
        print(f"{'='*60}\n")
        try:
            engine = compute_ratings_from_db(sport, args.db)
        except Exception as e:
            print(f"Error computing {sport} ratings: {e}")
