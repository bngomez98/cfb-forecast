"""Static configuration for the dynamic ratings model.

:data:`METRICS` names the five per-game observations each team is rated on.
:class:`StateConfig` carries the hyperparameters that govern how the latent
offense and defense states drift between games, how uncertainty inflates
across the offseason, and how home-field advantage enters the points model.

The configuration is intentionally small and immutable: everything that would
normally live in a sprawling config file is either a selected hyperparameter
(stored inside a trained model's JSON) or a validated constant here.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

#: Names of the five per-game metrics observed and modeled for each team.
METRICS = (
    "points",
    "yards_per_play",
    "yards_per_pass",
    "yards_per_rush",
    "first_downs_created_rate",
)


@dataclass(frozen=True)
class StateConfig:
    """Hyperparameters for the Kalman-style team-rating state.

    Attributes
    ----------
    process_variance:
        Weekly diffusion variance applied to each team's rating, in
        standardized squared units per week. Larger values let ratings move
        faster.
    offseason_carry:
        Multiplicative shrinkage of the rating mean across the offseason.
        A value of 1 keeps the rating; 0 resets it to the league average.
    prior_variance:
        Initial state variance before any games have been seen.
    offseason_variance:
        Additive variance added across the offseason to represent roster and
        coaching turnover that is not individually modeled.
    weekly_carry:
        Multiplicative shrinkage of the rating mean per elapsed week during
        the season.
    home_margin:
        Fixed home-field advantage, in points, applied symmetrically to the
        two sides' points-state updates.
    observation_correlation:
        Residual correlation between the two teams' observations in a single
        game, used in the joint Kalman update.
    """

    process_variance: float = 0.02
    offseason_carry: float = 0.8
    prior_variance: float = 0.5
    offseason_variance: float = 0.3
    weekly_carry: float = 0.995
    home_margin: float = 2.5
    observation_correlation: float = 0.10

    def __post_init__(self) -> None:
        if self.process_variance < 0 or self.prior_variance <= 0:
            raise ValueError(
                "State variances must be nonnegative; prior variance must be positive"
            )
        if not 0 < self.offseason_carry <= 1 or not 0 < self.weekly_carry <= 1:
            raise ValueError("Carry factors must be in (0, 1]")
        if not -1 < self.observation_correlation < 1:
            raise ValueError("Observation correlation must be in (-1, 1)")

    def to_dict(self) -> dict:
        """Return a plain dict suitable for JSON serialization."""
        return asdict(self)
