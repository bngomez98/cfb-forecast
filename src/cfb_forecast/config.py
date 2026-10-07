from dataclasses import asdict, dataclass

METRICS = ("points", "yards_per_play", "yards_per_pass", "yards_per_rush", "first_downs_created_rate")

@dataclass(frozen=True)
class StateConfig:
    process_variance: float = 0.02  # standardized squared units per week
    offseason_carry: float = 0.8
    prior_variance: float = 0.5
    offseason_variance: float = 0.3
    weekly_carry: float = 0.995
    home_margin: float = 2.5
    observation_correlation: float = 0.10

    def __post_init__(self):
        if self.process_variance < 0 or self.prior_variance <= 0:
            raise ValueError("State variances must be nonnegative; prior variance must be positive")
        if not 0 < self.offseason_carry <= 1 or not 0 < self.weekly_carry <= 1:
            raise ValueError("Carry factors must be in (0, 1]")
        if not -1 < self.observation_correlation < 1:
            raise ValueError("Observation correlation must be in (-1, 1)")

    def to_dict(self):
        return asdict(self)
