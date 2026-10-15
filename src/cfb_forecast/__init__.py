"""College football forecasts with explicit data cutoffs and provenance."""
__version__ = "0.1.0"

from .score import score_games
from .weekly import fps_weeks, run_cycle, run_week, validate_pending
