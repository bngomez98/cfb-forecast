"""Adaptive outcome blending: the self-reinforcing prediction layer.

This module is what makes the system an *application that learns over time*
rather than a static scorer. It keeps no team ratings and performs no
smoothing of the observed data; every game is kept exactly as recorded. What
adapts is the **blend** between competing prediction experts:

* the dynamic Gaussian score head (the production model),
* a nearest-neighbor matchup expert borrowed from similar past games
  (Hoegh et al., 2015),
* the betting market when a pre-kickoff line was captured (Cox et al., 2020
  show spreads are the strongest single regular-season benchmark),
* a constant league-base-rate expert (the no-signal floor any model must
  beat).

For each expert we maintain a Beta posterior over its accuracy, updated with
every settled game: ``Beta(a, b) -> Beta(a + win, b + loss)``. The posterior
mean is the expert's current credibility; the blend weights are softmax over
credibility-adjusted log-posteriors, so an expert that has been right lately
dominates tomorrow's forecasts and an expert that drifts into error is
automatically down-weighted. This is the standard Hedge / multiplicative-
weights bandit recipe adapted to sequential sports forecasting, and it is
deliberately rating-free: the state it maintains is about *models*, not teams.

Two safety properties are enforced by construction:

1. **No look-ahead.** Updates for a settled game use only that game's own
   realized winner against predictions frozen before kickoff. Weights used to
   produce a forecast always come from the ledger state *before* that game.
2. **Persistence.** The full posterior state serializes to JSON so the CLI,
   the dashboard and nightly workers share one continuously-learning brain.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

#: Experts tracked in the blend, in canonical order. ``ridge``/``logit``/
#: ``gbm`` are the conditional score heads (the production experts); the
#: legacy name ``model`` is accepted as an alias for ``ridge`` so ledgers and
#: saved states written before the head split keep loading without migration.
EXPERTS = ("ridge", "logit", "gbm", "matchup", "market", "base")

#: Alias map applied when loading persisted blend state from older schema.
_EXPERT_ALIASES = {"model": "ridge"}

#: Prior strength of the initial Beta(a, b) on each expert's accuracy. A
#: Beta(2, 2) prior starts every expert at 50% credibility with mild
#: confidence, so early settled games move beliefs without whipsawing.
PRIOR_ALPHA = 2.0
PRIOR_BETA = 2.0

#: Softmax temperature converting credibility differences into blend weights.
#: Smaller values make the blend winner-take-all faster; larger keep it soft.
BLEND_TEMPERATURE = 8.0

#: Floor weight so a temporarily cold expert can never be fully discarded
#: (nonstationarity insurance: a market that goes quiet or a model mid-refit
#: should still contribute a token amount until evidence says otherwise).
MIN_WEIGHT = 0.02


@dataclass
class ExpertState:
    """Beta-accuracy posterior for one expert plus its realized history."""

    name: str
    alpha: float = PRIOR_ALPHA
    beta: float = PRIOR_BETA
    games: int = 0
    correct: int = 0

    @property
    def credibility(self) -> float:
        """Posterior mean accuracy of this expert."""
        return self.alpha / (self.alpha + self.beta)

    @property
    def posterior_sd(self) -> float:
        """Standard deviation of the Beta posterior (uncertainty measure)."""
        a, b = self.alpha, self.beta
        total = a + b
        return math.sqrt(a * b / (total**2 * (total + 1)))

    def update(self, correct: bool, *, decay: float = 1.0) -> None:
        """Multiply-in one Bernoulli observation, optionally forgetting old games.

        ``decay < 1`` geometrically discounts all past evidence, giving the
        blend a recency bias suited to nonstationary seasons (roster turnover,
        rule changes). With ``decay == 1`` the posterior is the exact batch
        conjugate update.
        """
        if not 0 < decay <= 1:
            raise ValueError("decay must be in (0, 1]")
        self.alpha = (
            self.alpha * decay + (PRIOR_ALPHA * (1 - decay) if decay < 1 else 0.0) + (1.0 if correct else 0.0)
        )
        self.beta = (
            self.beta * decay + (PRIOR_BETA * (1 - decay) if decay < 1 else 0.0) + (0.0 if correct else 1.0)
        )
        self.games += 1
        self.correct += int(correct)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "alpha": self.alpha,
            "beta": self.beta,
            "games": self.games,
            "correct": self.correct,
        }

    @classmethod
    def from_dict(cls, value: dict) -> ExpertState:
        state = cls(value["name"], float(value["alpha"]), float(value["beta"]))
        state.games = int(value.get("games", 0))
        state.correct = int(value.get("correct", 0))
        return state


@dataclass
class AdaptiveBlend:
    """Continuously learned mixture over the expert pool.

    Attributes
    ----------
    experts:
        Mapping of expert name to :class:`ExpertState`.
    decay:
        Per-update geometric forgetting factor (1.0 = full memory).
    log:
        In-memory trail of settled-game updates (game_id, per-expert
        correctness, resulting weights). Cleared on load; the persisted
        ledger file is the source of truth.
    """

    experts: dict[str, ExpertState] = field(default_factory=dict)
    decay: float = 1.0
    log: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.experts:
            self.experts = {name: ExpertState(name) for name in EXPERTS}
        # Accept legacy expert names (e.g. "model" -> "ridge") from saved state.
        renamed = {}
        for name, state in list(self.experts.items()):
            canonical = _EXPERT_ALIASES.get(name, name)
            if canonical == name:
                renamed[name] = state
            elif canonical not in renamed:
                state.name = canonical
                renamed[canonical] = state
        self.experts = renamed
        missing = [name for name in EXPERTS if name not in self.experts]
        if missing:
            raise ValueError(f"Blend is missing experts: {missing}")
        if not 0 < self.decay <= 1:
            raise ValueError("decay must be in (0, 1]")

    # ------------------------------------------------------------------ weights
    def _canonical(self, predictions: dict) -> dict:
        """Route every incoming prediction onto this blend's actual expert keys.

        Older callers and ledgers name the production head ``model``; the
        current registry splits it into ``ridge``/``logit``/``gbm``.  The
        mapping is bidirectional so both naming generations work against
        either blend construction:

        * legacy ``model`` -> canonical ``ridge`` when the blend has no
          ``model`` expert (forward compatibility with old ledgers);
        * canonical ``ridge`` -> legacy ``model`` when the blend was built
          with the old key set (backward compatibility with new callers).

        A key that already exists on the blend is never overwritten by an
        alias, so explicit per-head predictions always win.
        """
        merged = dict(predictions)
        for source, target in list(_EXPERT_ALIASES.items()) + [
            (target, source) for source, target in _EXPERT_ALIASES.items()
        ]:
            if source in merged and source not in self.experts:
                if target in self.experts and target not in merged:
                    merged[target] = merged[source]
                merged.pop(source)
        return merged

    def weights(self) -> dict[str, float]:
        """Current softmax blend weights, floored at ``MIN_WEIGHT``."""
        raw = {name: BLEND_TEMPERATURE * _log_posterior(state) for name, state in self.experts.items()}
        shift = max(raw.values())
        exp = {name: math.exp(value - shift) for name, value in raw.items()}
        total = sum(exp.values())
        soft = {name: value / total for name, value in exp.items()}
        n = len(soft)
        floored = {name: MIN_WEIGHT + (1.0 - MIN_WEIGHT * n) * weight for name, weight in soft.items()}
        norm = sum(floored.values())
        return {name: weight / norm for name, weight in floored.items()}

    def blend_probability(self, predictions: dict[str, float | None]) -> tuple[float, dict[str, float]]:
        """Weighted home-win probability over whichever experts reported.

        Missing experts (no market line captured, empty neighbor pool) are
        renormalized away rather than imputed, and the returned effective
        weight map records exactly which experts shaped the number.
        """
        weights = self.weights()
        predictions = self._canonical(predictions)
        usable = {name: float(p) for name, p in predictions.items() if p is not None and name in weights}
        if not usable:
            raise ValueError("blend_probability requires at least one available expert prediction")
        mass = sum(weights[name] for name in usable)
        effective = {name: weights[name] / mass for name in usable}
        probability = sum(effective[name] * value for name, value in usable.items())
        return float(np.clip(probability, 1e-9, 1 - 1e-9)), effective

    # ------------------------------------------------------------------ updates
    def observe(self, game_id: int, predictions: dict[str, float | None], home_win: bool) -> dict:
        """Settle one game: score every expert that predicted, update posteriors.

        ``predictions`` holds each expert's *frozen pre-kickoff* home-win
        probability (None where the expert abstained). An expert counts as
        correct when its side of the coin matched the realized winner; an
        abstaining expert neither gains nor loses evidence.
        """
        predictions = self._canonical(predictions)
        record = {"game_id": int(game_id), "home_win": bool(home_win), "experts": {}}
        for name, state in self.experts.items():
            p = predictions.get(name)
            if p is None:
                record["experts"][name] = {"p": None, "correct": None}
                continue
            p = float(p)
            correct = (p >= 0.5) == bool(home_win)
            state.update(correct, decay=self.decay)
            record["experts"][name] = {"p": p, "correct": bool(correct)}
        record["weights_after"] = self.weights()
        self.log.append(record)
        return record

    # ------------------------------------------------------------- persistence
    def to_dict(self) -> dict:
        return {
            "schema_version": 1,
            "decay": self.decay,
            "experts": {name: state.to_dict() for name, state in self.experts.items()},
        }

    @classmethod
    def from_dict(cls, value: dict) -> AdaptiveBlend:
        if value.get("schema_version") != 1:
            raise ValueError("Unsupported adaptive-blend schema")
        experts = {name: ExpertState.from_dict(blob) for name, blob in value["experts"].items()}
        return cls(experts=experts, decay=float(value.get("decay", 1.0)))

    def save(self, path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2))
        return target

    @classmethod
    def load(cls, path) -> AdaptiveBlend:
        text = Path(path).read_text()
        if not text.strip():
            raise ValueError(f"Adaptive blend state at {path} is empty; run 'blend-init' first")
        return cls.from_dict(json.loads(text))


def _log_posterior(state: ExpertState) -> float:
    """Log of the Beta posterior density peak, safe for tiny posteriors."""
    a, b = max(state.alpha, 1e-6), max(state.beta, 1e-6)
    return (a - 1) * math.log(a / (a + b)) + (b - 1) * math.log(b / (a + b))


def base_rate_expert(win_share: float) -> float:
    """Home-win probability for the constant base-rate expert.

    Computed from the historical share of home wins among FBS-v-FBS games
    strictly available before the forecast cutoff; callers pass the share,
    this function clamps degenerate samples toward 0.5.
    """
    if not np.isfinite(win_share):
        return 0.5
    return float(np.clip(0.5 + 0.9 * (win_share - 0.5), 0.05, 0.95))
