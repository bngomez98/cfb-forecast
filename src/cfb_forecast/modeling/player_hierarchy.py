"""One conditional Gaussian posterior for team scores and player matchups.

This is a joint working likelihood, not a drive simulator. Shared random effects
enter player-rate observations AND team-score observations in one precision
matrix. Hyperparameters and normalization are conditional empirical estimates;
the saved artifact records them. No legacy score head or residual map is added.
"""
from dataclasses import asdict, dataclass
from itertools import product
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import splu

from .analytic import score_summary

ROLES = ("passing", "rushing", "receiving")
NOISE_CALIBRATION_FILE = "joint_noise_calibration.json"
ASSUMED_NOISE = {"method": "assumed_score_scale",
                 "note": "Historical score SD squared with no home/away correlation; not measured on unseen games"}


@dataclass(frozen=True)
class HierarchyConfig:
    team_variance: float = 0.5
    player_variance: float = 0.5
    defense_variance: float = 0.5
    matchup_variance: float = 0.1
    context_variance: float = 1.0
    season_innovation_variance: float = 0.3
    offseason_carry: float = 0.8
    max_uncertain_players: int = 10

    def __post_init__(self):
        for name, value in asdict(self).items():
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.offseason_carry > 1:
            raise ValueError("offseason_carry must be at most one")


class PlayerHierarchy:
    def __init__(self, config=None):
        self.config = config or HierarchyConfig()
        self.keys = []
        self.index = {}
        self.rows = []
        self.targets = []
        self.weights = []
        self.transitions = []
        self.rate_scales = {}
        self.context_center = None
        self.context_scale = None
        self.context_fill = None
        self.score_center = 0.
        self.score_scale = 1.
        self.cutoff = None
        self.latest_result = None
        self.noise_covariance_source = dict(ASSUMED_NOISE)

    def _key(self, *parts):
        key = json.dumps(parts, separators=(",", ":"))
        if key not in self.index:
            if hasattr(self, "posterior_mean"):
                raise ValueError("Prediction references effects absent from the frozen posterior")
            self.index[key] = len(self.keys)
            self.keys.append(key)
        return key

    def _player_terms(self, player, opponent, season):
        role, identity = player["role"], player["player_id"]
        return {self._key("player", identity, role, season): 1.,
                self._key("role_defense", int(opponent), role, season): -1.,
                self._key("matchup", identity, int(opponent), role): 1.}

    def _context(self, values):
        values = np.asarray(values, float)
        missing = ~np.isfinite(values)
        completed = np.where(missing, self.context_fill, values)
        return np.r_[np.where(self.context_active, (completed - self.context_center) / self.context_scale, 0.),
                     np.where(self.missing_active, missing.astype(float), 0.)]

    def score_design(self, team, opponent, season, venue, lineup, context):
        row = {self._key("score_intercept"): 1., self._key("venue"): float(venue),
               self._key("team_offense", int(team), int(season)): 1.,
               self._key("team_defense", int(opponent), int(season)): -1.}
        for j, value in enumerate(self._context(context)):
            row[self._key("context", j)] = float(value)
        # Add opportunity-weighted, centered role contributions. Missing roles
        # contribute no observed evidence; other roles are not rescaled.
        for player in lineup:
            if player["role"] not in self.rate_scales:
                continue
            if player["player_id"] == "unidentified_replacement":
                player = {**player, "player_id": f"unidentified_replacement:{team}:{season}"}
            share = float(player["opportunity_share"])
            for key, value in self._player_terms(player, opponent, int(season)).items():
                row[key] = row.get(key, 0.) + share * value
        return row

    def _observation(self, row, target, weight=1.):
        if not np.isfinite(target) or not np.isfinite(weight) or weight <= 0:
            raise ValueError("Nonfinite hierarchical observation")
        self.rows.append(row)
        self.targets.append(float(target))
        self.weights.append(float(weight))

    def _matrix(self, rows):
        rr, cc, vv = [], [], []
        for i, row in enumerate(rows):
            for key, value in row.items():
                rr.append(i); cc.append(self.index[key]); vv.append(value)
        return sparse.csr_matrix((vv, (rr, cc)), shape=(len(rows), len(self.keys)))

    @classmethod
    def fit(cls, games, players, context, future_designs, cutoff, config=None):
        """Fit solely from completed games available strictly before cutoff.

        future_designs registers new teams/players/matchups under their priors;
        it contains no outcomes. Historical score rows use observed participants
        conditional on their recorded opportunities, never as backtest features.
        """
        model = cls(config)
        at = pd.Timestamp(cutoff)
        if at.tzinfo is None:
            raise ValueError("cutoff must have a timezone")
        dates = pd.to_datetime(games.result_available_at, utc=True)
        if len(games) < 3 or not (dates < at).all():
            raise ValueError("Joint fit requires at least three eligible prior games")
        if not np.isfinite(games[["home_points", "away_points"]].to_numpy(float)).all():
            raise ValueError("Joint fit requires finite observed scores")
        model.cutoff = at.isoformat()
        model.latest_result = dates.max().isoformat()
        scores = games[["home_points", "away_points"]].to_numpy(float)
        model.score_center = float(scores.mean())
        model.score_scale = max(float(scores.std()), 1.)
        values = np.asarray(context, float)
        if values.ndim != 3 or values.shape[:2] != (len(games), 2):
            raise ValueError("Context must have shape (games, 2, features)")
        flat = values.reshape(len(games) * 2, values.shape[-1])
        count = np.isfinite(flat).sum(0)
        model.context_fill = np.divide(np.nansum(np.where(np.isfinite(flat), flat, np.nan), axis=0), count,
                                       out=np.zeros(flat.shape[1]), where=count > 0)
        completed = np.where(np.isfinite(flat), flat, model.context_fill)
        model.context_center = completed.mean(0)
        spread = completed.std(0)
        model.context_active = (count >= 3) & (spread > 1e-6)
        model.missing_active = (count > 0) & (count < len(flat))
        model.context_scale = np.where(spread > 1e-6, spread, 1.)
        allowed = set(games.game_id.astype(int))
        if len(players):
            players = players.loc[players.game_id.isin(allowed)].copy()
            if not (pd.to_datetime(players.available_at, utc=True) < at).all():
                raise ValueError("Player observation at/after the information cutoff")
        for role in ROLES:
            part = players.loc[players.role == role] if len(players) else pd.DataFrame()
            if part.empty:
                continue
            epa = pd.to_numeric(part.epa, errors="coerce").to_numpy(float)
            metric = "epa" if np.isfinite(epa).sum() >= 3 and np.nanstd(epa) > 1e-6 else "yards_per_opportunity"
            rate = epa if metric == "epa" else part.yards.to_numpy(float) / part.opportunities.to_numpy(float)
            finite = np.isfinite(rate)
            if finite.sum() < 3 or np.std(rate[finite]) < 1e-6:
                continue
            model.rate_scales[role] = {"metric": metric, "center": float(rate[finite].mean()),
                                       "scale": float(rate[finite].std())}
            for record, observed in zip(part.to_dict("records"), rate):
                if not np.isfinite(observed):
                    continue
                row = model._player_terms(record, record["opponent_id"], int(record["season"]))
                row[model._key("role_intercept", role)] = 1.
                scale = model.rate_scales[role]
                # Game-level player rate is the observation unit. Treating every
                # recorded opportunity as independent would overstate precision.
                model._observation(row, (observed - scale["center"]) / scale["scale"])
        if not model.rate_scales:
            raise ValueError("No variable, identified player-rate history is available")
        grouped = {} if players.empty else {int(k): v for k, v in players.groupby("game_id")}
        for slot, game in enumerate(games.itertuples(index=False)):
            part = grouped.get(int(game.game_id), pd.DataFrame())
            for side, (team, opponent) in enumerate(((game.home_id, game.away_id), (game.away_id, game.home_id))):
                lineup = observed_lineup(part, int(team))
                venue = 0 if game.neutral_site else (1 if side == 0 else -1)
                row = model.score_design(team, opponent, game.season, venue, lineup, values[slot, side])
                model._observation(row, (scores[slot, side] - model.score_center) / model.score_scale)
        for spec in future_designs:
            for lineup in availability_scenarios(spec["lineup"], model.config.max_uncertain_players):
                model.score_design(spec["team"], spec["opponent"], spec["season"], spec["venue"], lineup[1], spec["context"])
            for player in spec["lineup"]:
                if player["role"] in model.rate_scales:
                    model._player_terms(player, spec["opponent"], int(spec["season"]))
                    model._key("role_intercept", player["role"])
        nonroots = model._season_transitions()
        X = model._matrix(model.rows)
        w = np.asarray(model.weights)
        prior = np.array([model._prior_variance(json.loads(key)[0]) for key in model.keys])
        prior_precision = 1. / prior
        for key in nonroots:
            prior_precision[model.index[key]] = 0.
        Q = sparse.diags(prior_precision, format="csc")
        if model.transitions:
            T = model._matrix(model.transitions)
            Q = Q + T.T @ T / model.config.season_innovation_variance
        precision = Q + X.T @ sparse.diags(w) @ X
        model.precision = precision.tocsc()
        model.factor = _factorize(model.precision)
        model.posterior_mean = model.factor.solve(np.asarray(X.T @ (w * np.asarray(model.targets))).ravel())
        # Until calibrate_noise_covariance() is given chronological holdout
        # errors, the leftover score noise is an explicit assumption: the
        # historical score scale with independent home/away errors.
        model.noise_covariance = np.eye(2) * model.score_scale ** 2
        model.noise_covariance_source = dict(ASSUMED_NOISE)
        model.training_through_season = int(games.season.max())
        model.training_games = len(games)
        model.training_player_observations = len(model.targets) - 2 * len(games)
        return model

    def calibrate_noise_covariance(self, past_games, min_games=100):
        """Set the leftover score noise from chronological holdout errors.

        ``past_games`` holds one row per game forecast by a model fitted only
        on results available before that game (see holdout_noise_covariance).
        Errors from games used in the fit would understate the uncertainty.
        """
        noise, source = holdout_noise_covariance(past_games, min_games=min_games)
        self.noise_covariance = noise
        self.noise_covariance_source = source
        return noise

    def _prior_variance(self, group):
        if group in ("score_intercept", "role_intercept"):
            return 10.
        if group == "player":
            return self.config.player_variance
        if group == "role_defense":
            return self.config.defense_variance
        if group == "matchup":
            return self.config.matchup_variance
        if group in ("team_offense", "team_defense"):
            return self.config.team_variance
        return self.config.context_variance

    def _season_transitions(self):
        groups = {}
        nonroots = []
        for key in list(self.keys):
            parts = json.loads(key)
            if parts[0] in ("player", "team_offense", "team_defense", "role_defense"):
                groups.setdefault(tuple(parts[:-1]), []).append((int(parts[-1]), key))
        for entries in groups.values():
            ordered = sorted(entries)
            for (prev_year, previous), (year, current) in zip(ordered, ordered[1:]):
                carry = self.config.offseason_carry ** (year - prev_year)
                gap_scale = np.sqrt(year - prev_year)
                self.transitions.append({current: 1. / gap_scale, previous: -carry / gap_scale})
                nonroots.append(current)
        return nonroots

    def predict(self, home, away, market=None, players=True):
        specs = (home, away)
        scenarios = [availability_scenarios(s["lineup"], self.config.max_uncertain_players) for s in specs]
        if len(scenarios[0]) * len(scenarios[1]) > 2 ** self.config.max_uncertain_players:
            raise ValueError("Too many uncertain players for exact availability integration")
        components = []
        for hs, aws in product(*scenarios):
            probability = hs[0] * aws[0]
            if probability == 0:
                continue
            rows = [self.score_design(s["team"], s["opponent"], s["season"], s["venue"], lineup, s["context"])
                    for s, lineup in zip(specs, (hs[1], aws[1]))]
            X = self._matrix(rows).toarray()
            raw = self.score_center + self.score_scale * (X @ self.posterior_mean)
            means = np.maximum(raw, 0.)
            parameter_cov = (X @ self.factor.solve(X.T)) * self.score_scale ** 2
            covariance = self.noise_covariance + parameter_cov
            summary = score_summary(means, covariance, **(market or {}))
            components.append((probability, means, covariance, summary))
        means = sum(p * mu for p, mu, _, _ in components)
        covariance = sum(p * (cov + np.outer(mu - means, mu - means)) for p, mu, cov, _ in components)
        result = score_summary(means, covariance, **(market or {}))
        # Probabilities integrate the Gaussian mixture, not its moment match.
        for key in components[0][3]:
            if key.endswith("probability") or key.endswith("expected_net_per_unit"):
                result[key] = float(sum(p * summary[key] for p, _, _, summary in components))
        result.update(distribution_method="joint_player_gaussian_posterior",
                      probability_source="joint_player_posterior", model_spread=float(-means[0] + means[1]),
                      score_covariance=covariance.tolist(), availability_components=len(components),
                      intervals_method="moment_matched_gaussian_80",
                      noise_covariance_method=self.noise_covariance_source.get("method"),
                      player_forecasts=({"home": self.player_predictions(home), "away": self.player_predictions(away)}
                                        if players else None))
        return result

    def player_predictions(self, spec):
        scenarios = availability_scenarios(spec["lineup"], self.config.max_uncertain_players)
        players, rows = [], []
        for player in spec["lineup"]:
            if player["role"] not in self.rate_scales:
                continue
            row = self._player_terms(player, spec["opponent"], int(spec["season"]))
            row[self._key("role_intercept", player["role"])] = 1.
            players.append(player)
            rows.append(row)
        if not rows:
            return []
        # One multi-column solve for the whole lineup instead of one per player.
        X = self._matrix(rows)
        means = X @ self.posterior_mean
        spreads = np.asarray(X.multiply(self.factor.solve(X.T.toarray()).T).sum(1)).ravel()
        result = []
        for player, mean, spread in zip(players, means, spreads):
            scale = self.rate_scales[player["role"]]
            rate = scale["center"] + scale["scale"] * float(mean)
            variance = scale["scale"] ** 2 * (1 + float(spread))
            share = sum(p * sum(q["opportunity_share"] for q in lineup if q["player_id"] == player["player_id"] and q["role"] == player["role"])
                        for p, lineup in scenarios)
            result.append({"player_id": player["player_id"], "role": player["role"], "metric": scale["metric"],
                           "conditional_rate_mean": rate, "conditional_rate_variance": variance,
                           "expected_opportunity_share": share,
                           "availability_probability": player.get("availability_probability"),
                           "identity_scope": "team_season_name" if player["player_id"].startswith("legacy:") else "canonical"})
        return result

    def save(self, folder):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        sparse.save_npz(folder / "joint_player_precision.npz", self.precision)
        metadata = {"schema_version": 1, "family": "joint_player_gaussian", "config": asdict(self.config),
                    "keys": self.keys, "posterior_mean": self.posterior_mean.tolist(), "rate_scales": self.rate_scales,
                    "context_fill": self.context_fill.tolist(), "context_center": self.context_center.tolist(),
                    "context_scale": self.context_scale.tolist(), "context_active": self.context_active.tolist(),
                    "missing_active": self.missing_active.tolist(), "score_center": self.score_center,
                    "score_scale": self.score_scale, "noise_covariance": self.noise_covariance.tolist(),
                    "noise_covariance_source": self.noise_covariance_source,
                    "cutoff": self.cutoff, "latest_result": self.latest_result, "training_games": self.training_games,
                    "training_player_observations": self.training_player_observations,
                    "training_through_season": self.training_through_season}
        (folder / "joint_player_model.json").write_text(json.dumps(metadata, indent=2, allow_nan=False))

    @classmethod
    def load(cls, folder):
        folder = Path(folder)
        metadata = json.loads((folder / "joint_player_model.json").read_text())
        if metadata.pop("schema_version") != 1 or metadata.pop("family") != "joint_player_gaussian":
            raise ValueError("Unsupported joint hierarchy artifact")
        model = cls(HierarchyConfig(**metadata.pop("config")))
        for key, value in metadata.items():
            setattr(model, key, np.asarray(value) if key in ("posterior_mean", "context_fill", "context_center", "context_scale", "context_active", "missing_active", "noise_covariance") else value)
        model.index = {key: i for i, key in enumerate(model.keys)}
        model.precision = sparse.load_npz(folder / "joint_player_precision.npz")
        if model.precision.shape != (len(model.keys), len(model.keys)) or not np.isfinite(model.posterior_mean).all():
            raise ValueError("Invalid joint posterior dimensions or values")
        model.factor = _factorize(model.precision.tocsc())
        return model


def holdout_noise_covariance(predictions, min_games=100, floor=1.):
    """Leftover score noise measured on games the forecasting fit never saw.

    Each row is one game with ``home_points``, ``away_points``,
    ``expected_home_score``, ``expected_away_score`` and the posterior
    parameter covariance of that forecast (``parameter_cov_home``,
    ``parameter_cov_cross``, ``parameter_cov_away``). The predictive error
    covariance is noise plus parameter uncertainty, so the noise is the
    observed second moment of the errors (bias included, so a biased mean
    widens rather than hides the uncertainty) minus the average parameter
    covariance. Eigenvalues are floored at ``floor`` points squared.
    """
    required = ["home_points", "away_points", "expected_home_score", "expected_away_score",
                "parameter_cov_home", "parameter_cov_cross", "parameter_cov_away"]
    missing = [c for c in required if c not in predictions]
    if missing:
        raise ValueError(f"Holdout predictions lack columns: {missing}")
    frame = predictions[required + [c for c in ("start_date", "week", "result_available_at") if c in predictions]]
    frame = frame.loc[np.isfinite(frame[required].to_numpy(float)).all(1)]
    if len(frame) < min_games:
        raise ValueError(f"Noise calibration needs at least {min_games} holdout games, got {len(frame)}")
    errors = (frame[["home_points", "away_points"]].to_numpy(float)
              - frame[["expected_home_score", "expected_away_score"]].to_numpy(float))
    observed = errors.T @ errors / len(errors)
    parameter = np.array([[frame.parameter_cov_home.mean(), frame.parameter_cov_cross.mean()],
                          [frame.parameter_cov_cross.mean(), frame.parameter_cov_away.mean()]])
    values, vectors = np.linalg.eigh((observed - parameter + (observed - parameter).T) / 2)
    noise = vectors @ np.diag(np.maximum(values, floor)) @ vectors.T
    source = {"method": "chronological_holdout_residuals", "games": int(len(frame)),
              "observed_error_second_moment": observed.tolist(),
              "mean_parameter_covariance": parameter.tolist(),
              "mean_error": errors.mean(0).tolist(),
              "error_correlation": float(np.corrcoef(errors.T)[0, 1]),
              "eigenvalue_floor": float(floor), "floored": bool((values < floor).any())}
    if "week" in frame:
        # Reported so a real early/late-season difference can be seen; one
        # pooled covariance is used unless the evidence calls for a split.
        for phase, mask in (("weeks_1_4", frame.week.to_numpy(float) <= 4), ("weeks_5_plus", frame.week.to_numpy(float) > 4)):
            if mask.sum():
                part = errors[mask]
                source[f"{phase}_games"] = int(mask.sum())
                source[f"{phase}_error_second_moment"] = (part.T @ part / len(part)).tolist()
    if "result_available_at" in frame:
        source["calibrated_through"] = pd.to_datetime(frame.result_available_at, utc=True).max().isoformat()
    return noise, source


def _factorize(precision):
    """Sparse LU of the symmetric posterior precision.

    A symmetric minimum-degree ordering keeps fill-in roughly six times lower
    than the default column ordering on real seasons, so fitting and every
    posterior-variance solve are several times faster. The solution is the
    same linear system; only the elimination order changes.
    """
    return splu(precision, permc_spec="MMD_AT_PLUS_A", options={"SymmetricMode": True})


def observed_lineup(players, team):
    if players.empty:
        return []
    result = []
    for role, part in players.loc[players.team_id == team].groupby("role"):
        total = float(part.opportunities.sum())
        for row in part.to_dict("records"):
            result.append({"player_id": row["player_id"], "role": role,
                           "opportunity_share": float(row["opportunities"]) / total})
    return result


def availability_scenarios(lineup, max_uncertain=10):
    """Exact finite mixture for supplied availability probabilities.

    Missing availability stays unknown: participation is conditional on the
    supplied/inferred lineup, with no invented healthy probability. Shares of
    unavailable players are redistributed within the same role. An empty role
    uses an unidentified replacement with its own prior in score_design.
    """
    seen = set()
    totals = {}
    for player in lineup:
        key = (player["player_id"], player["role"])
        if not key[0] or key[1] not in ROLES or key in seen:
            raise ValueError("Invalid or duplicate player/role in lineup")
        seen.add(key)
        share = float(player["opportunity_share"])
        if not np.isfinite(share) or share < 0:
            raise ValueError("Invalid opportunity share")
        totals[key[1]] = totals.get(key[1], 0.) + share
        probability = player.get("availability_probability")
        if probability is not None and (not np.isfinite(probability) or not 0 <= probability <= 1):
            raise ValueError("Invalid availability probability")
    if any(not np.isclose(total, 1.) for total in totals.values()):
        raise ValueError("Role opportunity shares must sum to one")
    uncertain = [i for i, p in enumerate(lineup) if p.get("availability_probability") is not None and 0 < float(p["availability_probability"]) < 1]
    if len(uncertain) > max_uncertain:
        raise ValueError("Too many uncertain players for exact availability integration")
    scenarios = []
    for bits in product((0, 1), repeat=len(uncertain)):
        status = {i: bit for i, bit in zip(uncertain, bits)}
        probability = 1.
        kept = []
        for i, player in enumerate(lineup):
            p = player.get("availability_probability")
            active = status.get(i, p != 0)
            if i in status:
                probability *= float(p) if active else 1 - float(p)
            if active:
                kept.append({**player, **({"availability_probability": 1.} if p is not None else {})})
        for role in {p["role"] for p in lineup}:
            part = [p for p in kept if p["role"] == role]
            total = sum(p["opportunity_share"] for p in part)
            if total > 0:
                for player in part:
                    player["opportunity_share"] /= total
            else:
                kept = [p for p in kept if p["role"] != role]
                # Explicitly unidentified replacement, never a fabricated roster.
                kept.append({"player_id": "unidentified_replacement", "role": role, "opportunity_share": 1.})
        scenarios.append((probability, kept))
    return scenarios
