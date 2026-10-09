"""Final forecast: the model's projection informed by the market and external predictions.

Three margin signals are combined per game:

* model     the engine's projected home margin (always present)
* market    the latest pregame line at the cutoff, as a home margin (-home_spread)
* external  the mean projected home margin of external prediction sources at the cutoff
            (predicted scores, spreads, or win probabilities converted to margins)

``final_margin = w_model * model + w_market * market + w_external * external``

The weights are fitted on held-out games (2025: model held-out predictions, the
pregame displayed lines and the archived public prediction tracker), minimizing
squared margin error with the weights non-negative and summing to one. The market
is informative, not decisive: its weight is capped (``CFB_MARKET_MAX_WEIGHT``,
default 0.35) and can never exceed the model's own weight. When a signal is missing
for a game, the remaining weights are renormalized, so a game with no market line
is forecast from the model and external sources only.

Win probability is ``Phi(final_margin / sigma)`` with ``sigma`` fitted by maximum
likelihood on the same held-out games. Totals stay the model's own: there is no
held-out pregame total history to validate a total blend against.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import norm

from .logs import get

log = get("blend")

SIGNALS = ("model", "market", "external")
DEFAULT_PATH = Path("artifacts/backtest/blend/blend.json")


def market_cap():
    return float(os.environ.get("CFB_MARKET_MAX_WEIGHT", "0.35"))


# ---------------------------------------------------------------- fitting
def training_frame(predictions_path, market_path, archive_path, variant="pbp"):
    """Held-out games with all three signals as of each game's forecast cutoff."""
    p = pd.read_csv(predictions_path)
    p["start_date"] = pd.to_datetime(p.start_date, utc=True, format="mixed")
    p["cutoff"] = pd.to_datetime(p.forecast_cutoff, utc=True, format="mixed")
    m = pd.read_csv(market_path)
    m["observed_at"] = pd.to_datetime(m.observed_at, utc=True, format="mixed")
    m = m.merge(p[["game_id", "start_date", "cutoff"]], on="game_id")
    m = m[(m.observed_at <= m.cutoff) & (m.observed_at < m.start_date) & m.home_spread.notna()]
    m = m.sort_values("observed_at").drop_duplicates("game_id", keep="last")
    a = pd.read_csv(archive_path)[["game_id", "lineavg"]].dropna()
    f = p.merge(m[["game_id", "home_spread"]], on="game_id").merge(a, on="game_id")
    return pd.DataFrame({
        "game_id": f.game_id, "week": f.week, "actual_margin": f.actual_home_margin, "home_win": f.home_win,
        "model": f[f"{variant}_home_margin"], "market": -f.home_spread, "external": f.lineavg,
        "model_p": f[f"{variant}_p"],
    })


def fit_weights(X, y, cap):
    """Convex weights minimizing squared error, market <= min(cap, model weight)."""
    k = X.shape[1]
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1},
            {"type": "ineq", "fun": lambda w: cap - w[1]},
            {"type": "ineq", "fun": lambda w: w[0] - w[1]}]
    res = minimize(lambda w: np.mean((X @ w - y) ** 2), np.full(k, 1 / k), bounds=[(0, 1)] * k, constraints=cons, method="SLSQP")
    return np.clip(res.x, 0, 1) / np.clip(res.x, 0, 1).sum()


def fit_sigma(margin, home_win):
    nll = lambda s: -np.mean(home_win * norm.logcdf(margin / s) + (1 - home_win) * norm.logcdf(-margin / s))  # noqa: E731
    return float(minimize_scalar(nll, bounds=(5, 40), method="bounded").x)


def _metrics(margin, p, y, home_win):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {"margin_mae": float(np.mean(np.abs(margin - y))), "winner_accuracy": float(np.mean((p >= 0.5) == (home_win == 1))),
            "log_loss": float(-np.mean(home_win * np.log(p) + (1 - home_win) * np.log(1 - p))),
            "brier": float(np.mean((p - home_win) ** 2))}


def fit_blend(predictions_path="artifacts/backtest/predictions.csv", market_path="data/market_history.csv",
              archive_path="data/archive/matched_predictions.csv", out_path=DEFAULT_PATH, cap=None, folds=5):
    cap = market_cap() if cap is None else float(cap)
    t = training_frame(predictions_path, market_path, archive_path)
    if len(t) < 100:
        raise ValueError(f"Only {len(t)} held-out games have all signals; need at least 100 to fit the blend")
    X, y, hw = t[list(SIGNALS)].to_numpy(float), t.actual_margin.to_numpy(float), t.home_win.to_numpy(float)
    w = fit_weights(X, y, cap)
    sigma = fit_sigma(X @ w, hw)
    # Out-of-sample check: week-blocked cross-validation of the whole fitting procedure.
    weeks = np.array(sorted(t.week.unique()))
    oos_m, oos_p = np.empty(len(t)), np.empty(len(t))
    for fold in np.array_split(weeks, folds):
        test = t.week.isin(fold).to_numpy()
        wf = fit_weights(X[~test], y[~test], cap)
        sf = fit_sigma(X[~test] @ wf, hw[~test])
        oos_m[test] = X[test] @ wf
        oos_p[test] = norm.cdf(oos_m[test] / sf)
    model_sigma = fit_sigma(X[:, 0], hw)
    spec = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "weights": dict(zip(SIGNALS, map(float, w))),
        "sigma": sigma,
        "model_only_sigma": model_sigma,
        "market_max_weight": cap,
        "training": {"games": int(len(t)), "season": 2025, "model_variant": "pbp",
                     "market": "pregame displayed lines at each game's forecast cutoff (data/market_history.csv)",
                     "external": "archived public prediction-tracker average (data/archive/matched_predictions.csv: lineavg)"},
        "cross_validated": {
            "blend": _metrics(oos_m, oos_p, y, hw),
            "model_only": _metrics(X[:, 0], t.model_p.to_numpy(float), y, hw),
            "market_only": _metrics(X[:, 1], norm.cdf(X[:, 1] / fit_sigma(X[:, 1], hw)), y, hw),
            "external_only": _metrics(X[:, 2], norm.cdf(X[:, 2] / fit_sigma(X[:, 2], hw)), y, hw),
            "folds": folds, "blocking": "season week",
        },
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(spec, indent=2))
    log.info("blend weights %s, sigma %.2f (cv margin MAE %.2f vs model %.2f)", spec["weights"], sigma,
             spec["cross_validated"]["blend"]["margin_mae"], spec["cross_validated"]["model_only"]["margin_mae"])
    return spec


# ---------------------------------------------------------------- applying
def load_blend(path=DEFAULT_PATH):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else None


def external_margins(data_root, games, as_of, sigma):
    """Mean external projected home margin per game from picks published before the cutoff."""
    path = Path(data_root) / "sources" / "picks.csv"
    cols = ["game_id", "external_home_margin", "external_sources"]
    if not path.exists():
        return pd.DataFrame(columns=cols)
    p = pd.read_csv(path)
    p["retrieved_at"] = pd.to_datetime(p.retrieved_at, utc=True, format="mixed")
    p = p.drop(columns=["kickoff"], errors="ignore").merge(games[["game_id", "kickoff"]], on="game_id")
    p = p[(p.retrieved_at <= as_of) & (p.retrieved_at < p.kickoff)]
    if p.empty:
        return pd.DataFrame(columns=cols)
    p = p.sort_values("retrieved_at").drop_duplicates(["source", "game_id"], keep="last")
    margin = p.pick_home_score - p.pick_away_score
    margin = margin.fillna(-p.pick_home_spread)
    prob = p.pick_home_win_prob.clip(0.01, 0.99)
    margin = margin.fillna(pd.Series(sigma * norm.ppf(prob), index=p.index))
    p = p.assign(margin=margin).dropna(subset=["margin"])
    g = p.groupby("game_id")
    return pd.DataFrame({"external_home_margin": g.margin.mean(), "external_sources": g.source.apply(lambda s: ";".join(sorted(s)))}).reset_index()


def market_margins(market_path, games, as_of):
    from .market import read_market_history

    cols = ["game_id", "market_home_margin", "market_provider", "market_observed_at"]
    if not market_path or not Path(market_path).exists():
        return pd.DataFrame(columns=cols)
    m = read_market_history(market_path)
    if m.empty or "home_spread" not in m:
        return pd.DataFrame(columns=cols)
    m = m.merge(games[["game_id", "kickoff"]], on="game_id")
    m = m[(m.observed_at <= as_of) & (m.observed_at < m.kickoff) & m.home_spread.notna()]
    m = m.sort_values("observed_at").drop_duplicates("game_id", keep="last")
    return pd.DataFrame({"game_id": m.game_id, "market_home_margin": -m.home_spread, "market_provider": m.provider,
                         "market_observed_at": m.observed_at.map(lambda t: t.isoformat())})


def apply_blend(forecast_dir, data_root, blend_path=DEFAULT_PATH, market_path=None, as_of=None):
    """Write ``final_forecasts.csv`` and ``report.md`` next to the model forecast.

    ``as_of`` (default: now) is the information cutoff for market and external
    inputs; each must also be observed before the game's kickoff. The model part
    keeps its own forecast cutoff, which is never later than ``as_of``.
    """
    spec = load_blend(blend_path)
    if spec is None:
        raise FileNotFoundError(f"No fitted blend at {blend_path}; run `cfb-forecast blend-fit`")
    forecast_dir = Path(forecast_dir)
    f = pd.read_csv(forecast_dir / "forecasts.csv")
    f["kickoff"] = pd.to_datetime(f.kickoff, utc=True, format="mixed")
    model_as_of = pd.to_datetime(f.as_of, utc=True, format="mixed").max()
    as_of = max(pd.Timestamp(as_of).tz_convert("UTC") if as_of is not None else pd.Timestamp.now(tz="UTC"), model_as_of)
    out = f.merge(market_margins(market_path, f, as_of), on="game_id", how="left")
    out = out.merge(external_margins(data_root, f, as_of, spec["sigma"]), on="game_id", how="left")
    for col in ("market_home_margin", "external_home_margin"):
        out[col] = pd.to_numeric(out[col], errors="coerce").astype(float)
    w = spec["weights"]
    signals = {"model": out.projected_home_margin, "market": out.market_home_margin, "external": out.external_home_margin}
    # The external weight was fitted on a ~50-system average; a game covered by fewer
    # sources gets proportionally less of it (full weight from CFB_EXTERNAL_FULL_SOURCES on).
    full = max(1, int(os.environ.get("CFB_EXTERNAL_FULL_SOURCES", "3")))
    n_ext = out.external_sources.fillna("").map(lambda v: len([x for x in v.split(";") if x]))
    raw = {"model": pd.Series(w["model"], index=out.index), "market": pd.Series(w["market"], index=out.index),
           "external": w["external"] * (n_ext / full).clip(upper=1)}
    den = sum(raw[k] * signals[k].notna() for k in SIGNALS)
    for k in SIGNALS:
        out[f"weight_{k}"] = raw[k] * signals[k].notna() / den
    # Renormalizing can lift the market above its cap when other signals are missing; pull it back to the model.
    # Move the smallest amount e from market to model so that market <= cap and market <= model.
    excess = np.maximum.reduce([np.zeros(len(out)), (out.weight_market - spec["market_max_weight"]).to_numpy(),
                                ((out.weight_market - out.weight_model) / 2).to_numpy()])
    out["weight_market"] = out.weight_market - excess
    out["weight_model"] = out.weight_model + excess
    final_margin = sum(out[f"weight_{k}"] * signals[k].fillna(0) for k in SIGNALS)
    # Without any outside information the final forecast is exactly the model's own.
    model_only = out.market_home_margin.isna() & out.external_home_margin.isna()
    final_p = pd.Series(norm.cdf(final_margin / spec["sigma"]), index=out.index).where(~model_only, out.home_win_probability)
    shift = final_margin - out.projected_home_margin
    out = out.rename(columns={"home_win_probability": "model_home_win_probability", "projected_home_margin": "model_home_margin",
                              "expected_home_score": "model_home_score", "expected_away_score": "model_away_score"})
    out["home_win_probability"] = final_p
    out["projected_home_margin"] = final_margin
    out["expected_home_score"] = (out.model_home_score + shift / 2).clip(lower=0)
    out["expected_away_score"] = (out.model_away_score - shift / 2).clip(lower=0)
    out["market_shift_points"] = out.weight_market * (out.market_home_margin - out.model_home_margin)
    out["external_shift_points"] = out.weight_external * (out.external_home_margin - out.model_home_margin)
    out["blend_inputs"] = np.select([model_only, out.market_home_margin.isna(), out.external_home_margin.isna()],
                                    ["model", "model+external", "model+market"], "model+market+external")
    lead = ["game_id", "kickoff", "away_team", "home_team", "expected_away_score", "expected_home_score", "home_win_probability",
            "projected_home_margin", "projected_total", "blend_inputs", "model_home_win_probability", "model_home_margin",
            "market_home_margin", "external_home_margin", "market_shift_points", "external_shift_points",
            "weight_model", "weight_market", "weight_external", "market_provider", "market_observed_at", "external_sources"]
    out["blend_as_of"] = as_of.isoformat()
    out = out[lead + [c for c in out.columns if c not in lead]].sort_values("kickoff")
    out.to_csv(forecast_dir / "final_forecasts.csv", index=False)
    (forecast_dir / "report.md").write_text(render_report(out, forecast_dir, as_of, spec))
    return out


def _pct(x):
    return "—" if pd.isna(x) else f"{100 * x:.0f}%"


def _pts(x):
    return "—" if pd.isna(x) else f"{x:+.1f}"


def render_report(c, forecast_dir, as_of, spec):
    week = ""
    manifest = forecast_dir / "weekly_manifest.json"
    if manifest.exists():
        m = json.loads(manifest.read_text())
        m = m[0] if isinstance(m, list) else m
        week = f"Season {m.get('season')} · week {m.get('week')} · "
    w = spec["weights"]
    cv = spec["cross_validated"]
    lines = [
        "# Weekly forecast", "", f"{week}model cutoff {c.as_of.iloc[0]} · market/external cutoff {as_of.isoformat()} · {len(c)} games", "",
        f"Final forecast = model {w['model']:.0%} · market {w['market']:.0%} (cap {spec['market_max_weight']:.0%}) · "
        f"external {w['external']:.0%}, renormalized over the signals each game has.",
        f"Held-out 2025 cross-validation: margin MAE {cv['blend']['margin_mae']:.2f} (model alone {cv['model_only']['margin_mae']:.2f}), "
        f"log loss {cv['blend']['log_loss']:.3f} (model alone {cv['model_only']['log_loss']:.3f}).", "",
        f"- Market line available for {int(c.market_home_margin.notna().sum())} games; external predictions for {int(c.external_home_margin.notna().sum())}",
        "",
        "| Kickoff (UTC) | Away | Home | Forecast | Home win | Model margin | Market | External | Market shift |",
        "|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in c.itertuples():
        lines.append(f"| {r.kickoff:%a %m-%d %H:%M} | {r.away_team} | {r.home_team} | {r.expected_away_score:.0f}–{r.expected_home_score:.0f} "
                     f"| {_pct(r.home_win_probability)} | {_pts(r.model_home_margin)} | {_pts(r.market_home_margin)} "
                     f"| {_pts(r.external_home_margin)} | {_pts(r.market_shift_points)} |")
    lines += ["", "Margins are home minus away. Market shift = points the market moved the final margin away from the model. "
              "Forecasts are research outputs; no betting edge has been established.", ""]
    return "\n".join(lines)
