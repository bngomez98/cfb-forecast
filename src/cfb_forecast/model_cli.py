"""Operating cycle and command line for the joint football model.

    cfb-model ingest   --season 2026            load schedules, team rates, plays, drives from SportsDataverse files
    cfb-model derive   --seasons 2022 ... 2026  derived football observations and schedule conditions
    cfb-model fit      --seasons ... --cutoff T the joint posterior and its stored estimates
    cfb-model forecast --bundle ID --start D --end D
    cfb-model evaluate                          attach finals to forecasts and score them
    cfb-model backtest --season 2025 --cutoffs ...
    cfb-model cycle    --season 2026            the whole weekly cycle in order
"""
from __future__ import annotations

import argparse
import json
from contextlib import closing
from pathlib import Path

import numpy as np
import pandas as pd

from . import football_model as fm
from . import forecasting as fc
from . import model_spec as sp
from .database import connect
from .ingest import ingest_sportsdataverse
from .observations import derive_observations, derive_schedule_conditions, load_model_data
from .paths import artifacts_root, data_root


def default_database():
    import os
    return Path(os.environ.get("CFB_DATABASE") or artifacts_root() / "cfb_forecast.sqlite3")


def games_in_window(conn, start, end, season, only_open=True):
    q = """SELECT game_id, season, week, start_date, home_id, away_id FROM games WHERE fbs_game=1 AND season=? AND start_date >= ? AND start_date < ?"""
    if only_open:
        q += " AND result_usable=0"
    return pd.read_sql_query(q, conn, params=[int(season), pd.Timestamp(start, tz="UTC").isoformat() if pd.Timestamp(start).tzinfo is None else pd.Timestamp(start).isoformat(),
                                               pd.Timestamp(end, tz="UTC").isoformat() if pd.Timestamp(end).tzinfo is None else pd.Timestamp(end).isoformat()])


def periods_of(games, seasons):
    return sorted({seasons.index(int(r.season)) * sp.WEEKS + min(max(int(r.week), 1), sp.WEEKS) - 1 for r in games.itertuples(index=False)})


def fit_model(conn, artifacts, seasons, cutoff, steps=4000, lr=0.02, keep_periods=None, n_draws=200, seed=0, log=print, method="svi", warmup=500):
    """Fit the whole joint model at the cutoff, store the estimates, and return the saved bundle."""
    data = load_model_data(conn, seasons, cutoff=cutoff)
    f = fm.infer(data, method, steps=steps, lr=lr, seed=seed, keep_periods=keep_periods, log=log, warmup=warmup, samples=n_draws)
    bundle = fc.extract_bundle(f, n_draws=n_draws, seed=seed + 7)
    manifest = {"seasons": seasons, "cutoff": str(cutoff), "steps": steps, "method": method, "final_neg_elbo": None if f.losses is None else float(f.losses[-1]),
                "retrievals": [dict(zip(("source_id", "n", "last"), r)) for r in conn.execute("SELECT source_id, COUNT(*), MAX(retrieved_at) FROM source_retrievals GROUP BY source_id")]}
    fc.store_estimates(conn, bundle, manifest)
    bundle.save(Path(artifacts) / "bundles")
    return bundle, f


def forecast_window(conn, bundle, artifacts, start, end, run_id=None):
    seasons = bundle.meta["seasons"]
    games = games_in_window(conn, start, end, seasons[-1])
    if games.empty:
        return pd.DataFrame()
    fcst = fc.Forecaster(bundle, conn)
    out = fc.forecast_games(conn, bundle, games.game_id.tolist(), artifacts, run_id=run_id, forecaster=fcst)
    fc.store_availability(conn, bundle, fcst, games.game_id.tolist())
    return out


def backtest(conn, artifacts, season, cutoffs, horizon_days=28, seasons=None, steps=3000, n_draws=120, log=print):
    """Walk forward: at each cutoff fit on what was known, forecast the next window, then score against finals."""
    seasons = seasons or list(range(season - 3, season + 1))
    results = []
    for c in cutoffs:
        c = pd.Timestamp(c, tz="UTC")
        end = c + pd.Timedelta(days=horizon_days)
        window = pd.read_sql_query("SELECT game_id, season, week, start_date, home_id, away_id FROM games WHERE fbs_game=1 AND result_usable=1 AND season=? AND start_date>=? AND start_date<?",
                                   conn, params=[season, c.isoformat(), end.isoformat()])
        if window.empty:
            continue
        keep = periods_of(window, seasons)
        log(f"cutoff {c.date()}: fitting on data before it, forecasting {len(window)} games over periods {keep}")
        bundle, _ = fit_model(conn, artifacts, seasons, c.isoformat(), steps=steps, keep_periods=keep, n_draws=n_draws, log=None)
        fcst = fc.Forecaster(bundle, conn)
        out = fc.forecast_games(conn, bundle, window.game_id.tolist(), artifacts, run_id=f"backtest-{season}-{c:%Y%m%d}", information_cutoff=c.isoformat(), forecaster=fcst)
        results.append(out)
    ev = fc.evaluate_forecasts(conn, artifacts, now="2100-01-01T00:00:00+00:00")
    return fc.evaluation_summary(ev[ev.game_id.isin(pd.concat(results).game_id)] if results else ev), ev


def main(argv=None):
    ap = argparse.ArgumentParser(prog="cfb-model", description="Joint hierarchical Bayesian football model")
    ap.add_argument("--database", default=None)
    ap.add_argument("--artifacts", default=None)
    ap.add_argument("--data", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("ingest"); p.add_argument("--season", type=int, nargs="+", required=True); p.add_argument("--no-plays", action="store_true")
    p = sub.add_parser("derive"); p.add_argument("--seasons", type=int, nargs="+", required=True)
    p = sub.add_parser("fit"); p.add_argument("--seasons", type=int, nargs="+", required=True); p.add_argument("--cutoff", default=None)
    p.add_argument("--steps", type=int, default=4000); p.add_argument("--draws", type=int, default=200); p.add_argument("--method", choices=["svi", "nuts"], default="svi")
    p = sub.add_parser("forecast"); p.add_argument("--bundle", required=True); p.add_argument("--start", required=True); p.add_argument("--end", required=True)
    sub.add_parser("evaluate")
    p = sub.add_parser("backtest"); p.add_argument("--season", type=int, required=True); p.add_argument("--cutoffs", nargs="+", required=True)
    p.add_argument("--steps", type=int, default=3000); p.add_argument("--horizon-days", type=int, default=28)
    p = sub.add_parser("cycle"); p.add_argument("--season", type=int, required=True); p.add_argument("--history", type=int, default=4)
    p.add_argument("--steps", type=int, default=4000); p.add_argument("--no-ingest", action="store_true"); p.add_argument("--method", choices=["svi", "nuts"], default="svi")
    args = ap.parse_args(argv)
    artifacts = Path(args.artifacts) if args.artifacts else artifacts_root()
    db = Path(args.database) if args.database else default_database()
    data = Path(args.data) if args.data else data_root()
    with closing(connect(db)) as conn:
        if args.cmd == "ingest":
            for s in args.season:
                print(json.dumps(ingest_sportsdataverse(conn, data, s, artifacts, include_plays=not args.no_plays)))
        elif args.cmd == "derive":
            print(json.dumps(derive_observations(conn, args.seasons)), derive_schedule_conditions(conn))
        elif args.cmd == "fit":
            keep = None
            bundle, _ = fit_model(conn, artifacts, args.seasons, args.cutoff, steps=args.steps, n_draws=args.draws, keep_periods=keep, method=args.method)
            print(bundle.parameter_set_id)
        elif args.cmd == "forecast":
            bundle = fc.Bundle.load(artifacts / "bundles", args.bundle)
            out = forecast_window(conn, bundle, artifacts, args.start, args.end)
            print(out.to_string(index=False) if len(out) else "no open FBS games in the window")
        elif args.cmd == "evaluate":
            ev = fc.evaluate_forecasts(conn, artifacts)
            print(json.dumps(fc.evaluation_summary(ev), indent=2, default=float))
        elif args.cmd == "backtest":
            summary, _ = backtest(conn, artifacts, args.season, args.cutoffs, args.horizon_days, steps=args.steps)
            print(json.dumps(summary, indent=2, default=float))
        elif args.cmd == "cycle":
            seasons = list(range(args.season - args.history + 1, args.season + 1))
            if not args.no_ingest:
                for s in seasons:
                    ingest_sportsdataverse(conn, data, s, artifacts)
            derive_observations(conn, seasons)
            derive_schedule_conditions(conn)
            now = pd.Timestamp.now(tz="UTC")
            nxt = games_in_window(conn, now, now + pd.Timedelta(days=7), args.season)
            keep = periods_of(nxt, seasons) if len(nxt) else None
            bundle, _ = fit_model(conn, artifacts, seasons, now.isoformat(), steps=args.steps, keep_periods=keep, method=args.method)
            out = forecast_window(conn, bundle, artifacts, now, now + pd.Timedelta(days=7))
            fc.evaluate_forecasts(conn, artifacts)
            print(out.to_string(index=False) if len(out) else "no open FBS games in the next 7 days")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
