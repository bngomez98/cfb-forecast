import argparse
from pathlib import Path
import pandas as pd
from .paths import model_root


def main():
    parser=argparse.ArgumentParser(description="College football forecasting with reproducible cutoffs")
    sub=parser.add_subparsers(dest="command",required=True)
    fetch=sub.add_parser("fetch",help="Download free schedules and PBP summaries")
    fetch.add_argument("--data",default="data"); fetch.add_argument("--seasons",nargs="+",type=int,required=True)
    fetch.add_argument("--refresh",action="store_true"); fetch.add_argument("--raw-pbp",action="store_true")
    bt=sub.add_parser("backtest",help="Chronological model selection and held-out evaluation")
    bt.add_argument("--data",default="data"); bt.add_argument("--out",default="artifacts/training")
    bt.add_argument("--archive",default=None)
    bt.add_argument("--warmup",type=int,default=2021); bt.add_argument("--train-start",type=int,default=2022)
    bt.add_argument("--validation",type=int,default=2024); bt.add_argument("--test",type=int,default=2025)
    fc=sub.add_parser("forecast",help="Produce forecasts from a saved production model")
    fc.add_argument("--data",default="data"); fc.add_argument("--model",default=str(model_root()))
    fc.add_argument("--out",default="artifacts/forecast"); fc.add_argument("--as-of",required=True)
    fc.add_argument("--through",required=True); fc.add_argument("--season",type=int,required=True)
    fc.add_argument("--market",default=None); fc.add_argument("--market-provider",default=None); fc.add_argument("--draws",type=int,default=50000)
    fa=sub.add_parser("forecast-advance",help="Advance a saved exact forecast state when no new result became eligible")
    fa.add_argument("--previous",required=True); fa.add_argument("--data",default="data"); fa.add_argument("--model",default=str(model_root()))
    fa.add_argument("--out",required=True); fa.add_argument("--as-of",required=True); fa.add_argument("--through",required=True); fa.add_argument("--season",type=int,required=True)
    fa.add_argument("--market",default=None); fa.add_argument("--market-provider",default=None); fa.add_argument("--draws",type=int,default=50000)
    mi=sub.add_parser("market-import",help="Append timestamped market snapshots to the canonical market-history store")
    mi.add_argument("--input",required=True); mi.add_argument("--store",default="data/market_history.csv")
    mi.add_argument("--provider",default="unspecified"); mi.add_argument("--source",default=None)
    me=sub.add_parser("market-init",help="Create an empty canonical market-history store")
    me.add_argument("--store",default="data/market_history.csv")
    ms=sub.add_parser("market-snapshot-sdv",help="Capture the current free SportsDataverse ESPN spread/total into point-in-time history")
    ms.add_argument("--season",type=int,required=True); ms.add_argument("--store",default="data/market_history.csv"); ms.add_argument("--observed-at",default=None)
    mb=sub.add_parser("market-backfill-sdv",help="Download benchmark-only historical resolved/closing SportsDataverse lines")
    mb.add_argument("--seasons",nargs="+",type=int,required=True); mb.add_argument("--out",default="data/market_closing_history.csv")
    dv=sub.add_parser("distribution-backtest",help="Evaluate Gaussian score distributions on frozen held-out predictions")
    dv.add_argument("--predictions",default="artifacts/backtest/predictions.csv"); dv.add_argument("--models",default="artifacts/backtest/models"); dv.add_argument("--out",default="artifacts/backtest/distribution_validation.csv")
    hf=sub.add_parser("holdout-freeze",help="Freeze a forecast batch and exact inputs before kickoff")
    hf.add_argument("--forecast",required=True); hf.add_argument("--data",default="data"); hf.add_argument("--model",default=str(model_root()))
    hf.add_argument("--out",default="artifacts/holdouts"); hf.add_argument("--batch-id",default=None); hf.add_argument("--freeze-at",default=None)
    hs=sub.add_parser("holdout-settle",help="Evaluate a frozen prospective holdout against later final outcomes")
    hs.add_argument("--holdout",required=True); hs.add_argument("--data",default="data")
    hv=sub.add_parser("holdout-verify",help="Verify immutable prospective holdout hashes")
    hv.add_argument("--holdout",required=True)
    cu=sub.add_parser("chronology-update",help="Append only games whose fixed pregame forecast cutoff has arrived")
    cu.add_argument("--data",default="data"); cu.add_argument("--model",default=str(model_root()))
    cu.add_argument("--ledger",default="artifacts/chronological_holdout"); cu.add_argument("--season",type=int,required=True)
    cu.add_argument("--now",default=None); cu.add_argument("--lead-hours",type=float,default=24.0)
    cu.add_argument("--market",default=None); cu.add_argument("--market-provider",default=None); cu.add_argument("--draws",type=int,default=50000)
    cu.add_argument("--primary-lateness-minutes",type=float,default=60.0)
    cv=sub.add_parser("chronology-verify",help="Verify the chronological ledger and every immutable run")
    cv.add_argument("--ledger",default="artifacts/chronological_holdout")
    cs=sub.add_parser("chronology-settle",help="Settle every completed game in the chronological ledger")
    cs.add_argument("--ledger",default="artifacts/chronological_holdout"); cs.add_argument("--data",default="data"); cs.add_argument("--settled-at",default=None)
    cb=sub.add_parser("chronology-bootstrap",help="Start a chronological ledger from an existing frozen batch, discarding not-yet-due games")
    cb.add_argument("--frozen-batch",required=True); cb.add_argument("--ledger",default="artifacts/chronological_holdout")
    cb.add_argument("--lead-hours",type=float,default=24.0); cb.add_argument("--primary-lateness-minutes",type=float,default=60.0)
    pr=sub.add_parser("prospective-run",help="Capture market then append newly due games to the chronological prospective holdout")
    pr.add_argument("--data",default="data"); pr.add_argument("--model",default=str(model_root()))
    pr.add_argument("--ledger",default="artifacts/chronological_holdout"); pr.add_argument("--season",type=int,required=True)
    pr.add_argument("--now",default=None); pr.add_argument("--lead-hours",type=float,default=24.0)
    pr.add_argument("--market",default="data/market_history.csv"); pr.add_argument("--market-provider",default=None); pr.add_argument("--draws",type=int,default=50000)
    pr.add_argument("--primary-lateness-minutes",type=float,default=60.0); pr.add_argument("--skip-market-capture",action="store_true")
    wf=sub.add_parser("weekly-forecast",help="Forecast the next upcoming FBS slate")
    wf.add_argument("--data",default="data"); wf.add_argument("--model",default=str(model_root()))
    wf.add_argument("--out",default="artifacts"); wf.add_argument("--season",type=int,required=True)
    wf.add_argument("--week",type=int,default=None); wf.add_argument("--as-of",default=None)
    wf.add_argument("--market",default=None); wf.add_argument("--draws",type=int,default=50000)
    sr=sub.add_parser("sunday-run",help="Collect sources, train, validate and publish the Sunday slate")
    sr.add_argument("--data",default="data"); sr.add_argument("--model",default=None)
    sr.add_argument("--artifacts", "--out", dest="artifacts", default="artifacts")
    sr.add_argument("--season",type=int,required=True)
    sr.add_argument("--scheduled",action="store_true")
    sr.add_argument("--no-refresh",action="store_true")
    sr.add_argument("--timezone",default="America/Chicago")
    rr=sub.add_parser("results-refresh",help="Retrieve live scores from ESPN's public scoreboard API")
    rr.add_argument("--data",default="data"); rr.add_argument("--out",default="artifacts/results")
    rr.add_argument("--season",type=int,required=True); rr.add_argument("--week",type=int,default=None)
    rr.add_argument("--start",default=None); rr.add_argument("--end",default=None)
    sb=sub.add_parser("season-backtest",help="Reconstruct and grade weekly forecasts across a season")
    sb.add_argument("--data",default="data"); sb.add_argument("--model",default=str(model_root()))
    sb.add_argument("--out",default="artifacts/backtest"); sb.add_argument("--season",type=int,required=True)
    sb.add_argument("--draws",type=int,default=50000); sb.add_argument("--lead-hours",type=float,default=48.0)
    pf=sub.add_parser("preflight",help="Validate mounted data and artifact paths")
    pf.add_argument("--data",default="data"); pf.add_argument("--artifacts",default="artifacts"); pf.add_argument("--season",type=int,default=None)
    pa=sub.add_parser("probability-audit",help="Evaluate prior-season calibration, constrained fitting, and stress sensitivity on exported predictions")
    pa.add_argument("--input",required=True); pa.add_argument("--out",required=True)
    pa.add_argument("--probability-column",default="score_p"); pa.add_argument("--outcome-column",default="home_win")
    pa.add_argument("--min-games",type=int,default=100); pa.add_argument("--regularization",type=float,default=0.01)
    pa.add_argument("--max-change",type=float,default=None); pa.add_argument("--stress-radius",type=float,default=0.1)
    pa.add_argument("--model",default=None,help="Optionally write a separate candidate model next to the audit")
    pu=sub.add_parser("publish-update",help="Publish a completed forecast directory to SQLite, React, Streamlit, logs, and email")
    pu.add_argument("--run",required=True); pu.add_argument("--artifacts",default="artifacts")
    pu.add_argument("--database",default=None); pu.add_argument("--web",default=None); pu.add_argument("--run-id",default=None)
    pu.add_argument("--fetch-sources",action="store_true")
    pg=sub.add_parser("player-aggregates",help="Build per-player-game aggregates from raw play-by-play for the hierarchical player layer")
    pg.add_argument("--data",default="data"); pg.add_argument("--seasons",nargs="+",type=int,required=True)
    pg.add_argument("--pbp-dir",default=None); pg.add_argument("--refresh",action="store_true")
    sub.add_parser("sources-list",help="List the registered external web sources")
    sc=sub.add_parser("sources-collect",help="Fetch external web sources, interpret them and publish reconciled outputs")
    sc.add_argument("--data",default="data"); sc.add_argument("--artifacts",default="artifacts"); sc.add_argument("--season",type=int,required=True)
    sc.add_argument("--source",nargs="+",default=None,help="Source names (default: all)"); sc.add_argument("--tag",default=None,choices=["results","picks","lines","stats"])
    sc.add_argument("--week",type=int,default=None); sc.add_argument("--offline",action="store_true",help="Re-interpret the latest stored snapshots instead of fetching")
    sc.add_argument("--market",default=None,help="Market history store (default <data>/market_history.csv)")
    si=sub.add_parser("sources-import",help="Interpret a saved page, export or PDF as a source snapshot")
    si.add_argument("--file",required=True); si.add_argument("--source",required=True); si.add_argument("--season",type=int,required=True)
    si.add_argument("--data",default="data"); si.add_argument("--artifacts",default="artifacts"); si.add_argument("--market",default=None)
    si.add_argument("--observed-at",default=None,help="When the content was published/captured (UTC offset required); default now")
    hc=sub.add_parser("healthcheck",help="Exit 0 if the scheduler heartbeat is fresh (container health)")
    hc.add_argument("--artifacts",default="artifacts"); hc.add_argument("--max-age",type=int,default=600)
    from .app import add_arguments as app_arguments
    app_arguments(sub.add_parser("app",help="Run the whole application: scheduler (Sunday + daily forecast) plus dashboard"))
    from .scheduler import add_arguments as scheduler_arguments
    scheduler_arguments(sub.add_parser("scheduler",help="Run pipeline jobs automatically from the game calendar"))
    from .pipeline import add_arguments as pipeline_arguments
    pipeline_arguments(sub.add_parser("pipeline",help="Run an end-to-end pipeline: forecast, results, sources, backtest or data"))
    args=parser.parse_args()
    from .paths import cli_path
    for name in ("data", "artifacts", "pbp_dir", "model", "models", "out", "store", "ledger", "predictions", "archive", "market", "input", "previous", "forecast", "holdout", "frozen_batch", "run", "web", "database"):
        value = getattr(args, name, None)
        if value is not None:
            setattr(args, name, str(cli_path(value)))
    from .logs import configure
    configure()
    if args.command=="player-aggregates":
        from .players import build_aggregates
        print(build_aggregates(args.data,args.seasons,args.pbp_dir,args.refresh)); return
    if args.command=="sources-list":
        from .webdata import list_sources
        print(list_sources().to_string(index=False)); return
    if args.command=="sources-collect":
        from .webdata import collect
        report=collect(args.data,args.season,args.artifacts,args.source,args.tag,args.week,args.offline,args.market)
        rows=[{"source":k,**{f:v.get(f) for f in ("status","reason","results","picks","lines","team_stats")}} for k,v in report["sources"].items()]
        print(pd.DataFrame(rows).fillna("").to_string(index=False))
        print({k:v for k,v in report.items() if k not in ("sources",)}); return
    if args.command=="sources-import":
        from .webdata import import_file
        print(pd.Series(import_file(args.file,args.source,args.data,args.season,args.artifacts,args.observed_at,args.market)).to_string()); return
    if args.command=="healthcheck":
        from .scheduler import healthcheck
        raise SystemExit(healthcheck(args.artifacts,args.max_age))
    if args.command=="app":
        from .app import run_from_args as app_run
        raise SystemExit(app_run(args))
    if args.command=="scheduler":
        from .scheduler import run_from_args as scheduler_run
        raise SystemExit(scheduler_run(args))
    if args.command=="pipeline":
        from .pipeline import run_from_args
        raise SystemExit(run_from_args(args))
    if args.command=="fetch":
        from .data import fetch as run
        run(args.data,args.seasons,args.refresh,args.raw_pbp)
    elif args.command=="backtest":
        from .backtest import run_backtest
        result,spec=run_backtest(args.data,args.out,args.archive,args.warmup,args.train_start,args.validation,args.test)
        print(result.to_string(index=False)); print("selected production variant:",spec["selected_variant"])
    elif args.command=="forecast":
        from .forecast import forecast
        result=forecast(args.data,args.model,args.out,args.as_of,args.through,args.season,args.market,args.draws,args.market_provider)
        print(result[["away_team","home_team","expected_away_score","expected_home_score","home_win_probability","projected_home_margin"]].to_string(index=False))
    elif args.command=="forecast-advance":
        from .forecast import advance_forecast_snapshot
        result=advance_forecast_snapshot(args.previous,args.data,args.model,args.out,args.as_of,args.through,args.season,args.market,args.draws,args.market_provider)
        print(result[["away_team","home_team","expected_away_score","expected_home_score","home_win_probability","projected_home_margin"]].to_string(index=False))
    elif args.command=="market-import":
        from .market import import_market_file
        result=import_market_file(args.input,args.store,args.provider,args.source)
        print(f"market snapshots: {len(result)} -> {args.store}")
    elif args.command=="market-init":
        from .market import empty_market_history
        print(empty_market_history(args.store))
    elif args.command=="market-snapshot-sdv":
        from .market import capture_sportsdataverse_snapshot
        r=capture_sportsdataverse_snapshot(args.season,args.store,args.observed_at); print(f"market snapshots: {len(r)} -> {args.store}")
    elif args.command=="market-backfill-sdv":
        from .market import backfill_sportsdataverse_closing
        r=backfill_sportsdataverse_closing(args.seasons,args.out); print(f"closing-market rows: {len(r)} -> {args.out}")
    elif args.command=="distribution-backtest":
        from .distribution import validate_heldout_distribution
        import json
        preds=pd.read_csv(args.predictions); models={}
        for v in ["score","pbp"]:
            models[v]=json.loads((Path(args.models)/v/"heldout"/"model.json").read_text())
        r=validate_heldout_distribution(preds,models); Path(args.out).parent.mkdir(parents=True,exist_ok=True); r.to_csv(args.out,index=False); print(r.to_string(index=False))
    elif args.command=="holdout-freeze":
        from .holdout import freeze_holdout
        path,receipt=freeze_holdout(args.forecast,args.data,args.model,args.out,args.batch_id,args.freeze_at)
        print(path); print(pd.Series(receipt).to_string())
    elif args.command=="holdout-settle":
        from .holdout import settle_holdout
        _,summary=settle_holdout(args.holdout,args.data)
        print(pd.Series(summary).to_string())
    elif args.command=="holdout-verify":
        from .holdout import verify_holdout
        print(pd.Series(verify_holdout(args.holdout)).to_string())
    elif args.command=="chronology-update":
        from .chronology import run_chronological_update
        ledger,added,receipt=run_chronological_update(args.data,args.model,args.ledger,args.season,args.now,args.lead_hours,args.market,args.draws,args.market_provider,args.primary_lateness_minutes)
        if len(added):
            print(added[["away_team","home_team","kickoff","scheduled_cutoff","actual_forecast_at","actual_lead_hours","cutoff_delay_minutes","primary_cutoff_eligible","market_status"]].to_string(index=False))
        print(pd.Series(receipt).to_string())
    elif args.command=="chronology-verify":
        from .chronology import verify_chronological_ledger
        print(pd.Series(verify_chronological_ledger(args.ledger)).to_string())
    elif args.command=="chronology-settle":
        from .chronology import settle_chronological_ledger
        _,summary=settle_chronological_ledger(args.ledger,args.data,args.settled_at)
        print(pd.Series(summary).to_string())
    elif args.command=="chronology-bootstrap":
        from .chronology import bootstrap_from_frozen_batch
        ledger,manifest=bootstrap_from_frozen_batch(args.frozen_batch,args.ledger,args.lead_hours,args.primary_lateness_minutes)
        print(ledger[["away_team","home_team","kickoff","scheduled_cutoff","actual_forecast_at","actual_lead_hours","cutoff_delay_minutes","primary_cutoff_eligible","market_status"]].to_string(index=False))
        print(f"chronological games: {len(ledger)}")
    elif args.command=="prospective-run":
        from .collector import run_prospective_collection
        ledger,added,receipt=run_prospective_collection(args.data,args.model,args.ledger,args.season,args.now,args.lead_hours,args.market,args.market_provider,args.draws,args.primary_lateness_minutes,not args.skip_market_capture)
        if len(added):
            print(added[["away_team","home_team","kickoff","scheduled_cutoff","actual_forecast_at","actual_lead_hours","cutoff_delay_minutes","primary_cutoff_eligible","market_status"]].to_string(index=False))
        print(pd.Series(receipt).to_string())
    elif args.command=="results-refresh":
        from .results import refresh_results
        results,manifest=refresh_results(args.data,args.out,args.season,args.week,args.start,args.end)
        print(pd.DataFrame([manifest]).to_string(index=False))
        print(results[["game_id","status","is_final","away_team","away_points","home_team","home_points"]].tail(20).to_string(index=False))
    elif args.command=="season-backtest":
        from .season_backtest import run_season_backtest
        metrics,summary=run_season_backtest(args.data,args.model,args.out,args.season,args.draws,args.lead_hours)
        print(pd.DataFrame([summary]).to_string(index=False))
        print(metrics.to_string(index=False))
    elif args.command=="preflight":
        from .preflight import run_preflight
        run_preflight(args.data,args.artifacts,args.season)
    elif args.command=="probability-audit":
        from .probability import audit_probabilities
        import json
        summary=audit_probabilities(args.input,args.out,args.probability_column,args.outcome_column,args.min_games,args.regularization,args.model,args.max_change,args.stress_radius)
        print(json.dumps(summary,indent=2,default=str))
    elif args.command=="publish-update":
        from .publisher import publish_forecast_update
        report = publish_forecast_update(args.run, args.artifacts, args.database, args.web, args.run_id, fetch_sources=args.fetch_sources)
        print(pd.Series(report).to_string())
    elif args.command == "sunday-run":
        from .sunday import sunday_run
        report = sunday_run(args.data, args.artifacts, args.season, args.model, args.timezone, args.scheduled, not args.no_refresh)
        print(pd.Series(report).to_string())
    elif args.command == "weekly-forecast":
        from .weekly import run_weekly_forecast
        result,manifest=run_weekly_forecast(args.data,args.model,args.out,args.season,args.week,args.as_of,args.market,args.draws)
        print(pd.DataFrame([manifest]).to_string(index=False))
        print(result[["away_team","home_team","kickoff","expected_away_score","expected_home_score","home_win_probability","projected_home_margin"]].to_string(index=False))
