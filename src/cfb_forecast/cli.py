import argparse
from pathlib import Path
import pandas as pd


def main():
    parser=argparse.ArgumentParser(description="College football forecasting with reproducible cutoffs")
    sub=parser.add_subparsers(dest="command",required=True)
    fetch=sub.add_parser("fetch",help="Download free schedules and PBP summaries")
    fetch.add_argument("--data",default="data"); fetch.add_argument("--seasons",nargs="+",type=int,required=True)
    fetch.add_argument("--refresh",action="store_true"); fetch.add_argument("--raw-pbp",action="store_true")
    bt=sub.add_parser("backtest",help="Chronological model selection and held-out evaluation")
    bt.add_argument("--data",default="data"); bt.add_argument("--out",default="artifacts/backtest")
    bt.add_argument("--archive",default=None)
    bt.add_argument("--warmup",type=int,default=2021); bt.add_argument("--train-start",type=int,default=2022)
    bt.add_argument("--validation",type=int,default=2024); bt.add_argument("--test",type=int,default=2025)
    fc=sub.add_parser("forecast",help="Produce forecasts from a saved production model")
    fc.add_argument("--data",default="data"); fc.add_argument("--model",default="artifacts/backtest/model")
    fc.add_argument("--out",default="artifacts/forecast"); fc.add_argument("--as-of",required=True)
    fc.add_argument("--through",required=True); fc.add_argument("--season",type=int,required=True)
    fc.add_argument("--market",default=None); fc.add_argument("--market-provider",default=None); fc.add_argument("--draws",type=int,default=50000)
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
    wf=sub.add_parser("weekly-forecast",help="Forecast the next upcoming FBS slate")
    wf.add_argument("--data",default="data"); wf.add_argument("--model",default="artifacts/backtest/model")
    wf.add_argument("--out",default="artifacts"); wf.add_argument("--season",type=int,required=True)
    wf.add_argument("--week",type=int,default=None); wf.add_argument("--as-of",default=None)
    wf.add_argument("--market",default=None); wf.add_argument("--draws",type=int,default=50000)
    rr=sub.add_parser("results-refresh",help="Retrieve final scores of completed games from ESPN's public scoreboard API")
    rr.add_argument("--data",default="data"); rr.add_argument("--out",default="artifacts/results")
    rr.add_argument("--season",type=int,required=True); rr.add_argument("--week",type=int,default=None)
    rr.add_argument("--start",default=None); rr.add_argument("--end",default=None)
    sb=sub.add_parser("season-backtest",help="Point-in-time backtest: reconstruct each week's pre-kickoff forecasts and compare with final scores")
    sb.add_argument("--data",default="data"); sb.add_argument("--model",default="artifacts/backtest/model")
    sb.add_argument("--out",default="artifacts/backtest"); sb.add_argument("--season",type=int,required=True)
    sb.add_argument("--draws",type=int,default=50000); sb.add_argument("--lead-hours",type=float,default=48.0)
    pf=sub.add_parser("preflight",help="Validate mounted data and artifact paths")
    pf.add_argument("--data",default="data"); pf.add_argument("--artifacts",default="artifacts"); pf.add_argument("--season",type=int,default=None)
    sub.add_parser("sources-list",help="List the registered external sources")
    sc=sub.add_parser("sources-collect",help="Fetch external sources, interpret them and publish reconciled outputs")
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
    from .scheduler import add_arguments as scheduler_arguments
    scheduler_arguments(sub.add_parser("scheduler",help="Run pipeline jobs automatically from the game calendar"))
    from .pipeline import add_arguments as pipeline_arguments
    pipeline_arguments(sub.add_parser("pipeline",help="Run an end-to-end pipeline: weekly, results, backtest or data"))
    args=parser.parse_args()
    from .logs import configure
    configure()
    if args.command=="sources-list":
        from .sources import list_sources
        print(list_sources().to_string(index=False)); return
    if args.command=="sources-collect":
        from .sources import collect
        report=collect(args.data,args.season,args.artifacts,args.source,args.tag,args.week,args.offline,args.market)
        rows=[{"source":k,**{f:v.get(f) for f in ("status","reason","results","picks","lines","team_stats")}} for k,v in report["sources"].items()]
        print(pd.DataFrame(rows).fillna("").to_string(index=False))
        print({k:v for k,v in report.items() if k not in ("sources",)}); return
    if args.command=="sources-import":
        from .sources import import_file
        print(pd.Series(import_file(args.file,args.source,args.data,args.season,args.artifacts,args.observed_at,args.market)).to_string()); return
    if args.command=="healthcheck":
        from .scheduler import healthcheck
        raise SystemExit(healthcheck(args.artifacts,args.max_age))
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
    elif args.command=="weekly-forecast":
        from .weekly import run_weekly_forecast
        result,manifest=run_weekly_forecast(args.data,args.model,args.out,args.season,args.week,args.as_of,args.market,args.draws)
        print(pd.DataFrame([manifest]).to_string(index=False))
        print(result[["away_team","home_team","kickoff","expected_away_score","expected_home_score","home_win_probability","projected_home_margin"]].to_string(index=False))
