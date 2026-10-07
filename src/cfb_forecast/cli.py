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
    fa=sub.add_parser("forecast-advance",help="Advance a saved exact forecast state when no new result became eligible")
    fa.add_argument("--previous",required=True); fa.add_argument("--data",default="data"); fa.add_argument("--model",default="artifacts/backtest/model")
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
    hf.add_argument("--forecast",required=True); hf.add_argument("--data",default="data"); hf.add_argument("--model",default="artifacts/backtest/model")
    hf.add_argument("--out",default="artifacts/holdouts"); hf.add_argument("--batch-id",default=None); hf.add_argument("--freeze-at",default=None)
    hs=sub.add_parser("holdout-settle",help="Evaluate a frozen prospective holdout against later final outcomes")
    hs.add_argument("--holdout",required=True); hs.add_argument("--data",default="data")
    hv=sub.add_parser("holdout-verify",help="Verify immutable prospective holdout hashes")
    hv.add_argument("--holdout",required=True)
    cu=sub.add_parser("chronology-update",help="Append only games whose fixed pregame forecast cutoff has arrived")
    cu.add_argument("--data",default="data"); cu.add_argument("--model",default="artifacts/backtest/model")
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
    pr.add_argument("--data",default="data"); pr.add_argument("--model",default="artifacts/backtest/model")
    pr.add_argument("--ledger",default="artifacts/chronological_holdout"); pr.add_argument("--season",type=int,required=True)
    pr.add_argument("--now",default=None); pr.add_argument("--lead-hours",type=float,default=24.0)
    pr.add_argument("--market",default="data/market_history.csv"); pr.add_argument("--market-provider",default=None); pr.add_argument("--draws",type=int,default=50000)
    pr.add_argument("--primary-lateness-minutes",type=float,default=60.0); pr.add_argument("--skip-market-capture",action="store_true")
    wf=sub.add_parser("weekly-forecast",help="Forecast the next upcoming FBS slate")
    wf.add_argument("--data",default="data"); wf.add_argument("--model",default="artifacts/backtest/model")
    wf.add_argument("--out",default="artifacts"); wf.add_argument("--season",type=int,required=True)
    wf.add_argument("--week",type=int,default=None); wf.add_argument("--as-of",default=None)
    wf.add_argument("--market",default=None); wf.add_argument("--draws",type=int,default=50000)
    rr=sub.add_parser("results-refresh",help="Retrieve live scores from ESPN's public scoreboard API")
    rr.add_argument("--data",default="data"); rr.add_argument("--out",default="artifacts/results")
    rr.add_argument("--season",type=int,required=True); rr.add_argument("--week",type=int,default=None)
    rr.add_argument("--start",default=None); rr.add_argument("--end",default=None)
    args=parser.parse_args()
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
    elif args.command=="weekly-forecast":
        from .weekly import run_weekly_forecast
        result,manifest=run_weekly_forecast(args.data,args.model,args.out,args.season,args.week,args.as_of,args.market,args.draws)
        print(pd.DataFrame([manifest]).to_string(index=False))
        print(result[["away_team","home_team","kickoff","expected_away_score","expected_home_score","home_win_probability","projected_home_margin"]].to_string(index=False))
