import argparse
from pathlib import Path
import pandas as pd
from .paths import model_root


def main():
    parser=argparse.ArgumentParser(description="College football forecasting with reproducible cutoffs")
    sub=parser.add_subparsers(dest="command",required=True)
    fetch=sub.add_parser("fetch",help="Download free schedules and PBP summaries")
    fetch.add_argument("--data",default="data"); fetch.add_argument("--seasons",nargs="+",type=int,required=True)
    fetch.add_argument("--manifest", default=None, help="Pin source URLs and stored-file checksums to a JSON manifest")
    fetch.add_argument("--refresh",action="store_true"); fetch.add_argument("--raw-pbp",action="store_true")
    db=sub.add_parser("update-database", help="Validate local inputs and update the shared schedule and source catalog")
    db.add_argument("--data", default="data"); db.add_argument("--artifacts", default="artifacts")
    db.add_argument("--database", default=None); db.add_argument("--season", type=int, required=True)
    db.add_argument("--fetch-sources", action="store_true")
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
    fc.add_argument("--forecast-mode", choices=["football", "market_conditioned"], default="football", help="Football-only estimator inputs by default; quotes remain available for comparison")
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
    mb=sub.add_parser("market-backfill-sdv",help="Download historical SportsDataverse closing lines; forecasts read them only for games already played")
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
    jh=sub.add_parser("joint-holdout",help="Replay 2023-2025 week by week with the joint player model, calibrate score noise and tune prior settings on earlier seasons")
    jh.add_argument("--data",default="data"); jh.add_argument("--database",default=None)
    jh.add_argument("--out",default="artifacts/joint_holdout")
    jh.add_argument("--betting",nargs="*",default=[],help="SportsDataverse betting_YYYY.csv files for the closing-spread comparison")
    jh.add_argument("--legacy-data",default=None,help="Data folder with the season before the first fold's warmup, for the previous team-only model")
    jh.add_argument("--workers",type=int,default=1); jh.add_argument("--lead-hours",type=float,default=48.)
    jh.add_argument("--no-tune",action="store_true")
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
    cn=sub.add_parser("capture-news",help="Capture ESPN's college-football news feed and flag injury and roster items")
    cn.add_argument("--team-ids",nargs="*",type=int,default=None); cn.add_argument("--out",required=True)
    cn.add_argument("--observed-at",default=None); cn.add_argument("--timeout",type=float,default=15.0)
    cr=sub.add_parser("capture-historical-rosters",help="Download SportsDataverse rosters for every season in a range")
    cr.add_argument("--start-season",type=int,required=True); cr.add_argument("--end-season",type=int,required=True)
    cr.add_argument("--out",required=True); cr.add_argument("--cache-dir",default="data/rosters"); cr.add_argument("--timeout",type=float,default=60.0)
    sp=sub.add_parser("season-ppg",help="Team season-to-date points scored and allowed per game at a cutoff")
    sp.add_argument("--data",default="data"); sp.add_argument("--season",type=int,required=True); sp.add_argument("--as-of",required=True)
    sp.add_argument("--out",required=True); sp.add_argument("--attach-forecasts",default=None); sp.add_argument("--include-fcs",action="store_true")
    pp=sub.add_parser("player-ppg",help="Player season-to-date points per game from a player-game stats file")
    pp.add_argument("--player-games",required=True); pp.add_argument("--as-of",required=True); pp.add_argument("--season",type=int,default=None)
    pp.add_argument("--out",required=True); pp.add_argument("--min-games",type=int,default=1)
    for command in (fc, wf, sr):
        temporal = command.add_mutually_exclusive_group()
        temporal.add_argument("--temporal-features", dest="use_temporal_features", action="store_true", default=None, help="Fit lag, rolling-window and expanding history variables")
        temporal.add_argument("--no-temporal-features", dest="use_temporal_features", action="store_false", help="Ablate temporal variables while retaining team/player season rates")
        command.add_argument("--adjustment-ridge", type=float, default=None, help="Positive ridge penalty for the joint score residual model")
        command.add_argument("--adjustment-min-games", type=int, default=None, help="Minimum prior-season examples for adjustment fitting (at least 30)")
        command.add_argument("--adjustment-min-coverage", type=float, default=None, help="Observed fraction required for each learned feature, in (0,1]")
    for command_parser in (fc, fa, wf, sb):
        command_parser.add_argument("--legacy-score", action="store_true", help="Explicitly reproduce the previous team-only score head")
    pi = sub.add_parser("import-player-evidence", help="Import identified, timestamped player observations and lineups into the joint forecast database")
    pi.add_argument("--input", required=True)
    pi.add_argument("--database", default=None)
    ev=sub.add_parser("evaluate-run",help="Verify frozen forecast inputs and settle scores, winners and errors")
    ev.add_argument("--database",required=True); ev.add_argument("--run-id",required=True)
    ev.add_argument("--evidence-dir",default=None,help="Content-addressed input directory after moving a saved run")
    ev.add_argument("--data",default="data"); ev.add_argument("--as-of",required=True); ev.add_argument("--out",required=True)
    args=parser.parse_args()
    overrides = {key: getattr(args, field) for field, key in
                 (("adjustment_ridge", "ridge"), ("adjustment_min_games", "min_games"),
                  ("adjustment_min_coverage", "min_feature_coverage"), ("use_temporal_features", "use_temporal_features"))
                 if getattr(args, field, None) is not None}
    model_options = {"adjustment_config": overrides} if overrides else {}
    from .paths import cli_path
    for name in ("data", "artifacts", "model", "models", "out", "store", "ledger", "predictions", "archive", "market", "input", "previous", "forecast", "holdout", "frozen_batch", "run", "web", "database", "manifest"):
        value = getattr(args, name, None)
        if value is not None:
            setattr(args, name, str(cli_path(value)))
    if args.command == "import-player-evidence":
        import json
        from .ingest.player_matchups import import_player_evidence
        from .paths import database_path
        report = import_player_evidence(args.database or database_path(), json.loads(Path(args.input).read_text()))
        print(json.dumps(report, indent=2))
    if args.command=="evaluate-run":
        import json
        from contextlib import closing
        from .ingest.data import load_data
        from .storage.database import connect
        from .evaluation.evidence import settle_evidence
        schedules, _, _ = load_data(args.data)
        with closing(connect(args.database)) as conn:
            _, report = settle_evidence(conn,args.run_id,schedules,args.as_of,args.out,args.evidence_dir)
        print(json.dumps(report,indent=2))
    elif args.command=="fetch":
        from .ingest.data import fetch as run
        run(args.data,args.seasons,args.refresh,args.raw_pbp,manifest_path=args.manifest)
    elif args.command=="update-database":
        import json
        import os
        from .pipeline.database_update import update_database
        database = args.database or os.environ.get("CFB_DATABASE") or str(Path(args.artifacts) / "cfb_forecast.sqlite")
        print(json.dumps(update_database(args.data, database, args.season, args.artifacts, args.fetch_sources), indent=2))
    elif args.command=="backtest":
        from .evaluation.backtest import run_backtest
        result,spec=run_backtest(args.data,args.out,args.archive,args.warmup,args.train_start,args.validation,args.test)
        print(result.to_string(index=False)); print("selected production variant:",spec["selected_variant"])
    elif args.command=="forecast":
        from .modeling.forecast import forecast
        result=forecast(args.data,args.model,args.out,args.as_of,args.through,args.season,args.market,args.draws,args.market_provider,**model_options)
        result=forecast(args.data,args.model,args.out,args.as_of,args.through,args.season,args.market,args.draws,args.market_provider,legacy_score=args.legacy_score)
        result=forecast(args.data,args.model,args.out,args.as_of,args.through,args.season,args.market,args.draws,args.market_provider,forecast_mode=args.forecast_mode)
        print(result[["away_team","home_team","expected_away_score","expected_home_score","home_win_probability","projected_home_margin"]].to_string(index=False))
    elif args.command=="forecast-advance":
        from .modeling.forecast import advance_forecast_snapshot
        result=advance_forecast_snapshot(args.previous,args.data,args.model,args.out,args.as_of,args.through,args.season,args.market,args.draws,args.market_provider,legacy_score=args.legacy_score)
        print(result[["away_team","home_team","expected_away_score","expected_home_score","home_win_probability","projected_home_margin"]].to_string(index=False))
    elif args.command=="market-import":
        from .odds.market import import_market_file
        result=import_market_file(args.input,args.store,args.provider,args.source)
        print(f"market snapshots: {len(result)} -> {args.store}")
    elif args.command=="market-init":
        from .odds.market import empty_market_history
        print(empty_market_history(args.store))
    elif args.command=="market-snapshot-sdv":
        from .odds.market import capture_sportsdataverse_snapshot
        r=capture_sportsdataverse_snapshot(args.season,args.store,args.observed_at); print(f"market snapshots: {len(r)} -> {args.store}")
    elif args.command=="market-backfill-sdv":
        from .odds.market import backfill_sportsdataverse_closing
        r=backfill_sportsdataverse_closing(args.seasons,args.out); print(f"closing-market rows: {len(r)} -> {args.out}")
    elif args.command=="distribution-backtest":
        from .modeling.distribution import validate_heldout_distribution
        import json
        preds=pd.read_csv(args.predictions); models={}
        for v in ["score","pbp"]:
            models[v]=json.loads((Path(args.models)/v/"heldout"/"model.json").read_text())
        r=validate_heldout_distribution(preds,models); Path(args.out).parent.mkdir(parents=True,exist_ok=True); r.to_csv(args.out,index=False); print(r.to_string(index=False))
    elif args.command=="holdout-freeze":
        from .evaluation.holdout import freeze_holdout
        path,receipt=freeze_holdout(args.forecast,args.data,args.model,args.out,args.batch_id,args.freeze_at)
        print(path); print(pd.Series(receipt).to_string())
    elif args.command=="holdout-settle":
        from .evaluation.holdout import settle_holdout
        _,summary=settle_holdout(args.holdout,args.data)
        print(pd.Series(summary).to_string())
    elif args.command=="holdout-verify":
        from .evaluation.holdout import verify_holdout
        print(pd.Series(verify_holdout(args.holdout)).to_string())
    elif args.command=="chronology-update":
        from .evaluation.chronology import run_chronological_update
        ledger,added,receipt=run_chronological_update(args.data,args.model,args.ledger,args.season,args.now,args.lead_hours,args.market,args.draws,args.market_provider,args.primary_lateness_minutes)
        if len(added):
            print(added[["away_team","home_team","kickoff","scheduled_cutoff","actual_forecast_at","actual_lead_hours","cutoff_delay_minutes","primary_cutoff_eligible","market_status"]].to_string(index=False))
        print(pd.Series(receipt).to_string())
    elif args.command=="chronology-verify":
        from .evaluation.chronology import verify_chronological_ledger
        print(pd.Series(verify_chronological_ledger(args.ledger)).to_string())
    elif args.command=="chronology-settle":
        from .evaluation.chronology import settle_chronological_ledger
        _,summary=settle_chronological_ledger(args.ledger,args.data,args.settled_at)
        print(pd.Series(summary).to_string())
    elif args.command=="chronology-bootstrap":
        from .evaluation.chronology import bootstrap_from_frozen_batch
        ledger,manifest=bootstrap_from_frozen_batch(args.frozen_batch,args.ledger,args.lead_hours,args.primary_lateness_minutes)
        print(ledger[["away_team","home_team","kickoff","scheduled_cutoff","actual_forecast_at","actual_lead_hours","cutoff_delay_minutes","primary_cutoff_eligible","market_status"]].to_string(index=False))
        print(f"chronological games: {len(ledger)}")
    elif args.command=="prospective-run":
        from .ingest.collector import run_prospective_collection
        ledger,added,receipt=run_prospective_collection(args.data,args.model,args.ledger,args.season,args.now,args.lead_hours,args.market,args.market_provider,args.draws,args.primary_lateness_minutes,not args.skip_market_capture)
        if len(added):
            print(added[["away_team","home_team","kickoff","scheduled_cutoff","actual_forecast_at","actual_lead_hours","cutoff_delay_minutes","primary_cutoff_eligible","market_status"]].to_string(index=False))
        print(pd.Series(receipt).to_string())
    elif args.command=="results-refresh":
        from .ingest.results import refresh_results
        results,manifest=refresh_results(args.data,args.out,args.season,args.week,args.start,args.end)
        print(pd.DataFrame([manifest]).to_string(index=False))
        print(results[["game_id","status","is_final","away_team","away_points","home_team","home_points"]].tail(20).to_string(index=False))
    elif args.command=="joint-holdout":
        from .evaluation.joint_holdout import FOLDS, legacy_holdout_predictions, run_chronological_holdout
        from .paths import database_path
        legacy=None
        if args.legacy_data:
            legacy={test:legacy_holdout_predictions(args.legacy_data,args.out,test) for _,test in FOLDS}
        summary,_=run_chronological_holdout(args.data,args.database or database_path(),args.out,args.betting,legacy,
                                            workers=args.workers,lead_hours=args.lead_hours,tune=not args.no_tune)
        print(summary.to_string(index=False))
    elif args.command=="season-backtest":
        from .evaluation.season_backtest import run_season_backtest
        metrics,summary=run_season_backtest(args.data,args.model,args.out,args.season,args.draws,args.lead_hours,legacy_score=args.legacy_score)
        print(pd.DataFrame([summary]).to_string(index=False))
        print(metrics.to_string(index=False))
    elif args.command=="preflight":
        from .pipeline.preflight import run_preflight
        run_preflight(args.data,args.artifacts,args.season)
    elif args.command=="probability-audit":
        from .modeling.probability import audit_probabilities
        import json
        summary=audit_probabilities(args.input,args.out,args.probability_column,args.outcome_column,args.min_games,args.regularization,args.model,args.max_change,args.stress_radius)
        print(json.dumps(summary,indent=2,default=str))
    elif args.command=="publish-update":
        from .pipeline.publisher import publish_forecast_update
        report = publish_forecast_update(args.run, args.artifacts, args.database, args.web, args.run_id, fetch_sources=args.fetch_sources)
        print(pd.Series(report).to_string())
    elif args.command == "sunday-run":
        from .pipeline.sunday import sunday_run
        report = sunday_run(args.data, args.artifacts, args.season, args.model, args.timezone, args.scheduled, not args.no_refresh, **model_options)
        print(pd.Series(report).to_string())
    elif args.command == "weekly-forecast":
        from .pipeline.weekly import run_weekly_forecast
        result,manifest=run_weekly_forecast(args.data,args.model,args.out,args.season,args.week,args.as_of,args.market,args.draws,**model_options)
        result,manifest=run_weekly_forecast(args.data,args.model,args.out,args.season,args.week,args.as_of,args.market,args.draws,legacy_score=args.legacy_score)
        print(pd.DataFrame([manifest]).to_string(index=False))
        print(result[["away_team","home_team","kickoff","expected_away_score","expected_home_score","home_win_probability","projected_home_margin"]].to_string(index=False))
    elif args.command == "capture-news":
        from .ingest.capture_log import CAPTURE_LOG_FILENAME, append_capture
        from .ingest.news import fetch_news, fetch_news_for_teams, summarize_captures

        if args.team_ids:
            captures = fetch_news_for_teams(args.team_ids, observed_at=args.observed_at, timeout=args.timeout)
        else:
            captures = [fetch_news(observed_at=args.observed_at, timeout=args.timeout)]
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        frames = [capture.to_frame() for capture in captures]
        combined = pd.concat(frames, ignore_index=True) if any(not f.empty for f in frames) else frames[0]
        combined.to_csv(out / "news.csv", index=False)
        for capture in captures:
            append_capture(
                out / CAPTURE_LOG_FILENAME,
                source="news.espn.cfb",
                status=capture.status,
                observed_at=capture.observed_at,
                request_url=capture.request_url,
                extra={"team_id": capture.team_id, "items": capture.count, "injury_items": capture.injury_count},
            )
        print(pd.Series(summarize_captures(captures)).to_string())
    elif args.command == "capture-historical-rosters":
        from .ingest.capture_log import CAPTURE_LOG_FILENAME, append_capture
        from .ingest.rosters import fetch_historical_rosters_span, summarize_captures

        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        captures = fetch_historical_rosters_span(args.start_season, args.end_season, cache_dir=args.cache_dir, timeout=args.timeout)
        frames = [capture.to_frame() for capture in captures]
        combined = pd.concat(frames, ignore_index=True) if any(not f.empty for f in frames) else frames[0]
        combined.to_csv(out / "rosters_historical.csv", index=False)
        for capture in captures:
            append_capture(
                out / CAPTURE_LOG_FILENAME,
                source="rosters.sportsdataverse.historical",
                status=capture.status,
                observed_at=capture.observed_at,
                request_url=capture.request_url,
                extra={"season": capture.season, "player_count": capture.count},
            )
        print(pd.Series(summarize_captures(captures)).to_string())
    elif args.command == "season-ppg":
        from .ingest.capture_log import CAPTURE_LOG_FILENAME, append_capture
        from .ingest.data import load_data
        from .ingest.season_ppg import attach_ppg_to_forecasts, compute_season_ppg

        schedules, _, _ = load_data(args.data)
        ppg = compute_season_ppg(schedules, season=args.season, as_of=args.as_of, fbs_only=not args.include_fcs)
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        ppg.to_csv(out / "season_ppg.csv", index=False)
        append_capture(
            out / CAPTURE_LOG_FILENAME,
            source="season_ppg.local",
            status="captured" if len(ppg) else "no_games_before_cutoff",
            observed_at=args.as_of,
            extra={"season": args.season, "teams": int(len(ppg))},
        )
        if args.attach_forecasts:
            attach_ppg_to_forecasts(pd.read_csv(args.attach_forecasts), ppg).to_csv(out / "forecasts_with_ppg.csv", index=False)
        print(f"PPG for {len(ppg)} teams written to {out / 'season_ppg.csv'}")
    elif args.command == "player-ppg":
        from .ingest.capture_log import CAPTURE_LOG_FILENAME, append_capture
        from .ingest.player_ppg import compute_player_ppg, summarize_player_ppg

        source = Path(args.player_games)
        player_games = pd.read_parquet(source) if source.suffix.lower() == ".parquet" else pd.read_csv(source)
        result = compute_player_ppg(player_games, as_of=args.as_of, season=args.season, min_games=args.min_games)
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        result.to_csv(out / "player_ppg.csv", index=False)
        append_capture(
            out / CAPTURE_LOG_FILENAME,
            source="player_ppg.local",
            status="captured" if len(result) else "no_player_rows_before_cutoff",
            observed_at=args.as_of,
            extra={"season": args.season, "players": int(len(result)), "source_file": str(source)},
        )
        summary = summarize_player_ppg(result)
        print(f"PPG for {summary['players']} players written to {out / 'player_ppg.csv'}")
