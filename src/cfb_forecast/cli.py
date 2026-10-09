import argparse
from pathlib import Path

import pandas as pd

from ._logging import configure_cli_logging


def main():
    configure_cli_logging()
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
    sb=sub.add_parser("season-backtest",help="Reconstruct and grade weekly forecasts across a season")
    sb.add_argument("--data",default="data"); sb.add_argument("--model",default="artifacts/backtest/model")
    sb.add_argument("--out",default="artifacts/backtest"); sb.add_argument("--season",type=int,required=True)
    sb.add_argument("--draws",type=int,default=50000); sb.add_argument("--lead-hours",type=float,default=48.0)
    cw = sub.add_parser(
        "capture-weather",
        help="Capture pregame weather for a slate of games through Open-Meteo",
    )
    cw.add_argument("--forecasts", required=True, help="forecasts.csv from a forecast run")
    cw.add_argument("--venues", default="data/stadiums.csv")
    cw.add_argument("--out", required=True, help="Output directory for the enriched CSV and capture log")
    cw.add_argument("--timeout", type=float, default=15.0)
    ci = sub.add_parser(
        "capture-injuries",
        help="Capture ESPN injury reports for a list of team IDs",
    )
    ci.add_argument("--team-ids", nargs="+", type=int, required=True)
    ci.add_argument("--out", required=True)
    ci.add_argument("--timeout", type=float, default=15.0)
    cn = sub.add_parser(
        "capture-nfl",
        help="Capture the ESPN NFL scoreboard for a date (default: today, UTC)",
    )
    cn.add_argument("--date", default=None)
    cn.add_argument("--out", required=True)
    cn.add_argument("--timeout", type=float, default=15.0)
    cr = sub.add_parser(
        "capture-rosters",
        help="Capture current rosters from ESPN for a list of team IDs",
    )
    cr.add_argument("--team-ids", nargs="+", type=int, required=True)
    cr.add_argument("--out", required=True)
    cr.add_argument("--timeout", type=float, default=15.0)
    crh = sub.add_parser(
        "capture-historical-rosters",
        help="Download SportsDataverse's per-season roster parquet for all FBS teams",
    )
    crh.add_argument("--seasons", nargs="+", type=int, required=True)
    crh.add_argument("--out", required=True)
    crh.add_argument("--cache-dir", default="data/rosters")
    crh.add_argument("--timeout", type=float, default=60.0)
    cp = sub.add_parser(
        "season-ppg",
        help="Compute per-team season-to-date PPG scored and allowed at a cutoff",
    )
    cp.add_argument("--data", default="data")
    cp.add_argument("--season", type=int, required=True)
    cp.add_argument("--as-of", required=True)
    cp.add_argument("--out", required=True)
    cp.add_argument(
        "--attach-forecasts",
        default=None,
        help="If given, read this forecasts.csv and write a forecast file enriched with PPG columns",
    )
    cp.add_argument(
        "--include-fcs",
        action="store_true",
        help="Include FBS-vs-FCS games when computing PPG (default: FBS-only)",
    )
    cnw = sub.add_parser(
        "capture-news",
        help="Capture ESPN's public college-football news feed and label injury/roster items",
    )
    cnw.add_argument(
        "--team-ids",
        nargs="*",
        type=int,
        default=None,
        help="Optional list of team IDs; omit to capture the general feed",
    )
    cnw.add_argument("--out", required=True)
    cnw.add_argument("--injury-out", default=None,
                     help="Optional path for an injury-only CSV filtered from the capture")
    cnw.add_argument("--observed-at", default=None)
    cnw.add_argument("--timeout", type=float, default=15.0)
    chrs = sub.add_parser(
        "capture-historical-rosters-span",
        help="Download SportsDataverse rosters for every season in a range",
    )
    chrs.add_argument("--start-season", type=int, required=True)
    chrs.add_argument("--end-season", type=int, required=True)
    chrs.add_argument("--out", required=True)
    chrs.add_argument("--cache-dir", default="data/rosters")
    chrs.add_argument("--timeout", type=float, default=60.0)
    pp = sub.add_parser(
        "player-ppg",
        help="Compute per-player season-to-date PPG from a player-game stats parquet or CSV",
    )
    pp.add_argument("--player-games", required=True,
                    help="Path to a player-game stats parquet or CSV with scoring columns")
    pp.add_argument("--as-of", required=True)
    pp.add_argument("--season", type=int, default=None)
    pp.add_argument("--out", required=True)
    pp.add_argument("--min-games", type=int, default=1)
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
        import json

        from .distribution import validate_heldout_distribution
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
    elif args.command=="weekly-forecast":
        from .weekly import run_weekly_forecast
        result,manifest=run_weekly_forecast(args.data,args.model,args.out,args.season,args.week,args.as_of,args.market,args.draws)
        print(pd.DataFrame([manifest]).to_string(index=False))
        print(result[["away_team","home_team","kickoff","expected_away_score","expected_home_score","home_win_probability","projected_home_margin"]].to_string(index=False))
    elif args.command == "capture-weather":
        from .capture_log import CAPTURE_LOG_FILENAME, append_capture, read_capture_log, summarize_capture_log
        from .venues import load_venues
        from .weather import enrich_forecasts_with_weather

        forecasts = pd.read_csv(args.forecasts)
        venues = load_venues(args.venues)
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        enriched, captures = enrich_forecasts_with_weather(
            forecasts, venues, timeout=args.timeout
        )
        enriched.to_csv(out / "forecasts_with_weather.csv", index=False)
        log_path = out / CAPTURE_LOG_FILENAME
        for row in captures:
            append_capture(
                log_path,
                source="weather.open-meteo",
                status=row.get("weather_status", "unknown"),
                observed_at=row.get("weather_observed_at"),
                game_id=row.get("game_id"),
                request_url=row.get("weather_request_url"),
                extra={
                    "temperature_c": row.get("weather_temperature_c"),
                    "wind_speed_kph": row.get("weather_wind_speed_kph"),
                    "precipitation_mm": row.get("weather_precipitation_mm"),
                    "cloud_cover_percent": row.get("weather_cloud_cover_percent"),
                    "weather_code": row.get("weather_code"),
                },
            )
        summary = summarize_capture_log(read_capture_log(log_path))
        print(pd.Series(summary["status_counts_by_source"].get("weather.open-meteo", {})).to_string())
        print(f"wrote {len(enriched)} rows to {out}/forecasts_with_weather.csv")
    elif args.command == "capture-injuries":
        from .capture_log import CAPTURE_LOG_FILENAME, append_capture, read_capture_log, summarize_capture_log
        from .injuries import fetch_injuries_for_games, summarize_reports

        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        reports = fetch_injuries_for_games(args.team_ids, timeout=args.timeout)
        rows = []
        log_path = out / CAPTURE_LOG_FILENAME
        for report in reports:
            append_capture(
                log_path,
                source="injuries.espn.cfb",
                status=report.status,
                observed_at=report.observed_at,
                request_url=report.request_url,
                extra={"team_id": report.team_id, "player_count": report.count},
            )
            rows.extend(report.to_rows())
        pd.DataFrame(rows).to_csv(out / "injury_report.csv", index=False)
        print(pd.Series(summarize_reports(reports)).to_string())
        print(f"wrote {len(rows)} player rows to {out}/injury_report.csv")
    elif args.command == "capture-rosters":
        from .capture_log import CAPTURE_LOG_FILENAME, append_capture
        from .rosters import fetch_current_rosters, summarize_captures

        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        captures = fetch_current_rosters(args.team_ids, timeout=args.timeout)
        frames = []
        log_path = out / CAPTURE_LOG_FILENAME
        for capture in captures:
            append_capture(
                log_path,
                source="rosters.espn.cfb.current",
                status=capture.status,
                observed_at=capture.observed_at,
                request_url=capture.request_url,
                extra={"team_id": capture.team_id, "player_count": capture.count},
            )
            frames.append(capture.to_frame())
        pd.concat(frames, ignore_index=True).to_csv(
            out / "rosters_current.csv", index=False
        )
        print(pd.Series(summarize_captures(captures)).to_string())
        print(f"wrote {out}/rosters_current.csv")
    elif args.command == "capture-historical-rosters":
        from .capture_log import CAPTURE_LOG_FILENAME, append_capture
        from .rosters import fetch_historical_rosters, summarize_captures

        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        captures = []
        frames = []
        log_path = out / CAPTURE_LOG_FILENAME
        for season in args.seasons:
            capture = fetch_historical_rosters(
                season,
                cache_dir=args.cache_dir,
                timeout=args.timeout,
            )
            captures.append(capture)
            append_capture(
                log_path,
                source="rosters.sportsdataverse.historical",
                status=capture.status,
                observed_at=capture.observed_at,
                request_url=capture.request_url,
                extra={"season": season, "player_count": capture.count},
            )
            frames.append(capture.to_frame())
        pd.concat(frames, ignore_index=True).to_csv(
            out / "rosters_historical.csv", index=False
        )
        print(pd.Series(summarize_captures(captures)).to_string())
        print(f"wrote {out}/rosters_historical.csv")
    elif args.command == "season-ppg":
        from .capture_log import CAPTURE_LOG_FILENAME, append_capture
        from .data import load_data
        from .season_ppg import attach_ppg_to_forecasts, compute_season_ppg

        schedules, _, _ = load_data(args.data)
        ppg = compute_season_ppg(
            schedules,
            season=args.season,
            as_of=args.as_of,
            fbs_only=not args.include_fcs,
        )
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
            forecasts = pd.read_csv(args.attach_forecasts)
            enriched = attach_ppg_to_forecasts(forecasts, ppg)
            enriched.to_csv(out / "forecasts_with_ppg.csv", index=False)
            print(f"wrote {len(enriched)} enriched forecast rows to {out}/forecasts_with_ppg.csv")
        print(ppg.head(20).to_string(index=False))
        print(f"wrote PPG for {len(ppg)} teams to {out}/season_ppg.csv")
    elif args.command == "capture-news":
        from .capture_log import CAPTURE_LOG_FILENAME, append_capture
        from .news import (
            fetch_news,
            fetch_news_for_teams,
            filter_injury_items,
            summarize_captures,
        )

        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        if args.team_ids:
            captures = fetch_news_for_teams(
                args.team_ids,
                observed_at=args.observed_at,
                timeout=args.timeout,
            )
        else:
            captures = [
                fetch_news(
                    observed_at=args.observed_at,
                    timeout=args.timeout,
                )
            ]
        frames = [capture.to_frame() for capture in captures]
        combined = (
            pd.concat(frames, ignore_index=True)
            if any(not frame.empty for frame in frames)
            else frames[0]
        )
        combined.to_csv(out / "news.csv", index=False)
        log_path = out / CAPTURE_LOG_FILENAME
        for capture in captures:
            append_capture(
                log_path,
                source="news.espn.cfb",
                status=capture.status,
                observed_at=capture.observed_at,
                request_url=capture.request_url,
                extra={
                    "team_id": capture.team_id,
                    "items": capture.count,
                    "injury_items": capture.injury_count,
                    "roster_change_items": capture.roster_change_count,
                },
            )
        if args.injury_out:
            injury_frame = filter_injury_items(captures)
            injury_frame.to_csv(args.injury_out, index=False)
            print(f"wrote {len(injury_frame)} injury-flagged items to {args.injury_out}")
        print(pd.Series(summarize_captures(captures)).to_string())
        print(f"wrote {len(combined)} news rows to {out}/news.csv")
    elif args.command == "capture-historical-rosters-span":
        from .capture_log import CAPTURE_LOG_FILENAME, append_capture
        from .rosters import fetch_historical_rosters_span, summarize_captures

        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        captures = fetch_historical_rosters_span(
            args.start_season,
            args.end_season,
            cache_dir=args.cache_dir,
            timeout=args.timeout,
        )
        frames = [capture.to_frame() for capture in captures]
        combined = (
            pd.concat(frames, ignore_index=True)
            if any(not frame.empty for frame in frames)
            else frames[0]
        )
        combined.to_csv(out / "rosters_historical_span.csv", index=False)
        log_path = out / CAPTURE_LOG_FILENAME
        for capture in captures:
            append_capture(
                log_path,
                source="rosters.sportsdataverse.historical.span",
                status=capture.status,
                observed_at=capture.observed_at,
                request_url=capture.request_url,
                extra={"season": capture.season, "player_count": capture.count},
            )
        print(pd.Series(summarize_captures(captures)).to_string())
        print(
            f"wrote {len(combined)} roster rows across seasons "
            f"{args.start_season}-{args.end_season} to {out}/rosters_historical_span.csv"
        )
    elif args.command == "player-ppg":
        from .capture_log import CAPTURE_LOG_FILENAME, append_capture
        from .player_ppg import compute_player_ppg, summarize_player_ppg

        source = Path(args.player_games)
        if source.suffix.lower() == ".parquet":
            player_games = pd.read_parquet(source)
        else:
            player_games = pd.read_csv(source)
        result = compute_player_ppg(
            player_games,
            as_of=args.as_of,
            season=args.season,
            min_games=args.min_games,
        )
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        result.to_csv(out / "player_ppg.csv", index=False)
        append_capture(
            out / CAPTURE_LOG_FILENAME,
            source="player_ppg.local",
            status="captured" if len(result) else "no_player_rows_before_cutoff",
            observed_at=args.as_of,
            extra={
                "season": args.season,
                "players": int(len(result)),
                "min_games": args.min_games,
                "source_file": str(source),
            },
        )
        summary = summarize_player_ppg(result)
        print(f"players: {summary['players']}, mean PPG: {summary['mean_ppg']}")
        for row in summary["top_scorers"][:10]:
            print(
                f"  {row.get('player_name') or row['player_id']} "
                f"(team {row.get('team_id')}): "
                f"{row['ppg_scored']:.2f} across {row['games_played_to_date']} games"
            )
        print(f"wrote PPG for {len(result)} players to {out}/player_ppg.csv")
    elif args.command == "capture-nfl":
        from .capture_log import CAPTURE_LOG_FILENAME, append_capture
        from .nfl import fetch_nfl_scoreboard

        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        capture = fetch_nfl_scoreboard(args.date, timeout=args.timeout)
        log_path = out / CAPTURE_LOG_FILENAME
        append_capture(
            log_path,
            source="nfl.espn.scoreboard",
            status=capture.status,
            observed_at=capture.observed_at,
            request_url=capture.request_url,
            extra={"scoreboard_date": capture.date, "games_listed": len(capture.games)},
        )
        pd.DataFrame(capture.to_rows()).to_csv(out / "nfl_scoreboard.csv", index=False)
        print(pd.Series(capture.to_record()).to_string())
