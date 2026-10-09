"""Replay only available prior results, then freeze one as-of state snapshot."""
from pathlib import Path
from datetime import datetime,timezone
import json
import shutil
import numpy as np
import pandas as pd
from .config import StateConfig
from .data import load_data,timestamp,sha256
from .features import observations
from .ratings import DynamicRatings
from .model import load_model,ScoreHead
from .simulate import simulate_scores
from .distribution import probability_tables
from .market import market_at_cutoff,market_history_summary,read_market_history
from .availability import availability_snapshot
from .features import PLAYER_VARIANTS, attach_player_features


def _attach_player_block(F, targets, data_root, as_of, model):
    """Fit the player layer as of the forecast cutoff and append its feature block."""
    from .players import PlayerConfig, fit_player_ratings, load_aggregates, recent_usage, team_defense_rating, team_offense_ratings

    spec = model.get("player_layer", {})
    agg = load_aggregates(data_root)
    if agg.empty:
        raise ValueError("Model uses the player layer but no player_games_*.parquet aggregates were found; run player-aggregates")
    config = PlayerConfig(**spec["config"]) if spec.get("config") else PlayerConfig()
    ratings = fit_player_ratings(agg, as_of, config)
    usage = recent_usage(agg, as_of)
    team = team_offense_ratings(ratings, usage)
    offense = team.set_index("team_id").player_offense if len(team) else pd.Series(dtype=float)
    rows = []
    for r in targets.itertuples(index=False):
        ho, ao = float(offense.get(int(r.home_id), 0.0)), float(offense.get(int(r.away_id), 0.0))
        rows.append({"game_id": int(r.game_id), "home_player_offense": ho, "away_player_offense": ao,
                     "home_player_defense": team_defense_rating(ratings, r.home_id),
                     "away_player_defense": team_defense_rating(ratings, r.away_id),
                     "player_net_signal": 0.0})
    pf = pd.DataFrame(rows).set_index("game_id")
    return attach_player_features(F, targets.game_id.tolist(), pf)


def forecast(data_root,model_root,out_root,as_of,through,season,market_path=None,draws=50000,market_provider=None,save_distribution=True,target_game_ids=None):
    at=timestamp(as_of); end=timestamp(through)
    if end<=at:
        raise ValueError("Forecast window must end after as-of time")
    model=load_model(model_root)
    # A fitted head containing the forecast season would be retrospective leakage.
    if model["training_through_season"]>=season:
        raise ValueError("Model training must end before the forecast season")
    schedules,summaries,paths=load_data(data_root)
    population=schedules[schedules.fbs_game&schedules.season.between(model["warmup_season"],season)].copy().reset_index(drop=True)
    targets=population[(population.season==season)&(population.start_date>at)&(population.start_date<=end)].copy()
    if target_game_ids is not None:
        wanted={int(x) for x in target_game_ids}
        targets=targets[targets.game_id.astype(int).isin(wanted)].copy()
        missing=wanted-set(targets.game_id.astype(int))
        if missing:
            raise ValueError(f"Requested target game IDs are not eligible in the forecast window: {sorted(missing)}")
    if targets.empty:
        raise ValueError("No FBS-v-FBS games in the requested forecast window")
    past=population[population.result_usable&(population.result_available_at<at)].copy().reset_index(drop=True)
    if not (past.season==model["warmup_season"]).any():
        raise ValueError("Historical warmup season missing")
    teams=sorted(set(population.home_id)|set(population.away_id))
    state=DynamicRatings(teams,model["center"],model["scale"],StateConfig(**model["state_config"]))
    Y=observations(past,summaries)
    events=[]
    for year,g in population.groupby("season"):
        reset=g.start_date.min()-pd.Timedelta(days=21)
        if reset<at:
            events.append((reset.value,0,int(year)))
    records=list(past.itertuples(index=False))
    for k,r in enumerate(records):
        events.append((r.result_available_at.value,1,k))
    for tm,typ,k in sorted(events):
        event_at=pd.Timestamp(tm,tz="UTC")
        if typ==0:
            state.offseason(event_at)
        else:
            r=records[k]; state.update(r.home_id,r.away_id,r.neutral_site,Y[k],event_at)
    state.advance(at)
    F=np.stack([state.features(r.home_id,r.away_id,r.neutral_site,int(r.week)<=4) for r in targets.itertuples(index=False)])
    if model["variant"] in PLAYER_VARIANTS:
        F=_attach_player_block(F,targets,data_root,at,model)
    mu=ScoreHead.from_dict(model["head"]).predict(F)
    market=read_market_history(market_path) if market_path else None
    rows=[]; joint=[]; margins=[]; totals=[]
    for r,means in zip(targets.itertuples(index=False),mu):
        m=market_at_cutoff(market,r.game_id,at,r.start_date,provider=market_provider) if market is not None and not market.empty else {}
        hist=market_history_summary(market,r.game_id,at,r.start_date,provider=market_provider) if market is not None and not market.empty else {"market_snapshot_count":0}
        sim,samples=simulate_scores(means,model["score_covariance"],r.game_id,draws=draws,return_samples=True,**{k:v for k,v in m.items() if k in ["home_spread","total_line","home_decimal","away_decimal"]})
        if save_distribution:
            tabs=probability_tables(samples,r.game_id); joint.append(tabs["joint"]); margins.append(tabs["margin"]); totals.append(tabs["total"])
        rows.append({"game_id":int(r.game_id),"season":int(r.season),"home_team":r.home_team,"away_team":r.away_team,"kickoff":r.start_date.isoformat(),"as_of":at.isoformat(),"latest_result_available_at":state.latest_update.isoformat() if state.latest_update is not None else None,"prior_fbs_games":len(past),"model_variant":model["variant"],"head_training_through_season":model["training_through_season"],"prediction_method":"dynamic Gaussian state + regularized score head + prior-season residual covariance",**sim,**m,**hist})
    out=Path(out_root); out.mkdir(parents=True,exist_ok=True)
    market_file=None
    if market_path:
        mp=Path(market_path)
        market_file=f"market_history{mp.suffix.lower()}"
        shutil.copy2(mp,out/market_file)
    (out/"forecasts.json").write_text(json.dumps(rows,indent=2))
    pd.DataFrame(rows).to_csv(out/"forecasts.csv",index=False)
    distribution_files=[]
    if save_distribution:
        for stem,frames in [("score_distribution",joint),("margin_distribution",margins),("total_distribution",totals)]:
            table=pd.concat(frames,ignore_index=True)
            try:
                table.to_parquet(out/f"{stem}.parquet",index=False)
                distribution_files.append(f"{stem}.parquet")
            except (ImportError, ModuleNotFoundError):
                table.to_csv(out/f"{stem}.csv",index=False)
                distribution_files.append(f"{stem}.csv")
    np.savez_compressed(out/"state_snapshot.npz",team_ids=state.teams,mean=state.mean,cov=state.cov,features=F)
    avail=availability_snapshot(data_root,at,schedules,summaries,paths)
    (out/"data_availability.json").write_text(json.dumps(avail,indent=2))
    trace={"as_of":at.isoformat(),"through":end.isoformat(),"model_sha256":sha256(Path(model_root)/"model.json"),"source_hashes":{p.name:sha256(p) for p in paths},"market_sha256":sha256(market_path) if market_path else None,"market_file":market_file,"market_provider":market_provider,"used_game_ids":past.game_id.astype(int).tolist(),"latest_result_available_at":state.latest_update.isoformat(),"created_at":datetime.now(timezone.utc).isoformat(),"distribution_files":distribution_files,"target_game_ids":targets.game_id.astype(int).tolist(),"data_availability_check":avail["prospective_current_season_source_check"],"information_limit":"Downloaded record versions may postdate individual historical stat publication times. Prospective current-season source capture is checked separately in data_availability.json."}
    (out/"forecast_manifest.json").write_text(json.dumps(trace,indent=2))
    return pd.DataFrame(rows)
