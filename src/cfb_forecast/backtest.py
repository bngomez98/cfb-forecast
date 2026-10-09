"""Warmup 2021, fit 2022-23, select/calibrate 2024, evaluate 2025.

The test year is never used for hyperparameter or variant selection.
Production refitting occurs only after test predictions have been written.
"""
from datetime import datetime,timezone
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .config import METRICS,StateConfig
from .data import load_data,audit,sha256,timestamp
from .features import observations,normalization,attach_player_features,PLAYER_VARIANTS
from .ratings import build_features
from .players import load_aggregates,player_features_by_week,PlayerConfig
from .model import ScoreHead,residual_covariance,write_model
from .simulate import win_probability
from .evaluate import metrics,calibration,paired_bootstrap
from .baseline import evaluate_elo

def verified_archive(path,games):
    if path is None:
        return None,{}
    arc=pd.read_csv(path)
    required={"game_id","snapshot_utc","snapshot_commit","snapshot_file","phwin","lineavg","lineca"}
    if not required.issubset(arc):
        raise ValueError(f"Archive missing fields {sorted(required-set(arc))}")
    if arc.game_id.duplicated().any():
        raise ValueError("Archive must have one snapshot per game")
    meta_path=Path(path).parent/"snapshot_commits.json"
    if not meta_path.exists():
        raise ValueError("Archive needs snapshot_commits.json for independently retained commit metadata")
    meta={r["file"]:r for r in json.loads(meta_path.read_text())}
    for name,group in arc.groupby("snapshot_file"):
        if Path(name).name!=name or name not in meta:
            raise ValueError("Unrecognized archive snapshot file")
        commit=meta[name]
        if not group.snapshot_commit.eq(commit["sha"]).all() or not pd.to_datetime(group.snapshot_utc,utc=True).eq(timestamp(commit["date"])).all():
            raise ValueError(f"Snapshot commit/timestamp mismatch: {name}")
        raw_path=Path(path).parent/name
        if not raw_path.exists():
            raise ValueError(f"Missing original archive snapshot: {name}")
        raw=pd.read_csv(raw_path)
        if raw.duplicated(["home","road"]).any():
            raise ValueError(f"Ambiguous raw archive match: {name}")
        check=group.merge(raw[["home","road","phwin","lineavg","lineca"]],on=["home","road"],how="left",suffixes=("","_raw"),validate="many_to_one")
        for c in ["phwin","lineavg","lineca"]:
            if not np.allclose(check[c],check[c+"_raw"],equal_nan=True,rtol=1e-10,atol=1e-10):
                raise ValueError(f"Archive probabilities/lines differ from original snapshot: {name} {c}")
    arc=arc.merge(games[["game_id","start_date"]],on="game_id",validate="one_to_one")
    arc["snapshot_utc"]=pd.to_datetime(arc.snapshot_utc,utc=True)
    good=(arc.snapshot_utc<arc.start_date)&arc.snapshot_commit.astype(str).str.fullmatch(r"[a-fA-F0-9]{40}")&arc.phwin.between(0,1)
    arc=arc[good].drop(columns="start_date")
    return arc,dict(zip(arc.game_id.astype(int),arc.snapshot_utc))

def load_player_features(data_root,games,lead_hours=48.0):
    """Point-in-time player feature table for the backtest games, or None if no aggregates."""
    agg=load_aggregates(data_root)
    if agg.empty:
        return None
    return player_features_by_week(agg,games,lead_hours,PlayerConfig())


def run_backtest(data_root,out_root,archive_path=None,warmup=2021,train_start=2022,validation=2024,test=2025):
    if not warmup<train_start<validation<test:
        raise ValueError("Need warmup < training start < validation < test year")
    out=Path(out_root); out.mkdir(parents=True,exist_ok=True)
    schedules,summaries,paths=load_data(data_root)
    audit(schedules,summaries).to_csv(out/"data_audit.csv",index=False)
    games=schedules[schedules.fbs_game&schedules.result_usable&schedules.season.between(warmup,test)].copy().reset_index(drop=True)
    for year in range(warmup,test+1):
        if not (games.season==year).any():
            raise ValueError(f"No completed FBS-v-FBS games for required season {year}")
    Y=observations(games,summaries); center,scale=normalization(Y,games.season.eq(warmup).to_numpy())
    train=games.season.between(train_start,validation-1).to_numpy()
    val=games.season.eq(validation).to_numpy(); te=games.season.eq(test).to_numpy()
    final_train=games.season.between(train_start,validation).to_numpy()
    arc,explicit=verified_archive(archive_path,games)
    # Point-in-time player layer: hierarchical PBP player ratings as an extra feature block.
    # Present only when per-player-game aggregates are available; the state-only variants are unaffected.
    player_features=load_player_features(data_root,games)
    variants=("score","pbp","pbp_player") if player_features is not None else ("score","pbp")
    # Archived competitor timestamps define the same information cutoff for both.
    cache={}; trials=[]
    for q in (.005,.02,.06):
        for carry in (.5,.8):
            cfg=StateConfig(process_variance=q,offseason_carry=carry)
            F,cutoffs,latest=build_features(games,Y,center,scale,cfg,explicit,warmup)
            Fp=attach_player_features(F,games.game_id.tolist(),player_features) if player_features is not None else None
            cache[(q,carry)]=(F,Fp,cutoffs,latest)
            for variant in variants:
                feats=Fp if variant in PLAYER_VARIANTS else F
                for alpha in (10.,100.,1000.):
                    head=ScoreHead(variant,alpha).fit(feats,Y,train)
                    train_mu=head.predict(feats[train]); val_mu=head.predict(feats[val])
                    # In-sample covariance is only for validation-grid ranking;
                    # released covariance comes from out-of-fit validation residuals.
                    sigma=residual_covariance(Y[train,:,0],train_mu)
                    p=win_probability(val_mu,sigma)
                    scores=metrics((Y[val,0,0]>Y[val,1,0]).astype(int),p)
                    trials.append({"variant":variant,"q":q,"carry":carry,"alpha":alpha,**scores})
            print(f"computed chronological features q={q} carry={carry}",flush=True)
    pd.DataFrame(trials).to_csv(out/"validation_trials.csv",index=False)
    selected={v:min((r for r in trials if r["variant"]==v),key=lambda r:r["log_loss"]) for v in variants}
    chosen_variant=min(selected,key=lambda v:selected[v]["log_loss"])
    result=games.loc[te,["game_id","season","week","start_date","home_id","away_id","home_team","away_team","home_points","away_points","neutral_site"]].copy()
    result["home_win"]=(result.home_points>result.away_points).astype(int)
    result["actual_home_margin"]=result.home_points-result.away_points
    result["actual_total"]=result.home_points+result.away_points
    models={}; summaries_out=[]; production={}
    for variant,best in selected.items():
        F,Fp,cutoffs,latest=cache[(best["q"],best["carry"])]
        feats=Fp if variant in PLAYER_VARIANTS else F
        cal_head=ScoreHead(variant,best["alpha"]).fit(feats,Y,train)
        cov=residual_covariance(Y[val,:,0],cal_head.predict(feats[val]))
        head=ScoreHead(variant,best["alpha"]).fit(feats,Y,final_train)
        mu=head.predict(feats[te]); p=win_probability(mu,cov)
        result[f"{variant}_home_score"]=mu[:,0]; result[f"{variant}_away_score"]=mu[:,1]
        result[f"{variant}_home_margin"]=mu[:,0]-mu[:,1]; result[f"{variant}_total"]=mu.sum(axis=1)
        result[f"{variant}_p"]=p
        result["forecast_cutoff"]=cutoffs[te].astype(str).to_numpy()
        result["latest_result_available_at"]=[str(x) if x is not None else "" for x,yes in zip(latest,te) if yes]
        base={"schema_version":1,"variant":variant,"state_config":StateConfig(process_variance=best["q"],offseason_carry=best["carry"]).to_dict(),"metric_names":list(METRICS),"center":center.tolist(),"scale":scale.tolist(),"head":head.to_dict(),"score_covariance":cov.tolist(),"warmup_season":warmup,"train_start_season":train_start,"training_through_season":validation,"calibration_season":validation,"selection_season":validation,"source_hashes":{p.name:sha256(p) for p in paths},"input_note":"Retrospectively revised historical records; not immutable as-known statistics"}
        if variant in PLAYER_VARIANTS:
            base["player_layer"]={"lead_hours":48.0,"config":PlayerConfig().to_dict(),"feature_columns":["own_player_offense","opponent_player_defense"],"note":"Hierarchical PBP player ratings fit point-in-time per week; see players.py"}
        models[variant]=base
        write_model(out/"models"/variant/"heldout",base)
        m=metrics(result.home_win,p,mu[:,0]-mu[:,1],result.actual_home_margin,mu.sum(axis=1),result.actual_total,mu,Y[te,:,0])
        summaries_out.append({"model":variant,"season":test,**m})
        calibration(result.home_win,p).to_csv(out/f"calibration_{variant}.csv",index=False)
        # Refit for later forecasting; never overwrite held-out predictions.
        prod=dict(base); prod["training_through_season"]=test
        prod["head"]=ScoreHead(variant,best["alpha"]).fit(feats,Y,games.season.between(train_start,test).to_numpy()).to_dict()
        production[variant]=prod; write_model(out/"models"/variant/"production",prod)
    elo_p,elo_trials,elo_chosen=evaluate_elo(games,train,val,final_train,te,explicit)
    result["elo_p"]=elo_p
    pd.DataFrame(elo_trials).to_csv(out/"elo_validation.csv",index=False)
    summaries_out.append({"model":"elo","season":test,**metrics(result.home_win,elo_p)})
    calibration(result.home_win,elo_p).to_csv(out/"calibration_elo.csv",index=False)
    result.to_csv(out/"predictions.csv",index=False)
    summaries_out.append({"model":"always_home","season":test,**metrics(result.home_win,np.full(len(result),.5))})
    comparisons=[paired_bootstrap(result,"pbp_p","score_p"),paired_bootstrap(result,"score_p","elo_p"),paired_bootstrap(result,"pbp_p","elo_p")]
    if arc is not None and len(arc):
        matched=result.merge(arc,on="game_id",validate="one_to_one")
        matched=matched[np.isfinite(matched.phwin)&np.isfinite(matched.lineavg)].copy()
        matched["archive_p"]=matched.phwin
        # Tracker lineavg: positive means home advantage, verified in original archive.
        matched["archive_home_margin"]=matched.lineavg
        matched.to_csv(out/"archive_comparison.csv",index=False)
        if len(matched):
            for name,col,margin in [("archive","archive_p","archive_home_margin"),("score_matched","score_p","score_home_margin"),("pbp_matched","pbp_p","pbp_home_margin")]:
                summaries_out.append({"model":name,"season":test,**metrics(matched.home_win,matched[col],matched[margin],matched.actual_home_margin)})
            for variant in ("score","pbp"):
                comparisons.append(paired_bootstrap(matched,f"{variant}_p","archive_p"))
            # Recorded historical lineca is descriptive archive evidence. Odds and
            # original sportsbook timestamp are not established; no ROI reported.
            spread=[]
            for variant in ("score","pbp"):
                f=matched[matched.lineca.notna()].copy()
                edge=f[f"{variant}_home_margin"]-f.lineca
                actual=f.actual_home_margin-f.lineca
                chosen=np.where(edge>0,actual,-actual)
                for threshold in [0.,2.,4.]:
                    take=edge.abs()>threshold; vals=chosen[take]
                    spread.append({"variant":variant,"minimum_margin_edge":threshold,"games":int(take.sum()),"wins":int((vals>0).sum()),"losses":int((vals<0).sum()),"pushes":int((vals==0).sum()),"win_rate_excluding_pushes":float((vals>0).sum()/(vals!=0).sum()) if (vals!=0).sum() else None})
            pd.DataFrame(spread).to_csv(out/"archive_spread_diagnostics.csv",index=False)
    pd.DataFrame(summaries_out).to_csv(out/"metrics.csv",index=False)
    pd.DataFrame(comparisons).to_csv(out/"paired_comparisons.csv",index=False)
    write_model(out/"model",production[chosen_variant])
    spec={"created_at":datetime.now(timezone.utc).isoformat(),"warmup":warmup,"initial_train_seasons":list(range(train_start,validation)),"validation_season":validation,"test_season":test,"selected_variant":chosen_variant,"selected_settings":selected,"elo_selected_C":elo_chosen["C"],"evaluation_games":len(result),"all_cutoffs_before_kickoff":bool((pd.to_datetime(result.forecast_cutoff,utc=True)<result.start_date).all()),"all_latest_results_before_cutoff":bool((pd.to_datetime(result.latest_result_available_at,utc=True)<pd.to_datetime(result.forecast_cutoff,utc=True)).all()),"missing_pbp_does_not_remove_games":True,"source_hashes":{p.name:sha256(p) for p in paths},"archive_sha256":sha256(archive_path) if archive_path else None,"limitations":["Historical source statistics may have later revisions","Unknown final-whistle time uses kickoff plus eight hours","FBS-v-FBS rating and evaluation population","Empirical validation covariance includes state-estimation uncertainty; it is not added a second time","No learned player/injury availability, fitted EPA, or drive-level rules engine","Archive comparison is a public forecast tracker, not a bookmaker moneyline baseline","Test season has been studied in earlier work; this is reproducible historical evaluation, not a newly blind experiment"]}
    (out/"run_manifest.json").write_text(json.dumps(spec,indent=2))
    return pd.DataFrame(summaries_out),spec
