"""Immutable prospective holdout freezing, integrity verification, and later settlement."""
from __future__ import annotations

import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from .data import normalize_schedule, sha256, timestamp
from .distribution import discrete_crps
from .evaluate import metrics


def _hash_tree(root: Path, exclude=("holdout_manifest.json", "settlements")):
    result = {}
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(root).as_posix()
        if rel in exclude or any(rel.startswith(x.rstrip("/") + "/") for x in exclude):
            continue
        result[rel] = sha256(p)
    return result


def freeze_holdout(forecast_dir, data_root, model_root, holdout_root, batch_id=None, freeze_at=None):
    src=Path(forecast_dir); data=Path(data_root); model=Path(model_root); root=Path(holdout_root)
    manifest=json.loads((src/"forecast_manifest.json").read_text())
    pred=pd.read_csv(src/"forecasts.csv")
    freeze_at=timestamp(freeze_at or datetime.now(UTC))
    kickoff=pd.to_datetime(pred.kickoff,utc=True)
    if (kickoff<=freeze_at).any():
        bad=pred.loc[kickoff<=freeze_at,["game_id","away_team","home_team","kickoff"]]
        raise ValueError(f"Prospective holdout cannot freeze games at/after kickoff: {bad.to_dict('records')[:3]}")
    if manifest.get("data_availability_check") != "pass":
        raise ValueError("Forecast does not pass prospective current-season source availability check")
    # verify exact source/model bytes before freezing
    if sha256(model/"model.json") != manifest["model_sha256"]:
        raise ValueError("Model hash no longer matches forecast manifest")
    for name, expected in manifest["source_hashes"].items():
        p=data/name
        if not p.exists() or sha256(p)!=expected:
            raise ValueError(f"Source hash mismatch for {name}")
    batch_id=batch_id or f"prospective_{freeze_at.strftime('%Y%m%dT%H%M%SZ')}"
    dst=root/batch_id
    if dst.exists():
        raise FileExistsError(dst)
    dst.mkdir(parents=True)
    (dst/"forecast").mkdir(); (dst/"inputs"/"data").mkdir(parents=True); (dst/"inputs"/"model").mkdir(parents=True)
    for name in ["forecasts.csv","forecasts.json","forecast_manifest.json","data_availability.json","state_snapshot.npz","score_distribution.parquet","margin_distribution.parquet","total_distribution.parquet","score_distribution.csv","margin_distribution.csv","total_distribution.csv"]:
        p=src/name
        if p.exists(): shutil.copy2(p,dst/"forecast"/name)
    shutil.copy2(model/"model.json",dst/"inputs"/"model"/"model.json")
    for name in manifest["source_hashes"]:
        shutil.copy2(data/name,dst/"inputs"/"data"/name)
    if (data/"source_manifest.json").exists(): shutil.copy2(data/"source_manifest.json",dst/"inputs"/"data"/"source_manifest.json")
    # If market input was used, preserve it too.
    market_hash=manifest.get("market_sha256")
    market_file=manifest.get("market_file")
    if market_hash and market_file:
        market_copy=src/market_file
        if not market_copy.exists() or sha256(market_copy)!=market_hash:
            raise ValueError("Forecast market snapshot no longer matches manifest")
        shutil.copy2(market_copy,dst/"inputs"/market_file)
    hashes=_hash_tree(dst)
    receipt={
        "schema_version":1,
        "batch_id":batch_id,
        "status":"frozen_prospective_holdout",
        "frozen_at":freeze_at.isoformat(),
        "forecast_as_of":manifest["as_of"],
        "through":manifest["through"],
        "games":int(len(pred)),
        "earliest_kickoff":kickoff.min().isoformat(),
        "latest_kickoff":kickoff.max().isoformat(),
        "prediction_sha256":sha256(dst/"forecast"/"forecasts.csv"),
        "files":hashes,
        "integrity_sha256":hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest(),
        "settlement_rule":"Predictions and archived inputs are immutable. Later outcomes are written only under settlements/ after integrity verification.",
    }
    (dst/"holdout_manifest.json").write_text(json.dumps(receipt,indent=2))
    (dst/"FROZEN.txt").write_text("Prospective holdout. Do not edit forecast/ or inputs/. Integrity is checked before settlement.\n")
    # FROZEN is added after the first hash pass, so recompute once including it.
    hashes=_hash_tree(dst)
    receipt["files"]=hashes
    receipt["integrity_sha256"]=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    (dst/"holdout_manifest.json").write_text(json.dumps(receipt,indent=2))
    return dst, receipt


def verify_holdout(holdout_dir):
    root=Path(holdout_dir); receipt=json.loads((root/"holdout_manifest.json").read_text())
    hashes=_hash_tree(root)
    if hashes != receipt["files"]:
        missing=set(receipt["files"])-set(hashes); added=set(hashes)-set(receipt["files"])
        changed=[k for k in set(hashes)&set(receipt["files"]) if hashes[k]!=receipt["files"][k]]
        raise ValueError(f"Holdout integrity failure; missing={sorted(missing)}, added={sorted(added)}, changed={changed}")
    digest=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    if digest!=receipt["integrity_sha256"]:
        raise ValueError("Holdout integrity digest mismatch")
    return receipt


def _pmf_for(path, game_id, value_col):
    f=pd.read_parquet(path) if path.suffix==".parquet" else pd.read_csv(path); g=f[f.game_id==int(game_id)]
    if g.empty: return None
    return g[value_col].to_numpy(),g.probability.to_numpy()


def settle_holdout(holdout_dir, current_data_root, settled_at=None):
    root=Path(holdout_dir); receipt=verify_holdout(root)
    pred=pd.read_csv(root/"forecast"/"forecasts.csv")
    data_root=Path(current_data_root)
    schedule_paths=sorted(data_root.glob("schedule_*.csv"))
    if not schedule_paths:
        raise FileNotFoundError(f"No schedule CSV files in {data_root}")
    schedules=normalize_schedule(pd.concat([pd.read_csv(p) for p in schedule_paths],ignore_index=True))
    outcomes=schedules[["game_id","completed","result_usable","home_points","away_points","start_date"]].copy()
    merged=pred.merge(outcomes,on="game_id",how="left",validate="one_to_one")
    settled=merged[merged.result_usable.fillna(False)].copy()
    rows=[]
    margin_path=(root/"forecast"/"margin_distribution.parquet") if (root/"forecast"/"margin_distribution.parquet").exists() else (root/"forecast"/"margin_distribution.csv")
    total_path=(root/"forecast"/"total_distribution.parquet") if (root/"forecast"/"total_distribution.parquet").exists() else (root/"forecast"/"total_distribution.csv")
    for r in settled.itertuples(index=False):
        y=int(r.home_points>r.away_points); actual_margin=int(r.home_points-r.away_points); actual_total=int(r.home_points+r.away_points)
        rec={"game_id":int(r.game_id),"home_team":r.home_team,"away_team":r.away_team,"kickoff":r.kickoff,"home_points":int(r.home_points),"away_points":int(r.away_points),"home_win":y,"actual_margin":actual_margin,"actual_total":actual_total,"forecast_home_win_probability":float(r.home_win_probability),"projected_home_margin":float(r.projected_home_margin),"projected_total":float(r.projected_total)}
        if margin_path.exists():
            v,p=_pmf_for(margin_path,r.game_id,"margin"); rec["margin_crps"]=discrete_crps(v,p,actual_margin)
        if total_path.exists():
            v,p=_pmf_for(total_path,r.game_id,"total"); rec["total_crps"]=discrete_crps(v,p,actual_total)
        rows.append(rec)
    result=pd.DataFrame(rows)
    summary={"batch_id":receipt["batch_id"],"settled_at":timestamp(settled_at or datetime.now(UTC)).isoformat(),"forecast_games":int(len(pred)),"settled_games":int(len(result)),"unsettled_games":int(len(pred)-len(result))}
    if len(result):
        summary.update(metrics(result.home_win,result.forecast_home_win_probability,result.projected_home_margin,result.actual_margin,result.projected_total,result.actual_total))
        if "margin_crps" in result: summary["mean_margin_crps"]=float(result.margin_crps.mean())
        if "total_crps" in result: summary["mean_total_crps"]=float(result.total_crps.mean())
    out=root/"settlements"; out.mkdir(exist_ok=True)
    tag=timestamp(summary["settled_at"]).strftime("%Y%m%dT%H%M%SZ")
    result.to_csv(out/f"games_{tag}.csv",index=False)
    (out/f"summary_{tag}.json").write_text(json.dumps(summary,indent=2))
    return result,summary
