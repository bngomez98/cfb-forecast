"""Continuous awareness checks for data, models, forecasts, and storage."""
from __future__ import annotations
from datetime import datetime, timezone
import json
from pathlib import Path
import pandas as pd
from .data import load_data, sha256


def awareness_report(data_root, model_root, database_path):
    report={"checked_at":datetime.now(timezone.utc).isoformat(),"checks":{},"overall":"pass"}
    root=Path(data_root); model=Path(model_root)
    try:
        schedules,summaries,paths=load_data(root)
        report["checks"]["data"]={"status":"pass","schedule_rows":len(schedules),"summary_rows":len(summaries),"source_files":len(paths),"latest_schedule_date":schedules.start_date.max().isoformat()}
        report["checks"]["schema"]={"status":"pass","schedule_columns":sorted(schedules.columns.tolist()),"summary_columns":sorted(summaries.columns.tolist())}
    except Exception as exc:
        report["checks"]["data"]={"status":"fail","error":str(exc)}
    model_file=model/"model.json"
    try:
        value=json.loads(model_file.read_text())
        required=["schema_version","variant","training_through_season","head","score_covariance"]
        missing=[x for x in required if x not in value]
        report["checks"]["model"]={"status":"pass" if not missing else "fail","variant":value.get("variant"),"training_through_season":value.get("training_through_season"),"sha256":sha256(model_file),"missing_fields":missing}
    except Exception as exc:
        report["checks"]["model"]={"status":"fail","error":str(exc)}
    db=Path(database_path)
    report["checks"]["database"]={"status":"pass" if db.parent.exists() else "fail","path":str(db),"exists":db.exists()}
    if any(v.get("status")=="fail" for v in report["checks"].values()): report["overall"]="fail"
    return report
