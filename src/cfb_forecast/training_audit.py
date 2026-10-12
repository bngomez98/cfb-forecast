"""Evidence that historical team observations actually enter model fitting."""
import json
import numpy as np
from .config import METRICS
from .data import sha256
from .features import observations


def training_audit(games, summaries, paths, warmup, train_start, validation, test):
    rows = []
    for year in range(warmup, test + 1):
        season = games[games.season.eq(year)]
        if season.empty:
            raise ValueError(f"No completed FBS-v-FBS games for required season {year}")
        values = observations(season, summaries)
        counts = {metric: int(np.isfinite(values[:, :, index]).sum()) for index, metric in enumerate(METRICS)}
        missing = [metric for metric in METRICS[1:] if counts[metric] == 0]
        if missing:
            raise ValueError(f"Season {year} has no matched advanced observations for {missing}; restore its adv_team Parquet or CSV file")
        role = "warmup" if year < train_start else "training"
        if year == validation:
            role = "validation / calibration"
        elif year == test:
            role = "held-out evaluation / later production refit"
        rows.append({"season": year, "role": role, "games": len(season), "team_observations": 2 * len(season), "complete_advanced_rows": int(np.isfinite(values[:, :, 1:]).all(axis=2).sum()), **counts})
    return {"status": "verified", "seasons": rows, "features": list(METRICS), "source_files": [{"file": p.name, "format": p.suffix.lstrip("."), "sha256": sha256(p), "bytes": p.stat().st_size} for p in paths], "missing_policy": "Keep games with partial advanced coverage; never impute missing source facts as zero.", "chronology": "Warmup normalization; prior-result states; validation-only selection; test predictions saved before production refitting."}


def save_training_audit(out, *args):
    report = training_audit(*args)
    (out / "training_inputs.json").write_text(json.dumps(report, indent=2))
    return report
