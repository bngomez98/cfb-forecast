#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="${PROJECT_DIR:-/mnt/c/Users/brian/cfb-forecast}"
VENV_DIR="${VENV_DIR:-/home/brian/.venvs/cfb-forecast}"

[[ -x "$VENV_DIR/bin/python" ]] || { echo "Python venv not found: $VENV_DIR" >&2; exit 1; }
source "$VENV_DIR/bin/activate"
cd "$PROJECT_DIR"

python - <<'PY'
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

project = Path.cwd()
manifest_path = project / "data" / "source_manifest.json"
manifest = json.loads(manifest_path.read_text())

urls = {
    "ESPN scoreboard API": "https://site.api.espn.com/apis/site/v2/sports/football/college-football/scoreboard?limit=1",
    "SportsDataverse schedules release": "https://github.com/sportsdataverse/sportsdataverse-data/releases/tag/cfb_schedules",
    "SportsDataverse advanced-team release": "https://github.com/sportsdataverse/sportsdataverse-data/releases/tag/espn_cfb_adv_team",
    "SportsDataverse PBP release": "https://github.com/sportsdataverse/sportsdataverse-data/releases/tag/espn_cfb_pbp",
    "cfbfastR schedules repository": "https://github.com/sportsdataverse/cfbfastR-data",
    "cfbfastR data repository": "https://github.com/sportsdataverse/cfbfastR-cfb-data",
    "ncaa predictions repository": "https://github.com/ssmarathontrading/ncaa_predictions",
}

headers = {"User-Agent": "cfb-forecast-source-check/0.3"}
for label, url in urls.items():
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = response.read(512)
            if response.status < 200 or response.status >= 400:
                raise RuntimeError(f"HTTP {response.status}")
            if "ESPN scoreboard" in label:
                json.loads(body.decode("utf-8"))
            print(f"OK {label}: HTTP {response.status}")
    except Exception as exc:
        raise SystemExit(f"FAILED {label}: {exc}")

required_local = [
    project / "data" / "schedule_2026.csv",
    project / "data" / "adv_team_2026.parquet",
]
for path in required_local:
    if not path.exists() or path.stat().st_size == 0:
        raise SystemExit(f"FAILED local dataset: {path}")
    print(f"OK local dataset: {path.relative_to(project)}")

for name, metadata in manifest.get("files", {}).items():
    url = metadata.get("url")
    if not url:
        raise SystemExit(f"FAILED manifest entry without URL: {name}")
    print(f"OK manifest source: {name} -> {url}")

print("All ESPN and public GitHub source checks passed")
PY
