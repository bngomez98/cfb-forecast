"""Persist execution evidence for the read-only forecast desk."""
from datetime import datetime, timezone
import json

STAGES = [
    ("sources", "Source ingestion", "Schedules, Parquet summaries, authoritative results and identity checks."),
    ("context", "Pregame context", "Weather, rosters, injuries, team statistics and timestamped market quotes."),
    ("validation", "Outcome validation", "Unchanged prior predictions joined to final results; held-out context selection."),
    ("training", "Historical training", "Prior-result dynamic offense/defense states, ridge score heads and validation covariance."),
    ("probability", "Probability layers", "Gaussian score expectations, direct probability integration and validated context effects."),
    ("decisions", "Decision controls", "No-vig prices, expected returns, input coverage gates and exposure caps."),
    ("publication", "Immutable publication", "Source and model hashes, frozen predictions and completed-run receipt."),
]


class PipelineJournal:
    def __init__(self, path):
        self.path = path
        self.report = {"status": "running", "stages": [{"id": key, "name": name, "description": description, "status": "pending"} for key, name, description in STAGES]}

    def stage(self, key):
        now = datetime.now(timezone.utc).isoformat()
        for row in self.report["stages"]:
            if row["status"] == "running":
                row.update(status="completed", completed_at=now)
            if row["id"] == key:
                row.update(status="running", started_at=now)
        self.write()

    def finish(self, no_games=False):
        now = datetime.now(timezone.utc).isoformat()
        for row in self.report["stages"]:
            if row["status"] == "running":
                row.update(status="completed", completed_at=now)
            elif no_games and row["status"] == "pending":
                row.update(status="skipped", reason="No upcoming FBS games")
        self.report["status"] = "completed"
        self.write()

    def fail(self, error):
        self.report.update(status="failed", error=str(error))
        for row in self.report["stages"]:
            if row["status"] == "running":
                row.update(status="failed", error=str(error))
        self.write()

    def write(self):
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.report, indent=2))
        temporary.replace(self.path)
