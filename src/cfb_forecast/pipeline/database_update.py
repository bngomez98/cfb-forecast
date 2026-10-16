"""Explicit refresh of the shared source catalog and normalized schedule store."""
import json
from contextlib import closing
from pathlib import Path

from ..ingest.data import load_data
from ..ingest.sources import ingest_catalog
from ..storage.database import connect
from ..storage.football_database import register_source, record_retrieval, upsert_schedule


def update_database(data_root, database, season, artifacts, fetch_sources=False):
    schedules, _, files = load_data(data_root)
    schedules = schedules[schedules.season.eq(int(season))]
    if schedules.empty:
        raise ValueError(f"No schedule available for season {season}")
    destination = Path(artifacts) / "source_ingest" / str(season)
    receipts = ingest_catalog(season, destination) if fetch_sources else []
    from .publisher import register_sources
    with closing(connect(database)) as conn:
        registered = register_sources(conn, season)
        source_id = register_source(conn, "sportsdataverse_schedule", "sportsdataverse")
        games = upsert_schedule(conn, schedules, source_id)
        for path in files:
            record_retrieval(conn, source_id, path, metadata={"scope": "local_input", "season": int(season)})
        for receipt in receipts:
            source_id = register_source(conn, receipt["name"], receipt["name"], receipt.get("url"))
            path = destination / f"{receipt['name']}.bin"
            retrieval_id = record_retrieval(conn, source_id, path,
                                            status=receipt["status"], metadata=receipt)
            if receipt["status"] == "ok" and path.exists():
                content = path.read_text(errors="replace")
                try:
                    payload = json.loads(content)
                except json.JSONDecodeError:
                    payload = {"content": content, "format": "text"}
                rows = payload if isinstance(payload, list) else [payload]
                conn.executemany("INSERT INTO source_rows(retrieval_id,row_index,payload_json) VALUES(?,?,?)",
                                 [(retrieval_id, index, json.dumps(row)) for index, row in enumerate(rows)])
                conn.commit()
        from ..ingest.features import load_history_facts
        from ..storage.database import record_history_facts
        teams, players = load_history_facts(data_root)
        record_history_facts(conn, teams, players)
    return {"status": "updated", "season": int(season), "games": games,
            "registered_sources": registered, "input_files": len(files), "source_receipts": receipts,
            "database": str(database)}
