"""Point-in-time player evidence for the joint hierarchy.

The importer accepts normalized, identified source records, not guessed HTML
columns. Provider duplicates are alternate measurements, not additional games.
Legacy names are explicitly scoped to a team and season, never fuzzy matched.
"""
from contextlib import closing
from hashlib import sha256
import json

import numpy as np
import pandas as pd

from ..storage.database import connect
from .data import timestamp
from .features import load_history_facts

ROLES = ("passing", "rushing", "receiving")


def import_player_evidence(database, payload):
    """Atomically import identified observations and complete role lineups.

    ``epa`` is mean EPA per recorded opportunity, matching legacy facts.
    ``yards`` is total yardage, not yardage per opportunity.
    Each lineup row contains player_id, role, opportunity_share and optional
    availability_probability (an explicit supplied estimate, never inferred).
    """
    observations = payload.get("observations", [])
    lineups = payload.get("lineups", [])
    identities = payload.get("players", [])
    with closing(connect(database)) as conn, conn:
        for row in identities:
            if not row.get("player_id") or not row.get("display_name"):
                raise ValueError("Player identities require player_id and display_name")
            conn.execute("INSERT INTO player_identities VALUES(?,?) ON CONFLICT(player_id) DO UPDATE SET display_name=excluded.display_name",
                         (row["player_id"], row["display_name"]))
            for alias in row.get("source_ids", []):
                previous = conn.execute("SELECT player_id FROM player_source_ids WHERE provider=? AND source_player_id=?",
                                        (alias["provider"], str(alias["source_player_id"]))).fetchone()
                if previous and previous[0] != row["player_id"]:
                    raise ValueError("Source identity already maps to a different player")
                conn.execute("INSERT OR IGNORE INTO player_source_ids VALUES(?,?,?)",
                             (alias["provider"], str(alias["source_player_id"]), row["player_id"]))
        for row in observations:
            _provenance(row)
            if row["role"] not in ROLES or int(row["team_id"]) == int(row["opponent_id"]):
                raise ValueError("Invalid role or opponent")
            n = float(row["opportunities"])
            if not np.isfinite(n) or n <= 0:
                raise ValueError("Opportunities must be positive and finite")
            for key in ("yards", "epa"):
                if row.get(key) is not None and not np.isfinite(float(row[key])):
                    raise ValueError(f"{key} must be finite when supplied")
            conn.execute("INSERT OR REPLACE INTO player_observations VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                         (int(row["game_id"]), int(row["team_id"]), int(row["opponent_id"]), row["player_id"], row["role"], n,
                          row.get("yards"), row.get("epa"), timestamp(row["available_at"]).isoformat(),
                          row["provider"], row["source_url"], row["source_sha256"]))
        for row in lineups:
            _provenance(row)
            players = row["players"]
            seen = set()
            totals = {}
            for player in players:
                key = (player["player_id"], player["role"])
                if key in seen or key[1] not in ROLES:
                    raise ValueError("Duplicate player/role or invalid role in lineup")
                seen.add(key)
                if not conn.execute("SELECT 1 FROM player_identities WHERE player_id=?", (key[0],)).fetchone():
                    raise ValueError("Lineup references an unknown player identity")
                share = float(player["opportunity_share"])
                if not np.isfinite(share) or share < 0:
                    raise ValueError("Opportunity shares must be finite and nonnegative")
                totals[key[1]] = totals.get(key[1], 0.) + share
                p = player.get("availability_probability")
                if p is not None and (not np.isfinite(float(p)) or not 0 <= float(p) <= 1):
                    raise ValueError("Availability probability must lie in [0,1]")
            if any(not np.isclose(total, 1.) for total in totals.values()):
                raise ValueError("Opportunity shares must sum to one within each supplied role")
            conn.execute("INSERT OR REPLACE INTO player_lineups VALUES(?,?,?,?,?,?,?)",
                         (int(row["game_id"]), int(row["team_id"]), timestamp(row["observed_at"]).isoformat(), row["provider"],
                          row["source_url"], row["source_sha256"], json.dumps(players, allow_nan=False)))
        for row in payload.get("contexts", []):
            _provenance(row)
            features = row["features"]
            from ..modeling.forecast_model import KNOWN_COLUMNS
            if set(features) - set(KNOWN_COLUMNS):
                raise ValueError("Context contains unknown or outcome columns")
            if any(value is not None and not np.isfinite(float(value)) for value in features.values()):
                raise ValueError("Context values must be finite or null")
            conn.execute("INSERT OR REPLACE INTO game_context_snapshots VALUES(?,?,?,?,?,?)",
                         (int(row["game_id"]), timestamp(row["observed_at"]).isoformat(), row["provider"], row["source_url"],
                          row["source_sha256"], json.dumps(features, allow_nan=False)))
    return {"players": len(identities), "observations": len(observations), "lineups": len(lineups), "contexts": len(payload.get("contexts", []))}


def _provenance(row):
    if not all(row.get(key) for key in ("provider", "source_url", "source_sha256")):
        raise ValueError("Player evidence requires provider, source_url and source_sha256")


def player_evidence(data_root, database, games, cutoff):
    """Join observations to authoritative opponents and final-result availability.

    Legacy files have no historical publication receipts: result availability
    gates them and their provenance is labelled retrospective. Versioned SQL
    observations require BOTH result and observation availability before cutoff.
    """
    at = timestamp(cutoff)
    valid = games.loc[games.result_usable & (games.result_available_at < at)].copy()
    schedule = {int(r.game_id): r for r in valid.itertuples(index=False)}
    _, legacy = load_history_facts(data_root, database=database)
    rows = []
    with closing(connect(database)) as conn:
        aliases = {r[0]: r[1] for r in conn.execute("SELECT source_player_id, player_id FROM player_source_ids WHERE provider='sportsdataverse'")}
    if legacy is not None and len(legacy):
        for r in legacy.to_dict("records"):
            game = schedule.get(int(r["game_id"]))
            if game is None or r.get("position_group") not in ROLES:
                continue
            team = int(r["team_id"])
            if team not in (int(game.home_id), int(game.away_id)):
                continue
            n = r.get("plays")
            if n is None or not np.isfinite(n) or n <= 0:
                continue
            source_identity = r.get("source_player_id")
            if source_identity is not None and not pd.isna(source_identity):
                source_identity = str(int(source_identity)) if isinstance(source_identity, (int, float)) else str(source_identity)
                identity = aliases.get(source_identity, f"source:sportsdataverse:{source_identity}")
            else:
                identity = f"legacy:{game.season}:{team}:{r['player_name']}"
            rows.append({"game_id": int(game.game_id), "team_id": team,
                         "opponent_id": int(game.away_id if team == game.home_id else game.home_id),
                         "player_id": identity, "role": r["position_group"], "opportunities": float(n),
                         "yards": r.get("yards"), "epa": r.get("epa"), "available_at": game.result_available_at,
                         "provider": "legacy_retrospective", "season": int(game.season), "start_date": game.start_date})
    with closing(connect(database)) as conn:
        stored = pd.read_sql_query("SELECT * FROM player_observations", conn)
        lineups = pd.read_sql_query("SELECT * FROM player_lineups", conn)
    if len(stored):
        stored["available_at"] = pd.to_datetime(stored.available_at, utc=True)
        stored = stored.loc[stored.available_at < at].sort_values(["available_at", "provider"], kind="stable")
        # Contradictory provider measurements are not counted as independent games.
        # Prefer the latest eligible version; retain original source rows in SQL.
        stored = stored.drop_duplicates(["game_id", "team_id", "player_id", "role"], keep="last")
        for r in stored.to_dict("records"):
            game = schedule.get(int(r["game_id"]))
            if game is None:
                continue
            expected = game.away_id if int(r["team_id"]) == int(game.home_id) else game.home_id
            if int(r["team_id"]) not in (int(game.home_id), int(game.away_id)) or int(r["opponent_id"]) != int(expected):
                raise ValueError("Player observation contradicts authoritative game opponents")
            r.update(season=int(game.season), start_date=game.start_date)
            rows.append(r)
        # Never mix legacy aliases and canonical player records for the same
        # game/team/role. Canonical coverage replaces that legacy role slice.
        canonical = set(zip(stored.game_id, stored.team_id, stored.role))
        rows = [r for r in rows if r["provider"] != "legacy_retrospective" or
                (r["game_id"], r["team_id"], r["role"]) not in canonical]
    if len(lineups):
        lineups["observed_at"] = pd.to_datetime(lineups.observed_at, utc=True)
        lineups = lineups.loc[lineups.observed_at < at].sort_values(["observed_at", "provider"], kind="stable")
        lineups = lineups.drop_duplicates(["game_id", "team_id"], keep="last")
    observations = pd.DataFrame(rows)
    # Hash exactly the selected evidence, not a mutable SQLite file checksum.
    encoded = json.dumps({"observations": rows, "lineups": lineups.to_dict("records")}, default=str, sort_keys=True)
    return observations, lineups, sha256(encoded.encode()).hexdigest()
