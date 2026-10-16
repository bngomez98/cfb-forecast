"""Optional timestamped CFBD quotes, matched conservatively to scheduled games."""
import os
from pathlib import Path

import pandas as pd

from .external_sources import _cfbd_get
from .data import timestamp
from ..odds.decision import american_to_decimal
from ..odds.market import append_market_history


def capture_cfbd_prices(games, season, root, store):
    key = os.environ.get("CFBD_API_KEY")
    if not key or games.empty:
        return 0
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    snapshots = []
    for week in sorted(games.week.unique()):
        payload, receipt = _cfbd_get("/lines", {"year": int(season), "week": int(week)},
                                     root / f"cfbd_lines_{int(week)}.json", key)
        if not isinstance(payload, list) or receipt.get("status") != "ok":
            continue
        for event in payload:
            # Provider IDs alone are not sufficient evidence of a cross-source match.
            matched = games[games.home_team.str.casefold().eq(str(event.get("homeTeam", "")).casefold())
                            & games.away_team.str.casefold().eq(str(event.get("awayTeam", "")).casefold())]
            if len(matched) != 1:
                continue
            game = matched.iloc[0]
            kickoff = pd.to_datetime(event.get("startDate"), utc=True, errors="coerce")
            observed = timestamp(receipt["retrieved_at"])
            if pd.isna(kickoff) or kickoff != game.start_date or observed >= kickoff:
                continue
            for quote in event.get("lines", []):
                if not quote.get("provider"):
                    continue
                row = dict(game_id=int(game.game_id), provider=f"cfbd:{quote['provider']}",
                           observed_at=receipt["retrieved_at"], source=receipt["url"])
                for source, target in (("spread", "home_spread"), ("overUnder", "total_line")):
                    value = pd.to_numeric(quote.get(source), errors="coerce")
                    if pd.notna(value) and abs(float(value)) != float("inf"):
                        row[target] = float(value)
                for side in ("home", "away"):
                    value = quote.get(f"{side}Moneyline")
                    if value is not None:
                        try:
                            row[f"{side}_ml_decimal"] = american_to_decimal(value)
                        except (TypeError, ValueError):
                            pass
                if len(row) > 4:
                    snapshots.append(row)
    if snapshots:
        append_market_history(store, pd.DataFrame(snapshots))
    return len(snapshots)
