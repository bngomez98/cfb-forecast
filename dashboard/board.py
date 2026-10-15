"""Readable forecast pages: game cards grouped by day, odds, and a plain explanation."""
from datetime import timedelta
from html import escape
import json
from pathlib import Path

import pandas as pd
import streamlit as st
from cfb_forecast.paths import artifacts_root, project_root

# College football schedules are read in Eastern time.
ZONE = "America/New_York"
ZONE_LABEL = "ET"


def _read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


@st.cache_data(ttl=60)
def _load(paths, modified):
    for path in paths:
        path = Path(path)
        if not path.exists():
            continue
        if path.suffix == ".csv":
            rows = pd.read_csv(path)
            return {"forecasts": rows.to_dict("records"), "source": str(path)}
        board = _read_json(path)
        if board and board.get("forecasts"):
            return {**board, "source": str(path)}
    return {"forecasts": []}


def load_board():
    """Latest saved board: the run's own files first, then the published site copy."""
    latest = artifacts_root() / "latest"
    paths = (
        str(latest / "forecast_board.json"),
        str(latest / "forecasts.csv"),
        str(project_root() / "web" / "public" / "forecasts.json"),
    )
    modified = tuple(Path(p).stat().st_mtime_ns if Path(p).exists() else 0 for p in paths)
    board = _load(paths, modified)
    frame = pd.DataFrame(board.get("forecasts", []))
    if frame.empty:
        return board, frame
    frame["kickoff"] = pd.to_datetime(frame["kickoff"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["kickoff"]).sort_values("kickoff").reset_index(drop=True)
    frame["home_win_probability"] = pd.to_numeric(frame["home_win_probability"], errors="coerce").clip(0, 1)
    if "projected_home_margin" not in frame:
        frame["projected_home_margin"] = frame.expected_home_score - frame.expected_away_score
    return board, frame


def fair_moneyline(probability):
    """Fair American odds from a win chance, the same conversion the forecast uses."""
    if not 0 < probability < 1:
        return None
    if probability >= 0.5:
        return -round(100 * probability / (1 - probability))
    return round(100 * (1 - probability) / probability)


def _odds(value):
    if value is None or pd.isna(value):
        return "—"
    value = int(round(float(value)))
    return f"+{value}" if value > 0 else str(value)


def _spread(value):
    if value is None or pd.isna(value):
        return "—"
    if abs(value) < 0.05:
        return "Pick’em"
    return f"+{value:.1f}" if value > 0 else f"{value:.1f}"


def _pick_note(game):
    lead = abs(float(game.projected_home_margin))
    if lead < 0.05:
        return "Even"
    team = game.home_team if game.home_win_probability >= 0.5 else game.away_team
    return f"{team} by {lead:.1f}"


def _week_name(board):
    week = board.get("week")
    return f"Week {week}" if isinstance(week, int) else "This week"


def _as_of(board, frame):
    stamp = pd.to_datetime(board.get("as_of"), utc=True, errors="coerce")
    if pd.isna(stamp) and len(frame):
        stamp = frame.kickoff.min()
    return stamp


def _week_games(board, frame):
    """The board's own week: games from the forecast date through the next eight days."""
    start = _as_of(board, frame)
    if pd.isna(start):
        return frame
    return frame[frame.kickoff < start + timedelta(days=8)]


def _filters(board, frame, key):
    left, right = st.columns([2, 1])
    query = left.text_input("Find a team", key=f"{key}-query", placeholder="Find a team", label_visibility="collapsed")
    week_name = _week_name(board)
    choice = right.segmented_control(
        "Which games", [week_name, "All games"], default=week_name, key=f"{key}-range", label_visibility="collapsed"
    )
    view = frame if choice == "All games" else _week_games(board, frame)
    text = query.strip().lower()
    if text:
        names = (view.home_team.astype(str) + " " + view.away_team.astype(str)).str.lower()
        view = view[names.str.contains(text, regex=False)]
    return view, query


def _summary(board, frame):
    week = _week_games(board, frame)
    chance = week.home_win_probability.where(week.home_win_probability >= 0.5, 1 - week.home_win_probability)
    from results import season_accuracy
    graded = season_accuracy()
    checked = f"{graded[0]:.1%}" if graded else "Not graded yet"
    items = [
        (f"{_week_name(board)} games", str(len(week)) if len(week) else "—"),
        ("Average chance for the pick", f"{chance.mean():.0%}" if len(week) else "—"),
        ("Close calls (under 60%)", str(int((chance < 0.6).sum())) if len(week) else "—"),
        ("Season accuracy so far", checked),
    ]
    st.html('<div class="summary">' + "".join(
        f'<div class="summary-item"><span>{escape(label)}</span><strong>{escape(value)}</strong></div>' for label, value in items
    ) + "</div>")


def _time(stamp):
    local = stamp.tz_convert(ZONE)
    return f"{local.strftime('%I:%M %p').lstrip('0')} {ZONE_LABEL}"


def _prediction_card(game):
    home = float(game.home_win_probability)
    pick_home = home >= 0.5
    rows = [
        ("", game.away_team, game.expected_away_score, 1 - home, not pick_home),
        ("at ", game.home_team, game.expected_home_score, home, pick_home),
    ]
    body = "".join(
        f'<div class="row{" picked" if picked else ""}"><span class="team"><em>{escape(prefix)}</em>{escape(str(team))}</span>'
        f'<span class="num">{round(float(points))}</span>'
        f'<span class="chance"><b>{chance:.0%}</b><i><i style="width:{chance * 100:.1f}%"></i></i></span></div>'
        for prefix, team, points, chance, picked in rows
    )
    return (
        f'<article class="card"><header><span>{_time(game.kickoff)}</span><span class="pick-note">{escape(_pick_note(game))}</span></header>'
        f'<div class="row head"><span>Team</span><span>Points</span><span>Win chance</span></div>{body}</article>'
    )


def _line_card(game):
    home = float(game.home_win_probability)
    home_spread = game.model_spread if "model_spread" in game._fields and pd.notna(game.model_spread) else -float(game.projected_home_margin)
    book = getattr(game, "market_spread", None)
    has_book = book is not None and pd.notna(book)
    odds = {
        "home": getattr(game, "home_moneyline", None),
        "away": getattr(game, "away_moneyline", None),
    }
    odds = {side: value if value is not None and pd.notna(value) else fair_moneyline(home if side == "home" else 1 - home) for side, value in odds.items()}
    rows = [
        ("away", "", game.away_team, -home_spread, odds["away"], -float(book) if has_book else None, home < 0.5),
        ("home", "at ", game.home_team, home_spread, odds["home"], float(book) if has_book else None, home >= 0.5),
    ]
    extra = " four" if has_book else ""
    head = '<span>Team</span><span>Spread</span><span>Fair odds</span>' + ("<span>Book</span>" if has_book else "")
    body = "".join(
        f'<div class="row{extra}{" picked" if picked else ""}"><span class="team"><em>{escape(prefix)}</em>{escape(str(team))}</span>'
        f'<span class="num">{_spread(spread)}</span><span class="num">{_odds(price)}</span>'
        + (f'<span class="num">{_spread(book_line)}</span>' if has_book else "")
        + "</div>"
        for _, prefix, team, spread, price, book_line, picked in rows
    )
    return (
        f'<article class="card"><header><span>{_time(game.kickoff)}</span><span class="pick-note">{escape(_pick_note(game))}</span></header>'
        f'<div class="row head{extra}">{head}</div>{body}</article>'
    )


def _days(view, card):
    if view.empty:
        return
    local = view.kickoff.dt.tz_convert(ZONE)
    for day, games in view.groupby(local.dt.date, sort=True):
        title = pd.Timestamp(day).strftime("%A, %B %-d")
        st.html(f'<section class="day"><h2>{escape(title)}</h2><div class="cards">'
                + "".join(card(game) for game in games.itertuples(index=False)) + "</div></section>")


def _heading(title, board, frame):
    stamp = _as_of(board, frame)
    made = f"Forecast made {stamp.tz_convert(ZONE).strftime('%a, %b %-d')}" if not pd.isna(stamp) else "No forecast published yet"
    count = f" · {len(frame)} games on the board" if len(frame) else ""
    st.html(f'<div class="intro"><h1>{escape(title)}</h1><p class="updated">{escape(made + count)}</p></div>')


def predictions_page():
    board, frame = load_board()
    _heading(f"{_week_name(board)} predictions", board, frame)
    if frame.empty:
        st.info("No forecast has been published yet. The board fills after the weekly run.")
        return
    _summary(board, frame)
    view, query = _filters(board, frame, "predictions")
    if view.empty:
        st.write(f"No games match “{query}”.")
    _days(view, _prediction_card)
    if len(view):
        export = view.assign(kickoff=view.kickoff.dt.tz_convert(ZONE).dt.strftime("%Y-%m-%d %H:%M"))
        columns = [c for c in ["kickoff", "away_team", "home_team", "expected_away_score", "expected_home_score", "winner", "home_win_probability"] if c in export]
        st.download_button("Download these games", export[columns].to_csv(index=False), "predictions.csv", "text/csv")


def odds_page():
    board, frame = load_board()
    _heading("Odds", board, frame)
    if frame.empty:
        st.info("No forecast has been published yet.")
        return
    _summary(board, frame)
    view, query = _filters(board, frame, "odds")
    st.html('<p class="lede">The spread and fair odds are our forecast written the way a sportsbook would write them. '
            'They are not sportsbook prices. A Book column appears when a sportsbook line was recorded before the forecast.</p>')
    if view.empty:
        st.write(f"No games match “{query}”.")
    _days(view, _line_card)


def how_it_works_page():
    st.html('<div class="intro"><h1>How it works</h1></div>')
    st.html("""<div class="prose">
<p>Every Sunday the forecast is made for the coming week using only results that were final at that time.</p>
<ol>
<li>Each team has an offense and a defense rating, updated after every game it plays.</li>
<li>Those ratings, plus each team's recorded rates and rest, give an expected score for both teams.</li>
<li>When a betting line was recorded before the forecast, it adjusts the expected score. Games without one use a version of the forecast that never relies on a line.</li>
<li>The difference between the two expected scores becomes a win chance. The pick is the team with the higher chance.</li>
<li>The spread and fair odds on the Odds page are the same forecast written the way a sportsbook would write it.</li>
<li>After the games finish, the forecast is checked against the real results before the next Sunday run.</li>
</ol>
<p>Weather and injury reports are recorded with each forecast. They only start to adjust the win chance once enough weeks have been recorded to show that they help.</p>
<p class="note">These are forecasts, not guarantees, and not betting advice.</p>
</div>""")
