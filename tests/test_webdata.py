"""Tests for external-source extraction and reconciliation.

The page snippets below are hand-written fixtures covering common page layouts
(table rows, two-line scoreboards, single-line result lists, picks, odds and
embedded JSON). They are not copies of any site's real markup.
"""
import json

import pandas as pd
import pytest

from cfb_forecast.webdata import collection as collect_mod
from cfb_forecast.webdata.extract import parse_html, parse_json, parse_text
from cfb_forecast.webdata.fetch import store
from cfb_forecast.webdata.interpret import GameBook, interpret, numbers, team_stats
from cfb_forecast.webdata.teams import TeamIndex

GAMES = [
    # game_id, week, start, away, home, away_pts, home_pts, completed
    (1, 1, "2026-08-29T19:00:00Z", "San José State", "USC", 26, 42, True),
    (2, 1, "2026-08-29T19:30:00Z", "NC State", "Virginia", 8, 34, True),
    (3, 1, "2026-08-29T16:00:00Z", "North Carolina", "TCU", 15, 10, True),
    (4, 7, "2026-10-13T23:00:00Z", "Delaware", "Middle Tennessee", None, None, False),
    (5, 7, "2026-10-17T19:00:00Z", "Miami (OH)", "Ohio", None, None, False),
    (6, 7, "2026-10-17T23:00:00Z", "Florida State", "Miami", None, None, False),
]
RETRIEVED = pd.Timestamp("2026-10-10T12:00:00Z")


def schedule_frame():
    rows = []
    for gid, week, start, away, home, ap, hp, done in GAMES:
        rows.append({"game_id": gid, "season": 2026, "week": week, "start_date": start, "completed": done,
                     "neutral_site": False, "home_id": hash(home) % 10**6, "away_id": hash(away) % 10**6,
                     "home_team": home, "away_team": away, "home_points": hp, "away_points": ap,
                     "home_division": "fbs", "away_division": "fbs",
                     "home_abbreviation": home[:3].upper(), "away_abbreviation": away[:3].upper()})
    return pd.DataFrame(rows)


@pytest.fixture
def book():
    from cfb_forecast.data import normalize_schedule
    return GameBook(normalize_schedule(schedule_frame()))


def ctx(**kw):
    return {"season": 2026, "retrieved_at": RETRIEVED, "final_only": False, "allow_past_picks": False, **kw}


def by_game(records):
    return {r["game_id"]: r for r in records}


def test_team_aliases_and_ranks():
    idx = TeamIndex(["Miami", "Miami (OH)", "Ohio", "Ohio State", "San José State", "Middle Tennessee", "Texas A&M"])
    assert idx.resolve("Miami (OH) RedHawks") == "Miami (OH)"
    assert idx.resolve("Miami Hurricanes") == "Miami"
    assert idx.resolve("#7 Ohio St.") == "Ohio State"
    assert idx.resolve("Ohio Bobcats") == "Ohio"
    assert idx.resolve("San Jose St") == "San José State"
    assert idx.resolve("Middle Tenn") == "Middle Tennessee"
    assert idx.resolve("Texas A&M Aggies") == "Texas A&M"
    assert idx.resolve("Alabama") is None


def test_table_row_winner_first_is_oriented_to_schedule(book):
    html = """<table id="schedule"><tr><th>Wk</th><th>Date</th><th>Winner</th><th>Pts</th><th></th><th>Loser</th><th>Pts</th></tr>
    <tr><td>1</td><td>Aug 29, 2026</td><td data-stat="winner_school_name">(14) USC</td><td>42</td><td></td>
        <td data-stat="loser_school_name">San Jose State</td><td>26</td></tr>
    <tr><td>1</td><td>Aug 29, 2026</td><td>North Carolina</td><td>15</td><td>N</td><td>TCU</td><td>10</td></tr>
    <tr><td>7</td><td>Oct 13, 2026</td><td>Middle Tennessee</td><td></td><td>@</td><td>Delaware</td><td></td></tr></table>"""
    res = by_game(interpret(parse_html(html), book, "results", ctx(final_only=True))["results"])
    assert (res[1]["home_points"], res[1]["away_points"], res[1]["is_final"]) == (42, 26, True)
    assert (res[3]["home_team"], res[3]["home_points"], res[3]["away_points"]) == ("TCU", 10, 15)
    assert 4 not in res  # future game without scores is not a result


def test_two_line_scoreboard_and_records_ignored(book):
    html = """<div class="game"><div>Final</div>
      <div class="team">North Carolina (1-0) <b>15</b></div>
      <div class="team">TCU (0-1) <b>10</b></div></div>
      <div class="game"><div>NC State 8</div><div>Virginia 34 Final</div></div>"""
    res = by_game(interpret(parse_html(html), book, "results", ctx())["results"])
    assert (res[3]["away_points"], res[3]["home_points"], res[3]["is_final"]) == (15, 10, True)
    assert (res[2]["away_points"], res[2]["home_points"]) == (8, 34)


def test_single_line_with_several_games(book):
    doc = parse_text("Final: San Jose St. 26, USC 42   NC State 8, Virginia 34 (F)\n")
    res = by_game(interpret(doc, book, "results", ctx())["results"])
    assert (res[1]["home_points"], res[1]["away_points"]) == (42, 26)
    assert (res[2]["home_points"], res[2]["away_points"]) == (34, 8)


def test_unscheduled_pairs_and_future_scores_rejected(book):
    doc = parse_text("USC 42 Virginia 34 Final\nDelaware 21 Middle Tennessee 24 Final\n")
    out = interpret(doc, book, "results", ctx())
    assert out["results"] == []


def test_picks_probability_spread_and_explicit_pick(book):
    doc = parse_text("Delaware at Middle Tennessee | Pick: Middle Tennessee -3.5 | Delaware 41% Middle Tennessee 59%\n"
                     "Florida State 24 Miami 31 prediction\n")
    picks = by_game(interpret(doc, book, "picks", ctx())["picks"])
    p = picks[4]
    assert p["pick_team"] == "Middle Tennessee"
    assert p["pick_home_win_prob"] == pytest.approx(0.59)
    assert p["pick_home_spread"] == -3.5
    assert (picks[6]["pick_home_score"], picks[6]["pick_away_score"], picks[6]["pick_team"]) == (31, 24, "Miami")


def test_picks_after_kickoff_rejected(book):
    doc = parse_text("North Carolina 70% TCU 30%\n")
    assert interpret(doc, book, "picks", ctx())["picks"] == []


def test_lines_with_prices_and_total(book):
    doc = parse_text("Delaware +3.5 (-110) +145 | Middle Tennessee -3.5 (-110) -175 | O/U 52.5\n")
    line = by_game(interpret(doc, book, "lines", ctx())["lines"])[4]
    assert line["home_spread"] == -3.5
    assert line["total_line"] == 52.5
    assert (line["home_ml"], line["away_ml"]) == (-175, 145)
    assert line["away_ml_decimal"] == pytest.approx(2.45)
    assert line["home_ml_decimal"] == pytest.approx(1 + 100 / 175)


def test_number_classification():
    n = numbers(" #12 (5-1) 7:30 PM ET Oct 11 -6.5 (-110) o48.5 63% 27 ")
    assert (n["spread"], n["total"], n["prob"], n["scores"]) == (-6.5, 48.5, 0.63, [27.0])


def test_embedded_json_keeps_scores_with_teams(book):
    blob = {"events": [{"id": "9", "status": {"type": {"completed": True}}, "competitors": [
        {"homeAway": "home", "score": "34", "team": {"displayName": "Virginia Cavaliers", "id": "258"}},
        {"homeAway": "away", "score": "8", "team": {"displayName": "NC State Wolfpack", "id": "152"}}]},
        {"homeTeam": {"name": "USC Trojans", "score": 42}, "awayTeam": {"name": "San Jose State Spartans", "score": 26}, "status": "final"}]}
    html = f'<html><script id="__NEXT_DATA__" type="application/json">{json.dumps(blob)}</script><div>menu</div></html>'
    for doc in (parse_html(html), parse_json(json.dumps(blob))):
        res = by_game(interpret(doc, book, "results", ctx())["results"])
        assert (res[2]["home_points"], res[2]["away_points"], res[2]["is_final"]) == (34, 8, True)
        assert (res[1]["home_points"], res[1]["away_points"]) == (42, 26)


def test_team_stats_table(book):
    html = """<table><tr><th>Team</th><th>Gms</th><th>PPG</th></tr>
      <tr><td>USC</td><td>6</td><td>38.5</td></tr><tr><td>Virginia</td><td>6</td><td>27.1</td></tr></table>"""
    rows = {r["team"]: r for r in team_stats(parse_html(html), book)}
    assert rows["USC"]["ppg"] == 38.5 and rows["Virginia"]["gms"] == 6


def test_offline_collect_publishes_consensus_market_and_results(tmp_path, monkeypatch):
    data, artifacts = tmp_path / "data", tmp_path / "artifacts"
    data.mkdir()
    schedule_frame().to_csv(data / "schedule_2026.csv", index=False)
    when = RETRIEVED.to_pydatetime()
    store("cfbtxt_results", "https://cfbtxt.com/results/", b"USC 42 San Jose State 26 Final\nNC State 8 Virginia 34 Final\n",
          "text/plain", data, retrieved_at=when)
    store("cfbtrack_scores", "https://cfbtrack.com/scores", b"<div>San Jose St. 26</div><div>USC 41 Final</div>", "text/html", data, retrieved_at=when)
    store("fanduel_sportsbook", "https://sportsbook.fanduel.com/navigation/ncaaf",
          b"<div>Delaware +3.5 +145</div><div>Middle Tennessee -3.5 -175</div><div>O/U 52.5</div>", "text/html", data, retrieved_at=when)
    report = collect_mod.collect(data, 2026, artifacts, names=["cfbtxt_results", "cfbtrack_scores", "fanduel_sportsbook"], offline=True)
    assert report["sources"]["cfbtxt_results"]["results"] == 2
    assert report["sources"]["fanduel_sportsbook"]["lines"] == 1
    cons = pd.read_csv(data / "sources" / "results_crosscheck.csv").set_index("game_id")
    assert not cons.loc[1, "sources_agree"]          # 42 vs 41: conflict is flagged, not averaged
    assert cons.loc[2, "schedule_agrees"]
    exported = pd.read_csv(artifacts / "results" / "sources" / "results.csv")
    assert exported.game_id.tolist() == [2]          # only agreed, schedule-consistent finals are published
    market = pd.read_csv(data / "market_history.csv")
    assert market.loc[0, "provider"] == "fanduel_sportsbook" and market.loc[0, "home_spread"] == -3.5
    assert (data / "sources" / "collections" / "latest.json").exists()


def test_import_file_with_declared_publication_time(tmp_path):
    data, artifacts = tmp_path / "data", tmp_path / "artifacts"
    data.mkdir()
    schedule_frame().to_csv(data / "schedule_2026.csv", index=False)
    board = tmp_path / "board.txt"
    board.write_text("Delaware 20 Middle Tennessee 27\nMiami (OH) 17 Ohio 24 prediction\n")
    summary = collect_mod.import_file(board, "predictions_board", data, 2026, artifacts, observed_at="2026-10-08T15:00:00Z")
    assert summary["picks"] == 2
    picks = pd.read_csv(data / "sources" / "picks.csv").set_index("game_id")
    assert picks.loc[5, "pick_team"] == "Ohio" and picks.loc[5, "retrieved_at"].startswith("2026-10-08T15:00:00")


def _tiny_pdf(lines):
    """A minimal one-page PDF with one text line per entry (Helvetica)."""
    text = " ".join(f"({line}) Tj 0 -16 Td" for line in lines)
    stream = f"BT /F1 11 Tf 40 760 Td {text} ET".encode()
    objects = [b"<< /Type /Catalog /Pages 2 0 R >>",
               b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
               b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
               b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
               b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream"]
    out, offsets = bytearray(b"%PDF-1.4\n"), []
    for i, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    out += b"".join(b"%010d 00000 n \n" % o for o in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objects) + 1, xref)
    return bytes(out)


def test_pdf_predictions_board_import(tmp_path):
    data, artifacts = tmp_path / "data", tmp_path / "artifacts"
    data.mkdir()
    schedule_frame().to_csv(data / "schedule_2026.csv", index=False)
    pdf = tmp_path / "board.pdf"
    pdf.write_bytes(_tiny_pdf(["CFB Predictions Board", "Delaware 20 Middle Tennessee 27 (73%)", "Florida State 21 Miami 30"]))
    summary = collect_mod.import_file(pdf, "predictions_board", data, 2026, artifacts, observed_at="2026-10-08T15:00:00Z")
    assert summary["picks"] == 2
    picks = pd.read_csv(data / "sources" / "picks.csv").set_index("game_id")
    assert picks.loc[4, "pick_team"] == "Middle Tennessee"
    assert picks.loc[4, "pick_home_win_prob"] == pytest.approx(0.73)
    assert (picks.loc[6, "pick_home_score"], picks.loc[6, "pick_away_score"]) == (30, 21)
