"""Interpret extracted lines as results, picks, betting lines or team stats.

A candidate is accepted only when two team mentions resolve to teams that the
schedule says play each other. The numbers next to each team are then
classified (score, spread, moneyline, total, win probability) and oriented to
the schedule's home/away sides, so source ordering never flips a result.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

from .teams import TeamIndex, normalize

KINDS = ("results", "picks", "lines", "team_stats")

_MONTHS = {m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}
_DATE_ISO = re.compile(r"\b(20\d\d)-(\d{2})-(\d{2})")
_DATE_US = re.compile(r"\b(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?\b")
_DATE_TEXT = re.compile(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+(\d{1,2})(?:st|nd|rd|th)?(?:,?\s+(20\d\d))?", re.I)

_NOISE = [
    re.compile(r"\b(20\d\d)-(\d{2})-(\d{2})(?:[T ][\d:.]+(?:Z|[+-]\d{2}:?\d{2})?)?"),
    _DATE_US,
    _DATE_TEXT,
    re.compile(r"\b\d{1,2}:\d{2}(?:\s*[ap]\.?m\.?)?(?:\s*[ECMP][SD]?T\b)?", re.I),
    re.compile(r"\(\s*\d{1,2}\s*\)|#\s?\d{1,2}\b|\bNo\.\s?\d{1,2}\b"),  # poll ranks
    re.compile(r"\b20\d\d\b"),
    re.compile(r"\bweek\s*\d+\b", re.I),
    re.compile(r"\((?:\d{1,2}-\d{1,2}(?:-\d{1,2})?)(?:[,;]\s*\d{1,2}-\d{1,2}(?:\s*\w+)?)*\)"),  # W-L records
    re.compile(r"\b(?:Q[1-4]|[1-4](?:st|nd|rd|th))\b", re.I),
    re.compile(r"\b\d+(?:st|nd|rd|th)\b", re.I),
]
_JUICE = re.compile(r"\(\s*[+-]\d{3,4}\s*\)")  # spread/total prices such as (-110)
_PERCENT = re.compile(r"(?<![\d.])(\d{1,3}(?:\.\d+)?)\s*%")
_TOTAL = re.compile(r"(?:\bo/u\b|\bover/under\b|\btotal\b|\bou\b|(?<![a-z])[ou](?=\s?\d))\s*:?\s*(\d{2,3}(?:\.\d)?)", re.I)
_ML = re.compile(r"(?<![\w.])([+-]\d{3,4})(?![\d.])")
_SPREAD = re.compile(r"(?<![\w.])([+-]\d{1,2}(?:\.\d)?)(?![\d.%])|\b(pk|pick'?em|even)\b", re.I)
_PAIR = re.compile(r"(?<![\d.])(\d{1,2})\s*[-–]\s*(\d{1,2})(?![\d.])")
_SCORE = re.compile(r"(?<![\w.+\-])(\d{1,2}(?:\.\d)?)(?![\d.%:/])")
_FINAL = re.compile(r"(?i:\bfinal\b)|\bF/\d?OT\b|(?<![\w/])F(?![\w/])")
_LIVE = re.compile(r"\b(?:Q[1-4]|1st|2nd|3rd|4th|halftime|half|HT|in progress|live)\b", re.I)
_PICK = re.compile(r"\b(?:pick|prediction|predicted winner|winner|best bet|our pick|projected winner|take|lean)\b\s*[:\-]?\s*", re.I)


@dataclass
class Game:
    game_id: int
    season: int
    week: int
    kickoff: pd.Timestamp
    home: str
    away: str


class GameBook:
    """Scheduled FBS games indexed by unordered team pair."""

    def __init__(self, schedule):
        schedule = schedule.copy()
        schedule["start_date"] = pd.to_datetime(schedule.start_date, utc=True)
        if "fbs_game" in schedule:
            schedule = schedule[schedule.fbs_game.astype(str).str.lower().isin(["true", "1"])]
        self.index = TeamIndex.from_schedule(schedule)
        self.pairs = {}
        for r in schedule.itertuples(index=False):
            g = Game(int(r.game_id), int(r.season), int(r.week), r.start_date, str(r.home_team), str(r.away_team))
            self.pairs.setdefault(frozenset((g.home, g.away)), []).append(g)
        for games in self.pairs.values():
            games.sort(key=lambda g: g.kickoff)

    def choose(self, a, b, kind, retrieved_at, hint=None):
        games = self.pairs.get(frozenset((a, b)))
        if not games:
            return None
        if hint is not None:
            near = min(games, key=lambda g: abs(g.kickoff - hint))
            if abs(near.kickoff - hint) <= pd.Timedelta(days=4):
                return near
        if len(games) == 1:
            return games[0]
        if kind == "results":
            past = [g for g in games if g.kickoff <= retrieved_at]
            return past[-1] if past else None
        future = [g for g in games if g.kickoff >= retrieved_at - pd.Timedelta(hours=6)]
        return future[0] if future else None


def date_hint(text, season, retrieved_at):
    m = _DATE_ISO.search(text)
    if m:
        return pd.Timestamp(f"{m.group(1)}-{m.group(2)}-{m.group(3)}", tz="UTC")
    m = _DATE_TEXT.search(text)
    if m:
        month, day, year = _MONTHS[m.group(1)[:3].lower()], int(m.group(2)), m.group(3)
    else:
        m = _DATE_US.search(text)
        if not m:
            return None
        month, day, year = int(m.group(1)), int(m.group(2)), m.group(3)
        if not (1 <= month <= 12 and 1 <= day <= 31):
            return None
    if year:
        year = int(year) + (2000 if len(str(year)) == 2 else 0)
    else:
        year = season + 1 if month <= 2 else season  # bowl season runs into January
    try:
        return pd.Timestamp(year=year, month=month, day=day, tz="UTC") + pd.Timedelta(hours=18)
    except ValueError:
        return None


def _clean(segment):
    for pattern in _NOISE:
        segment = pattern.sub(" ", segment)
    return segment


def numbers(segment):
    """Classify the numbers in one team's text segment."""
    seg = _JUICE.sub(" ", _clean(segment))
    out = {"prob": None, "total": None, "ml": None, "spread": None, "scores": []}
    m = _PERCENT.search(seg)
    if m and float(m.group(1)) <= 100:
        out["prob"] = float(m.group(1)) / 100
    seg = _PERCENT.sub(" ", seg)
    m = _TOTAL.search(seg)
    if m and 20 <= float(m.group(1)) <= 120:
        out["total"] = float(m.group(1))
    seg = _TOTAL.sub(" ", seg)
    m = _ML.search(seg)
    if m:
        out["ml"] = int(m.group(1))
    seg = _ML.sub(" ", seg)
    m = _SPREAD.search(seg)
    if m:
        out["spread"] = 0.0 if m.group(2) else float(m.group(1))
        if abs(out["spread"]) > 60:
            out["spread"] = None
    seg = _SPREAD.sub(" ", seg)
    m = _PAIR.search(seg)
    if m:
        out["pair"] = (float(m.group(1)), float(m.group(2)))
        seg = _PAIR.sub(" ", seg)
    out["scores"] = [float(x) for x in _SCORE.findall(seg)]
    return out


def american_to_decimal(ml):
    if ml is None or ml == 0:
        return None
    return 1 + (ml / 100 if ml > 0 else 100 / abs(ml))


def _groups(hits, book):
    """Group consecutive team mentions into games.

    Two consecutive mentions of teams scheduled to play each other open a group;
    further consecutive mentions of either team ("A at B | Pick: B | A 41% B 59%")
    extend it. Each group is (start_index, end_index_exclusive).
    """
    i = 0
    while i + 1 < len(hits):
        a, b = hits[i][0], hits[i + 1][0]
        if a != b and frozenset((a, b)) in book.pairs:
            j = i + 2
            while j < len(hits) and hits[j][0] in (a, b):
                j += 1
            yield i, j
            i = j
        else:
            i += 1


def _merge(parts):
    """Combine the number summaries of every segment that follows one team's mentions."""
    out = {"prob": None, "total": None, "ml": None, "spread": None, "scores": []}
    for p in parts:
        for k in ("prob", "total", "ml", "spread"):
            if out[k] is None and p[k] is not None:
                out[k] = p[k]
        out["scores"].extend(p["scores"])
        if "pair" in p and "pair" not in out:
            out["pair"] = p["pair"]
    return out


def interpret_window(text, book, kind, ctx, status_text=""):
    """Yield candidate records for every scheduled pairing mentioned in ``text``."""
    hits = book.index.find(text)
    if len(hits) < 2:
        return
    groups = list(_groups(hits, book))
    for n, (start, stop) in enumerate(groups):
        seg_end = hits[groups[n + 1][0]][1] if n + 1 < len(groups) else len(text)
        seg_start = hits[groups[n - 1][1] - 1][2] if n > 0 else 0
        team_a, team_b = hits[start][0], hits[start + 1][0]
        segs = {team_a: [], team_b: []}
        for k in range(start, stop):
            nxt = hits[k + 1][1] if k + 1 < stop else seg_end
            segs[hits[k][0]].append(numbers(text[hits[k][2]:nxt]))
        na, nb = _merge(segs[team_a]), _merge(segs[team_b])
        context = text[seg_start:seg_end]
        sa = na["scores"][0] if na["scores"] else None
        sb = nb["scores"][0] if nb["scores"] else None
        layout = "adjacent"
        if sa is None and sb is None and "pair" in na and "pair" not in nb:
            sa, sb = na["pair"]
            layout = "dash-pair"
        elif sa is None and len(nb["scores"]) == 2 and " | " in text:
            sa, sb = nb["scores"]
            layout = "columns"
        hint = date_hint(context, ctx["season"], ctx["retrieved_at"])
        choose_kind = kind if kind != "auto" else ("results" if _FINAL.search(context) else "picks")
        game = book.choose(team_a, team_b, choose_kind, ctx["retrieved_at"], hint)
        if game is None:
            continue
        a_home = team_a == game.home
        home_n, home_s = (na, sa) if a_home else (nb, sb)
        away_n, away_s = (nb, sb) if a_home else (na, sa)
        base = {
            "game_id": game.game_id, "season": game.season, "week": game.week, "kickoff": game.kickoff.isoformat(),
            "home_team": game.home, "away_team": game.away, "layout": layout, "evidence": context.strip()[:300],
        }
        group_hits = hits[start:stop]
        for k in ([kind] if kind != "auto" else ["results", "picks", "lines"]):
            rec = _build(k, base, home_n, away_n, home_s, away_s, context, group_hits, game, ctx, status_text)
            if rec is not None:
                yield k, rec


def _build(kind, base, hn, an, hs, as_, context, hits, game, ctx, status_text=""):
    retrieved = ctx["retrieved_at"]
    if kind == "results":
        if hs is None or as_ is None or hs != int(hs) or as_ != int(as_):
            return None
        status = f"{context} {status_text}"
        final = bool(_FINAL.search(status)) or (ctx.get("final_only") and game.kickoff + pd.Timedelta(hours=4) <= retrieved)
        if _LIVE.search(status) and not _FINAL.search(status):
            final = False
        if game.kickoff > retrieved:
            return None  # a score for a game that has not started is not a result
        if final and hs == as_:
            return None  # college football games cannot end tied
        return {**base, "home_points": int(hs), "away_points": int(as_), "is_final": bool(final)}
    if kind == "lines":
        if game.kickoff <= retrieved:
            return None  # only pregame lines are usable market information
        spread = hn["spread"] if hn["spread"] is not None else (-an["spread"] if an["spread"] is not None else None)
        total = hn["total"] or an["total"]
        rec = {**base, "home_spread": spread, "total_line": total,
               "home_ml": hn["ml"], "away_ml": an["ml"],
               "home_ml_decimal": american_to_decimal(hn["ml"]), "away_ml_decimal": american_to_decimal(an["ml"])}
        return rec if any(rec[k] is not None for k in ("home_spread", "total_line", "home_ml_decimal", "away_ml_decimal")) else None
    if kind == "picks":
        pick = None
        for m in _PICK.finditer(context):
            after = [h for h in hits if 0 <= h[1] - m.end() <= 3 and h[0] in (game.home, game.away)]
            if after:
                pick = after[0][0]
                break
        prob = None
        if hn["prob"] is not None and an["prob"] is not None and hn["prob"] + an["prob"] > 0:
            prob = hn["prob"] / (hn["prob"] + an["prob"])
        elif hn["prob"] is not None:
            prob = hn["prob"]
        elif an["prob"] is not None:
            prob = 1 - an["prob"]
        spread = hn["spread"] if hn["spread"] is not None else (-an["spread"] if an["spread"] is not None else None)
        if pick is None and hs is not None and as_ is not None and hs != as_:
            pick = game.home if hs > as_ else game.away
        if pick is None and prob is not None and prob != 0.5:
            pick = game.home if prob > 0.5 else game.away
        if pick is None and prob is None and (hs is None or as_ is None):
            return None
        if game.kickoff <= retrieved - pd.Timedelta(hours=6) and not ctx.get("allow_past_picks"):
            return None  # a "pick" published after the game is not a forecast
        return {**base, "pick_team": pick, "pick_home_win_prob": prob, "pick_home_score": hs, "pick_away_score": as_, "pick_home_spread": spread}
    return None


def interpret(doc, book, kind, ctx, max_window=4):
    """All accepted records from a document, deduplicated per (kind, game_id)."""
    records, consumed = {}, set()
    lines = doc.lines
    has_team = [bool(book.index.find(line)) for line in lines]
    is_status = [not t and bool(_FINAL.search(line) or _LIVE.search(line)) and len(line) < 40 for line, t in zip(lines, has_team)]
    # Scoreboards put a status line ("Final", "Q3 4:12") either before or after each game's team lines.
    before = sum(1 for i in range(len(lines) - 1) if is_status[i] and has_team[i + 1])
    after = sum(1 for i in range(1, len(lines)) if is_status[i] and has_team[i - 1])
    status_offset = -1 if before > after else (1 if after > before else 0)
    for width in range(1, max_window + 1):
        for i in range(len(lines) - width + 1):
            span = range(i, i + width)
            if any(j in consumed for j in span):
                continue
            text = "  ||  ".join(lines[i:i + width])
            j = i - 1 if status_offset < 0 else i + width
            status_text = lines[j] if status_offset and 0 <= j < len(lines) and is_status[j] else ""
            found = False
            for k, rec in interpret_window(text, book, kind, ctx, status_text):
                records.setdefault((k, rec["game_id"]), rec)
                found = True
            if found:
                consumed.update(span)
    out = {k: [] for k in ("results", "picks", "lines")}
    for (k, _), rec in records.items():
        out[k].append(rec)
    return out


def team_stats(doc, book):
    """Rows of team-level stat tables: one record per resolved team with header-named columns."""
    records, header = [], None
    for row in doc.rows:
        texts = [c["text"] for c in row]
        teams = [book.index.resolve(t) for t in texts[:3]]
        named = [t for t in teams if t]
        numeric = sum(bool(re.fullmatch(r"[-+]?[\d,.]+%?", t)) for t in texts)
        if not named and numeric <= len(texts) // 3 and len(texts) >= 3:
            header = [c["stat"] or c["text"] or f"col{i}" for i, c in enumerate(row)]
            continue
        if len(named) != 1 or header is None:
            continue
        rec = {"team": named[0]}
        for i, cell in enumerate(row):
            name = cell["stat"] or (header[i] if i < len(header) else f"col{i}")
            key = normalize(name).replace(" ", "_") or f"col{i}"
            value = cell["text"].replace(",", "")
            rec[key] = pd.to_numeric(value.rstrip("%"), errors="coerce") if re.fullmatch(r"[-+]?[\d.]+%?", value) else cell["text"]
        records.append(rec)
    return records
