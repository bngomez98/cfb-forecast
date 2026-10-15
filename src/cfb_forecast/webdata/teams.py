"""Resolve free-text team names from any source to the schedule's canonical names.

Sites spell teams differently ("Ohio St.", "Miami (FL)", "UConn Huskies",
"(5) Georgia"). Every name is reduced to a normalized token sequence and matched
longest-first against the schedule's names, their abbreviations, automatic
"State"/"St." variants and a curated alias list. A row from a source is only
accepted when both of its teams resolve and those two teams are scheduled to
play each other, which rejects stray text that merely looks like a score.
"""
from __future__ import annotations

import re
import unicodedata

import pandas as pd

# Canonical schedule name -> alternative spellings used by other sites.
ALIASES = {
    "App State": ["Appalachian State", "Appalachian St", "App St"],
    "BYU": ["Brigham Young"],
    "Bowling Green": ["Bowling Green State", "BGSU"],
    "Central Michigan": ["CMU"],
    "Coastal Carolina": ["Coastal"],
    "East Carolina": ["ECU"],
    "Florida Atlantic": ["FAU"],
    "Florida International": ["FIU", "Florida Intl"],
    "Hawai'i": ["Hawaii"],
    "Jacksonville State": ["Jax State", "Jacksonville St"],
    "Kennesaw State": ["Kennesaw"],
    "LSU": ["Louisiana State"],
    "Louisiana": ["Louisiana-Lafayette", "Louisiana Lafayette", "UL Lafayette", "ULL"],
    "UL Monroe": ["Louisiana-Monroe", "Louisiana Monroe", "ULM"],
    "Massachusetts": ["UMass"],
    "Miami": ["Miami (FL)", "Miami FL", "Miami Florida", "Miami-Florida", "Miami Hurricanes"],
    "Miami (OH)": ["Miami Ohio", "Miami OH", "Miami-Ohio", "Miami RedHawks", "Miami (Ohio)"],
    "Middle Tennessee": ["Middle Tennessee State", "MTSU", "Middle Tenn"],
    "NC State": ["North Carolina State", "N.C. State"],
    "Northern Illinois": ["NIU"],
    "Ole Miss": ["Mississippi"],
    "Pittsburgh": ["Pitt"],
    "SMU": ["Southern Methodist"],
    "San José State": ["San Jose State", "SJSU"],
    "Sam Houston": ["Sam Houston State"],
    "South Florida": ["USF"],
    "Southern Miss": ["Southern Mississippi"],
    "TCU": ["Texas Christian"],
    "Texas A&M": ["TAMU"],
    "UAB": ["Alabama-Birmingham", "Alabama Birmingham"],
    "UCF": ["Central Florida"],
    "UConn": ["Connecticut"],
    "UNLV": ["Nevada-Las Vegas"],
    "USC": ["Southern California", "Southern Cal"],
    "UTEP": ["Texas-El Paso", "Texas El Paso"],
    "UTSA": ["Texas-San Antonio", "UT San Antonio"],
    "Western Kentucky": ["WKU"],
    "Western Michigan": ["WMU"],
    "Army": ["Army West Point"],
}

_RANK = re.compile(r"(?:\(\s*\d{1,2}\s*\)|#\s?\d{1,2}\b|\bNo\.\s?\d{1,2}\b)")


def normalize(text):
    """Lowercase ASCII tokens: accents stripped, '&' -> 'and', punctuation removed."""
    text = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode()
    text = text.replace("&", " and ").replace("'", "").replace("’", "")
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def strip_ranks(text):
    return _RANK.sub(" ", str(text))


def _variants(name):
    out = {name}
    if name.endswith(" State"):
        base = name[: -len(" State")]
        out |= {f"{base} St", f"{base} St."}
    return out


class TeamIndex:
    """Longest-match lookup of team mentions inside arbitrary text."""

    def __init__(self, names, abbreviations=None):
        self.names = sorted(set(names))
        self._phrases = {}
        for name in self.names:
            for alias in _variants(name) | set(ALIASES.get(name, [])):
                key = tuple(normalize(alias).split())
                if key:
                    # Never let an alias shadow a different real team's own name.
                    if key in self._phrases and normalize(self._phrases[key]) == " ".join(key):
                        continue
                    self._phrases[key] = name
        self._max = max((len(k) for k in self._phrases), default=1)
        # Abbreviations only match as standalone uppercase tokens ("UGA", "OSU" excluded if ambiguous).
        counts = pd.Series(list((abbreviations or {}).values())).value_counts() if abbreviations else pd.Series(dtype=int)
        self._abbr = {a: n for n, a in (abbreviations or {}).items() if a and counts.get(a, 0) == 1 and len(a) >= 2}

    @classmethod
    def from_schedule(cls, schedule):
        games = schedule[schedule.get("fbs_game", True).astype(bool)] if "fbs_game" in schedule else schedule
        names = pd.concat([games.home_team, games.away_team]).dropna().astype(str)
        abbr = {}
        for side in ("home", "away"):
            col = f"{side}_abbreviation"
            if col in games:
                for n, a in zip(games[f"{side}_team"], games[col]):
                    if isinstance(a, str):
                        abbr[str(n)] = a.upper()
        return cls(names.unique(), abbr)

    def resolve(self, text):
        """Return the canonical name if ``text`` is exactly one team, else None."""
        found = self.find(text)
        return found[0][0] if len(found) == 1 else None

    def find(self, text):
        """List of (canonical_name, start, end) character spans, longest match first, in order."""
        text = strip_ranks(text)
        tokens = [(m.group(0), m.start(), m.end()) for m in re.finditer(r"[A-Za-z0-9À-ɏ'&’]+", text)]
        norm = [normalize(t[0]).split() for t in tokens]
        hits, i = [], 0
        while i < len(tokens):
            matched = False
            for length in range(min(self._max, len(tokens) - i), 0, -1):
                key = tuple(w for part in norm[i:i + length] for w in part)
                if key in self._phrases:
                    hits.append((self._phrases[key], tokens[i][1], tokens[i + length - 1][2]))
                    i += length
                    matched = True
                    break
            if not matched:
                raw = tokens[i][0]
                if raw.isupper() and raw in self._abbr:
                    hits.append((self._abbr[raw], tokens[i][1], tokens[i][2]))
                i += 1
        return hits
