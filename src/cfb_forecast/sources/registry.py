"""Registered external sources.

``kind`` decides how a page is interpreted:

* ``results``    final scores (cross-checked against ESPN and each other)
* ``picks``      published predictions (graded like the model, never used as model inputs)
* ``lines``      betting lines, stamped with the retrieval time and appended to market history
* ``team_stats`` season team statistics tables
* ``auto``       results, picks and lines are all looked for

URLs may use ``{season}`` and ``{week}``. ``final_only`` marks pages that only
show a score after the game ends, so a score there counts as final once the
game is four hours past kickoff.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Source:
    name: str
    url: str
    kind: str
    description: str
    final_only: bool = False
    min_interval: float = 3.0
    needs_week: bool = False
    respect_robots: bool = True
    tags: tuple = field(default_factory=tuple)

    def resolve(self, season, week=None):
        if self.needs_week and week is None:
            raise ValueError(f"{self.name} needs --week")
        return self.url.format(season=season, week=week)


SOURCES = {s.name: s for s in [
    Source("sports_reference", "https://www.sports-reference.com/cfb/years/{season}-schedule.html", "results",
           "Sports-Reference season schedule and final scores", final_only=True, min_interval=4.0, tags=("results",)),
    Source("cfbtrack", "https://cfbtrack.com/", "auto", "CFB Track home page", tags=("results", "picks")),
    Source("cfbtrack_scores", "https://cfbtrack.com/scores", "results", "CFB Track scoreboard", tags=("results",)),
    Source("cfbtxt_results", "https://cfbtxt.com/results/", "results", "cfbtxt results list", final_only=True, tags=("results",)),
    Source("thescore", "https://www.thescore.com/ncaaf/events/{season}-{week}", "results",
           "theScore weekly NCAAF events", needs_week=True, tags=("results",)),
    Source("mrtimo_games", "https://mrtimo.github.io/cfb-games/games-{season}.html", "results",
           "mrtimo cfb-games season list", final_only=True, tags=("results",)),
    Source("footballdb_teamstats", "https://www.footballdb.com/college-football/stats/teamstat.html?cat=S&yr={season}&lg=FBS&conf=12",
           "team_stats", "FootballDB FBS team scoring stats (conference 12)", tags=("stats",)),
    Source("fanduel_sportsbook", "https://sportsbook.fanduel.com/navigation/ncaaf", "lines",
           "FanDuel Sportsbook NCAAF lines", min_interval=5.0, tags=("lines",)),
    Source("fanduel_research", "https://www.fanduel.com/research/ncaaf", "picks", "FanDuel Research NCAAF picks", tags=("picks",)),
    Source("pickswise", "https://www.pickswise.com/college-football/", "picks", "Pickswise college football picks", tags=("picks",)),
    Source("pickdawgz", "https://stats.pickdawgz.com/tools", "auto", "PickDawgz stats tools", tags=("picks", "lines")),
    Source("sportsforecaster", "https://sportsforecaster.com/", "picks", "SportsForecaster predictions", tags=("picks",)),
    Source("cfbpassport", "https://cfbpassport.com/", "auto", "CFB Passport", tags=("results", "picks")),
    Source("collegefootball_gg", "https://collegefootball.gg/tools/", "auto", "collegefootball.gg tools", tags=("picks", "lines")),
]}

# Files supplied by hand (saved pages, exports, PDFs) are imported with
# `cfb-forecast sources-import --source NAME --file PATH`; NAME may be any of the
# above or a new label such as this one for the weekly predictions-board PDF.
IMPORT_ONLY = {
    "predictions_board": Source("predictions_board", "", "picks", "CFB Predictions Board PDF (manual import)", tags=("picks",)),
}

# Not collected: https://en.wikipedia.org/wiki/Statistical_association_football_predictions is an
# encyclopedia article about soccer prediction methods, not a data feed.


def get_source(name):
    if name in SOURCES:
        return SOURCES[name]
    if name in IMPORT_ONLY:
        return IMPORT_ONLY[name]
    return Source(name, "", "auto", f"Imported file source '{name}'")


def select(names=None, tag=None):
    chosen = [SOURCES[n] for n in names] if names else list(SOURCES.values())
    if tag:
        wanted = {tag} if isinstance(tag, str) else set(tag)
        chosen = [s for s in chosen if wanted & set(s.tags)]
    return chosen
