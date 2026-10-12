import { NextResponse } from "next/server"
import { buildForecasts, type ForecastGame } from "@/lib/forecast-engine"

export const revalidate = 300
const ESPN = "https://site.api.espn.com/apis/site/v2/sports/football/college-football"
const SOURCES = [
  { name: "ESPN / SportsDataverse feed", url: ESPN, role: "live schedule, scores, rankings, odds" },
  { name: "cfbfastR data registry", url: "https://github.com/sportsdataverse/cfbfastR-cfb-data/blob/main/models/REGISTRY.md", role: "historical model and data definitions" },
  { name: "CFBfastr", url: "https://cfbfastr.sportsdataverse.org", role: "college football data package reference" },
  { name: "CFB.TXT", url: "https://cfbtxt.com", role: "results and schedule cross-check" },
  { name: "CollegeFootballData", url: "https://collegefootballdata.com", role: "FBS schedules and team statistics" },
  { name: "College Football Database", url: "https://www.collegefootballdb.com/college-football/stats/teamstat.html?group=...&cat=S&yr=2026&lg=FBS&conf=12", role: "team statistical context" },
  { name: "CollegeFootball.gg", url: "https://collegefootball.gg/tools", role: "public football tools" },
  { name: "CFB.TXT", url: "https://cfbtxt.com/results", role: "results cross-check" },
  { name: "TheScore", url: "https://www.thescore.com/ncaaf/events/2026-6", role: "schedule cross-check" },
  { name: "CFBTrack", url: "https://cfbtrack.com", role: "team and game context" },
  { name: "CFB Passport", url: "https://cfbpassport.com", role: "schedule context" },
  { name: "FanDuel NCAAF", url: "https://sportsbook.fanduel.com/navigation/ncaaf", role: "market lines when published" },
  { name: "CFB Games Tracker", url: "https://mrtimo.github.io/cfb-games/games-2026.html?conference=Big+Ten&week=Week+2&sort=thrill", role: "schedule cross-check" },
] as const

async function sourceStatus(source: (typeof SOURCES)[number]) {
  try {
    const response = await fetch(source.url, { method: "HEAD", signal: AbortSignal.timeout(3500), cache: "no-store" })
    return { ...source, status: response.ok ? "reachable" : "blocked", httpStatus: response.status }
  } catch {
    return { ...source, status: "unavailable", httpStatus: null }
  }
}

function normalizeEvent(event: any): ForecastGame | null {
  const competition = event.competitions?.[0]
  const home = competition?.competitors?.find((x: any) => x.homeAway === "home")
  const away = competition?.competitors?.find((x: any) => x.homeAway === "away")
  if (!home || !away) return null
  const odds = competition.odds?.[0]
  const spread = odds?.spread == null ? null : Number(odds.spread)
  const total = odds?.overUnder == null ? null : Number(odds.overUnder)
  return {
    id: String(event.id), date: event.date, week: Number(event.week?.number ?? 0),
    status: event.status?.type?.state ?? "pre",
    home: { id: String(home.team.id), name: home.team.displayName, abbreviation: home.team.abbreviation, score: home.score == null ? null : Number(home.score), rank: home.curatedRank?.current ?? null },
    away: { id: String(away.team.id), name: away.team.displayName, abbreviation: away.team.abbreviation, score: away.score == null ? null : Number(away.score), rank: away.curatedRank?.current ?? null },
    venue: competition.venue?.fullName ?? "Venue unavailable", spread, marketTotal: total, source: "ESPN site API"
  }
}

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url)
  const season = searchParams.get("season") ?? "2026"
  const week = searchParams.get("week") ?? "7"
  const cutoff = new Date().toISOString()
  try {
    const seasons = [Number(season) - 1, Number(season)]
    const requests = seasons.flatMap((year) => Array.from({ length: 15 }, (_, index) => ({ year, week: index + 1 })))
    const responses = await Promise.all(requests.map(async ({ year, week: weekNumber }) => {
      const response = await fetch(`${ESPN}/scoreboard?dates=${year}&week=${weekNumber}&seasontype=2&limit=300`, { next: { revalidate: 300 } })
      if (!response.ok) throw new Error(`ESPN returned ${response.status}`)
      return response.json()
    }))
    const games = new Map<string, ForecastGame>()
    responses.flatMap((response) => response.events ?? []).map(normalizeEvent).filter(Boolean).forEach((game) => games.set(game!.id, game!))
    const result = buildForecasts([...games.values()], cutoff)
    const sourceStatusResults = await Promise.all(SOURCES.map(sourceStatus))
    return NextResponse.json({ ...result, forecasts: result.forecasts.filter((game) => game.week === Number(week) && game.date.startsWith(`${season}-`)), season: Number(season), selectedWeek: Number(week), trainingSeasons: seasons, source: "ESPN/SportsDataverse-compatible FBS feed", sources: sourceStatusResults, dataCutoff: cutoff, methodology: { primary: "Point-in-time opponent-adjusted offense and defense", scoreLayer: "Gaussian margin with Poisson score check", betting: "Recorded spread comparison and fractional Kelly only when a real price exists", dataPolicy: "No simulated schedules, scores, player states, weather, or betting lines", weeklyControls: result.controls, backtestContract: result.dataContract } })
  } catch (error) {
    return NextResponse.json({ error: "Live source unavailable", detail: error instanceof Error ? error.message : "Unknown source error", source: "ESPN site API" }, { status: 503 })
  }
}
