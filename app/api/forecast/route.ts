import { NextResponse } from "next/server"
import { buildForecasts, type ForecastGame } from "@/lib/forecast-engine"

export const revalidate = 300
const ESPN = "https://site.api.espn.com/apis/site/v2/sports/football/college-football"

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
  const week = searchParams.get("week") ?? "4"
  const cutoff = new Date().toISOString()
  try {
    const responses = await Promise.all([1, 2, 3, 4, 5, 6].map(async (weekNumber) => {
      const response = await fetch(`${ESPN}/scoreboard?dates=${season}&week=${weekNumber}&seasontype=2&limit=300`, { next: { revalidate: 300 } })
      if (!response.ok) throw new Error(`ESPN returned ${response.status}`)
      return response.json()
    }))
    const games = new Map<string, ForecastGame>()
    responses.flatMap((response) => response.events ?? []).map(normalizeEvent).filter(Boolean).forEach((game) => games.set(game!.id, game!))
    const result = buildForecasts([...games.values()], cutoff)
    return NextResponse.json({ ...result, forecasts: result.forecasts.filter((game) => game.week === Number(week)), season: Number(season), selectedWeek: Number(week), source: "SportsDataverse-compatible ESPN site feed", dataCutoff: cutoff })
  } catch (error) {
    return NextResponse.json({ error: "Live source unavailable", detail: error instanceof Error ? error.message : "Unknown source error", source: "ESPN site API" }, { status: 503 })
  }
}
