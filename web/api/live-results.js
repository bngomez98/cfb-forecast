const SCOREBOARD = 'https://site.api.espn.com/apis/site/v2/sports/football/college-football/scoreboard'
const DAY = 24 * 60 * 60 * 1000

function dateKey(date) {
  return date.toISOString().slice(0, 10).replaceAll('-', '')
}

function parseDate(value, fallback) {
  if (value === undefined) return fallback
  if (typeof value !== 'string' || !value.trim()) throw new Error('Invalid date')
  const parsed = new Date(value)
  if (!Number.isFinite(parsed.getTime())) throw new Error('Invalid date')
  return parsed
}

function score(value) {
  if (value == null || value === '') return null
  const number = Number(value)
  return Number.isInteger(number) && number >= 0 ? number : null
}

function eventResult(event) {
  const competition = event.competitions?.[0]
  const competitors = competition?.competitors ?? []
  const home = competitors.find(team => team.homeAway === 'home')
  const away = competitors.find(team => team.homeAway === 'away')
  const type = competition?.status?.type ?? event.status?.type ?? {}
  const id = Number(event.id)
  if (!Number.isInteger(id) || id <= 0 || !home || !away) return null
  return {
    game_id: id,
    status: type.completed ? 'final' : type.state === 'in' ? 'live' : 'scheduled',
    status_detail: type.shortDetail ?? type.detail ?? null,
    period: competition?.status?.period ?? null,
    clock: competition?.status?.displayClock ?? null,
    home_points: score(home.score),
    away_points: score(away.score),
  }
}

export default async function handler(request, response) {
  if (request.method !== 'GET') return response.status(405).json({ error: 'Method not allowed' })
  const season = Number(request.query.season ?? 2026)
  if (!Number.isInteger(season) || season < 2000 || season > 2100) {
    return response.status(400).json({ error: 'Invalid season' })
  }
  const now = new Date()
  let start, end
  try {
    start = parseDate(request.query.start, new Date(now.getTime() - DAY))
    end = parseDate(request.query.end, new Date(now.getTime() + 7 * DAY))
    if (end < start || end - start > 9 * DAY) throw new Error('Invalid range')
  } catch {
    return response.status(400).json({ error: 'Provide valid dates in increasing order, at most nine days apart' })
  }
  const firstDay = Date.UTC(start.getUTCFullYear(), start.getUTCMonth(), start.getUTCDate())
  const lastDay = Date.UTC(end.getUTCFullYear(), end.getUTCMonth(), end.getUTCDate())
  const dates = Array.from({ length: (lastDay - firstDay) / DAY + 1 }, (_, index) => dateKey(new Date(firstDay + index * DAY)))
  try {
    const payloads = await Promise.all(dates.map(async date => {
      const url = `${SCOREBOARD}?dates=${date}&groups=80&limit=500`
      const upstream = await fetch(url, {
        headers: { 'User-Agent': 'fieldquant-cfb/1.0' }, signal: AbortSignal.timeout(15_000),
      })
      if (!upstream.ok) throw new Error(`Score provider returned ${upstream.status}`)
      return upstream.json()
    }))
    const rows = payloads.flatMap(payload => payload.events ?? [])
      .filter(event => Number(event.season?.year) === season).map(eventResult).filter(Boolean)
    const results = [...new Map(rows.map(row => [row.game_id, row])).values()]
    response.setHeader('Cache-Control', 's-maxage=30, stale-while-revalidate=30')
    return response.status(200).json({ season, updated_at: new Date().toISOString(), provider: 'ESPN public scoreboard', results })
  } catch {
    return response.status(502).json({ error: 'Live scores are temporarily unavailable' })
  }
}
