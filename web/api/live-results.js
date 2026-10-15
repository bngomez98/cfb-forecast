const SCOREBOARD = 'https://site.api.espn.com/apis/site/v2/sports/football/college-football/scoreboard'

function dateKey(date) {
  return date.toISOString().slice(0, 10).replaceAll('-', '')
}

function parseDate(value, fallback) {
  if (!value) return fallback
  const parsed = new Date(value)
  return Number.isNaN(parsed.getTime()) ? fallback : parsed
}

function eventResult(event) {
  const competition = event.competitions?.[0]
  const competitors = competition?.competitors ?? []
  const home = competitors.find(team => team.homeAway === 'home')
  const away = competitors.find(team => team.homeAway === 'away')
  const type = competition?.status?.type ?? event.status?.type ?? {}
  if (!event.id || !home || !away) return null
  return {
    game_id: Number(event.id),
    status: type.completed ? 'final' : type.state === 'in' ? 'live' : 'scheduled',
    status_detail: type.shortDetail ?? type.detail ?? null,
    period: competition?.status?.period ?? null,
    clock: competition?.status?.displayClock ?? null,
    home_points: home.score === undefined ? null : Number(home.score),
    away_points: away.score === undefined ? null : Number(away.score),
  }
}

export default async function handler(request, response) {
  if (request.method !== 'GET') return response.status(405).json({ error: 'Method not allowed' })
  const now = new Date()
  const start = parseDate(request.query.start, new Date(now.getTime() - 24 * 60 * 60 * 1000))
  const end = parseDate(request.query.end, new Date(now.getTime() + 7 * 24 * 60 * 60 * 1000))
  const span = Math.ceil((end.getTime() - start.getTime()) / (24 * 60 * 60 * 1000))
  if (span < 0 || span > 9) return response.status(400).json({ error: 'Date window must be 0 to 9 days' })

  const dates = Array.from({ length: span + 1 }, (_, index) => dateKey(new Date(start.getTime() + index * 24 * 60 * 60 * 1000)))
  try {
    const payloads = await Promise.all(dates.map(async date => {
      const url = `${SCOREBOARD}?dates=${date}&groups=80&limit=500`
      const upstream = await fetch(url, { headers: { 'User-Agent': 'fieldquant-cfb/1.0' } })
      if (!upstream.ok) throw new Error(`Score provider returned ${upstream.status}`)
      return upstream.json()
    }))
    const results = payloads.flatMap(payload => payload.events ?? []).map(eventResult).filter(Boolean)
    response.setHeader('Cache-Control', 's-maxage=30, stale-while-revalidate=30')
    return response.status(200).json({ season: 2026, updated_at: new Date().toISOString(), provider: 'ESPN public scoreboard', results })
  } catch (error) {
    return response.status(502).json({ error: 'Live scores are temporarily unavailable' })
  }
}
