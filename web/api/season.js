import { readFile } from 'node:fs/promises'
import path from 'node:path'

function isValidForecast(game) {
  return game && Number.isFinite(Number(game.game_id)) && game.away_team && game.home_team && game.kickoff && Number.isFinite(Number(game.home_win_probability))
}

export default async function handler(request, response) {
  if (request.method !== 'GET') return response.status(405).json({ error: 'Method not allowed' })
  const requestedSeason = Number(request.query.season ?? 2026)
  if (!Number.isInteger(requestedSeason) || requestedSeason < 2000 || requestedSeason > 2100) {
    return response.status(400).json({ error: 'season must be a valid four-digit year' })
  }

  try {
    const file = path.join(process.cwd(), process.cwd().endsWith(`${path.sep}web`) ? 'public' : 'web/public', 'forecasts.json')
    const board = JSON.parse(await readFile(file, 'utf8'))
    if (Number(board.season) !== requestedSeason) {
      return response.status(404).json({ error: `No published forecast is available for ${requestedSeason}` })
    }
    const forecasts = Array.isArray(board.forecasts) ? board.forecasts.filter(isValidForecast) : []
    const kickoffTimes = forecasts.map(game => new Date(game.kickoff).getTime()).filter(Number.isFinite)
    response.setHeader('Cache-Control', 's-maxage=60, stale-while-revalidate=300')
    return response.status(200).json({
      season: requestedSeason,
      run_id: board.run_id ?? null,
      as_of: board.as_of ?? null,
      logged_at: board.logged_at ?? null,
      week: board.week ?? null,
      model: board.model ?? null,
      games: forecasts.length,
      first_kickoff: kickoffTimes.length ? new Date(Math.min(...kickoffTimes)).toISOString() : null,
      last_kickoff: kickoffTimes.length ? new Date(Math.max(...kickoffTimes)).toISOString() : null,
      sources: Array.isArray(board.sources) ? board.sources : [],
      forecasts,
    })
  } catch {
    return response.status(503).json({ error: 'Published season data is temporarily unavailable' })
  }
}
