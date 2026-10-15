import { readFile } from 'node:fs/promises'
import path from 'node:path'

export default async function handler(request, response) {
  if (request.method !== 'GET') return response.status(405).json({ error: 'Method not allowed' })
  try {
    const file = path.join(process.cwd(), process.cwd().endsWith(`${path.sep}web`) ? 'public' : 'web/public', 'forecasts.json')
    const board = JSON.parse(await readFile(file, 'utf8'))
    response.setHeader('Cache-Control', 's-maxage=30, stale-while-revalidate=120')
    return response.status(200).json({
      run_id: board.run_id ?? null,
      as_of: board.as_of ?? null,
      logged_at: board.logged_at ?? null,
      status: board.pipeline_status ?? 'published',
      stages: Array.isArray(board.pipeline_stages) ? board.pipeline_stages : [],
    })
  } catch {
    return response.status(200).json({ run_id: null, as_of: null, logged_at: null, status: 'unavailable', stages: [] })
  }
}
