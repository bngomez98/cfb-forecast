import { readFile } from 'node:fs/promises'
import path from 'node:path'

export default async function handler(request, response) {
  if (request.method !== 'GET') return response.status(405).json({ error: 'Method not allowed' })
  try {
    const file = path.join(process.cwd(), process.cwd().endsWith(`${path.sep}web`) ? 'public' : 'web/public', 'forecasts.json')
    const board = JSON.parse(await readFile(file, 'utf8'))
    const versions = Array.isArray(board.learning_versions) ? board.learning_versions : []
    response.setHeader('Cache-Control', 's-maxage=60, stale-while-revalidate=300')
    return response.status(200).json({ versions })
  } catch {
    return response.status(200).json({ versions: [] })
  }
}
