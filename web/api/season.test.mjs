import test from 'node:test'
import assert from 'node:assert/strict'
import { mkdtemp, mkdir, writeFile, rm } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import handler from './season.js'

function response() {
  return { code: null, body: null, status(code) { this.code = code; return this },
    json(body) { this.body = body; return this }, setHeader() {} }
}

test('season endpoint serves saved provenance, prices and settled metrics', async () => {
  const directory = await mkdtemp(path.join(os.tmpdir(), 'cfb-season-'))
  const previous = process.cwd()
  try {
    await mkdir(path.join(directory, 'web/public'), { recursive: true })
    await writeFile(path.join(directory, 'web/public/forecasts.json'), JSON.stringify({
      season: 2026, as_of: '2026-10-11T15:00Z', validation: { settled_games: 4 }, statistics: [],
      forecasts: [{ game_id: 1, home_team: 'Home', away_team: 'Away', kickoff: '2026-10-17T19:00Z',
        home_win_probability: .6, expected_home_score: 28, expected_away_score: 24, home_book_moneyline: -125 },
        { game_id: 2, home_team: 'Home', away_team: 'Away', kickoff: 'bad', home_win_probability: null }],
    }))
    process.chdir(directory)
    const result = response()
    await handler({ method: 'GET', query: { season: '2026' } }, result)
    assert.equal(result.code, 200)
    assert.equal(result.body.forecasts.length, 1)
    assert.equal(result.body.forecasts[0].home_book_moneyline, -125)
    assert.equal(result.body.validation.settled_games, 4)
    const unavailable = response()
    await handler({ method: 'GET', query: { season: '2025' } }, unavailable)
    assert.equal(unavailable.code, 404)
  } finally {
    process.chdir(previous)
    await rm(directory, { recursive: true, force: true })
  }
})

test('season endpoint rejects writes and invalid years', async () => {
  const write = response()
  await handler({ method: 'POST', query: {} }, write)
  assert.equal(write.code, 405)
  const invalid = response()
  await handler({ method: 'GET', query: { season: 'invalid' } }, invalid)
  assert.equal(invalid.code, 400)
})
