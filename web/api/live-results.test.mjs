import test from 'node:test'
import assert from 'node:assert/strict'
import handler from './live-results.js'

function response() {
  return { code: null, body: null, status(code) { this.code = code; return this },
    json(body) { this.body = body; return this }, setHeader() {} }
}

test('invalid dates and reversed sub-day windows fail before fetching', async () => {
  for (const query of [{ start: 'not a date' }, { start: '2026-10-16T00:00Z', end: '2026-10-15T23:59Z' }]) {
    const result = response()
    await handler({ method: 'GET', query }, result)
    assert.equal(result.code, 400)
  }
})

test('score feed preserves season and missing scores and removes duplicates', async () => {
  const original = global.fetch
  const calls = []
  global.fetch = async url => {
    calls.push(url)
    const event = { id: '7', season: { year: 2026 }, competitions: [{ competitors: [
      { homeAway: 'home', score: '' }, { homeAway: 'away', score: '14' },
    ], status: { type: { completed: false, state: 'in' } } }] }
    return { ok: true, json: async () => ({ events: [event, { ...event, id: '8', season: { year: 2025 } }] }) }
  }
  try {
    const result = response()
    await handler({ method: 'GET', query: { start: '2026-10-15T23:00Z', end: '2026-10-16T01:00Z', season: '2026' } }, result)
    assert.equal(calls.length, 2)
    assert.equal(result.code, 200)
    assert.equal(result.body.results.length, 1)
    assert.equal(result.body.results[0].home_points, null)
    assert.equal(result.body.results[0].away_points, 14)
  } finally {
    global.fetch = original
  }
})
