import { useEffect, useMemo, useState } from 'react'
import { RefreshCw, Search } from 'lucide-react'

type Forecast = {
  game_id: number
  away_team: string
  home_team: string
  kickoff: string
  home_win_probability: number
  expected_home_score: number
  expected_away_score: number
  projected_home_margin: number
  model_spread?: number | null
  market_spread?: number | null
  home_moneyline?: number | null
  away_moneyline?: number | null
  winner: string
  confidence: number
}
type Source = { name: string; url: string }
type Board = { forecasts?: Forecast[]; sources?: Source[]; games?: number; as_of?: string; model?: string; week?: number | null; run_id?: string }
type Validation = { status: string; games?: number; log_loss?: number; brier?: number; margin_mae?: number }
type Page = 'predictions' | 'lines' | 'statistics' | 'methodology'
type Range = 'week' | 'all'

const PAGES: { id: Page; label: string }[] = [
  { id: 'predictions', label: 'Predictions' },
  { id: 'lines', label: 'Odds' },
  { id: 'statistics', label: 'Data' },
  { id: 'methodology', label: 'How it works' },
]
const DAY = 24 * 60 * 60 * 1000

function pct(value: number) { return `${Math.round(value * 100)}%` }
function homeChance(game: Forecast) { return Math.min(Math.max(Number(game.home_win_probability), 0), 1) }
function margin(game: Forecast) {
  const value = game.projected_home_margin ?? game.expected_home_score - game.expected_away_score
  return Number(value)
}
// Fair American odds from a win chance, the same conversion the forecast uses.
function fairMoneyline(probability: number) {
  if (!(probability > 0 && probability < 1)) return null
  return probability >= 0.5 ? -Math.round((100 * probability) / (1 - probability)) : Math.round((100 * (1 - probability)) / probability)
}
function oddsLabel(value?: number | null) {
  if (value == null || Number.isNaN(Number(value))) return '—'
  return value > 0 ? `+${value}` : String(value)
}
function spreadFor(game: Forecast, side: 'home' | 'away') {
  const home = game.model_spread ?? -margin(game)
  const value = side === 'home' ? home : -home
  if (value == null || Number.isNaN(value)) return '—'
  if (Math.abs(value) < 0.05) return 'Pick’em'
  return value > 0 ? `+${value.toFixed(1)}` : value.toFixed(1)
}
function bookSpreadFor(game: Forecast, side: 'home' | 'away') {
  if (game.market_spread == null || Number.isNaN(Number(game.market_spread))) return null
  const value = side === 'home' ? Number(game.market_spread) : -Number(game.market_spread)
  if (value === 0) return 'Pick’em'
  return value > 0 ? `+${value}` : String(value)
}
function pickLabel(game: Forecast) {
  const lead = Math.abs(margin(game))
  if (!(lead >= 0.05)) return 'Even'
  return `${homeChance(game) >= 0.5 ? game.home_team : game.away_team} by ${lead.toFixed(1)}`
}
function dayKey(value: string) {
  return new Intl.DateTimeFormat('en-US', { weekday: 'long', month: 'long', day: 'numeric' }).format(new Date(value))
}
function timeLabel(value: string) {
  return new Intl.DateTimeFormat('en-US', { hour: 'numeric', minute: '2-digit', timeZoneName: 'short' }).format(new Date(value))
}
function shortDate(value: Date) {
  return new Intl.DateTimeFormat('en-US', { weekday: 'short', month: 'short', day: 'numeric' }).format(value)
}

export function App() {
  const [page, setPage] = useState<Page>('predictions')
  const [forecasts, setForecasts] = useState<Forecast[]>([])
  const [sources, setSources] = useState<Source[]>([])
  const [week, setWeek] = useState<number | null>(null)
  const [validation, setValidation] = useState<Validation>({ status: 'awaiting_results' })
  const [loading, setLoading] = useState(false)
  const [updated, setUpdated] = useState<Date | null>(null)
  const [query, setQuery] = useState('')
  const [range, setRange] = useState<Range>('week')

  async function refresh() {
    setLoading(true)
    try {
      const published = await fetch('/forecasts.json', { cache: 'no-store' })
      if (published.ok) {
        const board = await published.json() as Board
        if (Array.isArray(board.forecasts)) setForecasts(board.forecasts)
        if (Array.isArray(board.sources)) setSources(board.sources)
        if (typeof board.week === 'number') setWeek(board.week)
        if (board.as_of) setUpdated(new Date(board.as_of))
      }
      const response = await fetch('/api/forecasts?season=2026')
      if (response.ok && response.headers.get('content-type')?.includes('json')) {
        const data = await response.json()
        if (Array.isArray(data.forecasts) && data.forecasts.length && data.forecasts[0].winner) setForecasts(data.forecasts)
        if (data.validation) setValidation(data.validation)
      }
    } catch {
      // The published board is enough; the API only exists on the self-hosted desk.
    } finally {
      setLoading(false)
    }
  }
  useEffect(() => { refresh() }, [])

  const sorted = useMemo(() => [...forecasts].sort((a, b) => new Date(a.kickoff).getTime() - new Date(b.kickoff).getTime()), [forecasts])
  // The board's own week: games from the forecast date through the following eight days.
  const weekEnd = useMemo(() => {
    const start = updated ?? (sorted.length ? new Date(sorted[0].kickoff) : null)
    return start ? start.getTime() + 8 * DAY : null
  }, [updated, sorted])
  const weekGames = useMemo(() => weekEnd == null ? sorted : sorted.filter(game => new Date(game.kickoff).getTime() < weekEnd), [sorted, weekEnd])
  const shown = useMemo(() => {
    const text = query.trim().toLowerCase()
    const pool = range === 'week' ? weekGames : sorted
    return text ? pool.filter(game => `${game.home_team} ${game.away_team}`.toLowerCase().includes(text)) : pool
  }, [query, range, weekGames, sorted])
  const days = useMemo(() => {
    const groups: { day: string; games: Forecast[] }[] = []
    for (const game of shown) {
      const day = dayKey(game.kickoff)
      if (!groups.length || groups[groups.length - 1].day !== day) groups.push({ day, games: [] })
      groups[groups.length - 1].games.push(game)
    }
    return groups
  }, [shown])
  const avgConfidence = weekGames.length ? weekGames.reduce((sum, game) => sum + Math.max(homeChance(game), 1 - homeChance(game)), 0) / weekGames.length : 0
  const closeGames = weekGames.filter(game => Math.max(homeChance(game), 1 - homeChance(game)) < 0.6).length
  const weekName = week ? `Week ${week}` : 'This week'

  return <div className="page">
    <header className="site-header">
      <div className="header-inner">
        <div className="brand"><span className="brand-mark">CF</span><span>College Football Forecasts</span></div>
        <nav aria-label="Pages">
          {PAGES.map(item => <button key={item.id} className={page === item.id ? 'active' : ''} aria-current={page === item.id ? 'page' : undefined} onClick={() => setPage(item.id)}>{item.label}</button>)}
        </nav>
      </div>
    </header>

    <main className="content">
      <section className="intro">
        <div>
          <h1>{page === 'predictions' ? `${weekName} predictions` : PAGES.find(item => item.id === page)!.label}</h1>
          <p className="updated">{updated ? `Forecast made ${shortDate(updated)}` : 'No forecast published yet'}{sorted.length ? ` · ${sorted.length} games on the board` : ''}</p>
        </div>
        <button className="reload" onClick={refresh} disabled={loading}><RefreshCw className={loading ? 'spin' : ''} aria-hidden /> {loading ? 'Loading' : 'Reload'}</button>
      </section>

      {(page === 'predictions' || page === 'lines') && <>
        <section className="summary" aria-label="Summary">
          <Summary label={`${weekName} games`} value={weekGames.length ? String(weekGames.length) : '—'} />
          <Summary label="Average chance for the pick" value={weekGames.length ? pct(avgConfidence) : '—'} />
          <Summary label="Close calls (under 60%)" value={weekGames.length ? String(closeGames) : '—'} />
          <Summary label="Last week checked" value={validation.status === 'settled' && validation.games ? `${validation.games} games` : 'Not yet'} />
        </section>

        <section className="controls">
          <label className="search"><Search aria-hidden /><input type="search" placeholder="Find a team" value={query} onChange={event => setQuery(event.target.value)} aria-label="Find a team" /></label>
          <div className="toggle" role="group" aria-label="Which games">
            <button className={range === 'week' ? 'active' : ''} onClick={() => setRange('week')}>{weekName}</button>
            <button className={range === 'all' ? 'active' : ''} onClick={() => setRange('all')}>All games</button>
          </div>
        </section>

        {page === 'lines' && <p className="lede">The spread and fair odds are our forecast written the way a sportsbook would write them. They are not sportsbook prices. A Book column appears when a sportsbook line was recorded before the forecast.</p>}

        {!sorted.length && <p className="empty">No forecast has been published yet. The board fills when the weekly run writes forecasts.json.</p>}
        {sorted.length > 0 && !shown.length && <p className="empty">No games match “{query}”.</p>}
        {days.map(group => <section className="day" key={group.day}>
          <h2>{group.day}</h2>
          <div className="cards">{group.games.map(game => page === 'predictions' ? <PredictionCard key={game.game_id} game={game} /> : <LineCard key={game.game_id} game={game} />)}</div>
        </section>)}
      </>}

      {page === 'statistics' && <section className="prose">
        <p>Before each game, the forecast reads what both teams have recorded in games already played. The previous season is the starting point, and every game this season updates it.</p>
        <ul>
          <li>Points scored and allowed</li>
          <li>Yards per play, success rate and explosive plays</li>
          <li>Expected points added and turnovers</li>
          <li>Passing, rushing and receiving totals</li>
          <li>Days of rest since each team's last game</li>
          <li>The betting line and total, when one was recorded before the forecast</li>
        </ul>
        <h2>Sources</h2>
        <ul className="sources">{(sources.length ? sources : [{ name: 'sportsdataverse', url: 'https://github.com/sportsdataverse/sportsdataverse-data' }]).map(source => <li key={source.name}><a href={source.url} target="_blank" rel="noreferrer">{source.name}</a></li>)}</ul>
      </section>}

      {page === 'methodology' && <section className="prose">
        <p>Every Sunday the forecast is made for the coming week using only results that were final at that time.</p>
        <ol>
          <li>Each team has an offense and a defense rating, updated after every game it plays.</li>
          <li>Those ratings, plus each team's recorded rates and rest, give an expected score for both teams.</li>
          <li>When a betting line was recorded before the forecast, it adjusts the expected score. Games without one use a version of the forecast that never relies on a line.</li>
          <li>The difference between the two expected scores becomes a win chance. The pick is the team with the higher chance.</li>
          <li>The spread and fair odds on the Odds page are the same forecast written the way a sportsbook would write it.</li>
          <li>After the games finish, the forecast is checked against the real results before the next Sunday run.</li>
        </ol>
        <p>Weather and injury reports are recorded with each forecast. They only start to adjust the win chance once enough weeks have been recorded to show that they help.</p>
        <p className="note">These are forecasts, not guarantees, and not betting advice.</p>
      </section>}
    </main>

    <footer className="site-footer">Forecasts, not guarantees. {updated ? `Published ${updated.toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' })}.` : ''}</footer>
  </div>
}

function Summary({ label, value }: { label: string; value: string }) {
  return <div className="summary-item"><span>{label}</span><strong>{value}</strong></div>
}

function PredictionCard({ game }: { game: Forecast }) {
  const home = homeChance(game)
  const pickHome = home >= 0.5
  const rows = [
    { side: 'away' as const, team: game.away_team, score: game.expected_away_score, chance: 1 - home, pick: !pickHome },
    { side: 'home' as const, team: game.home_team, score: game.expected_home_score, chance: home, pick: pickHome },
  ]
  return <article className="card">
    <header><span>{timeLabel(game.kickoff)}</span><span className="pick-note">{pickLabel(game)}</span></header>
    <div className="table" role="table" aria-label={`${game.away_team} at ${game.home_team}`}>
      <div className="row head" role="row"><span role="columnheader">Team</span><span role="columnheader">Points</span><span role="columnheader">Win chance</span></div>
      {rows.map(row => <div key={row.side} className={`row${row.pick ? ' picked' : ''}`} role="row">
        <span className="team" role="cell">{row.side === 'home' && <em>at</em>}{row.team}</span>
        <span className="num" role="cell">{Math.round(row.score)}</span>
        <span className="chance" role="cell"><b>{pct(row.chance)}</b><i aria-hidden><i style={{ width: `${row.chance * 100}%` }} /></i></span>
      </div>)}
    </div>
  </article>
}

function LineCard({ game }: { game: Forecast }) {
  const home = homeChance(game)
  const odds = {
    home: game.home_moneyline ?? fairMoneyline(home),
    away: game.away_moneyline ?? fairMoneyline(1 - home),
  }
  const hasBook = game.market_spread != null && !Number.isNaN(Number(game.market_spread))
  return <article className="card">
    <header><span>{timeLabel(game.kickoff)}</span><span className="pick-note">{pickLabel(game)}</span></header>
    <div className={`table${hasBook ? ' four' : ''}`} role="table" aria-label={`${game.away_team} at ${game.home_team}`}>
      <div className="row head" role="row"><span role="columnheader">Team</span><span role="columnheader">Spread</span><span role="columnheader">Fair odds</span>{hasBook && <span role="columnheader">Book</span>}</div>
      {(['away', 'home'] as const).map(side => <div key={side} className={`row${(side === 'home') === (home >= 0.5) ? ' picked' : ''}`} role="row">
        <span className="team" role="cell">{side === 'home' && <em>at</em>}{side === 'home' ? game.home_team : game.away_team}</span>
        <span className="num" role="cell">{spreadFor(game, side)}</span>
        <span className="num" role="cell">{oddsLabel(odds[side])}</span>
        {hasBook && <span className="num" role="cell">{bookSpreadFor(game, side)}</span>}
      </div>)}
    </div>
  </article>
}
