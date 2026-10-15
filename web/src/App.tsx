import { useEffect, useMemo, useState } from 'react'

type Forecast = {
  game_id: number
  season?: number
  away_team: string
  home_team: string
  game_date?: string
  kickoff: string
  home_win_probability: number
  expected_home_score: number
  expected_away_score: number
  projected_home_margin: number
  model_spread?: number | null
  market_spread?: number | null
  home_cover_probability?: number | null
  home_moneyline?: number | null
  away_moneyline?: number | null
  book_home_moneyline?: number | null
  book_away_moneyline?: number | null
  book_total?: number | null
  spread_open?: number | null
  total_open?: number | null
  formatted_spread?: string | null
  line_provider?: string | null
  cfbd_pregame_home_win_probability?: number | null
  venue?: string | null
  weather_condition?: string | null
  temperature_f?: number | null
  winner: string
  confidence: number
}
type Source = { name: string; url: string }
type Board = { forecasts?: Forecast[]; sources?: Source[]; games?: number; as_of?: string; model?: string; week?: number | null; run_id?: string }
type Validation = { status: string; games?: number; log_loss?: number; brier?: number; margin_mae?: number }
type Page = 'board' | 'lines' | 'sources' | 'method'
type SortKey = 'kickoff' | 'confidence' | 'margin' | 'total'

const number = new Intl.NumberFormat('en-US', { maximumFractionDigits: 1, minimumFractionDigits: 1 })

function pct(value: number) {
  return `${Math.round(value * 100)}%`
}
function signed(value: number) {
  return `${value > 0 ? '+' : ''}${number.format(value)}`
}
function moneyline(value?: number | null) {
  if (value == null || Number.isNaN(Number(value))) return '—'
  return value > 0 ? `+${value}` : String(value)
}
function dayLabel(iso?: string) {
  if (!iso || !/^\d{4}-\d{2}-\d{2}$/.test(iso)) return 'Date unavailable'
  const [year, month, day] = iso.split('-').map(Number)
  return new Intl.DateTimeFormat('en-US', { weekday: 'long', month: 'long', day: 'numeric', year: 'numeric', timeZone: 'UTC' }).format(new Date(Date.UTC(year, month - 1, day)))
}
function kickoffTime(value?: string) {
  if (!value) return 'Time unavailable'
  const parsed = new Date(value)
  if (Number.isNaN(parsed.getTime())) return 'Time unavailable'
  return new Intl.DateTimeFormat('en-US', { hour: 'numeric', minute: '2-digit', timeZone: 'America/Chicago', timeZoneName: 'short' }).format(parsed)
}
function confidenceOf(game: Forecast) {
  return game.confidence ?? Math.max(game.home_win_probability, 1 - game.home_win_probability)
}
function totalOf(game: Forecast) {
  return game.expected_away_score + game.expected_home_score
}
function spreadOf(game: Forecast) {
  return game.model_spread ?? -game.projected_home_margin
}
function spreadText(game: Forecast) {
  const spread = spreadOf(game)
  if (spread == null || Number.isNaN(spread)) return 'Spread unavailable'
  const favored = spread <= 0 ? game.home_team : game.away_team
  return `${favored} -${Math.abs(spread).toFixed(1)}`
}
function bookGap(game: Forecast) {
  if (game.market_spread == null || Number.isNaN(Number(game.market_spread))) return null
  return game.market_spread - spreadOf(game)
}
function slateDate(runId: string) {
  const match = runId.match(/(\d{4}-\d{2}-\d{2})$/)
  return match ? dayLabel(match[1]) : ''
}

export function App() {
  const [page, setPage] = useState<Page>('board')
  const [forecasts, setForecasts] = useState<Forecast[]>([])
  const [sources, setSources] = useState<Source[]>([])
  const [model, setModel] = useState('Score forecast')
  const [runId, setRunId] = useState('')
  const [validation, setValidation] = useState<Validation>({ status: 'awaiting_results' })
  const [live, setLive] = useState(false)
  const [updated, setUpdated] = useState<Date | null>(null)
  const [query, setQuery] = useState('')
  const [day, setDay] = useState('all')
  const [sort, setSort] = useState<SortKey>('kickoff')
  const [gapsOnly, setGapsOnly] = useState(false)
  const [selectedId, setSelectedId] = useState<number | null>(null)

  async function refresh() {
    setLive(true)
    try {
      const published = await fetch('/forecasts.json', { cache: 'no-store' })
      if (published.ok) {
        const board = await published.json() as Board
        if (Array.isArray(board.forecasts)) setForecasts(board.forecasts)
        if (Array.isArray(board.sources)) setSources(board.sources)
        if (board.model) setModel(board.model)
        if (board.run_id) setRunId(board.run_id)
        if (board.as_of) setUpdated(new Date(board.as_of))
      }
      const response = await fetch('/api/forecasts?season=2026')
      if (response.ok) {
        const data = await response.json()
        if (Array.isArray(data.forecasts) && data.forecasts.length && data.forecasts[0].winner) setForecasts(data.forecasts)
        if (data.validation) setValidation(data.validation)
      }
    } finally {
      setLive(false)
    }
  }
  useEffect(() => { refresh() }, [])

  const dates = useMemo(() => [...new Set(forecasts.map(game => game.game_date).filter((value): value is string => Boolean(value)))].sort(), [forecasts])
  const visible = useMemo(() => {
    const needle = query.trim().toLowerCase()
    const rows = forecasts.filter(game => {
      const named = !needle || `${game.away_team} ${game.home_team}`.toLowerCase().includes(needle)
      const dated = day === 'all' || game.game_date === day
      const gap = bookGap(game)
      const separated = !gapsOnly || (gap != null && Math.abs(gap) >= 3)
      return named && dated && separated
    })
    return rows.sort((a, b) => {
      if (sort === 'confidence') return confidenceOf(b) - confidenceOf(a)
      if (sort === 'margin') return Math.abs(b.projected_home_margin) - Math.abs(a.projected_home_margin)
      if (sort === 'total') return totalOf(b) - totalOf(a)
      return `${a.game_date ?? ''} ${a.kickoff}`.localeCompare(`${b.game_date ?? ''} ${b.kickoff}`)
    })
  }, [forecasts, query, day, gapsOnly, sort])
  const grouped = useMemo(() => {
    const groups = new Map<string, Forecast[]>()
    for (const game of visible) {
      const key = game.game_date || 'Date unavailable'
      groups.set(key, [...(groups.get(key) ?? []), game])
    }
    return [...groups.entries()]
  }, [visible])
  const selected = forecasts.find(game => game.game_id === selectedId) ?? null
  const average = forecasts.length ? forecasts.reduce((sum, game) => sum + confidenceOf(game), 0) / forecasts.length : 0
  const recordedBooks = forecasts.filter(game => game.market_spread != null).length
  const strongest = [...forecasts].sort((a, b) => confidenceOf(b) - confidenceOf(a)).slice(0, 3)
  const closest = [...forecasts].sort((a, b) => Math.abs(a.projected_home_margin) - Math.abs(b.projected_home_margin)).slice(0, 3)

  return <div className="app">
    <aside className="sidebar">
      <div className="brand"><strong>Forecast</strong><span>College football, 2026</span></div>
      <nav>
        <button className={page === 'board' ? 'active' : ''} onClick={() => setPage('board')}>Board</button>
        <button className={page === 'lines' ? 'active' : ''} onClick={() => setPage('lines')}>Lines</button>
        <button className={page === 'sources' ? 'active' : ''} onClick={() => setPage('sources')}>Sources</button>
        <button className={page === 'method' ? 'active' : ''} onClick={() => setPage('method')}>Method</button>
      </nav>
      <div className="sidebar-foot">
        <span className="eyebrow">{forecasts.length ? 'Published' : 'Waiting'}</span>
        <span>{slateDate(runId) || 'No slate date yet'}</span>
        <span>{updated ? updated.toLocaleString() : 'Not published yet'}</span>
      </div>
    </aside>
    <main className="main">
      <header className="topbar">
        <div>
          <div className="eyebrow">{runId || 'Weekly slate'}</div>
          <h1>{page === 'lines' ? 'Spreads and prices' : page === 'sources' ? 'Source catalog' : page === 'method' ? 'How a forecast is made' : 'This week’s games'}</h1>
          <p className="lede">{page === 'board' ? 'Each row is one game. The date is the Chicago calendar day of kickoff. The score is the model mean, and the pick is the side at or above 50 percent.' : page === 'lines' ? 'The model spread and the fair moneyline come from the same score forecast. A recorded book number is shown beside them.' : page === 'sources' ? 'These are the public catalogs the forecast run records. A catalog entry is not a claim that every field was present for every game.' : 'The forecast is a weekly point-in-time estimate. It is settled against completed games before the next week.'}</p>
        </div>
        <button className="refresh" onClick={refresh} disabled={live}>{live ? 'Refreshing' : 'Refresh'}</button>
      </header>
      <section className="metrics">
        <Metric label="Games" value={forecasts.length ? String(forecasts.length) : '—'} note={dates.length ? `${dates.length} kickoff dates` : 'No board yet'} />
        <Metric label="Average confidence" value={forecasts.length ? pct(average) : '—'} note="Larger of the two win probabilities" />
        <Metric label="Recorded books" value={forecasts.length ? String(recordedBooks) : '—'} note="Games with a market spread" />
        <Metric label="Prior week" value={validation.games ? `${validation.games}` : '—'} note={validation.status === 'settled' ? `Log loss ${validation.log_loss?.toFixed(3)}` : 'Awaiting completed games'} />
      </section>

      {page === 'board' && <>
        <div className="toolbar">
          <label className="tool">Search<input value={query} onChange={event => setQuery(event.target.value)} placeholder="Team" aria-label="Search teams" /></label>
          <label className="tool">Date<select value={day} onChange={event => setDay(event.target.value)} aria-label="Filter by date">{[['all', 'All dates'], ...dates.map(value => [value, dayLabel(value)])].map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select></label>
          <label className="tool">Sort<select value={sort} onChange={event => setSort(event.target.value as SortKey)} aria-label="Sort games"><option value="kickoff">Kickoff</option><option value="confidence">Confidence</option><option value="margin">Margin</option><option value="total">Total</option></select></label>
          <button className={gapsOnly ? 'chip on' : 'chip'} onClick={() => setGapsOnly(value => !value)} aria-pressed={gapsOnly}>Model and book differ by 3+</button>
        </div>
        <div className="layout">
          <section className="panel">
            {grouped.length ? grouped.map(([date, games]) => <div className="day" key={date}>
              <div className="day-head"><h2>{date === 'Date unavailable' ? date : dayLabel(date)}</h2><span className="quiet">{games.length} {games.length === 1 ? 'game' : 'games'}</span></div>
              {games.map(game => <GameRow key={game.game_id} game={game} selected={game.game_id === selectedId} onSelect={() => setSelectedId(game.game_id === selectedId ? null : game.game_id)} />)}
            </div>) : <p className="note">No games match this search.</p>}
          </section>
          <aside>{selected ? <GameDetail game={selected} /> : <SlateNotes strongest={strongest} closest={closest} onSelect={setSelectedId} />}</aside>
        </div>
      </>}

      {page === 'lines' && <section className="panel table-wrap">
        <table>
          <thead><tr><th>Date</th><th className="wrap">Game</th><th className="num">Score</th><th className="num">Total</th><th>Model spread</th><th className="num">Book</th><th className="num">Open</th><th className="num">Book total</th><th>Provider</th><th className="num">Gap</th><th className="num">Home cover</th><th className="num">Fair ML</th><th className="num">Book ML</th><th>Venue</th></tr></thead>
          <tbody>{visible.map(game => {
            const gap = bookGap(game)
            return <tr key={game.game_id}><td>{game.game_date || '—'}<div className="quiet">{kickoffTime(game.kickoff)}</div></td><td className="wrap"><button onClick={() => { setSelectedId(game.game_id); setPage('board') }}>{game.away_team} at {game.home_team}</button></td><td className="num">{number.format(game.expected_away_score)}–{number.format(game.expected_home_score)}</td><td className="num">{number.format(totalOf(game))}</td><td>{spreadText(game)}</td><td className="num">{game.market_spread == null ? '—' : signed(game.market_spread)}</td><td className="num">{game.spread_open == null ? '—' : signed(game.spread_open)}</td><td className="num">{game.book_total == null ? '—' : number.format(game.book_total)}</td><td>{game.line_provider || '—'}</td><td className="num">{gap == null ? '—' : signed(gap)}</td><td className="num">{game.home_cover_probability == null ? '—' : pct(game.home_cover_probability)}</td><td className="num">{moneyline(game.away_moneyline)} / {moneyline(game.home_moneyline)}</td><td className="num">{moneyline(game.book_away_moneyline)} / {moneyline(game.book_home_moneyline)}</td><td>{game.venue || '—'}</td></tr>
          })}</tbody>
        </table>
        <p className="note">Gap is the book’s home spread minus the model’s home spread. A positive gap means the book lists the home team as a larger underdog. Fair moneylines are converted from the win probability. They are not posted prices.</p>
      </section>}

      {page === 'sources' && <section className="panel">
        {(sources.length ? sources : []).map(source => <div className="source" key={source.name}><span>{source.name}</span><a href={source.url} target="_blank" rel="noreferrer">{source.url.replace(/^https?:\/\//, '')}</a></div>)}
        {!sources.length && <p className="note">The source list appears when a forecast board is published.</p>}
      </section>}

      {page === 'method' && <section className="panel method">
        <div><h2>Score</h2><p className="note">A hierarchical team state sets the score location from games already completed. Same-season and prior-season points and play rates enter before kickoff. Weather, rest, injuries, and a recorded line enter only when they are already known.</p></div>
        <div><h2>Winner and confidence</h2><p className="note">The winner is the side whose win probability is at least 50 percent. Confidence is the larger of the home win probability and its complement. Expected points are the model means, and the margin is the home mean minus the away mean.</p></div>
        <div><h2>Date</h2><p className="note">Every forecast keeps the kickoff timestamp and the Chicago calendar date of that kickoff. A game after midnight UTC can still belong to the previous evening in Chicago.</p></div>
        <div><h2>What this is not</h2><p className="note">This is not a live player-tracking feed and not a betting instruction. {model}</p></div>
      </section>}
      <footer><span>Model outputs, not guarantees.</span><span>{validation.status === 'settled' ? `Last settled week: log loss ${validation.log_loss?.toFixed(3)}, Brier ${validation.brier?.toFixed(3)}, margin MAE ${validation.margin_mae?.toFixed(1)}` : 'The previous week is scored after games become final.'}</span></footer>
    </main>
  </div>
}

function Metric({ label, value, note }: { label: string; value: string; note: string }) {
  return <article className="metric"><span>{label}</span><strong>{value}</strong><small>{note}</small></article>
}

function GameRow({ game, selected, onSelect }: { game: Forecast; selected: boolean; onSelect: () => void }) {
  return <button className={selected ? 'game selected' : 'game'} onClick={onSelect} aria-pressed={selected}>
    <span className="time">{kickoffTime(game.kickoff)}<span>{game.game_date || 'No date'}</span></span>
    <span className="matchup"><b className={game.winner === game.away_team ? 'favored' : ''}>{game.away_team}</b><b className={game.winner === game.home_team ? 'favored' : ''}>{game.home_team}</b></span>
    <span className="score"><strong>{number.format(game.expected_away_score)} – {number.format(game.expected_home_score)}</strong><span>Total {number.format(totalOf(game))}</span></span>
    <span className="pick"><strong>{game.winner}</strong><span>{pct(confidenceOf(game))} · {spreadText(game)}</span><span className="bar"><i style={{ width: `${confidenceOf(game) * 100}%` }} /></span></span>
  </button>
}

function GameDetail({ game }: { game: Forecast }) {
  const gap = bookGap(game)
  return <article className="detail panel">
    <div><span className="eyebrow">{dayLabel(game.game_date)} · {kickoffTime(game.kickoff)}</span><h2>{game.away_team} at {game.home_team}</h2></div>
    <div className="scoreline"><div><span>Away</span><strong>{number.format(game.expected_away_score)}</strong></div><div><span>Home</span><strong>{number.format(game.expected_home_score)}</strong></div></div>
    <div className="prob"><div><span>{game.away_team}</span><strong>{pct(1 - game.home_win_probability)}</strong></div><div><span>{game.home_team}</span><strong>{pct(game.home_win_probability)}</strong></div></div>
    <div className="detail-grid">
      <div><span>Pick</span><strong>{game.winner}</strong></div>
      <div><span>Confidence</span><strong>{pct(confidenceOf(game))}</strong></div>
      <div><span>Home margin</span><strong>{signed(game.projected_home_margin)}</strong></div>
      <div><span>Total</span><strong>{number.format(totalOf(game))}</strong></div>
      <div><span>Model spread</span><strong>{spreadText(game)}</strong></div>
      <div><span>Book spread</span><strong>{game.market_spread == null ? 'Not recorded' : signed(game.market_spread)}</strong></div>
      <div><span>Book minus model</span><strong>{gap == null ? '—' : signed(gap)}</strong></div>
      <div><span>Home cover probability</span><strong>{game.home_cover_probability == null ? '—' : pct(game.home_cover_probability)}</strong></div>
      <div><span>Away fair moneyline</span><strong>{moneyline(game.away_moneyline)}</strong></div>
      <div><span>Home fair moneyline</span><strong>{moneyline(game.home_moneyline)}</strong></div>
    </div>
    <p className="note">Game {game.game_id}{game.season ? ` · season ${game.season}` : ''}. The home spread is positive when the home team is the underdog.</p>
  </article>
}

function SlateNotes({ strongest, closest, onSelect }: { strongest: Forecast[]; closest: Forecast[]; onSelect: (id: number) => void }) {
  return <article className="detail panel">
    <div><span className="eyebrow">Slate notes</span><h2>Select a game</h2><p className="note">Open any row for the score, both win probabilities, the spread, and the fair moneyline.</p></div>
    <div className="mini"><span className="eyebrow">Strongest picks</span>{strongest.map(game => <button key={game.game_id} onClick={() => onSelect(game.game_id)}><span>{game.away_team} at {game.home_team}</span><strong>{pct(confidenceOf(game))}</strong></button>)}</div>
    <div className="mini"><span className="eyebrow">Closest games</span>{closest.map(game => <button key={game.game_id} onClick={() => onSelect(game.game_id)}><span>{game.game_date} · {game.away_team} at {game.home_team}</span><strong>{signed(game.projected_home_margin)}</strong></button>)}</div>
  </article>
}