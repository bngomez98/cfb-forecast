import { useEffect, useMemo, useState } from 'react'
import { Activity, ArrowUpRight, BarChart3, CalendarDays, CheckCircle2, Database, Gauge, RefreshCw, ShieldCheck, TriangleAlert } from 'lucide-react'

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
type Validation = { status: string; games?: number; log_loss?: number; brier?: number; margin_mae?: number; }
type Page = 'predictions' | 'lines' | 'statistics' | 'methodology'

function pct(value: number) { return `${Math.round(value * 100)}%` }
function spreadLabel(game: Forecast) {
  const spread = game.model_spread ?? (game.projected_home_margin != null ? -game.projected_home_margin : null)
  if (spread == null || Number.isNaN(spread)) return 'Spread unavailable'
  const favored = spread <= 0 ? game.home_team : game.away_team
  return `${favored} -${Math.abs(spread).toFixed(1)}`
}
function moneylineLabel(value?: number | null) {
  if (value == null || Number.isNaN(Number(value))) return '—'
  return value > 0 ? `+${value}` : String(value)
}
function dateLabel(value: string) {
  return new Intl.DateTimeFormat('en-US', { weekday: 'short', month: 'short', day: 'numeric', hour: 'numeric' }).format(new Date(value))
}

export function App() {
  const [page, setPage] = useState<Page>('predictions')
  const [forecasts, setForecasts] = useState<Forecast[]>([])
  const [sources, setSources] = useState<Source[]>([])
  const [model, setModel] = useState('Repository score head')
  const [week, setWeek] = useState<number | null>(null)
  const [runId, setRunId] = useState('')
  const [validation, setValidation] = useState<Validation>({ status: 'awaiting_results' })
  const [live, setLive] = useState(false)
  const [updated, setUpdated] = useState<Date | null>(null)

  async function refresh() {
    setLive(true)
    try {
      const published = await fetch('/forecasts.json', { cache: 'no-store' })
      if (published.ok) {
        const board = await published.json() as Board
        if (Array.isArray(board.forecasts)) setForecasts(board.forecasts)
        if (Array.isArray(board.sources)) setSources(board.sources)
        if (board.model) setModel(board.model)
        if (typeof board.week === 'number') setWeek(board.week)
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

  const avgConfidence = useMemo(() => forecasts.length ? forecasts.reduce((sum, game) => sum + (game.confidence ?? Math.max(game.home_win_probability, 1 - game.home_win_probability)), 0) / forecasts.length : 0, [forecasts])
  const titles: Record<Page, { heading: string; note: string }> = {
    predictions: { heading: 'Weekly forecast board', note: 'Kickoff, teams, expected score, winner, and confidence.' },
    lines: { heading: 'Spreads and moneylines', note: 'Model spread and fair American moneyline from the same win probability.' },
    statistics: { heading: 'Team and player statistics', note: 'Recorded play and scoring rates used before kickoff. They live in the forecast database.' },
    methodology: { heading: 'How the forecast is made', note: 'One model. Weekly point-in-time predictions, settled after the games.' },
  }

  return <div className="app-shell">
    <aside className="sidebar">
      <div className="brand"><div className="brand-mark">CF</div><div><strong>Forecast Desk</strong><span>2026 season model</span></div></div>
      <nav>
        <button className={page === 'predictions' ? 'active' : ''} onClick={() => setPage('predictions')}><Gauge /> Predictions</button>
        <button className={page === 'lines' ? 'active' : ''} onClick={() => setPage('lines')}><CalendarDays /> Lines</button>
        <button className={page === 'statistics' ? 'active' : ''} onClick={() => setPage('statistics')}><Database /> Statistics</button>
        <button className={page === 'methodology' ? 'active' : ''} onClick={() => setPage('methodology')}><BarChart3 /> Methodology</button>
      </nav>
      <div className="sidebar-foot"><div className="signal"><span className="dot" /> {forecasts.length ? 'Forecast published' : 'Awaiting a run'}</div><small>{model}<br />Updated {updated ? updated.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' }) : 'not yet'}</small></div>
    </aside>
    <main className="main-content">
      <header className="topbar"><div><div className="eyebrow">{week ? `2026 week ${week}` : 'Weekly slate'}{runId ? ` / ${runId}` : ''}</div><h1>{titles[page].heading}</h1><p>{titles[page].note}</p></div><button className="refresh" onClick={refresh} disabled={live}><RefreshCw className={live ? 'spin' : ''} /> {live ? 'Updating' : 'Refresh model'}</button></header>
      <section className="metrics"><Metric icon={<Activity />} label="Model state" value={forecasts.length ? 'Published' : 'Waiting'} note={model} good={forecasts.length > 0} /><Metric icon={<ShieldCheck />} label="Avg. confidence" value={forecasts.length ? pct(avgConfidence) : '—'} note="Winner probability" good /><Metric icon={<CheckCircle2 />} label="Last validation" value={validation.games ? `${validation.games} games` : 'Pending'} note={validation.status === 'settled' ? `Log loss ${validation.log_loss?.toFixed(3)}` : 'Awaiting completed results'} /><Metric icon={<Database />} label="Data sources" value={sources.length ? String(sources.length) : '—'} note="Stored in the source catalog" /></section>
      {page === 'predictions' && <div className="content-grid"><section className="panel forecast-panel"><div className="panel-head"><div><div className="eyebrow">Upcoming games</div><h2>Predicted outcomes</h2></div><span className="live-pill"><span className="dot" /> {forecasts.length ? 'Published' : 'Waiting'}</span></div><div className="forecast-list">{forecasts.length ? forecasts.map(game => <article className="game-row" key={game.game_id}><div className="game-meta"><span>{dateLabel(game.kickoff)}</span><span className="muted">Game {game.game_id}</span></div><div className="matchup"><div><strong>{game.away_team}</strong><span>away</span></div><div className="at">@</div><div><strong>{game.home_team}</strong><span>home</span></div></div><div className="score"><strong>{game.expected_away_score.toFixed(1)} — {game.expected_home_score.toFixed(1)}</strong><span>Expected score</span></div><div className="probability"><div><strong>{game.winner}</strong><span>{pct(game.confidence ?? Math.max(game.home_win_probability, 1 - game.home_win_probability))} confidence</span></div><div className="prob-bar"><i style={{ width: `${(game.confidence ?? game.home_win_probability) * 100}%` }} /></div><small>{pct(game.home_win_probability)} home win</small></div><ArrowUpRight className="row-arrow" /></article>) : <p className="panel-note">No forecast has been published yet. The board fills when the weekly pipeline writes forecasts.json.</p>}</div></section><aside className="right-column"><section className="panel validation-panel"><div className="panel-head"><div><div className="eyebrow">Ongoing learning</div><h2>Weekly validation</h2></div><BarChart3 /></div><div className="validation-status"><span className={validation.status === 'settled' ? 'status-good' : 'status-waiting'}>{validation.status === 'settled' ? 'Settled' : 'Awaiting results'}</span><span>Prior week</span></div><div className="validation-grid"><Stat label="Log loss" value={validation.log_loss?.toFixed(3) ?? '—'} /><Stat label="Brier score" value={validation.brier?.toFixed(3) ?? '—'} /><Stat label="Margin MAE" value={validation.margin_mae?.toFixed(1) ?? '—'} /></div><p className="panel-note">Each weekly forecast is settled against completed games before the next Sunday run.</p></section></aside></div>}
      {page === 'lines' && <section className="panel forecast-panel"><div className="panel-head"><div><div className="eyebrow">Sportsbook format</div><h2>Spread and moneyline</h2></div></div><div className="forecast-list">{forecasts.length ? forecasts.map(game => <article className="game-row" key={game.game_id}><div className="game-meta"><span>{dateLabel(game.kickoff)}</span><span className="muted">{game.away_team} @ {game.home_team}</span></div><div className="score"><strong>{spreadLabel(game)}</strong><span>Model spread</span></div><div className="probability"><div><strong>{moneylineLabel(game.home_moneyline)} / {moneylineLabel(game.away_moneyline)}</strong><span>Home / away fair ML</span></div><small>{game.market_spread == null ? 'Book spread unavailable' : `Book ${game.market_spread}`}</small></div></article>) : <p className="panel-note">No forecast has been published yet.</p>}</div></section>}
      {page === 'statistics' && <section className="panel"><div className="panel-head"><div><div className="eyebrow">Recorded rates</div><h2>What the model reads</h2></div></div><p className="panel-note">Team scoring, EPA, success rate, explosiveness, yards per play, turnovers, and player passing, rushing, and receiving totals are stored in team_game_facts and player_game_facts after each forecast run. Same-season games update those rates before the next kickoff. The previous season is the starting history. FootballDB college offense, defense, scoring, and player pages stay in the source catalog.</p><div className="source-row"><span>Sources in this checkout</span></div>{(sources.length ? sources.map(source => source.name) : ['cfbfastR', 'collegefootballdata', 'cfbtrack', 'cfbtracker', 'cfbresults', 'collegefootballdb', 'footballdb']).map(source => <div className="source-row" key={source}><span>{source}</span><span className="source-state"><i className="dot" /> Catalog</span></div>)}</section>}
      {page === 'methodology' && <section className="panel"><div className="panel-head"><div><div className="eyebrow">One model</div><h2>Weekly point-in-time forecast</h2></div></div><p className="panel-note">The hierarchical team state sets the score location from games already completed. Same-season and prior-season points and play rates enter that forecast before kickoff. Weather, rest, injuries, and a recorded line, when present, enter as known-future information. Expected scores are the model means. The winner is the side with home win probability at least 50 percent. Confidence is the larger of that probability and one minus it. Spread and moneyline on the Lines page are the same forecast written in sportsbook form. Each Sunday run is validated against finished games before the next week.</p><p className="panel-note">This is not a live player-tracking engine and not a betting instruction. Probabilities are model outputs.</p></section>}
      <footer><span><TriangleAlert /> Probabilities are model outputs, not guarantees.</span><span>Last published: {updated ? updated.toLocaleString() : 'not yet'}</span></footer>
    </main>
  </div>
}

function Metric({ icon, label, value, note, good = false }: { icon: React.ReactNode; label: string; value: string; note: string; good?: boolean }) { return <div className="metric"><div className="metric-icon">{icon}</div><div><span>{label}</span><strong className={good ? 'good-text' : ''}>{value}</strong><small>{note}</small></div></div> }
function Stat({ label, value }: { label: string; value: string }) { return <div><span>{label}</span><strong>{value}</strong></div> }
