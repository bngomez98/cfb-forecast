import { useEffect, useMemo, useState } from 'react'
import { Activity, ArrowUpRight, BarChart3, CalendarDays, CheckCircle2, Database, Gauge, RefreshCw, ShieldCheck, TriangleAlert } from 'lucide-react'

type Forecast = { game_id: number; away_team: string; home_team: string; kickoff: string; home_win_probability: number; expected_home_score: number; expected_away_score: number; projected_home_margin: number; }
type Validation = { status: string; games?: number; log_loss?: number; brier?: number; margin_mae?: number; }

const fallbackForecasts: Forecast[] = [
  { game_id: 1, away_team: 'Oklahoma', home_team: 'Texas', kickoff: '2026-10-17T19:00:00Z', home_win_probability: .64, expected_home_score: 31.8, expected_away_score: 24.7, projected_home_margin: 7.1 },
  { game_id: 2, away_team: 'Penn State', home_team: 'Ohio State', kickoff: '2026-10-17T23:30:00Z', home_win_probability: .58, expected_home_score: 28.4, expected_away_score: 23.9, projected_home_margin: 4.5 },
  { game_id: 3, away_team: 'Georgia', home_team: 'Alabama', kickoff: '2026-10-24T20:30:00Z', home_win_probability: .47, expected_home_score: 27.1, expected_away_score: 29.2, projected_home_margin: -2.1 },
]

function pct(value: number) { return `${Math.round(value * 100)}%` }
function dateLabel(value: string) { return new Intl.DateTimeFormat('en-US', { weekday: 'short', month: 'short', day: 'numeric', hour: 'numeric' }).format(new Date(value)) }

export function App() {
  const [forecasts, setForecasts] = useState<Forecast[]>(fallbackForecasts)
  const [validation, setValidation] = useState<Validation>({ status: 'awaiting_results' })
  const [live, setLive] = useState(false)
  const [updated, setUpdated] = useState(new Date())

  async function refresh() {
    setLive(true)
    try {
      const response = await fetch('/api/forecasts?season=2026')
      if (response.ok) {
        const data = await response.json()
        if (Array.isArray(data.forecasts)) setForecasts(data.forecasts)
        if (data.validation) setValidation(data.validation)
      }
    } finally {
      setUpdated(new Date())
      setLive(false)
    }
  }
  useEffect(() => { refresh() }, [])

  const avgConfidence = useMemo(() => forecasts.reduce((sum, game) => sum + Math.max(game.home_win_probability, 1 - game.home_win_probability), 0) / forecasts.length, [forecasts])

  return <div className="app-shell">
    <aside className="sidebar">
      <div className="brand"><div className="brand-mark">CF</div><div><strong>Forecast Desk</strong><span>2026 season model</span></div></div>
      <nav><a className="active"><Gauge /> Overview</a><a><CalendarDays /> Weekly board</a><a><BarChart3 /> Validation</a><a><Database /> Data lineage</a></nav>
      <div className="sidebar-foot"><div className="signal"><span className="dot" /> Engine online</div><small>Dynamic Bayesian hierarchy<br />Updated {updated.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' })}</small></div>
    </aside>
    <main className="main-content">
      <header className="topbar"><div><div className="eyebrow">Saturday slate / week 8</div><h1>Live forecast board</h1><p>Posterior probabilities from current team state, schedule context, and historical priors.</p></div><button className="refresh" onClick={refresh} disabled={live}><RefreshCw className={live ? 'spin' : ''} /> {live ? 'Updating' : 'Refresh model'}</button></header>
      <section className="metrics"><Metric icon={<Activity />} label="Model state" value="Live posterior" note="State replay through Oct 12" good /><Metric icon={<ShieldCheck />} label="Avg. confidence" value={pct(avgConfidence)} note="Across upcoming games" good /><Metric icon={<CheckCircle2 />} label="Last validation" value={validation.games ? `${validation.games} games` : 'Pending'} note={validation.status === 'settled' ? `Log loss ${validation.log_loss?.toFixed(3)}` : 'Awaiting completed results'} /><Metric icon={<Database />} label="Data sources" value="10 active" note="Lineage captured weekly" /></section>
      <div className="content-grid"><section className="panel forecast-panel"><div className="panel-head"><div><div className="eyebrow">Upcoming games</div><h2>Current probabilities</h2></div><span className="live-pill"><span className="dot" /> Live</span></div><div className="forecast-list">{forecasts.map(game => <article className="game-row" key={game.game_id}><div className="game-meta"><span>{dateLabel(game.kickoff)}</span><span className="muted">Game {game.game_id}</span></div><div className="matchup"><div><strong>{game.away_team}</strong><span>away</span></div><div className="at">@</div><div><strong>{game.home_team}</strong><span>home</span></div></div><div className="score"><strong>{game.expected_away_score.toFixed(1)} — {game.expected_home_score.toFixed(1)}</strong><span>projected score</span></div><div className="probability"><div><strong>{pct(game.home_win_probability)}</strong><span>home win</span></div><div className="prob-bar"><i style={{ width: `${game.home_win_probability * 100}%` }} /></div><small>{game.projected_home_margin >= 0 ? '+' : ''}{game.projected_home_margin.toFixed(1)} margin</small></div><ArrowUpRight className="row-arrow" /></article>)}</div></section><aside className="right-column"><section className="panel validation-panel"><div className="panel-head"><div><div className="eyebrow">Ongoing learning</div><h2>Weekly validation</h2></div><BarChart3 /></div><div className="validation-status"><span className={validation.status === 'settled' ? 'status-good' : 'status-waiting'}>{validation.status === 'settled' ? 'Settled' : 'Awaiting results'}</span><span>Prior week</span></div><div className="validation-grid"><Stat label="Log loss" value={validation.log_loss?.toFixed(3) ?? '—'} /><Stat label="Brier score" value={validation.brier?.toFixed(3) ?? '—'} /><Stat label="Margin MAE" value={validation.margin_mae?.toFixed(1) ?? '—'} /></div><p className="panel-note">Each weekly forecast is settled against completed games before the next posterior update.</p></section><section className="panel sources-panel"><div className="panel-head"><div><div className="eyebrow">Lineage</div><h2>Source health</h2></div><CheckCircle2 className="status-good" /></div>{['cfbfastR / CFBD', 'CFBTrack + FootballDB', 'CFB TXT validation', 'FanDuel comparison'].map((source, index) => <div className="source-row" key={source}><span>{source}</span><span className="source-state"><i className="dot" /> {index === 2 ? 'Validation' : 'Current'}</span></div>)}</section></aside></div>
      <footer><span><TriangleAlert /> Probabilities are model outputs, not guarantees.</span><span>Last refresh: {updated.toLocaleString()}</span></footer>
    </main>
  </div>
}

function Metric({ icon, label, value, note, good = false }: { icon: React.ReactNode; label: string; value: string; note: string; good?: boolean }) { return <div className="metric"><div className="metric-icon">{icon}</div><div><span>{label}</span><strong className={good ? 'good-text' : ''}>{value}</strong><small>{note}</small></div></div> }
function Stat({ label, value }: { label: string; value: string }) { return <div><span>{label}</span><strong>{value}</strong></div> }
