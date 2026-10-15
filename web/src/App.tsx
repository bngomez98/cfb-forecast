import { useEffect, useMemo, useState } from 'react'
import {
  Activity,
  ArrowRight,
  BarChart3,
  BookOpen,
  CalendarDays,
  ChevronDown,
  ChevronUp,
  CircleDot,
  Database,
  Gauge,
  Info,
  LineChart,
  RefreshCw,
  Search,
  ShieldCheck,
  SlidersHorizontal,
  Target,
  TrendingUp,
} from 'lucide-react'

type LiveResult = {
  game_id: number
  status: 'scheduled' | 'live' | 'final'
  status_detail?: string | null
  period?: number | null
  clock?: string | null
  home_points?: number | null
  away_points?: number | null
}
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
type Board = {
  forecasts?: Forecast[]
  sources?: Source[]
  games?: number
  as_of?: string
  model?: string
  week?: number | null
  run_id?: string
}
type Validation = { status: string; games?: number; log_loss?: number; brier?: number; margin_mae?: number }
type Page = 'forecast' | 'moneylines' | 'methodology' | 'data'
type Range = 'week' | 'all'
type Sort = 'kickoff' | 'confidence' | 'margin'

const PAGES: { id: Page; label: string }[] = [
  { id: 'forecast', label: 'Game picks' },
  { id: 'moneylines', label: 'Odds & value' },
  { id: 'methodology', label: 'How it works' },
  { id: 'data', label: 'Track record' },
]
const DAY = 24 * 60 * 60 * 1000

function pct(value: number, digits = 0) {
  return `${(value * 100).toFixed(digits)}%`
}
function homeChance(game: Forecast) {
  return Math.min(Math.max(Number(game.home_win_probability), 0), 1)
}
function margin(game: Forecast) {
  return Number(game.projected_home_margin ?? game.expected_home_score - game.expected_away_score)
}
function fairMoneyline(probability: number) {
  if (!(probability > 0 && probability < 1)) return null
  return probability >= 0.5
    ? -Math.round((100 * probability) / (1 - probability))
    : Math.round((100 * (1 - probability)) / probability)
}
function oddsLabel(value?: number | null) {
  if (value == null || Number.isNaN(Number(value))) return '—'
  return value > 0 ? `+${value}` : String(value)
}
function spreadFor(game: Forecast, side: 'home' | 'away') {
  const home = game.model_spread ?? -margin(game)
  const value = side === 'home' ? Number(home) : -Number(home)
  if (Number.isNaN(value)) return '—'
  if (Math.abs(value) < 0.05) return 'PK'
  return value > 0 ? `+${value.toFixed(1)}` : value.toFixed(1)
}
function bookSpreadFor(game: Forecast, side: 'home' | 'away') {
  if (game.market_spread == null || Number.isNaN(Number(game.market_spread))) return '—'
  const value = side === 'home' ? Number(game.market_spread) : -Number(game.market_spread)
  if (value === 0) return 'PK'
  return value > 0 ? `+${value}` : String(value)
}
function dayKey(value: string) {
  return new Intl.DateTimeFormat('en-US', { weekday: 'long', month: 'long', day: 'numeric' }).format(new Date(value))
}
function timeLabel(value: string) {
  return new Intl.DateTimeFormat('en-US', { hour: 'numeric', minute: '2-digit', timeZoneName: 'short' }).format(new Date(value))
}
function shortDate(value: Date) {
  return new Intl.DateTimeFormat('en-US', { month: 'short', day: 'numeric', year: 'numeric' }).format(value)
}
function confidenceLabel(value: number) {
  if (value >= 0.75) return 'Strong'
  if (value >= 0.62) return 'Moderate'
  return 'Lean'
}
function initials(team: string) {
  return team.split(/\s+/).map(word => word[0]).join('').slice(0, 2).toUpperCase()
}

export function App() {
  const [page, setPage] = useState<Page>('forecast')
  const [forecasts, setForecasts] = useState<Forecast[]>([])
  const [sources, setSources] = useState<Source[]>([])
  const [boardMeta, setBoardMeta] = useState<Board>({})
  const [validation, setValidation] = useState<Validation>({ status: 'awaiting_results' })
  const [loading, setLoading] = useState(false)
  const [query, setQuery] = useState('')
  const [range, setRange] = useState<Range>('week')
  const [sort, setSort] = useState<Sort>('kickoff')
  const [liveResults, setLiveResults] = useState<Record<number, LiveResult>>({})
  const [scoresUpdatedAt, setScoresUpdatedAt] = useState<Date | null>(null)

  async function refresh() {
    setLoading(true)
    try {
      const published = await fetch('/forecasts.json', { cache: 'no-store' })
      let currentForecasts = forecasts
      if (published.ok) {
        const board = await published.json() as Board
        setBoardMeta(board)
        if (Array.isArray(board.forecasts)) {
          currentForecasts = board.forecasts
          setForecasts(board.forecasts)
        }
        if (Array.isArray(board.sources)) setSources(board.sources)
      }
      const firstKickoff = currentForecasts.length ? new Date(Math.min(...currentForecasts.map(game => new Date(game.kickoff).getTime()))) : new Date()
      const start = new Date(Math.max(firstKickoff.getTime(), Date.now() - DAY))
      const end = new Date(start.getTime() + 8 * DAY)
      const response = await fetch(`/api/live-results?start=${encodeURIComponent(start.toISOString())}&end=${encodeURIComponent(end.toISOString())}`, { cache: 'no-store' })
      if (response.ok) {
        const data = await response.json() as { updated_at?: string; results?: LiveResult[] }
        if (Array.isArray(data.results)) setLiveResults(Object.fromEntries(data.results.map(result => [result.game_id, result])))
        if (data.updated_at) setScoresUpdatedAt(new Date(data.updated_at))
      }
    } catch {
      // The static publication is the production fallback when the self-hosted API is unavailable.
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    void refresh()
    const timer = window.setInterval(() => { void refresh() }, 60_000)
    return () => window.clearInterval(timer)
  }, [])
  useEffect(() => { window.scrollTo({ top: 0, behavior: 'smooth' }) }, [page])

  const updated = boardMeta.as_of ? new Date(boardMeta.as_of) : null
  const chronological = useMemo(() => [...forecasts].sort((a, b) => new Date(a.kickoff).getTime() - new Date(b.kickoff).getTime()), [forecasts])
  const weekEnd = useMemo(() => {
    const start = updated ?? (chronological.length ? new Date(chronological[0].kickoff) : null)
    return start ? start.getTime() + 8 * DAY : null
  }, [updated, chronological])
  const weekGames = useMemo(() => weekEnd == null ? chronological : chronological.filter(game => new Date(game.kickoff).getTime() < weekEnd), [chronological, weekEnd])
  const shown = useMemo(() => {
    const text = query.trim().toLowerCase()
    const pool = range === 'week' ? weekGames : chronological
    const filtered = text ? pool.filter(game => `${game.home_team} ${game.away_team}`.toLowerCase().includes(text)) : pool
    return [...filtered].sort((a, b) => {
      if (sort === 'confidence') return Math.max(homeChance(b), 1 - homeChance(b)) - Math.max(homeChance(a), 1 - homeChance(a))
      if (sort === 'margin') return Math.abs(margin(b)) - Math.abs(margin(a))
      return new Date(a.kickoff).getTime() - new Date(b.kickoff).getTime()
    })
  }, [query, range, sort, weekGames, chronological])
  const days = useMemo(() => {
    if (sort !== 'kickoff') return [{ day: sort === 'confidence' ? 'Ranked by model confidence' : 'Ranked by projected margin', games: shown }]
    const groups: { day: string; games: Forecast[] }[] = []
    for (const game of shown) {
      const day = dayKey(game.kickoff)
      if (!groups.length || groups[groups.length - 1].day !== day) groups.push({ day, games: [] })
      groups[groups.length - 1].games.push(game)
    }
    return groups
  }, [shown, sort])
  const avgConfidence = weekGames.length ? weekGames.reduce((sum, game) => sum + Math.max(homeChance(game), 1 - homeChance(game)), 0) / weekGames.length : 0
  const strongPicks = weekGames.filter(game => Math.max(homeChance(game), 1 - homeChance(game)) >= 0.7).length
  const closest = weekGames.length ? [...weekGames].sort((a, b) => Math.abs(margin(a)) - Math.abs(margin(b)))[0] : null
  const weekName = boardMeta.week ? `Week ${boardMeta.week}` : 'This week'

  return <div className="app-shell">
    <header className="topbar">
      <a className="brand" href="#top" onClick={() => setPage('forecast')} aria-label="FieldQuant home">
        <span className="brand-mark"><LineChart aria-hidden="true" /></span>
        <span><strong>FIELDQUANT</strong><small>FBS FORECAST DESK</small></span>
      </a>
      <nav aria-label="Primary navigation">
        {PAGES.map(item => <button key={item.id} className={page === item.id ? 'active' : ''} aria-current={page === item.id ? 'page' : undefined} onClick={() => setPage(item.id)}>{item.label}</button>)}
      </nav>
      <div className="model-status"><i aria-hidden="true" /> Picks updated</div>
    </header>

    <main id="top">
      {page === 'forecast' && <>
        <Hero weekName={weekName} updated={updated} games={weekGames.length} onRefresh={refresh} loading={loading} />
        <section className="content-section">
          <div className="live-score-strip" role="status"><span><i aria-hidden="true" /> Live 2026 scores refresh every 60 seconds</span><small>{scoresUpdatedAt ? `Last checked ${scoresUpdatedAt.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' })}` : 'Connecting to score feed'}</small></div>
          <div className="metric-grid" aria-label="Weekly forecast summary">
            <Metric icon={CalendarDays} label="Games in window" value={weekGames.length ? String(weekGames.length) : '—'} detail={`${weekName} slate`} />
            <Metric icon={Gauge} label="Average pick strength" value={weekGames.length ? pct(avgConfidence, 1) : '—'} detail="Average chance for the picked team" />
            <Metric icon={Target} label="Strong picks" value={weekGames.length ? String(strongPicks) : '—'} detail="Games at 70% or higher" />
            <Metric icon={CircleDot} label="Tightest matchup" value={closest ? `${Math.abs(margin(closest)).toFixed(1)} pts` : '—'} detail={closest ? `${closest.away_team} at ${closest.home_team}` : 'Awaiting slate'} />
          </div>
          <BoardControls query={query} setQuery={setQuery} range={range} setRange={setRange} sort={sort} setSort={setSort} weekName={weekName} />
          <Board days={days} mode="forecast" hasGames={chronological.length > 0} query={query} liveResults={liveResults} />
        </section>
      </>}

      {page === 'moneylines' && <MoneylinePage
        weekName={weekName}
        updated={updated}
        games={shown}
        allGames={chronological}
        query={query}
        setQuery={setQuery}
        range={range}
        setRange={setRange}
        sort={sort}
        setSort={setSort}
      />}
      {page === 'methodology' && <MethodologyPage model={boardMeta.model} />}
      {page === 'data' && <DataPage sources={sources} validation={validation} board={boardMeta} />}
    </main>

    <footer>
      <div><strong>FIELDQUANT</strong><span>Independent college football forecasting</span></div>
      <p>Probabilities are estimates, not guarantees. No wagers are placed.</p>
      <span>{updated ? `Run ${boardMeta.run_id ?? 'current'} · ${shortDate(updated)}` : 'Awaiting publication'}</span>
    </footer>
  </div>
}

function Hero({ weekName, updated, games, onRefresh, loading }: { weekName: string; updated: Date | null; games: number; onRefresh: () => void; loading: boolean }) {
  return <section className="hero">
    <div className="hero-inner">
      <div className="eyebrow"><span>2026 FBS</span><span>{weekName}</span><span>{games} games</span></div>
      <h1>Smarter picks for every game.</h1>
      <p>See who is more likely to win, the predicted score, and what a fair price looks like—all in one place.</p>
      <div className="hero-actions">
        <button className="primary-action" onClick={() => document.querySelector('.forecast-board')?.scrollIntoView({ behavior: 'smooth' })}>View the board <ArrowRight aria-hidden="true" /></button>
        <button className="secondary-action" onClick={onRefresh} disabled={loading}><RefreshCw className={loading ? 'spin' : ''} aria-hidden="true" />{loading ? 'Refreshing' : 'Refresh data'}</button>
      </div>
      <div className="run-meta"><ShieldCheck aria-hidden="true" /><span>Point-in-time forecast</span><i /> <span>{updated ? `Generated ${shortDate(updated)}` : 'No forecast published'}</span></div>
    </div>
    <div className="hero-visual" aria-hidden="true">
      <div className="field-line line-one" /><div className="field-line line-two" /><div className="field-line line-three" />
      <div className="hero-number">26</div><div className="hero-chart"><i /><i /><i /><i /><i /><i /><i /><i /></div>
    </div>
  </section>
}

function Metric({ icon: Icon, label, value, detail }: { icon: typeof Activity; label: string; value: string; detail: string }) {
  return <article className="metric-card">
    <div className="metric-label"><Icon aria-hidden="true" /><span>{label}</span></div>
    <strong>{value}</strong>
    <p>{detail}</p>
  </article>
}

function BoardControls({ query, setQuery, range, setRange, sort, setSort, weekName }: {
  query: string; setQuery: (value: string) => void; range: Range; setRange: (value: Range) => void; sort: Sort; setSort: (value: Sort) => void; weekName: string
}) {
  return <section className="board-controls" aria-label="Forecast filters">
    <label className="search-field"><Search aria-hidden="true" /><input type="search" placeholder="Search team" value={query} onChange={event => setQuery(event.target.value)} aria-label="Search team" /></label>
    <div className="segmented" role="group" aria-label="Date range">
      <button className={range === 'week' ? 'active' : ''} onClick={() => setRange('week')}>{weekName}</button>
      <button className={range === 'all' ? 'active' : ''} onClick={() => setRange('all')}>Full board</button>
    </div>
    <label className="sort-field"><SlidersHorizontal aria-hidden="true" /><span className="sr-only">Sort forecasts</span><select value={sort} onChange={event => setSort(event.target.value as Sort)}><option value="kickoff">Sort: Kickoff</option><option value="confidence">Sort: Confidence</option><option value="margin">Sort: Margin</option></select><ChevronDown aria-hidden="true" /></label>
  </section>
}

function Board({ days, mode, hasGames, query, liveResults = {} }: { days: { day: string; games: Forecast[] }[]; mode: 'forecast' | 'moneyline'; hasGames: boolean; query: string; liveResults?: Record<number, LiveResult> }) {
  if (!hasGames) return <div className="empty-state"><Database aria-hidden="true" /><h2>No forecast published</h2><p>The board fills when the weekly run writes the public forecast file.</p></div>
  if (!days.some(day => day.games.length)) return <div className="empty-state"><Search aria-hidden="true" /><h2>No matching games</h2><p>No team matches “{query}”.</p></div>
  return <div className="forecast-board">
    {days.map(day => <section className="day-group" key={day.day}>
      <div className="day-heading"><h2>{day.day}</h2><span>{day.games.length} {day.games.length === 1 ? 'game' : 'games'}</span></div>
      <div className="game-grid">{day.games.map(game => mode === 'forecast' ? <ForecastCard key={game.game_id} game={game} result={liveResults[game.game_id]} /> : <MoneylineRow key={game.game_id} game={game} />)}</div>
    </section>)}
  </div>
}

function ForecastCard({ game, result }: { game: Forecast; result?: LiveResult }) {
  const [expanded, setExpanded] = useState(false)
  const home = homeChance(game)
  const away = 1 - home
  const pickHome = home >= 0.5
  const confidence = Math.max(home, away)
  const projectedTotal = game.expected_home_score + game.expected_away_score
  const pickedTeam = pickHome ? game.home_team : game.away_team
  const detailsId = `game-details-${game.game_id}`
  const statusLabel = result?.status === 'final' ? 'Final' : result?.status === 'live' ? (result.status_detail ?? `${result.clock ?? ''} · Q${result.period ?? ''}`) : timeLabel(game.kickoff)
  return <article className={`game-card ${result?.status === 'live' ? 'is-live' : ''}`}>
    <header>
      <div><span className="live-dot" aria-hidden="true" /><span>{statusLabel}</span></div>
      <span className={`confidence ${confidence >= 0.7 ? 'high' : ''}`}>{confidenceLabel(confidence)} · {pct(confidence)}</span>
    </header>
    <div className="matchup">
      <TeamLine team={game.away_team} score={game.expected_away_score} actualScore={result?.away_points} chance={away} picked={!pickHome} side="Away" status={result?.status} />
      <TeamLine team={game.home_team} score={game.expected_home_score} actualScore={result?.home_points} chance={home} picked={pickHome} side="Home" status={result?.status} />
    </div>
    <div className="probability-track" aria-label={`${game.away_team} ${pct(away)}, ${game.home_team} ${pct(home)}`}><i style={{ width: `${away * 100}%` }} /><i style={{ width: `${home * 100}%` }} /></div>
    <dl className="game-details">
      <div><dt>Predicted spread</dt><dd>{pickedTeam} {spreadFor(game, pickHome ? 'home' : 'away')}</dd></div>
      <div><dt>Predicted points</dt><dd>{projectedTotal.toFixed(1)}</dd></div>
      <div><dt>Fair odds</dt><dd>{oddsLabel(fairMoneyline(confidence))}</dd></div>
    </dl>
    <button className="details-toggle" onClick={() => setExpanded(value => !value)} aria-expanded={expanded} aria-controls={detailsId}>
      {expanded ? 'Hide details' : 'Explain this pick'}
      {expanded ? <ChevronUp aria-hidden="true" /> : <ChevronDown aria-hidden="true" />}
    </button>
    {expanded && <div className="pick-explainer" id={detailsId}>
      <p><strong>Why {pickedTeam}?</strong> We give them a {pct(confidence)} chance to win and predict a final score of {game.expected_away_score.toFixed(0)}–{game.expected_home_score.toFixed(0)}.</p>
      <dl>
        <div><dt>Predicted spread</dt><dd>Our expected winning margin, shown in the same format as a sportsbook spread.</dd></div>
        <div><dt>Fair odds</dt><dd>The price that matches our win estimate before a sportsbook adds its margin.</dd></div>
      </dl>
    </div>}
  </article>
}

function TeamLine({ team, score, actualScore, chance, picked, side, status }: { team: string; score: number; actualScore?: number | null; chance: number; picked: boolean; side: string; status?: LiveResult['status'] }) {
  return <div className={`team-line ${picked ? 'picked' : ''}`}>
    <span className="team-badge" aria-hidden="true">{initials(team)}</span>
    <span className="team-name"><strong>{team}</strong><small>{side}{picked ? ' · Model pick' : ''}</small></span>
    <span className="projected-score" title={status && status !== 'scheduled' ? `Model projected ${score.toFixed(1)}` : 'Model projected score'}>{status && status !== 'scheduled' && actualScore != null ? actualScore.toFixed(0) : score.toFixed(1)}<small>{status && status !== 'scheduled' ? 'ACTUAL' : 'PRED'}</small></span>
    <span className="win-probability">{pct(chance)}</span>
  </div>
}

function MoneylinePage(props: {
  weekName: string; updated: Date | null; games: Forecast[]; allGames: Forecast[]; query: string; setQuery: (value: string) => void; range: Range; setRange: (value: Range) => void; sort: Sort; setSort: (value: Sort) => void
}) {
  const days = [{ day: 'Model prices', games: props.games }]
  return <>
    <PageHeader eyebrow="ODDS VIEW" title="Odds & value" description="Compare our estimated fair odds with the posted sportsbook price." updated={props.updated} />
    <section className="content-section">
      <div className="notice"><Info aria-hidden="true" /><div><strong>Use fair odds as your comparison point.</strong><p>If our fair price is −150, we estimate a 60% chance to win. A more expensive sportsbook price gives you less value; a cheaper price may give you more.</p></div></div>
      <BoardControls {...props} />
      <div className="moneyline-header" aria-hidden="true"><span>Matchup</span><span>Win probability</span><span>Fair moneyline</span><span>Published price</span></div>
      <Board days={days} mode="moneyline" hasGames={props.allGames.length > 0} query={props.query} />
      <section className="odds-guide">
        <div><BookOpen aria-hidden="true" /><h2>How to read American odds</h2></div>
        <div className="odds-formulas">
          <article><span>Favorite</span><strong>−150</strong><p>Risk $150 to profit $100. Break-even probability: 60.0%.</p></article>
          <article><span>Underdog</span><strong>+180</strong><p>Risk $100 to profit $180. Break-even probability: 35.7%.</p></article>
          <article><span>Model edge</span><strong>Probability gap</strong><p>Model win probability minus the sportsbook&apos;s implied probability after accounting for price.</p></article>
        </div>
      </section>
    </section>
  </>
}

function MoneylineRow({ game }: { game: Forecast }) {
  const home = homeChance(game)
  const away = 1 - home
  return <article className="moneyline-row">
    <div className="moneyline-matchup"><span>{timeLabel(game.kickoff)}</span><strong>{game.away_team} <em>at</em> {game.home_team}</strong></div>
    <div className="dual-value"><span>{game.away_team} <b>{pct(away, 1)}</b></span><span>{game.home_team} <b>{pct(home, 1)}</b></span></div>
    <div className="dual-value mono"><span>{oddsLabel(fairMoneyline(away))}</span><span>{oddsLabel(fairMoneyline(home))}</span></div>
    <div className="dual-value mono muted"><span>{oddsLabel(game.away_moneyline)}</span><span>{oddsLabel(game.home_moneyline)}</span></div>
  </article>
}

function PageHeader({ eyebrow, title, description, updated }: { eyebrow: string; title: string; description: string; updated?: Date | null }) {
  return <header className="page-header"><div className="page-header-inner"><span className="section-eyebrow">{eyebrow}</span><h1>{title}</h1><p>{description}</p>{updated && <small>Current run · {shortDate(updated)}</small>}</div></header>
}

const METHOD_STEPS = [
  { icon: Database, number: '01', title: 'Time-safe inputs', text: 'Schedules, final scores, per-game team statistics, current-season rates, rest, and captured market lines are timestamped. A result cannot enter a forecast until it was knowable, and future games marked final stop the run.' },
  { icon: Activity, number: '02', title: 'Dynamic team ratings', text: 'Every FBS team carries separate offense and defense states for points, yards per play, pass efficiency, rush efficiency, and first-down rate. Bayesian updates move uncertain teams more and established teams less.' },
  { icon: SlidersHorizontal, number: '03', title: 'Score expectation', text: 'A regularized regression converts the two teams’ rating states into expected points. Recent games receive more weight, home field is worth 2.5 points, neutral sites receive none, and ratings drift toward average between games.' },
  { icon: LineChart, number: '04', title: 'Context adjustment', text: 'Current scoring and efficiency rates, rest, and a pre-forecast spread or total can shift the score location. Missing-input indicators distinguish unavailable evidence from a genuinely average observation.' },
  { icon: BarChart3, number: '05', title: 'Joint distribution', text: 'Expected margin and total enter one Gaussian score distribution using historical residual covariance. That single distribution produces win, spread, and total probabilities so the outputs cannot contradict each other.' },
  { icon: ShieldCheck, number: '06', title: 'Walk-forward grading', text: 'Each forecast is immutable after publication. Later results are joined to the saved prediction and scored with log loss, Brier score, margin error, calibration, and hypothetical return—without rewriting history.' },
]

function MethodologyPage({ model }: { model?: string }) {
  return <>
    <PageHeader eyebrow="MODEL DOCUMENTATION" title="From raw games to one probability" description="A transparent, point-in-time forecasting pipeline built to answer one question consistently: how likely is each team to win?" />
    <section className="content-section methodology">
      <div className="method-intro"><div><span>PRODUCTION SPECIFICATION</span><h2>{model ?? 'Dynamic Gaussian state + regularized score head'}</h2></div><p>The model predicts scores first. Win probabilities, model spreads, totals, and fair moneylines are all transformations of the same score distribution—not separate opinions.</p></div>
      <div className="method-flow">{METHOD_STEPS.map(step => <article key={step.number}><div className="step-icon"><step.icon aria-hidden="true" /></div><span>{step.number}</span><h2>{step.title}</h2><p>{step.text}</p></article>)}</div>
      <section className="technical-grid">
        <article><span className="section-eyebrow">RATING LAYER</span><h2>Opponent-adjusted learning</h2><p>A strong performance against a strong opponent moves the latent state differently from the same raw output against a weak opponent. Uncertainty controls update size, and weekly mean reversion prevents stale early-season information from dominating indefinitely.</p><dl><div><dt>Home-field prior</dt><dd>+2.5 points</dd></div><div><dt>Season transition</dt><dd>Preseason shrinkage</dd></div><div><dt>Update form</dt><dd>Bayesian state</dd></div></dl></article>
        <article><span className="section-eyebrow">SCORE LAYER</span><h2>One coherent distribution</h2><p>The score head estimates each team&apos;s expected points. Historical residual covariance captures how paired scores vary around those means; the production probabilities use the resulting Gaussian approximation.</p><div className="equation"><span>Projected margin</span><strong>μ<sub>home</sub> − μ<sub>away</sub></strong></div><div className="equation"><span>Win probability</span><strong>P(margin &gt; 0)</strong></div></article>
      </section>
      <section className="guardrails"><div><ShieldCheck aria-hidden="true" /><h2>What the model refuses to do</h2></div><ul><li>It does not invent missing lines, weather, injuries, scores, or starters.</li><li>It does not use a result or line that was unavailable at forecast time.</li><li>It does not let a separate winner curve override the score distribution.</li><li>It does not activate weather or injury effects until enough graded forecasts show out-of-sample improvement.</li><li>It does not present hypothetical betting returns as realized performance.</li></ul></section>
      <section className="training-timeline"><span className="section-eyebrow">SEASON ROLES</span><h2>Walk-forward model development</h2><div><article><strong>2021</strong><span>Warmup</span><p>Initial team states are formed; these games are not used for final 2026 grading.</p></article><article><strong>2022–23</strong><span>Estimation</span><p>Fit the initial expected-points regression.</p></article><article><strong>2024</strong><span>Selection</span><p>Choose rating settings and regularization strength.</p></article><article><strong>2025</strong><span>Evaluation</span><p>Compare candidate systems on a later season.</p></article><article><strong>2026</strong><span>Forecast</span><p>Publish immutable weekly predictions.</p></article></div></section>
      <div className="disclosure"><Info aria-hidden="true" /><p><strong>Important limitation:</strong> because 2025 informed model selection, it is not a fully untouched test of the final specification. Published 2026 forecasts provide the cleanest prospective record.</p></div>
    </section>
  </>
}

function DataPage({ sources, validation, board }: { sources: Source[]; validation: Validation; board: Board }) {
  const gradedGames = validation.games ?? 0
  return <>
    <PageHeader eyebrow="TRACK RECORD" title="Published accuracy" description="Results are shown only after a published pick can be matched to a final score. No estimate or made-up accuracy percentage is displayed." />
    <section className="content-section data-page">
      <div className="validation-grid">
        <Metric icon={Database} label="Published picks" value={board.games ? String(board.games) : '—'} detail="Picks saved before kickoff" />
        <Metric icon={ShieldCheck} label="Graded games" value={String(gradedGames)} detail="Finished games matched to picks" />
        <Metric icon={Target} label="Average score error" value={validation.margin_mae != null ? `${validation.margin_mae.toFixed(2)} pts` : 'Not available'} detail="Average miss on winning margin" />
        <Metric icon={Gauge} label="Probability error" value={validation.brier != null ? validation.brier.toFixed(3) : 'Not available'} detail="0 is perfect; lower is better" />
      </div>
      <div className="accuracy-note"><Info aria-hidden="true" /><div><strong>{gradedGames > 0 ? `${gradedGames} games are included in these results.` : 'No completed games have been graded yet.'}</strong><p>{gradedGames > 0 ? `Log loss is ${validation.log_loss?.toFixed(3) ?? 'not available'}. It assigns a larger penalty when a high-confidence pick loses.` : 'The current file contains future picks, so an accuracy percentage would be misleading. Results will appear after those games finish and are matched to these saved picks.'}</p></div></div>
      <div className="data-columns">
        <section><span className="section-eyebrow">INPUT FAMILIES</span><h2>What enters the forecast</h2><ul className="check-list"><li><b>Game results</b><span>Final scores and game-level team statistics</span></li><li><b>Efficiency</b><span>Points, yards per play, pass, rush, and first-down rates</span></li><li><b>Situation</b><span>Venue, neutral-site status, and days of rest</span></li><li><b>Market context</b><span>Spread and total only when captured before forecast time</span></li><li><b>Conditional context</b><span>Weather and injuries recorded, but gated by validation</span></li></ul></section>
        <section><span className="section-eyebrow">PUBLIC SOURCES</span><h2>Traceable inputs</h2><div className="source-list">{(sources.length ? sources : [{ name: 'sportsdataverse', url: 'https://github.com/sportsdataverse/sportsdataverse-data' }]).map(source => <a key={source.name} href={source.url} target="_blank" rel="noreferrer"><span>{source.name.replace(/_/g, ' ')}</span><ArrowRight aria-hidden="true" /></a>)}</div></section>
      </div>
    </section>
  </>
}
