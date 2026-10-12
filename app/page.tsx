'use client'

import { useEffect, useMemo, useState } from 'react'

type Team = { id: string; name: string; abbreviation: string; score: number | null; rank: number | null }
type Game = { id: string; date: string; week: number; status: string; home: Team; away: Team; venue: string; spread: number | null; marketTotal: number | null; homeWin: number; awayWin: number; margin: number; total: number; homeExpectedPoints: number; awayExpectedPoints: number; cover: number | null; over: number | null; model: string; cutoff: string }
type Data = { forecasts: Game[]; ratings: { id: string; name?: string; abbreviation?: string; offense: number; defense: number; net: number; games: number }[]; fittedGames: number; backtest: { games: number; accuracy: number; logLoss: number; spreadGames: number; spreadMae: number; rows: { week: number; date: string; game: string; actualMargin: number; predictedMargin: number; spread: number | null; cover: number | null; winner: boolean }[] }; dataCutoff: string; source: string; sources?: { name: string; status: string; role: string }[] }

const percent = (value: number | null) => value == null ? '—' : `${(value * 100).toFixed(1)}%`
const margin = (value: number | null) => value == null ? '—' : `${value > 0 ? '+' : ''}${value.toFixed(1)}`
const dateLabel = (value: string) => new Date(value).toLocaleString([], { weekday: 'short', month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' })

export default function Home() {
  const [week, setWeek] = useState('7')
  const [tab, setTab] = useState('Upcoming games')
  const [query, setQuery] = useState('')
  const [data, setData] = useState<Data | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  async function load() {
    setLoading(true)
    setError('')
    try {
      const response = await fetch(`/api/forecast?season=2026&week=${week}`, { cache: 'no-store' })
      const json = await response.json()
      if (!response.ok) throw new Error(json.detail || json.error)
      setData(json)
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'The live FBS feed is unavailable')
    } finally { setLoading(false) }
  }

  useEffect(() => { load() }, [week])
  const games = useMemo(() => data?.forecasts.filter((game) => `${game.home.name} ${game.away.name}`.toLowerCase().includes(query.toLowerCase())) ?? [], [data, query])

  return <main className="terminal-shell">
    <header className="topbar">
      <div className="brand"><div className="brand-mark">FBS</div><div><strong>FORECAST DESK</strong><small>COLLEGE FOOTBALL · POINT-IN-TIME ANALYSIS</small></div></div>
      <div className="divider" />
      <label className="sr-only" htmlFor="week">Week</label><select id="week" value={week} onChange={(event) => setWeek(event.target.value)}>{Array.from({ length: 15 }, (_, index) => <option key={index + 1} value={index + 1}>Week {index + 1}</option>)}</select>
      <span className="live-chip"><i />{loading ? 'UPDATING' : 'LIVE FBS FEED'}</span><div className="top-spacer" />
      <div className="header-kpi"><b>{data?.fittedGames ?? '—'}</b><span>GAMES FITTED</span></div><div className="header-kpi"><b>{data?.forecasts.length ?? '—'}</b><span>WEEK {week} GAMES</span></div>
      <button className="refresh" onClick={load} disabled={loading}>{loading ? 'Updating' : 'Refresh'}</button>
    </header>
    <nav className="tabs" aria-label="Forecast views">{['Upcoming games', 'Team ratings', 'Point-in-time validation', 'Method & sources'].map((item) => <button key={item} className={tab === item ? 'active' : ''} onClick={() => setTab(item)}>{item}</button>)}</nav>
    <div className="content">
      <section className="hero-row"><div><p className="eyebrow">2026 SEASON · WEEK {week.padStart(2, '0')}</p><h1>{tab}</h1><p className="lede">Opponent-adjusted team strength, score distributions, and market comparisons built only from information available before kickoff.</p></div><div className="cutoff"><span>DATA CUTOFF</span><b>{data ? new Date(data.dataCutoff).toLocaleString([], { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' }) : '—'}</b><small>Forecasts update from live source records</small></div></section>
      {error && <div className="error-panel"><b>Live data unavailable</b><span>{error}</span><button onClick={load}>Try again</button></div>}
      {tab === 'Upcoming games' && <><div className="source-strip"><div><small>PRIMARY SOURCE</small><strong>{data?.source ?? 'Waiting for live FBS records'}</strong><small>Missing odds and features remain unavailable; no records are substituted.</small></div><div className="source-list">{data?.sources?.map((source) => <span key={source.name} className={`source-pill ${source.status !== 'reachable' ? 'muted-source' : ''}`} title={`${source.role} · ${source.status}`}><i />{source.name.split(' /')[0]}</span>)}</div></div><div className="toolbar"><div className="section-label"><span className="live-dot" />{games.length} scheduled games</div><input aria-label="Search teams" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search by team name" /></div><section className="game-grid">{loading && !data ? <LoadingCards /> : games.map((game) => <GameCard key={game.id} game={game} />)}{!loading && games.length === 0 && <div className="empty">No live games match this week or search.</div>}</section></>
      }
      {tab === 'Team ratings' && <Ratings data={data} />}
      {tab === 'Point-in-time validation' && <Validation data={data} />}
      {tab === 'Method & sources' && <Method data={data} />}
    </div>
  </main>
}

function LoadingCards() { return <><div className="game-card loading-card" aria-label="Loading forecast"><span /><span /><span /></div><div className="game-card loading-card" aria-hidden="true"><span /><span /><span /></div></> }

function GameCard({ game }: { game: Game }) { return <article className="game-card"><div className="game-meta"><span>{dateLabel(game.date)}</span><span className="status track">{game.venue}</span></div><div className="matchup"><div><b>{game.away.rank ? `#${game.away.rank} ` : ''}{ game.away.name}</b><small>AWAY · {percent(game.awayWin)}</small></div><span className="at">at</span><div className="home"><b>{game.home.rank ? `#${game.home.rank} ` : ''}{game.home.name}</b><small>HOME · {percent(game.homeWin)}</small></div><strong className="win">{margin(game.margin)}</strong></div><div className="market-row"><span>MODEL SCORE <b>{game.homeExpectedPoints.toFixed(1)}–{game.awayExpectedPoints.toFixed(1)}</b></span><span>WINNER <b>{game.homeWin >= .5 ? game.home.name : game.away.name}</b></span><span>SPREAD <b>{margin(game.spread)}</b></span><span>TOTAL <b>{game.marketTotal == null ? '—' : game.marketTotal.toFixed(1)}</b></span></div><div className="prob-bar"><i style={{ width: `${game.homeWin * 100}%` }} /><i style={{ width: `${game.awayWin * 100}%` }} /></div><div className="game-footnote">Home win {percent(game.homeWin)} · away win {percent(game.awayWin)} · cutoff {new Date(game.cutoff).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' })}</div></article> }

function Ratings({ data }: { data: Data | null }) { return <section className="panel table-panel"><div className="panel-head"><div><p className="eyebrow">OPPONENT-ADJUSTED STRENGTH</p><h2>Team ratings</h2></div><span className="muted mono">NET = OFFENSE − DEFENSE</span></div><table><thead><tr><th>TEAM</th><th>GAMES</th><th>OFFENSE</th><th>DEFENSE</th><th>NET RATING</th></tr></thead><tbody>{(data?.ratings ?? []).map((team) => <tr key={team.id}><td><b>{team.name ?? team.abbreviation}</b></td><td>{Math.floor(team.games)}</td><td className="positive">{margin(team.offense)}</td><td>{margin(team.defense)}</td><td><strong>{margin(team.net)}</strong></td></tr>)}</tbody></table></section> }

function Validation({ data }: { data: Data | null }) { const result = data?.backtest; return <section className="validation-grid"><div className="panel"><p className="eyebrow">WALK-FORWARD RESULTS</p><h2>Predictions scored against finals</h2><div className="compare"><div><span>Winner accuracy</span><b>{result ? percent(result.accuracy) : '—'}</b></div><div><span>Winner log loss</span><b>{result?.logLoss.toFixed(4) ?? '—'}</b></div><div><span>Spread MAE</span><b>{result?.spreadMae.toFixed(1) ?? '—'}</b></div></div><p className="muted validation-note">{result?.games ?? 0} completed games scored chronologically. Accuracy is descriptive; it is not a guarantee of future outcomes.</p></div><div className="panel table-panel"><div className="panel-head"><h2>Recent scored games</h2><span className="muted mono">ACTUAL VS MODEL</span></div><table><thead><tr><th>WEEK</th><th>GAME</th><th>ACTUAL</th><th>MODEL</th><th>SPREAD</th><th>WINNER</th></tr></thead><tbody>{(result?.rows ?? []).map((row, index) => <tr key={`${row.date}-${index}`}><td>{row.week}</td><td><b>{row.game}</b></td><td>{margin(row.actualMargin)}</td><td>{margin(row.predictedMargin)}</td><td>{margin(row.spread)}</td><td className={row.winner ? 'positive' : 'negative'}>{row.winner ? 'CORRECT' : 'INCORRECT'}</td></tr>)}</tbody></table></div></section> }

function Method({ data }: { data: Data | null }) { return <section className="panel method-long"><p className="eyebrow">MODEL & PROVENANCE</p><h2>How each forecast is made</h2><div className="method-list"><p><b>Strength state.</b> Completed FBS games are processed in time order. Opponent-adjusted offense and defense update after each final score, with a home-field term and shrinking updates as teams accumulate games.</p><p><b>Outcome layer.</b> Expected margin is converted to a win probability. Expected points are split into team score means, then checked with a discrete Poisson score layer.</p><p><b>Market layer.</b> A recorded spread or total is shown only when the live source reports one. The engine does not backfill lines, weather, players, or scores.</p><p><b>Validation.</b> Historical results are evaluated using a point-in-time cutoff so later results cannot enter an earlier forecast. The table reports accuracy, log loss, and margin error separately.</p><p><b>Sources.</b> {data?.sources?.map((source) => `${source.name} (${source.status})`).join(' · ') || 'Live source status will appear after the first successful refresh.'}</p></div></section> }
