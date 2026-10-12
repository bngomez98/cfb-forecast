"use client"

import { useMemo, useState } from "react"

type Game = { away: string; home: string; kickoff: string; spread: string; total: string; win: number; edge: string; status: "EDGE" | "TRACK" | "PASS"; color: string }

const games: Game[] = [
  { away: "Northwestern", home: "Indiana", kickoff: "FRI · 8:00 PM", spread: "IND -20.5", total: "49.5", win: 0.91, edge: "+3.8%", status: "EDGE", color: "green" },
  { away: "Texas", home: "Tennessee", kickoff: "SAT · 12:00 PM", spread: "TEX -4.0", total: "—", win: 0.64, edge: "+1.6%", status: "TRACK", color: "cyan" },
  { away: "Wisconsin", home: "Penn State", kickoff: "SAT · 5:00 PM", spread: "PSU -10.0", total: "44.5", win: 0.78, edge: "+2.4%", status: "EDGE", color: "green" },
  { away: "Central Michigan", home: "Miami", kickoff: "SAT · 6:30 PM", spread: "MIA -41.5", total: "54.5", win: 0.99, edge: "—", status: "PASS", color: "muted" },
]

const ratings = [
  ["Ohio State", "+24.8", "+18.1", "43.2"], ["Miami", "+20.7", "+16.4", "37.1"], ["Indiana", "+19.9", "+13.8", "33.7"], ["Oregon", "+18.6", "+14.1", "32.7"], ["Texas", "+17.2", "+15.0", "32.2"], ["Penn State", "+14.8", "+11.9", "28.1"],
]

function Metric({ label, value, note, tone = "cyan" }: { label: string; value: string; note: string; tone?: string }) {
  return <div className={`metric ${tone}`}><span>{label}</span><strong>{value}</strong><small>{note}</small></div>
}

function GameCard({ game, onOpen }: { game: Game; onOpen: (game: Game) => void }) {
  return <article className={`game-card ${game.status === "EDGE" ? "has-edge" : ""}`}>
    <div className="game-meta"><span>{game.kickoff}</span><span className={`status ${game.status.toLowerCase()}`}>{game.status}</span></div>
    <div className="matchup"><div><b>{game.away}</b><small>AWAY</small></div><span className="at">at</span><div className="home"><b>{game.home}</b><small>HOME</small></div><strong className="win">{Math.round(game.win * 100)}%</strong></div>
    <div className="market-row"><span>SPREAD <b>{game.spread}</b></span><span>TOTAL <b>{game.total}</b></span><span className="edge-value">MODEL EDGE <b>{game.edge}</b></span></div>
    <div className="prob-bar"><i style={{ width: `${game.win * 100}%` }} /><i style={{ width: `${(1 - game.win) * 100}%` }} /></div>
    <button className="text-button" onClick={() => onOpen(game)}>Open forecast <span>→</span></button>
  </article>
}

export default function Home() {
  const [tab, setTab] = useState("Live Board")
  const [selected, setSelected] = useState<Game | null>(null)
  const [query, setQuery] = useState("")
  const filtered = useMemo(() => games.filter((g) => `${g.away} ${g.home}`.toLowerCase().includes(query.toLowerCase())), [query])
  const tabs = ["Live Board", "Forecasts", "Line Movement", "Historical Validation", "Ratings", "Methodology"]

  return <main className="terminal-shell">
    <header className="topbar"><div className="brand"><div className="brand-mark">PI</div><div><strong>CFB-PIM FBS TERMINAL</strong><small>REAL-TIME · HISTORICAL · FORECAST</small></div></div><div className="divider" /><select aria-label="Season"><option>2026 season</option></select><select aria-label="Week"><option>Week 4 · 12 games</option></select><span className="live-chip"><i /> LIVE · ESPN</span><div className="top-spacer" /><div className="header-kpi"><b>0.5764</b><span>BEST HIST LL</span></div><div className="header-kpi"><b>3</b><span>QUALIFYING EDGES</span></div><button className="refresh">↻ Refresh feeds</button></header>
    <nav className="tabs" aria-label="Primary navigation">{tabs.map((item) => <button key={item} className={tab === item ? "active" : ""} onClick={() => setTab(item)}>{item}{item === "Live Board" && <em>12</em>}{item === "Historical Validation" && <em>3,136</em>}</button>)}</nav>
    <div className="content">
      <section className="hero-row"><div><p className="eyebrow">{tab === "Live Board" ? "WEEK 04 / 2026 SEASON" : "RESEARCH WORKSPACE"}</p><h1>{tab === "Live Board" ? "Forecast board" : tab}</h1><p className="lede">Calibrated probabilities first. Market decisions second. Every number carries its source and cutoff.</p></div><div className="pipeline"><span className="complete">DATA</span><i>→</i><span className="complete">FEATURES</span><i>→</i><span className="complete">RATINGS</span><i>→</i><span className="current">FORECAST</span><i>→</i><span>BACKTEST</span></div></section>
      <section className="metrics"><Metric label="SLATE" value="12" note="4 priced · 8 tracked" /><Metric label="FORECASTS BUILT" value="12" note="joint distributions" tone="violet" /><Metric label="QUALIFYING EDGES" value="3" note="τ 2.0% · EV > 0" tone="green" /><Metric label="GAMES FITTED" value="808" note="real 2026 outcomes" tone="amber" /><Metric label="DATA CUTOFF" value="09.25" note="2026 · 02:00 UTC" tone="cyan" /></section>
      {tab === "Live Board" || tab === "Forecasts" ? <>
        <section className="toolbar"><div className="search"><span>⌕</span><input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Filter team, conference, venue…" /></div><div className="segmented"><button className="selected">All</button><button>Upcoming</button><button>Final</button></div><div className="toolbar-spacer" /><span className="muted mono">{filtered.length} SHOWN · RESEARCH MODE</span></section>
        <section className="section-head"><div><p className="eyebrow">MARKET-AWARE FORECASTS</p><h2>Today's slate <span>·</span> <b>3 edges worth investigating</b></h2></div><button className="filter-button">Sort: model edge⌄</button></section>
        <section className="games-grid">{filtered.map((game) => <GameCard game={game} key={game.home} onOpen={setSelected} />)}</section>
      </> : tab === "Ratings" ? <Ratings /> : <Validation />}
      <section className="bottom-grid"><div className="panel"><div className="panel-head"><div><p className="eyebrow">MODEL HEALTH</p><h2>Validation snapshot</h2></div><span className="badge green-badge">OUT-OF-SAMPLE</span></div><div className="snapshot-grid"><div><strong>0.5406</strong><span>LOG LOSS</span><small>2025 held-out</small></div><div><strong>73.02%</strong><span>WIN ACCURACY</span><small>590 / 808 games</small></div><div><strong>0.1830</strong><span>BRIER SCORE</span><small>lower is better</small></div></div><div className="chart"><div className="chart-line" /><div className="chart-labels"><span>2021</span><span>2022</span><span>2023</span><span>2024</span><span>2025</span></div></div></div><div className="panel method-card"><p className="eyebrow">CURRENT STACK</p><h2>Evidence over complexity</h2><p>Dynamic strength is the current production layer. PBP and matchup interactions remain experimental until they beat the baseline in a clean walk-forward test.</p><div className="method-row"><span>Best historical model</span><b>Dynamic strength</b></div><div className="method-row"><span>Primary metric</span><b>Log loss</b></div><button className="outline-button" onClick={() => setTab("Methodology")}>Read methodology →</button></div></section>
    </div>
    {selected && <div className="drawer-backdrop" onClick={() => setSelected(null)}><aside className="drawer" onClick={(e) => e.stopPropagation()}><div className="drawer-head"><span>GAME FORECAST</span><button onClick={() => setSelected(null)}>×</button></div><div className="drawer-body"><p className="eyebrow">WEEK 04 · MARKET-AWARE</p><h2>{selected.home} <span>vs</span> {selected.away}</h2><div className="drawer-prob"><strong>{Math.round(selected.win * 100)}%</strong><span>MODEL HOME WIN PROBABILITY</span></div><div className="drawer-stats"><Metric label="PROJECTED MARGIN" value={selected.home === "Indiana" ? "+18.4" : "+6.2"} note="σ 13.6 points" tone="violet" /><Metric label="PROJECTED TOTAL" value={selected.total === "—" ? "52.8" : selected.total} note="σ 13.2 points" tone="amber" /></div><div className="note"><b>Model read</b><p>Signal is strongest when the dynamic rating and market layer agree. This forecast is logged before kickoff and will be scored after the game.</p></div><button className="outline-button full" onClick={() => setSelected(null)}>Close forecast</button></div></aside></div>}
  </main>
}

function Ratings() { return <section className="panel table-panel"><div className="panel-head"><div><p className="eyebrow">DYNAMIC TEAM STATES</p><h2>Power ratings</h2></div><span className="muted mono">PRE-GAME · SHRUNK TOWARD PRIOR</span></div><table><thead><tr><th>#</th><th>TEAM</th><th>OFFENSE</th><th>DEFENSE</th><th>NET RATING</th></tr></thead><tbody>{ratings.map((r, i) => <tr key={r[0]}><td>{String(i + 1).padStart(2, "0")}</td><td><b>{r[0]}</b></td><td className="positive">{r[1]}</td><td>{r[2]}</td><td className="positive">{r[3]}</td></tr>)}</tbody></table></section> }
function Validation() { return <section className="validation-grid"><div className="panel"><p className="eyebrow">2021–2025 WALK-FORWARD</p><h2>Model comparison</h2><div className="compare"><div><span>Dynamic strength</span><b>0.5764</b><i style={{ width: "83%" }} /></div><div><span>PBP extension</span><b>0.5842</b><i style={{ width: "78%" }} /></div><div><span>Calibrated Elo</span><b>0.5959</b><i style={{ width: "70%" }} /></div></div></div><div className="panel"><p className="eyebrow">HONEST LIMITS</p><h2>What this proves</h2><p className="lede">The model has forecast skill versus a coin-flip baseline. It does not establish that it beats closing lines or produces a profitable betting edge.</p><div className="note"><b>Next gate</b><p>Prospective holdout with timestamped market snapshots and calibration at the 60–80% bins.</p></div></div></section> }
