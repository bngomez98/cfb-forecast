export type Team = { id: string; name: string; abbreviation: string; score: number | null; rank: number | null }
export type ForecastGame = { id: string; date: string; week: number; status: string; home: Team; away: Team; venue: string; spread: number | null; marketTotal: number | null; source: string }
export type Forecast = ForecastGame & { homeWin: number; awayWin: number; margin: number; total: number; homeExpectedPoints: number; awayExpectedPoints: number; marginSd: number; totalSd: number; over: number | null; cover: number | null; kelly: number | null; poissonHomeWin: number; model: string; cutoff: string }

type Rating = { offense: number; defense: number; games: number }
const clamp = (x: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, x))
const logistic = (x: number) => 1 / (1 + Math.exp(-x))
const erf = (x: number) => { const sign = x < 0 ? -1 : 1; x = Math.abs(x); const t = 1 / (1 + 0.3275911 * x); const y = 1 - (((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t) * Math.exp(-x * x); return sign * y }
const normalCdf = (x: number) => 0.5 * (1 + erf(x / Math.sqrt(2)))

export function americanToProb(odds: number | null) {
  if (odds == null) return null
  return odds < 0 ? -odds / (-odds + 100) : 100 / (odds + 100)
}

export function fitDynamicRatings(games: ForecastGame[]) {
  const ratings = new Map<string, Rating>()
  const get = (id: string) => { if (!ratings.has(id)) ratings.set(id, { offense: 0, defense: 0, games: 0 }); return ratings.get(id)! }
  const completed = games.filter(g => g.status === "post" && g.home.score != null && g.away.score != null)
  for (const game of completed) { get(game.home.id); get(game.away.id) }
  const mean = completed.length ? completed.flatMap(g => [g.home.score!, g.away.score!]).reduce((a, b) => a + b, 0) / (completed.length * 2) : 28
  for (let pass = 0; pass < 35; pass++) for (const game of completed) {
    const h = get(game.home.id), a = get(game.away.id), hfa = 2.4
    const hExpected = mean + h.offense - a.defense + hfa
    const aExpected = mean + a.offense - h.defense
    const hResidual = game.home.score! - hExpected, aResidual = game.away.score! - aExpected
    const shrinkH = 0.18 / (1 + h.games * 0.08), shrinkA = 0.18 / (1 + a.games * 0.08)
    h.offense += hResidual * shrinkH; h.defense -= aResidual * shrinkH; a.offense += aResidual * shrinkA; a.defense -= hResidual * shrinkA
    h.games += 0.02; a.games += 0.02
  }
  return { ratings, mean, games: completed.length }
}

export function forecastGame(game: ForecastGame, fitted: ReturnType<typeof fitDynamicRatings>, cutoff: string): Forecast {
  const home = fitted.ratings.get(game.home.id) ?? { offense: 0, defense: 0, games: 0 }
  const away = fitted.ratings.get(game.away.id) ?? { offense: 0, defense: 0, games: 0 }
  const margin = clamp(2.4 + home.offense - away.defense - (away.offense - home.defense), -50, 50)
  const total = clamp(2 * fitted.mean + home.offense + away.offense - home.defense - away.defense, 20, 90)
  const uncertainty = 1 + 0.35 / Math.sqrt(1 + home.games + away.games)
  const marginSd = 13.6 * uncertainty, totalSd = 12.8 * uncertainty
  const homeExpectedPoints = clamp((total + margin) / 2, 1, 80)
  const awayExpectedPoints = clamp((total - margin) / 2, 1, 80)
  const marketHome = game.spread == null ? null : logistic(-game.spread / 6.5)
  const modelHome = logistic(margin / 6.5)
  const homeWin = marketHome == null ? modelHome : logistic(0.55 * (margin / 6.5) + 0.45 * Math.log(clamp(marketHome, .01, .99) / clamp(1 - marketHome, .01, .99)))
  const over = game.marketTotal == null ? null : 1 - normalCdf((game.marketTotal - total) / totalSd)
  const cover = game.spread == null ? null : 1 - normalCdf((-game.spread - margin) / marginSd)
  const poissonHomeWin = poissonWinProbability(homeExpectedPoints, awayExpectedPoints)
  const kelly = game.spread == null || cover == null ? null : kellyFraction(cover, 1.909)
  return { ...game, homeWin, awayWin: 1 - homeWin, margin, total, homeExpectedPoints, awayExpectedPoints, marginSd, totalSd, over, cover, kelly, poissonHomeWin, model: "Bayesian dynamic strength + Gaussian/Poisson score layer", cutoff }
}

export function poissonPmf(lambda: number, k: number) { let p = Math.exp(-lambda); for (let i = 1; i <= k; i++) p *= lambda / i; return p }
export function poissonWinProbability(homeLambda: number, awayLambda: number) {
  let home = 0
  for (let h = 0; h <= 80; h++) for (let a = 0; a < h; a++) home += poissonPmf(homeLambda, h) * poissonPmf(awayLambda, a)
  return clamp(home, 0.001, 0.999)
}
export function kellyFraction(probability: number, decimalOdds: number) { const b = decimalOdds - 1; return b <= 0 ? 0 : Math.max(0, (probability * decimalOdds - 1) / b) }
export function summarizeRatings(fitted: ReturnType<typeof fitDynamicRatings>) { return [...fitted.ratings.entries()].map(([id, r]) => ({ id, ...r, net: r.offense - r.defense })).sort((a, b) => b.net - a.net) }
export function backtestSummary() { return { games: 808, accuracy: 0.7302, logLoss: 0.5406, brier: 0.183, model: "PBP-selected / dynamic-strength production benchmark", caveat: "Historical benchmark from RESULTS.md; live ratings are refit only from completed source games." } }
export function buildForecasts(games: ForecastGame[], cutoff: string) { const fitted = fitDynamicRatings(games); return { forecasts: games.map(g => forecastGame(g, fitted, cutoff)), ratings: summarizeRatings(fitted), fittedGames: fitted.games, backtest: backtestSummary() } }
