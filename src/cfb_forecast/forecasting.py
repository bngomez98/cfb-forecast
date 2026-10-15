"""Forecasts from the joint model, their storage, and their evaluation.

A forecast is the posterior-predictive distribution of one game. For every joint posterior
draw of the states and parameters (teams, units, players, availability, matchup, possession
model) the exact score distribution of the game is computed from the possession model; the
forecast is the average over draws, which is the integral over parameter and state uncertainty.
"""
from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import nbinom

from . import football_model as fm
from . import model_spec as sp

L = 121                    # points 0..120 per team
M_FFT = 256
NMAX = 36                  # most possessions per team considered
GLOBAL_SITES = ["alpha", "eta", "loading", "gamma", "W_raw", "tau_W", "fg_distance_coef", "outcome_intercept", "outcome_coef",
                "outcome_field_position", "outcome_home", "outcome_covariates", "td_points_logits", "rare_off_logits",
                "rare_opp_logits", "drives_intercept", "drives_tempo", "drives_sustain", "drives_concentration", "sigma"] + \
               [f"{p}_{g}" for g in sp.PLAYER_GROUPS for p in ("part_intercept", "part_sd", "part_prior_usage", "part_node")]


def now_iso():
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Bundle:
    """Posterior draws needed to forecast, plus the identifiers to interpret them."""
    parameter_set_id: str
    draws: dict
    meta: dict

    def save(self, folder) -> Path:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(folder / f"{self.parameter_set_id}.npz", **self.draws)
        (folder / f"{self.parameter_set_id}.meta.pkl").write_bytes(pickle.dumps(self.meta))
        return folder / f"{self.parameter_set_id}.npz"

    @staticmethod
    def load(folder, parameter_set_id) -> "Bundle":
        folder = Path(folder)
        z = np.load(folder / f"{parameter_set_id}.npz")
        return Bundle(parameter_set_id, {k: z[k] for k in z.files}, pickle.loads((folder / f"{parameter_set_id}.meta.pkl").read_bytes()))


def extract_bundle(f: "fm.Fit", n_draws=200, batch=20, seed=7, calculation_version="1") -> Bundle:
    """Posterior-predictive draws of states (at the kept periods) and global parameters."""
    import jax
    from numpyro.infer import Predictive
    sites = GLOBAL_SITES + ["S_keep"] + [f"pstate_{g}" for g in sp.PLAYER_GROUPS]
    chunks = []
    for b, start in enumerate(range(0, n_draws, batch)):
        pred = Predictive(fm.model, guide=f.guide, params=f.params, num_samples=min(batch, n_draws - start), return_sites=sites)
        out = pred(jax.random.PRNGKey(seed + b), f.A)
        chunks.append({k: np.asarray(v) for k, v in out.items() if k in sites})
    draws = {k: np.concatenate([c[k] for c in chunks], axis=0) for k in chunks[0]}
    d = f.data
    meta = dict(
        seasons=d.seasons, team_ids=d.team_ids, team_index=d.team_index, cutoff=d.cutoff, cov_names=d.cov_names, cov_stats=d.cov_stats,
        keep_periods=[int(x) for x in np.asarray(f.A["keep_periods"])],
        nodes={g: dict(player_id=d.nodes[g]["player_id"], y=d.nodes[g]["y"], team=d.nodes[g]["team"], cov=d.nodes[g]["cov"],
                       n_obs=d.nodes[g]["n_obs"]) for g in d.nodes},
        fp_quantiles=np.quantile(d.drive_fp, [0.1, 0.3, 0.5, 0.7, 0.9]), stream_init=d.stream_init,
        unit=np.asarray(f.A["unit"]), calculation_version=calculation_version,
    )
    digest = hashlib.sha256(json.dumps([d.seasons, str(d.cutoff), len(d.obs_gd), float(f.losses[-1]), calculation_version], default=str).encode()).hexdigest()[:16]
    return Bundle(f"ps-{digest}", draws, meta)


# ------------------------------------------------------------------------ database storage

def store_estimates(conn, bundle: Bundle, source_manifest=None, forecast_games=None) -> None:
    """Persist the parameter set, state estimates and availability estimates (posterior means and sds)."""
    ps, dr, meta = bundle.parameter_set_id, bundle.draws, bundle.meta
    conn.execute("DELETE FROM statistical_parameter_sets WHERE parameter_set_id=?", (ps,))
    conn.execute("INSERT INTO statistical_parameter_sets(parameter_set_id,calculation_version,data_cutoff,source_manifest_json,created_at) VALUES(?,?,?,?,?)",
                 (ps, meta["calculation_version"], str(meta["cutoff"]), json.dumps(source_manifest or {}, default=str), now_iso()))
    rows = []
    for name in GLOBAL_SITES:
        if name not in dr or name.startswith("part_node"):
            continue
        arr = dr[name]
        mean, sd = arr.mean(0), arr.std(0)
        for idx in np.ndindex(*mean.shape) if mean.ndim else [()]:
            tag = name + ("[" + ",".join(map(str, idx)) + "]" if idx else "")
            rows.append((ps, tag, float(mean[idx]), float(sd[idx])))
    conn.executemany("INSERT OR REPLACE INTO statistical_parameter_values(parameter_set_id,name,estimated_value,uncertainty) VALUES(?,?,?,?)", rows)

    as_of = str(meta["cutoff"] or now_iso())
    S = dr["S_keep"][:, :, 0, :]                       # (draws, teams, K) at the first kept period
    mean, sd = S.mean(0), S.std(0)
    teams = meta["team_ids"]
    team_rows, unit_rows = [], []
    for g, comps in sp.TEAM_GROUPS.items():
        for k, comp in enumerate(comps):
            c = sp.TEAM_OFFSETS[g] + k
            for t, team_id in enumerate(teams):
                rec = (ps, int(team_id), None, comp, as_of, float(mean[t, c]), float(sd[t, c]))
                if g in ("OFF", "DEF"):
                    team_rows.append((rec[0], rec[1], "offense" if g == "OFF" else "defense", *rec[3:]))
                else:
                    unit_rows.append((rec[0], rec[1], g.lower(), *rec[3:]))
    conn.execute("DELETE FROM team_state_estimates WHERE parameter_set_id=?", (ps,))
    conn.execute("DELETE FROM unit_state_estimates WHERE parameter_set_id=?", (ps,))
    conn.execute("DELETE FROM player_state_estimates WHERE parameter_set_id=?", (ps,))
    conn.executemany("INSERT OR REPLACE INTO team_state_estimates(parameter_set_id,team_id,side,component,as_of,mean,sd) VALUES(?,?,?,?,?,?,?)", team_rows)
    conn.executemany("INSERT OR REPLACE INTO unit_state_estimates(parameter_set_id,team_id,unit,component,as_of,mean,sd) VALUES(?,?,?,?,?,?,?)", unit_rows)
    prow = []
    for g, comps in sp.PLAYER_GROUPS.items():
        ps_g = dr[f"pstate_{g}"]
        m_, s_ = ps_g.mean(0), ps_g.std(0)
        nd = meta["nodes"][g]
        latest = np.flatnonzero(nd["y"] == len(meta["seasons"]) - 1)
        for n in latest:
            for k, comp in enumerate(comps):
                prow.append((ps, str(nd["player_id"][n]), g, comp, as_of, float(m_[n, k]), float(s_[n, k])))
    conn.executemany("INSERT OR REPLACE INTO player_state_estimates(parameter_set_id,player_id,position_group,component,as_of,mean,sd) VALUES(?,?,?,?,?,?,?)", prow)
    conn.commit()


# ------------------------------------------------------------------------ score distribution

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def _pmf_sets(p_out, td_pmf, rare_off, rare_opp):
    """Per-possession point PMFs (draws, M_FFT) for the offense and the credit its defense gets."""
    n = p_out.shape[0]
    off = np.zeros((n, M_FFT))
    for k, pts in enumerate(sp.TD_POINTS):
        off[:, pts] += p_out[:, 1] * td_pmf[:, k]
    off[:, 3] += p_out[:, 2]
    for k, pts in enumerate(sp.RARE_POINTS):
        off[:, pts] += p_out[:, 7] * rare_off[:, k]
    off[:, 0] += 1.0 - off.sum(axis=1)
    credit = np.zeros((n, M_FFT))
    for k, pts in enumerate(sp.RARE_POINTS):
        credit[:, pts] += p_out[:, 7] * rare_opp[:, k]
    credit[:, 0] += 1.0 - credit.sum(axis=1)
    return off, credit


def _powers(pmf):
    """rfft(pmf)^n for n = 0..NMAX; shape (draws, NMAX+1, M_FFT//2+1)."""
    F = np.fft.rfft(pmf, axis=-1)
    out = np.ones((F.shape[0], NMAX + 1, F.shape[1]), complex)
    for n in range(1, NMAX + 1):
        out[:, n] = out[:, n - 1] * F
    return out


def game_distribution(p_home, p_away, td_pmf, rare_off, rare_opp, lam, phi, rng=None):
    """Exact joint score distribution for each draw, averaged. p_*: (draws, 8) outcome probabilities."""
    n = p_home.shape[0]
    off_h, cred_h = _pmf_sets(p_home, td_pmf, rare_off, rare_opp)
    off_a, cred_a = _pmf_sets(p_away, td_pmf, rare_off, rare_opp)
    Fh, Fa = _powers(off_h), _powers(off_a)
    Gh, Ga = _powers(cred_a), _powers(cred_h)       # home gets credit from away possessions, and vice versa
    # possession-count pairs: total T ~ NegBin, the extra possession of an odd total is a coin toss
    T = np.arange(0, 2 * NMAX)
    pT = nbinom.pmf(T[None, :], phi[:, None], phi[:, None] / (phi[:, None] + lam[:, None]))
    pairs, weights = [], []
    for t in T:
        if t % 2 == 0:
            pairs.append((t // 2, t // 2)); weights.append(pT[:, t])
        else:
            pairs.append(((t + 1) // 2, (t - 1) // 2)); weights.append(0.5 * pT[:, t])
            pairs.append(((t - 1) // 2, (t + 1) // 2)); weights.append(0.5 * pT[:, t])
    pairs = np.array(pairs)
    keep = pairs.max(axis=1) <= NMAX
    pairs, W = pairs[keep], np.stack(weights, axis=1)[:, keep]
    W = W / W.sum(axis=1, keepdims=True)
    J = np.zeros((L, L))
    for d in range(n):
        nh, na = pairs[:, 0], pairs[:, 1]
        H = np.fft.irfft(Fh[d, nh] * Gh[d, na], n=M_FFT, axis=-1)[:, :L].clip(0)
        A = np.fft.irfft(Fa[d, na] * Ga[d, nh], n=M_FFT, axis=-1)[:, :L].clip(0)
        J += (W[d][:, None] * H).T @ A
    J /= n
    return J / J.sum()


def summarize(J):
    pts = np.arange(L)
    home = J.sum(1)
    away = J.sum(0)
    diff = pts[:, None] - pts[None, :]
    margin = np.zeros(2 * L - 1)
    np.add.at(margin, (diff + L - 1).ravel(), J.ravel())
    total = np.zeros(2 * L - 1)
    np.add.at(total, (pts[:, None] + pts[None, :]).ravel(), J.ravel())
    p_tie = np.trace(J)
    p_home = J[np.tril_indices(L, -1)].sum()
    p_away = J[np.triu_indices(L, 1)].sum()
    win = p_home / (p_home + p_away)                 # overtime resolves ties in proportion to strength
    m_axis = np.arange(-(L - 1), L)
    e_margin = float((margin * m_axis).sum())
    e_total = float((total[: 2 * L - 1] * np.arange(2 * L - 1)).sum())
    return dict(home_points=float((home * pts).sum()), away_points=float((away * pts).sum()), home_win=float(win), away_win=float(1 - win),
                margin=margin, total=total, expected_margin=e_margin, margin_variance=float((margin * (m_axis - e_margin) ** 2).sum()),
                expected_total=e_total, total_variance=float((total * (np.arange(2 * L - 1) - e_total) ** 2).sum()), tie_mass=float(p_tie))


def cover_probability(margin_pmf, spread):
    """P(home covers) for a home spread (negative = home favored); pushes are split evenly."""
    m = np.arange(-(L - 1), L)
    edge = m + spread
    return float(margin_pmf[edge > 0].sum() + 0.5 * margin_pmf[edge == 0].sum())


def over_probability(total_pmf, line):
    t = np.arange(len(total_pmf))
    return float(total_pmf[t > line].sum() + 0.5 * total_pmf[t == line].sum())


# ------------------------------------------------------------------------ lineup and game features

def recent_group_fractions(conn, team_id, before, n_games=6):
    """Share of a team's carries/targets recently going to each player group (team-level context)."""
    q = """SELECT p.is_rush, p.is_pass, p.sack, p.rusher_id, p.receiver_id, g.start_date FROM plays p JOIN games g ON g.game_id=p.game_id
           WHERE p.offense_team_id=? AND p.scrimmage_play=1 AND g.start_date < ? AND g.result_usable=1
           ORDER BY g.start_date DESC LIMIT 1200"""
    df = pd.read_sql_query(q, conn, params=[int(team_id), str(before)])
    pos = pd.read_sql_query("SELECT player_id, position FROM players", conn).set_index("player_id").position
    if df.empty:
        return dict(qb_carry=0.12, rb_carry=0.75, rec_target=0.72)
    rush = df[df.is_rush == 1]
    passes = df[(df.is_pass == 1) & (df.sack == 0)]
    g = rush.rusher_id.map(pos)
    return dict(qb_carry=float((g == "QB").mean()) if len(rush) else 0.12, rb_carry=float((g == "RB").mean()) if len(rush) else 0.75,
                rec_target=float((passes.receiver_id.map(pos) == "REC").mean()) if len(passes) else 0.72)


class Forecaster:
    def __init__(self, bundle: Bundle, conn=None):
        self.b, self.d, self.m, self.conn = bundle, bundle.draws, bundle.meta, conn
        self.n = self.d["S_keep"].shape[0]
        self.Wm = self.d["W_raw"] * self.d["tau_W"][:, None, None]
        self.rng = np.random.default_rng(11)

    # ---- lineup: who plays and how the opportunities are shared (availability and usage)
    def lineup(self, team_idx, fractions):
        w = {}
        shares = {}
        for g, comps in sp.PLAYER_GROUPS.items():
            nd = self.m["nodes"][g]
            cand = np.flatnonzero((nd["team"] == team_idx) & (nd["y"] == len(self.m["seasons"]) - 1))
            if len(cand) == 0:
                shares[g] = (cand, np.zeros((self.n, 0)))
                continue
            a, su = self.d[f"part_intercept_{g}"][:, None], self.d[f"part_sd_{g}"][:, None]
            bp = self.d[f"part_prior_usage_{g}"][:, None]
            u = self.d[f"part_node_{g}"][:, cand]
            p = _sigmoid(a + su * u + bp * nd["cov"][cand, 0][None, :])
            z = self.rng.random(p.shape) < p
            none = ~z.any(axis=1)
            if none.any():
                z[none, p[none].argmax(axis=1)] = True
            usage = self.d[f"pstate_{g}"][:, cand, comps.index("usage_capacity")]
            sh = _softmax(np.where(z, usage, -1e9), axis=1) * z
            shares[g] = (cand, sh)
        frac = fractions
        roles = {
            "dropback": ("QB", 1.0), "carry_qb": ("QB", frac["qb_carry"]), "carry_rb": ("RB", frac["rb_carry"]),
            "target": ("REC", frac["rec_target"]), "rb_target": ("RB", 1.0),
        }
        for role, (g, scale) in roles.items():
            cand, sh = shares[g]
            w[role] = (g, cand, sh * scale)
        return w, shares

    def linear(self, off_idx, def_idx, kp, off_lineup):
        """Link-scale means for every stream of one game direction, per draw: (draws, S) and matchup index."""
        Sk = self.d["S_keep"][:, :, kp, :]
        pad = np.zeros((self.n, Sk.shape[1], 1))
        Sk = np.concatenate([Sk, pad], axis=2)
        Oi, Dj = Sk[:, off_idx], Sk[:, def_idx]
        team = (Oi[:, fm.COL_I] * fm.SG_I).sum(-1) + (Dj[:, fm.COL_J] * fm.SG_J).sum(-1)
        lam = np.where(fm.HAS_LOADING, self.d["loading"], 1.0)
        lin = team * lam
        Mx = np.einsum("dk,dkl,dl->d", Oi[:, fm.O0:fm.O1], self.Wm, Dj[:, fm.D0:fm.D1])
        lin = lin + np.where(fm.HAS_MATCHUP, self.d["gamma"], 0.0) * Mx[:, None]
        for s, spec in enumerate(sp.STREAMS):
            for pg, comp, role in spec.players:
                g, cand, w = off_lineup[role]
                if g != pg or len(cand) == 0:
                    continue
                st = self.d[f"pstate_{pg}"][:, cand, sp.PLAYER_GROUPS[pg].index(comp)]
                lin[:, s] += (w * st).sum(axis=1)
        return lin, Mx

    def outcome_probs(self, lin, Mx, h, xg):
        d = self.d
        feats = np.concatenate([lin[:, fm.KEY], Mx[:, None]], axis=1)
        base = d["outcome_intercept"] + np.einsum("dk,dkc->dc", feats, d["outcome_coef"]) + h * d["outcome_home"]
        if xg is not None and len(xg):
            base = base + np.einsum("k,dkc->dc", np.asarray(xg), d["outcome_covariates"][:, : len(xg)])
        probs = 0
        qs = self.m["fp_quantiles"]
        for q in qs:
            logit = np.concatenate([np.zeros((self.n, 1)), base + q * d["outcome_field_position"]], axis=1)
            probs = probs + _softmax(logit)
        return probs / len(qs)

    def forecast_game(self, home_idx, away_idx, kp, neutral, xg_home, xg_away, home_fr, away_fr):
        lh, _ = self.lineup(home_idx, home_fr)
        la, _ = self.lineup(away_idx, away_fr)
        lin_h, M_h = self.linear(home_idx, away_idx, kp, lh)       # home offense vs away defense
        lin_a, M_a = self.linear(away_idx, home_idx, kp, la)
        hs = 0.0 if neutral else 1.0
        p_h = self.outcome_probs(lin_h, M_h, hs, xg_home)
        p_a = self.outcome_probs(lin_a, M_a, -hs, xg_away)
        d = self.d
        td_pmf = _softmax(d["td_points_logits"])
        rare_off, rare_opp = _softmax(d["rare_off_logits"]), _softmax(d["rare_opp_logits"])
        ti, fi = sp.STREAM_INDEX["tempo_plays"], sp.STREAM_INDEX["first_down_rate"]
        lam = np.exp(d["drives_intercept"] + d["drives_tempo"] * (lin_h[:, ti] + lin_a[:, ti]) + d["drives_sustain"] * (lin_h[:, fi] + lin_a[:, fi]))
        J = game_distribution(p_h, p_a, td_pmf, rare_off, rare_opp, lam, d["drives_concentration"])
        return J, dict(outcome_home=p_h.mean(0), outcome_away=p_a.mean(0), possessions=float(lam.mean()))


# ------------------------------------------------------------------------ run + store forecasts

def game_covariates(conn, game_id, meta):
    """Standardized schedule/weather covariates for both directions of a game, in training order."""
    names, stats = meta["cov_names"], meta["cov_stats"]
    if not names:
        return None, None
    sc = pd.read_sql_query("SELECT * FROM schedule_conditions WHERE game_id=?", conn, params=[int(game_id)])
    if sc.empty:
        z = np.zeros(len(names))
        return z, z
    r = sc.iloc[0]
    raw = {"rest_diff": (r.home_days_rest, r.away_days_rest), "short_week_diff": (r.home_short_week, r.away_short_week),
           "bye_diff": (r.home_bye_week, r.away_bye_week), "travel_diff": (r.home_travel_distance, r.away_travel_distance)}
    home, away = [], []
    for n in names:
        if n in raw and not any(pd.isna(x) for x in raw[n]):
            a, b = float(raw[n][0]), float(raw[n][1])
            mean, sd = stats[n]
            home.append((a - b - mean) / sd)
            away.append((b - a - mean) / sd)
        else:
            home.append(0.0); away.append(0.0)
    return np.array(home), np.array(away)


def forecast_games(conn, bundle: Bundle, game_ids, artifacts, run_id=None, information_cutoff=None, forecaster=None):
    """Forecast the games, store forecasts/inputs/distribution artifact, and return a table of results."""
    fc = forecaster or Forecaster(bundle, conn)
    meta = bundle.meta
    games = pd.read_sql_query(f"SELECT * FROM games WHERE game_id IN ({','.join('?' * len(game_ids))})", conn, params=[int(g) for g in game_ids])
    run_id = run_id or f"run-{bundle.parameter_set_id}-{datetime.now(timezone.utc):%Y%m%dT%H%M%S}"
    cutoff = information_cutoff or str(meta["cutoff"] or now_iso())
    created = now_iso()
    conn.execute("INSERT OR REPLACE INTO forecast_runs(run_id,created_at,information_cutoff,parameter_set_id,status,manifest_json) VALUES(?,?,?,?,?,?)",
                 (run_id, created, cutoff, bundle.parameter_set_id, "completed", json.dumps({"games": len(games)})))
    dist_dir = Path(artifacts) / "forecast_distributions"
    dist_dir.mkdir(parents=True, exist_ok=True)
    out, Js, ids = [], [], []
    per_of_week = lambda season, week: meta["seasons"].index(int(season)) * sp.WEEKS + min(max(int(week), 1), sp.WEEKS) - 1
    for g in games.itertuples(index=False):
        hi, ai = meta["team_index"].get(int(g.home_id)), meta["team_index"].get(int(g.away_id))
        if hi is None or ai is None or int(g.season) not in meta["seasons"]:
            continue
        per = per_of_week(g.season, g.week)
        if per not in meta["keep_periods"]:
            raise ValueError(f"Period {per} of game {g.game_id} is not among the posterior periods kept at fit time {meta['keep_periods']}")
        kp = meta["keep_periods"].index(per)
        xh, xa = game_covariates(conn, g.game_id, meta)
        fh = recent_group_fractions(conn, g.home_id, g.start_date)
        fa = recent_group_fractions(conn, g.away_id, g.start_date)
        J, info = fc.forecast_game(hi, ai, kp, bool(g.neutral_site), xh, xa, fh, fa)
        s = summarize(J)
        distribution_ref = f"{run_id}.npz#{len(Js)}"
        cur = conn.execute("""INSERT OR REPLACE INTO forecasts(run_id,game_id,creation_timestamp,information_cutoff,parameter_set_id,home_expected_points,
            away_expected_points,home_win_probability,away_win_probability,expected_margin,margin_variance,expected_total,total_variance,
            forecast_distribution_artifact,team_estimate_reference,player_estimate_reference,availability_reference,weather_reference,market_quote_reference)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                           (run_id, int(g.game_id), created, cutoff, bundle.parameter_set_id, s["home_points"], s["away_points"], s["home_win"], s["away_win"],
                            s["expected_margin"], s["margin_variance"], s["expected_total"], s["total_variance"], distribution_ref,
                            f"team_state_estimates:{bundle.parameter_set_id}", f"player_state_estimates:{bundle.parameter_set_id}",
                            f"availability_estimates:{bundle.parameter_set_id}", f"weather:{g.game_id}", f"market_quotes:{g.game_id}"))
        fid = cur.lastrowid
        conn.executemany("INSERT INTO forecast_inputs(forecast_id,input_type,reference_table,reference_key,payload_json) VALUES(?,?,?,?,?)", [
            (fid, "team_states", "team_state_estimates", f"{bundle.parameter_set_id}:{g.home_id},{g.away_id}", "{}"),
            (fid, "schedule_conditions", "schedule_conditions", str(g.game_id), json.dumps({"home": None if xh is None else xh.tolist(), "away": None if xa is None else xa.tolist()})),
            (fid, "group_fractions", None, None, json.dumps({"home": fh, "away": fa})),
            (fid, "possessions", None, None, json.dumps({"expected_drives_per_team": info["possessions"] / 2}))])
        Js.append(J.astype(np.float32)); ids.append(int(g.game_id))
        out.append(dict(game_id=int(g.game_id), forecast_id=fid, home_team=g.home_team, away_team=g.away_team, kickoff=g.start_date,
                        home_points=s["home_points"], away_points=s["away_points"], home_win=s["home_win"], expected_margin=s["expected_margin"],
                        expected_total=s["expected_total"], margin_sd=float(np.sqrt(s["margin_variance"])), total_sd=float(np.sqrt(s["total_variance"]))))
    if Js:
        np.savez_compressed(dist_dir / f"{run_id}.npz", joint=np.stack(Js), game_ids=np.array(ids))
    conn.commit()
    return pd.DataFrame(out)


def store_availability(conn, bundle: Bundle, forecaster: Forecaster, game_ids) -> int:
    """Participation probability and expected opportunity share of each candidate player, per upcoming game."""
    meta, rows = bundle.meta, []
    games = pd.read_sql_query(f"SELECT game_id, home_id, away_id FROM games WHERE game_id IN ({','.join('?' * len(game_ids))})", conn, params=[int(g) for g in game_ids])
    as_of = str(meta["cutoff"] or now_iso())
    for g in games.itertuples(index=False):
        for team_id in (g.home_id, g.away_id):
            ti = meta["team_index"].get(int(team_id))
            if ti is None:
                continue
            _, shares = forecaster.lineup(ti, dict(qb_carry=0.12, rb_carry=0.75, rec_target=0.72))
            for grp, (cand, sh) in shares.items():
                nd = meta["nodes"][grp]
                a, su = bundle.draws[f"part_intercept_{grp}"][:, None], bundle.draws[f"part_sd_{grp}"][:, None]
                bp = bundle.draws[f"part_prior_usage_{grp}"][:, None]
                p = _sigmoid(a + su * bundle.draws[f"part_node_{grp}"][:, cand] + bp * nd["cov"][cand, 0][None, :])
                for j, n in enumerate(cand):
                    rows.append((bundle.parameter_set_id, int(g.game_id), str(nd["player_id"][n]), float(p[:, j].mean()), float(sh[:, j].mean()), as_of))
    conn.executemany("INSERT OR REPLACE INTO availability_estimates(parameter_set_id,game_id,player_id,participation_probability,expected_usage_share,as_of) VALUES(?,?,?,?,?,?)", rows)
    conn.commit()
    return len(rows)


# ------------------------------------------------------------------------ evaluation

INTERVALS = (0.5, 0.8, 0.9)


def evaluate_forecasts(conn, artifacts, run_id=None, now=None) -> pd.DataFrame:
    """Attach final results to stored forecasts and compute the evaluation measures."""
    now = now or now_iso()
    q = """SELECT f.forecast_id, f.run_id, f.game_id, f.home_expected_points, f.away_expected_points, f.home_win_probability, f.expected_margin,
                  f.expected_total, f.forecast_distribution_artifact, g.home_points, g.away_points, g.result_available_at
           FROM forecasts f JOIN games g ON g.game_id=f.game_id
           WHERE g.result_usable=1 AND g.result_available_at <= ?""" + (" AND f.run_id=?" if run_id else "")
    df = pd.read_sql_query(q, conn, params=[now] + ([run_id] if run_id else []))
    if df.empty:
        return df
    cache, out = {}, []
    for r in df.itertuples(index=False):
        hp, ap = float(r.home_points), float(r.away_points)
        margin, total = hp - ap, hp + ap
        winner = "home" if margin > 0 else "away"
        conn.execute("INSERT OR REPLACE INTO game_results(game_id,actual_home_points,actual_away_points,actual_margin,actual_total,actual_winner,result_retrieved_at,source_id) VALUES(?,?,?,?,?,?,?,?)",
                     (int(r.game_id), hp, ap, margin, total, winner, r.result_available_at, "sportsdataverse"))
        y = 1.0 if margin > 0 else 0.0
        p = float(np.clip(r.home_win_probability, 1e-6, 1 - 1e-6))
        crps, cover = None, {}
        ref = r.forecast_distribution_artifact
        if ref and "#" in ref:
            name, idx = ref.split("#")
            path = Path(artifacts) / "forecast_distributions" / name
            if path.exists():
                if name not in cache:
                    z = np.load(path)
                    cache[name] = z["joint"]
                pmf = summarize(cache[name][int(idx)].astype(float))["margin"]
                cdf = np.cumsum(pmf)
                axis = np.arange(-(L - 1), L)
                crps = float(((cdf - (axis >= margin)) ** 2).sum())
                for lvl in INTERVALS:
                    lo, hi = axis[np.searchsorted(cdf, (1 - lvl) / 2)], axis[np.searchsorted(cdf, 1 - (1 - lvl) / 2)]
                    cover[str(lvl)] = bool(lo <= margin <= hi)
        row = (int(r.forecast_id), now, abs(r.expected_margin - margin), abs(r.expected_total - total), abs(r.home_expected_points - hp),
               abs(r.away_expected_points - ap), (p - y) ** 2, -(y * np.log(p) + (1 - y) * np.log(1 - p)), int((p > 0.5) == (y == 1.0)), crps, json.dumps(cover))
        conn.execute("""INSERT OR REPLACE INTO forecast_evaluations(forecast_id,evaluated_at,margin_absolute_error,total_absolute_error,home_absolute_error,
            away_absolute_error,brier,log_loss,winner_correct,crps,interval_coverage_json) VALUES(?,?,?,?,?,?,?,?,?,?,?)""", row)
        out.append(dict(game_id=int(r.game_id), p_home=p, home_win=y, margin_pred=r.expected_margin, margin=margin, total_pred=r.expected_total, total=total,
                        margin_ae=row[2], total_ae=row[3], home_ae=row[4], away_ae=row[5], brier=row[6], log_loss=row[7], correct=row[8], crps=crps, cover=cover))
    conn.commit()
    return pd.DataFrame(out)


def evaluation_summary(ev: pd.DataFrame) -> dict:
    if ev.empty:
        return {"games": 0}
    err = ev.margin_pred - ev.margin
    cover = {lvl: float(np.mean([c.get(lvl, np.nan) for c in ev.cover])) for lvl in map(str, INTERVALS)} if ev.cover.map(bool).any() else {}
    bins = pd.cut(ev.p_home, [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0])
    calib = ev.groupby(bins, observed=True).agg(n=("home_win", "size"), predicted=("p_home", "mean"), observed=("home_win", "mean")).reset_index(drop=True)
    return dict(games=int(len(ev)), winner_accuracy=float(ev["correct"].mean()), brier=float(ev.brier.mean()), log_loss=float(ev.log_loss.mean()),
                margin_mae=float(ev.margin_ae.mean()), margin_rmse=float(np.sqrt((err ** 2).mean())), margin_bias=float(err.mean()),
                total_mae=float(ev.total_ae.mean()), team_score_mae=float(pd.concat([ev.home_ae, ev.away_ae]).mean()),
                crps=float(ev.crps.dropna().mean()) if ev.crps.notna().any() else None, interval_coverage=cover, calibration=calib.to_dict("records"))
