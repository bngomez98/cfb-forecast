"""The joint hierarchical Bayesian football model.

One probability model, one joint posterior over everything unknown:

* team offense, defense, offensive-line, defensive-front and special-teams states,
  weekly within-season drift, season-to-season evolution (A, B, Q) and population
  distributions (Sigma);
* quarterback, running-back and receiver states grouped by position;
* the observation equations tying every observed stream to the states of the teams,
  units and players involved, including home field, source bias and the matchup
  interaction O' W D;
* player participation and usage (availability and expected opportunity share);
* the possession model: tempo/possession counts and outcome probabilities for each
  possession conditioned on the same states, with the scoring-point distributions;
* external rating measurements of team strength.

Inference is stochastic variational inference over all of it at once (NumPyro/JAX).
"""
from __future__ import annotations

import numpy as np

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal
from numpyro.infer.initialization import init_to_median
from numpyro.optim import Adam

from . import model_spec as sp

S = len(sp.STREAMS)
K_TEAM = sp.K_TEAM
GROUPS = list(sp.TEAM_GROUPS)
KEY = [sp.STREAM_INDEX[n] for n in sp.KEY_STREAMS]
N_OUT = len(sp.OUTCOMES) - 1                      # free outcome classes ("punt" is the reference)
PAD = K_TEAM                                      # index of the zero column appended to the state vector
MAX_TERMS = 3


def _terms():
    ci, si, cj, sj = (np.full((S, MAX_TERMS), PAD, int), np.zeros((S, MAX_TERMS)), np.full((S, MAX_TERMS), PAD, int), np.zeros((S, MAX_TERMS)))
    for n, st in enumerate(sp.STREAMS):
        for t, (g, c, sign) in enumerate(st.i):
            ci[n, t], si[n, t] = sp.col(g, c), sign
        for t, (g, c, sign) in enumerate(st.j):
            cj[n, t], sj[n, t] = sp.col(g, c), sign
    return ci, si, cj, sj


COL_I, SG_I, COL_J, SG_J = _terms()
IS_B = np.array([s.kind == "B" for s in sp.STREAMS])
HAS_HOME = np.array([s.home for s in sp.STREAMS], float)
HAS_LOADING = np.array([s.loading for s in sp.STREAMS])
HAS_MATCHUP = np.array([s.matchup for s in sp.STREAMS])
FG = sp.STREAM_INDEX["fg_make"]
PASS_EPA = sp.STREAM_INDEX["pass_epa"]
TEMPO_K = KEY.index(sp.STREAM_INDEX["tempo_plays"])
FD_K = KEY.index(sp.STREAM_INDEX["first_down_rate"])
O0, O1 = sp.TEAM_OFFSETS["OFF"], sp.TEAM_OFFSETS["OFF"] + len(sp.TEAM_GROUPS["OFF"])
D0, D1 = sp.TEAM_OFFSETS["DEF"], sp.TEAM_OFFSETS["DEF"] + len(sp.TEAM_GROUPS["DEF"])
RATING_POINTS = np.array([0.0, 6.9, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0])   # offense points per outcome class (rare handled by PMF)


def component_units(data):
    """Prior scale of each state component, from the empirical scale of the streams that observe it."""
    unit = np.full(K_TEAM, 0.3)
    seen = np.zeros(K_TEAM, bool)
    obs_sd = {}
    for s in range(S):
        m = data.obs_s == s
        obs_sd[s] = 0.3 if IS_B[s] else (0.5 * float(np.std(data.obs_val[m])) if m.any() else 0.3)
    for n in range(S):
        for c in list(COL_I[n]) + list(COL_J[n]):
            if c != PAD and not seen[c]:
                unit[c], seen[c] = obs_sd[n], True
    punit = {}
    for g, comps in sp.PLAYER_GROUPS.items():
        u = np.full(len(comps), 0.3)
        for n, st in enumerate(sp.STREAMS):
            for pg, comp, _ in st.players:
                if pg == g:
                    u[comps.index(comp)] = obs_sd[n]
        u[comps.index("usage_capacity")] = 1.0
        punit[g] = u
    return unit, punit


def prepare(data, keep_periods=None):
    """Move the assembled data into JAX arrays plus static structure.

    ``keep_periods`` are the time periods whose team states are returned with every posterior draw
    (the weeks being forecast); by default the three periods after the last observed one.
    """
    unit, punit = component_units(data)
    j = jnp.asarray
    A = dict(
        T=data.T, Y=data.Y, n_gd=data.n_gd, W=sp.WEEKS, unit=j(unit, jnp.float32), punit={g: j(v, jnp.float32) for g, v in punit.items()},
        gd_i=j(data.gd_i), gd_j=j(data.gd_j), gd_per=j(data.gd_per), gd_h=j(data.gd_h, jnp.float32), gd_X=j(data.gd_X, jnp.float32),
        obs_gd=j(data.obs_gd), obs_s=j(data.obs_s), obs_n=j(data.obs_n, jnp.float32), obs_x=j(data.obs_x, jnp.float32),
        obs_val=j(data.obs_val, jnp.float32), obs_cov=j(data.obs_cov, jnp.float32), obs_src=j(data.obs_src),
        stream_scale=j(data.stream_scale, jnp.float32), stream_init=j(data.stream_init, jnp.float32), R=j(data.R, jnp.float32),
        R_dim=data.R.shape[2], n_cov=data.gd_X.shape[1], n_src=data.n_src,
        kind_b=j(IS_B[data.obs_s]), N_idx=j(np.flatnonzero(~IS_B[data.obs_s])), B_idx=j(np.flatnonzero(IS_B[data.obs_s])),
        nodes={}, entries={}, part={}, usage={},
        drive_gd=j(data.drive_gd), drive_class=j(data.drive_class), drive_fp=j(data.drive_fp, jnp.float32),
        td_idx=j(data.td_idx), rare_off_idx=j(data.rare_off_idx), rare_opp_idx=j(data.rare_opp_idx),
        game_gd1=j(data.game_gd1), game_gd2=j(data.game_gd2), game_drives=j(data.game_drives, jnp.float32),
        n_rating=len(data.rating_val),
        keep_periods=jnp.asarray(keep_periods if keep_periods is not None else _default_keep(data)),
    )
    for g in sp.PLAYER_GROUPS:
        nd = data.nodes[g]
        y = nd["y"]
        bounds = [(int(np.searchsorted(y, k)), int(np.searchsorted(y, k, side="right"))) for k in range(data.Y)]
        A["nodes"][g] = dict(n=len(y), bounds=bounds, parent=nd["parent"], cov=j(nd["cov"], jnp.float32))
        e = data.entries[g]
        A["entries"][g] = dict(row=j(e["row"]), node=j(e["node"]), comp=j(e["comp"]), w=j(e["w"], jnp.float32))
        p = data.participation[g]
        A["part"][g] = dict(node=j(p["node"]), gd=j(p["gd"]), z=j(p["z"], jnp.float32))
        u = data.usage[g]
        A["usage"][g] = dict(seg=j(u["seg"]), node=j(u["node"]), count=j(u["count"], jnp.float32))
    if A["n_rating"]:
        A.update(rating_team=j(data.rating_team), rating_per=j(data.rating_per), rating_src=j(data.rating_src), rating_val=j(data.rating_val, jnp.float32))
    return A


def _default_keep(data):
    last = int(data.gd_per.max()) if len(data.gd_per) else 0
    top = data.Y * sp.WEEKS - 1
    return [min(last + k, top) for k in (1, 2, 3)]


# ---------------------------------------------------------------------------- the model

def _mvn_chol(name, K, unit, scale_prior=1.0, lkj=2.0):
    sig = numpyro.sample(f"sig_{name}", dist.HalfNormal(scale_prior).expand([K]).to_event(1))
    L = numpyro.sample(f"L_{name}", dist.LKJCholesky(K, lkj))
    return (unit * sig)[:, None] * L


def team_states(P, A):
    """Stack the team-group states, shape (T, Y*W, K_TEAM): season start from persistence, weekly drift within."""
    T, Y, W = A["T"], A["Y"], A["W"]
    blocks = []
    for g in GROUPS:
        K = len(sp.TEAM_GROUPS[g])
        sl = slice(sp.TEAM_OFFSETS[g], sp.TEAM_OFFSETS[g] + K)
        unit = A["unit"][sl]
        LS, LQ = P[f"LS_{g}"], P[f"LQ_{g}"]
        Am = 0.5 * jnp.eye(K) + P[f"Araw_{g}"]
        Bm = P[f"B_{g}"]
        kap = unit * P[f"kap_{g}"]
        z0, eps = P[f"z0_{g}"], P[f"eps_{g}"]
        drift = jnp.cumsum(eps * kap, axis=2)
        drift = drift - drift[:, :, :1]
        starts, paths = [], []
        prev_end = None
        for y in range(Y):
            if y == 0:
                s0 = z0[:, 0] @ LS.T
            else:
                s0 = prev_end @ Am.T + A["R"][:, y] @ Bm.T + z0[:, y] @ LQ.T
            path = s0[:, None, :] + drift[:, y]
            paths.append(path)
            prev_end = path[:, -1]
        blocks.append(jnp.concatenate(paths, axis=1))
    return jnp.concatenate(blocks, axis=2)


def player_states(P, A, g):
    K = len(sp.PLAYER_GROUPS[g])
    nd = A["nodes"][g]
    LS, LQ = P[f"pLS_{g}"], P[f"pLQ_{g}"]
    Am = 0.5 * jnp.eye(K) + P[f"pAraw_{g}"]
    Bm = P[f"pB_{g}"]
    z = P[f"pz_{g}"]
    vals = jnp.zeros((nd["n"], K))
    parent = nd["parent"]
    for y, (lo, hi) in enumerate(nd["bounds"]):
        if hi <= lo:
            continue
        par = parent[lo:hi]
        has = jnp.asarray(par >= 0)
        prev = vals[jnp.asarray(np.maximum(par, 0))]
        mean = jnp.where(has[:, None], prev @ Am.T + nd["cov"][lo:hi] @ Bm.T, 0.0)
        noise = jnp.where(has[:, None], z[lo:hi] @ LQ.T, z[lo:hi] @ LS.T)
        vals = vals.at[lo:hi].set(mean + noise)
    return vals


def linear_predictors(P, A, S_all, players):
    """Link-scale mean of every (game direction, stream), without intercept/home/bias, plus the matchup index."""
    Spad = jnp.concatenate([S_all, jnp.zeros(S_all.shape[:2] + (1,))], axis=2)
    Oi = Spad[A["gd_i"], A["gd_per"]]
    Dj = Spad[A["gd_j"], A["gd_per"]]
    team = (Oi[:, COL_I] * SG_I).sum(-1) + (Dj[:, COL_J] * SG_J).sum(-1)
    lam = jnp.where(HAS_LOADING, P["loading"], 1.0)
    lin = team * lam
    Wm = P["W"]
    M = jnp.einsum("gk,kl,gl->g", Oi[:, O0:O1], Wm, Dj[:, D0:D1])
    lin = lin + jnp.where(HAS_MATCHUP, P["gamma"], 0.0) * M[:, None]
    flat = jnp.zeros(A["n_gd"] * S)
    for g, ps in players.items():
        e = A["entries"][g]
        flat = flat + jax.ops.segment_sum(e["w"] * ps[e["node"], e["comp"]], e["row"], num_segments=A["n_gd"] * S)
    return lin + flat.reshape(A["n_gd"], S), M


def model(A):
    T, Y = A["T"], A["Y"]
    P = {}
    # ---- hierarchical state parameters per team group
    for g in GROUPS:
        K = len(sp.TEAM_GROUPS[g])
        unit = A["unit"][sp.TEAM_OFFSETS[g]: sp.TEAM_OFFSETS[g] + K]
        P[f"LS_{g}"] = _mvn_chol(f"S_{g}", K, unit)
        P[f"LQ_{g}"] = _mvn_chol(f"Q_{g}", K, unit)
        P[f"kap_{g}"] = numpyro.sample(f"kap_{g}", dist.HalfNormal(0.2).expand([K]).to_event(1))
        P[f"Araw_{g}"] = numpyro.sample(f"Araw_{g}", dist.Normal(0, 0.3).expand([K, K]).to_event(2))
        P[f"B_{g}"] = numpyro.sample(f"B_{g}", dist.Normal(0, 0.3).expand([K, A["R_dim"]]).to_event(2))
        P[f"z0_{g}"] = numpyro.sample(f"z0_{g}", dist.Normal(0, 1).expand([T, Y, K]).to_event(3))
        P[f"eps_{g}"] = numpyro.sample(f"eps_{g}", dist.Normal(0, 1).expand([T, Y, A["W"], K]).to_event(4))
    S_all = team_states(P, A)
    # ---- player states grouped by position
    players = {}
    for g, comps in sp.PLAYER_GROUPS.items():
        K = len(comps)
        P[f"pLS_{g}"] = _mvn_chol(f"pS_{g}", K, A["punit"][g])
        P[f"pLQ_{g}"] = _mvn_chol(f"pQ_{g}", K, A["punit"][g])
        P[f"pAraw_{g}"] = numpyro.sample(f"pAraw_{g}", dist.Normal(0, 0.3).expand([K, K]).to_event(2))
        P[f"pB_{g}"] = numpyro.sample(f"pB_{g}", dist.Normal(0, 0.3).expand([K, 1]).to_event(2))
        P[f"pz_{g}"] = numpyro.sample(f"pz_{g}", dist.Normal(0, 1).expand([A["nodes"][g]["n"], K]).to_event(2))
        players[g] = player_states(P, A, g)
    # ---- observation equations
    P["alpha"] = numpyro.sample("alpha", dist.Normal(A["stream_init"], 1.0).to_event(1))
    P["eta"] = numpyro.sample("eta", dist.Normal(0, 0.2).expand([S]).to_event(1))
    P["loading"] = numpyro.sample("loading", dist.LogNormal(0, 0.5).expand([S]).to_event(1))
    P["gamma"] = numpyro.sample("gamma", dist.Normal(1.0, 0.5).expand([S]).to_event(1))
    P["W"] = numpyro.sample("W_raw", dist.Normal(0, 1).expand([O1 - O0, D1 - D0]).to_event(2))
    tauW = numpyro.sample("tau_W", dist.HalfNormal(0.3))
    P["W"] = P["W"] * tauW
    sigma = numpyro.sample("sigma", dist.HalfNormal(A["stream_scale"] * 1.5).to_event(1))
    phi = numpyro.sample("phi", dist.LogNormal(jnp.log(60.0), 1.0).expand([S]).to_event(1))
    fg_cov = numpyro.sample("fg_distance_coef", dist.Normal(0, 1.0))
    bias = None
    if A["n_src"] > 1:
        bias = numpyro.sample("source_bias", dist.Normal(0, 0.1).expand([A["n_src"] - 1, S]).to_event(2))

    numpyro.deterministic("S_keep", S_all[:, A["keep_periods"]])
    for g_, ps_ in players.items():
        numpyro.deterministic(f"pstate_{g_}", ps_)
    lin, M = linear_predictors(P, A, S_all, players)
    mu = lin + P["alpha"] + P["eta"] * HAS_HOME * A["gd_h"][:, None]
    mu_obs = mu[A["obs_gd"], A["obs_s"]] + jnp.where(A["obs_s"] == FG, fg_cov * A["obs_cov"], 0.0)
    if bias is not None:
        mu_obs = mu_obs + jnp.where(A["obs_src"] > 0, bias[jnp.maximum(A["obs_src"] - 1, 0), A["obs_s"]], 0.0)
    ni, bi = A["N_idx"], A["B_idx"]
    numpyro.sample("y_normal", dist.Normal(mu_obs[ni], sigma[A["obs_s"][ni]] / jnp.sqrt(A["obs_n"][ni])), obs=A["obs_val"][ni])
    pb = jax.nn.sigmoid(mu_obs[bi])
    ph = phi[A["obs_s"][bi]]
    numpyro.sample("y_binomial", dist.BetaBinomial(pb * ph + 1e-3, (1 - pb) * ph + 1e-3, total_count=A["obs_n"][bi]), obs=A["obs_x"][bi])

    # ---- player participation and expected opportunity share
    for g in sp.PLAYER_GROUPS:
        part = A["part"][g]
        a = numpyro.sample(f"part_intercept_{g}", dist.Normal(0.5, 1.0))
        su = numpyro.sample(f"part_sd_{g}", dist.HalfNormal(1.0))
        bprior = numpyro.sample(f"part_prior_usage_{g}", dist.Normal(0, 2.0))
        u = numpyro.sample(f"part_node_{g}", dist.Normal(0, 1).expand([A["nodes"][g]["n"]]).to_event(1))
        if len(part["z"]):
            logits = a + su * u[part["node"]] + bprior * A["nodes"][g]["cov"][part["node"], 0]
            numpyro.sample(f"z_{g}", dist.Bernoulli(logits=logits), obs=part["z"])
        use = A["usage"][g]
        comp = sp.PLAYER_GROUPS[g].index("usage_capacity")
        cap = players[g][:, comp]
        val = cap[use["node"]]
        seg = use["seg"]
        mx = jax.ops.segment_max(val, seg, num_segments=A["n_gd"])
        lse = mx + jnp.log(jax.ops.segment_sum(jnp.exp(val - mx[seg]), seg, num_segments=A["n_gd"]) + 1e-12)
        numpyro.factor(f"usage_{g}", jnp.sum(use["count"] * (val - lse[seg])))

    # ---- possessions: outcome model, points, tempo
    n_key = len(KEY)
    feats = jnp.concatenate([lin[:, KEY], M[:, None]], axis=1)
    b0 = numpyro.sample("outcome_intercept", dist.Normal(0, 2.0).expand([N_OUT]).to_event(1))
    Wc = numpyro.sample("outcome_coef", dist.Normal(0, 3.0).expand([n_key + 1, N_OUT]).to_event(2))
    fpc = numpyro.sample("outcome_field_position", dist.Normal(0, 1.0).expand([N_OUT]).to_event(1))
    hc = numpyro.sample("outcome_home", dist.Normal(0, 0.5).expand([N_OUT]).to_event(1))
    Cc = numpyro.sample("outcome_covariates", dist.Normal(0, 0.5).expand([max(A["n_cov"], 1), N_OUT]).to_event(2))
    dg = A["drive_gd"]
    logits = b0 + feats[dg] @ Wc + A["drive_fp"][:, None] * fpc + A["gd_h"][dg][:, None] * hc
    if A["n_cov"]:
        logits = logits + A["gd_X"][dg] @ Cc[: A["n_cov"]]
    logits = jnp.concatenate([jnp.zeros((logits.shape[0], 1)), logits], axis=1)
    numpyro.sample("drive_outcome", dist.Categorical(logits=logits), obs=A["drive_class"])
    td_l = numpyro.sample("td_points_logits", dist.Normal(0, 2.0).expand([len(sp.TD_POINTS)]).to_event(1))
    ro_l = numpyro.sample("rare_off_logits", dist.Normal(0, 2.0).expand([len(sp.RARE_POINTS)]).to_event(1))
    rp_l = numpyro.sample("rare_opp_logits", dist.Normal(0, 2.0).expand([len(sp.RARE_POINTS)]).to_event(1))
    for name, lg, idx in (("td", td_l, A["td_idx"]), ("rare_off", ro_l, A["rare_off_idx"]), ("rare_opp", rp_l, A["rare_opp_idx"])):
        m = idx >= 0
        numpyro.factor(f"{name}_points", jnp.sum(jnp.where(m, jax.nn.log_softmax(lg)[jnp.maximum(idx, 0)], 0.0)))
    c0 = numpyro.sample("drives_intercept", dist.Normal(jnp.log(24.0), 0.5))
    dt = numpyro.sample("drives_tempo", dist.Normal(0, 1.0))
    ds = numpyro.sample("drives_sustain", dist.Normal(0, 1.0))
    dphi = numpyro.sample("drives_concentration", dist.LogNormal(jnp.log(40.0), 1.0))
    g1, g2 = A["game_gd1"], A["game_gd2"]
    lam = jnp.exp(c0 + dt * (lin[g1, sp.STREAM_INDEX["tempo_plays"]] + lin[g2, sp.STREAM_INDEX["tempo_plays"]])
                  + ds * (lin[g1, sp.STREAM_INDEX["first_down_rate"]] + lin[g2, sp.STREAM_INDEX["first_down_rate"]]))
    numpyro.sample("drives_total", dist.NegativeBinomial2(lam, dphi), obs=A["game_drives"])

    # ---- external ratings measure a composite team strength (expected points per game above average)
    if A["n_rating"]:
        p_bar = jax.nn.softmax(jnp.concatenate([jnp.zeros(1), b0]))
        pts = jnp.asarray(RATING_POINTS, jnp.float32)
        Wfull = jnp.concatenate([jnp.zeros((n_key + 1, 1)), Wc], axis=1)
        g_k = (Wfull - (Wfull * p_bar).sum(1, keepdims=True)) @ (p_bar * pts)
        Spad = jnp.concatenate([S_all, jnp.zeros(S_all.shape[:2] + (1,))], axis=2)
        st = Spad[A["rating_team"], A["rating_per"]]
        ks = np.array(KEY)
        offense = (st[:, COL_I[ks]] * SG_I[ks]).sum(-1)
        defense = -(st[:, COL_J[ks]] * np.minimum(SG_J[ks], 0)).sum(-1)
        theta = 12.0 * ((offense + defense) * g_k[:n_key]).sum(-1)
        ra = numpyro.sample("rating_offset", dist.Normal(0, 10.0).expand([len(sp.RATING_SOURCES)]).to_event(1))
        rb = numpyro.sample("rating_slope", dist.HalfNormal(2.0).expand([len(sp.RATING_SOURCES)]).to_event(1))
        rs = numpyro.sample("rating_sd", dist.HalfNormal(10.0).expand([len(sp.RATING_SOURCES)]).to_event(1))
        numpyro.sample("ratings", dist.Normal(ra[A["rating_src"]] + rb[A["rating_src"]] * theta, rs[A["rating_src"]]), obs=A["rating_val"])


# ---------------------------------------------------------------------------- inference

class Fit:
    def __init__(self, data, A, guide, params, losses):
        self.data, self.A, self.guide, self.params, self.losses = data, A, guide, params, losses


def fit(data, steps=3000, lr=0.02, seed=0, log_every=250, log=print, keep_periods=None) -> Fit:
    """Fit the whole model at once by stochastic variational inference."""
    A = prepare(data, keep_periods)
    init = {"alpha": A["stream_init"], "sigma": A["stream_scale"]}
    guide = AutoNormal(model, init_loc_fn=init_to_median(num_samples=15), init_scale=0.02)
    schedule = numpyro.optim.optax_to_numpyro(__import__("optax").adamw(
        __import__("optax").cosine_decay_schedule(lr, steps, alpha=0.05)))
    svi = SVI(model, guide, schedule, Trace_ELBO())
    state = svi.init(jax.random.PRNGKey(seed), A)
    update = jax.jit(lambda s: svi.update(s, A))
    losses = []
    for step in range(steps):
        state, loss = update(state)
        losses.append(float(loss))
        if log and (step % log_every == 0 or step == steps - 1):
            log(f"step {step:5d}  -ELBO {loss:,.1f}")
    return Fit(data, A, guide, svi.get_params(state), np.array(losses))


def posterior_draws(f: Fit, n: int, seed: int = 1) -> dict:
    """Draw n joint samples of every latent and parameter from the fitted posterior."""
    return f.guide.sample_posterior(jax.random.PRNGKey(seed), f.params, sample_shape=(n,))


def posterior_median(f: Fit) -> dict:
    return f.guide.median(f.params)
