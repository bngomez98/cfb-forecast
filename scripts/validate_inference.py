"""Compare the variational posterior with exact NUTS sampling of the same joint model.

Runs on the synthetic league (known truth) and reports, for team offense components at the
forecast week, how closely the variational means match the exact posterior means and how
much narrower the variational standard deviations are.

    python scripts/validate_inference.py --steps 2500 --warmup 300 --samples 300
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from cfb_forecast import football_model as fm  # noqa: E402
from cfb_forecast import forecasting as fc  # noqa: E402
from cfb_forecast import model_spec as sp  # noqa: E402
from cfb_forecast.database import connect  # noqa: E402
from cfb_forecast.observations import derive_observations, load_model_data  # noqa: E402
from synthetic_league import build  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--samples", type=int, default=300)
    ap.add_argument("--teams", type=int, default=6)
    args = ap.parse_args(argv)
    conn = connect(tempfile.mkdtemp() + "/v.sqlite3")
    truth = build(conn, teams=args.teams, seasons=(2024, 2025), weeks=6, observed_weeks_last=3)
    seasons = [2024, 2025]
    derive_observations(conn, seasons)
    data = load_model_data(conn, seasons)
    keep = [sp.WEEKS + 3, sp.WEEKS + 4]
    svi = fm.infer(data, "svi", steps=args.steps, lr=0.03, log=None, keep_periods=keep)
    nuts = fm.infer(data, "nuts", warmup=args.warmup, samples=args.samples, log=print, keep_periods=keep)
    a = fc.extract_bundle(svi, n_draws=args.samples, batch=25)
    b = fc.extract_bundle(nuts, n_draws=args.samples, batch=25)
    report = {"nuts_divergences": nuts.info["divergences"], "mean_tree_steps": float(np.mean(nuts.info["steps"]))}
    for comp in ("pass_efficiency", "rush_efficiency", "success_creation"):
        c = sp.col("OFF", comp)
        sa, sb = a.draws["S_keep"][:, :, 0, c], b.draws["S_keep"][:, :, 0, c]
        report[comp] = dict(corr_of_means=float(np.corrcoef(sa.mean(0), sb.mean(0))[0, 1]),
                            sd_ratio_svi_over_nuts=float(sa.std(0).mean() / sb.std(0).mean()),
                            corr_with_truth_svi=float(np.corrcoef(sa.mean(0), truth["offense"] + truth["qb"])[0, 1]),
                            corr_with_truth_nuts=float(np.corrcoef(sb.mean(0), truth["offense"] + truth["qb"])[0, 1]))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
