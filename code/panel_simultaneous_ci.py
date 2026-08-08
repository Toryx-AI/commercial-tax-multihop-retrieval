#!/usr/bin/env python3
"""Simultaneous (Dunnett-type) confidence intervals for the twelve panel comparisons.

The problem this fixes
----------------------
Table 1 reports each entrant's own 95% interval. Each is 95% for that entrant
considered alone -- but a reader looks at all thirteen at once, and the chance
that all thirteen simultaneously contain their true values is well below 95%. So
the paper pairs multiplicity-corrected p-values with uncorrected intervals, and
then has to warn readers in a figure caption not to draw conclusions from the
intervals it is showing them.

A caption asking readers to distrust the plot is a sign the wrong thing is
plotted. What is actually being tested is the *difference* between each entrant
and the top one, so that is what should carry an interval.

The method
----------
Bootstrap max-t, the resampling form of Dunnett's procedure for many-vs-one
comparisons. Resample the evaluation questions; crucially, use the SAME resampled
questions for all twelve comparisons in each draw, because the twelve share a
reference and are strongly correlated -- and it is exactly that correlation that
makes the correction cheap. Take the largest studentized deviation across the
twelve in each draw; its 95th percentile is the critical value that makes all
twelve intervals hold at once.

Compare three critical values to see what the correction costs:
  marginal (no correction, wrong for a set)  1.96
  Bonferroni over 12 (ignores correlation)   ~2.87
  bootstrap max-t (uses the correlation)     computed here

Usage
-----
    python3 scripts/panel_simultaneous_ci.py
    python3 scripts/panel_simultaneous_ci.py --resamples 100000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from panel_multiplicity import PANEL, load  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resamples", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/panel_simultaneous_ci.json")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    vecs = {name: load(f, k) for name, f, k in PANEL}
    ref_name = max(vecs, key=lambda k: vecs[k].mean())
    ref = vecs[ref_name]
    n = len(ref)
    others = [k for k in vecs if k != ref_name]

    # Per-question difference vectors, all against the same reference.
    D = np.vstack([ref - vecs[k] for k in others]) * 100      # 12 x n, in points
    d_hat = D.mean(axis=1)
    se = D.std(axis=1, ddof=1) / np.sqrt(n)

    # Average pairwise correlation among the twelve difference vectors -- this is
    # what makes the simultaneous correction cheaper than Bonferroni.
    C = np.corrcoef(D)
    rho = (C.sum() - len(C)) / (len(C) * (len(C) - 1))

    # Bootstrap max-t. One shared resample of QUESTIONS per draw, applied to all
    # twelve comparisons, so the correlation structure is preserved.
    maxt = np.empty(a.resamples)
    B, step = a.resamples, 2000
    for s in range(0, B, step):
        m = min(step, B - s)
        idx = rng.integers(0, n, size=(m, n))
        boot = D[:, idx].mean(axis=2)                          # 12 x m
        t = (boot - d_hat[:, None]) / se[:, None]
        maxt[s:s + m] = np.abs(t).max(axis=0)

    c_sim = float(np.percentile(maxt, 95))
    c_marg = 1.959963985
    from math import erf, sqrt
    # Bonferroni critical value for 12 two-sided tests at 0.05, by bisection on Phi.
    lo, hi = 1.0, 5.0
    target = 1 - 0.05 / (2 * 12)
    for _ in range(200):
        mid = (lo + hi) / 2
        if 0.5 * (1 + erf(mid / sqrt(2))) < target:
            lo = mid
        else:
            hi = mid
    c_bonf = (lo + hi) / 2

    print(f"n = {n} questions, B = {a.resamples:,}, seed {a.seed}")
    print(f"reference (top entrant): {ref_name}  {100*ref.mean():.2f}")
    print(f"mean pairwise correlation among the 12 difference vectors: {rho:.3f}\n")
    print(f"critical values   marginal {c_marg:.3f}   bootstrap max-t {c_sim:.3f}   "
          f"Bonferroni {c_bonf:.3f}")
    print(f"  the correlation saves {100*(c_bonf-c_sim)/c_bonf:.1f}% of Bonferroni's widening\n")

    print(f"{'model':30s} {'diff':>7s} {'simultaneous 95% CI':>24s}  excl 0")
    rows = []
    for i, k in enumerate(others):
        lo_i, hi_i = d_hat[i] - c_sim * se[i], d_hat[i] + c_sim * se[i]
        excl = lo_i > 0 or hi_i < 0
        rows.append({"model": k, "diff_vs_top": float(d_hat[i]), "se": float(se[i]),
                     "sim_lo": float(lo_i), "sim_hi": float(hi_i), "excludes_zero": bool(excl)})
        print(f"{k:30s} {d_hat[i]:+7.2f}   [{lo_i:+6.2f}, {hi_i:+6.2f}]      {'yes' if excl else 'NO'}")

    n_excl = sum(r["excludes_zero"] for r in rows)
    survivors = [r["model"] for r in rows if not r["excludes_zero"]]
    print(f"\n{n_excl}/{len(rows)} separable from the top entrant, simultaneously at 95%.")
    print(f"not separable: {survivors if survivors else 'none'}")

    (REPO / a.out).write_text(json.dumps(
        {"reference": ref_name, "n": n, "resamples": a.resamples, "seed": a.seed,
         "mean_correlation": float(rho), "crit_marginal": c_marg,
         "crit_simultaneous_maxt": c_sim, "crit_bonferroni": float(c_bonf),
         "comparisons": rows}, indent=2))
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
