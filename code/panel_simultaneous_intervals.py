#!/usr/bin/env python3
"""Recompute the simultaneous (Dunnett-type) intervals and their max-t critical value.

The paper quotes a critical value of 2.78 against the marginal 1.96 and reports that the
simultaneous intervals reach the same verdict as the Holm correction. Until now no released
artifact carried either the critical value or the intervals: panel_multiplicity.json has the
Holm fields only. A reviewer flagged that as the one number in the paper a reader could not
check, which is a fair hit on a paper whose thesis is that unverifiable numbers are the
problem. This script closes it.

Method, stated so the value is reproducible rather than asserted. For each of the twelve
entrants compared against the common reference, form the paired per-question difference
vector d_j. On each bootstrap resample b (questions resampled once, jointly, so the twelve
comparisons stay dependent exactly as they are in the data):

    t_j^(b) = (mean(d_j^(b)) - mean(d_j)) / se(d_j^(b))

and record max_j |t_j^(b)|. The critical value is the 95th percentile of that maximum, which
is what makes the twelve intervals simultaneously valid at 95% rather than marginally. The
interval for comparison j is then mean(d_j) +/- c * se(d_j).

    python3 scripts/panel_simultaneous_intervals.py --resamples 100000

Writes data/results/panel_simultaneous_intervals.json.
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

sys.path.insert(0, "scripts")
from panel_confidence_intervals import SOURCES  # noqa: E402

SEED = 42


def load(metric: str) -> dict[str, np.ndarray]:
    out = {}
    for name, (path, key) in SOURCES.items():
        d = json.load(open(path))
        out[name] = np.asarray(d[key][metric], dtype=float)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resamples", type=int, default=100000)
    ap.add_argument("--reference", default="Nemotron-3-Embed-8B",
                    help="common reference; the paper's family uses the top entrant")
    ap.add_argument("--metric", default="r5")
    ap.add_argument("--out", default="data/results/panel_simultaneous_intervals.json")
    args = ap.parse_args()

    vecs = load(args.metric)
    ref = vecs[args.reference]
    names = [n for n in vecs if n != args.reference]

    # D[j] is the paired difference vector for comparison j, in Recall points.
    D = np.stack([(vecs[n] - ref) * 100 for n in names])          # (12, n_questions)
    k, n = D.shape
    obs_mean = D.mean(axis=1)
    obs_se = D.std(axis=1, ddof=1) / np.sqrt(n)

    rng = np.random.default_rng(SEED)
    maxt = np.empty(args.resamples)
    # Chunked so the (B, n) index matrix never has to exist all at once.
    step = max(1, 2_000_000 // n)
    done = 0
    while done < args.resamples:
        b = min(step, args.resamples - done)
        idx = rng.integers(0, n, size=(b, n))
        samp = D[:, idx]                                          # (k, b, n)
        m = samp.mean(axis=2)                                     # (k, b)
        se = samp.std(axis=2, ddof=1) / np.sqrt(n)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.abs(m - obs_mean[:, None]) / se
        t[~np.isfinite(t)] = 0.0        # a resample with zero variance carries no evidence
        maxt[done:done + b] = t.max(axis=0)
        done += b

    crit = float(np.percentile(maxt, 95))

    rows = []
    for j, name in enumerate(names):
        lo, hi = obs_mean[j] - crit * obs_se[j], obs_mean[j] + crit * obs_se[j]
        rows.append({"model": name, "diff": float(obs_mean[j]), "se": float(obs_se[j]),
                     "sim_ci_lo": float(lo), "sim_ci_hi": float(hi),
                     "excludes_zero": bool(lo > 0 or hi < 0),
                     "marginal_ci_lo": float(obs_mean[j] - 1.96 * obs_se[j]),
                     "marginal_ci_hi": float(obs_mean[j] + 1.96 * obs_se[j])})
    rows.sort(key=lambda r: r["diff"])

    result = {"reference": args.reference, "metric": args.metric, "n": int(n),
              "comparisons": k, "resamples": args.resamples, "seed": SEED,
              "method": "bootstrap max-t (Dunnett-type) over jointly resampled questions",
              "critical_value": crit, "marginal_critical_value": 1.96,
              "n_excluding_zero": sum(r["excludes_zero"] for r in rows),
              "rows": rows}
    json.dump(result, open(args.out, "w"), indent=2)

    print(f"reference {args.reference}  k={k}  n={n}  B={args.resamples}")
    print(f"max-t critical value: {crit:.4f}   (marginal 1.96)")
    for r in rows:
        flag = "excl 0" if r["excludes_zero"] else "INCLUDES 0"
        print(f"  {r['model']:32s} {r['diff']:+7.3f}  "
              f"[{r['sim_ci_lo']:+7.3f}, {r['sim_ci_hi']:+7.3f}]  {flag}")
    print(f"\n{result['n_excluding_zero']}/{k} exclude zero simultaneously")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
