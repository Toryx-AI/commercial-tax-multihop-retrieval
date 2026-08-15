#!/usr/bin/env python3
"""Recompute the anchor-reference (mirror) comparison family from the released vectors.

Three of four panel readers on v9 asked the same question: the paper's primary test
(Gemini embedding-001 vs NV-Embed-v2, +2.31, CI [0.91, 3.71], p=0.001) and the Recall@10
leader pair are quoted in the abstract, but every released interval file uses the *top
entrant* as the reference, so neither number can be checked against an artifact. That is a
reasonable objection, and the fix is not to argue about it: the per-question vectors are
released, so the interval is recomputable and this script recomputes it.

Same estimator, same seed, and the same B as the headline family, so the output is directly
comparable to panel_multiplicity.json rather than merely similar to it.

    python3 scripts/panel_anchor_family.py --resamples 100000

Writes data/results/panel_anchor_family.json.
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

sys.path.insert(0, "scripts")
from panel_confidence_intervals import SOURCES  # noqa: E402

ANCHOR = "NV-Embed-v2"
SEED = 42


def load(metric: str) -> dict[str, np.ndarray]:
    out = {}
    for name, (path, key) in SOURCES.items():
        d = json.load(open(path))
        out[name] = np.asarray(d[key][metric], dtype=float)
    return out


def paired(a: np.ndarray, b: np.ndarray, resamples: int) -> dict:
    """a minus b, paired: both sides always scored on the same resampled question set."""
    d = (a - b) * 100
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(d), size=(resamples, len(d)))
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    # +1 correction, so the smallest reportable p is 2/(B+1) rather than 0.
    p = 2 * min(((means <= 0).sum() + 1) / (resamples + 1),
                ((means >= 0).sum() + 1) / (resamples + 1))
    return {"diff": float(d.mean()), "ci_lo": float(lo), "ci_hi": float(hi),
            "p_raw": float(min(p, 1.0))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resamples", type=int, default=100000)
    ap.add_argument("--out", default="data/results/panel_anchor_family.json")
    args = ap.parse_args()

    result = {"reference": ANCHOR, "resamples": args.resamples, "seed": SEED,
              "estimator": "paired percentile bootstrap over evaluation questions",
              "note": "Reference is the research anchor, not the top entrant. The Holm "
                      "family excludes the pre-specified Gemini-vs-anchor primary test; "
                      "it is reported here for checkability, uncorrected and labelled.",
              "recall5": {}, "recall10": {}}

    for metric, slot in (("r5", "recall5"), ("r10", "recall10")):
        vecs = load(metric)
        anchor = vecs[ANCHOR]
        rows = []
        for name, v in vecs.items():
            if name == ANCHOR:
                continue
            r = paired(v, anchor, args.resamples)
            r["model"] = name
            r["score"] = float(v.mean() * 100)
            rows.append(r)
        rows.sort(key=lambda r: -r["score"])
        result[slot] = {"anchor_score": float(anchor.mean() * 100), "rows": rows}

    json.dump(result, open(args.out, "w"), indent=2)

    for slot in ("recall5", "recall10"):
        print(f"\n=== {slot}  (anchor {result[slot]['anchor_score']:.2f}) ===")
        for r in result[slot]["rows"]:
            print(f"  {r['model']:32s} {r['score']:6.2f}  "
                  f"{r['diff']:+6.2f}  [{r['ci_lo']:+6.2f}, {r['ci_hi']:+6.2f}]  p={r['p_raw']:.4f}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
