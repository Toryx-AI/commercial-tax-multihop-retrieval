#!/usr/bin/env python3
"""Exact paired-bootstrap p-values for the 13-model panel, with Holm-Bonferroni.

Why this exists
---------------
v7's panel caption makes a *uniqueness* claim -- "Only NV-Embed-v2 is statistically
indistinguishable from the top entrant; every other embedder is significantly below
it at 95% (paired bootstrap, all p<0.002)" -- across twelve pairwise comparisons
against a common reference, with no multiplicity correction stated.

Two separate problems, both fixable without new measurement:

1. A uniqueness claim is the conjunction of twelve rejections. Unadjusted at
   alpha=0.05 the family-wise error rate is 1-0.95^12 ~ 46%, so roughly one
   spurious rejection is expected somewhere in the family. The claim as worded is
   under-supported by the test as run.

2. "all p<0.002" is a censored bound, not a p-value. With 10,000 resamples the
   finest resolvable p is 1e-4, so exact values are available and the reader
   should not have to take a bound on trust.

This script computes the exact per-comparison p-values from the saved per-question
recall vectors and applies Holm-Bonferroni, so the paper can state the correction
and show the conclusion survives it rather than leaving a referee to check.

Usage
-----
    python3 scripts/panel_multiplicity.py
    python3 scripts/panel_multiplicity.py --resamples 10000 --seed 42
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
RES = REPO / "results"   # release copy: layout adapted from the research repo

# Where each entrant's per-question Recall@5 vector lives. The panel is assembled
# from four files because the entrants were measured by four different harnesses
# (API sweep, two GPU sweeps, self-hosted sweep) -- which is itself worth stating
# in the paper, since a reader cannot otherwise tell the vectors are commensurate.
# They are: same 1,000 questions, same order, same gold matching, same scorer.
PANEL = [
    ("Nemotron-3-Embed-8B",         "nemotron_variant_sweep_perq.json",   "title+text"),
    ("NV-Embed-v2",                 "nvembed_variant_sweep_perq.json",    "web-search|title+text"),
    ("Gemini embedding-001",        "api_panel_sweep_perq.json",          "gemini|title+text"),
    ("Nemotron-3-Embed-1B",         "api_panel_sweep_perq.json",          "nim-nemotron-1b|title+text"),
    ("Llama-Nemotron-Embed-1B-v2",  "api_panel_sweep_perq.json",          "nim-llama-nemotron-1b-v2|title+text"),
    ("Cohere Embed v4",             "api_panel_sweep_perq.json",          "cohere|title+text"),
    ("Qwen3-VL-Embedding-8B",       "selfhosted_panel_perq_vectors.json", "qwen3vl"),
    ("text-embedding-3-large",      "api_panel_sweep_perq.json",          "openai-large|title+text"),
    ("nv-embedqa-e5-v5",            "api_panel_sweep_perq.json",          "nim-e5-v5|title+text"),
    ("mxbai-embed-large-v1",        "selfhosted_panel_perq_vectors.json", "mxbai"),
    ("text-embedding-3-small",      "api_panel_sweep_perq.json",          "openai-small|title+text"),
    ("BGE-M3",                      "selfhosted_panel_perq_vectors.json", "bgem3"),
    ("voyage-3.5",                  "api_panel_sweep_perq.json",          "voyage|title+text"),
]


def load(fname: str, key: str) -> np.ndarray:
    d = json.loads((RES / fname).read_text())
    if "results" in d and isinstance(d["results"], list):        # self-hosted shape
        for e in d["results"]:
            if e.get("model") == key:
                return np.asarray(e["r5"], dtype=float)
        raise KeyError(f"{key} not in {fname}")
    return np.asarray(d[key]["r5"], dtype=float)


def paired_p(a: np.ndarray, b: np.ndarray, resamples: int, rng: np.random.Generator) -> tuple[float, float, float, float]:
    """Two-sided paired bootstrap on the per-question difference.

    p is the proportion of resampled mean differences that fall on or across zero,
    doubled. Reported with the +1 correction so a p of exactly 0 is never claimed:
    with B resamples the smallest defensible statement is p < 1/B.
    """
    d = a - b
    idx = rng.integers(0, len(d), size=(resamples, len(d)))
    boot = d[idx].mean(axis=1)
    obs = d.mean()
    lo, hi = np.percentile(boot, [2.5, 97.5])
    tail = (boot <= 0).sum() if obs > 0 else (boot >= 0).sum()
    p = min(1.0, 2.0 * (tail + 1) / (resamples + 1))
    return obs * 100, lo * 100, hi * 100, p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resamples", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/panel_multiplicity.json")
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    vecs = {name: load(f, k) for name, f, k in PANEL}
    n = len(next(iter(vecs.values())))
    assert all(len(v) == n for v in vecs.values()), "vectors are not the same length"

    ref_name = max(vecs, key=lambda k: vecs[k].mean())
    ref = vecs[ref_name]
    print(f"n = {n} questions, {a.resamples:,} resamples, seed {a.seed}")
    print(f"top entrant (reference): {ref_name}  Recall@5 = {100*ref.mean():.2f}\n")

    rows = []
    for name, v in vecs.items():
        if name == ref_name:
            continue
        diff, lo, hi, p = paired_p(ref, v, a.resamples, rng)
        rows.append({"model": name, "recall5": float(100 * v.mean()), "diff_vs_top": float(diff),
                     "ci_lo": float(lo), "ci_hi": float(hi), "p_raw": float(p)})

    # Holm-Bonferroni, step-down. Sort ascending by p; threshold alpha/(m-i).
    rows.sort(key=lambda r: r["p_raw"])
    m = len(rows)
    still_rejecting = True
    for i, r in enumerate(rows):
        thr = 0.05 / (m - i)
        r["holm_threshold"] = thr
        still_rejecting = bool(still_rejecting and r["p_raw"] <= thr)
        r["holm_reject"] = still_rejecting

    print(f"{'model':30s} {'R@5':>6s} {'diff':>7s} {'95% CI':>18s} {'p':>10s} {'Holm thr':>9s}  rej")
    for r in rows:
        ci = f"[{r['ci_lo']:+.2f},{r['ci_hi']:+.2f}]"
        print(f"{r['model']:30s} {r['recall5']:6.2f} {r['diff_vs_top']:+7.2f} {ci:>18s} "
              f"{r['p_raw']:10.5f} {r['holm_threshold']:9.5f}  {'yes' if r['holm_reject'] else 'NO'}")

    # The paper's claim is a uniqueness claim with a NAMED exception: every entrant
    # except NV-Embed-v2 is separable from the top. So "all twelve rejected" is the
    # wrong success condition -- it would mean the anchor is separable too, which
    # contradicts the headline. The claim holds iff the anchor is the sole survivor.
    ANCHOR = "NV-Embed-v2"
    n_rej = sum(r["holm_reject"] for r in rows)
    survivors = [r["model"] for r in rows if not r["holm_reject"]]
    print(f"\nHolm-Bonferroni at alpha=0.05 over m={m}: {n_rej}/{m} rejected.")
    if survivors == [ANCHOR]:
        print(f"Uniqueness claim SURVIVES correction: {ANCHOR} is the sole entrant not")
        print("separable from the top, and all other 11 are separable family-wise.")
    elif not survivors:
        print(f"Claim CHANGES: even {ANCHOR} is separable from the top after correction.")
    else:
        print("Uniqueness claim FAILS: more than the anchor survives. Not separable:")
        for k in survivors:
            print(f"  - {k}")

    out = REPO / a.out
    out.write_text(json.dumps({"reference": ref_name, "n": n, "resamples": a.resamples,
                               "seed": a.seed, "alpha": 0.05, "correction": "holm-bonferroni",
                               "comparisons": rows}, indent=2))
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
