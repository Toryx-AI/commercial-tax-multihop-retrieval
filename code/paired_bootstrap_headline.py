#!/usr/bin/env python3
"""paired_bootstrap_headline.py — paired bootstrap CI + sign test on the Paper 1 headline
comparison, now that both models have been measured on the same corpus format, the same
harness and the same GPU.

Inputs are the per-question recall vectors written by the two sweeps:
  data/results/nvembed_variant_sweep_perq.json    (key "<query variant>|<corpus format>")
  data/results/nemotron_variant_sweep_perq.json   (key "<corpus format>")

Both vectors are aligned: same 1,000 MuSiQue questions, same order, same gold sets, so the
difference can be bootstrapped as a paired statistic rather than two independent means.
"""
from __future__ import annotations

import json
import numpy as np

NVEMBED_PERQ = "data/results/nvembed_variant_sweep_perq.json"
NEMOTRON_PERQ = "data/results/nemotron_variant_sweep_perq.json"
NV_KEY = "web-search|title+text"        # NV-Embed-v2's best variant on the matched corpus
OUT = "data/results/headline_paired_bootstrap.json"
B = 10000
SEED = 42


def boot_ci(diff, b=B, seed=SEED, alpha=0.05):
    rng = np.random.default_rng(seed)
    n = len(diff)
    idx = rng.integers(0, n, size=(b, n))
    means = diff[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    # two-sided bootstrap p: how often does the resampled mean cross zero
    p = 2 * min((means <= 0).mean(), (means >= 0).mean())
    return float(diff.mean()), float(lo), float(hi), float(min(p, 1.0))


def main():
    nv = json.load(open(NVEMBED_PERQ))[NV_KEY]
    nm = json.load(open(NEMOTRON_PERQ))["title+text"]

    rows = []
    for k in ("r5", "r10"):
        a = np.asarray(nm[k], dtype=float)   # Nemotron
        b = np.asarray(nv[k], dtype=float)   # NV-Embed-v2
        assert len(a) == len(b), f"length mismatch at {k}: {len(a)} vs {len(b)}"
        d = (a - b) * 100.0                  # Nemotron minus NV-Embed, in points
        mean, lo, hi, p = boot_ci(d)

        wins = int((a > b).sum())
        losses = int((a < b).sum())
        ties = int((a == b).sum())

        rows.append({
            "metric": "Recall@5" if k == "r5" else "Recall@10",
            "nemotron": float(a.mean() * 100),
            "nvembed": float(b.mean() * 100),
            "diff_nemotron_minus_nvembed": mean,
            "ci95": [lo, hi],
            "bootstrap_p": p,
            "n": len(a),
            "questions_nemotron_better": wins,
            "questions_nvembed_better": losses,
            "questions_tied": ties,
            "significant_at_05": bool(lo > 0 or hi < 0),
        })

    print(f"{'metric':11s} {'Nemotron':>9} {'NV-Embed':>9} {'diff':>7}  {'95% CI':>18} {'p':>7}  sig")
    for r in rows:
        ci = f"[{r['ci95'][0]:+.2f}, {r['ci95'][1]:+.2f}]"
        print(f"{r['metric']:11s} {r['nemotron']:9.2f} {r['nvembed']:9.2f} "
              f"{r['diff_nemotron_minus_nvembed']:+7.2f}  {ci:>18} {r['bootstrap_p']:7.3f}  "
              f"{'YES' if r['significant_at_05'] else 'no'}")
        print(f"{'':11s} per-question: Nemotron better on {r['questions_nemotron_better']}, "
              f"NV-Embed better on {r['questions_nvembed_better']}, "
              f"tied on {r['questions_tied']}")

    json.dump({
        "comparison": "Nemotron-3-Embed-8B vs NV-Embed-v2, both title+text corpus, same harness/GPU",
        "nvembed_variant": NV_KEY,
        "bootstrap_resamples": B,
        "seed": SEED,
        "results": rows,
    }, open(OUT, "w"), indent=1)
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
