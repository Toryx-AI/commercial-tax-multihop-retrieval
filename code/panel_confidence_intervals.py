#!/usr/bin/env python3
"""panel_confidence_intervals.py — bootstrap confidence intervals for every embedder in the
Paper 1 panel, plus the paired difference against the top-ranked model.

The paper reports a 13-embedder panel as bare point estimates, with a headline separation of
0.24 Recall@5 points on n=1,000. This computes what that separation is actually worth.

All 13 rows are on the SAME corpus format (title+text), which required re-running the panel -
see docs/paper/ARTIFACT_MANIFEST_2026-08-05.md. NV-Embed-v2's original figure was measured on a
text-only corpus and is not used here.

Per-question sources (all aligned: same 1,000 questions, same order, same gold sets):
  nvembed_variant_sweep_perq.json   "web-search|title+text"
  nemotron_variant_sweep_perq.json  "title+text"
  api_panel_sweep_perq.json         "<provider>|title+text"
  selfhosted_panel_perq_vectors.json  "<model>"

  python3 scripts/panel_confidence_intervals.py
"""
from __future__ import annotations

import json, os
import numpy as np

R = "data/results"
B = 10000
SEED = 42

# display name -> (file, key)
SOURCES = {
    "Nemotron-3-Embed-8B":          (f"{R}/nemotron_variant_sweep_perq.json", "title+text"),
    "NV-Embed-v2":                  (f"{R}/nvembed_variant_sweep_perq.json", "web-search|title+text"),
    "Gemini embedding-001":         (f"{R}/api_panel_sweep_perq.json", "gemini|title+text"),
    "Nemotron-3-Embed-1B":          (f"{R}/api_panel_sweep_perq.json", "nim-nemotron-1b|title+text"),
    "Llama-Nemotron-Embed-1B-v2":   (f"{R}/api_panel_sweep_perq.json", "nim-llama-nemotron-1b-v2|title+text"),
    "Cohere Embed v4":              (f"{R}/api_panel_sweep_perq.json", "cohere|title+text"),
    "OpenAI text-embedding-3-large": (f"{R}/api_panel_sweep_perq.json", "openai-large|title+text"),
    "Qwen3-VL-Embedding-8B":        (f"{R}/selfhosted_panel_perq_vectors.json", "qwen3vl"),
    "nv-embedqa-e5-v5":             (f"{R}/api_panel_sweep_perq.json", "nim-e5-v5|title+text"),
    "mxbai-embed-large-v1":         (f"{R}/selfhosted_panel_perq_vectors.json", "mxbai"),
    "OpenAI text-embedding-3-small": (f"{R}/api_panel_sweep_perq.json", "openai-small|title+text"),
    "BGE-M3":                       (f"{R}/selfhosted_panel_perq_vectors.json", "bgem3"),
    "Voyage voyage-3.5":            (f"{R}/api_panel_sweep_perq.json", "voyage|title+text"),
}

LICENSE = {
    "Nemotron-3-Embed-8B": "Commercial", "NV-Embed-v2": "Non-commercial",
    "Gemini embedding-001": "Commercial", "Nemotron-3-Embed-1B": "Commercial",
    "Llama-Nemotron-Embed-1B-v2": "Commercial", "Cohere Embed v4": "Commercial",
    "OpenAI text-embedding-3-large": "Commercial", "Qwen3-VL-Embedding-8B": "Free/open",
    "nv-embedqa-e5-v5": "Commercial", "mxbai-embed-large-v1": "Free/open",
    "OpenAI text-embedding-3-small": "Commercial", "BGE-M3": "Free/open",
    "Voyage voyage-3.5": "Commercial",
}


def boot_mean_ci(x, b=B, seed=SEED, alpha=0.05):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(b, len(x)))
    means = x[idx].mean(axis=1) * 100
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(x.mean() * 100), float(lo), float(hi)


def boot_paired(a, b_, b=B, seed=SEED, alpha=0.05):
    """a minus b_, paired over questions."""
    d = (a - b_) * 100
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(b, len(d)))
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    p = 2 * min((means <= 0).mean(), (means >= 0).mean())
    return float(d.mean()), float(lo), float(hi), float(min(p, 1.0))


def main():
    vecs = {}
    for name, (path, key) in SOURCES.items():
        if not os.path.exists(path):
            print(f"!! missing {path} for {name}")
            continue
        d = json.load(open(path))
        if key not in d:
            print(f"!! key {key!r} not in {path} for {name} (have: {list(d)[:4]}...)")
            continue
        vecs[name] = {k: np.asarray(v, dtype=float) for k, v in d[key].items()}

    lens = {name: len(v["r5"]) for name, v in vecs.items()}
    assert len(set(lens.values())) == 1, f"misaligned question counts: {lens}"
    n = next(iter(lens.values()))
    print(f"{len(vecs)} embedders, n={n} questions each\n")

    rows = []
    for name, v in vecs.items():
        m5, lo5, hi5 = boot_mean_ci(v["r5"])
        m10, lo10, hi10 = boot_mean_ci(v["r10"])
        rows.append({"embedder": name, "license": LICENSE.get(name, ""),
                     "recall_at_5": m5, "ci95_r5": [lo5, hi5],
                     "recall_at_10": m10, "ci95_r10": [lo10, hi10]})
    rows.sort(key=lambda r: -r["recall_at_5"])

    top = rows[0]["embedder"]
    for r in rows:
        d, lo, hi, p = boot_paired(vecs[top]["r5"], vecs[r["embedder"]]["r5"])
        r["vs_top_r5"] = {"diff": d, "ci95": [lo, hi], "p": p,
                          "significant": bool(lo > 0 or hi < 0)}

    w = max(len(r["embedder"]) for r in rows)
    print(f"{'embedder':{w}s} {'licence':15s} {'R@5':>6} {'95% CI':>16} "
          f"{'R@10':>6} {'95% CI':>16}   vs top (R@5)")
    for r in rows:
        c5 = f"[{r['ci95_r5'][0]:.2f}, {r['ci95_r5'][1]:.2f}]"
        c10 = f"[{r['ci95_r10'][0]:.2f}, {r['ci95_r10'][1]:.2f}]"
        t = r["vs_top_r5"]
        vs = "—" if r["embedder"] == top else (
            f"{t['diff']:+.2f} [{t['ci95'][0]:+.2f},{t['ci95'][1]:+.2f}] "
            f"p={t['p']:.3f} {'SIG' if t['significant'] else 'ns'}")
        print(f"{r['embedder']:{w}s} {r['license']:15s} {r['recall_at_5']:6.2f} {c5:>16} "
              f"{r['recall_at_10']:6.2f} {c10:>16}   {vs}")

    out = f"{R}/panel_confidence_intervals.json"
    json.dump({"n_questions": n, "bootstrap_resamples": B, "seed": SEED,
               "corpus_format": "title+text (matched across all entrants)",
               "top_embedder": top, "rows": rows}, open(out, "w"), indent=1)
    print(f"\nsaved {out}")

    ns = [r["embedder"] for r in rows if r["embedder"] != top
          and not r["vs_top_r5"]["significant"]]
    print(f"\nStatistically indistinguishable from {top} at 95%: "
          f"{', '.join(ns) if ns else 'none'}")


if __name__ == "__main__":
    main()
