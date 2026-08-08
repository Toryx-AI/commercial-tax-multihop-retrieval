#!/usr/bin/env python3
"""Paired intervals for the claims v8 asserted without them.

Round 3 of the adversarial review found several places where the paper makes a
paired inferential claim and reports no uncertainty -- in a paper whose own
methodological argument is that "a field that reports point estimates without
intervals has no mechanism by which an error of exactly this shape would ever
surface." Fixing that with prose would be answering the criticism with the
behaviour it criticises.

Computes, from the saved per-question vectors:

  1. Recall@10 - Recall@5 lift, per entrant, with paired bootstrap CIs. v8 claims
     "every one of the thirteen gains" -- thirteen paired claims, zero intervals.
  2. The title+text vs text-only format penalty, per entrant, with paired CIs.
     v8 says format "penalises EVERY model ... by between 0.79 and 4.73 points",
     but 0.79 (text-embedding-3-large) is barely above that same model's 0.54
     run-to-run drift, so the universal quantifier needs checking at the low end.
  3. TOST equivalence sample size, done with the correct constants and with the
     assumption stated. v8 cites ~11,400, which back-solves to no standard
     formula; it corresponds to neither the true-difference-zero nor the
     observed-difference case.
  4. Panel p-values at raised B, because at B=10,000 the reported "exact"
     0.0002/0.0004 are literally the two-sided Monte-Carlo floor (2/B, 4/B) and
     carry no ordering information -- Voyage at 15.71 points behind gets the same
     p as Gemini at 2.55 behind.

Usage
-----
    python3 scripts/panel_intervals.py
    python3 scripts/panel_intervals.py --resamples 100000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
RES = REPO / "results"   # release copy: layout adapted from the research repo

# Reuse the panel definition rather than restating it -- one source of truth for
# which file and key each entrant's vector comes from.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from panel_multiplicity import PANEL, load  # noqa: E402

# Entrants measured in both corpus formats, and where the text-only arm lives.
FORMATS = [
    ("Nemotron-3-Embed-8B",        "nemotron_variant_sweep_perq.json", "title+text",            "text-only"),
    ("NV-Embed-v2",                "nvembed_variant_sweep_perq.json",  "web-search|title+text", "web-search|text-only"),
    ("Gemini embedding-001",       "api_panel_sweep_perq.json",        "gemini|title+text",     "gemini|text-only"),
    ("Nemotron-3-Embed-1B",        "api_panel_sweep_perq.json",        "nim-nemotron-1b|title+text",          "nim-nemotron-1b|text-only"),
    ("Llama-Nemotron-Embed-1B-v2", "api_panel_sweep_perq.json",        "nim-llama-nemotron-1b-v2|title+text", "nim-llama-nemotron-1b-v2|text-only"),
    ("Cohere Embed v4",            "api_panel_sweep_perq.json",        "cohere|title+text",     "cohere|text-only"),
    ("text-embedding-3-large",     "api_panel_sweep_perq.json",        "openai-large|title+text", "openai-large|text-only"),
    ("nv-embedqa-e5-v5",           "api_panel_sweep_perq.json",        "nim-e5-v5|title+text",  "nim-e5-v5|text-only"),
    ("text-embedding-3-small",     "api_panel_sweep_perq.json",        "openai-small|title+text", "openai-small|text-only"),
    ("voyage-3.5",                 "api_panel_sweep_perq.json",        "voyage|title+text",     "voyage|text-only"),
]


def load_metric(fname: str, key: str, metric: str) -> np.ndarray:
    d = json.loads((RES / fname).read_text())
    if "results" in d and isinstance(d["results"], list):
        for e in d["results"]:
            if e.get("model") == key:
                return np.asarray(e[metric], dtype=float)
        raise KeyError(key)
    return np.asarray(d[key][metric], dtype=float)


def paired_ci(d: np.ndarray, B: int, rng) -> tuple[float, float, float, float]:
    """Paired bootstrap on a per-question difference vector. Returns pts and p."""
    idx = rng.integers(0, len(d), size=(B, len(d)))
    boot = d[idx].mean(axis=1) * 100
    obs = d.mean() * 100
    lo, hi = np.percentile(boot, [2.5, 97.5])
    tail = (boot <= 0).sum() if obs > 0 else (boot >= 0).sum()
    return obs, lo, hi, min(1.0, 2.0 * (tail + 1) / (B + 1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resamples", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/panel_intervals.json")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    out: dict = {"resamples": a.resamples, "seed": a.seed}

    print(f"B = {a.resamples:,}, seed {a.seed}\n")

    print("=" * 78)
    print("1. Recall@10 minus Recall@5 lift, paired CIs")
    print("=" * 78)
    lifts = []
    for name, f, k in PANEL:
        r5 = load_metric(f, k, "r5")
        r10 = load_metric(f, k, "r10")
        obs, lo, hi, p = paired_ci(r10 - r5, a.resamples, rng)
        lifts.append({"model": name, "lift": obs, "lo": lo, "hi": hi, "p": p})
        print(f"  {name:30s} {obs:+6.2f}  [{lo:+.2f}, {hi:+.2f}]  p={p:.5f}")
    span = max(x["lift"] for x in lifts) - min(x["lift"] for x in lifts)
    allpos = all(x["lo"] > 0 for x in lifts)
    print(f"\n  span = {span:.2f} points; every CI strictly above zero: {allpos}")
    out["r10_lift"] = lifts

    print()
    print("=" * 78)
    print("2. Format penalty (title+text minus text-only), paired CIs")
    print("=" * 78)
    pens = []
    for name, f, kt, kx in FORMATS:
        d = load_metric(f, kt, "r5") - load_metric(f, kx, "r5")
        obs, lo, hi, p = paired_ci(d, a.resamples, rng)
        sig = lo > 0
        pens.append({"model": name, "penalty": obs, "lo": lo, "hi": hi, "p": p, "significant": bool(sig)})
        print(f"  {name:30s} {obs:+6.2f}  [{lo:+.2f}, {hi:+.2f}]  p={p:.5f}  {'' if sig else '<-- CI includes 0'}")
    n_sig = sum(x["significant"] for x in pens)
    print(f"\n  {n_sig}/{len(pens)} penalties are significant at 95%.")
    print(f"  'penalises every model' is {'supported' if n_sig == len(pens) else 'NOT supported at the low end'}.")
    out["format_penalty"] = pens

    print()
    print("=" * 78)
    print("3. TOST equivalence sample size, assumption stated")
    print("=" * 78)
    nm = load_metric("nemotron_variant_sweep_perq.json", "title+text", "r5")
    nv = load_metric("nvembed_variant_sweep_perq.json", "web-search|title+text", "r5")
    d = (nm - nv) * 100
    sd, n0 = d.std(ddof=1), len(d)
    obs = d.mean()
    z_a, z_b = 1.6448536269514722, 0.8416212335729143   # one-sided 0.05, power 0.80
    print(f"  observed difference {obs:+.3f}, per-question SD {sd:.3f}, SE at n={n0} is {sd/np.sqrt(n0):.3f}")
    for margin in (0.5, 1.0):
        for assumed, lbl in ((0.0, "true difference = 0"), (abs(obs), f"true difference = observed {abs(obs):.2f}")):
            slack = margin - assumed
            if slack <= 0:
                print(f"  margin +/-{margin}: unattainable under {lbl}")
                continue
            n = ((z_a + z_b) * sd / slack) ** 2
            print(f"  margin +/-{margin}, {lbl:36s} -> n >= {int(np.ceil(n)):,}")
    out["tost"] = {"observed": float(obs), "sd": float(sd), "n": int(n0)}

    (REPO / a.out).write_text(json.dumps(out, indent=2, default=float))
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
