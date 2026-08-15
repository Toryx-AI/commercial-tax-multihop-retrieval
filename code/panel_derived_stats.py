#!/usr/bin/env python3
"""panel_derived_stats.py -- every derived statistic Paper 1 quotes that is not already a
row in another released artifact, computed from the released per-question vectors.

Reviewers could reproduce the paired intervals and p-values from the artifacts but not
these, because nothing printed them:

  * the paired SD / SE behind the MDE and TOST sample sizes (S4.2, S4.3)
  * the minimum detectable effect at 80% power
  * the two TOST sample sizes (true difference 0, and the observed +0.24)
  * per-question win / loss / tie counts for the headline pair
  * the mean pairwise per-question Recall@5 correlation across the panel and the
    correlation between the two leaders (Figure 2 caption)
  * the bootstrap-free lift standard errors quoted for NV-Embed-v2 and nv-embedqa-e5-v5
    (Appendix D caption), which show the two identical lift rows are distinct vectors

Everything is a closed-form function of the released vectors; no resampling, no seed.

  python3 scripts/panel_derived_stats.py
"""
from __future__ import annotations

import itertools
import json

import numpy as np
from scipy.stats import norm

R = "data/results"
ALPHA = 0.05
POWER = 0.80
TOST_MARGIN = 0.5           # Recall@5 points, the buyer-relevant margin used in S4.2
TOP = "Nemotron-3-Embed-8B"
ANCHOR = "NV-Embed-v2"


def vec(x, k="r5") -> np.ndarray:
    x = x[k] if isinstance(x, dict) else x
    x = np.asarray(x, dtype=float)
    return x * 100 if x.mean() <= 1.0 else x


def load_panel() -> dict[str, dict[str, np.ndarray]]:
    nv = json.load(open(f"{R}/nvembed_variant_sweep_perq.json"))["web-search|title+text"]
    ne = json.load(open(f"{R}/nemotron_variant_sweep_perq.json"))["title+text"]
    api = json.load(open(f"{R}/api_panel_sweep_perq.json"))
    sh = json.load(open(f"{R}/selfhosted_panel_perq_vectors.json"))
    api_names = {"openai-small": "OpenAI text-embedding-3-small",
                 "openai-large": "OpenAI text-embedding-3-large",
                 "gemini": "Gemini embedding-001", "voyage": "Voyage voyage-3.5",
                 "cohere": "Cohere Embed v4", "nim-e5-v5": "nv-embedqa-e5-v5",
                 "nim-nemotron-1b": "Nemotron-3-Embed-1B",
                 "nim-llama-nemotron-1b-v2": "Llama-Nemotron-Embed-1B-v2"}
    sh_names = {"qwen3vl": "Qwen3-VL-Embedding-8B", "mxbai": "mxbai-embed-large-v1",
                "bgem3": "BGE-M3"}
    panel = {ANCHOR: {"r5": vec(nv, "r5"), "r10": vec(nv, "r10")},
             TOP: {"r5": vec(ne, "r5"), "r10": vec(ne, "r10")}}
    for k, name in api_names.items():
        src = api[f"{k}|title+text"]
        panel[name] = {"r5": vec(src, "r5"), "r10": vec(src, "r10")}
    for k, name in sh_names.items():
        panel[name] = {"r5": vec(sh[k], "r5"), "r10": vec(sh[k], "r10")}
    n = {len(v["r5"]) for v in panel.values()} | {len(v["r10"]) for v in panel.values()}
    assert n == {1000}, f"vector lengths differ: {n}"
    return panel


def main() -> None:
    panel = load_panel()
    a, b = panel[ANCHOR]["r5"], panel[TOP]["r5"]
    d = b - a
    sd = float(d.std(ddof=1))
    se = sd / np.sqrt(len(d))
    z = norm.ppf
    out = {
        "n": int(len(d)),
        "headline_pair": {
            "top": TOP, "anchor": ANCHOR, "metric": "Recall@5",
            "mean_diff_top_minus_anchor": float(d.mean()),
            "paired_sd": sd, "paired_se": float(se),
            "wins_top": int((d > 0).sum()), "wins_anchor": int((d < 0).sum()),
            "ties": int((d == 0).sum()),
        },
        "power": {
            "alpha": ALPHA, "power": POWER,
            "mde_two_sided": float((z(1 - ALPHA / 2) + z(POWER)) * se),
            "tost_margin": TOST_MARGIN,
            # each one-sided test at alpha, so z(1-alpha) not z(1-alpha/2); at a true
            # difference of zero both one-sided tests must pass, so the power term is
            # z(1-(1-POWER)/2) = z(0.90), not z(0.80) -- see S4.2
            "tost_n_true_diff_zero": float(((z(1 - ALPHA) + z(1 - (1 - POWER) / 2)) * sd
                                            / TOST_MARGIN) ** 2),
            "tost_n_true_diff_observed": float(((z(1 - ALPHA) + z(POWER)) * sd
                                                / (TOST_MARGIN - abs(d.mean()))) ** 2),
        },
    }
    names = sorted(panel)
    M = np.array([panel[n]["r5"] for n in names])
    C = np.corrcoef(M)
    pairs = list(itertools.combinations(range(len(names)), 2))
    out["per_question_r5_correlation"] = {
        "mean_pairwise": float(np.mean([C[i, j] for i, j in pairs])),
        "leaders": float(C[names.index(ANCHOR), names.index(TOP)]),
        "note": "Pearson over the 1,000 per-question Recall@5 scores; distinct from the "
                "difference-statistic correlation in panel_simultaneous_ci.json",
    }
    lifts = {}
    for name in names:
        l = panel[name]["r10"] - panel[name]["r5"]
        lifts[name] = {"mean": float(l.mean()), "se": float(l.std(ddof=1) / np.sqrt(len(l)))}
    out["r10_minus_r5_lift"] = lifts

    json.dump(out, open(f"{R}/panel_derived_stats.json", "w"), indent=1)
    h, p = out["headline_pair"], out["power"]
    print(f"{TOP} - {ANCHOR} @5: {h['mean_diff_top_minus_anchor']:+.4f}  "
          f"SD {h['paired_sd']:.2f}  SE {h['paired_se']:.3f}  "
          f"wins/losses/ties {h['wins_top']}/{h['wins_anchor']}/{h['ties']}")
    print(f"MDE (80% power, two-sided 5%): {p['mde_two_sided']:.2f}")
    print(f"TOST n at margin +/-{TOST_MARGIN}: {p['tost_n_true_diff_zero']:,.0f} (true diff 0), "
          f"{p['tost_n_true_diff_observed']:,.0f} (true diff observed)")
    c = out["per_question_r5_correlation"]
    print(f"per-question R@5 correlation: panel mean {c['mean_pairwise']:.3f}, leaders {c['leaders']:.3f}")
    for name in (ANCHOR, "nv-embedqa-e5-v5"):
        print(f"lift {name}: {lifts[name]['mean']:.3f} SE {lifts[name]['se']:.3f}")
    print(f"saved {R}/panel_derived_stats.json")


if __name__ == "__main__":
    main()
