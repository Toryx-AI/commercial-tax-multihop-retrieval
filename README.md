# When Benchmarks Meet Reality — replication artifacts

Harness code and measurement artifacts for *When Benchmarks Meet Reality: The True Cost of
Multi-Hop Retrieval* (Sanchez & Dehnad).

The paper argues that the multi-hop retrieval literature reports quality numbers stripped of the
two things a buyer needs — whether the retrieval backbone is legal to deploy commercially, and what
it costs to build. A paper making that argument owes you its own harness. This is it.

Everything runs against the public HippoRAG-2 MuSiQue split (11,656 passages, first 1,000 dev
questions). The corpus is not redistributed; fetch instructions are below.

## The result

All thirteen embedders, one harness, every passage embedded as `title\ntext`. Confidence intervals
are percentile bootstrap over the 1,000 evaluation questions, 10,000 resamples, seed 42.

| Embedder | Licence | R@5 | 95% CI | R@10 | vs top |
|---|---|---|---|---|---|
| Nemotron-3-Embed-8B | Commercial | 69.79 | [68.11, 71.44] | 77.54 | — |
| NV-Embed-v2 | **Non-commercial** | 69.55 | [67.92, 71.22] | 78.12 | +0.24, p=0.706 **ns** |
| Gemini embedding-001 | Commercial | 67.24 | [65.49, 69.01] | 76.35 | +2.55, p=0.001 sig |
| Nemotron-3-Embed-1B | Commercial | 64.32 | [62.64, 65.97] | 72.79 | +5.47 sig |
| Llama-Nemotron-Embed-1B-v2 | Commercial | 63.73 | [62.02, 65.42] | 71.35 | +6.06 sig |
| Cohere Embed v4 | Commercial | 60.21 | [58.44, 61.97] | 69.08 | +9.58 sig |
| Qwen3-VL-Embedding-8B | Free/open | 59.88 | [58.15, 61.60] | 68.53 | +9.91 sig |
| OpenAI text-embedding-3-large | Commercial | 59.48 | [57.69, 61.27] | 70.15 | +10.31 sig |
| nv-embedqa-e5-v5 | Commercial | 57.69 | [55.97, 59.35] | 66.27 | +12.10 sig |
| mxbai-embed-large-v1 | Free/open | 55.71 | [53.87, 57.54] | 64.35 | +14.08 sig |
| OpenAI text-embedding-3-small | Commercial | 55.38 | [53.62, 57.12] | 64.78 | +14.42 sig |
| BGE-M3 | Free/open | 54.93 | [53.23, 56.59] | 62.89 | +14.87 sig |
| Voyage voyage-3.5 | Commercial | 54.08 | [52.27, 55.92] | 63.68 | +15.71 sig |

**NV-Embed-v2 is the only entrant statistically indistinguishable from the top model.** The
commercial tax, measured against the non-commercial anchor:

| | diff | 95% CI | p | |
|---|---|---|---|---|
| Best commercial **before** Nemotron (Gemini embedding-001) | +2.31 | [+0.91, +3.69] | 0.002 | **significant** |
| Best commercial **after** Nemotron (Nemotron-3-Embed-8B) | −0.24 | [−1.41, +0.93] | 0.706 | not significant |

The tax was real and recent. It has closed at the frontier — not across the category: eleven of
twelve commercially-deployable entrants are still significantly behind.

## The correction this repository documents

An earlier draft reported NV-Embed-v2 at **67.1** R@5 and treated the gap against its published
69.4–69.7 as a finding about harness variance. It was our bug.

`code/build_nvembed_index.py` embeds passages as **text only** (line 54,
`texts = [p["text"] for p in corpus]`) while every other entrant received `"{title}\n{text}"`. On
MuSiQue the multi-hop chain runs through Wikipedia entity titles, so dropping them costs ~2.5 R@5
points — and it was dropped for exactly one model, the anchor the headline claim is asserted
against.

We re-ran the whole panel to check. Twelve of thirteen reproduce their original figures on
`title\ntext`, three to the decimal:

| Embedder | title+text | Δ vs draft | text-only |
|---|---|---|---|
| Gemini embedding-001 | 67.24 | **+0.00** | 65.78 |
| Llama-Nemotron-Embed-1B-v2 | 63.73 | **+0.00** | 59.81 |
| Nemotron-3-Embed-1B | 64.32 | **−0.00** | 60.86 |
| mxbai-embed-large-v1 | 55.71 | +0.01 | — |
| Qwen3-VL-Embedding-8B | 59.88 | −0.02 | — |
| BGE-M3 | 54.93 | +0.06 | — |
| Cohere Embed v4 | 60.21 | +0.08 | 57.92 |
| nv-embedqa-e5-v5 | 57.69 | +0.09 | 52.96 |
| Nemotron-3-Embed-8B | 69.79 | +0.10 | 67.25 |
| voyage-3.5 | 54.08 | +0.11 | 51.17 |
| OpenAI 3-small | 55.38 | +0.19 | 53.48 |
| OpenAI 3-large | 59.48 | −0.54 | 58.69 |
| **NV-Embed-v2** | **69.55** | **+2.48** | 67.07 |

Query-instruction choice does *not* explain the gap: across the text-only corpus every instructed
variant lands 66.6–67.1, and the original harness already used the best of them
(`results/nvembed_variant_sweep.json`).

The defective indexing script is published here unmodified, alongside the corrected one.

## Reproducing

Fetch the HippoRAG-2 MuSiQue split from the upstream project and place `musique_corpus.json` and
`musique.json` under `data/hipporag2/`.

NV-Embed-v2 and Nemotron-3-Embed-8B need different transformers major versions and cannot share an
environment:

```bash
# NV-Embed-v2 requires transformers 4.42.4 (custom modelling code)
pip install --target .tf442 "transformers==4.42.4"
PYTHONPATH=.tf442 python3 code/nvembed_variant_sweep.py

# Nemotron-3-Embed-8B requires transformers >= 5.2.0 (ministral3 architecture)
pip install --target .tf520 "transformers>=5.2.0" "sentence-transformers>=5.0.0"
rm -rf .tf520/torch*          # fall back to the system torch build
PYTHONPATH=.tf520 python3 code/nemotron_variant_sweep.py

# Hosted providers (OpenAI, Gemini, Voyage, Cohere/Bedrock, NVIDIA NIM). Resumable.
python3 code/api_embedder_panel_sweep.py

# Qwen3-VL / mxbai / BGE-M3 — CPU by default, they are small
EMBED_DEVICE=cpu python3 code/selfhosted_panel_perq.py

# Statistics (no GPU)
python3 code/paired_bootstrap_headline.py
python3 code/panel_confidence_intervals.py
```

GPU work ran on a single NVIDIA RTX A6000 (48GB). Both GPU models were measured on the same device
deliberately: the headline difference is 0.24 points, inside the range hardware and precision
differences can produce on their own.

Embedder conventions are not interchangeable:

- **NV-Embed-v2** — passages carry no instruction; queries carry `"Instruct: {task}\nQuery: "`.
- **Nemotron-3-Embed** — asymmetric prefixes, `encode_query` (`"query: "`) and `encode_document`
  (`"passage: "`). Mismatching the sides degrades retrieval silently.
- **Hosted providers** — each provider's own asymmetric mode (Cohere `input_type`, Gemini
  `task_type`, Voyage `input_type`, NIM `input_type`).
- **BGE-M3** — dense mode only here; no instruction prefix required.

NVIDIA's catalog identifiers are not stable across surfaces: `nemotron-3-embed-8b` 404s on the
hosted NIM endpoint (the 8B is self-host-only), and the HuggingFace name
`nemotron-3-embed-1b-bf16` 404s there too — the working NIM id is `nemotron-3-embed-1b`.

## Reader configuration

Every Answer F1 figure comes from `gpt-4o-mini`, one call per question, `temperature=0.0`,
`max_tokens=32`, single sample, no self-consistency. Passages serialised as `[i] {title}: {text}`
joined by blank lines. Prompts are in `code/eval_bgem3_reader_f1.py`. Scoring is max SQuAD
token-overlap F1 over the gold answer and its aliases.

## Files

| Path | What |
|---|---|
| `results/panel_confidence_intervals.json` | all 13 embedders, bootstrap CIs, paired tests vs top |
| `results/headline_paired_bootstrap.json` | Nemotron vs NV-Embed-v2, CIs and per-question win/loss/tie |
| `results/nvembed_variant_sweep.json` (+`_perq`) | NV-Embed-v2, 2 corpus formats × 4 query instructions |
| `results/nvembed_query_sweep.json` (+`_perq`) | instruction sweep, text-only corpus |
| `results/nemotron_variant_sweep.json` (+`_perq`) | Nemotron-3-Embed-8B, both corpus formats |
| `results/api_panel_sweep.json` (+`_perq`) | 8 hosted providers, both corpus formats |
| `results/selfhosted_panel_perq.json` (+`_vectors`) | Qwen3-VL, mxbai, BGE-M3 |
| `results/diag_embedder_recall_musique.json` | Qwen3-VL / mxbai query-format sweep (the 11.6-point swing) |
| `results/bge_m3_recall_musique.json`, `..._reader_f1_...` | BGE-M3 recall, Answer F1, measured API cost |

**Per-question files** (`*_perq*.json`) hold one recall fraction per scored question, in corpus
order, for every entrant. Any confidence interval or paired test in the paper can be recomputed
from these without a GPU — and comparisons we did not think to run are available to you.

Embedding matrices (`.npy`, 2.2GB total) are not in git. See the companion HuggingFace dataset.

## Licence

Code in `code/` is MIT (`LICENSE`). Measurement outputs in `results/` are CC BY 4.0
(`LICENSE-DATA`). Neither covers the upstream MuSiQue corpus or any third-party model weights,
which carry their own terms — including NV-Embed-v2, which is `cc-by-nc-4.0` and is the subject of
this paper.
