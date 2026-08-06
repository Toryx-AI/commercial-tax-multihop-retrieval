#!/usr/bin/env python3
"""nvembed_variant_sweep.py — audit NV-Embed-v2's MuSiQue Recall@5/@10 across the two
harness choices that the Paper 1 panel never varied for this model.

Why this exists
---------------
Paper 1 (PRECURSOR_commercial_tax_v5) reports NV-Embed-v2 at 67.1 Recall@5 on our own
harness, 2.3-2.6 points below its widely-cited 69.4-69.7, and the headline claim
("a commercial embedder now beats the research anchor") rests entirely on that gap.
Two harness choices were never swept for NV-Embed-v2 specifically:

  1. CORPUS FORMAT. build_nvembed_index.py embeds passages as text-only (line 54:
     `texts = [p["text"] for p in corpus]`), while every commercial embedder in the panel
     embeds `f"{title}\\n{text}"` (diag_embedder_recall.py, commercial_embed_retrieve.py).
     On MuSiQue the title carries entity signal, so this plausibly costs NV-Embed-v2 recall
     and is an apples-to-apples violation.
  2. QUERY INSTRUCTION. nvembed_encode_retrieve.py hardcodes ONE instruction. The panel's
     stated protocol ("tested multiple query-formatting variants and took the best") was
     applied to Qwen3-VL and mxbai (diag_embedder_recall.py) but not to NV-Embed-v2.

This script runs the full 2 (corpus format) x N (query instruction) factorial and saves
PER-QUESTION recall vectors so paired bootstrap CIs can be computed downstream.

Run (local A6000 or Spark):
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
    python3 scripts/nvembed_variant_sweep.py \\
      --out data/results/nvembed_variant_sweep.json

  --skip-title-corpus   reuse only the cached text-only corpus (query sweep only, no re-embed)
  --title-corpus-cache  where to cache the title+text corpus embedding (built once, ~30 min)
"""
from __future__ import annotations

import argparse, json, os, time
import numpy as np
import torch

from transformers import DynamicCache
if not hasattr(DynamicCache, "get_usable_length"):
    DynamicCache.get_usable_length = lambda self, *a, **kw: self.get_seq_length()

MODEL_NAME = "nvidia/NV-Embed-v2"
DATA = "data/hipporag2"
TEXT_ONLY_CACHE = "data/results/embeddings/nvembed/musique_corpus_emb.npy"
TITLE_TEXT_CACHE = "data/results/embeddings/nvembed/musique_corpus_emb_titletext.npy"

# NV-Embed-v2's documented usage wraps the task description as
#   "Instruct: {task}\nQuery: "
# and passes it via model.encode(instruction=...). Variants differ only in {task}.
QUERY_VARIANTS = [
    ("harness-original",
     "Instruct: Given a question, retrieve passages that answer the question\nQuery: "),
    ("multihop-qa",
     "Instruct: Given a multi-hop question, retrieve documents that can help answer the question\nQuery: "),
    ("web-search",
     "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "),
    ("no-instruction", ""),
]


def load_model(retries=6, wait=45):
    from transformers import AutoModel
    import transformers as _tf
    _v = tuple(int(x) for x in _tf.__version__.split(".")[:2])
    _dk = "dtype" if _v >= (4, 56) else "torch_dtype"
    kw = {"trust_remote_code": True, _dk: torch.bfloat16, "low_cpu_mem_usage": True}
    for i in range(retries):
        try:
            m = AutoModel.from_pretrained(MODEL_NAME, **kw)
            # custom NV-Embed loader ignores dtype -> lands fp32; cast on CPU THEN move.
            m = m.to(torch.bfloat16).cuda()
            m.eval()
            return m
        except (getattr(torch.cuda, "OutOfMemoryError", RuntimeError), RuntimeError) as e:
            if "out of memory" not in str(e).lower() or i == retries - 1:
                raise
            torch.cuda.empty_cache()
            print(f"[oom] load attempt {i+1} failed; wait {wait}s", flush=True)
            time.sleep(wait)


def encode(model, texts, instruction, max_length, batch_size, label=""):
    embs = []
    t0 = time.time()
    for i in range(0, len(texts), batch_size):
        with torch.no_grad():
            e = model.encode(texts[i:i + batch_size], instruction=instruction,
                             max_length=max_length)
        if isinstance(e, torch.Tensor):
            e = e.float().cpu().numpy()
        embs.append(e)
        done = min(i + batch_size, len(texts))
        if (i // batch_size) % 50 == 0 and i:
            rate = done / (time.time() - t0)
            print(f"    [{label}] {done}/{len(texts)} ({rate:.1f}/s)", flush=True)
    a = np.vstack(embs)
    n = np.linalg.norm(a, axis=1, keepdims=True); n[n == 0] = 1e-10
    return (a / n).astype(np.float32)


def per_question_recall(query_embs, corpus_embs, gold, ks=(5, 10)):
    """Return {k: [per-question recall fraction, ...]} — the vector, not just the mean,
    so downstream paired bootstrap CIs are possible."""
    sims = query_embs @ corpus_embs.T
    order = np.argsort(-sims, axis=1)[:, :max(ks)]
    out = {}
    for k in ks:
        vec = []
        for i in range(len(query_embs)):
            g = set(gold[i])
            if not g:
                continue
            vec.append(len(set(order[i, :k].tolist()) & g) / len(g))
        out[k] = vec
    return out


def build_gold(corpus, queries):
    tmap = {}
    for idx, p in enumerate(corpus):
        tmap.setdefault(p["text"], []).append(idx)
    gold = []
    for q in queries:
        gi = set()
        for p in q["paragraphs"]:
            if p.get("is_supporting") and p["paragraph_text"] in tmap:
                gi.update(tmap[p["paragraph_text"]])
        gold.append(list(gi))
    return gold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/results/nvembed_variant_sweep.json")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--query-max-length", type=int, default=128)
    ap.add_argument("--passage-max-length", type=int, default=512)
    ap.add_argument("--skip-title-corpus", action="store_true")
    ap.add_argument("--title-corpus-cache", default=TITLE_TEXT_CACHE)
    a = ap.parse_args()

    corpus = json.load(open(f"{DATA}/musique_corpus.json"))
    queries = json.load(open(f"{DATA}/musique.json"))
    qtext = [q["question"] for q in queries]
    gold = build_gold(corpus, queries)
    n_scored = sum(1 for g in gold if g)
    print(f"corpus={len(corpus)} queries={len(qtext)} scored={n_scored}", flush=True)

    model = load_model()
    print("model loaded", flush=True)

    corpora = {}
    corpora["text-only"] = np.load(TEXT_ONLY_CACHE)
    print(f"[corpus] text-only cached {corpora['text-only'].shape}", flush=True)

    if not a.skip_title_corpus:
        if os.path.exists(a.title_corpus_cache):
            corpora["title+text"] = np.load(a.title_corpus_cache)
            print(f"[corpus] title+text cached {corpora['title+text'].shape}", flush=True)
        else:
            print("[corpus] embedding title+text (one time, slow)...", flush=True)
            t0 = time.time()
            ptexts = [f'{p["title"]}\n{p["text"]}' for p in corpus]
            emb = encode(model, ptexts, "", a.passage_max_length, a.batch_size, "corpus")
            os.makedirs(os.path.dirname(a.title_corpus_cache), exist_ok=True)
            np.save(a.title_corpus_cache, emb)
            corpora["title+text"] = emb
            print(f"[corpus] done in {time.time()-t0:.0f}s {emb.shape}", flush=True)

    results, perq = [], {}
    for vname, instr in QUERY_VARIANTS:
        print(f"\n[query] {vname}", flush=True)
        qe = encode(model, qtext, instr, a.query_max_length, a.batch_size, vname)
        for cname, C in corpora.items():
            r = per_question_recall(qe, C, gold)
            row = {
                "query_variant": vname,
                "instruction": instr,
                "corpus_format": cname,
                "recall_at_5": float(np.mean(r[5]) * 100),
                "recall_at_10": float(np.mean(r[10]) * 100),
                "n_scored": len(r[5]),
            }
            results.append(row)
            perq[f"{vname}|{cname}"] = {"r5": r[5], "r10": r[10]}
            print(f"  {cname:12s} R@5={row['recall_at_5']:.2f}  R@10={row['recall_at_10']:.2f}",
                  flush=True)

    results.sort(key=lambda x: -x["recall_at_5"])
    print("\n=== SWEEP SUMMARY (sorted by R@5) ===")
    print(f"{'query variant':20s} {'corpus':12s} {'R@5':>7} {'R@10':>7}")
    for r in results:
        print(f"{r['query_variant']:20s} {r['corpus_format']:12s} "
              f"{r['recall_at_5']:7.2f} {r['recall_at_10']:7.2f}")

    payload = {
        "model": MODEL_NAME,
        "dataset": "musique",
        "corpus_size": len(corpus),
        "n_queries": len(qtext),
        "n_scored": n_scored,
        "paper_reported": {"recall_at_5": 67.1, "recall_at_10": 76.2},
        "literature_bar": {"recall_at_5_range": [69.4, 69.7]},
        "query_max_length": a.query_max_length,
        "passage_max_length": a.passage_max_length,
        "results": results,
        "best": results[0],
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(payload, open(a.out, "w"), indent=1)
    perq_path = a.out.replace(".json", "_perq.json")
    json.dump(perq, open(perq_path, "w"))
    print(f"\nsaved {a.out}\nsaved {perq_path}", flush=True)


if __name__ == "__main__":
    main()
