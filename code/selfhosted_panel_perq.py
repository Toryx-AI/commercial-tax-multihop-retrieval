#!/usr/bin/env python3
"""selfhosted_panel_perq.py — recover per-question recall vectors for the three self-hosted
panel entrants whose original runs saved aggregates only: Qwen3-VL-Embedding-8B,
mxbai-embed-large-v1 and BGE-M3.

Without per-question vectors no confidence interval or paired test can be computed for these
rows, which is why they were the last gap in the panel-wide CI column (see
docs/paper/ARTIFACT_MANIFEST_2026-08-05.md).

Each model's corpus embedding is already cached, so only the 1,000 queries are re-embedded.
Reproducing the published Recall@5 against the cached corpus is itself the check that the
cached corpus is the one that produced the paper's number.

Best query variants are carried over from the original runs (scripts/diag_embedder_recall.py,
scripts/eval_bgem3_recall.py), which swept formats and took the best:
  Qwen3-VL  - no instruction
  mxbai     - no prefix
  BGE-M3    - "Represent this sentence for searching relevant passages: " prefix

  python3 scripts/selfhosted_panel_perq.py --models qwen3vl,mxbai,bgem3
"""
from __future__ import annotations

import argparse, json, os, time, urllib.request
import numpy as np

DATA = "data/hipporag2"
QWEN_URL = os.environ.get("QWEN_EMBED_URL", "http://localhost:9500")
CACHES = {
    "qwen3vl": ("data/results/embeddings/kms_qwen3vl/corpus_emb.npy", (59.9, 68.5)),
    "mxbai":   ("data/results/embeddings/kms_mxbai/corpus_emb.npy",   (55.7, 64.4)),
    "bgem3":   ("data/results/embeddings/bge_m3/corpus_emb.npy",      (54.87, 62.87)),
}


def l2(a):
    a = np.asarray(a, dtype=np.float32)
    n = np.linalg.norm(a, axis=1, keepdims=True); n[n == 0] = 1e-10
    return a / n


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


def per_question_recall(qe, ce, gold, ks=(5, 10)):
    sims = qe @ ce.T
    order = np.argsort(-sims, axis=1)[:, :max(ks)]
    out = {}
    for k in ks:
        vec = []
        for i in range(len(qe)):
            g = set(gold[i])
            if not g:
                continue
            vec.append(len(set(order[i, :k].tolist()) & g) / len(g))
        out[k] = vec
    return out


def embed_qwen_http(texts, bs=32, label="qwen"):
    out = []
    for i in range(0, len(texts), bs):
        body = json.dumps({"texts": texts[i:i + bs], "instruction": ""}).encode()
        req = urllib.request.Request(QWEN_URL.rstrip("/") + "/embed", data=body,
                                     headers={"Content-Type": "application/json"})
        for attempt in range(4):
            try:
                out.extend(json.load(urllib.request.urlopen(req, timeout=240))["vectors"])
                break
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2 * (attempt + 1))
        if (i // bs) % 10 == 0:
            print(f"    [{label}] {min(i+bs,len(texts))}/{len(texts)}", flush=True)
    return out


def embed_mxbai(texts, label="mxbai", device=None):
    """335M params over 1,000 short queries — CPU is fast enough, and the shared GPU on this
    box is routinely full. device=None picks CPU unless EMBED_DEVICE says otherwise."""
    from sentence_transformers import SentenceTransformer
    dev = device or os.environ.get("EMBED_DEVICE", "cpu")
    m = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1", device=dev)
    return m.encode(texts, batch_size=32, show_progress_bar=True)


def embed_bgem3(texts, label="bgem3", device=None):
    """Same reasoning as mxbai (568M). fp16 is GPU-only, so it is off when running on CPU."""
    from FlagEmbedding import BGEM3FlagModel
    dev = device or os.environ.get("EMBED_DEVICE", "cpu")
    m = BGEM3FlagModel("BAAI/bge-m3", use_fp16=(dev != "cpu"), devices=dev)
    return m.encode(texts, max_length=512, batch_size=16)["dense_vecs"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="qwen3vl,mxbai,bgem3")
    ap.add_argument("--out", default="data/results/selfhosted_panel_perq.json")
    a = ap.parse_args()

    corpus = json.load(open(f"{DATA}/musique_corpus.json"))
    queries = json.load(open(f"{DATA}/musique.json"))
    qtext = [q["question"] for q in queries]
    gold = build_gold(corpus, queries)
    print(f"corpus={len(corpus)} queries={len(qtext)} scored={sum(1 for g in gold if g)}",
          flush=True)

    results, perq = [], {}
    if os.path.exists(a.out):
        results = json.loads(open(a.out).read()).get("results", [])
    done = {r["model"] for r in results}

    for name in [m.strip() for m in a.models.split(",") if m.strip()]:
        if name in done:
            print(f"[{name}] already done, skipping", flush=True)
            continue
        cache, (p5, p10) = CACHES[name]
        if not os.path.exists(cache):
            print(f"!! {name}: cached corpus missing at {cache}", flush=True)
            continue
        C = np.load(cache)
        print(f"\n[{name}] corpus cached {C.shape}", flush=True)
        try:
            if name == "qwen3vl":
                qe = l2(embed_qwen_http(qtext))
            elif name == "mxbai":
                qe = l2(embed_mxbai(qtext))
            else:
                qe = l2(embed_bgem3(
                    [f"Represent this sentence for searching relevant passages: {t}"
                     for t in qtext]))
        except Exception as e:
            print(f"!! {name} query embedding failed: {type(e).__name__}: {str(e)[:200]}",
                  flush=True)
            continue

        r = per_question_recall(qe, C, gold)
        row = {"model": name,
               "recall_at_5": float(np.mean(r[5]) * 100),
               "recall_at_10": float(np.mean(r[10]) * 100),
               "paper_reported_r5": p5, "paper_reported_r10": p10,
               "delta_r5_vs_paper": float(np.mean(r[5]) * 100) - p5,
               "n_scored": len(r[5])}
        results.append(row)
        perq[name] = {"r5": r[5], "r10": r[10]}
        print(f"  R@5={row['recall_at_5']:.2f} R@10={row['recall_at_10']:.2f} "
              f"(paper {p5}, {row['delta_r5_vs_paper']:+.2f})", flush=True)

        json.dump({"dataset": "musique", "results": results}, open(a.out, "w"), indent=1)
        pq = a.out.replace(".json", "_vectors.json")
        ex = json.loads(open(pq).read()) if os.path.exists(pq) else {}
        ex.update(perq)
        json.dump(ex, open(pq, "w"))

    print("\n=== SELF-HOSTED PANEL SUMMARY ===")
    for r in results:
        print(f"{r['model']:10s} R@5={r['recall_at_5']:6.2f} R@10={r['recall_at_10']:6.2f} "
              f"paper={r['paper_reported_r5']:6.2f} delta={r['delta_r5_vs_paper']:+.2f}")
    print(f"\nsaved {a.out}")


if __name__ == "__main__":
    main()
