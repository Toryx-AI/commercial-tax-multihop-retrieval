#!/usr/bin/env python3
"""BGE-M3 Recall@5/@10 on the exact Paper-1 MuSiQue harness (11,656 passages, 1,000 dev
questions, data/hipporag2/*.json — the same files/gold-computation used for every other
embedder in the panel). Dense-vector-only, no sparse/multi-vector, for apples-to-apples
comparability with the rest of the panel. Tests both the model's own documented default
(no instruction prefix) and an instruction-prefixed variant, consistent with how every
other open-weight embedder in this panel was tested (best-of-variants).
"""
import json, time
import numpy as np
from FlagEmbedding import BGEM3FlagModel

DATA = "data/hipporag2"
CORPUS_CACHE = "data/results/embeddings/bge_m3/corpus_emb.npy"


def recall(query_embs, corpus_embs, gold_list, ks=(5, 10)):
    sims = query_embs @ corpus_embs.T
    res = {}
    for k in ks:
        pq = []
        for i in range(len(query_embs)):
            g = set(gold_list[i])
            if not g:
                continue
            topk = set(np.argsort(-sims[i])[:k].tolist())
            pq.append(len(topk & g) / len(g))
        res[k] = float(np.mean(pq) * 100)
    return res


def main():
    corpus = json.load(open(f"{DATA}/musique_corpus.json"))
    queries = json.load(open(f"{DATA}/musique.json"))
    qtext = [q["question"] for q in queries]
    print(f"corpus={len(corpus)} queries={len(queries)}")

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
    n_missing = sum(1 for g in gold if not g)
    print(f"gold computed, {n_missing} queries with 0 gold passages")

    print("\nloading BAAI/bge-m3 (fp16)...")
    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

    import os
    if os.path.exists(CORPUS_CACHE):
        print(f"loading cached corpus embeddings from {CORPUS_CACHE}")
        corpus_emb = np.load(CORPUS_CACHE)
    else:
        print("embedding corpus (title+text, official default, no instruction)...")
        t0 = time.time()
        passage_texts = [f'{p["title"]}\n{p["text"]}' for p in corpus]
        corpus_emb = model.encode(passage_texts, max_length=1024, batch_size=64)["dense_vecs"]
        os.makedirs(os.path.dirname(CORPUS_CACHE), exist_ok=True)
        np.save(CORPUS_CACHE, corpus_emb)
        print(f"  corpus embedded in {time.time()-t0:.0f}s, shape={corpus_emb.shape}")

    variants = [
        ("bge-m3  bare query (official default)", qtext),
        ("bge-m3  instruction-prefixed", [f"Represent this sentence for searching relevant passages: {t}" for t in qtext]),
    ]

    rows = []
    for name, texts in variants:
        t0 = time.time()
        q_emb = model.encode(texts, max_length=512, batch_size=64)["dense_vecs"]
        r = recall(q_emb, corpus_emb, gold)
        print(f"  {name:42s} R@5={r[5]:.2f} R@10={r[10]:.2f}  ({time.time()-t0:.0f}s)")
        rows.append((name, r[5], r[10]))

    rows.sort(key=lambda x: -x[1])
    best = rows[0]
    print("\n=== BGE-M3 SUMMARY (best-of-variants, sorted by R@5) ===")
    for name, r5, r10 in rows:
        print(f"{name:42s} R@5={r5:6.2f} R@10={r10:6.2f}")
    print(f"\nBest variant: {best[0]} -> R@5={best[1]:.2f} R@10={best[2]:.2f}")
    print("\nFor reference, from the paper's panel:")
    print(f"{'Nemotron-3-Embed-8B (current leader)':42s} R@5= 69.69 R@10= 77.45")
    print(f"{'NV-Embed-v2 (non-commercial anchor)':42s} R@5= 67.10 R@10= 76.20")

    out = {
        "dataset": "musique", "n_queries": len(queries), "corpus_size": len(corpus),
        "model": "BAAI/bge-m3", "license": "MIT", "mode": "dense-only (no sparse/multi-vector)",
        "variants": [{"variant": n, "recall_at_5": r5, "recall_at_10": r10} for n, r5, r10 in rows],
        "best_variant": {"variant": best[0], "recall_at_5": best[1], "recall_at_10": best[2]},
    }
    json.dump(out, open("data/results/bge_m3_recall_musique.json", "w"), indent=2)
    print("\nsaved -> data/results/bge_m3_recall_musique.json")


if __name__ == "__main__":
    main()
