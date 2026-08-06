#!/usr/bin/env python3
"""Embedder/format diagnostic for MuSiQue Recall@5 (issue #102, Step 1b).

Question: is KMS qwen3-VL's R@5=48.2 a *format artifact* or a *real* weakness,
and where does the text-native **mxbai** embedder land? If a home embedder clears
~65+, structure can carry it; if both stay low, NV-Embed-v2 becomes the backend.

Cheap by design: the qwen3-VL **corpus** embedding is cached (data/results/
embeddings/kms_qwen3vl/corpus_emb.npy), so each qwen variant only re-embeds the
1,000 queries. mxbai embeds the corpus once (small/fast model) then reuses it.

Bars (MuSiQue R@5): NV-Embed-v2 69.7 · HippoRAG-2 74.7 (winner).
"""
import json, os, time, urllib.request
import numpy as np

DATA = "data/hipporag2"
QWEN = os.environ.get("QWEN_EMBED_URL", "http://localhost:9500")
MXBAI = os.environ.get("MXBAI_EMBED_URL", "http://localhost:9999")
QWEN_CACHE = "data/results/embeddings/kms_qwen3vl/corpus_emb.npy"
MXBAI_CACHE = "data/results/embeddings/kms_mxbai/corpus_emb.npy"


def embed(texts, url, instruction="", bs=32, label=""):
    out = []
    for i in range(0, len(texts), bs):
        body = json.dumps({"texts": texts[i:i + bs], "instruction": instruction}).encode()
        req = urllib.request.Request(url.rstrip("/") + "/embed", data=body,
                                     headers={"Content-Type": "application/json"})
        for attempt in range(3):
            try:
                out.extend(json.load(urllib.request.urlopen(req, timeout=180))["vectors"]); break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2 * (attempt + 1))
        if (i // bs) % 25 == 0:
            print(f"    [{label}] {min(i+bs,len(texts))}/{len(texts)}", flush=True)
    a = np.asarray(out, dtype=np.float32)
    n = np.linalg.norm(a, axis=1, keepdims=True); n[n == 0] = 1e-10
    return a / n


def recall(query_embs, corpus_embs, gold_list, ks=(2, 5)):
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

    # gold indices by exact passage-text match (0 missing, validated)
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

    rows = []

    # ---------- qwen3-VL: corpus cached, vary the QUERY instruction/format ----------
    qcorpus = np.load(QWEN_CACHE)
    print(f"\n[qwen3-VL] corpus cached {qcorpus.shape} — testing query formats")
    qwen_variants = [
        ("qwen  q=task-instr (baseline)", lambda: embed(qtext, QWEN,
            "Given a question, retrieve passages that answer the question", label="q.base")),
        ("qwen  q=no-instruction", lambda: embed(qtext, QWEN, "", label="q.none")),
        ("qwen  q=MTEB-web-search", lambda: embed(qtext, QWEN,
            "Given a web search query, retrieve relevant passages that answer the query", label="q.mteb")),
        ("qwen  q=Instruct/Query template", lambda: embed(
            [f"Instruct: Given a question, retrieve passages that answer it\nQuery:{t}" for t in qtext],
            QWEN, "", label="q.tmpl")),
    ]
    for name, fn in qwen_variants:
        r = recall(fn(), qcorpus, gold)
        rows.append((name, r[5], r[2])); print(f"  {name:38s} R@5={r[5]:.1f} R@2={r[2]:.1f}")

    # ---------- mxbai: embed corpus once (raw), vary the query prefix ----------
    if os.path.exists(MXBAI_CACHE):
        mcorpus = np.load(MXBAI_CACHE)
        print(f"\n[mxbai] corpus cached {mcorpus.shape}")
    else:
        print("\n[mxbai] embedding corpus (raw title+text)...")
        t0 = time.time()
        mcorpus = embed([f'{p["title"]}\n{p["text"]}' for p in corpus], MXBAI, "", label="mx.corpus")
        os.makedirs(os.path.dirname(MXBAI_CACHE), exist_ok=True)
        np.save(MXBAI_CACHE, mcorpus); print(f"  corpus embedded {time.time()-t0:.0f}s {mcorpus.shape}")
    mxbai_variants = [
        ("mxbai q=retrieval-prefix", lambda: embed(
            [f"Represent this sentence for searching relevant passages: {t}" for t in qtext],
            MXBAI, "", label="mx.pfx")),
        ("mxbai q=no-prefix", lambda: embed(qtext, MXBAI, "", label="mx.raw")),
    ]
    for name, fn in mxbai_variants:
        r = recall(fn(), mcorpus, gold)
        rows.append((name, r[5], r[2])); print(f"  {name:38s} R@5={r[5]:.1f} R@2={r[2]:.1f}")

    rows.sort(key=lambda x: -x[1])
    print("\n=== DIAGNOSTIC SUMMARY (sorted by R@5) ===")
    print(f"{'variant':40s} {'R@5':>6} {'R@2':>6}")
    for name, r5, r2 in rows:
        print(f"{name:40s} {r5:6.1f} {r2:6.1f}")
    print(f"{'-'*54}")
    print(f"{'NV-Embed-v2 (embedder bar)':40s} {69.7:6.1f}")
    print(f"{'HippoRAG-2 (winner)':40s} {74.7:6.1f}")
    out = {"dataset": "musique", "n_queries": len(queries),
           "bars": {"nv_embed_v2": 69.7, "hipporag2_winner": 74.7},
           "results": [{"variant": n, "recall_at_5": r5, "recall_at_2": r2} for n, r5, r2 in rows]}
    json.dump(out, open("data/results/diag_embedder_recall_musique.json", "w"), indent=2)
    print("\nsaved -> data/results/diag_embedder_recall_musique.json")


if __name__ == "__main__":
    main()
