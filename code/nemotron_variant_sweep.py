#!/usr/bin/env python3
"""nemotron_variant_sweep.py — measure Nemotron-3-Embed-8B on MuSiQue under both corpus
formats, so its Paper 1 number can be compared like-for-like with the corrected
NV-Embed-v2 figure from nvembed_variant_sweep.py.

Why this exists
---------------
Paper 1 reports Nemotron-3-Embed-8B at 69.69 / 77.45 (Recall@5/@10) and NV-Embed-v2 at
67.1 / 76.2. The NV-Embed number turned out to be a harness artifact: its passages were
embedded text-only (build_nvembed_index.py:54) while the rest of the panel got
"{title}\\n{text}". Corrected, NV-Embed-v2 is 69.55 / 78.12.

No script or per-question artifact survives for the Nemotron run, so we cannot tell from
source which corpus format produced 69.69. This re-measures both. Whichever format
reproduces 69.69 identifies what the paper actually did; the title+text row is the one
comparable to the corrected NV-Embed-v2 figure.

Model convention (model card + scripts/serve_lab_retrieval.py:90-97): asymmetric prefixes,
`encode_query` -> "query: ", `encode_document` -> "passage: ". Mismatching the sides
degrades retrieval silently. Requires transformers >= 5.2.0 (ministral3 architecture).

Run:
  PYTHONPATH=/research/afwerk/.tf520 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
    python3 scripts/nemotron_variant_sweep.py --out data/results/nemotron_variant_sweep.json
"""
from __future__ import annotations

import argparse, json, os, time
import numpy as np
import torch

MODEL_NAME = "nvidia/Nemotron-3-Embed-8B-BF16"
DATA = "data/hipporag2"
PAPER_REPORTED = {"recall_at_5": 69.69, "recall_at_10": 77.45}


def per_question_recall(query_embs, corpus_embs, gold, ks=(5, 10)):
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


def l2(a):
    a = np.asarray(a, dtype=np.float32)
    n = np.linalg.norm(a, axis=1, keepdims=True); n[n == 0] = 1e-10
    return a / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/results/nemotron_variant_sweep.json")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--cache-dir", default="data/results/embeddings/nemotron3_8b")
    a = ap.parse_args()

    from sentence_transformers import SentenceTransformer

    corpus = json.load(open(f"{DATA}/musique_corpus.json"))
    queries = json.load(open(f"{DATA}/musique.json"))
    qtext = [q["question"] for q in queries]
    gold = build_gold(corpus, queries)
    print(f"corpus={len(corpus)} queries={len(qtext)} scored={sum(1 for g in gold if g)}",
          flush=True)

    model = SentenceTransformer(MODEL_NAME, model_kwargs={"dtype": torch.bfloat16},
                                device="cuda")
    print("model loaded", flush=True)

    os.makedirs(a.cache_dir, exist_ok=True)
    formats = {
        "title+text": [f'{p["title"]}\n{p["text"]}' for p in corpus],
        "text-only": [p["text"] for p in corpus],
    }

    corpora = {}
    for cname, texts in formats.items():
        path = os.path.join(a.cache_dir, f"corpus_{cname.replace('+','_')}.npy")
        if os.path.exists(path):
            corpora[cname] = np.load(path)
            print(f"[corpus] {cname} cached {corpora[cname].shape}", flush=True)
            continue
        print(f"[corpus] embedding {cname} ...", flush=True)
        t0 = time.time()
        emb = l2(model.encode_document(texts, batch_size=a.batch_size,
                                       show_progress_bar=True))
        np.save(path, emb)
        corpora[cname] = emb
        print(f"[corpus] {cname} done {time.time()-t0:.0f}s {emb.shape}", flush=True)

    print("\n[query] encode_query ('query: ' prefix)", flush=True)
    qe = l2(model.encode_query(qtext, batch_size=a.batch_size, show_progress_bar=True))

    results, perq = [], {}
    for cname, C in corpora.items():
        r = per_question_recall(qe, C, gold)
        row = {"corpus_format": cname,
               "recall_at_5": float(np.mean(r[5]) * 100),
               "recall_at_10": float(np.mean(r[10]) * 100),
               "n_scored": len(r[5])}
        results.append(row)
        perq[cname] = {"r5": r[5], "r10": r[10]}
        print(f"  {cname:12s} R@5={row['recall_at_5']:.2f}  R@10={row['recall_at_10']:.2f}",
              flush=True)

    results.sort(key=lambda x: -x["recall_at_5"])
    print("\n=== NEMOTRON SUMMARY ===")
    print(f"paper reported: R@5={PAPER_REPORTED['recall_at_5']}  "
          f"R@10={PAPER_REPORTED['recall_at_10']}")
    for r in results:
        d5 = r["recall_at_5"] - PAPER_REPORTED["recall_at_5"]
        print(f"{r['corpus_format']:12s} R@5={r['recall_at_5']:6.2f} ({d5:+.2f})  "
              f"R@10={r['recall_at_10']:6.2f}")

    json.dump({"model": MODEL_NAME, "dataset": "musique",
               "corpus_size": len(corpus), "n_queries": len(qtext),
               "paper_reported": PAPER_REPORTED, "results": results,
               "best": results[0]}, open(a.out, "w"), indent=1)
    json.dump(perq, open(a.out.replace(".json", "_perq.json"), "w"))
    print(f"\nsaved {a.out}", flush=True)


if __name__ == "__main__":
    main()
