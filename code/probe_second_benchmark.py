#!/usr/bin/env python3
"""Does the paper's headline survive a second benchmark? A cheap probe before the full panel.

Paper 1's central claim -- NVIDIA's commercial Nemotron-3-Embed-8B *matches* the
non-commercial research anchor NV-Embed-v2 -- rests on one benchmark, MuSiQue under the
HippoRAG-2 protocol. That is the paper's largest structural weakness, and the fix (run the
whole 13-embedder panel on 2Wiki and HotpotQA) costs GPU hours across a fleet.

This runs only the two models the claim is about, on one extra benchmark, in the reported
configuration. If the match holds, the full panel is worth provisioning. If the anchor pulls
away, we have learned that for a couple of GPU-hours instead of finding out from a referee.

Deliberately faithful to the measured configuration rather than convenient:
  * corpus text is "{title}\\n{text}" -- the matched format the paper reports; the text-only
    variant is what caused the original v5 anchor bug, so it is not the default here.
  * NV-Embed-v2 uses the `web-search` query instruction, the variant the paper reports, with
    an empty passage instruction.
  * Nemotron-3-Embed-8B uses its asymmetric encode_query/encode_document pair ("query: " /
    "passage: "). Mismatching those sides degrades retrieval silently.

Gold sets differ by dataset and this is where a benchmark migration usually goes wrong:
MuSiQue marks supporting paragraphs inline (`paragraphs[].is_supporting`) and is matched on
paragraph text; 2Wiki and HotpotQA instead carry `supporting_facts` and are matched on title.
Title matching is only safe where titles are unique, so that is asserted, not assumed --
2Wiki has 6,119 passages under 6,119 distinct titles.

Run (Spark):
  # Nemotron needs transformers >= 5.2
  ~/ir-training-venv/bin/python scripts/probe_second_benchmark.py \\
      --model nemotron --dataset 2wikimultihopqa
  # NV-Embed needs the pinned 4.42.4 shadow
  PYTHONPATH=~/afwerk/.tf442 ~/faster-qwen3-tts/.venv/bin/python \\
      scripts/probe_second_benchmark.py --model nvembed --dataset 2wikimultihopqa
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

DATA = "data/hipporag2"

MODELS = {
    "nemotron": "nvidia/Nemotron-3-Embed-8B-BF16",
    "nvembed": "nvidia/NV-Embed-v2",
}

# The variant the paper reports for the anchor.
NVEMBED_QUERY_INSTRUCTION = (
    "Instruct: Given a web search query, retrieve relevant passages that answer the query\n"
    "Query: ")

# MuSiQue numbers the paper reports, for the sanity check that the harness still agrees
# with the manuscript before any second-benchmark number is believed.
MUSIQUE_REPORTED = {"nemotron": (69.79, 77.54), "nvembed": (69.55, 78.12)}


def l2(a) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    n = np.linalg.norm(a, axis=1, keepdims=True)
    n[n == 0] = 1e-10
    return a / n


def build_gold(corpus, queries, dataset: str) -> list[list[int]]:
    """Supporting-passage indices per question. Two schemas, and they are not interchangeable."""
    if dataset == "musique":
        tmap: dict = {}
        for idx, p in enumerate(corpus):
            tmap.setdefault(p["text"], []).append(idx)
        gold = []
        for q in queries:
            gi = set()
            for p in q["paragraphs"]:
                if p.get("is_supporting") and p["paragraph_text"] in tmap:
                    gi.update(tmap[p["paragraph_text"]])
            gold.append(sorted(gi))
        return gold

    # 2Wiki / HotpotQA: supporting_facts is [[title, sentence_idx], ...] and the corpus is
    # one passage per title, so gold is by title -- valid only under unique titles.
    tmap = {}
    for idx, p in enumerate(corpus):
        tmap.setdefault(p["title"], []).append(idx)
    dupes = {t: v for t, v in tmap.items() if len(v) > 1}
    if dupes:
        raise SystemExit(
            f"{len(dupes)} duplicate titles in {dataset}; title matching would silently "
            "conflate passages. Fix the gold builder before trusting any recall number.")
    gold, missing = [], 0
    for q in queries:
        gi = set()
        for sf in q["supporting_facts"]:
            t = sf[0]
            if t in tmap:
                gi.update(tmap[t])
            else:
                missing += 1
        gold.append(sorted(gi))
    if missing:
        raise SystemExit(f"{missing} supporting titles absent from the {dataset} corpus; "
                         "recall would be understated against an unreachable ceiling.")
    return gold


def per_question_recall(qe, ce, gold, ks=(5, 10)) -> dict[int, list[float]]:
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(MODELS), required=True)
    ap.add_argument("--dataset", default="2wikimultihopqa",
                    choices=["2wikimultihopqa", "hotpotqa", "musique"])
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--query-max-length", type=int, default=512)
    ap.add_argument("--passage-max-length", type=int, default=512)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    corpus = json.load(open(f"{DATA}/{a.dataset}_corpus.json"))
    queries = json.load(open(f"{DATA}/{a.dataset}.json"))
    gold = build_gold(corpus, queries, a.dataset)
    qtext = [q["question"] for q in queries]
    ptexts = [f'{p["title"]}\n{p["text"]}' for p in corpus]
    scored = sum(1 for g in gold if g)
    print(f"dataset={a.dataset} corpus={len(corpus)} queries={len(qtext)} scored={scored}",
          flush=True)

    name = MODELS[a.model]
    t0 = time.time()
    import torch
    from transformers import AutoModel

    if a.model == "nvembed":
        model = AutoModel.from_pretrained(name, trust_remote_code=True,
                                          torch_dtype=torch.bfloat16).cuda().eval()

        def enc(texts, instruction, maxlen, label):
            out = []
            for i in range(0, len(texts), a.batch_size):
                with torch.no_grad():
                    e = model.encode(texts[i:i + a.batch_size], instruction=instruction,
                                     max_length=maxlen)
                out.append(e.detach().float().cpu().numpy())
                if i % (a.batch_size * 50) == 0:
                    print(f"  [{label}] {i}/{len(texts)}", flush=True)
            return l2(np.vstack(out))

        ce = enc(ptexts, "", a.passage_max_length, "corpus")
        qe = enc(qtext, NVEMBED_QUERY_INSTRUCTION, a.query_max_length, "query")
    else:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(name, trust_remote_code=True,
                                    model_kwargs={"torch_dtype": torch.bfloat16})
        ce = l2(model.encode_document(ptexts, batch_size=a.batch_size,
                                      show_progress_bar=True))
        qe = l2(model.encode_query(qtext, batch_size=a.batch_size, show_progress_bar=True))

    perq = per_question_recall(qe, ce, gold)
    r5, r10 = float(np.mean(perq[5]) * 100), float(np.mean(perq[10]) * 100)
    print(f"\n{a.model} on {a.dataset}:  Recall@5 {r5:.2f}   Recall@10 {r10:.2f}   "
          f"({time.time() - t0:.0f}s)")
    if a.dataset == "musique":
        pr5, pr10 = MUSIQUE_REPORTED[a.model]
        print(f"  paper reports {pr5}/{pr10} -> delta {r5 - pr5:+.2f}/{r10 - pr10:+.2f}")

    out = a.out or f"data/results/probe_{a.model}_{a.dataset}.json"
    json.dump({"model": name, "dataset": a.dataset, "format": "title+text",
               "instruction": NVEMBED_QUERY_INSTRUCTION if a.model == "nvembed" else
                              "encode_query/encode_document",
               "corpus_size": len(corpus), "n_questions": len(qtext), "scored": scored,
               "recall_at_5": r5, "recall_at_10": r10,
               "perq": {"r5": perq[5], "r10": perq[10]}}, open(out, "w"))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
