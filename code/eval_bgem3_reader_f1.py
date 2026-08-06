#!/usr/bin/env python3
"""BGE-M3 Answer F1 + answering cost, matching Paper 1's §4.7 protocol exactly:
gpt-4o-mini fixed reader over the real top-k retrieved passages (BGE-M3's best
variant: instruction-prefixed), for all 1,000 MuSiQue questions, at k=5 and k=10.
SQuAD-style normalized token-overlap F1, max over answer aliases. Real API calls,
real measured token counts -> both batch and standard cost computed from those
real counts (not a token-count estimate).
"""
import json, os, re, string, time
from collections import Counter
import numpy as np
from openai import OpenAI
from FlagEmbedding import BGEM3FlagModel

DATA = "data/hipporag2"
CORPUS_CACHE = "data/results/embeddings/bge_m3/corpus_emb.npy"
MODEL = "gpt-4o-mini"

# real published rates, same as used throughout the paper (§3.5)
RATE_BATCH_IN, RATE_BATCH_OUT = 0.075 / 1_000_000, 0.30 / 1_000_000
RATE_STD_IN, RATE_STD_OUT = 0.15 / 1_000_000, 0.60 / 1_000_000

with open("/research/afwerk/.env") as f:
    for line in f:
        if line.startswith("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = line.strip().split("=", 1)[1].strip().strip('"').strip("'")
client = OpenAI()


def normalize_answer(s):
    def remove_articles(text): return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text): return " ".join(text.split())
    def remove_punc(text): return "".join(ch for ch in text if ch not in string.punctuation)
    def lower(text): return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return float(pred_tokens == gt_tokens)
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def max_f1(prediction, gold_answer, aliases):
    golds = [gold_answer] + list(aliases or [])
    return max(f1_score(prediction, g) for g in golds)


READER_SYSTEM = (
    "You answer questions using only the provided passages. Answer as concisely as "
    "possible — a short phrase or entity, not a full sentence. If the passages do "
    "not contain the answer, give your best guess in as few words as possible."
)


def build_prompt(question, passages):
    ctx = "\n\n".join(f"[{i+1}] {p['title']}: {p['text']}" for i, p in enumerate(passages))
    return f"Passages:\n{ctx}\n\nQuestion: {question}\nAnswer (short phrase only):"


def main():
    corpus = json.load(open(f"{DATA}/musique_corpus.json"))
    queries = json.load(open(f"{DATA}/musique.json"))
    print(f"corpus={len(corpus)} queries={len(queries)}")

    print("loading BAAI/bge-m3 + cached corpus embeddings...")
    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
    corpus_emb = np.load(CORPUS_CACHE)

    qtext_prefixed = [f"Represent this sentence for searching relevant passages: {q['question']}" for q in queries]
    q_emb = model.encode(qtext_prefixed, max_length=512, batch_size=64)["dense_vecs"]
    sims = q_emb @ corpus_emb.T
    top10_idx = np.argsort(-sims, axis=1)[:, :10]
    print(f"retrieval done, top10_idx shape={top10_idx.shape}")

    results = {5: [], 10: []}
    tokens = {5: {"in": 0, "out": 0}, 10: {"in": 0, "out": 0}}

    for k in (5, 10):
        print(f"\n=== running gpt-4o-mini reader at k={k} ===")
        t0 = time.time()
        for i, q in enumerate(queries):
            idxs = top10_idx[i][:k]
            passages = [corpus[j] for j in idxs]
            prompt = build_prompt(q["question"], passages)
            for attempt in range(3):
                try:
                    resp = client.chat.completions.create(
                        model=MODEL,
                        messages=[{"role": "system", "content": READER_SYSTEM},
                                  {"role": "user", "content": prompt}],
                        temperature=0.0, max_tokens=32,
                    )
                    break
                except Exception as e:
                    if attempt == 2:
                        raise
                    time.sleep(2 * (attempt + 1))
            pred = resp.choices[0].message.content.strip()
            f1 = max_f1(pred, q["answer"], q.get("answer_aliases"))
            results[k].append(f1)
            tokens[k]["in"] += resp.usage.prompt_tokens
            tokens[k]["out"] += resp.usage.completion_tokens
            if (i + 1) % 100 == 0:
                elapsed = time.time() - t0
                print(f"  {i+1}/{len(queries)}  mean_f1={np.mean(results[k])*100:.2f}  ({elapsed:.0f}s)")
        print(f"  k={k} done in {time.time()-t0:.0f}s")

    out = {"dataset": "musique", "n_queries": len(queries), "model": MODEL,
           "reader_embedder": "BAAI/bge-m3 (instruction-prefixed, best variant)"}
    print("\n=== BGE-M3 READER RESULTS ===")
    for k in (5, 10):
        mean_f1 = float(np.mean(results[k]) * 100)
        tin, tout = tokens[k]["in"], tokens[k]["out"]
        cost_batch = tin * RATE_BATCH_IN + tout * RATE_BATCH_OUT
        cost_std = tin * RATE_STD_IN + tout * RATE_STD_OUT
        print(f"k={k:2d}  Answer F1={mean_f1:.2f}  tokens_in={tin:,} tokens_out={tout:,}  "
              f"cost_batch=${cost_batch:.4f} cost_std=${cost_std:.4f}")
        out[f"k{k}"] = {"answer_f1": mean_f1, "tokens_in": tin, "tokens_out": tout,
                         "cost_batch_usd": cost_batch, "cost_std_usd": cost_std}

    print("\nFor reference, from the paper's panel (k=5 / k=10 Answer F1):")
    print("  Nemotron-3-Embed-8B  45.12 / 46.22")
    print("  OpenAI text-embedding-3-small (lowest in panel)  34.42 / 39.12")

    json.dump(out, open("data/results/bge_m3_reader_f1_musique.json", "w"), indent=2)
    print("\nsaved -> data/results/bge_m3_reader_f1_musique.json")


if __name__ == "__main__":
    main()
