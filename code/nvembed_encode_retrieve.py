#!/usr/bin/env python3
"""nvembed_encode_retrieve.py — embed query texts with NV-Embed-v2, return top-k corpus hits.

Stateless helper for the Self-Ask probe (#103, Track C): the dev orchestrator generates
sub-questions (OpenRouter LLM, key stays on dev), ships them here per hop, and we do the
NV-Embed encoding + corpus retrieval on Spark (where the model + vectors live). Internal
OOM-retry handles transient GPU contention with the resident production process.

Run on Spark:
  HF_HOME=/home/luis/afwerk/.hfcache PYTHONPATH=/home/luis/afwerk/.tf442 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    /home/luis/faster-qwen3-tts/.venv/bin/python scripts/nvembed_encode_retrieve.py \
    --in /tmp/subqs.jsonl --topk 10 --out /tmp/hits.json

  --in   : jsonl, one {"id": <str>, "text": <query>} per line
  --out  : json, {id: [[corpus_idx, score], ...]} top-k per query
"""
from __future__ import annotations

import argparse, json, time
import numpy as np
import torch

from transformers import DynamicCache
if not hasattr(DynamicCache, "get_usable_length"):
    DynamicCache.get_usable_length = lambda self, *a, **kw: self.get_seq_length()

MODEL_NAME = "nvidia/NV-Embed-v2"
QUERY_INSTRUCTION = "Instruct: Given a question, retrieve passages that answer the question\nQuery: "


def load_model(retries=8, wait=60, load_4bit=False):
    from transformers import AutoModel
    # transformers>=4.56 honors `dtype`; <=4.42 (the NV-Embed pin) honors `torch_dtype`.
    import transformers as _tf
    _v = tuple(int(x) for x in _tf.__version__.split(".")[:2])
    _dk = "dtype" if _v >= (4, 56) else "torch_dtype"
    kw = {"trust_remote_code": True, _dk: torch.bfloat16, "low_cpu_mem_usage": True}
    if load_4bit:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        kw["device_map"] = {"": 0}
    for i in range(retries):
        try:
            m = AutoModel.from_pretrained(MODEL_NAME, **kw)
            if not load_4bit:
                # custom NV-Embed loader ignores dtype -> lands fp32; cast on CPU, THEN move
                # (only ~15.7GB bf16 crosses to GPU instead of 31GB fp32 -> no OOM on 24GB).
                m = m.to(torch.bfloat16).cuda()
            m.eval()
            return m
        except (getattr(torch.cuda, "OutOfMemoryError", RuntimeError), RuntimeError) as e:
            if "out of memory" not in str(e).lower() or i == retries - 1:
                raise
            torch.cuda.empty_cache()
            print(f"[oom] model load attempt {i+1} failed; wait {wait}s", flush=True)
            time.sleep(wait)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--emb-dir", default="data/results/embeddings/nvembed")
    ap.add_argument("--corpus-emb", default="",
                    help="explicit path to the corpus embedding .npy; overrides --emb-dir/musique default")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--out", required=True)
    ap.add_argument("--load-4bit", action="store_true", help="4-bit NF4 load (~5GB) to share a busy GPU")
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.inp) if l.strip()]
    ids = [r["id"] for r in rows]
    texts = [r["text"] for r in rows]
    C = np.load(a.corpus_emb or f"{a.emb_dir}/musique_corpus_emb.npy")
    print(f"queries={len(texts)} corpus={C.shape[0]}", flush=True)

    model = load_model(load_4bit=a.load_4bit)
    embs = []
    for i in range(0, len(texts), a.batch_size):
        with torch.no_grad():
            e = model.encode(texts[i:i+a.batch_size], instruction=QUERY_INSTRUCTION, max_length=a.max_length)
        if isinstance(e, torch.Tensor):
            e = e.float().cpu().numpy()
        embs.append(e)
    arr = np.vstack(embs)
    n = np.linalg.norm(arr, axis=1, keepdims=True); n[n == 0] = 1e-10
    arr = (arr / n).astype(np.float32)

    sims = arr @ C.T
    out = {}
    for j, qid in enumerate(ids):
        top = np.argsort(-sims[j])[: a.topk]
        out[qid] = [[int(t), float(sims[j, t])] for t in top]
    json.dump(out, open(a.out, "w"))
    print(f"saved {len(out)} -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
