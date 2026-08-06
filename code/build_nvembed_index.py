#!/usr/bin/env python3
"""build_nvembed_index.py — embed a hipporag2 corpus with NV-Embed-v2 -> L2-normalized .npy.

Passages are embedded with NO instruction (NV-Embed-v2 convention) and text-only (no title),
matching the MuSiQue index the retrieval baseline used — so the retrieval *method* is identical
across MuSiQue / HotpotQA / 2Wiki (the point of the cross-dataset transfer experiment).

Run with the tf442 shadow (same recipe as nvembed_encode_retrieve.py):
  HF_HOME=/root/.hfcache PYTHONPATH=/root/.tf442 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python scripts/build_nvembed_index.py \
    --corpus data/hipporag2/hotpotqa_corpus.json \
    --out data/results/embeddings/nvembed/hotpotqa_corpus_emb.npy
"""
from __future__ import annotations

import argparse, json, os, time
import numpy as np
import torch

from transformers import DynamicCache
if not hasattr(DynamicCache, "get_usable_length"):
    DynamicCache.get_usable_length = lambda self, *a, **kw: self.get_seq_length()

MODEL_NAME = "nvidia/NV-Embed-v2"
PASSAGE_INSTRUCTION = ""  # passages carry no instruction


def load_model(retries=8, wait=60):
    from transformers import AutoModel
    for i in range(retries):
        try:
            m = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True,
                                          torch_dtype=torch.bfloat16).cuda()
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
    ap.add_argument("--corpus", required=True, help="hipporag2 corpus json: list of {idx,title,text}")
    ap.add_argument("--out", required=True, help="output .npy of L2-normalized passage embeddings")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=512)
    a = ap.parse_args()

    corpus = json.load(open(a.corpus))
    texts = [p["text"] for p in corpus]
    print(f"corpus={len(texts)} passages -> {a.out}", flush=True)

    model = load_model()
    embs = []
    t0 = time.time()
    for i in range(0, len(texts), a.batch_size):
        with torch.no_grad():
            e = model.encode(texts[i:i + a.batch_size], instruction=PASSAGE_INSTRUCTION,
                             max_length=a.max_length)
        if isinstance(e, torch.Tensor):
            e = e.float().cpu().numpy()
        embs.append(e)
        if (i // a.batch_size) % 50 == 0:
            done = min(i + a.batch_size, len(texts))
            rate = done / max(time.time() - t0, 1e-9)
            print(f"  {done}/{len(texts)}  ({rate:.0f}/s)", flush=True)
    arr = np.vstack(embs)
    n = np.linalg.norm(arr, axis=1, keepdims=True); n[n == 0] = 1e-10
    arr = (arr / n).astype(np.float32)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    np.save(a.out, arr)
    print(f"saved {arr.shape} -> {a.out}  in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
