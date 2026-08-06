#!/usr/bin/env python3
"""commercial_embed_retrieve.py — embed query texts with a commercially-licensed embedder
(mxbai-embed-large-v1 or Qwen3-VL-Embedding-8B), return top-k corpus hits.

Same stateless I/O contract as nvembed_encode_retrieve.py (issue #118, Task B): read
{"id":..., "text":...} lines, embed against a prebuilt corpus .npy, write
{id: [[corpus_idx, score], ...]} top-k per query. Used by selfask_probe.py's
--retriever commercial path so the self-ask decomposition loop can retrieve against a
commercially-clean embedder instead of NV-Embed-v2.

Encoding recipe matches the diagnostic that produced the cited 59.9 (qwen3-VL) / 55.7
(mxbai) raw MuSiQue R@5 floors (docs/research/task_b_infra_findings.md,
scripts/diag_embedder_recall.py): NO instruction/prefix on either side (best-scoring
variant for both embedders), corpus text = "{title}\n{text}".

  --embedder qwen3vl : HTTP call to the KMS embed server ($QWEN_EMBED_URL/embed)
  --embedder mxbai    : local sentence-transformers load (mixedbread-ai/mxbai-embed-large-v1);
                        the KMS mxbai HTTP server (:9999) was down at the time this was written
                        (docs/research/task_b_infra_findings.md) — this script does not depend on it.

Usage:
  python scripts/commercial_embed_retrieve.py --embedder mxbai \
    --in /tmp/subqs.jsonl --corpus-emb data/results/embeddings/kms_mxbai/corpus_emb.npy \
    --topk 10 --out /tmp/hits.json
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request

import numpy as np

QWEN_URL_DEFAULT = "http://localhost:9500"
MXBAI_MODEL = "mixedbread-ai/mxbai-embed-large-v1"

_mxbai_model = None


def embed_http(texts, url, instruction="", batch_size=32):
    """Batch_size is IGNORED here and forced to 1 (see note below) — the KMS qwen3vl embed
    server (production container kms-qwen3vl-embed on Spark, port 9500) throws an
    intermittent CUBLAS_STATUS_INTERNAL_ERROR on some batch>1 request shapes (confirmed via
    `docker logs kms-qwen3vl-embed`: a cublasGemmEx failure inside the Qwen3-VL attention
    forward pass — a GB10/Blackwell driver-maturity issue, not our request). batch_size=1
    verified stable over 20/20 sequential calls; this is shared production infra so we do
    not poke at the batch>1 path further. Slower (~1 req per sub-question) but reliable.

    Returns (vectors, ok_mask): ok_mask[i] is False for any text that never succeeded after
    the full retry budget. During Task B's RUN-1 pass (issue #118) we observed the CUBLAS
    error recur on the SAME text deterministically across a 40-attempt/~9min-span retry
    window for a small number of sub-questions — evidence this isn't purely transient for
    every input, it may be shape/content-triggered for a few specific texts. Previously a
    single such text raised and discarded the ENTIRE batch (up to 500 queries); now we skip
    just that text (logged) so one poisoned input can't block hundreds of good ones. The
    caller drops skipped ids from the output; selfask_probe.py already handles a hop's
    `hits` dict missing some qids gracefully (that query just doesn't advance this hop)."""
    out = []
    ok_mask = []
    for i in range(0, len(texts), 1):
        body = json.dumps({"texts": texts[i:i + 1], "instruction": instruction}).encode()
        req = urllib.request.Request(url.rstrip("/") + "/embed", data=body,
                                     headers={"Content-Type": "application/json"})
        # Retry budget widened after observing the documented CUBLAS blip recur during Task
        # B's multi-hour RUN-1 pass (issue #118) in bursty, multi-minute-long bad windows
        # (not brief blips) — the per-query span needs to comfortably outlast the longest
        # observed bad window (~6-7 min) rather than just retry a few times. Still not
        # touching the production container, just riding out transient windows client-side.
        max_attempts = 40
        vec = None
        for attempt in range(max_attempts):
            try:
                vec = json.load(urllib.request.urlopen(req, timeout=60))["vectors"][0]
                break
            except Exception as e:
                if attempt == max_attempts - 1:
                    print(f"[embed_http] giving up on item {i} after {max_attempts} attempts: {e}",
                          flush=True)
                else:
                    time.sleep(min(2.0 * (attempt + 1), 15.0))
        if vec is None:
            ok_mask.append(False)
        else:
            out.append(vec)
            ok_mask.append(True)
    return np.asarray(out, dtype=np.float32), ok_mask


def embed_mxbai_local(texts, batch_size=32):
    global _mxbai_model
    if _mxbai_model is None:
        from sentence_transformers import SentenceTransformer
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _mxbai_model = SentenceTransformer(MXBAI_MODEL, device=device)
    emb = _mxbai_model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                              show_progress_bar=False)
    return np.asarray(emb, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--embedder", required=True, choices=["mxbai", "qwen3vl"])
    ap.add_argument("--corpus-emb", required=True, help="path to the prebuilt corpus embedding .npy")
    ap.add_argument("--qwen-url", default=QWEN_URL_DEFAULT)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.inp) if l.strip()]
    ids = [r["id"] for r in rows]
    texts = [r["text"] for r in rows]
    C = np.load(a.corpus_emb)
    print(f"[{a.embedder}] queries={len(texts)} corpus={C.shape[0]}", flush=True)

    if a.embedder == "qwen3vl":
        arr, ok_mask = embed_http(texts, a.qwen_url, instruction="", batch_size=a.batch_size)
    else:
        arr = embed_mxbai_local(texts, batch_size=a.batch_size)
        ok_mask = [True] * len(texts)

    n_skipped = ok_mask.count(False)
    if n_skipped:
        print(f"[{a.embedder}] WARNING: {n_skipped}/{len(texts)} queries permanently failed "
              f"embedding (production embed-server error persisted past the full retry "
              f"budget) — skipped, not retried further this call.", flush=True)
    ok_ids = [qid for qid, ok in zip(ids, ok_mask) if ok]

    n = np.linalg.norm(arr, axis=1, keepdims=True)
    n[n == 0] = 1e-10
    arr = (arr / n).astype(np.float32)

    sims = arr @ C.T
    out = {}
    for j, qid in enumerate(ok_ids):
        top = np.argsort(-sims[j])[: a.topk]
        out[qid] = [[int(t), float(sims[j, t])] for t in top]
    json.dump(out, open(a.out, "w"))
    print(f"saved {len(out)} -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
