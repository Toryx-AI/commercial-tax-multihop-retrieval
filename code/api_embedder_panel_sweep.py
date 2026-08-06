#!/usr/bin/env python3
"""api_embedder_panel_sweep.py — re-measure the API-provider embedders of the Paper 1 panel
on BOTH corpus formats, saving per-question recall vectors.

Why this exists
---------------
The panel's NV-Embed-v2 row turned out to be measured on a text-only corpus while others got
"{title}\\n{text}" (see docs/paper/FINDING_2026-08-05_nvembed_title_artifact.md). No script or
per-question artifact survives for the API-provider rows, so we cannot tell from source which
format they used. Measuring both formats reproduces whichever the original run used - the same
forensic check that confirmed Nemotron-3-Embed-8B had been embedded title+text.

Each provider's documented asymmetric query/document mode is used, because ignoring it embeds
a question as though it were a document and silently costs recall.

Resumable: corpus and query embeddings are cached per (provider, format) under --cache-dir,
so an interrupted run re-uses what it already paid for.

  python3 scripts/api_embedder_panel_sweep.py --providers openai-small,openai-large,gemini,voyage,cohere
"""
from __future__ import annotations

import argparse, json, os, pathlib, time
import numpy as np

DATA = "data/hipporag2"
DEFAULT_CACHE = "data/results/embeddings/api_panel"
PAPER_REPORTED = {                       # Table 1 of the current draft, (R@5, R@10)
    "openai-small": (55.19, 64.18),
    "openai-large": (60.02, 70.28),
    "gemini":       (67.24, 76.35),
    "voyage":       (53.97, 63.65),
    "cohere":       (60.13, 69.08),
    "nim-e5-v5":               (57.6, 66.2),
    "nim-nemotron-1b":         (64.32, 72.79),
    "nim-llama-nemotron-1b-v2": (63.73, 71.35),
}


def load_env(path=".env"):
    for line in pathlib.Path(path).read_text().splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def l2(a):
    a = np.asarray(a, dtype=np.float32)
    n = np.linalg.norm(a, axis=1, keepdims=True); n[n == 0] = 1e-10
    return a / n


def retry(fn, tries=5, base=2.0, label=""):
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            if i == tries - 1:
                raise
            wait = base * (2 ** i)
            print(f"    [{label}] retry {i+1}/{tries-1} after {type(e).__name__}: "
                  f"{str(e)[:110]} (sleep {wait:.0f}s)", flush=True)
            time.sleep(wait)


# --------------------------------------------------------------------------- providers

def embed_openai(texts, is_query, model, batch=256, label=""):
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    out = []
    for i in range(0, len(texts), batch):
        chunk = [t.replace("\n", " ")[:30000] for t in texts[i:i + batch]]
        r = retry(lambda: client.embeddings.create(model=model, input=chunk), label=label)
        out.extend(d.embedding for d in r.data)
        if (i // batch) % 8 == 0:
            print(f"    [{label}] {min(i+batch,len(texts))}/{len(texts)}", flush=True)
    return out


def embed_gemini(texts, is_query, model="gemini-embedding-001", batch=64, label=""):
    import urllib.request
    key = os.environ["GOOGLE_API_KEY"]
    ttype = "RETRIEVAL_QUERY" if is_query else "RETRIEVAL_DOCUMENT"
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{model}:batchEmbedContents?key={key}")
    out = []
    for i in range(0, len(texts), batch):
        chunk = texts[i:i + batch]
        body = {"requests": [{"model": f"models/{model}",
                              "content": {"parts": [{"text": t[:20000]}]},
                              "taskType": ttype} for t in chunk]}

        def call():
            req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            return json.load(urllib.request.urlopen(req, timeout=300))
        r = retry(call, label=label)
        out.extend(e["values"] for e in r["embeddings"])
        if (i // batch) % 20 == 0:
            print(f"    [{label}] {min(i+batch,len(texts))}/{len(texts)}", flush=True)
    return out


def embed_voyage(texts, is_query, model="voyage-3.5", batch=96, label=""):
    import voyageai
    client = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])
    itype = "query" if is_query else "document"
    out = []
    for i in range(0, len(texts), batch):
        chunk = texts[i:i + batch]
        r = retry(lambda: client.embed(chunk, model=model, input_type=itype), label=label)
        out.extend(r.embeddings)
        if (i // batch) % 15 == 0:
            print(f"    [{label}] {min(i+batch,len(texts))}/{len(texts)}", flush=True)
    return out


def embed_cohere_bedrock(texts, is_query, model="cohere.embed-v4:0", batch=96, label=""):
    import boto3
    sess = boto3.Session(profile_name=os.environ.get("AWS_PROFILE", "credit"))
    rt = sess.client("bedrock-runtime", region_name=os.environ.get("AWS_REGION", "us-east-1"))
    itype = "search_query" if is_query else "search_document"
    out = []
    for i in range(0, len(texts), batch):
        chunk = [t[:8000] for t in texts[i:i + batch]]
        body = json.dumps({"texts": chunk, "input_type": itype,
                           "embedding_types": ["float"]})

        def call():
            r = rt.invoke_model(modelId=model, body=body,
                                accept="application/json", contentType="application/json")
            return json.loads(r["body"].read())
        r = retry(call, label=label)
        emb = r.get("embeddings")
        if isinstance(emb, dict):
            emb = emb.get("float")
        out.extend(emb)
        if (i // batch) % 15 == 0:
            print(f"    [{label}] {min(i+batch,len(texts))}/{len(texts)}", flush=True)
    return out


def embed_nim(texts, is_query, model, batch=32, label=""):
    """NVIDIA NIM hosted embeddings. The key lives in ~/.hermes/.env, not the repo .env.

    Model ids are not stable in NVIDIA's catalog: `nemotron-3-embed-1b-bf16` (the HuggingFace
    name) 404s here and the working id is `nemotron-3-embed-1b`; `nemotron-3-embed-8b` 404s
    entirely, consistent with the 8B being self-host-only.
    """
    import urllib.request
    key = os.environ.get("NVIDIA_API_KEY") or ""
    if not key:
        load_env("/home/luis/.hermes/.env")
        key = os.environ["NVIDIA_API_KEY"]
    itype = "query" if is_query else "passage"
    out = []
    for i in range(0, len(texts), batch):
        chunk = [t[:8000] for t in texts[i:i + batch]]
        body = {"input": chunk, "model": model, "input_type": itype,
                "encoding_format": "float", "truncate": "END"}

        def call():
            req = urllib.request.Request(
                "https://integrate.api.nvidia.com/v1/embeddings",
                data=json.dumps(body).encode(),
                headers={"Authorization": "Bearer " + key,
                         "Content-Type": "application/json",
                         "Accept": "application/json"})
            return json.load(urllib.request.urlopen(req, timeout=300))
        r = retry(call, tries=6, label=label)
        out.extend(d["embedding"] for d in r["data"])
        if (i // batch) % 40 == 0:
            print(f"    [{label}] {min(i+batch,len(texts))}/{len(texts)}", flush=True)
    return out


PROVIDERS = {
    "openai-small": lambda t, q, lb: embed_openai(t, q, "text-embedding-3-small", label=lb),
    "openai-large": lambda t, q, lb: embed_openai(t, q, "text-embedding-3-large", label=lb),
    "gemini":       lambda t, q, lb: embed_gemini(t, q, label=lb),
    "voyage":       lambda t, q, lb: embed_voyage(t, q, label=lb),
    "cohere":       lambda t, q, lb: embed_cohere_bedrock(t, q, label=lb),
    "nim-e5-v5":    lambda t, q, lb: embed_nim(t, q, "nvidia/nv-embedqa-e5-v5", label=lb),
    "nim-nemotron-1b": lambda t, q, lb: embed_nim(t, q, "nvidia/nemotron-3-embed-1b", label=lb),
    "nim-llama-nemotron-1b-v2": lambda t, q, lb: embed_nim(
        t, q, "nvidia/llama-nemotron-embed-1b-v2", label=lb),
}


# --------------------------------------------------------------------------- eval

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


def cached(path, fn):
    if os.path.exists(path):
        return np.load(path)
    a = l2(fn())
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, a)
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--providers", default=",".join(PROVIDERS))
    ap.add_argument("--cache-dir", default=DEFAULT_CACHE)
    ap.add_argument("--out", default="data/results/api_panel_sweep.json")
    a = ap.parse_args()
    load_env()

    corpus = json.load(open(f"{DATA}/musique_corpus.json"))
    queries = json.load(open(f"{DATA}/musique.json"))
    qtext = [q["question"] for q in queries]
    gold = build_gold(corpus, queries)
    formats = {
        "title+text": [f'{p["title"]}\n{p["text"]}' for p in corpus],
        "text-only": [p["text"] for p in corpus],
    }
    print(f"corpus={len(corpus)} queries={len(qtext)} scored={sum(1 for g in gold if g)}",
          flush=True)

    results, perq = [], {}
    outp = pathlib.Path(a.out)
    if outp.exists():                      # merge with any earlier partial run
        prev = json.loads(outp.read_text())
        results = prev.get("results", [])
    done = {(r["provider"], r["corpus_format"]) for r in results}

    for name in [p.strip() for p in a.providers.split(",") if p.strip()]:
        if name not in PROVIDERS:
            print(f"!! unknown provider {name}, skipping", flush=True)
            continue
        fn = PROVIDERS[name]
        cdir = os.path.join(a.cache_dir, name)
        try:
            qe = cached(os.path.join(cdir, "queries.npy"),
                        lambda: fn(qtext, True, f"{name}.q"))
        except Exception as e:
            print(f"!! {name} query embedding failed: {type(e).__name__}: {str(e)[:200]}",
                  flush=True)
            continue

        for cname, texts in formats.items():
            if (name, cname) in done:
                print(f"[{name}/{cname}] already done, skipping", flush=True)
                continue
            print(f"\n[{name}] corpus {cname}", flush=True)
            try:
                ce = cached(os.path.join(cdir, f"corpus_{cname.replace('+','_')}.npy"),
                            lambda: fn(texts, False, f"{name}.{cname}"))
            except Exception as e:
                print(f"!! {name}/{cname} failed: {type(e).__name__}: {str(e)[:200]}",
                      flush=True)
                continue
            r = per_question_recall(qe, ce, gold)
            row = {"provider": name, "corpus_format": cname,
                   "recall_at_5": float(np.mean(r[5]) * 100),
                   "recall_at_10": float(np.mean(r[10]) * 100),
                   "n_scored": len(r[5])}
            if name in PAPER_REPORTED:
                row["paper_reported_r5"], row["paper_reported_r10"] = PAPER_REPORTED[name]
                row["delta_r5_vs_paper"] = row["recall_at_5"] - row["paper_reported_r5"]
            results.append(row)
            perq[f"{name}|{cname}"] = {"r5": r[5], "r10": r[10]}
            d = f" (paper {row['paper_reported_r5']}, {row['delta_r5_vs_paper']:+.2f})" \
                if name in PAPER_REPORTED else ""
            print(f"  {cname:12s} R@5={row['recall_at_5']:.2f}  "
                  f"R@10={row['recall_at_10']:.2f}{d}", flush=True)

            outp.parent.mkdir(parents=True, exist_ok=True)
            json.dump({"dataset": "musique", "corpus_size": len(corpus),
                       "n_queries": len(qtext), "results": results},
                      open(a.out, "w"), indent=1)
            pq_path = a.out.replace(".json", "_perq.json")
            existing = json.loads(pathlib.Path(pq_path).read_text()) \
                if os.path.exists(pq_path) else {}
            existing.update(perq)
            json.dump(existing, open(pq_path, "w"))

    print("\n=== API PANEL SUMMARY ===")
    print(f"{'provider':14s} {'corpus':12s} {'R@5':>7} {'R@10':>7} {'paper R@5':>10} {'delta':>7}")
    for r in sorted(results, key=lambda x: (x["provider"], x["corpus_format"])):
        pr = r.get("paper_reported_r5", float("nan"))
        dl = r.get("delta_r5_vs_paper", float("nan"))
        print(f"{r['provider']:14s} {r['corpus_format']:12s} {r['recall_at_5']:7.2f} "
              f"{r['recall_at_10']:7.2f} {pr:10.2f} {dl:+7.2f}")
    print(f"\nsaved {a.out}")


if __name__ == "__main__":
    main()
