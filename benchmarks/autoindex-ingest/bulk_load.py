#!/usr/bin/env python3
"""Plain ES `_bulk` loader for the like-for-like ingest row (no autoindex client in the loop).

usage: python3 -I bulk_load.py --url http://127.0.0.1:9200 --index bench --corpus corpus.jsonl [--n N] [--threads T]
                               [--bulk-mb 8] [--out result.json] [--create-index 1]
Corpus lines are {"id": ..., "text": ...}; each becomes `{"index":{"_index":I,"_id":id}}` + `{"text": ...}`.
Bulks are cut at --bulk-mb; T client threads post concurrently; HTTP 429/503 (and per-item 429) are retried with
backoff and counted; any other item error is counted and reported. At the end: POST /I/_refresh, GET /I/_count.
Prints and writes JSON: wall_s (first byte sent -> refresh returned), searchable_s, docs, count, retries, item_errors.
"""
import argparse
import json
import queue
import threading
import time
import urllib.error
import urllib.request


def post(url, body, ctype="application/x-ndjson", timeout=600):
    req = urllib.request.Request(url, data=body, method="POST", headers={"Content-Type": ctype})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--index", default="bench")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--bulk-mb", type=float, default=8.0)
    ap.add_argument("--out", default="")
    ap.add_argument("--create-index", type=int, default=1)
    a = ap.parse_args()
    base = a.url.rstrip("/")
    if a.create_index:
        body = json.dumps({"mappings": {"properties": {"text": {"type": "text"}}}}).encode()
        req = urllib.request.Request(f"{base}/{a.index}", data=body, method="PUT", headers={"Content-Type": "application/json"})
        try:
            urllib.request.urlopen(req, timeout=60).read()
        except urllib.error.HTTPError as e:
            print("create index:", e.code, e.read()[:200])
    cut = int(a.bulk_mb * 1024 * 1024)
    q = queue.Queue(maxsize=a.threads * 2)
    stats = {"retries": 0, "item_errors": 0, "first_error": None, "bulks": 0}
    lock = threading.Lock()

    def worker():
        while True:
            item = q.get()
            if item is None:
                return
            body, ndocs = item
            delay = 0.25
            while True:
                status, resp = post(f"{base}/_bulk", body)
                if status in (429, 503):
                    with lock:
                        stats["retries"] += 1
                    time.sleep(delay)
                    delay = min(delay * 2, 8)
                    continue
                if status != 200:
                    with lock:
                        stats["item_errors"] += ndocs
                        stats["first_error"] = stats["first_error"] or f"HTTP {status}: {resp[:300]!r}"
                    break
                r = json.loads(resp)
                if not r.get("errors"):
                    break
                retry_lines = []
                lines = body.split(b"\n")
                for i, it in enumerate(r.get("items", [])):
                    op = next(iter(it.values()))
                    if op.get("status") == 429:
                        retry_lines += [lines[2 * i], lines[2 * i + 1]]
                    elif op.get("error"):
                        with lock:
                            stats["item_errors"] += 1
                            stats["first_error"] = stats["first_error"] or json.dumps(op.get("error"))[:300]
                if not retry_lines:
                    break
                with lock:
                    stats["retries"] += 1
                body = b"\n".join(retry_lines) + b"\n"
                ndocs = len(retry_lines) // 2
                time.sleep(delay)
                delay = min(delay * 2, 8)
            with lock:
                stats["bulks"] += 1

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(a.threads)]
    for t in threads:
        t.start()
    t0 = time.time()
    docs = 0
    buf = []
    size = 0
    nb = 0
    with open(a.corpus, "rb") as f:
        for line in f:
            if a.n and docs >= a.n:
                break
            d = json.loads(line)
            action = json.dumps({"index": {"_index": a.index, "_id": d["id"]}}).encode()
            doc = json.dumps({"text": d["text"]}, ensure_ascii=False).encode()
            buf.append(action)
            buf.append(doc)
            size += len(action) + len(doc) + 2
            docs += 1
            nb += 1
            if size >= cut:
                q.put((b"\n".join(buf) + b"\n", nb))
                buf, size, nb = [], 0, 0
    if buf:
        q.put((b"\n".join(buf) + b"\n", nb))
    for _ in threads:
        q.put(None)
    for t in threads:
        t.join()
    t_bulk = time.time() - t0
    status, _ = post(f"{base}/{a.index}/_refresh", b"", "application/json")
    t_ref = time.time() - t0
    with urllib.request.urlopen(f"{base}/{a.index}/_count", timeout=600) as r:
        count = json.loads(r.read()).get("count")
    res = {"docs_sent": docs, "count": count, "bulk_s": round(t_bulk, 2), "searchable_s": round(t_ref, 2),
           "refresh_status": status, "threads": a.threads, "bulk_mb": a.bulk_mb, **stats}
    print(json.dumps(res))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
