#!/usr/bin/env python3
"""Row-by-row identity of two ask runs (e.g. the dev/base release server over HTTP vs dev/shared in process, both with
the default exact linker): status, answer, evidence ids, program, verify verdict, final answer, decision count.

usage: e8_compare.py DIR_A DIR_B [--arms perworld,shared]
"""
import argparse, collections, json
from pathlib import Path

KEYS = ("status", "answer", "evidence_ids", "program", "final_status", "final_answer", "decisions", "candidate_ids")


def jl(p):
    with open(p, encoding="utf-8") as f:
        for l in f:
            if l.strip():
                yield json.loads(l)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--arms", default="perworld,shared")
    a = ap.parse_args()
    out = {}
    for arm in a.arms.split(","):
        A = {r["qid"]: r for r in jl(Path(a.a) / f"asks_{arm}.jsonl") if "http" not in r and not r.get("same_as_exact")}
        B = {r["qid"]: r for r in jl(Path(a.b) / f"asks_{arm}.jsonl") if "http" not in r and not r.get("same_as_exact")}
        both = sorted(set(A) & set(B))
        c = collections.Counter()
        ex = []
        for q in both:
            same = all(A[q].get(k) == B[q].get(k) for k in KEYS)
            v = (A[q].get("verify") or {}).get("veto") == (B[q].get("verify") or {}).get("veto")
            c["identical"] += same and v
            if not (same and v) and len(ex) < 5:
                ex.append({"qid": q, "diff": [k for k in KEYS if A[q].get(k) != B[q].get(k)]})
        out[arm] = {"compared": len(both), "identical": c["identical"], "examples": ex}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
