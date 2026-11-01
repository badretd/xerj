#!/usr/bin/env python3
"""Casefold pre-link rewrites of two exact-path files (e.g. v1 and v2 of the linker) on the questions both contain:
how many questions each version rewrites per arm and set, and the questions whose rewrite differs.

usage: e8_rewrites.py --corpus DIR A.jsonl B.jsonl [--examples N]
"""
import argparse, collections, json
from pathlib import Path


def jl(p):
    with open(p, encoding="utf-8") as f:
        for l in f:
            if l.strip():
                yield json.loads(l)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--examples", type=int, default=12)
    a = ap.parse_args()
    qs = {q["qid"]: q for q in jl(Path(a.corpus) / "questions.jsonl")}
    A = {r["qid"]: r for r in jl(a.a)}
    B = {r["qid"]: r for r in jl(a.b)}
    both = sorted(set(A) & set(B))
    c = collections.Counter()
    ex = []
    for qid in both:
        s = qs[qid]["set"]
        for arm in ("perworld_cf", "shared_cf"):
            ra, rb = A[qid][arm].get("rewritten"), B[qid][arm].get("rewritten")
            c[(arm, s, "rewritten A")] += ra is not None
            c[(arm, s, "rewritten B")] += rb is not None
            if ra != rb:
                c[(arm, s, "differ")] += 1
                if len(ex) < a.examples:
                    ex.append({"arm": arm, "set": s, "question": qs[qid]["question"], "A": ra, "B": rb,
                               "link A": A[qid][arm]["link"], "link B": B[qid][arm]["link"]})
    print(f"questions in both: {len(both)}")
    for k in sorted(c):
        print(" | ".join(k), c[k])
    for e in ex:
        print(json.dumps(e, ensure_ascii=False)[:700])


if __name__ == "__main__":
    main()
