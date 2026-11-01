#!/usr/bin/env python3
"""Latency subset of the development sample: a fixed random quarter of each set's development questions (seed 20261028),
store files symlinked. Exact path only (candidate retrieval timing); never scored for the gate.
usage: e8_latsample.py --corpus corpus_10k --out corpus_10k_lat [--frac 0.25]"""
import argparse, collections, json, os, random
from pathlib import Path
ap = argparse.ArgumentParser()
ap.add_argument("--corpus", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--frac", type=float, default=0.25)
a = ap.parse_args()
src, out = Path(a.corpus).resolve(), Path(a.out)
out.mkdir(parents=True, exist_ok=False)
for f in ("phase1.jsonl", "phase2.jsonl", "world_facts.jsonl", "corpus.json", "worlds_sample.jsonl"):
    os.symlink(src / f, out / f)
qs = [json.loads(l) for l in open(src / "questions.jsonl")]
by = collections.defaultdict(list)
for q in qs:
    if q.get("sample"):
        by[q["set"]].append(q["qid"])
keep = set()
for s, ids in sorted(by.items()):
    ids = sorted(ids)
    keep |= set(random.Random(f"e8-lat|20261028|{s}").sample(ids, max(1, round(len(ids) * a.frac))))
with open(out / "questions.jsonl", "w") as fo:
    for q in qs:
        q["sample"] = q["qid"] in keep
        fo.write(json.dumps(q, ensure_ascii=False) + "\n")
print({"questions": len(keep), "by_set": {s: sum(1 for i in ids if i in keep) for s, ids in by.items()}})
