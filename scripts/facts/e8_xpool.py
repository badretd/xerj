#!/usr/bin/env python3
"""Exact-path analysis corpus: every question EXCEPT those of the sealed new confirmation sample (the `sample` flag set on
all others), store files symlinked. Used only for exact-path (no model) development counts, never for the gate.
usage: e8_xpool.py --corpus corpus_10k --conf2 corpus_10k_conf2 --out corpus_10k_xpool"""
import argparse, json, os
from pathlib import Path
ap = argparse.ArgumentParser()
ap.add_argument("--corpus", required=True)
ap.add_argument("--conf2", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
src, out = Path(a.corpus).resolve(), Path(a.out)
out.mkdir(parents=True, exist_ok=False)
for f in ("phase1.jsonl", "phase2.jsonl", "world_facts.jsonl", "corpus.json"):
    os.symlink(src / f, out / f)
sealed_worlds = set()
for l in open(Path(a.conf2) / "questions.jsonl"):
    q = json.loads(l)
    if q["sample"]:
        sealed_worlds.add((q["world_file"], q["world_id"]))
n = {"kept": 0, "sealed": 0}
keep_w = set()
with open(out / "questions.jsonl", "w") as fo:
    for l in open(src / "questions.jsonl"):
        q = json.loads(l)
        # world-level: no question of a world that holds a sealed question (of any set)
        s = (q["world_file"], q["world_id"]) not in sealed_worlds
        q["sample"] = s
        n["kept" if s else "sealed"] += 1
        if s:
            keep_w.add((q["world_file"], q["world_id"]))
        fo.write(json.dumps(q, ensure_ascii=False) + "\n")
os.symlink(src / "worlds_sample.jsonl", out / "worlds_sample.jsonl")
n["worlds"] = len(keep_w)
(out / "xpool.json").write_text(json.dumps(n) + "\n")
print(n)
