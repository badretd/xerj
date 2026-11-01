#!/usr/bin/env python3
"""A fresh model sample over an existing E8 corpus (confirmation run): the same store files, new `sample` flags.

The development sample (e8_corpus.py SAMPLE) holds the questions of mquake_t worlds [0, 100), mquake_t_chain worlds
[0, 50), twowiki_test worlds [0, 200) and all of direction_test, pick_num_test and hotpot_test. The confirmation sample
holds the next worlds of the three sets that have more: mquake_t [100, 200), mquake_t_chain [50, 100), twowiki_test
[200, 400). Store files (phase1/phase2/world_facts/corpus.json) are symlinked, not copied.

usage: e8_conf_sample.py --corpus corpus_1m --root $XERJ_R_ROOT/runs/world --out corpus_1m_conf
"""
import argparse, collections, json, os
from pathlib import Path

CONF = {"mquake_t": (100, 200), "mquake_t_chain": (50, 100), "twowiki_test": (200, 400)}


def jl(p):
    with open(p, encoding="utf-8") as f:
        for l in f:
            if l.strip():
                yield json.loads(l)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    src, out, root = Path(a.corpus).resolve(), Path(a.out), Path(a.root)
    out.mkdir(parents=True, exist_ok=False)
    for f in ("phase1.jsonl", "phase2.jsonl", "world_facts.jsonl", "corpus.json"):
        os.symlink(src / f, out / f)
    qs = list(jl(src / "questions.jsonl"))
    files = sorted({q["world_file"] for q in qs if q["set"] in CONF})
    widx = {}
    worlds = {}
    for wf in files:
        for i, w in enumerate(jl(root / wf)):
            widx[(wf, w["world_id"])] = i
            worlds[(wf, w["world_id"])] = w
    n = collections.Counter()
    keep = set()
    with open(out / "questions.jsonl", "w", encoding="utf-8") as fo:
        for q in qs:
            lo, hi = CONF.get(q["set"], (0, 0))
            i = widx.get((q["world_file"], q["world_id"]), -1)
            s = lo <= i < hi
            n[f"{q['set']}: dev sample"] += bool(q.get("sample"))
            q["sample"] = s
            if s:
                keep.add((q["world_file"], q["world_id"]))
                n[f"{q['set']}: confirmation sample"] += 1
            fo.write(json.dumps(q, ensure_ascii=False) + "\n")
    with open(out / "worlds_sample.jsonl", "w", encoding="utf-8") as fo:
        for (wf, wid) in sorted(keep):
            v = {k: x for k, x in worlds[(wf, wid)].items() if k != "items"}
            fo.write(json.dumps({"world_file": wf, "world_id": wid, "world": v}, ensure_ascii=False) + "\n")
    n["worlds in the confirmation sample"] = len(keep)
    (out / "conf_sample.json").write_text(json.dumps({"rule": CONF, "counts": dict(n)}, indent=1) + "\n")
    print(json.dumps(dict(n), indent=1))


if __name__ == "__main__":
    main()
