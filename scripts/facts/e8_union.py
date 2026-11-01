#!/usr/bin/env python3
"""Union of several model samples over one corpus (one harness process for all of them: one projection rebuild), and the
split of its outputs back into one file set per sample (qids of each sample's questions.jsonl `sample` flag).
  e8_union.py make  --out corpus_10m_all --base corpus_10m SAMPLEDIR...      (store files symlinked from --base)
  e8_union.py split --sample SAMPLEDIR --src RUNDIR --dst RUNDIR             (exact.jsonl and model/*/asks_*.jsonl)"""
import argparse, json, os
from pathlib import Path


def qids(d):
    out = set()
    for l in open(Path(d) / "questions.jsonl"):
        q = json.loads(l)
        if q.get("sample"):
            out.add(q["qid"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["make", "split"])
    ap.add_argument("--out")
    ap.add_argument("--base")
    ap.add_argument("--sample")
    ap.add_argument("--src")
    ap.add_argument("--dst")
    ap.add_argument("dirs", nargs="*")
    a = ap.parse_args()
    if a.cmd == "make":
        out, base = Path(a.out), Path(a.base).resolve()
        out.mkdir(parents=True, exist_ok=False)
        for f in ("phase1.jsonl", "phase2.jsonl", "world_facts.jsonl", "corpus.json"):
            os.symlink(base / f, out / f)
        want = set()
        for d in a.dirs:
            want |= qids(d)
        # worlds of every sample (the scorer's per-world visibility)
        seen = set()
        with open(out / "worlds_sample.jsonl", "w") as fo:
            for d in a.dirs:
                for l in open(Path(d) / "worlds_sample.jsonl"):
                    w = json.loads(l)
                    k = (w["world_file"], w["world_id"])
                    if k not in seen:
                        seen.add(k)
                        fo.write(l if l.endswith("\n") else l + "\n")
        n = 0
        with open(out / "questions.jsonl", "w") as fo:
            for l in open(base / "questions.jsonl"):
                q = json.loads(l)
                q["sample"] = q["qid"] in want
                n += q["sample"]
                fo.write(json.dumps(q, ensure_ascii=False) + "\n")
        print({"sample": n, "worlds": len(seen)})
    else:
        keep = qids(a.sample)
        src, dst = Path(a.src), Path(a.dst)
        files = [Path("exact.jsonl")] + [p.relative_to(src) for p in src.glob("model/**/asks_*.jsonl")]
        for rel in files:
            if not (src / rel).exists():
                continue
            (dst / rel).parent.mkdir(parents=True, exist_ok=True)
            n = 0
            with open(src / rel) as fi, open(dst / rel, "w") as fo:
                for l in fi:
                    if json.loads(l)["qid"] in keep:
                        fo.write(l)
                        n += 1
            print(rel, n)


if __name__ == "__main__":
    main()
