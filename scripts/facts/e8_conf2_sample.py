#!/usr/bin/env python3
"""A NEW confirmation sample over an existing E8 corpus (round 2): worlds drawn at random with a recorded seed from the
worlds that no earlier model sample used, drawn before any result on them is looked at. Store files are symlinked.

Earlier model samples (excluded from the draw):
  development (e8_corpus.py SAMPLE): mquake_t worlds [0, 100), mquake_t_chain [0, 50), twowiki_test [0, 200), and every
      world of direction_test, pick_num_test and hotpot_test (so these three sets have no unused world left);
  round-1 confirmation (e8_conf_sample.py CONF): mquake_t [100, 200), mquake_t_chain [50, 100), twowiki_test [200, 400).
Draw: for each set, random.Random(f"e8-conf2|{seed}|{set}").sample(sorted unused world indices, N[set]); every question of
a drawn world is in the sample (the per-world namespaces hold exactly the drawn worlds).

usage: e8_conf2_sample.py --corpus corpus_1m --root $XERJ_R_ROOT/runs/world --out corpus_1m_conf2 [--seed 20261028]
"""
import argparse, collections, hashlib, json, os, random, subprocess
from pathlib import Path

USED = {"mquake_t": [(0, 100), (100, 200)], "mquake_t_chain": [(0, 50), (50, 100)], "twowiki_test": [(0, 200), (200, 400)]}
N = {"mquake_t": 250, "mquake_t_chain": 100, "twowiki_test": 300}
SEED = 20261028


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
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args()
    src, out, root = Path(a.corpus).resolve(), Path(a.out), Path(a.root)
    out.mkdir(parents=True, exist_ok=False)
    for f in ("phase1.jsonl", "phase2.jsonl", "world_facts.jsonl", "corpus.json"):
        os.symlink(src / f, out / f)
    qs = list(jl(src / "questions.jsonl"))
    files = sorted({q["world_file"] for q in qs if q["set"] in N})
    widx, worlds = {}, {}
    for wf in files:
        for i, w in enumerate(jl(root / wf)):
            widx[(wf, w["world_id"])] = i
            worlds[(wf, w["world_id"])] = w
    # unused world indices per set (worlds that hold at least one question of the set)
    unused = collections.defaultdict(set)
    for q in qs:
        if q["set"] not in N:
            continue
        i = widx[(q["world_file"], q["world_id"])]
        if not any(lo <= i < hi for lo, hi in USED[q["set"]]):
            unused[q["set"]].add(i)
    drawn = {}
    for s in sorted(N):
        pool = sorted(unused[s])
        rng = random.Random(f"e8-conf2|{a.seed}|{s}")
        drawn[s] = set(rng.sample(pool, min(N[s], len(pool))))
    n = collections.Counter()
    keep = set()
    qids = []
    with open(out / "questions.jsonl", "w", encoding="utf-8") as fo:
        for q in qs:
            i = widx.get((q["world_file"], q["world_id"]), -1)
            s = q["set"] in drawn and i in drawn[q["set"]]
            q["sample"] = s
            if s:
                keep.add((q["world_file"], q["world_id"]))
                n[f"{q['set']}: new confirmation sample (questions)"] += 1
                qids.append(q["qid"])
            fo.write(json.dumps(q, ensure_ascii=False) + "\n")
    with open(out / "worlds_sample.jsonl", "w", encoding="utf-8") as fo:
        for (wf, wid) in sorted(keep):
            v = {k: x for k, x in worlds[(wf, wid)].items() if k != "items"}
            fo.write(json.dumps({"world_file": wf, "world_id": wid, "world": v}, ensure_ascii=False) + "\n")
    for s in sorted(N):
        n[f"{s}: unused worlds in the pool"] = len(unused[s])
        n[f"{s}: worlds drawn"] = len(drawn[s])
    n["worlds in the sample"] = len(keep)
    qids.sort()
    man = {"rule": "random worlds per set from the unused pool", "seed": a.seed, "n_worlds": N, "excluded": USED,
           "drawn_world_indices": {s: sorted(v) for s, v in drawn.items()}, "counts": dict(n),
           "sample_qids_sha256": hashlib.sha256("\n".join(qids).encode()).hexdigest(),
           "drawn_at": subprocess.check_output(["date", "+%F %T %Z"]).decode().strip(),
           "source_questions_sha256": hashlib.sha256(open(src / "questions.jsonl", "rb").read()).hexdigest()}
    (out / "conf2_sample.json").write_text(json.dumps(man, indent=1) + "\n")
    print(json.dumps({k: man[k] for k in ("seed", "counts", "sample_qids_sha256", "drawn_at")}, indent=1))


if __name__ == "__main__":
    main()
