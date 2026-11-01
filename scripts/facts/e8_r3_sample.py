#!/usr/bin/env python3
"""E8 round 3 (24_shared3): a NEW development sample and the NEXT gate sample over an existing E8 corpus, both drawn
from the worlds that no earlier sample used, before any result on them is looked at.

Earlier samples (excluded from both draws):
  development (e8_corpus.py SAMPLE): mquake_t worlds [0, 100), mquake_t_chain [0, 50), twowiki_test [0, 200), and every
      world of direction_test, pick_num_test and hotpot_test (these three sets have no unused world);
  round-1 confirmation (e8_conf_sample.py): mquake_t [100, 200), mquake_t_chain [50, 100), twowiki_test [200, 400);
  round-2 confirmation (e8_conf2_sample.py, seed 20261028; scored, so development data now): its drawn world indices,
      read from its manifest (--conf2-manifest).
Draw rule (both samples): for each set, random.Random(f"e8-r3-{kind}|{seed}|{set}").sample(sorted pool, N[set]); the
gate sample's pool is the development draw's pool minus the development draw. Every question of a drawn world is in its
sample.

--mode dev   writes <out>/questions.jsonl (sample flags), worlds_sample.jsonl, r3_dev_sample.json and symlinks the store
             files (the per-world namespaces hold exactly the drawn worlds)
--mode gate  writes ONLY <out>/r3_gate_sample.json: the seed, the rule, the counts and the sha256 of the sorted question
             id list and of the sorted world list (the ids themselves are not written: they follow from the seed)

usage: e8_r3_sample.py --mode dev|gate --corpus corpus_10k --root $XERJ_R_ROOT/runs/world --conf2-manifest FILE
                       --out DIR --seed N [--dev-manifest FILE (gate mode)]
"""
import argparse, collections, hashlib, json, os, random, subprocess
from pathlib import Path

USED = {"mquake_t": [(0, 100), (100, 200)], "mquake_t_chain": [(0, 50), (50, 100)], "twowiki_test": [(0, 200), (200, 400)]}
N_DEV = {"mquake_t": 100, "mquake_t_chain": 100, "twowiki_test": 20}
N_GATE = {"mquake_t": 250, "mquake_t_chain": 100, "twowiki_test": 73}


def jl(p):
    with open(p, encoding="utf-8") as f:
        for l in f:
            if l.strip():
                yield json.loads(l)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("dev", "gate"), required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--conf2-manifest", required=True)
    ap.add_argument("--dev-manifest", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, required=True)
    a = ap.parse_args()
    src, out, root = Path(a.corpus).resolve(), Path(a.out), Path(a.root)
    conf2 = json.loads(Path(a.conf2_manifest).read_text())
    c2 = {s: set(v) for s, v in conf2["drawn_world_indices"].items()}
    N = N_DEV if a.mode == "dev" else N_GATE
    qs = list(jl(src / "questions.jsonl"))
    files = sorted({q["world_file"] for q in qs if q["set"] in N})
    widx, worlds = {}, {}
    for wf in files:
        for i, w in enumerate(jl(root / wf)):
            widx[(wf, w["world_id"])] = i
            worlds[(wf, w["world_id"])] = w
    pool = collections.defaultdict(set)
    for q in qs:
        if q["set"] not in N:
            continue
        i = widx[(q["world_file"], q["world_id"])]
        if any(lo <= i < hi for lo, hi in USED[q["set"]]) or i in c2.get(q["set"], set()):
            continue
        pool[q["set"]].add(i)
    excl_dev = {}
    if a.mode == "gate":
        dev = json.loads(Path(a.dev_manifest).read_text())
        excl_dev = {s: set(v) for s, v in dev["drawn_world_indices"].items()}
        for s in pool:
            pool[s] -= excl_dev.get(s, set())
    drawn = {}
    for s in sorted(N):
        p = sorted(pool[s])
        rng = random.Random(f"e8-r3-{a.mode}|{a.seed}|{s}")
        drawn[s] = set(rng.sample(p, min(N[s], len(p))))
    n = collections.Counter()
    keep, qids = set(), []
    for q in qs:
        i = widx.get((q["world_file"], q["world_id"]), -1)
        if q["set"] in drawn and i in drawn[q["set"]]:
            keep.add((q["world_file"], q["world_id"]))
            n[f"{q['set']}: questions"] += 1
            qids.append(q["qid"])
    for s in sorted(N):
        n[f"{s}: unused worlds in the pool"] = len(pool[s])
        n[f"{s}: worlds drawn"] = len(drawn[s])
    n["worlds in the sample"] = len(keep)
    n["questions in the sample"] = len(qids)
    qids.sort()
    wl = sorted(f"{wf}|{wid}" for wf, wid in keep)
    man = {"kind": f"E8 round 3 {a.mode} sample", "rule": "random worlds per set from the unused pool", "seed": a.seed,
           "rng": f"random.Random('e8-r3-{a.mode}|{a.seed}|<set>').sample(sorted pool, N)", "n_worlds": N,
           "excluded": {"ranges": USED, "round-2 confirmation (seed 20261028)": {s: len(v) for s, v in c2.items()},
                        "round-3 development sample": {s: len(v) for s, v in excl_dev.items()} if excl_dev else None},
           "counts": dict(n), "sample_qids_sha256": hashlib.sha256("\n".join(qids).encode()).hexdigest(),
           "sample_worlds_sha256": hashlib.sha256("\n".join(wl).encode()).hexdigest(),
           "drawn_at": subprocess.check_output(["date", "+%F %T %Z"]).decode().strip(),
           "source_questions_sha256": hashlib.sha256(open(src / "questions.jsonl", "rb").read()).hexdigest(),
           "conf2_manifest_sha256": hashlib.sha256(Path(a.conf2_manifest).read_bytes()).hexdigest()}
    out.mkdir(parents=True, exist_ok=(a.mode == "gate"))
    if a.mode == "dev":
        man["drawn_world_indices"] = {s: sorted(v) for s, v in drawn.items()}
        for f in ("phase1.jsonl", "phase2.jsonl", "world_facts.jsonl", "corpus.json"):
            os.symlink(src / f, out / f)
        qset = set(qids)
        with open(out / "questions.jsonl", "w", encoding="utf-8") as fo:
            for q in qs:
                q["sample"] = q["qid"] in qset
                fo.write(json.dumps(q, ensure_ascii=False) + "\n")
        with open(out / "worlds_sample.jsonl", "w", encoding="utf-8") as fo:
            for (wf, wid) in sorted(keep):
                v = {k: x for k, x in worlds[(wf, wid)].items() if k != "items"}
                fo.write(json.dumps({"world_file": wf, "world_id": wid, "world": v}, ensure_ascii=False) + "\n")
        (out / "r3_dev_sample.json").write_text(json.dumps(man, indent=1) + "\n")
    else:
        man["dev_manifest_sha256"] = hashlib.sha256(Path(a.dev_manifest).read_bytes()).hexdigest()
        (out / "r3_gate_sample.json").write_text(json.dumps(man, indent=1) + "\n")
    print(json.dumps({k: man[k] for k in ("kind", "seed", "counts", "sample_qids_sha256", "sample_worlds_sha256", "drawn_at")}, indent=1))


if __name__ == "__main__":
    main()
