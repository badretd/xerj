#!/usr/bin/env python3
"""E8 round 3 (24_shared3), model-track design note: which relations the shared-store run follows where it leaves the
per-world run, over the losses of an e8_attrib.py output. Counts of relation labels only (Wikidata property labels of
the stored facts); no question text, no entity names in the output.

Per loss with a record read at the departure decision d:
  departure relation        the relation of the record the shared run read at d
  previous -> departure     the relation of the last record the shared run read before d, then the departure relation
                            ("(question)" when d is the first read)
  decision index            d (0 = the first action)
  shared-run length         actions in the shared run's program (6 = the step limit)
usage: e8_design_counts.py --attrib FILE --corpus DIR --asks DIR --out FILE [--top 12]
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
    ap.add_argument("--attrib", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--asks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--top", type=int, default=12)
    a = ap.parse_args()
    at = json.loads(Path(a.attrib).read_text())
    need = {r["qid"] for r in at["rows"]}
    sh = {r["qid"]: r for r in jl(Path(a.asks) / "asks_shared.jsonl") if r["qid"] in need}
    want = {t["fact_id"] for q in need for t in (sh[q].get("trace") or []) if t.get("fact_id")}
    rel = {}
    corpus = Path(a.corpus)
    for b in jl(corpus / "phase1.jsonl"):
        for f in b.get("facts", []):
            if f["fact_id"] in want:
                rel[f["fact_id"]] = f["relation"]
        if len(rel) == len(want):
            break
    for b in jl(corpus / "phase2.jsonl"):
        for e in b.get("edits", []):
            for f in ([e["fact"]] if e.get("fact") else []) + list(e.get("facts") or []):
                if f["fact_id"] in want:
                    rel[f["fact_id"]] = f["relation"]
    dep = collections.defaultdict(collections.Counter)
    pair = collections.defaultdict(collections.Counter)
    dix = collections.defaultdict(collections.Counter)
    length = collections.defaultdict(collections.Counter)
    missing = 0
    for r in at["rows"]:
        cls, d = r["class"], r["departure_decision"]
        ts = sh[r["qid"]].get("trace") or []
        length[cls][len(sh[r["qid"]].get("program") or [])] += 1
        dix[cls][d] += 1
        if d is None or d >= len(ts) or not ts[d].get("fact_id"):
            dep[cls]["(not a record read)"] += 1
            continue
        fid = ts[d]["fact_id"]
        if fid not in rel:
            missing += 1
            dep[cls]["(record not found)"] += 1
            continue
        prev = "(question)"
        for t in ts[:d]:
            if t.get("fact_id") in rel:
                prev = rel[t["fact_id"]]
        dep[cls][rel[fid]] += 1
        pair[cls][f"{prev} -> {rel[fid]}"] += 1
    top = lambda c: dict(c.most_common(a.top))
    out = {"label": at["label"], "losses": len(at["rows"]), "records not found": missing,
           "departure relation by class": {k: top(v) for k, v in dep.items()},
           "previous -> departure relation by class": {k: top(v) for k, v in pair.items()},
           "departure decision index by class": {k: {str(x): n for x, n in sorted(v.items(), key=lambda y: str(y[0]))} for k, v in dix.items()},
           "shared-run program length by class": {k: dict(sorted(v.items())) for k, v in length.items()},
           "all classes: departure relation": top(sum(dep.values(), collections.Counter())),
           "all classes: previous -> departure": top(sum(pair.values(), collections.Counter()))}
    Path(a.out).write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
