#!/usr/bin/env python3
"""Causes of the shared-store losses of one scored run (main cohort, per world supported and shared not), from the
shared-arm ask rows: where the shared answer's evidence comes from.
  hub CF     evidence from the counterfactual direction hub world (MQuAKE-CF based: false facts about real entities)
  background evidence from a Wikidata5M background record (wd5m/...)
  other world evidence from another test world
  no answer  shared status UNKNOWN / CONFLICT where the gold is an ANSWER
usage: e8_causes.py --corpus DIR --score score.json --asks DIR [--asks-base DIR] [--variant cf] --exact exact.jsonl"""
import argparse, collections, json, os, sys
from pathlib import Path


def jl(p):
    with open(p) as f:
        for l in f:
            if l.strip():
                yield json.loads(l)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--losses", required=True, help="e8_losses.py --json output")
    ap.add_argument("--asks", required=True)
    ap.add_argument("--asks-base", default=None)
    a = ap.parse_args()
    qs = {q["qid"]: q for q in jl(Path(a.corpus) / "questions.jsonl")}
    wf = {w["world_id"]: set(w["facts"]) for w in jl(Path(a.corpus) / "world_facts.jsonl")}
    rows = {r["qid"]: r for r in jl(Path(a.asks) / "asks_shared.jsonl")}
    if a.asks_base:
        base = {r["qid"]: r for r in jl(Path(a.asks_base) / "asks_shared.jsonl")}
        for k, r in list(rows.items()):
            if r.get("same_as_exact"):
                rows[k] = base.get(k, r)
    c = collections.Counter()
    out = []
    for l in json.load(open(a.losses)):
        if l["kind"] != "shared lost":
            continue
        q = qs[l["qid"]]
        r = rows[l["qid"]]
        ev = r.get("evidence_ids") or []
        own = wf.get(q["world_id"], set())
        foreign = [e for e in ev if e not in own]
        hub = [e for e in foreign if e.startswith("hub-")]
        bg = [e for e in foreign if e.startswith("wd5m/") or e.startswith("0wd5m/")]
        if r.get("status") != "ANSWER" and q["gold_status"] == "ANSWER":
            why = "no answer (" + str(r.get("status")) + ")"
        elif hub:
            why = "cites the counterfactual hub world"
        elif bg:
            why = "cites a background record"
        elif foreign:
            why = "cites another test world"
        else:
            why = "own-world evidence, wrong decision"
        c[(q["set"], why)] += 1
        out.append({"qid": l["qid"], "set": q["set"], "why": why, "answer": r.get("answer"), "gold": q["gold_answers"][:2]})
    for k, v in sorted(c.items()):
        print(v, " | ".join(k))
    print(json.dumps({"by_cause": {f"{k[0]} | {k[1]}": v for k, v in sorted(c.items())}}))


if __name__ == "__main__":
    main()
