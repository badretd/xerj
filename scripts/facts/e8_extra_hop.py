#!/usr/bin/env python3
"""E8 round 3 (24_shared3), model-track design note: what the record the shared-store run took at its departure looks
like, over the losses of an e8_attrib.py output (counts only, no question text in the output).

Per loss (departure decision d, the shared run took a record (S, R, V) where the per-world run did something else):
  after the answer      the per-world run RETURNed at d and S is the per-world answer (normalized): one hop past it
  back edge             V (normalized) is an entity the program already had (a linked subject or an earlier value)
  relation in question  R shares a content word with the question (the RelPref rule of xerj-index: function words out,
                        plural s stripped, equal or a common prefix of >= 6 chars)
  same relation again   R is the relation of an earlier step of the same program
usage: e8_extra_hop.py --attrib FILE --corpus DIR --asks DIR [--asks-shared FILE] --out FILE
"""
import argparse, collections, json, re, unicodedata
from pathlib import Path

STOP = set("a an and are as at be been being by can could did do does for from had has have he her his how i if in into is it "
           "its of on or she that the their them there these they this those to was were what when where which who whom whose "
           "why will with would".split())


def jl(p):
    with open(p, encoding="utf-8") as f:
        for l in f:
            if l.strip():
                yield json.loads(l)


def content_words(s):
    out = []
    for w in re.split(r"[^0-9A-Za-zÀ-￿]+", s):
        if not w:
            continue
        w = w.lower()
        if w in STOP:
            continue
        if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        if w not in out:
            out.append(w)
    return out


def words_match(a, b):
    if a == b:
        return True
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n >= 6


def rel_score(question, phrase):
    q = content_words(question)
    return sum(1 for w in content_words(phrase) if any(words_match(x, w) for x in q))


def norm(s):
    s = unicodedata.normalize("NFKC", s or "").strip().lower()
    return " ".join(s.split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attrib", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--asks", required=True)
    ap.add_argument("--asks-shared", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    at = json.loads(Path(a.attrib).read_text())
    corpus = Path(a.corpus)
    qs = {q["qid"]: q for q in jl(corpus / "questions.jsonl")}
    need = {r["qid"] for r in at["rows"]}
    sh = {r["qid"]: r for r in jl(Path(a.asks_shared) if a.asks_shared else Path(a.asks) / "asks_shared.jsonl") if r["qid"] in need}
    pw = {r["qid"]: r for r in jl(Path(a.asks) / "asks_perworld.jsonl") if r["qid"] in need}
    # record contents: (subject, relation, value) of the facts the shared runs read (test part, phase 2, background ids)
    want = set()
    for q in need:
        for t in sh[q].get("trace") or []:
            if t.get("fact_id"):
                want.add(t["fact_id"])
    content = {}
    for b in jl(corpus / "phase1.jsonl"):
        for f in b.get("facts", []):
            if f["fact_id"] in want:
                content[f["fact_id"]] = (f["subject"], f["relation"], str(f.get("value")))
        if len(content) == len(want):
            break
    for b in jl(corpus / "phase2.jsonl"):
        for e in b.get("edits", []):
            for f in ([e["fact"]] if e.get("fact") else []) + list(e.get("facts") or []):
                if f["fact_id"] in want:
                    content[f["fact_id"]] = (f["subject"], f["relation"], str(f.get("value")))
    c = collections.Counter()
    per_class = collections.defaultdict(collections.Counter)
    for r in at["rows"]:
        q = qs[r["qid"]]
        s_, p_ = sh[r["qid"]], pw[r["qid"]]
        d = r["departure_decision"]
        ts = s_.get("trace") or []
        if d is None or d >= len(ts) or not ts[d].get("fact_id"):
            c["departure is not a record read"] += 1
            per_class[r["class"]]["departure is not a record read"] += 1
            continue
        fid = ts[d]["fact_id"]
        if fid not in content:
            c["record content not found (background beyond the scanned batches)"] += 1
            per_class[r["class"]]["record content not found"] += 1
            continue
        S, R, V = content[fid]
        tp = p_.get("trace") or []
        pw_ret = d < len(tp) and tp[d].get("op") == "RETURN"
        after = pw_ret and norm(S) == norm(p_.get("answer"))
        path = [norm(x) for x in ((s_.get("link") or {}).get("subjects") or [])]
        for t in ts[:d]:
            res = t.get("result") or ""
            m = re.match(r"VALUE text (.*)$", res)
            if m:
                path.append(norm(m.group(1)))
        back = norm(V) in path
        rel_in_q = rel_score(q["question"], R) > 0
        earlier_rels = set()
        for t in ts[:d]:
            if t.get("fact_id") in content:
                earlier_rels.add(content[t["fact_id"]][1])
        same_rel = R in earlier_rels
        for k, v in (("one hop past the per-world answer", after), ("back edge (value already on the path)", back),
                     ("relation mentioned in the question", rel_in_q), ("same relation as an earlier step", same_rel)):
            if v:
                c[k] += 1
                per_class[r["class"]][k] += 1
        c["records analysed"] += 1
        per_class[r["class"]]["records analysed"] += 1
        if after and not rel_in_q:
            c["one hop past the answer, relation not mentioned"] += 1
            per_class[r["class"]]["one hop past the answer, relation not mentioned"] += 1
        if after and rel_in_q:
            c["one hop past the answer, relation mentioned"] += 1
            per_class[r["class"]]["one hop past the answer, relation mentioned"] += 1
    out = {"label": at["label"], "losses": len(at["rows"]), "counts": dict(c),
           "by_class": {k: dict(v) for k, v in per_class.items()}}
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
