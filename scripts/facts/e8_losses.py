#!/usr/bin/env python3
"""Discordant main-cohort questions of one scored model run (shared vs per world): what the shared arm got wrong and why.

usage: e8_losses.py --corpus DIR --exact FILE --asks DIR [--asks-base DIR --variant cf] [--json OUT]
                   [--scorer-dir research/xerj_r_mvp/dev_v5/future_scientific]
"""
import argparse, collections, json, os, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
scorer = None  # the frozen dev_v5 normalizer (research/xerj_r_mvp/dev_v5/future_scientific/scorer.py); not vendored here


def load_scorer(d):
    global scorer
    d = d or os.environ.get("XERJ_R_SCORER_DIR") or str(HERE / "frozen" / "future_scientific")
    sys.path.insert(0, d)
    import scorer as sc  # noqa: E402
    scorer = sc


def jl(p):
    with open(p, encoding="utf-8") as f:
        for l in f:
            if l.strip():
                yield json.loads(l)


def correct(status, answer, q):
    if status != q["gold_status"]:
        return False
    if status != "ANSWER":
        return True
    k = scorer.normalize_answer(answer)
    return bool(k) and k in {scorer.normalize_answer(x) for x in q["gold_answers"] if x is not None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--exact", required=True)
    ap.add_argument("--asks", required=True)
    ap.add_argument("--asks-base", default=None)
    ap.add_argument("--variant", default="")
    ap.add_argument("--json", default=None)
    ap.add_argument("--scorer-dir", default=None, help="directory of the frozen scorer.py (default $XERJ_R_SCORER_DIR)")
    a = ap.parse_args()
    load_scorer(a.scorer_dir)
    qs = {q["qid"]: q for q in jl(Path(a.corpus) / "questions.jsonl")}
    wf = {w["world_id"]: set(w["facts"]) for w in jl(Path(a.corpus) / "world_facts.jsonl")}
    asks = {}
    for arm in ("perworld", "shared"):
        asks[arm] = {r["qid"]: r for r in jl(Path(a.asks) / f"asks_{arm}.jsonl")}
        if a.asks_base:
            base = {r["qid"]: r for r in jl(Path(a.asks_base) / f"asks_{arm}.jsonl")}
            for qid, r in list(asks[arm].items()):
                if r.get("same_as_exact"):
                    asks[arm][qid] = base.get(qid, r)
    need = set(asks["perworld"]) & set(asks["shared"])
    sfx = f"_{a.variant}" if a.variant else ""
    ex = {}
    for r in jl(a.exact):
        if r["qid"] in need:
            ex[r["qid"]] = r
    # the scorer's identifiability rule for inverse steps with no record in the world (key ['', relation]): not
    # identifiable when the shared store holds an active fact with that relation and a value the default linker links
    absent_inv, pairs = {}, set()
    for qid, r in ex.items():
        for st in qs[qid]["trace"]:
            k = st.get("key")
            if isinstance(k, list) and len(k) == 2 and k[0] == "" and st.get("direction") == "in":
                vals = r["perworld"].get("values") or r["shared"].get("values") or []
                absent_inv[qid] = (k[1], vals)
                pairs.update((k[1], v) for v in vals)
    held = collections.defaultdict(list)
    p1, p2, sup, wd, doc_of = set(), set(), set(), set(), {}
    if pairs:
        def note(f):
            v = f.get("value")
            if isinstance(v, str) and (f.get("relation"), v) in pairs:
                held[(f["relation"], v)].append(f["fact_id"])
        for b in jl(Path(a.corpus) / "phase1.jsonl"):
            for f in b.get("facts", []):
                if f.get("relation") in {r for r, _ in pairs}:
                    note(f); p1.add(f["fact_id"]); doc_of[f["fact_id"]] = (f["doc_id"], f["version"])
        for b in jl(Path(a.corpus) / "phase2.jsonl"):
            for e in b.get("edits", []):
                for f in ([e["fact"]] if e.get("fact") else []) + list(e.get("facts") or []):
                    note(f); p2.add(f["fact_id"]); doc_of[f["fact_id"]] = (f["doc_id"], f["version"])
                if e.get("op") == "correct":
                    sup.add(e["supersedes"])
                if e.get("op") == "withdraw_document":
                    wd.add((e["doc_id"], e["version"]))

    def absent_held(qid, phase):
        if qid not in absent_inv:
            return False
        rel, vals = absent_inv[qid]
        for v in vals:
            for fid in held.get((rel, v), ()):
                if phase == "before" and fid in p1:
                    return True
                if phase != "before" and fid not in sup and doc_of.get(fid) not in wd:
                    return True
        return False
    out = []
    cat = collections.Counter()
    for qid in sorted(need):
        q = qs[qid]
        r = ex.get(qid)
        if r is None:
            continue
        pw, sh = r[f"perworld{sfx}"], r[f"shared{sfx}"]
        ident = all(("key" not in s1) or (s2.get("status") == s1.get("status") and sorted(s2.get("values", [])) == sorted(s1.get("values", []))
                    and (not s1.get("inverse") or s2.get("inverse_subjects", 0) == s1.get("inverse_subjects", 0)))
                    for s1, s2 in zip(pw["steps"], sh["steps"]))
        if not correct(pw["oracle"]["status"], pw["oracle"]["answer"], q) or not ident or absent_held(qid, q["phase"]):
            continue
        rp, rs = asks["perworld"][qid], asks["shared"][qid]
        if "http" in rp or "http" in rs:
            continue

        def ok(rr):
            ev = rr.get("evidence_ids") or []
            c = correct(rr["status"], rr["answer"], q)
            return c and (rr["status"] != "ANSWER" or bool(ev))
        po, so = ok(rp), ok(rs)
        if po == so:
            continue
        orc = correct(sh["oracle"]["status"], sh["oracle"]["answer"], q)
        foreign = [e for e in (rs.get("evidence_ids") or []) if e not in wf.get(q["world_id"], set())]
        why = []
        if not orc:
            why.append("exact path wrong in shared (" + ("; ".join(sh.get("problems") or [])[:60] or "no problem text") + ")")
        if sh["gold_lookups_in_cands"] < sh["gold_lookups"]:
            why.append(f"gold lookups in K=16 {sh['gold_lookups_in_cands']}/{sh['gold_lookups']} (per world {pw['gold_lookups_in_cands']})")
        if sorted(map(str, sh["link"])) != sorted(map(str, pw["link"])):
            why.append("link differs")
        if foreign:
            why.append(f"{len(foreign)} foreign evidence")
        if not why:
            why.append("model decision differs only (same oracle, gold in candidates)")
        kind = "shared lost" if po else "shared won"
        cat[(kind, q["set"], q.get("form"), why[0][:40])] += 1
        out.append({"kind": kind, "qid": qid, "set": q["set"], "form": q.get("form"), "phase": q["phase"], "question": q["question"],
                    "gold": [q["gold_status"], q["gold_answers"][:3]], "pw": [rp["status"], rp["answer"]],
                    "sh": [rs["status"], rs["answer"], rs.get("final_status"), rs.get("final_answer")],
                    "sh_program": rs.get("program"), "pw_program": rp.get("program"),
                    "link_pw": pw["link"], "link_sh": sh["link"], "values_pw": pw.get("values"), "values_sh": sh.get("values"),
                    "why": why, "foreign": foreign[:4], "trace": q.get("trace"),
                    "casefold": rs.get("link") if isinstance(rs.get("link"), dict) else None})
    for k, v in sorted(cat.items()):
        print(v, " | ".join(str(x) for x in k))
    for o in out:
        print(json.dumps({k: o[k] for k in ("kind", "set", "form", "phase", "question", "gold", "pw", "sh", "why", "link_pw", "link_sh")}, ensure_ascii=False)[:900])
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
