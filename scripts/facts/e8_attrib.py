#!/usr/bin/env python3
"""E8 round 3 (24_shared3): where a shared-store answer first departs from the one-store-per-world answer, and why.

Losses = main-cohort questions that one store per world answers (raw supported) and the shared store does not (the gate's
metric; e8_score.py / e8_losses.py definitions, cohort rule replicated from e8_losses.py). For each loss the two runs'
executed actions are compared decision by decision on (op, traversal direction, CONTENT of the record the action read:
subject, relation, kind, value, unit, scope, validity). The shared store keeps one canonical copy of each content, the
per-world store the world's own copy, so content is the common key. The first decision where they differ is the
departure; its class:

  linker or manifest choice   the shared run linked other subjects, or the record the per-world run took at that decision
                              is not among the shared run's candidates there (base manifest or hop manifest)
  loop to the step limit      the shared run used all MAX_STEPS = 6 actions without RETURN (UNKNOWN, action budget)
  counterfactual hub fact     the shared run took a record of the direction test's hub world (hub-*: MQuAKE-CF edits)
  background fact             the shared run took a Wikidata5M background record (wd5m/*)
  fact from another world     the shared run took a record that is not a fact of the question's own world
  other                       the shared run took an own-world record or another op (e.g. RETURN, a comparison) while
                              the per-world choice was offered
Precedence in this order: a step-limit loop that began with a foreign record counts as a loop; the 2-D table
(what was taken at the departure x how the shared run ended) keeps both.

Interval: worlds of the main cohort resampled with replacement (B resamples, seed recorded), 95 % percentile interval of
each class count. Output: categories, counts and question ids only (no question text).

usage: e8_attrib.py --corpus DIR --exact FILE --asks DIR --variant cf --scorer-dir DIR --label NAME --out FILE.json
                    [--b 2000] [--seed 20261029] [--world-corpus DIR (test facts; default --corpus)]
"""
import argparse, collections, json, os, random, sys
from pathlib import Path

MAX_STEPS = 6
CLASSES = ["linker or manifest choice", "loop to the step limit", "counterfactual hub fact", "background fact",
           "fact from another world", "other"]
scorer = None


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


def content_key(f):  # e8_corpus.content_key
    v = f.get("value")
    v = ("b", v) if isinstance(v, bool) else ("s", "" if v is None else str(v))
    return (f["subject"], f["relation"], f.get("kind", "text"), v, f.get("unit", "") or "", f.get("scope", "default") or "default",
            f.get("valid_from"), f.get("valid_to"))


def edit_facts(e):
    return ([e["fact"]] if e.get("fact") else []) + list(e.get("facts") or [])


def pctl(v, q):
    v = sorted(x for x in v if x is not None)
    return v[min(len(v) - 1, int(round(q * (len(v) - 1))))] if v else None


def source(fid, own):
    if fid is None:
        return None
    if fid in own:
        return "own world"
    if fid.startswith("hub-"):
        return "counterfactual hub"
    if fid.startswith("wd5m/") or fid.startswith("0wd5m/"):
        return "background"
    return "another world"


def action(t, content):
    """(op, inverse, content key or None) of one executed trace entry"""
    fid = t.get("fact_id")
    return (t.get("op"), bool(t.get("inverse")), content(fid) if fid else None)


def main():
    global scorer
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--world-corpus", default=None, help="corpus dir whose phase1 test part / phase2 give the record contents")
    ap.add_argument("--exact", required=True)
    ap.add_argument("--asks", required=True, help="dir with asks_perworld.jsonl and asks_shared.jsonl")
    ap.add_argument("--asks-shared", default=None, help="another asks_shared.jsonl (an option's run) instead of --asks'")
    ap.add_argument("--variant", default="cf")
    ap.add_argument("--scorer-dir", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--b", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261029)
    a = ap.parse_args()
    sys.path.insert(0, a.scorer_dir)
    import scorer as sc  # noqa: E402  (frozen dev_v5 normalizer, read from its own directory)
    scorer = sc
    corpus = Path(a.corpus)
    wcorpus = Path(a.world_corpus) if a.world_corpus else corpus
    qs = {q["qid"]: q for q in jl(corpus / "questions.jsonl")}
    wf = {w["world_id"]: set(w["facts"]) for w in jl(corpus / "world_facts.jsonl")}
    asks = {"perworld": {r["qid"]: r for r in jl(Path(a.asks) / "asks_perworld.jsonl")},
            "shared": {r["qid"]: r for r in jl(Path(a.asks_shared) if a.asks_shared else Path(a.asks) / "asks_shared.jsonl")}}
    need = set(asks["perworld"]) & set(asks["shared"])
    sfx = f"_{a.variant}" if a.variant else ""
    ex = {r["qid"]: r for r in jl(a.exact) if r["qid"] in need}

    # ---- cohort (e8_losses.py / e8_score.py): per-world oracle correct, every gold step key identifiable, absent inverse
    absent_inv, pairs = {}, set()
    for qid, r in ex.items():
        for st in qs[qid]["trace"]:
            k = st.get("key")
            if isinstance(k, list) and len(k) == 2 and k[0] == "" and st.get("direction") == "in":
                vals = r["perworld"].get("values") or r["shared"].get("values") or []
                absent_inv[qid] = (k[1], vals)
                pairs.update((k[1], v) for v in vals)
    # record contents of the shared store's test part (phase-1 "test" batches, phase-2 edits) and the absent-inverse scan
    shared_content = {}
    held = collections.defaultdict(list)
    p1, sup_, wd, doc_of = set(), set(), set(), {}
    rels = {r for r, _ in pairs}

    def note(f):
        v = f.get("value")
        if isinstance(v, str) and (f.get("relation"), v) in pairs:
            held[(f["relation"], v)].append(f["fact_id"])
            doc_of[f["fact_id"]] = (f["doc_id"], f["version"])
    for b in jl(wcorpus / "phase1.jsonl"):
        if b.get("part") != "test" and not pairs:
            break  # the test part comes first; the background is only needed for the absent-inverse rule
        for f in b.get("facts", []):
            if b.get("part") == "test":
                shared_content[f["fact_id"]] = content_key(f)
            if f.get("relation") in rels:
                note(f); p1.add(f["fact_id"])
    for b in jl(wcorpus / "phase2.jsonl"):
        for e in b.get("edits", []):
            for f in edit_facts(e):
                shared_content[f["fact_id"]] = content_key(f)
                note(f)
            if e.get("op") == "correct":
                sup_.add(e["supersedes"])
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
                if phase != "before" and fid not in sup_ and doc_of.get(fid) not in wd:
                    return True
        return False
    # per-world record contents (the world's own fact ids)
    pw_content = {}
    for w in jl(corpus / "worlds_sample.jsonl"):
        ww = w["world"]
        m = {}
        for f in ww.get("base_facts", []):
            m[f["fact_id"]] = content_key(f)
        for e in ww.get("edits", []):
            for f in edit_facts(e):
                m[f["fact_id"]] = content_key(f)
        pw_content[w["world_id"]] = m

    main_q, losses, wins = [], [], []
    okmap = {}
    for qid in sorted(need):
        q, r = qs[qid], ex.get(qid)
        if r is None:
            continue
        pw, sh = r[f"perworld{sfx}"], r[f"shared{sfx}"]
        ident = all(("key" not in s1) or (s2.get("status") == s1.get("status") and sorted(s2.get("values", [])) == sorted(s1.get("values", []))
                    and (not s1.get("inverse") or s2.get("inverse_subjects", 0) == s1.get("inverse_subjects", 0)))
                    for s1, s2 in zip(pw["steps"], sh["steps"]))
        if not correct(pw["oracle"]["status"], pw["oracle"]["answer"], q) or not ident or absent_held(qid, q["phase"]):
            continue
        rp, rs = asks["perworld"][qid], asks["shared"][qid]
        if "http" in rp or "http" in rs or "skipped" in rp:
            continue
        main_q.append(qid)

        def ok(rr):
            ev = rr.get("evidence_ids") or []
            return correct(rr["status"], rr["answer"], q) and (rr["status"] != "ANSWER" or bool(ev))
        po, so = ok(rp), ok(rs)
        okmap[qid] = [int(po), int(so)]
        if po and not so:
            losses.append(qid)
        elif so and not po:
            wins.append(qid)

    rows = []
    for qid in losses:
        q = qs[qid]
        wid = q["world_id"]
        own = wf.get(wid, set())
        rp, rs = asks["perworld"][qid], asks["shared"][qid]
        pwc = pw_content.get(wid, {})
        cp = lambda fid: pwc.get(fid, ("?pw", fid))
        cs = lambda fid: shared_content.get(fid, ("?bg" if fid.startswith(("wd5m/", "0wd5m/")) else "?sh", fid))
        tp, ts = rp.get("trace") or [], rs.get("trace") or []
        d = None
        for i in range(max(len(tp), len(ts))):
            ap_ = action(tp[i], cp) if i < len(tp) else None
            as_ = action(ts[i], cs) if i < len(ts) else None
            if ap_ != as_:
                d = i
                break
        link_p = sorted(map(str, (rp.get("link") or {}).get("subjects") or [])) if isinstance(rp.get("link"), dict) else None
        link_s = sorted(map(str, (rs.get("link") or {}).get("subjects") or [])) if isinstance(rs.get("link"), dict) else None
        link_differs = link_p is not None and link_s is not None and link_p != link_s
        # the shared run's candidates at the departure decision
        offered = None
        pw_took = sh_took = None
        if d is not None:
            if d < len(tp) and tp[d].get("fact_id"):
                pw_took = cp(tp[d]["fact_id"])
                if d == 0:
                    cands = rs.get("candidate_ids") or []
                else:
                    cands = next((h["ids"] for h in (rs.get("hop_reads") or []) if h["decision"] == d), [])
                offered = pw_took in {cs(c) for c in cands}
            if d < len(ts):
                sh_took = ts[d].get("fact_id")
        sh_src = source(sh_took, own) if sh_took else None
        sh_op = ts[d]["op"] if (d is not None and d < len(ts)) else "(trace ended)"
        pw_op = tp[d]["op"] if (d is not None and d < len(tp)) else "(trace ended)"
        n_act = len(rs.get("program") or [])
        last_op = (rs.get("program") or [[None]])[-1][0] if rs.get("program") else None
        step_limit = rs["status"] == "UNKNOWN" and n_act >= MAX_STEPS and last_op not in ("RETURN", "ABSTAIN")
        fids = [t.get("fact_id") for t in ts if t.get("fact_id")]
        revisit = len(fids) != len(set(fids))
        if rs["status"] == "ANSWER":
            outcome = "wrong answer" if not correct(rs["status"], rs["answer"], q) else "answer without evidence"
        elif step_limit:
            outcome = "step limit (UNKNOWN)"
        else:
            outcome = f"{rs['status']} (model)"
        # primary class (precedence as in the docstring)
        if d is None:
            cls = "other"
        elif link_differs or offered is False:
            cls = "linker or manifest choice"
        elif step_limit:
            cls = "loop to the step limit"
        elif sh_src == "counterfactual hub":
            cls = "counterfactual hub fact"
        elif sh_src == "background":
            cls = "background fact"
        elif sh_src == "another world":
            cls = "fact from another world"
        else:
            cls = "other"
        took = (sh_src or f"op {sh_op}")
        rows.append({"qid": qid, "set": q["set"], "form": q.get("form"), "world": f"{q['world_file']}|{wid}",
                     "departure_decision": d, "per_world_action": pw_op, "shared_action": sh_op, "shared_took": took,
                     "per_world_choice_offered": offered, "link_differs": link_differs, "outcome": outcome,
                     "step_limit": step_limit, "record_revisit": revisit, "class": cls,
                     "shared_status": rs["status"], "per_world_steps": len(tp), "shared_steps": len(ts)})

    # ---- world-cluster bootstrap of each class count (worlds of the main cohort)
    worlds = sorted({f"{qs[q]['world_file']}|{qs[q]['world_id']}" for q in main_q})
    per_world = collections.defaultdict(collections.Counter)
    for r_ in rows:
        per_world[r_["world"]][r_["class"]] += 1
        per_world[r_["world"]]["all losses"] += 1
    for q_ in wins:
        per_world[f"{qs[q_]['world_file']}|{qs[q_]['world_id']}"]["wins"] += 1
    keys = CLASSES + ["all losses", "wins"]
    rng = random.Random(f"e8-attrib|{a.seed}|{a.label}")
    samples = {k: [] for k in keys}
    for _ in range(a.b):
        tot = collections.Counter()
        for _w in range(len(worlds)):
            tot.update(per_world.get(worlds[rng.randrange(len(worlds))], {}))
        for k in keys:
            samples[k].append(tot[k])
    obs = collections.Counter()
    for r_ in rows:
        obs[r_["class"]] += 1
    obs["all losses"] = len(rows)
    obs["wins"] = len(wins)

    def ci(v):
        v = sorted(v)
        return [v[int(0.025 * (len(v) - 1))], v[int(round(0.975 * (len(v) - 1)))]]
    table = {k: {"count": obs[k], "ci95": ci(samples[k]), "points": round(100.0 * obs[k] / max(1, len(main_q)), 2)} for k in keys}
    two_d = collections.Counter((r_["shared_took"], r_["outcome"]) for r_ in rows)
    by_set = collections.Counter((r_["set"], r_["class"]) for r_ in rows)
    dep = collections.Counter((r_["departure_decision"], r_["per_world_action"], r_["shared_action"]) for r_ in rows)
    out = {"label": a.label, "n_main": len(main_q), "worlds": len(worlds), "losses": len(rows), "wins": len(wins),
           "bootstrap": {"B": a.b, "seed": a.seed, "rng": f"random.Random('e8-attrib|{a.seed}|{a.label}')", "unit": "world"},
           "classes": table,
           "taken_at_departure_x_outcome": {f"{k[0]} | {k[1]}": v for k, v in sorted(two_d.items())},
           "by_set_x_class": {f"{k[0]} | {k[1]}": v for k, v in sorted(by_set.items())},
           "departure_decision_x_actions": {f"d{k[0]} | per world {k[1]} | shared {k[2]}": v for k, v in sorted(dep.items(), key=lambda x: (str(x[0][0]), x[0][1], x[0][2]))},
           "flags": {"record revisit in the shared trace": sum(r_["record_revisit"] for r_ in rows),
                     "link differs": sum(r_["link_differs"] for r_ in rows),
                     "per-world choice not offered": sum(1 for r_ in rows if r_["per_world_choice_offered"] is False),
                     "unmapped record contents (per world / shared)": [sum(1 for q_ in losses for t in (asks['perworld'][q_].get('trace') or []) if t.get('fact_id') and t['fact_id'] not in pw_content.get(qs[q_]['world_id'], {})),
                                                                      sum(1 for q_ in losses for t in (asks['shared'][q_].get('trace') or []) if t.get('fact_id') and t['fact_id'] not in shared_content and not t['fact_id'].startswith(('wd5m/', '0wd5m/')))]},
           "rows": rows, "wins_qids": wins, "main_ok": okmap,
           "shared_arm": {"decisions_mean": round(sum((asks["shared"][q_].get("decisions") or 0) for q_ in main_q) / max(1, len(main_q)), 3),
                          "client_ms_p50": pctl([asks["shared"][q_].get("client_ms") for q_ in main_q], 0.5),
                          "client_ms_p95": pctl([asks["shared"][q_].get("client_ms") for q_ in main_q], 0.95),
                          "guard_fallbacks": sum((asks["shared"][q_].get("fallbacks") or 0) for q_ in main_q),
                          "loop_stopped": sum(1 for q_ in main_q if ((asks["shared"][q_].get("answer_opts") or {}).get("loop_stopped"))),
                          "step_limit": sum(1 for q_ in main_q if asks["shared"][q_].get("status") == "UNKNOWN" and len(asks["shared"][q_].get("program") or []) >= MAX_STEPS and (asks["shared"][q_].get("program") or [[None]])[-1][0] not in ("RETURN", "ABSTAIN"))},
           "perworld_arm": {"decisions_mean": round(sum((asks["perworld"][q_].get("decisions") or 0) for q_ in main_q) / max(1, len(main_q)), 3),
                            "record_revisits": sum(1 for q_ in main_q if len([t.get("fact_id") for t in (asks["perworld"][q_].get("trace") or []) if t.get("fact_id")]) != len({t.get("fact_id") for t in (asks["perworld"][q_].get("trace") or []) if t.get("fact_id")}))}}
    Path(a.out).write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({k: out[k] for k in ("label", "n_main", "worlds", "losses", "wins", "classes", "flags")}, indent=1))
    print(json.dumps(out["taken_at_departure_x_outcome"], indent=1))


if __name__ == "__main__":
    main()
