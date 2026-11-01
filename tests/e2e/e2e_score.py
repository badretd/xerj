#!/usr/bin/env python3
"""Test 6 scoring, in the research tree (labels stay there). Takes the HTTP rows of e2e_bench.py and compares them with the research rows.

  python3 -I -B e2e_score.py --mvp research/xerj_r_mvp --http DIR_WITH_ROWS --out DIR

SELECT half only: no check-half label, row or score is read or written. Per precision: HTTP rows against the research rows of that
precision (per question: status, answer, program, evidence); served decisions against the frozen Python policy on the same rows; served
numbers from the fields the client actually gets; FP32 to w8 changes. Gate: every HTTP difference from the research row of the same
precision is explained, and the served numbers equal the frozen policy on the research rows.
"""
import argparse, collections, json, sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--mvp", required=True); ap.add_argument("--http", required=True); ap.add_argument("--out", required=True); ap.add_argument("--seed-dir", default="s29")
a = ap.parse_args()
MVP = Path(a.mvp).resolve(); HTTP = Path(a.http).resolve(); OUT = Path(a.out).resolve(); OUT.mkdir(parents=True, exist_ok=True)
sys.dont_write_bytecode = True
sys.path[:0] = [str(MVP / "decline_v1"), str(MVP / "tools")]
import evaluate as E                     # noqa: E402
import policy as P                       # noqa: E402
import replay as RP                      # noqa: E402
import hybrid_bench_score as HS          # noqa: E402
from common import load_labels, load_split  # noqa: E402

v2 = E.verify_frozen_v2(); cfg = dict(v2["policy"]); split = load_split()["items"]; order, labels = load_labels()
order = [i for i in order if split[i] == "select"]
labels = {i: labels[i] for i in order}
sc = HS.load_scorer(MVP / "dev_v5" / "future_scientific")
ROOT = MVP / "runs" / "hybrid_v2_eval" / "rented_pairs"
REF = {"f32": ROOT / "cf32" / "x68_cf32_C__external_executor" / a.seed_dir, "unset": ROOT / "cf32_int8" / "x68_cf32_C_int8__external_executor" / a.seed_dir}
ref = {k: {i: r for i, r in E.read_rows(v / "hybrid_v2_bench.rows.jsonl").items() if i in labels} for k, v in REF.items()}
worlds = RP.Worlds()
halves = {"select": order}


def key(r):
    return {"status": r.get("status"), "answer": r.get("answer"), "program": r.get("program"), "evidence_ids": r.get("evidence_ids")}


def prog(p):
    return [[s.get("op"), s.get("pointer"), s.get("direction")] for s in p or []]


def diff_fields(x, y):
    d = [k for k in ("status", "answer", "evidence_ids") if x.get(k) != y.get(k)]
    return d + (["program"] if prog(x.get("program")) != prog(y.get("program")) else [])


def wilson(k, n):
    u = P.hp.wilson_up(k, n)
    return None if u is None else round(u, 4)


def served(rows):
    """numbers from the served fields; `right` is the research rule"""
    res = {}
    for h, ids in halves.items():
        acc = wrong = cov = 0; den = 0; inscope_ok = 0; n_in = 0; reasons = collections.Counter(); oos_wrong = 0
        for i in ids:
            r, lab = rows[i], labels[i]
            o = {"accepted": r["final_status"] == "ANSWER", "answer": r["final_answer"]}
            gr = dict(r, gold_status=lab["gold_status"], gold_answer=lab.get("gold_answer"))
            source = "computable" if lab["scope"] == "IN_SCOPE" and lab["gold_status"] == "ANSWER" else "other"
            den += source == "computable"
            if o["accepted"]:
                acc += 1
                if P.right(gr, o):
                    cov += source == "computable"
                else:
                    wrong += 1
                    oos_wrong += lab["scope"] != "IN_SCOPE"
            else:
                reasons[r["reason_code"]] += 1
            if lab["scope"] == "IN_SCOPE":
                n_in += 1
                rec = dict(r, status=r["final_status"] if r["final_status"] != "UNSUPPORTED" else "UNKNOWN", answer=r["final_answer"])
                inscope_ok += HS.categorize(sc, rec, lab)["cat"] == "correct"
        res[h] = {"questions": len(ids), "accepted": acc, "wrong": wrong, "wrong_out_of_scope": oos_wrong, "wrong_among_accepted": round(wrong / acc, 4) if acc else None, "upper95_one_sided": wilson(wrong, acc),
                  "coverage_right": cov, "coverage_denominator": den, "coverage": round(cov / den, 4), "in_scope_n": n_in, "in_scope_correct_served": inscope_ok, "in_scope_correct_served_rate": round(inscope_ok / n_in, 4),
                  "reasons": dict(reasons.most_common())}
    return res


def python_numbers(rows):
    """the frozen policy on research rows, select half, gold from the private labels"""
    acc = wrong = cov = 0
    for i in order:
        o = P.stages(rows[i], worlds, None, cfg)["P3+D2+D2b"]; lab = labels[i]
        if o["accepted"]:
            acc += 1
            gr = dict(rows[i], gold_status=lab["gold_status"], gold_answer=lab.get("gold_answer"))
            if P.right(gr, o):
                cov += lab["scope"] == "IN_SCOPE" and lab["gold_status"] == "ANSWER"
            else:
                wrong += 1
    return {"accepted": acc, "wrong": wrong, "coverage_right": cov}


res = {"schema": "xerj-e2e-bench-v1", "labelling": "exposed development material; select half only, the half the rules were designed on; no fresh-question claim",
       "seed": a.seed_dir, "questions": len(order), "precisions": {}}
http = {}
for pr in ("unset", "f32"):
    f = HTTP / f"{pr}.rows.jsonl"
    if not f.exists():
        continue
    rows = {}
    for l in open(f, encoding="utf-8"):
        r = json.loads(l); rows[r["item_id"]] = r
    missing = [i for i in order if i not in rows]
    p = {"rows": len(rows), "missing": len(missing)}
    res["precisions"][pr] = p
    if missing:
        continue
    http[pr] = rows
    # 1. same as the research rows of this precision
    c = collections.Counter(); sel_diff = []
    for i in order:
        d = diff_fields(rows[i], ref[pr][i])
        c["identical" if not d else "different"] += 1
        for k in d:
            c["field_" + k] += 1
        if d and split[i] == "select":
            sel_diff.append({"item_id": i, "fields": d, "http": [rows[i]["status"], rows[i]["answer"]], "ref": [ref[pr][i]["status"], ref[pr][i]["answer"]]})
    p["vs_research_rows"] = dict(c); p["select_half_differences"] = sel_diff[:40]
    # 3. served decision = the Python policy on the HTTP rows
    c = collections.Counter(); py = {}
    for i in order:
        r = rows[i]; o = py[i] = P.stages({k: r.get(k) for k in ("item_id", "phase", "question", "status", "answer", "program", "execution_trace", "evidence_ids", "error")}, worlds, None, cfg)["P3+D2+D2b"]
        same = o["accepted"] == (r["final_status"] == "ANSWER") and (not o["accepted"] or o["answer"] == r["final_answer"]) and o["repaired"] == bool(r.get("repair"))
        c["same" if same else "different"] += 1
        if o["accepted"] and not r["evidence"]:
            c["accepted_without_citation"] += 1
        if not o["accepted"] and not r["reason_code"]:
            c["declined_without_reason"] += 1
        if not o["accepted"] and (r["evidence"] or r["final_answer"] is not None):
            c["declined_with_value_or_citation"] += 1
        if o["repaired"]:
            c["repairs"] += 1
            c["repairs_logged"] += bool((r.get("repair") or {}).get("original_program")) and r["evidence"] is not None
    p["served_vs_python_policy"] = dict(c)
    p["served"] = served(rows)
    p["python_policy_on_research_rows"] = python_numbers(ref[pr])


def cat(r, i):
    return HS.categorize(sc, r, labels[i])["cat"]


def changes(x, y):
    """questions, outcome changes, status-or-answer changes between two row sets, by half"""
    out = {}
    for h, ids in halves.items():
        o = sum(cat(x[i], i) != cat(y[i], i) for i in ids)
        s = sum((x[i].get("status"), x[i].get("answer")) != (y[i].get("status"), y[i].get("answer")) for i in ids)
        pr_ = sum(prog(x[i].get("program")) != prog(y[i].get("program")) for i in ids)
        out[h] = {"outcome_changed": o, "status_or_answer_changed": s, "program_changed": pr_}
    return out


res["research_f32_to_w8"] = changes(ref["f32"], ref["unset"])
gate = {}
if "unset" in http:
    w = http["unset"]
    res["http_w8_vs_research_f32"] = changes(w, ref["f32"])
    if "f32" in http:
        res["http_f32_to_http_w8"] = changes(http["f32"], w)
    unexplained = [i for i in order if diff_fields(w[i], ref["unset"][i])]
    gate["unexplained_question_differences"] = len(unexplained)
    gate["examples"] = unexplained[:20]
    rp_, sv = res["precisions"]["unset"]["python_policy_on_research_rows"], res["precisions"]["unset"]["served"]["select"]
    gate["metrics_equal_research_w8"] = all(rp_[k] == sv[k] for k in ("accepted", "wrong", "coverage_right"))
    gate["served_equals_python"] = res["precisions"]["unset"]["served_vs_python_policy"].get("different", 0) == 0
    gate["pass"] = not unexplained and gate["metrics_equal_research_w8"] and gate["served_equals_python"]
res["gate"] = gate
(OUT / "bench_compare.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
print(json.dumps({k: v for k, v in res.items() if k in ("gate", "research_f32_to_w8")}, indent=1))
for pr, p in res["precisions"].items():
    print(pr, p.get("vs_research_rows"), p.get("served_vs_python_policy"), {h: {k: v[k] for k in ("accepted", "wrong", "upper95_one_sided", "coverage")} for h, v in (p.get("served") or {}).items()})
