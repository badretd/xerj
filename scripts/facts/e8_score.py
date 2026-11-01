#!/usr/bin/env python3
"""E8 scoring: cohorts from the exact path, supported accuracy per world vs shared, paired bounds, visibility checks.

Cohorts (plan E8: "preserve an original answer only when its question remains identifiable after consolidation"), from the
exact-path file of facts_e8_shared.rs (all questions):
  main        the gold trace answers correctly in the per-world store, and in the shared store every gold step key has
              exactly the per-world value (forward: one active value, the same; inverse: the record's (relation, value) is
              held by exactly one subject)
  changed     the per-world gold trace is correct, but the shared store answers a gold step differently (an edit of
              another world, a second subject with the same value, a background fact): the question is not identifiable
              there; scored separately. Also: a gold step that is an inverse LOOKUP with no record in the world (key
              ['', relation], gold UNKNOWN: "Whose chairperson is X?") while the shared store holds an active fact with that
              relation and one of the question's linked values at the question's revision (another world or the background
              answers it)
  pw_fail     the per-world gold trace itself fails with the default item id (excluded, counted)

Model accuracy (asks_perworld / asks_shared, default item id, int8 + guard):
  correct     status = gold status and, for ANSWER, the frozen dev_v5 normalizer's key of the answer is a gold answer key
  supported   correct, and for ANSWER at least one cited record and every cited record visible at the requested revision in
              the asked namespace (the exact executor derives the answer from the cited records)
  own-world supported   supported, and every cited record a fact of the question's own world (canonical ids in the shared
              store): a correct answer through an equivalent fact of another world (USA head of state vs head of
              government) fails it; reported, not the gate metric
Paired difference shared - per world on `supported`: Newcombe's score interval for paired proportions (method 10), lower
limit of the two-sided 90 % interval = one-sided 95 % lower bound; plus a cluster-robust normal bound (clusters = worlds).
Gate (pre-declared, plan section 10): lower bound >= -0.01 -> PASS; else if the point difference >= -0.01 -> INCONCLUSIVE
(underpowered); else FAIL. The verdict uses the smaller of the two bounds.

Visibility (strict, every ask): V1 snapshot (an evidence record not visible at the requested revision), V2 namespace (an
evidence record that is not in the asked namespace), V3 foreign world (shared arm: an evidence record outside the question's
world content), V3 is reported per cohort; the gate counts V1 + V2 over all asks and V3 over the main cohort.

usage: e8_score.py --corpus DIR --exact FILE --asks DIR --out FILE.json [--label 10k] [--variant cf --asks-base DIR]
                   [--scorer-dir research/xerj_r_mvp/dev_v5/future_scientific]
"""
import argparse, collections, json, math, os, random, sys
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


def norm_set(xs):
    return {scorer.normalize_answer(x) for x in xs if x is not None}


def correct(status, answer, q):
    if status != q["gold_status"]:
        return False
    if status != "ANSWER":
        return True
    k = scorer.normalize_answer(answer)
    return bool(k) and k in norm_set(q["gold_answers"])


def wilson(x, n, z):
    if n == 0:
        return 0.0, 1.0
    p = x / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def newcombe_paired(a, b, c, d, z=1.6448536269514722):
    """difference p1 - p2 with p1 = (a + b)/n (candidate), p2 = (a + c)/n (baseline); b = candidate-only correct,
    c = baseline-only correct; returns (diff, lower, upper) of the two-sided interval at z"""
    n = a + b + c + d
    if n == 0:
        return None, None, None
    p1, p2 = (a + b) / n, (a + c) / n
    l1, u1 = wilson(a + b, n, z)
    l2, u2 = wilson(a + c, n, z)
    den = (a + b) * (c + d) * (a + c) * (b + d)
    phi = (a * d - b * c) / math.sqrt(den) if den > 0 else 0.0
    dl = math.sqrt(max(0.0, (p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2))
    du = math.sqrt(max(0.0, (u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2))
    diff = p1 - p2
    return diff, diff - dl, diff + du


def cluster_bound(diffs, clusters, z=1.6448536269514722):
    n = len(diffs)
    if n == 0:
        return None
    m = sum(diffs) / n
    by = collections.defaultdict(float)
    for d, c in zip(diffs, clusters):
        by[c] += d - m
    g = len(by)
    if g < 2:
        return None
    se = math.sqrt(sum(v * v for v in by.values()) * g / (g - 1)) / n
    return m - z * se


def verdict(lb, diff, margin=-0.01):
    if lb is None:
        return "INCONCLUSIVE"
    if lb >= margin:
        return "PASS"
    return "INCONCLUSIVE" if diff >= margin else "FAIL"


def pct(v, q):
    v = sorted(x for x in v if x is not None)
    if not v:
        return None
    return v[min(len(v) - 1, int(round(q * (len(v) - 1))))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--exact", required=True)
    ap.add_argument("--asks", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--variant", default="", help="'' = default (exact linker) arms; 'cf' = the casefold arms")
    ap.add_argument("--asks-base", default=None, help="asks of the exact linker: rows marked same_as_exact are taken from here")
    ap.add_argument("--scorer-dir", default=None, help="directory of the frozen scorer.py (default $XERJ_R_SCORER_DIR)")
    a = ap.parse_args()
    load_scorer(a.scorer_dir)
    corpus = Path(a.corpus)
    qs = {q["qid"]: q for q in jl(corpus / "questions.jsonl")}
    world_facts = {w["world_id"]: set(w["facts"]) for w in jl(corpus / "world_facts.jsonl")}
    ex = {}
    for r in jl(a.exact):
        ex[r["qid"]] = r
    sfx = f"_{a.variant}" if a.variant else ""
    # inverse steps with no record in the world (key ['', relation]): (relation, linked value) pairs to look up in the corpus
    absent_inv, need = {}, set()
    for qid, r in ex.items():
        for st in qs[qid]["trace"]:
            k = st.get("key")
            if isinstance(k, list) and len(k) == 2 and k[0] == "" and st.get("direction") == "in":
                # the values the default linker links in the world's own store: the cohort does not depend on the linker
                vals = r["perworld"].get("values") or r["shared"].get("values") or []
                absent_inv[qid] = (k[1], vals)
                need.update((k[1], v) for v in vals)
    held = collections.defaultdict(list)  # (relation, value) -> fact ids of the shared corpus

    def note(f):
        v = f.get("value")
        if isinstance(v, str) and (f.get("relation"), v) in need:
            held[(f["relation"], v)].append(f["fact_id"])
    # shared-store visibility from the corpus: phase-1 facts (visible at B), phase-2 facts, superseded and withdrawn
    p1, p2, sup, wd_docs, doc_of = set(), set(), set(), set(), {}
    for b in jl(corpus / "phase1.jsonl"):
        for f in b.get("facts", []):
            p1.add(f["fact_id"]); doc_of[f["fact_id"]] = (f["doc_id"], f["version"]); note(f)
    for b in jl(corpus / "phase2.jsonl"):
        for e in b.get("edits", []):
            for f in ([e["fact"]] if e.get("fact") else []) + list(e.get("facts") or []):
                p2.add(f["fact_id"]); doc_of[f["fact_id"]] = (f["doc_id"], f["version"]); note(f)
            if e.get("op") == "correct":
                sup.add(e["supersedes"])
            if e.get("op") == "withdraw_document":
                wd_docs.add((e["doc_id"], e["version"]))
    vis_b = p1
    vis_f = {f for f in (p1 | p2) if f not in sup and doc_of.get(f) not in wd_docs}
    # per-world visibility from the original worlds of the sample
    pw_vis = {}
    for w in jl(corpus / "worlds_sample.jsonl"):
        ww = w["world"]
        base = {f["fact_id"] for f in ww.get("base_facts", [])}
        docs = {}
        for f in ww.get("base_facts", []):
            docs[f["fact_id"]] = (f["doc_id"], f["version"])
        allf, s2, wd = set(base), set(), set()
        for e in ww.get("edits", []):
            for f in ([e["fact"]] if e.get("fact") else []) + list(e.get("facts") or []):
                allf.add(f["fact_id"]); docs[f["fact_id"]] = (f["doc_id"], f["version"])
            if e.get("op") == "correct":
                s2.add(e["supersedes"])
            if e.get("op") == "withdraw_document":
                wd.add((e["doc_id"], e["version"]))
        fin = {f for f in allf if f not in s2 and docs.get(f) not in wd}
        pw_vis[(w["world_file"], w["world_id"])] = (base, fin, allf)

    # ---------------------------------------------------------------------------------------------- exact path
    cohort = {}
    ec = collections.Counter()
    by_set = collections.defaultdict(collections.Counter)
    lm_us = {"perworld": [], "shared": []}
    for qid, r in ex.items():
        q = qs[qid]
        pw, sh = r[f"perworld{sfx}"], r[f"shared{sfx}"]
        lm_us["perworld"].append(pw["us_link_manifest"]); lm_us["shared"].append(sh["us_link_manifest"])
        pw_ok = correct(pw["oracle"]["status"], pw["oracle"]["answer"], q)
        sh_ok = correct(sh["oracle"]["status"], sh["oracle"]["answer"], q)
        # identifiability: every keyed step has the per-world view in the shared store
        ident = True
        why = []
        for s1, s2 in zip(pw["steps"], sh["steps"]):
            if "key" not in s1:
                continue
            if s2.get("status") != s1.get("status") or sorted(s2.get("values", [])) != sorted(s1.get("values", [])):
                ident = False; why.append("value" if s2.get("status") == s1.get("status") else f"status {s1.get('status')}->{s2.get('status')}")
            if s1.get("inverse") and s2.get("inverse_subjects", 0) != s1.get("inverse_subjects", 0):
                ident = False; why.append("inverse subjects")
        if qid in absent_inv:
            rel, vals = absent_inv[qid]
            seen = vis_b if q["phase"] == "before" else vis_f
            if any(fid in seen for v in vals for fid in held.get((rel, v), ())):
                ident = False; why.append("absent inverse key held in the shared store")
        if not pw_ok:
            c = "pw_fail"
        elif ident:
            c = "main"
        else:
            c = "changed"
        cohort[qid] = c
        st = by_set[q["set"]]
        st["questions"] += 1
        st[f"cohort {c}"] += 1
        st["perworld oracle correct"] += pw_ok
        st["shared oracle correct"] += sh_ok
        if c == "main":
            st["main: shared oracle correct"] += sh_ok
            st["main: base-manifest gold lookups found (shared)"] += sh["gold_lookups_in_cands"]
            st["main: base-manifest gold lookups found (perworld)"] += pw["gold_lookups_in_cands"]
            st["main: base-manifest gold lookups"] += sh["gold_lookups"]
            if not sh_ok:
                st["main: shared oracle wrong, problems: " + ("; ".join(sh["problems"])[:80] or "none")] += 1
        if c == "changed":
            for w in set(why):
                st[f"changed because: {w}"] += 1
            st["changed: shared oracle status " + str(sh["oracle"]["status"])] += 1
        for k in ("link", "values"):
            if len(sh[k]) > len(pw[k]):
                st[f"more {k} linked in shared"] += 1
    tot = collections.Counter()
    for s in by_set.values():
        tot.update(s)
    exact = {"by_set": {k: dict(v) for k, v in by_set.items()}, "total": dict(tot),
             "link_manifest_us": {k: {"p50": pct(v, 0.5), "p95": pct(v, 0.95), "p99": pct(v, 0.99)} for k, v in lm_us.items()}}

    # ---------------------------------------------------------------------------------------------- model asks
    model = None
    if a.asks:
        asks = {}
        merged = collections.Counter()
        for arm in ("perworld", "shared"):
            p = Path(a.asks) / f"asks_{arm}.jsonl"
            if p.exists():
                asks[arm] = {r["qid"]: r for r in jl(p)}
                if a.asks_base:
                    base = {r["qid"]: r for r in jl(Path(a.asks_base) / f"asks_{arm}.jsonl")}
                    for qid, r in list(asks[arm].items()):
                        if r.get("same_as_exact"):
                            if qid in base:
                                asks[arm][qid] = base[qid]
                                merged[f"{arm}: unchanged by the pre-link, exact-linker row used"] += 1
                            else:
                                del asks[arm][qid]
                                merged[f"{arm}: unchanged, no exact-linker row (dropped)"] += 1
                        else:
                            merged[f"{arm}: asked with the pre-link"] += 1
        shared_load = None
        model = {"arms": {}, "paired": {}, "visibility": {}, "latency": {}, "merged": dict(merged)}
        vis = collections.Counter()
        v3_by = collections.Counter()
        rows = []
        for qid in sorted(set(asks.get("perworld", {})) & set(asks.get("shared", {}))):
            q = qs[qid]
            rp, rs = asks["perworld"][qid], asks["shared"][qid]
            if "http" in rp or "http" in rs or "skipped" in rp:
                vis["asks with an HTTP error or skipped"] += 1
                continue
            wkey = (q["world_file"], q["world_id"])
            base, fin, allf = pw_vis[wkey]
            wf = world_facts.get(q["world_id"], set())
            res = {}
            for arm, r in (("perworld", rp), ("shared", rs)):
                ev = r.get("evidence_ids") or []
                before = q["phase"] == "before"
                if arm == "perworld":
                    visible, owned = (base if before else fin), allf
                    clean = all(e in visible for e in ev)
                else:
                    visible = vis_b if before else vis_f
                    clean = all(e in visible for e in ev)
                    owned = wf
                for kind in ("raw", "verified"):
                    stt = r["status"] if kind == "raw" else r["final_status"]
                    ans = r["answer"] if kind == "raw" else r["final_answer"]
                    ok = correct(stt, ans, q)
                    # supported: an ANSWER cites at least one record and every cited record is visible at the requested
                    # revision in the asked namespace (the exact executor derives the answer from the cited records)
                    sup = ok and (stt != "ANSWER" or (bool(ev) and clean))
                    own = sup and (stt != "ANSWER" or all(e in owned for e in ev))
                    res[(arm, kind)] = (ok, sup, own)
                # strict visibility
                if arm == "perworld":
                    visible = base if before else fin
                    for e in ev:
                        if e not in allf:
                            vis["V2 namespace (per-world arm: evidence outside its world/namespace)"] += 1
                        elif e not in visible:
                            vis["V1 snapshot (per-world arm)"] += 1
                else:
                    visible = vis_b if before else vis_f
                    for e in ev:
                        if e not in p1 and e not in p2:
                            vis["V2 namespace (shared arm: evidence outside the shared corpus)"] += 1
                        elif e not in visible:
                            vis["V1 snapshot (shared arm)"] += 1
                    foreign = [e for e in ev if e not in wf]
                    if foreign:
                        v3_by[(cohort.get(qid, "?"), "correct" if res[("shared", "raw")][0] else "wrong")] += 1
            rows.append((qid, q, res))
        model["visibility"] = {"strict": dict(vis), "V1+V2 total": sum(v for k, v in vis.items() if k.startswith(("V1", "V2"))),
                               "V3 foreign-world evidence (shared arm), by cohort / correctness": {f"{k[0]} {k[1]}": v for k, v in v3_by.items()},
                               "V3 in the main cohort": sum(v for k, v in v3_by.items() if k[0] == "main")}
        # accuracy tables
        def table(sel, name):
            out = {}
            for kind in ("raw", "verified"):
                for metric, idx in (("correct", 0), ("supported", 1), ("own-world supported", 2)):
                    a_ = b_ = c_ = d_ = 0
                    diffs, clus = [], []
                    for qid, q, res in sel:
                        s_ok = res[("shared", kind)][idx]
                        p_ok = res[("perworld", kind)][idx]
                        if s_ok and p_ok: a_ += 1
                        elif s_ok and not p_ok: b_ += 1
                        elif p_ok and not s_ok: c_ += 1
                        else: d_ += 1
                        diffs.append(int(s_ok) - int(p_ok)); clus.append(f"{q['world_file']}|{q['world_id']}")
                    n = a_ + b_ + c_ + d_
                    diff, lo, hi = newcombe_paired(a_, b_, c_, d_)
                    cb = cluster_bound(diffs, clus)
                    lb = min(x for x in (lo, cb) if x is not None) if (lo is not None) else None
                    out[f"{kind} {metric}"] = {"n": n, "perworld": (a_ + c_) / n if n else None, "shared": (a_ + b_) / n if n else None,
                                                "shared_only": b_, "perworld_only": c_, "diff": diff,
                                                "lb95_newcombe": lo, "lb95_cluster": cb, "verdict": verdict(lb, diff if diff is not None else 0)}
            return out
        by_c = collections.defaultdict(list)
        for row in rows:
            by_c[cohort.get(row[0], "no exact-path row")].append(row)
            by_c[f"{cohort.get(row[0], '?')} / {row[1]['set']}"].append(row)
        model["paired"] = {k: table(v, k) for k, v in sorted(by_c.items())}
        model["paired"]["all questions"] = table(rows, "all")
        # latency by arm
        for arm in ("perworld", "shared"):
            rr = [r for r in asks.get(arm, {}).values() if "timings_us" in r and r.get("timings_us")]
            lat = {"n": len(rr), "client_ms": {"p50": pct([r["client_ms"] for r in rr], .5), "p95": pct([r["client_ms"] for r in rr], .95)}}
            for k in ("link", "manifest", "model", "executor", "verify", "total"):
                v = [r["timings_us"].get(k) for r in rr]
                lat[f"{k}_ms"] = {"p50": None if pct(v, .5) is None else round(pct(v, .5) / 1e3, 3),
                                  "p95": None if pct(v, .95) is None else round(pct(v, .95) / 1e3, 3),
                                  "p99": None if pct(v, .99) is None else round(pct(v, .99) / 1e3, 3)}
            lat["guard_fallbacks"] = sum((r.get("fallbacks") or 0) for r in rr)
            lat["decisions"] = sum((r.get("decisions") or 0) for r in rr)
            model["latency"][arm] = lat
    out = {"label": a.label, "exact_path": exact, "model": model}
    Path(a.out).write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    # short print
    print(json.dumps({"exact_total": exact["total"], "link_manifest_us": exact["link_manifest_us"]}, indent=1))
    if model:
        for k in ("main", "changed", "all questions"):
            if k in model["paired"]:
                print(k, json.dumps(model["paired"][k].get("raw supported"), indent=None))
        print(json.dumps(model["visibility"], indent=1))


if __name__ == "__main__":
    main()
