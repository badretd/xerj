#!/usr/bin/env python3
"""E8 shared-store corpus (design gate E8; plan research/xerj_r_mvp/SHARED_STORE_PLAN.md).

Consolidates the test worlds of several sets into ONE globally consistent fact store and adds real Wikidata5M background
facts, so that the same questions can be asked per world (the reference: one store per world) and in one shared store.

Consolidation (plan E8: "construct a globally consistent entity space"):
- A base fact whose content (subject, relation, kind, value, unit, scope, valid_from, valid_to) was already written by an
  earlier world is not written again: every world that holds it shares the first copy (its fact id). MQuAKE-T holds 34,081
  base facts over 1,920 keys; without this every key would be 18 records.
- Edits are applied once, globally, after every base fact (phase 2). An edit that supersedes a fact some earlier world's edit
  already superseded with the same new content is the same edit (written once); with a different new content it collides
  and is not written (counted). Worlds that hold the old fact without that edit see the new value in the shared store: the
  exact-path oracle check puts their questions in the "changed by consolidation" cohort instead of the main cohort.
- Fact / document ids already used with different content are renamed with the world id (counted; none expected).
- Background (Wikidata5M transductive, 2019): subjects in a fixed hash order, all their triples, labels = first alias,
  relation phrases = the test worlds' phrase for the same P-id, else Wikidata5M's first alias. A background triple whose
  (subject, relation) key is a test key is skipped (it would contradict a test fact from a different dump date); other
  overlaps stay: same names, same relations and same values are the realistic distractors the plan asks for.
- Revisions: phase 1 = test bases, then background, in batches; phase 2 = the edits. Questions in phase "before" read the
  revision after phase 1 (B), "after" questions the revision after phase 2 (F).

Outputs in --out: phase1.jsonl (one _bulk body per line), phase2.jsonl, questions.jsonl, worlds_sample.jsonl (the original
worlds of the model sample, for the per-world namespaces), corpus.json (counts, sha256s, consolidation log).

usage: e8_corpus.py --root $XERJ_R_ROOT/runs/world --wd5m DIR --out DIR --background N [--batch-facts 20000]
"""
import argparse, collections, hashlib, json, os, sys
from pathlib import Path

# set -> (world file, item file replacing the worlds' items or None, label file); the order is the load order
SETS = [
    ("mquake_t", "mquake_lifecycle/model_test_t.jsonl", None, "mquake_lifecycle/private_labels/test_t.jsonl"),
    ("mquake_t_chain", "mquake_lifecycle/model_test_t.jsonl", "mquake_lifecycle/chain_stack_v2/items_test_t.jsonl",
     "mquake_lifecycle/chain_stack_v2/private_labels/test_t.jsonl"),
    ("direction_test", "direction_v1/model_test.jsonl", None, "direction_v1/private_labels/test.jsonl"),
    ("pick_num_test", "pick_num_v1/model_test.jsonl", None, "pick_num_v1/private_labels/test.jsonl"),
    ("twowiki_test", "twowiki_oracle/model_test.jsonl", None, "twowiki_oracle/private_labels/test.jsonl"),
    ("hotpot_test", "hotpot_oracle/model_test.jsonl", None, "hotpot_oracle/private_labels/test.jsonl"),
]
# model sample per set: (first N worlds of the world file, max items) -- the per-world namespaces hold exactly these worlds
SAMPLE = {"mquake_t": (100, 10**9), "mquake_t_chain": (50, 10**9), "direction_test": (1, 10**9),
          "pick_num_test": (100, 10**9), "twowiki_test": (200, 10**9), "hotpot_test": (54, 10**9)}


def jl(p):
    with open(p, encoding="utf-8") as f:
        for l in f:
            if l.strip():
                yield json.loads(l)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def content_key(f):
    v = f.get("value")
    v = ("b", v) if isinstance(v, bool) else ("s", "" if v is None else str(v))
    return (f["subject"], f["relation"], f.get("kind", "text"), v, f.get("unit", "") or "", f.get("scope", "default") or "default",
            f.get("valid_from"), f.get("valid_to"))


class Consolidator:
    def __init__(self):
        self.canon_by_content = {}     # content key -> canonical fact id
        self.fact_content = {}         # canonical fact id -> content key
        self.canon_of = {}             # (world id, original fact id) -> canonical fact id (None = collided)
        self.docs = {}                 # (doc_id, version) -> text (written)
        self.doc_map = {}              # (world id, doc_id, version) -> (doc_id, version) written
        self.superseded_by = {}        # canonical fact id -> canonical id of its replacement
        self.phase1_docs, self.phase1_facts, self.phase2 = [], [], []
        self.log = collections.Counter()
        self.world_facts = collections.defaultdict(set)  # world id -> canonical fact ids the world's facts map to
        self.test_keys = set()         # (subject, relation) of every test fact (base and edits)

    def _doc(self, wid, d):
        key = (d["doc_id"], d["version"])
        m = (wid, d["doc_id"], d["version"])
        if m in self.doc_map:
            return self.doc_map[m], False
        if key in self.docs:
            if self.docs[key] == d["text"]:
                self.doc_map[m] = key
                self.log["documents shared (same id, same text)"] += 1
                return key, False
            key = (f"e8r/{wid}/{d['doc_id']}", d["version"])
            self.log["documents renamed (id reused with other text)"] += 1
        self.docs[key] = d["text"]
        self.doc_map[m] = key
        return key, True

    def _fact_id(self, wid, f):
        fid = f["fact_id"]
        if fid in self.fact_content:
            fid = f"e8r/{wid}/{fid}"
            self.log["facts renamed (id reused with other content)"] += 1
        return fid

    def base_world(self, w):
        wid = w["world_id"]
        docs = {(d["doc_id"], d["version"]): d for d in w.get("base_documents", [])}
        for f in w.get("base_facts", []):
            self.test_keys.add((f["subject"], f["relation"]))
            ck = content_key(f)
            if ck in self.canon_by_content:
                c = self.canon_by_content[ck]
                self.canon_of[(wid, f["fact_id"])] = c
                self.world_facts[wid].add(c)
                self.log["base facts shared with an earlier world"] += 1
                continue
            d = docs.get((f["doc_id"], f["version"]))
            if d is None:
                raise SystemExit(f"{wid}: fact {f['fact_id']} cites a document outside base_documents")
            (doc_id, ver), new = self._doc(wid, d)
            if new:
                self.phase1_docs.append({"doc_id": doc_id, "version": ver, "text": d["text"]})
            fid = self._fact_id(wid, f)
            g = dict(f, fact_id=fid, doc_id=doc_id, version=ver)
            self.phase1_facts.append(g)
            self.canon_by_content[ck] = fid
            self.fact_content[fid] = ck
            self.canon_of[(wid, f["fact_id"])] = fid
            self.world_facts[wid].add(fid)
            self.log["base facts written"] += 1
        # base documents that no fact of the world cites are written too (text without facts); a document whose facts
        # were all shared with an earlier world is not (its facts already have a source)
        cited = {(f["doc_id"], f["version"]) for f in w.get("base_facts", [])}
        for (did, ver), d in docs.items():
            if (did, ver) not in cited and (wid, did, ver) not in self.doc_map:
                (doc_id, v2), new = self._doc(wid, d)
                if new:
                    self.phase1_docs.append({"doc_id": doc_id, "version": v2, "text": d["text"]})

    def _edit_fact(self, wid, f, supersedes):
        """the fact of an edit: returns the rewritten fact or None when it is the same as an earlier world's"""
        self.test_keys.add((f["subject"], f["relation"]))
        ck = content_key(f)
        fid = self._fact_id(wid, f)
        self.fact_content[fid] = ck
        self.canon_of[(wid, f["fact_id"])] = fid
        self.world_facts[wid].add(fid)
        return dict(f, fact_id=fid)

    def edits_world(self, w):
        wid = w["world_id"]
        for e in w.get("edits", []):
            op = e.get("op")
            if op == "correct":
                old = self.canon_of.get((wid, e["supersedes"]))
                if old is None:
                    self.log["edits dropped: superseded fact unknown or collided"] += 1
                    self.canon_of[(wid, e["fact"]["fact_id"])] = None
                    continue
                ck_new = content_key(e["fact"])
                if old in self.superseded_by:
                    prev = self.superseded_by[old]
                    if self.fact_content.get(prev) == ck_new:
                        self.canon_of[(wid, e["fact"]["fact_id"])] = prev
                        self.world_facts[wid].add(prev)
                        self.log["edits shared with an earlier world (same old fact, same new fact)"] += 1
                    else:
                        self.canon_of[(wid, e["fact"]["fact_id"])] = None
                        self.log["edits collided (old fact already replaced by another value)"] += 1
                    continue
                (doc_id, ver), new = self._doc(wid, e["doc"])
                f = self._edit_fact(wid, dict(e["fact"], doc_id=doc_id, version=ver), [old])
                g = {"op": "correct", "fact": f, "supersedes": old}
                g["doc"] = {"doc_id": doc_id, "version": ver, "text": e["doc"]["text"]} if new else None
                if g["doc"] is None:
                    raise SystemExit(f"{wid}: correct edit reuses document {doc_id}@{ver}")
                self.superseded_by[old] = f["fact_id"]
                self.phase2.append(g)
                self.log["edits written: correct"] += 1
            elif op in ("add_document", "add_conflict"):
                (doc_id, ver), new = self._doc(wid, e["doc"])
                if not new:
                    raise SystemExit(f"{wid}: {op} reuses document {doc_id}@{ver}")
                g = {"op": op, "doc": {"doc_id": doc_id, "version": ver, "text": e["doc"]["text"]}}
                if "facts" in e:
                    g["facts"] = [self._edit_fact(wid, dict(f, doc_id=doc_id, version=ver), []) for f in e["facts"]]
                if "fact" in e:
                    g["fact"] = self._edit_fact(wid, dict(e["fact"], doc_id=doc_id, version=ver), [])
                for k, v in e.items():
                    if k not in g and k not in ("doc", "facts", "fact", "serial"):
                        g[k] = v
                self.phase2.append(g)
                self.log[f"edits written: {op}"] += 1
            elif op == "withdraw_document":
                m = self.doc_map.get((wid, e["doc_id"], e["version"]))
                if m is None:
                    raise SystemExit(f"{wid}: withdraw_document of an unknown document {e['doc_id']}")
                self.phase2.append({"op": op, "doc_id": m[0], "version": m[1]})
                self.log["edits written: withdraw_document"] += 1
            else:
                raise SystemExit(f"{wid}: edit op {op!r} not handled")


def background(c, wd5m, n_target, rel_phrase_override, st, id_prefix="wd5m"):
    """Wikidata5M triples of subjects in sha256(Qid) order until n_target facts: yields (document, [facts]) chunks (one
    document per subject and at most 64 facts), streaming (nothing but the selected triples and labels is held)"""
    wd = Path(wd5m)
    rel = {}
    for l in open(wd / "wikidata5m_relation.txt", encoding="utf-8"):
        p = l.rstrip("\n").split("\t")
        if len(p) >= 2 and p[1].strip():
            rel[p[0]] = p[1].strip()
    rel.update(rel_phrase_override)
    # pass 1: triples per subject
    per = collections.Counter()
    tr = wd / "wikidata5m_transductive_train.txt"
    for l in open(tr, encoding="utf-8"):
        per[l.split("\t", 1)[0]] += 1
    order = sorted(per, key=lambda q: hashlib.sha256(q.encode()).digest())
    rank = {}
    tot = 0
    for i, q in enumerate(order):
        if tot >= n_target * 1.25 + 1000:  # head-room for skipped triples
            break
        rank[q] = i
        tot += per[q]
    del per, order
    # pass 2: the selected subjects' triples, compact ("P\tQ" strings)
    trip = collections.defaultdict(list)
    need = set(rank)
    for l in open(tr, encoding="utf-8"):
        s_, _, po = l.rstrip("\n").partition("\t")
        if s_ in rank:
            trip[s_].append(po)
            need.add(po.partition("\t")[2])
    label = {}
    for l in open(wd / "wikidata5m_entity.txt", encoding="utf-8"):
        q, _, rest = l.partition("\t")
        if q in need:
            a_ = rest.rstrip("\n").split("\t", 1)[0].strip()
            if a_ and " ".join(a_.split()) == a_:
                label[q] = a_
    del need
    n = 0
    for s_ in sorted(trip, key=lambda q: rank[q]):
        if n >= n_target:
            break
        S = label.get(s_)
        if S is None:
            st["subjects without a usable label"] += 1
            continue
        facts = []
        for po in sorted(set(trip[s_])):
            if n + len(facts) >= n_target:
                break
            p, _, o = po.partition("\t")
            V, R = label.get(o), rel.get(p)
            if V is None or R is None:
                st["triples without a usable object / relation label"] += 1
                continue
            if (S, R) in c.test_keys:
                st["triples on a test key (skipped)"] += 1
                continue
            facts.append((p, o, R, V))
        # one document per subject, at most 64 facts each: "S R V. S R2 V2. ..."
        for ci in range(0, len(facts), 64):
            chunk = facts[ci:ci + 64]
            did = f"{id_prefix}/{s_}/{ci // 64}"
            parts, spans, pos = [], [], 0
            for (p, o, R, V) in chunk:
                sent = f"{S} {R} {V}."
                a2 = pos + len(S) + 1 + len(R) + 1
                spans.append((a2, a2 + len(V)))
                parts.append(sent)
                pos += len(sent) + 1
            text = " ".join(parts)
            out = []
            for (p, o, R, V), (a2, b2) in zip(chunk, spans):
                assert text[a2:b2] == V, (text, a2, b2, V)
                out.append({"fact_id": f"{id_prefix}/{s_}/{p}/{o}", "subject": S, "relation": R, "kind": "text", "value": V,
                            "unit": "", "doc_id": did, "version": "v1", "start": a2, "end": b2, "quote": V,
                            "scope": "default"})
            n += len(out)
            st["background documents"] += 1
            yield {"doc_id": did, "version": "v1", "text": text}, out
        trip[s_] = None
    st["background facts"] = n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--wd5m", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--background", type=int, default=0, help="background facts to add (Wikidata5M)")
    ap.add_argument("--target-total", type=int, default=0, help="instead of --background: fill to this many facts in total")
    ap.add_argument("--batch-facts", type=int, default=20000)
    ap.add_argument("--bg-id-prefix", default="wd5m",
                    help="prefix of the background fact / document ids; manifests truncate tier 1 in fact-id order, so "
                         "'0wd5m' (sorts before every test id) is the adverse case")
    a = ap.parse_args()
    root, out = Path(a.root), Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    c = Consolidator()
    worlds_by_file = {}
    for name, wf, _, _ in SETS:
        if wf not in worlds_by_file:
            worlds_by_file[wf] = list(jl(root / wf))
    seen_files = []
    for name, wf, _, _ in SETS:
        if wf in seen_files:
            continue
        seen_files.append(wf)
        for w in worlds_by_file[wf]:
            c.base_world(w)
    n_test_base = len(c.phase1_facts)
    for wf in seen_files:
        for w in worlds_by_file[wf]:
            c.edits_world(w)
    # relation phrases of the test worlds by P-id (Wikidata relations)
    over = {}
    for wf in seen_files:
        for w in worlds_by_file[wf]:
            for f in w.get("base_facts", []):
                if f.get("relation_id", "").startswith("P"):
                    over.setdefault(f["relation_id"], f["relation"])
    bg_stats = collections.Counter()
    n_bg = a.background
    if a.target_total:
        n_bg = max(0, a.target_total - n_test_base - sum(len(e.get("facts", [])) + (1 if e.get("fact") else 0) for e in c.phase2))
    # phase 1 batches: test docs + facts first, then background (documents always in the batch of their first fact)
    def batches(docs, facts):
        dmap = {(d["doc_id"], d["version"]): d for d in docs}
        used = set()
        cur_d, cur_f = [], []
        for f in facts:
            k = (f["doc_id"], f["version"])
            if k not in used:
                used.add(k)
                cur_d.append(dmap[k])
            cur_f.append(f)
            if len(cur_f) >= a.batch_facts:
                yield {"documents": cur_d, "facts": cur_f}
                cur_d, cur_f = [], []
        rest = [d for k, d in dmap.items() if k not in used]
        if cur_f or rest:
            yield {"documents": cur_d + rest, "facts": cur_f}
    n_b = 0
    with open(out / "phase1.jsonl", "w", encoding="utf-8") as fo:
        for b in batches(c.phase1_docs, c.phase1_facts):
            fo.write(json.dumps({"part": "test", **b}, ensure_ascii=False) + "\n"); n_b += 1
        if n_bg:
            cur_d, cur_f = [], []
            for d, fs in background(c, a.wd5m, n_bg, over, bg_stats, a.bg_id_prefix):
                cur_d.append(d)
                cur_f.extend(fs)
                if len(cur_f) >= a.batch_facts:
                    fo.write(json.dumps({"part": "background", "documents": cur_d, "facts": cur_f}, ensure_ascii=False) + "\n")
                    n_b += 1
                    cur_d, cur_f = [], []
            if cur_f:
                fo.write(json.dumps({"part": "background", "documents": cur_d, "facts": cur_f}, ensure_ascii=False) + "\n")
                n_b += 1
    with open(out / "phase2.jsonl", "w", encoding="utf-8") as fo:
        fo.write(json.dumps({"part": "edits", "edits": c.phase2}, ensure_ascii=False) + "\n")
    # questions
    nq = collections.Counter()
    seen_q = set()
    sample_worlds = collections.defaultdict(set)
    with open(out / "questions.jsonl", "w", encoding="utf-8") as fo:
        for name, wf, itf, lf in SETS:
            labels = {l["item_id"]: l for l in jl(root / lf)}
            worlds = worlds_by_file[wf]
            widx = {w["world_id"]: i for i, w in enumerate(worlds)}
            if itf:
                pairs = [(v["world_id"], it) for v in jl(root / itf) for it in v["items"]]
            else:
                pairs = [(w["world_id"], it) for w in worlds for it in w.get("items", [])]
            n_sw, n_si = SAMPLE.get(name, (0, 0))
            for wid, it in pairs:
                lab = labels.get(it["item_id"])
                if lab is None:
                    nq[f"{name}: items without a label"] += 1
                    continue
                phase = it.get("phase") or lab.get("phase") or "after"
                gold = lab.get("gold_answers") or ([lab["gold_answer"]] if lab.get("gold_answer") is not None else [])
                key = (name, it["question"], phase, lab.get("gold_status"), json.dumps(sorted(map(str, gold))))
                if key in seen_q:
                    nq[f"{name}: duplicate questions (same text, phase, gold)"] += 1
                    continue
                seen_q.add(key)
                wi = widx[wid]
                samp = wi < n_sw
                if samp:
                    sample_worlds[wf].add(wid)
                support = lab.get("support_fact_ids") or []
                q = {"qid": f"{name}:{it['item_id']}", "set": name, "item_id": it["item_id"], "world_id": wid,
                     "world_file": wf, "question": it["question"], "phase": phase, "form": lab.get("form"),
                     "case_type": lab.get("case_type"), "gold_status": lab.get("gold_status"), "gold_answers": gold,
                     "trace": lab.get("trace"), "support_fact_ids": support,
                     "support_shared": [c.canon_of.get((wid, s)) for s in support], "sample": samp}
                fo.write(json.dumps(q, ensure_ascii=False) + "\n")
                nq[f"{name}: questions"] += 1
                if samp:
                    nq[f"{name}: model sample"] += 1
    with open(out / "worlds_sample.jsonl", "w", encoding="utf-8") as fo:
        for wf in seen_files:
            for w in worlds_by_file[wf]:
                if w["world_id"] in sample_worlds[wf]:
                    v = {k: x for k, x in w.items() if k != "items"}
                    fo.write(json.dumps({"world_file": wf, "world_id": w["world_id"], "world": v}, ensure_ascii=False) + "\n")
    # world membership of canonical facts (for the cross-world evidence check)
    with open(out / "world_facts.jsonl", "w", encoding="utf-8") as fo:
        for wid, s in c.world_facts.items():
            fo.write(json.dumps({"world_id": wid, "facts": sorted(s)}, ensure_ascii=False) + "\n")
    n_edit_facts = sum(len(e.get("facts", [])) + (1 if e.get("fact") else 0) for e in c.phase2)
    man = {"sets": [s[0] for s in SETS], "phase1_batches": n_b, "test_base_facts": n_test_base,
           "test_base_documents": len(c.phase1_docs), "phase2_edits": len(c.phase2), "phase2_new_facts": n_edit_facts,
           "background": dict(bg_stats), "total_facts": n_test_base + n_edit_facts + bg_stats["background facts"],
           "total_documents": len(c.phase1_docs) + bg_stats["background documents"] + sum(1 for e in c.phase2 if e.get("doc")),
           "consolidation": dict(c.log), "questions": dict(nq),
           "inputs": {s[0]: {k: (sha256_file(root / p) if p else None) for k, p in (("worlds", s[1]), ("items", s[2]), ("labels", s[3]))}
                      for s in SETS},
           "files": {}}
    for f in ("phase1.jsonl", "phase2.jsonl", "questions.jsonl", "worlds_sample.jsonl", "world_facts.jsonl"):
        man["files"][f] = sha256_file(out / f)
    (out / "corpus.json").write_text(json.dumps(man, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({k: man[k] for k in man if k not in ("inputs", "files")}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
