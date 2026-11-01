#!/usr/bin/env python3
"""Pick the test 2 sample from the SELECT half of the bench, one list per path, and write it as JSON. Runs where the research tree is.

Only questions where the frozen Python policy reaches the path on BOTH the FP32 and the weight-only row sets are candidates, so this
is a path test and says nothing about rates: the bench run (test 6) gives those. Order: sha256(item id), up to --n per path.

  python3 -I -B make_sample.py --mvp research/xerj_r_mvp --seed-dir s29 --out sample.json
"""
import argparse, hashlib, json, sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--mvp", required=True); ap.add_argument("--out", required=True); ap.add_argument("--n", type=int, default=5)
ap.add_argument("--seed-dir", default="s29")
a = ap.parse_args()
MVP = Path(a.mvp).resolve()
sys.dont_write_bytecode = True
sys.path[:0] = [str(MVP / "decline_v1"), str(MVP / "tools")]
import evaluate as E                     # noqa: E402
import policy as P                       # noqa: E402
import replay as RP                      # noqa: E402
from common import load_labels, load_split  # noqa: E402

cfg = dict(E.verify_frozen_v2()["policy"])
split = load_split()
order, labels = load_labels()
sel = [i for i in order if split["items"][i] == "select"]              # the check half is never read here
ROOT = MVP / "runs" / "hybrid_v2_eval" / "rented_pairs"
fam = {"f32": ROOT / "cf32" / "x68_cf32_C__external_executor", "w8": ROOT / "cf32_int8" / "x68_cf32_C_int8__external_executor"}
rows = {k: E.read_rows(v / a.seed_dir / "hybrid_v2_bench.rows.jsonl") for k, v in fam.items()}
worlds = RP.Worlds()
stage = {k: {i: P.stages(rows[k][i], worlds, None, cfg)["P3+D2+D2b"] for i in sel} for k in rows}
paths = {k: [] for k in ("lookup", "fk_chain", "comparison", "missing", "repaired", "multi_value", "counting", "uncomputable")}
FK = ("airport_country", "airport_region")


def both(i, f):
    return all(f(rows[k][i], stage[k][i]) for k in rows)


for i in sorted(sel, key=lambda x: hashlib.sha256(x.encode()).hexdigest()):
    lab = labels[i]
    ok = lambda r, o: o["accepted"] and P.right(r, o)
    declined = lambda r, o: not o["accepted"] and r["status"] == "ANSWER"
    fixed = lambda r, o: o["accepted"] and o["repaired"] and P.right(r, o)
    want = lab["scope"] == "IN_SCOPE"
    if want and both(i, fixed):
        paths["repaired"].append(i)
    if want and lab["gold_status"] == "UNKNOWN" and both(i, lambda r, o: r["status"] == "UNKNOWN"):
        paths["missing"].append(i)
    if want and lab["gold_status"] == "ANSWER" and both(i, lambda r, o: ok(r, o) and not o["repaired"]):
        fam_is_fk = lab["family"].startswith(FK)
        paths["fk_chain" if fam_is_fk else "lookup" if lab["form"] == "lookup" else "comparison"].append(i)
    if not want:
        sr = lab.get("scope_reason") or ""
        if sr.startswith("COUNT") and both(i, lambda r, o: not o["accepted"]):
            paths["counting"].append(i)
        elif "several" in sr.lower() and both(i, declined):
            paths["multi_value"].append(i)
        elif not sr.startswith("COUNT") and "several" not in sr.lower() and both(i, declined):
            paths["uncomputable"].append(i)
out = {"schema": "xerj-e2e-sample-v1", "half": "select", "note": "path test: candidates reach the path on both row sets", "n_candidates": {k: len(v) for k, v in paths.items()}, "items": []}
for k, ids in paths.items():
    for i in ids[:a.n]:
        lab = labels[i]
        out["items"].append({"path": k, "item_id": i, "gold_status": lab["gold_status"], "gold_answer": lab.get("gold_answer"), "family": lab["family"], "scope_reason": lab.get("scope_reason"),
                             "expect_answer": {k2: stage[k2][i]["answer"] for k2 in rows}})
Path(a.out).write_text(json.dumps(out, indent=1, ensure_ascii=False))
print(out["n_candidates"])
