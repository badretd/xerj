#!/usr/bin/env python3
"""E8 round 3 (24_shared3): per-set view of an options table (rows file of e8_r3_table.py): for each row and each set of
the main cohort, n, one store per world, shared, shared-only / per-world-only and the difference in points (supported
accuracy, e8_score.py's "main / <set>" entries), plus fixed / broken against the baseline row per set (e8_attrib.py's
main_ok). Development data only; counts, no question text.
usage: e8_r3_perset.py --rows ROWS.json --corpus DIR --out-md FILE --out-json FILE
"""
import argparse, collections, json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--out-json", required=True)
    a = ap.parse_args()
    rows = json.loads(Path(a.rows).read_text())
    set_of = {}
    with open(Path(a.corpus) / "questions.jsonl", encoding="utf-8") as f:
        for l in f:
            if l.strip():
                q = json.loads(l)
                set_of[q["qid"]] = q["set"]
    out, md = [], ["| store | option | set | n | per world | shared | shared-only / per-world-only | diff (pts) | vs baseline: fixed / broken |",
                   "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        sc = json.loads(Path(r["score"]).read_text())
        at = json.loads(Path(r["attrib"]).read_text())
        bl = json.loads(Path(r["baseline_attrib"]).read_text())
        fb = collections.defaultdict(lambda: [0, 0])
        for q, (po, so) in at["main_ok"].items():
            b = bl["main_ok"].get(q)
            if b is None:
                continue
            s = set_of.get(q, "?")
            if so and not b[1]:
                fb[s][0] += 1
            if b[1] and not so:
                fb[s][1] += 1
        for k, v in sc["model"]["paired"].items():
            if not k.startswith("main / "):
                continue
            s = k.split("/ ", 1)[1]
            m = v["raw supported"]
            fx, br = fb.get(s, [0, 0])
            md.append(f"| {r['size']} | {r['option']} | {s} | {m['n']} | {m['perworld']:.4f} | {m['shared']:.4f} | "
                      f"{m['shared_only']} / {m['perworld_only']} | {100 * m['diff']:+.2f} | {fx} / {br} |")
            out.append({"size": r["size"], "option": r["option"], "set": s, "n": m["n"], "perworld": m["perworld"],
                        "shared": m["shared"], "shared_only": m["shared_only"], "perworld_only": m["perworld_only"],
                        "diff": m["diff"], "fixed_vs_baseline": fx, "broken_vs_baseline": br})
    Path(a.out_md).write_text("\n".join(md) + "\n")
    Path(a.out_json).write_text(json.dumps(out, indent=1) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
