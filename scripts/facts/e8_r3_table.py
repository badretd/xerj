#!/usr/bin/env python3
"""E8 round 3 (24_shared3): the options table on the round-3 development sample (development data: no gate claim).

One row per store size x option: the gate-style numbers from e8_score.py (main cohort, raw supported, shared vs one store
per world, Newcombe / cluster lower bound), what the option changes against the S2 baseline of the same size (questions
the option fixes / breaks in the shared arm), the residual losses by class (e8_attrib.py), model decisions per question
and client time (indicative only: shared CPUs, concurrent asks); V3 = main-cohort shared-store answers that cite a record of
another world or of the background (scored as not supported), and the own-world supported difference (reported, not the
gate metric).

usage: e8_r3_table.py --rows ROWS.json --out-md FILE --out-json FILE
  ROWS.json: [{"size": "10K", "option": "...", "score": FILE, "attrib": FILE, "baseline_attrib": FILE}, ...]
"""
import argparse, json
from pathlib import Path

CLS = [("counterfactual hub fact", "hub"), ("loop to the step limit", "loop"), ("fact from another world", "other world"),
       ("background fact", "background"), ("linker or manifest choice", "linker/manifest"), ("other", "other")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--out-json", required=True)
    a = ap.parse_args()
    rows = json.loads(Path(a.rows).read_text())
    md = ["| store | option | n (main) | per world | shared | shared-only / per-world-only | diff (pts) | lower bound (pts) | V1+V2 | "
          "vs S2 baseline: fixed / broken | residual losses: " + " / ".join(c[1] for c in CLS) + " | decisions per question | "
          "client ms p50 / p95 | loop guard fired | V3 (main) | own-world supported diff (pts) |",
          "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---|---:|---:|---:|"]
    out = []
    for r in rows:
        sc = json.loads(Path(r["score"]).read_text())
        at = json.loads(Path(r["attrib"]).read_text())
        bl = json.loads(Path(r["baseline_attrib"]).read_text())
        m = sc["model"]["paired"]["main"]["raw supported"]
        lbs = [x for x in (m["lb95_newcombe"], m["lb95_cluster"]) if x is not None]
        lb = min(lbs) if lbs else None
        v = sc["model"]["visibility"]["V1+V2 total"]
        v3 = sc["model"]["visibility"].get("V3 in the main cohort")
        ow = sc["model"]["paired"]["main"].get("raw own-world supported") or {}
        ow_diff = ow.get("diff")
        fixed = broken = 0
        for q, (po, so) in at["main_ok"].items():
            b = bl["main_ok"].get(q)
            if b is None:
                continue
            if so and not b[1]:
                fixed += 1
            if b[1] and not so:
                broken += 1
        cl = at["classes"]
        res = " / ".join(str(cl[c[0]]["count"]) for c in CLS)
        sa = at["shared_arm"]
        md.append(f"| {r['size']} | {r['option']} | {m['n']} | {m['perworld']:.4f} | {m['shared']:.4f} | {m['shared_only']} / {m['perworld_only']} | "
                  f"{100 * m['diff']:+.2f} | {100 * lb:+.2f} | {v} | {fixed} / {broken} | {res} | {sa['decisions_mean']:.2f} | "
                  f"{sa['client_ms_p50']:.0f} / {sa['client_ms_p95']:.0f} | {sa['loop_stopped']} | {v3} | "
                  f"{'' if ow_diff is None else format(100 * ow_diff, '+.2f')} |")
        out.append({"size": r["size"], "option": r["option"], "n": m["n"], "perworld": m["perworld"], "shared": m["shared"],
                    "shared_only": m["shared_only"], "perworld_only": m["perworld_only"], "diff": m["diff"], "lb": lb,
                    "lb_newcombe": m["lb95_newcombe"], "lb_cluster": m["lb95_cluster"], "V1V2": v, "V3_main": v3,
                    "own_world_supported": ow,
                    "fixed_vs_s2": fixed, "broken_vs_s2": broken,
                    "residual_classes": {c[0]: {"count": cl[c[0]]["count"], "ci95": cl[c[0]]["ci95"]} for c in CLS},
                    "losses_ci95": cl["all losses"]["ci95"], "shared_arm": sa, "perworld_arm": at.get("perworld_arm"),
                    "all_questions_raw_supported": sc["model"]["paired"].get("all questions", {}).get("raw supported")})
    Path(a.out_md).write_text("\n".join(md) + "\n")
    Path(a.out_json).write_text(json.dumps(out, indent=1) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
