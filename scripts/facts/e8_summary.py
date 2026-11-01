#!/usr/bin/env python3
"""One line per scored run: main-cohort supported accuracy per world / shared, discordant counts, difference, lower bound
(smaller of Newcombe and cluster), verdict, visibility; e8_summary.py LABEL=score.json ... [--json out.json]"""
import json, sys
rows = []
args = [a for a in sys.argv[1:] if not a.startswith("--json")]
out = None
for a in sys.argv[1:]:
    if a.startswith("--json="):
        out = a.split("=", 1)[1]
for lab, p in [a.split("=", 1) for a in args]:
    try:
        d = json.load(open(p))
    except Exception as e:
        print(lab, "missing", e)
        continue
    t = d["exact_path"]["total"]
    m = d["model"]
    r = m["paired"]["main"]["raw supported"]
    lbs = [x for x in (r["lb95_newcombe"], r["lb95_cluster"]) if x is not None]
    lb = min(lbs) if lbs else None
    v = m["visibility"]
    ver = m["paired"]["main"].get("verified supported", {})
    row = {"label": lab, "n_main": r["n"], "perworld": r["perworld"], "shared": r["shared"], "shared_only": r["shared_only"],
           "perworld_only": r["perworld_only"], "diff": r["diff"], "lb95": lb, "lb95_newcombe": r["lb95_newcombe"],
           "lb95_cluster": r["lb95_cluster"], "verdict": r["verdict"], "changed": t.get("cohort changed"),
           "pw_fail": t.get("cohort pw_fail", 0), "V1V2": v["V1+V2 total"], "V3_main": v["V3 in the main cohort"],
           "verified_perworld": ver.get("perworld"), "verified_shared": ver.get("shared"),
           "oracle_shared_main": t.get("main: shared oracle correct"), "by_set": {}}
    for k, x in m["paired"].items():
        if k.startswith("main / "):
            rs = x["raw supported"]
            row["by_set"][k[7:]] = [rs["n"], rs["perworld"], rs["shared"], rs["shared_only"], rs["perworld_only"]]
    rows.append(row)
    print(f"{lab:28s} n={r['n']:5d} pw={r['perworld']:.4f} sh={r['shared']:.4f} +{r['shared_only']}/-{r['perworld_only']} "
          f"diff={100*r['diff']:+.2f} lb={100*lb:+.2f} {r['verdict']:12s} changed={t.get('cohort changed')} "
          f"V1+V2={v['V1+V2 total']} V3main={v['V3 in the main cohort']} | "
          + " ".join(f"{s}:{x[0]}/+{x[3]}-{x[4]}" for s, x in sorted(row["by_set"].items())))
if out:
    json.dump(rows, open(out, "w"), indent=1)
