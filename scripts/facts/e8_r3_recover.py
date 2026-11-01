#!/usr/bin/env python3
"""E8 round 3 (24_shared3): which of the baseline's losses an option recovers, by the loss's first-departure class
(e8_attrib.py outputs: the baseline's rows and classes, the option's main_ok). A loss is recovered when the option's
shared arm answers it (supported); broken = baseline shared right, option shared wrong. Counts only.
usage: e8_r3_recover.py --rows ROWS.json --out-md FILE --out-json FILE   (rows file of e8_r3_table.py)
"""
import argparse, collections, json
from pathlib import Path

CLS = ["counterfactual hub fact", "loop to the step limit", "fact from another world", "background fact",
       "linker or manifest choice", "other"]
SHORT = {"counterfactual hub fact": "hub", "loop to the step limit": "loop", "fact from another world": "other world",
         "background fact": "background", "linker or manifest choice": "linker/manifest", "other": "other"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--out-json", required=True)
    a = ap.parse_args()
    rows = json.loads(Path(a.rows).read_text())
    md = ["| store | option | recovered of the baseline's losses: " + " / ".join(SHORT[c] for c in CLS) + " | all | broken |",
          "|---|---|---|---:|---:|"]
    out = []
    for r in rows:
        at = json.loads(Path(r["attrib"]).read_text())
        bl = json.loads(Path(r["baseline_attrib"]).read_text())
        cls_of = {x["qid"]: x["class"] for x in bl["rows"]}
        tot, rec = collections.Counter(), collections.Counter()
        for q, c in cls_of.items():
            tot[c] += 1
            ok = at["main_ok"].get(q)
            if ok is not None and ok[1]:
                rec[c] += 1
        broken = sum(1 for q, (po, so) in at["main_ok"].items() if bl["main_ok"].get(q, [0, 0])[1] and not so)
        cells = " / ".join(f"{rec[c]} of {tot[c]}" if tot[c] else "-" for c in CLS)
        md.append(f"| {r['size']} | {r['option']} | {cells} | {sum(rec.values())} of {sum(tot.values())} | {broken} |")
        out.append({"size": r["size"], "option": r["option"], "baseline_losses": dict(tot), "recovered": dict(rec),
                    "broken": broken})
    Path(a.out_md).write_text("\n".join(md) + "\n")
    Path(a.out_json).write_text(json.dumps(out, indent=1) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
