#!/usr/bin/env python3
"""pass / fail per test from a pulled run directory"""
import glob, json, os, sys
d = sys.argv[1]
for f in sorted(glob.glob(os.path.join(d, "pulled", "func", "t?.json"))):
    r = json.load(open(f)); n = len(r["checks"]); ok = sum(c["ok"] for c in r["checks"])
    print(r["test"], "PASS" if r["pass"] else "FAIL" if r["pass"] is False else "NOT FINISHED", f"{ok}/{n}")
    for c in r["checks"]:
        if not c["ok"]:
            print("   -", c["what"], str(c["detail"])[:200])
p = os.path.join(d, "score", "bench_compare.json")
if os.path.exists(p):
    print("t6 gate", json.load(open(p))["gate"].get("pass"))
for f in sorted(glob.glob(os.path.join(d, "pulled", "bench", "latency.*.json"))):
    r = json.load(open(f)); print(os.path.basename(f), {k: round(v) for k, v in r["decline_v2"].items()}, r["rss_decline_v2"])
