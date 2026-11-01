#!/usr/bin/env python3
"""Test 8: counting is off. Every question of the given worlds file goes through `_ask` with the default rule and must be declined with COUNTING_OFF, no count served.

  python3 -B e2e_counting.py --bin SERVER --model DIR --worlds counting_worlds.jsonl --out DIR
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e2e_lib import Client, Report, Server, bulk_body, ns_of

ap = argparse.ArgumentParser()
for k in ("bin", "model", "worlds", "out"):
    ap.add_argument("--" + k, required=True)
ap.add_argument("--cpus", default="12-19"); ap.add_argument("--port", type=int, default=19881)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
srv = Server(a.bin, a.model, os.path.join(a.out, "data"), a.port, a.cpus)
srv.start()
ok = False
try:
    c = Client(a.port)
    r = Report("t8", a.out)
    n = 0
    for k, l in enumerate(open(a.worlds, encoding="utf-8")):
        if not l.strip():
            continue
        w = json.loads(l)
        ns = ns_of(f"count{k}")
        st, b = c.call("POST", f"/_facts/{ns}/_bulk", bulk_body(w))
        r.check(f"world {k} stored", st == 200, str(b)[:200])
        for it in w["items"]:
            n += 1
            st, o = c.call("POST", f"/_facts/{ns}/_ask", {"question": it["question"], "item_id": it["item_id"]})
            good = st == 200 and o.get("final_status") == "UNSUPPORTED" and o.get("final_answer") is None and o.get("reason_code") == "COUNTING_OFF" and not o.get("evidence")
            r.check(f"{it['item_id']}: declined with COUNTING_OFF", good, [st, o.get("final_status"), o.get("final_answer"), o.get("reason_code")] if isinstance(o, dict) else [st, str(o)[:200]])
    r.fact("questions", n)
    ok = r.end()
finally:
    srv.stop()
sys.exit(0 if ok else 1)
