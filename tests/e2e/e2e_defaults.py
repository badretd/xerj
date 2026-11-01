#!/usr/bin/env python3
"""Test 9: a plain server. Nothing but the model directory is set, and without it `_ask` is a clear 503.

  python3 -B e2e_defaults.py --bin SERVER --model DIR --worlds model_dev.jsonl --sample sample.json --out DIR
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e2e_lib import Client, Report, Server, bulk_body, load_worlds, ns_of

ap = argparse.ArgumentParser()
for k in ("bin", "model", "worlds", "sample", "out"):
    ap.add_argument("--" + k, required=True)
ap.add_argument("--cpus", default="12-19"); ap.add_argument("--port", type=int, default=19981)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
s = next(x for x in json.load(open(a.sample))["items"] if x["path"] == "lookup")
w = load_worlds(a.worlds, {s["item_id"]})[s["item_id"]]
q = w["items"][0]["question"]
r = Report("t9", a.out)
ns = ns_of("t9")

srv = Server(a.bin, None, os.path.join(a.out, "data_nomodel"), a.port, a.cpus, threads=0)
srv.start()
c = Client(a.port)
st, b = c.call("POST", f"/_facts/{ns}/_bulk", bulk_body(w))
r.check("no model: writes work", st == 200, str(b)[:200])
st, o = c.call("POST", f"/_facts/{ns}/_ask", {"question": q, "item_id": s["item_id"]})
r.check("no model: _ask is 503 MODEL_UNAVAILABLE, not retryable", st == 503 and "MODEL_UNAVAILABLE" in json.dumps(o) and o.get("retryable") is False, [st, str(o)[:300]])
srv.stop()

srv = Server(a.bin, a.model, os.path.join(a.out, "data_plain"), a.port, a.cpus, threads=0)
srv.start()
c = Client(a.port)
t0 = time.time()
while srv.rss()["VmRSS"] < 100 and time.time() - t0 < 60:
    time.sleep(0.5)
r.fact("rss_before_first_ask", srv.rss())
r.check("model dir only: the compiler loads before the first ask", srv.rss()["VmRSS"] >= 100, srv.rss())
c.call("POST", f"/_facts/{ns}/_bulk", bulk_body(w))
st, o = c.call("POST", f"/_facts/{ns}/_ask", {"question": q, "item_id": s["item_id"]})
r.check("model dir only: default rule is decline_v2 and the answer is served", st == 200 and o.get("verify", {}).get("option") == "decline_v2" and o.get("final_status") == "ANSWER", str(o)[:400])
r.fact("model_info", o.get("model_info"))
r.check("model dir only: weight-only INT8", "w8" in json.dumps(o.get("model_info")), o.get("model_info"))
r.fact("threads_seen", [l.strip() for l in open(f"/proc/{srv.real_pid()}/status") if l.startswith("Threads")])
srv.stop()
sys.exit(0 if r.end() else 1)
