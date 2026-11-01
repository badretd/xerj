#!/usr/bin/env python3
"""Test 6 and 7: the whole bench over HTTP, one namespace per world as the bench is built, then latency and RSS. Resumable.

  load    `_bulk` every world, acknowledged writes recorded in OUT/loaded.jsonl
  ask     real restart, then every question with verify=decline_v2 and the bench item id; rows in OUT/PRECISION.rows.jsonl
  latency real restart, one question at a time, THREADS threads per question, a fixed random sample; default response and decline_v2

  python3 -B e2e_bench.py --bin SERVER --model DIR --worlds model_dev.jsonl --out DIR --phase load|ask|latency --precision unset|f32
"""
import argparse, json, os, queue, random, statistics, sys, threading, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e2e_lib import Client, Server, bulk_body, load_worlds, now, ns_of

ap = argparse.ArgumentParser()
for k in ("bin", "model", "worlds", "out", "phase"):
    ap.add_argument("--" + k, required=True)
ap.add_argument("--ids", required=True, help="file of item ids to run, the SELECT half"); ap.add_argument("--precision", default="unset"); ap.add_argument("--cpus", default="12-19"); ap.add_argument("--port", type=int, default=19781)
ap.add_argument("--conc", type=int, default=8); ap.add_argument("--threads", type=int, default=1); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--latency-n", type=int, default=400)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
worlds = load_worlds(a.worlds, set(open(a.ids).read().split()))
ids = list(worlds)[:a.limit or None]
srv = Server(a.bin, a.model, os.path.join(a.out, "data"), a.port, a.cpus, threads=a.threads, precision=None if a.precision == "unset" else a.precision)
state = os.path.join(a.out, f"{a.phase}.{a.precision}.state.json")


def save(**k):
    d = {"phase": a.phase, "precision": a.precision, "updated": now(), **k}
    json.dump(d, open(state + ".tmp", "w"), indent=1)
    os.replace(state + ".tmp", state)


def done_ids(path, key="item_id"):
    if not os.path.exists(path):
        return set()
    out = set()
    good = open(path + ".clean", "w")
    for l in open(path, encoding="utf-8"):
        try:
            r = json.loads(l)
        except ValueError:
            continue
        out.add(r[key])
        good.write(l)
    good.close()
    os.replace(path + ".clean", path)
    return out


def pool(todo, fn, conc):
    q = queue.Queue()
    for x in todo:
        q.put(x)
    errs = []

    def run():
        c = Client(a.port)
        while True:
            try:
                x = q.get_nowait()
            except queue.Empty:
                return
            for t in range(6):
                try:
                    fn(c, x)
                    break
                except Exception as e:
                    if t == 5:
                        errs.append([x, repr(e)[:300]])
                    time.sleep(2 ** t)

    ts = [threading.Thread(target=run, daemon=True) for _ in range(conc)]
    for t in ts:
        t.start()
    while any(t.is_alive() for t in ts):
        save(remaining=q.qsize(), errors=len(errs), rss=srv.rss())
        time.sleep(5)
    return errs


def load():
    path = os.path.join(a.out, "loaded.jsonl")
    have = done_ids(path)
    todo = [i for i in ids if i not in have]
    f = open(path, "a", encoding="utf-8")
    lock = threading.Lock()

    def one(c, i):
        w = worlds[i]
        t = time.perf_counter()
        st, b = c.call("POST", f"/_facts/{ns_of(i)}/_bulk", bulk_body(w))
        if st != 200:
            raise RuntimeError((st, str(b)[:200]))
        with lock:
            f.write(json.dumps({"item_id": i, "to_rev": b["committed"]["to_rev"], "ms": (time.perf_counter() - t) * 1e3, "records": b["records_indexed"]}) + "\n")
            f.flush()

    t0 = time.time()
    errs = pool(todo, one, 4)
    save(remaining=0, errors=len(errs), first_errors=errs[:5], load_s=time.time() - t0, loaded=len(done_ids(path)), rss=srv.rss())


def row(i, o, ms):
    w = worlds[i]
    prog = [{"op": p[0], "pointer": p[1], **({"direction": "in"} if len(p) > 2 and p[2] else {})} for p in o.get("program") or []]
    trace = [{"step": t["step"], "op": t["op"], "pointer": t["pointer"], "result": t["result"], **({"direction": "in"} if t.get("inverse") else {})} for t in o.get("trace") or []]
    return {"item_id": i, "question": w["items"][0]["question"], "phase": w["items"][0].get("phase"), "status": o.get("status"), "answer": o.get("answer"), "program": prog, "execution_trace": trace,
            "evidence_ids": o.get("evidence_ids"), "error": o.get("error"), "final_status": o.get("final_status"), "final_answer": o.get("final_answer"),
            "final_evidence_ids": o.get("final_evidence_ids"), "reason_code": o.get("reason_code"), "decline": o.get("decline"), "repair": o.get("repair"), "evidence": o.get("evidence"),
            "verify": o.get("verify"), "decisions": o.get("decisions"), "client_ms": ms, "handler_ms": o.get("handler_ms"), "model_info": None}


def asks():
    path = os.path.join(a.out, f"{a.precision}.rows.jsonl")
    have = done_ids(path)
    todo = [i for i in ids if i not in have]
    f = open(path, "a", encoding="utf-8")
    lock = threading.Lock()
    info = {}
    bad = []

    def one(c, i):
        w = worlds[i]
        t = time.perf_counter()
        st, o = c.call("POST", f"/_facts/{ns_of(i)}/_ask", {"question": w["items"][0]["question"], "item_id": i, "verify": "decline_v2", "wait_for_ms": 120000})
        ms = (time.perf_counter() - t) * 1e3
        if st != 200:
            with lock:
                bad.append([i, st, str(o)[:200]])
            raise RuntimeError(st)
        info.setdefault("model_info", o.get("model_info"))
        with lock:
            f.write(json.dumps(row(i, o, ms), ensure_ascii=False) + "\n")
            f.flush()

    t0 = time.time()
    errs = pool(todo, one, a.conc)
    save(remaining=0, errors=len(errs), first_errors=errs[:5], transient=len(bad), first_transient=bad[:5], ask_s=time.time() - t0, conc=a.conc, threads=a.threads, rows=len(done_ids(path)), rss=srv.rss(), **info)


def pct(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(len(v) * p))]


def latency():
    rng = random.Random(20261031)
    pick = rng.sample(ids, min(a.latency_n, len(ids)))
    c = Client(a.port)
    res = {"n": len(pick), "threads": a.threads, "precision": a.precision, "rss_ready": srv.rss(), "note": "one question at a time, real HTTP, fresh revision head"}
    for name, verify in (("warmup", "decline_v2"), ("decline_v2", "decline_v2"), ("default", None), ("decline_v2_again", "decline_v2")):
        ms = []
        for i in pick if name != "warmup" else pick[:20]:
            b = {"question": worlds[i]["items"][0]["question"], "item_id": i, "wait_for_ms": 120000}
            if verify:
                b["verify"] = verify
            t = time.perf_counter()
            st, o = c.call("POST", f"/_facts/{ns_of(i)}/_ask", b)
            ms.append((time.perf_counter() - t) * 1e3)
            assert st == 200, (i, st, str(o)[:200])
        res[name] = {"p50": pct(ms, .5), "p95": pct(ms, .95), "p99": pct(ms, .99), "mean": statistics.mean(ms), "max": max(ms)}
        res["rss_" + name] = srv.rss()
        save(**res)
    json.dump(res, open(os.path.join(a.out, f"latency.{a.precision}.t{a.threads}.json"), "w"), indent=1)


print(now(), a.phase, a.precision, len(ids), flush=True)
ready = srv.start()
res0 = {"ready_s": ready, "rss_at_ready": srv.rss()}
print("ready", res0, flush=True)
json.dump(res0, open(os.path.join(a.out, f"{a.phase}.{a.precision}.t{a.threads}.ready.json"), "w"))
try:
    {"load": load, "ask": asks, "latency": latency}[a.phase]()
finally:
    print("rss", srv.rss(), flush=True)
    print("stop rc", srv.stop(), now(), flush=True)
