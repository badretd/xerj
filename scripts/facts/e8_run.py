#!/usr/bin/env python3
"""E8 shared store over the product path: load, restart, rebuild, ask (release server, HTTP, /_facts).

  load  start the server on a fresh data dir; POST the shared corpus to /_facts/{shared}/_bulk (phase 1 batches -> revision
        B, phase 2 edits -> revision F); POST every sampled world to /_facts/w-NNNN/_world (one namespace per world, the
        reference layout, in the same engine); flush; stop (SIGTERM). Per-batch timings, server RSS.
  ask   start the server on that data dir (a real restart), time the first access of each namespace (the projection is
        rebuilt from the engine's index), then ask the model sample with the default item id ("lab/q/" + sha256(question):
        no world id enters the run) in the per-world namespaces and in the shared namespace, at the revision of each
        question's phase, N asks in parallel. Writes asks_<arm>.jsonl (status, answer, evidence ids, program, verify,
        candidates, hop reads, stage timings, client latency).

  restart  start the server on a loaded data dir (a real restart), time boot and the first access of the shared namespace
        (the projection rebuilt from the engine index), record RSS, stop; --repeat N times (restart.json).

Large loads: --min-avail-gib G stops the load (server stopped by SIGTERM, the run marked "aborted") as soon as the host's
MemAvailable falls below G GiB (checked every 2 s and between batches); --skip-worlds leaves out the per-world namespaces.

Stdlib only. usage:
  e8_run.py load --bin BIN --corpus DIR --run DIR --port P --cpus C --cap 12G [--shared e8-shared] [--worlds-max N]
                 [--skip-worlds] [--flush-every N] [--min-avail-gib G]
  e8_run.py ask  --bin BIN --corpus DIR --run DIR --port P --cpus C --cap 12G --arms perworld,shared [--par 6] [--limit N]
                 [--precision int8-guard]
  e8_run.py restart --bin BIN --corpus DIR --run DIR --port P --cpus C --cap 12G [--repeat 3]
"""
import argparse, json, os, signal, subprocess, sys, threading, time, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def call(port, method, path, body=None, timeout=3600, raw=None):
    data = raw if raw is not None else (None if body is None else json.dumps(body, ensure_ascii=False).encode())
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data, method=method,
                                 headers={"Content-Type": "application/json"})

    def parse(b):
        try:
            return json.loads(b or b"null")
        except ValueError:
            return b.decode("utf-8", "replace")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, parse(r.read())
    except urllib.error.HTTPError as e:
        return e.code, parse(e.read())


def jl(p):
    with open(p, encoding="utf-8") as f:
        for l in f:
            if l.strip():
                yield json.loads(l)


def loadavg():
    return open("/proc/loadavg").read().split()[:3]


def memavail_gib():
    for l in open("/proc/meminfo"):
        if l.startswith("MemAvailable:"):
            return round(int(l.split()[1]) / 1048576, 1)


def now():
    return subprocess.check_output(["date", "+%F %T %Z"]).decode().strip()


class Server:
    """the release server under `capped` (memory cap, CPU set, nice 15, oom_score_adj 1000), loopback only"""

    def __init__(self, a, data_dir, run):
        self.a, self.data_dir, self.run, self.proc, self.pid = a, Path(data_dir), Path(run), None, None
        self.env = dict(os.environ, XERJ_FACTS_PRECISION=getattr(a, "precision", "int8-guard"))
        self.conf = self.run / "engine.toml"
        extra = ""
        if getattr(a, "engine_toml", None):
            # extra engine settings (e.g. a [merge] section for a 10M load), appended verbatim
            extra = Path(a.engine_toml).read_text()
        self.conf.write_text(f"[server]\nrest_port = {a.port - 1}\nes_compat_port = {a.port}\ngrpc_port = {a.port + 1}\n" + extra)
        self.peak = 0
        self._stop_poll = threading.Event()
        # --rss-trace: VmRSS every 0.25 s into <run>/rss.tsv (unix time, MiB), to see which phase holds the memory
        self.trace = open(self.run / "rss.tsv", "a") if getattr(a, "rss_trace", False) else None

    def _find_pid(self):
        # the server process: our binary, our data dir, and a descendant of our own child (never matched by name alone)
        want = str(self.data_dir)
        for d in os.listdir("/proc"):
            if not d.isdigit():
                continue
            try:
                cmd = open(f"/proc/{d}/cmdline", "rb").read().split(b"\0")
            except OSError:
                continue
            if not cmd or os.path.basename(cmd[0].decode(errors="replace")) != os.path.basename(self.a.bin):
                continue
            if want.encode() not in cmd:
                continue
            p = int(d)
            q = p
            for _ in range(16):
                try:
                    q = int(open(f"/proc/{q}/stat").read().rsplit(")", 1)[1].split()[1])
                except OSError:
                    break
                if q == self.proc.pid:
                    return p
                if q <= 1:
                    break
        return None

    def rss(self):
        """(VmRSS MiB, VmHWM MiB) of the server process"""
        if not self.pid:
            return None, None
        try:
            st = open(f"/proc/{self.pid}/status").read()
        except OSError:
            return None, None
        v = {l.split(":")[0]: int(l.split()[1]) for l in st.splitlines() if l.startswith(("VmRSS", "VmHWM"))}
        return round(v.get("VmRSS", 0) / 1024, 1), round(v.get("VmHWM", 0) / 1024, 1)

    def _poll(self):
        while not self._stop_poll.wait(0.25 if self.trace else 1.0):
            r, _ = self.rss()
            if r:
                self.peak = max(self.peak, r)
                if self.trace:
                    self.trace.write(f"{time.time():.2f}\t{r}\n")
                    self.trace.flush()

    def start(self):
        cmd = [self.a.bin, "--config", str(self.conf), "--data-dir", str(self.data_dir), "--insecure", "--no-auto-port"]
        if self.a.capped != "none":
            # memory cap, CPU set, nice 15, oom_score_adj 1000 (the engine build server's wrapper)
            cmd = [self.a.capped, self.a.cap, self.a.cpus] + cmd
        self.proc = subprocess.Popen(cmd, stdout=open(self.run / "engine.log", "ab"), stderr=subprocess.STDOUT, env=self.env,
                                     start_new_session=True)
        t0 = time.time()
        while time.time() - t0 < 900:
            if self.proc.poll() is not None:
                raise RuntimeError(f"engine exited with {self.proc.returncode}; see {self.run}/engine.log")
            try:
                if call(self.a.port, "GET", "/health/ready", timeout=5)[0] == 200:
                    self.pid = self._find_pid()
                    self._stop_poll.clear()
                    threading.Thread(target=self._poll, daemon=True).start()
                    return time.time() - t0
            except Exception:
                pass
            time.sleep(0.25)
        self.stop()
        raise RuntimeError("engine not ready after 900 s (stopped)")

    def stop(self):
        self._stop_poll.set()
        if self.proc and self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)  # our own child (sudo relays it to the server), by PID
            try:
                self.proc.wait(timeout=600)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        return self.proc.returncode if self.proc else None


class MemWatch:
    """stops the server when the host's MemAvailable falls below `gib` (0 = off); `tripped` holds the reading"""

    def __init__(self, srv, gib):
        self.srv, self.gib, self.tripped, self.min_seen = srv, gib, None, None
        self._stop = threading.Event()
        if gib > 0:
            threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while not self._stop.wait(2.0):
            try:
                self.check()
            except RuntimeError:
                return

    def check(self):
        m = memavail_gib()
        self.min_seen = m if self.min_seen is None else min(self.min_seen, m)
        if self.gib > 0 and m < self.gib and self.tripped is None:
            self.tripped = {"memavail_gib": m, "at": now(), "server_rss_mib": self.srv.rss()[0]}
            print(json.dumps({"memwatch": "MemAvailable below the floor, stopping the server", **self.tripped}), flush=True)
            self.srv.stop()
        if self.tripped is not None:
            raise RuntimeError(f"aborted: MemAvailable {self.tripped['memavail_gib']} GiB < {self.gib} GiB")

    def close(self):
        self._stop.set()


def cmd_load(a):
    corpus, run = Path(a.corpus), Path(a.run)
    run.mkdir(parents=True, exist_ok=False)
    data = run / "data"
    data.mkdir()
    srv = Server(a, data, run)
    rep = {"started": now(), "corpus": str(corpus), "bin": a.bin, "cpus": a.cpus, "cap": a.cap, "shared": a.shared,
           "flush_every": a.flush_every, "XERJ_MAX_PROCESS_MEMORY_MB": os.environ.get("XERJ_MAX_PROCESS_MEMORY_MB"),
           "min_avail_gib": a.min_avail_gib, "loadavg_start": loadavg(), "memavail_gib_start": memavail_gib()}
    rep["boot_s"] = round(srv.start(), 3)
    rep["rss_mib_after_boot"] = srv.rss()[0]
    watch = MemWatch(srv, a.min_avail_gib)
    batches = []
    try:
        t0 = time.time()
        for part, fname in (("phase1", "phase1.jsonl"), ("phase2", "phase2.jsonl")):
            with open(corpus / fname, "rb") as f:
                for i, line in enumerate(f):
                    if not line.strip():
                        continue
                    if a.min_avail_gib > 0:
                        watch.check()
                    meta = json.loads(line)
                    body = {k: meta[k] for k in ("documents", "facts", "edits") if k in meta}
                    tb = time.time()
                    retries = []
                    raw = json.dumps(body, ensure_ascii=False).encode()
                    while True:
                        st, r = call(a.port, "POST", f"/_facts/{a.shared}/_bulk", raw=raw)
                        if st in (429, 503) and len(retries) < 30:
                            if a.min_avail_gib > 0:
                                watch.check()
                            # back-pressure (memtable / RSS breaker): the batch committed nothing; wait and send it again
                            retries.append({"http": st, "at_s": round(time.time() - tb, 1), "rss_mib": srv.rss()[0],
                                            "reason": json.dumps(r)[:200]})
                            print(json.dumps({"retry": retries[-1], "part": part, "i": i}), flush=True)
                            time.sleep(10)
                            continue
                        break
                    dt = time.time() - tb
                    if st != 200:
                        raise RuntimeError(f"_bulk {part} #{i}: {st} {json.dumps(r)[:600]}")
                    b = {"part": part, "kind": meta.get("part"), "i": i, "facts": len(body.get("facts", [])),
                         "documents": len(body.get("documents", [])), "edits": len(body.get("edits", [])),
                         "bytes": len(line), "client_s": round(dt, 3), "committed": r.get("committed"),
                         "records_indexed": r.get("records_indexed"), "epochs": r.get("epochs_written"),
                         "timings_ms": r.get("timings_ms"), "rss_mib": srv.rss()[0], "retries": retries,
                         "loadavg": loadavg()}
                    if a.flush_every and part == "phase1" and (i + 1) % a.flush_every == 0:
                        # keep the memtables small during a large ingest (one engine flush per N batches)
                        tf = time.time()
                        fst, _ = call(a.port, "POST", "/_flush", {})
                        b["flush"] = {"http": fst, "s": round(time.time() - tf, 3), "rss_mib_after": srv.rss()[0]}
                    batches.append(b)
                    print(json.dumps(b), flush=True)
            if part == "phase1":
                rep["shared_B"] = batches[-1]["committed"]["to_rev"]
                rep["phase1_s"] = round(time.time() - t0, 3)
        rep["shared_F"] = batches[-1]["committed"]["to_rev"] if batches[-1]["committed"] else rep["shared_B"]
        rep["shared_load_s"] = round(time.time() - t0, 3)
        rep["rss_mib_after_shared"] = srv.rss()[0]
        # per-world namespaces (the reference layout) for the model sample, in the same engine
        worlds = {}
        t1 = time.time()
        n = 0
        for w in ([] if a.skip_worlds else jl(corpus / "worlds_sample.jsonl")):
            key = f"{w['world_file']}|{w['world_id']}"
            if key in worlds:
                continue
            if a.worlds_max and n >= a.worlds_max:
                break
            ns = f"w-{n:04d}"
            st, r = call(a.port, "POST", f"/_facts/{ns}/_world", w["world"])
            if st != 200:
                raise RuntimeError(f"_world {ns} ({key}): {st} {json.dumps(r)[:400]}")
            worlds[key] = {"ns": ns, "base": r["base_rev"], "final": r["committed"]["to_rev"] if r.get("committed") else r["base_rev"]}
            n += 1
        rep["perworld_namespaces"] = n
        rep["perworld_load_s"] = round(time.time() - t1, 3)
        rep["rss_mib_after_perworld"] = srv.rss()[0]
        st, r = call(a.port, "POST", "/_flush", {})
        rep["flush"] = st
        (run / "worlds_ns.json").write_text(json.dumps(worlds, indent=0) + "\n")
    except Exception as e:
        rep["error"] = str(e)[:2000]
        raise
    finally:
        watch.close()
        rep["memwatch"] = {"floor_gib": a.min_avail_gib, "tripped": watch.tripped, "min_memavail_gib_seen": watch.min_seen}
        rep["aborted"] = watch.tripped is not None
        rep["rss_mib_peak_polled"] = srv.peak
        rep["rss_mib_hwm"] = srv.rss()[1]
        rep["stop_rc"] = srv.stop()
        rep["loadavg_end"] = loadavg()
        rep["ended"] = now()
        with open(run / "load_batches.jsonl", "w") as f:
            for b in batches:
                f.write(json.dumps(b) + "\n")
        (run / "load.json").write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps(rep, indent=1))


def cmd_ask(a):
    corpus, run = Path(a.corpus), Path(a.run)
    load = json.loads((run / "load.json").read_text())
    worlds = json.loads((run / "worlds_ns.json").read_text())
    tag = a.tag or time.strftime("%H%M")
    out = run / f"ask_{tag}"
    out.mkdir(exist_ok=False)
    srv = Server(a, run / "data", out)
    rep = {"started": now(), "precision": a.precision, "par": a.par, "cpus": a.cpus, "cap": a.cap, "loadavg_start": loadavg(),
           "memavail_gib_start": memavail_gib(), "shared": load["shared"], "B": load["shared_B"], "F": load["shared_F"]}
    rep["boot_s"] = round(srv.start(), 3)
    rep["rss_mib_after_boot"] = srv.rss()[0]
    qs = [q for q in jl(corpus / "questions.jsonl") if q["sample"]]
    if a.sets:
        keep = set(a.sets.split(","))
        qs = [q for q in qs if q["set"] in keep]
    if a.limit:
        qs = qs[: a.limit]
    try:
        # first access = rebuild of the projection from the index (timed per namespace)
        t = time.time()
        st, r = call(a.port, "GET", f"/_facts/{load['shared']}")
        rep["shared_rebuild_s"] = round(time.time() - t, 3)
        rep["shared_status"] = r if st == 200 else {"http": st, "body": r}
        rep["rss_mib_after_shared_rebuild"] = srv.rss()[0]
        if "perworld" in a.arms:
            t = time.time()
            for key, w in worlds.items():
                st, r = call(a.port, "GET", f"/_facts/{w['ns']}")
                if st != 200:
                    raise RuntimeError(f"rebuild {w['ns']}: {st} {r}")
            rep["perworld_rebuild_s"] = round(time.time() - t, 3)
            rep["rss_mib_after_perworld_rebuild"] = srv.rss()[0]
        # warm the model once (load time reported separately)
        t = time.time()
        st, r = call(a.port, "POST", f"/_facts/{load['shared']}/_ask", {"question": "What is the capital of France?"})
        rep["model_first_ask_s"] = round(time.time() - t, 3)
        rep["model_first_ask_http"] = st
        rep["rss_mib_after_model_load"] = srv.rss()[0]

        def one(arm, q):
            if arm == "shared":
                ns, rev = load["shared"], (load["shared_B"] if q["phase"] == "before" else load["shared_F"])
            else:
                w = worlds.get(f"{q['world_file']}|{q['world_id']}")
                if w is None:
                    return {"qid": q["qid"], "arm": arm, "skipped": "world not loaded"}
                ns, rev = w["ns"], (w["base"] if q["phase"] == "before" else w["final"])
            body = {"question": q["question"], "revision": rev, "explain": True}
            t0 = time.time()
            st, r = call(a.port, "POST", f"/_facts/{ns}/_ask", body)
            dt = time.time() - t0
            if st != 200:
                return {"qid": q["qid"], "arm": arm, "ns": ns, "revision": rev, "http": st, "error": r, "client_ms": round(dt * 1e3, 1)}
            keep = ("status", "answer", "evidence_ids", "program", "final_status", "final_answer", "verify", "decisions",
                    "fallbacks", "timings_us", "candidate_ids", "candidate_inverse", "hop_reads", "item_id", "engine_error",
                    "published_rev", "trace")
            row = {"qid": q["qid"], "arm": arm, "ns": ns, "revision": rev, "client_ms": round(dt * 1e3, 1)}
            row.update({k: r.get(k) for k in keep})
            row["evidence"] = [{"fact_id": e.get("fact_id"), "subject": e.get("subject"), "relation": e.get("relation"),
                                "value": e.get("value")} for e in r.get("evidence") or []]
            return row

        for arm in a.arms.split(","):
            t = time.time()
            rows = []
            with ThreadPoolExecutor(max_workers=a.par) as ex:
                for row in ex.map(lambda q: one(arm, q), qs):
                    rows.append(row)
                    if len(rows) % 100 == 0:
                        print(f"{arm}: {len(rows)}/{len(qs)} {time.time() - t:.0f}s rss={srv.rss()[0]} load={loadavg()}", flush=True)
            rep[f"{arm}_s"] = round(time.time() - t, 1)
            rep[f"{arm}_questions"] = len(rows)
            rep[f"{arm}_http_errors"] = sum(1 for r in rows if "http" in r)
            rep[f"loadavg_after_{arm}"] = loadavg()
            with open(out / f"asks_{arm}.jsonl", "w") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
    finally:
        rep["rss_mib_peak_polled"] = srv.peak
        rep["rss_mib_hwm"] = srv.rss()[1]
        rep["stop_rc"] = srv.stop()
        rep["loadavg_end"] = loadavg()
        rep["ended"] = now()
        (out / "ask.json").write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps(rep, indent=1))


def cmd_restart(a):
    """restart the server on a loaded data dir: boot, first access of the shared namespace (projection rebuilt from the
    index), RSS; --repeat times; appends to restart.json"""
    run = Path(a.run)
    load = json.loads((run / "load.json").read_text())
    # the published revision a restart must come back with: the load's last revision, or, for a load that stopped
    # part-way, the last batch the server acknowledged (load_batches.jsonl)
    expected = load.get("shared_F")
    if expected is None:
        acked = [json.loads(l) for l in open(run / "load_batches.jsonl") if l.strip()]
        expected = acked[-1]["committed"]["to_rev"] if acked else None
    out = []
    for k in range(a.repeat):
        d = run / f"restart_{a.tag or time.strftime('%H%M')}_{k}"
        d.mkdir(exist_ok=False)
        srv = Server(a, run / "data", d)
        rep = {"started": now(), "repeat": k, "cpus": a.cpus, "cap": a.cap, "loadavg_start": loadavg(),
               "memavail_gib_start": memavail_gib()}
        watch = None
        try:
            rep["boot_s"] = round(srv.start(), 3)
            rep["rss_mib_after_boot"] = srv.rss()[0]
            watch = MemWatch(srv, a.min_avail_gib)
            t = time.time()
            st, r = call(a.port, "GET", f"/_facts/{load['shared']}")
            rep["shared_rebuild_s"] = round(time.time() - t, 3)
            rep["shared_status"] = r if st == 200 else {"http": st, "body": r}
            rep["expected_rev"] = expected
            rep["published_rev"] = r.get("published_rev") if st == 200 else None
            rep["published_rev_ok"] = st == 200 and r.get("published_rev") == expected
            rep["rss_mib_after_rebuild"] = srv.rss()[0]
            # a second access is served from the loaded projection
            t = time.time()
            st2, _ = call(a.port, "GET", f"/_facts/{load['shared']}")
            rep["second_access_ms"] = round((time.time() - t) * 1e3, 1)
            rep["second_access_http"] = st2
        except Exception as e:
            rep["error"] = str(e)[:2000]
        finally:
            if watch:
                watch.close()
                rep["memwatch"] = {"floor_gib": a.min_avail_gib, "tripped": watch.tripped, "min_memavail_gib_seen": watch.min_seen}
            rep["rss_mib_peak_polled"] = srv.peak
            rep["rss_mib_hwm"] = srv.rss()[1]
            t = time.time()
            rep["stop_rc"] = srv.stop()
            rep["stop_s"] = round(time.time() - t, 3)
            rep["loadavg_end"] = loadavg()
            rep["ended"] = now()
            (d / "restart.json").write_text(json.dumps(rep, indent=1) + "\n")
        out.append(rep)
        print(json.dumps({k2: rep.get(k2) for k2 in ("repeat", "boot_s", "shared_rebuild_s", "published_rev_ok",
                                                      "rss_mib_after_rebuild", "rss_mib_peak_polled", "stop_s", "loadavg_start",
                                                      "error")}), flush=True)
        if "error" in rep:
            break
    with open(run / "restart.jsonl", "a") as f:
        for rep in out:
            f.write(json.dumps(rep) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["load", "ask", "restart"])
    ap.add_argument("--bin", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--cpus", required=True)
    ap.add_argument("--cap", default="12G")
    ap.add_argument("--capped", default=os.environ.get("XERJ_CAPPED", "none"),
                    help="wrapper run as <capped> <cap> <cpus> <server...>; 'none' runs the server directly")
    ap.add_argument("--shared", default="e8-shared")
    ap.add_argument("--worlds-max", type=int, default=0)
    ap.add_argument("--flush-every", type=int, default=0, help="load: POST /_flush after every N phase-1 batches (0 = never)")
    ap.add_argument("--arms", default="perworld,shared")
    ap.add_argument("--sets", default="")
    ap.add_argument("--par", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--precision", default="int8-guard")
    ap.add_argument("--tag", default="")
    ap.add_argument("--skip-worlds", action="store_true", help="load: no per-world namespaces")
    ap.add_argument("--min-avail-gib", type=float, default=0.0,
                    help="load / restart: stop the server when the host's MemAvailable falls below this (0 = off)")
    ap.add_argument("--repeat", type=int, default=1, help="restart: number of restarts")
    ap.add_argument("--rss-trace", action="store_true", help="record the server's VmRSS every 0.25 s (<run dir>/rss.tsv)")
    ap.add_argument("--engine-toml", default="", help="a file whose text is appended to the server's engine.toml")
    a = ap.parse_args()
    {"load": cmd_load, "ask": cmd_ask, "restart": cmd_restart}[a.cmd](a)


if __name__ == "__main__":
    main()
