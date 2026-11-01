"""Shared helpers of the facts end-to-end tests: a real xerj server process and a small HTTP client. Standard library only."""
import hashlib, http.client, json, os, signal, subprocess, sys, time, urllib.parse

VERIFY = "decline_v2"
OUTCOME_KEYS = ("status", "answer", "final_status", "final_answer", "final_evidence_ids", "reason_code", "program", "evidence_ids")


def fp(fact_id):
    """a fact id as one path segment: bench ids hold slashes"""
    return urllib.parse.quote(fact_id, safe="")


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S %Z")


def ns_of(world_key):
    return "w" + hashlib.sha1(world_key.encode()).hexdigest()[:16]


class Client:
    """one keep-alive connection per thread; a dead connection raises OSError / http.client exceptions to the caller"""

    def __init__(self, port):
        self.port = port
        self.c = None

    def call(self, method, path, body=None, timeout=600):
        if self.c is None:
            self.c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=timeout)
        try:
            self.c.request(method, path, None if body is None else json.dumps(body), {"content-type": "application/json"})
            r = self.c.getresponse()
            raw = r.read()
        except Exception:
            self.c.close()
            self.c = None
            raise
        try:
            return r.status, json.loads(raw or b"null")
        except ValueError:
            return r.status, raw.decode("utf-8", "replace")


class Server:
    """the release binary as its own process; data dir survives stop, kill and start"""

    def __init__(self, bin, model, data_dir, port, cpus, threads=1, precision=None, env=None, nice=15):
        self.bin, self.model, self.data, self.port, self.cpus = bin, model, data_dir, port, cpus
        self.threads, self.precision, self.extra_env, self.nice = threads, precision, env or {}, nice
        self.p = None
        os.makedirs(data_dir, exist_ok=True)
        self.conf = os.path.join(data_dir, "..", os.path.basename(data_dir.rstrip("/")) + ".toml")
        with open(self.conf, "w") as f:
            f.write(f"[server]\nrest_port = {port - 1}\nes_compat_port = {port}\ngrpc_port = {port + 1}\n")
        self.logpath = data_dir.rstrip("/") + ".engine.log"

    def start(self, ready_timeout=1800):
        env = dict(os.environ, XERJ_FACTS_MODEL_DIR=self.model, XERJ_FACTS_PRELOAD="1", XERJ_FACTS_THREADS=str(self.threads), **self.extra_env)
        env.pop("XERJ_FACTS_PRECISION", None)
        if self.precision:
            env["XERJ_FACTS_PRECISION"] = self.precision
        log = open(self.logpath, "ab")
        self.p = subprocess.Popen(["nice", "-n", str(self.nice), "taskset", "-c", self.cpus, self.bin, "--config", self.conf, "--data-dir", self.data,
                                   "--insecure", "--no-auto-port"], stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        log.close()
        t0 = time.time()
        c = Client(self.port)
        while True:
            if self.p.poll() is not None:
                raise RuntimeError(f"server exited rc={self.p.returncode}, see {self.logpath}")
            try:
                if c.call("GET", "/health/ready", timeout=5)[0] == 200:
                    return time.time() - t0
            except OSError:
                pass
            except http.client.HTTPException:
                pass
            if time.time() - t0 > ready_timeout:
                raise RuntimeError("server not ready")
            time.sleep(0.25)

    def real_pid(self):
        """nice and taskset exec the server, so the pid is the server's own"""
        return self.p.pid

    def stop(self, timeout=300):
        self.p.terminate()
        try:
            self.p.wait(timeout)
        except subprocess.TimeoutExpired:
            self.p.kill()
            self.p.wait()
        return self.p.returncode

    def kill9(self):
        os.kill(self.real_pid(), signal.SIGKILL)
        self.p.wait()

    def rss(self):
        v = {}
        for l in open(f"/proc/{self.real_pid()}/status"):
            if l.startswith(("VmRSS", "VmHWM", "RssAnon", "RssFile")):
                v[l.split(":")[0]] = round(int(l.split()[1]) / 1024, 1)
        return v


class Report:
    """one result file per test, written after every check, so a dead run leaves what it had"""

    def __init__(self, name, out):
        self.name, self.path = name, os.path.join(out, f"{name}.json")
        self.r = {"test": name, "start": now(), "checks": [], "pass": None, "facts": {}}

    def check(self, what, ok, detail=None):
        self.r["checks"].append({"what": what, "ok": bool(ok), "detail": detail})
        if not ok:
            print(f"[{self.name}] FAIL {what}: {str(detail)[:400]}", flush=True)
        self.save()
        return ok

    def fact(self, k, v):
        self.r["facts"][k] = v
        self.save()

    def save(self):
        with open(self.path + ".tmp", "w") as f:
            json.dump(self.r, f, indent=1)
        os.replace(self.path + ".tmp", self.path)

    def end(self):
        self.r["pass"] = bool(self.r["checks"]) and all(c["ok"] for c in self.r["checks"])
        self.r["end"] = now()
        self.save()
        print(f"[{self.name}] {'PASS' if self.r['pass'] else 'FAIL'} {sum(c['ok'] for c in self.r['checks'])}/{len(self.r['checks'])}", flush=True)
        return self.r["pass"]


def load_worlds(path, keep=None):
    out = {}
    for l in open(path, encoding="utf-8"):
        if l.strip():
            w = json.loads(l)
            iid = w["items"][0]["item_id"]
            if keep is None or iid in keep:
                out[iid] = w
    return out


def bulk_body(w):
    return {"documents": w["base_documents"], "facts": w["base_facts"], "edits": w.get("edits", [])}


def ask(c, ns, question, item_id=None, revision=None, verify=VERIFY, timeout=600):
    b = {"question": question}
    if item_id:
        b["item_id"] = item_id
    if revision is not None:
        b["revision"] = revision
    if verify:
        b["verify"] = verify
    return c.call("POST", f"/_facts/{ns}/_ask", b, timeout=timeout)


def outcome(o):
    return {k: o.get(k) for k in OUTCOME_KEYS}
