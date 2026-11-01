#!/usr/bin/env python3
"""Per-thread CPU and memory sampler for a running process tree (no perf needed).

usage: procsample.py --out PREFIX --interval S --pid NAME=PID [--pid NAME=PID ...] [--progress FILE]

Every S seconds it reads /proc/PID/stat, /proc/PID/status and /proc/PID/task/*/{comm,stat} of each named pid and writes
  PREFIX.timeline.tsv  one row per sample: t, per process cpu_s / rss_mb (VmRSS) / threads / anon_mb (RssAnon +
                       RssShmem, what the memory breaker reads) / file_mb (RssFile), MemFree, hottest zone, client phase
  PREFIX.threads.json  at exit: per process, CPU seconds per thread group (thread name with digits stripped), the
                       process total (utime+stime of the whole process, dead threads included) and the unattributed rest,
                       peak RSS (VmHWM), peak and last RssAnon+RssShmem / RssFile.
It exits when every pid is gone (or on SIGTERM), writing the summary first.
"""
import argparse
import json
import os
import re
import signal
import sys
import time

HZ = os.sysconf("SC_CLK_TCK")


def read(p):
    try:
        with open(p) as f:
            return f.read()
    except OSError:
        return None


def stat_cpu(s):
    # fields after the ')' : state(0) ... utime(11) stime(12)
    rest = s.rsplit(")", 1)[1].split()
    return (int(rest[11]) + int(rest[12])) / HZ


def status_kb(st, key):
    m = re.search(r"^" + key + r":\s+(\d+)", st, re.M)
    return int(m.group(1)) if m else 0


def memfree():
    s = read("/proc/meminfo") or ""
    return status_kb(s, "MemFree") * 1024


def hottest():
    t = 0
    d = "/sys/class/thermal"
    for z in os.listdir(d):
        if z.startswith("thermal_zone"):
            v = read(os.path.join(d, z, "temp"))
            if v:
                t = max(t, int(v))
    return t / 1000


def group_name(comm):
    comm = comm.strip()
    return re.sub(r"[-_ ]?\d+$", "", comm) or comm


def last_phase(path):
    if not path or not os.path.exists(path):
        return ""
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            n = f.tell()
            f.seek(max(0, n - 65536))
            tail = f.read().decode("utf-8", "replace")
    except OSError:
        return ""
    for line in reversed(tail.splitlines()):
        if line.startswith("xerj-progress "):
            kv = dict(x.split("=", 1) for x in line.split()[1:] if "=" in x)
            return f"{kv.get('phase','')}:{kv.get('pct','')}:{kv.get('items','')}"
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--interval", type=float, default=2.0)
    ap.add_argument("--pid", action="append", default=[])
    ap.add_argument("--progress", default="")
    ap.add_argument("--dir", action="append", default=[], help="NAME=PATH: sample the directory's size, keep the peak")
    a = ap.parse_args()
    dirs = dict(spec.split("=", 1) for spec in a.dir)
    dir_peak = {n: 0 for n in dirs}

    def dir_bytes(path):
        total = 0
        for dp, _, fs in os.walk(path):
            for f in fs:
                try:
                    total += os.lstat(os.path.join(dp, f)).st_size
                except OSError:
                    pass
        return total
    procs = {}
    for spec in a.pid:
        name, pid = spec.split("=")
        procs[name] = int(pid)
    # per process: thread tid -> (group, last cpu)
    threads = {n: {} for n in procs}
    total = {n: 0.0 for n in procs}
    hwm = {n: 0 for n in procs}
    anon = {n: 0 for n in procs}
    filerss = {n: 0 for n in procs}
    anon_peak = {n: 0 for n in procs}
    file_peak = {n: 0 for n in procs}
    t0 = time.time()
    tl = open(a.out + ".timeline.tsv", "w")
    tl.write("t\t" + "\t".join(f"{n}_cpu_s\t{n}_rss_mb\t{n}_threads\t{n}_anon_mb\t{n}_file_mb" for n in procs)
             + "\tmemfree_gib\thottest_c\tphase\n")
    stop = {"flag": False}

    def on_term(*_):
        stop["flag"] = True

    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGINT, on_term)

    def summary():
        out = {}
        for n in procs:
            groups = {}
            for tid, (g, c) in threads[n].items():
                groups[g] = groups.get(g, 0.0) + c
            attributed = sum(groups.values())
            out[n] = {
                "pid": procs[n],
                "cpu_s_total": round(total[n], 2),
                "cpu_s_unattributed": round(max(0.0, total[n] - attributed), 2),
                "peak_rss_mb": round(hwm[n] / 1024, 1),
                "last_rss_anon_mb": round(anon[n] / 1024, 1),
                "last_rss_file_mb": round(filerss[n] / 1024, 1),
                "peak_rss_anon_mb": round(anon_peak[n] / 1024, 1),
                "peak_rss_file_mb": round(file_peak[n] / 1024, 1),
                "thread_groups_cpu_s": dict(sorted(((k, round(v, 2)) for k, v in groups.items()), key=lambda x: -x[1])),
            }
        out["wall_s"] = round(time.time() - t0, 1)
        out["dir_peak_bytes"] = dir_peak
        with open(a.out + ".threads.json", "w") as f:
            json.dump(out, f, indent=1)

    while not stop["flag"]:
        alive = False
        row = [f"{time.time() - t0:.1f}"]
        for n, pid in procs.items():
            s = read(f"/proc/{pid}/stat")
            st = read(f"/proc/{pid}/status")
            if s is None or st is None:
                row += ["", "", "", "", ""]
                continue
            alive = True
            try:
                total[n] = stat_cpu(s)
            except (IndexError, ValueError):
                pass
            hwm[n] = max(hwm[n], status_kb(st, "VmHWM"))
            anon[n] = status_kb(st, "RssAnon") + status_kb(st, "RssShmem")
            filerss[n] = status_kb(st, "RssFile")
            anon_peak[n] = max(anon_peak[n], anon[n])
            file_peak[n] = max(file_peak[n], filerss[n])
            nthreads = 0
            try:
                tids = os.listdir(f"/proc/{pid}/task")
            except OSError:
                tids = []
            for tid in tids:
                ts = read(f"/proc/{pid}/task/{tid}/stat")
                comm = read(f"/proc/{pid}/task/{tid}/comm")
                if ts is None or comm is None:
                    continue
                nthreads += 1
                try:
                    threads[n][tid] = (group_name(comm), stat_cpu(ts))
                except (IndexError, ValueError):
                    pass
            row += [f"{total[n]:.1f}", f"{status_kb(st, 'VmRSS') / 1024:.0f}", str(nthreads),
                    f"{anon[n] / 1024:.0f}", f"{filerss[n] / 1024:.0f}"]
        for n, path in dirs.items():
            dir_peak[n] = max(dir_peak[n], dir_bytes(path))
        row += [f"{memfree() / 2**30:.2f}", f"{hottest():.1f}", last_phase(a.progress)]
        tl.write("\t".join(row) + "\n")
        tl.flush()
        if not alive:
            break
        summary()
        time.sleep(a.interval)
    summary()


if __name__ == "__main__":
    main()
