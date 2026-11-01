#!/usr/bin/env python3
"""Stop a measured server before it takes memory the host needs (shared machine: a GPU job lives on the same RAM).

usage: rsswatch.py --pid PID --out FILE [--anon-mb 8192] [--memfree-gib 10] [--memfree-min-anon-mb 2048] [--interval 1]

Every interval it reads /proc/PID/status and /proc/meminfo. It stops the server (SIGTERM, SIGKILL after 30 s) when
  * RssAnon + RssShmem of the server (what the product's memory breaker reads) exceeds --anon-mb, or
  * MemFree falls below --memfree-gib while the server holds at least --memfree-min-anon-mb of anonymous memory.
Before signalling it writes FILE (JSON: reason, epoch, anon_mb, file_mb, memfree_gib). It exits when the pid is gone.
"""
import argparse
import json
import os
import re
import signal
import time


def kb(text, key):
    m = re.search(r"^" + key + r":\s+(\d+)", text, re.M)
    return int(m.group(1)) if m else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--anon-mb", type=float, default=8192)
    ap.add_argument("--memfree-gib", type=float, default=10)
    ap.add_argument("--memfree-min-anon-mb", type=float, default=2048)
    ap.add_argument("--interval", type=float, default=1.0)
    a = ap.parse_args()
    while True:
        try:
            st = open(f"/proc/{a.pid}/status").read()
        except OSError:
            return
        if "State:\tZ" in st:
            return
        anon = (kb(st, "RssAnon") + kb(st, "RssShmem")) / 1024
        filemb = kb(st, "RssFile") / 1024
        memfree = kb(open("/proc/meminfo").read(), "MemFree") / 2**20
        reason = None
        if anon > a.anon_mb:
            reason = f"server RssAnon+RssShmem {anon:.0f} MB > {a.anon_mb:.0f} MB"
        elif memfree < a.memfree_gib and anon >= a.memfree_min_anon_mb:
            reason = f"MemFree {memfree:.1f} GiB < {a.memfree_gib} GiB with the server holding {anon:.0f} MB anonymous"
        if reason:
            with open(a.out, "w") as f:
                json.dump({"reason": reason, "epoch": time.time(), "anon_mb": round(anon, 1),
                           "file_mb": round(filemb, 1), "memfree_gib": round(memfree, 2)}, f)
            try:
                os.kill(a.pid, signal.SIGCONT)
                os.kill(a.pid, signal.SIGTERM)
            except OSError:
                return
            for _ in range(60):
                time.sleep(0.5)
                if not os.path.exists(f"/proc/{a.pid}"):
                    return
            try:
                os.kill(a.pid, signal.SIGKILL)
            except OSError:
                pass
            return
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
