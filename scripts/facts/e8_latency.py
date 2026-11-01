#!/usr/bin/env python3
"""Candidate retrieval latency of exact-path runs (XERJ_E8_REPEAT passes): per pass and arm, p50 / p95 / p99 of the link +
base manifest time (us_link_manifest) and of the gold program on the executor incl. hop manifests (us_oracle_exec), with
the load average at the start / end of the pass. usage: e8_latency.py RUNDIR..."""
import json, sys
from pathlib import Path


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(round(q * (len(v) - 1))))] if v else None


for d in map(Path, sys.argv[1:]):
    summ = json.loads((d / "exact.summary.json").read_text())
    passes = summ.get("exact_passes") or [{"pass": 0, "file": str(d / "exact.jsonl")}]
    info = (d / "run_info.txt").read_text().splitlines()
    print(f"## {d.name}: {info[1][:24] if len(info) > 1 else ''} | build/apply s {summ.get('direct_apply_s') or summ.get('rebuild_s')} | "
          f"questions {summ.get('questions')}")
    for p in passes:
        rows = [json.loads(l) for l in open(p["file"])]
        for arm in ("shared", "shared_cf", "perworld"):
            lm = [r[arm]["us_link_manifest"] / 1000 for r in rows]
            ex = [r[arm]["us_oracle_exec"] / 1000 for r in rows]
            print(f"  pass {p['pass']} {arm:9s} n={len(rows)} link+manifest ms p50/p95/p99 {pct(lm,.5):.2f}/{pct(lm,.95):.2f}/{pct(lm,.99):.2f}"
                  f" | gold program incl. hop manifests ms p50/p95/p99 {pct(ex,.5):.2f}/{pct(ex,.95):.2f}/{pct(ex,.99):.2f}"
                  f" | load {p.get('loadavg_start')} -> {p.get('loadavg_end')} | pass s {round(p.get('s', 0), 1)}")
