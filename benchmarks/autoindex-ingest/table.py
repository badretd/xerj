#!/usr/bin/env python3
"""Markdown table of run summaries: table.py runs/A/summary.json runs/B/summary.json ...

Columns (NEWCORE_MAP.md 7.5): exit code, records the client reported / records `_count` returns, time to searchable
(active = wall minus guard pauses, and wall), server CPU seconds during the client run and including the background
work it left (merges, until the server's CPU went flat or IDLE_CAP_S), server peak RSS split into anonymous
(RssAnon+RssShmem, what the breaker reads) and file-backed (RssFile), bytes on disk, memory-breaker events.
Older summaries without a field print "–"."""
import json
import sys


def mb(v):
    return "–" if v is None else f"{v / 1e6:,.0f}"


def num(v, nd=0):
    if v is None:
        return "–"
    return f"{v:,.{nd}f}"


def searchable(s):
    if s.get("searchable_active_s") is None:
        return "–"
    return f"{num(s['searchable_active_s'])} / {num(s['searchable_wall_s'])}"


def records(s):
    rep = s.get("records_reported")
    cnt = s.get("t_count")
    return f"{num(rep) if rep is not None else '–'} / {num(cnt) if cnt is not None else '–'}"


def rss(s):
    a, f = s.get("server_peak_anon_mb"), s.get("server_peak_file_mb")
    if a is None:
        return f"VmHWM {num(s.get('server_peak_rss_mb'))}"
    return f"{num(a)} / {num(f)}"


def rss_idle(s):
    a, f = s.get("server_peak_anon_until_idle_mb"), s.get("server_peak_file_until_idle_mb")
    if a is None:
        return f"VmHWM {num(s.get('server_peak_rss_until_idle_mb'))}"
    return f"{num(a)} / {num(f)}"


def breaker(s):
    if s.get("breaker_engaged_run") is None:
        return "–"
    return (f"{s['breaker_engaged_run']} / {s['breaker_engaged_after']} "
            f"({s.get('client_shed_lines', 0)} shed, {s.get('client_pushback_lines', 0)} push-back)")


cols = [
    ("run", lambda s: s["label"]),
    ("exit", lambda s: s["autoindex_rc"]),
    ("records reported / `_count`", records),
    ("searchable s active / wall", searchable),
    ("server CPU-s run / until idle", lambda s: f"{num(s['server_cpu_s'])} / {num(s.get('server_cpu_until_idle_s'))}"),
    ("server peak MB anon / file (run)", rss),
    ("… until idle", rss_idle),
    ("disk MB end / peak", lambda s: f"{mb(s['server_data_bytes'])} / {mb(s.get('server_data_peak_bytes'))}"),
    ("breaker engaged run / after (client)", breaker),
]
rows = [json.load(open(p)) for p in sys.argv[1:]]
print("| " + " | ".join(c for c, _ in cols) + " |")
print("|" + "|".join("---" if i == 0 else "---:" for i in range(len(cols))) + "|")
for r in rows:
    vals = []
    for _, f in cols:
        try:
            vals.append(str(f(r)))
        except Exception:
            vals.append("–")
    print("| " + " | ".join(vals) + " |")
for r in rows:
    stop = r.get("stopped")
    extra = f"; stopped by the harness: {stop['reason']}" if stop else ""
    print(f"- {r['label']}: `{r.get('xerj_done')}`{extra}")
