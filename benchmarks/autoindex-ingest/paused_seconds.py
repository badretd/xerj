#!/usr/bin/env python3
"""Seconds the guard (guardd.py) kept registered groups stopped between two epoch times: sum of PAUSE->RESUME
intervals in guard.log clipped to [t0, t1].   usage: paused_seconds.py GUARD_LOG T0 T1"""
import sys, time
log, t0, t1 = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
paused_at = None
total = 0.0
n = 0
for line in open(log, errors="replace"):
    parts = line.split()
    if len(parts) < 3:
        continue
    try:
        t = time.mktime(time.strptime(parts[0] + " " + parts[1], "%Y-%m-%d %H:%M:%S"))
    except ValueError:
        continue
    if parts[2] == "PAUSE" and paused_at is None:
        paused_at = t
    elif parts[2] == "RESUME" and paused_at is not None:
        a, b = max(paused_at, t0), min(t, t1)
        if b > a:
            total += b - a
            n += 1
        paused_at = None
if paused_at is not None and t1 > max(paused_at, t0):
    total += t1 - max(paused_at, t0)
    n += 1
print(f"{total:.1f} {n}")
