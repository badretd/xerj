#!/usr/bin/env python3
"""Minimal reader for pprof protobuf profiles (pprof-rs `cpu.pb`, optionally gzipped): no Go toolchain needed.

usage: pprof_top.py PROFILE [--top N] [--folded OUT.folded] [--crate-depth D]
Prints: total samples; top functions by self and by cumulative samples; cumulative share per crate / module prefix
(`xerj_engine::bulk`, `xerj_fts::...`, std / alloc / libc ...); per thread (pprof-rs puts the thread name at the root).
"""
import argparse
import collections
import gzip
import sys


def varint(b, i):
    v = 0
    s = 0
    while True:
        c = b[i]
        i += 1
        v |= (c & 0x7F) << s
        if c < 0x80:
            return v, i
        s += 7


def fields(b):
    i = 0
    n = len(b)
    while i < n:
        key, i = varint(b, i)
        f, wt = key >> 3, key & 7
        if wt == 0:
            v, i = varint(b, i)
            yield f, wt, v
        elif wt == 2:
            ln, i = varint(b, i)
            yield f, wt, b[i:i + ln]
            i += ln
        elif wt == 1:
            yield f, wt, b[i:i + 8]
            i += 8
        elif wt == 5:
            yield f, wt, b[i:i + 4]
            i += 4
        else:
            raise ValueError(f"wire type {wt}")


def packed(v, wt):
    if wt == 0:
        return [v]
    out = []
    i = 0
    while i < len(v):
        x, i = varint(v, i)
        out.append(x)
    return out


def load(path):
    raw = open(path, "rb").read()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    strings, samples, locs, funcs = [], [], {}, {}
    for f, wt, v in fields(raw):
        if f == 6:
            strings.append(v.decode("utf-8", "replace"))
        elif f == 2:
            lids, vals = [], []
            for g, wt2, x in fields(v):
                if g == 1:
                    lids += packed(x, wt2)
                elif g == 2:
                    vals += packed(x, wt2)
            samples.append((lids, vals))
        elif f == 4:
            lid, lines = 0, []
            for g, wt2, x in fields(v):
                if g == 1:
                    lid = x
                elif g == 4:
                    fid = 0
                    for h, _, y in fields(x):
                        if h == 1:
                            fid = y
                    lines.append(fid)
            locs[lid] = lines
        elif f == 5:
            fid, name = 0, 0
            for g, _, x in fields(v):
                if g == 1:
                    fid = x
                elif g == 2:
                    name = x
            funcs[fid] = name
    return strings, samples, locs, funcs


def demangle_short(name):
    # keep rust paths readable: drop hash suffixes like ::h0123456789abcdef
    if "::h" in name and len(name.rsplit("::h", 1)[1]) == 16:
        name = name.rsplit("::h", 1)[0]
    return name


def module_of(name, depth):
    n = name
    if n.startswith("<"):
        # <T as Trait>::method -> T
        n = n[1:].split(" as ")[0].split(">")[0]
    parts = n.split("::")
    if len(parts) <= 1:
        return n[:60]
    return "::".join(parts[:depth])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("profile")
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--folded", default="")
    ap.add_argument("--crate-depth", type=int, default=2)
    ap.add_argument("--components", default="",
                    help="TSV file: name<TAB>regex; a sample goes to the deepest frame that matches a rule, except that "
                         "rules named ^name are context rules: checked first, in file order, against every frame")
    a = ap.parse_args()
    import re
    rules = []
    context = []  # (name, regex): a frame anywhere in the stack decides (e.g. work queued onto the merge pool, whose
    #               stack starts at the pool's spawn closure and never shows merge_pass_locked)
    if a.components:
        for line in open(a.components):
            if line.strip() and not line.startswith("#"):
                name, rx = line.rstrip("\n").split("\t", 1)
                if name.startswith("^"):
                    context.append((name[1:], re.compile(rx)))
                else:
                    rules.append((name, re.compile(rx)))
    strings, samples, locs, funcs = load(a.profile)
    total = 0
    self_c = collections.Counter()
    cum_c = collections.Counter()
    mod_c = collections.Counter()
    thread_c = collections.Counter()
    folded = collections.Counter()
    comp_c = collections.Counter()
    for lids, vals in samples:
        w = vals[0] if vals else 1
        total += w
        stack = []  # leaf first
        for lid in lids:
            for fid in locs.get(lid, []):
                stack.append(demangle_short(strings[funcs.get(fid, 0)] if funcs.get(fid, 0) < len(strings) else "?"))
        if not stack:
            continue
        self_c[stack[0]] += w
        for fn in set(stack):
            cum_c[fn] += w
        for m in set(module_of(fn, a.crate_depth) for fn in stack):
            mod_c[m] += w
        thread_c[stack[-1]] += w
        folded[";".join(reversed(stack))] += w
        if rules or context:
            comp = next((name for name, rx in context if any(rx.search(fn) for fn in stack)), None)
            if comp is None:
                comp = "other"
                for fn in stack:  # leaf first: the deepest matching frame decides
                    hit = next((name for name, rx in rules if rx.search(fn)), None)
                    if hit:
                        comp = hit
                        break
            comp_c[comp] += w
    print(f"total samples: {total}")
    print("\n== threads (root frame) ==")
    for k, v in thread_c.most_common(25):
        print(f"{100 * v / total:6.1f}%  {v:8d}  {k}")
    if rules or context:
        print("\n== components (context rules, then the deepest matching frame) ==")
        for k, v in comp_c.most_common():
            print(f"{100 * v / total:6.1f}%  {v:8d}  {k}")
    print("\n== modules (cumulative, depth %d) ==" % a.crate_depth)
    for k, v in mod_c.most_common(a.top):
        print(f"{100 * v / total:6.1f}%  {v:8d}  {k}")
    print("\n== functions by self ==")
    for k, v in self_c.most_common(a.top):
        print(f"{100 * v / total:6.1f}%  {v:8d}  {k[:160]}")
    print("\n== functions by cumulative ==")
    for k, v in cum_c.most_common(a.top):
        print(f"{100 * v / total:6.1f}%  {v:8d}  {k[:160]}")
    if a.folded:
        with open(a.folded, "w") as f:
            for k, v in folded.most_common():
                f.write(f"{k} {v}\n")


if __name__ == "__main__":
    main()
