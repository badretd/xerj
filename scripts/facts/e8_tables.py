#!/usr/bin/env python3
"""Markdown tables from e8_score.py outputs: e8_tables.py LABEL=score.json [LABEL=score.json ...]"""
import json, sys


def f3(x):
    return "-" if x is None else f"{x:.3f}"


def f4(x):
    return "-" if x is None else f"{x:+.4f}"


def main():
    runs = []
    for a in sys.argv[1:]:
        lab, p = a.split("=", 1)
        runs.append((lab, json.load(open(p))))
    print("### Exact path (all questions, no model)\n")
    print("| run | questions | main | changed | per-world oracle correct | shared oracle correct (main) | link+manifest p50 / p95 / p99 us (per world; shared) |")
    print("|---|---:|---:|---:|---:|---:|---|")
    for lab, d in runs:
        t = d["exact_path"]["total"]
        lm = d["exact_path"]["link_manifest_us"]
        main = t.get("cohort main", 0)
        print(f"| {lab} | {t.get('questions')} | {main} | {t.get('cohort changed', 0)} | {t.get('perworld oracle correct')} | "
              f"{t.get('main: shared oracle correct', 0)} ({t.get('main: shared oracle correct', 0) / max(1, main):.3f}) | "
              f"{lm['perworld']['p50']} / {lm['perworld']['p95']} / {lm['perworld']['p99']}; {lm['shared']['p50']} / {lm['shared']['p95']} / {lm['shared']['p99']} |")
    print()
    for lab, d in runs:
        print(f"#### {lab}: exact path by set\n")
        print("| set | questions | main | changed | shared oracle correct in main | gold LOOKUP records in the K=16 manifest, main (per world / shared) |")
        print("|---|---:|---:|---:|---:|---:|")
        for s, v in sorted(d["exact_path"]["by_set"].items()):
            m = v.get("cohort main", 0)
            print(f"| {s} | {v.get('questions')} | {m} | {v.get('cohort changed', 0)} | {v.get('main: shared oracle correct', 0)} | "
                  f"{v.get('main: base-manifest gold lookups found (perworld)', 0)} / {v.get('main: base-manifest gold lookups found (shared)', 0)} of {v.get('main: base-manifest gold lookups', 0)} |")
        print()
    print("### Model (int8 + guard), supported accuracy, shared - per world\n")
    for lab, d in runs:
        m = d.get("model")
        if not m:
            continue
        print(f"#### {lab}\n")
        print("| cohort | n | per world | shared | shared-only / per-world-only | diff | LB95 Newcombe | LB95 cluster | verdict |")
        print("|---|---:|---:|---:|---:|---:|---:|---:|---|")
        for k, v in m["paired"].items():
            r = v.get("raw supported")
            if not r or not r["n"]:
                continue
            print(f"| {k} | {r['n']} | {f3(r['perworld'])} | {f3(r['shared'])} | {r['shared_only']} / {r['perworld_only']} | {f4(r['diff'])} | "
                  f"{f4(r['lb95_newcombe'])} | {f4(r['lb95_cluster'])} | {r['verdict']} |")
        print()
        print("| cohort | raw correct (per world / shared, diff, LB95) | verified supported (per world / shared, diff) | own-world supported (per world / shared, diff) |")
        print("|---|---|---|---|")
        for k, v in m["paired"].items():
            if not k.startswith("main") and k != "all questions":
                continue
            c, vs, ow = v.get("raw correct"), v.get("verified supported"), v.get("raw own-world supported")
            if not c or not c["n"]:
                continue
            lb = min(x for x in (c["lb95_newcombe"], c["lb95_cluster"]) if x is not None)
            print(f"| {k} | {f3(c['perworld'])} / {f3(c['shared'])}, {f4(c['diff'])}, {f4(lb)} | {f3(vs['perworld'])} / {f3(vs['shared'])}, {f4(vs['diff'])} | "
                  f"{f3(ow['perworld']) if ow else '-'} / {f3(ow['shared']) if ow else '-'}, {f4(ow['diff']) if ow else '-'} |")
        print()
        print("visibility:", json.dumps(m["visibility"]))
        print()
        for arm, lat in m["latency"].items():
            print(f"latency {arm}: n={lat['n']} client p50/p95 {lat['client_ms']['p50']}/{lat['client_ms']['p95']} ms; "
                  + "; ".join(f"{k} {lat[k + '_ms']['p50']}/{lat[k + '_ms']['p95']}" for k in ("link", "manifest", "model", "executor", "total"))
                  + f"; guard fallbacks {lat['guard_fallbacks']} of {lat['decisions']} decisions")
        if m.get("merged"):
            print("merged:", json.dumps(m["merged"]))
        print()


if __name__ == "__main__":
    main()
