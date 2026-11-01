#!/usr/bin/env python3
"""The round-2 gate table: one row per store size x setting x sample from the score files that exist.
usage: e8_gate_table.py > gate_table.md (run in e8/)"""
import json, os
ROWS = [  # (size, setting, sample, score file)
    ("10K", "default", "development", "scores/b0_10k_dev/score_exact.json"),
    ("10K", "casefold v2", "development", "scores/b0_10k_dev/score_casefold.json"),
    ("10K", "S2", "development", "scores/b1_10k_dev/score_v3ra.json"),
    ("10K", "default", "round-1 confirmation", "scores/b0_10k_conf1/score_exact.json"),
    ("10K", "casefold v2", "round-1 confirmation", "scores/b0_10k_conf1/score_casefold.json"),
    ("10K", "S2", "round-1 confirmation", "scores/f_conf1_10k/score_s2.json"),
    ("10K", "default", "new confirmation", "scores/f_conf2_10k/score_exact.json"),
    ("10K", "casefold v2", "new confirmation", "scores/f_conf2_10k/score_cf2.json"),
    ("10K", "S2", "new confirmation", "scores/f_conf2_10k/score_s2.json"),
    ("1M", "default", "development", "scores/b0_1m_dev/score_exact.json"),
    ("1M", "casefold v2", "development", "scores/b0_1m_dev/score_casefold.json"),
    ("1M", "S2", "development", "scores/b2_1m_dev/score_s2.json"),
    ("1M", "default", "round-1 confirmation", "scores/b0_1m_conf1/score_exact.json"),
    ("1M", "casefold v2", "round-1 confirmation", "scores/b0_1m_conf1/score_casefold.json"),
    ("1M", "S2", "round-1 confirmation", "scores/f_conf1_1m/score_s2.json"),
    ("1M", "default", "new confirmation", "scores/f_conf2_1m/score_exact.json"),
    ("1M", "casefold v2", "new confirmation", "scores/f_conf2_1m/score_cf2.json"),
    ("1M", "S2", "new confirmation", "scores/f_conf2_1m/score_s2.json"),
    ("10M", "default", "development", "scores/f_dev_10m/score_exact.json"),
    ("10M", "casefold v2", "development", "scores/f_dev_10m/score_cf2.json"),
    ("10M", "S2", "development", "scores/f_dev_10m/score_s2.json"),
    ("10M", "default", "round-1 confirmation", "scores/f_conf1_10m/score_exact.json"),
    ("10M", "casefold v2", "round-1 confirmation", "scores/f_conf1_10m/score_cf2.json"),
    ("10M", "S2", "round-1 confirmation", "scores/f_conf1_10m/score_s2.json"),
    ("10M", "default", "new confirmation", "scores/f_conf2_10m/score_exact.json"),
    ("10M", "casefold v2", "new confirmation", "scores/f_conf2_10m/score_cf2.json"),
    ("10M", "S2", "new confirmation", "scores/f_conf2_10m/score_s2.json"),
]
print("| store | setting | sample | n (main) | per world | shared | shared-only / per-world-only | diff (pts) | lower bound (pts) | verdict | V1+V2 | V3 main | changed |")
print("|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|")
out = []
for size, setting, sample, p in ROWS:
    if not os.path.exists(p):
        print(f"| {size} | {setting} | {sample} | - | - | - | - | - | - | not run | - | - | - |")
        continue
    d = json.load(open(p))
    r = d["model"]["paired"]["main"]["raw supported"]
    lbs = [x for x in (r["lb95_newcombe"], r["lb95_cluster"]) if x is not None]
    lb = min(lbs) if lbs else None
    v = d["model"]["visibility"]
    t = d["exact_path"]["total"]
    print(f"| {size} | {setting} | {sample} | {r['n']} | {r['perworld']:.4f} | {r['shared']:.4f} | {r['shared_only']} / {r['perworld_only']} | "
          f"{100*r['diff']:+.2f} | {100*lb:+.2f} | {r['verdict']} | {v['V1+V2 total']} | {v['V3 in the main cohort']} | {t.get('cohort changed', 0)} |")
    out.append({"size": size, "setting": setting, "sample": sample, "n": r["n"], "perworld": r["perworld"], "shared": r["shared"],
                "shared_only": r["shared_only"], "perworld_only": r["perworld_only"], "diff": r["diff"], "lb": lb,
                "verdict": r["verdict"], "V1V2": v["V1+V2 total"], "V3_main": v["V3 in the main cohort"]})
json.dump(out, open("gate_table.json", "w"), indent=1)
