#!/usr/bin/env python3
"""Load / rebuild / memory table from e8 run dirs: e8_ops.py LABEL=RUN_DIR[,ASK_TAG][,EXACT_SUMMARY] ..."""
import json, sys
from pathlib import Path


def get(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return None


def main():
    print("| run | facts | records written | shared load (s) | records/s | engine RSS peak during load (MiB) | rebuild after restart, release server (s) | RSS after rebuild (MiB) | rebuild in-process test (s) | projection heap (MiB) |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for a in sys.argv[1:]:
        lab, rest = a.split("=", 1)
        parts = rest.split(",")
        run = Path(parts[0])
        tag = parts[1] if len(parts) > 1 else "a1"
        ex = get(parts[2]) if len(parts) > 2 else None
        load = get(run / "load.json") or {}
        ask = get(run / f"ask_{tag}" / "ask.json") or {}
        recs = 0
        bl = run / "load_batches.jsonl"
        if bl.exists():
            for l in open(bl):
                b = json.loads(l)
                recs += (b.get("records_indexed") or 0)
        facts = None
        st = (ex or {}).get("status_after_rebuild") or (ex or {}).get("status_after_apply") or ask.get("shared_status") or {}
        if isinstance(st, dict):
            facts = st.get("records")
        heap = st.get("projection_heap_bytes") if isinstance(st, dict) else None
        if isinstance(heap, dict):
            heap = heap.get("total") or sum(v for v in heap.values() if isinstance(v, (int, float)))
        ls = load.get("shared_load_s")
        print(f"| {lab} | {facts if facts is not None else '-'} | {recs or '-'} | {ls if ls is not None else '-'} | "
              f"{round(recs / ls) if recs and ls else '-'} | {load.get('rss_mib_peak_polled', '-')} | {ask.get('shared_rebuild_s', '-')} | "
              f"{ask.get('rss_mib_after_shared_rebuild', '-')} | {(ex or {}).get('rebuild_s', (ex or {}).get('direct_apply_s', '-'))} | "
              f"{round(heap / 2**20, 1) if isinstance(heap, (int, float)) else '-'} |")


if __name__ == "__main__":
    main()
