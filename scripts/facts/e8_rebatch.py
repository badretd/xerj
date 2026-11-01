#!/usr/bin/env python3
"""Split the phase-1 `_bulk` bodies of an E8 corpus into smaller batches (same ops, same order of facts).

The product refuses a batch of more than `limits.max_actions_per_bulk` entries (documents + facts + edits; 50,000 by
default); e8_corpus.py wrote the 10M corpus in batches of 50,000 facts plus their documents. Each fact keeps its place in
the fact sequence; each document goes into the batch of its first fact (a document is always written before the facts that
cite it); documents no fact of the line cites go into the line's last batch. The store revision is the number of ops, so
revisions B and F, the visible content at both and therefore every snapshot fingerprint stay the same.
Other corpus files are symlinked.

usage: e8_rebatch.py --corpus corpus_10m --out corpus_10m_b20k [--batch-facts 20000]
"""
import argparse, hashlib, json, os
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-facts", type=int, default=20000)
    a = ap.parse_args()
    src, out = Path(a.corpus).resolve(), Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    for f in ("phase2.jsonl", "questions.jsonl", "world_facts.jsonl", "worlds_sample.jsonl", "corpus.json"):
        if (src / f).exists():
            os.symlink(src / f, out / f)
    n_in = n_out = n_f = n_d = 0
    max_entries = 0
    hf_in, hf_out = hashlib.sha256(), hashlib.sha256()
    with open(src / "phase1.jsonl", encoding="utf-8") as fi, open(out / "phase1.jsonl", "w", encoding="utf-8") as fo:
        for line in fi:
            if not line.strip():
                continue
            b = json.loads(line)
            n_in += 1
            docs = {(d["doc_id"], d["version"]): d for d in b.get("documents", [])}
            facts = b.get("facts", [])
            for f in facts:
                hf_in.update(f["fact_id"].encode() + b"\n")
            used, chunks = set(), []
            for i in range(0, max(1, len(facts)), a.batch_facts):
                ch = facts[i:i + a.batch_facts]
                cd = []
                for f in ch:
                    k = (f["doc_id"], f["version"])
                    if k in docs and k not in used:
                        used.add(k)
                        cd.append(docs[k])
                chunks.append({"part": b.get("part"), "documents": cd, "facts": ch})
            chunks[-1]["documents"].extend(d for k, d in docs.items() if k not in used)
            for c in chunks:
                for f in c["facts"]:
                    hf_out.update(f["fact_id"].encode() + b"\n")
                n_f += len(c["facts"])
                n_d += len(c["documents"])
                max_entries = max(max_entries, len(c["facts"]) + len(c["documents"]))
                fo.write(json.dumps(c, ensure_ascii=False) + "\n")
                n_out += 1
    rep = {"source": str(src), "batch_facts": a.batch_facts, "lines_in": n_in, "batches_out": n_out, "facts": n_f,
           "documents": n_d, "max_entries_per_batch": max_entries,
           "fact_sequence_identical": hf_in.hexdigest() == hf_out.hexdigest()}
    (out / "rebatch.json").write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
