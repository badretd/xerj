//! XERJ connected memory as a document store + BM25 search engine, same protocol as bench_baselines/tantivy_bench and
//! LuceneBench: load the first N items of a corpus (JSONL {id, text}; loading is not timed), ingest them as source documents
//! (store + BM25 over documents), measure time until searchable, CPU time, peak RSS, heap bytes per component and the size of a
//! checkpoint on disk; then query latency (top-16, one thread, warm), throughput with T reader threads, a ranked top-64 file for
//! recall, the time until ONE newly written document is found by search (in-process), and the latency of a durable write
//! (op-log append + fdatasync) on a persistent memory.
//! usage: corpus_bench --corpus F --n N --queries F --ranked F.tsv --out F.json [--threads T] [--async 0|1] [--inc-ops 20]
//!                     [--dir D] [--checkpoint 0|1]
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use std::time::Instant;

use serde::Deserialize;
use serde_json::json;
use xerj_index::memory::{Memory, MemoryConfig};

#[derive(Deserialize)]
struct Item {
    id: String,
    text: String,
}

#[derive(Deserialize)]
struct Q {
    qid: String,
    question: String,
    max_gold_pos: usize,
}

fn arg(a: &HashMap<String, String>, k: &str, d: &str) -> String {
    a.get(k).cloned().unwrap_or_else(|| d.to_string())
}

fn proc_kb(field: &str) -> u64 {
    let s = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    s.lines()
        .find(|l| l.starts_with(field))
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(0)
}

fn cpu_seconds() -> f64 {
    let s = std::fs::read_to_string("/proc/self/stat").unwrap_or_default();
    let rest = s.rsplit(')').next().unwrap_or("");
    let f: Vec<&str> = rest.split_whitespace().collect();
    let ticks = |i: usize| f.get(i).and_then(|v| v.parse::<f64>().ok()).unwrap_or(0.0);
    (ticks(11) + ticks(12)) / 100.0
}

fn dir_bytes(p: &Path) -> u64 {
    let mut n = 0;
    if let Ok(rd) = std::fs::read_dir(p) {
        for e in rd.flatten() {
            let m = e.metadata().unwrap();
            n += if m.is_dir() {
                dir_bytes(&e.path())
            } else {
                m.len()
            };
        }
    }
    n
}

fn stats(mut v: Vec<f64>) -> serde_json::Value {
    if v.is_empty() {
        return json!(null);
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let q = |f: f64| v[((v.len() as f64 * f) as usize).min(v.len() - 1)] * 1000.0;
    json!({"n": v.len(), "mean_ms": v.iter().sum::<f64>() / v.len() as f64 * 1000.0, "p50_ms": q(0.5), "p95_ms": q(0.95),
           "p99_ms": q(0.99), "max_ms": v[v.len() - 1] * 1000.0})
}

fn ids(m: &Memory, q: &str, k: usize) -> Vec<String> {
    m.search_documents(q, k)
        .iter()
        .map(|h| m.store.doc_key_str(h.doc).0.to_string())
        .collect()
}

fn probe_words(text: &str) -> String {
    text.split_whitespace()
        .filter(|w| w.len() > 3)
        .take(12)
        .collect::<Vec<_>>()
        .join(" ")
}

fn main() {
    let raw: Vec<String> = std::env::args().collect();
    let mut a = HashMap::new();
    let mut i = 1;
    while i + 1 < raw.len() {
        a.insert(
            raw[i].trim_start_matches("--").to_string(),
            raw[i + 1].clone(),
        );
        i += 2;
    }
    let n: usize = arg(&a, "n", "1000").parse().unwrap();
    let threads: usize = arg(&a, "threads", "1").parse().unwrap();
    let async_bm25 = arg(&a, "async", "0") == "1";
    let inc_ops: usize = arg(&a, "inc-ops", "20").parse().unwrap();
    let out = arg(&a, "out", "xerj.json");
    let dir = arg(&a, "dir", "/tmp/xerj_corpus_bench");
    let do_ckpt = arg(&a, "checkpoint", "1") == "1";
    // --durable 1: ingest through a persistent memory (op log, one fdatasync at the end), skip checkpoint and queries, then
    // measure ONE write at a time at full scale with a unique-token probe (the update semantics of tantivy_bench / LuceneBench):
    // in-process (write -> search finds the token) and durable (write -> commit with fdatasync -> search finds the token)
    let durable_mode = arg(&a, "durable", "0") == "1";

    // ---- load corpus (not timed as indexing)
    let t_load = Instant::now();
    let f = BufReader::with_capacity(1 << 22, File::open(arg(&a, "corpus", "")).unwrap());
    let mut items: Vec<Item> = Vec::with_capacity(n + inc_ops);
    let mut text_bytes: u64 = 0;
    for line in f.lines() {
        if items.len() >= n + inc_ops {
            break;
        }
        let it: Item = serde_json::from_str(&line.unwrap()).unwrap();
        if items.len() < n {
            text_bytes += it.text.len() as u64;
        }
        items.push(it);
    }
    let extra: Vec<Item> = items.split_off(n.min(items.len()));
    let load_s = t_load.elapsed().as_secs_f64();
    let rss_loaded = proc_kb("VmRSS:") * 1024;

    // ---- ingest: store + BM25 over documents (no fact records in this corpus)
    let cfg = MemoryConfig {
        bm25_records: false,
        bm25_documents: true,
        async_bm25,
        ..MemoryConfig::default()
    };
    if durable_mode {
        let ddir = Path::new(&dir).join("durable_full");
        let _ = std::fs::remove_dir_all(&ddir);
        let mut m = Memory::open(&ddir, cfg.clone(), true).unwrap();
        let cpu0 = cpu_seconds();
        let t0 = Instant::now();
        for it in &items {
            m.add_document(&it.id, "1", &it.text).unwrap();
        }
        let add_s = t0.elapsed().as_secs_f64();
        m.commit().unwrap(); // one group commit + fdatasync for the whole bulk load
        let durable_s = t0.elapsed().as_secs_f64();
        let cpu = cpu_seconds() - cpu0;
        let log_bytes = dir_bytes(&ddir);
        let stride = (items.len() / (2 * inc_ops + 1)).max(1);
        let (mut vis, mut vok, mut dur, mut dok) = (Vec::new(), 0, Vec::new(), 0);
        for k in 0..inc_ops {
            let it = &items[(2 * k + 1) * stride % items.len()];
            let tok = format!("xupdtoken{}", k);
            let t = Instant::now();
            m.add_document(
                &format!("{}#upd{}", it.id, k),
                "1",
                &format!("{} {}", it.text, tok),
            )
            .unwrap();
            let ok = ids(&m, &tok, 1)
                .first()
                .map(|x| x.ends_with(&format!("#upd{}", k)))
                .unwrap_or(false);
            vis.push(t.elapsed().as_secs_f64());
            vok += ok as usize;
            let it2 = &items[(2 * k + 2) * stride % items.len()];
            let tok2 = format!("xdurtoken{}", k);
            let t = Instant::now();
            m.add_document(
                &format!("{}#dur{}", it2.id, k),
                "1",
                &format!("{} {}", it2.text, tok2),
            )
            .unwrap();
            m.commit().unwrap();
            let ok2 = ids(&m, &tok2, 1)
                .first()
                .map(|x| x.ends_with(&format!("#dur{}", k)))
                .unwrap_or(false);
            dur.push(t.elapsed().as_secs_f64());
            dok += ok2 as usize;
        }
        let res = json!({"system": "xerj-mem", "mode": "durable bulk load + unique-token writes at full scale", "n": items.len(),
            "text_bytes": text_bytes, "durable_bulk": {"add_s": add_s, "searchable_and_durable_s": durable_s, "cpu_s": cpu,
                "mb_per_s": text_bytes as f64 / 1e6 / durable_s, "op_log_bytes": log_bytes},
            "write_visible_in_process": {"stats": stats(vis), "found": vok, "of": inc_ops,
                "semantics": "add_document (op log appended, no fsync) -> BM25 search for a unique token finds it"},
            "write_visible_durable": {"stats": stats(dur), "found": dok, "of": inc_ops,
                "semantics": "add_document -> commit (fdatasync) -> BM25 search for a unique token finds it"},
            "peak_rss_bytes": proc_kb("VmHWM:") * 1024});
        std::fs::write(&out, serde_json::to_string_pretty(&res).unwrap()).unwrap();
        println!("{}", serde_json::to_string(&res).unwrap());
        drop(m);
        let _ = std::fs::remove_dir_all(&ddir);
        return;
    }
    let mut m = Memory::new(cfg.clone());
    let cpu0 = cpu_seconds();
    let t0 = Instant::now();
    for it in &items {
        m.add_document(&it.id, "1", &it.text).unwrap();
    }
    let add_s = t0.elapsed().as_secs_f64();
    let first = ids(&m, &items[0].text.chars().take(200).collect::<String>(), 1); // read barrier (async) + proves visibility
    let searchable_s = t0.elapsed().as_secs_f64();
    m.commit().unwrap();
    let commit_s = t0.elapsed().as_secs_f64();
    let build_cpu_s = cpu_seconds() - cpu0;
    let rss_peak_build = proc_kb("VmHWM:") * 1024;
    let hb = m.heap_bytes();
    let mut result = json!({
        "system": "xerj-mem", "version": env!("CARGO_PKG_VERSION"), "n": items.len(), "text_bytes": text_bytes,
        "mode": if async_bm25 { "async BM25 (writer thread + 1 indexer thread)" } else { "inline (1 thread)" },
        "knobs": {"bm25": {"k1": cfg.bm25.k1, "b": cfg.bm25.b}, "tokenizer": "reference LexicalIndex tokens (Python-exact \\w, casefold)",
                  "stored": "document text kept in the store (tantivy / Lucene runs store only ids)"},
        "load_s": load_s, "rss_after_loading_corpus_bytes": rss_loaded,
        "build": {"add_s": add_s, "searchable_s": searchable_s, "commit_s": commit_s, "cpu_s": build_cpu_s,
                  "docs_per_s": items.len() as f64 / searchable_s, "mb_per_s": text_bytes as f64 / 1e6 / searchable_s,
                  "peak_rss_bytes": rss_peak_build, "first_doc_found": first.first().map(|s| s == &items[0].id).unwrap_or(false),
                  "heap": {"store": hb.store, "bm25_documents": hb.bm25_documents, "subject_index": hb.subject_index, "total": hb.total()}},
    });
    eprintln!(
        "[corpus_bench] ingested {} docs ({:.2} GB text) in {:.1} s",
        items.len(),
        text_bytes as f64 / 1e9,
        searchable_s
    );

    // ---- checkpoint = the on-disk footprint (store columns incl. text + BM25 segments)
    if do_ckpt {
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let p = Path::new(&dir).join("memory.ckp");
        let t = Instant::now();
        let bytes = m.checkpoint_to(&p, 0).unwrap();
        let ckpt_s = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let m2 = Memory::load_checkpoint(&p, cfg.clone()).unwrap();
        let reload_s = t.elapsed().as_secs_f64();
        let ok = ids(&m2, &items[0].text.chars().take(200).collect::<String>(), 1)
            .first()
            .map(|s| s == &items[0].id)
            .unwrap_or(false);
        drop(m2);
        result["checkpoint"] = json!({"bytes": bytes, "write_s": ckpt_s, "reload_s": reload_s, "reload_search_ok": ok});
    }

    // ---- queries answerable at this N
    let qf = BufReader::new(File::open(arg(&a, "queries", "")).unwrap());
    let queries: Vec<Q> = qf
        .lines()
        .map(|l| serde_json::from_str::<Q>(&l.unwrap()).unwrap())
        .filter(|q| q.max_gold_pos < items.len())
        .collect();
    let mut rw = BufWriter::new(File::create(arg(&a, "ranked", "ranked.tsv")).unwrap());
    for q in &queries {
        writeln!(rw, "{}\t{}", q.qid, ids(&m, &q.question, 64).join(" ")).unwrap();
    }
    rw.flush().unwrap();
    let mut lat = Vec::with_capacity(queries.len());
    for q in &queries {
        let t = Instant::now();
        std::hint::black_box(ids(&m, &q.question, 16));
        lat.push(t.elapsed().as_secs_f64());
    }
    let tq = Instant::now();
    std::thread::scope(|s| {
        let chunk = (queries.len() + threads - 1) / threads.max(1);
        for part in queries.chunks(chunk.max(1)) {
            let m = &m;
            s.spawn(move || {
                for q in part {
                    std::hint::black_box(ids(m, &q.question, 16));
                }
            });
        }
    });
    let qps = queries.len() as f64 / tq.elapsed().as_secs_f64();
    result["query"] = json!({"n_queries": queries.len(), "latency_top16": stats(lat), "qps_threads": threads, "qps": qps});

    // ---- one write at a time on the big in-memory store: time until search finds the new document (in-process visibility)
    let mut vis = Vec::new();
    let mut found = 0;
    for it in extra.iter().take(inc_ops) {
        let probe = probe_words(&it.text);
        let t = Instant::now();
        m.add_document(&it.id, "1", &it.text).unwrap();
        let hit = ids(&m, &probe, 16).iter().any(|x| x == &it.id);
        vis.push(t.elapsed().as_secs_f64());
        found += hit as usize;
    }
    result["incremental_in_process"] = json!({"ops": vis.len(), "found_by_probe": found, "write_plus_search_latency": stats(vis),
        "semantics": "add_document + BM25 update + a 12-word probe search; in memory, no fsync (comparable to Lucene NRT)"});

    // ---- durable writes: a persistent memory (op log, group commit, fdatasync per commit)
    let ddir = Path::new(&dir).join("durable");
    let _ = std::fs::remove_dir_all(&ddir);
    let mut dm = Memory::open(&ddir, cfg.clone(), true).unwrap();
    for it in items.iter().take(1000) {
        dm.add_document(&it.id, "1", &it.text).unwrap();
    }
    dm.commit().unwrap();
    let mut dur = Vec::new();
    for it in extra.iter().take(inc_ops) {
        let t = Instant::now();
        dm.add_document(&format!("{}#durable", it.id), "1", &it.text)
            .unwrap();
        dm.commit().unwrap();
        dur.push(t.elapsed().as_secs_f64());
    }
    result["incremental_durable"] = json!({"ops": dur.len(), "write_commit_fsync_latency": stats(dur),
        "semantics": "add_document + op-log append + commit with fdatasync (durable on return); store of 1,000 documents"});
    drop(dm);
    result["peak_rss_bytes"] = json!(proc_kb("VmHWM:") * 1024);
    result["checkpoint_dir_bytes"] = json!(dir_bytes(Path::new(&dir)));
    std::fs::write(&out, serde_json::to_string_pretty(&result).unwrap()).unwrap();
    println!("{}", serde_json::to_string(&result["build"]).unwrap());
}
