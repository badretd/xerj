//! xerj-corpus: a large document collection in one xerj-index Memory (native BM25 over documents), built once, served from the
//! mmap checkpoint. The XERJ Lab's live "10 GB of Wikipedia" search uses it.
//!
//!   xerj-corpus build --corpus corpus.jsonl --out DIR [--n N]      corpus lines: {"id": "...", "text": "Title\nPassage ..."}
//!   xerj-corpus serve --dir DIR                                    JSON lines on stdin / stdout
//!
//! Requests: {"op":"stats"} | {"op":"search","query":"...","k":10,"fuzzy":true}
//! Typo tolerance at this scale: a query word (>= 4 letters) that is absent or very rare in the corpus is replaced by its most
//! frequent spelling within 1 edit (2 edits for >= 9 letters when 1 finds nothing), when that spelling is >= 50x more frequent.
//! Candidates are generated (deletes, transposes, replaces, inserts) and checked against the term dictionary in O(1) each, so
//! the 10 GB vocabulary is never scanned.
use serde::Deserialize;
use serde_json::{json, Value};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;
use std::time::Instant;
use xerj_index::{Memory, MemoryConfig};

fn arg(args: &[String], k: &str) -> Option<String> {
    args.iter()
        .position(|a| a == k)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

fn proc_kb(key: &str) -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with(key))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|v| v.parse().ok())
        })
        .unwrap_or(0)
}

fn cfg() -> MemoryConfig {
    MemoryConfig {
        bm25_records: false,
        bm25_documents: true,
        ..MemoryConfig::default()
    }
}

#[derive(Deserialize)]
struct Item {
    id: String,
    text: String,
}

fn build(args: &[String]) {
    let corpus = arg(args, "--corpus").expect("--corpus");
    let out = PathBuf::from(arg(args, "--out").expect("--out"));
    let n_max: usize = arg(args, "--n")
        .and_then(|v| v.parse().ok())
        .unwrap_or(usize::MAX);
    std::fs::create_dir_all(&out).expect("create --out");
    let t0 = Instant::now();
    let mut m = Memory::new(cfg());
    let f = BufReader::with_capacity(1 << 22, std::fs::File::open(&corpus).expect("open corpus"));
    let (mut n, mut bytes) = (0usize, 0u64);
    for line in f.lines() {
        if n >= n_max {
            break;
        }
        let line = line.expect("read");
        let it: Item = serde_json::from_str(&line).expect("corpus line");
        bytes += it.text.len() as u64;
        m.add_document(&it.id, "1", &it.text).expect("add_document");
        n += 1;
        if n % 1_000_000 == 0 {
            eprintln!(
                "[build] {n} passages, {:.1} GB, {:.0} s",
                bytes as f64 / 1e9,
                t0.elapsed().as_secs_f64()
            );
        }
    }
    m.commit().expect("commit");
    let build_s = t0.elapsed().as_secs_f64();
    let t1 = Instant::now();
    let ck = out.join("corpus.ckpt");
    let ck_bytes = m.checkpoint_to(&ck, 0).expect("checkpoint");
    let stats = json!({"passages": n, "text_bytes": bytes, "build_s": (build_s * 10.0).round() / 10.0,
                       "checkpoint_bytes": ck_bytes, "checkpoint_s": (t1.elapsed().as_secs_f64() * 10.0).round() / 10.0,
                       "peak_rss_bytes": proc_kb("VmHWM:") * 1024, "corpus": corpus, "built": "xerj-corpus build (native BM25, 1 thread, inline)"});
    std::fs::write(
        out.join("build.json"),
        serde_json::to_string_pretty(&stats).unwrap(),
    )
    .expect("write build.json");
    println!("{stats}");
}

fn analyze(text: &str) -> Vec<String> {
    let mut v = Vec::new();
    let mut scratch = String::new();
    xerj_index::bm25::tokens(text, &mut scratch, |t| v.push(t.to_string()));
    v
}

const ABC: &[u8] = b"abcdefghijklmnopqrstuvwxyz";

fn edits1(w: &[u8], out: &mut Vec<Vec<u8>>) {
    let n = w.len();
    for i in 0..n {
        let mut d = w.to_vec();
        d.remove(i);
        out.push(d);
    }
    for i in 0..n.saturating_sub(1) {
        let mut t = w.to_vec();
        t.swap(i, i + 1);
        out.push(t);
    }
    for i in 0..n {
        for &c in ABC {
            if w[i] != c {
                let mut r = w.to_vec();
                r[i] = c;
                out.push(r);
            }
        }
    }
    for i in 0..=n {
        for &c in ABC {
            let mut s = w.to_vec();
            s.insert(i, c);
            out.push(s);
        }
    }
}

/// most frequent known spelling within `max_d` edits: (word, df, distance)
fn best_spelling(m: &Memory, w: &str, max_d: usize) -> Option<(String, u32, usize)> {
    let b = w.as_bytes();
    let mut c1 = Vec::with_capacity(60 * b.len());
    edits1(b, &mut c1);
    let mut best: Option<(String, u32, usize)> = None;
    let consider = |cand: &[u8], d: usize, best: &mut Option<(String, u32, usize)>| {
        if let Ok(s) = std::str::from_utf8(cand) {
            let df = m.document_term_df(s).unwrap_or(0);
            if df > 0 && best.as_ref().is_none_or(|x| df > x.1) {
                *best = Some((s.to_string(), df, d));
            }
        }
    };
    for c in &c1 {
        consider(c, 1, &mut best);
    }
    if best.is_none() && max_d >= 2 {
        let mut c2 = Vec::with_capacity(60 * b.len());
        for c in c1.iter() {
            c2.clear();
            edits1(c, &mut c2);
            for x in &c2 {
                consider(x, 2, &mut best);
            }
        }
    }
    best
}

fn search(m: &Memory, req: &Value) -> Value {
    let q = req
        .get("query")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_string();
    if q.is_empty() {
        return json!({"ok": false, "error": "type a query"});
    }
    let k = req
        .get("k")
        .and_then(|v| v.as_u64())
        .unwrap_or(10)
        .clamp(1, 50) as usize;
    let fuzzy = req.get("fuzzy").and_then(|v| v.as_bool()).unwrap_or(true);
    let t0 = Instant::now();
    let toks = analyze(&q);
    let mut used: Vec<String> = Vec::with_capacity(toks.len());
    let mut exps = Vec::new();
    let mut seen: Vec<String> = Vec::new();
    for t in &toks {
        let mut out = t.clone();
        if fuzzy && t.len() >= 4 && t.bytes().all(|c| c.is_ascii_lowercase()) && !seen.contains(t) {
            seen.push(t.clone());
            let df = m.document_term_df(t).unwrap_or(0);
            if df < 20 {
                let max_d = if t.len() >= 9 { 2 } else { 1 };
                if let Some((w, wdf, d)) = best_spelling(m, t, max_d) {
                    if wdf >= 20 && wdf as u64 >= 50 * (df.max(1) as u64) {
                        exps.push(
                            json!({"word": t, "df": df, "to": w, "to_df": wdf, "distance": d}),
                        );
                        out = w;
                    }
                }
            }
        }
        used.push(out);
    }
    let expand_us = t0.elapsed().as_secs_f64() * 1e6;
    let q_used = used.join(" ");
    let t1 = Instant::now();
    let hits = m.search_documents(&q_used, k);
    let search_us = t1.elapsed().as_secs_f64() * 1e6;
    let st = &m.store;
    let rows: Vec<Value> = hits
        .iter()
        .map(|h| {
            let (id, _) = st.doc_key_str(h.doc);
            let text = st.doc_text(h.doc);
            let (title, body) = text.split_once('\n').unwrap_or(("", text));
            let mut cut = body.len().min(420);
            while !body.is_char_boundary(cut) {
                cut -= 1;
            }
            json!({"id": id, "title": title, "snippet": &body[..cut], "more": body.len() > cut, "score": (h.score * 1000.0).round() / 1000.0})
        })
        .collect();
    json!({"ok": true, "query": q, "query_used": q_used, "fuzzy": fuzzy, "expansions": exps, "hits": rows,
           "expand_us": (expand_us * 10.0).round() / 10.0, "search_us": (search_us * 10.0).round() / 10.0})
}

fn serve(args: &[String]) {
    let dir = PathBuf::from(arg(args, "--dir").expect("--dir"));
    let t0 = Instant::now();
    let m = Memory::load_checkpoint(&dir.join("corpus.ckpt"), cfg()).expect("load checkpoint");
    let load_ms = t0.elapsed().as_secs_f64() * 1e3;
    let build: Value = std::fs::read_to_string(dir.join("build.json"))
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or(Value::Null);
    let mut out = BufWriter::new(std::io::stdout());
    writeln!(out, "{}", json!({"ready": true, "pid": std::process::id(), "load_ms": (load_ms * 10.0).round() / 10.0, "build": build})).unwrap();
    out.flush().unwrap();
    for line in std::io::stdin().lock().lines() {
        let Ok(line) = line else { break };
        if line.trim().is_empty() {
            continue;
        }
        let t = Instant::now();
        let mut res = match serde_json::from_str::<Value>(&line) {
            Ok(req) => match req.get("op").and_then(|o| o.as_str()).unwrap_or("") {
                "stats" => {
                    json!({"ok": true, "build": build, "load_ms": (load_ms * 10.0).round() / 10.0,
                                  "rss_bytes": proc_kb("VmRSS:") * 1024, "pid": std::process::id()})
                }
                "search" => search(&m, &req),
                op => json!({"ok": false, "error": format!("unknown op {op:?}")}),
            },
            Err(e) => json!({"ok": false, "error": format!("bad json: {e}")}),
        };
        res["worker_ms"] = json!((t.elapsed().as_secs_f64() * 1e4).round() / 10.0);
        writeln!(out, "{res}").unwrap();
        out.flush().unwrap();
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("build") => build(&args),
        Some("serve") => serve(&args),
        _ => {
            eprintln!("usage: xerj-corpus build --corpus FILE --out DIR [--n N] | serve --dir DIR");
            std::process::exit(2);
        }
    }
}
