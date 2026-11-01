//! Component-level benchmark of xerj-fts on a real corpus: build segments through the crate's own
//! writer API, query them through `FtsSearcher`, measure latency / throughput / work and check that
//! the WAND fast path returns exactly what the generic path returns.
//!
//! ```text
//! fts_corpus_bench build  --corpus F.jsonl --out DIR [--n N] [--seg-docs 128000]
//!                         [--analyzer standard]
//! fts_corpus_bench query  --dir DIR --queries Q.jsonl --out R.json [--ranked R.tsv] [--k 16]
//!                         [--threads 4] [--passes 3] [--path wand|generic] [--tth 10000|exact]
//! fts_corpus_bench parity --dir DIR --queries Q.jsonl --out P.json [--ks 16] [--tth 10000]
//! ```
//!
//! * Corpus lines are `{"id": "...", "text": "..."}`; every line is one document with one `text`
//!   field (standard analyzer, positions on: the engine's default for a text field).
//!   `--analyzer NAME` indexes the field with another built-in analyzer of the registry (for
//!   example `english`: stop words + Snowball stemming, what a mapping's `"analyzer": "english"`
//!   selects); the name is recorded in `build.json` and `query` / `parity` analyse the questions
//!   with the same analyzer.
//! * `build` cuts a segment every `--seg-docs` documents (default 128 000 =
//!   `FLUSH_DOC_THRESHOLD_DEFAULT` in xerj-engine) and writes it with `FtsIndexWriter`
//!   (`configure_field` + `add_document` + `finish`). Segment-local doc ids are the ordinals inside
//!   the segment; the global order is (segment, ordinal). The corpus pages are dropped from the page
//!   cache behind the reader (`posix_fadvise`), so a long build does not grow the page cache.
//! * `query` mirrors what the engine does per request for a `match` query on one text field: the
//!   question is analysed with the field's analyzer, one `should` term clause per token (repeated
//!   tokens give repeated clauses, as in `query_node_to_fts_projected`), a single token becomes a
//!   plain term query; index-wide BM25 statistics (`CollectionStats`: field stats of every segment
//!   plus the per-segment `term_doc_freq` of the query's terms, built per request like the engine's
//!   #188 pre-pass) feed one `FtsSearcher` per segment; each segment returns its top k
//!   (`search_bounded_counting`, which takes the WAND path for this shape, or
//!   `search_bounded_observed` for the generic path); the per-segment lists are merged by (score
//!   desc, segment asc, doc asc). `--tth N` is `track_total_hits: N` the way the engine passes it
//!   (`total_count_limit`, the ES default 10 000): segment i is searched with the count limit
//!   `N + 1 - counted so far`, so the walk may stop counting (and rank with its MaxScore tail) once
//!   N + 1 matches are counted; `--tth exact` is `track_total_hits: true` (`u64::MAX`).
//!   One untimed warm-up pass, then `--passes` timed single-thread passes, then the throughput of
//!   `--threads` reader threads. Outside the timed regions the harness waits while the hottest
//!   thermal zone is above `QPERF_HOT_MC` (until below `QPERF_COOL_MC`; the benchmark guard sets
//!   both), and it re-measures every timed sample a guard pause landed in (`QPERF_PAUSE_LOG`).
//! * "postings in lists" = the postings of every clause's list in every segment (what a walk
//!   without skipping decodes). Postings decoded and BM25 term scores computed come from builds
//!   patched with thread-local counters (`work_counters`), the same rule on every branch.
//! * `parity`: per question and k, the WAND path and the generic path must return the same hits (doc
//!   ids, bit-identical f32 scores, order), per segment and after the merge; with `--tth exact` the
//!   totals must be equal, with `--tth N` the merged total must be exact below N + 1 and at least
//!   N + 1 otherwise (never above the exact total).
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;
use xerj_fts::{
    AnalyzerRegistry, BoolQuery, CollectionStats, FieldIndexConfig, FieldValues, FtsIndexReader,
    FtsIndexWriter, FtsSearcher, Query, ScoredHit, TermQuery,
};

const FIELD: &str = "text";

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

fn args() -> (String, HashMap<String, String>) {
    let raw: Vec<String> = std::env::args().collect();
    let mode = raw.get(1).cloned().unwrap_or_default();
    let mut a = HashMap::new();
    let mut i = 2;
    while i + 1 < raw.len() {
        a.insert(
            raw[i].trim_start_matches("--").to_string(),
            raw[i + 1].clone(),
        );
        i += 2;
    }
    (mode, a)
}

fn arg(a: &HashMap<String, String>, k: &str, d: &str) -> String {
    a.get(k).cloned().unwrap_or_else(|| d.to_string())
}

fn proc_kb(field: &str) -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .unwrap_or_default()
        .lines()
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

fn hottest_mc() -> u64 {
    let mut m = 0;
    if let Ok(rd) = std::fs::read_dir("/sys/class/thermal") {
        for e in rd.flatten() {
            if e.file_name().to_string_lossy().starts_with("thermal_zone") {
                if let Ok(t) = std::fs::read_to_string(e.path().join("temp")) {
                    m = m.max(t.trim().parse::<u64>().unwrap_or(0));
                }
            }
        }
    }
    m
}

fn loadavg() -> Vec<f64> {
    std::fs::read_to_string("/proc/loadavg")
        .unwrap_or_default()
        .split_whitespace()
        .take(3)
        .filter_map(|v| v.parse().ok())
        .collect()
}

fn memfree_gib() -> f64 {
    std::fs::read_to_string("/proc/meminfo")
        .unwrap_or_default()
        .lines()
        .find(|l| l.starts_with("MemFree:"))
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(0.0)
        / 1048576.0
}

/// Machine rule, outside timed regions: when the hottest zone is above `QPERF_HOT_MC` (set by the
/// benchmark guard; 88 C when unset), wait until it is below `QPERF_COOL_MC` (85 C when unset).
fn thermal_wait() -> f64 {
    let env = |k: &str, d: u64| {
        std::env::var(k)
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(d)
    };
    let (hot, cool) = (env("QPERF_HOT_MC", 88_000), env("QPERF_COOL_MC", 85_000));
    if hottest_mc() <= hot {
        return 0.0;
    }
    let t = Instant::now();
    while hottest_mc() >= cool {
        std::thread::sleep(std::time::Duration::from_millis(500));
    }
    t.elapsed().as_secs_f64()
}

fn epoch() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs_f64()
}

/// Intervals during which the benchmark guard had this process stopped (SIGSTOP at the thermal
/// limit), from the log it names in `QPERF_PAUSE_LOG` (`P <epoch>` / `R <epoch>` lines); an
/// interval still open runs to +inf. Empty when not run under the guard.
fn guard_pauses() -> Vec<(f64, f64)> {
    let Ok(path) = std::env::var("QPERF_PAUSE_LOG") else {
        return Vec::new();
    };
    let text = std::fs::read_to_string(path).unwrap_or_default();
    let mut out = Vec::new();
    let mut open: Option<f64> = None;
    for line in text.lines() {
        let mut it = line.split_whitespace();
        match (it.next(), it.next().and_then(|v| v.parse::<f64>().ok())) {
            (Some("P"), Some(t)) => open = Some(t),
            (Some("R"), Some(t)) => {
                if let Some(p) = open.take() {
                    out.push((p, t));
                }
            }
            _ => {}
        }
    }
    if let Some(p) = open {
        out.push((p, f64::INFINITY));
    }
    out
}

/// seconds of `[t0, t1]` the guard had the process stopped
fn paused_within(pauses: &[(f64, f64)], t0: f64, t1: f64) -> f64 {
    pauses
        .iter()
        .map(|&(a, b)| (b.min(t1) - a.max(t0)).max(0.0))
        .sum()
}

#[cfg(target_os = "linux")]
fn drop_cache(f: &File, upto: u64) {
    use std::os::fd::AsRawFd;
    extern "C" {
        fn posix_fadvise(fd: i32, offset: i64, len: i64, advice: i32) -> i32;
    }
    const POSIX_FADV_DONTNEED: i32 = 4;
    // SAFETY: advisory syscall on an open descriptor; no memory is touched.
    unsafe {
        posix_fadvise(f.as_raw_fd(), 0, upto as i64, POSIX_FADV_DONTNEED);
    }
}

#[cfg(not(target_os = "linux"))]
fn drop_cache(_f: &File, _upto: u64) {}

/// `--codec packed|efac` (efac = the experimental ZPS3 Elias-Fano + arithmetic full blocks; this
/// line is patched out of benchmark builds of branches without the codec switch)
fn codec_from(name: &str) -> xerj_fts::postings::BlockCodec {
    match name {
        "packed" => xerj_fts::postings::BlockCodec::Packed,
        "efac" => xerj_fts::postings::BlockCodec::EliasFanoAc,
        "nocounts" => xerj_fts::postings::BlockCodec::PackedNoCounts,
        _ => panic!("--codec packed|efac"),
    }
}

/// Byte accounting of every `.post` file in `dir`: file bytes (zstd envelope), the decompressed
/// postings the reader holds in RAM (the envelope's length field), and the block-index frame.
fn post_bytes(dir: &Path) -> Value {
    let (mut file, mut decompressed, mut frame, mut n) = (0u64, 0u64, 0u64, 0u64);
    for e in std::fs::read_dir(dir).unwrap().flatten() {
        let name = e.file_name().to_string_lossy().into_owned();
        if !name.ends_with(".post") {
            continue;
        }
        let bytes = std::fs::read(e.path()).unwrap_or_default();
        n += 1;
        file += bytes.len() as u64;
        if bytes.len() >= 8 && (&bytes[..3] == b"ZPS") {
            decompressed += u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as u64;
            if bytes.len() >= 16
                && u32::from_le_bytes(bytes[8..12].try_into().unwrap()) == 0x184D_2A5B
            {
                frame += 8 + u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as u64;
            }
        }
    }
    json!({"files": n, "file_bytes": file, "decompressed_postings_bytes": decompressed, "block_index_frame_bytes": frame})
}

fn dir_bytes(p: &Path) -> u64 {
    std::fs::read_dir(p)
        .map(|rd| {
            rd.flatten()
                .filter_map(|e| e.metadata().ok())
                .filter(|m| m.is_file())
                .map(|m| m.len())
                .sum()
        })
        .unwrap_or(0)
}

fn stats_ms(mut v: Vec<f64>) -> Value {
    if v.is_empty() {
        return json!(null);
    }
    v.sort_by(|a, b| a.total_cmp(b));
    let q = |f: f64| v[((v.len() as f64 * f) as usize).min(v.len() - 1)] * 1e3;
    json!({"n": v.len(), "mean_ms": v.iter().sum::<f64>() / v.len() as f64 * 1e3, "p50_ms": q(0.5),
           "p90_ms": q(0.9), "p95_ms": q(0.95), "p99_ms": q(0.99), "max_ms": v[v.len() - 1] * 1e3})
}

fn stats_u64(mut v: Vec<u64>) -> Value {
    if v.is_empty() {
        return json!(null);
    }
    v.sort_unstable();
    let q = |f: f64| v[((v.len() as f64 * f) as usize).min(v.len() - 1)];
    json!({"mean": v.iter().sum::<u64>() as f64 / v.len() as f64, "p50": q(0.5), "p95": q(0.95),
           "max": v[v.len() - 1], "sum": v.iter().sum::<u64>()})
}

// ------------------------------------------------------------------------------------------ build
fn build(a: &HashMap<String, String>) {
    let corpus = arg(a, "corpus", "");
    let out = PathBuf::from(arg(a, "out", ""));
    let n_max: usize = arg(a, "n", "0").parse().unwrap();
    let n_max = if n_max == 0 { usize::MAX } else { n_max };
    let seg_docs: usize = arg(a, "seg-docs", "128000").parse().unwrap();
    let codec_name = arg(a, "codec", "packed");
    let codec = codec_from(&codec_name);
    let analyzer_name = arg(a, "analyzer", "standard");
    std::fs::create_dir_all(&out).unwrap();
    let registry = Arc::new(AnalyzerRegistry::default());
    assert!(
        registry.get_analyzer(&analyzer_name).is_some(),
        "--analyzer {analyzer_name}: not a built-in analyzer"
    );
    let field_config = FieldIndexConfig {
        analyzer: analyzer_name.clone(),
        ..FieldIndexConfig::default()
    };
    let f = File::open(&corpus).expect("open corpus");
    let mut rd = BufReader::with_capacity(1 << 22, f.try_clone().unwrap());
    let mut ids = BufWriter::new(File::create(out.join("ids.txt")).unwrap());
    let la0 = loadavg();
    let free0 = memfree_gib();
    let mut min_free = free0;
    let cpu0 = cpu_seconds();
    let t0 = Instant::now();
    let (mut lines, mut text_bytes, mut consumed, mut dropped) = (0usize, 0u64, 0u64, 0u64);
    let mut segments: Vec<Value> = Vec::new();
    let mut finish_s = 0.0;
    let mut line = String::new();
    let mut eof = false;
    while !eof && lines < n_max {
        let seg_id = format!("seg-{:05}", segments.len());
        let mut w = FtsIndexWriter::new(&out, seg_id.clone(), Arc::clone(&registry))
            .with_block_codec(codec);
        w.configure_field(FIELD, field_config.clone());
        let mut ord = 0u32;
        let t_seg = Instant::now();
        while (ord as usize) < seg_docs && lines < n_max {
            line.clear();
            let nread = rd.read_line(&mut line).unwrap();
            if nread == 0 {
                eof = true;
                break;
            }
            consumed += nread as u64;
            if consumed - dropped >= 64 << 20 {
                dropped = consumed & !((1 << 20) - 1);
                drop_cache(&f, dropped);
            }
            let it: Item = serde_json::from_str(&line).expect("corpus line");
            text_bytes += it.text.len() as u64;
            let mut fields = HashMap::with_capacity(1);
            fields.insert(FIELD.to_string(), FieldValues::from(it.text));
            w.add_document(ord, &fields);
            writeln!(ids, "{}", it.id).unwrap();
            ord += 1;
            lines += 1;
        }
        if ord == 0 {
            break;
        }
        let add_s = t_seg.elapsed().as_secs_f64();
        let rss_before_finish = proc_kb("VmRSS:") * 1024;
        let t_fin = Instant::now();
        w.finish().expect("finish segment");
        let fin = t_fin.elapsed().as_secs_f64();
        finish_s += fin;
        let fr = memfree_gib();
        min_free = min_free.min(fr);
        segments.push(
            json!({"id": seg_id, "docs": ord, "add_s": add_s, "finish_s": fin,
                             "rss_before_finish_bytes": rss_before_finish}),
        );
        eprintln!(
            "[build] {seg_id}: {ord} docs, add {add_s:.1} s, finish {fin:.1} s; {lines} lines, {:.2} GB, {:.0} s, RSS {:.2} GB, \
             HWM {:.2} GB, MemFree {fr:.1} GiB, hottest {:.1} C",
            text_bytes as f64 / 1e9,
            t0.elapsed().as_secs_f64(),
            proc_kb("VmRSS:") as f64 / 1048576.0,
            proc_kb("VmHWM:") as f64 / 1048576.0,
            hottest_mc() as f64 / 1000.0
        );
    }
    ids.flush().unwrap();
    drop_cache(&f, consumed);
    let total_s = t0.elapsed().as_secs_f64();
    let paused_s = paused_within(&guard_pauses(), epoch() - total_s, epoch());
    let res = json!({
        "mode": "build", "corpus": corpus, "docs": lines, "text_bytes": text_bytes, "seg_docs": seg_docs,
        "segments": segments.len(), "field": FIELD, "analyzer": analyzer_name,
        "field_config": format!("{analyzer_name} analyzer, positions on"),
        "build": {"total_s": total_s, "paused_s": paused_s, "total_s_unpaused": total_s - paused_s,
                  "finish_s": finish_s, "cpu_s": cpu_seconds() - cpu0,
                  "docs_per_s": lines as f64 / total_s, "mb_per_s": text_bytes as f64 / 1e6 / total_s,
                  "peak_rss_bytes": proc_kb("VmHWM:") * 1024},
        "disk_bytes": dir_bytes(&out), "codec": codec_name, "postings": post_bytes(&out),
        "segment_list": segments,
        "machine": {"loadavg_start": la0, "loadavg_end": loadavg(), "memfree_gib_start": free0,
                    "memfree_gib_min_sampled": min_free, "hottest_c_end": hottest_mc() as f64 / 1000.0},
    });
    std::fs::write(
        out.join("build.json"),
        serde_json::to_string_pretty(&res).unwrap(),
    )
    .unwrap();
    println!("{}", serde_json::to_string(&res["build"]).unwrap());
}

// ------------------------------------------------------------------------------------------ load
struct Index {
    readers: Vec<Arc<FtsIndexReader>>,
    /// global doc number of each segment's ordinal 0
    base: Vec<u32>,
    ids: Vec<String>,
    registry: Arc<AnalyzerRegistry>,
    /// the field's analyzer as recorded by `build` (`standard` for indexes built before the option)
    analyzer: String,
    /// union of every segment's field stats (the per-request df is added per query)
    base_stats: CollectionStats,
    load_s: f64,
    docs: usize,
}

fn load(dir: &Path) -> Index {
    let t = Instant::now();
    let b: Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("build.json")).unwrap()).unwrap();
    let registry = Arc::new(AnalyzerRegistry::default());
    let mut readers = Vec::new();
    let mut base = Vec::new();
    let mut next = 0u32;
    let mut base_stats = CollectionStats::new();
    for s in b["segment_list"].as_array().unwrap() {
        let id = s["id"].as_str().unwrap();
        let r = FtsIndexReader::open(dir, id, &[FIELD]).expect("open segment");
        if let Some(fs) = r.field_stats(FIELD) {
            base_stats.add_field(FIELD, fs);
        }
        readers.push(Arc::new(r));
        base.push(next);
        next += s["docs"].as_u64().unwrap() as u32;
    }
    let ids: Vec<String> = BufReader::new(File::open(dir.join("ids.txt")).unwrap())
        .lines()
        .map(|l| l.unwrap())
        .collect();
    let analyzer = b["analyzer"].as_str().unwrap_or("standard").to_string();
    Index {
        readers,
        base,
        ids,
        registry,
        analyzer,
        base_stats,
        load_s: t.elapsed().as_secs_f64(),
        docs: next as usize,
    }
}

/// the engine's projection of `{"match": {"text": question}}` (operator OR, boost 1)
fn match_query(ix: &Index, question: &str) -> (Option<Query>, Vec<String>) {
    let analyzer = ix
        .registry
        .get_analyzer(&ix.analyzer)
        .expect("the index's analyzer");
    let toks: Vec<String> = analyzer
        .analyze(question)
        .into_iter()
        .map(|t| t.text)
        .collect();
    let q = match toks.len() {
        0 => None,
        1 => Some(Query::Term(TermQuery::new(FIELD, toks[0].clone()))),
        _ => {
            let mut b = BoolQuery::new().boost(1.0);
            for t in &toks {
                b = b.should(Query::Term(TermQuery::new(FIELD, t.clone())));
            }
            Some(Query::Bool(Box::new(b)))
        }
    };
    (q, toks)
}

/// index-wide statistics for this request: every segment's field stats + the df of each query term
fn request_stats(ix: &Index, toks: &[String]) -> Arc<CollectionStats> {
    let mut st = ix.base_stats.clone();
    let mut seen: Vec<&str> = Vec::with_capacity(toks.len());
    for t in toks {
        if seen.contains(&t.as_str()) {
            continue;
        }
        seen.push(t);
        for r in &ix.readers {
            if let Some(df) = r.term_doc_freq(FIELD, t) {
                st.add_doc_freq(FIELD, t, df as u64);
            }
        }
    }
    Arc::new(st)
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Path_ {
    /// `search_bounded_counting` with the engine's per-segment count limit (`None` = exact); `true` =
    /// hand each segment the worst score of the page merged so far (`search_bounded_counting_floor`,
    /// what a one-line change in xerj-engine's segment loop would do)
    Wand(Option<u64>, bool),
    /// `search_bounded_observed`: execute every clause, exact total
    Generic,
}

struct Answer {
    /// (segment, hit) best-first
    hits: Vec<(usize, ScoredHit)>,
    per_segment: Vec<(Vec<ScoredHit>, u64)>,
    total: u64,
}

fn run(ix: &Index, question: &str, k: usize, path: Path_) -> Answer {
    let (q, toks) = match_query(ix, question);
    let Some(q) = q else {
        return Answer {
            hits: Vec::new(),
            per_segment: Vec::new(),
            total: 0,
        };
    };
    let stats = request_stats(ix, &toks);
    let mut all: Vec<(usize, ScoredHit)> = Vec::new();
    let mut per_segment = Vec::with_capacity(ix.readers.len());
    let mut total = 0u64;
    for (si, r) in ix.readers.iter().enumerate() {
        let s = FtsSearcher::new(Arc::clone(r), Arc::clone(&ix.registry))
            .with_collection_stats(Some(Arc::clone(&stats)));
        let (hits, t) = match path {
            Path_::Wand(limit, use_floor) => {
                // xerj-engine `search_inner`: `limit + 1 - counted so far` (saturating), u64::MAX when exact
                let seg_limit =
                    limit.map_or(u64::MAX, |l| l.saturating_add(1).saturating_sub(total));
                let floor = if use_floor && all.len() >= k && k > 0 {
                    all.sort_by(|a, b| {
                        b.1.score
                            .total_cmp(&a.1.score)
                            .then(a.0.cmp(&b.0))
                            .then(a.1.doc_id.cmp(&b.1.doc_id))
                    });
                    all.truncate(k);
                    Some(all[k - 1].1.score)
                } else {
                    None
                };
                segment_page(&s, &q, k, seg_limit, floor)
            }
            Path_::Generic => s
                .search_bounded_observed(&q, k, false, |_| {})
                .expect("search"),
        };
        total += t;
        all.extend(hits.iter().cloned().map(|h| (si, h)));
        per_segment.push((hits, t));
    }
    all.sort_by(|a, b| {
        b.1.score
            .total_cmp(&a.1.score)
            .then(a.0.cmp(&b.0))
            .then(a.1.doc_id.cmp(&b.1.doc_id))
    });
    all.truncate(k);
    Answer {
        hits: all,
        per_segment,
        total,
    }
}

/// postings decoded by either path for this question: every clause's list in every segment
fn postings_visited(ix: &Index, question: &str) -> (u64, u32) {
    let (_, toks) = match_query(ix, question);
    let mut n = 0u64;
    for t in &toks {
        for r in &ix.readers {
            n += r.term_doc_freq(FIELD, t).unwrap_or(0) as u64;
        }
    }
    (n, toks.len() as u32)
}

/// `[postings decoded, BM25 term scores computed]` so far on this thread, in
/// builds patched with work counters (benchmark-only builds: the counters cost
/// time on the hot path, so latency is measured on unpatched builds); `None`
/// otherwise.
fn work_counters() -> Option<[u64; 2]> {
    None // WORK-COUNTERS-HOOK
}

fn read_queries(path: &str) -> Vec<Q> {
    BufReader::new(File::open(path).expect("queries"))
        .lines()
        .map(|l| serde_json::from_str::<Q>(&l.unwrap()).unwrap())
        .collect()
}

/// One segment's page through the match walk (benchmark builds of branches without the floor or the
/// count limit patch this one line: see qperf/fts/patch_build.py).
fn segment_page(
    s: &FtsSearcher,
    q: &Query,
    k: usize,
    limit: u64,
    floor: Option<f32>,
) -> (Vec<ScoredHit>, u64) {
    s.search_bounded_counting_floor(q, k, false, limit, floor)
        .expect("search") // COMPAT-LINE
}

fn parse_path(s: &str, tth: &str) -> Path_ {
    let limit = if tth == "exact" {
        None
    } else {
        Some(tth.parse::<u64>().expect("--tth N|exact"))
    };
    match s {
        "wand" => Path_::Wand(limit, false),
        "wand-floor" => Path_::Wand(limit, true),
        "generic" => Path_::Generic,
        _ => panic!("--path wand|wand-floor|generic"),
    }
}

// ------------------------------------------------------------------------------------------ query
fn query(a: &HashMap<String, String>) {
    let dir = PathBuf::from(arg(a, "dir", ""));
    let k: usize = arg(a, "k", "16").parse().unwrap();
    let threads: usize = arg(a, "threads", "4").parse().unwrap();
    let passes: usize = arg(a, "passes", "3").parse().unwrap();
    let qps_passes: usize = arg(a, "qps-passes", &passes.to_string()).parse().unwrap();
    let tth = arg(a, "tth", "10000");
    let path = parse_path(&arg(a, "path", "wand"), &tth);
    let la0 = loadavg();
    let ix = load(&dir);
    let rss_loaded = proc_kb("VmRSS:") * 1024;
    let qs = read_queries(&arg(a, "queries", ""));
    let warm: usize = arg(a, "warm", &qs.len().to_string()).parse().unwrap();
    let mut thermal_waits = 0.0;
    let t_warm = Instant::now();
    if let Some(rp) = a.get("ranked") {
        let mut rw = BufWriter::new(File::create(rp).unwrap());
        let all = arg(a, "ranked-all", "0") == "1";
        for q in qs.iter().filter(|q| all || q.max_gold_pos < ix.docs) {
            let ans = run(&ix, &q.question, 64, path);
            let v: Vec<&str> = ans
                .hits
                .iter()
                .map(|(s, h)| ix.ids[(ix.base[*s] + h.doc_id) as usize].as_str())
                .collect();
            writeln!(rw, "{}\t{}", q.qid, v.join(" ")).unwrap();
        }
        rw.flush().unwrap();
    }
    for q in qs.iter().take(warm) {
        std::hint::black_box(run(&ix, &q.question, k, path).hits.len());
    }
    let warm_s = t_warm.elapsed().as_secs_f64();
    let mut totals = vec![0u64; qs.len()];
    let mut lat = Vec::with_capacity(passes * qs.len());
    // (question, wall-clock start, end) of every timed sample, to find the ones a guard pause hit
    let mut spans: Vec<(usize, f64, f64)> = Vec::with_capacity(passes * qs.len());
    let t_timed0 = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs_f64();
    for _ in 0..passes {
        for (i, q) in qs.iter().enumerate() {
            if i % 8 == 0 {
                thermal_waits += thermal_wait();
            }
            let e0 = epoch();
            let t = Instant::now();
            let ans = std::hint::black_box(run(&ix, &q.question, k, path));
            let e = t.elapsed().as_secs_f64();
            lat.push(e);
            spans.push((i, e0, epoch()));
            totals[i] = ans.total;
        }
    }
    // a sample the guard stopped mid-flight measures the pause, not the query: measure it again
    let mut reruns = 0usize;
    for _ in 0..5 {
        let pauses = guard_pauses();
        let hit: Vec<usize> = (0..spans.len())
            .filter(|&j| paused_within(&pauses, spans[j].1, spans[j].2) > 0.0)
            .collect();
        if hit.is_empty() {
            break;
        }
        for j in hit {
            thermal_waits += thermal_wait();
            let q = &qs[spans[j].0];
            let e0 = epoch();
            let t = Instant::now();
            std::hint::black_box(run(&ix, &q.question, k, path).hits.len());
            lat[j] = t.elapsed().as_secs_f64();
            spans[j].1 = e0;
            spans[j].2 = epoch();
            reruns += 1;
        }
    }
    let mut per_q: Vec<Vec<f64>> = vec![Vec::new(); qs.len()];
    for (j, &(i, _, _)) in spans.iter().enumerate() {
        per_q[i].push(lat[j]);
    }
    let t_timed1 = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs_f64();
    let med: Vec<f64> = per_q
        .iter()
        .map(|v| {
            let mut v = v.clone();
            v.sort_by(|a, b| a.total_cmp(b));
            v[v.len() / 2]
        })
        .collect();
    let mut in_lists = Vec::with_capacity(qs.len());
    let mut clauses = Vec::with_capacity(qs.len());
    for q in &qs {
        let (v, c) = postings_visited(&ix, &q.question);
        in_lists.push(v);
        clauses.push(c as u64);
    }
    // work counters (untimed pass; only in builds whose `work_counters` reads real counters)
    let mut decoded = Vec::new();
    let mut scored = Vec::new();
    if work_counters().is_some() {
        for q in &qs {
            let before = work_counters().unwrap();
            std::hint::black_box(run(&ix, &q.question, k, path).hits.len());
            let after = work_counters().unwrap();
            decoded.push(after[0] - before[0]);
            scored.push(after[1] - before[1]);
        }
    }
    thermal_waits += thermal_wait();
    let qps_passes = if threads == 0 { 0 } else { qps_passes };
    let q0e = epoch();
    let tq = Instant::now();
    let ixr = &ix;
    std::thread::scope(|s| {
        let chunk = qs.len().div_ceil(threads.max(1));
        for part in qs.chunks(chunk.max(1)) {
            s.spawn(move || {
                for _ in 0..qps_passes {
                    for q in part {
                        std::hint::black_box(run(ixr, &q.question, k, path).hits.len());
                    }
                }
            });
        }
    });
    let qps_wall = tq.elapsed().as_secs_f64();
    let qps_paused = paused_within(&guard_pauses(), q0e, epoch());
    let qps = if qps_passes == 0 {
        0.0
    } else {
        (qps_passes * qs.len()) as f64 / (qps_wall - qps_paused).max(1e-9)
    };
    let res = json!({
        "mode": "query", "dir": dir, "path": format!("{path:?}"), "track_total_hits": tth, "k": k,
        "build": option_env!("FTS_BENCH_VARIANT").unwrap_or("unlabelled (set FTS_BENCH_VARIANT when building)"), "docs": ix.docs, "segments": ix.readers.len(), "analyzer": ix.analyzer,
        "n_queries": qs.len(), "passes": passes, "warm_queries": warm.min(qs.len()), "load_s": ix.load_s,
        "rss_after_load_bytes": rss_loaded, "warm_s": warm_s,
        "latency": stats_ms(lat), "latency_median_of_passes": stats_ms(med),
        "throughput": {"threads": threads, "qps": qps, "passes": qps_passes, "wall_s": qps_wall,
                       "guard_paused_s": qps_paused, "note": "qps over wall time minus guard pauses"},
        "samples_remeasured_after_guard_pause": reruns,
        "work": {"clauses": stats_u64(clauses),
                 "postings_in_lists": stats_u64(in_lists),
                 "postings_decoded": if decoded.is_empty() { json!("counting build only") } else { stats_u64(decoded) },
                 "postings_scored": if scored.is_empty() { json!("counting build only") } else { stats_u64(scored) },
                 "total_hits": stats_u64(totals)},
        "machine": {"loadavg_start": la0, "loadavg_end": loadavg(), "thermal_wait_s": thermal_waits,
                    "hottest_c_end": hottest_mc() as f64 / 1000.0, "timed_window_epoch": [t_timed0, t_timed1],
                    "rss_bytes": proc_kb("VmRSS:") * 1024, "memfree_gib_end": memfree_gib()},
    });
    std::fs::write(
        arg(a, "out", "query.json"),
        serde_json::to_string_pretty(&res).unwrap(),
    )
    .unwrap();
    println!(
        "{}",
        serde_json::to_string(&json!({"path": res["path"], "latency": res["latency"], "qps": qps}))
            .unwrap()
    );
}

// ------------------------------------------------------------------------------------------ parity
fn same_hits(a: &[ScoredHit], b: &[ScoredHit]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(x, y)| x.doc_id == y.doc_id && x.score.to_bits() == y.score.to_bits())
}

fn parity(a: &HashMap<String, String>) {
    let dir = PathBuf::from(arg(a, "dir", ""));
    let ks: Vec<usize> = arg(a, "ks", "16")
        .split(',')
        .map(|v| v.parse().unwrap())
        .collect();
    // --modes path:tth,... (default: --path / --tth): every mode is checked against ONE generic
    // answer per question and k (the generic path is the expensive part at scale)
    let modes: Vec<(String, Path_)> = match a.get("modes") {
        Some(m) => m
            .split(',')
            .map(|spec| {
                let (p, t) = spec.split_once(':').expect("--modes path:tth,...");
                (spec.to_string(), parse_path(p, t))
            })
            .collect(),
        None => {
            let tth = arg(a, "tth", "exact");
            let p = arg(a, "path", "wand");
            vec![(format!("{p}:{tth}"), parse_path(&p, &tth))]
        }
    };
    let ix = load(&dir);
    let qs = read_queries(&arg(a, "queries", ""));
    let t = Instant::now();
    #[derive(Default, Clone)]
    struct Tally {
        checks: usize,
        bad_merged: usize,
        bad_segments: usize,
        bad_totals: usize,
        examples: Vec<Value>,
    }
    let mut tallies: Vec<Tally> = vec![Tally::default(); modes.len()];
    for q in &qs {
        for &k in &ks {
            let g = run(&ix, &q.question, k, Path_::Generic);
            let gm: Vec<ScoredHit> = g.hits.iter().map(|x| x.1.clone()).collect();
            for ((_, wand), tally) in modes.iter().zip(tallies.iter_mut()) {
                let wand = *wand;
                let w = run(&ix, &q.question, k, wand);
                tally.checks += 1;
                let wm: Vec<ScoredHit> = w.hits.iter().map(|x| x.1.clone()).collect();
                let seg_ok = w.hits.iter().map(|x| x.0).eq(g.hits.iter().map(|x| x.0))
                    && same_hits(&wm, &gm);
                if !seg_ok {
                    tally.bad_merged += 1;
                    if tally.examples.len() < 5 {
                        tally.examples.push(json!({"qid": q.qid, "k": k,
                            "wand": w.hits.iter().map(|(s, h)| (s, h.doc_id, h.score)).collect::<Vec<_>>(),
                            "generic": g.hits.iter().map(|(s, h)| (s, h.doc_id, h.score)).collect::<Vec<_>>()}));
                    }
                }
                for (ws, gs) in w.per_segment.iter().zip(&g.per_segment) {
                    // with the cross-segment floor a segment returns only its hits at or above the
                    // floor: only the merged page is comparable
                    if !matches!(wand, Path_::Wand(_, true)) && !same_hits(&ws.0, &gs.0) {
                        tally.bad_segments += 1;
                    }
                    if matches!(wand, Path_::Wand(None, false)) && ws.1 != gs.1 {
                        tally.bad_totals += 1;
                    }
                }
                if let Path_::Wand(Some(limit), _) = wand {
                    // the merged total: exact below limit + 1, a lower bound of at least limit + 1
                    let ok = if g.total <= limit {
                        w.total == g.total
                    } else {
                        w.total > limit && w.total <= g.total
                    };
                    if !ok {
                        tally.bad_totals += 1;
                    }
                }
            }
        }
        thermal_wait();
    }
    let per_mode: Vec<Value> = modes
        .iter()
        .zip(&tallies)
        .map(|((name, _), t)| {
            json!({"mode": name, "checks": t.checks, "mismatches_merged_topk": t.bad_merged,
                   "mismatches_segment_topk": t.bad_segments, "mismatches_totals": t.bad_totals,
                   "examples": t.examples})
        })
        .collect();
    let bad: usize = tallies
        .iter()
        .map(|t| t.bad_merged + t.bad_segments + t.bad_totals)
        .sum();
    let res = json!({"mode": "parity", "dir": dir, "build": option_env!("FTS_BENCH_VARIANT").unwrap_or("unlabelled (set FTS_BENCH_VARIANT when building)"),
        "compared": "match walk (search_bounded_counting[_floor]) vs generic path (search_bounded_observed), per mode",
        "ks": ks, "queries": qs.len(), "segments": ix.readers.len(), "docs": ix.docs, "analyzer": ix.analyzer, "modes": per_mode,
        "mismatches_all_modes": bad,
        "tie_rule": "ScoredHit Ord: score desc, segment-local doc_id asc; merge: score desc, segment asc, doc asc",
        "total_rule": "exact mode: equal per segment; limit N: merged total exact up to N, else in (N, exact]",
        "score_rule": "bit-identical f32", "seconds": t.elapsed().as_secs_f64()});
    std::fs::write(
        arg(a, "out", "parity.json"),
        serde_json::to_string_pretty(&res).unwrap(),
    )
    .unwrap();
    println!("{}", serde_json::to_string(&res).unwrap());
    if bad > 0 {
        std::process::exit(1);
    }
}

fn vbyte_at(b: &[u8], pos: &mut usize) -> u64 {
    // xerj-fts vbyte: high bit set on the LAST byte
    let (mut v, mut shift) = (0u64, 0);
    loop {
        let c = b[*pos];
        *pos += 1;
        v |= ((c & 0x7f) as u64) << shift;
        if c & 0x80 != 0 {
            return v;
        }
        shift += 7;
    }
}

/// `bytes --dir DIR --out F.json`: the doc-id and frequency frame bytes of every FULL block of every
/// term with a block-index entry (parsed at the offsets the index stores; read-only), per codec
/// framing, plus the postings those blocks hold. Residuals (< 128 postings per term) and positions
/// are the same under both codecs and are reported as the remainder of the decompressed postings.
fn bytes(a: &HashMap<String, String>) {
    let dir = PathBuf::from(arg(a, "dir", ""));
    let ix = load(&dir);
    let b: Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("build.json")).unwrap()).unwrap();
    let (mut doc_bytes, mut freq_bytes, mut full_postings, mut blocks, mut terms) =
        (0u64, 0u64, 0u64, 0u64, 0u64);
    let mut magic = String::new();
    for (si, r) in ix.readers.iter().enumerate() {
        let seg = b["segment_list"][si]["id"].as_str().unwrap();
        let post = std::fs::read(dir.join(format!("{seg}.{FIELD}.post"))).unwrap();
        magic = String::from_utf8_lossy(&post[..4]).into_owned();
        let efac = &post[..4] == b"ZPS3";
        let Some(bi) = r.block_index(FIELD) else {
            continue;
        };
        let has_positions = r.field_has_positions(FIELD);
        let mut names = Vec::new();
        r.for_each_term(FIELD, |t| {
            names.push(t.to_string());
            true
        });
        for t in &names {
            let Some(tp) = r.lookup_term(FIELD, t) else {
                continue;
            };
            let Some(tb) = bi.term(tp.postings_offset) else {
                continue;
            };
            let Some(data) = r.postings_data(FIELD, &tp) else {
                continue;
            };
            let freqs_stored = has_positions && tp.total_term_frequency != tp.doc_frequency as u64;
            terms += 1;
            let nfull = tp.doc_frequency as usize / 128;
            for blk in 0..nfull {
                let mut pos = tb.offset[blk] as usize;
                let start = pos;
                if efac {
                    let l = data[pos] as u64;
                    pos += 1;
                    let u = vbyte_at(data, &mut pos);
                    pos += ((128 * l + 128 + ((u - 1) >> l) + 1) as usize).div_ceil(8);
                } else {
                    let w = data[pos] as usize;
                    pos += 1;
                    vbyte_at(data, &mut pos);
                    pos += 16 * w;
                }
                doc_bytes += (pos - start) as u64;
                if freqs_stored {
                    let fstart = pos;
                    if efac {
                        pos += 1;
                        let len = vbyte_at(data, &mut pos) as usize;
                        pos += len;
                    } else {
                        let w = data[pos] as usize;
                        pos += 1;
                        vbyte_at(data, &mut pos);
                        pos += 16 * w;
                    }
                    freq_bytes += (pos - fstart) as u64;
                }
                blocks += 1;
                full_postings += 128;
            }
        }
    }
    let decompressed = b["postings"]["decompressed_postings_bytes"]
        .as_u64()
        .unwrap_or(0);
    let res = json!({"mode": "bytes", "dir": dir, "envelope": magic, "codec": b["codec"],
        "terms_with_block_index": terms, "full_blocks": blocks, "full_block_postings": full_postings,
        "doc_frame_bytes": doc_bytes, "freq_frame_bytes": freq_bytes,
        "doc_bits_per_posting": doc_bytes as f64 * 8.0 / full_postings.max(1) as f64,
        "freq_bits_per_posting": freq_bytes as f64 * 8.0 / full_postings.max(1) as f64,
        "decompressed_postings_bytes": decompressed,
        "residuals_and_positions_bytes": decompressed.saturating_sub(doc_bytes + freq_bytes),
        "post_file_bytes": b["postings"]["file_bytes"], "block_index_frame_bytes": b["postings"]["block_index_frame_bytes"]});
    std::fs::write(
        arg(a, "out", "bytes.json"),
        serde_json::to_string_pretty(&res).unwrap(),
    )
    .unwrap();
    println!("{}", serde_json::to_string(&res).unwrap());
}

/// `attrib --dir DIR --out F.json`: every byte of the decompressed `.post` of the corpus field by
/// class (ZPS2 only), each class's size under zstd on its own, what the rest of the segment holds,
/// and what two position layouts would cost.
fn attrib(a: &HashMap<String, String>) {
    #[derive(Default)]
    struct Class {
        raw: Vec<u8>,
    }
    let dir = PathBuf::from(arg(a, "dir", ""));
    let level: i32 = arg(a, "level", "3").parse().unwrap();
    let ix = load(&dir);
    let b: Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("build.json")).unwrap()).unwrap();
    let names = [
        "block_doc_ids",
        "block_tf",
        "block_pos_counts",
        "block_pos_gaps",
        "tail_header",
        "tail_doc_ids",
        "tail_tf",
        "tail_pos_counts",
        "tail_pos_gaps",
    ];
    let mut c: Vec<Class> = names.iter().map(|_| Class::default()).collect();
    let (mut postings, mut positions, mut count_ne_tf, mut gap_1b, mut gap_2b, mut gap_big) =
        (0u64, 0u64, 0u64, 0u64, 0u64, 0u64);
    let mut gaps_all: Vec<u32> = Vec::new();
    let mut for128: Vec<u8> = Vec::new();
    // Whole-layout candidates: every stream as it would sit in the file.
    let (mut s1, mut s2_main, mut s2_pos, mut s3_main, mut s3_pos) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let mut front: Vec<u8> = Vec::new();
    let mut pos_stream_vb = 0u64;
    for (si, r) in ix.readers.iter().enumerate() {
        let _ = si;
        let has_positions = r.field_has_positions(FIELD);
        let mut terms = Vec::new();
        r.for_each_term(FIELD, |t| {
            terms.push(t.to_string());
            true
        });
        // Front-coded dictionary in 16-term buckets, as the study's: shared prefix, suffix.
        let mut prev: &[u8] = b"";
        for (i, t) in terms.iter().enumerate() {
            let b = t.as_bytes();
            let shared = if i % 16 == 0 {
                0
            } else {
                b.iter().zip(prev).take_while(|(x, y)| x == y).count()
            };
            xerj_fts::postings::vbyte_encode(shared as u32, &mut front);
            xerj_fts::postings::vbyte_encode((b.len() - shared) as u32, &mut front);
            front.extend_from_slice(&b[shared..]);
            prev = b;
        }
        let mut tps: Vec<_> = terms
            .iter()
            .filter_map(|t| r.lookup_term(FIELD, t))
            .collect();
        tps.sort_by_key(|t| t.postings_offset);
        for tp in &tps {
            let Some(data) = r.postings_data(FIELD, tp) else {
                continue;
            };
            let freqs = has_positions && tp.total_term_frequency != tp.doc_frequency as u64;
            let df = tp.doc_frequency as usize;
            let mut pos = 0usize;
            gaps_all.clear();
            // The group's gaps without its count, and the same split by first or later in the doc.
            let group = |pos: &mut usize,
                             counts: usize,
                             gaps: usize,
                             c: &mut Vec<Class>,
                             tf: Option<u64>,
                             st: &mut (u64, u64, u64, u64, u64),
                             ga: &mut Vec<u32>,
                             out: &mut Vec<u8>| {
                let s = *pos;
                let n = vbyte_at(data, pos);
                c[counts].raw.extend_from_slice(&data[s..*pos]);
                if tf.is_some_and(|t| t != n) {
                    st.0 += 1;
                }
                st.1 += n;
                for _ in 0..n {
                    let s = *pos;
                    let g = vbyte_at(data, pos);
                    ga.push(g as u32);
                    c[gaps].raw.extend_from_slice(&data[s..*pos]);
                    out.extend_from_slice(&data[s..*pos]);
                    match *pos - s {
                        1 => st.2 += 1,
                        2 => st.3 += 1,
                        _ => st.4 += 1,
                    }
                }
            };
            let mut st = (0u64, 0u64, 0u64, 0u64, 0u64);
            for _ in 0..df / 128 {
                let s = pos;
                let w = data[pos] as usize;
                pos += 1;
                vbyte_at(data, &mut pos);
                pos += 16 * w;
                c[0].raw.extend_from_slice(&data[s..pos]);
                s1.extend_from_slice(&data[s..pos]);
                s2_main.extend_from_slice(&data[s..pos]);
                s3_main.extend_from_slice(&data[s..pos]);
                let mut tfs = [1u64; 128];
                if freqs {
                    let s = pos;
                    let w = data[pos] as usize;
                    pos += 1;
                    let min = vbyte_at(data, &mut pos);
                    let end = pos + 16 * w;
                    if w == 0 {
                        tfs = [min; 128];
                    } else {
                        // Only the equal-count statistic needs them, and that needs the unpack.
                        tfs = [0; 128];
                    }
                    pos = end;
                    c[1].raw.extend_from_slice(&data[s..pos]);
                    s1.extend_from_slice(&data[s..pos]);
                    s2_main.extend_from_slice(&data[s..pos]);
                    s3_main.extend_from_slice(&data[s..pos]);
                }
                if has_positions {
                    let mut out = Vec::new();
                    for &tf in &tfs {
                        let tf = (tf != 0).then_some(tf);
                        group(&mut pos, 2, 3, &mut c, tf, &mut st, &mut gaps_all, &mut out);
                    }
                    s1.extend_from_slice(&out);
                    s2_pos.extend_from_slice(&out);
                    s3_pos.extend_from_slice(&out);
                }
                postings += 128;
            }
            let tail = df % 128;
            if tail > 0 {
                c[4].raw.push(data[pos]);
                s1.push(data[pos]);
                s2_main.push(data[pos]);
                s3_main.push(data[pos]);
                pos += 1;
                for _ in 0..tail {
                    let s = pos;
                    vbyte_at(data, &mut pos);
                    c[5].raw.extend_from_slice(&data[s..pos]);
                    s1.extend_from_slice(&data[s..pos]);
                    s2_main.extend_from_slice(&data[s..pos]);
                    s3_main.extend_from_slice(&data[s..pos]);
                    let mut tf = if has_positions { Some(1) } else { None };
                    if freqs {
                        let s = pos;
                        tf = Some(vbyte_at(data, &mut pos));
                        c[6].raw.extend_from_slice(&data[s..pos]);
                        s1.extend_from_slice(&data[s..pos]);
                        s2_main.extend_from_slice(&data[s..pos]);
                        s3_main.extend_from_slice(&data[s..pos]);
                    }
                    if has_positions {
                        let mut out = Vec::new();
                        group(&mut pos, 7, 8, &mut c, tf, &mut st, &mut gaps_all, &mut out);
                        s1.extend_from_slice(&out);
                        s2_main.extend_from_slice(&out);
                        s3_pos.extend_from_slice(&out);
                    }
                }
                postings += tail as u64;
            }
            assert_eq!(pos, data.len(), "walked past the term");
            count_ne_tf += st.0;
            positions += st.1;
            gap_1b += st.2;
            gap_2b += st.3;
            gap_big += st.4;
            pos_stream_vb += gaps_all.len() as u64;
            for chunk in gaps_all.chunks(128) {
                if chunk.len() == 128 {
                    use bitpacking::BitPacker;
                    let max = *chunk.iter().max().unwrap();
                    let w = (32 - max.leading_zeros()) as u8;
                    let lanes: [u32; 128] = chunk.try_into().unwrap();
                    let mut out = vec![0u8; 16 * w as usize];
                    bitpacking::BitPacker4x::new().compress(&lanes, &mut out, w);
                    for128.push(w);
                    for128.extend_from_slice(&out);
                } else {
                    for &g in chunk {
                        xerj_fts::postings::vbyte_encode(g, &mut for128);
                    }
                }
            }
        }
    }
    let mut classes = serde_json::Map::new();
    for (name, class) in names.iter().zip(&c) {
        let z = zstd::bulk::compress(&class.raw, level).unwrap();
        classes.insert(
            name.to_string(),
            json!({"raw_bytes": class.raw.len(), "zstd_bytes": z.len()}),
        );
    }
    let z = |b: &[u8]| zstd::bulk::compress(b, level).unwrap().len();
    let layouts = json!({
        "inline_no_counts": z(&s1),
        "block_positions_apart": [z(&s2_main), z(&s2_pos)],
        "all_positions_apart": [z(&s3_main), z(&s3_pos)],
        "front_coded_terms_raw": front.len(), "front_coded_terms_zstd": z(&front),
    });
    let mut ext: HashMap<String, u64> = HashMap::new();
    for e in std::fs::read_dir(&dir).unwrap().flatten() {
        let name = e.file_name().to_string_lossy().into_owned();
        let parts: Vec<&str> = name.rsplitn(3, '.').collect();
        let key = match parts.as_slice() {
            [x, f, _] => format!("{f}.{x}"),
            [x, ..] => x.to_string(),
            [] => continue,
        };
        *ext.entry(key).or_default() += e.metadata().map_or(0, |m| m.len());
    }
    let res = json!({"mode": "attrib", "dir": dir, "codec": b["codec"], "postings": postings,
        "positions": positions, "position_count_differs_from_tf": count_ne_tf,
        "gap_bytes_1": gap_1b, "gap_bytes_2": gap_2b, "gap_bytes_3_plus": gap_big,
        "classes": classes, "layouts": layouts, "files_by_field_and_ext": ext,
        "counterfactual_gaps_for128_raw": for128.len(),
        "counterfactual_gaps_for128_zstd": zstd::bulk::compress(&for128, level).unwrap().len(), "gaps_counted": pos_stream_vb,
        "post_file_bytes": b["postings"]["file_bytes"],
        "block_index_frame_bytes": b["postings"]["block_index_frame_bytes"],
        "decompressed_postings_bytes": b["postings"]["decompressed_postings_bytes"]});
    std::fs::write(
        arg(a, "out", "attrib.json"),
        serde_json::to_string_pretty(&res).unwrap(),
    )
    .unwrap();
    println!("{}", serde_json::to_string(&res).unwrap());
}

fn main() {
    let (mode, a) = args();
    match mode.as_str() {
        "build" => build(&a),
        "query" => query(&a),
        "parity" => parity(&a),
        "bytes" => bytes(&a),
        "attrib" => attrib(&a),
        _ => {
            eprintln!("usage: fts_corpus_bench build|query|parity|bytes ... (see the module docs)");
            std::process::exit(2);
        }
    }
}
