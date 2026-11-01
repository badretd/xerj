//! Engine-level benchmark of the request bodies the user tools send, in process (no HTTP).
//!
//! ```text
//! body_bench ingest --data DIR --corpus F.jsonl [--n 1000000] [--batch 2000]
//! body_bench query  --data DIR --queries Q.jsonl --out R.json [--shapes cli,mcp,match,...]
//!                   [--k 10] [--n-queries 200] [--warm 20] [--mapped id,text] [--tth 10000|exact]
//! ```
//!
//! * `ingest` creates index `bench` (`id` keyword, `text` text: the two fields of a
//!   `{"id", "text"}` JSONL corpus) under `DIR`, feeds the first `--n` lines through
//!   `Index::index_batch_turbo` (the bulk path), flushes, and runs background merge passes until
//!   the policy selects nothing.
//! * `query` reopens `DIR` and runs every question once per shape, single-threaded and
//!   sequential, after `--warm` untimed questions. Shapes:
//!   - `cli`: `xerj search`'s body (`xerj_common::code_query::search_query_body_for_mapping`
//!     with the mapping `--mapped`), as sent: `_source` citation list and `fields: ["_passage"]`;
//!   - `mcp`: the `xerj_search` MCP tool's body for a plain-string query: the same query, size
//!     `--k`, its `_source` default list and `fields: ["_passage"]`;
//!   - `cli_walk`: the CLI query with `_source: ["id"]` and no `fields` (walk + a minimal page);
//!   - `match`: `{"match": {"text": q}}`, `_source: ["id"]`;
//!   - `leg_mm`, `leg_phrase`, `leg_cov_half`, `leg_cov_all`, `no_phrase`, `no_cov`: the CLI
//!     query cut to one leg (or without one), `_source: ["id"]` — which leg costs what.
//! * The engine's own phase log (`XERJ_PHASE_LOG_MS=0`, the `warn!` line `search_inner` emits)
//!   is captured per request: `segment_loop` (the FTS walk and admission) and `hydrate+corpus`.
use std::collections::{BTreeSet, HashMap};
use std::io::{BufRead, BufReader, Write};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use serde_json::{json, Value};
use xerj_common::config::Config;
use xerj_common::types::{FieldConfig, FieldType, Schema};

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

/// Phase lines of the engine's slow-search log, captured from `tracing`.
#[derive(Clone, Default)]
struct Captured(Arc<Mutex<Vec<String>>>);

impl Write for Captured {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let s = String::from_utf8_lossy(buf);
        if s.contains("phases") {
            self.0.lock().unwrap().push(s.into_owned());
        }
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for Captured {
    type Writer = Captured;
    fn make_writer(&'a self) -> Self::Writer {
        self.clone()
    }
}

/// `name=12ms` pairs of the `phases` field.
fn phases_of(line: &str) -> HashMap<String, f64> {
    let mut out = HashMap::new();
    if let Some(start) = line.find("phases=") {
        let rest = &line[start + 7..];
        let rest = rest.trim_start_matches('"');
        let end = rest.find(" gates=").unwrap_or(rest.len());
        for kv in rest[..end].trim_end_matches('"').split_whitespace() {
            if let Some((k, v)) = kv.split_once('=') {
                if let Ok(ms) = v.trim_end_matches("ms").parse::<f64>() {
                    out.insert(k.to_string(), ms);
                }
            }
        }
    }
    out
}

fn stats(mut v: Vec<f64>) -> Value {
    if v.is_empty() {
        return json!(null);
    }
    v.sort_by(|a, b| a.total_cmp(b));
    let q = |f: f64| v[((v.len() as f64 * f) as usize).min(v.len() - 1)];
    json!({"n": v.len(), "mean": v.iter().sum::<f64>() / v.len() as f64, "p50": q(0.5),
           "p90": q(0.9), "p95": q(0.95), "p99": q(0.99), "max": v[v.len() - 1]})
}

fn config(data: &str) -> Config {
    let mut config = Config::default();
    config.server.data_dir = data.to_string();
    config
}

async fn ingest(a: &HashMap<String, String>) {
    let data = arg(a, "data", "");
    let corpus = arg(a, "corpus", "");
    let n: usize = arg(a, "n", "1000000").parse().unwrap();
    let batch: usize = arg(a, "batch", "2000").parse().unwrap();
    // `--facets 1`: Discover-style fields derived from the line number — two
    // keyword facets and a date — beside `id` / `text`.
    let facets = arg(a, "facets", "0") == "1";
    // `text` without doc values unless `--text-doc-values 1`: what a declared
    // or dynamically mapped `text` field gets through the ES API
    // (`doc_values_default`). `FieldConfig::new` keeps them, so an index built
    // through it carries the passages a second time in its `.dv` side-cars.
    let text_doc_values = arg(a, "text-doc-values", "0") == "1";
    let engine = xerj_engine::Engine::new(config(&data)).expect("engine");
    let mut schema = Schema::empty();
    schema
        .add_field(FieldConfig::new("id", FieldType::Keyword))
        .unwrap();
    let mut text = FieldConfig::new("text", FieldType::Text);
    text.options.doc_values = text_doc_values;
    schema.add_field(text).unwrap();
    if facets {
        for (f, t) in [
            ("cat", FieldType::Keyword),
            ("tag", FieldType::Keyword),
            ("ts", FieldType::Date),
        ] {
            schema.add_field(FieldConfig::new(f, t)).unwrap();
        }
    }
    engine.create_index("bench", schema).expect("create index");
    let idx = engine.get_index("bench").expect("index");
    let t0 = Instant::now();
    let rd = BufReader::with_capacity(1 << 22, std::fs::File::open(&corpus).expect("corpus"));
    let mut docs: Vec<(String, Value)> = Vec::with_capacity(batch);
    let mut done = 0usize;
    for line in rd.lines() {
        if done >= n {
            break;
        }
        let v: Value = serde_json::from_str(&line.unwrap()).expect("corpus line");
        let id = v["id"].as_str().unwrap().to_string();
        let text = v["text"].as_str().unwrap().to_string();
        let mut doc = json!({"id": id, "text": text});
        if facets {
            let n = done as u64;
            let day = chrono_day(n % 365);
            doc["cat"] = json!(format!("c{}", n % 9));
            doc["tag"] = json!(format!("t{}", (n / 3) % 4));
            doc["ts"] = json!(format!("{day}T{:02}:00:00.000Z", (n * 7) % 24));
        }
        docs.push((id.clone(), doc));
        done += 1;
        if docs.len() == batch {
            idx.index_batch_turbo(std::mem::take(&mut docs), false, false)
                .await
                .expect("bulk");
            if done.is_multiple_of(100_000) {
                eprintln!("[ingest] {done} docs, {:.0} s", t0.elapsed().as_secs_f64());
            }
        }
    }
    if !docs.is_empty() {
        idx.index_batch_turbo(docs, false, false)
            .await
            .expect("bulk");
    }
    let ingest_s = t0.elapsed().as_secs_f64();
    idx.flush().await.expect("flush");
    let flush_s = t0.elapsed().as_secs_f64() - ingest_s;
    let segs_after_flush = idx.segment_rows().len();
    let t1 = Instant::now();
    let mut passes = 0;
    loop {
        let merged = idx.run_merge_once().await.expect("merge");
        if merged == 0 {
            break;
        }
        passes += 1;
    }
    let merge_s = t1.elapsed().as_secs_f64();
    let res = json!({"mode": "ingest", "docs": done, "facets": facets, "text_doc_values": text_doc_values, "ingest_s": ingest_s, "flush_s": flush_s,
                     "segments_after_flush": segs_after_flush, "merge_passes": passes,
                     "merge_s": merge_s, "segments": idx.segment_rows().len()});
    println!("{res}");
    std::fs::write(
        std::path::Path::new(&data).join("bench_ingest.json"),
        serde_json::to_string_pretty(&res).unwrap(),
    )
    .unwrap();
}

/// `YYYY-MM-DD` of day `d` (0-based) of 2025.
fn chrono_day(d: u64) -> String {
    const LEN: [u64; 12] = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    let (mut m, mut d) = (0usize, d);
    while d >= LEN[m] {
        d -= LEN[m];
        m += 1;
    }
    format!("2025-{:02}-{:02}", m + 1, d + 1)
}

/// The CLI query cut to the legs `keep` selects (indices into its `should`).
fn cut_legs(cli: &Value, keep: &dyn Fn(usize, &Value) -> bool) -> Value {
    let should = cli["query"]["bool"]["should"].as_array().unwrap();
    let kept: Vec<Value> = should
        .iter()
        .enumerate()
        .filter(|(i, leg)| keep(*i, leg))
        .map(|(_, leg)| leg.clone())
        .collect();
    json!({"bool": {"should": kept}})
}

fn is_phrase(leg: &Value) -> bool {
    leg.get("match_phrase").is_some()
}

fn is_cov(leg: &Value) -> bool {
    leg.get("constant_score").is_some()
}

/// `track_total_hits` the way the HTTP layer sends it: the ES layer injects
/// 10,000 when a body leaves it out (`es_compat`), so the user tools' bodies
/// reach the engine with it; `exact` keeps the native parser's default (exact).
fn with_tth(mut body: Value, tth: &str) -> Value {
    if tth != "exact" && body.get("track_total_hits").is_none() {
        body["track_total_hits"] = json!(tth.parse::<u64>().expect("--tth N|exact"));
    }
    body
}

fn body(shape: &str, q: &str, k: usize, mapped: &BTreeSet<String>) -> Value {
    let numeric = BTreeSet::new();
    let cli =
        xerj_common::code_query::search_query_body_for_mapping(q, k, Some(mapped), Some(&numeric));
    let walk = |query: Value| json!({"size": k, "_source": ["id"], "query": query});
    match shape {
        "cli" => cli,
        "mcp" => json!({
            "query": cli["query"].clone(), "size": k,
            "_source": ["ax_path", "ax_file", "ax_locator", "path", "line", "start_line",
                        "language", "kind", "code", "title"],
            "fields": ["_passage"]
        }),
        "cli_walk" => walk(cli["query"].clone()),
        "match" => walk(json!({"match": {"text": q}})),
        // The console's Discover request (xerj-ux search-body.js): the match,
        // 25 hits, an exact total, `terms` facets and a daily histogram.
        // Needs an index ingested with `--facets 1`.
        "discover" => json!({
            "query": {"match": {"text": q}},
            "size": 25,
            "track_total_hits": true,
            "aggs": {
                "by__index": {"terms": {"field": "_index", "size": 8}},
                "by_cat": {"terms": {"field": "cat", "size": 8}},
                "by_tag": {"terms": {"field": "tag", "size": 8}},
                "by_date": {"date_histogram": {"field": "ts", "calendar_interval": "day"}}
            }
        }),
        "leg_mm" => walk(cut_legs(&cli, &|_, leg| leg.get("multi_match").is_some())),
        "leg_phrase" => walk(cut_legs(&cli, &|_, leg| is_phrase(leg))),
        "leg_cov_half" => walk(cut_legs(&cli, &|_, leg| {
            is_cov(leg) && leg.pointer("/constant_score/filter/bool").is_some()
        })),
        "leg_cov_all" => walk(cut_legs(&cli, &|_, leg| {
            is_cov(leg) && leg.pointer("/constant_score/filter/multi_match").is_some()
        })),
        "no_phrase" => walk(cut_legs(&cli, &|_, leg| !is_phrase(leg))),
        "no_cov" => walk(cut_legs(&cli, &|_, leg| !is_cov(leg))),
        other => panic!("unknown shape {other}"),
    }
}

async fn query(a: &HashMap<String, String>, captured: Captured) {
    let data = arg(a, "data", "");
    let k: usize = arg(a, "k", "10").parse().unwrap();
    let tth = arg(a, "tth", "10000");
    let nq: usize = arg(a, "n-queries", "200").parse().unwrap();
    let warm: usize = arg(a, "warm", "20").parse().unwrap();
    let shapes: Vec<String> = arg(a, "shapes", "cli,mcp,cli_walk,match")
        .split(',')
        .map(str::to_string)
        .collect();
    let mapped: BTreeSet<String> = arg(a, "mapped", "id,text")
        .split(',')
        .map(str::to_string)
        .collect();
    let questions: Vec<String> =
        BufReader::new(std::fs::File::open(arg(a, "queries", "")).unwrap())
            .lines()
            .map(|l| {
                serde_json::from_str::<Value>(&l.unwrap()).unwrap()["question"]
                    .as_str()
                    .unwrap()
                    .to_string()
            })
            .take(nq)
            .collect();
    let t_open = Instant::now();
    let engine = xerj_engine::Engine::new(config(&data)).expect("engine");
    let idx = engine.get_index("bench").expect("index");
    let open_s = t_open.elapsed().as_secs_f64();
    let segments = idx.segment_rows().len();
    let mut per_shape = serde_json::Map::new();
    for shape in &shapes {
        // the warm questions are the LAST ones, so the timed set is the same for every shape
        for q in questions.iter().rev().take(warm) {
            let req =
                xerj_query::parse_request(&with_tth(body(shape, q, k, &mapped), &tth)).unwrap();
            let _ = idx.search(&req).await;
        }
        captured.0.lock().unwrap().clear();
        let (mut lat, mut seg_loop, mut hydrate) = (Vec::new(), Vec::new(), Vec::new());
        let (mut totals, mut empty, mut timed_out) = (Vec::new(), 0usize, 0usize);
        let mut first_ids: Vec<Value> = Vec::new();
        let (mut errors, mut with_aggs) = (0usize, 0usize);
        let mut first_error: Option<String> = None;
        for q in &questions {
            let req =
                xerj_query::parse_request(&with_tth(body(shape, q, k, &mapped), &tth)).unwrap();
            captured.0.lock().unwrap().clear();
            let t = Instant::now();
            let r = idx.search(&req).await;
            lat.push(t.elapsed().as_secs_f64() * 1e3);
            let r = match r {
                Ok(r) => r,
                Err(e) => {
                    errors += 1;
                    first_error.get_or_insert_with(|| e.to_string());
                    continue;
                }
            };
            with_aggs += usize::from(r.aggs.is_some());
            let lines = captured.0.lock().unwrap().clone();
            if let Some(line) = lines.last() {
                if std::env::var_os("BODY_BENCH_PHASES").is_some() {
                    eprintln!(
                        "[phases] {shape} {:.1} ms | {}",
                        lat[lat.len() - 1],
                        line.trim()
                    );
                }
                let ph = phases_of(line);
                seg_loop.push(*ph.get("segment_loop").unwrap_or(&0.0));
                hydrate.push(*ph.get("hydrate+corpus").unwrap_or(&0.0));
            }
            totals.push(r.total.value as f64);
            empty += usize::from(r.hits.is_empty());
            timed_out += usize::from(r.timed_out);
            first_ids.push(json!(r
                .hits
                .iter()
                .map(|h| h.id.clone())
                .collect::<Vec<_>>()));
        }
        eprintln!(
            "[query] {shape}: p50 {:.1} ms",
            stats(lat.clone())["p50"].as_f64().unwrap_or(0.0)
        );
        per_shape.insert(
            shape.clone(),
            json!({"latency_ms": stats(lat), "phase_segment_loop_ms": stats(seg_loop),
                   "phase_hydrate_ms": stats(hydrate), "total_hits": stats(totals),
                   "empty_pages": empty, "timed_out": timed_out, "errors": errors,
                   "first_error": first_error, "with_aggs": with_aggs, "pages": first_ids}),
        );
    }
    let res = json!({"mode": "query", "data": data, "segments": segments, "open_s": open_s,
                     "k": k, "track_total_hits": tth, "questions": questions.len(), "warm": warm,
                     "mapped": mapped.iter().collect::<Vec<_>>(), "shapes": per_shape});
    std::fs::write(
        arg(a, "out", "body_bench.json"),
        serde_json::to_string_pretty(&res).unwrap(),
    )
    .unwrap();
}

fn main() {
    let (mode, a) = args();
    // the engine logs every search's phase breakdown at this threshold
    std::env::set_var("XERJ_PHASE_LOG_MS", "0");
    let captured = Captured::default();
    tracing_subscriber::fmt()
        .with_writer(captured.clone())
        .with_ansi(false)
        .with_max_level(tracing::Level::WARN)
        .init();
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .enable_all()
        .build()
        .unwrap();
    rt.block_on(async {
        match mode.as_str() {
            "ingest" => ingest(&a).await,
            "query" => query(&a, captured).await,
            _ => eprintln!("usage: body_bench ingest|query ..."),
        }
    });
}
