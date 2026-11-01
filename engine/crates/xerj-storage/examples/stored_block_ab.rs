//! Same-segment A/B of the stored encodings on a real corpus: one stored section built from a
//! whole JSONL file, as a fully merged segment holds it. Bytes by column, encode CPU, and the time
//! to read the rows of a search page from that section.
//!
//! ```text
//! XERJ_STORED_ROW_BLOCKS=1 stored_block_ab CORPUS.jsonl flush|merge|force [MERGE_LEVEL] [PAGES] [SEED]
//! ```
//!
//! Blocks and dictionary are chosen by `XERJ_STORED_ROW_BLOCKS` and `XERJ_STORED_BLOCK_DICT`, which
//! each process reads once. rc.100's encoding is `XERJ_STORED_ROW_BLOCKS=0 ... merge 3`.
use rand::{rngs::StdRng, Rng, SeedableRng};
use serde_json::{json, Map, Value};
use std::io::BufRead;
use std::time::Instant;
use xerj_storage::stored_codec::{
    decode_stored, decode_stored_v2_rows, encode_stored_v2, encode_stored_v2_forcemerge_at_level,
    encode_stored_v2_merge_at_level, stored_v2_column_sizes, StoredV2RowHydrationResult,
};

fn cpu_seconds() -> f64 {
    let stat = std::fs::read_to_string("/proc/self/stat").unwrap_or_default();
    let tail = stat.rsplit_once(')').map_or("", |t| t.1);
    let f: Vec<&str> = tail.split_whitespace().collect();
    let tick = |i: usize| f.get(i).and_then(|v| v.parse::<f64>().ok()).unwrap_or(0.0);
    (tick(11) + tick(12)) / 100.0
}

fn pct(sorted: &[f64], p: f64) -> f64 {
    sorted[((sorted.len() - 1) as f64 * p).round() as usize]
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let mode = a[2].as_str();
    let level: i32 = a.get(3).map_or(3, |s| s.parse().unwrap());
    let pages: usize = a.get(4).map_or(200, |s| s.parse().unwrap());
    let seed: u64 = a.get(5).map_or(29, |s| s.parse().unwrap());

    let file = std::fs::File::open(&a[1]).unwrap();
    let mut docs: Vec<Value> = Vec::new();
    let reader = std::io::BufReader::with_capacity(1 << 24, file);
    for (i, line) in reader.lines().enumerate() {
        let source: Map<String, Value> = serde_json::from_str(&line.unwrap()).unwrap();
        docs.push(json!({ "_id": format!("d{i}"), "_seq_no": i as u64, "_source": source }));
    }
    let n = docs.len();
    let body = serde_json::to_vec(&docs).unwrap();
    drop(docs);

    let (t, c) = (Instant::now(), cpu_seconds());
    let section = match mode {
        "flush" => encode_stored_v2(&body),
        "merge" => encode_stored_v2_merge_at_level(&body, level),
        "force" => encode_stored_v2_forcemerge_at_level(&body, level),
        m => panic!("unknown mode {m}"),
    };
    let (wall, cpu) = (t.elapsed().as_secs_f64(), cpu_seconds() - c);

    let columns = stored_v2_column_sizes(&section).expect("columnar");
    let (mut full_ms, mut rows_us) = (0.0, Vec::with_capacity(pages));
    let t = Instant::now();
    let decoded = decode_stored(&section).unwrap();
    full_ms += t.elapsed().as_secs_f64() * 1e3;
    assert!(decoded == body, "the section does not decode to its input");
    drop(decoded);
    let mut rng = StdRng::seed_from_u64(seed);
    for _ in 0..pages {
        let mut page: Vec<usize> = (0..25).map(|_| rng.gen_range(0..n)).collect();
        page.sort_unstable();
        page.dedup();
        let t = Instant::now();
        let r = decode_stored_v2_rows(&section, &page).unwrap();
        rows_us.push(t.elapsed().as_secs_f64() * 1e6);
        assert!(matches!(r, StoredV2RowHydrationResult::Hydrated { .. }));
    }
    rows_us.sort_by(f64::total_cmp);
    let out = json!({
        "docs": n, "input_bytes": body.len(), "mode": mode, "merge_level": level,
        "row_blocks": std::env::var("XERJ_STORED_ROW_BLOCKS").ok(),
        "dict": std::env::var("XERJ_STORED_BLOCK_DICT").ok(),
        "section_bytes": section.len(),
        "ratio": body.len() as f64 / section.len() as f64,
        "encode_wall_s": wall, "encode_cpu_s": cpu,
        "columns": columns.iter().map(|c| json!({
            "name": c.name, "codec": c.codec, "bytes": c.payload_bytes
        })).collect::<Vec<_>>(),
        "full_decode_ms": full_ms,
        "page25_us": {
            "p50": pct(&rows_us, 0.5), "p95": pct(&rows_us, 0.95), "p99": pct(&rows_us, 0.99)
        },
    });
    println!("{out}");
}
