//! BM25 diagnostics: pruned vs TAAT per question on the first N records of the scale corpus; prints the slowest ratios.
//!   cargo run --release -p xerj-exec --example bm25_diag -- 100000
use std::io::BufRead;
use std::time::Instant;
use xerj_index::bm25::{Bm25Backend, Bm25Params, SearchAlgo, SegmentedBm25};
use xerj_index::memory::{record_text_into, RecordText};
use xerj_store::world::{ingest_world, World};
use xerj_store::Store;

fn main() {
    let n: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(100_000);
    let home = std::env::var("HOME").unwrap();
    let rd = std::io::BufReader::new(
        std::fs::File::open(format!("{home}/xerj_mem_data/scale_worlds.jsonl")).unwrap(),
    );
    let (mut lines, mut total) = (Vec::new(), 0usize);
    for l in rd.lines() {
        let l = l.unwrap();
        total += World::parse(&l).unwrap().all_facts().count();
        lines.push(l);
        if total >= n {
            break;
        }
    }
    let worlds: Vec<World<'_>> = lines.iter().map(|l| World::parse(l).unwrap()).collect();
    let mut st = Store::default();
    for w in &worlds {
        ingest_world(&mut st, w).unwrap();
    }
    let mut b = SegmentedBm25::new(Bm25Params::default());
    let mut buf = String::new();
    for r in 0..st.num_records() as u32 {
        if st.head_visible(r) {
            record_text_into(&st, r, RecordText::Rich, &mut buf);
            b.add(r, &buf);
        }
    }
    b.commit();
    let qs: Vec<String> = worlds
        .iter()
        .flat_map(|w| w.items.iter().map(|i| i.question.to_string()))
        .step_by(4)
        .take(1000)
        .collect();
    let mut rows = Vec::new();
    for q in &qs {
        b.algo = SearchAlgo::Taat;
        let t = Instant::now();
        let a = b.search_with_ties(q, 16);
        let tt = t.elapsed().as_secs_f64() * 1e6;
        b.algo = SearchAlgo::Pruned;
        let t = Instant::now();
        let c = b.search_with_ties(q, 16);
        let tp = t.elapsed().as_secs_f64() * 1e6;
        assert_eq!(a, c);
        let (_, s) = b.explain(q, 16);
        rows.push((tp / tt, tp, tt, q.clone(), s));
    }
    rows.sort_by(|a, b| b.0.total_cmp(&a.0));
    for r in rows.iter().take(15) {
        println!(
            "ratio {:.2} pruned {:.0}us taat {:.0}us {:?} {:?}",
            r.0, r.1, r.2, r.3, r.4
        );
    }
    let mut tp: Vec<f64> = rows.iter().map(|r| r.1).collect();
    let mut tt: Vec<f64> = rows.iter().map(|r| r.2).collect();
    tp.sort_by(|a, b| a.total_cmp(b));
    tt.sort_by(|a, b| a.total_cmp(b));
    let p = |v: &[f64], x: f64| v[((v.len() - 1) as f64 * x) as usize];
    println!(
        "pruned p50 {:.0} p95 {:.0} | taat p50 {:.0} p95 {:.0}",
        p(&tp, 0.5),
        p(&tp, 0.95),
        p(&tt, 0.5),
        p(&tt, 0.95)
    );
}
