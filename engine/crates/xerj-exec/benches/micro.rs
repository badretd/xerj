//! criterion micro-benchmarks on a memory built from the first XERJ_MICRO_N (default 100 000) records of the scale corpus
//! ($XERJ_SCALE_DATA, default ~/xerj_mem_data/scale_worlds.jsonl). Single thread.
use criterion::{criterion_group, criterion_main, Criterion};
use std::hint::black_box;
use std::io::BufRead;
use xerj_exec::episode::{Episode, Protocol};
use xerj_index::linker::QBuf;
use xerj_index::{Memory, MemoryConfig};
use xerj_store::world::World;
use xerj_store::{FactIn, ValueIn};

fn load(n: usize) -> (Vec<String>, usize) {
    let home = std::env::var("HOME").unwrap_or_default();
    let path = std::env::var("XERJ_SCALE_DATA")
        .unwrap_or(format!("{home}/xerj_mem_data/scale_worlds.jsonl"));
    let rd = std::io::BufReader::new(std::fs::File::open(path).expect("scale corpus"));
    let (mut lines, mut total) = (Vec::new(), 0);
    for l in rd.lines() {
        let l = l.unwrap();
        let w = World::parse(&l).unwrap();
        total += w.all_facts().count();
        drop(w);
        lines.push(l);
        if total >= n {
            break;
        }
    }
    (lines, total)
}

fn benches(c: &mut Criterion) {
    let n: usize = std::env::var("XERJ_MICRO_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(100_000);
    let (lines, _) = load(n);
    let worlds: Vec<World<'_>> = lines.iter().map(|l| World::parse(l).unwrap()).collect();
    let mut mem = Memory::new(MemoryConfig::default());
    for w in &worlds {
        mem.ingest_world(w).unwrap();
    }
    mem.commit().unwrap();
    let qs: Vec<(String, String)> = worlds
        .iter()
        .flat_map(|w| {
            w.items
                .iter()
                .map(|i| (i.item_id.to_string(), i.question.to_string()))
        })
        .step_by(97)
        .take(256)
        .collect();
    let head = mem.head();
    let _ = head.fingerprint();
    let mut qb = QBuf::default();
    let mut g = c.benchmark_group(format!("xerj_mem_{n}"));
    g.sample_size(30)
        .measurement_time(std::time::Duration::from_secs(4));
    let mut i = 0usize;
    g.bench_function("link_spanhash", |b| {
        b.iter(|| {
            i = (i + 1) % qs.len();
            black_box(mem.link(&head, &qs[i].1, &mut qb))
        })
    });
    g.bench_function("link_plus_manifest_k16_exact", |b| {
        b.iter(|| {
            i = (i + 1) % qs.len();
            black_box(mem.candidate_manifest(&head, &qs[i].0, &qs[i].1, 16, &mut qb))
        })
    });
    let r0 = 12_345 % mem.store.num_records() as u32;
    let (s, r, sc) = (
        mem.store.subject(r0).to_string(),
        mem.store.relation(r0).to_string(),
        mem.store.scope(r0).to_string(),
    );
    g.bench_function("lookup", |b| b.iter(|| black_box(head.lookup(&s, &r, &sc))));
    g.bench_function("bm25_records_top16", |b| {
        b.iter(|| {
            i = (i + 1) % qs.len();
            black_box(mem.search_records(&qs[i].1, 16))
        })
    });
    g.bench_function("bm25_documents_top16", |b| {
        b.iter(|| {
            i = (i + 1) % qs.len();
            black_box(mem.search_documents(&qs[i].1, 16))
        })
    });
    let cands: Vec<Vec<u32>> = qs
        .iter()
        .map(|(id, q)| mem.candidate_manifest(&head, id, q, 16, &mut qb))
        .collect();
    g.bench_function("exec_v2_lookup_chain_return", |b| {
        b.iter(|| {
            i = (i + 1) % qs.len();
            let mut ep = Episode::new(&mem, head, Protocol::V2, cands[i].clone(), &qs[i].0, 16);
            let _ = ep.step(0, Some(0));
            let _ = ep.step(7, Some(0));
            let _ = ep.step(5, None);
            black_box(ep.answer)
        })
    });
    g.finish();
    let mut k = 0u64;
    c.bench_function(&format!("xerj_mem_{n}/write_doc_plus_fact"), |b| {
        b.iter(|| {
            k += 1;
            let doc = format!("micro-{k}");
            let subj = format!("Micro Entity {k}");
            let text = format!("{subj} is 7 m tall.");
            let st = subj.len() as i64 + 4;
            let fid = format!("{doc}@v1/h");
            mem.add_document(&doc, "v1", &text).unwrap();
            let f = FactIn {
                kind: "number",
                unit: "m",
                value: ValueIn::Str("7"),
                ..FactIn::text(&fid, &subj, "height", "7", (&doc, "v1"), (st, st + 1), "7")
            };
            black_box(mem.add_fact(&f, &[]).unwrap())
        })
    });
}

criterion_group!(g, benches);
criterion_main!(g);
