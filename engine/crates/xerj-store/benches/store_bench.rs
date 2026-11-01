//! criterion micro-benchmarks of the record store (single thread): ingest of synthetic facts, lookups, snapshot creation.
use criterion::{criterion_group, criterion_main, Criterion};
use std::hint::black_box;
use xerj_store::{FactIn, Store, StoreConfig, ValueIn};

fn build(n: usize) -> Store {
    let mut s = Store::new(StoreConfig::default());
    for i in 0..n {
        let doc = format!("doc{i}");
        let text = format!("Entity {i} is {} m tall.", i % 900 + 10);
        s.add_document(&doc, "v1", &text).unwrap();
        let v = format!("{}", i % 900 + 10);
        let start = text.find(" is ").unwrap() as i64 + 4;
        let fid = format!("f{i}");
        let subj = format!("Entity {i}");
        let f = FactIn {
            kind: "number",
            unit: "m",
            value: ValueIn::Str(&v),
            ..FactIn::text(
                &fid,
                &subj,
                "height",
                &v,
                (&doc, "v1"),
                (start, start + v.len() as i64),
                &v,
            )
        };
        s.add_fact(&f, &[]).unwrap();
    }
    s
}

fn benches(c: &mut Criterion) {
    c.bench_function("store/ingest_10k_doc+fact", |b| {
        b.iter(|| black_box(build(10_000).num_records()))
    });
    let s = build(100_000);
    let mut i = 0usize;
    c.bench_function("store/lookup_head_100k", |b| {
        b.iter(|| {
            i = (i + 7919) % 100_000;
            let subj = format!("Entity {i}");
            black_box(s.head().lookup(&subj, "height", "default").status)
        })
    });
    c.bench_function("store/snapshot_before_after", |b| {
        b.iter(|| {
            black_box(
                s.snapshot("2026-10-20", Some(s.revision() / 2))
                    .unwrap()
                    .rev,
            )
        })
    });
}

criterion_group!(g, benches);
criterion_main!(g);
