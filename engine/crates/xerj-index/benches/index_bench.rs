//! criterion micro-benchmarks of xerj-index on synthetic data (no corpus needed). Single thread.
//! * manifest keys: 4-lane ARMv8 SHA-256 with a shared-prefix midstate vs the sha2 crate (clone midstate + finalize)
//! * entity linking: span-hash probes vs tiered Aho-Corasick over 100k subjects
//! * BM25: pruned vs plain TAAT top-16 over 200k short records
use criterion::{criterion_group, criterion_main, Criterion};
use sha2::{Digest, Sha256};
use std::hint::black_box;
use xerj_index::bm25::{Bm25Backend, Bm25Params, SearchAlgo, SegmentedBm25};
use xerj_index::linker::{Linker, LinkerKind, QBuf};
use xerj_index::sha256x::PrefixSha256;
use xerj_index::SubjectIndex;
use xerj_store::{FactIn, Store};

const SYL: [&str; 20] = [
    "ka", "ri", "vo", "len", "tha", "mor", "si", "quel", "dra", "pen", "ul", "ves", "ter", "ga",
    "lio", "nas", "bra", "cu", "fen", "ho",
];

fn name(i: usize) -> String {
    let mut x = i as u64 * 2654435761 + 7;
    let mut w = |n: usize| {
        let mut s = String::new();
        for _ in 0..n {
            s.push_str(SYL[(x % 20) as usize]);
            x /= 20;
        }
        let mut c = s.chars();
        c.next()
            .map(|f| f.to_uppercase().collect::<String>() + c.as_str())
            .unwrap_or_default()
    };
    format!("{} {} {}", w(2), w(3), i)
}

fn keys(c: &mut Criterion) {
    let item = "world-test-00042/update/before";
    let fp = "5fa3b225e59c9c169f2986627174e41dce3d006f507947b49d7b91ae273b7dc3";
    let fids: Vec<String> = (0..4096)
        .map(|i| format!("s3w{i}:mquake-t-{i:05}/f{}@v1/P159", i % 17))
        .collect();
    let mut g = c.benchmark_group("manifest_keys_4096");
    g.bench_function("sha2_crate_midstate_clone", |b| {
        let mut base = Sha256::new();
        base.update(item.as_bytes());
        base.update(fp.as_bytes());
        b.iter(|| {
            let mut acc = 0u8;
            for f in &fids {
                let mut h = base.clone();
                h.update(f.as_bytes());
                acc ^= h.finalize()[0];
            }
            black_box(acc)
        })
    });
    let p = PrefixSha256::new(&[item.as_bytes(), fp.as_bytes()]);
    g.bench_function("xerj_prefix_sha256_x1", |b| {
        b.iter(|| {
            let mut acc = 0u8;
            for f in &fids {
                acc ^= p.digest(f.as_bytes())[0];
            }
            black_box(acc)
        })
    });
    g.bench_function("xerj_prefix_sha256_x4_lanes", |b| {
        b.iter(|| {
            let mut acc = 0u8;
            for c4 in fids.as_chunks::<4>().0 {
                let d = p.digest4([
                    c4[0].as_bytes(),
                    c4[1].as_bytes(),
                    c4[2].as_bytes(),
                    c4[3].as_bytes(),
                ]);
                acc ^= d[0][0] ^ d[1][0] ^ d[2][0] ^ d[3][0];
            }
            black_box(acc)
        })
    });
    g.finish();
}

fn linking(c: &mut Criterion) {
    let mut st = Store::default();
    let n = 100_000;
    for i in 0..n {
        let s = name(i);
        let d = format!("d{i}");
        let text = format!("{s} is red.");
        let st0 = s.len() as i64 + 4;
        st.add_document(&d, "v1", &text).unwrap();
        st.add_fact(
            &FactIn::text(
                &format!("f{i}"),
                &s,
                "color",
                "red",
                (&d, "v1"),
                (st0, st0 + 3),
                "red",
            ),
            &[],
        )
        .unwrap();
    }
    let mut idx = SubjectIndex::new();
    idx.catch_up(&st);
    let mut span = Linker::new(LinkerKind::SpanHash);
    span.catch_up(&st);
    let mut ac = Linker::new(LinkerKind::AhoCorasick);
    ac.catch_up(&st);
    ac.ac.refresh(&st.pools.subjects);
    let qs: Vec<String> = (0..256)
        .map(|i| format!("Was {} founded before {}?", name(i * 37), name(i * 91 + 5)))
        .collect();
    let head = st.head();
    let mut qb = QBuf::default();
    let mut g = c.benchmark_group("link_100k_subjects");
    let mut i = 0;
    g.bench_function("span_hash", |b| {
        b.iter(|| {
            i = (i + 1) % qs.len();
            black_box(span.link_exact(&head, &idx, &qs[i], &mut qb))
        })
    });
    g.bench_function("aho_corasick", |b| {
        b.iter(|| {
            i = (i + 1) % qs.len();
            black_box(ac.link_exact(&head, &idx, &qs[i], &mut qb))
        })
    });
    g.finish();
}

fn bm25(c: &mut Criterion) {
    let mut ix = SegmentedBm25::new(Bm25Params::default());
    let rel = [
        "date of birth",
        "place of birth",
        "country of citizenship",
        "height",
        "employer",
        "spouse",
    ];
    for i in 0..200_000u32 {
        let t = format!(
            "{} {} the value {} of {}",
            name(i as usize),
            rel[(i % 6) as usize],
            i % 997,
            name((i as usize * 7) % 5000)
        );
        ix.add(i, &t);
    }
    ix.commit();
    let qs: Vec<String> = (0..256)
        .map(|i| format!("What is the {} of {}?", rel[i % 6], name(i * 613 % 200_000)))
        .collect();
    let mut g = c.benchmark_group("bm25_200k_records_top16");
    let mut i = 0;
    ix.algo = SearchAlgo::Pruned;
    g.bench_function("pruned", |b| {
        b.iter(|| {
            i = (i + 1) % qs.len();
            black_box(ix.search(&qs[i], 16))
        })
    });
    let mut ix2 = ix;
    ix2.algo = SearchAlgo::Taat;
    g.bench_function("taat", |b| {
        b.iter(|| {
            i = (i + 1) % qs.len();
            black_box(ix2.search(&qs[i], 16))
        })
    });
    g.finish();
}

criterion_group! {
    name = g;
    config = Criterion::default().sample_size(20).measurement_time(std::time::Duration::from_secs(3));
    targets = keys, linking, bm25
}
criterion_main!(g);
