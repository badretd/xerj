//! Scale benchmark of the xerj-mem engine (custom harness; criterion micro-benchmarks are in benches/micro.rs).
//!
//!   cargo bench -p xerj-exec --bench scale -- --data ~/xerj_mem_data/scale_worlds.jsonl --sizes 1000,10000,100000,1000000 \
//!       --out ~/xerj_mem_bench
//!
//! Per size N (whole worlds of the scale corpus until >= N records; single thread unless noted):
//! * ingest, SEPARATELY per component: JSON parse; record store (contract checks, interning, MVCC, key index); subject /
//!   relation / entity-id indexes + linker vocabulary; BM25 over records; BM25 over documents; then the full `Memory`
//!   (all components maintained per op, read-your-writes) with and without background merges (1 extra thread);
//!   Aho-Corasick linker build (optional linker);
//! * RAM: live heap bytes of the full memory (counting allocator) per record, and per component;
//! * time-to-searchable / write-to-visible: one new document + fact, then the first lookup / BM25 hit / manifest that
//!   contains it (exact and revision fingerprints);
//! * query latency p50 / p95 over sampled questions: entity link (span-hash, Aho-Corasick), link + manifest K=16 (SHA-256
//!   exact, xxh3 fast), lookup, BM25 records / documents top-16, executor programs (v1 lookup, v1 compare, v2 chain);
//! * exact fingerprint (first computation at head), persistence: checkpoint size / write / reload, op log size / replay.

use std::alloc::{GlobalAlloc, Layout, System};
use std::io::{BufRead, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicIsize, Ordering};
use std::time::{Duration, Instant};
use xerj_exec::episode::{Episode, Protocol};
use xerj_index::bm25::{Bm25Backend, Bm25Params, SegmentedBm25};
use xerj_index::linker::{Linker, LinkerKind, QBuf};
use xerj_index::manifest::{candidate_manifest, ManifestHash};
use xerj_index::memory::{record_text_into, RecordText};
use xerj_index::{Memory, MemoryConfig, SubjectIndex};
use xerj_store::world::{ingest_world, World};
use xerj_store::{FactIn, FingerprintMode, Store, StoreConfig, ValueIn};

struct Counting;
static LIVE: AtomicIsize = AtomicIsize::new(0);
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        LIVE.fetch_add(l.size() as isize, Ordering::Relaxed);
        System.alloc(l)
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        LIVE.fetch_sub(l.size() as isize, Ordering::Relaxed);
        System.dealloc(p, l)
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        LIVE.fetch_add(n as isize - l.size() as isize, Ordering::Relaxed);
        System.realloc(p, l, n)
    }
}
#[global_allocator]
static A: Counting = Counting;

fn live() -> isize {
    LIVE.load(Ordering::Relaxed)
}

fn pct(v: &mut [Duration], p: f64) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_unstable();
    let i = ((p / 100.0) * (v.len() - 1) as f64).round() as usize;
    v[i].as_secs_f64() * 1e6
}

fn stats(v: &mut [Duration]) -> serde_json::Value {
    serde_json::json!({"p50_us": pct(v, 50.0), "p95_us": pct(v, 95.0), "n": v.len()})
}

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1).cloned())
}

fn n_records(w: &World<'_>) -> usize {
    w.base_facts.len()
        + w.edits
            .iter()
            .map(|e| e.facts.as_ref().map_or(0, |f| f.len()) + e.fact.is_some() as usize)
            .sum::<usize>()
}

fn exact_cfg() -> MemoryConfig {
    MemoryConfig {
        store: StoreConfig {
            fingerprint: FingerprintMode::Exact,
            store_id: String::new(),
        },
        ..MemoryConfig::default()
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let home = std::env::var("HOME").unwrap_or_default();
    let data = arg(&args, "--data").unwrap_or(format!("{home}/xerj_mem_data/scale_worlds.jsonl"));
    let sizes: Vec<usize> = arg(&args, "--sizes")
        .unwrap_or("1000,10000,100000,1000000".into())
        .split(',')
        .map(|s| s.parse().unwrap())
        .collect();
    let out = PathBuf::from(arg(&args, "--out").unwrap_or(format!("{home}/xerj_mem_bench")));
    let nq: usize = arg(&args, "--questions").map_or(1000, |s| s.parse().unwrap());
    std::fs::create_dir_all(&out).unwrap();
    let max_n = *sizes.iter().max().unwrap();
    // read the corpus prefix once
    let t0 = Instant::now();
    let mut lines: Vec<String> = Vec::new();
    let mut prefix_records: Vec<usize> = Vec::new();
    {
        let rd = std::io::BufReader::with_capacity(
            1 << 22,
            std::fs::File::open(&data).expect("scale corpus"),
        );
        let mut total = 0;
        for line in rd.lines() {
            let line = line.unwrap();
            let w = World::parse(&line).unwrap();
            total += n_records(&w);
            drop(w);
            lines.push(line);
            prefix_records.push(total);
            if total >= max_n {
                break;
            }
        }
    }
    eprintln!(
        "read {} worlds in {:.1}s",
        lines.len(),
        t0.elapsed().as_secs_f64()
    );
    let mut all = Vec::new();
    for &n in &sizes {
        let nw = prefix_records
            .iter()
            .position(|&t| t >= n)
            .map_or(lines.len(), |i| i + 1);
        let lines = &lines[..nw];
        let records = prefix_records[nw - 1];
        let bytes: usize = lines.iter().map(|l| l.len() + 1).sum();
        eprintln!(
            "== N={n}: {nw} worlds, {records} records, {:.1} MB JSON",
            bytes as f64 / 1e6
        );
        let mut row = serde_json::json!({"N": n, "worlds": nw, "records": records, "json_mb": bytes as f64 / 1e6});

        // ---------------------------------------------------------------- parse
        let t = Instant::now();
        let worlds: Vec<World<'_>> = lines.iter().map(|l| World::parse(l).unwrap()).collect();
        let t_parse = t.elapsed().as_secs_f64();
        row["parse_s"] = t_parse.into();
        row["parse_records_per_s"] = (records as f64 / t_parse).into();

        // ---------------------------------------------------------------- store only
        let t = Instant::now();
        let mut store = Store::new(StoreConfig::default());
        for w in &worlds {
            ingest_world(&mut store, w).unwrap();
        }
        let t_store = t.elapsed().as_secs_f64();
        row["store_ingest_s"] = t_store.into();
        row["store_records_per_s"] = (records as f64 / t_store).into();
        row["docs"] = store.num_docs().into();
        row["subjects"] = store.pools.subjects.len().into();
        row["relations"] = store.pools.relations.len().into();
        row["head_visible"] = store.num_head_visible().into();
        row["doc_text_mb"] = (store.docs.text.len() as f64 / 1e6).into();
        row["store_heap_mb"] = (store.heap_bytes() as f64 / 1e6).into();

        // the same with capacity reserved up front (a bulk load knows its size)
        {
            let docs_n = store.num_docs();
            let text_b = store.docs.text.len();
            let t = Instant::now();
            let mut st2 = Store::new(StoreConfig::default());
            st2.reserve(records, docs_n, text_b);
            for w in &worlds {
                ingest_world(&mut st2, w).unwrap();
            }
            row["store_reserved_records_per_s"] =
                (records as f64 / t.elapsed().as_secs_f64()).into();
        }

        // ---------------------------------------------------------------- subject / relation / entity indexes + linker vocabulary
        let t = Instant::now();
        let mut sidx = SubjectIndex::new();
        sidx.catch_up(&store);
        let mut lk = Linker::new(LinkerKind::SpanHash);
        lk.catch_up(&store);
        let t_idx = t.elapsed().as_secs_f64();
        row["subject_index_s"] = t_idx.into();
        row["subject_index_records_per_s"] = (records as f64 / t_idx).into();
        drop(sidx);

        // ---------------------------------------------------------------- BM25 records / documents (each alone)
        let mut buf = String::new();
        let t = Instant::now();
        let mut b = SegmentedBm25::new(Bm25Params::default());
        for r in 0..store.num_records() as u32 {
            record_text_into(&store, r, RecordText::Rich, &mut buf);
            b.add(r, &buf);
        }
        for r in 0..store.num_records() as u32 {
            if !store.head_visible(r) {
                record_text_into(&store, r, RecordText::Rich, &mut buf);
                b.delete(r, &buf);
            }
        }
        let t_add = t.elapsed().as_secs_f64();
        b.commit();
        let t_bm25r = t.elapsed().as_secs_f64();
        row["bm25_records_s"] = t_bm25r.into();
        row["bm25_records_searchable_s"] = t_add.into();
        row["bm25_records_per_s"] = (records as f64 / t_bm25r).into();
        row["bm25_records_stats"] = serde_json::to_value(b.stats()).unwrap();
        drop(b);
        let t = Instant::now();
        let mut b = SegmentedBm25::new(Bm25Params::default());
        let mut doc_bytes = 0usize;
        for d in 0..store.num_docs() as u32 {
            let tx = store.doc_text(d);
            doc_bytes += tx.len();
            b.add(d, tx);
        }
        for d in 0..store.num_docs() as u32 {
            if !store.doc_head_visible(d) {
                b.delete(d, store.doc_text(d));
            }
        }
        b.commit();
        let t_bm25d = t.elapsed().as_secs_f64();
        row["bm25_docs_s"] = t_bm25d.into();
        row["bm25_docs_per_s"] = (store.num_docs() as f64 / t_bm25d).into();
        row["bm25_docs_mb_per_s"] = (doc_bytes as f64 / 1e6 / t_bm25d).into();
        drop(b);

        // ---------------------------------------------------------------- Aho-Corasick linker build (optional linker)
        let t = Instant::now();
        let ac = Linker::new(LinkerKind::AhoCorasick);
        ac.ac.refresh(&store.pools.subjects);
        row["ac_linker_build_s"] = t.elapsed().as_secs_f64().into();
        row["ac_linker_bytes"] = ac.ac.stats().3.into();
        drop(ac);
        drop(store);

        // ---------------------------------------------------------------- full memory, per-op maintenance
        let h0 = live();
        let t = Instant::now();
        let mut mem = Memory::new(exact_cfg());
        for w in &worlds {
            mem.ingest_world(w).unwrap();
        }
        let t_full_visible = t.elapsed().as_secs_f64();
        mem.commit().unwrap();
        let t_full = t.elapsed().as_secs_f64();
        let heap = (live() - h0) as f64;
        row["memory_ingest_s"] = t_full.into();
        row["memory_ingest_searchable_s"] = t_full_visible.into();
        row["memory_records_per_s"] = (records as f64 / t_full).into();
        row["heap_bytes_per_record"] = (heap / records as f64).into();
        row["heap_mb"] = (heap / 1e6).into();
        row["heap_breakdown_mb"] = {
            let hb = mem.heap_bytes();
            serde_json::json!({"store": hb.store as f64 / 1e6, "subject_index": hb.subject_index as f64 / 1e6,
                "bm25_records": hb.bm25_records as f64 / 1e6, "bm25_documents": hb.bm25_documents as f64 / 1e6})
        };
        {
            let h1 = live();
            mem.shrink_to_fit();
            let shrunk = (live() - h0) as f64;
            row["heap_bytes_per_record_shrunk"] = (shrunk / records as f64).into();
            row["shrink_freed_mb"] = ((h1 - live()) as f64 / 1e6).into();
        }
        {
            // asynchronous BM25 (one indexer thread); the commit includes the final read barrier
            let t = Instant::now();
            let mut m3 = Memory::new(MemoryConfig {
                async_bm25: true,
                ..exact_cfg()
            });
            for w in &worlds {
                m3.ingest_world(w).unwrap();
            }
            let tv = t.elapsed().as_secs_f64();
            m3.commit().unwrap();
            let ta = t.elapsed().as_secs_f64();
            row["memory_async_writer_s"] = tv.into();
            row["memory_async_ingest_s"] = ta.into();
            row["memory_async_records_per_s"] = (records as f64 / ta).into();
            // same answers as inline
            if let Some(&(_, q)) = worlds
                .iter()
                .flat_map(|w| {
                    w.items
                        .iter()
                        .map(|i| (i.item_id.as_ref(), i.question.as_ref()))
                })
                .collect::<Vec<(&str, &str)>>()
                .first()
            {
                assert_eq!(m3.search_records(q, 16), mem.search_records(q, 16));
                assert_eq!(m3.search_documents(q, 16), mem.search_documents(q, 16));
            }
        }
        {
            let t = Instant::now();
            let mut m2 = Memory::new(MemoryConfig {
                background_merges: true,
                ..exact_cfg()
            });
            for w in &worlds {
                m2.ingest_world(w).unwrap();
            }
            let tv = t.elapsed().as_secs_f64();
            m2.commit().unwrap();
            row["memory_bg_merge_searchable_s"] = tv.into();
            row["memory_bg_merge_ingest_s"] = t.elapsed().as_secs_f64().into();
        }

        // ---------------------------------------------------------------- fingerprints
        let head = mem.head();
        let t = Instant::now();
        let fp = head.fingerprint();
        row["exact_fingerprint_first_s"] = t.elapsed().as_secs_f64().into();
        let _ = fp;

        // ---------------------------------------------------------------- queries
        let items: Vec<(&str, &str)> = worlds
            .iter()
            .flat_map(|w| {
                w.items
                    .iter()
                    .map(|i| (i.item_id.as_ref(), i.question.as_ref()))
            })
            .collect();
        let step = (items.len() / nq.max(1)).max(1);
        let qs: Vec<(&str, &str)> = items.iter().step_by(step).take(nq).copied().collect();
        let mut qb = QBuf::default();
        let mut t_link = Vec::new();
        let mut t_link_ac = Vec::new();
        let mut t_man = Vec::new();
        let mut t_man_fast = Vec::new();
        let mut t_man_sampled = Vec::new();
        let mut t_bm_r = Vec::new();
        let mut t_bm_d = Vec::new();
        let mut t_exec1 = Vec::new();
        let mut t_exec2 = Vec::new();
        let mut t_exec3 = Vec::new();
        let mut keys_hashed = 0usize;
        let mut linked_any = 0usize;
        let mut ac = Linker::new(LinkerKind::AhoCorasick);
        ac.catch_up(&mem.store);
        ac.ac.refresh(&mem.store.pools.subjects);
        let fps = head.fingerprint();
        // warm-up (lazy normalized index for CHAIN, scratch buffers)
        if let Some(&(id, q)) = qs.first() {
            let c = mem.candidate_manifest(&head, id, q, 16, &mut qb);
            let mut ep = Episode::new(&mem, head, Protocol::V2, c, id, 16);
            let _ = ep.step(0, Some(0));
            let _ = ep.current_hop();
        }
        for &(id, q) in &qs {
            let t = Instant::now();
            let ents = mem.link(&head, q, &mut qb);
            t_link.push(t.elapsed());
            linked_any += !ents.is_empty() as usize;
            let t = Instant::now();
            std::hint::black_box(ac.link_exact(&head, &mem.subjects, q, &mut qb));
            t_link_ac.push(t.elapsed());
            let t = Instant::now();
            let (cands, st) = mem.candidate_manifest_stats(&head, id, q, 16, &mut qb);
            t_man.push(t.elapsed());
            keys_hashed += st.keys_hashed;
            let t = Instant::now();
            let e2 = mem.link(&head, q, &mut qb);
            std::hint::black_box(candidate_manifest(
                &head,
                &mem.subjects,
                &e2,
                id,
                &fps,
                16,
                ManifestHash::Fast,
                None,
            ));
            t_man_fast.push(t.elapsed());
            let t = Instant::now();
            let e3 = mem.link(&head, q, &mut qb);
            std::hint::black_box(candidate_manifest(
                &head,
                &mem.subjects,
                &e3,
                id,
                &fps,
                16,
                ManifestHash::Sampled,
                None,
            ));
            t_man_sampled.push(t.elapsed());
            let t = Instant::now();
            std::hint::black_box(mem.search_records(q, 16));
            t_bm_r.push(t.elapsed());
            let t = Instant::now();
            std::hint::black_box(mem.search_documents(q, 16));
            t_bm_d.push(t.elapsed());
            // executor programs over the manifest
            let t = Instant::now();
            let mut ep = Episode::new(&mem, head, Protocol::V1, cands.clone(), id, 16);
            let _ = ep.step(0, Some(0));
            let _ = ep.step(5, None);
            std::hint::black_box(&ep.answer);
            t_exec1.push(t.elapsed());
            let t = Instant::now();
            let mut ep = Episode::new(&mem, head, Protocol::V1, cands.clone(), id, 16);
            let _ = ep.step(0, Some(0));
            let _ = ep.step(0, Some(1));
            let _ = ep.step(1, None);
            let _ = ep.step(5, None);
            std::hint::black_box(&ep.answer);
            t_exec2.push(t.elapsed());
            let t = Instant::now();
            let mut ep = Episode::new(&mem, head, Protocol::V2, cands, id, 16);
            let _ = ep.step(0, Some(0));
            let _ = ep.step(7, Some(0));
            let _ = ep.step(5, None);
            std::hint::black_box(&ep.answer);
            t_exec3.push(t.elapsed());
        }
        // lookups of random existing keys
        let mut t_look = Vec::new();
        let nrec = mem.store.num_records() as u64;
        let mut x = 0x9E37_79B9_7F4A_7C15u64;
        for _ in 0..2000 {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            let r = (x % nrec) as u32;
            let (s, rl, sc) = (
                mem.store.subject(r).to_string(),
                mem.store.relation(r).to_string(),
                mem.store.scope(r).to_string(),
            );
            let t = Instant::now();
            std::hint::black_box(head.lookup(&s, &rl, &sc));
            t_look.push(t.elapsed());
        }
        // the same questions with the plain TAAT BM25 (reference path), interleaved in this run for a noise-free comparison
        {
            let (mut tr, mut td) = (Vec::new(), Vec::new());
            let (mut pr, mut pd) = (Vec::new(), Vec::new());
            for &(_, q) in &qs {
                mem.set_bm25_algo(xerj_index::bm25::SearchAlgo::Taat);
                let t = Instant::now();
                let a = mem.search_records(q, 16);
                tr.push(t.elapsed());
                let t = Instant::now();
                let b2 = mem.search_documents(q, 16);
                td.push(t.elapsed());
                mem.set_bm25_algo(xerj_index::bm25::SearchAlgo::Pruned);
                let t = Instant::now();
                let a2 = mem.search_records(q, 16);
                pr.push(t.elapsed());
                let t = Instant::now();
                let b3 = mem.search_documents(q, 16);
                pd.push(t.elapsed());
                assert_eq!(a, a2, "pruned BM25 differs from TAAT on {q:?}");
                assert_eq!(b2, b3, "pruned BM25 differs from TAAT on {q:?}");
            }
            row["bm25_records_q_taat"] = stats(&mut tr);
            row["bm25_docs_q_taat"] = stats(&mut td);
            row["bm25_records_q_pruned_paired"] = stats(&mut pr);
            row["bm25_docs_q_pruned_paired"] = stats(&mut pd);
        }
        row["questions"] = qs.len().into();
        row["questions_linked"] = linked_any.into();
        row["manifest_keys_hashed_mean"] = (keys_hashed as f64 / qs.len().max(1) as f64).into();
        row["link_spanhash"] = stats(&mut t_link);
        row["link_ac"] = stats(&mut t_link_ac);
        row["manifest_exact"] = stats(&mut t_man);
        row["manifest_fast"] = stats(&mut t_man_fast);
        row["manifest_sampled"] = stats(&mut t_man_sampled);
        row["lookup"] = stats(&mut t_look);
        row["bm25_records_q"] = stats(&mut t_bm_r);
        row["bm25_docs_q"] = stats(&mut t_bm_d);
        row["exec_lookup_return"] = stats(&mut t_exec1);
        row["exec_compare"] = stats(&mut t_exec2);
        row["exec_chain_v2"] = stats(&mut t_exec3);

        // ---------------------------------------------------------------- write -> visible
        // one new document + fact per repetition; "write_to_X" = the write's duration + the duration of the first X query
        // after it (nothing is rebuilt in between; exact fingerprints must be recomputed for the new revision)
        let mut extra_records = 0usize;
        for (mode, label) in [
            (FingerprintMode::Exact, "exact"),
            (FingerprintMode::Revision, "revision"),
        ] {
            mem.store.set_fingerprint_mode(mode);
            let reps = if mode == FingerprintMode::Exact && records > 200_000 {
                5
            } else {
                100
            };
            let (mut w_look, mut w_bm, mut w_man, mut w_write) =
                (Vec::new(), Vec::new(), Vec::new(), Vec::new());
            for i in 0..reps {
                let subj = format!("Zqx{label}{i} Benchmark Entity");
                let doc = format!("bench-{label}-{i}");
                let v = format!("1{i}");
                let text = format!("{subj} is {v} m tall.");
                let start = (subj.chars().count() + 4) as i64;
                let fid = format!("{doc}@v1/height");
                let q = format!("What is the height of {subj}?");
                let t = Instant::now();
                mem.add_document(&doc, "v1", &text).unwrap();
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
                let a = mem.add_fact(&f, &[]).unwrap();
                let tw = t.elapsed();
                extra_records += 1;
                w_write.push(tw);
                let snap = mem.head();
                let t = Instant::now();
                let l = snap.lookup(&subj, "height", "default");
                assert_eq!(l.refs.as_slice(), &[a.rec]);
                w_look.push(tw + t.elapsed());
                let t = Instant::now();
                let hits = mem.search_records(&subj, 4);
                assert_eq!(hits.first().map(|h| h.doc), Some(a.rec));
                w_bm.push(tw + t.elapsed());
                let t = Instant::now();
                let m = mem.candidate_manifest(&snap, &doc, &q, 16, &mut qb);
                assert!(m.contains(&a.rec));
                w_man.push(tw + t.elapsed());
            }
            row[format!("write_{label}")] = stats(&mut w_write);
            row[format!("write_to_lookup_{label}")] = stats(&mut w_look);
            row[format!("write_to_bm25_{label}")] = stats(&mut w_bm);
            row[format!("write_to_manifest_{label}")] = stats(&mut w_man);
        }
        mem.store.set_fingerprint_mode(FingerprintMode::Exact);

        // ---------------------------------------------------------------- persistence
        let dir = out.join(format!("ckpt_{n}"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let ck = dir.join("memory.ckp");
        let t = Instant::now();
        let size = mem.checkpoint_to(&ck, 0).unwrap();
        row["checkpoint_write_s"] = t.elapsed().as_secs_f64().into();
        row["checkpoint_mb"] = (size as f64 / 1e6).into();
        drop(mem);
        let t = Instant::now();
        let m = Memory::load_checkpoint(&ck, exact_cfg()).unwrap();
        row["checkpoint_reload_s"] = t.elapsed().as_secs_f64().into();
        assert_eq!(m.store.num_records(), records + extra_records);
        drop(m);
        // op log: ingest through a logged memory, then reopen from the log only
        let ldir = dir.join("logged");
        {
            let mut lm = Memory::open(&ldir, exact_cfg(), false).unwrap();
            for w in &worlds {
                lm.ingest_world(w).unwrap();
            }
            lm.commit().unwrap();
        }
        let log_bytes = std::fs::metadata(ldir.join("ops.log")).unwrap().len();
        let t = Instant::now();
        let lm = Memory::open(&ldir, exact_cfg(), false).unwrap();
        row["log_replay_s"] = t.elapsed().as_secs_f64().into();
        row["log_mb"] = (log_bytes as f64 / 1e6).into();
        assert_eq!(lm.store.num_records(), records);
        drop(lm);
        let _ = std::fs::remove_dir_all(&dir);
        eprintln!("{}", serde_json::to_string(&row).unwrap());
        all.push(row);
        let mut f = std::fs::File::create(out.join("scale_results.json")).unwrap();
        writeln!(f, "{}", serde_json::to_string_pretty(&all).unwrap()).unwrap();
    }
}
