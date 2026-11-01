//! Parity with the frozen Python reference (research/xerj_r_mvp/tools/xerj_mem_parity_export.py fixtures).
//!
//! XERJ_PARITY_DIR (default $HOME/xerj_mem_parity) holds `<set>.parity.jsonl`; every world line is ingested into a fresh
//! `Memory` (exact fingerprints, SHA-256 manifests, reference BM25 text) and every exported reference output is
//! recomputed and compared: revisions, fingerprints, lookups (status / value / refs / reason), link_entities (span-hash
//! AND Aho-Corasick linkers, items and synthetic stress questions), candidate manifests at K = 16 / 4 / 64, gold-trace
//! executions (answer record incl. program and execution trace, problems, every hop manifest read, final reason), random
//! and mutated programs, the verify verdict, BM25 top-16 (LexicalIndex), normalize_entity and normalize_answer.
//! The test fails on any mismatch and prints a per-set table (also written to $XERJ_PARITY_DIR/RUST_PARITY.md).

use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashMap};
use std::io::BufRead;
use std::path::PathBuf;
use xerj_exec::episode::{run_gold, Answer, Episode, GoldStep, Protocol};
use xerj_exec::scorer::normalize_answer;
use xerj_exec::verify::{verify, VerifyMode};
use xerj_index::linker::{Linker, LinkerKind, QBuf};
use xerj_index::{Memory, MemoryConfig, RecordText};
use xerj_store::text::normalize_entity;
use xerj_store::world::{World, WriteSink};
use xerj_store::{Rev, Snapshot, StoreConfig};

#[derive(Deserialize)]
struct Fps {
    before: String,
    after: String,
}

#[derive(Deserialize)]
struct LookupJ {
    status: String,
    value: Option<(String, Value, String)>,
    refs: Vec<String>,
    reason: String,
}

#[derive(Deserialize)]
struct GoldJ {
    trace: Vec<Value>,
}

#[derive(Deserialize)]
struct HopJ {
    before_step: usize,
    fact_ids: Vec<String>,
    resolves: Vec<bool>,
    anchor: Option<String>,
}

#[derive(Deserialize)]
struct ExecJ {
    answer: Value,
    problems: Vec<String>,
    hops: Vec<HopJ>,
    top_reason: String,
    verify_vetoed: bool,
}

#[derive(Deserialize)]
struct ProgJ {
    actions: Vec<(i64, Option<i64>)>,
    answer: Value,
}

#[derive(Deserialize)]
struct ItemJ {
    item_id: String,
    phase: String,
    link: Vec<String>,
    manifest: BTreeMap<String, Vec<String>>,
    gold: Option<GoldJ>,
    exec: Option<ExecJ>,
    #[serde(default)]
    programs: Vec<ProgJ>,
    bm25: Vec<(String, f64)>,
}

#[derive(Deserialize)]
struct StressJ {
    question: String,
    link: Vec<String>,
}

#[derive(Deserialize)]
struct ParityWorld<'a> {
    set: String,
    protocol: String,
    #[serde(borrow)]
    world: World<'a>,
    base_revision: Rev,
    final_revision: Rev,
    fingerprints: Fps,
    lookup_keys: Vec<(String, String, String)>,
    lookups: HashMap<String, Vec<LookupJ>>,
    items: Vec<ItemJ>,
    stress: Vec<StressJ>,
    normalize_entity: BTreeMap<String, String>,
    normalize_answer: BTreeMap<String, String>,
}

/// (step before which the hop was read, fact ids, resolve flags, anchor)
type HopRead = (usize, Vec<String>, Vec<bool>, Option<String>);

#[derive(Default, Clone)]
struct Tally {
    worlds: u64,
    items: u64,
    checks: BTreeMap<&'static str, (u64, u64)>,
    examples: Vec<String>,
}

impl Tally {
    fn check(&mut self, what: &'static str, ok: bool, detail: impl FnOnce() -> String) {
        let e = self.checks.entry(what).or_default();
        e.1 += 1;
        if ok {
            e.0 += 1;
        } else if self.examples.len() < 12 {
            self.examples.push(format!("[{what}] {}", detail()));
        }
    }
    fn failures(&self) -> u64 {
        self.checks.values().map(|(ok, n)| n - ok).sum()
    }
}

fn answer_json(mem: &Memory, a: &Answer) -> Value {
    let st = &mem.store;
    let program: Vec<Value> = a
        .program
        .iter()
        .map(|x| {
            let mut o = json!({"op": x.op, "pointer": x.pointer});
            if x.illegal {
                o["illegal"] = json!(true);
            }
            o
        })
        .collect();
    let trace: Vec<Value> = a
        .execution_trace
        .iter()
        .map(|t| {
            let mut o =
                json!({"step": t.step, "op": t.op, "pointer": t.pointer, "result": t.result});
            if t.illegal {
                o["illegal"] = json!(true);
            }
            if let Some(h) = &t.hop_candidate_ids {
                o["hop_candidate_ids"] =
                    json!(h.iter().map(|&r| st.fact_id(r)).collect::<Vec<_>>());
            }
            o
        })
        .collect();
    json!({
        "status": a.status, "answer": a.answer, "error": a.error,
        "evidence_ids": a.evidence_ids.iter().map(|&r| st.fact_id(r)).collect::<Vec<_>>(),
        "program": program, "execution_trace": trace,
    })
}

fn gold_steps(trace: &[Value]) -> Vec<GoldStep> {
    trace
        .iter()
        .map(|s| GoldStep {
            op: s["op"].as_str().unwrap().to_string(),
            key: s.get("key").and_then(|k| k.as_array()).map(|k| {
                (
                    k[0].as_str().unwrap().to_string(),
                    k[1].as_str().unwrap().to_string(),
                )
            }),
            inverse: s.get("direction").and_then(|d| d.as_str()) == Some("in"),
        })
        .collect()
}

fn memory(reference_text: bool) -> Memory {
    Memory::new(MemoryConfig {
        store: StoreConfig::default(),
        record_text: if reference_text {
            RecordText::Reference
        } else {
            RecordText::Rich
        },
        ..MemoryConfig::default()
    })
}

/// BM25 rankings equal up to permutations inside runs of scores that agree to 1e-12 (float summation order)
fn bm25_equal(got: &[(String, f64)], exp: &[(String, f64)]) -> bool {
    if got.len() != exp.len() {
        return false;
    }
    let close = |a: f64, b: f64| (a - b).abs() <= 1e-12 * a.abs().max(b.abs()).max(1.0);
    let mut i = 0;
    while i < exp.len() {
        if !close(got[i].1, exp[i].1) {
            return false;
        }
        let mut j = i + 1;
        while j < exp.len() && close(exp[j].1, exp[i].1) {
            j += 1;
        }
        let mut a: Vec<&str> = got[i..j].iter().map(|x| x.0.as_str()).collect();
        let mut b: Vec<&str> = exp[i..j].iter().map(|x| x.0.as_str()).collect();
        a.sort();
        b.sort();
        if a != b && j < exp.len() {
            return false;
        }
        i = j;
    }
    true
}

fn check_world(line: &str, tally: &mut BTreeMap<String, Tally>) {
    let pw: ParityWorld = serde_json::from_str(line).expect("parity line");
    let t = tally.entry(pw.set.clone()).or_default();
    t.worlds += 1;
    let protocol = if pw.protocol == "v2" {
        Protocol::V2
    } else {
        Protocol::V1
    };
    let mut mem = memory(true);
    let revs = mem.ingest_world(&pw.world).expect("ingest");
    t.check(
        "revisions",
        (revs.base, revs.final_) == (pw.base_revision, pw.final_revision),
        || format!("{revs:?} vs {} {}", pw.base_revision, pw.final_revision),
    );
    // a base-only memory: BM25 at the "before" revision (BM25 indexes the head)
    let mut base_mem = memory(true);
    for d in &pw.world.base_documents {
        base_mem
            .sink_add_document(&d.doc_id, &d.version, &d.text)
            .unwrap();
    }
    for f in &pw.world.base_facts {
        let fo = f.to_fact_in();
        base_mem.sink_add_fact(&fo.get(), &[]).unwrap();
    }
    let snaps: HashMap<&str, Snapshot<'_>> = [
        ("before", mem.snapshot_at(revs.base)),
        ("after", mem.snapshot_at(revs.final_)),
    ]
    .into_iter()
    .collect();
    for (phase, fp) in [
        ("before", &pw.fingerprints.before),
        ("after", &pw.fingerprints.after),
    ] {
        let got = snaps[phase].fingerprint();
        t.check("fingerprints", &*got == fp.as_str(), || {
            format!("{phase}: {got} vs {fp}")
        });
    }
    for (phase, list) in &pw.lookups {
        let snap = snaps[phase.as_str()];
        for (k, exp) in pw.lookup_keys.iter().zip(list) {
            let l = snap.lookup(&k.0, &k.1, &k.2);
            let refs: Vec<&str> = l.refs.iter().map(|&r| mem.store.fact_id(r)).collect();
            let val_ok = match (&exp.value, l.value) {
                (None, None) => true,
                (Some((kind, v, unit)), Some(tv)) => {
                    kind == tv.kind.as_str()
                        && unit == tv.unit
                        && (if tv.kind == xerj_store::Kind::Boolean {
                            v.as_bool() == tv.as_bool()
                        } else {
                            v.as_str() == Some(tv.value)
                        })
                }
                _ => false,
            };
            let ok = exp.status == l.status.value_str()
                && val_ok
                && exp.refs == refs
                && exp.reason == l.reason;
            t.check("lookups", ok, || {
                format!(
                    "{phase} {k:?}: {:?} {:?} {refs:?} vs {} {:?} {:?}",
                    l.status, l.value, exp.status, exp.value, exp.refs
                )
            });
        }
    }
    let mut ac = Linker::new(LinkerKind::AhoCorasick);
    ac.catch_up(&mem.store);
    let mut buf = QBuf::default();
    let subj_names = |mem: &Memory, v: &[u32]| {
        v.iter()
            .map(|&s| mem.store.pools.subjects.get(s).to_string())
            .collect::<Vec<_>>()
    };
    let questions: HashMap<&str, &str> = pw
        .world
        .items
        .iter()
        .map(|i| (i.item_id.as_ref(), i.question.as_ref()))
        .collect();
    for it in &pw.items {
        t.items += 1;
        let snap = snaps[it.phase.as_str()];
        let q = questions[it.item_id.as_str()];
        let l1 = subj_names(&mem, &mem.link(&snap, q, &mut buf));
        t.check("link (span-hash)", l1 == it.link, || {
            format!("{}: {l1:?} vs {:?}", it.item_id, it.link)
        });
        let l2 = subj_names(&mem, &ac.link_exact(&snap, &mem.subjects, q, &mut buf));
        t.check("link (aho-corasick)", l2 == it.link, || {
            format!("{}: {l2:?} vs {:?}", it.item_id, it.link)
        });
        let mut cands16 = Vec::new();
        for (k, exp) in &it.manifest {
            let kk: usize = k.parse().unwrap();
            let m = mem.candidate_manifest(&snap, &it.item_id, q, kk, &mut buf);
            let got: Vec<&str> = m.iter().map(|&r| mem.store.fact_id(r)).collect();
            t.check("manifests", &got == exp, || {
                format!("{} K={k}: {got:?} vs {exp:?}", it.item_id)
            });
            if kk == 16 {
                cands16 = m;
            }
        }
        if let (Some(gold), Some(ex)) = (&it.gold, &it.exec) {
            let steps = gold_steps(&gold.trace);
            let mut ep = Episode::new(&mem, snap, protocol, cands16.clone(), &it.item_id, 16);
            // hop manifests read before each step (v2), recomputed along the same execution
            let mut hops_got: Vec<HopRead> = Vec::new();
            if protocol == Protocol::V2 {
                let mut probe =
                    Episode::new(&mem, snap, protocol, cands16.clone(), &it.item_id, 16);
                for (si, st) in steps.iter().enumerate() {
                    if probe.done {
                        break;
                    }
                    let h = probe.current_hop();
                    if !h.is_empty() {
                        hops_got.push((
                            si,
                            h.cands
                                .iter()
                                .map(|&r| mem.store.fact_id(r).to_string())
                                .collect(),
                            h.resolves.clone(),
                            h.anchor.clone(),
                        ));
                    }
                    let (op, ptr) = xerj_exec::episode::teacher_target(&mut probe, st);
                    if !(op >= 0 && probe.legal_ops()[op as usize]) {
                        break;
                    }
                    let _ = probe.step(op, ptr);
                }
            }
            let (ans, problems) = run_gold(&mut ep, &steps);
            let aj = answer_json(&mem, &ans);
            t.check("exec (gold traces)", aj == ex.answer, || {
                format!("{}: {aj} vs {}", it.item_id, ex.answer)
            });
            t.check("exec problems", problems == ex.problems, || {
                format!("{}: {problems:?} vs {:?}", it.item_id, ex.problems)
            });
            let top = ep.top_reason().unwrap_or("").to_string();
            t.check("exec final reason", top == ex.top_reason, || {
                format!("{}: {top:?} vs {:?}", it.item_id, ex.top_reason)
            });
            let hops_exp: Vec<HopRead> = ex
                .hops
                .iter()
                .map(|h| {
                    (
                        h.before_step,
                        h.fact_ids.clone(),
                        h.resolves.clone(),
                        h.anchor.clone(),
                    )
                })
                .collect();
            t.check("hop manifests", hops_got == hops_exp, || {
                format!("{}: {hops_got:?} vs {hops_exp:?}", it.item_id)
            });
            let v = verify(&mem, &snap, q, &ans, VerifyMode::Substring, &mut buf);
            t.check("verify rule", v.vetoed == ex.verify_vetoed, || {
                format!("{}: {} vs {}", it.item_id, v.vetoed, ex.verify_vetoed)
            });
            for p in &it.programs {
                let mut ep = Episode::new(&mem, snap, protocol, cands16.clone(), &it.item_id, 16);
                for &(op, ptr) in &p.actions {
                    if ep.done {
                        break;
                    }
                    let _ = ep.step(op, ptr);
                }
                let aj = answer_json(&mem, &ep.answer_or_unfinished());
                t.check("programs", aj == p.answer, || {
                    format!("{} {:?}: {aj} vs {}", it.item_id, p.actions, p.answer)
                });
            }
        }
        let bm = if it.phase == "before" {
            &base_mem
        } else {
            &mem
        };
        let hits: Vec<(String, f64)> = bm
            .search_records(q, 16)
            .into_iter()
            .map(|h| (bm.store.fact_id(h.doc).to_string(), h.score))
            .collect();
        t.check("bm25 (LexicalIndex)", bm25_equal(&hits, &it.bm25), || {
            format!("{} {q:?}: {hits:?} vs {:?}", it.item_id, it.bm25)
        });
    }
    let after = snaps["after"];
    for s in &pw.stress {
        let l1 = subj_names(&mem, &mem.link(&after, &s.question, &mut buf));
        t.check("link stress (span-hash)", l1 == s.link, || {
            format!("{:?}: {l1:?} vs {:?}", s.question, s.link)
        });
        let l2 = subj_names(
            &mem,
            &ac.link_exact(&after, &mem.subjects, &s.question, &mut buf),
        );
        t.check("link stress (aho-corasick)", l2 == s.link, || {
            format!("{:?}: {l2:?} vs {:?}", s.question, s.link)
        });
    }
    for (s, exp) in &pw.normalize_entity {
        let got = normalize_entity(s);
        t.check("normalize_entity", &got == exp, || {
            format!("{s:?}: {got:?} vs {exp:?}")
        });
    }
    for (s, exp) in &pw.normalize_answer {
        let got = normalize_answer(Some(s));
        t.check("normalize_answer", &got == exp, || {
            format!("{s:?}: {got:?} vs {exp:?}")
        });
    }
}

#[test]
fn parity_with_python_reference() {
    let dir = std::env::var("XERJ_PARITY_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(std::env::var("HOME").unwrap()).join("xerj_mem_parity"));
    let mut files: Vec<PathBuf> = match std::fs::read_dir(&dir) {
        Ok(rd) => rd
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.to_string_lossy().ends_with(".parity.jsonl"))
            .collect(),
        Err(_) => {
            eprintln!(
                "parity fixtures not found in {} (set XERJ_PARITY_DIR); skipping",
                dir.display()
            );
            return;
        }
    };
    files.sort();
    let only = std::env::var("XERJ_PARITY_SETS").ok();
    let mut tally: BTreeMap<String, Tally> = BTreeMap::new();
    let t0 = std::time::Instant::now();
    for f in &files {
        if let Some(o) = &only {
            if !o
                .split(',')
                .any(|s| f.file_name().unwrap().to_string_lossy() == format!("{s}.parity.jsonl"))
            {
                continue;
            }
        }
        let rd = std::io::BufReader::with_capacity(1 << 20, std::fs::File::open(f).unwrap());
        for line in rd.lines() {
            let line = line.unwrap();
            if line.starts_with("{\"header\"") {
                continue;
            }
            check_world(&line, &mut tally);
        }
    }
    let mut cols: Vec<&'static str> = Vec::new();
    for t in tally.values() {
        for k in t.checks.keys() {
            if !cols.contains(k) {
                cols.push(k);
            }
        }
    }
    let mut md = String::new();
    md.push_str(&format!(
        "| set | worlds | items | {} |\n",
        cols.join(" | ")
    ));
    md.push_str(&format!("|---|---:|---:|{}\n", "---:|".repeat(cols.len())));
    let mut fails = 0;
    for (name, t) in &tally {
        let cells: Vec<String> = cols
            .iter()
            .map(|c| {
                t.checks
                    .get(c)
                    .map_or("-".into(), |(ok, n)| format!("{ok}/{n}"))
            })
            .collect();
        md.push_str(&format!(
            "| {name} | {} | {} | {} |\n",
            t.worlds,
            t.items,
            cells.join(" | ")
        ));
        fails += t.failures();
    }
    println!("{md}");
    println!("parity checked in {:.1} s", t0.elapsed().as_secs_f64());
    for (name, t) in &tally {
        for e in &t.examples {
            println!("MISMATCH {name}: {e}");
        }
    }
    let _ = std::fs::write(dir.join("RUST_PARITY.md"), &md);
    assert_eq!(fails, 0, "{fails} parity mismatches");
}
