//! Stack-v5 (DIRECTION) parity with the dev_v5 Python reference (research/xerj_r_mvp/tools/xerj_mem_parity_export_v5.py
//! fixtures; reference code research/xerj_r_mvp/dev_v5/future_scientific/stack_v5.py on stack_v2 / stack_runner / xr_codecs).
//!
//! XERJ_PARITY_V5_DIR (default $HOME/xerj_mem_v5_data/parity_v5) holds `<set>.parity.jsonl`; every world line is ingested
//! into a fresh `Memory` (exact fingerprints, SHA-256 manifests) and every exported reference output is recomputed and
//! compared: revisions, fingerprints, link_entities, link_values (items and stress questions, both revisions), stack-v5
//! candidate manifests (fact ids + inverse flags) at every exported (K, max_inverse), gold-trace executions under
//! `Protocol::V5` (answer record incl. program / execution trace with direction, problems, final reason, final stack,
//! teacher actions, every hop manifest read with inverse flags, the same hops at other (K, max_inverse)), random / guided /
//! mutated programs (answer record, final stack), cross-snapshot programs (the item's manifest executed on the other
//! snapshot), the verify verdict AND the rule that fired for every execution (tools/verify_rows.py as written: no direction
//! exemption, so inverse answers are vetoed unless their subject occurs in the question; split into inverse / forward
//! answers), normalize_entity and normalize_answer.
//! XERJ_PARITY_V5_SETS = comma-separated set names (default: all files). The test fails on any mismatch, prints a per-set
//! table (also written to $XERJ_PARITY_V5_DIR/RUST_PARITY_V5.md) and coverage counters of the inverse paths.
//! Negative control: XERJ_PARITY_V5_EXPECT_MISMATCHES=N makes the test pass iff exactly N checks mismatch (they are printed).

use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashMap};
use std::io::BufRead;
use std::path::PathBuf;
use xerj_exec::episode::{run_gold, teacher_target, Answer, Episode, GoldStep};
use xerj_exec::result::result_text;
use xerj_exec::scorer::normalize_answer;
use xerj_exec::verify::{verify, VerifyMode};
use xerj_index::linker::QBuf;
use xerj_index::{Memory, MemoryConfig, RecordText};
use xerj_store::text::normalize_entity;
use xerj_store::world::World;
use xerj_store::{Kind, Rev, Snapshot, StoreConfig, TypedValue};

#[derive(Deserialize)]
struct Header {
    max_inverse: i64,
    #[serde(rename = "episode_K")]
    episode_k: usize,
}

#[derive(Deserialize)]
struct Fps {
    before: String,
    after: String,
}

#[derive(Deserialize)]
struct ManJ {
    #[serde(rename = "K")]
    k: usize,
    cap: i64,
    ids: Vec<String>,
    inverse: Vec<bool>,
}

#[derive(Deserialize)]
struct GoldJ {
    trace: Vec<Value>,
}

#[derive(Deserialize, Clone, PartialEq, Debug)]
struct HopVarJ {
    #[serde(rename = "K")]
    k: usize,
    cap: i64,
    fact_ids: Vec<String>,
    resolves: Vec<bool>,
    inverse: Vec<bool>,
    anchor: Option<String>,
}

#[derive(Deserialize)]
struct HopJ {
    before_step: usize,
    fact_ids: Vec<String>,
    resolves: Vec<bool>,
    inverse: Vec<bool>,
    anchor: Option<String>,
    value: (String, String, String),
    value_ids: Vec<String>,
    variants: Vec<HopVarJ>,
}

#[derive(Deserialize)]
struct ExecJ {
    answer: Value,
    problems: Vec<String>,
    hops: Vec<HopJ>,
    top_reason: String,
    actions: Vec<(i64, Option<i64>)>,
    stack: Value,
    verify_vetoed: bool,
    verify_reason: String,
    inverse_steps: usize,
    score: ScoreJ,
}

/// the reference scorer's verdict on the gold execution (informational: coverage counters)
#[derive(Deserialize)]
struct ScoreJ {
    supported: bool,
}

#[derive(Deserialize)]
struct ProgJ {
    kind: String,
    actions: Vec<(i64, Option<i64>)>,
    answer: Value,
    stack: Value,
    vetoed: bool,
    verify_reason: String,
}

#[derive(Deserialize)]
struct CrossJ {
    snapshot: String,
    programs: Vec<ProgJ>,
}

#[derive(Deserialize)]
struct ItemJ {
    item_id: String,
    phase: String,
    probe: bool,
    link: Vec<String>,
    values: Vec<String>,
    manifests: Vec<ManJ>,
    gold: Option<GoldJ>,
    exec: Option<ExecJ>,
    #[serde(default)]
    programs: Vec<ProgJ>,
    cross: Option<CrossJ>,
}

#[derive(Deserialize)]
struct StressJ {
    phase: String,
    question: String,
    values: Vec<String>,
}

#[derive(Deserialize)]
struct ParityWorld<'a> {
    set: String,
    #[serde(borrow)]
    world: World<'a>,
    base_revision: Rev,
    final_revision: Rev,
    fingerprints: Fps,
    items: Vec<ItemJ>,
    stress_values: Vec<StressJ>,
    normalize_entity: BTreeMap<String, String>,
    normalize_answer: BTreeMap<String, String>,
}

/// (step before which the hop was read, fact ids, resolve flags, inverse flags, anchor)
type HopRead = (usize, Vec<String>, Vec<bool>, Vec<bool>, Option<String>);

#[derive(Default, Clone)]
struct Tally {
    worlds: u64,
    items: u64,
    probes: u64,
    checks: BTreeMap<&'static str, (u64, u64)>,
    examples: Vec<String>,
    /// coverage of the stack-v5 paths, counted on the Rust executions that matched the reference
    coverage: BTreeMap<String, u64>,
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
    fn cover(&mut self, key: impl Into<String>) {
        *self.coverage.entry(key.into()).or_default() += 1;
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
            if x.inverse {
                o["direction"] = json!("in");
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
            if t.inverse {
                o["direction"] = json!("in");
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

/// the final stack as the export's stack_json: status, result text, references, reason, producing step
fn stack_json(mem: &Memory, ep: &Episode<'_>) -> Value {
    Value::Array(
        ep.stack
            .iter()
            .map(|(r, step)| {
                json!({"status": r.status.value_str(), "text": result_text(r), "reason": &*r.reason, "step": step,
                       "refs": r.refs.iter().map(|&x| mem.store.fact_id(x)).collect::<Vec<_>>()})
            })
            .collect(),
    )
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

fn memory() -> Memory {
    Memory::new(MemoryConfig {
        store: StoreConfig::default(),
        record_text: RecordText::Reference,
        bm25_records: false,
        bm25_documents: false,
        ..MemoryConfig::default()
    })
}

/// coverage counters of one executed answer + final stack (inverse steps and the inverse reasons reached)
fn cover(t: &mut Tally, ep: &Episode<'_>) {
    for a in &ep.trace {
        if a.inverse {
            t.cover(format!(
                "inverse {} -> {}",
                a.op,
                a.result
                    .as_deref()
                    .map(|r| if r.starts_with("VALUE") { "VALUE" } else { r })
                    .unwrap_or("-")
            ));
        }
    }
    for (r, _) in &ep.stack {
        let why: &str = &r.reason;
        if why.starts_with("inverse record") || why == "multiple subjects (set)" {
            t.cover(format!("reason: {why}"));
        }
    }
}

/// the verify verdict (vetoed, reason) of an answer, checked under "verify (inverse answers)" when the execution followed
/// an inverse candidate (the reference rule, tools/verify_rows.py, has no direction exemption) and "verify (forward answers)"
/// otherwise
#[allow(clippy::too_many_arguments)]
fn check_verify(
    t: &mut Tally,
    mem: &Memory,
    snap: &Snapshot<'_>,
    q: &str,
    ep: &Episode<'_>,
    ans: &Answer,
    exp: (bool, &str),
    buf: &mut QBuf,
    label: impl FnOnce() -> String,
) -> bool {
    let v = verify(mem, snap, q, ans, VerifyMode::Substring, buf);
    let inverse = ep.trace.iter().any(|x| x.inverse);
    let what = if inverse {
        "verify (inverse answers)"
    } else {
        "verify (forward answers)"
    };
    let ok = v.vetoed == exp.0 && v.reason == exp.1;
    t.check(what, ok, || {
        format!(
            "{}: ({}, {:?}) vs ({}, {:?})",
            label(),
            v.vetoed,
            v.reason,
            exp.0,
            exp.1
        )
    });
    if inverse && ans.status == "ANSWER" {
        t.cover("verify: inverse ANSWERs checked");
        if v.vetoed {
            t.cover(format!("verify: inverse ANSWERs vetoed ({})", v.reason));
        }
    }
    v.vetoed
}

fn kind_of(s: &str) -> Kind {
    Kind::parse(s).expect("value kind")
}

#[allow(clippy::too_many_arguments)]
fn run_prog<'m>(
    mem: &'m Memory,
    snap: Snapshot<'m>,
    cands: &[u32],
    inv: &[bool],
    item_id: &str,
    k: usize,
    max_inverse: i64,
    actions: &[(i64, Option<i64>)],
) -> Episode<'m> {
    let mut ep = Episode::new_v5(mem, snap, cands.to_vec(), inv.to_vec(), item_id, k);
    ep.max_inverse = max_inverse;
    for &(op, ptr) in actions {
        if ep.done {
            break;
        }
        let _ = ep.step(op, ptr);
    }
    ep
}

fn check_world(line: &str, hdr: &Header, tally: &mut BTreeMap<String, Tally>) {
    let pw: ParityWorld = serde_json::from_str(line).expect("parity line");
    let t = tally.entry(pw.set.clone()).or_default();
    t.worlds += 1;
    let k_ep = hdr.episode_k;
    let mut mem = memory();
    let revs = mem.ingest_world(&pw.world).expect("ingest");
    t.check(
        "revisions",
        (revs.base, revs.final_) == (pw.base_revision, pw.final_revision),
        || format!("{revs:?} vs {} {}", pw.base_revision, pw.final_revision),
    );
    let mem = mem;
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
    let mut buf = QBuf::default();
    let subj_names = |v: &[u32]| {
        v.iter()
            .map(|&s| mem.store.pools.subjects.get(s).to_string())
            .collect::<Vec<_>>()
    };
    let value_names = |v: &[u32]| {
        v.iter()
            .map(|&s| mem.store.pools.strs.get(s).to_string())
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
        if it.probe {
            t.probes += 1;
        }
        let snap = snaps[it.phase.as_str()];
        let q = questions[it.item_id.as_str()];
        let l1 = subj_names(&mem.link(&snap, q, &mut buf));
        t.check("link_entities", l1 == it.link, || {
            format!("{}: {l1:?} vs {:?}", it.item_id, it.link)
        });
        let lv = value_names(&mem.link_values(&snap, q, &mut buf));
        t.check("link_values (items)", lv == it.values, || {
            format!("{}: {lv:?} vs {:?}", it.item_id, it.values)
        });
        let mut cands16: Option<(Vec<u32>, Vec<bool>)> = None;
        for m in &it.manifests {
            let got = mem.candidate_manifest_v5_with(&snap, &it.item_id, q, m.k, m.cap, &mut buf);
            let ids: Vec<&str> = got.cands.iter().map(|&r| mem.store.fact_id(r)).collect();
            let ok = ids == m.ids && got.inverse == m.inverse;
            let what = if m.cap == hdr.max_inverse {
                "manifests (K 16/4/64)"
            } else {
                "manifests (other max_inverse)"
            };
            t.check(what, ok, || {
                format!(
                    "{} K={} cap={}: {ids:?} {:?} vs {:?} {:?}",
                    it.item_id, m.k, m.cap, got.inverse, m.ids, m.inverse
                )
            });
            if m.cap == hdr.max_inverse && ok {
                let n_inv = got.inverse.iter().filter(|&&b| b).count() as u64;
                if n_inv > 0 {
                    t.cover(format!("base manifests with inverse slots (K={})", m.k));
                    *t.coverage
                        .entry(format!("base manifest inverse slots (K={})", m.k))
                        .or_default() += n_inv;
                }
            }
            if m.k == k_ep && m.cap == hdr.max_inverse {
                cands16 = Some((got.cands, got.inverse));
            }
        }
        let (cands, inv) =
            cands16.expect("the episode manifest (K, default max_inverse) is exported");
        if let (Some(gold), Some(ex)) = (&it.gold, &it.exec) {
            let steps = gold_steps(&gold.trace);
            // hop manifests read before each step and the executed teacher actions, along the same execution
            let mut probe =
                Episode::new_v5(&mem, snap, cands.clone(), inv.clone(), &it.item_id, k_ep);
            probe.max_inverse = hdr.max_inverse;
            let mut hops_got: Vec<HopRead> = Vec::new();
            let mut acts_got: Vec<(i64, Option<i64>)> = Vec::new();
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
                        h.inverse.clone(),
                        h.anchor.clone(),
                    ));
                }
                let (op, ptr) = teacher_target(&mut probe, st);
                if !(op >= 0 && probe.legal_ops()[op as usize]) {
                    break;
                }
                acts_got.push((op, ptr));
                let _ = probe.step(op, ptr);
            }
            let hops_exp: Vec<HopRead> = ex
                .hops
                .iter()
                .map(|h| {
                    (
                        h.before_step,
                        h.fact_ids.clone(),
                        h.resolves.clone(),
                        h.inverse.clone(),
                        h.anchor.clone(),
                    )
                })
                .collect();
            t.check("hop manifests (read)", hops_got == hops_exp, || {
                format!("{}: {hops_got:?} vs {hops_exp:?}", it.item_id)
            });
            for h in &hops_got {
                let n_inv = h.3.iter().filter(|&&b| b).count() as u64;
                t.cover("hop manifests read");
                if n_inv > 0 {
                    t.cover("hop manifests read with inverse slots");
                    *t.coverage
                        .entry("hop manifest inverse slots".to_string())
                        .or_default() += n_inv;
                }
            }
            t.check("teacher actions", acts_got == ex.actions, || {
                format!("{}: {acts_got:?} vs {:?}", it.item_id, ex.actions)
            });
            let fp = snap.fingerprint();
            for h in &ex.hops {
                let v = TypedValue {
                    kind: kind_of(&h.value.0),
                    value: &h.value.1,
                    unit: &h.value.2,
                };
                let ids: Vec<&str> = h.value_ids.iter().map(|s| s.as_str()).collect();
                for var in &h.variants {
                    let got =
                        mem.hop_manifest_v5(&snap, &it.item_id, &fp, &v, &ids, var.k, var.cap);
                    let g = HopVarJ {
                        k: var.k,
                        cap: var.cap,
                        fact_ids: got
                            .cands
                            .iter()
                            .map(|&r| mem.store.fact_id(r).to_string())
                            .collect(),
                        resolves: got.resolves.clone(),
                        inverse: got.inverse.clone(),
                        anchor: got.anchor.clone(),
                    };
                    t.check("hop manifests (other K / max_inverse)", &g == var, || {
                        format!("{} step {}: {g:?} vs {var:?}", it.item_id, h.before_step)
                    });
                }
            }
            let mut ep = Episode::new_v5(&mem, snap, cands.clone(), inv.clone(), &it.item_id, k_ep);
            ep.max_inverse = hdr.max_inverse;
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
            let sj = stack_json(&mem, &ep);
            t.check("exec final stack", sj == ex.stack, || {
                format!("{}: {sj} vs {}", it.item_id, ex.stack)
            });
            let n_inv = ep.trace.iter().filter(|x| x.inverse).count();
            t.check("exec inverse steps", n_inv == ex.inverse_steps, || {
                format!("{}: {n_inv} vs {}", it.item_id, ex.inverse_steps)
            });
            let vetoed = check_verify(
                t,
                &mem,
                &snap,
                q,
                &ep,
                &ans,
                (ex.verify_vetoed, ex.verify_reason.as_str()),
                &mut buf,
                || format!("{} gold", it.item_id),
            );
            cover(t, &ep);
            if n_inv > 0 {
                t.cover("gold: executions with an inverse step");
                if ex.score.supported {
                    t.cover("gold: correct (supported) inverse answers");
                    if vetoed {
                        t.cover("gold: correct (supported) inverse answers vetoed by verify");
                    }
                }
            }
        }
        for p in &it.programs {
            let ep = run_prog(
                &mem,
                snap,
                &cands,
                &inv,
                &it.item_id,
                k_ep,
                hdr.max_inverse,
                &p.actions,
            );
            let ans = ep.answer_or_unfinished();
            let aj = answer_json(&mem, &ans);
            t.check("programs (answer)", aj == p.answer, || {
                format!(
                    "{} {} {:?}: {aj} vs {}",
                    it.item_id, p.kind, p.actions, p.answer
                )
            });
            let sj = stack_json(&mem, &ep);
            t.check("programs (final stack)", sj == p.stack, || {
                format!(
                    "{} {} {:?}: {sj} vs {}",
                    it.item_id, p.kind, p.actions, p.stack
                )
            });
            check_verify(
                t,
                &mem,
                &snap,
                q,
                &ep,
                &ans,
                (p.vetoed, p.verify_reason.as_str()),
                &mut buf,
                || format!("{} {}", it.item_id, p.kind),
            );
            cover(t, &ep);
        }
        if let Some(cr) = &it.cross {
            let osnap = snaps[cr.snapshot.as_str()];
            for p in &cr.programs {
                let ep = run_prog(
                    &mem,
                    osnap,
                    &cands,
                    &inv,
                    &it.item_id,
                    k_ep,
                    hdr.max_inverse,
                    &p.actions,
                );
                let ans = ep.answer_or_unfinished();
                let aj = answer_json(&mem, &ans);
                t.check("cross-snapshot (answer)", aj == p.answer, || {
                    format!(
                        "{} {} -> {} {:?}: {aj} vs {}",
                        it.item_id, p.kind, cr.snapshot, p.actions, p.answer
                    )
                });
                let sj = stack_json(&mem, &ep);
                t.check("cross-snapshot (final stack)", sj == p.stack, || {
                    format!(
                        "{} {} -> {}: {sj} vs {}",
                        it.item_id, p.kind, cr.snapshot, p.stack
                    )
                });
                check_verify(
                    t,
                    &mem,
                    &osnap,
                    q,
                    &ep,
                    &ans,
                    (p.vetoed, p.verify_reason.as_str()),
                    &mut buf,
                    || format!("{} {} cross", it.item_id, p.kind),
                );
                cover(t, &ep);
            }
        }
    }
    for s in &pw.stress_values {
        let snap = snaps[s.phase.as_str()];
        let got = value_names(&mem.link_values(&snap, &s.question, &mut buf));
        t.check("link_values (stress)", got == s.values, || {
            format!("{} {:?}: {got:?} vs {:?}", s.phase, s.question, s.values)
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
fn parity_v5_with_python_reference() {
    let dir = std::env::var("XERJ_PARITY_V5_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(std::env::var("HOME").unwrap())
                .join("xerj_mem_v5_data")
                .join("parity_v5")
        });
    let mut files: Vec<PathBuf> = match std::fs::read_dir(&dir) {
        Ok(rd) => rd
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.to_string_lossy().ends_with(".parity.jsonl"))
            .collect(),
        Err(_) => {
            eprintln!(
                "stack-v5 parity fixtures not found in {} (set XERJ_PARITY_V5_DIR); skipping",
                dir.display()
            );
            return;
        }
    };
    files.sort();
    let only = std::env::var("XERJ_PARITY_V5_SETS").ok();
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
        let mut hdr: Option<Header> = None;
        for line in rd.lines() {
            let line = line.unwrap();
            if line.starts_with("{\"header\"") {
                hdr = Some(serde_json::from_str(&line).expect("header"));
                continue;
            }
            check_world(&line, hdr.as_ref().expect("header line first"), &mut tally);
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
        "| set | worlds | items (probes) | {} |\n",
        cols.join(" | ")
    ));
    md.push_str(&format!("|---|---:|---:|{}\n", "---:|".repeat(cols.len())));
    let mut fails = 0;
    let mut total = Tally::default();
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
            "| {name} | {} | {} ({}) | {} |\n",
            t.worlds,
            t.items,
            t.probes,
            cells.join(" | ")
        ));
        fails += t.failures();
        total.worlds += t.worlds;
        total.items += t.items;
        total.probes += t.probes;
        for (k, (ok, n)) in &t.checks {
            let e = total.checks.entry(k).or_default();
            e.0 += ok;
            e.1 += n;
        }
        for (k, n) in &t.coverage {
            *total.coverage.entry(k.clone()).or_default() += n;
        }
    }
    let cells: Vec<String> = cols
        .iter()
        .map(|c| {
            total
                .checks
                .get(c)
                .map_or("-".into(), |(ok, n)| format!("{ok}/{n}"))
        })
        .collect();
    md.push_str(&format!(
        "| **total** | {} | {} ({}) | {} |\n",
        total.worlds,
        total.items,
        total.probes,
        cells.join(" | ")
    ));
    md.push_str("\nCoverage (stack-v5 paths on the matching Rust executions):\n\n| path | count |\n|---|---:|\n");
    for (k, n) in &total.coverage {
        md.push_str(&format!("| {k} | {n} |\n"));
    }
    println!("{md}");
    println!(
        "stack-v5 parity checked in {:.1} s; mismatches: {fails}",
        t0.elapsed().as_secs_f64()
    );
    for (name, t) in &tally {
        for e in &t.examples {
            println!("MISMATCH {name}: {e}");
        }
    }
    let _ = std::fs::write(dir.join("RUST_PARITY_V5.md"), &md);
    let expect: u64 = std::env::var("XERJ_PARITY_V5_EXPECT_MISMATCHES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    assert_eq!(
        fails, expect,
        "{fails} stack-v5 parity mismatches (expected {expect})"
    );
}
