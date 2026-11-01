//! E2 exact-path parity (design `research/product/XERJ-R_engine_integration_design_2026-10-26.md` §6.3), on the local
//! spike branch: every world of the parity fixtures goes through the product path — `POST /_facts/{ns}/_world` into a
//! tempdir engine, one namespace per world (the reference runs per-case stores) — then the engine is dropped and reopened,
//! so every projection is rebuilt from the index (epoch op headers plus the `doc` / `fact` payloads, design §2.5), and the
//! rebuilt projection must reproduce what the research path reproduces against the frozen Python reference
//! (`xerj-exec/tests/parity.rs`): revisions, exact fingerprints at both revisions, lookups at both, the span-hash and
//! Aho-Corasick linkers (items and stress questions), candidate manifests at every exported K, gold-trace executions
//! (answer record, problems, final reason, hop manifests), random and mutated programs, and the verify verdict.
//!
//! Not compared: BM25 (the product projection runs with BM25 off; the engine's own full-text index searches the `fact`
//! documents) and the two normalisers (pure functions, no product code involved).
//!
//! Inputs: XERJ_PARITY_DIR holds `<set>.parity.jsonl` (research/xerj_r_mvp/runs/xerj_mem_parity); without it the test
//! says so and returns. XERJ_PARITY_SETS = comma-separated set names; XERJ_E2_MAX_WORLDS caps the worlds per set;
//! XERJ_E2_REPORT names a file for the per-set table. Like the server, every test first raises its open-file soft limit
//! (`raise_nofile_limit`), and it stops its first engine like a server stop before the reopen (`end_session`;
//! XERJ_E2_REOPEN=drop keeps the spike's plain handle drop as a control).
#![cfg(feature = "facts")]

use axum::body::Body;
use axum::http::{Request, StatusCode};
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashMap};
use std::io::BufRead;
use std::path::{Path, PathBuf};
use tower::ServiceExt;
use xerj_facts::xerj_exec::episode::{
    run_gold, teacher_target, Answer, Episode, GoldStep, Protocol,
};
use xerj_facts::xerj_exec::verify::{verify, VerifyMode};
use xerj_facts::xerj_index::linker::{Linker, LinkerKind, QBuf};
use xerj_facts::xerj_index::Memory;
use xerj_facts::xerj_store::world::World;
use xerj_facts::xerj_store::{Rev, Snapshot};

// ------------------------------------------------------------------------------------------- fixture lines (parity.rs)

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
}

/// (step before which the hop was read, fact ids, resolve flags, anchor)
type HopRead = (usize, Vec<String>, Vec<bool>, Option<String>);

#[derive(Default)]
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

// ------------------------------------------------------------------------------------- the comparisons (parity.rs)

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

/// The comparisons of `parity.rs::check_world`, against the product's rebuilt projection instead of a fresh `Memory`.
fn check_world(pw: &ParityWorld<'_>, mem: &Memory, t: &mut Tally) {
    let protocol = if pw.protocol == "v2" {
        Protocol::V2
    } else {
        Protocol::V1
    };
    let snaps: HashMap<&str, Snapshot<'_>> = [
        ("before", mem.snapshot_at(pw.base_revision)),
        ("after", mem.snapshot_at(pw.final_revision)),
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
                        && (if tv.kind == xerj_facts::xerj_store::Kind::Boolean {
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
    let subj_names = |v: &[u32]| {
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
        let l1 = subj_names(&mem.link(&snap, q, &mut buf));
        t.check("link (span-hash)", l1 == it.link, || {
            format!("{}: {l1:?} vs {:?}", it.item_id, it.link)
        });
        let l2 = subj_names(&ac.link_exact(&snap, &mem.subjects, q, &mut buf));
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
            let mut ep = Episode::new(mem, snap, protocol, cands16.clone(), &it.item_id, 16);
            let mut hops_got: Vec<HopRead> = Vec::new();
            if protocol == Protocol::V2 {
                let mut probe = Episode::new(mem, snap, protocol, cands16.clone(), &it.item_id, 16);
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
                    let (op, ptr) = teacher_target(&mut probe, st);
                    if !(op >= 0 && probe.legal_ops()[op as usize]) {
                        break;
                    }
                    let _ = probe.step(op, ptr);
                }
            }
            let (ans, problems) = run_gold(&mut ep, &steps);
            let aj = answer_json(mem, &ans);
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
            let v = verify(mem, &snap, q, &ans, VerifyMode::Substring, &mut buf);
            t.check("verify rule", v.vetoed == ex.verify_vetoed, || {
                format!("{}: {} vs {}", it.item_id, v.vetoed, ex.verify_vetoed)
            });
            for p in &it.programs {
                let mut ep = Episode::new(mem, snap, protocol, cands16.clone(), &it.item_id, 16);
                for &(op, ptr) in &p.actions {
                    if ep.done {
                        break;
                    }
                    let _ = ep.step(op, ptr);
                }
                let aj = answer_json(mem, &ep.answer_or_unfinished());
                t.check("programs", aj == p.answer, || {
                    format!("{} {:?}: {aj} vs {}", it.item_id, p.actions, p.answer)
                });
            }
        }
    }
    let after = snaps["after"];
    for s in &pw.stress {
        let l1 = subj_names(&mem.link(&after, &s.question, &mut buf));
        t.check("link stress (span-hash)", l1 == s.link, || {
            format!("{:?}: {l1:?} vs {:?}", s.question, s.link)
        });
        let l2 = subj_names(&ac.link_exact(&after, &mem.subjects, &s.question, &mut buf));
        t.check("link stress (aho-corasick)", l2 == s.link, || {
            format!("{:?}: {l2:?} vs {:?}", s.question, s.link)
        });
    }
}

// ----------------------------------------------------------------------------------------------------- the product

fn config_for(dir: &Path) -> xerj_common::config::Config {
    let mut config = xerj_common::config::Config::default();
    config.server.data_dir = dir.to_string_lossy().into_owned();
    config.storage.wal_sync = xerj_common::config::WalSync::Async;
    config
}

/// A fresh app over `dir`; after the previous one was dropped this is a restart: the second boot sees only what reached
/// the disk, and every facts projection is rebuilt from its index on first access.
fn app_over(dir: &Path) -> (axum::Router, xerj_api::state::AppState) {
    let config = config_for(dir);
    let metrics = xerj_common::metrics::Metrics::new().expect("metrics");
    let engine = xerj_engine::Engine::new(config.clone()).expect("engine");
    let state = xerj_api::state::AppState::new(config, engine, metrics);
    (
        xerj_api::router::build_es_compat_router(state.clone()),
        state,
    )
}

/// Raise the open-file soft limit to the hard limit, once per process, as `xerj-server` does before anything else runs
/// (`raise_nofile_limit` in its main.rs). Every index holds segment files, a WAL and more, so hundreds of facts namespaces
/// need thousands of descriptors. The test process inherits the shell's soft limit instead (1,024 on the x86 build server):
/// there the engine ran out of descriptors after about 128 namespaces (EMFILE), so commits failed, an index could not be
/// opened again after the reopen (404 for its namespace), and `check_world` then panicked with "revision beyond head".
/// Returns (soft limit before, soft limit now) for the report.
#[allow(clippy::unnecessary_cast)] // the width of `rlim_t` depends on the target
fn raise_nofile_limit() -> (u64, u64) {
    static ONCE: std::sync::OnceLock<(u64, u64)> = std::sync::OnceLock::new();
    *ONCE.get_or_init(|| {
        let get = || {
            let mut lim = libc::rlimit {
                rlim_cur: 0,
                rlim_max: 0,
            };
            // SAFETY: getrlimit only writes the struct it is given.
            let rc = unsafe { libc::getrlimit(libc::RLIMIT_NOFILE, &mut lim) };
            (rc == 0).then_some(lim)
        };
        let Some(lim) = get() else {
            return (0, 0);
        };
        let hard = if lim.rlim_max == libc::RLIM_INFINITY {
            1_048_576
        } else {
            lim.rlim_max
        };
        if hard > lim.rlim_cur {
            let next = libc::rlimit {
                rlim_cur: hard,
                rlim_max: hard,
            };
            // SAFETY: setrlimit only reads the struct; failure leaves the inherited limit in place.
            let _ = unsafe { libc::setrlimit(libc::RLIMIT_NOFILE, &next) };
        }
        let now = get().map_or(lim.rlim_cur, |l| l.rlim_cur);
        (lim.rlim_cur as u64, now as u64)
    })
}

/// How session 1 ends before the engine is reopened on the same directory (`XERJ_E2_REOPEN`).
///
/// `stop` (the default) stops the engine the way `xerj-server` stops on SIGTERM (its shutdown step 15,
/// `Engine::flush_all_force`: every index's background merge scheduling is stopped first, then every index is flushed),
/// drops the handles and waits until every index object of the stopped engine is gone, so nothing of the first engine can
/// still write the data directory while the second engine opens and reads it.
///
/// `drop` only drops the handles, as the spike did; nothing then waits for the first engine's background work. That was
/// suspected for the failures on the x86 build server, but they came from the open-file limit (`raise_nofile_limit`): with
/// the limit raised both modes pass, and with the inherited limit both fail alike. Kept as a control.
async fn end_session(app: axum::Router, state: xerj_api::state::AppState) -> String {
    let mode = std::env::var("XERJ_E2_REOPEN").unwrap_or_else(|_| "stop".into());
    let (before, now) = raise_nofile_limit();
    let limit = format!("open-file soft limit {before} -> {now}");
    if mode == "drop" {
        drop(app);
        drop(state);
        return format!(
            "session 1 ended by dropping the handles only (XERJ_E2_REOPEN=drop); {limit}"
        );
    }
    assert_eq!(mode, "stop", "XERJ_E2_REOPEN = stop | drop");
    let t0 = std::time::Instant::now();
    state.engine.flush_all_force().await;
    let mut weak = Vec::new();
    for info in state.engine.list_indices().await {
        if let Ok(idx) = state.engine.get_index(&info.name) {
            weak.push(std::sync::Arc::downgrade(&idx));
        }
    }
    drop(app);
    drop(state);
    let alive = |w: &[std::sync::Weak<_>]| w.iter().filter(|x| x.strong_count() > 0).count();
    let deadline = t0 + std::time::Duration::from_secs(30);
    while alive(&weak) > 0 && std::time::Instant::now() < deadline {
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    let left = alive(&weak);
    if left > 0 {
        println!(
            "WARNING: {left} of {} indices of the stopped engine are still alive 30 s after its shutdown",
            weak.len()
        );
    }
    format!(
        "session 1 ended like a server stop (flush_all_force; {} of {} indices released after {:.2} s); {limit}",
        weak.len() - left,
        weak.len(),
        t0.elapsed().as_secs_f64()
    )
}

async fn send(app: &axum::Router, req: Request<Body>) -> (StatusCode, Value) {
    let response = app.clone().oneshot(req).await.expect("response");
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body");
    let body = if bytes.is_empty() {
        Value::Null
    } else {
        serde_json::from_slice(&bytes).unwrap_or(Value::Null)
    };
    (status, body)
}

fn post_json(path: &str, body: &Value) -> Request<Body> {
    Request::post(path)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .expect("request")
}

fn get(path: &str) -> Request<Body> {
    Request::get(path).body(Body::empty()).expect("request")
}

/// `(file set name, namespace, fixture line)` for every world the environment selects.
#[allow(clippy::type_complexity)]
fn fixture_worlds() -> Option<(PathBuf, Vec<(String, String, String)>)> {
    let dir = PathBuf::from(std::env::var_os("XERJ_PARITY_DIR")?);
    let only = std::env::var("XERJ_PARITY_SETS").ok();
    let cap: usize = std::env::var("XERJ_E2_MAX_WORLDS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(usize::MAX);
    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
        .ok()?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.to_string_lossy().ends_with(".parity.jsonl"))
        .collect();
    files.sort();
    let mut worlds = Vec::new();
    for f in files {
        let name = f.file_name().unwrap().to_string_lossy().into_owned();
        let set = name.trim_end_matches(".parity.jsonl").to_string();
        if let Some(o) = &only {
            if !o.split(',').any(|s| s == set) {
                continue;
            }
        }
        let rd = std::io::BufReader::with_capacity(1 << 20, std::fs::File::open(&f).unwrap());
        let mut n = 0usize;
        for line in rd.lines() {
            let line = line.unwrap();
            if line.starts_with("{\"header\"") {
                continue;
            }
            if n == cap {
                break;
            }
            worlds.push((set.clone(), format!("e2-{set}-{n}"), line));
            n += 1;
        }
    }
    Some((dir, worlds))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn facts_e2_exact_path_parity_after_a_reopen() {
    let Some((dir, worlds)) = fixture_worlds() else {
        eprintln!(
            "XERJ_PARITY_DIR not set or unreadable (research/xerj_r_mvp/runs/xerj_mem_parity); E2 parity not run"
        );
        return;
    };
    assert!(
        !worlds.is_empty(),
        "no worlds selected in {}",
        dir.display()
    );
    raise_nofile_limit();
    let data = tempfile::tempdir().expect("tempdir");
    let mut tally: BTreeMap<String, Tally> = BTreeMap::new();

    // Session 1: every world through `POST /_facts/{ns}/_world`, then a flush and a stop.
    let t_load = std::time::Instant::now();
    let reopen;
    {
        let (app, state) = app_over(data.path());
        for (_, ns, line) in &worlds {
            let pw: ParityWorld = serde_json::from_str(line).expect("parity line");
            let raw: Value = serde_json::from_str(line).expect("parity line");
            let t = tally.entry(pw.set.clone()).or_default();
            let (st, body) = send(
                &app,
                post_json(&format!("/_facts/{ns}/_world"), &raw["world"]),
            )
            .await;
            t.check("world committed", st == StatusCode::OK, || {
                format!("{ns}: {st} {body}")
            });
            let base = body["base_rev"].as_u64();
            let to = body["committed"]["to_rev"].as_u64();
            t.check(
                "commit revisions",
                base == Some(u64::from(pw.base_revision))
                    && to == Some(u64::from(pw.final_revision)),
                || {
                    format!(
                        "{ns}: base {base:?} to {to:?} vs {} {}",
                        pw.base_revision, pw.final_revision
                    )
                },
            );
        }
        let (st, body) = send(&app, post_json("/_flush", &json!({}))).await;
        assert_eq!(st, StatusCode::OK, "flush: {body}");
        reopen = end_session(app, state).await;
    }
    let load_s = t_load.elapsed().as_secs_f64();

    // Session 2: reopen; each projection is rebuilt from its index on first access, then compared.
    let t_check = std::time::Instant::now();
    {
        let (app, state) = app_over(data.path());
        for (_, ns, line) in &worlds {
            let pw: ParityWorld = serde_json::from_str(line).expect("parity line");
            let t = tally.entry(pw.set.clone()).or_default();
            t.worlds += 1;
            let (st, body) = send(&app, get(&format!("/_facts/{ns}"))).await;
            t.check("rebuilt after reopen", st == StatusCode::OK, || {
                format!("{ns}: {st} {body}")
            });
            let published = body["published_rev"].as_u64();
            t.check(
                "published revision",
                published == Some(u64::from(pw.final_revision)),
                || format!("{ns}: {published:?} vs {}", pw.final_revision),
            );
            if st != StatusCode::OK || published != Some(u64::from(pw.final_revision)) {
                // already a mismatch; the comparisons need both fixture revisions in the projection
                state.facts.remove(ns);
                continue;
            }
            let Some(n) = state.facts.get(ns) else {
                t.check("projection loaded", false, || ns.clone());
                continue;
            };
            {
                let mem = n.mem.read().expect("projection lock");
                check_world(&pw, &mem, t);
            }
            state.facts.remove(ns);
        }
    }
    let check_s = t_check.elapsed().as_secs_f64();

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
    println!(
        "E2 product path: {} worlds loaded in {load_s:.1} s, rebuilt after a reopen and compared in {check_s:.1} s; {reopen}",
        worlds.len()
    );
    for (name, t) in &tally {
        for e in &t.examples {
            println!("MISMATCH {name}: {e}");
        }
    }
    if let Some(p) = std::env::var_os("XERJ_E2_REPORT") {
        let _ = std::fs::write(
            p,
            format!(
                "{md}\n{} worlds loaded in {load_s:.1} s, rebuilt after a reopen and compared in {check_s:.1} s; {reopen}\n",
                worlds.len()
            ),
        );
    }
    assert_eq!(fails, 0, "{fails} E2 parity mismatches");
}

// ---------------------------------------------------------------------- stack-v5 (direction) fixtures (parity_v5.rs)

/// The same product path for the stack-v5 (direction) fixtures, research/xerj_r_mvp/runs/xerj_mem_parity_v5 (selected with
/// XERJ_PARITY_V5_DIR and XERJ_PARITY_V5_SETS; XERJ_E2_MAX_WORLDS caps worlds per set). The rebuilt projection must reproduce
/// what `xerj-exec/tests/parity_v5.rs` reproduces: link_entities, link_values (items and stress), stack-v5 manifests (fact ids
/// and inverse flags) at every exported (K, max_inverse), gold-trace executions under `Protocol::V5` (answer record with
/// directions, problems, final reason, final stack, inverse steps), teacher actions, every hop manifest read and its variants
/// at other (K, max_inverse), random / guided / mutated and cross-snapshot programs (answer and final stack), and the verify
/// verdict with the rule that fired. The pure normalisers are not compared.
mod v5 {
    use super::{app_over, end_session, get, post_json, send};
    use axum::http::StatusCode;
    use serde::Deserialize;
    use serde_json::{json, Value};
    use std::collections::{BTreeMap, HashMap};
    use std::io::BufRead;
    use std::path::PathBuf;
    use xerj_facts::xerj_exec::episode::{run_gold, teacher_target, Answer, Episode, GoldStep};
    use xerj_facts::xerj_exec::result::result_text;
    use xerj_facts::xerj_exec::verify::{verify, VerifyMode};
    use xerj_facts::xerj_index::linker::QBuf;
    use xerj_facts::xerj_index::Memory;
    use xerj_facts::xerj_store::world::World;
    use xerj_facts::xerj_store::{Kind, Rev, Snapshot, TypedValue};

    #[derive(Deserialize, Clone)]
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
    }

    type HopRead = (usize, Vec<String>, Vec<bool>, Vec<bool>, Option<String>);

    #[derive(Default)]
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
    ) {
        let v = verify(mem, snap, q, ans, VerifyMode::Substring, buf);
        let what = if ep.trace.iter().any(|x| x.inverse) {
            "verify (inverse answers)"
        } else {
            "verify (forward answers)"
        };
        t.check(what, v.vetoed == exp.0 && v.reason == exp.1, || {
            format!(
                "{}: ({}, {:?}) vs ({}, {:?})",
                label(),
                v.vetoed,
                v.reason,
                exp.0,
                exp.1
            )
        });
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

    /// The comparisons of `parity_v5.rs::check_world`, against the product's rebuilt projection.
    fn check_world(pw: &ParityWorld<'_>, hdr: &Header, mem: &Memory, t: &mut Tally) {
        let k_ep = hdr.episode_k;
        let snaps: HashMap<&str, Snapshot<'_>> = [
            ("before", mem.snapshot_at(pw.base_revision)),
            ("after", mem.snapshot_at(pw.final_revision)),
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
            let mut cands_ep: Option<(Vec<u32>, Vec<bool>)> = None;
            for m in &it.manifests {
                let got =
                    mem.candidate_manifest_v5_with(&snap, &it.item_id, q, m.k, m.cap, &mut buf);
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
                if m.k == k_ep && m.cap == hdr.max_inverse {
                    cands_ep = Some((got.cands, got.inverse));
                }
            }
            let (cands, inv) =
                cands_ep.expect("the episode manifest (K, default max_inverse) is exported");
            if let (Some(gold), Some(ex)) = (&it.gold, &it.exec) {
                let steps = gold_steps(&gold.trace);
                let mut probe =
                    Episode::new_v5(mem, snap, cands.clone(), inv.clone(), &it.item_id, k_ep);
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
                t.check("teacher actions", acts_got == ex.actions, || {
                    format!("{}: {acts_got:?} vs {:?}", it.item_id, ex.actions)
                });
                let fp = snap.fingerprint();
                for h in &ex.hops {
                    let v = TypedValue {
                        kind: Kind::parse(&h.value.0).expect("value kind"),
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
                let mut ep =
                    Episode::new_v5(mem, snap, cands.clone(), inv.clone(), &it.item_id, k_ep);
                ep.max_inverse = hdr.max_inverse;
                let (ans, problems) = run_gold(&mut ep, &steps);
                let aj = answer_json(mem, &ans);
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
                let sj = stack_json(mem, &ep);
                t.check("exec final stack", sj == ex.stack, || {
                    format!("{}: {sj} vs {}", it.item_id, ex.stack)
                });
                let n_inv = ep.trace.iter().filter(|x| x.inverse).count();
                t.check("exec inverse steps", n_inv == ex.inverse_steps, || {
                    format!("{}: {n_inv} vs {}", it.item_id, ex.inverse_steps)
                });
                check_verify(
                    t,
                    mem,
                    &snap,
                    q,
                    &ep,
                    &ans,
                    (ex.verify_vetoed, ex.verify_reason.as_str()),
                    &mut buf,
                    || format!("{} gold", it.item_id),
                );
            }
            for p in &it.programs {
                let ep = run_prog(
                    mem,
                    snap,
                    &cands,
                    &inv,
                    &it.item_id,
                    k_ep,
                    hdr.max_inverse,
                    &p.actions,
                );
                let ans = ep.answer_or_unfinished();
                let aj = answer_json(mem, &ans);
                t.check("programs (answer)", aj == p.answer, || {
                    format!(
                        "{} {} {:?}: {aj} vs {}",
                        it.item_id, p.kind, p.actions, p.answer
                    )
                });
                let sj = stack_json(mem, &ep);
                t.check("programs (final stack)", sj == p.stack, || {
                    format!(
                        "{} {} {:?}: {sj} vs {}",
                        it.item_id, p.kind, p.actions, p.stack
                    )
                });
                check_verify(
                    t,
                    mem,
                    &snap,
                    q,
                    &ep,
                    &ans,
                    (p.vetoed, p.verify_reason.as_str()),
                    &mut buf,
                    || format!("{} {}", it.item_id, p.kind),
                );
            }
            if let Some(cr) = &it.cross {
                let osnap = snaps[cr.snapshot.as_str()];
                for p in &cr.programs {
                    let ep = run_prog(
                        mem,
                        osnap,
                        &cands,
                        &inv,
                        &it.item_id,
                        k_ep,
                        hdr.max_inverse,
                        &p.actions,
                    );
                    let ans = ep.answer_or_unfinished();
                    let aj = answer_json(mem, &ans);
                    t.check("cross-snapshot (answer)", aj == p.answer, || {
                        format!(
                            "{} {} -> {}: {aj} vs {}",
                            it.item_id, p.kind, cr.snapshot, p.answer
                        )
                    });
                    let sj = stack_json(mem, &ep);
                    t.check("cross-snapshot (final stack)", sj == p.stack, || {
                        format!(
                            "{} {} -> {}: {sj} vs {}",
                            it.item_id, p.kind, cr.snapshot, p.stack
                        )
                    });
                    check_verify(
                        t,
                        mem,
                        &osnap,
                        q,
                        &ep,
                        &ans,
                        (p.vetoed, p.verify_reason.as_str()),
                        &mut buf,
                        || format!("{} {} cross", it.item_id, p.kind),
                    );
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
    }

    /// `(header, namespace, fixture line)` for every world the environment selects.
    #[allow(clippy::type_complexity)]
    fn fixture_worlds() -> Option<(PathBuf, Vec<(Header, String, String)>)> {
        let dir = PathBuf::from(std::env::var_os("XERJ_PARITY_V5_DIR")?);
        let only = std::env::var("XERJ_PARITY_V5_SETS").ok();
        let cap: usize = std::env::var("XERJ_E2_MAX_WORLDS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(usize::MAX);
        let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
            .ok()?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.to_string_lossy().ends_with(".parity.jsonl"))
            .collect();
        files.sort();
        let mut worlds = Vec::new();
        for f in files {
            let name = f.file_name().unwrap().to_string_lossy().into_owned();
            let set = name.trim_end_matches(".parity.jsonl").to_string();
            if let Some(o) = &only {
                if !o.split(',').any(|s| s == set) {
                    continue;
                }
            }
            let rd = std::io::BufReader::with_capacity(1 << 20, std::fs::File::open(&f).unwrap());
            let mut hdr: Option<Header> = None;
            let mut n = 0usize;
            for line in rd.lines() {
                let line = line.unwrap();
                if line.starts_with("{\"header\"") {
                    hdr = Some(serde_json::from_str(&line).expect("header"));
                    continue;
                }
                if n == cap {
                    break;
                }
                let h = hdr.clone().expect("header line first");
                worlds.push((h, format!("e2v5-{set}-{n}"), line));
                n += 1;
            }
        }
        Some((dir, worlds))
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn facts_e2_v5_exact_path_parity_after_a_reopen() {
        let Some((dir, worlds)) = fixture_worlds() else {
            eprintln!(
                "XERJ_PARITY_V5_DIR not set or unreadable (research/xerj_r_mvp/runs/xerj_mem_parity_v5); E2 v5 parity not run"
            );
            return;
        };
        assert!(
            !worlds.is_empty(),
            "no worlds selected in {}",
            dir.display()
        );
        super::raise_nofile_limit();
        let data = tempfile::tempdir().expect("tempdir");
        let mut tally: BTreeMap<String, Tally> = BTreeMap::new();
        let t_load = std::time::Instant::now();
        let reopen;
        {
            let (app, state) = app_over(data.path());
            for (_, ns, line) in &worlds {
                let pw: ParityWorld = serde_json::from_str(line).expect("parity line");
                let raw: Value = serde_json::from_str(line).expect("parity line");
                let t = tally.entry(pw.set.clone()).or_default();
                let (st, body) = send(
                    &app,
                    post_json(&format!("/_facts/{ns}/_world"), &raw["world"]),
                )
                .await;
                t.check("world committed", st == StatusCode::OK, || {
                    format!("{ns}: {st} {body}")
                });
                let base = body["base_rev"].as_u64();
                let to = body["committed"]["to_rev"].as_u64();
                t.check(
                    "commit revisions",
                    base == Some(u64::from(pw.base_revision))
                        && to == Some(u64::from(pw.final_revision)),
                    || {
                        format!(
                            "{ns}: base {base:?} to {to:?} vs {} {}",
                            pw.base_revision, pw.final_revision
                        )
                    },
                );
            }
            let (st, body) = send(&app, post_json("/_flush", &json!({}))).await;
            assert_eq!(st, StatusCode::OK, "flush: {body}");
            reopen = end_session(app, state).await;
        }
        let load_s = t_load.elapsed().as_secs_f64();
        let t_check = std::time::Instant::now();
        {
            let (app, state) = app_over(data.path());
            for (hdr, ns, line) in &worlds {
                let pw: ParityWorld = serde_json::from_str(line).expect("parity line");
                let t = tally.entry(pw.set.clone()).or_default();
                t.worlds += 1;
                let (st, body) = send(&app, get(&format!("/_facts/{ns}"))).await;
                t.check("rebuilt after reopen", st == StatusCode::OK, || {
                    format!("{ns}: {st} {body}")
                });
                let published = body["published_rev"].as_u64();
                t.check(
                    "published revision",
                    published == Some(u64::from(pw.final_revision)),
                    || format!("{ns}: {published:?} vs {}", pw.final_revision),
                );
                if st != StatusCode::OK || published != Some(u64::from(pw.final_revision)) {
                    // already a mismatch; the comparisons need both fixture revisions in the projection
                    state.facts.remove(ns);
                    continue;
                }
                let Some(n) = state.facts.get(ns) else {
                    t.check("projection loaded", false, || ns.clone());
                    continue;
                };
                {
                    let mem = n.mem.read().expect("projection lock");
                    check_world(&pw, hdr, &mem, t);
                }
                state.facts.remove(ns);
            }
        }
        let check_s = t_check.elapsed().as_secs_f64();
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
        println!(
            "E2 v5 product path: {} worlds loaded in {load_s:.1} s, rebuilt after a reopen and compared in {check_s:.1} s; {reopen}",
            worlds.len()
        );
        for (name, t) in &tally {
            for e in &t.examples {
                println!("MISMATCH {name}: {e}");
            }
        }
        if let Some(p) = std::env::var_os("XERJ_E2_V5_REPORT") {
            let _ = std::fs::write(
                p,
                format!(
                    "{md}\n{} worlds loaded in {load_s:.1} s, rebuilt after a reopen and compared in {check_s:.1} s; {reopen}\n",
                    worlds.len()
                ),
            );
        }
        assert_eq!(fails, 0, "{fails} E2 v5 parity mismatches");
    }
}

// ------------------------------------------------------------------- the 68M after a restart (design PR-7, frozen model)

/// E2 frozen-model parity on the local spike: the worlds behind the golden PyTorch fixtures
/// (research/xerj_r_mvp/runs/xerj_infer_export_v5/golden/fixtures/<model>.<set>.jsonl) go through the product path, the
/// engine restarts, and every golden question is answered by the product's compiler (`FactsService::model()`, the 68M export
/// in XERJ_FACTS_MODEL_DIR at XERJ_FACTS_PRECISION) on the rebuilt projection, through `answer_observed` (what `_ask` runs).
/// Compared with the golden row: the candidates the model reads (subject, relation, value, quote, inverse flag), the number
/// of decisions, each decision's input ids (exact), op and pointer (exact) and, in f32, op / pointer logits within 5e-4 (int8 +
/// guard must give identical actions, as xerj-infer's own golden test requires), and the final record (status, answer,
/// evidence ids, program with directions).
///
/// Env: XERJ_E2_GOLDEN_DIR (the fixtures directory), XERJ_E2_GOLDEN_MODEL (default x68_dir_s11), XERJ_R_ROOT
/// (research/xerj_r_mvp, for the worlds), XERJ_E2_MODEL_SETS (default: every set of `SETS` with a golden file),
/// XERJ_E2_MODEL_MAX (rows per set) after XERJ_E2_MODEL_SKIP (rows skipped at the start of each set; with MAX it splits one
/// long golden set over several processes),
/// XERJ_E2_MODEL_REPORT (a file for the table). Without XERJ_E2_GOLDEN_DIR or XERJ_FACTS_MODEL_DIR the test says so and returns.
mod model {
    use super::{app_over, end_session, get, post_json, send};
    use axum::http::StatusCode;
    use serde_json::{json, Value};
    use std::collections::{BTreeMap, HashMap, HashSet};
    use std::io::BufRead;
    use std::path::{Path, PathBuf};
    use xerj_facts::xerj_infer::episode::DecisionEvent;
    use xerj_facts::xerj_infer::fixtures::parse_row;
    use xerj_facts::xerj_infer::Session;
    use xerj_facts::xerj_serve::engine::{answer_observed, LinkerMode};
    use xerj_facts::xerj_store::world::World;

    /// golden set -> (world directory under XERJ_R_ROOT, split, item file replacing the worlds' items). Without
    /// XERJ_E2_MODEL_SETS every set whose golden file `<model>.<set>.jsonl` exists runs; the golden tool
    /// (tools/xerj_infer_golden_v5.py) reads the same worlds and items.
    const SETS: [(&str, &str, &str, Option<&str>); 12] = [
        ("direction_test", "runs/world/direction_v1", "test", None),
        ("direction_dev", "runs/world/direction_v1", "dev", None),
        (
            "factcons_mh_6k",
            "runs/world/factcons",
            "6k",
            Some("runs/world/factcons/chain_stack_v2/items_6k.jsonl"),
        ),
        (
            "factcons_mh_32k",
            "runs/world/factcons",
            "32k",
            Some("runs/world/factcons/chain_stack_v2/items_32k.jsonl"),
        ),
        (
            "factcons_mh_64k",
            "runs/world/factcons",
            "64k",
            Some("runs/world/factcons/chain_stack_v2/items_64k.jsonl"),
        ),
        ("factcons_sh_6k", "runs/world/factcons", "6k", None),
        ("factcons_sh_32k", "runs/world/factcons", "32k", None),
        ("factcons_sh_64k", "runs/world/factcons", "64k", None),
        (
            "mquake_t_chain",
            "runs/world/mquake_lifecycle",
            "test_t",
            Some("runs/world/mquake_lifecycle/chain_stack_v2/items_test_t.jsonl"),
        ),
        (
            "mquake_chain_dev",
            "runs/world/mquake_lifecycle",
            "dev",
            Some("runs/world/mquake_lifecycle/chain_stack_v2/items_dev.jsonl"),
        ),
        ("mquake_t", "runs/world/mquake_lifecycle", "test_t", None),
        ("pick_num_test", "runs/world/pick_num_v1", "test", None),
    ];

    struct Decision {
        ids: Vec<u32>,
        op: usize,
        pointer: Option<usize>,
        op_logits: Vec<f32>,
        ptr_logits: Vec<f32>,
    }

    #[derive(Default)]
    struct Tally {
        rows: u64,
        decisions: u64,
        fallbacks: u64,
        worst_logit_diff: f32,
        checks: BTreeMap<&'static str, (u64, u64)>,
        examples: Vec<String>,
        model_ms: Vec<f64>,
        total_ms: Vec<f64>,
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

    fn lines(path: &Path) -> Vec<String> {
        let f = std::fs::File::open(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        std::io::BufReader::with_capacity(1 << 20, f)
            .lines()
            .map(|l| l.unwrap())
            .filter(|l| !l.trim().is_empty())
            .collect()
    }

    fn max_diff(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return f32::INFINITY;
        }
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max)
    }

    fn pct(v: &mut [f64], q: f64) -> f64 {
        if v.is_empty() {
            return 0.0;
        }
        v.sort_by(|a, b| a.total_cmp(b));
        v[((v.len() as f64 * q) as usize).min(v.len() - 1)]
    }

    fn rss_mib() -> f64 {
        std::fs::read_to_string("/proc/self/status")
            .ok()
            .and_then(|s| {
                s.lines()
                    .find(|l| l.starts_with("VmRSS:"))
                    .and_then(|l| l.split_whitespace().nth(1))
                    .and_then(|kb| kb.parse::<f64>().ok())
            })
            .map_or(0.0, |kb| kb / 1024.0)
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn facts_e2_68m_after_a_reopen() {
        let (Some(golden), Some(_)) = (
            std::env::var_os("XERJ_E2_GOLDEN_DIR").map(PathBuf::from),
            std::env::var_os("XERJ_FACTS_MODEL_DIR"),
        ) else {
            eprintln!("XERJ_E2_GOLDEN_DIR or XERJ_FACTS_MODEL_DIR not set; E2 frozen-model parity not run");
            return;
        };
        let root =
            PathBuf::from(std::env::var("XERJ_R_ROOT").expect("XERJ_R_ROOT = research/xerj_r_mvp"));
        let model_name =
            std::env::var("XERJ_E2_GOLDEN_MODEL").unwrap_or_else(|_| "x68_dir_s11".into());
        let precision = std::env::var("XERJ_FACTS_PRECISION").unwrap_or_else(|_| "f32".into());
        let f32_logits = !precision.starts_with("int8");
        let only = std::env::var("XERJ_E2_MODEL_SETS").ok();
        let cap: usize = std::env::var("XERJ_E2_MODEL_MAX")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(usize::MAX);
        let skip: usize = std::env::var("XERJ_E2_MODEL_SKIP")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0);

        // golden rows and the worlds that hold their items
        struct Q {
            set: &'static str,
            line: String,
            ns: String,
            phase_after: bool,
        }
        let mut questions: Vec<Q> = Vec::new();
        let mut worlds: Vec<(String, String)> = Vec::new(); // (namespace, world line)
        for (set, data, split, items_file) in SETS {
            let golden_file = golden.join(format!("{model_name}.{set}.jsonl"));
            if let Some(o) = &only {
                if !o.split(',').any(|s| s == set) {
                    continue;
                }
            } else if !golden_file.exists() {
                continue;
            }
            let rows: Vec<String> = lines(&golden_file)
                .into_iter()
                .skip(skip)
                .take(cap)
                .collect();
            // item id -> (world id, phase) from the item file or the worlds' own items
            let mut item_world: HashMap<String, (String, String)> = HashMap::new();
            let world_lines = lines(&root.join(data).join(format!("model_{split}.jsonl")));
            if let Some(f) = items_file {
                for l in lines(&root.join(f)) {
                    let v: Value = serde_json::from_str(&l).unwrap();
                    let wid = v["world_id"].as_str().unwrap().to_string();
                    for it in v["items"].as_array().unwrap() {
                        item_world.insert(
                            it["item_id"].as_str().unwrap().to_string(),
                            (
                                wid.clone(),
                                it["phase"].as_str().unwrap_or("after").to_string(),
                            ),
                        );
                    }
                }
            } else {
                for wl in &world_lines {
                    let w = World::parse(wl).expect("world line");
                    let wid = w.world_id.as_deref().unwrap_or("").to_string();
                    for it in &w.items {
                        item_world.insert(
                            it.item_id.to_string(),
                            (
                                wid.clone(),
                                it.phase.as_deref().unwrap_or("after").to_string(),
                            ),
                        );
                    }
                }
            }
            let mut needed: HashSet<String> = HashSet::new();
            for r in &rows {
                let v: Value = serde_json::from_str(r).unwrap();
                let id = v["item_id"].as_str().unwrap();
                let (wid, _) = item_world
                    .get(id)
                    .unwrap_or_else(|| panic!("{set}: no world holds item {id}"));
                needed.insert(wid.clone());
            }
            let mut ns_of: HashMap<String, String> = HashMap::new();
            for wl in &world_lines {
                let w = World::parse(wl).expect("world line");
                let wid = w.world_id.as_deref().unwrap_or("").to_string();
                if needed.contains(&wid) && !ns_of.contains_key(&wid) {
                    let ns = format!("e2m-{set}-{}", ns_of.len());
                    ns_of.insert(wid, ns.clone());
                    worlds.push((ns, wl.clone()));
                }
            }
            for r in rows {
                let v: Value = serde_json::from_str(&r).unwrap();
                let (wid, phase) = item_world[v["item_id"].as_str().unwrap()].clone();
                questions.push(Q {
                    set,
                    ns: ns_of[&wid].clone(),
                    phase_after: phase != "before",
                    line: r,
                });
            }
        }
        assert!(!questions.is_empty(), "no golden rows selected");

        super::raise_nofile_limit();
        let data_dir = tempfile::tempdir().expect("tempdir");
        let mut revs: HashMap<String, (u64, u64)> = HashMap::new();
        let reopen;
        {
            let (app, state) = app_over(data_dir.path());
            for (ns, wl) in &worlds {
                let w: Value = serde_json::from_str(wl).unwrap();
                let (st, body) = send(&app, post_json(&format!("/_facts/{ns}/_world"), &w)).await;
                assert_eq!(st, StatusCode::OK, "{ns}: {body}");
                revs.insert(
                    ns.clone(),
                    (
                        body["base_rev"].as_u64().unwrap(),
                        body["committed"]["to_rev"].as_u64().unwrap(),
                    ),
                );
            }
            let (st, body) = send(&app, post_json("/_flush", &json!({}))).await;
            assert_eq!(st, StatusCode::OK, "flush: {body}");
            reopen = end_session(app, state).await;
        }

        let mut tally: BTreeMap<&'static str, Tally> = BTreeMap::new();
        let t0 = std::time::Instant::now();
        let (app, state) = app_over(data_dir.path());
        let slot = state.facts.model().expect("the 68M loads");
        let mut rebuilt: HashSet<String> = HashSet::new();
        let mut short: HashSet<String> = HashSet::new();
        for q in &questions {
            let t = tally.entry(q.set).or_default();
            t.rows += 1;
            let (base, fin) = revs[&q.ns];
            if rebuilt.insert(q.ns.clone()) {
                let (st, body) = send(&app, get(&format!("/_facts/{}", q.ns))).await;
                let published = body["published_rev"].as_u64();
                let ok = st == StatusCode::OK && published == Some(fin);
                t.check("rebuilt at the committed revision", ok, || {
                    format!("{}: {st} published {published:?} vs {fin}", q.ns)
                });
                if !ok {
                    short.insert(q.ns.clone());
                }
            }
            if short.contains(&q.ns) {
                continue; // already a mismatch; its rows cannot be asked at their revision
            }
            let n = state.facts.get(&q.ns).expect("projection loaded");
            let row = parse_row(&q.line).expect("golden row");
            let raw: Value = serde_json::from_str(&q.line).unwrap();
            let rev = if q.phase_after { fin } else { base } as u32;
            let mut rec: Vec<Decision> = Vec::new();
            let out = {
                let mem = n.mem.read().expect("projection lock");
                let snap = mem.snapshot_at(rev);
                let mut observe = |s: &Session, ev: &DecisionEvent| {
                    rec.push(Decision {
                        ids: s.input_ids(),
                        op: ev.op,
                        pointer: ev.pointer,
                        op_logits: ev.out.op_logits.clone(),
                        ptr_logits: ev.out.ptr_logits.clone(),
                    })
                };
                answer_observed(
                    &slot,
                    &mem,
                    snap,
                    &row.item_id,
                    &row.question,
                    LinkerMode::Exact,
                    true,
                    Some(&mut observe),
                )
            };
            let id = row.item_id.clone();
            // the candidates the model read
            let got_c: Vec<(String, String, String, String)> = out["candidates"]
                .as_array()
                .map(|a| {
                    a.iter()
                        .map(|c| {
                            let s = |i: usize| c[i].as_str().unwrap_or("").to_string();
                            (s(0), s(1), s(2), s(3))
                        })
                        .collect()
                })
                .unwrap_or_default();
            let exp_c: Vec<(String, String, String, String)> = row
                .cands
                .iter()
                .map(|c| {
                    (
                        c.subject.clone(),
                        c.relation.clone(),
                        c.value.clone(),
                        c.quote.clone(),
                    )
                })
                .collect();
            t.check("candidates", got_c == exp_c, || {
                format!("{id}: {got_c:?} vs {exp_c:?}")
            });
            let got_inv: Vec<bool> = out["candidate_inverse"]
                .as_array()
                .map(|a| a.iter().map(|b| b.as_bool().unwrap_or(false)).collect())
                .unwrap_or_default();
            let exp_inv: Vec<bool> = row.cands.iter().map(|c| c.inverse).collect();
            t.check("candidate inverse flags", got_inv == exp_inv, || {
                format!("{id}: {got_inv:?} vs {exp_inv:?}")
            });
            // the decisions
            t.decisions += rec.len() as u64;
            t.fallbacks += out["fallbacks"].as_u64().unwrap_or(0);
            t.check("decision count", rec.len() == row.decisions.len(), || {
                format!("{id}: {} vs {}", rec.len(), row.decisions.len())
            });
            for (d, g) in rec.iter().zip(&row.decisions) {
                t.check("decision input ids", d.ids == g.ids, || {
                    format!(
                        "{id} step {}: {} ids vs {}",
                        g.step,
                        d.ids.len(),
                        g.ids.len()
                    )
                });
                t.check(
                    "decision op + pointer",
                    d.op == g.op && d.pointer == g.pointer,
                    || {
                        format!(
                            "{id} step {}: ({}, {:?}) vs ({}, {:?})",
                            g.step, d.op, d.pointer, g.op, g.pointer
                        )
                    },
                );
                if f32_logits {
                    let diff = max_diff(&d.op_logits, &g.op_logits)
                        .max(max_diff(&d.ptr_logits, &g.ptr_logits));
                    t.worst_logit_diff = t.worst_logit_diff.max(diff);
                    t.check("decision logits (f32, 5e-4)", diff < 5e-4, || {
                        format!("{id} step {}: max |diff| {diff}", g.step)
                    });
                }
            }
            // the final record
            let fin_ok = out["status"].as_str() == Some(row.final_record.status.as_str())
                && out["answer"].as_str().map(str::to_string) == row.final_record.answer
                && out["evidence_ids"].as_array().map(|a| {
                    a.iter()
                        .filter_map(|v| v.as_str().map(str::to_string))
                        .collect::<Vec<_>>()
                }) == Some(row.final_record.evidence_ids.clone());
            t.check("final answer record", fin_ok, || {
                format!(
                    "{id}: {} {} {} vs {:?} {:?} {:?}",
                    out["status"],
                    out["answer"],
                    out["evidence_ids"],
                    row.final_record.status,
                    row.final_record.answer,
                    row.final_record.evidence_ids
                )
            });
            let got_p: Vec<(String, Option<i64>, bool)> = out["program"]
                .as_array()
                .map(|a| {
                    a.iter()
                        .map(|p| {
                            (
                                p[0].as_str().unwrap_or("").to_string(),
                                p[1].as_i64(),
                                p[2].as_str() == Some("in"),
                            )
                        })
                        .collect()
                })
                .unwrap_or_default();
            let exp_p: Vec<(String, Option<i64>, bool)> = raw["final"]["program"]
                .as_array()
                .map(|a| {
                    a.iter()
                        .map(|p| {
                            (
                                p["op"].as_str().unwrap_or("").to_string(),
                                p["pointer"].as_i64(),
                                p["direction"].as_str() == Some("in"),
                            )
                        })
                        .collect()
                })
                .unwrap_or_default();
            t.check("program with directions", got_p == exp_p, || {
                format!("{id}: {got_p:?} vs {exp_p:?}")
            });
            if let Some(us) = out["timings_us"]["model"].as_f64() {
                t.model_ms.push(us / 1e3);
            }
            if let Some(us) = out["timings_us"]["total"].as_f64() {
                t.total_ms.push(us / 1e3);
            }
        }
        let secs = t0.elapsed().as_secs_f64();

        let mut cols: Vec<&'static str> = Vec::new();
        for t in tally.values() {
            for k in t.checks.keys() {
                if !cols.contains(k) {
                    cols.push(k);
                }
            }
        }
        let mut md = format!(
            "precision {precision}, model {model_name}, {} questions (rows {skip}.. of each set) in {secs:.1} s, RSS {:.0} MiB at the end; {reopen}\n\n",
            questions.len(),
            rss_mib()
        );
        md.push_str(&format!(
            "| set | rows | decisions | guard fallbacks | worst f32 logit diff | model ms p50 / p95 | total ms p50 / p95 | {} |\n",
            cols.join(" | ")
        ));
        md.push_str(&format!(
            "|---|---:|---:|---:|---:|---:|---:|{}\n",
            "---:|".repeat(cols.len())
        ));
        let mut fails = 0;
        for (name, t) in tally.iter_mut() {
            let cells: Vec<String> = cols
                .iter()
                .map(|c| {
                    t.checks
                        .get(c)
                        .map_or("-".into(), |(ok, n)| format!("{ok}/{n}"))
                })
                .collect();
            let (m50, m95, t50, t95) = (
                pct(&mut t.model_ms, 0.5),
                pct(&mut t.model_ms, 0.95),
                pct(&mut t.total_ms, 0.5),
                pct(&mut t.total_ms, 0.95),
            );
            md.push_str(&format!(
                "| {name} | {} | {} | {} | {:.2e} | {m50:.0} / {m95:.0} | {t50:.0} / {t95:.0} | {} |\n",
                t.rows,
                t.decisions,
                t.fallbacks,
                t.worst_logit_diff,
                cells.join(" | ")
            ));
            fails += t.failures();
        }
        println!("{md}");
        for (name, t) in &tally {
            for e in &t.examples {
                println!("MISMATCH {name}: {e}");
            }
        }
        if let Some(p) = std::env::var_os("XERJ_E2_MODEL_REPORT") {
            let _ = std::fs::write(p, &md);
        }
        assert_eq!(fails, 0, "{fails} E2 frozen-model mismatches ({precision})");
    }
}
