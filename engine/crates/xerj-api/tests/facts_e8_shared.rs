//! E8 shared store, exact path without the model (design `research/product/XERJ-R_engine_integration_design_2026-10-26.md`
//! §6.4; integration plan E8): every test world in ONE facts namespace, against one store per world.
//!
//! The shared namespace is written over HTTP by the release server (`e8_run.py load`: `POST /_facts/{ns}/_bulk`, phase 1 =
//! every world's base documents and facts plus background facts, phase 2 = every edit), which is then stopped. This test
//! opens that data directory in a new process (a real restart), so the shared projection is rebuilt from the engine's index
//! through the product path (`GET /_facts/{ns}`), and compares it question by question with the per-world reference store
//! (`xerj_serve::engine::build_world`: the research path, which E2 proves identical to a per-world `/_facts` namespace):
//!
//! * linking (subjects and values) and the stack-v5 K = 16 candidate manifest, with the default item id
//!   ("lab/q/" + sha256(question)[:16]: no world id enters the run) at the revision of the question's phase;
//! * whether the gold LOOKUP records are in that manifest (candidate recall);
//! * the gold trace executed by the exact executor (oracle program: link, manifest, hop manifests, executor), i.e. the
//!   full-path answer under oracle intermediate entities;
//! * per gold step, the store's own lookup of the step key (status, active values) and, for inverse steps, the subjects
//!   that share the record's (relation, value): whether the question is still identifiable in the shared store.
//!
//! `XERJ_E8_DIRECT` instead of a data dir builds the shared projection straight from the corpus batches (the projection
//! `_bulk` maintains, without the engine index): for store sizes whose load through the index does not fit the run.
//!
//! Inputs (nothing runs without XERJ_E8_DATA_DIR or XERJ_E8_DIRECT): XERJ_E8_DATA_DIR (data dir of the stopped server), XERJ_E8_CORPUS
//! (questions.jsonl), XERJ_E8_RUN (load.json: namespace, revisions B and F), XERJ_R_ROOT (runs/world), XERJ_E8_OUT (JSONL,
//! one line per question), XERJ_E8_THREADS (default 4), XERJ_E8_MAX (questions cap).
//!
//! The second exact-path arm (`perworld_cf` / `shared_cf`) runs the opt-in linker XERJ_E8_CF_LINKER (default `casefold`;
//! `casefold_v3`) with the manifest choice XERJ_E8_CF_MANIFEST (default `reference`; `relation_aware`), as `_ask` runs them.
//! Model linker specs (XERJ_E8_MODEL_LINKER, comma-separated): a linker name, optionally with `+ra` for the relation-aware
//! manifest (`casefold_v3+ra`); each spec writes its own subdirectory.
//!
//! Round 3 (opt-in `xerj_facts::AskExt`, shared arm only; flags after the linker, `+`-separated, e.g.
//! `casefold_v3+ra+coherent`): `scope_own` / `scope_ownbg` (the question names its own world, without / with the
//! Wikidata5M background: base and hop manifests offer only those records), `nohub` (every collection but the
//! counterfactual hub world: as if that world had its own namespace), `prefer_own` (own-world records first where a cap
//! binds), `coherent` (no world named: hops only to records that share a world with every record the program read),
//! `loop_unknown` / `loop_first` (`xerj_exec::LoopGuard`). The record -> collections map is built from the corpus's
//! `world_facts.jsonl` (collection 0 = the background, `wd5m/` ids; worlds 1.. in world-id order). The per-world arm runs
//! only the loop guard (a per-world store holds one world).
#![cfg(feature = "facts")]

use axum::body::Body;
use axum::http::{Request, StatusCode};
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashMap};
use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Instant;
use tower::ServiceExt;
use xerj_facts::xerj_exec::episode::{run_gold, Episode, GoldStep};
use xerj_facts::xerj_index::direction::{max_inverse, RelPref};
use xerj_facts::xerj_index::linker::QBuf;
use xerj_facts::xerj_index::Memory;
use xerj_facts::xerj_serve::engine::{build_world, lab_item_id};
use xerj_facts::xerj_store::value::render_value;
use xerj_facts::xerj_store::{Rev, Snapshot, Status};

const K: usize = 16;

fn app_over(dir: &Path) -> (axum::Router, xerj_api::state::AppState) {
    let mut config = xerj_common::config::Config::default();
    config.server.data_dir = dir.to_string_lossy().into_owned();
    let metrics = xerj_common::metrics::Metrics::new().expect("metrics");
    let engine = xerj_engine::Engine::new(config.clone()).expect("engine");
    let state = xerj_api::state::AppState::new(config, engine, metrics);
    (
        xerj_api::router::build_es_compat_router(state.clone()),
        state,
    )
}

async fn send(app: &axum::Router, req: Request<Body>) -> (StatusCode, Value) {
    let response = app.clone().oneshot(req).await.expect("response");
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body");
    let body = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    (status, body)
}

fn rss_mib() -> (f64, f64) {
    let s = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    let kb = |k: &str| {
        s.lines()
            .find(|l| l.starts_with(k))
            .and_then(|l| l.split_whitespace().nth(1))
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(0.0)
    };
    (kb("VmRSS:") / 1024.0, kb("VmHWM:") / 1024.0)
}

fn lines(path: &Path) -> Vec<String> {
    let f = std::fs::File::open(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    std::io::BufReader::with_capacity(1 << 20, f)
        .lines()
        .map(|l| l.expect("line"))
        .filter(|l| !l.trim().is_empty())
        .collect()
}

fn gold_steps(trace: &Value) -> Vec<GoldStep> {
    trace
        .as_array()
        .map(|t| {
            t.iter()
                .map(|s| GoldStep {
                    op: s["op"].as_str().unwrap_or("").to_string(),
                    key: s.get("key").and_then(|k| k.as_array()).map(|k| {
                        (
                            k[0].as_str().unwrap_or("").to_string(),
                            k[1].as_str().unwrap_or("").to_string(),
                        )
                    }),
                    inverse: s.get("direction").and_then(|d| d.as_str()) == Some("in"),
                })
                .collect()
        })
        .unwrap_or_default()
}

/// The opt-in arm of the exact path: a linker (`casefold` | `casefold_v3`) and the manifest choice, as `_ask` runs them.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Variant {
    Reference,
    Casefold { v3: bool, relation_aware: bool },
}

/// One arm (a store at a revision) for one question; a `Variant` other than `Reference` = the opt-in pre-link
/// (`xerj_facts::casefold_link` / `casefold_v3_link`) before the exact pipeline, with the relation-aware manifest choice
/// when asked, as `_ask` with that `linker` / `manifest` runs it.
fn run_arm(
    mem: &Memory,
    rev: Rev,
    q: &Value,
    steps: &[GoldStep],
    buf: &mut QBuf,
    variant: Variant,
) -> Value {
    let st = &mem.store;
    let snap: Snapshot<'_> = mem.snapshot_at(rev);
    let original = q["question"].as_str().unwrap_or("");
    // the manifest salt is always the original question's
    let item_id = lab_item_id(original);
    let t0 = Instant::now();
    let (v3, relation_aware) = match variant {
        Variant::Reference => (false, false),
        Variant::Casefold { v3, relation_aware } => (v3, relation_aware),
    };
    let cl = (variant != Variant::Reference && !v3)
        .then(|| xerj_facts::casefold_link(mem, &snap, original));
    let sl = v3.then(|| xerj_facts::casefold_v3_link(mem, &snap, original));
    let question: &str = match (&cl, &sl) {
        (Some(c), _) => c.rewritten.as_str(),
        (_, Some(l)) => l.case.rewritten.as_str(),
        _ => original,
    };
    let ents: Vec<u32> = match &sl {
        Some(l) => l.subjects.clone(),
        None => mem.link(&snap, question, buf).to_vec(),
    };
    let link: Vec<String> = ents
        .iter()
        .map(|&s| st.pools.subjects.get(s).to_string())
        .collect();
    // tier 1 of the manifest: the visible records of the linked subjects (more than K = truncated in fact-id order)
    let tier1: usize = ents
        .iter()
        .map(|&s| {
            mem.subjects
                .records_of_subject(s)
                .iter()
                .filter(|&&r| snap.visible(r))
                .count()
        })
        .sum();
    let values: Vec<String> = mem
        .link_values(&snap, question, buf)
        .iter()
        .map(|&s| st.pools.strs.get(s).to_string())
        .collect();
    let pref = relation_aware.then(|| std::sync::Arc::new(RelPref::from_question(question)));
    let m = if variant == Variant::Reference {
        mem.candidate_manifest_v5(&snap, &item_id, question, K, buf)
    } else {
        let given = sl
            .as_ref()
            .filter(|l| l.changes_subjects())
            .map(|l| l.subjects.as_slice());
        mem.candidate_manifest_v5_opts(
            &snap,
            &item_id,
            question,
            K,
            max_inverse(),
            given,
            pref.as_deref(),
            buf,
        )
    };
    let us_manifest = t0.elapsed().as_secs_f64() * 1e6;
    let cands: Vec<&str> = m.cands.iter().map(|&r| st.fact_id(r)).collect();
    // gold LOOKUP steps (the base manifest's part of the trace): is a record with the step key and direction a candidate?
    let mut lookups = 0usize;
    let mut found = 0usize;
    for s in steps.iter().filter(|s| s.op == "LOOKUP") {
        if let Some((subj, rel)) = &s.key {
            lookups += 1;
            if m.cands.iter().enumerate().any(|(i, &r)| {
                st.subject(r) == subj
                    && st.relation(r) == rel
                    && m.inverse.get(i).copied().unwrap_or(false) == s.inverse
            }) {
                found += 1;
            }
        }
    }
    // the store's view of every gold step key
    let step_info: Vec<Value> = steps
        .iter()
        .map(|s| {
            let Some((subj, rel)) = &s.key else {
                return json!({"op": s.op});
            };
            let l = snap.lookup(subj, rel, "default");
            let vals: Vec<String> = l
                .refs
                .iter()
                .map(|&r| render_value(&st.typed_value(r)))
                .collect();
            let mut o = json!({"op": s.op, "key": [subj, rel], "inverse": s.inverse, "status": l.status.value_str(),
                               "refs": l.refs.len(), "values": vals});
            if s.inverse {
                // the subjects that share this record's (relation, value): more than one = the question is a set
                let mut subs: Vec<String> = Vec::new();
                for &r in &l.refs {
                    for x in mem.inverse_same_value(&snap, r) {
                        let n = st.subject(x).to_string();
                        if !subs.contains(&n) {
                            subs.push(n);
                        }
                    }
                }
                o["inverse_subjects"] = json!(subs.len());
                o["inverse_subjects_head"] = json!(subs.iter().take(5).collect::<Vec<_>>());
            }
            if matches!(l.status, Status::Conflict) {
                o["conflict"] = json!(true);
            }
            o
        })
        .collect();
    let mut ep = Episode::new_v5(mem, snap, m.cands.clone(), m.inverse.clone(), &item_id, K);
    ep.rel_pref = pref.clone();
    let t1 = Instant::now();
    let (ans, problems) = run_gold(&mut ep, steps);
    let us_exec = t1.elapsed().as_secs_f64() * 1e6;
    let mut o = json!({
        "rev": rev, "link": link, "tier1": tier1, "values": values, "cands": cands, "inv": m.inverse,
        "gold_lookups": lookups, "gold_lookups_in_cands": found,
        "oracle": {"status": ans.status, "answer": ans.answer,
                   "evidence_ids": ans.evidence_ids.iter().map(|&r| st.fact_id(r)).collect::<Vec<_>>()},
        "problems": problems, "steps": step_info,
        "us_link_manifest": (us_manifest * 10.0).round() / 10.0, "us_oracle_exec": (us_exec * 10.0).round() / 10.0,
    });
    if let Some(c) = cl.as_ref().or(sl.as_ref().map(|l| &l.case)) {
        o["casefold_matches"] = json!(c.matches.len());
        o["casefold_ambiguous"] = json!(c.ambiguous.len());
        if c.rewritten != original {
            o["rewritten"] = json!(c.rewritten);
        }
    }
    if let Some(l) = &sl {
        if !l.dropped.is_empty() {
            o["dropped"] = json!(l.dropped);
        }
        if !l.merged.is_empty() {
            o["merged"] = json!(l.merged);
        }
    }
    o
}

/// the exact path's opt-in arm from XERJ_E8_CF_LINKER / XERJ_E8_CF_MANIFEST
fn cf_variant() -> Variant {
    let v3 = match std::env::var("XERJ_E8_CF_LINKER").as_deref() {
        Err(_) | Ok("casefold") => false,
        Ok("casefold_v3") => true,
        Ok(o) => panic!("XERJ_E8_CF_LINKER {o}: casefold or casefold_v3"),
    };
    let relation_aware = match std::env::var("XERJ_E8_CF_MANIFEST").as_deref() {
        Err(_) | Ok("reference") => false,
        Ok("relation_aware") => true,
        Ok(o) => panic!("XERJ_E8_CF_MANIFEST {o}: reference or relation_aware"),
    };
    Variant::Casefold { v3, relation_aware }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn facts_e8_shared_store_exact_path() {
    let data = std::env::var_os("XERJ_E8_DATA_DIR").map(PathBuf::from);
    let direct = std::env::var_os("XERJ_E8_DIRECT").is_some();
    if data.is_none() && !direct {
        eprintln!("neither XERJ_E8_DATA_DIR nor XERJ_E8_DIRECT is set: the E8 shared-store harness is not run");
        return;
    }
    let corpus = PathBuf::from(std::env::var_os("XERJ_E8_CORPUS").expect("XERJ_E8_CORPUS"));
    let root =
        PathBuf::from(std::env::var_os("XERJ_R_ROOT").expect("XERJ_R_ROOT")).join("runs/world");
    let out_path = PathBuf::from(std::env::var_os("XERJ_E8_OUT").expect("XERJ_E8_OUT"));
    let threads: usize = std::env::var("XERJ_E8_THREADS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4);
    let cap: usize = std::env::var("XERJ_E8_MAX")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(usize::MAX);
    // XERJ_E8_SAMPLE_ONLY: only the model sample (the store-size sweep's large end)
    let sample_only = std::env::var_os("XERJ_E8_SAMPLE_ONLY").is_some();
    let mut summary = json!({"threads": threads, "direct": direct, "sample_only": sample_only});
    // the engine stays open while the projection is read
    let mut _engine: Option<(axum::Router, xerj_api::state::AppState)> = None;
    let (shared, rev_b, rev_f): (std::sync::Arc<xerj_facts::Namespace>, Rev, Rev) = if let Some(
        data,
    ) = data
    {
        let run = PathBuf::from(std::env::var_os("XERJ_E8_RUN").expect("XERJ_E8_RUN"));
        let load: Value = serde_json::from_str(
            &std::fs::read_to_string(run.join("load.json")).expect("load.json"),
        )
        .expect("load.json");
        let ns = load["shared"].as_str().expect("shared").to_string();
        let (rev_b, rev_f) = (
            load["shared_B"].as_u64().expect("B") as Rev,
            load["shared_F"].as_u64().expect("F") as Rev,
        );
        summary["namespace"] = json!(ns);
        // restart: open the server's data dir; the first access rebuilds the projection from the index
        let t = Instant::now();
        let (app, state) = app_over(&data);
        summary["engine_open_s"] = json!(t.elapsed().as_secs_f64());
        let t = Instant::now();
        let (st, body) = send(
            &app,
            Request::get(format!("/_facts/{ns}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        assert_eq!(st, StatusCode::OK, "rebuild of {ns}: {body}");
        summary["rebuild_s"] = json!(t.elapsed().as_secs_f64());
        summary["status_after_rebuild"] = body.clone();
        assert_eq!(
            body["published_rev"].as_u64(),
            Some(u64::from(rev_f)),
            "the rebuilt projection publishes the last committed revision"
        );
        let (rss, hwm) = rss_mib();
        summary["rss_mib_after_rebuild"] = json!(rss);
        summary["hwm_mib_after_rebuild"] = json!(hwm);
        let shared = state.facts.get(&ns).expect("projection loaded");
        _engine = Some((app, state));
        (shared, rev_b, rev_f)
    } else {
        // projection only (no engine index): every corpus batch applied as `POST /_facts/{ns}/_bulk` applies it
        use std::sync::atomic::Ordering;
        let n = xerj_facts::Namespace::new();
        let t = Instant::now();
        let mut rev_b: Rev = 0;
        let mut batches = 0usize;
        for (phase, file) in [("phase1", "phase1.jsonl"), ("phase2", "phase2.jsonl")] {
            let f = std::fs::File::open(corpus.join(file)).expect("corpus batch file");
            for l in std::io::BufReader::with_capacity(1 << 22, f).lines() {
                let l = l.expect("line");
                if l.trim().is_empty() {
                    continue;
                }
                let mut b: Value = serde_json::from_str(&l).expect("batch");
                let mut take = |k: &str| {
                    let v = b[k].take();
                    if v.is_null() {
                        json!([])
                    } else {
                        v
                    }
                };
                let world = json!({"base_documents": take("documents"), "base_facts": take("facts"), "edits": take("edits")});
                let ap = n.apply(&xerj_facts::Batch::World { world }).expect("apply");
                n.published.store(ap.to_rev, Ordering::SeqCst);
                batches += 1;
            }
            if phase == "phase1" {
                rev_b = n.published.load(Ordering::SeqCst);
            }
        }
        let rev_f = n.published.load(Ordering::SeqCst);
        summary["namespace"] = json!("direct");
        summary["direct_batches"] = json!(batches);
        summary["direct_apply_s"] = json!(t.elapsed().as_secs_f64());
        summary["status_after_apply"] = n.status(false);
        let (rss, hwm) = rss_mib();
        summary["rss_mib_after_apply"] = json!(rss);
        summary["hwm_mib_after_apply"] = json!(hwm);
        (std::sync::Arc::new(n), rev_b, rev_f)
    };
    summary["B"] = json!(rev_b);
    summary["F"] = json!(rev_f);

    // questions grouped by world (one per-world reference store each)
    let mut by_world: BTreeMap<(String, String), Vec<Value>> = BTreeMap::new();
    let mut n = 0usize;
    for l in lines(&corpus.join("questions.jsonl")) {
        if n == cap {
            break;
        }
        let q: Value = serde_json::from_str(&l).expect("question");
        if sample_only && q["sample"].as_bool() != Some(true) {
            continue;
        }
        let key = (
            q["world_file"].as_str().unwrap().to_string(),
            q["world_id"].as_str().unwrap().to_string(),
        );
        by_world.entry(key).or_default().push(q);
        n += 1;
    }
    // the worlds, read once per file
    let files: Vec<String> = {
        let mut f: Vec<String> = by_world.keys().map(|k| k.0.clone()).collect();
        f.dedup();
        f
    };
    let mut worlds: HashMap<(String, String), Value> = HashMap::new();
    for f in &files {
        for l in lines(&root.join(f)) {
            let w: Value = serde_json::from_str(&l).expect("world");
            let id = w["world_id"].as_str().unwrap_or("").to_string();
            if by_world.contains_key(&(f.clone(), id.clone())) {
                worlds.insert((f.clone(), id), w);
            }
        }
    }
    summary["questions"] = json!(n);
    summary["worlds"] = json!(by_world.len());

    let skip_exact = std::env::var("XERJ_E8_SKIP_EXACT").is_ok();
    let cfv = cf_variant();
    summary["exact_cf_arm"] = json!({"linker": std::env::var("XERJ_E8_CF_LINKER").unwrap_or_else(|_| "casefold".into()),
        "manifest": std::env::var("XERJ_E8_CF_MANIFEST").unwrap_or_else(|_| "reference".into())});
    // XERJ_E8_REPEAT = N passes of the exact path (latency runs); the last pass writes XERJ_E8_OUT, the others
    // <XERJ_E8_OUT stem>.pass<k>.jsonl
    let repeat: usize = std::env::var("XERJ_E8_REPEAT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1)
        .max(1);
    let loadavg = || {
        std::fs::read_to_string("/proc/loadavg")
            .unwrap_or_default()
            .split_whitespace()
            .take(3)
            .map(String::from)
            .collect::<Vec<_>>()
    };
    let t = Instant::now();
    let jobs: Vec<(&(String, String), &Vec<Value>)> = if skip_exact {
        Vec::new()
    } else {
        by_world.iter().collect()
    };
    let guard = shared.mem.read().expect("projection lock");
    let smem: &Memory = &guard;
    // warm the two fingerprints (O(N) once per revision, cached)
    let _ = smem.snapshot_at(rev_b).fingerprint();
    let _ = smem.snapshot_at(rev_f).fingerprint();
    summary["fingerprints_s"] = json!(t.elapsed().as_secs_f64());
    let mut passes = Vec::new();
    for pass in 0..repeat {
        let path = if pass + 1 == repeat {
            out_path.clone()
        } else {
            out_path.with_extension(format!("pass{pass}.jsonl"))
        };
        let out = Mutex::new(std::io::BufWriter::new(
            std::fs::File::create(&path).expect("out"),
        ));
        let next = std::sync::atomic::AtomicUsize::new(0);
        let tp = Instant::now();
        let la0 = loadavg();
        std::thread::scope(|sc| {
            for _ in 0..threads {
                sc.spawn(|| {
                let mut buf = QBuf::default();
                loop {
                    let i = next.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    let Some((key, qs)) = jobs.get(i) else { break };
                    let w = worlds.get(*key).expect("world of a question");
                    let built = build_world(w, None).expect("per-world reference store");
                    let mut rows = Vec::with_capacity(qs.len());
                    for q in qs.iter() {
                        let steps = gold_steps(&q["trace"]);
                        let before = q["phase"].as_str() == Some("before");
                        let pw_rev = if before { built.base } else { built.final_ };
                        let sh_rev = if before { rev_b } else { rev_f };
                        let pw = run_arm(&built.mem, pw_rev, q, &steps, &mut buf, Variant::Reference);
                        let sh = run_arm(smem, sh_rev, q, &steps, &mut buf, Variant::Reference);
                        let pwc = run_arm(&built.mem, pw_rev, q, &steps, &mut buf, cfv);
                        let shc = run_arm(smem, sh_rev, q, &steps, &mut buf, cfv);
                        rows.push(
                            json!({"qid": q["qid"], "perworld": pw, "shared": sh, "perworld_cf": pwc, "shared_cf": shc})
                                .to_string(),
                        );
                    }
                    let mut o = out.lock().unwrap();
                    for r in rows {
                        writeln!(o, "{r}").unwrap();
                    }
                }
            });
            }
        });
        out.lock().unwrap().flush().unwrap();
        passes.push(json!({"pass": pass, "file": path.display().to_string(), "s": tp.elapsed().as_secs_f64(),
                       "loadavg_start": la0, "loadavg_end": loadavg()}));
    }
    drop(guard);
    summary["exact_passes"] = json!(passes);
    summary["exact_path_s"] = json!(t.elapsed().as_secs_f64());
    let (rss, hwm) = rss_mib();
    summary["rss_mib_end"] = json!(rss);
    summary["hwm_mib_end"] = json!(hwm);
    if let Some(mo) = std::env::var_os("XERJ_E8_MODEL_OUT") {
        let linkers = std::env::var("XERJ_E8_MODEL_LINKER").unwrap_or_else(|_| "exact".into());
        let many = linkers.contains(',');
        let mut runs = Vec::new();
        for l in linkers.split(',') {
            let dir = if many {
                PathBuf::from(&mo).join(l)
            } else {
                PathBuf::from(&mo)
            };
            runs.push(model_arm(
                &shared, rev_b, rev_f, &by_world, &worlds, threads, l, &dir,
            ));
        }
        summary["model"] = json!(runs);
    }
    let sp = out_path.with_extension("summary.json");
    std::fs::write(&sp, serde_json::to_string_pretty(&summary).unwrap()).unwrap();
    eprintln!("{}", serde_json::to_string_pretty(&summary).unwrap());
}

/// The model arm in process (`XERJ_E8_MODEL_OUT` = output dir; with several linkers, one subdirectory each): every sampled
/// question through `xerj_facts::ask` (the
/// handler's own function: link, manifest, compiler decisions on the exact executor, verify) in the shared projection and
/// in a per-world namespace built from the world by `Namespace::apply` (the `_world` path without the index), with the
/// default item id and `XERJ_E8_MODEL_LINKER` (default "exact"). Writes asks_perworld.jsonl and asks_shared.jsonl in the
/// format of `e8_run.py ask`. `XERJ_E8_MODEL_SETS` limits the sets. The model is `XERJ_FACTS_MODEL_DIR` at
/// `XERJ_FACTS_PRECISION`.
#[allow(clippy::too_many_arguments)]
fn model_arm(
    shared: &xerj_facts::Namespace,
    rev_b: Rev,
    rev_f: Rev,
    by_world: &BTreeMap<(String, String), Vec<Value>>,
    worlds: &HashMap<(String, String), Value>,
    threads: usize,
    linker: &str,
    dir: &Path,
) -> Value {
    use std::sync::atomic::Ordering;
    std::fs::create_dir_all(dir).expect("model out dir");
    // a spec is a linker name, optionally "+ra" (the relation-aware manifest choice) and round-3 flags
    let mut parts = linker.split('+');
    let linker = parts.next().unwrap_or("exact").to_string();
    let mut manifest = None;
    let mut scope_kind: Option<&str> = None;
    let mut loop_guard = xerj_facts::xerj_exec::LoopGuard::Off;
    for f in parts {
        match f {
            "ra" => manifest = Some("relation_aware".to_string()),
            "scope_own" | "scope_ownbg" | "nohub" | "prefer_own" | "coherent" => {
                assert!(scope_kind.is_none(), "one scope flag per spec");
                scope_kind = Some(match f {
                    "scope_own" => "scope_own",
                    "scope_ownbg" => "scope_ownbg",
                    "nohub" => "nohub",
                    "prefer_own" => "prefer_own",
                    _ => "coherent",
                });
            }
            "loop_unknown" => loop_guard = xerj_facts::xerj_exec::LoopGuard::Unknown,
            "loop_first" => loop_guard = xerj_facts::xerj_exec::LoopGuard::FirstUse,
            o => panic!("spec flag {o}: ra, scope_own, scope_ownbg, nohub, prefer_own, coherent, loop_unknown, loop_first"),
        }
    }
    let ext_set = scope_kind.is_some() || loop_guard != xerj_facts::xerj_exec::LoopGuard::Off;
    // the shared store's record -> collections map (round-3 scope flags)
    let t_map = Instant::now();
    let cols = scope_kind.map(|_| collections_of_shared(shared));
    let map_s = t_map.elapsed().as_secs_f64();
    // XERJ_E8_MODEL_ARMS = "perworld,shared" (default) | "shared" | "perworld"
    let arms = std::env::var("XERJ_E8_MODEL_ARMS").unwrap_or_else(|_| "perworld,shared".into());
    let (do_pw, do_sh) = (arms.contains("perworld"), arms.contains("shared"));
    let sets: Option<Vec<String>> = std::env::var("XERJ_E8_MODEL_SETS")
        .ok()
        .map(|s| s.split(',').map(String::from).collect());
    let svc = xerj_facts::FactsService::from_env();
    let t = Instant::now();
    let slot = svc.model().expect("model");
    let load_s = t.elapsed().as_secs_f64();
    let jobs: Vec<(&(String, String), Vec<&Value>)> = by_world
        .iter()
        .map(|(k, qs)| {
            let v: Vec<&Value> = qs
                .iter()
                .filter(|q| q["sample"].as_bool() == Some(true))
                .filter(|q| {
                    sets.as_ref()
                        .is_none_or(|s| s.iter().any(|x| Some(x.as_str()) == q["set"].as_str()))
                })
                .collect();
            (k, v)
        })
        .filter(|(_, v)| !v.is_empty())
        .collect();
    let n_q: usize = jobs.iter().map(|(_, v)| v.len()).sum();
    let outs = Mutex::new((
        std::io::BufWriter::new(std::fs::File::create(dir.join("asks_perworld.jsonl")).unwrap()),
        std::io::BufWriter::new(std::fs::File::create(dir.join("asks_shared.jsonl")).unwrap()),
    ));
    let next = std::sync::atomic::AtomicUsize::new(0);
    let t = Instant::now();
    let row = |arm: &str,
               ns: &str,
               rev: Rev,
               q: &Value,
               r: Result<Value, xerj_facts::FactsError>,
               ms: f64|
     -> String {
        match r {
            Ok(v) => {
                let mut o = json!({"qid": q["qid"], "arm": arm, "ns": ns, "revision": rev, "client_ms": (ms * 10.0).round() / 10.0});
                for k in ["status", "answer", "evidence_ids", "program", "final_status", "final_answer", "verify", "decisions",
                          "fallbacks", "timings_us", "candidate_ids", "candidate_inverse", "hop_reads", "item_id", "engine_error",
                          "published_rev", "trace", "link", "question_used", "answer_opts"] {
                    o[k] = v.get(k).cloned().unwrap_or(Value::Null);
                }
                o["evidence"] = json!(v["evidence"].as_array().map(|a| a.iter().map(|e| json!({"fact_id": e["fact_id"],
                    "subject": e["subject"], "relation": e["relation"], "value": e["value"]})).collect::<Vec<_>>()));
                o.to_string()
            }
            Err(e) => json!({"qid": q["qid"], "arm": arm, "ns": ns, "revision": rev, "http": 500, "error": format!("{e:?}")}).to_string(),
        }
    };
    std::thread::scope(|sc| {
        for _ in 0..threads {
            sc.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::SeqCst);
                let Some((key, qs)) = jobs.get(i) else { break };
                let w = worlds.get(*key).expect("world");
                let n = xerj_facts::Namespace::new();
                let ap = n
                    .apply(&xerj_facts::Batch::World { world: w.clone() })
                    .expect("per-world namespace");
                n.published.store(ap.to_rev, Ordering::SeqCst);
                let (base, fin) = (ap.base_rev.unwrap_or(ap.to_rev), ap.to_rev);
                let mut lines = Vec::new();
                for q in qs {
                    let before = q["phase"].as_str() == Some("before");
                    let ask = |ns: &xerj_facts::Namespace, rev: Rev, shared_arm: bool| {
                        let req = xerj_facts::AskReq {
                            question: q["question"].as_str().unwrap_or("").to_string(),
                            revision: Some(rev),
                            item_id: None,
                            linker: Some(linker.clone()),
                            manifest: manifest.clone(),
                            verify: None,
                            explain: true,
                        };
                        // round 3: the scope only in the shared arm (its map names the shared store's records)
                        let scope = match (&cols, shared_arm) {
                            (Some(c), true) => Some(question_scope(c, scope_kind.unwrap(), q)),
                            _ => None,
                        };
                        let ext = xerj_facts::AskExt { scope, loop_guard };
                        let t0 = Instant::now();
                        let r = if ext_set {
                            xerj_facts::ask_ext(&slot, ns, &req, &ext)
                        } else {
                            xerj_facts::ask(&slot, ns, &req)
                        };
                        (r, t0.elapsed().as_secs_f64() * 1e3)
                    };
                    // with the casefold linker, a question the pre-link leaves unchanged runs exactly as with the exact
                    // linker: it is not asked again (the scorer takes the exact-linker row of the same arm)
                    // (not with the relation-aware manifest: a cap may bind in any manifest of the episode)
                    let unchanged = |ns: &xerj_facts::Namespace, rev: Rev| -> bool {
                        if manifest.is_some() || ext_set {
                            return false;
                        }
                        let g = ns.mem.read().expect("projection lock");
                        let snap = g.snapshot_at(rev);
                        let question = q["question"].as_str().unwrap_or("");
                        match linker.as_str() {
                            "casefold" => xerj_facts::casefold_link(&g, &snap, question).rewritten == question,
                            "casefold_v3" => {
                                let l = xerj_facts::casefold_v3_link(&g, &snap, question);
                                l.case.rewritten == question && !l.changes_subjects()
                            }
                            _ => false,
                        }
                    };
                    let same = |arm: &str, ns: &str, rev: Rev| {
                        json!({"qid": q["qid"], "arm": arm, "ns": ns, "revision": rev, "same_as_exact": true}).to_string()
                    };
                    let pr = if before { base } else { fin };
                    let lp = if !do_pw {
                        String::new()
                    } else if unchanged(&n, pr) {
                        same("perworld", "inproc", pr)
                    } else {
                        let (r, ms) = ask(&n, pr, false);
                        row("perworld", "inproc", pr, q, r, ms)
                    };
                    let sr = if before { rev_b } else { rev_f };
                    let ls = if !do_sh {
                        String::new()
                    } else if unchanged(shared, sr) {
                        same("shared", "shared", sr)
                    } else {
                        let (r, ms) = ask(shared, sr, true);
                        row("shared", "shared", sr, q, r, ms)
                    };
                    lines.push((lp, ls));
                }
                let mut o = outs.lock().unwrap();
                for (a, b) in lines {
                    if !a.is_empty() {
                        writeln!(o.0, "{a}").unwrap();
                    }
                    if !b.is_empty() {
                        writeln!(o.1, "{b}").unwrap();
                    }
                }
            });
        }
    });
    let mut o = outs.lock().unwrap();
    o.0.flush().unwrap();
    o.1.flush().unwrap();
    let (rss, hwm) = rss_mib();
    json!({"linker": linker, "manifest": manifest, "arms": arms, "questions": n_q, "worlds": jobs.len(), "model_load_s": load_s,
           "asks_s": t.elapsed().as_secs_f64(), "threads": threads, "rss_mib": rss, "hwm_mib": hwm,
           "scope": scope_kind, "loop_guard": loop_guard.name(),
           "collections_map": cols.as_ref().map(|c| json!({"records": c.map.len(), "collections": c.n, "hub": c.hub.len(), "build_s": map_s})),
           "model": slot.info()})
}

/// Round 3: the shared store's record -> collections map from the corpus (`XERJ_E8_CORPUS/world_facts.jsonl`): collection
/// 0 = the Wikidata5M background (fact ids `wd5m/...`, `0wd5m/...`), worlds 1.. in world-id order (a fact written once for
/// several worlds belongs to each); `hub` = the collections of the counterfactual hub worlds (world ids `hub-...`).
struct SharedCollections {
    map: std::sync::Arc<xerj_facts::xerj_index::scope::RecCollections>,
    world_col: HashMap<String, u32>,
    hub: Vec<u32>,
    /// every collection but the hub worlds (sorted)
    not_hub: Vec<u32>,
    n: usize,
}

fn collections_of_shared(shared: &xerj_facts::Namespace) -> SharedCollections {
    let corpus = PathBuf::from(std::env::var_os("XERJ_E8_CORPUS").expect("XERJ_E8_CORPUS"));
    let mut worlds: Vec<(String, Vec<String>)> = lines(&corpus.join("world_facts.jsonl"))
        .iter()
        .map(|l| {
            let w: Value = serde_json::from_str(l).expect("world_facts line");
            let facts = w["facts"]
                .as_array()
                .map(|a| {
                    a.iter()
                        .filter_map(|f| f.as_str().map(String::from))
                        .collect()
                })
                .unwrap_or_default();
            (w["world_id"].as_str().unwrap_or("").to_string(), facts)
        })
        .collect();
    worlds.sort_by(|a, b| a.0.cmp(&b.0));
    let mut world_col = HashMap::new();
    let mut of_fact: HashMap<&str, Vec<u32>> = HashMap::new();
    let mut hub = Vec::new();
    for (i, (wid, facts)) in worlds.iter().enumerate() {
        let c = i as u32 + 1;
        world_col.insert(wid.clone(), c);
        if wid.starts_with("hub-") {
            hub.push(c);
        }
        for f in facts {
            of_fact.entry(f.as_str()).or_default().push(c);
        }
    }
    let g = shared.mem.read().expect("projection lock");
    let st = &g.store;
    let of: Vec<Vec<u32>> = (0..st.num_records())
        .map(|r| {
            let fid = st.fact_id(r as xerj_facts::xerj_store::RecId);
            if fid.starts_with("wd5m/") || fid.starts_with("0wd5m/") {
                vec![0]
            } else {
                of_fact.get(fid).cloned().unwrap_or_default()
            }
        })
        .collect();
    let n = worlds.len() + 1;
    let not_hub: Vec<u32> = (0..n as u32).filter(|c| !hub.contains(c)).collect();
    SharedCollections {
        map: std::sync::Arc::new(xerj_facts::xerj_index::scope::RecCollections::from_lists(
            of,
        )),
        world_col,
        hub,
        not_hub,
        n,
    }
}

/// Round 3: the scope of one shared-arm question for a scope flag
fn question_scope(
    c: &SharedCollections,
    kind: &str,
    q: &Value,
) -> xerj_facts::xerj_index::scope::QuestionScope {
    use xerj_facts::xerj_index::scope::{QuestionScope, ScopeMode};
    let own = *c
        .world_col
        .get(q["world_id"].as_str().unwrap_or(""))
        .expect("the question's world is in world_facts.jsonl");
    let mode = match kind {
        "scope_own" => ScopeMode::Filter(vec![own]),
        "scope_ownbg" => ScopeMode::Filter(vec![0, own]),
        "nohub" => ScopeMode::Filter(c.not_hub.clone()),
        "prefer_own" => ScopeMode::Prefer(vec![own]),
        _ => ScopeMode::Coherent,
    };
    QuestionScope {
        map: c.map.clone(),
        mode,
    }
}
