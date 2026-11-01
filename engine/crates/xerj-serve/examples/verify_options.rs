//! verify_options: the verify-rule options judged on development data (research tool; round 3, 54_verifier3, 2026-10-29).
//! Nothing here is served: the engine's choice is `AnswerOpts::verify` (default `VerifyRule::Served`).
//!
//! ```text
//! verify_options run    --models NAME=DIR --precision f32|int8 [--int8-mode guard|smooth|raw]
//!                       --worlds FILE [--items FILE] --set NAME [--shard I/N] [--max-worlds N] --out FILE
//! verify_options replay --rows FILE --worlds FILE [--items FILE] --set NAME --model NAME [--shard I/N] --out FILE
//! ```
//!
//! * `run` answers every item with the model exactly as `_ask` serves a question by default (`answer_stoppable_opts`,
//!   exact linker, `AnswerOpts::default()`, candidates listed), then replays the served program on the item's stack-v5
//!   manifest through `Episode::new_v5`, which gives back the answer record with its execution trace and evidence.
//! * `replay` does the same with the programs of saved per-question rows (tools/eval_heldout_rows.py rows, rollout
//!   predictions: `program` as `[{op, pointer, direction?}]` or `[[op, pointer, dir?]]`) instead of running a model.
//!
//! The replayed record is judged by every rule: `served` (`verify_v5`, the default), the opt-in values of `VerifyRule`
//! (`bindings`, `bindings_path`, `bindings_all`) through `verify_v5_with`, the engine's own dispatch, and the prototype's
//! `bindings_ab_path` (bindings for (a) too; compared, not offered). `bound` / `bound_all` / `substr` list the question's
//! subjects as the snapshot's bindings, the bindings over every written subject and the raw substring scan see them. One JSON line per item: identity, the served
//! response (run) or the row's answer (replay), the replay and whether it reproduces the served record, the verdicts.
//! `--worlds` = memory-world-v1 lines; items come from `--items` (lines `{group_id, items}`) or from the worlds; phase
//! "before" reads the base snapshot, anything else the final one. `--shard I/N` takes worlds I, I+N, I+2N, ... (0-based).

use serde_json::{json, Value};
use std::collections::HashMap;
use std::io::{BufRead, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;
use xerj_exec::episode::{Episode, Protocol};
use xerj_exec::{Answer, Verdict};
use xerj_index::{Memory, MemoryConfig, QBuf};
use xerj_infer::Precision;
use xerj_serve::engine::{answer_stoppable_opts, AnswerOpts, LinkerMode, ModelSlot};
use xerj_serve::verify_bound::{verify_bound, verify_v5_with, Opts, VerifyRule};
use xerj_store::world::World;
use xerj_store::{RecId, Snapshot};

const K_REPLAY: usize = 16;

fn arg(args: &[String], k: &str) -> Option<String> {
    args.iter()
        .position(|a| a == k)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

fn lines(path: &Path) -> Vec<String> {
    let rd = std::io::BufReader::with_capacity(
        1 << 20,
        std::fs::File::open(path).unwrap_or_else(|e| panic!("{}: {e}", path.display())),
    );
    rd.lines()
        .map(|l| l.expect("read line"))
        .filter(|l| !l.trim().is_empty())
        .collect()
}

fn memory() -> Memory {
    // the engine projection's configuration (xerj-facts `Namespace::new`, xerj-serve `build_world`): BM25 off
    Memory::new(MemoryConfig {
        bm25_records: false,
        bm25_documents: false,
        ..MemoryConfig::default()
    })
}

/// (op, pointer) of one saved program step: `{op, pointer}` objects or `[op, pointer, dir]` arrays
fn step_of(s: &Value) -> (String, Option<i64>) {
    match s {
        Value::Array(a) => (
            a.first().and_then(Value::as_str).unwrap_or("").to_string(),
            a.get(1).and_then(Value::as_i64),
        ),
        Value::Object(o) => (
            o.get("op")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string(),
            o.get("pointer").and_then(Value::as_i64),
        ),
        Value::String(op) => (op.clone(), None),
        _ => (String::new(), None),
    }
}

/// a row's served status / answer under the names the research tools use
fn row_status(r: &Value) -> (Option<String>, Option<String>) {
    let st = r
        .get("status")
        .or_else(|| r.get("predicted_status"))
        .and_then(Value::as_str)
        .map(str::to_string);
    let an = r
        .get("answer")
        .or_else(|| r.get("predicted_answer"))
        .and_then(Value::as_str)
        .map(str::to_string);
    (st, an)
}

struct Replayed {
    ans: Answer,
    cands: Vec<RecId>,
    /// every saved step had a pointer field (a program of op names only cannot be replayed)
    complete: bool,
}

fn replay(
    mem: &Memory,
    snap: Snapshot<'_>,
    item_id: &str,
    question: &str,
    k: usize,
    program: &[Value],
    buf: &mut QBuf,
) -> Replayed {
    let m = mem.candidate_manifest_v5(&snap, item_id, question, k, buf);
    let mut ep = Episode::new_v5(mem, snap, m.cands.clone(), m.inverse.clone(), item_id, k);
    let mut complete = true;
    for s in program {
        if ep.done {
            break;
        }
        if matches!(s, Value::String(_)) {
            complete = false;
        }
        let (op, ptr) = step_of(s);
        let Some(id) = Protocol::V5.op_id(&op) else {
            complete = false;
            break;
        };
        let _ = ep.step(id, ptr);
    }
    Replayed {
        ans: ep.answer_or_unfinished(),
        cands: m.cands,
        complete,
    }
}

const RULES: [&str; 5] = [
    "served",
    "bindings",
    "bindings_path",
    "bindings_all",
    "bindings_ab_path",
];

fn verdicts(
    mem: &Memory,
    snap: &Snapshot<'_>,
    question: &str,
    cands: &[RecId],
    ans: &Answer,
    buf: &mut QBuf,
) -> [Verdict; 5] {
    [
        verify_v5_with(VerifyRule::Served, mem, snap, question, cands, ans, buf),
        verify_v5_with(VerifyRule::Bindings, mem, snap, question, cands, ans, buf),
        verify_v5_with(
            VerifyRule::BindingsPath,
            mem,
            snap,
            question,
            cands,
            ans,
            buf,
        ),
        verify_v5_with(
            VerifyRule::BindingsAll,
            mem,
            snap,
            question,
            cands,
            ans,
            buf,
        ),
        verify_bound(
            mem,
            snap,
            question,
            cands,
            ans,
            Opts {
                bind_b: true,
                bind_a: true,
                path: true,
                bind_all: false,
            },
            buf,
        ),
    ]
}

/// the subject names of syms
fn names(mem: &Memory, syms: &[u32]) -> Vec<String> {
    syms.iter()
        .map(|&s| mem.store.pools.subjects.get(s).to_string())
        .collect()
}

struct Item {
    item_id: String,
    question: String,
    phase: Option<String>,
}

fn items_of(world: &World<'_>, item_sets: &HashMap<String, Vec<Item>>) -> Vec<Item> {
    let gid = world.group_id.as_deref().unwrap_or("").to_string();
    let wid = world.world_id.as_deref().unwrap_or("").to_string();
    if let Some(v) = item_sets.get(&gid).or_else(|| item_sets.get(&wid)) {
        return v
            .iter()
            .map(|i| Item {
                item_id: i.item_id.clone(),
                question: i.question.clone(),
                phase: i.phase.clone(),
            })
            .collect();
    }
    world
        .items
        .iter()
        .map(|i| Item {
            item_id: i.item_id.to_string(),
            question: i.question.to_string(),
            phase: i.phase.as_ref().map(|p| p.to_string()),
        })
        .collect()
}

fn load_item_sets(path: Option<String>) -> HashMap<String, Vec<Item>> {
    let mut out: HashMap<String, Vec<Item>> = HashMap::new();
    let Some(p) = path else { return out };
    for l in lines(Path::new(&p)) {
        let v: Value = serde_json::from_str(&l).expect("items line");
        let key = v
            .get("group_id")
            .or_else(|| v.get("world_id"))
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        for it in v["items"].as_array().into_iter().flatten() {
            out.entry(key.clone()).or_default().push(Item {
                item_id: it["item_id"].as_str().unwrap().to_string(),
                question: it["question"].as_str().unwrap().to_string(),
                phase: it.get("phase").and_then(Value::as_str).map(str::to_string),
            });
        }
    }
    out
}

fn fact_ids(mem: &Memory, ids: &[RecId]) -> Vec<String> {
    ids.iter()
        .map(|&r| mem.store.fact_id(r).to_string())
        .collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).cloned().unwrap_or_default();
    if mode != "run" && mode != "replay" {
        eprintln!("usage: verify_options run|replay ... (see the file's header)");
        std::process::exit(2);
    }
    let worlds_path = PathBuf::from(arg(&args, "--worlds").expect("--worlds"));
    let set = arg(&args, "--set").expect("--set");
    let out_path = arg(&args, "--out").expect("--out");
    let (shard, nshards) = match arg(&args, "--shard") {
        Some(s) => {
            let (a, b) = s.split_once('/').expect("--shard I/N");
            (a.parse::<usize>().unwrap(), b.parse::<usize>().unwrap())
        }
        None => (0, 1),
    };
    let max_worlds = arg(&args, "--max-worlds").map(|s| s.parse::<usize>().unwrap());
    let item_sets = load_item_sets(arg(&args, "--items"));

    // run: one model slot (precision as the product's XERJ_FACTS_PRECISION; one thread per question, the default)
    let slot = if mode == "run" {
        let spec = arg(&args, "--models").expect("--models NAME=DIR");
        let (name, dir) = spec.split_once('=').expect("--models NAME=DIR");
        let precision = match arg(&args, "--precision").as_deref() {
            Some("f32") => Precision::F32,
            Some("int8") => Precision::Int8,
            Some("w8") => Precision::W8,
            other => panic!("--precision f32|w8|int8 (got {other:?})"),
        };
        let int8_mode = arg(&args, "--int8-mode").unwrap_or_else(|| "guard".into());
        Some(ModelSlot::load(name, Path::new(dir), precision, &int8_mode).expect("model"))
    } else {
        None
    };
    let model_name = match &slot {
        Some(s) => s.name.clone(),
        None => arg(&args, "--model").expect("--model NAME (replay)"),
    };
    let k = slot.as_ref().map_or(K_REPLAY, |s| s.k);
    // replay: the saved rows by item id
    let rows: HashMap<String, Value> = match arg(&args, "--rows") {
        Some(p) if mode == "replay" => lines(Path::new(&p))
            .iter()
            .map(|l| {
                let v: Value = serde_json::from_str(l).expect("row");
                (v["item_id"].as_str().unwrap().to_string(), v)
            })
            .collect(),
        _ => HashMap::new(),
    };

    let mut out = BufWriter::new(std::fs::File::create(&out_path).expect("create --out"));
    let t0 = Instant::now();
    let mut buf = QBuf::default();
    let (mut n_items, mut n_worlds, mut bad_worlds) = (0u64, 0u64, 0u64);
    let world_lines = lines(&worlds_path);
    for (wi, wl) in world_lines.iter().enumerate() {
        if max_worlds.is_some_and(|m| wi >= m) {
            break;
        }
        if wi % nshards != shard {
            continue;
        }
        let world = match World::parse(wl) {
            Ok(w) => w,
            Err(e) => {
                eprintln!("[verify_options] world {wi}: parse error {e}");
                bad_worlds += 1;
                continue;
            }
        };
        let items = items_of(&world, &item_sets);
        if mode == "replay" && !items.iter().any(|i| rows.contains_key(&i.item_id)) {
            continue;
        }
        let mut mem = memory();
        let revs = match mem.ingest_world(&world) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("[verify_options] world {wi}: rejected by the store {e:?}");
                bad_worlds += 1;
                continue;
            }
        };
        n_worlds += 1;
        let wname = world
            .group_id
            .as_deref()
            .or(world.world_id.as_deref())
            .unwrap_or("")
            .to_string();
        for it in &items {
            let rev = if it.phase.as_deref() == Some("before") {
                revs.base
            } else {
                revs.final_
            };
            let snap = mem.snapshot_at(rev);
            let (served, program, status, answer): (
                Value,
                Vec<Value>,
                Option<String>,
                Option<String>,
            ) = if let Some(s) = &slot {
                let o = answer_stoppable_opts(
                    s,
                    &mem,
                    snap,
                    &it.item_id,
                    &it.question,
                    LinkerMode::Exact,
                    true,
                    None,
                    None,
                    &AnswerOpts::default(),
                )
                .expect("no stop flag");
                let program = o["program"].as_array().cloned().unwrap_or_default();
                let st = o["status"].as_str().map(str::to_string);
                let an = o["answer"].as_str().map(str::to_string);
                (o, program, st, an)
            } else {
                let Some(r) = rows.get(&it.item_id) else {
                    continue;
                };
                let program = r["program"].as_array().cloned().unwrap_or_default();
                let (st, an) = row_status(r);
                (r.clone(), program, st, an)
            };
            let rp = replay(&mem, snap, &it.item_id, &it.question, k, &program, &mut buf);
            let vs = verdicts(&mem, &snap, &it.question, &rp.cands, &rp.ans, &mut buf);
            let mut rules = serde_json::Map::new();
            for (name, v) in RULES.iter().zip(vs.iter()) {
                rules.insert(
                    name.to_string(),
                    json!({"veto": v.vetoed, "reason": v.reason}),
                );
            }
            let ev = fact_ids(&mem, &rp.ans.evidence_ids);
            let same_answer = status.as_deref() == Some(rp.ans.status) && answer == rp.ans.answer;
            let mut line = json!({
                "set": set, "model": model_name, "mode": mode, "world": wname, "item_id": it.item_id,
                "phase": it.phase, "question": it.question, "rev": rev,
                "status": status, "answer": answer,
                "program": program,
                "replay": {"status": rp.ans.status, "answer": rp.ans.answer, "evidence_ids": ev, "complete": rp.complete,
                           "same_answer": same_answer,
                           "program_ops": rp.ans.program.iter().map(|a| json!([a.op, a.pointer, a.inverse.then_some("in")])).collect::<Vec<_>>()},
                "rules": rules,
                "n_bound": mem.link(&snap, &it.question, &mut buf).len(),
                "n_substr2": mem.linker.substring_subjects(&mem.store, &it.question, &mut buf, 2).len(),
                "bound": names(&mem, &mem.link(&snap, &it.question, &mut buf)),
                "bound_all": names(&mem, &mem.linker.link_any_state(&mem.store, &it.question, &mut buf)),
                "substr": names(&mem, &mem.linker.substring_subjects(&mem.store, &it.question, &mut buf, 8)),
            });
            if slot.is_some() {
                let served_ev: Vec<String> = served["evidence_ids"]
                    .as_array()
                    .map(|a| {
                        a.iter()
                            .filter_map(|x| x.as_str().map(str::to_string))
                            .collect()
                    })
                    .unwrap_or_default();
                let served_cands: Vec<String> = served["candidate_ids"]
                    .as_array()
                    .map(|a| {
                        a.iter()
                            .filter_map(|x| x.as_str().map(str::to_string))
                            .collect()
                    })
                    .unwrap_or_default();
                let same_evidence = served_ev == ev;
                let same_manifest = served_cands == fact_ids(&mem, &rp.cands);
                line["served"] = json!({
                    "veto": served["verify"]["veto"], "reason": served["verify"]["reason"], "rule": served["verify"]["rule"],
                    "final_status": served["final_status"], "evidence_ids": served["evidence_ids"],
                    "same_evidence": same_evidence, "same_manifest": same_manifest,
                    "same_verdict": served["verify"]["veto"].as_bool() == Some(vs[0].vetoed),
                });
                line["decisions"] = served["decisions"].clone();
                line["fallbacks"] = served["fallbacks"].clone();
                line["model_ms"] =
                    json!(served["timings_us"]["model"].as_f64().map(|u| u / 1000.0));
                line["latency_ms"] = served["latency_ms"].clone();
                line["engine_error"] = served["engine_error"].clone();
            } else {
                for key in [
                    "gold_status",
                    "gold_answer",
                    "supported_success",
                    "ok",
                    "form",
                    "case_type",
                    "template",
                ] {
                    if let Some(v) = served.get(key) {
                        line[format!("row_{key}")] = v.clone();
                    }
                }
            }
            writeln!(out, "{line}").unwrap();
            n_items += 1;
            if n_items % 100 == 0 {
                eprintln!(
                    "[verify_options] {set} {model_name} shard {shard}/{nshards}: {n_items} items, {n_worlds} worlds, {:.0} s",
                    t0.elapsed().as_secs_f64()
                );
                out.flush().unwrap();
            }
        }
    }
    out.flush().unwrap();
    eprintln!(
        "{}",
        json!({"set": set, "model": model_name, "mode": mode, "shard": format!("{shard}/{nshards}"), "items": n_items,
               "worlds": n_worlds, "bad_worlds": bad_worlds, "wall_s": t0.elapsed().as_secs_f64()})
    );
}
