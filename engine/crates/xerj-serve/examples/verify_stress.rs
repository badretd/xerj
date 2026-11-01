//! verify_stress: the verify-rule options on the stack-v5 parity fixtures' random / guided / mutated programs (research
//! tool; round 3, 54_verifier3, 2026-10-29). The served model makes few wrong answers on the engine's development sets, so
//! these programs show what each rule rejects among many wrong ones (round 1: `verify_bound_on_fixture_programs`, on test
//! fixtures; here one line per program ANSWER, with its provenance, for the development fixtures).
//!
//! ```text
//! verify_stress --parity FILE.parity.jsonl --set NAME --out FILE
//! ```
//!
//! Every item whose gold-trace answer is itself correct (`exec.score.exact_correct`) is taken; each of its programs is
//! replayed on the item's own stack-v5 manifest (the fixture's `episode_K` and `max_inverse`) and snapshot. A program
//! ANSWER is `correct` when its status and normalized answer equal the gold answer's, and `supported` when its evidence ids
//! also equal the gold answer's (the provenance the scorer's supported success asks for). Rules: served (`verify_v5`),
//! bindings, bindings_path, bindings_all (the opt-in `VerifyRule` values) and the prototype's bindings_ab_path.

use serde_json::{json, Value};
use std::collections::HashMap;
use std::io::{BufRead, BufWriter, Write};
use std::path::Path;
use xerj_exec::episode::Episode;
use xerj_exec::scorer::normalize_answer;
use xerj_index::{Memory, MemoryConfig, QBuf};
use xerj_serve::verify_bound::{verify_bound, verify_v5_with, Opts, VerifyRule};
use xerj_store::world::World;

/// the subject names of syms
fn names(mem: &Memory, syms: &[u32]) -> Vec<String> {
    syms.iter()
        .map(|&s| mem.store.pools.subjects.get(s).to_string())
        .collect()
}

fn arg(args: &[String], k: &str) -> Option<String> {
    args.iter()
        .position(|a| a == k)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let parity = arg(&args, "--parity").expect("--parity");
    let set = arg(&args, "--set").expect("--set");
    let out_path = arg(&args, "--out").expect("--out");
    let rd = std::io::BufReader::with_capacity(
        1 << 22,
        std::fs::File::open(Path::new(&parity)).expect("open --parity"),
    );
    let mut out = BufWriter::new(std::fs::File::create(&out_path).expect("create --out"));
    let mut header: Option<Value> = None;
    let mut buf = QBuf::default();
    let (mut n_items, mut n_progs) = (0u64, 0u64);
    for line in rd.lines() {
        let line = line.expect("read");
        if line.trim().is_empty() {
            continue;
        }
        let v: Value = serde_json::from_str(&line).expect("json");
        if v.get("header").is_some() {
            header = Some(v);
            continue;
        }
        let h = header.as_ref().expect("header first");
        let k = h["episode_K"].as_u64().unwrap() as usize;
        let max_inverse = h["max_inverse"].as_i64().unwrap();
        let world_text = v["world"].to_string();
        let world = World::parse(&world_text).expect("world");
        let wname = world
            .group_id
            .as_deref()
            .or(world.world_id.as_deref())
            .unwrap_or("")
            .to_string();
        let questions: HashMap<String, String> = world
            .items
            .iter()
            .map(|i| (i.item_id.to_string(), i.question.to_string()))
            .collect();
        let mut mem = Memory::new(MemoryConfig {
            bm25_records: false,
            bm25_documents: false,
            ..MemoryConfig::default()
        });
        let revs = mem.ingest_world(&world).expect("ingest");
        for it in v["items"].as_array().unwrap() {
            let ex = &it["exec"];
            if !ex["score"]["exact_correct"].as_bool().unwrap_or(false) {
                continue;
            }
            n_items += 1;
            let item_id = it["item_id"].as_str().unwrap();
            let q = questions[item_id].as_str();
            let rev = if it["phase"].as_str() == Some("before") {
                revs.base
            } else {
                revs.final_
            };
            let snap = mem.snapshot_at(rev);
            let m = mem.candidate_manifest_v5_with(&snap, item_id, q, k, max_inverse, &mut buf);
            let gold_status = ex["answer"]["status"].as_str().unwrap_or("");
            let gold_answer = normalize_answer(ex["answer"]["answer"].as_str());
            let mut gold_ev: Vec<String> = ex["answer"]["evidence_ids"]
                .as_array()
                .map(|a| {
                    a.iter()
                        .filter_map(|x| x.as_str().map(str::to_string))
                        .collect()
                })
                .unwrap_or_default();
            gold_ev.sort();
            for (pi, p) in it["programs"].as_array().into_iter().flatten().enumerate() {
                let mut ep =
                    Episode::new_v5(&mem, snap, m.cands.clone(), m.inverse.clone(), item_id, k);
                ep.max_inverse = max_inverse;
                for a in p["actions"].as_array().unwrap() {
                    if ep.done {
                        break;
                    }
                    let _ = ep.step(a[0].as_i64().unwrap(), a[1].as_i64());
                }
                let ans = ep.answer_or_unfinished();
                if ans.status != "ANSWER" {
                    continue;
                }
                n_progs += 1;
                let correct = gold_status == "ANSWER"
                    && normalize_answer(ans.answer.as_deref()) == gold_answer;
                let mut ev: Vec<String> = ans
                    .evidence_ids
                    .iter()
                    .map(|&r| mem.store.fact_id(r).to_string())
                    .collect();
                ev.sort();
                let supported = correct && ev == gold_ev;
                let vs = [
                    verify_v5_with(VerifyRule::Served, &mem, &snap, q, &m.cands, &ans, &mut buf),
                    verify_v5_with(
                        VerifyRule::Bindings,
                        &mem,
                        &snap,
                        q,
                        &m.cands,
                        &ans,
                        &mut buf,
                    ),
                    verify_v5_with(
                        VerifyRule::BindingsPath,
                        &mem,
                        &snap,
                        q,
                        &m.cands,
                        &ans,
                        &mut buf,
                    ),
                    verify_v5_with(
                        VerifyRule::BindingsAll,
                        &mem,
                        &snap,
                        q,
                        &m.cands,
                        &ans,
                        &mut buf,
                    ),
                    verify_bound(
                        &mem,
                        &snap,
                        q,
                        &m.cands,
                        &ans,
                        Opts {
                            bind_b: true,
                            bind_a: true,
                            path: true,
                            bind_all: false,
                        },
                        &mut buf,
                    ),
                ];
                let mut rules = serde_json::Map::new();
                for (name, vd) in [
                    "served",
                    "bindings",
                    "bindings_path",
                    "bindings_all",
                    "bindings_ab_path",
                ]
                .iter()
                .zip(vs.iter())
                {
                    rules.insert(
                        name.to_string(),
                        json!({"veto": vd.vetoed, "reason": vd.reason}),
                    );
                }
                let line = json!({
                    "set": set, "world": wname, "item_id": item_id, "phase": it["phase"], "question": q,
                    "program_index": pi, "kind": p["kind"], "answer": ans.answer, "gold_answer": ex["answer"]["answer"],
                    "correct": correct, "supported": supported, "evidence_ids": ev, "gold_evidence_ids": gold_ev,
                    "program": ans.program.iter().map(|a| json!([a.op, a.pointer, a.inverse.then_some("in")])).collect::<Vec<_>>(),
                    "rules": rules,
                    "n_bound": mem.link(&snap, q, &mut buf).len(),
                    "n_substr2": mem.linker.substring_subjects(&mem.store, q, &mut buf, 2).len(),
                    "bound": names(&mem, &mem.link(&snap, q, &mut buf)),
                    "bound_all": names(&mem, &mem.linker.link_any_state(&mem.store, q, &mut buf)),
                    "substr": names(&mem, &mem.linker.substring_subjects(&mem.store, q, &mut buf, 8)),
                });
                writeln!(out, "{line}").unwrap();
            }
        }
    }
    out.flush().unwrap();
    eprintln!(
        "{}",
        json!({"set": set, "items": n_items, "program_answers": n_progs})
    );
}
