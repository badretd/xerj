//! Served-response evaluation of the verify-rule prototype (`verify_bound`) on the engine spike's runs
//! (research/xerj_r_mvp/runs/spike_facts_e2e/<run>/results.jsonl, written by tools/spike_facts_e2e.py).
//!
//! For every served row the world is ingested into a fresh `Memory` (the same reference store the engine's projection
//! rebuilds), the item's stack-v5 manifest is recomputed at the row's revision, and the served program is replayed through
//! `Episode::new_v5`, which gives back the served answer record with its evidence. The test first checks that the replay
//! reproduces the served status / answer and the served `verify_v5` verdict; then it scores the final served response under
//! each rule: correct answers withheld, wrong answers accepted, coverage (final ANSWER) and raw vs verified accuracy.
//! The model is not run: the served programs are fixed, and the verify step depends only on the question, the answer record
//! and the store.
//!
//! XERJ_VERIFY_EVAL_RUNS = comma-separated run directories; XERJ_R_ROOT = research/xerj_r_mvp (the runs' relative `data`
//! paths); XERJ_VERIFY_EVAL_OUT = a file for the table. Without XERJ_VERIFY_EVAL_RUNS the test says so and returns.

use serde_json::Value;
use std::collections::{BTreeMap, HashMap};
use std::io::BufRead;
use std::path::{Path, PathBuf};
use xerj_exec::episode::{Episode, Protocol};
use xerj_exec::Verdict;
use xerj_index::{Memory, MemoryConfig, QBuf};
use xerj_serve::verify_bound::{verify_bound, Opts};
use xerj_serve::verify_v5::verify_v5;
use xerj_store::world::World;

const K: usize = 16;

/// The rules compared: the served one and the prototype's variants.
const RULES: [&str; 5] = [
    "served (verify_v5)",
    "bindings for (b)",
    "bindings for (b) + path",
    "bindings for (a)(b) + path",
    "no verify",
];

#[derive(Default, Clone)]
struct Score {
    n: u64,
    raw_answered: u64,
    raw_correct: u64,
    vetoed: u64,
    withheld_correct: u64,
    accepted_wrong: u64,
    final_answered: u64,
    final_correct: u64,
}

fn lines(path: &Path) -> Vec<String> {
    let rd = std::io::BufReader::with_capacity(
        1 << 20,
        std::fs::File::open(path).unwrap_or_else(|e| panic!("{}: {e}", path.display())),
    );
    rd.lines()
        .map(|l| l.unwrap())
        .filter(|l| !l.trim().is_empty())
        .collect()
}

fn memory() -> Memory {
    // the engine projection's configuration (xerj-facts `Namespace::new`): BM25 off; verify does not read it
    Memory::new(MemoryConfig {
        bm25_records: false,
        bm25_documents: false,
        ..MemoryConfig::default()
    })
}

struct RunEval {
    name: String,
    rows: u64,
    replay_ok: u64,
    served_verdict_ok: u64,
    mismatches: Vec<String>,
    scores: Vec<Score>,
    reasons: Vec<BTreeMap<&'static str, u64>>,
}

fn eval_run(root: &Path, run: &Path) -> RunEval {
    let summary: Value =
        serde_json::from_str(&std::fs::read_to_string(run.join("summary.json")).unwrap()).unwrap();
    let data = root.join(summary["data"].as_str().unwrap());
    let split = summary["split"].as_str().unwrap();
    let n_worlds = summary["worlds"].as_u64().unwrap() as usize;
    let world_lines: Vec<String> = lines(&data.join(format!("model_{split}.jsonl")))
        .into_iter()
        .take(n_worlds)
        .collect();
    // items: the run's item file (keyed by world_id) or the worlds' own items
    let mut item_sets: HashMap<String, Vec<(String, String)>> = HashMap::new();
    if let Some(f) = summary["items_file"].as_str() {
        for l in lines(&root.join(f)) {
            let v: Value = serde_json::from_str(&l).unwrap();
            let wid = v["world_id"].as_str().unwrap().to_string();
            for it in v["items"].as_array().unwrap() {
                item_sets.entry(wid.clone()).or_default().push((
                    it["item_id"].as_str().unwrap().to_string(),
                    it["question"].as_str().unwrap().to_string(),
                ));
            }
        }
    }
    let rows: Vec<Value> = lines(&run.join("results.jsonl"))
        .iter()
        .map(|l| serde_json::from_str(l).unwrap())
        .filter(|r: &Value| r.get("status").is_some())
        .collect();
    let mut by_item: HashMap<String, &Value> = HashMap::new();
    for r in &rows {
        by_item.insert(r["item_id"].as_str().unwrap().to_string(), r);
    }
    let mut ev = RunEval {
        name: run.file_name().unwrap().to_string_lossy().into_owned(),
        rows: 0,
        replay_ok: 0,
        served_verdict_ok: 0,
        mismatches: Vec::new(),
        scores: vec![Score::default(); RULES.len()],
        reasons: vec![BTreeMap::new(); RULES.len()],
    };
    let mut buf = QBuf::default();
    for wl in &world_lines {
        let world = World::parse(wl).expect("world line");
        let wid = world.world_id.as_deref().unwrap_or("").to_string();
        let items: Vec<(String, String)> = match item_sets.get(&wid) {
            Some(v) => v.clone(),
            None => world
                .items
                .iter()
                .map(|i| (i.item_id.to_string(), i.question.to_string()))
                .collect(),
        };
        let mut mem = memory();
        mem.ingest_world(&world).expect("ingest");
        for (item_id, question) in &items {
            let Some(row) = by_item.get(item_id) else {
                continue;
            };
            ev.rows += 1;
            let rev = row["revision"].as_u64().unwrap() as u32;
            let snap = mem.snapshot_at(rev);
            let m = mem.candidate_manifest_v5(&snap, item_id, question, K, &mut buf);
            let mut ep =
                Episode::new_v5(&mem, snap, m.cands.clone(), m.inverse.clone(), item_id, K);
            for step in row["program"].as_array().unwrap() {
                if ep.done {
                    break;
                }
                let op = step[0].as_str().unwrap();
                let ptr = step[1].as_i64();
                let id = Protocol::V5.op_id(op).unwrap_or_else(|| panic!("op {op}"));
                let _ = ep.step(id, ptr);
            }
            let ans = ep.answer_or_unfinished();
            let served_status = row["status"].as_str().unwrap();
            let served_answer = row["answer"].as_str().map(str::to_string);
            if ans.status == served_status && ans.answer == served_answer {
                ev.replay_ok += 1;
            } else if ev.mismatches.len() < 10 {
                ev.mismatches.push(format!(
                    "replay {item_id}: {} {:?} vs served {served_status} {served_answer:?}",
                    ans.status, ans.answer
                ));
            }
            let served_veto = row["verify"]["veto"].as_bool().unwrap_or(false);
            let verdicts: [Verdict; 5] = [
                verify_v5(&mem, question, &m.cands, &ans, &mut buf),
                verify_bound(
                    &mem,
                    &snap,
                    question,
                    &m.cands,
                    &ans,
                    Opts {
                        bind_b: true,
                        bind_a: false,
                        path: false,
                        bind_all: false,
                    },
                    &mut buf,
                ),
                verify_bound(
                    &mem,
                    &snap,
                    question,
                    &m.cands,
                    &ans,
                    Opts::default(),
                    &mut buf,
                ),
                verify_bound(
                    &mem,
                    &snap,
                    question,
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
                Verdict {
                    vetoed: false,
                    reason: "",
                },
            ];
            if verdicts[0].vetoed == served_veto {
                ev.served_verdict_ok += 1;
            } else if ev.mismatches.len() < 10 {
                ev.mismatches.push(format!(
                    "served verdict {item_id}: recomputed {} ({}) vs served {served_veto}",
                    verdicts[0].vetoed, verdicts[0].reason
                ));
            }
            let correct_raw = row["correct_raw"].as_bool().unwrap();
            let gold_unknown = row["gold_status"].as_str() == Some("UNKNOWN");
            let answered = served_status == "ANSWER";
            for (i, v) in verdicts.iter().enumerate() {
                let s = &mut ev.scores[i];
                s.n += 1;
                s.raw_answered += u64::from(answered);
                s.raw_correct += u64::from(correct_raw);
                let veto = answered && v.vetoed;
                s.vetoed += u64::from(veto);
                s.withheld_correct += u64::from(veto && correct_raw);
                s.accepted_wrong += u64::from(answered && !veto && !correct_raw);
                s.final_answered += u64::from(answered && !veto);
                // a vetoed answer is served as UNKNOWN: correct only when the gold is UNKNOWN
                s.final_correct += u64::from(if veto { gold_unknown } else { correct_raw });
                if veto {
                    *ev.reasons[i].entry(v.reason).or_default() += 1;
                }
            }
        }
    }
    ev
}

#[test]
fn verify_bound_on_the_spike_runs() {
    let Ok(runs) = std::env::var("XERJ_VERIFY_EVAL_RUNS") else {
        eprintln!("XERJ_VERIFY_EVAL_RUNS not set (research/xerj_r_mvp/runs/spike_facts_e2e/<run>, comma-separated); not run");
        return;
    };
    let root =
        PathBuf::from(std::env::var("XERJ_R_ROOT").expect("XERJ_R_ROOT = research/xerj_r_mvp"));
    let mut md = String::new();
    md.push_str("| run | rule | n | raw answered | raw correct | vetoed | correct withheld | wrong accepted | coverage | raw acc | verified acc |\n");
    md.push_str("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n");
    let mut fidelity_failures = 0u64;
    let mut reason_lines = Vec::new();
    for run in runs.split(',').filter(|s| !s.is_empty()) {
        let t0 = std::time::Instant::now();
        let ev = eval_run(&root, Path::new(run));
        for (i, s) in ev.scores.iter().enumerate() {
            let pct = |x: u64| 100.0 * x as f64 / s.n.max(1) as f64;
            md.push_str(&format!(
                "| {} | {} | {} | {} | {} | {} | {} | {} | {:.1}% | {:.1}% | {:.1}% |\n",
                ev.name,
                RULES[i],
                s.n,
                s.raw_answered,
                s.raw_correct,
                s.vetoed,
                s.withheld_correct,
                s.accepted_wrong,
                pct(s.final_answered),
                pct(s.raw_correct),
                pct(s.final_correct)
            ));
            if !ev.reasons[i].is_empty() {
                reason_lines.push(format!("{} / {}: {:?}", ev.name, RULES[i], ev.reasons[i]));
            }
        }
        println!(
            "{}: {} rows, replay = served on {}, served verdict reproduced on {} ({:.1} s)",
            ev.name,
            ev.rows,
            ev.replay_ok,
            ev.served_verdict_ok,
            t0.elapsed().as_secs_f64()
        );
        for m in &ev.mismatches {
            println!("  MISMATCH {m}");
        }
        fidelity_failures += (ev.rows - ev.replay_ok) + (ev.rows - ev.served_verdict_ok);
    }
    println!("{md}");
    for l in &reason_lines {
        println!("veto reasons {l}");
    }
    if let Ok(p) = std::env::var("XERJ_VERIFY_EVAL_OUT") {
        let _ = std::fs::write(p, format!("{md}\n{}\n", reason_lines.join("\n")));
    }
    assert_eq!(
        fidelity_failures, 0,
        "the replay must reproduce the served answers and verdicts"
    );
}

/// The same rules on many wrong answers: the stack-v5 parity fixtures' random / guided / mutated programs
/// (research/xerj_r_mvp/runs/xerj_mem_parity_v5, XERJ_PARITY_V5_DIR, sets in XERJ_VERIFY_STRESS_SETS), replayed on their own
/// item's manifest and snapshot. A program's answer counts as correct when it equals the item's gold-trace answer (status
/// and normalized answer) and that gold answer is itself correct (`score.exact_correct`); items without a correct gold answer
/// are skipped. The served spike runs hold three wrong answers in all; this measures what each rule rejects.
#[test]
fn verify_bound_on_fixture_programs() {
    use xerj_exec::scorer::normalize_answer;
    let (Ok(dir), Ok(sets)) = (
        std::env::var("XERJ_PARITY_V5_DIR"),
        std::env::var("XERJ_VERIFY_STRESS_SETS"),
    ) else {
        eprintln!("XERJ_PARITY_V5_DIR / XERJ_VERIFY_STRESS_SETS not set; not run");
        return;
    };
    let mut md = String::new();
    md.push_str("| set | rule | program ANSWERs | correct | wrong | vetoed | correct withheld | wrong accepted | wrong rejected |\n");
    md.push_str("|---|---|---:|---:|---:|---:|---:|---:|---:|\n");
    let mut buf = QBuf::default();
    for set in sets.split(',').filter(|s| !s.is_empty()) {
        let mut scores = vec![Score::default(); RULES.len()];
        let mut wrong_n = 0u64;
        let mut correct_n = 0u64;
        let mut header: Option<Value> = None;
        for line in lines(&Path::new(&dir).join(format!("{set}.parity.jsonl"))) {
            let v: Value = serde_json::from_str(&line).unwrap();
            if v.get("header").is_some() {
                header = Some(v);
                continue;
            }
            let h = header.as_ref().expect("header first");
            let k = h["episode_K"].as_u64().unwrap() as usize;
            let max_inverse = h["max_inverse"].as_i64().unwrap();
            let world_text = v["world"].to_string();
            let world = World::parse(&world_text).expect("world");
            let questions: HashMap<String, String> = world
                .items
                .iter()
                .map(|i| (i.item_id.to_string(), i.question.to_string()))
                .collect();
            let mut mem = memory();
            let revs = mem.ingest_world(&world).expect("ingest");
            for it in v["items"].as_array().unwrap() {
                let ex = &it["exec"];
                if !ex["score"]["exact_correct"].as_bool().unwrap_or(false) {
                    continue;
                }
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
                for p in it["programs"].as_array().into_iter().flatten() {
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
                    let correct = gold_status == "ANSWER"
                        && normalize_answer(ans.answer.as_deref()) == gold_answer;
                    if correct {
                        correct_n += 1;
                    } else {
                        wrong_n += 1;
                    }
                    let verdicts: [Verdict; 5] = [
                        verify_v5(&mem, q, &m.cands, &ans, &mut buf),
                        verify_bound(
                            &mem,
                            &snap,
                            q,
                            &m.cands,
                            &ans,
                            Opts {
                                bind_b: true,
                                bind_a: false,
                                path: false,
                                bind_all: false,
                            },
                            &mut buf,
                        ),
                        verify_bound(&mem, &snap, q, &m.cands, &ans, Opts::default(), &mut buf),
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
                        Verdict {
                            vetoed: false,
                            reason: "",
                        },
                    ];
                    for (i, vd) in verdicts.iter().enumerate() {
                        let s = &mut scores[i];
                        s.n += 1;
                        s.vetoed += u64::from(vd.vetoed);
                        s.withheld_correct += u64::from(vd.vetoed && correct);
                        s.accepted_wrong += u64::from(!vd.vetoed && !correct);
                    }
                }
            }
        }
        for (i, s) in scores.iter().enumerate() {
            md.push_str(&format!(
                "| {set} | {} | {} | {correct_n} | {wrong_n} | {} | {} | {} | {} |\n",
                RULES[i],
                s.n,
                s.vetoed,
                s.withheld_correct,
                s.accepted_wrong,
                wrong_n - s.accepted_wrong
            ));
        }
    }
    println!("{md}");
    if let Ok(p) = std::env::var("XERJ_VERIFY_STRESS_OUT") {
        let _ = std::fs::write(p, &md);
    }
}
