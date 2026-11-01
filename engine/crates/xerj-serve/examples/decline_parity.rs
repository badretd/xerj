//! Research tool (e2e1, never served): the Rust decline policy v2 on recorded bench rows, for comparison with the Python oracle
//! (research/xerj_r_mvp/runs/e2e1/s3/oracle_policy.py, diff_parity.py).
//!
//!   decline_parity --worlds model_dev.jsonl --rows hybrid_v2_bench.rows.jsonl --out engine.jsonl [--threads 4]
//!
//! For every row: the world goes through the engine's store and index (`build_world`), the stack-v5 manifest is built as the
//! engine builds it, the row's recorded program is executed by the exact executor (`Episode`), and `decline_v2::decide` judges it.
//! No model runs: this checks the policy, repair and executor on the recorded programs. Gold fields are not read.

use serde_json::{json, Value};
use std::collections::HashMap;
use std::io::{BufRead, Write};
use xerj_exec::{Episode, Protocol};
use xerj_index::QBuf;
use xerj_serve::decline_v2::replay::{row_of, EpisodeReplay};
use xerj_serve::decline_v2::{decide, Read, Replay, Row, Step, Swapped, TraceStep};
use xerj_serve::engine::build_world;

fn arg(a: &[String], k: &str) -> Option<String> {
    a.iter()
        .position(|x| x == k)
        .and_then(|i| a.get(i + 1))
        .cloned()
}

fn lines(p: &str) -> Vec<String> {
    std::io::BufReader::new(std::fs::File::open(p).expect(p))
        .lines()
        .map(|l| l.expect("line"))
        .filter(|l| !l.trim().is_empty())
        .collect()
}

fn row_from(v: &Value) -> Row {
    let s = |k: &str| v[k].as_str().map(str::to_string);
    Row {
        question: s("question").expect("question"),
        status: s("status").expect("status"),
        answer: s("answer"),
        program: v["program"]
            .as_array()
            .expect("program")
            .iter()
            .map(|p| Step {
                op: p["op"].as_str().expect("op").to_string(),
                pointer: p["pointer"].as_i64(),
            })
            .collect(),
        trace: v["execution_trace"]
            .as_array()
            .expect("trace")
            .iter()
            .map(|t| TraceStep {
                op: t["op"].as_str().expect("op").to_string(),
                pointer: t["pointer"].as_i64(),
                result: t["result"].as_str().map(str::to_string),
                hop_candidate_ids: t["hop_candidate_ids"].as_array().map(|h| {
                    h.iter()
                        .map(|x| x.as_str().expect("fact id").to_string())
                        .collect()
                }),
            })
            .collect(),
        evidence_ids: v["evidence_ids"]
            .as_array()
            .expect("evidence")
            .iter()
            .map(|x| x.as_str().expect("fact id").to_string())
            .collect(),
    }
}

/// the live replay, with replay.replay_matches instead of "the row is the episode"
struct Recorded<'a, 'm> {
    inner: EpisodeReplay<'a, 'm>,
    same: bool,
}

impl Replay for Recorded<'_, '_> {
    fn matches(&mut self, _r: &Row) -> bool {
        self.same
    }
    fn operand_reads(&mut self, r: &Row) -> Vec<Read> {
        self.inner.operand_reads(r)
    }
    fn execute_swapped(&mut self, r: &Row) -> Option<Swapped> {
        self.inner.execute_swapped(r)
    }
}

/// compares what replay.replay_matches compares, with `direction` against the executor's inverse flag
fn same_as_recorded(rec: &Value, r: &Row, ans: &xerj_exec::Answer) -> bool {
    let live = |t: usize| ans.execution_trace.get(t);
    let rows_equal = rec["status"].as_str() == Some(ans.status)
        && rec["answer"].as_str() == ans.answer.as_deref()
        && r.evidence_ids.len() == ans.evidence_ids.len();
    let trace = rec["execution_trace"].as_array().map_or(0, |a| a.len());
    rows_equal
        && trace == ans.execution_trace.len()
        && (0..trace).all(|t| {
            let e = &rec["execution_trace"][t];
            live(t).is_some_and(|l| {
                e["step"].as_u64() == Some(l.step as u64) && e["direction"].is_string() == l.inverse
            })
        })
}

fn run_one(world: &Value, rec: &Value) -> Value {
    let item_id = rec["item_id"].as_str().expect("item_id");
    let want = row_from(rec);
    let built = build_world(world, None).expect("world");
    let item = world["items"]
        .as_array()
        .and_then(|a| a.iter().find(|i| i["item_id"] == rec["item_id"]))
        .expect("item");
    let rev = if item["phase"].as_str() == Some("before") {
        built.base
    } else {
        built.final_
    };
    let snap = built.mem.snapshot_at(rev);
    let mut buf = QBuf::default();
    let k = 16usize;
    let question = item["question"].as_str().expect("question");
    let m = built
        .mem
        .candidate_manifest_v5(&snap, item_id, question, k, &mut buf);
    let (cands, inverse) = (m.cands.clone(), m.inverse.clone());
    let mut ep = Episode::from_manifest_v5(&built.mem, snap, m, item_id, k);
    for s in &want.program {
        if ep.done {
            break;
        }
        let op = Protocol::V5.op_id(&s.op).expect("op");
        ep.step(op, s.pointer).expect("step");
    }
    let ans = ep.answer_or_unfinished();
    let live = row_of(&want.question, &ans, &built.mem);
    let same = live == want && same_as_recorded(rec, &want, &ans);
    let mut r = Recorded {
        inner: EpisodeReplay {
            ep: &ep,
            mem: &built.mem,
            snap,
            cands: &cands,
            inverse: &inverse,
            item_id,
            k,
        },
        same,
    };
    let d = decide(&want, &mut r);
    let steps = |p: &[Step]| {
        p.iter()
            .map(|s| json!([s.op, s.pointer]))
            .collect::<Vec<_>>()
    };
    let (fs, fa) = d.served(&want.status);
    json!({"item_id": item_id, "replay_ok": same, "accepted": d.accepted, "answer": d.answer, "code": d.policy_code,
           "repaired": d.repair.is_some(), "detail": d.detail, "reason_code": d.reason_code,
           "p3_accepted": d.stages.p3_accepted, "d2_accepted": d.stages.d2_accepted, "d2_code": d.stages.d2_code,
           "repaired_program": d.repair.as_ref().map(|x| steps(&x.program)),
           "repaired_evidence": d.repair.as_ref().map(|x| x.evidence_ids.clone()),
           "final_status": fs, "final_answer": fa})
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let worlds_path = arg(&a, "--worlds").expect("--worlds");
    let rows_path = arg(&a, "--rows").expect("--rows");
    let out = arg(&a, "--out").expect("--out");
    let threads: usize = arg(&a, "--threads").map_or(4, |s| s.parse().expect("--threads"));
    let t0 = std::time::Instant::now();
    let worlds: Vec<Value> = lines(&worlds_path)
        .iter()
        .map(|l| serde_json::from_str(l).expect("world"))
        .collect();
    let mut by_item: HashMap<String, usize> = HashMap::new();
    for (i, w) in worlds.iter().enumerate() {
        for it in w["items"].as_array().expect("items") {
            by_item.insert(it["item_id"].as_str().expect("id").to_string(), i);
        }
    }
    let recs: Vec<Value> = lines(&rows_path)
        .iter()
        .map(|l| serde_json::from_str(l).expect("row"))
        .collect();
    let chunk = recs.len().div_ceil(threads.max(1));
    let mut res: Vec<Vec<Value>> = Vec::new();
    std::thread::scope(|s| {
        let hs: Vec<_> = recs
            .chunks(chunk)
            .map(|c| {
                let (worlds, by_item) = (&worlds, &by_item);
                s.spawn(move || {
                    c.iter()
                        .map(|r| run_one(&worlds[by_item[r["item_id"].as_str().expect("id")]], r))
                        .collect::<Vec<Value>>()
                })
            })
            .collect();
        res = hs.into_iter().map(|h| h.join().expect("worker")).collect();
    });
    let mut f = std::io::BufWriter::new(std::fs::File::create(&out).expect("out"));
    let mut n = 0;
    for v in res.iter().flatten() {
        writeln!(f, "{v}").expect("write");
        n += 1;
    }
    eprintln!("{n} rows, {:.1} s", t0.elapsed().as_secs_f64());
}
