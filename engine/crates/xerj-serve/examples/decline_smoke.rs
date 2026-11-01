//! Research tool (e2e1, never served): a real model answers through `answer_stoppable_opts` with `verify: decline_v2`, to check the
//! response fields end to end at the engine (no HTTP). Each item runs twice: default and decline_v2.
//!
//!   decline_smoke --model DIR --worlds model_dev.jsonl --items ITEM_IDS.txt --out OUT.jsonl

use serde_json::{json, Value};
use std::io::{BufRead, Write};
use std::path::Path;
use xerj_infer::Precision;
use xerj_serve::engine::{
    answer_stoppable_opts, build_world, AnswerOpts, LinkerMode, ModelSlot, VerifyRule,
};

fn arg(a: &[String], k: &str) -> String {
    a.iter()
        .position(|x| x == k)
        .and_then(|i| a.get(i + 1))
        .cloned()
        .unwrap_or_else(|| panic!("{k}"))
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let slot = ModelSlot::load(
        "smoke",
        Path::new(&arg(&a, "--model")),
        Precision::F32,
        "raw",
    )
    .expect("model");
    let want: std::collections::BTreeSet<String> = std::fs::read_to_string(arg(&a, "--items"))
        .expect("items")
        .lines()
        .map(str::to_string)
        .collect();
    let mut out = std::io::BufWriter::new(std::fs::File::create(arg(&a, "--out")).expect("out"));
    for line in
        std::io::BufReader::new(std::fs::File::open(arg(&a, "--worlds")).expect("worlds")).lines()
    {
        let w: Value = serde_json::from_str(&line.expect("line")).expect("world");
        for it in w["items"].as_array().expect("items") {
            let id = it["item_id"].as_str().expect("id");
            if !want.contains(id) {
                continue;
            }
            let built = build_world(&w, None).expect("world");
            let rev = if it["phase"].as_str() == Some("before") {
                built.base
            } else {
                built.final_
            };
            let snap = built.mem.snapshot_at(rev);
            let q = it["question"].as_str().expect("question");
            let run = |o: &AnswerOpts| {
                answer_stoppable_opts(
                    &slot,
                    &built.mem,
                    snap,
                    id,
                    q,
                    LinkerMode::Exact,
                    false,
                    None,
                    None,
                    o,
                )
                .expect("answer")
            };
            let plain = run(&AnswerOpts::default());
            let d = run(&AnswerOpts {
                verify: VerifyRule::DeclineV2,
                ..AnswerOpts::default()
            });
            let same_raw = [
                "status",
                "answer",
                "evidence_ids",
                "program",
                "trace",
                "error",
            ]
            .iter()
            .all(|k| plain[*k] == d[*k]);
            let keys_added: Vec<&String> = d
                .as_object()
                .expect("object")
                .keys()
                .filter(|k| plain.get(k.as_str()).is_none())
                .collect();
            let plain_has_decline = ["reason_code", "repair", "decline", "final_evidence_ids"]
                .iter()
                .any(|k| plain.get(*k).is_some());
            let v = json!({"item_id": id, "same_raw": same_raw, "status": d["status"], "answer": d["answer"], "final_status": d["final_status"],
                "final_answer": d["final_answer"], "reason_code": d["reason_code"], "repair": d.get("repair"), "decline": d["decline"],
                "final_evidence_ids": d["final_evidence_ids"], "verify": d["verify"], "plain_final_status": plain["final_status"],
                "plain_has_decline_fields": plain_has_decline,
                "keys_added": keys_added, "program": d["program"]});
            writeln!(out, "{v}").expect("write");
            out.flush().expect("flush");
        }
    }
}
