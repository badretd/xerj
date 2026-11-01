//! Research tool (e2e1, never served): the Rust decline port against the frozen Python on random strings
//! (research/xerj_r_mvp/runs/e2e1/s3/fuzz_gen.py). Exit code 1 on any difference; every difference is printed.
//!
//!   decline_fuzz FUZZ.jsonl

use serde_json::{json, Value};
use std::io::BufRead;
use xerj_serve::decline_v2::pyre::{Budget, DEFAULT_BUDGET};
use xerj_serve::decline_v2::{p3, roles as ro, rules as r, Read, Row, Step, TraceStep, P3};

fn row_of(v: &Value, q: &str) -> Row {
    Row {
        question: q.to_string(),
        status: v["status"].as_str().unwrap().to_string(),
        answer: v["answer"].as_str().map(str::to_string),
        program: v["program"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| Step {
                op: p["op"].as_str().unwrap().to_string(),
                pointer: p["pointer"].as_i64(),
            })
            .collect(),
        trace: v["execution_trace"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| TraceStep {
                op: t["op"].as_str().unwrap().to_string(),
                pointer: t["pointer"].as_i64(),
                result: t["result"].as_str().map(str::to_string),
                hop_candidate_ids: t["hop_candidate_ids"]
                    .as_array()
                    .map(|h| h.iter().map(|x| x.as_str().unwrap().to_string()).collect()),
            })
            .collect(),
        evidence_ids: v["evidence_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_str().unwrap().to_string())
            .collect(),
    }
}

fn main() {
    let p = std::env::args().nth(1).expect("FUZZ.jsonl");
    let (mut n, mut bad) = (0u64, 0u64);
    for line in std::io::BufReader::new(std::fs::File::open(p).unwrap()).lines() {
        let v: Value = serde_json::from_str(&line.unwrap()).unwrap();
        n += 1;
        let b = Budget::new(DEFAULT_BUDGET);
        let q = v["q"].as_str().unwrap();
        let q2 = v["q2"].as_str().unwrap();
        let row = row_of(&v["row"], q);
        let mut diff = |what: &str, want: &Value, got: &Value| {
            if want != got {
                bad += 1;
                println!(
                    "DIFF line {n} {what}\n  q={q:?}\n  q2={q2:?}\n  python={want}\n  rust  ={got}"
                );
            }
        };
        let ms = r::mask(q, &b).unwrap();
        diff("mask", &v["mask"], &json!(ms));
        diff(
            "tokens",
            &v["tokens"],
            &json!(r::tokenize(&ms, &b).unwrap()),
        );
        diff("p3", &v["p3"], &json!(p3(&row, &b).unwrap() == P3::Accept));
        let d = r::check(q, &row, &b);
        diff(
            "d2",
            &v["d2"],
            &d.map_or(Value::Null, |d| json!([d.code, d.rule, d.detail])),
        );
        if row.status == "ANSWER" {
            let all: Vec<Value> = r::check_all(q, &row, &b)
                .unwrap()
                .iter()
                .map(|d| json!([d.code, d.detail]))
                .collect();
            diff("d2_all", &v["d2_all"], &json!(all));
        }
        let (n1, n2) = (v["n1"].as_str().unwrap(), v["n2"].as_str().unwrap());
        let (sk, sp) = ro::skeleton(q2, [n1, n2], &b).unwrap();
        diff("skeleton", &v["skeleton"], &json!(sk));
        diff("spans", &v["spans"], &json!(sp));
        let reads: Vec<Read> = v["reads"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| Read {
                step: x[0].as_u64().unwrap() as usize,
                op: x[1].as_str().unwrap().to_string(),
                subject: x[2].as_str().map(str::to_string),
                relation: x[3].as_str().map(str::to_string),
                fact_id: x[4].as_str().map(str::to_string),
                result: x[5].as_str().map(str::to_string),
            })
            .collect();
        let kind = |x: &Read| -> Option<String> {
            let s = x.result.as_deref()?;
            s.starts_with("VALUE ")
                .then(|| s.split_whitespace().nth(1).map(str::to_string))
                .flatten()
        };
        let (k1, k2) = (kind(&reads[0]), kind(&reads[1]));
        let got = match ro::read_roles(q2, [n1, n2], [k1.as_deref(), k2.as_deref()], &b).unwrap() {
            ro::Roles::Undetermined(u) => json!({"undetermined": u}),
            ro::Roles::Read {
                subject,
                reference,
                expected,
                why,
            } => {
                json!({"subject": subject, "reference": reference, "expected": expected, "why": why})
            }
        };
        diff("roles", &v["roles"], &got);
        let mut row2 = row.clone();
        row2.question = q2.to_string();
        let g = ro::check(q2, &row2, &reads, &b).unwrap();
        diff(
            "role",
            &v["role"],
            &json!([g.code, g.detail, g.applies, g.repairable]),
        );
    }
    println!("{n} cases, {bad} differences");
    std::process::exit(i32::from(bad > 0));
}
