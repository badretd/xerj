//! xerj-ask: XERJ-R end to end in Rust, JSON lines.
//!
//!   xerj-ask serve --models v2_ee=DIR,v1_rrt=DIR [--precision w8|f32|int8] [--int8-mode guard|smooth|raw]
//!       one request per stdin line, one response per stdout line (the XERJ Lab worker)
//!   xerj-ask batch --models ... --in requests.jsonl --out responses.jsonl
//!
//! Requests:
//!   {"op":"ping"}
//!   {"op":"answer","world":{memory-world-v1},"apply":[edit indices] (default all),"snapshot":"base"|"final","question":"...",
//!    "item_id":"..." (default lab/q/<sha256[:16]>),"models":["v2_ee",...] (default all),"linker":"exact"|"tolerant",
//!    "with_store":false}
//!   {"op":"answer_batch","world":{...},"questions":[{"item_id","question","snapshot","variant",...}],"models":[...],
//!    "linkers":["exact","tolerant"]}
//!   {"op":"search","text":"...","questions":["..."],"k":3}            sentence search, exact and fuzzy
use serde_json::{json, Value};
use std::io::{BufRead, BufWriter, Write};
use std::path::PathBuf;
use std::time::Instant;
use xerj_infer::{CpuFeatures, Kernels, Precision};
use xerj_serve::engine::{
    answer, build_world, lab_item_id, round1, store_rows, LinkerMode, ModelSlot,
};
use xerj_serve::textsearch::TextIndex;

fn arg(args: &[String], k: &str) -> Option<String> {
    args.iter()
        .position(|a| a == k)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

fn pick<'a>(slots: &'a [ModelSlot], req: &Value) -> Vec<&'a ModelSlot> {
    match req.get("models").and_then(|m| m.as_array()) {
        Some(names) => names
            .iter()
            .filter_map(|n| n.as_str())
            .filter_map(|n| slots.iter().find(|s| s.name == n))
            .collect(),
        None => slots.iter().collect(),
    }
}

fn apply_of(req: &Value) -> Option<Vec<usize>> {
    req.get("apply").and_then(|a| a.as_array()).map(|a| {
        a.iter()
            .filter_map(|x| x.as_u64())
            .map(|x| x as usize)
            .collect()
    })
}

fn handle(req: &Value, slots: &[ModelSlot]) -> Value {
    let op = req.get("op").and_then(|o| o.as_str()).unwrap_or("");
    match op {
        "ping" => {
            let f = CpuFeatures::detect();
            json!({"ok": true, "models": slots.iter().map(|s| s.info()).collect::<Vec<_>>(),
                   "kernels": format!("{:?}", Kernels::detect()), "cpu": format!("{:?}", f), "pid": std::process::id()})
        }
        "answer" => {
            let Some(world) = req.get("world") else {
                return json!({"ok": false, "error": "no world"});
            };
            let question = req
                .get("question")
                .and_then(|q| q.as_str())
                .unwrap_or("")
                .trim()
                .to_string();
            if question.is_empty() {
                return json!({"ok": false, "error": "type a question"});
            }
            let built = match build_world(world, apply_of(req).as_deref()) {
                Ok(b) => b,
                Err(e) => return json!({"ok": false, "error": e}),
            };
            let rev = if req.get("snapshot").and_then(|s| s.as_str()) == Some("base") {
                built.base
            } else {
                built.final_
            };
            let snap = built.mem.snapshot_at(rev);
            let item_id = req
                .get("item_id")
                .and_then(|s| s.as_str())
                .map(String::from)
                .unwrap_or_else(|| lab_item_id(&question));
            let linker = LinkerMode::parse(
                req.get("linker")
                    .and_then(|s| s.as_str())
                    .unwrap_or("exact"),
            );
            let results: Vec<Value> = pick(slots, req)
                .iter()
                .map(|s| answer(s, &built.mem, snap, &item_id, &question, linker, true))
                .collect();
            let mut out = json!({"ok": true, "results": results, "item_id": item_id, "world_ms": round1(built.build_ms),
                                 "snapshot": {"revision": rev, "fingerprint": &*snap.fingerprint(), "facts": snap.visible_records().len()}});
            if req
                .get("with_store")
                .and_then(|b| b.as_bool())
                .unwrap_or(false)
            {
                out["store"] = store_rows(&built.mem, &snap);
            }
            out
        }
        "answer_batch" => {
            let Some(world) = req.get("world") else {
                return json!({"ok": false, "error": "no world"});
            };
            let built = match build_world(world, apply_of(req).as_deref()) {
                Ok(b) => b,
                Err(e) => return json!({"ok": false, "error": e, "id": req.get("id")}),
            };
            let linkers: Vec<LinkerMode> = req
                .get("linkers")
                .and_then(|l| l.as_array())
                .map(|a| {
                    a.iter()
                        .filter_map(|x| x.as_str())
                        .map(LinkerMode::parse)
                        .collect()
                })
                .unwrap_or_else(|| vec![LinkerMode::Exact]);
            let models = pick(slots, req);
            let with_cands = req
                .get("with_cands")
                .and_then(|b| b.as_bool())
                .unwrap_or(false);
            let mut rows = Vec::new();
            for q in req
                .get("questions")
                .and_then(|q| q.as_array())
                .map(|v| v.as_slice())
                .unwrap_or(&[])
            {
                let question = q.get("question").and_then(|s| s.as_str()).unwrap_or("");
                let rev = if q.get("snapshot").and_then(|s| s.as_str()) == Some("base") {
                    built.base
                } else {
                    built.final_
                };
                let snap = built.mem.snapshot_at(rev);
                let item_id = q
                    .get("item_id")
                    .and_then(|s| s.as_str())
                    .map(String::from)
                    .unwrap_or_else(|| lab_item_id(question));
                for &lk in &linkers {
                    for s in &models {
                        let mut r = answer(s, &built.mem, snap, &item_id, question, lk, with_cands);
                        if let (Some(o), Some(qo)) = (r.as_object_mut(), q.as_object()) {
                            for (k, v) in qo {
                                if k != "question" && k != "item_id" {
                                    o.insert(format!("q_{k}"), v.clone());
                                }
                            }
                        }
                        rows.push(r);
                    }
                }
            }
            json!({"ok": true, "id": req.get("id"), "world_ms": round1(built.build_ms), "results": rows})
        }
        "search" => {
            let text = req.get("text").and_then(|t| t.as_str()).unwrap_or("");
            let k = req.get("k").and_then(|x| x.as_u64()).unwrap_or(3) as usize;
            let ix = TextIndex::build(text);
            let qs: Vec<&str> = req
                .get("questions")
                .and_then(|q| q.as_array())
                .map(|a| a.iter().filter_map(|x| x.as_str()).collect())
                .unwrap_or_default();
            let res: Vec<Value> = qs.iter().map(|q| json!({"exact": ix.search(text, q, k, false), "fuzzy": ix.search(text, q, k, true)})).collect();
            json!({"ok": true, "id": req.get("id"), "sentences": ix.sentences.len(), "build_us": round1(ix.build_us), "results": res})
        }
        _ => json!({"ok": false, "error": format!("unknown op {op:?}")}),
    }
}

fn load_slots(args: &[String]) -> Vec<ModelSlot> {
    let precision = match arg(args, "--precision").as_deref().unwrap_or("w8") {
        "f32" => Precision::F32,
        "int8" => Precision::Int8,
        _ => Precision::W8,
    };
    let mode = arg(args, "--int8-mode").unwrap_or_else(|| "guard".into());
    let mut slots = Vec::new();
    for spec in arg(args, "--models")
        .unwrap_or_default()
        .split(',')
        .filter(|s| !s.is_empty())
    {
        let (name, dir) = spec
            .split_once('=')
            .unwrap_or_else(|| panic!("--models NAME=DIR,... (got {spec})"));
        match ModelSlot::load(name, &PathBuf::from(dir), precision, &mode) {
            Ok(s) => slots.push(s),
            Err(e) => eprintln!("[xerj-ask] cannot load {name}: {e}"),
        }
    }
    slots
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let cmd = args.get(1).map(String::as_str).unwrap_or("");
    match cmd {
        "serve" => {
            let slots = load_slots(&args);
            let stdin = std::io::stdin();
            let mut out = BufWriter::new(std::io::stdout());
            writeln!(out, "{}", json!({"ready": true, "pid": std::process::id(), "models": slots.iter().map(|s| s.info()).collect::<Vec<_>>()})).unwrap();
            out.flush().unwrap();
            for line in stdin.lock().lines() {
                let Ok(line) = line else { break };
                if line.trim().is_empty() {
                    continue;
                }
                let t0 = Instant::now();
                let mut res = match serde_json::from_str::<Value>(&line) {
                    Ok(req) => handle(&req, &slots),
                    Err(e) => json!({"ok": false, "error": format!("bad json: {e}")}),
                };
                res["worker_ms"] = json!(round1(t0.elapsed().as_secs_f64() * 1e3));
                writeln!(out, "{res}").unwrap();
                out.flush().unwrap();
            }
        }
        "batch" => {
            let slots = load_slots(&args);
            let inp = arg(&args, "--in").expect("--in");
            let outp = arg(&args, "--out").expect("--out");
            let f = std::io::BufReader::new(std::fs::File::open(&inp).expect("open --in"));
            let mut out = BufWriter::new(std::fs::File::create(&outp).expect("create --out"));
            let t0 = Instant::now();
            let mut n = 0;
            for line in f.lines() {
                let line = line.expect("read");
                if line.trim().is_empty() {
                    continue;
                }
                let req: Value = serde_json::from_str(&line).expect("json request");
                let res = handle(&req, &slots);
                writeln!(out, "{res}").unwrap();
                n += 1;
            }
            out.flush().unwrap();
            eprintln!(
                "{}",
                json!({"requests": n, "wall_s": round1(t0.elapsed().as_secs_f64()), "models": slots.iter().map(|s| s.info()).collect::<Vec<_>>()})
            );
        }
        _ => {
            eprintln!("usage: xerj-ask serve|batch --models NAME=DIR,... [--precision w8|f32|int8] [--int8-mode guard|smooth|raw] [--in F --out F]");
            std::process::exit(2);
        }
    }
}
