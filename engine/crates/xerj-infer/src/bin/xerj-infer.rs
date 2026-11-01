//! xerj-infer CLI: `info`, `parity` (golden fixtures from PyTorch) and `bench` (latency / throughput / RSS).
//!
//!   xerj-infer info   --model DIR [--precision f32|w8|int8]
//!   xerj-infer parity --model DIR --fixtures F.jsonl[,G.jsonl] [--precision f32|w8|int8] [--limit N] [--verbose]
//!   xerj-infer bench  --model DIR --fixtures F.jsonl[,...] [--precision f32|w8|int8] [--rounds R] [--threads T] [--limit N]
//!   xerj-infer w8check --model DIR --fakequant DIR2 [--json OUT]   (w8 rounding of DIR == the f32 tensors of export DIR2)
//! common: --f32-isa scalar|neon|avx2|avx512   --i8-isa scalar|dot|i8mm|avx2|vnni   --json OUT (machine-readable result)

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Instant;
use xerj_infer::codec;
use xerj_infer::episode::Guard;
use xerj_infer::episode::{self, DecisionEvent};
use xerj_infer::fixtures::{self, FixtureRow, ReplayExecutor};
use xerj_infer::kernels::{CpuFeatures, F32Isa, I8Isa, Kernels};
use xerj_infer::{Engine, Int8Calib, LoadOptions, Model, Precision, Session};

struct Args {
    cmd: String,
    kv: BTreeMap<String, String>,
    flags: Vec<String>,
}

fn parse_args() -> Args {
    let mut it = std::env::args().skip(1);
    let cmd = it.next().unwrap_or_else(|| "help".into());
    let mut kv = BTreeMap::new();
    let mut flags = Vec::new();
    let rest: Vec<String> = it.collect();
    let mut i = 0;
    while i < rest.len() {
        let a = &rest[i];
        if let Some(k) = a.strip_prefix("--") {
            if i + 1 < rest.len() && !rest[i + 1].starts_with("--") {
                kv.insert(k.to_string(), rest[i + 1].clone());
                i += 2;
                continue;
            }
            flags.push(k.to_string());
        }
        i += 1;
    }
    Args { cmd, kv, flags }
}

impl Args {
    fn get(&self, k: &str) -> Option<&str> {
        self.kv.get(k).map(|s| s.as_str())
    }
    fn flag(&self, k: &str) -> bool {
        self.flags.iter().any(|f| f == k)
    }
    fn usize(&self, k: &str, d: usize) -> usize {
        self.get(k).and_then(|v| v.parse().ok()).unwrap_or(d)
    }
}

fn kernels(a: &Args) -> Kernels {
    let f = CpuFeatures::detect();
    let mut k = Kernels::detect();
    if let Some(s) = a.get("f32-isa") {
        k.f32_isa = match s {
            "scalar" => F32Isa::Scalar,
            #[cfg(target_arch = "aarch64")]
            "neon" => F32Isa::Neon,
            #[cfg(target_arch = "x86_64")]
            "avx2" => F32Isa::Avx2,
            #[cfg(target_arch = "x86_64")]
            "avx512" => F32Isa::Avx512,
            other => panic!("unknown --f32-isa {other}"),
        };
        assert!(
            f.supports_f32(k.f32_isa),
            "CPU does not support {}",
            k.f32_isa.name()
        );
    }
    if let Some(s) = a.get("i8-isa") {
        k.i8_isa = match s {
            "scalar" => I8Isa::Scalar,
            #[cfg(target_arch = "aarch64")]
            "dot" => I8Isa::NeonDot,
            #[cfg(target_arch = "aarch64")]
            "i8mm" => I8Isa::NeonI8mm,
            #[cfg(target_arch = "x86_64")]
            "avx2" => I8Isa::Avx2,
            #[cfg(target_arch = "x86_64")]
            "vnni" => I8Isa::Avx512Vnni,
            other => panic!("unknown --i8-isa {other}"),
        };
        assert!(
            f.supports_i8(k.i8_isa),
            "CPU does not support {}",
            k.i8_isa.name()
        );
    }
    match a.get("kleidi") {
        Some("off") => k.kleidi = false,
        Some("on") => {
            assert!(
                xerj_infer::kernels::kleidi_available(),
                "KleidiAI backend not compiled in (feature kleidiai) or no FEAT_I8MM"
            );
            k.kleidi = true;
        }
        _ => {}
    }
    if k.i8_isa != xerj_infer::kernels::CpuFeatures::detect().best_i8() {
        k.kleidi = false;
    }
    k
}

/// default calibration file of a kernel set (the KleidiAI path quantizes activations differently: its own thresholds)
fn calib_name(k: &Kernels) -> &'static str {
    if k.kleidi {
        "int8_calib.kleidiai.json"
    } else {
        "int8_calib.json"
    }
}

fn precision(a: &Args) -> Precision {
    match a.get("precision").unwrap_or("f32") {
        "f32" => Precision::F32,
        "int8" => Precision::Int8,
        "w8" => Precision::W8,
        other => panic!("unknown --precision {other}"),
    }
}

/// int8 variants: raw (per-row W8A8), smooth (+ calibrated smoothing), guard (smooth + f32 recompute of close decisions)
fn int8_mode(a: &Args) -> &str {
    a.get("int8-mode").unwrap_or("guard")
}

/// --intra T: threads per question (weights split into T output-channel shards, attention heads spread); 1 = off
fn intra(a: &Args) -> usize {
    a.usize("intra", 1).max(1)
}

/// a session with the intra-question pool attached (when --intra > 1)
fn session_with<'e>(
    e: &'e Engine,
    pool: &Option<std::sync::Arc<xerj_infer::pool::Pool>>,
) -> Session<'e> {
    let mut s = e.session();
    s.set_pool(pool.clone());
    s
}

fn make_pool(a: &Args) -> Option<std::sync::Arc<xerj_infer::pool::Pool>> {
    (intra(a) > 1).then(|| std::sync::Arc::new(xerj_infer::pool::Pool::new(intra(a))))
}

fn load_engine(a: &Args, dir: &std::path::Path) -> (Engine, Option<Guard>) {
    let k = kernels(a);
    let prec = precision(a);
    let mut opt = LoadOptions::new(prec, k);
    opt.shards = intra(a);
    let mut guard = None;
    if prec == Precision::Int8 && int8_mode(a) != "raw" {
        let path = a
            .get("calib")
            .map(PathBuf::from)
            .unwrap_or_else(|| dir.join(calib_name(&k)));
        match Int8Calib::load(&path) {
            Ok(c) => {
                if int8_mode(a) == "guard" {
                    opt.keep_f32 = true;
                    guard = Some(Guard {
                        tau_op: c.tau_op,
                        tau_ptr: c.tau_ptr,
                    });
                }
                opt.calib = Some(c);
            }
            Err(e) => eprintln!("warning: no int8 calibration ({e}); running raw int8"),
        }
    }
    (Engine::load_with(dir, &opt).expect("load"), guard)
}

/// x86: the AVX2 + FMA element-wise / attention kernels are in use (XERJ_AVX2_OPS)
fn avx2_ops() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        xerj_infer::kernels::x86ops::enabled()
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

fn load_rows(a: &Args) -> Vec<FixtureRow> {
    load_rows_opt(a, false)
}

fn load_rows_opt(a: &Args, strip: bool) -> Vec<FixtureRow> {
    let mut rows = Vec::new();
    for p in a.get("fixtures").expect("--fixtures").split(',') {
        let mut r =
            fixtures::load_opt(&PathBuf::from(p), strip).unwrap_or_else(|e| panic!("{p}: {e}"));
        let lim = a.usize("limit", usize::MAX);
        r.truncate(lim);
        rows.extend(r);
    }
    rows
}

fn proc_status(key: &str) -> Option<f64> {
    let s = std::fs::read_to_string("/proc/self/status").ok()?;
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix(key) {
            let kb: f64 = rest
                .trim_start_matches(':')
                .trim()
                .trim_end_matches("kB")
                .trim()
                .parse()
                .ok()?;
            return Some(kb / 1024.0);
        }
    }
    None
}

fn mean_of(v: &[f64]) -> f64 {
    if v.is_empty() {
        f64::NAN
    } else {
        v.iter().sum::<f64>() / v.len() as f64
    }
}

fn pct(v: &mut [f64], p: f64) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let i = ((p * (v.len() - 1) as f64).round() as usize).min(v.len() - 1);
    v[i]
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> (f64, usize) {
    // masked entries (f32::MIN) must coincide; returns (max diff over unmasked entries, mask mismatches)
    let mut worst = 0f64;
    let mut mm = 0;
    if a.len() != b.len() {
        return (f64::INFINITY, a.len().max(b.len()));
    }
    for (x, y) in a.iter().zip(b) {
        let (mx, my) = (*x <= f32::MIN, *y <= f32::MIN);
        if mx || my {
            if mx != my {
                mm += 1;
            }
            continue;
        }
        worst = worst.max((*x as f64 - *y as f64).abs());
    }
    (worst, mm)
}

fn info(a: &Args) {
    let dir = PathBuf::from(a.get("model").expect("--model"));
    let k = kernels(a);
    let rss0 = proc_status("VmRSS").unwrap_or(f64::NAN);
    let (e, _) = load_engine(a, &dir);
    let rss1 = proc_status("VmRSS").unwrap_or(f64::NAN);
    {
        let t = std::time::Instant::now();
        let tok2 = xerj_infer::tokenizer::HfTokenizer::from_file(&dir.join("tokenizer.json"))
            .expect("tokenizer");
        let rss2 = proc_status("VmRSS").unwrap_or(f64::NAN);
        println!("RSS MB: process start {rss0:.1}; after model + tokenizer {rss1:.1}; one more tokenizer instance +{:.1} MB ({:.1} ms)", rss2 - rss1, t.elapsed().as_secs_f64() * 1e3);
        drop(tok2);
    }
    let c = &e.model.cfg;
    let feats = CpuFeatures::detect();
    println!("model {} arm={} stack={:?} codec={:?} K={} layers={} d={} heads={} window={} fingerprint={:?}", dir.display(), c.arm.name(), c.stack, c.codec, c.k, c.layers, c.hidden, c.heads, c.window, c.fingerprint);
    println!(
        "precision {} f32-isa {} i8-isa {} cpu {:?}",
        e.model.precision.name(),
        k.f32_isa.name(),
        k.i8_name(),
        feats
    );
    println!("load {:.1} ms (+ tokenizer {:.1} ms); resident weights {:.2} MB; embedding table {:.2} MB (mmap)", e.model.load_ms, e.tokenizer_ms, e.model.resident_weight_bytes() as f64 / 1e6, e.model.embedding_bytes() as f64 / 1e6);
}

#[derive(Default)]
struct Parity {
    questions: usize,
    decisions: usize,
    ids_mismatch: usize,
    max_op: f64,
    max_ptr: f64,
    mask_mismatch: usize,
    op_agree: usize,
    action_agree: usize,
    mem_max: f64,
    hopmem_max: f64,
    reg_max: f64,
    free_identical: usize,
    worst: Vec<(f64, String)>,
    diverged: Vec<String>,
    fallbacks_forced: usize,
    fallbacks_free: usize,
    fallback_questions: usize,
    // stack-v5 coverage of the compared decisions
    hop_decisions: usize,
    inverse_input_decisions: usize,
    inverse_actions: usize,
    inverse_actions_agree: usize,
    max_abs_ptr_logit: f64,
}

fn parity_row(
    sess: &mut Session,
    row: &FixtureRow,
    p: &mut Parity,
    verbose: bool,
    guard: Option<Guard>,
) {
    let d = sess.m.cfg.hidden;
    let w = d + 4;
    let slots = sess.m.cfg.codec.slots();
    let k = sess.m.cfg.k;
    // ---- forced: every recorded decision with the recorded history / registers / hop
    {
        let mut ex = ReplayExecutor::new(row, true);
        let mut obs = |s: &Session, ev: &DecisionEvent| {
            let rec = &row.decisions[ev.step];
            p.decisions += 1;
            if s.input_ids() != rec.ids {
                p.ids_mismatch += 1;
                if verbose {
                    eprintln!(
                        "{} d{}: input ids differ (rust {} vs py {})",
                        row.item_id,
                        ev.step,
                        s.input_ids().len(),
                        rec.ids.len()
                    );
                }
            }
            let (dop, _) = max_abs_diff(&ev.out.op_logits, &rec.op_logits);
            let (dptr, mm) = max_abs_diff(&ev.out.ptr_logits, &rec.ptr_logits);
            p.max_op = p.max_op.max(dop);
            p.max_ptr = p.max_ptr.max(dptr);
            p.mask_mismatch += mm;
            p.worst
                .push((dop.max(dptr), format!("{} d{}", row.item_id, ev.step)));
            for &v in &rec.ptr_logits {
                if v > f32::MIN {
                    p.max_abs_ptr_logit = p.max_abs_ptr_logit.max(v.abs() as f64);
                }
            }
            p.hop_decisions += rec.hop.is_some() as usize;
            p.inverse_input_decisions += (row.cands.iter().any(|f| f.inverse)
                || rec
                    .hop
                    .as_ref()
                    .is_some_and(|h| h.cands.iter().any(|f| f.inverse)))
                as usize;
            p.inverse_actions += rec.trace.inverse as usize;
            if ev.op == rec.op {
                p.op_agree += 1;
            }
            if ev.op == rec.op && ev.pointer == rec.pointer {
                p.action_agree += 1;
                p.inverse_actions_agree += rec.trace.inverse as usize;
            } else if verbose {
                let lg =
                    |v: &[f32], p: Option<usize>| p.map(|i| v[i]).unwrap_or(*v.last().unwrap());
                let fields = |p: Option<usize>| -> String {
                    match p {
                        Some(i) if i < row.cands.len() => {
                            format!("{} | {}", row.cands[i].subject, row.cands[i].relation)
                        }
                        Some(i) => format!("hop{}", i),
                        None => "NULL".into(),
                    }
                };
                eprintln!(
                    "{} d{}: action rust ({}, {:?}) vs py ({}, {:?}); py logits {:.4} vs {:.4} (margin {:.4}); rust {:.4} vs {:.4}; [{}] vs [{}]",
                    row.item_id, ev.step, ev.op, ev.pointer, rec.op, rec.pointer,
                    lg(&rec.ptr_logits, ev.pointer), lg(&rec.ptr_logits, rec.pointer), lg(&rec.ptr_logits, rec.pointer) - lg(&rec.ptr_logits, ev.pointer),
                    lg(&ev.out.ptr_logits, ev.pointer), lg(&ev.out.ptr_logits, rec.pointer), fields(ev.pointer), fields(rec.pointer)
                );
            }
            let mv = s.memory_values();
            if ev.step == 0 {
                if let Some(mem) = &row.memory {
                    let n = mem.len().min(k * slots * w);
                    let (dm, _) = max_abs_diff(&mv[..n], &mem[..n]);
                    p.mem_max = p.mem_max.max(dm);
                }
            }
            if let Some(hm) = &rec.hop_memory {
                if mv.len() >= 2 * k * slots * w {
                    let (dm, _) =
                        max_abs_diff(&mv[k * slots * w..2 * k * slots * w], &hm[..k * slots * w]);
                    p.hopmem_max = p.hopmem_max.max(dm);
                }
            }
            if let Some((rf, rm)) = &rec.registers {
                let mut f = [0f32; 64];
                let mut m = [false; 4];
                codec::register_features(&rec.stack, &mut f, &mut m);
                let (dr, _) = max_abs_diff(&f, rf);
                p.reg_max = p.reg_max.max(dr);
                if m.as_slice() != rm.as_slice() {
                    p.reg_max = f64::INFINITY;
                }
            }
        };
        let obs_dyn: &mut dyn FnMut(&Session, &DecisionEvent) = &mut obs;
        let st = episode::run_episode_guarded(
            sess,
            &row.question,
            &row.cands,
            &mut ex,
            Some(obs_dyn),
            guard,
        )
        .expect("forced episode");
        p.fallbacks_forced += st.fallbacks;
    }
    // ---- free running: the engine's own actions drive the replay executor
    let mut ex = ReplayExecutor::new(row, false);
    let st = episode::run_episode_guarded(sess, &row.question, &row.cands, &mut ex, None, guard)
        .expect("free episode");
    p.fallbacks_free += st.fallbacks;
    p.fallback_questions += (st.fallbacks > 0) as usize;
    if ex.identical_episode() {
        p.free_identical += 1;
    } else {
        p.diverged.push(format!(
            "{} (decision {:?})",
            row.item_id, ex.first_divergence
        ));
    }
    p.questions += 1;
}

fn parity(a: &Args) {
    let dir = PathBuf::from(a.get("model").expect("--model"));
    let k = kernels(a);
    let prec = precision(a);
    let (e, guard) = load_engine(a, &dir);
    let rows = load_rows(a);
    let verbose = a.flag("verbose");
    let pool = make_pool(a);
    let mut sess = session_with(&e, &pool);
    let mut p = Parity::default();
    let t0 = Instant::now();
    for row in &rows {
        parity_row(&mut sess, row, &mut p, verbose, guard);
    }
    p.worst.sort_by(|x, y| y.0.partial_cmp(&x.0).unwrap());
    let res = serde_json::json!({
        "model": dir.display().to_string(), "arm": e.model.cfg.arm.name(), "stack": format!("{:?}", e.model.cfg.stack),
        "precision": prec.name(), "f32_isa": k.f32_isa.name(), "i8_isa": k.i8_name(), "intra_threads": intra(a),
        "avx2_ops": avx2_ops(),
        "int8_mode": if prec == Precision::Int8 { serde_json::json!(if guard.is_some() { "guard" } else if e.model.calib.is_some() { "smooth" } else { "raw" }) } else { serde_json::Value::Null },
        "guard": guard.map(|g| serde_json::json!({"tau_op": g.tau_op, "tau_ptr": g.tau_ptr})),
        "fallback_decisions_forced": p.fallbacks_forced, "fallback_decisions_free": p.fallbacks_free, "fallback_questions_free": p.fallback_questions,
        "fixtures": a.get("fixtures"), "questions": p.questions, "decisions": p.decisions,
        "input_ids_mismatch": p.ids_mismatch, "max_abs_op_logit_diff": p.max_op, "max_abs_ptr_logit_diff": p.max_ptr,
        "masked_slot_mismatch": p.mask_mismatch, "op_agreement": p.op_agree as f64 / p.decisions.max(1) as f64,
        "action_agreement": p.action_agree as f64 / p.decisions.max(1) as f64, "actions_identical": p.action_agree,
        "memory_max_abs_diff": p.mem_max, "hop_memory_max_abs_diff": p.hopmem_max, "register_max_abs_diff": p.reg_max,
        "max_abs_ptr_logit_reference": p.max_abs_ptr_logit,
        "coverage": {"decisions_with_hop_block": p.hop_decisions, "decisions_with_inverse_lines": p.inverse_input_decisions,
                     "inverse_actions_recorded": p.inverse_actions, "inverse_actions_identical": p.inverse_actions_agree},
        "free_running_identical_episodes": p.free_identical, "free_running_identical_fraction": p.free_identical as f64 / p.questions.max(1) as f64,
        "worst": p.worst.iter().take(5).map(|(d, s)| format!("{s}: {d:.3e}")).collect::<Vec<_>>(),
        "diverged": p.diverged.iter().take(20).collect::<Vec<_>>(),
        "seconds": t0.elapsed().as_secs_f64(),
    });
    println!("{}", serde_json::to_string_pretty(&res).unwrap());
    if let Some(out) = a.get("json") {
        std::fs::write(out, serde_json::to_string_pretty(&res).unwrap()).expect("write json");
    }
}

/// per-decision wall times of the measured questions (--decision-times): the first decision includes the question's
/// prefix forward, a fallback decision includes its f32 recompute (prefix + history from scratch)
#[derive(Default)]
struct DecTimes {
    first: Vec<f64>,
    later: Vec<f64>,
    fallback: Vec<f64>,
}

fn run_free_timed(
    sess: &mut Session,
    row: &FixtureRow,
    guard: Option<Guard>,
    dt: &mut DecTimes,
) -> (usize, bool, usize) {
    let mut ex = ReplayExecutor::new(row, false);
    let mut last = Instant::now();
    let mut obs = |_s: &Session, ev: &DecisionEvent| {
        let now = Instant::now();
        let ms = (now - last).as_secs_f64() * 1e3;
        last = now;
        if ev.fallback {
            dt.fallback.push(ms);
        } else if ev.step == 0 {
            dt.first.push(ms);
        } else {
            dt.later.push(ms);
        }
    };
    let od: &mut dyn FnMut(&Session, &DecisionEvent) = &mut obs;
    let st =
        episode::run_episode_guarded(sess, &row.question, &row.cands, &mut ex, Some(od), guard)
            .expect("episode");
    (st.decisions, ex.identical_episode(), st.fallbacks)
}

/// one free-running question; `fresh`: on a new session (as the engine's answer() creates one per question) instead of the
/// reused one
fn run_free_opt(
    e: &Engine,
    pool: &Option<std::sync::Arc<xerj_infer::pool::Pool>>,
    sess: &mut Session,
    row: &FixtureRow,
    guard: Option<Guard>,
    fresh: bool,
) -> (usize, bool, usize) {
    if fresh {
        let mut s = session_with(e, pool);
        run_free(&mut s, row, guard)
    } else {
        run_free(sess, row, guard)
    }
}

fn run_free(sess: &mut Session, row: &FixtureRow, guard: Option<Guard>) -> (usize, bool, usize) {
    let mut ex = ReplayExecutor::new(row, false);
    let st = episode::run_episode_guarded(sess, &row.question, &row.cands, &mut ex, None, guard)
        .expect("episode");
    (st.decisions, ex.identical_episode(), st.fallbacks)
}

fn bench(a: &Args) {
    let dir = PathBuf::from(a.get("model").expect("--model"));
    let k = kernels(a);
    let prec = precision(a);
    let t0 = Instant::now();
    let (e, guard) = load_engine(a, &dir);
    let load_ms = t0.elapsed().as_secs_f64() * 1e3;
    let rss_after_load = proc_status("VmRSS").unwrap_or(f64::NAN);
    let rows = load_rows_opt(a, true);
    let rounds = a.usize("rounds", 3);
    let threads = a.usize("threads", 1);
    let rss_after_fixtures = proc_status("VmRSS").unwrap_or(f64::NAN);
    let pool = make_pool(a);
    // warm-up pass (also checks that every episode replays); its per-question times are the "cold" latency: every text segment is
    // tokenized for the first time and the touched embedding rows are paged in (the measured rounds reuse the segment cache)
    let mut identical = 0;
    let mut cold_ms = Vec::new();
    {
        let mut sess = session_with(&e, &pool);
        for row in &rows {
            let t = Instant::now();
            identical += run_free(&mut sess, row, guard).1 as usize;
            cold_ms.push(t.elapsed().as_secs_f64() * 1e3);
        }
    }
    let cold_mean = cold_ms.iter().sum::<f64>() / cold_ms.len().max(1) as f64;
    let mut lat_ms = Vec::new();
    let mut decisions = 0usize;
    let mut fallbacks = 0usize;
    let wall;
    let _ = xerj_infer::prof::take();
    let mut profile = serde_json::Value::Null;
    let fresh = a.flag("fresh");
    let timed_dec = a.flag("decision-times");
    let mut dt = DecTimes::default();
    if threads <= 1 {
        let mut sess = session_with(&e, &pool);
        let tw = Instant::now();
        for _ in 0..rounds {
            for row in &rows {
                let t = Instant::now();
                let (nd, _, fb) = if timed_dec {
                    run_free_timed(&mut sess, row, guard, &mut dt)
                } else {
                    run_free_opt(&e, &pool, &mut sess, row, guard, fresh)
                };
                lat_ms.push(t.elapsed().as_secs_f64() * 1e3);
                decisions += nd;
                fallbacks += fb;
            }
        }
        wall = tw.elapsed().as_secs_f64();
        if xerj_infer::prof::enabled() {
            let nq = (rounds * rows.len()).max(1) as f64;
            let parts = xerj_infer::prof::take();
            let covered: f64 = parts.iter().map(|p| p.1).sum();
            let mut m = serde_json::Map::new();
            for (name, sec) in parts {
                m.insert(name.to_string(), serde_json::json!(1e3 * sec / nq));
            }
            m.insert(
                "other".into(),
                serde_json::json!(1e3 * (wall - covered) / nq),
            );
            profile = serde_json::Value::Object(m);
        }
    } else {
        let tw = Instant::now();
        let results: Vec<(Vec<f64>, usize)> = std::thread::scope(|s| {
            let hs: Vec<_> = (0..threads)
                .map(|tid| {
                    let e = &e;
                    let rows = &rows;
                    s.spawn(move || {
                        let mut sess = e.session();
                        let mut lat = Vec::new();
                        let mut nd = 0;
                        for _ in 0..rounds {
                            for row in rows.iter().skip(tid).step_by(threads) {
                                let t = Instant::now();
                                nd += run_free(&mut sess, row, guard).0;
                                lat.push(t.elapsed().as_secs_f64() * 1e3);
                            }
                        }
                        (lat, nd)
                    })
                })
                .collect();
            hs.into_iter().map(|h| h.join().unwrap()).collect()
        });
        wall = tw.elapsed().as_secs_f64();
        for (l, nd) in results {
            lat_ms.extend(l);
            decisions += nd;
        }
    }
    let n = lat_ms.len();
    let mean = lat_ms.iter().sum::<f64>() / n.max(1) as f64;
    let mut l2 = lat_ms.clone();
    let (p50, p90, p95) = (pct(&mut l2, 0.5), pct(&mut l2, 0.9), pct(&mut l2, 0.95));
    let mut py_lat: Vec<f64> = rows.iter().map(|r| r.py_latency_s * 1e3).collect();
    let mut py_dec: Vec<f64> = rows.iter().map(|r| r.py_decide_s * 1e3).collect();
    let py_mean = py_lat.iter().sum::<f64>() / py_lat.len().max(1) as f64;
    let res = serde_json::json!({
        "model": dir.display().to_string(), "arm": e.model.cfg.arm.name(), "stack": format!("{:?}", e.model.cfg.stack),
        "precision": prec.name(), "f32_isa": k.f32_isa.name(), "i8_isa": k.i8_name(), "threads": threads, "rounds": rounds,
        "intra_threads": intra(a), "avx2_ops": avx2_ops(), "fresh_session_per_question": a.flag("fresh"),
        "decision_ms": if timed_dec { serde_json::json!({
            "first_incl_prefix": {"n": dt.first.len(), "mean": mean_of(&dt.first), "p50": pct(&mut dt.first.clone(), 0.5), "p95": pct(&mut dt.first.clone(), 0.95)},
            "later": {"n": dt.later.len(), "mean": mean_of(&dt.later), "p50": pct(&mut dt.later.clone(), 0.5), "p95": pct(&mut dt.later.clone(), 0.95)},
            "fallback_incl_f32_recompute": {"n": dt.fallback.len(), "mean": mean_of(&dt.fallback), "p50": pct(&mut dt.fallback.clone(), 0.5), "p95": pct(&mut dt.fallback.clone(), 0.95)},
        }) } else { serde_json::Value::Null },
        "int8_mode": if prec == Precision::Int8 { serde_json::json!(if guard.is_some() { "guard" } else if e.model.calib.is_some() { "smooth" } else { "raw" }) } else { serde_json::Value::Null },
        "fallback_decisions": fallbacks,
        "fixtures": a.get("fixtures"), "questions": rows.len(), "episodes_replayed_identically": identical,
        "latency_ms": {"mean": mean, "p50": p50, "p90": p90, "p95": p95},
        "cold_first_pass_latency_ms": {"mean": cold_mean, "p50": pct(&mut cold_ms, 0.5), "p95": pct(&mut cold_ms, 0.95)},
        "questions_per_s": n as f64 / wall, "decisions_per_s": decisions as f64 / wall, "decisions_per_question": decisions as f64 / n.max(1) as f64,
        "load_ms_total": load_ms, "load_ms_model": e.model.load_ms, "load_ms_tokenizer": e.tokenizer_ms,
        "rss_mb": {"after_load": rss_after_load, "after_fixtures": rss_after_fixtures, "final": proc_status("VmRSS"), "peak_hwm": proc_status("VmHWM"),
                   "final_anon": proc_status("RssAnon"), "final_file": proc_status("RssFile")},
        "resident_weight_mb": e.model.resident_weight_bytes() as f64 / 1e6, "embedding_table_mb_mmap": e.model.embedding_bytes() as f64 / 1e6,
        "weights_file_mb": std::fs::metadata(&e.model.weights_path).map(|m| m.len() as f64 / 1e6).unwrap_or(f64::NAN),
        "profile_ms_per_question": profile,
        "python_same_questions": {"answer_item_ms": {"mean": py_mean, "p50": pct(&mut py_lat, 0.5), "p95": pct(&mut py_lat, 0.95)},
                                   "decide_ms": {"p50": pct(&mut py_dec, 0.5), "p95": pct(&mut py_dec, 0.95)}},
    });
    println!("{}", serde_json::to_string_pretty(&res).unwrap());
    if let Some(out) = a.get("json") {
        std::fs::write(out, serde_json::to_string_pretty(&res).unwrap()).expect("write json");
    }
}

/// int8 calibration on DEV fixtures: (1) per-channel input maxima of every quantized linear from the f32 model ->
/// SmoothQuant factors s_j = max|X_j|^alpha / max|W_j|^(1-alpha), alpha chosen from --alphas (default 0.3,0.5,0.7) by the
/// smallest pointer-logit error on the calibration decisions; (2) the smoothed int8 model vs f32 on the same decisions ->
/// largest logit errors -> guard thresholds tau = safety * 2 * max error (an int8 argmax whose margin exceeds twice the
/// error bound equals the f32 argmax; safety default 2.0 covers test errors larger than the dev maximum).
fn calibrate(a: &Args) {
    let dir = PathBuf::from(a.get("model").expect("--model"));
    let k = kernels(a);
    let rows = load_rows(a);
    let alphas: Vec<f32> = a
        .get("alphas")
        .unwrap_or("0.3,0.5,0.7")
        .split(',')
        .filter_map(|v| v.parse().ok())
        .collect();
    let safety: f32 = a.get("safety").and_then(|v| v.parse().ok()).unwrap_or(2.0);
    // (1) f32 statistics + reference logits of every forced decision
    let ef = Engine::load(&dir, Precision::F32, k).expect("load f32");
    let mut stats = std::collections::HashMap::new();
    let mut f32_logits: Vec<Vec<(Vec<f32>, Vec<f32>, [bool; 10])>> = Vec::new();
    {
        let mut sess = ef.session();
        sess.set_calibration(true);
        for row in &rows {
            let mut ex = ReplayExecutor::new(row, true);
            let mut got = Vec::new();
            let mut obs = |_s: &Session, ev: &DecisionEvent| {
                got.push((
                    ev.out.op_logits.clone(),
                    ev.out.ptr_logits.clone(),
                    ev.legal,
                ))
            };
            let od: &mut dyn FnMut(&Session, &DecisionEvent) = &mut obs;
            episode::run_episode(&mut sess, &row.question, &row.cands, &mut ex, Some(od))
                .expect("f32 episode");
            f32_logits.push(got);
        }
        sess.merge_calibration(&mut stats);
    }
    let smooth_for = |alpha: f32| -> Int8Calib {
        let mut c = Int8Calib {
            alpha,
            ..Default::default()
        };
        for lin in ef.model.linears() {
            if !Model::quantizable_name_pub(&lin.name) {
                continue;
            }
            let (Some(xs), Some(ws)) = (stats.get(&lin.name), lin.col_absmax()) else {
                continue;
            };
            let s: Vec<f32> = xs
                .iter()
                .zip(&ws)
                .map(|(x, w)| {
                    (x.max(1e-5).powf(alpha) / w.max(1e-5).powf(1.0 - alpha)).clamp(1e-3, 1e3)
                })
                .collect();
            c.smooth.insert(lin.name.clone(), s);
        }
        c
    };
    // (2) smoothed int8 vs f32 logits on the same forced decisions
    let errors = |c: &Int8Calib| -> (f64, f64) {
        let mut opt = LoadOptions::new(Precision::Int8, k);
        opt.calib = Some(c.clone());
        let ei = Engine::load_with(&dir, &opt).expect("load int8");
        let (mut eop, mut eptr) = (0f64, 0f64);
        let mut sess = ei.session();
        for (row, f) in rows.iter().zip(&f32_logits) {
            let mut ex = ReplayExecutor::new(row, true);
            let mut i = 0;
            let mut obs = |_s: &Session, ev: &DecisionEvent| {
                let (fo, fp, legal) = &f[i];
                for j in 0..10 {
                    if legal[j] {
                        eop = eop.max((ev.out.op_logits[j] - fo[j]).abs() as f64);
                    }
                }
                let (dp, _) = max_abs_diff(&ev.out.ptr_logits, fp);
                eptr = eptr.max(dp);
                i += 1;
            };
            let od: &mut dyn FnMut(&Session, &DecisionEvent) = &mut obs;
            episode::run_episode(&mut sess, &row.question, &row.cands, &mut ex, Some(od))
                .expect("int8 episode");
        }
        (eop, eptr)
    };
    let mut best: Option<(Int8Calib, f64, f64)> = None;
    let mut tried = Vec::new();
    for &al in &alphas {
        let c = smooth_for(al);
        let (eop, eptr) = errors(&c);
        tried.push(serde_json::json!({"alpha": al, "max_err_op": eop, "max_err_ptr": eptr}));
        if best.as_ref().map_or(true, |b| eptr < b.2) {
            best = Some((c, eop, eptr));
        }
    }
    let (mut calib, eop, eptr) = best.expect("at least one alpha");
    calib.max_err_op = eop as f32;
    calib.max_err_ptr = eptr as f32;
    calib.tau_op = (2.0 * safety as f64 * eop) as f32;
    calib.tau_ptr = (2.0 * safety as f64 * eptr) as f32;
    calib.source = format!(
        "{} ({} questions, alpha {} chosen from {:?} by the pointer error, safety {safety})",
        a.get("fixtures").unwrap_or(""),
        rows.len(),
        calib.alpha,
        alphas
    );
    let out = a
        .get("out")
        .map(PathBuf::from)
        .unwrap_or_else(|| dir.join(calib_name(&k)));
    let mut j = calib.to_json();
    j["alphas_tried"] = serde_json::json!(tried);
    std::fs::write(&out, serde_json::to_string_pretty(&j).unwrap()).expect("write calib");
    println!(
        "{}",
        serde_json::json!({"out": out.display().to_string(), "alpha": calib.alpha, "linears_smoothed": calib.smooth.len(), "max_err_op": calib.max_err_op,
        "max_err_ptr": calib.max_err_ptr, "tau_op": calib.tau_op, "tau_ptr": calib.tau_ptr, "questions": rows.len(), "tried": tried})
    );
}

/// GEMM microbenchmark at the engine's shapes (random data): int8 quantize + GEMM and f32 GEMM per kernel ISA
fn gemmbench(a: &Args) {
    use xerj_infer::kernels::{Epi, GemmScratch, PackedF32, PackedI8};
    let shapes: [(usize, usize, usize); 10] = [
        (430, 768, 256),
        (430, 256, 256),
        (430, 768, 256),
        (430, 256, 384),
        (430, 1166, 256),
        (430, 256, 1166),
        (35, 768, 256),
        (35, 256, 384),
        (8, 768, 256),
        (1, 768, 256),
    ];
    let k0 = kernels(a);
    let feats = CpuFeatures::detect();
    let mut isas = vec![k0.i8_isa];
    #[cfg(target_arch = "aarch64")]
    if feats.dotprod && k0.i8_isa != I8Isa::NeonDot {
        isas.push(I8Isa::NeonDot);
    }
    let mut seed = 0x12345678u64;
    let mut rnd = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        ((seed >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    };
    for (m, n, k) in shapes {
        let x: Vec<f32> = (0..m * k).map(|_| rnd()).collect();
        let w: Vec<f32> = (0..n * k).map(|_| rnd() * 0.1).collect();
        let b: Vec<f32> = (0..n).map(|_| rnd()).collect();
        let pw = PackedI8::quantize(&w, n, k, Some(&b));
        let pf = PackedF32::pack(&w, n, k, Some(&b));
        let mut y = vec![0f32; m * n];
        let mut s = GemmScratch::default();
        let reps = ((2e8 / (m * n * k) as f64) as usize).max(3);
        for &isa in &isas {
            let kern = Kernels {
                f32_isa: k0.f32_isa,
                i8_isa: isa,
                kleidi: false,
            };
            let (mut tq, mut tg) = (0f64, 0f64);
            for r in 0..reps + 2 {
                let t0 = Instant::now();
                kern.quantize(&x, m, k, None, &mut s);
                let t1 = Instant::now();
                kern.gemm_i8(&pw, &mut y, n, Epi::STORE, &s);
                let t2 = Instant::now();
                if r >= 2 {
                    tq += (t1 - t0).as_secs_f64();
                    tg += (t2 - t1).as_secs_f64();
                }
            }
            tq /= reps as f64;
            tg /= reps as f64;
            let ops = 2.0 * (m * n * k) as f64;
            println!("{:<22} M={m:4} N={n:4} K={k:4}  quant {:8.2} us  gemm {:8.2} us  total {:8.2} us  {:6.1} GOPS(gemm) {:6.1} GOPS(total)", isa.name(), tq * 1e6, tg * 1e6, (tq + tg) * 1e6, ops / tg / 1e9, ops / (tq + tg) / 1e9);
        }
        #[cfg(all(target_arch = "aarch64", xerj_kleidiai))]
        if xerj_infer::kernels::kleidi_available() {
            let kw = xerj_infer::kernels::kleidi::KaiRhs::pack(
                &pw.row_major(),
                n,
                k,
                &pw.scales[..n],
                pw.bias.as_deref(),
            );
            let mut ks = xerj_infer::kernels::kleidi::KaiScratch::default();
            let mut t = 0f64;
            for r in 0..reps + 2 {
                let t0 = Instant::now();
                xerj_infer::kernels::kleidi::gemm(&kw, &x, m, None, &mut y, n, Epi::STORE, &mut ks);
                if r >= 2 {
                    t += t0.elapsed().as_secs_f64();
                }
            }
            t /= reps as f64;
            println!(
                "{:<22} M={m:4} N={n:4} K={k:4}  quant+gemm {:8.2} us  {:6.1} GOPS(total)",
                "kleidiai i8mm 16x4",
                t * 1e6,
                2.0 * (m * n * k) as f64 / t / 1e9
            );
        }
        let mut tf = 0f64;
        for r in 0..reps.min(50) + 2 {
            let t0 = Instant::now();
            k0.gemm_f32(&x, m, &pf, &mut y, n, Epi::STORE, &mut s);
            if r >= 2 {
                tf += t0.elapsed().as_secs_f64();
            }
        }
        tf /= reps.min(50) as f64;
        println!(
            "{:<22} M={m:4} N={n:4} K={k:4}  gemm {:8.2} us  {:6.1} GFLOP/s",
            format!("f32 {}", k0.f32_isa.name()),
            tf * 1e6,
            2.0 * (m * n * k) as f64 / tf / 1e9
        );
    }
}

/// GEMM shapes of a model size for `gemmcmp`: "68m" (the stack-v5 direction executor: d 512, GeGLU 768, capacity 2,318;
/// rows of a first decision, a later decision and a history segment) or "m,n,k;m,n,k;..."
fn gemm_shapes(spec: &str) -> Vec<(usize, usize, usize)> {
    let spec = match spec {
        "68m" => "350,1536,512;350,512,512;350,1536,512;350,512,768;300,1536,512;300,512,768;280,2318,512;280,512,2318;24,1536,512;24,512,768;1,1536,512",
        s => s,
    };
    spec.split(';')
        .filter(|t| !t.trim().is_empty())
        .map(|t| {
            let v: Vec<usize> = t
                .split(',')
                .map(|x| x.trim().parse().expect("m,n,k"))
                .collect();
            assert_eq!(v.len(), 3, "shape {t}: m,n,k");
            (v[0], v[1], v[2])
        })
        .collect()
}

/// x86: the round-1 and round-2 AVX2 GEMM kernels on the same random data, alternating, with every output compared bit
/// for bit. `--shapes 68m|m,n,k;...` (default 68m), `--reps R` (default 7), `--epi store|add|bias` (default bias).
#[cfg(target_arch = "x86_64")]
fn gemmcmp(a: &Args) {
    use xerj_infer::kernels::{x86, AVec, Epi, GemmScratch, PackedF32, PackedI8};
    let shapes = gemm_shapes(a.get("shapes").unwrap_or("68m"));
    let reps = a.usize("reps", 7).max(1);
    let k0 = Kernels::detect();
    assert!(
        k0.f32_isa == F32Isa::Avx2 || k0.f32_isa == F32Isa::Avx512,
        "needs AVX2"
    );
    let mut seed = 0x2468_ace1u64;
    let mut rnd = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        ((seed >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    };
    let mut all_same = true;
    for (m, n, k) in shapes {
        let x: Vec<f32> = (0..m * k).map(|_| rnd()).collect();
        let w: Vec<f32> = (0..n * k).map(|_| rnd() * 0.1).collect();
        let b: Vec<f32> = (0..n).map(|_| rnd()).collect();
        let y0: Vec<f32> = (0..m * n).map(|_| rnd()).collect();
        let (bias, epi) = match a.get("epi").unwrap_or("bias") {
            "store" => (None, Epi::STORE),
            "add" => (None, Epi::ADD),
            _ => (Some(&b[..]), Epi::gated_add(0.7)),
        };
        let pw = PackedI8::quantize(&w, n, k, bias);
        let pf = PackedF32::pack(&w, n, k, bias);
        let mut s = GemmScratch::default();
        k0.quantize(&x, m, k, None, &mut s);
        let mut ap = AVec::new();
        let (mut ti, mut tf) = ([Vec::new(), Vec::new()], [Vec::new(), Vec::new()]);
        let mut yi = [y0.clone(), y0.clone()];
        let mut yf = [y0.clone(), y0.clone()];
        for rep in 0..reps + 1 {
            for v in 0..2 {
                let mut y = y0.clone();
                let t = Instant::now();
                // SAFETY: AVX2 + FMA checked by Kernels::detect; y holds m rows of n
                unsafe {
                    if v == 0 {
                        x86::gemm_i8_avx2_ptr(&s.q, &pw, y.as_mut_ptr(), n, epi)
                    } else {
                        x86::gemm_i8_avx2_v2_ptr(&s.q, &pw, y.as_mut_ptr(), n, epi)
                    }
                }
                let dt = t.elapsed().as_secs_f64() * 1e6;
                if rep > 0 {
                    ti[v].push(dt);
                }
                yi[v] = y;
                let mut y = y0.clone();
                let t = Instant::now();
                // SAFETY: as above
                unsafe {
                    if v == 0 {
                        x86::gemm_f32_avx2_ptr(&x, m, &pf, y.as_mut_ptr(), n, epi, &mut ap)
                    } else {
                        x86::gemm_f32_avx2_v2_ptr(&x, m, &pf, y.as_mut_ptr(), n, epi, &mut ap)
                    }
                }
                let dt = t.elapsed().as_secs_f64() * 1e6;
                if rep > 0 {
                    tf[v].push(dt);
                }
                yf[v] = y;
            }
        }
        let same = |p: &[f32], q: &[f32]| p.iter().zip(q).all(|(u, v)| u.to_bits() == v.to_bits());
        let (si, sf) = (same(&yi[0], &yi[1]), same(&yf[0], &yf[1]));
        all_same &= si && sf;
        let med = |v: &mut Vec<f64>| {
            v.sort_by(|p, q| p.partial_cmp(q).unwrap());
            v[v.len() / 2]
        };
        let macs = (m * n * k) as f64;
        let (i0, i1, f0, f1) = (
            med(&mut ti[0]),
            med(&mut ti[1]),
            med(&mut tf[0]),
            med(&mut tf[1]),
        );
        println!(
            "M={m:4} N={n:4} K={k:4}  int8 r1 {i0:9.1} us {:5.1} GMAC/s  r2 {i1:9.1} us {:5.1} GMAC/s  x{:.2} identical={si}  |  f32 r1 {f0:9.1} us {:5.1} GMAC/s  r2 {f1:9.1} us {:5.1} GMAC/s  x{:.2} identical={sf}",
            macs / i0 / 1e3, macs / i1 / 1e3, i0 / i1, macs / f0 / 1e3, macs / f1 / 1e3, f0 / f1
        );
    }
    println!("all identical: {all_same}");
    if !all_same {
        std::process::exit(1);
    }
}

/// x86: the round-1 attention (transposed keys) and the round-2 attention (blocked keys; values row-major, then
/// head-major) on the same random cache, at the shapes of a decision (`--t` cached + `--n` new positions, all queried:
/// the hop block) and of a prefix (`--n` new positions from 0), global and sliding window 64, alternating; outputs
/// compared bit for bit. Prints ns per (query, key) pair of the attended range.
#[cfg(target_arch = "x86_64")]
fn attncmp(a: &Args) {
    use xerj_infer::kernels::ops;
    use xerj_infer::model::LayerKv;
    let (heads, hd) = (8usize, 64usize);
    let d = heads * hd;
    let t0 = a.usize("t", 350);
    let n = a.usize("n", 300);
    let reps = a.usize("reps", 9).max(1);
    let mut seed = 0x9e37_79b9u64;
    let mut rnd = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        ((seed >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    };
    let total = t0 + n;
    let keys: Vec<f32> = (0..total * d).map(|_| rnd() * 1.5).collect();
    let vals: Vec<f32> = (0..total * d).map(|_| rnd()).collect();
    let mut kv1 = LayerKv::default();
    kv1.ensure(total, d);
    let blocked = |vhead: bool| {
        let mut kv = LayerKv::default();
        kv.blocked = true;
        kv.vhead = vhead;
        kv.heads = heads;
        kv.hd = hd;
        kv.ensure(total, d);
        kv
    };
    let (mut kv2, mut kv3) = (blocked(false), blocked(true));
    for p in 0..total {
        kv1.put(p, &keys[p * d..(p + 1) * d], &vals[p * d..(p + 1) * d]);
        kv2.put(p, &keys[p * d..(p + 1) * d], &vals[p * d..(p + 1) * d]);
        kv3.put(p, &keys[p * d..(p + 1) * d], &vals[p * d..(p + 1) * d]);
    }
    kv1.transpose_keys(0, total, d);
    let mut all_same = true;
    for (label, pos) in [
        ("decision", (t0..total).collect::<Vec<usize>>()),
        ("prefix", (0..n).collect::<Vec<usize>>()),
    ] {
        let q: Vec<f32> = (0..pos.len() * d).map(|_| rnd()).collect();
        for window in [None, Some(64usize)] {
            let pairs: usize = pos
                .iter()
                .map(|&p| p + 1 - window.map_or(0, |w| (p + 1).saturating_sub(w)))
                .sum::<usize>()
                * heads;
            let (mut s1, mut s2, mut s3) = (Vec::new(), Vec::new(), Vec::new());
            let mut o1 = vec![0f32; pos.len() * d];
            let (mut o2, mut o3) = (o1.clone(), o1.clone());
            let (mut t1, mut t2, mut t3) = (Vec::new(), Vec::new(), Vec::new());
            for _ in 0..reps {
                let t = Instant::now();
                ops::self_attention(&q, &pos, &kv1, heads, hd, window, &mut s1, &mut o1);
                t1.push(t.elapsed().as_secs_f64());
                let t = Instant::now();
                ops::self_attention(&q, &pos, &kv2, heads, hd, window, &mut s2, &mut o2);
                t2.push(t.elapsed().as_secs_f64());
                let t = Instant::now();
                ops::self_attention(&q, &pos, &kv3, heads, hd, window, &mut s3, &mut o3);
                t3.push(t.elapsed().as_secs_f64());
            }
            let same = o1.iter().zip(&o2).all(|(x, y)| x.to_bits() == y.to_bits())
                && o1.iter().zip(&o3).all(|(x, y)| x.to_bits() == y.to_bits());
            all_same &= same;
            for tv in [&mut t1, &mut t2, &mut t3] {
                tv.sort_by(|x, y| x.partial_cmp(y).unwrap());
            }
            let (m1, m2, m3) = (t1[reps / 2], t2[reps / 2], t3[reps / 2]);
            let ns = |m: f64| m * 1e9 / pairs as f64;
            println!(
                "{label:<8} window={:<4} queries={:4} pairs={:9}  r1 {:8.3} ms {:5.2} ns/pair  r2 {:8.3} ms {:5.2} ns/pair  r2 head-major values {:8.3} ms {:5.2} ns/pair  x{:.2} / x{:.2}  identical={same}",
                window.map_or("-".to_string(), |w| w.to_string()),
                pos.len(),
                pairs,
                m1 * 1e3,
                ns(m1),
                m2 * 1e3,
                ns(m2),
                m3 * 1e3,
                ns(m3),
                m1 / m2,
                m2 / m3
            );
        }
    }
    println!("all identical: {all_same}");
    if !all_same {
        std::process::exit(1);
    }
}

#[cfg(not(target_arch = "x86_64"))]
fn attncmp(_a: &Args) {
    eprintln!("attncmp: x86_64 only");
    std::process::exit(2);
}

#[cfg(not(target_arch = "x86_64"))]
fn gemmcmp(_a: &Args) {
    eprintln!("gemmcmp: x86_64 only");
    std::process::exit(2);
}

/// the bridge between the two INT8 implementations: every 2-D tensor of the f32 export `--model`, rounded by the engine's
/// `w8_quantize_row` and dequantized, must equal the same tensor of `--fakequant` (an export of tools/int8_fakequant.py's
/// checkpoint), and every other tensor must be bit-identical. A rounded -0.0 equals 0.0: it changes no product.
fn w8check(a: &Args) {
    use xerj_infer::kernels::w8_quantize_row;
    use xerj_infer::safetensors::SafeTensors;
    let open = |k: &str| {
        let p =
            PathBuf::from(a.get(k).unwrap_or_else(|| panic!("--{k}"))).join("model.safetensors");
        SafeTensors::open(&p).expect("open")
    };
    let (m, f) = (open("model"), open("fakequant"));
    let mut names: Vec<&String> = m.tensors.keys().collect();
    names.sort();
    let emb = "core.backbone.model.embeddings.tok_embeddings.weight";
    let (mut quantized, mut copied, mut params, mut bad, mut signed_zero) =
        (0, 0, 0usize, 0usize, 0usize);
    for name in names {
        let (w, shape) = m.f32(name).expect("tensor");
        let (fq, fshape) = f.f32(name).expect("fakequant tensor");
        assert_eq!(shape, fshape, "{name}");
        if shape.len() == 2 && name != emb {
            let k = shape[1];
            let mut q = vec![0i8; k];
            for (r, fr) in w.chunks_exact(k).zip(fq.chunks_exact(k)) {
                let scale = w8_quantize_row(r, &mut q);
                for (qv, fv) in q.iter().zip(fr) {
                    let v = *qv as f32 * scale;
                    bad += (v != *fv) as usize;
                    signed_zero += (v == 0.0 && v.to_bits() != fv.to_bits()) as usize;
                }
            }
            quantized += 1;
            params += w.len();
        } else {
            bad += w
                .iter()
                .zip(fq.iter())
                .filter(|(x, y)| x.to_bits() != y.to_bits())
                .count();
            copied += 1;
        }
    }
    let res = serde_json::json!({"model": a.get("model"), "fakequant": a.get("fakequant"), "tensors_quantized": quantized,
        "tensors_identical_rule": copied, "params_quantized": params, "values_different": bad, "signed_zero_only_differences": signed_zero});
    println!("{}", serde_json::to_string_pretty(&res).unwrap());
    if let Some(out) = a.get("json") {
        std::fs::write(out, serde_json::to_string_pretty(&res).unwrap()).expect("write json");
    }
    if bad > 0 {
        std::process::exit(1);
    }
}

fn main() {
    let a = parse_args();
    match a.cmd.as_str() {
        "info" => info(&a),
        "parity" => parity(&a),
        "bench" => bench(&a),
        "calibrate" => calibrate(&a),
        "w8check" => w8check(&a),
        "gemmbench" => gemmbench(&a),
        "gemmcmp" => gemmcmp(&a),
        "attncmp" => attncmp(&a),
        _ => {
            eprintln!("usage: xerj-infer info|parity|bench --model DIR [--fixtures F.jsonl,...] [--precision f32|w8|int8] ...");
            std::process::exit(2);
        }
    }
}
