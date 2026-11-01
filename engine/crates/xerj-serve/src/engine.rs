//! One question end to end: world -> Memory -> (tolerant) link -> exact K=16 manifest -> model decisions (int8 + guard)
//! on the exact executor -> answer record -> verify rule. Mirrors rollout.answer_item (frozen stack_v2_r1; dev_v5 for the
//! stack-v5 direction executor: v5 manifest with inverse candidates, EpisodeV5) as the XERJ Lab runs it: item id "lab/q/" +
//! sha256(question)[:16] unless given, oracle-free label-free manifest, the model's own free-running program.

use crate::adapter::{candidate_fields, kind_name, ExecAdapter};
use crate::tolerant::{tolerant_link, LinkInfo};
use crate::verify_bound::verify_v5_with;
pub use crate::verify_bound::VerifyRule;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;
use xerj_exec::{verify, Episode, LoopGuard, Protocol, VerifyMode};
use xerj_index::direction::{max_inverse, RecScope, RelPref};
use xerj_index::scope::{QuestionScope, ScopeMode};
use xerj_index::{rel_phrase, Memory, MemoryConfig, QBuf};
use xerj_infer::config::{Arm, StackProtocol};
use xerj_infer::episode::{run_episode_guarded, Guard};
use xerj_infer::{Engine, Int8Calib, Kernels, LoadOptions, Model, Precision, TextEncoder};
use xerj_infer::{Session, SessionBufs};
use xerj_store::value::render_value;
use xerj_store::{RecId, Rev, Snapshot, World};

pub struct ModelSlot {
    pub name: String,
    pub dir: PathBuf,
    pub engine: Engine,
    pub guard: Option<Guard>,
    pub protocol: Protocol,
    pub k: usize,
    pub int8_mode: &'static str,
    pub load_ms: f64,
    pub checkpoint: Value,
    /// answer-speed options the slot was loaded with (default: one thread, no session reuse)
    pub speed: SpeedOptions,
    pool: Option<std::sync::Arc<xerj_infer::pool::Pool>>,
    /// buffers of finished questions (KV caches, activations, tokenized segments) when `speed.reuse_sessions`
    spare: std::sync::Mutex<Vec<SessionBufs>>,
    /// a question holds the slot's thread pool (see [`ModelSlot::session`])
    pool_busy: AtomicBool,
    /// `SpeedOptions::queue`: the turn of the one question that computes (the others wait for it)
    turn: std::sync::Mutex<()>,
}

/// Answer-speed options of a model slot. The default is the original behaviour: one thread per question and a new session
/// (KV caches, buffers, tokenized-segment cache) for every question. Neither option changes a decision: the threads split
/// each GEMM by output channels and the attention by heads (every value computed exactly as single-threaded), and a reused
/// session is reset by `Session::begin` exactly as the parity tools reuse one session for all their questions.
#[derive(Debug, Clone, Copy, Default)]
pub struct SpeedOptions {
    /// threads per question (0 or 1 = single-threaded); the weights are split into this many output-channel shards. The
    /// threads go to one question at a time; questions that arrive meanwhile run single-threaded (see
    /// [`ModelSlot::session`]).
    pub threads: usize,
    /// keep the buffers of finished questions for the next ones (at most `MAX_SPARE` sets)
    pub reuse_sessions: bool,
    /// with `threads` > 1: concurrent questions take turns on all the threads, one question computing at a time while
    /// the others wait for their turn, instead of the default policy (the pool to one question, the others
    /// single-threaded beside it on the same cores). Off by default; decisions are identical either way.
    pub queue: bool,
}

/// spare session buffer sets kept by a slot (one per concurrently answered question, up to this many)
pub const MAX_SPARE: usize = 4;

/// One question's claim on the slot's thread pool ([`ModelSlot::session`]). Dropping it releases the pool when this
/// question held it, also when the episode panics, so a failed question never keeps the pool.
pub struct QuestionGuard<'a> {
    pool_busy: &'a AtomicBool,
    /// the question got the slot's thread pool (no other question held it when this one started)
    pub threaded: bool,
    /// `SpeedOptions::queue`: this question's turn, released after the pool (field order)
    _turn: Option<std::sync::MutexGuard<'a, ()>>,
}

impl QuestionGuard<'_> {
    /// start a question; it is `threaded` when a pool exists and no other question holds it
    fn begin(pool_busy: &AtomicBool, has_pool: bool) -> QuestionGuard<'_> {
        let threaded = has_pool
            && pool_busy
                .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
                .is_ok();
        QuestionGuard {
            pool_busy,
            threaded,
            _turn: None,
        }
    }

    /// start a question after waiting for its turn (`SpeedOptions::queue`): only the question holding the turn computes,
    /// so it always gets the pool
    fn begin_turn<'a>(
        pool_busy: &'a AtomicBool,
        has_pool: bool,
        turn: &'a std::sync::Mutex<()>,
    ) -> QuestionGuard<'a> {
        let t = turn.lock().unwrap_or_else(|e| e.into_inner());
        let mut q = QuestionGuard::begin(pool_busy, has_pool);
        q._turn = Some(t);
        q
    }
}

impl Drop for QuestionGuard<'_> {
    fn drop(&mut self) {
        if self.threaded {
            self.pool_busy.store(false, Ordering::SeqCst);
        }
    }
}

impl ModelSlot {
    #[cfg(feature = "tokenizer")]
    pub fn load(
        name: &str,
        dir: &Path,
        precision: Precision,
        int8_mode: &str,
    ) -> Result<ModelSlot, String> {
        Self::load_inner(name, dir, precision, int8_mode, |opt| {
            Engine::load_with(dir, opt).map_err(|e| format!("{name}: {e}"))
        })
    }

    /// As `load`, with a caller-supplied text encoder (a host that already ships a tokenizer, e.g. the XERJ engine, builds
    /// this crate without the `tokenizer` feature). The caller checks that its tokenizer is the export's (sha256 in config.json).
    pub fn load_with_encoder(
        name: &str,
        dir: &Path,
        precision: Precision,
        int8_mode: &str,
        tok: Box<dyn TextEncoder>,
    ) -> Result<ModelSlot, String> {
        Self::load_inner(name, dir, precision, int8_mode, move |opt| {
            let model = Model::load_with(dir, opt).map_err(|e| format!("{name}: {e}"))?;
            Ok(Engine {
                model,
                tok,
                tokenizer_ms: 0.0,
            })
        })
    }

    /// `load_with_encoder` with answer-speed options (see [`SpeedOptions`])
    pub fn load_with_encoder_speed(
        name: &str,
        dir: &Path,
        precision: Precision,
        int8_mode: &str,
        tok: Box<dyn TextEncoder>,
        speed: SpeedOptions,
    ) -> Result<ModelSlot, String> {
        Self::load_inner_speed(name, dir, precision, int8_mode, speed, move |opt| {
            let model = Model::load_with(dir, opt).map_err(|e| format!("{name}: {e}"))?;
            Ok(Engine {
                model,
                tok,
                tokenizer_ms: 0.0,
            })
        })
    }

    /// A session for one question and its claim on the thread pool (keep the guard until the question is done): reused
    /// buffers when `speed.reuse_sessions` and a spare set exists. With `speed.threads > 1` the slot's pool goes to one
    /// question at a time: a question gets it when no other question holds it, and questions that start meanwhile run
    /// single-threaded side by side instead of queueing their parallel loops behind the one shared pool (with the pool
    /// shared by 4-8 concurrent questions the throughput fell below the single-threaded default: 1.9-2.0 vs 2.2-2.5
    /// questions per second on 4 CPUs). Decisions are identical either way: a sharded linear computes every output value
    /// with the same kernel with or without a pool, and the attention and element-wise loops run the same per-head /
    /// per-row code.
    ///
    /// With `speed.queue` (off by default) a question first waits for its turn: one question computes at a time, on all
    /// the threads, and the others wait instead of running single-threaded beside it on the same cores.
    pub fn session(&self) -> (Session<'_>, QuestionGuard<'_>) {
        let q = if self.speed.queue && self.pool.is_some() {
            QuestionGuard::begin_turn(&self.pool_busy, true, &self.turn)
        } else {
            QuestionGuard::begin(&self.pool_busy, self.pool.is_some())
        };
        let spare = if self.speed.reuse_sessions {
            self.spare.lock().ok().and_then(|mut v| v.pop())
        } else {
            None
        };
        let m = &self.engine.model;
        let tok = self.engine.tok.as_ref();
        let mut s = match spare {
            Some(b) => Session::from_bufs(m, tok, b),
            None => Session::new(m, tok),
        };
        s.set_pool(if q.threaded { self.pool.clone() } else { None });
        (s, q)
    }

    /// hand a finished question's session back (kept for the next question when `speed.reuse_sessions`)
    pub fn recycle(&self, s: Session<'_>) {
        if self.speed.reuse_sessions {
            if let Ok(mut v) = self.spare.lock() {
                if v.len() < MAX_SPARE {
                    v.push(s.into_bufs());
                }
            }
        }
    }

    fn load_inner(
        name: &str,
        dir: &Path,
        precision: Precision,
        int8_mode: &str,
        build: impl FnOnce(&LoadOptions) -> Result<Engine, String>,
    ) -> Result<ModelSlot, String> {
        Self::load_inner_speed(
            name,
            dir,
            precision,
            int8_mode,
            SpeedOptions::default(),
            build,
        )
    }

    fn load_inner_speed(
        name: &str,
        dir: &Path,
        precision: Precision,
        int8_mode: &str,
        speed: SpeedOptions,
        build: impl FnOnce(&LoadOptions) -> Result<Engine, String>,
    ) -> Result<ModelSlot, String> {
        let t0 = Instant::now();
        let k = Kernels::detect();
        let mut opt = LoadOptions::new(precision, k);
        let threads = speed.threads.max(1);
        opt.shards = threads;
        let mut guard = None;
        let mut mode = match precision {
            Precision::F32 => "f32",
            Precision::W8 => "w8",
            Precision::Int8 => "raw",
        };
        if precision == Precision::Int8 && int8_mode != "raw" {
            match Int8Calib::load(&dir.join("int8_calib.json")) {
                Ok(c) => {
                    if int8_mode == "guard" {
                        opt.keep_f32 = true;
                        guard = Some(Guard {
                            tau_op: c.tau_op,
                            tau_ptr: c.tau_ptr,
                        });
                        mode = "guard";
                    } else {
                        mode = "smooth";
                    }
                    opt.calib = Some(c);
                }
                Err(e) => eprintln!("[xerj-ask] {name}: no int8 calibration ({e}); raw int8"),
            }
        }
        let engine = build(&opt)?;
        let cfg = &engine.model.cfg;
        let protocol = match cfg.stack {
            StackProtocol::V1 => Protocol::V1,
            StackProtocol::V2 => Protocol::V2,
            StackProtocol::V5 => Protocol::V5,
        };
        let k = cfg.k;
        let checkpoint = std::fs::read_to_string(dir.join("config.json"))
            .ok()
            .and_then(|s| serde_json::from_str::<Value>(&s).ok())
            .and_then(|v| v.get("checkpoint").cloned())
            .unwrap_or(Value::Null);
        Ok(ModelSlot {
            name: name.to_string(),
            dir: dir.to_path_buf(),
            engine,
            guard,
            protocol,
            k,
            int8_mode: mode,
            load_ms: t0.elapsed().as_secs_f64() * 1e3,
            checkpoint,
            speed,
            pool: (threads > 1).then(|| std::sync::Arc::new(xerj_infer::pool::Pool::new(threads))),
            spare: std::sync::Mutex::new(Vec::new()),
            pool_busy: AtomicBool::new(false),
            turn: std::sync::Mutex::new(()),
        })
    }

    pub fn info(&self) -> Value {
        let cfg = &self.engine.model.cfg;
        let mut v = json!({"name": self.name, "dir": self.dir.display().to_string(),
               "arm": match cfg.arm { Arm::Rrt => "rrt", Arm::ExternalExecutor => "external_executor" },
               "stack": cfg.stack.name(),
               "k": self.k, "int8_mode": self.int8_mode, "load_ms": round1(self.load_ms), "checkpoint": self.checkpoint});
        // the answer-speed options only when set: the default response stays exactly as before
        if self.speed.threads > 1 || self.speed.reuse_sessions {
            v["threads"] = json!(self.speed.threads.max(1));
            v["reuse_sessions"] = json!(self.speed.reuse_sessions);
        }
        if self.speed.queue && self.pool.is_some() {
            v["queue"] = json!(true);
        }
        v
    }
}

pub fn round1(x: f64) -> f64 {
    (x * 10.0).round() / 10.0
}
pub fn round3(x: f64) -> f64 {
    (x * 1000.0).round() / 1000.0
}

pub fn lab_item_id(question: &str) -> String {
    let d = xerj_infer::sha256::digest(question.as_bytes());
    format!("lab/q/{}", &xerj_infer::sha256::hex(&d)[..16])
}

/// A world (memory-world-v1) with the edits `apply` (indices; None = all) applied, as the XERJ Lab builds it.
pub struct BuiltWorld {
    pub mem: Memory,
    pub base: Rev,
    pub final_: Rev,
    pub build_ms: f64,
}

pub fn build_world(world: &Value, apply: Option<&[usize]>) -> Result<BuiltWorld, String> {
    let t0 = Instant::now();
    let mut w = world.clone();
    if let (Some(idx), Some(edits)) = (apply, world.get("edits").and_then(|e| e.as_array())) {
        let mut idx: Vec<usize> = idx.iter().copied().filter(|&i| i < edits.len()).collect();
        idx.sort_unstable();
        idx.dedup();
        w["edits"] = Value::Array(idx.iter().map(|&i| edits[i].clone()).collect());
    }
    if let Some(o) = w.as_object_mut() {
        o.remove("items");
    }
    let s = serde_json::to_string(&w).map_err(|e| e.to_string())?;
    let parsed = World::parse(&s).map_err(|e| format!("world: {e}"))?;
    let cfg = MemoryConfig {
        bm25_records: false,
        bm25_documents: false,
        ..MemoryConfig::default()
    };
    let mut mem = Memory::new(cfg);
    let revs = mem
        .ingest_world(&parsed)
        .map_err(|e| format!("world rejected by the store: {e:?}"))?;
    Ok(BuiltWorld {
        mem,
        base: revs.base,
        final_: revs.final_,
        build_ms: t0.elapsed().as_secs_f64() * 1e3,
    })
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LinkerMode {
    Exact,
    Tolerant,
    /// tolerant names + relation-word typo correction
    TolerantWords,
}

impl LinkerMode {
    pub fn parse(s: &str) -> LinkerMode {
        match s {
            "tolerant" => LinkerMode::Tolerant,
            "tolerant_words" => LinkerMode::TolerantWords,
            _ => LinkerMode::Exact,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            LinkerMode::Exact => "exact",
            LinkerMode::Tolerant => "tolerant",
            LinkerMode::TolerantWords => "tolerant_words",
        }
    }
}

fn link_json(li: &LinkInfo) -> Value {
    json!({"mode": li.mode, "exact": li.exact_subjects, "rewritten": li.rewritten, "capitalized": li.capitalized,
           "matches": li.matches.iter().map(|m| json!({"surface": m.surface, "subject": m.subject, "distance": m.distance})).collect::<Vec<_>>(),
           "ambiguous": li.ambiguous.iter().map(|(s, c)| json!({"surface": s, "candidates": c})).collect::<Vec<_>>(),
           "word_fixes": li.word_fixes.iter().map(|(a, b, d)| json!({"surface": a, "to": b, "distance": d})).collect::<Vec<_>>()})
}

/// Answer one question with one model on a snapshot of a built world.
pub fn answer(
    slot: &ModelSlot,
    mem: &Memory,
    snap: Snapshot<'_>,
    item_id: &str,
    question: &str,
    linker: LinkerMode,
    with_cands: bool,
) -> Value {
    answer_observed(slot, mem, snap, item_id, question, linker, with_cands, None)
}

/// [`answer`] with an observer called on every model decision with the session (its decision input ids) and the decision
/// (op, pointer, logits): the E2 frozen-model parity check records them against the golden PyTorch fixtures.
#[allow(clippy::too_many_arguments)]
pub fn answer_observed(
    slot: &ModelSlot,
    mem: &Memory,
    snap: Snapshot<'_>,
    item_id: &str,
    question: &str,
    linker: LinkerMode,
    with_cands: bool,
    observe: Option<&mut dyn FnMut(&xerj_infer::Session, &xerj_infer::episode::DecisionEvent)>,
) -> Value {
    match answer_stoppable(
        slot, mem, snap, item_id, question, linker, with_cands, observe, None,
    ) {
        Ok(v) => v,
        Err(s) => unreachable!("no stop flag, yet stopped after {} decisions", s.decisions),
    }
}

/// Why [`answer_stoppable`] has no answer: its host asked it to stop (a deadline passed, or the client went away) before
/// the compiler finished the program.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Stopped {
    /// model decisions taken before the stop
    pub decisions: usize,
}

/// [`answer_observed`] that its host can stop between two model decisions. Once `stop` is set, the episode ends before its
/// next decision (a decision in progress completes) and the result is `Err(Stopped)`: an episode cut short has no answer,
/// not even an UNKNOWN, so nothing partial can be served. A flag set before the start skips the question entirely. With
/// `stop` = `None`, or a flag that stays unset, every decision and the answer record are [`answer_observed`]'s: the flag
/// is read only between decisions.
#[allow(clippy::too_many_arguments)]
pub fn answer_stoppable(
    slot: &ModelSlot,
    mem: &Memory,
    snap: Snapshot<'_>,
    item_id: &str,
    question: &str,
    linker: LinkerMode,
    with_cands: bool,
    observe: Option<&mut dyn FnMut(&xerj_infer::Session, &xerj_infer::episode::DecisionEvent)>,
    stop: Option<Arc<AtomicBool>>,
) -> Result<Value, Stopped> {
    answer_stoppable_opts(
        slot,
        mem,
        snap,
        item_id,
        question,
        linker,
        with_cands,
        observe,
        stop,
        &AnswerOpts::default(),
    )
}

/// EXTENSION options of one answer (stack-v5 manifests, record scope, loop guard and verify rule). The default is the
/// reference pipeline: [`answer_stoppable`] passes it, and with it [`answer_stoppable_opts`] runs the very same calls.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct AnswerOpts {
    /// the linked subjects (subject syms of `mem.store.pools.subjects`) instead of the exact linker's on the question;
    /// None = the exact linker (a shared-store linker of the host decides them, e.g. xerj-facts `casefold_v3`)
    pub ents: Option<Vec<u32>>,
    /// question-aware choice of the records a manifest keeps where a cap binds, in the base and every hop manifest
    /// (`xerj_index::direction::RelPref`); false = the reference choice
    pub relation_aware: bool,
    /// the records this question's manifests may offer (`xerj_index::scope`: Filter / Prefer = the base and every hop
    /// manifest, Coherent = the hop manifests after the first read); None = every record (the reference)
    pub scope: Option<QuestionScope>,
    /// what a LOOKUP / CHAIN does that reads a record the program already read (`xerj_exec::LoopGuard`); Off = the
    /// reference
    pub loop_guard: LoopGuard,
    /// the verify rule that judges a stack-v5 ANSWER ([`VerifyRule`]); `Served` (the default) = `verify_v5`, the rule
    /// `answer()` has always served. It runs after the episode, so it changes no decision, program or raw answer: only
    /// `verify` and `final_status` / `final_answer`. Stack-v1 / v2 slots keep their own rule.
    pub verify: VerifyRule,
}

impl AnswerOpts {
    /// the reference manifests and episode (no manifest or episode extension in use: `ents`, `relation_aware`, `scope`,
    /// `loop_guard`). `verify` is a separate choice: it is read after the episode and never changes a manifest or a
    /// decision.
    pub fn is_reference(&self) -> bool {
        self.ents.is_none()
            && !self.relation_aware
            && self.scope.is_none()
            && self.loop_guard == LoopGuard::Off
    }
}

/// [`answer_stoppable`] with [`AnswerOpts`] (EXTENSION, opt-in; only the stack-v5 protocol reads them). With
/// `AnswerOpts::default()` every call, decision and output field is [`answer_stoppable`]'s.
#[allow(clippy::too_many_arguments)]
pub fn answer_stoppable_opts(
    slot: &ModelSlot,
    mem: &Memory,
    snap: Snapshot<'_>,
    item_id: &str,
    question: &str,
    linker: LinkerMode,
    with_cands: bool,
    observe: Option<&mut dyn FnMut(&xerj_infer::Session, &xerj_infer::episode::DecisionEvent)>,
    stop: Option<Arc<AtomicBool>>,
    opts: &AnswerOpts,
) -> Result<Value, Stopped> {
    if stop.as_ref().is_some_and(|s| s.load(Ordering::Acquire)) {
        return Err(Stopped { decisions: 0 });
    }
    let t0 = Instant::now();
    let mut buf = QBuf::default();
    let (q, link) = match linker {
        LinkerMode::Exact => (question.to_string(), None),
        LinkerMode::Tolerant | LinkerMode::TolerantWords => {
            let mut li = tolerant_link(mem, &snap, question, &mut buf);
            if linker == LinkerMode::TolerantWords {
                let protect: Vec<String> = li
                    .exact_subjects
                    .iter()
                    .cloned()
                    .chain(li.matches.iter().map(|m| m.subject.clone()))
                    .collect();
                let tok = slot.engine.tok.as_ref();
                let known = |w: &str| {
                    tok.encode(&format!(" {w}"))
                        .map(|ids| ids.len() == 1)
                        .unwrap_or(true)
                };
                let (q2, fixes) = crate::tolerant::correct_relation_words(
                    mem,
                    &snap,
                    &li.rewritten,
                    &protect,
                    &known,
                );
                if !fixes.is_empty() {
                    li.rewritten = q2;
                    li.word_fixes = fixes;
                    if li.mode == "exact" {
                        li.mode = "words";
                    }
                }
            }
            (li.rewritten.clone(), Some(li))
        }
    };
    let t_link = t0.elapsed().as_secs_f64() * 1e6;
    // stack-v5: the v5 manifest (linked subjects' records + inverse candidates of linked values, MAX_INVERSE from
    // XERJ_R_MAX_INVERSE or 8), one inverse flag per slot
    let v5_ext = slot.protocol == Protocol::V5 && !opts.is_reference();
    let pref = (v5_ext && opts.relation_aware).then(|| Arc::new(RelPref::from_question(&q)));
    let (cands, inverse): (Vec<RecId>, Vec<bool>) = if slot.protocol == Protocol::V5 {
        let m = if let Some((sc, set)) = opts.scope.as_ref().and_then(|sc| match &sc.mode {
            ScopeMode::Filter(s) | ScopeMode::Prefer(s) if v5_ext => Some((sc, s)),
            _ => None,
        }) {
            // EXTENSION (opt-in): the base manifest over the question's scope
            let keep = |r: RecId| sc.map.in_any(r, set);
            let rs = match sc.mode {
                ScopeMode::Prefer(_) => RecScope::Prefer(&keep),
                _ => RecScope::Filter(&keep),
            };
            mem.candidate_manifest_v5_scope(
                &snap,
                item_id,
                &q,
                slot.k,
                max_inverse(),
                opts.ents.as_deref(),
                pref.as_deref(),
                rs,
                &mut buf,
            )
        } else if v5_ext {
            mem.candidate_manifest_v5_opts(
                &snap,
                item_id,
                &q,
                slot.k,
                max_inverse(),
                opts.ents.as_deref(),
                pref.as_deref(),
                &mut buf,
            )
        } else {
            mem.candidate_manifest_v5(&snap, item_id, &q, slot.k, &mut buf)
        };
        (m.cands, m.inverse)
    } else {
        (
            mem.candidate_manifest(&snap, item_id, &q, slot.k, &mut buf),
            Vec::new(),
        )
    };
    let is_inv = |i: usize| inverse.get(i).copied().unwrap_or(false);
    let fields: Vec<_> = cands
        .iter()
        .enumerate()
        .map(|(i, &r)| candidate_fields(&mem.store, r).with_inverse(is_inv(i)))
        .collect();
    let t_manifest = t0.elapsed().as_secs_f64() * 1e6 - t_link;
    let mut ep = if slot.protocol == Protocol::V5 {
        Episode::new_v5(mem, snap, cands.clone(), inverse.clone(), item_id, slot.k)
    } else {
        Episode::new(mem, snap, slot.protocol, cands.clone(), item_id, slot.k)
    };
    if let Some(p) = &pref {
        ep.rel_pref = Some(p.clone());
    }
    if v5_ext {
        ep.scope = opts.scope.clone().map(Arc::new);
        ep.loop_guard = opts.loop_guard;
    }
    let mut ad = ExecAdapter::new(ep);
    let stop_flag = stop.clone();
    if let Some(stop) = stop {
        ad = ad.with_stop(stop);
    }
    let t1 = Instant::now();
    let (mut sess, in_flight) = slot.session();
    // a question that waited for its turn (`SpeedOptions::queue`) and was given up meanwhile does not start
    if slot.speed.queue && stop_flag.is_some_and(|s| s.load(Ordering::Acquire)) {
        slot.recycle(sess);
        drop(in_flight);
        return Err(Stopped { decisions: 0 });
    }
    let run = run_episode_guarded(&mut sess, &q, &fields, &mut ad, observe, slot.guard);
    if ad.stopped() {
        return Err(Stopped {
            decisions: run.as_ref().map_or(0, |s| s.decisions),
        });
    }
    let t_episode = t1.elapsed().as_secs_f64() * 1e6;
    slot.recycle(sess);
    drop(in_flight);
    let store = &mem.store;
    let fid = |r: RecId| store.fact_id(r).to_string();
    let (stats, err) = match run {
        Ok(s) => (Some(s), None),
        Err(e) => (None, Some(e.to_string())),
    };
    let ans = ad.ep.answer_or_unfinished();
    let t2 = Instant::now();
    let (v, rule) = match slot.protocol {
        // `VerifyRule::Served` (the default) is `verify_v5` with the very same arguments as before
        Protocol::V5 => (
            verify_v5_with(opts.verify, mem, &snap, &q, &cands, &ans, &mut buf),
            opts.verify.rule(),
        ),
        Protocol::V2 => (
            verify_chain_aware(mem, &q, &ans, &mut buf),
            "chain-aware (v2)",
        ),
        Protocol::V1 => (
            verify(mem, &snap, &q, &ans, VerifyMode::Substring, &mut buf),
            "verify_rows (v1 reference)",
        ),
    };
    // the opt-in decline policy reads the raw answer as it was measured; `verify` above is reported only
    let decline = (opts.verify == VerifyRule::DeclineV2).then(|| {
        crate::decline_v2::judge(
            &q, &ad.ep, &ans, mem, snap, &cands, &inverse, item_id, slot.k,
        )
    });
    let t_verify = t2.elapsed().as_secs_f64() * 1e6;
    let trace: Vec<Value> = ans
        .execution_trace
        .iter()
        .map(|t| {
            let f = match (t.op.as_str(), t.pointer) {
                ("LOOKUP", Some(p)) if p >= 0 && (p as usize) < cands.len() => Value::String(fid(cands[p as usize])),
                ("CHAIN", Some(p)) => t.hop_candidate_ids.as_ref().and_then(|h| h.get(p.max(0) as usize)).map(|&r| Value::String(fid(r))).unwrap_or(Value::Null),
                _ => Value::Null,
            };
            json!({"step": t.step, "op": t.op, "pointer": t.pointer, "fact_id": f, "result": t.result, "illegal": t.illegal, "inverse": t.inverse})
        })
        .collect();
    let total_us = t0.elapsed().as_secs_f64() * 1e6;
    let model_us = (t_episode - ad.exec_us).max(0.0);
    let mut out = json!({
        "model": slot.name, "linker": linker.name(), "question": question, "question_used": q, "item_id": item_id,
        "status": ans.status, "answer": ans.answer, "evidence_ids": ans.evidence_ids.iter().map(|&r| fid(r)).collect::<Vec<_>>(),
        "error": ans.error, "trace": trace,
        // stack-v5: [op, pointer, "in" | null] (dev_v5 actions carry "direction": "in" on inverse steps)
        "program": ans.program.iter().map(|a| if slot.protocol == Protocol::V5 { json!([a.op, a.pointer, a.inverse.then_some("in")]) } else { json!([a.op, a.pointer]) }).collect::<Vec<_>>(),
        "verify": {"applies": ans.status == "ANSWER", "veto": v.vetoed, "reason": v.reason, "rule": rule},
        "final_status": if v.vetoed { "UNKNOWN" } else { ans.status },
        "final_answer": if v.vetoed { None } else { ans.answer.clone() },
        "decisions": stats.as_ref().map(|s| s.decisions), "fallbacks": stats.as_ref().map(|s| s.fallbacks),
        "computed_tokens": stats.as_ref().map(|s| s.computed_tokens), "input_tokens": stats.as_ref().map(|s| s.input_tokens),
        "engine_error": err,
        "timings_us": {"link": round1(t_link), "manifest": round1(t_manifest), "model": round1(model_us), "executor": round1(ad.exec_us),
                        "verify": round1(t_verify), "total": round1(total_us)},
        "latency_ms": round3(total_us / 1000.0),
    });
    if let Some(li) = link {
        out["link"] = link_json(&li);
    }
    if slot.protocol == Protocol::V5 && opts.verify != VerifyRule::Served {
        // only when an opt-in verify rule judged the answer: the default response keeps its fields
        out["verify"]["option"] = json!(opts.verify.name());
    }
    if let Some(d) = &decline {
        // only under `verify: decline_v2`: the default response keeps its fields
        let raw: Vec<String> = ans.evidence_ids.iter().map(|&r| fid(r)).collect();
        let j = d.to_json(ans.status, &raw);
        out["final_status"] = j["final_status"].clone();
        out["final_answer"] = j["final_answer"].clone();
        out["final_evidence_ids"] = j["final_evidence_ids"].clone();
        out["reason_code"] = j["reason_code"].clone();
        if let Some(r) = j.get("repair") {
            out["repair"] = r.clone();
        }
        out["decline"] = j;
    }
    if v5_ext {
        // only when an extension is in use: the default response keeps its fields
        out["answer_opts"] = json!({"relation_aware": opts.relation_aware,
            "subjects": opts.ents.as_ref().map(|e| e.iter().map(|&s| mem.store.pools.subjects.get(s).to_string()).collect::<Vec<_>>())});
        // round-3 extensions: keys only when in use (the round-2 options' output stays as it was)
        if let Some(sc) = &opts.scope {
            out["answer_opts"]["scope"] = json!(sc.name());
        }
        if opts.loop_guard != LoopGuard::Off {
            out["answer_opts"]["loop_guard"] = json!(opts.loop_guard.name());
            out["answer_opts"]["loop_stopped"] = json!(ad
                .ep
                .loop_stopped
                .map(|(t, first)| json!({"step": t, "first_read": first})));
        }
    }
    if with_cands {
        out["candidate_ids"] = json!(cands.iter().map(|&r| fid(r)).collect::<Vec<_>>());
        out["candidates"] = json!(fields
            .iter()
            .map(|f| json!([f.subject, f.relation, f.value, f.quote]))
            .collect::<Vec<_>>());
        if slot.protocol == Protocol::V5 {
            out["candidate_inverse"] = json!(inverse);
        }
        out["hop_reads"] = json!(ad.hop_reads.iter().map(|(d, ids)| json!({"decision": d, "ids": ids.iter().map(|&r| fid(r)).collect::<Vec<_>>()})).collect::<Vec<_>>());
    }
    Ok(out)
}

/// The verify rule for stack-v2 programs. tools/verify_rows.py (`xerj_exec::verify`, stack-v1) vetoes an ANSWER whose cited
/// record's subject does not occur in the question; a CHAIN answer always cites the bridge record, whose subject is the value
/// reached by the previous hop (in "Who leads the city where TV Land is based?" the cited "New York City" record), so the v1 rule
/// would withhold every correct chain answer. Here subjects of records selected by a CHAIN step are exempt from rule (a); rule
/// (b) is unchanged (a question that names >= 2 store subjects needs >= 2 distinct cited subjects).
pub fn verify_chain_aware(
    mem: &Memory,
    question: &str,
    ans: &xerj_exec::Answer,
    buf: &mut QBuf,
) -> xerj_exec::Verdict {
    if ans.status != "ANSWER" {
        return xerj_exec::Verdict {
            vetoed: false,
            reason: "",
        };
    }
    let store = &mem.store;
    let mut exempt: Vec<&str> = Vec::new();
    for t in &ans.execution_trace {
        if t.op == "CHAIN" && !t.illegal {
            if let (Some(p), Some(h)) = (t.pointer, t.hop_candidate_ids.as_ref()) {
                if let Some(&r) = h.get(p.max(0) as usize) {
                    exempt.push(store.subject(r));
                }
            }
        }
    }
    let mut subs: Vec<&str> = Vec::new();
    for &r in &ans.evidence_ids {
        let s = store.subject(r);
        if !subs.contains(&s) {
            subs.push(s);
        }
    }
    if subs
        .iter()
        .any(|s| !question.contains(s) && !exempt.contains(s))
    {
        return xerj_exec::Verdict {
            vetoed: true,
            reason: "cited subject not in question (nor reached by CHAIN)",
        };
    }
    if subs.len() < 2 && mem.linker.substring_subjects(store, question, buf, 2).len() >= 2 {
        return xerj_exec::Verdict {
            vetoed: true,
            reason: "question names >= 2 subjects, answer cites < 2",
        };
    }
    xerj_exec::Verdict {
        vetoed: false,
        reason: "",
    }
}

/// the store listing shown in the Lab: every record with its visibility in the snapshot
pub fn store_rows(mem: &Memory, snap: &Snapshot<'_>) -> Value {
    let st = &mem.store;
    let rows: Vec<Value> = (0..st.num_records() as RecId)
        .map(|r| {
            let v = st.typed_value(r);
            json!({"fact_id": st.fact_id(r), "subject": st.subject(r), "relation": rel_phrase(st.relation(r)), "kind": kind_name(v.kind),
                   "value": render_value(&v), "quote": st.quote(r), "visible": snap.visible(r)})
        })
        .collect();
    json!(rows)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_pool_goes_to_one_question_at_a_time() {
        let busy = AtomicBool::new(false);
        let a = QuestionGuard::begin(&busy, true);
        assert!(a.threaded);
        let b = QuestionGuard::begin(&busy, true);
        assert!(!b.threaded, "a holds the pool: b runs single-threaded");
        drop(a);
        let c = QuestionGuard::begin(&busy, true);
        assert!(
            c.threaded,
            "the pool is free again, though b is still running"
        );
        drop(b);
        assert!(busy.load(Ordering::SeqCst), "c still holds it");
        drop(c);
        assert!(!busy.load(Ordering::SeqCst));
        assert!(
            !QuestionGuard::begin(&busy, false).threaded,
            "no pool, no threads"
        );
        assert!(!busy.load(Ordering::SeqCst));
    }

    #[test]
    fn with_the_queue_questions_take_turns_on_all_the_threads() {
        use std::sync::mpsc;
        let busy = AtomicBool::new(false);
        let turn = std::sync::Mutex::new(());
        let a = QuestionGuard::begin_turn(&busy, true, &turn);
        assert!(a.threaded);
        std::thread::scope(|s| {
            let (tx, rx) = mpsc::channel();
            let (busy, turn) = (&busy, &turn);
            s.spawn(move || {
                let b = QuestionGuard::begin_turn(busy, true, turn);
                tx.send(b.threaded).unwrap();
            });
            assert!(
                rx.recv_timeout(std::time::Duration::from_millis(200))
                    .is_err(),
                "b waits while a has its turn"
            );
            drop(a);
            assert_eq!(
                rx.recv_timeout(std::time::Duration::from_secs(10)),
                Ok(true),
                "then b gets the turn and the pool"
            );
        });
        assert!(!busy.load(Ordering::SeqCst), "b released the pool");
        assert!(turn.try_lock().is_ok(), "and its turn");
    }

    #[test]
    fn a_panicking_question_releases_the_pool() {
        let busy = AtomicBool::new(false);
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let q = QuestionGuard::begin(&busy, true);
            assert!(q.threaded);
            panic!("episode failed");
        }));
        assert!(r.is_err());
        assert!(!busy.load(Ordering::SeqCst));
        assert!(QuestionGuard::begin(&busy, true).threaded);
    }
}

#[cfg(test)]
mod stop_tests {
    use super::*;
    use xerj_infer::episode::ActionExecutor;

    /// The stop flag ends a free-running episode before its next decision, and only when set: without it, or while it
    /// stays unset, the adapter reports exactly the episode's own `done`.
    #[test]
    fn the_stop_flag_ends_the_episode_before_its_next_decision() {
        let text = "Liberia is governed by George Weah.";
        let world = json!({
            "world_id": "stop",
            "base_documents": [{"doc_id": "d1", "version": "v1", "text": text}],
            "base_facts": [{"fact_id": "f1", "subject": "Liberia", "relation": "head_of_government", "kind": "text",
                            "value": "George Weah", "doc_id": "d1", "version": "v1", "start": 0, "end": text.len(),
                            "quote": text}],
            "edits": []
        });
        let built = build_world(&world, None).expect("world");
        let snap = built.mem.snapshot_at(built.final_);
        let mut buf = QBuf::default();
        let q = "Who governs Liberia?";
        let m = built
            .mem
            .candidate_manifest_v5(&snap, "t/stop", q, 16, &mut buf);
        assert!(!m.cands.is_empty(), "the manifest holds f1");
        let episode = || {
            Episode::new_v5(
                &built.mem,
                snap,
                m.cands.clone(),
                m.inverse.clone(),
                "t/stop",
                16,
            )
        };

        let plain = ExecAdapter::new(episode());
        assert!(!plain.done() && !plain.stopped());

        let stop = Arc::new(AtomicBool::new(false));
        let mut ad = ExecAdapter::new(episode()).with_stop(stop.clone());
        assert!(!ad.done() && !ad.stopped(), "an unset flag changes nothing");
        // one legal decision runs as without the flag
        let legal = ad.legal_ops();
        assert!(legal[xerj_infer::episode::LOOKUP], "LOOKUP is legal first");
        ad.step(xerj_infer::episode::LOOKUP, Some(0)).expect("step");
        assert!(!ad.done() && !ad.stopped());
        stop.store(true, Ordering::Release);
        assert!(
            ad.done(),
            "a set flag ends the episode before its next decision"
        );
        assert!(ad.stopped(), "and marks it as cut short");
        assert!(!ad.ep.done, "the episode itself did not finish");
    }
}
