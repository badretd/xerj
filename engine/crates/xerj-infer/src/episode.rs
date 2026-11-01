//! Free-running episode driver (rollout._execute / rollout._execute_v2, also for stack "v5"): decision → legal argmax op →
//! argmax pointer → executor step, until the executor finishes. The exact executor (stack runner, store, manifests) is NOT part
//! of this crate: it is reached through [`ActionExecutor`] (implemented by xerj-exec in production and by the fixture replay
//! executor in the parity tests).
//!
//! Stack-v5 (direction) adds no operation and no pointer slot: the executor marks inverse candidates in the base manifest it
//! hands to [`run_episode`] and in its hop manifests ([`CandidateFields::inverse`]), and marks executed inverse hops in its trace
//! ([`HistoryStep::inverse`]); selection, legality and the pointer layout are stack-v2's.

use crate::codec::StackEntry;
use crate::config::{Arm, StackProtocol};
use crate::error::Result;
use crate::inputs::{self, CandidateFields, HistoryStep};
use crate::session::{DecisionOut, HopInput, Session};

pub const N_OPS: usize = 10;
pub const LOOKUP: usize = 0;
pub const ABSTAIN: usize = 6;
pub const CHAIN: usize = 7;
pub const OPS_V1: [&str; N_OPS] = [
    "LOOKUP",
    "LT",
    "GT",
    "EQ",
    "NOT",
    "RETURN",
    "ABSTAIN",
    "RESERVED7",
    "RESERVED8",
    "RESERVED9",
];
pub const OPS_V2: [&str; N_OPS] = [
    "LOOKUP", "LT", "GT", "EQ", "NOT", "RETURN", "ABSTAIN", "CHAIN", "PICK_LT", "PICK_GT",
];

/// the hop manifest the next stack-v2 / v5 decision reads
#[derive(Debug, Clone, Copy)]
pub struct HopView<'a> {
    /// hop candidates in manifest order (stack-v5: `inverse` set on the inverse ones)
    pub cands: &'a [CandidateFields],
    /// per candidate: the record resolves to the chained value (stack-v5: true for forward and inverse tier-1 records)
    pub resolves: &'a [bool],
    /// nonzero identity of the manifest (the memory arm re-encodes only when it changes)
    pub key: u64,
}

/// What the engine needs from the exact executor between decisions.
pub trait ActionExecutor {
    /// legality mask over the 10 operations for the next decision
    fn legal_ops(&self) -> [bool; N_OPS];
    /// execute one action: op id, pointer = base candidate position (LOOKUP) / hop candidate position (CHAIN) / None = NULL
    fn step(&mut self, op: usize, pointer: Option<usize>) -> Result<()>;
    fn done(&self) -> bool;
    /// executed actions (op name, pointer, result text, stack-v5 inverse flag), one per decision so far
    fn trace(&self) -> &[HistoryStep];
    /// executed stack, bottom -> top (rrt registers)
    fn stack(&self) -> &[StackEntry];
    /// stack-v2 / v5: hop manifest of the next decision (None = no hop manifest)
    fn hop(&self) -> Option<HopView<'_>>;
}

#[inline]
fn masked(v: f32, allowed: bool) -> f32 {
    if !allowed || v.is_nan() {
        f32::NEG_INFINITY
    } else {
        v
    }
}

/// rollout.choose_op: argmax over the legal operations; ABSTAIN when no legal operation has a usable score
pub fn choose_op(logits: &[f32], legal: &[bool; N_OPS]) -> usize {
    let mut best = f32::NEG_INFINITY;
    let mut arg = None;
    for (i, &v) in logits.iter().enumerate().take(N_OPS) {
        let s = masked(v, legal[i]);
        if s > best {
            best = s;
            arg = Some(i);
        }
    }
    match arg {
        Some(i) if legal[i] => i,
        _ => ABSTAIN,
    }
}

/// rollout.choose_pointer (stack-v1): argmax over the real candidates and NULL (last index); NULL -> None
pub fn choose_pointer_v1(logits: &[f32], n_cands: usize) -> Option<usize> {
    let n = logits.len();
    let null = n - 1;
    let mut best = f32::NEG_INFINITY;
    let mut arg = None;
    for (i, &v) in logits.iter().enumerate() {
        let ok = i < n_cands.min(null) || i == null;
        let s = masked(v, ok);
        if s > best {
            best = s;
            arg = Some(i);
        }
    }
    match arg {
        Some(i) if i != null => Some(i),
        _ => None,
    }
}

/// stack_v2.choose_pointer: argmax over the slots of the op (LOOKUP: base 0..n_base, CHAIN: hop K..K+n_hop, NULL 2K)
pub fn choose_pointer_v2(
    logits: &[f32],
    op: usize,
    n_base: usize,
    n_hop: usize,
    k: usize,
) -> Option<usize> {
    let mut best = f32::NEG_INFINITY;
    let mut arg = None;
    for (i, &v) in logits.iter().enumerate().take(2 * k + 1) {
        let ok = i == 2 * k
            || (op == LOOKUP && i < n_base.min(k))
            || (op == CHAIN && i >= k && i < k + n_hop.min(k));
        let s = masked(v, ok);
        if s > best {
            best = s;
            arg = Some(i);
        }
    }
    match arg {
        Some(p) if p != 2 * k => Some(if op == LOOKUP { p } else { p - k }),
        _ => None,
    }
}

/// one decision as observed by the driver
pub struct DecisionEvent<'a> {
    pub step: usize,
    pub out: &'a DecisionOut,
    pub legal: [bool; N_OPS],
    pub op: usize,
    pub pointer: Option<usize>,
    /// int8 guard: this decision was recomputed with the f32 weights
    pub fallback: bool,
}

#[derive(Debug, Clone, Default)]
pub struct EpisodeStats {
    pub decisions: usize,
    pub computed_tokens: usize,
    pub input_tokens: usize,
    pub program: Vec<(usize, Option<usize>)>,
    /// int8 guard: decisions recomputed with the f32 weights
    pub fallbacks: usize,
}

/// int8 guard: decisions whose winning margin is below these thresholds are recomputed with the f32 weights
#[derive(Debug, Clone, Copy)]
pub struct Guard {
    pub tau_op: f32,
    pub tau_ptr: f32,
}

/// top-1 minus top-2 over the allowed entries (+inf with fewer than two)
fn margin<I: Iterator<Item = f32>>(vals: I) -> f32 {
    let (mut a, mut b) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
    for v in vals {
        if v.is_nan() {
            continue;
        }
        if v > a {
            b = a;
            a = v;
        } else if v > b {
            b = v;
        }
    }
    if b == f32::NEG_INFINITY {
        f32::INFINITY
    } else {
        a - b
    }
}

/// (op margin, pointer margin) of a decision under the selection rules of choose_op / choose_pointer
pub fn margins(
    out: &DecisionOut,
    legal: &[bool; N_OPS],
    op: usize,
    stack: StackProtocol,
    n_base: usize,
    n_hop: usize,
    k: usize,
) -> (f32, f32) {
    let om = margin(
        out.op_logits
            .iter()
            .enumerate()
            .filter(|(i, _)| *i < N_OPS && legal[*i])
            .map(|(_, v)| *v),
    );
    let pm = match stack {
        StackProtocol::V1 if op == LOOKUP => {
            let null = out.ptr_logits.len() - 1;
            margin(
                out.ptr_logits
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| *i < n_base.min(null) || *i == null)
                    .map(|(_, v)| *v),
            )
        }
        StackProtocol::V2 | StackProtocol::V5 if op == LOOKUP || op == CHAIN => margin(
            out.ptr_logits
                .iter()
                .enumerate()
                .filter(|(i, _)| {
                    *i == 2 * k
                        || (op == LOOKUP && *i < n_base.min(k))
                        || (op == CHAIN && *i >= k && *i < k + n_hop.min(k))
                })
                .map(|(_, v)| *v),
        ),
        _ => f32::INFINITY,
    };
    (om, pm)
}

fn choose(
    out: &DecisionOut,
    legal: &[bool; N_OPS],
    stack: StackProtocol,
    n_base: usize,
    n_hop: usize,
    k: usize,
) -> (usize, Option<usize>) {
    let op = choose_op(&out.op_logits, legal);
    let pointer = match stack {
        StackProtocol::V1 => {
            if op == LOOKUP {
                choose_pointer_v1(&out.ptr_logits, n_base)
            } else {
                None
            }
        }
        StackProtocol::V2 | StackProtocol::V5 => {
            if op == LOOKUP || op == CHAIN {
                choose_pointer_v2(&out.ptr_logits, op, n_base, n_hop, k)
            } else {
                None
            }
        }
    };
    (op, pointer)
}

/// Run one question to the end with the given executor.
pub fn run_episode<E: ActionExecutor>(
    sess: &mut Session,
    question: &str,
    cands: &[CandidateFields],
    exec: &mut E,
    observe: Option<&mut dyn FnMut(&Session, &DecisionEvent)>,
) -> Result<EpisodeStats> {
    run_episode_guarded(sess, question, cands, exec, observe, None)
}

/// As [`run_episode`]; with an int8 model and a guard, a decision whose margin is below the guard thresholds is recomputed
/// with the f32 weights (from scratch: prefix + the full history, so its result is exactly the f32 decision); the next
/// decisions continue in int8 on top of the f32-built caches (and are guarded again).
pub fn run_episode_guarded<E: ActionExecutor>(
    sess: &mut Session,
    question: &str,
    cands: &[CandidateFields],
    exec: &mut E,
    mut observe: Option<&mut dyn FnMut(&Session, &DecisionEvent)>,
    guard: Option<Guard>,
) -> Result<EpisodeStats> {
    let cfg = &sess.m.cfg;
    let (arm, stack, k) = (cfg.arm, cfg.stack, cfg.k);
    let can_guard =
        guard.is_some() && sess.m.keep_f32 && sess.m.precision == crate::model::Precision::Int8;
    sess.set_force_f32(false);
    sess.begin(question, cands)?;
    let mut st = EpisodeStats::default();
    let mut t = 0;
    let mut hist_all: Vec<String> = Vec::new();
    while !exec.done() {
        let history = if t > 0 {
            let tr = exec.trace();
            let h = tr.get(t - 1).ok_or_else(|| {
                crate::error::Error::Executor("trace shorter than the decisions".into())
            })?;
            hist_all.push(inputs::history_segment(h, arm, stack, k));
            Some(hist_all[t - 1].clone())
        } else {
            None
        };
        let regs = if arm == Arm::Rrt && t > 0 {
            Some(exec.stack())
        } else {
            None
        };
        let hopv = if stack.has_hops() { exec.hop() } else { None };
        let hop = hopv.map(|h| HopInput {
            cands: h.cands,
            resolves: h.resolves,
            key: h.key,
        });
        let n_hop = hopv.map_or(0, |h| h.cands.len());
        let mut out = sess.decide(history.as_deref(), regs, hop.as_ref())?;
        let legal = exec.legal_ops();
        let (mut op, mut pointer) = choose(&out, &legal, stack, cands.len(), n_hop, k);
        let mut fell_back = false;
        if let (true, Some(g)) = (can_guard, guard) {
            let (om, pm) = margins(&out, &legal, op, stack, cands.len(), n_hop, k);
            if om < g.tau_op || pm < g.tau_ptr {
                st.fallbacks += 1;
                fell_back = true;
                sess.set_force_f32(true);
                sess.begin(question, cands)?;
                let ids = sess.tokenize_segments(&hist_all)?;
                out = sess.decide_ids(&ids, regs, hop.as_ref(), None)?;
                sess.set_force_f32(false);
                let c = choose(&out, &legal, stack, cands.len(), n_hop, k);
                op = c.0;
                pointer = c.1;
            }
        }
        if let Some(f) = observe.as_mut() {
            f(
                sess,
                &DecisionEvent {
                    step: t,
                    out: &out,
                    legal,
                    op,
                    pointer,
                    fallback: fell_back,
                },
            );
        }
        st.decisions += 1;
        st.computed_tokens += out.computed_tokens;
        st.input_tokens += out.input_tokens;
        st.program.push((op, pointer));
        exec.step(op, pointer)?;
        t += 1;
    }
    sess.set_force_f32(false);
    Ok(st)
}
