//! The exact action runner: stack_runner.Episode (stack-v1), stack_v2.EpisodeV2 (CHAIN, PICK_LT, PICK_GT, per-hop
//! manifests) and stack_v5.EpisodeV5 (stack-v2 + traversal direction: LOOKUP / CHAIN on an INVERSE candidate follow the
//! record from its value back to its subject), bit-for-bit in statuses, answers, evidence, errors, programs and execution
//! traces.
//!
//! Stack-v5 (`Protocol::V5`, `Episode::new_v5`): the base manifest carries one inverse flag per slot (`cand_inverse`, from
//! `Memory::candidate_manifest_v5`), hop manifests are `Memory::hop_manifest_v5` (inverse slots in `HopManifest::inverse`).
//! LOOKUP / CHAIN on a forward candidate behave exactly as stack-v2. On an inverse candidate (S, R, V) (`inverse_result`):
//! the record's own key lookup(S, R, scope) CONFLICT -> CONFLICT "inverse record: conflicting active values"; not VALUE ->
//! UNKNOWN "inverse record not active (<lookup reason>)"; a VALUE that no longer denotes V -> UNKNOWN "inverse record not
//! active (value changed)"; >= 2 subjects among the active text records (S', R, V') with the same relation and scope whose
//! value denotes V -> UNKNOWN "multiple subjects (set)" (references: those records); else VALUE text S with the key's
//! records as references (CHAIN: the whole path), origin ids {subject_id}. The action and the trace entry carry
//! `inverse = true` (Python `"direction": "in"`). Legality masks are stack-v2's.

use crate::result::{compare, merge_refs, negate, result_text, CmpOp, Res};
use smallvec::SmallVec;
use std::borrow::Cow;
use std::sync::Arc;
use xerj_index::direction::{max_inverse, RecScope, RelPref};
use xerj_index::scope::{intersect_sorted, QuestionScope, ScopeMode};
use xerj_index::{CandManifest, HopManifest, Memory};
use xerj_store::text::normalize_entity;
use xerj_store::value::{canonical_of_u32, date_year, render_value};
use xerj_store::{Kind, RecId, Snapshot, Status, TypedValue};

pub const OPS: [&str; 10] = [
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
pub const OPS_V2: [&str; 10] = [
    "LOOKUP", "LT", "GT", "EQ", "NOT", "RETURN", "ABSTAIN", "CHAIN", "PICK_LT", "PICK_GT",
];
pub const LOOKUP: i64 = 0;
pub const RETURN: i64 = 5;
pub const ABSTAIN: i64 = 6;
pub const CHAIN: i64 = 7;
pub const PICK_LT: i64 = 8;
pub const PICK_GT: i64 = 9;
pub const MAX_STACK: usize = 4;
pub const MAX_STEPS: usize = 6;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Protocol {
    /// stack_runner.Episode
    V1,
    /// stack_v2.EpisodeV2
    V2,
    /// stack_v5.EpisodeV5 (stack-v2 + traversal direction)
    V5,
}

impl Protocol {
    pub fn ops(self) -> &'static [&'static str; 10] {
        match self {
            Protocol::V1 => &OPS,
            Protocol::V2 | Protocol::V5 => &OPS_V2,
        }
    }
    pub fn op_id(self, name: &str) -> Option<i64> {
        self.ops().iter().position(|o| *o == name).map(|i| i as i64)
    }
    /// CHAIN / PICK_LT / PICK_GT with per-hop manifests (stack-v2 and stack-v5)
    pub fn has_hops(self) -> bool {
        matches!(self, Protocol::V2 | Protocol::V5)
    }
}

/// EXTENSION (opt-in; [`LoopGuard::Off`] = the reference): what a LOOKUP / CHAIN does that reads a record the episode
/// already read. Why (E8, one shared store, round 3): in a shared store a chain can cycle between two facts of different
/// worlds ("UK, head of state, Elizabeth II" and "Elizabeth II, country of citizenship, UK") until MAX_STEPS.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LoopGuard {
    /// the reference: no check (a cycling program runs until MAX_STEPS, then UNKNOWN "action_budget_exhausted")
    #[default]
    Off,
    /// the episode ends at once with UNKNOWN ("loop guard: ...")
    Unknown,
    /// the episode ends with the result of the step that first read the record: ANSWER with that step's value and
    /// evidence, CONFLICT, or UNKNOWN
    FirstUse,
}

impl LoopGuard {
    pub fn name(self) -> &'static str {
        match self {
            LoopGuard::Off => "off",
            LoopGuard::Unknown => "unknown",
            LoopGuard::FirstUse => "first_use",
        }
    }

    /// "off" | "unknown" | "first_use"
    pub fn parse(s: &str) -> Option<LoopGuard> {
        match s {
            "off" => Some(LoopGuard::Off),
            "unknown" => Some(LoopGuard::Unknown),
            "first_use" => Some(LoopGuard::FirstUse),
            _ => None,
        }
    }
}

/// one entry of `program` (actions)
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Action {
    pub op: String,
    pub pointer: Option<i64>,
    pub illegal: bool,
    /// stack-v5: the action followed an INVERSE candidate (Python `"direction": "in"`)
    pub inverse: bool,
}

/// one entry of `execution_trace`
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TraceEntry {
    pub step: usize,
    pub op: String,
    pub pointer: Option<i64>,
    pub result: Option<String>,
    pub illegal: bool,
    /// CHAIN steps (stack-v2): the fact ids of the hop manifest the pointer read
    pub hop_candidate_ids: Option<Vec<RecId>>,
    /// stack-v5: an executed inverse LOOKUP / CHAIN (Python `"direction": "in"`; the decision history renders " IN" after
    /// the pointer)
    pub inverse: bool,
}

/// the answer record of a finished episode
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Answer {
    /// "ANSWER" | "UNKNOWN" | "CONFLICT"
    pub status: &'static str,
    pub answer: Option<String>,
    /// sorted(set(evidence)) by fact id
    pub evidence_ids: Vec<RecId>,
    pub error: Option<String>,
    pub program: Vec<Action>,
    pub execution_trace: Vec<TraceEntry>,
}

/// provenance of a stack entry produced by LOOKUP / CHAIN (stack_v2.Origin)
#[derive(Clone, Debug)]
struct Origin<'s> {
    subject: Option<&'s str>,
    value_ids: SmallVec<[&'s str; 2]>,
}

pub struct Episode<'m> {
    pub mem: &'m Memory,
    pub snap: Snapshot<'m>,
    pub protocol: Protocol,
    pub cands: Vec<RecId>,
    pub item_id: String,
    pub k: usize,
    pub fingerprint: Arc<str>,
    pub stack: Vec<(Res<'m>, usize)>,
    pub actions: Vec<Action>,
    pub trace: Vec<TraceEntry>,
    pub done: bool,
    pub answer: Option<Answer>,
    /// stack-v5: one flag per base-manifest slot, true = inverse candidate (`CandManifest::inverse`; empty = all forward)
    pub cand_inverse: Vec<bool>,
    /// stack-v5: MAX_INVERSE of the hop manifests (default `xerj_index::direction::max_inverse()`: XERJ_R_MAX_INVERSE or 8)
    pub max_inverse: i64,
    /// EXTENSION (opt-in; None = the reference): question-aware choice of the hop-manifest records where a cap binds
    /// (`xerj_index::direction::RelPref`)
    pub rel_pref: Option<Arc<RelPref>>,
    /// EXTENSION (opt-in; None = the reference): the records the hop manifests may offer (`xerj_index::scope`); a
    /// `ScopeMode::Coherent` scope follows the records this episode reads
    pub scope: Option<Arc<QuestionScope>>,
    /// EXTENSION (opt-in; `LoopGuard::Off` = the reference)
    pub loop_guard: LoopGuard,
    /// [`LoopGuard`]: (the action that re-read a record, the step that first read it) when the guard ended the episode
    pub loop_stopped: Option<(usize, usize)>,
    origins: Vec<(usize, Origin<'m>)>,
    hops: Vec<(usize, Arc<HopManifest>)>,
    /// extensions only (scope or loop guard set): the records read by LOOKUP / CHAIN, the step and its result
    reads: Vec<(RecId, usize, Res<'m>)>,
    /// `ScopeMode::Coherent`: the collections every record read so far belongs to (None before the first read)
    chain_cols: Option<SmallVec<[u32; 4]>>,
}

fn no_hop() -> Arc<HopManifest> {
    Arc::new(HopManifest::default())
}

impl<'m> Episode<'m> {
    /// stack_runner.Episode(snapshot, cands) / stack_v2.new_episode(snapshot, cands, item_id, K, entity_ids, fingerprint)
    pub fn new(
        mem: &'m Memory,
        snap: Snapshot<'m>,
        protocol: Protocol,
        cands: Vec<RecId>,
        item_id: &str,
        k: usize,
    ) -> Episode<'m> {
        let fingerprint = snap.fingerprint();
        Episode {
            mem,
            snap,
            protocol,
            cands,
            item_id: item_id.to_string(),
            k,
            fingerprint,
            stack: Vec::with_capacity(MAX_STACK),
            actions: Vec::with_capacity(MAX_STEPS),
            trace: Vec::with_capacity(MAX_STEPS),
            done: false,
            answer: None,
            cand_inverse: Vec::new(),
            max_inverse: max_inverse(),
            rel_pref: None,
            scope: None,
            loop_guard: LoopGuard::Off,
            loop_stopped: None,
            origins: Vec::new(),
            hops: Vec::new(),
            reads: Vec::new(),
            chain_cols: None,
        }
    }

    /// stack_v5.new_episode(snapshot, cands, item_id, K, entity_ids, fingerprint): `inverse` = one flag per candidate
    /// (`CandManifest::inverse`; empty = no inverse candidate)
    pub fn new_v5(
        mem: &'m Memory,
        snap: Snapshot<'m>,
        cands: Vec<RecId>,
        inverse: Vec<bool>,
        item_id: &str,
        k: usize,
    ) -> Episode<'m> {
        assert!(
            inverse.is_empty() || inverse.len() == cands.len(),
            "one inverse flag per candidate"
        );
        let mut ep = Episode::new(mem, snap, Protocol::V5, cands, item_id, k);
        ep.cand_inverse = inverse;
        ep
    }

    /// `Episode::new_v5` from a stack-v5 base manifest
    pub fn from_manifest_v5(
        mem: &'m Memory,
        snap: Snapshot<'m>,
        m: CandManifest,
        item_id: &str,
        k: usize,
    ) -> Episode<'m> {
        Episode::new_v5(mem, snap, m.cands, m.inverse, item_id, k)
    }

    /// is base-manifest slot `k` an inverse candidate? (stack-v5; false otherwise)
    #[inline]
    pub fn cand_is_inverse(&self, k: usize) -> bool {
        self.cand_inverse.get(k).copied().unwrap_or(false)
    }

    fn origin(&self, step: usize) -> Option<&Origin<'m>> {
        self.origins.iter().find(|o| o.0 == step).map(|o| &o.1)
    }

    /// the subject of the record step `step` read (Python `ep.origins[step].subject`): None for a step without an origin or a
    /// NULL / invalid pointer. Read by the decline layer's role check (xerj-serve `decline_v2`).
    pub fn origin_subject(&self, step: usize) -> Option<&'m str> {
        self.origin(step).and_then(|o| o.subject)
    }

    fn is_source(&self, idx_from_top: usize) -> bool {
        let n = self.stack.len();
        n > idx_from_top && self.origin(self.stack[n - 1 - idx_from_top].1).is_some()
    }

    /// the legality mask of the next action
    pub fn legal_ops(&self) -> [bool; 10] {
        let n = self.stack.len();
        let top = self.stack.last().map(|t| t.0.status);
        let mut m = [
            n < MAX_STACK,
            n >= 2,
            n >= 2,
            n >= 2,
            n >= 1
                && matches!(
                    top,
                    Some(Status::True | Status::False | Status::Unknown | Status::Conflict)
                ),
            n >= 1,
            true,
            false,
            false,
            false,
        ];
        if self.protocol.has_hops() {
            m[7] = n >= 1 && self.is_source(0);
            m[8] = n >= 2 && self.is_source(0) && self.is_source(1);
            m[9] = m[8];
        }
        m
    }

    fn finish(
        &mut self,
        status: &'static str,
        answer: Option<String>,
        evidence: &[RecId],
        error: Option<String>,
    ) {
        self.done = true;
        let store = self.snap.store;
        let mut ev: Vec<RecId> = evidence.to_vec();
        ev.sort_unstable_by(|&a, &b| store.fact_id(a).cmp(store.fact_id(b)));
        ev.dedup();
        self.answer = Some(Answer {
            status,
            answer,
            evidence_ids: ev,
            error,
            program: self.actions.clone(),
            execution_trace: self.trace.clone(),
        });
    }

    /// value ids of the records that produced a value (stack_v2.value_ids_of)
    fn value_ids_of(&self, records: &[RecId]) -> SmallVec<[&'m str; 2]> {
        let store = self.snap.store;
        let mut v: SmallVec<[&'m str; 2]> = SmallVec::new();
        for &r in records {
            if let Some(x) = store.value_id(r) {
                if !v.contains(&x) {
                    v.push(x);
                }
            }
        }
        v
    }

    /// stack_v2.EpisodeV2.current_hop / stack_v5.EpisodeV5.current_hop: the hop manifest the next decision reads (empty
    /// unless the top entry is a text VALUE produced by LOOKUP / CHAIN and some record resolves to it; stack-v5: forward or
    /// inverse)
    pub fn current_hop(&mut self) -> Arc<HopManifest> {
        if !self.protocol.has_hops() {
            return no_hop();
        }
        let Some((r, step)) = self.stack.last() else {
            return no_hop();
        };
        let step = *step;
        let Some(origin) = self.origin(step) else {
            return no_hop();
        };
        let v = match (r.status, r.value) {
            (Status::Value, Some(v)) if v.kind == Kind::Text => v,
            _ => return no_hop(),
        };
        if let Some(h) = self.hops.iter().find(|h| h.0 == step) {
            return h.1.clone();
        }
        let ids: SmallVec<[&str; 2]> = origin.value_ids.clone();
        // EXTENSION (opt-in): a scoped hop manifest (a Coherent scope only once the records read share a collection)
        let scoped: Option<(Arc<QuestionScope>, SmallVec<[u32; 4]>)> = match &self.scope {
            Some(sc) if self.protocol == Protocol::V5 => match &sc.mode {
                ScopeMode::Filter(set) | ScopeMode::Prefer(set) => {
                    Some((sc.clone(), SmallVec::from_slice(set)))
                }
                ScopeMode::Coherent => match &self.chain_cols {
                    Some(c) if !c.is_empty() => Some((sc.clone(), c.clone())),
                    _ => None,
                },
            },
            _ => None,
        };
        let h = Arc::new(if let Some((sc, set)) = scoped {
            let map = sc.map.clone();
            let keep = |r: RecId| map.in_any(r, &set);
            let rs = match sc.mode {
                ScopeMode::Prefer(_) => RecScope::Prefer(&keep),
                _ => RecScope::Filter(&keep),
            };
            self.mem.hop_manifest_v5_scope(
                &self.snap,
                &self.item_id,
                &self.fingerprint,
                &v,
                &ids,
                self.k,
                self.max_inverse,
                self.rel_pref.as_deref(),
                rs,
            )
        } else if self.protocol == Protocol::V5 {
            match &self.rel_pref {
                None => self.mem.hop_manifest_v5(
                    &self.snap,
                    &self.item_id,
                    &self.fingerprint,
                    &v,
                    &ids,
                    self.k,
                    self.max_inverse,
                ),
                Some(p) => self.mem.hop_manifest_v5_pref(
                    &self.snap,
                    &self.item_id,
                    &self.fingerprint,
                    &v,
                    &ids,
                    self.k,
                    self.max_inverse,
                    Some(p.as_ref()),
                ),
            }
        } else {
            self.mem.hop_manifest(
                &self.snap,
                &self.item_id,
                &self.fingerprint,
                &v,
                &ids,
                self.k,
            )
        });
        self.hops.push((step, h.clone()));
        h
    }

    /// Execute one action (op id, pointer = candidate position or None for NULL). Err only when the episode is done.
    pub fn step(&mut self, op: i64, pointer: Option<i64>) -> Result<(), &'static str> {
        if self.done {
            return Err("episode already finished");
        }
        let ops = self.protocol.ops();
        let legal = self.legal_ops();
        if !(0..10).contains(&op) || !legal[op as usize] {
            let name = if (0..10).contains(&op) {
                ops[op as usize].to_string()
            } else {
                op.to_string()
            };
            let is_ptr = name == "LOOKUP" || (self.protocol.has_hops() && name == "CHAIN");
            let ptr = if is_ptr { pointer } else { None };
            let t = self.actions.len();
            self.actions.push(Action {
                op: name.clone(),
                pointer: ptr,
                illegal: true,
                inverse: false,
            });
            self.trace.push(TraceEntry {
                step: t,
                op: name.clone(),
                pointer: ptr,
                result: None,
                illegal: true,
                hop_candidate_ids: None,
                inverse: false,
            });
            self.finish("UNKNOWN", None, &[], Some(format!("illegal action {name}")));
            return Ok(());
        }
        if self.loop_guard == LoopGuard::Off && self.scope.is_none() {
            // the reference
            self.dispatch(op, pointer);
            return Ok(());
        }
        // EXTENSION (opt-in): the loop guard, and the reads a scoped or guarded episode follows
        let target = self.target_record(op, pointer);
        if let Some(r) = target {
            if self.loop_guard != LoopGuard::Off {
                if let Some(j) = self.reads.iter().position(|x| x.0 == r) {
                    self.loop_stop(op, pointer, j);
                    return Ok(());
                }
            }
        }
        let t = self.actions.len();
        self.dispatch(op, pointer);
        if let Some(r) = target {
            self.note_read(r, t);
        }
        Ok(())
    }

    /// a legal action (stack_v5.EpisodeV5.step / stack_v2 / stack-v1 dispatch)
    fn dispatch(&mut self, op: i64, pointer: Option<i64>) {
        if self.protocol == Protocol::V5 {
            // stack_v5.EpisodeV5.step: a legal LOOKUP / CHAIN whose pointer names an inverse candidate
            let p = pointer.filter(|&p| p >= 0).map(|p| p as usize);
            if op == LOOKUP {
                if let Some(p) = p.filter(|&p| p < self.cands.len() && self.cand_is_inverse(p)) {
                    self.inverse_lookup(p);
                    return;
                }
            } else if op == CHAIN {
                let hop = self.current_hop();
                if let Some(p) = p.filter(|&p| p < hop.len() && hop.is_inverse(p)) {
                    self.inverse_chain(p, hop);
                    return;
                }
            }
        }
        if self.protocol.has_hops() && op >= CHAIN {
            self.step_v2(op, pointer);
            return;
        }
        self.step_v1(op, pointer);
    }

    /// EXTENSION: the record a legal LOOKUP / CHAIN with this pointer reads (None for other ops and NULL pointers)
    fn target_record(&mut self, op: i64, pointer: Option<i64>) -> Option<RecId> {
        let p = pointer.filter(|&p| p >= 0)? as usize;
        if op == LOOKUP {
            self.cands.get(p).copied()
        } else if op == CHAIN && self.protocol.has_hops() {
            self.current_hop().cands.get(p).copied()
        } else {
            None
        }
    }

    /// EXTENSION: step `t` read record `r` (a Coherent scope narrows to the collections of every record read)
    fn note_read(&mut self, r: RecId, t: usize) {
        let Some(res) = self.stack.last().filter(|e| e.1 == t).map(|e| e.0.clone()) else {
            return;
        };
        self.reads.push((r, t, res));
        if let Some(sc) = &self.scope {
            if sc.mode == ScopeMode::Coherent {
                let of = sc.map.of(r);
                self.chain_cols = Some(match &self.chain_cols {
                    None => SmallVec::from_slice(of),
                    Some(c) => intersect_sorted(c, of),
                });
            }
        }
    }

    /// EXTENSION ([`LoopGuard`]): the legal action `op` reads the record that step `self.reads[j].1` read: it is recorded
    /// (result none) and the episode ends as the guard says
    fn loop_stop(&mut self, op: i64, pointer: Option<i64>, j: usize) {
        let name = self.protocol.ops()[op as usize].to_string();
        let t = self.actions.len();
        let slot = pointer.unwrap_or(0).max(0) as usize;
        let (inverse, hop_ids) = if op == CHAIN {
            let hop = self.current_hop();
            (hop.is_inverse(slot), Some(hop.cands.clone()))
        } else {
            (self.cand_is_inverse(slot), None)
        };
        self.actions.push(Action {
            op: name.clone(),
            pointer,
            illegal: false,
            inverse,
        });
        self.trace.push(TraceEntry {
            step: t,
            op: name,
            pointer,
            result: None,
            illegal: false,
            hop_candidate_ids: hop_ids,
            inverse,
        });
        let (_, first, res) = self.reads[j].clone();
        self.loop_stopped = Some((t, first));
        let why = format!("loop guard: step {t} reads the record step {first} read");
        match (self.loop_guard, res.status, res.value) {
            (LoopGuard::FirstUse, Status::Value, Some(v)) => {
                self.finish("ANSWER", Some(render_value(&v)), &res.refs, None)
            }
            (LoopGuard::FirstUse, Status::Conflict, _) => {
                self.finish("CONFLICT", None, &res.refs, Some(why))
            }
            _ => self.finish("UNKNOWN", None, &[], Some(why)),
        }
    }

    fn step_v1(&mut self, op: i64, pointer: Option<i64>) {
        let store = self.snap.store;
        let name = OPS[op as usize];
        let t = self.actions.len();
        let is_lookup = op == LOOKUP;
        self.actions.push(Action {
            op: name.to_string(),
            pointer: if is_lookup { pointer } else { None },
            illegal: false,
            inverse: false,
        });
        match op {
            0 => {
                let valid = matches!(pointer, Some(p) if p >= 0 && (p as usize) < self.cands.len());
                let r = if valid {
                    let rec = self.cands[pointer.unwrap() as usize];
                    Res::from_lookup(self.snap.lookup_record_key(rec))
                } else {
                    Res::unknown(SmallVec::new(), "NULL pointer")
                };
                if self.protocol.has_hops() {
                    let records: SmallVec<[RecId; 4]> = if valid {
                        r.refs.clone()
                    } else {
                        SmallVec::new()
                    };
                    let subject = if valid {
                        Some(store.subject(self.cands[pointer.unwrap() as usize]))
                    } else {
                        None
                    };
                    let value_ids = if r.status == Status::Value {
                        self.value_ids_of(&records)
                    } else {
                        SmallVec::new()
                    };
                    self.origins.push((t, Origin { subject, value_ids }));
                }
                self.stack.push((r, t));
            }
            1..=3 => {
                let (b, _) = self.stack.pop().unwrap();
                let (a, _) = self.stack.pop().unwrap();
                let cmp = match op {
                    1 => CmpOp::Lt,
                    2 => CmpOp::Gt,
                    _ => CmpOp::Eq,
                };
                let r = compare(store, &a, &b, cmp);
                self.stack.push((r, t));
            }
            4 => {
                let (a, _) = self.stack.pop().unwrap();
                let r =
                    negate(&a).expect("legality guarantees a logical / unknown / conflict operand");
                self.stack.push((r, t));
            }
            5 => {
                let r = self.stack.last().unwrap().0.clone();
                match r.status {
                    Status::Value => {
                        let v = r.value.expect("value");
                        self.finish("ANSWER", Some(render_value(&v)), &r.refs, None);
                    }
                    Status::True | Status::False => self.finish(
                        "ANSWER",
                        Some(
                            if r.status == Status::True {
                                "yes"
                            } else {
                                "no"
                            }
                            .to_string(),
                        ),
                        &r.refs,
                        None,
                    ),
                    Status::Conflict => self.finish("CONFLICT", None, &r.refs, None),
                    Status::Unknown => self.finish("UNKNOWN", None, &r.refs, None),
                }
            }
            6 => self.finish("UNKNOWN", None, &[], Some("abstained".into())),
            _ => unreachable!(),
        }
        let result = if !self.stack.is_empty() && op != RETURN && op != ABSTAIN {
            Some(result_text(&self.stack.last().unwrap().0))
        } else {
            None
        };
        self.trace.push(TraceEntry {
            step: t,
            op: name.to_string(),
            pointer: if is_lookup { pointer } else { None },
            result,
            illegal: false,
            hop_candidate_ids: None,
            inverse: false,
        });
        if self.done {
            if let Some(a) = self.answer.as_mut() {
                a.execution_trace = self.trace.clone();
            }
        }
        if !self.done && self.actions.len() >= MAX_STEPS {
            self.finish("UNKNOWN", None, &[], Some("action_budget_exhausted".into()));
        }
    }

    fn step_v2(&mut self, op: i64, pointer: Option<i64>) {
        let store = self.snap.store;
        let name = OPS_V2[op as usize];
        let t = self.actions.len();
        let mut entry = TraceEntry {
            step: t,
            op: name.to_string(),
            pointer: if op == CHAIN { pointer } else { None },
            result: None,
            illegal: false,
            hop_candidate_ids: None,
            inverse: false,
        };
        if op == CHAIN {
            let hop = self.current_hop();
            self.actions.push(Action {
                op: name.to_string(),
                pointer,
                illegal: false,
                inverse: false,
            });
            let (prev, _) = self.stack.pop().unwrap();
            let mut origin = Origin {
                subject: None,
                value_ids: SmallVec::new(),
            };
            let r = if prev.status == Status::Conflict {
                Res::conflict(prev.refs.clone(), "CHAIN on a conflicting value")
            } else if prev.status != Status::Value {
                Res::unknown(prev.refs.clone(), "CHAIN on a missing value")
            } else if !matches!(pointer, Some(p) if p >= 0 && (p as usize) < hop.len()) {
                Res::unknown(prev.refs.clone(), "NULL pointer")
            } else if !hop.resolves[pointer.unwrap() as usize] {
                Res::unknown(
                    prev.refs.clone(),
                    "record subject does not resolve to the value",
                )
            } else {
                let f = hop.cands[pointer.unwrap() as usize];
                let look = self.snap.lookup_record_key(f);
                let path = merge_refs(store, &prev.refs, &look.refs);
                let records = look.refs.clone();
                let r = match look.status {
                    Status::Value => Res {
                        status: Status::Value,
                        value: look.value,
                        refs: path,
                        reason: Cow::Borrowed(""),
                    },
                    Status::Conflict => Res::conflict(path, look.reason),
                    _ => Res::unknown(prev.refs.clone(), look.reason),
                };
                let value_ids = if look.status == Status::Value {
                    self.value_ids_of(&records)
                } else {
                    SmallVec::new()
                };
                origin = Origin {
                    subject: Some(store.subject(f)),
                    value_ids,
                };
                r
            };
            self.stack.push((r, t));
            self.origins.retain(|o| o.0 != t);
            self.origins.push((t, origin));
            entry.hop_candidate_ids = Some(hop.cands.clone());
        } else {
            self.actions.push(Action {
                op: name.to_string(),
                pointer: None,
                illegal: false,
                inverse: false,
            });
            let (b, sb) = self.stack.pop().unwrap();
            let (a, sa) = self.stack.pop().unwrap();
            let refs = merge_refs(store, &a.refs, &b.refs);
            let r = match pick_compare(store, &a, &b, op) {
                Err(bad) => bad,
                Ok((chosen, gran)) => {
                    let subject = self
                        .origin(if chosen == 0 { sa } else { sb })
                        .and_then(|o| o.subject);
                    match subject {
                        Some(s) => Res {
                            status: Status::Value,
                            value: Some(TypedValue {
                                kind: Kind::Text,
                                value: s,
                                unit: "",
                            }),
                            refs,
                            reason: Cow::Owned(format!("{name} at {gran} granularity")),
                        },
                        None => Res::unknown(refs, "operand without a record subject"),
                    }
                }
            };
            self.stack.push((r, t));
        }
        entry.result = Some(result_text(&self.stack.last().unwrap().0));
        self.trace.push(entry);
        if !self.done && self.actions.len() >= MAX_STEPS {
            self.finish("UNKNOWN", None, &[], Some("action_budget_exhausted".into()));
        }
    }

    /// stack_v5.EpisodeV5._budget
    fn budget(&mut self) {
        if !self.done && self.actions.len() >= MAX_STEPS {
            self.finish("UNKNOWN", None, &[], Some("action_budget_exhausted".into()));
        }
    }

    /// stack_v5.inverse_result: the exact result of following inverse candidate `rec` (S, R, V) from V back to S, and the
    /// record's subject_id
    fn inverse_result(&self, rec: RecId) -> (Res<'m>, Option<&'m str>) {
        let store = self.snap.store;
        let sid = store.subject_id(rec);
        let look = self.snap.lookup_record_key(rec);
        match look.status {
            Status::Conflict => {
                return (
                    Res::conflict(look.refs, "inverse record: conflicting active values"),
                    sid,
                )
            }
            Status::Value => {}
            _ => {
                return (
                    Res::unknown(
                        look.refs,
                        format!("inverse record not active ({})", look.reason),
                    ),
                    sid,
                )
            }
        }
        let lv = look.value.expect("a VALUE lookup carries its value");
        let fv = store.typed_value(rec);
        // normalize_entity(f.value.value): Python's str() of the record's own value
        let fs = if fv.kind == Kind::Boolean {
            if fv.value == "true" {
                "True"
            } else {
                "False"
            }
        } else {
            fv.value
        };
        if !(lv.kind == Kind::Text && normalize_entity(lv.value) == normalize_entity(fs)) {
            return (
                Res::unknown(look.refs, "inverse record not active (value changed)"),
                sid,
            );
        }
        let same = self.mem.inverse_same_value(&self.snap, rec);
        let first_subject = same.first().map(|&g| store.recs.subject[g as usize]);
        if same
            .iter()
            .any(|&g| Some(store.recs.subject[g as usize]) != first_subject)
        {
            return (
                Res::unknown(merge_refs(store, &same, &[]), "multiple subjects (set)"),
                sid,
            );
        }
        (
            Res {
                status: Status::Value,
                value: Some(TypedValue {
                    kind: Kind::Text,
                    value: store.subject(rec),
                    unit: "",
                }),
                refs: look.refs,
                reason: Cow::Borrowed(""),
            },
            sid,
        )
    }

    /// stack_v5.EpisodeV5._inverse_lookup (a legal LOOKUP on inverse base slot `p`)
    fn inverse_lookup(&mut self, p: usize) {
        let store = self.snap.store;
        let t = self.actions.len();
        let rec = self.cands[p];
        self.actions.push(Action {
            op: "LOOKUP".to_string(),
            pointer: Some(p as i64),
            illegal: false,
            inverse: true,
        });
        let (r, sid) = self.inverse_result(rec);
        let ok = r.status == Status::Value;
        let value_ids: SmallVec<[&'m str; 2]> = match (ok, sid) {
            (true, Some(s)) => SmallVec::from_slice(&[s]),
            _ => SmallVec::new(),
        };
        let result = result_text(&r);
        self.stack.push((r, t));
        self.origins.retain(|o| o.0 != t);
        self.origins.push((
            t,
            Origin {
                subject: Some(store.subject(rec)),
                value_ids,
            },
        ));
        self.trace.push(TraceEntry {
            step: t,
            op: "LOOKUP".to_string(),
            pointer: Some(p as i64),
            result: Some(result),
            illegal: false,
            hop_candidate_ids: None,
            inverse: true,
        });
        self.budget();
    }

    /// stack_v5.EpisodeV5._inverse_chain (a legal CHAIN on inverse hop slot `p` of `hop`)
    fn inverse_chain(&mut self, p: usize, hop: Arc<HopManifest>) {
        let store = self.snap.store;
        let t = self.actions.len();
        let rec = hop.cands[p];
        self.actions.push(Action {
            op: "CHAIN".to_string(),
            pointer: Some(p as i64),
            illegal: false,
            inverse: true,
        });
        let (prev, _) = self.stack.pop().unwrap();
        let mut origin = Origin {
            subject: None,
            value_ids: SmallVec::new(),
        };
        let r = if prev.status == Status::Conflict {
            Res::conflict(prev.refs.clone(), "CHAIN on a conflicting value")
        } else if prev.status != Status::Value {
            Res::unknown(prev.refs.clone(), "CHAIN on a missing value")
        } else {
            let (look, sid) = self.inverse_result(rec);
            let path = merge_refs(store, &prev.refs, &look.refs);
            match look.status {
                Status::Value => {
                    origin = Origin {
                        subject: Some(store.subject(rec)),
                        value_ids: sid.into_iter().collect(),
                    };
                    Res {
                        status: Status::Value,
                        value: look.value,
                        refs: path,
                        reason: Cow::Borrowed(""),
                    }
                }
                Status::Conflict => Res::conflict(path, look.reason),
                _ => Res::unknown(path, look.reason),
            }
        };
        let result = result_text(&r);
        self.stack.push((r, t));
        self.origins.retain(|o| o.0 != t);
        self.origins.push((t, origin));
        self.trace.push(TraceEntry {
            step: t,
            op: "CHAIN".to_string(),
            pointer: Some(p as i64),
            result: Some(result),
            illegal: false,
            hop_candidate_ids: Some(hop.cands.clone()),
            inverse: true,
        });
        self.budget();
    }

    /// the episode's answer record, or the "unfinished" fallback of stack_v2_check.verify_item
    pub fn answer_or_unfinished(&self) -> Answer {
        self.answer.clone().unwrap_or(Answer {
            status: "UNKNOWN",
            answer: None,
            evidence_ids: Vec::new(),
            error: Some("unfinished".into()),
            program: self.actions.clone(),
            execution_trace: self.trace.clone(),
        })
    }

    /// reason of the top stack entry (diagnostics, e.g. PICK granularity)
    pub fn top_reason(&self) -> Option<&str> {
        self.stack.last().map(|t| t.0.reason.as_ref())
    }
}

/// stack_v2.pick_compare: Err(result) for CONFLICT / UNKNOWN outcomes, Ok((chosen operand 0 = a / 1 = b, granularity))
pub fn pick_compare<'s>(
    store: &xerj_store::Store,
    a: &Res<'s>,
    b: &Res<'s>,
    op: i64,
) -> Result<(u8, &'static str), Res<'s>> {
    let refs = merge_refs(store, &a.refs, &b.refs);
    if a.status == Status::Conflict || b.status == Status::Conflict {
        return Err(Res::conflict(refs, "Conflicting operand"));
    }
    if a.status != Status::Value || b.status != Status::Value {
        return Err(Res::unknown(refs, "Missing operand"));
    }
    let (va, vb) = (a.value.unwrap(), b.value.unwrap());
    let mut gran = "value";
    // year granularity: a date contributes TypedValue("number", str(year), "year")
    let ya;
    let yb;
    let (xa, xb) = if va.kind == Kind::Date && vb.kind == Kind::Date {
        gran = "day";
        (va, vb)
    } else if ((va.kind == Kind::Date && vb.kind == Kind::Number)
        || (va.kind == Kind::Number && vb.kind == Kind::Date))
        && [va, vb]
            .iter()
            .all(|v| v.kind == Kind::Date || v.unit == "year")
    {
        gran = "year";
        ya = (va.kind == Kind::Date).then(|| canonical_of_u32(date_year(va.value)));
        yb = (vb.kind == Kind::Date).then(|| canonical_of_u32(date_year(vb.value)));
        let fa = match &ya {
            Some(s) => TypedValue {
                kind: Kind::Number,
                value: s.as_str(),
                unit: "year",
            },
            None => va,
        };
        let fb = match &yb {
            Some(s) => TypedValue {
                kind: Kind::Number,
                value: s.as_str(),
                unit: "year",
            },
            None => vb,
        };
        (fa, fb)
    } else {
        if va.kind == Kind::Number
            && vb.kind == Kind::Number
            && va.unit == "year"
            && vb.unit == "year"
        {
            gran = "year";
        }
        (va, vb)
    };
    let ra = Res {
        status: Status::Value,
        value: Some(xa),
        refs: a.refs.clone(),
        reason: Cow::Borrowed(""),
    };
    let rb = Res {
        status: Status::Value,
        value: Some(xb),
        refs: b.refs.clone(),
        reason: Cow::Borrowed(""),
    };
    let lt = compare(store, &ra, &rb, CmpOp::Lt);
    let gt = compare(store, &ra, &rb, CmpOp::Gt);
    let tf = |s: Status| matches!(s, Status::True | Status::False);
    if !tf(lt.status) || !tf(gt.status) {
        let why = if !lt.reason.is_empty() {
            lt.reason.clone()
        } else {
            gt.reason.clone()
        };
        return Err(Res::unknown(refs, format!("Incomparable operands ({why})")));
    }
    if lt.status == Status::False && gt.status == Status::False {
        return Err(Res::unknown(refs, format!("Tie at {gran} granularity")));
    }
    let smaller = if lt.status == Status::True { 0 } else { 1 };
    Ok((if op == PICK_LT { smaller } else { 1 - smaller }, gran))
}

/// One gold trace step over fact KEYS (private labels; parity tests and teacher programs only).
#[derive(Clone, Debug)]
pub struct GoldStep {
    pub op: String,
    pub key: Option<(String, String)>,
    /// stack-v5: the step carries `"direction": "in"` (stack_v5.teacher_target maps it to INVERSE positions only, a forward
    /// step to forward positions only); ignored by stack-v1 / v2
    pub inverse: bool,
}

/// stack_v2.teacher_target / stack_runner.teacher_actions / stack_v5.teacher_target: (op id, pointer) of a gold step in the
/// current episode state
pub fn teacher_target(ep: &mut Episode<'_>, step: &GoldStep) -> (i64, Option<i64>) {
    let store = ep.snap.store;
    let op = ep.protocol.op_id(&step.op).unwrap_or(-1);
    let v5 = ep.protocol == Protocol::V5;
    let has_key =
        |r: RecId, key: &(String, String)| store.subject(r) == key.0 && store.relation(r) == key.1;
    match (op, &step.key) {
        (LOOKUP, Some(k)) => {
            let p = (0..ep.cands.len()).find(|&i| {
                (!v5 || ep.cand_is_inverse(i) == step.inverse) && has_key(ep.cands[i], k)
            });
            (op, p.map(|p| p as i64))
        }
        (CHAIN, Some(k)) => {
            let hop = ep.current_hop();
            let p = (0..hop.len())
                .find(|&j| (!v5 || hop.is_inverse(j) == step.inverse) && has_key(hop.cands[j], k));
            (op, p.map(|p| p as i64))
        }
        _ => (op, None),
    }
}

/// Run a gold trace the way tools/stack_v2_check.verify_item (v2) / tools/build_direction_worlds.py verify (v5) or the
/// stack-v1 teacher (v1) do: returns the answer record and the problems (illegal teacher op / episode finished early / not
/// terminated).
pub fn run_gold(ep: &mut Episode<'_>, trace: &[GoldStep]) -> (Answer, Vec<String>) {
    let mut problems = Vec::new();
    for (t, st) in trace.iter().enumerate() {
        if ep.done {
            problems.push(format!("episode finished before step {t}"));
            break;
        }
        let (op, ptr) = teacher_target(ep, st);
        if ep.protocol.has_hops() && !(op >= 0 && ep.legal_ops()[op as usize]) {
            problems.push(format!("teacher op {} illegal at {t}", st.op));
            break;
        }
        let _ = ep.step(op, ptr);
    }
    if !ep.done {
        problems.push("teacher program does not terminate".into());
    }
    (ep.answer_or_unfinished(), problems)
}

#[cfg(test)]
mod extension_tests {
    use super::*;
    use xerj_index::scope::RecCollections;
    use xerj_index::{Memory, MemoryConfig};
    use xerj_store::FactIn;

    /// three facts that let a chain cycle: Borujerdi -citizenship-> Iran -head of government-> Raisi -citizenship-> Iran
    fn cycle_memory() -> (Memory, [RecId; 3]) {
        let mut mem = Memory::new(MemoryConfig {
            bm25_records: false,
            bm25_documents: false,
            ..MemoryConfig::default()
        });
        mem.add_document("d", "v1", "abcd efgh").expect("doc");
        for (id, s, r, v) in [
            ("f0", "Borujerdi", "country of citizenship", "Iran"),
            ("f1", "Iran", "head of government", "Raisi"),
            ("f2", "Raisi", "country of citizenship", "Iran"),
        ] {
            let fi = FactIn::text(id, s, r, v, ("d", "v1"), (0, 4), "abcd");
            mem.add_fact(&fi, &[]).expect("fact");
        }
        let r = |id: &str| mem.store.rec_of(id).expect("record");
        let recs = [r("f0"), r("f1"), r("f2")];
        (mem, recs)
    }

    /// the hop-manifest slot of record `r` read forward (panics when it is not offered)
    fn slot(ep: &mut Episode<'_>, r: RecId) -> Option<i64> {
        let h = ep.current_hop();
        (0..h.len())
            .find(|&j| h.cands[j] == r && !h.is_inverse(j))
            .map(|j| j as i64)
    }

    /// LOOKUP f0, CHAIN f1, CHAIN f2, then CHAIN f1 again: the program the shared-store loops of E8 run
    fn run_cycle(ep: &mut Episode<'_>, recs: [RecId; 3]) {
        ep.step(LOOKUP, Some(0)).unwrap();
        let p = slot(ep, recs[1]).expect("f1 offered");
        ep.step(CHAIN, Some(p)).unwrap();
        let p = slot(ep, recs[2]).expect("f2 offered");
        ep.step(CHAIN, Some(p)).unwrap();
        assert!(!ep.done);
        let p = slot(ep, recs[1]).expect("f1 offered again");
        ep.step(CHAIN, Some(p)).unwrap();
    }

    #[test]
    fn the_loop_guard_ends_a_cycle_and_off_is_the_reference() {
        let (mem, recs) = cycle_memory();
        let snap = mem.head();
        let mk = |g: LoopGuard| {
            let mut ep = Episode::new_v5(&mem, snap, vec![recs[0]], vec![false], "t/loop", 16);
            ep.loop_guard = g;
            ep
        };
        // reference: the re-read runs (Raisi again), the episode goes on towards MAX_STEPS
        let mut off = mk(LoopGuard::Off);
        run_cycle(&mut off, recs);
        assert!(!off.done && off.loop_stopped.is_none());
        assert_eq!(off.trace[3].result.as_deref(), Some("VALUE text Raisi"));
        // unknown: the re-read ends the episode with UNKNOWN
        let mut unk = mk(LoopGuard::Unknown);
        run_cycle(&mut unk, recs);
        assert!(unk.done);
        let a = unk.answer.clone().unwrap();
        assert_eq!((a.status, a.answer.as_deref()), ("UNKNOWN", None));
        assert!(a.error.as_deref().unwrap().starts_with("loop guard"));
        assert_eq!(unk.loop_stopped, Some((3, 1)));
        assert_eq!(a.program.len(), 4, "the guarded action is recorded");
        // first_use: the re-read ends the episode with the value step 1 produced and its evidence
        let mut fu = mk(LoopGuard::FirstUse);
        run_cycle(&mut fu, recs);
        let a = fu.answer.clone().unwrap();
        assert_eq!((a.status, a.answer.as_deref()), ("ANSWER", Some("Raisi")));
        let mut ev = vec![recs[0], recs[1]];
        ev.sort_unstable_by(|&x, &y| mem.store.fact_id(x).cmp(mem.store.fact_id(y)));
        assert_eq!(a.evidence_ids, ev);
        // identical up to the guarded action
        assert_eq!(fu.trace[..3], off.trace[..3]);
        assert_eq!(unk.trace[..3], off.trace[..3]);
    }

    #[test]
    fn a_coherent_scope_hides_hop_records_of_other_collections() {
        let (mem, recs) = cycle_memory();
        let snap = mem.head();
        // f0 in collection 1, f1 in 1 and 2, f2 in 2 only (written once for two worlds, another world's fact)
        let mut of: Vec<SmallVec<[u32; 2]>> = vec![SmallVec::new(); mem.store.num_records()];
        of[recs[0] as usize] = SmallVec::from_slice(&[1]);
        of[recs[1] as usize] = SmallVec::from_slice(&[1, 2]);
        of[recs[2] as usize] = SmallVec::from_slice(&[2]);
        let map = Arc::new(RecCollections::new(of));
        for (mode, hidden) in [
            (None, false),
            (Some(ScopeMode::Coherent), true),
            (Some(ScopeMode::Filter(vec![1])), true),
            (Some(ScopeMode::Filter(vec![1, 2])), false),
            (Some(ScopeMode::Prefer(vec![1])), false),
        ] {
            let mut ep = Episode::new_v5(&mem, snap, vec![recs[0]], vec![false], "t/scope", 16);
            ep.scope = mode.clone().map(|m| {
                Arc::new(QuestionScope {
                    map: map.clone(),
                    mode: m,
                })
            });
            ep.step(LOOKUP, Some(0)).unwrap();
            let p = slot(&mut ep, recs[1]).expect("f1 offered");
            ep.step(CHAIN, Some(p)).unwrap();
            // the value is Raisi: f2 (Raisi's own fact, collection 2 only) resolves forward
            assert_eq!(slot(&mut ep, recs[2]).is_none(), hidden, "{mode:?}");
        }
    }
}
