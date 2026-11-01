//! The exact executor (xerj-exec `Episode`, stack-v1 / stack-v2 / stack-v5) behind xerj-infer's `ActionExecutor`: the glue that
//! was a fixture replay until now. Candidate strings are `xr_codecs.candidate_fields`, the history is the episode's own execution
//! trace, the rrt registers are built from the executed stack, and the stack-v2 / v5 hop manifest is the episode's `current_hop`.
//! Stack-v5: inverse hop candidates carry the `<= ` flag and executed inverse steps the ` IN` history marker (base-manifest
//! flags are the caller's: `CandManifest::inverse`).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;
use xerj_exec::{Episode, Res};
use xerj_index::rel_phrase;
use xerj_infer::codec::{StackEntry, Status as CStatus, TypedValue as CValue};
use xerj_infer::episode::{ActionExecutor, HopView, N_OPS};
use xerj_infer::inputs::{CandidateFields, HistoryStep};
use xerj_store::value::render_value;
use xerj_store::{Kind, RecId, Status, Store};

pub fn kind_name(k: Kind) -> &'static str {
    match k {
        Kind::Text => "text",
        Kind::Date => "date",
        Kind::Number => "number",
        Kind::Boolean => "boolean",
    }
}

/// xr_codecs.candidate_fields: (subject, relation phrase, "kind: rendered value", quote)
pub fn candidate_fields(store: &Store, r: RecId) -> CandidateFields {
    let v = store.typed_value(r);
    CandidateFields {
        subject: store.subject(r).to_string(),
        relation: rel_phrase(store.relation(r)).to_string(),
        value: format!("{}: {}", kind_name(v.kind), render_value(&v)),
        quote: store.quote(r).to_string(),
        inverse: false, // forward; stack-v5 callers flag inverse slots with `with_inverse`
    }
}

fn stack_entry(r: &Res<'_>, step: usize) -> StackEntry {
    let status = match r.status {
        Status::Value => CStatus::Value,
        Status::True => CStatus::True,
        Status::False => CStatus::False,
        Status::Unknown => CStatus::Unknown,
        Status::Conflict => CStatus::Conflict,
    };
    let value = match (r.status, r.value) {
        (Status::Value, Some(v)) => Some(match v.kind {
            Kind::Text => CValue::Text(v.value.to_string()),
            Kind::Date => CValue::Date(v.value.to_string()),
            Kind::Number => CValue::Number(v.value.to_string()),
            Kind::Boolean => CValue::Boolean(v.value == "true"),
        }),
        _ => None,
    };
    StackEntry {
        status,
        value,
        step,
    }
}

fn fnv(s: &str) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// xerj-exec Episode as an xerj-infer ActionExecutor
pub struct ExecAdapter<'m> {
    pub ep: Episode<'m>,
    hist: Vec<HistoryStep>,
    stack: Vec<StackEntry>,
    hop_fields: Vec<CandidateFields>,
    hop_resolves: Vec<bool>,
    hop_key: u64,
    /// per decision that read a hop manifest: (decision index, hop candidate fact ids)
    pub hop_reads: Vec<(usize, Vec<RecId>)>,
    hop_ids: Vec<RecId>,
    /// time spent inside the exact executor (steps + hop manifests), microseconds
    pub exec_us: f64,
    /// The host's request to stop (its deadline passed, or its client went away). [`ActionExecutor::done`] reports it, so
    /// the free-running episode ends before its next decision. `None` (the reference behaviour) = never.
    stop: Option<Arc<AtomicBool>>,
}

impl<'m> ExecAdapter<'m> {
    pub fn new(ep: Episode<'m>) -> Self {
        let mut a = ExecAdapter {
            ep,
            hist: Vec::new(),
            stack: Vec::new(),
            hop_fields: Vec::new(),
            hop_resolves: Vec::new(),
            hop_key: 0,
            hop_reads: Vec::new(),
            hop_ids: Vec::new(),
            exec_us: 0.0,
            stop: None,
        };
        a.refresh();
        a
    }

    /// End the episode before its next decision once `stop` is set (checked only between decisions: a decision in
    /// progress completes). While `stop` stays unset every decision is the one [`ExecAdapter::new`] gives.
    pub fn with_stop(mut self, stop: Arc<AtomicBool>) -> Self {
        self.stop = Some(stop);
        self
    }

    /// The host asked the episode to stop and it had not finished: its answer record is incomplete and must not be
    /// served (not even as an UNKNOWN).
    pub fn stopped(&self) -> bool {
        !self.ep.done && self.stop_requested()
    }

    fn stop_requested(&self) -> bool {
        self.stop
            .as_ref()
            .is_some_and(|s| s.load(Ordering::Acquire))
    }

    fn refresh(&mut self) {
        self.stack = self
            .ep
            .stack
            .iter()
            .map(|(r, s)| stack_entry(r, *s))
            .collect();
        self.hop_fields.clear();
        self.hop_resolves.clear();
        self.hop_ids.clear();
        self.hop_key = 0;
        if self.ep.protocol.has_hops() && !self.ep.done {
            let h = self.ep.current_hop();
            if !h.cands.is_empty() {
                let store = self.ep.snap.store;
                self.hop_fields = h
                    .cands
                    .iter()
                    .enumerate()
                    .map(|(j, &r)| candidate_fields(store, r).with_inverse(h.is_inverse(j)))
                    .collect();
                self.hop_resolves = h.resolves.clone();
                self.hop_ids = h.cands.clone();
                let mut s = h.anchor.clone().unwrap_or_default();
                s.push('#');
                for (j, &r) in h.cands.iter().enumerate() {
                    if h.is_inverse(j) {
                        s.push('<');
                    }
                    s.push_str(store.fact_id(r));
                    s.push(';');
                }
                self.hop_key = fnv(&s) | 1;
            }
        }
    }
}

impl ActionExecutor for ExecAdapter<'_> {
    fn legal_ops(&self) -> [bool; N_OPS] {
        self.ep.legal_ops()
    }

    fn step(&mut self, op: usize, pointer: Option<usize>) -> xerj_infer::Result<()> {
        let t0 = Instant::now();
        if !self.hop_ids.is_empty() {
            self.hop_reads.push((self.hist.len(), self.hop_ids.clone()));
        }
        let n = self.ep.trace.len();
        self.ep
            .step(op as i64, pointer.map(|p| p as i64))
            .map_err(|e| xerj_infer::Error::Executor(e.to_string()))?;
        match self.ep.trace.get(n) {
            Some(t) => self.hist.push(HistoryStep {
                op: t.op.clone(),
                pointer: t.pointer.map(|p| p.max(0) as usize),
                result: t.result.clone(),
                inverse: t.inverse,
            }),
            None => {
                return Err(xerj_infer::Error::Executor(
                    "executor did not record the action".into(),
                ))
            }
        }
        self.refresh();
        self.exec_us += t0.elapsed().as_secs_f64() * 1e6;
        Ok(())
    }

    fn done(&self) -> bool {
        self.ep.done || self.stop_requested()
    }

    fn trace(&self) -> &[HistoryStep] {
        &self.hist
    }

    fn stack(&self) -> &[StackEntry] {
        &self.stack
    }

    fn hop(&self) -> Option<HopView<'_>> {
        if self.hop_fields.is_empty() {
            None
        } else {
            Some(HopView {
                cands: &self.hop_fields,
                resolves: &self.hop_resolves,
                key: self.hop_key,
            })
        }
    }
}
