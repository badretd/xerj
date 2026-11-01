//! Golden fixtures exported from PyTorch (`research/xerj_r_mvp/tools/xerj_infer_golden.py` for stack v1 / v2 and
//! `tools/xerj_infer_golden_v5.py` for stack v5, format `xerj-infer-golden-v1`; the v5 tool adds the optional keys
//! `cands_inverse`, hop `inverse` and trace `direction`) and the replay executor that stands in for the exact executor in parity
//! tests and benchmarks.
//!
//! A fixture row is one question: the base candidate manifest, and per decision the exact input ids, spans, hop manifest,
//! memory / register tensors, the typed stack, the legality mask, op / pointer logits, the chosen action and the trace entry
//! the Python executor produced. The replay executor answers `legal_ops` / `stack` / `hop` / `trace` from the recorded episode
//! and accepts an action only if it equals the recorded one (free-running mode: the episode stops at the first divergence;
//! forced mode: the recorded action is applied anyway and the mismatch is counted).

use crate::codec::{StackEntry, Status, TypedValue};
use crate::episode::{ActionExecutor, HopView, N_OPS};
use crate::error::{Error, Result};
use crate::inputs::{CandidateFields, HistoryStep};
use serde_json::Value;
use std::io::BufRead;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct HopFixture {
    pub cands: Vec<CandidateFields>,
    pub resolves: Vec<bool>,
    pub anchor: String,
    pub key: u64,
}

#[derive(Debug, Clone)]
pub struct FixtureDecision {
    pub step: usize,
    pub ids: Vec<u32>,
    pub spans: Option<Vec<(usize, usize)>>,
    pub hop_spans: Option<Vec<(usize, usize)>>,
    pub hop: Option<HopFixture>,
    pub hop_memory: Option<Vec<f32>>,
    pub registers: Option<(Vec<f32>, Vec<bool>)>,
    pub stack: Vec<StackEntry>,
    pub legal: [bool; N_OPS],
    pub op_logits: Vec<f32>,
    pub ptr_logits: Vec<f32>,
    pub op: usize,
    pub pointer: Option<usize>,
    pub trace: HistoryStep,
}

#[derive(Debug, Clone)]
pub struct FinalRecord {
    pub status: String,
    pub answer: Option<String>,
    pub evidence_ids: Vec<String>,
    pub error: Option<String>,
}

#[derive(Debug, Clone)]
pub struct FixtureRow {
    pub set: String,
    pub item_id: String,
    pub question: String,
    pub arm: String,
    pub stack: String,
    pub k: usize,
    pub cands: Vec<CandidateFields>,
    pub memory: Option<Vec<f32>>,
    pub decisions: Vec<FixtureDecision>,
    pub final_record: FinalRecord,
    /// Python answer_item latency (manifest + codec + decoder + executor) and time inside arm.decide, seconds
    pub py_latency_s: f64,
    pub py_decide_s: f64,
}

pub fn b64decode(s: &str) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(s.len() * 3 / 4);
    let mut buf = 0u32;
    let mut nbits = 0;
    for c in s.bytes() {
        let v = match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            b'=' => break,
            b'\n' | b'\r' | b' ' => continue,
            _ => return Err(Error::Format(format!("bad base64 byte {c}"))),
        } as u32;
        buf = (buf << 6) | v;
        nbits += 6;
        if nbits >= 8 {
            nbits -= 8;
            out.push((buf >> nbits) as u8);
            buf &= (1 << nbits) - 1;
        }
    }
    Ok(out)
}

fn f32s(v: &Value) -> Result<Vec<f32>> {
    let s = v
        .as_str()
        .ok_or_else(|| Error::Format("expected a base64 string".into()))?;
    let b = b64decode(s)?;
    Ok(b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

/// candidate rows `[subject, relation, value, quote]`; `inverse` = the stack-v5 flags (absent = all forward)
fn cands(v: &Value, inverse: Option<&Value>) -> Result<Vec<CandidateFields>> {
    let arr = v.as_array().ok_or_else(|| Error::Format("cands".into()))?;
    let inv: Vec<bool> = inverse
        .and_then(|x| x.as_array())
        .map(|a| a.iter().map(|b| b.as_bool().unwrap_or(false)).collect())
        .unwrap_or_default();
    if !inv.is_empty() && inv.len() != arr.len() {
        return Err(Error::Format(format!(
            "{} inverse flags for {} candidates",
            inv.len(),
            arr.len()
        )));
    }
    arr.iter()
        .enumerate()
        .map(|(i, f)| {
            let a = f
                .as_array()
                .ok_or_else(|| Error::Format("cand fields".into()))?;
            let s = |i: usize| a.get(i).and_then(|x| x.as_str()).unwrap_or("").to_string();
            Ok(CandidateFields {
                subject: s(0),
                relation: s(1),
                value: s(2),
                quote: s(3),
                inverse: inv.get(i).copied().unwrap_or(false),
            })
        })
        .collect()
}

fn spans(v: &Value) -> Option<Vec<(usize, usize)>> {
    v.as_array().map(|a| {
        a.iter()
            .filter_map(|p| {
                p.as_array().map(|x| {
                    (
                        x[0].as_u64().unwrap_or(0) as usize,
                        x[1].as_u64().unwrap_or(0) as usize,
                    )
                })
            })
            .collect()
    })
}

fn stack_entry(v: &Value) -> Result<StackEntry> {
    let status = Status::parse(v.get("status").and_then(|s| s.as_str()).unwrap_or(""))
        .ok_or_else(|| Error::Format("stack status".into()))?;
    let kind = v.get("kind").and_then(|k| k.as_str());
    let val = v.get("value");
    let value = match kind {
        None => None,
        Some("text") => Some(TypedValue::Text(
            val.and_then(|x| x.as_str()).unwrap_or("").into(),
        )),
        Some("date") => Some(TypedValue::Date(
            val.and_then(|x| x.as_str()).unwrap_or("").into(),
        )),
        Some("number") => Some(TypedValue::Number(
            val.and_then(|x| x.as_str()).unwrap_or("0").into(),
        )),
        Some("boolean") => Some(TypedValue::Boolean(
            val.and_then(|x| x.as_bool()).unwrap_or(false),
        )),
        Some(k) => return Err(Error::Format(format!("value kind {k}"))),
    };
    Ok(StackEntry {
        status,
        value,
        step: v.get("step").and_then(|s| s.as_u64()).unwrap_or(0) as usize,
    })
}

fn fnv(s: &str) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h | 1
}

fn opt_usize(v: Option<&Value>) -> Option<usize> {
    v.and_then(|x| x.as_u64()).map(|x| x as usize)
}

pub fn parse_row(line: &str) -> Result<FixtureRow> {
    let v: Value =
        serde_json::from_str(line).map_err(|e| Error::Format(format!("fixture json: {e}")))?;
    let gs = |k: &str| v.get(k).and_then(|x| x.as_str()).unwrap_or("").to_string();
    let mut decisions = Vec::new();
    for d in v
        .get("decisions")
        .and_then(|x| x.as_array())
        .ok_or_else(|| Error::Format("decisions".into()))?
    {
        let ids = d
            .get("ids")
            .and_then(|x| x.as_array())
            .ok_or_else(|| Error::Format("ids".into()))?
            .iter()
            .map(|x| x.as_u64().unwrap_or(0) as u32)
            .collect();
        let hop = match d.get("hop") {
            Some(h) if !h.is_null() => {
                let anchor = h
                    .get("anchor")
                    .and_then(|x| x.as_str())
                    .unwrap_or("")
                    .to_string();
                let c = cands(h.get("cands").unwrap_or(&Value::Null), h.get("inverse"))?;
                let ids: String = c
                    .iter()
                    .map(|f| {
                        format!(
                            "{}{}|{}|{}|{};",
                            if f.inverse { "<=" } else { "" },
                            f.subject,
                            f.relation,
                            f.value,
                            f.quote
                        )
                    })
                    .collect();
                Some(HopFixture {
                    key: fnv(&format!("{anchor}#{ids}")),
                    cands: c,
                    resolves: h
                        .get("resolves")
                        .and_then(|x| x.as_array())
                        .map(|a| a.iter().map(|b| b.as_bool().unwrap_or(false)).collect())
                        .unwrap_or_default(),
                    anchor,
                })
            }
            _ => None,
        };
        let registers = match d.get("registers") {
            Some(r) if !r.is_null() => Some((
                f32s(r.get("feats").unwrap_or(&Value::Null))?,
                r.get("mask")
                    .and_then(|x| x.as_array())
                    .map(|a| a.iter().map(|b| b.as_bool().unwrap_or(false)).collect())
                    .unwrap_or_default(),
            )),
            _ => None,
        };
        let mut legal = [false; N_OPS];
        if let Some(a) = d.get("legal").and_then(|x| x.as_array()) {
            for (i, b) in a.iter().enumerate().take(N_OPS) {
                legal[i] = b.as_bool().unwrap_or(false);
            }
        }
        let tr = d
            .get("trace")
            .ok_or_else(|| Error::Format("trace".into()))?;
        decisions.push(FixtureDecision {
            step: d.get("step").and_then(|x| x.as_u64()).unwrap_or(0) as usize,
            ids,
            spans: d.get("spans").and_then(spans),
            hop_spans: d.get("hop_spans").and_then(spans),
            hop,
            hop_memory: match d.get("hop_memory") {
                Some(h) if !h.is_null() => Some(f32s(h)?),
                _ => None,
            },
            registers,
            stack: d
                .get("stack")
                .and_then(|x| x.as_array())
                .map(|a| a.iter().map(stack_entry).collect::<Result<Vec<_>>>())
                .transpose()?
                .unwrap_or_default(),
            legal,
            op_logits: f32s(d.get("op_logits").unwrap_or(&Value::Null))?,
            ptr_logits: f32s(d.get("ptr_logits").unwrap_or(&Value::Null))?,
            op: d.get("op").and_then(|x| x.as_u64()).unwrap_or(0) as usize,
            pointer: opt_usize(d.get("pointer")),
            trace: HistoryStep {
                op: tr
                    .get("op")
                    .and_then(|x| x.as_str())
                    .unwrap_or("")
                    .to_string(),
                pointer: opt_usize(tr.get("pointer")),
                result: tr.get("result").and_then(|x| x.as_str()).map(String::from),
                inverse: tr.get("direction").and_then(|x| x.as_str()) == Some("in"),
            },
        });
    }
    let fin = v
        .get("final")
        .ok_or_else(|| Error::Format("final".into()))?;
    let py = v.get("python");
    Ok(FixtureRow {
        set: gs("set"),
        item_id: gs("item_id"),
        question: gs("question"),
        arm: gs("arm"),
        stack: gs("stack"),
        k: v.get("K").and_then(|x| x.as_u64()).unwrap_or(16) as usize,
        cands: cands(
            v.get("cands").unwrap_or(&Value::Null),
            v.get("cands_inverse"),
        )?,
        memory: match v.get("memory") {
            Some(m) if !m.is_null() => Some(f32s(m)?),
            _ => None,
        },
        decisions,
        final_record: FinalRecord {
            status: fin
                .get("status")
                .and_then(|x| x.as_str())
                .unwrap_or("")
                .to_string(),
            answer: fin.get("answer").and_then(|x| x.as_str()).map(String::from),
            evidence_ids: fin
                .get("evidence_ids")
                .and_then(|x| x.as_array())
                .map(|a| {
                    a.iter()
                        .filter_map(|s| s.as_str().map(String::from))
                        .collect()
                })
                .unwrap_or_default(),
            error: fin.get("error").and_then(|x| x.as_str()).map(String::from),
        },
        py_latency_s: py
            .and_then(|p| p.get("latency_s"))
            .and_then(|x| x.as_f64())
            .unwrap_or(0.0),
        py_decide_s: py
            .and_then(|p| p.get("decide_s"))
            .and_then(|x| x.as_f64())
            .unwrap_or(0.0),
    })
}

impl FixtureRow {
    /// drop the parity-only payload (memory tensors, logits, ids) and keep what the replay executor needs (benchmarks)
    pub fn strip(&mut self) {
        self.memory = None;
        for d in &mut self.decisions {
            d.ids = Vec::new();
            d.hop_memory = None;
            d.registers = None;
            d.op_logits = Vec::new();
            d.ptr_logits = Vec::new();
            d.spans = None;
            d.hop_spans = None;
        }
    }
}

pub fn load(path: &Path) -> Result<Vec<FixtureRow>> {
    load_opt(path, false)
}

/// load rows; `strip`: keep only the replay data (each row stripped right after parsing)
pub fn load_opt(path: &Path, strip: bool) -> Result<Vec<FixtureRow>> {
    let f = std::fs::File::open(path).map_err(|e| Error::Io(format!("{}: {e}", path.display())))?;
    let mut rows = Vec::new();
    for line in std::io::BufReader::new(f).lines() {
        let line = line.map_err(|e| Error::Io(e.to_string()))?;
        if line.trim().is_empty() {
            continue;
        }
        let mut r = parse_row(&line)?;
        if strip {
            r.strip();
        }
        rows.push(r);
    }
    Ok(rows)
}

/// replays the recorded executor of one fixture row
pub struct ReplayExecutor<'a> {
    row: &'a FixtureRow,
    t: usize,
    forced: bool,
    trace: Vec<HistoryStep>,
    /// first decision whose action differed from the recorded one
    pub first_divergence: Option<usize>,
    pub mismatches: usize,
    pub actions: Vec<(usize, Option<usize>)>,
}

impl<'a> ReplayExecutor<'a> {
    pub fn new(row: &'a FixtureRow, forced: bool) -> Self {
        ReplayExecutor {
            row,
            t: 0,
            forced,
            trace: Vec::new(),
            first_divergence: None,
            mismatches: 0,
            actions: Vec::new(),
        }
    }
    /// the recorded final answer is the episode's answer iff no action diverged
    pub fn identical_episode(&self) -> bool {
        self.first_divergence.is_none() && self.t == self.row.decisions.len()
    }
    pub fn position(&self) -> usize {
        self.t
    }
}

impl ActionExecutor for ReplayExecutor<'_> {
    fn legal_ops(&self) -> [bool; N_OPS] {
        self.row.decisions[self.t].legal
    }
    fn step(&mut self, op: usize, pointer: Option<usize>) -> Result<()> {
        let rec = &self.row.decisions[self.t];
        self.actions.push((op, pointer));
        if (op, pointer) != (rec.op, rec.pointer) {
            self.mismatches += 1;
            if self.first_divergence.is_none() {
                self.first_divergence = Some(self.t);
            }
            if !self.forced {
                self.t = self.row.decisions.len();
                return Ok(());
            }
        }
        self.trace.push(rec.trace.clone());
        self.t += 1;
        Ok(())
    }
    fn done(&self) -> bool {
        self.t >= self.row.decisions.len()
    }
    fn trace(&self) -> &[HistoryStep] {
        &self.trace
    }
    fn stack(&self) -> &[StackEntry] {
        &self.row.decisions[self.t].stack
    }
    fn hop(&self) -> Option<HopView<'_>> {
        self.row.decisions[self.t].hop.as_ref().map(|h| HopView {
            cands: &h.cands,
            resolves: &h.resolves,
            key: h.key,
        })
    }
}
