//! decline_v1/replay.py over the exact executor: the operand reads of a program (which subject each LOOKUP read) and the repair
//! (the two operand reads swapped, executed by a new `Episode` on the same manifest, so a repaired answer comes from the executor,
//! never from a rewrite of the answer text).

use super::{Read, Replay, Row, Step, Swapped, TraceStep};
use xerj_exec::{Answer, Episode, Protocol};
use xerj_index::Memory;
use xerj_store::{RecId, Snapshot};

/// the row of a finished stack-v5 episode
pub fn row_of(question: &str, ans: &Answer, mem: &Memory) -> Row {
    let store = &mem.store;
    Row {
        question: question.to_string(),
        status: ans.status.to_string(),
        answer: ans.answer.clone(),
        program: ans
            .program
            .iter()
            .map(|a| Step {
                op: a.op.clone(),
                pointer: a.pointer,
            })
            .collect(),
        trace: ans
            .execution_trace
            .iter()
            .map(|t| TraceStep {
                op: t.op.clone(),
                pointer: t.pointer,
                result: t.result.clone(),
                hop_candidate_ids: t
                    .hop_candidate_ids
                    .as_ref()
                    .map(|h| h.iter().map(|&r| store.fact_id(r).to_string()).collect()),
            })
            .collect(),
        evidence_ids: ans
            .evidence_ids
            .iter()
            .map(|&r| store.fact_id(r).to_string())
            .collect(),
    }
}

/// replay.swapped_program: [LOOKUP a, LOOKUP b, CMP, (NOT,) RETURN] -> [LOOKUP b, LOOKUP a, CMP, (NOT,) RETURN]; None for any other shape
pub fn swapped_program(program: &[Step]) -> Option<Vec<Step>> {
    let core: Vec<&Step> = program.iter().filter(|s| s.op != "RETURN").collect();
    if core.len() < 3
        || core[0].op != "LOOKUP"
        || core[1].op != "LOOKUP"
        || !matches!(core[2].op.as_str(), "LT" | "GT")
    {
        return None;
    }
    if core.len() > 4 || (core.len() == 4 && core[3].op != "NOT") {
        return None;
    }
    if program.last()?.op != "RETURN" {
        return None;
    }
    let mut sw = vec![core[1].clone(), core[0].clone()];
    sw.extend(core[2..].iter().map(|s| (*s).clone()));
    sw.push(Step {
        op: "RETURN".to_string(),
        pointer: None,
    });
    Some(sw)
}

/// The finished episode of a question and how to start another on the same manifest.
pub struct EpisodeReplay<'a, 'm> {
    pub ep: &'a Episode<'m>,
    pub mem: &'m Memory,
    pub snap: Snapshot<'m>,
    pub cands: &'a [RecId],
    pub inverse: &'a [bool],
    pub item_id: &'a str,
    pub k: usize,
}

impl<'m> EpisodeReplay<'_, 'm> {
    fn fresh(&self) -> Episode<'m> {
        let mut e = Episode::new_v5(
            self.mem,
            self.snap,
            self.cands.to_vec(),
            self.inverse.to_vec(),
            self.item_id,
            self.k,
        );
        e.max_inverse = self.ep.max_inverse;
        e.rel_pref = self.ep.rel_pref.clone();
        e.scope = self.ep.scope.clone();
        e.loop_guard = self.ep.loop_guard;
        e
    }
}

impl Replay for EpisodeReplay<'_, '_> {
    /// the engine's row IS this episode: there is nothing to replay
    fn matches(&mut self, _row: &Row) -> bool {
        true
    }

    fn operand_reads(&mut self, row: &Row) -> Vec<Read> {
        let store = &self.mem.store;
        let mut out = Vec::new();
        for (t, s) in row.program.iter().enumerate() {
            let Some(tr) = self.ep.trace.get(t) else {
                break;
            };
            if !matches!(s.op.as_str(), "LOOKUP" | "CHAIN") {
                continue;
            }
            let (mut fid, mut rel) = (None, None);
            if let Some(p) = s.pointer.filter(|&p| p >= 0).map(|p| p as usize) {
                if s.op == "LOOKUP" {
                    if let Some(&r) = self.cands.get(p) {
                        fid = Some(store.fact_id(r).to_string());
                        rel = Some(store.relation(r).to_string());
                    }
                } else if let Some(&r) = tr.hop_candidate_ids.as_ref().and_then(|h| h.get(p)) {
                    let f = store.fact_id(r).to_string();
                    let last = f.rsplit('/').next().unwrap_or("");
                    rel = Some(last.split('#').next().unwrap_or("").replace('-', " "));
                    fid = Some(f);
                }
            }
            out.push(Read {
                step: t,
                op: s.op.clone(),
                subject: self.ep.origin_subject(t).map(str::to_string),
                relation: rel,
                fact_id: fid,
                result: tr.result.clone(),
            });
        }
        out
    }

    fn execute_swapped(&mut self, row: &Row) -> Option<Swapped> {
        let sw = swapped_program(&row.program)?;
        let mut e = self.fresh();
        for s in &sw {
            if e.done {
                break;
            }
            let op = Protocol::V5.op_id(&s.op)?;
            e.step(op, s.pointer).ok()?;
        }
        let a = e.answer.as_ref();
        Some(Swapped {
            status: a.map_or("", |x| x.status).to_string(),
            answer: a.and_then(|x| x.answer.clone()),
            evidence_ids: a.map_or(Vec::new(), |x| {
                x.evidence_ids
                    .iter()
                    .map(|&r| self.mem.store.fact_id(r).to_string())
                    .collect()
            }),
            program: sw,
        })
    }
}
