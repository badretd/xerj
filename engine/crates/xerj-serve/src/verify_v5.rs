//! The verify rule for stack-v5 (direction) programs. This is an EXTENSION, not reference behaviour.
//!
//! `tools/verify_rows.py` (`xerj_exec::verify`) vetoes an ANSWER whose cited record's subject does not occur in the question. For
//! an inverse step (LOOKUP or CHAIN on a `<=` candidate) the cited record's subject IS the answer, so the reference rule vetoes
//! every correct inverse answer: 1,422 of 1,422 gold inverse answers in the v5 parity fixtures (2026-10-26). The production v5
//! evaluation applies no verify at all. This rule extends `verify_chain_aware` the same way that rule extended the reference
//! for CHAIN:
//! - (a) The subject of every record read by a CHAIN step (forward or inverse) or by an inverse LOOKUP is exempt from "cited
//!   subject in the question". An inverse LOOKUP's record must instead have its VALUE in the question: the name the question
//!   starts from. An inverse CHAIN's value is the previous hop's result, exempt like the forward bridge.
//! - (b) A question that names >= 2 store subjects needs >= 2 distinct cited names. An inverse record contributes its value
//!   next to the cited subjects.
//!
//! The answer record itself is never changed: callers report the raw status/answer and the verdict separately.

use xerj_exec::{Answer, Verdict};
use xerj_index::{Memory, QBuf};
use xerj_store::RecId;

pub const RULE: &str = "direction-aware (v5 extension of verify_chain_aware)";

/// `cands` = the base manifest the episode ran on (an inverse LOOKUP's pointer indexes it).
pub fn verify_v5(
    mem: &Memory,
    question: &str,
    cands: &[RecId],
    ans: &Answer,
    buf: &mut QBuf,
) -> Verdict {
    if ans.status != "ANSWER" {
        return Verdict {
            vetoed: false,
            reason: "",
        };
    }
    let store = &mem.store;
    let mut exempt: Vec<&str> = Vec::new();
    let mut inverse_values: Vec<&str> = Vec::new();
    for t in &ans.execution_trace {
        if t.illegal {
            continue;
        }
        let Some(p) = t.pointer.filter(|&p| p >= 0).map(|p| p as usize) else {
            continue;
        };
        let rec = match t.op.as_str() {
            "CHAIN" => t.hop_candidate_ids.as_ref().and_then(|h| h.get(p)).copied(),
            "LOOKUP" if t.inverse => cands.get(p).copied(),
            _ => None,
        };
        let Some(r) = rec else { continue };
        exempt.push(store.subject(r));
        if t.inverse {
            let v = store.typed_value(r).value;
            if t.op == "LOOKUP" && !question.contains(v) {
                return Verdict {
                    vetoed: true,
                    reason: "inverse record's value not in question",
                };
            }
            inverse_values.push(v);
        }
    }
    let mut names: Vec<&str> = Vec::new();
    for &r in &ans.evidence_ids {
        let s = store.subject(r);
        if !names.contains(&s) {
            names.push(s);
        }
    }
    if names
        .iter()
        .any(|s| !question.contains(s) && !exempt.contains(s))
    {
        return Verdict {
            vetoed: true,
            reason: "cited subject not in question (nor reached by CHAIN or an inverse step)",
        };
    }
    for v in inverse_values {
        if !names.contains(&v) {
            names.push(v);
        }
    }
    if names.len() < 2 && mem.linker.substring_subjects(store, question, buf, 2).len() >= 2 {
        return Verdict {
            vetoed: true,
            reason: "question names >= 2 subjects, answer cites < 2",
        };
    }
    Verdict {
        vetoed: false,
        reason: "",
    }
}
