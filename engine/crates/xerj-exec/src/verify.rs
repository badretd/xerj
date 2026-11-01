//! The deterministic verify step (research/xerj_r_mvp/tools/verify_rows.py), label-free: an ANSWER becomes UNKNOWN when
//! (a) a cited record's subject string does not occur in the question, or (b) the question names two or more store
//! subjects but the answer cites fewer than two distinct subjects.
//!
//! `VerifyMode::Substring` is the reference rule (plain `s in q` over every subject of the store's records, any state).
//! In a large merged store plain substrings over-trigger (b) (short subject names occur inside other words), so
//! `VerifyMode::Linked` counts the snapshot subjects the exact linker finds (word boundaries) instead; it equals the
//! reference on stores whose subjects never occur inside other words.

use crate::episode::Answer;
use xerj_index::{Memory, QBuf};
use xerj_store::Snapshot;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VerifyMode {
    Substring,
    Linked,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Verdict {
    pub vetoed: bool,
    /// "cited subject not in question" | "question names >= 2 subjects, answer cites < 2" | ""
    pub reason: &'static str,
}

/// Apply the rule to an answer record (fact ids resolved through the memory's store).
pub fn verify(
    mem: &Memory,
    snap: &Snapshot<'_>,
    question: &str,
    ans: &Answer,
    mode: VerifyMode,
    buf: &mut QBuf,
) -> Verdict {
    if ans.status != "ANSWER" {
        return Verdict {
            vetoed: false,
            reason: "",
        };
    }
    let store = &mem.store;
    let mut subs: smallvec::SmallVec<[&str; 4]> = smallvec::SmallVec::new();
    for &r in &ans.evidence_ids {
        let s = store.subject(r);
        if !subs.contains(&s) {
            subs.push(s);
        }
    }
    if subs.iter().any(|s| !question.contains(s)) {
        return Verdict {
            vetoed: true,
            reason: "cited subject not in question",
        };
    }
    if subs.len() < 2 {
        let named = match mode {
            VerifyMode::Substring => mem.linker.substring_subjects(store, question, buf, 2).len(),
            VerifyMode::Linked => mem.link(snap, question, buf).len(),
        };
        if named >= 2 {
            return Verdict {
                vetoed: true,
                reason: "question names >= 2 subjects, answer cites < 2",
            };
        }
    }
    Verdict {
        vetoed: false,
        reason: "",
    }
}

/// The verified answer record: vetoed answers become UNKNOWN (no answer; the cited evidence is kept for audit).
pub fn apply(ans: &Answer, v: &Verdict) -> Answer {
    if !v.vetoed {
        return ans.clone();
    }
    let mut a = ans.clone();
    a.status = "UNKNOWN";
    a.answer = None;
    a.error = Some(format!("verify: {}", v.reason));
    a
}
