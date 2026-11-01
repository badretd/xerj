//! Decline policy v2 of the XERJ-R decline layer in Rust (e2e1, 2026-10-31): P3 (scope guard + copy guard), D2 (question-program
//! consistency) and D2b (role / direction check, variant "repair"). A port, not a redesign: the frozen Python
//! (`tools/hybrid_bench_policy.py`, `decline_v1/{rules,roles,replay,policy}.py`, `runs/decline_v1/frozen_rules_v2.json`) is the
//! oracle, and the parity tools compare every decision with it. The D1 threshold of v2 is not here: it reads a confidence export
//! that FP32 checkpoints do not have (decline report section 9.6).
//!
//! Opt-in per question (`VerifyRule::DeclineV2`, the `_ask` option `verify: "decline_v2"`); the default response is unchanged. In
//! this mode the policy reads the raw answer, exactly as it was measured: `verify_v5` still runs and is reported, but does not veto.
//!
//! Beside the Python code (`policy_code`) every outcome carries a `reason_code`. The Python policy has one code "P3" for "not
//! accepted by the guards"; `reason_code` says which guard or raw status it was, and names counting. The split never changes a
//! decision. `COUNTING_OFF`: counting is off in this build, so a question with counting wording is declined whatever the model
//! did, and is served as UNSUPPORTED (not UNKNOWN: the store may hold the items; the system does not count them).

pub mod pyre;
pub mod pystr;
#[rustfmt::skip]
pub mod pytables;
pub mod replay;
pub mod roles;
pub mod rules;
#[rustfmt::skip]
pub mod words;

#[cfg(test)]
mod tests;

use pyre::{Budget, OverBudget, Regex, DEFAULT_BUDGET};
use pystr::norm;
use serde_json::{json, Value};
use std::sync::LazyLock;

/// sha256 of the frozen Python files this port was generated from (`frozen_rules_v2.json` holds the same hashes)
pub use words::FROZEN_SHA256;

/// the served status of a declined answer (owner decision 2 pending: rules.py's stated default)
pub const DECLINED_STATUS: &str = "UNSUPPORTED";

/// one entry of a program (`{"op": .., "pointer": ..}`)
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Step {
    pub op: String,
    pub pointer: Option<i64>,
}

/// one entry of an execution trace: the fields the rules read
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TraceStep {
    pub op: String,
    pub pointer: Option<i64>,
    pub result: Option<String>,
    pub hop_candidate_ids: Option<Vec<String>>,
}

/// the model's output for one question: all the policy reads (no gold field, by construction)
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct Row {
    pub question: String,
    pub status: String,
    pub answer: Option<String>,
    pub program: Vec<Step>,
    pub trace: Vec<TraceStep>,
    pub evidence_ids: Vec<String>,
}

/// replay.operand_reads: one LOOKUP / CHAIN step of the executed program
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Read {
    pub step: usize,
    pub op: String,
    pub subject: Option<String>,
    pub relation: Option<String>,
    pub fact_id: Option<String>,
    pub result: Option<String>,
}

/// replay.execute_swapped
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Swapped {
    pub status: String,
    pub answer: Option<String>,
    pub evidence_ids: Vec<String>,
    pub program: Vec<Step>,
}

/// what D2b needs from the executor: the live `Episode` in the engine (`replay::EpisodeReplay`), a stub in the unit tests
pub trait Replay {
    /// replay.replay_matches: does the program executed here give the row?
    fn matches(&mut self, row: &Row) -> bool;
    fn operand_reads(&mut self, row: &Row) -> Vec<Read>;
    /// the program with its two operand reads swapped, executed on the same manifest; None when the shape does not allow it
    fn execute_swapped(&mut self, row: &Row) -> Option<Swapped>;
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Repair {
    pub detail: String,
    pub original_answer: Option<String>,
    pub original_program: Vec<Step>,
    pub program: Vec<Step>,
    pub evidence_ids: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stages {
    pub p3_accepted: bool,
    pub d2_accepted: bool,
    /// the code of the stage P3+D2 (None = accepted)
    pub d2_code: Option<String>,
}

/// the policy's outcome for one question
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Decision {
    pub accepted: bool,
    /// the answer the policy stage holds (policy.py): for a decline, the raw answer that is NOT served
    pub answer: Option<String>,
    /// the code of the Python stage P3+D2+D2b (None = accepted unchanged)
    pub policy_code: Option<String>,
    /// the reason served: None = accepted unchanged, `D2R_REPAIRED`, or why nothing is served
    pub reason_code: Option<String>,
    pub detail: String,
    pub repair: Option<Repair>,
    pub stages: Stages,
}

struct Guards {
    set_q: Regex,
    pick: Regex,
    count: Regex,
}

/// tools/hybrid_bench_policy.py: SET_Q and PICK_RE as frozen; `count` = the counting alternatives of SET_Q (labels a decline, decides nothing)
static GUARDS: LazyLock<Guards> = LazyLock::new(|| Guards {
    set_q: Regex::new(
        r"\b(how many|number of|count|list|name all|show all|every|all (the )?works|which works|what works|tallest|largest|biggest|smallest|shortest|earliest|latest|oldest|newest|most|least|average|total)\b",
        true,
    ),
    pick: Regex::new(r":\s*.+\s+or\s+.+\?*$", true),
    count: Regex::new(r"\b(how many|number of|count)\b", true),
});

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum P3 {
    Accept,
    NotAnswer,
    Scope,
    Copy,
}

/// hybrid_bench_policy.decide(row, "P3"), naming the guard that refused
pub fn p3(row: &Row, b: &Budget) -> Result<P3, OverBudget> {
    if row.status != "ANSWER" {
        return Ok(P3::NotAnswer);
    }
    let q: Vec<char> = row.question.chars().collect();
    if GUARDS.set_q.search(&q, b)?.is_some() && GUARDS.pick.search(&q, b)?.is_none() {
        return Ok(P3::Scope);
    }
    let last = row.program.iter().rev().find(|s| s.op != "RETURN");
    if let (Some(s), Some(a)) = (last, row.answer.as_deref()) {
        if matches!(s.op.as_str(), "LOOKUP" | "CHAIN")
            && !a.is_empty()
            && norm(&row.question).contains(&norm(a))
        {
            return Ok(P3::Copy);
        }
    }
    Ok(P3::Accept)
}

/// counting wording: the labelling rule of `COUNTING_OFF`. Decides nothing the policy does not decide: it only names a decline.
pub fn counting_question(question: &str, b: &Budget) -> Result<bool, OverBudget> {
    let q: Vec<char> = question.chars().collect();
    if GUARDS.count.search(&q, b)?.is_some() && GUARDS.pick.search(&q, b)?.is_none() {
        return Ok(true);
    }
    let qi = rules::analyse_question(question, b)?;
    Ok(rules::aggregate_cues(&qi).iter().any(|(_, f)| f == "count")
        || !rules::count_comparison_cues(&qi, b)?.is_empty())
}

/// the whole policy over one executed program. Never panics on a question; every internal failure declines (fail closed).
pub fn decide(row: &Row, replay: &mut dyn Replay) -> Decision {
    decide_budget(row, replay, DEFAULT_BUDGET)
}

pub fn decide_budget(row: &Row, replay: &mut dyn Replay, steps: u64) -> Decision {
    let budget = Budget::new(steps);
    let counting = counting_question(&row.question, &budget);
    let mut d = decide_with(row, replay, &budget, counting);
    // the label only: every decline of a counting question names counting, whichever rule fired first. `policy_code`, the
    // decision and the served status are the frozen policy's, and a rule error stays a rule error.
    if !d.accepted && matches!(counting, Ok(true)) {
        d.reason_code = Some("COUNTING_OFF".to_string());
    }
    d
}

fn decide_with(
    row: &Row,
    replay: &mut dyn Replay,
    b: &Budget,
    counting: Result<bool, OverBudget>,
) -> Decision {
    let raw_reason = |status: &str| format!("RAW_{status}");
    let decline = |policy: &str, reason: String, detail: String, stages: Stages| Decision {
        accepted: false,
        answer: row.answer.clone(),
        policy_code: Some(policy.to_string()),
        reason_code: Some(reason),
        detail,
        repair: None,
        stages,
    };
    let early = Stages {
        p3_accepted: false,
        d2_accepted: false,
        d2_code: Some("P3".to_string()),
    };
    let over = || "the question exceeds the rule step budget".to_string();
    let p = match p3(row, b) {
        Ok(p) => p,
        Err(OverBudget) => return decline("P3", "D2X_RULE_ERROR".to_string(), over(), early),
    };
    let is_counting = counting.unwrap_or(true);
    match p {
        P3::NotAnswer => {
            let reason = if is_counting {
                "COUNTING_OFF".to_string()
            } else {
                raw_reason(&row.status)
            };
            return decline(
                "P3",
                reason,
                format!("the model returned {}", row.status),
                early,
            );
        }
        P3::Scope => {
            let reason = if is_counting {
                "COUNTING_OFF"
            } else {
                "P3_SCOPE"
            };
            return decline("P3", reason.to_string(), "scope guard".to_string(), early);
        }
        P3::Copy => {
            return decline(
                "P3",
                "P3_COPY".to_string(),
                "copy guard: the value is text of the question".to_string(),
                early,
            )
        }
        P3::Accept => {}
    }
    if let Some(d) = rules::check(&row.question, row, b) {
        let stages = Stages {
            p3_accepted: true,
            d2_accepted: false,
            d2_code: Some(d.code.to_string()),
        };
        let counted = matches!(d.code, "D2C_AGGREGATE" | "D2C_COUNT_COMPARISON") && is_counting;
        let reason = if counted { "COUNTING_OFF" } else { d.code };
        return decline(d.code, reason.to_string(), d.detail, stages);
    }
    let stages = Stages {
        p3_accepted: true,
        d2_accepted: true,
        d2_code: None,
    };
    // D2b (policy.d2b, variant "repair")
    if !replay.matches(row) {
        let code = "D2R_REPLAY_MISMATCH";
        return decline(
            code,
            code.to_string(),
            "the program does not replay to the scored row (fail closed)".to_string(),
            stages,
        );
    }
    let reads = replay.operand_reads(row);
    let v = match roles::check(&row.question, row, &reads, b) {
        Ok(v) => v,
        Err(OverBudget) => {
            return decline(
                "D2X_RULE_ERROR",
                "D2X_RULE_ERROR".to_string(),
                over(),
                stages,
            )
        }
    };
    let Some(code) = v.code else {
        return Decision {
            accepted: true,
            answer: row.answer.clone(),
            policy_code: None,
            reason_code: None,
            detail: v.detail,
            repair: None,
            stages,
        };
    };
    if code == "D2R_DIRECTION_REVERSED" && v.repairable {
        return match replay.execute_swapped(row) {
            Some(sw) if sw.status == "ANSWER" => Decision {
                accepted: true,
                answer: sw.answer.clone(),
                policy_code: Some("D2R_REPAIRED".to_string()),
                reason_code: Some("D2R_REPAIRED".to_string()),
                detail: v.detail.clone(),
                repair: Some(Repair {
                    detail: v.detail,
                    original_answer: row.answer.clone(),
                    original_program: row.program.clone(),
                    program: sw.program,
                    evidence_ids: sw.evidence_ids,
                }),
                stages,
            },
            _ => decline(
                "D2R_REPAIR_NOT_ANSWER",
                "D2R_REPAIR_NOT_ANSWER".to_string(),
                "the swapped program does not give an answer".to_string(),
                stages,
            ),
        };
    }
    decline(code, code.to_string(), v.detail, stages)
}

/// The verify step of `VerifyRule::DeclineV2` over a finished episode. Only stack-v5 programs were measured: another protocol declines.
#[allow(clippy::too_many_arguments)]
pub fn judge<'m>(
    question: &str,
    ep: &xerj_exec::Episode<'m>,
    ans: &xerj_exec::Answer,
    mem: &'m xerj_index::Memory,
    snap: xerj_store::Snapshot<'m>,
    cands: &[xerj_store::RecId],
    inverse: &[bool],
    item_id: &str,
    k: usize,
) -> Decision {
    let row = replay::row_of(question, ans, mem);
    if ep.protocol != xerj_exec::Protocol::V5 {
        return Decision {
            accepted: false,
            answer: row.answer.clone(),
            policy_code: Some("D2X_RULE_ERROR".to_string()),
            reason_code: Some("DECLINE_PROTOCOL".to_string()),
            detail: "decline_v2 reads stack-v5 programs only".to_string(),
            repair: None,
            stages: Stages {
                p3_accepted: false,
                d2_accepted: false,
                d2_code: None,
            },
        };
    }
    let mut r = replay::EpisodeReplay {
        ep,
        mem,
        snap,
        cands,
        inverse,
        item_id,
        k,
    };
    decide(&row, &mut r)
}

impl Decision {
    /// (final_status, final_answer) served for a raw status. A raw UNKNOWN / CONFLICT passes through with its reason, except on a
    /// counting question: that is UNSUPPORTED like every counting decline.
    pub fn served(&self, raw_status: &str) -> (&'static str, Option<String>) {
        if self.accepted {
            return ("ANSWER", self.answer.clone());
        }
        match raw_status {
            "UNKNOWN" if self.reason_code.as_deref() != Some("COUNTING_OFF") => ("UNKNOWN", None),
            "CONFLICT" if self.reason_code.as_deref() != Some("COUNTING_OFF") => ("CONFLICT", None),
            _ => (DECLINED_STATUS, None),
        }
    }

    /// the `decline` object of a response; every repair is logged here, with the program and answer it replaced
    pub fn to_json(&self, raw_status: &str, raw_evidence: &[String]) -> Value {
        let (status, answer) = self.served(raw_status);
        let steps = |p: &[Step]| {
            p.iter()
                .map(|s| json!([s.op, s.pointer]))
                .collect::<Vec<_>>()
        };
        let evidence: Vec<String> = match (&self.repair, self.accepted) {
            (Some(r), _) => r.evidence_ids.clone(),
            (None, true) => raw_evidence.to_vec(),
            (None, false) => Vec::new(),
        };
        let mut v = json!({
            "policy": "decline_v2", "stage": "P3+D2+D2b", "accepted": self.accepted, "reason_code": self.reason_code,
            "policy_code": self.policy_code, "detail": self.detail, "final_status": status, "final_answer": answer,
            "final_evidence_ids": evidence,
        });
        if let Some(r) = &self.repair {
            v["repair"] = json!({"code": "D2R_REPAIRED", "detail": r.detail, "original_answer": r.original_answer,
                "repaired_answer": self.answer, "original_program": steps(&r.original_program),
                "repaired_program": steps(&r.program), "repaired_evidence_ids": r.evidence_ids});
        }
        v
    }
}
