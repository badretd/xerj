//! Decision-input text, exactly as future_scientific/inputs.py (stack-v1), stack_v2.py (stack-v2) and dev_v5 inputs.py /
//! stack_v2.py (stack-v5) build it. Every piece is tokenized SEPARATELY (segment by segment, no special tokens) and the token
//! lists are concatenated:
//!
//! ```text
//! [CLS] "Question: {q}\n" [external_executor: "Facts:\n" + one segment per candidate line] "Steps:" + one segment per prior
//! action + suffix
//! suffix = "\nNext:"  |  (stack-v2 / v5 external_executor with a hop manifest) "\nChain facts:\n" + hop lines + "Next:"
//! ```
//! candidate line `"[k] subject | relation | kind: value | quote\n"`, hop line `"[K+j] [= ]subject | ...\n"`; a prior action
//! renders as `" LOOKUP [k];"`, `" LOOKUP [none];"`, `" CHAIN [K+j];"`, `" LT;"`, ...; the external executor appends
//! `" -> {result}"` before `";"` when the action produced a result text.
//!
//! Stack-v5 (traversal direction): an INVERSE candidate (followed from its value back to its subject; stack_v5.InverseCand) is
//! rendered `"[k] <= subject | ..."` in the base block AND in the hop block (where `"<= "` replaces the resolve marker `"= "`,
//! inverse hop candidates always resolve); an executed inverse hop renders `" IN"` after its pointer
//! (`" LOOKUP [3] IN -> VALUE text Liberia;"`). Without inverse candidates every string is the stack-v2 one.

use crate::config::{Arm, StackProtocol};

/// the four model-visible strings of one candidate record (xr_codecs.candidate_fields): subject, relation phrase,
/// "kind: rendered value", source quote; plus the stack-v5 direction of the candidate
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CandidateFields {
    pub subject: String,
    pub relation: String,
    pub value: String,
    pub quote: String,
    /// stack-v5: an INVERSE candidate (the record is offered value -> subject; the executor answers with its subject). Rendered
    /// with the `"<= "` marker; always false under stack v1 / v2
    pub inverse: bool,
}

impl CandidateFields {
    pub fn new(subject: &str, relation: &str, value: &str, quote: &str) -> Self {
        CandidateFields {
            subject: subject.into(),
            relation: relation.into(),
            value: value.into(),
            quote: quote.into(),
            inverse: false,
        }
    }
    /// the same record as a stack-v5 inverse candidate (or forward, `inverse = false`)
    pub fn with_inverse(mut self, inverse: bool) -> Self {
        self.inverse = inverse;
        self
    }
    pub fn fields(&self) -> [&str; 4] {
        [&self.subject, &self.relation, &self.value, &self.quote]
    }
}

pub const BOS: u32 = 50281;
pub const FACTS: &str = "Facts:\n";
pub const STEPS: &str = "Steps:";
pub const NEXT: &str = "\nNext:";
pub const HOP_HEADER: &str = "\nChain facts:\n";
pub const HOP_NEXT: &str = "Next:";
/// stack-v2: marker of a hop candidate whose subject resolves to the chained value
pub const HOP_MARK: &str = "= ";
/// stack-v5: marker of an inverse candidate line (base and hop blocks)
pub const INVERSE_MARK: &str = "<= ";
/// stack-v5: history marker of an executed inverse hop, after its pointer
pub const INVERSE_HISTORY: &str = " IN";

pub fn question_segment(question: &str) -> String {
    format!("Question: {question}\n")
}

/// inputs.candidate_line: `"[k] [<= ]subject | relation | value | quote\n"`
pub fn candidate_line(k: usize, f: &CandidateFields) -> String {
    let mark = if f.inverse { INVERSE_MARK } else { "" };
    format!(
        "[{k}] {mark}{} | {} | {} | {}\n",
        f.subject, f.relation, f.value, f.quote
    )
}

/// stack_v2.hop_line: `"[k] [<= |= ]subject | ...\n"` (inverse first, then the resolve marker)
pub fn hop_line(k: usize, f: &CandidateFields, resolved: bool) -> String {
    let mark = if f.inverse {
        INVERSE_MARK
    } else if resolved {
        HOP_MARK
    } else {
        ""
    };
    format!(
        "[{k}] {mark}{} | {} | {} | {}\n",
        f.subject, f.relation, f.value, f.quote
    )
}

/// one executed action as the decision text shows it
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct HistoryStep {
    pub op: String,
    pub pointer: Option<usize>,
    pub result: Option<String>,
    /// stack-v5: the action followed an inverse candidate (trace `"direction": "in"`); rendered as `" IN"` after the pointer
    pub inverse: bool,
}

pub fn history_segment(h: &HistoryStep, arm: Arm, stack: StackProtocol, k: usize) -> String {
    let mut s = format!(" {}", h.op);
    if h.op == "LOOKUP" {
        match h.pointer {
            Some(p) => s.push_str(&format!(" [{p}]")),
            None => s.push_str(" [none]"),
        }
    } else if h.op == "CHAIN" && stack.has_hops() {
        match h.pointer {
            Some(p) => s.push_str(&format!(" [{}]", k + p)),
            None => s.push_str(" [none]"),
        }
    }
    // stack_v2.build_decision_input (dev_v5): `if h.get("direction") == "in": s += " IN"` (the stack-v1 builder has no direction)
    if h.inverse && stack.has_hops() {
        s.push_str(INVERSE_HISTORY);
    }
    if arm == Arm::ExternalExecutor {
        if let Some(r) = &h.result {
            if !r.is_empty() {
                s.push_str(" -> ");
                s.push_str(r);
            }
        }
    }
    s.push(';');
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn v5_markers() {
        let f = CandidateFields::new(
            "Liberia",
            "head of government",
            "text: George Weah",
            "George Weah",
        );
        assert_eq!(
            candidate_line(3, &f),
            "[3] Liberia | head of government | text: George Weah | George Weah\n"
        );
        let fi = f.clone().with_inverse(true);
        assert_eq!(
            candidate_line(3, &fi),
            "[3] <= Liberia | head of government | text: George Weah | George Weah\n"
        );
        assert_eq!(
            hop_line(17, &f, true),
            "[17] = Liberia | head of government | text: George Weah | George Weah\n"
        );
        assert_eq!(
            hop_line(17, &f, false),
            "[17] Liberia | head of government | text: George Weah | George Weah\n"
        );
        assert_eq!(
            hop_line(17, &fi, true),
            "[17] <= Liberia | head of government | text: George Weah | George Weah\n"
        );
        let h = HistoryStep {
            op: "LOOKUP".into(),
            pointer: Some(3),
            result: Some("VALUE text Liberia".into()),
            inverse: true,
        };
        assert_eq!(
            history_segment(&h, Arm::ExternalExecutor, StackProtocol::V5, 16),
            " LOOKUP [3] IN -> VALUE text Liberia;"
        );
        let c = HistoryStep {
            op: "CHAIN".into(),
            pointer: Some(2),
            result: Some("UNKNOWN".into()),
            inverse: true,
        };
        assert_eq!(
            history_segment(&c, Arm::ExternalExecutor, StackProtocol::V5, 16),
            " CHAIN [18] IN -> UNKNOWN;"
        );
        let r = HistoryStep {
            op: "RETURN".into(),
            ..Default::default()
        };
        assert_eq!(
            history_segment(&r, Arm::ExternalExecutor, StackProtocol::V5, 16),
            " RETURN;"
        );
        // forward steps are the stack-v2 strings
        let fwd = HistoryStep {
            inverse: false,
            ..h
        };
        assert_eq!(
            history_segment(&fwd, Arm::ExternalExecutor, StackProtocol::V5, 16),
            " LOOKUP [3] -> VALUE text Liberia;"
        );
        assert_eq!(
            history_segment(&fwd, Arm::ExternalExecutor, StackProtocol::V2, 16),
            " LOOKUP [3] -> VALUE text Liberia;"
        );
    }
}
