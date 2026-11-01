//! PROTOTYPE (research branch `verify-bindings`, 2026-10-27): the stack-v5 verify rule over exact entity bindings. This is
//! not the reference rule and changes no default; `verify_v5` stays what `answer()` serves.
//!
//! Rule (b) of the reference (`tools/verify_rows.py`, `xerj_exec::verify`) and of its chain / direction extensions
//! (`verify_chain_aware`, `verify_v5`) says that a question naming >= 2 store subjects needs >= 2 distinct cited names, and
//! it counts the named subjects with a raw substring scan over every subject string of the store. In a large store names
//! nest: "Queen Victoria I" contains another subject's name, and "Prime Minister of Albania" contains "Albania". The scan
//! then counts two subjects in a question that names one, and the rule withholds correct single-hop answers (FC SH 262K: 11
//! of 97 in the engine spike).
//!
//! Here the question's entities are its exact bindings: the reference linker's matches (`Memory::link`), which take the
//! snapshot's subjects longest first, on word boundaries, and mask what a longer match covered, so a nested name never binds
//! twice. The evidence checks follow the executed path and its directions:
//! - (a) as `verify_v5`: a cited forward record's subject must occur in the question unless a CHAIN step or an inverse step
//!   reached it; an inverse LOOKUP's record must have its value in the question. With `Opts::bind_a`, "occurs" becomes "is a
//!   binding" (a subject among the linked subjects, a value among the linked text values).
//! - (b) a question with >= 2 bindings needs >= 2 distinct cited names (subjects, and an inverse record's value).
//! - (c) with `Opts::path`, every CHAIN record must continue the path: its subject (forward) or its value (inverse) is the
//!   name the previous read arrived at (a forward record's value, an inverse record's subject), compared after
//!   `normalize_entity`.
//!
//! The answer record itself is never changed: callers report the raw status/answer and the verdict separately.
//!
//! OPT-IN (round 3, 2026-10-29): [`VerifyRule`] lets a caller of `answer_stoppable_opts` (`AnswerOpts::verify`, the
//! product's `_ask` option `verify`) choose this rule per question. The default, `VerifyRule::Served`, is `verify_v5`: the
//! served rule stays what it was until the owner decides.
//!
//! `Opts::bind_all` (round 3): the snapshot's bindings miss a subject whose records were all withdrawn, so a question that
//! names it next to a visible subject binds one name, and an answer that read the visible subject twice ("is X's country
//! the same as Y's?" answered from X alone) passes (b), where the substring count vetoes it. With `bind_all` rule (b)
//! counts the reference masking over every subject ever written (`Linker::link_any_state`): the substring rule's
//! vocabulary without its nested-name double count.

use crate::verify_v5::verify_v5;
use xerj_exec::{Answer, Verdict};
use xerj_index::{Memory, QBuf};
use xerj_store::text::normalize_entity;
use xerj_store::{RecId, Snapshot};

pub const RULE: &str = "exact bindings (prototype; v5 direction and path aware)";

/// Which verify rule judges a stack-v5 ANSWER (EXTENSION, opt-in per question).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum VerifyRule {
    /// `verify_v5`, the rule `answer()` has always served: rule (b) counts raw substrings of every store subject
    #[default]
    Served,
    /// [`verify_bound`] with `Opts { bind_b: true, bind_a: false, path: false }`: rule (b) counts the question's exact
    /// bindings (longest non-overlapping linker matches), so a nested store name no longer counts as a second subject
    Bindings,
    /// [`verify_bound`] with [`Opts::default`]: bindings for (b) and path continuity for every CHAIN record
    BindingsPath,
    /// [`verify_bound`] with `Opts { bind_b: true, bind_all: true, .. }`: rule (b) counts the reference masking over every
    /// subject ever written (any state), so a nested name does not count twice and a withdrawn subject still counts
    BindingsAll,
    /// the frozen decline policy v2 ([`crate::decline_v2`]): P3 + D2 + D2b on the raw answer, as it was measured. `verify_v5` still
    /// runs and is reported, but does not veto; the response gains `reason_code`, `repair` and `decline`
    DeclineV2,
}

impl VerifyRule {
    /// every value `parse` accepts, the default first
    pub const NAMES: [&'static str; 5] = [
        "served",
        "bindings",
        "bindings_path",
        "bindings_all",
        "decline_v2",
    ];

    pub fn parse(s: &str) -> Option<VerifyRule> {
        match s {
            "served" => Some(VerifyRule::Served),
            "bindings" => Some(VerifyRule::Bindings),
            "bindings_path" => Some(VerifyRule::BindingsPath),
            "bindings_all" => Some(VerifyRule::BindingsAll),
            "decline_v2" => Some(VerifyRule::DeclineV2),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            VerifyRule::Served => "served",
            VerifyRule::Bindings => "bindings",
            VerifyRule::BindingsPath => "bindings_path",
            VerifyRule::BindingsAll => "bindings_all",
            VerifyRule::DeclineV2 => "decline_v2",
        }
    }

    /// the `verify.rule` text of a response judged by this rule (`Served`: `verify_v5::RULE`, as before)
    pub fn rule(self) -> &'static str {
        match self {
            VerifyRule::Served => crate::verify_v5::RULE,
            VerifyRule::Bindings => "exact bindings for (b) (opt-in)",
            VerifyRule::BindingsPath => "exact bindings for (b) + CHAIN path continuity (opt-in)",
            VerifyRule::BindingsAll => "exact bindings over every written subject for (b) (opt-in)",
            VerifyRule::DeclineV2 => {
                "decline policy v2: P3 + D2 + D2b; verify_v5 reported, no veto (opt-in)"
            }
        }
    }

    /// the prototype options of this rule (None = `verify_v5`)
    pub fn opts(self) -> Option<Opts> {
        match self {
            VerifyRule::Served | VerifyRule::DeclineV2 => None,
            VerifyRule::Bindings => Some(Opts {
                bind_b: true,
                bind_a: false,
                path: false,
                bind_all: false,
            }),
            VerifyRule::BindingsPath => Some(Opts::default()),
            VerifyRule::BindingsAll => Some(Opts {
                bind_b: true,
                bind_a: false,
                path: false,
                bind_all: true,
            }),
        }
    }
}

/// The verdict of `rule` on a stack-v5 answer record. `VerifyRule::Served` calls `verify_v5` with the same arguments as
/// `answer()` always did; `snap` is read only by the binding rules (the exact linker runs on it).
pub fn verify_v5_with(
    rule: VerifyRule,
    mem: &Memory,
    snap: &Snapshot<'_>,
    question: &str,
    cands: &[RecId],
    ans: &Answer,
    buf: &mut QBuf,
) -> Verdict {
    match rule.opts() {
        None => verify_v5(mem, question, cands, ans, buf),
        Some(o) => verify_bound(mem, snap, question, cands, ans, o, buf),
    }
}

/// Which parts of the prototype apply; `Opts::default()` is the proposal (bindings for (b), path continuity).
#[derive(Clone, Copy, Debug)]
pub struct Opts {
    /// (b) counts the linker's bindings instead of raw substrings of every store subject
    pub bind_b: bool,
    /// (a) requires bindings instead of substring containment
    pub bind_a: bool,
    /// (c) CHAIN records must continue the path
    pub path: bool,
    /// with `bind_b`: (b) counts the reference masking over every subject ever written, any state
    /// (`Linker::link_any_state`), instead of the snapshot's bindings
    pub bind_all: bool,
}

impl Default for Opts {
    fn default() -> Self {
        Opts {
            bind_b: true,
            bind_a: false,
            path: true,
            bind_all: false,
        }
    }
}

/// `cands` = the base manifest the episode ran on (an inverse LOOKUP's pointer indexes it).
pub fn verify_bound(
    mem: &Memory,
    snap: &Snapshot<'_>,
    question: &str,
    cands: &[RecId],
    ans: &Answer,
    opts: Opts,
    buf: &mut QBuf,
) -> Verdict {
    if ans.status != "ANSWER" {
        return Verdict {
            vetoed: false,
            reason: "",
        };
    }
    let store = &mem.store;
    let bound_subjects: Vec<&str> = mem
        .link(snap, question, buf)
        .iter()
        .map(|&s| store.pools.subjects.get(s))
        .collect();
    let bound_values: Vec<&str> = if opts.bind_a {
        mem.link_values(snap, question, buf)
            .iter()
            .map(|&s| store.pools.strs.get(s))
            .collect()
    } else {
        Vec::new()
    };
    let in_question = |s: &str| {
        if opts.bind_a {
            bound_subjects.contains(&s)
        } else {
            question.contains(s)
        }
    };
    let value_in_question = |v: &str| {
        if opts.bind_a {
            bound_values.contains(&v) || bound_subjects.contains(&v)
        } else {
            question.contains(v)
        }
    };
    let mut exempt: Vec<&str> = Vec::new();
    let mut inverse_values: Vec<&str> = Vec::new();
    // the name the previous read arrived at: a forward record's value, an inverse record's subject
    let mut previous_name: Option<&str> = None;
    for t in &ans.execution_trace {
        if t.illegal {
            continue;
        }
        let rec = t
            .pointer
            .filter(|&p| p >= 0)
            .map(|p| p as usize)
            .and_then(|p| match t.op.as_str() {
                "CHAIN" => t.hop_candidate_ids.as_ref().and_then(|h| h.get(p)).copied(),
                "LOOKUP" => cands.get(p).copied(),
                _ => None,
            });
        let Some(r) = rec else { continue };
        let subject = store.subject(r);
        let v = store.typed_value(r).value;
        if t.op == "CHAIN" || t.inverse {
            exempt.push(subject);
        }
        if t.inverse {
            if t.op == "LOOKUP" && !value_in_question(v) {
                return Verdict {
                    vetoed: true,
                    reason: "inverse record's value not in question",
                };
            }
            inverse_values.push(v);
        }
        if opts.path && t.op == "CHAIN" {
            // the name the hop starts from: the subject of a forward record, the value of an inverse one
            let from = if t.inverse { v } else { subject };
            let continues =
                previous_name.is_some_and(|p| normalize_entity(p) == normalize_entity(from));
            if !continues {
                return Verdict {
                    vetoed: true,
                    reason: "CHAIN record does not continue the path",
                };
            }
        }
        previous_name = Some(if t.inverse { subject } else { v });
    }
    let mut names: Vec<&str> = Vec::new();
    for &r in &ans.evidence_ids {
        let s = store.subject(r);
        if !names.contains(&s) {
            names.push(s);
        }
    }
    if names.iter().any(|s| !in_question(s) && !exempt.contains(s)) {
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
    let named = if opts.bind_b && opts.bind_all {
        mem.linker.link_any_state(store, question, buf).len()
    } else if opts.bind_b {
        bound_subjects.len()
    } else {
        mem.linker.substring_subjects(store, question, buf, 2).len()
    };
    if names.len() < 2 && named >= 2 {
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
