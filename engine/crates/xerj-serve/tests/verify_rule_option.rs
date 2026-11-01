//! The opt-in verify rules (`VerifyRule`, `AnswerOpts::verify`; round 3, 2026-10-29): the defect they address, the
//! default they keep, and a wrong answer they still withhold.
//!
//! The defect (engine spike 2026-10-27, FC SH 262K: 11 of 97 correct answers withheld): rule (b) of the served rule
//! (`verify_v5`) counts every store subject that occurs in the question as a raw substring. When store names nest
//! ("Albania" inside "Prime Minister of Albania") a question that names one subject counts two, and a correct answer that
//! cites one record is withheld. The binding rules count the exact linker's bindings (longest non-overlapping matches).
//!
//! A second defect, of the `bindings` prototype itself: the snapshot's bindings miss a subject whose records were all
//! withdrawn, so "is A's country the same as B's?" answered by reading A twice (a self-comparison, always "yes") passes
//! rule (b) when B was withdrawn, while the substring count withholds it (the dev data's F10 withdrawal cases, and the
//! round-1 fixtures' "+15 wrong accepted"). `bindings_all` counts the names over every subject ever written.
//!
//! The programs are replayed through the stack-v5 executor on the item's own manifest, as the engine runs them; no model.

use serde_json::json;
use xerj_exec::episode::{Episode, Protocol};
use xerj_exec::Answer;
use xerj_index::{Memory, MemoryConfig, QBuf};
use xerj_serve::verify_bound::{verify_v5_with, VerifyRule};
use xerj_serve::verify_v5::{verify_v5, RULE};
use xerj_store::world::World;
use xerj_store::RecId;

const K: usize = 16;
const ITEM: &str = "verify-rule-option/q0";
const QUESTION: &str = "Who holds the office of Prime Minister of Albania?";

fn fact(doc: &str, text: &str, subject: &str, relation: &str, quote: &str) -> serde_json::Value {
    let start = text.find(quote).expect("quote in the document");
    json!({"fact_id": format!("{doc}@v1/{relation}"), "subject": subject, "relation": relation, "kind": "text",
           "value": quote, "unit": "", "doc_id": doc, "version": "v1", "start": start, "end": start + quote.len(),
           "quote": quote, "scope": "default"})
}

/// three subjects; "Albania" nests inside "Prime Minister of Albania"
fn world_line() -> String {
    let d1 = "Albania is a country in Europe. Its capital is Tirana.";
    let d2 = "The Prime Minister of Albania is Edi Rama.";
    let d3 = "Kosovo is a country in Europe. Its capital is Pristina.";
    json!({
        "group_id": "verify-rule-option", "world_id": "verify-rule-option",
        "base_documents": [
            {"doc_id": "w/d1", "version": "v1", "text": d1},
            {"doc_id": "w/d2", "version": "v1", "text": d2},
            {"doc_id": "w/d3", "version": "v1", "text": d3},
        ],
        "base_facts": [
            fact("w/d1", d1, "Albania", "capital", "Tirana"),
            fact("w/d2", d2, "Prime Minister of Albania", "officeholder", "Edi Rama"),
            fact("w/d3", d3, "Kosovo", "capital", "Pristina"),
        ],
        "edits": [],
        "items": [{"item_id": ITEM, "question": QUESTION, "phase": "after"}],
    })
    .to_string()
}

fn memory(line: &str) -> (Memory, u32) {
    let w = World::parse(line).expect("world");
    let mut mem = Memory::new(MemoryConfig {
        bm25_records: false,
        bm25_documents: false,
        ..MemoryConfig::default()
    });
    let revs = mem.ingest_world(&w).expect("ingest");
    (mem, revs.final_)
}

/// LOOKUP the record of (`subject`, `relation`) in the item's manifest, then RETURN: the answer record and the manifest
fn lookup_answer(mem: &Memory, rev: u32, subject: &str, relation: &str) -> (Answer, Vec<RecId>) {
    let snap = mem.snapshot_at(rev);
    let mut buf = QBuf::default();
    let m = mem.candidate_manifest_v5(&snap, ITEM, QUESTION, K, &mut buf);
    let ptr = m
        .cands
        .iter()
        .position(|&r| mem.store.subject(r) == subject && mem.store.relation(r) == relation)
        .unwrap_or_else(|| panic!("({subject}, {relation}) is not in the manifest"));
    let mut ep = Episode::new_v5(mem, snap, m.cands.clone(), m.inverse.clone(), ITEM, K);
    ep.step(Protocol::V5.op_id("LOOKUP").unwrap(), Some(ptr as i64))
        .expect("LOOKUP");
    ep.step(Protocol::V5.op_id("RETURN").unwrap(), None)
        .expect("RETURN");
    (ep.answer_or_unfinished(), m.cands)
}

fn withheld(rule: VerifyRule, mem: &Memory, rev: u32, ans: &Answer, cands: &[RecId]) -> bool {
    let snap = mem.snapshot_at(rev);
    let mut buf = QBuf::default();
    verify_v5_with(rule, mem, &snap, QUESTION, cands, ans, &mut buf).vetoed
}

#[test]
fn the_binding_rules_keep_a_correct_answer_to_a_question_with_a_nested_store_name() {
    let line = world_line();
    let (mem, rev) = memory(&line);
    let (ans, cands) = lookup_answer(&mem, rev, "Prime Minister of Albania", "officeholder");
    assert_eq!(
        (ans.status, ans.answer.as_deref()),
        ("ANSWER", Some("Edi Rama"))
    );
    // the fix, opt-in: the question binds one subject, the answer cites it
    assert!(!withheld(VerifyRule::Bindings, &mem, rev, &ans, &cands));
    assert!(!withheld(VerifyRule::BindingsPath, &mem, rev, &ans, &cands));
    assert!(!withheld(VerifyRule::BindingsAll, &mem, rev, &ans, &cands));
    // the default is unchanged: the served rule still counts "Albania" a second time and withholds the answer
    assert!(withheld(VerifyRule::Served, &mem, rev, &ans, &cands));
}

/// KNOWN DEFECT of the served (default) rule, kept failing on purpose: the same correct answer is withheld.
#[test]
#[ignore = "known defect (54_verifier3, 2026-10-29): the served verify rule (b) withholds a correct answer when store names nest; opt-in fix: verify=bindings; run with --ignored"]
fn known_defect_the_served_rule_keeps_a_correct_answer_to_a_question_with_a_nested_store_name() {
    let line = world_line();
    let (mem, rev) = memory(&line);
    let (ans, cands) = lookup_answer(&mem, rev, "Prime Minister of Albania", "officeholder");
    assert_eq!(ans.status, "ANSWER");
    assert!(
        !withheld(VerifyRule::Served, &mem, rev, &ans, &cands),
        "the served rule withholds the correct answer \"Edi Rama\""
    );
}

#[test]
fn every_rule_withholds_an_answer_that_cites_a_subject_the_question_does_not_name() {
    let line = world_line();
    let (mem, rev) = memory(&line);
    let (ans, cands) = lookup_answer(&mem, rev, "Kosovo", "capital");
    assert_eq!(
        (ans.status, ans.answer.as_deref()),
        ("ANSWER", Some("Pristina"))
    );
    for rule in [
        VerifyRule::Served,
        VerifyRule::Bindings,
        VerifyRule::BindingsPath,
        VerifyRule::BindingsAll,
    ] {
        assert!(withheld(rule, &mem, rev, &ans, &cands), "{rule:?}");
    }
}

#[test]
fn the_default_rule_is_the_served_rule() {
    assert_eq!(VerifyRule::default(), VerifyRule::Served);
    assert_eq!(VerifyRule::NAMES[0], "served");
    for name in VerifyRule::NAMES {
        assert_eq!(VerifyRule::parse(name).map(VerifyRule::name), Some(name));
    }
    assert_eq!(VerifyRule::parse("fuzzy"), None);
    assert_eq!(VerifyRule::Served.rule(), RULE);
    assert_eq!(VerifyRule::Served.opts().map(|_| ()), None);
    // `Served` is `verify_v5` itself, verdict and reason
    let line = world_line();
    let (mem, rev) = memory(&line);
    for (subject, relation) in [
        ("Prime Minister of Albania", "officeholder"),
        ("Kosovo", "capital"),
        ("Albania", "capital"),
    ] {
        let (ans, cands) = lookup_answer(&mem, rev, subject, relation);
        let snap = mem.snapshot_at(rev);
        let mut buf = QBuf::default();
        let a = verify_v5(&mem, QUESTION, &cands, &ans, &mut buf);
        let b = verify_v5_with(
            VerifyRule::Served,
            &mem,
            &snap,
            QUESTION,
            &cands,
            &ans,
            &mut buf,
        );
        assert_eq!((a.vetoed, a.reason), (b.vetoed, b.reason), "{subject}");
    }
}

const WQ_ITEM: &str = "verify-rule-option/withdrawn/q0";
const WQ: &str = "Is the country of Prapen River the same as the country of Coryra River?";

/// two rivers; every record of "Prapen River" is withdrawn by the world's edit, so the final snapshot cannot bind it
fn withdrawn_world_line() -> String {
    let d1 = "Coryra River flows through Hallend.";
    let d2 = "Prapen River flows through Fallow.";
    json!({
        "group_id": "verify-rule-option-withdrawn", "world_id": "verify-rule-option-withdrawn",
        "base_documents": [
            {"doc_id": "w2/d1", "version": "v1", "text": d1},
            {"doc_id": "w2/d2", "version": "v1", "text": d2},
        ],
        "base_facts": [
            fact("w2/d1", d1, "Coryra River", "country", "Hallend"),
            fact("w2/d2", d2, "Prapen River", "country", "Fallow"),
        ],
        "edits": [{"op": "withdraw_document", "doc_id": "w2/d2", "version": "v1"}],
        "items": [{"item_id": WQ_ITEM, "question": WQ, "phase": "after"}],
    })
    .to_string()
}

/// LOOKUP Coryra River's country twice, EQ, RETURN: a self-comparison, "yes" whatever the question asks
fn self_comparison(mem: &Memory, rev: u32) -> (Answer, Vec<RecId>) {
    let snap = mem.snapshot_at(rev);
    let mut buf = QBuf::default();
    let m = mem.candidate_manifest_v5(&snap, WQ_ITEM, WQ, K, &mut buf);
    let ptr = m
        .cands
        .iter()
        .position(|&r| mem.store.subject(r) == "Coryra River")
        .expect("Coryra River's record in the manifest");
    assert!(
        !m.cands
            .iter()
            .any(|&r| mem.store.subject(r) == "Prapen River"),
        "the withdrawn subject is not in the final snapshot's manifest"
    );
    let mut ep = Episode::new_v5(mem, snap, m.cands.clone(), m.inverse.clone(), WQ_ITEM, K);
    for (op, p) in [
        ("LOOKUP", Some(ptr as i64)),
        ("LOOKUP", Some(ptr as i64)),
        ("EQ", None),
        ("RETURN", None),
    ] {
        ep.step(Protocol::V5.op_id(op).unwrap(), p).expect(op);
    }
    (ep.answer_or_unfinished(), m.cands)
}

fn withheld_wq(rule: VerifyRule, mem: &Memory, rev: u32, ans: &Answer, cands: &[RecId]) -> bool {
    let snap = mem.snapshot_at(rev);
    let mut buf = QBuf::default();
    verify_v5_with(rule, mem, &snap, WQ, cands, ans, &mut buf).vetoed
}

#[test]
fn the_served_and_bindings_all_rules_withhold_a_self_comparison_when_the_other_subject_was_withdrawn(
) {
    let line = withdrawn_world_line();
    let (mem, rev) = memory(&line);
    let (ans, cands) = self_comparison(&mem, rev);
    assert_eq!((ans.status, ans.answer.as_deref()), ("ANSWER", Some("yes")));
    assert!(withheld_wq(VerifyRule::Served, &mem, rev, &ans, &cands));
    assert!(withheld_wq(
        VerifyRule::BindingsAll,
        &mem,
        rev,
        &ans,
        &cands
    ));
}

/// KNOWN DEFECT of the `bindings` prototype (kept failing on purpose): the snapshot's bindings miss the withdrawn subject,
/// so the self-comparison passes rule (b). Fixed by `bindings_all`.
#[test]
#[ignore = "known defect (54_verifier3, 2026-10-29): verify=bindings accepts a self-comparison when the other named subject was withdrawn; fix: verify=bindings_all; run with --ignored"]
fn known_defect_the_bindings_rule_withholds_a_self_comparison_when_the_other_subject_was_withdrawn()
{
    let line = withdrawn_world_line();
    let (mem, rev) = memory(&line);
    let (ans, cands) = self_comparison(&mem, rev);
    assert_eq!(ans.status, "ANSWER");
    assert!(
        withheld_wq(VerifyRule::Bindings, &mem, rev, &ans, &cands),
        "verify=bindings accepts the self-comparison \"yes\""
    );
}
