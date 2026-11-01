//! D2b, the role / direction check for binary comparisons: a line-by-line port of the frozen
//! research/xerj_r_mvp/decline_v1/roles.py (sha256 in `words::FROZEN_SHA256`). Do not change a rule here: the Python is the oracle.

use super::pyre::{Budget, OverBudget, Regex};
use super::pystr::lower;
use super::rules::second_word;
use super::words as w;
use super::{Read, Row};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::LazyLock;

fn has(list: &[&str], s: &str) -> bool {
    list.binary_search(&s).is_ok()
}

const ENT: &str = "§([01])§";
const CLAUSE: &str = "[^,;?:§]*?";

struct Rx {
    ent: Regex,
    ref_after: Regex,
    ref_before: Regex,
    ref_than: Regex,
    ref_passive: Regex,
    ref_temporal: Regex,
    negation: Regex,
    poss: Regex,
    dq: Regex,
    sq: Regex,
    paren: Regex,
    words: Regex,
}

static RX: LazyLock<Rx> = LazyLock::new(|| Rx {
    ent: Regex::new(ENT, false),
    ref_after: Regex::new(
        &format!(
            r"\b(?:compared (?:with|to)|as compared (?:with|to)|relative to|versus|vs\.?|against|in comparison (?:with|to)|in relation to)\b{CLAUSE}{ENT}"
        ),
        false,
    ),
    ref_before: Regex::new(
        &format!(
            r"{ENT}[^,;?:§]{{0,40}}?\bas (?:the |a )?(?:reference|baseline|benchmark|base|comparison(?: point)?|yardstick|standard)\b"
        ),
        false,
    ),
    ref_than: Regex::new(&format!(r"\bthan\b{CLAUSE}{ENT}"), false),
    ref_passive: Regex::new(
        &format!(
            r"\b(?:exceeded|surpassed|outranked|outnumbered|topped|beaten|bettered|outstripped|overtaken|outdone)(?: [a-z-]+){{0,3}}? by\b{CLAUSE}{ENT}"
        ),
        false,
    ),
    ref_temporal: Regex::new(&format!(r"\b(before|after)\b{CLAUSE}{ENT}"), false),
    negation: Regex::new(r"(?<![\w-])(?:not|never|no)(?![\w-])|n't\b", false),
    poss: Regex::new(r"(?<=[\w)\]§])'s(?![\w])", false),
    dq: Regex::new(r#""[^"§]*""#, false),
    sq: Regex::new(
        r#"(?<![\w])'(?=\S)(?:[^'§]|'(?=\w))*?(?<=\S)'(?![\w])"#,
        false,
    ),
    paren: Regex::new(r"\([^()§]*\)", false),
    words: Regex::new(r"[a-z]+(?:-[a-z]+)*", false),
});

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RoleVerdict {
    pub code: Option<&'static str>,
    pub detail: String,
    pub applies: bool,
    pub repairable: bool,
}

fn verdict(code: Option<&'static str>, detail: String, applies: bool) -> RoleVerdict {
    RoleVerdict {
        code,
        detail,
        applies,
        repairable: false,
    }
}

fn norm_space(s: &str) -> String {
    s.replace(['\u{2019}', '\u{2018}'], "'")
}

fn chars(s: &str) -> Vec<char> {
    s.chars().collect()
}

fn text(s: &[char]) -> String {
    s.iter().collect()
}

/// word-bounded occurrences of an entity name in the question ([(start, end)], code points)
pub fn mention_spans(
    question: &str,
    name: &str,
    b: &Budget,
) -> Result<Vec<(usize, usize)>, OverBudget> {
    if name.is_empty() {
        return Ok(Vec::new());
    }
    let q = chars(&norm_space(question));
    let n = norm_space(name);
    let mut out = Vec::new();
    for flag in [false, true] {
        out = Regex::word_bounded_literal(&n, flag)
            .find_all(&q, b)?
            .iter()
            .map(|m| (m.start, m.end))
            .collect();
        if !out.is_empty() {
            break;
        }
    }
    Ok(out)
}

type Spans = [Vec<(usize, usize)>; 2];

/// the question in lower case with the two operands' mentions replaced by §0§ / §1§; (skeleton, spans per operand)
pub fn skeleton(
    question: &str,
    names: [&str; 2],
    b: &Budget,
) -> Result<(String, Spans), OverBudget> {
    let q = norm_space(question);
    let qc = chars(&q);
    let mut spans: Spans = [
        mention_spans(&q, names[0], b)?,
        mention_spans(&q, names[1], b)?,
    ];
    for i in 0..2 {
        let j = 1 - i;
        let other = spans[j].clone();
        spans[i].retain(|s| {
            !other
                .iter()
                .any(|&(a, z)| a <= s.0 && s.1 <= z && (a, z) != *s)
        });
    }
    let mut marks: Vec<(usize, usize, usize)> = (0..2)
        .flat_map(|i| spans[i].iter().map(move |&(s, e)| (s, e, i)))
        .collect();
    marks.sort_unstable();
    let mut out = String::new();
    let mut pos = 0;
    for (s, e, i) in marks {
        if s < pos {
            continue;
        }
        out.push_str(&lower(&text(&qc[pos..s])));
        out.push_str(&format!("§{i}§"));
        pos = e;
    }
    out.push_str(&lower(&text(&qc[pos..])));
    let rx = &*RX;
    let sk = chars(&out);
    let sk = rx.poss.sub(&sk, " poss", b)?;
    let sk = rx.dq.sub(&sk, " qt ", b)?;
    let mut sk = rx.sq.sub(&sk, " qt ", b)?;
    for _ in 0..3 {
        sk = rx.paren.sub(&sk, " ", b)?;
    }
    Ok((text(&sk), spans))
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Roles {
    Undetermined(String),
    Read {
        subject: usize,
        reference: usize,
        expected: &'static str,
        why: String,
    },
}

fn group_int(s: &[char], g: Option<(usize, usize)>) -> usize {
    g.map_or(0, |(a, _)| usize::from(s[a] == '1'))
}

const AGE: [(&str, Option<&str>, Option<&str>); 3] = [
    ("older", Some("LT"), Some("GT")),
    ("newer", Some("GT"), None),
    ("younger", Some("GT"), Some("LT")),
];

fn undetermined(s: impl Into<String>) -> Result<Roles, OverBudget> {
    Ok(Roles::Undetermined(s.into()))
}

pub fn read_roles(
    question: &str,
    names: [&str; 2],
    kinds: [Option<&str>; 2],
    b: &Budget,
) -> Result<Roles, OverBudget> {
    let (sk, spans) = skeleton(question, names, b)?;
    if spans[0].is_empty() || spans[1].is_empty() {
        return undetermined("operand not named");
    }
    let rx = &*RX;
    let skc = chars(&sk);
    let mut refs: BTreeMap<&str, BTreeSet<usize>> = BTreeMap::new();
    for (name, r) in [
        ("compared", &rx.ref_after),
        ("as_reference", &rx.ref_before),
        ("than", &rx.ref_than),
        ("passive", &rx.ref_passive),
    ] {
        for m in r.find_all(&skc, b)? {
            refs.entry(name)
                .or_default()
                .insert(group_int(&skc, m.groups[0]));
        }
    }
    let mut temporal: BTreeMap<String, BTreeSet<usize>> = BTreeMap::new();
    for m in rx.ref_temporal.find_all(&skc, b)? {
        let (a, z) = m.groups[0].unwrap_or((0, 0));
        temporal
            .entry(text(&skc[a..z]))
            .or_default()
            .insert(group_int(&skc, m.groups[1]));
    }
    let mut cand: BTreeSet<usize> = refs.values().flatten().copied().collect();
    if cand.is_empty() && !temporal.is_empty() {
        cand = temporal.values().flatten().copied().collect();
    }
    if cand.len() != 1 {
        return undetermined(if cand.is_empty() {
            "no reference marker"
        } else {
            "both operands in reference positions"
        });
    }
    let reference = *cand.iter().next().unwrap_or(&0);
    let subject = 1 - reference;
    let neutral = rx.ent.sub(&skc, " ", b)?;
    let words: Vec<String> = rx
        .words
        .find_all(&neutral, b)?
        .iter()
        .map(|m| text(&neutral[m.start..m.end]))
        .collect();
    if rx.negation.search(&neutral, b)?.is_some() {
        return undetermined("negated comparison");
    }
    let mut dirs: BTreeSet<&'static str> = BTreeSet::new();
    for wd in &words {
        if has(w::GT_WORDS, wd) {
            dirs.insert("GT");
        } else if has(w::LT_WORDS, wd) {
            dirs.insert("LT");
        } else if let Some(a) = AGE.iter().find(|a| a.0 == wd) {
            let kind = if kinds[0] == kinds[1] { kinds[0] } else { None };
            let d = match kind {
                Some("date") => a.1,
                Some("number") => a.2,
                _ => None,
            };
            let Some(d) = d else {
                return undetermined(format!("'{wd}' without a known operand type"));
            };
            dirs.insert(d);
        }
    }
    let mut why: Vec<String> = Vec::new();
    if refs.contains_key("passive") {
        if dirs.iter().any(|d| *d != "GT") {
            return undetermined("passive comparative with a conflicting direction word");
        }
        dirs = BTreeSet::from(["LT"]);
        why.push("passive agent".to_string());
    }
    if dirs.is_empty() && !temporal.is_empty() {
        if temporal.len() == 1 {
            dirs.insert(if temporal.contains_key("before") {
                "LT"
            } else {
                "GT"
            });
        }
        why.push("before / after".to_string());
    }
    if dirs.len() != 1 {
        return undetermined(if dirs.is_empty() {
            "no direction"
        } else {
            "conflicting direction words"
        });
    }
    if refs.is_empty() {
        why.push("temporal".to_string());
    } else {
        why.extend(refs.keys().map(|k| k.to_string())); // BTreeMap: sorted(refs)
    }
    Ok(Roles::Read {
        subject,
        reference,
        expected: dirs.iter().next().copied().unwrap_or("GT"),
        why: why.join(", "),
    })
}

fn value_kind(result: Option<&str>) -> Option<String> {
    let s = result.unwrap_or("");
    if s.starts_with("VALUE ") {
        second_word(s)
    } else {
        None
    }
}

fn flip(op: &'static str) -> &'static str {
    if op == "GT" {
        "LT"
    } else {
        "GT"
    }
}

/// D2b verdict of one accepted row; `reads` = the operand reads of the row's own program
pub fn check(
    question: &str,
    row: &Row,
    reads: &[Read],
    b: &Budget,
) -> Result<RoleVerdict, OverBudget> {
    let core: Vec<&str> = row
        .program
        .iter()
        .filter(|s| s.op != "RETURN")
        .map(|s| s.op.as_str())
        .collect();
    let shape = matches!(core.len(), 3 | 4)
        && core[0] == "LOOKUP"
        && core[1] == "LOOKUP"
        && matches!(core[2], "LT" | "GT")
        && (core.len() == 3 || core[3] == "NOT");
    if !shape {
        return Ok(verdict(
            None,
            "not a binary LT / GT comparison of two lookups".into(),
            false,
        ));
    }
    let negated = core.len() == 4;
    if reads.len() < 2 {
        return Ok(verdict(None, "operands not resolved".into(), true));
    }
    let (s1, s2) = match (&reads[0].subject, &reads[1].subject) {
        (Some(a), Some(c)) => (a.as_str(), c.as_str()),
        _ => return Ok(verdict(None, "operand without a subject".into(), true)),
    };
    if s1 == s2 {
        return Ok(verdict(
            Some("D2R_SAME_OPERAND"),
            format!("both operands read '{s1}'"),
            true,
        ));
    }
    let (r1, r2) = (reads[0].relation.as_deref(), reads[1].relation.as_deref());
    if r1.unwrap_or("") != r2.unwrap_or("") {
        return Ok(verdict(
            Some("D2R_RELATION_MISMATCH"),
            format!("'{}' vs '{}'", r1.unwrap_or("None"), r2.unwrap_or("None")),
            true,
        ));
    }
    let m1 = mention_spans(question, s1, b)?;
    let m2 = mention_spans(question, s2, b)?;
    if m1.is_empty() || m2.is_empty() {
        let missing = if m1.is_empty() { s1 } else { s2 };
        return Ok(verdict(
            Some("D2R_OPERAND_NOT_NAMED"),
            format!("'{missing}' is not named in the question"),
            true,
        ));
    }
    let k1 = value_kind(reads[0].result.as_deref());
    let k2 = value_kind(reads[1].result.as_deref());
    let roles = read_roles(question, [s1, s2], [k1.as_deref(), k2.as_deref()], b)?;
    let (subject, expected) = match roles {
        Roles::Undetermined(u) => {
            return Ok(verdict(None, format!("roles undetermined: {u}"), true))
        }
        Roles::Read {
            subject, expected, ..
        } => (subject, expected),
    };
    let op: &'static str = if core[2] == "GT" { "GT" } else { "LT" };
    let rel = if subject == 0 { op } else { flip(op) };
    if !negated && rel == expected {
        return Ok(verdict(None, "consistent".into(), true));
    }
    if !negated && rel == flip(expected) {
        return Ok(RoleVerdict {
            code: Some("D2R_DIRECTION_REVERSED"),
            detail: format!("asked {expected} (subject vs reference), program computes {rel}"),
            applies: true,
            repairable: true,
        });
    }
    if negated && rel == expected {
        return Ok(verdict(
            Some("D2R_DIRECTION_NEGATED"),
            format!("asked {expected}, program computes NOT {rel}"),
            true,
        ));
    }
    Ok(verdict(
        None,
        "consistent except on ties (NOT of the reversed relation)".into(),
        true,
    ))
}
