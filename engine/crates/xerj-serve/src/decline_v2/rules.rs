//! D2, the question-program consistency checks: a line-by-line port of the frozen research/xerj_r_mvp/decline_v1/rules.py
//! (frozen 2026-10-30 05:53:17 CDT, sha256 in `words::FROZEN_SHA256`). Function names and order follow the Python file; word
//! lists are generated from it (`words.rs`), patterns are its text. Do not change a rule here: the Python file is the oracle.

use super::pyre::{Budget, OverBudget, Regex};
use super::pystr::{
    first_is_lower, first_is_upper, is_alpha, lower, py_len, split_ws, strip, upper,
};
use super::words as w;
use super::{Row, TraceStep};
use std::collections::BTreeSet;
use std::sync::LazyLock;

/// membership in a generated (sorted) word list
fn has(list: &[&str], s: &str) -> bool {
    list.binary_search(&s).is_ok()
}

fn lookup<'a>(tab: &'a [(&'a str, &'a str)], k: &str) -> Option<&'a str> {
    tab.binary_search_by_key(&k, |e| e.0).ok().map(|i| tab[i].1)
}

const MEASURE_ASIDE: [&str; 9] = [
    "recorded",
    "catalogue",
    "catalog",
    "known",
    "elapsed",
    "whole",
    "full",
    "complete",
    "exact",
];

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Decline {
    pub code: &'static str,
    pub rule: &'static str,
    pub detail: String,
}

struct Rx {
    dq: Regex,
    sq: Regex,
    paren: Regex,
    numtok: Regex,
    caps: Regex,
    token: Regex,
    poss_first: Regex,
    poss_left: Regex,
    camel: Regex,
    id_rel: Regex,
    code_like: Regex,
    year4: Regex,
    date_iso: Regex,
    date_slash: Regex,
    time_colon: Regex,
    leading_num: Regex,
}

static RX: LazyLock<Rx> = LazyLock::new(|| Rx {
    dq: Regex::new(r#""[^"]*""#, false),
    sq: Regex::new(
        r#"(?<![\w])'(?=\S)(?:[^']|'(?=\w))*?(?<=\S)'(?![\w])"#,
        false,
    ),
    paren: Regex::new(r"\([^()]*\)", false),
    numtok: Regex::new(
        r"(?<![\w])[\w.+\-/:=<>≥≤–]*\d[\w.+\-/:=<>≥≤–]*(?![\w])",
        false,
    ),
    caps: Regex::new(r"\b[A-Z][A-Z0-9]{1,}(?:[-/][A-Z0-9]+)*\b", false),
    token: Regex::new(
        r"QTITLE|PAREN|VALTOK|CODETOK|POSS|[^\W\d_](?:[\w\-/]*[^\W_])?|[?.;:,!]",
        false,
    ),
    poss_first: Regex::new(r"(?<=[\w)\]])'s(?![\w])", false),
    poss_left: Regex::new(r"(?<=\s)'s(?![\w])", false),
    camel: Regex::new(r"[a-z][A-Z]", false),
    id_rel: Regex::new(
        r"(^|[-_ ])(id|ids|identifier|identifiers|code|codes)$",
        false,
    ),
    code_like: Regex::full(r"[A-Za-z0-9][A-Za-z0-9\-_/.]*"),
    year4: Regex::full(r"\d{4}"),
    date_iso: Regex::new(r"\d{4}-\d{2}(-\d{2})?", false),
    date_slash: Regex::new(r"\d{1,2}/\d{1,2}/\d{2,4}", false),
    time_colon: Regex::new(r"\d{1,2}:\d{2}", false),
    leading_num: Regex::new(r"\s*-?\d", false),
});

fn chars(s: &str) -> Vec<char> {
    s.chars().collect()
}

fn text(s: &[char]) -> String {
    s.iter().collect()
}

/// `_QUOTES`
fn translate_quotes(q: &str) -> String {
    q.chars()
        .map(|c| match c {
            '\u{2019}' | '\u{2018}' | '\u{2032}' | '\u{B4}' => '\'',
            '\u{201C}' | '\u{201D}' | '\u{AB}' | '\u{BB}' => '"',
            c => c,
        })
        .collect()
}

/// quoted titles -> QTITLE, parentheticals -> PAREN, tokens with a digit -> VALTOK, all-caps codes -> CODETOK, possessive 's -> POSS
pub fn mask(question: &str, b: &Budget) -> Result<String, OverBudget> {
    let rx = &*RX;
    let q = chars(&translate_quotes(question));
    let q = rx.poss_first.sub(&q, " POSS ", b)?;
    let q = rx.dq.sub(&q, " QTITLE ", b)?;
    let mut q = rx.sq.sub(&q, " QTITLE ", b)?;
    for _ in 0..3 {
        q = rx.paren.sub(&q, " PAREN ", b)?;
    }
    let q = rx.numtok.sub(&q, " VALTOK ", b)?;
    let q = rx.caps.sub_with(&q, b, |m| {
        let s = text(m);
        if has(w::KEEP_CAPS, &upper(&s)) || s == "IDs" {
            s
        } else {
            " CODETOK ".to_string()
        }
    })?;
    let q = rx.poss_left.sub(&q, " POSS ", b)?;
    Ok(text(&q))
}

pub fn tokenize(masked: &str, b: &Budget) -> Result<Vec<String>, OverBudget> {
    let cs = chars(masked);
    Ok(RX
        .token
        .find_all(&cs, b)?
        .iter()
        .map(|m| text(&cs[m.start..m.end]))
        .collect())
}

fn is_masked(t: &str) -> bool {
    has(w::MASKED, t)
}

fn is_punct(t: &str) -> bool {
    matches!(t, "?" | "." | ";" | ":" | "," | "!")
}

fn is_plural(tok: &str) -> bool {
    let mut lw = lower(tok);
    if is_masked(tok) || has(w::UNITS, &lw) || py_len(&lw) < 3 {
        return false;
    }
    if has(w::IRREGULAR_PLURALS, &lw) {
        return !has(w::UNITS, &lw);
    }
    if lw == "ids" {
        return true;
    }
    if has(w::SINGULAR_S, &lw) || has(w::AUX, &lw) || has(w::DETERMINERS, &lw) {
        return false;
    }
    if lw.contains('-') {
        lw = lw.rsplit('-').next().unwrap_or("").to_string();
        if lw == "ids" {
            return true;
        }
        if py_len(&lw) < 3 || has(w::SINGULAR_S, &lw) {
            return false;
        }
    }
    lw.ends_with('s')
        && !["ss", "us", "is", "ous", "ics", "'s"]
            .iter()
            .any(|e| lw.ends_with(e))
        && lw.chars().rev().nth(1) != Some('\'')
}

fn is_attr_word(tok: &str, b: &Budget) -> Result<bool, OverBudget> {
    if is_masked(tok) || is_punct(tok) {
        return Ok(false);
    }
    let up = upper(tok);
    if up == "ID" || up == "IDS" || tok == "IDs" {
        return Ok(true);
    }
    if py_len(tok) == 1 {
        return Ok(false);
    }
    if first_is_upper(tok) {
        return Ok(RX.camel.search(&chars(tok), b)?.is_some());
    }
    let lw = lower(tok);
    Ok(!has(w::AUX, &lw) && !has(w::PREPOSITIONS, &lw) && !has(w::DETERMINERS, &lw))
}

#[derive(Clone, Debug)]
pub struct Clause {
    pub kind: &'static str,
    pub toks: Vec<String>,
    pub opener: String,
    pub span: Vec<String>,
    pub sentence: usize,
    pub question_mark: bool,
}

#[derive(Clone, Debug)]
pub struct QInfo {
    pub masked: String,
    pub tokens: Vec<String>,
    pub clauses: Vec<Clause>,
    pub has_polar: bool,
    pub has_wh: bool,
    pub has_request: bool,
    pub has_question_mark: bool,
    pub low: Vec<String>,
    pub spans: Vec<Vec<String>>,
    pub choice: bool,
    pub has_action: bool,
}

fn sentences(tokens: &[String]) -> Vec<Vec<String>> {
    let mut out: Vec<Vec<String>> = Vec::new();
    let mut cur: Vec<String> = Vec::new();
    for (i, t) in tokens.iter().enumerate() {
        cur.push(t.clone());
        let stop = matches!(t.as_str(), "?" | ";" | "!")
            || (t == "."
                && !(i > 0 && {
                    let p = &tokens[i - 1];
                    (py_len(p) == 1 && p.chars().next().is_some_and(is_alpha))
                        || has(w::ABBREV, &lower(p))
                }));
        if stop {
            out.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out.into_iter()
        .filter(|s| {
            s.iter()
                .any(|t| !matches!(t.as_str(), "?" | "." | ";" | "!" | "," | ":"))
        })
        .collect()
}

fn split_clauses(sent: &[String]) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut start: Option<usize> = None;
    for (i, t) in sent.iter().enumerate() {
        let low = lower(t);
        let nxt = sent.get(i + 1).map(|x| lower(x)).unwrap_or_default();
        if t == "," || t == ":" {
            if let Some(s) = start {
                out.push((s, i));
            }
            start = None;
            continue;
        }
        if matches!(low.as_str(), "and" | "then" | "but")
            && start.is_some()
            && (has(w::WH, &nxt)
                || has(w::REQUEST_VERBS, &nxt)
                || (has(w::AUX, &nxt) && low != "and"))
        {
            out.push((start.unwrap(), i));
            start = Some(i);
            continue;
        }
        if low == "then" && start.is_some() {
            out.push((start.unwrap(), i));
            start = None;
            continue;
        }
        if start.is_none() {
            start = Some(i);
        }
    }
    if let Some(s) = start {
        out.push((s, sent.len()));
    }
    out
}

fn opener(cl: &[String]) -> (String, usize) {
    let mut k = 0;
    while k < cl.len() && has(w::LEADING_FILLERS, &lower(&cl[k])) {
        k += 1;
    }
    (cl.get(k).map(|t| lower(t)).unwrap_or_default(), k)
}

fn participle(toks: &[String], i: usize) -> bool {
    let wd = lower(&toks[i]);
    if !(wd.ends_with("ing") && py_len(&wd) > 4) {
        return false;
    }
    let nxt = toks.get(i + 1).map(String::as_str).unwrap_or("");
    let l = lower(nxt);
    has(w::DETERMINERS, &l)
        || has(w::PREPOSITIONS, &l)
        || is_masked(nxt)
        || first_is_upper(nxt)
        || has(w::FORMAT_OBJECTS, &l)
        || matches!(l.as_str(), "ties" | "nulls" | "missing" | "duplicates")
}

fn verb_object(cl: &[String], j: usize) -> bool {
    if j >= cl.len() {
        return false;
    }
    let t = &cl[j];
    let l = lower(t);
    if has(w::DETERMINERS, &l)
        || matches!(
            l.as_str(),
            "me" | "us" | "my" | "our" | "him" | "her" | "them" | "it"
        )
        || is_masked(t)
        || first_is_upper(t)
    {
        return true;
    }
    j + 1 < cl.len()
        && (is_masked(&cl[j + 1]) || first_is_upper(&cl[j + 1]))
        && first_is_lower(t)
        && !has(w::PREPOSITIONS, &l)
}

fn span(cl: &[String], start: usize, wh: bool, b: &Budget) -> Result<Vec<String>, OverBudget> {
    let mut out: Vec<String> = Vec::new();
    let mut i = start;
    if wh && i < cl.len() && matches!(lower(&cl[i]).as_str(), "is" | "are" | "was" | "were") {
        i += 1;
    }
    for j in i..cl.len().min(i + 8) {
        if cl[j] == "POSS" {
            i = j + 1;
            break;
        }
        let l = lower(&cl[j]);
        if has(w::PREPOSITIONS, &l) || has(w::VERB_STOPS, &l) {
            break;
        }
    }
    while i < cl.len() {
        let t = &cl[i];
        let low = lower(t);
        if matches!(t.as_str(), "?" | "." | ";" | "!" | ",") {
            if t == ","
                && i + 1 < cl.len()
                && (is_attr_word(&cl[i + 1], b)? || has(w::DETERMINERS, &lower(&cl[i + 1])))
            {
                out.push(t.clone());
                i += 1;
                continue;
            }
            break;
        }
        if low == "in"
            && i + 1 < cl.len()
            && (has(w::UNITS, &lower(&cl[i + 1])) || cl[i + 1] == "PAREN")
        {
            i += 2;
            continue;
        }
        if low == "in"
            && i + 2 < cl.len()
            && has(w::UNITS, &lower(&cl[i + 2]))
            && !is_masked(&cl[i + 1])
        {
            i += 3;
            continue;
        }
        if low == "or" && i + 1 < cl.len() && has(w::FORMAT_OBJECTS, &lower(&cl[i + 1])) {
            break;
        }
        if !out.is_empty()
            && (has(w::PREPOSITIONS, &low)
                || has(w::RELATIVES, &low)
                || has(w::VERB_STOPS, &low)
                || participle(cl, i))
        {
            break;
        }
        if out.is_empty() && has(w::PREPOSITIONS, &low) && low != "of" {
            break;
        }
        out.push(t.clone());
        i += 1;
    }
    Ok(out)
}

pub fn analyse_question(question: &str, b: &Budget) -> Result<QInfo, OverBudget> {
    let masked = mask(question, b)?;
    let tokens = tokenize(&masked, b)?;
    let mut clauses: Vec<Clause> = Vec::new();
    let sents = sentences(&tokens);
    for (si, sent) in sents.iter().enumerate() {
        let qmark = sent.last().is_some_and(|t| t == "?");
        for (a, z) in split_clauses(sent) {
            let cl = &sent[a..z];
            let (mut op, k) = opener(cl);
            let mut kind: &'static str = "context";
            let mut sp: Vec<String> = Vec::new();
            if has(w::CONDITIONAL, &op) {
                kind = "conditional";
            } else if has(w::AUX, &op) {
                kind = if qmark { "polar" } else { "context" };
                if matches!(op.as_str(), "do" | "does")
                    && k + 1 < cl.len()
                    && lower(&cl[k + 1]) == "not"
                {
                    kind = "format";
                }
                let nx: Vec<String> = cl[(k + 1).min(cl.len())..(k + 4).min(cl.len())]
                    .iter()
                    .map(|t| lower(t))
                    .collect();
                if nx.len() >= 2
                    && nx[0] == "you"
                    && (has(w::REQUEST_VERBS, &nx[1])
                        || (nx[1] == "please" && nx.len() > 2 && has(w::REQUEST_VERBS, &nx[2])))
                    && matches!(op.as_str(), "can" | "could" | "would" | "will")
                {
                    let v = k + 2 + usize::from(nx[1] == "please");
                    let mut st = v + 1;
                    while st < cl.len() && matches!(lower(&cl[st]).as_str(), "me" | "us") {
                        st += 1;
                    }
                    if !(st < cl.len() && matches!(lower(&cl[st]).as_str(), "whether" | "if")) {
                        kind = "request";
                        sp = span(sent, a + st, false, b)?;
                    }
                } else if nx.len() >= 2
                    && nx[0] == "you"
                    && nx[1] == "know"
                    && matches!(op.as_str(), "do" | "does" | "did")
                {
                    let st = k + 3;
                    if st < cl.len() && has(w::WH, &lower(&cl[st])) {
                        kind = "wh";
                        op = lower(&cl[st]);
                        sp = span(sent, a + st + 1, true, b)?;
                    } else if !(st < cl.len()
                        && matches!(lower(&cl[st]).as_str(), "whether" | "if"))
                    {
                        kind = "request";
                        sp = span(sent, a + st, false, b)?;
                    }
                }
            } else if has(w::WH, &op) {
                let next_aux = k + 1 < cl.len() && has(w::AUX, &lower(&cl[k + 1]));
                if matches!(op.as_str(), "when" | "where") && !qmark && !next_aux {
                    kind = "conditional";
                } else if matches!(op.as_str(), "where" | "when") && !qmark && a > 0 {
                    kind = "conditional";
                } else {
                    kind = "wh";
                    sp = span(sent, a + k + 1, true, b)?;
                }
            } else if (has(w::REQUEST_VERBS, &op)
                && !(k + 1 < cl.len()
                    && matches!(
                        lower(&cl[k + 1]).as_str(),
                        "and" | "or" | "," | "." | "?" | ";" | "&"
                    ))
                && k + 1 < cl.len())
                || (op == "i"
                    && k + 1 < cl.len()
                    && matches!(lower(&cl[k + 1]).as_str(), "need" | "want" | "would"))
            {
                let mut start = k + 1;
                if op == "look" && start < cl.len() && lower(&cl[start]) == "up" {
                    start += 1;
                }
                if op == "i" {
                    start = k + 2;
                    while start < cl.len()
                        && matches!(
                            lower(&cl[start]).as_str(),
                            "like" | "to" | "know" | "see" | "have" | "get"
                        )
                    {
                        start += 1;
                    }
                }
                let obj: Vec<&String> = cl[start.min(cl.len())..(start + 3).min(cl.len())]
                    .iter()
                    .filter(|t| !has(w::DETERMINERS, &lower(t)))
                    .collect();
                let mut st2 = start;
                while st2 < cl.len() && matches!(lower(&cl[st2]).as_str(), "me" | "us" | "out") {
                    st2 += 1;
                }
                if !obj.is_empty() && has(w::FORMAT_OBJECTS, &lower(obj[0])) {
                    kind = "format";
                } else if st2 < cl.len() && matches!(lower(&cl[st2]).as_str(), "whether" | "if") {
                    kind = "polar";
                } else {
                    kind = "request";
                    sp = span(sent, a + start, false, b)?;
                }
            } else if has(w::ACTION_VERBS, &op) && verb_object(cl, k + 1) {
                kind = "action";
            } else if qmark {
                for j in 1..cl.len() {
                    let prev = lower(&cl[j - 1]);
                    if has(w::WH, &lower(&cl[j]))
                        && (has(w::PREPOSITIONS, &prev)
                            || matches!(prev.as_str(), "is" | "are" | "was" | "were"))
                    {
                        kind = "wh";
                        sp = span(sent, a + j + 1, true, b)?;
                        op = lower(&cl[j]);
                        break;
                    }
                }
            }
            clauses.push(Clause {
                kind,
                toks: cl.to_vec(),
                opener: op,
                span: sp,
                sentence: si,
                question_mark: qmark,
            });
        }
    }
    let low: Vec<String> = clauses
        .iter()
        .filter(|c| !matches!(c.kind, "conditional" | "format"))
        .flat_map(|c| c.toks.iter().map(|t| lower(t)))
        .collect();
    let spans: Vec<Vec<String>> = clauses
        .iter()
        .filter(|c| matches!(c.kind, "wh" | "request") && !c.span.is_empty())
        .map(|c| c.span.clone())
        .collect();
    let mut choice = false;
    for c in &clauses {
        let tl = clause_low(c);
        for (j, t) in tl.iter().enumerate() {
            if t == "or" && 0 < j && j + 1 < tl.len() {
                let left = &c.toks[j - 1];
                let mut r = j + 1;
                while r + 1 < tl.len() && has(w::DETERMINERS, &tl[r]) {
                    r += 1;
                }
                let right = &c.toks[r];
                if (is_masked(left) || first_is_upper(left) || py_len(left) == 1)
                    && (is_masked(right) || first_is_upper(right) || py_len(right) == 1)
                {
                    let up = upper(right);
                    if !has(w::FORMAT_OBJECTS, &up) && up != "UNKNOWN" && up != "NULL" {
                        choice = true;
                    }
                }
            }
        }
    }
    Ok(QInfo {
        has_polar: clauses.iter().any(|c| c.kind == "polar"),
        has_wh: clauses.iter().any(|c| c.kind == "wh"),
        has_request: clauses.iter().any(|c| c.kind == "request"),
        has_question_mark: tokens.iter().any(|t| t == "?"),
        has_action: clauses.iter().any(|c| c.kind == "action"),
        masked,
        tokens,
        clauses,
        low,
        spans,
        choice,
    })
}

#[derive(Clone, Debug)]
pub struct PInfo {
    pub ops: Vec<String>,
    pub final_op: Option<String>,
    pub value_type: Option<String>,
    pub value: Option<String>,
    pub returned_relation: Option<String>,
    pub operand_relations: Vec<String>,
    pub answer: Option<String>,
    pub yes_no: bool,
}

pub fn relation_of(fact_id: &str) -> String {
    fact_id
        .rsplit('/')
        .next()
        .unwrap_or("")
        .split('#')
        .next()
        .unwrap_or("")
        .to_string()
}

fn result_type(res: Option<&str>) -> (Option<String>, Option<String>) {
    let s = res.unwrap_or("");
    if s.starts_with("VALUE ") {
        let mut p = s.splitn(3, ' ');
        p.next();
        let ty = p.next().unwrap_or("text").to_string();
        let val = p.next().unwrap_or("").to_string();
        return (Some(ty), Some(val));
    }
    if s == "TRUE" || s == "FALSE" {
        return (Some("yesno".to_string()), Some(lower(s)));
    }
    (None, None)
}

pub fn analyse_program(row: &Row) -> PInfo {
    let ops: Vec<String> = row
        .program
        .iter()
        .filter(|s| s.op != "RETURN")
        .map(|s| s.op.clone())
        .collect();
    let trace: Vec<&TraceStep> = row.trace.iter().filter(|t| t.op != "RETURN").collect();
    let final_op = ops.last().cloned();
    let (mut vt, mut val) = (None, None);
    if let Some(t) = trace.last() {
        (vt, val) = result_type(t.result.as_deref());
    }
    let a = row
        .answer
        .as_deref()
        .map(|s| lower(strip(s)).trim_end_matches('.').to_string())
        .unwrap_or_default();
    let yes_no = matches!(a.as_str(), "yes" | "no" | "true" | "false")
        || matches!(vt.as_deref(), Some("yesno" | "boolean"));
    let rels: Vec<String> = row.evidence_ids.iter().map(|e| relation_of(e)).collect();
    let mut returned = None;
    if let Some(t) = trace.last() {
        if t.op == "CHAIN" {
            if let (Some(hop), Some(p)) = (&t.hop_candidate_ids, t.pointer) {
                if !hop.is_empty() && p >= 0 && (p as usize) < hop.len() {
                    returned = Some(relation_of(&hop[p as usize]));
                }
            }
        }
    }
    let uniq: BTreeSet<&String> = rels.iter().collect();
    if returned.is_none()
        && matches!(final_op.as_deref(), Some("LOOKUP" | "CHAIN"))
        && !rels.is_empty()
        && uniq.len() == 1
    {
        returned = Some(rels[0].clone());
    }
    PInfo {
        ops,
        final_op,
        value_type: vt,
        value: val,
        returned_relation: returned,
        operand_relations: uniq.into_iter().cloned().collect(),
        answer: row.answer.clone(),
        yes_no,
    }
}

fn is_id_relation(rel: Option<&str>, b: &Budget) -> Result<bool, OverBudget> {
    let Some(rel) = rel.filter(|r| !r.is_empty()) else {
        return Ok(false);
    };
    let r = lower(rel);
    Ok(RX.id_rel.search(&chars(&r), b)?.is_some()
        || matches!(r.as_str(), "identifier" | "id" | "code"))
}

fn is_code_like(v: Option<&str>, b: &Budget) -> Result<bool, OverBudget> {
    let s = strip(v.unwrap_or(""));
    if s.is_empty() || s.contains(' ') || RX.code_like.match_at_start(&chars(s), b)?.is_none() {
        return Ok(false);
    }
    // s is ASCII here (the pattern); str.isupper(): some cased character, none lower
    let upper_like =
        s.chars().any(|c| c.is_ascii_uppercase()) && !s.chars().any(|c| c.is_ascii_lowercase());
    Ok(s.chars().any(|c| c.is_ascii_digit()) || (upper_like && py_len(s) <= 12))
}

fn year_or_date_like(v: Option<&str>, b: &Budget) -> Result<bool, OverBudget> {
    let s = chars(strip(v.unwrap_or("")));
    Ok(RX.year4.match_at_start(&s, b)?.is_some()
        || RX.date_iso.match_at_start(&s, b)?.is_some()
        || RX.date_slash.match_at_start(&s, b)?.is_some()
        || RX.time_colon.match_at_start(&s, b)?.is_some())
}

fn has_phrase(low: &[String], phrase: &[&str]) -> bool {
    let n = phrase.len();
    if low.len() < n {
        return false;
    }
    (0..=low.len() - n).any(|i| (0..n).all(|k| low[i + k] == phrase[k]))
}

fn eligible(c: &Clause, i: usize) -> bool {
    let t = &c.toks[i];
    if first_is_lower(t) {
        return true;
    }
    let mut k = 0;
    while k < c.toks.len() && has(w::LEADING_FILLERS, &lower(&c.toks[k])) {
        k += 1;
    }
    let nxt = c.toks.get(i + 1).map(String::as_str).unwrap_or("");
    i == k && !(first_is_upper(nxt) && !is_masked(nxt))
}

fn content_clauses(q: &QInfo) -> impl Iterator<Item = &Clause> {
    q.clauses
        .iter()
        .filter(|c| !matches!(c.kind, "conditional" | "format"))
}

fn clause_low(c: &Clause) -> Vec<String> {
    c.toks.iter().map(|t| lower(t)).collect()
}

fn multi_value_cues(q: &QInfo, b: &Budget) -> Result<Vec<String>, OverBudget> {
    let mut cues: Vec<String> = Vec::new();
    const SKIP: [&str; 6] = [
        "recorded",
        "associated",
        "corresponding",
        "respective",
        "unmodified",
        "reported",
    ];
    const PH: [&[&str]; 15] = [
        &["together", "with"],
        &["along", "with"],
        &["alongside"],
        &["as", "well", "as"],
        &["in", "addition", "to"],
        &["as", "a", "pair"],
        &["pair"],
        &["pairs"],
        &["and", "its"],
        &["with", "its"],
        &["and", "their"],
        &["both"],
        &["each"],
        &["every"],
        &["all"],
    ];
    for sp in &q.spans {
        let sl: Vec<String> = sp.iter().map(|t| lower(t)).collect();
        for (i, t) in sl.iter().enumerate() {
            if (t == "and" || t == "&") && 0 < i && i + 1 < sp.len() {
                let left = &sp[i - 1];
                let mut j = i + 1;
                while j < sp.len()
                    && (has(w::DETERMINERS, &sl[j]) || SKIP.contains(&sl[j].as_str()))
                {
                    j += 1;
                }
                if let Some(right) = sp.get(j) {
                    let ll = lower(left);
                    if is_attr_word(left, b)?
                        && is_attr_word(right, b)?
                        && !has(w::UNITS, &ll)
                        && ll != "and"
                    {
                        cues.push(format!("'{left} and {right}'"));
                    }
                }
            }
            if t == ","
                && 0 < i
                && i + 1 < sp.len()
                && is_attr_word(&sp[i - 1], b)?
                && is_attr_word(&sp[i + 1], b)?
                && sl[i..].iter().any(|x| x == "and")
            {
                cues.push(format!("list '{}, {}'", sp[i - 1], sp[i + 1]));
            }
        }
        for ph in PH {
            if has_phrase(&sl, ph) {
                cues.push(ph.join(" "));
            }
        }
    }
    for c in content_clauses(q) {
        if c.kind == "request" && matches!(c.opener.as_str(), "list" | "enumerate") {
            cues.push(format!("'{}' request", c.opener));
        }
    }
    for c in content_clauses(q) {
        let tl = clause_low(c);
        if c.kind == "wh" && tl.first().is_some_and(|t| t == "and") {
            cues.push(format!(
                "second question ('and {}')",
                tl.get(1).map(String::as_str).unwrap_or("")
            ));
        }
    }
    let wh_sent: BTreeSet<usize> = q
        .clauses
        .iter()
        .filter(|c| c.kind == "wh" && c.question_mark)
        .map(|c| c.sentence)
        .collect();
    if wh_sent.len() >= 2 {
        cues.push("two wh-questions".to_string());
    }
    for c in content_clauses(q) {
        if matches!(c.kind, "request" | "wh") {
            let tl = clause_low(c);
            let phs: [&[&str]; 4] = [
                &["as", "well", "as"],
                &["together", "with"],
                &["as", "a", "pair"],
                &["alongside"],
            ];
            for ph in phs {
                let j = ph.join(" ");
                if has_phrase(&tl, ph) && !cues.contains(&j) {
                    cues.push(j);
                }
            }
        }
    }
    Ok(cues)
}

fn set_cues(q: &QInfo, b: &Budget) -> Result<Vec<String>, OverBudget> {
    let mut cues: Vec<String> = Vec::new();
    for c in content_clauses(q) {
        let tl = clause_low(c);
        for (i, t) in tl.iter().enumerate() {
            if !eligible(c, i) {
                continue;
            }
            if has(w::SET_WORDS, t) {
                let prev = if i > 0 { tl[i - 1].as_str() } else { "" };
                if matches!(t.as_str(), "share" | "shares")
                    && (has(w::DETERMINERS, prev)
                        || has(w::WH, prev)
                        || matches!(prev, "market" | "per" | "of"))
                {
                    continue;
                }
                if matches!(t.as_str(), "distinct" | "unique" | "remaining") {
                    let mut plural = false;
                    for j in i + 1..tl.len().min(i + 4) {
                        if is_plural(&c.toks[j]) && is_attr_word(&c.toks[j], b)? {
                            plural = true;
                            break;
                        }
                    }
                    if !plural {
                        continue;
                    }
                }
                cues.push(t.clone());
            }
            if t == "in" && tl.get(i + 1).is_some_and(|x| x == "common") && eligible(c, i + 1) {
                cues.push("in common".to_string());
            }
            if t == "but" && tl.get(i + 1).is_some_and(|x| x == "not" || x == "no") {
                cues.push(format!("but {}", tl[i + 1]));
            }
            if t == "set" && tl.get(i + 1).is_some_and(|x| x == "of") {
                cues.push("set of".to_string());
            }
            if t == "sets" {
                cues.push(t.clone());
            }
        }
        for ph in w::EXIST_PHRASES {
            if has_phrase(&tl, ph) {
                if *ph == ["any"]
                    && (has_phrase(&tl, &["if", "any"]) || has_phrase(&tl, &["any", "other"]))
                {
                    continue;
                }
                if *ph == ["some"] && c.kind != "polar" {
                    continue;
                }
                cues.push(format!("existential '{}'", ph.join(" ")));
            }
        }
    }
    Ok(cues)
}

fn measure_after(tl: &[String], i: usize) -> bool {
    let mut j = i;
    while j < tl.len() && MEASURE_ASIDE.contains(&tl[j].as_str()) {
        j += 1;
    }
    j < tl.len() && has(w::UNITS, &tl[j])
}

/// (word, family) pairs of aggregation / extremum / grouping / arithmetic cues outside conditional and format clauses
pub fn aggregate_cues(q: &QInfo) -> Vec<(String, String)> {
    let mut cues: Vec<(String, String)> = Vec::new();
    let fam = |t: &str, d: &str| lookup(w::CUE_FAMILY, t).unwrap_or(d).to_string();
    for c in content_clauses(q) {
        let tl = clause_low(c);
        for (i, t) in tl.iter().enumerate() {
            if !eligible(c, i) {
                continue;
            }
            let t = t.as_str();
            let prev = if i > 0 { tl[i - 1].as_str() } else { "" };
            let nxt = tl.get(i + 1).map(String::as_str).unwrap_or("");
            if t == "how" && nxt == "many" {
                if !measure_after(&tl, i + 2) {
                    cues.push(("how many".into(), "count".into()));
                }
            } else if t == "number" && nxt == "of" {
                if !measure_after(&tl, i + 2)
                    && (matches!(
                        prev,
                        "" | "the"
                            | "a"
                            | "an"
                            | "total"
                            | "what"
                            | "which"
                            | "how"
                            | "of"
                            | "same"
                            | "greater"
                            | "smaller"
                            | "higher"
                            | "lower"
                            | "larger"
                            | "give"
                            | "return"
                            | "find"
                            | "report"
                            | "count"
                    ) || has(w::VERB_STOPS, prev))
                {
                    cues.push(("number of".into(), "count".into()));
                }
            } else if has(w::AGG_WORDS, t) {
                let back = &tl[i.saturating_sub(4)..i];
                if matches!(t, "mean" | "means")
                    && (matches!(nxt, "that" | "by" | "to" | "?" | "." | "")
                        || back
                            .iter()
                            .any(|x| matches!(x.as_str(), "does" | "do" | "did")))
                {
                    continue;
                }
                if t == "count" && nxt == "as" {
                    continue;
                }
                cues.push((t.to_string(), fam(t, "agg")));
            } else if has(w::SUPERLATIVES, t) {
                cues.push((t.to_string(), fam(t, "ext")));
            } else if has(w::EXTREMUM_WORDS, t) {
                if matches!(t, "least" | "most") && prev == "at" {
                    continue;
                }
                if t == "most" && nxt == "of" {
                    continue;
                }
                if matches!(t, "first" | "last")
                    && ((matches!(nxt, "name" | "names" | "time") && t == "last" && nxt == "time")
                        || matches!(nxt, "name" | "names")
                        || prev == "at")
                {
                    continue;
                }
                if matches!(t, "last" | "first") && has(w::TIME_NOUNS, nxt) && nxt != "time" {
                    continue;
                }
                if t == "top" && nxt == "of" {
                    continue;
                }
                if t == "bottom" && nxt == "of" {
                    continue;
                }
                cues.push((t.to_string(), fam(t, "ext")));
            } else if has(w::ARITH_WORDS, t) {
                if matches!(t, "difference" | "differences")
                    && !matches!(nxt, "between" | "in" | "of")
                {
                    continue;
                }
                cues.push((t.to_string(), "arith".into()));
            } else if t == "per" {
                if has(w::UNITS, prev)
                    || has(w::UNITS, nxt)
                    || prev == "valtok"
                    || prev == "percent"
                    || matches!(
                        nxt,
                        "capita"
                            | "annum"
                            | "person"
                            | "head"
                            | "item"
                            | "piece"
                            | "share"
                            | "unit"
                            | "gallon"
                            | "serving"
                            | "cent"
                    )
                {
                    continue;
                }
                cues.push(("per".into(), "group".into()));
            }
        }
        for ph in w::GROUP_PHRASES {
            if has_phrase(&tl, ph) {
                cues.push((ph.join(" "), "group".into()));
            }
        }
    }
    cues
}

pub fn count_comparison_cues(q: &QInfo, b: &Budget) -> Result<Vec<String>, OverBudget> {
    let mut cues = Vec::new();
    for c in content_clauses(q) {
        let tl = clause_low(c);
        for (i, t) in tl.iter().enumerate() {
            if has(w::COUNT_COMPARATIVES, t) && eligible(c, i) {
                if t == "less" && tl.get(i + 1).is_some_and(|x| x == "than") {
                    continue;
                }
                for j in i + 1..tl.len().min(i + 4) {
                    if matches!(tl[j].as_str(), "than" | "?" | "." | ",") {
                        break;
                    }
                    if is_plural(&c.toks[j]) && is_attr_word(&c.toks[j], b)? {
                        cues.push(format!("{t} ... {}", tl[j]));
                        break;
                    }
                }
            }
        }
    }
    Ok(cues)
}

fn comparative_cues(q: &QInfo) -> Vec<String> {
    let mut out = Vec::new();
    for c in content_clauses(q) {
        if !matches!(c.kind, "polar" | "wh") {
            continue;
        }
        let tl = clause_low(c);
        for (i, t) in tl.iter().enumerate() {
            if !eligible(c, i) {
                continue;
            }
            if has(w::COMPARATIVES, t) {
                out.push(t.clone());
            } else if matches!(t.as_str(), "more" | "less")
                && i + 1 < tl.len()
                && tl[i + 1] != "than"
                && !is_plural(&c.toks[i + 1])
            {
                out.push(format!("{t} {}", tl[i + 1]));
            }
        }
    }
    out
}

fn requests_id(q: &QInfo, b: &Budget) -> Result<bool, OverBudget> {
    for sp in &q.spans {
        for (i, t) in sp.iter().enumerate() {
            let low = lower(t);
            let base = if low.contains('-') {
                low.rsplit('-').next().unwrap_or("").to_string()
            } else {
                low.clone()
            };
            if has(w::ID_WORDS, &low)
                || has(w::ID_WORDS, &base)
                || t == "IDs"
                || (low == "number" && i > 0 && is_attr_word(&sp[i - 1], b)?)
            {
                let nxt = sp.get(i + 1).map(String::as_str).unwrap_or("");
                if matches!(nxt, "VALTOK" | "CODETOK") {
                    continue;
                }
                return Ok(true);
            }
        }
    }
    Ok(false)
}

fn requested_place_field(q: &QInfo) -> Option<String> {
    for sp in &q.spans {
        let heads: Vec<String> = sp
            .iter()
            .filter(|t| !has(w::DETERMINERS, &lower(t)) && !is_masked(t) && t.as_str() != ",")
            .map(|t| lower(t))
            .collect();
        let named: Vec<String> = heads
            .iter()
            .filter(|h| {
                !matches!(
                    h.as_str(),
                    "name" | "names" | "label" | "labels" | "value" | "values"
                )
            })
            .cloned()
            .collect();
        let heads = if named.is_empty() { heads } else { named };
        if let Some(h) = heads.last() {
            let h = if let Some(s) = h.strip_suffix("ies") {
                format!("{s}y")
            } else if h.ends_with('s') && has(w::PLACE_FIELDS, &h[..h.len() - 1]) {
                h[..h.len() - 1].to_string()
            } else {
                h.clone()
            };
            if has(w::PLACE_FIELDS, &h) {
                return Some(h);
            }
        }
    }
    None
}

fn requested_types(q: &QInfo) -> BTreeSet<&'static str> {
    let mut types = BTreeSet::new();
    for c in content_clauses(q) {
        if c.kind != "wh" {
            continue;
        }
        let tl = clause_low(c);
        let k = tl.iter().position(|t| *t == c.opener).unwrap_or(0);
        let nxt = tl.get(k + 1).map(String::as_str).unwrap_or("");
        let op = c.opener.as_str();
        if op == "when" && (has(w::AUX, nxt) || nxt.is_empty()) {
            types.insert("date");
        }
        if op == "where" && (has(w::AUX, nxt) || nxt.is_empty()) {
            types.insert("place");
        }
        if matches!(op, "who" | "whom")
            && (has(w::AUX, nxt) || has(w::VERB_STOPS, nxt) || nxt.ends_with("ed"))
        {
            types.insert("person");
        }
        if op == "how" && has(w::MEASURE_ADJ, nxt) {
            types.insert("number");
        }
        if op == "how" && nxt == "many" && measure_after(&tl, k + 2) {
            types.insert("number");
        }
    }
    for sp in &q.spans {
        let heads: Vec<String> = sp
            .iter()
            .map(|t| lower(t))
            .filter(|x| !has(w::DETERMINERS, x) && !is_masked(x))
            .collect();
        let Some(h) = heads.last() else { continue };
        if has(w::DATE_HEADS, h) {
            types.insert("date");
        }
        if has(w::PLACE_HEADS, h) {
            types.insert("place");
        }
        if matches!(
            h.as_str(),
            "amount"
                | "price"
                | "cost"
                | "quantity"
                | "length"
                | "width"
                | "height"
                | "depth"
                | "distance"
                | "weight"
                | "mass"
                | "magnitude"
                | "elevation"
                | "area"
                | "size"
                | "volume"
                | "population"
                | "speed"
                | "temperature"
        ) {
            types.insert("number");
        }
    }
    types
}

fn split_field(r: &str) -> Vec<&str> {
    r.split(['-', '_', ' ']).collect()
}

fn cue_matched_by_field(fam: &str, p: &PInfo) -> bool {
    let Some(stems) = w::CUE_FIELD_STEMS.iter().find(|e| e.0 == fam).map(|e| e.1) else {
        return false;
    };
    let mut rels: Vec<&str> = Vec::new();
    if let Some(r) = p.returned_relation.as_deref().filter(|r| !r.is_empty()) {
        rels.push(r);
    }
    rels.extend(p.operand_relations.iter().map(String::as_str));
    rels.iter().any(|r| {
        let l = lower(r);
        let parts = split_field(&l);
        stems.iter().any(|s| parts.contains(s))
    })
}

/// a Python f-string `{x}` of an optional text
fn opt(s: &Option<String>) -> &str {
    s.as_deref().unwrap_or("None")
}

fn unique_join(v: &[String], sep: &str) -> String {
    let mut seen: Vec<&String> = Vec::new();
    for x in v {
        if !seen.contains(&x) {
            seen.push(x);
        }
    }
    seen.iter()
        .map(|s| s.as_str())
        .collect::<Vec<_>>()
        .join(sep)
}

const SCALAR_OPS: [&str; 8] = [
    "LOOKUP", "CHAIN", "LT", "GT", "EQ", "NOT", "PICK_LT", "PICK_GT",
];
const ORDER_OPS: [&str; 4] = ["LT", "GT", "PICK_LT", "PICK_GT"];
const COMPARE_OPS: [&str; 3] = ["LT", "GT", "EQ"];
const PICK_OPS: [&str; 2] = ["PICK_LT", "PICK_GT"];

pub fn check_all(question: &str, row: &Row, b: &Budget) -> Result<Vec<Decline>, OverBudget> {
    let q = analyse_question(question, b)?;
    let p = analyse_program(row);
    let mut out: Vec<Decline> = Vec::new();
    let any_op = |set: &[&str]| p.ops.iter().any(|o| set.contains(&o.as_str()));
    let scalar = !p.ops.is_empty() && p.ops.iter().all(|o| SCALAR_OPS.contains(&o.as_str()));
    let d = |code, rule, detail: String| Decline { code, rule, detail };
    // (a)
    if scalar {
        let mv = multi_value_cues(&q, b)?;
        if !mv.is_empty() {
            out.push(d("D2A_MULTI_VALUE", "a", unique_join(&mv, "; ")));
        }
    }
    // (b)
    if scalar {
        let sc = set_cues(&q, b)?;
        if !sc.is_empty() {
            let ex: Vec<String> = sc
                .iter()
                .filter(|s| s.starts_with("existential"))
                .cloned()
                .collect();
            let st: Vec<String> = sc
                .iter()
                .filter(|s| !s.starts_with("existential"))
                .cloned()
                .collect();
            if !st.is_empty() {
                out.push(d("D2B_SET", "b", unique_join(&st, "; ")));
            }
            if !ex.is_empty() {
                out.push(d("D2B_EXISTENTIAL", "b", unique_join(&ex, "; ")));
            }
        }
    }
    // (c)
    let ag = aggregate_cues(&q);
    let mut unmatched: Vec<String> = Vec::new();
    for (word, fam) in &ag {
        if matches!(fam.as_str(), "max" | "min" | "first" | "last" | "ext")
            && any_op(&PICK_OPS)
            && q.choice
        {
            continue;
        }
        if cue_matched_by_field(fam, &p) {
            continue;
        }
        unmatched.push(word.clone());
    }
    if !unmatched.is_empty() && scalar {
        let fams: BTreeSet<&str> = ag
            .iter()
            .filter(|(wd, _)| unmatched.contains(wd))
            .map(|(_, f)| f.as_str())
            .collect();
        let code = if ["count", "sum", "avg", "agg"]
            .iter()
            .any(|f| fams.contains(f))
        {
            "D2C_AGGREGATE"
        } else if fams.contains("group") {
            "D2C_GROUP"
        } else if fams.len() == 1 && fams.contains("arith") {
            "D2C_ARITHMETIC"
        } else {
            "D2C_EXTREMUM"
        };
        out.push(d(code, "c", unique_join(&unmatched, ", ")));
    }
    let cc = count_comparison_cues(&q, b)?;
    if !cc.is_empty() && scalar {
        out.push(d("D2C_COUNT_COMPARISON", "c", cc.join("; ")));
    }
    let comp = comparative_cues(&q);
    if !comp.is_empty() && scalar {
        if !(any_op(&COMPARE_OPS) || any_op(&PICK_OPS)) {
            out.push(d("D2C_COMPARATIVE_NO_COMPARISON", "c", comp.join(", ")));
        } else if !any_op(&ORDER_OPS) && !q.low.iter().any(|x| has(w::EQUALITY_CUES, x)) {
            out.push(d("D2C_COMPARATIVE_EQ_ONLY", "c", comp.join(", ")));
        }
    }
    // (d)
    if q.has_action && !(q.has_polar || q.has_wh || q.has_request || q.has_question_mark) {
        out.push(d(
            "D2D_ACTION_REQUEST",
            "d",
            "the text asks for an action, not an answer".to_string(),
        ));
    }
    if p.yes_no && !q.has_polar && (q.has_wh || q.has_request) {
        out.push(d(
            "D2D_YES_NO_FOR_WH",
            "d",
            format!(
                "yes / no answer to a {}",
                if q.has_wh { "wh-question" } else { "request" }
            ),
        ));
    }
    if !p.yes_no && q.has_polar && !q.has_wh && !q.has_request && !q.choice && p.answer.is_some() {
        out.push(d(
            "D2D_VALUE_FOR_YES_NO",
            "d",
            "a value answer to a yes / no question".to_string(),
        ));
    }
    let wants_id = requests_id(&q, b)?;
    let fop = p.final_op.as_deref();
    let lookup_final = matches!(fop, Some("LOOKUP" | "CHAIN"));
    let mut id_rel: Option<String> = None;
    if lookup_final && is_id_relation(p.returned_relation.as_deref(), b)? {
        id_rel = p.returned_relation.clone();
    } else if fop.is_some_and(|o| COMPARE_OPS.contains(&o) || PICK_OPS.contains(&o) || o == "NOT") {
        for r in &p.operand_relations {
            if is_id_relation(Some(r), b)? {
                id_rel = Some(r.clone());
                break;
            }
        }
    }
    if let Some(r) = id_rel.as_deref().filter(|r| !r.is_empty()) {
        if !wants_id {
            out.push(d(
                "D2D_ID_NOT_REQUESTED",
                "d",
                format!("identifier field '{r}' not asked for"),
            ));
        }
    }
    if wants_id && !p.yes_no && lookup_final {
        let value_or_answer = p.value.as_deref().or(p.answer.as_deref());
        if matches!(p.value_type.as_deref(), Some("date" | "boolean"))
            || (p
                .returned_relation
                .as_deref()
                .is_some_and(|r| !r.is_empty())
                && !is_id_relation(p.returned_relation.as_deref(), b)?
                && !is_code_like(value_or_answer, b)?)
        {
            out.push(d(
                "D2D_ID_REQUESTED_NOT_ID",
                "d",
                format!(
                    "identifier requested, got {} from '{}'",
                    opt(&p.value_type),
                    opt(&p.returned_relation)
                ),
            ));
        }
    }
    let pf = requested_place_field(&q);
    if let (Some(pf), true, Some(rel), false) = (
        pf.as_ref(),
        lookup_final,
        p.returned_relation.as_deref().filter(|r| !r.is_empty()),
        p.yes_no,
    ) {
        let l = lower(rel);
        let last = split_field(&l).pop().unwrap_or("");
        if has(w::PLACE_FIELDS, last) && lookup(w::PLACE_LEVEL, last) != lookup(w::PLACE_LEVEL, pf)
        {
            out.push(d(
                "D2D_PLACE_FIELD_MISMATCH",
                "d",
                format!("{pf} requested, field '{rel}' returned"),
            ));
        }
    }
    let rt = requested_types(&q);
    if !p.yes_no
        && matches!(fop, Some("LOOKUP" | "CHAIN" | "PICK_LT" | "PICK_GT"))
        && p.value_type.as_deref().is_some_and(|v| !v.is_empty())
    {
        let v = p.value.as_deref().or(p.answer.as_deref());
        let vt = p.value_type.as_deref().unwrap_or("");
        if rt.contains("date") && !(vt == "date" || year_or_date_like(v, b)?) {
            out.push(d("D2D_TYPE_DATE", "d", format!("date requested, got {vt}")));
        }
        if rt.contains("number")
            && vt != "number"
            && RX
                .leading_num
                .match_at_start(&chars(v.unwrap_or("")), b)?
                .is_none()
        {
            out.push(d(
                "D2D_TYPE_NUMBER",
                "d",
                format!("amount requested, got {vt}"),
            ));
        }
        if rt.contains("person")
            && (matches!(vt, "number" | "date" | "boolean")
                || is_id_relation(p.returned_relation.as_deref(), b)?)
        {
            out.push(d(
                "D2D_TYPE_PERSON",
                "d",
                format!("person requested, got {vt}"),
            ));
        }
        if rt.contains("place") && matches!(vt, "number" | "date" | "boolean") {
            out.push(d(
                "D2D_TYPE_PLACE",
                "d",
                format!("place requested, got {vt}"),
            ));
        }
    }
    Ok(out)
}

/// the D2 decision of one answer: the first failing check; `row.status` other than ANSWER is not checked
pub fn check(question: &str, row: &Row, b: &Budget) -> Option<Decline> {
    if row.status != "ANSWER" {
        return None;
    }
    match check_all(question, row, b) {
        Ok(f) => f.into_iter().next(),
        Err(OverBudget) => Some(Decline {
            code: "D2X_RULE_ERROR",
            rule: "x",
            detail: "OverBudget: the question exceeds the rule step budget".to_string(),
        }),
    }
}

/// `str.split()` as `_value_kind` uses it, shared with roles
pub fn second_word(s: &str) -> Option<String> {
    split_ws(s).get(1).map(|x| x.to_string())
}
