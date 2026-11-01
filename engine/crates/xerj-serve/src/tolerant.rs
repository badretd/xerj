//! Tolerant entity linking with a visible rewrite (EXTENSION, off by default; the exact linker stays the reference).
//!
//! The exact linker (`xr_codecs.link_entities`) finds a subject only when the question contains it verbatim, so
//! "velquel tower" or "Velqel Tower" link nothing, the manifest falls back to unrelated records and the reader answers
//! wrong or abstains. Here, after the exact pass, the remaining question words are compared with the snapshot's subjects
//! on their `normalize_entity` form with spaces removed (case, punctuation and spacing differences cost nothing) using a
//! bounded Damerau-Levenshtein distance:
//!
//! | subject length (normalized, no spaces) | edits allowed |
//! |---|---|
//! | < 5 | 0 (case / punctuation / spacing only) |
//! | 5-8 | 1 |
//! | 9-15 | 2 |
//! | >= 16 | 3 |
//!
//! Windows of n-1..n+1 question words are tried for a subject of n words. Ties between two different subjects at the best
//! distance are AMBIGUOUS and link nothing (no guessing between "Ivana Trump" and "Ivanka Trump"). Accepted matches are
//! written back into the question as the canonical subject string, so the exact manifest, the model and the verify rule
//! all see the stored name; the rewrite is returned for display. Cost: O(#active subjects x #question words) per question,
//! fine for per-case and playground stores; a large store needs a trigram / SymSpell candidate index first (not built).

use std::collections::HashSet;
use xerj_index::{Memory, QBuf};
use xerj_store::text::normalize_entity;
use xerj_store::{RecId, Snapshot};

#[derive(Debug, Clone)]
pub struct Match {
    /// byte span in the original question
    pub start: usize,
    pub end: usize,
    pub surface: String,
    pub subject: String,
    pub distance: usize,
}

#[derive(Debug, Clone)]
pub struct LinkInfo {
    /// "exact" (nothing changed), "case" (only case / punctuation / spacing), "fuzzy" (typos corrected)
    pub mode: &'static str,
    pub exact_subjects: Vec<String>,
    pub matches: Vec<Match>,
    pub ambiguous: Vec<(String, Vec<String>)>,
    pub rewritten: String,
    /// relation-word typo corrections (surface, replacement, distance); filled by `correct_relation_words`
    pub word_fixes: Vec<(String, String, usize)>,
    /// the first letter was lower case and was capitalized (sentence case)
    pub capitalized: bool,
}

pub fn allowed_edits(len: usize) -> usize {
    match len {
        0..=4 => 0,
        5..=8 => 1,
        9..=15 => 2,
        _ => 3,
    }
}

/// optimal-string-alignment distance with an upper bound (returns bound + 1 when exceeded)
pub fn osa_bounded(a: &[char], b: &[char], bound: usize) -> usize {
    let (n, m) = (a.len(), b.len());
    if n.abs_diff(m) > bound {
        return bound + 1;
    }
    let mut prev2 = vec![0usize; m + 1];
    let mut prev: Vec<usize> = (0..=m).collect();
    let mut cur = vec![0usize; m + 1];
    for i in 1..=n {
        cur[0] = i;
        let mut row_min = cur[0];
        for j in 1..=m {
            let cost = usize::from(a[i - 1] != b[j - 1]);
            let mut v = (prev[j] + 1).min(cur[j - 1] + 1).min(prev[j - 1] + cost);
            if i > 1 && j > 1 && a[i - 1] == b[j - 2] && a[i - 2] == b[j - 1] {
                v = v.min(prev2[j - 2] + 1);
            }
            cur[j] = v;
            row_min = row_min.min(v);
        }
        if row_min > bound {
            return bound + 1;
        }
        std::mem::swap(&mut prev2, &mut prev);
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[m].min(bound + 1)
}

fn squash(s: &str) -> Vec<char> {
    normalize_entity(s).chars().filter(|c| *c != ' ').collect()
}

/// word spans (byte offsets) of the question: maximal runs of alphanumeric chars
fn words(q: &str) -> Vec<(usize, usize)> {
    let mut v = Vec::new();
    let mut st: Option<usize> = None;
    for (i, c) in q.char_indices() {
        if c.is_alphanumeric() {
            if st.is_none() {
                st = Some(i);
            }
        } else if let Some(s) = st.take() {
            v.push((s, i));
        }
    }
    if let Some(s) = st {
        v.push((s, q.len()));
    }
    v
}

/// spans of the exactly linked subjects (first occurrence at word boundaries)
fn exact_spans(q: &str, subjects: &[String]) -> Vec<(usize, usize)> {
    let mut out: Vec<(usize, usize)> = Vec::new();
    for s in subjects {
        let mut from = 0;
        while let Some(p) = q[from..].find(s.as_str()) {
            let a = from + p;
            let b = a + s.len();
            let left_ok = q[..a]
                .chars()
                .next_back()
                .is_none_or(|c| !c.is_alphanumeric());
            let right_ok = q[b..].chars().next().is_none_or(|c| !c.is_alphanumeric());
            if left_ok && right_ok && !out.iter().any(|&(x, y)| a < y && x < b) {
                out.push((a, b));
                break;
            }
            from = a + s.chars().next().map_or(1, |c| c.len_utf8());
        }
    }
    out
}

/// distinct subject strings of the records visible in `snap`
pub fn active_subjects(mem: &Memory, snap: &Snapshot<'_>) -> Vec<String> {
    let store = &mem.store;
    let mut seen: HashSet<&str> = HashSet::new();
    let mut out = Vec::new();
    for r in 0..store.num_records() as RecId {
        if snap.visible(r) {
            let s = store.subject(r);
            if seen.insert(s) {
                out.push(s.to_string());
            }
        }
    }
    out
}

pub fn tolerant_link(mem: &Memory, snap: &Snapshot<'_>, q: &str, buf: &mut QBuf) -> LinkInfo {
    let exact: Vec<String> = mem
        .link(snap, q, buf)
        .iter()
        .map(|&s| mem.store.pools.subjects.get(s).to_string())
        .collect();
    let protected = exact_spans(q, &exact);
    let ws = words(q);
    let free: Vec<bool> = ws
        .iter()
        .map(|&(a, b)| !protected.iter().any(|&(x, y)| a < y && x < b))
        .collect();
    let wsq: Vec<Vec<char>> = ws.iter().map(|&(a, b)| squash(&q[a..b])).collect();
    // candidates: (distance, -len, subject index, first word, last word)
    let subjects = active_subjects(mem, snap);
    let mut cands: Vec<(usize, isize, usize, usize, usize)> = Vec::new();
    for (si, s) in subjects.iter().enumerate() {
        if exact.contains(s) {
            continue;
        }
        let ss = squash(s);
        if ss.is_empty() {
            continue;
        }
        let n_words = normalize_entity(s)
            .split(' ')
            .filter(|w| !w.is_empty())
            .count()
            .max(1);
        let bound = allowed_edits(ss.len());
        for w in n_words.saturating_sub(1).max(1)..=n_words + 1 {
            if w > ws.len() {
                continue;
            }
            for i in 0..=ws.len() - w {
                if !(i..i + w).all(|k| free[k]) {
                    continue;
                }
                let mut win: Vec<char> = Vec::new();
                for k in i..i + w {
                    win.extend_from_slice(&wsq[k]);
                }
                if win.len().abs_diff(ss.len()) > bound {
                    continue;
                }
                let d = osa_bounded(&win, &ss, bound);
                if d <= bound {
                    cands.push((d, -(ss.len() as isize), si, i, i + w - 1));
                }
            }
        }
    }
    cands.sort();
    let mut used: Vec<(usize, usize)> = Vec::new();
    let mut matches: Vec<Match> = Vec::new();
    let mut ambiguous: Vec<(String, Vec<String>)> = Vec::new();
    let mut taken_subjects: HashSet<usize> = HashSet::new();
    for idx in 0..cands.len() {
        let (d, _, si, i, j) = cands[idx];
        if taken_subjects.contains(&si) || used.iter().any(|&(x, y)| i <= y && x <= j) {
            continue;
        }
        // another subject equally close on an overlapping window -> ambiguous, link nothing there
        let rivals: Vec<String> = cands
            .iter()
            .filter(|c| {
                c.0 == d && c.2 != si && c.3 <= j && i <= c.4 && !taken_subjects.contains(&c.2)
            })
            .map(|c| subjects[c.2].clone())
            .collect();
        if !rivals.is_empty() {
            let mut all = vec![subjects[si].clone()];
            for r in rivals {
                if !all.contains(&r) {
                    all.push(r);
                }
            }
            ambiguous.push((q[ws[i].0..ws[j].1].to_string(), all));
            used.push((i, j));
            continue;
        }
        used.push((i, j));
        taken_subjects.insert(si);
        matches.push(Match {
            start: ws[i].0,
            end: ws[j].1,
            surface: q[ws[i].0..ws[j].1].to_string(),
            subject: subjects[si].clone(),
            distance: d,
        });
    }
    matches.sort_by_key(|m| m.start);
    let mut rewritten = String::with_capacity(q.len() + 16);
    let mut last = 0;
    for m in &matches {
        rewritten.push_str(&q[last..m.start]);
        rewritten.push_str(&m.subject);
        last = m.end;
    }
    rewritten.push_str(&q[last..]);
    // sentence case: the models were trained on questions that start with a capital ("which city ..." -> "Which city ...");
    // a question that starts inside a stored name keeps the name as stored
    let mut capitalized = false;
    if let Some((i, c)) = rewritten.char_indices().find(|(_, c)| c.is_alphabetic()) {
        let in_name = exact_spans(&rewritten, &exact)
            .iter()
            .chain(
                exact_spans(
                    &rewritten,
                    &matches
                        .iter()
                        .map(|m| m.subject.clone())
                        .collect::<Vec<_>>(),
                )
                .iter(),
            )
            .any(|&(a, b)| a <= i && i < b);
        if c.is_lowercase() && !in_name {
            let up: String = c.to_uppercase().collect();
            rewritten.replace_range(i..i + c.len_utf8(), &up);
            capitalized = true;
        }
    }
    let mode = if matches.is_empty() && !capitalized {
        "exact"
    } else if matches.iter().all(|m| m.distance == 0) {
        "case"
    } else {
        "fuzzy"
    };
    LinkInfo {
        mode,
        exact_subjects: exact,
        matches,
        ambiguous,
        rewritten,
        word_fixes: Vec::new(),
        capitalized,
    }
}

/// "headquarter" vs "headquarters", "founded" vs "found": the same word with an inflection ending, not a typo
fn inflection_of(a: &str, b: &str) -> bool {
    let (short, long) = if a.len() <= b.len() { (a, b) } else { (b, a) };
    long.starts_with(short)
        && matches!(
            &long[short.len()..],
            "s" | "es" | "d" | "ed" | "ing" | "er" | "ers" | "ly" | "al"
        )
}

/// A typo in a relation word ("hieght", "ocupation"): question words of >= 5 letters outside the linked names that are not a
/// whole-word token of the model's tokenizer (`is_known`: a misspelling is split into pieces, a real English word mostly is
/// not) and that are within the edit budget of exactly one relation-phrase word of the snapshot are rewritten to that word.
/// Known words are never touched, so "height" is not turned into "weight" when only weights are stored. Returns the new
/// question and the (surface, replacement, distance) list.
pub fn correct_relation_words(
    mem: &Memory,
    snap: &Snapshot<'_>,
    q: &str,
    protect: &[String],
    is_known: &dyn Fn(&str) -> bool,
) -> (String, Vec<(String, String, usize)>) {
    let store = &mem.store;
    let mut vocab: Vec<String> = Vec::new();
    let mut seen_rel: HashSet<&str> = HashSet::new();
    for r in 0..store.num_records() as RecId {
        if !snap.visible(r) {
            continue;
        }
        let rel = store.relation(r);
        if !seen_rel.insert(rel) {
            continue;
        }
        for w in xerj_index::rel_phrase(rel).split(|c: char| !c.is_alphanumeric()) {
            let w = w.to_lowercase();
            if w.chars().count() >= 4 && !vocab.contains(&w) {
                vocab.push(w);
            }
        }
    }
    let spans = exact_spans(q, protect);
    let mut out = String::with_capacity(q.len());
    let mut fixes = Vec::new();
    let mut last = 0;
    for (a, b) in words(q) {
        let w = &q[a..b];
        let n = w.chars().count();
        if n < 5
            || !w.chars().all(|c| c.is_alphabetic())
            || spans.iter().any(|&(x, y)| a < y && x < b)
        {
            continue;
        }
        let lw = w.to_lowercase();
        if vocab.contains(&lw) || is_known(&lw) || vocab.iter().any(|v| inflection_of(&lw, v)) {
            continue;
        }
        let wc: Vec<char> = lw.chars().collect();
        let bound = allowed_edits(n).min(2);
        let mut best: Vec<(usize, &String)> = Vec::new();
        for v in &vocab {
            let vc: Vec<char> = v.chars().collect();
            let d = osa_bounded(&wc, &vc, bound);
            if d <= bound {
                best.push((d, v));
            }
        }
        let Some(dmin) = best.iter().map(|x| x.0).min() else {
            continue;
        };
        let top: Vec<&String> = best.iter().filter(|x| x.0 == dmin).map(|x| x.1).collect();
        if top.len() != 1 || dmin == 0 {
            continue;
        }
        let mut rep = top[0].clone();
        if w.chars().next().is_some_and(|c| c.is_uppercase()) {
            let mut cs = rep.chars();
            rep = cs
                .next()
                .map(|c| c.to_uppercase().collect::<String>() + cs.as_str())
                .unwrap_or_default();
        }
        out.push_str(&q[last..a]);
        out.push_str(&rep);
        last = b;
        fixes.push((w.to_string(), rep, dmin));
    }
    out.push_str(&q[last..]);
    (out, fixes)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d(a: &str, b: &str, k: usize) -> usize {
        let a: Vec<char> = a.chars().collect();
        let b: Vec<char> = b.chars().collect();
        osa_bounded(&a, &b, k)
    }

    #[test]
    fn distance() {
        assert_eq!(d("velqueltower", "velqueltower", 2), 0);
        assert_eq!(d("velqeltower", "velqueltower", 2), 1);
        assert_eq!(d("vleqqueltower", "velqueltower", 2), 2);
        assert_eq!(d("abc", "xyz", 1), 2);
        assert_eq!(d("ivanktarump", "ivankatrump", 2), 1);
        assert!(inflection_of("headquarter", "headquarters"));
        assert!(!inflection_of("hieght", "height"));
    }
}
