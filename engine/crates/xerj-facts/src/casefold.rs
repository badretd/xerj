//! Opt-in case-insensitive subject linking for large shared stores (`_ask` with `"linker": "casefold"`; off by default:
//! the exact linker stays the reference and the default).
//!
//! Why (E8, one shared store): the exact linker finds a subject only when the question contains its stored name verbatim.
//! In a per-world store that rarely matters, because a world of at most K = 16 records puts every record in the candidate
//! list anyway; in a shared store the unlinked question gets 16 unrelated records. 2WikiMultihopQA questions are title
//! cased ("Did the movies Cock Of The Roost and Nobody'S Perfect originate from the same country?") while the stored
//! subjects are not ("Cock of the Roost", "NoBody's Perfect"): at 10 thousand shared facts the gold trace then finds its
//! records for 377 of 791 questions instead of 791.
//!
//! What: before the exact pipeline runs, every name-like span of the question whose `normalize_entity` form (NFKC,
//! casefold, punctuation as spaces, whitespace collapsed) equals that of exactly ONE subject active in the snapshot is
//! written back as the stored subject string, so linking, the manifest, the compiler and the verify rule all see the
//! stored name, and the rewrite is returned. Lookup goes through the store's lazily built normalized-name index
//! (`SubjectIndex::subjects_with_norm`, also used by stack-v2 hop resolution): O(question words x window) hash lookups per
//! question, whatever the store size.
//!
//! Rules that keep it from linking ordinary words in a large store: a span must contain an upper-case letter or a digit;
//! a one-word span must start with an upper-case letter and must not be the question's first word (sentence case is not
//! evidence of a name); spans are taken longest first, left to right, without overlap; a span already equal to its
//! subject is left as it is.
//!
//! Only letter case is bridged (second version, measured at 1 million shared facts): a candidate from the normalized
//! index is accepted only if the question span, widened over the subject's own leading / trailing punctuation, equals
//! the stored name up to letter case. The normalized form also drops punctuation, which linked the word "TV" to a
//! background subject ".TV". When several active subjects pass (the same name stored in two spellings, e.g. a test
//! world's "When Do You Commit Suicide? (1932 film)" and a background alias "when do you commit suicide? (1932 film)"),
//! the one whose letter case is closest to the question's wins; a tie is ambiguous and is not rewritten. A span with an
//! upper-case letter is never rewritten to an all-lower-case stored name: at 10 million facts the background's
//! lower-cased aliases ("india?", "emmanuel") turned the value mentions "India" and "Emmanuel Macron" into links to
//! unrelated subjects.

//!
//! Third version (`"linker": "casefold_v3"`, [`casefold_v3_link`], round 2 of E8; opt-in, `casefold` keeps the v2 rules
//! above): the v2 rewrite plus an inflection fallback (a name-like span that names no subject in any letter case, and whose
//! last word looks plural, is tried in the singular: "Big Macs" -> the stored "Big Mac"; hash lookups in the same
//! normalized-name index, so the cost does not grow with the store), and then, on the exact linker's subjects of the
//! rewritten question: a subject whose whole name is an English function word is not linked when the question uses it
//! only lower case or as its first word (background subjects "and", "Whose" at 10 million facts), and every active subject
//! spelled the same up to letter case is linked with it (the same entity stored in two spellings by two sources, "Pat
//! DeWine" / "Pat Dewine"), except an all-lower-case spelling of a capitalized name (a background alias). The linked
//! subjects go to the manifest as given ([`xerj_serve::engine::AnswerOpts::ents`]).

use smallvec::SmallVec;
use xerj_index::{Memory, QBuf};
use xerj_store::text::normalize_entity;
use xerj_store::Snapshot;

/// Longest subject name considered, in question words.
pub const MAX_WORDS: usize = 12;

/// One accepted rewrite (byte span of the original question).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CaseMatch {
    pub start: usize,
    pub end: usize,
    pub surface: String,
    pub subject: String,
}

#[derive(Clone, Debug, Default)]
pub struct CaseLink {
    /// the question with the accepted spans replaced by the stored subject strings
    pub rewritten: String,
    pub matches: Vec<CaseMatch>,
    /// name-like spans whose normalized form names several active subjects (not rewritten)
    pub ambiguous: Vec<(String, Vec<String>)>,
}

/// word spans (byte offsets): maximal runs of alphanumeric chars
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

fn name_like(surface: &str, one_word: bool, first_word: bool) -> bool {
    if one_word {
        return !first_word && surface.chars().next().is_some_and(char::is_uppercase);
    }
    surface.chars().any(|c| c.is_uppercase() || c.is_numeric())
}

/// Widen `[a, b)` over the punctuation the subject itself starts / ends with ("(film)" after "Target Nevada").
fn widen(q: &str, a: usize, b: usize, subject: &str) -> (usize, usize) {
    let lead: String = subject
        .chars()
        .take_while(|c| !c.is_alphanumeric())
        .collect();
    let trail: String = {
        let mut t: Vec<char> = subject
            .chars()
            .rev()
            .take_while(|c| !c.is_alphanumeric())
            .collect();
        t.reverse();
        t.into_iter().collect()
    };
    let a2 = if !lead.is_empty() && q[..a].ends_with(lead.as_str()) {
        a - lead.len()
    } else {
        a
    };
    let b2 = if !trail.is_empty() && q[b..].starts_with(trail.as_str()) {
        b + trail.len()
    } else {
        b
    };
    (a2, b2)
}

/// `a` and `b` are the same string up to letter case (char by char, Unicode lower case).
fn same_but_case(a: &str, b: &str) -> bool {
    let mut x = a.chars().flat_map(char::to_lowercase);
    let mut y = b.chars().flat_map(char::to_lowercase);
    loop {
        match (x.next(), y.next()) {
            (None, None) => return true,
            (Some(p), Some(q)) if p == q => {}
            _ => return false,
        }
    }
}

/// Number of characters whose letter case differs (for strings that are [`same_but_case`]).
fn case_distance(a: &str, b: &str) -> usize {
    a.chars().zip(b.chars()).filter(|(p, q)| p != q).count()
}

/// The case-insensitive rewrite of `question` against the subjects active in `snap`.
pub fn casefold_link(mem: &Memory, snap: &Snapshot<'_>, question: &str) -> CaseLink {
    casefold_link_with(mem, snap, question, false)
}

/// singular forms of a span whose last word looks plural ("Macs" -> "Mac", "Boxes" -> "Box", "Cities" -> "City")
fn singulars(surface: &str) -> SmallVec<[String; 3]> {
    let mut out: SmallVec<[String; 3]> = SmallVec::new();
    let last_start = surface
        .char_indices()
        .rev()
        .find(|(_, c)| !c.is_alphanumeric())
        .map_or(0, |(i, c)| i + c.len_utf8());
    let (head, last) = surface.split_at(last_start);
    if last.chars().count() < 4 || !last.chars().all(char::is_alphabetic) {
        return out;
    }
    let lower = last.to_lowercase();
    if lower.ends_with("ss") || !lower.ends_with('s') {
        return out;
    }
    if lower.ends_with("ies") {
        let stem = &last[..last.len() - 3];
        let y = if last.ends_with("IES") { "Y" } else { "y" };
        out.push(format!("{head}{stem}{y}"));
    }
    if lower.ends_with("es") {
        out.push(format!("{head}{}", &last[..last.len() - 2]));
    }
    out.push(format!("{head}{}", &last[..last.len() - 1]));
    out
}

/// [`casefold_link`] with the v3 inflection fallback when `inflect` (see the module documentation); `inflect` = false is
/// exactly [`casefold_link`].
pub fn casefold_link_with(
    mem: &Memory,
    snap: &Snapshot<'_>,
    question: &str,
    inflect: bool,
) -> CaseLink {
    let ws = words(question);
    let mut out = CaseLink::default();
    let mut found: SmallVec<[u32; 4]> = SmallVec::new();
    let mut i = 0usize;
    while i < ws.len() {
        let mut took = 0usize;
        for n in (1..=MAX_WORDS.min(ws.len() - i)).rev() {
            let (a, b) = (ws[i].0, ws[i + n - 1].1);
            let surface = &question[a..b];
            if !name_like(surface, n == 1, i == 0) {
                continue;
            }
            let norm = normalize_entity(surface);
            if norm.is_empty() {
                continue;
            }
            found.clear();
            mem.subjects
                .subjects_with_norm(&mem.store, &norm, &mut found);
            let mut names: Vec<&str> = Vec::new();
            for &s in &found {
                if mem.subjects.subject_active(snap, s) {
                    let name = mem.store.pools.subjects.get(s);
                    if !names.contains(&name) {
                        names.push(name);
                    }
                }
            }
            // only letter case is bridged: the widened span must equal the stored name up to case
            let prev_end = out.matches.last().map_or(0, |m| m.end);
            let span_upper = surface.chars().any(char::is_uppercase);
            let mut cands: Vec<(usize, &str, usize, usize)> = Vec::new();
            for &name in &names {
                if span_upper && !name.chars().any(char::is_uppercase) {
                    continue;
                }
                let (a2, b2) = widen(question, a, b, name);
                // never reach back into the previous rewrite
                let a2 = a2.max(prev_end);
                let span = &question[a2..b2];
                if same_but_case(span, name) {
                    cands.push((case_distance(span, name), name, a2, b2));
                }
            }
            if cands.is_empty() && inflect && n >= 1 && !(n == 1 && i == 0) {
                // v3: a plural-looking span that names nothing is tried in the singular (letter case bridged as above)
                let mut alt: Vec<(usize, String)> = Vec::new();
                for sv in singulars(surface) {
                    let nn = normalize_entity(&sv);
                    if nn.is_empty() {
                        continue;
                    }
                    found.clear();
                    mem.subjects.subjects_with_norm(&mem.store, &nn, &mut found);
                    for &f in &found {
                        if !mem.subjects.subject_active(snap, f) {
                            continue;
                        }
                        let name = mem.store.pools.subjects.get(f);
                        if span_upper && !name.chars().any(char::is_uppercase) {
                            continue;
                        }
                        if same_but_case(&sv, name) && !alt.iter().any(|x| x.1 == name) {
                            alt.push((case_distance(&sv, name), name.to_string()));
                        }
                    }
                    if !alt.is_empty() {
                        break;
                    }
                }
                if !alt.is_empty() {
                    alt.sort();
                    if alt.len() > 1 && alt[0].0 == alt[1].0 {
                        out.ambiguous.push((
                            surface.to_string(),
                            alt.iter().map(|c| c.1.clone()).collect(),
                        ));
                    } else if a >= prev_end {
                        out.matches.push(CaseMatch {
                            start: a,
                            end: b,
                            surface: surface.to_string(),
                            subject: alt[0].1.clone(),
                        });
                    }
                    took = n;
                    break;
                }
            }
            if cands.is_empty() {
                continue;
            }
            cands.sort();
            if cands.len() > 1 && cands[0].0 == cands[1].0 {
                out.ambiguous.push((
                    surface.to_string(),
                    cands.iter().map(|c| c.1.to_string()).collect(),
                ));
                took = n;
                break;
            }
            let (dist, name, a2, b2) = cands[0];
            if dist > 0 {
                out.matches.push(CaseMatch {
                    start: a2,
                    end: b2,
                    surface: question[a2..b2].to_string(),
                    subject: name.to_string(),
                });
            }
            took = n;
            break;
        }
        i += took.max(1);
    }
    let mut s = String::with_capacity(question.len() + 16);
    let mut at = 0usize;
    for m in &out.matches {
        s.push_str(&question[at..m.start]);
        s.push_str(&m.subject);
        at = m.end;
    }
    s.push_str(&question[at..]);
    out.rewritten = s;
    out
}

/// Closed-class English words (question words, articles, prepositions, conjunctions, auxiliaries, pronouns).
pub const FUNCTION_WORDS: &[&str] = &[
    "a", "about", "after", "all", "also", "an", "and", "any", "are", "as", "at", "be", "been",
    "before", "being", "both", "but", "by", "can", "could", "did", "do", "does", "each", "either",
    "for", "from", "had", "has", "have", "he", "her", "here", "hers", "him", "his", "how", "i",
    "if", "in", "into", "is", "it", "its", "me", "my", "neither", "no", "nor", "not", "of", "on",
    "or", "our", "she", "should", "so", "some", "than", "that", "the", "their", "them", "then",
    "there", "these", "they", "this", "those", "to", "under", "up", "us", "was", "we", "were",
    "what", "when", "where", "whether", "which", "while", "who", "whom", "whose", "why", "will",
    "with", "would", "you", "your",
];

/// `name` is one English function word, in any letter case
pub fn is_function_word(name: &str) -> bool {
    let l = name.to_lowercase();
    FUNCTION_WORDS.contains(&l.as_str())
}

/// The question uses `name` (verbatim, as whole words) only as its first word or with no upper-case letter at all, i.e.
/// never as a capitalized name inside the sentence.
fn only_sentence_case_or_lower(question: &str, name: &str) -> bool {
    if !name.chars().any(char::is_uppercase) {
        return true;
    }
    let first_word_start = question
        .char_indices()
        .find(|(_, c)| c.is_alphanumeric())
        .map_or(0, |(i, _)| i);
    let mut from = 0usize;
    while let Some(off) = question[from..].find(name) {
        let st = from + off;
        let en = st + name.len();
        let before_ok = question[..st]
            .chars()
            .next_back()
            .is_none_or(|c| !c.is_alphanumeric());
        let after_ok = question[en..]
            .chars()
            .next()
            .is_none_or(|c| !c.is_alphanumeric());
        if before_ok && after_ok && st != first_word_start {
            return false;
        }
        from = en;
    }
    true
}

/// The `casefold_v3` link of one question ([`casefold_v3_link`]).
#[derive(Clone, Debug, Default)]
pub struct SharedLink {
    /// the v2 rewrite with the v3 inflection fallback (`rewritten` is the question the pipeline runs on)
    pub case: CaseLink,
    /// the exact linker's subjects in the rewritten question (subject syms, reference order)
    pub exact: Vec<u32>,
    /// the subjects passed to the manifest: `exact` minus `dropped`, plus `merged`
    pub subjects: Vec<u32>,
    /// function-word subjects not linked
    pub dropped: Vec<String>,
    /// (linked subject, the same name in another letter case, linked too)
    pub merged: Vec<(String, String)>,
}

impl SharedLink {
    /// the manifest sees other subjects than the exact linker's
    pub fn changes_subjects(&self) -> bool {
        self.subjects != self.exact
    }
}

/// The opt-in third version of the shared-store linker (`"linker": "casefold_v3"`; module documentation).
pub fn casefold_v3_link(mem: &Memory, snap: &Snapshot<'_>, question: &str) -> SharedLink {
    let case = casefold_link_with(mem, snap, question, true);
    let mut buf = QBuf::default();
    let exact: Vec<u32> = mem.link(snap, &case.rewritten, &mut buf).to_vec();
    let mut out = SharedLink {
        case,
        exact: exact.clone(),
        ..SharedLink::default()
    };
    let pool = &mem.store.pools.subjects;
    for &s in &exact {
        let name = pool.get(s);
        if is_function_word(name) && only_sentence_case_or_lower(&out.case.rewritten, name) {
            out.dropped.push(name.to_string());
            continue;
        }
        if !out.subjects.contains(&s) {
            out.subjects.push(s);
        }
    }
    let kept = out.subjects.clone();
    let mut found: SmallVec<[u32; 4]> = SmallVec::new();
    for s in kept {
        let name = pool.get(s);
        let norm = normalize_entity(name);
        if norm.is_empty() {
            continue;
        }
        found.clear();
        mem.subjects
            .subjects_with_norm(&mem.store, &norm, &mut found);
        for &c in &found {
            if c == s || out.subjects.contains(&c) || !mem.subjects.subject_active(snap, c) {
                continue;
            }
            let other = pool.get(c);
            // as in the rewrite: a capitalized name is never joined by an all-lower-case spelling (background aliases
            // such as "quaqtaq airport" next to the test world's "Quaqtaq Airport")
            let upper = |x: &str| x.chars().any(char::is_uppercase);
            if same_but_case(other, name) && (upper(other) || !upper(name)) {
                out.subjects.push(c);
                out.merged.push((name.to_string(), other.to_string()));
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use xerj_index::MemoryConfig;
    use xerj_store::{FactIn, ValueIn};

    fn mem_with(subjects: &[&str]) -> Memory {
        let mut m = Memory::new(MemoryConfig {
            bm25_records: false,
            bm25_documents: false,
            ..MemoryConfig::default()
        });
        for (i, s) in subjects.iter().enumerate() {
            let text = format!("{s} is in Ruritania.");
            let doc = format!("d{i}");
            let fid = format!("f{i}");
            m.add_document(&doc, "v1", &text).unwrap();
            let start = s.chars().count() as i64 + 7;
            let f = FactIn {
                fact_id: &fid,
                subject: s,
                relation: "country",
                kind: "text",
                value: ValueIn::Str("Ruritania"),
                unit: "",
                doc_id: &doc,
                version: "v1",
                start,
                end: start + 9,
                quote: "Ruritania",
                scope: "default",
                subject_id: None,
                value_id: None,
                valid_from: None,
                valid_to: None,
                writer_version: "test",
            };
            m.add_fact(&f, &[]).unwrap();
        }
        m
    }

    #[test]
    fn title_cased_names_are_rewritten_to_the_stored_subject() {
        let m = mem_with(&[
            "Cock of the Roost",
            "NoBody's Perfect",
            "Target Nevada (film)",
        ]);
        let snap = m.head();
        let l = casefold_link(
            &m,
            &snap,
            "Did the movies Cock Of The Roost and Nobody'S Perfect, originate from the same country?",
        );
        assert_eq!(
            l.rewritten,
            "Did the movies Cock of the Roost and NoBody's Perfect, originate from the same country?"
        );
        assert_eq!(l.matches.len(), 2);
        let l = casefold_link(
            &m,
            &snap,
            "Are Target Nevada (Film) and Cock of the Roost from the same country?",
        );
        assert_eq!(
            l.rewritten, "Are Target Nevada (film) and Cock of the Roost from the same country?",
            "the closing parenthesis is part of the subject; the exact name is left alone"
        );
        assert_eq!(l.matches.len(), 1);
    }

    #[test]
    fn ordinary_words_and_sentence_case_are_not_linked() {
        let m = mem_with(&["WHO", "Country", "Paris"]);
        let snap = m.head();
        let q = "Who is the head of the country where paris is?";
        let l = casefold_link(&m, &snap, q);
        assert_eq!(l.rewritten, q, "{:?}", l.matches);
        assert!(l.matches.is_empty());
    }

    #[test]
    fn a_name_two_subjects_spell_equally_close_is_ambiguous() {
        let m = mem_with(&["Ivana Trump", "IVANA TRUMP"]);
        let snap = m.head();
        // 5 case differences from either spelling
        let l = casefold_link(&m, &snap, "Where was IVANA trump born?");
        assert_eq!(l.rewritten, "Where was IVANA trump born?");
        assert_eq!(l.ambiguous.len(), 1);
        assert_eq!(l.ambiguous[0].1.len(), 2);
    }

    #[test]
    fn the_spelling_closest_to_the_question_wins() {
        let m = mem_with(&[
            "When Do You Commit Suicide? (1932 film)",
            "when do you commit suicide? (1932 film)",
            "Ivana Trump",
            "IVANA TRUMP",
        ]);
        let snap = m.head();
        let l = casefold_link(
            &m,
            &snap,
            "Did the movies When Do You Commit Suicide? (1932 Film) and Cock of the Roost originate from the same country?",
        );
        assert_eq!(
            l.rewritten,
            "Did the movies When Do You Commit Suicide? (1932 film) and Cock of the Roost originate from the same country?"
        );
        assert!(l.ambiguous.is_empty());
        let l = casefold_link(&m, &snap, "Where was Ivana trump born?");
        assert_eq!(l.rewritten, "Where was Ivana Trump born?");
    }

    #[test]
    fn a_capitalized_mention_is_not_rewritten_to_a_lower_case_alias() {
        let m = mem_with(&[
            "india?",
            "emmanuel",
            "when do you commit suicide? (1932 film)",
        ]);
        let snap = m.head();
        for q in [
            "Whose country of citizenship is India?",
            "What is the capital of the country whose head of state is Emmanuel Macron?",
            "Did the movies When Do You Commit Suicide? (1932 Film) and Cock of the Roost originate from the same country?",
        ] {
            let l = casefold_link(&m, &snap, q);
            assert_eq!(l.rewritten, q, "{:?}", l.matches);
        }
    }

    #[test]
    fn punctuation_is_not_bridged() {
        let m = mem_with(&[".TV", "Playhouse 90", "Target Nevada (film)"]);
        let snap = m.head();
        let q = "Who is the head of government of the country of origin of the TV series \"Playhouse 90\"?";
        let l = casefold_link(&m, &snap, q);
        assert_eq!(l.rewritten, q, "{:?}", l.matches);
        let q = "Is Target Nevada Film a western?";
        let l = casefold_link(&m, &snap, q);
        assert_eq!(l.rewritten, q, "{:?}", l.matches);
        let l = casefold_link(&m, &snap, "Is Target Nevada (Film) a western?");
        assert_eq!(l.rewritten, "Is Target Nevada (film) a western?");
    }

    #[test]
    fn v3_links_plural_mentions_to_the_singular_subject() {
        let m = mem_with(&["Big Mac", "Android Jelly Bean", "Box"]);
        let snap = m.head();
        let l = casefold_v3_link(&m, &snap, "In which country were Big Macs first sold?");
        assert_eq!(
            l.case.rewritten,
            "In which country were Big Mac first sold?"
        );
        assert_eq!(l.subjects.len(), 1);
        let l = casefold_v3_link(&m, &snap, "Who develops Android Jelly Beans?");
        assert_eq!(l.case.rewritten, "Who develops Android Jelly Bean?");
        // v2 leaves plurals alone; one-word plurals at the start are not names
        let l = casefold_link(&m, &snap, "Who develops Android Jelly Beans?");
        assert_eq!(l.rewritten, "Who develops Android Jelly Beans?");
        let l = casefold_v3_link(&m, &snap, "Boxes are made where?");
        assert_eq!(l.case.rewritten, "Boxes are made where?");
    }

    #[test]
    fn v3_drops_function_word_subjects_used_as_words_and_keeps_titles() {
        let m = mem_with(&["Whose", "and", "Her", "Up", "Floyd Gottfredson"]);
        let snap = m.head();
        let l = casefold_v3_link(
            &m,
            &snap,
            "Whose official language is the language of Floyd Gottfredson and his father?",
        );
        let names: Vec<&str> = l
            .subjects
            .iter()
            .map(|&s| m.store.pools.subjects.get(s))
            .collect();
        assert_eq!(names, vec!["Floyd Gottfredson"]);
        let mut dropped = l.dropped.clone();
        dropped.sort();
        assert_eq!(dropped, vec!["Whose".to_string(), "and".to_string()]);
        // film titles that are function words stay linked where the question capitalizes them inside the sentence
        let l = casefold_v3_link(
            &m,
            &snap,
            "Did the movies Up and Her originate from the same country?",
        );
        let mut names: Vec<&str> = l
            .subjects
            .iter()
            .map(|&s| m.store.pools.subjects.get(s))
            .collect();
        names.sort();
        assert_eq!(names, vec!["Her", "Up"]);
        assert_eq!(l.dropped, vec!["and".to_string()]);
    }

    #[test]
    fn v3_links_every_spelling_of_a_name_that_differs_only_in_case() {
        let m = mem_with(&["Pat DeWine", "Pat Dewine", "Mike DeWine"]);
        let snap = m.head();
        let l = casefold_v3_link(&m, &snap, "Who is the father of Pat Dewine?");
        let names: Vec<&str> = l
            .subjects
            .iter()
            .map(|&s| m.store.pools.subjects.get(s))
            .collect();
        assert_eq!(names, vec!["Pat Dewine", "Pat DeWine"]);
        assert!(l.changes_subjects());
        assert_eq!(
            l.merged,
            vec![("Pat Dewine".to_string(), "Pat DeWine".to_string())]
        );
        // a name stored once is linked as the exact linker links it
        let l = casefold_v3_link(&m, &snap, "Who is the father of Mike DeWine?");
        assert!(!l.changes_subjects());
        // an all-lower-case alias does not join a capitalized name (it does join a lower-case one)
        let m = mem_with(&["Quaqtaq Airport", "quaqtaq airport", "quaqtaq"]);
        let snap = m.head();
        let l = casefold_v3_link(&m, &snap, "Is Quaqtaq Airport in Canada?");
        assert!(!l.changes_subjects(), "{:?}", l.merged);
    }

    #[test]
    fn only_subjects_active_in_the_snapshot_count() {
        let m = mem_with(&["Cock of the Roost"]);
        let before = m.snapshot_at(0);
        let l = casefold_link(&m, &before, "Is Cock Of The Roost a film?");
        assert_eq!(l.rewritten, "Is Cock Of The Roost a film?");
        let l = casefold_link(&m, &m.head(), "Is Cock Of The Roost a film?");
        assert_eq!(l.rewritten, "Is Cock of the Roost a film?");
    }
}
