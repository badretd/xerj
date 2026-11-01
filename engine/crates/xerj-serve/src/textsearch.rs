//! "Where in my text is the answer?": the pasted text is split into sentences, each sentence is a document of an xerj-index
//! Memory (native BM25, the same index as the 10 GB test), and a question returns the best sentences. `fuzzy` adds typo
//! tolerance on the query side: a query word that occurs in no sentence is expanded to the closest vocabulary word(s)
//! within the tolerant linker's edit budget (case is already ignored by the analyzer). BM25 has no notion of synonyms:
//! a paraphrase that shares no word with its sentence cannot be found, by design.

use crate::tolerant::{allowed_edits, osa_bounded};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::time::Instant;
use xerj_index::{Memory, MemoryConfig};

pub struct TextIndex {
    pub mem: Memory,
    pub sentences: Vec<(usize, usize)>,
    /// analyzer token -> number of sentences containing it
    pub vocab: HashMap<String, u32>,
    pub build_us: f64,
}

/// sentence spans (byte offsets): split after . ! ? (plus closing quotes / brackets) followed by whitespace, and at blank lines
pub fn split_sentences(text: &str) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let b: Vec<(usize, char)> = text.char_indices().collect();
    let mut start = 0usize;
    let mut i = 0usize;
    while i < b.len() {
        let (pos, c) = b[i];
        let mut cut: Option<usize> = None;
        if c == '\n' && i + 1 < b.len() && b[i + 1].1 == '\n' {
            cut = Some(pos);
        } else if matches!(c, '.' | '!' | '?') {
            let mut j = i + 1;
            while j < b.len() && matches!(b[j].1, '"' | '\'' | ')' | ']' | '”' | '’') {
                j += 1;
            }
            if (j >= b.len() || b[j].1.is_whitespace()) && !(c == '.' && no_break_after(&b, i)) {
                cut = Some(if j < b.len() { b[j].0 } else { text.len() });
            }
        }
        if let Some(end) = cut {
            let seg = &text[start..end];
            if !seg.trim().is_empty() {
                let lead = seg.len() - seg.trim_start().len();
                let trail = seg.len() - seg.trim_end().len();
                out.push((start + lead, end - trail));
            }
            start = end;
        }
        i += 1;
    }
    if start < text.len() && !text[start..].trim().is_empty() {
        let seg = &text[start..];
        let lead = seg.len() - seg.trim_start().len();
        let trail = seg.len() - seg.trim_end().len();
        out.push((start + lead, text.len() - trail));
    }
    out
}

/// a period that does not end a sentence: after an initial ("Richard G. Wilkinson", "U.S."), after a common abbreviation
/// ("Dr.", "e.g."), or when the next word starts in lower case or with a digit ("approx. 5")
fn no_break_after(b: &[(usize, char)], i: usize) -> bool {
    const ABBR: [&str; 30] = [
        "mr", "mrs", "ms", "dr", "prof", "st", "sr", "jr", "vs", "etc", "eg", "ie", "no", "fig",
        "al", "inc", "ltd", "co", "corp", "gen", "gov", "sen", "rep", "mt", "ft", "jan", "feb",
        "aug", "sept", "approx",
    ];
    let mut k = i;
    let mut word = String::new();
    while k > 0 {
        let ch = b[k - 1].1;
        if ch.is_alphabetic() || ch == '.' {
            word.insert(0, ch);
            k -= 1;
        } else {
            break;
        }
    }
    let letters: String = word
        .chars()
        .filter(|c| c.is_alphabetic())
        .collect::<String>()
        .to_lowercase();
    if letters.chars().count() == 1
        || ABBR.contains(&letters.as_str())
        || (word.contains('.') && letters.chars().count() <= 3)
    {
        return true;
    }
    let mut j = i + 1;
    while j < b.len() && b[j].1.is_whitespace() {
        j += 1;
    }
    j < b.len() && (b[j].1.is_lowercase() || b[j].1.is_ascii_digit())
}

fn analyze(text: &str) -> Vec<String> {
    let mut v = Vec::new();
    let mut scratch = String::new();
    xerj_index::bm25::tokens(text, &mut scratch, |t| v.push(t.to_string()));
    v
}

impl TextIndex {
    pub fn build(text: &str) -> TextIndex {
        let t0 = Instant::now();
        let cfg = MemoryConfig {
            bm25_records: false,
            bm25_documents: true,
            ..MemoryConfig::default()
        };
        let mut mem = Memory::new(cfg);
        let sentences = split_sentences(text);
        let mut vocab: HashMap<String, u32> = HashMap::new();
        for (i, &(a, b)) in sentences.iter().enumerate() {
            let s = &text[a..b];
            mem.add_document(&format!("s{i}"), "v1", s)
                .expect("fresh document ids");
            let mut seen: Vec<String> = analyze(s);
            seen.sort();
            seen.dedup();
            for t in seen {
                *vocab.entry(t).or_insert(0) += 1;
            }
        }
        TextIndex {
            mem,
            sentences,
            vocab,
            build_us: t0.elapsed().as_secs_f64() * 1e6,
        }
    }

    /// expansions for query words absent from the text: (word, [(vocabulary word, distance)])
    pub fn expansions(&self, query: &str) -> Vec<(String, Vec<(String, usize)>)> {
        let mut out = Vec::new();
        let mut seen: Vec<String> = Vec::new();
        for w in analyze(query) {
            if seen.contains(&w) || self.vocab.contains_key(&w) {
                continue;
            }
            seen.push(w.clone());
            let wc: Vec<char> = w.chars().collect();
            let bound = allowed_edits(wc.len());
            if bound == 0 {
                continue;
            }
            let mut best: Vec<(usize, u32, String)> = Vec::new();
            for (v, &df) in &self.vocab {
                let vc: Vec<char> = v.chars().collect();
                if vc.len().abs_diff(wc.len()) > bound {
                    continue;
                }
                let d = osa_bounded(&wc, &vc, bound);
                if d <= bound {
                    best.push((d, df, v.clone()));
                }
            }
            if best.is_empty() {
                continue;
            }
            let dmin = best.iter().map(|x| x.0).min().unwrap();
            let mut keep: Vec<(usize, u32, String)> =
                best.into_iter().filter(|x| x.0 == dmin).collect();
            keep.sort_by(|a, b| b.1.cmp(&a.1).then(a.2.cmp(&b.2)));
            keep.truncate(2);
            out.push((w, keep.into_iter().map(|(d, _, v)| (v, d)).collect()));
        }
        out
    }

    pub fn search(&self, text: &str, query: &str, k: usize, fuzzy: bool) -> Value {
        let t0 = Instant::now();
        let exp = if fuzzy {
            self.expansions(query)
        } else {
            Vec::new()
        };
        let mut q = query.to_string();
        for (_, alts) in &exp {
            for (v, _) in alts {
                q.push(' ');
                q.push_str(v);
            }
        }
        let hits = self.mem.search_documents(&q, k);
        let us = t0.elapsed().as_secs_f64() * 1e6;
        let rows: Vec<Value> = hits
            .iter()
            .filter_map(|h| {
                let (doc_id, _) = self.mem.store.doc_key_str(h.doc);
                let i: usize = doc_id.trim_start_matches('s').parse().ok()?;
                let (a, b) = self.sentences[i];
                Some(json!({"sentence": i, "start": a, "end": b, "text": &text[a..b], "score": (h.score * 1000.0).round() / 1000.0}))
            })
            .collect();
        json!({"query": query, "query_used": q, "fuzzy": fuzzy, "hits": rows,
               "expansions": exp.iter().map(|(w, a)| json!({"word": w, "to": a.iter().map(|(v, d)| json!([v, d])).collect::<Vec<_>>()})).collect::<Vec<_>>(),
               "search_us": (us * 10.0).round() / 10.0})
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sentences() {
        let t = "Anna was born in 1990. She works as a nurse!  Where? In Oslo.\n\nNew para";
        let s: Vec<&str> = split_sentences(t).iter().map(|&(a, b)| &t[a..b]).collect();
        assert_eq!(
            s,
            vec![
                "Anna was born in 1990.",
                "She works as a nurse!",
                "Where?",
                "In Oslo.",
                "New para"
            ]
        );
    }

    #[test]
    fn initials_and_abbreviations() {
        let t = "British researchers Richard G. Wilkinson and Kate Pickett found it. Dr. Smith agreed, e.g. in 2009. The U.S. data came last.";
        let s: Vec<&str> = split_sentences(t).iter().map(|&(a, b)| &t[a..b]).collect();
        assert_eq!(
            s,
            vec![
                "British researchers Richard G. Wilkinson and Kate Pickett found it.",
                "Dr. Smith agreed, e.g. in 2009.",
                "The U.S. data came last."
            ]
        );
    }

    #[test]
    fn fuzzy_finds_typo() {
        let t = "The tower is 312 metres tall. It was completed in 1931. It stands in Quillmere.";
        let ix = TextIndex::build(t);
        let r = ix.search(t, "When was it compelted?", 1, true);
        assert_eq!(r["hits"][0]["sentence"], 1);
    }
}
