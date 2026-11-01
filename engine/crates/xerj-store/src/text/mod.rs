//! Python-exact text primitives shared by every xerj-mem crate.
//!
//! The reference semantics depend on four predicates / mappings of the CPython that ran it (3.12, Unicode 15.0): `re`'s `\w`
//! (word boundaries of `xr_codecs.link_entities`, tokens of `reference_core.LexicalIndex`), `str.casefold()`,
//! `unicodedata.category(c)[0] == "P"` and `str.isspace()` (`stack_v2.normalize_entity`). `py_tables.rs` is generated from
//! that interpreter (research/xerj_r_mvp/tools/xerj_mem_unicode_tables.py), so these functions agree with it on every code
//! point. Rust's own `char::is_alphanumeric` / `to_lowercase` differ (e.g. Other_Alphabetic marks, final sigma), so they are
//! never used on parity paths. NFKC comes from the `unicode-normalization` crate (Unicode 16 vs Python's 15.0: identical on
//! all characters assigned in 15.0).

mod py_tables;

pub use py_tables::PY_UNIDATA_VERSION;

use std::borrow::Cow;
use unicode_normalization::UnicodeNormalization;

const fn ascii_word_table() -> [bool; 128] {
    let mut t = [false; 128];
    let mut i = 0;
    while i < 128 {
        let c = i as u8;
        t[i] = c.is_ascii_alphanumeric() || c == b'_';
        i += 1;
    }
    t
}

static ASCII_WORD: [bool; 128] = ascii_word_table();

#[inline]
fn in_ranges(ranges: &[(u32, u32)], cp: u32) -> bool {
    let i = ranges.partition_point(|r| r.0 <= cp);
    i > 0 && cp <= ranges[i - 1].1
}

/// Python `re` `\w` for str patterns (Unicode alphanumerics + underscore, CPython's exact definition).
#[inline]
pub fn is_py_word(c: char) -> bool {
    let cp = c as u32;
    if cp < 0x80 {
        ASCII_WORD[cp as usize]
    } else {
        in_ranges(py_tables::WORD_RANGES, cp)
    }
}

/// `str.isspace()` (also `str.split()` / `str.strip()` and `re` `\s` for str patterns).
#[inline]
pub fn is_py_space(c: char) -> bool {
    let cp = c as u32;
    if cp < 0x80 {
        matches!(cp, 0x09..=0x0D | 0x1C..=0x20)
    } else {
        py_tables::SPACE_CHARS.binary_search(&cp).is_ok()
    }
}

/// `unicodedata.category(c).startswith("P")`.
#[inline]
pub fn is_py_punct(c: char) -> bool {
    let cp = c as u32;
    if cp < 0x80 {
        (c as u8).is_ascii_punctuation()
            && !matches!(c, '$' | '+' | '<' | '=' | '>' | '^' | '`' | '|' | '~')
    } else {
        in_ranges(py_tables::PUNCT_RANGES, cp)
    }
}

/// Append `str.casefold()` of one char.
#[inline]
pub fn casefold_char_into(c: char, out: &mut String) {
    let cp = c as u32;
    if cp < 0x80 {
        out.push((c as u8).to_ascii_lowercase() as char);
    } else {
        match py_tables::CASEFOLD.binary_search_by_key(&cp, |e| e.0) {
            Ok(i) => out.push_str(py_tables::CASEFOLD[i].1),
            Err(_) => out.push(c),
        }
    }
}

/// Append `s.casefold()`.
pub fn casefold_into(s: &str, out: &mut String) {
    if s.is_ascii() {
        out.reserve(s.len());
        for b in s.bytes() {
            out.push(b.to_ascii_lowercase() as char);
        }
        return;
    }
    for c in s.chars() {
        casefold_char_into(c, out);
    }
}

/// `s.casefold()`, borrowed when already folded ASCII.
pub fn casefold(s: &str) -> Cow<'_, str> {
    if s.is_ascii() && !s.bytes().any(|b| b.is_ascii_uppercase()) {
        return Cow::Borrowed(s);
    }
    let mut out = String::with_capacity(s.len());
    casefold_into(s, &mut out);
    Cow::Owned(out)
}

/// `stack_v2.normalize_entity`: NFKC, casefold, every P* char -> space, whitespace runs collapsed, stripped.
pub fn normalize_entity(s: &str) -> String {
    let nfkc: Cow<'_, str> = if s.is_ascii() {
        Cow::Borrowed(s)
    } else {
        Cow::Owned(s.nfkc().collect())
    };
    let mut folded = String::with_capacity(nfkc.len());
    casefold_into(&nfkc, &mut folded);
    let mut out = String::with_capacity(folded.len());
    let mut pending = false;
    for c in folded.chars() {
        let c = if is_py_punct(c) { ' ' } else { c };
        if is_py_space(c) {
            pending = !out.is_empty();
            continue;
        }
        if pending {
            out.push(' ');
            pending = false;
        }
        out.push(c);
    }
    out
}

/// `json.dumps(s, ensure_ascii=False)` string literal (Python's escaping: `\"`, `\\`, `\n`, `\r`, `\t`, `\b`, `\f`, other
/// C0 controls as lowercase `\u00XX`; everything else, including DEL and U+2028/2029, raw UTF-8).
pub fn py_json_str(s: &str, out: &mut Vec<u8>) {
    out.push(b'"');
    let bytes = s.as_bytes();
    let mut start = 0;
    for (i, &b) in bytes.iter().enumerate() {
        if b >= 0x20 && b != b'"' && b != b'\\' {
            continue;
        }
        out.extend_from_slice(&bytes[start..i]);
        match b {
            b'"' => out.extend_from_slice(b"\\\""),
            b'\\' => out.extend_from_slice(b"\\\\"),
            b'\n' => out.extend_from_slice(b"\\n"),
            b'\r' => out.extend_from_slice(b"\\r"),
            b'\t' => out.extend_from_slice(b"\\t"),
            0x08 => out.extend_from_slice(b"\\b"),
            0x0C => out.extend_from_slice(b"\\f"),
            _ => {
                const HEX: &[u8; 16] = b"0123456789abcdef";
                out.extend_from_slice(b"\\u00");
                out.push(HEX[(b >> 4) as usize]);
                out.push(HEX[(b & 15) as usize]);
            }
        }
        start = i + 1;
    }
    out.extend_from_slice(&bytes[start..]);
    out.push(b'"');
}

/// Number of code points (Python `len(str)`).
#[inline]
pub fn char_len(s: &str) -> usize {
    if s.is_ascii() {
        s.len()
    } else {
        s.chars().count()
    }
}

/// Byte range of the code-point span `[start, end)` of `s` (`ascii` = s is ASCII, then offsets coincide). None when the
/// span exceeds the text.
pub fn char_span_to_bytes(
    s: &str,
    ascii: bool,
    start: usize,
    end: usize,
) -> Option<(usize, usize)> {
    if ascii {
        return if end <= s.len() {
            Some((start, end))
        } else {
            None
        };
    }
    let mut bs = None;
    let mut n = 0usize;
    for (bi, _) in s.char_indices() {
        if n == start {
            bs = Some(bi);
        }
        if n == end {
            return bs.map(|b| (b, bi));
        }
        n += 1;
    }
    if n == end {
        return bs.map(|b| (b, s.len()));
    }
    None
}

/// Python-exact `\w+` tokens of `text.casefold()` (reference_core.LexicalIndex.tokens), appended to `out` as owned
/// strings; `scratch` is reused. The fast path covers ASCII; other text goes through the generated casefold / word tables.
pub fn py_word_tokens(text: &str, scratch: &mut String, mut emit: impl FnMut(&str)) {
    if text.is_ascii() {
        // ASCII: tokens are emitted as slices of `text` (copied only when they contain uppercase letters)
        let b = text.as_bytes();
        let mut i = 0;
        while i < b.len() {
            while i < b.len() && !ASCII_WORD[b[i] as usize] {
                i += 1;
            }
            let st = i;
            let mut upper = false;
            while i < b.len() && ASCII_WORD[b[i] as usize] {
                upper |= b[i].is_ascii_uppercase();
                i += 1;
            }
            if i > st {
                if upper {
                    scratch.clear();
                    for &c in &b[st..i] {
                        scratch.push(c.to_ascii_lowercase() as char);
                    }
                    emit(scratch);
                } else {
                    emit(&text[st..i]);
                }
            }
        }
        scratch.clear();
        return;
    }
    let mut folded = String::with_capacity(text.len() + 8);
    casefold_into(text, &mut folded);
    scratch.clear();
    for c in folded.chars() {
        if is_py_word(c) {
            scratch.push(c);
        } else if !scratch.is_empty() {
            emit(scratch);
            scratch.clear();
        }
    }
    if !scratch.is_empty() {
        emit(scratch);
        scratch.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn word_and_space() {
        assert!(
            is_py_word('a')
                && is_py_word('_')
                && is_py_word('9')
                && is_py_word('é')
                && is_py_word('中')
        );
        assert!(!is_py_word('-') && !is_py_word(' ') && !is_py_word('\u{0903}')); // Devanagari visarga: Mc, not \w in Python
        assert!(is_py_space('\u{a0}') && is_py_space('\u{1c}') && !is_py_space('\u{200b}'));
        assert!(
            is_py_punct('-')
                && is_py_punct('.')
                && is_py_punct('«')
                && !is_py_punct('$')
                && !is_py_punct('+')
        );
    }

    #[test]
    fn ascii_fast_paths_match_tables() {
        for cp in 0u32..128 {
            let c = char::from_u32(cp).unwrap();
            assert_eq!(
                is_py_word(c),
                in_ranges(py_tables::WORD_RANGES, cp),
                "word {cp}"
            );
            assert_eq!(
                is_py_punct(c),
                in_ranges(py_tables::PUNCT_RANGES, cp),
                "punct {cp}"
            );
            assert_eq!(
                is_py_space(c),
                py_tables::SPACE_CHARS.binary_search(&cp).is_ok(),
                "space {cp}"
            );
            let mut a = String::new();
            casefold_char_into(c, &mut a);
            let b = match py_tables::CASEFOLD.binary_search_by_key(&cp, |e| e.0) {
                Ok(i) => py_tables::CASEFOLD[i].1.to_string(),
                Err(_) => c.to_string(),
            };
            assert_eq!(a, b, "fold {cp}");
        }
    }

    #[test]
    fn folding_and_entities() {
        assert_eq!(casefold("Straße ΣΑΣ"), "strasse σασ");
        assert_eq!(normalize_entity("J. R. R. Tolkien"), "j r r tolkien");
        assert_eq!(normalize_entity("Spider-Man"), "spider man");
        assert_eq!(normalize_entity("  ﬁne\u{a0}Café  "), "fine café");
    }

    #[test]
    fn json_escape() {
        let mut v = Vec::new();
        py_json_str("a\"b\\c\nd\u{1}é\u{7f}", &mut v);
        assert_eq!(
            String::from_utf8(v).unwrap(),
            "\"a\\\"b\\\\c\\nd\\u0001é\u{7f}\""
        );
    }

    #[test]
    fn spans() {
        assert_eq!(char_span_to_bytes("héllo", false, 1, 3), Some((1, 4)));
        assert_eq!(char_span_to_bytes("héllo", false, 3, 5), Some((4, 6)));
        assert_eq!(char_span_to_bytes("héllo", false, 3, 6), None);
    }

    #[test]
    fn tokens() {
        let mut s = String::new();
        let mut t = Vec::new();
        py_word_tokens(
            "Was Thanasmor Falvocu Works founded on 3 February 1984?",
            &mut s,
            |x| t.push(x.to_string()),
        );
        assert_eq!(
            t,
            [
                "was",
                "thanasmor",
                "falvocu",
                "works",
                "founded",
                "on",
                "3",
                "february",
                "1984"
            ]
        );
        t.clear();
        py_word_tokens("Straße l'été", &mut s, |x| t.push(x.to_string()));
        assert_eq!(t, ["strasse", "l", "été"]);
    }
}
