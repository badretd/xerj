//! scorer.normalize_answer (xerj-r-scorer-1.0), for parity tests and the "reproduces the official answer" check:
//! NFC, whitespace runs -> one space, strip, lower; then a valid ISO / "D Month YYYY" / "Month D, YYYY" date -> ISO; a
//! leading decimal number (comma thousands groups allowed) + optional unit suffix -> exact plain decimal + unit; else the
//! normalized string. ASCII digits only (Python's `\d` also accepts other Unicode digits; documented divergence).

use unicode_normalization::UnicodeNormalization;
use xerj_store::text::is_py_space;
use xerj_store::value::{canonical_decimal, render_canonical_decimal};

const MONTHS: [&str; 12] = [
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
];

fn month(s: &str) -> Option<u32> {
    MONTHS
        .iter()
        .position(|m| *m == s || &m[..3] == s)
        .map(|i| i as u32 + 1)
}

fn date_or_none(y: u32, m: u32, d: u32) -> Option<String> {
    if !(1..=9999).contains(&y)
        || !(1..=12).contains(&m)
        || d == 0
        || d > xerj_store::value::days_in_month(y, m)
    {
        return None;
    }
    Some(format!("{y:04}-{m:02}-{d:02}"))
}

fn digits(s: &str, min: usize, max: usize) -> Option<u32> {
    if s.len() < min || s.len() > max || !s.bytes().all(|c| c.is_ascii_digit()) {
        return None;
    }
    s.parse().ok()
}

fn as_date(t: &str) -> Option<String> {
    let b = t.as_bytes();
    if b.len() == 10 && b[4] == b'-' && b[7] == b'-' {
        if let (Some(y), Some(m), Some(d)) = (
            digits(&t[..4], 4, 4),
            digits(&t[5..7], 2, 2),
            digits(&t[8..], 2, 2),
        ) {
            return date_or_none(y, m, d);
        }
    }
    let parts: Vec<&str> = t.split(' ').collect();
    if parts.len() == 3 {
        // ^(\d{1,2}) ([a-z]+),? (\d{4})$
        let mid = parts[1].strip_suffix(',').unwrap_or(parts[1]);
        if let (Some(d), true, Some(y)) = (
            digits(parts[0], 1, 2),
            !mid.is_empty() && mid.bytes().all(|c| c.is_ascii_lowercase()),
            digits(parts[2], 4, 4),
        ) {
            if let Some(m) = month(mid) {
                return date_or_none(y, m, d);
            }
            return None;
        }
        // ^([a-z]+) (\d{1,2}),? (\d{4})$
        let dd = parts[1].strip_suffix(',').unwrap_or(parts[1]);
        if parts[0].bytes().all(|c| c.is_ascii_lowercase()) && !parts[0].is_empty() {
            if let (Some(d), Some(y)) = (digits(dd, 1, 2), digits(parts[2], 4, 4)) {
                if let Some(m) = month(parts[0]) {
                    return date_or_none(y, m, d);
                }
            }
        }
    }
    None
}

/// `_NUM` match: (number text, unit suffix)
fn split_number(t: &str) -> Option<(&str, &str)> {
    let b = t.as_bytes();
    let mut i = 0;
    if i < b.len() && (b[i] == b'+' || b[i] == b'-') {
        i += 1;
    }
    let ds = i;
    while i < b.len() && b[i].is_ascii_digit() {
        i += 1;
    }
    let first = i - ds;
    if first == 0 {
        return None;
    }
    // (?:\d{1,3}(?:,\d{3})+|\d+): try comma groups when the first run has <= 3 digits and a ",ddd" follows
    let mut j = i;
    if first <= 3 {
        let mut k = i;
        let mut groups = 0;
        while k + 4 <= b.len()
            && b[k] == b','
            && b[k + 1..k + 4].iter().all(|c| c.is_ascii_digit())
            && (k + 4 == b.len() || !b[k + 4].is_ascii_digit())
        {
            k += 4;
            groups += 1;
        }
        if groups > 0 {
            j = k;
        }
    }
    let mut i = j;
    if i + 1 < b.len() && b[i] == b'.' && b[i + 1].is_ascii_digit() {
        i += 1;
        while i < b.len() && b[i].is_ascii_digit() {
            i += 1;
        }
    }
    let no_exp = i;
    if i < b.len() && b[i] == b'e' {
        let mut k = i + 1;
        if k < b.len() && (b[k] == b'+' || b[k] == b'-') {
            k += 1;
        }
        let st = k;
        while k < b.len() && b[k].is_ascii_digit() {
            k += 1;
        }
        if k > st {
            i = k;
        }
    }
    // the regex backtracks out of the optional exponent when the remainder cannot be a unit (shorter digit runs, fewer
    // comma groups or a dropped fraction never help: the remainder would start with a digit, ',' or '.')
    for end in [i, no_exp] {
        let rest = &t[end..];
        if rest.is_empty() {
            return Some((&t[..end], ""));
        }
        // (?:\s*([^\d\s,.+\-].*))?$
        let unit = rest.trim_start_matches(is_py_space);
        if let Some(c) = unit.chars().next() {
            if !(c.is_ascii_digit() || is_py_space(c) || matches!(c, ',' | '.' | '+' | '-')) {
                return Some((&t[..end], unit));
            }
        }
        if end == no_exp {
            break;
        }
    }
    None
}

fn plain_decimal(canon: &str) -> String {
    match canon.split_once('E') {
        None => canon.to_string(),
        Some((_, e)) => {
            let e: i64 = e.parse().unwrap_or(0);
            if (-40..=40).contains(&e) {
                render_canonical_decimal(canon)
            } else {
                canon.to_string()
            }
        }
    }
}

fn as_number(t: &str) -> Option<String> {
    let (num, unit) = split_number(t)?;
    let canon = canonical_decimal(&num.replace(',', "")).ok()?;
    let unit = unit.trim_matches(is_py_space);
    let out = format!("{} {}", plain_decimal(&canon), unit);
    Some(out.trim_matches(is_py_space).to_string())
}

/// scorer.normalize_answer for a string answer ("" for None)
pub fn normalize_answer(s: Option<&str>) -> String {
    let Some(s) = s else { return String::new() };
    let nfc: String = s.nfc().collect();
    let mut t = String::with_capacity(nfc.len());
    let mut ws = false;
    for c in nfc.chars() {
        if is_py_space(c) {
            ws = true;
            continue;
        }
        if ws && !t.is_empty() {
            t.push(' ');
        }
        ws = false;
        t.push(c);
    }
    let t = t.to_lowercase();
    if t.is_empty() {
        return t;
    }
    if let Some(d) = as_date(&t) {
        return d;
    }
    if let Some(n) = as_number(&t) {
        return n;
    }
    t
}

/// scorer.answers_match
pub fn answers_match(predicted: Option<&str>, accepted: &[&str]) -> bool {
    let k = normalize_answer(predicted);
    !k.is_empty() && accepted.iter().any(|g| normalize_answer(Some(g)) == k)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalization() {
        assert_eq!(normalize_answer(Some("  3 February 1984 ")), "1984-02-03");
        assert_eq!(normalize_answer(Some("February 3, 1984")), "1984-02-03");
        assert_eq!(normalize_answer(Some("Feb 30, 1984")), "feb 30, 1984");
        assert_eq!(normalize_answer(Some("589 m")), "589 m");
        assert_eq!(normalize_answer(Some("589.0 m")), "589 m");
        assert_eq!(normalize_answer(Some("589m")), "589 m");
        assert_eq!(normalize_answer(Some("36,490")), "36490");
        assert_eq!(normalize_answer(Some("1,2")), "1,2");
        assert_eq!(normalize_answer(Some("12 345")), "12 345");
        assert_eq!(normalize_answer(Some("Bill  de Blasio")), "bill de blasio");
        assert_eq!(normalize_answer(None), "");
    }
}
