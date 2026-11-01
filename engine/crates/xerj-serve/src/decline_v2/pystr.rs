//! The Python `str` behaviour the frozen rules depend on (CPython's own Unicode tables, `pytables.rs`, generated): `lower()` and
//! `upper()` with full mappings and the final-sigma rule, `strip()`, whitespace `split()`, and one-character predicates. Rust's
//! `char` methods use another Unicode version and other definitions (`\w`, `is_uppercase`), so they are not used here.

use super::pytables as t;

fn in_ranges(tab: &[(u32, u32)], c: char) -> bool {
    let c = c as u32;
    tab.binary_search_by(|&(a, b)| {
        if c < a {
            std::cmp::Ordering::Greater
        } else if c > b {
            std::cmp::Ordering::Less
        } else {
            std::cmp::Ordering::Equal
        }
    })
    .is_ok()
}

/// regex `\w` (`str.isalnum()` or `_`)
#[inline]
pub fn is_word(c: char) -> bool {
    if c.is_ascii() {
        c.is_ascii_alphanumeric() || c == '_'
    } else {
        in_ranges(t::WORD, c)
    }
}

/// regex `\d` (Unicode decimal digits)
#[inline]
pub fn is_digit(c: char) -> bool {
    if c.is_ascii() {
        c.is_ascii_digit()
    } else {
        in_ranges(t::DIGIT, c)
    }
}

/// regex `\s` and `str.isspace()`
#[inline]
pub fn is_space(c: char) -> bool {
    in_ranges(t::SPACE, c)
}

pub fn is_alpha(c: char) -> bool {
    in_ranges(t::ALPHA, c)
}

pub fn is_upper(c: char) -> bool {
    in_ranges(t::UPPER, c)
}

pub fn is_lower(c: char) -> bool {
    in_ranges(t::LOWER, c)
}

/// `s[:1].isupper()`
pub fn first_is_upper(s: &str) -> bool {
    s.chars().next().is_some_and(is_upper)
}

/// `s[:1].islower()`
pub fn first_is_lower(s: &str) -> bool {
    s.chars().next().is_some_and(is_lower)
}

/// the IGNORECASE class of a character: two characters match under `re.I` iff their classes are equal
#[inline]
pub fn ignorecase_canon(c: char) -> u32 {
    let u = c as u32;
    match t::IGNORECASE_CANON.binary_search_by_key(&u, |e| e.0) {
        Ok(i) => t::IGNORECASE_CANON[i].1,
        Err(_) => u,
    }
}

fn map_full(tab: &[(u32, &[u32])], c: char, out: &mut String) {
    match tab.binary_search_by_key(&(c as u32), |e| e.0) {
        Ok(i) => {
            for &x in tab[i].1 {
                if let Some(m) = char::from_u32(x) {
                    out.push(m);
                }
            }
        }
        Err(_) => out.push(c),
    }
}

/// CPython `handle_capital_sigma`
fn final_sigma(cs: &[char], i: usize) -> bool {
    let mut j = i;
    let mut before = None;
    while j > 0 {
        j -= 1;
        if !in_ranges(t::CASE_IGNORABLE, cs[j]) {
            before = Some(cs[j]);
            break;
        }
    }
    if !before.is_some_and(|c| in_ranges(t::CASED_NOT_IGNORABLE, c)) {
        return false;
    }
    for &c in &cs[i + 1..] {
        if !in_ranges(t::CASE_IGNORABLE, c) {
            return !in_ranges(t::CASED_NOT_IGNORABLE, c);
        }
    }
    true
}

/// `s.lower()`
pub fn lower(s: &str) -> String {
    if s.is_ascii() {
        return s.to_ascii_lowercase();
    }
    let cs: Vec<char> = s.chars().collect();
    let mut out = String::with_capacity(s.len());
    for (i, &c) in cs.iter().enumerate() {
        if c == '\u{3A3}' {
            out.push(if final_sigma(&cs, i) {
                '\u{3C2}'
            } else {
                '\u{3C3}'
            });
        } else {
            map_full(t::LOWER_MAP, c, &mut out);
        }
    }
    out
}

/// `s.upper()`
pub fn upper(s: &str) -> String {
    if s.is_ascii() {
        return s.to_ascii_uppercase();
    }
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        map_full(t::UPPER_MAP, c, &mut out);
    }
    out
}

/// `s.strip()`
pub fn strip(s: &str) -> &str {
    s.trim_matches(is_space)
}

/// `s.split()`
pub fn split_ws(s: &str) -> Vec<&str> {
    s.split(is_space).filter(|p| !p.is_empty()).collect()
}

/// `len(s)` (code points)
pub fn py_len(s: &str) -> usize {
    s.chars().count()
}

/// the policy tool's `norm`: `s.strip().lower().rstrip(".")`
pub fn norm(s: &str) -> String {
    lower(strip(s)).trim_end_matches('.').to_string()
}
