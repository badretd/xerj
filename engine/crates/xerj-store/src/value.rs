//! Typed values (reference_core.TypedValue): kinds, exact decimal canonicalization / comparison / rendering, ISO dates.
//!
//! Numbers are kept as reference_core.canonical_decimal strings ("185E0", "15E-1", "0"): exact, no float rounding, and
//! equality of canonical strings is numeric equality. Comparison and `format(Decimal, "f")` rendering work on that form
//! directly (no big-number arithmetic needed).

use crate::error::*;
use std::cmp::Ordering;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum Kind {
    Text = 0,
    Date = 1,
    Number = 2,
    Boolean = 3,
}

impl Kind {
    pub fn parse(s: &str) -> Option<Kind> {
        Some(match s {
            "text" => Kind::Text,
            "date" => Kind::Date,
            "number" => Kind::Number,
            "boolean" => Kind::Boolean,
            _ => return None,
        })
    }
    pub fn as_str(self) -> &'static str {
        match self {
            Kind::Text => "text",
            Kind::Date => "date",
            Kind::Number => "number",
            Kind::Boolean => "boolean",
        }
    }
    pub fn from_u8(v: u8) -> Option<Kind> {
        Some(match v {
            0 => Kind::Text,
            1 => Kind::Date,
            2 => Kind::Number,
            3 => Kind::Boolean,
            _ => return None,
        })
    }
}

/// A typed value. `value` is the canonical string (numbers: canonical_decimal; booleans: "true" / "false").
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TypedValue<'a> {
    pub kind: Kind,
    pub value: &'a str,
    pub unit: &'a str,
}

impl<'a> TypedValue<'a> {
    pub fn boolean(b: bool) -> TypedValue<'static> {
        TypedValue {
            kind: Kind::Boolean,
            value: if b { "true" } else { "false" },
            unit: "",
        }
    }
    pub fn as_bool(&self) -> Option<bool> {
        (self.kind == Kind::Boolean).then(|| self.value == "true")
    }
}

/// `xr_codecs.render_value`: numbers `format(Decimal, "f")` + unit (stripped), booleans yes / no, else the value.
pub fn render_value(v: &TypedValue<'_>) -> String {
    match v.kind {
        Kind::Number => {
            let mut s = render_canonical_decimal(v.value);
            if !v.unit.is_empty() {
                s.push(' ');
                s.push_str(v.unit);
            }
            // f"{...} {unit}".strip(): a unit is a nonempty string; strip only matters for units with outer whitespace
            if v.unit
                .chars()
                .next_back()
                .is_some_and(crate::text::is_py_space)
            {
                let t = s.trim_end_matches(crate::text::is_py_space).len();
                s.truncate(t);
            }
            s
        }
        Kind::Boolean => {
            if v.value == "true" {
                "yes".into()
            } else {
                "no".into()
            }
        }
        _ => v.value.to_string(),
    }
}

/// `render_value` appended to `out` (no allocation for text / date values)
pub fn render_value_into(v: &TypedValue<'_>, out: &mut String) {
    match v.kind {
        Kind::Number => out.push_str(&render_value(v)),
        Kind::Boolean => out.push_str(if v.value == "true" { "yes" } else { "no" }),
        _ => out.push_str(v.value),
    }
}

/// `xr_codecs.result_text` of a VALUE: "VALUE {kind} {render}".
pub fn value_text(v: &TypedValue<'_>) -> String {
    format!("VALUE {} {}", v.kind.as_str(), render_value(v))
}

// ------------------------------------------------------------------------------------------------ dates

/// reference_core.iso_date: exact YYYY-MM-DD and a valid proleptic Gregorian date (year 1..=9999).
pub fn validate_iso_date(s: &str) -> Result<(), ContractError> {
    let b = s.as_bytes();
    let digits = |r: std::ops::Range<usize>| b[r].iter().all(|c| c.is_ascii_digit());
    if b.len() != 10
        || b[4] != b'-'
        || b[7] != b'-'
        || !digits(0..4)
        || !digits(5..7)
        || !digits(8..10)
    {
        // Python's \d also matches non-ASCII digits, which date.fromisoformat then rejects: same outcome (an error)
        return Err(ContractError::new(E_DATE_FORMAT));
    }
    let num = |r: std::ops::Range<usize>| b[r].iter().fold(0u32, |a, c| a * 10 + (c - b'0') as u32);
    let (y, m, d) = (num(0..4), num(5..7), num(8..10));
    if y == 0 {
        return Err(ContractError::owned(format!("year {y} is out of range")));
    }
    if !(1..=12).contains(&m) {
        return Err(ContractError::new("month must be in 1..12"));
    }
    if d == 0 || d > days_in_month(y, m) {
        return Err(ContractError::new("day is out of range for month"));
    }
    Ok(())
}

#[inline]
pub fn days_in_month(y: u32, m: u32) -> u32 {
    match m {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        _ => {
            if (y.is_multiple_of(4) && !y.is_multiple_of(100)) || y.is_multiple_of(400) {
                29
            } else {
                28
            }
        }
    }
}

/// year of a validated ISO date (`int(str(v.value)[:4])`)
#[inline]
pub fn date_year(s: &str) -> u32 {
    s.as_bytes()[..4]
        .iter()
        .fold(0u32, |a, c| a * 10 + (c - b'0') as u32)
}

// ------------------------------------------------------------------------------------------------ decimals

/// reference_core.canonical_decimal(Decimal(s)): the exact canonical form "[-]coefE<exp>" (coefficient without leading /
/// trailing zeros) or "0". Accepts Python's Decimal string syntax (surrounding whitespace, underscores, sign, fraction,
/// exponent); infinities / NaNs fail "Numeric values must be finite" like the reference. Non-ASCII digits (accepted by
/// Python's Decimal) are rejected as "Invalid decimal" (documented divergence; never seen in the data).
pub fn canonical_decimal(input: &str) -> Result<String, ContractError> {
    // CPython's _decimal: strip surrounding whitespace, then drop every underscore, then parse
    let trimmed = input.trim_matches(crate::text::is_py_space);
    let cleaned: String;
    let s = if trimmed.contains('_') {
        cleaned = trimmed.chars().filter(|&c| c != '_').collect();
        cleaned.as_str()
    } else {
        trimmed
    };
    let b = s.as_bytes();
    let mut i = 0;
    let neg = match b.first() {
        Some(b'-') => {
            i = 1;
            true
        }
        Some(b'+') => {
            i = 1;
            false
        }
        _ => false,
    };
    let body = &s[i..];
    let lower = body.to_ascii_lowercase();
    if lower == "inf" || lower == "infinity" {
        return Err(ContractError::new(E_NUM_FINITE));
    }
    for p in ["nan", "snan"] {
        if let Some(rest) = lower.strip_prefix(p) {
            if rest.bytes().all(|c| c.is_ascii_digit()) {
                return Err(ContractError::new(E_NUM_FINITE));
            }
        }
    }
    let invalid = || ContractError::new(E_NUM_INVALID);
    let mut coef = String::with_capacity(b.len());
    let mut n_frac: i64 = 0;
    let mut seen_digit = false;
    while i < b.len() && b[i].is_ascii_digit() {
        coef.push(b[i] as char);
        seen_digit = true;
        i += 1;
    }
    if i < b.len() && b[i] == b'.' {
        i += 1;
        while i < b.len() && b[i].is_ascii_digit() {
            coef.push(b[i] as char);
            n_frac += 1;
            seen_digit = true;
            i += 1;
        }
    }
    if !seen_digit {
        return Err(invalid());
    }
    let mut exp: i64 = 0;
    if i < b.len() && (b[i] == b'e' || b[i] == b'E') {
        i += 1;
        let eneg = match b.get(i) {
            Some(b'-') => {
                i += 1;
                true
            }
            Some(b'+') => {
                i += 1;
                false
            }
            _ => false,
        };
        let st = i;
        while i < b.len() && b[i].is_ascii_digit() {
            exp = exp
                .checked_mul(10)
                .and_then(|e| e.checked_add((b[i] - b'0') as i64))
                .ok_or_else(invalid)?;
            i += 1;
        }
        if i == st {
            return Err(invalid());
        }
        if eneg {
            exp = -exp;
        }
    }
    if i != b.len() {
        return Err(invalid());
    }
    let mut exp = exp.checked_sub(n_frac).ok_or_else(invalid)?;
    let lead = coef.bytes().take_while(|&c| c == b'0').count();
    if lead == coef.len() {
        return Ok("0".to_string());
    }
    let mut digits = &coef[lead..];
    let trail = digits.bytes().rev().take_while(|&c| c == b'0').count();
    digits = &digits[..digits.len() - trail];
    exp = exp.checked_add(trail as i64).ok_or_else(invalid)?;
    Ok(format!("{}{}E{}", if neg { "-" } else { "" }, digits, exp))
}

/// (negative, coefficient digits, exponent) of a canonical decimal; "0" -> (false, "", 0)
#[inline]
fn split_canonical(c: &str) -> (bool, &str, i64) {
    if c == "0" {
        return (false, "", 0);
    }
    let (neg, rest) = match c.strip_prefix('-') {
        Some(r) => (true, r),
        None => (false, c),
    };
    let e = rest.find('E').unwrap_or(rest.len());
    let exp = rest
        .get(e + 1..)
        .and_then(|x| x.parse::<i64>().ok())
        .unwrap_or(0);
    (neg, &rest[..e], exp)
}

/// Exact numeric order of two canonical decimals.
pub fn cmp_canonical_decimal(a: &str, b: &str) -> Ordering {
    if a == b {
        return Ordering::Equal;
    }
    let (an, ad, ae) = split_canonical(a);
    let (bn, bd, be) = split_canonical(b);
    let sign = |neg: bool, d: &str| {
        if d.is_empty() {
            0
        } else if neg {
            -1
        } else {
            1
        }
    };
    let (sa, sb) = (sign(an, ad), sign(bn, bd));
    if sa != sb {
        return sa.cmp(&sb);
    }
    if sa == 0 {
        return Ordering::Equal;
    }
    // magnitude: adjusted exponent first (position of the most significant digit), then the digits left-aligned
    let adj_a = ad.len() as i128 + ae as i128;
    let adj_b = bd.len() as i128 + be as i128;
    let mag = adj_a.cmp(&adj_b).then_with(|| {
        let (x, y) = (ad.as_bytes(), bd.as_bytes());
        let n = x.len().max(y.len());
        for k in 0..n {
            let cx = x.get(k).copied().unwrap_or(b'0');
            let cy = y.get(k).copied().unwrap_or(b'0');
            if cx != cy {
                return cx.cmp(&cy);
            }
        }
        Ordering::Equal
    });
    if sa < 0 {
        mag.reverse()
    } else {
        mag
    }
}

/// `format(Decimal(canonical), "f")`.
pub fn render_canonical_decimal(c: &str) -> String {
    let (neg, digits, exp) = split_canonical(c);
    if digits.is_empty() {
        return "0".to_string();
    }
    let mut out = String::with_capacity(digits.len() + 4);
    if neg {
        out.push('-');
    }
    if exp >= 0 {
        out.push_str(digits);
        for _ in 0..exp {
            out.push('0');
        }
    } else {
        let pos = digits.len() as i64 + exp;
        if pos > 0 {
            out.push_str(&digits[..pos as usize]);
            out.push('.');
            out.push_str(&digits[pos as usize..]);
        } else {
            out.push_str("0.");
            for _ in 0..(-pos) {
                out.push('0');
            }
            out.push_str(digits);
        }
    }
    out
}

/// canonical decimal of a non-negative integer (the PICK year conversion `TypedValue("number", str(year), "year")`)
pub fn canonical_of_u32(v: u32) -> String {
    canonical_decimal(&v.to_string()).expect("integer is a valid decimal")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical() {
        for (i, o) in [
            ("185", "185E0"),
            ("180", "18E1"),
            ("185.0", "185E0"),
            ("1.50", "15E-1"),
            ("-0", "0"),
            ("0.000", "0"),
            ("0012", "12E0"),
            ("1e5", "1E5"),
            (" +7_000 ", "7E3"),
            ("-1.25E-3", "-125E-5"),
            (".5", "5E-1"),
            ("5.", "5E0"),
        ] {
            assert_eq!(canonical_decimal(i).unwrap(), o, "{i}");
        }
        for bad in ["", "abc", "1.2.3", "1e", "--1", "1 2"] {
            assert_eq!(
                canonical_decimal(bad).unwrap_err().message(),
                E_NUM_INVALID,
                "{bad}"
            );
        }
        for nf in ["inf", "-Infinity", "NaN", "sNaN12"] {
            assert_eq!(
                canonical_decimal(nf).unwrap_err().message(),
                E_NUM_FINITE,
                "{nf}"
            );
        }
    }

    #[test]
    fn compare_and_render() {
        let c = |x: &str| canonical_decimal(x).unwrap();
        let pairs = [
            ("1", "2"),
            ("-5", "3"),
            ("0", "0.1"),
            ("-0.5", "0"),
            ("99", "100"),
            ("1.5", "1.25"),
            ("-2", "-10"),
            ("1e3", "999"),
        ];
        for (a, b) in pairs {
            let fa: f64 = a.parse().unwrap();
            let fb: f64 = b.parse().unwrap();
            assert_eq!(
                cmp_canonical_decimal(&c(a), &c(b)),
                fa.partial_cmp(&fb).unwrap(),
                "{a} {b}"
            );
        }
        assert_eq!(render_canonical_decimal("15E-1"), "1.5");
        assert_eq!(render_canonical_decimal("1E5"), "100000");
        assert_eq!(render_canonical_decimal("1E-7"), "0.0000001");
        assert_eq!(render_canonical_decimal("-125E-5"), "-0.00125");
        assert_eq!(render_canonical_decimal("0"), "0");
        let v = TypedValue {
            kind: Kind::Number,
            value: "185E0",
            unit: "cm",
        };
        assert_eq!(render_value(&v), "185 cm");
        assert_eq!(value_text(&v), "VALUE number 185 cm");
    }

    #[test]
    fn dates() {
        assert!(validate_iso_date("2024-02-29").is_ok());
        assert!(validate_iso_date("2023-02-29").is_err());
        assert!(validate_iso_date("0000-01-01").is_err());
        assert!(validate_iso_date("2024-1-01").is_err());
        assert_eq!(date_year("1993-05-01"), 1993);
    }
}
