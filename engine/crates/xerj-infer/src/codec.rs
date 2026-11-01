//! Memory codecs of the rrt arm (xr_codecs.c1_memory / c1b_memory, stack_v2.hop_memory) and the comparison16 register codec
//! (xr_codecs.register_features).

use crate::config::MemoryCodec;
use crate::error::{Error, Result};
use crate::inputs::CandidateFields;
use crate::model::Embedding;
use crate::tokenizer::TextEncoder;

pub const MAX_FIELD_TOKENS: usize = 16;
pub const ROLE_WIDTH: usize = 4;
pub const C1B_RMS_EPS: f32 = 1e-12;
/// stack-v2: role column set on the subject slot of a resolving hop record
pub const RESOLVED_ROLE_COLUMN: usize = 3;

/// Encode up to `k` candidate records into `values` [k, slots, d + 4] and `slot_mask` [k, slots] (records beyond the
/// candidates are padding: zeros / false). `resolves`: stack-v2 hop flags (role column 3 of the subject slot).
pub fn encode_memory(
    codec: MemoryCodec,
    cands: &[CandidateFields],
    resolves: Option<&[bool]>,
    k: usize,
    tok: &dyn TextEncoder,
    emb: &Embedding,
    values: &mut [f32],
    slot_mask: &mut [bool],
) -> Result<()> {
    let _t = crate::prof::start(crate::prof::Cat::Codec);
    let d = emb.dim;
    let w = d + ROLE_WIDTH;
    let slots = codec.slots();
    if values.len() < k * slots * w || slot_mask.len() < k * slots {
        return Err(Error::Input("memory buffers too small".into()));
    }
    values[..k * slots * w].fill(0.0);
    slot_mask[..k * slots].fill(false);
    let mut field = String::new();
    for (r, f) in cands.iter().take(k).enumerate() {
        let fs = f.fields();
        for s in 0..slots {
            field.clear();
            if codec == MemoryCodec::C1b {
                field.push(' ');
            }
            field.push_str(fs[s]);
            let mut ids = tok.encode(&field)?;
            ids.truncate(MAX_FIELD_TOKENS);
            let v = &mut values[(r * slots + s) * w..(r * slots + s + 1) * w];
            for &id in &ids {
                let row = emb.row(id);
                for i in 0..d {
                    v[i] += row[i];
                }
            }
            let cnt = (ids.len() as f32).max(1.0);
            for x in v[..d].iter_mut() {
                *x /= cnt;
            }
            if codec == MemoryCodec::C1b {
                let mut ms = 0f32;
                for &x in v[..d].iter() {
                    ms += x * x;
                }
                ms /= d as f32;
                let rms = ms.max(C1B_RMS_EPS).sqrt();
                for x in v[..d].iter_mut() {
                    *x /= rms;
                }
            }
            v[d + s] = 1.0;
            slot_mask[r * slots + s] = true;
        }
        if let Some(res) = resolves {
            if res.get(r).copied().unwrap_or(false) {
                values[(r * slots) * w + d + RESOLVED_ROLE_COLUMN] = 1.0;
            }
        }
    }
    Ok(())
}

/// executor result status (reference_core.Status), in the register codec's STATUS_ORDER
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    Value = 0,
    True = 1,
    False = 2,
    Unknown = 3,
    Conflict = 4,
}

impl Status {
    pub fn parse(s: &str) -> Option<Status> {
        Some(match s.to_ascii_uppercase().as_str() {
            "VALUE" => Status::Value,
            "TRUE" => Status::True,
            "FALSE" => Status::False,
            "UNKNOWN" => Status::Unknown,
            "CONFLICT" => Status::Conflict,
            _ => return None,
        })
    }
}

/// typed value of a VALUE result (reference_core.TypedValue); numbers are canonical decimal strings ("25E-1", "0")
#[derive(Debug, Clone, PartialEq)]
pub enum TypedValue {
    Text(String),
    Date(String),
    Number(String),
    Boolean(bool),
}

/// one executed stack entry: (Result, producing step)
#[derive(Debug, Clone, PartialEq)]
pub struct StackEntry {
    pub status: Status,
    pub value: Option<TypedValue>,
    pub step: usize,
}

pub const REGISTER_COUNT: usize = 4;
pub const REGISTER_WIDTH: usize = 16;

fn day_of_year(y: i32, m: u32, d: u32) -> u32 {
    let leap = (y % 4 == 0 && y % 100 != 0) || y % 400 == 0;
    let cum = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];
    cum[(m - 1) as usize] + d + if leap && m > 2 { 1 } else { 0 }
}

/// decimal string -> (f64 value, adjusted exponent + 1) as Python's Decimal: float(dec), dec.copy_abs().adjusted() + 1
fn decimal_parts(s: &str) -> Option<(f64, f64, bool)> {
    let x: f64 = s.parse().ok()?;
    let t = s.trim();
    let neg = t.starts_with('-');
    let body = t.trim_start_matches(['-', '+']);
    let (mant, exp) = match body.find(['E', 'e']) {
        Some(i) => (&body[..i], body[i + 1..].parse::<i64>().ok()?),
        None => (body, 0),
    };
    let (int_part, frac_part) = match mant.find('.') {
        Some(i) => (&mant[..i], &mant[i + 1..]),
        None => (mant, ""),
    };
    let digits: String = format!("{int_part}{frac_part}");
    let exponent = exp - frac_part.len() as i64;
    let trimmed = digits.trim_start_matches('0');
    let ndig = if trimmed.is_empty() {
        1
    } else {
        trimmed.len() as i64
    };
    let adjusted = exponent + ndig - 1;
    Some((x, adjusted as f64 + 1.0, neg))
}

/// comparison16 registers for the stack (bottom -> top) produced by actions <= action_step: feats [4, 16], mask [4]
pub fn register_features(
    stack: &[StackEntry],
    feats: &mut [f32; REGISTER_COUNT * REGISTER_WIDTH],
    mask: &mut [bool; REGISTER_COUNT],
) {
    feats.fill(0.0);
    mask.fill(false);
    let start = stack.len().saturating_sub(REGISTER_COUNT);
    for (slot, e) in stack[start..].iter().enumerate() {
        let f = &mut feats[slot * REGISTER_WIDTH..(slot + 1) * REGISTER_WIDTH];
        f[e.status as usize] = 1.0;
        // kind: the value's kind for VALUE results, boolean for TRUE / FALSE
        let kind: Option<usize> = match (&e.status, &e.value) {
            (Status::Value, Some(TypedValue::Date(_))) => Some(0),
            (Status::Value, Some(TypedValue::Number(_))) => Some(1),
            (Status::Value, Some(TypedValue::Text(_))) => Some(2),
            (Status::Value, Some(TypedValue::Boolean(_))) => Some(3),
            (Status::True, _) | (Status::False, _) => Some(3),
            _ => None,
        };
        if let Some(kd) = kind {
            f[5 + kd] = 1.0;
        }
        f[9 + slot] = 1.0;
        f[13] = (e.step as f64 / 8.0) as f32;
        match kind {
            Some(0) => {
                if let Some(TypedValue::Date(s)) = &e.value {
                    let y: i32 = s.get(0..4).and_then(|v| v.parse().ok()).unwrap_or(1900);
                    let m: u32 = s.get(5..7).and_then(|v| v.parse().ok()).unwrap_or(1);
                    let dd: u32 = s.get(8..10).and_then(|v| v.parse().ok()).unwrap_or(1);
                    f[14] = ((y - 1900) as f64 / 200.0) as f32;
                    f[15] = (day_of_year(y, m, dd) as f64 / 366.0) as f32;
                }
            }
            Some(1) => {
                if let Some(TypedValue::Number(s)) = &e.value {
                    if let Some((x, adj1, neg)) = decimal_parts(s) {
                        let mag = if x.is_finite() {
                            (x.abs() + 1.0).log10()
                        } else {
                            adj1
                        };
                        let sign = if x.is_finite() {
                            x
                        } else if neg {
                            -1.0
                        } else {
                            1.0
                        };
                        f[14] = ((mag.min(60.0) / 6.0).copysign(sign)) as f32;
                    }
                }
            }
            Some(3) => {
                let truth = e.status == Status::True
                    || (e.status == Status::Value && e.value == Some(TypedValue::Boolean(true)));
                f[14] = if truth { 1.0 } else { -1.0 };
            }
            _ => {}
        }
        mask[slot] = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registers_basic() {
        let st = vec![
            StackEntry {
                status: Status::Value,
                value: Some(TypedValue::Date("2020-03-01".into())),
                step: 0,
            },
            StackEntry {
                status: Status::Value,
                value: Some(TypedValue::Number("25E-1".into())),
                step: 1,
            },
            StackEntry {
                status: Status::True,
                value: None,
                step: 2,
            },
        ];
        let mut f = [0f32; 64];
        let mut m = [false; 4];
        register_features(&st, &mut f, &mut m);
        assert_eq!(m, [true, true, true, false]);
        assert_eq!(f[0], 1.0);
        assert_eq!(f[5], 1.0);
        assert_eq!(f[9], 1.0);
        assert!((f[14] - 0.6).abs() < 1e-7);
        assert!((f[15] - (61.0 / 366.0) as f32).abs() < 1e-7);
        assert!((f[16 + 14] - ((3.5f64).log10() / 6.0) as f32).abs() < 1e-7);
        assert_eq!(f[32 + 14], 1.0);
        assert_eq!(f[32 + 8], 1.0);
    }
}
