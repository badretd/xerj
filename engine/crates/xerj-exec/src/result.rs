//! reference_core.Result / Executor.compare / Executor.negate and xr_codecs.result_text over store-borrowed values.

use smallvec::SmallVec;
use std::borrow::Cow;
use std::cmp::Ordering;
use xerj_store::value::{cmp_canonical_decimal, value_text};
use xerj_store::{Kind, Lookup, RecId, Status, Store, TypedValue};

/// reference_core.Result. `refs` are fact ids as record ids, kept sorted by fact-id string and unique (every reference
/// path produces `tuple(sorted(set(...)))` or a lookup's snapshot-ordered tuple).
#[derive(Clone, Debug, PartialEq)]
pub struct Res<'s> {
    pub status: Status,
    pub value: Option<TypedValue<'s>>,
    pub refs: SmallVec<[RecId; 4]>,
    pub reason: Cow<'static, str>,
}

impl<'s> Res<'s> {
    pub fn unknown(refs: SmallVec<[RecId; 4]>, reason: impl Into<Cow<'static, str>>) -> Res<'s> {
        Res {
            status: Status::Unknown,
            value: None,
            refs,
            reason: reason.into(),
        }
    }
    pub fn conflict(refs: SmallVec<[RecId; 4]>, reason: impl Into<Cow<'static, str>>) -> Res<'s> {
        Res {
            status: Status::Conflict,
            value: None,
            refs,
            reason: reason.into(),
        }
    }
    pub fn from_lookup(l: Lookup<'s>) -> Res<'s> {
        Res {
            status: l.status,
            value: l.value,
            refs: l.refs,
            reason: Cow::Borrowed(l.reason),
        }
    }
}

/// sorted(set(a) | set(b)) by fact-id string
pub fn merge_refs(store: &Store, a: &[RecId], b: &[RecId]) -> SmallVec<[RecId; 4]> {
    let mut v: SmallVec<[RecId; 4]> = SmallVec::with_capacity(a.len() + b.len());
    v.extend_from_slice(a);
    v.extend_from_slice(b);
    let p = &store.pools.fact_ids;
    v.sort_unstable_by(|&x, &y| p.get(x).cmp(p.get(y)));
    v.dedup();
    v
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CmpOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

/// order of two typed values of the same (kind, unit) for number / date
pub fn order_values(a: &TypedValue<'_>, b: &TypedValue<'_>) -> Ordering {
    match a.kind {
        Kind::Number => cmp_canonical_decimal(a.value, b.value),
        _ => a.value.cmp(b.value),
    }
}

/// reference_core.Executor.compare
pub fn compare<'s>(store: &Store, a: &Res<'s>, b: &Res<'s>, op: CmpOp) -> Res<'s> {
    let refs = merge_refs(store, &a.refs, &b.refs);
    if a.status == Status::Conflict || b.status == Status::Conflict {
        return Res::conflict(refs, "Conflicting operand");
    }
    if a.status != Status::Value || b.status != Status::Value {
        return Res::unknown(refs, "Missing operand");
    }
    let (va, vb) = (
        a.value.expect("VALUE carries a value"),
        b.value.expect("VALUE carries a value"),
    );
    if (va.kind, va.unit) != (vb.kind, vb.unit) {
        return Res::unknown(refs, "Incomparable types or units");
    }
    if !matches!(op, CmpOp::Eq | CmpOp::Ne) && !matches!(va.kind, Kind::Number | Kind::Date) {
        return Res::unknown(refs, "Ordering needs numeric or date values");
    }
    let truth = match op {
        CmpOp::Eq | CmpOp::Ne => {
            let eq = match va.kind {
                Kind::Number => cmp_canonical_decimal(va.value, vb.value) == Ordering::Equal,
                _ => va.value == vb.value,
            };
            eq == (op == CmpOp::Eq)
        }
        CmpOp::Lt => order_values(&va, &vb) == Ordering::Less,
        CmpOp::Le => order_values(&va, &vb) != Ordering::Greater,
        CmpOp::Gt => order_values(&va, &vb) == Ordering::Greater,
        CmpOp::Ge => order_values(&va, &vb) != Ordering::Less,
    };
    Res {
        status: if truth { Status::True } else { Status::False },
        value: None,
        refs,
        reason: Cow::Borrowed(""),
    }
}

/// reference_core.Executor.negate (None = ContractError "NOT requires a logical result")
pub fn negate<'s>(r: &Res<'s>) -> Option<Res<'s>> {
    match r.status {
        Status::Unknown | Status::Conflict => Some(r.clone()),
        Status::True => Some(Res {
            status: Status::False,
            value: None,
            refs: r.refs.clone(),
            reason: Cow::Borrowed(""),
        }),
        Status::False => Some(Res {
            status: Status::True,
            value: None,
            refs: r.refs.clone(),
            reason: Cow::Borrowed(""),
        }),
        Status::Value => None,
    }
}

/// xr_codecs.result_text
pub fn result_text(r: &Res<'_>) -> String {
    match (r.status, r.value) {
        (Status::Value, Some(v)) => value_text(&v),
        (s, _) => s.name().to_string(),
    }
}
