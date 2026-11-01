//! Label-free candidate manifests, exactly as the reference builds them:
//!
//! * `candidate_manifest` (xr_codecs.candidate_manifest): tier 1 = records of the linked subjects (snapshot order = fact-id
//!   order), tier 2 = records of other subjects with a tier-1 relation, tier 3 = the rest; tiers 2-3 ranked by
//!   `manifest_selection_key` = sha256(item_id + fingerprint + fact_id), the first K kept, then ordered by
//!   `manifest_order_key` = sha256("order|" + item_id + fingerprint + fact_id).
//! * `hop_manifest` (stack_v2.hop_manifest): tier 1 = records whose subject RESOLVES to the chained text value (entity ids
//!   when both sides carry them, else normalize_entity equality), tier 2 = other records with a tier-1 relation, tier 3 =
//!   the rest; keys sha256(salt + item_id + "|" + value_anchor + "|" + fingerprint + "|" + fact_id).
//!
//! Instead of sorting whole tiers (the reference sorts every same-relation record and, when tiers 1-2 are short, every
//! record of the snapshot), only the K - |tier 1| smallest keys are kept in a bounded max-heap; the hash prefix (salt,
//! item id, fingerprint) is absorbed once and its SHA-256 midstate cloned per record. Tier 2 / 3 candidates come from the
//! relation index, so tier 3 is only scanned when tiers 1-2 hold fewer than K records.
//!
//! `ManifestHash::Fast` keeps every rule but replaces SHA-256 by xxh3-128 (seeded with the same prefix): a different,
//! equally label-free order, ~10x cheaper per record for very large same-relation tiers.

use crate::subjects::SubjectIndex;
use smallvec::SmallVec;
use std::collections::BinaryHeap;
use xerj_store::text::{normalize_entity, py_json_str};
use xerj_store::{Kind, RecId, Snapshot, TypedValue, NONE};

pub const ORDER_SALT: &str = "order|";
pub const HOP_SELECT_SALT: &str = "stack-v2 hop select|";
pub const HOP_ORDER_SALT: &str = "stack-v2 hop order|";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ManifestHash {
    /// the reference keys (bit-identical manifests)
    Sha256,
    /// xxh3-128 keys (same tiers and rules, different order; faster on huge tiers)
    Fast,
    /// EXTENSION: tiers 2-3 as a seeded uniform sample (O(K) instead of O(tier size)) + xxh3 order keys. Hash ranking
    /// of a tier = a uniformly random subset of it; sampling draws the same distribution directly. Falls back to the
    /// full Fast scan when eligible records are too sparse to sample. Deterministic per (item id, fingerprint).
    Sampled,
}

pub type Key = [u8; 32];

/// A hash key function with its prefix absorbed once.
#[derive(Clone)]
pub enum KeyFn {
    /// SHA-256 midstate of the prefix (4-lane ARMv8 SHA-256 when available)
    Sha(crate::sha256x::PrefixSha256),
    Fast(u64, u64),
}

impl KeyFn {
    pub fn new(hash: ManifestHash, prefix: &[&str]) -> KeyFn {
        match hash {
            ManifestHash::Sha256 => {
                let parts: Vec<&[u8]> = prefix.iter().map(|p| p.as_bytes()).collect();
                KeyFn::Sha(crate::sha256x::PrefixSha256::new(&parts))
            }
            ManifestHash::Fast | ManifestHash::Sampled => {
                let mut buf = Vec::with_capacity(256);
                for p in prefix {
                    buf.extend_from_slice(p.as_bytes());
                }
                let h = xxhash_rust::xxh3::xxh3_128(&buf);
                KeyFn::Fast(h as u64, (h >> 64) as u64)
            }
        }
    }

    #[inline]
    pub fn key(&self, fact_id: &str) -> Key {
        match self {
            KeyFn::Sha(p) => p.digest(fact_id.as_bytes()),
            KeyFn::Fast(s1, s2) => fast_key(*s1, *s2, fact_id),
        }
    }

    /// keys of 8 fact ids (the x86-64 AVX2 SHA-256 path compresses them in lockstep)
    #[inline]
    pub fn key8(&self, f: [&str; 8]) -> [Key; 8] {
        match self {
            KeyFn::Sha(p) => p.digest8(core::array::from_fn(|l| f[l].as_bytes())),
            KeyFn::Fast(s1, s2) => core::array::from_fn(|l| fast_key(*s1, *s2, f[l])),
        }
    }

    #[inline]
    pub fn key4(&self, f: [&str; 4]) -> [Key; 4] {
        match self {
            KeyFn::Sha(p) => p.digest4([
                f[0].as_bytes(),
                f[1].as_bytes(),
                f[2].as_bytes(),
                f[3].as_bytes(),
            ]),
            KeyFn::Fast(s1, s2) => [
                fast_key(*s1, *s2, f[0]),
                fast_key(*s1, *s2, f[1]),
                fast_key(*s1, *s2, f[2]),
                fast_key(*s1, *s2, f[3]),
            ],
        }
    }
}

#[inline]
fn fast_key(s1: u64, s2: u64, fact_id: &str) -> Key {
    let a = xxhash_rust::xxh3::xxh3_64_with_seed(fact_id.as_bytes(), s1);
    let b = xxhash_rust::xxh3::xxh3_64_with_seed(fact_id.as_bytes(), s2);
    let mut out = [0u8; 32];
    out[..8].copy_from_slice(&a.to_be_bytes());
    out[8..16].copy_from_slice(&b.to_be_bytes());
    out
}

/// keys hashed together by [`TopM`] (the AVX2 SHA-256 width; aarch64 runs two 4-lane batches)
const LANES: usize = 8;

/// keeps the m smallest (key, record) pairs; candidates are hashed eight at a time
pub(crate) struct TopM<'s> {
    m: usize,
    heap: BinaryHeap<(Key, RecId)>,
    pub(crate) seen: usize,
    pending: [(RecId, &'s str); LANES],
    npend: usize,
}

impl<'s> TopM<'s> {
    pub(crate) fn new(m: usize) -> TopM<'s> {
        TopM {
            m,
            heap: BinaryHeap::with_capacity(m + 1),
            seen: 0,
            pending: [(0, ""); LANES],
            npend: 0,
        }
    }
    #[inline]
    fn push_key(&mut self, k: Key, r: RecId) {
        if self.heap.len() < self.m {
            self.heap.push((k, r));
        } else if let Some(mut top) = self.heap.peek_mut() {
            if k < top.0 {
                *top = (k, r);
            }
        }
    }
    #[inline]
    pub(crate) fn offer(&mut self, kf: &KeyFn, r: RecId, fact_id: &'s str) {
        self.seen += 1;
        if self.m == 0 {
            return;
        }
        self.pending[self.npend] = (r, fact_id);
        self.npend += 1;
        if self.npend == LANES {
            let p = self.pending;
            let ks = kf.key8(core::array::from_fn(|l| p[l].1));
            for (k, (r, _)) in ks.into_iter().zip(p) {
                self.push_key(k, r);
            }
            self.npend = 0;
        }
    }
    /// the kept (key, record) pairs, unsorted (pending records hashed first); [`crate::direction`]'s threaded scans merge
    /// these per range
    pub(crate) fn into_keyed(mut self, kf: &KeyFn) -> Vec<(Key, RecId)> {
        for i in 0..self.npend {
            let (r, f) = self.pending[i];
            let k = kf.key(f);
            self.push_key(k, r);
        }
        self.heap.into_vec()
    }
    pub(crate) fn into_sorted(mut self, kf: &KeyFn) -> Vec<RecId> {
        for i in 0..self.npend {
            let (r, f) = self.pending[i];
            let k = kf.key(f);
            self.push_key(k, r);
        }
        let mut v = self.heap.into_vec();
        v.sort_unstable_by_key(|a| a.0);
        v.into_iter().map(|x| x.1).collect()
    }
}

/// Sort `chosen` by the order key.
pub(crate) fn order_by(snap: &Snapshot<'_>, kf: &KeyFn, chosen: &mut [RecId]) {
    let store = snap.store;
    let mut keyed: SmallVec<[(Key, RecId); 32]> = SmallVec::with_capacity(chosen.len());
    let mut i = 0;
    while i + 4 <= chosen.len() {
        let c = &chosen[i..i + 4];
        let ks = kf.key4([
            store.fact_id(c[0]),
            store.fact_id(c[1]),
            store.fact_id(c[2]),
            store.fact_id(c[3]),
        ]);
        for (k, &r) in ks.into_iter().zip(c) {
            keyed.push((k, r));
        }
        i += 4;
    }
    for &r in &chosen[i..] {
        keyed.push((kf.key(store.fact_id(r)), r));
    }
    keyed.sort_unstable_by_key(|a| a.0);
    for (i, (_, r)) in keyed.into_iter().enumerate() {
        chosen[i] = r;
    }
}

/// Statistics of one manifest build (for profiling / benchmarks).
#[derive(Clone, Copy, Debug, Default)]
pub struct ManifestStats {
    pub first: usize,
    pub second_eligible: usize,
    pub rest_scanned: usize,
    pub keys_hashed: usize,
}

/// Seeded uniform sample (without replacement) of up to `m` eligible records from the concatenation of `lists`;
/// None when eligible records are too sparse (the caller falls back to an exact scan).
pub(crate) fn sample_lists(
    lists: &[&[RecId]],
    m: usize,
    seed: u64,
    eligible: impl Fn(RecId) -> bool,
) -> Option<Vec<RecId>> {
    let total: usize = lists.iter().map(|l| l.len()).sum();
    if m == 0 {
        return Some(Vec::new());
    }
    if total < 64 * m {
        return None;
    }
    let mut x = seed | 1;
    let mut next = || {
        // splitmix64
        x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = x;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    let mut picked: Vec<RecId> = Vec::with_capacity(m);
    let mut draws = 0usize;
    while picked.len() < m {
        draws += 1;
        if draws > 32 * m + 64 {
            return None;
        }
        let mut i = (((next() >> 11) as u128 * total as u128) >> 53) as usize;
        let mut li = 0;
        while i >= lists[li].len() {
            i -= lists[li].len();
            li += 1;
        }
        let r = lists[li][i];
        if eligible(r) && !picked.contains(&r) {
            picked.push(r);
        }
    }
    Some(picked)
}

/// xr_codecs.candidate_manifest given the linked subjects `ents` (subject syms): K record ids in manifest order.
#[allow(clippy::too_many_arguments)]
pub fn candidate_manifest(
    snap: &Snapshot<'_>,
    idx: &SubjectIndex,
    ents: &[u32],
    item_id: &str,
    fingerprint: &str,
    k: usize,
    hash: ManifestHash,
    stats: Option<&mut ManifestStats>,
) -> Vec<RecId> {
    let store = snap.store;
    let fid = |r: RecId| store.fact_id(r);
    // tier 1: records of the linked subjects, snapshot (fact id) order
    let mut first: Vec<RecId> = Vec::new();
    for &s in ents {
        for &r in idx.records_of_subject(s) {
            if snap.visible(r) {
                first.push(r);
            }
        }
    }
    first.sort_unstable_by(|&a, &b| fid(a).cmp(fid(b)));
    first.dedup();
    let mut st = ManifestStats {
        first: first.len(),
        ..Default::default()
    };
    let order = KeyFn::new(hash, &[ORDER_SALT, item_id, fingerprint]);
    if first.len() >= k {
        first.truncate(k);
        order_by(snap, &order, &mut first);
        st.keys_hashed = k;
        if let Some(s) = stats {
            *s = st;
        }
        return first;
    }
    let sel = KeyFn::new(hash, &[item_id, fingerprint]);
    let mut rels: SmallVec<[u32; 8]> = SmallVec::new();
    for &r in &first {
        let rl = store.recs.relation[r as usize];
        if !rels.contains(&rl) {
            rels.push(rl);
        }
    }
    let m = k - first.len();
    if hash == ManifestHash::Sampled {
        let seed = xxhash_rust::xxh3::xxh3_64(format!("sample|{item_id}|{fingerprint}").as_bytes());
        let lists2: Vec<&[RecId]> = rels.iter().map(|&rl| idx.records_of_relation(rl)).collect();
        let elig2 = |r: RecId| snap.visible(r) && !ents.contains(&store.recs.subject[r as usize]);
        if let Some(mut picked) = sample_lists(&lists2, m, seed, elig2) {
            st.keys_hashed = picked.len();
            let mut chosen = first;
            chosen.append(&mut picked);
            order_by(snap, &order, &mut chosen);
            if let Some(s) = stats {
                *s = st;
            }
            return chosen;
        }
    }
    // tier 2: other subjects, same relations
    let mut second = TopM::new(m);
    for &rl in &rels {
        for &r in idx.records_of_relation(rl) {
            if snap.visible(r) && !ents.contains(&store.recs.subject[r as usize]) {
                second.offer(&sel, r, fid(r));
            }
        }
    }
    st.second_eligible = second.seen;
    let mut chosen = first;
    let need_rest = m.saturating_sub(second.seen);
    st.keys_hashed = second.seen;
    chosen.extend(second.into_sorted(&sel));
    if need_rest > 0 {
        // tier 3: every other visible record (relation outside tier 1's)
        let mut rest = TopM::new(need_rest);
        for rl in 0..idx.by_relation.len() as u32 {
            if rels.contains(&rl) {
                continue;
            }
            for &r in idx.records_of_relation(rl) {
                if snap.visible(r) {
                    rest.offer(&sel, r, fid(r));
                }
            }
        }
        st.rest_scanned = rest.seen;
        st.keys_hashed += rest.seen;
        chosen.extend(rest.into_sorted(&sel));
    }
    chosen.truncate(k);
    st.keys_hashed += chosen.len();
    order_by(snap, &order, &mut chosen);
    if let Some(s) = stats {
        *s = st;
    }
    chosen
}

/// stack_v2.HopManifest (+ stack-v5 inverse flags)
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct HopManifest {
    pub cands: Vec<RecId>,
    pub resolves: Vec<bool>,
    pub anchor: Option<String>,
    /// stack-v5 (`direction::hop_manifest_v5`): per slot, true = an INVERSE candidate (stack_v5.InverseCand: followed from
    /// its value back to its subject; rendered "[k] <= ..."). Empty for stack-v2 manifests (no inverse candidates).
    pub inverse: Vec<bool>,
}

impl HopManifest {
    pub fn is_empty(&self) -> bool {
        self.cands.is_empty()
    }
    pub fn len(&self) -> usize {
        self.cands.len()
    }
    /// is slot `j` an inverse (stack-v5) candidate? (false for stack-v2 manifests and out-of-range slots)
    #[inline]
    pub fn is_inverse(&self, j: usize) -> bool {
        self.inverse.get(j).copied().unwrap_or(false)
    }
}

/// stack_v2.value_anchor: json.dumps({"kind", "value", "unit", "ids": sorted(ids)}, sort_keys=True, ensure_ascii=False)
/// (default separators ", " and ": ")
pub fn value_anchor(v: &TypedValue<'_>, value_ids: &[&str]) -> String {
    let mut ids: SmallVec<[&str; 4]> = value_ids.iter().copied().collect();
    ids.sort_unstable();
    ids.dedup();
    let mut b: Vec<u8> = Vec::with_capacity(64 + v.value.len());
    b.extend_from_slice(b"{\"ids\": [");
    for (i, id) in ids.iter().enumerate() {
        if i > 0 {
            b.extend_from_slice(b", ");
        }
        py_json_str(id, &mut b);
    }
    b.extend_from_slice(b"], \"kind\": ");
    py_json_str(v.kind.as_str(), &mut b);
    b.extend_from_slice(b", \"unit\": ");
    py_json_str(v.unit, &mut b);
    b.extend_from_slice(b", \"value\": ");
    if v.kind == Kind::Boolean {
        b.extend_from_slice(v.value.as_bytes());
    } else {
        py_json_str(v.value, &mut b);
    }
    b.push(b'}');
    String::from_utf8(b).expect("utf-8")
}

/// stack_v2.resolves for one record (the rule; `value_ids` sorted or not)
pub fn resolves(
    snap: &Snapshot<'_>,
    idx: &SubjectIndex,
    v: &TypedValue<'_>,
    value_ids: &[&str],
    r: RecId,
) -> bool {
    if v.kind != Kind::Text {
        return false;
    }
    let store = snap.store;
    if let Some(sid) = store.subject_id(r) {
        if !value_ids.is_empty() {
            return value_ids.contains(&sid);
        }
    }
    let _ = idx;
    normalize_entity(v.value) == normalize_entity(store.subject(r))
}

/// the resolving records of a chained text value (tier 1 of a hop manifest), snapshot order
pub fn resolving_records(
    snap: &Snapshot<'_>,
    idx: &SubjectIndex,
    v: &TypedValue<'_>,
    value_ids: &[&str],
) -> Vec<RecId> {
    let store = snap.store;
    let mut out: Vec<RecId> = Vec::new();
    if v.kind != Kind::Text {
        return out;
    }
    let nv = normalize_entity(v.value);
    let mut subs: SmallVec<[u32; 4]> = SmallVec::new();
    idx.subjects_with_norm(store, &nv, &mut subs);
    if !value_ids.is_empty() {
        for vid in value_ids {
            if let Some(e) = store.pools.ent_ids.lookup(vid) {
                for &r in idx.records_of_subject_id(e) {
                    if snap.visible(r) {
                        out.push(r);
                    }
                }
            }
        }
        for &s in &subs {
            for &r in idx.records_of_subject(s) {
                if store.recs.subject_id[r as usize] == NONE && snap.visible(r) {
                    out.push(r);
                }
            }
        }
    } else {
        for &s in &subs {
            for &r in idx.records_of_subject(s) {
                if snap.visible(r) {
                    out.push(r);
                }
            }
        }
    }
    out.sort_unstable_by(|&a, &b| store.fact_id(a).cmp(store.fact_id(b)));
    out.dedup();
    out
}

/// stack_v2.hop_manifest(item_id, snapshot.facts, fingerprint, value, value_ids, K, entity_ids)
#[allow(clippy::too_many_arguments)]
pub fn hop_manifest(
    snap: &Snapshot<'_>,
    idx: &SubjectIndex,
    item_id: &str,
    fingerprint: &str,
    v: &TypedValue<'_>,
    value_ids: &[&str],
    k: usize,
    hash: ManifestHash,
) -> HopManifest {
    let store = snap.store;
    let first = resolving_records(snap, idx, v, value_ids);
    if first.is_empty() {
        return HopManifest::default();
    }
    let fid = |r: RecId| store.fact_id(r);
    let anchor = value_anchor(v, value_ids);
    let sel = KeyFn::new(
        hash,
        &[
            HOP_SELECT_SALT,
            item_id,
            "|",
            &anchor,
            "|",
            fingerprint,
            "|",
        ],
    );
    let order = KeyFn::new(
        hash,
        &[HOP_ORDER_SALT, item_id, "|", &anchor, "|", fingerprint, "|"],
    );
    let mut first_sorted_ids: Vec<RecId> = first.clone();
    first_sorted_ids.sort_unstable();
    let in_first = |r: RecId| first_sorted_ids.binary_search(&r).is_ok();
    let mut rels: SmallVec<[u32; 8]> = SmallVec::new();
    for &r in &first {
        let rl = store.recs.relation[r as usize];
        if !rels.contains(&rl) {
            rels.push(rl);
        }
    }
    let mut chosen: Vec<RecId>;
    let sampled = if hash == ManifestHash::Sampled && first.len() < k {
        let seed = xxhash_rust::xxh3::xxh3_64(
            format!("hop sample|{item_id}|{anchor}|{fingerprint}").as_bytes(),
        );
        let lists2: Vec<&[RecId]> = rels.iter().map(|&rl| idx.records_of_relation(rl)).collect();
        sample_lists(&lists2, k - first.len(), seed, |r| {
            snap.visible(r) && !in_first(r)
        })
    } else {
        None
    };
    if first.len() >= k {
        chosen = first[..k].to_vec();
    } else if let Some(mut picked) = sampled {
        chosen = first.clone();
        chosen.append(&mut picked);
    } else {
        let m = k - first.len();
        let mut second = TopM::new(m);
        for &rl in &rels {
            for &r in idx.records_of_relation(rl) {
                if snap.visible(r) && !in_first(r) {
                    second.offer(&sel, r, fid(r));
                }
            }
        }
        let need_rest = m.saturating_sub(second.seen);
        chosen = first.clone();
        chosen.extend(second.into_sorted(&sel));
        if need_rest > 0 {
            let mut rest = TopM::new(need_rest);
            for rl in 0..idx.by_relation.len() as u32 {
                if rels.contains(&rl) {
                    continue;
                }
                for &r in idx.records_of_relation(rl) {
                    if snap.visible(r) && !in_first(r) {
                        rest.offer(&sel, r, fid(r));
                    }
                }
            }
            chosen.extend(rest.into_sorted(&sel));
        }
        chosen.truncate(k);
    }
    order_by(snap, &order, &mut chosen);
    let resolves = chosen.iter().map(|&r| in_first(r)).collect();
    HopManifest {
        cands: chosen,
        resolves,
        anchor: Some(anchor),
        inverse: Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn anchor_format() {
        let v = TypedValue {
            kind: Kind::Text,
            value: "New York City",
            unit: "",
        };
        assert_eq!(
            value_anchor(&v, &["Q60"]),
            r#"{"ids": ["Q60"], "kind": "text", "unit": "", "value": "New York City"}"#
        );
        assert_eq!(
            value_anchor(&v, &[]),
            r#"{"ids": [], "kind": "text", "unit": "", "value": "New York City"}"#
        );
        assert_eq!(
            value_anchor(&v, &["Q2", "Q1"]),
            r#"{"ids": ["Q1", "Q2"], "kind": "text", "unit": "", "value": "New York City"}"#
        );
    }

    #[test]
    fn sha_keys_match_python() {
        use sha2::{Digest, Sha256};
        // hashlib.sha256(("order|" + "it" + "fp" + "f1").encode()).hexdigest()
        let kf = KeyFn::new(ManifestHash::Sha256, &[ORDER_SALT, "it", "fp"]);
        let k = kf.key("f1");
        let expect: [u8; 32] = Sha256::digest(b"order|itfpf1").into();
        assert_eq!(k, expect);
        let k4 = kf.key4(["f1", "f2", "a-much-longer-fact-id/with@v1/relation", "f1"]);
        assert_eq!(k4[0], expect);
        assert_eq!(
            k4[2],
            <[u8; 32]>::from(Sha256::digest(
                b"order|itfpa-much-longer-fact-id/with@v1/relation"
            ))
        );
    }
}
