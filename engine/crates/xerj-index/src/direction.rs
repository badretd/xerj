//! Stack-v5 TRAVERSAL DIRECTION (dev_v5 `stack_v5.py`, 2026-10-23): INVERSE candidates (records followed from their VALUE
//! back to their SUBJECT) in the base and hop manifests, the label-free value linker, and the value-side lookups the exact
//! executor's inverse checks read. Everything here is additive: the stack-v1 / v2 paths never call it.
//!
//! * [`link_values`] (`stack_v5.link_values`): snapshot TEXT values (`str.strip()` non-empty) that occur in the question as
//!   whole words, longest first, matched spans masked, returned in question order: the value-side twin of
//!   `xr_codecs.link_entities`, run with the same span-hash occurrence finder and masking simulation
//!   (`linker::link_simulate_with`) over the text-value vocabulary.
//! * [`candidate_manifest_v5`] (`stack_v5.candidate_manifest`): tier 1 = records of the linked subjects (snapshot order),
//!   tier 1-IN = text records whose value is a linked value and whose subject is not linked, at most `max_inverse` of them
//!   ranked by sha256("stack-v5 inverse select|" + item_id + "||" + fingerprint + "|" + fact_id); tier 2 = records of other
//!   subjects with a relation of tier 1 or 1-IN (minus 1-IN), tier 3 = the rest, both by the v2 selection key; K kept, ordered
//!   by the v2 order key. Inverse slots are flagged ([`CandManifest::inverse`]).
//! * [`hop_manifest_v5`] (`stack_v5.hop_manifest`): tier 1 = the records whose SUBJECT resolves to the chained text value
//!   (stack-v2 rule) + the records whose VALUE resolves to it ([`value_resolving_records`]: the record's value_id among the
//!   chained value's ids when both carry ids, else normalize_entity equality; text records; not forward), at most
//!   `max_inverse` by sha256("stack-v5 inverse select|" + item_id + "|" + anchor + "|" + fingerprint + "|" + fact_id); no hop
//!   manifest when both are empty; tiers 2-3 and the salted order as stack-v2; `resolves` is true for forward AND inverse
//!   tier-1 records, `inverse` flags the inverse slots.
//! * [`same_value_records`] (the set check of `stack_v5.inverse_result`): the active text records with the record's relation
//!   and scope whose value denotes the same value (value ids when both carry one, else normalized strings).
//!
//! A record appears in a manifest once; forward takes precedence over inverse; records of the inverse pool beyond the cap
//! stay eligible for tiers 2-3 as plain (forward) records, exactly as the reference.
//!
//! The value-side tables ([`ValueTables`]: text value -> records, value_id -> text records, normalized value -> values, and
//! the shape of the linkable vocabulary) are built lazily on the first stack-v5 read and caught up incrementally (O(new
//! records)) behind a `RwLock`, so v1 / v2 workloads pay nothing at ingest.
//!
//! MAX_INVERSE: [`max_inverse`] reads `XERJ_R_MAX_INVERSE` once (default 8, the value every v5 checkpoint was trained with;
//! 0 = no inverse candidates, i.e. the stack-v2 manifests). The cap is applied like Python's `[:n]` (a negative n drops
//! records from the end of the ranking).
//!
//! EXTENSION (opt-in, never on the reference path): [`RelPref`], a question-aware choice of the records a manifest keeps
//! when a cap binds ([`candidate_manifest_v5_pref`], [`hop_manifest_v5_pref`]). See [`RelPref`]; with `None` both are the
//! reference functions above, and with `Some` the manifest differs from the reference only where a cap binds.
//!
//! EXTENSION (opt-in, exact): [`manifest_threads`] (`XERJ_R_MANIFEST_THREADS`, default 1) splits the tier-2 / tier-3 scans
//! of the stack-v5 base and hop manifests over scoped threads. Those scans key every record of the relations the tier-1
//! records use (every other record for tier 3), so their cost grows with the store (about 0.1 s median and 0.6-0.9 s at the
//! 95th percentile per base manifest at 10 million facts on one x86 core). The kept records are the same: see
//! [`scan_top_m`].

use crate::linker::{link_simulate_with, span_hash_occurrences, QBuf, SubjectShape};
use crate::manifest::{
    order_by, resolving_records, sample_lists, value_anchor, HopManifest, KeyFn, ManifestHash,
    TopM, HOP_ORDER_SALT, HOP_SELECT_SALT, ORDER_SALT,
};
use crate::subjects::SubjectIndex;
use foldhash::fast::FixedState;
use hashbrown::HashMap;
use smallvec::SmallVec;
use std::sync::{OnceLock, RwLock, RwLockReadGuard};
use xerj_store::text::{is_py_space, is_py_word, normalize_entity};
use xerj_store::{Kind, RecId, Snapshot, Store, StrPool, TypedValue, NONE};

/// stack_v5.INV_SELECT_SALT
pub const INV_SELECT_SALT: &str = "stack-v5 inverse select|";
/// stack_v5.MAX_INVERSE without XERJ_R_MAX_INVERSE
pub const DEFAULT_MAX_INVERSE: i64 = 8;

/// `int(os.environ.get("XERJ_R_MAX_INVERSE", "8"))` (read once per process; an unparsable value falls back to 8 where
/// Python would fail at import)
pub fn max_inverse() -> i64 {
    static V: OnceLock<i64> = OnceLock::new();
    *V.get_or_init(|| parse_max_inverse(std::env::var("XERJ_R_MAX_INVERSE").ok().as_deref()))
}

/// the cap a XERJ_R_MAX_INVERSE value denotes (None = unset = 8)
pub fn parse_max_inverse(v: Option<&str>) -> i64 {
    match v {
        None => DEFAULT_MAX_INVERSE,
        Some(s) => s.trim().parse::<i64>().unwrap_or(DEFAULT_MAX_INVERSE),
    }
}

/// Default of `XERJ_R_MANIFEST_PAR_MIN`: a tier-2 / tier-3 scan over fewer records stays on the calling thread.
pub const DEFAULT_MANIFEST_PAR_MIN: usize = 65_536;

/// EXTENSION (opt-in, exact): `(threads, min records)` for the tier-2 / tier-3 scans of the stack-v5 manifests, read once
/// per process like `XERJ_R_MAX_INVERSE`: `XERJ_R_MANIFEST_THREADS` (unset, unparsable, 0 or 1 = the reference loop on the
/// calling thread) and `XERJ_R_MANIFEST_PAR_MIN` (default [`DEFAULT_MANIFEST_PAR_MIN`]; a scan over fewer records runs on
/// the calling thread).
pub fn manifest_threads() -> (usize, usize) {
    static V: OnceLock<(usize, usize)> = OnceLock::new();
    *V.get_or_init(|| {
        parse_manifest_threads(
            std::env::var("XERJ_R_MANIFEST_THREADS").ok().as_deref(),
            std::env::var("XERJ_R_MANIFEST_PAR_MIN").ok().as_deref(),
        )
    })
}

/// the `(threads, min records)` that `XERJ_R_MANIFEST_THREADS` / `XERJ_R_MANIFEST_PAR_MIN` values denote
pub fn parse_manifest_threads(threads: Option<&str>, par_min: Option<&str>) -> (usize, usize) {
    let t = threads
        .and_then(|s| s.trim().parse::<usize>().ok())
        .unwrap_or(1)
        .clamp(1, 64);
    let m = par_min
        .and_then(|s| s.trim().parse::<usize>().ok())
        .unwrap_or(DEFAULT_MANIFEST_PAR_MIN);
    (t, m)
}

/// number of elements Python's `seq[:n]` keeps from a sequence of length `len`
#[inline]
pub fn py_take(len: usize, n: i64) -> usize {
    if n >= 0 {
        len.min(n as usize)
    } else {
        len.saturating_sub(n.unsigned_abs() as usize)
    }
}

// ================================================================================================ value tables

/// Value-side record tables over a `Store` (every record ever written; readers filter visibility).
#[derive(Default)]
pub struct ValueTables {
    /// text value sym (the store's `strs` pool) -> text records with exactly that value (write order)
    by_value: HashMap<u32, SmallVec<[RecId; 2]>, FixedState>,
    /// value_id sym (the store's `ent_ids` pool) -> text records carrying it
    by_value_id: HashMap<u32, SmallVec<[RecId; 2]>, FixedState>,
    /// normalize_entity of the distinct text values (pool id = norm id)
    norm_names: StrPool,
    /// norm id -> text value syms
    norm_values: Vec<SmallVec<[u32; 2]>>,
    /// vocabulary shape of the LINKABLE text values (non-blank), for the span-hash finder and the masking simulation
    pub shape: SubjectShape,
    indexed: usize,
}

fn shape_add(sh: &mut SubjectShape, sym: u32, t: &str) {
    let (Some(f), Some(l)) = (t.chars().next(), t.chars().next_back()) else {
        return;
    };
    let len = t.chars().count() as u32;
    sh.max_len = sh.max_len.max(len);
    sh.first_any.insert(f);
    sh.last_any.insert(l);
    if f == ' ' || l == ' ' {
        sh.odd.push(sym);
        return;
    }
    if !is_py_word(f) {
        sh.first_nonword.insert(f);
    }
    if !is_py_word(l) {
        sh.last_nonword.insert(l);
    }
}

impl ValueTables {
    fn catch_up(&mut self, store: &Store) {
        let n = store.num_records();
        if self.indexed > n {
            *self = ValueTables::default();
        }
        for i in self.indexed..n {
            if store.recs.kind[i] != Kind::Text {
                continue;
            }
            let r = i as RecId;
            let v = store.recs.value[i];
            let e = self.by_value.entry(v).or_default();
            let new = e.is_empty();
            e.push(r);
            if new {
                let t = store.pools.strs.get(v);
                let nid = self.norm_names.intern(&normalize_entity(t));
                if nid as usize == self.norm_values.len() {
                    self.norm_values.push(SmallVec::new());
                }
                self.norm_values[nid as usize].push(v);
                if !t.chars().all(is_py_space) {
                    shape_add(&mut self.shape, v, t);
                }
            }
            let vid = store.recs.value_id[i];
            if vid != NONE {
                self.by_value_id.entry(vid).or_default().push(r);
            }
        }
        self.indexed = n;
        self.shape.seen = self.by_value.len();
    }

    /// records indexed so far
    pub fn indexed(&self) -> usize {
        self.indexed
    }

    /// text records whose value is the `strs` sym `value` (any state)
    #[inline]
    pub fn records_of_value(&self, value: u32) -> &[RecId] {
        self.by_value.get(&value).map_or(&[], |v| v.as_slice())
    }

    /// text records whose value_id is the `ent_ids` sym `ent` (any state)
    #[inline]
    pub fn records_of_value_id(&self, ent: u32) -> &[RecId] {
        self.by_value_id.get(&ent).map_or(&[], |v| v.as_slice())
    }

    /// text value syms whose normalize_entity equals `normalized`
    #[inline]
    pub fn values_with_norm(&self, normalized: &str) -> &[u32] {
        match self.norm_names.lookup(normalized) {
            Some(id) => &self.norm_values[id as usize],
            None => &[],
        }
    }

    /// is `sym` a text value of some record whose `str.strip()` is non-empty (link_values' vocabulary, any state)?
    #[inline]
    pub fn is_linkable(&self, pool: &StrPool, sym: u32) -> bool {
        self.by_value.contains_key(&sym) && !pool.get(sym).chars().all(is_py_space)
    }

    /// is `sym` the value of a text record visible in `snap`?
    #[inline]
    pub fn value_active(&self, snap: &Snapshot<'_>, sym: u32) -> bool {
        self.records_of_value(sym).iter().any(|&r| snap.visible(r))
    }

    pub fn heap_bytes(&self) -> usize {
        let e = std::mem::size_of::<(u32, SmallVec<[RecId; 2]>)>() + 1;
        let spill = |m: &HashMap<u32, SmallVec<[RecId; 2]>, FixedState>| {
            m.values()
                .filter(|v| v.spilled())
                .map(|v| v.capacity() * 4)
                .sum::<usize>()
        };
        self.by_value.capacity() * e
            + spill(&self.by_value)
            + self.by_value_id.capacity() * e
            + spill(&self.by_value_id)
            + self.norm_names.heap_bytes()
            + self.norm_values.capacity() * std::mem::size_of::<SmallVec<[u32; 2]>>()
            + self.shape.odd.capacity() * 4
    }
}

/// Lazily built, incrementally caught-up [`ValueTables`] (held by `Memory`).
#[derive(Default)]
pub struct ValueIndex {
    inner: RwLock<ValueTables>,
}

impl ValueIndex {
    pub fn new() -> Self {
        Self::default()
    }

    /// the tables, caught up with every record of `store` (built on first use)
    pub fn tables(&self, store: &Store) -> RwLockReadGuard<'_, ValueTables> {
        loop {
            {
                let g = self.inner.read().unwrap();
                if g.indexed == store.num_records() {
                    return g;
                }
            }
            self.inner.write().unwrap().catch_up(store);
        }
    }

    pub fn heap_bytes(&self) -> usize {
        self.inner.read().unwrap().heap_bytes()
    }
}

// ================================================================================================ value linker

/// `stack_v5.link_values(question, snapshot)` as `strs` syms of the linked values, in question order.
pub fn link_values(
    snap: &Snapshot<'_>,
    t: &ValueTables,
    q: &str,
    b: &mut QBuf,
) -> SmallVec<[u32; 8]> {
    let pool = &snap.store.pools.strs;
    b.load(q);
    span_hash_occurrences(pool, &t.shape, q, b);
    b.occs.retain(|o| t.is_linkable(pool, o.subject));
    link_simulate_with(pool, &t.shape.odd, |s| t.value_active(snap, s), b)
}

// ================================================================================================ relation-aware caps (opt-in)

/// Function words that never decide which relation a question asks for.
const STOP: &[&str] = &[
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "by", "can", "could", "did", "do",
    "does", "for", "from", "had", "has", "have", "he", "her", "his", "how", "i", "if", "in",
    "into", "is", "it", "its", "of", "on", "or", "she", "that", "the", "their", "them", "there",
    "these", "they", "this", "those", "to", "was", "were", "what", "when", "where", "which", "who",
    "whom", "whose", "why", "will", "with", "would",
];

/// lowercased alphanumeric words of `s`, function words dropped, a plural `s` stripped from words longer than 3 chars
fn content_words(s: &str) -> SmallVec<[String; 8]> {
    let mut out: SmallVec<[String; 8]> = SmallVec::new();
    for w in s
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
    {
        let w = w.to_lowercase();
        if STOP.contains(&w.as_str()) {
            continue;
        }
        let w = if w.chars().count() > 3 && w.ends_with('s') && !w.ends_with("ss") {
            w[..w.len() - 1].to_string()
        } else {
            w
        };
        if !out.contains(&w) {
            out.push(w);
        }
    }
    out
}

/// two content words match: equal, or a common prefix of at least 6 chars ("citizen" / "citizenship")
fn words_match(a: &str, b: &str) -> bool {
    if a == b {
        return true;
    }
    let n = a.chars().zip(b.chars()).take_while(|(x, y)| x == y).count();
    n >= 6
}

/// EXTENSION (opt-in; the reference never builds one): question-aware choice of the records a stack-v5 manifest keeps when a
/// cap binds.
///
/// Why (E8, one shared store): the reference keeps tier 1 in fact-id order when the linked subjects (base manifest) or the
/// records resolving to the chained value (hop manifest) number more than K, and it keeps MAX_INVERSE inverse candidates by
/// a hash key. In one store per world the caps rarely bind; in a large shared store a popular entity (a country) or value
/// (a language) has many more records, and the record of the question's relation can be dropped: which records survive
/// then depends on fact-id spelling or on the hash, not on the question.
///
/// What: when a cap binds, the records are grouped by relation and the groups ordered by how many content words of the
/// question their relation phrase shares ([`RelPref::score`]; ties keep the reference order of the groups' first records);
/// the kept records are then taken round-robin over the groups (each group in the reference order), so every relation of
/// the entity is represented before a second record of any relation. When no cap binds nothing changes: the manifest is the
/// reference manifest, record for record and in the reference order.
#[derive(Clone, Debug, Default)]
pub struct RelPref {
    words: SmallVec<[String; 8]>,
}

impl RelPref {
    /// the content words of `question`
    pub fn from_question(question: &str) -> RelPref {
        RelPref {
            words: content_words(question),
        }
    }

    /// number of distinct content words of `phrase` that match a content word of the question
    pub fn score(&self, phrase: &str) -> u32 {
        content_words(phrase)
            .iter()
            .filter(|w| self.words.iter().any(|q| words_match(q, w)))
            .count() as u32
    }

    /// the `n` records of `recs` (given in the reference preference order) to keep, in keep order: relation groups by
    /// descending score, round-robin over the groups. `recs.len() <= n` keeps `recs` as given.
    pub fn keep(&self, store: &Store, recs: &[RecId], n: usize) -> Vec<RecId> {
        if recs.len() <= n {
            return recs.to_vec();
        }
        // relation groups in order of first appearance, each in the reference order
        let mut groups: Vec<(u32, Vec<RecId>)> = Vec::new();
        for &r in recs {
            let rl = store.recs.relation[r as usize];
            match groups.iter_mut().find(|g| g.0 == rl) {
                Some(g) => g.1.push(r),
                None => groups.push((rl, vec![r])),
            }
        }
        let mut scored: Vec<(u32, usize, &Vec<RecId>)> = groups
            .iter()
            .enumerate()
            .map(|(i, g)| {
                let phrase = crate::memory::rel_phrase(store.relation(g.1[0]));
                (self.score(phrase), i, &g.1)
            })
            .collect();
        // higher score first; ties: the group whose first record comes first in the reference order
        scored.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
        let mut out: Vec<RecId> = Vec::with_capacity(n);
        let mut round = 0usize;
        while out.len() < n {
            let mut any = false;
            for (_, _, g) in &scored {
                if let Some(&r) = g.get(round) {
                    any = true;
                    out.push(r);
                    if out.len() == n {
                        break;
                    }
                }
            }
            if !any {
                break;
            }
            round += 1;
        }
        out
    }
}

/// [`select_by_key`] with an optional [`RelPref`]: the `n` records to keep, in keep order (the reference: smallest keys,
/// sorted by key)
fn select_by_key_pref(
    store: &Store,
    pool: &[RecId],
    kf: &KeyFn,
    n: usize,
    pref: Option<&RelPref>,
) -> Vec<RecId> {
    match pref {
        Some(p) if pool.len() > n => {
            // the whole pool in the reference (key) order, then the question-aware keep
            let ranked = select_by_key(store, pool, kf, pool.len());
            p.keep(store, &ranked, n)
        }
        _ => select_by_key(store, pool, kf, n),
    }
}

// ================================================================================================ manifests

/// A stack-v5 base manifest: K record ids in manifest order + one inverse flag per slot.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CandManifest {
    pub cands: Vec<RecId>,
    /// per slot: true = an INVERSE candidate (stack_v5.InverseCand; text line "[k] <= subject | relation | kind: value |
    /// quote", memory role column 2 of the subject slot)
    pub inverse: Vec<bool>,
}

impl CandManifest {
    pub fn len(&self) -> usize {
        self.cands.len()
    }
    pub fn is_empty(&self) -> bool {
        self.cands.is_empty()
    }
    /// is slot `k` an inverse candidate? (false out of range)
    #[inline]
    pub fn is_inverse(&self, k: usize) -> bool {
        self.inverse.get(k).copied().unwrap_or(false)
    }
}

/// the `n` records of `pool` with the smallest keys (sorted by key)
fn select_by_key(store: &Store, pool: &[RecId], kf: &KeyFn, n: usize) -> Vec<RecId> {
    let mut top = TopM::new(n);
    for &r in pool {
        top.offer(kf, r, store.fact_id(r));
    }
    top.into_sorted(kf)
}

/// EXTENSION (opt-in, exact; [`manifest_threads`]): what a tier-2 / tier-3 loop over `lists` computes, i.e. the `m`
/// records with the smallest keys among the records `r` of `lists` with `keep(r)`, sorted by key, and how many records were
/// offered (`TopM::seen`). With `threads > 1` and at least `par_min` records in `lists`, the concatenation of `lists` is
/// cut into `threads` contiguous ranges, each range is scanned into its own [`TopM`] of `m` on a scoped thread, and the
/// per-range winners are merged by key. A key is the full 32-byte SHA-256 (or the xxh3 pair) of the record's fact id
/// under one prefix, and fact ids are unique, so the `m` smallest keys, their order and the offered count do not depend on
/// the split: the result equals the single-threaded loop's.
pub(crate) fn scan_top_m(
    store: &Store,
    kf: &KeyFn,
    lists: &[&[RecId]],
    m: usize,
    keep: &(dyn Fn(RecId) -> bool + Sync),
    threads: usize,
    par_min: usize,
) -> (Vec<RecId>, usize) {
    let total: usize = lists.iter().map(|l| l.len()).sum();
    if threads <= 1 || m == 0 || total < par_min.max(2) {
        let (mut keyed, seen) = scan_range(store, kf, lists, m, keep);
        keyed.sort_unstable_by_key(|a| a.0);
        return (keyed.into_iter().map(|x| x.1).collect(), seen);
    }
    let per = total.div_ceil(threads);
    let mut ranges: Vec<Vec<&[RecId]>> = Vec::with_capacity(threads);
    let mut cur: Vec<&[RecId]> = Vec::new();
    let mut room = per;
    for &l in lists {
        let mut rest = l;
        while !rest.is_empty() {
            let take = room.min(rest.len());
            cur.push(&rest[..take]);
            rest = &rest[take..];
            room -= take;
            if room == 0 {
                ranges.push(std::mem::take(&mut cur));
                room = per;
            }
        }
    }
    if !cur.is_empty() {
        ranges.push(cur);
    }
    let parts: Vec<(Vec<(crate::manifest::Key, RecId)>, usize)> = std::thread::scope(|s| {
        let handles: Vec<_> = ranges[1..]
            .iter()
            .map(|pieces| s.spawn(move || scan_range(store, kf, pieces, m, keep)))
            .collect();
        let mut out = Vec::with_capacity(ranges.len());
        out.push(scan_range(store, kf, &ranges[0], m, keep));
        for h in handles {
            out.push(h.join().expect("manifest scan thread panicked"));
        }
        out
    });
    let mut seen = 0;
    let mut all: Vec<(crate::manifest::Key, RecId)> = Vec::with_capacity(parts.len() * m);
    for (p, n) in parts {
        seen += n;
        all.extend(p);
    }
    all.sort_unstable_by_key(|a| a.0);
    all.truncate(m);
    (all.into_iter().map(|x| x.1).collect(), seen)
}

/// one range of [`scan_top_m`]: the kept (key, record) pairs (unsorted) and the offered count
fn scan_range(
    store: &Store,
    kf: &KeyFn,
    pieces: &[&[RecId]],
    m: usize,
    keep: &(dyn Fn(RecId) -> bool + Sync),
) -> (Vec<(crate::manifest::Key, RecId)>, usize) {
    let mut top = TopM::new(m);
    for &p in pieces {
        for &r in p {
            if keep(r) {
                top.offer(kf, r, store.fact_id(r));
            }
        }
    }
    let seen = top.seen;
    (top.into_keyed(kf), seen)
}

fn rels_of(store: &Store, recs: &[RecId], out: &mut SmallVec<[u32; 8]>) {
    for &r in recs {
        let rl = store.recs.relation[r as usize];
        if !out.contains(&rl) {
            out.push(rl);
        }
    }
}

/// `stack_v5.candidate_manifest(item, snapshot, K, entity_ids, max_inverse)` given the linked subjects `ents` (subject syms)
/// and the linked values `vals` (`strs` syms of [`link_values`]).
#[allow(clippy::too_many_arguments)]
pub fn candidate_manifest_v5(
    snap: &Snapshot<'_>,
    idx: &SubjectIndex,
    t: &ValueTables,
    ents: &[u32],
    vals: &[u32],
    item_id: &str,
    fingerprint: &str,
    k: usize,
    hash: ManifestHash,
    max_inverse: i64,
) -> CandManifest {
    candidate_manifest_v5_pref(
        snap,
        idx,
        t,
        ents,
        vals,
        item_id,
        fingerprint,
        k,
        hash,
        max_inverse,
        None,
    )
}

/// [`candidate_manifest_v5`] with an optional [`RelPref`] (EXTENSION, opt-in): `None` is the reference; `Some` changes only
/// which records are kept where a cap binds (tier 1 above K, the inverse pool above `max_inverse`).
#[allow(clippy::too_many_arguments)]
pub fn candidate_manifest_v5_pref(
    snap: &Snapshot<'_>,
    idx: &SubjectIndex,
    t: &ValueTables,
    ents: &[u32],
    vals: &[u32],
    item_id: &str,
    fingerprint: &str,
    k: usize,
    hash: ManifestHash,
    max_inverse: i64,
    pref: Option<&RelPref>,
) -> CandManifest {
    let store = snap.store;
    let fid = |r: RecId| store.fact_id(r);
    let linked = |r: RecId| ents.contains(&store.recs.subject[r as usize]);
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
    let order = KeyFn::new(hash, &[ORDER_SALT, item_id, fingerprint]);
    if first.len() >= k {
        // (first + inverse + ...)[:K] = first[:K]
        if let Some(p) = pref {
            first = p.keep(store, &first, k);
        }
        first.truncate(k);
        order_by(snap, &order, &mut first);
        let n = first.len();
        return CandManifest {
            cands: first,
            inverse: vec![false; n],
        };
    }
    // tier 1-IN: text records with a linked value whose subject is not linked, <= max_inverse by the inverse key
    let mut pool: Vec<RecId> = Vec::new();
    for &v in vals {
        for &r in t.records_of_value(v) {
            if snap.visible(r) && !linked(r) {
                pool.push(r);
            }
        }
    }
    pool.sort_unstable();
    pool.dedup();
    let invk = KeyFn::new(
        hash,
        &[INV_SELECT_SALT, item_id, "|", "", "|", fingerprint, "|"],
    );
    let inverse = select_by_key_pref(store, &pool, &invk, py_take(pool.len(), max_inverse), pref);
    let mut inv_sorted = inverse.clone();
    inv_sorted.sort_unstable();
    let in_inv = |r: RecId| inv_sorted.binary_search(&r).is_ok();
    let mut chosen: Vec<RecId> = first.clone();
    chosen.extend(inverse.iter().copied().take(k - first.len()));
    if chosen.len() < k {
        let mut rels: SmallVec<[u32; 8]> = SmallVec::new();
        rels_of(store, &first, &mut rels);
        rels_of(store, &inverse, &mut rels);
        let m = k - chosen.len();
        let sel = KeyFn::new(hash, &[item_id, fingerprint]);
        let mut sampled = None;
        if hash == ManifestHash::Sampled {
            let seed =
                xxhash_rust::xxh3::xxh3_64(format!("sample|{item_id}|{fingerprint}").as_bytes());
            let lists2: Vec<&[RecId]> =
                rels.iter().map(|&rl| idx.records_of_relation(rl)).collect();
            sampled = sample_lists(&lists2, m, seed, |r| {
                snap.visible(r) && !linked(r) && !in_inv(r)
            });
        }
        let (threads, par_min) = manifest_threads();
        if let Some(mut picked) = sampled {
            chosen.append(&mut picked);
        } else if threads > 1 {
            // EXTENSION (opt-in, exact): the same tiers 2 and 3, scanned on `threads` threads ([`scan_top_m`])
            let lists: Vec<&[RecId]> = rels.iter().map(|&rl| idx.records_of_relation(rl)).collect();
            let keep2 = |r: RecId| snap.visible(r) && !linked(r) && !in_inv(r);
            let (second, seen) = scan_top_m(store, &sel, &lists, m, &keep2, threads, par_min);
            let need_rest = m.saturating_sub(seen);
            chosen.extend(second);
            if need_rest > 0 {
                let lists: Vec<&[RecId]> = (0..idx.by_relation.len() as u32)
                    .filter(|rl| !rels.contains(rl))
                    .map(|rl| idx.records_of_relation(rl))
                    .collect();
                let keep3 = |r: RecId| snap.visible(r);
                let (rest, _) =
                    scan_top_m(store, &sel, &lists, need_rest, &keep3, threads, par_min);
                chosen.extend(rest);
            }
        } else {
            // tier 2: other subjects, a relation of tier 1 / 1-IN, not 1-IN
            let mut second = TopM::new(m);
            for &rl in &rels {
                for &r in idx.records_of_relation(rl) {
                    if snap.visible(r) && !linked(r) && !in_inv(r) {
                        second.offer(&sel, r, fid(r));
                    }
                }
            }
            let need_rest = m.saturating_sub(second.seen);
            chosen.extend(second.into_sorted(&sel));
            if need_rest > 0 {
                // tier 3: every other visible record (relation outside tiers 1 / 1-IN)
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
                chosen.extend(rest.into_sorted(&sel));
            }
        }
        chosen.truncate(k);
    }
    order_by(snap, &order, &mut chosen);
    let inverse_flags = chosen.iter().map(|&r| in_inv(r)).collect();
    CandManifest {
        cands: chosen,
        inverse: inverse_flags,
    }
}

/// `stack_v5.value_resolves` over the snapshot: the visible TEXT records whose value denotes the chained text value `v`
/// (value_id among `value_ids` when both carry ids, else normalize_entity equality), sorted by fact id.
pub fn value_resolving_records(
    snap: &Snapshot<'_>,
    t: &ValueTables,
    v: &TypedValue<'_>,
    value_ids: &[&str],
) -> Vec<RecId> {
    let store = snap.store;
    let mut out: Vec<RecId> = Vec::new();
    if v.kind != Kind::Text {
        return out;
    }
    let nv = normalize_entity(v.value);
    if !value_ids.is_empty() {
        for vid in value_ids {
            if let Some(e) = store.pools.ent_ids.lookup(vid) {
                for &r in t.records_of_value_id(e) {
                    if snap.visible(r) {
                        out.push(r);
                    }
                }
            }
        }
        for &sym in t.values_with_norm(&nv) {
            for &r in t.records_of_value(sym) {
                if store.recs.value_id[r as usize] == NONE && snap.visible(r) {
                    out.push(r);
                }
            }
        }
    } else {
        for &sym in t.values_with_norm(&nv) {
            for &r in t.records_of_value(sym) {
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

/// `stack_v5.hop_manifest(item_id, snapshot.facts, fingerprint, value, value_ids, K, entity_ids, max_inverse)`
#[allow(clippy::too_many_arguments)]
pub fn hop_manifest_v5(
    snap: &Snapshot<'_>,
    idx: &SubjectIndex,
    t: &ValueTables,
    item_id: &str,
    fingerprint: &str,
    v: &TypedValue<'_>,
    value_ids: &[&str],
    k: usize,
    hash: ManifestHash,
    max_inverse: i64,
) -> HopManifest {
    hop_manifest_v5_pref(
        snap,
        idx,
        t,
        item_id,
        fingerprint,
        v,
        value_ids,
        k,
        hash,
        max_inverse,
        None,
    )
}

/// [`hop_manifest_v5`] with an optional [`RelPref`] (EXTENSION, opt-in): `None` is the reference; `Some` changes only which
/// records are kept where a cap binds (tier 1 above K, the inverse pool above `max_inverse`).
#[allow(clippy::too_many_arguments)]
pub fn hop_manifest_v5_pref(
    snap: &Snapshot<'_>,
    idx: &SubjectIndex,
    t: &ValueTables,
    item_id: &str,
    fingerprint: &str,
    v: &TypedValue<'_>,
    value_ids: &[&str],
    k: usize,
    hash: ManifestHash,
    max_inverse: i64,
    pref: Option<&RelPref>,
) -> HopManifest {
    if v.kind != Kind::Text {
        return HopManifest::default();
    }
    let store = snap.store;
    let fid = |r: RecId| store.fact_id(r);
    let first = resolving_records(snap, idx, v, value_ids);
    let mut first_sorted: Vec<RecId> = first.clone();
    first_sorted.sort_unstable();
    let in_first = |r: RecId| first_sorted.binary_search(&r).is_ok();
    let anchor = value_anchor(v, value_ids);
    let pool: Vec<RecId> = value_resolving_records(snap, t, v, value_ids)
        .into_iter()
        .filter(|&r| !in_first(r))
        .collect();
    let invk = KeyFn::new(
        hash,
        &[
            INV_SELECT_SALT,
            item_id,
            "|",
            &anchor,
            "|",
            fingerprint,
            "|",
        ],
    );
    let inverse = select_by_key_pref(store, &pool, &invk, py_take(pool.len(), max_inverse), pref);
    if first.is_empty() && inverse.is_empty() {
        return HopManifest::default();
    }
    let mut inv_sorted = inverse.clone();
    inv_sorted.sort_unstable();
    let in_inv = |r: RecId| inv_sorted.binary_search(&r).is_ok();
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
    let mut chosen: Vec<RecId>;
    if first.len() >= k {
        chosen = match pref {
            Some(p) => p.keep(store, &first, k),
            None => first[..k].to_vec(),
        };
    } else {
        chosen = first.clone();
        chosen.extend(inverse.iter().copied().take(k - first.len()));
        if chosen.len() < k {
            let mut rels: SmallVec<[u32; 8]> = SmallVec::new();
            rels_of(store, &first, &mut rels);
            rels_of(store, &inverse, &mut rels);
            let m = k - chosen.len();
            let mut sampled = None;
            if hash == ManifestHash::Sampled {
                let seed = xxhash_rust::xxh3::xxh3_64(
                    format!("hop sample|{item_id}|{anchor}|{fingerprint}").as_bytes(),
                );
                let lists2: Vec<&[RecId]> =
                    rels.iter().map(|&rl| idx.records_of_relation(rl)).collect();
                sampled = sample_lists(&lists2, m, seed, |r| {
                    snap.visible(r) && !in_first(r) && !in_inv(r)
                });
            }
            let (threads, par_min) = manifest_threads();
            if let Some(mut picked) = sampled {
                chosen.append(&mut picked);
            } else if threads > 1 {
                // EXTENSION (opt-in, exact): the same tiers 2 and 3 on `threads` threads ([`scan_top_m`])
                let lists: Vec<&[RecId]> =
                    rels.iter().map(|&rl| idx.records_of_relation(rl)).collect();
                let keep2 = |r: RecId| snap.visible(r) && !in_first(r) && !in_inv(r);
                let (second, seen) = scan_top_m(store, &sel, &lists, m, &keep2, threads, par_min);
                let need_rest = m.saturating_sub(seen);
                chosen.extend(second);
                if need_rest > 0 {
                    let lists: Vec<&[RecId]> = (0..idx.by_relation.len() as u32)
                        .filter(|rl| !rels.contains(rl))
                        .map(|rl| idx.records_of_relation(rl))
                        .collect();
                    let keep3 = |r: RecId| snap.visible(r);
                    let (rest, _) =
                        scan_top_m(store, &sel, &lists, need_rest, &keep3, threads, par_min);
                    chosen.extend(rest);
                }
            } else {
                let mut second = TopM::new(m);
                for &rl in &rels {
                    for &r in idx.records_of_relation(rl) {
                        if snap.visible(r) && !in_first(r) && !in_inv(r) {
                            second.offer(&sel, r, fid(r));
                        }
                    }
                }
                let need_rest = m.saturating_sub(second.seen);
                chosen.extend(second.into_sorted(&sel));
                if need_rest > 0 {
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
                    chosen.extend(rest.into_sorted(&sel));
                }
            }
            chosen.truncate(k);
        }
    }
    order_by(snap, &order, &mut chosen);
    let resolves = chosen.iter().map(|&r| in_first(r) || in_inv(r)).collect();
    let inverse_flags = chosen.iter().map(|&r| in_inv(r)).collect();
    HopManifest {
        cands: chosen,
        resolves,
        anchor: Some(anchor),
        inverse: inverse_flags,
    }
}

// ================================================================================================ record scopes (opt-in)

/// EXTENSION (opt-in; never on the reference path): how a scoped manifest treats the records with `keep(r)` (the records
/// of a question's scope, `crate::scope`). The reference functions above are unchanged; [`candidate_manifest_v5_scope`]
/// and [`hop_manifest_v5_scope`] are their scoped twins.
#[derive(Clone, Copy)]
pub enum RecScope<'a> {
    /// the manifest is built as if the store held only the records with `keep(r)`: every visibility test of the reference
    /// manifest (tiers 1, 1-IN, 2 and 3) is `visible(r) && keep(r)`
    Filter(&'a (dyn Fn(RecId) -> bool + Sync)),
    /// the records with `keep(r)` come first where the reference keeps a prefix of tier 1 (above K) or of the inverse pool
    /// (above `max_inverse`); everything else (tiers 2 and 3, the order) is the reference's
    Prefer(&'a (dyn Fn(RecId) -> bool + Sync)),
}

impl RecScope<'_> {
    /// may the record be offered at all? (`Prefer` hides nothing)
    #[inline]
    fn admits(&self, r: RecId) -> bool {
        match self {
            RecScope::Filter(k) => k(r),
            RecScope::Prefer(_) => true,
        }
    }
}

/// [`RecScope::Prefer`]: the `n` records of `recs` (given in the reference preference order) to keep: those with `keep(r)`
/// first, then the others, each part by the reference rule (its first records) or the relation-aware one (`pref`)
fn keep_prefer(
    store: &Store,
    recs: &[RecId],
    n: usize,
    pref: Option<&RelPref>,
    keep: &(dyn Fn(RecId) -> bool + Sync),
) -> Vec<RecId> {
    let (inn, out): (Vec<RecId>, Vec<RecId>) = recs.iter().partition(|&&r| keep(r));
    let take = |v: &[RecId], m: usize| -> Vec<RecId> {
        match pref {
            Some(p) => p.keep(store, v, m),
            None => v[..m.min(v.len())].to_vec(),
        }
    };
    let mut o = take(&inn, n);
    if o.len() < n {
        let rest = n - o.len();
        o.extend(take(&out, rest));
    }
    o
}

/// the inverse candidates of a scoped manifest: the reference choice ([`select_by_key_pref`]) over the admitted pool, with
/// [`RecScope::Prefer`] the kept records first where the cap binds
fn select_inverse_scoped(
    store: &Store,
    pool: &[RecId],
    kf: &KeyFn,
    n: usize,
    pref: Option<&RelPref>,
    scope: RecScope<'_>,
) -> Vec<RecId> {
    match scope {
        RecScope::Prefer(keep) if pool.len() > n => {
            let ranked = select_by_key(store, pool, kf, pool.len());
            keep_prefer(store, &ranked, n, pref, keep)
        }
        _ => select_by_key_pref(store, pool, kf, n, pref),
    }
}

/// [`candidate_manifest_v5_pref`] over a record scope (EXTENSION, opt-in; see [`RecScope`]). With a scope that admits and
/// keeps every record the result is [`candidate_manifest_v5_pref`]'s, record for record (unit test
/// `scoped_manifests_with_a_full_scope_are_the_reference`).
#[allow(clippy::too_many_arguments)]
pub fn candidate_manifest_v5_scope(
    snap: &Snapshot<'_>,
    idx: &SubjectIndex,
    t: &ValueTables,
    ents: &[u32],
    vals: &[u32],
    item_id: &str,
    fingerprint: &str,
    k: usize,
    hash: ManifestHash,
    max_inverse: i64,
    pref: Option<&RelPref>,
    scope: RecScope<'_>,
) -> CandManifest {
    let store = snap.store;
    let fid = |r: RecId| store.fact_id(r);
    let linked = |r: RecId| ents.contains(&store.recs.subject[r as usize]);
    let vis = |r: RecId| snap.visible(r) && scope.admits(r);
    // tier 1: records of the linked subjects, snapshot (fact id) order
    let mut first: Vec<RecId> = Vec::new();
    for &s in ents {
        for &r in idx.records_of_subject(s) {
            if vis(r) {
                first.push(r);
            }
        }
    }
    first.sort_unstable_by(|&a, &b| fid(a).cmp(fid(b)));
    first.dedup();
    let order = KeyFn::new(hash, &[ORDER_SALT, item_id, fingerprint]);
    if first.len() >= k {
        first = match (scope, pref) {
            (RecScope::Prefer(keep), _) => keep_prefer(store, &first, k, pref, keep),
            (RecScope::Filter(_), Some(p)) => p.keep(store, &first, k),
            (RecScope::Filter(_), None) => first,
        };
        first.truncate(k);
        order_by(snap, &order, &mut first);
        let n = first.len();
        return CandManifest {
            cands: first,
            inverse: vec![false; n],
        };
    }
    // tier 1-IN: text records with a linked value whose subject is not linked, <= max_inverse by the inverse key
    let mut pool: Vec<RecId> = Vec::new();
    for &v in vals {
        for &r in t.records_of_value(v) {
            if vis(r) && !linked(r) {
                pool.push(r);
            }
        }
    }
    pool.sort_unstable();
    pool.dedup();
    let invk = KeyFn::new(
        hash,
        &[INV_SELECT_SALT, item_id, "|", "", "|", fingerprint, "|"],
    );
    let inverse = select_inverse_scoped(
        store,
        &pool,
        &invk,
        py_take(pool.len(), max_inverse),
        pref,
        scope,
    );
    let mut inv_sorted = inverse.clone();
    inv_sorted.sort_unstable();
    let in_inv = |r: RecId| inv_sorted.binary_search(&r).is_ok();
    let mut chosen: Vec<RecId> = first.clone();
    chosen.extend(inverse.iter().copied().take(k - first.len()));
    if chosen.len() < k {
        let mut rels: SmallVec<[u32; 8]> = SmallVec::new();
        rels_of(store, &first, &mut rels);
        rels_of(store, &inverse, &mut rels);
        let m = k - chosen.len();
        let sel = KeyFn::new(hash, &[item_id, fingerprint]);
        let mut sampled = None;
        if hash == ManifestHash::Sampled {
            let seed =
                xxhash_rust::xxh3::xxh3_64(format!("sample|{item_id}|{fingerprint}").as_bytes());
            let lists2: Vec<&[RecId]> =
                rels.iter().map(|&rl| idx.records_of_relation(rl)).collect();
            sampled = sample_lists(&lists2, m, seed, |r| vis(r) && !linked(r) && !in_inv(r));
        }
        let (threads, par_min) = manifest_threads();
        if let Some(mut picked) = sampled {
            chosen.append(&mut picked);
        } else if threads > 1 {
            let lists: Vec<&[RecId]> = rels.iter().map(|&rl| idx.records_of_relation(rl)).collect();
            let keep2 = |r: RecId| vis(r) && !linked(r) && !in_inv(r);
            let (second, seen) = scan_top_m(store, &sel, &lists, m, &keep2, threads, par_min);
            let need_rest = m.saturating_sub(seen);
            chosen.extend(second);
            if need_rest > 0 {
                let lists: Vec<&[RecId]> = (0..idx.by_relation.len() as u32)
                    .filter(|rl| !rels.contains(rl))
                    .map(|rl| idx.records_of_relation(rl))
                    .collect();
                let keep3 = |r: RecId| vis(r);
                let (rest, _) =
                    scan_top_m(store, &sel, &lists, need_rest, &keep3, threads, par_min);
                chosen.extend(rest);
            }
        } else {
            // tier 2: other subjects, a relation of tier 1 / 1-IN, not 1-IN
            let mut second = TopM::new(m);
            for &rl in &rels {
                for &r in idx.records_of_relation(rl) {
                    if vis(r) && !linked(r) && !in_inv(r) {
                        second.offer(&sel, r, fid(r));
                    }
                }
            }
            let need_rest = m.saturating_sub(second.seen);
            chosen.extend(second.into_sorted(&sel));
            if need_rest > 0 {
                // tier 3: every other admitted visible record (relation outside tiers 1 / 1-IN)
                let mut rest = TopM::new(need_rest);
                for rl in 0..idx.by_relation.len() as u32 {
                    if rels.contains(&rl) {
                        continue;
                    }
                    for &r in idx.records_of_relation(rl) {
                        if vis(r) {
                            rest.offer(&sel, r, fid(r));
                        }
                    }
                }
                chosen.extend(rest.into_sorted(&sel));
            }
        }
        chosen.truncate(k);
    }
    order_by(snap, &order, &mut chosen);
    let inverse_flags = chosen.iter().map(|&r| in_inv(r)).collect();
    CandManifest {
        cands: chosen,
        inverse: inverse_flags,
    }
}

/// [`hop_manifest_v5_pref`] over a record scope (EXTENSION, opt-in; see [`RecScope`]). With a scope that admits and keeps
/// every record the result is [`hop_manifest_v5_pref`]'s, record for record.
#[allow(clippy::too_many_arguments)]
pub fn hop_manifest_v5_scope(
    snap: &Snapshot<'_>,
    idx: &SubjectIndex,
    t: &ValueTables,
    item_id: &str,
    fingerprint: &str,
    v: &TypedValue<'_>,
    value_ids: &[&str],
    k: usize,
    hash: ManifestHash,
    max_inverse: i64,
    pref: Option<&RelPref>,
    scope: RecScope<'_>,
) -> HopManifest {
    if v.kind != Kind::Text {
        return HopManifest::default();
    }
    let store = snap.store;
    let fid = |r: RecId| store.fact_id(r);
    let vis = |r: RecId| snap.visible(r) && scope.admits(r);
    let first: Vec<RecId> = resolving_records(snap, idx, v, value_ids)
        .into_iter()
        .filter(|&r| scope.admits(r))
        .collect();
    let mut first_sorted: Vec<RecId> = first.clone();
    first_sorted.sort_unstable();
    let in_first = |r: RecId| first_sorted.binary_search(&r).is_ok();
    let anchor = value_anchor(v, value_ids);
    let pool: Vec<RecId> = value_resolving_records(snap, t, v, value_ids)
        .into_iter()
        .filter(|&r| !in_first(r) && scope.admits(r))
        .collect();
    let invk = KeyFn::new(
        hash,
        &[
            INV_SELECT_SALT,
            item_id,
            "|",
            &anchor,
            "|",
            fingerprint,
            "|",
        ],
    );
    let inverse = select_inverse_scoped(
        store,
        &pool,
        &invk,
        py_take(pool.len(), max_inverse),
        pref,
        scope,
    );
    if first.is_empty() && inverse.is_empty() {
        return HopManifest::default();
    }
    let mut inv_sorted = inverse.clone();
    inv_sorted.sort_unstable();
    let in_inv = |r: RecId| inv_sorted.binary_search(&r).is_ok();
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
    let mut chosen: Vec<RecId>;
    if first.len() >= k {
        chosen = match (scope, pref) {
            (RecScope::Prefer(keep), _) => keep_prefer(store, &first, k, pref, keep),
            (RecScope::Filter(_), Some(p)) => p.keep(store, &first, k),
            (RecScope::Filter(_), None) => first[..k].to_vec(),
        };
    } else {
        chosen = first.clone();
        chosen.extend(inverse.iter().copied().take(k - first.len()));
        if chosen.len() < k {
            let mut rels: SmallVec<[u32; 8]> = SmallVec::new();
            rels_of(store, &first, &mut rels);
            rels_of(store, &inverse, &mut rels);
            let m = k - chosen.len();
            let mut sampled = None;
            if hash == ManifestHash::Sampled {
                let seed = xxhash_rust::xxh3::xxh3_64(
                    format!("hop sample|{item_id}|{anchor}|{fingerprint}").as_bytes(),
                );
                let lists2: Vec<&[RecId]> =
                    rels.iter().map(|&rl| idx.records_of_relation(rl)).collect();
                sampled = sample_lists(&lists2, m, seed, |r| vis(r) && !in_first(r) && !in_inv(r));
            }
            let (threads, par_min) = manifest_threads();
            if let Some(mut picked) = sampled {
                chosen.append(&mut picked);
            } else if threads > 1 {
                let lists: Vec<&[RecId]> =
                    rels.iter().map(|&rl| idx.records_of_relation(rl)).collect();
                let keep2 = |r: RecId| vis(r) && !in_first(r) && !in_inv(r);
                let (second, seen) = scan_top_m(store, &sel, &lists, m, &keep2, threads, par_min);
                let need_rest = m.saturating_sub(seen);
                chosen.extend(second);
                if need_rest > 0 {
                    let lists: Vec<&[RecId]> = (0..idx.by_relation.len() as u32)
                        .filter(|rl| !rels.contains(rl))
                        .map(|rl| idx.records_of_relation(rl))
                        .collect();
                    let keep3 = |r: RecId| vis(r);
                    let (rest, _) =
                        scan_top_m(store, &sel, &lists, need_rest, &keep3, threads, par_min);
                    chosen.extend(rest);
                }
            } else {
                let mut second = TopM::new(m);
                for &rl in &rels {
                    for &r in idx.records_of_relation(rl) {
                        if vis(r) && !in_first(r) && !in_inv(r) {
                            second.offer(&sel, r, fid(r));
                        }
                    }
                }
                let need_rest = m.saturating_sub(second.seen);
                chosen.extend(second.into_sorted(&sel));
                if need_rest > 0 {
                    let mut rest = TopM::new(need_rest);
                    for rl in 0..idx.by_relation.len() as u32 {
                        if rels.contains(&rl) {
                            continue;
                        }
                        for &r in idx.records_of_relation(rl) {
                            if vis(r) {
                                rest.offer(&sel, r, fid(r));
                            }
                        }
                    }
                    chosen.extend(rest.into_sorted(&sel));
                }
            }
            chosen.truncate(k);
        }
    }
    order_by(snap, &order, &mut chosen);
    let resolves = chosen.iter().map(|&r| in_first(r) || in_inv(r)).collect();
    let inverse_flags = chosen.iter().map(|&r| in_inv(r)).collect();
    HopManifest {
        cands: chosen,
        resolves,
        anchor: Some(anchor),
        inverse: inverse_flags,
    }
}

// ================================================================================================ executor support

/// The records of `stack_v5.inverse_result`'s set check for record `r`: every visible text record with `r`'s relation and
/// scope whose value denotes the same value as `r`'s (`stack_v5.same_value`: value ids when both carry one, else
/// normalize_entity of the strings), sorted by fact id. Empty when `r` is not a text record.
pub fn same_value_records(snap: &Snapshot<'_>, t: &ValueTables, r: RecId) -> Vec<RecId> {
    let store = snap.store;
    let i = r as usize;
    let mut out: Vec<RecId> = Vec::new();
    if store.recs.kind[i] != Kind::Text {
        return out;
    }
    let (rel, scope, fvid) = (
        store.recs.relation[i],
        store.recs.scope[i],
        store.recs.value_id[i],
    );
    let ok = |g: RecId| {
        let j = g as usize;
        store.recs.relation[j] == rel && store.recs.scope[j] == scope && snap.visible(g)
    };
    if fvid != NONE {
        for &g in t.records_of_value_id(fvid) {
            if ok(g) {
                out.push(g);
            }
        }
    }
    let nf = normalize_entity(store.pools.strs.get(store.recs.value[i]));
    for &sym in t.values_with_norm(&nf) {
        for &g in t.records_of_value(sym) {
            if ok(g) && (fvid == NONE || store.recs.value_id[g as usize] == NONE) {
                out.push(g);
            }
        }
    }
    out.sort_unstable_by(|&a, &b| store.fact_id(a).cmp(store.fact_id(b)));
    out.dedup();
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn python_slice_lengths() {
        assert_eq!(py_take(5, 8), 5);
        assert_eq!(py_take(5, 2), 2);
        assert_eq!(py_take(5, 0), 0);
        assert_eq!(py_take(5, -1), 4);
        assert_eq!(py_take(5, -9), 0);
        assert_eq!(parse_max_inverse(None), 8);
        assert_eq!(parse_max_inverse(Some(" 3 ")), 3);
        assert_eq!(parse_max_inverse(Some("0")), 0);
        assert_eq!(parse_max_inverse(Some("x")), 8);
    }

    #[test]
    fn inverse_key_matches_python() {
        use sha2::{Digest, Sha256};
        // hashlib.sha256(("stack-v5 inverse select|" + "it" + "|" + "" + "|" + "fp" + "|" + "f1").encode()).digest()
        let kf = KeyFn::new(
            ManifestHash::Sha256,
            &[INV_SELECT_SALT, "it", "|", "", "|", "fp", "|"],
        );
        let expect: [u8; 32] = Sha256::digest(b"stack-v5 inverse select|it||fp|f1").into();
        assert_eq!(kf.key("f1"), expect);
    }

    #[test]
    fn manifest_thread_settings_parse() {
        assert_eq!(
            parse_manifest_threads(None, None),
            (1, DEFAULT_MANIFEST_PAR_MIN)
        );
        assert_eq!(parse_manifest_threads(Some("4"), Some("0")), (4, 0));
        assert_eq!(
            parse_manifest_threads(Some(" 0 "), Some("x")),
            (1, DEFAULT_MANIFEST_PAR_MIN)
        );
        assert_eq!(
            parse_manifest_threads(Some("1000"), None),
            (64, DEFAULT_MANIFEST_PAR_MIN)
        );
    }

    /// the threaded tier-2 / tier-3 scan keeps the same records in the same order as the single-threaded loop
    #[test]
    fn threaded_scan_equals_the_single_threaded_loop() {
        use crate::memory::{Memory, MemoryConfig};
        use xerj_store::FactIn;
        let mut mem = Memory::new(MemoryConfig {
            bm25_records: false,
            bm25_documents: false,
            ..MemoryConfig::default()
        });
        mem.add_document("d", "v1", "abcd efgh").expect("doc");
        let n = 3000;
        let names: Vec<(String, String, String, String)> = (0..n)
            .map(|i| {
                (
                    format!("f{i:05}"),
                    format!("subject {}", i % 97),
                    format!("relation {}", i % 7),
                    format!("value {i}"),
                )
            })
            .collect();
        for (f, s, r, v) in &names {
            let fi = FactIn::text(f, s, r, v, ("d", "v1"), (0, 4), "abcd");
            mem.add_fact(&fi, &[]).expect("fact");
        }
        let store = &mem.store;
        let recs: Vec<RecId> = (0..n as RecId).collect();
        // uneven lists, like the records of a few relations
        let lists: Vec<&[RecId]> = vec![
            &recs[..1],
            &recs[1..700],
            &recs[700..701],
            &recs[701..2999],
            &recs[2999..],
        ];
        let keep = |r: RecId| r % 5 != 3;
        for hash in [ManifestHash::Sha256, ManifestHash::Fast] {
            for item in ["lab/q/0001", "lab/q/0002"] {
                let kf = KeyFn::new(hash, &[HOP_SELECT_SALT, item, "|", "a", "|", "fp", "|"]);
                for m in [0usize, 1, 5, 16, 100, 2399, 2400, 5000] {
                    let one = scan_top_m(store, &kf, &lists, m, &keep, 1, 0);
                    // the reference loop itself
                    let mut top = TopM::new(m);
                    for &l in &lists {
                        for &r in l {
                            if keep(r) {
                                top.offer(&kf, r, store.fact_id(r));
                            }
                        }
                    }
                    let seen = top.seen;
                    assert_eq!(one, (top.into_sorted(&kf), seen), "{hash:?} {item} m={m}");
                    assert_eq!(one.1, 2400);
                    for threads in [2usize, 3, 4, 7, 64] {
                        assert_eq!(
                            scan_top_m(store, &kf, &lists, m, &keep, threads, 0),
                            one,
                            "{hash:?} {item} m={m} threads={threads}"
                        );
                    }
                    // below the threshold the calling thread scans alone
                    assert_eq!(scan_top_m(store, &kf, &lists, m, &keep, 4, n + 1), one);
                }
            }
        }
    }

    /// a small store whose subjects are also values (hops resolve, inverse candidates exist, caps bind with small K)
    fn scope_test_memory() -> crate::memory::Memory {
        use crate::memory::{Memory, MemoryConfig};
        use xerj_store::FactIn;
        let mut mem = Memory::new(MemoryConfig {
            bm25_records: false,
            bm25_documents: false,
            ..MemoryConfig::default()
        });
        mem.add_document("d", "v1", "abcd efgh").expect("doc");
        let names = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta"];
        let rels = [
            "capital",
            "head of state",
            "country of citizenship",
            "official language",
        ];
        for i in 0..90usize {
            let s = names[i % 6];
            let r = rels[(i / 6) % 4];
            let v = if i % 3 == 0 {
                names[(i * 7 + 1) % 6].to_string()
            } else {
                format!("value {i}")
            };
            let id = format!("f{i:03}");
            let fi = FactIn::text(&id, s, r, &v, ("d", "v1"), (0, 4), "abcd");
            mem.add_fact(&fi, &[]).expect("fact");
        }
        mem
    }

    /// a scope that admits and keeps every record (Filter with keep = true, Prefer with any keep where no record is
    /// preferred over another) gives the reference manifests record for record, base and hop, with and without RelPref
    #[test]
    fn scoped_manifests_with_a_full_scope_are_the_reference() {
        let mem = scope_test_memory();
        let snap = mem.head();
        let t = mem.values.tables(&mem.store);
        let all = |_: RecId| true;
        let none = |_: RecId| false;
        let names = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta"];
        let mut buf = crate::linker::QBuf::default();
        let mut compared = 0;
        for q in [
            "What is the capital of Alpha?",
            "Who is the head of state of Beta and Gamma?",
            "Whose official language is Delta?",
        ] {
            let ents = mem.link(&snap, q, &mut buf);
            let vals = link_values(&snap, &t, q, &mut buf);
            let rp = RelPref::from_question(q);
            for item in ["lab/q/0001", "lab/q/0002"] {
                for (k, mi) in [(16usize, 8i64), (4, 2), (2, 1), (16, 0)] {
                    for pref in [None, Some(&rp)] {
                        let reference = candidate_manifest_v5_pref(
                            &snap,
                            &mem.subjects,
                            &t,
                            &ents,
                            &vals,
                            item,
                            "fp",
                            k,
                            ManifestHash::Sha256,
                            mi,
                            pref,
                        );
                        for scope in [
                            RecScope::Filter(&all),
                            RecScope::Prefer(&all),
                            RecScope::Prefer(&none),
                        ] {
                            let s = candidate_manifest_v5_scope(
                                &snap,
                                &mem.subjects,
                                &t,
                                &ents,
                                &vals,
                                item,
                                "fp",
                                k,
                                ManifestHash::Sha256,
                                mi,
                                pref,
                                scope,
                            );
                            assert_eq!(s, reference, "base {q} {item} k={k} mi={mi}");
                            compared += 1;
                        }
                        for n in names {
                            let v = TypedValue {
                                kind: Kind::Text,
                                value: n,
                                unit: "",
                            };
                            let reference = hop_manifest_v5_pref(
                                &snap,
                                &mem.subjects,
                                &t,
                                item,
                                "fp",
                                &v,
                                &[],
                                k,
                                ManifestHash::Sha256,
                                mi,
                                pref,
                            );
                            for scope in [
                                RecScope::Filter(&all),
                                RecScope::Prefer(&all),
                                RecScope::Prefer(&none),
                            ] {
                                let s = hop_manifest_v5_scope(
                                    &snap,
                                    &mem.subjects,
                                    &t,
                                    item,
                                    "fp",
                                    &v,
                                    &[],
                                    k,
                                    ManifestHash::Sha256,
                                    mi,
                                    pref,
                                    scope,
                                );
                                assert_eq!(
                                    (&s.cands, &s.resolves, &s.inverse, &s.anchor),
                                    (
                                        &reference.cands,
                                        &reference.resolves,
                                        &reference.inverse,
                                        &reference.anchor
                                    ),
                                    "hop {n} {item} k={k} mi={mi}"
                                );
                                compared += 1;
                            }
                        }
                    }
                }
            }
        }
        assert!(compared > 400);
    }

    /// Filter offers only the kept records (as if the store held nothing else); Prefer keeps the kept records of an
    /// over-full tier 1 first
    #[test]
    fn scoped_manifests_filter_and_prefer() {
        let mem = scope_test_memory();
        let snap = mem.head();
        let t = mem.values.tables(&mem.store);
        // records 0, 12, 24, ... are out of scope: Alpha's tier 1 (records 0, 6, 12, ... in fact-id order) mixes both
        let keep = |r: RecId| !r.is_multiple_of(12);
        let mut buf = crate::linker::QBuf::default();
        let q = "What is the capital of Alpha?";
        let ents = mem.link(&snap, q, &mut buf);
        let vals = link_values(&snap, &t, q, &mut buf);
        for k in [16usize, 4] {
            let m = candidate_manifest_v5_scope(
                &snap,
                &mem.subjects,
                &t,
                &ents,
                &vals,
                "lab/q/1",
                "fp",
                k,
                ManifestHash::Sha256,
                8,
                None,
                RecScope::Filter(&keep),
            );
            assert!(
                !m.cands.is_empty() && m.cands.iter().all(|&r| keep(r)),
                "base k={k}: {:?}",
                m.cands
            );
            for n in ["Alpha", "Beta", "Gamma"] {
                let v = TypedValue {
                    kind: Kind::Text,
                    value: n,
                    unit: "",
                };
                let h = hop_manifest_v5_scope(
                    &snap,
                    &mem.subjects,
                    &t,
                    "lab/q/1",
                    "fp",
                    &v,
                    &[],
                    k,
                    ManifestHash::Sha256,
                    8,
                    None,
                    RecScope::Filter(&keep),
                );
                assert!(h.cands.iter().all(|&r| keep(r)), "hop {n} k={k}");
                // every kept record resolving to the value is offered when tier 1 fits
                let kept_first: Vec<RecId> =
                    crate::manifest::resolving_records(&snap, &mem.subjects, &v, &[])
                        .into_iter()
                        .filter(|&r| keep(r))
                        .collect();
                if kept_first.len() <= k {
                    assert!(
                        kept_first.iter().all(|r| h.cands.contains(r)),
                        "hop {n} k={k}"
                    );
                }
            }
        }
        // Prefer: tier 1 of Alpha's hop manifest is over-full at K = 4; the kept records come first
        let v = TypedValue {
            kind: Kind::Text,
            value: "Alpha",
            unit: "",
        };
        let first = crate::manifest::resolving_records(&snap, &mem.subjects, &v, &[]);
        assert!(first.len() > 4);
        let kept: Vec<RecId> = first.iter().copied().filter(|&r| keep(r)).collect();
        let h = hop_manifest_v5_scope(
            &snap,
            &mem.subjects,
            &t,
            "lab/q/1",
            "fp",
            &v,
            &[],
            4,
            ManifestHash::Sha256,
            8,
            None,
            RecScope::Prefer(&keep),
        );
        let reference = hop_manifest_v5_pref(
            &snap,
            &mem.subjects,
            &t,
            "lab/q/1",
            "fp",
            &v,
            &[],
            4,
            ManifestHash::Sha256,
            8,
            None,
        );
        let mut a = h.cands.clone();
        a.sort_unstable();
        let mut b: Vec<RecId> = kept.iter().copied().take(4).collect();
        b.sort_unstable();
        assert_eq!(a, b, "the first four kept records of tier 1");
        assert_ne!(
            h.cands, reference.cands,
            "the reference keeps the first four of tier 1, kept or not"
        );
    }

    #[test]
    fn relation_words_score_the_question() {
        let p = RelPref::from_question(
            "Whose official language is the languages spoken, written or signed of Floyd Gottfredson?",
        );
        assert_eq!(p.score("official language"), 2);
        assert_eq!(p.score("languages spoken, written or signed"), 4);
        assert_eq!(p.score("country of citizenship"), 0);
        let p = RelPref::from_question(
            "Who is the chief executive of the country Alan Hollinghurst is a citizen of?",
        );
        assert_eq!(
            p.score("country of citizenship"),
            2,
            "citizen ~ citizenship, country"
        );
        assert_eq!(p.score("head of state"), 0);
        assert_eq!(RelPref::from_question("Which is it?").score("is"), 0);
    }
}
