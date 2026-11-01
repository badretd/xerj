//! The versioned record store (reference_core.MemoryStore semantics, columnar and indexed).
//!
//! * Every successful write takes the next revision (exactly as the reference: add_document, add_fact, withdraw_fact and
//!   withdraw_document each +1), so revision numbers, and therefore exact snapshot fingerprints, match Python.
//! * MVCC-lite: a record is visible at revision r iff `added <= r < vis_end`, where `vis_end = min(retired, document
//!   withdrawal)` is maintained eagerly on supersede / withdraw (`retired` alone answers "already withdrawn?", as the
//!   reference's `_withdrawn_facts`). A `Snapshot` is (store, revision, valid_at): O(1) to construct for any revision, so
//!   "before" / "after" answers need no copy and no rebuild; reads filter by visibility.
//! * Columnar records (struct of arrays), strings interned in `StrPool` arenas; a (subject, relation, scope) key index with
//!   intra-key chains (`next_in_key`) makes lookups O(versions of the key) instead of the reference's full scan.
//! * Writes are O(1) amortized and immediately visible to the next read (read-your-writes); nothing is rebuilt.

use crate::error::*;
use crate::pool::StrPool;
use crate::text;
use crate::value::{self, Kind, TypedValue};
use foldhash::fast::FixedState;
use hashbrown::HashMap;
use sha2::{Digest, Sha256};
use smallvec::SmallVec;
use std::sync::{Arc, Mutex};

pub type Rev = u32;
pub type RecId = u32;
pub type DocIx = u32;
/// "no id" sentinel of optional u32 columns
pub const NONE: u32 = u32::MAX;
/// "never" sentinel of revision columns (retired / withdrawn / vis_end)
pub const OPEN: Rev = Rev::MAX;

pub const DEFAULT_SCOPE: &str = "default";
/// writer_version of facts built by xr_codecs._fact (part of every exact fingerprint)
pub const ORACLE_WRITER: &str = "memory-world-oracle-v1";
/// reference_core.Fact's default writer_version
pub const FIXTURE_WRITER: &str = "fixture-v1";
/// xr_codecs.VALID_AT, the valid_at of every snapshot in the reference experiments
pub const VALID_AT: &str = "2026-10-20";

/// reference_core.Status
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Status {
    Value,
    True,
    False,
    Unknown,
    Conflict,
}

impl Status {
    /// `Status.X.name` (result_text of non-VALUE results)
    pub fn name(self) -> &'static str {
        match self {
            Status::Value => "VALUE",
            Status::True => "TRUE",
            Status::False => "FALSE",
            Status::Unknown => "UNKNOWN",
            Status::Conflict => "CONFLICT",
        }
    }
    /// `Status.X.value`
    pub fn value_str(self) -> &'static str {
        match self {
            Status::Value => "value",
            Status::True => "true",
            Status::False => "false",
            Status::Unknown => "unknown",
            Status::Conflict => "conflict",
        }
    }
}

/// A fact's raw value as written (a string for text / date / number, a bool for boolean)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ValueIn<'a> {
    Str(&'a str),
    Bool(bool),
    /// any other JSON type (rejected like the reference: "Nonboolean values require a nonempty string" / bool rule)
    Other,
}

/// One fact record to write (the memory-world-v1 record fields + reference_core.Fact's optional fields)
#[derive(Clone, Debug)]
pub struct FactIn<'a> {
    pub fact_id: &'a str,
    pub subject: &'a str,
    pub relation: &'a str,
    pub kind: &'a str,
    pub value: ValueIn<'a>,
    pub unit: &'a str,
    pub doc_id: &'a str,
    pub version: &'a str,
    pub start: i64,
    pub end: i64,
    pub quote: &'a str,
    pub scope: &'a str,
    pub subject_id: Option<&'a str>,
    pub value_id: Option<&'a str>,
    pub valid_from: Option<&'a str>,
    pub valid_to: Option<&'a str>,
    pub writer_version: &'a str,
}

impl<'a> FactIn<'a> {
    /// a text fact with the oracle writer, default scope, no ids / validity (tests and tools)
    pub fn text(
        fact_id: &'a str,
        subject: &'a str,
        relation: &'a str,
        value: &'a str,
        doc: (&'a str, &'a str),
        span: (i64, i64),
        quote: &'a str,
    ) -> Self {
        FactIn {
            fact_id,
            subject,
            relation,
            kind: "text",
            value: ValueIn::Str(value),
            unit: "",
            doc_id: doc.0,
            version: doc.1,
            start: span.0,
            end: span.1,
            quote,
            scope: DEFAULT_SCOPE,
            subject_id: None,
            value_id: None,
            valid_from: None,
            valid_to: None,
            writer_version: ORACLE_WRITER,
        }
    }
}

/// How snapshot fingerprints (the seed of the label-free manifest order) are computed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FingerprintMode {
    /// reference_core.Snapshot.fingerprint bit for bit: sha256 of the canonical JSON of every visible document and fact.
    /// O(N log N) per new (revision, valid_at), cached; manifests then equal Python's exactly.
    Exact,
    /// sha256("xerj-mem/revision-fp/v1|{store_id}|{revision}|{valid_at}"): O(1) for any revision, deterministic for a
    /// given op sequence (revisions are), different from Python's value (manifests keep every tier / hashing rule but
    /// order differently). For large stores with frequent writes.
    Revision,
}

#[derive(Clone, Debug)]
pub struct StoreConfig {
    pub fingerprint: FingerprintMode,
    /// mixed into Revision fingerprints (empty by default: identical op sequences give identical fingerprints)
    pub store_id: String,
}

impl Default for StoreConfig {
    fn default() -> Self {
        StoreConfig {
            fingerprint: FingerprintMode::Exact,
            store_id: String::new(),
        }
    }
}

/// A validated `YYYY-MM-DD` date (Copy, no allocation).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ValidAt([u8; 10]);

impl ValidAt {
    pub fn parse(s: &str) -> Result<ValidAt, ContractError> {
        value::validate_iso_date(s)?;
        let mut b = [0u8; 10];
        b.copy_from_slice(s.as_bytes());
        Ok(ValidAt(b))
    }
    #[inline]
    pub fn as_str(&self) -> &str {
        // SAFETY: validated ASCII
        unsafe { std::str::from_utf8_unchecked(&self.0) }
    }
}

/// All interned string domains
#[derive(Clone, Default)]
pub struct Pools {
    /// fact ids; the pool id IS the record id
    pub fact_ids: StrPool,
    pub subjects: StrPool,
    pub relations: StrPool,
    /// canonical values, quotes and validity dates
    pub strs: StrPool,
    pub units: StrPool,
    pub scopes: StrPool,
    /// entity ids (subject_id / value_id)
    pub ent_ids: StrPool,
    pub doc_ids: StrPool,
    pub versions: StrPool,
    pub writers: StrPool,
}

/// Record columns (struct of arrays, indexed by RecId)
#[derive(Clone, Default)]
pub struct Records {
    pub subject: Vec<u32>,
    pub relation: Vec<u32>,
    pub scope: Vec<u32>,
    pub kind: Vec<Kind>,
    pub value: Vec<u32>,
    pub unit: Vec<u16>,
    pub doc: Vec<DocIx>,
    pub start: Vec<u32>,
    pub end: Vec<u32>,
    pub quote: Vec<u32>,
    pub subject_id: Vec<u32>,
    pub value_id: Vec<u32>,
    pub writer: Vec<u16>,
    pub added: Vec<Rev>,
    pub retired: Vec<Rev>,
    pub vis_end: Vec<Rev>,
    pub next_in_key: Vec<RecId>,
    /// sparse validity intervals (valid_from, valid_to) as strs ids, NONE = unbounded
    pub interval: HashMap<RecId, (u32, u32), FixedState>,
}

/// Document columns (indexed by DocIx)
#[derive(Clone, Default)]
pub struct Docs {
    pub doc_id: Vec<u32>,
    pub version: Vec<u32>,
    pub text_end: Vec<u64>,
    pub text: Vec<u8>,
    pub ascii: Vec<bool>,
    pub added: Vec<Rev>,
    pub withdrawn: Vec<Rev>,
    /// records citing the document (withdrawal propagation)
    pub recs: Vec<SmallVec<[RecId; 3]>>,
}

/// Result of a successful add_fact
#[derive(Clone, Debug)]
pub struct AddedFact {
    pub rec: RecId,
    pub rev: Rev,
    /// superseded records that were visible at head before this write
    pub hidden: SmallVec<[RecId; 2]>,
}

/// reference_core.Snapshot.lookup / Executor.lookup result
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Lookup<'s> {
    pub status: Status,
    pub value: Option<TypedValue<'s>>,
    /// fact ids of the active assertions for the key, fact-id order
    pub refs: SmallVec<[RecId; 4]>,
    pub reason: &'static str,
}

pub const REASON_NO_ASSERTION: &str = "No active assertion for this exact key";
pub const REASON_DISAGREE: &str = "Active assertions disagree";

pub struct Store {
    cfg: StoreConfig,
    rev: Rev,
    pub pools: Pools,
    pub recs: Records,
    pub docs: Docs,
    key_head: HashMap<(u32, u32, u32), RecId, FixedState>,
    doc_key: HashMap<(u32, u32), DocIx, FixedState>,
    head_visible: u64,
    fp_cache: Mutex<Vec<(Rev, ValidAt, Arc<str>)>>,
    doc_sha: Mutex<Vec<[u8; 32]>>,
    /// last document resolved by add_fact (consecutive facts usually cite the same document)
    last_doc: DocIx,
}

impl Default for Store {
    fn default() -> Self {
        Store::new(StoreConfig::default())
    }
}

impl Clone for Store {
    fn clone(&self) -> Self {
        Store {
            cfg: self.cfg.clone(),
            rev: self.rev,
            pools: self.pools.clone(),
            recs: self.recs.clone(),
            docs: self.docs.clone(),
            key_head: self.key_head.clone(),
            doc_key: self.doc_key.clone(),
            head_visible: self.head_visible,
            fp_cache: Mutex::new(self.fp_cache.lock().unwrap().clone()),
            doc_sha: Mutex::new(self.doc_sha.lock().unwrap().clone()),
            last_doc: self.last_doc,
        }
    }
}

const SEED: u64 = 0x5845_524a_7374_6f72;

impl Store {
    pub fn new(cfg: StoreConfig) -> Store {
        Store {
            cfg,
            rev: 0,
            pools: Pools::default(),
            recs: Records {
                interval: HashMap::with_hasher(FixedState::with_seed(SEED)),
                ..Default::default()
            },
            docs: Docs::default(),
            key_head: HashMap::with_hasher(FixedState::with_seed(SEED)),
            doc_key: HashMap::with_hasher(FixedState::with_seed(SEED)),
            head_visible: 0,
            fp_cache: Mutex::new(Vec::new()),
            doc_sha: Mutex::new(Vec::new()),
            last_doc: NONE,
        }
    }

    /// Reserve capacity for a bulk load of about `records` facts and `docs` documents (fewer rehashes / reallocations).
    pub fn reserve(&mut self, records: usize, docs: usize, text_bytes: usize) {
        let p = &mut self.pools;
        p.fact_ids.reserve(records, records * 40);
        p.strs.reserve(records, records * 16);
        p.doc_ids.reserve(docs, docs * 40);
        p.versions.reserve(docs / 4, docs * 4);
        p.subjects.reserve(records / 4, records * 5);
        let r = &mut self.recs;
        for v in [
            &mut r.subject,
            &mut r.relation,
            &mut r.scope,
            &mut r.value,
            &mut r.doc,
            &mut r.start,
            &mut r.end,
            &mut r.quote,
            &mut r.subject_id,
            &mut r.value_id,
            &mut r.added,
            &mut r.retired,
            &mut r.vis_end,
            &mut r.next_in_key,
        ] {
            v.reserve(records);
        }
        r.kind.reserve(records);
        r.unit.reserve(records);
        r.writer.reserve(records);
        self.key_head.reserve(records);
        let d = &mut self.docs;
        for v in [
            &mut d.doc_id,
            &mut d.version,
            &mut d.added,
            &mut d.withdrawn,
        ] {
            v.reserve(docs);
        }
        d.text_end.reserve(docs);
        d.text.reserve(text_bytes);
        d.ascii.reserve(docs);
        d.recs.reserve(docs);
        self.doc_key.reserve(docs);
    }

    /// Release spare capacity of every column, pool and index (after a bulk load).
    pub fn shrink_to_fit(&mut self) {
        let p = &mut self.pools;
        for pool in [
            &mut p.fact_ids,
            &mut p.subjects,
            &mut p.relations,
            &mut p.strs,
            &mut p.units,
            &mut p.scopes,
            &mut p.ent_ids,
            &mut p.doc_ids,
            &mut p.versions,
            &mut p.writers,
        ] {
            pool.shrink_to_fit();
        }
        let r = &mut self.recs;
        for v in [
            &mut r.subject,
            &mut r.relation,
            &mut r.scope,
            &mut r.value,
            &mut r.doc,
            &mut r.start,
            &mut r.end,
            &mut r.quote,
            &mut r.subject_id,
            &mut r.value_id,
            &mut r.added,
            &mut r.retired,
            &mut r.vis_end,
            &mut r.next_in_key,
        ] {
            v.shrink_to_fit();
        }
        r.kind.shrink_to_fit();
        r.unit.shrink_to_fit();
        r.writer.shrink_to_fit();
        r.interval.shrink_to_fit();
        self.key_head.shrink_to_fit();
        let d = &mut self.docs;
        for v in [
            &mut d.doc_id,
            &mut d.version,
            &mut d.added,
            &mut d.withdrawn,
        ] {
            v.shrink_to_fit();
        }
        d.text_end.shrink_to_fit();
        d.text.shrink_to_fit();
        d.ascii.shrink_to_fit();
        d.recs.shrink_to_fit();
        for v in d.recs.iter_mut() {
            if v.spilled() {
                v.shrink_to_fit();
            }
        }
        self.doc_key.shrink_to_fit();
    }

    pub fn config(&self) -> &StoreConfig {
        &self.cfg
    }
    pub fn set_fingerprint_mode(&mut self, mode: FingerprintMode) {
        self.cfg.fingerprint = mode;
        self.fp_cache.lock().unwrap().clear();
    }
    /// current (head) revision
    #[inline]
    pub fn revision(&self) -> Rev {
        self.rev
    }
    #[inline]
    pub fn num_records(&self) -> usize {
        self.recs.subject.len()
    }
    #[inline]
    pub fn num_docs(&self) -> usize {
        self.docs.doc_id.len()
    }
    /// records visible at the head revision (ignoring validity intervals)
    #[inline]
    pub fn num_head_visible(&self) -> u64 {
        self.head_visible
    }

    // -------------------------------------------------------------------------------------------- accessors
    #[inline]
    pub fn fact_id(&self, r: RecId) -> &str {
        self.pools.fact_ids.get(r)
    }
    #[inline]
    pub fn subject(&self, r: RecId) -> &str {
        self.pools.subjects.get(self.recs.subject[r as usize])
    }
    #[inline]
    pub fn relation(&self, r: RecId) -> &str {
        self.pools.relations.get(self.recs.relation[r as usize])
    }
    #[inline]
    pub fn scope(&self, r: RecId) -> &str {
        self.pools.scopes.get(self.recs.scope[r as usize])
    }
    #[inline]
    pub fn quote(&self, r: RecId) -> &str {
        self.pools.strs.get(self.recs.quote[r as usize])
    }
    #[inline]
    pub fn typed_value(&self, r: RecId) -> TypedValue<'_> {
        let i = r as usize;
        TypedValue {
            kind: self.recs.kind[i],
            value: self.pools.strs.get(self.recs.value[i]),
            unit: self.pools.units.get(self.recs.unit[i] as u32),
        }
    }
    #[inline]
    pub fn subject_id(&self, r: RecId) -> Option<&str> {
        let s = self.recs.subject_id[r as usize];
        (s != NONE).then(|| self.pools.ent_ids.get(s))
    }
    #[inline]
    pub fn value_id(&self, r: RecId) -> Option<&str> {
        let s = self.recs.value_id[r as usize];
        (s != NONE).then(|| self.pools.ent_ids.get(s))
    }
    pub fn writer_version(&self, r: RecId) -> &str {
        self.pools.writers.get(self.recs.writer[r as usize] as u32)
    }
    pub fn rec_doc_key(&self, r: RecId) -> (&str, &str) {
        self.doc_key_str(self.recs.doc[r as usize])
    }
    pub fn doc_key_str(&self, d: DocIx) -> (&str, &str) {
        (
            self.pools.doc_ids.get(self.docs.doc_id[d as usize]),
            self.pools.versions.get(self.docs.version[d as usize]),
        )
    }
    #[inline]
    pub fn doc_text(&self, d: DocIx) -> &str {
        let i = d as usize;
        let s = if i == 0 {
            0
        } else {
            self.docs.text_end[i - 1] as usize
        };
        // SAFETY: whole &str values only
        unsafe { std::str::from_utf8_unchecked(&self.docs.text[s..self.docs.text_end[i] as usize]) }
    }
    /// record id of a fact id (any state)
    #[inline]
    pub fn rec_of(&self, fact_id: &str) -> Option<RecId> {
        self.pools.fact_ids.lookup(fact_id)
    }
    pub fn doc_of(&self, doc_id: &str, version: &str) -> Option<DocIx> {
        let d = self.pools.doc_ids.lookup(doc_id)?;
        let v = self.pools.versions.lookup(version)?;
        self.doc_key.get(&(d, v)).copied()
    }
    /// record visible at the head revision (validity intervals not considered)
    #[inline]
    pub fn head_visible(&self, r: RecId) -> bool {
        self.recs.vis_end[r as usize] == OPEN
    }
    #[inline]
    pub fn doc_head_visible(&self, d: DocIx) -> bool {
        self.docs.withdrawn[d as usize] == OPEN
    }
    /// chain head of a (subject, relation, scope) key (newest record first, then `recs.next_in_key`)
    #[inline]
    pub fn key_chain_head(&self, subject: u32, relation: u32, scope: u32) -> RecId {
        self.key_head
            .get(&(subject, relation, scope))
            .copied()
            .unwrap_or(NONE)
    }

    // -------------------------------------------------------------------------------------------- writes
    fn next_rev(&mut self) -> Result<Rev, ContractError> {
        if self.rev >= OPEN - 1 {
            return Err(ContractError::new(E_CAPACITY));
        }
        self.rev += 1;
        Ok(self.rev)
    }

    /// reference_core.MemoryStore.add_document
    pub fn add_document(
        &mut self,
        doc_id: &str,
        version: &str,
        text: &str,
    ) -> Result<DocIx, ContractError> {
        if doc_id.is_empty() || version.is_empty() {
            return Err(ContractError::new(E_DOC_REQUIRES));
        }
        if self.doc_of(doc_id, version).is_some() {
            return Err(ContractError::new(E_DOC_OVERWRITE));
        }
        if self.docs.doc_id.len() >= NONE as usize - 1 {
            return Err(ContractError::new(E_CAPACITY));
        }
        let rev = self.next_rev()?;
        let d = self.docs.doc_id.len() as DocIx;
        let di = self.pools.doc_ids.intern(doc_id);
        let vi = self.pools.versions.intern(version);
        self.doc_key.insert((di, vi), d);
        self.docs.doc_id.push(di);
        self.docs.version.push(vi);
        self.docs.text.extend_from_slice(text.as_bytes());
        self.docs.text_end.push(self.docs.text.len() as u64);
        self.docs.ascii.push(text.is_ascii());
        self.docs.added.push(rev);
        self.docs.withdrawn.push(OPEN);
        self.docs.recs.push(SmallVec::new());
        Ok(d)
    }

    /// Validates a fact (value typing, span, ids, interval, identity, source availability, quote, supersession) exactly in
    /// the reference's order without changing anything; returns the canonical value and the document.
    fn check_fact<'f>(
        &self,
        f: &FactIn<'f>,
        supersedes: &[&str],
        doc: Option<DocIx>,
    ) -> Result<(Kind, std::borrow::Cow<'f, str>, DocIx, u64), ContractError> {
        use std::borrow::Cow;
        let kind = Kind::parse(f.kind).ok_or(ContractError::new(E_KIND))?;
        let canon: Cow<'f, str> = match (kind, f.value) {
            (Kind::Boolean, ValueIn::Bool(b)) if f.unit.is_empty() => {
                Cow::Borrowed(if b { "true" } else { "false" })
            }
            (Kind::Boolean, _) => return Err(ContractError::new(E_BOOL)),
            (_, ValueIn::Str(s)) if !s.is_empty() => match kind {
                Kind::Date => {
                    value::validate_iso_date(s)?;
                    if !f.unit.is_empty() {
                        return Err(ContractError::new(E_DATE_UNIT));
                    }
                    Cow::Borrowed(s)
                }
                Kind::Number => Cow::Owned(value::canonical_decimal(s)?),
                _ => {
                    if !f.unit.is_empty() {
                        return Err(ContractError::new(E_TEXT_UNIT));
                    }
                    Cow::Borrowed(s)
                }
            },
            _ => return Err(ContractError::new(E_NONEMPTY)),
        };
        // SourceDocument.reference(start, end) -> SourceRef.extract: the span must lie inside the exact text
        let d = doc.ok_or(ContractError::new(E_FACT_SOURCE))?;
        let text = self.doc_text(d);
        let nchars = if self.docs.ascii[d as usize] {
            text.len()
        } else {
            text::char_len(text)
        };
        if !(0 <= f.start && f.start < f.end && f.end as u64 <= nchars as u64) {
            return Err(ContractError::new(E_SPAN));
        }
        // Fact.__post_init__
        if f.fact_id.is_empty()
            || f.subject.is_empty()
            || f.relation.is_empty()
            || f.scope.is_empty()
            || f.writer_version.is_empty()
        {
            return Err(ContractError::new(E_FACT_IDS));
        }
        if let Some(v) = f.valid_from {
            value::validate_iso_date(v)?;
        }
        if let Some(v) = f.valid_to {
            value::validate_iso_date(v)?;
        }
        if let (Some(a), Some(b)) = (f.valid_from, f.valid_to) {
            if a >= b {
                return Err(ContractError::new(E_INTERVAL));
            }
        }
        // MemoryStore.add_fact
        let fid_hash = self.pools.fact_ids.hash_str(f.fact_id);
        if self
            .pools
            .fact_ids
            .lookup_hashed(f.fact_id, fid_hash)
            .is_some()
        {
            return Err(ContractError::new(E_FACT_OVERWRITE));
        }
        if self.docs.withdrawn[d as usize] != OPEN {
            return Err(ContractError::new(E_FACT_SOURCE));
        }
        let (bs, be) = text::char_span_to_bytes(
            text,
            self.docs.ascii[d as usize],
            f.start as usize,
            f.end as usize,
        )
        .ok_or(ContractError::new(E_SPAN))?;
        if &text[bs..be] != f.quote {
            return Err(ContractError::new(E_QUOTE));
        }
        for old in supersedes {
            let r = match self.pools.fact_ids.lookup(old) {
                Some(r) if self.recs.retired[r as usize] == OPEN => r,
                _ => return Err(ContractError::new(E_SUPERSEDE_TARGET)),
            };
            let same = self.subject(r) == f.subject
                && self.relation(r) == f.relation
                && self.scope(r) == f.scope;
            if !same {
                return Err(ContractError::new(E_SUPERSEDE_KEY));
            }
        }
        Ok((kind, canon, d, fid_hash))
    }

    /// reference_core.MemoryStore.add_fact(fact, supersedes)
    pub fn add_fact(
        &mut self,
        f: &FactIn<'_>,
        supersedes: &[&str],
    ) -> Result<AddedFact, ContractError> {
        let cached = self.last_doc;
        let doc = if cached != NONE && self.doc_key_str(cached) == (f.doc_id, f.version) {
            Some(cached)
        } else {
            self.doc_of(f.doc_id, f.version)
        };
        if let Some(d) = doc {
            self.last_doc = d;
        }
        let (kind, canon, d, fid_hash) = self.check_fact(f, supersedes, doc)?;
        if self.recs.subject.len() >= NONE as usize - 1
            || self.pools.units.len() >= u16::MAX as usize
            || self.pools.writers.len() >= u16::MAX as usize
        {
            return Err(ContractError::new(E_CAPACITY));
        }
        let rev = self.next_rev()?;
        let rid = match self.pools.fact_ids.insert_new_hashed(f.fact_id, fid_hash) {
            Ok(r) => r,
            Err(_) => unreachable!("checked"),
        };
        debug_assert_eq!(rid as usize, self.recs.subject.len());
        let s = self.pools.subjects.intern(f.subject);
        let rl = self.pools.relations.intern(f.relation);
        let sc = self.pools.scopes.intern(f.scope);
        let v = self.pools.strs.intern(&canon);
        let u = self.pools.units.intern(f.unit) as u16;
        let q = self.pools.strs.intern(f.quote);
        let sid = f.subject_id.map_or(NONE, |x| self.pools.ent_ids.intern(x));
        let vid = f.value_id.map_or(NONE, |x| self.pools.ent_ids.intern(x));
        let w = self.pools.writers.intern(f.writer_version) as u16;
        let r = &mut self.recs;
        r.subject.push(s);
        r.relation.push(rl);
        r.scope.push(sc);
        r.kind.push(kind);
        r.value.push(v);
        r.unit.push(u);
        r.doc.push(d);
        r.start.push(f.start as u32);
        r.end.push(f.end as u32);
        r.quote.push(q);
        r.subject_id.push(sid);
        r.value_id.push(vid);
        r.writer.push(w);
        r.added.push(rev);
        r.retired.push(OPEN);
        r.vis_end.push(OPEN);
        let head = self.key_head.insert((s, rl, sc), rid).unwrap_or(NONE);
        self.recs.next_in_key.push(head);
        if f.valid_from.is_some() || f.valid_to.is_some() {
            let a = f.valid_from.map_or(NONE, |x| self.pools.strs.intern(x));
            let b = f.valid_to.map_or(NONE, |x| self.pools.strs.intern(x));
            self.recs.interval.insert(rid, (a, b));
        }
        self.docs.recs[d as usize].push(rid);
        self.head_visible += 1;
        let mut hidden = SmallVec::new();
        for old in supersedes {
            let o = self.pools.fact_ids.lookup(old).expect("checked") as usize;
            self.recs.retired[o] = rev;
            if self.recs.vis_end[o] == OPEN {
                hidden.push(o as RecId);
                self.head_visible -= 1;
                self.recs.vis_end[o] = rev;
            }
        }
        Ok(AddedFact {
            rec: rid,
            rev,
            hidden,
        })
    }

    /// reference_core.MemoryStore.withdraw_fact; returns the record and whether it was visible at head
    pub fn withdraw_fact(&mut self, fact_id: &str) -> Result<(RecId, bool), ContractError> {
        let r = match self.pools.fact_ids.lookup(fact_id) {
            Some(r) if self.recs.retired[r as usize] == OPEN => r,
            _ => return Err(ContractError::new(E_FACT_WITHDRAW)),
        };
        let rev = self.next_rev()?;
        let i = r as usize;
        self.recs.retired[i] = rev;
        let was = self.recs.vis_end[i] == OPEN;
        if was {
            self.recs.vis_end[i] = rev;
            self.head_visible -= 1;
        }
        Ok((r, was))
    }

    /// reference_core.MemoryStore.withdraw_document; returns the document and the records it hid at head
    pub fn withdraw_document(
        &mut self,
        doc_id: &str,
        version: &str,
    ) -> Result<(DocIx, SmallVec<[RecId; 8]>), ContractError> {
        let d = match self.doc_of(doc_id, version) {
            Some(d) if self.docs.withdrawn[d as usize] == OPEN => d,
            _ => return Err(ContractError::new(E_DOC_WITHDRAW)),
        };
        let rev = self.next_rev()?;
        self.docs.withdrawn[d as usize] = rev;
        let mut hidden = SmallVec::new();
        for k in 0..self.docs.recs[d as usize].len() {
            let r = self.docs.recs[d as usize][k];
            if self.recs.vis_end[r as usize] == OPEN {
                self.recs.vis_end[r as usize] = rev;
                self.head_visible -= 1;
                hidden.push(r);
            }
        }
        Ok((d, hidden))
    }

    // -------------------------------------------------------------------------------------------- snapshots
    /// reference_core.MemoryStore.snapshot(valid_at=, revision=): O(1)
    pub fn snapshot(
        &self,
        valid_at: &str,
        revision: Option<Rev>,
    ) -> Result<Snapshot<'_>, ContractError> {
        let va = ValidAt::parse(valid_at)?;
        let rev = revision.unwrap_or(self.rev);
        if rev > self.rev {
            return Err(ContractError::new(E_SNAPSHOT_REV));
        }
        Ok(Snapshot {
            store: self,
            rev,
            valid_at: va,
        })
    }

    /// head snapshot at xr_codecs.VALID_AT
    pub fn head(&self) -> Snapshot<'_> {
        Snapshot {
            store: self,
            rev: self.rev,
            valid_at: ValidAt::parse(VALID_AT).unwrap(),
        }
    }

    pub fn at(&self, rev: Rev, valid_at: ValidAt) -> Snapshot<'_> {
        assert!(rev <= self.rev, "revision beyond head");
        Snapshot {
            store: self,
            rev,
            valid_at,
        }
    }

    fn ensure_doc_sha(&self) -> std::sync::MutexGuard<'_, Vec<[u8; 32]>> {
        let mut g = self.doc_sha.lock().unwrap();
        let n = self.num_docs();
        while g.len() < n {
            let d = g.len() as DocIx;
            let h: [u8; 32] = Sha256::digest(self.doc_text(d).as_bytes()).into();
            g.push(h);
        }
        g
    }

    /// sha256 hex of a document's exact UTF-8 text (SourceDocument.sha256)
    pub fn doc_sha256_hex(&self, d: DocIx) -> String {
        hex32(&self.ensure_doc_sha()[d as usize])
    }

    fn fingerprint(&self, rev: Rev, va: ValidAt) -> Arc<str> {
        {
            let c = self.fp_cache.lock().unwrap();
            if let Some((_, _, fp)) = c.iter().find(|(r, v, _)| *r == rev && *v == va) {
                return fp.clone();
            }
        }
        let fp: Arc<str> = match self.cfg.fingerprint {
            FingerprintMode::Exact => self.exact_fingerprint(rev, va).into(),
            FingerprintMode::Revision => {
                let s = format!(
                    "xerj-mem/revision-fp/v1|{}|{}|{}",
                    self.cfg.store_id,
                    rev,
                    va.as_str()
                );
                hex32(&Sha256::digest(s.as_bytes()).into()).into()
            }
        };
        let mut c = self.fp_cache.lock().unwrap();
        if c.len() >= 16 {
            c.remove(0);
        }
        c.push((rev, va, fp.clone()));
        fp
    }

    /// reference_core.Snapshot.fingerprint: digest({"revision", "valid_at", "documents": [(doc_id, version, sha256)],
    /// "facts": [asdict(f)]}) with json.dumps(sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    fn exact_fingerprint(&self, rev: Rev, va: ValidAt) -> String {
        let snap = Snapshot {
            store: self,
            rev,
            valid_at: va,
        };
        let mut docs: Vec<DocIx> = (0..self.num_docs() as DocIx)
            .filter(|&d| {
                self.docs.added[d as usize] <= rev && rev < self.docs.withdrawn[d as usize]
            })
            .collect();
        docs.sort_unstable_by(|&a, &b| self.doc_key_str(a).cmp(&self.doc_key_str(b)));
        let facts = snap.facts_sorted();
        let shas = self.ensure_doc_sha();
        let mut h = Sha256::new();
        let mut buf: Vec<u8> = Vec::with_capacity(1 << 16);
        let mut itoa = itoa_buf();
        buf.extend_from_slice(b"{\"documents\":[");
        for (k, &d) in docs.iter().enumerate() {
            if k > 0 {
                buf.push(b',');
            }
            let (di, vi) = self.doc_key_str(d);
            buf.push(b'[');
            text::py_json_str(di, &mut buf);
            buf.push(b',');
            text::py_json_str(vi, &mut buf);
            buf.extend_from_slice(b",\"");
            push_hex32(&shas[d as usize], &mut buf);
            buf.extend_from_slice(b"\"]");
            if buf.len() > 60_000 {
                h.update(&buf);
                buf.clear();
            }
        }
        buf.extend_from_slice(b"],\"facts\":[");
        for (k, &r) in facts.iter().enumerate() {
            if k > 0 {
                buf.push(b',');
            }
            self.write_fact_json(r, &shas, &mut buf, &mut itoa);
            if buf.len() > 60_000 {
                h.update(&buf);
                buf.clear();
            }
        }
        buf.extend_from_slice(b"],\"revision\":");
        buf.extend_from_slice(fmt_u64(&mut itoa, rev as u64));
        buf.extend_from_slice(b",\"valid_at\":");
        text::py_json_str(va.as_str(), &mut buf);
        buf.push(b'}');
        h.update(&buf);
        hex32(&h.finalize().into())
    }

    /// json.dumps(asdict(fact), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    fn write_fact_json(&self, r: RecId, shas: &[[u8; 32]], buf: &mut Vec<u8>, itoa: &mut [u8; 24]) {
        let i = r as usize;
        let d = self.recs.doc[i];
        let (doc_id, version) = self.doc_key_str(d);
        buf.extend_from_slice(b"{\"fact_id\":");
        text::py_json_str(self.fact_id(r), buf);
        buf.extend_from_slice(b",\"quote\":");
        text::py_json_str(self.quote(r), buf);
        buf.extend_from_slice(b",\"relation\":");
        text::py_json_str(self.relation(r), buf);
        buf.extend_from_slice(b",\"scope\":");
        text::py_json_str(self.scope(r), buf);
        buf.extend_from_slice(b",\"source\":{\"doc_id\":");
        text::py_json_str(doc_id, buf);
        buf.extend_from_slice(b",\"end\":");
        buf.extend_from_slice(fmt_u64(itoa, self.recs.end[i] as u64));
        buf.extend_from_slice(b",\"sha256\":\"");
        push_hex32(&shas[d as usize], buf);
        buf.extend_from_slice(b"\",\"start\":");
        buf.extend_from_slice(fmt_u64(itoa, self.recs.start[i] as u64));
        buf.extend_from_slice(b",\"version\":");
        text::py_json_str(version, buf);
        buf.extend_from_slice(b"},\"subject\":");
        text::py_json_str(self.subject(r), buf);
        let (vf, vt) = self.recs.interval.get(&r).copied().unwrap_or((NONE, NONE));
        buf.extend_from_slice(b",\"valid_from\":");
        if vf == NONE {
            buf.extend_from_slice(b"null");
        } else {
            text::py_json_str(self.pools.strs.get(vf), buf);
        }
        buf.extend_from_slice(b",\"valid_to\":");
        if vt == NONE {
            buf.extend_from_slice(b"null");
        } else {
            text::py_json_str(self.pools.strs.get(vt), buf);
        }
        let tv = self.typed_value(r);
        buf.extend_from_slice(b",\"value\":{\"kind\":");
        text::py_json_str(tv.kind.as_str(), buf);
        buf.extend_from_slice(b",\"unit\":");
        text::py_json_str(tv.unit, buf);
        buf.extend_from_slice(b",\"value\":");
        if tv.kind == Kind::Boolean {
            buf.extend_from_slice(tv.value.as_bytes());
        } else {
            text::py_json_str(tv.value, buf);
        }
        buf.extend_from_slice(b"},\"writer_version\":");
        text::py_json_str(self.writer_version(r), buf);
        buf.push(b'}');
    }

    /// Approximate heap bytes of the store (pools, columns, documents, key index).
    pub fn heap_bytes(&self) -> usize {
        let p = &self.pools;
        let pools = p.fact_ids.heap_bytes()
            + p.subjects.heap_bytes()
            + p.relations.heap_bytes()
            + p.strs.heap_bytes()
            + p.units.heap_bytes()
            + p.scopes.heap_bytes()
            + p.ent_ids.heap_bytes()
            + p.doc_ids.heap_bytes()
            + p.versions.heap_bytes()
            + p.writers.heap_bytes();
        let r = &self.recs;
        let cols = (r.subject.capacity()
            + r.relation.capacity()
            + r.scope.capacity()
            + r.value.capacity()
            + r.doc.capacity()
            + r.start.capacity()
            + r.end.capacity()
            + r.quote.capacity()
            + r.subject_id.capacity()
            + r.value_id.capacity()
            + r.added.capacity()
            + r.retired.capacity()
            + r.vis_end.capacity()
            + r.next_in_key.capacity())
            * 4
            + (r.unit.capacity() + r.writer.capacity()) * 2
            + r.kind.capacity()
            + r.interval.capacity() * 13;
        let d = &self.docs;
        let docs = (d.doc_id.capacity()
            + d.version.capacity()
            + d.added.capacity()
            + d.withdrawn.capacity())
            * 4
            + d.text_end.capacity() * 8
            + d.text.capacity()
            + d.ascii.capacity()
            + d.recs.capacity() * std::mem::size_of::<SmallVec<[RecId; 3]>>()
            + d.recs
                .iter()
                .filter(|v| v.spilled())
                .map(|v| v.capacity() * 4)
                .sum::<usize>();
        let idx = self.key_head.capacity() * (12 + 4 + 1) + self.doc_key.capacity() * (8 + 4 + 1);
        pools + cols + docs + idx + self.doc_sha.lock().unwrap().capacity() * 32
    }

    // -------------------------------------------------------------------------------------------- persistence support
    pub(crate) fn raw_parts(&self) -> (Rev, &StoreConfig) {
        (self.rev, &self.cfg)
    }

    /// Rebuild a store from loaded columns (checkpoint): recomputes the key and document indexes.
    pub(crate) fn from_parts(
        cfg: StoreConfig,
        rev: Rev,
        pools: Pools,
        mut recs: Records,
        docs: Docs,
    ) -> Result<Store, String> {
        let n = recs.subject.len();
        if pools.fact_ids.len() != n {
            return Err("fact id pool and record columns disagree".into());
        }
        let mut key_head: HashMap<(u32, u32, u32), RecId, FixedState> =
            HashMap::with_capacity_and_hasher(n / 2 + 1, FixedState::with_seed(SEED));
        recs.next_in_key.clear();
        recs.next_in_key.reserve(n);
        let mut head_visible = 0u64;
        for r in 0..n {
            let head = key_head
                .insert(
                    (recs.subject[r], recs.relation[r], recs.scope[r]),
                    r as RecId,
                )
                .unwrap_or(NONE);
            recs.next_in_key.push(head);
            if recs.vis_end[r] == OPEN {
                head_visible += 1;
            }
        }
        let mut doc_key: HashMap<(u32, u32), DocIx, FixedState> =
            HashMap::with_capacity_and_hasher(docs.doc_id.len(), FixedState::with_seed(SEED));
        for d in 0..docs.doc_id.len() {
            doc_key.insert((docs.doc_id[d], docs.version[d]), d as DocIx);
        }
        Ok(Store {
            cfg,
            rev,
            pools,
            recs,
            docs,
            key_head,
            doc_key,
            head_visible,
            fp_cache: Mutex::new(Vec::new()),
            doc_sha: Mutex::new(Vec::new()),
            last_doc: NONE,
        })
    }
}

#[inline]
fn itoa_buf() -> [u8; 24] {
    [0u8; 24]
}

#[inline]
fn fmt_u64(buf: &mut [u8; 24], mut v: u64) -> &[u8] {
    let mut i = buf.len();
    if v == 0 {
        i -= 1;
        buf[i] = b'0';
    }
    while v > 0 {
        i -= 1;
        buf[i] = b'0' + (v % 10) as u8;
        v /= 10;
    }
    &buf[i..]
}

const HEX: &[u8; 16] = b"0123456789abcdef";

pub fn push_hex32(h: &[u8; 32], out: &mut Vec<u8>) {
    for &b in h {
        out.push(HEX[(b >> 4) as usize]);
        out.push(HEX[(b & 15) as usize]);
    }
}

pub fn hex32(h: &[u8; 32]) -> String {
    let mut v = Vec::with_capacity(64);
    push_hex32(h, &mut v);
    // SAFETY: hex digits
    unsafe { String::from_utf8_unchecked(v) }
}

/// A read view of the store at one revision and valid_at date (reference_core.Snapshot without copying anything).
#[derive(Clone, Copy)]
pub struct Snapshot<'s> {
    pub store: &'s Store,
    pub rev: Rev,
    pub valid_at: ValidAt,
}

impl<'s> Snapshot<'s> {
    /// fact visible: added <= rev < min(retired, document withdrawal), and valid at `valid_at`
    #[inline]
    pub fn visible(&self, r: RecId) -> bool {
        let i = r as usize;
        let recs = &self.store.recs;
        if !(recs.added[i] <= self.rev && self.rev < recs.vis_end[i]) {
            return false;
        }
        if recs.interval.is_empty() {
            return true;
        }
        self.interval_ok(r)
    }

    #[cold]
    fn interval_ok(&self, r: RecId) -> bool {
        match self.store.recs.interval.get(&r) {
            None => true,
            Some(&(a, b)) => {
                let when = self.valid_at.as_str();
                (a == NONE || self.store.pools.strs.get(a) <= when)
                    && (b == NONE || when < self.store.pools.strs.get(b))
            }
        }
    }

    #[inline]
    pub fn doc_visible(&self, d: DocIx) -> bool {
        let i = d as usize;
        self.store.docs.added[i] <= self.rev && self.rev < self.store.docs.withdrawn[i]
    }

    /// visible records in id (insertion) order: O(N) scan
    pub fn visible_records(&self) -> Vec<RecId> {
        (0..self.store.num_records() as RecId)
            .filter(|&r| self.visible(r))
            .collect()
    }

    /// visible records in the reference's snapshot order (sorted by fact id)
    pub fn facts_sorted(&self) -> Vec<RecId> {
        let mut v = self.visible_records();
        let p = &self.store.pools.fact_ids;
        v.sort_unstable_by(|&a, &b| p.get(a).cmp(p.get(b)));
        v
    }

    /// reference_core.Snapshot.fingerprint (or the configured Revision fingerprint), cached per (revision, valid_at)
    pub fn fingerprint(&self) -> Arc<str> {
        self.store.fingerprint(self.rev, self.valid_at)
    }

    /// reference_core.Snapshot.lookup(subject, relation, scope)
    pub fn lookup(&self, subject: &str, relation: &str, scope: &str) -> Lookup<'s> {
        let p = &self.store.pools;
        match (
            p.subjects.lookup(subject),
            p.relations.lookup(relation),
            p.scopes.lookup(scope),
        ) {
            (Some(s), Some(r), Some(c)) => self.lookup_key(s, r, c),
            _ => Lookup {
                status: Status::Unknown,
                value: None,
                refs: SmallVec::new(),
                reason: REASON_NO_ASSERTION,
            },
        }
    }

    /// lookup of the full key of record `r` (the stack runner's LOOKUP of a candidate)
    pub fn lookup_record_key(&self, r: RecId) -> Lookup<'s> {
        let i = r as usize;
        let recs = &self.store.recs;
        self.lookup_key(recs.subject[i], recs.relation[i], recs.scope[i])
    }

    pub fn lookup_key(&self, subject: u32, relation: u32, scope: u32) -> Lookup<'s> {
        let st = self.store;
        let mut refs: SmallVec<[RecId; 4]> = SmallVec::new();
        let mut r = st.key_chain_head(subject, relation, scope);
        while r != NONE {
            if self.visible(r) {
                refs.push(r);
            }
            r = st.recs.next_in_key[r as usize];
        }
        if refs.is_empty() {
            return Lookup {
                status: Status::Unknown,
                value: None,
                refs,
                reason: REASON_NO_ASSERTION,
            };
        }
        if refs.len() > 1 {
            let p = &st.pools.fact_ids;
            refs.sort_unstable_by(|&a, &b| p.get(a).cmp(p.get(b)));
        }
        let first = refs[0] as usize;
        let key = (
            st.recs.kind[first],
            st.recs.value[first],
            st.recs.unit[first],
        );
        let agree = refs.iter().all(|&x| {
            let i = x as usize;
            (st.recs.kind[i], st.recs.value[i], st.recs.unit[i]) == key
        });
        if !agree {
            return Lookup {
                status: Status::Conflict,
                value: None,
                refs,
                reason: REASON_DISAGREE,
            };
        }
        Lookup {
            status: Status::Value,
            value: Some(st.typed_value(refs[0])),
            refs,
            reason: "",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store_ab() -> Store {
        // reference_core.smoke_demo
        let mut s = Store::default();
        for (name, deadline) in [("A", "2026-11-02"), ("B", "2026-11-14")] {
            let text = format!("Project {name} deadline is {deadline}.");
            s.add_document(name, "v1", &text).unwrap();
            let start = text.find(deadline).unwrap() as i64;
            let fid = format!("f-{name}");
            let f = FactIn {
                kind: "date",
                writer_version: FIXTURE_WRITER,
                ..FactIn::text(
                    &fid,
                    name,
                    "deadline",
                    deadline,
                    (name, "v1"),
                    (start, start + 10),
                    deadline,
                )
            };
            s.add_fact(&f, &[]).unwrap();
        }
        s
    }

    #[test]
    fn smoke_demo_fingerprint_matches_python() {
        let s = store_ab();
        let snap = s.snapshot("2026-10-20", None).unwrap();
        // python3 reference_core.py --smoke -> snapshot_fingerprint (reference_pkg frozen copy, 2026-10-21)
        assert_eq!(
            &*snap.fingerprint(),
            "5fa3b225e59c9c169f2986627174e41dce3d006f507947b49d7b91ae273b7dc3"
        );
        let l = snap.lookup("A", "deadline", "default");
        assert_eq!(l.status, Status::Value);
        assert_eq!(l.value.unwrap().value, "2026-11-02");
    }

    #[test]
    fn mvcc_supersede_withdraw() {
        let mut s = Store::default();
        s.add_document("d1", "v1", "X is red. Y is blue.").unwrap();
        s.add_fact(
            &FactIn::text("f1", "X", "color", "red", ("d1", "v1"), (5, 8), "red"),
            &[],
        )
        .unwrap();
        s.add_fact(
            &FactIn::text("f2", "Y", "color", "blue", ("d1", "v1"), (15, 19), "blue"),
            &[],
        )
        .unwrap();
        let base = s.revision();
        s.add_document("d1", "v2", "X is green.").unwrap();
        let a = s
            .add_fact(
                &FactIn::text("f3", "X", "color", "green", ("d1", "v2"), (5, 10), "green"),
                &["f1"],
            )
            .unwrap();
        assert_eq!(a.hidden.as_slice(), &[0]);
        assert_eq!(
            s.add_fact(
                &FactIn::text("f4", "X", "color", "green", ("d1", "v2"), (5, 10), "green"),
                &["f1"]
            )
            .unwrap_err()
            .message(),
            E_SUPERSEDE_TARGET
        );
        assert_eq!(
            s.add_fact(
                &FactIn::text("f3", "X", "color", "green", ("d1", "v2"), (5, 10), "green"),
                &[]
            )
            .unwrap_err()
            .message(),
            E_FACT_OVERWRITE
        );
        assert_eq!(
            s.add_fact(
                &FactIn::text("f5", "X", "color", "gree", ("d1", "v2"), (5, 10), "gree"),
                &[]
            )
            .unwrap_err()
            .message(),
            E_QUOTE
        );
        let va = ValidAt::parse(VALID_AT).unwrap();
        assert_eq!(
            s.at(base, va)
                .lookup("X", "color", "default")
                .value
                .unwrap()
                .value,
            "red"
        );
        assert_eq!(
            s.head()
                .lookup("X", "color", "default")
                .value
                .unwrap()
                .value,
            "green"
        );
        let (_, hid) = s.withdraw_document("d1", "v1").unwrap();
        assert_eq!(hid.as_slice(), &[1]);
        assert_eq!(
            s.head().lookup("Y", "color", "default").status,
            Status::Unknown
        );
        assert_eq!(
            s.at(base, va).lookup("Y", "color", "default").status,
            Status::Value
        );
        assert_eq!(
            s.withdraw_document("d1", "v1").unwrap_err().message(),
            E_DOC_WITHDRAW
        );
        // conflict: a second active assertion with another value
        s.add_document("d9", "v1", "X is teal.").unwrap();
        s.add_fact(
            &FactIn::text("f6", "X", "color", "teal", ("d9", "v1"), (5, 9), "teal"),
            &[],
        )
        .unwrap();
        let l = s.head().lookup("X", "color", "default");
        assert_eq!(l.status, Status::Conflict);
        assert_eq!(
            l.refs.iter().map(|&r| s.fact_id(r)).collect::<Vec<_>>(),
            ["f3", "f6"]
        );
        assert_eq!(s.num_head_visible(), 2);
    }

    #[test]
    fn numbers_canonical_equal_values_do_not_conflict() {
        let mut s = Store::default();
        s.add_document("d", "v1", "185 and 185.0").unwrap();
        let mk = |id: &'static str, v: &'static str, a: i64, b: i64| FactIn {
            kind: "number",
            unit: "cm",
            ..FactIn::text(id, "P", "height", v, ("d", "v1"), (a, b), v)
        };
        s.add_fact(&mk("h1", "185", 0, 3), &[]).unwrap();
        s.add_fact(&mk("h2", "185.0", 8, 13), &[]).unwrap();
        let l = s.head().lookup("P", "height", "default");
        assert_eq!(l.status, Status::Value);
        assert_eq!(l.value.unwrap().value, "185E0");
    }
}
