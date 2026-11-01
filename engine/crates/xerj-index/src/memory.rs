//! `Memory`: the store plus every index, kept consistent on each write (read-your-writes: a write is visible to the
//! next query with no rebuild), with optional op log and checkpoints.
//!
//! Index maintenance per write (all O(1) amortized): subject / relation / entity-id lists (append), linker vocabulary
//! shape (new subjects only), BM25 over record text (add; superseded / withdrawn records deleted) and over documents
//! (add; withdrawn documents deleted). BM25 reflects the HEAD revision (as a production search index); snapshot reads
//! at older revisions use the exact store / manifest paths, never BM25 statistics.

use crate::bm25::{Bm25Backend, Bm25Params, Hit, SegmentedBm25};
use crate::direction::{self, CandManifest, ValueIndex, ValueTables};
use crate::linker::{link_normalized, Linker, LinkerKind, QBuf};
use crate::manifest::{self, HopManifest, ManifestHash, ManifestStats};
use crate::subjects::SubjectIndex;
use smallvec::SmallVec;
use std::path::{Path, PathBuf};
use std::sync::RwLockReadGuard;
use xerj_store::checkpoint::{read_store, write_store};
use xerj_store::ckpt::{SectionWriter, Sections};
use xerj_store::log::{replay, LogWriter, Op};
use xerj_store::value::render_value_into;
use xerj_store::world::{ingest_world, World, WorldRevs, WriteSink};
use xerj_store::{
    AddedFact, ContractError, DocIx, FactIn, RecId, Rev, Snapshot, Store, StoreConfig, StoreError,
    TypedValue, ValidAt, VALID_AT,
};

/// xr_codecs.REL_PHRASE (relation -> question phrase; other relations are their own phrase)
pub fn rel_phrase(relation: &str) -> &str {
    match relation {
        "birth_date" => "date of birth",
        "height" => "height",
        "nationality" => "nationality",
        "occupation" => "occupation",
        "founding_date" => "founding date",
        "employees" => "number of employees",
        "headquarters" => "headquarters city",
        "completion_date" => "completion date",
        "city" => "city",
        "length" => "length",
        "discharge" => "discharge",
        "country" => "country",
        other => other,
    }
}

/// Which text of a record the record BM25 index sees.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RecordText {
    /// reference_core.LexicalIndex: f"{subject} {relation} {quote}" (parity tests)
    Reference,
    /// subject + relation phrase + rendered value + quote (default)
    Rich,
}

#[derive(Clone, Debug)]
pub struct MemoryConfig {
    pub store: StoreConfig,
    pub linker: LinkerKind,
    pub manifest_hash: ManifestHash,
    pub bm25_records: bool,
    pub bm25_documents: bool,
    pub record_text: RecordText,
    pub bm25: Bm25Params,
    pub bm25_seal_postings: usize,
    pub background_merges: bool,
    /// maintain BM25 on one indexer thread (read-your-writes kept by a read barrier); inline when false
    pub async_bm25: bool,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        MemoryConfig {
            store: StoreConfig::default(),
            linker: LinkerKind::SpanHash,
            manifest_hash: ManifestHash::Sha256,
            bm25_records: true,
            bm25_documents: true,
            record_text: RecordText::Rich,
            bm25: Bm25Params::default(),
            bm25_seal_postings: 1 << 20,
            background_merges: false,
            async_bm25: false,
        }
    }
}

const T_BM25_REC: u32 = 0x10000;
const T_BM25_DOC: u32 = 0x20000;
const CKPT_FILE: &str = "memory.ckp";
const LOG_FILE: &str = "ops.log";

pub struct Memory {
    pub cfg: MemoryConfig,
    pub store: Store,
    pub subjects: SubjectIndex,
    /// stack-v5 value-side tables (text value / value_id / normalized value -> records, linkable-value shape); built lazily
    /// on the first stack-v5 read and caught up incrementally, never touched by v1 / v2 reads or by writes
    pub values: ValueIndex,
    pub linker: Linker,
    pub bm25_rec: Option<Box<dyn Bm25Backend>>,
    pub bm25_doc: Option<Box<dyn Bm25Backend>>,
    /// async mode: both backends live on the indexer thread (bm25_rec / bm25_doc are then None)
    bm25_async: Option<crate::indexer::AsyncBm25>,
    log: Option<LogWriter>,
    dir: Option<PathBuf>,
    text_buf: String,
}

fn new_bm25(cfg: &MemoryConfig) -> Box<dyn Bm25Backend> {
    let mut b = SegmentedBm25::new(cfg.bm25);
    b.seal_postings = cfg.bm25_seal_postings;
    if cfg.background_merges {
        b = b.with_background_merges();
    }
    Box::new(b)
}

/// record text for BM25 (allocation reused through `buf`)
pub fn record_text_into(store: &Store, r: RecId, mode: RecordText, buf: &mut String) {
    buf.clear();
    buf.push_str(store.subject(r));
    buf.push(' ');
    match mode {
        RecordText::Reference => buf.push_str(store.relation(r)),
        RecordText::Rich => {
            buf.push_str(rel_phrase(store.relation(r)));
            buf.push(' ');
            render_value_into(&store.typed_value(r), buf);
        }
    }
    buf.push(' ');
    buf.push_str(store.quote(r));
}

impl Memory {
    pub fn new(cfg: MemoryConfig) -> Memory {
        let store = Store::new(cfg.store.clone());
        let mut bm25_rec = cfg.bm25_records.then(|| new_bm25(&cfg));
        let mut bm25_doc = cfg.bm25_documents.then(|| new_bm25(&cfg));
        let bm25_async = cfg
            .async_bm25
            .then(|| crate::indexer::AsyncBm25::new(bm25_rec.take(), bm25_doc.take()));
        Memory {
            linker: Linker::new(cfg.linker),
            cfg,
            store,
            subjects: SubjectIndex::new(),
            values: ValueIndex::new(),
            bm25_rec,
            bm25_doc,
            bm25_async,
            log: None,
            dir: None,
            text_buf: String::new(),
        }
    }

    /// run `f` on the record backend (after the async read barrier)
    pub fn with_record_index<R>(&self, f: impl FnOnce(&dyn Bm25Backend) -> R) -> Option<R> {
        match &self.bm25_async {
            Some(a) => a.records().as_deref().map(f),
            None => self.bm25_rec.as_deref().map(f),
        }
    }

    /// run `f` on the document backend (after the async read barrier)
    pub fn with_document_index<R>(&self, f: impl FnOnce(&dyn Bm25Backend) -> R) -> Option<R> {
        match &self.bm25_async {
            Some(a) => a.documents().as_deref().map(f),
            None => self.bm25_doc.as_deref().map(f),
        }
    }

    fn with_backends_mut(&mut self, mut f: impl FnMut(&mut dyn Bm25Backend, bool)) {
        match &self.bm25_async {
            Some(a) => {
                if let Some(b) = a.records().as_deref_mut() {
                    f(b, true);
                }
                if let Some(b) = a.documents().as_deref_mut() {
                    f(b, false);
                }
            }
            None => {
                if let Some(b) = self.bm25_rec.as_deref_mut() {
                    f(b, true);
                }
                if let Some(b) = self.bm25_doc.as_deref_mut() {
                    f(b, false);
                }
            }
        }
    }

    /// Install loaded backends (keeps keys absent from the checkpoint as built by `new`), honouring async mode.
    fn install_backends(
        &mut self,
        rec: Option<Box<dyn Bm25Backend>>,
        doc: Option<Box<dyn Bm25Backend>>,
    ) {
        let (mut r0, mut d0) = match self.bm25_async.take() {
            Some(a) => a.into_backends(),
            None => (self.bm25_rec.take(), self.bm25_doc.take()),
        };
        if rec.is_some() {
            r0 = rec;
        }
        if doc.is_some() {
            d0 = doc;
        }
        if self.cfg.async_bm25 {
            self.bm25_async = Some(crate::indexer::AsyncBm25::new(r0, d0));
        } else {
            self.bm25_rec = r0;
            self.bm25_doc = d0;
        }
    }

    /// Replace a BM25 backend (e.g. a tantivy adapter); must be called on an empty memory.
    pub fn set_record_backend(&mut self, b: Box<dyn Bm25Backend>) {
        assert_eq!(self.store.num_records(), 0, "swap backends before writing");
        assert!(self.bm25_async.is_none(), "swap backends in inline mode");
        self.bm25_rec = Some(b);
    }
    pub fn set_document_backend(&mut self, b: Box<dyn Bm25Backend>) {
        assert_eq!(self.store.num_docs(), 0, "swap backends before writing");
        assert!(self.bm25_async.is_none(), "swap backends in inline mode");
        self.bm25_doc = Some(b);
    }

    fn rec_text(&mut self, r: RecId) -> String {
        let mut buf = std::mem::take(&mut self.text_buf);
        record_text_into(&self.store, r, self.cfg.record_text, &mut buf);
        buf
    }

    // -------------------------------------------------------------------------------------------- writes
    pub fn add_document(
        &mut self,
        doc_id: &str,
        version: &str,
        text: &str,
    ) -> Result<DocIx, ContractError> {
        let d = self.store.add_document(doc_id, version, text)?;
        if let Some(a) = &self.bm25_async {
            a.add_document(d, text);
        } else if let Some(b) = self.bm25_doc.as_mut() {
            b.add(d, text);
        }
        if let Some(l) = self.log.as_mut() {
            l.append(&Op::AddDocument {
                doc_id,
                version,
                text,
            });
        }
        Ok(d)
    }

    pub fn add_fact(
        &mut self,
        f: &FactIn<'_>,
        supersedes: &[&str],
    ) -> Result<AddedFact, ContractError> {
        let a = self.store.add_fact(f, supersedes)?;
        self.subjects.add_record(&self.store, a.rec);
        if self.store.pools.subjects.len() > self.linker.shape.seen {
            self.linker.catch_up(&self.store);
        }
        if self.bm25_async.is_some() && self.cfg.bm25_records {
            let t = self.rec_text(a.rec);
            self.bm25_async.as_ref().unwrap().add_record(a.rec, &t);
            self.text_buf = t;
            for &h in &a.hidden {
                let t = self.rec_text(h);
                self.bm25_async.as_ref().unwrap().delete_record(h, &t);
                self.text_buf = t;
            }
        } else if self.bm25_rec.is_some() {
            let t = self.rec_text(a.rec);
            self.bm25_rec.as_mut().unwrap().add(a.rec, &t);
            self.text_buf = t;
            for &h in &a.hidden {
                let t = self.rec_text(h);
                self.bm25_rec.as_mut().unwrap().delete(h, &t);
                self.text_buf = t;
            }
        }
        if let Some(l) = self.log.as_mut() {
            l.append(&Op::AddFact {
                fact: f.clone(),
                supersedes: supersedes.iter().copied().collect(),
            });
        }
        Ok(a)
    }

    pub fn withdraw_fact(&mut self, fact_id: &str) -> Result<(RecId, bool), ContractError> {
        let (r, was) = self.store.withdraw_fact(fact_id)?;
        if was && self.bm25_async.is_some() && self.cfg.bm25_records {
            let t = self.rec_text(r);
            self.bm25_async.as_ref().unwrap().delete_record(r, &t);
            self.text_buf = t;
        } else if was && self.bm25_rec.is_some() {
            let t = self.rec_text(r);
            self.bm25_rec.as_mut().unwrap().delete(r, &t);
            self.text_buf = t;
        }
        if let Some(l) = self.log.as_mut() {
            l.append(&Op::WithdrawFact { fact_id });
        }
        Ok((r, was))
    }

    pub fn withdraw_document(
        &mut self,
        doc_id: &str,
        version: &str,
    ) -> Result<(DocIx, SmallVec<[RecId; 8]>), ContractError> {
        let (d, hidden) = self.store.withdraw_document(doc_id, version)?;
        if let Some(a) = &self.bm25_async {
            if self.cfg.bm25_documents {
                a.delete_document(d, self.store.doc_text(d));
            }
            if self.cfg.bm25_records {
                for &h in &hidden {
                    let t = self.rec_text(h);
                    self.bm25_async.as_ref().unwrap().delete_record(h, &t);
                    self.text_buf = t;
                }
            }
        } else {
            if let Some(b) = self.bm25_doc.as_mut() {
                let text = self.store.doc_text(d).to_string();
                b.delete(d, &text);
            }
            if self.bm25_rec.is_some() {
                for &h in &hidden {
                    let t = self.rec_text(h);
                    self.bm25_rec.as_mut().unwrap().delete(h, &t);
                    self.text_buf = t;
                }
            }
        }
        if let Some(l) = self.log.as_mut() {
            l.append(&Op::WithdrawDocument { doc_id, version });
        }
        Ok((d, hidden))
    }

    /// xr_codecs.world_store through this memory (indexes maintained per op)
    pub fn ingest_world(&mut self, w: &World<'_>) -> Result<WorldRevs, ContractError> {
        ingest_world(self, w)
    }

    /// Seal in-memory index segments, install finished merges, flush the log (group commit).
    pub fn commit(&mut self) -> std::io::Result<()> {
        self.with_backends_mut(|b, _| b.commit());
        if let Some(l) = self.log.as_mut() {
            l.commit()?;
        }
        Ok(())
    }

    // -------------------------------------------------------------------------------------------- reads
    pub fn head(&self) -> Snapshot<'_> {
        self.store.head()
    }

    pub fn snapshot_at(&self, rev: Rev) -> Snapshot<'_> {
        self.store.at(rev, ValidAt::parse(VALID_AT).unwrap())
    }

    /// xr_codecs.link_entities (subject syms, reference order)
    pub fn link(&self, snap: &Snapshot<'_>, question: &str, buf: &mut QBuf) -> SmallVec<[u32; 8]> {
        self.linker.link_exact(snap, &self.subjects, question, buf)
    }

    /// xr_codecs.candidate_manifest(item, snapshot, K)
    pub fn candidate_manifest(
        &self,
        snap: &Snapshot<'_>,
        item_id: &str,
        question: &str,
        k: usize,
        buf: &mut QBuf,
    ) -> Vec<RecId> {
        let ents = self.link(snap, question, buf);
        let fp = snap.fingerprint();
        manifest::candidate_manifest(
            snap,
            &self.subjects,
            &ents,
            item_id,
            &fp,
            k,
            self.cfg.manifest_hash,
            None,
        )
    }

    /// candidate_manifest with build statistics
    pub fn candidate_manifest_stats(
        &self,
        snap: &Snapshot<'_>,
        item_id: &str,
        question: &str,
        k: usize,
        buf: &mut QBuf,
    ) -> (Vec<RecId>, ManifestStats) {
        let ents = self.link(snap, question, buf);
        let fp = snap.fingerprint();
        let mut st = ManifestStats::default();
        let m = manifest::candidate_manifest(
            snap,
            &self.subjects,
            &ents,
            item_id,
            &fp,
            k,
            self.cfg.manifest_hash,
            Some(&mut st),
        );
        (m, st)
    }

    /// stack_v2.hop_manifest for a chained text value
    pub fn hop_manifest(
        &self,
        snap: &Snapshot<'_>,
        item_id: &str,
        fingerprint: &str,
        value: &TypedValue<'_>,
        value_ids: &[&str],
        k: usize,
    ) -> HopManifest {
        manifest::hop_manifest(
            snap,
            &self.subjects,
            item_id,
            fingerprint,
            value,
            value_ids,
            k,
            self.cfg.manifest_hash,
        )
    }

    // -------------------------------------------------------------------------------------------- stack-v5 (direction)
    /// the stack-v5 value tables, caught up with the store (built on first use)
    pub fn value_tables(&self) -> RwLockReadGuard<'_, ValueTables> {
        self.values.tables(&self.store)
    }

    /// `stack_v5.link_values(question, snapshot)`: the linked text values as `strs` syms (question order); the strings are
    /// `self.store.pools.strs.get(sym)`
    pub fn link_values(
        &self,
        snap: &Snapshot<'_>,
        question: &str,
        buf: &mut QBuf,
    ) -> SmallVec<[u32; 8]> {
        let t = self.values.tables(&self.store);
        direction::link_values(snap, &t, question, buf)
    }

    /// `stack_v5.candidate_manifest(item, snapshot, K, entity_ids)` with MAX_INVERSE from XERJ_R_MAX_INVERSE (default 8):
    /// K records in manifest order + one inverse flag per slot
    pub fn candidate_manifest_v5(
        &self,
        snap: &Snapshot<'_>,
        item_id: &str,
        question: &str,
        k: usize,
        buf: &mut QBuf,
    ) -> CandManifest {
        self.candidate_manifest_v5_with(snap, item_id, question, k, direction::max_inverse(), buf)
    }

    /// `stack_v5.candidate_manifest(..., max_inverse)` with an explicit cap (0 = the stack-v2 manifest)
    pub fn candidate_manifest_v5_with(
        &self,
        snap: &Snapshot<'_>,
        item_id: &str,
        question: &str,
        k: usize,
        max_inverse: i64,
        buf: &mut QBuf,
    ) -> CandManifest {
        let ents = self.link(snap, question, buf);
        let t = self.values.tables(&self.store);
        let vals = direction::link_values(snap, &t, question, buf);
        let fp = snap.fingerprint();
        direction::candidate_manifest_v5(
            snap,
            &self.subjects,
            &t,
            &ents,
            &vals,
            item_id,
            &fp,
            k,
            self.cfg.manifest_hash,
            max_inverse,
        )
    }

    /// `stack_v5.hop_manifest` for a chained text value (inverse slots flagged in `HopManifest::inverse`)
    #[allow(clippy::too_many_arguments)]
    pub fn hop_manifest_v5(
        &self,
        snap: &Snapshot<'_>,
        item_id: &str,
        fingerprint: &str,
        value: &TypedValue<'_>,
        value_ids: &[&str],
        k: usize,
        max_inverse: i64,
    ) -> HopManifest {
        let t = self.values.tables(&self.store);
        direction::hop_manifest_v5(
            snap,
            &self.subjects,
            &t,
            item_id,
            fingerprint,
            value,
            value_ids,
            k,
            self.cfg.manifest_hash,
            max_inverse,
        )
    }

    /// EXTENSION (opt-in): [`Memory::candidate_manifest_v5_with`] with the linked subjects given (`ents`; None = the exact
    /// linker on `question`) and a [`direction::RelPref`] (None = the reference choice where a cap binds). With both None
    /// this is `candidate_manifest_v5_with`.
    #[allow(clippy::too_many_arguments)]
    pub fn candidate_manifest_v5_opts(
        &self,
        snap: &Snapshot<'_>,
        item_id: &str,
        question: &str,
        k: usize,
        max_inverse: i64,
        ents: Option<&[u32]>,
        pref: Option<&direction::RelPref>,
        buf: &mut QBuf,
    ) -> CandManifest {
        let linked;
        let ents: &[u32] = match ents {
            Some(e) => e,
            None => {
                linked = self.link(snap, question, buf);
                &linked
            }
        };
        let t = self.values.tables(&self.store);
        let vals = direction::link_values(snap, &t, question, buf);
        let fp = snap.fingerprint();
        direction::candidate_manifest_v5_pref(
            snap,
            &self.subjects,
            &t,
            ents,
            &vals,
            item_id,
            &fp,
            k,
            self.cfg.manifest_hash,
            max_inverse,
            pref,
        )
    }

    /// EXTENSION (opt-in): [`Memory::hop_manifest_v5`] with a [`direction::RelPref`] (None = the reference)
    #[allow(clippy::too_many_arguments)]
    pub fn hop_manifest_v5_pref(
        &self,
        snap: &Snapshot<'_>,
        item_id: &str,
        fingerprint: &str,
        value: &TypedValue<'_>,
        value_ids: &[&str],
        k: usize,
        max_inverse: i64,
        pref: Option<&direction::RelPref>,
    ) -> HopManifest {
        let t = self.values.tables(&self.store);
        direction::hop_manifest_v5_pref(
            snap,
            &self.subjects,
            &t,
            item_id,
            fingerprint,
            value,
            value_ids,
            k,
            self.cfg.manifest_hash,
            max_inverse,
            pref,
        )
    }

    /// EXTENSION (opt-in): [`Memory::candidate_manifest_v5_opts`] over a record scope (`direction::RecScope`: only the
    /// records a question may read, or those first where a cap binds)
    #[allow(clippy::too_many_arguments)]
    pub fn candidate_manifest_v5_scope(
        &self,
        snap: &Snapshot<'_>,
        item_id: &str,
        question: &str,
        k: usize,
        max_inverse: i64,
        ents: Option<&[u32]>,
        pref: Option<&direction::RelPref>,
        scope: direction::RecScope<'_>,
        buf: &mut QBuf,
    ) -> CandManifest {
        let linked;
        let ents: &[u32] = match ents {
            Some(e) => e,
            None => {
                linked = self.link(snap, question, buf);
                &linked
            }
        };
        let t = self.values.tables(&self.store);
        let vals = direction::link_values(snap, &t, question, buf);
        let fp = snap.fingerprint();
        direction::candidate_manifest_v5_scope(
            snap,
            &self.subjects,
            &t,
            ents,
            &vals,
            item_id,
            &fp,
            k,
            self.cfg.manifest_hash,
            max_inverse,
            pref,
            scope,
        )
    }

    /// EXTENSION (opt-in): [`Memory::hop_manifest_v5_pref`] over a record scope (`direction::RecScope`)
    #[allow(clippy::too_many_arguments)]
    pub fn hop_manifest_v5_scope(
        &self,
        snap: &Snapshot<'_>,
        item_id: &str,
        fingerprint: &str,
        value: &TypedValue<'_>,
        value_ids: &[&str],
        k: usize,
        max_inverse: i64,
        pref: Option<&direction::RelPref>,
        scope: direction::RecScope<'_>,
    ) -> HopManifest {
        let t = self.values.tables(&self.store);
        direction::hop_manifest_v5_scope(
            snap,
            &self.subjects,
            &t,
            item_id,
            fingerprint,
            value,
            value_ids,
            k,
            self.cfg.manifest_hash,
            max_inverse,
            pref,
            scope,
        )
    }

    /// the set check of `stack_v5.inverse_result` for record `r` (`direction::same_value_records`)
    pub fn inverse_same_value(&self, snap: &Snapshot<'_>, r: RecId) -> Vec<RecId> {
        let t = self.values.tables(&self.store);
        direction::same_value_records(snap, &t, r)
    }

    /// EXTENSION: candidate manifest with a BM25-assisted fallback when the exact linker finds nothing: normalized
    /// subject linking first, then the BM25 top records (head index, filtered to `snap`) as tier 1. Identical to
    /// `candidate_manifest` whenever the exact linker links something. Returns (manifest, which path was used).
    pub fn candidate_manifest_with_fallback(
        &self,
        snap: &Snapshot<'_>,
        item_id: &str,
        question: &str,
        k: usize,
        buf: &mut QBuf,
    ) -> (Vec<RecId>, &'static str) {
        let ents = self.link(snap, question, buf);
        let fp = snap.fingerprint();
        if !ents.is_empty() {
            return (
                manifest::candidate_manifest(
                    snap,
                    &self.subjects,
                    &ents,
                    item_id,
                    &fp,
                    k,
                    self.cfg.manifest_hash,
                    None,
                ),
                "exact",
            );
        }
        let norm = link_normalized(snap, &self.subjects, question, 12);
        if !norm.is_empty() {
            return (
                manifest::candidate_manifest(
                    snap,
                    &self.subjects,
                    &norm,
                    item_id,
                    &fp,
                    k,
                    self.cfg.manifest_hash,
                    None,
                ),
                "normalized",
            );
        }
        let mut subjects: SmallVec<[u32; 8]> = SmallVec::new();
        for h in self.search_records(question, k) {
            if snap.visible(h.doc) {
                let s = self.store.recs.subject[h.doc as usize];
                if !subjects.contains(&s) {
                    subjects.push(s);
                }
            }
            if subjects.len() >= 4 {
                break;
            }
        }
        (
            manifest::candidate_manifest(
                snap,
                &self.subjects,
                &subjects,
                item_id,
                &fp,
                k,
                self.cfg.manifest_hash,
                None,
            ),
            "bm25",
        )
    }

    /// Query algorithm of the native BM25 backends (no-op for other backends).
    pub fn set_bm25_algo(&mut self, algo: crate::bm25::SearchAlgo) {
        self.with_backends_mut(|b, _| {
            if let Some(nb) = b.as_any_mut().downcast_mut::<SegmentedBm25>() {
                nb.algo = algo;
            }
        });
    }

    /// BM25 over record text (head revision); ties broken by fact id like reference_core.LexicalIndex.search
    pub fn search_records(&self, query: &str, k: usize) -> Vec<Hit> {
        let Some(mut hits) = self.with_record_index(|b| b.search_with_ties(query, k)) else {
            return Vec::new();
        };
        let cmp = |a: &Hit, b: &Hit| {
            b.score
                .total_cmp(&a.score)
                .then_with(|| self.store.fact_id(a.doc).cmp(self.store.fact_id(b.doc)))
        };
        if hits.len() > k {
            hits.select_nth_unstable_by(k - 1, cmp);
            hits.truncate(k);
        }
        hits.sort_unstable_by(cmp);
        hits
    }

    /// document frequency of an analyzed term in the document index (None when the backend cannot tell)
    pub fn document_term_df(&self, term: &str) -> Option<u32> {
        self.with_document_index(|b| b.term_df(term)).flatten()
    }

    /// BM25 over source documents (head revision; withdrawn versions excluded)
    pub fn search_documents(&self, query: &str, k: usize) -> Vec<Hit> {
        self.with_document_index(|b| b.search(query, k))
            .unwrap_or_default()
    }

    /// Release spare capacity everywhere (call after a bulk load; seals in-memory index segments).
    pub fn shrink_to_fit(&mut self) {
        let _ = self.commit();
        self.store.shrink_to_fit();
        self.subjects.shrink_to_fit();
    }

    pub fn heap_bytes(&self) -> MemoryBytes {
        MemoryBytes {
            store: self.store.heap_bytes(),
            subject_index: self.subjects.heap_bytes(),
            linker_ac: self.linker.ac.stats().3,
            bm25_records: self.with_record_index(|b| b.heap_bytes()).unwrap_or(0),
            bm25_documents: self.with_document_index(|b| b.heap_bytes()).unwrap_or(0),
        }
    }

    // -------------------------------------------------------------------------------------------- persistence
    /// Open (or create) a persistent memory in `dir`: load `memory.ckp` if present, replay `ops.log` past it, keep
    /// appending to the log. `durable` = fdatasync at every commit.
    pub fn open(dir: &Path, cfg: MemoryConfig, durable: bool) -> Result<Memory, StoreError> {
        std::fs::create_dir_all(dir)?;
        let ck = dir.join(CKPT_FILE);
        let (mut m, from) = if ck.exists() {
            let s = Sections::open(&ck)?;
            let (store, meta) = read_store(&s)?;
            let mut m = Memory::new(MemoryConfig {
                store: store.config().clone(),
                ..cfg.clone()
            });
            m.store = store;
            m.subjects.catch_up(&m.store);
            m.linker.catch_up(&m.store);
            let load = |tag: u32| -> Result<Box<dyn Bm25Backend>, StoreError> {
                let mut b = SegmentedBm25::read_sections(&s, tag, cfg.bm25)?;
                b.seal_postings = cfg.bm25_seal_postings;
                Ok(Box::new(if cfg.background_merges {
                    b.with_background_merges()
                } else {
                    b
                }))
            };
            let rec = if cfg.bm25_records && s.has(T_BM25_REC) {
                Some(load(T_BM25_REC)?)
            } else {
                None
            };
            let doc = if cfg.bm25_documents && s.has(T_BM25_DOC) {
                Some(load(T_BM25_DOC)?)
            } else {
                None
            };
            m.install_backends(rec, doc);
            (m, meta.log_offset)
        } else {
            (Memory::new(cfg), 0)
        };
        let logp = dir.join(LOG_FILE);
        let mut valid = None;
        if logp.exists() {
            let st = replay(&logp, from, |op| m.apply_op(op))?;
            valid = Some(st.valid_len);
        }
        m.log = Some(LogWriter::open(&logp, durable, valid)?);
        m.dir = Some(dir.to_path_buf());
        Ok(m)
    }

    fn apply_op(&mut self, op: &Op<'_>) -> Result<(), ContractError> {
        match op {
            Op::AddDocument {
                doc_id,
                version,
                text,
            } => self.add_document(doc_id, version, text).map(|_| ()),
            Op::AddFact { fact, supersedes } => self.add_fact(fact, supersedes).map(|_| ()),
            Op::WithdrawFact { fact_id } => self.withdraw_fact(fact_id).map(|_| ()),
            Op::WithdrawDocument { doc_id, version } => {
                self.withdraw_document(doc_id, version).map(|_| ())
            }
        }
    }

    /// Write a checkpoint (store columns + native BM25 segments) covering the whole log; returns its size in bytes.
    pub fn checkpoint(&mut self) -> Result<u64, StoreError> {
        let dir = self.dir.clone().ok_or_else(|| {
            StoreError::Format("memory has no directory (use Memory::open)".into())
        })?;
        self.commit()?;
        let log_offset = self.log.as_ref().map_or(0, |l| l.len());
        self.checkpoint_to(&dir.join(CKPT_FILE), log_offset)
    }

    /// Write a checkpoint file to `path` (atomic rename).
    pub fn checkpoint_to(&mut self, path: &Path, log_offset: u64) -> Result<u64, StoreError> {
        let tmp = path.with_extension("tmp");
        let mut w = SectionWriter::create(&tmp)?;
        write_store(&self.store, &mut w, log_offset)?;
        let mut err = None;
        self.with_backends_mut(|b, is_rec| {
            // only the native backend persists itself; others are rebuilt from the store on open
            if let Some(nb) = b.as_any_mut().downcast_mut::<SegmentedBm25>() {
                if let Err(e) =
                    nb.write_sections(&mut w, if is_rec { T_BM25_REC } else { T_BM25_DOC })
                {
                    err = Some(e);
                }
            }
        });
        if let Some(e) = err {
            return Err(e);
        }
        let n = w.finish()?;
        std::fs::rename(&tmp, path)?;
        Ok(n)
    }

    /// Load a checkpoint file without a log (benchmarks: reload time).
    pub fn load_checkpoint(path: &Path, cfg: MemoryConfig) -> Result<Memory, StoreError> {
        let s = Sections::open(path)?;
        let (store, _) = read_store(&s)?;
        let mut m = Memory::new(MemoryConfig {
            store: store.config().clone(),
            ..cfg.clone()
        });
        m.store = store;
        m.subjects.catch_up(&m.store);
        m.linker.catch_up(&m.store);
        let rec: Option<Box<dyn Bm25Backend>> = if cfg.bm25_records && s.has(T_BM25_REC) {
            Some(Box::new(SegmentedBm25::read_sections(
                &s, T_BM25_REC, cfg.bm25,
            )?))
        } else {
            None
        };
        let doc: Option<Box<dyn Bm25Backend>> = if cfg.bm25_documents && s.has(T_BM25_DOC) {
            Some(Box::new(SegmentedBm25::read_sections(
                &s, T_BM25_DOC, cfg.bm25,
            )?))
        } else {
            None
        };
        m.install_backends(rec, doc);
        Ok(m)
    }
}

/// Heap bytes per component.
#[derive(Clone, Copy, Debug, Default, serde::Serialize)]
pub struct MemoryBytes {
    pub store: usize,
    pub subject_index: usize,
    pub linker_ac: usize,
    pub bm25_records: usize,
    pub bm25_documents: usize,
}

impl MemoryBytes {
    pub fn total(&self) -> usize {
        self.store + self.subject_index + self.linker_ac + self.bm25_records + self.bm25_documents
    }
}

impl WriteSink for Memory {
    fn sink_add_document(
        &mut self,
        doc_id: &str,
        version: &str,
        text: &str,
    ) -> Result<DocIx, ContractError> {
        self.add_document(doc_id, version, text)
    }
    fn sink_add_fact(
        &mut self,
        f: &FactIn<'_>,
        supersedes: &[&str],
    ) -> Result<AddedFact, ContractError> {
        self.add_fact(f, supersedes)
    }
    fn sink_withdraw_fact(&mut self, fact_id: &str) -> Result<(RecId, bool), ContractError> {
        self.withdraw_fact(fact_id)
    }
    fn sink_withdraw_document(
        &mut self,
        doc_id: &str,
        version: &str,
    ) -> Result<(DocIx, SmallVec<[RecId; 8]>), ContractError> {
        self.withdraw_document(doc_id, version)
    }
    fn sink_revision(&self) -> Rev {
        self.store.revision()
    }
}
