//! Incremental record indexes over a `Store`: subject -> records, relation -> records, entity id (subject_id) -> records,
//! and (lazily) normalized subject name -> subjects. All lists hold every record ever written (MVCC history); readers
//! filter by snapshot visibility, so one index serves every revision and "before" / "after" snapshots.
//!
//! Maintenance is O(1) per written record (append to dense, sym-indexed vectors; no hashing besides the store's own
//! interning). The normalized-name index (stack-v2 string resolution) is built on first use and then caught up
//! incrementally, so workloads that never chain pay nothing for it at ingest.

use smallvec::SmallVec;
use std::sync::RwLock;
use xerj_store::text::normalize_entity;
use xerj_store::{RecId, Snapshot, Store, StrPool, NONE};

#[derive(Default)]
pub struct NormIndex {
    /// normalized names (pool id = norm id)
    pub names: StrPool,
    /// subject sym -> norm id (filled up to `norm_of_subject.len()` subjects)
    pub norm_of_subject: Vec<u32>,
    /// norm id -> subject syms
    pub subjects_of_norm: Vec<SmallVec<[u32; 2]>>,
}

impl NormIndex {
    fn catch_up(&mut self, subjects: &StrPool) {
        for s in self.norm_of_subject.len() as u32..subjects.len() as u32 {
            let n = normalize_entity(subjects.get(s));
            let id = self.names.intern(&n);
            if id as usize == self.subjects_of_norm.len() {
                self.subjects_of_norm.push(SmallVec::new());
            }
            self.subjects_of_norm[id as usize].push(s);
            self.norm_of_subject.push(id);
        }
    }
}

#[derive(Default)]
pub struct SubjectIndex {
    /// subject sym -> records (write order)
    pub by_subject: Vec<SmallVec<[RecId; 4]>>,
    /// relation sym -> records (write order)
    pub by_relation: Vec<Vec<RecId>>,
    /// entity id sym (subject_id) -> records
    pub by_subject_id: Vec<SmallVec<[RecId; 2]>>,
    /// lazily built normalized-name index
    norm: RwLock<NormIndex>,
    indexed: usize,
}

impl SubjectIndex {
    pub fn new() -> Self {
        Self::default()
    }

    /// number of records indexed so far
    pub fn indexed(&self) -> usize {
        self.indexed
    }

    /// Index every record of `store` not indexed yet (call after writes; O(new records)).
    pub fn catch_up(&mut self, store: &Store) {
        let n = store.num_records();
        for r in self.indexed..n {
            self.add_record(store, r as RecId);
        }
    }

    #[inline]
    pub fn add_record(&mut self, store: &Store, r: RecId) {
        debug_assert_eq!(r as usize, self.indexed);
        let i = r as usize;
        let s = store.recs.subject[i] as usize;
        if s >= self.by_subject.len() {
            self.by_subject.resize_with(s + 1, SmallVec::new);
        }
        self.by_subject[s].push(r);
        let rl = store.recs.relation[i] as usize;
        if rl >= self.by_relation.len() {
            self.by_relation.resize_with(rl + 1, Vec::new);
        }
        self.by_relation[rl].push(r);
        let sid = store.recs.subject_id[i];
        if sid != NONE {
            let sid = sid as usize;
            if sid >= self.by_subject_id.len() {
                self.by_subject_id.resize_with(sid + 1, SmallVec::new);
            }
            self.by_subject_id[sid].push(r);
        }
        self.indexed = i + 1;
    }

    #[inline]
    pub fn records_of_subject(&self, subject: u32) -> &[RecId] {
        self.by_subject
            .get(subject as usize)
            .map_or(&[], |v| v.as_slice())
    }

    #[inline]
    pub fn records_of_relation(&self, relation: u32) -> &[RecId] {
        self.by_relation
            .get(relation as usize)
            .map_or(&[], |v| v.as_slice())
    }

    #[inline]
    pub fn records_of_subject_id(&self, ent: u32) -> &[RecId] {
        self.by_subject_id
            .get(ent as usize)
            .map_or(&[], |v| v.as_slice())
    }

    /// does `subject` have a record visible in `snap` (i.e. is it in `{f.subject for f in snapshot.facts}`)?
    #[inline]
    pub fn subject_active(&self, snap: &Snapshot<'_>, subject: u32) -> bool {
        self.records_of_subject(subject)
            .iter()
            .any(|&r| snap.visible(r))
    }

    /// subject syms whose normalize_entity equals `normalize_entity(text)` (builds / catches up the lazy index)
    pub fn subjects_with_norm(
        &self,
        store: &Store,
        normalized: &str,
        out: &mut SmallVec<[u32; 4]>,
    ) {
        {
            let g = self.norm.read().unwrap();
            if g.norm_of_subject.len() == store.pools.subjects.len() {
                if let Some(id) = g.names.lookup(normalized) {
                    out.extend_from_slice(&g.subjects_of_norm[id as usize]);
                }
                return;
            }
        }
        let mut g = self.norm.write().unwrap();
        g.catch_up(&store.pools.subjects);
        if let Some(id) = g.names.lookup(normalized) {
            out.extend_from_slice(&g.subjects_of_norm[id as usize]);
        }
    }

    /// normalized name of a subject (lazy index)
    pub fn norm_of_subject(&self, store: &Store, subject: u32) -> String {
        {
            let g = self.norm.read().unwrap();
            if let Some(&id) = g.norm_of_subject.get(subject as usize) {
                return g.names.get(id).to_string();
            }
        }
        let mut g = self.norm.write().unwrap();
        g.catch_up(&store.pools.subjects);
        let id = g.norm_of_subject[subject as usize];
        g.names.get(id).to_string()
    }

    pub fn shrink_to_fit(&mut self) {
        self.by_subject.shrink_to_fit();
        for v in self.by_subject.iter_mut() {
            if v.spilled() {
                v.shrink_to_fit();
            }
        }
        self.by_relation.shrink_to_fit();
        for v in self.by_relation.iter_mut() {
            v.shrink_to_fit();
        }
        self.by_subject_id.shrink_to_fit();
    }

    pub fn heap_bytes(&self) -> usize {
        let sv = std::mem::size_of::<SmallVec<[RecId; 4]>>();
        let sv2 = std::mem::size_of::<SmallVec<[RecId; 2]>>();
        let subj = self.by_subject.capacity() * sv
            + self
                .by_subject
                .iter()
                .filter(|v| v.spilled())
                .map(|v| v.capacity() * 4)
                .sum::<usize>();
        let rel = self.by_relation.capacity() * 24
            + self
                .by_relation
                .iter()
                .map(|v| v.capacity() * 4)
                .sum::<usize>();
        let ent = self.by_subject_id.capacity() * sv2
            + self
                .by_subject_id
                .iter()
                .filter(|v| v.spilled())
                .map(|v| v.capacity() * 4)
                .sum::<usize>();
        let g = self.norm.read().unwrap();
        let norm = g.names.heap_bytes()
            + g.norm_of_subject.capacity() * 4
            + g.subjects_of_norm.capacity() * std::mem::size_of::<SmallVec<[u32; 2]>>();
        subj + rel + ent + norm
    }
}
