//! Native segmented BM25 (see `super` for the design summary).
//!
//! * Term dictionary: one global hash interner (term -> term id) with exact live df per term; postings everywhere refer
//!   to global term ids, so sealed segments need no per-segment dictionary beyond a sorted term-id array.
//! * In-memory segment: per-term linked posting chains in three flat arrays (O(1) append, no per-term allocation) plus
//!   per-term max tf / min doc length; searchable as soon as `add` returns (read-your-writes).
//! * Sealed segment: postings in 128-doc blocks, doc deltas and tf bit-packed (codec.rs), per-block skip entries (last doc,
//!   offset) and per-term score bounds (max tf, min doc length); doc-id range [lo, hi); data in a `Blob` (owned after a
//!   seal / merge, a zero-copy mmap range after reload).
//! * Deletes: a live bitset + exact df / N / total length updates; dead postings are skipped at query time and dropped
//!   by merges. Merges: tiered (merge_factor segments of one size class -> one), on a background thread when enabled,
//!   installed at the next write / commit.
//! * Query (default `Pruned`, exact): query terms sorted by their score upper bound (max over segments of the bound
//!   from the term's max tf and min doc length); this order is also the canonical summation order of every path.
//!   Phase 1 scores the high-bound terms term-at-a-time over every segment (tight decode loops into an epoch-stamped
//!   dense accumulator). As soon as the summed bounds of the remaining terms fall strictly below theta = the k-th best
//!   partial score (a lower bound of the final k-th score, maintained exactly from the previous top k plus the docs a
//!   term changed, and only when a switch is possible), no unseen doc can enter the top k: phase 2 keeps only candidates
//!   with partial + remaining bound >= theta and adds the remaining low-bound terms (stopwords) either by scanning their
//!   postings against candidate marks or by block-skipping `advance` per candidate, whichever is cheaper, pruning as
//!   bounds shrink. Contributions are added in the canonical order in both phases, so scores are bit-identical to
//!   `Taat` and ties stay ties (callers break them by their own key). Bounds and comparisons carry a safety margin.
//!   Worst case (no selective term) = TAAT.

use super::codec::{encode_postings_skips, BlockCursor, PostingIter};
use super::{Bm25Backend, Bm25Params, Bm25Stats, Hit};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use xerj_store::ckpt::{Blob, SectionWriter, Sections};
use xerj_store::{StoreError, StrPool};

const NONE: u32 = u32::MAX;
/// term-id cache slots (power of two)
const TCACHE: usize = 8192;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SearchAlgo {
    /// exact pruned term-at-a-time (default; see module docs)
    Pruned,
    /// plain term-at-a-time over every posting (reference path)
    Taat,
}

pub struct Segment {
    pub lo: u32,
    pub hi: u32,
    terms: Vec<u32>,
    offs: Vec<u64>,
    counts: Vec<u32>,
    max_tf: Vec<u32>,
    min_len: Vec<u32>,
    /// per term: range [skip_off[i], skip_off[i+1]) of `skip_last`; `skip_start` has one extra entry per term (tail start)
    skip_off: Vec<u32>,
    skip_last: Vec<u32>,
    skip_start: Vec<u32>,
    data: Blob,
    pub postings: u64,
}

impl Segment {
    #[inline]
    fn find(&self, tid: u32) -> Option<usize> {
        self.terms.binary_search(&tid).ok()
    }
    #[inline]
    fn postings_of(&self, tid: u32) -> Option<PostingIter<'_>> {
        let i = self.find(tid)?;
        let d = self.data.as_slice();
        Some(PostingIter::new(
            &d[self.offs[i] as usize..self.offs[i + 1] as usize],
            self.lo,
            self.counts[i] as usize,
        ))
    }
    #[inline]
    fn cursor(&self, i: usize) -> BlockCursor<'_> {
        let d = self.data.as_slice();
        let (s0, s1) = (self.skip_off[i] as usize, self.skip_off[i + 1] as usize);
        // skip_start holds (full blocks + 1) entries per term: offset by the term index
        BlockCursor::new(
            &d[self.offs[i] as usize..self.offs[i + 1] as usize],
            self.lo,
            self.counts[i] as usize,
            &self.skip_last[s0..s1],
            &self.skip_start[s0 + i..s1 + i + 1],
        )
    }
    fn heap_bytes(&self) -> usize {
        self.terms.capacity() * 4
            + self.offs.capacity() * 8
            + (self.counts.capacity()
                + self.max_tf.capacity()
                + self.min_len.capacity()
                + self.skip_off.capacity())
                * 4
            + (self.skip_last.capacity() + self.skip_start.capacity()) * 4
            + self.data.heap_bytes()
    }
    fn mapped_bytes(&self) -> usize {
        if self.data.is_mapped() {
            self.data.as_slice().len()
        } else {
            0
        }
    }
    /// size class for the tiered merge policy
    fn level(&self, base: u64, factor: u64) -> u32 {
        // sealed segments hold ~base postings -> class 0; a merge of `factor` of them -> class 1, ...
        let mut l = 0;
        let mut cap = base.max(1).saturating_mul(2);
        while self.postings > cap {
            cap = cap.saturating_mul(factor);
            l += 1;
        }
        l
    }
}

struct SegBuilder {
    lo: u32,
    hi: u32,
    terms: Vec<u32>,
    offs: Vec<u64>,
    counts: Vec<u32>,
    max_tf: Vec<u32>,
    min_len: Vec<u32>,
    skip_off: Vec<u32>,
    skip_last: Vec<u32>,
    skip_start: Vec<u32>,
    data: Vec<u8>,
    postings: u64,
}

impl SegBuilder {
    fn new(lo: u32, hi: u32) -> SegBuilder {
        SegBuilder {
            lo,
            hi,
            terms: Vec::new(),
            offs: vec![0],
            counts: Vec::new(),
            max_tf: Vec::new(),
            min_len: Vec::new(),
            skip_off: vec![0],
            skip_last: Vec::new(),
            skip_start: Vec::new(),
            data: Vec::new(),
            postings: 0,
        }
    }
    fn push(&mut self, tid: u32, docs: &[u32], tfs: &[u32], doc_len: &[u32]) {
        if docs.is_empty() {
            return;
        }
        let mut starts = Vec::new();
        encode_postings_skips(
            self.lo,
            docs,
            tfs,
            &mut self.data,
            Some((&mut self.skip_last, &mut starts)),
        );
        self.skip_start.extend_from_slice(&starts);
        self.terms.push(tid);
        self.offs.push(self.data.len() as u64);
        self.counts.push(docs.len() as u32);
        self.max_tf.push(tfs.iter().copied().max().unwrap_or(1));
        self.min_len.push(
            docs.iter()
                .map(|&d| doc_len.get(d as usize).copied().unwrap_or(0))
                .min()
                .unwrap_or(0),
        );
        self.skip_off.push(self.skip_last.len() as u32);
        self.postings += docs.len() as u64;
    }
    fn finish(mut self) -> Segment {
        self.data.shrink_to_fit();
        self.skip_last.shrink_to_fit();
        self.skip_start.shrink_to_fit();
        Segment {
            lo: self.lo,
            hi: self.hi,
            terms: self.terms,
            offs: self.offs,
            counts: self.counts,
            max_tf: self.max_tf,
            min_len: self.min_len,
            skip_off: self.skip_off,
            skip_last: self.skip_last,
            skip_start: self.skip_start,
            data: Blob::owned(self.data),
            postings: self.postings,
        }
    }
}

/// Per-term mutable state, co-located (one cache line access per term occurrence at ingest): live df and the term's
/// posting chain + bounds in the in-memory segment.
#[derive(Clone, Copy)]
#[repr(C, align(32))]
struct TermSt {
    df: u32,
    head: u32,
    tail: u32,
    max_tf: u32,
    min_len: u32,
}

impl Default for TermSt {
    fn default() -> Self {
        TermSt {
            df: 0,
            head: NONE,
            tail: NONE,
            max_tf: 0,
            min_len: 0,
        }
    }
}

#[derive(Default)]
struct MemSeg {
    lo: u32,
    hi: u32,
    touched: Vec<u32>,
    p_doc: Vec<u32>,
    p_tf: Vec<u32>,
    p_next: Vec<u32>,
}

impl MemSeg {
    #[inline]
    fn push(&mut self, ts: &mut TermSt, tid: u32, doc: u32, tf: u32, len: u32) {
        let idx = self.p_doc.len() as u32;
        self.p_doc.push(doc);
        self.p_tf.push(tf);
        self.p_next.push(NONE);
        if ts.head == NONE {
            ts.head = idx;
            self.touched.push(tid);
            ts.max_tf = tf;
            ts.min_len = len;
        } else {
            self.p_next[ts.tail as usize] = idx;
            ts.max_tf = ts.max_tf.max(tf);
            ts.min_len = ts.min_len.min(len);
        }
        ts.tail = idx;
    }
    fn heap_bytes(&self) -> usize {
        self.touched.capacity() * 4
            + (self.p_doc.capacity() + self.p_tf.capacity() + self.p_next.capacity()) * 4
    }
}

/// Diagnostics of the last pruned search (`SegmentedBm25::explain`).
#[derive(Clone, Copy, Debug, Default, serde::Serialize)]
pub struct PrunedStats {
    pub terms: usize,
    pub phase1_terms: usize,
    pub postings_phase1: u64,
    pub postings_total: u64,
    pub touched: usize,
    pub cands_phase2: usize,
    pub rescored: usize,
}

#[derive(Default)]
struct Scratch {
    last: PrunedStats,
    /// docs whose accumulator phase-1 terms changed since theta was last computed (pruned search)
    updated: Vec<u32>,
    /// best partial score so far (theta <= it: no switch is possible while the remaining bound exceeds it)
    max_acc: f64,
    /// candidate marks (phase 2 scan mode) and their epoch
    cmark: Vec<u32>,
    cepoch: u32,
    acc: Vec<f64>,
    stamp: Vec<u32>,
    epoch: u32,
    touched: Vec<u32>,
    q: Vec<(u32, f64)>,
    tok: String,
}

struct MergeJob {
    inputs: Vec<Arc<Segment>>,
    live: Arc<Vec<u64>>,
    doc_len: Arc<Vec<u32>>,
}

struct MergeDone {
    inputs: Vec<(u32, u32)>,
    seg: Segment,
    ms: f64,
}

fn merge_segments(inputs: &[Arc<Segment>], live: &[u64], doc_len: &[u32]) -> Segment {
    let lo = inputs.first().map_or(0, |s| s.lo);
    let hi = inputs.last().map_or(0, |s| s.hi);
    let mut b = SegBuilder::new(lo, hi);
    let mut pos = vec![0usize; inputs.len()];
    let (mut docs, mut tfs) = (Vec::new(), Vec::new());
    loop {
        let mut tid = NONE;
        for (k, s) in inputs.iter().enumerate() {
            if let Some(&t) = s.terms.get(pos[k]) {
                tid = tid.min(t);
            }
        }
        if tid == NONE {
            break;
        }
        docs.clear();
        tfs.clear();
        for (k, s) in inputs.iter().enumerate() {
            if s.terms.get(pos[k]) == Some(&tid) {
                let i = pos[k];
                let d = s.data.as_slice();
                let mut it = PostingIter::new(
                    &d[s.offs[i] as usize..s.offs[i + 1] as usize],
                    s.lo,
                    s.counts[i] as usize,
                );
                while let Some((ds, ts)) = it.next_block() {
                    for (&doc, &tf) in ds.iter().zip(ts) {
                        if live
                            .get((doc >> 6) as usize)
                            .is_some_and(|w| w >> (doc & 63) & 1 == 1)
                        {
                            docs.push(doc);
                            tfs.push(tf);
                        }
                    }
                }
                pos[k] += 1;
            }
        }
        b.push(tid, &docs, &tfs, doc_len);
    }
    b.finish()
}

pub struct SegmentedBm25 {
    params: Bm25Params,
    terms: StrPool,
    ts: Vec<TermSt>,
    doc_len: Vec<u32>,
    live: Vec<u64>,
    n_live: u64,
    total_len: u64,
    mem: MemSeg,
    segs: Vec<Arc<Segment>>,
    /// seal the in-memory segment beyond this many postings
    pub seal_postings: usize,
    /// merge this many segments of one size class
    pub merge_factor: usize,
    /// query algorithm (Pruned by default; Taat as the reference path)
    pub algo: SearchAlgo,
    background: bool,
    merge_tx: Option<mpsc::Sender<MergeJob>>,
    done_rx: Option<Mutex<mpsc::Receiver<MergeDone>>>,
    merge_busy: bool,
    seals: u64,
    merges: u64,
    seal_ms: f64,
    merge_ms: f64,
    scratch: Mutex<Vec<Scratch>>,
    tok: String,
    tids: Vec<u32>,
    tcache: Vec<(u64, u32)>,
}

impl SegmentedBm25 {
    pub fn new(params: Bm25Params) -> SegmentedBm25 {
        SegmentedBm25 {
            params,
            terms: StrPool::new(),
            ts: Vec::new(),
            doc_len: Vec::new(),
            live: Vec::new(),
            n_live: 0,
            total_len: 0,
            mem: MemSeg::default(),
            segs: Vec::new(),
            seal_postings: 1 << 20,
            merge_factor: 8,
            algo: SearchAlgo::Pruned,
            background: false,
            merge_tx: None,
            done_rx: None,
            merge_busy: false,
            seals: 0,
            merges: 0,
            seal_ms: 0.0,
            merge_ms: 0.0,
            scratch: Mutex::new(Vec::new()),
            tok: String::new(),
            tids: Vec::new(),
            tcache: vec![(0, 0); TCACHE],
        }
    }

    /// Run merges on one background thread (installed at the next add / commit).
    pub fn with_background_merges(mut self) -> Self {
        let (tx, rx) = mpsc::channel::<MergeJob>();
        let (dtx, drx) = mpsc::channel::<MergeDone>();
        std::thread::Builder::new()
            .name("xerj-bm25-merge".into())
            .spawn(move || {
                while let Ok(job) = rx.recv() {
                    let t = Instant::now();
                    let seg = merge_segments(&job.inputs, &job.live, &job.doc_len);
                    let inputs = job.inputs.iter().map(|s| (s.lo, s.hi)).collect();
                    if dtx
                        .send(MergeDone {
                            inputs,
                            seg,
                            ms: t.elapsed().as_secs_f64() * 1e3,
                        })
                        .is_err()
                    {
                        break;
                    }
                }
            })
            .expect("spawn merge thread");
        self.background = true;
        self.merge_tx = Some(tx);
        self.done_rx = Some(Mutex::new(drx));
        self
    }

    #[inline]
    fn is_live(&self, doc: u32) -> bool {
        let w = (doc >> 6) as usize;
        w < self.live.len() && (self.live[w] >> (doc & 63)) & 1 == 1
    }

    fn term_id(&mut self, t: &str) -> u32 {
        // direct-mapped cache in front of the term dictionary: frequent terms skip the big-table probe
        let h = self.terms.hash_str(t);
        let slot = (h as usize) & (TCACHE - 1);
        let (ch, cid) = self.tcache[slot];
        if ch == h && cid != 0 && self.terms.get(cid - 1) == t {
            return cid - 1;
        }
        let id = match self.terms.lookup_hashed(t, h) {
            Some(id) => id,
            None => self.terms.insert_new_hashed(t, h).unwrap_or_else(|e| e),
        };
        self.tcache[slot] = (h, id + 1);
        if id as usize == self.ts.len() {
            self.ts.push(TermSt::default());
        }
        id
    }

    fn seal(&mut self) {
        if self.mem.p_doc.is_empty() {
            self.mem.lo = self.mem.hi;
            return;
        }
        let t0 = Instant::now();
        let mut touched = std::mem::take(&mut self.mem.touched);
        touched.sort_unstable();
        let mut b = SegBuilder::new(self.mem.lo, self.mem.hi);
        let (mut docs, mut tfs) = (Vec::new(), Vec::new());
        for &tid in &touched {
            docs.clear();
            tfs.clear();
            let mut p = self.ts[tid as usize].head;
            while p != NONE {
                docs.push(self.mem.p_doc[p as usize]);
                tfs.push(self.mem.p_tf[p as usize]);
                p = self.mem.p_next[p as usize];
            }
            b.push(tid, &docs, &tfs, &self.doc_len);
            self.ts[tid as usize].head = NONE;
            self.ts[tid as usize].tail = NONE;
        }
        touched.clear();
        self.mem.touched = touched;
        self.mem.p_doc.clear();
        self.mem.p_tf.clear();
        self.mem.p_next.clear();
        self.mem.lo = self.mem.hi;
        self.segs.push(Arc::new(b.finish()));
        self.seals += 1;
        self.seal_ms += t0.elapsed().as_secs_f64() * 1e3;
        self.maybe_merge();
    }

    fn install_merges(&mut self, block: bool) {
        let Some(rx) = &self.done_rx else { return };
        let msg = {
            let rx = rx.lock().unwrap();
            if block && self.merge_busy {
                rx.recv().ok()
            } else {
                rx.try_recv().ok()
            }
        };
        if let Some(done) = msg {
            self.merge_busy = false;
            let first = self
                .segs
                .iter()
                .position(|s| (s.lo, s.hi) == done.inputs[0]);
            if let Some(i) = first {
                let n = done.inputs.len();
                let ok = i + n <= self.segs.len()
                    && self.segs[i..i + n]
                        .iter()
                        .zip(&done.inputs)
                        .all(|(s, r)| (s.lo, s.hi) == *r);
                if ok {
                    self.segs
                        .splice(i..i + n, std::iter::once(Arc::new(done.seg)));
                    self.merges += 1;
                    self.merge_ms += done.ms;
                }
            }
            self.maybe_merge();
        }
    }

    /// tiered policy: merge the trailing run of `merge_factor` segments that share a size class
    fn maybe_merge(&mut self) {
        if self.merge_busy || self.segs.len() < self.merge_factor {
            return;
        }
        let base = self.seal_postings as u64;
        let f = self.merge_factor as u64;
        let n = self.segs.len();
        let lvl = self.segs[n - 1].level(base, f);
        let run = self
            .segs
            .iter()
            .rev()
            .take_while(|s| s.level(base, f) == lvl)
            .count();
        if run < self.merge_factor {
            return;
        }
        let inputs: Vec<Arc<Segment>> = self.segs[n - run..].to_vec();
        if self.background {
            let job = MergeJob {
                inputs,
                live: Arc::new(self.live.clone()),
                doc_len: Arc::new(self.doc_len.clone()),
            };
            if self.merge_tx.as_ref().unwrap().send(job).is_ok() {
                self.merge_busy = true;
            }
        } else {
            let t0 = Instant::now();
            let seg = merge_segments(&inputs, &self.live, &self.doc_len);
            self.segs.truncate(n - run);
            self.segs.push(Arc::new(seg));
            self.merges += 1;
            self.merge_ms += t0.elapsed().as_secs_f64() * 1e3;
            self.maybe_merge();
        }
    }

    /// Force-merge everything into one segment (dead postings dropped); waits for a running background merge.
    pub fn force_merge(&mut self) {
        self.seal();
        self.install_merges(true);
        if self.segs.len() > 1 {
            let t0 = Instant::now();
            let seg = merge_segments(&self.segs, &self.live, &self.doc_len);
            self.segs = vec![Arc::new(seg)];
            self.merges += 1;
            self.merge_ms += t0.elapsed().as_secs_f64() * 1e3;
        }
    }

    pub fn num_segments(&self) -> usize {
        self.segs.len() + (!self.mem.p_doc.is_empty()) as usize
    }

    /// reference_core.LexicalIndex term contribution (identical float expression)
    #[inline(always)]
    fn contrib(&self, idf: f64, tf: u32, len: u32, avg: f64) -> f64 {
        let (k1, b) = (self.params.k1, self.params.b);
        let tf = tf as f64;
        let norm = tf + k1 * (1.0 - b + b * len as f64 / avg);
        idf * tf * (k1 + 1.0) / norm
    }

    /// query terms: distinct known tokens with df > 0, in first-occurrence order, with their idf
    fn query_terms(&self, query: &str, s: &mut Scratch) {
        s.q.clear();
        let n = self.n_live as i64;
        let mut tok = std::mem::take(&mut s.tok);
        let mut q = std::mem::take(&mut s.q);
        super::tokens(query, &mut tok, |t| {
            if let Some(tid) = self.terms.lookup(t) {
                if !q.iter().any(|x| x.0 == tid) {
                    let df = self.ts[tid as usize].df as i64;
                    if df > 0 {
                        let idf = (1.0 + ((n - df) as f64 + 0.5) / (df as f64 + 0.5)).ln();
                        q.push((tid, idf));
                    }
                }
            }
        });
        s.tok = tok;
        s.q = q;
    }

    fn taat(&self, s: &mut Scratch, avg: f64) -> Vec<Hit> {
        s.epoch = s.epoch.wrapping_add(1);
        if s.epoch == 0 {
            s.stamp.iter_mut().for_each(|x| *x = 0);
            s.epoch = 1;
        }
        let n = self.doc_len.len();
        if s.acc.len() < n {
            s.acc.resize(n, 0.0);
            s.stamp.resize(n, 0);
        }
        s.touched.clear();
        let epoch = s.epoch;
        let q = std::mem::take(&mut s.q);
        // canonical summation order (shared with the pruned path): bound desc, then query position
        let order = self.canonical_order(&q, avg);
        for &(qi, _) in &order {
            let (tid, idf) = q[qi];
            let visit = |doc: u32, tf: u32, s: &mut Scratch| {
                if !self.is_live(doc) {
                    return;
                }
                let d = doc as usize;
                let sc = self.contrib(idf, tf, self.doc_len[d], avg);
                if s.stamp[d] != epoch {
                    s.stamp[d] = epoch;
                    s.acc[d] = sc;
                    s.touched.push(doc);
                } else {
                    s.acc[d] += sc;
                }
            };
            for seg in &self.segs {
                if let Some(mut it) = seg.postings_of(tid) {
                    while let Some((docs, tfs)) = it.next_block() {
                        for k in 0..docs.len() {
                            visit(docs[k], tfs[k], s);
                        }
                    }
                }
            }
            if (tid as usize) < self.ts.len() {
                let mut p = self.ts[tid as usize].head;
                while p != NONE {
                    visit(self.mem.p_doc[p as usize], self.mem.p_tf[p as usize], s);
                    p = self.mem.p_next[p as usize];
                }
            }
        }
        s.q = q;
        s.touched
            .iter()
            .map(|&d| Hit {
                doc: d,
                score: s.acc[d as usize],
            })
            .filter(|h| h.score > 0.0)
            .collect()
    }

    /// TAAT of one term over every segment into the accumulator (new docs allowed); records the docs it changed
    fn taat_term(&self, s: &mut Scratch, tid: u32, idf: f64, avg: f64) {
        let epoch = s.epoch;
        let visit = |doc: u32, tf: u32, s: &mut Scratch| {
            if !self.is_live(doc) {
                return;
            }
            let d = doc as usize;
            let sc = self.contrib(idf, tf, self.doc_len[d], avg);
            let v = if s.stamp[d] != epoch {
                s.stamp[d] = epoch;
                s.touched.push(doc);
                sc
            } else {
                s.acc[d] + sc
            };
            s.acc[d] = v;
            if v > s.max_acc {
                s.max_acc = v;
            }
            s.updated.push(doc);
        };
        self.for_postings(tid, |doc, tf| visit(doc, tf, s));
    }

    /// every posting (doc, tf) of a term over all segments, in doc order
    #[inline]
    fn for_postings(&self, tid: u32, mut f: impl FnMut(u32, u32)) {
        for seg in &self.segs {
            if let Some(mut it) = seg.postings_of(tid) {
                while let Some((docs, tfs)) = it.next_block() {
                    for k in 0..docs.len() {
                        f(docs[k], tfs[k]);
                    }
                }
            }
        }
        if (tid as usize) < self.ts.len() {
            let mut p = self.ts[tid as usize].head;
            while p != NONE {
                f(self.mem.p_doc[p as usize], self.mem.p_tf[p as usize]);
                p = self.mem.p_next[p as usize];
            }
        }
    }

    /// calls f(doc, tf) for every posting of `tid` at the (ascending) docs of `cands` (block-skipping cursors)
    fn probe(&self, tid: u32, cands: &[u32], mut f: impl FnMut(usize, u32)) {
        let mut ci = 0usize;
        for seg in &self.segs {
            while ci < cands.len() && cands[ci] < seg.lo {
                ci += 1;
            }
            if ci >= cands.len() {
                return;
            }
            if cands[ci] >= seg.hi {
                continue;
            }
            let end = ci + cands[ci..].partition_point(|&d| d < seg.hi);
            if let Some(i) = seg.find(tid) {
                let mut c = seg.cursor(i);
                for (j, &d) in cands[ci..end].iter().enumerate() {
                    c.advance(d);
                    if c.doc == NONE {
                        break;
                    }
                    if c.doc == d {
                        f(ci + j, c.tf());
                    }
                }
            }
            ci = end;
        }
        if ci < cands.len() && (tid as usize) < self.ts.len() && self.ts[tid as usize].head != NONE
        {
            let mut p = self.ts[tid as usize].head;
            for (j, &d) in cands[ci..].iter().enumerate() {
                while p != NONE && self.mem.p_doc[p as usize] < d {
                    p = self.mem.p_next[p as usize];
                }
                if p == NONE {
                    break;
                }
                if self.mem.p_doc[p as usize] == d {
                    f(ci + j, self.mem.p_tf[p as usize]);
                }
            }
        }
    }

    /// the canonical term order of a query: (index, bound) sorted by bound desc, then query position. Every path adds a
    /// doc's contributions in this order, so equal contribution sets give bit-identical scores (ties stay ties)
    fn canonical_order(&self, q: &[(u32, f64)], avg: f64) -> Vec<(usize, f64)> {
        let mut order: Vec<(usize, f64)> = q
            .iter()
            .enumerate()
            .map(|(i, &(tid, idf))| (i, self.term_bound(tid, idf, avg)))
            .collect();
        order.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        order
    }

    /// upper bound of a term's contribution over all segments
    fn term_bound(&self, tid: u32, idf: f64, avg: f64) -> f64 {
        let mut ub: f64 = 0.0;
        for seg in &self.segs {
            if let Some(i) = seg.find(tid) {
                ub = ub.max(self.contrib(idf, seg.max_tf[i], seg.min_len[i], avg));
            }
        }
        let t = tid as usize;
        if t < self.ts.len() && self.ts[t].head != NONE {
            ub = ub.max(self.contrib(idf, self.ts[t].max_tf, self.ts[t].min_len, avg));
        }
        ub * (1.0 + 1e-9) + 1e-12
    }

    /// k-th largest accumulated score among `docs` (-inf when fewer than k)
    fn kth(acc: &[f64], docs: &[u32], k: usize, buf: &mut Vec<f64>) -> f64 {
        if docs.len() < k {
            return f64::NEG_INFINITY;
        }
        buf.clear();
        buf.extend(docs.iter().map(|&d| acc[d as usize]));
        let (_, kth, _) = buf.select_nth_unstable_by(k - 1, |a, b| b.total_cmp(a));
        *kth
    }

    /// exact pruned TAAT (module docs): returns the top k + ties with canonical scores
    fn pruned(&self, s: &mut Scratch, k: usize, avg: f64) -> Vec<Hit> {
        s.epoch = s.epoch.wrapping_add(1);
        if s.epoch == 0 {
            s.stamp.iter_mut().for_each(|x| *x = 0);
            s.epoch = 1;
        }
        let n = self.doc_len.len();
        if s.acc.len() < n {
            s.acc.resize(n, 0.0);
            s.stamp.resize(n, 0);
        }
        if s.cmark.len() < n {
            s.cmark.resize(n, 0);
        }
        s.touched.clear();
        let q = std::mem::take(&mut s.q);
        let m = q.len();
        let margin = |t: f64| t.abs() * 1e-9 + 1e-12;
        let order = self.canonical_order(&q, avg);
        let mut suffix = vec![0f64; m + 1];
        for i in (0..m).rev() {
            suffix[i] = suffix[i + 1] + order[i].1;
        }
        // phase 1: full TAAT while unseen docs could still reach the top k. theta = the exact k-th best partial score
        // (a lower bound of the final k-th score), maintained from the previous top k plus the docs the term changed
        let mut theta = f64::NEG_INFINITY;
        let mut top: Vec<u32> = Vec::with_capacity(k);
        let mut pool: Vec<(f64, u32)> = Vec::new();
        let mut done = 0usize;
        s.updated.clear();
        s.max_acc = f64::NEG_INFINITY;
        while done < m {
            let (qi, _) = order[done];
            self.taat_term(s, q[qi].0, q[qi].1, avg);
            done += 1;
            if done < m && suffix[done] >= s.max_acc {
                continue; // theta <= max_acc <= remaining bound: no switch possible yet; keep collecting changed docs
            }
            s.cepoch = s.cepoch.wrapping_add(1).max(1);
            pool.clear();
            for &d in top.iter().chain(s.updated.iter()) {
                if s.cmark[d as usize] != s.cepoch {
                    s.cmark[d as usize] = s.cepoch;
                    pool.push((s.acc[d as usize], d));
                }
            }
            s.updated.clear();
            if pool.len() >= k {
                pool.select_nth_unstable_by(k - 1, |a, b| b.0.total_cmp(&a.0));
                theta = pool[k - 1].0;
                top.clear();
                top.extend(pool[..k].iter().map(|x| x.1));
            } else {
                top.clear();
                top.extend(pool.iter().map(|x| x.1));
            }
            if suffix[done] < theta - margin(theta) {
                break;
            }
        }
        s.last = PrunedStats {
            terms: m,
            phase1_terms: done,
            postings_phase1: order[..done]
                .iter()
                .map(|&(qi, _)| self.ts[q[qi].0 as usize].df as u64)
                .sum(),
            postings_total: q.iter().map(|&(t, _)| self.ts[t as usize].df as u64).sum(),
            touched: s.touched.len(),
            ..PrunedStats::default()
        };
        // phase 2: remaining (low-bound) terms only for candidates that can still reach theta; per term, scan its
        // postings against the candidate marks or probe the candidates with block-skipping cursors, whichever is cheaper
        let mut cands: Vec<u32> = s
            .touched
            .iter()
            .copied()
            .filter(|&d| s.acc[d as usize] + suffix[done] >= theta - margin(theta))
            .collect();
        let mut sorted = false;
        s.last.cands_phase2 = if done < m { cands.len() } else { 0 };
        let mut buf = Vec::new();
        // candidate marks are set once; pruned candidates keep their (now irrelevant) mark: accumulating into them is harmless
        s.cepoch = s.cepoch.wrapping_add(1).max(1);
        let ce = s.cepoch;
        if done < m {
            for &d in &cands {
                s.cmark[d as usize] = ce;
            }
        }
        while done < m && !cands.is_empty() {
            let (qi, _) = order[done];
            let (tid, idf) = q[qi];
            if (self.ts[tid as usize].df as usize) < 16 * cands.len() {
                let (acc, cmark, dl) = (&mut s.acc, &s.cmark, &self.doc_len);
                self.for_postings(tid, |doc, tf| {
                    let d = doc as usize;
                    if cmark[d] == ce {
                        acc[d] += self.contrib(idf, tf, dl[d], avg);
                    }
                });
            } else {
                if !sorted {
                    cands.sort_unstable();
                    sorted = true;
                }
                let acc = &mut s.acc;
                let dl = &self.doc_len;
                self.probe(tid, &cands, |j, tf| {
                    let d = cands[j] as usize;
                    acc[d] += self.contrib(idf, tf, dl[d], avg);
                });
            }
            done += 1;
            // prune with the phase-1 theta (a valid lower bound); the exact k-th is selected once at the end
            let rest = suffix[done];
            if done < m {
                cands.retain(|&d| s.acc[d as usize] + rest >= theta - margin(theta));
            }
        }
        // every candidate was scored term by term in the canonical order (phase 1 TAAT, then phase 2): its accumulator
        // IS the canonical score. Return the k-th score and everything tied with or above it.
        s.last.rescored = 0;
        let kth = Self::kth(&s.acc, &cands, k, &mut buf);
        let hits = cands
            .iter()
            .map(|&d| Hit {
                doc: d,
                score: s.acc[d as usize],
            })
            .filter(|h| h.score > 0.0 && h.score >= kth)
            .collect();
        s.q = q;
        hits
    }

    fn search_impl(&self, query: &str, k: usize, ties: bool) -> Vec<Hit> {
        if k == 0 || self.n_live == 0 {
            return Vec::new();
        }
        let mut s = self.scratch.lock().unwrap().pop().unwrap_or_default();
        self.query_terms(query, &mut s);
        let avg = self.total_len as f64 / (self.n_live.max(1)) as f64;
        let mut hits = if self.algo == SearchAlgo::Pruned && s.q.len() > 1 {
            self.pruned(&mut s, k, avg)
        } else {
            self.taat(&mut s, avg)
        };
        self.scratch.lock().unwrap().push(s);
        let cmp = |a: &Hit, b: &Hit| b.score.total_cmp(&a.score).then(a.doc.cmp(&b.doc));
        if hits.len() > k {
            hits.select_nth_unstable_by(k - 1, cmp);
            let kth = hits[k - 1].score;
            if ties {
                hits.retain(|h| h.score >= kth);
            } else {
                hits.truncate(k);
            }
        }
        hits.sort_unstable_by(cmp);
        hits
    }

    /// pruned search + its diagnostics
    pub fn explain(&self, query: &str, k: usize) -> (Vec<Hit>, PrunedStats) {
        let mut s = self.scratch.lock().unwrap().pop().unwrap_or_default();
        self.query_terms(query, &mut s);
        let avg = self.total_len as f64 / (self.n_live.max(1)) as f64;
        let hits = if s.q.len() > 1 {
            self.pruned(&mut s, k, avg)
        } else {
            Vec::new()
        };
        let st = s.last;
        self.scratch.lock().unwrap().push(s);
        (hits, st)
    }

    // -------------------------------------------------------------------------------------------- persistence
    /// Write this index as sections `base..` of a checkpoint (seals first; waits for a running merge).
    pub fn write_sections(&mut self, w: &mut SectionWriter, base: u32) -> Result<(), StoreError> {
        self.seal();
        self.install_merges(true);
        let meta = serde_json::json!({
            "format": 2, "k1": self.params.k1, "b": self.params.b, "n_live": self.n_live, "total_len": self.total_len,
            "segments": self.segs.len(), "next_doc": self.mem.hi,
        });
        w.section(base, &serde_json::to_vec(&meta)?)?;
        let (tb, te) = self.terms.raw_parts();
        w.section(base + 1, tb)?;
        w.pod(base + 2, te)?;
        let df: Vec<u32> = self.ts.iter().map(|t| t.df).collect();
        w.pod(base + 3, &df)?;
        w.pod(base + 4, &self.doc_len)?;
        w.pod(base + 5, &self.live)?;
        for (i, s) in self.segs.iter().enumerate() {
            let t = base + 16 + 16 * i as u32;
            w.pod(t, &[s.lo as u64, s.hi as u64, s.postings])?;
            w.pod(t + 1, &s.terms)?;
            w.pod(t + 2, &s.offs)?;
            w.pod(t + 3, &s.counts)?;
            w.pod(t + 4, &s.max_tf)?;
            w.pod(t + 5, &s.min_len)?;
            w.pod(t + 6, &s.skip_off)?;
            w.pod(t + 7, &s.skip_last)?;
            w.pod(t + 8, &s.skip_start)?;
            w.section(t + 9, s.data.as_slice())?;
        }
        Ok(())
    }

    /// Load from checkpoint sections (segment postings stay memory-mapped).
    pub fn read_sections(
        s: &Sections,
        base: u32,
        params: Bm25Params,
    ) -> Result<SegmentedBm25, StoreError> {
        let meta: serde_json::Value = serde_json::from_slice(s.bytes(base)?)?;
        if meta["format"].as_u64() != Some(2) {
            return Err(StoreError::Format("unsupported bm25 section format".into()));
        }
        let mut x = SegmentedBm25::new(params);
        x.terms = StrPool::from_raw_parts(s.bytes(base + 1)?.to_vec(), s.vec(base + 2)?)
            .map_err(StoreError::Format)?;
        let df: Vec<u32> = s.vec(base + 3)?;
        x.ts = df
            .into_iter()
            .map(|df| TermSt {
                df,
                ..TermSt::default()
            })
            .collect();
        x.doc_len = s.vec(base + 4)?;
        x.live = s.vec(base + 5)?;
        x.n_live = meta["n_live"].as_u64().unwrap_or(0);
        x.total_len = meta["total_len"].as_u64().unwrap_or(0);
        let nseg = meta["segments"].as_u64().unwrap_or(0) as u32;
        for i in 0..nseg {
            let t = base + 16 + 16 * i;
            let h: Vec<u64> = s.vec(t)?;
            x.segs.push(Arc::new(Segment {
                lo: h[0] as u32,
                hi: h[1] as u32,
                postings: h[2],
                terms: s.vec(t + 1)?,
                offs: s.vec(t + 2)?,
                counts: s.vec(t + 3)?,
                max_tf: s.vec(t + 4)?,
                min_len: s.vec(t + 5)?,
                skip_off: s.vec(t + 6)?,
                skip_last: s.vec(t + 7)?,
                skip_start: s.vec(t + 8)?,
                data: s.blob(t + 9)?,
            }));
        }
        let next = meta["next_doc"].as_u64().unwrap_or(0) as u32;
        x.mem.lo = next;
        x.mem.hi = next;
        Ok(x)
    }
}

impl Bm25Backend for SegmentedBm25 {
    fn term_df(&self, term: &str) -> Option<u32> {
        Some(
            self.terms
                .lookup(term)
                .map(|tid| self.ts[tid as usize].df)
                .unwrap_or(0),
        )
    }
    fn name(&self) -> &'static str {
        "xerj-native-segmented"
    }

    fn add(&mut self, doc: u32, text: &str) {
        if self.background && self.merge_busy {
            self.install_merges(false);
        }
        assert!(
            doc >= self.mem.hi || (self.mem.p_doc.is_empty() && doc >= self.mem.lo),
            "bm25 doc ids must increase"
        );
        if self.mem.p_doc.is_empty() {
            self.mem.lo = doc;
        }
        let mut tok = std::mem::take(&mut self.tok);
        let mut tids = std::mem::take(&mut self.tids);
        tids.clear();
        super::tokens(text, &mut tok, |t| tids.push(self.term_id(t)));
        let len = tids.len() as u32;
        tids.sort_unstable();
        let d = doc as usize;
        if d >= self.doc_len.len() {
            self.doc_len.resize(d + 1, 0);
        }
        self.doc_len[d] = len;
        let mut i = 0;
        while i < tids.len() {
            let t = tids[i];
            let mut j = i + 1;
            while j < tids.len() && tids[j] == t {
                j += 1;
            }
            let st = &mut self.ts[t as usize];
            st.df += 1;
            self.mem.push(st, t, doc, (j - i) as u32, len);
            i = j;
        }
        self.tok = tok;
        self.tids = tids;
        let w = d >> 6;
        if w >= self.live.len() {
            self.live.resize(w + 1, 0);
        }
        self.live[w] |= 1 << (d & 63);
        self.n_live += 1;
        self.total_len += len as u64;
        self.mem.hi = doc + 1;
        if self.mem.p_doc.len() >= self.seal_postings {
            self.seal();
        }
    }

    fn delete(&mut self, doc: u32, text: &str) {
        if !self.is_live(doc) {
            return;
        }
        let d = doc as usize;
        self.live[d >> 6] &= !(1 << (d & 63));
        self.n_live -= 1;
        self.total_len -= self.doc_len[d] as u64;
        let mut tok = std::mem::take(&mut self.tok);
        let mut tids = std::mem::take(&mut self.tids);
        tids.clear();
        super::tokens(text, &mut tok, |t| {
            if let Some(id) = self.terms.lookup(t) {
                tids.push(id);
            }
        });
        tids.sort_unstable();
        tids.dedup();
        for &t in &tids {
            self.ts[t as usize].df -= 1;
        }
        self.tok = tok;
        self.tids = tids;
    }

    fn commit(&mut self) {
        self.seal();
        self.install_merges(false);
        // release the in-memory segment's buffers after a batch (they regrow on the next adds)
        self.mem.p_doc = Vec::new();
        self.mem.p_tf = Vec::new();
        self.mem.p_next = Vec::new();
        self.mem.touched = Vec::new();
    }

    fn search(&self, query: &str, k: usize) -> Vec<Hit> {
        self.search_impl(query, k, false)
    }

    fn search_with_ties(&self, query: &str, k: usize) -> Vec<Hit> {
        self.search_impl(query, k, true)
    }

    fn live_docs(&self) -> u64 {
        self.n_live
    }

    fn heap_bytes(&self) -> usize {
        self.terms.heap_bytes()
            + self.ts.capacity() * std::mem::size_of::<TermSt>()
            + self.doc_len.capacity() * 4
            + self.live.capacity() * 8
            + self.mem.heap_bytes()
            + self.segs.iter().map(|s| s.heap_bytes()).sum::<usize>()
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn stats(&self) -> Bm25Stats {
        Bm25Stats {
            live_docs: self.n_live,
            terms: self.terms.len() as u64,
            segments: self.num_segments() as u64,
            mem_postings: self.mem.p_doc.len() as u64,
            sealed_postings: self.segs.iter().map(|s| s.postings).sum(),
            seals: self.seals,
            merges: self.merges,
            merge_ms: self.merge_ms,
            seal_ms: self.seal_ms,
            heap_bytes: self.heap_bytes() as u64,
            mapped_bytes: self.segs.iter().map(|s| s.mapped_bytes() as u64).sum(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn brute(docs: &[(u32, &str, bool)], q: &str, k1: f64, b: f64) -> Vec<(u32, f64)> {
        // the reference LexicalIndex over the live docs
        let toks = |t: &str| {
            let mut s = String::new();
            let mut v = Vec::new();
            super::super::tokens(t, &mut s, |x| v.push(x.to_string()));
            v
        };
        let live: Vec<_> = docs.iter().filter(|d| d.2).collect();
        let rows: Vec<Vec<String>> = live.iter().map(|d| toks(d.1)).collect();
        let n = rows.len() as f64;
        let avg = rows.iter().map(|r| r.len()).sum::<usize>() as f64 / n.max(1.0);
        let mut qt = toks(q);
        qt.sort();
        qt.dedup();
        let mut out = Vec::new();
        for (d, row) in live.iter().zip(&rows) {
            let len = row.len() as f64;
            let mut sc = 0.0;
            for t in &qt {
                let tf = row.iter().filter(|x| *x == t).count() as f64;
                if tf > 0.0 {
                    let df = rows.iter().filter(|r| r.contains(t)).count() as f64;
                    let idf = (1.0 + (n - df + 0.5) / (df + 0.5)).ln();
                    sc += idf * tf * (k1 + 1.0) / (tf + k1 * (1.0 - b + b * len / avg));
                }
            }
            if sc > 0.0 {
                out.push((d.0, sc));
            }
        }
        out.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        out
    }

    fn build(
        words: &[&str],
        ndocs: u32,
        seal: usize,
        factor: usize,
        seed0: u64,
    ) -> (SegmentedBm25, Vec<(u32, String, bool)>) {
        let mut docs: Vec<(u32, String, bool)> = Vec::new();
        let mut ix = SegmentedBm25::new(Bm25Params::default());
        ix.seal_postings = seal;
        ix.merge_factor = factor;
        let mut seed = seed0;
        let mut rnd = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (seed >> 33) as usize
        };
        for d in 0..ndocs {
            let n = 1 + rnd() % 12;
            // Zipf-ish: low word indexes are frequent ("stopwords")
            let text: Vec<&str> = (0..n)
                .map(|_| words[(rnd() % words.len()).min(rnd() % words.len())])
                .collect();
            let text = text.join(" ");
            ix.add(d, &text);
            docs.push((d, text, true));
            if d % 7 == 3 {
                let victim = (rnd() % (d as usize + 1)) as u32;
                if docs[victim as usize].2 {
                    ix.delete(victim, &docs[victim as usize].1.clone());
                    docs[victim as usize].2 = false;
                }
            }
        }
        (ix, docs)
    }

    #[test]
    fn matches_bruteforce_across_seals_merges_deletes() {
        let words = [
            "the",
            "of",
            "is",
            "alpha",
            "beta",
            "gamma",
            "delta",
            "Ünïcode",
            "straße",
            "x1",
            "y_2",
            "zeta",
            "eta",
            "omega",
            "kappa",
            "rho",
        ];
        for (seal, factor, seed) in [(64usize, 3usize, 12345u64), (1 << 20, 8, 7), (500, 2, 99)] {
            let (mut ix, docs) = build(&words, 3000, seal, factor, seed);
            let view: Vec<(u32, &str, bool)> =
                docs.iter().map(|d| (d.0, d.1.as_str(), d.2)).collect();
            for round in 0..2 {
                for q in [
                    "alpha beta",
                    "STRASSE gamma the",
                    "ünïcode",
                    "x1 y_2 zeta eta of is",
                    "the of is",
                    "omega kappa rho the",
                    "nothing",
                ] {
                    for k in [1usize, 5, 25] {
                        let exp = brute(&view, q, 1.2, 0.75);
                        for algo in [SearchAlgo::Pruned, SearchAlgo::Taat] {
                            ix.algo = algo;
                            let got = ix.search(q, k);
                            let e: Vec<_> = exp.iter().take(k).collect();
                            assert_eq!(got.len(), e.len(), "{q} {algo:?} k={k}");
                            for (g, e) in got.iter().zip(&e) {
                                assert!(
                                    (g.score - e.1).abs() < 1e-9,
                                    "{q} {algo:?}: {g:?} vs {e:?}"
                                );
                            }
                            assert_eq!(
                                got.iter().map(|h| h.doc).collect::<Vec<_>>(),
                                e.iter().map(|h| h.0).collect::<Vec<_>>(),
                                "{q} {algo:?} k={k}"
                            );
                            // ties: every doc tied with the k-th score is returned
                            let t = ix.search_with_ties(q, k);
                            if let Some(kth) = e.last() {
                                let n_ties = exp.iter().filter(|x| x.1 >= kth.1).count();
                                assert_eq!(t.len(), n_ties, "{q} {algo:?} ties k={k}");
                            }
                        }
                        // pruned and TAAT scores are bit-identical (canonical summation order)
                        ix.algo = SearchAlgo::Pruned;
                        let a = ix.search(q, k);
                        ix.algo = SearchAlgo::Taat;
                        let b = ix.search(q, k);
                        assert_eq!(a, b, "{q} k={k}");
                    }
                }
                if round == 0 {
                    ix.force_merge();
                    assert_eq!(ix.num_segments(), 1);
                }
            }
        }
    }
}
