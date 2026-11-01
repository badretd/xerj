//! BM25 over the same memory: one index over record text, one over source documents, fed by the store's writes.
//!
//! `Bm25Backend` is the small seam the rest of the engine uses, so a tantivy-backed implementation can replace the
//! native one if it wins the comparison (README: adapter sketch). The native backend (`native::SegmentedBm25`) follows
//! production designs: a mutable in-memory segment (searchable immediately = read-your-writes), immutable sealed
//! segments with block bit-packed delta-encoded postings, tiered merges on a background thread, a hash term dictionary
//! with exact LIVE document frequencies (deletes keep df / N / avgdl exact, as the reference LexicalIndex rebuilt per
//! snapshot), and mmap reload of sealed segments.
//!
//! Scoring is reference_core.LexicalIndex's: tokens = Python `\w+` of `casefold()`, idf = ln(1 + (N - df + 0.5) / (df +
//! 0.5)), tf * (k1 + 1) / (tf + k1 * (1 - b + b * len / avglen)), k1 = 1.2, b = 0.75, query terms deduplicated, ties by
//! the caller's key (records: fact id).

pub mod codec;
pub mod native;

pub use native::{PrunedStats, SearchAlgo, SegmentedBm25};

/// One search hit (doc = the caller's id: record id or document index)
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Hit {
    pub doc: u32,
    pub score: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bm25Params {
    pub k1: f64,
    pub b: f64,
}

impl Default for Bm25Params {
    fn default() -> Self {
        Bm25Params { k1: 1.2, b: 0.75 }
    }
}

/// Counters for benchmarks and diagnostics.
#[derive(Clone, Copy, Debug, Default, serde::Serialize)]
pub struct Bm25Stats {
    pub live_docs: u64,
    pub terms: u64,
    pub segments: u64,
    pub mem_postings: u64,
    pub sealed_postings: u64,
    pub seals: u64,
    pub merges: u64,
    pub merge_ms: f64,
    pub seal_ms: f64,
    pub heap_bytes: u64,
    pub mapped_bytes: u64,
}

/// The BM25 seam (native segmented index today; a tantivy adapter can implement the same five calls).
pub trait Bm25Backend: Send + Sync {
    fn name(&self) -> &'static str;
    /// index `text` under `doc` (ids increase with every add)
    fn add(&mut self, doc: u32, text: &str);
    /// remove `doc`; `text` is the text it was indexed with (lets backends keep df exact without a forward index)
    fn delete(&mut self, doc: u32, text: &str);
    /// make every add / delete durable in the index structure (native: seal the in-memory segment, install finished
    /// background merges; adds are searchable before this call)
    fn commit(&mut self);
    /// top `k` by (score desc, doc asc)
    fn search(&self, query: &str, k: usize) -> Vec<Hit>;
    /// top `k` plus every further hit tied with the k-th score (callers that break ties by their own key)
    fn search_with_ties(&self, query: &str, k: usize) -> Vec<Hit> {
        self.search(query, k)
    }
    fn live_docs(&self) -> u64;
    /// document frequency of an analyzed term (None = the backend cannot tell); read-only, used for query-side typo tolerance
    fn term_df(&self, _term: &str) -> Option<u32> {
        None
    }
    fn heap_bytes(&self) -> usize;
    fn stats(&self) -> Bm25Stats;
    /// downcasting hook (checkpointing persists native backends directly)
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any;
}

/// reference_core.LexicalIndex.tokens: Python `\w+` over `text.casefold()`
pub fn tokens(text: &str, scratch: &mut String, emit: impl FnMut(&str)) {
    xerj_store::text::py_word_tokens(text, scratch, emit)
}
