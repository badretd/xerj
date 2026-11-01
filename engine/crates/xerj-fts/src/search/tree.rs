//! Block-max MaxScore over a scoring TREE of term leaves (opt-in,
//! `XERJ_FTS_TREE_WALK`).
//!
//! [`FtsSearcher::wand_should_bool`] serves the flat `bool{should:[term..]}`
//! that a one-field `match` projects to. The bodies the product's own clients
//! send project to deeper trees, and those take the generic path, which
//! executes every clause into a `Vec<ScoredHit>` and folds the lists through
//! hash maps (`execute_bool`) — O(postings) hash-map work per clause:
//!
//! * `multi_match` `most_fields` over several fields (the recall leg of
//!   `xerj search` and of the MCP `xerj_search` tool):
//!   `bool{should:[bool{should:[title:t..]}, bool{should:[text:t..]}]}`;
//! * the coverage clauses `xerj search` appends (#2327): a per-token
//!   `bool{should:[..], minimum_should_match: k}` and an `operator: and`
//!   copy, `bool{should:[bool{must:[title:t..]}, bool{must:[text:t..]}]}`;
//! * the console's one-`match`-per-field `bool`.
//!
//! This walk evaluates such a tree document at a time, with one block cursor
//! per DISTINCT `(field, term)`, and prunes with the bounds of the flat walk
//! applied to the tree's sum.
//!
//! ## Supported trees
//!
//! `Bool` nodes with `must` and `should` children (any
//! `minimum_should_match`; no `filter`, no `must_not`), `Term` leaves, and
//! `Phrase` leaves (materialised once through [`FtsSearcher::execute`], the
//! generic path's own evaluation). Every boost finite and `>= 0`, at most
//! [`MAX_LEAVES`] leaves, a `Bool` at the root. Anything else returns
//! `Ok(None)` and the caller takes the generic path unchanged.
//!
//! ## Why the output is the generic path's
//!
//! * **Matching.** A `Bool` matches `d` iff every `must` child matches and
//!   at least `min_should` `should` children match, where `min_should` is
//!   `execute_bool`'s minimum (`min_should_match`, else 1 when the bool has
//!   no `must` and at least one `should`, else 0), raised to 1 when there is
//!   no `must` (`execute_bool` then takes the `should` union as its
//!   candidates, so at least one `should` has to match).
//! * **Scores, bit for bit.** A term leaf contributes
//!   `score_term(df, tf, dl) * boost` (f32), exactly as `scan_term` forms it;
//!   a `Bool` sums its matching children in f32 starting from 0.0, `must`
//!   children first and then `should` children, each in clause order — the
//!   order `execute_bool`'s `score_map` accumulates in — and multiplies by
//!   its boost; a phrase leaf contributes the generic path's own hit score.
//! * **The page.** A source (a distinct term list, or a phrase's hit list)
//!   contributes to the root at most `weight * raw`, where `weight` is the
//!   sum over the source's leaf occurrences of the product of the boosts on
//!   the path (leaf boost included) and `raw` its unboosted score; the
//!   matching constraints (`must`, minimum should match) only remove
//!   contributions, so a document's root score is at most the sum of
//!   `weight * bound` over the sources that can match it. The bounds are
//!   the flat walk's (block impacts, `impact_bound`), inflated by the same
//!   relative 1e-4, which covers the f32 rounding of a tree of at most
//!   [`MAX_LEAVES`] leaves (each f32 operation adds at most 2^-24 relative
//!   error to a sum of non-negative terms). A document is skipped only when
//!   that bound is strictly below `theta` (the page's worst score, never
//!   below the caller's floor), so it could not enter; every other document
//!   is evaluated exactly and offered to the same [`TopN`]. The page is
//!   therefore the generic page — or, with a floor, the generic page's
//!   documents scoring at least the floor, which is the contract of
//!   [`FtsSearcher::search_bounded_counting_floor`].
//! * **The total** is exact below `count_limit`, otherwise at least
//!   `count_limit` and never above the exact total: a source reached from
//!   the root through plain disjunctions only (no `must`, minimum 1 on every
//!   bool of the path) is a subset of the match set, so its length is a
//!   lower bound; a tree made only of plain disjunctions matches exactly the
//!   union of its sources (counted through a bitset); any other tree is
//!   counted document at a time, stopping at `count_limit`.

use super::{
    impact_bound, inflate, norm_c_table, FtsSearcher, Query, ScoredHit, TermQuery, TopN,
    WalkCounters,
};
use crate::bm25::Bm25Scorer;
use crate::index::FieldLengths;
use crate::postings::{BlockCursor, TermPostings};
use anyhow::Result;

/// Leaf occurrences (term and phrase leaves, repeats included) above which
/// the walk declines: the rounding margin of the bounds (module doc) holds
/// for trees up to this size.
pub(crate) const MAX_LEAVES: usize = 512;

/// Whether this process routes the supported query trees through the tree
/// walk: on, unless `XERJ_FTS_TREE_WALK` = `0` / `false` / `off` / `no` (read
/// once). The walk returns the page and totals of the generic path
/// (`tests/tree_walk_parity.rs`), so the switch only opts out of the speed-up.
pub fn tree_walk_enabled() -> bool {
    if let Some(on) = TREE_WALK_THREAD_OVERRIDE.with(std::cell::Cell::get) {
        return on;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        xerj_common::resource::switch_setting(std::env::var("XERJ_FTS_TREE_WALK").ok().as_deref())
    })
}

thread_local! {
    /// An override of [`tree_walk_enabled`] for this thread, for the engine's
    /// parity tests: a searcher is built by the engine, past any builder.
    #[doc(hidden)]
    pub static TREE_WALK_THREAD_OVERRIDE: std::cell::Cell<Option<bool>> =
        const { std::cell::Cell::new(None) };
}

/// The query tree compiled against one segment.
enum Node {
    /// a term leaf: its source and the term query's own boost
    Term { src: usize, boost: f32 },
    /// a materialised leaf (a phrase): its source; the score carries the boost
    Mat { src: usize },
    /// matches nothing in this segment (term absent, phrase without hits)
    Never,
    Bool {
        must: Vec<Node>,
        should: Vec<Node>,
        /// matching `should` children needed (see the module doc)
        min_should: u32,
        boost: f32,
    },
}

/// Where a source's documents come from.
enum Spec<'q> {
    Term {
        tq: &'q TermQuery,
        tp: TermPostings,
    },
    /// a materialised hit list, ascending by doc id
    Mat {
        docs: Vec<u32>,
        scores: Vec<f32>,
    },
}

/// The compiled tree with its sources.
struct Plan<'q> {
    root: Node,
    specs: Vec<Spec<'q>>,
    /// per source: sum over its occurrences of the product of path boosts
    weight: Vec<f64>,
    /// per source: reached from the root through plain disjunctions only
    pure: Vec<bool>,
    /// every bool of the tree is a plain disjunction (no `must`, minimum 1)
    all_pure: bool,
}

struct Builder<'q> {
    specs: Vec<Spec<'q>>,
    weight: Vec<f64>,
    pure: Vec<bool>,
    all_pure: bool,
    leaves: usize,
}

/// One cursor of the walk.
struct Src<'a> {
    /// term source: its block cursor (`None` for a materialised list or a
    /// list the reader could not open, which then matches nothing)
    cur: Option<BlockCursor<'a>>,
    bm25: Option<Bm25Scorer>,
    score_df: u64,
    lengths: Option<FieldLengths<'a>>,
    table: usize,
    idf: f64,
    k1p1: f64,
    /// materialised source: index into `Plan::specs`, and the position
    mat: Option<usize>,
    pos: usize,
    /// sum of the path-boost products over the source's occurrences
    w: f64,
    /// unboosted bound over the whole list (term) / the list's maximum (mat)
    ub: f64,
    bb_blk: usize,
    bb_val: f64,
    sh: usize,
    /// `raw` is the unboosted score at `raw_doc`
    raw: f32,
    raw_doc: u32,
    /// bound of `w * raw` at the candidate being evaluated
    at_d: f64,
    /// at the candidate being evaluated: whether the source may hold it
    /// (false: it cannot), and the unboosted bound of its score there
    may: bool,
    bd: f64,
}

impl Src<'_> {
    #[inline]
    fn doc(&self, specs: &[Spec<'_>]) -> u32 {
        if let Some(c) = &self.cur {
            return c.doc;
        }
        match self.mat {
            Some(m) => match &specs[m] {
                Spec::Mat { docs, .. } => docs.get(self.pos).copied().unwrap_or(u32::MAX),
                Spec::Term { .. } => u32::MAX,
            },
            None => u32::MAX,
        }
    }

    #[inline]
    fn next(&mut self) {
        if let Some(c) = &mut self.cur {
            c.next();
        } else {
            self.pos += 1;
        }
    }

    #[inline]
    fn advance(&mut self, target: u32, specs: &[Spec<'_>]) {
        if let Some(c) = &mut self.cur {
            if c.doc < target {
                c.advance(target);
            }
            return;
        }
        if let Some(m) = self.mat {
            if let Spec::Mat { docs, .. } = &specs[m] {
                let from = self.pos.min(docs.len());
                self.pos = from + docs[from..].partition_point(|&x| x < target);
            }
        }
    }

    /// The unboosted score at `d` (the source is on `d`).
    #[inline]
    fn raw_at(&mut self, d: u32, specs: &[Spec<'_>]) -> f32 {
        if self.raw_doc == d {
            return self.raw;
        }
        let v = if let (Some(c), Some(bm25)) = (&self.cur, &self.bm25) {
            let dl = self.lengths.and_then(|l| l.get(d)).unwrap_or(1) as u32;
            bm25.score_term(self.score_df, c.tf(), dl)
        } else if let Some(m) = self.mat {
            match &specs[m] {
                Spec::Mat { scores, .. } => scores[self.pos],
                Spec::Term { .. } => 0.0,
            }
        } else {
            0.0
        };
        self.raw = v;
        self.raw_doc = d;
        v
    }

    #[inline]
    fn block_bound(&mut self, b: usize, norm_c: &[f64; 256]) -> f64 {
        if b == self.bb_blk {
            return self.bb_val;
        }
        let Some(t) = self.cur.as_ref().and_then(|c| c.blocks().copied()) else {
            return self.ub;
        };
        let v = impact_bound(t.block_impacts(b), norm_c, self.k1p1, self.idf);
        self.bb_blk = b;
        self.bb_val = v;
        v
    }

    /// Unboosted bound of the block holding the current document.
    #[inline]
    fn current_block_bound(&mut self, norm_c: &[f64; 256]) -> f64 {
        match &self.cur {
            Some(c) if c.blocks().is_some() => {
                let b = c.block();
                self.block_bound(b, norm_c)
            }
            _ => self.ub,
        }
    }

    /// Last doc id the current block bound covers (`u32::MAX`: the rest of
    /// the list).
    #[inline]
    fn current_block_last_doc(&self) -> u32 {
        match &self.cur {
            Some(c) => c.block_postings().0.last().copied().unwrap_or(u32::MAX),
            None => u32::MAX,
        }
    }

    /// Unboosted bound at `d` without decoding; `None` when the source
    /// cannot hold `d` (its cursor is past `d`, or the block index puts `d`
    /// beyond its list).
    #[inline]
    fn bound_at(&mut self, d: u32, norm_c: &[f64; 256], specs: &[Spec<'_>]) -> Option<f64> {
        let doc = self.doc(specs);
        if doc == u32::MAX || doc > d {
            return None;
        }
        if doc == d {
            return Some(self.current_block_bound(norm_c));
        }
        let Some(t) = self.cur.as_ref().and_then(|c| c.blocks().copied()) else {
            return Some(self.ub);
        };
        let blk = self.cur.as_ref().map_or(0, |c| c.block());
        let b = crate::postings::gallop(t.last_doc, self.sh.max(blk), d);
        self.sh = b;
        if b >= t.len() {
            return None;
        }
        Some(self.block_bound(b, norm_c))
    }
}

/// Exact score of `node` at `d` (`None`: no match), every source sitting on
/// or past `d` — or before `d` only when it cannot match `d`.
fn eval(node: &Node, d: u32, srcs: &mut [Src<'_>], specs: &[Spec<'_>]) -> Option<f32> {
    match node {
        Node::Never => None,
        Node::Term { src, boost } => {
            let s = &mut srcs[*src];
            if s.doc(specs) != d {
                return None;
            }
            Some(s.raw_at(d, specs) * *boost)
        }
        Node::Mat { src } => {
            let s = &mut srcs[*src];
            if s.doc(specs) != d {
                return None;
            }
            Some(s.raw_at(d, specs))
        }
        Node::Bool {
            must,
            should,
            min_should,
            boost,
        } => {
            let mut sum = 0.0f32;
            for c in must {
                sum += eval(c, d, srcs, specs)?;
            }
            let mut matched = 0u32;
            for c in should {
                if let Some(s) = eval(c, d, srcs, specs) {
                    matched += 1;
                    sum += s;
                }
            }
            if matched < *min_should {
                return None;
            }
            Some(sum * *boost)
        }
    }
}

/// Upper bound of `node`'s score at the candidate from the sources' `may`
/// and `bd` (set by the page pass); `None` when the node cannot match it.
/// Every matching source has `may` set and a raw score at most `bd`, and a
/// node that matches has its constraints met by matching children, so this
/// bounds [`eval`] from above (in reals; the bounds carry the walk's margin).
fn tree_bound(node: &Node, srcs: &[Src<'_>]) -> Option<f64> {
    match node {
        Node::Never => None,
        Node::Term { src, boost } => {
            let s = &srcs[*src];
            s.may.then(|| s.bd * f64::from(*boost))
        }
        Node::Mat { src } => {
            let s = &srcs[*src];
            s.may.then_some(s.bd)
        }
        Node::Bool {
            must,
            should,
            min_should,
            boost,
        } => {
            let mut sum = 0.0f64;
            for c in must {
                sum += tree_bound(c, srcs)?;
            }
            let mut can = 0u32;
            for c in should {
                if let Some(b) = tree_bound(c, srcs) {
                    can += 1;
                    sum += b;
                }
            }
            if can < *min_should {
                return None;
            }
            Some(sum * f64::from(*boost))
        }
    }
}

/// Whether `node` matches `d` (sources positioned as for [`eval`]).
fn matches(node: &Node, d: u32, srcs: &[Src<'_>], specs: &[Spec<'_>]) -> bool {
    match node {
        Node::Never => false,
        Node::Term { src, .. } | Node::Mat { src } => srcs[*src].doc(specs) == d,
        Node::Bool {
            must,
            should,
            min_should,
            ..
        } => {
            if !must.iter().all(|c| matches(c, d, srcs, specs)) {
                return false;
            }
            if *min_should == 0 {
                return true;
            }
            let mut matched = 0u32;
            for c in should {
                if matches(c, d, srcs, specs) {
                    matched += 1;
                    if matched >= *min_should {
                        return true;
                    }
                }
            }
            false
        }
    }
}

impl FtsSearcher {
    /// Compile `q` into a node; `Ok(None)`: a shape this walk does not take.
    fn tree_compile<'q>(
        &self,
        q: &'q Query,
        path: f64,
        pure_chain: bool,
        b: &mut Builder<'q>,
    ) -> Result<Option<Node>> {
        match q {
            Query::Term(tq) => {
                if !tq.boost.is_finite() || tq.boost < 0.0 {
                    return Ok(None);
                }
                b.leaves += 1;
                if b.leaves > MAX_LEAVES {
                    return Ok(None);
                }
                let existing = b.specs.iter().position(|s| match s {
                    Spec::Term { tq: o, .. } => o.field == tq.field && o.term == tq.term,
                    Spec::Mat { .. } => false,
                });
                let src = match existing {
                    Some(i) => i,
                    None => {
                        let Some(tp) = self.reader.lookup_term(&tq.field, &tq.term) else {
                            return Ok(Some(Node::Never));
                        };
                        b.specs.push(Spec::Term { tq, tp });
                        b.weight.push(0.0);
                        b.pure.push(false);
                        b.specs.len() - 1
                    }
                };
                b.weight[src] += path * f64::from(tq.boost);
                b.pure[src] |= pure_chain;
                Ok(Some(Node::Term {
                    src,
                    boost: tq.boost,
                }))
            }
            Query::Phrase(pq) => {
                if !pq.boost.is_finite() || pq.boost < 0.0 {
                    return Ok(None);
                }
                b.leaves += 1;
                if b.leaves > MAX_LEAVES {
                    return Ok(None);
                }
                let mut hits = self.execute(q, false)?;
                if hits.is_empty() {
                    return Ok(Some(Node::Never));
                }
                hits.sort_unstable_by_key(|h| h.doc_id);
                let docs: Vec<u32> = hits.iter().map(|h| h.doc_id).collect();
                let scores: Vec<f32> = hits.iter().map(|h| h.score).collect();
                if scores.iter().any(|s| !s.is_finite() || *s < 0.0) {
                    return Ok(None);
                }
                b.specs.push(Spec::Mat { docs, scores });
                b.weight.push(path);
                b.pure.push(pure_chain);
                Ok(Some(Node::Mat {
                    src: b.specs.len() - 1,
                }))
            }
            Query::Bool(bq) => {
                if !bq.filter.is_empty()
                    || !bq.must_not.is_empty()
                    || !bq.boost.is_finite()
                    || bq.boost < 0.0
                {
                    return Ok(None);
                }
                // `execute_bool`'s minimum; with no `must` the candidates are
                // the `should` union, so at least one `should` has to match.
                let min_should = bq.min_should_match.unwrap_or(if bq.must.is_empty() {
                    u32::from(!bq.should.is_empty())
                } else {
                    0
                });
                let min_should = if bq.must.is_empty() {
                    min_should.max(1)
                } else {
                    min_should
                };
                let disjunction = bq.must.is_empty() && min_should == 1;
                if !disjunction {
                    b.all_pure = false;
                }
                let child_path = path * f64::from(bq.boost);
                let mut must = Vec::with_capacity(bq.must.len());
                let mut never = false;
                for c in &bq.must {
                    let Some(n) = self.tree_compile(c, child_path, false, b)? else {
                        return Ok(None);
                    };
                    never |= matches!(n, Node::Never);
                    must.push(n);
                }
                let mut should = Vec::with_capacity(bq.should.len());
                let mut live = 0u32;
                for c in &bq.should {
                    let Some(n) = self.tree_compile(c, child_path, pure_chain && disjunction, b)?
                    else {
                        return Ok(None);
                    };
                    live += u32::from(!matches!(n, Node::Never));
                    should.push(n);
                }
                if never || live < min_should {
                    return Ok(Some(Node::Never));
                }
                Ok(Some(Node::Bool {
                    must,
                    should,
                    min_should,
                    boost: bq.boost,
                }))
            }
            _ => Ok(None),
        }
    }

    fn tree_plan<'q>(&self, query: &'q Query) -> Result<Option<Plan<'q>>> {
        if !matches!(query, Query::Bool(_)) {
            return Ok(None);
        }
        let mut b = Builder {
            specs: Vec::new(),
            weight: Vec::new(),
            pure: Vec::new(),
            all_pure: true,
            leaves: 0,
        };
        let Some(root) = self.tree_compile(query, 1.0, true, &mut b)? else {
            return Ok(None);
        };
        if b.weight.iter().any(|w| !w.is_finite() || *w < 0.0) {
            return Ok(None);
        }
        Ok(Some(Plan {
            root,
            specs: b.specs,
            weight: b.weight,
            pure: b.pure,
            all_pure: b.all_pure,
        }))
    }

    /// Cursors over the plan's sources (`want_tf == false`: counting only).
    fn tree_sources<'a>(
        &'a self,
        plan: &Plan<'_>,
        want_tf: bool,
        norm_tables: &mut Vec<(String, Box<[f64; 256]>)>,
    ) -> Vec<Src<'a>> {
        let mut out = Vec::with_capacity(plan.specs.len());
        for (i, spec) in plan.specs.iter().enumerate() {
            let mut s = Src {
                cur: None,
                bm25: None,
                score_df: 0,
                lengths: None,
                table: 0,
                idf: 0.0,
                k1p1: 0.0,
                mat: None,
                pos: 0,
                w: plan.weight[i],
                ub: 0.0,
                bb_blk: usize::MAX,
                bb_val: 0.0,
                sh: 0,
                raw: 0.0,
                raw_doc: u32::MAX,
                at_d: 0.0,
                may: false,
                bd: 0.0,
            };
            match spec {
                Spec::Mat { scores, .. } => {
                    s.mat = Some(i);
                    let best = scores.iter().copied().fold(0.0f32, f32::max);
                    s.ub = inflate(f64::from(best));
                }
                Spec::Term { tq, tp } => {
                    s.cur = self.reader.block_cursor_with(&tq.field, tp, want_tf);
                    if want_tf {
                        let bm25 = self.make_scorer(&tq.field);
                        s.score_df = self.scoring_df(&tq.field, &tq.term, tp.doc_frequency as u64);
                        s.lengths = self.reader.field_lengths(&tq.field);
                        s.table = match norm_tables.iter().position(|(f, _)| *f == tq.field) {
                            Some(t) => t,
                            None => {
                                norm_tables.push((tq.field.clone(), norm_c_table(&bm25)));
                                norm_tables.len() - 1
                            }
                        };
                        s.idf = f64::from(bm25.idf(s.score_df));
                        s.k1p1 = f64::from(bm25.k1 + 1.0);
                        if let Some(c) = &s.cur {
                            s.ub = if let Some(blocks) = c.blocks() {
                                impact_bound(
                                    blocks.term_impacts,
                                    &norm_tables[s.table].1,
                                    s.k1p1,
                                    s.idf,
                                )
                            } else if c.num_blocks() == 1 {
                                // the whole list is decoded already: its exact maximum
                                let (docs, tfs) = c.block_postings();
                                let mut best = 0.0f32;
                                for (&d, &tf) in docs.iter().zip(tfs) {
                                    let dl = s.lengths.and_then(|l| l.get(d)).unwrap_or(1) as u32;
                                    best = best.max(bm25.score_term(s.score_df, tf, dl));
                                }
                                inflate(f64::from(best))
                            } else {
                                // no block index: the supremum over any (tf, dl)
                                inflate(f64::from(bm25.idf(s.score_df) * (bm25.k1 + 1.0)))
                            };
                        }
                        s.bm25 = Some(bm25);
                    }
                }
            }
            out.push(s);
        }
        out
    }

    /// Top-`cap` evaluation of a supported query tree (module doc); `Ok(None)`
    /// for any other shape.
    pub(super) fn tree_walk(
        &self,
        query: &Query,
        cap: usize,
        count_limit: u64,
        floor: Option<f32>,
    ) -> Result<Option<(Vec<ScoredHit>, u64)>> {
        let Some(plan) = self.tree_plan(query)? else {
            return Ok(None);
        };
        let mut walk = WalkCounters::default();
        let total = self.tree_count(&plan, count_limit, &mut walk);
        let mut top = TopN::new(cap);
        if cap > 0 && !plan.specs.is_empty() && !matches!(plan.root, Node::Never) {
            let mut norm_tables: Vec<(String, Box<[f64; 256]>)> = Vec::new();
            let mut srcs = self.tree_sources(&plan, true, &mut norm_tables);
            self.tree_page(&plan, &mut srcs, &norm_tables, floor, &mut top, &mut walk);
            for s in &srcs {
                if let Some(c) = &s.cur {
                    walk.postings_decoded += c.postings_decoded;
                    walk.blocks_decoded += c.blocks_decoded;
                }
            }
        }
        if let Some(stats) = &self.walk_stats {
            walk.publish(stats);
        }
        let (hits, _) = top.finish();
        Ok(Some((hits, total)))
    }

    /// The total (module doc).
    fn tree_count(&self, plan: &Plan<'_>, count_limit: u64, walk: &mut WalkCounters) -> u64 {
        if count_limit == 0 || matches!(plan.root, Node::Never) || plan.specs.is_empty() {
            return 0;
        }
        let mut no_tables = Vec::new();
        let mut srcs = self.tree_sources(plan, false, &mut no_tables);
        // a plain-disjunction path from the root: a subset of the match set
        let mut lower = 0u64;
        for (i, s) in srcs.iter().enumerate() {
            if !plan.pure[i] {
                continue;
            }
            let len = match (&s.cur, &plan.specs[i]) {
                (Some(c), _) => u64::from(c.doc_frequency()),
                (None, Spec::Mat { docs, .. }) => docs.len() as u64,
                (None, Spec::Term { .. }) => 0,
            };
            lower = lower.max(len);
        }
        if lower >= count_limit {
            return count_limit;
        }
        let mut steps = 0u32;
        if plan.all_pure {
            // the match set is the union of the sources
            let mut bits: Vec<u64> = Vec::new();
            let set = |bits: &mut Vec<u64>, d: u32| {
                let w = (d / 64) as usize;
                if bits.len() <= w {
                    bits.resize(w + 1, 0);
                }
                bits[w] |= 1u64 << (d % 64);
            };
            for (i, s) in srcs.iter_mut().enumerate() {
                if let Some(cur) = &mut s.cur {
                    while cur.doc != u32::MAX {
                        let (docs, _) = cur.block_postings();
                        let last = *docs.last().unwrap_or(&0);
                        for &d in docs {
                            set(&mut bits, d);
                        }
                        walk.postings_counted += docs.len() as u64;
                        cur.advance(last.saturating_add(1));
                        steps = steps.wrapping_add(1);
                        if steps & 4095 == 0 && self.deadline_hit() {
                            return bits.iter().map(|w| u64::from(w.count_ones())).sum();
                        }
                    }
                } else if let Spec::Mat { docs, .. } = &plan.specs[i] {
                    for &d in docs {
                        set(&mut bits, d);
                    }
                }
            }
            return bits.iter().map(|w| u64::from(w.count_ones())).sum();
        }
        // document at a time over the union, stopping at the limit
        let mut count = 0u64;
        loop {
            steps = steps.wrapping_add(1);
            if steps & 4095 == 0 && self.deadline_hit() {
                return count;
            }
            let mut d = u32::MAX;
            for s in &srcs {
                d = d.min(s.doc(&plan.specs));
            }
            if d == u32::MAX {
                return count;
            }
            if matches(&plan.root, d, &srcs, &plan.specs) {
                count += 1;
                if count >= count_limit {
                    return count;
                }
            }
            for s in srcs.iter_mut() {
                if s.doc(&plan.specs) == d {
                    s.next();
                    walk.postings_counted += 1;
                }
            }
        }
    }

    /// The match set of a supported tree as a bitset over the segment's doc
    /// ids (what [`FtsSearcher::matching_docs`] returns): `Ok(None)` for a
    /// shape this walk does not take, `Ok(Some(None))` when the deadline cut
    /// the evaluation short.
    pub(super) fn tree_matching_docs(&self, query: &Query) -> Result<Option<Option<Vec<u64>>>> {
        let Some(plan) = self.tree_plan(query)? else {
            return Ok(None);
        };
        let mut bits: Vec<u64> = Vec::new();
        if matches!(plan.root, Node::Never) || plan.specs.is_empty() {
            return Ok(Some(if self.deadline_tripped() {
                None
            } else {
                Some(bits)
            }));
        }
        let set = |bits: &mut Vec<u64>, d: u32| {
            let w = (d / 64) as usize;
            if bits.len() <= w {
                bits.resize(w + 1, 0);
            }
            bits[w] |= 1u64 << (d % 64);
        };
        let mut no_tables = Vec::new();
        let mut srcs = self.tree_sources(&plan, false, &mut no_tables);
        let mut steps = 0u32;
        if plan.all_pure {
            for (i, s) in srcs.iter_mut().enumerate() {
                if let Some(cur) = &mut s.cur {
                    while cur.doc != u32::MAX {
                        let (docs, _) = cur.block_postings();
                        let last = *docs.last().unwrap_or(&0);
                        for &d in docs {
                            set(&mut bits, d);
                        }
                        cur.advance(last.saturating_add(1));
                        steps = steps.wrapping_add(1);
                        if steps & 4095 == 0 && self.deadline_hit() {
                            return Ok(Some(None));
                        }
                    }
                } else if let Spec::Mat { docs, .. } = &plan.specs[i] {
                    for &d in docs {
                        set(&mut bits, d);
                    }
                }
            }
        } else {
            loop {
                steps = steps.wrapping_add(1);
                if steps & 4095 == 0 && self.deadline_hit() {
                    return Ok(Some(None));
                }
                let mut d = u32::MAX;
                for s in &srcs {
                    d = d.min(s.doc(&plan.specs));
                }
                if d == u32::MAX {
                    break;
                }
                if matches(&plan.root, d, &srcs, &plan.specs) {
                    set(&mut bits, d);
                }
                for s in srcs.iter_mut() {
                    if s.doc(&plan.specs) == d {
                        s.next();
                    }
                }
            }
        }
        // a phrase materialised under a tripped deadline is a subset
        if self.deadline_tripped() {
            return Ok(Some(None));
        }
        Ok(Some(Some(bits)))
    }

    /// The page pass: block-max MaxScore over the sources, exact tree
    /// evaluation of every document that may enter (module doc).
    fn tree_page(
        &self,
        plan: &Plan<'_>,
        srcs: &mut [Src<'_>],
        norm_tables: &[(String, Box<[f64; 256]>)],
        floor: Option<f32>,
        top: &mut TopN,
        walk: &mut WalkCounters,
    ) {
        let specs = &plan.specs;
        let n = srcs.len();
        // sources by whole-list bound ascending (ties: source order)
        let mut by_ub: Vec<usize> = (0..n).collect();
        by_ub.sort_by(|&a, &b| {
            (srcs[a].w * srcs[a].ub)
                .total_cmp(&(srcs[b].w * srcs[b].ub))
                .then_with(|| a.cmp(&b))
        });
        let mut essential = vec![true; n];
        let mut ne_sum = 0.0f64;
        let mut ne_order: Vec<usize> = Vec::with_capacity(n);
        let mut partitioned_at = f64::NEG_INFINITY;
        let mut since_poll = 0u32;
        let mut offered = 0.0f32;
        loop {
            since_poll = since_poll.wrapping_add(1);
            if since_poll & 4095 == 0 && self.deadline_hit() {
                break;
            }
            // the caller's floor, raised by the walks beside this one
            let floor = self.floor_now(floor);
            let theta = match (top.worst(), floor) {
                (Some(w), Some(f)) => Some(f64::from(w.score.max(f))),
                (Some(w), None) => Some(f64::from(w.score)),
                (None, Some(f)) => Some(f64::from(f)),
                (None, None) => None,
            };
            if let Some(theta) = theta {
                if theta > partitioned_at {
                    partitioned_at = theta;
                    let mut sum = 0.0f64;
                    ne_sum = 0.0;
                    ne_order.clear();
                    for &i in &by_ub {
                        sum += srcs[i].w * srcs[i].ub;
                        essential[i] = sum >= theta;
                        if !essential[i] {
                            ne_sum = sum;
                            ne_order.push(i);
                        }
                    }
                    ne_order.reverse();
                }
            }
            // candidate: the smallest current doc of the essential sources
            let mut d = u32::MAX;
            for i in 0..n {
                if essential[i] {
                    d = d.min(srcs[i].doc(specs));
                }
            }
            if d == u32::MAX {
                break;
            }
            if let Some(theta) = theta {
                walk.candidates += 1;
                // (a) window: up to the first end of an essential current block
                let mut window = ne_sum;
                let mut at_d = ne_sum;
                let mut wend = u32::MAX;
                for i in 0..n {
                    if !essential[i] {
                        continue;
                    }
                    let doc = srcs[i].doc(specs);
                    srcs[i].may = doc == d;
                    srcs[i].bd = 0.0;
                    if doc == u32::MAX {
                        continue;
                    }
                    let t = table_at(norm_tables, srcs[i].table);
                    let raw_bb = srcs[i].current_block_bound(t);
                    let bb = srcs[i].w * raw_bb;
                    window += bb;
                    if doc == d {
                        at_d += bb;
                        srcs[i].bd = raw_bb;
                    }
                    wend = wend.min(srcs[i].current_block_last_doc());
                }
                if window < theta {
                    walk.windows_skipped += 1;
                    if wend == u32::MAX {
                        break;
                    }
                    for i in 0..n {
                        if essential[i] {
                            srcs[i].advance(wend.saturating_add(1), specs);
                        }
                    }
                    continue;
                }
                // (b) candidate filter: essential block bounds first ...
                if at_d < theta {
                    advance_essential(srcs, &essential, d, specs);
                    continue;
                }
                // ... then the non-essential sources' block bounds at `d`
                if !ne_order.is_empty() {
                    let mut ub_d = at_d - ne_sum;
                    for &j in &ne_order {
                        let t = table_at(norm_tables, srcs[j].table);
                        let b = srcs[j].bound_at(d, t, specs);
                        srcs[j].may = b.is_some();
                        srcs[j].bd = b.unwrap_or(0.0);
                        srcs[j].at_d = srcs[j].w * srcs[j].bd;
                        ub_d += srcs[j].at_d;
                    }
                    if ub_d < theta {
                        advance_essential(srcs, &essential, d, specs);
                        continue;
                    }
                }
                // ... then the same bounds through the tree: a `must` child
                // that cannot hold `d`, or fewer `should` children that can
                // than the minimum, take a bool's whole bound away
                if tree_bound(&plan.root, srcs).is_none_or(|b| b < theta) {
                    walk.tree_pruned += 1;
                    advance_essential(srcs, &essential, d, specs);
                    continue;
                }
            }
            // (c) exact bound of the essential sources on `d`, then the
            // non-essential ones, largest bound first, stopping as soon as
            // the rest cannot lift `d` to `theta`
            let mut partial = 0.0f64;
            for i in 0..n {
                if essential[i] && srcs[i].doc(specs) == d {
                    let w = srcs[i].w;
                    partial += w * f64::from(srcs[i].raw_at(d, specs));
                }
            }
            let mut skip = false;
            if let Some(theta) = theta {
                let mut rem: f64 = ne_order.iter().map(|&j| srcs[j].at_d).sum();
                for &j in &ne_order {
                    if inflate(partial) + rem < theta {
                        skip = true;
                        break;
                    }
                    rem -= srcs[j].at_d;
                    if !srcs[j].may {
                        continue;
                    }
                    srcs[j].advance(d, specs);
                    if srcs[j].doc(specs) == d {
                        let w = srcs[j].w;
                        partial += w * f64::from(srcs[j].raw_at(d, specs));
                    }
                }
                if !skip && inflate(partial) < theta {
                    skip = true;
                }
            }
            if !skip {
                // every source that may hold `d` sits on it or past it now: the
                // loop above advanced each non-essential one that may hold it,
                // and one that cannot hold `d` never matches in `eval`
                if let Some(score) = eval(&plan.root, d, srcs, specs) {
                    // below the caller's floor: cannot enter the merged page
                    if floor.is_none_or(|f| score >= f) {
                        walk.scored += 1;
                        top.push(ScoredHit {
                            doc_id: d,
                            score,
                            explanation: None,
                        });
                        self.share_floor(top, &mut offered);
                    }
                }
            }
            advance_essential(srcs, &essential, d, specs);
        }
    }
}

/// Norm table of a term source (materialised sources never read it).
static NO_TABLE: [f64; 256] = [0.0; 256];

#[inline]
fn table_at(norm_tables: &[(String, Box<[f64; 256]>)], t: usize) -> &[f64; 256] {
    norm_tables.get(t).map_or(&NO_TABLE, |(_, b)| &**b)
}

/// Move every essential source sitting on `d` to its next document.
#[inline]
fn advance_essential(srcs: &mut [Src<'_>], essential: &[bool], d: u32, specs: &[Spec<'_>]) {
    for (s, &e) in srcs.iter_mut().zip(essential) {
        if e && s.doc(specs) == d {
            s.next();
        }
    }
}
