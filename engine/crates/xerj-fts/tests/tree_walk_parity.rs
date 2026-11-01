//! Parity of the tree walk (`FtsSearcher::with_tree_walk(true)`: nested `bool`s of terms and phrases
//! under block-max MaxScore) with the generic path (`search_bounded_observed`, every clause executed
//! and folded through `execute_bool`).
//!
//! Contract under test (`FtsSearcher::search_bounded_counting_floor`, the same as the flat walk's):
//! * the page (top `cap`) is the generic path's page bit for bit: same doc ids, same f32 score bits,
//!   same order (`ScoredHit`: score desc, then doc id asc);
//! * the total is exact when it is below `count_limit`, otherwise at least `count_limit` and never
//!   above the exact total;
//! * with a floor, a segment walk hands back its top `cap` among the documents scoring at least the
//!   floor, so the merged page over several segments is the generic merged page;
//! * `matching_docs` is the generic match set.
//!
//! The trees are the shapes the product's clients send (the `xerj search` / MCP body: a
//! `most_fields` `multi_match`, an exact-phrase leg, the #2327 coverage clauses with
//! `minimum_should_match` and the `operator: and` copy; the console's one-`match`-per-field
//! `bool`) and random nested `bool`s with `must`, `should`, minimum-should-match values, boosts
//! (zero included), repeated clauses and absent terms. Every check also runs on the segment with its
//! `.post` block-index frame stripped (the layout written before the frame existed).
use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::Arc;
use tempfile::TempDir;
use xerj_fts::{
    AnalyzerRegistry, BoolQuery, FieldIndexConfig, FieldValues, FtsIndexReader, FtsIndexWriter,
    FtsSearcher, PhraseQuery, Query, ScoredHit, SharedFloor, TermQuery, WalkStats,
};

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n.max(1)
    }
    /// Zipf-like index in [0, n): low indexes are far more frequent
    fn zipf(&mut self, n: usize) -> usize {
        let r = (self.below(1_000_000) as f64 + 0.5) / 1_000_000.0;
        ((r.powf(3.2)) * n as f64) as usize % n
    }
}

const VOCAB: usize = 300;
const FIELDS: [&str; 3] = ["body", "title", "tags"];

fn word(i: usize) -> String {
    format!("w{i}")
}

/// `docs` documents with fields `body` (positions), `title` (positions) and `tags` (positions);
/// bodies repeat short word runs so that multi-word phrases occur.
fn build_segment(dir: &Path, segment: &str, docs: usize, seed: u64) {
    let registry = Arc::new(AnalyzerRegistry::default());
    let mut writer = FtsIndexWriter::new(dir, segment, Arc::clone(&registry));
    for f in FIELDS {
        writer.configure_field(f, FieldIndexConfig::default());
    }
    let mut rng = Rng(seed);
    for d in 0..docs {
        let len = if rng.below(10) == 0 {
            1 + rng.below(4) as usize
        } else {
            5 + rng.below(90) as usize
        };
        let mut body: Vec<String> = Vec::with_capacity(len + 8);
        while body.len() < len {
            let w = rng.zipf(VOCAB);
            if rng.below(15) == 0 {
                // a fixed run: phrases "w{w} w{w+1} w{w+2}" occur in many documents
                for k in 0..3 {
                    body.push(word((w + k) % VOCAB));
                }
            } else {
                body.push(word(w));
                if rng.below(12) == 0 {
                    for _ in 0..1 + rng.below(5) {
                        body.push(word(w));
                    }
                }
            }
        }
        let title: Vec<String> = (0..1 + rng.below(6))
            .map(|_| word(rng.zipf(VOCAB / 2)))
            .collect();
        let tags: Vec<String> = (0..rng.below(4)).map(|_| word(rng.zipf(40))).collect();
        let mut fields: HashMap<String, FieldValues> = HashMap::new();
        fields.insert("body".into(), FieldValues::from(body.join(" ")));
        fields.insert("title".into(), FieldValues::from(title.join(" ")));
        if !tags.is_empty() {
            fields.insert("tags".into(), FieldValues::from(tags.join(" ")));
        }
        writer.add_document(d as u32, &fields);
    }
    writer.finish().unwrap();
}

/// Remove the block-index frame of every `.post` file of `segment` (see blockmax_parity.rs).
fn strip_block_index(dir: &Path, segment: &str) -> usize {
    let mut stripped = 0;
    for e in std::fs::read_dir(dir).unwrap().flatten() {
        let name = e.file_name().to_string_lossy().into_owned();
        if !name.starts_with(&format!("{segment}.")) || !name.ends_with(".post") {
            continue;
        }
        let bytes = std::fs::read(e.path()).unwrap();
        if bytes.len() < 16 || &bytes[..4] != b"ZPS2" {
            continue;
        }
        let magic = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
        if magic & 0xFFFF_FFF0 != 0x184D_2A50 {
            continue;
        }
        let size = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
        let mut out = bytes[..8].to_vec();
        out.extend_from_slice(&bytes[16 + size..]);
        std::fs::write(e.path(), out).unwrap();
        stripped += 1;
    }
    stripped
}

fn reader(dir: &Path, segment: &str) -> Arc<FtsIndexReader> {
    Arc::new(FtsIndexReader::open(dir, segment, &FIELDS).unwrap())
}

/// (generic, tree walk) searchers over the same reader.
///
/// `XERJ_TREE_PARITY_OFF=1` builds the second searcher WITHOUT the tree walk: the fail-before
/// demonstration (the walk-count assertions then fail, the parity assertions still pass).
fn searchers(r: &Arc<FtsIndexReader>, stats: &Arc<WalkStats>) -> (FtsSearcher, FtsSearcher) {
    let reg = Arc::new(AnalyzerRegistry::default());
    let on = std::env::var("XERJ_TREE_PARITY_OFF").map_or(true, |v| v != "1");
    (
        FtsSearcher::new(Arc::clone(r), Arc::clone(&reg)).with_tree_walk(false),
        FtsSearcher::new(Arc::clone(r), reg)
            .with_tree_walk(on)
            .with_walk_stats(Some(Arc::clone(stats))),
    )
}

fn page(hits: &[ScoredHit]) -> Vec<(u32, u32)> {
    hits.iter().map(|h| (h.doc_id, h.score.to_bits())).collect()
}

fn term(field: &str, t: &str, boost: f32) -> Query {
    let mut tq = TermQuery::new(field, t);
    tq.boost = boost;
    Query::Term(tq)
}

fn should_of(qs: Vec<Query>, boost: f32, msm: Option<u32>) -> Query {
    let mut b = BoolQuery::new().boost(boost);
    for q in qs {
        b = b.should(q);
    }
    if let Some(m) = msm {
        b = b.min_should_match(m);
    }
    Query::Bool(Box::new(b))
}

fn must_of(qs: Vec<Query>, boost: f32) -> Query {
    let mut b = BoolQuery::new().boost(boost);
    for q in qs {
        b = b.must(q);
    }
    Query::Bool(Box::new(b))
}

fn tokens(rng: &mut Rng, n: usize) -> Vec<String> {
    (0..n)
        .map(|_| match rng.below(12) {
            0 => "absentterm".to_string(),
            1..=4 => word(rng.below(VOCAB as u64) as usize), // uniform: rare
            _ => word(rng.zipf(VOCAB)),                      // Zipf: common
        })
        .collect()
}

/// The `xerj search` / MCP body as the engine projects it, over `fields`.
fn client_body(rng: &mut Rng, fields: &[&str]) -> Query {
    let n = 3 + rng.below(12) as usize;
    let toks = tokens(rng, n);
    // multi_match most_fields: one should-bool of the tokens per field
    let mm = should_of(
        fields
            .iter()
            .map(|f| should_of(toks.iter().map(|t| term(f, t, 1.0)).collect(), 1.0, None))
            .collect(),
        1.0,
        None,
    );
    // exact phrase on the content field (boost 4): a run that exists, or the whole token list
    let start = rng.zipf(VOCAB);
    let phrase_terms: Vec<String> = if rng.below(2) == 0 {
        (0..2 + rng.below(2) as usize)
            .map(|k| word((start + k) % VOCAB))
            .collect()
    } else {
        toks.clone()
    };
    let phrase = Query::Phrase(PhraseQuery {
        field: "body".into(),
        terms: phrase_terms,
        slop: 0,
        boost: 4.0,
    });
    // coverage: per-token multi_match under minimum_should_match = half, and the AND copy
    let need = (n as u32).div_ceil(2);
    let per_token: Vec<Query> = toks
        .iter()
        .map(|t| should_of(fields.iter().map(|f| term(f, t, 1.0)).collect(), 1.0, None))
        .collect();
    let cov1 = should_of(per_token, 1.0, Some(need));
    let cov2 = should_of(
        fields
            .iter()
            .map(|f| must_of(toks.iter().map(|t| term(f, t, 1.0)).collect(), 1.0))
            .collect(),
        1.0,
        None,
    );
    should_of(vec![mm, phrase, cov1, cov2], 1.0, None)
}

/// The console's body: one `match` per text field, minimum_should_match 1.
fn console_body(rng: &mut Rng) -> Query {
    let n = 2 + rng.below(10) as usize;
    let toks = tokens(rng, n);
    should_of(
        FIELDS
            .iter()
            .map(|f| should_of(toks.iter().map(|t| term(f, t, 1.0)).collect(), 1.0, None))
            .collect(),
        1.0,
        Some(1),
    )
}

/// A random nested bool of terms and phrases.
fn random_tree(rng: &mut Rng, depth: usize) -> Query {
    let leaf = |rng: &mut Rng| -> Query {
        let f = FIELDS[rng.below(3) as usize];
        let boost = match rng.below(9) {
            0 => 0.0,
            1 => 2.5,
            2 => 0.3,
            _ => 1.0,
        };
        if rng.below(10) == 0 {
            let s = rng.zipf(VOCAB);
            Query::Phrase(PhraseQuery {
                field: "body".into(),
                terms: (0..2).map(|k| word((s + k) % VOCAB)).collect(),
                slop: rng.below(2) as u32,
                boost: 1.0 + rng.below(3) as f32,
            })
        } else {
            term(f, &tokens(rng, 1)[0], boost)
        }
    };
    let mut b = BoolQuery::new();
    if rng.below(5) == 0 {
        b = b.boost(1.5);
    }
    let nm = if depth > 0 && rng.below(3) == 0 {
        1 + rng.below(3)
    } else {
        0
    };
    let ns = 1 + rng.below(6);
    for _ in 0..nm {
        let c = if depth > 0 && rng.below(3) == 0 {
            random_tree(rng, depth - 1)
        } else {
            leaf(rng)
        };
        b = b.must(c);
    }
    for _ in 0..ns {
        let c = if depth > 0 && rng.below(3) == 0 {
            random_tree(rng, depth - 1)
        } else {
            leaf(rng)
        };
        b = b.should(c);
    }
    match rng.below(5) {
        0 => b = b.min_should_match(rng.below(ns + 1) as u32),
        1 => b = b.min_should_match(0),
        _ => {}
    }
    Query::Bool(Box::new(b))
}

fn queries(seed: u64, n: usize) -> Vec<Query> {
    let mut rng = Rng(seed);
    (0..n)
        .map(|i| match i % 4 {
            0 => client_body(&mut rng, &["title", "body"]),
            1 => console_body(&mut rng),
            _ => random_tree(&mut rng, 2),
        })
        .collect()
}

fn check(generic: &FtsSearcher, tree: &FtsSearcher, queries: &[Query], label: &str) -> usize {
    let mut checks = 0;
    for (qi, q) in queries.iter().enumerate() {
        let (all, exact_total) = generic
            .search_bounded_observed(q, usize::MAX, false, |_| {})
            .unwrap();
        for cap in [1usize, 5, 10, 64] {
            let (want, _) = generic
                .search_bounded_observed(q, cap, false, |_| {})
                .unwrap();
            for limit in [0u64, 7, 1_000, u64::MAX] {
                let (got, total) = tree.search_bounded_counting(q, cap, false, limit).unwrap();
                assert_eq!(
                    page(&got),
                    page(&want),
                    "{label}: query {qi} cap {cap} limit {limit}: page\n{q:?}"
                );
                if exact_total < limit {
                    assert_eq!(
                        total, exact_total,
                        "{label}: query {qi} cap {cap} limit {limit}: total"
                    );
                } else {
                    assert!(
                        total >= limit && total <= exact_total,
                        "{label}: query {qi} cap {cap} limit {limit}: total {total} vs exact {exact_total}"
                    );
                }
                checks += 1;
            }
        }
        // the match set
        let mut want_bits: Vec<u64> = Vec::new();
        for h in &all {
            let w = (h.doc_id / 64) as usize;
            if want_bits.len() <= w {
                want_bits.resize(w + 1, 0);
            }
            want_bits[w] |= 1u64 << (h.doc_id % 64);
        }
        let mut got_bits = tree.matching_docs(q).unwrap().expect("no deadline");
        let trim = |v: &mut Vec<u64>| {
            while v.last() == Some(&0) {
                v.pop();
            }
        };
        trim(&mut want_bits);
        trim(&mut got_bits);
        assert_eq!(got_bits, want_bits, "{label}: query {qi}: matching_docs");
    }
    checks
}

#[test]
fn tree_walk_page_total_and_match_set_equal_the_generic_path() {
    let dir = TempDir::new().unwrap();
    build_segment(dir.path(), "seg-t", 5_000, 0x2545_f491_4f6c_dd1d);
    let qs = queries(0x9e37_79b9_7f4a_7c15, 48);
    let stats = Arc::new(WalkStats::default());
    let r = reader(dir.path(), "seg-t");
    assert!(
        r.block_index("body").is_some(),
        "the body field carries a block index"
    );
    let (generic, tree) = searchers(&r, &stats);
    let checks = check(&generic, &tree, &qs, "as written");
    assert!(checks > 0);
    // fail-before: the client shapes are served by the tree walk (the generic path publishes no
    // walk), and the walk scores fewer documents than it would visit without pruning
    let walks = stats.searches.load(Relaxed);
    assert!(
        walks >= (qs.len() * 16 / 2) as u64,
        "tree walk served {walks} searches"
    );
    let scored = stats.scored.load(Relaxed);
    let candidates = stats.candidates.load(Relaxed);
    assert!(
        scored > 0 && candidates > 0,
        "scored {scored} candidates {candidates}"
    );
    drop((generic, tree, r));
    // the same segment in the layout written before the block index existed
    assert!(strip_block_index(dir.path(), "seg-t") >= 2);
    let r = reader(dir.path(), "seg-t");
    assert!(r.block_index("body").is_none());
    let (generic, tree) = searchers(&r, &stats);
    check(&generic, &tree, &qs, "frame stripped");
}

#[test]
fn tree_walk_parity_on_short_and_single_block_lists() {
    for (docs, seed) in [
        (1usize, 7u64),
        (5, 11),
        (127, 13),
        (128, 17),
        (129, 19),
        (300, 23),
    ] {
        let dir = TempDir::new().unwrap();
        build_segment(dir.path(), "seg-s", docs, seed);
        let stats = Arc::new(WalkStats::default());
        let r = reader(dir.path(), "seg-s");
        let (generic, tree) = searchers(&r, &stats);
        check(
            &generic,
            &tree,
            &queries(seed ^ 0xdead_beef, 16),
            &format!("{docs} docs"),
        );
    }
}

/// The pruning is real on the client body: with a full page the walk evaluates far fewer documents
/// than the body matches (the generic path folds every match).
#[test]
fn tree_walk_prunes_the_client_body() {
    let dir = TempDir::new().unwrap();
    build_segment(dir.path(), "seg-p", 20_000, 0x1234_5678);
    let r = reader(dir.path(), "seg-p");
    let stats = Arc::new(WalkStats::default());
    let (generic, tree) = searchers(&r, &stats);
    let mut rng = Rng(0xabcdef);
    let mut matched = 0u64;
    let mut n = 0u64;
    for _ in 0..400 {
        if n >= 12 {
            break;
        }
        let q = client_body(&mut rng, &["title", "body"]);
        let (_, exact) = generic
            .search_bounded_observed(&q, 10, false, |_| {})
            .unwrap();
        if exact < 2_000 {
            continue;
        }
        let (want, _) = generic
            .search_bounded_observed(&q, 10, false, |_| {})
            .unwrap();
        let (got, _) = tree.search_bounded_counting(&q, 10, false, 10_001).unwrap();
        assert_eq!(page(&got), page(&want));
        matched += exact;
        n += 1;
    }
    assert_eq!(
        n, 12,
        "found 12 client bodies matching at least 2,000 documents"
    );
    assert_eq!(
        stats.searches.load(Relaxed),
        12,
        "the tree walk served every body"
    );
    assert!(
        stats.tree_pruned.load(Relaxed) > 0,
        "the constraint-aware bound dropped candidates"
    );
    let scored = stats.scored.load(Relaxed);
    assert!(
        scored * 4 < matched,
        "the walk scored {scored} documents of {matched} matches"
    );
}

/// The cross-segment floor (`search_bounded_counting_floor`): walking segments in order and handing
/// each the worst score of the merged page so far ends with the generic merged page.
#[test]
fn tree_walk_cross_segment_floor_keeps_the_merged_page() {
    let dir = TempDir::new().unwrap();
    let segments = ["seg-0", "seg-1", "seg-2"];
    for (i, seg) in segments.iter().enumerate() {
        build_segment(dir.path(), seg, 1_500 + 500 * i, 0x5151 + i as u64);
    }
    let stats = Arc::new(WalkStats::default());
    let readers: Vec<Arc<FtsIndexReader>> =
        segments.iter().map(|s| reader(dir.path(), s)).collect();
    let pairs: Vec<(FtsSearcher, FtsSearcher)> =
        readers.iter().map(|r| searchers(r, &stats)).collect();
    let merge = |pages: &[Vec<ScoredHit>], cap: usize| -> Vec<(usize, u32, u32)> {
        let mut all: Vec<(usize, u32, u32)> = pages
            .iter()
            .enumerate()
            .flat_map(|(s, p)| p.iter().map(move |h| (s, h.doc_id, h.score.to_bits())))
            .collect();
        all.sort_by(|a, b| {
            f32::from_bits(b.2)
                .total_cmp(&f32::from_bits(a.2))
                .then(a.0.cmp(&b.0))
                .then(a.1.cmp(&b.1))
        });
        all.truncate(cap);
        all
    };
    for (qi, q) in queries(0x7777, 24).iter().enumerate() {
        for cap in [1usize, 10, 64] {
            let generic: Vec<Vec<ScoredHit>> = pairs
                .iter()
                .map(|(g, _)| g.search_bounded_observed(q, cap, false, |_| {}).unwrap().0)
                .collect();
            let expected = merge(&generic, cap);
            for limit in [0u64, u64::MAX] {
                let mut pages: Vec<Vec<ScoredHit>> = Vec::new();
                for (_, t) in &pairs {
                    let merged = merge(&pages, cap);
                    let floor = (merged.len() == cap).then(|| f32::from_bits(merged[cap - 1].2));
                    let (hits, _) = t
                        .search_bounded_counting_floor(q, cap, false, limit, floor)
                        .unwrap();
                    if let Some(f) = floor {
                        assert!(
                            hits.iter().all(|h| h.score >= f),
                            "query {qi}: a hit below the floor"
                        );
                    }
                    pages.push(hits);
                }
                assert_eq!(
                    merge(&pages, cap),
                    expected,
                    "query {qi} cap {cap} limit {limit}: merged page"
                );
            }
        }
    }
}

/// Shapes the tree walk does not take keep the generic path (and the same answer).
#[test]
fn unsupported_shapes_keep_the_generic_path() {
    let dir = TempDir::new().unwrap();
    build_segment(dir.path(), "seg-u", 800, 99);
    let stats = Arc::new(WalkStats::default());
    let r = reader(dir.path(), "seg-u");
    let (generic, tree) = searchers(&r, &stats);
    let with_must_not = Query::Bool(Box::new(
        BoolQuery::new()
            .should(term("body", "w1", 1.0))
            .should(should_of(vec![term("title", "w2", 1.0)], 1.0, None))
            .must_not(term("body", "w3", 1.0)),
    ));
    let negative_boost = should_of(
        vec![
            should_of(vec![term("body", "w1", -1.0)], 1.0, None),
            term("title", "w2", 1.0),
        ],
        1.0,
        None,
    );
    for q in [with_must_not, negative_boost] {
        let (want, wt) = generic
            .search_bounded_observed(&q, 10, false, |_| {})
            .unwrap();
        let (got, gt) = tree
            .search_bounded_counting(&q, 10, false, u64::MAX)
            .unwrap();
        assert_eq!(page(&got), page(&want));
        assert_eq!(gt, wt);
    }
    assert_eq!(stats.searches.load(Relaxed), 0, "no tree walk ran");
}

/// More seeds on demand: `XERJ_TREE_PARITY_SEEDS=<n>` runs the parity check over `n` more segments
/// and query sets (0 = skipped, the default).
#[test]
fn tree_walk_parity_more_seeds() {
    let n: u64 = std::env::var("XERJ_TREE_PARITY_SEEDS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    for k in 0..n {
        let dir = TempDir::new().unwrap();
        let docs = 200 + (k as usize * 977) % 6_000;
        build_segment(dir.path(), "seg-m", docs, 0x51ed_270b + k);
        let stats = Arc::new(WalkStats::default());
        let r = reader(dir.path(), "seg-m");
        let (generic, tree) = searchers(&r, &stats);
        check(
            &generic,
            &tree,
            &queries(0xc0ff_ee00 + k, 40),
            &format!("seed {k} ({docs} docs)"),
        );
    }
}

/// The pages of several segments merged as the engine merges them: score desc, then segment, then
/// doc id; `(segment, doc id, score bits)`, the first `cap`.
fn merge_pages(pages: &[Vec<ScoredHit>], cap: usize) -> Vec<(usize, u32, u32)> {
    let mut all: Vec<(usize, u32, u32)> = pages
        .iter()
        .enumerate()
        .flat_map(|(s, p)| p.iter().map(move |h| (s, h.doc_id, h.score.to_bits())))
        .collect();
    all.sort_by(|a, b| {
        f32::from_bits(b.2)
            .total_cmp(&f32::from_bits(a.2))
            .then(a.0.cmp(&b.0))
            .then(a.1.cmp(&b.1))
    });
    all.truncate(cap);
    all
}

/// Flat `match`-shaped queries (`bool{should: [term..]}` on `body`), the flat walk's shape.
fn flat_queries(seed: u64, n: usize) -> Vec<Query> {
    let mut rng = Rng(seed);
    (0..n)
        .map(|_| {
            let k = 2 + rng.below(6) as usize;
            let ts = tokens(&mut rng, k);
            should_of(ts.iter().map(|t| term("body", t, 1.0)).collect(), 1.0, None)
        })
        .collect()
}

/// The shared floor (`FtsSearcher::with_shared_floor`; the engine's opt-in `XERJ_FTS_LIVE_FLOOR`):
/// the walks of one search's segments read one floor while they run, and each raises it to its full
/// page's worst score.
/// * A shared floor holding `f` (at most the page's worst score) acts as the caller's floor `f`: the
///   same hits bit for bit and the same total, in the flat walk and the tree walk; and a walk whose
///   page fills leaves that page's worst score in it.
/// * Raised while the segments' walks run concurrently (by the walks themselves and by a racing
///   thread, to any value up to the merged page's worst score), it never costs the merged page a
///   document: the segments' hits merge into the generic merged page, totals unchanged.
/// * It prunes: with the floor at the page's worst score from the start, the walks score fewer
///   documents than without it. FAIL-BEFORE: a walk that ignores the shared floor scores as many,
///   and leaves the floor unset (the publication assertions fail).
#[test]
fn shared_floor_keeps_the_merged_page_and_prunes() {
    let reg = Arc::new(AnalyzerRegistry::default());
    // ── one segment: a preset shared floor is the caller's floor; a full page publishes ──
    let dir = TempDir::new().unwrap();
    build_segment(dir.path(), "seg-f", 4_000, 0x0f10_0f10);
    let r = reader(dir.path(), "seg-f");
    let generic = FtsSearcher::new(Arc::clone(&r), Arc::clone(&reg)).with_tree_walk(false);
    let tree = FtsSearcher::new(Arc::clone(&r), Arc::clone(&reg)).with_tree_walk(true);
    let mut qs = queries(0x5eed_f100, 24);
    qs.extend(flat_queries(0xf1a7, 12));
    let (mut checked, mut published) = (0usize, 0usize);
    for (qi, q) in qs.iter().enumerate() {
        let (all, _) = generic
            .search_bounded_observed(q, usize::MAX, false, |_| {})
            .unwrap();
        for cap in [1usize, 10, 64] {
            if all.len() < cap || all[cap - 1].score <= 0.0 {
                continue;
            }
            let worst = all[cap - 1].score;
            let mut floors = vec![
                worst,
                all[(cap - 1 + all.len() - 1) / 2].score,
                all[all.len() - 1].score,
            ];
            floors.retain(|f| *f > 0.0 && *f <= worst);
            for f in floors {
                for limit in [7u64, u64::MAX] {
                    let (want, want_total) = tree
                        .search_bounded_counting_floor(q, cap, false, limit, Some(f))
                        .unwrap();
                    let shared = Arc::new(SharedFloor::new());
                    shared.raise(f);
                    let s = FtsSearcher::new(Arc::clone(&r), Arc::clone(&reg))
                        .with_tree_walk(true)
                        .with_shared_floor(Some(Arc::clone(&shared)));
                    let (got, got_total) = s
                        .search_bounded_counting_floor(q, cap, false, limit, None)
                        .unwrap();
                    assert_eq!(
                        page(&got),
                        page(&want),
                        "query {qi} cap {cap} floor {f}: page"
                    );
                    assert_eq!(
                        got_total, want_total,
                        "query {qi} cap {cap} floor {f}: total"
                    );
                    let now = shared.get().expect("a raised floor stays set");
                    assert!(
                        now >= f && now <= worst,
                        "query {qi} cap {cap}: floor {now} outside [{f}, {worst}]"
                    );
                    checked += 1;
                }
            }
            // a fresh shared floor: the walk's full page publishes its worst score
            let stats = Arc::new(WalkStats::default());
            let shared = Arc::new(SharedFloor::new());
            let s = FtsSearcher::new(Arc::clone(&r), Arc::clone(&reg))
                .with_tree_walk(true)
                .with_walk_stats(Some(Arc::clone(&stats)))
                .with_shared_floor(Some(Arc::clone(&shared)));
            let (got, _) = s.search_bounded_counting(q, cap, false, u64::MAX).unwrap();
            let (want, _) = generic
                .search_bounded_observed(q, cap, false, |_| {})
                .unwrap();
            assert_eq!(page(&got), page(&want), "query {qi} cap {cap}: page");
            if stats.searches.load(Relaxed) == 1 {
                // served by the flat or the tree walk (the generic path has no shared floor)
                assert_eq!(
                    shared.get(),
                    Some(worst),
                    "query {qi} cap {cap}: published floor"
                );
                published += 1;
            }
        }
    }
    assert!(checked >= 200, "preset-floor checks: {checked}");
    assert!(
        published >= 60,
        "walks that published their page's floor: {published}"
    );

    // ── concurrent segments, the floor rising while they walk ──
    let segments = ["seg-c0", "seg-c1", "seg-c2", "seg-c3"];
    for (i, seg) in segments.iter().enumerate() {
        build_segment(dir.path(), seg, 1_000 + 700 * i, 0xc0c0 + i as u64);
    }
    let readers: Vec<Arc<FtsIndexReader>> =
        segments.iter().map(|s| reader(dir.path(), s)).collect();
    let mut cq = queries(0x7a11_f100, 16);
    cq.extend(flat_queries(0x7a12, 8));
    let mut merged_checks = 0usize;
    for (qi, q) in cq.iter().enumerate() {
        for cap in [1usize, 10, 64] {
            let generic_runs: Vec<(Vec<ScoredHit>, u64)> = readers
                .iter()
                .map(|r| {
                    let g = FtsSearcher::new(Arc::clone(r), Arc::clone(&reg)).with_tree_walk(false);
                    let (page, _) = g.search_bounded_observed(q, cap, false, |_| {}).unwrap();
                    let (_, exact) = g
                        .search_bounded_observed(q, usize::MAX, false, |_| {})
                        .unwrap();
                    (page, exact)
                })
                .collect();
            let generic_pages: Vec<Vec<ScoredHit>> =
                generic_runs.iter().map(|(p, _)| p.clone()).collect();
            let expected = merge_pages(&generic_pages, cap);
            let merged_worst = (expected.len() == cap).then(|| f32::from_bits(expected[cap - 1].2));
            for round in 0..4u64 {
                let shared = Arc::new(SharedFloor::new());
                let done = AtomicBool::new(false);
                let runs: Vec<(Vec<ScoredHit>, u64)> = std::thread::scope(|scope| {
                    if let Some(w) = merged_worst.filter(|w| *w > 0.0) {
                        let shared = Arc::clone(&shared);
                        let done = &done;
                        scope.spawn(move || {
                            let mut rng = Rng(0x9ace_0001 ^ ((qi as u64) << 8) ^ round);
                            while !done.load(Relaxed) {
                                // any value up to the merged page's worst score
                                let u = (rng.below(1_000) + 1) as f32 / 1_000.0;
                                shared.raise((w * u).min(w));
                                std::thread::yield_now();
                            }
                        });
                    }
                    let walkers: Vec<_> = readers
                        .iter()
                        .map(|r| {
                            let s = FtsSearcher::new(Arc::clone(r), Arc::clone(&reg))
                                .with_tree_walk(true)
                                .with_shared_floor(Some(Arc::clone(&shared)));
                            scope.spawn(move || {
                                s.search_bounded_counting(q, cap, false, u64::MAX).unwrap()
                            })
                        })
                        .collect();
                    let runs = walkers.into_iter().map(|h| h.join().unwrap()).collect();
                    done.store(true, Relaxed);
                    runs
                });
                let pages: Vec<Vec<ScoredHit>> = runs.iter().map(|(p, _)| p.clone()).collect();
                assert_eq!(
                    merge_pages(&pages, cap),
                    expected,
                    "query {qi} cap {cap} round {round}: merged page"
                );
                for (si, ((_, total), (_, exact))) in runs.iter().zip(&generic_runs).enumerate() {
                    assert_eq!(total, exact, "query {qi} cap {cap} segment {si}: total");
                }
                if let (Some(w), Some(now)) = (merged_worst, shared.get()) {
                    assert!(
                        now <= w,
                        "query {qi} cap {cap}: floor {now} above the page's {w}"
                    );
                }
                merged_checks += 1;
            }
        }
    }
    assert!(
        merged_checks >= 250,
        "concurrent merges checked: {merged_checks}"
    );

    // ── it prunes: the floor at the page's worst score from the start ──
    let dir_p = TempDir::new().unwrap();
    build_segment(dir_p.path(), "seg-p2", 20_000, 0x1234_5678);
    let rp = reader(dir_p.path(), "seg-p2");
    let gp = FtsSearcher::new(Arc::clone(&rp), Arc::clone(&reg)).with_tree_walk(false);
    let mut rng = Rng(0xabcdef);
    let (without, with) = (
        Arc::new(WalkStats::default()),
        Arc::new(WalkStats::default()),
    );
    let mut n = 0;
    for _ in 0..400 {
        if n >= 12 {
            break;
        }
        let q = client_body(&mut rng, &["title", "body"]);
        let (_, exact) = gp.search_bounded_observed(&q, 10, false, |_| {}).unwrap();
        if exact < 2_000 {
            continue;
        }
        let (want, _) = gp.search_bounded_observed(&q, 10, false, |_| {}).unwrap();
        let plain = FtsSearcher::new(Arc::clone(&rp), Arc::clone(&reg))
            .with_tree_walk(true)
            .with_walk_stats(Some(Arc::clone(&without)));
        let (a, _) = plain
            .search_bounded_counting(&q, 10, false, 10_001)
            .unwrap();
        let shared = Arc::new(SharedFloor::new());
        shared.raise(want[want.len() - 1].score);
        let floored = FtsSearcher::new(Arc::clone(&rp), Arc::clone(&reg))
            .with_tree_walk(true)
            .with_walk_stats(Some(Arc::clone(&with)))
            .with_shared_floor(Some(shared));
        let (b, _) = floored
            .search_bounded_counting(&q, 10, false, 10_001)
            .unwrap();
        assert_eq!(page(&a), page(&want));
        assert_eq!(page(&b), page(&want));
        n += 1;
    }
    assert_eq!(
        n, 12,
        "found 12 client bodies matching at least 2,000 documents"
    );
    let (sa, sb) = (without.scored.load(Relaxed), with.scored.load(Relaxed));
    eprintln!(
        "shared floor: {checked} preset-floor checks, {published} walks published, \
         {merged_checks} concurrent merges; scored {sb} documents with the floor, {sa} without"
    );
    assert!(
        sb < sa,
        "with the shared floor the walks scored {sb} documents, without it {sa}"
    );
}
