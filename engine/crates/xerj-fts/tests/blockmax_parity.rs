//! Parity of the `match`-shaped disjunction walk (`search_bounded_counting`, the WAND / MaxScore
//! path) with the generic path (`search_bounded_observed`, every clause executed and folded).
//!
//! Contract under test (`FtsSearcher::search_bounded_counting`):
//! * the page (top `cap`) is the generic path's page bit for bit: same doc ids, same f32 score bits,
//!   same order (`ScoredHit`: score desc, then doc id asc);
//! * the total is exact when it is below `count_limit`, otherwise at least `count_limit` and never
//!   above the exact total.
//!
//! The corpus is large enough for many 128-posting blocks per common term (stopword-like terms
//! occur in most documents), lengths and term frequencies vary, clauses repeat and carry boosts,
//! bool boosts vary, and some clauses name terms or fields that do not exist. Every check runs on
//! the segment as written and on the same segment with the `.post` block-index frame stripped (the
//! layout of segments written before the frame existed), so the fallback walk is held to the same
//! contract.
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use tempfile::TempDir;
use xerj_fts::{
    AnalyzerRegistry, BoolQuery, FieldIndexConfig, FieldValues, FtsIndexReader, FtsIndexWriter,
    FtsSearcher, Query, ScoredHit, TermQuery,
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

const VOCAB: usize = 400;

fn word(i: usize) -> String {
    format!("w{i}")
}

/// `docs` documents with fields `body` (positions), `title` (positions) and `tags` (docs-only)
fn build_segment(dir: &Path, segment: &str, docs: usize, seed: u64) {
    let registry = Arc::new(AnalyzerRegistry::default());
    let mut writer = FtsIndexWriter::new(dir, segment, Arc::clone(&registry));
    writer.configure_field("body", FieldIndexConfig::default());
    writer.configure_field("title", FieldIndexConfig::default());
    writer.configure_field(
        "tags",
        FieldIndexConfig {
            store_positions: false,
            ..FieldIndexConfig::default()
        },
    );
    let mut rng = Rng(seed);
    for d in 0..docs {
        // lengths 1..=120 with a long tail of short documents; repeated words give tf > 1
        let len = if rng.below(10) == 0 {
            1 + rng.below(4) as usize
        } else {
            5 + rng.below(116) as usize
        };
        let mut body: Vec<String> = Vec::with_capacity(len);
        for _ in 0..len {
            let w = rng.zipf(VOCAB);
            body.push(word(w));
            if rng.below(12) == 0 {
                // bursts: the same word several times in one document
                for _ in 0..1 + rng.below(6) {
                    body.push(word(w));
                }
            }
        }
        let title: Vec<String> = (0..1 + rng.below(5))
            .map(|_| word(rng.zipf(VOCAB / 2)))
            .collect();
        let tags: Vec<String> = (0..rng.below(4))
            .map(|_| format!("t{}", rng.zipf(30)))
            .collect();
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

/// Remove a zstd skippable frame that directly follows the 8-byte ZPS2 header of every `.post` file
/// of `segment` (the block index), leaving the envelope exactly as pre-block-index writers made it.
/// Returns how many files had one.
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

fn searcher(dir: &Path, segment: &str) -> FtsSearcher {
    let reader = Arc::new(FtsIndexReader::open(dir, segment, &["body", "title", "tags"]).unwrap());
    FtsSearcher::new(reader, Arc::new(AnalyzerRegistry::default()))
}

fn page(hits: &[ScoredHit]) -> Vec<(u32, u32)> {
    hits.iter().map(|h| (h.doc_id, h.score.to_bits())).collect()
}

fn queries(seed: u64, n: usize) -> Vec<Query> {
    let mut rng = Rng(seed);
    let mut out = Vec::with_capacity(n);
    for q in 0..n {
        let clauses = 2 + rng.below(14) as usize;
        let mut bq = BoolQuery::new();
        let mut last: Option<String> = None;
        for c in 0..clauses {
            let (field, term) = match rng.below(20) {
                0 => ("title", word(rng.zipf(VOCAB / 2))),
                1 => ("tags", format!("t{}", rng.zipf(30))),
                2 => ("body", "absentterm".to_string()),
                3 if last.is_some() => ("body", last.clone().unwrap()), // repeated clause
                4..=7 => ("body", word(rng.below(VOCAB as u64) as usize)), // uniform: rare terms
                _ => ("body", word(rng.zipf(VOCAB))),                   // Zipf: common terms
            };
            let mut tq = TermQuery::new(field, term.clone());
            if (q + c) % 9 == 0 {
                tq.boost = 2.5;
            }
            bq = bq.should(Query::Term(tq));
            last = Some(term);
        }
        if q % 6 == 0 {
            bq = bq.boost(1.75);
        }
        out.push(Query::Bool(Box::new(bq)));
    }
    out
}

fn check(searcher: &FtsSearcher, queries: &[Query], label: &str) -> usize {
    let mut checks = 0;
    for (qi, q) in queries.iter().enumerate() {
        for cap in [1usize, 10, 64] {
            let (generic, exact_total) = searcher
                .search_bounded_observed(q, cap, false, |_| {})
                .unwrap();
            let (wand, wand_total) = searcher.search_bounded(q, cap, false).unwrap();
            assert_eq!(
                page(&wand),
                page(&generic),
                "{label}: query {qi} cap {cap}: exact walk page"
            );
            assert_eq!(
                wand_total, exact_total,
                "{label}: query {qi} cap {cap}: exact walk total"
            );
            for limit in [0u64, 7, 1_000, u64::MAX] {
                let (got, total) = searcher
                    .search_bounded_counting(q, cap, false, limit)
                    .unwrap();
                assert_eq!(
                    page(&got),
                    page(&generic),
                    "{label}: query {qi} cap {cap} limit {limit}: page"
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
    }
    checks
}

#[test]
fn match_walk_page_and_total_equal_the_generic_path() {
    let dir = TempDir::new().unwrap();
    build_segment(dir.path(), "seg-a", 6_000, 0x2545_f491_4f6c_dd1d);
    let qs = queries(0x9e37_79b9_7f4a_7c15, 40);
    let as_written = FtsIndexReader::open(dir.path(), "seg-a", &["body", "title", "tags"]).unwrap();
    assert!(
        as_written.block_index("body").is_some() && as_written.block_index("tags").is_some(),
        "fields with 128+ posting lists carry a block index"
    );
    drop(as_written);
    let checks = check(&searcher(dir.path(), "seg-a"), &qs, "as written");
    assert!(checks > 0);
    // the same segment in the layout written before the block index existed
    assert!(strip_block_index(dir.path(), "seg-a") >= 2);
    let stripped = FtsIndexReader::open(dir.path(), "seg-a", &["body", "title", "tags"]).unwrap();
    assert!(stripped.block_index("body").is_none());
    drop(stripped);
    check(&searcher(dir.path(), "seg-a"), &qs, "frame stripped");
}

#[test]
fn match_walk_parity_on_short_and_single_block_lists() {
    // few documents: every list is a residual or one or two blocks
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
        let qs = queries(seed ^ 0xdead_beef, 12);
        check(&searcher(dir.path(), "seg-s"), &qs, &format!("{docs} docs"));
    }
}

/// The cross-segment floor (`search_bounded_counting_floor`): a caller that walks segments in order
/// and hands each one the worst score of its merged page so far must end with the same merged page
/// as merging every segment's generic page. Merge order: score desc, then segment, then doc id.
#[test]
fn cross_segment_floor_keeps_the_merged_page() {
    let dir = TempDir::new().unwrap();
    let segments = ["seg-0", "seg-1", "seg-2", "seg-3"];
    for (i, seg) in segments.iter().enumerate() {
        build_segment(dir.path(), seg, 1_500 + 400 * i, 0x5151 + i as u64);
    }
    let searchers: Vec<FtsSearcher> = segments.iter().map(|s| searcher(dir.path(), s)).collect();
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
    for (qi, q) in queries(0x7777, 30).iter().enumerate() {
        for cap in [1usize, 10, 64] {
            let generic: Vec<Vec<ScoredHit>> = searchers
                .iter()
                .map(|s| s.search_bounded_observed(q, cap, false, |_| {}).unwrap().0)
                .collect();
            let expected = merge(&generic, cap);
            for limit in [0u64, u64::MAX] {
                let mut pages: Vec<Vec<ScoredHit>> = Vec::new();
                for s in &searchers {
                    let merged = merge(&pages, cap);
                    let floor = (merged.len() == cap).then(|| f32::from_bits(merged[cap - 1].2));
                    let (hits, _) = s
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
