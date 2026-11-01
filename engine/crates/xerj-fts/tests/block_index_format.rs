//! The block index lives in a zstd skippable frame inside the ZPS2 `.post` envelope. These tests pin
//! the compatibility story: readers that predate the frame (they hand the bytes after the 8-byte
//! header to `zstd::bulk::decompress`) still get exactly the postings; the new reader finds the frame
//! on flushed and on merged segments; a segment without it (stripped) still answers, unchanged.
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use tempfile::TempDir;
use xerj_fts::{
    AnalyzerRegistry, BoolQuery, FieldIndexConfig, FieldValues, FtsIndexReader, FtsIndexWriter,
    FtsMergeSource, FtsSearcher, Query, ScoredHit, TermQuery,
};

const SKIPPABLE: u32 = 0x184D_2A5B;

fn registry() -> Arc<AnalyzerRegistry> {
    Arc::new(AnalyzerRegistry::default())
}

/// `docs` documents over a small vocabulary: "common" in every document, "mid" in every third,
/// "rare" in every 97th, plus filler; lengths vary.
fn write_segment(dir: &Path, id: &str, docs: u32, offset: u32) {
    let mut w = FtsIndexWriter::new(dir, id, registry());
    w.configure_field("body", FieldIndexConfig::default());
    for i in 0..docs {
        let g = i + offset;
        let mut text = format!("common filler{} pad{}", g % 13, g % 5);
        if g.is_multiple_of(3) {
            text.push_str(" mid mid");
        }
        if g.is_multiple_of(97) {
            text.push_str(" rare");
        }
        for _ in 0..(g % 7) {
            text.push_str(" more words here");
        }
        let fields: HashMap<String, FieldValues> =
            [("body".to_owned(), FieldValues::from(text))].into();
        w.add_document(i, &fields);
    }
    w.finish().unwrap();
}

fn post_path(dir: &Path, id: &str) -> std::path::PathBuf {
    dir.join(format!("{id}.body.post"))
}

fn query() -> Query {
    Query::Bool(Box::new(
        BoolQuery::new()
            .should(Query::Term(TermQuery::new("body", "rare")))
            .should(Query::Term(TermQuery::new("body", "mid")))
            .should(Query::Term(TermQuery::new("body", "common")))
            .should(Query::Term(TermQuery::new("body", "filler3"))),
    ))
}

fn page(hits: &[ScoredHit]) -> Vec<(u32, u32)> {
    hits.iter().map(|h| (h.doc_id, h.score.to_bits())).collect()
}

#[test]
fn readers_that_predate_the_frame_decompress_the_same_postings() {
    let dir = TempDir::new().unwrap();
    write_segment(dir.path(), "seg-a", 3_000, 0);
    let bytes = std::fs::read(post_path(dir.path(), "seg-a")).unwrap();
    assert_eq!(&bytes[..4], b"ZPS2");
    let uncompressed_len = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
    let magic = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
    assert_eq!(magic, SKIPPABLE, "the block-index frame follows the header");
    let frame_len = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
    // what every older reader does: decompress everything after the header
    let old_view = zstd::bulk::decompress(&bytes[8..], uncompressed_len).unwrap();
    // the postings frame alone
    let postings_only = zstd::bulk::decompress(&bytes[16 + frame_len..], uncompressed_len).unwrap();
    assert_eq!(old_view.len(), uncompressed_len);
    assert_eq!(old_view, postings_only, "zstd must skip the frame");

    let reader = FtsIndexReader::open(dir.path(), "seg-a", &["body"]).unwrap();
    let ix = reader.block_index("body").expect("block index parsed");
    assert!(
        ix.terms() >= 2 && ix.blocks() >= 3_000 / 128,
        "{} terms, {} blocks",
        ix.terms(),
        ix.blocks()
    );
}

#[test]
fn stripping_the_frame_leaves_a_pre_frame_segment_with_the_same_answers() {
    let dir = TempDir::new().unwrap();
    write_segment(dir.path(), "seg-a", 2_000, 0);
    let with = FtsSearcher::new(
        Arc::new(FtsIndexReader::open(dir.path(), "seg-a", &["body"]).unwrap()),
        registry(),
    );
    let mut before = Vec::new();
    for cap in [1usize, 5, 40] {
        for limit in [0u64, 10, u64::MAX] {
            before.push(
                with.search_bounded_counting(&query(), cap, false, limit)
                    .unwrap(),
            );
        }
    }
    drop(with);
    let path = post_path(dir.path(), "seg-a");
    let bytes = std::fs::read(&path).unwrap();
    let frame_len = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
    let mut stripped = bytes[..8].to_vec();
    stripped.extend_from_slice(&bytes[16 + frame_len..]);
    std::fs::write(&path, stripped).unwrap();
    let reader = Arc::new(FtsIndexReader::open(dir.path(), "seg-a", &["body"]).unwrap());
    assert!(reader.block_index("body").is_none());
    let without = FtsSearcher::new(reader, registry());
    let mut i = 0;
    for cap in [1usize, 5, 40] {
        for limit in [0u64, 10, u64::MAX] {
            let (hits, total) = without
                .search_bounded_counting(&query(), cap, false, limit)
                .unwrap();
            assert_eq!(page(&hits), page(&before[i].0), "cap {cap} limit {limit}");
            assert_eq!(total, before[i].1, "cap {cap} limit {limit}");
            i += 1;
        }
    }
}

#[test]
fn merged_segments_carry_the_block_index_and_answer_like_the_generic_path() {
    let dir = TempDir::new().unwrap();
    write_segment(dir.path(), "seg-a", 1_500, 0);
    write_segment(dir.path(), "seg-b", 1_700, 1_500);
    let a = FtsIndexReader::open(dir.path(), "seg-a", &["body"]).unwrap();
    let b = FtsIndexReader::open(dir.path(), "seg-b", &["body"]).unwrap();
    let map_a: Vec<Option<u32>> = (0..1_500).map(Some).collect();
    let map_b: Vec<Option<u32>> = (0..1_700).map(|i| Some(1_500 + i)).collect();
    let mut merged = FtsIndexWriter::new(dir.path(), "seg-m", registry());
    merged.configure_field("body", FieldIndexConfig::default());
    merged
        .merge_from_segments(&[
            FtsMergeSource {
                reader: &a,
                doc_map: &map_a,
            },
            FtsMergeSource {
                reader: &b,
                doc_map: &map_b,
            },
        ])
        .unwrap();
    merged.finish().unwrap();
    let reader = Arc::new(FtsIndexReader::open(dir.path(), "seg-m", &["body"]).unwrap());
    let ix = reader
        .block_index("body")
        .expect("merged segment has a block index");
    assert!(ix.blocks() >= 3_200 / 128);
    let s = FtsSearcher::new(reader, registry());
    for cap in [1usize, 7, 64] {
        let (generic, exact) = s
            .search_bounded_observed(&query(), cap, false, |_| {})
            .unwrap();
        for limit in [0u64, 5, 500, u64::MAX] {
            let (hits, total) = s
                .search_bounded_counting(&query(), cap, false, limit)
                .unwrap();
            assert_eq!(page(&hits), page(&generic), "cap {cap} limit {limit}");
            if exact < limit {
                assert_eq!(total, exact);
            } else {
                assert!(
                    total >= limit && total <= exact,
                    "{total} vs {exact} at {limit}"
                );
            }
        }
    }
}

/// The wide `.post` header (`"ZPS2" | u32::MAX | "ZPW8" | u64 len`, written for postings of
/// 4 GiB - 1 and more) puts the frames 12 bytes later. Re-wrapping a small segment's postings in
/// that form must leave the reader's answers and its block index unchanged, and a reader older
/// than the form (it hands the bytes after the 8th to zstd with the `u32` as the capacity) must get
/// a frame error, not postings.
#[test]
fn the_wide_header_form_reads_the_same_postings_and_older_readers_refuse_it() {
    let dir = TempDir::new().unwrap();
    write_segment(dir.path(), "seg-a", 2_000, 0);
    let answers = |dir: &Path| {
        let reader = Arc::new(FtsIndexReader::open(dir, "seg-a", &["body"]).unwrap());
        assert!(reader.block_index("body").is_some(), "block index found");
        let searcher = FtsSearcher::new(reader, registry());
        let mut out = Vec::new();
        for cap in [1usize, 5, 40] {
            for limit in [0u64, 10, u64::MAX] {
                let (hits, total) = searcher
                    .search_bounded_counting(&query(), cap, false, limit)
                    .unwrap();
                out.push((page(&hits), total));
            }
        }
        out
    };
    let before = answers(dir.path());

    let path = post_path(dir.path(), "seg-a");
    let bytes = std::fs::read(&path).unwrap();
    assert_eq!(&bytes[..4], b"ZPS2");
    let len = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
    let mut wide = b"ZPS2".to_vec();
    wide.extend_from_slice(&u32::MAX.to_le_bytes());
    wide.extend_from_slice(b"ZPW8");
    wide.extend_from_slice(&u64::from(len).to_le_bytes());
    wide.extend_from_slice(&bytes[8..]);
    std::fs::write(&path, &wide).unwrap();

    assert_eq!(answers(dir.path()), before, "same hits, scores and totals");
    assert!(
        zstd::bulk::decompress(&wide[8..], u32::MAX as usize).is_err(),
        "an older reader must fail on the wide form, not read it as postings"
    );
}
