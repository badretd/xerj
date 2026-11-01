//! A segment records the decoded size of its stored section.
//!
//! What a merge holds in memory is a multiple of its inputs' DECODED stored
//! bytes (the survivors' JSON, the merged buffer, the parsed tree), but the
//! only size a segment carried was its size on disk, and the ratio between
//! the two depends on the data: about 3x for prose, 5-6x for the packed
//! lexical vectors of `semantic_text`, and about 40x for the documents below.
//! The decoded size cannot be read back cheaply either: a stored column
//! written with one-shot zstd has no content size in its frame header. The
//! writer of a segment knows it, so the segment now records it
//! (`SegmentMeta::stored_decoded_bytes`, surfaced as
//! `SegmentInfo::stored_decoded_bytes`): a flush from the documents it
//! charged to the memtable, a merge from the merged JSON it holds.

use serde_json::{json, Value};
use tempfile::TempDir;
use xerj_common::config::Config;
use xerj_common::types::{FieldConfig, FieldType, Schema};
use xerj_engine::{Engine, Index};
use xerj_query::parse_request;

const BATCHES: usize = 4;
const DOCS_PER_BATCH: usize = 400;
/// 16 KiB of one repeating sentence per document plus a short unique tail.
const BLOB_BYTES: usize = 16 * 1024;

fn make_engine(dir: &TempDir) -> Engine {
    let mut config = Config::default();
    config.server.data_dir = dir.path().to_str().unwrap().to_string();
    // One FTS memtable shard: one flush writes one segment.
    config.engine.ingest_shards = 1;
    Engine::new(config).expect("engine::new")
}

fn schema() -> Schema {
    let mut schema = Schema::empty();
    schema
        .fields
        .push(FieldConfig::new("title", FieldType::Keyword));
    schema
}

fn document(n: usize) -> Value {
    let sentence = "the quick brown fox jumps over the lazy dog while the band plays on ";
    let mut blob = sentence.repeat(BLOB_BYTES / sentence.len() + 1);
    blob.truncate(BLOB_BYTES);
    json!({ "title": format!("doc-{n}"), "blob": format!("{blob} {n}") })
}

async fn total_hits(idx: &Index) -> u64 {
    let request = parse_request(&json!({
        "query": {"match_all": {}}, "size": 0, "track_total_hits": true
    }))
    .unwrap();
    idx.search(&request).await.unwrap().total.value
}

#[tokio::test(flavor = "multi_thread")]
async fn flushed_and_merged_segments_record_their_decoded_stored_size() {
    // No background merge during the test: the one merge below runs by hand.
    std::env::set_var("XERJ_MERGE_INTERVAL_SECS", "3600");
    let dir = TempDir::new().unwrap();
    let engine = make_engine(&dir);
    engine.create_index("blobs", schema()).unwrap();
    let idx = engine.get_index("blobs").unwrap();
    for batch in 0..BATCHES {
        for n in batch * DOCS_PER_BATCH..(batch + 1) * DOCS_PER_BATCH {
            idx.index_document(Some(n.to_string()), document(n))
                .await
                .unwrap();
        }
        idx.flush().await.unwrap();
    }
    let all = (BATCHES * DOCS_PER_BATCH) as u64;
    assert_eq!(total_hits(&idx).await, all);

    // Every flushed segment records the size of what it decodes to: ~16 KiB
    // per document plus the per-document JSON around it, many times its size
    // on disk.
    let segments = idx.segment_rows();
    assert_eq!(segments.len(), BATCHES, "{segments:?}");
    let expected = (DOCS_PER_BATCH * BLOB_BYTES) as u64;
    for meta in &segments {
        assert!(
            meta.stored_decoded_bytes >= expected
                && meta.stored_decoded_bytes <= expected + expected / 4,
            "flushed segment records {} decoded bytes for {} documents of {} bytes",
            meta.stored_decoded_bytes,
            DOCS_PER_BATCH,
            BLOB_BYTES
        );
        assert!(
            meta.size_bytes * 10 < meta.stored_decoded_bytes,
            "the test needs segments that compress well: {} on disk, {} decoded",
            meta.size_bytes,
            meta.stored_decoded_bytes
        );
    }

    // The merged segment records its own decoded size: the sum of its inputs'.
    // The budget has room for the merge whatever the host's memory.
    std::env::set_var("XERJ_MAX_PROCESS_MEMORY_MB", "4096");
    assert_eq!(idx.run_merge_once().await.unwrap(), 1);
    let after = idx.segment_rows();
    assert_eq!(after.len(), 1, "{after:?}");
    let merged = after[0].stored_decoded_bytes;
    let inputs: u64 = segments.iter().map(|meta| meta.stored_decoded_bytes).sum();
    assert!(
        merged + merged / 10 >= inputs && merged <= inputs + inputs / 10,
        "merged segment records {merged} decoded bytes, its inputs {inputs}"
    );
    assert_eq!(total_hits(&idx).await, all);
    std::env::remove_var("XERJ_MAX_PROCESS_MEMORY_MB");
}
