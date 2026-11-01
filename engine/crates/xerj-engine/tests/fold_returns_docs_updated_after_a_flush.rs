//! `Index::fold_match_sources` (the one-pass walk behind the facts rebuild and the `significant_text` aggregation)
//! must return the live copy of every document exactly once, including a document updated after a flush.
//!
//! Such an update leaves the pre-update copy in the older segment, which the walk reaches first. The fold used to mark
//! the id as seen on that stale copy, then skip the copy as superseded, and then drop the live copy in the newer segment
//! as a duplicate: it returned neither, and the facts rebuild after a restart failed with "no fact document". The scan
//! arms check liveness before they mark an id seen; the fold now does the same.

use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::sync::Arc;
use tempfile::TempDir;
use xerj_common::config::Config;
use xerj_common::types::Schema;
use xerj_engine::{Engine, Index};
use xerj_query::ast::QueryNode;

fn make_engine(dir: &TempDir) -> Engine {
    let mut config = Config::default();
    config.server.data_dir = dir.path().to_str().unwrap().to_string();
    Engine::new(config).expect("engine::new")
}

/// `k` -> every `v` the fold returned for it, in walk order.
async fn fold_all(index: &Arc<Index>) -> BTreeMap<String, Vec<i64>> {
    let mut out: BTreeMap<String, Vec<i64>> = BTreeMap::new();
    index
        .fold_match_sources(&QueryNode::MatchAll, &mut |src: &Value| {
            let k = src["k"].as_str().expect("k").to_string();
            out.entry(k)
                .or_default()
                .push(src["v"].as_i64().expect("v"));
        })
        .await;
    out
}

fn expected() -> BTreeMap<String, Vec<i64>> {
    BTreeMap::from([("a".to_string(), vec![2]), ("b".to_string(), vec![2])])
}

/// Segment 1: a, b, c (v1). Segment 2: a (v2) and the delete of c. Memtable: b (v2), unless `flush_last`.
async fn seed(index: &Arc<Index>, flush_last: bool) {
    for k in ["a", "b", "c"] {
        index
            .index_document(Some(k.into()), json!({"k": k, "v": 1}))
            .await
            .unwrap();
    }
    index.flush().await.unwrap();
    index
        .index_document(Some("a".into()), json!({"k": "a", "v": 2}))
        .await
        .unwrap();
    assert!(index.delete_document("c").await.unwrap());
    index.flush().await.unwrap();
    index
        .index_document(Some("b".into()), json!({"k": "b", "v": 2}))
        .await
        .unwrap();
    if flush_last {
        index.flush().await.unwrap();
    }
}

fn assert_stale_copy_is_on_disk(index: &Arc<Index>) {
    assert!(
        index.store_snapshot().segments.len() >= 2,
        "test setup: the pre-update copy of `a` must still sit in an older segment than its live copy"
    );
}

#[tokio::test]
async fn fold_returns_the_live_copy_of_a_doc_updated_after_a_flush() {
    let dir = TempDir::new().unwrap();
    let engine = make_engine(&dir);
    engine
        .create_index("fold_updated", Schema::empty())
        .unwrap();
    let index = engine.get_index("fold_updated").unwrap();
    seed(&index, false).await;
    assert_stale_copy_is_on_disk(&index);

    assert_eq!(fold_all(&index).await, expected());
}

#[tokio::test]
async fn fold_returns_the_live_copy_of_a_doc_updated_after_a_flush_after_a_reopen() {
    let dir = TempDir::new().unwrap();
    {
        let engine = make_engine(&dir);
        engine.create_index("fold_reopen", Schema::empty()).unwrap();
        let index = engine.get_index("fold_reopen").unwrap();
        seed(&index, true).await;
        assert_eq!(fold_all(&index).await, expected(), "before the reopen");
    }
    let engine = make_engine(&dir);
    let index = engine.get_index("fold_reopen").unwrap();
    assert_stale_copy_is_on_disk(&index);

    assert_eq!(fold_all(&index).await, expected(), "after the reopen");
}
