//! A `semantic_text` field whose documents carry passage vectors is never
//! served by the HNSW graph (the passage guard, `passage_chunks_require_exact`,
//! sends every kNN / `semantic` query on it to the exact passage scorer), so
//! ingest must not pay for graph inserts on it.
//!
//! Before: every document of such a field was inserted into the graph, one at
//! a time under the graph's write lock — 23.3 % of all server CPU on a
//! 100,000-passage `xerj autoindex` run of Wikipedia passages, most of which
//! are longer than the 512-character semantic chunk. After: no node is added
//! for a passage-scored field, `semantic` results are unchanged (the exact
//! scorer served them before and after), and a field whose texts stay under
//! one chunk keeps its graph and its ANN path.

use serde_json::{json, Value};
use tempfile::TempDir;
use xerj_common::config::Config;
use xerj_common::types::{EmbeddingConfig, FieldConfig, FieldType, Schema};
use xerj_engine::{Engine, Index};
use xerj_query::parse_request;

const DIMS: usize = 64;
const DOCS: usize = 300;

fn make_engine(dir: &TempDir) -> Engine {
    let mut config = Config::default();
    config.server.data_dir = dir.path().to_str().unwrap().to_string();
    Engine::new(config).expect("engine::new")
}

/// `body` as `semantic_text` with the built-in lexical embedder, its vector
/// companion `body_vector` (what the ES mapping layer derives).
fn semantic_schema() -> Schema {
    let mut schema = Schema::empty();
    let mut body = FieldConfig::new("body", FieldType::Text);
    body.options.dimensions = Some(DIMS);
    body.embedding = Some(EmbeddingConfig {
        endpoint: None,
        model: None,
        target_field: Some("body_vector".into()),
    });
    schema.add_field(body).unwrap();
    let mut vector = FieldConfig::new("body_vector", FieldType::Vector);
    vector.options.dimensions = Some(DIMS);
    vector.options.similarity = Some("cosine".into());
    schema.add_field(vector).unwrap();
    schema
}

const TOPICS: [&str; 6] = [
    "volcano",
    "glacier",
    "harbour",
    "orchard",
    "comet",
    "monastery",
];

/// A body of about `len` characters about one topic.
fn body(n: usize, len: usize) -> String {
    let topic = TOPICS[n % TOPICS.len()];
    let mut text = format!("Document {n} is about the {topic}. ");
    while text.len() < len {
        text.push_str(&format!(
            "The {topic} record {n} continues with details about the {topic} and its history. "
        ));
    }
    text
}

async fn seed(engine: &Engine, name: &str, len: usize) -> std::sync::Arc<Index> {
    engine.create_index(name, semantic_schema()).unwrap();
    let idx = engine.get_index(name).unwrap();
    for n in 0..DOCS {
        idx.index_document(Some(format!("d{n}")), json!({ "body": body(n, len) }))
            .await
            .unwrap();
    }
    idx
}

async fn semantic_top(idx: &Index, text: &str) -> Vec<String> {
    let request = parse_request(&json!({
        "query": {"semantic": {"field": "body", "query": text, "k": 5}},
        "size": 5
    }))
    .unwrap();
    idx.search(&request)
        .await
        .unwrap()
        .hits
        .into_iter()
        .map(|hit| hit.id)
        .collect()
}

fn as_u64(stats: &Value, key: &str) -> u64 {
    stats[key]
        .as_u64()
        .unwrap_or_else(|| panic!("{key} in {stats}"))
}

#[tokio::test(flavor = "multi_thread")]
async fn a_passage_scored_field_adds_no_graph_nodes_and_keeps_its_semantic_results() {
    let dir = TempDir::new().unwrap();
    let engine = make_engine(&dir);

    // 1,500-character bodies: every document is chunked into passages.
    let long = seed(&engine, "passages", 1500).await;
    let stats = long.hnsw_stats().await;
    assert_eq!(
        stats["semantic_ann_blocked_by_passages"],
        json!(true),
        "the corpus must be passage-scored for this test to mean anything: {stats}"
    );
    assert_eq!(
        as_u64(&stats, "doc_coverage"),
        0,
        "graph inserts were paid for a field ANN can never serve: {stats}"
    );
    assert!(
        as_u64(&stats, "vector_doc_count") >= DOCS as u64,
        "skipped documents still count as vector documents, so the coverage \
         gate reads incomplete: {stats}"
    );
    assert_eq!(stats["covered"], json!(false), "{stats}");
    // The exact passage scorer answers: the first hits are on topic.
    let top = semantic_top(&long, "volcano history").await;
    assert!(!top.is_empty());
    for id in &top {
        let n: usize = id[1..].parse().unwrap();
        assert_eq!(
            TOPICS[n % TOPICS.len()],
            "volcano",
            "off-topic hit {id} in {top:?}"
        );
    }
    // Overwrites and deletes of skipped documents keep the counters sane.
    long.index_document(Some("d0".into()), json!({ "body": body(6, 1500) }))
        .await
        .unwrap();
    long.delete_document("d1").await.unwrap();
    let stats = long.hnsw_stats().await;
    assert_eq!(as_u64(&stats, "doc_coverage"), 0, "{stats}");
    assert_eq!(stats["covered"], json!(false), "{stats}");

    // 200-character bodies stay one chunk: the graph is still built and the
    // field stays ANN-servable.
    let short = seed(&engine, "single-chunk", 200).await;
    let stats = short.hnsw_stats().await;
    assert_eq!(
        stats["semantic_ann_blocked_by_passages"],
        json!(false),
        "{stats}"
    );
    assert_eq!(as_u64(&stats, "doc_coverage"), DOCS as u64, "{stats}");
    assert_eq!(stats["covered"], json!(true), "{stats}");
    assert!(!semantic_top(&short, "glacier history").await.is_empty());
}
