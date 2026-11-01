//! New segments use the compact postings codec, and the data directory says so before the first
//! one exists: a build that cannot read it refuses at open instead of misreading the postings.
use serde_json::{json, Value};
use tempfile::TempDir;
use xerj_common::config::Config;
use xerj_common::types::{FieldConfig, FieldType, Schema};
use xerj_engine::Engine;
use xerj_query::parse_request;

fn make_engine(dir: &TempDir) -> Engine {
    let mut config = Config::default();
    config.server.data_dir = dir.path().to_str().unwrap().to_string();
    Engine::new(config).expect("engine::new")
}

fn schema() -> Schema {
    let mut schema = Schema::empty();
    schema
        .add_field(FieldConfig::new("body", FieldType::Text))
        .unwrap();
    schema
}

fn marker_version(dir: &TempDir) -> u64 {
    let raw = std::fs::read(find(dir.path(), "xerj_meta.json")[0].clone()).unwrap();
    serde_json::from_slice::<Value>(&raw).unwrap()["format_version"]
        .as_u64()
        .unwrap()
}

fn find(root: &std::path::Path, suffix: &str) -> Vec<std::path::PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(dir).unwrap().flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.to_string_lossy().ends_with(suffix) {
                out.push(path);
            }
        }
    }
    out
}

async fn hits(engine: &Engine, body: Value) -> Vec<String> {
    let idx = engine.get_index("docs").unwrap();
    let request = parse_request(&body).unwrap();
    let response = idx.search(&request).await.unwrap();
    response.hits.iter().map(|h| h.id.clone()).collect()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_flush_fences_the_directory_and_phrases_still_answer() {
    let dir = TempDir::new().unwrap();
    let engine = make_engine(&dir);
    engine.create_index("docs", schema()).unwrap();
    let idx = engine.get_index("docs").unwrap();
    for i in 0..300u32 {
        let body = format!("alpha beta gamma number {i} alpha delta beta");
        idx.index_document(Some(format!("d{i}")), json!({ "body": body }))
            .await
            .unwrap();
    }
    idx.flush().await.unwrap();
    let posts = find(dir.path(), ".body.post");
    assert!(!posts.is_empty());
    for post in &posts {
        assert_eq!(&std::fs::read(post).unwrap()[..4], b"ZPS4");
    }
    assert_eq!(marker_version(&dir), 4);

    let phrase = json!({"query": {"match_phrase": {"body": "number 7 alpha"}}, "size": 5});
    let term = json!({"query": {"match": {"body": "gamma"}}, "size": 400});
    assert_eq!(hits(&engine, phrase.clone()).await, ["d7"]);
    assert_eq!(hits(&engine, term.clone()).await.len(), 300);
    drop(idx);
    drop(engine);

    let engine = make_engine(&dir);
    assert_eq!(hits(&engine, phrase).await, ["d7"]);
    assert_eq!(hits(&engine, term).await.len(), 300);
    assert_eq!(marker_version(&dir), 4);
}
