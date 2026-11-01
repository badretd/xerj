//! Text left in the original file, through the engine: ingest, flush, merge, search, GET-shaped
//! projections, and an edited file. Its own test binary: the root is process-wide.
use serde_json::{json, Value};
use tempfile::TempDir;
use xerj_common::config::Config;
use xerj_common::types::{FieldConfig, FieldType, Schema};
use xerj_engine::Engine;
use xerj_query::parse_request;
use xerj_storage::stored_codec::decode_stored_preserving_derived;
use xerj_storage::stored_codec::source_ref::{set_root, ERROR_FIELD, REF_FIELD};
use xerj_storage::{SectionType, SegmentReader};

const INDEX: &str = "refs";

fn schema() -> Schema {
    let mut schema = Schema::empty();
    schema
        .add_field(FieldConfig::new("text", FieldType::Text))
        .unwrap();
    schema
}

async fn sources(engine: &Engine, query: Value) -> Vec<Value> {
    let idx = engine.get_index(INDEX).unwrap();
    let request = parse_request(&query).unwrap();
    let result = idx.search(&request).await.unwrap();
    result.hits.iter().map(|h| h.source.clone()).collect()
}

/// Text bytes left in stored sections, read the way a merge reads them.
fn stored_text_bytes(dir: &std::path::Path) -> usize {
    let mut total = 0;
    for entry in std::fs::read_dir(dir).unwrap().flatten() {
        let path = entry.path();
        if path.is_dir() {
            total += stored_text_bytes(&path);
        } else if path.extension().is_some_and(|e| e == "seg") {
            let reader = SegmentReader::open(&path).unwrap();
            let Some(raw) = reader.section(SectionType::Stored).unwrap() else {
                continue;
            };
            let bytes = decode_stored_preserving_derived(raw).unwrap();
            let docs: Vec<Value> = serde_json::from_slice(&bytes).unwrap();
            total += docs
                .iter()
                .filter_map(|d| d["_source"]["text"].as_str())
                .map(str::len)
                .sum::<usize>();
        }
    }
    total
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn searches_read_the_text_from_the_file_and_an_edit_gives_an_error() {
    let data = TempDir::new().unwrap();
    let refs = TempDir::new().unwrap();
    let file = refs.path().join("orig.jsonl");
    // Segments below 128 rows are stored whole, so this needs a few thousand rows.
    // Words that do not repeat: a section of repeats loses to LZ4 over the whole JSON, whose
    // size net then keeps the documents whole.
    let mut x = 0x9E37_79B9_7F4A_7C15u64;
    let mut word = move || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        format!("{x:x}")
    };
    let lines: Vec<String> = (0..2_000)
        .map(|i| {
            let words: Vec<String> = (0..40).map(|_| word()).collect();
            let text = format!("needle{i} {}", words.join(" "));
            json!({"id": format!("{i}#0"), "text": text}).to_string()
        })
        .collect();
    std::fs::write(&file, lines.join("\n") + "\n").unwrap();
    let opts = xerj_pagedir::BuildOptions {
        article_key: xerj_pagedir::ArticleKey::None,
        ..Default::default()
    };
    xerj_pagedir::build(&file, &refs.path().join("d1.pgdir"), opts).unwrap();
    std::fs::write(refs.path().join("d1.source"), file.to_str().unwrap()).unwrap();
    set_root(Some(refs.path().to_owned()));

    let mut config = Config::default();
    config.server.data_dir = data.path().to_str().unwrap().to_string();
    let engine = Engine::new(config.clone()).unwrap();
    engine.create_index(INDEX, schema()).unwrap();
    let idx = engine.get_index(INDEX).unwrap();
    for (i, line) in lines.iter().enumerate() {
        let mut doc: Value = serde_json::from_str(line).unwrap();
        doc[REF_FIELD] = json!({
            "dir": "d1", "p": i, "fields": ["text"],
            "h": format!("{:016x}", xerj_pagedir::line_hash(line.as_bytes())),
        });
        idx.index_document(Some(format!("d{i}")), doc)
            .await
            .unwrap();
        if i == 999 {
            idx.flush().await.unwrap();
        }
    }
    idx.flush().await.unwrap();
    assert_eq!(
        stored_text_bytes(data.path()),
        0,
        "flush must strip the text"
    );
    assert!(idx.force_merge(1).await.unwrap() > 0);

    assert_eq!(
        stored_text_bytes(data.path()),
        0,
        "a merge must keep rows stripped"
    );

    let want: Value = serde_json::from_str(&lines[37]).unwrap();
    let query = json!({"query": {"match": {"text": "needle37"}}, "size": 3});
    let got = sources(&engine, query.clone()).await;
    assert_eq!(got.len(), 1);
    assert_eq!(got[0].get(ERROR_FIELD), None);
    assert_eq!(got[0]["text"], want["text"]);
    assert!(got[0].get(ERROR_FIELD).is_none());

    // A projection on the stripped field still answers.
    let projected = json!({"query": {"match": {"text": "needle37"}}, "_source": ["text"]});
    assert_eq!(sources(&engine, projected).await[0]["text"], want["text"]);

    // An edit that keeps the size: the row errors, it never shows other text.
    let raw = std::fs::read(&file).unwrap();
    let at = raw.windows(9).position(|w| w == b"needle37 ").unwrap();
    let mut edited = raw.clone();
    edited[at] = b'N';
    std::fs::write(&file, &edited).unwrap();
    set_root(Some(refs.path().to_owned()));
    // A restart empties the decoded-row caches, which still hold the verified text.
    drop(idx);
    drop(engine);
    let engine = Engine::new(config).unwrap();
    let got = sources(&engine, query).await;
    assert_eq!(got.len(), 1);
    assert_eq!(got[0]["text"], "");
    assert!(got[0][ERROR_FIELD]
        .as_str()
        .unwrap()
        .starts_with("stale file"));
    set_root(None);
}
