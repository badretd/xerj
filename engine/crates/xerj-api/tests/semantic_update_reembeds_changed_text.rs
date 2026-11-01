//! A `semantic_text` field's vectors must follow its text through every
//! read-modify-write update path.
//!
//! Partial `_update`, scripted `_update`, `_update_by_query` and `_bulk`
//! update actions read the current document back (vectors included) and
//! re-index it. The vectors of the OLD text came back as if the caller had
//! supplied them, so the write kept them for the NEW text: the document stored
//! text B with the pooled and chunk vectors of text A, and a `semantic` query
//! scored it like the text it no longer has. An update must embed changed text
//! exactly as indexing the same document does, and leave the vectors (and the
//! passage offsets kept with them) of unchanged text alone.
//!
//! Elasticsearch is referenced for wire semantics only; no ES code is
//! reproduced here.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use serde_json::{json, Value};
use tower::ServiceExt;

async fn app() -> (axum::Router, tempfile::TempDir) {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut config = xerj_common::config::Config::default();
    config.server.data_dir = dir.path().to_string_lossy().into_owned();
    config.storage.wal_sync = xerj_common::config::WalSync::Async;
    let metrics = xerj_common::metrics::Metrics::new().expect("metrics");
    let engine = xerj_engine::Engine::new(config.clone()).expect("engine");
    let state = xerj_api::state::AppState::new(config, engine, metrics);
    (xerj_api::router::build_es_compat_router(state), dir)
}

async fn send(app: &axum::Router, method: &str, path: &str, body: Value) -> (StatusCode, Value) {
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method(method)
                .uri(path)
                .header("content-type", "application/json")
                .body(if body.is_null() {
                    Body::empty()
                } else {
                    Body::from(body.to_string())
                })
                .unwrap(),
        )
        .await
        .unwrap();
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    (
        status,
        serde_json::from_slice(&bytes).unwrap_or(Value::Null),
    )
}

/// ~790 characters each, so every text is two semantic chunks (512 / 64):
/// the pooled vector AND `sem_vector_chunks` are exercised.
fn text_a() -> String {
    "Alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike. "
        .repeat(10)
        .trim()
        .to_string()
}

fn text_b() -> String {
    "Zebras graze near quantum laboratories while violins tune their strings slowly. "
        .repeat(10)
        .trim()
        .to_string()
}

fn text_c() -> String {
    "Mountain rivers carve granite canyons beneath silent glaciers every spring season. "
        .repeat(10)
        .trim()
        .to_string()
}

const QUERY_B: &str = "zebras graze near quantum laboratories while violins tune";

/// Index `t` with one `semantic_text` field and a keyword, and documents
/// `a` (text A, the one the tests update), `b` (text B, the reference a fresh
/// index produces) and `u` (an unrelated text C).
async fn seeded() -> (axum::Router, tempfile::TempDir) {
    let (app, dir) = app().await;
    let (status, response) = send(
        &app,
        "PUT",
        "/t",
        json!({"mappings": {"properties": {
            "sem": {"type": "semantic_text"},
            "tag": {"type": "keyword"}
        }}}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "create: {response}");
    for (id, text) in [("a", text_a()), ("b", text_b()), ("u", text_c())] {
        let (status, response) = send(
            &app,
            "PUT",
            &format!("/t/_doc/{id}?refresh=true"),
            json!({"sem": text, "tag": "x"}),
        )
        .await;
        assert_eq!(status, StatusCode::CREATED, "index {id}: {response}");
    }
    (app, dir)
}

async fn source(app: &axum::Router, id: &str) -> Value {
    let (status, response) = send(app, "GET", &format!("/t/_doc/{id}"), Value::Null).await;
    assert_eq!(status, StatusCode::OK, "get {id}: {response}");
    response["_source"].clone()
}

/// `(id, score)` of every hit of a `semantic` query, best first.
async fn semantic_hits(app: &axum::Router, query: &str) -> Vec<(String, f64)> {
    let (status, response) = send(
        app,
        "POST",
        "/t/_search",
        json!({"query": {"semantic": {"field": "sem", "query": query}}, "_source": false}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "search: {response}");
    response["hits"]["hits"]
        .as_array()
        .expect("hits")
        .iter()
        .map(|hit| {
            (
                hit["_id"].as_str().unwrap().to_string(),
                hit["_score"].as_f64().unwrap(),
            )
        })
        .collect()
}

/// After `a`'s text became B, `a` must carry exactly the vectors a fresh
/// index of B produces (document `b`) and score exactly like `b`, above the
/// unrelated `u`.
async fn assert_a_was_reembedded_as_b(app: &axum::Router, a_before: &Value, path: &str) {
    let a = source(app, "a").await;
    let b = source(app, "b").await;
    assert_eq!(a["sem"], json!(text_b()), "{path}: the text changed");
    assert!(
        a["sem_vector"].is_array() && a["sem_vector_chunks"].is_array(),
        "{path}: vectors present: {a}"
    );
    assert_ne!(
        a["sem_vector"], a_before["sem_vector"],
        "{path}: the pooled vector of the OLD text was kept for the new text"
    );
    assert_eq!(
        a["sem_vector"], b["sem_vector"],
        "{path}: pooled vector must equal a fresh index of the same text"
    );
    assert_eq!(
        a["sem_vector_chunks"], b["sem_vector_chunks"],
        "{path}: chunk vectors must equal a fresh index of the same text"
    );

    let hits = semantic_hits(app, QUERY_B).await;
    let score = |id: &str| {
        hits.iter()
            .find(|(hit, _)| hit == id)
            .unwrap_or_else(|| panic!("{path}: {id} missing from {hits:?}"))
            .1
    };
    assert_eq!(
        score("a"),
        score("b"),
        "{path}: two documents holding the same text must score the same: {hits:?}"
    );
    assert!(
        score("a") > score("u"),
        "{path}: the updated document must outrank an unrelated one: {hits:?}"
    );
    let top: std::collections::HashSet<&str> =
        hits.iter().take(2).map(|(id, _)| id.as_str()).collect();
    assert_eq!(
        top,
        ["a", "b"]
            .into_iter()
            .collect::<std::collections::HashSet<&str>>(),
        "{path}: ranking {hits:?}"
    );
}

#[tokio::test]
async fn partial_update_reembeds_changed_semantic_text() {
    let (app, _dir) = seeded().await;
    let before = source(&app, "a").await;
    let (status, response) = send(
        &app,
        "POST",
        "/t/_update/a?refresh=true",
        json!({"doc": {"sem": text_b()}}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "update: {response}");
    assert_eq!(response["result"], "updated", "{response}");
    assert_a_was_reembedded_as_b(&app, &before, "partial _update").await;
}

#[tokio::test]
async fn scripted_update_reembeds_changed_semantic_text() {
    let (app, _dir) = seeded().await;
    let before = source(&app, "a").await;
    let (status, response) = send(
        &app,
        "POST",
        "/t/_update/a?refresh=true",
        json!({"script": {
            "source": "ctx._source.sem = params.t",
            "params": {"t": text_b()}
        }}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "update: {response}");
    assert_eq!(response["result"], "updated", "{response}");
    assert_a_was_reembedded_as_b(&app, &before, "scripted _update").await;
}

#[tokio::test]
async fn update_by_query_reembeds_changed_semantic_text() {
    let (app, _dir) = seeded().await;
    let before = source(&app, "a").await;
    let (status, response) = send(
        &app,
        "POST",
        "/t/_update_by_query?refresh=true",
        json!({
            "query": {"ids": {"values": ["a"]}},
            "script": {"source": "ctx._source.sem = params.t", "params": {"t": text_b()}}
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "update_by_query: {response}");
    assert_eq!(response["updated"], 1, "{response}");
    assert_a_was_reembedded_as_b(&app, &before, "_update_by_query").await;
}

#[tokio::test]
async fn bulk_update_action_reembeds_changed_semantic_text() {
    let (app, _dir) = seeded().await;
    let before = source(&app, "a").await;
    let body = format!(
        "{}\n{}\n",
        json!({"update": {"_index": "t", "_id": "a"}}),
        json!({"doc": {"sem": text_b()}})
    );
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/_bulk?refresh=true")
                .header("content-type", "application/x-ndjson")
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let response: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    assert_eq!(status, StatusCode::OK, "bulk: {response}");
    assert_eq!(response["errors"], false, "{response}");
    assert_a_was_reembedded_as_b(&app, &before, "_bulk update").await;
}

/// The negative: updates that leave the text alone must not touch its
/// vectors. Every path, byte-identical (the GET renders the stored packed f32
/// as the same number array only if the stored bits are unchanged).
#[tokio::test]
async fn updates_that_do_not_touch_the_text_keep_its_vectors_byte_identical() {
    let (app, _dir) = seeded().await;
    let before = source(&app, "a").await;
    assert!(before["sem_vector_chunks"].is_array(), "{before}");

    let steps = [
        (
            "partial _update of another field",
            "/t/_update/a?refresh=true",
            json!({"doc": {"tag": "y"}}),
        ),
        (
            "scripted _update of another field",
            "/t/_update/a?refresh=true",
            json!({"script": {"source": "ctx._source.tag = params.v", "params": {"v": "z"}}}),
        ),
        (
            "_update_by_query of another field",
            "/t/_update_by_query?refresh=true",
            json!({
                "query": {"ids": {"values": ["a"]}},
                "script": {"source": "ctx._source.tag = params.v", "params": {"v": "w"}}
            }),
        ),
        (
            "partial _update that writes the same text",
            "/t/_update/a?refresh=true",
            json!({"doc": {"sem": text_a(), "tag": "v"}}),
        ),
    ];
    for (what, path, body) in steps {
        let (status, response) = send(&app, "POST", path, body).await;
        assert_eq!(status, StatusCode::OK, "{what}: {response}");
        let after = source(&app, "a").await;
        assert_eq!(after["sem"], before["sem"], "{what}: text unchanged");
        assert_eq!(
            after["sem_vector"].to_string(),
            before["sem_vector"].to_string(),
            "{what}: pooled vector must stay byte-identical"
        );
        assert_eq!(
            after["sem_vector_chunks"].to_string(),
            before["sem_vector_chunks"].to_string(),
            "{what}: chunk vectors must stay byte-identical"
        );
    }
    assert_eq!(source(&app, "a").await["tag"], "v");
}

/// The passage offsets stored with generated vectors are part of what an
/// update that does not touch the text must leave alone: without them a
/// `semantic` query can no longer say which passage of the document matched.
#[tokio::test]
async fn an_update_that_does_not_touch_the_text_keeps_its_passage_offsets() {
    let (app, _dir) = seeded().await;
    let passage_of_a = |response: &Value| -> Option<Value> {
        response["hits"]["hits"]
            .as_array()?
            .iter()
            .find(|hit| hit["_id"] == "a")
            .map(|hit| hit["fields"]["_passage"].clone())
    };
    let query = json!({
        "query": {"semantic": {"field": "sem", "query": "alpha bravo charlie delta echo"}},
        "fields": ["_passage"],
        "_source": false
    });
    let (status, response) = send(&app, "POST", "/t/_search", query.clone()).await;
    assert_eq!(status, StatusCode::OK, "{response}");
    let before = passage_of_a(&response).expect("a matches");
    assert!(
        before.is_array(),
        "passage offered before the update: {response}"
    );

    let (status, response) = send(
        &app,
        "POST",
        "/t/_update/a?refresh=true",
        json!({"doc": {"tag": "y"}}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{response}");

    let (status, response) = send(&app, "POST", "/t/_search", query).await;
    assert_eq!(status, StatusCode::OK, "{response}");
    assert_eq!(
        passage_of_a(&response),
        Some(before),
        "an update of another field dropped the passage offsets: {response}"
    );
}
