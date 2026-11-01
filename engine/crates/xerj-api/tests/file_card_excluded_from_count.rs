//! A file-card anchor (`ax_locator: "file"`) lives in the dataset index so a
//! graph edge can resolve it. It is not a content record. A bare `_count` —
//! what `xerj autoindex map` reads, and what an agent curls — must not count
//! it. An explicit query still does, and `_search` can still return the card.

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

async fn json_req(
    app: &axum::Router,
    method: &str,
    path: &str,
    body: Value,
) -> (StatusCode, Value) {
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method(method)
                .uri(path)
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .expect("request"),
        )
        .await
        .expect("response");
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body");
    (
        status,
        serde_json::from_slice(&bytes).unwrap_or(Value::Null),
    )
}

#[tokio::test]
async fn bare_count_skips_file_cards_and_a_term_query_still_sees_them() {
    let (app, _dir) = app().await;
    let (status, body) = json_req(
        &app,
        "PUT",
        "/docs",
        json!({
            "mappings": { "properties": {
                "body": { "type": "text" },
                "ax_locator": { "type": "keyword" }
            }}
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "create index: {body}");

    for (id, doc) in [
        ("s0", json!({"body": "the design", "ax_locator": "s0"})),
        ("s1", json!({"body": "the rest", "ax_locator": "s1"})),
        ("card", json!({"body": "README.md", "ax_locator": "file"})),
    ] {
        let (status, body) = json_req(&app, "PUT", &format!("/docs/_doc/{id}"), doc).await;
        assert!(status.is_success(), "index {id}: {status} {body}");
    }
    let (status, body) = json_req(&app, "POST", "/docs/_refresh", json!({})).await;
    assert!(status.is_success(), "refresh: {status} {body}");

    let (status, count) = json_req(&app, "GET", "/docs/_count", Value::Null).await;
    assert_eq!(status, StatusCode::OK, "{count}");
    assert_eq!(
        count["count"].as_u64(),
        Some(2),
        "the file card must not inflate a bare count: {count}"
    );

    let (status, glob) = json_req(&app, "GET", "/docs*/_count", Value::Null).await;
    assert_eq!(status, StatusCode::OK, "{glob}");
    assert_eq!(glob["count"].as_u64(), Some(2), "{glob}");

    let (status, cards) = json_req(
        &app,
        "POST",
        "/docs/_count",
        json!({"query": {"term": {"ax_locator": "file"}}}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{cards}");
    assert_eq!(
        cards["count"].as_u64(),
        Some(1),
        "an explicit card query still counts the anchor: {cards}"
    );

    let (status, search) = json_req(
        &app,
        "POST",
        "/docs/_search",
        json!({"size": 0, "track_total_hits": true, "query": {"match_all": {}}}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{search}");
    assert_eq!(
        search.pointer("/hits/total/value").and_then(Value::as_u64),
        Some(3),
        "the card stays in the index: {search}"
    );
}

/// `_stats.search.query_total` of one index: every `Index::search` call, a
/// result-cache hit included, adds one.
async fn query_total(app: &axum::Router, index: &str) -> u64 {
    let (status, stats) = json_req(app, "GET", &format!("/{index}/_stats"), Value::Null).await;
    assert_eq!(status, StatusCode::OK, "{stats}");
    stats
        .pointer("/_all/primaries/search/query_total")
        .and_then(Value::as_u64)
        .unwrap_or_else(|| panic!("no query_total in {stats}"))
}

/// An index whose mapping has no `ax_locator` field cannot hold a file card,
/// so a bare `_count` and `_cat/indices` answer from the stat count without
/// running a search. On an unmapped field the card query is answered by the
/// stored-document scan, which decodes every document of every segment, and
/// the result cache is invalidated by every write: a monitor polling
/// `_cat/indices` during a bulk load paid one whole-corpus scan per poll.
#[tokio::test]
async fn an_index_without_ax_locator_counts_without_a_search() {
    let (app, _dir) = app().await;
    let (status, body) = json_req(
        &app,
        "PUT",
        "/plain",
        json!({
            "mappings": { "properties": {
                "id": { "type": "keyword" },
                "text": { "type": "text" }
            }}
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "create index: {body}");
    for i in 0..3 {
        let (status, body) = json_req(
            &app,
            "PUT",
            &format!("/plain/_doc/p{i}"),
            json!({"id": format!("p{i}"), "text": "a passage"}),
        )
        .await;
        assert!(status.is_success(), "index p{i}: {status} {body}");
    }
    let (status, body) = json_req(&app, "POST", "/plain/_refresh", json!({})).await;
    assert!(status.is_success(), "refresh: {status} {body}");

    let before = query_total(&app, "plain").await;
    let (status, count) = json_req(&app, "GET", "/plain/_count", Value::Null).await;
    assert_eq!(status, StatusCode::OK, "{count}");
    assert_eq!(count["count"].as_u64(), Some(3), "{count}");
    let (status, cat) = json_req(&app, "GET", "/_cat/indices/plain?format=json", Value::Null).await;
    assert_eq!(status, StatusCode::OK, "{cat}");
    assert_eq!(
        cat.pointer("/0/docs.count").and_then(Value::as_str),
        Some("3"),
        "{cat}"
    );
    assert_eq!(
        query_total(&app, "plain").await,
        before,
        "a bare _count and _cat/indices on an index without ax_locator must not run a search"
    );
}
