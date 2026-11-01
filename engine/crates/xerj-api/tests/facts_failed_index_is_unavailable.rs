//! A facts namespace whose index exists on disk but cannot be opened at boot is unavailable, not missing.
//!
//! The engine answers such an index with 503 `no_shard_available_action_exception` and the open error (issue #206:
//! "a name that failed to open is not 'not found'"), but `/_facts/{ns}` mapped every `get_index` error to 404 "no facts
//! namespace", which tells a client that the namespace holds no data. Seen in the E2 coverage runs on the x86 build server:
//! with the test process at a 1,024 open-file limit, one committed namespace's index could not be opened after the reopen,
//! and `GET /_facts/{ns}` answered 404 for it.
#![cfg(feature = "facts")]

use axum::body::Body;
use axum::http::{Request, StatusCode};
use serde_json::{json, Value};
use std::path::Path;
use tower::ServiceExt;

fn app_over(dir: &Path) -> (axum::Router, xerj_api::state::AppState) {
    let mut config = xerj_common::config::Config::default();
    config.server.data_dir = dir.to_string_lossy().into_owned();
    let metrics = xerj_common::metrics::Metrics::new().expect("metrics");
    let engine = xerj_engine::Engine::new(config.clone()).expect("engine");
    let state = xerj_api::state::AppState::new(config, engine, metrics);
    (
        xerj_api::router::build_es_compat_router(state.clone()),
        state,
    )
}

async fn send(
    app: &axum::Router,
    method: &str,
    path: &str,
    body: Option<Value>,
) -> (StatusCode, Value) {
    let req = Request::builder()
        .method(method)
        .uri(path)
        .header("content-type", "application/json")
        .body(body.map_or_else(Body::empty, |b| Body::from(b.to_string())))
        .expect("request");
    let response = app.clone().oneshot(req).await.expect("response");
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body");
    (
        status,
        serde_json::from_slice(&bytes).unwrap_or(Value::Null),
    )
}

fn world() -> Value {
    json!({
        "world_id": "t1",
        "base_documents": [{"doc_id": "d1", "version": "v1", "text": "Liberia is governed by George Weah."}],
        "base_facts": [{"fact_id": "f1", "subject": "Liberia", "relation": "head_of_government", "kind": "text",
                        "value": "George Weah", "doc_id": "d1", "version": "v1", "start": 0, "end": 35,
                        "quote": "Liberia is governed by George Weah."}],
        "edits": []
    })
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_namespace_whose_index_did_not_open_is_unavailable_not_missing() {
    let dir = tempfile::tempdir().expect("tempdir");
    {
        let (app, state) = app_over(dir.path());
        let (st, body) = send(&app, "POST", "/_facts/kb/_world", Some(world())).await;
        assert_eq!(st, StatusCode::OK, "{body}");
        let (st, body) = send(&app, "GET", "/_facts/kb", None).await;
        assert_eq!(
            (st, &body["published_rev"]),
            (StatusCode::OK, &json!(2)),
            "{body}"
        );
        state.engine.flush_all_force().await;
    }
    // Break the namespace's index the way the engine's own issue #206 test breaks one (a corrupt snapshot manifest).
    let snapshot = dir
        .path()
        .join(".xerj-memory-kb-facts")
        .join("snapshot.json");
    assert!(snapshot.exists(), "{} after a flush", snapshot.display());
    std::fs::write(&snapshot, b"{not json").expect("corrupt the manifest");

    let (app, state) = app_over(dir.path());
    assert_eq!(
        state.engine.list_failed_indices().len(),
        1,
        "the fixture must produce exactly one failed index"
    );
    let lookup = json!({"subject": "Liberia", "relation": "head_of_government"});
    let ask = json!({"question": "Who is the head of government of Liberia?"});
    for (method, path, body) in [
        ("GET", "/_facts/kb", None),
        ("POST", "/_facts/kb/_lookup", Some(lookup)),
        ("POST", "/_facts/kb/_ask", Some(ask)),
    ] {
        let (st, out) = send(&app, method, path, body).await;
        assert_eq!(
            st,
            StatusCode::SERVICE_UNAVAILABLE,
            "{method} {path} on a namespace whose index did not open: {out}"
        );
        assert_eq!(
            out["error"]["type"], "no_shard_available_action_exception",
            "{method} {path}: {out}"
        );
    }
    // A namespace that never existed is still "not found".
    let (st, out) = send(&app, "GET", "/_facts/never", None).await;
    assert_eq!(st, StatusCode::NOT_FOUND, "{out}");
}
