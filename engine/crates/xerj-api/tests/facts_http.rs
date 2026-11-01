//! `/_facts` over HTTP without a compiler model (design §5.3, PR-4 / PR-6): operational failures are HTTP errors with
//! a stable code, never a factual UNKNOWN.
//!
//! * `501 FACTS_CLUSTER_UNSUPPORTED` on every route of a node configured for a cluster;
//! * `503 ENRICHMENT_PENDING` (`retryable`, `Retry-After`) when the namespace's projection is being rebuilt and the
//!   caller's wait runs out; without a wait the read waits for the rebuild, as before;
//! * `503 MODEL_UNAVAILABLE`, sticky until restart and `retryable: false`;
//! * `504 ASK_DEADLINE` when every ask slot stays busy past the deadline;
//! * `429 ASK_QUEUE_FULL` at once when every slot is busy and the line of waiting asks is full;
//! * size limits on `_ask` / `_lookup` bodies, questions, item ids, linker, manifest and verify names and write batches,
//!   answered as JSON.
//!
//! The last two tests run the real compiler and are skipped unless `XERJ_FACTS_MODEL_DIR` and the golden fixtures are
//! set (the engine build server's `env.sh`).

#![cfg(feature = "facts")]

use axum::body::Body;
use axum::http::{HeaderMap, Request, StatusCode};
use http_body_util::BodyExt;
use serde_json::{json, Value};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tower::ServiceExt;
use xerj_api::router::build_es_compat_router;
use xerj_api::state::AppState;
use xerj_common::{config::Config, metrics::Metrics};
use xerj_engine::Engine;
use xerj_facts::{AskGate, FactsService};

/// An open (`--insecure`-style) node over `dir`; `configure` adjusts the config, and the facts service has no model and
/// no limits unless the test installs them.
fn node(dir: &std::path::Path, configure: impl FnOnce(&mut Config)) -> AppState {
    let mut config = Config::default();
    config.server.data_dir = dir.to_str().unwrap().to_string();
    config.storage.wal_sync = xerj_common::config::WalSync::Async;
    configure(&mut config);
    let metrics = Metrics::new().expect("metrics");
    let engine = Engine::new(config.clone()).expect("engine");
    let mut state = AppState::new(config, engine, metrics);
    state.facts = Arc::new(FactsService::new(None, "f32"));
    state
}

async fn send(
    app: &axum::Router,
    method: &str,
    uri: &str,
    body: &str,
) -> (StatusCode, Value, HeaderMap) {
    let req = Request::builder()
        .method(method)
        .uri(uri)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .expect("request");
    let resp = app.clone().oneshot(req).await.expect("response");
    let status = resp.status();
    let headers = resp.headers().clone();
    let bytes = resp.into_body().collect().await.expect("body").to_bytes();
    let json: Value = serde_json::from_slice(&bytes)
        .unwrap_or_else(|_| Value::String(String::from_utf8_lossy(&bytes).into_owned()));
    (status, json, headers)
}

fn world() -> Value {
    let text = "Liberia is governed by George Weah.";
    json!({
        "world_id": "t1",
        "base_documents": [{"doc_id": "d1", "version": "v1", "text": text}],
        "base_facts": [{"fact_id": "f1", "subject": "Liberia", "relation": "head_of_government", "kind": "text",
                        "value": "George Weah", "doc_id": "d1", "version": "v1", "start": 0, "end": text.len(),
                        "quote": text}],
        "edits": []
    })
}

const LOOKUP: &str = r#"{"subject":"Liberia","relation":"head_of_government"}"#;

async fn load(app: &axum::Router, ns: &str) {
    let (s, b, _) = send(
        app,
        "POST",
        &format!("/_facts/{ns}/_world"),
        &world().to_string(),
    )
    .await;
    assert_eq!(s, StatusCode::OK, "{b}");
}

fn error_type(b: &Value) -> &str {
    b["error"]["type"].as_str().unwrap_or("")
}

#[test]
fn the_api_and_the_store_name_the_same_index() {
    assert_eq!(
        xerj_api::authz::facts_index("alpha"),
        xerj_facts::facts_index("alpha")
    );
    assert_eq!(
        xerj_api::authz::facts_index("alpha"),
        ".xerj-memory-alpha-facts"
    );
    const {
        assert!(
            xerj_api::authz::FACTS_SUFFIX_RESERVED,
            "a build with /_facts reserves the suffix"
        )
    };
}

/// v1 is single-node: every route answers 501 with a stable code on a node configured for a cluster.
#[tokio::test]
async fn a_cluster_node_answers_501_on_every_route() {
    let dir = tempfile::tempdir().expect("tempdir");
    let app = build_es_compat_router(node(dir.path(), |c| {
        c.server.bind_address = "127.0.0.1".to_string();
        c.cluster.enabled = true;
        c.cluster.auth_secret = "test-secret-0123456789abcdef".to_string();
    }));
    for (method, uri, body) in [
        ("GET", "/_facts", String::new()),
        ("GET", "/_facts/alpha", String::new()),
        ("POST", "/_facts/alpha/_world", world().to_string()),
        ("POST", "/_facts/alpha/_bulk", "{}".to_string()),
        ("POST", "/_facts/alpha/_lookup", LOOKUP.to_string()),
        (
            "POST",
            "/_facts/alpha/_ask",
            r#"{"question":"Who governs Liberia?"}"#.to_string(),
        ),
        ("DELETE", "/_facts/alpha/f1", String::new()),
        ("DELETE", "/_facts/alpha", String::new()),
    ] {
        let (s, b, _) = send(&app, method, uri, &body).await;
        assert_eq!(s, StatusCode::NOT_IMPLEMENTED, "{method} {uri}: {b}");
        assert_eq!(error_type(&b), "FACTS_CLUSTER_UNSUPPORTED", "{b}");
    }
}

/// A projection that is being rebuilt is a retryable 503, never an answer: the test holds the namespace writer (as a
/// long write or another rebuild does) with the projection dropped, so a rebuild cannot finish.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_lagging_projection_is_503_enrichment_pending_never_unknown() {
    let dir = tempfile::tempdir().expect("tempdir");
    let state = node(dir.path(), |_| {});
    let app = build_es_compat_router(state.clone());
    load(&app, "alpha").await;
    state.facts.remove("alpha");
    let held = state.facts.writer.lock().await;

    let ask = r#"{"question":"Who governs Liberia?","wait_for_ms":100}"#;
    let lookup = r#"{"subject":"Liberia","relation":"head_of_government","wait_for_ms":100}"#;
    for (method, uri, body) in [
        ("POST", "/_facts/alpha/_ask", ask),
        ("POST", "/_facts/alpha/_lookup", lookup),
        ("GET", "/_facts/alpha?wait_for_ms=100", ""),
        // a deadline bounds the wait as well
        (
            "POST",
            "/_facts/alpha/_ask",
            r#"{"question":"Who governs Liberia?","timeout_ms":100}"#,
        ),
    ] {
        let t0 = Instant::now();
        let (s, b, h) = send(&app, method, uri, body).await;
        assert_eq!(s, StatusCode::SERVICE_UNAVAILABLE, "{method} {uri}: {b}");
        assert_eq!(error_type(&b), "ENRICHMENT_PENDING", "{b}");
        assert_eq!(b["error"]["retryable"], true, "{b}");
        assert_eq!(h.get("retry-after").map(|v| v.to_str().unwrap()), Some("1"));
        assert!(
            b.get("status").is_some_and(|s| s == 503) && b.get("answer").is_none(),
            "{b}"
        );
        assert!(
            t0.elapsed() < Duration::from_secs(5),
            "answered at the wait, not after it"
        );
    }

    // without a wait the read waits for the rebuild, as before (off by default)
    let app2 = app.clone();
    let waiting =
        tokio::spawn(async move { send(&app2, "POST", "/_facts/alpha/_lookup", LOOKUP).await });
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert!(
        !waiting.is_finished(),
        "a read without wait_for_ms waits for the projection"
    );
    drop(held);
    let (s, b, _) = waiting.await.expect("join");
    assert_eq!(
        (s, b["status"].as_str()),
        (StatusCode::OK, Some("VALUE")),
        "{b}"
    );
    // the rebuild started by the refused reads went on and registered the projection
    assert!(state.facts.get("alpha").is_some());
    let (s, b, _) = send(&app, "POST", "/_facts/alpha/_lookup", lookup).await;
    assert_eq!(
        (s, b["status"].as_str()),
        (StatusCode::OK, Some("VALUE")),
        "{b}"
    );
}

/// A missing or broken model directory is `503 MODEL_UNAVAILABLE` with `retryable: false`, and it is sticky: the
/// directory is not read again until the node restarts, so the answer does not change when the files do.
#[tokio::test]
async fn model_unavailable_is_sticky_and_not_retryable() {
    let dir = tempfile::tempdir().expect("tempdir");
    let model_dir = tempfile::tempdir().expect("model dir");
    let mut state = node(dir.path(), |_| {});
    state.facts = Arc::new(FactsService::new(
        Some(model_dir.path().to_path_buf()),
        "int8-guard",
    ));
    let app = build_es_compat_router(state.clone());
    load(&app, "alpha").await;
    let (s, b, _) = send(&app, "GET", "/_facts", "").await;
    assert_eq!(
        (s, b["model"]["state"].as_str()),
        (StatusCode::OK, Some("not_loaded")),
        "{b}"
    );

    let ask = r#"{"question":"Who governs Liberia?"}"#;
    let (s1, b1, h1) = send(&app, "POST", "/_facts/alpha/_ask", ask).await;
    assert_eq!(s1, StatusCode::SERVICE_UNAVAILABLE, "{b1}");
    assert_eq!(error_type(&b1), "MODEL_UNAVAILABLE", "{b1}");
    assert_eq!(b1["error"]["retryable"], false, "{b1}");
    assert!(
        h1.get("retry-after").is_none(),
        "a sticky error must not invite a retry"
    );
    assert!(
        b1["error"]["reason"]
            .as_str()
            .unwrap()
            .contains("config.json"),
        "{b1}"
    );

    // the directory changes; the error does not (a reload would now fail on the tokenizer instead)
    std::fs::write(
        model_dir.path().join("config.json"),
        r#"{"tokenizer":{"sha256":"0"}}"#,
    )
    .unwrap();
    let (s2, b2, _) = send(&app, "POST", "/_facts/alpha/_ask", ask).await;
    assert_eq!((s2, &b2), (s1, &b1), "sticky until restart");
    let (_, b, _) = send(&app, "GET", "/_facts", "").await;
    assert_eq!(b["model"]["state"], "unavailable", "{b}");

    // no model configured at all: the same code
    let dir2 = tempfile::tempdir().expect("tempdir");
    let mut bare = node(dir2.path(), |_| {});
    bare.facts = Arc::new(FactsService::new(None, "f32"));
    let app = build_es_compat_router(bare);
    load(&app, "alpha").await;
    let (s, b, _) = send(&app, "POST", "/_facts/alpha/_ask", ask).await;
    assert_eq!(
        (s, error_type(&b)),
        (StatusCode::SERVICE_UNAVAILABLE, "MODEL_UNAVAILABLE"),
        "{b}"
    );
    assert!(
        b["error"]["reason"]
            .as_str()
            .unwrap()
            .contains("XERJ_FACTS_MODEL_DIR"),
        "{b}"
    );
}

/// With every ask slot busy past the deadline, `_ask` answers `504 ASK_DEADLINE` (stage `queue`) instead of waiting.
/// The slot is taken through the node's own gate, as a running ask holds it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_full_ask_queue_is_504_ask_deadline() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut state = node(dir.path(), |_| {});
    let gate = AskGate::new(Some(1), Some(Duration::from_secs(30)));
    state.facts = Arc::new(FactsService::new(None, "f32").with_ask_gate(gate.clone()));
    let app = build_es_compat_router(state);
    load(&app, "alpha").await;

    let busy = tokio::spawn(async move {
        gate.run(Instant::now(), None, || {
            std::thread::sleep(Duration::from_millis(1500))
        })
        .await
    });
    tokio::time::sleep(Duration::from_millis(100)).await;
    let (_, b, _) = send(&app, "GET", "/_facts", "").await;
    assert_eq!(
        b["ask_limits"],
        json!({"max_concurrent": 1, "running": 1, "max_queued": null, "waiting": 0, "deadline_ms": 30000,
               "config_error": null})
    );

    let t0 = Instant::now();
    let (s, b, h) = send(
        &app,
        "POST",
        "/_facts/alpha/_ask",
        r#"{"question":"Who governs Liberia?","timeout_ms":200}"#,
    )
    .await;
    assert_eq!(s, StatusCode::GATEWAY_TIMEOUT, "{b}");
    assert_eq!(error_type(&b), "ASK_DEADLINE", "{b}");
    assert_eq!(
        (
            b["error"]["stage"].as_str(),
            b["error"]["deadline_ms"].as_u64()
        ),
        (Some("queue"), Some(200)),
        "{b}"
    );
    assert_eq!(b["error"]["retryable"], true);
    assert!(h.get("retry-after").is_some());
    assert!(
        t0.elapsed() < Duration::from_millis(1200),
        "answered at the deadline"
    );

    busy.await.expect("join").expect("the busy slot ran");
    // the slot is free again: the ask now reaches the (missing) model
    let (s, b, _) = send(
        &app,
        "POST",
        "/_facts/alpha/_ask",
        r#"{"question":"Who governs Liberia?","timeout_ms":5000}"#,
    )
    .await;
    assert_eq!(
        (s, error_type(&b)),
        (StatusCode::SERVICE_UNAVAILABLE, "MODEL_UNAVAILABLE"),
        "{b}"
    );
}

/// With every slot busy and the line full (`max_queued` = 1 here, one ask already waiting), `_ask` is refused at once
/// with `429 ASK_QUEUE_FULL` (`retryable`, `Retry-After`) instead of waiting for an answer that would come too late; the
/// waiting ask keeps its place and runs when the slot frees.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_full_line_is_429_ask_queue_full_at_once() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut state = node(dir.path(), |_| {});
    let gate = AskGate::new(Some(1), None)
        .with_max_queued(Some(1))
        .expect("valid limits");
    state.facts = Arc::new(FactsService::new(None, "f32").with_ask_gate(gate.clone()));
    let app = build_es_compat_router(state);
    load(&app, "alpha").await;

    let g1 = gate.clone();
    let busy = tokio::spawn(async move {
        g1.run(Instant::now(), None, || {
            std::thread::sleep(Duration::from_millis(1500))
        })
        .await
    });
    tokio::time::sleep(Duration::from_millis(100)).await;
    let app2 = app.clone();
    let waiting = tokio::spawn(async move {
        send(
            &app2,
            "POST",
            "/_facts/alpha/_ask",
            r#"{"question":"Who governs Liberia?"}"#,
        )
        .await
    });
    tokio::time::sleep(Duration::from_millis(200)).await;
    let (_, b, _) = send(&app, "GET", "/_facts", "").await;
    assert_eq!(
        (
            &b["ask_limits"]["running"],
            &b["ask_limits"]["waiting"],
            &b["ask_limits"]["max_queued"]
        ),
        (&json!(1), &json!(1), &json!(1)),
        "{b}"
    );

    let t0 = Instant::now();
    let (s, b, h) = send(
        &app,
        "POST",
        "/_facts/alpha/_ask",
        r#"{"question":"Who governs Liberia?"}"#,
    )
    .await;
    assert_eq!(
        (s, error_type(&b)),
        (StatusCode::TOO_MANY_REQUESTS, "ASK_QUEUE_FULL"),
        "{b}"
    );
    assert_eq!(
        (
            b["error"]["retryable"].as_bool(),
            b["error"]["waiting"].as_u64(),
            b["error"]["max_queued"].as_u64()
        ),
        (Some(true), Some(1), Some(1)),
        "{b}"
    );
    assert_eq!(h.get("retry-after").map(|v| v.to_str().unwrap()), Some("1"));
    assert!(b.get("answer").is_none(), "{b}");
    assert!(
        t0.elapsed() < Duration::from_millis(1000),
        "refused at once, not after the busy slot"
    );

    busy.await.expect("join").expect("the busy slot ran");
    // the waiting ask got the slot and reached the (missing) model
    let (s, b, _) = waiting.await.expect("join");
    assert_eq!(
        (s, error_type(&b)),
        (StatusCode::SERVICE_UNAVAILABLE, "MODEL_UNAVAILABLE"),
        "{b}"
    );
    let (_, b, _) = send(&app, "GET", "/_facts", "").await;
    assert_eq!(
        (&b["ask_limits"]["running"], &b["ask_limits"]["waiting"]),
        (&json!(0), &json!(0)),
        "{b}"
    );
}

/// Requests outside the limits are refused before any work, as JSON a client can read.
#[tokio::test]
async fn requests_outside_the_limits_are_refused_as_json() {
    let dir = tempfile::tempdir().expect("tempdir");
    let app = build_es_compat_router(node(dir.path(), |c| c.limits.max_actions_per_bulk = 3));
    load(&app, "alpha").await;
    let ask = |extra: Value| {
        let mut b = json!({"question": "Who governs Liberia?"});
        for (k, v) in extra.as_object().unwrap() {
            b[k] = v.clone();
        }
        b.to_string()
    };
    for (body, want) in [
        (
            ask(json!({"question": "x".repeat(4097)})),
            "question is 4097 bytes",
        ),
        (
            ask(json!({"item_id": "i".repeat(257)})),
            "item_id is 257 bytes",
        ),
        (ask(json!({"linker": "fuzzy"})), "linker [fuzzy]"),
        (ask(json!({"manifest": "fuzzy"})), "manifest [fuzzy]"),
        (ask(json!({"verify": "fuzzy"})), "verify [fuzzy]"),
        ("{not json".to_string(), "request body"),
        (r#"{"revision": 1}"#.to_string(), "question"),
    ] {
        let (s, b, _) = send(&app, "POST", "/_facts/alpha/_ask", &body).await;
        assert_eq!(
            (s, error_type(&b)),
            (StatusCode::BAD_REQUEST, "facts_bad_request"),
            "{b}"
        );
        assert!(
            b["error"]["reason"].as_str().unwrap().contains(want),
            "{want}: {b}"
        );
    }
    // a question of exactly the limit passes the check (and then needs the model)
    let (s, b, _) = send(
        &app,
        "POST",
        "/_facts/alpha/_ask",
        &ask(json!({"question": "x".repeat(4096)})),
    )
    .await;
    assert_eq!(
        (s, error_type(&b)),
        (StatusCode::SERVICE_UNAVAILABLE, "MODEL_UNAVAILABLE"),
        "{b}"
    );
    // the opt-in shared-store values pass the check too (and then need the model)
    for extra in [
        json!({"linker": "casefold_v3"}),
        json!({"manifest": "relation_aware"}),
        json!({"linker": "casefold_v3", "manifest": "reference"}),
        json!({"verify": "served"}),
        json!({"verify": "bindings"}),
        json!({"verify": "bindings_path"}),
        json!({"verify": "bindings_all"}),
    ] {
        let (s, b, _) = send(&app, "POST", "/_facts/alpha/_ask", &ask(extra.clone())).await;
        assert_eq!(
            (s, error_type(&b)),
            (StatusCode::SERVICE_UNAVAILABLE, "MODEL_UNAVAILABLE"),
            "{extra}: {b}"
        );
    }

    // bodies over 64 KiB, before they are parsed
    let huge = ask(json!({"pad": "p".repeat(70_000)}));
    for uri in ["/_facts/alpha/_ask", "/_facts/alpha/_lookup"] {
        let (s, b, _) = send(&app, "POST", uri, &huge).await;
        assert_eq!(
            (s, error_type(&b)),
            (StatusCode::PAYLOAD_TOO_LARGE, "facts_request_too_large"),
            "{uri}: {b}"
        );
    }
}

/// A write batch over `limits.max_actions_per_bulk` (3 here) is refused before anything is written.
#[tokio::test]
async fn a_write_batch_over_the_bulk_cap_writes_nothing() {
    let dir = tempfile::tempdir().expect("tempdir");
    let app = build_es_compat_router(node(dir.path(), |c| c.limits.max_actions_per_bulk = 3));
    let mut big = world();
    big["edits"] = json!([{"op": "withdraw_document", "doc_id": "d1", "version": "v1"},
                          {"op": "withdraw_document", "doc_id": "d1", "version": "v1"}]);
    for (uri, body) in [
        ("/_facts/beta/_world", big.to_string()),
        (
            "/_facts/beta/_bulk",
            json!({"documents": big["base_documents"], "facts": big["base_facts"],
                                      "edits": big["edits"]})
            .to_string(),
        ),
    ] {
        let (s, b, _) = send(&app, "POST", uri, &body).await;
        assert_eq!(
            (s, error_type(&b)),
            (StatusCode::PAYLOAD_TOO_LARGE, "facts_batch_too_large"),
            "{uri}: {b}"
        );
        assert!(
            b["error"]["reason"].as_str().unwrap().contains("4 entries"),
            "{b}"
        );
    }
    let (s, b, _) = send(&app, "GET", "/_facts/beta", "").await;
    assert_eq!(s, StatusCode::NOT_FOUND, "nothing was created: {b}");
}

/// A batch the store refuses is a 400 with the store's own (reference) message, and nothing of it is committed: the
/// namespace stays at its revision and keeps answering.
#[tokio::test]
async fn a_contract_error_is_a_400_with_the_reference_message_and_commits_nothing() {
    let dir = tempfile::tempdir().expect("tempdir");
    let app = build_es_compat_router(node(dir.path(), |_| {}));
    load(&app, "alpha").await;
    let bad = json!({
        "documents": [{"doc_id": "d2", "version": "v1", "text": "Ana Bel is 170 cm tall."}],
        "facts": [{"fact_id": "f2", "subject": "Ana Bel", "relation": "height", "kind": "number", "value": "170",
                   "unit": "cm", "doc_id": "nope", "version": "v1", "start": 11, "end": 14, "quote": "170"}]
    });
    let (s, b, _) = send(&app, "POST", "/_facts/alpha/_bulk", &bad.to_string()).await;
    assert_eq!(
        (s, error_type(&b)),
        (StatusCode::BAD_REQUEST, "facts_contract_error"),
        "{b}"
    );
    assert!(
        b["error"]["reason"]
            .as_str()
            .unwrap()
            .contains("Fact must reference an available source version"),
        "the store's reference message: {b}"
    );
    let (s, b, _) = send(&app, "GET", "/_facts/alpha", "").await;
    assert_eq!(s, StatusCode::OK, "{b}");
    assert_eq!(
        (b["revision"].as_u64(), b["published_rev"].as_u64()),
        (Some(2), Some(2)),
        "{b}"
    );
    let (s, b, _) = send(&app, "POST", "/_facts/alpha/_lookup", LOOKUP).await;
    assert_eq!(
        (s, b["status"].as_str()),
        (StatusCode::OK, Some("VALUE")),
        "{b}"
    );
}

/// `GET /_facts/{ns}/{fact_id}`: the record and its visibility at a committed revision; a correction hides the old
/// value at the head but not before it; unknown ids, unknown namespaces and uncommitted revisions are clear errors.
#[tokio::test]
async fn a_fact_reads_with_its_visibility_at_a_revision() {
    let dir = tempfile::tempdir().expect("tempdir");
    let app = build_es_compat_router(node(dir.path(), |_| {}));
    let mut w = world();
    w["base_documents"]
        .as_array_mut()
        .unwrap()
        .push(json!({"doc_id": "d2", "version": "v1", "text": "Ana Bel is 170 cm tall."}));
    w["base_facts"].as_array_mut().unwrap().push(json!({"fact_id": "f2", "subject": "Ana Bel", "relation": "height",
        "kind": "number", "value": "170", "unit": "cm", "doc_id": "d2", "version": "v1", "start": 11, "end": 14,
        "quote": "170"}));
    w["edits"] = json!([{"op": "correct", "doc": {"doc_id": "d2", "version": "v2", "text": "Ana Bel is 171 cm tall."},
                         "fact": {"fact_id": "f3", "subject": "Ana Bel", "relation": "height", "kind": "number",
                                  "value": "171", "unit": "cm", "doc_id": "d2", "version": "v2", "start": 11, "end": 14,
                                  "quote": "171"},
                         "supersedes": "f2"}]);
    let (s, b, _) = send(&app, "POST", "/_facts/alpha/_world", &w.to_string()).await;
    assert_eq!(s, StatusCode::OK, "{b}");
    let head = b["committed"]["to_rev"].as_u64().unwrap();

    let (s, f2, _) = send(&app, "GET", "/_facts/alpha/f2", "").await;
    assert_eq!(s, StatusCode::OK, "{f2}");
    let (_, f3, _) = send(&app, "GET", "/_facts/alpha/f3", "").await;
    assert_eq!(
        (f2["visible"].as_bool(), f3["visible"].as_bool()),
        (Some(false), Some(true)),
        "{f2} {f3}"
    );
    assert_eq!(
        (
            &f3["namespace"],
            &f3["fact_id"],
            &f3["snapshot"]["revision"]
        ),
        (&json!("alpha"), &json!("f3"), &json!(head))
    );
    let before = f3["added_rev"].as_u64().unwrap() - 1;
    let (_, f2b, _) = send(
        &app,
        "GET",
        &format!("/_facts/alpha/f2?revision={before}"),
        "",
    )
    .await;
    let (_, f3b, _) = send(
        &app,
        "GET",
        &format!("/_facts/alpha/f3?revision={before}"),
        "",
    )
    .await;
    assert_eq!(
        (f2b["visible"].as_bool(), f3b["visible"].as_bool()),
        (Some(true), Some(false)),
        "{f2b} {f3b}"
    );

    let ahead = format!("/_facts/alpha/f1?revision={}", head + 1);
    for (uri, status, code) in [
        (
            "/_facts/alpha/nope",
            StatusCode::NOT_FOUND,
            "fact_not_found",
        ),
        (
            ahead.as_str(),
            StatusCode::CONFLICT,
            "REVISION_NOT_COMMITTED",
        ),
        (
            "/_facts/alpha/f1?revision=x",
            StatusCode::BAD_REQUEST,
            "facts_bad_request",
        ),
    ] {
        let (s, b, _) = send(&app, "GET", uri, "").await;
        assert_eq!((s, error_type(&b)), (status, code), "{uri}: {b}");
    }
    let (s, b, _) = send(&app, "GET", "/_facts/nobody/f1", "").await;
    assert_eq!(s, StatusCode::NOT_FOUND, "{b}");
    // a withdrawn fact stays readable, hidden from that revision on
    let (s, b, _) = send(&app, "DELETE", "/_facts/alpha/f1", "").await;
    assert_eq!(s, StatusCode::OK, "{b}");
    let (_, f1, _) = send(&app, "GET", "/_facts/alpha/f1", "").await;
    let (_, f1old, _) = send(
        &app,
        "GET",
        &format!("/_facts/alpha/f1?revision={head}"),
        "",
    )
    .await;
    assert_eq!(
        (f1["visible"].as_bool(), f1old["visible"].as_bool()),
        (Some(false), Some(true)),
        "{f1} {f1old}"
    );
}

/// A write reports the node's real WAL policy (design §2.4), not a fixed sentence.
#[tokio::test]
async fn a_write_reports_the_nodes_wal_policy() {
    for (sync, want) in [
        (xerj_common::config::WalSync::Async, "wal_sync=async"),
        (
            xerj_common::config::WalSync::Batched,
            "wal_sync=batched(100ms)",
        ),
        (xerj_common::config::WalSync::Sync, "wal_sync=sync"),
    ] {
        let dir = tempfile::tempdir().expect("tempdir");
        let app = build_es_compat_router(node(dir.path(), |c| {
            c.storage.wal_sync = sync;
            c.storage.wal_batch_ms = 100;
        }));
        let (s, b, _) = send(&app, "POST", "/_facts/alpha/_world", &world().to_string()).await;
        assert_eq!(s, StatusCode::OK, "{b}");
        assert_eq!(b["durability"], want, "{b}");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// With the real compiler (skipped unless the build server's env.sh is sourced)
// ─────────────────────────────────────────────────────────────────────────────

fn golden_rows(n: usize) -> Option<(std::path::PathBuf, Vec<Value>)> {
    let model_dir = std::env::var_os("XERJ_FACTS_MODEL_DIR")?;
    let golden = std::env::var_os("FIX_GOLDEN_DIR")?;
    let model = std::env::var("XERJ_E2_GOLDEN_MODEL").unwrap_or_else(|_| "x68_dir_s11".into());
    let path = std::path::Path::new(&golden).join(format!("{model}.direction_test.jsonl"));
    let text = std::fs::read_to_string(path).ok()?;
    let rows = text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .take(n)
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    Some((std::path::PathBuf::from(model_dir), rows))
}

/// The worlds of the direction test set, by world id.
fn direction_worlds() -> Option<std::collections::HashMap<String, Value>> {
    let root = std::env::var_os("XERJ_R_ROOT")?;
    let p = std::path::Path::new(&root).join("runs/world/direction_v1/model_test.jsonl");
    let text = std::fs::read_to_string(p).ok()?;
    Some(
        text.lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| serde_json::from_str::<Value>(l).unwrap())
            .map(|w| (w["world_id"].as_str().unwrap_or("").to_string(), w))
            .collect(),
    )
}

/// The gate changes nothing about an answer: through `_ask` with a concurrency limit and a deadline the compiler
/// gives the golden decision's status, answer, evidence and program; a 1 ms deadline is a 504, never an UNKNOWN.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn with_the_real_compiler_the_gate_changes_no_answer() {
    let (Some((model_dir, rows)), Some(worlds)) = (golden_rows(3), direction_worlds()) else {
        eprintln!("skipped: XERJ_FACTS_MODEL_DIR, FIX_GOLDEN_DIR and XERJ_R_ROOT are not all set");
        return;
    };
    let precision = std::env::var("XERJ_FACTS_PRECISION").unwrap_or_else(|_| "int8-guard".into());
    let dir = tempfile::tempdir().expect("tempdir");
    let mut state = node(dir.path(), |_| {});
    state.facts = Arc::new(
        FactsService::new(Some(model_dir), &precision)
            .with_ask_gate(AskGate::new(Some(1), Some(Duration::from_secs(120)))),
    );
    let app = build_es_compat_router(state);
    let mut checked = 0;
    for (i, r) in rows.iter().enumerate() {
        let item = r["item_id"].as_str().unwrap();
        let Some((wid, w)) = worlds.iter().find(|(_, w)| {
            w["items"]
                .as_array()
                .is_some_and(|its| its.iter().any(|it| it["item_id"] == item))
        }) else {
            panic!("no world holds {item}");
        };
        let phase = w["items"]
            .as_array()
            .unwrap()
            .iter()
            .find(|it| it["item_id"] == item)
            .unwrap()["phase"]
            .as_str()
            .unwrap_or("after")
            .to_string();
        let ns = format!("gold-{i}");
        let (s, b, _) = send(
            &app,
            "POST",
            &format!("/_facts/{ns}/_world"),
            &w.to_string(),
        )
        .await;
        assert_eq!(s, StatusCode::OK, "{wid}: {b}");
        let rev = if phase == "before" {
            b["base_rev"].clone()
        } else {
            b["committed"]["to_rev"].clone()
        };
        let ask = json!({"question": r["question"], "revision": rev, "item_id": item}).to_string();
        let (s, got, _) = send(&app, "POST", &format!("/_facts/{ns}/_ask"), &ask).await;
        assert_eq!(s, StatusCode::OK, "{item}: {got}");
        let g = &r["final"];
        let program: Vec<Value> = g["program"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| {
                json!([
                    p["op"],
                    p["pointer"],
                    if p["direction"] == "in" {
                        json!("in")
                    } else {
                        Value::Null
                    }
                ])
            })
            .collect();
        assert_eq!(got["status"], g["status"], "{item}");
        assert_eq!(got["answer"], g["answer"], "{item}");
        assert_eq!(got["evidence_ids"], g["evidence_ids"], "{item}");
        assert_eq!(got["program"], json!(program), "{item}");
        checked += 1;
        if i == 0 {
            let short = json!({"question": r["question"], "revision": rev, "item_id": item, "timeout_ms": 1}).to_string();
            let (s, b, _) = send(&app, "POST", &format!("/_facts/{ns}/_ask"), &short).await;
            assert_eq!(
                (s, error_type(&b)),
                (StatusCode::GATEWAY_TIMEOUT, "ASK_DEADLINE"),
                "{b}"
            );
            assert!(
                b.get("answer").is_none() && b.get("status") == Some(&json!(504)),
                "{b}"
            );
        }
    }
    eprintln!(
        "real compiler through the gate: {checked} / {} golden answers identical",
        rows.len()
    );
    assert_eq!(checked, rows.len());
}

/// A stop ends the compiler before its next decision and leaves no answer, not even an UNKNOWN; without a stop the same
/// question gives the golden program. The observer sets the stop flag at the first decision, as the gate does when the
/// deadline passes or the client goes away during it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn with_the_real_compiler_a_stop_ends_the_ask_after_the_decision_in_progress() {
    use std::sync::atomic::{AtomicBool, Ordering};
    use xerj_facts::xerj_infer::episode::DecisionEvent;
    use xerj_facts::xerj_infer::Session;
    use xerj_facts::xerj_serve::engine::{answer_stoppable, LinkerMode, Stopped};

    let (Some((model_dir, rows)), Some(worlds)) = (golden_rows(12), direction_worlds()) else {
        eprintln!("skipped: XERJ_FACTS_MODEL_DIR, FIX_GOLDEN_DIR and XERJ_R_ROOT are not all set");
        return;
    };
    // a golden question whose program takes at least two decisions
    let Some(r) = rows.iter().find(|r| {
        r["final"]["program"]
            .as_array()
            .is_some_and(|p| p.len() >= 2)
    }) else {
        panic!(
            "no golden row with two decisions among the first {}",
            rows.len()
        );
    };
    let item = r["item_id"].as_str().unwrap();
    let question = r["question"].as_str().unwrap();
    let w = worlds
        .values()
        .find(|w| {
            w["items"]
                .as_array()
                .is_some_and(|its| its.iter().any(|it| it["item_id"] == item))
        })
        .expect("the golden row's world");
    let phase_before = w["items"]
        .as_array()
        .unwrap()
        .iter()
        .find(|it| it["item_id"] == item)
        .unwrap()["phase"]
        == "before";
    let precision = std::env::var("XERJ_FACTS_PRECISION").unwrap_or_else(|_| "int8-guard".into());
    let dir = tempfile::tempdir().expect("tempdir");
    let mut state = node(dir.path(), |_| {});
    state.facts = Arc::new(FactsService::new(Some(model_dir), &precision));
    let app = build_es_compat_router(state.clone());
    let (s, b, _) = send(&app, "POST", "/_facts/stop/_world", &w.to_string()).await;
    assert_eq!(s, StatusCode::OK, "{b}");
    let rev = if phase_before {
        b["base_rev"].as_u64().unwrap()
    } else {
        b["committed"]["to_rev"].as_u64().unwrap()
    } as u32;
    let slot = state.facts.model().expect("the model loads");
    let n = state.facts.get("stop").expect("projection");

    let run = |stop_at: Option<usize>| {
        let mem = n.mem.read().expect("projection lock");
        let snap = mem.snapshot_at(rev);
        let stop = Arc::new(AtomicBool::new(false));
        let mut decisions = 0usize;
        let mut observe = |_: &Session, _: &DecisionEvent| {
            decisions += 1;
            if Some(decisions) == stop_at {
                stop.store(true, Ordering::Release);
            }
        };
        let out = answer_stoppable(
            &slot,
            &mem,
            snap,
            item,
            question,
            LinkerMode::Exact,
            false,
            Some(&mut observe),
            Some(stop.clone()),
        );
        (out, decisions)
    };

    let (out, decisions) = run(Some(1));
    assert_eq!(
        out.as_ref().map(|_| ()),
        Err(&Stopped { decisions: 1 }),
        "stopped after the decision in progress"
    );
    assert_eq!(decisions, 1, "no decision after the stop");

    let (out, decisions) = run(None);
    let out = out.expect("an unset flag changes nothing");
    let golden: Vec<Value> = r["final"]["program"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| {
            json!([
                p["op"],
                p["pointer"],
                if p["direction"] == "in" {
                    json!("in")
                } else {
                    Value::Null
                }
            ])
        })
        .collect();
    assert_eq!(out["program"], json!(golden), "{item}");
    assert_eq!(out["status"], r["final"]["status"], "{item}");
    assert_eq!(out["answer"], r["final"]["answer"], "{item}");
    assert_eq!(decisions, golden.len(), "one decision per program step");
    eprintln!(
        "stop test on {item}: stopped after 1 of {} decisions; unstopped run identical to golden",
        golden.len()
    );
}
