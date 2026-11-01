//! A facts namespace is a security boundary (design `XERJ-R_engine_integration_design_2026-10-26.md` §5.4, PR-4).
//!
//! Namespace `ns` keeps its facts in `.xerj-memory-{ns}-facts`, inside the reserved `.xerj-memory-` prefix, so the doors
//! into it are `/_facts/{ns}/…` *and* every generic door into an index: the ES-compat index routes, the body-named
//! targets of `_bulk` / `_msearch` / `_mget`, the native router and wildcard reads. A boundary that closed `/_facts` only
//! would be a boundary in name. This file walks every door with every kind of caller: the superuser, a key granted the
//! namespace, a key granted another namespace, a key granted only the namespace's memory and edges indices, an unscoped
//! key and a share-link guest. It also pins the in-handler check (the routes stay closed on a router mounted without the
//! authorization middleware), the filtered listing, the identical 403 for an absent and a forbidden namespace, the
//! reserved `-facts` suffix of `/_memory` and `/_graph`, and the audit entries.
//!
//! The node has no compiler model (`FactsService::new(None, …)`): `_ask` that passes authorization answers
//! `503 MODEL_UNAVAILABLE`, which is how these tests tell "authorized" from "refused" without loading a model.

#![cfg(feature = "facts")]

use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use serde_json::{json, Value};
use std::sync::Arc;
use tower::ServiceExt;
use xerj_api::router::{build_es_compat_router, build_native_router};
use xerj_api::state::AppState;
use xerj_common::{config::Config, metrics::Metrics};
use xerj_engine::Engine;

const ADMIN_KEY: &str = "admin-key-for-the-facts-boundary-test";

fn admin() -> String {
    format!("ApiKey {ADMIN_KEY}")
}

/// An auth-enabled node over `dir` with no compiler model and nothing read from the process environment (the build
/// server exports `XERJ_FACTS_MODEL_DIR`).
fn state_over(dir: &std::path::Path) -> AppState {
    let mut config = Config::default();
    config.server.data_dir = dir.to_str().unwrap().to_string();
    config.auth.enabled = true;
    config.auth.admin_api_key = ADMIN_KEY.to_string();
    let metrics = Metrics::new().expect("metrics");
    let engine = Engine::new(config.clone()).expect("engine");
    let mut state = AppState::new(config, engine, metrics);
    state.facts = Arc::new(xerj_facts::FactsService::new(None, "f32"));
    state
}

async fn send(
    app: &axum::Router,
    method: &str,
    uri: &str,
    auth: &str,
    body: &str,
) -> (StatusCode, Value) {
    let mut builder = Request::builder()
        .method(method)
        .uri(uri)
        .header("content-type", "application/json");
    if !auth.is_empty() {
        builder = builder.header("authorization", auth);
    }
    let req = builder.body(Body::from(body.to_string())).expect("request");
    let resp = app.clone().oneshot(req).await.expect("response");
    let status = resp.status();
    let bytes = resp.into_body().collect().await.expect("body").to_bytes();
    let json: Value = serde_json::from_slice(&bytes)
        .unwrap_or_else(|_| Value::String(String::from_utf8_lossy(&bytes).into_owned()));
    (status, json)
}

async fn mint(app: &axum::Router, body: Value) -> String {
    let (status, resp) = send(
        app,
        "POST",
        "/_security/api_key",
        &admin(),
        &body.to_string(),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "minting a key: {resp}");
    format!("ApiKey {}", resp["encoded"].as_str().expect("encoded key"))
}

/// A key holding `privileges` on exactly `indices`.
async fn key_for(app: &axum::Router, name: &str, indices: &[&str], privileges: &[&str]) -> String {
    mint(
        app,
        json!({"name": name, "role_descriptors": {name: {"indices": [{"names": indices, "privileges": privileges}]}}}),
    )
    .await
}

/// One memory world with one document and one fact: `{subject} is governed by {value}.`
fn world(subject: &str, value: &str) -> Value {
    let text = format!("{subject} is governed by {value}.");
    json!({
        "world_id": subject.to_lowercase(),
        "base_documents": [{"doc_id": "d1", "version": "v1", "text": text}],
        "base_facts": [{"fact_id": "f1", "subject": subject, "relation": "head_of_government", "kind": "text",
                        "value": value, "doc_id": "d1", "version": "v1", "start": 0, "end": text.len(),
                        "quote": text}],
        "edits": []
    })
}

/// alice and bob, each with one fact; bob's value is the secret nothing of alice's may ever show.
const SECRET: &str = "Zorbo Vex";

async fn seed(app: &axum::Router) {
    for (ns, w) in [
        ("alice", world("Liberia", "George Weah")),
        ("bob", world("Atlantis", SECRET)),
    ] {
        let (status, body) = send(
            app,
            "POST",
            &format!("/_facts/{ns}/_world"),
            &admin(),
            &w.to_string(),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "seeding {ns}: {body}");
        assert_eq!(body["committed"]["to_rev"], 2, "{body}");
    }
}

/// Every `/_facts` route of namespace `ns`, with a body each would accept.
fn facts_routes(ns: &str) -> Vec<(&'static str, String, String)> {
    let w = world("Mu", "Ka").to_string();
    vec![
        ("GET", format!("/_facts/{ns}"), String::new()),
        ("GET", format!("/_facts/{ns}/f1"), String::new()),
        (
            "POST",
            format!("/_facts/{ns}/_lookup"),
            r#"{"subject":"Atlantis","relation":"head_of_government"}"#.into(),
        ),
        (
            "POST",
            format!("/_facts/{ns}/_ask"),
            r#"{"question":"Who governs Atlantis?"}"#.into(),
        ),
        ("POST", format!("/_facts/{ns}/_world"), w),
        (
            "POST",
            format!("/_facts/{ns}/_bulk"),
            r#"{"documents":[],"facts":[],"edits":[]}"#.into(),
        ),
        ("DELETE", format!("/_facts/{ns}/f1"), String::new()),
        ("DELETE", format!("/_facts/{ns}"), String::new()),
    ]
}

fn is_security_403(status: StatusCode, body: &Value) -> bool {
    status == StatusCode::FORBIDDEN && body.to_string().contains("security_exception")
}

// ─────────────────────────────────────────────────────────────────────────────

/// Every door into bob's facts, walked with a key granted alice's facts index (read and write).
#[tokio::test]
async fn a_scoped_caller_cannot_reach_another_namespace_through_any_door() {
    let dir = tempfile::tempdir().expect("tempdir");
    let state = state_over(dir.path());
    let app = build_es_compat_router(state.clone());
    let native = build_native_router(state);
    seed(&app).await;
    let alice = key_for(
        &app,
        "alice-agent",
        &[".xerj-memory-alice-facts"],
        &["read", "write"],
    )
    .await;

    // 1. the facts API: every route
    for (method, uri, body) in facts_routes("bob") {
        let (status, resp) = send(&app, method, &uri, &alice, &body).await;
        assert!(
            is_security_403(status, &resp),
            "{method} {uri}: {status} {resp}"
        );
        assert!(
            !resp.to_string().contains(SECRET),
            "{method} {uri} leaked: {resp}"
        );
    }

    // 2. the raw index: read, write, refresh, destroy, percent-encoded
    for (method, uri, body) in [
        (
            "POST",
            "/.xerj-memory-bob-facts/_search",
            r#"{"query":{"match_all":{}}}"#,
        ),
        ("GET", "/.xerj-memory-bob-facts/_doc/f:f1", ""),
        (
            "POST",
            "/.xerj-memory-bob-facts/_doc/forged",
            r#"{"rt":"epoch","from_rev":3,"to_rev":3,"ops":"[]"}"#,
        ),
        ("POST", "/.xerj-memory-bob-facts/_refresh", ""),
        ("GET", "/.xerj-memory-bob-facts/_mapping", ""),
        ("DELETE", "/.xerj-memory-bob-facts", ""),
        ("POST", "/%2Exerj-memory-bob-facts/_search", "{}"),
    ] {
        let (status, resp) = send(&app, method, uri, &alice, body).await;
        assert_eq!(status, StatusCode::FORBIDDEN, "{method} {uri}: {resp}");
    }

    // 3. body-named targets
    for (uri, body) in [
        ("/_bulk", "{\"index\":{\"_index\":\".xerj-memory-bob-facts\",\"_id\":\"e:9999999999\"}}\n{\"rt\":\"epoch\"}\n"),
        ("/_bulk", "{\"delete\":{\"_index\":\".xerj-memory-bob-facts\",\"_id\":\"f:f1\"}}\n"),
        ("/_msearch", "{\"index\":\".xerj-memory-bob-facts\"}\n{\"query\":{\"match_all\":{}}}\n"),
        ("/_mget", "{\"docs\":[{\"_index\":\".xerj-memory-bob-facts\",\"_id\":\"f:f1\"}]}"),
    ] {
        let (status, resp) = send(&app, "POST", uri, &alice, body).await;
        assert_eq!(status, StatusCode::FORBIDDEN, "body-named {uri}: {resp}");
    }

    // 4. wildcard reads are answered over alice's own indices only
    for uri in [
        "/.xerj-memory-*/_search",
        "/*/_search",
        "/_all/_search",
        "/.xerj-memory-bob*/_search",
    ] {
        let (status, resp) = send(&app, "POST", uri, &alice, "{}").await;
        assert_eq!(status, StatusCode::OK, "{uri}: {resp}");
        assert!(
            !resp.to_string().contains(SECRET),
            "{uri} leaked bob: {resp}"
        );
    }
    let (status, resp) = send(
        &app,
        "POST",
        "/.xerj-memory-alice-facts/_search",
        &alice,
        "{}",
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{resp}");

    // 5. the native router
    for (method, uri) in [
        ("POST", "/v1/indices/.xerj-memory-bob-facts/search"),
        ("GET", "/v1/indices/.xerj-memory-bob-facts"),
        ("DELETE", "/v1/indices/.xerj-memory-bob-facts"),
    ] {
        let (status, resp) = send(&native, method, uri, &alice, r#"{"query":{}}"#).await;
        assert_eq!(
            status,
            StatusCode::FORBIDDEN,
            "native {method} {uri}: {resp}"
        );
    }

    // bob's facts are intact
    let (status, resp) = send(
        &app,
        "POST",
        "/_facts/bob/_lookup",
        &admin(),
        r#"{"subject":"Atlantis","relation":"head_of_government"}"#,
    )
    .await;
    assert_eq!(
        (status, resp["status"].as_str()),
        (StatusCode::OK, Some("VALUE")),
        "{resp}"
    );
}

/// A namespace that does not exist is refused exactly like one that does, so status codes cannot map the node.
#[tokio::test]
async fn an_absent_and_a_forbidden_namespace_get_the_same_403() {
    let dir = tempfile::tempdir().expect("tempdir");
    let app = build_es_compat_router(state_over(dir.path()));
    seed(&app).await;
    let alice = key_for(
        &app,
        "alice-agent",
        &[".xerj-memory-alice-facts"],
        &["read", "write"],
    )
    .await;
    for (method, uri, body) in facts_routes("bob") {
        let absent = uri.replace("/bob", "/carol");
        let (s1, b1) = send(&app, method, &uri, &alice, &body).await;
        let (s2, b2) = send(&app, method, &absent, &alice, &body).await;
        assert_eq!(s1, StatusCode::FORBIDDEN, "{method} {uri}: {b1}");
        assert_eq!(s2, s1, "{method} {absent}: {b2}");
        assert_eq!(
            b2.to_string().replace("carol", "bob"),
            b1.to_string(),
            "{method}: the bodies differ beyond the name the caller sent"
        );
    }
}

/// Inside its grant a key keeps the use it was given, and each op needs its own privilege: `_ask` and `_lookup` read,
/// `_world` / `_bulk` / a fact's withdrawal write, dropping the namespace manages. Grants on the namespace's memory and
/// edges indices do not reach its facts.
#[tokio::test]
async fn each_op_needs_its_own_privilege_on_the_facts_index() {
    let dir = tempfile::tempdir().expect("tempdir");
    let app = build_es_compat_router(state_over(dir.path()));
    seed(&app).await;
    let reader = key_for(&app, "reader", &[".xerj-memory-alice-facts"], &["read"]).await;
    let writer = key_for(
        &app,
        "writer",
        &[".xerj-memory-alice-facts"],
        &["read", "write"],
    )
    .await;
    let manager = key_for(&app, "manager", &[".xerj-memory-alice-facts"], &["all"]).await;
    let memory_only = key_for(
        &app,
        "memory-only",
        &[".xerj-memory-alice", ".xerj-memory-alice-edges"],
        &["all"],
    )
    .await;

    let lookup = r#"{"subject":"Liberia","relation":"head_of_government"}"#;
    let (s, b) = send(&app, "POST", "/_facts/alice/_lookup", &reader, lookup).await;
    assert_eq!(
        (s, b["status"].as_str()),
        (StatusCode::OK, Some("VALUE")),
        "{b}"
    );
    let (s, b) = send(&app, "GET", "/_facts/alice", &reader, "").await;
    assert_eq!(s, StatusCode::OK, "{b}");
    // authorized, then refused for the missing model (not for the key)
    let (s, b) = send(
        &app,
        "POST",
        "/_facts/alice/_ask",
        &reader,
        r#"{"question":"Who governs Liberia?"}"#,
    )
    .await;
    assert_eq!(
        (s, b["error"]["type"].as_str()),
        (StatusCode::SERVICE_UNAVAILABLE, Some("MODEL_UNAVAILABLE")),
        "{b}"
    );
    let (s, b) = send(&app, "GET", "/_facts/alice/f1", &reader, "").await;
    assert_eq!(
        (s, b["value"].as_str(), b["visible"].as_bool()),
        (StatusCode::OK, Some("George Weah"), Some(true)),
        "{b}"
    );
    // the reads are the first four routes; the rest need more than read
    for (method, uri, body) in facts_routes("alice").into_iter().skip(4) {
        let (s, b) = send(&app, method, &uri, &reader, &body).await;
        assert!(is_security_403(s, &b), "reader {method} {uri}: {s} {b}");
    }

    let (s, b) = send(&app, "POST", "/_facts/alice/_bulk", &writer, &json!({
        "documents": [{"doc_id": "d2", "version": "v1", "text": "Ana Bel is 170 cm tall."}],
        "facts": [{"fact_id": "f2", "subject": "Ana Bel", "relation": "height", "kind": "number", "value": "170",
                   "unit": "cm", "doc_id": "d2", "version": "v1", "start": 11, "end": 14, "quote": "170"}]
    }).to_string()).await;
    assert_eq!(
        (s, &b["committed"]),
        (StatusCode::OK, &json!({"from_rev": 3, "to_rev": 4})),
        "{b}"
    );
    let (s, b) = send(&app, "DELETE", "/_facts/alice/f2", &writer, "").await;
    assert_eq!(
        (s, &b["committed"]),
        (StatusCode::OK, &json!({"from_rev": 5, "to_rev": 5})),
        "{b}"
    );
    let (s, b) = send(&app, "DELETE", "/_facts/alice", &writer, "").await;
    assert!(
        is_security_403(s, &b),
        "dropping a namespace needs manage: {s} {b}"
    );

    for (method, uri, body) in facts_routes("alice") {
        let (s, b) = send(&app, method, &uri, &memory_only, &body).await;
        assert!(
            is_security_403(s, &b),
            "memory-only {method} {uri}: {s} {b}"
        );
    }

    let (s, b) = send(&app, "DELETE", "/_facts/alice", &manager, "").await;
    assert_eq!(s, StatusCode::OK, "{b}");
    let (s, b) = send(&app, "GET", "/_facts/alice", &manager, "").await;
    assert_eq!(s, StatusCode::NOT_FOUND, "{b}");
}

/// An unscoped key (minted without role descriptors) holds nothing in the reserved prefix, and a share-link guest
/// gets no `/_facts` route at all, not even the listing.
#[tokio::test]
async fn unscoped_keys_and_share_link_guests_are_refused() {
    let dir = tempfile::tempdir().expect("tempdir");
    let app = build_es_compat_router(state_over(dir.path()));
    seed(&app).await;
    let unscoped = mint(&app, json!({"name": "legacy"})).await;
    for (method, uri, body) in facts_routes("alice") {
        let (s, b) = send(&app, method, &uri, &unscoped, &body).await;
        assert!(is_security_403(s, &b), "unscoped {method} {uri}: {s} {b}");
    }
    let (s, b) = send(&app, "GET", "/_facts", &unscoped, "").await;
    assert_eq!(s, StatusCode::OK, "{b}");
    assert_eq!(
        b["namespaces"],
        json!([]),
        "an unscoped key reads no facts namespace: {b}"
    );

    // a guest of an ordinary shared index
    let (s, _) = send(&app, "PUT", "/casefile", &admin(), "").await;
    assert_eq!(s, StatusCode::OK);
    let (s, share) = send(&app, "POST", "/_share", &admin(), r#"{"index":"casefile"}"#).await;
    assert_eq!(s, StatusCode::OK, "{share}");
    let (s, claimed) = send(
        &app,
        "POST",
        "/_share/claim",
        "",
        &json!({"id": share["share_id"], "passcode": share["passcode"]}).to_string(),
    )
    .await;
    assert_eq!(s, StatusCode::OK, "{claimed}");
    let guest = format!("ApiKey {}", claimed["api_key"].as_str().expect("api_key"));
    let mut routes = facts_routes("alice");
    routes.push(("GET", "/_facts".to_string(), String::new()));
    routes.push((
        "POST",
        "/.xerj-memory-alice-facts/_search".to_string(),
        "{}".to_string(),
    ));
    for (method, uri, body) in routes {
        let (s, b) = send(&app, method, &uri, &guest, &body).await;
        assert_eq!(s, StatusCode::FORBIDDEN, "guest {method} {uri}: {b}");
    }
    // and a share cannot name a facts index
    let (s, b) = send(
        &app,
        "POST",
        "/_share",
        &admin(),
        r#"{"index":".xerj-memory-alice-facts"}"#,
    )
    .await;
    assert_eq!(s, StatusCode::BAD_REQUEST, "{b}");
}

/// `GET /_facts` lists the namespaces the caller may read, nothing else (design §5.1 "read, filtered"). The listing
/// used to return every namespace in memory, with its revision, to any authenticated key.
#[tokio::test]
async fn the_listing_shows_only_what_the_caller_may_read() {
    let dir = tempfile::tempdir().expect("tempdir");
    let app = build_es_compat_router(state_over(dir.path()));
    seed(&app).await;
    let reader = key_for(&app, "reader", &[".xerj-memory-alice-facts"], &["read"]).await;
    let names = |b: &Value| -> Vec<String> {
        b["namespaces"]
            .as_array()
            .unwrap()
            .iter()
            .map(|n| n["namespace"].as_str().unwrap().to_string())
            .collect()
    };
    let (s, b) = send(&app, "GET", "/_facts", &reader, "").await;
    assert_eq!(s, StatusCode::OK, "{b}");
    assert_eq!(names(&b), ["alice"], "{b}");
    assert!(
        !b.to_string().contains("bob"),
        "the listing leaked bob: {b}"
    );
    assert_eq!(b["namespaces"][0]["state"], "ready");
    assert_eq!(b["namespaces"][0]["published_rev"], 2);
    let (s, b) = send(&app, "GET", "/_facts", &admin(), "").await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(names(&b), ["alice", "bob"], "{b}");
}

/// The handlers authorize on their own, so the routes stay closed on a router mounted without the authorization
/// middleware (the `memory_api` discipline: "a handler that authorizes must be safe on any router it is mounted on").
#[tokio::test]
async fn the_handlers_authorize_without_the_middleware() {
    let dir = tempfile::tempdir().expect("tempdir");
    let state = state_over(dir.path());
    let app = build_es_compat_router(state.clone());
    seed(&app).await;
    let alice = key_for(
        &app,
        "alice-agent",
        &[".xerj-memory-alice-facts"],
        &["read", "write"],
    )
    .await;
    let bare = xerj_api::facts_api::routes().with_state(state);
    for (method, uri, body) in facts_routes("bob") {
        let (s, b) = send(&bare, method, &uri, &alice, &body).await;
        assert!(
            is_security_403(s, &b),
            "bare router {method} {uri}: {s} {b}"
        );
    }
    let (s, b) = send(&bare, "GET", "/_facts", &alice, "").await;
    assert_eq!(s, StatusCode::OK);
    assert!(!b.to_string().contains("bob"), "{b}");
    let (s, b) = send(
        &bare,
        "POST",
        "/_facts/alice/_lookup",
        &alice,
        r#"{"subject":"Liberia","relation":"head_of_government"}"#,
    )
    .await;
    assert_eq!(
        (s, b["status"].as_str()),
        (StatusCode::OK, Some("VALUE")),
        "{b}"
    );
    // no credential at all: refused, never treated as the superuser
    let (s, b) = send(
        &bare,
        "POST",
        "/_facts/alice/_lookup",
        "",
        r#"{"subject":"Liberia","relation":"head_of_government"}"#,
    )
    .await;
    assert_eq!(s, StatusCode::FORBIDDEN, "{b}");
}

/// `-facts` is reserved in memory-namespace and brain names: `/_memory/alice-facts` would write memories into
/// facts namespace `alice`'s index, and brain `alice-facts` would read it as its nodes.
#[tokio::test]
async fn memory_and_graph_refuse_the_facts_suffix() {
    let dir = tempfile::tempdir().expect("tempdir");
    let app = build_es_compat_router(state_over(dir.path()));
    seed(&app).await;
    let reason = "namespace suffix '-facts' is reserved for facts indices";
    for (method, uri, body) in [
        ("POST", "/_memory/alice-facts", r#"{"text":"a memory"}"#),
        (
            "POST",
            "/_memory/alice-facts/_recall",
            r#"{"query":"George"}"#,
        ),
        ("GET", "/_memory/alice-facts", ""),
        (
            "POST",
            "/_graph/alice-facts/link",
            r#"{"src":"a","type":"mentions","dst":"b"}"#,
        ),
        ("GET", "/_graph/alice-facts/overview", ""),
        (
            "POST",
            "/_facts/alice-facts/_world",
            r#"{"base_documents":[]}"#,
        ),
    ] {
        let (s, b) = send(&app, method, uri, &admin(), body).await;
        assert_eq!(s, StatusCode::BAD_REQUEST, "{method} {uri}: {b}");
        assert!(b.to_string().contains(reason), "{method} {uri}: {b}");
    }
    // the facts index is not listed as a memory namespace
    let (s, b) = send(&app, "GET", "/_memory", &admin(), "").await;
    assert_eq!(s, StatusCode::OK, "{b}");
    assert!(!b.to_string().contains("alice-facts"), "{b}");
    // and alice's facts are untouched
    let (s, b) = send(
        &app,
        "POST",
        "/.xerj-memory-alice-facts/_count",
        &admin(),
        "{}",
    )
    .await;
    assert_eq!(
        (s, b["count"].as_u64()),
        (StatusCode::OK, Some(3)),
        "1 doc + 1 fact + 1 epoch: {b}"
    );
}

async fn audit_entries(native: &axum::Router) -> Vec<Value> {
    let (status, body) = send(native, "GET", "/_audit/_search?size=1000", &admin(), "").await;
    assert_eq!(status, StatusCode::OK, "audit read: {body}");
    body["entries"].as_array().cloned().unwrap_or_default()
}

/// Writes are audited against the namespace's facts index whatever their outcome, a refused question is audited as
/// a refused read, and an answered question is not written to the audit ring at all (it is not a write).
#[tokio::test]
async fn facts_writes_and_refusals_are_audited_against_the_facts_index() {
    let dir = tempfile::tempdir().expect("tempdir");
    let state = state_over(dir.path());
    let app = build_es_compat_router(state.clone());
    let native = build_native_router(state);
    seed(&app).await;
    let writer = key_for(
        &app,
        "writer",
        &[".xerj-memory-alice-facts"],
        &["read", "write"],
    )
    .await;
    let (s, _) = send(&app, "DELETE", "/_facts/alice/f1", &writer, "").await;
    assert_eq!(s, StatusCode::OK);
    let (s, _) = send(
        &app,
        "POST",
        "/_facts/alice/_ask",
        &writer,
        r#"{"question":"Who governs Liberia?"}"#,
    )
    .await;
    assert_eq!(s, StatusCode::SERVICE_UNAVAILABLE, "authorized (no model)");
    let (s, _) = send(
        &app,
        "POST",
        "/_facts/bob/_ask",
        &writer,
        r#"{"question":"Who governs Atlantis?"}"#,
    )
    .await;
    assert_eq!(s, StatusCode::FORBIDDEN);
    let (s, _) = send(&app, "POST", "/_facts/bob/_bulk", &writer, "{}").await;
    assert_eq!(s, StatusCode::FORBIDDEN);

    let entries = audit_entries(&native).await;
    let find = |op: &str, resource: &str, outcome: &str| {
        entries
            .iter()
            .any(|e| e["op"] == op && e["resource"] == resource && e["outcome"] == outcome)
    };
    assert!(
        find("facts.world", ".xerj-memory-alice-facts", "ok"),
        "{entries:#?}"
    );
    assert!(
        find("facts.withdraw", ".xerj-memory-alice-facts", "ok"),
        "{entries:#?}"
    );
    assert!(
        find("facts.ask", ".xerj-memory-bob-facts", "denied"),
        "{entries:#?}"
    );
    assert!(
        find("facts.bulk", ".xerj-memory-bob-facts", "denied"),
        "{entries:#?}"
    );
    assert!(
        !entries
            .iter()
            .any(|e| e["op"] == "facts.ask" && e["resource"] == ".xerj-memory-alice-facts"),
        "an allowed question is not a write: {entries:#?}"
    );
    assert!(
        !entries
            .iter()
            .any(|e| e["resource"] == "_facts" || e["op"] == "facts.post"),
        "no entry may name the endpoint instead of the namespace: {entries:#?}"
    );
}
