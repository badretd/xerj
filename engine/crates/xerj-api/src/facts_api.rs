//! SPIKE `/_facts` (feature `facts`; local branch `spike/facts-68m`, never pushed). The XERJ-R facts store, exact executor
//! and learned compiler served by the engine. Design: `research/product/XERJ-R_engine_integration_design_2026-10-26.md`,
//! step 0. The logic lives in `xerj-facts`; this module only composes the ES-compat handlers, as `memory_api.rs` does.
//!
//! ```text
//! GET    /_facts                         readable namespaces, residency and evictions, compiler status, `_ask` limits
//! GET    /_facts/{ns}[?fingerprint=true] namespace status (revision, published, epochs, records, cleanup_pending)
//! DELETE /_facts/{ns}                    drop the namespace (index + projection)
//! POST   /_facts/{ns}/_world             one memory-world-v1 batch (documents, facts, edits) -> {committed, base_rev}
//! POST   /_facts/{ns}/_bulk              {documents?, facts?, edits?}: the same as one batch
//! GET    /_facts/{ns}/{fact_id}[?revision] one fact and its visibility at a committed revision, no model
//! DELETE /_facts/{ns}/{fact_id}          withdraw_fact (history stays readable)
//! POST   /_facts/{ns}/_lookup            {subject, relation, scope?, revision?} -> VALUE | UNKNOWN | CONFLICT, no model
//! POST   /_facts/{ns}/_ask               {question, revision?, item_id?, linker?, manifest?, verify?, explain?} -> answer
//!                                       + evidence + program
//! ```
//!
//! Commit protocol (one writer, design §2.3):
//! 2. apply the batch to the projection (contract checks first);
//! 3. bulk-index its `doc` / `fact` records;
//! 4. write its `epoch` documents (op headers, at most `EPOCH_OPS` each), one bulk call per epoch in revision order. The batch
//!    commits when its last epoch is written: every epoch names the batch's last revision, and recovery keeps a batch only
//!    with all of its epochs. A failed epoch write answers 503 `COMMIT_OUTCOME_UNKNOWN`;
//! 5. re-index the repair-owned fields of the older records the batch retired, hid or withdrew (partial updates). The batch
//!    is committed whatever happens here: a failed repair is reported (`repair.deferred`) and finished by the next rebuild;
//! 6. publish the batch's last revision, then acknowledge.
//!
//! Any failure before the batch commits drops the projection, and the next access rebuilds it. Recovery (§2.5) is one
//! consistent walk over the index (`fold_match_sources`) that collects the epochs and the record payloads, a replay of the
//! committed chain through the reference write API, then the cleanup the replay asks for, before any write takes the next
//! revision: delete the records and epochs of the batch that never committed (orphans) and finish the step-5 repairs a kill
//! cut off. A cleanup the engine refuses (e.g. under its memory breaker) leaves the replayed projection serving reads at the
//! committed head (`cleanup_pending` in the status; design §2.3: repairing until the next write); every write retries the
//! cleanup first and answers 503 `NAMESPACE_RECOVERING` while it is refused. Residue of more than one batch past the
//! committed head is no crash's (a lost write, or a walk that missed a committed epoch): it is never deleted, and the
//! namespace answers 503 `NAMESPACE_RECOVERING` to reads and writes until someone looks at it. Operational failures are
//! HTTP errors and never a factual UNKNOWN.
//!
//! Projections are derived state. `XERJ_FACTS_MAX_RESIDENT_NAMESPACES=N` (unset by default: no eviction) keeps at most N in
//! memory and evicts the least recently used; the next access rebuilds an evicted namespace through the same recovery.
//!
//! Every step has a crash point (`xerj_facts::failpoint`, feature `facts-failpoints`, test-only and off by default):
//! `commit.after_apply`, `commit.records` (before each records bulk call), `commit.after_records`, `commit.epoch` (before
//! each epoch write), `commit.after_epochs`, `commit.repair` (before each repair bulk call), `commit.after_repair`,
//! `commit.after_publish`; in a rebuild `recover.after_replay`, `recover.delete` (before each delete bulk call),
//! `recover.after_delete`, `recover.repair` (before each repair bulk call) and `recover.after_repair`.
//!
//! Hardening (design §5.3–§5.4, PR-4 / PR-6):
//! - **Authorization in the handler.** Every handler checks the caller's privilege on the namespace's facts index
//!   (`authz::facts_index`) itself, before anything that could reveal whether the namespace exists, as `memory_api`
//!   does: the routes stay closed on a router mounted without `authz::authz_middleware`. `GET /_facts` lists only the
//!   namespaces the caller may read.
//! - **Single node.** With `cluster.enabled = true` every route answers `501 FACTS_CLUSTER_UNSUPPORTED`.
//! - **Sizes.** `_ask` and `_lookup` bodies are capped at [`MAX_READ_BODY_BYTES`], questions at [`MAX_QUESTION_BYTES`];
//!   a `_world` / `_bulk` batch holds at most `limits.max_actions_per_bulk` entries (documents + facts + edits).
//! - **A lagging projection is a 503, never an UNKNOWN.** A namespace whose projection is not in memory (first access
//!   after a restart, dropped after a failed write, or evicted) is rebuilt from the index, with the recovery above, by
//!   a background task. A read that passes `wait_for_ms` (or runs under a deadline) waits at most that long, then gets
//!   `503 ENRICHMENT_PENDING` with `retryable: true` and `Retry-After`; the rebuild goes on and the retry finds it.
//!   Without either it waits for the rebuild, as before.
//! - **`_ask` admission** (`xerj_facts::AskGate`): an optional bound on concurrent asks, an optional bound on the asks
//!   waiting for a slot and an optional deadline (`XERJ_FACTS_MAX_CONCURRENT_ASKS`, `XERJ_FACTS_MAX_QUEUED_ASKS`,
//!   `XERJ_FACTS_ASK_DEADLINE_MS`; a request may shorten the deadline with `timeout_ms`). A full line is
//!   `429 ASK_QUEUE_FULL` at once; a missed deadline is `504 ASK_DEADLINE`, and the compiler of an ask nobody waits for
//!   any more (deadline passed, or the client went away) stops before its next decision and frees its slot. All are off
//!   unless configured.
//! - **Model errors are sticky.** A missing or broken model directory is `503 MODEL_UNAVAILABLE` with
//!   `retryable: false` until the node restarts.

/// The facts compiler preload at server start-up (`XERJ_FACTS_PRELOAD`, on when a model directory is set): see
/// [`xerj_facts::start_early_preload`]. Re-exported for `xerj-server`, which reaches the facts crate through this one.
pub use xerj_facts::start_early_preload;

use crate::auth::Principal;
use crate::authz;
use crate::es_compat;
use crate::extract::OptionalJson;
use crate::state::AppState;
use axum::extract::rejection::BytesRejection;
use axum::extract::{DefaultBodyLimit, Path, Query, State};
use axum::http::{header, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use bytes::Bytes;
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use xerj_engine::rbac::Privilege;
use xerj_facts::failpoint::{self, Action};
use xerj_facts::{
    facts_index, facts_mapping, now_ms, parse_epoch, Applied, AskReq, Batch, Cleanup, FactsError,
    GateError, LookupReq, Namespace, Payloads, Recovered,
};
use xerj_query::ast::QueryNode;

/// `doc` / `fact` documents (or repairs, or deletes) per bulk call (the engine caps a bulk at
/// `limits.max_actions_per_bulk`, 50,000 by default).
const RECORDS_PER_BULK: usize = 10_000;

/// A recovery cleanup of at least this many deletes and repairs, or a step 5 of at least this many repairs, flushes the
/// facts index first (see `projection_locked`).
const CLEANUP_FLUSH_MIN: usize = 1_000;

fn records_per_bulk() -> usize {
    failpoint::records_per_bulk(RECORDS_PER_BULK)
}

/// Largest request body `_ask` and `_lookup` accept: a question and a few options. The write routes keep the node's
/// `limits.max_body_bytes`.
pub const MAX_READ_BODY_BYTES: usize = 64 * 1024;
/// Longest question `_ask` accepts, in bytes. The compiler reads the whole question at every decision, so its cost grows
/// with the length; the reference questions are a few hundred bytes.
pub const MAX_QUESTION_BYTES: usize = 4096;
/// Longest `item_id` (the manifest salt) `_ask` accepts, in bytes.
pub const MAX_ITEM_ID_BYTES: usize = 256;
/// The `linker` values `_ask` accepts: `xerj_serve::engine::LinkerMode`, plus the opt-in `casefold` pre-link
/// (`xerj_facts::casefold_link`) and its third version `casefold_v3` (`xerj_facts::casefold_v3_link`). Any other value used
/// to run the exact linker silently; it is refused now.
const LINKERS: &[&str] = &[
    "exact",
    "tolerant",
    "tolerant_words",
    "casefold",
    "casefold_v3",
];
/// The `manifest` values `_ask` accepts: the reference manifests (default) or the opt-in `relation_aware` choice where a
/// manifest cap binds (`xerj_index::direction::RelPref`).
const MANIFESTS: &[&str] = &["reference", "relation_aware"];
/// The `verify` values `_ask` accepts (`xerj_facts::VerifyRule`); without one, `decline_v2` decides what is served. None
/// changes the program or the raw answer.
const VERIFY_RULES: &[&str] = &xerj_facts::VerifyRule::NAMES;
/// `Retry-After` of the retryable 503 / 504 answers, in seconds.
const RETRY_AFTER_SECS: u64 = 1;

pub fn routes() -> Router<AppState> {
    Router::new()
        .route("/_facts", get(list))
        .route("/_facts/:ns", get(status).delete(drop_namespace))
        .route("/_facts/:ns/_world", post(world))
        .route("/_facts/:ns/_bulk", post(bulk))
        // route-level limits win over the router's `DefaultBodyLimit::disable()` (as `/_share/claim` does)
        .route(
            "/_facts/:ns/_lookup",
            post(lookup).layer(DefaultBodyLimit::max(MAX_READ_BODY_BYTES)),
        )
        .route(
            "/_facts/:ns/_ask",
            post(ask).layer(DefaultBodyLimit::max(MAX_READ_BODY_BYTES)),
        )
        .route("/_facts/:ns/:fact_id", get(get_fact).delete(withdraw_fact))
}

fn err(status: StatusCode, code: &str, reason: impl Into<String>) -> Response {
    (
        status,
        Json(json!({"error": {"type": code, "reason": reason.into()}, "status": status.as_u16()})),
    )
        .into_response()
}

/// [`err`] with more fields in `error` (`retryable`, `stage`, ...). A retryable answer also carries `Retry-After`.
fn err_with(status: StatusCode, code: &str, reason: impl Into<String>, extra: Value) -> Response {
    let mut error = json!({"type": code, "reason": reason.into()});
    let mut retryable = false;
    if let (Some(e), Value::Object(more)) = (error.as_object_mut(), extra) {
        for (k, v) in more {
            if k == "retryable" {
                retryable = v.as_bool().unwrap_or(false);
            }
            e.insert(k, v);
        }
    }
    let mut resp = (
        status,
        Json(json!({"error": error, "status": status.as_u16()})),
    )
        .into_response();
    if retryable {
        resp.headers_mut()
            .insert(header::RETRY_AFTER, HeaderValue::from(RETRY_AFTER_SECS));
    }
    resp
}

/// `503 ENRICHMENT_PENDING`: the namespace's projection is behind its index (being rebuilt) and did not catch up within
/// the caller's wait. Never an UNKNOWN: nothing was read.
fn enrichment_pending(ns: &str, waited: Duration) -> Response {
    err_with(
        StatusCode::SERVICE_UNAVAILABLE,
        "ENRICHMENT_PENDING",
        format!(
            "the projection of facts namespace [{ns}] is being rebuilt from its index and was not ready within {} ms; \
             nothing was read: retry",
            waited.as_millis()
        ),
        json!({"retryable": true, "waited_ms": waited.as_millis() as u64}),
    )
}

fn facts_err(e: FactsError) -> Response {
    match e {
        FactsError::BadRequest(m) => err(StatusCode::BAD_REQUEST, "facts_bad_request", m),
        FactsError::NotCommitted(m) => err(StatusCode::CONFLICT, "REVISION_NOT_COMMITTED", m),
        // sticky until restart (`FactsService::model`): retrying cannot help, so the client is told not to
        FactsError::ModelUnavailable(m) => err_with(
            StatusCode::SERVICE_UNAVAILABLE,
            "MODEL_UNAVAILABLE",
            m,
            json!({"retryable": false}),
        ),
        // only an ask the gate stopped waiting for is stopped, so its client already has its 504; kept consistent
        FactsError::Stopped { decisions } => err_with(
            StatusCode::GATEWAY_TIMEOUT,
            "ASK_DEADLINE",
            format!(
                "the ask was stopped after {decisions} compiler decisions; nothing was answered"
            ),
            json!({"retryable": true, "stage": "model"}),
        ),
        FactsError::Internal(m) => {
            err(StatusCode::INTERNAL_SERVER_ERROR, "facts_internal_error", m)
        }
    }
}

async fn drain_json(resp: Response) -> (StatusCode, Value) {
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap_or_default();
    (
        status,
        serde_json::from_slice::<Value>(&bytes).unwrap_or(Value::Null),
    )
}

/// The namespace rules of `/_memory` (`memory_api::validate_namespace`), which in a build with this module also reserve
/// the `-facts` suffix: `.xerj-memory-{ns}-facts` must never be another namespace's facts index.
fn check_ns(ns: &str) -> Result<(), Response> {
    crate::memory_api::validate_namespace(ns)
        .map_err(|m| err(StatusCode::BAD_REQUEST, "invalid_namespace", m))
}

/// v1 is single-node (design §5.4): a node configured for a cluster refuses every `/_facts` route.
fn cluster_gate(state: &AppState) -> Result<(), Response> {
    if state.config.cluster.enabled {
        return Err(err(
            StatusCode::NOT_IMPLEMENTED,
            "FACTS_CLUSTER_UNSUPPORTED",
            "the facts store runs on a single node in this version, and this node is configured for a cluster \
             (cluster.enabled = true)",
        ));
    }
    Ok(())
}

/// What every namespace handler checks first, in this order: the name's shape (it reveals nothing), the caller's
/// `privilege` on the namespace's facts index (before anything that could tell whether the namespace exists, so an
/// absent and a forbidden namespace get the same 403), then the single-node rule.
fn preflight(
    state: &AppState,
    principal: &Principal,
    ns: &str,
    privilege: Privilege,
) -> Result<(), Response> {
    check_ns(ns)?;
    authz::authorize_index(principal, &authz::facts_index(ns), privilege)?;
    cluster_gate(state)
}

/// A JSON request body, or the error a client can act on (axum's own rejections are plain text).
fn parse_body<T: serde::de::DeserializeOwned>(
    body: Result<Bytes, BytesRejection>,
) -> Result<T, Response> {
    let bytes = body.map_err(|rej| {
        let status = rej.status();
        let code = if status == StatusCode::PAYLOAD_TOO_LARGE {
            "facts_request_too_large"
        } else {
            "facts_bad_request"
        };
        err(status, code, rej.body_text())
    })?;
    serde_json::from_slice(&bytes).map_err(|e| {
        err(
            StatusCode::BAD_REQUEST,
            "facts_bad_request",
            format!("request body: {e}"),
        )
    })
}

/// `wait_for_ms` as a duration (`None` = wait as long as it takes).
fn wait_of(ms: Option<u64>) -> Option<Duration> {
    ms.map(Duration::from_millis)
}

fn index_exists(state: &AppState, index: &str) -> bool {
    state.engine.get_index(index).is_ok()
}

async fn ensure_index(state: &AppState, index: &str) -> Result<(), Response> {
    if index_exists(state, index) {
        return Ok(());
    }
    let resp = es_compat::create_index(
        State(state.clone()),
        Path(index.to_string()),
        OptionalJson(Some(facts_mapping())),
    )
    .await
    .into_response();
    let (status, body) = drain_json(resp).await;
    if status.is_success() || index_exists(state, index) {
        return Ok(());
    }
    Err(err(status, "facts_index_create_failed", body.to_string()))
}

/// One bulk call into the facts index.
enum Write<'a> {
    /// whole `(_id, _source)` documents
    Index(&'a [(String, Value)]),
    /// `(_id, partial source)`: merged into the existing documents (the repair-owned fields)
    Update(&'a [(String, Value)]),
    Delete(&'a [String]),
}

/// A failed bulk call: the HTTP status to report and the first failing item (or the response).
struct BulkError {
    status: StatusCode,
    reason: String,
}

impl BulkError {
    fn into_response(self, code: &str) -> Response {
        err(self.status, code, self.reason)
    }
}

/// One bulk call into the facts index; an item-level error fails the whole call.
async fn bulk_write(state: &AppState, index: &str, w: Write<'_>) -> Result<(), BulkError> {
    let mut nd = String::new();
    let mut line = |v: Value| {
        nd.push_str(&v.to_string());
        nd.push('\n');
    };
    match w {
        Write::Index(docs) => {
            for (id, src) in docs {
                line(json!({"index": {"_id": id}}));
                line(src.clone());
            }
        }
        Write::Update(docs) => {
            for (id, fields) in docs {
                line(json!({"update": {"_id": id}}));
                line(json!({"doc": fields}));
            }
        }
        Write::Delete(ids) => {
            for id in ids {
                line(json!({"delete": {"_id": id}}));
            }
        }
    }
    if nd.is_empty() {
        return Ok(());
    }
    let resp = es_compat::bulk_ops(
        State(state.clone()),
        Path(index.to_string()),
        Query(HashMap::new()),
        bytes::Bytes::from(nd),
    )
    .await
    .into_response();
    let (status, body) = drain_json(resp).await;
    if !status.is_success() || body.get("errors").and_then(Value::as_bool).unwrap_or(true) {
        let first = body
            .get("items")
            .and_then(Value::as_array)
            .and_then(|items| {
                items.iter().find(|i| {
                    i.as_object()
                        .is_some_and(|o| o.values().any(|a| a.get("error").is_some()))
                })
            })
            .cloned()
            .unwrap_or(body);
        return Err(BulkError {
            status: if status.is_success() {
                StatusCode::SERVICE_UNAVAILABLE
            } else {
                status
            },
            reason: first.to_string(),
        });
    }
    Ok(())
}

/// The namespace's projection, rebuilt from the committed epochs on first access. `None` = no facts index; an index that
/// exists but is unavailable is an error response, never `None`.
/// The caller holds the writer lock, so a rebuild never interleaves with a write. Authorization already happened in the
/// middleware (`Target::Facts`); the rebuild reads the namespace's own index directly.
async fn projection_locked(state: &AppState, ns: &str) -> Result<Option<Arc<Namespace>>, Response> {
    if let Some(n) = state.facts.get(ns) {
        return Ok(Some(n));
    }
    let idx = match state.engine.get_index(&facts_index(ns)) {
        Ok(idx) => idx,
        Err(e) => {
            // Only an index the engine does not know is "no facts namespace". An index that exists on disk but did not
            // open (engine issue #206), or a node whose cluster state blocked the boot, is unavailable: answer with the
            // engine's own error (503 with the open error). A 404 there would tell the client the namespace holds no data.
            let e = xerj_common::XerjError::from(e);
            if matches!(e, xerj_common::XerjError::IndexNotFound { .. }) {
                return Ok(None);
            }
            return Err(crate::error::ApiError::new(e).into_response());
        }
    };
    // design §2.5: one consistent walk over every live document collects the epoch markers and the record payloads
    let t0 = Instant::now();
    let mut epochs = Vec::new();
    let mut payloads = Payloads::default();
    let mut bad: Option<String> = None;
    idx.fold_match_sources(&QueryNode::MatchAll, &mut |src: &Value| {
        if bad.is_some() {
            return;
        }
        let r = match parse_epoch(src) {
            Ok(Some(e)) => {
                epochs.push(e);
                Ok(())
            }
            Ok(None) => payloads.add_source(src),
            Err(m) => Err(m),
        };
        if let Err(m) = r {
            bad = Some(m);
        }
    })
    .await;
    if let Some(m) = bad {
        return Err(err(
            StatusCode::INTERNAL_SERVER_ERROR,
            "facts_corrupt_record",
            m,
        ));
    }
    epochs.sort_by_key(|e| (e.to, e.from));
    let walk_ms = t0.elapsed().as_millis() as u64;
    let rec: Recovered =
        tokio::task::spawn_blocking(move || Namespace::recover(&epochs, &payloads))
            .await
            .map_err(|e| {
                err(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "facts_internal_error",
                    e.to_string(),
                )
            })?
            .map_err(|e| err(StatusCode::INTERNAL_SERVER_ERROR, "facts_replay_failed", e))?;
    let replay_ms = t0.elapsed().as_millis() as u64 - walk_ms;
    if let Some(why) = &rec.blocked {
        // not the residue of one interrupted batch: deleting it could destroy committed writes (Namespace::recover)
        tracing::error!(
            namespace = ns,
            published = rec.ns.published.load(std::sync::atomic::Ordering::SeqCst),
            records = rec.unexplained.len(),
            first = %rec.unexplained.first().map(String::as_str).unwrap_or(""),
            "facts recovery blocked: {why}"
        );
        return Err(err(
            StatusCode::SERVICE_UNAVAILABLE,
            "NAMESPACE_RECOVERING",
            format!("facts recovery of [{ns}] is blocked: {why}"),
        ));
    }
    // §2.5 steps 2 and 4, before a write takes the next revision: delete what never committed, finish the repairs a kill
    // cut off. Both are idempotent; an unfinished cleanup is redone by the next write or the next rebuild.
    let t_clean = Instant::now();
    recovery_point(ns, "recover.after_replay")?;
    let cleanup = Cleanup {
        orphans: rec.orphans,
        repairs: rec.repairs,
    };
    let (orphans, repairs) = (cleanup.orphans.len(), cleanup.repairs.len());
    let done = run_cleanup(state, ns, &cleanup).await;
    if !rec.unexplained.is_empty() {
        tracing::warn!(
            namespace = ns,
            records = rec.unexplained.len(),
            first = %rec.unexplained[0],
            "facts records outside the committed state at or below the head: no crash explains them; left in place"
        );
    }
    let n = Arc::new(rec.ns);
    let flush_ms = match done {
        Ok(flush_ms) => flush_ms,
        Err(reason) => {
            // design §2.3 / §2.5: the replayed projection holds exactly the committed chain, so it serves reads at the
            // committed head; the next write finishes the cleanup first (`finish_cleanup`)
            tracing::warn!(
                namespace = ns,
                orphans,
                repairs,
                reason = %reason,
                "facts recovery cleanup did not finish: reads are served at the committed head, writes wait for it"
            );
            if let Ok(mut p) = n.pending_cleanup.lock() {
                *p = Some(cleanup);
            }
            0
        }
    };
    tracing::info!(
        namespace = ns,
        published = n.published.load(std::sync::atomic::Ordering::SeqCst),
        epochs = n.epochs.load(std::sync::atomic::Ordering::SeqCst),
        ignored = rec.ignored,
        orphans,
        repairs,
        unexplained = rec.unexplained.len(),
        cleanup_pending = n
            .pending_cleanup
            .lock()
            .map(|p| p.is_some())
            .unwrap_or(true),
        walk_ms,
        replay_ms,
        flush_ms,
        cleanup_ms = t_clean.elapsed().as_millis() as u64,
        ms = t0.elapsed().as_millis() as u64,
        "facts projection rebuilt from the index"
    );
    insert_projection(state, ns, n.clone());
    Ok(Some(n))
}

/// Load a projection (writer lock held) and log the evictions `XERJ_FACTS_MAX_RESIDENT_NAMESPACES` asks for.
fn insert_projection(state: &AppState, ns: &str, n: Arc<Namespace>) {
    for old in state.facts.insert(ns, n) {
        tracing::info!(
            namespace = %old,
            loaded = ns,
            "facts projection evicted (XERJ_FACTS_MAX_RESIDENT_NAMESPACES); its next access rebuilds it from the index"
        );
    }
}

/// §2.5 steps 2 and 4: delete the orphans, then re-index the stale repair-owned fields. `Ok`: the flush time (ms);
/// `Err`: why it did not finish (nothing is lost: the cleanup is idempotent and derived again by any rebuild).
async fn run_cleanup(state: &AppState, ns: &str, c: &Cleanup) -> Result<u64, String> {
    let index = facts_index(ns);
    // A delete or an update of a document the engine still holds in its memtable walks every term of that memtable
    // (xerj-engine `MemTable::remove`), so a large cleanup right after a restart (the WAL replay put every record in the
    // memtable) is quadratic: 30,010 orphans took 64-84 s on the x86 build server, 3.1-3.6 s after a flush that moves
    // them into a segment, where each delete or update is a tombstone plus a write. The flush itself costs about 1.7 s for
    // 30,000 records, more than a few deletes or updates save, so only a large cleanup flushes first.
    let mut flush_ms = 0;
    if c.orphans.len() + c.repairs.len() >= CLEANUP_FLUSH_MIN {
        let t = Instant::now();
        let idx = state
            .engine
            .get_index(&index)
            .map_err(|e| format!("facts index: {e}"))?;
        idx.flush()
            .await
            .map_err(|e| format!("flush before the cleanup: {e}"))?;
        flush_ms = t.elapsed().as_millis() as u64;
    }
    for chunk in c.orphans.chunks(records_per_bulk()) {
        if cleanup_point("recover.delete")? {
            continue;
        }
        bulk_write(state, &index, Write::Delete(chunk))
            .await
            .map_err(|e| e.reason)?;
    }
    cleanup_point("recover.after_delete")?;
    for chunk in c.repairs.chunks(records_per_bulk()) {
        if cleanup_point("recover.repair")? {
            continue;
        }
        bulk_write(state, &index, Write::Update(chunk))
            .await
            .map_err(|e| e.reason)?;
    }
    cleanup_point("recover.after_repair")?;
    Ok(flush_ms)
}

/// Before a write takes the next revision (writer lock held): finish the cleanup its namespace's rebuild could not.
/// Still refused: 503 `NAMESPACE_RECOVERING`, and the projection keeps serving reads at the committed head.
async fn finish_cleanup(state: &AppState, ns: &str, n: &Namespace) -> Result<(), Response> {
    let pending = n.pending_cleanup.lock().ok().and_then(|mut p| p.take());
    let Some(c) = pending else {
        return Ok(());
    };
    let t = Instant::now();
    match run_cleanup(state, ns, &c).await {
        Ok(flush_ms) => {
            tracing::info!(
                namespace = ns,
                orphans = c.orphans.len(),
                repairs = c.repairs.len(),
                flush_ms,
                ms = t.elapsed().as_millis() as u64,
                "facts recovery cleanup finished before a write"
            );
            Ok(())
        }
        Err(reason) => {
            if let Ok(mut p) = n.pending_cleanup.lock() {
                *p = Some(c);
            }
            Err(err(
                StatusCode::SERVICE_UNAVAILABLE,
                "NAMESPACE_RECOVERING",
                format!(
                    "facts recovery of [{ns}] did not finish: {reason}; reads are served at the committed head, \
                     writes wait for the cleanup"
                ),
            ))
        }
    }
}

/// A crash point of the cleanup (a no-op without the feature `facts-failpoints`). `Err`: the cleanup fails;
/// `Ok(true)`: skip the one write the point guards.
fn cleanup_point(name: &str) -> Result<bool, String> {
    match failpoint::hit(name) {
        Action::Continue => Ok(false),
        Action::Skip => Ok(true),
        Action::Error => Err(format!("failpoint {name}")),
    }
}

/// A crash point of the rebuild before its cleanup (a no-op without the feature `facts-failpoints`). `Err`: the rebuild
/// fails; `Ok(true)`: skip the one write the point guards.
fn recovery_point(ns: &str, name: &str) -> Result<bool, Response> {
    match failpoint::hit(name) {
        Action::Continue => Ok(false),
        Action::Skip => Ok(true),
        Action::Error => Err(err(
            StatusCode::SERVICE_UNAVAILABLE,
            "NAMESPACE_RECOVERING",
            format!("facts recovery of [{ns}] did not finish: failpoint {name}"),
        )),
    }
}

/// The namespace's projection for a read, waiting at most `wait` for it (`None` = as long as it takes).
///
/// A projection that is not in memory is rebuilt by a task of its own, under the writer lock, so a rebuild is never
/// abandoned half-way when a caller stops waiting: the next request finds it done (design §2.4, "required projection
/// visible"). A caller whose wait runs out gets `503 ENRICHMENT_PENDING`, never an answer from a partial projection.
///
/// Detached work and the visibility rule (`xerj_engine::index_guard`): the task runs under the request's own rule
/// (`index_guard::current` + `scoped_opt`), and the namespace was authorized by the handler before this is called.
async fn projection_within(
    state: &AppState,
    ns: &str,
    wait: Option<Duration>,
) -> Result<Option<Arc<Namespace>>, Response> {
    if let Some(n) = state.facts.get(ns) {
        return Ok(Some(n));
    }
    let rule = xerj_engine::index_guard::current();
    let (st, name) = (state.clone(), ns.to_string());
    let rebuild = tokio::spawn(xerj_engine::index_guard::scoped_opt(rule, async move {
        let _w = st.facts.writer.lock().await;
        projection_locked(&st, &name).await
    }));
    let joined = match wait {
        None => rebuild.await,
        Some(d) => match tokio::time::timeout(d, rebuild).await {
            Ok(joined) => joined,
            Err(_) => return Err(enrichment_pending(ns, d)),
        },
    };
    joined.map_err(|e| {
        err(
            StatusCode::INTERNAL_SERVER_ERROR,
            "facts_internal_error",
            format!("projection rebuild: {e}"),
        )
    })?
}

/// How durable an acknowledged batch is (design §2.4, "durable"): the node's WAL fsync policy, which this store does
/// not change. Power loss is not covered in `batched` or `async` mode.
fn durability(state: &AppState) -> String {
    let storage = &state.config.storage;
    match storage.wal_sync {
        xerj_common::config::WalSync::Sync => "wal_sync=sync".to_string(),
        xerj_common::config::WalSync::Batched => {
            format!("wal_sync=batched({}ms)", storage.wal_batch_ms)
        }
        xerj_common::config::WalSync::Async => "wal_sync=async".to_string(),
    }
}

/// A crash point of the commit protocol (a no-op without the feature `facts-failpoints`). `Err`: the step fails as a failed
/// engine write would; `Ok(true)`: skip the one write the point guards.
fn point(name: &str) -> Result<bool, Response> {
    match failpoint::hit(name) {
        Action::Continue => Ok(false),
        Action::Skip => Ok(true),
        Action::Error => Err(err(
            StatusCode::SERVICE_UNAVAILABLE,
            "facts_write_failed",
            format!("failpoint {name}"),
        )),
    }
}

/// Stage timings of one written batch (ms) and the state of its step-5 repairs.
struct Written {
    records_ms: f64,
    epoch_ms: f64,
    repair_ms: f64,
    epochs: usize,
    /// why step 5 did not finish (the batch is committed; the next rebuild finishes the repairs)
    repair_deferred: Option<String>,
}

/// Steps 3 to 6 of the commit protocol for an applied batch: the `doc` / `fact` records, the epoch markers (the commit),
/// the step-5 repairs, then publishing `to_rev`. On `Err` the caller drops the projection, and the next access rebuilds it
/// from the index: the markers that were written decide what is committed.
async fn write_batch(
    state: &AppState,
    index: &str,
    n: &Namespace,
    applied: &Applied,
) -> Result<Written, Response> {
    point("commit.after_apply")?;
    // step 3: the records; nothing is committed before the batch's last marker
    let t_records = Instant::now();
    for chunk in applied.records.chunks(records_per_bulk()) {
        if point("commit.records")? {
            continue;
        }
        bulk_write(state, index, Write::Index(chunk))
            .await
            .map_err(|e| e.into_response("facts_write_failed"))?;
    }
    let records_ms = t_records.elapsed().as_secs_f64() * 1e3;
    point("commit.after_records")?;
    // step 4: the markers, one bulk call each and in revision order; the last one commits the batch. A failed call leaves
    // the outcome unknown: the rebuild decides by the markers' presence.
    let committed_at = now_ms();
    let t_epoch = Instant::now();
    for (id, src) in &applied.epochs {
        if point("commit.epoch")? {
            continue;
        }
        let mut src = src.clone();
        src["committed_at_ms"] = json!(committed_at);
        bulk_write(state, index, Write::Index(&[(id.clone(), src)]))
            .await
            .map_err(|e| {
                err(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "COMMIT_OUTCOME_UNKNOWN",
                    e.reason,
                )
            })?;
    }
    let epoch_ms = t_epoch.elapsed().as_secs_f64() * 1e3;
    point("commit.after_epochs")?;
    // step 5: the older records' repair-owned fields. The batch is committed whatever happens here.
    let t_repair = Instant::now();
    let mut repair_deferred = None;
    if applied.repairs.len() >= CLEANUP_FLUSH_MIN {
        // Updating records the engine still holds in its memtable is quadratic (see `projection_locked`): 4,849 repairs
        // took 23.8 s of a 27.2 s commit on the x86 build server. A flush first makes each one a tombstone plus a write.
        // A failed flush only costs time.
        if let Ok(idx) = state.engine.get_index(index) {
            if let Err(e) = idx.flush().await {
                tracing::warn!(index, error = %e, "flush before the step-5 repairs failed; repairing anyway");
            }
        }
    }
    for chunk in applied.repairs.chunks(records_per_bulk()) {
        match point("commit.repair") {
            Ok(true) => continue,
            Ok(false) => {}
            Err(_) => {
                repair_deferred = Some("failpoint commit.repair".to_string());
                break;
            }
        }
        if let Err(e) = bulk_write(state, index, Write::Update(chunk)).await {
            repair_deferred = Some(e.reason);
            break;
        }
    }
    let repair_ms = t_repair.elapsed().as_secs_f64() * 1e3;
    point("commit.after_repair")?;
    // step 6: publish
    n.published
        .store(applied.to_rev, std::sync::atomic::Ordering::SeqCst);
    n.epochs.fetch_add(
        applied.epochs.len() as u32,
        std::sync::atomic::Ordering::SeqCst,
    );
    point("commit.after_publish")?;
    Ok(Written {
        records_ms,
        epoch_ms,
        repair_ms,
        epochs: applied.epochs.len(),
        repair_deferred,
    })
}

/// One write batch under the commit protocol of the module doc.
async fn commit(state: &AppState, ns: &str, batch: Batch) -> Response {
    if let Err(r) = check_ns(ns) {
        return r;
    }
    let t0 = Instant::now();
    let svc = state.facts.clone();
    let _w = svc.writer.lock().await;
    let index = facts_index(ns);
    if let Err(r) = ensure_index(state, &index).await {
        return r;
    }
    let n = match projection_locked(state, ns).await {
        Ok(Some(n)) => n,
        Ok(None) => {
            let n = Arc::new(Namespace::new());
            insert_projection(state, ns, n.clone());
            n
        }
        Err(r) => return r,
    };
    if let Err(r) = finish_cleanup(state, ns, &n).await {
        return r;
    }
    let n2 = n.clone();
    let applied = match tokio::task::spawn_blocking(move || n2.apply(&batch)).await {
        Ok(Ok(a)) => a,
        Ok(Err(m)) => {
            svc.remove(ns); // the projection may hold a prefix of the batch
            return err(StatusCode::BAD_REQUEST, "facts_contract_error", m);
        }
        Err(e) => {
            svc.remove(ns);
            return err(
                StatusCode::INTERNAL_SERVER_ERROR,
                "facts_internal_error",
                e.to_string(),
            );
        }
    };
    if applied.to_rev < applied.from_rev {
        return Json(json!({"namespace": ns, "committed": null, "note": "empty batch"}))
            .into_response();
    }
    let w = match write_batch(state, &index, &n, &applied).await {
        Ok(w) => w,
        Err(r) => {
            svc.remove(ns);
            return r;
        }
    };
    if w.repair_deferred.is_some() {
        // committed and published; the next access rebuilds the projection, and the rebuild finishes the repairs
        svc.remove(ns);
    }
    Json(json!({
        "namespace": ns, "index": index,
        "committed": {"from_rev": applied.from_rev, "to_rev": applied.to_rev}, "base_rev": applied.base_rev,
        "records_indexed": applied.records.len(), "epochs_written": w.epochs,
        "repair": {"records": applied.repairs.len(), "deferred": w.repair_deferred},
        "durability": durability(state),
        "timings_ms": {"apply": applied.apply_ms, "records": w.records_ms, "epoch": w.epoch_ms, "repair": w.repair_ms,
                       "total": t0.elapsed().as_secs_f64() * 1e3},
    }))
    .into_response()
}

/// The entries of one write batch (documents + facts + edits), checked against `limits.max_actions_per_bulk` (0 = no
/// limit), the cap of an engine bulk: one batch holds the writer lock and its records in memory until it commits.
fn check_batch_size(state: &AppState, world: &Value) -> Result<(), Response> {
    let max = state.config.limits.max_actions_per_bulk;
    let len = |k: &str| world.get(k).and_then(Value::as_array).map_or(0, Vec::len);
    let n = len("base_documents") + len("base_facts") + len("edits");
    if max > 0 && n > max {
        return Err(err(
            StatusCode::PAYLOAD_TOO_LARGE,
            "facts_batch_too_large",
            format!(
                "this batch holds {n} entries (documents + facts + edits); limits.max_actions_per_bulk is {max}: \
                 split it into several batches"
            ),
        ));
    }
    Ok(())
}

async fn world(
    State(state): State<AppState>,
    Path(ns): Path<String>,
    principal: Principal,
    body: Result<Bytes, BytesRejection>,
) -> Response {
    if let Err(r) = preflight(&state, &principal, &ns, Privilege::WriteIndex) {
        return r;
    }
    let world: Value = match parse_body(body) {
        Ok(w) => w,
        Err(r) => return r,
    };
    if !world.is_object() {
        return err(
            StatusCode::BAD_REQUEST,
            "facts_bad_request",
            "body must be a memory-world-v1 object",
        );
    }
    if let Err(r) = check_batch_size(&state, &world) {
        return r;
    }
    commit(&state, &ns, Batch::World { world }).await
}

async fn bulk(
    State(state): State<AppState>,
    Path(ns): Path<String>,
    principal: Principal,
    body: Result<Bytes, BytesRejection>,
) -> Response {
    if let Err(r) = preflight(&state, &principal, &ns, Privilege::WriteIndex) {
        return r;
    }
    let body: Value = match parse_body(body) {
        Ok(b) => b,
        Err(r) => return r,
    };
    let pick = |k: &str| body.get(k).cloned().unwrap_or_else(|| json!([]));
    let world = json!({"base_documents": pick("documents"), "base_facts": pick("facts"), "edits": pick("edits")});
    if let Err(r) = check_batch_size(&state, &world) {
        return r;
    }
    commit(&state, &ns, Batch::World { world }).await
}

async fn withdraw_fact(
    State(state): State<AppState>,
    Path((ns, fact_id)): Path<(String, String)>,
    principal: Principal,
) -> Response {
    if let Err(r) = preflight(&state, &principal, &ns, Privilege::WriteIndex) {
        return r;
    }
    commit(&state, &ns, Batch::WithdrawFact { fact_id }).await
}

fn no_namespace(ns: &str) -> Response {
    err(
        StatusCode::NOT_FOUND,
        "index_not_found_exception",
        format!("no facts namespace [{ns}]"),
    )
}

/// `_lookup` body: the lookup, plus how long to wait for a projection that is being rebuilt.
#[derive(Deserialize)]
struct LookupBody {
    #[serde(flatten)]
    req: LookupReq,
    #[serde(default)]
    wait_for_ms: Option<u64>,
}

async fn lookup(
    State(state): State<AppState>,
    Path(ns): Path<String>,
    principal: Principal,
    body: Result<Bytes, BytesRejection>,
) -> Response {
    if let Err(r) = preflight(&state, &principal, &ns, Privilege::ReadIndex) {
        return r;
    }
    let LookupBody { req, wait_for_ms } = match parse_body(body) {
        Ok(b) => b,
        Err(r) => return r,
    };
    let n = match projection_within(&state, &ns, wait_of(wait_for_ms)).await {
        Ok(Some(n)) => n,
        Ok(None) => return no_namespace(&ns),
        Err(r) => return r,
    };
    match xerj_facts::lookup(&n, &req) {
        Ok(v) => Json(v).into_response(),
        Err(e) => facts_err(e),
    }
}

/// `GET /_facts/{ns}/{fact_id}[?revision=&wait_for_ms=]`: one fact record and its visibility at a committed revision
/// (default: the published head), read from the projection; no model (design §5.1). Withdrawn and superseded records
/// stay readable with `visible: false`; a record of a batch that is not committed yet is not shown.
async fn get_fact(
    State(state): State<AppState>,
    Path((ns, fact_id)): Path<(String, String)>,
    principal: Principal,
    Query(q): Query<HashMap<String, String>>,
) -> Response {
    if let Err(r) = preflight(&state, &principal, &ns, Privilege::ReadIndex) {
        return r;
    }
    let num = |k: &str| -> Result<Option<u64>, Response> {
        match q.get(k).map(|v| v.parse::<u64>()) {
            None => Ok(None),
            Some(Ok(n)) => Ok(Some(n)),
            Some(Err(_)) => Err(err(
                StatusCode::BAD_REQUEST,
                "facts_bad_request",
                format!("{k} must be a whole number"),
            )),
        }
    };
    let (revision, wait) = match (num("revision"), num("wait_for_ms")) {
        (Ok(r), Ok(w)) => (r, w),
        (Err(e), _) | (_, Err(e)) => return e,
    };
    let Ok(revision) = revision.map(u32::try_from).transpose() else {
        return err(
            StatusCode::BAD_REQUEST,
            "facts_bad_request",
            "revision is out of range",
        );
    };
    match projection_within(&state, &ns, wait.map(Duration::from_millis)).await {
        Ok(Some(n)) => match xerj_facts::get_fact(&n, &fact_id, revision) {
            Ok(Some(mut v)) => {
                v["namespace"] = json!(ns);
                Json(v).into_response()
            }
            Ok(None) => err(
                StatusCode::NOT_FOUND,
                "fact_not_found",
                format!("no fact [{fact_id}] in facts namespace [{ns}]"),
            ),
            Err(e) => facts_err(e),
        },
        Ok(None) => no_namespace(&ns),
        Err(r) => r,
    }
}

/// `_ask` body: the question (`AskReq`), plus `wait_for_ms` (how long to wait for a projection being rebuilt) and
/// `timeout_ms` (this request's deadline; it can only shorten the server's).
#[derive(Deserialize)]
struct AskBody {
    #[serde(flatten)]
    req: AskReq,
    #[serde(default)]
    wait_for_ms: Option<u64>,
    #[serde(default)]
    timeout_ms: Option<u64>,
}

/// Size and shape limits of a question, checked before any work.
fn check_ask(req: &AskReq) -> Result<(), Response> {
    let bad = |m: String| Err(err(StatusCode::BAD_REQUEST, "facts_bad_request", m));
    if req.question.len() > MAX_QUESTION_BYTES {
        return bad(format!(
            "question is {} bytes; at most {MAX_QUESTION_BYTES} are accepted",
            req.question.len()
        ));
    }
    if let Some(id) = &req.item_id {
        if id.len() > MAX_ITEM_ID_BYTES {
            return bad(format!(
                "item_id is {} bytes; at most {MAX_ITEM_ID_BYTES} are accepted",
                id.len()
            ));
        }
    }
    if let Some(l) = &req.linker {
        if !LINKERS.contains(&l.as_str()) {
            return bad(format!("linker [{l}] is not one of {}", LINKERS.join(", ")));
        }
    }
    if let Some(m) = &req.manifest {
        if !MANIFESTS.contains(&m.as_str()) {
            return bad(format!(
                "manifest [{m}] is not one of {}",
                MANIFESTS.join(", ")
            ));
        }
    }
    if let Some(v) = &req.verify {
        if !VERIFY_RULES.contains(&v.as_str()) {
            return bad(format!(
                "verify [{v}] is not one of {}",
                VERIFY_RULES.join(", ")
            ));
        }
    }
    Ok(())
}

async fn ask(
    State(state): State<AppState>,
    Path(ns): Path<String>,
    principal: Principal,
    body: Result<Bytes, BytesRejection>,
) -> Response {
    let t0 = Instant::now();
    if let Err(r) = preflight(&state, &principal, &ns, Privilege::ReadIndex) {
        return r;
    }
    let AskBody {
        req,
        wait_for_ms,
        timeout_ms,
    } = match parse_body(body) {
        Ok(b) => b,
        Err(r) => return r,
    };
    if let Err(r) = check_ask(&req) {
        return r;
    }
    if let Some(e) = &state.facts.config_error {
        return err_with(
            StatusCode::SERVICE_UNAVAILABLE,
            "FACTS_CONFIG_INVALID",
            format!("the node's _ask limits could not be read: {e}"),
            json!({"retryable": false}),
        );
    }
    let gate = state.facts.ask_gate.clone();
    let deadline = gate.effective_deadline(timeout_ms.map(Duration::from_millis));
    // the wait for the projection is part of the deadline too
    let wait = match (wait_of(wait_for_ms), deadline) {
        (Some(w), Some(d)) => Some(w.min(d)),
        (w, d) => w.or(d),
    };
    let n = match projection_within(&state, &ns, wait).await {
        Ok(Some(n)) => n,
        Ok(None) => return no_namespace(&ns),
        Err(r) => return r,
    };
    // The blocking closure owns only the projection and the service: no engine handle, no name resolution. The gate
    // sets `stop` when it stops waiting (deadline, or this request dropped), and the compiler stops before its next
    // decision.
    let svc = state.facts.clone();
    let res = gate
        .run_stoppable(t0, deadline, move |stop| {
            let slot = svc.model()?;
            xerj_facts::ask_stoppable(&slot, &n, &req, Some(stop))
        })
        .await;
    match res {
        Ok(Ok(mut v)) => {
            v["namespace"] = json!(ns);
            v["handler_ms"] = json!(t0.elapsed().as_secs_f64() * 1e3);
            Json(v).into_response()
        }
        Ok(Err(e)) => facts_err(e),
        Err(GateError::Deadline { stage, waited_ms }) => err_with(
            StatusCode::GATEWAY_TIMEOUT,
            "ASK_DEADLINE",
            format!(
                "no answer within the deadline of {} ms ({}); nothing was answered",
                deadline.map_or(0, |d| d.as_millis()),
                if stage == "queue" {
                    "every ask slot stayed busy"
                } else {
                    "the compiler was still running and stops before its next decision"
                }
            ),
            json!({"retryable": true, "stage": stage, "waited_ms": waited_ms,
                   "deadline_ms": deadline.map(|d| d.as_millis() as u64)}),
        ),
        Err(GateError::QueueFull {
            waiting,
            max_queued,
        }) => err_with(
            StatusCode::TOO_MANY_REQUESTS,
            "ASK_QUEUE_FULL",
            format!(
                "every ask slot is busy and {waiting} asks are already waiting (at most {max_queued} wait); \
                 nothing was run: retry later"
            ),
            json!({"retryable": true, "waiting": waiting, "max_queued": max_queued}),
        ),
        Err(GateError::Failed(e)) => {
            err(StatusCode::INTERNAL_SERVER_ERROR, "facts_internal_error", e)
        }
    }
}

async fn status(
    State(state): State<AppState>,
    Path(ns): Path<String>,
    principal: Principal,
    Query(q): Query<HashMap<String, String>>,
) -> Response {
    if let Err(r) = preflight(&state, &principal, &ns, Privilege::ReadIndex) {
        return r;
    }
    let wait = match q.get("wait_for_ms").map(|v| v.parse::<u64>()) {
        None => None,
        Some(Ok(ms)) => Some(Duration::from_millis(ms)),
        Some(Err(_)) => {
            return err(
                StatusCode::BAD_REQUEST,
                "facts_bad_request",
                "wait_for_ms must be a whole number of milliseconds",
            )
        }
    };
    match projection_within(&state, &ns, wait).await {
        Ok(Some(n)) => {
            let fp = q.get("fingerprint").is_some_and(|v| v == "true");
            let mut v = tokio::task::spawn_blocking(move || n.status(fp))
                .await
                .unwrap_or(Value::Null);
            v["namespace"] = json!(ns);
            v["index"] = json!(facts_index(&ns));
            Json(v).into_response()
        }
        Ok(None) => no_namespace(&ns),
        Err(r) => r,
    }
}

/// `GET /_facts`: the facts namespaces the caller may read (design §5.1 "read, filtered"), each `ready` (projection in
/// memory) or `cold` (rebuilt on first access), plus the compiler status and the `_ask` limits. A namespace the caller
/// cannot read is not listed, so the listing reveals nothing a 403 would not.
async fn list(State(state): State<AppState>, principal: Principal) -> Response {
    if let Err(r) = cluster_gate(&state) {
        return r;
    }
    use std::sync::atomic::Ordering::SeqCst;
    let loaded: HashMap<String, Arc<Namespace>> = state.facts.loaded().into_iter().collect();
    let mut namespaces = Vec::new();
    for idx in state.engine.list_indices().await {
        let Some(ns) = idx
            .name
            .strip_prefix(authz::RESERVED_INDEX_PREFIX)
            .and_then(|rest| rest.strip_suffix(authz::FACTS_INDEX_SUFFIX))
        else {
            continue;
        };
        if check_ns(ns).is_err()
            || authz::authorize_index(&principal, &idx.name, Privilege::ReadIndex).is_err()
        {
            continue;
        }
        namespaces.push(match loaded.get(ns) {
            Some(n) => json!({"namespace": ns, "index": idx.name, "state": "ready",
                              "published_rev": n.published.load(SeqCst), "epochs": n.epochs.load(SeqCst)}),
            None => json!({"namespace": ns, "index": idx.name, "state": "cold"}),
        });
    }
    namespaces.sort_by(|a, b| a["namespace"].as_str().cmp(&b["namespace"].as_str()));
    // `loaded` keeps the spike's shape (the namespaces with a projection in memory), filtered the same way
    let loaded_view: Vec<Value> = namespaces
        .iter()
        .filter(|v| v["state"] == "ready")
        .map(|v| json!({"namespace": v["namespace"], "published_rev": v["published_rev"], "epochs": v["epochs"]}))
        .collect();
    let (max_resident, evictions) = state.facts.residency();
    let gate = &state.facts.ask_gate;
    Json(json!({
        "namespaces": namespaces, "loaded": loaded_view, "model": state.facts.model_status(),
        "max_resident_namespaces": max_resident, "evictions": evictions,
        "ask_limits": {"max_concurrent": gate.max_concurrent(), "running": gate.running(),
                       "max_queued": gate.max_queued(), "waiting": gate.waiting(),
                       "deadline_ms": gate.deadline().map(|d| d.as_millis() as u64),
                       "config_error": state.facts.config_error},
        "spike": true,
    }))
    .into_response()
}

async fn drop_namespace(
    State(state): State<AppState>,
    Path(ns): Path<String>,
    principal: Principal,
) -> Response {
    if let Err(r) = preflight(&state, &principal, &ns, Privilege::AdminIndex) {
        return r;
    }
    let _w = state.facts.writer.lock().await;
    state.facts.remove(&ns);
    let index = facts_index(&ns);
    if !index_exists(&state, &index) {
        return no_namespace(&ns);
    }
    let resp = es_compat::delete_index(
        State(state.clone()),
        Path(index.clone()),
        Query(es_compat::DeleteIndexParams::default()),
    )
    .await
    .into_response();
    let (code, body) = drain_json(resp).await;
    (
        code,
        Json(json!({"namespace": ns, "index": index, "result": body})),
    )
        .into_response()
}
