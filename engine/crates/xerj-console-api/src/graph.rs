//! Session-authorized Second-Brain graph reads (issue #936).
//!
//! The data plane's `/_graph/{brain}/ego|overview` routes authenticate an
//! engine *credential* (`Authorization` header — API key), and the
//! session-authenticated data-sources proxy deliberately refuses the reserved
//! `.xerj-memory-*` namespace (RC10 B1, pinned by
//! `tests/console_cannot_read_a_brain.rs`). Both decisions are right; together
//! they meant a signed-in console operator could not see any brain at all on
//! an auth-enabled engine — the default posture — while a share-link *guest*,
//! who holds a key, could.
//!
//! This module is the console's own read path for the graph. It authorizes
//! the **console session's role** and then reads the engine in-process (the
//! same `Engine` handle `data_sources` already uses), so no data-plane
//! credential is involved and the data plane's Authorization-header boundary
//! is untouched.
//!
//! ## The authorization rule
//!
//! Console roles `owner` and `admin` may read brains; `editor` and `viewer`
//! may not. Brains are node-local today: there is no per-tenant ownership
//! model anywhere yet, so the operator tier is the node's operator — the same
//! tier that may issue invites (`auth::magic::issue` gates `owner|admin`).
//! When a tenant model lands, this gate is where a per-user allow-list goes.
//!
//! No existence oracle, mirroring the discipline the search proxy set: a
//! role-refused read and a nonexistent brain return **byte-identical**
//! responses (same status, same body modulo the brain name the caller
//! supplied), so a session that may not read brains cannot enumerate them.
//! The `brains` listing is a flat role gate — 403 for a disallowed role —
//! which reveals no per-brain fact.
//!
//! ## Endpoints
//!
//! ```text
//! GET  /_xerj-console/api/v1/graph/brains              list brains (+ nodes_index)
//! GET  /_xerj-console/api/v1/graph/{brain}/ego         §4.3 neighborhood
//! GET  /_xerj-console/api/v1/graph/{brain}/overview    §4.4 brain stats
//! POST /_xerj-console/api/v1/graph/{brain}/edges/_search   ES _search on the
//!                                                     brain's edges index only
//! POST /_xerj-console/api/v1/graph/{brain}/nodes/_search   ES _search on ONE
//!                                                     of the brain's nodes
//!                                                     indices (meta-doc set)
//! ```
//!
//! The read endpoints re-derive their answers with the same engine machinery
//! the data plane uses — `xerj_engine::graph::Index::graph_expand` for the
//! traversal, `Index::search` for the bounded `ids` hydrations and the
//! overview aggregations — and answer in the same JSON contract
//! (`xerj-second-brain/1`) so the SPA consumes one shape whether it reads
//! through the console session (here) or a scoped key (data plane). The
//! implementation mirrors `xerj-api/src/graph_api.rs` handler-for-handler;
//! each mirrored block names its source. Deliberate differences:
//!
//! - the `ego?nodes_index=` override is refused (400): on the data plane it
//!   is authorized against the caller's grants, but a console session holds
//!   no grants, and silently ignoring it would change semantics between the
//!   two surfaces. Hydration always resolves through the brain's meta doc.
//! - the reserved namespace is reached only through these brain-scoped
//!   routes. The generic data-sources proxy still refuses it (B1 unchanged),
//!   and `link`/`unlink` have no console path — brains are written by the
//!   engine's own writers (`xerj brain`, agents with keys), read here.
//!
//! `/v1/metrics` for the Second Brain dashboard's "searches per index" tile
//! deliberately stays key-only: the metrics surface is an ops plane, and the
//! tile already renders the refusal honestly.

use std::collections::{HashMap, HashSet};

use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde::Deserialize;
use serde_json::{json, Map, Value};

use crate::auth::sessions::AuthSession;
use crate::auth::store;
use crate::error::{ConsoleApiError, ConsoleResult};
use crate::response::ok;
use crate::state::ConsoleState;
use xerj_common::types::RESERVED_INDEX_PREFIX;
use xerj_engine::graph::{
    GraphDirection, GraphEdgeLite, GraphExpandRequest, GRAPH_HOPS_CAP_REASON,
};
use xerj_engine::Engine;

/// Contract version string, returned by every read endpoint. Byte-identical
/// to `xerj_api::graph_api::GRAPH_CONTRACT` — the console answers the same
/// contract the data plane does.
const GRAPH_CONTRACT: &str = "xerj-second-brain/1";

/// Reserved `_id` of the per-brain meta document (SECOND_BRAIN_SPEC §2.5).
pub(crate) const BRAIN_META_ID: &str = "__xerj-brain-meta";

/// Max returned edges for `ego` (same clamp as the data plane).
const MAX_EGO_LIMIT: usize = 1000;

/// Max seed ids accepted by `ego`'s `nodes=` param (same cap as the data
/// plane; excess seeds are dropped and counted in `not_shown.frontier_clipped`).
const EGO_SEEDS_CAP: usize = 64;

/// Max dangling node ids listed verbatim in `not_shown.dangling_ids`.
const MAX_DANGLING_LISTED: usize = 50;

/// Max brains listed by `GET /brains`; the excess is counted, never silently
/// dropped. Real nodes hold a handful of brains; the cap is a bound on
/// per-brain meta-doc reads, not a product limit.
const BRAINS_LISTED_CAP: usize = 256;

/// Edges index name for a brain (SECOND_BRAIN_SPEC §1) — same construction as
/// `xerj_api::authz::brain_edges_index`, which names the resource a
/// data-plane grant must hold.
pub(crate) fn edges_index(brain: &str) -> String {
    format!("{RESERVED_INDEX_PREFIX}{brain}-edges")
}

/// Default nodes index for a brain: the agent-memory namespace of the same
/// name. Autoindex brains override this via the §2.5 meta doc.
fn default_nodes_index(brain: &str) -> String {
    format!("{RESERVED_INDEX_PREFIX}{brain}")
}

/// Brain-name validation (SECOND_BRAIN_SPEC §1): byte-identical rules to
/// `xerj_api::graph_api::validate_brain` (lowercase/digit start, `[a-z0-9._-]`,
/// ≤200 chars, no `..`, `-edges` suffix reserved). A copy, not a shared fn,
/// matching the existing deliberate duplication (`memory_api`,
/// `xerj_autoindex::detect`) — the console crate does not depend on
/// `xerj-api`, on purpose (see `lib.rs`).
pub(crate) fn validate_brain(brain: &str) -> Result<(), String> {
    if brain.is_empty() {
        return Err("brain name must not be empty".into());
    }
    if brain.len() > 200 {
        return Err("brain name too long (max 200 chars)".into());
    }
    let first = brain.chars().next().unwrap();
    if !(first.is_ascii_lowercase() || first.is_ascii_digit()) {
        return Err("brain name must start with a lowercase letter or digit".into());
    }
    for c in brain.chars() {
        let ok = c.is_ascii_lowercase() || c.is_ascii_digit() || matches!(c, '_' | '-' | '.');
        if !ok {
            return Err(format!(
                "brain name contains illegal character '{c}' (allowed: a-z 0-9 _ - .)"
            ));
        }
    }
    if brain.contains("..") {
        return Err("brain name must not contain '..'".into());
    }
    if brain.ends_with("-edges") {
        return Err("namespace suffix '-edges' is reserved for graph edge indices".into());
    }
    // `-facts`: brain `kb-facts`'s default nodes index is facts namespace `kb`'s index. Reserved in builds with the
    // facts store only (feature `facts`), like the other copies of this rule.
    if cfg!(feature = "facts") && brain.ends_with("-facts") {
        return Err("namespace suffix '-facts' is reserved for facts indices".into());
    }
    Ok(())
}

/// May this console session read brains? The node's operator tier — the same
/// roles that may issue invites (`auth::magic::issue`). See module docs for
/// the rule and its no-tenant-model caveat. Public because
/// `tests/console_graph_read.rs` pins the tier boundary; not part of the
/// HTTP surface.
#[doc(hidden)]
pub fn may_read_brains(user: &store::User) -> bool {
    matches!(user.role.as_str(), "owner" | "admin")
}

/// The 404 every "you may not know about this brain" path returns — identical
/// for a role-refused read and a missing brain, modulo the caller-supplied
/// name. Mirrors the data plane's unknown-brain 404 shape (`graph_error`)
/// so the SPA sees one contract.
fn brain_not_found(brain: &str) -> Response {
    (
        StatusCode::NOT_FOUND,
        Json(json!({
            "error": {
                "type": "graph_error",
                "reason": format!("brain '{brain}' does not exist (no edges index)"),
            },
            "status": 404,
        })),
    )
        .into_response()
}

/// A uniform graph error response (same shape as the data plane's).
fn graph_error(status: StatusCode, reason: impl Into<String>) -> Response {
    (
        status,
        Json(json!({
            "error": { "type": "graph_error", "reason": reason.into() },
            "status": status.as_u16(),
        })),
    )
        .into_response()
}

/// Validate the brain name, then apply the role gate in the oracle-safe
/// order: a session that may not read brains gets exactly the response a
/// nonexistent brain produces. Malformed names 400 for every role — a
/// validation error leaks no existence fact. `Some(response)` = return it.
fn gate_brain(user: &store::User, brain: &str) -> Option<Response> {
    if let Err(reason) = validate_brain(brain) {
        return Some(graph_error(StatusCode::BAD_REQUEST, reason));
    }
    if !may_read_brains(user) {
        return Some(brain_not_found(brain));
    }
    None
}

/// Run one ES-shaped `_search` body against one index, in-process. The same
/// pattern `data_sources::search` uses (`parse_request` → `Index::search`);
/// errors surface as a Console 400/500 like every other console read.
async fn run_search(
    engine: &Engine,
    index: &str,
    body: &Value,
) -> ConsoleResult<xerj_query::executor::SearchResult> {
    let idx = engine.get_index(index)?;
    let req = xerj_query::parser::parse_request(body)
        .map_err(|e| ConsoleApiError::BadRequest(e.to_string()))?;
    idx.search(&req)
        .await
        .map_err(|e| ConsoleApiError::Internal(e.to_string()))
}

/// One document by id, via the ids-prefilter fast path (the same read the
/// data plane's `get_doc` resolves to). `None` when absent.
async fn get_doc_source(engine: &Engine, index: &str, id: &str) -> ConsoleResult<Option<Value>> {
    let body = json!({ "query": { "ids": { "values": [id] } }, "size": 1 });
    let r = run_search(engine, index, &body).await?;
    Ok(r.hits.into_iter().next().map(|h| h.source))
}

/// The concrete names in a `nodes_index` value — `xerj brain` records a
/// multi-dataset folder as one comma-joined string (`"ax-mail,ax-pdfs"`).
/// Mirrors `graph_api::nodes_index_names`.
fn nodes_index_names(nodes_index: &str) -> impl Iterator<Item = &str> {
    nodes_index
        .split(',')
        .map(str::trim)
        .filter(|n| !n.is_empty())
}

/// Resolve the nodes index for a brain: the meta doc's `nodes_index`, else
/// the `.xerj-memory-{brain}` default. Mirrors `graph_api::resolve_nodes_index`
/// (explicit-param override excluded — see module docs).
pub(crate) async fn resolve_nodes_index(engine: &Engine, brain: &str, edges: &str) -> String {
    if let Some(src) = get_doc_source(engine, edges, BRAIN_META_ID)
        .await
        .ok()
        .flatten()
    {
        if let Some(ni) = src.get("nodes_index").and_then(Value::as_str) {
            if !ni.is_empty() {
                return ni.to_string();
            }
        }
    }
    default_nodes_index(brain)
}

/// Parse a query-string timestamp: decimal epoch-ms first, RFC3339 second
/// (mirrors `graph_api::parse_ms_str`).
fn parse_ms_str(s: &str) -> Option<i64> {
    if let Ok(n) = s.parse::<i64>() {
        return Some(n);
    }
    chrono::DateTime::parse_from_rfc3339(s)
        .ok()
        .map(|dt| dt.timestamp_millis())
}

/// Server "now" in epoch milliseconds.
fn now_ms() -> i64 {
    chrono::Utc::now().timestamp_millis()
}

// ─────────────────────────────────────────────────────────────────────────────
// GET /brains — discovery listing
// ─────────────────────────────────────────────────────────────────────────────

/// `GET /_xerj-console/api/v1/graph/brains` — every brain on this node, with
/// the `nodes_index` its meta doc records. Replaces the SPA's direct
/// `_cat/indices/.xerj-memory-*` walk + per-brain meta-doc reads, both of
/// which 401 for a session. Flat role gate: 403 for a role that may not read
/// brains (reveals no per-brain fact), listing for the operator tier.
pub async fn brains(
    State(state): State<ConsoleState>,
    sess: AuthSession,
) -> ConsoleResult<Response> {
    if !may_read_brains(&sess.user) {
        return Err(ConsoleApiError::Forbidden(format!(
            "the console role '{}' may not read brains",
            sess.user.role
        )));
    }

    // Brain = a reserved `.xerj-memory-{brain}-edges` index that exists.
    let mut names: Vec<String> = state
        .engine
        .index_name_list()
        .into_iter()
        .filter(|n| {
            n.starts_with(RESERVED_INDEX_PREFIX)
                && n.ends_with("-edges")
                && n.len() > RESERVED_INDEX_PREFIX.len() + "-edges".len()
        })
        .map(|n| n[RESERVED_INDEX_PREFIX.len()..n.len() - "-edges".len()].to_string())
        .filter(|b| validate_brain(b).is_ok())
        .collect();
    names.sort();

    let total = names.len();
    let clipped = total.saturating_sub(BRAINS_LISTED_CAP);
    names.truncate(BRAINS_LISTED_CAP);

    let mut brains: Vec<Value> = Vec::with_capacity(names.len());
    for brain in names {
        let edges = edges_index(&brain);
        let nodes_index = resolve_nodes_index(&state.engine, &brain, &edges).await;
        brains.push(json!({
            "name": brain,
            "nodes_index": nodes_index,
            "contract": GRAPH_CONTRACT,
        }));
    }

    Ok(ok(
        json!({
            "brains": brains,
            "total": total,
            "not_shown": { "brains_clipped": clipped as u64 },
        }),
        None,
    ))
}

// ─────────────────────────────────────────────────────────────────────────────
// GET /{brain}/ego — bounded neighborhood
// ─────────────────────────────────────────────────────────────────────────────

/// Query params for `GET /graph/{brain}/ego` — the data plane's §4.3 set,
/// minus the `nodes_index` override (refused here; see module docs).
#[derive(Debug, Default, Deserialize)]
pub struct EgoParams {
    #[serde(default)]
    pub node: Option<String>,
    #[serde(default)]
    pub nodes: Option<String>,
    #[serde(default)]
    pub hops: Option<u64>,
    #[serde(default)]
    pub direction: Option<String>,
    #[serde(default)]
    pub types: Option<String>,
    #[serde(default)]
    pub limit: Option<u64>,
    #[serde(default)]
    pub as_of: Option<String>,
    #[serde(default)]
    pub include_expired: Option<bool>,
    #[serde(default)]
    pub include_nodes: Option<bool>,
    #[serde(default)]
    pub nodes_index: Option<String>,
    #[serde(default)]
    pub include_evidence: Option<bool>,
}

/// `GET /_xerj-console/api/v1/graph/{brain}/ego` — the §4.3 neighborhood of
/// one node (or ≤64 via `nodes=`). Mirrors `graph_api::ego` step for step;
/// the authorization halves differ (console role gate instead of Principal
/// grants, in-process hydrations instead of composed ES-compat handlers).
pub async fn ego(
    State(state): State<ConsoleState>,
    sess: AuthSession,
    Path(brain): Path<String>,
    Query(params): Query<EgoParams>,
) -> Response {
    if let Some(resp) = gate_brain(&sess.user, &brain) {
        return resp;
    }
    if let Some(ni) = params.nodes_index.as_deref() {
        let _ = ni;
        return graph_error(
            StatusCode::BAD_REQUEST,
            "`nodes_index` is not supported on the console path: hydration resolves \
             through the brain's meta document",
        );
    }
    if params.node.is_some() && params.nodes.is_some() {
        return graph_error(
            StatusCode::BAD_REQUEST,
            "`node` and `nodes` are mutually exclusive — `nodes` is the multi-seed form",
        );
    }
    // Seed list: `node` is the 1-element case of `nodes`. Deduped preserving
    // order (the order is part of the `reachable` contract), then clamped to
    // EGO_SEEDS_CAP with the clip counted — never silent.
    let raw_seeds: Vec<String> = match (params.node.as_deref(), params.nodes.as_deref()) {
        (Some(n), None) if !n.is_empty() => vec![n.to_string()],
        (None, Some(ns)) => ns
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(String::from)
            .collect(),
        _ => Vec::new(),
    };
    let mut seeds: Vec<String> = Vec::with_capacity(raw_seeds.len());
    {
        let mut seen: HashSet<&str> = HashSet::with_capacity(raw_seeds.len());
        for id in &raw_seeds {
            if seen.insert(id.as_str()) {
                seeds.push(id.clone());
            }
        }
    }
    if seeds.is_empty() {
        return graph_error(
            StatusCode::BAD_REQUEST,
            "`node` (or comma-separated `nodes`) is required: the node id(s) to expand from",
        );
    }
    let mut seeds_clipped = 0u64;
    if seeds.len() > EGO_SEEDS_CAP {
        seeds_clipped = (seeds.len() - EGO_SEEDS_CAP) as u64;
        seeds.truncate(EGO_SEEDS_CAP);
    }
    let hops = params.hops.unwrap_or(1);
    if hops == 0 || hops > 2 {
        return graph_error(StatusCode::BAD_REQUEST, GRAPH_HOPS_CAP_REASON);
    }
    let direction_str = params.direction.as_deref().unwrap_or("both");
    let direction = match direction_str {
        "out" => GraphDirection::Out,
        "in" => GraphDirection::In,
        "both" => GraphDirection::Both,
        other => {
            return graph_error(
                StatusCode::BAD_REQUEST,
                format!("`direction` must be 'out', 'in', or 'both' (got '{other}')"),
            );
        }
    };
    let types: Option<Vec<String>> = params.types.as_deref().map(|t| {
        t.split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(String::from)
            .collect()
    });
    let types = types.filter(|t| !t.is_empty());
    let limit = (params.limit.unwrap_or(100) as usize).clamp(1, MAX_EGO_LIMIT);
    let as_of = match params.as_of.as_deref() {
        None => now_ms(),
        Some(s) => match parse_ms_str(s) {
            Some(ms) => ms,
            None => {
                return graph_error(
                    StatusCode::BAD_REQUEST,
                    "`as_of` must be an epoch-ms number or an RFC3339 string",
                );
            }
        },
    };
    let include_expired = params.include_expired.unwrap_or(false);
    let include_nodes = params.include_nodes.unwrap_or(false);
    let include_evidence = params.include_evidence.unwrap_or(true);

    let edges = edges_index(&brain);
    let Ok(idx) = state.engine.get_index(&edges) else {
        return brain_not_found(&brain);
    };

    let req = GraphExpandRequest {
        frontier: seeds.clone(),
        hops: hops as u8,
        direction,
        types,
        as_of_ms: as_of,
        include_expired,
        max_result_edges: limit,
    };
    let result = match idx.graph_expand(&req) {
        Ok(r) => r,
        Err(e) => return graph_error(StatusCode::BAD_REQUEST, e.to_string()),
    };

    // Per-edge direction relative to the expansion (mirrors graph_api::ego):
    // an edge discovered at hop h is "out" iff its src was in hop h's
    // frontier; the hop-1 frontier is the seed set, the hop-2 frontier is
    // every endpoint hop 1 discovered.
    let frontier1: HashSet<&str> = seeds.iter().map(String::as_str).collect();
    let mut frontier2: HashSet<&str> = HashSet::new();
    for e in result.edges.iter().filter(|e| e.hop == 1) {
        for id in [e.src.as_str(), e.dst.as_str()] {
            if !frontier1.contains(id) {
                frontier2.insert(id);
            }
        }
    }
    let edge_direction = |e: &GraphEdgeLite| -> &'static str {
        let frontier = if e.hop == 1 { &frontier1 } else { &frontier2 };
        if frontier.contains(e.src.as_str()) {
            "out"
        } else {
            "in"
        }
    };

    // Post-traversal hydration (bounded ≤ limit ≤ 1000): ONE `ids` search on
    // the edges index for evidence/envelope fields — rides the ids-prefilter
    // fast path; the hop itself never touched `_source`.
    let mut hydrated: HashMap<String, Value> = HashMap::new();
    if include_evidence && !result.edges.is_empty() {
        let ids: Vec<&str> = result.edges.iter().map(|e| e.edge_id.as_str()).collect();
        let body = json!({
            "query": { "ids": { "values": ids } },
            "size": ids.len(),
            "_source": ["edge_id", "created_at", "detector", "confidence", "evidence", "expired_at"],
        });
        if let Ok(r) = run_search(&state.engine, &edges, &body).await {
            for h in r.hits {
                if let Some(id) = h.source.get("edge_id").and_then(Value::as_str) {
                    hydrated.insert(id.to_string(), h.source);
                }
            }
        }
    }

    // Node summaries (`include_nodes=true`): ONE `ids` search per concrete
    // nodes-index name for every reachable id. Ids that resolve nowhere are
    // DANGLING — the edge is kept, the honesty is counted, and up to 50 ids
    // are listed.
    let mut nodes_obj = Map::new();
    let mut dangling_nodes = 0u64;
    let mut dangling_ids: Vec<String> = Vec::new();
    if include_nodes {
        let nodes_index = resolve_nodes_index(&state.engine, &brain, &edges).await;
        let mut found: HashSet<String> = HashSet::new();
        for name in nodes_index_names(&nodes_index) {
            // A nodes index that does not exist (edges asserted through the
            // API alone) contributes nothing; every reachable id is then
            // reported dangling, never fabricated.
            if state.engine.get_index(name).is_err() {
                continue;
            }
            let ids: Vec<&str> = result.reachable.iter().map(String::as_str).collect();
            let body = json!({
                "query": { "ids": { "values": ids } },
                "size": ids.len(),
                "_source": ["title", "text", "body", "ax_path"],
            });
            let Ok(r) = run_search(&state.engine, name, &body).await else {
                continue;
            };
            for h in r.hits {
                let id = h.id.clone();
                let title = h
                    .source
                    .get("title")
                    .and_then(Value::as_str)
                    .map(Value::from)
                    .unwrap_or(Value::Null);
                let preview = h
                    .source
                    .get("text")
                    .and_then(Value::as_str)
                    .or_else(|| h.source.get("body").and_then(Value::as_str))
                    .map(|t| Value::from(t.chars().take(160).collect::<String>()))
                    .unwrap_or(Value::Null);
                // Truthful file label: the autoindex writer stamps `ax_path`
                // on every note; null when absent — never fabricated.
                let path = h
                    .source
                    .get("ax_path")
                    .and_then(Value::as_str)
                    .map(Value::from)
                    .unwrap_or(Value::Null);
                found.insert(id.clone());
                nodes_obj.entry(id).or_insert(json!({
                    "title": title,
                    "preview": preview,
                    "path": path,
                    "index": name,
                }));
            }
        }
        let mut dangling: Vec<&String> = result
            .reachable
            .iter()
            .filter(|id| !found.contains(*id))
            .collect();
        dangling.sort();
        dangling_nodes = dangling.len() as u64;
        dangling_ids = dangling
            .into_iter()
            .take(MAX_DANGLING_LISTED)
            .cloned()
            .collect();
    }

    // Edges in the §3.2 stable order (the engine already sorted), each
    // annotated with its expansion direction; `invalid_at` is JSON null when
    // unset (response-side null is fine — the omit-rule binds stored docs).
    let edges_json: Vec<Value> = result
        .edges
        .iter()
        .map(|e| {
            let mut obj = json!({
                "edge_id": e.edge_id,
                "src": e.src,
                "dst": e.dst,
                "type": e.edge_type,
                "weight": e.weight,
                "hop": e.hop,
                "direction": edge_direction(e),
                "valid_at": e.valid_at_ms,
                "invalid_at": e.invalid_at_ms.map(Value::from).unwrap_or(Value::Null),
            });
            if let Some(src) = hydrated.get(&e.edge_id) {
                for field in [
                    "created_at",
                    "detector",
                    "confidence",
                    "evidence",
                    "expired_at",
                ] {
                    if let Some(v) = src.get(field) {
                        obj[field] = v.clone();
                    }
                }
            }
            obj
        })
        .collect();

    // Neighbors: first-discovery order following the sorted edge list,
    // excluding the seed nodes; `via_edge` is the first sorted edge that
    // reached each one.
    let mut neighbors: Vec<Value> = Vec::new();
    let mut seen: HashSet<&str> = seeds.iter().map(String::as_str).collect();
    for e in &result.edges {
        for id in [e.src.as_str(), e.dst.as_str()] {
            if seen.insert(id) {
                neighbors.push(json!({ "id": id, "hop": e.hop, "via_edge": e.edge_id }));
            }
        }
    }

    // `seeds` echoes the ADMITTED seed list (post-dedupe, post-clamp) so the
    // caller can bookkeep exactly what was expanded; `node` stays on the
    // response whenever there is exactly one seed (the 1-element case keeps
    // its historical shape). Handler-clipped seeds fold into the same
    // `frontier_clipped` counter the engine uses — one honesty channel.
    let mut resp = json!({
        "brain": brain,
        "contract": GRAPH_CONTRACT,
        "seeds": seeds,
        "as_of": as_of,
        "hops": hops,
        "direction": direction_str,
        "edges": edges_json,
        "neighbors": neighbors,
        "not_shown": {
            "edges_clipped": result.stats.edges_clipped,
            "frontier_clipped": result.stats.frontier_clipped + seeds_clipped,
            "expired_excluded": result.stats.expired_excluded,
            "type_filtered": result.stats.type_filtered,
            "segments_without_columns": result.stats.segments_without_columns,
            "memtable_docs_scanned": result.stats.memtable_docs_scanned,
            "dangling_nodes": dangling_nodes,
            "dangling_ids": dangling_ids,
        }
    });
    if let [only] = resp["seeds"].as_array().map(Vec::as_slice).unwrap_or(&[]) {
        // §8.5 is a normative instance including key order: `node` sits
        // between `contract` and `seeds`, so rebuild in place rather than
        // appending (which would serialize `node` last).
        let only = only.clone();
        let old = std::mem::take(resp.as_object_mut().expect("ego response is an object"));
        let mut ordered = serde_json::Map::with_capacity(old.len() + 1);
        for (k, v) in old {
            if k == "seeds" {
                ordered.insert("node".to_string(), only.clone());
            }
            ordered.insert(k, v);
        }
        resp = Value::Object(ordered);
    }
    if include_nodes {
        resp["nodes"] = Value::Object(nodes_obj);
    }
    Json(resp).into_response()
}

// ─────────────────────────────────────────────────────────────────────────────
// GET /{brain}/overview — brain-level stats
// ─────────────────────────────────────────────────────────────────────────────

/// Query params for `GET /graph/{brain}/overview` (§4.4; same as the data
/// plane's `OverviewParams`).
#[derive(Debug, Default, Deserialize)]
pub struct OverviewParams {
    #[serde(default)]
    pub as_of: Option<String>,
    #[serde(default)]
    pub top: Option<u64>,
    #[serde(default)]
    pub histogram_interval: Option<String>,
}

/// Terms-agg buckets → `[{"<key_name>": key, "<count_name>": doc_count}]`,
/// plus the agg's `sum_other_doc_count` (the not-listed tail, reported
/// in-band). Mirrors `graph_api::terms_list`.
fn terms_list(aggs: &Value, agg: &str, key_name: &str, count_name: &str) -> (Vec<Value>, u64) {
    let buckets = aggs
        .pointer(&format!("/{agg}/buckets"))
        .and_then(Value::as_array);
    let list = buckets
        .map(|bs| {
            bs.iter()
                .map(|b| {
                    json!({
                        key_name: b.get("key").cloned().unwrap_or(Value::Null),
                        count_name: b.get("doc_count").cloned().unwrap_or(json!(0)),
                    })
                })
                .collect()
        })
        .unwrap_or_default();
    let other = aggs
        .pointer(&format!("/{agg}/sum_other_doc_count"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    (list, other)
}

/// The honesty marker for the default embedder (§0 invariant 8): the built-in
/// embedder is LEXICAL feature-hashing — no surface may imply neural
/// semantics. Mirrors `graph_api::embedder_id`, read off the same engine
/// config the data plane's AppState holds.
fn embedder_id(engine: &Engine) -> String {
    let emb = &engine.config().embedding;
    match emb.mode.as_str() {
        "neural" => emb.neural_model.clone(),
        "proxy" if !emb.default_model.is_empty() => emb.default_model.clone(),
        "proxy" => "proxy".into(),
        "auto" if !emb.default_endpoint.is_empty() && !emb.default_model.is_empty() => {
            emb.default_model.clone()
        }
        "auto" if !emb.default_endpoint.is_empty() => "proxy".into(),
        _ => "lexical-feature-hash".into(),
    }
}

/// `GET /_xerj-console/api/v1/graph/{brain}/overview` — totals, live slice
/// (types/detectors/hubs), the created-over-time histogram, and the notes
/// total. Mirrors `graph_api::overview`; exactly three searches on the edges
/// index plus one size-0 count per concrete nodes-index name.
pub async fn overview(
    State(state): State<ConsoleState>,
    sess: AuthSession,
    Path(brain): Path<String>,
    Query(params): Query<OverviewParams>,
) -> Response {
    if let Some(resp) = gate_brain(&sess.user, &brain) {
        return resp;
    }
    let as_of = match params.as_of.as_deref() {
        None => now_ms(),
        Some(s) => match parse_ms_str(s) {
            Some(ms) => ms,
            None => {
                return graph_error(
                    StatusCode::BAD_REQUEST,
                    "`as_of` must be an epoch-ms number or an RFC3339 string",
                );
            }
        },
    };
    let top = (params.top.unwrap_or(10) as usize).clamp(1, 50);
    let interval = params.histogram_interval.as_deref().unwrap_or("day");
    if !matches!(interval, "day" | "hour") {
        return graph_error(
            StatusCode::BAD_REQUEST,
            "`histogram_interval` must be 'day' or 'hour'",
        );
    }

    let edges = edges_index(&brain);
    if state.engine.get_index(&edges).is_err() {
        // Same body the data plane returns for an unknown brain (the SPA
        // treats `exists: false` as a valid answer, not an error) — and the
        // same body a role-refused read got above.
        return (
            StatusCode::NOT_FOUND,
            Json(json!({ "brain": brain, "contract": GRAPH_CONTRACT, "exists": false })),
        )
            .into_response();
    }

    // 1. Totals: every edge ever asserted (`exists src` excludes the meta doc).
    let totals_body = json!({
        "query": { "exists": { "field": "src" } },
        "size": 0,
        "track_total_hits": true,
    });
    let totals = match run_search(&state.engine, &edges, &totals_body).await {
        Ok(v) => v,
        Err(e) => return graph_error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    };
    let total = totals.total.value;

    // 2. Live slice at as_of, with the type/detector/hub breakdowns.
    let live_body = json!({
        "query": {
            "bool": {
                "filter": [
                    { "exists": { "field": "src" } },
                    { "range": { "valid_at": { "lte": as_of } } }
                ],
                "must_not": [
                    { "range": { "invalid_at": { "lte": as_of } } }
                ]
            }
        },
        "size": 0,
        "track_total_hits": true,
        "aggs": {
            "by_type":     { "terms": { "field": "type",     "size": top } },
            "by_detector": { "terms": { "field": "detector", "size": top } },
            "top_src":     { "terms": { "field": "src",      "size": top } },
            "top_dst":     { "terms": { "field": "dst",      "size": top } },
        },
    });
    let live_resp = match run_search(&state.engine, &edges, &live_body).await {
        Ok(v) => v,
        Err(e) => return graph_error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    };
    let live = live_resp.total.value;
    let aggs = live_resp.aggs.clone().unwrap_or(Value::Null);
    let (types, types_other) = terms_list(&aggs, "by_type", "type", "live");
    let (detectors, detectors_other) = terms_list(&aggs, "by_detector", "detector", "live");
    let (hubs_out, hubs_out_other) = terms_list(&aggs, "top_src", "id", "live_edges");
    let (hubs_in, hubs_in_other) = terms_list(&aggs, "top_dst", "id", "live_edges");

    // 3. Created-over-time histogram (all asserted edges, not just live —
    // the timeline shows assertion activity, invalidation does not erase it).
    let timeline_body = json!({
        "query": { "exists": { "field": "src" } },
        "size": 0,
        "aggs": {
            "created": {
                "date_histogram": { "field": "created_at", "calendar_interval": interval }
            }
        },
    });
    let timeline = match run_search(&state.engine, &edges, &timeline_body).await {
        Ok(v) => v,
        Err(e) => return graph_error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    };
    let created_over_time: Vec<Value> = timeline
        .aggs
        .as_ref()
        .and_then(|a| a.pointer("/created/buckets"))
        .and_then(Value::as_array)
        .map(|bs| {
            bs.iter()
                .map(|b| {
                    json!({
                        "t": b.get("key").cloned().unwrap_or(Value::Null),
                        "count": b.get("doc_count").cloned().unwrap_or(json!(0)),
                    })
                })
                .collect()
        })
        .unwrap_or_default();

    let nodes_index = resolve_nodes_index(&state.engine, &brain, &edges).await;

    // 4. Notes total: one size-0 count per concrete nodes-index name. A brain
    // whose nodes index was never created (edges asserted through the API
    // alone) truthfully has 0 stored notes — reported as such, never
    // fabricated from edge endpoints.
    let mut nodes_total: u64 = 0;
    for name in nodes_index_names(&nodes_index) {
        if state.engine.get_index(name).is_err() {
            continue;
        }
        let count_body = json!({
            "query": { "match_all": {} },
            "size": 0,
            "track_total_hits": true,
        });
        if let Ok(r) = run_search(&state.engine, name, &count_body).await {
            nodes_total = nodes_total.saturating_add(r.total.value);
        }
    }

    Json(json!({
        "brain": brain,
        "contract": GRAPH_CONTRACT,
        "exists": true,
        "as_of": as_of,
        "nodes_index": nodes_index,
        "nodes": { "total": nodes_total },
        "embedder": embedder_id(&state.engine),
        "edges": {
            "total": total,
            "live": live,
            "invalidated": total.saturating_sub(live),
        },
        "types": types,
        "detectors": detectors,
        "hubs": { "out": hubs_out, "in": hubs_in },
        "created_over_time": created_over_time,
        "not_shown": {
            "types_not_listed": types_other,
            "detectors_not_listed": detectors_other,
            "hubs_out_not_listed": hubs_out_other,
            "hubs_in_not_listed": hubs_in_other,
        }
    }))
    .into_response()
}

// ─────────────────────────────────────────────────────────────────────────────
// POST /{brain}/edges/_search — the brain's edges index, ES wire shape
// ─────────────────────────────────────────────────────────────────────────────

/// Run an ES-shaped `_search` against one index and answer in the ES wire
/// shape (the mapping `data_sources::search` already established for
/// session-authenticated searches).
async fn es_wire_search(engine: &Engine, index: &str, body: &Value) -> Response {
    let result = match run_search(engine, index, body).await {
        Ok(r) => r,
        Err(e) => return e.into_response(),
    };
    let hits: Vec<Value> = result
        .hits
        .iter()
        .map(|h| {
            json!({
                "_index": index,
                "_id": h.id,
                "_score": h.score,
                "_source": h.source,
            })
        })
        .collect();
    Json(json!({
        "took": result.took_ms,
        "timed_out": result.timed_out,
        "hits": {
            "total": { "value": result.total.value, "relation": result.total.relation },
            "max_score": result.max_score,
            "hits": hits,
        },
        "aggregations": result.aggs,
    }))
    .into_response()
}

/// `POST /_xerj-console/api/v1/graph/{brain}/edges/_search` — an ES `_search`
/// against exactly this brain's edges index, for the dashboard's
/// recent-retirements and file-type-crossing reads (both used to go direct to
/// `/.xerj-memory-{brain}-edges/_search`, which 401s for a session). The
/// index is pinned server-side — the caller cannot name another index — so
/// this opens the reserved namespace only through the same role gate as every
/// other console graph read, and the generic data-sources proxy keeps
/// refusing it (RC10 B1 unchanged).
pub async fn edges_search(
    State(state): State<ConsoleState>,
    sess: AuthSession,
    Path(brain): Path<String>,
    Json(body): Json<Value>,
) -> Response {
    if let Some(resp) = gate_brain(&sess.user, &brain) {
        return resp;
    }
    if state.engine.get_index(&edges_index(&brain)).is_err() {
        return brain_not_found(&brain);
    }
    es_wire_search(&state.engine, &edges_index(&brain), &body).await
}

// ─────────────────────────────────────────────────────────────────────────────
// POST /{brain}/nodes/_search — one of the brain's nodes indices
// ─────────────────────────────────────────────────────────────────────────────

/// Query params for `POST /graph/{brain}/nodes/_search`.
#[derive(Debug, Default, Deserialize)]
pub struct NodesSearchParams {
    /// WHICH of the brain's nodes indices to search — mandatory when the
    /// meta doc records more than one (`xerj brain` over a multi-dataset
    /// folder writes `"ax-mail,ax-pdfs"`), ignored-correct when it records
    /// exactly one. Must be one of the brain's own names; anything else is
    /// the same 404 an unknown brain gets.
    #[serde(default)]
    pub index: Option<String>,
}

/// `POST /_xerj-console/api/v1/graph/{brain}/nodes/_search` — an ES `_search`
/// against ONE of the brain's nodes indices, resolved from the brain's own
/// meta doc (never caller-chosen): the dashboard's name hydration, notes
/// tally and FIND reads. Multi-index brains fan out client-side (the SPA
/// merges the per-index answers), so no multi-index merge semantics are
/// invented here.
pub async fn nodes_search(
    State(state): State<ConsoleState>,
    sess: AuthSession,
    Path(brain): Path<String>,
    Query(params): Query<NodesSearchParams>,
    Json(body): Json<Value>,
) -> Response {
    if let Some(resp) = gate_brain(&sess.user, &brain) {
        return resp;
    }
    let edges = edges_index(&brain);
    if state.engine.get_index(&edges).is_err() {
        return brain_not_found(&brain);
    }
    let nodes_index = resolve_nodes_index(&state.engine, &brain, &edges).await;
    let names: Vec<&str> = nodes_index_names(&nodes_index).collect();
    let target = match params.index.as_deref() {
        Some(want) => {
            if !names.contains(&want) {
                return brain_not_found(&brain);
            }
            want
        }
        None => match names.as_slice() {
            [only] => only,
            _ => {
                return graph_error(
                    StatusCode::BAD_REQUEST,
                    format!(
                        "this brain's nodes_index spans {} indices; pass ?index= with one of: {}",
                        names.len(),
                        names.join(", ")
                    ),
                );
            }
        },
    };
    if state.engine.get_index(target).is_err() {
        // A brain can hold edges before its nodes index is created — that is
        // an honest empty answer, not a missing brain.
        return es_wire_search_empty_ok();
    }
    es_wire_search(&state.engine, target, &body).await
}

/// An empty ES `_search` response for a nodes index that does not exist
/// (edges asserted through the API alone): zero hits, honestly.
fn es_wire_search_empty_ok() -> Response {
    Json(json!({
        "took": 0,
        "timed_out": false,
        "hits": {
            "total": { "value": 0, "relation": "eq" },
            "max_score": null,
            "hits": [],
        },
        "aggregations": {},
    }))
    .into_response()
}

#[cfg(test)]
mod brain_name_tests {
    use super::validate_brain;

    /// The console's copy of the brain-name rule agrees with `xerj_api::graph_api` and `xerj_autoindex::detect`:
    /// `-edges` always, `-facts` in builds with the facts store (brain `kb-facts`'s default nodes index is facts
    /// namespace `kb`'s index).
    #[test]
    fn brain_names_reserve_the_edges_and_facts_suffixes() {
        assert!(validate_brain("notes").is_ok());
        assert_eq!(
            validate_brain("kb-edges").unwrap_err(),
            "namespace suffix '-edges' is reserved for graph edge indices"
        );
        let r = validate_brain("kb-facts");
        if cfg!(feature = "facts") {
            assert_eq!(
                r.unwrap_err(),
                "namespace suffix '-facts' is reserved for facts indices"
            );
        } else {
            assert!(r.is_ok(), "{r:?}");
        }
        assert!(validate_brain("kb-factsheet").is_ok());
    }
}
