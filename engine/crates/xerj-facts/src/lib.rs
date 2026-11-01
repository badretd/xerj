//! SPIKE (local branch `spike/facts-68m`, never pushed): the XERJ-R facts store, exact executor and learned compiler inside
//! the XERJ engine. This is step 0 of `research/product/XERJ-R_engine_integration_design_2026-10-26.md`. Its numbers are demo
//! numbers: the E2 parity gate and the PR sequence of that design are not claimed here.
//!
//! * **System of record**: the product index `.xerj-memory-{ns}-facts`, written by `xerj-api/src/facts_api.rs` in the record
//!   layout of design §2.2: a `doc` document per source version (its text), a `fact` document per fact (the canonical fields
//!   for the ES DSL plus the exact write input), and `epoch` documents that hold op headers only (op, revision, payload key;
//!   at most [`EPOCH_OPS`] per epoch). An epoch's presence is its commit.
//! * **Projection**: xerj_mem's `Memory` (the reference store, linker and manifests), rebuilt by replaying the committed op
//!   headers in revision order with the payloads of the `doc` / `fact` documents, through the same write API that first
//!   applied them ([`Recorder`]). Every op takes exactly one revision and the replay checks each one, so revisions and exact
//!   fingerprints are the reference's. It changes only through [`Namespace::apply`].
//! * **Compiler**: an xerj-infer export directory (`XERJ_FACTS_MODEL_DIR`), loaded lazily on the first ask. Its text encoder
//!   is the engine's own `tokenizers` 0.21; the export's tokenizer.json is identified by its sha256.
//!
//! * **Commit and recovery** (design §2.3 / §2.5): a batch commits when the last of its epoch documents is written; each
//!   epoch carries the batch's last revision (`batch_to_rev`), so a batch split over several epochs is committed whole or not
//!   at all. After the markers, the older records the batch retired, hid or withdrew get their repair-owned fields
//!   (`retired_rev`, `vis_end_rev`, `withdrawn_rev`) re-indexed (step 5). A rebuild ([`Namespace::recover`]) replays the
//!   committed chain, then lists what to clean up: the records and epochs of batches that never committed (orphans of a
//!   failure or a process kill before the markers) and the committed records whose repair-owned fields a kill left stale.
//!   The executor reads only the projection; the fields serve raw product queries over the records.
//!
//! Spike limitations, stated rather than hidden:
//! - One writer per process. No cluster mode.

mod casefold;
pub mod failpoint;
mod gate;
mod tok;

pub use casefold::{
    casefold_link, casefold_link_with, casefold_v3_link, is_function_word, CaseLink, CaseMatch,
    SharedLink, FUNCTION_WORDS,
};
pub use gate::{
    AskGate, GateError, StopFlag, ENV_ASK_DEADLINE_MS, ENV_MAX_CONCURRENT_ASKS, ENV_MAX_QUEUED_ASKS,
};
pub use tok::EngineTokenizer;
/// The research crates behind the projection, for the E2 parity harness (`xerj-api/tests/facts_e2_parity.rs`), which drives
/// the product path and must compare its rebuilt projection with the same reference outputs the research path reproduces.
pub use {xerj_exec, xerj_index, xerj_infer, xerj_serve, xerj_store};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use smallvec::SmallVec;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, RwLock};
use std::time::Instant;
use xerj_index::{rel_phrase, Memory, MemoryConfig};
use xerj_infer::Precision;
use xerj_serve::adapter::kind_name;
/// The `_ask` option `verify` ([`AskReq::verify`]): which rule judges an ANSWER.
pub use xerj_serve::engine::VerifyRule;
use xerj_serve::engine::{
    answer_stoppable_opts, lab_item_id, AnswerOpts, LinkerMode, ModelSlot, SpeedOptions,
};
use xerj_store::value::render_value;
use xerj_store::{
    ingest_world, AddedFact, ContractError, DocIx, FactIn, RecId, Rev, Status, Store, ValueIn,
    World, WriteSink, DEFAULT_SCOPE, OPEN,
};

/// 3: records carry the repair-owned fields (`retired_rev`, `vis_end_rev`, `withdrawn_rev`), and records and epochs the last
/// revision of their batch (`batch_to_rev`). 2: epochs hold op headers, payloads live in the `doc` / `fact` documents (1 held whole
/// batches). Older records are read as they are: a missing field is open, an epoch without `batch_to_rev` is its own batch.
pub const SCHEMA_VERSION: u32 = 3;
/// Op headers per `epoch` document at most (design §2.3). Keeps every epoch far below the engine's 8 MiB limit for one indexed
/// document, whatever the batch size.
pub const EPOCH_OPS: usize = 10_000;

/// The product index that holds one namespace's facts (inside the reserved `.xerj-memory-` prefix).
pub fn facts_index(ns: &str) -> String {
    format!(".xerj-memory-{ns}-facts")
}

/// Mapping of the facts index. `ops`, the fact `input` and the document `text` are stored and not indexed.
pub fn facts_mapping() -> Value {
    json!({"mappings": {"properties": {
        "rt": {"type": "keyword"},
        "fact_id": {"type": "keyword"}, "subject": {"type": "keyword"}, "relation": {"type": "keyword"},
        "rel_phrase": {"type": "keyword"}, "kind": {"type": "keyword"}, "value": {"type": "keyword"},
        "rendered": {"type": "keyword"}, "unit": {"type": "keyword"}, "scope": {"type": "keyword"},
        "quote": {"type": "text"}, "subject_id": {"type": "keyword"}, "value_id": {"type": "keyword"},
        "writer_version": {"type": "keyword"}, "doc_id": {"type": "keyword"}, "version": {"type": "keyword"},
        "supersedes": {"type": "keyword"}, "added_rev": {"type": "long"},
        "input": {"type": "text", "index": false}, "text": {"type": "text", "index": false},
        "from_rev": {"type": "long"}, "to_rev": {"type": "long"},
        "ops": {"type": "text", "index": false}, "committed_at_ms": {"type": "long"},
        "batch_to_rev": {"type": "long"}, "retired_rev": {"type": "long"},
        "vis_end_rev": {"type": "long"}, "withdrawn_rev": {"type": "long"},
        "schema_version": {"type": "long"}
    }}})
}

/// `_id` of the `doc` record of one source version.
pub fn doc_record_id(doc_id: &str, version: &str) -> String {
    format!("d:{doc_id}@{version}")
}

/// `_id` of the `fact` record of one fact.
pub fn fact_record_id(fact_id: &str) -> String {
    format!("f:{fact_id}")
}

/// `_id` of the `epoch` record that ends at `to_rev`.
pub fn epoch_record_id(to_rev: Rev) -> String {
    format!("e:{to_rev:010}")
}

fn open_or(rev: Rev) -> Option<Rev> {
    (rev != OPEN).then_some(rev)
}

/// The repair-owned fields of a `fact` record at the store's head (design §2.2): the revision that retired it (superseded or
/// withdrawn) and the one that ended its visibility (retired, or its source withdrawn); `null` = open.
pub fn fact_fields(st: &Store, r: RecId) -> Value {
    let i = r as usize;
    json!({"retired_rev": open_or(st.recs.retired[i]), "vis_end_rev": open_or(st.recs.vis_end[i])})
}

/// The repair-owned field of a `doc` record at the store's head: the revision that withdrew it; `null` = open.
pub fn doc_fields(st: &Store, d: DocIx) -> Value {
    json!({"withdrawn_rev": open_or(st.docs.withdrawn[d as usize])})
}

/// A revision field of a stored record: missing or `null` = open.
fn stored_rev(src: &Value, k: &str) -> Result<Option<Rev>, String> {
    match src.get(k) {
        None | Some(Value::Null) => Ok(None),
        Some(v) => v
            .as_u64()
            .filter(|&r| r < OPEN as u64)
            .map(|r| Some(r as Rev))
            .ok_or_else(|| format!("{k} is not a revision: {v}")),
    }
}

/// One write batch as the API receives it.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum Batch {
    /// A memory-world-v1 batch: base documents, base facts, then edits, in the reference op order. `items` are dropped.
    World {
        world: Value,
    },
    WithdrawFact {
        fact_id: String,
    },
}

/// One op header of an `epoch` document (design §2.2 `ops`): the op, the revision it took and the key of its payload. The
/// payloads (document text, fact write input) live in the `doc` / `fact` documents.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum OpHeader {
    AddDocument {
        rev: Rev,
        doc_id: String,
        version: String,
    },
    AddFact {
        rev: Rev,
        fact_id: String,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        supersedes: Vec<String>,
    },
    WithdrawFact {
        rev: Rev,
        fact_id: String,
    },
    WithdrawDocument {
        rev: Rev,
        doc_id: String,
        version: String,
    },
}

impl OpHeader {
    pub fn rev(&self) -> Rev {
        match self {
            OpHeader::AddDocument { rev, .. }
            | OpHeader::AddFact { rev, .. }
            | OpHeader::WithdrawFact { rev, .. }
            | OpHeader::WithdrawDocument { rev, .. } => *rev,
        }
    }
}

/// The exact write input of one fact (every `FactIn` field), as stored in its `fact` document. The replay feeds it back to
/// `add_fact`, so the store canonicalizes the same input the same way.
pub fn fact_input_json(f: &FactIn<'_>) -> Value {
    let value = match f.value {
        ValueIn::Str(s) => json!(s),
        ValueIn::Bool(b) => json!(b),
        ValueIn::Other => Value::Null,
    };
    json!({"fact_id": f.fact_id, "subject": f.subject, "relation": f.relation, "kind": f.kind, "value": value, "unit": f.unit,
           "doc_id": f.doc_id, "version": f.version, "start": f.start, "end": f.end, "quote": f.quote, "scope": f.scope,
           "subject_id": f.subject_id, "value_id": f.value_id, "valid_from": f.valid_from, "valid_to": f.valid_to,
           "writer_version": f.writer_version})
}

fn input_str<'a>(v: &'a Value, k: &str) -> Result<&'a str, String> {
    v.get(k)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("fact input without {k}"))
}

fn input_i64(v: &Value, k: &str) -> Result<i64, String> {
    v.get(k)
        .and_then(Value::as_i64)
        .ok_or_else(|| format!("fact input without {k}"))
}

/// The `FactIn` of a stored write input ([`fact_input_json`]).
pub fn fact_input_from_json(v: &Value) -> Result<FactIn<'_>, String> {
    let value = match v.get("value") {
        Some(Value::String(s)) => ValueIn::Str(s),
        Some(Value::Bool(b)) => ValueIn::Bool(*b),
        _ => ValueIn::Other,
    };
    Ok(FactIn {
        fact_id: input_str(v, "fact_id")?,
        subject: input_str(v, "subject")?,
        relation: input_str(v, "relation")?,
        kind: input_str(v, "kind")?,
        value,
        unit: input_str(v, "unit")?,
        doc_id: input_str(v, "doc_id")?,
        version: input_str(v, "version")?,
        start: input_i64(v, "start")?,
        end: input_i64(v, "end")?,
        quote: input_str(v, "quote")?,
        scope: input_str(v, "scope")?,
        subject_id: v.get("subject_id").and_then(Value::as_str),
        value_id: v.get("value_id").and_then(Value::as_str),
        valid_from: v.get("valid_from").and_then(Value::as_str),
        valid_to: v.get("valid_to").and_then(Value::as_str),
        writer_version: input_str(v, "writer_version")?,
    })
}

/// The reference write API with every successful op recorded as an op header. A failing op mutates nothing and records
/// nothing. `_world` drives `ingest_world` through it, so the op order is the reference's.
pub struct Recorder<'m> {
    pub mem: &'m mut Memory,
    pub ops: Vec<OpHeader>,
    /// per `add_fact`: the new record and its write input ([`fact_input_json`])
    pub facts: Vec<(RecId, Value)>,
}

impl WriteSink for Recorder<'_> {
    fn sink_add_document(
        &mut self,
        doc_id: &str,
        version: &str,
        text: &str,
    ) -> Result<DocIx, ContractError> {
        let d = self.mem.add_document(doc_id, version, text)?;
        self.ops.push(OpHeader::AddDocument {
            rev: self.mem.store.revision(),
            doc_id: doc_id.into(),
            version: version.into(),
        });
        Ok(d)
    }
    fn sink_add_fact(
        &mut self,
        f: &FactIn<'_>,
        supersedes: &[&str],
    ) -> Result<AddedFact, ContractError> {
        let a = self.mem.add_fact(f, supersedes)?;
        self.ops.push(OpHeader::AddFact {
            rev: a.rev,
            fact_id: f.fact_id.into(),
            supersedes: supersedes.iter().map(|s| s.to_string()).collect(),
        });
        self.facts.push((a.rec, fact_input_json(f)));
        Ok(a)
    }
    fn sink_withdraw_fact(&mut self, fact_id: &str) -> Result<(RecId, bool), ContractError> {
        let r = self.mem.withdraw_fact(fact_id)?;
        self.ops.push(OpHeader::WithdrawFact {
            rev: self.mem.store.revision(),
            fact_id: fact_id.into(),
        });
        Ok(r)
    }
    fn sink_withdraw_document(
        &mut self,
        doc_id: &str,
        version: &str,
    ) -> Result<(DocIx, SmallVec<[RecId; 8]>), ContractError> {
        let r = self.mem.withdraw_document(doc_id, version)?;
        self.ops.push(OpHeader::WithdrawDocument {
            rev: self.mem.store.revision(),
            doc_id: doc_id.into(),
            version: version.into(),
        });
        Ok(r)
    }
    fn sink_revision(&self) -> Rev {
        self.mem.store.revision()
    }
}

/// What one applied batch produced.
pub struct Applied {
    /// first revision taken by the batch (`to_rev < from_rev` means the batch was empty)
    pub from_rev: Rev,
    pub to_rev: Rev,
    /// `World`: the revision after the base documents and facts (before the edits)
    pub base_rev: Option<Rev>,
    /// `(_id, _source)` of the `doc` / `fact` documents the batch created
    pub records: Vec<(String, Value)>,
    /// `(_id, _source)` of the batch's `epoch` documents (at most `epoch_ops` op headers each) in revision order; the
    /// caller adds `committed_at_ms`
    pub epochs: Vec<(String, Value)>,
    /// `(_id, partial source)` of the step-5 repairs: the repair-owned fields of the records older than the batch that it
    /// retired, hid or withdrew, at the batch's last revision (an idempotent partial update)
    pub repairs: Vec<(String, Value)>,
    pub apply_ms: f64,
}

/// One `epoch` document: the op headers of revisions `from ..= to`, part of the batch that ends at `batch_to`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EpochRec {
    pub from: Rev,
    pub to: Rev,
    /// the last revision of the epoch's batch; the batch commits only with all of its epochs (= `to` before schema 3)
    pub batch_to: Rev,
    pub ops: Vec<OpHeader>,
}

/// The [`EpochRec`] of an `epoch` document source; `None` for the other record types.
pub fn parse_epoch(src: &Value) -> Result<Option<EpochRec>, String> {
    if src.get("rt").and_then(Value::as_str) != Some("epoch") {
        return Ok(None);
    }
    let (Some(from), Some(to), Some(ops)) = (
        src["from_rev"].as_u64(),
        src["to_rev"].as_u64(),
        src["ops"].as_str(),
    ) else {
        return Err(format!(
            "epoch document without from_rev / to_rev / ops: {}",
            src["to_rev"]
        ));
    };
    let batch_to = src
        .get("batch_to_rev")
        .and_then(Value::as_u64)
        .unwrap_or(to);
    if from == 0 || to < from || batch_to < to || batch_to >= OPEN as u64 {
        return Err(format!(
            "epoch {from}..{to} with batch_to_rev {batch_to}: not a revision range"
        ));
    }
    let ops: Vec<OpHeader> =
        serde_json::from_str(ops).map_err(|e| format!("epoch {from}..{to}: {e}"))?;
    Ok(Some(EpochRec {
        from: from as Rev,
        to: to as Rev,
        batch_to: batch_to as Rev,
        ops,
    }))
}

/// A `doc` / `fact` record as the walk saw it: its key, the revision that wrote it and its stored repair-owned fields.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Seen {
    Doc {
        doc_id: String,
        version: String,
        added: Option<Rev>,
        batch_to: Option<Rev>,
        withdrawn: Option<Rev>,
    },
    Fact {
        fact_id: String,
        added: Option<Rev>,
        batch_to: Option<Rev>,
        retired: Option<Rev>,
        vis_end: Option<Rev>,
    },
}

/// The payloads of the `doc` / `fact` documents, keyed as the op headers name them.
#[derive(Default)]
pub struct Payloads {
    /// (doc_id, version) -> text
    pub docs: HashMap<(String, String), String>,
    /// fact_id -> write input ([`fact_input_json`]) as its stored JSON text. The replay parses it when it applies the
    /// fact. A parsed `Value` per fact (17 keys) held about 2 KB more than the text, which is what made a rebuild of
    /// 6.9 million facts outgrow a 26 GB memory limit.
    pub facts: HashMap<String, Box<str>>,
    /// every `doc` / `fact` record walked, for the recovery's cleanup and repair ([`Namespace::recover`])
    pub seen: Vec<Seen>,
}

impl Payloads {
    /// Take the payload of one `doc` / `fact` document source; other record types are ignored.
    pub fn add_source(&mut self, src: &Value) -> Result<(), String> {
        match src.get("rt").and_then(Value::as_str) {
            Some("doc") => {
                let (Some(d), Some(v), Some(t)) = (
                    src["doc_id"].as_str(),
                    src["version"].as_str(),
                    src["text"].as_str(),
                ) else {
                    return Err(format!(
                        "doc document without doc_id / version / text: {}",
                        src["doc_id"]
                    ));
                };
                self.docs
                    .insert((d.to_string(), v.to_string()), t.to_string());
                self.seen.push(Seen::Doc {
                    doc_id: d.to_string(),
                    version: v.to_string(),
                    added: stored_rev(src, "added_rev").map_err(|e| format!("doc {d}@{v}: {e}"))?,
                    batch_to: stored_rev(src, "batch_to_rev")
                        .map_err(|e| format!("doc {d}@{v}: {e}"))?,
                    withdrawn: stored_rev(src, "withdrawn_rev")
                        .map_err(|e| format!("doc {d}@{v}: {e}"))?,
                });
            }
            Some("fact") => {
                let (Some(id), Some(input)) = (src["fact_id"].as_str(), src["input"].as_str())
                else {
                    return Err(format!(
                        "fact document without fact_id / input: {}",
                        src["fact_id"]
                    ));
                };
                // A stored input that does not parse is a corrupt record, as before; only the text is kept.
                serde_json::from_str::<Value>(input)
                    .map_err(|e| format!("fact {id}: input: {e}"))?;
                self.facts.insert(id.to_string(), input.into());
                self.seen.push(Seen::Fact {
                    fact_id: id.to_string(),
                    added: stored_rev(src, "added_rev").map_err(|e| format!("fact {id}: {e}"))?,
                    batch_to: stored_rev(src, "batch_to_rev")
                        .map_err(|e| format!("fact {id}: {e}"))?,
                    retired: stored_rev(src, "retired_rev")
                        .map_err(|e| format!("fact {id}: {e}"))?,
                    vis_end: stored_rev(src, "vis_end_rev")
                        .map_err(|e| format!("fact {id}: {e}"))?,
                });
            }
            _ => {}
        }
        Ok(())
    }
}

/// The cleanup a rebuild asks for (design §2.5 steps 2 and 4): the uncommitted records and epochs to delete and the
/// committed records whose repair-owned fields to re-index. Idempotent; a rebuild after any crash derives it again.
#[derive(Debug, Default)]
pub struct Cleanup {
    /// `_id`s to delete
    pub orphans: Vec<String>,
    /// `(_id, partial source)` to merge
    pub repairs: Vec<(String, Value)>,
}

/// One namespace's projection: the reference memory plus the committed (published) revision.
pub struct Namespace {
    pub mem: RwLock<Memory>,
    /// The highest committed revision. Readers default to it; uncommitted revisions are never served.
    pub published: AtomicU32,
    /// committed `epoch` documents
    pub epochs: AtomicU32,
    /// A cleanup the rebuild could not finish (the engine refused its deletes or updates, e.g. under its memory
    /// breaker). The replayed projection holds exactly the committed chain and the executor never reads orphans or
    /// repair-owned fields, so it serves reads meanwhile (design §2.3: "marked `repairing` until repairs finish (before
    /// the next write)"); the next write must finish the cleanup first.
    pub pending_cleanup: Mutex<Option<Cleanup>>,
}

impl Default for Namespace {
    fn default() -> Self {
        Self::new()
    }
}

impl Namespace {
    pub fn new() -> Namespace {
        // BM25 off: the engine's own full-text index already searches the `fact` documents.
        let cfg = MemoryConfig {
            bm25_records: false,
            bm25_documents: false,
            ..MemoryConfig::default()
        };
        Namespace {
            mem: RwLock::new(Memory::new(cfg)),
            published: AtomicU32::new(0),
            epochs: AtomicU32::new(0),
            pending_cleanup: Mutex::new(None),
        }
    }

    /// Apply one batch to the projection, with the contract checks in the reference order. On a contract error the projection
    /// may hold a prefix of the batch; the caller must then discard it, and the next access rebuilds it from the committed
    /// epochs.
    pub fn apply(&self, batch: &Batch) -> Result<Applied, String> {
        self.apply_chunked(batch, failpoint::epoch_ops(EPOCH_OPS))
    }

    /// [`Namespace::apply`] with at most `epoch_ops` op headers per epoch document.
    pub fn apply_chunked(&self, batch: &Batch, epoch_ops: usize) -> Result<Applied, String> {
        let t0 = Instant::now();
        let mut guard = self
            .mem
            .write()
            .map_err(|_| "facts projection lock poisoned".to_string())?;
        let mem: &mut Memory = &mut guard;
        let rev0 = mem.store.revision();
        let docs0 = mem.store.num_docs() as DocIx;
        let mut rec = Recorder {
            mem,
            ops: Vec::new(),
            facts: Vec::new(),
        };
        let base_rev = match batch {
            Batch::World { world } => {
                let mut w = world.clone();
                if let Some(o) = w.as_object_mut() {
                    o.remove("items");
                }
                let s = serde_json::to_string(&w).map_err(|e| e.to_string())?;
                let parsed = World::parse(&s).map_err(|e| format!("world: {e}"))?;
                Some(
                    ingest_world(&mut rec, &parsed)
                        .map_err(|e| e.to_string())?
                        .base,
                )
            }
            Batch::WithdrawFact { fact_id } => {
                rec.sink_withdraw_fact(fact_id).map_err(|e| e.to_string())?;
                None
            }
        };
        let Recorder { mem, ops, facts } = rec;
        let to_rev = mem.store.revision();
        let st = &mem.store;
        let mut records = Vec::with_capacity(ops.len());
        // add_document appends documents in op order: the k-th AddDocument header is document docs0 + k
        let mut d = docs0;
        for h in &ops {
            if let OpHeader::AddDocument {
                rev,
                doc_id,
                version,
            } = h
            {
                let mut src = json!({"rt": "doc", "doc_id": doc_id, "version": version, "text": st.doc_text(d),
                                     "added_rev": rev, "batch_to_rev": to_rev, "schema_version": SCHEMA_VERSION});
                merge_fields(&mut src, doc_fields(st, d));
                records.push((doc_record_id(doc_id, version), src));
                d += 1;
            }
        }
        let fact_ops = ops.iter().filter_map(|h| match h {
            OpHeader::AddFact {
                rev, supersedes, ..
            } => Some((*rev, supersedes)),
            _ => None,
        });
        for ((rev, supersedes), (r, input)) in fact_ops.zip(&facts) {
            let r = *r;
            let v = st.typed_value(r);
            let (doc_id, version) = st.rec_doc_key(r);
            let mut src = json!({"rt": "fact", "fact_id": st.fact_id(r), "subject": st.subject(r), "relation": st.relation(r),
                       "rel_phrase": rel_phrase(st.relation(r)), "kind": kind_name(v.kind), "value": v.value, "unit": v.unit,
                       "rendered": render_value(&v), "scope": st.scope(r), "quote": st.quote(r),
                       "subject_id": st.subject_id(r), "value_id": st.value_id(r), "writer_version": st.writer_version(r),
                       "doc_id": doc_id, "version": version, "supersedes": supersedes, "added_rev": rev,
                       "batch_to_rev": to_rev, "input": input.to_string(), "schema_version": SCHEMA_VERSION});
            merge_fields(&mut src, fact_fields(st, r));
            records.push((fact_record_id(st.fact_id(r)), src));
        }
        let mut epochs = Vec::new();
        for c in ops.chunks(epoch_ops.max(1)) {
            let (from, to) = (c[0].rev(), c[c.len() - 1].rev());
            let headers = serde_json::to_string(c).map_err(|e| e.to_string())?;
            epochs.push((
                epoch_record_id(to),
                json!({"rt": "epoch", "from_rev": from, "to_rev": to, "batch_to_rev": to_rev, "ops": headers,
                       "schema_version": SCHEMA_VERSION}),
            ));
        }
        let repairs = step5_repairs(st, &ops, rev0 + 1);
        Ok(Applied {
            from_rev: rev0 + 1,
            to_rev,
            base_rev,
            records,
            epochs,
            repairs,
            apply_ms: t0.elapsed().as_secs_f64() * 1e3,
        })
    }

    /// Rebuild a projection from the walked epochs and the payloads of the `doc` / `fact` documents: replay the committed
    /// chain ([`committed_chain`]) through the reference write API. Every op must take the revision its header recorded.
    /// Returns the projection and the number of epochs outside the chain (not committed).
    pub fn replay(epochs: &[EpochRec], p: &Payloads) -> Result<(Namespace, usize), String> {
        let chain = committed_chain(epochs);
        let ns = Namespace::replay_chain(epochs, &chain, p)?;
        Ok((ns, epochs.len() - chain.len()))
    }

    fn replay_chain(
        epochs: &[EpochRec],
        chain: &[usize],
        p: &Payloads,
    ) -> Result<Namespace, String> {
        let ns = Namespace::new();
        let mut head: Rev = 0;
        {
            let mut guard = ns
                .mem
                .write()
                .map_err(|_| "facts projection lock poisoned".to_string())?;
            let mem: &mut Memory = &mut guard;
            for &i in chain {
                let EpochRec { from, to, ops, .. } = &epochs[i];
                debug_assert_eq!(*from, head + 1);
                for h in ops {
                    let r = match h {
                        OpHeader::AddDocument {
                            doc_id, version, ..
                        } => {
                            let text = p
                                .docs
                                .get(&(doc_id.clone(), version.clone()))
                                .ok_or_else(|| format!("epoch {from}..{to}: no doc document for {doc_id}@{version}"))?;
                            mem.add_document(doc_id, version, text).map(|_| ())
                        }
                        OpHeader::AddFact {
                            fact_id,
                            supersedes,
                            ..
                        } => {
                            let input = p.facts.get(fact_id).ok_or_else(|| {
                                format!("epoch {from}..{to}: no fact document for {fact_id}")
                            })?;
                            // the text parsed when it was walked (`Payloads::add_source`)
                            let input: Value = serde_json::from_str(input).map_err(|e| {
                                format!("epoch {from}..{to}: fact {fact_id}: input: {e}")
                            })?;
                            let f = fact_input_from_json(&input)
                                .map_err(|e| format!("epoch {from}..{to}: fact {fact_id}: {e}"))?;
                            let sup: Vec<&str> = supersedes.iter().map(String::as_str).collect();
                            mem.add_fact(&f, &sup).map(|_| ())
                        }
                        OpHeader::WithdrawFact { fact_id, .. } => {
                            mem.withdraw_fact(fact_id).map(|_| ())
                        }
                        OpHeader::WithdrawDocument {
                            doc_id, version, ..
                        } => mem.withdraw_document(doc_id, version).map(|_| ()),
                    };
                    r.map_err(|e| format!("epoch {from}..{to}: replay of {h:?} failed: {e}"))?;
                    let got = mem.store.revision();
                    if got != h.rev() {
                        return Err(format!("epoch {from}..{to}: {h:?} took revision {got}"));
                    }
                }
                if mem.store.revision() != *to {
                    return Err(format!(
                        "replay of epoch {from}..{to} reached revision {}",
                        mem.store.revision()
                    ));
                }
                head = *to;
                ns.epochs.fetch_add(1, Ordering::SeqCst);
            }
        }
        ns.published.store(head, Ordering::SeqCst);
        Ok(ns)
    }

    /// Recovery (design §2.5): [`Namespace::replay`], then what the rebuilt projection says about the walked records.
    ///
    /// The residue past the committed head is the epochs outside the chain that start after the head and the `doc` / `fact`
    /// records written after the head whose key the projection does not hold. A batch that failed or was killed before its
    /// last epoch leaves exactly that: its records and some of its epochs. One writer per namespace, and a cleanup before
    /// every write, leave at most one such batch, so all of its residue names the same `batch_to_rev`. That residue is
    /// returned as `orphans`, to delete before the next write (which also keeps a stale epoch from ever extending the chain
    /// once later batches reach its revisions). Residue of several batches is no crash's: a gap in the chain from a lost
    /// write (power loss) or an index walk that missed a committed epoch. Deleting it could destroy committed writes, so
    /// nothing is deleted and `blocked` says why; the caller must not serve or write the namespace then.
    ///
    /// Records outside the committed state at or below the head cannot come from a crash either (a batch commits whole,
    /// and the cleanup runs before any write); they are harmless to later writes, left in place and reported as
    /// `unexplained`. `repairs` are the committed records whose repair-owned fields differ from the projection's (a kill
    /// between the markers and the end of step 5).
    pub fn recover(epochs: &[EpochRec], p: &Payloads) -> Result<Recovered, String> {
        let chain = committed_chain(epochs);
        let ns = Namespace::replay_chain(epochs, &chain, p)?;
        let head = ns.published.load(Ordering::SeqCst);
        let in_chain: HashSet<usize> = chain.iter().copied().collect();
        // residue past the head, with the batch each item names (`None`: a schema-2 record, which names none)
        let mut residue: Vec<(String, Option<Rev>)> = Vec::new();
        let mut unexplained = Vec::new();
        for (i, e) in epochs.iter().enumerate() {
            if !in_chain.contains(&i) {
                let id = epoch_record_id(e.to);
                if e.from > head {
                    residue.push((id, Some(e.batch_to)));
                } else {
                    unexplained.push(id);
                }
            }
        }
        let mut unknown = |id: String, added: Option<Rev>, batch_to: Option<Rev>| match added {
            Some(r) if r > head => residue.push((id, batch_to)),
            _ => unexplained.push(id),
        };
        let mut repairs = Vec::new();
        {
            let mem = ns
                .mem
                .read()
                .map_err(|_| "facts projection lock poisoned".to_string())?;
            let st = &mem.store;
            for s in &p.seen {
                match s {
                    Seen::Doc {
                        doc_id,
                        version,
                        added,
                        batch_to,
                        withdrawn,
                    } => match st.doc_of(doc_id, version) {
                        None => unknown(doc_record_id(doc_id, version), *added, *batch_to),
                        Some(d) => {
                            if *withdrawn != open_or(st.docs.withdrawn[d as usize]) {
                                repairs.push((doc_record_id(doc_id, version), doc_fields(st, d)));
                            }
                        }
                    },
                    Seen::Fact {
                        fact_id,
                        added,
                        batch_to,
                        retired,
                        vis_end,
                    } => match st.rec_of(fact_id) {
                        None => unknown(fact_record_id(fact_id), *added, *batch_to),
                        Some(r) => {
                            let i = r as usize;
                            if (*retired, *vis_end)
                                != (open_or(st.recs.retired[i]), open_or(st.recs.vis_end[i]))
                            {
                                repairs.push((fact_record_id(fact_id), fact_fields(st, r)));
                            }
                        }
                    },
                }
            }
        }
        let batches: BTreeSet<Rev> = residue.iter().filter_map(|(_, b)| *b).collect();
        let (mut orphans, blocked) = if batches.len() <= 1 {
            (
                residue.into_iter().map(|(id, _)| id).collect::<Vec<_>>(),
                None,
            )
        } else {
            let why = format!(
                "{} records and epochs past the committed head {head} belong to {} batches (batch_to_rev {:?}): \
                 not the residue of one interrupted batch (a lost write or an incomplete index walk); nothing deleted",
                residue.len(),
                batches.len(),
                batches.iter().take(8).collect::<Vec<_>>()
            );
            unexplained.extend(residue.into_iter().map(|(id, _)| id));
            (Vec::new(), Some(why))
        };
        orphans.sort();
        unexplained.sort();
        repairs.sort_by(|a, b| a.0.cmp(&b.0));
        Ok(Recovered {
            ignored: epochs.len() - chain.len(),
            ns,
            orphans,
            unexplained,
            repairs,
            blocked,
        })
    }

    /// Status of the namespace. The exact fingerprint is O(N) and is computed only on request.
    pub fn status(&self, fingerprint: bool) -> Value {
        let Ok(mem) = self.mem.read() else {
            return json!({"error": "facts projection lock poisoned"});
        };
        let published = self.published.load(Ordering::SeqCst);
        let mut out = json!({
            "revision": mem.store.revision(), "published_rev": published, "epochs": self.epochs.load(Ordering::SeqCst),
            "records": mem.store.num_records(), "documents": mem.store.num_docs(), "visible_at_head": mem.store.num_head_visible(),
            "projection_heap_bytes": mem.store.heap_bytes(),
        });
        if fingerprint {
            out["fingerprint_exact"] = json!(mem.snapshot_at(published).fingerprint().to_string());
        }
        if let Ok(p) = self.pending_cleanup.lock() {
            if let Some(c) = p.as_ref() {
                out["cleanup_pending"] =
                    json!({"orphans": c.orphans.len(), "repairs": c.repairs.len()});
            }
        }
        out
    }
}

/// A projection rebuilt by [`Namespace::recover`] and the cleanup it asks for.
pub struct Recovered {
    pub ns: Namespace,
    /// epochs outside the committed chain
    pub ignored: usize,
    /// `_id`s of the uncommitted records to delete (epochs, `doc` and `fact` records), sorted
    pub orphans: Vec<String>,
    /// `_id`s of records outside the committed state that no crash explains: left in place, sorted
    pub unexplained: Vec<String>,
    /// Why the residue past the head is not one interrupted batch's: then nothing may be deleted, and the namespace must
    /// not be served or written until someone looks at it (`None` = recovered)
    pub blocked: Option<String>,
    /// `(_id, partial source)` of the committed records whose repair-owned fields are stale, sorted by `_id`
    pub repairs: Vec<(String, Value)>,
}

/// The committed epochs in revision order (indices into `epochs`): the contiguous chain from revision 1, batch by batch. A
/// batch (`from ..= batch_to`, one or more epochs) is committed only when all of its epochs are present and contiguous. The
/// chain ends at the first revision that no complete batch continues; an epoch of an incomplete batch that starts there
/// does not hide a complete one that starts at the same revision.
pub fn committed_chain(epochs: &[EpochRec]) -> Vec<usize> {
    let mut by_from: HashMap<Rev, Vec<usize>> = HashMap::new();
    for (i, e) in epochs.iter().enumerate() {
        by_from.entry(e.from).or_default().push(i);
    }
    let mut chain = Vec::new();
    let mut head: Rev = 0;
    'batches: while let Some(starts) = by_from.get(&(head + 1)) {
        for &i in starts {
            let bt = epochs[i].batch_to;
            let mut group = vec![i];
            let mut end = epochs[i].to;
            while end < bt {
                let next = by_from
                    .get(&(end + 1))
                    .and_then(|v| v.iter().copied().find(|&j| epochs[j].batch_to == bt));
                let Some(j) = next else { break };
                group.push(j);
                end = epochs[j].to;
            }
            if end == bt {
                chain.extend(group);
                head = bt;
                continue 'batches;
            }
        }
        break;
    }
    chain
}

/// Step 5 of the commit protocol: `(_id, partial source)` with the repair-owned fields of every record older than the batch
/// (added before `from`) that the batch's ops retired (a correction's `supersedes`, `withdraw_fact`), hid (a withdrawn
/// source's facts) or withdrew (`withdraw_document`).
fn step5_repairs(st: &Store, ops: &[OpHeader], from: Rev) -> Vec<(String, Value)> {
    let mut facts = BTreeSet::new();
    let mut docs = BTreeSet::new();
    let fact = |id: &str, facts: &mut BTreeSet<RecId>| {
        if let Some(r) = st.rec_of(id) {
            if st.recs.added[r as usize] < from {
                facts.insert(r);
            }
        }
    };
    for h in ops {
        match h {
            OpHeader::AddFact { supersedes, .. } => {
                for old in supersedes {
                    fact(old, &mut facts);
                }
            }
            OpHeader::WithdrawFact { fact_id, .. } => fact(fact_id, &mut facts),
            OpHeader::WithdrawDocument {
                doc_id, version, ..
            } => {
                if let Some(d) = st.doc_of(doc_id, version) {
                    if st.docs.added[d as usize] < from {
                        docs.insert(d);
                    }
                    for &r in &st.docs.recs[d as usize] {
                        let i = r as usize;
                        if st.recs.added[i] < from
                            && st.recs.vis_end[i] != OPEN
                            && st.recs.vis_end[i] >= from
                        {
                            facts.insert(r);
                        }
                    }
                }
            }
            OpHeader::AddDocument { .. } => {}
        }
    }
    let mut out: Vec<(String, Value)> = docs
        .into_iter()
        .map(|d| {
            let (doc_id, version) = st.doc_key_str(d);
            (doc_record_id(doc_id, version), doc_fields(st, d))
        })
        .chain(
            facts
                .into_iter()
                .map(|r| (fact_record_id(st.fact_id(r)), fact_fields(st, r))),
        )
        .collect();
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

fn merge_fields(src: &mut Value, fields: Value) {
    if let (Some(o), Value::Object(f)) = (src.as_object_mut(), fields) {
        o.extend(f);
    }
}

/// Errors that are operational, never a factual UNKNOWN (plan §5).
#[derive(Debug)]
pub enum FactsError {
    BadRequest(String),
    /// The requested revision is not committed (409).
    NotCommitted(String),
    /// The model directory is missing or broken; sticky (503).
    ModelUnavailable(String),
    /// The caller stopped waiting (deadline or a dropped request) and the compiler stopped after this many decisions:
    /// there is no answer, not even an UNKNOWN.
    Stopped {
        decisions: usize,
    },
    Internal(String),
}

#[derive(Debug, Deserialize)]
pub struct AskReq {
    pub question: String,
    /// snapshot revision; default = the published head
    #[serde(default)]
    pub revision: Option<Rev>,
    /// manifest salt; default "lab/q/" + sha256(question)[:16] (parity runs pass the fixture's item id)
    #[serde(default)]
    pub item_id: Option<String>,
    /// "exact" (the reference, the default) | "tolerant" | "tolerant_words" | "casefold" (case-insensitive names, for large
    /// shared stores; [`casefold_link`]) | "casefold_v3" (casefold + plurals + function-word and case-variant subject rules;
    /// [`casefold_v3_link`])
    #[serde(default)]
    pub linker: Option<String>,
    /// "reference" (the default) | "relation_aware": which records the stack-v5 manifests keep where a cap binds (tier 1
    /// above K, inverse candidates above MAX_INVERSE; `xerj_index::direction::RelPref`). Opt-in, for large shared stores;
    /// where no cap binds the manifests are the reference manifests.
    #[serde(default)]
    pub manifest: Option<String>,
    /// "decline_v2" (the default: the decline policy decides what is served; read `final_status`, `final_answer`,
    /// `final_evidence_ids` and `reason_code`, while `status` and `answer` stay the raw program's) | "served" (the stack-v5 rule
    /// `verify_v5`, which accepts wrong answers far more often and stays for comparison) | "bindings" (rule (b) counts the question's exact
    /// bindings, so a store name nested in a longer one, "Albania" in "Prime Minister of Albania", is not a second subject)
    /// | "bindings_path" (bindings, and every CHAIN record must continue the path) | "bindings_all" (bindings over every
    /// subject ever written, so a subject whose facts were withdrawn still counts). Opt-in ([`VerifyRule`]): the verify step
    /// runs after the compiler, so the program and the raw `status` / `answer` are the default's; only `verify`,
    /// `final_status` and `final_answer` can differ.
    #[serde(default)]
    pub verify: Option<String>,
    /// adds the candidates and the hop reads
    #[serde(default)]
    pub explain: bool,
}

#[derive(Debug, Deserialize)]
pub struct LookupReq {
    pub subject: String,
    pub relation: String,
    #[serde(default)]
    pub scope: Option<String>,
    #[serde(default)]
    pub revision: Option<Rev>,
}

fn status_name(s: Status) -> &'static str {
    match s {
        Status::Value => "VALUE",
        Status::True => "TRUE",
        Status::False => "FALSE",
        Status::Unknown => "UNKNOWN",
        Status::Conflict => "CONFLICT",
    }
}

fn read_rev(ns: &Namespace, revision: Option<Rev>) -> Result<Rev, FactsError> {
    let published = ns.published.load(Ordering::SeqCst);
    let rev = revision.unwrap_or(published);
    if rev > published {
        return Err(FactsError::NotCommitted(format!(
            "revision {rev} is not committed (published {published})"
        )));
    }
    Ok(rev)
}

/// Exact key lookup at a committed revision; no model.
pub fn lookup(ns: &Namespace, req: &LookupReq) -> Result<Value, FactsError> {
    let rev = read_rev(ns, req.revision)?;
    let mem = ns
        .mem
        .read()
        .map_err(|_| FactsError::Internal("facts projection lock poisoned".into()))?;
    let snap = mem.snapshot_at(rev);
    let l = snap.lookup(
        &req.subject,
        &req.relation,
        req.scope.as_deref().unwrap_or(DEFAULT_SCOPE),
    );
    let st = &mem.store;
    Ok(json!({
        "status": status_name(l.status),
        "value": l.value.as_ref().map(|v| json!({"kind": kind_name(v.kind), "value": v.value, "unit": v.unit, "rendered": render_value(v)})),
        "refs": l.refs.iter().map(|&r| st.fact_id(r)).collect::<Vec<_>>(),
        "reason": l.reason, "snapshot": {"revision": rev},
    }))
}

/// One fact record and its visibility at a committed revision (default: the published head); no model
/// (`GET /_facts/{ns}/{fact_id}`, design §5.1). Withdrawn and superseded records stay readable with `visible: false`.
/// `Ok(None)`: no committed record has this id. A record of a batch that is applied to the projection but not yet
/// published is not shown.
pub fn get_fact(
    ns: &Namespace,
    fact_id: &str,
    revision: Option<Rev>,
) -> Result<Option<Value>, FactsError> {
    let rev = read_rev(ns, revision)?;
    let published = ns.published.load(Ordering::SeqCst);
    let mem = ns
        .mem
        .read()
        .map_err(|_| FactsError::Internal("facts projection lock poisoned".into()))?;
    let st = &mem.store;
    let Some(r) = st
        .rec_of(fact_id)
        .filter(|&r| st.recs.added[r as usize] <= published)
    else {
        return Ok(None);
    };
    let v = st.typed_value(r);
    let (doc_id, version) = st.rec_doc_key(r);
    Ok(Some(json!({
        "fact_id": st.fact_id(r), "subject": st.subject(r), "relation": st.relation(r),
        "relation_phrase": rel_phrase(st.relation(r)), "kind": kind_name(v.kind), "value": render_value(&v),
        "scope": st.scope(r), "quote": st.quote(r), "source": {"doc_id": doc_id, "version": version},
        "added_rev": st.recs.added[r as usize], "visible": mem.snapshot_at(rev).visible(r),
        "snapshot": {"revision": rev}, "published_rev": published,
    })))
}

/// The rule of one ask: `decline_v2` when the request names none, a refused name is a 400.
fn verify_rule(name: Option<&str>) -> Result<VerifyRule, FactsError> {
    match name {
        None => Ok(VerifyRule::DeclineV2),
        Some(v) => VerifyRule::parse(v).ok_or_else(|| {
            FactsError::BadRequest(format!(
                "verify [{v}] is not one of {}",
                VerifyRule::NAMES.join(", ")
            ))
        }),
    }
}

/// One question: link, exact manifest, compiler decisions on the live exact executor, verify, plus the cited records.
pub fn ask(slot: &ModelSlot, ns: &Namespace, req: &AskReq) -> Result<Value, FactsError> {
    ask_stoppable(slot, ns, req, None)
}

/// [`ask`] that stops between two compiler decisions once `stop` is set ([`AskGate::run_stoppable`] sets it when the
/// caller stops waiting): the result is then [`FactsError::Stopped`], never an answer. While `stop` stays unset the
/// decisions and the answer are [`ask`]'s.
pub fn ask_stoppable(
    slot: &ModelSlot,
    ns: &Namespace,
    req: &AskReq,
    stop: Option<StopFlag>,
) -> Result<Value, FactsError> {
    ask_inner(slot, ns, req, stop, &AskExt::default())
}

/// EXTENSION (opt-in; [`AskExt::default`] = [`ask`]): options of one question that the HTTP API does not carry (how a
/// request would name them is an owner decision): which records the question may read, and the loop guard. Measured on
/// the E8 shared store (round 3, `tests/facts_e8_shared.rs`); nothing in the served path sets them.
#[derive(Clone, Debug, Default)]
pub struct AskExt {
    /// the records the question's manifests may offer (`xerj_index::scope`); None = every record
    pub scope: Option<xerj_index::scope::QuestionScope>,
    /// what an action does that reads a record the program already read (`xerj_exec::LoopGuard`)
    pub loop_guard: xerj_exec::LoopGuard,
}

/// [`ask`] with [`AskExt`] options (EXTENSION, opt-in). With `AskExt::default()` every decision and output field is
/// [`ask`]'s.
pub fn ask_ext(
    slot: &ModelSlot,
    ns: &Namespace,
    req: &AskReq,
    ext: &AskExt,
) -> Result<Value, FactsError> {
    ask_inner(slot, ns, req, None, ext)
}

fn ask_inner(
    slot: &ModelSlot,
    ns: &Namespace,
    req: &AskReq,
    stop: Option<StopFlag>,
    ext: &AskExt,
) -> Result<Value, FactsError> {
    if req.question.trim().is_empty() {
        return Err(FactsError::BadRequest("question must not be empty".into()));
    }
    let rev = read_rev(ns, req.revision)?;
    let mem = ns
        .mem
        .read()
        .map_err(|_| FactsError::Internal("facts projection lock poisoned".into()))?;
    let snap = mem.snapshot_at(rev);
    let item_id = req
        .item_id
        .clone()
        .unwrap_or_else(|| lab_item_id(&req.question));
    let linker_name = req.linker.as_deref().unwrap_or("exact");
    let relation_aware = match req.manifest.as_deref() {
        None | Some("reference") => false,
        Some("relation_aware") => true,
        Some(m) => {
            return Err(FactsError::BadRequest(format!(
                "manifest [{m}] is not one of reference, relation_aware"
            )))
        }
    };
    let verify = verify_rule(req.verify.as_deref())?;
    let stopped = |s: xerj_serve::engine::Stopped| FactsError::Stopped {
        decisions: s.decisions,
    };
    let mut out = if linker_name == "casefold" || linker_name == "casefold_v3" {
        // opt-in (E8 shared store): case-insensitive pre-link, then the exact pipeline on the rewritten question; the item id
        // (manifest salt) stays the original question's. The stop flag reaches the compiler as on the default path.
        let cl;
        let sl;
        let (rewritten, ents, link) = if linker_name == "casefold" {
            cl = casefold_link(&mem, &snap, &req.question);
            let link = json!({
                "mode": "casefold", "rewritten": cl.rewritten,
                "matches": cl.matches.iter().map(|m| json!({"surface": m.surface, "subject": m.subject})).collect::<Vec<_>>(),
                "ambiguous": cl.ambiguous.iter().map(|(s, c)| json!({"surface": s, "candidates": c})).collect::<Vec<_>>(),
            });
            (cl.rewritten.as_str(), None, link)
        } else {
            // v3: the linked subjects go to the manifest as decided here (function-word subjects out, case variants in)
            sl = casefold_v3_link(&mem, &snap, &req.question);
            let pool = &mem.store.pools.subjects;
            let link = json!({
                "mode": "casefold_v3", "rewritten": sl.case.rewritten,
                "matches": sl.case.matches.iter().map(|m| json!({"surface": m.surface, "subject": m.subject})).collect::<Vec<_>>(),
                "ambiguous": sl.case.ambiguous.iter().map(|(s, c)| json!({"surface": s, "candidates": c})).collect::<Vec<_>>(),
                "subjects": sl.subjects.iter().map(|&s| pool.get(s)).collect::<Vec<_>>(),
                "dropped": sl.dropped, "merged": sl.merged.iter().map(|(a, b)| json!([a, b])).collect::<Vec<_>>(),
            });
            let ents = sl.changes_subjects().then(|| sl.subjects.clone());
            (sl.case.rewritten.as_str(), ents, link)
        };
        let opts = AnswerOpts {
            ents,
            relation_aware,
            scope: ext.scope.clone(),
            loop_guard: ext.loop_guard,
            verify,
        };
        let mut o = answer_stoppable_opts(
            slot,
            &mem,
            snap,
            &item_id,
            rewritten,
            LinkerMode::Exact,
            req.explain,
            None,
            stop,
            &opts,
        )
        .map_err(stopped)?;
        o["question"] = json!(req.question);
        o["linker"] = json!(linker_name);
        o["link"] = link;
        o
    } else {
        let opts = AnswerOpts {
            ents: None,
            relation_aware,
            scope: ext.scope.clone(),
            loop_guard: ext.loop_guard,
            verify,
        };
        answer_stoppable_opts(
            slot,
            &mem,
            snap,
            &item_id,
            &req.question,
            LinkerMode::parse(linker_name),
            req.explain,
            None,
            stop,
            &opts,
        )
        .map_err(stopped)?
    };
    let st = &mem.store;
    // Under `verify: decline_v2` the citations are those of the served answer: a repaired answer cites the repaired program,
    // a declined one cites nothing (`evidence_ids` keeps the raw program's), and each carries where the quote sits in which
    // text. Without that option the block is the raw program's, field for field.
    let served = out.get("final_evidence_ids").is_some();
    let evidence: Vec<Value> = out[if served { "final_evidence_ids" } else { "evidence_ids" }]
        .as_array()
        .map(|ids| {
            ids.iter()
                .filter_map(|id| id.as_str())
                .filter_map(|id| st.rec_of(id))
                .map(|r| {
                    let v = st.typed_value(r);
                    let (doc_id, version) = st.rec_doc_key(r);
                    let mut e = json!({"fact_id": st.fact_id(r), "subject": st.subject(r), "relation": rel_phrase(st.relation(r)),
                           "kind": kind_name(v.kind), "value": render_value(&v), "scope": st.scope(r), "quote": st.quote(r),
                           "source": {"doc_id": doc_id, "version": version}});
                    if served {
                        let i = r as usize;
                        e["source"]["locator"] = json!({"unit": "char", "start": st.recs.start[i], "end": st.recs.end[i]});
                        e["source"]["content_sha256"] = json!(st.doc_sha256_hex(st.recs.doc[i]));
                    }
                    e
                })
                .collect()
        })
        .unwrap_or_default();
    out["evidence"] = json!(evidence);
    out["snapshot"] = json!({"revision": rev});
    out["published_rev"] = json!(ns.published.load(Ordering::SeqCst));
    out["model_info"] = slot.info();
    Ok(out)
}

/// A loaded projection and the access tick of its last use (the eviction order).
struct Resident {
    ns: Arc<Namespace>,
    used: u64,
}

/// The per-process service: namespace projections plus the lazily loaded compiler.
pub struct FactsService {
    namespaces: Mutex<BTreeMap<String, Resident>>,
    /// At most this many projections stay in memory (`XERJ_FACTS_MAX_RESIDENT_NAMESPACES`; unset, the default: all of
    /// them). Loading one more evicts the least recently used other projection. A projection is derived state: the next
    /// access rebuilds an evicted one from the index (the recovery path of design §2.5), so the namespace's revisions,
    /// fingerprints and answers are unchanged; only that access waits for the rebuild (design §2.4: the projection lags
    /// "only during a rebuild or eviction").
    max_resident: Option<usize>,
    /// access clock of the eviction order
    clock: AtomicU64,
    evictions: AtomicU64,
    /// one writer per process (spike); also serializes rebuilds with writes
    pub writer: tokio::sync::Mutex<()>,
    model_dir: Option<PathBuf>,
    precision: Precision,
    int8_mode: &'static str,
    /// answer-speed options (`XERJ_FACTS_THREADS`); default: one thread, a new session per question
    speed: SpeedOptions,
    /// the compiler, loaded once: on the first ask, or at start-up by the preload thread (`XERJ_FACTS_PRELOAD`)
    model: ModelCell,
    /// Admission of `_ask` (concurrency limit, deadline); both off unless configured ([`AskGate`]).
    pub ask_gate: AskGate,
    /// A limit in the environment that could not be read: every `_ask` is refused with it (no guessed limit).
    pub config_error: Option<String>,
}

/// `XERJ_FACTS_PRELOAD`: unset loads the compiler at start-up when a model directory is set; a value turns it on only
/// when [`preload_on`] reads it so.
fn preload_wanted(v: Option<&str>) -> bool {
    v.is_none() || preload_on(v)
}

/// Switch values that mean on (anything else, or unset, is off)
fn preload_on(v: Option<&str>) -> bool {
    matches!(
        v.map(|s| s.trim().to_ascii_lowercase()).as_deref(),
        Some("1" | "true" | "yes" | "on")
    )
}

/// The compiler cell of a service: empty until the model is loaded (first ask or preload), then the load's result.
type ModelCell = Arc<OnceLock<Result<Arc<ModelSlot>, String>>>;

/// `XERJ_FACTS_PRECISION` as (precision, int8 mode). The default, for anything not named below, is `w8`: weight-only INT8
/// (int8 weights, f32 activations, f32 embedding table), which keeps the model's behaviour within noise at half the weight
/// bytes. `f32` is the exact reference. `int8` and `int8-guard` / `int8+guard` are W8A8: never the default, because 8-bit
/// activations change this model's decisions (research/product/XERJ-R_INT8_CPU_serving_2026-10-30.md).
fn parse_precision(precision: &str) -> (Precision, &'static str) {
    match precision {
        "f32" => (Precision::F32, "f32"),
        "int8" => (Precision::Int8, "raw"),
        "int8-guard" | "int8+guard" => (Precision::Int8, "guard"),
        _ => (Precision::W8, "w8"),
    }
}

/// Threads per question when `XERJ_FACTS_THREADS` does not name a number: half the cores, between 1 and 4.
fn auto_threads(cores: usize) -> usize {
    (cores / 2).clamp(1, 4)
}

/// `XERJ_FACTS_THREADS` and `XERJ_FACTS_THREAD_QUEUE` as speed options on a machine of `cores` cores. Threads unset,
/// unreadable or 0: [`auto_threads`]. Buffers are always kept between questions.
fn speed_from(threads: Option<&str>, queue: Option<&str>, cores: usize) -> SpeedOptions {
    let threads = threads
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|&n| n >= 1)
        .unwrap_or_else(|| auto_threads(cores));
    SpeedOptions {
        threads: threads.min(64),
        reuse_sessions: true,
        queue: preload_on(queue),
    }
}

/// [`speed_from`] from the environment
fn speed_from_env() -> SpeedOptions {
    speed_from(
        std::env::var("XERJ_FACTS_THREADS").ok().as_deref(),
        std::env::var("XERJ_FACTS_THREAD_QUEUE").ok().as_deref(),
        std::thread::available_parallelism().map_or(1, usize::from),
    )
}

/// What a compiler load depends on: model directory, precision, int8 mode, speed options.
#[derive(Debug, Clone, PartialEq)]
struct LoadKey {
    dir: PathBuf,
    precision: Precision,
    int8_mode: &'static str,
    threads: usize,
    reuse_sessions: bool,
    queue: bool,
}

/// The compiler load started by [`start_early_preload`] (process-wide; at most one).
static EARLY_PRELOAD: OnceLock<(LoadKey, ModelCell)> = OnceLock::new();

/// The settings `FactsService::from_env` and [`start_early_preload`] both read, with the preload switch
fn env_load_settings() -> (Option<LoadKey>, bool) {
    let dir = std::env::var_os("XERJ_FACTS_MODEL_DIR").map(PathBuf::from);
    let (precision, int8_mode) =
        parse_precision(&std::env::var("XERJ_FACTS_PRECISION").unwrap_or_default());
    let speed = speed_from_env();
    let key = dir.map(|dir| LoadKey {
        dir,
        precision,
        int8_mode,
        threads: speed.threads,
        reuse_sessions: speed.reuse_sessions,
        queue: speed.queue,
    });
    (
        key,
        preload_wanted(std::env::var("XERJ_FACTS_PRELOAD").ok().as_deref()),
    )
}

/// Start the compiler load in a background thread into a fresh cell (the same thread `spawn_preload` used to start).
fn spawn_load(cell: &ModelCell, key: &LoadKey) {
    let cell = Arc::clone(cell);
    let dir = key.dir.clone();
    let (precision, int8_mode) = (key.precision, key.int8_mode);
    let speed = SpeedOptions {
        threads: key.threads,
        reuse_sessions: key.reuse_sessions,
        queue: key.queue,
    };
    let _ = std::thread::Builder::new()
        .name("xerj-facts-preload".to_string())
        .spawn(move || {
            cell.get_or_init(|| load_slot(Some(&dir), precision, int8_mode, speed));
        });
}

/// `XERJ_FACTS_PRELOAD` (on when a model directory is set) from the very start of the server's start-up: when the preload is on and a
/// model directory is configured, start the compiler load now, in a background thread, so it runs while the engine
/// opens its indices; the [`FactsService::from_env`] created afterwards adopts this load (same directory, precision and
/// speed options) instead of starting its own, which used to begin only about 0.2 s before the server listened. Returns
/// whether a load was started. With the preload off the model loads on the first ask.
pub fn start_early_preload() -> bool {
    let (Some(key), true) = env_load_settings() else {
        return false;
    };
    let mut started = false;
    EARLY_PRELOAD.get_or_init(|| {
        let cell: ModelCell = Arc::new(OnceLock::new());
        spawn_load(&cell, &key);
        started = true;
        (key, cell)
    });
    started
}

/// The early load's cell when it was started for exactly these settings
fn adopt_early(early: Option<&(LoadKey, ModelCell)>, key: &LoadKey) -> Option<ModelCell> {
    early.filter(|(k, _)| k == key).map(|(_, c)| Arc::clone(c))
}

/// Load the compiler of a service (the one place both the first ask and the preload thread go through).
fn load_slot(
    dir: Option<&Path>,
    precision: Precision,
    int8_mode: &str,
    speed: SpeedOptions,
) -> Result<Arc<ModelSlot>, String> {
    let dir = dir.ok_or_else(|| "XERJ_FACTS_MODEL_DIR is not set".to_string())?;
    load_model(dir, precision, int8_mode, speed).map(Arc::new)
}

impl FactsService {
    /// `XERJ_FACTS_MODEL_DIR` (an xerj-infer export), `XERJ_FACTS_PRECISION` = w8 (default) | f32 | int8 | int8-guard, and the
    /// `_ask` limits `XERJ_FACTS_MAX_CONCURRENT_ASKS` / `XERJ_FACTS_ASK_DEADLINE_MS` / `XERJ_FACTS_MAX_QUEUED_ASKS` (unset =
    /// no limit).
    /// `XERJ_FACTS_THREADS` = N sets the threads per question; unset, half the cores between 1 and 4. Buffers are kept
    /// between questions. Decisions are identical at every thread count (see `xerj_serve::engine::SpeedOptions`).
    /// `XERJ_FACTS_THREAD_QUEUE` = 1 (unset by default: off; only with `XERJ_FACTS_THREADS` >= 2) makes concurrent
    /// questions take turns on all the threads, one computing at a time, instead of running the others single-threaded
    /// beside it (see `xerj_serve::engine::SpeedOptions::queue`); decisions are identical either way.
    /// `XERJ_FACTS_PRELOAD`, on when unset and a model directory is set (0 turns it off), loads the compiler in a background
    /// thread at start-up instead of on the first ask, so the first question after a restart does not wait for the load (a question that arrives during the
    /// load waits for that same load). Same model, same options, same result: only the moment of the load moves.
    /// `XERJ_FACTS_MAX_RESIDENT_NAMESPACES` = N (unset by default: no eviction) keeps at most N projections in memory.
    /// A load started earlier by [`start_early_preload`] with the same settings is adopted, not repeated.
    pub fn from_env() -> FactsService {
        let model_dir = std::env::var_os("XERJ_FACTS_MODEL_DIR").map(PathBuf::from);
        let precision = std::env::var("XERJ_FACTS_PRECISION").unwrap_or_default();
        let mut svc = FactsService::new(model_dir, &precision);
        match AskGate::from_env() {
            Ok(g) => svc.ask_gate = g,
            Err(e) => svc.config_error = Some(e),
        }
        svc.speed = speed_from_env();
        svc.max_resident = std::env::var("XERJ_FACTS_MAX_RESIDENT_NAMESPACES")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .filter(|&n| n >= 1);
        if svc.model_dir.is_some()
            && preload_wanted(std::env::var("XERJ_FACTS_PRELOAD").ok().as_deref())
        {
            match adopt_early(EARLY_PRELOAD.get(), &svc.load_key().expect("model dir")) {
                Some(cell) => svc.model = cell,
                None => svc.spawn_preload(),
            }
        }
        svc
    }

    /// what this service's compiler load depends on (None without a model directory)
    fn load_key(&self) -> Option<LoadKey> {
        self.model_dir.clone().map(|dir| LoadKey {
            dir,
            precision: self.precision,
            int8_mode: self.int8_mode,
            threads: self.speed.threads,
            reuse_sessions: self.speed.reuse_sessions,
            queue: self.speed.queue,
        })
    }

    /// A service with this model directory (`None` = no model configured) and precision (`f32` | `int8` |
    /// `int8-guard`; anything else is w8, as in [`FactsService::from_env`]), no `_ask` limits, and nothing read from the
    /// environment: one thread per question, no eviction, no preload. Tests build their node with it so parallel tests
    /// never race on process-wide variables.
    pub fn new(model_dir: Option<PathBuf>, precision: &str) -> FactsService {
        let (precision, int8_mode) = parse_precision(precision);
        FactsService {
            namespaces: Mutex::new(BTreeMap::new()),
            max_resident: None,
            clock: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
            writer: tokio::sync::Mutex::new(()),
            model_dir,
            precision,
            int8_mode,
            speed: SpeedOptions::default(),
            model: Arc::new(OnceLock::new()),
            ask_gate: AskGate::default(),
            config_error: None,
        }
    }

    /// Start loading the compiler in a background thread. If the thread cannot be spawned, nothing changes: the first ask
    /// loads it as before.
    fn spawn_preload(&self) {
        if let Some(key) = self.load_key() {
            spawn_load(&self.model, &key);
        }
    }

    /// The same service with these `_ask` limits.
    pub fn with_ask_gate(mut self, gate: AskGate) -> FactsService {
        self.ask_gate = gate;
        self
    }

    pub fn get(&self, ns: &str) -> Option<Arc<Namespace>> {
        let mut m = self.namespaces.lock().ok()?;
        let r = m.get_mut(ns)?;
        r.used = self.clock.fetch_add(1, Ordering::Relaxed);
        Some(r.ns.clone())
    }

    /// Load a projection; beyond `XERJ_FACTS_MAX_RESIDENT_NAMESPACES` evict the least recently used others and return
    /// their names. Called with the writer lock held (as every load is), so no write of an evicted namespace is in
    /// flight; a reader that already holds its projection finishes on it.
    pub fn insert(&self, ns: &str, n: Arc<Namespace>) -> Vec<String> {
        let mut evicted = Vec::new();
        if let Ok(mut m) = self.namespaces.lock() {
            let used = self.clock.fetch_add(1, Ordering::Relaxed);
            m.insert(ns.to_string(), Resident { ns: n, used });
            if let Some(max) = self.max_resident {
                while m.len() > max {
                    let Some(lru) = m
                        .iter()
                        .filter(|(k, _)| k.as_str() != ns)
                        .min_by_key(|(_, r)| r.used)
                        .map(|(k, _)| k.clone())
                    else {
                        break;
                    };
                    m.remove(&lru);
                    evicted.push(lru);
                }
            }
        }
        self.evictions
            .fetch_add(evicted.len() as u64, Ordering::Relaxed);
        evicted
    }

    /// `XERJ_FACTS_MAX_RESIDENT_NAMESPACES` (`None`: no eviction, the default) and the evictions so far.
    pub fn residency(&self) -> (Option<usize>, u64) {
        (self.max_resident, self.evictions.load(Ordering::Relaxed))
    }

    /// Drop a projection; the next access rebuilds it from the index.
    pub fn remove(&self, ns: &str) {
        if let Ok(mut m) = self.namespaces.lock() {
            m.remove(ns);
        }
    }

    pub fn loaded(&self) -> Vec<(String, Arc<Namespace>)> {
        self.namespaces
            .lock()
            .map(|m| m.iter().map(|(k, r)| (k.clone(), r.ns.clone())).collect())
            .unwrap_or_default()
    }

    /// The compiler, loaded on first use (or by the preload thread). A load error is sticky until restart.
    pub fn model(&self) -> Result<Arc<ModelSlot>, FactsError> {
        self.model
            .get_or_init(|| {
                load_slot(
                    self.model_dir.as_deref(),
                    self.precision,
                    self.int8_mode,
                    self.speed,
                )
            })
            .clone()
            .map_err(FactsError::ModelUnavailable)
    }

    pub fn model_status(&self) -> Value {
        match self.model.get() {
            None => {
                json!({"state": if self.model_dir.is_some() { "not_loaded" } else { "not_configured" },
                           "dir": self.model_dir.as_ref().map(|d| d.display().to_string())})
            }
            Some(Ok(slot)) => json!({"state": "loaded", "info": slot.info()}),
            Some(Err(e)) => json!({"state": "unavailable", "error": e}),
        }
    }
}

/// Load an xerj-infer export with the engine tokenizer, after checking that tokenizer.json is the export's (sha256).
fn load_model(
    dir: &Path,
    precision: Precision,
    int8_mode: &str,
    speed: SpeedOptions,
) -> Result<ModelSlot, String> {
    let cfg_text = std::fs::read_to_string(dir.join("config.json"))
        .map_err(|e| format!("{}/config.json: {e}", dir.display()))?;
    let cfg: Value = serde_json::from_str(&cfg_text).map_err(|e| format!("config.json: {e}"))?;
    let tok_path = dir.join("tokenizer.json");
    let want = cfg
        .pointer("/tokenizer/sha256")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let got = xerj_infer::tokenizer::sha256_file(&tok_path).map_err(|e| e.to_string())?;
    if want != got {
        return Err(format!(
            "tokenizer.json sha256 {got} does not match the export's {want}"
        ));
    }
    let tok = EngineTokenizer::from_file(&tok_path)?;
    ModelSlot::load_with_encoder_speed("facts", dir, precision, int8_mode, Box::new(tok), speed)
}

/// Milliseconds since the Unix epoch (commit stamps).
pub fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

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

    /// every op type: documents, a number fact, a correction (supersedes) and a document withdrawal
    fn world_with_edits() -> Value {
        json!({
            "world_id": "t2",
            "base_documents": [{"doc_id": "d1", "version": "v1", "text": "Liberia is governed by George Weah."},
                               {"doc_id": "d2", "version": "v1", "text": "Ana Bel is 170 cm tall."}],
            "base_facts": [{"fact_id": "f1", "subject": "Liberia", "relation": "head_of_government", "kind": "text",
                            "value": "George Weah", "doc_id": "d1", "version": "v1", "start": 0, "end": 35,
                            "quote": "Liberia is governed by George Weah."},
                           {"fact_id": "f2", "subject": "Ana Bel", "relation": "height", "kind": "number", "value": "170",
                            "unit": "cm", "doc_id": "d2", "version": "v1", "start": 11, "end": 14, "quote": "170"}],
            "edits": [{"op": "correct", "doc": {"doc_id": "d2", "version": "v2", "text": "Ana Bel is 171 cm tall."},
                       "fact": {"fact_id": "f3", "subject": "Ana Bel", "relation": "height", "kind": "number", "value": "171",
                                "unit": "cm", "doc_id": "d2", "version": "v2", "start": 11, "end": 14, "quote": "171"},
                       "supersedes": "f2"},
                      {"op": "withdraw_document", "doc_id": "d1", "version": "v1"}]
        })
    }

    /// what a rebuild reads back from the index: the epoch documents and the record payloads
    fn committed(ap: &Applied) -> (Vec<EpochRec>, Payloads) {
        let epochs = ap
            .epochs
            .iter()
            .map(|(_, src)| parse_epoch(src).expect("epoch").expect("an epoch document"))
            .collect();
        let mut p = Payloads::default();
        for (_, src) in &ap.records {
            p.add_source(src).expect("payload");
        }
        (epochs, p)
    }

    /// the index after `batches` were written: their records with the step-5 repairs applied (partial updates), and the
    /// epochs; `skip_repairs_of` = the batches whose step 5 a crash cut off
    fn index_after(batches: &[&Applied], skip_repairs_of: &[usize]) -> BTreeMap<String, Value> {
        let mut idx = BTreeMap::new();
        for (k, ap) in batches.iter().enumerate() {
            for (id, src) in ap.records.iter().chain(&ap.epochs) {
                idx.insert(id.clone(), src.clone());
            }
            if !skip_repairs_of.contains(&k) {
                for (id, fields) in &ap.repairs {
                    let rec = idx
                        .get_mut(id)
                        .expect("a repair targets an existing record");
                    merge_fields(rec, fields.clone());
                }
            }
        }
        idx
    }

    /// what a rebuild walks: every record of the index
    fn walk(idx: &BTreeMap<String, Value>) -> (Vec<EpochRec>, Payloads) {
        let mut epochs = Vec::new();
        let mut p = Payloads::default();
        for src in idx.values() {
            match parse_epoch(src).expect("epoch") {
                Some(e) => epochs.push(e),
                None => p.add_source(src).expect("payload"),
            }
        }
        epochs.sort_by_key(|e| (e.to, e.from));
        (epochs, p)
    }

    fn bulk(world: Value) -> Batch {
        Batch::World { world }
    }

    /// the edits of [`world_with_edits`] as a second batch over the base sources of the first
    fn base_then_edits() -> (Value, Value) {
        let w = world_with_edits();
        let base = json!({"world_id": "t2", "base_documents": w["base_documents"], "base_facts": w["base_facts"],
                          "edits": []});
        let edits = json!({"base_documents": [], "base_facts": [], "edits": w["edits"]});
        (base, edits)
    }

    fn fingerprint(n: &Namespace, rev: Rev) -> String {
        n.mem
            .read()
            .unwrap()
            .snapshot_at(rev)
            .fingerprint()
            .to_string()
    }

    #[test]
    fn replay_reproduces_revisions_and_fingerprints() {
        let a = Namespace::new();
        let ap = a
            .apply(&Batch::World {
                world: world_with_edits(),
            })
            .expect("apply");
        assert_eq!(
            (ap.from_rev, ap.to_rev, ap.base_rev),
            (1, 7, Some(4)),
            "4 base ops, a correction (2 ops), a withdrawal"
        );
        assert_eq!(ap.epochs.len(), 1);
        assert_eq!(ap.records.len(), 6, "3 doc + 3 fact documents");
        a.published.store(ap.to_rev, Ordering::SeqCst);
        let (epochs, payloads) = committed(&ap);
        let (b, ignored) = Namespace::replay(&epochs, &payloads).expect("replay");
        assert_eq!(ignored, 0);
        assert_eq!(b.published.load(Ordering::SeqCst), ap.to_rev);
        for rev in [4, 7] {
            assert_eq!(
                fingerprint(&a, rev),
                fingerprint(&b, rev),
                "exact fingerprint at revision {rev}"
            );
        }
    }

    #[test]
    fn walked_fact_payloads_keep_the_stored_input_text() {
        let a = Namespace::new();
        let ap = a
            .apply(&Batch::World {
                world: world_with_edits(),
            })
            .expect("apply");
        let (_, p) = committed(&ap);
        let mut n = 0;
        for (_, src) in &ap.records {
            if src["rt"] == "fact" {
                let id = src["fact_id"].as_str().expect("fact_id");
                assert_eq!(
                    Some(&*p.facts[id]),
                    src["input"].as_str(),
                    "fact {id}: the walk keeps the stored text"
                );
                n += 1;
            }
        }
        assert_eq!(n, 3);
    }

    #[test]
    fn a_fact_record_whose_input_does_not_parse_is_still_corrupt() {
        let mut p = Payloads::default();
        let src = json!({"rt": "fact", "fact_id": "f1", "input": "{\"fact_id\": ", "added_rev": 1, "batch_to_rev": 1});
        let e = p.add_source(&src).expect_err("an input that is not JSON");
        assert!(e.starts_with("fact f1: input: "), "{e}");
        assert!(p.facts.is_empty() && p.seen.is_empty());
    }

    #[test]
    fn one_op_per_epoch_and_a_gap() {
        let a = Namespace::new();
        let ap = a
            .apply_chunked(
                &Batch::World {
                    world: world_with_edits(),
                },
                1,
            )
            .expect("apply");
        assert_eq!(ap.epochs.len(), 7, "one epoch per op, one revision per op");
        let (epochs, payloads) = committed(&ap);
        assert!(
            epochs.iter().all(|e| e.batch_to == 7),
            "every epoch names the batch's end"
        );
        let (b, _) = Namespace::replay(&epochs, &payloads).expect("replay");
        assert_eq!(fingerprint(&a, 7), fingerprint(&b, 7));
        // schema-2 epochs (no batch_to_rev: each epoch its own batch). Epoch 3..3 missing: 1..2 are committed, 4..7 come
        // after a gap
        let gapped: Vec<_> = epochs
            .iter()
            .filter(|e| e.from != 3)
            .map(|e| EpochRec {
                batch_to: e.to,
                ..e.clone()
            })
            .collect();
        let (c, ignored) = Namespace::replay(&gapped, &payloads).expect("replay");
        assert_eq!((c.published.load(Ordering::SeqCst), ignored), (2, 4));
        assert_eq!(fingerprint(&a, 2), fingerprint(&c, 2));
    }

    /// A batch split over several epochs commits with its last epoch: any missing epoch (a process kill between the epoch
    /// writes, or a lost write) leaves the whole batch uncommitted, and recovery lists all of its records and epochs as
    /// orphans.
    #[test]
    fn a_batch_commits_only_with_all_of_its_epochs() {
        let a = Namespace::new();
        let ap = a
            .apply_chunked(&bulk(world_with_edits()), 2)
            .expect("apply");
        assert_eq!(ap.epochs.len(), 4, "1-2, 3-4, 5-6, 7");
        let (epochs, payloads) = committed(&ap);
        for drop in 0..epochs.len() {
            let part: Vec<_> = epochs
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != drop)
                .map(|(_, e)| e.clone())
                .collect();
            let r = Namespace::recover(&part, &payloads).expect("recover");
            assert_eq!(
                r.ns.published.load(Ordering::SeqCst),
                0,
                "epoch {drop} missing"
            );
            assert_eq!(r.ignored, 3);
            assert_eq!(r.orphans.len(), 3 + ap.records.len(), "{:?}", r.orphans);
            assert!(r.repairs.is_empty() && r.unexplained.is_empty() && r.blocked.is_none());
        }
        let r = Namespace::recover(&epochs, &payloads).expect("recover");
        assert_eq!((r.ns.published.load(Ordering::SeqCst), r.ignored), (7, 0));
        assert!(r.orphans.is_empty() && r.repairs.is_empty());
    }

    /// New records carry the repair-owned fields as of their batch's end.
    #[test]
    fn records_carry_the_repair_owned_fields() {
        let a = Namespace::new();
        let ap = a.apply(&bulk(world_with_edits())).expect("apply");
        let rec: HashMap<&str, &Value> =
            ap.records.iter().map(|(id, v)| (id.as_str(), v)).collect();
        let f = |id: &str, k: &str| rec[id][k].clone();
        // revisions: d1 1, d2 2, f1 3, f2 4, d2@v2 5, f3 6 (supersedes f2), withdraw d1@v1 7
        assert_eq!(
            (f("f:f2", "retired_rev"), f("f:f2", "vis_end_rev")),
            (json!(6), json!(6))
        );
        assert_eq!(
            (f("f:f1", "retired_rev"), f("f:f1", "vis_end_rev")),
            (Value::Null, json!(7))
        );
        assert_eq!(
            (f("f:f3", "retired_rev"), f("f:f3", "vis_end_rev")),
            (Value::Null, Value::Null)
        );
        assert_eq!(f("d:d1@v1", "withdrawn_rev"), json!(7));
        assert_eq!(f("d:d2@v1", "withdrawn_rev"), Value::Null);
        assert!(ap.repairs.is_empty(), "nothing older than the batch");
    }

    /// Step 5 lists exactly the older records a batch changes, and a rebuild of the index that has them applied needs no
    /// cleanup; a withdraw_fact batch repairs its one record.
    #[test]
    fn step5_repairs_the_older_records_a_batch_changes() {
        let (base, edits) = base_then_edits();
        let a = Namespace::new();
        let ap1 = a.apply(&bulk(base)).expect("base");
        let ap2 = a.apply(&bulk(edits)).expect("edits");
        assert_eq!((ap2.from_rev, ap2.to_rev), (5, 7));
        let ids: Vec<&str> = ap2.repairs.iter().map(|(id, _)| id.as_str()).collect();
        assert_eq!(ids, ["d:d1@v1", "f:f1", "f:f2"]);
        let get = |id: &str| ap2.repairs.iter().find(|(i, _)| i == id).unwrap().1.clone();
        assert_eq!(get("f:f2"), json!({"retired_rev": 6, "vis_end_rev": 6}));
        assert_eq!(get("f:f1"), json!({"retired_rev": null, "vis_end_rev": 7}));
        assert_eq!(get("d:d1@v1"), json!({"withdrawn_rev": 7}));
        let ap3 = a
            .apply(&Batch::WithdrawFact {
                fact_id: "f3".into(),
            })
            .expect("withdraw");
        assert_eq!(
            ap3.repairs,
            vec![(
                "f:f3".to_string(),
                json!({"retired_rev": 8, "vis_end_rev": 8})
            )]
        );
        let idx = index_after(&[&ap1, &ap2, &ap3], &[]);
        let (epochs, p) = walk(&idx);
        let r = Namespace::recover(&epochs, &p).expect("recover");
        assert_eq!(r.ns.published.load(Ordering::SeqCst), 8);
        assert!(r.orphans.is_empty(), "{:?}", r.orphans);
        assert!(r.repairs.is_empty(), "{:?}", r.repairs);
        assert_eq!(fingerprint(&a, 8), fingerprint(&r.ns, 8));
    }

    /// A kill between a batch's markers and the end of its step 5: the batch is committed, and recovery lists exactly the
    /// repairs step 5 would have written.
    #[test]
    fn recovery_finishes_an_interrupted_step5() {
        let (base, edits) = base_then_edits();
        let a = Namespace::new();
        let ap1 = a.apply(&bulk(base)).expect("base");
        let ap2 = a.apply(&bulk(edits)).expect("edits");
        let idx = index_after(&[&ap1, &ap2], &[1]);
        let (epochs, p) = walk(&idx);
        let r = Namespace::recover(&epochs, &p).expect("recover");
        assert_eq!(r.ns.published.load(Ordering::SeqCst), 7);
        assert!(r.orphans.is_empty());
        assert_eq!(r.repairs, ap2.repairs);
    }

    /// A kill (or a failed write) before a batch's markers: its records stay in the index without an epoch. Recovery keeps
    /// the committed prefix and lists exactly the uncommitted batch's records as orphans; an epoch after a gap is one too.
    #[test]
    fn recovery_lists_uncommitted_records_and_epochs() {
        let (base, edits) = base_then_edits();
        let a = Namespace::new();
        let ap1 = a.apply(&bulk(base)).expect("base");
        let ap2 = a.apply(&bulk(edits)).expect("edits");
        let mut idx = index_after(&[&ap1], &[]);
        for (id, src) in &ap2.records {
            idx.insert(id.clone(), src.clone()); // step 3 of batch 2 done, no marker
        }
        let (epochs, p) = walk(&idx);
        let r = Namespace::recover(&epochs, &p).expect("recover");
        assert_eq!((r.ns.published.load(Ordering::SeqCst), r.ignored), (4, 0));
        let mut want: Vec<String> = ap2.records.iter().map(|(id, _)| id.clone()).collect();
        want.sort();
        assert_eq!(r.orphans, want);
        assert!(r.repairs.is_empty() && r.unexplained.is_empty() && r.blocked.is_none());
        assert_eq!(fingerprint(&a, 4), fingerprint(&r.ns, 4));
    }

    /// Residue of two batches past the head (here the records of batch 5..7 without its epoch, and a whole batch 8..8 after
    /// that gap: what a walk that missed batch 5..7's epoch would see) is no crash's: nothing is deleted, the recovery
    /// is blocked.
    #[test]
    fn residue_of_several_batches_blocks_the_recovery() {
        let (base, edits) = base_then_edits();
        let a = Namespace::new();
        let ap1 = a.apply(&bulk(base)).expect("base");
        let ap2 = a.apply(&bulk(edits)).expect("edits");
        let ap3 = a
            .apply(&Batch::WithdrawFact {
                fact_id: "f3".into(),
            })
            .expect("withdraw");
        let mut idx = index_after(&[&ap1, &ap2, &ap3], &[]);
        idx.remove(&epoch_record_id(7));
        let (epochs, p) = walk(&idx);
        let r = Namespace::recover(&epochs, &p).expect("recover");
        assert_eq!(r.ns.published.load(Ordering::SeqCst), 4);
        assert!(r.orphans.is_empty(), "{:?}", r.orphans);
        let why = r.blocked.expect("blocked");
        assert!(why.contains("2 batches"), "{why}");
        let mut want: Vec<String> = ap2.records.iter().map(|(id, _)| id.clone()).collect();
        want.push(epoch_record_id(8));
        want.sort();
        assert_eq!(r.unexplained, want);
    }

    /// An epoch of an uncommitted batch that starts where the chain ends must not hide a complete batch that a later write
    /// committed at the same revisions (an orphan the cleanup has not removed yet).
    #[test]
    fn an_incomplete_batch_does_not_hide_a_committed_one() {
        let (base, edits) = base_then_edits();
        let a = Namespace::new();
        let ap1 = a.apply(&bulk(base)).expect("base");
        let ap2 = a.apply(&bulk(edits)).expect("edits");
        let mut idx = index_after(&[&ap1, &ap2], &[]);
        // a stale first epoch of a two-epoch batch 5..12 whose second epoch never arrived
        idx.insert(
            epoch_record_id(6),
            json!({"rt": "epoch", "from_rev": 5, "to_rev": 6, "batch_to_rev": 12,
                   "ops": r#"[{"op":"withdraw_fact","rev":5,"fact_id":"f1"},{"op":"withdraw_fact","rev":6,"fact_id":"f2"}]"#}),
        );
        let (epochs, p) = walk(&idx);
        let r = Namespace::recover(&epochs, &p).expect("recover");
        assert_eq!((r.ns.published.load(Ordering::SeqCst), r.ignored), (7, 1));
        // it starts at or below the head: no crash leaves that, so it is reported and left in place
        assert!(r.orphans.is_empty());
        assert_eq!(r.unexplained, vec![epoch_record_id(6)]);
        assert_eq!(fingerprint(&a, 7), fingerprint(&r.ns, 7));
    }

    /// A record of an unknown key that was written at or below the head is not an orphan of a crash: never deleted.
    #[test]
    fn unknown_records_at_or_below_the_head_are_left_in_place() {
        let (base, edits) = base_then_edits();
        let a = Namespace::new();
        let ap1 = a.apply(&bulk(base)).expect("base");
        let ap2 = a.apply(&bulk(edits)).expect("edits");
        let mut idx = index_after(&[&ap1, &ap2], &[]);
        let mut stray = idx["f:f1"].clone();
        stray["fact_id"] = json!("f9");
        stray["added_rev"] = json!(3);
        idx.insert(fact_record_id("f9"), stray);
        let (epochs, p) = walk(&idx);
        let r = Namespace::recover(&epochs, &p).expect("recover");
        assert_eq!(r.ns.published.load(Ordering::SeqCst), 7);
        assert!(r.orphans.is_empty() && r.repairs.is_empty());
        assert_eq!(r.unexplained, vec![fact_record_id("f9")]);
    }

    #[test]
    fn epochs_with_impossible_ranges_are_refused() {
        for (from, to, bt) in [(0, 1, 1), (3, 2, 3), (2, 4, 3)] {
            let src = json!({"rt": "epoch", "from_rev": from, "to_rev": to, "batch_to_rev": bt, "ops": "[]"});
            assert!(parse_epoch(&src).is_err(), "{from}..{to} batch {bt}");
        }
        let old = json!({"rt": "epoch", "from_rev": 1, "to_rev": 2, "ops": "[]"});
        assert_eq!(
            parse_epoch(&old).unwrap().unwrap().batch_to,
            2,
            "schema 2: its own batch"
        );
    }

    /// FactConsolidation 262K (11.6 MB as one epoch) was refused by the engine's 8 MiB limit for one document: epochs hold
    /// op headers only, at most EPOCH_OPS each, whatever the batch size.
    #[test]
    fn large_batch_epochs_stay_small() {
        let n = 6_000;
        let filler = "x ".repeat(1_000);
        let mut docs = Vec::new();
        let mut facts = Vec::new();
        for i in 0..n {
            let (subject, value) = (format!("Entity{i}"), format!("Person{i}"));
            let text = format!("{subject} is governed by {value}. {filler}");
            let start = subject.len() + " is governed by ".len();
            docs.push(json!({"doc_id": format!("d{i}"), "version": "v1", "text": text}));
            facts.push(json!({"fact_id": format!("f{i}"), "subject": subject, "relation": "head_of_government", "kind": "text",
                              "value": value, "doc_id": format!("d{i}"), "version": "v1", "start": start,
                              "end": start + value.len(), "quote": value}));
        }
        let world =
            json!({"world_id": "big", "base_documents": docs, "base_facts": facts, "edits": []});
        assert!(
            world.to_string().len() > 8 << 20,
            "the batch itself is over the 8 MiB document limit"
        );
        let a = Namespace::new();
        let ap = a.apply(&Batch::World { world }).expect("apply");
        assert_eq!(ap.to_rev as usize, 2 * n);
        assert_eq!(ap.epochs.len(), (2 * n).div_ceil(EPOCH_OPS));
        for (id, src) in ap.epochs.iter().chain(&ap.records) {
            assert!(
                src.to_string().len() < 1 << 20,
                "{id} is {} bytes",
                src.to_string().len()
            );
        }
        let (epochs, payloads) = committed(&ap);
        let (b, ignored) = Namespace::replay(&epochs, &payloads).expect("replay");
        assert_eq!(
            (b.published.load(Ordering::SeqCst), ignored),
            (ap.to_rev, 0)
        );
        assert_eq!(fingerprint(&a, ap.to_rev), fingerprint(&b, ap.to_rev));
    }

    #[test]
    fn a_missing_payload_fails_the_rebuild() {
        let a = Namespace::new();
        let ap = a.apply(&Batch::World { world: world() }).expect("apply");
        let (epochs, _) = committed(&ap);
        let e = Namespace::replay(&epochs, &Payloads::default())
            .err()
            .expect("no payloads");
        assert!(e.contains("no doc document for d1@v1"), "{e}");
    }

    #[test]
    fn fact_input_round_trips() {
        let f = FactIn {
            fact_id: "f9",
            subject: "Ana Bel",
            relation: "height",
            kind: "number",
            value: ValueIn::Str("170"),
            unit: "cm",
            doc_id: "d2",
            version: "v1",
            start: 11,
            end: 14,
            quote: "170",
            scope: "default",
            subject_id: Some("Q1"),
            value_id: None,
            valid_from: Some("2020-01-01"),
            valid_to: None,
            writer_version: "oracle",
        };
        let j = fact_input_json(&f);
        assert_eq!(fact_input_json(&fact_input_from_json(&j).unwrap()), j);
        let b = FactIn {
            value: ValueIn::Bool(true),
            kind: "boolean",
            ..f
        };
        let jb = fact_input_json(&b);
        assert_eq!(fact_input_json(&fact_input_from_json(&jb).unwrap()), jb);
    }

    #[test]
    fn epoch_after_a_gap_is_not_committed() {
        let (n, ignored) = Namespace::replay(
            &[EpochRec {
                from: 3,
                to: 3,
                batch_to: 3,
                ops: vec![OpHeader::WithdrawFact {
                    rev: 3,
                    fact_id: "f1".into(),
                }],
            }],
            &Payloads::default(),
        )
        .expect("replay");
        assert_eq!(ignored, 1);
        assert_eq!(n.published.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn lookup_reads_the_committed_revision_only() {
        let n = Namespace::new();
        let ap = n.apply(&Batch::World { world: world() }).expect("apply");
        let req = LookupReq {
            subject: "Liberia".into(),
            relation: "head_of_government".into(),
            scope: None,
            revision: None,
        };
        // not yet published: the default snapshot is revision 0, where nothing is visible
        assert_eq!(lookup(&n, &req).unwrap()["status"], "UNKNOWN");
        n.published.store(ap.to_rev, Ordering::SeqCst);
        let v = lookup(&n, &req).unwrap();
        assert_eq!(v["status"], "VALUE");
        assert_eq!(v["refs"][0], "f1");
        let ahead = LookupReq {
            revision: Some(ap.to_rev + 1),
            ..req
        };
        assert!(matches!(
            lookup(&n, &ahead),
            Err(FactsError::NotCommitted(_))
        ));
    }

    #[test]
    fn preload_is_on_when_unset_and_a_value_can_turn_it_off() {
        assert!(preload_wanted(None));
        assert!(preload_wanted(Some("1")));
        assert!(!preload_wanted(Some("0")));
        assert!(!preload_wanted(Some("")));
        for v in ["1", "true", "TRUE", " yes ", "on"] {
            assert!(preload_on(Some(v)), "{v:?} turns the preload on");
        }
        for v in [
            None,
            Some(""),
            Some("0"),
            Some("false"),
            Some("off"),
            Some("2"),
            Some("preload"),
        ] {
            assert!(!preload_on(v), "{v:?} leaves the preload off");
        }
    }

    #[test]
    fn the_default_precision_is_weight_only_int8_and_w8a8_must_be_named() {
        for unset_or_other in ["", "w8", "typo"] {
            assert_eq!(parse_precision(unset_or_other), (Precision::W8, "w8"));
        }
        assert_eq!(parse_precision("f32"), (Precision::F32, "f32"));
        assert_eq!(parse_precision("int8"), (Precision::Int8, "raw"));
        assert_eq!(parse_precision("int8-guard"), (Precision::Int8, "guard"));
    }

    #[test]
    fn an_early_preload_is_adopted_only_with_the_same_settings() {
        let key = |precision: &str, threads: Option<&str>| {
            let (precision, int8_mode) = parse_precision(precision);
            let speed = speed_from(threads, None, 8);
            LoadKey {
                dir: PathBuf::from("/m"),
                precision,
                int8_mode,
                threads: speed.threads,
                reuse_sessions: speed.reuse_sessions,
                queue: speed.queue,
            }
        };
        let cell: ModelCell = Arc::new(OnceLock::new());
        let early = (key("int8-guard", Some("4")), Arc::clone(&cell));
        let got = adopt_early(Some(&early), &key("int8+guard", Some(" 4 ")));
        assert!(
            got.is_some_and(|c| Arc::ptr_eq(&c, &cell)),
            "same settings: the early load"
        );
        for other in [
            key("f32", Some("4")),
            key("int8", Some("4")),
            key("int8-guard", Some("2")),
        ] {
            assert!(adopt_early(Some(&early), &other).is_none(), "{other:?}");
        }
        let mut elsewhere = key("int8-guard", Some("4"));
        elsewhere.dir = PathBuf::from("/other");
        assert!(adopt_early(Some(&early), &elsewhere).is_none());
        let mut queued = key("int8-guard", Some("4"));
        queued.queue = true;
        assert!(adopt_early(Some(&early), &queued).is_none());
        assert!(adopt_early(None, &key("int8-guard", Some("4"))).is_none());
        // the speed options as from_env reads them
        assert_eq!(speed_from(Some("4"), None, 8).threads, 4);
        assert!(speed_from(Some("1"), None, 8).reuse_sessions);
        assert!(
            !speed_from(Some("4"), None, 8).queue,
            "the queue is off by default"
        );
        assert!(speed_from(Some("4"), Some("1"), 8).queue);
        assert!(!speed_from(Some("4"), Some("0"), 8).queue);
        assert_eq!(speed_from(Some("999"), None, 8).threads, 64);
    }

    #[test]
    fn an_ask_that_names_no_rule_gets_decline_v2_and_served_stays_available() {
        assert!(matches!(verify_rule(None), Ok(VerifyRule::DeclineV2)));
        assert!(matches!(
            verify_rule(Some("served")),
            Ok(VerifyRule::Served)
        ));
        assert!(matches!(
            verify_rule(Some("decline_v2")),
            Ok(VerifyRule::DeclineV2)
        ));
        assert!(matches!(
            verify_rule(Some("nonsense")),
            Err(FactsError::BadRequest(_))
        ));
    }

    #[test]
    fn threads_are_half_the_cores_between_one_and_four_unless_named() {
        for (cores, want) in [
            (1, 1),
            (2, 1),
            (3, 1),
            (4, 2),
            (6, 3),
            (8, 4),
            (20, 4),
            (128, 4),
        ] {
            for unset in [None, Some("0"), Some("x")] {
                let s = speed_from(unset, None, cores);
                assert_eq!(s.threads, want, "{cores} cores, {unset:?}");
                assert!(s.reuse_sessions);
            }
        }
        assert_eq!(speed_from(Some("1"), None, 20).threads, 1);
        assert_eq!(speed_from(Some("7"), None, 2).threads, 7);
    }

    #[test]
    fn a_failed_load_is_the_same_error_for_every_caller() {
        // no model directory: the load fails the same way whether the preload thread or the first ask runs it
        let e = load_slot(None, Precision::F32, "f32", SpeedOptions::default()).err();
        assert_eq!(e.as_deref(), Some("XERJ_FACTS_MODEL_DIR is not set"));
        let cell: Arc<OnceLock<Result<Arc<ModelSlot>, String>>> = Arc::new(OnceLock::new());
        let c2 = Arc::clone(&cell);
        std::thread::spawn(move || {
            c2.get_or_init(|| load_slot(None, Precision::F32, "f32", SpeedOptions::default()));
        })
        .join()
        .expect("preload thread");
        assert!(matches!(cell.get(), Some(Err(e)) if e == "XERJ_FACTS_MODEL_DIR is not set"));
    }

    #[test]
    fn status_reports_a_pending_cleanup() {
        let n = Namespace::new();
        assert!(n.status(false).get("cleanup_pending").is_none());
        *n.pending_cleanup.lock().unwrap() = Some(Cleanup {
            orphans: vec!["f:x".into(), "e:0000000009".into()],
            repairs: vec![],
        });
        assert_eq!(
            n.status(false)["cleanup_pending"],
            json!({"orphans": 2, "repairs": 0})
        );
    }

    #[test]
    fn eviction_keeps_the_most_recently_used_projections() {
        let names = |svc: &FactsService| -> Vec<String> {
            svc.loaded().into_iter().map(|(k, _)| k).collect()
        };
        let mut svc = FactsService::from_env();
        svc.max_resident = Some(2);
        assert!(svc.insert("a", Arc::new(Namespace::new())).is_empty());
        assert!(svc.insert("b", Arc::new(Namespace::new())).is_empty());
        assert!(svc.get("a").is_some(), "a is now more recent than b");
        assert_eq!(svc.insert("c", Arc::new(Namespace::new())), vec!["b"]);
        assert_eq!(names(&svc), vec!["a", "c"]);
        // reloading a loaded namespace replaces it and evicts nothing
        assert!(svc.insert("c", Arc::new(Namespace::new())).is_empty());
        assert_eq!(svc.insert("d", Arc::new(Namespace::new())), vec!["a"]);
        assert_eq!(names(&svc), vec!["c", "d"]);
        assert_eq!(svc.residency(), (Some(2), 2));
        // the default keeps every projection
        let mut all = FactsService::from_env();
        all.max_resident = None;
        for ns in ["a", "b", "c", "d"] {
            assert!(all.insert(ns, Arc::new(Namespace::new())).is_empty());
        }
        assert_eq!(names(&all).len(), 4);
        assert_eq!(all.residency(), (None, 0));
    }

    /// `GET /_facts/{ns}/{fact_id}`: one record with its visibility at a committed revision; superseded and withdrawn
    /// records stay readable; nothing of a batch that is not published yet is shown.
    #[test]
    fn get_fact_reads_one_record_at_a_committed_revision() {
        let n = Namespace::new();
        let ap = n
            .apply(&Batch::World {
                world: world_with_edits(),
            })
            .expect("apply");
        // applied but not published: not shown
        assert!(get_fact(&n, "f1", None).unwrap().is_none());
        n.published.store(ap.to_rev, Ordering::SeqCst);
        // f3 corrects f2 (supersedes): at the head f2 is hidden and f3 visible
        let f2 = get_fact(&n, "f2", None).unwrap().expect("f2");
        let f3 = get_fact(&n, "f3", None).unwrap().expect("f3");
        assert_eq!(
            (f2["visible"].as_bool(), f3["visible"].as_bool()),
            (Some(false), Some(true))
        );
        assert_eq!(
            (&f3["subject"], &f3["relation"], &f3["kind"]),
            (&json!("Ana Bel"), &json!("height"), &json!("number"))
        );
        assert!(f3["value"].as_str().unwrap().contains("171"), "{f3}");
        assert_eq!(f3["source"], json!({"doc_id": "d2", "version": "v2"}));
        assert_eq!(f3["snapshot"]["revision"], ap.to_rev);
        // just before the correction: f2 visible, f3 not
        let before = f3["added_rev"].as_u64().unwrap() as Rev - 1;
        let f2b = get_fact(&n, "f2", Some(before)).unwrap().expect("f2");
        let f3b = get_fact(&n, "f3", Some(before)).unwrap().expect("f3");
        assert_eq!(
            (f2b["visible"].as_bool(), f3b["visible"].as_bool()),
            (Some(true), Some(false))
        );
        // f1's document was withdrawn: hidden at the head, still readable
        assert_eq!(
            get_fact(&n, "f1", None).unwrap().expect("f1")["visible"],
            false
        );
        assert!(get_fact(&n, "nope", None).unwrap().is_none());
        assert!(matches!(
            get_fact(&n, "f1", Some(ap.to_rev + 1)),
            Err(FactsError::NotCommitted(_))
        ));
    }
}
