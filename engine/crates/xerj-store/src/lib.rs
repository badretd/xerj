//! # xerj-store — the versioned record store of the XERJ-R connected memory
//!
//! Python-exact port of `reference_core.MemoryStore` / `Snapshot` (research/xerj_r_mvp/reference_pkg) as used by
//! `xr_codecs.world_store`, built for speed: columnar records, interned strings, a (subject, relation, scope) key index,
//! MVCC-lite revisions (snapshots at any revision are O(1)), an append-only binary op log and mmap checkpoints.
//!
//! * [`store`]: `Store` (writes with the reference's contract checks), `Snapshot` (visibility, lookup, fingerprint).
//! * [`world`]: memory-world-v1 JSON and the reference ingest order (`ingest_world`).
//! * [`value`]: typed values, exact decimals and dates, rendering.
//! * [`text`]: Python-exact `\w`, casefold, punctuation, whitespace, JSON escaping, `normalize_entity`.
//! * [`log`], [`ckpt`], [`checkpoint`]: persistence.

pub mod checkpoint;
pub mod ckpt;
pub mod error;
pub mod log;
pub mod pool;
pub mod store;
pub mod text;
pub mod value;
pub mod world;

pub use error::{ContractError, StoreError};
pub use pool::StrPool;
pub use store::{
    AddedFact, DocIx, FactIn, FingerprintMode, Lookup, RecId, Rev, Snapshot, Status, Store,
    StoreConfig, ValidAt, ValueIn, DEFAULT_SCOPE, NONE, OPEN, ORACLE_WRITER, VALID_AT,
};
pub use value::{Kind, TypedValue};
pub use world::{ingest_world, World, WorldRevs, WriteSink};
