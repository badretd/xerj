//! # xerj-index — linking, manifests and BM25 over the XERJ-R memory
//!
//! * [`subjects`]: subject / relation / entity-id record lists, normalized-name index (incremental, MVCC-aware).
//! * [`linker`]: `xr_codecs.link_entities`, exact, via span-hash probes or tiered Aho-Corasick automata.
//! * [`manifest`]: `candidate_manifest` and stack-v2 `hop_manifest`, exact (SHA-256 keys) or fast (xxh3 keys).
//! * [`direction`]: stack-v5 (traversal direction): `link_values`, base / hop manifests with inverse candidates (flagged per
//!   slot), the value-side tables of the executor's inverse checks (lazy, incremental).
//! * [`bm25`]: the BM25 seam (`Bm25Backend`) and the native segmented implementation (records and documents).
//! * [`memory`]: `Memory` = store + all indexes, read-your-writes, op log, checkpoints.
//! * [`scope`] (EXTENSION, opt-in): a host's record -> collections map and the scope one question may read (scoped
//!   manifests: `direction::candidate_manifest_v5_scope`, `direction::hop_manifest_v5_scope`).

pub mod bm25;
pub mod direction;
pub mod indexer;
pub mod linker;
pub mod manifest;
pub mod memory;
pub mod scope;
pub mod sha256x;
pub mod subjects;

pub use bm25::{Bm25Backend, Bm25Params, Hit, SegmentedBm25};
pub use direction::{CandManifest, RecScope, RelPref, ValueIndex, ValueTables};
pub use linker::{Linker, LinkerKind, QBuf};
pub use manifest::{candidate_manifest, hop_manifest, HopManifest, ManifestHash};
pub use memory::{rel_phrase, Memory, MemoryConfig, RecordText};
pub use scope::{QuestionScope, RecCollections, ScopeMode};
pub use subjects::SubjectIndex;
