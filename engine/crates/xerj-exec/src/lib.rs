//! # xerj-exec — the exact executor of the XERJ-R memory
//!
//! * [`result`]: `reference_core.Result`, `Executor.compare` / `negate`, `xr_codecs.result_text`.
//! * [`episode`]: `stack_runner.Episode` (stack-v1: LOOKUP LT GT EQ NOT RETURN ABSTAIN), `stack_v2.EpisodeV2` (+ CHAIN
//!   with per-hop manifests and id / normalized-string resolution, PICK_LT / PICK_GT with the year granularity rule) and
//!   `stack_v5.EpisodeV5` (`Protocol::V5`: + traversal direction, LOOKUP / CHAIN on inverse candidates), answer records with
//!   programs and execution traces; `run_gold` replays gold traces (parity / teacher programs).
//! * [`verify`]: the deterministic verify rule of tools/verify_rows.py.
//! * [`scorer`]: `scorer.normalize_answer` (parity checks).

pub mod episode;
pub mod result;
pub mod scorer;
pub mod verify;

pub use episode::{run_gold, Action, Answer, Episode, GoldStep, LoopGuard, Protocol, TraceEntry};
pub use result::{compare, negate, result_text, CmpOp, Res};
pub use verify::{verify, Verdict, VerifyMode};
