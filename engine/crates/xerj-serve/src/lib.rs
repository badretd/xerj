//! xerj-serve: XERJ-R end to end in Rust (2026-10-22). Until now the Rust pieces were tested one by one (store / index /
//! executor parity with Python; xerj-infer decisions against recorded PyTorch episodes with a replay executor). This crate
//! joins them: a question goes through the store's linker and exact manifest, the 17M compiler (int8 + guard) decides on the
//! live exact executor, and the verify rule checks the cited subjects; `xerj-ask` exposes it as JSON lines for the
//! robustness tests and the XERJ Lab.
//!
//! * `adapter`: xerj-exec `Episode` as xerj-infer's `ActionExecutor` (candidate strings, history, registers, hop manifests)
//! * `engine`: model slots, world building (memory-world-v1 + applied edits), one answer with per-stage timings
//! * `tolerant`: optional typo / case tolerant entity linking with a visible rewrite (extension; exact is the reference)
//! * `textsearch`: sentence search over a pasted text (native BM25 + optional typo expansion)
//! * `verify_v5`: the direction-aware verify rule for stack-v5 answers (extension; the reference rule vetoes inverse answers)
//! * `verify_bound`: the exact-binding verify rule (prototype) and the opt-in `VerifyRule` choice (`AnswerOpts::verify`;
//!   default `Served` = `verify_v5`)
//! * `decline_v2`: the frozen decline policy v2 (P3 + D2 + D2b) as an opt-in verify rule (`VerifyRule::DeclineV2`), a port checked
//!   against the Python oracle

pub mod adapter;
pub mod decline_v2;
pub mod engine;
pub mod textsearch;
pub mod tolerant;
pub mod verify_bound;
pub mod verify_v5;
