//! xerj-infer — CPU inference engine for the trained XERJ-R executor models (17M; the 68M stack-v5 direction executor).
//!
//! * arms: `rrt` (memory path: C1b memory attention + register attention after every decoder layer, pointer over records) and
//!   `external_executor` (candidate lines in the text, pointer over their pooled hidden states); stack protocols v1 and v2,
//!   and v5 (stack-v2 + traversal direction: inverse candidates, external_executor only)
//! * backbone: any Ettin decoder size, shapes from the export's config.json (ModernBERT-decoder: e.g. 17M = 7 layers, d=256,
//!   4 heads, GeGLU 384; 68M = 19 layers, d=512, 8 heads, GeGLU 768; RoPE θ=160000, global attention every third layer,
//!   sliding window 64 otherwise)
//! * precisions: exact f32 ([`Precision::F32`], parity reference) and int8 ([`Precision::Int8`]: per-channel int8 weights ×
//!   per-row dynamic int8 activations with SMMLA / SDOT / AVX-512 VNNI / AVX2 kernels selected at run time)
//!   Weight-only INT8 ([`Precision::W8`]): int8 weights, dequantized per call, f32 activations; the embedding table stays f32
//! * exact prefix reuse across the decisions of one question (see [`session`])
//!
//! The exact executor (stack runner, store, manifests) is outside this crate; [`episode::ActionExecutor`] is the interface.

pub mod codec;
pub mod config;
pub mod episode;
pub mod error;
pub mod fixtures;
pub mod inputs;
pub mod kernels;
pub mod model;
pub mod pool;
pub mod prof;
pub mod safetensors;
pub mod session;
pub mod sha256;
pub mod tokenizer;

pub use config::{Arm, MemoryCodec, ModelConfig, StackProtocol};
pub use episode::{run_episode, ActionExecutor, HopView};
pub use error::{Error, Result};
pub use inputs::CandidateFields;
pub use kernels::{CpuFeatures, F32Isa, I8Isa, Kernels};
pub use model::{Int8Calib, LoadOptions, Model, Precision};
pub use session::{DecisionOut, Session, SessionBufs};
pub use tokenizer::TextEncoder;

use std::path::Path;

/// model + tokenizer, shared (read-only) by any number of sessions / threads
pub struct Engine {
    pub model: Model,
    pub tok: Box<dyn TextEncoder>,
    pub tokenizer_ms: f64,
}

impl Engine {
    /// load an export directory with the HF reference tokenizer (`<dir>/tokenizer.json`, else the path recorded in
    /// config.json; its sha256 must equal the recorded one)
    #[cfg(feature = "tokenizer")]
    pub fn load(dir: &Path, precision: Precision, kernels: Kernels) -> Result<Engine> {
        Self::load_with(dir, &model::LoadOptions::new(precision, kernels))
    }

    /// load with explicit options (int8 calibration / guard)
    #[cfg(feature = "tokenizer")]
    pub fn load_with(dir: &Path, opt: &model::LoadOptions) -> Result<Engine> {
        let model = Model::load_with(dir, opt)?;
        let t0 = std::time::Instant::now();
        let local = dir.join("tokenizer.json");
        let path = if local.exists() {
            local
        } else {
            model
                .cfg
                .tokenizer_path
                .clone()
                .ok_or_else(|| Error::Format("no tokenizer.json".into()))?
        };
        if let Some(want) = &model.cfg.tokenizer_sha256 {
            let got = tokenizer::sha256_file(&path)?;
            if &got != want {
                return Err(Error::Tokenizer(format!(
                    "{} sha256 {got} != exported {want}",
                    path.display()
                )));
            }
        }
        let tok = tokenizer::HfTokenizer::from_file(&path)?;
        let tokenizer_ms = t0.elapsed().as_secs_f64() * 1e3;
        Ok(Engine {
            model,
            tok: Box::new(tok),
            tokenizer_ms,
        })
    }

    /// load with a caller-provided text encoder
    pub fn with_encoder(
        dir: &Path,
        precision: Precision,
        kernels: Kernels,
        tok: Box<dyn TextEncoder>,
    ) -> Result<Engine> {
        let model = Model::load(dir, precision, kernels)?;
        Ok(Engine {
            model,
            tok,
            tokenizer_ms: 0.0,
        })
    }

    pub fn session(&self) -> Session<'_> {
        Session::new(&self.model, self.tok.as_ref())
    }
}
