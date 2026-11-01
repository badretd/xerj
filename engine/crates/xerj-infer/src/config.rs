//! Model configuration exported by `research/xerj_r_mvp/tools/xerj_infer_export.py` (17M, stack v1 / v2) and
//! `tools/xerj_infer_export_v5.py` (dev_v5 code, any Ettin size, stack v5) (`config.json`, format `xerj-infer-export-v1`; the v5
//! exporter only adds optional keys).

use crate::error::{Error, Result};
use serde_json::Value;
use std::path::{Path, PathBuf};

/// The two executor arms the engine runs (arms.py).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arm {
    /// memory path: C1/C1b memory attention after every decoder layer + structured register attention; pointer over records
    Rrt,
    /// text path: the 16 candidate lines (and stack-v2 hop lines) are part of the decision text; pointer = attention from the
    /// decision state to the mean-pooled final hidden states of each line (+ a learned NULL key)
    ExternalExecutor,
}

impl Arm {
    pub fn name(self) -> &'static str {
        match self {
            Arm::Rrt => "rrt",
            Arm::ExternalExecutor => "external_executor",
        }
    }
}

/// Stack protocol the checkpoint was trained with (stack_runner.py / stack_v2.py / dev_v5 stack_v5.py).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StackProtocol {
    V1,
    V2,
    /// stack-v2 + traversal direction (dev_v5 `stack_v5.py`, `--direction`): the same ops, legality masks, pointer layout and
    /// hop manifests as V2; candidates may be INVERSE (followed from value to subject), rendered as `[k] <= ...` lines, and an
    /// executed inverse hop renders ` IN` after its pointer in the history
    V5,
}

impl StackProtocol {
    pub fn name(self) -> &'static str {
        match self {
            StackProtocol::V1 => "v1",
            StackProtocol::V2 => "v2",
            StackProtocol::V5 => "v5",
        }
    }

    /// stack-v2 family (V2, V5): CHAIN / PICK operations, hop manifests, pointer layout [base K | hop K | NULL]
    pub fn has_hops(self) -> bool {
        matches!(self, StackProtocol::V2 | StackProtocol::V5)
    }
}

/// Memory codec of the rrt arm (xr_codecs.py).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryCodec {
    /// subject / relation phrase / typed value / quote, raw mean-pooled input embeddings (4 slots per record)
    C1,
    /// subject + relation phrase with a leading space, RMS 1 (2 slots per record)
    C1b,
}

impl MemoryCodec {
    pub fn slots(self) -> usize {
        match self {
            MemoryCodec::C1 => 4,
            MemoryCodec::C1b => 2,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub arm: Arm,
    pub stack: StackProtocol,
    pub codec: MemoryCodec,
    /// candidate manifest budget (records per manifest)
    pub k: usize,
    pub hidden: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub layers: usize,
    pub intermediate: usize,
    pub vocab: usize,
    /// per layer: sliding-window (local) attention
    pub sliding: Vec<bool>,
    /// keys j visible from query i on a sliding layer: i - window < j <= i
    pub window: usize,
    pub rope_theta_global: f64,
    pub rope_theta_local: f64,
    pub norm_eps: f32,
    pub max_positions: usize,
    pub n_ops: usize,
    pub memory_width: usize,
    pub register_width: usize,
    pub register_count: usize,
    pub cond_eps: f32,
    pub capacity_bottleneck: Option<usize>,
    pub capacity_eps: f32,
    pub bos: u32,
    pub pad: u32,
    pub tokenizer_path: Option<PathBuf>,
    pub tokenizer_sha256: Option<String>,
    pub fingerprint: Option<String>,
}

fn get<'a>(v: &'a Value, path: &[&str]) -> Result<&'a Value> {
    let mut cur = v;
    for p in path {
        cur = cur
            .get(*p)
            .ok_or_else(|| Error::Format(format!("config.json: missing {}", path.join("."))))?;
    }
    Ok(cur)
}

fn get_usize(v: &Value, path: &[&str]) -> Result<usize> {
    get(v, path)?
        .as_u64()
        .map(|x| x as usize)
        .ok_or_else(|| Error::Format(format!("config.json: {} is not an integer", path.join("."))))
}

fn get_f64(v: &Value, path: &[&str]) -> Result<f64> {
    get(v, path)?
        .as_f64()
        .ok_or_else(|| Error::Format(format!("config.json: {} is not a number", path.join("."))))
}

fn get_str<'a>(v: &'a Value, path: &[&str]) -> Result<&'a str> {
    get(v, path)?
        .as_str()
        .ok_or_else(|| Error::Format(format!("config.json: {} is not a string", path.join("."))))
}

impl ModelConfig {
    pub fn from_file(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| Error::Io(format!("{}: {e}", path.display())))?;
        let v: Value = serde_json::from_str(&text)
            .map_err(|e| Error::Format(format!("{}: {e}", path.display())))?;
        Self::from_json(&v)
    }

    pub fn from_json(v: &Value) -> Result<Self> {
        let fmt = get_str(v, &["format"])?;
        if fmt != "xerj-infer-export-v1" {
            return Err(Error::Format(format!("unsupported export format {fmt}")));
        }
        let arm = match get_str(v, &["arm"])? {
            "rrt" => Arm::Rrt,
            "external_executor" => Arm::ExternalExecutor,
            other => return Err(Error::Unsupported(format!("arm {other}"))),
        };
        let stack = match get_str(v, &["stack"])? {
            "v1" => StackProtocol::V1,
            "v2" => StackProtocol::V2,
            "v5" => StackProtocol::V5,
            other => return Err(Error::Unsupported(format!("stack protocol {other}"))),
        };
        if arm == Arm::Rrt && stack == StackProtocol::V5 {
            // the rrt memory codec would carry the inverse flag in role column 2 of the subject slot (xr_codecs.c1b_memory); no rrt
            // stack-v5 checkpoint exists to verify it against, so it is refused rather than run unverified
            return Err(Error::Unsupported(
                "rrt under stack v5 (not parity-tested: no rrt v5 checkpoint)".into(),
            ));
        }
        let codec = match get_str(v, &["memory_codec"])? {
            "C1" => MemoryCodec::C1,
            "C1b" => MemoryCodec::C1b,
            other => return Err(Error::Unsupported(format!("memory codec {other}"))),
        };
        let b = &["backbone"];
        let hidden = get_usize(v, &[b[0], "hidden_size"])?;
        let heads = get_usize(v, &[b[0], "num_attention_heads"])?;
        let layers = get_usize(v, &[b[0], "num_hidden_layers"])?;
        let sliding: Vec<bool> = get(v, &[b[0], "sliding"])?
            .as_array()
            .ok_or_else(|| Error::Format("backbone.sliding".into()))?
            .iter()
            .map(|x| x.as_bool().unwrap_or(false))
            .collect();
        if sliding.len() != layers {
            return Err(Error::Format(
                "backbone.sliding length != num_hidden_layers".into(),
            ));
        }
        for flag in ["norm_bias", "attention_bias", "mlp_bias"] {
            if get(v, &[b[0], flag])?.as_bool() != Some(false) {
                return Err(Error::Unsupported(format!("backbone.{flag} = true")));
            }
        }
        if get_str(v, &[b[0], "hidden_activation"])? != "gelu" {
            return Err(Error::Unsupported(
                "hidden_activation other than exact gelu".into(),
            ));
        }
        let ex = &["executor"];
        let opt_usize = |name: &str| -> Option<usize> {
            v.get(ex[0])
                .and_then(|e| e.get(name))
                .and_then(|x| x.as_u64())
                .map(|x| x as usize)
        };
        let opt_f32 = |name: &str| -> Option<f32> {
            v.get(ex[0])
                .and_then(|e| e.get(name))
                .and_then(|x| x.as_f64())
                .map(|x| x as f32)
        };
        if arm == Arm::Rrt {
            if v.get("executor")
                .and_then(|e| e.get("shared_conditioner"))
                .and_then(|x| x.as_bool())
                != Some(true)
            {
                return Err(Error::Unsupported(
                    "rrt without a shared conditioner".into(),
                ));
            }
            let cl: Vec<usize> = get(v, &["executor", "conditioned_layers"])?
                .as_array()
                .ok_or_else(|| Error::Format("executor.conditioned_layers".into()))?
                .iter()
                .filter_map(|x| x.as_u64().map(|y| y as usize))
                .collect();
            if cl != (0..layers).collect::<Vec<_>>() {
                return Err(Error::Unsupported(format!(
                    "conditioned layers {cl:?} (expected every layer)"
                )));
            }
        }
        let tok = v.get("tokenizer");
        Ok(ModelConfig {
            arm,
            stack,
            codec,
            k: get_usize(v, &["K"])?,
            hidden,
            heads,
            head_dim: hidden / heads,
            layers,
            intermediate: get_usize(v, &[b[0], "intermediate_size"])?,
            vocab: get_usize(v, &[b[0], "vocab_size"])?,
            sliding,
            window: get_usize(v, &[b[0], "sliding_window"])?,
            rope_theta_global: get_f64(v, &[b[0], "global_rope_theta"])?,
            rope_theta_local: get_f64(v, &[b[0], "local_rope_theta"])?,
            norm_eps: get_f64(v, &[b[0], "norm_eps"])? as f32,
            max_positions: get_usize(v, &[b[0], "max_position_embeddings"])?,
            n_ops: get_usize(v, &["executor", "n_ops"])?,
            memory_width: opt_usize("memory_width").unwrap_or(0),
            register_width: opt_usize("register_width").unwrap_or(0),
            register_count: opt_usize("register_count").unwrap_or(0),
            cond_eps: opt_f32("conditioner_ln_eps").unwrap_or(1e-5),
            capacity_bottleneck: opt_usize("capacity_bottleneck"),
            capacity_eps: opt_f32("capacity_ln_eps").unwrap_or(1e-5),
            bos: get_usize(v, &["tokens", "bos"])? as u32,
            pad: get_usize(v, &["tokens", "pad"])? as u32,
            tokenizer_path: tok
                .and_then(|t| t.get("path"))
                .and_then(|p| p.as_str())
                .map(PathBuf::from),
            tokenizer_sha256: tok
                .and_then(|t| t.get("sha256"))
                .and_then(|p| p.as_str())
                .map(String::from),
            fingerprint: v
                .get("checkpoint")
                .and_then(|c| c.get("fingerprint"))
                .and_then(|f| f.as_str())
                .map(String::from),
        })
    }

    /// records the memory / pointer layout holds: K (stack-v1) or 2K (stack-v2 / v5: base manifest ++ hop manifest)
    pub fn records(&self) -> usize {
        if self.stack.has_hops() {
            2 * self.k
        } else {
            self.k
        }
    }

    /// divisor of the external executor's pointer logits: float32(sqrt(hidden)) (arms.Arm.decide divides by math.sqrt(d); for
    /// a power-of-two root such as 16 at d = 256 this equals the former multiplication by 1/16 bit for bit)
    pub fn pointer_divisor(&self) -> f32 {
        (self.hidden as f64).sqrt() as f32
    }
}
