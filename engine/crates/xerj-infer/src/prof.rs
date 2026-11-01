//! Opt-in wall-clock profiler (XERJ_PROFILE=1): per-thread accumulated time per component. `perf` is not available to
//! unprivileged users on the benchmark hosts, so the bench harness reports this breakdown instead.

use std::cell::RefCell;
use std::sync::OnceLock;
use std::time::Instant;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cat {
    Tokenize = 0,
    Embed,
    Norm,
    GemmF32,
    Quantize,
    GemmI8,
    Rope,
    SelfAttn,
    CrossAttn,
    Act,
    Codec,
    Heads,
}

pub const NAMES: [&str; 12] = [
    "tokenize",
    "embed",
    "layernorm",
    "gemm_f32",
    "quantize",
    "gemm_i8",
    "rope_kv",
    "self_attn",
    "cross_attn",
    "gelu_geglu",
    "memory_codec",
    "heads_pointer",
];

static ON: OnceLock<bool> = OnceLock::new();

thread_local! {
    static ACC: RefCell<[f64; 12]> = const { RefCell::new([0.0; 12]) };
}

#[inline(always)]
pub fn enabled() -> bool {
    *ON.get_or_init(|| {
        std::env::var("XERJ_PROFILE")
            .map(|v| v == "1")
            .unwrap_or(false)
    })
}

pub struct Timer(Option<(Cat, Instant)>);

#[inline(always)]
pub fn start(c: Cat) -> Timer {
    if enabled() {
        Timer(Some((c, Instant::now())))
    } else {
        Timer(None)
    }
}

impl Drop for Timer {
    #[inline(always)]
    fn drop(&mut self) {
        if let Some((c, t)) = self.0 {
            let dt = t.elapsed().as_secs_f64();
            ACC.with(|a| a.borrow_mut()[c as usize] += dt);
        }
    }
}

/// (name, seconds) of this thread, and reset
pub fn take() -> Vec<(&'static str, f64)> {
    ACC.with(|a| {
        let mut a = a.borrow_mut();
        let v = NAMES.iter().zip(a.iter()).map(|(n, s)| (*n, *s)).collect();
        *a = [0.0; 12];
        v
    })
}
