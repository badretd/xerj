//! Model weights in run-time layout and the forward building blocks (ModernBERT-decoder layers, the shared rrt
//! ReferenceConditioner, the external executor's capacity block, the executor heads).
//!
//! Reproduces (transformers 4.57.1 `modeling_modernbert_decoder.py`, reference_pkg `model_reference.py` / `ettin_adapter.py`,
//! future_scientific `arms.py`):
//! * embeddings: LayerNorm(E[id]) (no bias); layer 0 has no attention norm (identity)
//! * attention: q/k/v projections (no bias), RoPE (rotate-half, θ = 160000), scale 1/√64, causal; sliding layers see keys
//!   i-window < j <= i; Wo (no bias); residual
//! * MLP: Wi → [input | gate], gelu(input) * gate, Wo; residual (pre-norm, no bias anywhere)
//! * rrt: after EVERY layer the shared conditioner: h += tanh(g_m)·MHA(LN_m(h), memory) then, when registers exist,
//!   h += tanh(g_r)·MHA(LN_r(h), registers) (nn.MultiheadAttention, 4 heads, key padding masks, NULL token always visible)
//! * external_executor: capacity block after the last layer: h + up(gelu(down(LN(h))))
//! * final LayerNorm; op head (10) and pointer query on the last token.

use crate::config::{Arm, ModelConfig};
use crate::error::{Error, Result};
use crate::kernels::{self, Epi, GemmScratch, Kernels, PackedF32, PackedI8, PackedW8};
use crate::safetensors::SafeTensors;
use std::path::Path;
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Precision {
    /// exact f32 path (SIMD f32 GEMM): parity reference
    F32,
    /// int8 weights (per-channel) × int8 activations (per-row, dynamic) for every backbone / conditioner / capacity-block
    /// projection; embeddings, memory/register projections, memory/register K/V and heads stay f32
    Int8,
    /// weight-only INT8: every linear of the external executor (backbone, capacity block, heads) holds int8 weights with one
    /// f32 scale per output channel (tools/int8_fakequant.py's values) and computes in f32, dequantizing on every call.
    /// Activations stay f32; the embedding table, which the output head is tied to, and the norms stay f32
    W8,
}

impl Precision {
    pub fn name(self) -> &'static str {
        match self {
            Precision::F32 => "f32",
            Precision::Int8 => "int8",
            Precision::W8 => "w8",
        }
    }
}

/// a linear layer y = x·Wᵀ + b in the run-time format of its precision. Int8 linears may carry a smoothing vector (activations
/// are multiplied by `pre` before quantization, weights were multiplied by its inverse) and, for the int8 guard, the f32
/// weights as well.
pub struct Linear {
    pub n: usize,
    pub k: usize,
    pub name: String,
    pub id: usize,
    f32w: Option<PackedF32>,
    i8w: Option<PackedI8>,
    w8w: Option<PackedW8>,
    pre: Option<Vec<f32>>,
    #[cfg(all(target_arch = "aarch64", xerj_kleidiai))]
    kai: Option<kernels::kleidi::KaiRhs>,
    /// output-channel shards (first channel, weights) for intra-question parallelism; empty = the weights above. A sharded
    /// linear holds no full weights: every output channel is computed by exactly one shard with the same kernel and inputs
    /// as unsharded, so the results are bit-identical whatever the thread count.
    shards: Vec<(usize, Linear)>,
}

/// smallest GEMM (m * n * k multiply-adds) that runs its shards in parallel
const PAR_MIN_MACS: usize = 1 << 18;

/// how to build one linear
#[derive(Clone, Copy)]
pub enum LinMode<'a> {
    F32,
    I8 {
        smooth: Option<&'a [f32]>,
        keep_f32: bool,
    },
    W8,
}

impl Linear {
    pub fn new(
        name: &str,
        id: usize,
        w: &[f32],
        n: usize,
        k: usize,
        bias: Option<&[f32]>,
        mode: LinMode,
    ) -> Self {
        match mode {
            LinMode::F32 => Linear {
                n,
                k,
                name: name.into(),
                id,
                f32w: Some(PackedF32::pack(w, n, k, bias)),
                i8w: None,
                w8w: None,
                pre: None,
                #[cfg(all(target_arch = "aarch64", xerj_kleidiai))]
                kai: None,
                shards: Vec::new(),
            },
            LinMode::W8 => Linear {
                n,
                k,
                name: name.into(),
                id,
                f32w: None,
                i8w: None,
                w8w: Some(PackedW8::quantize(w, n, k, bias)),
                pre: None,
                #[cfg(all(target_arch = "aarch64", xerj_kleidiai))]
                kai: None,
                shards: Vec::new(),
            },
            LinMode::I8 { smooth, keep_f32 } => {
                let f32w = if keep_f32 {
                    Some(PackedF32::pack(w, n, k, bias))
                } else {
                    None
                };
                let (i8w, pre) = match smooth {
                    Some(sv) => {
                        let mut ws = w.to_vec();
                        for c in 0..n {
                            for j in 0..k {
                                ws[c * k + j] *= sv[j];
                            }
                        }
                        (
                            PackedI8::quantize(&ws, n, k, bias),
                            Some(sv.iter().map(|v| 1.0 / v).collect()),
                        )
                    }
                    None => (PackedI8::quantize(w, n, k, bias), None),
                };
                Linear {
                    n,
                    k,
                    name: name.into(),
                    id,
                    f32w,
                    #[cfg(all(target_arch = "aarch64", xerj_kleidiai))]
                    kai: if kernels::kleidi_available() {
                        Some(kernels::kleidi::KaiRhs::pack(
                            &i8w.row_major(),
                            n,
                            k,
                            &i8w.scales[..n],
                            i8w.bias.as_deref(),
                        ))
                    } else {
                        None
                    },
                    i8w: Some(i8w),
                    w8w: None,
                    pre,
                    shards: Vec::new(),
                }
            }
        }
    }

    pub fn is_int8(&self) -> bool {
        self.i8w.is_some() || self.shards.first().is_some_and(|(_, l)| l.i8w.is_some())
    }

    /// as `new`, split into `parts` output-channel shards of a multiple of 16 channels (no split when parts <= 1 or the
    /// layer is narrower than 32 channels)
    #[allow(clippy::too_many_arguments)]
    pub fn new_sharded(
        name: &str,
        id: usize,
        w: &[f32],
        n: usize,
        k: usize,
        bias: Option<&[f32]>,
        mode: LinMode,
        parts: usize,
    ) -> Self {
        if parts <= 1 || n < 32 {
            return Linear::new(name, id, w, n, k, bias, mode);
        }
        let step = n.div_ceil(parts).div_ceil(16) * 16;
        let mut shards = Vec::new();
        let mut c0 = 0;
        while c0 < n {
            let c1 = (c0 + step).min(n);
            let sh = Linear::new(
                name,
                id,
                &w[c0 * k..c1 * k],
                c1 - c0,
                k,
                bias.map(|b| &b[c0..c1]),
                mode,
            );
            shards.push((c0, sh));
            c0 = c1;
        }
        let pre = shards[0].1.pre.clone();
        Linear {
            n,
            k,
            name: name.into(),
            id,
            f32w: None,
            i8w: None,
            w8w: None,
            pre,
            #[cfg(all(target_arch = "aarch64", xerj_kleidiai))]
            kai: None,
            shards,
        }
    }

    /// number of output-channel shards (0 = unsharded)
    pub fn shard_count(&self) -> usize {
        self.shards.len()
    }

    fn par_pool(
        &self,
        m: usize,
        simd: bool,
        s: &GemmScratch,
    ) -> Option<std::sync::Arc<crate::pool::Pool>> {
        s.pool
            .as_ref()
            .filter(|p| p.threads() > 1 && simd && m * self.n * self.k >= PAR_MIN_MACS)
            .cloned()
    }

    /// sharded f32 GEMM (parallel over the shards when a pool is attached and the GEMM is large enough)
    #[allow(clippy::too_many_arguments)]
    fn shards_f32(
        &self,
        kern: &Kernels,
        x: &[f32],
        m: usize,
        y: &mut [f32],
        ldy: usize,
        epi: Epi,
        s: &mut GemmScratch,
    ) {
        let _t = crate::prof::start(crate::prof::Cat::GemmF32);
        assert!(y.len() >= (m - 1) * ldy + self.n && x.len() >= m * self.k);
        let yp = crate::pool::SendPtr(y.as_mut_ptr());
        match self.par_pool(m, kern.f32_isa != kernels::F32Isa::Scalar, s) {
            Some(pool) => {
                let t = pool.threads();
                if s.tls.len() < t {
                    s.tls.resize_with(t, Default::default);
                }
                let tls = &s.tls;
                pool.run(self.shards.len(), &|i| {
                    let (c0, sh) = &self.shards[i];
                    let mut ap = tls[i % t].lock().unwrap_or_else(|e| e.into_inner());
                    // SAFETY: shard i writes columns c0..c0 + sh.n of rows 0..m only (disjoint between shards; y bounds
                    // asserted above); the caller waits for every shard
                    unsafe {
                        kern.gemm_f32_ptr(
                            x,
                            m,
                            sh.f32w.as_ref().unwrap(),
                            yp.get().add(*c0),
                            ldy,
                            epi,
                            &mut ap,
                        )
                    };
                });
            }
            None => {
                for (c0, sh) in &self.shards {
                    // SAFETY: as above, one shard at a time
                    unsafe {
                        kern.gemm_f32_ptr(
                            x,
                            m,
                            sh.f32w.as_ref().unwrap(),
                            yp.get().add(*c0),
                            ldy,
                            epi,
                            &mut s.apack,
                        )
                    };
                }
            }
        }
    }

    /// sharded weight-only INT8 GEMM: each shard is dequantized and computed by the f32 kernel, one participant per shard
    #[allow(clippy::too_many_arguments)]
    fn shards_w8(
        &self,
        kern: &Kernels,
        x: &[f32],
        m: usize,
        y: &mut [f32],
        ldy: usize,
        epi: Epi,
        s: &mut GemmScratch,
    ) {
        let _t = crate::prof::start(crate::prof::Cat::GemmF32);
        assert!(y.len() >= (m - 1) * ldy + self.n && x.len() >= m * self.k);
        let yp = crate::pool::SendPtr(y.as_mut_ptr());
        match self.par_pool(m, kern.f32_isa != kernels::F32Isa::Scalar, s) {
            Some(pool) => {
                let t = pool.threads();
                if s.tls.len() < t {
                    s.tls.resize_with(t, Default::default);
                }
                if s.tls_w8.len() < t {
                    s.tls_w8.resize_with(t, Default::default);
                }
                let (tls, tls_w8) = (&s.tls, &s.tls_w8);
                pool.run(self.shards.len(), &|i| {
                    let (c0, sh) = &self.shards[i];
                    let mut ap = tls[i % t].lock().unwrap_or_else(|e| e.into_inner());
                    let mut w = tls_w8[i % t].lock().unwrap_or_else(|e| e.into_inner());
                    sh.w8w.as_ref().unwrap().dequant_into(&mut w);
                    // SAFETY: shard i writes columns c0..c0 + sh.n of rows 0..m only (disjoint between shards; y bounds
                    // asserted above); the caller waits for every shard
                    unsafe { kern.gemm_f32_ptr(x, m, &w, yp.get().add(*c0), ldy, epi, &mut ap) };
                });
            }
            None => {
                for (c0, sh) in &self.shards {
                    sh.w8w.as_ref().unwrap().dequant_into(&mut s.w8);
                    // SAFETY: as above, one shard at a time
                    unsafe {
                        kern.gemm_f32_ptr(x, m, &s.w8, yp.get().add(*c0), ldy, epi, &mut s.apack)
                    };
                }
            }
        }
    }

    /// sharded int8 GEMM of the activations already quantized into s.q
    fn shards_i8(&self, kern: &Kernels, y: &mut [f32], ldy: usize, epi: Epi, s: &GemmScratch) {
        let _t = crate::prof::start(crate::prof::Cat::GemmI8);
        let m = s.q.m;
        if m == 0 {
            return;
        }
        assert!(y.len() >= (m - 1) * ldy + self.n);
        let yp = crate::pool::SendPtr(y.as_mut_ptr());
        let q = &s.q;
        let run = |i: usize| {
            let (c0, sh) = &self.shards[i];
            // SAFETY: shard i writes columns c0..c0 + sh.n of rows 0..m only (disjoint between shards; bounds asserted)
            unsafe { kern.gemm_i8_ptr(sh.i8w.as_ref().unwrap(), yp.get().add(*c0), ldy, epi, q) };
        };
        match self.par_pool(m, kern.i8_isa != kernels::I8Isa::Scalar, s) {
            Some(pool) => pool.run(self.shards.len(), &run),
            None => (0..self.shards.len()).for_each(run),
        }
    }

    /// forward of a sharded linear (see `forward`)
    #[allow(clippy::too_many_arguments)]
    fn forward_shards(
        &self,
        kern: &Kernels,
        x: &[f32],
        m: usize,
        y: &mut [f32],
        ldy: usize,
        epi: Epi,
        s: &mut GemmScratch,
    ) {
        let sh0 = &self.shards[0].1;
        if sh0.w8w.is_some() {
            return self.shards_w8(kern, x, m, y, ldy, epi, s);
        }
        let use_f32 = sh0.f32w.is_some() && (sh0.i8w.is_none() || s.force_f32);
        if use_f32 {
            if let Some(c) = s.calib.as_mut() {
                let v = &mut c[self.id];
                v.resize(self.k, 0.0);
                for r in 0..m {
                    for (j, a) in x[r * self.k..(r + 1) * self.k].iter().enumerate() {
                        v[j] = v[j].max(a.abs());
                    }
                }
            }
            self.shards_f32(kern, x, m, y, ldy, epi, s);
        } else {
            kern.quantize_par(x, m, self.k, self.pre.as_deref(), s);
            self.shards_i8(kern, y, ldy, epi, s);
        }
    }

    /// y[m, ldy] = epi(x[m, k]·Wᵀ + b)
    #[inline]
    pub fn forward(
        &self,
        kern: &Kernels,
        x: &[f32],
        m: usize,
        y: &mut [f32],
        ldy: usize,
        epi: Epi,
        s: &mut GemmScratch,
    ) {
        if m == 0 {
            return;
        }
        if !self.shards.is_empty() {
            return self.forward_shards(kern, x, m, y, ldy, epi, s);
        }
        let use_f32 = self.f32w.is_some() && (self.i8w.is_none() || s.force_f32);
        if use_f32 {
            if let Some(c) = s.calib.as_mut() {
                let v = &mut c[self.id];
                v.resize(self.k, 0.0);
                for r in 0..m {
                    for (j, a) in x[r * self.k..(r + 1) * self.k].iter().enumerate() {
                        v[j] = v[j].max(a.abs());
                    }
                }
            }
            kern.gemm_f32(x, m, self.f32w.as_ref().unwrap(), y, ldy, epi, s);
        } else if let Some(w) = &self.w8w {
            let mut d = std::mem::take(&mut s.w8);
            w.dequant_into(&mut d);
            kern.gemm_f32(x, m, &d, y, ldy, epi, s);
            s.w8 = d;
        } else if let Some(w) = &self.i8w {
            #[cfg(all(target_arch = "aarch64", xerj_kleidiai))]
            if kern.kleidi && m >= kernels::kleidi::KAI_MIN_M {
                if let Some(kw) = &self.kai {
                    let _t = crate::prof::start(crate::prof::Cat::GemmI8);
                    kernels::kleidi::gemm(kw, x, m, self.pre.as_deref(), y, ldy, epi, &mut s.kai);
                    return;
                }
            }
            kern.quantize(x, m, self.k, self.pre.as_deref(), s);
            kern.gemm_i8(w, y, ldy, epi, s);
        }
    }

    pub fn bytes(&self) -> usize {
        #[cfg(all(target_arch = "aarch64", xerj_kleidiai))]
        let kai = self.kai.as_ref().map_or(0, |w| w.bytes());
        #[cfg(not(all(target_arch = "aarch64", xerj_kleidiai)))]
        let kai = 0;
        self.f32w.as_ref().map_or(0, |w| w.bytes())
            + self.i8w.as_ref().map_or(0, |w| w.bytes())
            + self.w8w.as_ref().map_or(0, |w| w.bytes())
            + self.pre.as_ref().map_or(0, |p| 4 * p.len())
            + kai
            + self.shards.iter().map(|(_, l)| l.bytes()).sum::<usize>()
    }

    /// y = epi(LayerNorm(x)·Wᵀ + b): int8 linears fuse the LayerNorm into the activation quantization; otherwise the
    /// LayerNorm output goes through `xn`
    #[allow(clippy::too_many_arguments)]
    pub fn forward_ln(
        &self,
        kern: &Kernels,
        x: &[f32],
        m: usize,
        ln_w: &[f32],
        ln_b: Option<&[f32]>,
        eps: f32,
        xn: &mut Vec<f32>,
        y: &mut [f32],
        ldy: usize,
        epi: Epi,
        s: &mut GemmScratch,
    ) {
        if m == 0 {
            return;
        }
        if !self.shards.is_empty() {
            let sh0 = &self.shards[0].1;
            let use_f32 = sh0.f32w.is_some() && (sh0.i8w.is_none() || s.force_f32);
            if !use_f32
                && sh0.w8w.is_none()
                && kern.quantize_ln_par(x, m, self.k, ln_w, ln_b, eps, self.pre.as_deref(), s)
            {
                self.shards_i8(kern, y, ldy, epi, s);
                return;
            }
            grow(xn, m * self.k);
            {
                let _t = crate::prof::start(crate::prof::Cat::Norm);
                // rows over the pool (bit-identical: row-wise)
                kernels::ops::layer_norm_pool(x, m, self.k, ln_w, ln_b, eps, xn, s.pool.as_deref());
            }
            self.forward_shards(kern, &xn[..m * self.k], m, y, ldy, epi, s);
            return;
        }
        let use_f32 = self.f32w.is_some() && (self.i8w.is_none() || s.force_f32);
        #[cfg(all(target_arch = "aarch64", xerj_kleidiai))]
        let use_kai = kern.kleidi && m >= kernels::kleidi::KAI_MIN_M && self.kai.is_some();
        #[cfg(not(all(target_arch = "aarch64", xerj_kleidiai)))]
        let use_kai = false;
        if !use_f32 && !use_kai {
            if let Some(w) = &self.i8w {
                if kern.quantize_ln(x, m, self.k, ln_w, ln_b, eps, self.pre.as_deref(), s) {
                    kern.gemm_i8(w, y, ldy, epi, s);
                    return;
                }
            }
        }
        grow(xn, m * self.k);
        {
            let _t = crate::prof::start(crate::prof::Cat::Norm);
            kernels::ops::layer_norm(x, m, self.k, ln_w, ln_b, eps, xn);
        }
        self.forward(kern, &xn[..m * self.k], m, y, ldy, epi, s);
    }

    /// max |W[:, j]| per input channel (from the f32 weights)
    pub fn col_absmax(&self) -> Option<Vec<f32>> {
        if !self.shards.is_empty() {
            let mut out = vec![0f32; self.k];
            for (_, l) in &self.shards {
                for (o, v) in out.iter_mut().zip(l.col_absmax()?) {
                    *o = o.max(v);
                }
            }
            return Some(out);
        }
        let w = self.f32w.as_ref()?;
        let d = w.data.as_slice();
        let mut out = vec![0f32; self.k];
        for p in 0..w.panels() {
            for kk in 0..self.k {
                for j in 0..crate::kernels::NR {
                    out[kk] = out[kk].max(d[(p * self.k + kk) * crate::kernels::NR + j].abs());
                }
            }
        }
        Some(out)
    }

    /// is this linear quantized under int8 (and so a smoothing / calibration target)
    pub fn quantizable_name(name: &str) -> bool {
        name.starts_with("layers.")
            || name == "cond.mem.q"
            || name == "cond.mem.out"
            || name == "cond.reg.q"
            || name == "cond.reg.out"
            || name.starts_with("cap.")
    }

    /// bytes of the weights the int8 path reads (int8 part only when both formats are held)
    pub fn fast_bytes(&self) -> usize {
        if !self.shards.is_empty() {
            return self.shards.iter().map(|(_, l)| l.fast_bytes()).sum();
        }
        if let Some(w) = &self.w8w {
            return w.bytes();
        }
        match &self.i8w {
            Some(w) => w.bytes() + self.pre.as_ref().map_or(0, |p| 4 * p.len()),
            None => self.f32w.as_ref().map_or(0, |w| w.bytes()),
        }
    }
}

/// int8 calibration (int8_calib.json): smoothing vectors per linear and the guard thresholds
#[derive(Debug, Clone, Default)]
pub struct Int8Calib {
    pub alpha: f32,
    pub smooth: std::collections::HashMap<String, Vec<f32>>,
    /// guard: an int8 decision whose op margin (top-1 minus top-2 legal op logit) is below tau_op, or whose pointer margin is
    /// below tau_ptr, is recomputed with the f32 weights
    pub tau_op: f32,
    pub tau_ptr: f32,
    pub max_err_op: f32,
    pub max_err_ptr: f32,
    pub source: String,
}

impl Int8Calib {
    pub fn load(path: &Path) -> Result<Int8Calib> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| Error::Io(format!("{}: {e}", path.display())))?;
        let v: serde_json::Value = serde_json::from_str(&text)
            .map_err(|e| Error::Format(format!("{}: {e}", path.display())))?;
        let f = |k: &str| v.get(k).and_then(|x| x.as_f64()).unwrap_or(0.0) as f32;
        let mut smooth = std::collections::HashMap::new();
        if let Some(o) = v.get("smooth").and_then(|x| x.as_object()) {
            for (name, arr) in o {
                let vals: Vec<f32> = arr
                    .as_array()
                    .map(|a| a.iter().map(|x| x.as_f64().unwrap_or(1.0) as f32).collect())
                    .unwrap_or_default();
                smooth.insert(name.clone(), vals);
            }
        }
        Ok(Int8Calib {
            alpha: f("alpha"),
            smooth,
            tau_op: f("tau_op"),
            tau_ptr: f("tau_ptr"),
            max_err_op: f("max_err_op"),
            max_err_ptr: f("max_err_ptr"),
            source: v
                .get("source")
                .and_then(|x| x.as_str())
                .unwrap_or("")
                .to_string(),
        })
    }

    pub fn to_json(&self) -> serde_json::Value {
        let mut sm = serde_json::Map::new();
        let mut names: Vec<&String> = self.smooth.keys().collect();
        names.sort();
        for n in names {
            sm.insert(n.clone(), serde_json::json!(self.smooth[n]));
        }
        serde_json::json!({"format": "xerj-infer-int8-calib-v1", "alpha": self.alpha, "tau_op": self.tau_op, "tau_ptr": self.tau_ptr,
                           "max_err_op": self.max_err_op, "max_err_ptr": self.max_err_ptr, "source": self.source, "smooth": sm})
    }
}

/// load-time options
#[derive(Clone)]
pub struct LoadOptions {
    pub precision: Precision,
    pub kernels: Kernels,
    /// int8: also keep the f32 weights of the quantized linears (needed by the guard fallback)
    pub keep_f32: bool,
    /// int8 smoothing / guard calibration; None = none
    pub calib: Option<Int8Calib>,
    /// split the backbone and capacity-block linears into this many output-channel shards (intra-question parallelism with
    /// a `Pool` of that many threads; results are bit-identical). 1 = unsharded (default)
    pub shards: usize,
}

impl LoadOptions {
    pub fn new(precision: Precision, kernels: Kernels) -> Self {
        LoadOptions {
            precision,
            kernels,
            keep_f32: false,
            calib: None,
            shards: 1,
        }
    }
}

pub struct Layer {
    /// None for layer 0 (identity)
    pub attn_norm: Option<Vec<f32>>,
    /// fused q | k | v projection [3d, d]
    pub qkv: Linear,
    pub wo: Linear,
    pub mlp_norm: Vec<f32>,
    /// [2·inter, d]: input rows then gate rows
    pub wi: Linear,
    pub wo_mlp: Linear,
    pub sliding: bool,
}

/// one MultiheadAttention of the conditioner (query from the hidden state, keys/values from memory or register tokens)
pub struct CrossAttn {
    pub norm_w: Vec<f32>,
    pub norm_b: Vec<f32>,
    pub q: Linear,
    /// K and V projections [2d, d] with bias, f32 (applied to the few memory / register tokens)
    pub kv: Linear,
    pub out: Linear,
    /// tanh(gate)
    pub gate: f32,
}

pub struct Conditioner {
    pub mem: CrossAttn,
    pub reg: CrossAttn,
    /// memory projection [d, memory_width] + bias (f32)
    pub mem_proj: Linear,
    /// register projection [d, register_width] + bias (f32)
    pub reg_proj: Linear,
    pub null_memory: Vec<f32>,
    pub null_register: Vec<f32>,
}

pub struct Capacity {
    pub norm_w: Vec<f32>,
    pub norm_b: Vec<f32>,
    pub down: Linear,
    pub up: Linear,
    pub width: usize,
}

/// input embedding table, a zero-copy view of the memory-mapped safetensors file (only rows that are used become resident)
pub struct Embedding {
    st: Arc<SafeTensors>,
    offset: usize,
    pub vocab: usize,
    pub dim: usize,
}

impl Embedding {
    #[inline]
    pub fn row(&self, id: u32) -> &[f32] {
        let id = (id as usize).min(self.vocab - 1);
        let map = self.st.map();
        let start = self.offset + id * self.dim * 4;
        let bytes = &map[start..start + self.dim * 4];
        debug_assert_eq!(bytes.as_ptr() as usize % 4, 0);
        // SAFETY: alignment checked at load (offset % 4 == 0, map is page aligned), f32 has no invalid bit patterns.
        unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, self.dim) }
    }
}

pub struct RopeTable {
    pub half: usize,
    /// [positions][half]
    pub cos: Vec<f32>,
    pub sin: Vec<f32>,
    pub inv_freq: Vec<f32>,
}

impl RopeTable {
    pub fn new(inv_freq: Vec<f32>, positions: usize) -> Self {
        let half = inv_freq.len();
        let mut cos = vec![0f32; positions * half];
        let mut sin = vec![0f32; positions * half];
        for p in 0..positions {
            for i in 0..half {
                // torch: freqs = inv_freq(f32) @ position(f32) in f32, then cos/sin in f32
                let a = inv_freq[i] * p as f32;
                cos[p * half + i] = (a as f64).cos() as f32;
                sin[p * half + i] = (a as f64).sin() as f32;
            }
        }
        RopeTable {
            half,
            cos,
            sin,
            inv_freq,
        }
    }

    pub fn positions(&self) -> usize {
        self.cos.len() / self.half.max(1)
    }

    /// table rows for a position inside the table
    #[inline]
    pub fn row(&self, p: usize) -> Option<(&[f32], &[f32])> {
        if p < self.positions() {
            Some((
                &self.cos[p * self.half..(p + 1) * self.half],
                &self.sin[p * self.half..(p + 1) * self.half],
            ))
        } else {
            None
        }
    }

    /// cos/sin for one position (computed on the fly beyond the table)
    #[inline]
    pub fn at(&self, p: usize, cos: &mut [f32], sin: &mut [f32]) {
        if p < self.positions() {
            cos.copy_from_slice(&self.cos[p * self.half..(p + 1) * self.half]);
            sin.copy_from_slice(&self.sin[p * self.half..(p + 1) * self.half]);
        } else {
            for i in 0..self.half {
                let a = self.inv_freq[i] * p as f32;
                cos[i] = (a as f64).cos() as f32;
                sin[i] = (a as f64).sin() as f32;
            }
        }
    }
}

pub struct Model {
    pub cfg: ModelConfig,
    pub precision: Precision,
    pub kernels: Kernels,
    pub emb: Embedding,
    pub emb_norm: Vec<f32>,
    pub final_norm: Vec<f32>,
    pub layers: Vec<Layer>,
    pub cond: Option<Conditioner>,
    pub cap: Option<Capacity>,
    /// op head [n_ops, d] + bias (f32)
    pub op_head: Linear,
    /// pointer query [d, d], no bias (f32)
    pub pointer_query: Linear,
    /// external executor NULL key
    pub null_key: Option<Vec<f32>>,
    pub rope_global: RopeTable,
    pub rope_local: RopeTable,
    pub load_ms: f64,
    pub weights_path: std::path::PathBuf,
    /// int8 calibration in use (smoothing + guard thresholds)
    pub calib: Option<Int8Calib>,
    /// linear names by id
    pub linear_names: Vec<String>,
    pub keep_f32: bool,
    /// int8 path: self-attention with BFMMLA on bf16 copies of Q / K / P / V (aarch64 FEAT_BF16; XERJ_BF16_ATTN=0 disables)
    pub bf16_attention: bool,
}

const RT: &str = "core.backbone.model.";

fn bf16_attention_available(hd: usize) -> bool {
    if std::env::var("XERJ_BF16_ATTN").is_ok_and(|v| v == "0") || hd != 64 {
        return false;
    }
    #[cfg(target_arch = "aarch64")]
    {
        std::arch::is_aarch64_feature_detected!("bf16")
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        false
    }
}

impl Model {
    /// load an export directory (config.json + model.safetensors) with the given precision and kernel set
    pub fn load(dir: &Path, precision: Precision, kernels: Kernels) -> Result<Model> {
        Self::load_with(dir, &LoadOptions::new(precision, kernels))
    }

    pub fn load_with(dir: &Path, opt: &LoadOptions) -> Result<Model> {
        let (precision, kernels) = (opt.precision, opt.kernels);
        let t0 = std::time::Instant::now();
        let cfg = ModelConfig::from_file(&dir.join("config.json"))?;
        let path = dir.join("model.safetensors");
        let st = Arc::new(SafeTensors::open(&path)?);
        let d = cfg.hidden;
        let inter = cfg.intermediate;
        let int8 = precision == Precision::Int8;
        let w8 = precision == Precision::W8;
        if w8 && cfg.arm != Arm::ExternalExecutor {
            return Err(Error::Unsupported(format!(
                "precision w8 is defined for the external executor only, not {}",
                cfg.arm.name()
            )));
        }
        // diagnostics: XERJ_INT8_KEEP_F32 = comma list of components kept in f32 under int8 (qkv, wo, wi, wo_mlp, cond_q,
        // cond_out, cap_down, cap_up, layerN)
        let keep: Vec<String> = std::env::var("XERJ_INT8_KEEP_F32")
            .ok()
            .map(|v| v.split(',').map(|x| x.trim().to_string()).collect())
            .unwrap_or_default();
        let q8 = |name: &str, layer: Option<usize>| -> bool {
            int8 && !keep
                .iter()
                .any(|k| k == name || layer.is_some_and(|l| *k == format!("layer{l}")))
        };
        let names = std::cell::RefCell::new(Vec::<String>::new());
        let calib = opt.calib.as_ref();
        // a linear: int8 (smoothed when the calibration has a vector for it) when quantizable under int8, else f32
        let lin = |name: String,
                   kind: &str,
                   layer: Option<usize>,
                   w: &[f32],
                   n: usize,
                   k: usize,
                   bias: Option<&[f32]>|
         -> Linear {
            let mut nm = names.borrow_mut();
            let id = nm.len();
            nm.push(name.clone());
            let mode = if w8 {
                LinMode::W8
            } else if !kind.is_empty() && q8(kind, layer) {
                LinMode::I8 {
                    smooth: calib
                        .and_then(|c| c.smooth.get(&name))
                        .filter(|v| v.len() == k)
                        .map(|v| v.as_slice()),
                    keep_f32: opt.keep_f32,
                }
            } else {
                LinMode::F32
            };
            let parts = if matches!(kind, "qkv" | "wo" | "wi" | "wo_mlp" | "cap_down" | "cap_up") {
                opt.shards
            } else {
                1
            };
            Linear::new_sharded(&name, id, w, n, k, bias, mode, parts)
        };
        let emb_name = format!("{RT}embeddings.tok_embeddings.weight");
        let einfo = st.info(&emb_name)?;
        if einfo.shape != [cfg.vocab, d] {
            return Err(Error::Format(format!("embedding shape {:?}", einfo.shape)));
        }
        let eoff = st.data_offset(&emb_name)?;
        if (st.map().as_ptr() as usize + eoff) % 4 != 0 {
            return Err(Error::Format(
                "embedding table is not 4-byte aligned in the file".into(),
            ));
        }
        let emb = Embedding {
            st: st.clone(),
            offset: eoff,
            vocab: cfg.vocab,
            dim: d,
        };
        let emb_norm = st.f32_vec(&format!("{RT}embeddings.norm.weight"), &[d])?;
        let final_norm = st.f32_vec(&format!("{RT}final_norm.weight"), &[d])?;
        let mut layers = Vec::with_capacity(cfg.layers);
        for l in 0..cfg.layers {
            let p = format!("{RT}layers.{l}.");
            let attn_norm = if l == 0 {
                None
            } else {
                Some(st.f32_vec(&format!("{p}attn_norm.weight"), &[d])?)
            };
            let mut qkv = st.f32_vec(&format!("{p}attn.q_proj.weight"), &[d, d])?;
            qkv.extend(st.f32_vec(&format!("{p}attn.k_proj.weight"), &[d, d])?);
            qkv.extend(st.f32_vec(&format!("{p}attn.v_proj.weight"), &[d, d])?);
            let wo = st.f32_vec(&format!("{p}attn.Wo.weight"), &[d, d])?;
            let wi = st.f32_vec(&format!("{p}mlp.Wi.weight"), &[2 * inter, d])?;
            let wo_mlp = st.f32_vec(&format!("{p}mlp.Wo.weight"), &[d, inter])?;
            layers.push(Layer {
                attn_norm,
                qkv: lin(
                    format!("layers.{l}.qkv"),
                    "qkv",
                    Some(l),
                    &qkv,
                    3 * d,
                    d,
                    None,
                ),
                wo: lin(format!("layers.{l}.wo"), "wo", Some(l), &wo, d, d, None),
                mlp_norm: st.f32_vec(&format!("{p}mlp_norm.weight"), &[d])?,
                wi: lin(
                    format!("layers.{l}.wi"),
                    "wi",
                    Some(l),
                    &wi,
                    2 * inter,
                    d,
                    None,
                ),
                wo_mlp: lin(
                    format!("layers.{l}.wo_mlp"),
                    "wo_mlp",
                    Some(l),
                    &wo_mlp,
                    d,
                    inter,
                    None,
                ),
                sliding: cfg.sliding[l],
            });
        }
        let cond = if cfg.arm == Arm::Rrt {
            let c = "core.conditioners.0.";
            let cross = |name: &str, norm: &str, gate: &str, short: &str| -> Result<CrossAttn> {
                let w = st.f32_vec(&format!("{c}{name}.in_proj_weight"), &[3 * d, d])?;
                let b = st.f32_vec(&format!("{c}{name}.in_proj_bias"), &[3 * d])?;
                let ow = st.f32_vec(&format!("{c}{name}.out_proj.weight"), &[d, d])?;
                let ob = st.f32_vec(&format!("{c}{name}.out_proj.bias"), &[d])?;
                let g = st.scalar(&format!("{c}{gate}"))?;
                Ok(CrossAttn {
                    norm_w: st.f32_vec(&format!("{c}{norm}.weight"), &[d])?,
                    norm_b: st.f32_vec(&format!("{c}{norm}.bias"), &[d])?,
                    q: lin(
                        format!("cond.{short}.q"),
                        "cond_q",
                        None,
                        &w[..d * d],
                        d,
                        d,
                        Some(&b[..d]),
                    ),
                    kv: lin(
                        format!("cond.{short}.kv"),
                        "",
                        None,
                        &w[d * d..],
                        2 * d,
                        d,
                        Some(&b[d..]),
                    ),
                    out: lin(
                        format!("cond.{short}.out"),
                        "cond_out",
                        None,
                        &ow,
                        d,
                        d,
                        Some(&ob),
                    ),
                    gate: (g as f64).tanh() as f32,
                })
            };
            let mw = cfg.memory_width;
            let rw = cfg.register_width;
            Some(Conditioner {
                mem: cross("memory_attention", "memory_norm", "memory_gate", "mem")?,
                reg: cross(
                    "register_attention",
                    "register_norm",
                    "register_gate",
                    "reg",
                )?,
                mem_proj: lin(
                    "cond.mem_proj".into(),
                    "",
                    None,
                    &st.f32_vec("core.references.memory_projection.weight", &[d, mw])?,
                    d,
                    mw,
                    Some(&st.f32_vec("core.references.memory_projection.bias", &[d])?),
                ),
                reg_proj: lin(
                    "cond.reg_proj".into(),
                    "",
                    None,
                    &st.f32_vec("core.references.register_projection.weight", &[d, rw])?,
                    d,
                    rw,
                    Some(&st.f32_vec("core.references.register_projection.bias", &[d])?),
                ),
                null_memory: st.f32_vec("core.references.null_memory", &[1, 1, d])?,
                null_register: st.f32_vec("core.references.null_register", &[1, 1, d])?,
            })
        } else {
            None
        };
        let cap = match cfg.capacity_bottleneck {
            Some(bn) => Some(Capacity {
                norm_w: st.f32_vec("block.norm.weight", &[d])?,
                norm_b: st.f32_vec("block.norm.bias", &[d])?,
                down: lin(
                    "cap.down".into(),
                    "cap_down",
                    None,
                    &st.f32_vec("block.down.weight", &[bn, d])?,
                    bn,
                    d,
                    Some(&st.f32_vec("block.down.bias", &[bn])?),
                ),
                up: lin(
                    "cap.up".into(),
                    "cap_up",
                    None,
                    &st.f32_vec("block.up.weight", &[d, bn])?,
                    d,
                    bn,
                    Some(&st.f32_vec("block.up.bias", &[d])?),
                ),
                width: bn,
            }),
            None => None,
        };
        let n_ops = cfg.n_ops;
        let op_head = lin(
            "op_head".into(),
            "",
            None,
            &st.f32_vec("core.operation_head.weight", &[n_ops, d])?,
            n_ops,
            d,
            Some(&st.f32_vec("core.operation_head.bias", &[n_ops])?),
        );
        let pointer_query = lin(
            "pointer_query".into(),
            "",
            None,
            &st.f32_vec("core.pointer_query.weight", &[d, d])?,
            d,
            d,
            None,
        );
        let null_key = if cfg.arm == Arm::ExternalExecutor {
            Some(st.f32_vec("null_key", &[d])?)
        } else {
            None
        };
        let half = cfg.head_dim / 2;
        let rope_global = RopeTable::new(st.f32_vec("rope.inv_freq_global", &[half])?, 2048);
        let rope_local = RopeTable::new(st.f32_vec("rope.inv_freq_local", &[half])?, 2048);
        // every tensor except the embedding table was copied into run-time buffers: drop the file pages the copies touched
        st.release_pages();
        let bf16_attention = precision == Precision::Int8 && bf16_attention_available(cfg.head_dim);
        let load_ms = t0.elapsed().as_secs_f64() * 1e3;
        Ok(Model {
            cfg,
            precision,
            kernels,
            emb,
            emb_norm,
            final_norm,
            layers,
            cond,
            cap,
            op_head,
            pointer_query,
            null_key,
            rope_global,
            rope_local,
            load_ms,
            weights_path: path,
            calib: opt.calib.clone(),
            linear_names: names.into_inner(),
            keep_f32: opt.keep_f32,
            bf16_attention,
        })
    }

    /// bytes of the run-time weights held in memory (packed projections, norms, tables; the memory-mapped embedding table is
    /// reported separately because it is paged in lazily)
    pub fn resident_weight_bytes(&self) -> usize {
        let mut b = 0;
        for l in &self.layers {
            b += l.qkv.bytes()
                + l.wo.bytes()
                + l.wi.bytes()
                + l.wo_mlp.bytes()
                + 4 * (l.mlp_norm.len() + l.attn_norm.as_ref().map_or(0, |v| v.len()));
        }
        if let Some(c) = &self.cond {
            for x in [&c.mem, &c.reg] {
                b += x.q.bytes() + x.kv.bytes() + x.out.bytes() + 8 * x.norm_w.len();
            }
            b += c.mem_proj.bytes() + c.reg_proj.bytes();
        }
        if let Some(c) = &self.cap {
            b += c.down.bytes() + c.up.bytes();
        }
        b += self.op_head.bytes() + self.pointer_query.bytes();
        b += 4
            * (self.rope_global.cos.len()
                + self.rope_global.sin.len()
                + self.rope_local.cos.len()
                + self.rope_local.sin.len());
        b
    }

    /// every linear, in id order
    pub fn linears(&self) -> Vec<&Linear> {
        let mut v: Vec<&Linear> = Vec::new();
        for l in &self.layers {
            v.extend([&l.qkv, &l.wo, &l.wi, &l.wo_mlp]);
        }
        if let Some(c) = &self.cond {
            v.extend([
                &c.mem.q,
                &c.mem.kv,
                &c.mem.out,
                &c.reg.q,
                &c.reg.kv,
                &c.reg.out,
                &c.mem_proj,
                &c.reg_proj,
            ]);
        }
        if let Some(c) = &self.cap {
            v.extend([&c.down, &c.up]);
        }
        v.extend([&self.op_head, &self.pointer_query]);
        v.sort_by_key(|l| l.id);
        v
    }

    /// is a linear of this name quantized under int8 (a smoothing / calibration target)
    pub fn quantizable_name_pub(name: &str) -> bool {
        Linear::quantizable_name(name)
    }

    pub fn embedding_bytes(&self) -> usize {
        self.emb.vocab * self.emb.dim * 4
    }
}

/// per-layer K/V cache: keys row-major [capacity, d] and transposed [d, capacity + 16] (the aarch64 attention kernel reads
/// 16 consecutive keys of one dimension per load), values row-major [capacity, d]; post-RoPE keys
#[derive(Default)]
pub struct LayerKv {
    pub k: kernels::AVec<f32>,
    pub kt: kernels::AVec<f32>,
    pub v: kernels::AVec<f32>,
    pub rows: usize,
    /// int8 path (aarch64 FEAT_BF16): bf16 copies for the BFMMLA attention — keys in pair layout
    /// [head][rows/2 + 16][hd/4][8], values in transposed pair layout [head][rows/4 + 8][hd/2][8]
    pub kb: kernels::AVec<u16>,
    pub vb: kernels::AVec<u16>,
    pub bf16: bool,
    pub heads: usize,
    pub hd: usize,
    /// x86 round-2 layout (AVX2 + FMA attention, head_dim 64; see [`blocked_kv_layout`]): post-RoPE keys in 16-key blocks
    /// per head, `kblk[h * kblk_head_stride() + (p / 16) * 1024 + c * 16 + p % 16]` = key p, dimension c of head h (each
    /// block a contiguous, 64-byte aligned 4 KB run), values in `v` (see `vhead`); `k` and `kt` are not kept
    pub blocked: bool,
    pub kblk: kernels::AVec<f32>,
    /// blocked layout only: values head-major, `v[h * vhead_stride() + p * hd + c]` = value p, dimension c of head h (one
    /// head's value rows contiguous, 256 bytes apart for head_dim 64), instead of row-major `v[p * heads * hd + h * hd +
    /// c]`. With rows 2 KB apart the P·V loop's value rows map to a few L1 sets and are evicted between the tiles and
    /// dimension chunks that reuse them. Chosen with the layout ([`blocked_values_head_major`]); the same floats either way.
    pub vhead: bool,
}

/// the blocked layout keeps its values head-major (`LayerKv::vhead`): the default; `XERJ_X86_R1=vrows` keeps the
/// row-major values (A/B checks)
pub fn blocked_values_head_major() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        kernels::x86ops::r2_vhead()
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        true
    }
}

/// the blocked x86 key layout and the round-2 attention kernel are used for this head size: x86-64 with AVX2 + FMA
/// (`XERJ_AVX2_OPS` not 0), head_dim 64, and `XERJ_X86_R1` not keeping `attn` (the round-1 transposed-key layout)
pub fn blocked_kv_layout(hd: usize) -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        hd == 64 && kernels::x86ops::r2_attn()
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = hd;
        false
    }
}

impl LayerKv {
    /// row stride of the transposed keys
    #[inline]
    pub fn kts(&self) -> usize {
        self.rows + 16
    }

    /// floats per head of the blocked keys (`kblk`): rows / 16 blocks of 64 x 16
    #[inline]
    pub fn kblk_head_stride(&self) -> usize {
        self.rows / 16 * 1024
    }

    /// floats per head of the head-major values (`vhead`): rows x hd
    #[inline]
    pub fn vhead_stride(&self) -> usize {
        self.rows * self.hd
    }

    /// choose the key layout for `heads` x `hd` before rows are stored: blocked when [`blocked_kv_layout`] holds (values
    /// head-major when [`blocked_values_head_major`] holds too). The choice is fixed for the process (CPU features and
    /// environment), so a cache never changes layout once filled.
    pub fn set_layout(&mut self, heads: usize, hd: usize) {
        let want = blocked_kv_layout(hd);
        let vhead = want && blocked_values_head_major();
        if want != self.blocked || vhead != self.vhead {
            assert!(
                self.rows == 0,
                "the key layout of an allocated cache cannot change"
            );
            self.blocked = want;
            self.vhead = vhead;
        }
        if want {
            self.heads = heads;
            self.hd = hd;
        }
    }

    #[inline]
    pub fn pair_cap(&self) -> usize {
        self.rows / 2 + 16
    }

    #[inline]
    pub fn chunk_cap(&self) -> usize {
        self.rows / 4 + 8
    }

    /// enable the bf16 copies (int8 path on FEAT_BF16 CPUs)
    pub fn enable_bf16(&mut self, heads: usize, hd: usize) {
        if !self.bf16 {
            self.bf16 = true;
            self.heads = heads;
            self.hd = hd;
            let (rows, d) = (self.rows, heads * hd);
            if rows > 0 {
                self.alloc_bf16();
                for p in 0..rows {
                    self.put_bf16_from_f32(p, d);
                }
            }
        }
    }

    fn alloc_bf16(&mut self) {
        let (h, hd) = (self.heads, self.hd);
        self.kb = kernels::AVec::zeroed(h * self.pair_cap() * (hd / 4) * 8);
        self.vb = kernels::AVec::zeroed(h * self.chunk_cap() * (hd / 2) * 8);
    }

    fn put_bf16_from_f32(&mut self, p: usize, d: usize) {
        let key: Vec<f32> = self.k.as_slice()[p * d..(p + 1) * d].to_vec();
        let val: Vec<f32> = self.v.as_slice()[p * d..(p + 1) * d].to_vec();
        self.put_bf16(p, &key, &val);
    }

    /// bf16 copies of rows p0..p1 from the f32 row-major caches (keys: 4-value runs; values: one 8-lane entry per
    /// (4-position chunk, dimension pair), read-modify-write at partial chunks)
    pub fn put_bf16_rows(&mut self, p0: usize, p1: usize) {
        if p1 <= p0 {
            return;
        }
        let (heads, hd) = (self.heads, self.hd);
        let d = heads * hd;
        let (pc, cc) = (self.pair_cap(), self.chunk_cap());
        #[cfg(target_arch = "aarch64")]
        if hd == 64 {
            // SAFETY: rows p0..p1 exist in the f32 caches; bf16 caches hold pc pairs / cc chunks per head.
            unsafe {
                kernels::neon::put_bf16_rows64(
                    self.k.as_slice().as_ptr(),
                    self.v.as_slice().as_ptr(),
                    d,
                    heads,
                    p0,
                    p1,
                    self.kb.as_mut_slice().as_mut_ptr(),
                    pc,
                    self.vb.as_mut_slice().as_mut_ptr(),
                    cc,
                );
            }
            return;
        }
        let (nc, ndp) = (hd / 4, hd / 2);
        let k = self.k.as_slice();
        let v = self.v.as_slice();
        let kb = self.kb.as_mut_slice();
        for p in p0..p1 {
            let key = &k[p * d..(p + 1) * d];
            for h in 0..heads {
                let base = (h * pc + p / 2) * nc * 8 + (p % 2) * 4;
                for c in 0..nc {
                    let src = &key[h * hd + 4 * c..h * hd + 4 * c + 4];
                    let dst = &mut kb[base + c * 8..base + c * 8 + 4];
                    for i in 0..4 {
                        dst[i] = kernels::bf16(src[i]);
                    }
                }
            }
        }
        let vb = self.vb.as_mut_slice();
        for j in p0 / 4..p1.div_ceil(4) {
            for h in 0..heads {
                let base = (h * cc + j) * ndp * 8;
                for q in 0..ndp {
                    let e = &mut vb[base + q * 8..base + q * 8 + 8];
                    for l in 0..4 {
                        let p = 4 * j + l;
                        if p >= p0 && p < p1 {
                            e[l] = kernels::bf16(v[p * d + h * hd + 2 * q]);
                            e[4 + l] = kernels::bf16(v[p * d + h * hd + 2 * q + 1]);
                        }
                    }
                }
            }
        }
    }

    /// bf16 copies of the key / value of position p
    #[inline]
    pub fn put_bf16(&mut self, p: usize, key: &[f32], val: &[f32]) {
        let (heads, hd) = (self.heads, self.hd);
        let (pc, cc) = (self.pair_cap(), self.chunk_cap());
        let (nc, ndp) = (hd / 4, hd / 2);
        let kb = self.kb.as_mut_slice();
        for h in 0..heads {
            let base = (h * pc + p / 2) * nc * 8 + (p % 2) * 4;
            for c in 0..nc {
                for i in 0..4 {
                    kb[base + c * 8 + i] = kernels::bf16(key[h * hd + 4 * c + i]);
                }
            }
        }
        let vb = self.vb.as_mut_slice();
        let (j, l) = (p / 4, p % 4);
        for h in 0..heads {
            let base = (h * cc + j) * ndp * 8;
            for q in 0..ndp {
                vb[base + q * 8 + l] = kernels::bf16(val[h * hd + 2 * q]);
                vb[base + q * 8 + 4 + l] = kernels::bf16(val[h * hd + 2 * q + 1]);
            }
        }
    }

    pub fn ensure(&mut self, rows: usize, d: usize) {
        if self.blocked && self.rows < rows {
            assert_eq!(
                self.heads * self.hd,
                d,
                "blocked layout: set_layout before ensure"
            );
            let cap = rows.next_power_of_two().max(64);
            let mut v = kernels::AVec::zeroed(cap * d);
            let mut kblk = kernels::AVec::zeroed(cap * d);
            if self.rows > 0 {
                if self.vhead {
                    let (ovs, nvs) = (self.vhead_stride(), cap * self.hd);
                    for h in 0..self.heads {
                        v.as_mut_slice()[h * nvs..h * nvs + ovs]
                            .copy_from_slice(&self.v.as_slice()[h * ovs..(h + 1) * ovs]);
                    }
                } else {
                    let keep = self.rows * d;
                    v.as_mut_slice()[..keep].copy_from_slice(&self.v.as_slice()[..keep]);
                }
                let (ohs, nhs) = (self.kblk_head_stride(), cap / 16 * 1024);
                for h in 0..self.heads {
                    kblk.as_mut_slice()[h * nhs..h * nhs + ohs]
                        .copy_from_slice(&self.kblk.as_slice()[h * ohs..(h + 1) * ohs]);
                }
            }
            self.v = v;
            self.kblk = kblk;
            self.rows = cap;
            return;
        }
        if self.rows < rows {
            let cap = rows.next_power_of_two().max(64);
            let old_rows = self.rows;
            let mut k = kernels::AVec::zeroed(cap * d);
            let mut v = kernels::AVec::zeroed(cap * d);
            let mut kt = kernels::AVec::zeroed(d * (cap + 16));
            let keep = self.rows * d;
            if keep > 0 {
                k.as_mut_slice()[..keep].copy_from_slice(&self.k.as_slice()[..keep]);
                v.as_mut_slice()[..keep].copy_from_slice(&self.v.as_slice()[..keep]);
                let (old, ots) = (self.kt.as_slice(), self.rows + 16);
                let new = kt.as_mut_slice();
                for i in 0..d {
                    new[i * (cap + 16)..i * (cap + 16) + self.rows]
                        .copy_from_slice(&old[i * ots..i * ots + self.rows]);
                }
            }
            self.k = k;
            self.v = v;
            self.kt = kt;
            self.rows = cap;
            if self.bf16 {
                self.alloc_bf16();
                for p in 0..old_rows {
                    self.put_bf16_from_f32(p, d);
                }
            }
        }
    }

    /// store the post-RoPE key and the value of position p (row-major copies; call `transpose_keys` afterwards). Blocked
    /// layout: the value row and the key's 64-dimension columns of every head
    #[inline]
    pub fn put(&mut self, p: usize, key: &[f32], val: &[f32]) {
        let d = key.len();
        if self.blocked {
            self.put_v(p, val);
            let (hs, hd) = (self.kblk_head_stride(), self.hd);
            let kb = self.kblk.as_mut_slice();
            for h in 0..self.heads {
                let base = h * hs + (p / 16) * 1024 + p % 16;
                for c in 0..hd {
                    kb[base + c * 16] = key[h * hd + c];
                }
            }
            return;
        }
        self.k.as_mut_slice()[p * d..(p + 1) * d].copy_from_slice(key);
        self.v.as_mut_slice()[p * d..(p + 1) * d].copy_from_slice(val);
    }

    /// store the value row of position p (head-major layout: each head's part into that head's rows)
    #[inline]
    pub fn put_v(&mut self, p: usize, val: &[f32]) {
        let d = val.len();
        if self.vhead {
            let (vs, hd) = (self.vhead_stride(), self.hd);
            assert!(p < self.rows && d == self.heads * hd);
            let v = self.v.as_mut_slice();
            for h in 0..self.heads {
                v[h * vs + p * hd..h * vs + (p + 1) * hd]
                    .copy_from_slice(&val[h * hd..(h + 1) * hd]);
            }
            return;
        }
        self.v.as_mut_slice()[p * d..(p + 1) * d].copy_from_slice(val);
    }

    /// value of position p, dimension i (either layout; tests and diagnostics)
    pub fn value_at(&self, p: usize, i: usize, d: usize) -> f32 {
        if self.vhead {
            let (h, c) = (i / self.hd, i % self.hd);
            self.v.as_slice()[h * self.vhead_stride() + p * self.hd + c]
        } else {
            self.v.as_slice()[p * d + i]
        }
    }

    /// blocked layout: the keys of positions p0..p1 from rows of a matrix (key of position p at
    /// `src[(p - p0) * stride..][..d]`), x86 8 x 8 transposes (pure data movement, the same floats as `put`)
    pub fn put_keys_blocked(&mut self, src: &[f32], stride: usize, p0: usize, p1: usize) {
        assert!(self.blocked && p1 <= self.rows);
        let d = self.heads * self.hd;
        assert!(p1 <= p0 || src.len() >= (p1 - p0 - 1) * stride + d);
        #[cfg(target_arch = "x86_64")]
        {
            assert!(self.hd == 64 && self.kblk.len() >= self.heads * self.kblk_head_stride());
            // SAFETY: AVX2 + FMA (blocked_kv_layout); source rows and the destination blocks are in bounds (asserted)
            unsafe {
                kernels::x86ops::store_keys_blocked(
                    src.as_ptr(),
                    stride,
                    self.heads,
                    self.kblk.as_mut_slice().as_mut_ptr(),
                    self.kblk_head_stride(),
                    p0,
                    p1,
                )
            };
        }
        #[cfg(not(target_arch = "x86_64"))]
        for p in p0..p1 {
            let key: Vec<f32> = src[(p - p0) * stride..(p - p0) * stride + d].to_vec();
            let (hs, hd) = (self.kblk_head_stride(), self.hd);
            let kb = self.kblk.as_mut_slice();
            for h in 0..self.heads {
                for c in 0..hd {
                    kb[h * hs + (p / 16) * 1024 + c * 16 + p % 16] = key[h * hd + c];
                }
            }
        }
    }

    /// key of position p, dimension i (either layout; tests and diagnostics)
    pub fn key_at(&self, p: usize, i: usize, d: usize) -> f32 {
        if self.blocked {
            let (h, c) = (i / self.hd, i % self.hd);
            self.kblk.as_slice()[h * self.kblk_head_stride() + (p / 16) * 1024 + c * 16 + p % 16]
        } else {
            self.k.as_slice()[p * d + i]
        }
    }

    /// refresh the transposed keys of rows p0..p1 from the row-major keys (dimension-major loop: contiguous writes,
    /// the row-major reads stay in L1)
    pub fn transpose_keys(&mut self, p0: usize, p1: usize, d: usize) {
        if self.blocked {
            // the blocked layout stores its keys in `put` / `put_keys_blocked`
            return;
        }
        let kts = self.kts();
        #[cfg(target_arch = "x86_64")]
        if kernels::x86ops::enabled() && p1 > p0 {
            assert!(p1 <= self.rows && self.k.len() >= self.rows * d && self.kt.len() >= d * kts);
            // SAFETY: AVX2 + FMA checked at run time; rows p0..p1 of k ([rows, d]) and columns p0..p1 of kt ([d, kts])
            // are in bounds (asserted). Pure data movement.
            unsafe {
                kernels::x86ops::transpose_keys(
                    self.k.as_slice().as_ptr(),
                    d,
                    self.kt.as_mut_slice().as_mut_ptr(),
                    kts,
                    p0,
                    p1,
                )
            };
            return;
        }
        let k = self.k.as_slice();
        let kt = self.kt.as_mut_slice();
        for i in 0..d {
            let dst = &mut kt[i * kts + p0..i * kts + p1];
            for (j, x) in dst.iter_mut().enumerate() {
                *x = k[(p0 + j) * d + i];
            }
        }
    }
}

/// reusable activation buffers of one session (grown on demand, never shrunk)
#[derive(Default)]
pub struct Buffers {
    pub xn: Vec<f32>,
    pub qkv: Vec<f32>,
    pub att: Vec<f32>,
    pub u: Vec<f32>,
    pub g: Vec<f32>,
    pub capb: Vec<f32>,
    pub qc: Vec<f32>,
    pub xc: Vec<f32>,
    pub scores: Vec<f32>,
    pub pos: Vec<usize>,
    pub cosb: Vec<f32>,
    pub sinb: Vec<f32>,
    pub gemm: GemmScratch,
    pub bfq: Vec<u16>,
    pub bfp: Vec<u16>,
}

#[inline]
pub fn grow(v: &mut Vec<f32>, n: usize) {
    if v.len() < n {
        v.resize(n, 0.0);
    }
}

/// memory / register tokens already projected to K and V for the conditioner
#[derive(Default)]
pub struct CrossKv {
    /// [S, 2d]: K | V per token
    pub kv: Vec<f32>,
    /// indices of the visible tokens
    pub valid: Vec<usize>,
    pub tokens: usize,
    /// K | V rows of the visible tokens only, [valid.len(), 2d] (what the attention reads)
    pub compact: Vec<f32>,
    /// 0..valid.len()
    pub ident: Vec<usize>,
    /// visible keys transposed [d, n + 16]
    pub kt: Vec<f32>,
}

impl CrossKv {
    /// rebuild the compact visible-token rows after `kv` / `valid` changed
    pub fn compact(&mut self, d: usize) {
        let n = self.valid.len();
        self.compact.resize(n * 2 * d, 0.0);
        for (i, &t) in self.valid.iter().enumerate() {
            self.compact[i * 2 * d..(i + 1) * 2 * d]
                .copy_from_slice(&self.kv[t * 2 * d..(t + 1) * 2 * d]);
        }
        self.ident.clear();
        self.ident.extend(0..n);
        let kts = n + 16;
        self.kt.clear();
        self.kt.resize(d * kts, 0.0);
        for (i, &t) in self.valid.iter().enumerate() {
            for c in 0..d {
                self.kt[c * kts + i] = self.kv[t * 2 * d + c];
            }
        }
    }
}

impl Model {
    #[inline]
    fn d(&self) -> usize {
        self.cfg.hidden
    }

    /// x[n, d] = LayerNorm(E[ids])
    pub fn embed(&self, ids: &[u32], x: &mut [f32]) {
        let _t = crate::prof::start(crate::prof::Cat::Embed);
        let d = self.d();
        for (r, &id) in ids.iter().enumerate() {
            kernels::ops::layer_norm_row(
                self.emb.row(id),
                &self.emb_norm,
                None,
                self.cfg.norm_eps,
                &mut x[r * d..(r + 1) * d],
            );
        }
    }

    /// One decoder layer over `n` new rows at absolute positions pos0..pos0+n (`x`: [n, d], their input hidden states).
    /// Their K/V are written into `kv` (rows pos0..); queries attend to cache rows 0..=pos.
    /// `need`: when Some, only these local rows' outputs are computed (attention, Wo, MLP) and written compactly to
    /// `out_c` ([need.len(), d]); `x` is left with the layer inputs. When None, `x` is updated in place to the layer output.
    pub fn layer_forward(
        &self,
        l: usize,
        x: &mut [f32],
        n: usize,
        pos0: usize,
        kv: &mut LayerKv,
        need: Option<&[usize]>,
        out_c: &mut Vec<f32>,
        b: &mut Buffers,
    ) {
        let d = self.d();
        let cfg = &self.cfg;
        let layer = &self.layers[l];
        let kern = &self.kernels;
        let hd = cfg.head_dim;
        let heads = cfg.heads;
        kv.set_layout(heads, hd);
        kv.ensure(pos0 + n, d);
        // pre-norm
        grow(&mut b.xn, n * d);
        grow(&mut b.qkv, n * 3 * d);
        match &layer.attn_norm {
            Some(w) => layer.qkv.forward_ln(
                kern,
                &x[..n * d],
                n,
                w,
                None,
                cfg.norm_eps,
                &mut b.xn,
                &mut b.qkv,
                3 * d,
                Epi::STORE,
                &mut b.gemm,
            ),
            None => layer.qkv.forward(
                kern,
                &x[..n * d],
                n,
                &mut b.qkv,
                3 * d,
                Epi::STORE,
                &mut b.gemm,
            ),
        }
        // RoPE on q and k, then scatter k / v into the cache
        let rope = if layer.sliding {
            &self.rope_local
        } else {
            &self.rope_global
        };
        let half = hd / 2;
        grow(&mut b.cosb, half);
        grow(&mut b.sinb, half);
        if self.bf16_attention && !kv.bf16 {
            kv.enable_bf16(heads, hd);
        }
        let trope = crate::prof::start(crate::prof::Cat::Rope);
        #[cfg(target_arch = "x86_64")]
        let pooled = match b.gemm.pool.clone() {
            Some(pool) if pool.threads() > 1 && n >= 32 && kernels::x86ops::enabled() => {
                rope_kv_pool(&pool, &mut b.qkv, n, pos0, d, heads, hd, rope, kv);
                true
            }
            _ => false,
        };
        #[cfg(not(target_arch = "x86_64"))]
        let pooled = false;
        #[cfg(target_arch = "x86_64")]
        if kv.blocked && !pooled {
            // round 2: RoPE in place, the value rows, then the keys straight from the q | k | v rows into their blocks
            for r in 0..n {
                let p = pos0 + r;
                let (cs, sn): (&[f32], &[f32]) = match rope.row(p) {
                    Some(t) => t,
                    None => {
                        rope.at(p, &mut b.cosb[..half], &mut b.sinb[..half]);
                        (&b.cosb[..half], &b.sinb[..half])
                    }
                };
                let row = &mut b.qkv[r * 3 * d..(r + 1) * 3 * d];
                // SAFETY: AVX2 + FMA (blocked layout, head_dim 64); bit-identical to ops::rope_head
                unsafe { kernels::x86ops::rope_qk64(row, d, heads, cs, sn) };
                kv.put_v(p, &row[2 * d..3 * d]);
            }
            kv.put_keys_blocked(&b.qkv[d..], 3 * d, pos0, pos0 + n);
        }
        let serial_rows = if pooled || kv.blocked { 0 } else { n };
        for r in 0..serial_rows {
            let p = pos0 + r;
            let (cs, sn): (&[f32], &[f32]) = match rope.row(p) {
                Some(t) => t,
                None => {
                    rope.at(p, &mut b.cosb[..half], &mut b.sinb[..half]);
                    (&b.cosb[..half], &b.sinb[..half])
                }
            };
            let row = &mut b.qkv[r * 3 * d..(r + 1) * 3 * d];
            #[cfg(target_arch = "aarch64")]
            if hd == 64 {
                for h in 0..heads {
                    // SAFETY: 64-lane head slices inside the row; cos/sin hold 32 values
                    unsafe {
                        kernels::neon::rope_head64(
                            row[h * hd..].as_mut_ptr(),
                            cs.as_ptr(),
                            sn.as_ptr(),
                        );
                        kernels::neon::rope_head64(
                            row[d + h * hd..].as_mut_ptr(),
                            cs.as_ptr(),
                            sn.as_ptr(),
                        );
                    }
                }
                kv.put(p, &row[d..2 * d], &row[2 * d..3 * d]);
                continue;
            }
            #[cfg(target_arch = "x86_64")]
            if kernels::x86ops::enabled() {
                // SAFETY: AVX2 + FMA checked at run time (bit-identical to ops::rope_head)
                unsafe { kernels::x86ops::rope_qk(row, d, heads, hd, cs, sn) };
                kv.put(p, &row[d..2 * d], &row[2 * d..3 * d]);
                continue;
            }
            for h in 0..heads {
                kernels::ops::rope_head(&mut row[h * hd..(h + 1) * hd], cs, sn);
                kernels::ops::rope_head(&mut row[d + h * hd..d + (h + 1) * hd], cs, sn);
            }
            kv.put(p, &row[d..2 * d], &row[2 * d..3 * d]);
        }
        if !pooled && !kv.blocked {
            kv.transpose_keys(pos0, pos0 + n, d);
        }
        if kv.bf16 {
            kv.put_bf16_rows(pos0, pos0 + n);
        }
        drop(trope);
        // query rows
        let window = if layer.sliding {
            Some(cfg.window)
        } else {
            None
        };
        b.pos.clear();
        match need {
            Some(rows) => b.pos.extend(rows.iter().map(|&r| pos0 + r)),
            None => b.pos.extend(pos0..pos0 + n),
        }
        let nq = b.pos.len();
        if nq == 0 {
            return;
        }
        grow(&mut b.qc, nq * d);
        for (i, &p) in b.pos.iter().enumerate() {
            let r = p - pos0;
            b.qc[i * d..(i + 1) * d].copy_from_slice(&b.qkv[r * 3 * d..r * 3 * d + d]);
        }
        grow(&mut b.att, nq * d);
        let pos = std::mem::take(&mut b.pos);
        let tatt = crate::prof::start(crate::prof::Cat::SelfAttn);
        #[cfg(target_arch = "aarch64")]
        let use_bf16 = kv.bf16 && !b.gemm.force_f32;
        #[cfg(not(target_arch = "aarch64"))]
        let use_bf16 = false;
        if use_bf16 {
            #[cfg(target_arch = "aarch64")]
            {
                let pooled = match b.gemm.pool.clone() {
                    Some(pool) => kernels::ops::self_attention_bf16_pool(
                        &b.qc[..nq * d],
                        &pos,
                        kv,
                        heads,
                        hd,
                        window,
                        &pool,
                        &mut b.gemm.tls_scores,
                        &mut b.att,
                    ),
                    None => false,
                };
                if !pooled {
                    kernels::ops::self_attention_bf16(
                        &b.qc[..nq * d],
                        &pos,
                        kv,
                        heads,
                        hd,
                        window,
                        &mut b.scores,
                        &mut b.bfq,
                        &mut b.bfp,
                        &mut b.att,
                    );
                }
            }
        } else {
            let pooled = match b.gemm.pool.clone() {
                Some(pool) => kernels::ops::self_attention_pool(
                    &b.qc[..nq * d],
                    &pos,
                    kv,
                    heads,
                    hd,
                    window,
                    &pool,
                    &mut b.gemm.tls_scores,
                    &mut b.att,
                ),
                None => false,
            };
            if !pooled {
                kernels::ops::self_attention(
                    &b.qc[..nq * d],
                    &pos,
                    kv,
                    heads,
                    hd,
                    window,
                    &mut b.scores,
                    &mut b.att,
                );
            }
        }
        drop(tatt);
        b.pos = pos;
        // residual stream rows: in place (all rows) or compacted (needed rows)
        let h: &mut [f32] = match need {
            Some(rows) => {
                grow(out_c, nq * d);
                for (i, &r) in rows.iter().enumerate() {
                    out_c[i * d..(i + 1) * d].copy_from_slice(&x[r * d..(r + 1) * d]);
                }
                &mut out_c[..nq * d]
            }
            None => &mut x[..n * d],
        };
        layer
            .wo
            .forward(kern, &b.att[..nq * d], nq, h, d, Epi::ADD, &mut b.gemm);
        // MLP
        let inter = cfg.intermediate;
        grow(&mut b.u, nq * 2 * inter);
        layer.wi.forward_ln(
            kern,
            h,
            nq,
            &layer.mlp_norm,
            None,
            cfg.norm_eps,
            &mut b.xn,
            &mut b.u,
            2 * inter,
            Epi::STORE,
            &mut b.gemm,
        );
        grow(&mut b.g, nq * inter);
        {
            let _t = crate::prof::start(crate::prof::Cat::Act);
            kernels::ops::geglu_pool(&b.u, nq, inter, &mut b.g, b.gemm.pool.as_deref());
        }
        layer
            .wo_mlp
            .forward(kern, &b.g[..nq * inter], nq, h, d, Epi::ADD, &mut b.gemm);
    }

    /// project memory or register tokens to the conditioner's K | V
    pub fn cross_kv(
        &self,
        which: &CrossAttn,
        tokens: &[f32],
        s: usize,
        valid: &[bool],
        out: &mut CrossKv,
        b: &mut Buffers,
    ) {
        let d = self.d();
        out.kv.resize(s * 2 * d, 0.0);
        which.kv.forward(
            &self.kernels,
            tokens,
            s,
            &mut out.kv,
            2 * d,
            Epi::STORE,
            &mut b.gemm,
        );
        out.valid.clear();
        out.valid.extend((0..s).filter(|&i| valid[i]));
        out.tokens = s;
        out.compact(d);
    }

    /// conditioner hook on `n` rows: memory attention, then register attention when registers are present
    pub fn condition(
        &self,
        x: &mut [f32],
        n: usize,
        mem: &CrossKv,
        regs: Option<&CrossKv>,
        b: &mut Buffers,
    ) {
        let c = self.cond.as_ref().expect("rrt conditioner");
        let d = self.d();
        let kern = &self.kernels;
        let one = |ca: &CrossAttn, kvs: &CrossKv, x: &mut [f32], b: &mut Buffers| {
            grow(&mut b.qc, n * d);
            ca.q.forward_ln(
                kern,
                &x[..n * d],
                n,
                &ca.norm_w,
                Some(&ca.norm_b),
                self.cfg.cond_eps,
                &mut b.xn,
                &mut b.qc,
                d,
                Epi::STORE,
                &mut b.gemm,
            );
            grow(&mut b.att, n * d);
            let _t = crate::prof::start(crate::prof::Cat::CrossAttn);
            kernels::ops::cross_attention(
                &b.qc[..n * d],
                n,
                kvs,
                self.cfg.heads,
                self.cfg.head_dim,
                &mut b.scores,
                &mut b.att,
            );
            drop(_t);
            ca.out.forward(
                kern,
                &b.att[..n * d],
                n,
                x,
                d,
                Epi::gated_add(ca.gate),
                &mut b.gemm,
            );
        };
        one(&c.mem, mem, x, b);
        if let Some(r) = regs {
            one(&c.reg, r, x, b);
        }
    }

    /// capacity block on `n` rows in place: x += up(gelu(down(LN(x))))
    pub fn capacity(&self, x: &mut [f32], n: usize, b: &mut Buffers) {
        let c = self.cap.as_ref().expect("capacity block");
        let d = self.d();
        grow(&mut b.capb, n * c.width);
        c.down.forward_ln(
            &self.kernels,
            &x[..n * d],
            n,
            &c.norm_w,
            Some(&c.norm_b),
            self.cfg.capacity_eps,
            &mut b.xn,
            &mut b.capb,
            c.width,
            Epi::STORE,
            &mut b.gemm,
        );
        {
            let _t = crate::prof::start(crate::prof::Cat::Act);
            kernels::ops::gelu_inplace_pool(&mut b.capb[..n * c.width], b.gemm.pool.as_deref());
        }
        c.up.forward(
            &self.kernels,
            &b.capb[..n * c.width],
            n,
            x,
            d,
            Epi::ADD,
            &mut b.gemm,
        );
    }

    /// final LayerNorm of n rows in place
    pub fn final_norm(&self, x: &mut [f32], n: usize, b: &mut Buffers) {
        let d = self.d();
        grow(&mut b.xn, n * d);
        kernels::ops::layer_norm_pool(
            x,
            n,
            d,
            &self.final_norm,
            None,
            self.cfg.norm_eps,
            &mut b.xn,
            b.gemm.pool.as_deref(),
        );
        x[..n * d].copy_from_slice(&b.xn[..n * d]);
    }

    /// op logits [n_ops] and pointer query [d] of the decision state
    pub fn heads(&self, last: &[f32], op: &mut [f32], q: &mut [f32], b: &mut Buffers) {
        let _t = crate::prof::start(crate::prof::Cat::Heads);
        self.op_head.forward(
            &self.kernels,
            last,
            1,
            op,
            self.cfg.n_ops,
            Epi::STORE,
            &mut b.gemm,
        );
        self.pointer_query.forward(
            &self.kernels,
            last,
            1,
            q,
            self.cfg.hidden,
            Epi::STORE,
            &mut b.gemm,
        );
    }
}

/// RoPE of q / k, the K / V cache rows and the transposed keys of rows pos0..pos0 + n with the rows spread over a pool
/// (x86 AVX2 path): every row is rotated by `x86ops::rope_qk` and copied exactly as in the single-threaded loop of
/// `layer_forward`, the transposed keys are the same copies, so the caches are bit-identical.
#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
fn rope_kv_pool(
    pool: &crate::pool::Pool,
    qkv: &mut [f32],
    n: usize,
    pos0: usize,
    d: usize,
    heads: usize,
    hd: usize,
    rope: &RopeTable,
    kv: &mut LayerKv,
) {
    use crate::pool::SendPtr;
    assert!(qkv.len() >= n * 3 * d && pos0 + n <= kv.rows && heads * hd == d && hd <= 128);
    if kv.blocked {
        return rope_kv_pool_blocked(pool, qkv, n, pos0, d, heads, hd, rope, kv);
    }
    assert!(kv.k.len() >= kv.rows * d && kv.v.len() >= kv.rows * d && kv.kt.len() >= d * kv.kts());
    let t = pool.threads();
    let half = hd / 2;
    let qp = SendPtr(qkv.as_mut_ptr());
    let kp = SendPtr(kv.k.as_mut_slice().as_mut_ptr());
    let vp = SendPtr(kv.v.as_mut_slice().as_mut_ptr());
    let chunk = n.div_ceil(t);
    pool.run(t, &|i| {
        let mut cb = [0f32; 64];
        let mut sb = [0f32; 64];
        for r in (i * chunk).min(n)..((i + 1) * chunk).min(n) {
            let p = pos0 + r;
            let (cs, sn): (&[f32], &[f32]) = match rope.row(p) {
                Some(t) => t,
                None => {
                    rope.at(p, &mut cb[..half], &mut sb[..half]);
                    (&cb[..half], &sb[..half])
                }
            };
            // SAFETY: row r of qkv and cache row p belong to this task only (disjoint row ranges; bounds asserted);
            // AVX2 + FMA checked by the caller
            unsafe {
                let row = std::slice::from_raw_parts_mut(qp.get().add(r * 3 * d), 3 * d);
                kernels::x86ops::rope_qk(row, d, heads, hd, cs, sn);
                std::ptr::copy_nonoverlapping(row.as_ptr().add(d), kp.get().add(p * d), d);
                std::ptr::copy_nonoverlapping(row.as_ptr().add(2 * d), vp.get().add(p * d), d);
            }
        }
    });
    let kts = kv.kts();
    let ktp = SendPtr(kv.kt.as_mut_slice().as_mut_ptr());
    let kc = SendPtr(kv.k.as_mut_slice().as_mut_ptr());
    let pchunk = n.div_ceil(t).div_ceil(8) * 8;
    pool.run(t, &|i| {
        let (a, b) = (pos0 + (i * pchunk).min(n), pos0 + ((i + 1) * pchunk).min(n));
        if a < b {
            // SAFETY: columns a..b of the transposed keys belong to this task only; k is only read here
            unsafe { kernels::x86ops::transpose_keys(kc.get(), d, ktp.get(), kts, a, b) };
        }
    });
}

/// `rope_kv_pool` for the blocked x86 key layout (round 2): rows spread over the pool for RoPE and the value rows, then
/// 8-aligned key ranges for the blocked keys. Every row and key is processed by the same code as the serial path, so the
/// caches are bit-identical.
#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
fn rope_kv_pool_blocked(
    pool: &crate::pool::Pool,
    qkv: &mut [f32],
    n: usize,
    pos0: usize,
    d: usize,
    heads: usize,
    hd: usize,
    rope: &RopeTable,
    kv: &mut LayerKv,
) {
    use crate::pool::SendPtr;
    assert!(kv.blocked && hd == 64 && heads * hd == d && kv.heads == heads);
    assert!(kv.v.len() >= kv.rows * d && kv.kblk.len() >= heads * kv.kblk_head_stride());
    let t = pool.threads();
    let half = hd / 2;
    let qp = SendPtr(qkv.as_mut_ptr());
    let vp = SendPtr(kv.v.as_mut_slice().as_mut_ptr());
    let (vhead, vhs) = (kv.vhead, kv.vhead_stride());
    let chunk = n.div_ceil(t);
    pool.run(t, &|i| {
        let mut cb = [0f32; 64];
        let mut sb = [0f32; 64];
        for r in (i * chunk).min(n)..((i + 1) * chunk).min(n) {
            let p = pos0 + r;
            let (cs, sn): (&[f32], &[f32]) = match rope.row(p) {
                Some(t) => t,
                None => {
                    rope.at(p, &mut cb[..half], &mut sb[..half]);
                    (&cb[..half], &sb[..half])
                }
            };
            // SAFETY: row r of qkv and value row p (of every head) belong to this task only (disjoint row ranges; bounds
            // asserted: p < rows, so h * vhs + (p + 1) * hd <= heads * rows * hd = rows * d); AVX2 + FMA checked by the
            // caller
            unsafe {
                let row = std::slice::from_raw_parts_mut(qp.get().add(r * 3 * d), 3 * d);
                kernels::x86ops::rope_qk64(row, d, heads, cs, sn);
                if vhead {
                    for h in 0..heads {
                        std::ptr::copy_nonoverlapping(
                            row.as_ptr().add(2 * d + h * hd),
                            vp.get().add(h * vhs + p * hd),
                            hd,
                        );
                    }
                } else {
                    std::ptr::copy_nonoverlapping(row.as_ptr().add(2 * d), vp.get().add(p * d), d);
                }
            }
        }
    });
    let hs = kv.kblk_head_stride();
    let kbp = SendPtr(kv.kblk.as_mut_slice().as_mut_ptr());
    let qc = SendPtr(qkv.as_mut_ptr());
    // key ranges cut at multiples of 8 (an aligned 8-key group is written by one task)
    let pchunk = n.div_ceil(t);
    let cut = |i: usize| -> usize {
        if i == 0 {
            pos0
        } else if i >= t {
            pos0 + n
        } else {
            ((pos0 + i * pchunk) / 8 * 8).clamp(pos0, pos0 + n)
        }
    };
    pool.run(t, &|i| {
        let (a, b) = (cut(i), cut(i + 1));
        if a < b {
            // SAFETY: keys a..b belong to this task only (their 8-key groups are not shared); source rows a - pos0..b - pos0
            // of qkv are only read here; AVX2 + FMA checked by the caller
            unsafe {
                kernels::x86ops::store_keys_blocked(
                    qc.get().add((a - pos0) * 3 * d + d),
                    3 * d,
                    heads,
                    kbp.get(),
                    hs,
                    a,
                    b,
                )
            };
        }
    });
}
