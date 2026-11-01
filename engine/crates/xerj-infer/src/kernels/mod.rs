//! GEMM kernels with runtime ISA dispatch.
//!
//! Every linear layer computes `y = x · Wᵀ + b` for `x: [M, K]` (row-major activations) and a PyTorch weight `W: [N, K]`.
//! Two weight formats, each packed once at load time:
//!
//! * **f32** — panels of `NR = 16` output channels: `data[p][k][j] = W[16p + j][k]` (zero padded). Micro-kernels: NEON 6×16
//!   (by-element FMLA over an `A` block packed as `[k][6]`), AVX2+FMA 6×16, AVX-512F 12×16, scalar.
//! * **int8** (per-output-channel symmetric scales, activations quantized per row at run time, symmetric, `[-127, 127]`) —
//!   one layout for every ISA: blocks of 8 channels × 8 k, `data[nb][kb][c][kk] = q(W[8nb + c][8kb + kk])` (64 bytes). That is
//!   exactly the `SMMLA` right operand (two channels × 8 k per 16-byte register), the `SDOT` operand (row-duplicated `A`, pairwise
//!   lane sums), the AVX2 `VPMADDUBSW` operand (sign trick, exact for `[-127, 127]`) and the AVX-512 `VPDPBUSD` operand
//!   (`A + 128` offset corrected with per-channel sums). `A` is packed as row pairs: `[M/2][kb][16]` = row `2i` k0..8 | row
//!   `2i+1` k0..8 — the `SMMLA` left operand. Micro-kernels: NEON i8mm 8×8 (`SMMLA`), NEON dotprod 4×8 (`SDOT`), AVX-512 VNNI
//!   8×8, AVX2 4×8, scalar.
//!
//! The epilogue fuses bias, an output scale `alpha` and an optional accumulate into `y` (residual and gated residual adds).

#[cfg(all(target_arch = "aarch64", xerj_kleidiai))]
pub mod kleidi;
#[cfg(target_arch = "aarch64")]
pub mod neon;
pub mod ops;
pub mod scalar;
#[cfg(target_arch = "x86_64")]
pub mod x86;
#[cfg(target_arch = "x86_64")]
pub mod x86ops;

/// f32 -> bf16 bits, round to nearest even
#[inline(always)]
pub fn bf16(x: f32) -> u16 {
    let b = x.to_bits();
    if x.is_nan() {
        return ((b >> 16) | 0x40) as u16;
    }
    ((b + 0x7fff + ((b >> 16) & 1)) >> 16) as u16
}

/// f32 panel width (output channels per packed panel)
pub const NR: usize = 16;
/// int8 block: 8 channels × 8 k
pub const QN: usize = 8;
pub const QK: usize = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum F32Isa {
    Scalar,
    #[cfg(target_arch = "aarch64")]
    Neon,
    #[cfg(target_arch = "x86_64")]
    Avx2,
    #[cfg(target_arch = "x86_64")]
    Avx512,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum I8Isa {
    Scalar,
    #[cfg(target_arch = "aarch64")]
    NeonDot,
    #[cfg(target_arch = "aarch64")]
    NeonI8mm,
    #[cfg(target_arch = "x86_64")]
    Avx2,
    #[cfg(target_arch = "x86_64")]
    Avx512Vnni,
}

impl F32Isa {
    pub fn name(self) -> &'static str {
        match self {
            F32Isa::Scalar => "scalar",
            #[cfg(target_arch = "aarch64")]
            F32Isa::Neon => "neon",
            #[cfg(target_arch = "x86_64")]
            F32Isa::Avx2 => "avx2+fma",
            #[cfg(target_arch = "x86_64")]
            F32Isa::Avx512 => "avx512f",
        }
    }
    /// rows per micro-kernel tile (A packing width)
    pub fn mr(self) -> usize {
        match self {
            #[cfg(target_arch = "x86_64")]
            F32Isa::Avx512 => 12,
            _ => 6,
        }
    }
}

impl I8Isa {
    pub fn name(self) -> &'static str {
        match self {
            I8Isa::Scalar => "scalar",
            #[cfg(target_arch = "aarch64")]
            I8Isa::NeonDot => "neon-dotprod(sdot)",
            #[cfg(target_arch = "aarch64")]
            I8Isa::NeonI8mm => "neon-i8mm(smmla)",
            #[cfg(target_arch = "x86_64")]
            I8Isa::Avx2 => "avx2(vpmaddubsw)",
            #[cfg(target_arch = "x86_64")]
            I8Isa::Avx512Vnni => "avx512-vnni(vpdpbusd)",
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct CpuFeatures {
    pub neon: bool,
    pub dotprod: bool,
    pub i8mm: bool,
    pub sve2: bool,
    pub fp16: bool,
    pub avx2: bool,
    pub fma: bool,
    pub avx512f: bool,
    pub avx512bw: bool,
    pub avx512vnni: bool,
}

impl CpuFeatures {
    pub fn detect() -> Self {
        #[allow(unused_mut)]
        let mut f = CpuFeatures::default();
        #[cfg(target_arch = "aarch64")]
        {
            f.neon = std::arch::is_aarch64_feature_detected!("neon");
            f.dotprod = std::arch::is_aarch64_feature_detected!("dotprod");
            f.i8mm = std::arch::is_aarch64_feature_detected!("i8mm");
            f.sve2 = std::arch::is_aarch64_feature_detected!("sve2");
            f.fp16 = std::arch::is_aarch64_feature_detected!("fp16");
        }
        #[cfg(target_arch = "x86_64")]
        {
            f.avx2 = is_x86_feature_detected!("avx2");
            f.fma = is_x86_feature_detected!("fma");
            f.avx512f = is_x86_feature_detected!("avx512f");
            f.avx512bw = is_x86_feature_detected!("avx512bw");
            f.avx512vnni = is_x86_feature_detected!("avx512vnni");
        }
        f
    }

    pub fn best_f32(&self) -> F32Isa {
        #[cfg(target_arch = "aarch64")]
        if self.neon {
            return F32Isa::Neon;
        }
        #[cfg(target_arch = "x86_64")]
        {
            if self.avx512f {
                return F32Isa::Avx512;
            }
            if self.avx2 && self.fma {
                return F32Isa::Avx2;
            }
        }
        F32Isa::Scalar
    }

    pub fn best_i8(&self) -> I8Isa {
        #[cfg(target_arch = "aarch64")]
        {
            if self.i8mm {
                return I8Isa::NeonI8mm;
            }
            if self.dotprod {
                return I8Isa::NeonDot;
            }
        }
        #[cfg(target_arch = "x86_64")]
        {
            if self.avx512f && self.avx512bw && self.avx512vnni {
                return I8Isa::Avx512Vnni;
            }
            if self.avx2 {
                return I8Isa::Avx2;
            }
        }
        I8Isa::Scalar
    }

    pub fn supports_f32(&self, isa: F32Isa) -> bool {
        match isa {
            F32Isa::Scalar => true,
            #[cfg(target_arch = "aarch64")]
            F32Isa::Neon => self.neon,
            #[cfg(target_arch = "x86_64")]
            F32Isa::Avx2 => self.avx2 && self.fma,
            #[cfg(target_arch = "x86_64")]
            F32Isa::Avx512 => self.avx512f,
        }
    }

    pub fn supports_i8(&self, isa: I8Isa) -> bool {
        match isa {
            I8Isa::Scalar => true,
            #[cfg(target_arch = "aarch64")]
            I8Isa::NeonDot => self.dotprod,
            #[cfg(target_arch = "aarch64")]
            I8Isa::NeonI8mm => self.i8mm,
            #[cfg(target_arch = "x86_64")]
            I8Isa::Avx2 => self.avx2,
            #[cfg(target_arch = "x86_64")]
            I8Isa::Avx512Vnni => self.avx512f && self.avx512bw && self.avx512vnni,
        }
    }
}

/// 64-byte aligned buffer (cache-line aligned SIMD loads, no reallocation in the hot loop once grown)
pub struct AVec<T: Copy + Default> {
    buf: Vec<T>,
    off: usize,
    len: usize,
}

impl<T: Copy + Default> AVec<T> {
    pub fn new() -> Self {
        AVec {
            buf: Vec::new(),
            off: 0,
            len: 0,
        }
    }
    pub fn zeroed(len: usize) -> Self {
        let mut v = Self::new();
        v.resize(len);
        v.as_mut_slice().fill(T::default());
        v
    }
    /// set the logical length; grows (and realigns) only when the capacity is exceeded. Contents are unspecified after a grow.
    pub fn resize(&mut self, len: usize) {
        let sz = std::mem::size_of::<T>().max(1);
        let pad = 64 / sz + 1;
        if self.buf.len() < len + pad || self.off + len > self.buf.len() {
            let cap = (len + pad).max(self.buf.len() * 3 / 2);
            self.buf = vec![T::default(); cap];
            let addr = self.buf.as_ptr() as usize;
            self.off = ((64 - addr % 64) % 64) / sz;
        }
        self.len = len;
    }
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        &self.buf[self.off..self.off + self.len]
    }
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.buf[self.off..self.off + self.len]
    }
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub fn bytes(&self) -> usize {
        self.buf.len() * std::mem::size_of::<T>()
    }
}

impl<T: Copy + Default> Default for AVec<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Copy + Default> Clone for AVec<T> {
    fn clone(&self) -> Self {
        let mut v = AVec::new();
        v.resize(self.len);
        v.as_mut_slice().copy_from_slice(self.as_slice());
        v
    }
}

/// f32 weights packed in panels of NR output channels
#[derive(Clone, Default)]
pub struct PackedF32 {
    pub n: usize,
    pub k: usize,
    pub data: AVec<f32>,
    pub bias: Option<Vec<f32>>,
}

impl PackedF32 {
    /// `w`: PyTorch Linear weight [n, k] row-major
    pub fn pack(w: &[f32], n: usize, k: usize, bias: Option<&[f32]>) -> Self {
        assert_eq!(w.len(), n * k);
        let panels = n.div_ceil(NR);
        let mut data = AVec::zeroed(panels * k * NR);
        let d = data.as_mut_slice();
        for p in 0..panels {
            for kk in 0..k {
                for j in 0..NR {
                    let c = p * NR + j;
                    if c < n {
                        d[(p * k + kk) * NR + j] = w[c * k + kk];
                    }
                }
            }
        }
        PackedF32 {
            n,
            k,
            data,
            bias: bias.map(|b| b.to_vec()),
        }
    }

    pub fn panels(&self) -> usize {
        self.n.div_ceil(NR)
    }

    pub fn bytes(&self) -> usize {
        self.data.len() * 4 + self.bias.as_ref().map_or(0, |b| b.len() * 4)
    }
}

/// weight-only INT8 (`Precision::W8`): one row of a PyTorch weight [n, k] as tools/int8_fakequant.py rounds it. Returns the
/// row's scale and writes q: scale = max(max|w|, 1e-12) / 127, q = round(w / scale) (halves to even, as torch.round), clipped to
/// [-127, 127], all in f32.
pub fn w8_quantize_row(w: &[f32], q: &mut [i8]) -> f32 {
    debug_assert_eq!(w.len(), q.len());
    let amax = w.iter().fold(0f32, |a, v| a.max(v.abs()));
    let scale = amax.max(1e-12) / 127.0;
    for (d, v) in q.iter_mut().zip(w) {
        *d = (v / scale).round_ties_even().clamp(-127.0, 127.0) as i8;
    }
    scale
}

/// weight-only INT8 weights: the layout of [`PackedF32`] (panels of NR output channels) with i8 values and one f32 scale per
/// output channel. The compute stays f32: every call dequantizes into a [`PackedF32`] buffer and runs the f32 GEMM on it, so
/// activations are never quantized and the result is the f32 path's on the rounded weights.
#[derive(Clone, Default)]
pub struct PackedW8 {
    pub n: usize,
    pub k: usize,
    pub data: AVec<i8>,
    /// [panels * NR], zero on the padding channels
    pub scales: Vec<f32>,
    pub bias: Option<Vec<f32>>,
}

impl PackedW8 {
    /// `w`: PyTorch Linear weight [n, k] row-major
    pub fn quantize(w: &[f32], n: usize, k: usize, bias: Option<&[f32]>) -> Self {
        assert_eq!(w.len(), n * k);
        let panels = n.div_ceil(NR);
        let mut data = AVec::zeroed(panels * k * NR);
        let mut scales = vec![0f32; panels * NR];
        let mut row = vec![0i8; k];
        let d = data.as_mut_slice();
        for c in 0..n {
            scales[c] = w8_quantize_row(&w[c * k..(c + 1) * k], &mut row);
            let (p, j) = (c / NR, c % NR);
            for (kk, q) in row.iter().enumerate() {
                d[(p * k + kk) * NR + j] = *q;
            }
        }
        PackedW8 {
            n,
            k,
            data,
            scales,
            bias: bias.map(|b| b.to_vec()),
        }
    }

    pub fn panels(&self) -> usize {
        self.n.div_ceil(NR)
    }

    /// `out` = the f32 weights q * scale in the f32 panel layout (no allocation once `out` has grown)
    pub fn dequant_into(&self, out: &mut PackedF32) {
        let (panels, k) = (self.panels(), self.k);
        out.n = self.n;
        out.k = k;
        out.data.resize(panels * k * NR);
        let (src, dst) = (self.data.as_slice(), out.data.as_mut_slice());
        for ((scale, sp), dp) in self
            .scales
            .chunks_exact(NR)
            .zip(src.chunks_exact(k * NR))
            .zip(dst.chunks_exact_mut(k * NR))
        {
            let scale: &[f32; NR] = scale.try_into().unwrap();
            for (s, d) in sp.chunks_exact(NR).zip(dp.chunks_exact_mut(NR)) {
                let s: &[i8; NR] = s.try_into().unwrap();
                let d: &mut [f32; NR] = d.try_into().unwrap();
                for j in 0..NR {
                    d[j] = s[j] as f32 * scale[j];
                }
            }
        }
        match (&self.bias, &mut out.bias) {
            (Some(b), Some(o)) => {
                o.clear();
                o.extend_from_slice(b);
            }
            (Some(b), o) => *o = Some(b.clone()),
            (None, o) => *o = None,
        }
    }

    pub fn bytes(&self) -> usize {
        self.data.len() + self.scales.len() * 4 + self.bias.as_ref().map_or(0, |b| b.len() * 4)
    }
}

/// int8 weights, per-output-channel symmetric scale, 8×8 blocks
#[derive(Clone)]
pub struct PackedI8 {
    pub n: usize,
    pub k: usize,
    /// k padded to a multiple of 8
    pub kp: usize,
    pub data: AVec<i8>,
    /// per channel (padded to a multiple of 8, 0 for padding)
    pub scales: Vec<f32>,
    /// per channel Σ_k q (for the u8 offset trick of VPDPBUSD)
    pub colsum: Vec<i32>,
    pub bias: Option<Vec<f32>>,
}

impl PackedI8 {
    pub fn quantize(w: &[f32], n: usize, k: usize, bias: Option<&[f32]>) -> Self {
        assert_eq!(w.len(), n * k);
        let np = n.div_ceil(QN) * QN;
        let kp = k.div_ceil(QK) * QK;
        let mut q = vec![0i8; np * kp];
        let mut scales = vec![0f32; np];
        let mut colsum = vec![0i32; np];
        for c in 0..n {
            let row = &w[c * k..(c + 1) * k];
            let amax = row.iter().fold(0f32, |m, &x| m.max(x.abs()));
            let s = if amax > 0.0 { amax / 127.0 } else { 1.0 };
            scales[c] = if amax > 0.0 { s } else { 0.0 };
            let inv = 1.0 / s;
            let mut sum = 0i32;
            for kk in 0..k {
                let v = (row[kk] * inv).round().clamp(-127.0, 127.0) as i8;
                q[c * kp + kk] = v;
                sum += v as i32;
            }
            colsum[c] = sum;
        }
        let mut data = AVec::zeroed(np * kp);
        let d = data.as_mut_slice();
        let (nbs, kbs) = (np / QN, kp / QK);
        for nb in 0..nbs {
            for kb in 0..kbs {
                for c in 0..QN {
                    for kk in 0..QK {
                        d[((nb * kbs + kb) * QN + c) * QK + kk] =
                            q[(nb * QN + c) * kp + kb * QK + kk];
                    }
                }
            }
        }
        PackedI8 {
            n,
            k,
            kp,
            data,
            scales,
            colsum,
            bias: bias.map(|b| b.to_vec()),
        }
    }

    pub fn bytes(&self) -> usize {
        self.data.len()
            + self.scales.len() * 4
            + self.colsum.len() * 4
            + self.bias.as_ref().map_or(0, |b| b.len() * 4)
    }

    /// int8 weights [n, k] row-major (unpadded)
    pub fn row_major(&self) -> Vec<i8> {
        let mut q = vec![0i8; self.n * self.k];
        let kbs = self.kp / QK;
        let d = self.data.as_slice();
        for c in 0..self.n {
            let (nb, cc) = (c / QN, c % QN);
            for kk in 0..self.k {
                let (kb, k8) = (kk / QK, kk % QK);
                q[c * self.k + kk] = d[((nb * kbs + kb) * QN + cc) * QK + k8];
            }
        }
        q
    }

    /// dequantized weight (for tests): [n, k]
    pub fn dequant(&self) -> Vec<f32> {
        let mut w = vec![0f32; self.n * self.k];
        let kbs = self.kp / QK;
        let d = self.data.as_slice();
        for c in 0..self.n {
            let (nb, cc) = (c / QN, c % QN);
            for kk in 0..self.k {
                let (kb, k8) = (kk / QK, kk % QK);
                w[c * self.k + kk] =
                    d[((nb * kbs + kb) * QN + cc) * QK + k8] as f32 * self.scales[c];
            }
        }
        w
    }
}

/// activations quantized per row (symmetric int8) in the row-pair layout `[M/2][kb][16]`
pub struct QActs {
    pub m: usize,
    pub k: usize,
    pub kp: usize,
    pub data: AVec<i8>,
    pub scales: AVec<f32>,
}

impl QActs {
    pub fn new() -> Self {
        QActs {
            m: 0,
            k: 0,
            kp: 0,
            data: AVec::new(),
            scales: AVec::new(),
        }
    }
    pub fn prepare(&mut self, m: usize, k: usize) {
        self.m = m;
        self.k = k;
        self.kp = k.div_ceil(QK) * QK;
        let mp = m.div_ceil(2) * 2;
        self.data.resize(mp * self.kp);
        self.scales.resize(mp);
    }
    /// byte offset of (row pair p, k block kb)
    #[inline]
    pub fn pair_offset(&self, p: usize, kb: usize) -> usize {
        (p * (self.kp / QK) + kb) * 16
    }
}

impl Default for QActs {
    fn default() -> Self {
        Self::new()
    }
}

/// epilogue: `y = (accumulate ? y : 0) + alpha * (x·Wᵀ + b)`
#[derive(Debug, Clone, Copy)]
pub struct Epi {
    pub alpha: f32,
    pub accumulate: bool,
}

impl Epi {
    pub const STORE: Epi = Epi {
        alpha: 1.0,
        accumulate: false,
    };
    pub const ADD: Epi = Epi {
        alpha: 1.0,
        accumulate: true,
    };
    pub fn gated_add(alpha: f32) -> Epi {
        Epi {
            alpha,
            accumulate: true,
        }
    }
}

/// apply the epilogue to one output row segment (`acc` holds x·Wᵀ for columns c0..c0+acc.len())
#[inline(always)]
pub fn epi_store(y: &mut [f32], acc: &[f32], bias: Option<&[f32]>, epi: Epi) {
    match (bias, epi.accumulate) {
        (Some(b), false) => {
            for i in 0..acc.len() {
                y[i] = epi.alpha * (acc[i] + b[i]);
            }
        }
        (None, false) => {
            for i in 0..acc.len() {
                y[i] = epi.alpha * acc[i];
            }
        }
        (Some(b), true) => {
            for i in 0..acc.len() {
                y[i] += epi.alpha * (acc[i] + b[i]);
            }
        }
        (None, true) => {
            for i in 0..acc.len() {
                y[i] += epi.alpha * acc[i];
            }
        }
    }
}

/// rows per task when `m` rows are split over `t` threads: an even count (row pairs stay together), at least 2
pub fn par_chunk(m: usize, t: usize) -> usize {
    (m.div_ceil(t.max(1)) + 1) & !1
}

/// Runtime-selected kernel set + per-thread scratch for packing activations.
#[derive(Debug, Clone, Copy)]
pub struct Kernels {
    pub f32_isa: F32Isa,
    pub i8_isa: I8Isa,
    /// use the vendored KleidiAI SMMLA kernel for int8 GEMMs with M >= 16 (feature `kleidiai`, FEAT_I8MM)
    pub kleidi: bool,
}

/// is the KleidiAI backend compiled in and usable on this CPU
pub fn kleidi_available() -> bool {
    #[cfg(all(target_arch = "aarch64", xerj_kleidiai))]
    {
        CpuFeatures::detect().i8mm
    }
    #[cfg(not(all(target_arch = "aarch64", xerj_kleidiai)))]
    {
        false
    }
}

/// per-session scratch (no allocation once grown)
#[derive(Default)]
pub struct GemmScratch {
    pub apack: AVec<f32>,
    pub q: QActs,
    /// intra-question parallelism (sharded linears, attention heads); None = single-threaded
    pub pool: Option<std::sync::Arc<crate::pool::Pool>>,
    /// per-participant packing buffers of the parallel f32 GEMM
    pub tls: Vec<std::sync::Mutex<AVec<f32>>>,
    /// weight-only INT8: the dequantized weights of the running linear (or shard)
    pub w8: PackedF32,
    /// weight-only INT8: per-participant dequantized shard of the parallel GEMM
    pub tls_w8: Vec<std::sync::Mutex<PackedF32>>,
    /// per-participant attention scratch (pooled attention)
    pub tls_scores: Vec<std::sync::Mutex<ops::AttnScratch>>,
    /// run every dual-format linear with its f32 weights (int8 guard fallback)
    pub force_f32: bool,
    /// calibration: per linear id, per input channel max |x| (recorded by f32 forwards when Some)
    pub calib: Option<Vec<Vec<f32>>>,
    #[cfg(all(target_arch = "aarch64", xerj_kleidiai))]
    pub kai: kleidi::KaiScratch,
}

impl Kernels {
    pub fn detect() -> Self {
        let f = CpuFeatures::detect();
        Kernels {
            f32_isa: f.best_f32(),
            i8_isa: f.best_i8(),
            kleidi: kleidi_available(),
        }
    }

    pub fn scalar() -> Self {
        Kernels {
            f32_isa: F32Isa::Scalar,
            i8_isa: I8Isa::Scalar,
            kleidi: false,
        }
    }

    /// description of the int8 backend
    pub fn i8_name(&self) -> String {
        if self.kleidi {
            format!("kleidiai-i8mm(M>=16) + {}", self.i8_isa.name())
        } else {
            self.i8_isa.name().to_string()
        }
    }

    /// y[m, ldy] (first w.n columns) = epilogue(x[m, k] · Wᵀ + b)
    pub fn gemm_f32(
        &self,
        x: &[f32],
        m: usize,
        w: &PackedF32,
        y: &mut [f32],
        ldy: usize,
        epi: Epi,
        s: &mut GemmScratch,
    ) {
        debug_assert!(x.len() >= m * w.k && y.len() >= (m.max(1) - 1) * ldy + w.n);
        if m == 0 {
            return;
        }
        let _t = crate::prof::start(crate::prof::Cat::GemmF32);
        match self.f32_isa {
            F32Isa::Scalar => scalar::gemm_f32(x, m, w, y, ldy, epi),
            #[cfg(target_arch = "aarch64")]
            F32Isa::Neon => unsafe { neon::gemm_f32(x, m, w, y, ldy, epi, &mut s.apack) },
            #[cfg(target_arch = "x86_64")]
            F32Isa::Avx2 if x86::r2_gemm() => unsafe {
                x86::gemm_f32_avx2_v2_ptr(x, m, w, y.as_mut_ptr(), ldy, epi, &mut s.apack)
            },
            #[cfg(target_arch = "x86_64")]
            F32Isa::Avx2 => unsafe { x86::gemm_f32_avx2(x, m, w, y, ldy, epi, &mut s.apack) },
            #[cfg(target_arch = "x86_64")]
            F32Isa::Avx512 => unsafe { x86::gemm_f32_avx512(x, m, w, y, ldy, epi, &mut s.apack) },
        }
        let _ = &s;
    }

    /// quantize x[m, k] per row into s.q (row-pair layout); `pre`: per-channel factors applied first (smoothing)
    pub fn quantize(
        &self,
        x: &[f32],
        m: usize,
        k: usize,
        pre: Option<&[f32]>,
        s: &mut GemmScratch,
    ) {
        let _t = crate::prof::start(crate::prof::Cat::Quantize);
        s.q.prepare(m, k);
        match self.i8_isa {
            #[cfg(target_arch = "aarch64")]
            I8Isa::NeonDot | I8Isa::NeonI8mm => unsafe {
                neon::quantize_rows(x, m, k, pre, &mut s.q)
            },
            // SAFETY: AVX2 + FMA checked at run time (x86ops::enabled); bit-identical to the scalar path
            #[cfg(target_arch = "x86_64")]
            I8Isa::Avx2 | I8Isa::Avx512Vnni if x86ops::enabled() => unsafe {
                x86ops::quantize_rows(x, m, k, pre, &mut s.q)
            },
            _ => scalar::quantize_rows(x, m, k, pre, &mut s.q),
        }
    }

    /// `quantize` with the rows split over the pool attached to `s` (x86 AVX2 path; elsewhere, or without a pool, plain
    /// `quantize`). Bit-identical: every row is quantized by the same code, only the rows are spread over the threads.
    pub fn quantize_par(
        &self,
        x: &[f32],
        m: usize,
        k: usize,
        pre: Option<&[f32]>,
        s: &mut GemmScratch,
    ) {
        #[cfg(target_arch = "x86_64")]
        if let Some(pool) = self.par_rows_pool(m, s) {
            let _t = crate::prof::start(crate::prof::Cat::Quantize);
            assert!(x.len() >= m * k && pre.is_none_or(|p| p.len() >= k));
            s.q.prepare(m, k);
            let kbs = s.q.kp / QK;
            let dp = crate::pool::SendPtr(s.q.data.as_mut_slice().as_mut_ptr());
            let sp = crate::pool::SendPtr(s.q.scales.as_mut_slice().as_mut_ptr());
            let chunk = par_chunk(m, pool.threads());
            pool.run(pool.threads(), &|i| {
                let (r0, r1) = (i * chunk, ((i + 1) * chunk).min(m));
                if r0 < r1 {
                    // SAFETY: AVX2 + FMA checked (par_rows_pool); rows r0..r1 (even bounds) write their own pair slots
                    // and scales only (QActs sized by prepare)
                    unsafe {
                        x86ops::quantize_rows_range(x, r0, r1, k, pre, kbs, dp.get(), sp.get())
                    };
                }
            });
            // SAFETY: as above
            unsafe { x86ops::pad_odd_row(m, kbs, dp.get(), sp.get()) };
            return;
        }
        self.quantize(x, m, k, pre, s)
    }

    /// `quantize_ln` with the rows split over the pool attached to `s` (see `quantize_par`)
    #[allow(clippy::too_many_arguments)]
    pub fn quantize_ln_par(
        &self,
        x: &[f32],
        m: usize,
        k: usize,
        w: &[f32],
        b: Option<&[f32]>,
        eps: f32,
        pre: Option<&[f32]>,
        s: &mut GemmScratch,
    ) -> bool {
        #[cfg(target_arch = "x86_64")]
        if k % 8 == 0 {
            if let Some(pool) = self.par_rows_pool(m, s) {
                let _t = crate::prof::start(crate::prof::Cat::Quantize);
                assert!(x.len() >= m * k && w.len() >= k);
                s.q.prepare(m, k);
                let kbs = s.q.kp / QK;
                let dp = crate::pool::SendPtr(s.q.data.as_mut_slice().as_mut_ptr());
                let sp = crate::pool::SendPtr(s.q.scales.as_mut_slice().as_mut_ptr());
                let chunk = par_chunk(m, pool.threads());
                pool.run(pool.threads(), &|i| {
                    let (r0, r1) = (i * chunk, ((i + 1) * chunk).min(m));
                    if r0 < r1 {
                        // SAFETY: as in quantize_par
                        unsafe {
                            x86ops::quantize_rows_ln_range(
                                x,
                                r0,
                                r1,
                                k,
                                w,
                                b,
                                eps,
                                pre,
                                kbs,
                                dp.get(),
                                sp.get(),
                            )
                        };
                    }
                });
                // SAFETY: as above
                unsafe { x86ops::pad_odd_row(m, kbs, dp.get(), sp.get()) };
                return true;
            }
        }
        self.quantize_ln(x, m, k, w, b, eps, pre, s)
    }

    /// the pool for a row-parallel x86 quantization (AVX2 int8 kernels in use, enough rows), if any
    #[cfg(target_arch = "x86_64")]
    fn par_rows_pool(
        &self,
        m: usize,
        s: &GemmScratch,
    ) -> Option<std::sync::Arc<crate::pool::Pool>> {
        s.pool
            .as_ref()
            .filter(|p| {
                p.threads() > 1
                    && m >= 32
                    && matches!(self.i8_isa, I8Isa::Avx2 | I8Isa::Avx512Vnni)
                    && x86ops::enabled()
            })
            .cloned()
    }

    /// fused LayerNorm + quantization into s.q; returns false when no fused kernel exists (caller falls back)
    #[allow(clippy::too_many_arguments)]
    pub fn quantize_ln(
        &self,
        x: &[f32],
        m: usize,
        k: usize,
        w: &[f32],
        b: Option<&[f32]>,
        eps: f32,
        pre: Option<&[f32]>,
        s: &mut GemmScratch,
    ) -> bool {
        match self.i8_isa {
            #[cfg(target_arch = "aarch64")]
            I8Isa::NeonDot | I8Isa::NeonI8mm if k % 8 == 0 => {
                let _t = crate::prof::start(crate::prof::Cat::Quantize);
                s.q.prepare(m, k);
                unsafe { neon::quantize_rows_ln(x, m, k, w, b, eps, pre, &mut s.q) };
                true
            }
            // bit-identical to ops::layer_norm followed by the scalar quantization (the unfused x86 path)
            #[cfg(target_arch = "x86_64")]
            I8Isa::Avx2 | I8Isa::Avx512Vnni if k % 8 == 0 && x86ops::enabled() => {
                let _t = crate::prof::start(crate::prof::Cat::Quantize);
                s.q.prepare(m, k);
                // SAFETY: AVX2 + FMA checked at run time; k % 8 == 0
                unsafe { x86ops::quantize_rows_ln(x, m, k, w, b, eps, pre, &mut s.q) };
                true
            }
            _ => {
                let _ = (x, m, k, w, b, eps, pre, s);
                false
            }
        }
    }

    /// `gemm_f32` with a raw output pointer and an explicit packing buffer, no profiler timer (the parallel shards)
    ///
    /// # Safety
    /// `yp + r * ldy + c` must be writable for r < m, c < w.n, and no other thread may access those elements meanwhile.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gemm_f32_ptr(
        &self,
        x: &[f32],
        m: usize,
        w: &PackedF32,
        yp: *mut f32,
        ldy: usize,
        epi: Epi,
        apack: &mut AVec<f32>,
    ) {
        if m == 0 {
            return;
        }
        match self.f32_isa {
            F32Isa::Scalar => scalar::gemm_f32(
                x,
                m,
                w,
                std::slice::from_raw_parts_mut(yp, (m - 1) * ldy + w.n),
                ldy,
                epi,
            ),
            #[cfg(target_arch = "aarch64")]
            F32Isa::Neon => neon::gemm_f32_ptr(x, m, w, yp, ldy, epi, apack),
            #[cfg(target_arch = "x86_64")]
            F32Isa::Avx2 if x86::r2_gemm() => {
                x86::gemm_f32_avx2_v2_ptr(x, m, w, yp, ldy, epi, apack)
            }
            #[cfg(target_arch = "x86_64")]
            F32Isa::Avx2 => x86::gemm_f32_avx2_ptr(x, m, w, yp, ldy, epi, apack),
            #[cfg(target_arch = "x86_64")]
            F32Isa::Avx512 => x86::gemm_f32_avx512_ptr(x, m, w, yp, ldy, epi, apack),
        }
        let _ = apack;
    }

    /// `gemm_i8` with a raw output pointer, no profiler timer (the parallel shards)
    ///
    /// # Safety
    /// as `gemm_f32_ptr`, with m = q.m
    pub unsafe fn gemm_i8_ptr(&self, w: &PackedI8, yp: *mut f32, ldy: usize, epi: Epi, q: &QActs) {
        if q.m == 0 {
            return;
        }
        match self.i8_isa {
            I8Isa::Scalar => scalar::gemm_i8(
                q,
                w,
                std::slice::from_raw_parts_mut(yp, (q.m - 1) * ldy + w.n),
                ldy,
                epi,
            ),
            #[cfg(target_arch = "aarch64")]
            I8Isa::NeonDot => neon::gemm_i8_sdot_ptr(q, w, yp, ldy, epi),
            #[cfg(target_arch = "aarch64")]
            I8Isa::NeonI8mm => neon::gemm_i8_smmla_ptr(q, w, yp, ldy, epi),
            #[cfg(target_arch = "x86_64")]
            I8Isa::Avx2 if x86::r2_gemm() => x86::gemm_i8_avx2_v2_ptr(q, w, yp, ldy, epi),
            #[cfg(target_arch = "x86_64")]
            I8Isa::Avx2 => x86::gemm_i8_avx2_ptr(q, w, yp, ldy, epi),
            #[cfg(target_arch = "x86_64")]
            I8Isa::Avx512Vnni => x86::gemm_i8_avx512vnni_ptr(q, w, yp, ldy, epi),
        }
    }

    /// y[m, ldy] (first w.n columns) = epilogue(dequant(q · Wᵀ) + b) for activations already in s.q
    pub fn gemm_i8(&self, w: &PackedI8, y: &mut [f32], ldy: usize, epi: Epi, s: &GemmScratch) {
        let q = &s.q;
        debug_assert_eq!(q.kp, w.kp);
        if q.m == 0 {
            return;
        }
        let _t = crate::prof::start(crate::prof::Cat::GemmI8);
        match self.i8_isa {
            I8Isa::Scalar => scalar::gemm_i8(q, w, y, ldy, epi),
            #[cfg(target_arch = "aarch64")]
            I8Isa::NeonDot => unsafe { neon::gemm_i8_sdot(q, w, y, ldy, epi) },
            #[cfg(target_arch = "aarch64")]
            I8Isa::NeonI8mm => unsafe { neon::gemm_i8_smmla(q, w, y, ldy, epi) },
            #[cfg(target_arch = "x86_64")]
            I8Isa::Avx2 if x86::r2_gemm() => unsafe {
                x86::gemm_i8_avx2_v2_ptr(q, w, y.as_mut_ptr(), ldy, epi)
            },
            #[cfg(target_arch = "x86_64")]
            I8Isa::Avx2 => unsafe { x86::gemm_i8_avx2(q, w, y, ldy, epi) },
            #[cfg(target_arch = "x86_64")]
            I8Isa::Avx512Vnni => unsafe { x86::gemm_i8_avx512vnni(q, w, y, ldy, epi) },
        }
    }
}
