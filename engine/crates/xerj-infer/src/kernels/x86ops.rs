//! x86-64 AVX2 + FMA versions of the non-GEMM ops, selected at run time. The XERJ engine builds x86-64 for `x86-64-v2`
//! (SSE4.2, no AVX), so the portable code in `ops.rs` / `scalar.rs` compiles to SSE only and every `mul_add` becomes a libm
//! `fmaf` call; on a Broadwell Xeon that portable code took about two thirds of a question's time (self-attention alone half).
//! `XERJ_AVX2_OPS=0` keeps the portable code (A/B checks).
//!
//! Numerics:
//! * GELU / GeGLU: the portable functions themselves, compiled a second time with AVX2 + FMA enabled (`#[inline(always)]`
//!   bodies inside `#[target_feature]` wrappers). Same operations in the same order, no reassociation (Rust never
//!   contracts or reassociates float ops), and `mul_add` is a fused multiply-add in both builds: bit-identical.
//! * LayerNorm (round 2): intrinsics with the portable code's operations in its order (its 8-lane partial sums are one
//!   vector; LLVM kept most of them scalar in the AVX2 build of the portable code): bit-identical. RoPE (head_dim 64):
//!   likewise (`rope_qk64`).
//! * int8 activation quantization (also fused with LayerNorm): intrinsics that perform the scalar path's operations in its
//!   order (round half away from zero, as `f32::round`): bit-identical to `scalar::quantize_rows` of the LayerNorm output.
//! * transposed key copies: pure data movement.
//! * self-attention (head_dim 64): the NEON kernel's scheme with 16-query blocks: q·k accumulated over the head dimension in
//!   order with FMA, the exact two-pass softmax (max, exp with `ops::exp`'s algorithm, sum, normalize), P·V accumulated over
//!   the keys in order with FMA. Only the summation order differs from the portable code (f32 rounding).

use super::{QActs, QK};
use std::arch::x86_64::*;
use std::sync::OnceLock;

static ON: OnceLock<bool> = OnceLock::new();

/// Round-1 code kept for A/B checks of the round-2 x86 changes: `XERJ_X86_R1` = a comma list of components (`gemm`,
/// `attn`, `vrows`) or `all`. Unset (the default): every round-2 kernel is in use. Every round-2 change is bit-identical
/// to the round-1 code it replaces, so the switch only changes speed. `vrows` keeps the row-major value cache of the
/// first round-2 layout under the round-2 attention (see `LayerKv::vhead`).
pub fn r1_kept(component: &str) -> bool {
    std::env::var("XERJ_X86_R1").is_ok_and(|v| {
        v.split(',').any(|c| {
            let c = c.trim();
            c == component || c == "all"
        })
    })
}

static R2_ATTN: OnceLock<bool> = OnceLock::new();

/// the round-2 attention (blocked key layout, `attend_block64_r2`) is in use: AVX2 + FMA enabled and `XERJ_X86_R1` not
/// keeping `attn`
#[inline]
pub fn r2_attn() -> bool {
    *R2_ATTN.get_or_init(|| enabled() && !r1_kept("attn"))
}

static R2_VHEAD: OnceLock<bool> = OnceLock::new();

/// the blocked caches keep their values head-major (`LayerKv::vhead`, each head's value rows contiguous): the default;
/// `XERJ_X86_R1=vrows` keeps the row-major values (the same floats read by the same kernel: only the speed changes)
#[inline]
pub fn r2_vhead() -> bool {
    *R2_VHEAD.get_or_init(|| !r1_kept("vrows"))
}

static R2_LN: OnceLock<bool> = OnceLock::new();

/// the explicit AVX2 LayerNorm (`layer_norm_row_v`) is in use: the default; `XERJ_X86_R1=ln` keeps the portable code
/// compiled for AVX2 (bit-identical, A/B checks)
#[inline]
pub fn r2_ln() -> bool {
    *R2_LN.get_or_init(|| !r1_kept("ln"))
}

/// AVX2 + FMA present and not disabled with XERJ_AVX2_OPS=0
#[inline]
pub fn enabled() -> bool {
    *ON.get_or_init(|| {
        is_x86_feature_detected!("avx2")
            && is_x86_feature_detected!("fma")
            && std::env::var("XERJ_AVX2_OPS").map_or(true, |v| v != "0")
    })
}

// ------------------------------------------------------------------------------------------------ portable code, AVX2 build

/// `ops::layer_norm` for AVX2 + FMA: `layer_norm_row_v` per row (`XERJ_X86_R1=ln`: the portable code compiled for
/// AVX2 + FMA)
#[target_feature(enable = "avx2,fma")]
pub unsafe fn layer_norm(
    x: &[f32],
    m: usize,
    d: usize,
    w: &[f32],
    b: Option<&[f32]>,
    eps: f32,
    y: &mut [f32],
) {
    let v = r2_ln();
    for r in 0..m {
        let (xr, yr) = (&x[r * d..(r + 1) * d], &mut y[r * d..(r + 1) * d]);
        if v {
            layer_norm_row_v(xr, w, b, eps, yr);
        } else {
            super::ops::layer_norm_row_portable(xr, w, b, eps, yr);
        }
    }
}

/// `ops::layer_norm_row` for AVX2 + FMA (see `layer_norm`)
#[target_feature(enable = "avx2,fma")]
pub unsafe fn layer_norm_row(x: &[f32], w: &[f32], b: Option<&[f32]>, eps: f32, y: &mut [f32]) {
    if r2_ln() {
        layer_norm_row_v(x, w, b, eps, y)
    } else {
        super::ops::layer_norm_row_portable(x, w, b, eps, y)
    }
}

/// `ops::layer_norm_row_portable` with intrinsics (round 2), bit-identical: the statistics by `ln_stats` (the portable
/// code's 8-lane partial sums as one vector, the same lane fold, mean, variance and rstd), then
/// y = (x - mean) * rstd * w (+ b) per element in the portable order (separate multiplies and add, no FMA contraction).
/// The AVX2 build of the portable code keeps most of the lane sums scalar; this is about 3x faster on a 512-wide row.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn layer_norm_row_v(x: &[f32], w: &[f32], b: Option<&[f32]>, eps: f32, y: &mut [f32]) {
    let n = x.len();
    assert!(w.len() >= n && y.len() >= n && b.is_none_or(|b| b.len() >= n));
    let (mean, rstd) = ln_stats(x.as_ptr(), n, eps);
    let (vm, vr) = (_mm256_set1_ps(mean), _mm256_set1_ps(rstd));
    let (xp, wp, yp) = (x.as_ptr(), w.as_ptr(), y.as_mut_ptr());
    let chunks = n / 8;
    for c in 0..chunks {
        let i = c * 8;
        let mut v = _mm256_mul_ps(
            _mm256_mul_ps(_mm256_sub_ps(ldu(xp.add(i)), vm), vr),
            ldu(wp.add(i)),
        );
        if let Some(bb) = b {
            v = _mm256_add_ps(v, ldu(bb.as_ptr().add(i)));
        }
        _mm256_storeu_ps(yp.add(i), v);
    }
    for i in chunks * 8..n {
        let v = (x[i] - mean) * rstd * w[i];
        y[i] = match b {
            Some(bb) => v + bb[i],
            None => v,
        };
    }
}

/// `ops::geglu` compiled for AVX2 + FMA
#[target_feature(enable = "avx2,fma")]
pub unsafe fn geglu(u: &[f32], m: usize, h: usize, g: &mut [f32]) {
    super::ops::geglu_portable(u, m, h, g)
}

/// `ops::gelu_inplace` compiled for AVX2 + FMA
#[target_feature(enable = "avx2,fma")]
pub unsafe fn gelu_inplace(x: &mut [f32]) {
    super::ops::gelu_inplace_portable(x)
}

/// `ops::rope_head` on every head of a q | k row pair, compiled for AVX2 + FMA
#[target_feature(enable = "avx2,fma")]
pub unsafe fn rope_qk(
    row: &mut [f32],
    d: usize,
    heads: usize,
    hd: usize,
    cos: &[f32],
    sin: &[f32],
) {
    for h in 0..heads {
        super::ops::rope_head_portable(&mut row[h * hd..(h + 1) * hd], cos, sin);
        super::ops::rope_head_portable(&mut row[d + h * hd..d + (h + 1) * hd], cos, sin);
    }
}

/// `ops::rope_head` on every head (head_dim 64) of a q | k row pair with explicit AVX2 code (round 2): per element the
/// portable function's operations in its order, `x1 * cos - x2 * sin` and `x2 * cos + x1 * sin` (separate multiplies,
/// no FMA contraction), so the result is bit-identical; loads are single VLDDQU instructions (see [`ldu`])
#[target_feature(enable = "avx2,fma")]
pub unsafe fn rope_qk64(row: &mut [f32], d: usize, heads: usize, cos: &[f32], sin: &[f32]) {
    assert!(row.len() >= 2 * d && heads * 64 <= d && cos.len() >= 32 && sin.len() >= 32);
    let (cp, sp) = (cos.as_ptr(), sin.as_ptr());
    let c = [ldu(cp), ldu(cp.add(8)), ldu(cp.add(16)), ldu(cp.add(24))];
    let s = [ldu(sp), ldu(sp.add(8)), ldu(sp.add(16)), ldu(sp.add(24))];
    let base = row.as_mut_ptr();
    for part in 0..2 {
        for h in 0..heads {
            let x = base.add(part * d + h * 64);
            for i in 0..4 {
                let x1 = ldu(x.add(8 * i));
                let x2 = ldu(x.add(32 + 8 * i));
                let a = _mm256_sub_ps(_mm256_mul_ps(x1, c[i]), _mm256_mul_ps(x2, s[i]));
                let b = _mm256_add_ps(_mm256_mul_ps(x2, c[i]), _mm256_mul_ps(x1, s[i]));
                _mm256_storeu_ps(x.add(8 * i), a);
                _mm256_storeu_ps(x.add(32 + 8 * i), b);
            }
        }
    }
}

// ------------------------------------------------------------------------------------------------ helpers

#[inline(always)]
unsafe fn hmax(v: __m256) -> f32 {
    let mut t = [0f32; 8];
    _mm256_storeu_ps(t.as_mut_ptr(), v);
    t.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b))
}

#[inline(always)]
unsafe fn hsum(v: __m256) -> f32 {
    let mut t = [0f32; 8];
    _mm256_storeu_ps(t.as_mut_ptr(), v);
    t.iter().sum()
}

/// One unaligned 256-bit load as a single instruction (VLDDQU). The engine builds x86-64 for `x86-64-v2`, whose tuning
/// makes LLVM split every unaligned `_mm256_loadu_ps` into a 128-bit load plus a 128-bit insert from memory (three
/// uops, the insert on a vector ALU port); VLDDQU is one load uop on AVX2 CPUs. The same 32 bytes either way.
#[inline(always)]
pub unsafe fn ldu(p: *const f32) -> __m256 {
    _mm256_castsi256_ps(_mm256_lddqu_si256(p as *const __m256i))
}

/// round half away from zero (`f32::round`), exact for |x| < 2^23
#[inline(always)]
unsafe fn round_away(x: __m256) -> __m256 {
    let sign = _mm256_set1_ps(-0.0);
    let t = _mm256_round_ps::<{ _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC }>(x);
    let frac = _mm256_andnot_ps(sign, _mm256_sub_ps(x, t));
    let up = _mm256_cmp_ps::<_CMP_GE_OQ>(frac, _mm256_set1_ps(0.5));
    let one = _mm256_or_ps(_mm256_set1_ps(1.0), _mm256_and_ps(x, sign));
    _mm256_add_ps(t, _mm256_and_ps(up, one))
}

/// e^x on 8 lanes: `ops::exp`'s range reduction (round half away), polynomial and constants, FMA where it uses mul_add
#[inline(always)]
pub unsafe fn exp8(x: __m256) -> __m256 {
    let x = _mm256_min_ps(
        _mm256_max_ps(x, _mm256_set1_ps(-87.336_55)),
        _mm256_set1_ps(88.376_26),
    );
    let n = round_away(_mm256_mul_ps(x, _mm256_set1_ps(std::f32::consts::LOG2_E)));
    let r = _mm256_fmadd_ps(n, _mm256_set1_ps(-0.693_359_4), x);
    let r = _mm256_fmadd_ps(n, _mm256_set1_ps(2.121_944_4e-4), r);
    let mut p = _mm256_set1_ps(1.987_569_1e-4);
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.398_199_9e-3));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(8.333_452e-3));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(4.166_579_6e-2));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.666_666_5e-1));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(5.000_000_1e-1));
    let r2 = _mm256_mul_ps(r, r);
    let y = _mm256_add_ps(_mm256_fmadd_ps(p, r2, r), _mm256_set1_ps(1.0));
    let e = _mm256_slli_epi32::<23>(_mm256_add_epi32(
        _mm256_cvttps_epi32(n),
        _mm256_set1_epi32(127),
    ));
    _mm256_mul_ps(y, _mm256_castsi256_ps(e))
}

// ------------------------------------------------------------------------------------------------ int8 activation quantization

/// 8 values (already multiplied by 127 / amax) -> 8 int8 at `d`: round half away, clamp to [-127, 127]
#[inline(always)]
unsafe fn store_q8(d: *mut i8, v: __m256) {
    let r = _mm256_min_ps(
        _mm256_max_ps(round_away(v), _mm256_set1_ps(-127.0)),
        _mm256_set1_ps(127.0),
    );
    let i = _mm256_cvttps_epi32(r);
    let w = _mm_packs_epi32(_mm256_castsi256_si128(i), _mm256_extracti128_si256::<1>(i));
    _mm_storel_epi64(d as *mut __m128i, _mm_packs_epi16(w, w));
}

/// `scalar::quantize_rows` with AVX2 (bit-identical): per-row symmetric int8 into the row-pair layout; `pre` = smoothing
#[target_feature(enable = "avx2,fma")]
pub unsafe fn quantize_rows(x: &[f32], m: usize, k: usize, pre: Option<&[f32]>, q: &mut QActs) {
    let kbs = q.kp / QK;
    let dp = q.data.as_mut_slice().as_mut_ptr();
    let sp = q.scales.as_mut_slice().as_mut_ptr();
    quantize_rows_range(x, 0, m, k, pre, kbs, dp, sp);
    pad_odd_row(m, kbs, dp, sp);
}

/// the zero padding row of an odd row count (its outputs are never stored)
#[inline(always)]
pub unsafe fn pad_odd_row(m: usize, kbs: usize, dp: *mut i8, sp: *mut f32) {
    if m % 2 == 1 {
        let pair = m / 2;
        for kb in 0..kbs {
            std::ptr::write_bytes(dp.add((pair * kbs + kb) * 16 + 8), 0, 8);
        }
        *sp.add(m) = 0.0;
    }
}

/// rows r0..r1 of `quantize_rows` into the row-pair layout at `dp` (kbs k-blocks per pair) and the scales at `sp`: only
/// these rows' bytes and scales are written, so disjoint row ranges with even bounds can run in parallel
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn quantize_rows_range(
    x: &[f32],
    r0: usize,
    r1: usize,
    k: usize,
    pre: Option<&[f32]>,
    kbs: usize,
    dp: *mut i8,
    sp: *mut f32,
) {
    let pp = pre.map(|p| p.as_ptr());
    let sign = _mm256_set1_ps(-0.0);
    for r in r0..r1 {
        let row = x.as_ptr().add(r * k);
        let load = |i: usize| -> __m256 {
            let v = ldu(row.add(i));
            match pp {
                Some(p) => _mm256_mul_ps(v, ldu(p.add(i))),
                None => v,
            }
        };
        let elem = |i: usize| -> f32 {
            match pp {
                Some(p) => *row.add(i) * *p.add(i),
                None => *row.add(i),
            }
        };
        let mut mx = _mm256_setzero_ps();
        let mut i = 0;
        while i + 8 <= k {
            mx = _mm256_max_ps(mx, _mm256_andnot_ps(sign, load(i)));
            i += 8;
        }
        let mut amax = hmax(mx).max(0.0);
        while i < k {
            amax = amax.max(elem(i).abs());
            i += 1;
        }
        let (s, inv) = if amax > 0.0 {
            (amax / 127.0, 127.0 / amax)
        } else {
            (0.0, 0.0)
        };
        *sp.add(r) = s;
        let (pair, half) = (r / 2, r % 2);
        let vinv = _mm256_set1_ps(inv);
        for kb in 0..kbs {
            let d = dp.add((pair * kbs + kb) * 16 + half * 8);
            let i0 = kb * QK;
            if i0 + 8 <= k {
                store_q8(d, _mm256_mul_ps(load(i0), vinv));
            } else {
                for kk in 0..QK {
                    let ii = i0 + kk;
                    *d.add(kk) = if ii < k {
                        (elem(ii) * inv).round().clamp(-127.0, 127.0) as i8
                    } else {
                        0
                    };
                }
            }
        }
    }
}

/// LayerNorm statistics exactly as `ops::layer_norm_row_portable`: (mean, rstd)
#[inline(always)]
unsafe fn ln_stats(row: *const f32, k: usize, eps: f32) -> (f32, f32) {
    let chunks = k / 8;
    let mut s = _mm256_setzero_ps();
    for c in 0..chunks {
        s = _mm256_add_ps(s, ldu(row.add(c * 8)));
    }
    let mut t = [0f32; 8];
    _mm256_storeu_ps(t.as_mut_ptr(), s);
    let mut sum: f32 = t.iter().sum();
    for i in chunks * 8..k {
        sum += *row.add(i);
    }
    let mean = sum / k as f32;
    let vm = _mm256_set1_ps(mean);
    let mut q = _mm256_setzero_ps();
    for c in 0..chunks {
        let d = _mm256_sub_ps(ldu(row.add(c * 8)), vm);
        q = _mm256_add_ps(q, _mm256_mul_ps(d, d));
    }
    _mm256_storeu_ps(t.as_mut_ptr(), q);
    let mut var: f32 = t.iter().sum();
    for i in chunks * 8..k {
        let d = *row.add(i) - mean;
        var += d * d;
    }
    var /= k as f32;
    (mean, 1.0 / (var.max(0.0) + eps).sqrt())
}

/// fused LayerNorm + int8 quantization: bit-identical to `ops::layer_norm` (w, b, eps) followed by `quantize_rows` with
/// `pre` (the LayerNorm output is recomputed in both passes, never stored). Requires k % 8 == 0.
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn quantize_rows_ln(
    x: &[f32],
    m: usize,
    k: usize,
    w: &[f32],
    b: Option<&[f32]>,
    eps: f32,
    pre: Option<&[f32]>,
    q: &mut QActs,
) {
    debug_assert!(k % 8 == 0);
    let kbs = q.kp / QK;
    let dp = q.data.as_mut_slice().as_mut_ptr();
    let sp = q.scales.as_mut_slice().as_mut_ptr();
    quantize_rows_ln_range(x, 0, m, k, w, b, eps, pre, kbs, dp, sp);
    pad_odd_row(m, kbs, dp, sp);
}

/// rows r0..r1 of `quantize_rows_ln` (see `quantize_rows_range`)
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn quantize_rows_ln_range(
    x: &[f32],
    r0: usize,
    r1: usize,
    k: usize,
    w: &[f32],
    b: Option<&[f32]>,
    eps: f32,
    pre: Option<&[f32]>,
    kbs: usize,
    dp: *mut i8,
    sp: *mut f32,
) {
    debug_assert!(k % 8 == 0);
    let wp = w.as_ptr();
    let bp = b.map(|b| b.as_ptr());
    let pp = pre.map(|p| p.as_ptr());
    let sign = _mm256_set1_ps(-0.0);
    for r in r0..r1 {
        let row = x.as_ptr().add(r * k);
        let (mean, rstd) = ln_stats(row, k, eps);
        let (vm, vr) = (_mm256_set1_ps(mean), _mm256_set1_ps(rstd));
        // y = (x - mean) * rstd * w (+ b), then * pre: the operations of layer_norm_row and quantize_rows, in order
        let y = |i: usize| -> __m256 {
            let mut v = _mm256_mul_ps(
                _mm256_mul_ps(_mm256_sub_ps(ldu(row.add(i)), vm), vr),
                ldu(wp.add(i)),
            );
            if let Some(bb) = bp {
                v = _mm256_add_ps(v, ldu(bb.add(i)));
            }
            if let Some(p) = pp {
                v = _mm256_mul_ps(v, ldu(p.add(i)));
            }
            v
        };
        let mut mx = _mm256_setzero_ps();
        let mut i = 0;
        while i < k {
            mx = _mm256_max_ps(mx, _mm256_andnot_ps(sign, y(i)));
            i += 8;
        }
        let amax = hmax(mx).max(0.0);
        let (s, inv) = if amax > 0.0 {
            (amax / 127.0, 127.0 / amax)
        } else {
            (0.0, 0.0)
        };
        *sp.add(r) = s;
        let (pair, half) = (r / 2, r % 2);
        let vinv = _mm256_set1_ps(inv);
        for kb in 0..kbs {
            store_q8(
                dp.add((pair * kbs + kb) * 16 + half * 8),
                _mm256_mul_ps(y(kb * QK), vinv),
            );
        }
    }
}

// ------------------------------------------------------------------------------------------------ transposed keys

/// `LayerKv::transpose_keys` with 8 x 8 register transposes: kt[i * kts + p] = k[p * d + i] for rows p0..p1, all i < d
#[target_feature(enable = "avx2,fma")]
pub unsafe fn transpose_keys(
    k: *const f32,
    d: usize,
    kt: *mut f32,
    kts: usize,
    p0: usize,
    p1: usize,
) {
    let mut p = p0;
    if d % 8 == 0 {
        while p + 8 <= p1 {
            for i0 in (0..d).step_by(8) {
                let r = [
                    _mm256_loadu_ps(k.add(p * d + i0)),
                    _mm256_loadu_ps(k.add((p + 1) * d + i0)),
                    _mm256_loadu_ps(k.add((p + 2) * d + i0)),
                    _mm256_loadu_ps(k.add((p + 3) * d + i0)),
                    _mm256_loadu_ps(k.add((p + 4) * d + i0)),
                    _mm256_loadu_ps(k.add((p + 5) * d + i0)),
                    _mm256_loadu_ps(k.add((p + 6) * d + i0)),
                    _mm256_loadu_ps(k.add((p + 7) * d + i0)),
                ];
                let t0 = _mm256_unpacklo_ps(r[0], r[1]);
                let t1 = _mm256_unpackhi_ps(r[0], r[1]);
                let t2 = _mm256_unpacklo_ps(r[2], r[3]);
                let t3 = _mm256_unpackhi_ps(r[2], r[3]);
                let t4 = _mm256_unpacklo_ps(r[4], r[5]);
                let t5 = _mm256_unpackhi_ps(r[4], r[5]);
                let t6 = _mm256_unpacklo_ps(r[6], r[7]);
                let t7 = _mm256_unpackhi_ps(r[6], r[7]);
                let s0 = _mm256_shuffle_ps::<0x44>(t0, t2);
                let s1 = _mm256_shuffle_ps::<0xEE>(t0, t2);
                let s2 = _mm256_shuffle_ps::<0x44>(t1, t3);
                let s3 = _mm256_shuffle_ps::<0xEE>(t1, t3);
                let s4 = _mm256_shuffle_ps::<0x44>(t4, t6);
                let s5 = _mm256_shuffle_ps::<0xEE>(t4, t6);
                let s6 = _mm256_shuffle_ps::<0x44>(t5, t7);
                let s7 = _mm256_shuffle_ps::<0xEE>(t5, t7);
                let o = [
                    _mm256_permute2f128_ps::<0x20>(s0, s4),
                    _mm256_permute2f128_ps::<0x20>(s1, s5),
                    _mm256_permute2f128_ps::<0x20>(s2, s6),
                    _mm256_permute2f128_ps::<0x20>(s3, s7),
                    _mm256_permute2f128_ps::<0x31>(s0, s4),
                    _mm256_permute2f128_ps::<0x31>(s1, s5),
                    _mm256_permute2f128_ps::<0x31>(s2, s6),
                    _mm256_permute2f128_ps::<0x31>(s3, s7),
                ];
                for (j, v) in o.iter().enumerate() {
                    _mm256_storeu_ps(kt.add((i0 + j) * kts + p), *v);
                }
            }
            p += 8;
        }
    }
    for i in 0..d {
        for pp in p..p1 {
            *kt.add(i * kts + pp) = *k.add(pp * d + i);
        }
    }
}

// ------------------------------------------------------------------------------------------------ self-attention, head_dim 64

/// query rows per attention call
pub const QB: usize = 16;

/// floats of `scores` scratch `attend_block64` needs for blocks spanning at most `maxk` keys
pub fn attn_scratch(maxk: usize) -> usize {
    QB * (maxk + 32)
}

/// Softmax attention of up to 16 query rows (one head, head_dim 64) over keys [lo, hi]. Keys TRANSPOSED: `kt[c * kts + j]`
/// = key j, dimension c of this head, with `kts` >= hi + 16 (up to 15 slack columns are read; their scores are discarded).
/// Values row-major: `v[j * vs + c]`. Query r (< nq) sees keys lo_r[r]..=hi_r[r] inside [lo, hi]; rows nq..16 of `q` /
/// `out` may repeat a valid row (never written). `scores`: `attn_scratch(hi - lo + 1)` floats.
///
/// Scores: 16-key blocks outer (the 4 KB K^T block stays in L1 for all 4-query tiles), 4 queries x 16 keys per tile,
/// accumulated over the 64 dimensions in order with FMA, times `scale`. Softmax per row over its own range (exact two-pass:
/// max, `exp8`, sum, multiply by 1 / sum; zeros elsewhere). P·V: 64-key blocks outer (the 16 KB V block stays in L1),
/// 4 queries x 16 dimensions per pass, keys in order with FMA, accumulated in the output rows across blocks.
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn attend_block64(
    q: &[*const f32; QB],
    nq: usize,
    kt: *const f32,
    kts: usize,
    v: *const f32,
    vs: usize,
    lo: usize,
    hi: usize,
    lo_r: &[usize; QB],
    hi_r: &[usize; QB],
    scale: f32,
    scores: *mut f32,
    out: &[*mut f32; QB],
) {
    debug_assert!(nq >= 1 && nq <= QB && hi >= lo);
    let nk = hi + 1 - lo;
    let ns = nk.div_ceil(16) * 16;
    let ntiles = nq.div_ceil(4);
    // per tile: key range relative to lo (union of its rows)
    let mut ta = [usize::MAX; QB / 4];
    let mut tb = [0usize; QB / 4];
    for r in 0..nq {
        let t = r / 4;
        ta[t] = ta[t].min(lo_r[r] - lo);
        tb[t] = tb[t].max(hi_r[r] - lo);
    }
    let vsc = _mm256_set1_ps(scale);
    // ---- scores
    let mut j = 0;
    while j < nk {
        for t in 0..ntiles {
            if j + 16 <= ta[t] || j > tb[t] {
                continue;
            }
            let qp = [q[4 * t], q[4 * t + 1], q[4 * t + 2], q[4 * t + 3]];
            let mut acc = [[_mm256_setzero_ps(); 2]; 4];
            let kb = kt.add(lo + j);
            for c in 0..64 {
                let kr = kb.add(c * kts);
                let k0 = _mm256_loadu_ps(kr);
                let k1 = _mm256_loadu_ps(kr.add(8));
                for r in 0..4 {
                    let qc = _mm256_broadcast_ss(&*qp[r].add(c));
                    acc[r][0] = _mm256_fmadd_ps(qc, k0, acc[r][0]);
                    acc[r][1] = _mm256_fmadd_ps(qc, k1, acc[r][1]);
                }
            }
            for r in 0..4 {
                let row = scores.add((4 * t + r) * ns + j);
                _mm256_storeu_ps(row, _mm256_mul_ps(acc[r][0], vsc));
                _mm256_storeu_ps(row.add(8), _mm256_mul_ps(acc[r][1], vsc));
            }
        }
        j += 16;
    }
    // ---- softmax per row over its own range; normalized in place, zeros elsewhere (and in the padding rows)
    for r in 0..nq {
        let row = scores.add(r * ns);
        let (a, b) = (lo_r[r] - lo, hi_r[r] - lo);
        let mut vm = _mm256_set1_ps(f32::NEG_INFINITY);
        let mut t = a;
        while t + 8 <= b + 1 {
            vm = _mm256_max_ps(vm, _mm256_loadu_ps(row.add(t)));
            t += 8;
        }
        let mut mx = hmax(vm);
        while t <= b {
            mx = mx.max(*row.add(t));
            t += 1;
        }
        let vmx = _mm256_set1_ps(mx);
        let mut sum = _mm256_setzero_ps();
        let mut t = a;
        while t + 8 <= b + 1 {
            let e = exp8(_mm256_sub_ps(_mm256_loadu_ps(row.add(t)), vmx));
            _mm256_storeu_ps(row.add(t), e);
            sum = _mm256_add_ps(sum, e);
            t += 8;
        }
        let mut s = hsum(sum);
        while t <= b {
            let e = super::ops::exp(*row.add(t) - mx);
            *row.add(t) = e;
            s += e;
            t += 1;
        }
        let vi = _mm256_set1_ps(1.0 / s);
        let mut t = a;
        while t + 8 <= b + 1 {
            _mm256_storeu_ps(row.add(t), _mm256_mul_ps(_mm256_loadu_ps(row.add(t)), vi));
            t += 8;
        }
        let inv = 1.0 / s;
        while t <= b {
            *row.add(t) *= inv;
            t += 1;
        }
        for t in 0..a {
            *row.add(t) = 0.0;
        }
        for t in b + 1..ns {
            *row.add(t) = 0.0;
        }
    }
    for r in nq..4 * ntiles {
        std::ptr::write_bytes(scores.add(r * ns), 0, ns);
    }
    // ---- out = P · V
    for r in 0..nq {
        std::ptr::write_bytes(out[r], 0, 64);
    }
    let mut j0 = 0;
    while j0 < nk {
        let j1 = (j0 + 64).min(nk);
        for t in 0..ntiles {
            let (a, b) = (ta[t].max(j0), tb[t].min(j1 - 1));
            if a > b {
                continue;
            }
            let rows = (nq - 4 * t).min(4);
            let pr = [
                scores.add(4 * t * ns),
                scores.add((4 * t + 1) * ns),
                scores.add((4 * t + 2) * ns),
                scores.add((4 * t + 3) * ns),
            ];
            for c in 0..4 {
                let mut acc = [[_mm256_setzero_ps(); 2]; 4];
                for r in 0..rows {
                    let o = out[4 * t + r].add(16 * c);
                    acc[r][0] = _mm256_loadu_ps(o);
                    acc[r][1] = _mm256_loadu_ps(o.add(8));
                }
                for j in a..=b {
                    let vr = v.add((lo + j) * vs + 16 * c);
                    let v0 = _mm256_loadu_ps(vr);
                    let v1 = _mm256_loadu_ps(vr.add(8));
                    for r in 0..4 {
                        let p = _mm256_broadcast_ss(&*pr[r].add(j));
                        acc[r][0] = _mm256_fmadd_ps(p, v0, acc[r][0]);
                        acc[r][1] = _mm256_fmadd_ps(p, v1, acc[r][1]);
                    }
                }
                for r in 0..rows {
                    let o = out[4 * t + r].add(16 * c);
                    _mm256_storeu_ps(o, acc[r][0]);
                    _mm256_storeu_ps(o.add(8), acc[r][1]);
                }
            }
        }
        j0 = j1;
    }
}

// ------------------------------------------------------------------------------------------------ round 2: blocked keys

/// 8 x 8 transpose of the rows r[0..8] (8 floats each): out[j] lane i = r[i] lane j
#[inline(always)]
unsafe fn transpose8(r: [__m256; 8]) -> [__m256; 8] {
    let t0 = _mm256_unpacklo_ps(r[0], r[1]);
    let t1 = _mm256_unpackhi_ps(r[0], r[1]);
    let t2 = _mm256_unpacklo_ps(r[2], r[3]);
    let t3 = _mm256_unpackhi_ps(r[2], r[3]);
    let t4 = _mm256_unpacklo_ps(r[4], r[5]);
    let t5 = _mm256_unpackhi_ps(r[4], r[5]);
    let t6 = _mm256_unpacklo_ps(r[6], r[7]);
    let t7 = _mm256_unpackhi_ps(r[6], r[7]);
    let s0 = _mm256_shuffle_ps::<0x44>(t0, t2);
    let s1 = _mm256_shuffle_ps::<0xEE>(t0, t2);
    let s2 = _mm256_shuffle_ps::<0x44>(t1, t3);
    let s3 = _mm256_shuffle_ps::<0xEE>(t1, t3);
    let s4 = _mm256_shuffle_ps::<0x44>(t4, t6);
    let s5 = _mm256_shuffle_ps::<0xEE>(t4, t6);
    let s6 = _mm256_shuffle_ps::<0x44>(t5, t7);
    let s7 = _mm256_shuffle_ps::<0xEE>(t5, t7);
    [
        _mm256_permute2f128_ps::<0x20>(s0, s4),
        _mm256_permute2f128_ps::<0x20>(s1, s5),
        _mm256_permute2f128_ps::<0x20>(s2, s6),
        _mm256_permute2f128_ps::<0x20>(s3, s7),
        _mm256_permute2f128_ps::<0x31>(s0, s4),
        _mm256_permute2f128_ps::<0x31>(s1, s5),
        _mm256_permute2f128_ps::<0x31>(s2, s6),
        _mm256_permute2f128_ps::<0x31>(s3, s7),
    ]
}

/// Keys of positions p0..p1 into the blocked layout of `LayerKv::kblk` (head_dim 64): the key of position p is the
/// `heads * 64` floats at `src + (p - p0) * stride`; `dst[h * hs + (p / 16) * 1024 + c * 16 + p % 16]` = its dimension c
/// of head h. Aligned 8-key groups go through 8 x 8 register transposes (one aligned 32-byte store per dimension), the
/// rest element by element. Pure data movement.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn store_keys_blocked(
    src: *const f32,
    stride: usize,
    heads: usize,
    dst: *mut f32,
    hs: usize,
    p0: usize,
    p1: usize,
) {
    let one = |p: usize| {
        let key = src.add((p - p0) * stride);
        for h in 0..heads {
            let base = dst.add(h * hs + (p / 16) * 1024 + p % 16);
            for c in 0..64 {
                *base.add(c * 16) = *key.add(h * 64 + c);
            }
        }
    };
    let mut p = p0;
    while p < p1 && p % 8 != 0 {
        one(p);
        p += 1;
    }
    while p + 8 <= p1 {
        let rows = src.add((p - p0) * stride);
        for h in 0..heads {
            // 64-byte aligned buffer, block offset a multiple of 1024 floats, p % 16 in {0, 8}: 32-byte aligned stores
            let base = dst.add(h * hs + (p / 16) * 1024 + p % 16);
            for c0 in (0..64).step_by(8) {
                let col = h * 64 + c0;
                let r = [
                    ldu(rows.add(col)),
                    ldu(rows.add(stride + col)),
                    ldu(rows.add(2 * stride + col)),
                    ldu(rows.add(3 * stride + col)),
                    ldu(rows.add(4 * stride + col)),
                    ldu(rows.add(5 * stride + col)),
                    ldu(rows.add(6 * stride + col)),
                    ldu(rows.add(7 * stride + col)),
                ];
                let o = transpose8(r);
                for (j, v) in o.iter().enumerate() {
                    _mm256_store_ps(base.add((c0 + j) * 16), *v);
                }
            }
        }
        p += 8;
    }
    while p < p1 {
        one(p);
        p += 1;
    }
}

// ------------------------------------------------------------------------------------------------ round 2: attention

/// query rows per round-2 attention call: a multiple of 4, so the P·V tiles (4 consecutive rows) are those of round 1
pub const QB2: usize = 24;
/// query rows per Q·K tile of the round-2 kernel (12 accumulators: FMA-throughput-bound instead of latency-bound)
const QT: usize = 6;

/// floats of `scores` scratch `attend_block64_r2` needs for blocks spanning at most `maxk` keys (incl. 32-byte alignment)
pub fn attn_scratch2(maxk: usize) -> usize {
    QB2 * (maxk + 32) + 8
}

/// Round-2 softmax attention of up to `QB2` query rows of one head (head_dim 64) over keys [lo, hi], keys in the blocked
/// layout (`kb` = this head's blocks: key j, dimension c at `kb[(j / 16) * 1024 + c * 16 + j % 16]`), value j, dimension c
/// at `v[j * vs + c]` (`v` 32-byte aligned and `vs` a multiple of 8: this head's contiguous value rows with `vs` = 64, the
/// default `LayerKv::vhead` layout, or the row-major cache with `vs` = heads * 64). Query r (< nq) sees keys
/// lo_r[r]..=hi_r[r]; rows nq..QB2 of `q` / `out` may repeat a valid row (never written). `scores`: `attn_scratch2(hi + 1)`
/// floats.
///
/// Bit-identical to `attend_block64` on the same rows (the same per-element operations in the same order):
/// * scores: q·k accumulated over the 64 dimensions in order with FMA, times `scale`. Each score is its own FMA chain, so
///   the tiling is free: 16-key blocks aligned to the cache blocks (aligned loads, contiguous 4 KB per block), 6-row
///   tiles (12 independent accumulators);
/// * softmax per row over its own range, the round-1 code with the row's own first key as the vector origin (the same
///   8-lane grouping of the max, the exponentials and the sum);
/// * P·V: 4-row tiles of consecutive rows over the union of their key ranges, as in round 1 (the same terms, including
///   the zero probabilities of a tile's other rows), keys in order with FMA, accumulated in the output rows across
///   64-key blocks; dimensions in chunks of 24 + 24 + 16 (12 accumulators for the first two).
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn attend_block64_r2(
    q: &[*const f32; QB2],
    nq: usize,
    kb: *const f32,
    v: *const f32,
    vs: usize,
    lo: usize,
    hi: usize,
    lo_r: &[usize; QB2],
    hi_r: &[usize; QB2],
    scale: f32,
    scores: &mut [f32],
    out: &[*mut f32; QB2],
) {
    debug_assert!(nq >= 1 && nq <= QB2 && hi >= lo);
    let base = lo & !15;
    let nk = hi + 1 - base;
    let ns = nk.div_ceil(16) * 16;
    let n4 = nq.div_ceil(4);
    let n6 = nq.div_ceil(QT);
    // 32-byte aligned rows of the scores scratch
    let sp = {
        let p = scores.as_mut_ptr();
        let mis = (p as usize / 4) % 8;
        p.add((8 - mis) % 8)
    };
    debug_assert!(scores.len() >= QB2 * ns + 8);
    // per tile: key range relative to `base` (union of its rows)
    let mut ta6 = [usize::MAX; QB2 / QT];
    let mut tb6 = [0usize; QB2 / QT];
    let mut ta4 = [usize::MAX; QB2 / 4];
    let mut tb4 = [0usize; QB2 / 4];
    for r in 0..nq {
        let (a, b) = (lo_r[r] - base, hi_r[r] - base);
        ta6[r / QT] = ta6[r / QT].min(a);
        tb6[r / QT] = tb6[r / QT].max(b);
        ta4[r / 4] = ta4[r / 4].min(a);
        tb4[r / 4] = tb4[r / 4].max(b);
    }
    let vsc = _mm256_set1_ps(scale);
    // ---- scores: 16-key cache blocks outer, 6-row tiles
    let mut j = 0;
    while j < ns {
        let kblk = kb.add(((base + j) / 16) * 1024);
        for t in 0..n6 {
            if j + 16 <= ta6[t] || j > tb6[t] {
                continue;
            }
            let qp = [
                q[QT * t],
                q[QT * t + 1],
                q[QT * t + 2],
                q[QT * t + 3],
                q[QT * t + 4],
                q[QT * t + 5],
            ];
            let mut acc = [[_mm256_setzero_ps(); 2]; QT];
            for c in 0..64 {
                let k0 = _mm256_load_ps(kblk.add(c * 16));
                let k1 = _mm256_load_ps(kblk.add(c * 16 + 8));
                for r in 0..QT {
                    let qc = _mm256_broadcast_ss(&*qp[r].add(c));
                    acc[r][0] = _mm256_fmadd_ps(qc, k0, acc[r][0]);
                    acc[r][1] = _mm256_fmadd_ps(qc, k1, acc[r][1]);
                }
            }
            for r in 0..QT {
                let row = sp.add((QT * t + r) * ns + j);
                _mm256_store_ps(row, _mm256_mul_ps(acc[r][0], vsc));
                _mm256_store_ps(row.add(8), _mm256_mul_ps(acc[r][1], vsc));
            }
        }
        j += 16;
    }
    // ---- softmax per row over its own range (the round-1 code); normalized in place, zeros elsewhere
    for r in 0..nq {
        let row = sp.add(r * ns);
        let (a, b) = (lo_r[r] - base, hi_r[r] - base);
        let mut vm = _mm256_set1_ps(f32::NEG_INFINITY);
        let mut t = a;
        while t + 8 <= b + 1 {
            vm = _mm256_max_ps(vm, ldu(row.add(t)));
            t += 8;
        }
        let mut mx = hmax(vm);
        while t <= b {
            mx = mx.max(*row.add(t));
            t += 1;
        }
        let vmx = _mm256_set1_ps(mx);
        let mut sum = _mm256_setzero_ps();
        let mut t = a;
        while t + 8 <= b + 1 {
            let e = exp8(_mm256_sub_ps(ldu(row.add(t)), vmx));
            _mm256_storeu_ps(row.add(t), e);
            sum = _mm256_add_ps(sum, e);
            t += 8;
        }
        let mut s = hsum(sum);
        while t <= b {
            let e = super::ops::exp(*row.add(t) - mx);
            *row.add(t) = e;
            s += e;
            t += 1;
        }
        let vi = _mm256_set1_ps(1.0 / s);
        let mut t = a;
        while t + 8 <= b + 1 {
            _mm256_storeu_ps(row.add(t), _mm256_mul_ps(ldu(row.add(t)), vi));
            t += 8;
        }
        let inv = 1.0 / s;
        while t <= b {
            *row.add(t) *= inv;
            t += 1;
        }
        for t in 0..a {
            *row.add(t) = 0.0;
        }
        for t in b + 1..ns {
            *row.add(t) = 0.0;
        }
    }
    for r in nq..4 * n4 {
        std::ptr::write_bytes(sp.add(r * ns), 0, ns);
    }
    // ---- out = P · V: 64-key blocks outer (the V block stays in L1), 4-row tiles, dimension chunks 24 / 24 / 16
    for r in 0..nq {
        std::ptr::write_bytes(out[r], 0, 64);
    }
    let mut j0 = 0;
    while j0 < nk {
        let j1 = (j0 + 64).min(nk);
        for t in 0..n4 {
            let (a, b) = (ta4[t].max(j0), tb4[t].min(j1 - 1));
            if a > b {
                continue;
            }
            let rows = (nq - 4 * t).min(4);
            let pr = [
                sp.add(4 * t * ns),
                sp.add((4 * t + 1) * ns),
                sp.add((4 * t + 2) * ns),
                sp.add((4 * t + 3) * ns),
            ];
            let o = [out[4 * t], out[4 * t + 1], out[4 * t + 2], out[4 * t + 3]];
            pv_chunk3(&pr, &o, rows, v.add(base * vs), vs, a, b, 0);
            pv_chunk3(&pr, &o, rows, v.add(base * vs), vs, a, b, 24);
            pv_chunk2(&pr, &o, rows, v.add(base * vs), vs, a, b, 48);
        }
        j0 = j1;
    }
}

/// P·V of a 4-row tile for dimensions c0..c0 + 24 over keys a..=b (relative to `v`): out += Σ_j p[r][j] · v[j]
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn pv_chunk3(
    pr: &[*mut f32; 4],
    o: &[*mut f32; 4],
    rows: usize,
    v: *const f32,
    vs: usize,
    a: usize,
    b: usize,
    c0: usize,
) {
    let mut acc = [[_mm256_setzero_ps(); 3]; 4];
    for r in 0..rows {
        for i in 0..3 {
            acc[r][i] = ldu(o[r].add(c0 + 8 * i));
        }
    }
    // keys a..=b in order (an index for the probabilities, a pointer step for the value rows)
    let mut vr = v.add(a * vs + c0);
    let mut j = a;
    while j <= b {
        // 32-byte aligned: v is, vs is a multiple of 8, c0 is 0 or 24
        let v0 = _mm256_load_ps(vr);
        let v1 = _mm256_load_ps(vr.add(8));
        let v2 = _mm256_load_ps(vr.add(16));
        for r in 0..4 {
            let p = _mm256_broadcast_ss(&*pr[r].add(j));
            acc[r][0] = _mm256_fmadd_ps(p, v0, acc[r][0]);
            acc[r][1] = _mm256_fmadd_ps(p, v1, acc[r][1]);
            acc[r][2] = _mm256_fmadd_ps(p, v2, acc[r][2]);
        }
        vr = vr.add(vs);
        j += 1;
    }
    for r in 0..rows {
        for i in 0..3 {
            _mm256_storeu_ps(o[r].add(c0 + 8 * i), acc[r][i]);
        }
    }
}

/// as `pv_chunk3` for dimensions c0..c0 + 16
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn pv_chunk2(
    pr: &[*mut f32; 4],
    o: &[*mut f32; 4],
    rows: usize,
    v: *const f32,
    vs: usize,
    a: usize,
    b: usize,
    c0: usize,
) {
    let mut acc = [[_mm256_setzero_ps(); 2]; 4];
    for r in 0..rows {
        for i in 0..2 {
            acc[r][i] = ldu(o[r].add(c0 + 8 * i));
        }
    }
    let mut vr = v.add(a * vs + c0);
    let mut j = a;
    while j <= b {
        let v0 = _mm256_load_ps(vr);
        let v1 = _mm256_load_ps(vr.add(8));
        for r in 0..4 {
            let p = _mm256_broadcast_ss(&*pr[r].add(j));
            acc[r][0] = _mm256_fmadd_ps(p, v0, acc[r][0]);
            acc[r][1] = _mm256_fmadd_ps(p, v1, acc[r][1]);
        }
        vr = vr.add(vs);
        j += 1;
    }
    for r in 0..rows {
        for i in 0..2 {
            _mm256_storeu_ps(o[r].add(c0 + 8 * i), acc[r][i]);
        }
    }
}
