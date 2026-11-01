//! aarch64 NEON kernels: f32 6×16 (FMLA by element), int8 8×8 SMMLA (FEAT_I8MM), int8 4×8 SDOT (FEAT_DotProd),
//! per-row int8 quantization. SMMLA is emitted with inline asm (the `vmmlaq_s32` intrinsic is still unstable in Rust).

use super::{AVec, Epi, PackedF32, PackedI8, QActs, NR, QK, QN};
use std::arch::aarch64::*;
use std::arch::asm;

const MR: usize = 6;

/// pack rows [r0, r0+rows) of x[m, k] as [k][MR] (rows >= m are zero)
#[inline(always)]
unsafe fn pack_a(x: &[f32], m: usize, k: usize, r0: usize, dst: *mut f32) {
    let rows = (m - r0).min(MR);
    for kk in 0..k {
        let d = dst.add(kk * MR);
        for r in 0..MR {
            *d.add(r) = if r < rows {
                *x.get_unchecked((r0 + r) * k + kk)
            } else {
                0.0
            };
        }
    }
}

#[inline(always)]
unsafe fn store16(
    y: *mut f32,
    acc: &[float32x4_t; 4],
    bias: Option<*const f32>,
    epi: Epi,
    nc: usize,
) {
    if nc == NR {
        let al = vdupq_n_f32(epi.alpha);
        for c in 0..4 {
            let mut v = acc[c];
            if let Some(b) = bias {
                v = vaddq_f32(v, vld1q_f32(b.add(4 * c)));
            }
            v = vmulq_f32(v, al);
            if epi.accumulate {
                v = vaddq_f32(vld1q_f32(y.add(4 * c)), v);
            }
            vst1q_f32(y.add(4 * c), v);
        }
    } else {
        let mut t = [0f32; NR];
        for c in 0..4 {
            vst1q_f32(t.as_mut_ptr().add(4 * c), acc[c]);
        }
        let ys = std::slice::from_raw_parts_mut(y, nc);
        let bs = bias.map(|b| std::slice::from_raw_parts(b, nc));
        super::epi_store(ys, &t[..nc], bs, epi);
    }
}

/// f32 GEMM: y = epi(x · Wᵀ + b)
#[target_feature(enable = "neon")]
pub unsafe fn gemm_f32(
    x: &[f32],
    m: usize,
    w: &PackedF32,
    y: &mut [f32],
    ldy: usize,
    epi: Epi,
    apack: &mut AVec<f32>,
) {
    gemm_f32_ptr(x, m, w, y.as_mut_ptr(), ldy, epi, apack)
}

/// `gemm_f32` with a raw output pointer: row r, column c at `yp + r * ldy + c` (c < w.n)
#[target_feature(enable = "neon")]
pub unsafe fn gemm_f32_ptr(
    x: &[f32],
    m: usize,
    w: &PackedF32,
    yp: *mut f32,
    ldy: usize,
    epi: Epi,
    apack: &mut AVec<f32>,
) {
    let k = w.k;
    let rbs = m.div_ceil(MR);
    apack.resize(rbs * k * MR);
    let ap = apack.as_mut_slice().as_mut_ptr();
    for rb in 0..rbs {
        pack_a(x, m, k, rb * MR, ap.add(rb * k * MR));
    }
    let wd = w.data.as_slice().as_ptr();
    for p in 0..w.panels() {
        let c0 = p * NR;
        let nc = (w.n - c0).min(NR);
        let b = wd.add(p * k * NR);
        let bias = w.bias.as_ref().map(|bb| bb.as_ptr().add(c0));
        for rb in 0..rbs {
            let a = ap.add(rb * k * MR) as *const f32;
            let rows = (m - rb * MR).min(MR);
            let z = vdupq_n_f32(0.0);
            let mut acc = [[z; 4]; MR];
            for kk in 0..k {
                let bk = b.add(kk * NR);
                let b0 = vld1q_f32(bk);
                let b1 = vld1q_f32(bk.add(4));
                let b2 = vld1q_f32(bk.add(8));
                let b3 = vld1q_f32(bk.add(12));
                let a03 = vld1q_f32(a.add(kk * MR));
                let a45 = vld1_f32(a.add(kk * MR + 4));
                macro_rules! row_q {
                    ($r:expr, $l:expr) => {
                        acc[$r][0] = vfmaq_laneq_f32::<$l>(acc[$r][0], b0, a03);
                        acc[$r][1] = vfmaq_laneq_f32::<$l>(acc[$r][1], b1, a03);
                        acc[$r][2] = vfmaq_laneq_f32::<$l>(acc[$r][2], b2, a03);
                        acc[$r][3] = vfmaq_laneq_f32::<$l>(acc[$r][3], b3, a03);
                    };
                }
                macro_rules! row_d {
                    ($r:expr, $l:expr) => {
                        acc[$r][0] = vfmaq_lane_f32::<$l>(acc[$r][0], b0, a45);
                        acc[$r][1] = vfmaq_lane_f32::<$l>(acc[$r][1], b1, a45);
                        acc[$r][2] = vfmaq_lane_f32::<$l>(acc[$r][2], b2, a45);
                        acc[$r][3] = vfmaq_lane_f32::<$l>(acc[$r][3], b3, a45);
                    };
                }
                row_q!(0, 0);
                row_q!(1, 1);
                row_q!(2, 2);
                row_q!(3, 3);
                row_d!(4, 0);
                row_d!(5, 1);
            }
            for r in 0..rows {
                store16(yp.add((rb * MR + r) * ldy + c0), &acc[r], bias, epi, nc);
            }
        }
    }
}

/// per-row symmetric int8 quantization (round half away from zero, as the scalar path) into the row-pair layout;
/// `pre`: per-channel factors multiplied in first (SmoothQuant-style smoothing), recomputed in both passes (no temporary)
#[target_feature(enable = "neon")]
pub unsafe fn quantize_rows(x: &[f32], m: usize, k: usize, pre: Option<&[f32]>, q: &mut QActs) {
    let kbs = q.kp / QK;
    let dp = q.data.as_mut_slice().as_mut_ptr();
    let sp = q.scales.as_mut_slice().as_mut_ptr();
    for r in 0..m {
        let row = x.as_ptr().add(r * k);
        let pp = pre.map(|p| p.as_ptr());
        let load = |i: usize| -> float32x4_t {
            let v = vld1q_f32(row.add(i));
            match pp {
                Some(p) => vmulq_f32(v, vld1q_f32(p.add(i))),
                None => v,
            }
        };
        let elem = |i: usize| -> f32 {
            match pp {
                Some(p) => *row.add(i) * *p.add(i),
                None => *row.add(i),
            }
        };
        let mut mx = vdupq_n_f32(0.0);
        let mut i = 0;
        while i + 4 <= k {
            mx = vmaxq_f32(mx, vabsq_f32(load(i)));
            i += 4;
        }
        let mut amax = vmaxvq_f32(mx);
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
        let vinv = vdupq_n_f32(inv);
        let lo = vdupq_n_s32(-127);
        let hi = vdupq_n_s32(127);
        for kb in 0..kbs {
            let d = dp.add((pair * kbs + kb) * 16 + half * 8);
            let i0 = kb * QK;
            if i0 + 8 <= k {
                let v0 = vcvtaq_s32_f32(vmulq_f32(load(i0), vinv));
                let v1 = vcvtaq_s32_f32(vmulq_f32(load(i0 + 4), vinv));
                let v0 = vminq_s32(vmaxq_s32(v0, lo), hi);
                let v1 = vminq_s32(vmaxq_s32(v1, lo), hi);
                let h = vcombine_s16(vmovn_s32(v0), vmovn_s32(v1));
                vst1_s8(d, vmovn_s16(h));
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
    if m % 2 == 1 {
        let pair = m / 2;
        for kb in 0..kbs {
            let d = dp.add((pair * kbs + kb) * 16 + 8);
            vst1_s8(d, vdup_n_s8(0));
        }
        *sp.add(m) = 0.0;
    }
}

#[inline(always)]
unsafe fn smmla(acc: int32x4_t, a: int8x16_t, b: int8x16_t) -> int32x4_t {
    let mut r = acc;
    asm!("smmla {0:v}.4s, {1:v}.16b, {2:v}.16b", inout(vreg) r, in(vreg) a, in(vreg) b, options(pure, nomem, nostack));
    r
}

/// `vdotq_s32` as inline asm: the intrinsic is unstable before Rust 1.98 (`stdarch_neon_dotprod`), and the XERJ engine builds
/// with 1.97.1. Same instruction (SDOT Vd.4S, Vn.16B, Vm.16B), same result; callers run under `target_feature = "dotprod"`.
#[inline(always)]
unsafe fn sdot(acc: int32x4_t, a: int8x16_t, b: int8x16_t) -> int32x4_t {
    let mut r = acc;
    asm!("sdot {0:v}.4s, {1:v}.16b, {2:v}.16b", inout(vreg) r, in(vreg) a, in(vreg) b, options(pure, nomem, nostack));
    r
}

/// store 4 int32 accumulators of one row (columns c..c+4) through the epilogue
#[inline(always)]
unsafe fn store_q4(
    y: *mut f32,
    acc: int32x4_t,
    sa: f32,
    sw: *const f32,
    bias: Option<*const f32>,
    epi: Epi,
    nc: usize,
) {
    let mut v = vmulq_f32(vcvtq_f32_s32(acc), vmulq_n_f32(vld1q_f32(sw), sa));
    if nc >= 4 {
        if let Some(b) = bias {
            v = vaddq_f32(v, vld1q_f32(b));
        }
        v = vmulq_n_f32(v, epi.alpha);
        if epi.accumulate {
            v = vaddq_f32(vld1q_f32(y), v);
        }
        vst1q_f32(y, v);
    } else if nc > 0 {
        let mut t = [0f32; 4];
        vst1q_f32(t.as_mut_ptr(), v);
        let ys = std::slice::from_raw_parts_mut(y, nc);
        let bs = bias.map(|b| std::slice::from_raw_parts(b, nc));
        super::epi_store(ys, &t[..nc], bs, epi);
    }
}

#[inline(always)]
unsafe fn tile_smmla<const P: usize>(
    a: *const i8,
    astride: usize,
    b: *const i8,
    kbs: usize,
) -> [[int32x4_t; 4]; P] {
    let z = vdupq_n_s32(0);
    let mut acc = [[z; 4]; P];
    for kb in 0..kbs {
        let bk = b.add(kb * QN * QK);
        let b0 = vld1q_s8(bk);
        let b1 = vld1q_s8(bk.add(16));
        let b2 = vld1q_s8(bk.add(32));
        let b3 = vld1q_s8(bk.add(48));
        for i in 0..P {
            let ai = vld1q_s8(a.add(i * astride + kb * 16));
            acc[i][0] = smmla(acc[i][0], ai, b0);
            acc[i][1] = smmla(acc[i][1], ai, b1);
            acc[i][2] = smmla(acc[i][2], ai, b2);
            acc[i][3] = smmla(acc[i][3], ai, b3);
        }
    }
    acc
}

#[inline(always)]
unsafe fn store_smmla_tile<const P: usize>(
    acc: &[[int32x4_t; 4]; P],
    r0: usize,
    m: usize,
    c0: usize,
    nc: usize,
    q: &QActs,
    w: &PackedI8,
    y: *mut f32,
    ldy: usize,
    epi: Epi,
) {
    let sa = q.scales.as_slice();
    let sw = w.scales.as_ptr().add(c0);
    let bias = w.bias.as_ref().map(|b| b.as_ptr().add(c0));
    let bias_hi = bias.map(|b| b.add(4));
    for i in 0..P {
        let r = r0 + 2 * i;
        let a = &acc[i];
        let s64 = |v: int32x4_t| vreinterpretq_s64_s32(v);
        let lo0 = vreinterpretq_s32_s64(vzip1q_s64(s64(a[0]), s64(a[1])));
        let hi0 = vreinterpretq_s32_s64(vzip1q_s64(s64(a[2]), s64(a[3])));
        let lo1 = vreinterpretq_s32_s64(vzip2q_s64(s64(a[0]), s64(a[1])));
        let hi1 = vreinterpretq_s32_s64(vzip2q_s64(s64(a[2]), s64(a[3])));
        if r < m {
            let yr = y.add(r * ldy + c0);
            store_q4(yr, lo0, sa[r], sw, bias, epi, nc.min(4));
            store_q4(
                yr.add(4),
                hi0,
                sa[r],
                sw.add(4),
                bias_hi,
                epi,
                nc.saturating_sub(4),
            );
        }
        if r + 1 < m {
            let yr = y.add((r + 1) * ldy + c0);
            store_q4(yr, lo1, sa[r + 1], sw, bias, epi, nc.min(4));
            store_q4(
                yr.add(4),
                hi1,
                sa[r + 1],
                sw.add(4),
                bias_hi,
                epi,
                nc.saturating_sub(4),
            );
        }
    }
}

/// int8 GEMM with SMMLA (FEAT_I8MM): 8×8 tiles (4 row pairs × 4 channel pairs)
#[target_feature(enable = "neon,i8mm")]
pub unsafe fn gemm_i8_smmla(q: &QActs, w: &PackedI8, y: &mut [f32], ldy: usize, epi: Epi) {
    gemm_i8_smmla_ptr(q, w, y.as_mut_ptr(), ldy, epi)
}

/// `gemm_i8_smmla` with a raw output pointer
#[target_feature(enable = "neon,i8mm")]
pub unsafe fn gemm_i8_smmla_ptr(q: &QActs, w: &PackedI8, yp: *mut f32, ldy: usize, epi: Epi) {
    let kbs = w.kp / QK;
    let m = q.m;
    let pairs = m.div_ceil(2);
    let astride = kbs * 16;
    let ad = q.data.as_slice().as_ptr();
    let wd = w.data.as_slice().as_ptr();
    // row blocks of up to 64 rows (32 pairs: 16-32 KB of A stays in L1) x all column blocks
    let rb_pairs = 32;
    let mut pb0 = 0;
    while pb0 < pairs {
        let pb1 = (pb0 + rb_pairs).min(pairs);
        for nb in 0..w.n.div_ceil(QN) {
            let c0 = nb * QN;
            let nc = (w.n - c0).min(QN);
            let b = wd.add(nb * kbs * QN * QK);
            let mut p = pb0;
            while p + 4 <= pb1 {
                let acc = tile_smmla::<4>(ad.add(p * astride), astride, b, kbs);
                store_smmla_tile::<4>(&acc, 2 * p, m, c0, nc, q, w, yp, ldy, epi);
                p += 4;
            }
            smmla_tail(
                pb1 - p,
                ad.add(p * astride),
                astride,
                b,
                kbs,
                2 * p,
                m,
                c0,
                nc,
                q,
                w,
                yp,
                ldy,
                epi,
            );
        }
        pb0 = pb1;
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn smmla_tail(
    left: usize,
    a: *const i8,
    astride: usize,
    b: *const i8,
    kbs: usize,
    r0: usize,
    m: usize,
    c0: usize,
    nc: usize,
    q: &QActs,
    w: &PackedI8,
    yp: *mut f32,
    ldy: usize,
    epi: Epi,
) {
    match left {
        3 => {
            let acc = tile_smmla::<3>(a, astride, b, kbs);
            store_smmla_tile::<3>(&acc, r0, m, c0, nc, q, w, yp, ldy, epi);
        }
        2 => {
            let acc = tile_smmla::<2>(a, astride, b, kbs);
            store_smmla_tile::<2>(&acc, r0, m, c0, nc, q, w, yp, ldy, epi);
        }
        1 => {
            let acc = tile_smmla::<1>(a, astride, b, kbs);
            store_smmla_tile::<1>(&acc, r0, m, c0, nc, q, w, yp, ldy, epi);
        }
        _ => {}
    }
}

/// int8 GEMM with SDOT (FEAT_DotProd, no I8MM): 4×8 tiles; each row's 8 k values are duplicated into both halves of the
/// register, so one SDOT per (row, channel pair) yields [c k0-3, c k4-7, c' k0-3, c' k4-7], summed pairwise at the end
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn gemm_i8_sdot(q: &QActs, w: &PackedI8, y: &mut [f32], ldy: usize, epi: Epi) {
    gemm_i8_sdot_ptr(q, w, y.as_mut_ptr(), ldy, epi)
}

/// `gemm_i8_sdot` with a raw output pointer
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn gemm_i8_sdot_ptr(q: &QActs, w: &PackedI8, yp: *mut f32, ldy: usize, epi: Epi) {
    let kbs = w.kp / QK;
    let m = q.m;
    let astride = kbs * 16;
    let ad = q.data.as_slice().as_ptr();
    let wd = w.data.as_slice().as_ptr();
    let sa = q.scales.as_slice();
    for nb in 0..w.n.div_ceil(QN) {
        let c0 = nb * QN;
        let nc = (w.n - c0).min(QN);
        let b = wd.add(nb * kbs * QN * QK);
        let sw = w.scales.as_ptr().add(c0);
        let bias = w.bias.as_ref().map(|bb| bb.as_ptr().add(c0));
        let mut r0 = 0;
        while r0 < m {
            let rows = (m - r0).min(4);
            let z = vdupq_n_s32(0);
            let mut acc = [[z; 4]; 4];
            let p0 = ad.add((r0 / 2) * astride);
            let p1 = ad.add((r0 / 2 + 1).min(m.div_ceil(2) - 1) * astride);
            for kb in 0..kbs {
                let bk = b.add(kb * QN * QK);
                let bv = [
                    vld1q_s8(bk),
                    vld1q_s8(bk.add(16)),
                    vld1q_s8(bk.add(32)),
                    vld1q_s8(bk.add(48)),
                ];
                let a0 = vreinterpretq_s64_s8(vld1q_s8(p0.add(kb * 16)));
                let a1 = vreinterpretq_s64_s8(vld1q_s8(p1.add(kb * 16)));
                let rv = [
                    vreinterpretq_s8_s64(vdupq_laneq_s64::<0>(a0)),
                    vreinterpretq_s8_s64(vdupq_laneq_s64::<1>(a0)),
                    vreinterpretq_s8_s64(vdupq_laneq_s64::<0>(a1)),
                    vreinterpretq_s8_s64(vdupq_laneq_s64::<1>(a1)),
                ];
                for r in 0..4 {
                    for j in 0..4 {
                        acc[r][j] = sdot(acc[r][j], bv[j], rv[r]);
                    }
                }
            }
            for r in 0..rows {
                let lo = vpaddq_s32(acc[r][0], acc[r][1]);
                let hi = vpaddq_s32(acc[r][2], acc[r][3]);
                let yr = yp.add((r0 + r) * ldy + c0);
                store_q4(yr, lo, sa[r0 + r], sw, bias, epi, nc.min(4));
                store_q4(
                    yr.add(4),
                    hi,
                    sa[r0 + r],
                    sw.add(4),
                    bias.map(|bb| bb.add(4)),
                    epi,
                    nc.saturating_sub(4),
                );
            }
            r0 += 4;
        }
    }
}

// ------------------------------------------------------------------------------------------------ attention (head_dim 64)

#[inline(always)]
unsafe fn hsum4(a: float32x4_t, b: float32x4_t, c: float32x4_t, d: float32x4_t) -> float32x4_t {
    vpaddq_f32(vpaddq_f32(a, b), vpaddq_f32(c, d))
}

/// e^x on 4 lanes (same algorithm and constants as ops::exp)
#[inline(always)]
pub unsafe fn exp4(x: float32x4_t) -> float32x4_t {
    let x = vminq_f32(
        vmaxq_f32(x, vdupq_n_f32(-87.336_55)),
        vdupq_n_f32(88.376_26),
    );
    let n = vrndaq_f32(vmulq_f32(x, vdupq_n_f32(std::f32::consts::LOG2_E)));
    let r = vfmaq_f32(x, n, vdupq_n_f32(-0.693_359_4));
    let r = vfmaq_f32(r, n, vdupq_n_f32(2.121_944_4e-4));
    let mut p = vdupq_n_f32(1.987_569_1e-4);
    p = vfmaq_f32(vdupq_n_f32(1.398_199_9e-3), p, r);
    p = vfmaq_f32(vdupq_n_f32(8.333_452e-3), p, r);
    p = vfmaq_f32(vdupq_n_f32(4.166_579_6e-2), p, r);
    p = vfmaq_f32(vdupq_n_f32(1.666_666_5e-1), p, r);
    p = vfmaq_f32(vdupq_n_f32(5.000_000_1e-1), p, r);
    let r2 = vmulq_f32(r, r);
    let y = vaddq_f32(vfmaq_f32(r, p, r2), vdupq_n_f32(1.0));
    let e = vshlq_n_s32::<23>(vaddq_s32(vcvtq_s32_f32(n), vdupq_n_s32(127)));
    vmulq_f32(y, vreinterpretq_f32_s32(e))
}

/// Softmax attention of up to 4 query rows (one head, head_dim 64) over keys [lo, hi] of a K/V store with row stride
/// `stride` (floats; head offset already applied to `k` / `v`). `lo_r[r]..=hi_r[r]` is query r's own visible range inside
/// [lo, hi]. `scores` needs 4 * (hi - lo + 4) floats. Output rows: `out + r * ostride`.
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn attend_block64(
    q: [*const f32; 4],
    nq: usize,
    k: *const f32,
    v: *const f32,
    stride: usize,
    lo: usize,
    hi: usize,
    lo_r: [usize; 4],
    hi_r: [usize; 4],
    scale: f32,
    scores: *mut f32,
    out: [*mut f32; 4],
) {
    let nk = hi + 1 - lo;
    let ns = nk.div_ceil(4) * 4; // padded row length of the score block
                                 // q rows in registers (16 vectors each would be 64 regs: stream them per 4-dim chunk instead)
    let qp = [
        q[0],
        q[if nq > 1 { 1 } else { 0 }],
        q[if nq > 2 { 2 } else { 0 }],
        q[if nq > 3 { 3 } else { 0 }],
    ];
    let vs = vdupq_n_f32(scale);
    // ---- scores: 4 queries x 4 keys per block
    let mut j = 0;
    while j < nk {
        let kk = [
            k.add((lo + j) * stride),
            k.add((lo + (j + 1).min(nk - 1)) * stride),
            k.add((lo + (j + 2).min(nk - 1)) * stride),
            k.add((lo + (j + 3).min(nk - 1)) * stride),
        ];
        let z = vdupq_n_f32(0.0);
        let mut acc = [[z; 4]; 4];
        for c in 0..16 {
            let kv = [
                vld1q_f32(kk[0].add(4 * c)),
                vld1q_f32(kk[1].add(4 * c)),
                vld1q_f32(kk[2].add(4 * c)),
                vld1q_f32(kk[3].add(4 * c)),
            ];
            for r in 0..4 {
                let qv = vld1q_f32(qp[r].add(4 * c));
                acc[r][0] = vfmaq_f32(acc[r][0], qv, kv[0]);
                acc[r][1] = vfmaq_f32(acc[r][1], qv, kv[1]);
                acc[r][2] = vfmaq_f32(acc[r][2], qv, kv[2]);
                acc[r][3] = vfmaq_f32(acc[r][3], qv, kv[3]);
            }
        }
        for r in 0..4 {
            let s = vmulq_f32(hsum4(acc[r][0], acc[r][1], acc[r][2], acc[r][3]), vs);
            vst1q_f32(scores.add(r * ns + j), s);
        }
        j += 4;
    }
    // ---- softmax per query row over its own range (others -> 0)
    for r in 0..nq {
        let row = scores.add(r * ns);
        let (a, b) = (lo_r[r] - lo, hi_r[r] - lo);
        let mut mx = f32::NEG_INFINITY;
        for t in a..=b {
            mx = mx.max(*row.add(t));
        }
        for t in 0..a {
            *row.add(t) = 0.0;
        }
        let vm = vdupq_n_f32(mx);
        let mut sum = vdupq_n_f32(0.0);
        let mut t = a;
        while t + 4 <= b + 1 {
            let e = exp4(vsubq_f32(vld1q_f32(row.add(t)), vm));
            vst1q_f32(row.add(t), e);
            sum = vaddq_f32(sum, e);
            t += 4;
        }
        let mut s = vaddvq_f32(sum);
        while t <= b {
            let e = super::ops::exp(*row.add(t) - mx);
            *row.add(t) = e;
            s += e;
            t += 1;
        }
        for t in b + 1..ns {
            *row.add(t) = 0.0;
        }
        let inv = 1.0 / s;
        let vi = vdupq_n_f32(inv);
        let mut t = 0;
        while t < ns {
            vst1q_f32(row.add(t), vmulq_f32(vld1q_f32(row.add(t)), vi));
            t += 4;
        }
    }
    for r in nq..4 {
        for t in 0..ns {
            *scores.add(r * ns + t) = 0.0;
        }
    }
    // ---- out = P · V, 16 dims per pass (4 rows x 4 vectors accumulators)
    for c in 0..4 {
        let z = vdupq_n_f32(0.0);
        let mut acc = [[z; 4]; 4];
        for j in 0..nk {
            let vr = v.add((lo + j) * stride + 16 * c);
            let vv = [
                vld1q_f32(vr),
                vld1q_f32(vr.add(4)),
                vld1q_f32(vr.add(8)),
                vld1q_f32(vr.add(12)),
            ];
            for r in 0..4 {
                let p = vdupq_n_f32(*scores.add(r * ns + j));
                acc[r][0] = vfmaq_f32(acc[r][0], p, vv[0]);
                acc[r][1] = vfmaq_f32(acc[r][1], p, vv[1]);
                acc[r][2] = vfmaq_f32(acc[r][2], p, vv[2]);
                acc[r][3] = vfmaq_f32(acc[r][3], p, vv[3]);
            }
        }
        for r in 0..nq {
            let o = out[r].add(16 * c);
            vst1q_f32(o, acc[r][0]);
            vst1q_f32(o.add(4), acc[r][1]);
            vst1q_f32(o.add(8), acc[r][2]);
            vst1q_f32(o.add(12), acc[r][3]);
        }
    }
}

// ------------------------------------------------------------------------------------------------ element-wise

/// erf on 4 lanes: the same rational approximation and operation order as ops::erf (bit-identical results; a
/// reciprocal-estimate + Newton replacement of the division measured no faster on Cortex-X925)
#[inline(always)]
pub unsafe fn erf4(a: float32x4_t) -> float32x4_t {
    let x = vminq_f32(vmaxq_f32(a, vdupq_n_f32(-4.0)), vdupq_n_f32(4.0));
    let x2 = vmulq_f32(x, x);
    let mut p = vfmaq_f32(
        vdupq_n_f32(2.770_681_4e-8),
        x2,
        vdupq_n_f32(-2.726_142_3e-10),
    );
    p = vfmaq_f32(vdupq_n_f32(-2.101_024e-6), x2, p);
    p = vfmaq_f32(vdupq_n_f32(-5.692_506_4e-5), x2, p);
    p = vfmaq_f32(vdupq_n_f32(-7.349_906_3e-4), x2, p);
    p = vfmaq_f32(vdupq_n_f32(-2.954_600e-3), x2, p);
    p = vfmaq_f32(vdupq_n_f32(-1.609_603_3e-2), x2, p);
    p = vmulq_f32(p, x);
    let mut q = vfmaq_f32(
        vdupq_n_f32(-2.133_740_6e-4),
        x2,
        vdupq_n_f32(-1.456_607_2e-5),
    );
    q = vfmaq_f32(vdupq_n_f32(-1.682_827e-3), x2, q);
    q = vfmaq_f32(vdupq_n_f32(-7.373_329e-3), x2, q);
    q = vfmaq_f32(vdupq_n_f32(-1.426_474e-2), x2, q);
    vdivq_f32(p, q)
}

/// exact GELU on 4 lanes: x * 0.5 * (1 + erf(x / sqrt 2)), ops::gelu's order
#[inline(always)]
pub unsafe fn gelu4(x: float32x4_t) -> float32x4_t {
    let e = erf4(vmulq_f32(x, vdupq_n_f32(std::f32::consts::FRAC_1_SQRT_2)));
    vmulq_f32(
        vmulq_f32(x, vdupq_n_f32(0.5)),
        vaddq_f32(vdupq_n_f32(1.0), e),
    )
}

#[target_feature(enable = "neon")]
pub unsafe fn gelu_inplace(x: &mut [f32]) {
    let n = x.len();
    let p = x.as_mut_ptr();
    let mut i = 0;
    while i + 4 <= n {
        vst1q_f32(p.add(i), gelu4(vld1q_f32(p.add(i))));
        i += 4;
    }
    while i < n {
        *p.add(i) = super::ops::gelu(*p.add(i));
        i += 1;
    }
}

/// g[r, j] = gelu(u[r, j]) * u[r, h + j]
#[target_feature(enable = "neon")]
pub unsafe fn geglu(u: &[f32], m: usize, h: usize, g: &mut [f32]) {
    for r in 0..m {
        let a = u.as_ptr().add(r * 2 * h);
        let b = a.add(h);
        let o = g.as_mut_ptr().add(r * h);
        let mut j = 0;
        while j + 4 <= h {
            vst1q_f32(
                o.add(j),
                vmulq_f32(gelu4(vld1q_f32(a.add(j))), vld1q_f32(b.add(j))),
            );
            j += 4;
        }
        while j < h {
            *o.add(j) = super::ops::gelu(*a.add(j)) * *b.add(j);
            j += 1;
        }
    }
}

/// RoPE (rotate-half) on one head of 64 lanes; products and sums rounded separately as in PyTorch
#[target_feature(enable = "neon")]
pub unsafe fn rope_head64(x: *mut f32, cos: *const f32, sin: *const f32) {
    for i in (0..32).step_by(4) {
        let a = vld1q_f32(x.add(i));
        let b = vld1q_f32(x.add(32 + i));
        let c = vld1q_f32(cos.add(i));
        let s = vld1q_f32(sin.add(i));
        vst1q_f32(x.add(i), vsubq_f32(vmulq_f32(a, c), vmulq_f32(b, s)));
        vst1q_f32(x.add(32 + i), vaddq_f32(vmulq_f32(b, c), vmulq_f32(a, s)));
    }
}

/// Softmax attention of up to 4 query rows (one head, head_dim 64) over keys [lo, hi], keys stored TRANSPOSED:
/// `kt[d * kts + j]` = key j, dim d (this head), `kts` >= hi + 16 (slack read, masked); values row-major `v[j * vs + d]`.
/// `lo_r[r]..=hi_r[r]` = query r's visible keys. `scores` needs 4 * (hi - lo + 20) floats.
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn attend_block64_t(
    q: [*const f32; 4],
    nq: usize,
    kt: *const f32,
    kts: usize,
    v: *const f32,
    vs: usize,
    lo: usize,
    hi: usize,
    lo_r: [usize; 4],
    hi_r: [usize; 4],
    scale: f32,
    scores: *mut f32,
    out: [*mut f32; 4],
) {
    let nk = hi + 1 - lo;
    let ns = nk.div_ceil(16) * 16 + 4;
    let qp = [
        q[0],
        q[if nq > 1 { 1 } else { 0 }],
        q[if nq > 2 { 2 } else { 0 }],
        q[if nq > 3 { 3 } else { 0 }],
    ];
    let vsc = vdupq_n_f32(scale);
    // ---- scores: 4 queries x 16 keys per block, by-element FMLA over d (no horizontal reductions)
    let mut j = 0;
    while j < nk {
        let z = vdupq_n_f32(0.0);
        let mut acc = [[z; 4]; 4];
        let kb = kt.add(lo + j);
        for d4 in 0..16 {
            let qv = [
                vld1q_f32(qp[0].add(4 * d4)),
                vld1q_f32(qp[1].add(4 * d4)),
                vld1q_f32(qp[2].add(4 * d4)),
                vld1q_f32(qp[3].add(4 * d4)),
            ];
            macro_rules! step {
                ($l:expr) => {{
                    let kr = kb.add((4 * d4 + $l) * kts);
                    let k0 = vld1q_f32(kr);
                    let k1 = vld1q_f32(kr.add(4));
                    let k2 = vld1q_f32(kr.add(8));
                    let k3 = vld1q_f32(kr.add(12));
                    for r in 0..4 {
                        acc[r][0] = vfmaq_laneq_f32::<$l>(acc[r][0], k0, qv[r]);
                        acc[r][1] = vfmaq_laneq_f32::<$l>(acc[r][1], k1, qv[r]);
                        acc[r][2] = vfmaq_laneq_f32::<$l>(acc[r][2], k2, qv[r]);
                        acc[r][3] = vfmaq_laneq_f32::<$l>(acc[r][3], k3, qv[r]);
                    }
                }};
            }
            step!(0);
            step!(1);
            step!(2);
            step!(3);
        }
        for r in 0..4 {
            let row = scores.add(r * ns + j);
            for c in 0..4 {
                vst1q_f32(row.add(4 * c), vmulq_f32(acc[r][c], vsc));
            }
        }
        j += 16;
    }
    // ---- softmax per query row over its own range (others -> 0)
    let mut inv = [0f32; 4];
    for r in 0..nq {
        let row = scores.add(r * ns);
        let (a, b) = (lo_r[r] - lo, hi_r[r] - lo);
        let mut vm = vdupq_n_f32(f32::NEG_INFINITY);
        let mut t = a;
        while t + 4 <= b + 1 {
            vm = vmaxq_f32(vm, vld1q_f32(row.add(t)));
            t += 4;
        }
        let mut mx = vmaxvq_f32(vm);
        while t <= b {
            mx = mx.max(*row.add(t));
            t += 1;
        }
        for t in 0..a {
            *row.add(t) = 0.0;
        }
        let vmx = vdupq_n_f32(mx);
        let mut sum = vdupq_n_f32(0.0);
        let mut t = a;
        while t + 4 <= b + 1 {
            let e = exp4(vsubq_f32(vld1q_f32(row.add(t)), vmx));
            vst1q_f32(row.add(t), e);
            sum = vaddq_f32(sum, e);
            t += 4;
        }
        let mut s = vaddvq_f32(sum);
        while t <= b {
            let e = super::ops::exp(*row.add(t) - mx);
            *row.add(t) = e;
            s += e;
            t += 1;
        }
        for t in b + 1..ns {
            *row.add(t) = 0.0;
        }
        inv[r] = 1.0 / s;
    }
    for r in nq..4 {
        for t in 0..ns {
            *scores.add(r * ns + t) = 0.0;
        }
    }
    // ---- normalize and transpose: pt[j][r] = P[r][j] (one 4-lane vector per key for the P·V loop)
    let pt = scores.add(4 * ns);
    let vi = [
        vdupq_n_f32(inv[0]),
        vdupq_n_f32(inv[1]),
        vdupq_n_f32(inv[2]),
        vdupq_n_f32(inv[3]),
    ];
    let mut t = 0;
    while t < ns {
        let a0 = vmulq_f32(vld1q_f32(scores.add(t)), vi[0]);
        let a1 = vmulq_f32(vld1q_f32(scores.add(ns + t)), vi[1]);
        let a2 = vmulq_f32(vld1q_f32(scores.add(2 * ns + t)), vi[2]);
        let a3 = vmulq_f32(vld1q_f32(scores.add(3 * ns + t)), vi[3]);
        let t0 = vreinterpretq_f64_f32(vtrn1q_f32(a0, a1));
        let t1 = vreinterpretq_f64_f32(vtrn2q_f32(a0, a1));
        let t2 = vreinterpretq_f64_f32(vtrn1q_f32(a2, a3));
        let t3 = vreinterpretq_f64_f32(vtrn2q_f32(a2, a3));
        vst1q_f32(pt.add(4 * t), vreinterpretq_f32_f64(vtrn1q_f64(t0, t2)));
        vst1q_f32(pt.add(4 * t + 4), vreinterpretq_f32_f64(vtrn1q_f64(t1, t3)));
        vst1q_f32(pt.add(4 * t + 8), vreinterpretq_f32_f64(vtrn2q_f64(t0, t2)));
        vst1q_f32(
            pt.add(4 * t + 12),
            vreinterpretq_f32_f64(vtrn2q_f64(t1, t3)),
        );
        t += 4;
    }
    // ---- out = P · V: per key, one 4-lane vector of the 4 rows' probabilities; 16 dims per pass
    for c in 0..4 {
        let z = vdupq_n_f32(0.0);
        let mut acc = [[z; 4]; 4];
        for j in 0..nk {
            let pv = vld1q_f32(pt.add(4 * j));
            let vr = v.add((lo + j) * vs + 16 * c);
            let vv = [
                vld1q_f32(vr),
                vld1q_f32(vr.add(4)),
                vld1q_f32(vr.add(8)),
                vld1q_f32(vr.add(12)),
            ];
            macro_rules! row {
                ($r:expr) => {
                    acc[$r][0] = vfmaq_laneq_f32::<$r>(acc[$r][0], vv[0], pv);
                    acc[$r][1] = vfmaq_laneq_f32::<$r>(acc[$r][1], vv[1], pv);
                    acc[$r][2] = vfmaq_laneq_f32::<$r>(acc[$r][2], vv[2], pv);
                    acc[$r][3] = vfmaq_laneq_f32::<$r>(acc[$r][3], vv[3], pv);
                };
            }
            row!(0);
            row!(1);
            row!(2);
            row!(3);
        }
        for r in 0..nq {
            let o = out[r].add(16 * c);
            vst1q_f32(o, acc[r][0]);
            vst1q_f32(o.add(4), acc[r][1]);
            vst1q_f32(o.add(8), acc[r][2]);
            vst1q_f32(o.add(12), acc[r][3]);
        }
    }
}

/// fused LayerNorm + per-row int8 quantization: row r of the output is LN(x_r) * w (+ b), times `pre`, quantized into the
/// row-pair layout without materializing the LayerNorm output
#[target_feature(enable = "neon")]
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
    let kbs = q.kp / QK;
    let dp = q.data.as_mut_slice().as_mut_ptr();
    let sp = q.scales.as_mut_slice().as_mut_ptr();
    let wp = w.as_ptr();
    let bp = b.map(|b| b.as_ptr());
    let pp = pre.map(|p| p.as_ptr());
    debug_assert!(k % 8 == 0);
    for r in 0..m {
        let row = x.as_ptr().add(r * k);
        // stats (two-pass, as ops::layer_norm_row)
        let mut s0 = vdupq_n_f32(0.0);
        let mut s1 = vdupq_n_f32(0.0);
        let mut i = 0;
        while i + 8 <= k {
            s0 = vaddq_f32(s0, vld1q_f32(row.add(i)));
            s1 = vaddq_f32(s1, vld1q_f32(row.add(i + 4)));
            i += 8;
        }
        let mean = vaddvq_f32(vaddq_f32(s0, s1)) / k as f32;
        let vmean = vdupq_n_f32(mean);
        let mut q0 = vdupq_n_f32(0.0);
        let mut q1 = vdupq_n_f32(0.0);
        let mut i = 0;
        while i + 8 <= k {
            let d0 = vsubq_f32(vld1q_f32(row.add(i)), vmean);
            let d1 = vsubq_f32(vld1q_f32(row.add(i + 4)), vmean);
            q0 = vfmaq_f32(q0, d0, d0);
            q1 = vfmaq_f32(q1, d1, d1);
            i += 8;
        }
        let var = vaddvq_f32(vaddq_f32(q0, q1)) / k as f32;
        let rstd = vdupq_n_f32(1.0 / (var.max(0.0) + eps).sqrt());
        let y = |i: usize| -> float32x4_t {
            let mut v = vmulq_f32(
                vmulq_f32(vsubq_f32(vld1q_f32(row.add(i)), vmean), rstd),
                vld1q_f32(wp.add(i)),
            );
            if let Some(bb) = bp {
                v = vaddq_f32(v, vld1q_f32(bb.add(i)));
            }
            if let Some(p) = pp {
                v = vmulq_f32(v, vld1q_f32(p.add(i)));
            }
            v
        };
        let mut mx = vdupq_n_f32(0.0);
        let mut i = 0;
        while i + 4 <= k {
            mx = vmaxq_f32(mx, vabsq_f32(y(i)));
            i += 4;
        }
        let amax = vmaxvq_f32(mx);
        let (s, inv) = if amax > 0.0 {
            (amax / 127.0, 127.0 / amax)
        } else {
            (0.0, 0.0)
        };
        *sp.add(r) = s;
        let (pair, half) = (r / 2, r % 2);
        let vinv = vdupq_n_f32(inv);
        let lo = vdupq_n_s32(-127);
        let hi = vdupq_n_s32(127);
        for kb in 0..kbs {
            let d = dp.add((pair * kbs + kb) * 16 + half * 8);
            let i0 = kb * QK;
            let v0 = vminq_s32(vmaxq_s32(vcvtaq_s32_f32(vmulq_f32(y(i0), vinv)), lo), hi);
            let v1 = vminq_s32(
                vmaxq_s32(vcvtaq_s32_f32(vmulq_f32(y(i0 + 4), vinv)), lo),
                hi,
            );
            vst1_s8(d, vmovn_s16(vcombine_s16(vmovn_s32(v0), vmovn_s32(v1))));
        }
    }
    if m % 2 == 1 {
        let pair = m / 2;
        for kb in 0..kbs {
            vst1_s8(dp.add((pair * kbs + kb) * 16 + 8), vdup_n_s8(0));
        }
        *sp.add(m) = 0.0;
    }
}

// ------------------------------------------------------------------------------------------------ bf16 attention (int8 path)

/// f32 -> bf16, round to nearest even
#[inline(always)]
pub fn bf16(x: f32) -> u16 {
    let b = x.to_bits();
    if x.is_nan() {
        return ((b >> 16) | 0x40) as u16;
    }
    ((b + 0x7fff + ((b >> 16) & 1)) >> 16) as u16
}

#[inline(always)]
unsafe fn bfmmla(acc: float32x4_t, a: uint16x8_t, b: uint16x8_t) -> float32x4_t {
    let mut r = acc;
    asm!("bfmmla {0:v}.4s, {1:v}.8h, {2:v}.8h", inout(vreg) r, in(vreg) a, in(vreg) b, options(pure, nomem, nostack));
    r
}

/// 4 lanes f32 -> 4 bf16 (round to nearest even), as a u16x4
#[inline(always)]
unsafe fn bf16x4(v: float32x4_t) -> uint16x4_t {
    let b = vreinterpretq_u32_f32(v);
    let lsb = vandq_u32(vshrq_n_u32::<16>(b), vdupq_n_u32(1));
    let r = vaddq_u32(vaddq_u32(b, vdupq_n_u32(0x7fff)), lsb);
    vshrn_n_u32::<16>(r)
}

/// Softmax attention of up to 4 query rows (one head, head_dim 64) with BFMMLA (bf16 inputs, f32 accumulation).
/// Keys in pair layout `kb[(pair * 16 + chunk) * 8 + half * 4 + i]` (pair = position / 2), values in transposed pair layout
/// `vb[(chunk4 * 32 + dimpair) * 8 + half * 4 + lane]` (chunk4 = position / 4); both head-local, with slack beyond `hi`.
/// `scores` needs 4 * (hi - lo + 24) floats, `qb` 256 u16, `pb` 4 * (hi - lo + 24) u16.
#[target_feature(enable = "neon,bf16")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn attend_block64_bf16(
    q: [*const f32; 4],
    nq: usize,
    kb: *const u16,
    vb: *const u16,
    lo: usize,
    hi: usize,
    lo_r: [usize; 4],
    hi_r: [usize; 4],
    scale: f32,
    scores: *mut f32,
    qb: *mut u16,
    pb: *mut u16,
    out: [*mut f32; 4],
) {
    let lo4 = lo & !3;
    let nk = hi + 1 - lo4;
    let ns = nk.div_ceil(16) * 16;
    let qp = [
        q[0],
        q[if nq > 1 { 1 } else { 0 }],
        q[if nq > 2 { 2 } else { 0 }],
        q[if nq > 3 { 3 } else { 0 }],
    ];
    // queries -> bf16 pair layout [qpair][chunk][8]
    for p in 0..2 {
        for c in 0..16 {
            let a = bf16x4(vld1q_f32(qp[2 * p].add(4 * c)));
            let b = bf16x4(vld1q_f32(qp[2 * p + 1].add(4 * c)));
            vst1q_u16(qb.add((p * 16 + c) * 8), vcombine_u16(a, b));
        }
    }
    let vs = vdupq_n_f32(scale);
    let s64 = |v: float32x4_t| vreinterpretq_f64_f32(v);
    let f32v = |v: float64x2_t| vreinterpretq_f32_f64(v);
    // scores, 4 queries x 16 keys per block (16 independent accumulators hide the BFMMLA latency)
    let pair0 = lo4 / 2;
    let mut j = 0;
    while j < nk {
        let z = vdupq_n_f32(0.0);
        let mut acc = [[z; 8]; 2];
        let kbase = kb.add((pair0 + j / 2) * 128);
        for c in 0..16 {
            let q0 = vld1q_u16(qb.add(c * 8));
            let q1 = vld1q_u16(qb.add((16 + c) * 8));
            for kp in 0..8 {
                let kv = vld1q_u16(kbase.add((kp * 16 + c) * 8));
                acc[0][kp] = bfmmla(acc[0][kp], q0, kv);
                acc[1][kp] = bfmmla(acc[1][kp], q1, kv);
            }
        }
        for p in 0..2 {
            let a = &acc[p];
            let r0 = scores.add((2 * p) * ns + j);
            let r1 = scores.add((2 * p + 1) * ns + j);
            for h in 0..4 {
                vst1q_f32(
                    r0.add(4 * h),
                    vmulq_f32(f32v(vzip1q_f64(s64(a[2 * h]), s64(a[2 * h + 1]))), vs),
                );
                vst1q_f32(
                    r1.add(4 * h),
                    vmulq_f32(f32v(vzip2q_f64(s64(a[2 * h]), s64(a[2 * h + 1]))), vs),
                );
            }
        }
        j += 16;
    }
    // softmax per row over its own range (others -> 0), probabilities -> bf16 pair layout
    let mut inv = [0f32; 4];
    for r in 0..4 {
        let row = scores.add(r * ns);
        if r >= nq {
            for t in 0..ns {
                *row.add(t) = 0.0;
            }
            continue;
        }
        let (a, b) = (lo_r[r] - lo4, hi_r[r] - lo4);
        let mut vmx = vdupq_n_f32(f32::NEG_INFINITY);
        let mut t = a;
        while t + 4 <= b + 1 {
            vmx = vmaxq_f32(vmx, vld1q_f32(row.add(t)));
            t += 4;
        }
        let mut mx = vmaxvq_f32(vmx);
        while t <= b {
            mx = mx.max(*row.add(t));
            t += 1;
        }
        for t in 0..a {
            *row.add(t) = 0.0;
        }
        let vm = vdupq_n_f32(mx);
        let mut sum = vdupq_n_f32(0.0);
        let mut t = a;
        while t + 4 <= b + 1 {
            let e = exp4(vsubq_f32(vld1q_f32(row.add(t)), vm));
            vst1q_f32(row.add(t), e);
            sum = vaddq_f32(sum, e);
            t += 4;
        }
        let mut s = vaddvq_f32(sum);
        while t <= b {
            let e = super::ops::exp(*row.add(t) - mx);
            *row.add(t) = e;
            s += e;
            t += 1;
        }
        for t in b + 1..ns {
            *row.add(t) = 0.0;
        }
        inv[r] = 1.0 / s;
    }
    let nch = ns / 4;
    for p in 0..2 {
        let (va, vbb) = (vdupq_n_f32(inv[2 * p]), vdupq_n_f32(inv[2 * p + 1]));
        let (ra, rb) = (scores.add(2 * p * ns), scores.add((2 * p + 1) * ns));
        for c in 0..nch {
            let a = bf16x4(vmulq_f32(vld1q_f32(ra.add(4 * c)), va));
            let b = bf16x4(vmulq_f32(vld1q_f32(rb.add(4 * c)), vbb));
            vst1q_u16(pb.add((p * nch + c) * 8), vcombine_u16(a, b));
        }
    }
    // out = P · V: 16 dims (8 dim pairs) per pass
    let c0 = lo4 / 4;
    let used = nk.div_ceil(4);
    for pass in 0..4 {
        let z = vdupq_n_f32(0.0);
        let mut acc = [[z; 8]; 2];
        for c in 0..used {
            let p0 = vld1q_u16(pb.add(c * 8));
            let p1 = vld1q_u16(pb.add((nch + c) * 8));
            let vbase = vb.add(((c0 + c) * 32 + pass * 8) * 8);
            for dp in 0..8 {
                let vv = vld1q_u16(vbase.add(dp * 8));
                acc[0][dp] = bfmmla(acc[0][dp], p0, vv);
                acc[1][dp] = bfmmla(acc[1][dp], p1, vv);
            }
        }
        for p in 0..2 {
            for dp in (0..8).step_by(2) {
                let lo_row = f32v(vzip1q_f64(s64(acc[p][dp]), s64(acc[p][dp + 1])));
                let hi_row = f32v(vzip2q_f64(s64(acc[p][dp]), s64(acc[p][dp + 1])));
                if 2 * p < nq {
                    vst1q_f32(out[2 * p].add(pass * 16 + 2 * dp), lo_row);
                }
                if 2 * p + 1 < nq {
                    vst1q_f32(out[2 * p + 1].add(pass * 16 + 2 * dp), hi_row);
                }
            }
        }
    }
}

/// bf16 cache writes for rows p0..p1 (head_dim 64): keys into the pair layout, values into the transposed pair layout
/// (4x4 transposes per 4-position chunk; partial chunks keep the lanes of other positions)
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn put_bf16_rows64(
    k: *const f32,
    v: *const f32,
    d: usize,
    heads: usize,
    p0: usize,
    p1: usize,
    kb: *mut u16,
    pc: usize,
    vb: *mut u16,
    cc: usize,
) {
    for p in p0..p1 {
        let key = k.add(p * d);
        for h in 0..heads {
            let base = kb.add((h * pc + p / 2) * 128 + (p % 2) * 4);
            for c in 0..16 {
                vst1_u16(base.add(c * 8), bf16x4(vld1q_f32(key.add(h * 64 + 4 * c))));
            }
        }
    }
    let mut j = p0 / 4;
    while 4 * j < p1 {
        let full = 4 * j >= p0 && 4 * j + 4 <= p1;
        for h in 0..heads {
            let base = vb.add((h * cc + j) * 256);
            if full {
                let r = [
                    v.add((4 * j) * d + h * 64),
                    v.add((4 * j + 1) * d + h * 64),
                    v.add((4 * j + 2) * d + h * 64),
                    v.add((4 * j + 3) * d + h * 64),
                ];
                for g in 0..16 {
                    // rows = positions, columns = dims 4g..4g+4  ->  columns of positions per dim
                    let a0 = vld1q_f32(r[0].add(4 * g));
                    let a1 = vld1q_f32(r[1].add(4 * g));
                    let a2 = vld1q_f32(r[2].add(4 * g));
                    let a3 = vld1q_f32(r[3].add(4 * g));
                    let t0 = vreinterpretq_f64_f32(vtrn1q_f32(a0, a1));
                    let t1 = vreinterpretq_f64_f32(vtrn2q_f32(a0, a1));
                    let t2 = vreinterpretq_f64_f32(vtrn1q_f32(a2, a3));
                    let t3 = vreinterpretq_f64_f32(vtrn2q_f32(a2, a3));
                    let d0 = bf16x4(vreinterpretq_f32_f64(vtrn1q_f64(t0, t2)));
                    let d1 = bf16x4(vreinterpretq_f32_f64(vtrn1q_f64(t1, t3)));
                    let d2 = bf16x4(vreinterpretq_f32_f64(vtrn2q_f64(t0, t2)));
                    let d3 = bf16x4(vreinterpretq_f32_f64(vtrn2q_f64(t1, t3)));
                    // dim pair (4g, 4g+1) and (4g+2, 4g+3)
                    vst1q_u16(base.add((2 * g) * 8), vcombine_u16(d0, d1));
                    vst1q_u16(base.add((2 * g + 1) * 8), vcombine_u16(d2, d3));
                }
            } else {
                for l in 0..4 {
                    let p = 4 * j + l;
                    if p < p0 || p >= p1 {
                        continue;
                    }
                    let row = v.add(p * d + h * 64);
                    for q in 0..32 {
                        *base.add(q * 8 + l) = bf16(*row.add(2 * q));
                        *base.add(q * 8 + 4 + l) = bf16(*row.add(2 * q + 1));
                    }
                }
            }
        }
        j += 1;
    }
}
