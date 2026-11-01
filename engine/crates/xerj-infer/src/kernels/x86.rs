//! x86-64 kernels for customer servers: f32 AVX2+FMA 6×16 and AVX-512F 12×16; int8 AVX2 4×8 (VPMADDUBSW with the sign trick:
//! |a| · (w·sign(a)), exact because both operands are in [-127, 127]) and AVX-512 VNNI 8×8 (VPDPBUSD on a + 128, corrected
//! with per-channel weight sums). Same packed layouts as the aarch64 kernels (see kernels/mod.rs).
//! Tested with qemu-user where the emulator supports the ISA (see README); not benchmarked on x86 hardware.

use super::{AVec, Epi, PackedF32, PackedI8, QActs, NR, QK, QN};
use std::arch::x86_64::*;
use std::sync::OnceLock;

static R2_GEMM: OnceLock<bool> = OnceLock::new();

/// the round-2 AVX2 GEMM kernels (`gemm_f32_avx2_v2_ptr`, `gemm_i8_avx2_v2_ptr`) are in use: the default; `XERJ_X86_R1=gemm`
/// (or `all`) keeps the round-1 kernels for A/B checks. Both give bit-identical results.
pub fn r2_gemm() -> bool {
    *R2_GEMM.get_or_init(|| !super::x86ops::r1_kept("gemm"))
}

#[inline(always)]
unsafe fn pack_a(x: &[f32], m: usize, k: usize, r0: usize, mr: usize, dst: *mut f32) {
    let rows = (m - r0).min(mr);
    for kk in 0..k {
        let d = dst.add(kk * mr);
        for r in 0..mr {
            *d.add(r) = if r < rows {
                *x.get_unchecked((r0 + r) * k + kk)
            } else {
                0.0
            };
        }
    }
}

#[inline(always)]
unsafe fn store_spill(y: *mut f32, t: &[f32], bias: Option<*const f32>, epi: Epi, nc: usize) {
    let ys = std::slice::from_raw_parts_mut(y, nc);
    let bs = bias.map(|b| std::slice::from_raw_parts(b, nc));
    super::epi_store(ys, &t[..nc], bs, epi);
}

#[target_feature(enable = "avx2,fma")]
unsafe fn store8_avx2(y: *mut f32, v: __m256, bias: Option<*const f32>, epi: Epi, nc: usize) {
    if nc >= 8 {
        let mut v = v;
        if let Some(b) = bias {
            v = _mm256_add_ps(v, _mm256_loadu_ps(b));
        }
        v = _mm256_mul_ps(v, _mm256_set1_ps(epi.alpha));
        if epi.accumulate {
            v = _mm256_add_ps(_mm256_loadu_ps(y), v);
        }
        _mm256_storeu_ps(y, v);
    } else if nc > 0 {
        let mut t = [0f32; 8];
        _mm256_storeu_ps(t.as_mut_ptr(), v);
        store_spill(y, &t, bias, epi, nc);
    }
}

/// f32 GEMM, AVX2 + FMA, 6×16 tiles
#[target_feature(enable = "avx2,fma")]
pub unsafe fn gemm_f32_avx2(
    x: &[f32],
    m: usize,
    w: &PackedF32,
    y: &mut [f32],
    ldy: usize,
    epi: Epi,
    apack: &mut AVec<f32>,
) {
    gemm_f32_avx2_ptr(x, m, w, y.as_mut_ptr(), ldy, epi, apack)
}

/// `gemm_f32_avx2` with a raw output pointer: row r, column c at `yp + r * ldy + c` (c < w.n)
#[target_feature(enable = "avx2,fma")]
pub unsafe fn gemm_f32_avx2_ptr(
    x: &[f32],
    m: usize,
    w: &PackedF32,
    yp: *mut f32,
    ldy: usize,
    epi: Epi,
    apack: &mut AVec<f32>,
) {
    const MR: usize = 6;
    let k = w.k;
    let rbs = m.div_ceil(MR);
    apack.resize(rbs * k * MR);
    let ap = apack.as_mut_slice().as_mut_ptr();
    for rb in 0..rbs {
        pack_a(x, m, k, rb * MR, MR, ap.add(rb * k * MR));
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
            let mut acc = [[_mm256_setzero_ps(); 2]; MR];
            for kk in 0..k {
                let b0 = _mm256_loadu_ps(b.add(kk * NR));
                let b1 = _mm256_loadu_ps(b.add(kk * NR + 8));
                for r in 0..MR {
                    let av = _mm256_broadcast_ss(&*a.add(kk * MR + r));
                    acc[r][0] = _mm256_fmadd_ps(av, b0, acc[r][0]);
                    acc[r][1] = _mm256_fmadd_ps(av, b1, acc[r][1]);
                }
            }
            for r in 0..rows {
                let yr = yp.add((rb * MR + r) * ldy + c0);
                store8_avx2(yr, acc[r][0], bias, epi, nc.min(8));
                store8_avx2(
                    yr.add(8),
                    acc[r][1],
                    bias.map(|bb| bb.add(8)),
                    epi,
                    nc.saturating_sub(8),
                );
            }
        }
    }
}

/// f32 GEMM, AVX-512F, 12×16 tiles
#[target_feature(enable = "avx512f")]
pub unsafe fn gemm_f32_avx512(
    x: &[f32],
    m: usize,
    w: &PackedF32,
    y: &mut [f32],
    ldy: usize,
    epi: Epi,
    apack: &mut AVec<f32>,
) {
    gemm_f32_avx512_ptr(x, m, w, y.as_mut_ptr(), ldy, epi, apack)
}

/// `gemm_f32_avx512` with a raw output pointer
#[target_feature(enable = "avx512f")]
pub unsafe fn gemm_f32_avx512_ptr(
    x: &[f32],
    m: usize,
    w: &PackedF32,
    yp: *mut f32,
    ldy: usize,
    epi: Epi,
    apack: &mut AVec<f32>,
) {
    const MR: usize = 12;
    let k = w.k;
    let rbs = m.div_ceil(MR);
    apack.resize(rbs * k * MR);
    let ap = apack.as_mut_slice().as_mut_ptr();
    for rb in 0..rbs {
        pack_a(x, m, k, rb * MR, MR, ap.add(rb * k * MR));
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
            let mut acc = [_mm512_setzero_ps(); MR];
            for kk in 0..k {
                let bv = _mm512_loadu_ps(b.add(kk * NR));
                for r in 0..MR {
                    acc[r] = _mm512_fmadd_ps(_mm512_set1_ps(*a.add(kk * MR + r)), bv, acc[r]);
                }
            }
            for r in 0..rows {
                let yr = yp.add((rb * MR + r) * ldy + c0);
                if nc == NR {
                    let mut v = acc[r];
                    if let Some(bb) = bias {
                        v = _mm512_add_ps(v, _mm512_loadu_ps(bb));
                    }
                    v = _mm512_mul_ps(v, _mm512_set1_ps(epi.alpha));
                    if epi.accumulate {
                        v = _mm512_add_ps(_mm512_loadu_ps(yr), v);
                    }
                    _mm512_storeu_ps(yr, v);
                } else {
                    let mut t = [0f32; NR];
                    _mm512_storeu_ps(t.as_mut_ptr(), acc[r]);
                    store_spill(yr, &t, bias, epi, nc);
                }
            }
        }
    }
}

#[target_feature(enable = "avx2,fma")]
unsafe fn finish_i8_avx2(
    y: *mut f32,
    iacc: __m256i,
    sa: f32,
    sw: *const f32,
    bias: Option<*const f32>,
    epi: Epi,
    nc: usize,
) {
    let v = _mm256_mul_ps(
        _mm256_cvtepi32_ps(iacc),
        _mm256_mul_ps(_mm256_loadu_ps(sw), _mm256_set1_ps(sa)),
    );
    store8_avx2(y, v, bias, epi, nc);
}

/// int8 GEMM, AVX2: 4 rows × 8 channels; VPMADDUBSW(|a|, w·sign(a)) then VPMADDWD with ones
#[target_feature(enable = "avx2,fma")]
pub unsafe fn gemm_i8_avx2(q: &QActs, w: &PackedI8, y: &mut [f32], ldy: usize, epi: Epi) {
    gemm_i8_avx2_ptr(q, w, y.as_mut_ptr(), ldy, epi)
}

/// `gemm_i8_avx2` with a raw output pointer
#[target_feature(enable = "avx2,fma")]
pub unsafe fn gemm_i8_avx2_ptr(q: &QActs, w: &PackedI8, yp: *mut f32, ldy: usize, epi: Epi) {
    let kbs = w.kp / QK;
    let m = q.m;
    let astride = kbs * 16;
    let ad = q.data.as_slice().as_ptr();
    let wd = w.data.as_slice().as_ptr();
    let sa = q.scales.as_slice();
    let ones = _mm256_set1_epi16(1);
    let perm = _mm256_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7);
    for nb in 0..w.n.div_ceil(QN) {
        let c0 = nb * QN;
        let nc = (w.n - c0).min(QN);
        let b = wd.add(nb * kbs * QN * QK);
        let sw = w.scales.as_ptr().add(c0);
        let bias = w.bias.as_ref().map(|bb| bb.as_ptr().add(c0));
        let mut r0 = 0;
        while r0 < m {
            let rows = (m - r0).min(4);
            let mut acc = [[_mm256_setzero_si256(); 2]; 4];
            for kb in 0..kbs {
                let b01 = _mm256_loadu_si256(b.add(kb * 64) as *const __m256i);
                let b23 = _mm256_loadu_si256(b.add(kb * 64 + 32) as *const __m256i);
                for r in 0..rows {
                    let rr = r0 + r;
                    let src = ad.add((rr / 2) * astride + kb * 16 + (rr % 2) * 8) as *const i64;
                    let a = _mm256_set1_epi64x(std::ptr::read_unaligned(src));
                    let aa = _mm256_sign_epi8(a, a);
                    let p0 = _mm256_maddubs_epi16(aa, _mm256_sign_epi8(b01, a));
                    let p1 = _mm256_maddubs_epi16(aa, _mm256_sign_epi8(b23, a));
                    acc[r][0] = _mm256_add_epi32(acc[r][0], _mm256_madd_epi16(p0, ones));
                    acc[r][1] = _mm256_add_epi32(acc[r][1], _mm256_madd_epi16(p1, ones));
                }
            }
            for r in 0..rows {
                let h = _mm256_hadd_epi32(acc[r][0], acc[r][1]);
                let c = _mm256_permutevar8x32_epi32(h, perm);
                finish_i8_avx2(
                    yp.add((r0 + r) * ldy + c0),
                    c,
                    sa[r0 + r],
                    sw,
                    bias,
                    epi,
                    nc,
                );
            }
            r0 += 4;
        }
    }
}

/// int8 GEMM, AVX-512 VNNI: 8 rows × 8 channels; VPDPBUSD on (a + 128) as u8, minus 128·Σw per channel
#[target_feature(enable = "avx512f,avx512bw,avx512vnni,avx2,fma")]
pub unsafe fn gemm_i8_avx512vnni(q: &QActs, w: &PackedI8, y: &mut [f32], ldy: usize, epi: Epi) {
    gemm_i8_avx512vnni_ptr(q, w, y.as_mut_ptr(), ldy, epi)
}

/// `gemm_i8_avx512vnni` with a raw output pointer
#[target_feature(enable = "avx512f,avx512bw,avx512vnni,avx2,fma")]
pub unsafe fn gemm_i8_avx512vnni_ptr(q: &QActs, w: &PackedI8, yp: *mut f32, ldy: usize, epi: Epi) {
    let kbs = w.kp / QK;
    let m = q.m;
    let astride = kbs * 16;
    let ad = q.data.as_slice().as_ptr();
    let wd = w.data.as_slice().as_ptr();
    let sa = q.scales.as_slice();
    let flip = _mm512_set1_epi8(-128i8);
    for nb in 0..w.n.div_ceil(QN) {
        let c0 = nb * QN;
        let nc = (w.n - c0).min(QN);
        let b = wd.add(nb * kbs * QN * QK);
        let sw = w.scales.as_ptr().add(c0);
        let bias = w.bias.as_ref().map(|bb| bb.as_ptr().add(c0));
        let corr = _mm256_mullo_epi32(
            _mm256_loadu_si256(w.colsum.as_ptr().add(c0) as *const __m256i),
            _mm256_set1_epi32(128),
        );
        let mut r0 = 0;
        while r0 < m {
            let rows = (m - r0).min(8);
            let mut acc = [_mm512_setzero_si512(); 8];
            for kb in 0..kbs {
                let bv = _mm512_loadu_si512(b.add(kb * 64) as *const _);
                for r in 0..rows {
                    let rr = r0 + r;
                    let src = ad.add((rr / 2) * astride + kb * 16 + (rr % 2) * 8) as *const i64;
                    let a =
                        _mm512_xor_si512(_mm512_set1_epi64(std::ptr::read_unaligned(src)), flip);
                    acc[r] = _mm512_dpbusd_epi32(acc[r], a, bv);
                }
            }
            for r in 0..rows {
                let t = _mm512_add_epi32(acc[r], _mm512_srli_epi64::<32>(acc[r]));
                let c = _mm256_sub_epi32(_mm512_cvtepi64_epi32(t), corr);
                finish_i8_avx2(
                    yp.add((r0 + r) * ldy + c0),
                    c,
                    sa[r0 + r],
                    sw,
                    bias,
                    epi,
                    nc,
                );
            }
            r0 += 8;
        }
    }
}

// ------------------------------------------------------------------------------------------------ round 2

/// L2 bytes of packed f32 panels visited per row block by `gemm_f32_avx2_v2_ptr`
const F32_GROUP_BYTES: usize = 96 * 1024;

/// f32 GEMM, AVX2 + FMA, 6×16 tiles (round 2). Bit-identical to `gemm_f32_avx2_ptr`: every output element is the same
/// FMA chain over k in order from zero, followed by the same epilogue. Differences:
/// * the packed panels are read with aligned 256-bit loads (they are 64-byte aligned). The engine builds x86-64 for
///   `x86-64-v2`, whose tuning makes LLVM split every *unaligned* 256-bit load into two 128-bit halves (vmovups xmm +
///   vinsertf128: three uops instead of one, the insert competing with the FMAs for ports 0 / 1);
/// * the panels are visited in groups of about `F32_GROUP_BYTES` per row block, so a packed A block is reused from L1 /
///   L2 by every panel of the group instead of being streamed again from L3 for each panel.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn gemm_f32_avx2_v2_ptr(
    x: &[f32],
    m: usize,
    w: &PackedF32,
    yp: *mut f32,
    ldy: usize,
    epi: Epi,
    apack: &mut AVec<f32>,
) {
    const MR: usize = 6;
    let k = w.k;
    if m == 0 || k == 0 {
        return gemm_f32_avx2_ptr(x, m, w, yp, ldy, epi, apack);
    }
    let rbs = m.div_ceil(MR);
    apack.resize(rbs * k * MR);
    let ap = apack.as_mut_slice().as_mut_ptr();
    for rb in 0..rbs {
        pack_a(x, m, k, rb * MR, MR, ap.add(rb * k * MR));
    }
    let wd = w.data.as_slice().as_ptr();
    assert!(
        wd as usize % 32 == 0,
        "packed f32 panels must be 32-byte aligned"
    );
    let panels = w.panels();
    let group = (F32_GROUP_BYTES / (k * NR * 4)).clamp(1, panels.max(1));
    let mut g0 = 0;
    while g0 < panels {
        let g1 = (g0 + group).min(panels);
        for rb in 0..rbs {
            let a = ap.add(rb * k * MR) as *const f32;
            let rows = (m - rb * MR).min(MR);
            for p in g0..g1 {
                let c0 = p * NR;
                let nc = (w.n - c0).min(NR);
                let b = wd.add(p * k * NR);
                let bias = w.bias.as_ref().map(|bb| bb.as_ptr().add(c0));
                let mut acc = [[_mm256_setzero_ps(); 2]; MR];
                // one k step: every accumulator gets its FMA for this kk (the chain of each output element stays
                // kk = 0, 1, 2, ... in order); the loop runs two steps per iteration (fewer loop uops: the kernel is
                // close to the front-end limit of 4 uops per cycle)
                let step = |kk: usize, acc: &mut [[__m256; 2]; MR]| {
                    // aligned: panel p starts at p * k * 64 bytes of a 64-byte aligned buffer, row kk at 64 * kk
                    let b0 = _mm256_load_ps(b.add(kk * NR));
                    let b1 = _mm256_load_ps(b.add(kk * NR + 8));
                    for r in 0..MR {
                        let av = _mm256_broadcast_ss(&*a.add(kk * MR + r));
                        acc[r][0] = _mm256_fmadd_ps(av, b0, acc[r][0]);
                        acc[r][1] = _mm256_fmadd_ps(av, b1, acc[r][1]);
                    }
                };
                let mut kk = 0;
                while kk + 2 <= k {
                    step(kk, &mut acc);
                    step(kk + 1, &mut acc);
                    kk += 2;
                }
                if kk < k {
                    step(kk, &mut acc);
                }
                for r in 0..rows {
                    let yr = yp.add((rb * MR + r) * ldy + c0);
                    store8_avx2(yr, acc[r][0], bias, epi, nc.min(8));
                    store8_avx2(
                        yr.add(8),
                        acc[r][1],
                        bias.map(|bb| bb.add(8)),
                        epi,
                        nc.saturating_sub(8),
                    );
                }
            }
        }
        g0 = g1;
    }
}

/// one k block (8 k) of one row of the AVX2 int8 kernel: acc0 / acc1 += the 4 + 4 channels' partial sums (exact)
#[inline(always)]
unsafe fn i8_row_step(
    a8: *const i8,
    b01: __m256i,
    b23: __m256i,
    ones: __m256i,
    acc0: &mut __m256i,
    acc1: &mut __m256i,
) {
    let a = _mm256_set1_epi64x(std::ptr::read_unaligned(a8 as *const i64));
    let aa = _mm256_sign_epi8(a, a);
    let p0 = _mm256_maddubs_epi16(aa, _mm256_sign_epi8(b01, a));
    let p1 = _mm256_maddubs_epi16(aa, _mm256_sign_epi8(b23, a));
    *acc0 = _mm256_add_epi32(*acc0, _mm256_madd_epi16(p0, ones));
    *acc1 = _mm256_add_epi32(*acc1, _mm256_madd_epi16(p1, ones));
}

/// int8 GEMM, AVX2, 4 rows × 8 channels (round 2). The same exact integer dot products as `gemm_i8_avx2_ptr` (VPMADDUBSW
/// with the sign trick, both operands in [-127, 127] so no i16 saturation, VPMADDWD with ones, i32 sums: exact, so the
/// i32 of every output is the same in any order) and the same epilogue, so the results are bit-identical. Differences:
/// aligned weight loads (no 128-bit halves under the x86-64-v2 tuning) and full 4-row tiles in a loop without per-row
/// branches (two row-pair pointers advanced by 16 bytes per k block).
#[target_feature(enable = "avx2,fma")]
pub unsafe fn gemm_i8_avx2_v2_ptr(q: &QActs, w: &PackedI8, yp: *mut f32, ldy: usize, epi: Epi) {
    let kbs = w.kp / QK;
    let m = q.m;
    let astride = kbs * 16;
    let ad = q.data.as_slice().as_ptr();
    let wd = w.data.as_slice().as_ptr();
    assert!(
        wd as usize % 32 == 0,
        "packed int8 weights must be 32-byte aligned"
    );
    let sa = q.scales.as_slice();
    let ones = _mm256_set1_epi16(1);
    let perm = _mm256_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7);
    let full = m & !3;
    for nb in 0..w.n.div_ceil(QN) {
        let c0 = nb * QN;
        let nc = (w.n - c0).min(QN);
        let b = wd.add(nb * kbs * QN * QK);
        let sw = w.scales.as_ptr().add(c0);
        let bias = w.bias.as_ref().map(|bb| bb.as_ptr().add(c0));
        let mut r0 = 0;
        while r0 < full {
            // rows r0..r0 + 4 = row pairs r0 / 2 and r0 / 2 + 1 (r0 is even): [pair][kb][16] = row 2i | row 2i + 1
            let pa = ad.add((r0 / 2) * astride);
            let pb = pa.add(astride);
            let z = _mm256_setzero_si256();
            let (mut a00, mut a01, mut a10, mut a11) = (z, z, z, z);
            let (mut a20, mut a21, mut a30, mut a31) = (z, z, z, z);
            for kb in 0..kbs {
                let b01 = _mm256_load_si256(b.add(kb * 64) as *const __m256i);
                let b23 = _mm256_load_si256(b.add(kb * 64 + 32) as *const __m256i);
                let o = kb * 16;
                i8_row_step(pa.add(o), b01, b23, ones, &mut a00, &mut a01);
                i8_row_step(pa.add(o + 8), b01, b23, ones, &mut a10, &mut a11);
                i8_row_step(pb.add(o), b01, b23, ones, &mut a20, &mut a21);
                i8_row_step(pb.add(o + 8), b01, b23, ones, &mut a30, &mut a31);
            }
            for (r, (x0, x1)) in [(a00, a01), (a10, a11), (a20, a21), (a30, a31)]
                .into_iter()
                .enumerate()
            {
                let h = _mm256_hadd_epi32(x0, x1);
                let c = _mm256_permutevar8x32_epi32(h, perm);
                finish_i8_avx2(
                    yp.add((r0 + r) * ldy + c0),
                    c,
                    sa[r0 + r],
                    sw,
                    bias,
                    epi,
                    nc,
                );
            }
            r0 += 4;
        }
        // the last 1-3 rows
        for r in full..m {
            let pr = ad.add((r / 2) * astride + (r % 2) * 8);
            let z = _mm256_setzero_si256();
            let (mut x0, mut x1) = (z, z);
            for kb in 0..kbs {
                let b01 = _mm256_load_si256(b.add(kb * 64) as *const __m256i);
                let b23 = _mm256_load_si256(b.add(kb * 64 + 32) as *const __m256i);
                i8_row_step(pr.add(kb * 16), b01, b23, ones, &mut x0, &mut x1);
            }
            let h = _mm256_hadd_epi32(x0, x1);
            let c = _mm256_permutevar8x32_epi32(h, perm);
            finish_i8_avx2(yp.add(r * ldy + c0), c, sa[r], sw, bias, epi, nc);
        }
    }
}
