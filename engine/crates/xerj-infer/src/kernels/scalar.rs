//! Portable reference kernels (any CPU). Also the numerical reference the SIMD kernels are unit-tested against.

use super::{epi_store, Epi, PackedF32, PackedI8, QActs, NR, QK, QN};

/// y = epi(x · Wᵀ + b), f32
pub fn gemm_f32(x: &[f32], m: usize, w: &PackedF32, y: &mut [f32], ldy: usize, epi: Epi) {
    let k = w.k;
    let d = w.data.as_slice();
    let mut acc = [0f32; NR];
    for p in 0..w.panels() {
        let c0 = p * NR;
        let nc = (w.n - c0).min(NR);
        let panel = &d[p * k * NR..(p + 1) * k * NR];
        for r in 0..m {
            acc.fill(0.0);
            let xr = &x[r * k..(r + 1) * k];
            for (kk, &xv) in xr.iter().enumerate() {
                let b = &panel[kk * NR..kk * NR + NR];
                for j in 0..NR {
                    acc[j] += xv * b[j];
                }
            }
            let bias = w.bias.as_ref().map(|b| &b[c0..c0 + nc]);
            epi_store(
                &mut y[r * ldy + c0..r * ldy + c0 + nc],
                &acc[..nc],
                bias,
                epi,
            );
        }
    }
}

/// per-row symmetric int8 quantization into the row-pair layout
pub fn quantize_rows(x: &[f32], m: usize, k: usize, pre: Option<&[f32]>, q: &mut QActs) {
    let kbs = q.kp / QK;
    let (data, scales) = (q.data.as_mut_slice(), q.scales.as_mut_slice());
    for r in 0..m {
        let xr = &x[r * k..(r + 1) * k];
        let elem = |i: usize| -> f32 {
            match pre {
                Some(p) => xr[i] * p[i],
                None => xr[i],
            }
        };
        let amax = (0..k).fold(0f32, |a, i| a.max(elem(i).abs()));
        let (s, inv) = if amax > 0.0 {
            (amax / 127.0, 127.0 / amax)
        } else {
            (0.0, 0.0)
        };
        scales[r] = s;
        let (pair, half) = (r / 2, r % 2);
        for kb in 0..kbs {
            let base = (pair * kbs + kb) * 16 + half * 8;
            for kk in 0..QK {
                let i = kb * QK + kk;
                data[base + kk] = if i < k {
                    (elem(i) * inv).round().clamp(-127.0, 127.0) as i8
                } else {
                    0
                };
            }
        }
    }
    if m % 2 == 1 {
        // padding row of the last pair: zeros (its outputs are never stored)
        let pair = m / 2;
        for kb in 0..kbs {
            let base = (pair * kbs + kb) * 16 + 8;
            data[base..base + 8].fill(0);
        }
        scales[m] = 0.0;
    }
}

/// y = epi(dequant(q · Wᵀ) + b)
pub fn gemm_i8(q: &QActs, w: &PackedI8, y: &mut [f32], ldy: usize, epi: Epi) {
    let kbs = w.kp / QK;
    let wd = w.data.as_slice();
    let ad = q.data.as_slice();
    let sa = q.scales.as_slice();
    let mut acc = [0f32; QN];
    for nb in 0..w.n.div_ceil(QN) {
        let c0 = nb * QN;
        let nc = (w.n - c0).min(QN);
        for r in 0..q.m {
            let (pair, half) = (r / 2, r % 2);
            let mut iacc = [0i32; QN];
            for kb in 0..kbs {
                let a = &ad[(pair * kbs + kb) * 16 + half * 8..][..8];
                let b = &wd[(nb * kbs + kb) * QN * QK..][..QN * QK];
                for c in 0..QN {
                    let mut s = 0i32;
                    for kk in 0..QK {
                        s += a[kk] as i32 * b[c * QK + kk] as i32;
                    }
                    iacc[c] += s;
                }
            }
            for c in 0..nc {
                acc[c] = iacc[c] as f32 * (sa[r] * w.scales[c0 + c]);
            }
            let bias = w.bias.as_ref().map(|b| &b[c0..c0 + nc]);
            epi_store(
                &mut y[r * ldy + c0..r * ldy + c0 + nc],
                &acc[..nc],
                bias,
                epi,
            );
        }
    }
}
