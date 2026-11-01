//! SIMD kernels vs the scalar reference on random data (every ISA the CPU supports).
//! f32: max relative difference (summation order differs); int8: bit-identical (integer accumulation + identical epilogue).

use xerj_infer::kernels::{
    CpuFeatures, Epi, F32Isa, GemmScratch, I8Isa, Kernels, PackedF32, PackedI8,
};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
    fn vec(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| self.next() * scale).collect()
    }
}

fn f32_isas() -> Vec<F32Isa> {
    let f = CpuFeatures::detect();
    let mut v = vec![];
    #[cfg(target_arch = "aarch64")]
    v.push(F32Isa::Neon);
    #[cfg(target_arch = "x86_64")]
    {
        v.push(F32Isa::Avx2);
        v.push(F32Isa::Avx512);
    }
    v.into_iter().filter(|i| f.supports_f32(*i)).collect()
}

fn i8_isas() -> Vec<I8Isa> {
    let f = CpuFeatures::detect();
    let mut v = vec![];
    #[cfg(target_arch = "aarch64")]
    {
        v.push(I8Isa::NeonDot);
        v.push(I8Isa::NeonI8mm);
    }
    #[cfg(target_arch = "x86_64")]
    {
        v.push(I8Isa::Avx2);
        v.push(I8Isa::Avx512Vnni);
    }
    v.into_iter().filter(|i| f.supports_i8(*i)).collect()
}

const MS: [usize; 9] = [1, 2, 3, 5, 6, 7, 8, 13, 37];
const NS: [usize; 5] = [10, 16, 33, 256, 1166];
const KS: [usize; 5] = [16, 37, 256, 260, 384];

#[test]
fn f32_gemm_matches_scalar() {
    let mut rng = Rng(0x9e3779b97f4a7c15);
    let isas = f32_isas();
    eprintln!("f32 ISAs under test: {:?}", isas);
    for &m in &MS {
        for &n in &NS {
            for &k in &KS {
                let x = rng.vec(m * k, 1.0);
                let w = rng.vec(n * k, 0.1);
                let b = rng.vec(n, 0.5);
                let pw = PackedF32::pack(&w, n, k, Some(&b));
                for epi in [Epi::STORE, Epi::gated_add(0.37)] {
                    let y0 = rng.vec(m * n, 1.0);
                    let mut yr = y0.clone();
                    let mut s = GemmScratch::default();
                    Kernels::scalar().gemm_f32(&x, m, &pw, &mut yr, n, epi, &mut s);
                    // reference sanity vs naive triple loop
                    for r in 0..m {
                        for c in 0..n {
                            let mut acc = 0f64;
                            for kk in 0..k {
                                acc += x[r * k + kk] as f64 * w[c * k + kk] as f64;
                            }
                            let want = if epi.accumulate {
                                y0[r * n + c] as f64
                            } else {
                                0.0
                            } + epi.alpha as f64 * (acc + b[c] as f64);
                            assert!(
                                (yr[r * n + c] as f64 - want).abs() < 1e-4 * (1.0 + want.abs()),
                                "scalar m{m} n{n} k{k}"
                            );
                        }
                    }
                    for &isa in &isas {
                        let kern = Kernels {
                            f32_isa: isa,
                            i8_isa: I8Isa::Scalar,
                            kleidi: false,
                        };
                        let mut y = y0.clone();
                        kern.gemm_f32(&x, m, &pw, &mut y, n, epi, &mut s);
                        for i in 0..m * n {
                            let d = (y[i] - yr[i]).abs();
                            assert!(
                                d <= 2e-5 * (1.0 + yr[i].abs()),
                                "{} m{m} n{n} k{k}: {} vs {}",
                                isa.name(),
                                y[i],
                                yr[i]
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn int8_gemm_bit_identical_to_scalar() {
    let mut rng = Rng(0x2545f4914f6cdd1d);
    let isas = i8_isas();
    eprintln!("int8 ISAs under test: {:?}", isas);
    for &m in &MS {
        for &n in &NS {
            for &k in &KS {
                let x = rng.vec(m * k, 2.0);
                let w = rng.vec(n * k, 0.1);
                let b = rng.vec(n, 0.5);
                let pw = PackedI8::quantize(&w, n, k, Some(&b));
                for epi in [Epi::STORE, Epi::ADD, Epi::gated_add(-0.61)] {
                    let y0 = rng.vec(m * n, 1.0);
                    let mut yr = y0.clone();
                    let mut s = GemmScratch::default();
                    let sk = Kernels::scalar();
                    sk.quantize(&x, m, k, None, &mut s);
                    sk.gemm_i8(&pw, &mut yr, n, epi, &s);
                    // the dequantized product is close to the f32 product
                    let wd = pw.dequant();
                    for r in 0..m {
                        for c in 0..n {
                            let mut acc = 0f64;
                            for kk in 0..k {
                                acc += x[r * k + kk] as f64 * wd[c * k + kk] as f64;
                            }
                            let want = if epi.accumulate {
                                y0[r * n + c] as f64
                            } else {
                                0.0
                            } + epi.alpha as f64 * (acc + b[c] as f64);
                            let tol = 0.005 * (k as f64).sqrt();
                            assert!(
                                (yr[r * n + c] as f64 - want).abs() < tol,
                                "scalar int8 m{m} n{n} k{k}: {} vs {want}",
                                yr[r * n + c]
                            );
                        }
                    }
                    for &isa in &isas {
                        let kern = Kernels {
                            f32_isa: F32Isa::Scalar,
                            i8_isa: isa,
                            kleidi: false,
                        };
                        let mut s2 = GemmScratch::default();
                        kern.quantize(&x, m, k, None, &mut s2);
                        assert_eq!(
                            s2.q.scales.as_slice()[..m],
                            s.q.scales.as_slice()[..m],
                            "{} quant scales",
                            isa.name()
                        );
                        let mut y = y0.clone();
                        kern.gemm_i8(&pw, &mut y, n, epi, &s2);
                        for i in 0..m * n {
                            assert_eq!(
                                y[i].to_bits(),
                                yr[i].to_bits(),
                                "{} m{m} n{n} k{k} i{i}: {} vs {}",
                                isa.name(),
                                y[i],
                                yr[i]
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn smoothing_and_fused_layernorm_quantization() {
    use xerj_infer::kernels::ops;
    let mut rng = Rng(0x51ed270b27f3a1c3);
    for &isa in &i8_isas() {
        let kern = Kernels {
            f32_isa: F32Isa::Scalar,
            i8_isa: isa,
            kleidi: false,
        };
        for &(m, k) in &[(1usize, 256usize), (7, 256), (37, 384), (5, 1168)] {
            let x = rng.vec(m * k, 3.0);
            let pre: Vec<f32> = (0..k).map(|_| 0.5 + rng.next().abs() * 2.0).collect();
            // smoothing factors: quantize(x, pre) == quantize(x * pre) bit for bit, and equal to the scalar path
            let xs: Vec<f32> = x
                .iter()
                .zip(pre.iter().cycle())
                .map(|(a, b)| a * b)
                .collect();
            let (mut s1, mut s2, mut s3) = (
                GemmScratch::default(),
                GemmScratch::default(),
                GemmScratch::default(),
            );
            kern.quantize(&x, m, k, Some(&pre), &mut s1);
            kern.quantize(&xs, m, k, None, &mut s2);
            Kernels::scalar().quantize(&x, m, k, Some(&pre), &mut s3);
            assert_eq!(
                s1.q.data.as_slice(),
                s2.q.data.as_slice(),
                "{} pre-scale",
                isa.name()
            );
            assert_eq!(
                s1.q.data.as_slice(),
                s3.q.data.as_slice(),
                "{} vs scalar",
                isa.name()
            );
            // fused LayerNorm + quantization vs LayerNorm then quantization: within one level
            let w: Vec<f32> = (0..k).map(|_| rng.next() * 2.0).collect();
            let b: Vec<f32> = (0..k).map(|_| rng.next() * 0.3).collect();
            let mut ln = vec![0f32; m * k];
            ops::layer_norm(&x, m, k, &w, Some(&b), 1e-5, &mut ln);
            let mut sa = GemmScratch::default();
            let mut sb = GemmScratch::default();
            if kern.quantize_ln(&x, m, k, &w, Some(&b), 1e-5, Some(&pre), &mut sa) {
                let lns: Vec<f32> = ln
                    .iter()
                    .zip(pre.iter().cycle())
                    .map(|(a, b)| a * b)
                    .collect();
                kern.quantize(&lns, m, k, None, &mut sb);
                let (da, db) = (sa.q.data.as_slice(), sb.q.data.as_slice());
                let off = da
                    .iter()
                    .zip(db)
                    .filter(|(p, q)| (**p as i32 - **q as i32).abs() > 1)
                    .count();
                let diff1 = da.iter().zip(db).filter(|(p, q)| p != q).count();
                assert_eq!(
                    off,
                    0,
                    "{} fused LN quantization off by more than one level",
                    isa.name()
                );
                assert!(
                    diff1 * 1000 <= da.len(),
                    "{} fused LN: {diff1} of {} values differ",
                    isa.name(),
                    da.len()
                );
            }
        }
    }
}

/// x86 AVX2 + FMA element-wise ops (x86ops): bit-identical to the portable code they replace
#[cfg(target_arch = "x86_64")]
#[test]
fn x86_ops_bit_identical_to_portable() {
    use xerj_infer::kernels::{ops, x86ops};
    if !x86ops::enabled() {
        eprintln!("x86ops disabled (no AVX2 + FMA or XERJ_AVX2_OPS=0): skipped");
        return;
    }
    let mut rng = Rng(0x2545f4914f6cdd1d);
    // LayerNorm (with and without bias), several widths incl. a tail and rows shorter than one vector
    for &(m, k) in &[
        (1usize, 512usize),
        (7, 512),
        (3, 37),
        (5, 2318),
        (2, 768),
        (4, 13),
        (2, 5),
    ] {
        let x = rng.vec(m * k, 3.0);
        let w = rng.vec(k, 2.0);
        let b = rng.vec(k, 0.3);
        for bias in [None, Some(&b[..])] {
            let mut ya = vec![0f32; m * k];
            let mut yb = vec![0f32; m * k];
            let mut yc = vec![0f32; m * k];
            ops::layer_norm(&x, m, k, &w, bias, 1e-5, &mut ya);
            for r in 0..m {
                ops::layer_norm_row_portable(
                    &x[r * k..(r + 1) * k],
                    &w,
                    bias,
                    1e-5,
                    &mut yb[r * k..(r + 1) * k],
                );
                ops::layer_norm_row(
                    &x[r * k..(r + 1) * k],
                    &w,
                    bias,
                    1e-5,
                    &mut yc[r * k..(r + 1) * k],
                );
            }
            assert!(
                ya.iter().zip(&yb).all(|(p, q)| p.to_bits() == q.to_bits()),
                "layer_norm {m}x{k}"
            );
            assert!(
                yc.iter().zip(&yb).all(|(p, q)| p.to_bits() == q.to_bits()),
                "layer_norm_row {m}x{k}"
            );
        }
        // fused LayerNorm + quantization == LayerNorm then scalar quantization (k % 8 == 0 only)
        if k % 8 == 0 {
            let pre: Vec<f32> = (0..k).map(|_| 0.5 + rng.next().abs() * 2.0).collect();
            let kern = Kernels::detect();
            let mut sa = GemmScratch::default();
            assert!(kern.quantize_ln(&x, m, k, &w, Some(&b), 1e-5, Some(&pre), &mut sa));
            let mut ln = vec![0f32; m * k];
            for r in 0..m {
                ops::layer_norm_row_portable(
                    &x[r * k..(r + 1) * k],
                    &w,
                    Some(&b),
                    1e-5,
                    &mut ln[r * k..(r + 1) * k],
                );
            }
            let mut sb = GemmScratch::default();
            Kernels::scalar().quantize(&ln, m, k, Some(&pre), &mut sb);
            assert_eq!(
                sa.q.data.as_slice(),
                sb.q.data.as_slice(),
                "fused LN quantization {m}x{k}"
            );
            assert!(sa
                .q
                .scales
                .as_slice()
                .iter()
                .zip(sb.q.scales.as_slice())
                .all(|(p, q)| p.to_bits() == q.to_bits()));
        }
    }
    // GeGLU and GELU, including large |x| (erf clamp) and a tail
    for &(m, h) in &[(1usize, 768usize), (9, 768), (2, 13)] {
        let u = rng.vec(m * 2 * h, 6.0);
        let mut ga = vec![0f32; m * h];
        let mut gb = vec![0f32; m * h];
        ops::geglu(&u, m, h, &mut ga);
        ops::geglu_portable(&u, m, h, &mut gb);
        assert!(
            ga.iter().zip(&gb).all(|(p, q)| p.to_bits() == q.to_bits()),
            "geglu {m}x{h}"
        );
        let mut xa = u.clone();
        let mut xb = u.clone();
        ops::gelu_inplace(&mut xa);
        ops::gelu_inplace_portable(&mut xb);
        assert!(
            xa.iter().zip(&xb).all(|(p, q)| p.to_bits() == q.to_bits()),
            "gelu"
        );
    }
    // RoPE of every head of a q | k row
    let (heads, hd) = (8usize, 64usize);
    let d = heads * hd;
    let cs = rng.vec(hd / 2, 1.0);
    let sn = rng.vec(hd / 2, 1.0);
    let row = rng.vec(3 * d, 2.0);
    let mut ra = row.clone();
    let mut rb = row.clone();
    let mut rc = row.clone();
    unsafe { x86ops::rope_qk(&mut ra, d, heads, hd, &cs, &sn) };
    // round 2: explicit AVX2 RoPE (head_dim 64), on a misaligned row too
    unsafe { x86ops::rope_qk64(&mut rc, d, heads, &cs, &sn) };
    let mut rd = vec![0f32; 3 * d + 3];
    rd[3..].copy_from_slice(&row);
    unsafe { x86ops::rope_qk64(&mut rd[3..], d, heads, &cs, &sn) };
    for h in 0..heads {
        ops::rope_head_portable(&mut rb[h * hd..(h + 1) * hd], &cs, &sn);
        ops::rope_head_portable(&mut rb[d + h * hd..d + (h + 1) * hd], &cs, &sn);
    }
    for (name, r) in [
        ("rope", &ra[..]),
        ("rope_qk64", &rc[..]),
        ("rope_qk64 misaligned", &rd[3..]),
    ] {
        assert!(
            r.iter().zip(&rb).all(|(p, q)| p.to_bits() == q.to_bits()),
            "{name}"
        );
    }
    // the single-instruction unaligned load returns the same bits as the split load
    let buf = rng.vec(64, 3.0);
    for off in 0..9 {
        let (mut a, mut b) = ([0f32; 8], [0f32; 8]);
        unsafe {
            use std::arch::x86_64::*;
            _mm256_storeu_ps(a.as_mut_ptr(), x86ops::ldu(buf.as_ptr().add(off)));
            _mm256_storeu_ps(b.as_mut_ptr(), _mm256_loadu_ps(buf.as_ptr().add(off)));
        }
        assert_eq!(a.map(f32::to_bits), b.map(f32::to_bits), "ldu offset {off}");
    }
    // exp8 == ops::exp lane by lane (incl. exact .5 ties of x * log2(e) are rare; a dense sweep covers the range)
    let mut x = -90.0f32;
    while x < 90.0 {
        let v: Vec<f32> = (0..8).map(|i| x + i as f32 * 1.37e-3).collect();
        let mut o = [0f32; 8];
        unsafe {
            use std::arch::x86_64::*;
            _mm256_storeu_ps(o.as_mut_ptr(), x86ops::exp8(_mm256_loadu_ps(v.as_ptr())));
        }
        for i in 0..8 {
            assert_eq!(o[i].to_bits(), ops::exp(v[i]).to_bits(), "exp8({})", v[i]);
        }
        x += 0.0137;
    }
}

/// x86 transposed-key copies and attention vs the portable code
#[cfg(target_arch = "x86_64")]
#[test]
fn x86_attention_matches_portable() {
    use xerj_infer::kernels::{ops, x86ops};
    use xerj_infer::model::LayerKv;
    if !x86ops::enabled() {
        eprintln!("x86ops disabled: skipped");
        return;
    }
    let mut rng = Rng(0x7a3b9c1d5e2f4a6b);
    let (heads, hd) = (8usize, 64usize);
    let d = heads * hd;
    for &t in &[1usize, 5, 17, 64, 100, 333, 830] {
        let mut kv = LayerKv::default();
        kv.ensure(t, d);
        for p in 0..t {
            let key = rng.vec(d, 1.5);
            let val = rng.vec(d, 1.0);
            kv.put(p, &key, &val);
        }
        kv.transpose_keys(0, t, d);
        // the transposed copy equals the row-major keys
        let kts = kv.kts();
        for i in 0..d {
            for p in 0..t {
                assert_eq!(
                    kv.kt.as_slice()[i * kts + p].to_bits(),
                    kv.k.as_slice()[p * d + i].to_bits()
                );
            }
        }
        // query sets: every position (prefix), the last few (suffix), a sparse sorted set (needed rows)
        let all: Vec<usize> = (0..t).collect();
        let tail: Vec<usize> = (t.saturating_sub(7)..t).collect();
        let sparse: Vec<usize> = (0..t).filter(|p| p % 7 == 3 || *p == t - 1).collect();
        for pos in [&all, &tail, &sparse] {
            let nq = pos.len();
            let q = rng.vec(nq * d, 1.0);
            for window in [None, Some(64usize)] {
                let mut sa = Vec::new();
                let mut sb = Vec::new();
                let mut oa = vec![0f32; nq * d];
                let mut ob = vec![0f32; nq * d];
                ops::self_attention(&q, pos, &kv, heads, hd, window, &mut sa, &mut oa);
                ops::self_attention_portable(&q, pos, &kv, heads, hd, window, &mut sb, &mut ob);
                let scale = ob.iter().fold(0f32, |a, &v| a.max(v.abs())).max(1e-3);
                let worst = oa
                    .iter()
                    .zip(&ob)
                    .fold(0f32, |a, (x, y)| a.max((x - y).abs()));
                assert!(
                    worst <= 2e-6 * scale.max(1.0),
                    "attention t={t} nq={nq} window={window:?}: max diff {worst}"
                );
            }
        }
    }
}

/// output-channel shards (intra-question parallelism): bit-identical to the unsharded linear for every thread count,
/// f32 and int8 (smoothed), plain and fused-LayerNorm forwards
#[test]
fn sharded_linear_bit_identical() {
    use std::sync::Arc;
    use xerj_infer::model::{LinMode, Linear};
    use xerj_infer::pool::Pool;
    let mut rng = Rng(0x0123456789abcdef);
    let kern = Kernels::detect();
    let pools: Vec<Option<Arc<Pool>>> = vec![
        None,
        Some(Arc::new(Pool::new(2))),
        Some(Arc::new(Pool::new(4))),
    ];
    for &(n, k) in &[
        (512usize, 512usize),
        (1536, 512),
        (2318, 512),
        (512, 768),
        (40, 16),
    ] {
        let w = rng.vec(n * k, 0.05);
        let bias = rng.vec(n, 0.2);
        let smooth: Vec<f32> = (0..k).map(|_| 0.5 + rng.next().abs()).collect();
        let lnw = rng.vec(k, 1.5);
        for mode in [
            LinMode::F32,
            LinMode::I8 {
                smooth: None,
                keep_f32: false,
            },
            LinMode::I8 {
                smooth: Some(&smooth),
                keep_f32: true,
            },
            LinMode::W8,
        ] {
            let full = Linear::new("t", 0, &w, n, k, Some(&bias), mode);
            for parts in [2usize, 3, 4] {
                let sh = Linear::new_sharded("t", 0, &w, n, k, Some(&bias), mode, parts);
                for &m in &[1usize, 3, 37, 200] {
                    let x = rng.vec(m * k, 2.0);
                    for epi in [Epi::STORE, Epi::ADD] {
                        for force_f32 in [false, true] {
                            let y0 = rng.vec(m * n, 1.0);
                            let mut yr = y0.clone();
                            let mut s = GemmScratch::default();
                            s.force_f32 = force_f32;
                            full.forward(&kern, &x, m, &mut yr, n, epi, &mut s);
                            let mut xn = Vec::new();
                            let mut yl = y0.clone();
                            full.forward_ln(
                                &kern, &x, m, &lnw, None, 1e-5, &mut xn, &mut yl, n, epi, &mut s,
                            );
                            for pool in &pools {
                                let mut s2 = GemmScratch::default();
                                s2.force_f32 = force_f32;
                                s2.pool = pool.clone();
                                let mut ys = y0.clone();
                                sh.forward(&kern, &x, m, &mut ys, n, epi, &mut s2);
                                assert!(
                                    yr.iter().zip(&ys).all(|(a, b)| a.to_bits() == b.to_bits()),
                                    "forward n={n} k={k} m={m} parts={parts} f32={force_f32} pool={:?}",
                                    pool.as_ref().map(|p| p.threads())
                                );
                                let mut xn2 = Vec::new();
                                let mut ys = y0.clone();
                                sh.forward_ln(
                                    &kern, &x, m, &lnw, None, 1e-5, &mut xn2, &mut ys, n, epi,
                                    &mut s2,
                                );
                                assert!(
                                    yl.iter().zip(&ys).all(|(a, b)| a.to_bits() == b.to_bits()),
                                    "forward_ln n={n} k={k} m={m} parts={parts}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}

/// weight-only INT8 rounds as tools/int8_fakequant.py: scale = max|w| / 127, torch.round (halves to even), clip ±127
#[test]
fn w8_rounding_is_int8_fakequant() {
    use xerj_infer::kernels::w8_quantize_row;
    let mut q = [0i8; 4];
    let s = w8_quantize_row(&[127.0, 2.5, 3.5, -127.0], &mut q);
    assert_eq!(s, 1.0);
    assert_eq!(q, [127, 2, 4, -127], "halves go to the even neighbour");
    // an all-zero row keeps the clamped scale and rounds to zero
    let s = w8_quantize_row(&[0.0; 4], &mut q);
    assert_eq!(s, 1e-12f32 / 127.0);
    assert_eq!(q, [0; 4]);
    // one value is exactly the row maximum: it lands on ±127 and nothing leaves the range
    let s = w8_quantize_row(&[0.3, -0.7, 0.1, 0.2], &mut q);
    assert_eq!(s, 0.7f32 / 127.0);
    assert_eq!(q[1], -127);
}

/// a w8 linear computes exactly what the f32 linear computes on the rounded weights: same kernel, same bits, every f32 ISA
#[test]
fn w8_linear_is_f32_on_rounded_weights() {
    use xerj_infer::kernels::{w8_quantize_row, PackedW8};
    use xerj_infer::model::{LinMode, Linear};
    let mut rng = Rng(0x7e57ab1e);
    for kern_isa in f32_isas() {
        let kern = Kernels {
            f32_isa: kern_isa,
            ..Kernels::detect()
        };
        for &(n, k) in &[
            (512usize, 512usize),
            (1536, 512),
            (40, 16),
            (15, 7),
            (17, 33),
        ] {
            let w = rng.vec(n * k, 0.05);
            let bias = rng.vec(n, 0.2);
            let mut rounded = vec![0f32; n * k];
            let mut q = vec![0i8; k];
            for c in 0..n {
                let scale = w8_quantize_row(&w[c * k..(c + 1) * k], &mut q);
                for j in 0..k {
                    rounded[c * k + j] = q[j] as f32 * scale;
                }
            }
            let a = Linear::new("t", 0, &w, n, k, Some(&bias), LinMode::W8);
            let b = Linear::new("t", 0, &rounded, n, k, Some(&bias), LinMode::F32);
            assert!(a.bytes() < b.bytes(), "w8 holds fewer bytes than f32");
            let mut s = GemmScratch::default();
            for &m in &[1usize, 5, 37] {
                let x = rng.vec(m * k, 2.0);
                for epi in [Epi::STORE, Epi::ADD] {
                    let y0 = rng.vec(m * n, 1.0);
                    let (mut ya, mut yb) = (y0.clone(), y0.clone());
                    a.forward(&kern, &x, m, &mut ya, n, epi, &mut s);
                    b.forward(&kern, &x, m, &mut yb, n, epi, &mut s);
                    assert!(
                        ya.iter().zip(&yb).all(|(p, r)| p.to_bits() == r.to_bits()),
                        "{} n={n} k={k} m={m}",
                        kern_isa.name()
                    );
                }
            }
            // a shorter linear after a longer one reuses the buffers: nothing of the previous weights or bias may remain
            let w8 = PackedW8::quantize(&w, n, k, None);
            let mut d = PackedF32::pack(&rng.vec(4096, 1.0), 2, 2048, Some(&[1.0, 2.0]));
            w8.dequant_into(&mut d);
            assert!(d.bias.is_none());
            assert_eq!((d.n, d.k, d.data.len()), (n, k, n.div_ceil(16) * k * 16));
        }
    }
}

/// row-parallel GeGLU / GELU over a pool: bit-identical to the single-threaded functions
#[test]
fn pooled_elementwise_bit_identical() {
    use xerj_infer::kernels::ops;
    use xerj_infer::pool::Pool;
    let mut rng = Rng(0x5851f42d4c957f2d);
    for threads in [2usize, 3, 4] {
        let pool = Pool::new(threads);
        for &(m, h) in &[
            (1usize, 768usize),
            (31, 768),
            (32, 768),
            (333, 768),
            (97, 13),
        ] {
            let u = rng.vec(m * 2 * h, 5.0);
            let mut ga = vec![0f32; m * h];
            let mut gb = vec![0f32; m * h];
            ops::geglu(&u, m, h, &mut ga);
            ops::geglu_pool(&u, m, h, &mut gb, Some(&pool));
            assert!(
                ga.iter().zip(&gb).all(|(a, b)| a.to_bits() == b.to_bits()),
                "geglu_pool {m}x{h} t={threads}"
            );
        }
        for &(m, d) in &[
            (1usize, 512usize),
            (31, 512),
            (32, 512),
            (333, 512),
            (97, 37),
        ] {
            let x = rng.vec(m * d, 3.0);
            let w = rng.vec(d, 1.5);
            let b = rng.vec(d, 0.3);
            for bias in [None, Some(&b[..])] {
                let mut ya = vec![0f32; m * d];
                let mut yb = vec![0f32; m * d];
                ops::layer_norm(&x, m, d, &w, bias, 1e-5, &mut ya);
                ops::layer_norm_pool(&x, m, d, &w, bias, 1e-5, &mut yb, Some(&pool));
                assert!(
                    ya.iter().zip(&yb).all(|(a, b)| a.to_bits() == b.to_bits()),
                    "layer_norm_pool {m}x{d} t={threads}"
                );
            }
        }
        for n in [5usize, 1 << 14, (1 << 15) + 7, 333 * 2318] {
            let x = rng.vec(n, 5.0);
            let mut xa = x.clone();
            let mut xb = x.clone();
            ops::gelu_inplace(&mut xa);
            ops::gelu_inplace_pool(&mut xb, Some(&pool));
            assert!(
                xa.iter().zip(&xb).all(|(a, b)| a.to_bits() == b.to_bits()),
                "gelu_inplace_pool n={n} t={threads}"
            );
        }
    }
}

/// round 2 (x86): the AVX2 GEMM kernels v2 against the round-1 kernels, bit for bit (f32 and int8, every epilogue, edge
/// shapes: row counts below and around a tile, channel counts off the panel width, odd k)
#[cfg(target_arch = "x86_64")]
#[test]
fn x86_gemm_v2_bit_identical_to_round1() {
    use xerj_infer::kernels::{x86, AVec};
    let f = CpuFeatures::detect();
    if !(f.avx2 && f.fma) {
        eprintln!("no AVX2 + FMA: skipped");
        return;
    }
    let mut rng = Rng(0x3c6ef372fe94f82b);
    for &m in &[1usize, 2, 3, 4, 5, 6, 7, 8, 13, 37, 64] {
        for &(n, k) in &[
            (10usize, 16usize),
            (16, 37),
            (33, 256),
            (256, 260),
            (1166, 384),
            (512, 768),
        ] {
            let x = rng.vec(m * k, 2.0);
            let w = rng.vec(n * k, 0.1);
            let b = rng.vec(n, 0.5);
            for (bias, epi) in [
                (None, Epi::STORE),
                (Some(&b[..]), Epi::ADD),
                (Some(&b[..]), Epi::gated_add(-0.61)),
            ] {
                let pf = PackedF32::pack(&w, n, k, bias);
                let pw = PackedI8::quantize(&w, n, k, bias);
                let y0 = rng.vec(m * n, 1.0);
                let mut s = GemmScratch::default();
                Kernels::detect().quantize(&x, m, k, None, &mut s);
                let mut ap = AVec::new();
                let (mut f1, mut f2, mut i1, mut i2) =
                    (y0.clone(), y0.clone(), y0.clone(), y0.clone());
                // SAFETY: AVX2 + FMA checked above; the outputs hold m rows of n
                unsafe {
                    x86::gemm_f32_avx2_ptr(&x, m, &pf, f1.as_mut_ptr(), n, epi, &mut ap);
                    x86::gemm_f32_avx2_v2_ptr(&x, m, &pf, f2.as_mut_ptr(), n, epi, &mut ap);
                    x86::gemm_i8_avx2_ptr(&s.q, &pw, i1.as_mut_ptr(), n, epi);
                    x86::gemm_i8_avx2_v2_ptr(&s.q, &pw, i2.as_mut_ptr(), n, epi);
                }
                assert!(
                    f1.iter().zip(&f2).all(|(a, b)| a.to_bits() == b.to_bits()),
                    "f32 v2 m{m} n{n} k{k}"
                );
                assert!(
                    i1.iter().zip(&i2).all(|(a, b)| a.to_bits() == b.to_bits()),
                    "int8 v2 m{m} n{n} k{k}"
                );
            }
        }
    }
}

/// round 2 (x86): the blocked key layout holds exactly the keys and values (element stores, 8 x 8 transposes from a row
/// matrix with unaligned ranges, growth; values row-major and head-major), and the round-2 attention over it is
/// bit-identical to the round-1 attention over the transposed keys, single-threaded and spread over pools of 2-4 threads
#[cfg(target_arch = "x86_64")]
#[test]
fn x86_r2_attention_bit_identical_to_round1() {
    use std::sync::{Arc, Mutex};
    use xerj_infer::kernels::{ops, x86ops};
    use xerj_infer::model::LayerKv;
    use xerj_infer::pool::Pool;
    if !x86ops::enabled() {
        eprintln!("x86ops disabled: skipped");
        return;
    }
    let mut rng = Rng(0x1f83d9abfb41bd6b);
    let (heads, hd) = (8usize, 64usize);
    let d = heads * hd;
    let blocked = |vhead: bool| {
        let mut kv = LayerKv::default();
        kv.blocked = true;
        kv.vhead = vhead;
        kv.heads = heads;
        kv.hd = hd;
        kv
    };
    let pools = [
        Arc::new(Pool::new(2)),
        Arc::new(Pool::new(3)),
        Arc::new(Pool::new(4)),
    ];
    for &t in &[1usize, 5, 17, 64, 100, 333, 830] {
        let keys = rng.vec(t * d, 1.5);
        let vals = rng.vec(t * d, 1.0);
        let mut kv1 = LayerKv::default();
        kv1.ensure(t, d);
        for p in 0..t {
            kv1.put(p, &keys[p * d..(p + 1) * d], &vals[p * d..(p + 1) * d]);
        }
        kv1.transpose_keys(0, t, d);
        // blocked caches, values row-major and head-major: [element stores, grown in steps] per value layout
        let mut caches = Vec::new();
        for vhead in [false, true] {
            // element stores
            let mut kv2 = blocked(vhead);
            kv2.ensure(t, d);
            for p in 0..t {
                kv2.put(p, &keys[p * d..(p + 1) * d], &vals[p * d..(p + 1) * d]);
            }
            // grown in steps, keys stored from a row matrix (stride 3d as in layer_forward) in unaligned ranges
            let mut kv3 = blocked(vhead);
            let mut p = 0;
            let mut step = 3;
            while p < t {
                let p1 = (p + step).min(t);
                kv3.ensure(p1, d);
                let mut rows = vec![0f32; (p1 - p) * 3 * d];
                for r in p..p1 {
                    rows[(r - p) * 3 * d + d..(r - p) * 3 * d + 2 * d]
                        .copy_from_slice(&keys[r * d..(r + 1) * d]);
                    kv3.put_v(r, &vals[r * d..(r + 1) * d]);
                }
                kv3.put_keys_blocked(&rows[d..], 3 * d, p, p1);
                p = p1;
                step = step * 2 + 5;
            }
            caches.push(kv2);
            caches.push(kv3);
        }
        for kv in &caches {
            for p in 0..t {
                for i in 0..d {
                    assert_eq!(
                        kv.key_at(p, i, d).to_bits(),
                        keys[p * d + i].to_bits(),
                        "t={t} key {p}/{i}"
                    );
                    assert_eq!(
                        kv.value_at(p, i, d).to_bits(),
                        vals[p * d + i].to_bits(),
                        "t={t} vhead={} value {p}/{i}",
                        kv.vhead
                    );
                }
            }
            if !kv.vhead {
                assert_eq!(kv.v.as_slice()[..t * d], vals[..]);
            }
        }
        let all: Vec<usize> = (0..t).collect();
        let tail: Vec<usize> = (t.saturating_sub(7)..t).collect();
        let sparse: Vec<usize> = (0..t).filter(|p| p % 7 == 3 || *p == t - 1).collect();
        let late: Vec<usize> = (t / 2..t).collect();
        for pos in [&all, &tail, &sparse, &late] {
            let nq = pos.len();
            let q = rng.vec(nq * d, 1.0);
            for window in [None, Some(64usize)] {
                let mut s1 = Vec::new();
                let mut o1 = vec![0f32; nq * d];
                ops::self_attention(&q, pos, &kv1, heads, hd, window, &mut s1, &mut o1);
                for kv in &caches {
                    let mut s2 = Vec::new();
                    let mut o2 = vec![0f32; nq * d];
                    ops::self_attention(&q, pos, kv, heads, hd, window, &mut s2, &mut o2);
                    assert!(
                        o1.iter().zip(&o2).all(|(a, b)| a.to_bits() == b.to_bits()),
                        "r2 attention t={t} nq={nq} window={window:?} vhead={}",
                        kv.vhead
                    );
                }
                for pool in &pools {
                    for kv in [&caches[1], &caches[3]] {
                        let mut tls: Vec<Mutex<ops::AttnScratch>> = Vec::new();
                        let mut o3 = vec![0f32; nq * d];
                        assert!(ops::self_attention_pool(
                            &q, pos, kv, heads, hd, window, pool, &mut tls, &mut o3
                        ));
                        assert!(
                            o1.iter().zip(&o3).all(|(a, b)| a.to_bits() == b.to_bits()),
                            "pooled r2 attention t={t} nq={nq} threads={} vhead={}",
                            pool.threads(),
                            kv.vhead
                        );
                    }
                }
            }
        }
    }
}
