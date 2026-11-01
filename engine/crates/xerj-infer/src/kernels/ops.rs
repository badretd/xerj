//! Element-wise and attention ops (f32). Written as fixed-width chunk loops so LLVM vectorizes them for the target
//! (NEON on aarch64; SSE2 baseline / AVX2 with -C target-cpu on x86-64). The attention inner products use 4-key blocks with
//! independent accumulators.

/// LayerNorm of one row: y = (x - mean) * rstd * w (+ b)   (PyTorch CPU order: (x + (-mean)) * rstd * gamma + beta)
#[inline]
pub fn layer_norm_row(x: &[f32], w: &[f32], b: Option<&[f32]>, eps: f32, y: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if super::x86ops::enabled() {
        // SAFETY: AVX2 + FMA checked at run time
        unsafe { super::x86ops::layer_norm_row(x, w, b, eps, y) };
        return;
    }
    layer_norm_row_portable(x, w, b, eps, y)
}

/// the portable LayerNorm row (also compiled for AVX2 + FMA by `x86ops`, bit-identical)
#[doc(hidden)]
#[inline(always)]
pub fn layer_norm_row_portable(x: &[f32], w: &[f32], b: Option<&[f32]>, eps: f32, y: &mut [f32]) {
    let n = x.len();
    let mut s = [0f32; 8];
    let chunks = n / 8;
    for c in 0..chunks {
        for j in 0..8 {
            s[j] += x[c * 8 + j];
        }
    }
    let mut sum: f32 = s.iter().sum();
    for &v in &x[chunks * 8..] {
        sum += v;
    }
    let mean = sum / n as f32;
    let mut q = [0f32; 8];
    for c in 0..chunks {
        for j in 0..8 {
            let d = x[c * 8 + j] - mean;
            q[j] += d * d;
        }
    }
    let mut var: f32 = q.iter().sum();
    for &v in &x[chunks * 8..] {
        let d = v - mean;
        var += d * d;
    }
    var /= n as f32;
    let rstd = 1.0 / (var.max(0.0) + eps).sqrt();
    match b {
        Some(b) => {
            for i in 0..n {
                y[i] = (x[i] - mean) * rstd * w[i] + b[i];
            }
        }
        None => {
            for i in 0..n {
                y[i] = (x[i] - mean) * rstd * w[i];
            }
        }
    }
}

/// LayerNorm of `m` rows of width `d`
pub fn layer_norm(
    x: &[f32],
    m: usize,
    d: usize,
    w: &[f32],
    b: Option<&[f32]>,
    eps: f32,
    y: &mut [f32],
) {
    #[cfg(target_arch = "x86_64")]
    if super::x86ops::enabled() {
        // SAFETY: AVX2 + FMA checked at run time
        unsafe { super::x86ops::layer_norm(x, m, d, w, b, eps, y) };
        return;
    }
    for r in 0..m {
        layer_norm_row_portable(
            &x[r * d..(r + 1) * d],
            w,
            b,
            eps,
            &mut y[r * d..(r + 1) * d],
        );
    }
}

/// `layer_norm` with the rows split over a pool (bit-identical: every row is normalized by `layer_norm`); plain
/// `layer_norm` without a pool or for few rows
#[allow(clippy::too_many_arguments)]
pub fn layer_norm_pool(
    x: &[f32],
    m: usize,
    d: usize,
    w: &[f32],
    b: Option<&[f32]>,
    eps: f32,
    y: &mut [f32],
    pool: Option<&crate::pool::Pool>,
) {
    match pool.filter(|p| p.threads() > 1 && m >= 32) {
        Some(pool) => {
            assert!(x.len() >= m * d && y.len() >= m * d);
            let t = pool.threads();
            let chunk = m.div_ceil(t);
            let yp = crate::pool::SendPtr(y.as_mut_ptr());
            pool.run(t, &|i| {
                let (r0, r1) = (i * chunk, ((i + 1) * chunk).min(m));
                if r0 < r1 {
                    // SAFETY: rows r0..r1 of y are a range no other task touches (disjoint chunks, bounds asserted)
                    let ys = unsafe {
                        std::slice::from_raw_parts_mut(yp.get().add(r0 * d), (r1 - r0) * d)
                    };
                    layer_norm(&x[r0 * d..r1 * d], r1 - r0, d, w, b, eps, ys);
                }
            });
        }
        None => layer_norm(x, m, d, w, b, eps, y),
    }
}

/// erf(x) for f32: rational minimax approximation on [-4, 4] (the XLA / Eigen float erf; |error| < 2e-7)
#[inline(always)]
pub fn erf(a: f32) -> f32 {
    let x = a.clamp(-4.0, 4.0);
    let x2 = x * x;
    let mut p = x2.mul_add(-2.726_142_3e-10, 2.770_681_4e-8);
    p = x2.mul_add(p, -2.101_024e-6);
    p = x2.mul_add(p, -5.692_506_4e-5);
    p = x2.mul_add(p, -7.349_906_3e-4);
    p = x2.mul_add(p, -2.954_600e-3);
    p = x2.mul_add(p, -1.609_603_3e-2);
    p *= x;
    let mut q = x2.mul_add(-1.456_607_2e-5, -2.133_740_6e-4);
    q = x2.mul_add(q, -1.682_827e-3);
    q = x2.mul_add(q, -7.373_329e-3);
    q = x2.mul_add(q, -1.426_474e-2);
    p / q
}

/// exact GELU: x * 0.5 * (1 + erf(x / sqrt 2))   (PyTorch GeluKernel order)
#[inline(always)]
pub fn gelu(x: f32) -> f32 {
    x * 0.5 * (1.0 + erf(x * std::f32::consts::FRAC_1_SQRT_2))
}

/// GeGLU of `m` rows: g[r, j] = gelu(u[r, j]) * u[r, h + j] for u: [m, 2h]
pub fn geglu(u: &[f32], m: usize, h: usize, g: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is part of the aarch64 baseline; slices hold m rows
        assert!(u.len() >= m * 2 * h && g.len() >= m * h);
        unsafe { super::neon::geglu(u, m, h, g) };
        return;
    }
    #[cfg(target_arch = "x86_64")]
    if super::x86ops::enabled() {
        // SAFETY: AVX2 + FMA checked at run time
        unsafe { super::x86ops::geglu(u, m, h, g) };
        return;
    }
    #[allow(unreachable_code)]
    geglu_portable(u, m, h, g);
}

/// `geglu` with the rows split over a pool (bit-identical: every row is computed by `geglu`); plain `geglu` without a
/// pool or for few rows
pub fn geglu_pool(u: &[f32], m: usize, h: usize, g: &mut [f32], pool: Option<&crate::pool::Pool>) {
    match pool.filter(|p| p.threads() > 1 && m >= 32) {
        Some(pool) => {
            assert!(u.len() >= m * 2 * h && g.len() >= m * h);
            let t = pool.threads();
            let chunk = m.div_ceil(t);
            let gp = crate::pool::SendPtr(g.as_mut_ptr());
            pool.run(t, &|i| {
                let (r0, r1) = (i * chunk, ((i + 1) * chunk).min(m));
                if r0 < r1 {
                    // SAFETY: rows r0..r1 of g are a range no other task touches (disjoint chunks, bounds asserted)
                    let gs = unsafe {
                        std::slice::from_raw_parts_mut(gp.get().add(r0 * h), (r1 - r0) * h)
                    };
                    geglu(&u[r0 * 2 * h..r1 * 2 * h], r1 - r0, h, gs);
                }
            });
        }
        None => geglu(u, m, h, g),
    }
}

/// `gelu_inplace` with the values split over a pool (bit-identical); plain `gelu_inplace` without a pool or for few values
pub fn gelu_inplace_pool(x: &mut [f32], pool: Option<&crate::pool::Pool>) {
    let n = x.len();
    match pool.filter(|p| p.threads() > 1 && n >= 1 << 14) {
        Some(pool) => {
            let t = pool.threads();
            let chunk = n.div_ceil(t).div_ceil(16) * 16;
            let xp = crate::pool::SendPtr(x.as_mut_ptr());
            pool.run(t, &|i| {
                let (a, b) = ((i * chunk).min(n), ((i + 1) * chunk).min(n));
                if a < b {
                    // SAFETY: values a..b are a range no other task touches
                    gelu_inplace(unsafe { std::slice::from_raw_parts_mut(xp.get().add(a), b - a) });
                }
            });
        }
        None => gelu_inplace(x),
    }
}

/// the portable GeGLU (also compiled for AVX2 + FMA by `x86ops`, bit-identical)
#[doc(hidden)]
#[inline(always)]
pub fn geglu_portable(u: &[f32], m: usize, h: usize, g: &mut [f32]) {
    for r in 0..m {
        let ur = &u[r * 2 * h..(r + 1) * 2 * h];
        let gr = &mut g[r * h..(r + 1) * h];
        let (a, b) = ur.split_at(h);
        for j in 0..h {
            gr[j] = gelu(a[j]) * b[j];
        }
    }
}

pub fn gelu_inplace(x: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { super::neon::gelu_inplace(x) };
        return;
    }
    #[cfg(target_arch = "x86_64")]
    if super::x86ops::enabled() {
        // SAFETY: AVX2 + FMA checked at run time
        unsafe { super::x86ops::gelu_inplace(x) };
        return;
    }
    #[allow(unreachable_code)]
    gelu_inplace_portable(x);
}

/// the portable in-place GELU (also compiled for AVX2 + FMA by `x86ops`, bit-identical)
#[doc(hidden)]
#[inline(always)]
pub fn gelu_inplace_portable(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = gelu(*v);
    }
}

/// e^x for f32 (Cephes-style range reduction + degree-6 polynomial; ~1 ulp on the softmax range)
#[inline(always)]
pub fn exp(x: f32) -> f32 {
    let x = x.clamp(-87.336_55, 88.376_26);
    let n = (x * std::f32::consts::LOG2_E).round();
    let r = n.mul_add(-0.693_359_4, x);
    let r = n.mul_add(2.121_944_4e-4, r);
    let mut p = 1.987_569_1e-4f32;
    p = p.mul_add(r, 1.398_199_9e-3);
    p = p.mul_add(r, 8.333_452e-3);
    p = p.mul_add(r, 4.166_579_6e-2);
    p = p.mul_add(r, 1.666_666_5e-1);
    p = p.mul_add(r, 5.000_000_1e-1);
    let r2 = r * r;
    let y = p.mul_add(r2, r) + 1.0;
    let e = ((n as i32 + 127) as u32) << 23;
    y * f32::from_bits(e)
}

/// RoPE on one head vector (rotate-half convention): for i < h/2: (x_i, x_{i+h/2}) -> (x_i c - x_{i+h/2} s, x_{i+h/2} c + x_i s)
#[inline]
pub fn rope_head(x: &mut [f32], cos: &[f32], sin: &[f32]) {
    rope_head_portable(x, cos, sin)
}

/// the portable RoPE of one head (also compiled for AVX2 + FMA by `x86ops`, bit-identical)
#[doc(hidden)]
#[inline(always)]
pub fn rope_head_portable(x: &mut [f32], cos: &[f32], sin: &[f32]) {
    let half = x.len() / 2;
    let (a, b) = x.split_at_mut(half);
    for i in 0..half {
        let (x1, x2) = (a[i], b[i]);
        a[i] = x1 * cos[i] - x2 * sin[i];
        b[i] = x2 * cos[i] + x1 * sin[i];
    }
}

#[inline(always)]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    let mut acc = [0f32; 16];
    let chunks = n / 16;
    for c in 0..chunks {
        for j in 0..16 {
            acc[j] += a[c * 16 + j] * b[c * 16 + j];
        }
    }
    let mut s = 0f32;
    for j in 0..16 {
        s += acc[j];
    }
    for i in chunks * 16..n {
        s += a[i] * b[i];
    }
    s
}

/// four dot products of `q` with four keys at once (shared q loads, independent accumulators)
#[inline(always)]
fn dot4(q: &[f32], k0: &[f32], k1: &[f32], k2: &[f32], k3: &[f32]) -> [f32; 4] {
    let n = q.len();
    let mut a0 = [0f32; 8];
    let mut a1 = [0f32; 8];
    let mut a2 = [0f32; 8];
    let mut a3 = [0f32; 8];
    let chunks = n / 8;
    for c in 0..chunks {
        for j in 0..8 {
            let qv = q[c * 8 + j];
            a0[j] += qv * k0[c * 8 + j];
            a1[j] += qv * k1[c * 8 + j];
            a2[j] += qv * k2[c * 8 + j];
            a3[j] += qv * k3[c * 8 + j];
        }
    }
    let mut out = [
        a0.iter().sum::<f32>(),
        a1.iter().sum::<f32>(),
        a2.iter().sum::<f32>(),
        a3.iter().sum::<f32>(),
    ];
    for i in chunks * 8..n {
        out[0] += q[i] * k0[i];
        out[1] += q[i] * k1[i];
        out[2] += q[i] * k2[i];
        out[3] += q[i] * k3[i];
    }
    out
}

#[inline(always)]
fn axpy(y: &mut [f32], a: f32, x: &[f32]) {
    for i in 0..y.len() {
        y[i] += a * x[i];
    }
}

/// softmax-weighted sum for one query head over keys `idx(j)`, j in 0..nk:
/// out = Σ_j softmax(scale · q·k_j) v_j
#[inline]
fn attend_one<F: Fn(usize) -> usize>(
    q: &[f32],
    kc: &[f32],
    vc: &[f32],
    stride: usize,
    off: usize,
    nk: usize,
    key_row: F,
    scale: f32,
    scores: &mut [f32],
    out: &mut [f32],
) {
    let hd = q.len();
    let mut j = 0;
    while j + 4 <= nk {
        let r = [key_row(j), key_row(j + 1), key_row(j + 2), key_row(j + 3)];
        let s = dot4(
            q,
            &kc[r[0] * stride + off..r[0] * stride + off + hd],
            &kc[r[1] * stride + off..r[1] * stride + off + hd],
            &kc[r[2] * stride + off..r[2] * stride + off + hd],
            &kc[r[3] * stride + off..r[3] * stride + off + hd],
        );
        scores[j] = s[0] * scale;
        scores[j + 1] = s[1] * scale;
        scores[j + 2] = s[2] * scale;
        scores[j + 3] = s[3] * scale;
        j += 4;
    }
    while j < nk {
        let r = key_row(j);
        scores[j] = dot(q, &kc[r * stride + off..r * stride + off + hd]) * scale;
        j += 1;
    }
    let mut mx = f32::NEG_INFINITY;
    for &s in &scores[..nk] {
        mx = mx.max(s);
    }
    let mut sum = 0f32;
    for s in scores[..nk].iter_mut() {
        *s = exp(*s - mx);
        sum += *s;
    }
    let inv = 1.0 / sum;
    out.fill(0.0);
    for j in 0..nk {
        let r = key_row(j);
        axpy(
            out,
            scores[j] * inv,
            &vc[r * stride + off..r * stride + off + hd],
        );
    }
}

/// causal self-attention for query rows at absolute positions `pos[i]` (q rows already RoPE'd, layout [nq, heads*hd]),
/// over the cache rows 0..=pos (or the last `window` of them). out: [nq, heads*hd]
#[allow(clippy::too_many_arguments)]
pub fn self_attention(
    q: &[f32],
    pos: &[usize],
    kv: &crate::model::LayerKv,
    heads: usize,
    hd: usize,
    window: Option<usize>,
    scores: &mut Vec<f32>,
    out: &mut [f32],
) {
    let d = heads * hd;
    let scale = 1.0 / (hd as f32).sqrt();
    let maxk = pos.iter().copied().max().map_or(0, |p| p + 1);
    let lo_of = |p: usize| match window {
        Some(w) => (p + 1).saturating_sub(w),
        None => 0,
    };
    #[cfg(target_arch = "aarch64")]
    if hd == 64 {
        if scores.len() < 8 * (maxk + 24) {
            scores.resize(8 * (maxk + 24), 0.0);
        }
        let kts = kv.kts();
        let kt = kv.kt.as_slice();
        let vc = kv.v.as_slice();
        for h in 0..heads {
            let mut i = 0;
            while i < pos.len() {
                let bq = (pos.len() - i).min(4);
                let mut lo_r = [0usize; 4];
                let mut hi_r = [0usize; 4];
                let mut qp = [std::ptr::null::<f32>(); 4];
                let mut op = [std::ptr::null_mut::<f32>(); 4];
                for r in 0..4 {
                    let rr = i + r.min(bq - 1);
                    lo_r[r] = lo_of(pos[rr]);
                    hi_r[r] = pos[rr];
                    qp[r] = q[rr * d + h * hd..].as_ptr();
                    op[r] = out[rr * d + h * hd..].as_mut_ptr();
                }
                let lo = lo_r[..bq].iter().copied().min().unwrap();
                let hi = hi_r[..bq].iter().copied().max().unwrap();
                // SAFETY: keys lo..=hi (+ up to 15 slack columns) exist in the transposed cache, values in rows lo..=hi.
                unsafe {
                    super::neon::attend_block64_t(
                        qp,
                        bq,
                        kt[h * hd * kts..].as_ptr(),
                        kts,
                        vc[h * hd..].as_ptr(),
                        d,
                        lo,
                        hi,
                        lo_r,
                        hi_r,
                        scale,
                        scores.as_mut_ptr(),
                        op,
                    );
                }
                i += bq;
            }
        }
        return;
    }
    #[cfg(target_arch = "x86_64")]
    if hd == 64 && kv.blocked {
        // round 2: blocked keys (LayerKv::set_layout chose them, so AVX2 + FMA are enabled)
        let need = super::x86ops::attn_scratch2(maxk);
        if scores.len() < need {
            scores.resize(need, 0.0);
        }
        assert!(q.len() >= pos.len() * d && out.len() >= pos.len() * d);
        let op = out.as_mut_ptr();
        for h in 0..heads {
            // SAFETY: AVX2 + FMA enabled (blocked layout); see x86_attention_head_r2
            unsafe { x86_attention_head_r2(h, q, pos, kv, heads, hd, window, scale, scores, op) };
        }
        return;
    }
    #[cfg(target_arch = "x86_64")]
    if hd == 64 && super::x86ops::enabled() {
        let need = super::x86ops::attn_scratch(maxk);
        if scores.len() < need {
            scores.resize(need, 0.0);
        }
        let op = out.as_mut_ptr();
        for h in 0..heads {
            // SAFETY: AVX2 + FMA checked; see x86_attention_head
            unsafe { x86_attention_head(h, q, pos, kv, heads, hd, window, scale, scores, op) };
        }
        return;
    }
    let _ = (d, &lo_of, maxk, scale);
    self_attention_portable(q, pos, kv, heads, hd, window, scores, out)
}

/// one head of the x86 AVX2 self-attention (head_dim 64): query blocks of 16 rows; writes columns h*64..h*64+64 of the
/// output rows through `out` (the base of [pos.len(), heads * 64])
///
/// # Safety
/// AVX2 + FMA available; `scores` holds `x86ops::attn_scratch(max pos + 1)` floats; `out` is valid for pos.len() rows of
/// heads * 64 floats and no other thread accesses this head's columns meanwhile.
#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
unsafe fn x86_attention_head(
    h: usize,
    q: &[f32],
    pos: &[usize],
    kv: &crate::model::LayerKv,
    heads: usize,
    hd: usize,
    window: Option<usize>,
    scale: f32,
    scores: &mut [f32],
    out: *mut f32,
) {
    use super::x86ops::QB;
    let d = heads * hd;
    let kts = kv.kts();
    let kt = kv.kt.as_slice();
    let vc = kv.v.as_slice();
    let lo_of = |p: usize| match window {
        Some(w) => (p + 1).saturating_sub(w),
        None => 0,
    };
    let mut i = 0;
    while i < pos.len() {
        let bq = (pos.len() - i).min(QB);
        let mut lo_r = [0usize; QB];
        let mut hi_r = [0usize; QB];
        let mut qp = [std::ptr::null::<f32>(); QB];
        let mut op = [std::ptr::null_mut::<f32>(); QB];
        for r in 0..QB {
            let rr = i + r.min(bq - 1);
            lo_r[r] = lo_of(pos[rr]);
            hi_r[r] = pos[rr];
            qp[r] = q[rr * d + h * hd..].as_ptr();
            op[r] = out.add(rr * d + h * hd);
        }
        let lo = lo_r[..bq].iter().copied().min().unwrap();
        let hi = hi_r[..bq].iter().copied().max().unwrap();
        // keys lo..=hi (+ up to 15 slack columns, kts = rows + 16) exist in the transposed cache and values in rows lo..=hi
        super::x86ops::attend_block64(
            &qp,
            bq,
            kt[h * hd * kts..].as_ptr(),
            kts,
            vc[h * hd..].as_ptr(),
            d,
            lo,
            hi,
            &lo_r,
            &hi_r,
            scale,
            scores.as_mut_ptr(),
            &op,
        );
        i += bq;
    }
}

/// one head of the round-2 x86 self-attention over the blocked keys (head_dim 64): query blocks of `QB2` rows (a
/// multiple of 4: the P·V tiles are those of round 1); writes columns h*64..h*64+64 of the output rows through `out`
///
/// # Safety
/// AVX2 + FMA available; `kv` in the blocked layout with rows 0..=max(pos) stored; `scores` holds
/// `x86ops::attn_scratch2(max pos + 1)` floats; `out` is valid for pos.len() rows of heads * 64 floats and no other thread
/// accesses this head's columns meanwhile.
#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
unsafe fn x86_attention_head_r2(
    h: usize,
    q: &[f32],
    pos: &[usize],
    kv: &crate::model::LayerKv,
    heads: usize,
    hd: usize,
    window: Option<usize>,
    scale: f32,
    scores: &mut [f32],
    out: *mut f32,
) {
    use super::x86ops::QB2;
    let d = heads * hd;
    debug_assert!(kv.blocked && kv.heads == heads && hd == 64);
    let kb = kv.kblk.as_slice()[h * kv.kblk_head_stride()..].as_ptr();
    // values: this head's contiguous rows (head-major cache, the default) or its columns of the row-major cache
    let (vc, vs) = if kv.vhead {
        (kv.v.as_slice()[h * kv.vhead_stride()..].as_ptr(), hd)
    } else {
        (kv.v.as_slice()[h * hd..].as_ptr(), d)
    };
    let lo_of = |p: usize| match window {
        Some(w) => (p + 1).saturating_sub(w),
        None => 0,
    };
    let mut i = 0;
    while i < pos.len() {
        let bq = (pos.len() - i).min(QB2);
        let mut lo_r = [0usize; QB2];
        let mut hi_r = [0usize; QB2];
        let mut qp = [std::ptr::null::<f32>(); QB2];
        let mut op = [std::ptr::null_mut::<f32>(); QB2];
        for r in 0..QB2 {
            let rr = i + r.min(bq - 1);
            lo_r[r] = lo_of(pos[rr]);
            hi_r[r] = pos[rr];
            qp[r] = q[rr * d + h * hd..].as_ptr();
            op[r] = out.add(rr * d + h * hd);
        }
        let lo = lo_r[..bq].iter().copied().min().unwrap();
        let hi = hi_r[..bq].iter().copied().max().unwrap();
        assert!(hi < kv.rows);
        super::x86ops::attend_block64_r2(
            &qp, bq, kb, vc, vs, lo, hi, &lo_r, &hi_r, scale, scores, &op,
        );
        i += bq;
    }
}

/// per-participant scratch of the pooled attention (scores; bf16 query / probability copies on the aarch64 int8 path)
#[derive(Default)]
pub struct AttnScratch {
    pub scores: Vec<f32>,
    pub qb: Vec<u16>,
    pub pb: Vec<u16>,
}

/// `self_attention` with the heads spread over a pool (bit-identical: each head is computed exactly as single-threaded).
/// Returns false, doing nothing, when this CPU / head size has no parallel path (x86 AVX2 + FMA, head_dim 64 only); the
/// caller then runs `self_attention`.
#[allow(clippy::too_many_arguments)]
pub fn self_attention_pool(
    q: &[f32],
    pos: &[usize],
    kv: &crate::model::LayerKv,
    heads: usize,
    hd: usize,
    window: Option<usize>,
    pool: &crate::pool::Pool,
    tls: &mut Vec<std::sync::Mutex<AttnScratch>>,
    out: &mut [f32],
) -> bool {
    #[cfg(target_arch = "aarch64")]
    if hd == 64 && pool.threads() > 1 {
        // the serial aarch64 f32 path of `self_attention`, one head per task (not yet run on aarch64 hardware)
        let d = heads * hd;
        assert!(q.len() >= pos.len() * d && out.len() >= pos.len() * d);
        let scale = 1.0 / (hd as f32).sqrt();
        let maxk = pos.iter().copied().max().map_or(0, |p| p + 1);
        let lo_of = |p: usize| match window {
            Some(w) => (p + 1).saturating_sub(w),
            None => 0,
        };
        let t = pool.threads();
        if tls.len() < t {
            tls.resize_with(t, Default::default);
        }
        let tls = &*tls;
        let base = crate::pool::SendPtr(out.as_mut_ptr());
        let kts = kv.kts();
        let kt = kv.kt.as_slice();
        let vc = kv.v.as_slice();
        pool.run(heads, &|h| {
            let mut g = tls[h % t].lock().unwrap_or_else(|e| e.into_inner());
            let scores = &mut g.scores;
            if scores.len() < 8 * (maxk + 24) {
                scores.resize(8 * (maxk + 24), 0.0);
            }
            let ob = base.get();
            let mut i = 0;
            while i < pos.len() {
                let bq = (pos.len() - i).min(4);
                let mut lo_r = [0usize; 4];
                let mut hi_r = [0usize; 4];
                let mut qp = [std::ptr::null::<f32>(); 4];
                let mut op = [std::ptr::null_mut::<f32>(); 4];
                for r in 0..4 {
                    let rr = i + r.min(bq - 1);
                    lo_r[r] = lo_of(pos[rr]);
                    hi_r[r] = pos[rr];
                    qp[r] = q[rr * d + h * hd..].as_ptr();
                    // SAFETY: row rr < pos.len() of the [pos.len(), d] output (asserted); head h's columns only
                    op[r] = unsafe { ob.add(rr * d + h * hd) };
                }
                let lo = lo_r[..bq].iter().copied().min().unwrap();
                let hi = hi_r[..bq].iter().copied().max().unwrap();
                // SAFETY: keys lo..=hi (+ up to 15 slack columns) exist in the transposed cache, values in rows lo..=hi.
                unsafe {
                    super::neon::attend_block64_t(
                        qp,
                        bq,
                        kt[h * hd * kts..].as_ptr(),
                        kts,
                        vc[h * hd..].as_ptr(),
                        d,
                        lo,
                        hi,
                        lo_r,
                        hi_r,
                        scale,
                        scores.as_mut_ptr(),
                        op,
                    );
                }
                i += bq;
            }
        });
        return true;
    }
    #[cfg(target_arch = "x86_64")]
    if hd == 64 && kv.blocked && pool.threads() > 1 {
        let d = heads * hd;
        assert!(q.len() >= pos.len() * d && out.len() >= pos.len() * d);
        let scale = 1.0 / (hd as f32).sqrt();
        let maxk = pos.iter().copied().max().map_or(0, |p| p + 1);
        let need = super::x86ops::attn_scratch2(maxk);
        let t = pool.threads();
        if tls.len() < t {
            tls.resize_with(t, Default::default);
        }
        let tls = &*tls;
        let op = crate::pool::SendPtr(out.as_mut_ptr());
        pool.run(heads, &|h| {
            let mut g = tls[h % t].lock().unwrap_or_else(|e| e.into_inner());
            let sc = &mut g.scores;
            if sc.len() < need {
                sc.resize(need, 0.0);
            }
            // SAFETY: AVX2 + FMA enabled (blocked layout); head h writes only its own columns of the output rows
            unsafe { x86_attention_head_r2(h, q, pos, kv, heads, hd, window, scale, sc, op.get()) };
        });
        return true;
    }
    #[cfg(target_arch = "x86_64")]
    if hd == 64 && super::x86ops::enabled() && pool.threads() > 1 {
        let d = heads * hd;
        assert!(q.len() >= pos.len() * d && out.len() >= pos.len() * d);
        let scale = 1.0 / (hd as f32).sqrt();
        let maxk = pos.iter().copied().max().map_or(0, |p| p + 1);
        let need = super::x86ops::attn_scratch(maxk);
        let t = pool.threads();
        if tls.len() < t {
            tls.resize_with(t, Default::default);
        }
        let tls = &*tls;
        let op = crate::pool::SendPtr(out.as_mut_ptr());
        pool.run(heads, &|h| {
            let mut g = tls[h % t].lock().unwrap_or_else(|e| e.into_inner());
            let sc = &mut g.scores;
            if sc.len() < need {
                sc.resize(need, 0.0);
            }
            // SAFETY: AVX2 + FMA checked; head h writes only its own columns of the output rows (bounds asserted)
            unsafe { x86_attention_head(h, q, pos, kv, heads, hd, window, scale, sc, op.get()) };
        });
        return true;
    }
    let _ = (q, pos, kv, heads, hd, window, pool, tls, out);
    false
}

/// the portable self-attention (any CPU, any head size): one query and head at a time
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn self_attention_portable(
    q: &[f32],
    pos: &[usize],
    kv: &crate::model::LayerKv,
    heads: usize,
    hd: usize,
    window: Option<usize>,
    scores: &mut Vec<f32>,
    out: &mut [f32],
) {
    let d = heads * hd;
    let scale = 1.0 / (hd as f32).sqrt();
    let maxk = pos.iter().copied().max().map_or(0, |p| p + 1);
    let lo_of = |p: usize| match window {
        Some(w) => (p + 1).saturating_sub(w),
        None => 0,
    };
    let (kc, vc) = (kv.k.as_slice(), kv.v.as_slice());
    if scores.len() < maxk {
        scores.resize(maxk, 0.0);
    }
    for (i, &p) in pos.iter().enumerate() {
        let lo = lo_of(p);
        let nk = p + 1 - lo;
        for h in 0..heads {
            let off = h * hd;
            attend_one(
                &q[i * d + off..i * d + off + hd],
                kc,
                vc,
                d,
                off,
                nk,
                |j| lo + j,
                scale,
                &mut scores[..nk],
                &mut out[i * d + off..i * d + off + hd],
            );
        }
    }
}

/// bf16 (BFMMLA) variant of `self_attention` for the int8 path (aarch64 FEAT_BF16, head_dim 64): reads the bf16 copies
/// of the cache
#[cfg(target_arch = "aarch64")]
#[allow(clippy::too_many_arguments)]
pub fn self_attention_bf16(
    q: &[f32],
    pos: &[usize],
    kv: &crate::model::LayerKv,
    heads: usize,
    hd: usize,
    window: Option<usize>,
    scores: &mut Vec<f32>,
    qb: &mut Vec<u16>,
    pb: &mut Vec<u16>,
    out: &mut [f32],
) {
    assert!(hd == 64 && kv.bf16);
    let d = heads * hd;
    let scale = 1.0 / (hd as f32).sqrt();
    let maxk = pos.iter().copied().max().map_or(0, |p| p + 1);
    if scores.len() < 4 * (maxk + 24) {
        scores.resize(4 * (maxk + 24), 0.0);
    }
    if qb.len() < 256 {
        qb.resize(256, 0);
    }
    // two query pairs x ceil(nk / 8) * 2 chunks x 8 values
    if pb.len() < 4 * (maxk + 24) {
        pb.resize(4 * (maxk + 24), 0);
    }
    let lo_of = |p: usize| match window {
        Some(w) => (p + 1).saturating_sub(w),
        None => 0,
    };
    let (pc, cc) = (kv.pair_cap(), kv.chunk_cap());
    let kb = kv.kb.as_slice();
    let vb = kv.vb.as_slice();
    for h in 0..heads {
        let mut i = 0;
        while i < pos.len() {
            let bq = (pos.len() - i).min(4);
            let mut lo_r = [0usize; 4];
            let mut hi_r = [0usize; 4];
            let mut qp = [std::ptr::null::<f32>(); 4];
            let mut op = [std::ptr::null_mut::<f32>(); 4];
            for r in 0..4 {
                let rr = i + r.min(bq - 1);
                lo_r[r] = lo_of(pos[rr]);
                hi_r[r] = pos[rr];
                qp[r] = q[rr * d + h * hd..].as_ptr();
                op[r] = out[rr * d + h * hd..].as_mut_ptr();
            }
            let lo = lo_r[..bq].iter().copied().min().unwrap();
            let hi = hi_r[..bq].iter().copied().max().unwrap();
            // SAFETY: bf16 caches hold rows/2 + 16 key pairs and rows/4 + 8 value chunks per head (slack beyond hi).
            unsafe {
                super::neon::attend_block64_bf16(
                    qp,
                    bq,
                    kb[h * pc * 128..].as_ptr(),
                    vb[h * cc * 256..].as_ptr(),
                    lo,
                    hi,
                    lo_r,
                    hi_r,
                    scale,
                    scores.as_mut_ptr(),
                    qb.as_mut_ptr(),
                    pb.as_mut_ptr(),
                    op,
                );
            }
            i += bq;
        }
    }
}

/// `self_attention_bf16` with the heads spread over a pool: each head runs exactly the serial code with its own scratch
/// (bit-identical by construction; compiled for aarch64, not yet run on aarch64 hardware)
#[cfg(target_arch = "aarch64")]
#[allow(clippy::too_many_arguments)]
pub fn self_attention_bf16_pool(
    q: &[f32],
    pos: &[usize],
    kv: &crate::model::LayerKv,
    heads: usize,
    hd: usize,
    window: Option<usize>,
    pool: &crate::pool::Pool,
    tls: &mut Vec<std::sync::Mutex<AttnScratch>>,
    out: &mut [f32],
) -> bool {
    if !(hd == 64 && kv.bf16 && pool.threads() > 1) {
        return false;
    }
    let d = heads * hd;
    assert!(q.len() >= pos.len() * d && out.len() >= pos.len() * d);
    let scale = 1.0 / (hd as f32).sqrt();
    let maxk = pos.iter().copied().max().map_or(0, |p| p + 1);
    let lo_of = |p: usize| match window {
        Some(w) => (p + 1).saturating_sub(w),
        None => 0,
    };
    let t = pool.threads();
    if tls.len() < t {
        tls.resize_with(t, Default::default);
    }
    let tls = &*tls;
    let base = crate::pool::SendPtr(out.as_mut_ptr());
    let (pc, cc) = (kv.pair_cap(), kv.chunk_cap());
    let kb = kv.kb.as_slice();
    let vb = kv.vb.as_slice();
    pool.run(heads, &|h| {
        let mut g = tls[h % t].lock().unwrap_or_else(|e| e.into_inner());
        let AttnScratch { scores, qb, pb } = &mut *g;
        if scores.len() < 4 * (maxk + 24) {
            scores.resize(4 * (maxk + 24), 0.0);
        }
        if qb.len() < 256 {
            qb.resize(256, 0);
        }
        if pb.len() < 4 * (maxk + 24) {
            pb.resize(4 * (maxk + 24), 0);
        }
        let ob = base.get();
        let mut i = 0;
        while i < pos.len() {
            let bq = (pos.len() - i).min(4);
            let mut lo_r = [0usize; 4];
            let mut hi_r = [0usize; 4];
            let mut qp = [std::ptr::null::<f32>(); 4];
            let mut op = [std::ptr::null_mut::<f32>(); 4];
            for r in 0..4 {
                let rr = i + r.min(bq - 1);
                lo_r[r] = lo_of(pos[rr]);
                hi_r[r] = pos[rr];
                qp[r] = q[rr * d + h * hd..].as_ptr();
                // SAFETY: row rr < pos.len() of the [pos.len(), d] output (asserted); head h's columns only
                op[r] = unsafe { ob.add(rr * d + h * hd) };
            }
            let lo = lo_r[..bq].iter().copied().min().unwrap();
            let hi = hi_r[..bq].iter().copied().max().unwrap();
            // SAFETY: bf16 caches hold rows/2 + 16 key pairs and rows/4 + 8 value chunks per head (slack beyond hi).
            unsafe {
                super::neon::attend_block64_bf16(
                    qp,
                    bq,
                    kb[h * pc * 128..].as_ptr(),
                    vb[h * cc * 256..].as_ptr(),
                    lo,
                    hi,
                    lo_r,
                    hi_r,
                    scale,
                    scores.as_mut_ptr(),
                    qb.as_mut_ptr(),
                    pb.as_mut_ptr(),
                    op,
                );
            }
            i += bq;
        }
    });
    true
}

/// cross-attention of `nq` query rows over the visible memory / register tokens (compact K | V rows). out: [nq, heads*hd]
pub fn cross_attention(
    q: &[f32],
    nq: usize,
    kvs: &crate::model::CrossKv,
    heads: usize,
    hd: usize,
    scores: &mut Vec<f32>,
    out: &mut [f32],
) {
    let d = heads * hd;
    let scale = 1.0 / (hd as f32).sqrt();
    let nk = kvs.ident.len();
    if nk == 0 {
        out[..nq * d].fill(0.0);
        return;
    }
    #[cfg(target_arch = "aarch64")]
    if hd == 64 {
        if scores.len() < 8 * (nk + 24) {
            scores.resize(8 * (nk + 24), 0.0);
        }
        let kts = nk + 16;
        for h in 0..heads {
            let mut i = 0;
            while i < nq {
                let bq = (nq - i).min(4);
                let mut qp = [std::ptr::null::<f32>(); 4];
                let mut op = [std::ptr::null_mut::<f32>(); 4];
                for r in 0..4 {
                    let rr = i + r.min(bq - 1);
                    qp[r] = q[rr * d + h * hd..].as_ptr();
                    op[r] = out[rr * d + h * hd..].as_mut_ptr();
                }
                // SAFETY: kt holds d rows of nk + 16 columns; compact V rows have stride 2d.
                unsafe {
                    super::neon::attend_block64_t(
                        qp,
                        bq,
                        kvs.kt[h * hd * kts..].as_ptr(),
                        kts,
                        kvs.compact[d + h * hd..].as_ptr(),
                        2 * d,
                        0,
                        nk - 1,
                        [0; 4],
                        [nk - 1; 4],
                        scale,
                        scores.as_mut_ptr(),
                        op,
                    );
                }
                i += bq;
            }
        }
        return;
    }
    if scores.len() < nk {
        scores.resize(nk, 0.0);
    }
    let (k, v) = (&kvs.compact[..], &kvs.compact[d..]);
    for i in 0..nq {
        for h in 0..heads {
            let off = h * hd;
            attend_one(
                &q[i * d + off..i * d + off + hd],
                k,
                v,
                2 * d,
                off,
                nk,
                |j| j,
                scale,
                &mut scores[..nk],
                &mut out[i * d + off..i * d + off + hd],
            );
        }
    }
}

/// dot product of two equal-length f32 vectors (pointer logits)
pub fn dotf(a: &[f32], b: &[f32]) -> f32 {
    dot(a, b)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// f64 reference: Maclaurin series for |x| <= 3, Lentz continued fraction of erfc beyond
    fn erf64(x: f64) -> f64 {
        let z = x.abs();
        let v = if z <= 3.0 {
            let (mut term, mut sum, mut n) = (z, z, 0.0f64);
            loop {
                n += 1.0;
                term *= -z * z / n;
                let t = term / (2.0 * n + 1.0);
                sum += t;
                if t.abs() < 1e-17 {
                    break;
                }
            }
            sum * 2.0 / std::f64::consts::PI.sqrt()
        } else {
            // erfc(z) = exp(-z²)/sqrt(pi) * 1/(z + 1/2/(z + 1/(z + 3/2/(z + ...))))
            let mut f = z;
            for k in (1..200).rev() {
                f = z + (k as f64 / 2.0) / f;
            }
            1.0 - (-z * z).exp() / std::f64::consts::PI.sqrt() / f
        };
        if x < 0.0 {
            -v
        } else {
            v
        }
    }

    #[test]
    fn erf_accuracy() {
        let mut worst = 0f64;
        let mut x = -6.0f64;
        while x <= 6.0 {
            let e = (erf(x as f32) as f64 - erf64(x)).abs();
            worst = worst.max(e);
            x += 1e-3;
        }
        // the reference itself is accurate to ~1.2e-7
        assert!(worst < 4e-7, "erf max error {worst}");
    }

    #[test]
    fn exp_accuracy() {
        let mut worst = 0f64;
        let mut x = -80.0f64;
        while x <= 80.0 {
            let r = (x as f32 as f64).exp();
            let e = ((exp(x as f32) as f64 - r) / r).abs();
            worst = worst.max(e);
            x += 7e-3;
        }
        assert!(worst < 3e-7, "exp max rel error {worst}");
    }
}
