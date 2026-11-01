//! SHA-256 of (shared prefix ‖ suffix) for many suffixes: the manifest keys sha256(item_id + fingerprint + fact_id).
//!
//! * The prefix's full 64-byte blocks are absorbed once (midstate); only the residue + suffix + padding (usually 2
//!   blocks) is compressed per key.
//! * On aarch64 with the SHA-2 extension, 4 keys are compressed in lockstep with the ARMv8 SHA-256 instructions
//!   (`sha256h`, `sha256h2`, `sha256su0`, `sha256su1`). One stream is latency-bound (every 4-round group waits on the
//!   previous group's two SHA instructions); four independent streams fill those pipeline slots, so throughput rises
//!   several-fold. Elsewhere (or for odd-sized tails) a portable scalar compression is used.
//! * On x86-64 with AVX2 (no SHA extension on the Broadwell build server), [`PrefixSha256::digest8`] compresses 8 keys in
//!   lockstep, one 32-bit lane per key in 256-bit registers (the FIPS 180-4 rounds written with vector adds, shifts and
//!   logic ops); keys whose padded tails have fewer blocks keep their state from their last block. Same digests.
//! * Digests are bit-identical to `hashlib.sha256(...)` (tested against the sha2 crate).

const K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

const H0: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

/// portable compression of one block
fn compress_scalar(state: &mut [u32; 8], block: &[u8]) {
    let mut w = [0u32; 64];
    for i in 0..16 {
        w[i] = u32::from_be_bytes(block[4 * i..4 * i + 4].try_into().unwrap());
    }
    for i in 16..64 {
        let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
        let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16]
            .wrapping_add(s0)
            .wrapping_add(w[i - 7])
            .wrapping_add(s1);
    }
    let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = *state;
    for i in 0..64 {
        let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
        let ch = (e & f) ^ (!e & g);
        let t1 = h
            .wrapping_add(s1)
            .wrapping_add(ch)
            .wrapping_add(K[i])
            .wrapping_add(w[i]);
        let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
        let maj = (a & b) ^ (a & c) ^ (b & c);
        let t2 = s0.wrapping_add(maj);
        h = g;
        g = f;
        f = e;
        e = d.wrapping_add(t1);
        d = c;
        c = b;
        b = a;
        a = t1.wrapping_add(t2);
    }
    for (s, v) in state.iter_mut().zip([a, b, c, d, e, f, g, h]) {
        *s = s.wrapping_add(v);
    }
}

#[cfg(target_arch = "aarch64")]
mod arm {
    use super::K;
    use core::arch::aarch64::*;

    /// one block, one stream (ARMv8 SHA-256 instructions)
    #[target_feature(enable = "sha2")]
    pub unsafe fn compress1(state: &mut [u32; 8], block: &[u8]) {
        let mut st = [*state];
        compress_n::<1>(&mut st, [block]);
        *state = st[0];
    }

    /// N independent blocks in lockstep (N streams; the compiler interleaves the independent dependency chains)
    #[target_feature(enable = "sha2")]
    pub unsafe fn compress_n<const N: usize>(states: &mut [[u32; 8]; N], blocks: [&[u8]; N]) {
        let mut abcd: [uint32x4_t; N] = [vdupq_n_u32(0); N];
        let mut efgh: [uint32x4_t; N] = [vdupq_n_u32(0); N];
        let mut m0: [uint32x4_t; N] = [vdupq_n_u32(0); N];
        let mut m1 = m0;
        let mut m2 = m0;
        let mut m3 = m0;
        for l in 0..N {
            abcd[l] = vld1q_u32(states[l].as_ptr());
            efgh[l] = vld1q_u32(states[l].as_ptr().add(4));
            let b = blocks[l].as_ptr();
            m0[l] = vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(b)));
            m1[l] = vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(b.add(16))));
            m2[l] = vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(b.add(32))));
            m3[l] = vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(b.add(48))));
        }
        let (abcd0, efgh0) = (abcd, efgh);
        macro_rules! group {
            ($m:ident, $k:expr) => {{
                let kv = vld1q_u32(K.as_ptr().add($k * 4));
                for l in 0..N {
                    let wk = vaddq_u32($m[l], kv);
                    let a = abcd[l];
                    abcd[l] = vsha256hq_u32(a, efgh[l], wk);
                    efgh[l] = vsha256h2q_u32(efgh[l], a, wk);
                }
            }};
        }
        macro_rules! sched {
            ($a:ident, $b:ident, $c:ident, $d:ident) => {{
                for l in 0..N {
                    $a[l] = vsha256su1q_u32(vsha256su0q_u32($a[l], $b[l]), $c[l], $d[l]);
                }
            }};
        }
        for r in 0..3 {
            group!(m0, 4 * r);
            sched!(m0, m1, m2, m3);
            group!(m1, 4 * r + 1);
            sched!(m1, m2, m3, m0);
            group!(m2, 4 * r + 2);
            sched!(m2, m3, m0, m1);
            group!(m3, 4 * r + 3);
            sched!(m3, m0, m1, m2);
        }
        group!(m0, 12);
        group!(m1, 13);
        group!(m2, 14);
        group!(m3, 15);
        for l in 0..N {
            vst1q_u32(states[l].as_mut_ptr(), vaddq_u32(abcd[l], abcd0[l]));
            vst1q_u32(states[l].as_mut_ptr().add(4), vaddq_u32(efgh[l], efgh0[l]));
        }
    }
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use super::K;
    use core::arch::x86_64::*;

    /// rotate every lane right by R (L = 32 - R); a macro, so it expands inside the AVX2 function
    macro_rules! rotr {
        ($x:expr, $r:literal, $l:literal) => {
            _mm256_or_si256(_mm256_srli_epi32::<$r>($x), _mm256_slli_epi32::<$l>($x))
        };
    }

    /// 8 independent blocks in lockstep (one u32 lane per stream)
    #[target_feature(enable = "avx2")]
    pub unsafe fn compress8(states: &mut [[u32; 8]; 8], blocks: [&[u8]; 8]) {
        let mut lane = [0u32; 8];
        let mut w = [_mm256_setzero_si256(); 64];
        for (t, wt) in w.iter_mut().enumerate().take(16) {
            for l in 0..8 {
                let b = &blocks[l][4 * t..4 * t + 4];
                lane[l] = u32::from_be_bytes([b[0], b[1], b[2], b[3]]);
            }
            *wt = _mm256_loadu_si256(lane.as_ptr() as *const __m256i);
        }
        for t in 16..64 {
            let x = w[t - 15];
            let s0 = _mm256_xor_si256(
                _mm256_xor_si256(rotr!(x, 7, 25), rotr!(x, 18, 14)),
                _mm256_srli_epi32::<3>(x),
            );
            let y = w[t - 2];
            let s1 = _mm256_xor_si256(
                _mm256_xor_si256(rotr!(y, 17, 15), rotr!(y, 19, 13)),
                _mm256_srli_epi32::<10>(y),
            );
            w[t] = _mm256_add_epi32(
                _mm256_add_epi32(w[t - 16], s0),
                _mm256_add_epi32(w[t - 7], s1),
            );
        }
        let mut v = [_mm256_setzero_si256(); 8];
        for (j, vj) in v.iter_mut().enumerate() {
            for l in 0..8 {
                lane[l] = states[l][j];
            }
            *vj = _mm256_loadu_si256(lane.as_ptr() as *const __m256i);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = v;
        for t in 0..64 {
            let s1 = _mm256_xor_si256(
                _mm256_xor_si256(rotr!(e, 6, 26), rotr!(e, 11, 21)),
                rotr!(e, 25, 7),
            );
            let ch = _mm256_xor_si256(_mm256_and_si256(e, f), _mm256_andnot_si256(e, g));
            let t1 = _mm256_add_epi32(
                _mm256_add_epi32(_mm256_add_epi32(h, s1), ch),
                _mm256_add_epi32(_mm256_set1_epi32(K[t] as i32), w[t]),
            );
            let s0 = _mm256_xor_si256(
                _mm256_xor_si256(rotr!(a, 2, 30), rotr!(a, 13, 19)),
                rotr!(a, 22, 10),
            );
            let maj = _mm256_xor_si256(
                _mm256_xor_si256(_mm256_and_si256(a, b), _mm256_and_si256(a, c)),
                _mm256_and_si256(b, c),
            );
            let t2 = _mm256_add_epi32(s0, maj);
            h = g;
            g = f;
            f = e;
            e = _mm256_add_epi32(d, t1);
            d = c;
            c = b;
            b = a;
            a = _mm256_add_epi32(t1, t2);
        }
        let fin = [a, b, c, d, e, f, g, h];
        for j in 0..8 {
            let s = _mm256_add_epi32(fin[j], v[j]);
            _mm256_storeu_si256(lane.as_mut_ptr() as *mut __m256i, s);
            for l in 0..8 {
                states[l][j] = lane[l];
            }
        }
    }
}

#[inline]
fn has_avx2() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::arch::is_x86_feature_detected!("avx2")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

#[inline]
fn has_sha2() -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        std::arch::is_aarch64_feature_detected!("sha2")
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        false
    }
}

/// SHA-256 with a fixed prefix absorbed once.
#[derive(Clone)]
pub struct PrefixSha256 {
    mid: [u32; 8],
    rest: [u8; 64],
    rest_len: usize,
    prefix_len: u64,
    hw: bool,
    /// x86-64 AVX2 available: [`PrefixSha256::digest8`] runs 8 lanes in lockstep
    avx2: bool,
}

/// the padded tail blocks of one message (at most `MAX_TAIL` blocks; longer messages use the generic path)
const MAX_TAIL: usize = 4;

impl PrefixSha256 {
    pub fn new(parts: &[&[u8]]) -> PrefixSha256 {
        let hw = has_sha2();
        let mut st = H0;
        let mut buf = [0u8; 64];
        let mut n = 0usize;
        let mut total = 0u64;
        for p in parts {
            total += p.len() as u64;
            for &b in *p {
                buf[n] = b;
                n += 1;
                if n == 64 {
                    compress(&mut st, &buf, hw);
                    n = 0;
                }
            }
        }
        PrefixSha256 {
            mid: st,
            rest: buf,
            rest_len: n,
            prefix_len: total,
            hw,
            avx2: has_avx2(),
        }
    }

    /// padded tail (residue ‖ suffix ‖ 0x80 ‖ 0* ‖ bit length) into `out`; returns the number of blocks
    #[inline]
    fn tail(&self, suffix: &[u8], out: &mut [u8; 64 * MAX_TAIL]) -> usize {
        let len = self.rest_len + suffix.len();
        let nblocks = (len + 9).div_ceil(64);
        debug_assert!(nblocks <= MAX_TAIL);
        out[..self.rest_len].copy_from_slice(&self.rest[..self.rest_len]);
        out[self.rest_len..len].copy_from_slice(suffix);
        out[len] = 0x80;
        let end = nblocks * 64;
        for b in &mut out[len + 1..end - 8] {
            *b = 0;
        }
        let bits = (self.prefix_len + suffix.len() as u64) * 8;
        out[end - 8..end].copy_from_slice(&bits.to_be_bytes());
        nblocks
    }

    #[inline]
    fn fits(&self, suffix: &[u8]) -> bool {
        self.rest_len + suffix.len() + 9 <= 64 * MAX_TAIL
    }

    /// sha256(prefix ‖ suffix)
    pub fn digest(&self, suffix: &[u8]) -> [u8; 32] {
        if !self.fits(suffix) {
            return self.digest_long(suffix);
        }
        let mut t = [0u8; 64 * MAX_TAIL];
        let nb = self.tail(suffix, &mut t);
        let mut st = self.mid;
        for b in 0..nb {
            compress(&mut st, &t[64 * b..64 * b + 64], self.hw);
        }
        out(&st)
    }

    fn digest_long(&self, suffix: &[u8]) -> [u8; 32] {
        let mut st = self.mid;
        let mut buf = Vec::with_capacity(self.rest_len + suffix.len() + 72);
        buf.extend_from_slice(&self.rest[..self.rest_len]);
        buf.extend_from_slice(suffix);
        let bits = (self.prefix_len + suffix.len() as u64) * 8;
        buf.push(0x80);
        while buf.len() % 64 != 56 {
            buf.push(0);
        }
        buf.extend_from_slice(&bits.to_be_bytes());
        let (blocks, _) = buf.as_chunks::<64>();
        for c in blocks {
            compress(&mut st, c, self.hw);
        }
        out(&st)
    }

    /// sha256(prefix ‖ s) for 4 suffixes; lanes compressed in lockstep when their tails have equal block counts
    pub fn digest4(&self, s: [&[u8]; 4]) -> [[u8; 32]; 4] {
        #[cfg(target_arch = "aarch64")]
        if self.hw && s.iter().all(|x| self.fits(x)) {
            let mut t = [[0u8; 64 * MAX_TAIL]; 4];
            let mut nb = [0usize; 4];
            for l in 0..4 {
                nb[l] = self.tail(s[l], &mut t[l]);
            }
            if nb.iter().all(|&n| n == nb[0]) {
                let mut st = [self.mid; 4];
                for b in 0..nb[0] {
                    let o = 64 * b;
                    // SAFETY: hw == sha2 feature detected at runtime
                    unsafe {
                        arm::compress_n::<4>(
                            &mut st,
                            [
                                &t[0][o..o + 64],
                                &t[1][o..o + 64],
                                &t[2][o..o + 64],
                                &t[3][o..o + 64],
                            ],
                        )
                    };
                }
                return [out(&st[0]), out(&st[1]), out(&st[2]), out(&st[3])];
            }
        }
        [
            self.digest(s[0]),
            self.digest(s[1]),
            self.digest(s[2]),
            self.digest(s[3]),
        ]
    }

    /// sha256(prefix ‖ s) for 8 suffixes. On x86-64 with AVX2 the 8 lanes are compressed in lockstep (a lane whose padded
    /// tail has fewer blocks keeps the state of its own last block); elsewhere two [`PrefixSha256::digest4`] calls.
    pub fn digest8(&self, s: [&[u8]; 8]) -> [[u8; 32]; 8] {
        #[cfg(not(target_arch = "x86_64"))]
        let _ = self.avx2;
        #[cfg(target_arch = "x86_64")]
        if self.avx2 && s.iter().all(|x| self.fits(x)) {
            let mut t = [[0u8; 64 * MAX_TAIL]; 8];
            let mut nb = [0usize; 8];
            for l in 0..8 {
                nb[l] = self.tail(s[l], &mut t[l]);
            }
            let max_nb = nb.iter().copied().max().unwrap_or(0);
            let mut st = [self.mid; 8];
            for b in 0..max_nb {
                let o = 64 * b;
                let keep = st;
                // SAFETY: avx2 == AVX2 detected at runtime; every lane slice is one 64-byte block of its tail buffer
                // (lanes past their own last block read zeros and get their state back below)
                unsafe {
                    x86::compress8(
                        &mut st,
                        [
                            &t[0][o..o + 64],
                            &t[1][o..o + 64],
                            &t[2][o..o + 64],
                            &t[3][o..o + 64],
                            &t[4][o..o + 64],
                            &t[5][o..o + 64],
                            &t[6][o..o + 64],
                            &t[7][o..o + 64],
                        ],
                    )
                };
                for l in 0..8 {
                    if b >= nb[l] {
                        st[l] = keep[l];
                    }
                }
            }
            return core::array::from_fn(|l| out(&st[l]));
        }
        let a = self.digest4([s[0], s[1], s[2], s[3]]);
        let b = self.digest4([s[4], s[5], s[6], s[7]]);
        [a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3]]
    }

    /// whether the 4-lane hardware path is available
    pub fn hardware(&self) -> bool {
        self.hw
    }
}

#[inline]
fn compress(st: &mut [u32; 8], block: &[u8], hw: bool) {
    #[cfg(target_arch = "aarch64")]
    if hw {
        // SAFETY: hw == sha2 feature detected at runtime
        unsafe { arm::compress1(st, block) };
        return;
    }
    let _ = hw;
    compress_scalar(st, block);
}

#[inline]
fn out(st: &[u32; 8]) -> [u8; 32] {
    let mut o = [0u8; 32];
    for (i, w) in st.iter().enumerate() {
        o[4 * i..4 * i + 4].copy_from_slice(&w.to_be_bytes());
    }
    o
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};

    #[test]
    fn matches_sha2_crate() {
        let prefixes: [&[u8]; 4] = [b"", b"order|", b"world-test-00000/old/before5fa3b225e59c9c169f2986627174e41dce3d006f507947b49d7b91ae273b7dc3", &[b'x'; 200]];
        for p in prefixes {
            let h = PrefixSha256::new(&[p]);
            for n in [
                0usize, 1, 3, 54, 55, 56, 63, 64, 65, 100, 119, 120, 183, 247, 300,
            ] {
                let suffix: Vec<u8> = (0..n).map(|i| (i * 7 + 3) as u8).collect();
                let mut full = p.to_vec();
                full.extend_from_slice(&suffix);
                let want: [u8; 32] = Sha256::digest(&full).into();
                assert_eq!(h.digest(&suffix), want, "prefix {} suffix {n}", p.len());
                let s4 = h.digest4([&suffix, &suffix[..n / 2], &suffix, &suffix[..n / 3]]);
                assert_eq!(s4[0], want);
                let mut f2 = p.to_vec();
                f2.extend_from_slice(&suffix[..n / 2]);
                assert_eq!(s4[1], <[u8; 32]>::from(Sha256::digest(&f2)));
                // 8 lanes of different lengths (mixed tail block counts; the AVX2 path on x86-64)
                let lens = [
                    n,
                    n / 2,
                    n / 3,
                    0,
                    n.saturating_sub(1),
                    (n * 2) / 3,
                    n / 5,
                    1.min(n),
                ];
                let s8 = h.digest8(core::array::from_fn(|l| &suffix[..lens[l]]));
                for l in 0..8 {
                    let mut f = p.to_vec();
                    f.extend_from_slice(&suffix[..lens[l]]);
                    assert_eq!(
                        s8[l],
                        <[u8; 32]>::from(Sha256::digest(&f)),
                        "lane {l} len {}",
                        lens[l]
                    );
                }
            }
        }
        // many fact-id-shaped suffixes, every lane mix of 1- and 2-block tails
        let h = PrefixSha256::new(&[b"lab/q/0123456789abcdef", &[b'f'; 64]]);
        let ids: Vec<String> = (0..64)
            .map(|i| format!("wd5m/Q{}/P{}/Q{}", i * 7919, i % 50, i * 104729 % 999983))
            .collect();
        for c in ids.chunks(8) {
            let s8 = h.digest8(core::array::from_fn(|l| c[l].as_bytes()));
            for l in 0..8 {
                assert_eq!(s8[l], h.digest(c[l].as_bytes()));
            }
        }
        let mut st = H0;
        let mut buf = [0u8; 64];
        buf[0] = 0x80;
        compress_scalar(&mut st, &buf);
        assert_eq!(out(&st), <[u8; 32]>::from(Sha256::digest(b"")));
    }
}
