//! Bitvectors with select and rank samples.
//!
//! Bit `t` is bit `t & 7` of byte `t >> 3`, which on little-endian word access is bit `t & 63` of
//! word `t >> 6`. The words are loaded once, so every query works on `u64`s.

/// One select sample per this many ones; one rank sample per this many bits.
pub const SAMPLE_SHIFT: u32 = 9;
pub const SAMPLE: u64 = 1 << SAMPLE_SHIFT;

#[derive(Debug, Clone, Default)]
pub struct Bits {
    words: Vec<u64>,
    nbits: u64,
}

impl Bits {
    /// `bytes` is the packed section; it must hold at least `nbits` bits.
    pub fn from_le_bytes(bytes: &[u8], nbits: u64) -> Option<Self> {
        if (bytes.len() as u64) < nbits.div_ceil(8) {
            return None;
        }
        let mut words = Vec::with_capacity(bytes.len().div_ceil(8));
        for chunk in bytes.chunks(8) {
            let mut w = [0u8; 8];
            w[..chunk.len()].copy_from_slice(chunk);
            words.push(u64::from_le_bytes(w));
        }
        Some(Self { words, nbits })
    }

    pub fn len(&self) -> u64 {
        self.nbits
    }

    pub fn is_empty(&self) -> bool {
        self.nbits == 0
    }

    #[inline]
    fn word(&self, i: u64) -> u64 {
        self.words.get(i as usize).copied().unwrap_or(0)
    }

    pub fn get(&self, t: u64) -> bool {
        self.word(t >> 6) >> (t & 63) & 1 == 1
    }

    /// Bit position of the one with index `j`, using `samples` (one per 512 ones). `None` when the
    /// bitvector holds fewer ones, which only a damaged directory can cause.
    pub fn select1(&self, samples: &[u64], j: u64) -> Option<u64> {
        let s = (j >> SAMPLE_SHIFT) as usize;
        let mut t = *samples.get(s)?;
        let mut rem = j & (SAMPLE - 1);
        while t < self.nbits {
            let w = self.word(t >> 6) >> (t & 63);
            let avail = 64 - (t & 63);
            let w = if t + avail > self.nbits {
                w & ((1u64 << (self.nbits - t)) - 1)
            } else {
                w
            };
            let c = w.count_ones() as u64;
            if c > rem {
                return Some(t + select_in_word(w, rem as u32) as u64);
            }
            rem -= c;
            t += avail;
        }
        None
    }

    /// Position of the `q`-th zero, or `nbits` if there is none. Binary search over the select
    /// samples (sample `s` has `S[s] - 512 s` zeros before it), then a scan.
    pub fn select0(&self, samples: &[u64], q: u64) -> u64 {
        let usable = samples.len().saturating_sub(1);
        let (mut t, mut rem) = (0u64, q);
        if usable > 0 && samples[0] <= q {
            let (mut lo, mut hi) = (0usize, usable - 1);
            while lo < hi {
                let mid = (lo + hi).div_ceil(2);
                if samples[mid] - ((mid as u64) << SAMPLE_SHIFT) <= q {
                    lo = mid;
                } else {
                    hi = mid - 1;
                }
            }
            t = samples[lo];
            rem = q - (t - ((lo as u64) << SAMPLE_SHIFT));
        }
        while t < self.nbits {
            let avail = 64 - (t & 63);
            let span = avail.min(self.nbits - t);
            let w = !(self.word(t >> 6) >> (t & 63));
            let w = if span == 64 {
                w
            } else {
                w & ((1u64 << span) - 1)
            };
            let c = w.count_ones() as u64;
            if c > rem {
                return t + select_in_word(w, rem as u32) as u64;
            }
            rem -= c;
            t += span;
        }
        self.nbits
    }

    /// Position of the last zero below `pos`.
    pub fn zero_before(&self, pos: u64) -> Option<u64> {
        self.last_before(pos, true)
    }

    /// Position of the last one below `pos`.
    pub fn one_before(&self, pos: u64) -> Option<u64> {
        self.last_before(pos, false)
    }

    fn last_before(&self, pos: u64, invert: bool) -> Option<u64> {
        let mut t = pos;
        while t > 0 {
            let lo = (t - 1) & !63;
            let mut w = self.word(lo >> 6);
            if invert {
                w = !w;
            }
            let span = t - lo;
            if span < 64 {
                w &= (1u64 << span) - 1;
            }
            if w != 0 {
                return Some(lo + 63 - w.leading_zeros() as u64);
            }
            t = lo;
        }
        None
    }

    /// Position of the first zero at or after `pos`, or `nbits`.
    pub fn zero_from(&self, pos: u64) -> u64 {
        let mut t = pos;
        while t < self.nbits {
            let avail = 64 - (t & 63);
            let w = !(self.word(t >> 6) >> (t & 63));
            let w = if avail == 64 {
                w
            } else {
                w & ((1u64 << avail) - 1)
            };
            if w != 0 {
                return (t + w.trailing_zeros() as u64).min(self.nbits);
            }
            t += avail;
        }
        self.nbits
    }

    /// Number of ones in `[0, t)`, using `samples` (one per 512 bits).
    pub fn rank1(&self, samples: &[u64], t: u64) -> u64 {
        let b = t >> SAMPLE_SHIFT;
        let mut r = samples.get(b as usize).copied().unwrap_or(0);
        let mut cur = b << SAMPLE_SHIFT;
        while cur < t {
            let span = (t - cur).min(64);
            let w = self.word(cur >> 6);
            r += if span == 64 {
                w.count_ones()
            } else {
                (w & ((1u64 << span) - 1)).count_ones()
            } as u64;
            cur += span;
        }
        r
    }
}

/// Index of the `r`-th set bit of `w`. Callers guarantee it exists.
#[inline]
fn select_in_word(mut w: u64, r: u32) -> u32 {
    for _ in 0..r {
        w &= w - 1;
    }
    w.trailing_zeros()
}

/// Appends bits low to high into packed bytes.
#[derive(Debug, Default)]
pub struct BitWriter {
    bytes: Vec<u8>,
    nbits: u64,
}

impl BitWriter {
    pub fn with_capacity_bits(bits: u64) -> Self {
        Self {
            bytes: Vec::with_capacity(bits.div_ceil(8) as usize),
            nbits: 0,
        }
    }

    pub fn push(&mut self, bit: bool) {
        if self.nbits & 7 == 0 {
            self.bytes.push(0);
        }
        if bit {
            *self.bytes.last_mut().expect("just pushed") |= 1 << (self.nbits & 7);
        }
        self.nbits += 1;
    }

    pub fn push_ones(&mut self, mut n: u64) {
        while n > 0 && self.nbits & 7 != 0 {
            self.push(true);
            n -= 1;
        }
        let whole = n >> 3;
        self.bytes.resize(self.bytes.len() + whole as usize, 0xff);
        self.nbits += whole << 3;
        for _ in 0..(n & 7) {
            self.push(true);
        }
    }

    pub fn nbits(&self) -> u64 {
        self.nbits
    }

    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }
}

/// `counts[p]` ones then one zero for every bucket: `(packed, nbits, select samples + sentinel)`.
pub fn bucket_bitvector(counts: &[u32]) -> (Vec<u8>, u64, Vec<u64>) {
    let total: u64 = counts.iter().map(|&c| c as u64).sum();
    let mut w = BitWriter::with_capacity_bits(total + counts.len() as u64);
    let mut samples = Vec::with_capacity((total >> SAMPLE_SHIFT) as usize + 2);
    let (mut next, mut seen) = (0u64, 0u64);
    for &c in counts {
        let c = c as u64;
        if c != 0 {
            let start = w.nbits();
            let end = seen + c;
            while next < end {
                samples.push(start + (next - seen));
                next += SAMPLE;
            }
            seen = end;
            w.push_ones(c);
        }
        w.push(false);
    }
    let nbits = w.nbits();
    samples.push(nbits);
    (w.into_bytes(), nbits, samples)
}

/// Integers as u32 when every value fits, else u64: `(width, little-endian bytes)`.
pub fn int_array(values: &[u64]) -> (u32, Vec<u8>) {
    if values.iter().all(|&v| v < 1 << 32) {
        let mut out = Vec::with_capacity(values.len() * 4);
        for &v in values {
            out.extend_from_slice(&(v as u32).to_le_bytes());
        }
        (4, out)
    } else {
        let mut out = Vec::with_capacity(values.len() * 8);
        for &v in values {
            out.extend_from_slice(&v.to_le_bytes());
        }
        (8, out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::StdRng, Rng, SeedableRng};

    fn from_bools(v: &[bool]) -> Bits {
        let mut w = BitWriter::default();
        for &b in v {
            w.push(b);
        }
        let n = w.nbits();
        Bits::from_le_bytes(&w.into_bytes(), n).unwrap()
    }

    #[test]
    fn queries_match_a_plain_scan() {
        let mut rng = StdRng::seed_from_u64(7);
        for density in [0.02, 0.5, 0.98] {
            for len in [1usize, 63, 64, 65, 511, 512, 513, 5000] {
                let v: Vec<bool> = (0..len).map(|_| rng.gen_bool(density)).collect();
                let b = from_bools(&v);
                let ones: Vec<u64> = (0..len as u64).filter(|&t| v[t as usize]).collect();
                let zeros: Vec<u64> = (0..len as u64).filter(|&t| !v[t as usize]).collect();
                let mut sel: Vec<u64> = ones.iter().step_by(512).copied().collect();
                sel.push(len as u64);
                for (j, &p) in ones.iter().enumerate() {
                    assert_eq!(b.select1(&sel, j as u64), Some(p));
                }
                assert_eq!(b.select1(&sel, ones.len() as u64), None);
                for (q, &p) in zeros.iter().enumerate() {
                    assert_eq!(b.select0(&sel, q as u64), p, "len {len} q {q}");
                }
                assert_eq!(b.select0(&sel, zeros.len() as u64), len as u64);
                let mut rs = vec![];
                let mut r = 0;
                for t in 0..=len {
                    if t % 512 == 0 {
                        rs.push(r);
                    }
                    if t < len && v[t] {
                        r += 1;
                    }
                }
                let mut r = 0;
                for t in 0..=len as u64 {
                    assert_eq!(b.rank1(&rs, t), r);
                    let want_zero = (0..t).rev().find(|&x| !v[x as usize]);
                    let want_one = (0..t).rev().find(|&x| v[x as usize]);
                    assert_eq!(b.zero_before(t), want_zero);
                    assert_eq!(b.one_before(t), want_one);
                    let want = (t..len as u64)
                        .find(|&x| !v[x as usize])
                        .unwrap_or(len as u64);
                    assert_eq!(b.zero_from(t), want);
                    if (t as usize) < len && v[t as usize] {
                        r += 1;
                    }
                }
            }
        }
    }

    #[test]
    fn push_ones_equals_single_pushes() {
        let mut a = BitWriter::default();
        let mut b = BitWriter::default();
        for n in [0u64, 1, 3, 8, 13, 64, 7] {
            a.push_ones(n);
            a.push(false);
            for _ in 0..n {
                b.push(true);
            }
            b.push(false);
        }
        assert_eq!(a.nbits(), b.nbits());
        assert_eq!(a.into_bytes(), b.into_bytes());
    }
}
