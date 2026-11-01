//! Posting-list codec: blocks of 128 postings, doc deltas and (tf - 1) bit-packed at the block's maximal width (Lucene /
//! tantivy style); the final partial block as varints ((delta << 1) | (tf == 1), then tf when != 1). Width-specialized
//! unpacking (const generics) lets the compiler unroll and vectorize the hot decode loop.

pub const BLOCK: usize = 128;

#[inline]
pub fn put_varint(out: &mut Vec<u8>, mut v: u64) {
    while v >= 0x80 {
        out.push((v as u8) | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

#[inline]
pub fn get_varint(b: &[u8], i: &mut usize) -> u64 {
    let mut v = 0u64;
    let mut shift = 0;
    loop {
        let c = b[*i];
        *i += 1;
        v |= ((c & 0x7f) as u64) << shift;
        if c < 0x80 {
            return v;
        }
        shift += 7;
    }
}

#[inline]
fn bits_for(max: u32) -> u32 {
    32 - max.leading_zeros()
}

/// Pack 128 values of width `w` (16 * w bytes).
pub fn pack128(vals: &[u32; BLOCK], w: u32, out: &mut Vec<u8>) {
    if w == 0 {
        return;
    }
    let mut acc: u64 = 0;
    let mut n: u32 = 0;
    for &v in vals.iter() {
        acc |= (v as u64) << n;
        n += w;
        while n >= 8 {
            out.push(acc as u8);
            acc >>= 8;
            n -= 8;
        }
    }
    debug_assert_eq!(n, 0);
}

#[inline(always)]
fn unpack_w<const W: u32>(src: &[u8], out: &mut [u32; BLOCK]) {
    if W == 0 {
        *out = [0; BLOCK];
        return;
    }
    let nbytes = 16 * W as usize;
    let src = &src[..nbytes];
    let mask: u64 = (1u64 << W) - 1;
    // value j occupies bits [j*W, j*W + W): one unaligned little-endian u64 window covers it (shift <= 7, W <= 32);
    // with W a constant the loop fully unrolls and the tail branch resolves at compile time
    for (j, o) in out.iter_mut().enumerate() {
        let bit = j * W as usize;
        let byte = bit >> 3;
        let shift = (bit & 7) as u32;
        let word = if byte + 8 <= nbytes {
            u64::from_le_bytes(src[byte..byte + 8].try_into().unwrap())
        } else {
            let mut tmp = [0u8; 8];
            tmp[..nbytes - byte].copy_from_slice(&src[byte..]);
            u64::from_le_bytes(tmp)
        };
        *o = ((word >> shift) & mask) as u32;
    }
}

/// Unpack 128 values of width `w` from `src` (16 * w bytes).
pub fn unpack128(src: &[u8], w: u32, out: &mut [u32; BLOCK]) {
    macro_rules! dispatch {
        ($($n:literal)*) => {
            match w {
                $($n => unpack_w::<$n>(src, out),)*
                _ => panic!("bad bit width {w}"),
            }
        };
    }
    dispatch!(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32);
}

/// Encode one posting list (docs strictly increasing, all >= `lo`; tfs >= 1).
pub fn encode_postings(lo: u32, docs: &[u32], tfs: &[u32], out: &mut Vec<u8>) {
    encode_postings_skips(lo, docs, tfs, out, None)
}

/// Encode one posting list and, for every full block, record its last doc and its start offset (relative to the list's
/// first byte) in `skips` = (last_docs, starts); one extra start = the tail's offset.
pub fn encode_postings_skips(
    lo: u32,
    docs: &[u32],
    tfs: &[u32],
    out: &mut Vec<u8>,
    mut skips: Option<(&mut Vec<u32>, &mut Vec<u32>)>,
) {
    let base = out.len();
    let n = docs.len();
    let mut prev = lo;
    let mut i = 0;
    let mut d = [0u32; BLOCK];
    let mut t = [0u32; BLOCK];
    while i + BLOCK <= n {
        let mut md = 0;
        let mut mt = 0;
        for k in 0..BLOCK {
            let doc = docs[i + k];
            d[k] = doc - prev;
            prev = doc;
            t[k] = tfs[i + k] - 1;
            md |= d[k];
            mt |= t[k];
        }
        let (wd, wt) = (bits_for(md), bits_for(mt));
        if let Some((last, starts)) = skips.as_mut() {
            last.push(prev);
            starts.push((out.len() - base) as u32);
        }
        out.push(wd as u8);
        out.push(wt as u8);
        pack128(&d, wd, out);
        pack128(&t, wt, out);
        i += BLOCK;
    }
    if let Some((_, starts)) = skips.as_mut() {
        starts.push((out.len() - base) as u32);
    }
    while i < n {
        let delta = (docs[i] - prev) as u64;
        prev = docs[i];
        let tf = tfs[i];
        put_varint(out, (delta << 1) | (tf == 1) as u64);
        if tf != 1 {
            put_varint(out, tf as u64);
        }
        i += 1;
    }
}

/// Streaming decoder over one encoded posting list.
pub struct PostingIter<'a> {
    src: &'a [u8],
    pos: usize,
    left: usize,
    prev: u32,
    buf_d: [u32; BLOCK],
    buf_t: [u32; BLOCK],
    buf_len: usize,
    buf_i: usize,
}

impl<'a> PostingIter<'a> {
    pub fn new(src: &'a [u8], lo: u32, count: usize) -> Self {
        PostingIter {
            src,
            pos: 0,
            left: count,
            prev: lo,
            buf_d: [0; BLOCK],
            buf_t: [0; BLOCK],
            buf_len: 0,
            buf_i: 0,
        }
    }

    /// Decode the next full block (if any) into the internal buffers; returns its slices (docs absolute, tfs >= 1).
    #[inline]
    pub fn next_block(&mut self) -> Option<(&[u32], &[u32])> {
        if self.left == 0 {
            return None;
        }
        if self.left >= BLOCK {
            let wd = self.src[self.pos] as u32;
            let wt = self.src[self.pos + 1] as u32;
            self.pos += 2;
            unpack128(&self.src[self.pos..], wd, &mut self.buf_d);
            self.pos += 16 * wd as usize;
            unpack128(&self.src[self.pos..], wt, &mut self.buf_t);
            self.pos += 16 * wt as usize;
            let mut p = self.prev;
            for k in 0..BLOCK {
                p += self.buf_d[k];
                self.buf_d[k] = p;
                self.buf_t[k] += 1;
            }
            self.prev = p;
            self.left -= BLOCK;
            self.buf_len = BLOCK;
        } else {
            let n = self.left;
            for k in 0..n {
                let v = get_varint(self.src, &mut self.pos);
                self.prev += (v >> 1) as u32;
                self.buf_d[k] = self.prev;
                self.buf_t[k] = if v & 1 == 1 {
                    1
                } else {
                    get_varint(self.src, &mut self.pos) as u32
                };
            }
            self.left = 0;
            self.buf_len = n;
        }
        self.buf_i = 0;
        Some((&self.buf_d[..self.buf_len], &self.buf_t[..self.buf_len]))
    }
}

/// Random-access cursor over one encoded posting list with block skipping (`advance`), for MaxScore.
pub struct BlockCursor<'a> {
    src: &'a [u8],
    lo: u32,
    n: usize,
    skip_last: &'a [u32],
    skip_start: &'a [u32],
    blk: usize,
    docs: [u32; BLOCK],
    tfs: [u32; BLOCK],
    len: usize,
    i: usize,
    /// current doc (u32::MAX when exhausted)
    pub doc: u32,
}

impl<'a> BlockCursor<'a> {
    /// `skip_last` = last doc of each full block, `skip_start` = start offset of each full block + the tail's start
    pub fn new(
        src: &'a [u8],
        lo: u32,
        n: usize,
        skip_last: &'a [u32],
        skip_start: &'a [u32],
    ) -> Self {
        let mut c = BlockCursor {
            src,
            lo,
            n,
            skip_last,
            skip_start,
            blk: 0,
            docs: [0; BLOCK],
            tfs: [0; BLOCK],
            len: 0,
            i: 0,
            doc: u32::MAX,
        };
        if n > 0 {
            c.decode(0);
        }
        c
    }

    #[inline]
    fn nfull(&self) -> usize {
        self.n / BLOCK
    }

    #[inline]
    fn has_block(&self, b: usize) -> bool {
        b < self.nfull() || (b == self.nfull() && !self.n.is_multiple_of(BLOCK))
    }

    fn decode(&mut self, b: usize) {
        self.blk = b;
        let mut pos = self.skip_start[b] as usize;
        let prev = if b == 0 {
            self.lo
        } else {
            self.skip_last[b - 1]
        };
        if b < self.nfull() {
            let wd = self.src[pos] as u32;
            let wt = self.src[pos + 1] as u32;
            pos += 2;
            unpack128(&self.src[pos..], wd, &mut self.docs);
            pos += 16 * wd as usize;
            unpack128(&self.src[pos..], wt, &mut self.tfs);
            let mut p = prev;
            for k in 0..BLOCK {
                p += self.docs[k];
                self.docs[k] = p;
                self.tfs[k] += 1;
            }
            self.len = BLOCK;
        } else {
            let m = self.n % BLOCK;
            let mut p = prev;
            for k in 0..m {
                let v = get_varint(self.src, &mut pos);
                p += (v >> 1) as u32;
                self.docs[k] = p;
                self.tfs[k] = if v & 1 == 1 {
                    1
                } else {
                    get_varint(self.src, &mut pos) as u32
                };
            }
            self.len = m;
        }
        self.i = 0;
        self.doc = self.docs[0];
    }

    #[inline]
    pub fn tf(&self) -> u32 {
        self.tfs[self.i]
    }

    #[inline]
    pub fn next(&mut self) {
        self.i += 1;
        if self.i < self.len {
            self.doc = self.docs[self.i];
        } else if self.has_block(self.blk + 1) {
            self.decode(self.blk + 1);
        } else {
            self.doc = u32::MAX;
        }
    }

    /// move to the first posting with doc >= target
    pub fn advance(&mut self, target: u32) {
        if self.doc >= target {
            return;
        }
        // skip whole blocks whose last doc is < target (no decoding)
        if self.blk < self.nfull() && self.skip_last[self.blk] < target {
            let nf = self.nfull();
            let mut b = self.blk + 1;
            while b < nf && self.skip_last[b] < target {
                b += 1;
            }
            if !self.has_block(b) {
                self.doc = u32::MAX;
                return;
            }
            self.decode(b);
        }
        while self.doc < target {
            self.next();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cursor_advance() {
        for n in [1usize, 127, 128, 129, 300, 1000] {
            let docs: Vec<u32> = (0..n as u32).map(|i| 50 + i * 5).collect();
            let tfs: Vec<u32> = (0..n as u32).map(|i| 1 + i % 3).collect();
            let mut out = Vec::new();
            let (mut last, mut starts) = (Vec::new(), Vec::new());
            encode_postings_skips(50, &docs, &tfs, &mut out, Some((&mut last, &mut starts)));
            let mut c = BlockCursor::new(&out, 50, n, &last, &starts);
            let mut got = Vec::new();
            while c.doc != u32::MAX {
                got.push((c.doc, c.tf()));
                c.next();
            }
            assert_eq!(
                got,
                docs.iter()
                    .copied()
                    .zip(tfs.iter().copied())
                    .collect::<Vec<_>>()
            );
            for target in [0u32, 51, 55, 700, 2000, 5000, 6000] {
                let mut c = BlockCursor::new(&out, 50, n, &last, &starts);
                c.advance(target);
                let exp = docs.iter().position(|&d| d >= target);
                match exp {
                    Some(i) => assert_eq!((c.doc, c.tf()), (docs[i], tfs[i]), "n={n} t={target}"),
                    None => assert_eq!(c.doc, u32::MAX),
                }
            }
        }
    }

    #[test]
    fn roundtrip() {
        for n in [0usize, 1, 5, 127, 128, 129, 300, 1000] {
            let docs: Vec<u32> = (0..n as u32).map(|i| 1000 + i * 8 + (i % 7)).collect();
            let tfs: Vec<u32> = (0..n as u32)
                .map(|i| 1 + (i % 5 == 0) as u32 * (i % 11))
                .collect();
            let mut out = Vec::new();
            encode_postings(1000, &docs, &tfs, &mut out);
            let mut it = PostingIter::new(&out, 1000, n);
            let (mut gd, mut gt) = (Vec::new(), Vec::new());
            while let Some((d, t)) = it.next_block() {
                gd.extend_from_slice(d);
                gt.extend_from_slice(t);
            }
            assert_eq!(gd, docs, "n={n}");
            assert_eq!(gt, tfs, "n={n}");
        }
    }

    #[test]
    fn widths() {
        for w in 0..=32u32 {
            let mask = if w == 32 {
                u32::MAX
            } else {
                (1u32 << w).wrapping_sub(1)
            };
            let mut v = [0u32; BLOCK];
            for (i, x) in v.iter_mut().enumerate() {
                *x = (i as u32).wrapping_mul(2654435761) & mask;
            }
            let mut out = Vec::new();
            pack128(&v, w, &mut out);
            assert_eq!(out.len(), 16 * w as usize);
            let mut back = [0u32; BLOCK];
            unpack128(&out, w, &mut back);
            assert_eq!(back, v, "w={w}");
        }
    }
}
