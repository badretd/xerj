//! Read side: locate, read, check.
use crate::bits::{Bits, SAMPLE};
use crate::format::{
    mtime_ns, read_full_at, source_identity, Header, F_ARTICLES, F_ENDS_NL, F_MTIME, KEY_MAX, MAGIC,
    SECTION,
    TAIL,
};
use crate::{Error, Result};
use memchr::{memchr, memchr_iter, memrchr};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

const REC: &[u8] = b"{\"id\": \"";

/// Article key of a record that begins `{"id": "<article>#<passage>`: the bytes before `#`, or the
/// whole id. `None` when `rec` does not begin like a record.
pub fn jsonl_id_key(rec: &[u8]) -> Option<&[u8]> {
    if !rec.starts_with(REC) {
        return None;
    }
    let rest = &rec[REC.len()..];
    let q = memchr(b'"', rest)?;
    let head = &rest[..q];
    Some(match memchr(b'#', head) {
        Some(h) => &head[..h],
        None => head,
    })
}

/// The JSON `"id"` of a record line; `None` when it is not an object with a string id.
pub fn jsonl_record_id(rec: &[u8]) -> Option<String> {
    if rec.starts_with(REC) {
        let rest = &rec[REC.len()..];
        if let Some(q) = memchr(b'"', rest) {
            if memchr(b'\\', &rest[..q]).is_none() {
                return std::str::from_utf8(&rest[..q]).ok().map(str::to_owned);
            }
        }
        let mut de = serde_json::Deserializer::from_slice(&rec[REC.len() - 1..]);
        return serde::Deserialize::deserialize(&mut de).ok();
    }
    let v: serde_json::Value = serde_json::from_slice(rec).ok()?;
    v.get("id")?.as_str().map(str::to_owned)
}

fn id_matches(line: &[u8], expect: &str) -> bool {
    let e = expect.as_bytes();
    if line.starts_with(REC)
        && line[REC.len()..].starts_with(e)
        && line.get(REC.len() + e.len()) == Some(&b'"')
        && memchr(b'"', e).is_none()
        && memchr(b'\\', e).is_none()
    {
        return true;
    }
    jsonl_record_id(line).as_deref() == Some(expect)
}

fn describe(line: &[u8]) -> String {
    match jsonl_record_id(line) {
        Some(id) => format!("has id {id:?}"),
        None => {
            let shown = String::from_utf8_lossy(&line[..line.len().min(24)]);
            format!("is not a record (it begins {shown:?})")
        }
    }
}

/// First 8 bytes of the line's SHA-256 (trailing newline excluded), little-endian. Stored at index
/// time, it also catches an edit inside the right passage, which an id check cannot see.
pub fn line_hash(line: &[u8]) -> u64 {
    let line = line.strip_suffix(b"\n").unwrap_or(line);
    let d = Sha256::digest(line);
    u64::from_le_bytes(d[..8].try_into().expect("8 bytes"))
}

/// What a read must find.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Expect<'a> {
    /// The line's JSON `"id"` equals this.
    Id(&'a str),
    /// [`line_hash`] of the line equals this.
    LineHash(u64),
}

/// Where a passage's line sits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Located {
    pub page: u64,
    /// Rank among the records that start in this page.
    pub rank: u64,
    /// Records that start in this page.
    pub count: u64,
}

#[derive(Debug, Clone, Copy)]
pub struct ReadOptions {
    /// Fetch one more 4 KiB with the last record of a page, which nearly always crosses. One device
    /// read instead of two on a cold file (185 µs against 383 µs).
    pub prefetch: bool,
}

impl Default for ReadOptions {
    fn default() -> Self {
        Self { prefetch: true }
    }
}

#[derive(Debug, Default, Clone)]
struct ArticleLevel {
    ab: Bits,
    abs: Vec<u64>,
    bb: Bits,
    bbs: Vec<u64>,
    bbr: Vec<u64>,
}

/// A directory loaded in memory.
#[derive(Debug, Clone)]
pub struct PageDir {
    header: Header,
    shift: u32,
    pb: Bits,
    pbs: Vec<u64>,
    articles: Option<ArticleLevel>,
    bytes: u64,
}

/// Loads a directory file (version 1 or 2) and checks its body hash.
pub fn open(path: &Path) -> Result<PageDir> {
    PageDir::from_bytes(&std::fs::read(path)?, true)
}

fn int_section(payload: &[u8], width: u32, name: &str) -> Result<Vec<u64>> {
    match width {
        4 => Ok(payload
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().expect("4 bytes")) as u64)
            .collect()),
        8 => Ok(payload
            .chunks_exact(8)
            .map(|c| u64::from_le_bytes(c.try_into().expect("8 bytes")))
            .collect()),
        w => Err(Error::Format(format!("section {name}: width {w}"))),
    }
}

impl PageDir {
    /// `verify` checks the body hash; pass `false` only when the caller checked the file itself.
    pub fn from_bytes(blob: &[u8], verify: bool) -> Result<Self> {
        if blob.len() < 8 || &blob[..8] != MAGIC {
            return Err(Error::Format("not a page directory".into()));
        }
        let h = Header::parse(blob)?;
        if h.sample as u64 != SAMPLE {
            return Err(Error::Format(format!("unsupported sample {}", h.sample)));
        }
        if !h.page_size.is_power_of_two() || h.page_size < 16 {
            return Err(Error::Format(format!("bad page size {}", h.page_size)));
        }
        let hsize = h.header_size as usize;
        if hsize > blob.len() {
            return Err(Error::Format("truncated header".into()));
        }
        if verify && <[u8; 32]>::from(Sha256::digest(&blob[hsize..])) != h.body_sha {
            return Err(Error::Format("directory body checksum mismatch".into()));
        }
        let mut sec: HashMap<String, (&[u8], u32, u64, u64)> = HashMap::new();
        let table = h.fixed_size();
        for j in 0..h.n_sections as usize {
            let at = table + j * SECTION;
            let e = blob
                .get(at..at + SECTION)
                .ok_or_else(|| Error::Format("truncated section table".into()))?;
            let u = |o: usize| u64::from_le_bytes(e[o..o + 8].try_into().expect("8 bytes"));
            let name: Vec<u8> = e[..4].iter().take_while(|&&c| c != 0).copied().collect();
            let name = String::from_utf8_lossy(&name).into_owned();
            let width = u32::from_le_bytes(e[4..8].try_into().expect("4 bytes"));
            let (off, len) = (u(8) as usize, u(16) as usize);
            let payload = off
                .checked_add(len)
                .and_then(|end| blob.get(off..end))
                .ok_or_else(|| Error::Format(format!("section {name} is out of bounds")))?;
            sec.insert(name, (payload, width, u(24), u(32)));
        }
        let get = |name: &str| {
            sec.get(name)
                .copied()
                .ok_or_else(|| Error::Format(format!("section {name} is missing")))
        };
        let bits = |name: &str| -> Result<Bits> {
            let (p, _, n, _) = get(name)?;
            Bits::from_le_bytes(p, n)
                .ok_or_else(|| Error::Format(format!("section {name} is too short")))
        };
        let ints = |name: &str| -> Result<Vec<u64>> {
            let (p, w, _, _) = get(name)?;
            int_section(p, w, name)
        };
        let pb = bits("PBV")?;
        let pbs = ints("PBS")?;
        let articles = if h.flags & F_ARTICLES != 0 {
            Some(ArticleLevel {
                ab: bits("ABV")?,
                abs: ints("ABS")?,
                bb: bits("BBV")?,
                bbs: ints("BBS")?,
                bbr: ints("BBR")?,
            })
        } else {
            None
        };
        Ok(Self {
            shift: h.page_size.trailing_zeros(),
            header: h,
            pb,
            pbs,
            articles,
            bytes: blob.len() as u64,
        })
    }

    pub fn version(&self) -> u32 {
        self.header.version
    }
    pub fn passages(&self) -> u64 {
        self.header.passages
    }
    pub fn pages(&self) -> u64 {
        self.header.pages
    }
    pub fn articles(&self) -> u64 {
        self.header.articles
    }
    pub fn page_size(&self) -> u32 {
        self.header.page_size
    }
    pub fn file_size(&self) -> u64 {
        self.header.file_size
    }
    /// The file's modification time at build time; `None` in a version 1 directory.
    pub fn mtime_ns(&self) -> Option<i64> {
        (self.header.version >= 2 && self.header.flags & F_MTIME != 0)
            .then_some(self.header.mtime_ns)
    }
    pub fn key_spec(&self) -> &str {
        &self.header.key_spec
    }
    /// Size of the directory file.
    pub fn nbytes(&self) -> u64 {
        self.bytes
    }

    /// Errors unless the file has the size and head/tail hashes recorded at build time. Required
    /// when a source is opened, but not sufficient: edits whose size changes cancel pass it.
    pub fn check_source(&self, f: &File) -> Result<()> {
        let id = source_identity(f, self.header.ident_bytes)?;
        let (h, t) = (&self.header.head_sha, &self.header.tail_sha);
        if id.size != self.header.file_size || id.head_sha != *h || id.tail_sha != *t {
            return Err(Error::SourceMismatch(format!(
                "source differs from the indexed file (size {} vs {}, head {}, tail {})",
                id.size,
                self.header.file_size,
                if id.head_sha == *h { "ok" } else { "differs" },
                if id.tail_sha == *t { "ok" } else { "differs" },
            )));
        }
        Ok(())
    }

    /// Fast hint from one `stat`: `Some(false)` if size or mtime differs, `Some(true)` if both are
    /// equal, `None` if this directory stores no mtime and the size is equal. Tools can set mtime
    /// back, so `Some(true)` proves nothing; it never replaces an expectation on reads.
    pub fn check_stat(&self, f: &File) -> Result<Option<bool>> {
        let md = f.metadata()?;
        if md.len() != self.header.file_size {
            return Ok(Some(false));
        }
        Ok(self.mtime_ns().map(|m| m == mtime_ns(&md)))
    }

    /// Page, rank and page count of passage `i`.
    pub fn locate(&self, i: u64) -> Result<Located> {
        if i >= self.header.passages {
            return Err(Error::Invalid(format!(
                "passage {i} is past the end ({})",
                self.header.passages
            )));
        }
        let pos = self
            .pb
            .select1(&self.pbs, i)
            .ok_or_else(|| Error::Format("passage bitvector is short".into()))?;
        let z0 = self.pb.zero_before(pos);
        let z1 = self.pb.zero_from(pos + 1);
        let start = z0.map_or(0, |z| z + 1);
        Ok(Located {
            page: pos - i,
            rank: pos - start,
            count: z1 - start,
        })
    }

    /// First passage that starts in page `p` and the number that start there. The passages that
    /// overlap `p` are these plus, unless byte 0 of `p` starts a record, the one before them.
    pub fn passages_in_page(&self, p: u64) -> Result<(u64, u64)> {
        if p >= self.header.pages {
            return Err(Error::Invalid(format!("page {p} is past the end")));
        }
        let start = if p == 0 {
            0
        } else {
            self.pb.select0(&self.pbs, p - 1) + 1
        };
        let z1 = self.pb.zero_from(start);
        Ok((start - p, z1 - start))
    }

    fn read_page(&self, f: &File, loc: Located, prefetch: bool) -> Result<Vec<u8>> {
        let pg = self.header.page_size as usize;
        let want = if prefetch && loc.rank + 1 == loc.count {
            pg + TAIL
        } else {
            pg
        };
        let mut buf = vec![0u8; want];
        let got = read_full_at(f, &mut buf, loc.page << self.shift)?;
        buf.truncate(got);
        Ok(buf)
    }

    /// Passage `i`'s whole line, newline included, read through its page and checked. Returns the
    /// line only if it is what `expect` says, so a changed file gives an error, never wrong text.
    pub fn read(&self, f: &File, i: u64, expect: Expect<'_>, opts: ReadOptions) -> Result<Vec<u8>> {
        let (line, page) = self.read_line(f, i, opts)?;
        match expect {
            Expect::Id(id) if !id_matches(&line, id) => Err(Error::IdMismatch {
                passage: i,
                page,
                expected: id.to_owned(),
                found: describe(&line),
            }),
            Expect::LineHash(h) => {
                let found = line_hash(&line);
                if found == h {
                    Ok(line)
                } else {
                    Err(Error::HashMismatch {
                        passage: i,
                        page,
                        expected: h,
                        found,
                    })
                }
            }
            _ => Ok(line),
        }
    }

    /// The line with no check of what it is. **For tests and tools only**: after an edit this can
    /// return a different, well-formed line with no error.
    pub fn read_unchecked(&self, f: &File, i: u64, opts: ReadOptions) -> Result<Vec<u8>> {
        Ok(self.read_line(f, i, opts)?.0)
    }

    fn read_line(&self, f: &File, i: u64, opts: ReadOptions) -> Result<(Vec<u8>, u64)> {
        let loc = self.locate(i)?;
        let buf = self.read_page(f, loc, opts.prefetch)?;
        let start = self
            .kth_start(&buf, loc)
            .map_err(|m| Error::Stale(format!("passage {i}, page {}: {m}", loc.page)))?;
        let line = match memchr(b'\n', &buf[start..]) {
            Some(e) => buf[start..start + e + 1].to_vec(),
            None => {
                let off = (loc.page << self.shift) + buf.len() as u64;
                let data = buf[start..].to_vec();
                self.collect(f, data, off, 1)?.pop().expect("one record")
            }
        };
        Ok((line, loc.page))
    }

    /// Offset in `buf` of the `rank`-th record that starts in this page.
    fn kth_start(&self, buf: &[u8], loc: Located) -> std::result::Result<usize, String> {
        let pg = self.header.page_size as usize;
        let l = buf.len().min(pg);
        let body = &buf[..l.saturating_sub(1)];
        let interior = memchr_iter(b'\n', body).count() as u64;
        let (k, c) = (loc.rank, loc.count);
        let m = if interior == c {
            k
        } else if interior + 1 == c {
            if k == 0 {
                return Ok(0);
            }
            k - 1
        } else {
            return Err(format!(
                "page has {interior} interior line starts, the directory expects {} or {c}",
                c.saturating_sub(1)
            ));
        };
        if m <= interior >> 1 {
            let mut it = memchr_iter(b'\n', body);
            let j = it.nth(m as usize).ok_or("page changed while reading")?;
            Ok(j + 1)
        } else {
            let mut end = body.len();
            let mut j = 0;
            for _ in 0..(interior - m) {
                j = memrchr(b'\n', &body[..end]).ok_or("page changed while reading")?;
                end = j;
            }
            Ok(j + 1)
        }
    }

    /// Records from `data`, which starts at a record start, reading on page by page.
    fn collect(&self, f: &File, mut data: Vec<u8>, mut off: u64, count: u64) -> Result<Vec<Vec<u8>>> {
        let pg = self.header.page_size as usize;
        let mut recs = Vec::new();
        let mut cur = 0usize;
        while (recs.len() as u64) < count {
            if let Some(e) = memchr(b'\n', &data[cur..]) {
                recs.push(data[cur..cur + e + 1].to_vec());
                cur += e + 1;
                continue;
            }
            let mut more = vec![0u8; pg];
            let got = read_full_at(f, &mut more, off)?;
            if got == 0 {
                // A line may end without a newline only where the indexed file did. Anywhere else
                // the file was cut, and this is a piece of a line.
                let last_line = self.header.flags & F_ENDS_NL == 0 && off == self.header.file_size;
                if cur < data.len() && last_line {
                    recs.push(data[cur..].to_vec());
                }
                break;
            }
            data.drain(..cur);
            cur = 0;
            data.extend_from_slice(&more[..got]);
            off += got as u64;
        }
        if (recs.len() as u64) < count {
            return Err(Error::Stale("the file ended before the expected records".into()));
        }
        Ok(recs)
    }

    fn level(&self) -> Result<&ArticleLevel> {
        self.articles
            .as_ref()
            .ok_or_else(|| Error::Invalid("this directory has no article level".into()))
    }

    /// Article index of passage `i`.
    pub fn article_of(&self, i: u64) -> Result<u64> {
        let lv = self.level()?;
        self.check_passage(i)?;
        Ok(lv.bb.rank1(&lv.bbr, i + 1) - 1)
    }

    fn check_passage(&self, i: u64) -> Result<()> {
        if i >= self.header.passages {
            return Err(Error::Invalid(format!("passage {i} is past the end")));
        }
        Ok(())
    }

    /// The id passage `i` had at build time, `<article_key>#<n>`. The key comes from the caller's
    /// index; `n`, the passage's number within its article, comes from the article level.
    pub fn expected_id(&self, i: u64, article_key: &str) -> Result<String> {
        let lv = self.level()?;
        self.check_passage(i)?;
        let first = lv.bb.one_before(i + 1).unwrap_or(0);
        Ok(format!("{article_key}#{}", i - first))
    }

    /// First passage and passage count of article `a`.
    pub fn article_passages(&self, a: u64) -> Result<(u64, u64)> {
        let lv = self.level()?;
        let total = self.header.articles;
        if a >= total {
            return Err(Error::Invalid(format!("article {a} is past the end")));
        }
        let damaged = || Error::Format("boundary bitvector is short".into());
        let first = lv.bb.select1(&lv.bbs, a).ok_or_else(damaged)?;
        let end = if a + 1 < total {
            lv.bb.select1(&lv.bbs, a + 1).ok_or_else(damaged)?
        } else {
            self.header.passages
        };
        Ok((first, end - first))
    }

    /// Start page of article `a`, its rank among the articles starting there, and their count.
    pub fn article_start(&self, a: u64) -> Result<Located> {
        let lv = self.level()?;
        if a >= self.header.articles {
            return Err(Error::Invalid(format!("article {a} is past the end")));
        }
        let pos = lv
            .ab
            .select1(&lv.abs, a)
            .ok_or_else(|| Error::Format("article bitvector is short".into()))?;
        let start = lv.ab.zero_before(pos).map_or(0, |z| z + 1);
        Ok(Located {
            page: pos - a,
            rank: pos - start,
            count: lv.ab.zero_from(pos + 1) - start,
        })
    }

    fn check_article(&self, a: u64, lines: &[Vec<u8>], key: &str, page: u64) -> Result<()> {
        let first = self.article_passages(a)?.0;
        for (j, line) in lines.iter().enumerate() {
            let exp = format!("{key}#{j}");
            if !id_matches(line, &exp) {
                return Err(Error::IdMismatch {
                    passage: first + j as u64,
                    page,
                    expected: exp,
                    found: describe(line),
                });
            }
        }
        Ok(())
    }

    /// All passages of article `a`, every line checked against `<key>#<0, 1, 2, ...>`.
    pub fn read_article(&self, f: &File, a: u64, key: &str, opts: ReadOptions) -> Result<Vec<Vec<u8>>> {
        let (first, m) = self.article_passages(a)?;
        let loc = self.locate(first)?;
        let buf = self.read_page(f, loc, opts.prefetch)?;
        let start = self
            .kth_start(&buf, loc)
            .map_err(|e| Error::Stale(format!("article {a}, page {}: {e}", loc.page)))?;
        let off = (loc.page << self.shift) + buf.len() as u64;
        let lines = self.collect(f, buf[start..].to_vec(), off, m)?;
        self.check_article(a, &lines, key, loc.page)?;
        Ok(lines)
    }

    /// All passages of article `a` found from the article level alone, by the record key in the
    /// start page, without the passage level. Every line is checked as in [`Self::read_article`].
    pub fn read_article_by_start_page(&self, f: &File, a: u64, key: &str) -> Result<Vec<Vec<u8>>> {
        if self.header.key_spec != "jsonl-id" {
            return Err(Error::Invalid(format!(
                "no record-key parser for key spec {:?}",
                self.header.key_spec
            )));
        }
        let loc = self.article_start(a)?;
        let pg = self.header.page_size as usize;
        let base = loc.page << self.shift;
        let mut buf = vec![0u8; pg + TAIL];
        let got = read_full_at(f, &mut buf, base)?;
        buf.truncate(got);
        let l = got.min(pg);
        let prefix = |s: usize| &buf[s..buf.len().min(s + KEY_MAX)];
        let starts: Vec<usize> = memchr_iter(b'\n', &buf[..l.saturating_sub(1)])
            .map(|j| j + 1)
            .collect();
        let mut keys = Vec::with_capacity(starts.len());
        for &s in &starts {
            let k = jsonl_id_key(prefix(s))
                .ok_or_else(|| Error::Stale("a line in the page does not begin like a record".into()))?;
            keys.push(k);
        }
        let mut heads: Vec<usize> = (1..starts.len())
            .filter(|&t| keys[t] != keys[t - 1])
            .map(|t| starts[t])
            .collect();
        let rem = loc.count as i64 - heads.len() as i64;
        let k0 = jsonl_id_key(prefix(0));
        if starts.is_empty() {
            if loc.count != 1 || k0.is_none() {
                return Err(Error::Stale("article start page without a record start".into()));
            }
            heads = vec![0];
        } else if rem == 1 {
            let at = if k0.is_some() && k0 == Some(keys[0]) { 0 } else { starts[0] };
            heads.insert(0, at);
        } else if rem == 2 {
            if k0.is_none() {
                return Err(Error::Stale("byte 0 should begin an article".into()));
            }
            heads.insert(0, starts[0]);
            heads.insert(0, 0);
        } else if rem != 0 {
            return Err(Error::Stale(format!(
                "page has {} article changes, the directory expects {}",
                heads.len(),
                loc.count
            )));
        }
        let s0 = *heads
            .get(loc.rank as usize)
            .ok_or_else(|| Error::Stale("article rank beyond the article starts in the page".into()))?;
        let count = self.article_passages(a)?.1;
        let off = base + buf.len() as u64;
        let lines = self.collect(f, buf[s0..].to_vec(), off, count)?;
        self.check_article(a, &lines, key, loc.page)?;
        Ok(lines)
    }
}
