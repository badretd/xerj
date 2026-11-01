//! On-disk format, version 2, little-endian.
//!
//! ```text
//! header (192 B)  magic "XPGDIR\x1a\n", version u32, header_size u32, page_size u32, flags u32,
//!                 sample u32 (512), n_sections u32, passages u64, pages u64, file_size u64,
//!                 articles u64, ident_bytes u64 (1 MiB), head_sha256 32 B, tail_sha256 32 B,
//!                 body_sha256 32 B (everything after the header), key_spec 16 B (ASCII,
//!                 NUL-padded, "jsonl-id"), mtime_ns i64 (0 when flag bit 2 is clear)
//! section table   40 B each: name 4 B, width u32 (bytes per integer of a sample array, 0 for a
//!                 bitvector), offset u64, length u64, n u64 (bits or entries), ones u64
//! sections        8-byte aligned. PBV/PBS passage bitvector + select samples; ABV/ABS article
//!                 start-page bitvector + select samples; BBV/BBS/BBR boundary bitvector + select
//!                 samples + rank samples.
//! ```
//!
//! Flags: bit 0 article level, bit 1 file ends with `\n`, bit 2 mtime_ns present. Select arrays end
//! with a sentinel equal to the bitvector's bit count. Version 1 is the same without `mtime_ns`: its
//! header is 184 B and the section table starts there.
use crate::{Error, Result};
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io;
use std::path::Path;

pub const MAGIC: &[u8; 8] = b"XPGDIR\x1a\n";
pub const VERSION: u32 = 2;
pub const IDENT_BYTES: u64 = 1 << 20;
pub const TAIL: usize = 4096;
pub const KEY_MAX: usize = 256;
pub const HEADER_V1: usize = 184;
pub const HEADER_V2: usize = 192;
pub const SECTION: usize = 40;

pub const F_ARTICLES: u32 = 1;
pub const F_ENDS_NL: u32 = 2;
pub const F_MTIME: u32 = 4;

/// One section of a directory being written.
pub struct Section {
    pub name: [u8; 4],
    pub width: u32,
    pub payload: Vec<u8>,
    pub n: u64,
    pub ones: u64,
}

/// Header fields, as stored.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Header {
    pub version: u32,
    pub header_size: u32,
    pub page_size: u32,
    pub flags: u32,
    pub sample: u32,
    pub n_sections: u32,
    pub passages: u64,
    pub pages: u64,
    pub file_size: u64,
    pub articles: u64,
    pub ident_bytes: u64,
    pub head_sha: [u8; 32],
    pub tail_sha: [u8; 32],
    pub body_sha: [u8; 32],
    pub key_spec: String,
    pub mtime_ns: i64,
}

fn u32_at(b: &[u8], off: usize) -> u32 {
    u32::from_le_bytes(b[off..off + 4].try_into().expect("4 bytes"))
}

fn u64_at(b: &[u8], off: usize) -> u64 {
    u64::from_le_bytes(b[off..off + 8].try_into().expect("8 bytes"))
}

fn sha_at(b: &[u8], off: usize) -> [u8; 32] {
    b[off..off + 32].try_into().expect("32 bytes")
}

impl Header {
    /// Parses the header. Checks that it is whole and that the version is known.
    pub fn parse(blob: &[u8]) -> Result<Self> {
        if blob.len() < HEADER_V1 || &blob[..8] != MAGIC {
            return Err(Error::Format("not a page directory".into()));
        }
        let version = u32_at(blob, 8);
        match version {
            1 => {}
            2 if blob.len() >= HEADER_V2 => {}
            v => return Err(Error::Format(format!("unsupported version {v}"))),
        }
        let kspec = &blob[168..184];
        let end = kspec.iter().rposition(|&c| c != 0).map_or(0, |p| p + 1);
        let key_spec = std::str::from_utf8(&kspec[..end])
            .map_err(|_| Error::Format("key_spec is not ASCII".into()))?
            .to_owned();
        Ok(Self {
            version,
            header_size: u32_at(blob, 12),
            page_size: u32_at(blob, 16),
            flags: u32_at(blob, 20),
            sample: u32_at(blob, 24),
            n_sections: u32_at(blob, 28),
            passages: u64_at(blob, 32),
            pages: u64_at(blob, 40),
            file_size: u64_at(blob, 48),
            articles: u64_at(blob, 56),
            ident_bytes: u64_at(blob, 64),
            head_sha: sha_at(blob, 72),
            tail_sha: sha_at(blob, 104),
            body_sha: sha_at(blob, 136),
            key_spec,
            mtime_ns: if version >= 2 {
                i64::from_le_bytes(blob[184..192].try_into().expect("8 bytes"))
            } else {
                0
            },
        })
    }

    /// Size of the fixed part, which is where the section table starts.
    pub fn fixed_size(&self) -> usize {
        if self.version == 1 {
            HEADER_V1
        } else {
            HEADER_V2
        }
    }
}

/// Lays out a whole directory file.
pub fn encode(mut h: Header, sections: Vec<Section>) -> Vec<u8> {
    let header_size = (HEADER_V2 + sections.len() * SECTION + 7) & !7;
    let mut body = Vec::with_capacity(sections.iter().map(|s| s.payload.len() + 8).sum());
    let mut table = Vec::with_capacity(sections.len() * SECTION);
    for s in &sections {
        table.extend_from_slice(&s.name);
        table.extend_from_slice(&s.width.to_le_bytes());
        table.extend_from_slice(&((header_size + body.len()) as u64).to_le_bytes());
        table.extend_from_slice(&(s.payload.len() as u64).to_le_bytes());
        table.extend_from_slice(&s.n.to_le_bytes());
        table.extend_from_slice(&s.ones.to_le_bytes());
        body.extend_from_slice(&s.payload);
        body.resize(body.len().next_multiple_of(8), 0);
    }
    h.header_size = header_size as u32;
    h.n_sections = sections.len() as u32;
    h.body_sha = Sha256::digest(&body).into();
    let mut out = Vec::with_capacity(header_size + body.len());
    out.extend_from_slice(MAGIC);
    for v in [
        h.version,
        h.header_size,
        h.page_size,
        h.flags,
        h.sample,
        h.n_sections,
    ] {
        out.extend_from_slice(&v.to_le_bytes());
    }
    for v in [h.passages, h.pages, h.file_size, h.articles, h.ident_bytes] {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out.extend_from_slice(&h.head_sha);
    out.extend_from_slice(&h.tail_sha);
    out.extend_from_slice(&h.body_sha);
    let mut spec = [0u8; 16];
    let b = h.key_spec.as_bytes();
    spec[..b.len().min(16)].copy_from_slice(&b[..b.len().min(16)]);
    out.extend_from_slice(&spec);
    out.extend_from_slice(&h.mtime_ns.to_le_bytes());
    out.extend_from_slice(&table);
    out.resize(header_size, 0);
    out.extend_from_slice(&body);
    out
}

/// A file's size and the hashes of its first and last megabyte.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceIdentity {
    pub size: u64,
    pub head_sha: [u8; 32],
    pub tail_sha: [u8; 32],
}

/// `pread` that fills `buf` unless the file ends first; returns the bytes read.
pub fn read_full_at(f: &File, buf: &mut [u8], offset: u64) -> io::Result<usize> {
    let mut got = 0;
    while got < buf.len() {
        match read_at(f, &mut buf[got..], offset + got as u64) {
            Ok(0) => break,
            Ok(n) => got += n,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(got)
}

#[cfg(unix)]
fn read_at(f: &File, buf: &mut [u8], offset: u64) -> io::Result<usize> {
    std::os::unix::fs::FileExt::read_at(f, buf, offset)
}

#[cfg(windows)]
fn read_at(f: &File, buf: &mut [u8], offset: u64) -> io::Result<usize> {
    std::os::windows::fs::FileExt::seek_read(f, buf, offset)
}

/// Identity of an open file.
pub fn source_identity(f: &File, ident_bytes: u64) -> io::Result<SourceIdentity> {
    let size = f.metadata()?.len();
    let n = ident_bytes.min(size) as usize;
    let mut buf = vec![0u8; n];
    let got = read_full_at(f, &mut buf, 0)?;
    let head_sha = Sha256::digest(&buf[..got]).into();
    let got = read_full_at(f, &mut buf, size.saturating_sub(ident_bytes))?;
    let tail_sha = Sha256::digest(&buf[..got]).into();
    Ok(SourceIdentity {
        size,
        head_sha,
        tail_sha,
    })
}

/// Modification time of a file in nanoseconds, signed like `st_mtime_ns`.
pub fn mtime_ns(md: &std::fs::Metadata) -> i64 {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        md.mtime()
            .saturating_mul(1_000_000_000)
            .saturating_add(md.mtime_nsec())
    }
    #[cfg(not(unix))]
    {
        md.modified()
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map_or(0, |d| d.as_nanos() as i64)
    }
}

/// Writes `data` to `path` so that a crash leaves either the old file or the whole new one.
pub fn write_atomic(path: &Path, data: &[u8]) -> io::Result<()> {
    use std::io::Write;
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(".tmp");
    let tmp = std::path::PathBuf::from(tmp);
    {
        let mut f = File::create(&tmp)?;
        f.write_all(data)?;
        f.sync_all()?;
    }
    std::fs::rename(&tmp, path)?;
    #[cfg(unix)]
    if let Some(dir) = path.parent() {
        let dir = if dir.as_os_str().is_empty() {
            Path::new(".")
        } else {
            dir
        };
        File::open(dir)?.sync_all()?;
    }
    Ok(())
}
