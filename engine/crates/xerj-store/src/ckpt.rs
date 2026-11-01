//! Checkpoint container: one file of tagged, 8-byte aligned sections, written sequentially and read through mmap.
//!
//! `SectionWriter` streams sections (`XERJCKP1`, then per section: u32 tag, u32 0, u64 length, data, zero padding to 8).
//! `Sections` maps a file and indexes its sections; `bytes(tag)` is a zero-copy slice of the mapping, `blob(tag)` a
//! refcounted handle (`Blob`) that immutable index structures (BM25 segments) keep instead of copying their data, and
//! `vec::<T>(tag)` copies a plain little-endian column into a `Vec<T>`.

use crate::error::StoreError;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::Arc;

pub const CKPT_MAGIC: &[u8; 8] = b"XERJCKP1";

#[cfg(not(target_endian = "little"))]
compile_error!("xerj-mem checkpoints assume a little-endian target");

/// Plain fixed-size integer columns that can be written / read as raw little-endian bytes.
pub trait Pod: Copy + Default + 'static {}
impl Pod for u8 {}
impl Pod for u16 {}
impl Pod for u32 {}
impl Pod for u64 {}
impl Pod for f32 {}

pub fn pod_bytes<T: Pod>(v: &[T]) -> &[u8] {
    // SAFETY: Pod types have no padding and any bit pattern is valid
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

pub fn pod_vec<T: Pod>(b: &[u8]) -> Result<Vec<T>, StoreError> {
    let sz = std::mem::size_of::<T>();
    if !b.len().is_multiple_of(sz) {
        return Err(StoreError::Format(
            "section length is not a multiple of the element size".into(),
        ));
    }
    let n = b.len() / sz;
    let mut v: Vec<T> = Vec::with_capacity(n);
    // SAFETY: capacity n, bytes copied cover exactly n elements of a Pod type
    unsafe {
        std::ptr::copy_nonoverlapping(b.as_ptr(), v.as_mut_ptr() as *mut u8, b.len());
        v.set_len(n);
    }
    Ok(v)
}

/// Immutable bytes, owned or a range of a shared mapping.
#[derive(Clone)]
pub enum Blob {
    Owned(Arc<[u8]>),
    Mapped(Arc<memmap2::Mmap>, usize, usize),
}

impl Blob {
    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        match self {
            Blob::Owned(v) => v,
            Blob::Mapped(m, s, n) => &m[*s..*s + *n],
        }
    }
    pub fn owned(v: Vec<u8>) -> Blob {
        Blob::Owned(v.into())
    }
    pub fn is_mapped(&self) -> bool {
        matches!(self, Blob::Mapped(..))
    }
    /// resident heap bytes (mapped blobs live in the page cache, not the heap)
    pub fn heap_bytes(&self) -> usize {
        match self {
            Blob::Owned(v) => v.len(),
            Blob::Mapped(..) => 0,
        }
    }
}

impl std::fmt::Debug for Blob {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Blob({} bytes, {})",
            self.as_slice().len(),
            if self.is_mapped() { "mapped" } else { "owned" }
        )
    }
}

pub struct SectionWriter {
    w: BufWriter<File>,
    pos: u64,
}

impl SectionWriter {
    pub fn create(path: &Path) -> Result<SectionWriter, StoreError> {
        let mut w = BufWriter::with_capacity(1 << 20, File::create(path)?);
        w.write_all(CKPT_MAGIC)?;
        Ok(SectionWriter { w, pos: 8 })
    }

    pub fn section(&mut self, tag: u32, data: &[u8]) -> Result<(), StoreError> {
        self.w.write_all(&tag.to_le_bytes())?;
        self.w.write_all(&0u32.to_le_bytes())?;
        self.w.write_all(&(data.len() as u64).to_le_bytes())?;
        self.w.write_all(data)?;
        let pad = (8 - data.len() % 8) % 8;
        self.w.write_all(&[0u8; 8][..pad])?;
        self.pos += 16 + data.len() as u64 + pad as u64;
        Ok(())
    }

    pub fn pod<T: Pod>(&mut self, tag: u32, v: &[T]) -> Result<(), StoreError> {
        self.section(tag, pod_bytes(v))
    }

    /// flush and fsync
    pub fn finish(mut self) -> Result<u64, StoreError> {
        self.w.flush()?;
        self.w.get_ref().sync_all()?;
        Ok(self.pos)
    }
}

pub struct Sections {
    map: Arc<memmap2::Mmap>,
    table: Vec<(u32, usize, usize)>,
}

impl Sections {
    pub fn open(path: &Path) -> Result<Sections, StoreError> {
        let f = File::open(path)?;
        // SAFETY: checkpoint files are written once (renamed into place) and never modified afterwards
        let map = unsafe { memmap2::Mmap::map(&f)? };
        if map.len() < 8 || &map[..8] != CKPT_MAGIC {
            return Err(StoreError::Format(format!(
                "{} is not a xerj-mem checkpoint",
                path.display()
            )));
        }
        let mut table = Vec::new();
        let mut i = 8usize;
        while i + 16 <= map.len() {
            let tag = u32::from_le_bytes(map[i..i + 4].try_into().unwrap());
            let n = u64::from_le_bytes(map[i + 8..i + 16].try_into().unwrap()) as usize;
            let s = i + 16;
            if s + n > map.len() {
                return Err(StoreError::Format("truncated checkpoint section".into()));
            }
            table.push((tag, s, n));
            i = s + n + (8 - n % 8) % 8;
        }
        Ok(Sections {
            map: Arc::new(map),
            table,
        })
    }

    fn find(&self, tag: u32) -> Result<(usize, usize), StoreError> {
        self.table
            .iter()
            .find(|t| t.0 == tag)
            .map(|t| (t.1, t.2))
            .ok_or_else(|| StoreError::Format(format!("checkpoint section {tag:#x} missing")))
    }

    pub fn has(&self, tag: u32) -> bool {
        self.table.iter().any(|t| t.0 == tag)
    }

    pub fn bytes(&self, tag: u32) -> Result<&[u8], StoreError> {
        let (s, n) = self.find(tag)?;
        Ok(&self.map[s..s + n])
    }

    pub fn blob(&self, tag: u32) -> Result<Blob, StoreError> {
        let (s, n) = self.find(tag)?;
        Ok(Blob::Mapped(self.map.clone(), s, n))
    }

    pub fn vec<T: Pod>(&self, tag: u32) -> Result<Vec<T>, StoreError> {
        pod_vec(self.bytes(tag)?)
    }

    pub fn tags(&self) -> impl Iterator<Item = u32> + '_ {
        self.table.iter().map(|t| t.0)
    }
}
