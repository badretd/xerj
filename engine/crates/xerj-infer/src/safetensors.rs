//! Minimal read-only safetensors reader over a memory map (F32 tensors only; that is what the exporter writes).
//! Tensors are returned as zero-copy `&[f32]` views when the data is 4-byte aligned in the map (it is for files written by
//! the Python `safetensors` package, whose header is padded to 8 bytes), otherwise copied.

use crate::error::{Error, Result};
use memmap2::Mmap;
use serde_json::Value;
use std::borrow::Cow;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub shape: Vec<usize>,
    pub start: usize,
    pub end: usize,
}

pub struct SafeTensors {
    map: Mmap,
    data_start: usize,
    pub tensors: HashMap<String, TensorInfo>,
}

impl SafeTensors {
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path).map_err(|e| Error::Io(format!("{}: {e}", path.display())))?;
        // SAFETY: the file is opened read-only and treated as immutable for the life of the map.
        let map = unsafe { Mmap::map(&file) }
            .map_err(|e| Error::Io(format!("mmap {}: {e}", path.display())))?;
        if map.len() < 8 {
            return Err(Error::Format("safetensors file too short".into()));
        }
        let n = u64::from_le_bytes(map[0..8].try_into().unwrap()) as usize;
        if 8 + n > map.len() {
            return Err(Error::Format("safetensors header exceeds file".into()));
        }
        let header: Value = serde_json::from_slice(&map[8..8 + n])
            .map_err(|e| Error::Format(format!("safetensors header: {e}")))?;
        let obj = header
            .as_object()
            .ok_or_else(|| Error::Format("safetensors header is not an object".into()))?;
        let mut tensors = HashMap::new();
        for (name, info) in obj {
            if name == "__metadata__" {
                continue;
            }
            let dtype = info.get("dtype").and_then(|d| d.as_str()).unwrap_or("");
            if dtype != "F32" {
                return Err(Error::Unsupported(format!(
                    "tensor {name} has dtype {dtype}, expected F32"
                )));
            }
            let shape: Vec<usize> = info
                .get("shape")
                .and_then(|s| s.as_array())
                .ok_or_else(|| Error::Format(format!("{name}: shape")))?
                .iter()
                .map(|x| x.as_u64().unwrap_or(0) as usize)
                .collect();
            let offs = info
                .get("data_offsets")
                .and_then(|s| s.as_array())
                .ok_or_else(|| Error::Format(format!("{name}: offsets")))?;
            let (start, end) = (
                offs[0].as_u64().unwrap_or(0) as usize,
                offs[1].as_u64().unwrap_or(0) as usize,
            );
            let numel: usize = shape.iter().product();
            if end < start || end - start != 4 * numel || 8 + n + end > map.len() {
                return Err(Error::Format(format!("{name}: inconsistent data offsets")));
            }
            tensors.insert(name.clone(), TensorInfo { shape, start, end });
        }
        Ok(SafeTensors {
            map,
            data_start: 8 + n,
            tensors,
        })
    }

    pub fn info(&self, name: &str) -> Result<&TensorInfo> {
        self.tensors
            .get(name)
            .ok_or_else(|| Error::Format(format!("missing tensor {name}")))
    }

    pub fn has(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    /// raw little-endian bytes of a tensor
    pub fn bytes(&self, name: &str) -> Result<&[u8]> {
        let t = self.info(name)?;
        Ok(&self.map[self.data_start + t.start..self.data_start + t.end])
    }

    /// f32 view (zero-copy when aligned) and its shape
    pub fn f32(&self, name: &str) -> Result<(Cow<'_, [f32]>, Vec<usize>)> {
        let shape = self.info(name)?.shape.clone();
        let b = self.bytes(name)?;
        if cfg!(target_endian = "little") && (b.as_ptr() as usize) % 4 == 0 {
            // SAFETY: alignment checked, length is a multiple of 4 (checked at open), f32 has no invalid bit patterns.
            let s = unsafe { std::slice::from_raw_parts(b.as_ptr() as *const f32, b.len() / 4) };
            Ok((Cow::Borrowed(s), shape))
        } else {
            let v = b
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            Ok((Cow::Owned(v), shape))
        }
    }

    /// f32 tensor with an expected shape, copied
    pub fn f32_vec(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
        let (v, s) = self.f32(name)?;
        if s != shape {
            return Err(Error::Format(format!(
                "{name}: shape {s:?}, expected {shape:?}"
            )));
        }
        Ok(v.into_owned())
    }

    /// scalar tensor (shape [] or [1])
    pub fn scalar(&self, name: &str) -> Result<f32> {
        let (v, _) = self.f32(name)?;
        if v.len() != 1 {
            return Err(Error::Format(format!("{name}: not a scalar")));
        }
        Ok(v[0])
    }

    /// release the resident pages of the map (they were copied into packed buffers at load; embedding rows fault back in
    /// lazily on first use). Read-only shared file mapping: MADV_DONTNEED only drops page-table entries, contents are kept.
    pub fn release_pages(&self) {
        // SAFETY: read-only file-backed mapping; dropping its pages cannot change what later reads observe.
        #[cfg(unix)]
        unsafe {
            let _ = self
                .map
                .unchecked_advise(memmap2::UncheckedAdvice::DontNeed);
        }
    }

    /// the whole map (keeps zero-copy views alive)
    pub fn map(&self) -> &Mmap {
        &self.map
    }

    pub fn data_offset(&self, name: &str) -> Result<usize> {
        Ok(self.data_start + self.info(name)?.start)
    }
}
