//! Tokenization. The reference is the Hugging Face fast tokenizer (Rust `tokenizers` 0.22.2, the version the Python
//! reference runs) loaded from the pinned Ettin-17M `tokenizer.json`; segments are encoded without special tokens, as
//! `tok(text, add_special_tokens=False)["input_ids"]` does in inputs.py / xr_codecs.py.

use crate::error::{Error, Result};

pub trait TextEncoder: Send + Sync {
    /// token ids of one text segment, no special tokens
    fn encode(&self, text: &str) -> Result<Vec<u32>>;
}

#[cfg(feature = "tokenizer")]
pub struct HfTokenizer {
    inner: tokenizers::Tokenizer,
}

#[cfg(feature = "tokenizer")]
impl HfTokenizer {
    pub fn from_file(path: &std::path::Path) -> Result<Self> {
        let inner = tokenizers::Tokenizer::from_file(path)
            .map_err(|e| Error::Tokenizer(format!("{}: {e}", path.display())))?;
        Ok(HfTokenizer { inner })
    }
}

#[cfg(feature = "tokenizer")]
impl TextEncoder for HfTokenizer {
    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let enc = self
            .inner
            .encode_fast(text, false)
            .map_err(|e| Error::Tokenizer(e.to_string()))?;
        Ok(enc.get_ids().to_vec())
    }
}

/// sha256 of a file (tokenizer identity check against the export's config.json)
pub fn sha256_file(path: &std::path::Path) -> Result<String> {
    let data = std::fs::read(path).map_err(|e| Error::Io(format!("{}: {e}", path.display())))?;
    Ok(crate::sha256::hex(&crate::sha256::digest(&data)))
}
