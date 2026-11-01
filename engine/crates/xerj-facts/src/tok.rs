//! The compiler's text encoder over the engine's own `tokenizers` build (0.21, fancy-regex). The export's reference
//! tokenizer is the Python reference's (`tokenizers` 0.22.2); the file identity is checked by sha256 at load, and the token
//! ids must match the golden decision inputs (checked by the spike harness, not assumed).

use xerj_infer::TextEncoder;

pub struct EngineTokenizer {
    inner: tokenizers::Tokenizer,
}

impl EngineTokenizer {
    pub fn from_file(path: &std::path::Path) -> Result<Self, String> {
        tokenizers::Tokenizer::from_file(path)
            .map(|inner| EngineTokenizer { inner })
            .map_err(|e| format!("{}: {e}", path.display()))
    }
}

impl TextEncoder for EngineTokenizer {
    /// token ids of one segment without special tokens, as `tok(text, add_special_tokens=False)` in the reference
    fn encode(&self, text: &str) -> xerj_infer::Result<Vec<u32>> {
        let enc = self
            .inner
            .encode_fast(text, false)
            .map_err(|e| xerj_infer::Error::Tokenizer(e.to_string()))?;
        Ok(enc.get_ids().to_vec())
    }
}
