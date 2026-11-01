//! Text left in the original file, the indexer's half (see
//! `xerj_storage::stored_codec::source_ref`). With `XERJ_AUTOINDEX_SOURCE_REF=1` and
//! `XERJ_SOURCE_REF_DIR` set, the long string fields of an uncompressed JSONL file get a reference
//! instead of a copy in the segment. The server decides per row and keeps the text on any doubt.
//! Off by default, because the index then depends on the file staying as it is.
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::fs::File;
use std::path::{Path, PathBuf};
use xerj_pagedir::{ArticleKey, BuildOptions, PageDir, ReadOptions};

/// Same name as `xerj_storage::stored_codec::source_ref::REF_FIELD`; this crate talks to the server
/// over HTTP and does not link storage.
const REF_FIELD: &str = "__xerj_source_ref__";

/// Shorter strings cost less to store than to reference.
const MIN_FIELD_BYTES: usize = 256;

pub fn root_from_env() -> Option<PathBuf> {
    let on = std::env::var("XERJ_AUTOINDEX_SOURCE_REF")
        .is_ok_and(|v| matches!(v.trim(), "1" | "true" | "on" | "yes"));
    on.then(|| std::env::var_os("XERJ_SOURCE_REF_DIR"))
        .flatten()
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
}

/// A file with a directory on disk, ready to reference.
pub struct Prepared {
    name: String,
    dir: PageDir,
    file: File,
}

impl Prepared {
    /// Builds the file's directory under `root` unless this version of the file has one. `Err`
    /// means this file keeps its text in the index; indexing goes on.
    pub fn open(root: &Path, path: &Path) -> anyhow::Result<Self> {
        std::fs::create_dir_all(root)?;
        let path = path.canonicalize()?;
        let md = std::fs::metadata(&path)?;
        let modified = md.modified()?.duration_since(std::time::UNIX_EPOCH)?;
        let mut id = Sha256::new();
        id.update(path.to_string_lossy().as_bytes());
        id.update(md.len().to_le_bytes());
        id.update(modified.as_nanos().to_le_bytes());
        let digest = id.finalize();
        let name: String = digest[..16].iter().map(|b| format!("{b:02x}")).collect();
        let out = root.join(format!("{name}.pgdir"));
        if !out.exists() {
            let opts = BuildOptions {
                article_key: ArticleKey::None,
                ..Default::default()
            };
            xerj_pagedir::build(&path, &out, opts)?;
        }
        let note = root.join(format!("{name}.source"));
        if !note.exists() {
            let tmp = root.join(format!("{name}.source.tmp"));
            std::fs::write(&tmp, path.to_string_lossy().as_bytes())?;
            std::fs::rename(&tmp, &note)?;
        }
        Ok(Self {
            dir: xerj_pagedir::open(&out)?,
            file: File::open(&path)?,
            name,
        })
    }

    /// Adds the reference to a record read at byte `offset`. Does nothing when the file has moved on
    /// from its directory or no field is long enough.
    pub fn attach(&self, fields: &mut Map<String, Value>, offset: u64) {
        let Ok(passage) = self.dir.passage_at(&self.file, offset) else {
            return;
        };
        let Ok(line) = self
            .dir
            .read_unchecked(&self.file, passage, ReadOptions::default())
        else {
            return;
        };
        let Ok(parsed) = serde_json::from_slice::<Map<String, Value>>(&line) else {
            return;
        };
        let names: Vec<&String> = fields
            .iter()
            .filter(|(key, value)| {
                value
                    .as_str()
                    .is_some_and(|text| text.len() >= MIN_FIELD_BYTES)
                    && parsed.get(*key) == Some(*value)
            })
            .map(|(key, _)| key)
            .collect();
        if names.is_empty() {
            return;
        }
        let reference = json!({
            "dir": self.name, "p": passage, "fields": names,
            "h": format!("{:016x}", xerj_pagedir::line_hash(&line)),
        });
        fields.insert(REF_FIELD.to_owned(), reference);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn long_fields_get_a_reference_and_short_ones_do_not() {
        let t = tempfile::tempdir().unwrap();
        let path = t.path().join("c.jsonl");
        let long = "word ".repeat(100);
        let lines = [
            json!({"id": "1#0", "text": long}).to_string(),
            json!({"id": "1#1", "text": "short"}).to_string(),
        ];
        std::fs::write(&path, lines.join("\n") + "\n").unwrap();
        let prepared = Prepared::open(&t.path().join("refs"), &path).unwrap();

        let mut record = serde_json::from_str::<Map<String, Value>>(&lines[0]).unwrap();
        prepared.attach(&mut record, 0);
        let reference = &record[REF_FIELD];
        assert_eq!(reference["p"], 0);
        assert_eq!(reference["fields"], json!(["text"]));

        let at = lines[0].len() as u64 + 1;
        let mut short = serde_json::from_str::<Map<String, Value>>(&lines[1]).unwrap();
        prepared.attach(&mut short, at);
        assert!(short.get(REF_FIELD).is_none());

        // A value the indexer changed is not in the file: no reference.
        let mut edited = serde_json::from_str::<Map<String, Value>>(&lines[0]).unwrap();
        edited.insert("text".into(), json!("x".repeat(400)));
        prepared.attach(&mut edited, 0);
        assert!(edited.get(REF_FIELD).is_none());

        // The same file again reuses its directory.
        Prepared::open(&t.path().join("refs"), &path).unwrap();
    }
}
