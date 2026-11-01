//! Text left in the original file (page pointers).
//!
//! An index built with this option keeps no copy of a line-oriented file's long fields. A document
//! carries `__xerj_source_ref__`:
//!
//! ```json
//! {"dir": "<directory name>", "p": 12345, "h": "<16 hex>", "fields": ["text"]}
//! ```
//!
//! `dir` names `<name>.pgdir` and `<name>.source` (the file's path) under the server's
//! `XERJ_SOURCE_REF_DIR`. `p` is the line's passage number and `h` its [`xerj_pagedir::line_hash`].
//!
//! * **Encode** ([`strip_source`]): each named field is replaced by `""` only after the line read
//!   through the directory has the stored hash and the field's value equals the document's. Any
//!   doubt keeps the text in the segment, so ingest never loses data. The empty value keeps the
//!   column, so projections work; the marker `"s":true` says the row needs its text back.
//! * **Decode** ([`rehydrate_source_map`]): the line is read and its hash checked before any text
//!   is returned. A changed or missing file gives `__xerj_source_ref_error__` in `_source` and the
//!   field stays empty: an error, never wrong text.
//! * Merges read rows with the marker in place and stay stripped; they never touch the file.
//!
//! A client cannot read other files through this: a directory has to exist under the server's root
//! (only the indexer on that host writes there), its name cannot leave the root, and a read needs
//! the hash of the line it asks for.
use serde_json::{Map, Value};
use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock, RwLock};
use xerj_pagedir::{Expect, PageDir, ReadOptions};

pub const REF_FIELD: &str = "__xerj_source_ref__";
pub const ERROR_FIELD: &str = "__xerj_source_ref_error__";
const STRIPPED_KEY: &str = "s";

fn root_slot() -> &'static RwLock<Option<PathBuf>> {
    static ROOT: OnceLock<RwLock<Option<PathBuf>>> = OnceLock::new();
    ROOT.get_or_init(|| {
        RwLock::new(
            std::env::var_os("XERJ_SOURCE_REF_DIR")
                .filter(|v| !v.is_empty())
                .map(PathBuf::from),
        )
    })
}

/// Sets the root in place of `XERJ_SOURCE_REF_DIR` (tests and embedders).
pub fn set_root(root: Option<PathBuf>) {
    *root_slot().write().unwrap() = root;
    cache().lock().unwrap().clear();
}

struct Opened {
    dir: PageDir,
    file: File,
}

fn cache() -> &'static Mutex<HashMap<String, Arc<Opened>>> {
    static CACHE: OnceLock<Mutex<HashMap<String, Arc<Opened>>>> = OnceLock::new();
    CACHE.get_or_init(Default::default)
}

fn valid_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 128
        && name
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || c == b'-' || c == b'_')
}

fn open_at(root: &Path, name: &str) -> Result<Opened, String> {
    let dir = xerj_pagedir::open(&root.join(format!("{name}.pgdir"))).map_err(|e| e.to_string())?;
    let source = std::fs::read_to_string(root.join(format!("{name}.source")))
        .map_err(|e| format!("source note of {name}: {e}"))?;
    let file = File::open(source.trim_end_matches('\n'))
        .map_err(|e| format!("stale file: cannot open the original of {name}: {e}"))?;
    Ok(Opened { dir, file })
}

/// Open directories stay cached; a failed open does not, so a restored file works again. A file
/// replaced under an open descriptor keeps serving its old bytes, whose hashes still hold.
fn opened(name: &str) -> Result<Arc<Opened>, String> {
    if !valid_name(name) {
        return Err(format!("bad directory name {name:?}"));
    }
    let guard = root_slot().read().unwrap();
    let root = guard
        .as_deref()
        .ok_or_else(|| "XERJ_SOURCE_REF_DIR is not set on this server".to_string())?;
    if let Some(found) = cache().lock().unwrap().get(name) {
        return Ok(found.clone());
    }
    let fresh = Arc::new(open_at(root, name)?);
    cache()
        .lock()
        .unwrap()
        .insert(name.to_owned(), fresh.clone());
    Ok(fresh)
}

fn read_line(reference: &Map<String, Value>) -> Result<Map<String, Value>, String> {
    let name = reference
        .get("dir")
        .and_then(Value::as_str)
        .ok_or("reference without dir")?;
    let passage = reference
        .get("p")
        .and_then(Value::as_u64)
        .ok_or("reference without p")?;
    let hash = reference
        .get("h")
        .and_then(Value::as_str)
        .and_then(|h| u64::from_str_radix(h, 16).ok())
        .ok_or("reference without h")?;
    let open = opened(name)?;
    let line = open
        .dir
        .read(
            &open.file,
            passage,
            Expect::LineHash(hash),
            ReadOptions::default(),
        )
        .map_err(|e| e.to_string())?;
    serde_json::from_slice(&line).map_err(|e| format!("stale file: line {passage}: {e}"))
}

fn fields(reference: &Map<String, Value>) -> impl Iterator<Item = &str> {
    reference
        .get("fields")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
}

/// Encode side: `source` with its named fields emptied, or `None` to store it as it is.
pub fn strip_source(source: &Value) -> Option<Value> {
    let obj = source.as_object()?;
    let reference = obj.get(REF_FIELD)?.as_object()?;
    if reference.contains_key(STRIPPED_KEY) {
        return None;
    }
    let line = read_line(reference).ok()?;
    let names: Vec<&str> = fields(reference).collect();
    let same = |name: &str| {
        matches!(
            (obj.get(name), line.get(name)),
            (Some(a @ Value::String(_)), Some(b)) if a == b
        )
    };
    if names.is_empty() || !names.iter().copied().all(same) {
        return None;
    }
    let mut out = obj.clone();
    for name in &names {
        out.insert((*name).to_owned(), Value::String(String::new()));
    }
    let marked = out.get_mut(REF_FIELD)?.as_object_mut()?;
    marked.insert(STRIPPED_KEY.to_owned(), Value::Bool(true));
    Some(Value::Object(out))
}

/// Decode side: put the text back. Returns whether anything changed.
pub fn rehydrate_source_map(map: &mut Map<String, Value>) -> bool {
    let Some(reference) = map.get(REF_FIELD).and_then(Value::as_object) else {
        return false;
    };
    if !reference.contains_key(STRIPPED_KEY) {
        return false;
    }
    let reference = reference.clone();
    let line = match read_line(&reference) {
        Ok(line) => line,
        Err(error) => {
            map.insert(ERROR_FIELD.to_owned(), Value::String(error));
            return true;
        }
    };
    for name in fields(&reference) {
        if let Some(value) = line.get(name) {
            map.insert(name.to_owned(), value.clone());
        }
    }
    if let Some(marked) = map.get_mut(REF_FIELD).and_then(Value::as_object_mut) {
        marked.remove(STRIPPED_KEY);
    }
    true
}

#[cfg(test)]
mod tests {
    use super::super::*;
    use super::*;
    use serde_json::json;

    /// Serialises tests over the process-wide root.
    static LOCK: Mutex<()> = Mutex::new(());

    struct Fixture {
        _guard: std::sync::MutexGuard<'static, ()>,
        tmp: tempfile::TempDir,
        file: PathBuf,
        lines: Vec<String>,
    }

    fn fixture(n: usize) -> Fixture {
        let guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let tmp = tempfile::tempdir().unwrap();
        let file = tmp.path().join("orig.jsonl");
        let lines: Vec<String> = (0..n)
            .map(|i| {
                json!({"id": format!("{}#{}", i / 4, i % 4), "text": format!("line {i} {}", "w ".repeat(i % 90))})
                    .to_string()
                    .replace("{\"id\":\"", "{\"id\": \"")
            })
            .collect();
        std::fs::write(&file, lines.join("\n") + "\n").unwrap();
        xerj_pagedir::build(
            &file,
            &tmp.path().join("d1.pgdir"),
            xerj_pagedir::BuildOptions::default(),
        )
        .unwrap();
        std::fs::write(tmp.path().join("d1.source"), file.to_str().unwrap()).unwrap();
        set_root(Some(tmp.path().to_owned()));
        Fixture {
            _guard: guard,
            tmp,
            file,
            lines,
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            set_root(None);
        }
    }

    fn doc(f: &Fixture, i: usize) -> Value {
        let line: Value = serde_json::from_str(&f.lines[i]).unwrap();
        json!({
            "_id": format!("d{i}"), "_seq_no": i, "_source": {
                "id": line["id"], "text": line["text"], "kind": "k",
                REF_FIELD: {
                    "dir": "d1", "p": i, "fields": ["text"],
                    "h": format!("{:016x}", xerj_pagedir::line_hash(f.lines[i].as_bytes())),
                },
            }
        })
    }

    fn section(f: &Fixture, n: usize) -> (Vec<u8>, Vec<u8>) {
        let docs: Vec<Value> = (0..n).map(|i| doc(f, i)).collect();
        let json = serde_json::to_vec(&docs).unwrap();
        (encode_stored_v2(&json), json)
    }

    #[test]
    fn stripped_rows_read_back_the_same_and_stay_small() {
        let f = fixture(500);
        let (stripped, json) = section(&f, 500);
        let mut full: Vec<Value> = serde_json::from_slice(&json).unwrap();
        let back: Vec<Value> = serde_json::from_slice(&decode_stored(&stripped).unwrap()).unwrap();
        assert_eq!(back.len(), full.len());
        for doc in &mut full {
            doc.as_object_mut().unwrap().remove("_seq_no");
        }
        for (b, want) in back.iter().zip(&full) {
            assert_eq!(b["_source"], want["_source"]);
        }
        set_root(None);
        let kept = encode_stored_v2(&json);
        set_root(Some(f.tmp.path().to_owned()));
        assert!(
            stripped.len() < kept.len(),
            "{} vs {}",
            stripped.len(),
            kept.len()
        );
        // The rewrite path keeps the row stripped without opening the file.
        std::fs::remove_file(&f.file).unwrap();
        set_root(Some(f.tmp.path().to_owned()));
        let rewrite = decode_stored_for_rewrite(&stripped).unwrap();
        let merged = encode_stored_v2_merge_at_level(&rewrite, 3);
        assert!(merged.len() < kept.len());
    }

    #[test]
    fn a_projection_gets_the_text() {
        let f = fixture(300);
        let (stripped, _) = section(&f, 300);
        let rows = match decode_stored_v2_rows_projected(&stripped, &[7, 250], &["text"]).unwrap() {
            StoredV2RowHydrationResult::Hydrated { rows, .. } => rows,
            other => panic!("{other:?}"),
        };
        let want: Value = serde_json::from_str(&f.lines[250]).unwrap();
        assert_eq!(rows[1].source.get("text"), Some(&want["text"]));
        assert_eq!(rows[1].source.len(), 1);
    }

    #[test]
    fn a_changed_file_gives_an_error_never_wrong_text() {
        let f = fixture(300);
        let (stripped, _) = section(&f, 300);
        // Edit keeps the size: only the per-read hash can see it.
        let mut changed = std::fs::read(&f.file).unwrap();
        let at = changed.windows(6).position(|w| w == b"line 5").unwrap();
        changed[at] = b'L';
        std::fs::write(&f.file, &changed).unwrap();
        set_root(Some(f.tmp.path().to_owned()));
        let back: Vec<Value> = serde_json::from_slice(&decode_stored(&stripped).unwrap()).unwrap();
        for (i, b) in back.iter().enumerate() {
            let want: Value = serde_json::from_str(&f.lines[i]).unwrap();
            let source = &b["_source"];
            if source.get(ERROR_FIELD).is_some() {
                assert_eq!(source["text"], "");
                assert!(source[ERROR_FIELD]
                    .as_str()
                    .unwrap()
                    .starts_with("stale file"));
            } else {
                assert_eq!(source["text"], want["text"], "row {i}");
            }
        }
        assert!(back.iter().any(|b| b["_source"].get(ERROR_FIELD).is_some()));
    }

    #[test]
    fn doubt_keeps_the_text_in_the_segment() {
        let f = fixture(300);
        // Wrong hash, a value that differs from the line, an unknown directory, a name that tries
        // to leave the root.
        type Edit = Box<dyn Fn(&mut Value)>;
        let cases: Vec<Edit> = vec![
            Box::new(|d| d["_source"][REF_FIELD]["h"] = json!("0000000000000000")),
            Box::new(|d| d["_source"]["text"] = json!("not the line")),
            Box::new(|d| d["_source"][REF_FIELD]["dir"] = json!("nope")),
            Box::new(|d| d["_source"][REF_FIELD]["dir"] = json!("../d1")),
        ];
        for edit in &cases {
            let docs: Vec<Value> = (0..300)
                .map(|i| {
                    let mut d = doc(&f, i);
                    edit(&mut d);
                    d
                })
                .collect();
            let json = serde_json::to_vec(&docs).unwrap();
            let section = encode_stored_v2(&json);
            let back: Vec<Value> =
                serde_json::from_slice(&decode_stored(&section).unwrap()).unwrap();
            for (b, d) in back.iter().zip(&docs) {
                assert_eq!(b["_source"]["text"], d["_source"]["text"]);
            }
        }
    }

    #[test]
    fn no_root_keeps_the_text() {
        let f = fixture(300);
        set_root(None);
        let (kept, json) = section(&f, 300);
        let docs: Vec<Value> = serde_json::from_slice(&json).unwrap();
        let back: Vec<Value> = serde_json::from_slice(&decode_stored(&kept).unwrap()).unwrap();
        assert_eq!(back[9]["_source"]["text"], docs[9]["_source"]["text"]);
    }
}
