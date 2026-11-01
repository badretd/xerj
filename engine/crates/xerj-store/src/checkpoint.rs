//! Store checkpoint: every pool arena and column written raw (no per-record encoding), reloaded by mmap + memcpy; only
//! the hash tables (pool lookups, key index, document keys) are rebuilt on load. A checkpoint records the op-log offset
//! it covers, so `open` = load checkpoint + replay the log tail.

use crate::ckpt::{SectionWriter, Sections};
use crate::error::StoreError;
use crate::pool::StrPool;
use crate::store::{Docs, FingerprintMode, Pools, Records, Store, StoreConfig, NONE};
use crate::value::Kind;
use foldhash::fast::FixedState;
use hashbrown::HashMap;
use smallvec::SmallVec;
use std::path::Path;

const T_META: u32 = 0x100;
const T_POOL: u32 = 0x200; // + 2 * k (bytes), + 2 * k + 1 (ends)
const T_REC: u32 = 0x300; // + column index
const T_DOC: u32 = 0x400; // + column index

const POOL_NAMES: [&str; 10] = [
    "fact_ids",
    "subjects",
    "relations",
    "strs",
    "units",
    "scopes",
    "ent_ids",
    "doc_ids",
    "versions",
    "writers",
];

fn pools_ref(p: &Pools) -> [&StrPool; 10] {
    [
        &p.fact_ids,
        &p.subjects,
        &p.relations,
        &p.strs,
        &p.units,
        &p.scopes,
        &p.ent_ids,
        &p.doc_ids,
        &p.versions,
        &p.writers,
    ]
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
pub struct CheckpointMeta {
    pub format: u32,
    pub revision: u32,
    pub records: u64,
    pub documents: u64,
    pub fingerprint_mode: String,
    pub store_id: String,
    /// op-log byte offset covered by this checkpoint (replay continues from here)
    pub log_offset: u64,
    pub pools: Vec<String>,
}

/// Write the store's sections into `w` (callers add their own sections, e.g. indexes, to the same file).
pub fn write_store(
    store: &Store,
    w: &mut SectionWriter,
    log_offset: u64,
) -> Result<(), StoreError> {
    let (rev, cfg) = store.raw_parts();
    let meta = CheckpointMeta {
        format: 1,
        revision: rev,
        records: store.num_records() as u64,
        documents: store.num_docs() as u64,
        fingerprint_mode: match cfg.fingerprint {
            FingerprintMode::Exact => "exact".into(),
            FingerprintMode::Revision => "revision".into(),
        },
        store_id: cfg.store_id.clone(),
        log_offset,
        pools: POOL_NAMES.iter().map(|s| s.to_string()).collect(),
    };
    w.section(T_META, &serde_json::to_vec(&meta)?)?;
    for (k, p) in pools_ref(&store.pools).iter().enumerate() {
        let (bytes, ends) = p.raw_parts();
        w.section(T_POOL + 2 * k as u32, bytes)?;
        w.pod(T_POOL + 2 * k as u32 + 1, ends)?;
    }
    let r = &store.recs;
    w.pod(T_REC, &r.subject)?;
    w.pod(T_REC + 1, &r.relation)?;
    w.pod(T_REC + 2, &r.scope)?;
    let kinds: Vec<u8> = r.kind.iter().map(|k| *k as u8).collect();
    w.pod(T_REC + 3, &kinds)?;
    w.pod(T_REC + 4, &r.value)?;
    w.pod(T_REC + 5, &r.unit)?;
    w.pod(T_REC + 6, &r.doc)?;
    w.pod(T_REC + 7, &r.start)?;
    w.pod(T_REC + 8, &r.end)?;
    w.pod(T_REC + 9, &r.quote)?;
    w.pod(T_REC + 10, &r.subject_id)?;
    w.pod(T_REC + 11, &r.value_id)?;
    w.pod(T_REC + 12, &r.writer)?;
    w.pod(T_REC + 13, &r.added)?;
    w.pod(T_REC + 14, &r.retired)?;
    w.pod(T_REC + 15, &r.vis_end)?;
    let mut iv: Vec<u32> = Vec::with_capacity(r.interval.len() * 3);
    let mut keys: Vec<_> = r.interval.iter().collect();
    keys.sort_unstable_by_key(|e| *e.0);
    for (&rid, &(a, b)) in keys {
        iv.extend_from_slice(&[rid, a, b]);
    }
    w.pod(T_REC + 16, &iv)?;
    let d = &store.docs;
    w.pod(T_DOC, &d.doc_id)?;
    w.pod(T_DOC + 1, &d.version)?;
    w.pod(T_DOC + 2, &d.text_end)?;
    w.section(T_DOC + 3, &d.text)?;
    let ascii: Vec<u8> = d.ascii.iter().map(|&b| b as u8).collect();
    w.pod(T_DOC + 4, &ascii)?;
    w.pod(T_DOC + 5, &d.added)?;
    w.pod(T_DOC + 6, &d.withdrawn)?;
    let mut offs: Vec<u32> = Vec::with_capacity(d.recs.len() + 1);
    let mut vals: Vec<u32> = Vec::with_capacity(store.num_records());
    offs.push(0);
    for v in &d.recs {
        vals.extend_from_slice(v);
        offs.push(vals.len() as u32);
    }
    w.pod(T_DOC + 7, &offs)?;
    w.pod(T_DOC + 8, &vals)?;
    Ok(())
}

/// Checkpoint just the store to `path` (atomic: written to `path.tmp`, then renamed).
pub fn save_store(store: &Store, path: &Path, log_offset: u64) -> Result<u64, StoreError> {
    let tmp = path.with_extension("tmp");
    let mut w = SectionWriter::create(&tmp)?;
    write_store(store, &mut w, log_offset)?;
    let n = w.finish()?;
    std::fs::rename(&tmp, path)?;
    Ok(n)
}

/// Load the store sections of a checkpoint; returns the store and its metadata.
pub fn read_store(s: &Sections) -> Result<(Store, CheckpointMeta), StoreError> {
    let meta: CheckpointMeta = serde_json::from_slice(s.bytes(T_META)?)?;
    if meta.format != 1 {
        return Err(StoreError::Format(format!(
            "unsupported checkpoint format {}",
            meta.format
        )));
    }
    let mut pools: Vec<StrPool> = Vec::with_capacity(10);
    for k in 0..10u32 {
        let bytes = s.bytes(T_POOL + 2 * k)?.to_vec();
        let ends: Vec<u32> = s.vec(T_POOL + 2 * k + 1)?;
        pools.push(StrPool::from_raw_parts(bytes, ends).map_err(StoreError::Format)?);
    }
    let mut it = pools.into_iter();
    let mut next = || it.next().unwrap();
    let pools = Pools {
        fact_ids: next(),
        subjects: next(),
        relations: next(),
        strs: next(),
        units: next(),
        scopes: next(),
        ent_ids: next(),
        doc_ids: next(),
        versions: next(),
        writers: next(),
    };
    let kinds: Vec<u8> = s.vec(T_REC + 3)?;
    let kind = kinds
        .iter()
        .map(|&k| Kind::from_u8(k).ok_or_else(|| StoreError::Format("bad kind".into())))
        .collect::<Result<Vec<_>, _>>()?;
    let iv: Vec<u32> = s.vec(T_REC + 16)?;
    let mut interval = HashMap::with_hasher(FixedState::with_seed(0x5845_524a_7374_6f72));
    let (triples, _) = iv.as_chunks::<3>();
    for c in triples {
        interval.insert(c[0], (c[1], c[2]));
    }
    let recs = Records {
        subject: s.vec(T_REC)?,
        relation: s.vec(T_REC + 1)?,
        scope: s.vec(T_REC + 2)?,
        kind,
        value: s.vec(T_REC + 4)?,
        unit: s.vec(T_REC + 5)?,
        doc: s.vec(T_REC + 6)?,
        start: s.vec(T_REC + 7)?,
        end: s.vec(T_REC + 8)?,
        quote: s.vec(T_REC + 9)?,
        subject_id: s.vec(T_REC + 10)?,
        value_id: s.vec(T_REC + 11)?,
        writer: s.vec(T_REC + 12)?,
        added: s.vec(T_REC + 13)?,
        retired: s.vec(T_REC + 14)?,
        vis_end: s.vec(T_REC + 15)?,
        next_in_key: Vec::new(),
        interval,
    };
    let n = recs.subject.len();
    for col in [
        &recs.relation,
        &recs.scope,
        &recs.value,
        &recs.doc,
        &recs.start,
        &recs.end,
        &recs.quote,
        &recs.subject_id,
        &recs.value_id,
        &recs.added,
        &recs.retired,
        &recs.vis_end,
    ] {
        if col.len() != n {
            return Err(StoreError::Format("record columns differ in length".into()));
        }
    }
    if recs.kind.len() != n || recs.unit.len() != n || recs.writer.len() != n {
        return Err(StoreError::Format("record columns differ in length".into()));
    }
    let offs: Vec<u32> = s.vec(T_DOC + 7)?;
    let vals: Vec<u32> = s.vec(T_DOC + 8)?;
    let mut drecs = Vec::with_capacity(offs.len().saturating_sub(1));
    for w in offs.windows(2) {
        drecs.push(SmallVec::from_slice(&vals[w[0] as usize..w[1] as usize]));
    }
    let docs = Docs {
        doc_id: s.vec(T_DOC)?,
        version: s.vec(T_DOC + 1)?,
        text_end: s.vec(T_DOC + 2)?,
        text: s.bytes(T_DOC + 3)?.to_vec(),
        ascii: s
            .vec::<u8>(T_DOC + 4)?
            .into_iter()
            .map(|b| b != 0)
            .collect(),
        added: s.vec(T_DOC + 5)?,
        withdrawn: s.vec(T_DOC + 6)?,
        recs: drecs,
    };
    if std::str::from_utf8(&docs.text).is_err() {
        return Err(StoreError::Format("document arena is not UTF-8".into()));
    }
    let cfg = StoreConfig {
        fingerprint: if meta.fingerprint_mode == "revision" {
            FingerprintMode::Revision
        } else {
            FingerprintMode::Exact
        },
        store_id: meta.store_id.clone(),
    };
    for &x in recs.subject_id.iter().chain(recs.value_id.iter()) {
        if x != NONE && x as usize >= pools.ent_ids.len() {
            return Err(StoreError::Format("entity id out of range".into()));
        }
    }
    let store =
        Store::from_parts(cfg, meta.revision, pools, recs, docs).map_err(StoreError::Format)?;
    Ok((store, meta))
}

pub fn load_store(path: &Path) -> Result<(Store, CheckpointMeta), StoreError> {
    let s = Sections::open(path)?;
    read_store(&s)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::FactIn;

    #[test]
    fn checkpoint_roundtrip() {
        let mut s = Store::default();
        s.add_document("d1", "v1", "Ännä is 5 m tall. Bo is blue.")
            .unwrap();
        s.add_fact(
            &FactIn {
                kind: "number",
                unit: "m",
                ..FactIn::text("f1", "Ännä", "height", "5", ("d1", "v1"), (8, 9), "5")
            },
            &[],
        )
        .unwrap();
        s.add_fact(
            &FactIn::text("f2", "Bo", "color", "blue", ("d1", "v1"), (24, 28), "blue"),
            &[],
        )
        .unwrap();
        s.add_document("d1", "v2", "Bo is red.").unwrap();
        s.add_fact(
            &FactIn::text("f3", "Bo", "color", "red", ("d1", "v2"), (6, 9), "red"),
            &["f2"],
        )
        .unwrap();
        let fp = s.head().fingerprint();
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("store.ckp");
        save_store(&s, &p, 42).unwrap();
        let (t, meta) = load_store(&p).unwrap();
        assert_eq!(meta.log_offset, 42);
        assert_eq!(t.revision(), s.revision());
        assert_eq!(&*t.head().fingerprint(), &*fp);
        assert_eq!(
            t.head()
                .lookup("Bo", "color", "default")
                .value
                .unwrap()
                .value,
            "red"
        );
        let va = crate::store::ValidAt::parse(crate::store::VALID_AT).unwrap();
        assert_eq!(
            t.at(3, va)
                .lookup("Bo", "color", "default")
                .value
                .unwrap()
                .value,
            "blue"
        );
    }
}
