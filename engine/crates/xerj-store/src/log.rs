//! Append-only binary op log (the durable form of the store; checkpoints are an optimisation on top).
//!
//! File = `XERJLOG1` + frames; frame = u32 LE payload length, u32 LE crc32(payload), payload. Payload = op tag + fields
//! (strings as varint length + UTF-8, integers as varints). Only ops that the store ACCEPTED are logged, as the caller
//! wrote them (raw values, not canonical forms), so a replay re-runs the same contract checks and reproduces the same
//! revisions. Group commit: `append` encodes into a buffer, `commit` issues one write (+ optional fdatasync). Replay maps
//! the file and decodes zero-copy; a torn or corrupt tail frame ends the replay and is reported (`valid_len`) so the
//! writer can truncate it.

use crate::error::{ContractError, StoreError};
use crate::store::{FactIn, ValueIn};
use smallvec::SmallVec;
use std::fs::{File, OpenOptions};
use std::io::{Seek, SeekFrom, Write};
use std::path::Path;

pub const LOG_MAGIC: &[u8; 8] = b"XERJLOG1";

const T_ADD_DOC: u8 = 1;
const T_ADD_FACT: u8 = 2;
const T_WITHDRAW_FACT: u8 = 3;
const T_WITHDRAW_DOC: u8 = 4;

/// One logged write (a transient borrowed view; never stored in bulk).
#[derive(Debug, Clone)]
#[allow(clippy::large_enum_variant)]
pub enum Op<'a> {
    AddDocument {
        doc_id: &'a str,
        version: &'a str,
        text: &'a str,
    },
    AddFact {
        fact: FactIn<'a>,
        supersedes: SmallVec<[&'a str; 1]>,
    },
    WithdrawFact {
        fact_id: &'a str,
    },
    WithdrawDocument {
        doc_id: &'a str,
        version: &'a str,
    },
}

#[inline]
fn put_varint(out: &mut Vec<u8>, mut v: u64) {
    while v >= 0x80 {
        out.push((v as u8) | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

#[inline]
fn put_str(out: &mut Vec<u8>, s: &str) {
    put_varint(out, s.len() as u64);
    out.extend_from_slice(s.as_bytes());
}

#[inline]
fn zigzag(v: i64) -> u64 {
    ((v << 1) ^ (v >> 63)) as u64
}

#[inline]
fn unzigzag(v: u64) -> i64 {
    ((v >> 1) as i64) ^ -((v & 1) as i64)
}

/// Encode one op payload (without frame) into `out`.
pub fn encode_op(op: &Op<'_>, out: &mut Vec<u8>) {
    match op {
        Op::AddDocument {
            doc_id,
            version,
            text,
        } => {
            out.push(T_ADD_DOC);
            put_str(out, doc_id);
            put_str(out, version);
            put_str(out, text);
        }
        Op::AddFact {
            fact: f,
            supersedes,
        } => {
            out.push(T_ADD_FACT);
            for s in [f.fact_id, f.subject, f.relation, f.kind] {
                put_str(out, s);
            }
            match f.value {
                ValueIn::Str(s) => {
                    out.push(0);
                    put_str(out, s);
                }
                ValueIn::Bool(b) => out.push(if b { 2 } else { 1 }),
                ValueIn::Other => out.push(3),
            }
            for s in [f.unit, f.doc_id, f.version] {
                put_str(out, s);
            }
            put_varint(out, zigzag(f.start));
            put_varint(out, zigzag(f.end));
            put_str(out, f.quote);
            put_str(out, f.scope);
            let flags = (f.subject_id.is_some() as u8)
                | (f.value_id.is_some() as u8) << 1
                | (f.valid_from.is_some() as u8) << 2
                | (f.valid_to.is_some() as u8) << 3;
            out.push(flags);
            for s in [f.subject_id, f.value_id, f.valid_from, f.valid_to]
                .into_iter()
                .flatten()
            {
                put_str(out, s);
            }
            put_str(out, f.writer_version);
            put_varint(out, supersedes.len() as u64);
            for s in supersedes {
                put_str(out, s);
            }
        }
        Op::WithdrawFact { fact_id } => {
            out.push(T_WITHDRAW_FACT);
            put_str(out, fact_id);
        }
        Op::WithdrawDocument { doc_id, version } => {
            out.push(T_WITHDRAW_DOC);
            put_str(out, doc_id);
            put_str(out, version);
        }
    }
}

struct Cur<'a> {
    b: &'a [u8],
    i: usize,
}

impl<'a> Cur<'a> {
    fn u8(&mut self) -> Option<u8> {
        let v = *self.b.get(self.i)?;
        self.i += 1;
        Some(v)
    }
    fn varint(&mut self) -> Option<u64> {
        let mut v = 0u64;
        let mut shift = 0;
        loop {
            let c = self.u8()?;
            if shift >= 64 {
                return None;
            }
            v |= ((c & 0x7f) as u64) << shift;
            if c < 0x80 {
                return Some(v);
            }
            shift += 7;
        }
    }
    fn str(&mut self) -> Option<&'a str> {
        let n = self.varint()? as usize;
        let s = self.b.get(self.i..self.i.checked_add(n)?)?;
        self.i += n;
        std::str::from_utf8(s).ok()
    }
}

/// Decode one payload.
pub fn decode_op(payload: &[u8]) -> Option<Op<'_>> {
    let mut c = Cur { b: payload, i: 0 };
    let op = match c.u8()? {
        T_ADD_DOC => Op::AddDocument {
            doc_id: c.str()?,
            version: c.str()?,
            text: c.str()?,
        },
        T_ADD_FACT => {
            let (fact_id, subject, relation, kind) = (c.str()?, c.str()?, c.str()?, c.str()?);
            let value = match c.u8()? {
                0 => ValueIn::Str(c.str()?),
                1 => ValueIn::Bool(false),
                2 => ValueIn::Bool(true),
                3 => ValueIn::Other,
                _ => return None,
            };
            let (unit, doc_id, version) = (c.str()?, c.str()?, c.str()?);
            let start = unzigzag(c.varint()?);
            let end = unzigzag(c.varint()?);
            let (quote, scope) = (c.str()?, c.str()?);
            let flags = c.u8()?;
            let subject_id = if flags & 1 != 0 { Some(c.str()?) } else { None };
            let value_id = if flags & 2 != 0 { Some(c.str()?) } else { None };
            let valid_from = if flags & 4 != 0 { Some(c.str()?) } else { None };
            let valid_to = if flags & 8 != 0 { Some(c.str()?) } else { None };
            let writer_version = c.str()?;
            let n = c.varint()? as usize;
            let mut supersedes = SmallVec::new();
            for _ in 0..n {
                supersedes.push(c.str()?);
            }
            Op::AddFact {
                fact: FactIn {
                    fact_id,
                    subject,
                    relation,
                    kind,
                    value,
                    unit,
                    doc_id,
                    version,
                    start,
                    end,
                    quote,
                    scope,
                    subject_id,
                    value_id,
                    valid_from,
                    valid_to,
                    writer_version,
                },
                supersedes,
            }
        }
        T_WITHDRAW_FACT => Op::WithdrawFact { fact_id: c.str()? },
        T_WITHDRAW_DOC => Op::WithdrawDocument {
            doc_id: c.str()?,
            version: c.str()?,
        },
        _ => return None,
    };
    (c.i == payload.len()).then_some(op)
}

/// Group-commit log writer.
pub struct LogWriter {
    file: File,
    buf: Vec<u8>,
    frame: Vec<u8>,
    durable: bool,
    written: u64,
    ops: u64,
}

impl LogWriter {
    /// Create a new log (truncating) or open an existing one for appending (`valid_len` from a replay truncates a torn tail).
    pub fn open(
        path: &Path,
        durable: bool,
        valid_len: Option<u64>,
    ) -> Result<LogWriter, StoreError> {
        let exists = path.exists();
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(path)?;
        let mut written = file.metadata()?.len();
        if !exists || written == 0 {
            file.write_all(LOG_MAGIC)?;
            written = LOG_MAGIC.len() as u64;
        } else if let Some(v) = valid_len {
            if v < written {
                file.set_len(v)?;
                written = v;
            }
        }
        file.seek(SeekFrom::Start(written))?;
        Ok(LogWriter {
            file,
            buf: Vec::with_capacity(1 << 20),
            frame: Vec::with_capacity(256),
            durable,
            written,
            ops: 0,
        })
    }

    /// Buffer one op (visible in memory already; durable after `commit`).
    pub fn append(&mut self, op: &Op<'_>) {
        self.frame.clear();
        encode_op(op, &mut self.frame);
        let crc = crc32fast::hash(&self.frame);
        self.buf
            .extend_from_slice(&(self.frame.len() as u32).to_le_bytes());
        self.buf.extend_from_slice(&crc.to_le_bytes());
        self.buf.extend_from_slice(&self.frame);
        self.ops += 1;
        if self.buf.len() >= (8 << 20) {
            let _ = self.flush_buf();
        }
    }

    fn flush_buf(&mut self) -> std::io::Result<()> {
        if !self.buf.is_empty() {
            self.file.write_all(&self.buf)?;
            self.written += self.buf.len() as u64;
            self.buf.clear();
        }
        Ok(())
    }

    /// Write buffered ops (one write call) and, when durable, fdatasync.
    pub fn commit(&mut self) -> std::io::Result<()> {
        self.flush_buf()?;
        if self.durable {
            self.file.sync_data()?;
        }
        Ok(())
    }

    /// bytes in the file after the last commit (+ buffered)
    pub fn len(&self) -> u64 {
        self.written + self.buf.len() as u64
    }
    pub fn is_empty(&self) -> bool {
        self.len() <= LOG_MAGIC.len() as u64
    }
    pub fn ops(&self) -> u64 {
        self.ops
    }
}

/// Outcome of a replay.
#[derive(Debug, Clone, Copy, Default)]
pub struct ReplayStats {
    pub ops: u64,
    /// byte length of the valid prefix (a torn / corrupt tail starts here)
    pub valid_len: u64,
    pub torn_tail: bool,
}

/// Replay `path` from byte offset `from` (>= 8; a checkpoint records its log offset), calling `apply` per op.
pub fn replay(
    path: &Path,
    from: u64,
    mut apply: impl FnMut(&Op<'_>) -> Result<(), ContractError>,
) -> Result<ReplayStats, StoreError> {
    let file = File::open(path)?;
    let len = file.metadata()?.len();
    if len < LOG_MAGIC.len() as u64 {
        return Ok(ReplayStats {
            ops: 0,
            valid_len: 0,
            torn_tail: len > 0,
        });
    }
    // SAFETY: the log is append-only and not modified while mapped by this process
    let map = unsafe { memmap2::Mmap::map(&file)? };
    if &map[..8] != LOG_MAGIC {
        return Err(StoreError::Format("not a xerj-mem log".into()));
    }
    let mut i = from.max(8) as usize;
    let mut st = ReplayStats {
        ops: 0,
        valid_len: i as u64,
        torn_tail: false,
    };
    while i < map.len() {
        if i + 8 > map.len() {
            st.torn_tail = true;
            break;
        }
        let n = u32::from_le_bytes(map[i..i + 4].try_into().unwrap()) as usize;
        let crc = u32::from_le_bytes(map[i + 4..i + 8].try_into().unwrap());
        let Some(payload) = map.get(i + 8..i + 8 + n) else {
            st.torn_tail = true;
            break;
        };
        if crc32fast::hash(payload) != crc {
            st.torn_tail = true;
            break;
        }
        let op = decode_op(payload)
            .ok_or_else(|| StoreError::Format(format!("undecodable op at byte {i}")))?;
        apply(&op)?;
        st.ops += 1;
        i += 8 + n;
        st.valid_len = i as u64;
    }
    Ok(st)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::{Store, DEFAULT_SCOPE, ORACLE_WRITER};

    #[test]
    fn roundtrip_and_torn_tail() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("ops.log");
        let mut w = LogWriter::open(&p, false, None).unwrap();
        w.append(&Op::AddDocument {
            doc_id: "d",
            version: "v1",
            text: "X is 5 m.",
        });
        let f = FactIn {
            fact_id: "f",
            subject: "X",
            relation: "height",
            kind: "number",
            value: ValueIn::Str("5"),
            unit: "m",
            doc_id: "d",
            version: "v1",
            start: 5,
            end: 6,
            quote: "5",
            scope: DEFAULT_SCOPE,
            subject_id: Some("Q1"),
            value_id: None,
            valid_from: None,
            valid_to: Some("2030-01-01"),
            writer_version: ORACLE_WRITER,
        };
        w.append(&Op::AddFact {
            fact: f,
            supersedes: SmallVec::new(),
        });
        w.append(&Op::WithdrawFact { fact_id: "f" });
        w.commit().unwrap();
        // torn tail: a partial frame
        {
            let mut fh = OpenOptions::new().append(true).open(&p).unwrap();
            fh.write_all(&[9, 0, 0, 0, 1, 2]).unwrap();
        }
        let mut s = Store::default();
        let st = replay(&p, 0, |op| match op {
            Op::AddDocument {
                doc_id,
                version,
                text,
            } => s.add_document(doc_id, version, text).map(|_| ()),
            Op::AddFact { fact, supersedes } => s.add_fact(fact, supersedes).map(|_| ()),
            Op::WithdrawFact { fact_id } => s.withdraw_fact(fact_id).map(|_| ()),
            Op::WithdrawDocument { doc_id, version } => {
                s.withdraw_document(doc_id, version).map(|_| ())
            }
        })
        .unwrap();
        assert_eq!(st.ops, 3);
        assert!(st.torn_tail);
        assert_eq!(s.revision(), 3);
        assert_eq!(s.subject_id(0), Some("Q1"));
        let w2 = LogWriter::open(&p, false, Some(st.valid_len)).unwrap();
        assert_eq!(w2.len(), st.valid_len);
    }
}
