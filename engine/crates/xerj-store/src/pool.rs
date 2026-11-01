//! `StrPool`: an append-only string interner. Strings live back to back in one byte arena (no per-string allocation);
//! a SwissTable of u32 ids (4 bytes + 1 control byte per slot) maps a string to its dense id. Ids are assigned in insertion
//! order, so a pool doubles as a column (e.g. fact ids: pool id == record id).

use foldhash::fast::FixedState;
use hashbrown::HashTable;
use std::hash::BuildHasher;

const SEED: u64 = 0x5845_524a_2d6d_656d; // "XERJ-mem"

#[derive(Clone)]
pub struct StrPool {
    bytes: Vec<u8>,
    ends: Vec<u32>,
    table: HashTable<u32>,
    state: FixedState,
}

impl Default for StrPool {
    fn default() -> Self {
        Self::new()
    }
}

impl StrPool {
    pub fn new() -> Self {
        StrPool {
            bytes: Vec::new(),
            ends: Vec::new(),
            table: HashTable::new(),
            state: FixedState::with_seed(SEED),
        }
    }

    pub fn with_capacity(strings: usize, bytes: usize) -> Self {
        StrPool {
            bytes: Vec::with_capacity(bytes),
            ends: Vec::with_capacity(strings),
            table: HashTable::with_capacity(strings),
            state: FixedState::with_seed(SEED),
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.ends.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.ends.is_empty()
    }

    #[inline]
    pub fn hash_str(&self, s: &str) -> u64 {
        self.state.hash_one(s.as_bytes())
    }

    #[inline]
    fn range(ends: &[u32], id: u32) -> (usize, usize) {
        let i = id as usize;
        let start = if i == 0 { 0 } else { ends[i - 1] as usize };
        (start, ends[i] as usize)
    }

    /// The string of `id` (panics on an unknown id).
    #[inline]
    pub fn get(&self, id: u32) -> &str {
        let (s, e) = Self::range(&self.ends, id);
        // SAFETY: only whole &str values are ever appended to the arena
        unsafe { std::str::from_utf8_unchecked(&self.bytes[s..e]) }
    }

    /// Id of `s` if interned.
    #[inline]
    pub fn lookup(&self, s: &str) -> Option<u32> {
        self.lookup_hashed(s, self.hash_str(s))
    }

    #[inline]
    pub fn lookup_hashed(&self, s: &str, hash: u64) -> Option<u32> {
        let (bytes, ends) = (&self.bytes, &self.ends);
        self.table
            .find(hash, |&id| {
                let (a, b) = Self::range(ends, id);
                &bytes[a..b] == s.as_bytes()
            })
            .copied()
    }

    /// Id of `s`, interning it when new.
    #[inline]
    pub fn intern(&mut self, s: &str) -> u32 {
        match self.insert_new(s) {
            Ok(id) | Err(id) => id,
        }
    }

    /// Interns `s` when new (`Ok(id)`); `Err(existing id)` when already present.
    pub fn insert_new(&mut self, s: &str) -> Result<u32, u32> {
        let hash = self.hash_str(s);
        self.insert_new_hashed(s, hash)
    }

    /// `insert_new` with a precomputed `hash_str(s)` (callers that already probed with `lookup_hashed`)
    pub fn insert_new_hashed(&mut self, s: &str, hash: u64) -> Result<u32, u32> {
        let StrPool {
            bytes,
            ends,
            table,
            state,
        } = self;
        let eq = |&id: &u32| {
            let (a, b) = Self::range(ends, id);
            &bytes[a..b] == s.as_bytes()
        };
        if let Some(&id) = table.find(hash, eq) {
            return Err(id);
        }
        let id = ends.len() as u32;
        assert!(id < u32::MAX, "StrPool: more than u32::MAX strings");
        bytes.extend_from_slice(s.as_bytes());
        assert!(
            bytes.len() <= u32::MAX as usize,
            "StrPool: arena exceeds 4 GiB"
        );
        ends.push(bytes.len() as u32);
        let (bytes, ends) = (&*bytes, &*ends);
        table.insert_unique(hash, id, |&x| {
            let (a, b) = Self::range(ends, x);
            state.hash_one(&bytes[a..b])
        });
        Ok(id)
    }

    /// Appends `s` WITHOUT a uniqueness check or table entry (used by checkpoint loading, which rebuilds the table).
    fn push_raw(&mut self, s: &[u8]) {
        self.bytes.extend_from_slice(s);
        self.ends.push(self.bytes.len() as u32);
    }

    /// Iterator over (id, string).
    pub fn iter(&self) -> impl Iterator<Item = (u32, &str)> + '_ {
        (0..self.ends.len() as u32).map(move |i| (i, self.get(i)))
    }

    /// Reserve room for `strings` more strings of `bytes` more bytes.
    pub fn reserve(&mut self, strings: usize, bytes: usize) {
        self.bytes.reserve(bytes);
        self.ends.reserve(strings);
        let (b, e, st) = (&self.bytes, &self.ends, &self.state);
        self.table.reserve(strings, |&x| {
            let (a, z) = Self::range(e, x);
            st.hash_one(&b[a..z])
        });
    }

    /// Release spare capacity (after a bulk load).
    pub fn shrink_to_fit(&mut self) {
        self.bytes.shrink_to_fit();
        self.ends.shrink_to_fit();
        let (b, e, st) = (&self.bytes, &self.ends, &self.state);
        self.table.shrink_to_fit(|&x| {
            let (a, z) = Self::range(e, x);
            st.hash_one(&b[a..z])
        });
    }

    /// Heap bytes held (arena + ends + table).
    pub fn heap_bytes(&self) -> usize {
        self.bytes.capacity() + self.ends.capacity() * 4 + self.table.capacity() * (4 + 1)
    }

    /// Raw parts for persistence: (arena, ends).
    pub fn raw_parts(&self) -> (&[u8], &[u32]) {
        (&self.bytes, &self.ends)
    }

    /// Rebuild from raw parts (validates UTF-8 boundaries cheaply: the whole arena must be UTF-8 and ends monotone).
    pub fn from_raw_parts(bytes: Vec<u8>, ends: Vec<u32>) -> Result<Self, String> {
        if std::str::from_utf8(&bytes).is_err() {
            return Err("pool arena is not UTF-8".into());
        }
        let mut prev = 0u32;
        for &e in &ends {
            if e < prev || e as usize > bytes.len() {
                return Err("pool ends not monotone".into());
            }
            let s = &bytes[prev as usize..e as usize];
            if std::str::from_utf8(s).is_err() {
                return Err("pool string not on a char boundary".into());
            }
            prev = e;
        }
        let mut p = StrPool {
            bytes,
            ends,
            table: HashTable::new(),
            state: FixedState::with_seed(SEED),
        };
        p.rebuild_table();
        Ok(p)
    }

    fn rebuild_table(&mut self) {
        let n = self.ends.len();
        let mut table = HashTable::with_capacity(n);
        let (bytes, ends, state) = (&self.bytes, &self.ends, &self.state);
        for id in 0..n as u32 {
            let (a, b) = Self::range(ends, id);
            let h = state.hash_one(&bytes[a..b]);
            table.insert_unique(h, id, |&x| {
                let (a, b) = Self::range(ends, x);
                state.hash_one(&bytes[a..b])
            });
        }
        self.table = table;
    }

    #[allow(dead_code)]
    pub(crate) fn extend_raw<'a>(&mut self, items: impl Iterator<Item = &'a [u8]>) {
        for s in items {
            self.push_raw(s);
        }
        self.rebuild_table();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intern_roundtrip() {
        let mut p = StrPool::new();
        let a = p.intern("alpha");
        let b = p.intern("beta");
        assert_eq!(p.intern("alpha"), a);
        assert_eq!(p.insert_new("beta"), Err(b));
        assert_eq!(p.get(a), "alpha");
        assert_eq!(p.lookup("gamma"), None);
        for i in 0..10_000 {
            p.intern(&format!("k{i}"));
        }
        assert_eq!(p.lookup("k9999").map(|i| p.get(i)), Some("k9999"));
        let (bytes, ends) = p.raw_parts();
        let q = StrPool::from_raw_parts(bytes.to_vec(), ends.to_vec()).unwrap();
        assert_eq!(q.lookup("k77"), p.lookup("k77"));
        assert_eq!(q.len(), p.len());
    }
}
