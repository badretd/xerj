//! Entity linking, Python-exact (`xr_codecs.link_entities`) and fast.
//!
//! Reference: snapshot subjects sorted by (-len, s) (len in code points); each subject's FIRST occurrence in the
//! progressively masked question that satisfies `(?<!\w)` / `(?!\w)` (Python's Unicode `\w`) is taken and masked with
//! spaces; the result is the found subjects ordered by position. The reference does one regex search per snapshot subject
//! (O(#subjects x |q|)); here:
//!
//! 1. an occurrence finder lists the raw occurrences of store subjects in the question in O(|q|)-ish time:
//!    * `SpanHash` (default): probe the subject interner's hash table with every span that can possibly become a valid
//!      match: spans starting at a word start (or at a non-word char that begins some subject) and ending at a word end
//!      (or at a non-word char that ends some subject). Zero build cost, zero extra memory, O(1) per new subject.
//!    * `AhoCorasick`: overlapping matches of tiered Aho-Corasick automata (aho-corasick crate, MatchKind::Standard; the
//!      brief's leftmost-longest mode cannot reproduce the reference: with subjects "A B" and "B C D" in "A B C D" it
//!      reports "A B" where the reference, longest first, links "B C D"). New subjects go to a lazily rebuilt tail
//!      automaton; tails are sealed into tiers that merge geometrically (amortized O(log N) rebuilds per subject).
//! 2. the masking simulation replays the reference loop over only the subjects that occur, longest first.
//!
//! Why the occurrence lists are complete (proof sketch, see README): masking replaces chars by U+0020. A subject that
//! neither starts nor ends with U+0020 can never newly match in the masked text (an overlapped masked span would lie
//! strictly inside it, but every masked span is at least as long as it), so its only candidates are its original
//! occurrences, valid iff they overlap no masked position and their boundaries hold in the masked text; and a subject
//! that starts (ends) with a word char can only start (end) where the ORIGINAL text has a word boundary. Subjects that
//! start or end with U+0020 ("odd") are matched by a direct scan of the masked text in their turn.

use crate::subjects::SubjectIndex;
use aho_corasick::{AhoCorasick, AhoCorasickKind, MatchKind};
use hashbrown::HashSet;
use smallvec::SmallVec;
use std::sync::RwLock;
use xerj_store::text::{is_py_word, normalize_entity};
use xerj_store::{Snapshot, Store, StrPool};

/// A raw occurrence of a subject in a question: code-point span [start, end).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Occ {
    pub start: u32,
    pub end: u32,
    pub subject: u32,
}

/// Set of chars (ASCII bitmap + overflow set).
#[derive(Default, Clone)]
pub struct CharSet {
    ascii: [u64; 2],
    other: HashSet<char>,
}

impl CharSet {
    #[inline]
    pub fn insert(&mut self, c: char) {
        let cp = c as u32;
        if cp < 128 {
            self.ascii[(cp >> 6) as usize] |= 1 << (cp & 63);
        } else {
            self.other.insert(c);
        }
    }
    #[inline]
    pub fn contains(&self, c: char) -> bool {
        let cp = c as u32;
        if cp < 128 {
            self.ascii[(cp >> 6) as usize] & (1 << (cp & 63)) != 0
        } else {
            !self.other.is_empty() && self.other.contains(&c)
        }
    }
}

/// Shape statistics of the subject vocabulary (maintained in O(1) per new subject).
#[derive(Default, Clone)]
pub struct SubjectShape {
    /// non-word (and not U+0020) chars that begin / end some subject
    pub first_nonword: CharSet,
    pub last_nonword: CharSet,
    /// first / last chars of every subject (substring enumeration for the verify rule)
    pub first_any: CharSet,
    pub last_any: CharSet,
    /// longest subject in code points
    pub max_len: u32,
    /// subjects starting or ending with U+0020 (matched by direct scan)
    pub odd: Vec<u32>,
    pub seen: usize,
}

impl SubjectShape {
    pub fn catch_up(&mut self, pool: &StrPool) {
        for s in self.seen as u32..pool.len() as u32 {
            let t = pool.get(s);
            let (Some(f), Some(l)) = (t.chars().next(), t.chars().next_back()) else {
                continue;
            };
            let len = t.chars().count() as u32;
            self.max_len = self.max_len.max(len);
            self.first_any.insert(f);
            self.last_any.insert(l);
            if f == ' ' || l == ' ' {
                self.odd.push(s);
                continue;
            }
            if !is_py_word(f) {
                self.first_nonword.insert(f);
            }
            if !is_py_word(l) {
                self.last_nonword.insert(l);
            }
        }
        self.seen = pool.len();
    }
}

/// Per-call buffers (reuse one per thread to keep linking allocation-free).
#[derive(Default)]
pub struct QBuf {
    pub chars: Vec<char>,
    /// byte offset of char i (n + 1 entries)
    pub byte_at: Vec<u32>,
    pub word: Vec<bool>,
    /// char index of a byte offset (valid at char boundaries; AC matches)
    pub char_at_byte: Vec<u32>,
    pub masked: Vec<bool>,
    pub occs: Vec<Occ>,
    starts: Vec<u32>,
    ends: Vec<u32>,
    groups: Vec<Group>,
    found: Vec<(u32, u32)>,
    odd_chars: Vec<char>,
}

#[derive(Clone, Copy)]
struct Group {
    subject: u32,
    len: u32,
    lo: u32,
    hi: u32,
    odd: bool,
}

impl QBuf {
    pub fn load(&mut self, q: &str) {
        self.chars.clear();
        self.byte_at.clear();
        self.word.clear();
        self.char_at_byte.clear();
        self.char_at_byte.resize(q.len() + 1, u32::MAX);
        for (b, c) in q.char_indices() {
            self.char_at_byte[b] = self.chars.len() as u32;
            self.byte_at.push(b as u32);
            self.chars.push(c);
            self.word.push(is_py_word(c));
        }
        self.byte_at.push(q.len() as u32);
        self.char_at_byte[q.len()] = self.chars.len() as u32;
        self.masked.clear();
        self.masked.resize(self.chars.len(), false);
        self.occs.clear();
    }
    #[inline]
    pub fn n(&self) -> usize {
        self.chars.len()
    }
    #[inline]
    fn slice<'q>(&self, q: &'q str, i: u32, j: u32) -> &'q str {
        &q[self.byte_at[i as usize] as usize..self.byte_at[j as usize] as usize]
    }
}

// ================================================================================================ span-hash finder

/// Raw occurrences that can become valid reference matches (see module docs), via subject-interner probes.
pub fn span_hash_occurrences(pool: &StrPool, shape: &SubjectShape, q: &str, b: &mut QBuf) {
    let n = b.n() as u32;
    b.starts.clear();
    b.ends.clear();
    for i in 0..n {
        let iu = i as usize;
        let w = b.word[iu];
        if (w && (i == 0 || !b.word[iu - 1])) || (!w && shape.first_nonword.contains(b.chars[iu])) {
            b.starts.push(i);
        }
    }
    for j in 1..=n {
        let ju = j as usize;
        let w = b.word[ju - 1];
        if (w && (j == n || !b.word[ju])) || (!w && shape.last_nonword.contains(b.chars[ju - 1])) {
            b.ends.push(j);
        }
    }
    let mut e0 = 0usize;
    for si in 0..b.starts.len() {
        let i = b.starts[si];
        while e0 < b.ends.len() && b.ends[e0] <= i {
            e0 += 1;
        }
        for ei in e0..b.ends.len() {
            let j = b.ends[ei];
            if j - i > shape.max_len {
                break;
            }
            let s = b.slice(q, i, j);
            if let Some(sym) = pool.lookup(s) {
                b.occs.push(Occ {
                    start: i,
                    end: j,
                    subject: sym,
                });
            }
        }
    }
}

/// Distinct store subjects (any state) that occur in `q` as plain substrings (`s in q`), counting up to `cap`.
pub fn span_hash_substring_subjects(
    pool: &StrPool,
    shape: &SubjectShape,
    q: &str,
    b: &mut QBuf,
    cap: usize,
    out: &mut SmallVec<[u32; 4]>,
) {
    let n = b.n() as u32;
    for i in 0..n {
        if !shape.first_any.contains(b.chars[i as usize]) {
            continue;
        }
        let jmax = n.min(i + shape.max_len);
        for j in i + 1..=jmax {
            if !shape.last_any.contains(b.chars[(j - 1) as usize]) {
                continue;
            }
            if let Some(sym) = pool.lookup(b.slice(q, i, j)) {
                if !out.contains(&sym) {
                    out.push(sym);
                    if out.len() >= cap {
                        return;
                    }
                }
            }
        }
    }
}

// ================================================================================================ Aho-Corasick finder

struct Tier {
    lo: u32,
    hi: u32,
    ac: AhoCorasick,
}

#[derive(Default)]
struct AcState {
    tiers: Vec<Tier>,
    tail: Option<Tier>,
    /// subjects [0, sealed) live in `tiers`
    sealed: u32,
    builds: u64,
    build_bytes: u64,
}

/// Tiered Aho-Corasick automata over the subject interner (overlapping search).
pub struct AcFinder {
    state: RwLock<AcState>,
    pub tail_max: u32,
}

impl Default for AcFinder {
    fn default() -> Self {
        AcFinder {
            state: RwLock::new(AcState::default()),
            tail_max: 4096,
        }
    }
}

fn build_tier(pool: &StrPool, lo: u32, hi: u32) -> Tier {
    let ac = AhoCorasick::builder()
        .match_kind(MatchKind::Standard)
        .kind(Some(AhoCorasickKind::ContiguousNFA))
        .build((lo..hi).map(|s| pool.get(s)))
        .expect("aho-corasick build");
    Tier { lo, hi, ac }
}

impl AcFinder {
    /// Bring the automata up to date with `pool` (tail rebuild, or seal + geometric merge).
    pub fn refresh(&self, pool: &StrPool) {
        let n = pool.len() as u32;
        {
            let g = self.state.read().unwrap();
            let covered = g.tail.as_ref().map_or(g.sealed, |t| t.hi);
            if covered >= n {
                return;
            }
        }
        let mut g = self.state.write().unwrap();
        let covered = g.tail.as_ref().map_or(g.sealed, |t| t.hi);
        if covered >= n {
            return;
        }
        let bytes = |lo: u32, hi: u32| (lo..hi).map(|s| pool.get(s).len() as u64).sum::<u64>();
        if n - g.sealed > self.tail_max {
            let lo = g.sealed;
            g.build_bytes += bytes(lo, n);
            g.tiers.push(build_tier(pool, lo, n));
            g.builds += 1;
            g.sealed = n;
            g.tail = None;
            while g.tiers.len() >= 2 {
                let k = g.tiers.len();
                let (a, b) = (&g.tiers[k - 2], &g.tiers[k - 1]);
                if (b.hi - b.lo) * 2 < (a.hi - a.lo) {
                    break;
                }
                let (lo, hi) = (a.lo, b.hi);
                g.tiers.truncate(k - 2);
                g.build_bytes += bytes(lo, hi);
                g.tiers.push(build_tier(pool, lo, hi));
                g.builds += 1;
            }
        } else {
            let lo = g.sealed;
            g.build_bytes += bytes(lo, n);
            g.tail = Some(build_tier(pool, lo, n));
            g.builds += 1;
        }
    }

    /// all overlapping raw occurrences (any subject state)
    pub fn occurrences(&self, pool: &StrPool, q: &str, b: &mut QBuf) {
        self.refresh(pool);
        let g = self.state.read().unwrap();
        for t in g.tiers.iter().chain(g.tail.iter()) {
            for m in t.ac.find_overlapping_iter(q) {
                let s = b.char_at_byte[m.start()];
                let e = b.char_at_byte[m.end()];
                b.occs.push(Occ {
                    start: s,
                    end: e,
                    subject: t.lo + m.pattern().as_u32(),
                });
            }
        }
    }

    /// (automaton builds so far, pattern bytes built, number of tiers, memory bytes of the automata)
    pub fn stats(&self) -> (u64, u64, usize, usize) {
        let g = self.state.read().unwrap();
        let mem = g
            .tiers
            .iter()
            .chain(g.tail.iter())
            .map(|t| t.ac.memory_usage())
            .sum();
        (
            g.builds,
            g.build_bytes,
            g.tiers.len() + g.tail.is_some() as usize,
            mem,
        )
    }
}

// ================================================================================================ linker

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LinkerKind {
    SpanHash,
    AhoCorasick,
}

pub struct Linker {
    pub kind: LinkerKind,
    pub shape: SubjectShape,
    pub ac: AcFinder,
}

impl Linker {
    pub fn new(kind: LinkerKind) -> Linker {
        Linker {
            kind,
            shape: SubjectShape::default(),
            ac: AcFinder::default(),
        }
    }

    /// account for subjects interned since the last call (O(new subjects); AC automata are rebuilt lazily on read)
    pub fn catch_up(&mut self, store: &Store) {
        self.shape.catch_up(&store.pools.subjects);
    }

    fn occurrences(&self, store: &Store, q: &str, b: &mut QBuf) {
        match self.kind {
            LinkerKind::SpanHash => span_hash_occurrences(&store.pools.subjects, &self.shape, q, b),
            LinkerKind::AhoCorasick => self.ac.occurrences(&store.pools.subjects, q, b),
        }
    }

    /// `xr_codecs.link_entities(question, snapshot)` as subject syms (reference order: by position).
    pub fn link_exact(
        &self,
        snap: &Snapshot<'_>,
        idx: &SubjectIndex,
        q: &str,
        b: &mut QBuf,
    ) -> SmallVec<[u32; 8]> {
        let store = snap.store;
        b.load(q);
        self.occurrences(store, q, b);
        link_simulate(snap, idx, &self.shape, q, b)
    }

    /// EXTENSION (the opt-in verify rule `bindings_all`; no manifest reads it): the reference's masking loop
    /// ([`link_simulate_with`]: longest names first, word boundaries, a name inside a longer matched name masked) over
    /// every store subject ever written, any state: the vocabulary [`Linker::substring_subjects`] scans, so a subject
    /// whose records were all withdrawn still counts, but a name nested in a longer matched name does not. Syms in
    /// reference order.
    pub fn link_any_state(&self, store: &Store, q: &str, b: &mut QBuf) -> SmallVec<[u32; 8]> {
        b.load(q);
        self.occurrences(store, q, b);
        link_simulate_with(&store.pools.subjects, &self.shape.odd, |_| true, b)
    }

    /// Distinct store subjects (every record ever written, any state) occurring in `q` as substrings, up to `cap`
    /// (tools/verify_rows.py: `{s for s in by_group[group] if s in q}`).
    pub fn substring_subjects(
        &self,
        store: &Store,
        q: &str,
        b: &mut QBuf,
        cap: usize,
    ) -> SmallVec<[u32; 4]> {
        let mut out = SmallVec::new();
        b.load(q);
        match self.kind {
            LinkerKind::SpanHash => span_hash_substring_subjects(
                &store.pools.subjects,
                &self.shape,
                q,
                b,
                cap,
                &mut out,
            ),
            LinkerKind::AhoCorasick => {
                self.ac.occurrences(&store.pools.subjects, q, b);
                for o in &b.occs {
                    if !out.contains(&o.subject) {
                        out.push(o.subject);
                        if out.len() >= cap {
                            break;
                        }
                    }
                }
            }
        }
        out
    }
}

/// The reference's masking loop over the raw occurrences in `b.occs` (+ the odd subjects).
pub fn link_simulate(
    snap: &Snapshot<'_>,
    idx: &SubjectIndex,
    shape: &SubjectShape,
    q: &str,
    b: &mut QBuf,
) -> SmallVec<[u32; 8]> {
    let _ = q;
    link_simulate_with(
        &snap.store.pools.subjects,
        &shape.odd,
        |s| idx.subject_active(snap, s),
        b,
    )
}

/// The reference's masking loop for any vocabulary of interned strings (`link_entities` over subjects, stack-v5
/// `link_values` over text values): `b.occs` holds the raw occurrences (syms of `pool`), `odd` the vocabulary strings that
/// start or end with U+0020 (scanned directly), `active(sym)` says whether a string belongs to the snapshot's vocabulary.
/// Returns the found syms in reference order (by position, then string).
pub fn link_simulate_with(
    pool: &StrPool,
    odd: &[u32],
    active: impl Fn(u32) -> bool,
    b: &mut QBuf,
) -> SmallVec<[u32; 8]> {
    let n = b.n() as u32;
    b.occs.sort_unstable_by_key(|o| (o.subject, o.start));
    b.occs.dedup();
    b.groups.clear();
    let mut i = 0usize;
    while i < b.occs.len() {
        let s = b.occs[i].subject;
        let mut j = i + 1;
        while j < b.occs.len() && b.occs[j].subject == s {
            j += 1;
        }
        if active(s) {
            let o = b.occs[i];
            b.groups.push(Group {
                subject: s,
                len: o.end - o.start,
                lo: i as u32,
                hi: j as u32,
                odd: false,
            });
        }
        i = j;
    }
    for &s in odd {
        if !b.groups.iter().any(|g| g.subject == s) && active(s) {
            let len = pool.get(s).chars().count() as u32;
            if len <= n {
                b.groups.push(Group {
                    subject: s,
                    len,
                    lo: 0,
                    hi: 0,
                    odd: true,
                });
            }
        }
    }
    if b.groups.is_empty() {
        return SmallVec::new();
    }
    // odd subjects that also produced raw occurrences are scanned directly instead
    for g in b.groups.iter_mut() {
        if !g.odd {
            let t = pool.get(g.subject);
            if t.starts_with(' ') || t.ends_with(' ') {
                g.odd = true;
            }
        }
    }
    b.groups.sort_unstable_by(|x, y| {
        y.len
            .cmp(&x.len)
            .then_with(|| pool.get(x.subject).cmp(pool.get(y.subject)))
    });
    b.found.clear();
    let groups = std::mem::take(&mut b.groups);
    for g in &groups {
        let hit = if g.odd {
            scan_masked(b, pool.get(g.subject), g.len)
        } else {
            let mut hit = None;
            for k in g.lo..g.hi {
                let o = b.occs[k as usize];
                let (s, e) = (o.start as usize, o.end as usize);
                if b.masked[s..e].iter().any(|&m| m) {
                    continue;
                }
                let before_ok = s == 0 || !(b.word[s - 1] && !b.masked[s - 1]);
                let after_ok = e == n as usize || !(b.word[e] && !b.masked[e]);
                if before_ok && after_ok {
                    hit = Some(o.start);
                    break;
                }
            }
            hit
        };
        if let Some(st) = hit {
            for p in st..st + g.len {
                b.masked[p as usize] = true;
            }
            b.found.push((st, g.subject));
        }
    }
    b.groups = groups;
    b.found
        .sort_unstable_by(|x, y| x.0.cmp(&y.0).then_with(|| pool.get(x.1).cmp(pool.get(y.1))));
    b.found.iter().map(|x| x.1).collect()
}

/// leftmost valid match of `subject` in the masked text (direct scan; odd subjects only)
fn scan_masked(b: &mut QBuf, subject: &str, len: u32) -> Option<u32> {
    b.odd_chars.clear();
    b.odd_chars.extend(subject.chars());
    let n = b.n();
    let len = len as usize;
    if len > n {
        return None;
    }
    let c_at = |b: &QBuf, p: usize| if b.masked[p] { ' ' } else { b.chars[p] };
    let w_at = |b: &QBuf, p: usize| !b.masked[p] && b.word[p];
    'outer: for s in 0..=n - len {
        for k in 0..len {
            if c_at(b, s + k) != b.odd_chars[k] {
                continue 'outer;
            }
        }
        let before_ok = s == 0 || !w_at(b, s - 1);
        let after_ok = s + len == n || !w_at(b, s + len);
        if before_ok && after_ok {
            return Some(s as u32);
        }
    }
    None
}

// ================================================================================================ extension: normalized linking

/// EXTENSION (not reference semantics): subjects whose `normalize_entity` form occurs token-aligned in the normalized
/// question, leftmost-longest, active in `snap`. Recovers case / punctuation variants the exact linker misses
/// ("Course For Collision" vs subject "Course for Collision"). Used by the BM25-assisted fallback, never by the exact
/// manifest.
pub fn link_normalized(
    snap: &Snapshot<'_>,
    idx: &SubjectIndex,
    q: &str,
    max_tokens: usize,
) -> SmallVec<[u32; 8]> {
    let store = snap.store;
    let nq = normalize_entity(q);
    let toks: Vec<(usize, usize)> = {
        let mut v = Vec::new();
        let mut st = None;
        for (i, c) in nq.char_indices() {
            if c == ' ' {
                if let Some(s) = st.take() {
                    v.push((s, i));
                }
            } else if st.is_none() {
                st = Some(i);
            }
        }
        if let Some(s) = st {
            v.push((s, nq.len()));
        }
        v
    };
    let mut out: SmallVec<[u32; 8]> = SmallVec::new();
    let mut t = 0;
    let mut cand: SmallVec<[u32; 4]> = SmallVec::new();
    while t < toks.len() {
        let mut best: Option<(usize, SmallVec<[u32; 4]>)> = None;
        for e in (t..toks.len().min(t + max_tokens)).rev() {
            let span = &nq[toks[t].0..toks[e].1];
            cand.clear();
            idx.subjects_with_norm(store, span, &mut cand);
            let act: SmallVec<[u32; 4]> = cand
                .iter()
                .copied()
                .filter(|&s| idx.subject_active(snap, s))
                .collect();
            if !act.is_empty() {
                best = Some((e, act));
                break;
            }
        }
        match best {
            Some((e, act)) => {
                for s in act {
                    if !out.contains(&s) {
                        out.push(s);
                    }
                }
                t = e + 1;
            }
            None => t += 1,
        }
    }
    out
}
