//! Posting list encoding and decoding for xerj FTS.
//!
//! ## Format
//!
//! Posting lists are split into **128-doc blocks** (matching Lucene's default).
//! Within each block, doc IDs are delta-encoded and PFOR-packed using the
//! `bitpacking` crate for SIMD-accelerated bit-manipulation.
//!
//! Per-term layout written into the `.post` blob (see
//! [`PostingsWriter::encode_term`]):
//!
//! ```text
//! ┌─────────────────────────────────────────┐
//! │  Block 0: packed doc_id deltas          │
//! │           packed term_freqs (positioned)│
//! │           positions (vbyte, positioned) │
//! ├─────────────────────────────────────────┤
//! │  Block 1 …                              │
//! ├─────────────────────────────────────────┤
//! │  Residual (< 128 docs, vbyte)           │
//! └─────────────────────────────────────────┘
//! ```
//!
//! ## Block framing: ZPS2 (current) and ZPS1 (legacy)
//!
//! The `.post` envelope magic doubles as the inner-framing version.  A `ZPS2`
//! file uses the framing below; a `ZPS1` file uses the older
//! `[num_bits ≥ 1][u32 byte_len][payload]` framing, which the reader still
//! decodes (see [`PostCodec`]).
//!
//! ZPS2 packed streams (doc-id deltas, term freqs) are frame-of-reference
//! coded per block — the same shape tantivy uses for sorted blocks (a vint
//! base plus bit-packed residuals; `postings/compression/mod.rs`
//! `uncompress_block_sorted` in tantivy @ 6182c6062) — with three extra rules
//! that remove bytes the old framing paid even when they carried no
//! information:
//!
//! ```text
//!   [width: u8]        0 = every lane equals `min`, no payload follows
//!                     1..=32 = bit width of (value − min) lanes
//!   [min: vbyte]       the frame's base, added back on decode
//!   [payload]          128 × width / 8 bytes — length DERIVED, never stored
//! ```
//!
//! An all-equal block (dense-field doc-id deltas of 1, a term whose freqs are
//! all k) costs 2–3 bytes total instead of the 21 the old framing paid.  The
//! old `u32 byte_len` was pure overhead: it is always `128 × width / 8`.
//!
//! The frame's `min` is subtracted when that narrows the width, and also on
//! narrow blocks (≤ [`ZPS2_FOR_MAX_EQUAL_WIDTH`] bits) where it normalises
//! level-shifted repeats into identical payloads for the outer zstd pass.
//! On WIDE equal-width blocks the subtraction only rewrites lanes zstd would
//! have matched verbatim, so those emit `min = 0` with raw lanes — see the
//! constant's doc comment for the measured story.
//!
//! ## Term-frequency elision
//!
//! A positioned term whose `total_term_frequency == doc_frequency` has, by
//! construction, `term_freq == 1` in every document (df positive ints
//! summing to df).  The writer omits the term's ENTIRE freq stream — full
//! blocks and residual vbytes alike — and the reader re-derives the same
//! predicate from the `.meta` record it already loaded (tantivy gates its
//! freq block on `term_has_freq`, `postings/serializer.rs::write_block`;
//! Lucene 50 elides per term when ttf == df).  Docs-only fields have always
//! done this via `store_positions = false`.
//!
//! Residual docs (< 128) at the end use variable-byte encoding.  In docs-only
//! mode (`store_positions = false`) the freq and position sub-blocks are
//! omitted and the reader synthesises `term_freq = 1`.
//!
//! The per-term `(doc_frequency, total_term_frequency, offset)` header lives in
//! the FST term dictionary / `.meta` file, **not** inline in the `.post` blob.
//!
//! ## Skip-list acceleration: NOT implemented
//!
//! There is **no on-disk skip table**.  [`PostingsWriter::encode_term`] computes
//! a [`SkipEntry`] vector in memory (one entry every [`SKIP_INTERVAL`] blocks)
//! but it is **never serialised** — the caller in `xerj-fts::index` discards it,
//! and the whole `.post` blob is wrapped in a Zstd envelope, so intra-blob byte
//! offsets would not be usable as seek targets anyway.  Consequently
//! [`PostingsReader::advance_to`] performs a **linear scan**, not a skip-list
//! seek.  Traversal is correct, just unaccelerated; the [`SkipEntry`] machinery
//! is scaffolding for a future block-skip implementation.

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::io::{self, Cursor};

/// Number of docs per PFOR block (must be a multiple of 128 for bitpacking).
pub const BLOCK_SIZE: usize = 128;

/// Cadence at which [`PostingsWriter::encode_term`] emits an in-memory
/// [`SkipEntry`].  NOTE: these entries are never persisted or consulted — see
/// the module docs' "Skip-list acceleration: NOT implemented" note.
pub const SKIP_INTERVAL: usize = 8; // every 1024 docs

// ── Term metadata ─────────────────────────────────────────────────────────────

/// Metadata stored in the term dictionary alongside the FST key.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TermPostings {
    /// Number of documents that contain this term (≡ Lucene's `docFreq`).
    pub doc_frequency: u32,
    /// Sum of term frequency across all documents (≡ Lucene's `totalTermFreq`).
    pub total_term_frequency: u64,
    /// Byte offset into the postings file where this term's data starts.
    pub postings_offset: u64,
    /// Byte length of the postings data for this term.
    pub postings_length: u32,
}

impl TermPostings {
    /// Serialise to a fixed-size 20-byte record.
    pub fn encode(&self) -> [u8; 20] {
        let mut buf = [0u8; 20];
        let mut c = Cursor::new(buf.as_mut_slice());
        c.write_u32::<LittleEndian>(self.doc_frequency).unwrap();
        c.write_u64::<LittleEndian>(self.total_term_frequency)
            .unwrap();
        c.write_u64::<LittleEndian>(self.postings_offset).unwrap();
        // last 4 bytes: length — split from offset to stay 20 B
        // rewrite cleanly:
        let mut out = [0u8; 20];
        out[0..4].copy_from_slice(&self.doc_frequency.to_le_bytes());
        out[4..12].copy_from_slice(&self.total_term_frequency.to_le_bytes());
        out[12..20].copy_from_slice(&self.postings_offset.to_le_bytes());
        out
    }

    /// Parse from the 20-byte record produced by [`encode`].
    /// `postings_length` and `postings_offset` are stored separately in the
    /// FST value (offset only fits in 64 bits); length is stored here.
    pub fn decode(buf: &[u8]) -> io::Result<Self> {
        if buf.len() < 20 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "buffer too short for TermPostings",
            ));
        }
        let mut c = Cursor::new(buf);
        let doc_frequency = c.read_u32::<LittleEndian>()?;
        let total_term_frequency = c.read_u64::<LittleEndian>()?;
        let postings_offset = c.read_u64::<LittleEndian>()?;
        Ok(Self {
            doc_frequency,
            total_term_frequency,
            postings_offset,
            postings_length: 0, // filled by caller from separate source
        })
    }
}

// ── Skip table entry ──────────────────────────────────────────────────────────

/// A single would-be skip pointer into a term's block stream.
///
/// **Currently unused scaffolding.**  [`PostingsWriter::encode_term`] builds a
/// `Vec<SkipEntry>` but the value is discarded by its only caller and is never
/// written to the `.post` file, so no reader ever consults it.  It exists to
/// mark the intended shape of a future block-skip index; see the module docs.
#[derive(Debug, Clone)]
pub struct SkipEntry {
    /// Highest doc ID in the preceding block range.
    pub max_doc_id: u32,
    /// Byte offset of the first byte of the corresponding block.
    pub byte_offset: u32,
}

// ── PostingsWriter ────────────────────────────────────────────────────────────

/// Accumulated posting data for one term during index construction.
///
/// `positions` is empty (capacity 0, no heap allocation) when the writer
/// is configured with `store_positions = false` — the common case for
/// `keyword`, `ip`, numeric, and status-like log fields.  Skipping
/// positions here turns each posting from `32 + 24 + Vec<u32>` bytes
/// down to just `32 + 24` bytes of stack, and eliminates the per-occurrence
/// `Vec::push` that dominates ingest time for high-cardinality
/// repeating-value fields (e.g. nginx `method` with 27 000 occurrences
/// of `"GET"` per segment, each currently allocating a 1-element position
/// list into a growable Vec).
#[derive(Default)]
struct RawPosting {
    doc_id: u32,
    term_freq: u32,
    positions: Vec<u32>,
}

/// Accumulates postings for all terms in one field/segment, then serialises
/// them into the compact block format.
pub struct PostingsWriter {
    /// Per-term posting data, sorted by doc_id within each term.
    postings: std::collections::BTreeMap<String, Vec<RawPosting>>,
    /// When `false`, `add_occurrence` discards positions and `encode_term`
    /// writes a zero-length positions block per 128-doc block.  Saves ~60 %
    /// of `.post` bytes on keyword fields.
    store_positions: bool,
}

impl PostingsWriter {
    pub fn new() -> Self {
        Self {
            postings: std::collections::BTreeMap::new(),
            store_positions: true,
        }
    }

    /// Build a writer that omits positions.  Use for `keyword`, numeric,
    /// and other exact-match fields — they never answer phrase queries.
    pub fn new_no_positions() -> Self {
        Self {
            postings: std::collections::BTreeMap::new(),
            store_positions: false,
        }
    }

    pub fn store_positions(&self) -> bool {
        self.store_positions
    }

    /// Record one occurrence of `term` in `doc_id` at the given `position`.
    ///
    /// When the writer was constructed without positions, `position` is
    /// ignored and no heap growth happens in the `positions` Vec.
    pub fn add_occurrence(&mut self, term: &str, doc_id: u32, position: u32) {
        // Lookup-first to avoid the `term.to_owned()` String allocation
        // that `entry()` forced on EVERY occurrence (13M+ allocs per 1M
        // docs at flush time).  Misses (first occurrence of a term) pay
        // one extra BTreeMap descent — postings are Zipf-shaped, so hits
        // dominate.
        if let Some(list) = self.postings.get_mut(term) {
            if let Some(last) = list.last_mut() {
                if last.doc_id == doc_id {
                    last.term_freq += 1;
                    if self.store_positions {
                        last.positions.push(position);
                    }
                    return;
                }
            }
            list.push(RawPosting {
                doc_id,
                term_freq: 1,
                positions: if self.store_positions {
                    vec![position]
                } else {
                    Vec::new()
                },
            });
            return;
        }
        self.postings.insert(
            term.to_owned(),
            vec![RawPosting {
                doc_id,
                term_freq: 1,
                positions: if self.store_positions {
                    vec![position]
                } else {
                    Vec::new()
                },
            }],
        );
    }

    /// Append a whole run of already-decoded postings for `term`.
    ///
    /// This is the merge path's ingest entry point.  `add_occurrence` is the
    /// INDEXING entry point: it takes one token at a time because that is
    /// what an analyzer produces.  A merge has no analyzer — it has one
    /// [`DecodedPosting`] per (term, document) straight out of a source
    /// segment's posting list — so replaying it occurrence by occurrence
    /// would pay a `BTreeMap<String, _>` descent per TOKEN.  Taking the run
    /// in one call pays one descent per (term, source segment) instead:
    /// measured at roughly half the replay's CPU on a 100 MiB corpus, and it
    /// also moves each posting's position vector in rather than copying it.
    ///
    /// Positions are dropped when this writer omits them.  `term_freq` is
    /// stored as given — the `.post` format records the frequency and the
    /// position count as separate fields, and a merge must reproduce both
    /// exactly as the source segment had them.
    ///
    /// Two obligations on the caller, both of which a merge satisfies by
    /// construction:
    ///
    /// * a document may appear at most ONCE per term across every run (each
    ///   merged ordinal comes from exactly one source document, and a posting
    ///   list holds one entry per document), because this appends without
    ///   coalescing equal doc ids;
    /// * runs may arrive out of doc order — a merged segment interleaves its
    ///   inputs' documents — so [`Self::sort_postings_by_doc`] MUST run before
    ///   `encode_term`.
    pub fn extend_postings(&mut self, term: &str, run: Vec<DecodedPosting>) {
        if run.is_empty() {
            return;
        }
        let store_positions = self.store_positions;
        let convert = move |posting: DecodedPosting| RawPosting {
            doc_id: posting.doc_id,
            term_freq: posting.term_freq,
            positions: if store_positions {
                posting.positions
            } else {
                Vec::new()
            },
        };
        match self.postings.get_mut(term) {
            Some(list) => {
                list.reserve(run.len());
                list.extend(run.into_iter().map(convert));
            }
            None => {
                self.postings
                    .insert(term.to_owned(), run.into_iter().map(convert).collect());
            }
        }
    }

    /// Restore the ascending-doc-id invariant every posting list must hold
    /// before it is encoded.
    ///
    /// The indexing path never needs this: it walks documents in ordinal
    /// order, so each list is built ascending.  The merge path replays
    /// several source segments whose surviving documents interleave in the
    /// merged ordinal space, so a shared term's list arrives as k ascending
    /// runs.  `encode_block_doc_ids` delta-encodes with a plain `-`, which
    /// would underflow on an out-of-order pair.
    ///
    /// `sort_by_key` is the stable, run-detecting sort, so k concatenated
    /// ascending runs cost O(n log k), not O(n log n); lists that are already
    /// ordered are detected by the scan below and never sorted at all.
    pub fn sort_postings_by_doc(&mut self) {
        for list in self.postings.values_mut() {
            if list.windows(2).any(|pair| pair[0].doc_id > pair[1].doc_id) {
                list.sort_by_key(|posting| posting.doc_id);
            }
        }
    }

    /// Direct `(doc_frequency, total_term_frequency)` lookup for ONE term.
    ///
    /// O(log T + postings(term)) — use this in per-term loops.  The segment
    /// writer (`write_field_static`) previously resolved stats via
    /// `term_stats().find(term)`, which walks the term map from the start
    /// *and sums every earlier term's postings* on each call — O(T² × P)
    /// across the write loop.  Invisible on 625-doc flush segments, but a
    /// merged segment over a float field (≈1 distinct term per doc) spun
    /// 100 % CPU for HOURS at 3 M docs, pinning rayon workers and starving
    /// every search + flush behind it (the read-under-write collapse).
    pub fn stats_for(&self, term: &str) -> Option<(u32, u64)> {
        self.postings.get(term).map(|postings| {
            let doc_freq = postings.len() as u32;
            let ttf: u64 = postings.iter().map(|p| p.term_freq as u64).sum();
            (doc_freq, ttf)
        })
    }

    /// Returns an iterator over `(term, doc_frequency, total_term_frequency)`
    /// without consuming self (for building the FST term dictionary).
    pub fn term_stats(&self) -> impl Iterator<Item = (&str, u32, u64)> {
        self.postings.iter().map(|(term, postings)| {
            let doc_freq = postings.len() as u32;
            let ttf: u64 = postings.iter().map(|p| p.term_freq as u64).sum();
            (term.as_str(), doc_freq, ttf)
        })
    }

    /// Encode the posting list for `term` into `output`.
    ///
    /// Returns `(postings_offset, skip_table)` where `postings_offset` is
    /// the byte position of the encoded data within `output` at the time of
    /// writing (caller tracks the global byte offset).
    ///
    /// NOTE: the returned `skip_table` is **not** written into `output` and is
    /// currently discarded by the caller — only the packed blocks and residual
    /// are serialised.  See the module docs' "Skip-list acceleration: NOT
    /// implemented" note.
    pub fn encode_term(&self, term: &str, output: &mut Vec<u8>) -> Option<(u64, Vec<SkipEntry>)> {
        self.encode_term_inner(term, output, None, BlockCodec::Packed)
            .map(|(offset, skips, _)| (offset, skips))
    }

    /// [`Self::encode_term`] plus the term's block-index entry (see
    /// [`BlockIndex`]): for a term with at least one full block, every
    /// block's last doc id, its byte offset inside the term's encoded bytes
    /// and its competitive `(tf, norm byte)` pairs, plus the term's own
    /// competitive pairs.  `norm_of(doc_id)` returns the document's stored
    /// norm byte for this field.  The encoded bytes are identical to
    /// [`Self::encode_term`]'s.
    pub fn encode_term_indexed(
        &self,
        term: &str,
        output: &mut Vec<u8>,
        norm_of: &dyn Fn(u32) -> u8,
    ) -> Option<(u64, Option<TermBlockData>)> {
        self.encode_term_indexed_with(term, output, norm_of, BlockCodec::Packed)
    }

    /// [`Self::encode_term_indexed`] with an explicit full-block codec
    /// ([`BlockCodec::EliasFanoAc`] is the experimental ZPS3 layout).
    pub fn encode_term_indexed_with(
        &self,
        term: &str,
        output: &mut Vec<u8>,
        norm_of: &dyn Fn(u32) -> u8,
        codec: BlockCodec,
    ) -> Option<(u64, Option<TermBlockData>)> {
        self.encode_term_inner(term, output, Some(norm_of), codec)
            .map(|(offset, _, blocks)| (offset, blocks))
    }

    fn encode_term_inner(
        &self,
        term: &str,
        output: &mut Vec<u8>,
        norm_of: Option<&dyn Fn(u32) -> u8>,
        codec: BlockCodec,
    ) -> Option<(u64, Vec<SkipEntry>, Option<TermBlockData>)> {
        let postings = self.postings.get(term)?;

        let start_offset = output.len() as u64;
        let mut skip_table: Vec<SkipEntry> = Vec::new();
        let mut block_count = 0usize;
        let mut i = 0usize;
        let n = postings.len();
        // The block index covers terms with at least one full block: a
        // residual-only list is one short decode, nothing to skip.
        let mut blocks = match norm_of {
            Some(_) if n >= BLOCK_SIZE => Some(TermBlockData::default()),
            _ => None,
        };
        let mut impacts = ImpactBuilder::default();
        // the frequency a reader will score with: docs-only fields store none
        // and synthesise 1
        let store_positions = self.store_positions;
        let scored_tf = |p: &RawPosting| if store_positions { p.term_freq } else { 1 };
        let record = |blocks: &mut Option<TermBlockData>,
                      impacts: &mut ImpactBuilder,
                      run: &[RawPosting],
                      offset: usize| {
            if let (Some(b), Some(norm_of)) = (blocks.as_mut(), norm_of) {
                b.last_doc.push(run[run.len() - 1].doc_id);
                b.offset.push(offset as u32);
                for p in run {
                    impacts.add(scored_tf(p), norm_of(p.doc_id));
                }
                impacts.drain_into(&mut b.impacts, MAX_BLOCK_IMPACTS);
                b.impact_end.push(b.impacts.len() as u32);
            }
        };
        // Term-frequency elision: df positive integers summing to df iff
        // every one of them is 1.  The reader re-derives this SAME predicate
        // from the `.meta` record (ttf == df), so the decision costs no
        // bytes on disk and the two sides cannot drift.
        let ttf: u64 = postings.iter().map(|p| p.term_freq as u64).sum();
        let freq_omitted = self.store_positions && ttf == n as u64;
        // Track the last doc_id written, so that delta[0] of the next block
        // is the gap from the previous block's final doc.  The reader
        // (`decode_next_full_block`) assumes that the first delta in each
        // block is relative to `last_doc_id`, not absolute.
        let mut prev_block_last_doc_id: u32 = 0;

        while i + BLOCK_SIZE <= n {
            // Full block
            let block = &postings[i..i + BLOCK_SIZE];
            let block_offset = output.len() - start_offset as usize;

            if block_count.is_multiple_of(SKIP_INTERVAL) {
                skip_table.push(SkipEntry {
                    max_doc_id: block[BLOCK_SIZE - 1].doc_id,
                    byte_offset: block_offset as u32,
                });
            }

            match codec {
                BlockCodec::Packed | BlockCodec::PackedNoCounts => {
                    encode_block_doc_ids(block, output, prev_block_last_doc_id)
                }
                BlockCodec::EliasFanoAc => {
                    let base = if block_count == 0 {
                        0
                    } else {
                        prev_block_last_doc_id + 1
                    };
                    encode_block_doc_ids_ef(block, output, base)
                }
            }
            if self.store_positions && !freq_omitted {
                match codec {
                    BlockCodec::Packed | BlockCodec::PackedNoCounts => {
                        encode_block_freqs(block, output)
                    }
                    BlockCodec::EliasFanoAc => encode_block_freqs_ac(block, output),
                }
            }
            if self.store_positions {
                encode_block_positions(block, output, codec != BlockCodec::PackedNoCounts);
            }
            // docs-only mode: emit nothing after the packed doc-id block.
            // The reader is told about the mode via the field meta and
            // skips the freq/position decode paths entirely.
            record(&mut blocks, &mut impacts, block, block_offset);

            prev_block_last_doc_id = block[BLOCK_SIZE - 1].doc_id;
            i += BLOCK_SIZE;
            block_count += 1;
        }

        // Residual (< BLOCK_SIZE docs) — delta-encoded from the last full
        // block's final doc_id as well.
        if i < n {
            let residual = &postings[i..];
            let block_offset = output.len() - start_offset as usize;
            encode_residual(
                residual,
                output,
                prev_block_last_doc_id,
                self.store_positions,
                freq_omitted,
                codec != BlockCodec::PackedNoCounts,
            );
            record(&mut blocks, &mut impacts, residual, block_offset);
        }

        if let (Some(b), Some(norm_of)) = (blocks.as_mut(), norm_of) {
            for p in postings {
                impacts.add(scored_tf(p), norm_of(p.doc_id));
            }
            impacts.drain_into(&mut b.term_impacts, MAX_TERM_IMPACTS);
        }
        Some((start_offset, skip_table, blocks))
    }

    /// Whether every posting holds exactly `term_freq` positions, the condition
    /// for [`BlockCodec::PackedNoCounts`]. A field that fails it (no indexing
    /// path makes one, but the format never forbade it) keeps its counts.
    pub fn positions_match_freqs(&self) -> bool {
        !self.store_positions
            || self
                .postings
                .values()
                .flatten()
                .all(|p| p.positions.len() as u64 == p.term_freq as u64)
    }

    pub fn terms(&self) -> impl Iterator<Item = &str> {
        self.postings.keys().map(|s| s.as_str())
    }

    pub fn is_empty(&self) -> bool {
        self.postings.is_empty()
    }
}

impl Default for PostingsWriter {
    fn default() -> Self {
        Self::new()
    }
}

// ── Block encoding helpers ────────────────────────────────────────────────────

/// Width at or below which a block always takes the frame-of-reference
/// form, even when subtracting `min` would not narrow the width.
///
/// The whole-file zstd envelope is the second compression stage, and the
/// two stages interact in opposite directions by block width (measured on
/// the 100k `benchmarks/index-size` harness, A/B against both always-FOR
/// and never-FOR-at-equal-width encoders):
///
/// * NARROW blocks (≤ 4 bits): the packed payload is a short, highly
///   regular bit pattern.  FOR normalises level-shifted repeats — freqs
///   5/6 and freqs 9/10 both pack to the same `[0,1,0,1,…]` lanes — so
///   equal payloads recur across blocks and zstd matches them exactly.
///   Skipping the subtraction at equal width cost `body.post` 16 KB.
/// * WIDE blocks: the lanes are quasi-random bits, so the only cross-block
///   repetition zstd can find is payload EQUALITY, which subtracting a
///   varying per-block `min` destroys.  FOR-at-equal-width inflated the
///   compressed `top_doc` field 12.7 % while its uncompressed stream
///   shrank — pure post-compression loss.
const ZPS2_FOR_MAX_EQUAL_WIDTH: u8 = 4;

/// Bit-pack a 128-element `u32` array into the output buffer in the ZPS2
/// framing: `[width: u8][min: vbyte][payload]` with the payload omitted
/// entirely when every lane equals `min` (width 0).  Frame-of-reference
/// subtracts `min` first, so the width tracks the block's SPREAD, not its
/// maximum — a block of freqs 100..=110 packs at 4 bits, not 7.
fn pack_u32_block_v2(values: &[u32; BLOCK_SIZE], output: &mut Vec<u8>) {
    use bitpacking::BitPacker;
    let min = *values.iter().min().unwrap_or(&0);
    let max = *values.iter().max().unwrap_or(&0);
    let spread = max - min;

    if spread == 0 {
        output.push(0u8);
        vbyte_encode(min, output);
        return;
    }

    let num_bits_framed = (32 - spread.leading_zeros()) as u8;
    let num_bits_raw = (32 - max.leading_zeros()) as u8; // 1..=32, max > 0 here
    let (min, num_bits) =
        if num_bits_framed < num_bits_raw || num_bits_raw <= ZPS2_FOR_MAX_EQUAL_WIDTH {
            (min, num_bits_framed)
        } else {
            (0, num_bits_raw)
        };
    output.push(num_bits);
    vbyte_encode(min, output);

    let mut lanes = [0u32; BLOCK_SIZE];
    for (lane, &value) in lanes.iter_mut().zip(values.iter()) {
        *lane = value - min;
    }
    // bitpacking::BitPacker4x::BLOCK_LEN == 128; byte_len is derived on read.
    let byte_len = BLOCK_SIZE * num_bits as usize / 8;
    let mut compressed = vec![0u8; byte_len];
    bitpacking::BitPacker4x::new().compress(&lanes, &mut compressed, num_bits);
    output.extend_from_slice(&compressed);
}

/// Decode a 128-element block previously written by `pack_u32_block_v2`.
/// Reads the cursor forward past the data and fills `out`.
fn unpack_u32_block_v2(
    data: &[u8],
    cursor: &mut Cursor<&[u8]>,
    out: &mut [u32; BLOCK_SIZE],
) -> io::Result<()> {
    use bitpacking::BitPacker;
    let num_bits = cursor.read_u8()?;
    let min = vbyte_decode(cursor)?;
    if num_bits == 0 {
        *out = [min; BLOCK_SIZE];
        return Ok(());
    }
    if num_bits > 32 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "postings block: invalid bit width",
        ));
    }
    let byte_len = BLOCK_SIZE * num_bits as usize / 8;
    let start = cursor.position() as usize;
    let end = start + byte_len;
    if end > data.len() {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "postings data truncated",
        ));
    }
    bitpacking::BitPacker4x::new().decompress(&data[start..end], out, num_bits);
    for value in out.iter_mut() {
        *value = value.wrapping_add(min);
    }
    cursor.set_position(end as u64);
    Ok(())
}

/// Decode a 128-element block in the legacy ZPS1 framing
/// `[num_bits ≥ 1][u32 byte_len][payload]`.  Old segments stay readable.
fn unpack_u32_block_v1(
    data: &[u8],
    cursor: &mut Cursor<&[u8]>,
    out: &mut [u32; BLOCK_SIZE],
) -> io::Result<()> {
    use bitpacking::BitPacker;
    let num_bits = cursor.read_u8()?;
    let byte_len = cursor.read_u32::<LittleEndian>()? as usize;
    let start = cursor.position() as usize;
    let end = start + byte_len;
    if end > data.len() {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "postings data truncated",
        ));
    }
    let compressed = &data[start..end];
    bitpacking::BitPacker4x::new().decompress(compressed, out, num_bits);
    cursor.set_position(end as u64);
    Ok(())
}

fn encode_block_doc_ids(block: &[RawPosting], output: &mut Vec<u8>, prev_last_doc_id: u32) {
    debug_assert_eq!(block.len(), BLOCK_SIZE);

    // Delta-encode doc IDs.  The first delta is relative to the previous
    // block's final doc_id (or 0 for the very first block) so the reader
    // can reconstruct absolute positions by running-sum across blocks.
    let mut deltas = [0u32; BLOCK_SIZE];
    deltas[0] = block[0].doc_id - prev_last_doc_id;
    for j in 1..BLOCK_SIZE {
        deltas[j] = block[j].doc_id - block[j - 1].doc_id;
    }

    pack_u32_block_v2(&deltas, output);
}

fn encode_block_freqs(block: &[RawPosting], output: &mut Vec<u8>) {
    debug_assert_eq!(block.len(), BLOCK_SIZE);

    let mut freqs = [0u32; BLOCK_SIZE];
    for (i, p) in block.iter().enumerate() {
        freqs[i] = p.term_freq;
    }

    pack_u32_block_v2(&freqs, output);
}

fn encode_block_positions(block: &[RawPosting], output: &mut Vec<u8>, counts: bool) {
    // Positions: variable-byte encode per-document, delta within document.
    // No length prefix — the reader decodes exactly BLOCK_SIZE documents'
    // (count + deltas) groups, which pins the stream's end; the u32 the old
    // framing spent per block was information the structure already had.
    for posting in block {
        if counts {
            vbyte_encode(posting.positions.len() as u32, output);
        }
        let mut prev = 0u32;
        for &pos in &posting.positions {
            vbyte_encode(pos - prev, output);
            prev = pos;
        }
    }
}

fn encode_residual(
    residual: &[RawPosting],
    output: &mut Vec<u8>,
    prev_last_doc_id: u32,
    store_positions: bool,
    freq_omitted: bool,
    counts: bool,
) {
    // Mark as residual with a sentinel: count byte
    output.push(residual.len() as u8);

    // The residual continues the delta chain from the last full block.
    let mut prev_doc = prev_last_doc_id;
    for posting in residual {
        let delta = posting.doc_id - prev_doc;
        prev_doc = posting.doc_id;
        vbyte_encode(delta, output);

        // docs-only mode: skip freq and positions.  Reader synthesises
        // term_freq = 1 when decoding this field.  `freq_omitted` is the
        // positioned variant of the same elision — a term with ttf == df
        // has term_freq 1 everywhere, so there is nothing to store.
        if store_positions && !freq_omitted {
            vbyte_encode(posting.term_freq, output);
        }
        if store_positions {
            if counts {
                vbyte_encode(posting.positions.len() as u32, output);
            }
            let mut prev_pos = 0u32;
            for &pos in &posting.positions {
                vbyte_encode(pos - prev_pos, output);
                prev_pos = pos;
            }
        }
    }
}

// ── Full-block codecs: packed (ZPS1/ZPS2) and Elias-Fano + arithmetic (ZPS3) ──

/// How a writer encodes the doc ids and frequencies of FULL 128-posting
/// blocks (residuals and positions are the same in both).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum BlockCodec {
    /// ZPS2: frame-of-reference bit packing (the format every reader knows).
    #[default]
    Packed,
    /// ZPS3, EXPERIMENTAL: per-block Elias-Fano doc ids and arithmetic-coded
    /// frequencies (the compression study's "exact compressed" codec).
    /// Readers that predate ZPS3 would misread such a segment, so writing it
    /// needs a data-directory format gate before any production use; it
    /// exists to measure the size / speed trade-off behind the same block
    /// index and walk.
    EliasFanoAc,
    /// ZPS4: ZPS2 without the vbyte position count of every posting, which always
    /// equals its term frequency. On the 1M corpus that count was 93 M bytes of
    /// the 418 MB of decompressed postings. Refused by older builds through the
    /// data-directory format gate.
    PackedNoCounts,
}

/// Doc ids of one full block (absolute), any codec. `prev_last` is the
/// previous block's last doc (0 before the first block).
#[inline]
fn decode_full_block_docs(
    codec: PostCodec,
    data: &[u8],
    cur: &mut Cursor<&[u8]>,
    prev_last: u32,
    first_block: bool,
    out: &mut [u32; BLOCK_SIZE],
) -> io::Result<()> {
    match codec {
        PostCodec::V1 | PostCodec::V2 { .. } | PostCodec::V4 { .. } => {
            let mut deltas = [0u32; BLOCK_SIZE];
            match codec {
                PostCodec::V1 => unpack_u32_block_v1(data, cur, &mut deltas)?,
                _ => unpack_u32_block_v2(data, cur, &mut deltas)?,
            }
            let mut prev = prev_last;
            for (d, &delta) in out.iter_mut().zip(deltas.iter()) {
                prev = prev.wrapping_add(delta);
                *d = prev;
            }
            Ok(())
        }
        PostCodec::V3 { .. } => {
            let base = if first_block {
                0
            } else {
                prev_last.wrapping_add(1)
            };
            decode_block_doc_ids_ef(data, cur, base, out)
        }
    }
}

/// Frequencies of one full block, any codec.
#[inline]
fn decode_full_block_freqs(
    codec: PostCodec,
    data: &[u8],
    cur: &mut Cursor<&[u8]>,
    out: &mut [u32; BLOCK_SIZE],
) -> io::Result<()> {
    match codec {
        PostCodec::V1 => unpack_u32_block_v1(data, cur, out),
        PostCodec::V2 { .. } | PostCodec::V4 { .. } => unpack_u32_block_v2(data, cur, out),
        PostCodec::V3 { .. } => decode_block_freqs_ac(data, cur, out),
    }
}

/// Step over one full block's frequency frame, any codec.
fn skip_full_block_freqs(codec: PostCodec, data: &[u8], cur: &mut Cursor<&[u8]>) -> io::Result<()> {
    match codec {
        PostCodec::V1 | PostCodec::V2 { .. } | PostCodec::V4 { .. } => {
            skip_u32_block(codec, data, cur)
        }
        PostCodec::V3 { .. } => {
            cur.read_u8()?;
            let len = vbyte_decode(cur)? as usize;
            let end = cur.position() as usize + len;
            if end > data.len() {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "postings data truncated",
                ));
            }
            cur.set_position(end as u64);
            Ok(())
        }
    }
}

/// Elias-Fano over one full block: values `doc - base` (non-decreasing),
/// universe `u = last + 1`, `l = floor(log2(u / 128))` low bits each, high
/// parts in unary.  Layout: `[l: u8][u: vbyte][low bits, then high bits,
/// LSB first]`.
fn encode_block_doc_ids_ef(block: &[RawPosting], output: &mut Vec<u8>, base: u32) {
    debug_assert_eq!(block.len(), BLOCK_SIZE);
    let n = BLOCK_SIZE;
    let rel: Vec<u64> = block.iter().map(|p| (p.doc_id - base) as u64).collect();
    let u = rel[n - 1] + 1;
    let l = if u <= n as u64 {
        0
    } else {
        63 - (u / n as u64).leading_zeros()
    };
    output.push(l as u8);
    vbyte_encode(u as u32, output);
    let high_bits = n + ((u - 1) >> l) as usize + 1;
    let total = n * l as usize + high_bits;
    let mut words = vec![0u64; total.div_ceil(64) + 1];
    let low_mask = if l == 0 { 0 } else { (1u64 << l) - 1 };
    for (j, &r) in rel.iter().enumerate() {
        if l > 0 {
            let bit = j * l as usize;
            let v = r & low_mask;
            words[bit / 64] |= v << (bit % 64);
            if bit % 64 + l as usize > 64 {
                words[bit / 64 + 1] |= v >> (64 - bit % 64);
            }
        }
        let p = n * l as usize + (r >> l) as usize + j;
        words[p / 64] |= 1u64 << (p % 64);
    }
    let nbytes = total.div_ceil(8);
    for i in 0..nbytes {
        output.push((words[i / 8] >> (8 * (i % 8))) as u8);
    }
}

fn decode_block_doc_ids_ef(
    data: &[u8],
    cur: &mut Cursor<&[u8]>,
    base: u32,
    out: &mut [u32; BLOCK_SIZE],
) -> io::Result<()> {
    let n = BLOCK_SIZE;
    let l = cur.read_u8()? as usize;
    let u = vbyte_decode(cur)? as u64;
    if l > 31 || u == 0 {
        return Err(invalid("Elias-Fano block header"));
    }
    let high_bits = n + ((u - 1) >> l) as usize + 1;
    let total = n * l + high_bits;
    let nbytes = total.div_ceil(8);
    let start = cur.position() as usize;
    let end = start + nbytes;
    if end > data.len() {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "postings data truncated",
        ));
    }
    // copy into aligned words (the block is at most 128 * 31 + 3 * 128 bits)
    let mut w = [0u64; 70];
    let nwords = nbytes.div_ceil(8);
    if nwords + 1 > w.len() {
        return Err(invalid("Elias-Fano block too large"));
    }
    for (i, chunk) in data[start..end].chunks(8).enumerate() {
        let mut b = [0u8; 8];
        b[..chunk.len()].copy_from_slice(chunk);
        w[i] = u64::from_le_bytes(b);
    }
    let low_mask = if l == 0 { 0 } else { (1u64 << l) - 1 };
    let mut p = n * l; // high part cursor (bits)
    for (j, slot) in out.iter_mut().enumerate() {
        let low = if l == 0 {
            0
        } else {
            let bit = j * l;
            let (i, o) = (bit / 64, bit % 64);
            let mut v = w[i] >> o;
            if o + l > 64 {
                v |= w[i + 1] << (64 - o);
            }
            v & low_mask
        };
        // next set bit at or after p
        loop {
            let i = p / 64;
            if i >= nwords {
                return Err(invalid("Elias-Fano high bits"));
            }
            let bits = w[i] >> (p % 64);
            if bits != 0 {
                p += bits.trailing_zeros() as usize;
                break;
            }
            p = (i + 1) * 64;
        }
        let high = (p - n * l - j) as u64;
        *slot = base.wrapping_add(((high << l) | low) as u32);
        p += 1;
    }
    cur.set_position(end as u64);
    Ok(())
}

/// LZMA-style adaptive binary range coder (32-bit range, 64-bit low with
/// carry, 15-bit probabilities), ported from the compression study
/// (bench_compress/xcomp xpost.rs, C1.10c).
struct RcEnc {
    low: u64,
    range: u32,
    cache: u8,
    cache_size: u64,
    out: Vec<u8>,
    started: bool,
}

impl RcEnc {
    fn new() -> Self {
        RcEnc {
            low: 0,
            range: 0xFFFF_FFFF,
            cache: 0,
            cache_size: 1,
            out: Vec::new(),
            started: false,
        }
    }
    fn shift_low(&mut self) {
        if (self.low as u32) < 0xFF00_0000 || (self.low >> 32) != 0 {
            let carry = (self.low >> 32) as u8;
            let mut temp = self.cache;
            loop {
                let byte = temp.wrapping_add(carry);
                if self.started {
                    self.out.push(byte);
                } else {
                    // the first byte of the stream is always 0: not stored
                    self.started = true;
                }
                temp = 0xFF;
                self.cache_size -= 1;
                if self.cache_size == 0 {
                    break;
                }
            }
            self.cache = ((self.low >> 24) & 0xFF) as u8;
        }
        self.cache_size += 1;
        self.low = (self.low & 0x00FF_FFFF) << 8;
    }
    #[inline]
    fn encode(&mut self, p0: u32, bit: u32) {
        let bound = (self.range >> 15) * p0;
        if bit == 0 {
            self.range = bound;
        } else {
            self.low += bound as u64;
            self.range -= bound;
        }
        while self.range < (1 << 24) {
            self.range <<= 8;
            self.shift_low();
        }
    }
    fn finish(mut self) -> Vec<u8> {
        for k in (1..=4).rev() {
            let m = 1u64 << (8 * k);
            let v = (self.low + m - 1) & !(m - 1);
            if v < self.low + self.range as u64 {
                self.low = v;
                break;
            }
        }
        for _ in 0..5 {
            self.shift_low();
        }
        while self.out.last() == Some(&0) {
            self.out.pop();
        }
        self.out
    }
}

struct RcDec<'a> {
    buf: &'a [u8],
    pos: usize,
    range: u32,
    code: u32,
}

impl<'a> RcDec<'a> {
    fn new(buf: &'a [u8]) -> Self {
        let mut d = RcDec {
            buf,
            pos: 0,
            range: 0xFFFF_FFFF,
            code: 0,
        };
        for _ in 0..4 {
            d.code = (d.code << 8) | d.next() as u32;
        }
        d
    }
    #[inline]
    fn next(&mut self) -> u8 {
        let b = self.buf.get(self.pos).copied().unwrap_or(0);
        self.pos += 1;
        b
    }
    #[inline]
    fn decode(&mut self, p0: u32) -> u32 {
        let bound = (self.range >> 15) * p0;
        let bit = if self.code < bound {
            self.range = bound;
            0
        } else {
            self.code -= bound;
            self.range -= bound;
            1
        };
        while self.range < (1 << 24) {
            self.range <<= 8;
            self.code = (self.code << 8) | self.next() as u32;
        }
        bit
    }
}

/// Adaptive bit model: Krichevsky-Trofimov estimate, counts halved past 65535.
#[derive(Clone, Copy)]
struct Ctx {
    c0: u32,
    c1: u32,
}

impl Ctx {
    #[inline]
    fn p0(&self) -> u32 {
        let p = (((2 * self.c0 as u64 + 1) << 15) / (2 * (self.c0 + self.c1) as u64 + 2)) as u32;
        p.clamp(1, 32767)
    }
    #[inline]
    fn update(&mut self, bit: u32) {
        if bit == 0 {
            self.c0 += 1;
        } else {
            self.c1 += 1;
        }
        if self.c0 + self.c1 > 65535 {
            self.c0 = self.c0.div_ceil(2);
            self.c1 = self.c1.div_ceil(2);
        }
    }
    /// prior of the "tf > 1" flag from the block's exception rate `q / 255`
    fn prior(q: u32) -> Ctx {
        Ctx {
            c0: ((255 - q) * 16 + 127) / 255,
            c1: (q * 16 + 127) / 255,
        }
    }
}

/// Frequencies of one full block: `[q: u8][len: vbyte][stream]`, `q` = the
/// block's share of tf > 1 in 1/255 steps (the flag's prior), the stream =
/// per posting an adaptive "tf > 1" flag and, when set, `tf - 1` as
/// exp-Golomb (adaptive unary length, raw mantissa).
fn encode_block_freqs_ac(block: &[RawPosting], output: &mut Vec<u8>) {
    let exc = block.iter().filter(|p| p.term_freq > 1).count();
    let q = ((exc as f64 / block.len() as f64) * 255.0).round() as u32;
    let mut rc = RcEnc::new();
    let mut flag = Ctx::prior(q);
    let mut lens = [Ctx { c0: 0, c1: 0 }; 33];
    for p in block {
        let bit = (p.term_freq > 1) as u32;
        rc.encode(flag.p0(), bit);
        flag.update(bit);
        if bit == 1 {
            let w = p.term_freq - 1; // >= 1
            let k = (32 - w.leading_zeros()) - 1;
            for ctx in lens.iter_mut().take(k as usize) {
                rc.encode(ctx.p0(), 1);
                ctx.update(1);
            }
            rc.encode(lens[k as usize].p0(), 0);
            lens[k as usize].update(0);
            for i in (0..k).rev() {
                rc.encode(16384, (w >> i) & 1);
            }
        }
    }
    let stream = rc.finish();
    output.push(q as u8);
    vbyte_encode(stream.len() as u32, output);
    output.extend_from_slice(&stream);
}

fn decode_block_freqs_ac(
    data: &[u8],
    cur: &mut Cursor<&[u8]>,
    out: &mut [u32; BLOCK_SIZE],
) -> io::Result<()> {
    let q = cur.read_u8()? as u32;
    let len = vbyte_decode(cur)? as usize;
    let start = cur.position() as usize;
    let end = start + len;
    if end > data.len() {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "postings data truncated",
        ));
    }
    let mut rc = RcDec::new(&data[start..end]);
    let mut flag = Ctx::prior(q);
    let mut lens = [Ctx { c0: 0, c1: 0 }; 33];
    for slot in out.iter_mut() {
        let bit = rc.decode(flag.p0());
        flag.update(bit);
        if bit == 0 {
            *slot = 1;
        } else {
            let mut k = 0usize;
            loop {
                let b = rc.decode(lens[k].p0());
                lens[k].update(b);
                if b == 0 {
                    break;
                }
                k += 1;
                if k >= 32 {
                    return Err(invalid("arithmetic-coded frequency"));
                }
            }
            let mut w = 1u32;
            for _ in 0..k {
                w = (w << 1) | rc.decode(16384);
            }
            *slot = w + 1;
        }
    }
    cur.set_position(end as u64);
    Ok(())
}

// ── Variable-byte encoding ────────────────────────────────────────────────────

/// Encodes `value` as a variable-byte integer into `buf`.
pub fn vbyte_encode(mut value: u32, buf: &mut Vec<u8>) {
    loop {
        let byte = (value & 0x7F) as u8;
        value >>= 7;
        if value == 0 {
            buf.push(byte | 0x80); // high bit = last byte
            break;
        } else {
            buf.push(byte);
        }
    }
}

/// Decodes a variable-byte integer from `cursor`.
pub fn vbyte_decode(cursor: &mut Cursor<&[u8]>) -> io::Result<u32> {
    let mut result = 0u32;
    let mut shift = 0u32;
    loop {
        let byte = cursor.read_u8()?;
        result |= ((byte & 0x7F) as u32) << shift;
        if byte & 0x80 != 0 {
            return Ok(result);
        }
        shift += 7;
        if shift >= 35 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "vbyte overflow"));
        }
    }
}

/// Encodes `value` as a variable-byte `u64` into `buf`.
///
/// Same convention as [`vbyte_encode`] (high bit set on the final byte) so
/// the pair is self-consistent; this is the wider lane the `.meta` ZFM5
/// record streams need, where `total_term_frequency` and `postings_offset`
/// exceed the `u32` vbyte range on multi-GB `.post` files.
pub fn uvarint_encode(mut value: u64, buf: &mut Vec<u8>) {
    loop {
        let byte = (value & 0x7F) as u8;
        value >>= 7;
        if value == 0 {
            buf.push(byte | 0x80); // high bit = last byte
            break;
        } else {
            buf.push(byte);
        }
    }
}

/// Decodes a variable-byte `u64` from `cursor`.
pub fn uvarint_decode(cursor: &mut Cursor<&[u8]>) -> io::Result<u64> {
    let mut result = 0u64;
    let mut shift = 0u32;
    loop {
        let byte = cursor.read_u8()?;
        result |= ((byte & 0x7F) as u64) << shift;
        if byte & 0x80 != 0 {
            return Ok(result);
        }
        shift += 7;
        if shift >= 70 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "uvarint overflow",
            ));
        }
    }
}

/// Decode one document's position group — a vbyte count followed by that
/// many delta-encoded positions.
fn decode_position_group(cursor: &mut Cursor<&[u8]>) -> io::Result<Vec<u32>> {
    #[cfg(test)]
    POSITION_GROUPS_MATERIALISED.with(|c| c.set(c.get() + 1));
    let count = vbyte_decode(cursor)? as usize;
    let mut positions = Vec::with_capacity(count);
    let mut prev = 0u32;
    for _ in 0..count {
        let delta = vbyte_decode(cursor)?;
        prev += delta;
        positions.push(prev);
    }
    Ok(positions)
}

/// [`decode_position_group`] into a caller-owned buffer (cleared first), so
/// a cursor reading the positions of many documents allocates once.
fn decode_position_group_into(cursor: &mut Cursor<&[u8]>, out: &mut Vec<u32>) -> io::Result<()> {
    #[cfg(test)]
    POSITION_GROUPS_MATERIALISED.with(|c| c.set(c.get() + 1));
    out.clear();
    let count = vbyte_decode(cursor)? as usize;
    out.reserve(count);
    let mut prev = 0u32;
    for _ in 0..count {
        let delta = vbyte_decode(cursor)?;
        prev += delta;
        out.push(prev);
    }
    Ok(())
}

/// Step over one document's position group — the same vbyte count and
/// deltas [`decode_position_group`] reads — without materialising it.
///
/// A vbyte ends at the byte whose high bit is set ([`vbyte_encode`]), so the
/// deltas are skipped by counting terminator bytes. A delta longer than the
/// five bytes a `u32` needs is the same `vbyte overflow` error the decoder
/// reports.
fn skip_position_group(cursor: &mut Cursor<&[u8]>) -> io::Result<()> {
    let count = vbyte_decode(cursor)? as usize;
    skip_position_deltas(cursor, count)
}

fn skip_position_deltas(cursor: &mut Cursor<&[u8]>, count: usize) -> io::Result<()> {
    let data: &[u8] = cursor.get_ref();
    let mut pos = cursor.position() as usize;
    let mut remaining = count;
    let mut run = 0u32;
    while remaining > 0 {
        let Some(&byte) = data.get(pos) else {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "postings positions truncated",
            ));
        };
        pos += 1;
        if byte & 0x80 != 0 {
            remaining -= 1;
            run = 0;
        } else {
            run += 1;
            if run >= 5 {
                return Err(io::Error::new(io::ErrorKind::InvalidData, "vbyte overflow"));
            }
        }
    }
    cursor.set_position(pos as u64);
    Ok(())
}

/// Step over a document's positions: its group, or for
/// [`PostCodec::V4`] exactly `tf` deltas.
fn skip_positions_of(cursor: &mut Cursor<&[u8]>, codec: PostCodec, tf: u32) -> io::Result<()> {
    if matches!(codec, PostCodec::V4 { .. }) {
        skip_position_deltas(cursor, tf as usize)
    } else {
        skip_position_group(cursor)
    }
}

/// Decode a document's positions, see [`skip_positions_of`].
fn decode_positions_of_into(
    cursor: &mut Cursor<&[u8]>,
    codec: PostCodec,
    tf: u32,
    out: &mut Vec<u32>,
) -> io::Result<()> {
    if !matches!(codec, PostCodec::V4 { .. }) {
        return decode_position_group_into(cursor, out);
    }
    #[cfg(test)]
    POSITION_GROUPS_MATERIALISED.with(|c| c.set(c.get() + 1));
    out.clear();
    out.reserve(tf as usize);
    let mut prev = 0u32;
    for _ in 0..tf {
        prev += vbyte_decode(cursor)?;
        out.push(prev);
    }
    Ok(())
}

#[cfg(test)]
thread_local! {
    /// How many per-document position groups were decoded into a `Vec`.
    /// A scorer that reads only `(doc_id, term_freq)` must leave this at 0.
    pub(crate) static POSITION_GROUPS_MATERIALISED: std::cell::Cell<usize> =
        const { std::cell::Cell::new(0) };
}

// ── Block index: skip offsets + competitive impacts (ZBM1) ───────────────────
//
// A field's `.post` envelope is `"ZPS2" | u32 uncompressed_len | zstd frame`.
// Writers that know the block index put it in a zstd SKIPPABLE frame between
// the length and the postings frame.  zstd decoders skip such frames by
// specification, so every reader that hands the bytes after the header to
// `zstd::bulk::decompress` (all readers before this one) still gets exactly
// the postings, and `uncompressed_len` keeps describing the postings frame
// alone.  Readers that know the frame parse it once at open.
//
// Per term with at least one full block, the index stores every block's
// last doc id and byte offset inside the term's encoded bytes — a block is
// then decodable on its own (doc-id deltas are relative to the previous
// block's last doc, and the doc and freq frames come first, so positions
// are never touched) — and its competitive `(tf, norm byte)` pairs: the
// pairs no other posting of the block dominates (tf' >= tf with norm'
// <= norm).  BM25's `tf_norm` grows with tf and falls with the document
// length, and the length is a non-decreasing function of the norm byte, so
// the block's best `tf_norm` under ANY k1, b and avgdl is the best over its
// pairs: an exact block upper bound that survives the per-request,
// index-wide statistics (#188).  Lucene keeps per-block "competitive
// impacts" for the same reason (re-derived here, no code taken).

/// zstd skippable-frame magic (`0x184D2A50..=0x184D2A5F`) of the block-index
/// frame inside a ZPS2 `.post` envelope.
pub const BLOCK_INDEX_FRAME_MAGIC: u32 = 0x184D_2A5B;
const ZBM1_MAGIC: &[u8; 4] = b"ZBM1";
const ZBM1_VERSION: u8 = 1;
/// Competitive pairs kept per block; a longer frontier is folded (see
/// [`ImpactBuilder::drain_into`]), which keeps it an upper bound.
pub const MAX_BLOCK_IMPACTS: usize = 4;
/// Competitive pairs kept per term.
pub const MAX_TERM_IMPACTS: usize = 8;

/// One competitive `(term frequency, norm byte)` pair.  `tf == 255` means
/// "255 or more": a bound computed from it must use the `tf -> inf` limit.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Impact {
    pub tf: u8,
    pub norm: u8,
}

impl Impact {
    pub const TF_SATURATED: u8 = 255;
}

/// Accumulates `(tf, norm)` observations and emits their competitive
/// frontier: ascending norm, strictly ascending tf.
#[derive(Default)]
struct ImpactBuilder {
    /// max tf seen per norm byte (0 = none)
    best: Vec<u32>,
    touched: Vec<u8>,
}

impl ImpactBuilder {
    #[inline]
    fn add(&mut self, tf: u32, norm: u8) {
        if self.best.is_empty() {
            self.best.resize(256, 0);
        }
        let slot = &mut self.best[norm as usize];
        if *slot == 0 {
            self.touched.push(norm);
        }
        *slot = (*slot).max(tf.max(1));
    }

    /// Append the frontier of everything added since the last drain to
    /// `out`, at most `cap` pairs, and reset.  Folding merges the two
    /// neighbours with the smallest tf gap into (the larger tf, the smaller
    /// norm), a pair that dominates both, so the result still bounds every
    /// posting.
    fn drain_into(&mut self, out: &mut Vec<Impact>, cap: usize) {
        self.touched.sort_unstable();
        let start = out.len();
        let mut run = 0u8;
        for &norm in &self.touched {
            let code = self.best[norm as usize].min(Impact::TF_SATURATED as u32) as u8;
            if code > run {
                out.push(Impact { tf: code, norm });
                run = code;
            }
            self.best[norm as usize] = 0;
        }
        self.touched.clear();
        while out.len() - start > cap.max(1) {
            let frontier = &out[start..];
            let i = (0..frontier.len() - 1)
                .min_by_key(|&i| frontier[i + 1].tf - frontier[i].tf)
                .unwrap_or(0);
            let merged = Impact {
                tf: frontier[i + 1].tf,
                norm: frontier[i].norm,
            };
            out[start + i] = merged;
            out.remove(start + i + 1);
        }
    }
}

/// One term's block-index entry as the writer produces it.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct TermBlockData {
    /// last doc id of every block (full blocks, then the residual)
    pub last_doc: Vec<u32>,
    /// byte offset of every block inside the term's encoded bytes
    pub offset: Vec<u32>,
    /// `impacts[impact_end[b - 1]..impact_end[b]]` are block `b`'s pairs
    pub impact_end: Vec<u32>,
    pub impacts: Vec<Impact>,
    /// the term's own competitive pairs (over every posting)
    pub term_impacts: Vec<Impact>,
}

/// Collects one field's [`TermBlockData`] and serialises the block-index
/// frame.
#[derive(Default)]
pub struct BlockIndexBuilder {
    keys: Vec<u64>,
    first_block: Vec<u32>,
    term_imp_end: Vec<u32>,
    last_doc: Vec<u32>,
    offset: Vec<u32>,
    blk_imp_end: Vec<u32>,
    term_imps: Vec<Impact>,
    blk_imps: Vec<Impact>,
}

impl BlockIndexBuilder {
    /// Add a term; `postings_offset` (the term's `TermPostings::postings_offset`)
    /// must increase from call to call.
    pub fn add(&mut self, postings_offset: u64, t: &TermBlockData) {
        debug_assert!(self.keys.last().is_none_or(|&k| k < postings_offset));
        debug_assert_eq!(t.last_doc.len(), t.offset.len());
        debug_assert_eq!(t.last_doc.len(), t.impact_end.len());
        self.keys.push(postings_offset);
        self.first_block.push(self.last_doc.len() as u32);
        self.term_imps.extend_from_slice(&t.term_impacts);
        self.term_imp_end.push(self.term_imps.len() as u32);
        let base = self.blk_imps.len() as u32;
        self.last_doc.extend_from_slice(&t.last_doc);
        self.offset.extend_from_slice(&t.offset);
        self.blk_imp_end
            .extend(t.impact_end.iter().map(|&e| base + e));
        self.blk_imps.extend_from_slice(&t.impacts);
    }

    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    /// The complete skippable frame (magic, size, payload); empty when no
    /// term was added, so a field without long lists keeps the old bytes.
    pub fn finish(self, zstd_level: i32) -> io::Result<Vec<u8>> {
        if self.keys.is_empty() {
            return Ok(Vec::new());
        }
        let mut raw: Vec<u8> = Vec::with_capacity(
            16 + self.keys.len() * 16
                + self.last_doc.len() * 12
                + (self.term_imps.len() + self.blk_imps.len()) * 2,
        );
        for n in [
            self.keys.len(),
            self.last_doc.len(),
            self.term_imps.len(),
            self.blk_imps.len(),
        ] {
            raw.extend_from_slice(&(n as u32).to_le_bytes());
        }
        for k in &self.keys {
            raw.extend_from_slice(&k.to_le_bytes());
        }
        for col in [
            &self.first_block,
            &self.term_imp_end,
            &self.last_doc,
            &self.offset,
            &self.blk_imp_end,
        ] {
            for v in col.iter() {
                raw.extend_from_slice(&v.to_le_bytes());
            }
        }
        for imp in self.term_imps.iter().chain(&self.blk_imps) {
            raw.push(imp.tf);
            raw.push(imp.norm);
        }
        let compressed = zstd::bulk::compress(&raw, zstd_level)?;
        let payload_len = 4 + 4 + 4 + compressed.len();
        let mut out = Vec::with_capacity(8 + payload_len);
        out.extend_from_slice(&BLOCK_INDEX_FRAME_MAGIC.to_le_bytes());
        out.extend_from_slice(&(payload_len as u32).to_le_bytes());
        out.extend_from_slice(ZBM1_MAGIC);
        out.extend_from_slice(&[ZBM1_VERSION, BLOCK_SIZE as u8, 0, 0]);
        out.extend_from_slice(&(raw.len() as u32).to_le_bytes());
        out.extend_from_slice(&compressed);
        Ok(out)
    }
}

/// If `body` (a ZPS2 envelope after its 8-byte header) starts with the
/// block-index frame, return `(frame payload, rest)`.
pub fn split_block_index_frame(body: &[u8]) -> Option<(&[u8], &[u8])> {
    if body.len() < 8 || u32::from_le_bytes(body[..4].try_into().ok()?) != BLOCK_INDEX_FRAME_MAGIC {
        return None;
    }
    let size = u32::from_le_bytes(body[4..8].try_into().ok()?) as usize;
    let end = 8usize.checked_add(size)?;
    (end <= body.len()).then(|| (&body[8..end], &body[end..]))
}

/// One field's parsed block index (see the section notes above).
#[derive(Debug, Default)]
pub struct BlockIndex {
    keys: Vec<u64>,
    first_block: Vec<u32>,
    term_imp_end: Vec<u32>,
    last_doc: Vec<u32>,
    offset: Vec<u32>,
    blk_imp_end: Vec<u32>,
    term_imps: Vec<Impact>,
    blk_imps: Vec<Impact>,
}

fn invalid(msg: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, format!("block index: {msg}"))
}

impl BlockIndex {
    /// Parse a frame payload produced by [`BlockIndexBuilder::finish`]
    /// (without the 8-byte skippable-frame header).  Every structural
    /// invariant the cursor relies on is checked here.
    pub fn parse(payload: &[u8]) -> io::Result<Self> {
        if payload.len() < 12 || &payload[..4] != ZBM1_MAGIC {
            return Err(invalid("bad magic"));
        }
        if payload[4] != ZBM1_VERSION || payload[5] as usize != BLOCK_SIZE {
            return Err(invalid("unsupported version or block size"));
        }
        let raw_len = u32::from_le_bytes(payload[8..12].try_into().unwrap()) as usize;
        let raw = zstd::bulk::decompress(&payload[12..], raw_len)?;
        if raw.len() != raw_len || raw.len() < 16 {
            return Err(invalid("length mismatch"));
        }
        let rd = |i: usize| u32::from_le_bytes(raw[4 * i..4 * i + 4].try_into().unwrap()) as usize;
        let (n_terms, n_blocks, n_timps, n_bimps) = (rd(0), rd(1), rd(2), rd(3));
        let expect = 16usize
            .checked_add(
                n_terms
                    .checked_mul(8 + 4 + 4)
                    .ok_or_else(|| invalid("overflow"))?,
            )
            .and_then(|v| v.checked_add(n_blocks.checked_mul(12)?))
            .and_then(|v| v.checked_add((n_timps.checked_add(n_bimps)?).checked_mul(2)?))
            .ok_or_else(|| invalid("overflow"))?;
        if expect != raw.len() {
            return Err(invalid("section sizes do not add up"));
        }
        let mut pos = 16usize;
        let take_u64 = |n: usize, pos: &mut usize| -> Vec<u64> {
            let v = raw[*pos..*pos + 8 * n]
                .chunks_exact(8)
                .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
                .collect();
            *pos += 8 * n;
            v
        };
        let keys = take_u64(n_terms, &mut pos);
        let take_u32 = |n: usize, pos: &mut usize| -> Vec<u32> {
            let v = raw[*pos..*pos + 4 * n]
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            *pos += 4 * n;
            v
        };
        let first_block = take_u32(n_terms, &mut pos);
        let term_imp_end = take_u32(n_terms, &mut pos);
        let last_doc = take_u32(n_blocks, &mut pos);
        let offset = take_u32(n_blocks, &mut pos);
        let blk_imp_end = take_u32(n_blocks, &mut pos);
        let imps = |n: usize, pos: &mut usize| -> Vec<Impact> {
            let v = raw[*pos..*pos + 2 * n]
                .chunks_exact(2)
                .map(|c| Impact {
                    tf: c[0],
                    norm: c[1],
                })
                .collect();
            *pos += 2 * n;
            v
        };
        let term_imps = imps(n_timps, &mut pos);
        let blk_imps = imps(n_bimps, &mut pos);
        let ix = BlockIndex {
            keys,
            first_block,
            term_imp_end,
            last_doc,
            offset,
            blk_imp_end,
            term_imps,
            blk_imps,
        };
        ix.validate()?;
        Ok(ix)
    }

    fn validate(&self) -> io::Result<()> {
        let ascending = |v: &[u32], max: usize| {
            v.windows(2).all(|w| w[0] <= w[1]) && v.last().is_none_or(|&x| x as usize <= max)
        };
        if !self.keys.windows(2).all(|w| w[0] < w[1]) {
            return Err(invalid("keys not strictly ascending"));
        }
        if !ascending(&self.first_block, self.last_doc.len())
            || self.first_block.first().is_some_and(|&f| f != 0)
        {
            return Err(invalid("block ranges"));
        }
        // every term and every block owns at least one pair: an empty range
        // would read as a zero bound and prune real matches
        let strictly = |v: &[u32], max: usize| {
            v.first().is_none_or(|&x| x > 0)
                && v.windows(2).all(|w| w[0] < w[1])
                && v.last().is_none_or(|&x| x as usize <= max)
        };
        if !strictly(&self.term_imp_end, self.term_imps.len())
            || !strictly(&self.blk_imp_end, self.blk_imps.len())
        {
            return Err(invalid("impact ranges"));
        }
        for t in 0..self.keys.len() {
            let (b0, b1) = self.block_range(t);
            if b1 <= b0 {
                return Err(invalid("term without blocks"));
            }
            if !self.last_doc[b0..b1].windows(2).all(|w| w[0] < w[1])
                || !self.offset[b0..b1].windows(2).all(|w| w[0] < w[1])
            {
                return Err(invalid("block docs / offsets not ascending"));
            }
        }
        Ok(())
    }

    #[inline]
    fn block_range(&self, t: usize) -> (usize, usize) {
        let b0 = self.first_block[t] as usize;
        let b1 = self
            .first_block
            .get(t + 1)
            .map_or(self.last_doc.len(), |&b| b as usize);
        (b0, b1)
    }

    /// The entry of the term whose postings start at `postings_offset`.
    pub fn term(&self, postings_offset: u64) -> Option<TermBlocks<'_>> {
        let t = self.keys.binary_search(&postings_offset).ok()?;
        let (b0, b1) = self.block_range(t);
        let ti0 = if t == 0 {
            0
        } else {
            self.term_imp_end[t - 1] as usize
        };
        let ti1 = self.term_imp_end[t] as usize;
        let imp_base = if b0 == 0 { 0 } else { self.blk_imp_end[b0 - 1] };
        Some(TermBlocks {
            last_doc: &self.last_doc[b0..b1],
            offset: &self.offset[b0..b1],
            imp_end: &self.blk_imp_end[b0..b1],
            imp_base,
            impacts: &self.blk_imps,
            term_impacts: &self.term_imps[ti0..ti1],
        })
    }

    pub fn terms(&self) -> usize {
        self.keys.len()
    }

    pub fn blocks(&self) -> usize {
        self.last_doc.len()
    }

    /// Heap bytes held by the parsed index (segment-cache accounting).
    pub fn heap_bytes(&self) -> u64 {
        (self.keys.len() * 8
            + (self.first_block.len()
                + self.term_imp_end.len()
                + self.last_doc.len()
                + self.offset.len()
                + self.blk_imp_end.len())
                * 4
            + (self.term_imps.len() + self.blk_imps.len()) * 2) as u64
    }
}

/// A term's view into its field's [`BlockIndex`].
#[derive(Clone, Copy, Debug)]
pub struct TermBlocks<'a> {
    /// last doc id per block (strictly ascending)
    pub last_doc: &'a [u32],
    /// byte offset per block inside the term's encoded bytes
    pub offset: &'a [u32],
    imp_end: &'a [u32],
    imp_base: u32,
    impacts: &'a [Impact],
    /// competitive pairs over the whole list
    pub term_impacts: &'a [Impact],
}

impl<'a> TermBlocks<'a> {
    pub fn len(&self) -> usize {
        self.last_doc.len()
    }

    pub fn is_empty(&self) -> bool {
        self.last_doc.is_empty()
    }

    /// Competitive pairs of block `b`.
    #[inline]
    pub fn block_impacts(&self, b: usize) -> &'a [Impact] {
        let start = if b == 0 {
            self.imp_base
        } else {
            self.imp_end[b - 1]
        } as usize;
        &self.impacts[start..self.imp_end[b] as usize]
    }
}

// ── BlockCursor: positions-free, block-skipping posting cursor ─────────────

/// Step over one packed 128-value frame (ZPS1 / ZPS2 framing) without unpacking it.
fn skip_u32_block(codec: PostCodec, data: &[u8], cur: &mut Cursor<&[u8]>) -> io::Result<()> {
    let num_bits = cur.read_u8()?;
    let len = match codec {
        PostCodec::V1 => cur.read_u32::<LittleEndian>()? as usize,
        PostCodec::V2 { .. } | PostCodec::V3 { .. } | PostCodec::V4 { .. } => {
            vbyte_decode(cur)?;
            if num_bits > 32 {
                return Err(invalid("invalid bit width"));
            }
            BLOCK_SIZE * num_bits as usize / 8
        }
    };
    let end = cur.position() as usize + len;
    if end > data.len() {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "postings data truncated",
        ));
    }
    cur.set_position(end as u64);
    Ok(())
}

/// First index `i >= from` with `v[i] >= target` (exponential search, then
/// binary search): cheap for the short forward hops cursors make.
#[inline]
pub(crate) fn gallop(v: &[u32], from: usize, target: u32) -> usize {
    let n = v.len();
    if from >= n || v[from] >= target {
        return from.min(n);
    }
    let mut lo = from;
    let mut step = 1usize;
    let mut hi = from + 1;
    while hi < n && v[hi] < target {
        lo = hi;
        step *= 2;
        hi = from + step;
    }
    let hi = hi.min(n);
    lo + 1 + v[lo + 1..hi].partition_point(|&d| d < target)
}

const NO_BLOCK: usize = usize::MAX;

/// Cursor over one term's postings that yields `(doc_id, term_freq)` and
/// never decodes positions.
///
/// With a [`TermBlocks`] entry every block is reached through its stored
/// offset, so [`Self::advance`] jumps over whole blocks without reading
/// them, and the position stream is not even scanned.  Without one (a
/// segment written before the block index, or a residual-only list) blocks
/// are decoded in order and position bytes are stepped over, which is the
/// positions-free [`PostingsReader`] walk.  `doc == u32::MAX` once the list
/// is exhausted (or its bytes fail to decode, as `PostingsReader::next`
/// stops on error).
pub struct BlockCursor<'a> {
    data: &'a [u8],
    df: u32,
    has_positions: bool,
    freq_omitted: bool,
    codec: PostCodec,
    blocks: Option<TermBlocks<'a>>,
    nfull: usize,
    nblocks: usize,
    /// `false`: full blocks' frequency frames are not unpacked (`tf()` reads 1)
    want_tf: bool,
    /// sequential mode: where the next block starts, and its delta base
    next_pos: usize,
    prev_last: u32,
    blk: usize,
    docs: [u32; BLOCK_SIZE],
    tfs: [u32; BLOCK_SIZE],
    len: usize,
    i: usize,
    /// current document (`u32::MAX` when exhausted)
    pub doc: u32,
    /// work counters: blocks and postings decoded
    pub blocks_decoded: u64,
    pub postings_decoded: u64,
    /// Where the decoded FULL block's position groups start (one group per
    /// document, in document order), when the frames before them were read
    /// or stepped over (`pos_known`); see [`Self::positions`].
    pos_base: usize,
    pos_known: bool,
    /// The next unread group of the decoded full block: its index and byte
    /// offset, so a forward walk through one block steps over each group once.
    pos_next: (usize, usize),
    /// The residual block's group offset per document (its groups are
    /// interleaved with the doc deltas and frequencies).
    res_pos: [u32; BLOCK_SIZE],
}

impl<'a> BlockCursor<'a> {
    /// `codec` / `has_positions` as [`PostingsReader::new_with_codec`] takes
    /// them; `blocks` = the term's block-index entry when the segment has one.
    /// The cursor starts on the first posting.
    pub fn new(
        data: &'a [u8],
        doc_frequency: u32,
        has_positions: bool,
        codec: PostCodec,
        blocks: Option<TermBlocks<'a>>,
    ) -> Self {
        Self::with_frequencies(data, doc_frequency, has_positions, codec, blocks, true)
    }

    /// [`Self::new`], optionally without unpacking the frequency frames of
    /// full blocks (`want_tf == false`: document ids only, `tf()` reads 1 —
    /// counting passes).
    pub fn with_frequencies(
        data: &'a [u8],
        doc_frequency: u32,
        has_positions: bool,
        codec: PostCodec,
        blocks: Option<TermBlocks<'a>>,
        want_tf: bool,
    ) -> Self {
        let nfull = doc_frequency as usize / BLOCK_SIZE;
        let nblocks = nfull + usize::from(!(doc_frequency as usize).is_multiple_of(BLOCK_SIZE));
        let freq_omitted = match codec {
            PostCodec::V1 => !has_positions,
            PostCodec::V2 { freq_omitted }
            | PostCodec::V3 { freq_omitted }
            | PostCodec::V4 { freq_omitted } => {
                freq_omitted || !has_positions
            }
        };
        // an index entry that does not describe this list is ignored
        let blocks = blocks.filter(|b| {
            b.len() == nblocks && b.offset.last().is_none_or(|&o| (o as usize) < data.len())
        });
        let mut c = BlockCursor {
            data,
            df: doc_frequency,
            has_positions,
            freq_omitted,
            codec,
            blocks,
            nfull,
            nblocks,
            want_tf,
            next_pos: 0,
            prev_last: 0,
            blk: NO_BLOCK,
            docs: [0; BLOCK_SIZE],
            tfs: [0; BLOCK_SIZE],
            len: 0,
            i: 0,
            doc: u32::MAX,
            blocks_decoded: 0,
            postings_decoded: 0,
            pos_base: 0,
            pos_known: false,
            pos_next: (0, 0),
            res_pos: [0; BLOCK_SIZE],
        };
        if nblocks > 0 {
            c.decode(0);
        }
        c
    }

    /// The current document's positions, ascending, decoded into `out`
    /// (cleared first) — the values [`PostingsReader`] yields for it.
    ///
    /// `Ok(false)` when this cursor cannot read them: a docs-only field, the
    /// legacy ZPS1 framing (its full blocks keep positions in one
    /// length-prefixed run), or a cursor built without frequencies on a
    /// segment with a block index (the frequency frame was not stepped over,
    /// so the positions' start is unknown). Callers fall back to a
    /// [`PostingsReader`] then.
    ///
    /// Only the groups up to the current document are touched: within a
    /// full block the cursor remembers the next unread group, so walking a
    /// block forward steps over every group once.
    pub fn positions(&mut self, out: &mut Vec<u32>) -> io::Result<bool> {
        out.clear();
        if self.doc == u32::MAX || !self.has_positions {
            return Ok(false);
        }
        let mut cur = Cursor::new(self.data);
        if self.blk < self.nfull {
            if !self.pos_known {
                return Ok(false);
            }
            let (mut idx, off) = if self.pos_next.0 <= self.i {
                self.pos_next
            } else {
                (0, self.pos_base)
            };
            cur.set_position(off as u64);
            while idx < self.i {
                skip_positions_of(&mut cur, self.codec, self.tfs[idx])?;
                idx += 1;
            }
            decode_positions_of_into(&mut cur, self.codec, self.tfs[self.i], out)?;
            self.pos_next = (self.i + 1, cur.position() as usize);
        } else {
            cur.set_position(u64::from(self.res_pos[self.i]));
            decode_positions_of_into(&mut cur, self.codec, self.tfs[self.i], out)?;
        }
        Ok(true)
    }

    /// The term's block-index entry, if the cursor uses one.
    pub fn blocks(&self) -> Option<&TermBlocks<'a>> {
        self.blocks.as_ref()
    }

    pub fn doc_frequency(&self) -> u32 {
        self.df
    }

    /// Number of blocks (full blocks + the residual).
    pub fn num_blocks(&self) -> usize {
        self.nblocks
    }

    /// Index of the decoded block (the one holding `doc`).
    pub fn block(&self) -> usize {
        self.blk
    }

    /// The decoded block's documents and frequencies.
    pub fn block_postings(&self) -> (&[u32], &[u32]) {
        (&self.docs[..self.len], &self.tfs[..self.len])
    }

    #[inline]
    pub fn tf(&self) -> u32 {
        self.tfs[self.i]
    }

    fn exhaust(&mut self) {
        self.doc = u32::MAX;
        self.i = 0;
        self.len = 0;
    }

    /// Decode block `b` (a later block than the current one); on corrupt
    /// bytes the cursor is exhausted.
    fn decode(&mut self, b: usize) {
        if self.decode_inner(b).is_err() || self.len == 0 {
            self.exhaust();
            return;
        }
        self.blk = b;
        self.i = 0;
        self.doc = self.docs[0];
        self.prev_last = self.docs[self.len - 1];
        self.blocks_decoded += 1;
        self.postings_decoded += self.len as u64;
        self.pos_next = (0, self.pos_base);
    }

    fn decode_inner(&mut self, b: usize) -> io::Result<()> {
        let (pos, base) = match &self.blocks {
            Some(t) => (
                t.offset[b] as usize,
                if b == 0 { 0 } else { t.last_doc[b - 1] },
            ),
            None => {
                if b != self.blk.wrapping_add(1) {
                    return Err(invalid("sequential cursor asked to jump"));
                }
                (self.next_pos, self.prev_last)
            }
        };
        if pos > self.data.len() {
            return Err(invalid("block offset past the term's bytes"));
        }
        let mut cur = Cursor::new(self.data);
        cur.set_position(pos as u64);
        if b < self.nfull {
            decode_full_block_docs(
                self.codec,
                self.data,
                &mut cur,
                base,
                b == 0,
                &mut self.docs,
            )?;
            // Without position counts, sequential mode needs the frequencies to
            // step over the positions.
            let tf_steps = self.blocks.is_none() && matches!(self.codec, PostCodec::V4 { .. });
            if self.has_positions && !self.freq_omitted && (self.want_tf || tf_steps) {
                decode_full_block_freqs(self.codec, self.data, &mut cur, &mut self.tfs)?;
            } else {
                if self.has_positions && !self.freq_omitted && self.blocks.is_none() {
                    // sequential mode still has to step over the frame
                    skip_full_block_freqs(self.codec, self.data, &mut cur)?;
                }
                self.tfs = [1; BLOCK_SIZE];
            }
            // The position groups follow the frames read (or stepped over)
            // above — known unless a frequency frame was left unread.
            self.pos_known = self.has_positions
                && !matches!(self.codec, PostCodec::V1)
                && (self.freq_omitted || self.want_tf || self.blocks.is_none());
            self.pos_base = cur.position() as usize;
            if self.blocks.is_none() && self.has_positions {
                // sequential mode must find the next block: step over positions
                match self.codec {
                    PostCodec::V1 => {
                        let len = cur.read_u32::<LittleEndian>()? as u64;
                        cur.set_position(cur.position() + len);
                    }
                    PostCodec::V2 { .. } | PostCodec::V3 { .. } | PostCodec::V4 { .. } => {
                        for k in 0..BLOCK_SIZE {
                            skip_positions_of(&mut cur, self.codec, self.tfs[k])?;
                        }
                    }
                }
            }
            self.len = BLOCK_SIZE;
        } else {
            let count = cur.read_u8()? as usize;
            if count == 0
                || count >= BLOCK_SIZE
                || count != self.df as usize - self.nfull * BLOCK_SIZE
            {
                return Err(invalid("residual count"));
            }
            let mut prev = base;
            for k in 0..count {
                prev = prev.wrapping_add(vbyte_decode(&mut cur)?);
                self.docs[k] = prev;
                self.tfs[k] = if self.has_positions {
                    let tf = if self.freq_omitted {
                        1
                    } else {
                        vbyte_decode(&mut cur)?
                    };
                    self.res_pos[k] = cur.position() as u32;
                    skip_positions_of(&mut cur, self.codec, tf)?;
                    tf
                } else {
                    1
                };
            }
            self.len = count;
        }
        self.next_pos = cur.position() as usize;
        Ok(())
    }

    /// Move to the next posting.
    #[inline]
    pub fn next(&mut self) {
        if self.doc == u32::MAX {
            return;
        }
        self.i += 1;
        if self.i < self.len {
            self.doc = self.docs[self.i];
        } else if self.blk + 1 < self.nblocks {
            self.decode(self.blk + 1);
        } else {
            self.exhaust();
        }
    }

    /// Move to the first posting with `doc >= target` (no-op when already
    /// there).  With a block index, blocks whose last doc is below `target`
    /// are jumped over without being read.
    pub fn advance(&mut self, target: u32) {
        if self.doc >= target {
            return;
        }
        if self.docs[self.len - 1] < target {
            let next = match &self.blocks {
                Some(t) => gallop(t.last_doc, self.blk + 1, target),
                None => {
                    // sequential: decode forward until a block reaches target
                    loop {
                        if self.blk + 1 >= self.nblocks {
                            self.exhaust();
                            return;
                        }
                        self.decode(self.blk + 1);
                        if self.doc == u32::MAX {
                            return;
                        }
                        if self.docs[self.len - 1] >= target {
                            break;
                        }
                    }
                    self.blk
                }
            };
            if next >= self.nblocks {
                self.exhaust();
                return;
            }
            if next != self.blk {
                self.decode(next);
                if self.doc == u32::MAX {
                    return;
                }
            }
        }
        let j = self.i + self.docs[self.i..self.len].partition_point(|&d| d < target);
        self.i = j;
        self.doc = self.docs[j];
    }

    /// Index of the block that would hold `target` (the first block, from
    /// the decoded one on, whose last doc is `>= target`), without decoding;
    /// `num_blocks()` when every remaining block ends before `target`.
    /// Needs the block index (returns the decoded block otherwise).
    #[inline]
    pub fn shallow_block(&self, target: u32) -> usize {
        match &self.blocks {
            Some(t) => gallop(t.last_doc, self.blk.min(self.nblocks), target),
            None => self.blk,
        }
    }
}

// ── PostingsReader ────────────────────────────────────────────────────────────

/// Inner framing of a term's postings blob — set by the `.post` envelope
/// magic the segment was written with (see `xerj_fts::index`).
///
/// `V2` is the ZPS2 framing (frame-of-reference blocks, derived lengths,
/// width-0 constant blocks).  `freq_omitted` marks a positioned term whose
/// `ttf == df`, whose freq stream was therefore never written; the reader
/// synthesises `term_freq = 1`, exactly as docs-only fields always have.
///
/// `V1` is the legacy ZPS1 framing; segments written before the ZPS2 bump
/// decode through it unchanged.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PostCodec {
    V1,
    V2 {
        freq_omitted: bool,
    },
    /// ZPS3 (experimental, see [`BlockCodec::EliasFanoAc`]): ZPS2 with each
    /// full block's doc ids Elias-Fano coded and its frequencies
    /// arithmetic-coded; residuals and positions as ZPS2.
    V3 {
        freq_omitted: bool,
    },
    /// ZPS4: ZPS2 without the per-document position count, which always equals
    /// the term frequency (see [`BlockCodec::PackedNoCounts`]).
    V4 {
        freq_omitted: bool,
    },
}

impl PostCodec {
    /// [`Self::v2_for`] for a ZPS3 envelope.
    pub fn v3_for(has_positions: bool, total_term_frequency: u64, doc_frequency: u32) -> Self {
        PostCodec::V3 {
            freq_omitted: has_positions && total_term_frequency == doc_frequency as u64,
        }
    }

    /// [`Self::v2_for`] for a ZPS4 envelope.
    pub fn v4_for(has_positions: bool, total_term_frequency: u64, doc_frequency: u32) -> Self {
        PostCodec::V4 {
            freq_omitted: has_positions && total_term_frequency == doc_frequency as u64,
        }
    }

    /// The codec decision a ZPS2 reader makes for one term from facts the
    /// `.meta` record already carries.  The writer applies the identical
    /// predicate at encode time, so the two cannot drift.
    pub fn v2_for(has_positions: bool, total_term_frequency: u64, doc_frequency: u32) -> Self {
        PostCodec::V2 {
            freq_omitted: has_positions && total_term_frequency == doc_frequency as u64,
        }
    }
}

/// Iterator over decoded doc IDs from a posting list.
pub struct PostingsReader<'a> {
    data: &'a [u8],
    cursor: Cursor<&'a [u8]>,
    /// Total doc count for this term (determines how many blocks to decode).
    doc_frequency: u32,
    /// Docs consumed so far.
    docs_read: u32,
    /// Current block's decoded doc IDs (up to BLOCK_SIZE).
    block_docs: Vec<u32>,
    /// Current block's decoded term freqs.
    block_freqs: Vec<u32>,
    /// Current block's decoded positions per doc.
    block_positions: Vec<Vec<u32>>,
    /// Index within the current block.
    block_idx: usize,
    /// Number of full blocks.
    num_full_blocks: usize,
    /// Full blocks processed so far.
    blocks_processed: usize,
    /// Whether we are in the residual section.
    in_residual: bool,
    /// Residual entries (decoded all at once).
    residual: Vec<DecodedPosting>,
    /// Index within the residual.
    residual_idx: usize,
    /// Last doc ID seen (for delta decoding across blocks).
    last_doc_id: u32,
    /// `true` when the posting list contains freq + position blocks.
    /// `false` for docs-only fields (keyword, numeric, ip), in which
    /// case the reader synthesises `term_freq = 1` and empty positions
    /// for every posting.
    has_positions: bool,
    /// Inner block framing (ZPS1 legacy vs ZPS2) + freq-elision marker.
    codec: PostCodec,
    /// `false` after [`Self::without_positions`]: the position stream is
    /// stepped over, not decoded, and postings come back with no positions.
    positions_wanted: bool,
}

#[derive(Debug, Clone)]
pub struct DecodedPosting {
    pub doc_id: u32,
    pub term_freq: u32,
    pub positions: Vec<u32>,
}

impl<'a> PostingsReader<'a> {
    /// Create a reader over the raw posting bytes for one term.
    /// Defaults to the positioned format.  For docs-only fields use
    /// [`PostingsReader::new_with_positions`] with `has_positions =
    /// false` — the reader will skip freq / position decoding and
    /// synthesise `term_freq = 1` for every posting.
    pub fn new(data: &'a [u8], doc_frequency: u32) -> Self {
        Self::new_with_positions(data, doc_frequency, true)
    }

    /// Like [`new`] but explicit about whether the posting list carries
    /// term frequencies + positions.
    ///
    /// This constructs the **legacy ZPS1** reader.  Segments written with
    /// the ZPS2 envelope must go through [`Self::new_with_codec`] — the
    /// block framing differs — which is what `FtsIndexReader`'s postings
    /// factory does for every caller.
    pub fn new_with_positions(data: &'a [u8], doc_frequency: u32, has_positions: bool) -> Self {
        Self::new_with_codec(data, doc_frequency, has_positions, PostCodec::V1)
    }

    /// Like [`new_with_positions`] but explicit about the inner block
    /// framing, so a ZPS2 segment (FOR blocks, derived lengths, optional
    /// freq elision) and a legacy ZPS1 segment decode through the same
    /// iterator.
    pub fn new_with_codec(
        data: &'a [u8],
        doc_frequency: u32,
        has_positions: bool,
        codec: PostCodec,
    ) -> Self {
        let num_full_blocks = doc_frequency as usize / BLOCK_SIZE;
        Self {
            data,
            cursor: Cursor::new(data),
            doc_frequency,
            docs_read: 0,
            block_docs: Vec::new(),
            block_freqs: Vec::new(),
            block_positions: Vec::new(),
            block_idx: 0,
            num_full_blocks,
            blocks_processed: 0,
            in_residual: false,
            residual: Vec::new(),
            residual_idx: 0,
            last_doc_id: 0,
            has_positions,
            codec,
            positions_wanted: true,
        }
    }

    /// Stop materialising positions.
    ///
    /// The reader still walks the position stream — the block layout is
    /// positions-dependent, so it cannot be ignored — but steps over its
    /// bytes instead of building one `Vec<u32>` per posting, and every
    /// posting comes back with empty `positions`. For callers that read only
    /// `(doc_id, term_freq)`, such as BM25 over a disjunction of terms: on a
    /// `match` of a question against 1 M Wikipedia passages the per-posting
    /// `Vec` allocate/clone/free was a third of the query's CPU.
    pub fn without_positions(mut self) -> Self {
        self.positions_wanted = false;
        self
    }

    /// Load the next block (or the residual) when the current one is used up.
    /// `None` when the list is exhausted or the bytes do not decode.
    fn ensure_loaded(&mut self) -> Option<()> {
        if self.docs_read >= self.doc_frequency {
            return None;
        }
        if !self.in_residual && self.block_idx >= self.block_docs.len() {
            if self.blocks_processed < self.num_full_blocks {
                self.decode_next_full_block().ok()?;
            } else {
                // Switch to residual
                self.decode_residual().ok()?;
                self.in_residual = true;
            }
        }
        if self.in_residual && self.residual_idx >= self.residual.len() {
            return None;
        }
        Some(())
    }

    /// Advance to the next posting. Returns `None` when exhausted.
    ///
    /// An inherent cursor method rather than an `Iterator` impl: decoding
    /// borrows the reader's internal buffers, which doesn't fit `Iterator`'s
    /// `Item` lifetime, so `next` stays inherent by design.
    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> Option<DecodedPosting> {
        self.ensure_loaded()?;

        if self.in_residual {
            let posting = self.residual[self.residual_idx].clone();
            self.residual_idx += 1;
            self.docs_read += 1;
            return Some(posting);
        }

        let doc_id = self.block_docs[self.block_idx];
        let term_freq = self.block_freqs[self.block_idx];
        let positions = if self.positions_wanted {
            self.block_positions[self.block_idx].clone()
        } else {
            Vec::new()
        };
        self.block_idx += 1;
        self.docs_read += 1;

        Some(DecodedPosting {
            doc_id,
            term_freq,
            positions,
        })
    }

    /// Advance to the next posting and return only its `(doc_id, term_freq)`.
    ///
    /// Same walk as [`Self::next`] without building a [`DecodedPosting`];
    /// together with [`Self::without_positions`] a step allocates nothing.
    pub fn next_doc(&mut self) -> Option<(u32, u32)> {
        self.ensure_loaded()?;

        let step = if self.in_residual {
            let posting = &self.residual[self.residual_idx];
            self.residual_idx += 1;
            (posting.doc_id, posting.term_freq)
        } else {
            let step = (
                self.block_docs[self.block_idx],
                self.block_freqs[self.block_idx],
            );
            self.block_idx += 1;
            step
        };
        self.docs_read += 1;
        Some(step)
    }

    fn decode_next_full_block(&mut self) -> io::Result<()> {
        // Read doc_id deltas
        let mut docs = [0u32; BLOCK_SIZE];
        decode_full_block_docs(
            self.codec,
            self.data,
            &mut self.cursor,
            self.last_doc_id,
            self.blocks_processed == 0,
            &mut docs,
        )?;

        // Absolute doc IDs, into the reader's own buffer: a block is decoded
        // once per 128 postings, and allocating fresh `Vec`s for it on every
        // block was measurable on million-posting stopwords.
        self.block_docs.resize(BLOCK_SIZE, 0);
        self.block_docs.copy_from_slice(&docs);
        self.last_doc_id = self.block_docs[BLOCK_SIZE - 1];

        // A positioned term whose freq stream was elided (ttf == df) or a
        // docs-only field: every frequency is 1 by construction, so the
        // block's freq bytes simply are not there.
        let freq_stored = self.has_positions && !self.freq_omitted();
        self.block_freqs.resize(BLOCK_SIZE, 1);
        if freq_stored {
            let mut freqs = [0u32; BLOCK_SIZE];
            decode_full_block_freqs(self.codec, self.data, &mut self.cursor, &mut freqs)?;
            self.block_freqs.copy_from_slice(&freqs);
        } else {
            self.block_freqs.fill(1);
        }

        if self.has_positions {
            if self.positions_wanted {
                self.block_positions = self.decode_block_positions()?;
            } else {
                self.skip_block_positions()?;
                self.block_positions.clear();
            }
        } else if self.positions_wanted {
            // Docs-only mode: synthesise empty positions.
            self.block_positions = vec![Vec::new(); BLOCK_SIZE];
        } else {
            self.block_positions.clear();
        }

        self.block_idx = 0;
        self.blocks_processed += 1;

        Ok(())
    }

    /// Step over one block's position stream — the bytes
    /// [`Self::decode_block_positions`] would decode — without building it.
    fn skip_block_positions(&mut self) -> io::Result<()> {
        match self.codec {
            PostCodec::V1 => {
                let pos_byte_len = self.cursor.read_u32::<LittleEndian>()? as usize;
                let pos_end = self.cursor.position() as usize + pos_byte_len;
                if pos_end > self.data.len() {
                    return Err(io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        "postings positions truncated",
                    ));
                }
                self.cursor.set_position(pos_end as u64);
            }
            PostCodec::V2 { .. } | PostCodec::V3 { .. } | PostCodec::V4 { .. } => {
                for k in 0..BLOCK_SIZE {
                    let tf = self.block_freqs[k];
                    skip_positions_of(&mut self.cursor, self.codec, tf)?;
                }
            }
        }
        Ok(())
    }

    /// `true` when this term's freq stream was never written — either a
    /// docs-only field or (ZPS2) a positioned term with `ttf == df`.
    fn freq_omitted(&self) -> bool {
        match self.codec {
            PostCodec::V1 => false,
            PostCodec::V2 { freq_omitted }
            | PostCodec::V3 { freq_omitted }
            | PostCodec::V4 { freq_omitted } => freq_omitted || !self.has_positions,
        }
    }

    /// Decode one block's positions (BLOCK_SIZE documents' worth of
    /// `count + deltas` vbyte groups).
    ///
    /// ZPS1 prefixes the stream with a `u32` byte length and the groups
    /// decode from that slice; ZPS2 dropped the prefix — the group count is
    /// structurally exactly BLOCK_SIZE, so the groups decode straight off
    /// the main cursor and pin its end themselves.
    fn decode_block_positions(&mut self) -> io::Result<Vec<Vec<u32>>> {
        let mut positions: Vec<Vec<u32>> = Vec::with_capacity(BLOCK_SIZE);
        match self.codec {
            PostCodec::V1 => {
                let pos_byte_len = self.cursor.read_u32::<LittleEndian>()? as usize;
                let pos_start = self.cursor.position() as usize;
                let pos_end = pos_start + pos_byte_len;
                if pos_end > self.data.len() {
                    return Err(io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        "postings positions truncated",
                    ));
                }
                let mut pos_cursor = Cursor::new(&self.data[pos_start..pos_end]);
                for _ in 0..BLOCK_SIZE {
                    positions.push(decode_position_group(&mut pos_cursor)?);
                }
                self.cursor.set_position(pos_end as u64);
            }
            PostCodec::V2 { .. } | PostCodec::V3 { .. } => {
                for _ in 0..BLOCK_SIZE {
                    positions.push(decode_position_group(&mut self.cursor)?);
                }
            }
            PostCodec::V4 { .. } => {
                for k in 0..BLOCK_SIZE {
                    let mut group = Vec::new();
                    let tf = self.block_freqs[k];
                    decode_positions_of_into(&mut self.cursor, self.codec, tf, &mut group)?;
                    positions.push(group);
                }
            }
        }
        Ok(positions)
    }

    fn decode_residual(&mut self) -> io::Result<()> {
        let count = self.cursor.read_u8()? as usize;
        let mut result = Vec::with_capacity(count);
        let mut prev_doc = self.last_doc_id;

        for _ in 0..count {
            let doc_delta = vbyte_decode(&mut self.cursor)?;
            let doc_id = prev_doc + doc_delta;
            prev_doc = doc_id;

            let (term_freq, positions) = if self.has_positions {
                // An elided freq stream (ttf == df) means every posting's
                // frequency is 1 — the residual's freq vbytes were never
                // written either, mirroring `encode_residual`.
                let term_freq = if self.freq_omitted() {
                    1u32
                } else {
                    vbyte_decode(&mut self.cursor)?
                };
                if !self.positions_wanted {
                    skip_positions_of(&mut self.cursor, self.codec, term_freq)?;
                    (term_freq, Vec::new())
                } else {
                    let pos_count = if matches!(self.codec, PostCodec::V4 { .. }) {
                        term_freq as usize
                    } else {
                        vbyte_decode(&mut self.cursor)? as usize
                    };
                    let mut positions = Vec::with_capacity(pos_count);
                    let mut prev_pos = 0u32;
                    for _ in 0..pos_count {
                        let delta = vbyte_decode(&mut self.cursor)?;
                        prev_pos += delta;
                        positions.push(prev_pos);
                    }
                    (term_freq, positions)
                }
            } else {
                (1u32, Vec::new())
            };

            result.push(DecodedPosting {
                doc_id,
                term_freq,
                positions,
            });
        }

        self.residual = result;
        self.residual_idx = 0;
        Ok(())
    }

    /// Advance to the first posting whose `doc_id >= target`.
    ///
    /// This is a **linear scan** — it calls [`Self::next`] until a doc_id at or
    /// past `target` is reached (or the list is exhausted).  There is no
    /// skip-list acceleration: the `.post` blob carries no on-disk skip table
    /// (see the module docs), so no block-level seek is possible.  Correct, but
    /// O(n) in the number of postings skipped.
    pub fn advance_to(&mut self, target: u32) -> Option<DecodedPosting> {
        loop {
            match self.next() {
                Some(p) if p.doc_id >= target => return Some(p),
                Some(_) => continue,
                None => return None,
            }
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn build_postings(pairs: &[(u32, u32, &[u32])]) -> (Vec<u8>, u32, u64) {
        let mut writer = PostingsWriter::new();
        for &(doc_id, _freq, positions) in pairs {
            for &pos in positions {
                writer.add_occurrence("test", doc_id, pos);
            }
        }
        let mut data = Vec::new();
        writer.encode_term("test", &mut data);
        let doc_freq = pairs.len() as u32;
        let ttf: u64 = pairs.iter().map(|&(_, freq, _)| freq as u64).sum();
        (data, doc_freq, ttf)
    }

    /// The reader a ZPS2 segment constructs for writer output: V2 framing
    /// with the freq-elision predicate the writer itself applied.
    fn v2_reader(data: &[u8], doc_freq: u32, ttf: u64) -> PostingsReader<'_> {
        PostingsReader::new_with_codec(data, doc_freq, true, PostCodec::v2_for(true, ttf, doc_freq))
    }

    #[test]
    fn roundtrip_small_posting_list() {
        let postings: Vec<(u32, u32, &[u32])> =
            vec![(1, 2, &[0, 5]), (3, 1, &[2]), (7, 3, &[0, 1, 2])];
        let (data, doc_freq, ttf) = build_postings(&postings);
        let mut reader = v2_reader(&data, doc_freq, ttf);

        let p = reader.next().unwrap();
        assert_eq!(p.doc_id, 1);
        assert_eq!(p.term_freq, 2);
        assert_eq!(p.positions, vec![0, 5]);

        let p = reader.next().unwrap();
        assert_eq!(p.doc_id, 3);

        let p = reader.next().unwrap();
        assert_eq!(p.doc_id, 7);
        assert_eq!(p.term_freq, 3);

        assert!(reader.next().is_none());
    }

    #[test]
    fn roundtrip_full_block() {
        // Create exactly 128 docs
        let mut writer = PostingsWriter::new();
        for i in 0u32..128 {
            writer.add_occurrence("term", i * 2, i); // even doc ids, one position each
        }
        let mut data = Vec::new();
        writer.encode_term("term", &mut data);

        // Every frequency is 1 => ttf == df => the freq stream is elided and
        // this exercises the synthesised-freq path on a full block.
        let mut reader = v2_reader(&data, 128, 128);
        let mut last_doc = u32::MAX;
        let mut count = 0u32;
        while let Some(p) = reader.next() {
            assert!(
                p.doc_id < last_doc || last_doc == u32::MAX || p.doc_id > last_doc,
                "doc IDs must be monotonically increasing"
            );
            if last_doc != u32::MAX {
                assert!(p.doc_id > last_doc);
            }
            assert_eq!(p.term_freq, 1, "elided freqs decode as 1");
            assert_eq!(p.positions, vec![count], "positions still round-trip");
            last_doc = p.doc_id;
            count += 1;
        }
        assert_eq!(count, 128);
    }

    #[test]
    fn vbyte_roundtrip() {
        for &v in &[0u32, 1, 127, 128, 255, 16383, 16384, u32::MAX / 2] {
            let mut buf = Vec::new();
            vbyte_encode(v, &mut buf);
            let decoded = vbyte_decode(&mut Cursor::new(buf.as_slice())).unwrap();
            assert_eq!(decoded, v, "vbyte failed for {}", v);
        }
    }

    /// `advance_to` is a *linear* scan (there is no on-disk skip table — see
    /// the module docs).  This asserts it still lands on the correct posting:
    /// the first doc_id at or past the target, across a block boundary, and
    /// that it returns `None` once the list is exhausted.
    #[test]
    fn advance_to_linear_scans_correctly() {
        // 200 docs => one full 128-doc block + a 72-doc residual.
        // doc_id = i * 3, so ids are 0, 3, 6, … 597.
        let mut writer = PostingsWriter::new();
        for i in 0u32..200 {
            writer.add_occurrence("term", i * 3, i);
        }
        let mut data = Vec::new();
        writer.encode_term("term", &mut data);

        // Target lands exactly on a doc (150 = 50 * 3), inside the first block.
        let mut reader = v2_reader(&data, 200, 200);
        let hit = reader.advance_to(150).expect("expected a hit at/after 150");
        assert_eq!(hit.doc_id, 150, "must return the exact match when present");

        // Target between two docs (301 is not a multiple of 3) crossing into
        // the residual section => first doc strictly greater is 303.
        let mut reader = v2_reader(&data, 200, 200);
        let hit = reader.advance_to(301).expect("expected a hit at/after 301");
        assert_eq!(hit.doc_id, 303, "must return the first doc_id >= target");

        // Target past the end yields None.
        let mut reader = v2_reader(&data, 200, 200);
        assert!(
            reader.advance_to(600).is_none(),
            "advance past the last doc must exhaust the reader"
        );
    }

    /// The writer always emits the ZPS2 framing now; a hand-built legacy
    /// ZPS1 blob must still decode through `PostCodec::V1`.  The fixture is
    /// one docs-only full block whose lanes are [0, 1, 1, …] — the first
    /// delta is the gap from the previous block's last doc (0), so the
    /// lanes reconstruct docs 0..=127.
    #[test]
    fn zps1_legacy_block_bytes_still_decode() {
        let mut data: Vec<u8> = Vec::new();
        data.push(1u8); // num_bits = 1
        data.extend_from_slice(&16u32.to_le_bytes()); // byte_len = 128 * 1 / 8
                                                      // 1-bit lanes: lane 0 = 0, lanes 1..=127 = 1 => 0xFE then 0xFF × 15.
        data.push(0xFE);
        data.extend(std::iter::repeat_n(0xFFu8, 15));
        // No residual: df is an exact multiple of BLOCK_SIZE.

        let mut reader = PostingsReader::new_with_positions(&data, 128, false);
        let mut expected = 0u32;
        let mut count = 0u32;
        while let Some(p) = reader.next() {
            assert_eq!(
                p.doc_id, expected,
                "contiguous deltas of 1 reconstruct 0..127"
            );
            assert_eq!(p.term_freq, 1, "docs-only synthesises freq 1");
            expected += 1;
            count += 1;
        }
        assert_eq!(count, 128);
    }

    /// `without_positions` + `next_doc` must walk exactly the `(doc_id,
    /// term_freq)` sequence the positions-decoding `next` walks — across full
    /// blocks and the residual, for varying and freq-elided (ttf == df) terms —
    /// and must not decode a single position group into a `Vec`.
    #[test]
    fn without_positions_walks_the_same_docs_and_freqs() {
        for n in [1usize, 127, 128, 129, 300, 1000] {
            for tf_one in [false, true] {
                let mut writer = PostingsWriter::new();
                let mut expected: Vec<(u32, u32)> = Vec::new();
                for i in 0..n as u32 {
                    let doc = i * 3 + 1;
                    let tf = if tf_one { 1 } else { 1 + i % 5 };
                    for p in 0..tf {
                        writer.add_occurrence("t", doc, p * 7 + i % 3);
                    }
                    expected.push((doc, tf));
                }
                let mut data = Vec::new();
                writer.encode_term("t", &mut data);
                let ttf: u64 = expected.iter().map(|&(_, tf)| tf as u64).sum();

                let mut full = v2_reader(&data, n as u32, ttf);
                let mut decoded = Vec::new();
                while let Some(p) = full.next() {
                    assert_eq!(p.positions.len() as u32, p.term_freq);
                    decoded.push((p.doc_id, p.term_freq));
                }
                assert_eq!(decoded, expected, "next(), n={n} tf_one={tf_one}");

                POSITION_GROUPS_MATERIALISED.with(|c| c.set(0));
                let mut lean = v2_reader(&data, n as u32, ttf).without_positions();
                let mut walked = Vec::new();
                while let Some(step) = lean.next_doc() {
                    walked.push(step);
                }
                assert_eq!(walked, expected, "next_doc(), n={n} tf_one={tf_one}");
                assert_eq!(
                    POSITION_GROUPS_MATERIALISED.with(|c| c.get()),
                    0,
                    "a positions-free walk must not build position Vecs"
                );

                // `next` on a positions-free reader: same docs, no positions.
                let mut lean_next = v2_reader(&data, n as u32, ttf).without_positions();
                let mut seen = Vec::new();
                while let Some(p) = lean_next.next() {
                    assert!(p.positions.is_empty());
                    seen.push((p.doc_id, p.term_freq));
                }
                assert_eq!(seen, expected, "next() without positions, n={n}");
            }
        }
    }

    /// The legacy ZPS1 framing prefixes each block's position stream with its
    /// byte length; a positions-free reader jumps over it and lands on the
    /// next block exactly where the decoding reader does.
    #[test]
    fn without_positions_steps_over_a_zps1_position_stream() {
        let block = |data: &mut Vec<u8>| {
            // doc-id deltas: lane 0 = 0 (gap from the previous block's last
            // doc), lanes 1..=127 = 1; 1-bit lanes, 16-byte payload.
            data.push(1u8);
            data.extend_from_slice(&16u32.to_le_bytes());
            data.push(0xFE);
            data.extend(std::iter::repeat_n(0xFFu8, 15));
            // freqs: every lane 1.
            data.push(1u8);
            data.extend_from_slice(&16u32.to_le_bytes());
            data.extend(std::iter::repeat_n(0xFFu8, 16));
            // positions: u32 byte length, then 128 groups of (count 1, delta 0).
            data.extend_from_slice(&256u32.to_le_bytes());
            for _ in 0..BLOCK_SIZE {
                data.push(0x81);
                data.push(0x80);
            }
        };
        let mut data: Vec<u8> = Vec::new();
        block(&mut data);
        block(&mut data);

        let collect = |mut r: PostingsReader<'_>| {
            let mut out = Vec::new();
            while let Some(step) = r.next_doc() {
                out.push(step);
            }
            out
        };
        let decoded = collect(PostingsReader::new_with_positions(&data, 256, true));
        let lean =
            collect(PostingsReader::new_with_positions(&data, 256, true).without_positions());
        assert_eq!(decoded.len(), 256);
        assert_eq!(decoded[127], (127, 1));
        assert_eq!(
            decoded[128],
            (127, 1),
            "second block's lane 0 is a zero gap"
        );
        assert_eq!(lean, decoded);
    }

    /// A ZPS2 constant block — dense doc IDs, so every delta is 1 — encodes
    /// to `[width 0][vbyte 1]` and nothing else, where ZPS1 paid
    /// 5 header/payload-formality bytes plus a 16-byte payload.
    #[test]
    fn zps2_constant_block_collapses_to_width_zero() {
        let mut writer = PostingsWriter::new();
        for i in 1u32..=128 {
            writer.add_occurrence("term", i, i);
        }
        let mut data = Vec::new();
        writer.encode_term("term", &mut data);

        assert_eq!(&data[..2], &[0u8, 0x81], "width-0 block then vbyte(1)");
        // Positions (128 groups of count=1, delta=i+1) dominate what is left;
        // the doc-id stream itself is over after two bytes.
        assert!(
            data.len() < 4 * 128,
            "whole blob must stay well under the old framing's ~21 B/doc block"
        );

        let mut reader = v2_reader(&data, 128, 128);
        for expected in 1u32..=128 {
            let p = reader.next().expect("128 postings");
            assert_eq!(p.doc_id, expected);
        }
        assert!(reader.next().is_none());
    }

    /// Frame-of-reference packs the block's SPREAD, not its maximum: a freq
    /// stream of 100..=101 needs 7 bits raw, 1 bit after subtracting min.
    #[test]
    fn zps2_for_packs_spread_not_max() {
        let mut writer = PostingsWriter::new();
        for i in 0u32..128 {
            // doc i gets freq alternating 100/101 via that many occurrences
            // of a distinct position count: freq = 100 + (i % 2).
            let freq = 100 + (i % 2);
            for pos in 0u32..freq {
                writer.add_occurrence("term", i * 10, pos);
            }
        }
        let mut data = Vec::new();
        writer.encode_term("term", &mut data);

        // Doc-id deltas: first 10 (0-0), then 10 repeatedly => constant 10
        // after the first lane… all lanes are 10 except lane 0 which is
        // doc 0 => delta 0? No: block[0].doc_id - 0 = 0, then +10 each.
        // spread = 10 => width 4.  Freq lanes: min 100, spread 1 => width 1.
        // Walk: [width][vbyte min][payload] for docids then freqs.
        let mut cursor = Cursor::new(data.as_slice());
        let docid_width = cursor.read_u8().unwrap();
        let docid_min = vbyte_decode(&mut cursor).unwrap();
        assert_eq!((docid_width, docid_min), (4u8, 0u32), "docid lanes 0..=10");
        cursor.set_position(cursor.position() + 128 * 4 / 8);

        let freq_width = cursor.read_u8().unwrap();
        let freq_min = vbyte_decode(&mut cursor).unwrap();
        assert_eq!((freq_width, freq_min), (1u8, 100u32), "freq lanes 100/101");

        let ttf: u64 = (0..128u32).map(|i| (100 + (i % 2)) as u64).sum();
        let df = 128u32;
        let mut reader = v2_reader(&data, df, ttf);
        for i in 0u32..128 {
            let p = reader.next().expect("128 postings");
            assert_eq!(p.doc_id, i * 10);
            assert_eq!(p.term_freq, 100 + (i % 2));
            assert_eq!(p.positions.len(), (100 + (i % 2)) as usize);
        }
        assert!(reader.next().is_none());
    }

    /// A WIDE block whose min and max share a binade gains no width from the
    /// frame subtraction — deltas [1, 100, 100, …] need 7 bits either way —
    /// so the encoder must emit `min = 0` with raw lanes, keeping the
    /// payload bytes identical to what ZPS1 carried (the outer zstd pass
    /// matches them across blocks; FOR-always measurably inflated the
    /// compressed `top_doc` field on the 100k harness).
    #[test]
    fn zps2_skips_for_on_wide_equal_width_blocks() {
        let mut writer = PostingsWriter::new();
        // doc 1 then gaps of 100: lanes [1, 100, 100, …], min 1, max 100 —
        // 7 bits framed (spread 99) and 7 bits raw, and 7 > the narrow cap.
        let mut doc = 1u32;
        for i in 0u32..128 {
            writer.add_occurrence("term", doc, i);
            doc += 100;
        }
        let mut data = Vec::new();
        writer.encode_term("term", &mut data);

        // [width 7][vbyte 0] — NOT [width 7][vbyte 1]: same width either
        // way and too wide for FOR's pattern-normalisation to pay, so the
        // raw lanes are kept for zstd to match.
        assert_eq!(&data[..2], &[7u8, 0x80], "wide equal-width block stays raw");

        let mut reader = v2_reader(&data, 128, 128);
        let mut expected = 1u32;
        for _ in 0..128 {
            let p = reader.next().expect("128 postings");
            assert_eq!(p.doc_id, expected);
            expected += 100;
        }
        assert!(reader.next().is_none());
    }

    /// The NARROW counterpart: an equal-width block at or below the cap
    /// takes the frame even though the width is unchanged, because FOR
    /// normalises level-shifted repeats into identical payloads.  Deltas
    /// [1, 15, 15, …] are 4 bits framed or raw — the frame is chosen.
    #[test]
    fn zps2_frames_narrow_equal_width_blocks() {
        let mut writer = PostingsWriter::new();
        let mut doc = 1u32;
        for i in 0u32..128 {
            writer.add_occurrence("term", doc, i);
            doc += 15;
        }
        let mut data = Vec::new();
        writer.encode_term("term", &mut data);

        assert_eq!(
            &data[..2],
            &[4u8, 0x81],
            "narrow equal-width block takes the frame (min = 1)"
        );

        let mut reader = v2_reader(&data, 128, 128);
        let mut expected = 1u32;
        for _ in 0..128 {
            let p = reader.next().expect("128 postings");
            assert_eq!(p.doc_id, expected);
            expected += 15;
        }
        assert!(reader.next().is_none());
    }

    /// Two terms over the same documents: one all freq 1 (ttf == df), one
    /// with a single freq-2 document.  The elided term's blob must be
    /// strictly smaller and both must round-trip with exact term_freqs.
    #[test]
    fn freq_elision_shrinks_and_roundtrips() {
        const ONE_POSITION: &[u32] = &[0];
        const TWO_POSITIONS: &[u32] = &[0, 1];
        let elided: Vec<(u32, u32, &[u32])> = (0u32..300).map(|i| (i, 1, ONE_POSITION)).collect();
        let mut repeated = elided.clone();
        repeated[5] = (5, 2, TWO_POSITIONS);

        let (data_elided, df_e, ttf_e) = build_postings(&elided);
        let (data_repeated, df_r, ttf_r) = build_postings(&repeated);
        assert_eq!((df_e, ttf_e), (300, 300));
        assert_eq!((df_r, ttf_r), (300, 301));
        assert!(
            data_elided.len() < data_repeated.len(),
            "elided freq stream must save bytes ({} vs {})",
            data_elided.len(),
            data_repeated.len()
        );

        let mut reader = v2_reader(&data_elided, df_e, ttf_e);
        for i in 0u32..300 {
            let p = reader.next().expect("300 postings");
            assert_eq!(p.term_freq, 1, "elided freqs synthesise 1");
            assert_eq!(p.doc_id, i);
        }
        assert!(reader.next().is_none());

        let mut reader = v2_reader(&data_repeated, df_r, ttf_r);
        for i in 0u32..300 {
            let p = reader.next().expect("300 postings");
            assert_eq!(p.term_freq, if i == 5 { 2 } else { 1 });
        }
        assert!(reader.next().is_none());
    }

    // ── block index + BlockCursor ──────────────────────────────────────────

    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0 >> 33
        }
    }

    /// One term's list: `(doc, tf)` per posting, written through a writer
    /// with or without positions; returns (encoded bytes, block-index entry,
    /// expected `(doc, scored tf)` walk, df, ttf).
    type IndexedList = (Vec<u8>, Option<TermBlockData>, Vec<(u32, u32)>, u32, u64);

    fn indexed_list(
        n: usize,
        positions: bool,
        tf_one: bool,
        seed: u64,
        norm_of: &dyn Fn(u32) -> u8,
    ) -> IndexedList {
        indexed_list_with(n, positions, tf_one, seed, norm_of, BlockCodec::Packed)
    }

    fn indexed_list_with(
        n: usize,
        positions: bool,
        tf_one: bool,
        seed: u64,
        norm_of: &dyn Fn(u32) -> u8,
        codec: BlockCodec,
    ) -> IndexedList {
        let mut rng = Lcg(seed);
        let mut writer = if positions {
            PostingsWriter::new()
        } else {
            PostingsWriter::new_no_positions()
        };
        let mut doc = rng.next() as u32 % 50;
        let mut expected = Vec::with_capacity(n);
        let mut ttf = 0u64;
        for _ in 0..n {
            // gaps 1..=40, occasionally huge (sparse lists, wide blocks)
            doc += 1
                + (rng.next() % 40) as u32
                + if rng.next().is_multiple_of(97) {
                    100_000
                } else {
                    0
                };
            let tf = if tf_one {
                1
            } else if rng.next().is_multiple_of(50) {
                256 + (rng.next() % 300) as u32 // saturating impacts
            } else {
                1 + (rng.next() % 4) as u32
            };
            for p in 0..tf {
                writer.add_occurrence("t", doc, p * 3);
            }
            ttf += tf as u64;
            expected.push((doc, if positions { tf } else { 1 }));
        }
        let mut data = Vec::new();
        let (off, blocks) = writer
            .encode_term_indexed_with("t", &mut data, norm_of, codec)
            .unwrap();
        assert_eq!(off, 0);
        if codec == BlockCodec::Packed {
            // identical bytes to the plain encoder
            let mut plain = Vec::new();
            writer.encode_term("t", &mut plain);
            assert_eq!(
                plain, data,
                "encode_term_indexed must not change the encoding"
            );
        }
        (data, blocks, expected, n as u32, ttf)
    }

    fn parsed(blocks: &TermBlockData) -> BlockIndex {
        let mut b = BlockIndexBuilder::default();
        b.add(0, blocks);
        let frame = b.finish(3).unwrap();
        let (payload, rest) = split_block_index_frame(&frame).expect("frame header");
        assert!(rest.is_empty());
        BlockIndex::parse(payload).unwrap()
    }

    fn walk(mut c: BlockCursor<'_>) -> Vec<(u32, u32)> {
        let mut out = Vec::new();
        while c.doc != u32::MAX {
            out.push((c.doc, c.tf()));
            c.next();
        }
        out
    }

    #[test]
    fn block_cursor_walks_and_advances_like_the_linear_reader() {
        let norm_of = |d: u32| (d.wrapping_mul(2654435761) >> 24) as u8;
        let mut case = 0u64;
        for n in [1usize, 5, 127, 128, 129, 255, 256, 257, 1000, 5003] {
            for (positions, tf_one) in [(true, false), (true, true), (false, false)] {
                case += 1;
                let (data, blocks, expected, df, ttf) =
                    indexed_list(n, positions, tf_one, case, &norm_of);
                assert_eq!(
                    blocks.is_some(),
                    n >= BLOCK_SIZE,
                    "n={n}: index only for full blocks"
                );
                let codec = PostCodec::v2_for(positions, ttf, df);
                // the reader this crate had: positions-free linear walk
                let mut reader =
                    PostingsReader::new_with_codec(&data, df, positions, codec).without_positions();
                let mut linear = Vec::new();
                while let Some(step) = reader.next_doc() {
                    linear.push(step);
                }
                assert_eq!(linear, expected, "n={n} positions={positions}");
                let index = blocks.as_ref().map(parsed);
                let term = index.as_ref().map(|ix| ix.term(0).unwrap());
                for tb in [term, None] {
                    let label = format!(
                        "n={n} positions={positions} tf_one={tf_one} indexed={}",
                        tb.is_some()
                    );
                    assert_eq!(
                        walk(BlockCursor::new(&data, df, positions, codec, tb)),
                        expected,
                        "{label}: walk"
                    );
                    // document ids only: the same documents
                    let docs_only: Vec<u32> = walk(BlockCursor::with_frequencies(
                        &data, df, positions, codec, tb, false,
                    ))
                    .into_iter()
                    .map(|(d, _)| d)
                    .collect();
                    let docs: Vec<u32> = expected.iter().map(|&(d, _)| d).collect();
                    assert_eq!(docs_only, docs, "{label}: docs-only walk");
                    // advance: single jumps from a fresh cursor, and a rising sequence
                    let mut rng = Lcg(case * 31 + 7);
                    let max = expected.last().unwrap().0 + 10;
                    let mut targets: Vec<u32> = (0..60).map(|_| rng.next() as u32 % max).collect();
                    targets.extend(expected.iter().step_by(37).map(|e| e.0));
                    for &t in &targets {
                        let mut c = BlockCursor::new(&data, df, positions, codec, tb);
                        c.advance(t);
                        match expected.iter().find(|e| e.0 >= t) {
                            Some(&(d, tf)) => {
                                assert_eq!((c.doc, c.tf()), (d, tf), "{label}: advance({t})")
                            }
                            None => {
                                assert_eq!(c.doc, u32::MAX, "{label}: advance({t}) past the end")
                            }
                        }
                    }
                    targets.sort_unstable();
                    let mut c = BlockCursor::new(&data, df, positions, codec, tb);
                    for &t in &targets {
                        c.advance(t);
                        let want = expected.iter().find(|e| e.0 >= t).map_or(u32::MAX, |e| e.0);
                        assert_eq!(c.doc, want, "{label}: rising advance({t})");
                    }
                }
                // impacts: every posting is dominated by a stored pair of its block and of its term
                if let (Some(tb), Some(ix)) = (blocks.as_ref(), index.as_ref()) {
                    let t = ix.term(0).unwrap();
                    assert_eq!(t.len(), n.div_ceil(BLOCK_SIZE));
                    assert!(t.term_impacts.len() <= MAX_TERM_IMPACTS && !t.term_impacts.is_empty());
                    let dominated = |pairs: &[Impact], tf: u32, norm: u8| {
                        pairs
                            .iter()
                            .any(|p| p.tf as u32 >= tf.min(255) && p.norm <= norm)
                    };
                    for (b, chunk) in expected.chunks(BLOCK_SIZE).enumerate() {
                        let pairs = t.block_impacts(b);
                        assert!(!pairs.is_empty() && pairs.len() <= MAX_BLOCK_IMPACTS);
                        assert_eq!(t.last_doc[b], chunk.last().unwrap().0);
                        for &(d, tf) in chunk {
                            assert!(
                                dominated(pairs, tf, norm_of(d)),
                                "block {b} misses ({tf}, {})",
                                norm_of(d)
                            );
                            assert!(
                                dominated(t.term_impacts, tf, norm_of(d)),
                                "term misses ({tf}, {})",
                                norm_of(d)
                            );
                        }
                    }
                    assert_eq!(tb.last_doc.len(), t.len());
                }
            }
        }
    }

    #[test]
    fn impact_frontier_is_minimal_saturates_and_folds() {
        // a 10-point staircase: tf 1..=10 at norms 10, 20, ..., 100, plus dominated noise
        let mut b = ImpactBuilder::default();
        for k in 1..=10u32 {
            b.add(k, (k * 10) as u8);
            b.add(k, (k * 10 + 5) as u8); // dominated by (k, k*10)
        }
        let mut full = Vec::new();
        b.drain_into(&mut full, 64);
        assert_eq!(full.len(), 10);
        assert!(full
            .windows(2)
            .all(|w| w[0].tf < w[1].tf && w[0].norm < w[1].norm));
        let mut b = ImpactBuilder::default();
        for k in 1..=10u32 {
            b.add(k, (k * 10) as u8);
        }
        let mut folded = Vec::new();
        b.drain_into(&mut folded, 4);
        assert_eq!(folded.len(), 4);
        for k in 1..=10u32 {
            assert!(
                folded
                    .iter()
                    .any(|p| p.tf as u32 >= k && p.norm as u32 <= k * 10),
                "fold lost ({k}, {})",
                k * 10
            );
        }
        // tf >= 255 saturates; a later 255 at a larger norm is dominated
        let mut b = ImpactBuilder::default();
        b.add(300, 7);
        b.add(1000, 9);
        b.add(2, 1);
        let mut sat = Vec::new();
        b.drain_into(&mut sat, 4);
        assert_eq!(
            sat,
            vec![Impact { tf: 2, norm: 1 }, Impact { tf: 255, norm: 7 }]
        );
        // the builder resets between drains
        let mut b2 = Vec::new();
        b.add(3, 3);
        b.drain_into(&mut b2, 4);
        assert_eq!(b2, vec![Impact { tf: 3, norm: 3 }]);
    }

    #[test]
    fn block_index_parse_rejects_damage() {
        let norm_of = |_d: u32| 9u8;
        let (_, blocks, _, _, _) = indexed_list(700, true, false, 99, &norm_of);
        let blocks = blocks.unwrap();
        let mut builder = BlockIndexBuilder::default();
        builder.add(40, &blocks);
        let frame = builder.finish(3).unwrap();
        let (payload, _) = split_block_index_frame(&frame).unwrap();
        assert!(BlockIndex::parse(payload).is_ok());
        assert!(
            BlockIndex::parse(&payload[..payload.len() - 3]).is_err(),
            "truncated"
        );
        let mut bad = payload.to_vec();
        bad[0] = b'X';
        assert!(BlockIndex::parse(&bad).is_err(), "magic");
        let mut bad = payload.to_vec();
        bad[4] = 2;
        assert!(BlockIndex::parse(&bad).is_err(), "version");
        // a block without impacts would read as a zero bound: refused
        let mut hollow = blocks.clone();
        hollow.impact_end[1] = hollow.impact_end[0];
        let mut builder = BlockIndexBuilder::default();
        builder.add(40, &hollow);
        let frame = builder.finish(3).unwrap();
        let (payload, _) = split_block_index_frame(&frame).unwrap();
        assert!(BlockIndex::parse(payload).is_err(), "empty impact range");
        // a frame header claiming more bytes than present is not a frame
        let mut short = frame.clone();
        short.truncate(frame.len() - 1);
        assert!(split_block_index_frame(&short).is_none());
        assert!(split_block_index_frame(b"ZPS2abcd").is_none());
    }

    #[test]
    fn elias_fano_ac_blocks_decode_like_packed_ones() {
        blocks_decode_like_packed_ones(BlockCodec::EliasFanoAc, PostCodec::v3_for);
    }

    #[test]
    fn no_count_blocks_decode_like_packed_ones() {
        blocks_decode_like_packed_ones(BlockCodec::PackedNoCounts, PostCodec::v4_for);
    }

    fn blocks_decode_like_packed_ones(
        block_codec: BlockCodec,
        post_codec: fn(bool, u64, u32) -> PostCodec,
    ) {
        let norm_of = |d: u32| (d % 251) as u8;
        let mut case = 500u64;
        for n in [128usize, 129, 255, 256, 1000, 4097] {
            for (positions, tf_one) in [(true, false), (true, true), (false, false)] {
                case += 1;
                let (packed, pblocks, expected, df, ttf) =
                    indexed_list_with(n, positions, tf_one, case, &norm_of, BlockCodec::Packed);
                let (efac, eblocks, expected2, _, _) = indexed_list_with(
                    n,
                    positions,
                    tf_one,
                    case,
                    &norm_of,
                    block_codec,
                );
                assert_eq!(expected, expected2);
                assert_eq!(
                    pblocks.as_ref().map(|b| &b.last_doc),
                    eblocks.as_ref().map(|b| &b.last_doc)
                );
                let codec = post_codec(positions, ttf, df);
                let label = format!("n={n} positions={positions} tf_one={tf_one}");
                // the linear reader (generic path, merges) on ZPS3 blocks
                let mut reader =
                    PostingsReader::new_with_codec(&efac, df, positions, codec).without_positions();
                let mut linear = Vec::new();
                while let Some(step) = reader.next_doc() {
                    linear.push(step);
                }
                assert_eq!(linear, expected, "{label}: linear");
                if positions {
                    // positions still decode (they follow the coded frames)
                    let mut full = PostingsReader::new_with_codec(&efac, df, positions, codec);
                    let mut same = PostingsReader::new_with_codec(
                        &packed,
                        df,
                        positions,
                        PostCodec::v2_for(positions, ttf, df),
                    );
                    let mut count = 0;
                    while let Some(p) = full.next() {
                        assert_eq!(p.positions.len() as u32, p.term_freq, "{label}: positions");
                        let q = same.next().unwrap();
                        assert_eq!(p.positions, q.positions, "{label}: positions");
                        count += 1;
                    }
                    assert_eq!(count, n);
                    // The cursor reads a document's positions at any point of the walk.
                    let term = eblocks.as_ref().map(parsed);
                    let term = term.as_ref().map(|ix| ix.term(0).unwrap());
                    for tb in [term, None] {
                        let mut c = BlockCursor::new(&efac, df, positions, codec, tb);
                        let mut same = PostingsReader::new_with_codec(
                            &packed,
                            df,
                            positions,
                            PostCodec::v2_for(positions, ttf, df),
                        );
                        let mut out = Vec::new();
                        while c.doc != u32::MAX {
                            let q = same.next().unwrap();
                            assert_eq!(c.doc, q.doc_id, "{label}");
                            if c.positions(&mut out).unwrap() {
                                assert_eq!(out, q.positions, "{label}: cursor positions");
                            }
                            c.next();
                        }
                    }
                }
                // the block cursor, indexed and sequential
                let index = eblocks.as_ref().map(parsed);
                let term = index.as_ref().map(|ix| ix.term(0).unwrap());
                for tb in [term, None] {
                    assert_eq!(
                        walk(BlockCursor::new(&efac, df, positions, codec, tb)),
                        expected,
                        "{label}: walk"
                    );
                    let docs_only: Vec<u32> = walk(BlockCursor::with_frequencies(
                        &efac, df, positions, codec, tb, false,
                    ))
                    .into_iter()
                    .map(|(d, _)| d)
                    .collect();
                    assert_eq!(
                        docs_only,
                        expected.iter().map(|e| e.0).collect::<Vec<_>>(),
                        "{label}: docs only"
                    );
                    let mut c = BlockCursor::new(&efac, df, positions, codec, tb);
                    for e in expected.iter().step_by(61) {
                        c.advance(e.0);
                        assert_eq!((c.doc, c.tf()), *e, "{label}: advance");
                    }
                }
                if !tf_one && positions && block_codec == BlockCodec::EliasFanoAc {
                    assert!(
                        efac.len() < packed.len() + n,
                        "{label}: {} vs {}",
                        efac.len(),
                        packed.len()
                    );
                }
            }
        }
    }
}
