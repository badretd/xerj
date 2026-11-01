# Block skip tables and block-max impacts for xerj-fts postings

Status: design for `perf/fts-blockmax` (on top of `fix/engine-query-latency-and-memory`, commit
40cb72e "positions-free BM25 term walk; MaxScore tail once track_total_hits is counted").
Scope: `crates/xerj-fts` only (postings format, reader, `match`-shaped disjunction traversal).

## 1. Why

A `match` query is planned as `bool { should: [term, term, ...] }` (one clause per analysed token,
repeats included) and runs through `FtsSearcher::wand_should_bool` once per segment. Measured on
the first 1,000,000 passages of the 10 GB Wikipedia corpus (8 segments of 128,000 documents, one
`text` field, standard analyzer, positions on; 50 HotpotQA questions, 17.5 clauses and 4.85 M
postings per question on average, one thread):

| build | track_total_hits | p50 | p95 |
|---|---|---:|---:|
| main (42f4798) | exact (only mode) | 352 ms | 3,612 ms |
| 40cb72e | exact | 160 ms | 2,003 ms |
| 40cb72e | 10,000 (ES default) | 116 ms | 310 ms |

(`examples/fts_corpus_bench.rs`; the full 1,337-question tables are in the commit bodies.) Every
variant still touches every posting of every clause:

* the format has no skip table: `PostingsReader::advance_to` is a linear `next()` loop, and a block's
  position stream has no stored length (ZPS2), so even a positions-free reader must read every
  position byte to find the next block;
* the only score bound is per term and loose: `idf * (k1 + 1) * boost`, the supremum over any
  (tf, dl). A term that occurs once per passage scores about `idf * 1.0` at average length, so the
  bound is about 2x the truth and the MaxScore partition keeps too many clauses essential;
* the MaxScore tail moves non-essential cursors with `next_doc()` up to each candidate, so it decodes
  their lists end to end (stopwords included) although it scores almost nothing.

## 2. Format extension: a block index frame inside the `.post` envelope

### 2.1 Where it lives

The `.post` side-car is `"ZPS2" | u32 uncompressed_len | zstd frame(postings)`. The block index is
written as a **zstd skippable frame** between the length and the postings frame:

```text
"ZPS2" | u32 uncompressed_len | skippable frame (magic 0x184D2A5B, u32 size, "ZBM1" payload) | zstd frame
```

* Old readers are unchanged and still correct: they hand everything after byte 8 to
  `zstd::bulk::decompress`, and zstd skips skippable frames by specification ("decoders shall skip
  them"); `uncompressed_len` still describes the postings frame only.
* New readers parse the frame when it is present and fall back to today's walk when it is not, so
  every existing segment (ZPS1, ZPL1, raw, ZPS2 without the frame) reads unchanged.
* No new file and no new file role: `xerj-storage`'s `valid_flush_artifact_set`, the `.complete`
  manifest (name, size, CRC of each file) and the bundle packer accept the segment family as it is.
  A new side-car would have been an "unknown role" to every older binary (its recovery path treats
  such a family as incomplete), and appending to `.meta` is impossible because the ZFM5 decoder
  rejects trailing bytes.
* Flush and merge share `write_field_static`, so both produce the frame.

### 2.2 What the frame holds (`ZBM1`, versioned by its magic)

For every term with at least one full 128-posting block (df >= 128), sorted by `postings_offset`:

| array | per | meaning |
|---|---|---|
| `postings_offset: u64` | term | key: the `TermPostings::postings_offset` the FST / `.meta` already returns |
| `first_block: u32` | term (+1) | index of the term's first block in the block arrays |
| `term_impacts` | term | the term's competitive (tf, norm) pairs (below) |
| `last_doc: u32` | block | last document id of the block (residual included as the final block) |
| `offset: u32` | block | byte offset of the block inside the term's postings bytes |
| `impacts` | block | the block's competitive (tf, norm) pairs |

The payload is itself zstd-compressed (it is decoded once at open into flat `Vec`s). A block is
decodable from `(offset[b], last_doc[b - 1])` alone: doc-id deltas are relative to the previous
block's last doc, and the doc and freq frames come first in the block, so the position stream is
never read (it is not even scanned to find the next block).

### 2.3 Impacts: the competitive (tf, norm byte) frontier

BM25 here is `idf(df) * tf_norm(tf, dl) * boost`, with `tf_norm` increasing in `tf` and decreasing in
`dl`, and `dl` a monotone function of the stored 1-byte norm (`norm_u8_to_u16`). `idf`, `avgdl` and
`N` change per request (index-wide `CollectionStats`, #188), so a score cannot be precomputed. What
is stable is the set of (tf, norm byte) pairs that are not dominated by another posting of the block
(higher-or-equal tf with lower-or-equal norm): the maximum of `tf_norm` over the block equals its
maximum over that frontier for every `k1 >= 0`, `b in [0, 1]`, `avgdl > 0` (Lucene stores the same
"competitive impacts" per block for the same reason). Rules:

* at most 4 pairs per block (8 per term); a longer frontier is folded by merging neighbours into
  (larger tf, smaller norm), which dominates both, so the bound stays an upper bound;
* tf is stored as `min(tf, 255)` and 255 means "at least 255": its `tf_norm` is bounded by `k1 + 1`;
* the block bound used at query time is `idf * boost * max(tf_norm(pair))` computed in f64 and
  inflated by a relative 1e-4 (the same margin as the 40cb72e MaxScore tail), so f32 rounding of the
  real score (the f32 `tf_norm` is not exactly monotone for huge tf) can never exceed it.

Size, from the 1 M build: about 14-16 bytes per 128 postings before compression (u32 last doc, u32
offset, frontier of 1-3 pairs), well under 1 % of the decompressed postings with positions.

### 2.4 Reader

`FtsIndexReader::open` parses the frame per field (`Option<BlockIndex>`); lookup by
`postings_offset` is a binary search. Norms move from `Vec<(u32, u16, u8)>` + binary search per
`field_length` to a dense `Vec<u8>` + a 256-entry length table (O(1), 1 byte per document instead of
8; `field_norm_bytes` iterates the dense array).

## 3. Traversal (as implemented)

`wand_should_bool` keeps its contract (40cb72e): the page is the generic path's page bit for bit
(each matched clause contributes `score_term(df, tf, dl) * clause boost` in f32, summed in clause
order, then `* bq.boost`; `TopN` with `ScoredHit` order: score desc, doc asc); the total is exact
below `count_limit` and a lower bound of at least `count_limit` (never above the exact total) past
it. The design above first kept counting and ranking in one doc-ordered pass; the implementation
splits them, because the total does not need scores and the page does not need every document:

* **Cursors.** One `postings::BlockCursor` per clause: `next`, `advance(target)` (galloping search
  over `last_doc`, then one block decode of docs + freqs from its stored offset), `shallow_block`
  (locate a block without decoding). A term without an index entry (old segment, or df < 128 where
  the whole list is one residual) walks its blocks in order and steps over position bytes, the
  positions-free walk 40cb72e introduced.
* **Bounds.** Per clause, a term bound from the term's pairs, else the exact maximum of a one-block
  list (it is decoded anyway), else the old supremum `idf * (k1 + 1) * boost`; per block, the bound
  of its pairs. All bounds are f64 from the scorer's own f32 constants (`k1 * (1 - b + b * dl /
  avgdl)` tabulated per norm byte per field per segment) and inflated by a relative 1e-4.
* **Total.** The union holds at least as many documents as its longest list: when that list
  reaches `count_limit`, the total is `count_limit` with nothing decoded. Otherwise
  (`track_total_hits: true`, or a union that may stay below the limit) every distinct list's
  document ids are OR-ed into a bitset (frequency frames are not unpacked) and counted exactly.
* **Page.** Block-max MaxScore. While the page is not full every document of every list is scored.
  Once it is, with `theta` = its worst score:
  1. Partition by term bounds (ascending): non-essential = the longest run whose f64 sum, times the
     bool boost, stays below `theta`. Re-partitioned whenever `theta` rises (a clause never returns
     to essential, so non-essential cursors only ever move forward).
  2. Candidate `d` = smallest current doc of the essential cursors.
  3. Window skip: if the essential clauses' current block bounds plus the non-essential term bounds
     stay below `theta`, no document up to the first end of an essential current block can enter:
     the essential cursors jump past it (blocks in between are not read).
  4. Candidate filter: the bounds of the essential blocks sitting on `d` plus the non-essential term
     bounds; then, if still competitive, the non-essential clauses' block bounds at `d` (located
     without decoding; 0 when a clause has no posting at `d`).
  5. Exact evaluation: essential contributions, then non-essential clauses probed largest bound
     first (one block decode each), stopping as soon as the remaining bounds cannot lift `d` to
     `theta`; the f32 score is formed in clause order and pushed.
  A skipped document's f32 score is below `theta` (the bound exceeds it by the margin), so it could
  not be on the final page (`theta` only grows); ties at `theta` are scored, because `TopN` admits
  an equal score with a smaller doc id.
* **Exact totals** (`track_total_hits: true`): the count pass decodes every document id of every
  list once; the page pass still skips. This is the Elasticsearch contract: `track_total_hits`
  defaults to 10,000, after which `hits.total` is `{"value": 10000, "relation": "gte"}`, and that
  relaxation is what makes skipping the count legal; a request that asks for exact totals keeps them.
* **Cross-segment floor** (additive API, `search_bounded_counting_floor`): each segment's walk starts
  from an empty page unless the caller passes the worst score of the page it has merged so far;
  with it, the walk prunes from its first document and returns the segment's top `cap` among its
  documents scoring at least the floor. The merged page is unchanged (tested). Lucene shares the
  minimum competitive score across leaves for the same reason. `xerj-engine`'s segment loop
  (`Index::search_inner`, segments without deletes, score-ordered pages) passes the lowest score
  of `all_hits` once it holds a full page; the engine test
  `match_walk_takes_the_merged_page_floor_and_keeps_the_page` checks that the page and the total
  are the same with and without it.

### 3.1 Experimental block codec (ZPS3)

An exact compressed full-block codec (Elias-Fano document ids, arithmetic-coded tf) behind the same
block index was measured on the 1 M build: full-block doc ids 73.9 -> 63.6 MB, tf 26.1 -> 12.6 MB,
decompressed postings 418.1 -> 394.4 MB, `.post` files 325.1 -> 314.2 MB, and a walk about 2x slower
(track_total_hits 10,000: p50 9.1 -> 18.4 ms). It is not part of this branch: it changes the `.post`
envelope magic, which a binary without it would misread (an unknown magic falls through to the
pre-envelope raw path), so it needs a data-dir format gate first. It is kept on `exp/zps3`.

## 4. Not in this change (recorded)

* Positions in a separate stream (so the doc/freq blocks are contiguous and smaller in memory), and
  uncompressed or chunk-compressed postings that can be memory-mapped instead of inflated whole at
  open: both are format changes with their own compatibility story.
* The ZPS3 codec (above): a format gate, then a 10 GB measurement.

## 5. Tests

* Parity (written first, kept green): randomized multi-block corpora (Zipf vocabulary, varying tf and
  lengths), duplicate and boosted clauses, bool boosts, caps 1..100, count limits 0..u64::MAX; the
  page must equal the generic path (`search_bounded_observed`) bit for bit and the totals must obey
  the contract above; once on segments with the block index and once on the same segments with the
  frame stripped (old-segment fallback).
* Format: block index round trip on random and real-shaped lists (offsets decode the same postings
  as a linear walk, frontier dominance, the 255 cap, the fold), an old-reader check (plain
  `zstd::bulk::decompress` of a new envelope returns the old bytes), merge output carries the frame.
* Benchmark: `examples/fts_corpus_bench.rs` (main, 40cb72e and this branch side by side; 1 M and
  10 GB; latency p50/p95 one thread, QPS with 4 threads, postings decoded and scored, parity with the
  generic path, recall@16 with `eval_recall.py`).
