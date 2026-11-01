//! Page-pointer directory for text that stays in the original file.
//!
//! File systems read whole pages, so a byte-exact offset buys no I/O over a page number. A
//! directory stores only the page in which each record (one line of a line-oriented file) starts
//! and finds the line inside the page at read time. For 11.3 M passages the passage level is
//! 1.82 MB; the same index built on byte offsets costs about 21 MB more.
//!
//! # Levels
//!
//! * Passage level (always): the start-page sequence as a bitvector. For every page, one `1` per
//!   record that starts in it, then one `0`, plus a select sample per 512 ones.
//! * Article level (optional): the start page of each article's first passage, and a
//!   passage-to-article boundary bitvector. An article is a run of consecutive records with the
//!   same key. Postings keep article ids; the directory maps an article to its lines.
//!
//! # Reads check themselves
//!
//! A page pointer finds a line by counting newlines, so after an edit it can land on a different,
//! well-formed line. Size and head/tail hashes do not catch every edit: bytes inserted in one place
//! and deleted in another pass them, and mtime can be set back. **Production reads therefore pass
//! `expect`**: [`PageDir::read`] returns the line only if its JSON `"id"` equals the expected id,
//! and [`Error::IdMismatch`] otherwise. Passing no expectation is possible only through
//! [`PageDir::read_unchecked`], which is meant for tests and tools.
//!
//! The id check cannot see an edit inside the right passage's own text. A caller that must notice
//! those compares [`line_hash`] with one stored at index time, through [`Expect::LineHash`].
//!
//! # Format
//!
//! Little-endian, version 2, written and read here byte for byte the same as the research module
//! `page_dir.py` (version 1 is read too). See `docs` in [`format`].
pub mod bits;
pub mod build;
pub mod format;
mod read;

pub use build::{build, ArticleKey, BuildInfo, BuildOptions};
pub use format::{source_identity, SourceIdentity, MAGIC, VERSION};
pub use read::{
    jsonl_id_key, jsonl_record_id, line_hash, open, Expect, Located, PageDir, ReadOptions,
};

use thiserror::Error as ThisError;

/// Everything that can go wrong. Every staleness error means the same thing: the file no longer
/// has the bytes the directory was built from, and no text was returned.
#[derive(Debug, ThisError)]
pub enum Error {
    #[error(transparent)]
    Io(#[from] std::io::Error),
    /// The directory file itself is damaged or of an unknown version.
    #[error("page directory: {0}")]
    Format(String),
    /// The caller used the API outside its bounds, for example a passage number past the end.
    #[error("page directory: {0}")]
    Invalid(String),
    /// The source's size or head/tail hashes differ from those recorded at build time.
    #[error("stale file: {0}")]
    SourceMismatch(String),
    /// A page does not hold the lines the directory expects.
    #[error("stale file: {0}")]
    Stale(String),
    /// A line was read but is not the expected one.
    #[error("stale file: passage {passage} should have id {expected:?}, but the line read from page {page} {found}; the file changed after the directory was built, so rebuild the directory")]
    IdMismatch {
        passage: u64,
        page: u64,
        expected: String,
        /// Plain words for what was there: `has id "..."` or `is not a record ...`.
        found: String,
    },
    /// A line was read but its hash is not the one stored at index time.
    #[error("stale file: passage {passage} on page {page} has hash {found:016x}, the index stored {expected:016x}; the file changed after indexing, so re-index it")]
    HashMismatch {
        passage: u64,
        page: u64,
        expected: u64,
        found: u64,
    },
    /// The file changed while a directory was being built.
    #[error("the file changed during the build")]
    ChangedDuringBuild,
    /// A record has no article key.
    #[error("record {record} at byte {offset} has no article key")]
    NoArticleKey { record: u64, offset: u64 },
}

impl Error {
    /// True for every error that means the original file changed.
    pub fn is_stale(&self) -> bool {
        matches!(
            self,
            Error::SourceMismatch(_)
                | Error::Stale(_)
                | Error::IdMismatch { .. }
                | Error::HashMismatch { .. }
        )
    }
}

pub type Result<T> = std::result::Result<T, Error>;
