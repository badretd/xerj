//! Errors. `ContractError` mirrors `reference_core.ContractError`: the same checks fail with the same messages.

use std::borrow::Cow;
use std::fmt;

/// Malformed input or an invalid provenance / completeness claim (reference_core.ContractError).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContractError(pub Cow<'static, str>);

impl ContractError {
    #[inline]
    pub const fn new(msg: &'static str) -> Self {
        ContractError(Cow::Borrowed(msg))
    }
    pub fn owned(msg: String) -> Self {
        ContractError(Cow::Owned(msg))
    }
    pub fn message(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for ContractError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for ContractError {}

/// Errors of persistence and parsing (I/O, corrupt files, JSON) plus contract violations found while replaying.
#[derive(Debug)]
pub enum StoreError {
    Contract(ContractError),
    Io(std::io::Error),
    Format(String),
    Json(serde_json::Error),
}

impl fmt::Display for StoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StoreError::Contract(e) => write!(f, "contract: {e}"),
            StoreError::Io(e) => write!(f, "io: {e}"),
            StoreError::Format(e) => write!(f, "format: {e}"),
            StoreError::Json(e) => write!(f, "json: {e}"),
        }
    }
}

impl std::error::Error for StoreError {}

impl From<ContractError> for StoreError {
    fn from(e: ContractError) -> Self {
        StoreError::Contract(e)
    }
}
impl From<std::io::Error> for StoreError {
    fn from(e: std::io::Error) -> Self {
        StoreError::Io(e)
    }
}
impl From<serde_json::Error> for StoreError {
    fn from(e: serde_json::Error) -> Self {
        StoreError::Json(e)
    }
}

// message constants (exact reference_core wording)
pub(crate) const E_DOC_REQUIRES: &str = "Document requires ID, version, and exact Unicode text";
pub(crate) const E_DOC_OVERWRITE: &str = "Document version identity cannot be overwritten";
pub(crate) const E_FACT_OVERWRITE: &str = "Fact identity cannot be overwritten";
pub(crate) const E_FACT_SOURCE: &str = "Fact must reference an available source version";
pub(crate) const E_QUOTE: &str = "Quote does not equal the referenced source span";
pub(crate) const E_SPAN: &str = "Invalid nonempty half-open code-point span";
pub(crate) const E_SUPERSEDE_TARGET: &str = "Supersession target must be an active stored fact";
pub(crate) const E_SUPERSEDE_KEY: &str = "Supersession cannot retract a different fact key";
pub(crate) const E_FACT_WITHDRAW: &str = "Fact is absent or already withdrawn";
pub(crate) const E_DOC_WITHDRAW: &str = "Document is absent or already withdrawn";
pub(crate) const E_FACT_IDS: &str = "Fact identifiers and writer version must be nonempty";
pub(crate) const E_INTERVAL: &str = "Fact validity interval must be nonempty";
pub(crate) const E_SNAPSHOT_REV: &str = "Invalid snapshot revision";
pub(crate) const E_KIND: &str = "Unsupported value kind";
pub(crate) const E_BOOL: &str = "Boolean requires a bool and no unit";
pub(crate) const E_NONEMPTY: &str = "Nonboolean values require a nonempty string";
pub(crate) const E_DATE_FORMAT: &str = "Dates must have exact YYYY-MM-DD precision";
pub(crate) const E_DATE_UNIT: &str = "Dates do not have a numeric unit";
pub(crate) const E_NUM_FINITE: &str = "Numeric values must be finite";
pub(crate) const E_NUM_INVALID: &str = "Invalid decimal";
pub(crate) const E_TEXT_UNIT: &str = "Text does not have a numeric unit";
pub(crate) const E_CAPACITY: &str = "Store capacity exceeded (u32 ids / revisions)";
