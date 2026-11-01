use super::*;
use serde::de::Deserializer as _;
use std::cell::Cell;
use std::collections::{HashMap, HashSet};

/// Logical work performed by row-selective hydration.
///
/// These fields are not allocator or RSS measurements. In particular,
/// decompressed buffers are counted by byte length while JSON values are
/// counted as objects, so the fields must never be summed into "memory used".
#[derive(Debug, Default, PartialEq, Eq)]
pub struct StoredV2RowHydrationStats {
    pub encoded_rows_visited: usize,
    pub selected_json_values: usize,
    pub selected_dictionary_entries: usize,
    pub decompressed_buffer_bytes: usize,
    pub output_values_cloned: usize,
}

#[derive(Debug, PartialEq)]
pub struct StoredV2HydratedRow {
    pub ordinal: usize,
    pub id: serde_json::Value,
    pub seq_no: serde_json::Value,
    pub source: serde_json::Map<String, serde_json::Value>,
}

#[derive(Debug, PartialEq)]
pub enum StoredV2RowHydrationResult {
    NotV2,
    Hydrated {
        rows: Vec<StoredV2HydratedRow>,
        stats: StoredV2RowHydrationStats,
    },
    /// Valid storage whose dependency shape cannot be hydrated selectively.
    /// The caller must use the compatibility path for the whole request.
    UnsupportedDependencyShape {
        column: String,
    },
}

type IdentityColumns = (usize, Vec<serde_json::Value>, Vec<serde_json::Value>);

pub(super) fn decode_v2_identity_columns_controlled(
    bytes: &[u8],
    checkpoint: &mut dyn FnMut(StoredDecodeCheckpoint) -> std::ops::ControlFlow<()>,
) -> Result<StoredDecodeRun<Option<IdentityColumns>>> {
    let directory = parse_v2_directory(&bytes[4..])?;
    let id_index = directory
        .columns
        .iter()
        .position(|column| column.name == "__id")
        .ok_or_else(|| StorageError::Other(anyhow::anyhow!("v2 missing __id column")))?;
    let seq_index = directory
        .columns
        .iter()
        .position(|column| column.name == "__seq_no")
        .ok_or_else(|| StorageError::Other(anyhow::anyhow!("v2 missing __seq_no column")))?;
    let mut selected = Vec::new();
    selected
        .try_reserve_exact(directory.num_docs)
        .map_err(|error| {
            StorageError::Other(anyhow::anyhow!(
                "cannot reserve {} identity row ordinals: {error}",
                directory.num_docs
            ))
        })?;
    selected.extend(0..directory.num_docs);
    let mut stats = StoredV2RowHydrationStats::default();
    let mut dict_ids = HashMap::new();
    let mut decode =
        |index| -> Result<std::result::Result<Vec<serde_json::Value>, Option<String>>> {
            if decode_cancelled(checkpoint, StoredDecodePhase::BeforeColumn, index, 0) {
                return Ok(Err(None));
            }
            let result = select_column_rows(
                index,
                &directory,
                &selected,
                &mut stats,
                &mut dict_ids,
                checkpoint,
                false,
            )?;
            if result.is_ok()
                && decode_cancelled(
                    checkpoint,
                    StoredDecodePhase::AfterColumn,
                    index,
                    directory.num_docs,
                )
            {
                return Ok(Err(None));
            }
            Ok(result)
        };
    let ids = match decode(id_index)? {
        Ok(values) => values,
        Err(None) => return Ok(StoredDecodeRun::Cancelled),
        Err(Some(_)) => return Ok(StoredDecodeRun::Complete(None)),
    };
    let seq_nos = match decode(seq_index)? {
        Ok(values) => values,
        Err(None) => return Ok(StoredDecodeRun::Cancelled),
        Err(Some(_)) => return Ok(StoredDecodeRun::Complete(None)),
    };
    Ok(StoredDecodeRun::Complete(Some((
        directory.num_docs,
        ids,
        seq_nos,
    ))))
}

/// `_id` of every row of a columnar (ZBS2/ZBS3) stored section, in row
/// order, decoded from the identity columns alone: no `_source` column is
/// read. `Ok(None)` for a non-columnar section or an identity shape that
/// cannot be read selectively; the caller decodes the whole section instead.
pub fn decode_stored_v2_ids(bytes: &[u8]) -> Result<Option<Vec<serde_json::Value>>> {
    let bytes = if bytes.len() >= 4 && &bytes[..4] == STORED_V3_MAGIC {
        split_stored_v3(bytes)?.1
    } else {
        bytes
    };
    if !is_columnar_stored_magic(bytes) {
        return Ok(None);
    }
    let mut no_cancel = |_| std::ops::ControlFlow::Continue(());
    match decode_v2_identity_columns_controlled(bytes, &mut no_cancel)? {
        StoredDecodeRun::Complete(Some((num_docs, ids, _))) if ids.len() == num_docs => {
            Ok(Some(ids))
        }
        StoredDecodeRun::Complete(_) | StoredDecodeRun::Cancelled => Ok(None),
    }
}

struct SelectedRowsVisitor<'a> {
    selected: &'a [usize],
    expected_rows: usize,
    column_index: usize,
    checkpoint: &'a mut dyn FnMut(StoredDecodeCheckpoint) -> std::ops::ControlFlow<()>,
    cancelled: &'a Cell<bool>,
}

impl<'de> serde::de::Visitor<'de> for SelectedRowsVisitor<'_> {
    type Value = Vec<serde_json::Value>;

    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("a JSON array matching the declared stored row count")
    }

    fn visit_seq<A>(self, mut sequence: A) -> std::result::Result<Self::Value, A::Error>
    where
        A: serde::de::SeqAccess<'de>,
    {
        let mut selected_index = 0usize;
        let mut values = Vec::with_capacity(self.selected.len());
        for row in 0..self.expected_rows {
            if self.selected.get(selected_index).copied() == Some(row) {
                values.push(
                    sequence
                        .next_element::<serde_json::Value>()?
                        .ok_or_else(|| serde::de::Error::custom("stored column ended early"))?,
                );
                selected_index += 1;
            } else {
                sequence
                    .next_element::<serde::de::IgnoredAny>()?
                    .ok_or_else(|| serde::de::Error::custom("stored column ended early"))?;
            }
            let processed = row + 1;
            if processed.is_multiple_of(STORED_DECODE_CHECK_INTERVAL)
                && (self.checkpoint)(StoredDecodeCheckpoint {
                    phase: StoredDecodePhase::Rows,
                    column_index: self.column_index,
                    rows_processed: processed,
                })
                .is_break()
            {
                self.cancelled.set(true);
                return Err(serde::de::Error::custom("stored decode cancelled"));
            }
        }
        if sequence.next_element::<serde::de::IgnoredAny>()?.is_some() {
            return Err(serde::de::Error::custom(
                "stored column exceeds declared row count",
            ));
        }
        Ok(values)
    }
}

fn select_json_rows<R: std::io::Read>(
    reader: R,
    selected: &[usize],
    expected_rows: usize,
    context: &str,
    column_index: usize,
    checkpoint: &mut dyn FnMut(StoredDecodeCheckpoint) -> std::ops::ControlFlow<()>,
) -> Result<std::result::Result<Vec<serde_json::Value>, ()>> {
    let cancelled = Cell::new(false);
    let mut deserializer = serde_json::Deserializer::from_reader(buffered_json_stream(reader));
    let values = match deserializer.deserialize_seq(SelectedRowsVisitor {
        selected,
        expected_rows,
        column_index,
        checkpoint,
        cancelled: &cancelled,
    }) {
        Ok(values) => values,
        Err(_) if cancelled.get() => return Ok(Err(())),
        Err(error) => {
            return Err(StorageError::Other(anyhow::anyhow!("{context}: {error}")));
        }
    };
    deserializer
        .end()
        .map_err(|error| StorageError::Other(anyhow::anyhow!("{context}: {error}")))?;
    Ok(Ok(values))
}

/// [`select_json_rows`] over a column already decompressed in memory. The
/// slice parser skips unselected rows several times faster than the
/// byte-at-a-time reader parser; validation is the same.
fn select_json_rows_slice(
    bytes: &[u8],
    selected: &[usize],
    expected_rows: usize,
    context: &str,
    column_index: usize,
    checkpoint: &mut dyn FnMut(StoredDecodeCheckpoint) -> std::ops::ControlFlow<()>,
) -> Result<std::result::Result<Vec<serde_json::Value>, ()>> {
    let cancelled = Cell::new(false);
    let mut deserializer = serde_json::Deserializer::from_slice(bytes);
    let values = match deserializer.deserialize_seq(SelectedRowsVisitor {
        selected,
        expected_rows,
        column_index,
        checkpoint,
        cancelled: &cancelled,
    }) {
        Ok(values) => values,
        Err(_) if cancelled.get() => return Ok(Err(())),
        Err(error) => {
            return Err(StorageError::Other(anyhow::anyhow!("{context}: {error}")));
        }
    };
    deserializer
        .end()
        .map_err(|error| StorageError::Other(anyhow::anyhow!("{context}: {error}")))?;
    Ok(Ok(values))
}

fn unpack_id(packed: &[u8], bit_width: u8, ordinal: usize) -> Option<u32> {
    let width = bit_width as usize;
    if !(1..=32).contains(&width) {
        return None;
    }
    let bit_position = ordinal.checked_mul(width)?;
    let byte_position = bit_position / 8;
    let shift = bit_position % 8;
    let needed = (width + shift).div_ceil(8);
    let bytes = packed.get(byte_position..byte_position.checked_add(needed)?)?;
    let mut word = 0u64;
    for (offset, byte) in bytes.iter().enumerate() {
        word |= (*byte as u64) << (offset * 8);
    }
    Some(((word >> shift) & ((1u64 << width) - 1)) as u32)
}

struct SelectedDict {
    values: Vec<serde_json::Value>,
    ids: Vec<u32>,
    selected_entries: usize,
    decompressed_bytes: usize,
}

fn decode_dict_rows(
    payload: &[u8],
    selected: &[usize],
    num_docs: usize,
    column_index: usize,
    checkpoint: &mut dyn FnMut(StoredDecodeCheckpoint) -> std::ops::ControlFlow<()>,
) -> Result<std::result::Result<SelectedDict, ()>> {
    let mut cursor = Cursor::new(payload);
    let dict_count = cursor
        .read_u32::<LittleEndian>()
        .map_err(|error| StorageError::Other(anyhow::anyhow!("dict_count: {error}")))?
        as usize;
    if dict_count > num_docs {
        return Err(StorageError::Other(anyhow::anyhow!(
            "dict_count {dict_count} exceeds num_docs {num_docs}"
        )));
    }
    let bit_width = cursor
        .read_u8()
        .map_err(|error| StorageError::Other(anyhow::anyhow!("bit_width: {error}")))?;
    if !(1..=32).contains(&bit_width) {
        return Err(StorageError::Other(anyhow::anyhow!(
            "dict invalid bit_width {bit_width}"
        )));
    }
    let dict_len = cursor
        .read_u32::<LittleEndian>()
        .map_err(|error| StorageError::Other(anyhow::anyhow!("dict_zstd_len: {error}")))?
        as usize;
    let dict_start = cursor.position() as usize;
    let dict_end = dict_start
        .checked_add(dict_len)
        .filter(|end| *end <= payload.len())
        .ok_or_else(|| StorageError::Other(anyhow::anyhow!("dict payload truncated")))?;
    cursor.set_position(dict_end as u64);
    let ids_len = cursor
        .read_u32::<LittleEndian>()
        .map_err(|error| StorageError::Other(anyhow::anyhow!("ids_len: {error}")))?
        as usize;
    if ids_len != num_docs {
        return Err(StorageError::Other(anyhow::anyhow!(
            "dict ids_len {ids_len} != num_docs {num_docs}"
        )));
    }
    let packed_len = cursor
        .read_u32::<LittleEndian>()
        .map_err(|error| StorageError::Other(anyhow::anyhow!("packed_zstd_len: {error}")))?
        as usize;
    let packed_start = cursor.position() as usize;
    let packed_end = packed_start
        .checked_add(packed_len)
        .filter(|end| *end == payload.len())
        .ok_or_else(|| StorageError::Other(anyhow::anyhow!("dict packed truncated/trailing")))?;
    let expected_packed_len = num_docs
        .checked_mul(bit_width as usize)
        .ok_or_else(|| StorageError::Other(anyhow::anyhow!("dict packed length overflow")))?
        .div_ceil(8);
    let packed = decode_zstd_exact(
        &payload[packed_start..packed_end],
        expected_packed_len,
        "packed zstd",
    )?;
    let used_bits = num_docs * bit_width as usize;
    if !used_bits.is_multiple_of(8) {
        let used_in_last = used_bits % 8;
        let padding_mask = !((1u8 << used_in_last) - 1);
        if packed.last().is_some_and(|byte| byte & padding_mask != 0) {
            return Err(StorageError::Other(anyhow::anyhow!(
                "dict non-zero packed padding"
            )));
        }
    }
    let null_id = dict_count as u32;
    // Validate every encoded id without constructing per-row Values.
    for ordinal in 0..num_docs {
        let id = unpack_id(&packed, bit_width, ordinal)
            .ok_or_else(|| StorageError::Other(anyhow::anyhow!("dict id truncated")))?;
        if id > null_id {
            return Err(StorageError::Other(anyhow::anyhow!(
                "dict id {id} outside dictionary"
            )));
        }
        if (ordinal + 1).is_multiple_of(STORED_DECODE_CHECK_INTERVAL)
            && checkpoint(StoredDecodeCheckpoint {
                phase: StoredDecodePhase::Rows,
                column_index,
                rows_processed: ordinal + 1,
            })
            .is_break()
        {
            return Ok(Err(()));
        }
    }
    let ids: Vec<u32> = selected
        .iter()
        .map(|ordinal| {
            unpack_id(&packed, bit_width, *ordinal)
                .ok_or_else(|| StorageError::Other(anyhow::anyhow!("dict selected id truncated")))
        })
        .collect::<Result<_>>()?;
    let mut needed: Vec<usize> = ids
        .iter()
        .copied()
        .filter(|id| *id != null_id)
        .map(|id| id as usize)
        .collect();
    needed.sort_unstable();
    needed.dedup();
    let decoder = zstd::stream::read::Decoder::new(&payload[dict_start..dict_end])
        .map_err(|error| StorageError::Other(anyhow::anyhow!("dict zstd: {error}")))?;
    let entries = match select_json_rows(
        decoder,
        &needed,
        dict_count,
        "dict json",
        column_index,
        checkpoint,
    )? {
        Ok(entries) => entries,
        Err(()) => return Ok(Err(())),
    };
    let entries: HashMap<usize, serde_json::Value> = needed.iter().copied().zip(entries).collect();
    let values = ids
        .iter()
        .map(|id| {
            if *id == null_id {
                Ok(serde_json::Value::Null)
            } else {
                entries.get(&(*id as usize)).cloned().ok_or_else(|| {
                    StorageError::Other(anyhow::anyhow!("selected dict entry missing"))
                })
            }
        })
        .collect::<Result<_>>()?;
    Ok(Ok(SelectedDict {
        values,
        ids,
        selected_entries: needed.len(),
        decompressed_bytes: packed.len(),
    }))
}

fn decode_cross_dep_rows(
    column: &V2ColumnRef<'_>,
    selected: &[usize],
    source_ids: &[u32],
    num_docs: usize,
    stats: &mut StoredV2RowHydrationStats,
    column_index: usize,
    checkpoint: &mut dyn FnMut(StoredDecodeCheckpoint) -> std::ops::ControlFlow<()>,
) -> Result<std::result::Result<Vec<serde_json::Value>, ()>> {
    let body = decode_cross_dep_body(column.payload, num_docs)?;
    stats.decompressed_buffer_bytes += body.len();
    let mut cursor = Cursor::new(body.as_slice());
    let _source_index = cursor
        .read_u32::<LittleEndian>()
        .map_err(|error| StorageError::Other(anyhow::anyhow!("cross_dep src_ix: {error}")))?;
    let dict_count = cursor
        .read_u32::<LittleEndian>()
        .map_err(|error| StorageError::Other(anyhow::anyhow!("cross_dep dict_count: {error}")))?
        as usize;
    if dict_count > num_docs {
        return Err(StorageError::Other(anyhow::anyhow!(
            "cross_dep dict_count {dict_count} exceeds num_docs {num_docs}"
        )));
    }
    let mut modes = Vec::with_capacity(dict_count);
    for _ in 0..dict_count {
        modes.push(
            cursor
                .read_i64::<LittleEndian>()
                .map_err(|error| StorageError::Other(anyhow::anyhow!("cross_dep mode: {error}")))?,
        );
    }
    let exception_count = cursor
        .read_u32::<LittleEndian>()
        .map_err(|error| StorageError::Other(anyhow::anyhow!("cross_dep exc_count: {error}")))?
        as usize;
    if exception_count > num_docs {
        return Err(StorageError::Other(anyhow::anyhow!(
            "cross_dep exc_count {exception_count} exceeds num_docs {num_docs}"
        )));
    }
    let wanted: HashSet<u32> = selected.iter().map(|ordinal| *ordinal as u32).collect();
    let mut exceptions = HashMap::with_capacity(wanted.len().min(exception_count));
    let mut position = cursor.position() as usize;
    let mut previous = 0u32;
    for exception_index in 0..exception_count {
        let delta = u32::try_from(read_varint(&body, &mut position)?)
            .map_err(|_| StorageError::Other(anyhow::anyhow!("cross_dep ordinal overflow")))?;
        let value = read_zigzag_i64(&body, &mut position)?;
        let ordinal = previous
            .checked_add(delta)
            .ok_or_else(|| StorageError::Other(anyhow::anyhow!("cross_dep ordinal overflow")))?;
        if ordinal as usize >= num_docs || (exception_index > 0 && ordinal <= previous) {
            return Err(StorageError::Other(anyhow::anyhow!(
                "cross_dep invalid exception ordinal {ordinal}"
            )));
        }
        if wanted.contains(&ordinal) {
            exceptions.insert(ordinal, value);
        }
        previous = ordinal;
        if (exception_index + 1).is_multiple_of(STORED_DECODE_CHECK_INTERVAL)
            && checkpoint(StoredDecodeCheckpoint {
                phase: StoredDecodePhase::Rows,
                column_index,
                rows_processed: exception_index + 1,
            })
            .is_break()
        {
            return Ok(Err(()));
        }
    }
    if position != body.len() {
        return Err(StorageError::Other(anyhow::anyhow!(
            "cross_dep trailing exception bytes"
        )));
    }
    Ok(Ok(selected
        .iter()
        .zip(source_ids)
        .map(|(ordinal, source_id)| {
            if let Some(value) = exceptions.get(&(*ordinal as u32)).copied() {
                if value == i64::MIN + 1 {
                    serde_json::Value::Null
                } else {
                    serde_json::Value::Number(value.into())
                }
            } else {
                let value = modes.get(*source_id as usize).copied().unwrap_or(i64::MIN);
                if value == i64::MIN {
                    serde_json::Value::Null
                } else {
                    serde_json::Value::Number(value.into())
                }
            }
        })
        .collect()))
}

fn repeat_constant_rows_controlled(
    value: serde_json::Value,
    output_count: usize,
    column_index: usize,
    checkpoint: &mut dyn FnMut(StoredDecodeCheckpoint) -> std::ops::ControlFlow<()>,
) -> Result<std::result::Result<Vec<serde_json::Value>, ()>> {
    let mut values = Vec::new();
    values.try_reserve_exact(output_count).map_err(|error| {
        StorageError::Other(anyhow::anyhow!(
            "cannot reserve {output_count} selected constant values: {error}"
        ))
    })?;
    for output_index in 0..output_count {
        values.push(value.clone());
        let processed = output_index + 1;
        if processed.is_multiple_of(STORED_DECODE_CHECK_INTERVAL)
            && checkpoint(StoredDecodeCheckpoint {
                phase: StoredDecodePhase::Rows,
                column_index,
                rows_processed: processed,
            })
            .is_break()
        {
            return Ok(Err(()));
        }
    }
    Ok(Ok(values))
}

/// `in_memory`: decompress a JSON column whole and parse it as a slice
/// (faster) instead of streaming it. Only the non-cancellable parallel
/// hydration asks for it, so cancellation still never waits on a whole
/// column's decompression.
#[allow(clippy::too_many_arguments)]
fn select_column_rows(
    column_index: usize,
    directory: &V2Directory<'_>,
    selected: &[usize],
    stats: &mut StoredV2RowHydrationStats,
    dict_ids: &mut HashMap<usize, Vec<u32>>,
    checkpoint: &mut dyn FnMut(StoredDecodeCheckpoint) -> std::ops::ControlFlow<()>,
    in_memory: bool,
) -> Result<std::result::Result<Vec<serde_json::Value>, Option<String>>> {
    let column = &directory.columns[column_index];
    let values = match column.codec {
        ColCodec::RawJson if in_memory => {
            stats.encoded_rows_visited += directory.num_docs;
            let frame = decode_raw_payload(column.payload)?;
            let raw = zstd::stream::decode_all(frame)
                .map_err(|error| StorageError::Other(anyhow::anyhow!("raw zstd: {error}")))?;
            stats.decompressed_buffer_bytes += raw.len();
            match select_json_rows_slice(
                &raw,
                selected,
                directory.num_docs,
                "raw json",
                column_index,
                checkpoint,
            )? {
                Ok(values) => values,
                Err(()) => return Ok(Err(None)),
            }
        }
        ColCodec::RawJson => {
            stats.encoded_rows_visited += directory.num_docs;
            let frame = decode_raw_payload(column.payload)?;
            let decoder = zstd::stream::read::Decoder::new(frame)
                .map_err(|error| StorageError::Other(anyhow::anyhow!("raw zstd: {error}")))?;
            match select_json_rows(
                decoder,
                selected,
                directory.num_docs,
                "raw json",
                column_index,
                checkpoint,
            )? {
                Ok(values) => values,
                Err(()) => return Ok(Err(None)),
            }
        }
        ColCodec::RawJsonBlocks => {
            // Only the blocks that hold a selected row are decompressed.
            let blocks = RawBlocks::parse(column.payload, directory.num_docs)?;
            let mut values = Vec::with_capacity(selected.len());
            let mut next = 0usize;
            for block in 0..blocks.len() {
                let Some(&row) = selected.get(next) else {
                    break;
                };
                let rows = blocks.rows(block);
                if row >= rows.end {
                    continue;
                }
                let local: Vec<usize> = selected[next..]
                    .iter()
                    .take_while(|&&row| row < rows.end)
                    .map(|&row| row - rows.start)
                    .collect();
                let raw = blocks.decode(block)?;
                stats.encoded_rows_visited += rows.len();
                stats.decompressed_buffer_bytes += raw.len();
                match select_json_rows_slice(
                    &raw,
                    &local,
                    rows.len(),
                    "raw json block",
                    column_index,
                    checkpoint,
                )? {
                    Ok(block_values) => values.extend(block_values),
                    Err(()) => return Ok(Err(None)),
                }
                next += local.len();
            }
            if let Some(row) = selected.get(next) {
                return Err(StorageError::Other(anyhow::anyhow!(
                    "selected stored row {row} is past the column's {} rows",
                    directory.num_docs
                )));
            }
            values
        }
        ColCodec::Lz4Json => {
            let raw = lz4_flex::decompress_size_prepended(column.payload)
                .map_err(|error| StorageError::Other(anyhow::anyhow!("lz4 decode: {error}")))?;
            stats.decompressed_buffer_bytes += raw.len();
            stats.encoded_rows_visited += directory.num_docs;
            match select_json_rows(
                raw.as_slice(),
                selected,
                directory.num_docs,
                "lz4 json",
                column_index,
                checkpoint,
            )? {
                Ok(values) => values,
                Err(()) => return Ok(Err(None)),
            }
        }
        ColCodec::Constant => {
            let value: serde_json::Value = serde_json::from_slice(column.payload)
                .map_err(|error| StorageError::Other(anyhow::anyhow!("constant: {error}")))?;
            match repeat_constant_rows_controlled(value, selected.len(), column_index, checkpoint)?
            {
                Ok(values) => values,
                Err(()) => return Ok(Err(None)),
            }
        }
        ColCodec::DictBitpack => {
            let decoded = match decode_dict_rows(
                column.payload,
                selected,
                directory.num_docs,
                column_index,
                checkpoint,
            )? {
                Ok(decoded) => decoded,
                Err(()) => return Ok(Err(None)),
            };
            stats.selected_dictionary_entries += decoded.selected_entries;
            stats.decompressed_buffer_bytes += decoded.decompressed_bytes;
            dict_ids.insert(column_index, decoded.ids);
            decoded.values
        }
        ColCodec::TypedInt if stored_row_select_enabled() => {
            // `XERJ_STORED_ROW_SELECT`: only the frames holding a selected
            // row are unpacked (`decode_typed_int_rows`), the same values.
            stats.encoded_rows_visited += selected.len();
            decode_typed_int_rows(column.payload, directory.num_docs, selected)?
        }
        ColCodec::TypedInt => {
            // Full materialisation is the honest shape here: the RawJson arm
            // decompresses the entire column anyway, and TYPED_INT columns
            // decode in ~value-count time with no zstd at all.
            stats.encoded_rows_visited += directory.num_docs;
            let all = decode_typed_int(column.payload, directory.num_docs)?;
            selected.iter().map(|&row| all[row].clone()).collect()
        }
        ColCodec::CopyOf => {
            let source_index = copy_of_source_index(column.payload)?;
            let Some(source) = directory.columns.get(source_index) else {
                return Err(StorageError::Other(anyhow::anyhow!(
                    "copy_of source index {source_index} out of range"
                )));
            };
            if source.codec == ColCodec::CopyOf {
                // The writer never chains references. A chained segment is
                // still decodable via the full fixpoint decoder, so decline
                // selective hydration rather than erroring — the caller uses
                // the compatibility path.
                return Ok(Err(Some(column.name.to_string())));
            }
            // The selected rows of the source column ARE this column's
            // rows — tail-call into its own decode (checkpoints and stats
            // included) and propagate the result shape unchanged.
            return select_column_rows(
                source_index,
                directory,
                selected,
                stats,
                dict_ids,
                checkpoint,
                in_memory,
            );
        }
        ColCodec::CrossDep => {
            let source_index = cross_dep_source_index(column.payload, directory.num_docs)?;
            let Some(source) = directory.columns.get(source_index) else {
                return Err(StorageError::Other(anyhow::anyhow!(
                    "cross_dep source index {source_index} out of range"
                )));
            };
            // Current encoder guarantees CROSS_DEP sources remain DICT. Valid
            // historical/handcrafted chains use the compatibility decoder.
            if source.codec != ColCodec::DictBitpack {
                // Unsupported is not an escape hatch for malformed storage:
                // validate this column's complete body and exception stream
                // before asking the caller to use the compatibility path.
                let placeholder_source_ids = vec![0; selected.len()];
                match decode_cross_dep_rows(
                    column,
                    selected,
                    &placeholder_source_ids,
                    directory.num_docs,
                    stats,
                    column_index,
                    checkpoint,
                )? {
                    Ok(_) => {}
                    Err(()) => return Ok(Err(None)),
                }
                return Ok(Err(Some(column.name.to_string())));
            }
            if let std::collections::hash_map::Entry::Vacant(entry) = dict_ids.entry(source_index) {
                let decoded = match decode_dict_rows(
                    source.payload,
                    selected,
                    directory.num_docs,
                    source_index,
                    checkpoint,
                )? {
                    Ok(decoded) => decoded,
                    Err(()) => return Ok(Err(None)),
                };
                stats.selected_dictionary_entries += decoded.selected_entries;
                stats.decompressed_buffer_bytes += decoded.decompressed_bytes;
                entry.insert(decoded.ids);
            }
            match decode_cross_dep_rows(
                column,
                selected,
                &dict_ids[&source_index],
                directory.num_docs,
                stats,
                column_index,
                checkpoint,
            )? {
                Ok(values) => values,
                Err(()) => return Ok(Err(None)),
            }
        }
    };
    stats.selected_json_values += values.len();
    Ok(Ok(values))
}

/// Reconstruct only selected ZBS2 documents without the canonical full decode.
pub fn decode_stored_v2_rows(
    bytes: &[u8],
    ordinals: &[usize],
) -> Result<StoredV2RowHydrationResult> {
    match decode_stored_v2_rows_projected_controlled(bytes, ordinals, None, |_| {
        std::ops::ControlFlow::Continue(())
    })? {
        StoredDecodeRun::Complete(result) => Ok(result),
        StoredDecodeRun::Cancelled => unreachable!("non-cancelling compatibility wrapper"),
    }
}

/// Reconstruct selected ZBS2 rows and only the requested top-level `_source`
/// fields. Identity and sequence metadata are always returned.
///
/// Missing requested fields remain absent, matching `_source` object
/// semantics. An empty projection returns identity-only rows.
pub fn decode_stored_v2_rows_projected(
    bytes: &[u8],
    ordinals: &[usize],
    source_fields: &[&str],
) -> Result<StoredV2RowHydrationResult> {
    match decode_stored_v2_rows_projected_controlled(bytes, ordinals, Some(source_fields), |_| {
        std::ops::ControlFlow::Continue(())
    })? {
        StoredDecodeRun::Complete(result) => Ok(result),
        StoredDecodeRun::Cancelled => unreachable!("non-cancelling compatibility wrapper"),
    }
}

pub fn decode_stored_v2_rows_controlled<F>(
    bytes: &[u8],
    ordinals: &[usize],
    checkpoint: F,
) -> Result<StoredDecodeRun<StoredV2RowHydrationResult>>
where
    F: FnMut(StoredDecodeCheckpoint) -> std::ops::ControlFlow<()>,
{
    decode_stored_v2_rows_projected_controlled(bytes, ordinals, None, checkpoint)
}

pub fn decode_stored_v2_rows_projected_controlled<F>(
    bytes: &[u8],
    ordinals: &[usize],
    source_fields: Option<&[&str]>,
    checkpoint: F,
) -> Result<StoredDecodeRun<StoredV2RowHydrationResult>>
where
    F: FnMut(StoredDecodeCheckpoint) -> std::ops::ControlFlow<()>,
{
    hydrate_v2_rows(bytes, ordinals, source_fields, false, true, checkpoint)
}

/// [`decode_stored_v2_rows_projected`] without the sequence column: every
/// row's `seq_no` is `Null` and `__seq_no` is not read. For callers that read
/// only `_id` and `_source` (the engine's cold top-k identity read under
/// `XERJ_STORED_ROW_SELECT`): a RAW_JSON `__seq_no` stream is otherwise parsed
/// to its end for one row (70 ms on a 1.17 M-row segment).
pub fn decode_stored_v2_rows_projected_no_seq(
    bytes: &[u8],
    ordinals: &[usize],
    source_fields: &[&str],
) -> Result<StoredV2RowHydrationResult> {
    match hydrate_v2_rows(bytes, ordinals, Some(source_fields), false, false, |_| {
        std::ops::ControlFlow::Continue(())
    })? {
        StoredDecodeRun::Complete(result) => Ok(result),
        StoredDecodeRun::Cancelled => unreachable!("non-cancelling compatibility wrapper"),
    }
}

/// [`decode_stored_v2_rows_projected_parallel`] without the sequence column
/// (see [`decode_stored_v2_rows_projected_no_seq`]).
pub fn decode_stored_v2_rows_projected_parallel_no_seq(
    bytes: &[u8],
    ordinals: &[usize],
    source_fields: Option<&[&str]>,
) -> Result<StoredV2RowHydrationResult> {
    match hydrate_v2_rows(bytes, ordinals, source_fields, true, false, |_| {
        std::ops::ControlFlow::Continue(())
    })? {
        StoredDecodeRun::Complete(result) => Ok(result),
        StoredDecodeRun::Cancelled => unreachable!("non-cancelling compatibility wrapper"),
    }
}

/// Like [`decode_stored_v2_rows_projected`] (`None` = every field), but
/// decodes the wanted columns concurrently on the rayon pool. Each column is
/// an independent stream, so a cold top-k hydration that must walk every
/// column of a segment for a handful of rows costs roughly its largest
/// column instead of the sum of all of them. Output is identical to the
/// serial decoder.
pub fn decode_stored_v2_rows_projected_parallel(
    bytes: &[u8],
    ordinals: &[usize],
    source_fields: Option<&[&str]>,
) -> Result<StoredV2RowHydrationResult> {
    match hydrate_v2_rows(bytes, ordinals, source_fields, true, true, |_| {
        std::ops::ControlFlow::Continue(())
    })? {
        StoredDecodeRun::Complete(result) => Ok(result),
        StoredDecodeRun::Cancelled => unreachable!("non-cancelling parallel wrapper"),
    }
}

/// One stored row, for a read by `_id` (GET) that must not decode the whole
/// section: the `_id` and every `_source` field of row `ordinal`, with
/// `seq_no` left `Null` (`__seq_no` is not read; a GET takes sequence numbers
/// from the version map). The engine reads a row this way when its hydration
/// budget cannot keep the segment's decoded stored section, so that a GET no
/// longer decodes that section on every call.
///
/// The row is the one [`decode_stored_v2_rows`] returns for `ordinal`, field
/// for field and in the same order (the `_source` of [`decode_stored`]'s
/// canonical row): every column goes through the same per-codec reader, and
/// validates the same way, as there, with three differences that only skip
/// work:
///
/// - a `COPY_OF` column takes the value its source column already produced
///   instead of decoding that source a second time;
/// - a `TYPED_INT` column unpacks only the 128-value frame that holds the row
///   ([`decode_typed_int_rows`], the reader `XERJ_STORED_ROW_SELECT` selects
///   for the other row reads; the same values as the full unpack);
/// - a ZBS3 section's present explicit nulls are collected for this row only,
///   not into a set over the whole segment.
///
/// Memory stays bounded by the row, not by the section: the columns decode
/// concurrently, each through its streaming reader (a `RAW_JSON` frame or a
/// dictionary is streamed through the zstd window, a `RAW_JSON_BLOCKS`
/// column decompresses only the block that holds the row, a dictionary
/// column its packed id array), and only the selected values are
/// materialised. The time is that of streaming the largest column frame
/// once, unless the section is row-blocked.
pub fn decode_stored_v2_row_for_get(
    bytes: &[u8],
    ordinal: usize,
) -> Result<StoredV2RowHydrationResult> {
    use rayon::prelude::*;
    let (row_present_nulls, bytes) = if bytes.len() >= 4 && &bytes[..4] == STORED_V3_MAGIC {
        split_stored_v3_for_row(bytes, ordinal)?
    } else {
        (Vec::new(), bytes)
    };
    if !is_columnar_stored_magic(bytes) {
        return Ok(StoredV2RowHydrationResult::NotV2);
    }
    let directory = parse_v2_directory(&bytes[4..])?;
    let id_index = directory
        .columns
        .iter()
        .position(|column| column.name == "__id")
        .ok_or_else(|| StorageError::Other(anyhow::anyhow!("v2 missing __id column")))?;
    let seq_index = directory
        .columns
        .iter()
        .position(|column| column.name == "__seq_no")
        .ok_or_else(|| StorageError::Other(anyhow::anyhow!("v2 missing __seq_no column")))?;
    if ordinal >= directory.num_docs {
        return Err(StorageError::Other(anyhow::anyhow!(
            "selected row outside stored section"
        )));
    }
    let selected = [ordinal];
    // Every column but the sequence numbers and the COPY_OF twins. Each
    // column is an independent stream; errors and declined shapes surface in
    // column order, as in the parallel hydration.
    let read: Vec<usize> = (0..directory.columns.len())
        .filter(|&index| index != seq_index && directory.columns[index].codec != ColCodec::CopyOf)
        .collect();
    let decoded: Vec<Result<(usize, ColumnRows, StoredV2RowHydrationStats)>> = read
        .par_iter()
        .map(|&index| {
            let mut local_stats = StoredV2RowHydrationStats::default();
            let column = &directory.columns[index];
            if column.codec == ColCodec::TypedInt {
                local_stats.encoded_rows_visited += selected.len();
                let values = decode_typed_int_rows(column.payload, directory.num_docs, &selected)?;
                return Ok((index, Ok(values), local_stats));
            }
            let mut local_dict_ids = HashMap::new();
            let mut no_cancel = |_: StoredDecodeCheckpoint| std::ops::ControlFlow::Continue(());
            let values = select_column_rows(
                index,
                &directory,
                &selected,
                &mut local_stats,
                &mut local_dict_ids,
                &mut no_cancel,
                false,
            )?;
            Ok((index, values, local_stats))
        })
        .collect();
    let mut stats = StoredV2RowHydrationStats::default();
    let mut values: Vec<Option<serde_json::Value>> =
        (0..directory.columns.len()).map(|_| None).collect();
    for result in decoded {
        let (index, rows, local_stats) = result?;
        stats.encoded_rows_visited += local_stats.encoded_rows_visited;
        stats.selected_json_values += local_stats.selected_json_values;
        stats.selected_dictionary_entries += local_stats.selected_dictionary_entries;
        stats.decompressed_buffer_bytes += local_stats.decompressed_buffer_bytes;
        match rows {
            Ok(mut rows) => values[index] = Some(single_row(&mut rows, index)?),
            Err(Some(column)) => {
                return Ok(StoredV2RowHydrationResult::UnsupportedDependencyShape { column });
            }
            Err(None) => return Err(row_read_cancelled()),
        }
    }
    // COPY_OF columns: the source column's value of this row. A source that
    // was not read above (`__seq_no`, or a chained COPY_OF, which the writer
    // never emits) goes through the regular reader, which declines a chain.
    for (index, column) in directory.columns.iter().enumerate() {
        if index == seq_index || column.codec != ColCodec::CopyOf {
            continue;
        }
        let source_index = copy_of_source_index(column.payload)?;
        if directory.columns.get(source_index).is_none() {
            return Err(StorageError::Other(anyhow::anyhow!(
                "copy_of source index {source_index} out of range"
            )));
        }
        let value = match values[source_index].clone() {
            Some(value) => {
                stats.selected_json_values += 1;
                value
            }
            None => {
                let mut dict_ids = HashMap::new();
                let mut no_cancel = |_: StoredDecodeCheckpoint| std::ops::ControlFlow::Continue(());
                match select_column_rows(
                    index,
                    &directory,
                    &selected,
                    &mut stats,
                    &mut dict_ids,
                    &mut no_cancel,
                    false,
                )? {
                    Ok(mut rows) => single_row(&mut rows, index)?,
                    Err(Some(column)) => {
                        return Ok(StoredV2RowHydrationResult::UnsupportedDependencyShape {
                            column,
                        });
                    }
                    Err(None) => return Err(row_read_cancelled()),
                }
            }
        };
        values[index] = Some(value);
    }
    // The assembly of `hydrate_v2_rows`: directory order, nulls dropped
    // unless they were present explicit nulls.
    let mut source = serde_json::Map::new();
    for (column_index, column) in directory.columns.iter().enumerate() {
        if column_index == id_index || column_index == seq_index {
            continue;
        }
        let Some(value) = values[column_index].take() else {
            continue;
        };
        if !value.is_null() {
            source.insert(column.name.to_string(), value);
            stats.output_values_cloned += 1;
        } else if row_present_nulls.contains(&(column_index as u32)) {
            source.insert(column.name.to_string(), serde_json::Value::Null);
            stats.output_values_cloned += 1;
        }
    }
    stats.output_values_cloned += 2;
    let id = values[id_index]
        .take()
        .ok_or_else(|| StorageError::Other(anyhow::anyhow!("v2 __id column not read")))?;
    Ok(StoredV2RowHydrationResult::Hydrated {
        rows: vec![StoredV2HydratedRow {
            ordinal,
            id,
            seq_no: serde_json::Value::Null,
            source,
        }],
        stats,
    })
}

/// One column's selected rows as `select_column_rows` returns them: the
/// values, or `Err(Some(column))` for a dependency shape it declines and
/// `Err(None)` for a cancellation.
type ColumnRows = std::result::Result<Vec<serde_json::Value>, Option<String>>;

/// The one value a single-row read of column `index` produced.
fn single_row(rows: &mut Vec<serde_json::Value>, index: usize) -> Result<serde_json::Value> {
    match (rows.pop(), rows.is_empty()) {
        (Some(value), true) => Ok(value),
        _ => Err(StorageError::Other(anyhow::anyhow!(
            "stored column {index} did not return exactly one row"
        ))),
    }
}

/// `select_column_rows` reports a cancellation only when its checkpoint asks
/// for one, and the GET row read passes a checkpoint that never does.
fn row_read_cancelled() -> StorageError {
    StorageError::Other(anyhow::anyhow!("stored row read cancelled"))
}

/// [`split_stored_v3`] for one row: the columns of row `ordinal` that hold a
/// present explicit null, and the inner ZBS2 blob. The framing is validated
/// as there; entries of other rows are skipped, not collected.
fn split_stored_v3_for_row(bytes: &[u8], ordinal: usize) -> Result<(Vec<u32>, &[u8])> {
    let count = bytes
        .get(4..8)
        .ok_or_else(|| StorageError::Other(anyhow::anyhow!("v3 present-null count truncated")))
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()) as usize)?;
    let entries_start = 8usize;
    let entries_len = count
        .checked_mul(8)
        .ok_or_else(|| StorageError::Other(anyhow::anyhow!("v3 present-null count overflow")))?;
    let inner_start = entries_start
        .checked_add(entries_len)
        .ok_or_else(|| StorageError::Other(anyhow::anyhow!("v3 present-null section overflow")))?;
    let entries = bytes
        .get(entries_start..inner_start)
        .ok_or_else(|| StorageError::Other(anyhow::anyhow!("v3 present-null section truncated")))?;
    let mut columns = Vec::new();
    for pair in entries.as_chunks::<8>().0 {
        let doc = u32::from_le_bytes(pair[0..4].try_into().unwrap());
        if doc as usize == ordinal {
            columns.push(u32::from_le_bytes(pair[4..8].try_into().unwrap()));
        }
    }
    let inner = bytes
        .get(inner_start..)
        .ok_or_else(|| StorageError::Other(anyhow::anyhow!("v3 inner blob missing")))?;
    if !is_columnar_stored_magic(inner) {
        return Err(StorageError::Other(anyhow::anyhow!(
            "v3 inner blob is not ZBS2"
        )));
    }
    Ok((columns, inner))
}

fn hydrate_v2_rows<F>(
    bytes: &[u8],
    ordinals: &[usize],
    source_fields: Option<&[&str]>,
    parallel: bool,
    with_seq: bool,
    mut checkpoint: F,
) -> Result<StoredDecodeRun<StoredV2RowHydrationResult>>
where
    F: FnMut(StoredDecodeCheckpoint) -> std::ops::ControlFlow<()>,
{
    // #415: a ZBS3 wrapper carries a sparse present-null sidecar around an
    // inner ZBS2 blob. Strip it so the hydration below runs on the ZBS2 body,
    // and re-insert the present explicit nulls for hydrated rows (see the
    // assembly loop). A plain ZBS2 segment yields an empty sidecar.
    let (present_nulls, bytes) = if bytes.len() >= 4 && &bytes[..4] == STORED_V3_MAGIC {
        split_stored_v3(bytes)?
    } else {
        (Vec::new(), bytes)
    };
    if !is_columnar_stored_magic(bytes) {
        return Ok(StoredDecodeRun::Complete(StoredV2RowHydrationResult::NotV2));
    }
    let present_null_set: HashSet<(u32, u32)> = present_nulls.iter().copied().collect();
    let directory = parse_v2_directory(&bytes[4..])?;
    let id_index = directory
        .columns
        .iter()
        .position(|column| column.name == "__id")
        .ok_or_else(|| StorageError::Other(anyhow::anyhow!("v2 missing __id column")))?;
    let seq_index = directory
        .columns
        .iter()
        .position(|column| column.name == "__seq_no")
        .ok_or_else(|| StorageError::Other(anyhow::anyhow!("v2 missing __seq_no column")))?;
    let mut selected = ordinals.to_vec();
    selected.sort_unstable();
    selected.dedup();
    if selected
        .iter()
        .any(|ordinal| *ordinal >= directory.num_docs)
    {
        return Err(StorageError::Other(anyhow::anyhow!(
            "selected row outside stored section"
        )));
    }
    if selected.is_empty() {
        return Ok(StoredDecodeRun::Complete(
            StoredV2RowHydrationResult::Hydrated {
                rows: Vec::new(),
                stats: StoredV2RowHydrationStats::default(),
            },
        ));
    }
    let mut stats = StoredV2RowHydrationStats::default();
    let requested: Option<HashSet<&str>> =
        source_fields.map(|fields| fields.iter().copied().collect());
    // #1122/#2433: a row whose derived semantic vectors were stripped gets
    // them rebuilt from its text when the projection wants them, so decode
    // every column for it and trim back to the projection afterwards.
    let wants_derived = {
        let mut targets = directory
            .columns
            .iter()
            .filter_map(|column| derived_vectors::metadata_target(column.name))
            .peekable();
        targets.peek().is_some()
            && match requested.as_ref() {
                None => true,
                Some(fields) => targets.any(|target| {
                    fields.contains(target)
                        || fields.contains(derived_vectors::chunks_name(target).as_str())
                }),
            }
    };
    let allowed: Option<HashSet<String>> = if wants_derived {
        requested
            .as_ref()
            .map(|fields| fields.iter().map(|field| field.to_string()).collect())
    } else {
        None
    };
    let requested = if wants_derived { None } else { requested };
    let mut columns: Vec<Option<Vec<serde_json::Value>>> =
        (0..directory.columns.len()).map(|_| None).collect();
    let mut dict_ids = HashMap::new();
    let wanted_columns: Vec<usize> = (0..directory.columns.len())
        .filter(|&index| {
            index == id_index
                || (with_seq && index == seq_index)
                || (index != seq_index
                    && requested
                        .as_ref()
                        .is_none_or(|fields| fields.contains(directory.columns[index].name)))
        })
        .collect();
    if parallel && wanted_columns.len() > 1 {
        use rayon::prelude::*;
        // Each task gets its own stats and dictionary-id scratch: a CROSS_DEP
        // column re-derives its DICT source ids locally, which is the same
        // work the serial path does once and shares.
        let decoded: Vec<_> = wanted_columns
            .par_iter()
            .map(|&index| {
                let mut local_stats = StoredV2RowHydrationStats::default();
                let mut local_dict_ids = HashMap::new();
                let mut noop = |_: StoredDecodeCheckpoint| std::ops::ControlFlow::Continue(());
                select_column_rows(
                    index,
                    &directory,
                    &selected,
                    &mut local_stats,
                    &mut local_dict_ids,
                    &mut noop,
                    true,
                )
                .map(|values| (index, values, local_stats))
            })
            .collect();
        for result in decoded {
            let (index, values, local_stats) = result?;
            stats.encoded_rows_visited += local_stats.encoded_rows_visited;
            stats.selected_json_values += local_stats.selected_json_values;
            stats.selected_dictionary_entries += local_stats.selected_dictionary_entries;
            stats.decompressed_buffer_bytes += local_stats.decompressed_buffer_bytes;
            match values {
                Ok(values) => columns[index] = Some(values),
                Err(Some(column)) => {
                    return Ok(StoredDecodeRun::Complete(
                        StoredV2RowHydrationResult::UnsupportedDependencyShape { column },
                    ));
                }
                Err(None) => return Ok(StoredDecodeRun::Cancelled),
            }
        }
    }
    for index in wanted_columns {
        if columns[index].is_some() {
            continue;
        }
        if decode_cancelled(&mut checkpoint, StoredDecodePhase::BeforeColumn, index, 0) {
            return Ok(StoredDecodeRun::Cancelled);
        }
        match select_column_rows(
            index,
            &directory,
            &selected,
            &mut stats,
            &mut dict_ids,
            &mut checkpoint,
            false,
        )? {
            Ok(values) => columns[index] = Some(values),
            Err(Some(column)) => {
                return Ok(StoredDecodeRun::Complete(
                    StoredV2RowHydrationResult::UnsupportedDependencyShape { column },
                ));
            }
            Err(None) => return Ok(StoredDecodeRun::Cancelled),
        }
        if decode_cancelled(
            &mut checkpoint,
            StoredDecodePhase::AfterColumn,
            index,
            directory.num_docs,
        ) {
            return Ok(StoredDecodeRun::Cancelled);
        }
    }
    let mut rows = Vec::with_capacity(selected.len());
    for (selected_index, ordinal) in selected.into_iter().enumerate() {
        let mut source = serde_json::Map::new();
        for (column_index, column) in directory.columns.iter().enumerate() {
            if column_index == id_index || column_index == seq_index {
                continue;
            }
            let Some(values) = columns[column_index].as_ref() else {
                continue;
            };
            let value = values[selected_index].clone();
            if !value.is_null() {
                source.insert(column.name.to_string(), value);
                stats.output_values_cloned += 1;
            } else if present_null_set.contains(&(ordinal as u32, column_index as u32)) {
                // Present explicit null (#415) — re-insert, unlike an absent field.
                source.insert(column.name.to_string(), serde_json::Value::Null);
                stats.output_values_cloned += 1;
            }
        }
        if wants_derived {
            derived_vectors::rehydrate_source_map(&mut source);
            if let Some(allowed) = allowed.as_ref() {
                source.retain(|name, _| allowed.contains(name));
            }
        }
        stats.output_values_cloned += 2;
        rows.push(StoredV2HydratedRow {
            ordinal,
            id: columns[id_index].as_ref().unwrap()[selected_index].clone(),
            seq_no: if with_seq {
                columns[seq_index].as_ref().unwrap()[selected_index].clone()
            } else {
                serde_json::Value::Null
            },
            source,
        });
    }
    Ok(StoredDecodeRun::Complete(
        StoredV2RowHydrationResult::Hydrated { rows, stats },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn assemble(num_docs: usize, columns: &[(&str, ColCodec, Vec<u8>)]) -> Vec<u8> {
        let mut out = STORED_V2_MAGIC.to_vec();
        out.write_u32::<LittleEndian>(num_docs as u32).unwrap();
        out.write_u32::<LittleEndian>(columns.len() as u32).unwrap();
        for (name, codec, payload) in columns {
            out.write_u16::<LittleEndian>(name.len() as u16).unwrap();
            out.extend_from_slice(name.as_bytes());
            out.push(*codec as u8);
            out.write_u32::<LittleEndian>(payload.len() as u32).unwrap();
            out.extend_from_slice(payload);
        }
        out
    }

    fn raw(values: &[serde_json::Value]) -> Vec<u8> {
        zstd::encode_all(
            Cursor::new(serde_json::to_vec(values).unwrap()),
            STORED_ZSTD_LEVEL,
        )
        .unwrap()
    }

    fn fixture(n: usize, body_codec: ColCodec) -> Vec<u8> {
        let ids: Vec<_> = (0..n).map(|i| json!(format!("doc-{i}"))).collect();
        let seq: Vec<_> = (0..n).map(|i| json!(i)).collect();
        let body: Vec<_> = (0..n)
            .map(|i| json!({"row": i, "text": format!("payload-{i}")}))
            .collect();
        let body_bytes = serde_json::to_vec(&body).unwrap();
        let body_payload = match body_codec {
            ColCodec::RawJson => zstd::encode_all(Cursor::new(body_bytes), 3).unwrap(),
            ColCodec::Lz4Json => lz4_flex::compress_prepend_size(&body_bytes),
            _ => unreachable!(),
        };
        assemble(
            n,
            &[
                ("__id", ColCodec::RawJson, raw(&ids)),
                ("__seq_no", ColCodec::RawJson, raw(&seq)),
                ("body", body_codec, body_payload),
            ],
        )
    }

    /// `*_no_seq` readers on a section whose `__seq_no` is a RAW_JSON stream
    /// (the layout of the 10 GB benchmark segments): ids and bodies equal the
    /// regular readers', `seq_no` is null, and the `__seq_no` stream is not
    /// read - the regular reader parses it to its end for any row
    /// (fail-before: `encoded_rows_visited` drops by exactly the column's rows).
    #[test]
    fn rows_without_seq_skip_the_seq_stream() {
        let n = 5_000;
        let blob = fixture(n, ColCodec::RawJson);
        let rows = [4_999usize, 0, 2_500, 17];
        let unpack = |r: StoredV2RowHydrationResult| match r {
            StoredV2RowHydrationResult::Hydrated { rows, stats } => (rows, stats),
            other => panic!("rows must hydrate: {other:?}"),
        };
        let (with, with_stats) =
            unpack(decode_stored_v2_rows_projected(&blob, &rows, &[]).unwrap());
        let (without, without_stats) =
            unpack(decode_stored_v2_rows_projected_no_seq(&blob, &rows, &[]).unwrap());
        assert_eq!(with.len(), 4);
        for (a, b) in with.iter().zip(&without) {
            assert_eq!((a.ordinal, &a.id, &a.source), (b.ordinal, &b.id, &b.source));
            assert_eq!(a.seq_no, json!(a.ordinal));
            assert!(b.seq_no.is_null());
        }
        assert_eq!(
            with_stats.encoded_rows_visited - without_stats.encoded_rows_visited,
            n,
            "the __seq_no stream is not read"
        );
        let (with, _) =
            unpack(decode_stored_v2_rows_projected_parallel(&blob, &rows, None).unwrap());
        let (without, without_stats) =
            unpack(decode_stored_v2_rows_projected_parallel_no_seq(&blob, &rows, None).unwrap());
        for (a, b) in with.iter().zip(&without) {
            assert_eq!((a.ordinal, &a.id, &a.source), (b.ordinal, &b.id, &b.source));
            assert!(b.source.contains_key("body"), "the body column is read");
        }
        assert_eq!(
            without_stats.encoded_rows_visited,
            2 * n,
            "__id and body only"
        );
    }

    /// #2422: `select_json_rows` reads its column through a buffer.
    /// `serde_json`'s reader pulls one byte per `read`, and over zstd's
    /// streaming decoder every `read` was a `ZSTD_decompressStream` call, so
    /// one selected row of a large column cost one zstd call per byte.
    #[test]
    fn selected_rows_read_the_column_in_large_chunks_not_byte_by_byte() {
        struct CountingReader<'a> {
            inner: &'a [u8],
            reads: &'a Cell<usize>,
        }
        impl std::io::Read for CountingReader<'_> {
            fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
                self.reads.set(self.reads.get() + 1);
                self.inner.read(buf)
            }
        }
        let rows: Vec<_> = (0..4096)
            .map(|i| json!({"row": i, "text": "x".repeat(200)}))
            .collect();
        let column = serde_json::to_vec(&rows).unwrap();
        let reads = Cell::new(0);
        let selected = select_json_rows(
            CountingReader {
                inner: &column,
                reads: &reads,
            },
            &[7, 4000],
            rows.len(),
            "test json",
            0,
            &mut |_| std::ops::ControlFlow::Continue(()),
        )
        .unwrap()
        .unwrap();
        assert_eq!(selected, vec![rows[7].clone(), rows[4000].clone()]);
        let chunk_reads = column.len().div_ceil(JSON_STREAM_READ_BUFFER) + 2;
        assert!(
            reads.get() <= chunk_reads,
            "{} reads for a {} byte column; a byte-at-a-time parse makes one per byte",
            reads.get(),
            column.len()
        );
    }

    #[test]
    fn raw_and_lz4_select_rows_and_reject_trailing_json() {
        for codec in [ColCodec::RawJson, ColCodec::Lz4Json] {
            let encoded = fixture(128, codec);
            let StoredV2RowHydrationResult::Hydrated { rows, stats } =
                decode_stored_v2_rows(&encoded, &[127, 0, 7, 7]).unwrap()
            else {
                panic!("expected hydration")
            };
            assert_eq!(
                rows.iter().map(|row| row.ordinal).collect::<Vec<_>>(),
                [0, 7, 127]
            );
            assert_eq!(rows[1].source["body"]["row"], 7);
            assert_eq!(stats.output_values_cloned, 9);
            if codec == ColCodec::Lz4Json {
                assert!(stats.decompressed_buffer_bytes > 0);
            }
        }

        let mut bad_values = serde_json::to_vec(&vec![json!("x"); 4]).unwrap();
        bad_values.extend_from_slice(b"[]");
        let bad = assemble(
            4,
            &[
                ("__id", ColCodec::RawJson, raw(&vec![json!("x"); 4])),
                ("__seq_no", ColCodec::RawJson, raw(&vec![json!(0); 4])),
                (
                    "bad",
                    ColCodec::Lz4Json,
                    lz4_flex::compress_prepend_size(&bad_values),
                ),
            ],
        );
        assert!(decode_stored_v2_rows(&bad, &[0]).is_err());
    }

    /// The parallel hydration parses each JSON column as an in-memory slice;
    /// its rows must equal the streaming path's, and it must still reject a
    /// column with trailing JSON or a row count other than the directory's.
    #[test]
    fn parallel_hydration_matches_streaming_and_validates_columns() {
        for codec in [ColCodec::RawJson, ColCodec::Lz4Json] {
            let encoded = fixture(300, codec);
            let ordinals = [299, 0, 7, 7, 128];
            for fields in [None, Some(&["body"][..]), Some(&[][..])] {
                let serial = match fields {
                    Some(fields) => decode_stored_v2_rows_projected(&encoded, &ordinals, fields),
                    None => decode_stored_v2_rows(&encoded, &ordinals),
                }
                .unwrap();
                let parallel =
                    decode_stored_v2_rows_projected_parallel(&encoded, &ordinals, fields).unwrap();
                let (
                    StoredV2RowHydrationResult::Hydrated { rows: a, .. },
                    StoredV2RowHydrationResult::Hydrated { rows: b, .. },
                ) = (serial, parallel)
                else {
                    panic!("expected hydration")
                };
                assert_eq!(a.len(), 4);
                assert_eq!(a.len(), b.len());
                for (a, b) in a.iter().zip(&b) {
                    assert_eq!(
                        (a.ordinal, &a.id, &a.seq_no, &a.source),
                        (b.ordinal, &b.id, &b.seq_no, &b.source)
                    );
                }
            }
        }
        for tail in [&b"[]"[..], &b""[..]] {
            let mut values = serde_json::to_vec(&vec![json!("x"); 4]).unwrap();
            values.extend_from_slice(tail);
            let rows = if tail.is_empty() { 3 } else { 4 };
            let bad = assemble(
                rows,
                &[
                    ("__id", ColCodec::RawJson, raw(&vec![json!("x"); rows])),
                    ("__seq_no", ColCodec::RawJson, raw(&vec![json!(0); rows])),
                    (
                        "bad",
                        ColCodec::RawJson,
                        zstd::encode_all(Cursor::new(values), 3).unwrap(),
                    ),
                    ("other", ColCodec::RawJson, raw(&vec![json!(1); rows])),
                ],
            );
            assert!(decode_stored_v2_rows_projected_parallel(&bad, &[0], None).is_err());
        }
    }

    /// The identity-only id list equals the `_id`s of the full decode.
    #[test]
    fn identity_ids_match_the_full_decode() {
        for codec in [ColCodec::RawJson, ColCodec::Lz4Json] {
            let encoded = fixture(200, codec);
            let ids = decode_stored_v2_ids(&encoded).unwrap().unwrap();
            let full: Vec<serde_json::Value> =
                serde_json::from_slice(&super::super::decode_stored(&encoded).unwrap()).unwrap();
            assert_eq!(ids.len(), full.len());
            for (id, doc) in ids.iter().zip(&full) {
                assert_eq!(id, &doc["_id"]);
            }
        }
        assert!(decode_stored_v2_ids(b"not columnar").unwrap().is_none());
    }

    #[test]
    fn dictionary_nulls_boundaries_and_corruption_are_validated() {
        let values = vec![json!("a"), json!("b"), json!("c")];
        let ids = vec![0, 1, 3, 2, 0, 3, 1, 2];
        let encoded = assemble(
            ids.len(),
            &[
                (
                    "__id",
                    ColCodec::DictBitpack,
                    encode_dict_bitpack(&values, &ids, STORED_ZSTD_LEVEL),
                ),
                (
                    "__seq_no",
                    ColCodec::Constant,
                    serde_json::to_vec(&json!(9)).unwrap(),
                ),
            ],
        );
        let StoredV2RowHydrationResult::Hydrated { rows, stats } =
            decode_stored_v2_rows(&encoded, &[0, 2, 7]).unwrap()
        else {
            panic!("expected hydration")
        };
        assert_eq!(rows[0].id, "a");
        assert!(rows[1].id.is_null());
        assert_eq!(rows[2].id, "c");
        assert_eq!(stats.selected_dictionary_entries, 2);

        for width in 1..=32 {
            let max = if width == 32 {
                u32::MAX
            } else {
                (1u32 << width) - 1
            };
            let packed = bitpack_u32(&[0, max, max / 2], width);
            assert_eq!(unpack_id(&packed, width, 1), Some(max));
            assert_eq!(unpack_id(&packed, width, 2), Some(max / 2));
            assert_eq!(unpack_id(&packed[..packed.len() - 1], width, 2), None);
        }

        let mut corrupt = encode_dict_bitpack(&[json!("only")], &[0, 0, 0], STORED_ZSTD_LEVEL);
        // Rebuild the packed stream with id=3, which is outside {entry,null}.
        let dict_len = u32::from_le_bytes(corrupt[5..9].try_into().unwrap()) as usize;
        let packed_offset = 9 + dict_len + 8;
        let bad_packed = zstd::encode_all(Cursor::new(bitpack_u32(&[0, 3, 0], 2)), 3).unwrap();
        corrupt.truncate(packed_offset);
        corrupt.extend_from_slice(&bad_packed);
        corrupt[4] = 2;
        let length_offset = 9 + dict_len + 4;
        corrupt[length_offset..length_offset + 4]
            .copy_from_slice(&(bad_packed.len() as u32).to_le_bytes());
        let bad = assemble(
            3,
            &[
                ("__id", ColCodec::DictBitpack, corrupt),
                (
                    "__seq_no",
                    ColCodec::Constant,
                    serde_json::to_vec(&json!(0)).unwrap(),
                ),
            ],
        );
        assert!(decode_stored_v2_rows(&bad, &[0]).is_err());

        let mut padding = encode_dict_bitpack(&[json!("only")], &[0, 0, 0], STORED_ZSTD_LEVEL);
        let dict_len = u32::from_le_bytes(padding[5..9].try_into().unwrap()) as usize;
        let packed_length_offset = 9 + dict_len + 4;
        let packed_offset = packed_length_offset + 4;
        let mut unpacked = zstd::decode_all(&padding[packed_offset..]).unwrap();
        unpacked[0] |= 0b1000_0000;
        let repacked = zstd::encode_all(Cursor::new(unpacked), 3).unwrap();
        padding.truncate(packed_offset);
        padding.extend_from_slice(&repacked);
        padding[packed_length_offset..packed_offset]
            .copy_from_slice(&(repacked.len() as u32).to_le_bytes());
        let bad_padding = assemble(
            3,
            &[
                ("__id", ColCodec::DictBitpack, padding),
                (
                    "__seq_no",
                    ColCodec::Constant,
                    serde_json::to_vec(&json!(0)).unwrap(),
                ),
            ],
        );
        assert!(decode_stored_v2_rows(&bad_padding, &[0]).is_err());
    }

    #[test]
    fn selection_validation_empty_fast_path_and_fixed_r_work() {
        assert!(matches!(
            decode_stored_v2_rows(b"plain", &[0]).unwrap(),
            StoredV2RowHydrationResult::NotV2
        ));
        let encoded = fixture(128, ColCodec::RawJson);
        assert!(decode_stored_v2_rows(&encoded, &[128]).is_err());
        let StoredV2RowHydrationResult::Hydrated { rows, stats } =
            decode_stored_v2_rows(&encoded, &[]).unwrap()
        else {
            panic!("expected hydration")
        };
        assert!(rows.is_empty());
        assert_eq!(stats, StoredV2RowHydrationStats::default());

        let small = decode_stored_v2_rows(&fixture(128, ColCodec::RawJson), &[1, 9, 17]).unwrap();
        let large = decode_stored_v2_rows(&fixture(1024, ColCodec::RawJson), &[1, 9, 17]).unwrap();
        let selected = |result| match result {
            StoredV2RowHydrationResult::Hydrated { rows, stats } => (
                rows.len(),
                stats.selected_json_values,
                stats.output_values_cloned,
            ),
            _ => panic!("expected hydration"),
        };
        assert_eq!(selected(small), selected(large));
    }

    #[test]
    fn controlled_hydration_has_deterministic_order_and_distinct_cancellation() {
        let encoded = fixture(256, ColCodec::RawJson);
        let mut observed = Vec::new();
        let run = decode_stored_v2_rows_controlled(&encoded, &[0, 7], |point| {
            observed.push(point);
            if point.phase == StoredDecodePhase::Rows && point.rows_processed == 128 {
                std::ops::ControlFlow::Break(())
            } else {
                std::ops::ControlFlow::Continue(())
            }
        })
        .unwrap();
        assert_eq!(run, StoredDecodeRun::Cancelled);
        assert_eq!(
            observed[0],
            StoredDecodeCheckpoint {
                phase: StoredDecodePhase::BeforeColumn,
                column_index: 0,
                rows_processed: 0,
            }
        );
        assert_eq!(
            observed[1],
            StoredDecodeCheckpoint {
                phase: StoredDecodePhase::Rows,
                column_index: 0,
                rows_processed: 128,
            }
        );

        let complete = decode_stored_v2_rows_controlled(&encoded, &[0, 7], |_| {
            std::ops::ControlFlow::Continue(())
        })
        .unwrap();
        assert_eq!(
            complete,
            StoredDecodeRun::Complete(decode_stored_v2_rows(&encoded, &[0, 7]).unwrap())
        );
    }

    #[test]
    fn controlled_constant_hydration_is_fallible_cancellable_and_preserves_parity() {
        let encoded = assemble(
            256,
            &[
                (
                    "__id",
                    ColCodec::Constant,
                    serde_json::to_vec(&json!("same-id")).unwrap(),
                ),
                (
                    "__seq_no",
                    ColCodec::Constant,
                    serde_json::to_vec(&json!(7)).unwrap(),
                ),
                (
                    "body",
                    ColCodec::Constant,
                    serde_json::to_vec(&json!({"kind": "constant"})).unwrap(),
                ),
            ],
        );
        let selected: Vec<_> = (0..256).collect();
        let mut observed = Vec::new();
        let cancelled = decode_stored_v2_rows_controlled(&encoded, &selected, |point| {
            observed.push(point);
            if point.phase == StoredDecodePhase::Rows && point.rows_processed == 128 {
                std::ops::ControlFlow::Break(())
            } else {
                std::ops::ControlFlow::Continue(())
            }
        })
        .unwrap();
        assert_eq!(cancelled, StoredDecodeRun::Cancelled);
        assert_eq!(
            observed[1],
            StoredDecodeCheckpoint {
                phase: StoredDecodePhase::Rows,
                column_index: 0,
                rows_processed: 128,
            }
        );

        let complete = decode_stored_v2_rows_controlled(&encoded, &[255, 0, 7, 7], |_| {
            std::ops::ControlFlow::Continue(())
        })
        .unwrap();
        assert_eq!(
            complete,
            StoredDecodeRun::Complete(decode_stored_v2_rows(&encoded, &[255, 0, 7, 7]).unwrap())
        );
        let StoredDecodeRun::Complete(StoredV2RowHydrationResult::Hydrated { rows, .. }) = complete
        else {
            panic!("expected completed constant hydration")
        };
        assert_eq!(
            rows.iter().map(|row| row.ordinal).collect::<Vec<_>>(),
            [0, 7, 255]
        );
        assert!(rows
            .iter()
            .all(|row| row.source["body"] == json!({"kind": "constant"})));

        let reserve_error =
            repeat_constant_rows_controlled(json!(null), usize::MAX, 0, &mut |_| {
                std::ops::ControlFlow::Continue(())
            })
            .unwrap_err();
        let expected = format!("cannot reserve {} selected constant values", usize::MAX);
        assert!(
            reserve_error.to_string().contains(&expected),
            "allocation failure must identify the requested output count: {reserve_error}"
        );
    }

    #[test]
    fn controlled_hydration_cancellation_precedes_unvisited_corruption() {
        let mut encoded = fixture(256, ColCodec::RawJson);
        let (offset, payload_len) = {
            let directory = parse_v2_directory(&encoded[4..]).unwrap();
            let last = directory.columns.last().unwrap();
            (
                last.payload.as_ptr() as usize - encoded.as_ptr() as usize,
                last.payload.len(),
            )
        };
        encoded[offset + payload_len / 2] ^= 0xff;

        assert_eq!(
            decode_stored_v2_rows_controlled(&encoded, &[0], |_| {
                std::ops::ControlFlow::Break(())
            })
            .unwrap(),
            StoredDecodeRun::Cancelled
        );
        assert!(decode_stored_v2_rows_controlled(&encoded, &[0], |_| {
            std::ops::ControlFlow::Continue(())
        })
        .is_err());
    }

    #[test]
    fn controlled_hydration_stops_before_malformed_row_129() {
        let mut json = String::from("[");
        for row in 0..128 {
            if row != 0 {
                json.push(',');
            }
            json.push_str(&row.to_string());
        }
        json.push_str(",row_129_is_invalid]");
        let malformed = zstd::encode_all(Cursor::new(json.as_bytes()), STORED_ZSTD_LEVEL).unwrap();
        let encoded = assemble(
            256,
            &[
                (
                    "__id",
                    ColCodec::Constant,
                    serde_json::to_vec(&json!("id")).unwrap(),
                ),
                (
                    "__seq_no",
                    ColCodec::Constant,
                    serde_json::to_vec(&json!(1)).unwrap(),
                ),
                ("bad", ColCodec::RawJson, malformed),
            ],
        );
        assert_eq!(
            decode_stored_v2_rows_controlled(&encoded, &[0], |point| {
                if point.column_index == 2
                    && point.phase == StoredDecodePhase::Rows
                    && point.rows_processed == 128
                {
                    std::ops::ControlFlow::Break(())
                } else {
                    std::ops::ControlFlow::Continue(())
                }
            })
            .unwrap(),
            StoredDecodeRun::Cancelled
        );
        assert!(
            decode_stored_v2_rows_controlled(&encoded, &[0], |_| {
                std::ops::ControlFlow::Continue(())
            })
            .is_err(),
            "without cancellation the parser must consume malformed row 129"
        );
    }

    #[test]
    fn malformed_framing_and_unsupported_dependency_are_explicit() {
        let encoded = fixture(4, ColCodec::RawJson);
        assert!(decode_stored_v2_rows(&encoded[..encoded.len() - 1], &[0]).is_err());

        let valid_cross_dep = |source_index: u32| {
            let mut bytes = vec![0];
            bytes.extend_from_slice(&source_index.to_le_bytes());
            bytes.extend_from_slice(&0u32.to_le_bytes());
            bytes.extend_from_slice(&0u32.to_le_bytes());
            bytes
        };
        let unsupported = assemble(
            1,
            &[
                ("__id", ColCodec::RawJson, raw(&[json!("a")])),
                ("__seq_no", ColCodec::RawJson, raw(&[json!(0)])),
                ("dep", ColCodec::CrossDep, valid_cross_dep(0)),
            ],
        );
        assert!(matches!(
            decode_stored_v2_rows(&unsupported, &[0]).unwrap(),
            StoredV2RowHydrationResult::UnsupportedDependencyShape { .. }
        ));

        let malformed = assemble(
            1,
            &[
                ("__id", ColCodec::RawJson, raw(&[json!("a")])),
                ("__seq_no", ColCodec::RawJson, raw(&[json!(0)])),
                ("dep", ColCodec::CrossDep, vec![0, 0, 0, 0, 0]),
            ],
        );
        assert!(decode_stored_v2_rows(&malformed, &[0]).is_err());
    }

    #[test]
    fn selected_cross_dep_exceptions_and_nulls_match_full_semantics() {
        let mut dependent = vec![0];
        dependent.extend_from_slice(&0u32.to_le_bytes()); // source column
        dependent.extend_from_slice(&2u32.to_le_bytes());
        dependent.extend_from_slice(&10i64.to_le_bytes());
        dependent.extend_from_slice(&20i64.to_le_bytes());
        dependent.extend_from_slice(&2u32.to_le_bytes());
        write_varint(&mut dependent, 1);
        write_zigzag_i64(&mut dependent, i64::MIN + 1);
        write_varint(&mut dependent, 2);
        write_zigzag_i64(&mut dependent, 99);
        let encoded = assemble(
            4,
            &[
                (
                    "__id",
                    ColCodec::DictBitpack,
                    encode_dict_bitpack(
                        &[json!("a"), json!("b")],
                        &[0, 1, 0, 1],
                        STORED_ZSTD_LEVEL,
                    ),
                ),
                ("__seq_no", ColCodec::CrossDep, dependent),
            ],
        );
        let StoredV2RowHydrationResult::Hydrated { rows, .. } =
            decode_stored_v2_rows(&encoded, &[0, 1, 2, 3]).unwrap()
        else {
            panic!("expected hydration")
        };
        let canonical: Vec<serde_json::Value> =
            serde_json::from_slice(&decode_stored(&encoded).unwrap()).unwrap();
        for (row, full) in rows.iter().zip(&canonical) {
            assert_eq!(row.id, full["_id"]);
            assert_eq!(row.seq_no, full["_seq_no"]);
            assert_eq!(
                serde_json::Value::Object(row.source.clone()),
                full["_source"]
            );
        }
        assert_eq!(rows[0].seq_no, 10);
        assert!(rows[1].seq_no.is_null());
        assert_eq!(rows[2].seq_no, 10);
        assert_eq!(rows[3].seq_no, 99);
    }

    #[test]
    fn adaptive_encoder_rows_match_canonical_decode_across_source_shapes() {
        let sources: Vec<_> = (0..256)
            .map(|i| {
                json!({
                    "company": if i % 3 == 0 { "Acme" } else { "Globex" },
                    "quarter": (i % 4) + 1,
                    "constant": true,
                    "nested": {"amount": i as f64 / 7.0, "tags": ["finance", i]},
                    "nullable": if i % 11 == 0 { serde_json::Value::Null } else { json!(i % 5) }
                })
            })
            .collect();
        let ids: Vec<_> = (0..256).map(|i| format!("doc-{i}")).collect();
        let docs: Vec<_> = (0..256)
            .map(|i| (ids[i].as_str(), i as u64, &sources[i]))
            .collect();
        let encoded = encode_stored_v2_from_values_nojson(&docs);
        // The `nullable` field is a present explicit null on some rows, so the
        // segment is wrapped as ZBS3 (#415); hydration must still match canonical.
        assert_eq!(&encoded[..4], STORED_V3_MAGIC);
        let full: Vec<serde_json::Value> =
            serde_json::from_slice(&decode_stored(&encoded).unwrap()).unwrap();
        let selected = [0, 7, 42, 128, 255];
        let StoredV2RowHydrationResult::Hydrated { rows, .. } =
            decode_stored_v2_rows(&encoded, &selected).unwrap()
        else {
            panic!("current encoder must have a selectively supported dependency shape")
        };
        for (row, ordinal) in rows.iter().zip(selected) {
            assert_eq!(row.id, full[ordinal]["_id"]);
            assert_eq!(row.seq_no, full[ordinal]["_seq_no"]);
            assert_eq!(
                serde_json::Value::Object(row.source.clone()),
                full[ordinal]["_source"]
            );
        }
    }

    #[test]
    fn row_projection_preserves_sparse_parity_and_skips_unrequested_payloads() {
        let count = 128usize;
        let ids: Vec<_> = (0..count).map(|row| json!(format!("id-{row}"))).collect();
        let seqs: Vec<_> = (0..count).map(|row| json!(row)).collect();
        let wanted: Vec<_> = (0..count)
            .map(|row| {
                if row % 3 == 0 {
                    json!(format!("quarter-{row}"))
                } else {
                    serde_json::Value::Null
                }
            })
            .collect();
        let encoded = assemble(
            count,
            &[
                ("__id", ColCodec::RawJson, raw(&ids)),
                ("__seq_no", ColCodec::RawJson, raw(&seqs)),
                ("wanted", ColCodec::RawJson, raw(&wanted)),
                // Valid framing, deliberately corrupt payload. A projected
                // read must not touch an unrelated column.
                ("unrelated", ColCodec::RawJson, b"not-zstd".to_vec()),
            ],
        );
        let selected = [0, 1, 42, 127];
        let StoredV2RowHydrationResult::Hydrated { rows, stats } =
            decode_stored_v2_rows_projected(&encoded, &selected, &["wanted"]).unwrap()
        else {
            panic!("expected projected hydration")
        };
        assert_eq!(rows.len(), selected.len());
        for (row, ordinal) in rows.iter().zip(selected) {
            assert_eq!(row.id, ids[ordinal]);
            assert_eq!(row.seq_no, seqs[ordinal]);
            if wanted[ordinal].is_null() {
                assert!(!row.source.contains_key("wanted"));
            } else {
                assert_eq!(row.source["wanted"], wanted[ordinal]);
            }
            assert!(!row.source.contains_key("unrelated"));
        }
        assert_eq!(stats.selected_json_values, selected.len() * 3);
        assert!(decode_stored_v2_rows(&encoded, &selected).is_err());

        let StoredV2RowHydrationResult::Hydrated { rows, .. } =
            decode_stored_v2_rows_projected(&encoded, &[7], &[]).unwrap()
        else {
            panic!("expected identity-only hydration")
        };
        assert!(rows[0].source.is_empty());
    }

    /// Documents whose adaptive-encoder section carries dictionary,
    /// typed-int, constant, COPY_OF and JSON columns and present explicit
    /// nulls (ZBS3), with floats whose text form is not trivial.
    fn get_row_fixture(n: usize) -> (Vec<String>, Vec<serde_json::Value>) {
        let ids: Vec<String> = (0..n).map(|i| format!("doc-{i}")).collect();
        let sources = (0..n)
            .map(|i| {
                let mut source = json!({
                    "kind": (["alpha", "beta", "gamma"][i % 3]),
                    "n": i as i64 - 1_000,
                    "schema": 7,
                    "echo": format!("doc-{i}"),
                    "ratio": i as f64 / 7.0,
                    "body": {
                        "text": format!(
                            "row {i}: \"quoted\" back\\slash caf\u{e9} \u{1F600} tab\t {}",
                            "filler ".repeat(i % 17)
                        ),
                        "tags": ["t", i, null, true],
                        "deep": {"x": [1.5, -0.0, 1e23, 5e-324, f64::MAX]},
                    },
                });
                if i.is_multiple_of(7) {
                    source["maybe"] = serde_json::Value::Null;
                } else if !i.is_multiple_of(5) {
                    source["maybe"] = json!(i % 11);
                }
                source
            })
            .collect();
        (ids, sources)
    }

    /// The GET row read gives each row's `_id` and the `_source` of the
    /// canonical full decode, the same JSON text once serialised (field order
    /// included: `Value` equality ignores a map's key order), on plain and
    /// row-blocked sections; its values are those of the regular row reader,
    /// its `seq_no` is null, and on a row-blocked section it walks fewer rows
    /// (one block per blocked column).
    #[test]
    fn get_row_read_matches_the_canonical_rows_in_both_layouts() {
        const N: usize = 3_000;
        let (ids, sources) = get_row_fixture(N);
        let docs: Vec<_> = (0..N)
            .map(|i| (ids[i].as_str(), i as u64 + 1, &sources[i]))
            .collect();
        let rows_to_read: Vec<usize> = (0..N)
            .filter(|i| i % 5 == 0 || i % 7 == 0 || *i < 3 || *i == N - 1)
            .collect();
        let mut visited = Vec::new();
        for blocked in [false, true] {
            ROW_BLOCKS_TEST_OVERRIDE.with(|cell| cell.set(Some(blocked)));
            let encoded = encode_stored_v2_from_values_nojson(&docs);
            ROW_BLOCKS_TEST_OVERRIDE.with(|cell| cell.set(None));
            assert_eq!(&encoded[..4], STORED_V3_MAGIC, "explicit nulls: ZBS3");
            assert_eq!(stored_v2_has_row_blocks(&encoded), blocked);
            let canonical: Vec<serde_json::Value> =
                serde_json::from_slice(&decode_stored(&encoded).unwrap()).unwrap();
            let mut rows_visited = 0;
            for &row in &rows_to_read {
                let StoredV2RowHydrationResult::Hydrated { rows, stats } =
                    decode_stored_v2_row_for_get(&encoded, row).unwrap()
                else {
                    panic!("row {row}: expected a hydrated row");
                };
                rows_visited += stats.encoded_rows_visited;
                let [got] = rows.as_slice() else {
                    panic!("row {row}: expected exactly one row");
                };
                assert_eq!(got.ordinal, row);
                assert_eq!(got.id, canonical[row]["_id"]);
                assert!(got.seq_no.is_null());
                let got_source = serde_json::Value::Object(got.source.clone());
                // The engine parses the serialised row back, as the slice
                // path parses its slice of the canonical text.
                let reparsed: serde_json::Value =
                    serde_json::from_slice(&serde_json::to_vec(&got_source).unwrap()).unwrap();
                assert_eq!(
                    serde_json::to_string(&reparsed).unwrap(),
                    serde_json::to_string(&canonical[row]["_source"]).unwrap(),
                    "row {row}, row blocks {blocked}"
                );
                let StoredV2RowHydrationResult::Hydrated { rows: regular, .. } =
                    decode_stored_v2_rows(&encoded, &[row]).unwrap()
                else {
                    panic!("row {row}: the regular reader declined");
                };
                assert_eq!(
                    serde_json::to_string(&got_source).unwrap(),
                    serde_json::to_string(&serde_json::Value::Object(regular[0].source.clone()))
                        .unwrap(),
                    "row {row}, row blocks {blocked}: regular reader"
                );
            }
            visited.push(rows_visited);
        }
        assert!(
            visited[1] < visited[0],
            "rows walked: plain {}, row-blocked {}",
            visited[0],
            visited[1]
        );
    }

    /// COPY_OF columns that echo the identity columns: a twin of `__id` takes
    /// the id the read already has, a twin of `__seq_no` (which a GET does
    /// not otherwise read) goes through the regular reader; a twin of a later
    /// twin is declined, as the regular reader declines it. Without a twin,
    /// `__seq_no` is never read.
    #[test]
    fn get_row_read_resolves_identity_twins_and_declines_forward_chains() {
        let n = 200usize;
        let ids: Vec<_> = (0..n).map(|i| json!(format!("id-{i}"))).collect();
        let seqs: Vec<_> = (0..n).map(|i| json!(i * 3)).collect();
        let copy = |source: u32| source.to_le_bytes().to_vec();
        let twins = assemble(
            n,
            &[
                ("__id", ColCodec::RawJson, raw(&ids)),
                ("__seq_no", ColCodec::RawJson, raw(&seqs)),
                ("id_echo", ColCodec::CopyOf, copy(0)),
                ("seq_echo", ColCodec::CopyOf, copy(1)),
            ],
        );
        let canonical: Vec<serde_json::Value> =
            serde_json::from_slice(&decode_stored(&twins).unwrap()).unwrap();
        for row in [0, 1, 99, n - 1] {
            let StoredV2RowHydrationResult::Hydrated { rows, .. } =
                decode_stored_v2_row_for_get(&twins, row).unwrap()
            else {
                panic!("row {row}: expected a hydrated row");
            };
            assert_eq!(rows[0].id, ids[row]);
            assert_eq!(
                serde_json::to_string(&serde_json::Value::Object(rows[0].source.clone())).unwrap(),
                serde_json::to_string(&canonical[row]["_source"]).unwrap()
            );
        }
        let corrupt_seq = assemble(
            n,
            &[
                ("__id", ColCodec::RawJson, raw(&ids)),
                ("__seq_no", ColCodec::RawJson, b"not-zstd".to_vec()),
                ("id_echo", ColCodec::CopyOf, copy(0)),
            ],
        );
        let StoredV2RowHydrationResult::Hydrated { rows, .. } =
            decode_stored_v2_row_for_get(&corrupt_seq, 5).unwrap()
        else {
            panic!("a GET row read does not read __seq_no");
        };
        assert_eq!(rows[0].source["id_echo"], ids[5]);
        let forward_chain = assemble(
            n,
            &[
                ("__id", ColCodec::RawJson, raw(&ids)),
                ("__seq_no", ColCodec::RawJson, raw(&seqs)),
                ("b", ColCodec::CopyOf, copy(3)),
                ("a", ColCodec::CopyOf, copy(0)),
            ],
        );
        let declined = StoredV2RowHydrationResult::UnsupportedDependencyShape {
            column: "b".to_string(),
        };
        assert_eq!(
            decode_stored_v2_rows(&forward_chain, &[3]).unwrap(),
            declined
        );
        assert_eq!(
            decode_stored_v2_row_for_get(&forward_chain, 3).unwrap(),
            declined
        );
    }

    /// A legacy (non-columnar) section is declined, a row outside the section
    /// is an error.
    #[test]
    fn get_row_read_declines_legacy_sections_and_rejects_rows_outside() {
        let legacy = encode_stored_lz4(br#"[{"_id":"a","_seq_no":1,"_source":{"x":1}}]"#);
        assert_eq!(
            decode_stored_v2_row_for_get(&legacy, 0).unwrap(),
            StoredV2RowHydrationResult::NotV2
        );
        let encoded = fixture(130, ColCodec::RawJson);
        assert!(decode_stored_v2_row_for_get(&encoded, 130).is_err());
        let StoredV2RowHydrationResult::Hydrated { rows, .. } =
            decode_stored_v2_row_for_get(&encoded, 129).unwrap()
        else {
            panic!("expected a hydrated row");
        };
        assert_eq!(rows[0].source["body"]["row"], 129);
    }
}
