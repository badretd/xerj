//! Scans a line-per-record file and writes its directory.
use crate::bits::{bucket_bitvector, int_array, SAMPLE, SAMPLE_SHIFT};
use crate::format::{
    encode, mtime_ns, source_identity, write_atomic, Header, Section, F_ARTICLES, F_ENDS_NL,
    F_MTIME, IDENT_BYTES, VERSION,
};
use crate::read::jsonl_id_key;
use crate::{Error, Result};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// How records group into articles.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ArticleKey {
    /// `{"id": "<article>#<passage>"`, the bytes before `#`.
    #[default]
    JsonlId,
    /// Passage level only.
    None,
}

#[derive(Debug, Clone, Copy)]
pub struct BuildOptions {
    pub page_size: u32,
    pub article_key: ArticleKey,
}

impl Default for BuildOptions {
    fn default() -> Self {
        Self {
            page_size: 4096,
            article_key: ArticleKey::JsonlId,
        }
    }
}

/// What a build produced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BuildInfo {
    pub passages: u64,
    pub pages: u64,
    pub articles: u64,
    pub file_size: u64,
    pub mtime_ns: i64,
    pub page_size: u32,
    pub bytes: u64,
    pub sections: BTreeMap<String, u64>,
}

/// Builds the directory of `source` at `out`. A crash leaves either the old directory or the whole
/// new one. Fails if the file changes during the scan.
pub fn build(source: &Path, out: &Path, opts: BuildOptions) -> Result<BuildInfo> {
    let page_size = opts.page_size;
    if page_size < 16 || !page_size.is_power_of_two() {
        return Err(Error::Invalid(
            "page_size must be a power of two >= 16".into(),
        ));
    }
    let shift = page_size.trailing_zeros();
    let with_articles = opts.article_key == ArticleKey::JsonlId;

    let f = File::open(source)?;
    let md0 = f.metadata()?;
    let ident = source_identity(&f, IDENT_BYTES)?;
    let size = ident.size;
    let pages = (size + page_size as u64 - 1) >> shift;
    let mut pcount = vec![0u32; pages as usize];
    let mut acount = vec![0u32; if with_articles { pages as usize } else { 0 }];
    let mut astarts: Vec<u64> = Vec::new();
    let (mut n, mut pos, mut ends_nl) = (0u64, 0u64, false);
    {
        let mut r = BufReader::with_capacity(1 << 24, &f);
        let mut line = Vec::new();
        let mut prev: Option<Vec<u8>> = None;
        loop {
            line.clear();
            let got = r.read_until(b'\n', &mut line)?;
            if got == 0 {
                break;
            }
            let p = (pos >> shift) as usize;
            let slot = pcount
                .get_mut(p)
                .ok_or(Error::ChangedDuringBuild)?;
            *slot += 1;
            if with_articles {
                let k = jsonl_id_key(&line).ok_or(Error::NoArticleKey {
                    record: n,
                    offset: pos,
                })?;
                if prev.as_deref() != Some(k) {
                    acount[p] += 1;
                    astarts.push(n);
                    prev = Some(k.to_vec());
                }
            }
            pos += got as u64;
            n += 1;
            ends_nl = line.last() == Some(&b'\n');
        }
    }
    let md1 = f.metadata()?;
    if pos != size
        || md1.len() != md0.len()
        || mtime_ns(&md1) != mtime_ns(&md0)
        || source_identity(&f, IDENT_BYTES)? != ident
    {
        return Err(Error::ChangedDuringBuild);
    }

    let (pb, pb_bits, pbs) = bucket_bitvector(&pcount);
    let (w, pbs_b) = int_array(&pbs);
    let mut sections = vec![
        Section {
            name: *b"PBV\0",
            width: 0,
            payload: pb,
            n: pb_bits,
            ones: n,
        },
        Section {
            name: *b"PBS\0",
            width: w,
            n: pbs.len() as u64,
            payload: pbs_b,
            ones: 0,
        },
    ];
    let articles = astarts.len() as u64;
    let mut flags = F_MTIME | if ends_nl { F_ENDS_NL } else { 0 };
    if with_articles {
        flags |= F_ARTICLES;
        let (ab, ab_bits, abs) = bucket_bitvector(&acount);
        let mut bb = vec![0u8; n.div_ceil(8) as usize];
        for &s in &astarts {
            bb[(s >> 3) as usize] |= 1 << (s & 7);
        }
        let mut bbs: Vec<u64> = astarts.iter().step_by(SAMPLE as usize).copied().collect();
        bbs.push(n);
        let mut bbr = Vec::with_capacity((n >> SAMPLE_SHIFT) as usize + 1);
        let mut j = 0usize;
        for b in 0..=(n >> SAMPLE_SHIFT) {
            let lim = b << SAMPLE_SHIFT;
            while j < astarts.len() && astarts[j] < lim {
                j += 1;
            }
            bbr.push(j as u64);
        }
        let (w1, abs_b) = int_array(&abs);
        let (w2, bbs_b) = int_array(&bbs);
        let (w3, bbr_b) = int_array(&bbr);
        sections.extend([
            Section {
                name: *b"ABV\0",
                width: 0,
                payload: ab,
                n: ab_bits,
                ones: articles,
            },
            Section {
                name: *b"ABS\0",
                width: w1,
                n: abs.len() as u64,
                payload: abs_b,
                ones: 0,
            },
            Section {
                name: *b"BBV\0",
                width: 0,
                payload: bb,
                n,
                ones: articles,
            },
            Section {
                name: *b"BBS\0",
                width: w2,
                n: bbs.len() as u64,
                payload: bbs_b,
                ones: 0,
            },
            Section {
                name: *b"BBR\0",
                width: w3,
                n: bbr.len() as u64,
                payload: bbr_b,
                ones: 0,
            },
        ]);
    }
    let section_sizes = sections
        .iter()
        .map(|s| {
            let name = s.name.iter().take_while(|&&c| c != 0).copied().collect();
            (String::from_utf8(name).expect("ASCII"), s.payload.len() as u64)
        })
        .collect();
    let header = Header {
        version: VERSION,
        header_size: 0,
        page_size,
        flags,
        sample: SAMPLE as u32,
        n_sections: 0,
        passages: n,
        pages,
        file_size: size,
        articles,
        ident_bytes: IDENT_BYTES,
        head_sha: ident.head_sha,
        tail_sha: ident.tail_sha,
        body_sha: [0; 32],
        key_spec: if with_articles {
            "jsonl-id".into()
        } else {
            String::new()
        },
        mtime_ns: mtime_ns(&md0),
    };
    let blob = encode(header, sections);
    write_atomic(out, &blob)?;
    Ok(BuildInfo {
        passages: n,
        pages,
        articles,
        file_size: size,
        mtime_ns: mtime_ns(&md0),
        page_size,
        bytes: blob.len() as u64,
        sections: section_sizes,
    })
}
