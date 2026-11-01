//! Directories written by the research module `page_dir.py` read here, and directories written here
//! are the same bytes. The fixtures were made by `page_dir.build` (versions 2 and 1).
use std::fs::{self, File};
use xerj_pagedir::{build, open, ArticleKey, BuildOptions, Expect, ReadOptions};

const DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");

fn lines() -> Vec<Vec<u8>> {
    fs::read(format!("{DIR}/small.jsonl"))
        .unwrap()
        .split_inclusive(|&c| c == b'\n')
        .map(<[u8]>::to_vec)
        .collect()
}

fn read_all(dir_file: &str) {
    let d = open(format!("{DIR}/{dir_file}").as_ref()).unwrap();
    let f = File::open(format!("{DIR}/small.jsonl")).unwrap();
    d.check_source(&f).unwrap();
    let lines = lines();
    assert_eq!(d.passages(), lines.len() as u64);
    for (i, line) in lines.iter().enumerate() {
        let id: serde_json::Value = serde_json::from_slice(line).unwrap();
        let id = id["id"].as_str().unwrap();
        let got = d
            .read(&f, i as u64, Expect::Id(id), ReadOptions::default())
            .unwrap();
        assert_eq!(&got, line);
    }
}

#[test]
fn version_2_directories_read() {
    read_all("small.v2.4096.pgdir");
    read_all("small.v2.256.pgdir");
    read_all("small.v2.256.p.pgdir");
}

#[test]
fn version_1_directory_reads() {
    let d = open(format!("{DIR}/small.v1.4096.pgdir").as_ref()).unwrap();
    assert_eq!(d.version(), 1);
    assert_eq!(d.mtime_ns(), None);
    read_all("small.v1.4096.pgdir");
}

/// Same bytes as Python except the mtime field, which belongs to the file's own copy.
fn same_bytes(name: &str, opts: BuildOptions) {
    let t = tempfile::tempdir().unwrap();
    let src = t.path().join("small.jsonl");
    fs::copy(format!("{DIR}/small.jsonl"), &src).unwrap();
    let out = t.path().join("x.pgdir");
    build(&src, &out, opts).unwrap();
    let mut mine = fs::read(&out).unwrap();
    let theirs = fs::read(format!("{DIR}/{name}")).unwrap();
    assert_eq!(mine.len(), theirs.len());
    mine[184..192].copy_from_slice(&theirs[184..192]);
    assert!(mine == theirs, "{name} differs");
}

#[test]
fn directories_written_here_are_the_python_bytes() {
    let art = ArticleKey::JsonlId;
    let opts = |page_size, article_key| BuildOptions {
        page_size,
        article_key,
    };
    same_bytes("small.v2.4096.pgdir", opts(4096, art));
    same_bytes("small.v2.256.pgdir", opts(256, art));
    same_bytes("small.v2.256.p.pgdir", opts(256, ArticleKey::None));
}
