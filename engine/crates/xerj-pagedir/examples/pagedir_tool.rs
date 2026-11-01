//! Builds a directory, or reads a whole file back through one and compares every line with an
//! exact-offset read. `verify` checks each line's id against itself, from the sequential scan.
//!
//! ```text
//! pagedir_tool build SOURCE OUT [PAGE_SIZE]
//! pagedir_tool verify SOURCE DIR
//! ```
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::time::Instant;
use xerj_pagedir::{build, jsonl_record_id, open, BuildOptions, Expect, ReadOptions};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<String> = std::env::args().collect();
    match a.get(1).map(String::as_str) {
        Some("build") => {
            let opts = BuildOptions {
                page_size: a.get(4).map_or(Ok(4096), |s| s.parse())?,
                ..Default::default()
            };
            let t = Instant::now();
            let info = build(a[2].as_ref(), a[3].as_ref(), opts)?;
            println!("{info:?} in {:.1} s", t.elapsed().as_secs_f64());
        }
        Some("verify") => {
            let d = open(a[3].as_ref())?;
            let f = File::open(&a[2])?;
            d.check_source(&f)?;
            let t = Instant::now();
            let (mut n, mut bad) = (0u64, 0u64);
            let mut line = Vec::new();
            let mut r = BufReader::with_capacity(1 << 24, File::open(&a[2])?);
            while r.read_until(b'\n', &mut line)? > 0 {
                let id = jsonl_record_id(&line).ok_or("line without an id")?;
                let got = d.read(&f, n, Expect::Id(&id), ReadOptions::default())?;
                bad += (got != line) as u64;
                n += 1;
                line.clear();
            }
            println!(
                "{n} passages of {} read, {bad} differ, {:.1} s",
                d.passages(),
                t.elapsed().as_secs_f64()
            );
            if bad > 0 || n != d.passages() {
                std::process::exit(1);
            }
        }
        _ => {
            eprintln!("usage: pagedir_tool build SOURCE OUT [PAGE_SIZE] | verify SOURCE DIR");
            std::process::exit(2);
        }
    }
    Ok(())
}
