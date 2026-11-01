//! Port of `test_page_dir.py` and the self-check tests: every read is compared with an exact-offset
//! read, on files built to hit the edges of the format.
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::fs::{self, File};
use std::path::Path;
use xerj_pagedir::{build, open, ArticleKey, BuildOptions, Error, Expect, ReadOptions};

const OPT: ReadOptions = ReadOptions { prefetch: true };
const NOFETCH: ReadOptions = ReadOptions { prefetch: false };

fn rec(rng: &mut StdRng, art: u64, k: u64, textlen: usize) -> Vec<u8> {
    const L: &[u8] = b"abcdefghijklmnopqrstuvwxyz     ";
    let t: Vec<u8> = (0..textlen).map(|_| L[rng.gen_range(0..L.len())]).collect();
    let mut r = format!("{{\"id\": \"{art}#{k}\", \"text\": \"").into_bytes();
    r.extend_from_slice(&t);
    r.extend_from_slice(b"\"}\n");
    r
}

fn gen_random(rng: &mut StdRng, nrec: usize, mix: &[(usize, usize)]) -> Vec<Vec<u8>> {
    let (mut out, mut art, mut k) = (Vec::new(), 1000u64, 0u64);
    while out.len() < nrec {
        if rng.gen_bool(0.15) || k > 40 {
            art += rng.gen_range(1..50);
            k = 0;
        }
        let (a, b) = mix[rng.gen_range(0..mix.len())];
        let len = rng.gen_range(a..b);
        out.push(rec(rng, art, k, len));
        k += 1;
    }
    out
}

fn key(line: &[u8]) -> Vec<u8> {
    let h = line.iter().position(|&c| c == b'#').unwrap();
    line[8..h].to_vec()
}

fn id_of(line: &[u8]) -> String {
    let e = line[8..].iter().position(|&c| c == b'"').unwrap();
    String::from_utf8(line[8..8 + e].to_vec()).unwrap()
}

fn check(dir: &Path, name: &str, lines: &[Vec<u8>], sizes: &[u32], strip_nl: bool) {
    let mut lines = lines.to_vec();
    if strip_nl {
        lines.last_mut().unwrap().pop();
    }
    let data: Vec<u8> = lines.concat();
    let path = dir.join(format!("{name}.jsonl"));
    fs::write(&path, &data).unwrap();
    let mut starts = Vec::new();
    let mut pos = 0u64;
    for l in &lines {
        starts.push(pos);
        pos += l.len() as u64;
    }
    let keys: Vec<Vec<u8>> = lines.iter().map(|l| key(l)).collect();
    let arts: Vec<usize> = (0..lines.len())
        .filter(|&i| i == 0 || keys[i] != keys[i - 1])
        .collect();
    let (n, na) = (lines.len() as u64, arts.len() as u64);
    let f = File::open(&path).unwrap();
    for &pg in sizes {
        let sh = pg.trailing_zeros();
        let out = dir.join(format!("{name}.{pg}.pgdir"));
        let opts = BuildOptions {
            page_size: pg,
            article_key: ArticleKey::JsonlId,
        };
        build(&path, &out, opts).unwrap();
        let d = open(&out).unwrap();
        d.check_source(&f).unwrap();
        assert_eq!(d.check_stat(&f).unwrap(), Some(true));
        assert_eq!((d.passages(), d.articles(), d.file_size()), (n, na, pos));
        assert_eq!(d.pages(), pos.div_ceil(pg as u64));
        let pages: Vec<u64> = starts.iter().map(|s| s >> sh).collect();
        for (i, line) in lines.iter().enumerate() {
            let i64_ = i as u64;
            let p = pages[i];
            let first = pages.iter().position(|&x| x == p).unwrap();
            let count = pages.iter().filter(|&&x| x == p).count() as u64;
            let loc = d.locate(i64_).unwrap();
            assert_eq!(
                (loc.page, loc.rank, loc.count),
                (p, (i - first) as u64, count),
                "{name} page {pg} passage {i}"
            );
            for o in [OPT, NOFETCH] {
                assert_eq!(&d.read_unchecked(&f, i64_, o).unwrap(), line);
                let id = id_of(line);
                assert_eq!(&d.read(&f, i64_, Expect::Id(&id), o).unwrap(), line);
                let h = xerj_pagedir::line_hash(line);
                assert_eq!(&d.read(&f, i64_, Expect::LineHash(h), o).unwrap(), line);
            }
        }
        for (i, &s) in starts.iter().enumerate() {
            assert_eq!(d.passage_at(&f, s).unwrap(), i as u64, "{name} page {pg}");
        }
        let mut before = 0;
        for p in 0..d.pages() {
            let c = pages.iter().filter(|&&x| x == p).count() as u64;
            assert_eq!(d.passages_in_page(p).unwrap(), (before, c));
            before += c;
        }
        let mut ends = arts[1..].to_vec();
        ends.push(lines.len());
        for (a, (&s, &e)) in arts.iter().zip(&ends).enumerate() {
            let a = a as u64;
            assert_eq!(d.article_passages(a).unwrap(), (s as u64, (e - s) as u64));
            for i in s..e {
                assert_eq!(d.article_of(i as u64).unwrap(), a);
            }
            let start = d.article_start(a).unwrap();
            assert_eq!(start.page, pages[s]);
            let k = String::from_utf8(keys[s].clone()).unwrap();
            // The research corpus numbers passages from 0 inside each article; this file's runs can
            // restart or continue a number, so only runs numbered that way are read with the check.
            let numbered = (s..e).all(|i| id_of(&lines[i]) == format!("{k}#{}", i - s));
            if numbered {
                let want = lines[s..e].to_vec();
                assert_eq!(d.read_article(&f, a, &k, OPT).unwrap(), want);
                assert_eq!(d.read_article(&f, a, &k, NOFETCH).unwrap(), want);
                assert_eq!(d.read_article_by_start_page(&f, a, &k).unwrap(), want);
            }
            for i in s..e {
                assert_eq!(
                    d.expected_id(i as u64, &k).unwrap(),
                    format!("{k}#{}", i - s)
                );
            }
        }
        let out2 = dir.join(format!("{name}.{pg}.p.pgdir"));
        let opts = BuildOptions {
            page_size: pg,
            article_key: ArticleKey::None,
        };
        build(&path, &out2, opts).unwrap();
        let d2 = open(&out2).unwrap();
        assert!(fs::metadata(&out2).unwrap().len() < fs::metadata(&out).unwrap().len());
        for (i, line) in lines.iter().enumerate() {
            assert_eq!(&d2.read_unchecked(&f, i as u64, OPT).unwrap(), line);
        }
        assert!(d2.article_of(0).is_err(), "no article level");
    }
}

#[test]
fn edge_files() {
    let t = tempfile::tempdir().unwrap();
    let mut rng = StdRng::seed_from_u64(7);
    let mix = [(0, 6), (10, 200), (300, 2500), (5000, 20000)];
    let lines = gen_random(&mut rng, 600, &mix);
    for strip in [false, true] {
        let name = format!("rnd{strip}");
        check(t.path(), &name, &lines, &[16, 64, 256, 4096, 65536], strip);
    }
    // More than 64 record starts in one 4 KB page.
    let tiny = gen_random(&mut rng, 1500, &[(0, 3)]);
    check(t.path(), "tiny", &tiny, &[16, 256, 4096], false);
    // Lines that end exactly at page ends.
    let aligned: Vec<Vec<u8>> = (0..300u64)
        .map(|i| {
            let r = rec(&mut rng, 5 + i / 3, i % 3, 0);
            let mut x = r[..r.len() - 3].to_vec();
            x.resize(61, b'a');
            x.extend_from_slice(b"\"}\n");
            x
        })
        .collect();
    assert!(aligned.iter().all(|x| x.len() == 64));
    check(
        t.path(),
        "aligned",
        &aligned,
        &[16, 32, 64, 128, 4096],
        false,
    );
    // Records of several pages: long runs of zeros in the passage bitvector.
    let big = gen_random(&mut rng, 60, &[(9000, 40000)]);
    check(t.path(), "big", &big, &[16, 4096, 65536], false);
    let one = vec![rec(&mut rng, 1, 0, 10)];
    check(t.path(), "one", &one, &[16, 4096], false);
    check(t.path(), "one_nonl", &one, &[16, 4096], true);
    // Key 7 again after key 8 is a new run.
    let rep: Vec<Vec<u8>> = [(7, 0), (7, 1), (8, 0), (7, 2), (7, 3)]
        .iter()
        .map(|&(a, k)| rec(&mut rng, a, k, 50))
        .collect();
    check(t.path(), "rep", &rep, &[16, 64, 4096], false);
}

#[test]
fn empty_file() {
    let t = tempfile::tempdir().unwrap();
    let p = t.path().join("e.jsonl");
    fs::write(&p, b"").unwrap();
    let out = t.path().join("e.pgdir");
    build(&p, &out, BuildOptions::default()).unwrap();
    let d = open(&out).unwrap();
    d.check_source(&File::open(&p).unwrap()).unwrap();
    assert_eq!((d.passages(), d.pages(), d.articles()), (0, 0, 0));
    assert!(d.locate(0).is_err());
}

#[test]
fn a_line_without_a_key_stops_an_article_build() {
    let t = tempfile::tempdir().unwrap();
    let p = t.path().join("x.jsonl");
    fs::write(&p, b"{\"id\": \"1#0\"}\nplain text\n").unwrap();
    let out = t.path().join("x.pgdir");
    let e = build(&p, &out, BuildOptions::default()).unwrap_err();
    assert!(matches!(
        e,
        Error::NoArticleKey {
            record: 1,
            offset: 14
        }
    ));
    assert!(!out.exists());
    let opts = BuildOptions {
        article_key: ArticleKey::None,
        ..Default::default()
    };
    build(&p, &out, opts).unwrap();
}

#[test]
fn a_damaged_directory_is_refused() {
    let t = tempfile::tempdir().unwrap();
    let p = t.path().join("x.jsonl");
    fs::write(&p, b"{\"id\": \"1#0\", \"t\": \"a\"}\n").unwrap();
    let out = t.path().join("x.pgdir");
    build(&p, &out, BuildOptions::default()).unwrap();
    let good = fs::read(&out).unwrap();
    // A flipped bit in the body, a cut file, a wrong magic, an unknown version.
    let mut body = good.clone();
    *body.last_mut().unwrap() ^= 1;
    fs::write(&out, &body).unwrap();
    assert!(matches!(open(&out), Err(Error::Format(_))));
    fs::write(&out, &good[..100]).unwrap();
    assert!(matches!(open(&out), Err(Error::Format(_))));
    let mut magic = good.clone();
    magic[0] = 0;
    fs::write(&out, &magic).unwrap();
    assert!(matches!(open(&out), Err(Error::Format(_))));
    let mut ver = good.clone();
    ver[8] = 9;
    fs::write(&out, &ver).unwrap();
    assert!(matches!(open(&out), Err(Error::Format(_))));
    // Every truncation either fails to open or was not needed by the header.
    for cut in 0..good.len() {
        fs::write(&out, &good[..cut]).unwrap();
        assert!(open(&out).is_err(), "cut at {cut}");
    }
}

#[derive(Clone, Copy)]
enum Check {
    None,
    Id,
    Hash,
}

/// Reads every passage of an edited file. Returns (correct, stale errors, wrong lines returned).
fn sweep(d: &xerj_pagedir::PageDir, f: &File, lines: &[Vec<u8>], check: Check) -> (u32, u32, u32) {
    let (mut ok, mut stale, mut wrong) = (0, 0, 0);
    for (i, line) in lines.iter().enumerate() {
        let r = match check {
            Check::None => d.read_unchecked(f, i as u64, OPT),
            Check::Id => d.read(f, i as u64, Expect::Id(&id_of(line)), OPT),
            Check::Hash => {
                let h = xerj_pagedir::line_hash(line);
                d.read(f, i as u64, Expect::LineHash(h), OPT)
            }
        };
        match r {
            Ok(got) if got == *line => ok += 1,
            Ok(_) => wrong += 1,
            Err(e) => {
                assert!(e.is_stale(), "{e}");
                stale += 1;
            }
        }
    }
    (ok, stale, wrong)
}

#[test]
fn an_edited_file_gives_errors_never_wrong_text() {
    let t = tempfile::tempdir().unwrap();
    let mut rng = StdRng::seed_from_u64(29);
    let lines = gen_random(&mut rng, 3000, &[(40, 200), (200, 800)]);
    let data: Vec<u8> = lines.concat();
    let p = t.path().join("c.jsonl");
    let out = t.path().join("c.pgdir");
    fs::write(&p, &data).unwrap();
    build(&p, &out, BuildOptions::default()).unwrap();
    let d = open(&out).unwrap();

    let mut wrong_without_check = 0;
    for (ins, frac) in [(1usize, 0.0f64), (7, 0.5), (100, 0.0), (1000, 0.5)] {
        let x = (frac * data.len() as f64) as usize;
        let mut e = data[..x].to_vec();
        e.resize(x + ins, b'x');
        e.extend_from_slice(&data[x..]);
        fs::write(&p, &e).unwrap();
        let f = File::open(&p).unwrap();
        assert!(matches!(d.check_source(&f), Err(Error::SourceMismatch(_))));
        assert_eq!(d.check_stat(&f).unwrap(), Some(false));
        wrong_without_check += sweep(&d, &f, &lines, Check::None).2;
        // The id check cannot see the one line the bytes landed inside: its id is still right.
        let (_, stale, wrong) = sweep(&d, &f, &lines, Check::Id);
        assert!(wrong <= 1, "insert {ins} at {frac}: {wrong}");
        assert!(stale > 0);
        let (_, stale, wrong) = sweep(&d, &f, &lines, Check::Hash);
        assert_eq!(wrong, 0, "insert {ins} at {frac}");
        assert!(stale > 0);
    }
    assert!(
        wrong_without_check > 0,
        "the check must have something to catch"
    );

    // The same-size edit: size and the hashes of the first and last MiB are unchanged, so nothing
    // but the per-read check can see it. The file must exceed 2 MiB for this to hold.
    let pad = |lines: &mut Vec<Vec<u8>>, rng: &mut StdRng| {
        let mut art = 900_000;
        while lines.iter().map(Vec::len).sum::<usize>() < 3 << 20 {
            lines.push(rec(rng, art, 0, 2000));
            art += 1;
        }
    };
    let mut big = lines.clone();
    pad(&mut big, &mut rng);
    let data: Vec<u8> = big.concat();
    fs::write(&p, &data).unwrap();
    build(&p, &out, BuildOptions::default()).unwrap();
    let d = open(&out).unwrap();
    let (a, b) = (data.len() * 2 / 5, data.len() * 3 / 5);
    let mut e = data[..a].to_vec();
    e.resize(a + 100, b'y');
    e.extend_from_slice(&data[a..b]);
    e.extend_from_slice(&data[b + 100..]);
    assert_eq!(e.len(), data.len());
    fs::write(&p, &e).unwrap();
    let f = File::open(&p).unwrap();
    d.check_source(&f)
        .expect("this edit passes the identity check");
    assert!(
        sweep(&d, &f, &big, Check::None).2 > 0,
        "unchecked reads go wrong"
    );
    let (_, stale, wrong) = sweep(&d, &f, &big, Check::Id);
    assert!(wrong <= 2, "{wrong}");
    assert!(stale > 0);
    let (_, stale, wrong) = sweep(&d, &f, &big, Check::Hash);
    assert_eq!(wrong, 0);
    assert!(stale > 0);
}

#[test]
fn an_edit_inside_the_right_passage_is_caught_by_the_hash() {
    let t = tempfile::tempdir().unwrap();
    let p = t.path().join("c.jsonl");
    let line = b"{\"id\": \"1#0\", \"t\": \"hello\"}\n";
    fs::write(&p, line).unwrap();
    let out = t.path().join("c.pgdir");
    build(&p, &out, BuildOptions::default()).unwrap();
    let d = open(&out).unwrap();
    let h = xerj_pagedir::line_hash(line);
    fs::write(&p, b"{\"id\": \"1#0\", \"t\": \"jello\"}\n").unwrap();
    let f = File::open(&p).unwrap();
    assert!(d.read(&f, 0, Expect::Id("1#0"), OPT).is_ok(), "same id");
    let e = d.read(&f, 0, Expect::LineHash(h), OPT).unwrap_err();
    assert!(matches!(e, Error::HashMismatch { .. }));
    assert!(e.to_string().starts_with("stale file:"), "{e}");
}

#[test]
fn a_truncated_file_gives_an_error() {
    let t = tempfile::tempdir().unwrap();
    let mut rng = StdRng::seed_from_u64(3);
    let lines = gen_random(&mut rng, 500, &[(40, 400)]);
    let data: Vec<u8> = lines.concat();
    let p = t.path().join("c.jsonl");
    let out = t.path().join("c.pgdir");
    fs::write(&p, &data).unwrap();
    build(&p, &out, BuildOptions::default()).unwrap();
    let d = open(&out).unwrap();
    let f = fs::OpenOptions::new().write(true).open(&p).unwrap();
    f.set_len(data.len() as u64 / 2).unwrap();
    let f = File::open(&p).unwrap();
    let (_, stale, wrong) = sweep(&d, &f, &lines, Check::Id);
    assert_eq!(wrong, 0, "a cut line must not pass as a whole one");
    assert!(stale > 0);
}

#[test]
fn escaped_ids_and_ids_in_other_positions_are_parsed() {
    let t = tempfile::tempdir().unwrap();
    let p = t.path().join("c.jsonl");
    fs::write(
        &p,
        concat!(
            "{\"id\": \"a\\\"b#0\", \"t\": 1}\n",
            "{\"t\": 2, \"id\": \"c#0\"}\n",
            "{\"id\": 7}\n",
            "not json\n"
        ),
    )
    .unwrap();
    let out = t.path().join("c.pgdir");
    let opts = BuildOptions {
        article_key: ArticleKey::None,
        ..Default::default()
    };
    build(&p, &out, opts).unwrap();
    let d = open(&out).unwrap();
    let f = File::open(&p).unwrap();
    assert!(d.read(&f, 0, Expect::Id("a\"b#0"), OPT).is_ok());
    assert!(d.read(&f, 1, Expect::Id("c#0"), OPT).is_ok());
    assert!(matches!(
        d.read(&f, 2, Expect::Id("7"), OPT),
        Err(Error::IdMismatch { .. })
    ));
    let e = d.read(&f, 3, Expect::Id("x"), OPT).unwrap_err();
    assert!(e.to_string().contains("is not a record"), "{e}");
}
