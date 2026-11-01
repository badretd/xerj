//! Golden parity against PyTorch (needs the exported models and fixtures; skipped when XERJ_INFER_DATA is unset).
//!
//!   XERJ_INFER_DATA=~/xerj_infer_data cargo test --release -p xerj-infer --test parity -- --nocapture
//!
//! For every <model>.<set>.jsonl fixture under $XERJ_INFER_DATA/fixtures (first XERJ_INFER_PARITY_N questions, default 25):
//! f32 must reproduce PyTorch's decision input ids exactly, op / pointer logits within 5e-4, identical actions and identical
//! free-running episodes (final answers); int8 with the guard (when int8_calib.json exists) must give identical actions.
#![cfg(feature = "tokenizer")]

use std::path::PathBuf;
use xerj_infer::episode::{self, DecisionEvent, Guard};
use xerj_infer::fixtures::{self, ReplayExecutor};
use xerj_infer::{Engine, Int8Calib, Kernels, LoadOptions, Precision, Session};

fn data_dir() -> Option<PathBuf> {
    std::env::var("XERJ_INFER_DATA").ok().map(PathBuf::from)
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .filter(|(x, y)| **x > f32::MIN && **y > f32::MIN)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

#[test]
fn golden_parity() {
    let Some(dir) = data_dir() else {
        eprintln!("XERJ_INFER_DATA not set: parity test skipped");
        return;
    };
    let n: usize = std::env::var("XERJ_INFER_PARITY_N")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(25);
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir.join("fixtures"))
        .unwrap()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "jsonl"))
        .collect();
    files.sort();
    assert!(!files.is_empty(), "no fixtures");
    for f in files {
        let name = f.file_name().unwrap().to_string_lossy().to_string();
        let model = name.split('.').next().unwrap().to_string();
        let mdir = dir.join("weights").join(&model);
        let mut rows = fixtures::load(&f).unwrap();
        rows.truncate(n);
        // ---- f32
        let e = Engine::load(&mdir, Precision::F32, Kernels::detect()).unwrap();
        let mut sess = e.session();
        let (mut worst, mut decisions) = (0f32, 0usize);
        for row in &rows {
            let mut ex = ReplayExecutor::new(row, true);
            let mut obs = |s: &Session, ev: &DecisionEvent| {
                let rec = &row.decisions[ev.step];
                assert_eq!(
                    s.input_ids(),
                    rec.ids,
                    "{name} {} d{}: input ids",
                    row.item_id,
                    ev.step
                );
                worst = worst
                    .max(max_diff(&ev.out.op_logits, &rec.op_logits))
                    .max(max_diff(&ev.out.ptr_logits, &rec.ptr_logits));
                assert_eq!(
                    (ev.op, ev.pointer),
                    (rec.op, rec.pointer),
                    "{name} {} d{}: action",
                    row.item_id,
                    ev.step
                );
                decisions += 1;
            };
            let od: &mut dyn FnMut(&Session, &DecisionEvent) = &mut obs;
            episode::run_episode(&mut sess, &row.question, &row.cands, &mut ex, Some(od)).unwrap();
            let mut free = ReplayExecutor::new(row, false);
            episode::run_episode(&mut sess, &row.question, &row.cands, &mut free, None).unwrap();
            assert!(
                free.identical_episode(),
                "{name} {}: free-running episode diverged",
                row.item_id
            );
        }
        assert!(worst < 5e-4, "{name}: max logit diff {worst}");
        eprintln!("{name}: f32 {} questions, {decisions} decisions identical, max |logit diff| {worst:.2e}", rows.len());
        // ---- int8 + guard
        let cpath = mdir.join("int8_calib.json");
        if let Ok(c) = Int8Calib::load(&cpath) {
            let guard = Guard {
                tau_op: c.tau_op,
                tau_ptr: c.tau_ptr,
            };
            let mut opt = LoadOptions::new(Precision::Int8, Kernels::detect());
            opt.keep_f32 = true;
            opt.calib = Some(c);
            let e8 = Engine::load_with(&mdir, &opt).unwrap();
            let mut s8 = e8.session();
            let mut same = 0usize;
            let mut total = 0usize;
            for row in &rows {
                let mut free = ReplayExecutor::new(row, false);
                episode::run_episode_guarded(
                    &mut s8,
                    &row.question,
                    &row.cands,
                    &mut free,
                    None,
                    Some(guard),
                )
                .unwrap();
                same += free.identical_episode() as usize;
                total += 1;
            }
            assert_eq!(
                same, total,
                "{name}: int8+guard final answers {same}/{total}"
            );
            eprintln!("{name}: int8+guard {same}/{total} identical episodes");
        }
    }
}
