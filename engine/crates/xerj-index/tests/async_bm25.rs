//! Async BM25 (indexer thread) must answer exactly like inline maintenance, immediately after every write.
use xerj_index::{Memory, MemoryConfig};
use xerj_store::FactIn;

#[test]
fn async_equals_inline_with_read_your_writes() {
    let mut a = Memory::new(MemoryConfig {
        async_bm25: true,
        ..MemoryConfig::default()
    });
    let mut b = Memory::new(MemoryConfig::default());
    let colors = ["red", "green", "blue", "teal", "amber"];
    for i in 0..400u32 {
        let subj = format!("Entity {} {}", i % 37, colors[(i % 5) as usize]);
        let doc = format!("d{i}");
        let val = colors[((i * 7) % 5) as usize];
        let text = format!("{subj} is painted {val} today.");
        let st = (subj.len() + " is painted ".len()) as i64;
        let fid = format!("f{i}");
        for m in [&mut a, &mut b] {
            m.add_document(&doc, "v1", &text).unwrap();
            m.add_fact(
                &FactIn::text(
                    &fid,
                    &subj,
                    "color",
                    val,
                    (&doc, "v1"),
                    (st, st + val.len() as i64),
                    val,
                ),
                &[],
            )
            .unwrap();
        }
        if i % 9 == 4 {
            let old = format!("f{}", i - 2);
            for m in [&mut a, &mut b] {
                let _ = m.withdraw_fact(&old);
            }
        }
        if i % 13 == 6 {
            let d = format!("d{}", i - 3);
            for m in [&mut a, &mut b] {
                let _ = m.withdraw_document(&d, "v1");
            }
        }
        let q = format!("what color is entity {} {}", i % 37, val);
        assert_eq!(
            a.search_records(&q, 8),
            b.search_records(&q, 8),
            "records after write {i}"
        );
        assert_eq!(
            a.search_documents(&q, 8),
            b.search_documents(&q, 8),
            "documents after write {i}"
        );
    }
    a.commit().unwrap();
    b.commit().unwrap();
    assert_eq!(
        a.search_records("entity 3 red", 20),
        b.search_records("entity 3 red", 20)
    );
}
