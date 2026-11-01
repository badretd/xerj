//! Asynchronous BM25 maintenance: the writer appends index ops to a batch (texts copied into one buffer per batch) and
//! hands full batches to one indexer thread that owns both BM25 backends. Every BM25 read (and commit / checkpoint) first
//! flushes the pending batch and waits until the indexer has applied everything sent so far, so read-your-writes holds
//! exactly as in inline mode; only the work moves off the writer thread (ingest ~ max(store, BM25) instead of the sum).

use crate::bm25::Bm25Backend;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::JoinHandle;

const BATCH_OPS: usize = 1024;

/// the two backends handed back by `into_backends` (records, documents)
pub type Backends = (Option<Box<dyn Bm25Backend>>, Option<Box<dyn Bm25Backend>>);

#[derive(Clone, Copy)]
enum Kind {
    AddRec,
    DelRec,
    AddDoc,
    DelDoc,
}

#[derive(Default)]
struct Batch {
    ops: Vec<(Kind, u32, u32, u32)>,
    text: String,
}

struct Shared {
    rec: Mutex<Option<Box<dyn Bm25Backend>>>,
    doc: Mutex<Option<Box<dyn Bm25Backend>>>,
    applied: AtomicU64,
    wake: Mutex<()>,
    cv: Condvar,
}

struct Pending {
    batch: Batch,
    sent: u64,
    tx: Option<mpsc::Sender<Batch>>,
}

pub struct AsyncBm25 {
    shared: Arc<Shared>,
    pending: Mutex<Pending>,
    handle: Option<JoinHandle<()>>,
}

impl AsyncBm25 {
    pub fn new(rec: Option<Box<dyn Bm25Backend>>, doc: Option<Box<dyn Bm25Backend>>) -> AsyncBm25 {
        let shared = Arc::new(Shared {
            rec: Mutex::new(rec),
            doc: Mutex::new(doc),
            applied: AtomicU64::new(0),
            wake: Mutex::new(()),
            cv: Condvar::new(),
        });
        let (tx, rx) = mpsc::channel::<Batch>();
        let sh = shared.clone();
        let handle = std::thread::Builder::new()
            .name("xerj-bm25-indexer".into())
            .spawn(move || {
                while let Ok(b) = rx.recv() {
                    {
                        let mut rec = sh.rec.lock().unwrap();
                        let mut doc = sh.doc.lock().unwrap();
                        for &(k, id, s, e) in &b.ops {
                            let t = &b.text[s as usize..e as usize];
                            match k {
                                Kind::AddRec => rec.as_mut().map(|x| x.add(id, t)),
                                Kind::DelRec => rec.as_mut().map(|x| x.delete(id, t)),
                                Kind::AddDoc => doc.as_mut().map(|x| x.add(id, t)),
                                Kind::DelDoc => doc.as_mut().map(|x| x.delete(id, t)),
                            };
                        }
                    }
                    sh.applied.fetch_add(b.ops.len() as u64, Ordering::Release);
                    let _g = sh.wake.lock().unwrap();
                    sh.cv.notify_all();
                }
            })
            .expect("spawn indexer");
        AsyncBm25 {
            shared,
            pending: Mutex::new(Pending {
                batch: Batch::default(),
                sent: 0,
                tx: Some(tx),
            }),
            handle: Some(handle),
        }
    }

    fn push(&self, k: Kind, id: u32, text: &str) {
        let mut p = self.pending.lock().unwrap();
        let s = p.batch.text.len() as u32;
        p.batch.text.push_str(text);
        let e = p.batch.text.len() as u32;
        p.batch.ops.push((k, id, s, e));
        if p.batch.ops.len() >= BATCH_OPS || p.batch.text.len() > (4 << 20) {
            Self::send(&mut p);
        }
    }

    fn send(p: &mut Pending) {
        if p.batch.ops.is_empty() {
            return;
        }
        let b = std::mem::take(&mut p.batch);
        p.sent += b.ops.len() as u64;
        if let Some(tx) = &p.tx {
            tx.send(b).expect("indexer thread alive");
        }
    }

    pub fn add_record(&self, id: u32, text: &str) {
        self.push(Kind::AddRec, id, text)
    }
    pub fn delete_record(&self, id: u32, text: &str) {
        self.push(Kind::DelRec, id, text)
    }
    pub fn add_document(&self, id: u32, text: &str) {
        self.push(Kind::AddDoc, id, text)
    }
    pub fn delete_document(&self, id: u32, text: &str) {
        self.push(Kind::DelDoc, id, text)
    }

    /// Read barrier: flush the pending batch and wait until the indexer applied every op sent so far.
    pub fn sync(&self) {
        let target = {
            let mut p = self.pending.lock().unwrap();
            Self::send(&mut p);
            p.sent
        };
        if self.shared.applied.load(Ordering::Acquire) >= target {
            return;
        }
        let mut g = self.shared.wake.lock().unwrap();
        while self.shared.applied.load(Ordering::Acquire) < target {
            g = self.shared.cv.wait(g).unwrap();
        }
    }

    /// the records backend after a read barrier
    pub fn records(&self) -> MutexGuard<'_, Option<Box<dyn Bm25Backend>>> {
        self.sync();
        self.shared.rec.lock().unwrap()
    }

    /// the documents backend after a read barrier
    pub fn documents(&self) -> MutexGuard<'_, Option<Box<dyn Bm25Backend>>> {
        self.sync();
        self.shared.doc.lock().unwrap()
    }

    /// stop the thread and return the backends (after applying everything)
    pub fn into_backends(mut self) -> Backends {
        self.sync();
        self.pending.lock().unwrap().tx = None;
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
        let rec = self.shared.rec.lock().unwrap().take();
        let doc = self.shared.doc.lock().unwrap().take();
        (rec, doc)
    }
}

impl Drop for AsyncBm25 {
    fn drop(&mut self) {
        if let Ok(mut p) = self.pending.lock() {
            p.tx = None;
        }
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}
