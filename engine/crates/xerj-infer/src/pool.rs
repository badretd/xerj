//! A small fixed thread pool for the parallel loops inside one question (intra-question parallelism, off by default).
//!
//! `Pool::run(n, f)` runs `f(i)` for every i in 0..n: participant p (the caller is participant 0, workers 1..T-1) runs the
//! tasks p, p + T, p + 2T, ... and `run` returns when all tasks are done. The work split never changes what a task computes:
//! the users split GEMMs by output-channel shards and the attention by heads, so every output value is computed by the same
//! code on the same inputs as in a single-threaded run (bit-identical results).
//!
//! Workers spin for a short while after each job (the forward pass dispatches several jobs per layer, microseconds apart)
//! and then sleep on a condition variable until the next job. One job runs at a time: concurrent sessions that share a pool
//! serialize their parallel loops. A panic inside a task is caught in the worker and re-raised in the caller.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

/// how long an idle worker waits actively (spinning, then yielding) before it sleeps
const SPIN: Duration = Duration::from_micros(300);
/// pure spinning before a waiting thread starts to yield its core (keeps an oversubscribed host usable)
const PURE_SPIN: Duration = Duration::from_micros(20);

struct Job<'a> {
    f: &'a (dyn Fn(usize) + Sync),
    n: usize,
    threads: usize,
}

struct Shared {
    /// bumped once per job (the workers wait for a change)
    gen: AtomicUsize,
    /// address of the current `Job` (valid until `pending` reaches 0)
    job: AtomicUsize,
    /// workers that have not finished the current job
    pending: AtomicUsize,
    panicked: AtomicBool,
    stop: AtomicBool,
    sleepers: AtomicUsize,
    lock: Mutex<()>,
    cv: Condvar,
}

pub struct Pool {
    sh: Arc<Shared>,
    workers: Vec<std::thread::JoinHandle<()>>,
    threads: usize,
    run_lock: Mutex<()>,
}

impl Pool {
    /// a pool of `threads` participants (the caller plus `threads - 1` worker threads)
    pub fn new(threads: usize) -> Pool {
        let threads = threads.max(1);
        let sh = Arc::new(Shared {
            gen: AtomicUsize::new(0),
            job: AtomicUsize::new(0),
            pending: AtomicUsize::new(0),
            panicked: AtomicBool::new(false),
            stop: AtomicBool::new(false),
            sleepers: AtomicUsize::new(0),
            lock: Mutex::new(()),
            cv: Condvar::new(),
        });
        let workers = (1..threads)
            .map(|p| {
                let sh = sh.clone();
                std::thread::Builder::new()
                    .name(format!("xerj-infer-{p}"))
                    .spawn(move || worker(p, &sh))
                    .expect("spawn xerj-infer worker")
            })
            .collect();
        Pool {
            sh,
            workers,
            threads,
            run_lock: Mutex::new(()),
        }
    }

    /// participants (caller included)
    pub fn threads(&self) -> usize {
        self.threads
    }

    /// run `f(i)` for i in 0..n across the participants; returns when every task is done
    pub fn run(&self, n: usize, f: &(dyn Fn(usize) + Sync)) {
        if self.threads == 1 || n <= 1 {
            for i in 0..n {
                f(i);
            }
            return;
        }
        let _g = self.run_lock.lock().unwrap_or_else(|e| e.into_inner());
        let job = Job {
            f,
            n,
            threads: self.threads,
        };
        let sh = &*self.sh;
        sh.job.store(&job as *const Job as usize, Ordering::SeqCst);
        sh.pending.store(self.threads - 1, Ordering::SeqCst);
        sh.gen.fetch_add(1, Ordering::SeqCst);
        if sh.sleepers.load(Ordering::SeqCst) > 0 {
            let _l = sh.lock.lock().unwrap_or_else(|e| e.into_inner());
            sh.cv.notify_all();
        }
        let mine = catch_unwind(AssertUnwindSafe(|| {
            let mut i = 0;
            while i < n {
                f(i);
                i += self.threads;
            }
        }));
        // the workers read `job` (on this stack) until they decrement `pending`: wait for all of them, even after a panic
        let t0 = Instant::now();
        let mut spins = 0u32;
        while sh.pending.load(Ordering::Acquire) != 0 {
            spins = spins.wrapping_add(1);
            if spins % 64 == 0 && t0.elapsed() >= PURE_SPIN {
                std::thread::yield_now();
            } else {
                std::hint::spin_loop();
            }
        }
        if let Err(e) = mine {
            std::panic::resume_unwind(e);
        }
        if sh.panicked.swap(false, Ordering::SeqCst) {
            panic!("xerj-infer pool: a worker task panicked");
        }
    }
}

fn worker(p: usize, sh: &Shared) {
    // generation 0 = no job yet: a worker that starts after the first `run` began still sees that job (a later job cannot
    // begin before every worker finished the previous one, so no generation is ever skipped)
    let mut seen = 0usize;
    loop {
        // wait for the next job: spin, then sleep
        let t0 = Instant::now();
        let mut spins = 0u32;
        loop {
            if sh.stop.load(Ordering::Relaxed) {
                return;
            }
            let g = sh.gen.load(Ordering::Acquire);
            if g != seen {
                seen = g;
                break;
            }
            spins = spins.wrapping_add(1);
            if spins % 64 != 0 {
                std::hint::spin_loop();
                continue;
            }
            let waited = t0.elapsed();
            if waited < PURE_SPIN {
                continue;
            }
            if waited < SPIN {
                std::thread::yield_now();
                continue;
            }
            let guard = sh.lock.lock().unwrap_or_else(|e| e.into_inner());
            sh.sleepers.fetch_add(1, Ordering::SeqCst);
            if sh.gen.load(Ordering::SeqCst) == seen && !sh.stop.load(Ordering::SeqCst) {
                let _ = sh
                    .cv
                    .wait_timeout(guard, Duration::from_millis(50))
                    .unwrap_or_else(|e| e.into_inner());
            }
            sh.sleepers.fetch_sub(1, Ordering::SeqCst);
        }
        // SAFETY: `run` keeps the job alive until `pending` reaches 0, which needs this worker's decrement below.
        let job = unsafe { &*(sh.job.load(Ordering::SeqCst) as *const Job) };
        let r = catch_unwind(AssertUnwindSafe(|| {
            let mut i = p;
            while i < job.n {
                (job.f)(i);
                i += job.threads;
            }
        }));
        if r.is_err() {
            sh.panicked.store(true, Ordering::SeqCst);
        }
        sh.pending.fetch_sub(1, Ordering::Release);
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        self.sh.stop.store(true, Ordering::SeqCst);
        {
            let _l = self.sh.lock.lock().unwrap_or_else(|e| e.into_inner());
            self.sh.cv.notify_all();
        }
        for w in self.workers.drain(..) {
            let _ = w.join();
        }
    }
}

/// a raw pointer that may be shared with the pool's tasks (each task writes a disjoint part of the pointee)
#[derive(Clone, Copy)]
pub struct SendPtr<T>(pub *mut T);
// SAFETY: the users write disjoint elements from different tasks and the caller waits for all tasks (`Pool::run`).
unsafe impl<T> Send for SendPtr<T> {}
unsafe impl<T> Sync for SendPtr<T> {}

impl<T> SendPtr<T> {
    /// the pointer (a method, so closures capture the whole wrapper, not its non-Sync field)
    #[inline(always)]
    pub fn get(self) -> *mut T {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU64;

    #[test]
    fn runs_every_task_once() {
        for threads in [1, 2, 3, 4] {
            let pool = Pool::new(threads);
            for n in [0usize, 1, 2, 3, 7, 64] {
                let hits: Vec<AtomicU64> = (0..n).map(|_| AtomicU64::new(0)).collect();
                for _ in 0..50 {
                    pool.run(n, &|i| {
                        hits[i].fetch_add(1, Ordering::Relaxed);
                    });
                }
                assert!(hits.iter().all(|h| h.load(Ordering::Relaxed) == 50));
            }
            // after the workers fell asleep
            std::thread::sleep(Duration::from_millis(5));
            let s = AtomicU64::new(0);
            pool.run(10, &|i| {
                s.fetch_add(i as u64, Ordering::Relaxed);
            });
            assert_eq!(s.load(Ordering::Relaxed), 45);
        }
    }

    #[test]
    fn worker_panic_reaches_the_caller() {
        let pool = Pool::new(3);
        let r = catch_unwind(AssertUnwindSafe(|| {
            pool.run(3, &|i| {
                if i == 1 {
                    panic!("task 1");
                }
            })
        }));
        assert!(r.is_err());
        // the pool still works afterwards
        let s = AtomicU64::new(0);
        pool.run(4, &|_| {
            s.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(s.load(Ordering::Relaxed), 4);
    }
}
