//! Admission for `_ask` (design §4 "Execution"): a bound on concurrent asks, a bound on the asks waiting for a slot, and
//! a per-request deadline.
//!
//! An ask runs the learned compiler on one blocking-pool thread for seconds on a server CPU, so without a bound a burst of
//! asks takes every blocking thread and every core. [`AskGate`] holds three optional limits:
//!
//! * `max_concurrent`: at most this many asks run at once; the rest wait in arrival order (a fair `tokio` semaphore).
//! * `max_queued` (needs `max_concurrent`): at most this many asks wait for a slot; one more is refused at once with
//!   [`GateError::QueueFull`] (the API's `429 ASK_QUEUE_FULL`), instead of waiting for an answer that would come too
//!   late. `0` = nobody waits.
//! * `deadline`: an ask that has not finished within it (queueing included) is answered with [`GateError::Deadline`],
//!   which the API maps to `504 ASK_DEADLINE`.
//!
//! **Stopping abandoned work.** When the gate stops waiting for an ask (its deadline passed, or the request was dropped
//! because the client went away), it sets the ask's stop flag ([`StopFlag`]). The compiler reads it between two model
//! decisions (`xerj_serve::engine::answer_stoppable`), so the abandoned ask ends after the decision in progress and frees
//! its slot, instead of running to its natural end while the next asks wait behind it. A stopped ask has no answer at
//! all, so nothing partial is ever served. The slot is still released only when the work has really ended: the limit
//! counts work in flight, never only the requests still waiting. The flag is armed only under a limit or a deadline;
//! without either an abandoned ask runs to its end, as before.
//!
//! All limits are **off by default** (`None`): the defaults of the served engine are the owner's decision. They are set
//! with `XERJ_FACTS_MAX_CONCURRENT_ASKS`, `XERJ_FACTS_MAX_QUEUED_ASKS` and `XERJ_FACTS_ASK_DEADLINE_MS` (read once, see
//! [`AskGate::from_env`]); a request may ask for a shorter deadline of its own (`timeout_ms`), never a longer one.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{OwnedSemaphorePermit, Semaphore, TryAcquireError};

/// Environment variable: the most asks that run at once (unset, empty or 0 = no limit).
pub const ENV_MAX_CONCURRENT_ASKS: &str = "XERJ_FACTS_MAX_CONCURRENT_ASKS";
/// Environment variable: the server-wide deadline of one ask in milliseconds (unset, empty or 0 = none).
pub const ENV_ASK_DEADLINE_MS: &str = "XERJ_FACTS_ASK_DEADLINE_MS";
/// Environment variable: the most asks that wait for a slot (unset or empty = no bound; `0` = none wait). Needs
/// [`ENV_MAX_CONCURRENT_ASKS`].
pub const ENV_MAX_QUEUED_ASKS: &str = "XERJ_FACTS_MAX_QUEUED_ASKS";

/// An ask's stop flag: set by the gate when it stops waiting for the ask; the compiler reads it between two decisions.
pub type StopFlag = Arc<AtomicBool>;

/// Why the gate did not deliver a result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GateError {
    /// The deadline passed. `stage` is `"queue"` (no slot became free in time) or `"model"` (the computation was still
    /// running); `waited_ms` is the time spent in the gate.
    Deadline { stage: &'static str, waited_ms: u64 },
    /// Every slot was busy and `max_queued` asks were already waiting: refused at once, nothing ran.
    QueueFull { waiting: usize, max_queued: usize },
    /// The computation panicked or was cancelled (the blocking pool shut down).
    Failed(String),
}

/// The limits, the semaphore that enforces the first and the count of asks waiting for it. Cheap to share (`Arc` inside).
#[derive(Clone, Debug, Default)]
pub struct AskGate {
    slots: Option<Arc<Semaphore>>,
    max_concurrent: Option<usize>,
    max_queued: Option<usize>,
    waiting: Arc<AtomicUsize>,
    deadline: Option<Duration>,
}

impl AskGate {
    /// `max_concurrent` = `None` or `Some(0)`: no limit. `deadline` = `None` or zero: none. No bound on the asks waiting
    /// for a slot (see [`AskGate::with_max_queued`]).
    pub fn new(max_concurrent: Option<usize>, deadline: Option<Duration>) -> AskGate {
        let max_concurrent = max_concurrent.filter(|&n| n > 0);
        AskGate {
            slots: max_concurrent.map(|n| Arc::new(Semaphore::new(n))),
            max_concurrent,
            max_queued: None,
            waiting: Arc::new(AtomicUsize::new(0)),
            deadline: deadline.filter(|d| !d.is_zero()),
        }
    }

    /// The same gate with at most `max_queued` asks waiting for a slot (`None` = no bound, `Some(0)` = nobody waits).
    /// Without a concurrency limit nothing ever waits, so the bound needs `max_concurrent`.
    pub fn with_max_queued(mut self, max_queued: Option<usize>) -> Result<AskGate, String> {
        if max_queued.is_some() && self.max_concurrent.is_none() {
            return Err(format!(
                "{ENV_MAX_QUEUED_ASKS} bounds the asks waiting for a slot and needs {ENV_MAX_CONCURRENT_ASKS}"
            ));
        }
        self.max_queued = max_queued;
        Ok(self)
    }

    /// The limits from [`ENV_MAX_CONCURRENT_ASKS`], [`ENV_ASK_DEADLINE_MS`] and [`ENV_MAX_QUEUED_ASKS`]. A value that is
    /// not a whole number, or a queue bound without a concurrency limit, is reported in the error (the server refuses
    /// to guess an operator's limit).
    pub fn from_env() -> Result<AskGate, String> {
        AskGate::from_values(
            std::env::var(ENV_MAX_CONCURRENT_ASKS).ok().as_deref(),
            std::env::var(ENV_ASK_DEADLINE_MS).ok().as_deref(),
            std::env::var(ENV_MAX_QUEUED_ASKS).ok().as_deref(),
        )
    }

    /// [`AskGate::from_env`] over the three raw values (`None` = unset).
    pub fn from_values(
        max_concurrent: Option<&str>,
        deadline_ms: Option<&str>,
        max_queued: Option<&str>,
    ) -> Result<AskGate, String> {
        let read = |name: &str, v: Option<&str>| -> Result<Option<u64>, String> {
            match v.map(str::trim) {
                None | Some("") => Ok(None),
                Some(t) => t
                    .parse::<u64>()
                    .map(Some)
                    .map_err(|_| format!("{name}={t:?} is not a whole number")),
            }
        };
        let max = read(ENV_MAX_CONCURRENT_ASKS, max_concurrent)?.map(|n| n as usize);
        let deadline = read(ENV_ASK_DEADLINE_MS, deadline_ms)?.map(Duration::from_millis);
        let queued = read(ENV_MAX_QUEUED_ASKS, max_queued)?.map(|n| n as usize);
        AskGate::new(max, deadline).with_max_queued(queued)
    }

    pub fn max_concurrent(&self) -> Option<usize> {
        self.max_concurrent
    }

    pub fn max_queued(&self) -> Option<usize> {
        self.max_queued
    }

    pub fn deadline(&self) -> Option<Duration> {
        self.deadline
    }

    /// Asks waiting for a slot now (only known when a limit is set).
    pub fn waiting(&self) -> Option<usize> {
        self.slots
            .as_ref()
            .map(|_| self.waiting.load(Ordering::Acquire))
    }

    /// Asks running now (only known when a limit is set).
    pub fn running(&self) -> Option<usize> {
        match (&self.slots, self.max_concurrent) {
            (Some(s), Some(n)) => Some(n.saturating_sub(s.available_permits())),
            _ => None,
        }
    }

    /// The deadline that applies to one request: the shorter of the server's and the request's own (`timeout_ms`).
    pub fn effective_deadline(&self, requested: Option<Duration>) -> Option<Duration> {
        match (self.deadline, requested.filter(|d| !d.is_zero())) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        }
    }

    /// Run `work` on the blocking pool under the gate, with `deadline` counted from `started` (the start of the request:
    /// time already spent before the gate counts too). `work` cannot be stopped early: see [`AskGate::run_stoppable`].
    pub async fn run<T, F>(
        &self,
        started: Instant,
        deadline: Option<Duration>,
        work: F,
    ) -> Result<T, GateError>
    where
        T: Send + 'static,
        F: FnOnce() -> T + Send + 'static,
    {
        self.run_stoppable(started, deadline, move |_| work()).await
    }

    /// [`AskGate::run`] for work that checks a stop flag: the flag is set when this future stops waiting for the work
    /// without its result (the deadline passed, or the future was dropped because the client went away), so the work
    /// can end early and free its slot. Armed only when the gate has a concurrency limit or the request a deadline.
    pub async fn run_stoppable<T, F>(
        &self,
        started: Instant,
        deadline: Option<Duration>,
        work: F,
    ) -> Result<T, GateError>
    where
        T: Send + 'static,
        F: FnOnce(StopFlag) -> T + Send + 'static,
    {
        let left = |d: Duration| d.saturating_sub(started.elapsed());
        let waited = || started.elapsed().as_millis() as u64;
        let permit = match &self.slots {
            None => None,
            Some(slots) => Some(self.admit(slots, started, deadline).await?),
        };
        let stop: StopFlag = Arc::new(AtomicBool::new(false));
        // Set on every exit of this future (and harmless after the work has finished).
        let _stop_on_exit = if self.slots.is_some() || deadline.is_some() {
            Some(StopOnDrop(stop.clone()))
        } else {
            None
        };
        // The slot moves into the computation: it is released when the work ends, not when the caller stops waiting.
        let handle = tokio::task::spawn_blocking(move || {
            let _slot = permit;
            work(stop)
        });
        let joined = match deadline {
            None => handle.await,
            Some(d) => match tokio::time::timeout(left(d), handle).await {
                Ok(joined) => joined,
                Err(_) => {
                    return Err(GateError::Deadline {
                        stage: "model",
                        waited_ms: waited(),
                    })
                }
            },
        };
        joined.map_err(|e| GateError::Failed(e.to_string()))
    }

    /// A slot for one ask: at once if one is free; otherwise a place in line (unless `max_queued` asks already wait),
    /// until a slot frees or the deadline passes.
    async fn admit(
        &self,
        slots: &Arc<Semaphore>,
        started: Instant,
        deadline: Option<Duration>,
    ) -> Result<OwnedSemaphorePermit, GateError> {
        // A free slot is taken without queueing. The semaphore is fair: a released slot goes to the asks already
        // waiting, so this never jumps the line.
        match slots.clone().try_acquire_owned() {
            Ok(permit) => return Ok(permit),
            Err(TryAcquireError::Closed) => {
                return Err(GateError::Failed("the ask slots are closed".into()))
            }
            Err(TryAcquireError::NoPermits) => {}
        }
        let ahead = self.waiting.fetch_add(1, Ordering::AcqRel);
        let _in_line = LeaveLine(self.waiting.clone());
        if let Some(max_queued) = self.max_queued {
            if ahead >= max_queued {
                // a slot may have freed since the first try
                return slots
                    .clone()
                    .try_acquire_owned()
                    .map_err(|_| GateError::QueueFull {
                        waiting: ahead,
                        max_queued,
                    });
            }
        }
        let acquire = slots.clone().acquire_owned();
        let got = match deadline {
            None => acquire.await,
            Some(d) => {
                match tokio::time::timeout(d.saturating_sub(started.elapsed()), acquire).await {
                    Ok(got) => got,
                    Err(_) => {
                        return Err(GateError::Deadline {
                            stage: "queue",
                            waited_ms: started.elapsed().as_millis() as u64,
                        })
                    }
                }
            }
        };
        got.map_err(|e| GateError::Failed(e.to_string()))
    }
}

/// Sets an ask's stop flag when dropped.
struct StopOnDrop(StopFlag);

impl Drop for StopOnDrop {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Release);
    }
}

/// Takes an ask out of the waiting count when dropped (it got a slot, timed out, was refused or was dropped).
struct LeaveLine(Arc<AtomicUsize>);

impl Drop for LeaveLine {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::AcqRel);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn ms(n: u64) -> Duration {
        Duration::from_millis(n)
    }

    /// A unit of "model work": sleeps on the blocking thread, counts how many run at once.
    fn work(
        live: Arc<AtomicUsize>,
        peak: Arc<AtomicUsize>,
        sleep: Duration,
    ) -> impl FnOnce() -> usize + Send + 'static {
        move || {
            let now = live.fetch_add(1, Ordering::SeqCst) + 1;
            peak.fetch_max(now, Ordering::SeqCst);
            std::thread::sleep(sleep);
            live.fetch_sub(1, Ordering::SeqCst);
            now
        }
    }

    #[test]
    fn limits_are_off_by_default() {
        let g = AskGate::default();
        assert_eq!(
            (g.max_concurrent(), g.deadline(), g.running()),
            (None, None, None)
        );
        let g = AskGate::new(Some(0), Some(Duration::ZERO));
        assert_eq!((g.max_concurrent(), g.deadline()), (None, None));
        assert_eq!(g.effective_deadline(None), None);
        assert_eq!(g.effective_deadline(Some(ms(5))), Some(ms(5)));
        let g = AskGate::new(None, Some(ms(100)));
        assert_eq!(g.effective_deadline(None), Some(ms(100)));
        assert_eq!(
            g.effective_deadline(Some(ms(500))),
            Some(ms(100)),
            "a request cannot extend the server deadline"
        );
        assert_eq!(g.effective_deadline(Some(ms(20))), Some(ms(20)));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn without_a_limit_asks_run_side_by_side() {
        let g = AskGate::default();
        let (live, peak) = (Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0)));
        let t0 = Instant::now();
        let a = g.run(t0, None, work(live.clone(), peak.clone(), ms(300)));
        let b = g.run(t0, None, work(live.clone(), peak.clone(), ms(300)));
        let (a, b) = tokio::join!(a, b);
        assert!(a.is_ok() && b.is_ok());
        assert_eq!(peak.load(Ordering::SeqCst), 2);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn the_limit_queues_the_rest() {
        let g = AskGate::new(Some(1), None);
        let (live, peak) = (Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0)));
        let t0 = Instant::now();
        let runs: Vec<_> = (0..3)
            .map(|_| {
                let (g, w) = (g.clone(), work(live.clone(), peak.clone(), ms(100)));
                async move { g.run(t0, None, w).await }
            })
            .collect();
        for r in futures_join_all(runs).await {
            assert_eq!(r, Ok(1), "each ran alone");
        }
        assert_eq!(peak.load(Ordering::SeqCst), 1);
        assert!(
            t0.elapsed() >= ms(300),
            "three 100 ms asks one after the other"
        );
        assert_eq!(g.running(), Some(0));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_slow_ask_hits_the_deadline_and_keeps_its_slot_until_it_ends() {
        let g = AskGate::new(Some(1), Some(ms(1_000)));
        let (live, peak) = (Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0)));
        let t0 = Instant::now();
        // the request asks for 50 ms; the work takes 800 ms
        let r = g
            .run(
                t0,
                g.effective_deadline(Some(ms(50))),
                work(live.clone(), peak.clone(), ms(800)),
            )
            .await;
        assert!(
            matches!(r, Err(GateError::Deadline { stage: "model", .. })),
            "{r:?}"
        );
        assert!(
            t0.elapsed() < ms(700),
            "answered at the deadline, not after the work"
        );
        // the abandoned computation still holds the only slot: the next ask waits in the queue and times out there
        assert_eq!(g.running(), Some(1));
        let t1 = Instant::now();
        let r2 = g
            .run(t1, Some(ms(50)), work(live.clone(), peak.clone(), ms(10)))
            .await;
        assert!(
            matches!(r2, Err(GateError::Deadline { stage: "queue", .. })),
            "{r2:?}"
        );
        // once it ends, the slot is free again
        tokio::time::sleep(ms(1_000)).await;
        assert_eq!(g.running(), Some(0));
        let r3 = g
            .run(
                Instant::now(),
                Some(ms(500)),
                work(live, peak.clone(), ms(10)),
            )
            .await;
        assert_eq!(r3, Ok(1));
        assert_eq!(peak.load(Ordering::SeqCst), 1);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn time_spent_before_the_gate_counts() {
        let g = AskGate::default();
        let started = Instant::now() - ms(200);
        let r = g
            .run(started, Some(ms(100)), || std::thread::sleep(ms(300)))
            .await;
        assert!(matches!(r, Err(GateError::Deadline { .. })), "{r:?}");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_panicking_ask_is_an_error_and_frees_its_slot() {
        let g = AskGate::new(Some(1), None);
        let r: Result<(), _> = g.run(Instant::now(), None, || panic!("boom")).await;
        assert!(matches!(r, Err(GateError::Failed(_))), "{r:?}");
        assert_eq!(g.running(), Some(0));
    }

    /// `futures::future::join_all` without the dependency: the gate's futures are polled concurrently by spawning them.
    async fn futures_join_all<F>(fs: Vec<F>) -> Vec<F::Output>
    where
        F: std::future::Future + Send + 'static,
        F::Output: Send + 'static,
    {
        let handles: Vec<_> = fs.into_iter().map(tokio::spawn).collect();
        let mut out = Vec::new();
        for h in handles {
            out.push(h.await.expect("join"));
        }
        out
    }

    /// The parser behind `from_env` (the process environment is shared by parallel tests, so it is not mutated here).
    #[test]
    fn env_values_must_be_whole_numbers() {
        let g = AskGate::from_values(Some("2"), Some(" 1500 "), None).expect("valid");
        assert_eq!(
            (g.max_concurrent(), g.deadline(), g.max_queued()),
            (Some(2), Some(ms(1500)), None)
        );
        let g = AskGate::from_values(None, Some(""), Some("")).expect("unset");
        assert_eq!(
            (g.max_concurrent(), g.deadline(), g.max_queued()),
            (None, None, None)
        );
        let g = AskGate::from_values(Some("0"), Some("0"), None).expect("zero = off");
        assert_eq!((g.max_concurrent(), g.deadline()), (None, None));
        let e = AskGate::from_values(Some("two"), None, None).unwrap_err();
        assert!(
            e.contains(ENV_MAX_CONCURRENT_ASKS) && e.contains("two"),
            "{e}"
        );
        assert!(AskGate::from_values(None, Some("-5"), None).is_err());
        assert!(AskGate::from_values(None, Some("1.5"), None).is_err());
        // the line bound: 0 is a real value (nobody waits), and it needs a concurrency limit
        let g = AskGate::from_values(Some("2"), None, Some("0")).expect("valid");
        assert_eq!((g.max_concurrent(), g.max_queued()), (Some(2), Some(0)));
        let e = AskGate::from_values(None, None, Some("3")).unwrap_err();
        assert!(
            e.contains(ENV_MAX_QUEUED_ASKS) && e.contains(ENV_MAX_CONCURRENT_ASKS),
            "{e}"
        );
        assert!(AskGate::from_values(Some("2"), None, Some("x")).is_err());
    }

    /// "Model work" that checks its stop flag every 10 ms, as the compiler does between decisions; it ends by itself
    /// after `total`. Returns whether it was stopped; `seen` records it for the caller after the gate has answered.
    fn stoppable(
        total: Duration,
        seen: Arc<AtomicBool>,
    ) -> impl FnOnce(StopFlag) -> bool + Send + 'static {
        move |stop| {
            let t0 = Instant::now();
            while t0.elapsed() < total {
                if stop.load(Ordering::Acquire) {
                    seen.store(true, Ordering::Release);
                    return true;
                }
                std::thread::sleep(ms(10));
            }
            false
        }
    }

    /// Polls until no ask runs; returns how long that took.
    async fn until_idle(g: &AskGate, limit: Duration) -> Duration {
        let t0 = Instant::now();
        while g.running() != Some(0) && t0.elapsed() < limit {
            tokio::time::sleep(ms(10)).await;
        }
        t0.elapsed()
    }

    /// A missed deadline stops the work at its next check and frees the slot, instead of holding it until the work's
    /// natural end; the next ask runs at once.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_missed_deadline_stops_the_work_and_frees_its_slot() {
        let g = AskGate::new(Some(1), Some(ms(100)));
        let seen = Arc::new(AtomicBool::new(false));
        let t0 = Instant::now();
        let r = g
            .run_stoppable(t0, g.deadline(), stoppable(ms(5_000), seen.clone()))
            .await;
        assert!(
            matches!(r, Err(GateError::Deadline { stage: "model", .. })),
            "{r:?}"
        );
        let idle = until_idle(&g, ms(3_000)).await;
        assert!(seen.load(Ordering::Acquire), "the work saw the stop flag");
        assert!(
            idle < ms(1_000),
            "the slot was freed {idle:?} after the deadline, not at the work's end"
        );
        let r2 = g.run(Instant::now(), Some(ms(1_000)), || 7).await;
        assert_eq!(r2, Ok(7));
    }

    /// A request dropped while its work runs (the client went away) stops the work too.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_dropped_request_stops_its_work() {
        let g = AskGate::new(Some(1), None);
        let seen = Arc::new(AtomicBool::new(false));
        let g2 = g.clone();
        let w = stoppable(ms(5_000), seen.clone());
        let task = tokio::spawn(async move { g2.run_stoppable(Instant::now(), None, w).await });
        tokio::time::sleep(ms(150)).await;
        assert_eq!(g.running(), Some(1));
        task.abort();
        let idle = until_idle(&g, ms(3_000)).await;
        assert!(seen.load(Ordering::Acquire), "the work saw the stop flag");
        assert!(idle < ms(1_000), "freed {idle:?} after the drop");
    }

    /// Off by default: without a limit or a deadline nothing is stopped, and an abandoned ask runs to its end.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn without_a_limit_or_a_deadline_nothing_is_stopped() {
        let g = AskGate::default();
        let seen = Arc::new(AtomicBool::new(false));
        let done = Arc::new(AtomicBool::new(false));
        let (w, d2) = (stoppable(ms(300), seen.clone()), done.clone());
        let g2 = g.clone();
        let task = tokio::spawn(async move {
            g2.run_stoppable(Instant::now(), None, move |stop| {
                let stopped = w(stop);
                d2.store(true, Ordering::Release);
                stopped
            })
            .await
        });
        tokio::time::sleep(ms(50)).await;
        task.abort();
        tokio::time::sleep(ms(600)).await;
        assert!(done.load(Ordering::Acquire), "the work ran to its end");
        assert!(!seen.load(Ordering::Acquire), "and was never asked to stop");
    }

    /// With every slot busy and `max_queued` asks waiting, one more ask is refused at once; the waiting asks keep their
    /// place and run when a slot frees.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_full_line_is_refused_at_once() {
        let g = AskGate::new(Some(1), None)
            .with_max_queued(Some(1))
            .expect("valid");
        let t0 = Instant::now();
        let g1 = g.clone();
        let busy = tokio::spawn(async move {
            g1.run(Instant::now(), None, || std::thread::sleep(ms(400)))
                .await
        });
        tokio::time::sleep(ms(50)).await;
        let g2 = g.clone();
        let waiter = tokio::spawn(async move { g2.run(Instant::now(), None, || 2).await });
        tokio::time::sleep(ms(50)).await;
        assert_eq!((g.running(), g.waiting()), (Some(1), Some(1)));
        let t1 = Instant::now();
        let r = g.run(Instant::now(), None, || 3).await;
        assert_eq!(
            r,
            Err(GateError::QueueFull {
                waiting: 1,
                max_queued: 1
            })
        );
        assert!(t1.elapsed() < ms(100), "refused at once");
        assert!(busy.await.expect("join").is_ok());
        assert_eq!(waiter.await.expect("join"), Ok(2), "the waiting ask ran");
        assert!(t0.elapsed() >= ms(400));
        assert_eq!((g.running(), g.waiting()), (Some(0), Some(0)));

        // 0 = nobody waits: busy slot -> refused, free slot -> runs
        let g = AskGate::new(Some(1), None)
            .with_max_queued(Some(0))
            .expect("valid");
        let g1 = g.clone();
        let busy = tokio::spawn(async move {
            g1.run(Instant::now(), None, || std::thread::sleep(ms(200)))
                .await
        });
        tokio::time::sleep(ms(50)).await;
        assert!(matches!(
            g.run(Instant::now(), None, || 0).await,
            Err(GateError::QueueFull { .. })
        ));
        busy.await.expect("join").expect("ran");
        assert_eq!(g.run(Instant::now(), None, || 5).await, Ok(5));
        // without a concurrency limit there is no line to bound
        assert!(AskGate::new(None, Some(ms(10)))
            .with_max_queued(Some(1))
            .is_err());
    }
}
