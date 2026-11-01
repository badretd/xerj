//! Test-only crash points at the commit and recovery steps of the facts store (design §2.3 steps 2-6 and §2.5; gate E4:
//! process kill around the commit steps). They exist only with the feature `failpoints`, which is off by default and in no
//! shipped feature set; without it every [`hit`] is an inlined `Action::Continue` and costs nothing.
//!
//! `XERJ_FACTS_FAILPOINTS` is a comma-separated list of `name[@n]=action`: the `n`-th hit (1-based, default 1) of the point
//! `name` in this process runs `action`:
//! - `stop`: append the line `name@n pid` to the file named by `XERJ_FACTS_FAILPOINT_FILE` (when set) and to stderr, then
//!   block the calling thread for good. The process stays exactly at that step until a harness kills it (SIGKILL by PID).
//! - `error`: the step fails as a failed engine write would, without a crash.
//! - `skip`: the one write the point guards is not performed (a lost write, as after a power loss). Points that guard no
//!   single write treat it as `continue`.
//! - `continue`: the hit is counted and nothing else happens.
//!
//! The points (`commit.*` in `xerj-api/src/facts_api.rs`, `recover.*` in its projection rebuild) are listed in the crash
//! harness of the durability stream; each name is one place in the code.

/// What the caller of a point does next.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Action {
    /// Go on as if there were no point.
    Continue,
    /// Fail this step with an error (no crash).
    Error,
    /// Do not perform the write this point guards.
    Skip,
}

/// Visit the point `name`. Without the feature `failpoints` this is always [`Action::Continue`].
#[cfg(not(feature = "failpoints"))]
#[inline(always)]
pub fn hit(_name: &str) -> Action {
    Action::Continue
}

/// Visit the point `name` (feature `failpoints`): runs the action configured for this hit in `XERJ_FACTS_FAILPOINTS`.
#[cfg(feature = "failpoints")]
pub fn hit(name: &str) -> Action {
    imp::hit(name)
}

/// Op headers per epoch document: `default`, or `XERJ_FACTS_TEST_EPOCH_OPS` with the feature `failpoints`, so the tests can
/// split a small batch over several epochs.
#[cfg(not(feature = "failpoints"))]
#[inline(always)]
pub fn epoch_ops(default: usize) -> usize {
    default
}

/// Op headers per epoch document (feature `failpoints`): `XERJ_FACTS_TEST_EPOCH_OPS` when set, else `default`.
#[cfg(feature = "failpoints")]
pub fn epoch_ops(default: usize) -> usize {
    imp::env_usize("XERJ_FACTS_TEST_EPOCH_OPS").unwrap_or(default)
}

/// Documents per bulk call of the facts writes: `default`, or `XERJ_FACTS_TEST_RECORDS_PER_BULK` with the feature
/// `failpoints`, so the tests can split a small batch over several calls.
#[cfg(not(feature = "failpoints"))]
#[inline(always)]
pub fn records_per_bulk(default: usize) -> usize {
    default
}

/// Documents per bulk call (feature `failpoints`): `XERJ_FACTS_TEST_RECORDS_PER_BULK` when set, else `default`.
#[cfg(feature = "failpoints")]
pub fn records_per_bulk(default: usize) -> usize {
    imp::env_usize("XERJ_FACTS_TEST_RECORDS_PER_BULK").unwrap_or(default)
}

#[cfg(feature = "failpoints")]
mod imp {
    use super::Action;
    use std::collections::HashMap;
    use std::io::Write;
    use std::sync::{Mutex, OnceLock};

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(super) enum Kind {
        Stop,
        Error,
        Skip,
        Continue,
    }

    #[derive(Clone, Debug, PartialEq, Eq)]
    pub(super) struct Spec {
        pub name: String,
        pub nth: u64,
        pub kind: Kind,
    }

    /// Parse `name[@n]=action[,name[@n]=action...]`.
    pub(super) fn parse(s: &str) -> Result<Vec<Spec>, String> {
        let mut out = Vec::new();
        for part in s.split(',').map(str::trim).filter(|p| !p.is_empty()) {
            let (point, action) = part
                .split_once('=')
                .ok_or_else(|| format!("'{part}': expected name[@n]=action"))?;
            let (name, nth) = match point.split_once('@') {
                Some((name, n)) => (
                    name,
                    n.parse::<u64>().ok().filter(|&n| n >= 1).ok_or_else(|| {
                        format!("'{part}': the hit number must be an integer >= 1")
                    })?,
                ),
                None => (point, 1),
            };
            if name.is_empty() {
                return Err(format!("'{part}': empty point name"));
            }
            let kind = match action {
                "stop" => Kind::Stop,
                "error" => Kind::Error,
                "skip" => Kind::Skip,
                "continue" => Kind::Continue,
                other => {
                    return Err(format!(
                        "'{part}': unknown action '{other}' (stop | error | skip | continue)"
                    ))
                }
            };
            out.push(Spec {
                name: name.to_string(),
                nth,
                kind,
            });
        }
        Ok(out)
    }

    pub(super) fn env_usize(k: &str) -> Option<usize> {
        std::env::var(k).ok().map(|v| {
            v.parse::<usize>()
                .ok()
                .filter(|&n| n >= 1)
                .unwrap_or_else(|| panic!("{k}={v}: expected an integer >= 1"))
        })
    }

    fn specs() -> &'static [Spec] {
        static SPECS: OnceLock<Vec<Spec>> = OnceLock::new();
        SPECS.get_or_init(|| match std::env::var("XERJ_FACTS_FAILPOINTS") {
            // a test-only build with a wrong spec must not run as if no point were set
            Ok(v) => parse(&v).unwrap_or_else(|e| panic!("XERJ_FACTS_FAILPOINTS: {e}")),
            Err(_) => Vec::new(),
        })
    }

    pub(super) fn hit(name: &str) -> Action {
        let specs = specs();
        if specs.is_empty() {
            return Action::Continue;
        }
        static HITS: OnceLock<Mutex<HashMap<String, u64>>> = OnceLock::new();
        let n = {
            let mut hits = HITS
                .get_or_init(Default::default)
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            let c = hits.entry(name.to_string()).or_insert(0);
            *c += 1;
            *c
        };
        match specs
            .iter()
            .find(|s| s.name == name && s.nth == n)
            .map(|s| s.kind)
        {
            None | Some(Kind::Continue) => Action::Continue,
            Some(Kind::Error) => Action::Error,
            Some(Kind::Skip) => Action::Skip,
            Some(Kind::Stop) => stop(name, n),
        }
    }

    /// Report the stop, then hold this thread until the process is killed from outside.
    fn stop(name: &str, n: u64) -> ! {
        let line = format!("{name}@{n} {}\n", std::process::id());
        if let Some(path) = std::env::var_os("XERJ_FACTS_FAILPOINT_FILE") {
            if let Ok(mut f) = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
            {
                let _ = f.write_all(line.as_bytes());
            }
        }
        eprint!("XERJ_FACTS_FAILPOINT stop {line}");
        loop {
            std::thread::sleep(std::time::Duration::from_secs(3600));
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn specs_parse() {
            let s =
                parse("commit.epoch@2=skip, commit.after_epochs=stop,recover.after_walk@3=error")
                    .unwrap();
            assert_eq!(
                s,
                vec![
                    Spec {
                        name: "commit.epoch".into(),
                        nth: 2,
                        kind: Kind::Skip
                    },
                    Spec {
                        name: "commit.after_epochs".into(),
                        nth: 1,
                        kind: Kind::Stop
                    },
                    Spec {
                        name: "recover.after_walk".into(),
                        nth: 3,
                        kind: Kind::Error
                    },
                ]
            );
            assert!(parse("").unwrap().is_empty());
            for bad in [
                "commit.epoch",
                "commit.epoch@0=stop",
                "commit.epoch@x=stop",
                "=stop",
                "a=explode",
            ] {
                assert!(parse(bad).is_err(), "{bad}");
            }
        }
    }
}
