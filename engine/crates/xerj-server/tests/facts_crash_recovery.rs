//! Crash safety of `/_facts` (design gate E4: a process kill around the commit steps; no power-loss claim), against the
//! real binary and a real `SIGKILL`.
//!
//! The crash points (`xerj_facts::failpoint`) exist only with the test-only feature `facts-failpoints`, so this file
//! compiles to nothing without it:
//!
//! ```text
//! cargo test --profile ci-test -p xerj-server --features facts,facts-failpoints --test facts_crash_recovery
//! ```
//!
//! Each test writes acknowledged batches, arms one crash point for the next batch (the server reports the stop in a
//! marker file and holds there), kills the process by its PID, boots again on the same data directory and checks what
//! the design promises: every acknowledged batch is there, the interrupted batch is either all there or not at all, and
//! the index holds nothing of a batch that never committed and no stale retirement field. A wider matrix with a
//! fault-free twin server, random kill timing and large batches is the durability stream's harness (engine_dev
//! `30_durability/crash/crash_harness.py`); this file is the part that runs under `cargo test`.
//!
//! Linux/macOS only, like `cluster_metadata_survives_sigkill.rs`.

#![cfg(all(unix, feature = "facts-failpoints"))]

use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

fn three_free_ports() -> (u16, u16, u16) {
    let l1 = TcpListener::bind("127.0.0.1:0").unwrap();
    let l2 = TcpListener::bind("127.0.0.1:0").unwrap();
    let l3 = TcpListener::bind("127.0.0.1:0").unwrap();
    (
        l1.local_addr().unwrap().port(),
        l2.local_addr().unwrap().port(),
        l3.local_addr().unwrap().port(),
    )
}

/// One HTTP/1.1 request over a fresh connection: `(status, body as JSON)` (`Value::Null` when the body is not JSON).
fn request(
    port: u16,
    method: &str,
    path: &str,
    body: Option<&Value>,
) -> std::io::Result<(u16, Value)> {
    let mut stream = TcpStream::connect(("127.0.0.1", port))?;
    stream.set_read_timeout(Some(Duration::from_secs(120)))?;
    stream.set_write_timeout(Some(Duration::from_secs(30)))?;
    let payload = body.map(Value::to_string).unwrap_or_default();
    let head = format!(
        "{method} {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\
         Content-Type: application/json\r\nContent-Length: {}\r\n\r\n",
        payload.len()
    );
    stream.write_all(head.as_bytes())?;
    stream.write_all(payload.as_bytes())?;
    stream.flush()?;
    let mut reader = BufReader::new(stream);
    let mut status_line = String::new();
    reader.read_line(&mut status_line)?;
    let status: u16 = status_line
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line)? == 0 || line == "\r\n" {
            break;
        }
    }
    let mut text = String::new();
    reader.read_to_string(&mut text)?;
    Ok((status, serde_json::from_str(&text).unwrap_or(Value::Null)))
}

fn ok(port: u16, method: &str, path: &str, body: Option<&Value>) -> Value {
    let (st, b) = request(port, method, path, body).expect("request");
    assert_eq!(st, 200, "{method} {path}: {b}");
    b
}

/// A running node, owned by the test. Dropping it (a failed assertion unwinding through the test) kills and reaps the
/// process, so a failing test never leaves a server running on its temporary data directory; [`sigkill`] is the
/// deliberate kill of the test itself.
struct Running(Option<Child>);

impl Drop for Running {
    fn drop(&mut self) {
        if let Some(mut child) = self.0.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

/// One engine on a data directory, with the facts write sizes of the test (`epoch_ops`: op headers per epoch).
struct Node {
    dir: PathBuf,
    data: PathBuf,
    ports: (u16, u16, u16),
    epoch_ops: Option<usize>,
}

impl Node {
    fn new(dir: &Path, epoch_ops: Option<usize>) -> Node {
        let data = dir.join("data");
        std::fs::create_dir_all(&data).unwrap();
        Node {
            dir: dir.to_path_buf(),
            data,
            ports: three_free_ports(),
            epoch_ops,
        }
    }

    fn es(&self) -> u16 {
        self.ports.2
    }

    /// Boot the binary; `points` arms crash points (`XERJ_FACTS_FAILPOINTS`), reported in `marker`.
    fn boot(&self, points: Option<(&str, &Path)>) -> Running {
        let (rest, grpc, es) = self.ports;
        let config = format!(
            "[server]\nbind_address = \"127.0.0.1\"\nrest_port = {rest}\ngrpc_port = {grpc}\nes_compat_port = {es}\n\
             data_dir = \"{}\"\n",
            self.data.to_string_lossy().replace('\\', "/")
        );
        let config_path = self.dir.join("xerj.toml");
        std::fs::write(&config_path, config).unwrap();
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_xerj"));
        cmd.arg("--config")
            .arg(&config_path)
            .arg("--insecure")
            .arg("--no-auto-port")
            .env_remove("XERJ_FACTS_FAILPOINTS")
            .env_remove("XERJ_FACTS_FAILPOINT_FILE")
            .env_remove("XERJ_FACTS_TEST_EPOCH_OPS")
            .env_remove("XERJ_FACTS_TEST_RECORDS_PER_BULK")
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        if let Some(n) = self.epoch_ops {
            cmd.env("XERJ_FACTS_TEST_EPOCH_OPS", n.to_string());
        }
        if let Some((spec, marker)) = points {
            cmd.env("XERJ_FACTS_FAILPOINTS", spec)
                .env("XERJ_FACTS_FAILPOINT_FILE", marker);
        }
        let node = Running(Some(cmd.spawn().expect("spawn xerj")));
        let deadline = Instant::now() + Duration::from_secs(120);
        while Instant::now() < deadline {
            if let Ok((200, _)) = request(es, "GET", "/_cluster/health", None) {
                return node;
            }
            std::thread::sleep(Duration::from_millis(100));
        }
        panic!("xerj did not become ready on port {es}");
    }
}

/// `kill -9` by PID and reap: no shutdown hook runs.
fn sigkill(mut node: Running) {
    let mut child = node.0.take().expect("a running node");
    unsafe {
        libc::kill(child.id() as libc::pid_t, libc::SIGKILL);
    }
    let status = child.wait().expect("reap child");
    assert!(
        !status.success(),
        "the node exited cleanly instead of being killed: {status:?}"
    );
}

/// Send `body` to `path` on a thread, wait until the server reports the armed stop in `marker`, kill it there.
fn crash_during(node: &Node, child: Running, path: &str, body: &Value, marker: &Path) -> String {
    let (es, path_s, body_s) = (node.es(), path.to_string(), body.clone());
    let t = std::thread::spawn(move || request(es, "POST", &path_s, Some(&body_s)));
    let deadline = Instant::now() + Duration::from_secs(120);
    let stop = loop {
        let text = std::fs::read_to_string(marker).unwrap_or_default();
        if !text.trim().is_empty() {
            break text.trim().to_string();
        }
        assert!(
            Instant::now() < deadline,
            "the armed crash point was never reached"
        );
        assert!(
            !t.is_finished(),
            "the batch finished without reaching the armed crash point"
        );
        std::thread::sleep(Duration::from_millis(5));
    };
    sigkill(child);
    let reply = t.join().expect("request thread");
    assert!(
        !matches!(reply, Ok((200, _))),
        "a batch stopped before its acknowledgement was acknowledged: {reply:?}"
    );
    stop
}

fn doc(id: &str, text: &str) -> Value {
    json!({"doc_id": id, "version": "v1", "text": text})
}

/// `subject` "has height {value} cm." in document `doc_id`; the quote is the value.
fn height(fact_id: &str, doc_id: &str, subject: &str, value: &str) -> (Value, Value) {
    let text = format!("{subject} is {value} cm tall.");
    let start = subject.len() + 4;
    (
        doc(doc_id, &text),
        json!({"fact_id": fact_id, "subject": subject, "relation": "height", "kind": "number", "value": value,
               "unit": "cm", "doc_id": doc_id, "version": "v1", "start": start, "end": start + value.len(),
               "quote": value}),
    )
}

/// `n` new subjects with one height fact each, as one `_bulk` body; fact ids `{tag}-f{i}`.
fn new_facts(tag: &str, n: usize) -> Value {
    let (mut docs, mut facts) = (Vec::new(), Vec::new());
    for i in 0..n {
        let (d, f) = height(
            &format!("{tag}-f{i}"),
            &format!("{tag}-d{i}"),
            &format!("{tag} subject {i}"),
            "170",
        );
        docs.push(d);
        facts.push(f);
    }
    json!({"documents": docs, "facts": facts, "edits": []})
}

/// A correction of fact `{tag}-f{i}` for each `i`: a new document version whose fact supersedes it.
fn corrections(tag: &str, of: &[usize]) -> Value {
    let edits: Vec<Value> = of
        .iter()
        .map(|&i| {
            let (d, f) = height(
                &format!("{tag}-c{i}"),
                &format!("{tag}-cd{i}"),
                &format!("{tag} subject {i}"),
                "171",
            );
            json!({"op": "correct", "doc": d, "fact": f, "supersedes": format!("{tag}-f{i}")})
        })
        .collect();
    json!({"documents": [], "facts": [], "edits": edits})
}

fn committed_to(reply: &Value) -> u64 {
    reply["committed"]["to_rev"]
        .as_u64()
        .expect("committed.to_rev")
}

fn published(es: u16, ns: &str) -> u64 {
    ok(es, "GET", &format!("/_facts/{ns}"), None)["published_rev"]
        .as_u64()
        .expect("published_rev")
}

fn lookup(es: u16, ns: &str, subject: &str) -> Value {
    ok(
        es,
        "POST",
        &format!("/_facts/{ns}/_lookup"),
        Some(&json!({"subject": subject, "relation": "height"})),
    )
}

/// Documents of the facts index matching `query`, through the product's own search API (what a raw query sees).
fn raw_count(es: u16, ns: &str, query: Value) -> u64 {
    ok(
        es,
        "POST",
        &format!("/.xerj-memory-{ns}-facts/_count"),
        Some(&json!({"query": query})),
    )["count"]
        .as_u64()
        .expect("count")
}

/// Records and epochs written past revision `rev`: an interrupted batch's residue.
fn residue_past(es: u16, ns: &str, rev: u64) -> u64 {
    raw_count(
        es,
        ns,
        json!({"bool": {"should": [{"range": {"added_rev": {"gt": rev}}}, {"range": {"from_rev": {"gt": rev}}}]}}),
    )
}

#[test]
fn a_kill_before_the_commit_marker_rolls_the_batch_back_and_leaves_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let node = Node::new(dir.path(), None);
    let es = node.es();
    let ns = "crash-a";
    let child = node.boot(None);
    let base = committed_to(&ok(
        es,
        "POST",
        &format!("/_facts/{ns}/_bulk"),
        Some(&new_facts("a", 8)),
    ));
    sigkill(child);

    let marker = dir.path().join("marker");
    let child = node.boot(Some(("commit.after_records@1=stop", &marker)));
    let batch = new_facts("b", 8);
    let stop = crash_during(
        &node,
        child,
        &format!("/_facts/{ns}/_bulk"),
        &batch,
        &marker,
    );
    assert!(stop.starts_with("commit.after_records@1 "), "{stop}");

    let child = node.boot(None);
    // the records of the killed batch are in the index until the namespace's first access recovers it
    assert_eq!(
        residue_past(es, ns, base),
        16,
        "8 doc + 8 fact records of the killed batch"
    );
    assert_eq!(published(es, ns), base, "the killed batch never committed");
    assert_eq!(
        residue_past(es, ns, base),
        0,
        "the recovery deleted the killed batch's records"
    );
    assert_eq!(
        lookup(es, ns, "a subject 3")["status"],
        "VALUE",
        "acknowledged before the kill"
    );
    assert_eq!(
        lookup(es, ns, "b subject 3")["status"],
        "UNKNOWN",
        "never committed"
    );
    // the retry commits it whole; it survives a kill right after its acknowledgement
    let to = committed_to(&ok(
        es,
        "POST",
        &format!("/_facts/{ns}/_bulk"),
        Some(&batch),
    ));
    assert_eq!(to, base + 16);
    sigkill(child);
    let child = node.boot(None);
    assert_eq!(published(es, ns), to);
    assert_eq!(lookup(es, ns, "b subject 3")["status"], "VALUE");
    sigkill(child);
}

#[test]
fn a_kill_between_the_epochs_of_a_batch_rolls_back_all_of_it() {
    let dir = tempfile::tempdir().unwrap();
    // 4 op headers per epoch: the 16 ops of the crash batch take 4 epochs
    let node = Node::new(dir.path(), Some(4));
    let es = node.es();
    let ns = "crash-e";
    let child = node.boot(None);
    let base = committed_to(&ok(
        es,
        "POST",
        &format!("/_facts/{ns}/_bulk"),
        Some(&new_facts("a", 4)),
    ));
    let epochs_before = raw_count(es, ns, json!({"term": {"rt": "epoch"}}));
    sigkill(child);

    let marker = dir.path().join("marker");
    let child = node.boot(Some(("commit.epoch@3=stop", &marker)));
    crash_during(
        &node,
        child,
        &format!("/_facts/{ns}/_bulk"),
        &new_facts("b", 8),
        &marker,
    );

    let child = node.boot(None);
    assert_eq!(
        raw_count(es, ns, json!({"term": {"rt": "epoch"}})),
        epochs_before + 2,
        "two of the batch's four epochs were written before the kill"
    );
    assert_eq!(
        published(es, ns),
        base,
        "a batch commits with its last epoch, not with a prefix"
    );
    assert_eq!(
        residue_past(es, ns, base),
        0,
        "its records and its two epochs are gone"
    );
    assert_eq!(
        raw_count(es, ns, json!({"term": {"rt": "epoch"}})),
        epochs_before
    );
    assert_eq!(lookup(es, ns, "b subject 0")["status"], "UNKNOWN");
    sigkill(child);
}

#[test]
fn a_kill_after_the_commit_marker_keeps_the_batch_and_finishes_its_repairs() {
    let dir = tempfile::tempdir().unwrap();
    let node = Node::new(dir.path(), None);
    let es = node.es();
    let ns = "crash-r";
    let child = node.boot(None);
    ok(
        es,
        "POST",
        &format!("/_facts/{ns}/_bulk"),
        Some(&new_facts("a", 4)),
    );
    sigkill(child);

    let marker = dir.path().join("marker");
    let child = node.boot(Some(("commit.repair@1=stop", &marker)));
    crash_during(
        &node,
        child,
        &format!("/_facts/{ns}/_bulk"),
        &corrections("a", &[1, 2]),
        &marker,
    );

    let child = node.boot(None);
    let old = format!("/.xerj-memory-{ns}-facts/_doc/f:a-f1");
    assert!(
        ok(es, "GET", &old, None)["_source"]["retired_rev"].is_null(),
        "step 5 never ran before the kill"
    );
    let head = published(es, ns);
    assert_eq!(
        head, 12,
        "8 base ops + 2 corrections of 2 ops: committed by its marker"
    );
    let rec = ok(es, "GET", &old, None)["_source"].clone();
    assert_eq!(
        (rec["retired_rev"].as_u64(), rec["vis_end_rev"].as_u64()),
        (Some(10), Some(10)),
        "the recovery finished step 5: {rec}"
    );
    let l = lookup(es, ns, "a subject 1");
    assert_eq!(
        (l["status"].as_str(), l["refs"][0].as_str()),
        (Some("VALUE"), Some("a-c1"))
    );
    sigkill(child);
}

#[test]
fn a_kill_during_the_recovery_cleanup_converges_on_the_next_boot() {
    let dir = tempfile::tempdir().unwrap();
    let node = Node::new(dir.path(), None);
    let es = node.es();
    let ns = "crash-c";
    let child = node.boot(None);
    let base = committed_to(&ok(
        es,
        "POST",
        &format!("/_facts/{ns}/_bulk"),
        Some(&new_facts("a", 4)),
    ));
    sigkill(child);

    let marker = dir.path().join("marker");
    let child = node.boot(Some(("commit.after_records@1=stop", &marker)));
    crash_during(
        &node,
        child,
        &format!("/_facts/{ns}/_bulk"),
        &new_facts("b", 4),
        &marker,
    );

    // the first access after the restart recovers the namespace; kill it before the recovery deletes anything
    let marker2 = dir.path().join("marker2");
    let child = node.boot(Some(("recover.delete@1=stop", &marker2)));
    let t = std::thread::spawn(move || request(es, "GET", "/_facts/crash-c", None));
    let deadline = Instant::now() + Duration::from_secs(120);
    while std::fs::read_to_string(&marker2)
        .unwrap_or_default()
        .trim()
        .is_empty()
    {
        assert!(
            Instant::now() < deadline && !t.is_finished(),
            "the recovery never reached its cleanup"
        );
        std::thread::sleep(Duration::from_millis(5));
    }
    sigkill(child);
    let _ = t.join();

    let child = node.boot(None);
    assert_eq!(
        residue_past(es, ns, base),
        8,
        "nothing was deleted before the second kill"
    );
    assert_eq!(published(es, ns), base);
    assert_eq!(residue_past(es, ns, base), 0);
    sigkill(child);
}

#[test]
fn acknowledged_batches_survive_a_kill_right_after_their_acknowledgement() {
    let dir = tempfile::tempdir().unwrap();
    let node = Node::new(dir.path(), None);
    let es = node.es();
    let ns = "crash-k";
    let child = node.boot(None);
    let mut to = 0;
    for k in 0..6 {
        to = committed_to(&ok(
            es,
            "POST",
            &format!("/_facts/{ns}/_bulk"),
            Some(&new_facts(&format!("k{k}"), 5)),
        ));
    }
    sigkill(child);
    let child = node.boot(None);
    assert_eq!(published(es, ns), to);
    for k in 0..6 {
        assert_eq!(
            lookup(es, ns, &format!("k{k} subject 4"))["status"],
            "VALUE",
            "batch {k}"
        );
    }
    sigkill(child);
}

/// Step 5 re-indexes older records. When a record's first copy is already in a segment and the updated copy is flushed
/// into a newer one, the rebuild's index walk (`Index::fold_match_sources`) must return the updated copy: it once marked
/// the id as seen on the stale copy, skipped that copy as superseded, and then dropped the live copy as a duplicate, so
/// the rebuild failed with `facts_replay_failed` ("no fact document") and the namespace answered 500 after any restart.
#[test]
fn a_record_updated_after_a_flush_is_found_by_the_rebuild() {
    let dir = tempfile::tempdir().unwrap();
    let node = Node::new(dir.path(), None);
    let es = node.es();
    let ns = "crash-u";
    let child = node.boot(None);
    ok(
        es,
        "POST",
        &format!("/_facts/{ns}/_bulk"),
        Some(&new_facts("a", 4)),
    );
    ok(
        es,
        "POST",
        &format!("/.xerj-memory-{ns}-facts/_flush"),
        Some(&json!({})),
    );
    let to = committed_to(&ok(
        es,
        "POST",
        &format!("/_facts/{ns}/_bulk"),
        Some(&corrections("a", &[1, 2])),
    ));
    ok(
        es,
        "POST",
        &format!("/.xerj-memory-{ns}-facts/_flush"),
        Some(&json!({})),
    );
    sigkill(child);

    let child = node.boot(None);
    let (st, body) = request(es, "GET", &format!("/_facts/{ns}"), None).expect("status");
    assert_eq!(st, 200, "the rebuild after the restart: {body}");
    assert_eq!(body["published_rev"].as_u64(), Some(to));
    let rec = ok(
        es,
        "GET",
        &format!("/.xerj-memory-{ns}-facts/_doc/f:a-f1"),
        None,
    )["_source"]
        .clone();
    assert_eq!(rec["retired_rev"].as_u64(), Some(10), "{rec}");
    sigkill(child);
}

/// The engine can refuse a recovery's cleanup (its memory breaker answers 429 to the deletes of a large namespace after a
/// restart). The namespace then stays readable at its committed head: the replayed projection holds exactly the committed
/// chain and never reads the orphans (design §2.3 / §2.5). Writes wait: each one retries the cleanup first and answers 503
/// `NAMESPACE_RECOVERING` while it is refused.
#[test]
fn a_refused_recovery_cleanup_keeps_reads_and_holds_writes() {
    let dir = tempfile::tempdir().unwrap();
    let node = Node::new(dir.path(), None);
    let es = node.es();
    let ns = "crash-p";
    let child = node.boot(None);
    let base = committed_to(&ok(
        es,
        "POST",
        &format!("/_facts/{ns}/_bulk"),
        Some(&new_facts("a", 4)),
    ));
    sigkill(child);

    let marker = dir.path().join("marker");
    let child = node.boot(Some(("commit.after_records@1=stop", &marker)));
    crash_during(
        &node,
        child,
        &format!("/_facts/{ns}/_bulk"),
        &new_facts("b", 4),
        &marker,
    );

    // the restart's cleanup is refused on the first access (a read) and on the first write
    let marker2 = dir.path().join("marker2");
    let child = node.boot(Some((
        "recover.delete@1=error,recover.delete@2=error",
        &marker2,
    )));
    let st = ok(es, "GET", &format!("/_facts/{ns}"), None);
    assert_eq!(st["published_rev"].as_u64(), Some(base), "{st}");
    assert_eq!(
        st["cleanup_pending"],
        json!({"orphans": 8, "repairs": 0}),
        "{st}"
    );
    assert_eq!(lookup(es, ns, "a subject 1")["status"], "VALUE");
    assert_eq!(lookup(es, ns, "b subject 1")["status"], "UNKNOWN");
    assert_eq!(residue_past(es, ns, base), 8, "nothing deleted yet");

    let (code, body) = request(
        es,
        "POST",
        &format!("/_facts/{ns}/_bulk"),
        Some(&new_facts("c", 2)),
    )
    .expect("write");
    assert_eq!(
        (code, body["error"]["type"].as_str()),
        (503, Some("NAMESPACE_RECOVERING")),
        "{body}"
    );
    assert_eq!(residue_past(es, ns, base), 8);

    // the next write finishes the cleanup first, then commits at the next revision
    let to = committed_to(&ok(
        es,
        "POST",
        &format!("/_facts/{ns}/_bulk"),
        Some(&new_facts("c", 2)),
    ));
    assert_eq!(to, base + 4);
    assert_eq!(
        raw_count(es, ns, json!({"prefix": {"doc_id": "b-"}})),
        0,
        "the killed batch's records are gone"
    );
    let st = ok(es, "GET", &format!("/_facts/{ns}"), None);
    assert!(st.get("cleanup_pending").is_none(), "{st}");
    assert_eq!(lookup(es, ns, "c subject 1")["status"], "VALUE");
    sigkill(child);
}
