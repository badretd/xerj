//! Opt-in eviction of facts projections (`XERJ_FACTS_MAX_RESIDENT_NAMESPACES`; unset, the default, keeps every
//! projection), against the real binary. An evicted namespace is rebuilt from its index on its next access (the recovery
//! path of design §2.5) with the same published revision, exact fingerprint and lookups, and its next write continues at
//! the next revision.
//!
//! ```text
//! cargo test --profile ci-test -p xerj-server --features facts --test facts_eviction
//! ```

#![cfg(all(unix, feature = "facts"))]

use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

fn free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

/// One HTTP/1.1 request over a fresh connection: `(status, body as JSON)`.
fn request(port: u16, method: &str, path: &str, body: Option<&Value>) -> (u16, Value) {
    let mut stream = TcpStream::connect(("127.0.0.1", port)).expect("connect");
    stream
        .set_read_timeout(Some(Duration::from_secs(120)))
        .unwrap();
    let payload = body.map(Value::to_string).unwrap_or_default();
    let head = format!(
        "{method} {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\
         Content-Type: application/json\r\nContent-Length: {}\r\n\r\n",
        payload.len()
    );
    stream.write_all(head.as_bytes()).unwrap();
    stream.write_all(payload.as_bytes()).unwrap();
    let mut reader = BufReader::new(stream);
    let mut status_line = String::new();
    reader.read_line(&mut status_line).unwrap();
    let status = status_line
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line).unwrap() == 0 || line == "\r\n" {
            break;
        }
    }
    let mut text = String::new();
    reader.read_to_string(&mut text).unwrap();
    (status, serde_json::from_str(&text).unwrap_or(Value::Null))
}

fn ok(port: u16, method: &str, path: &str, body: Option<&Value>) -> Value {
    let (st, b) = request(port, method, path, body);
    assert_eq!(st, 200, "{method} {path}: {b}");
    b
}

/// The server, killed and reaped when the test ends or fails.
struct Running(Child);

impl Drop for Running {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

/// Boot the binary on `dir`; `max_resident` sets `XERJ_FACTS_MAX_RESIDENT_NAMESPACES`. Returns the node and its port.
fn boot(dir: &Path, max_resident: Option<usize>) -> (Running, u16) {
    let (rest, grpc, es) = (free_port(), free_port(), free_port());
    let data = dir.join("data");
    std::fs::create_dir_all(&data).unwrap();
    let config = format!(
        "[server]\nbind_address = \"127.0.0.1\"\nrest_port = {rest}\ngrpc_port = {grpc}\nes_compat_port = {es}\n\
         data_dir = \"{}\"\n",
        data.to_string_lossy().replace('\\', "/")
    );
    let config_path = dir.join("xerj.toml");
    std::fs::write(&config_path, config).unwrap();
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_xerj"));
    cmd.arg("--config")
        .arg(&config_path)
        .arg("--insecure")
        .arg("--no-auto-port")
        .env_remove("XERJ_FACTS_MAX_RESIDENT_NAMESPACES")
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    if let Some(n) = max_resident {
        cmd.env("XERJ_FACTS_MAX_RESIDENT_NAMESPACES", n.to_string());
    }
    let node = Running(cmd.spawn().expect("spawn xerj"));
    let deadline = Instant::now() + Duration::from_secs(120);
    while Instant::now() < deadline {
        if TcpStream::connect(("127.0.0.1", es)).is_ok()
            && request(es, "GET", "/_cluster/health", None).0 == 200
        {
            return (node, es);
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    panic!("xerj did not become ready on port {es}");
}

/// `n` new subjects with one height fact each; returns the batch's last revision.
fn write(es: u16, ns: &str, tag: &str, n: usize) -> u64 {
    let (mut docs, mut facts) = (Vec::new(), Vec::new());
    for i in 0..n {
        let subject = format!("{tag} subject {i}");
        let text = format!("{subject} is 170 cm tall.");
        let start = subject.len() + 4;
        docs.push(json!({"doc_id": format!("{tag}-d{i}"), "version": "v1", "text": text}));
        facts.push(json!({"fact_id": format!("{tag}-f{i}"), "subject": subject, "relation": "height",
                          "kind": "number", "value": "170", "unit": "cm", "doc_id": format!("{tag}-d{i}"),
                          "version": "v1", "start": start, "end": start + 3, "quote": "170"}));
    }
    let body = json!({"documents": docs, "facts": facts, "edits": []});
    ok(es, "POST", &format!("/_facts/{ns}/_bulk"), Some(&body))["committed"]["to_rev"]
        .as_u64()
        .expect("committed.to_rev")
}

/// `(published_rev, exact fingerprint)` of a namespace (loads it when it is not loaded).
fn state(es: u16, ns: &str) -> (u64, String) {
    let v = ok(es, "GET", &format!("/_facts/{ns}?fingerprint=true"), None);
    (
        v["published_rev"].as_u64().expect("published_rev"),
        v["fingerprint_exact"]
            .as_str()
            .expect("fingerprint_exact")
            .to_string(),
    )
}

fn lookup(es: u16, ns: &str, subject: &str) -> Value {
    ok(
        es,
        "POST",
        &format!("/_facts/{ns}/_lookup"),
        Some(&json!({"subject": subject, "relation": "height"})),
    )
}

/// `(loaded namespaces, max_resident_namespaces, evictions)` from `GET /_facts`.
fn residency(es: u16) -> (Vec<String>, Value, u64) {
    let v = ok(es, "GET", "/_facts", None);
    let loaded = v["loaded"]
        .as_array()
        .expect("loaded")
        .iter()
        .map(|l| l["namespace"].as_str().unwrap().to_string())
        .collect();
    (
        loaded,
        v["max_resident_namespaces"].clone(),
        v["evictions"].as_u64().expect("evictions"),
    )
}

#[test]
fn an_evicted_namespace_is_rebuilt_unchanged_on_its_next_access() {
    let dir = tempfile::tempdir().unwrap();
    let (_node, es) = boot(dir.path(), Some(1));
    let a1 = write(es, "ev-a", "a", 6);
    let before = state(es, "ev-a");
    assert_eq!(before.0, a1);
    let a_lookup = lookup(es, "ev-a", "a subject 3");
    assert_eq!(a_lookup["status"], "VALUE");

    let b1 = write(es, "ev-b", "b", 4);
    assert_eq!(
        residency(es),
        (vec!["ev-b".to_string()], json!(1), 1),
        "loading ev-b evicted ev-a"
    );

    // the next access rebuilds ev-a from its index: the same revision, exact fingerprint and lookup
    assert_eq!(state(es, "ev-a"), before);
    assert_eq!(lookup(es, "ev-a", "a subject 3"), a_lookup);
    assert_eq!(residency(es), (vec!["ev-a".to_string()], json!(1), 2));

    // a write to the evicted ev-b rebuilds it first and continues at its next revision (2 documents + 2 facts)
    let b2 = write(es, "ev-b", "c", 2);
    assert_eq!(b2, b1 + 4);
    assert_eq!(lookup(es, "ev-b", "b subject 0")["status"], "VALUE");
    assert_eq!(lookup(es, "ev-b", "c subject 1")["status"], "VALUE");
    // and the namespaces stay apart
    assert_eq!(lookup(es, "ev-a", "b subject 0")["status"], "UNKNOWN");
    assert_eq!(state(es, "ev-a"), before);
    assert_eq!(residency(es).2, 4);
}

#[test]
fn without_the_setting_every_projection_stays_loaded() {
    let dir = tempfile::tempdir().unwrap();
    let (_node, es) = boot(dir.path(), None);
    for ns in ["keep-a", "keep-b", "keep-c"] {
        write(es, ns, ns, 2);
    }
    assert_eq!(
        residency(es),
        (
            vec![
                "keep-a".to_string(),
                "keep-b".to_string(),
                "keep-c".to_string()
            ],
            Value::Null,
            0
        )
    );
}
