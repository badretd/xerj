# autoindex ingest harness

Measures `xerj autoindex` end to end against a fresh server of the same binary: time until searchable (from the start
of the client until `GET /<prefix>-*/_count` returns the records the client reported; "active" subtracts the seconds the
guard kept the run stopped), CPU seconds of server and client (and the server's background CPU until it goes idle), peak
RSS of both with the server's split into anonymous (`RssAnon+RssShmem`, what the memory breaker reads) and file-backed
(`RssFile`), bytes on disk (server data dir, client state dir), memory-breaker events (server engagements during the
client run and after it, client shed/push-back lines), and records indexed. Every number in `NEWCORE_MAP.md` section 7
comes from these scripts.

| file | what |
|---|---|
| `run_autoindex.sh LABEL BIN INPUT_DIR OUT_ROOT [flags]` | one run (server + `xerj autoindex` pinned to `CORES`, explicit `MEMCAP_MB`), writes `OUT_ROOT/LABEL/summary.json`; optional server pprof capture (`PROFILE_CPU_S`, needs `--features debug-profiling`); optional client wall cap `CAP_S`; waits for the server's background work (merges) to go idle (`IDLE_CAP_S`, 900) and reports that CPU and memory too |
| `table.py runs/*/summary.json` | the markdown table of `NEWCORE_MAP.md` 7.5 |
| `run_10gb_autoindex.sh OUT LABEL=BIN …` | the 10 GB protocol for a cool node: autoindex row and plain `_bulk` row per binary, 4 cores, thermal guard (SIGSTOP > `T_PAUSE`), MemAvailable abort, `results.json` + `results.md` |
| `procsample.py` | per-thread CPU (by thread name) and RSS timeline (VmRSS, anonymous, file-backed) of server and client from `/proc` (no `perf` needed) |
| `rsswatch.py` | stops the measured server (SIGTERM, then SIGKILL) when its anonymous RSS passes `RSS_STOP_MB` (8192) or MemFree falls below `MEMFREE_STOP_GIB` (10) while it holds ≥ 2 GiB: a shared host must not lose its RAM to a test server; the summary records why |
| `pprof_top.py PROFILE [--components components.tsv]` | reads a pprof `cpu.pb` (pprof-rs) without Go: top functions, modules, and a component split |
| `components.tsv` | component rules for `pprof_top.py` |
| `paused_seconds.py` | seconds a run spent stopped by the thermal / memory guard |
| `column_codec_offline.py CORPUS.jsonl [N]` | offline zstd sizing of the `text` and packed-vector stored columns (the 7.3 codec table; needs `zstandard`) |
| `bulk_load.py` | plain `_bulk` loader for the like-for-like row (`{id, text}` JSONL, N threads, 8 MiB bulks, 429 retry) |

Corpus: `bench_10gb/data/corpus.jsonl` (11,295,654 Wikipedia passages `{id, text}`, 10.48 GB) and slices of its first
lines. A run of `xerj autoindex` on one JSONL file elects `semantic_text` for `text` (average ≥ 80 characters of prose),
so the default path exercises the lexical embedder, its vector companion and the HNSW logic as users get them.

## The 10 GB protocol (for a cool node)

Build one release binary per ref, **each worktree in its own target directory**:

```sh
git worktree add --detach ../x-main 37774bd78                                   # main the fixes branch off
git worktree add --detach ../x-perf perf/fix/engine-query-latency-and-memory    # query/memory fixes
git worktree add --detach ../x-newcore newcore                                  # this branch
for w in x-main x-perf x-newcore; do
  (cd ../$w/engine && cargo build --release -p xerj-server && cp target/release/xerj $HOME/xerj-$w)
done
# check that each binary carries its own code (counts > 0 = present):
#   main: none of the strings below; perf: the first two; newcore: all three
for w in x-main x-perf x-newcore; do
  printf '%s run-budget=%s rssshmem=%s sender=%s\n' $w \
    "$(grep -c -a 'memory circuit breaker has held this run' $HOME/xerj-$w)" \
    "$(grep -c -a 'RssShmem:' $HOME/xerj-$w)" "$(grep -c -a 'autoindex-sender' $HOME/xerj-$w)"
done
```

Do not share one `CARGO_TARGET_DIR` between worktrees of this workspace. Cargo names a workspace crate's artifacts by
its path *relative to the workspace root*, so every worktree writes and reads the same artifact files, and it reuses an
artifact whenever the worktree's own sources are older than it. A second worktree built into the same directory
then links the first worktree's crates. On 2026-10-24 a "main" binary built that way after the perf branch was perf's
crates plus main's `xerj-server`, and the 10-24 "main" rows in `NEWCORE_MAP.md` 7.5 are marked invalid for that
reason. If a shared directory cannot be avoided, `touch` every `*.rs` of the worktree before building it, so that all
of its crates are recompiled, and check the binary as above.

Then, with the corpus copied to the node:

```sh
CORPUS=$HOME/bench10g/data/corpus.jsonl CORES=15,16,17,18 MEMCAP_MB=16384 \
  bash benchmarks/autoindex-ingest/run_10gb_autoindex.sh $HOME/run10g \
  main=$HOME/xerj-x-main perf=$HOME/xerj-x-perf newcore=$HOME/xerj-x-newcore
```

It writes `$HOME/run10g/results.md` (one row per binary and mode: autoindex end to end, plain `_bulk` with 4 client
threads): exit code, records indexed vs expected, wall seconds, CPU seconds (server + client), CPU seconds the server
still spent on background work after the client finished, peak RSS, bytes on disk, thermal pauses. Rows already
measured are skipped on a re-run. `AUTOINDEX_FLAGS=--no-semantic` gives the text-only variant; `MODES=autoindex` skips
the bulk row. Each run is capped at `CAP_S` (4 h); a capped run is reported as such.
