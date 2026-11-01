#!/bin/bash
# 10 GB ingest protocol for one or more `xerj` product binaries (autoindex end to end + a plain `_bulk` row), meant for a
# cool node. Nothing is built here: pass binaries built from the branches to compare.
#
#   bash run_10gb_autoindex.sh OUT_DIR LABEL=/path/to/xerj [LABEL=/path/to/xerj ...]
#
# env (defaults in brackets):
#   CORPUS   [required]  11,295,654 passages {id,text}
#   N_MAIN   [11295614]  passages indexed (the first N_MAIN lines; the bench10g protocol keeps 40 for one-at-a-time writes)
#   CORES    [15,16,17,18]  server AND client pinned to these 4 big cores (as Lucene / tantivy / rc.98 were)
#   MEMCAP_MB [16384]  XERJ_MAX_PROCESS_MEMORY_MB for every server (rc.98's auto tier on a 128 GB node was 16384)
#   WORKERS  [4]       autoindex --workers
#   MODES    [autoindex,bulk]   which rows to run per binary (bulk = bulk_load.py, 4 client threads, 8 MiB bulks)
#   AUTOINDEX_FLAGS []  extra flags for every autoindex run (e.g. "--no-semantic")
#   CAP_S    [14400]   wall-clock cap per run (timeout --foreground; a capped run is reported as such)
#   T_PAUSE / T_RESUME [90000 / 85000]  thermal rule (millidegrees): SIGSTOP the run's process groups above T_PAUSE,
#                                        SIGCONT below T_RESUME; pauses are logged and counted in the results
#   MEMAVAIL_MIN_GIB [10]  abort a run (SIGTERM) when MemAvailable falls below this, recorded as mem_abort
#   PORT     [19500]
# Output: OUT_DIR/<label>.<mode>/ (logs, monitor timeline, summary.json) and OUT_DIR/results.json + results.md.
# Idempotent: a row whose summary.json exists is skipped (delete it to re-run).
set -u
OUT=${1:?usage: run_10gb_autoindex.sh OUT_DIR LABEL=BIN ...}; shift
[ $# -ge 1 ] || { echo "need at least one LABEL=BIN"; exit 2; }
H=$(cd "$(dirname "$0")" && pwd)
CORPUS=${CORPUS:?path to corpus.jsonl}
N_MAIN=${N_MAIN:-11295614}; CORES=${CORES:-15,16,17,18}; MEMCAP_MB=${MEMCAP_MB:-16384}; WORKERS=${WORKERS:-4}
MODES=${MODES:-autoindex,bulk}; AUTOINDEX_FLAGS=${AUTOINDEX_FLAGS:-}; CAP_S=${CAP_S:-14400}
T_PAUSE=${T_PAUSE:-90000}; T_RESUME=${T_RESUME:-85000}; MEMAVAIL_MIN_GIB=${MEMAVAIL_MIN_GIB:-10}; PORT=${PORT:-19500}
mkdir -p "$OUT"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$OUT/progress.log"; }

# 1. input: one regular file holding the first N_MAIN passages (autoindex refuses a symlink that leaves the folder)
IN=$OUT/input
if [ ! -s "$IN/corpus.jsonl" ] || [ "$(wc -l < "$IN/corpus.jsonl")" -ne "$N_MAIN" ]; then
  mkdir -p "$IN"; log "writing $IN/corpus.jsonl (first $N_MAIN lines of $CORPUS)"
  head -n "$N_MAIN" "$CORPUS" > "$IN/corpus.jsonl.part" && mv "$IN/corpus.jsonl.part" "$IN/corpus.jsonl"
fi
log "input $(stat -c %s "$IN/corpus.jsonl") bytes, $N_MAIN passages; cores $CORES; memcap ${MEMCAP_MB} MiB; modes $MODES"

hottest() { cat /sys/class/thermal/thermal_zone*/temp 2>/dev/null | sort -n | tail -1; }
memavail_gib() { awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo; }
cpu_s() { python3 -c "import os,sys;s=open('/proc/%s/stat'%sys.argv[1]).read().rsplit(')',1)[1].split();print((int(s[11])+int(s[12]))/os.sysconf('SC_CLK_TCK'))" "$1" 2>/dev/null || echo 0; }

# guard GROUPS... : thermal + memory rule over the given process groups until the first group exits.
guard_loop() {
  local paused=0 pauses=0 max=0 t
  while kill -0 "$1" 2>/dev/null; do
    t=$(hottest); [ "${t:-0}" -gt "$max" ] && max=$t
    if [ $paused -eq 0 ] && [ "${t:-0}" -gt "$T_PAUSE" ]; then
      for g in "$@"; do kill -STOP -"$g" 2>/dev/null; done; paused=1; pauses=$((pauses + 1))
      echo "$(date '+%F %T') pause at $((t / 1000)) C" >> "$W/thermal.log"
    elif [ $paused -eq 1 ] && [ "${t:-0}" -lt "$T_RESUME" ]; then
      for g in "$@"; do kill -CONT -"$g" 2>/dev/null; done; paused=0
      echo "$(date '+%F %T') resume at $((t / 1000)) C" >> "$W/thermal.log"
    fi
    if [ "$(memavail_gib)" -lt "$MEMAVAIL_MIN_GIB" ]; then
      echo "$(date '+%F %T') MemAvailable below ${MEMAVAIL_MIN_GIB} GiB: aborting" >> "$W/thermal.log"
      echo 1 > "$W/mem_abort"
      for g in "$@"; do kill -CONT -"$g" 2>/dev/null; kill -TERM -"$g" 2>/dev/null; done
      break
    fi
    sleep 1
  done
  echo "{\"thermal_pauses\": $pauses, \"max_zone_c\": $((max / 1000))}" > "$W/guard.json"
}

start_server() {   # start_server BIN DATA_DIR LOG -> sets SPID
  env XERJ_MAX_PROCESS_MEMORY_MB="$MEMCAP_MB" XERJ_DISABLE_FEEDBACK=true \
    setsid taskset -c "$CORES" "$1" --insecure --data-dir "$2" --port "$PORT" > "$3" 2>&1 < /dev/null &
  SPID=$!
  for _ in $(seq 1 480); do
    curl -fs -m1 "http://127.0.0.1:$PORT/_cluster/health" > /dev/null 2>&1 && return 0
    kill -0 "$SPID" 2>/dev/null || return 1
    sleep 0.25
  done
  return 1
}
stop_server() { kill -TERM "$SPID" 2>/dev/null; for _ in $(seq 1 240); do kill -0 "$SPID" 2>/dev/null || break; sleep 0.5; done; kill -9 "$SPID" 2>/dev/null; }

for spec in "$@"; do
  LABEL=${spec%%=*}; BIN=${spec#*=}
  [ -x "$BIN" ] || { log "skip $LABEL: $BIN is not executable"; continue; }
  for MODE in ${MODES//,/ }; do
    W=$OUT/$LABEL.$MODE
    [ -s "$W/summary.json" ] && { log "$LABEL.$MODE done already"; continue; }
    while [ "$(hottest)" -gt "$T_RESUME" ]; do sleep 10; done   # start cool
    rm -rf "$W"; mkdir -p "$W/data" "$W/state"
    log "$LABEL.$MODE: $("$BIN" --version 2>&1 | head -1)"
    start_server "$BIN" "$W/data" "$W/server.log" || { log "$LABEL.$MODE: server did not start"; stop_server; continue; }
    c0=$(cpu_s "$SPID"); T0=$(date +%s.%N)
    if [ "$MODE" = autoindex ]; then
      # shellcheck disable=SC2086
      XERJ_DISABLE_FEEDBACK=true setsid timeout --foreground -k 120 "$CAP_S" taskset -c "$CORES" /usr/bin/time -v -o "$W/client.time" \
        "$BIN" autoindex "$IN" --url "http://127.0.0.1:$PORT" --prefix ax --state-dir "$W/state" --workers "$WORKERS" \
        --max-minutes 0 --progress plain --progress-interval 30 --disable-feedback $AUTOINDEX_FLAGS \
        > "$W/autoindex.out" 2> "$W/autoindex.err" < /dev/null &
    else
      setsid timeout --foreground -k 120 "$CAP_S" taskset -c "$CORES" /usr/bin/time -v -o "$W/client.time" \
        python3 -I "$H/bulk_load.py" --url "http://127.0.0.1:$PORT" --index bench --corpus "$IN/corpus.jsonl" \
        --threads 4 --bulk-mb 8 --out "$W/bulk.json" > "$W/bulk.out" 2> "$W/bulk.err" < /dev/null &
    fi
    CPID=$!
    # the sampled client is the process doing the work (under timeout and /usr/bin/time)
    XPID=""
    for _ in $(seq 1 40); do
      XPID=$(pgrep -n -f -- "--state-dir $W/state" || pgrep -n -f -- "--out $W/bulk.json" || true)
      [ -n "$XPID" ] && break; sleep 0.25
    done
    python3 "$H/procsample.py" --out "$W/sample" --interval 30 --pid server="$SPID" --pid client="${XPID:-$CPID}" --progress "$W/autoindex.err" \
      --dir data="$W/data" --dir state="$W/state" &
    SAMP=$!
    guard_loop "$CPID" "$SPID" &
    GUARD=$!
    wait "$CPID"; RC=$?
    T1=$(date +%s.%N)
    wait "$GUARD" 2>/dev/null
    c1=$(cpu_s "$SPID"); hwm=$(awk '/VmHWM/{print $2}' "/proc/$SPID/status" 2>/dev/null || echo 0)
    # background work the run left behind (flushes, merges): until the server's CPU stays flat for 20 s (cap IDLE_CAP_S)
    idle_t0=$(date +%s); prev=$c1; flat=0
    while [ $(( $(date +%s) - idle_t0 )) -lt "${IDLE_CAP_S:-1800}" ]; do
      sleep 10; cur=$(cpu_s "$SPID")
      if python3 -c "import sys;sys.exit(0 if float('$cur')-float('$prev')<1.0 else 1)"; then flat=$((flat+1)); else flat=0; fi
      prev=$cur; [ $flat -ge 2 ] && break
    done
    c_idle=$prev; idle_after=$(( $(date +%s) - idle_t0 ))
    if [ "$MODE" = autoindex ]; then
      curl -s -XPOST "http://127.0.0.1:$PORT/ax-*/_refresh" > /dev/null
      curl -s "http://127.0.0.1:$PORT/ax-*/_count" > "$W/count.json"
    else
      curl -s "http://127.0.0.1:$PORT/bench/_count" > "$W/count.json"
    fi
    curl -s "http://127.0.0.1:$PORT/_cat/indices?format=json&bytes=b" > "$W/indices.json"
    kill "$SAMP" 2>/dev/null; wait "$SAMP" 2>/dev/null
    stop_server; sync
    data_b=$(du -sb "$W/data" | cut -f1); state_b=$(du -sb "$W/state" | cut -f1)
    python3 - "$W" "$LABEL" "$MODE" "$RC" "$T0" "$T1" "$c0" "$c1" "${hwm:-0}" "$data_b" "$state_b" "$N_MAIN" "$c_idle" "$idle_after" <<'EOF'
import json, os, sys
w, label, mode, rc, t0, t1, c0, c1, hwm, db, sb, n, cidle, idle_after = sys.argv[1:]
def client_time():
    out = {}
    try:
        for line in open(os.path.join(w, "client.time")):
            line = line.strip()
            if line.startswith("User time"): out["user_s"] = float(line.split(":")[-1])
            elif line.startswith("System time"): out["sys_s"] = float(line.split(":")[-1])
            elif line.startswith("Maximum resident set size"): out["rss_mb"] = int(line.split(":")[-1]) / 1024
    except OSError:
        pass
    return out
def jload(name):
    try: return json.load(open(os.path.join(w, name)))
    except Exception: return {}
ct = client_time()
done = []
if os.path.exists(os.path.join(w, "autoindex.err")):
    done = [l.strip() for l in open(os.path.join(w, "autoindex.err"), errors="replace") if l.startswith("xerj-done")]
s = {"label": label, "mode": mode, "rc": int(rc), "capped": int(rc) == 124, "mem_abort": os.path.exists(os.path.join(w, "mem_abort")),
     "wall_s": round(float(t1) - float(t0), 1), "server_cpu_s": round(float(c1) - float(c0), 1),
     "client_cpu_s": round(ct.get("user_s", 0) + ct.get("sys_s", 0), 1), "server_peak_rss_mb": round(int(hwm) / 1024, 1),
     "client_peak_rss_mb": round(ct.get("rss_mb", 0), 1), "server_data_bytes": int(db), "client_state_bytes": int(sb),
     "count": jload("count.json").get("count"), "expected": int(n), "xerj_done": done[-1] if done else None,
     "guard": jload("guard.json"), "bulk": jload("bulk.json") or None}
s["cpu_s_total"] = round(s["server_cpu_s"] + s["client_cpu_s"], 1)
s["server_cpu_until_idle_s"] = round(float(cidle) - float(c0), 1)
s["server_idle_after_s"] = int(idle_after)
peaks = jload("sample.threads.json").get("dir_peak_bytes", {})
s["server_data_peak_bytes"], s["client_state_peak_bytes"] = peaks.get("data"), peaks.get("state")
s["all_indexed"] = s["count"] == s["expected"]
# anonymous (RssAnon+RssShmem, what the memory breaker reads) vs file-backed peak over the run and its idle tail
srv = jload("sample.threads.json").get("server", {})
s["server_peak_anon_mb"], s["server_peak_file_mb"] = srv.get("peak_rss_anon_mb"), srv.get("peak_rss_file_mb")
try:
    log = open(os.path.join(w, "server.log"), errors="replace").read()
    s["breaker_engaged"] = log.count("engaging the parent memory circuit breaker")
except OSError:
    s["breaker_engaged"] = None
try:
    err = open(os.path.join(w, "autoindex.err"), errors="replace").read()
    s["client_shed_lines"] = err.count("server is shedding load")
except OSError:
    s["client_shed_lines"] = None
json.dump(s, open(os.path.join(w, "summary.json"), "w"), indent=1)
print(json.dumps(s))
EOF
    log "$LABEL.$MODE rc=$RC: $(cat "$W/summary.json" | tr -d '\n' | cut -c1-400)"
    rm -rf "$W/data" "$W/state"
  done
done

python3 - "$OUT" <<'EOF'
import glob, json, os, sys
out = sys.argv[1]
rows = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(out, "*", "summary.json")))]
json.dump(rows, open(os.path.join(out, "results.json"), "w"), indent=1)
with open(os.path.join(out, "results.md"), "w") as f:
    f.write("| run | rc | all indexed | wall s | CPU s (server + client) | server CPU s until idle | peak RSS server / client MB | server peak anon / file MB | breaker engaged (client shed) | server data GB | client state GB | thermal pauses |\n")
    f.write("|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for r in rows:
        f.write(f"| {r['label']}.{r['mode']} | {r['rc']} | {r['all_indexed']} ({r['count']}/{r['expected']}) | {r['wall_s']} | "
                f"{r['cpu_s_total']} ({r['server_cpu_s']} + {r['client_cpu_s']}) | {r.get('server_cpu_until_idle_s')} | "
                f"{r['server_peak_rss_mb']} / {r['client_peak_rss_mb']} | "
                f"{r.get('server_peak_anon_mb')} / {r.get('server_peak_file_mb')} | "
                f"{r.get('breaker_engaged')} ({r.get('client_shed_lines')}) | "
                f"{r['server_data_bytes'] / 1e9:.2f} | {r['client_state_bytes'] / 1e9:.2f} | {r.get('guard', {}).get('thermal_pauses')} |\n")
print(open(os.path.join(out, "results.md")).read())
EOF
