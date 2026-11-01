#!/bin/bash
# One measured `xerj autoindex` run against a fresh server of the same binary.
#   run_autoindex.sh LABEL XERJ_BIN INPUT_DIR OUT_ROOT [extra autoindex flags...]
# env: PORT (19400), CORES (15,16,17,18: server and client share 4 big cores), MEMCAP_MB (8192 -> XERJ_MAX_PROCESS_MEMORY_MB),
#      WORKERS (4), PROFILE_CPU_S / PROFILE_DELAY_S / PROFILE_HZ (server pprof capture; needs a debug-profiling build),
#      GUARD_DIR (registry of guardd.py: thermal 88/85 C and MemFree pause), KEEP_DATA (1 = keep the data dir),
#      CAP_S (0 = none: wall cap of the client), IDLE_CAP_S (900: wait for the server's background work),
#      RSS_STOP_MB (8192: stop the server above this RssAnon+RssShmem), MEMFREE_STOP_GIB (10: stop it when MemFree
#      falls below this while it holds >= 2 GiB anonymous; see rsswatch.py)
# Writes OUT_ROOT/LABEL/{server.log,autoindex.out,autoindex.err,sample.*,summary.json}.
set -u
LABEL=$1; BIN=$2; INPUT=$3; ROOT=$4; shift 4
PORT=${PORT:-19400}; CORES=${CORES:-15,16,17,18}; MEMCAP_MB=${MEMCAP_MB:-8192}; WORKERS=${WORKERS:-4}
H=$(cd "$(dirname "$0")" && pwd)
GUARD_DIR=${GUARD_DIR:-}
W=$ROOT/$LABEL
rm -rf "$W"; mkdir -p "$W/data" "$W/state" "$W/prof"
reg() { [ -n "$GUARD_DIR" ] && echo "$2" > "$GUARD_DIR/reg/$1"; }
# hold every group already registered with the guard (background builds) stopped for the whole timed run
HELD=()
if [ -n "$GUARD_DIR" ] && [ "${HOLD_BUILDS:-1}" = "1" ]; then
  for f in "$GUARD_DIR"/reg/*; do [ -e "$f" ] || continue; g=$(basename "$f"); touch "$GUARD_DIR/hold/$g"; HELD+=("$g"); done
  [ ${#HELD[@]} -gt 0 ] && sleep 2
fi
release_held() { for g in "${HELD[@]}"; do rm -f "$GUARD_DIR/hold/$g"; done; }
trap release_held EXIT
unreg() { [ -n "$GUARD_DIR" ] && rm -f "$GUARD_DIR/reg/$1"; }

prof_env=()
if [ -n "${PROFILE_CPU_S:-}" ]; then
  prof_env=(XERJ_DEBUG_PROFILE_DIR="$W/prof" XERJ_DEBUG_CPU_SECONDS="$PROFILE_CPU_S" XERJ_DEBUG_PROFILE_DELAY_SECONDS="${PROFILE_DELAY_S:-0}" XERJ_DEBUG_CPU_HZ="${PROFILE_HZ:-99}")
fi
env "${prof_env[@]}" XERJ_MAX_PROCESS_MEMORY_MB="$MEMCAP_MB" XERJ_DISABLE_FEEDBACK=true \
  setsid taskset -c "$CORES" nice -n 15 "$BIN" --insecure --data-dir "$W/data" --port "$PORT" > "$W/server.log" 2>&1 < /dev/null &
SPID=$!
reg "$SPID" "xerj server $LABEL"
up=0
for _ in $(seq 1 240); do
  if curl -fs -m1 "http://127.0.0.1:$PORT/_cluster/health" > /dev/null 2>&1; then up=1; break; fi
  kill -0 "$SPID" 2>/dev/null || break
  sleep 0.25
done
if [ $up -ne 1 ]; then echo "server did not come up"; tail -20 "$W/server.log"; kill "$SPID" 2>/dev/null; unreg "$SPID"; exit 1; fi
server_cpu0=$(python3 -c "import os;s=open('/proc/$SPID/stat').read().rsplit(')',1)[1].split();print((int(s[11])+int(s[12]))/os.sysconf('SC_CLK_TCK'))")
python3 "$H/rsswatch.py" --pid "$SPID" --out "$W/stop.json" --anon-mb "${RSS_STOP_MB:-8192}" --memfree-gib "${MEMFREE_STOP_GIB:-10}" &
WATCH=$!

T0=$(date +%s.%N)
cap=()
[ "${CAP_S:-0}" -gt 0 ] && cap=(timeout --foreground -k 60 "$CAP_S")
XERJ_DISABLE_FEEDBACK=true setsid "${cap[@]}" taskset -c "$CORES" nice -n 15 /usr/bin/time -v -o "$W/client.time" \
  "$BIN" autoindex "$INPUT" --url "http://127.0.0.1:$PORT" --prefix t --state-dir "$W/state" --workers "$WORKERS" \
  --max-minutes 0 --progress plain --progress-interval 5 --disable-feedback "$@" > "$W/autoindex.out" 2> "$W/autoindex.err" < /dev/null &
CPID=$!
reg "$CPID" "xerj autoindex $LABEL"
# the sampled client is the child of /usr/bin/time
sleep 0.5
XPID=$(pgrep -P "$CPID" -f "autoindex" | head -1)
[ -n "${cap[*]}" ] && XPID=$(pgrep -P "$(pgrep -P "$CPID" | head -1)" -f "autoindex" | head -1)
SAMPLER_T0=$(date +%s.%N)
python3 "$H/procsample.py" --out "$W/sample" --interval 2 --pid server="$SPID" --pid client="${XPID:-$CPID}" --progress "$W/autoindex.err" \
  --dir data="$W/data" --dir state="$W/state" &
SAMPLER=$!
wait "$CPID"; RC=$?
T1=$(date +%s.%N)
unreg "$CPID"
server_cpu1=$(python3 -c "import os;s=open('/proc/$SPID/stat').read().rsplit(')',1)[1].split();print((int(s[11])+int(s[12]))/os.sysconf('SC_CLK_TCK'))")
server_hwm=$(awk '/VmHWM/{print $2}' /proc/$SPID/status)
# time to searchable: poll the target count until it reaches the records the client reported (cap IDLE_CAP_S)
expect=$(grep -o "records=[0-9]*" "$W/autoindex.err" | tail -1 | cut -d= -f2)
T_SEARCHABLE=""
if [ -n "$expect" ] && [ "$expect" -gt 0 ]; then
  ts0=$(date +%s)
  while [ $(( $(date +%s) - ts0 )) -lt "${IDLE_CAP_S:-900}" ]; do
    c=$(curl -s -m 30 "http://127.0.0.1:$PORT/t-*/_count" | python3 -c "import json,sys;print(json.load(sys.stdin).get('count',0))" 2>/dev/null || echo 0)
    if [ "${c:-0}" -ge "$expect" ]; then T_SEARCHABLE=$(date +%s.%N); break; fi
    kill -0 "$SPID" 2>/dev/null || break
    sleep 1
  done
fi
# background work the run left behind (merges, flushes): wait until the server's CPU stays flat for 10 s (cap IDLE_CAP_S)
IDLE_CAP_S=${IDLE_CAP_S:-900}
idle_t0=$(date +%s); prev=$server_cpu1; flat=0
while [ $(( $(date +%s) - idle_t0 )) -lt "$IDLE_CAP_S" ]; do
  sleep 5
  kill -0 "$SPID" 2>/dev/null || break
  cur=$(python3 -c "import os;s=open('/proc/$SPID/stat').read().rsplit(')',1)[1].split();print((int(s[11])+int(s[12]))/os.sysconf('SC_CLK_TCK'))" 2>/dev/null || echo "$prev")
  if python3 -c "import sys;sys.exit(0 if float('$cur')-float('$prev')<0.5 else 1)"; then flat=$((flat+1)); else flat=0; fi
  prev=$cur
  [ $flat -ge 2 ] && break
done
server_cpu_idle=$prev; idle_after_s=$(( $(date +%s) - idle_t0 ))
server_hwm_idle=$(awk '/VmHWM/{print $2}' /proc/$SPID/status 2>/dev/null || echo 0)
[ -n "$server_hwm_idle" ] || server_hwm_idle=0
curl -s "http://127.0.0.1:$PORT/_cat/indices?format=json&bytes=b" > "$W/indices.json"
curl -s "http://127.0.0.1:$PORT/t-*/_count" > "$W/count_t.json"
curl -s -XPOST "http://127.0.0.1:$PORT/autoindex-catalog/_search" -H 'content-type: application/json' \
  -d '{"size":5,"query":{"term":{"doc_kind":"run"}}}' > "$W/catalog_run.json"
sync
data_bytes=$(du -sb "$W/data" | cut -f1); state_bytes=$(du -sb "$W/state" | cut -f1)
kill "$SAMPLER" 2>/dev/null; wait "$SAMPLER" 2>/dev/null
if [ -n "${PROFILE_CPU_S:-}" ]; then
  # wait for the capture window to close before stopping the server
  for _ in $(seq 1 400); do [ -e "$W/prof/cpu.pb" ] || [ -e "$W/prof/cpu-error.json" ] && break; sleep 1; done
fi
kill -TERM "$SPID" 2>/dev/null
for _ in $(seq 1 120); do kill -0 "$SPID" 2>/dev/null || break; sleep 0.5; done
kill -9 "$SPID" 2>/dev/null
unreg "$SPID"
kill "$WATCH" 2>/dev/null; wait "$WATCH" 2>/dev/null
paused="0 0"
[ -n "$GUARD_DIR" ] && paused=$(python3 "$H/paused_seconds.py" "$GUARD_DIR/guard.log" "${T0%.*}" "$(date +%s)")
paused_search="0 0"
[ -n "$GUARD_DIR" ] && [ -n "$T_SEARCHABLE" ] && paused_search=$(python3 "$H/paused_seconds.py" "$GUARD_DIR/guard.log" "${T0%.*}" "${T_SEARCHABLE%.*}")
python3 - "$W" "$RC" "$T0" "$T1" "$server_cpu0" "$server_cpu1" "$server_hwm" "$data_bytes" "$state_bytes" "$LABEL" "$paused" "$server_cpu_idle" "$idle_after_s" "$server_hwm_idle" "${T_SEARCHABLE:-}" "$paused_search" "$SAMPLER_T0" <<'EOF'
import json, re, sys, datetime
w, rc, t0, t1, c0, c1, hwm, db, sb, label, paused, cidle, idle_after, hwm_idle, tsearch, paused_search, st0 = sys.argv[1:]
def client_time():
    out = {}
    try:
        for line in open(f"{w}/client.time"):
            line = line.strip()
            if line.startswith("User time"): out["user_s"] = float(line.split(":")[-1])
            elif line.startswith("System time"): out["sys_s"] = float(line.split(":")[-1])
            elif line.startswith("Maximum resident set size"): out["peak_rss_mb"] = int(line.split(":")[-1]) / 1024
    except OSError:
        pass
    return out
done = [l.strip() for l in open(f"{w}/autoindex.err", errors="replace") if l.startswith("xerj-done")]
try:
    count = json.load(open(f"{w}/count_t.json")).get("count")
except Exception:
    count = None
try:
    indices = json.load(open(f"{w}/indices.json"))
except Exception:
    indices = []
ct = client_time()
s = {"label": label, "autoindex_rc": int(rc), "xerj_done": done[-1] if done else None,
     "wall_s": round(float(t1) - float(t0), 1),
     "server_cpu_s": round(float(c1) - float(c0), 1), "client_cpu_s": round(ct.get("user_s", 0) + ct.get("sys_s", 0), 1),
     "server_peak_rss_mb": round(int(hwm) / 1024, 1), "client_peak_rss_mb": round(ct.get("peak_rss_mb", 0), 1),
     "server_data_bytes": int(db), "client_state_bytes": int(sb), "t_count": count,
     "indices": [{k: i.get(k) for k in ("index", "docs.count", "store.size")} for i in indices]}
s["cpu_s_total"] = round(s["server_cpu_s"] + s["client_cpu_s"], 1)
s["server_cpu_until_idle_s"] = round(float(cidle) - float(c0), 1)
s["server_idle_after_s"] = int(idle_after)
s["server_peak_rss_until_idle_mb"] = round(int(hwm_idle) / 1024, 1)
try:
    peaks = json.load(open(f"{w}/sample.threads.json")).get("dir_peak_bytes", {})
    s["server_data_peak_bytes"] = peaks.get("data")
    s["client_state_peak_bytes"] = peaks.get("state")
except Exception:
    pass
s["guard_paused_s"], s["guard_pauses"] = float(paused.split()[0]), int(paused.split()[1])
s["wall_minus_paused_s"] = round(s["wall_s"] - s["guard_paused_s"], 1)
m = re.search(r"records=(\d+)", s["xerj_done"] or "")
s["records_reported"] = int(m.group(1)) if m else None
if tsearch:
    s["searchable_wall_s"] = round(float(tsearch) - float(t0), 1)
    s["searchable_active_s"] = round(s["searchable_wall_s"] - float(paused_search.split()[0]), 1)
# peak anonymous (RssAnon+RssShmem) vs file-backed (RssFile) RSS of the server: during the client run, and until idle
run_until = float(t1) - float(st0)
peaks = {"run": [0, 0], "idle": [0, 0]}
try:
    rows = [l.rstrip("\n").split("\t") for l in open(f"{w}/sample.timeline.tsv")]
    head = rows[0]
    ia, if_ = head.index("server_anon_mb"), head.index("server_file_mb")
    for r in rows[1:]:
        if len(r) <= if_ or not r[ia]:
            continue
        a, f = float(r[ia]), float(r[if_])
        for k in (["run", "idle"] if float(r[0]) <= run_until else ["idle"]):
            peaks[k][0] = max(peaks[k][0], a)
            peaks[k][1] = max(peaks[k][1], f)
except (OSError, ValueError, IndexError):
    pass
s["server_peak_anon_mb"], s["server_peak_file_mb"] = peaks["run"]
s["server_peak_anon_until_idle_mb"], s["server_peak_file_until_idle_mb"] = peaks["idle"]
# memory-breaker events: server engagements during the client run / after it; client-side shed and push-back lines
eng_run = eng_after = 0
t1f = float(t1)
for line in open(f"{w}/server.log", errors="replace"):
    if "engaging the parent memory circuit breaker" not in line:
        continue
    line = re.sub(r"\x1b\[[0-9;]*m", "", line)
    try:
        ts = datetime.datetime.strptime(line[:26], "%Y-%m-%dT%H:%M:%S.%f").replace(tzinfo=datetime.timezone.utc).timestamp()
    except ValueError:
        eng_run += 1
        continue
    if ts <= t1f:
        eng_run += 1
    else:
        eng_after += 1
s["breaker_engaged_run"], s["breaker_engaged_after"] = eng_run, eng_after
err = open(f"{w}/autoindex.err", errors="replace").read()
s["client_shed_lines"] = err.count("server is shedding load")
s["client_pushback_lines"] = err.count("server pushed back")
try:
    s["stopped"] = json.load(open(f"{w}/stop.json"))
except (OSError, ValueError):
    s["stopped"] = None
json.dump(s, open(f"{w}/summary.json", "w"), indent=1)
print(json.dumps(s))
EOF
[ "${KEEP_DATA:-0}" = "1" ] || rm -rf "$W/data" "$W/state"
exit "$RC"
