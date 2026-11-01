#!/bin/bash
# One command for the facts end-to-end tests: bash tests/e2e/run.sh [TAG]     (from any directory; TAG = run name, default "main")
# Builds the release server from the committed heads of dev/e2e1 (git archive, so nothing uncommitted counts), copies it to the run host,
# runs tests 1 to 7 there detached, pulls the results and scores test 6 here, where the labels are. Safe to rerun: finished steps are
# skipped, a run that died resumes. To repeat a test, use a new TAG, or delete its .ok marker on the run host.
# env: RUN_HOST BUILD_HOST (both required) CPUS (12-19) CONC (8) BIN (a built server here: skips the build) MVP (research tree) MODEL (export on the run host) COUNTING_WORLDS (a dev worlds file here: runs test 8) T5_MIN (10) FUNC_ONLY (1,2,3,4,5)
set -u
TAG=${1:-main}
HERE=$(cd "$(dirname "$0")" && pwd); PROD=$(cd "$HERE/../.." && pwd); MEM=$(cd "$PROD/../xerj_mem" && pwd)
MVP=${MVP:?the research tree}
RUN_HOST=${RUN_HOST:?ssh host that runs the tests}; BUILD_HOST=${BUILD_HOST:?ssh host that builds}; CPUS=${CPUS:-12-19}
BASE='xerj-extractor/engine_e2e1'; RD=$BASE/e2e/$TAG
MODEL=${MODEL:-xerj-extractor/engine_e2e1/s1/export/x68_cC_s29}
WORLDS=xerj-extractor/research/xerj_r_mvp/runs/world/hybrid_v2_bench/model_dev.jsonl
OUT=$MVP/runs/e2e1/s5/$TAG; mkdir -p "$OUT"
log(){ echo "$(date '+%F %T %Z') $*" | tee -a "$OUT/run.log"; }
HP=$(git -C "$PROD" rev-parse HEAD); HM=$(git -C "$MEM" rev-parse HEAD)
log "heads product $HP xerj_mem $HM"
[ -f "$OUT/select_ids.txt" ] || python3 -I -B -c "import json,sys; print('\n'.join(k for k, v in json.load(open(sys.argv[1]))['items'].items() if v == 'select'))" "$MVP/runs/decline_v1/split.json" > "$OUT/select_ids.txt" || exit 3
[ -f "$MVP/runs/e2e1/s5/sample.json" ] || python3 -I -B "$HERE/make_sample.py" --mvp "$MVP" --out "$MVP/runs/e2e1/s5/sample.json" || exit 3

if [ -z "${BIN:-}" ]; then
# build
ssh "$BUILD_HOST" "mkdir -p $BASE/snap_e2e/product $BASE/snap_e2e/xerj_mem; cat $BASE/e2e_build/head 2>/dev/null" > "$OUT/.head"
if [ "$(cat "$OUT/.head")" != "$HP $HM" ]; then
  log "build on $BUILD_HOST"
  git -C "$PROD" archive HEAD engine | ssh "$BUILD_HOST" "tar -x -C $BASE/snap_e2e/product"
  git -C "$MEM" archive HEAD | ssh "$BUILD_HOST" "tar -x -C $BASE/snap_e2e/xerj_mem"
  scp -q "$HERE/build_e2e.sh" "$BUILD_HOST:$BASE/build_e2e.sh"
  ssh "$BUILD_HOST" "rm -f $BASE/e2e_build/BUILD_DONE; cd $BASE; nohup setsid bash build_e2e.sh > e2e_build.out 2>&1 < /dev/null &"
fi
until ssh "$BUILD_HOST" "test -f $BASE/e2e_build/BUILD_DONE"; do sleep 20; done
ssh "$BUILD_HOST" "head -1 $BASE/e2e_build/build.rc" | grep -qx 0 || { log "BUILD FAILED, see $BUILD_HOST:$BASE/e2e_build/build.log"; exit 4; }
ssh "$BUILD_HOST" "echo '$HP $HM' > $BASE/e2e_build/head"

fi

# copy and run, detached
ssh "$RUN_HOST" "mkdir -p $RD/bin $RD/tests; ln -sfn \$HOME/$MODEL $RD/model; ln -sfn \$HOME/$WORLDS $RD/worlds"
if [ -n "${BIN:-}" ]; then ssh "$RUN_HOST" "test -f $RD/bin/xerj" || scp -q "$BIN" "$RUN_HOST:$RD/bin/xerj"
else ssh "$RUN_HOST" "test -f $RD/bin/xerj" || scp -q -3 "$BUILD_HOST:$BASE/target_e2e/release/xerj" "$RUN_HOST:$RD/bin/xerj"; fi
scp -q "$HERE"/e2e_lib.py "$HERE"/e2e_functional.py "$HERE"/e2e_bench.py "$HERE"/e2e_counting.py "$HERE"/node.sh "$RUN_HOST:$RD/tests/"
scp -q "$MVP/runs/e2e1/s5/sample.json" "$RUN_HOST:$RD/sample.json"
scp -q "$OUT/select_ids.txt" "$RUN_HOST:$RD/select_ids.txt"
[ -n "${COUNTING_WORLDS:-}" ] && scp -q "$COUNTING_WORLDS" "$RUN_HOST:$RD/counting.jsonl"
ssh "$RUN_HOST" "test -f $RD/DONE || { cd $RD; CPUS=$CPUS CONC=${CONC:-8} T5_MIN=${T5_MIN:-10} FUNC_ONLY=${FUNC_ONLY:-1,2,3,4,5} nohup setsid bash tests/node.sh \$PWD > driver.out 2>&1 < /dev/null & }"
log "running on $RUN_HOST:$RD (progress.log there)"
until ssh "$RUN_HOST" "test -f $RD/DONE"; do sleep 60; done

# pull, score
for p in progress.log func/*.json func/*.log bench/*.json bench/*.rows.jsonl '*.log'; do
  mkdir -p "$OUT/pulled/$(dirname "$p")"; scp -q "$RUN_HOST:$RD/$p" "$OUT/pulled/$(dirname "$p")/" 2>/dev/null
done
python3 -I -B "$HERE/e2e_score.py" --mvp "$MVP" --http "$OUT/pulled/bench" --out "$OUT/score" | tee "$OUT/score.txt"
python3 -I -B "$HERE/summary.py" "$OUT" | tee "$OUT/summary.txt"
