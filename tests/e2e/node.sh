#!/bin/bash
# The tests on the run host, detached. CPU only, nice 15 through the server, cores CPUS. Finished steps are skipped; one copy (lock).
#   bash node.sh DIR     DIR holds bin/xerj, tests/, sample.json, select_ids.txt, model -> export, worlds -> model_dev.jsonl
D=$1; cd "$D" || exit 2
: "${CPUS:=12-19}" "${T5_MIN:=10}" "${FUNC_ONLY:=1,2,3,4,5}"
exec 9>"$D/lock"; flock -n 9 || { echo already running; exit 1; }
log(){ echo "$(date '+%F %T %Z') $*" >> "$D/progress.log"; }
step(){ n=$1; shift; [ -f "$D/$n.ok" ] && return 0; log "start $n"; taskset -c "$CPUS" nice -n 15 "$@" > "$D/$n.log" 2>&1; rc=$?; log "$n rc=$rc"; [ $rc = 0 ] && touch "$D/$n.ok"; return $rc; }
PY="python3 -B"; T=$D/tests; COMMON="--bin $D/bin/xerj --model $D/model --worlds $D/worlds"
step functional $PY $T/e2e_functional.py $COMMON --sample $D/sample.json --out $D/func --cpus $CPUS --only "$FUNC_ONLY" --t5-minutes "$T5_MIN"
echo "functional rc=$?" >> "$D/rc.log"
step bench_load $PY $T/e2e_bench.py $COMMON --ids $D/select_ids.txt --out $D/bench --phase load --cpus $CPUS || { log "FAILED bench_load"; touch "$D/DONE"; exit 1; }
step bench_w8 $PY $T/e2e_bench.py $COMMON --ids $D/select_ids.txt --out $D/bench --phase ask --precision unset --cpus $CPUS --conc ${CONC:-8} --threads 1
step bench_f32 $PY $T/e2e_bench.py $COMMON --ids $D/select_ids.txt --out $D/bench --phase ask --precision f32 --cpus $CPUS --conc ${CONC:-8} --threads 1
L=$(echo "$CPUS" | sed 's/-.*//'); L4=$L-$((L+3))
step latency_w8_t4 $PY $T/e2e_bench.py $COMMON --ids $D/select_ids.txt --out $D/bench --phase latency --precision unset --cpus $L4 --threads 4
step latency_f32_t4 $PY $T/e2e_bench.py $COMMON --ids $D/select_ids.txt --out $D/bench --phase latency --precision f32 --cpus $L4 --threads 4
step latency_w8_t1 $PY $T/e2e_bench.py $COMMON --ids $D/select_ids.txt --out $D/bench --phase latency --precision unset --cpus $L --threads 1
log DONE; touch "$D/DONE"
