#!/bin/bash
# Release xerj-server --features facts from git archives on the build host. nice 15, 4 threads, cores 0-3, CPU only. One copy (lock).
R=$HOME/xerj-extractor/engine_e2e1; S=$R/snap_e2e; O=$R/e2e_build; mkdir -p $O
exec 7>$O/lock; flock -n 7 || { echo already running; exit 1; }
export PATH=$HOME/.cargo/bin:$PATH CARGO_NET_OFFLINE=true
[ -d $R/target_e2e ] || cp -al $R/target_product_head $R/target_e2e
cd $S/product/engine
export CARGO_HOME=$R/cargo_home_product RUSTUP_HOME=$HOME/.rustup CARGO_TARGET_DIR=$R/target_e2e
nice -n 15 taskset -c 0-3 cargo build --release --locked --offline -p xerj-server --features facts -j 4 > $O/build.log 2>&1
echo $? > $O/build.rc; sha256sum $R/target_e2e/release/xerj >> $O/build.rc; date >> $O/build.rc; touch $O/BUILD_DONE
