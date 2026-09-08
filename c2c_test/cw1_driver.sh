#!/bin/bash
# Historical execution guard; no server is configured here.
printf '%s\n' 'Historical RESC recipe: execution disabled after infrastructure sanitation. Review a new laboratory-only launcher before running.' >&2
exit 64
# CW1 pre-registered sweep driver. config/env-only. NO src edits.
source /legacy-research/owner/miniconda3/etc/profile.d/conda.sh; conda activate clustersim-py311 2>/dev/null
ROOT=/legacy-research/owner/cudatest/RESC-c2c-wt
cd $ROOT/c2c_test
export PATH=$PWD/ptxas_shim:/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=$ROOT/build/lib:$HOME/miniconda3/envs/clustersim-py311/lib:${LD_LIBRARY_PATH:-}
BIN=$PWD/pipe_chain
CFG=$PWD/cfg_h20.config
OUT=$PWD/cw1_runs
mkdir -p $OUT
MAXJOBS=${MAXJOBS:-4}

run_cell () {
  local tag=$1 ring=$2 nc=$3 ns=$4 lag=$5 tl=$6
  local d=$OUT/$tag
  mkdir -p $d
  cp $CFG $d/gpgpusim.config
  ( cd $d; RESC_GEN_RING=$ring timeout 5000 $BIN $nc $ns $lag $tl > run.log 2>&1; echo "DONE_CELL $tag rc=$?" ) &
}

throttle () {
  while [ "$(jobs -rp | wc -l)" -ge "$MAXJOBS" ]; do sleep 5; done
}

echo "CW1_SWEEP_START $(date +%F_%T)"

# H1 concurrency: nc in {4,16,64,128,256}, ns=32 tl=4 lag=2, arms 1/2/3
for nc in 4 16 64 128 256; do
  for ring in 1 2 3; do
    throttle
    run_cell "h1_nc${nc}_ring${ring}" $ring $nc 32 2 4
  done
done

# H2 capacity: tl in {16,64,256}, nc=64 ns=32 lag=2, arms 1/2/3
# (tl=4/nc=64 already covered by H1 nc=64)
for tl in 16 64 256; do
  for ring in 1 2 3; do
    throttle
    run_cell "h2_tl${tl}_ring${ring}" $ring 64 32 2 $tl
  done
done

wait
echo "CW1_SWEEP_ALL_DONE $(date +%F_%T)"
