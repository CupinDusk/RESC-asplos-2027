#!/bin/bash
# Historical execution guard; no server is configured here.
printf '%s\n' 'Historical RESC recipe: execution disabled after infrastructure sanitation. Review a new laboratory-only launcher before running.' >&2
exit 64
# WORKING-SET / RATIO TEST. Pre-registration: resc-workingset-prereg.md, SHA-256
#   dd80ae5277a31ddaad044a2b575142f3cb27118baa0987e723abff16ca57d057 (amended
#   premise; every cell here uses W=3 so the ring window CANNOT confound it --
#   forwarding needs W > lag = 2, and W=3 passes).
#
# Q1 = the DECISIVE cell, and it does double duty:
#   (a) ratio test: CL=128 on a 64KB L1 -> 2*CL/L1 = 256/512 = 50%, the SAME
#       ratio as CL=64 on 32KB which forwards 100%. Ratio model predicts ~100%.
#   (b) the UNCONFOUNDED KILL-C2 test: live set fits with 2x headroom at a W
#       that passes the window. At 32KB this same CL=128 forwards 0%.
#   KILL-E1 fires if Q1 still forwards ~0 -> boundary is ABSOLUTE in comm_lines,
#   not a ratio to L1, and I stop proposing capacity models.
# Q2 = control at 75% ratio (CL=192 on 64KB) -> predict ~5%, like CL=96 on 32KB.
# P1 = interpolation at 32KB, CL=112 -> 87.5% ratio -> predict in (0%, 4.76%).
#
# Retention OFF everywhere. Concurrency 2. NEVER broad-pkill.
set +u
source /legacy-research/owner/miniconda3/etc/profile.d/conda.sh
conda activate clustersim-py311
T=/legacy-research/owner/cudatest/RESC-c2c-wt/c2c_test
cd $T
export PATH=$T/ptxas_shim:/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/legacy-research/owner/cudatest/RESC-c2c-wt/build/lib:$HOME/miniconda3/envs/clustersim-py311/lib:$LD_LIBRARY_PATH
W=$T/retain/ws; BIN=$T/retain/bin; mkdir -p $W
NCL=2; NS=32; LAG=2; PV=32; RW=3; TO=14400; MAXJOBS=2
running(){ ps -eo comm --no-headers|grep -cE "^(mf_arm|mfr_arm|m_R|rt_arm|mu_arm)"; }
throttle(){ while [ "$(running)" -ge $MAXJOBS ]; do sleep 20; done; }
launch(){ # tag cfg CL
  local tag=$1 cfg=$2 CL=$3 d=$W/${1}_cl${3}
  [ -f "$d/exit.code" ] && [ "$(cat $d/exit.code)" = "0" ] && return
  [ -f "$d/run.log" ] && [ ! -f "$d/exit.code" ] && { echo "IN-FLIGHT $d"; return; }
  mkdir -p "$d"; cp $T/$cfg "$d/gpgpusim.config"
  printf "\n-gpgpu_cluster_barrier_invalidate_l1 3\n" >> "$d/gpgpusim.config"
  rm -f "$d/exit.code"; throttle
  echo "LAUNCH $tag cfg=$cfg W=$RW CL=$CL ($(date +%H:%M:%S))"
  ( cd "$d" && timeout $TO env RESC_QUIET=1 RESC_GEN_RING=$RW RESC_MW=1 \
      $BIN/rt_arm2 $NCL $NS $LAG $CL $PV > run.log 2>&1; echo $? > exit.code ) &
}
echo "### WORKING-SET TEST ($(date))"
echo "### so md5: $(md5sum /legacy-research/owner/cudatest/RESC-c2c-wt/build/lib/libcudart.so.12|awk "{print \$1}")"
echo "### 64KB cfg md5: $(md5sum $T/cfg_h20_L1_64KB.config|awk "{print \$1}")"
launch Q1 cfg_h20_L1_64KB.config 128
launch Q2 cfg_h20_L1_64KB.config 192
wait
launch P1 cfg_h20_L1_32KB.config 112
wait
echo WS_TEST_DONE
