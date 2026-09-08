#!/bin/bash
# Historical execution guard; no server is configured here.
printf '%s\n' 'Historical RESC recipe: execution disabled after infrastructure sanitation. Review a new laboratory-only launcher before running.' >&2
exit 64
# THE GENERATION-RING x L1-CAPACITY CLIFF -- test runner.
#
# Pre-registration: resc-ring-capacity-cliff-prereg.md, SHA-256
#   12f0e7c4539015b64f8aeee181ca0c67b4059257ad9085564ec3c6ac29076cd5
#   frozen 2026-09-04T10:13:52Z, BEFORE any of these cells existed (verified:
#   no cliff dir, no CL=80/96 cell, no W=1 or W=5 run anywhere).
#
# HYPOTHESIS: a producer must hold W x comm_lines marked lines; the L1 holds
# nset x assoc = 4 x 64 = 256 lines. Forwarding works while W*CL <= 256 and
# collapses once W*CL > 256. Retention is IRRELEVANT to this, so every cell here
# runs with retention OFF to remove it as a variable.
#
# TEST A -- move the ring depth, hold comm_lines (the cliff must MOVE):
#   T-A1  W=5 CL=64   -> 320 > 256 -> predict ~0%   (currently 100% at W=3)
#   T-A2  W=1 CL=128  -> 128 <= 256 -> predict HIGH  (currently   0% at W=3)
#   T-A3  W=1 CL=64   ->  64 <= 256 -> predict HIGH  (control)
#   T-A4  W=5 CL=128  -> 640 > 256 -> predict ~0%    (control)
# TEST B -- locate the boundary at W=3 (predicted crossing at 256/3 = 85.3):
#   T-B1  W=3 CL=80   -> 240 <= 256 -> predict HIGH
#   T-B2  W=3 CL=96   -> 288 > 256 -> predict ~0%
#   These two differ by one sweep step and are assigned OPPOSITE outcomes, so
#   they are the sharpest available refutation of the hypothesis.
#
# KILL-C1 fires if T-A1 still forwards >>0 (cliff does not move with W).
# KILL-C2 fires if T-A2 still forwards ~0 (does not recover when made to fit)
#         -- that would be WORSE news than the hypothesis being true.
# KILL-C3 fires if T-B1 and T-B2 land on the same side.
# KILL-C4 fires on BITIDENTICAL=0 or reservation_fails != 0 anywhere.
#
# CONCURRENCY CAP 2, counted across ALL users' sims. NEVER broad-pkill.
set +u
source /legacy-research/owner/miniconda3/etc/profile.d/conda.sh
conda activate clustersim-py311
T=/legacy-research/owner/cudatest/RESC-c2c-wt/c2c_test
cd $T
export PATH=$T/ptxas_shim:/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/legacy-research/owner/cudatest/RESC-c2c-wt/build/lib:$HOME/miniconda3/envs/clustersim-py311/lib:$LD_LIBRARY_PATH

W=$T/retain/cliff
BIN=$T/retain/bin
mkdir -p $W
CFG=${CFG:-cfg_h20_L1_32KB.config}
NCL=2; NS=32; LAG=2; PV=32
TO=${TO:-14400}
MAXJOBS=${MAXJOBS:-2}

running_sims(){ ps -eo comm --no-headers | grep -cE '^(mf_arm|mfr_arm|m_R|rt_arm|mu_arm)'; }
throttle(){ while [ "$(running_sims)" -ge $MAXJOBS ]; do sleep 20; done; }

launch(){  # $1=tag $2=ring_W $3=comm_lines
  local tag=$1 RW=$2 CL=$3 d
  d=$W/${tag}_w${RW}_cl${CL}_pv${PV}
  [ -f "$d/exit.code" ] && [ "$(cat "$d/exit.code")" = "0" ] && return
  [ -f "$d/run.log" ] && [ ! -f "$d/exit.code" ] && { echo "IN-FLIGHT $d"; return; }
  mkdir -p "$d"
  cp $T/$CFG "$d/gpgpusim.config"
  printf "\n-gpgpu_cluster_barrier_invalidate_l1 3\n" >> "$d/gpgpusim.config"
  rm -f "$d/exit.code"
  throttle
  echo "LAUNCH $d  (W*CL=$((RW*CL)) vs 256 lines) ($(date +%H:%M:%S))"
  # RETENTION DELIBERATELY OFF (RESC_RETAIN unset) -- the hypothesis says it is
  # irrelevant here, so it is removed as a variable entirely.
  ( cd "$d" && timeout $TO env RESC_QUIET=1 RESC_GEN_RING=$RW RESC_MW=1 \
      $BIN/rt_arm2 $NCL $NS $LAG $CL $PV > run.log 2>&1; echo $? > exit.code ) &
}

echo "### RING-CAPACITY CLIFF TEST cfg=$CFG ($(date))"
echo "### cfg md5: $(md5sum $T/$CFG | awk '{print $1}')"
echo "### so md5:  $(md5sum /legacy-research/owner/cudatest/RESC-c2c-wt/build/lib/libcudart.so.12 | awk '{print $1}')"
echo "### L1 = 4 sets x 64 ways = 256 lines; predicted cliff at W*CL = 256"

# Test B FIRST: it is the sharpest test (one sweep step apart, opposite predictions).
echo "### TEST B -- boundary location at W=3 ($(date +%H:%M:%S))"
launch TB1 3 80
launch TB2 3 96
wait
# Test A: does the cliff MOVE with W?
echo "### TEST A -- cliff must move with ring depth ($(date +%H:%M:%S))"
launch TA1 5 64
launch TA2 1 128
wait
launch TA3 1 64
launch TA4 5 128
wait
echo "CLIFF_TEST_DONE"
