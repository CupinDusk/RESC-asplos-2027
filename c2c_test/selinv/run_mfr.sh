#!/bin/bash
# Historical execution guard; no server is configured here.
printf '%s\n' 'Historical RESC recipe: execution disabled after infrastructure sanitation. Review a new laboratory-only launcher before running.' >&2
exit 64
# CROSSED CELL: marked fraction AND reuse together.
# Pre-registration: markfrac-reuse-prereg.md, SHA-256 2a1d772392e7d187...,
# frozen 2026-09-03T17:29:08Z, BEFORE any reuse run existed.
#
# DECISION RULE ORDER MATTERS: R=1 is the ABORT-CHECK and must be evaluated FIRST.
# It must reproduce arm0@0 = 114006 and arm3@3 = 114919 EXACTLY (deterministic
# simulator, 0-cycle tolerance).  This script therefore runs R=1 alone, and the
# R>=2 sweep is launched separately only after that check passes.
#
# Fixed at the FA-4 anchor: priv_lines=32 (f=0.1111), comm_lines=4, NC=2, NS=32, LAG=2.
# Fairness rule: baselines at knobs {0,1}; RESC arms at knobs {2,3}; knob 0 is NOT
# legal for a RESC arm.  arm2/arm3 are the same binary, discipline set by env.
#
# CONCURRENCY CAP 3, counted across every mf_arm/mfr_arm process on the shared box.
set +u
source /legacy-research/owner/miniconda3/etc/profile.d/conda.sh
conda activate clustersim-py311
T=/legacy-research/owner/cudatest/RESC-c2c-wt/c2c_test
cd $T
export PATH=$T/ptxas_shim:/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/legacy-research/owner/cudatest/RESC-c2c-wt/build/lib:$HOME/miniconda3/envs/clustersim-py311/lib:$LD_LIBRARY_PATH
W=$T/selinv/mfr
BIN=$W/bin
mkdir -p $W

NCL=2; NS=32; LAG=2; CL=4; PV=32
TO=9000
MAXJOBS=3
RLIST="${RLIST:-1}"

running_sims(){ ps -eo comm --no-headers | grep -cE '^(mf_arm|mfr_arm)'; }
throttle(){ while [ "$(running_sims)" -ge $MAXJOBS ]; do sleep 10; done; }

launch(){  # $1=knob $2=R $3=arm
  local inv=$1 R=$2 arm=$3 bin G MW d
  case $arm in
    arm0) bin=mfr_arm0; G=1; MW=0;;
    arm1) bin=mfr_arm1; G=0; MW=0;;
    arm2) bin=mfr_arm2; G=2; MW=0;;
    arm3) bin=mfr_arm2; G=3; MW=1;;
  esac
  d=$W/k${inv}_R${R}_${arm}
  [ -f $d/exit.code ] && [ "$(cat $d/exit.code)" = "0" ] && return
  [ -f $d/run.log ] && [ ! -f $d/exit.code ] && { echo "IN-FLIGHT $d"; return; }
  mkdir -p $d
  cp $T/cfg_h20_l2_60mb.config $d/gpgpusim.config
  printf "\n-gpgpu_cluster_barrier_invalidate_l1 %s\n" $inv >> $d/gpgpusim.config
  rm -f $d/exit.code
  throttle
  echo "LAUNCH $d ($(date +%H:%M:%S))"
  ( cd $d && timeout $TO env RESC_QUIET=1 RESC_GEN_RING=$G RESC_MW=$MW \
      $BIN/$bin $NCL $NS $LAG $CL $PV $R > run.log 2>&1; echo $? > exit.code ) &
}

for R in $RLIST; do
  echo "### R=$R ($(date +%H:%M:%S))"
  for arm in arm0 arm1; do for inv in 0 1; do launch $inv $R $arm; done; done
  for arm in arm2 arm3; do for inv in 2 3; do launch $inv $R $arm; done; done
done
wait
echo "MFR_DONE RLIST=$RLIST"
