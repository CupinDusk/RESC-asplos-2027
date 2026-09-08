#!/bin/bash
# Historical execution guard; no server is configured here.
printf '%s\n' 'Historical RESC recipe: execution disabled after infrastructure sanitation. Review a new laboratory-only launcher before running.' >&2
exit 64
# MARKED-FRACTION SWEEP -- RESUME, informative points first.
# Pre-registration: markfrac-prereg.md (SHA-256 3ebf6393..., frozen 2026-09-03T13:06:21Z)
#   + addendum markfrac-prereg-addendum-01.md (b02af321..., frozen 2026-09-03T13:17:24Z)
#     which adds priv_lines=32 => f=0.1111, the FlashAttention-4 2-CTA real-kernel anchor.
#
# NOTHING about the experiment changes here -- same points, same arms, same knobs,
# same fixed communication, same fairness rule.  ONLY the ORDER changes: the original
# runner put priv_lines=1024 (the slowest and least informative point) first, which
# front-loaded ~2.6h before any informative point.  This resumes with the real-kernel
# anchor and the mid-sweep points first.
#
# NO-CLOBBER: any run with exit.code == 0 is SKIPPED, so the 3 priv=1024 runs still
# executing (detached, reparented) are neither duplicated nor disturbed.  Their
# subshells write their own exit.code.
#
# CONCURRENCY CAP 3 -- counts ALL mf_arm processes on the box, including the ones
# this script did not start, so the cap holds across the handover.
set +u
source /legacy-research/owner/miniconda3/etc/profile.d/conda.sh
conda activate clustersim-py311
T=/legacy-research/owner/cudatest/RESC-c2c-wt/c2c_test
cd $T
export PATH=$T/ptxas_shim:/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/legacy-research/owner/cudatest/RESC-c2c-wt/build/lib:$HOME/miniconda3/envs/clustersim-py311/lib:$LD_LIBRARY_PATH
W=$T/selinv/mf/sweep
BIN=$T/selinv/mf/bin
mkdir -p $W

NCL=2; NS=32; LAG=2; CL=4      # communication: FIXED for every point
TO=9000
MAXJOBS=3

# Count every simulator process on the box, not just our children.
running_sims(){ ps -eo comm --no-headers | grep -c '^mf_arm'; }
throttle(){ while [ "$(running_sims)" -ge $MAXJOBS ]; do sleep 10; done; }

launch(){  # $1=knob $2=priv_lines $3=arm
  local inv=$1 pv=$2 arm=$3 bin G MW d
  case $arm in
    arm0) bin=mf_arm0; G=1; MW=0;;
    arm1) bin=mf_arm1; G=0; MW=0;;
    arm2) bin=mf_arm2; G=2; MW=0;;
    arm3) bin=mf_arm2; G=3; MW=1;;   # LAG=2 -> G=LAG+1=3
  esac
  d=$W/k${inv}_v${pv}_${arm}
  if [ -f $d/exit.code ] && [ "$(cat $d/exit.code)" = "0" ]; then
    echo "SKIP (done) $d"; return
  fi
  # a run still in flight (dir exists, no exit.code, log growing) must not be duplicated
  if pgrep -f "$BIN/$bin $NCL $NS $LAG $CL $pv\$" > /dev/null 2>&1; then
    echo "SKIP (in flight) $d"; return
  fi
  mkdir -p $d
  cp $T/cfg_h20_l2_60mb.config $d/gpgpusim.config
  printf "\n-gpgpu_cluster_barrier_invalidate_l1 %s\n" $inv >> $d/gpgpusim.config
  throttle
  echo "LAUNCH $d  ($(date +%H:%M:%S))"
  ( cd $d && timeout $TO env RESC_QUIET=1 RESC_GEN_RING=$G RESC_MW=$MW \
      $BIN/$bin $NCL $NS $LAG $CL $pv > run.log 2>&1; echo $? > exit.code ) &
}

# Informative first: the real-kernel anchor (32 => f=0.1111), then the cheap ends,
# then outward to the expensive tail.  1024 last -- 3 of its 8 runs are already in flight.
for pv in 32 64 16 128 4 0 256 512 1024; do
  echo "### point priv_lines=$pv  ($(date +%H:%M:%S))"
  for arm in arm0 arm1; do
    for inv in 0 1; do launch $inv $pv $arm; done
  done
  for arm in arm2 arm3; do
    for inv in 2 3; do launch $inv $pv $arm; done
  done
done
wait
# the reparented priv=1024 runs may still be going; wait for the box to drain
while [ "$(running_sims)" -gt 0 ]; do sleep 20; done
echo "MF_SWEEP_ALL_DONE"
