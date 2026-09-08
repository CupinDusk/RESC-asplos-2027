#!/bin/bash
# Historical execution guard; no server is configured here.
printf '%s\n' 'Historical RESC recipe: execution disabled after infrastructure sanitation. Review a new laboratory-only launcher before running.' >&2
exit 64
# MARKED-FRACTION SWEEP -- PASS 3, repairs the in-flight-guard bug in run_markfrac2.sh.
#
# THE BUG (found and recorded, not hidden): run_markfrac2.sh guarded against
# duplicating an already-running job with
#     pgrep -f "$BIN/$bin $NCL $NS $LAG $CL $pv$"
# but the KNOB is not on the command line -- it is written into gpgpusim.config in the
# run directory.  So every arm/knob variant at the same priv_lines has an IDENTICAL
# command line, and the guard skipped 17 runs that had never been launched, e.g. it
# skipped k1_v128_arm1 because k0_v128_arm1 happened to be running.
#
# No result was corrupted by this -- the skipped runs simply never ran, and the
# no-clobber rule (exit.code == 0 => skip) means nothing completed is redone.
#
# THE FIX: identify a run by its DIRECTORY (which encodes knob, priv and arm), and
# treat "directory exists with a run.log but no exit.code" as in-flight.  This pass
# re-launches exactly the runs that are missing, for every point, both the parent
# sweep's arms and addendum 02's arm2@{0,1} diagnostic columns.
#
# Pre-registration: markfrac-prereg.md (3ebf6393..., 2026-09-03T13:06:21Z)
#   + addendum 01 (b02af321..., priv=32 anchor) + addendum 02 (7a2d243d..., arm2@{0,1}).
# Nothing about the experiment changes -- same points, arms, knobs, fairness rule.
#
# CONCURRENCY CAP 3, counting every mf_arm process on the shared box.
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

NCL=2; NS=32; LAG=2; CL=4
TO=9000
MAXJOBS=3

running_sims(){ ps -eo comm --no-headers | grep -c '^mf_arm'; }
throttle(){ while [ "$(running_sims)" -ge $MAXJOBS ]; do sleep 10; done; }

# wait for the earlier passes to stop launching, so we do not race them
while pgrep -f "bash run_markfrac2.sh" > /dev/null || pgrep -f "bash run_markfrac_diag.sh" > /dev/null; do
  sleep 30
done
echo "earlier passes finished, starting repair pass ($(date +%H:%M:%S))"

launch(){  # $1=knob $2=priv_lines $3=arm
  local inv=$1 pv=$2 arm=$3 bin G MW d
  case $arm in
    arm0) bin=mf_arm0; G=1; MW=0;;
    arm1) bin=mf_arm1; G=0; MW=0;;
    arm2) bin=mf_arm2; G=2; MW=0;;
    arm3) bin=mf_arm2; G=3; MW=1;;
  esac
  d=$W/k${inv}_v${pv}_${arm}
  if [ -f $d/exit.code ] && [ "$(cat $d/exit.code)" = "0" ]; then return; fi
  # in-flight = has a log but no exit.code yet (identified by DIRECTORY, not cmdline)
  if [ -f $d/run.log ] && [ ! -f $d/exit.code ]; then
    echo "IN-FLIGHT (leave alone) $d"; return
  fi
  mkdir -p $d
  cp $T/cfg_h20_l2_60mb.config $d/gpgpusim.config
  printf "\n-gpgpu_cluster_barrier_invalidate_l1 %s\n" $inv >> $d/gpgpusim.config
  throttle
  echo "LAUNCH $d ($(date +%H:%M:%S))"
  ( cd $d && timeout $TO env RESC_QUIET=1 RESC_GEN_RING=$G RESC_MW=$MW \
      $BIN/$bin $NCL $NS $LAG $CL $pv > run.log 2>&1; echo $? > exit.code ) &
}

for pv in 32 64 16 128 4 0 256 512 1024; do
  echo "### point priv_lines=$pv ($(date +%H:%M:%S))"
  for arm in arm0 arm1; do for inv in 0 1; do launch $inv $pv $arm; done; done
  for arm in arm2 arm3; do for inv in 2 3; do launch $inv $pv $arm; done; done
  # addendum 02 diagnostic columns
  for inv in 0 1; do launch $inv $pv arm2; done
done
wait
while [ "$(running_sims)" -gt 0 ]; do sleep 20; done
echo "MF_REPAIR_DONE"
