#!/bin/bash
# Historical execution guard; no server is configured here.
printf '%s\n' 'Historical RESC recipe: execution disabled after infrastructure sanitation. Review a new laboratory-only launcher before running.' >&2
exit 64
# MARKED-FRACTION SWEEP -- FINISHER.  Replaces run_markfrac_diag.sh and
# run_markfrac3.sh, both of which were stopped for two real hazards:
#
#   HAZARD 1 (correctness): run_markfrac_diag.sh had NO in-flight guard -- it skipped
#     only cells with exit.code == 0.  A cell already RUNNING (started by another pass)
#     has no exit.code yet, so diag would have launched a SECOND simulator into the same
#     directory, both writing run.log and exit.code.  That corrupts the cell.
#   HAZARD 2 (liveness): both waited on `grep -q MF_SWEEP_ALL_DONE mf_master2.log`, and
#     pass2's own terminator is `while running_sims > 0`, which counts EVERY mf_arm
#     process on the box including the priority pass's.  So the marker can be starved
#     indefinitely and neither pass would ever start.
#
# This finisher: (a) identifies a cell by its DIRECTORY, never by command line (the knob
# is not on the command line -- that was the original run_markfrac2.sh bug); (b) treats
# "run.log present, exit.code absent" as in-flight and leaves it alone; (c) re-runs any
# cell whose exit.code is non-zero (e.g. the exit-124 timeout) under TO=9000; (d) waits
# for nothing except the global concurrency cap; (e) loops until every one of the 90
# cells is exit 0, so it cannot terminate early on a transient.
#
# The experiment is UNCHANGED: same binaries, knobs, fixed communication, fairness rule,
# and the frozen 9-point set.  Only scheduling and safety change.
#
# Pre-registration: markfrac-prereg.md (3ebf6393...), addendum 01 (b02af321...),
# addendum 02 (7a2d243d..., the arm2@{0,1} diagnostic columns included below).
set +u
source /legacy-research/owner/miniconda3/etc/profile.d/conda.sh
conda activate clustersim-py311
T=/legacy-research/owner/cudatest/RESC-c2c-wt/c2c_test
cd $T
export PATH=$T/ptxas_shim:/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/legacy-research/owner/cudatest/RESC-c2c-wt/build/lib:$HOME/miniconda3/envs/clustersim-py311/lib:$LD_LIBRARY_PATH
W=$T/selinv/mf/sweep
BIN=$T/selinv/mf/bin
NCL=2; NS=32; LAG=2; CL=4
TO=9000
MAXJOBS=3

running_sims(){ ps -eo comm --no-headers | grep -c '^mf_arm'; }
throttle(){ while [ "$(running_sims)" -ge $MAXJOBS ]; do sleep 10; done; }

launch(){  # $1=knob $2=priv_lines $3=arm
  local inv=$1 pv=$2 arm=$3 bin G MW d
  case $arm in
    arm0) bin=mf_arm0; G=1; MW=0;;
    arm1) bin=mf_arm1; G=0; MW=0;;
    arm2) bin=mf_arm2; G=2; MW=0;;
    arm3) bin=mf_arm2; G=3; MW=1;;
  esac
  d=$W/k${inv}_v${pv}_${arm}
  # already good -> never redo
  if [ -f "$d/exit.code" ] && [ "$(cat "$d/exit.code")" = "0" ]; then return; fi
  # in flight (someone else is running it) -> never duplicate
  if [ -f "$d/run.log" ] && [ ! -f "$d/exit.code" ]; then return; fi
  # otherwise: never run, or failed (e.g. 124) -> (re)run it
  mkdir -p "$d"
  cp $T/cfg_h20_l2_60mb.config "$d/gpgpusim.config"
  printf "\n-gpgpu_cluster_barrier_invalidate_l1 %s\n" $inv >> "$d/gpgpusim.config"
  rm -f "$d/exit.code"
  throttle
  echo "LAUNCH $d ($(date +%H:%M:%S))"
  ( cd "$d" && timeout $TO env RESC_QUIET=1 RESC_GEN_RING=$G RESC_MW=$MW \
      $BIN/$bin $NCL $NS $LAG $CL $pv > run.log 2>&1; echo $? > exit.code ) &
}

# cheap/decision-relevant points first, expensive tail last
PTS="32 16 64 4 0 128 256 512 1024"

pass=0
while :; do
  pass=$((pass+1))
  echo "### finisher pass $pass ($(date +%H:%M:%S))"
  for pv in $PTS; do
    for arm in arm0 arm1; do for inv in 0 1; do launch $inv $pv $arm; done; done
    for arm in arm2 arm3; do for inv in 2 3; do launch $inv $pv $arm; done; done
    for inv in 0 1; do launch $inv $pv arm2; done    # addendum-02 diagnostics
  done
  wait
  # count what is still outstanding
  todo=0
  for pv in $PTS; do
    for spec in arm0:0 arm0:1 arm1:0 arm1:1 arm2:2 arm2:3 arm3:2 arm3:3 arm2:0 arm2:1; do
      a=${spec%:*}; k=${spec#*:}
      d=$W/k${k}_v${pv}_${a}
      if [ ! -f "$d/exit.code" ] || [ "$(cat "$d/exit.code")" != "0" ]; then todo=$((todo+1)); fi
    done
  done
  echo "### after pass $pass: $todo cells outstanding ($(date +%H:%M:%S))"
  [ "$todo" = "0" ] && break
  # if others still hold cells in flight, wait for them rather than spinning
  if [ "$(running_sims)" -gt 0 ]; then sleep 60; else sleep 10; fi
  [ $pass -ge 40 ] && { echo "### giving up after $pass passes, $todo outstanding"; break; }
done
echo "MF_FINISHER_DONE outstanding=$todo"
