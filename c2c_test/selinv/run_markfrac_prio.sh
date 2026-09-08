#!/bin/bash
# Historical execution guard; no server is configured here.
printf '%s\n' 'Historical RESC recipe: execution disabled after infrastructure sanitation. Review a new laboratory-only launcher before running.' >&2
exit 64
# MARKED-FRACTION SWEEP -- PRIORITY PASS: finish the DECISION-RELEVANT points first.
#
# WHY THIS EXISTS (scheduling only -- the experiment is unchanged):
# The three earlier runners are serialized: pass2 -> diag -> pass3.  pass2 is currently
# grinding through its expensive tail (priv=512, 1024; ~3.8 wall-hours at cap 3), and
# neither diag nor pass3 launches anything until it finishes.  But the cells that decide
# KILL-W1/W2/W3 -- above all the f=0.1111 real-kernel anchor (priv=32) -- are CHEAP
# (~2 min each) and are stuck behind that tail.
#
# This pass runs ONLY the missing cells of the cheap, decision-relevant points, in
# ascending cost order.  It changes NOTHING about the experiment: same binaries, same
# knobs, same fixed communication, same fairness rule, same pre-registered points.  It
# only reorders WHEN cells are computed.  The later passes' no-clobber rule
# (exit.code == 0 => skip) means they will simply skip whatever this pass completes.
#
# Pre-registration: markfrac-prereg.md (3ebf6393...), addendum 01 (b02af321..., the
# priv=32 anchor), addendum 02 (7a2d243d..., the arm2@{0,1} diagnostic columns).
#
# CONCURRENCY: shares the SAME global cap of 3 with every other runner, counted across
# all mf_arm processes on the shared box.
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
  [ -f $d/exit.code ] && [ "$(cat $d/exit.code)" = "0" ] && return
  # in-flight, identified by DIRECTORY state (never by command line -- the knob is not
  # on the command line; that was the bug in run_markfrac2.sh)
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

# Anchor FIRST (f = 0.1111, the published real-kernel ratio), then the rest of the
# cheap points outward.  The expensive tail (256/512/1024) is deliberately left to the
# already-running passes.
for pv in 32 16 64 4 0 128; do
  echo "### priority point priv_lines=$pv ($(date +%H:%M:%S))"
  for arm in arm0 arm1; do for inv in 0 1; do launch $inv $pv $arm; done; done
  for arm in arm2 arm3; do for inv in 2 3; do launch $inv $pv $arm; done; done
  for inv in 0 1; do launch $inv $pv arm2; done      # addendum 02 diagnostic columns
done
wait
echo "MF_PRIORITY_DONE"
