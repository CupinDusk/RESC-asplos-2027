#!/bin/bash
# Historical execution guard; no server is configured here.
printf '%s\n' 'Historical RESC recipe: execution disabled after infrastructure sanitation. Review a new laboratory-only launcher before running.' >&2
exit 64
# MARKED-FRACTION SWEEP -- where does marked-line selectivity actually pay?
# Pre-registration: resc-asplos-2027/markfrac-prereg.md (frozen BEFORE this ran).
#
# Workload: markfrac.cu = pipe_chain + a per-block PRIVATE READ-ONLY tile.
#   Communication is held byte-for-byte FIXED across the whole sweep
#   (comm_lines=4, lag=2, nstages=32, num_clusters=2, cluster_size=8).
#   ONLY priv_lines changes, so   f = comm_lines/(comm_lines+priv_lines).
#   priv_lines=0  =>  f=1.0  =>  the kernel IS pipe_chain (self-consistency anchor).
#
# FAIRNESS RULE (prereg S1.2, inherited from selective-invalidate-prereg.md S3):
# knobs 2/3 model RESC HARDWARE and are NEVER given to a baseline.  Each baseline
# runs at BOTH its legal options (0 = sync_restrict escape, 1 = plain acquire) and
# is credited with the FASTER -- the pairing LEAST favourable to RESC.
#   arm0 (cg, correct incumbent) -> knobs 0,1     [no RESC hw]
#   arm1 (ca overlay / conservative DSM stand-in) -> knobs 0,1  [no RESC hw]
#   arm2 (lesioned RESC, G=2, MW=0)  -> knobs 2,3  [has RESC hw]
#   arm3 (full RESC,     G=3, MW=1)  -> knobs 2,3  [has RESC hw]
# arm2/arm3 are the SAME BINARY (mf_arm2); discipline chosen by env at run time.
#
# CONCURRENCY IS CAPPED AT 3.  The machine is shared and loaded (~70-84); previous
# over-subscription caused exit-124 timeouts.  Do NOT raise this.
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
TO=5400                        # generous; machine is shared
MAXJOBS=3

throttle(){ while [ "$(jobs -rp | wc -l)" -ge $MAXJOBS ]; do sleep 5; done; }

launch(){  # $1=knob $2=priv_lines $3=arm
  local inv=$1 pv=$2 arm=$3 bin G MW d
  case $arm in
    arm0) bin=mf_arm0; G=1; MW=0;;
    arm1) bin=mf_arm1; G=0; MW=0;;
    arm2) bin=mf_arm2; G=2; MW=0;;
    arm3) bin=mf_arm2; G=3; MW=1;;   # LAG=2 -> G=LAG+1=3
  esac
  d=$W/k${inv}_v${pv}_${arm}
  # no-clobber: never silently redo a completed run
  if [ -f $d/exit.code ] && [ "$(cat $d/exit.code)" = "0" ]; then
    echo "SKIP  (done) $d"; return
  fi
  mkdir -p $d
  cp $T/cfg_h20_l2_60mb.config $d/gpgpusim.config
  printf "\n-gpgpu_cluster_barrier_invalidate_l1 %s\n" $inv >> $d/gpgpusim.config
  throttle
  echo "LAUNCH $d"
  ( cd $d && timeout $TO env RESC_QUIET=1 RESC_GEN_RING=$G RESC_MW=$MW \
      $BIN/$bin $NCL $NS $LAG $CL $pv > run.log 2>&1; echo $? > exit.code ) &
}

# Largest private tiles first: they are the slowest, so starting them early keeps
# the 3 slots busy instead of trailing one long run at the end.
for pv in 1024 512 256 128 64 16 4 0; do
  for arm in arm0 arm1; do
    for inv in 0 1; do launch $inv $pv $arm; done
  done
  for arm in arm2 arm3; do
    for inv in 2 3; do launch $inv $pv $arm; done
  done
done
wait
echo "MF_SWEEP_ALL_DONE"
