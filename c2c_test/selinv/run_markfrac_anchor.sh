#!/bin/bash
# Historical execution guard; no server is configured here.
printf '%s\n' 'Historical RESC recipe: execution disabled after infrastructure sanitation. Review a new laboratory-only launcher before running.' >&2
exit 64
# MARKED-FRACTION SWEEP -- ADDENDUM POINT priv_lines=32 => f=0.1111.
# Pre-registration addendum: markfrac-prereg-addendum-01.md, SHA-256 b02af321...,
# frozen 2026-09-03T13:17:24Z while ZERO sweep results existed.
# f=0.1111 matches FlashAttention-4 2-CTA backward (128 KB private : 16 KB dS via DSMEM).
# Same fixed communication and same fairness rule as the parent sweep.
# Waits for the parent sweep to finish so total concurrency never exceeds 3.
set +u
source /legacy-research/owner/miniconda3/etc/profile.d/conda.sh
conda activate clustersim-py311
T=/legacy-research/owner/cudatest/RESC-c2c-wt/c2c_test
cd $T
export PATH=$T/ptxas_shim:/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/legacy-research/owner/cudatest/RESC-c2c-wt/build/lib:$HOME/miniconda3/envs/clustersim-py311/lib:$LD_LIBRARY_PATH
W=$T/selinv/mf/sweep
BIN=$T/selinv/mf/bin
NCL=2; NS=32; LAG=2; CL=4; TO=5400; MAXJOBS=3; PV=32

# wait for parent sweep
while ! grep -q MF_SWEEP_ALL_DONE $T/selinv/mf_master.log 2>/dev/null; do sleep 60; done
echo "parent sweep done, starting anchor point priv=$PV"

throttle(){ while [ "$(jobs -rp | wc -l)" -ge $MAXJOBS ]; do sleep 5; done; }
launch(){
  local inv=$1 arm=$2 bin G MW d
  case $arm in
    arm0) bin=mf_arm0; G=1; MW=0;;
    arm1) bin=mf_arm1; G=0; MW=0;;
    arm2) bin=mf_arm2; G=2; MW=0;;
    arm3) bin=mf_arm2; G=3; MW=1;;
  esac
  d=$W/k${inv}_v${PV}_${arm}
  if [ -f $d/exit.code ] && [ "$(cat $d/exit.code)" = "0" ]; then echo "SKIP (done) $d"; return; fi
  mkdir -p $d
  cp $T/cfg_h20_l2_60mb.config $d/gpgpusim.config
  printf "\n-gpgpu_cluster_barrier_invalidate_l1 %s\n" $inv >> $d/gpgpusim.config
  throttle
  echo "LAUNCH $d"
  ( cd $d && timeout $TO env RESC_QUIET=1 RESC_GEN_RING=$G RESC_MW=$MW \
      $BIN/$bin $NCL $NS $LAG $CL $PV > run.log 2>&1; echo $? > exit.code ) &
}
for arm in arm0 arm1; do for inv in 0 1; do launch $inv $arm; done; done
for arm in arm2 arm3; do for inv in 2 3; do launch $inv $arm; done; done
wait
echo "MF_ANCHOR_DONE"
