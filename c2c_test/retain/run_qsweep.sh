#!/bin/bash
# Historical execution guard; no server is configured here.
printf '%s\n' 'Historical RESC recipe: execution disabled after infrastructure sanitation. Review a new laboratory-only launcher before running.' >&2
exit 64
# RETENTION QUOTA SWEEP. Prereg: retention-quota-sweep-prereg.md
#   SHA-256 1890bc0aec277923dfa9b85e9f301cf799eaf1f9ca740a26c9e5bc36acbfd876
# Varies ONLY RESC_RETAIN_Q in {25,50,90} + R0 control. CL=128 PV=32 32KB L1, W=3.
set +u
source /legacy-research/owner/miniconda3/etc/profile.d/conda.sh
conda activate clustersim-py311
T=/legacy-research/owner/cudatest/RESC-c2c-wt/c2c_test
cd $T
export PATH=$T/ptxas_shim:/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/legacy-research/owner/cudatest/RESC-c2c-wt/build/lib:$HOME/miniconda3/envs/clustersim-py311/lib:$LD_LIBRARY_PATH
W=$T/retain/qsweep; BIN=$T/retain/bin; mkdir -p $W
NCL=2; NS=32; LAG=2; CL=128; PV=32; RW=3; TO=14400; MAXJOBS=2
running(){ ps -eo comm --no-headers|grep -cE "^(mf_arm|mfr_arm|m_R|rt_arm|mu_arm)"; }
throttle(){ while [ "$(running)" -ge $MAXJOBS ]; do sleep 20; done; }
launch(){ # tag  retain_on  q
  local tag=$1 ron=$2 q=$3 d=$W/$1
  [ -f "$d/exit.code" ] && [ "$(cat $d/exit.code)" = "0" ] && return
  [ -f "$d/run.log" ] && [ ! -f "$d/exit.code" ] && { echo "IN-FLIGHT $d"; return; }
  mkdir -p "$d"; cp $T/cfg_h20_L1_32KB.config "$d/gpgpusim.config"
  printf "\n-gpgpu_cluster_barrier_invalidate_l1 3\n" >> "$d/gpgpusim.config"
  rm -f "$d/exit.code"; throttle
  echo "LAUNCH $tag retain=$ron Q=$q ($(date +%H:%M:%S))"
  if [ "$ron" = "0" ]; then
    ( cd "$d" && nice -n 10 timeout $TO env RESC_QUIET=1 RESC_GEN_RING=$RW RESC_MW=1 \
        $BIN/rt_arm2 $NCL $NS $LAG $CL $PV > run.log 2>&1; echo $? > exit.code ) &
  else
    ( cd "$d" && nice -n 10 timeout $TO env RESC_QUIET=1 RESC_GEN_RING=$RW RESC_MW=1 \
        RESC_RETAIN=1 RESC_RETAIN_Q=$q \
        $BIN/rt_arm2 $NCL $NS $LAG $CL $PV > run.log 2>&1; echo $? > exit.code ) &
  fi
}
echo "### QUOTA SWEEP ($(date))"
echo "### so md5: $(md5sum /legacy-research/owner/cudatest/RESC-c2c-wt/build/lib/libcudart.so.12|awk "{print \$1}")"
echo "### cfg md5: $(md5sum $T/cfg_h20_L1_32KB.config|awk "{print \$1}")"
launch R0_ctl  0 0
launch Q25     1 25
wait
launch Q50     1 50
launch Q90     1 90
wait
echo QSWEEP_DONE
