#!/bin/bash
# Historical execution guard; no server is configured here.
printf '%s\n' 'Historical RESC recipe: execution disabled after infrastructure sanitation. Review a new laboratory-only launcher before running.' >&2
exit 64
# DECISIVE ATTRIBUTION ARM (A8 = A2 with RESC_NO_FWD=1). Prereg c72a076e...
# Single variable vs A2: cross-SM L1->L1 forwarding disabled. Same binary/so/config/knob/argv.
set +u
source /legacy-research/owner/miniconda3/etc/profile.d/conda.sh
conda activate clustersim-py311
T=/legacy-research/owner/cudatest/RESC-c2c-wt/c2c_test
cd $T
export PATH=$T/ptxas_shim:/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/legacy-research/owner/cudatest/RESC-c2c-wt/build/lib:$HOME/miniconda3/envs/clustersim-py311/lib:$LD_LIBRARY_PATH
W=$T/composed/sweep; BIN=$T/composed/bin/v124
CFG=cfg_h20_smem228.config
TO=${TO:-21600}
MAXJOBS=${MAXJOBS:-2}
running_sims(){ ps -u hyren -o comm --no-headers | grep -c "^cx_arm" ; }
throttle(){ while [ "$(running_sims)" -ge $MAXJOBS ]; do sleep 20; done; }
launch(){ # $1=pointname $2=args
  local pn=$1 args="$2" d
  d=$W/${pn}_A8
  [ -f "$d/exit.code" ] && [ "$(cat "$d/exit.code")" = "0" ] && { echo "DONE $d"; return; }
  [ -f "$d/run.log" ] && [ ! -f "$d/exit.code" ] && { echo "IN-FLIGHT $d"; return; }
  throttle
  mkdir -p "$d"
  cp $T/$CFG "$d/gpgpusim.config"
  printf "\n-gpgpu_cluster_barrier_invalidate_l1 %d\n" 3 >> "$d/gpgpusim.config"
  rm -f "$d/exit.code"
  echo "LAUNCH $d NO_FWD=1 knob=3 args=\"$args\" ($(date +%H:%M:%S))"
  ( cd "$d" && timeout $TO nice -n 10 env RESC_QUIET=1 RESC_GEN_RING=3 RESC_MW=1 \
      RESC_RETAIN=1 RESC_RETAIN_Q=100 RESC_NO_FWD=1 \
      $BIN/cx_arm2 $args > run.log 2>&1; echo $? > exit.code ) &
}
echo "### NO_FWD ARM  cfg=$CFG  ($(date))"
echo "### so md5:  $(md5sum /legacy-research/owner/cudatest/RESC-c2c-wt/build/lib/libcudart.so.12|awk "{print \$1}")"
echo "### bin md5: $(md5sum $BIN/cx_arm2|awk "{print \$1}")"
pass=0
while :; do
  pass=$((pass+1)); echo "### pass $pass ($(date +%H:%M:%S))"
  launch L1P "2 32 4 64 256"
  launch OCC "2 32 4 64 1024"
  sleep 60
  ok=0
  for p in L1P OCC; do [ -f "$W/${p}_A8/exit.code" ] && [ "$(cat $W/${p}_A8/exit.code)" = "0" ] && ok=$((ok+1)); done
  [ $ok -eq 2 ] && { echo "### ALL DONE ($(date))"; break; }
  [ $pass -ge 200 ] && { echo "### GIVING UP after $pass passes"; break; }
done
